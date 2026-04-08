"""
pipeline/graph.py — LangGraph swarm definition.

Graph topology (per chunk):

  START
    │
    ▼
  [simulate_question]        ← Llama 3 generates question (10% adversarial)
    │
    ▼
  [retrieve_context]         ← pgvector + BM25 hybrid search
    │
    ▼
  [generate_answer]          ← Llama 3 generates grounded answer
    │
    ▼
  [judge_answer]             ← gpt-4o-mini (cascade → o1-mini if confidence < 90%)
    │
    ├── score >= threshold ──► [write_output]  ──► END (status=ready)
    │
    └── score < threshold ───► retry_count < MAX_RETRIES?
                                  │ yes → back to [simulate_question]
                                  │ no  → [mark_unresolvable] → END

All DB mutations happen inside the write_output and mark_unresolvable nodes
using explicit SQLAlchemy transactions (ACID idempotency, Self-Check patch).
"""

from __future__ import annotations

import logging
import uuid

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from agents.expert import generate_answer, retrieve_context
from agents.judge import judge_answer
from agents.simulator import simulate_question
from config.settings import settings
from db import repository as repo
from pipeline.state import FoundryState
from pipeline.watermark import build_watermark_description, compute_watermark_hash
from utils.output import JSONLWriter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt stored in every ChatML record
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE. "
    "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
    "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie\"."
)


# ---------------------------------------------------------------------------
# Node wrappers (inject DB session via closure)
# ---------------------------------------------------------------------------

def make_retrieve_node(session: Session):
    def _retrieve(state: FoundryState) -> dict:
        return retrieve_context(state, session)
    return _retrieve


def make_write_node(session: Session, writer: JSONLWriter):
    def _write(state: FoundryState) -> dict:
        chunk_id = uuid.UUID(state["chunk"]["id"])

        # Write to JSONL (atomic append)
        record_index, watermark_hash = writer.write_sample(
            question=state["question"],
            answer=state["answer"],
            system_prompt=_SYSTEM_PROMPT,
        )

        # Persist sample row + mark chunk ready (single transaction → ACID)
        try:
            sample = repo.insert_sample(
                session,
                chunk_id=chunk_id,
                question=state["question"],
                answer=state["answer"],
                system_prompt=_SYSTEM_PROMPT,
                is_adversarial=state.get("is_adversarial", False),
                quality_score=state.get("quality_score"),
                judge_model=state.get("judge_model"),
                judge_reasoning=state.get("judge_reasoning"),
                batch_id=state.get("batch_id", ""),
            )
            repo.mark_sample_written(
                session,
                sample_id=sample.id,
                record_index=record_index,
                watermark_hash=watermark_hash,
            )
            repo.finalize_chunk(session, chunk_id=chunk_id, success=True)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("Failed to persist sample for chunk %s: %s", chunk_id, exc)
            raise

        if watermark_hash:
            technique = record_index // settings.watermark_interval
            repo.register_watermark(
                session,
                batch_id=state.get("batch_id", "default"),
                client_id=settings.client_id,
                watermark_signature=build_watermark_description(technique),
                watermark_hash=watermark_hash,
                record_indices=[record_index],
                total_records=writer.record_count,
            )
            session.commit()

        logger.info(
            "✓ Record %d written (chunk=%s score=%.2f%s)",
            record_index,
            state["chunk"]["id"][:8],
            state.get("quality_score", 0.0),
            " [watermarked]" if watermark_hash else "",
        )
        return {"status": "ready", "record_index": record_index}

    return _write


def make_unresolvable_node(session: Session):
    def _unresolvable(state: FoundryState) -> dict:
        chunk_id = uuid.UUID(state["chunk"]["id"])
        try:
            repo.finalize_chunk(
                session,
                chunk_id=chunk_id,
                success=False,
                error=f"Max retries reached. Last judge score: {state.get('quality_score', 0):.2f}",
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("Failed to mark chunk unresolvable: %s", exc)

        logger.warning(
            "✗ Chunk %s marked unresolvable after %d retries",
            state["chunk"]["id"][:8],
            state.get("retry_count", 0),
        )
        return {"status": "unresolvable"}

    return _unresolvable


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_judge(state: FoundryState) -> str:
    score = state.get("quality_score", 0.0)
    retries = state.get("retry_count", 0)

    if score >= settings.quality_threshold:
        return "write_output"

    if retries < settings.max_retries_per_chunk:
        logger.info(
            "Score %.2f < %.2f — retry %d/%d for chunk %s",
            score,
            settings.quality_threshold,
            retries + 1,
            settings.max_retries_per_chunk,
            state["chunk"]["id"][:8],
        )
        return "retry"

    return "mark_unresolvable"


def increment_retry(state: FoundryState) -> dict:
    """Pseudo-node: bump retry counter before looping back."""
    return {"retry_count": state.get("retry_count", 0) + 1}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(session: Session, writer: JSONLWriter) -> StateGraph:
    graph = StateGraph(FoundryState)

    # Add nodes
    graph.add_node("simulate_question", simulate_question)
    graph.add_node("retrieve_context", make_retrieve_node(session))
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("judge_answer", judge_answer)
    graph.add_node("write_output", make_write_node(session, writer))
    graph.add_node("mark_unresolvable", make_unresolvable_node(session))
    graph.add_node("increment_retry", increment_retry)

    # Linear pipeline
    graph.add_edge(START, "simulate_question")
    graph.add_edge("simulate_question", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("generate_answer", "judge_answer")

    # Conditional routing after Judge
    graph.add_conditional_edges(
        "judge_answer",
        route_after_judge,
        {
            "write_output": "write_output",
            "retry": "increment_retry",
            "mark_unresolvable": "mark_unresolvable",
        },
    )

    # Retry loop: increment counter → back to Simulator
    graph.add_edge("increment_retry", "simulate_question")

    # Terminal edges
    graph.add_edge("write_output", END)
    graph.add_edge("mark_unresolvable", END)

    return graph.compile()
