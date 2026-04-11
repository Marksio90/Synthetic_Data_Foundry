"""
pipeline/graph.py — LangGraph swarm definition.

Graph topology (per chunk, multi-turn):

  START
    │
    ▼
  [simulate_question]      ← Groq/Llama generates initial question (10% adversarial)
    │
    ▼
  [retrieve_context]       ← pgvector + BM25 hybrid search
    │
    ▼
  [generate_answer]        ← Groq/Llama grounded answer (uses conversation history)
    │
    ▼
  [judge_answer]           ← gpt-4o-mini (cascade → gpt-4o if confidence < threshold)
    │
    ├── score >= threshold ──► [append_turn]
    │                              │
    │                              ├── turn < MAX_TURNS ──► [simulate_followup]
    │                              │                              │
    │                              │                          (loops back to retrieve_context)
    │                              │
    │                              └── turn >= MAX_TURNS ──► [write_output] ──► END
    │
    └── score < threshold ───► (turn==0) retry_count < MAX_RETRIES?
                                  │ yes → [increment_retry] → [simulate_question]
                                  │ no  → [mark_unresolvable] → END
                               (turn>0) → [write_output] (write partial conversation) → END

All DB mutations happen inside write_output and mark_unresolvable nodes.
"""

from __future__ import annotations

import logging
import uuid

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from agents.expert import generate_answer, retrieve_context
from agents.judge import judge_answer
from agents.simulator import simulate_followup, simulate_question
from config.settings import settings
from db import repository as repo
from pipeline.state import FoundryState
from pipeline.watermark import build_watermark_description, compute_watermark_hash
from utils.output import JSONLWriter

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE. "
    "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
    "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie\"."
)


# ---------------------------------------------------------------------------
# Node wrappers
# ---------------------------------------------------------------------------

def make_retrieve_node(session: Session):
    def _retrieve(state: FoundryState) -> dict:
        return retrieve_context(state, session)
    return _retrieve


def append_turn(state: FoundryState) -> dict:
    """Accumulate the current Q&A turn into conversation_history."""
    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": state["question"]})
    history.append({"role": "assistant", "content": state["answer"]})
    return {
        "conversation_history": history,
        "turn_count": state.get("turn_count", 0) + 1,
        # Reset per-turn fields for the next turn
        "question": "",
        "answer": "",
        "quality_score": 0.0,
        "retry_count": 0,
    }


def route_after_append(state: FoundryState) -> str:
    """After appending a turn: continue if budget allows, else write."""
    if state.get("turn_count", 0) < settings.max_turns:
        return "simulate_followup"
    return "write_output"


def make_write_node(session: Session, writer: JSONLWriter):
    def _write(state: FoundryState) -> dict:
        chunk_id = uuid.UUID(state["chunk"]["id"])

        # Build full ChatML messages from accumulated conversation_history
        history = state.get("conversation_history", [])
        if history:
            messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + list(history)
        else:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": state["answer"]},
            ]

        # Pre-check refusal cap BEFORE any I/O — avoids DB commit for skipped records
        if writer.should_skip(messages):
            try:
                repo.finalize_chunk(session, chunk_id=chunk_id, success=True)
                session.commit()
            except Exception as exc:
                session.rollback()
                logger.error("Failed to finalize capped chunk %s: %s", chunk_id, exc)
            logger.info("⊘ Refusal capped — chunk=%s not written", state["chunk"]["id"][:8])
            return {"status": "ready", "record_index": -1}

        # ── Phase 1: Persist to DB first (ACID: DB commit before JSONL write) ───────
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        first_asst = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        try:
            sample = repo.insert_sample(
                session,
                chunk_id=chunk_id,
                question=first_user,
                answer=first_asst,
                system_prompt=_SYSTEM_PROMPT,
                is_adversarial=state.get("is_adversarial", False),
                quality_score=state.get("quality_score"),
                judge_model=state.get("judge_model"),
                judge_reasoning=state.get("judge_reasoning"),
                batch_id=state.get("batch_id", ""),
            )
            repo.finalize_chunk(session, chunk_id=chunk_id, success=True)
            session.commit()  # ← DB committed BEFORE file write (prevents desync)
        except Exception as exc:
            session.rollback()
            logger.error("Failed to persist sample for chunk %s: %s", chunk_id, exc)
            raise

        # ── Phase 2: Write to JSONL (local I/O — rarely fails) ──────────────────────
        record_index, watermark_hash = writer.write_conversation(messages)

        # ── Phase 3: Update DB with record_index + watermark (best-effort) ──────────
        try:
            repo.mark_sample_written(
                session,
                sample_id=sample.id,
                record_index=record_index,
                watermark_hash=watermark_hash,
            )
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
        except Exception as exc:
            session.rollback()
            logger.warning("Could not update record_index for sample %s: %s", sample.id, exc)

        turns = state.get("turn_count", 0)
        logger.info(
            "✓ Record %d written (chunk=%s turns=%d score=%.2f%s)",
            record_index,
            state["chunk"]["id"][:8],
            turns,
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
    turn = state.get("turn_count", 0)

    if score >= settings.quality_threshold:
        return "append_turn"

    # Follow-up turn failed quality check → write partial conversation (still valuable)
    if turn > 0:
        logger.info(
            "Follow-up score %.2f < %.2f at turn %d — writing partial conversation for chunk %s",
            score, settings.quality_threshold, turn, state["chunk"]["id"][:8],
        )
        return "write_output"

    # First turn: retry if budget allows
    if retries < settings.max_retries_per_chunk:
        logger.info(
            "Score %.2f < %.2f — retry %d/%d for chunk %s",
            score, settings.quality_threshold,
            retries + 1, settings.max_retries_per_chunk,
            state["chunk"]["id"][:8],
        )
        return "retry"

    return "mark_unresolvable"


def increment_retry(state: FoundryState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(session: Session, writer: JSONLWriter) -> StateGraph:
    graph = StateGraph(FoundryState)

    # Nodes
    graph.add_node("simulate_question",  simulate_question)
    graph.add_node("simulate_followup",  simulate_followup)
    graph.add_node("retrieve_context",   make_retrieve_node(session))
    graph.add_node("generate_answer",    generate_answer)
    graph.add_node("judge_answer",       judge_answer)
    graph.add_node("append_turn",        append_turn)
    graph.add_node("write_output",       make_write_node(session, writer))
    graph.add_node("mark_unresolvable",  make_unresolvable_node(session))
    graph.add_node("increment_retry",    increment_retry)

    # Main pipeline (turn 0)
    graph.add_edge(START,                "simulate_question")
    graph.add_edge("simulate_question",  "retrieve_context")
    graph.add_edge("retrieve_context",   "generate_answer")
    graph.add_edge("generate_answer",    "judge_answer")

    # After judge: route to append_turn, retry, write_output, or unresolvable
    graph.add_conditional_edges(
        "judge_answer",
        route_after_judge,
        {
            "append_turn":       "append_turn",
            "retry":             "increment_retry",
            "write_output":      "write_output",
            "mark_unresolvable": "mark_unresolvable",
        },
    )

    # Retry loop
    graph.add_edge("increment_retry",   "simulate_question")

    # After appending: continue to follow-up or write
    graph.add_conditional_edges(
        "append_turn",
        route_after_append,
        {
            "simulate_followup": "simulate_followup",
            "write_output":      "write_output",
        },
    )

    # Follow-up loop: simulate → retrieve → generate → judge (reuses shared nodes)
    graph.add_edge("simulate_followup", "retrieve_context")

    # Terminal edges
    graph.add_edge("write_output",       END)
    graph.add_edge("mark_unresolvable",  END)

    return graph.compile()
