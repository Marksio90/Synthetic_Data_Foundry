"""
pipeline/graph.py — LangGraph swarm definition (Gods Finger v2).

Graph topology (per chunk, multi-turn):

  START
    │
    ▼
  [simulate_question]      ← 8 perspektyw, 10% adversarial
    │
    ▼
  [retrieve_context]       ← pgvector + BM25 hybrid search
    │
    ▼
  [generate_answer]        ← Ollama qwen2.5:14b / OpenAI (fallback)
    │
    ▼
  [constitutional_revision] ← self-critique + revision; original → rejected (DPO/ORPO/KTO)
    │
    ▼
  [judge_answer]           ← gpt-4o-mini wielowymiarowy (cascade → gpt-4o)
    │
    ├── score >= threshold ──► [append_turn]
    │                              │
    │                              ├── turn < MAX_TURNS ──► [simulate_followup]
    │                              │                              │
    │                              │                   (→ retrieve_context)
    │                              │
    │                              └── turn >= MAX_TURNS ──► [write_output] ──► END
    │
    └── score < threshold ───► (turn==0) retry_count < MAX_RETRIES?
                                  │ yes → [increment_retry] → [retrieve_context]
                                  │ no  → [mark_unresolvable] → END
                               (turn>0) → [write_output] → END
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from agents.constitutional import constitutional_revision
from agents.expert import generate_answer, retrieve_context
from agents.judge import judge_answer
from agents.simulator import simulate_followup, simulate_question
from config.settings import settings
from db import repository as repo
from pipeline.state import FoundryState
from pipeline.watermark import build_watermark_description, compute_watermark_hash
from utils.classifier import classify_question
from utils.dedup import MinHashDeduplicator
from utils.output import DPOWriter, JSONLWriter, KTOWriter, ORPOWriter

logger = logging.getLogger(__name__)

_SYSTEM_PROMPTS: dict[str, str] = {
    "cfo": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy CFO. "
        "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
        "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\""
    ),
    "prawnik": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy radcy prawnego. "
        "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
        "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\""
    ),
    "audytor": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy biegłego rewidenta ESG. "
        "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
        "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\""
    ),
    "analityk": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy analityka finansowego. "
        "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
        "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\""
    ),
    "regulator": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy regulatora/nadzorcy. "
        "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
        "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\""
    ),
    "akademik": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy badacza akademickiego. "
        "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
        "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\""
    ),
    "dziennikarz": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, wyjaśniającym przepisy przystępnym językiem. "
        "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
        "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\""
    ),
    "inwestor": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy inwestora instytucjonalnego. "
        "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
        "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\""
    ),
}
_DEFAULT_SYSTEM_PROMPT = (
    "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE. "
    "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
    "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\""
)


# ---------------------------------------------------------------------------
# Node wrappers
# ---------------------------------------------------------------------------

def make_retrieve_node(session: Session):
    def _retrieve(state: FoundryState) -> dict:
        return retrieve_context(state, session)
    return _retrieve


def append_turn(state: FoundryState) -> dict:
    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": state["question"]})
    history.append({"role": "assistant", "content": state["answer"]})
    return {
        "conversation_history": history,
        "turn_count": state.get("turn_count", 0) + 1,
        "question": "",
        "answer": "",
        "quality_score": 0.0,
        "retry_count": 0,
        "constitutional_critique": None,
    }


def route_after_append(state: FoundryState) -> str:
    if state.get("turn_count", 0) < settings.max_turns:
        return "simulate_followup"
    return "write_output"


def make_write_node(
    session: Session,
    writer: JSONLWriter,
    dpo_writer: Optional[DPOWriter] = None,
    orpo_writer: Optional[ORPOWriter] = None,
    kto_writer: Optional[KTOWriter] = None,
    deduplicator: Optional[MinHashDeduplicator] = None,
):
    def _write(state: FoundryState) -> dict:
        chunk_id = uuid.UUID(state["chunk"]["id"])
        perspective = state.get("perspective", "cfo")
        system_prompt = _SYSTEM_PROMPTS.get(perspective, _DEFAULT_SYSTEM_PROMPT)

        history = state.get("conversation_history", [])
        if history:
            messages = [{"role": "system", "content": system_prompt}] + list(history)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": state["answer"]},
            ]

        # Refusal cap
        if writer.should_skip(messages):
            try:
                repo.finalize_chunk(session, chunk_id=chunk_id, success=True)
                session.commit()
            except Exception as exc:
                session.rollback()
                logger.error("Failed to finalize capped chunk %s: %s", chunk_id, exc)
            logger.info("⊘ Refusal capped — chunk=%s", state["chunk"]["id"][:8])
            return {"status": "ready", "record_index": -1}

        # Near-duplicate detection
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        if deduplicator and first_user and deduplicator.is_duplicate(first_user):
            try:
                repo.finalize_chunk(session, chunk_id=chunk_id, success=True)
                session.commit()
            except Exception as exc:
                session.rollback()
                logger.error("Failed to finalize dedup chunk %s: %s", chunk_id, exc)
            logger.info("⊘ Near-duplicate — chunk=%s", state["chunk"]["id"][:8])
            return {"status": "ready", "record_index": -1}

        q_type, difficulty = classify_question(first_user) if first_user else ("factual", "medium")

        # Phase 1: DB commit (ACID)
        first_asst = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        rejected = state.get("rejected_answer") or ""
        try:
            sample = repo.insert_sample(
                session,
                chunk_id=chunk_id,
                question=first_user,
                answer=first_asst,
                system_prompt=system_prompt,
                is_adversarial=state.get("is_adversarial", False),
                quality_score=state.get("quality_score"),
                judge_model=state.get("judge_model"),
                judge_reasoning=state.get("judge_reasoning"),
                batch_id=state.get("batch_id", ""),
                perspective=perspective,
                conversation_json=messages,
                question_type=q_type,
                difficulty=difficulty,
                rejected_answer=rejected or None,
            )
            repo.finalize_chunk(session, chunk_id=chunk_id, success=True)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("Failed to persist sample for chunk %s: %s", chunk_id, exc)
            raise

        # Phase 2: Write SFT JSONL
        metadata = {
            "perspective": perspective,
            "question_type": q_type,
            "difficulty": difficulty,
            "source_document": state["chunk"].get("source_document", ""),
            "batch_id": state.get("batch_id", ""),
            "judge_details": state.get("judge_details", {}),
        }
        record_index, watermark_hash = writer.write_conversation(messages, metadata=metadata)

        # Phase 3: Write preference pairs (DPO / ORPO / KTO)
        if rejected and record_index >= 0:
            prompt_msgs = [m for m in messages if m["role"] in ("system", "user")][:2]
            if dpo_writer:
                dpo_writer.write_pair(prompt_msgs, first_asst, rejected)
            if orpo_writer:
                orpo_writer.write_pair(prompt_msgs, first_asst, rejected)
            if kto_writer:
                kto_writer.write_pair(prompt_msgs, first_asst, rejected)

        # Phase 4: Update DB + watermark registry
        try:
            repo.mark_sample_written(session, sample.id, record_index, watermark_hash)
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

        logger.info(
            "✓ Record %d (chunk=%s turns=%d score=%.2f type=%s diff=%s persp=%s%s)",
            record_index, state["chunk"]["id"][:8], state.get("turn_count", 0),
            state.get("quality_score", 0.0), q_type, difficulty, perspective,
            " [watermarked]" if watermark_hash else "",
        )
        return {"status": "ready", "record_index": record_index, "question_type": q_type, "difficulty": difficulty}

    return _write


def make_unresolvable_node(session: Session):
    def _unresolvable(state: FoundryState) -> dict:
        chunk_id = uuid.UUID(state["chunk"]["id"])
        try:
            repo.finalize_chunk(
                session, chunk_id=chunk_id, success=False,
                error=f"Max retries. Last score: {state.get('quality_score', 0):.2f}",
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("Failed to mark chunk unresolvable: %s", exc)
        logger.warning("✗ Chunk %s unresolvable after %d retries", state["chunk"]["id"][:8], state.get("retry_count", 0))
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

    if turn > 0:
        logger.info(
            "Follow-up score %.2f < %.2f at turn %d — writing partial (chunk=%s)",
            score, settings.quality_threshold, turn, state["chunk"]["id"][:8],
        )
        return "write_output"

    if retries < settings.max_retries_per_chunk:
        logger.info(
            "Score %.2f < %.2f — retry %d/%d (chunk=%s)",
            score, settings.quality_threshold, retries + 1, settings.max_retries_per_chunk,
            state["chunk"]["id"][:8],
        )
        return "retry"

    return "mark_unresolvable"


def increment_retry(state: FoundryState) -> dict:
    new_retry = state.get("retry_count", 0) + 1
    logger.info(
        "Retry %d for chunk=%s perspective=%s (last score=%.2f)",
        new_retry, state["chunk"]["id"][:8], state.get("perspective", "?"), state.get("quality_score", 0.0),
    )
    return {"retry_count": new_retry, "rejected_answer": state.get("answer", "")}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    session: Session,
    writer: JSONLWriter,
    dpo_writer: Optional[DPOWriter] = None,
    orpo_writer: Optional[ORPOWriter] = None,
    kto_writer: Optional[KTOWriter] = None,
    deduplicator: Optional[MinHashDeduplicator] = None,
) -> StateGraph:
    graph = StateGraph(FoundryState)

    graph.add_node("simulate_question",       simulate_question)
    graph.add_node("simulate_followup",       simulate_followup)
    graph.add_node("retrieve_context",        make_retrieve_node(session))
    graph.add_node("generate_answer",         generate_answer)
    graph.add_node("constitutional_revision", constitutional_revision)
    graph.add_node("judge_answer",            judge_answer)
    graph.add_node("append_turn",             append_turn)
    graph.add_node("write_output",            make_write_node(session, writer, dpo_writer, orpo_writer, kto_writer, deduplicator))
    graph.add_node("mark_unresolvable",       make_unresolvable_node(session))
    graph.add_node("increment_retry",         increment_retry)

    graph.add_edge(START,                       "simulate_question")
    graph.add_edge("simulate_question",         "retrieve_context")
    graph.add_edge("retrieve_context",          "generate_answer")
    graph.add_edge("generate_answer",           "constitutional_revision")
    graph.add_edge("constitutional_revision",   "judge_answer")

    graph.add_conditional_edges(
        "judge_answer", route_after_judge,
        {"append_turn": "append_turn", "retry": "increment_retry",
         "write_output": "write_output", "mark_unresolvable": "mark_unresolvable"},
    )

    graph.add_edge("increment_retry",   "retrieve_context")

    graph.add_conditional_edges(
        "append_turn", route_after_append,
        {"simulate_followup": "simulate_followup", "write_output": "write_output"},
    )

    graph.add_edge("simulate_followup",   "retrieve_context")
    graph.add_edge("write_output",        END)
    graph.add_edge("mark_unresolvable",   END)

    return graph.compile()
