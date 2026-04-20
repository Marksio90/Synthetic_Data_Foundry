"""
pipeline/graph.py — LangGraph swarm definition (Gods Finger v2) - ENTERPRISE EDITION.

Graf orkiestracji wielu agentów AI generujących dane syntetyczne.
Topologia (per chunk, multi-turn):

  START
    │
    ▼
  [simulate_question]       ← 8 perspektyw, 10% adversarial, z telemetrią
    │
    ▼
  [retrieve_context]        ← pgvector + BM25 hybrid search
    │
    ▼
  [generate_answer]         ← LLM Primary / Fallback
    │
    ▼
  [constitutional_revision] ← self-critique + revision; original → rejected (DPO/ORPO/KTO)
    │
    ▼
  [judge_answer]            ← Ocena wielowymiarowa
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
import time
import uuid
from functools import wraps
from typing import Optional, Any, Dict, Callable

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

# Importy Agentów (założenie spójności z resztą platformy)
from agents.constitutional import constitutional_revision
from agents.expert import generate_answer, retrieve_context
from agents.judge import judge_answer
from agents.simulator import simulate_followup, simulate_question
from config.settings import settings
from db import repository as repo
from pipeline.state import FoundryState
from pipeline.watermark import build_watermark_description
from utils.classifier import classify_question
from utils.dedup import MinHashDeduplicator
from utils.output import DPOWriter, JSONLWriter, KTOWriter, ORPOWriter

logger = logging.getLogger("foundry.pipeline.graph")


# ---------------------------------------------------------------------------
# Telemetria i Bezpieczeństwo Węzłów (PRO Feature)
# ---------------------------------------------------------------------------
def with_telemetry(node_name: str) -> Callable:
    """
    Dekorator telemetrii dla węzłów LangGraph.
    Automatycznie mierzy czas wykonania węzła, loguje ID przetwarzanego chunka
    oraz zabezpiecza graf przed wyciekiem pełnych Exception Tracebacks do konsumenta.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: FoundryState, *args, **kwargs) -> Dict[str, Any]:
            chunk_id_short = state.get("chunk", {}).get("id", "UNKNOWN")[:8]
            turn = state.get("turn_count", 0)
            
            start_time = time.perf_counter()
            logger.debug(f"[Graph:{node_name}] Uruchamianie węzła dla chunk={chunk_id_short} (Turn {turn})...")
            
            try:
                result = func(state, *args, **kwargs)
                
                elapsed = time.perf_counter() - start_time
                logger.debug(f"[Graph:{node_name}] Zakończono sukcesem w {elapsed:.3f}s (chunk={chunk_id_short})")
                
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(
                    f"[Graph:{node_name}] ❌ BŁĄD KRYTYCZNY po {elapsed:.3f}s (chunk={chunk_id_short}): {str(e)}", 
                    exc_info=True
                )
                # Ponowne rzucenie błędu, by Graph wiedział o awarii, 
                # ale mamy pewność, że wszystko zostało zalogowane.
                raise e
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Konfiguracja System Promptów (Słownik Typowany)
# ---------------------------------------------------------------------------
_DEFAULT_SYSTEM_PROMPT = (
    "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE. "
    "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
    "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\""
)

_SYSTEM_PROMPTS: Dict[str, str] = {
    "cfo": "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy CFO. Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\"",
    "prawnik": "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy radcy prawnego. Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\"",
    "audytor": "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy biegłego rewidenta ESG. Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\"",
    "analityk": "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy analityka finansowego. Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\"",
    "regulator": "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy regulatora/nadzorcy. Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\"",
    "akademik": "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy badacza akademickiego. Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\"",
    "dziennikarz": "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, wyjaśniającym przepisy przystępnym językiem. Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\"",
    "inwestor": "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy inwestora instytucjonalnego. Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\"",
}


# ---------------------------------------------------------------------------
# Definicje Węzłów (Nodes) 
# ---------------------------------------------------------------------------

def make_retrieve_node(session: Session) -> Callable:
    @with_telemetry("retrieve_context")
    def _retrieve(state: FoundryState) -> dict:
        return retrieve_context(state, session)
    return _retrieve


@with_telemetry("append_turn")
def append_turn(state: FoundryState) -> dict:
    """Dołącza najnowszą iterację Q&A do historii konwersacji."""
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


def make_write_node(
    session: Session,
    writer: JSONLWriter,
    dpo_writer: Optional[DPOWriter] = None,
    orpo_writer: Optional[ORPOWriter] = None,
    kto_writer: Optional[KTOWriter] = None,
    deduplicator: Optional[MinHashDeduplicator] = None,
) -> Callable:
    @with_telemetry("write_output")
    def _write(state: FoundryState) -> dict:
        """
        Zapisuje ostateczny stan do plików (JSONL, DPO) oraz bazy PostgreSQL.
        Wykorzystuje wzorzec bezpiecznych transakcji ACID (DB Flush -> File Write -> DB Commit).
        """
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

        # 1. Sprawdzenie Odmowy (Refusal Cap)
        if writer.should_skip(messages):
            try:
                repo.finalize_chunk(session, chunk_id=chunk_id, success=True)
                session.commit()
            except Exception as exc:
                session.rollback()
                logger.error("Błąd podczas finalizowania chunk-a odmownego %s: %s", chunk_id, exc)
            logger.info("⊘ Odrzucono z powodu limitu odmów — chunk=%s", state["chunk"]["id"][:8])
            return {"status": "ready", "record_index": -1}

        # 2. Sprawdzenie Duplikatów (Semantyczne/LSH)
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        if deduplicator and first_user and deduplicator.is_duplicate(first_user):
            try:
                repo.finalize_chunk(session, chunk_id=chunk_id, success=True)
                session.commit()
            except Exception as exc:
                session.rollback()
                logger.error("Błąd podczas finalizowania chunk-a zduplikowanego %s: %s", chunk_id, exc)
            logger.info("⊘ Zidentyfikowano niemal identyczny duplikat — chunk=%s", state["chunk"]["id"][:8])
            return {"status": "ready", "record_index": -1}

        q_type, difficulty = classify_question(first_user) if first_user else ("factual", "medium")
        first_asst = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        rejected = state.get("rejected_answer") or ""
        
        # -----------------------------------------------------------------------
        # FAZA TRANSAKCJI ACID (KRYTYCZNE MIEJSCE DLA INTEGRALNOŚCI DANYCH)
        # -----------------------------------------------------------------------
        try:
            # Etap 1: Wstępny Insert do Bazy Danych (Rezerwacja ID)
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
            
            # Wymuszamy wygenerowanie ID w bazie bez zamykania transakcji (Flush)
            session.flush()

            # Etap 2: Operacje dyskowe I/O (Pliki Treningowe)
            metadata = {
                "perspective": perspective,
                "question_type": q_type,
                "difficulty": difficulty,
                "source_document": state["chunk"].get("source_document", ""),
                "batch_id": state.get("batch_id", ""),
                "judge_details": state.get("judge_details", {}),
            }
            record_index, watermark_hash = writer.write_conversation(messages, metadata=metadata)

            if rejected and record_index >= 0:
                prompt_msgs = [m for m in messages if m["role"] in ("system", "user")][:2]
                if dpo_writer: dpo_writer.write_pair(prompt_msgs, first_asst, rejected)
                if orpo_writer: orpo_writer.write_pair(prompt_msgs, first_asst, rejected)
                if kto_writer: kto_writer.write_pair(prompt_msgs, first_asst, rejected)

            # Etap 3: Uzupełnienie Bazy Danych o Metadane Pliku i Znak Wodny
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
                
            repo.finalize_chunk(session, chunk_id=chunk_id, success=True)
            
            # Etap 4: Finalny Commit. 
            # Jeśli zapis plików by padł, wykonano by automatyczny Rollback przez `except`.
            session.commit()

            logger.info(
                "✓ ZAPISANO Z SUKCESEM: Rekord %d (chunk=%s, turny=%d, ocena=%.2f, typ=%s, trudność=%s, persp=%s%s)",
                record_index, state["chunk"]["id"][:8], state.get("turn_count", 0),
                state.get("quality_score", 0.0), q_type, difficulty, perspective,
                " [watermarked]" if watermark_hash else "",
            )
            return {"status": "ready", "record_index": record_index, "question_type": q_type, "difficulty": difficulty}

        except Exception as exc:
            # Wycofanie bazy danych w przypadku awarii pliku lub połączenia
            session.rollback()
            logger.error("KRYTYCZNY BŁĄD ZAPISU (ACID Rollback) dla chunk %s. Błąd: %s", chunk_id, exc, exc_info=True)
            raise

    return _write


def make_unresolvable_node(session: Session) -> Callable:
    @with_telemetry("mark_unresolvable")
    def _unresolvable(state: FoundryState) -> dict:
        chunk_id = uuid.UUID(state["chunk"]["id"])
        try:
            repo.finalize_chunk(
                session, chunk_id=chunk_id, success=False,
                error=f"Osiągnięto limit powtórzeń. Ostatnia ocena Sędziego: {state.get('quality_score', 0):.2f}",
            )
            session.commit()
            logger.warning("✗ Chunk %s uznany za NIEROZWIĄZYWALNY po %d próbach.", state["chunk"]["id"][:8], state.get("retry_count", 0))
        except Exception as exc:
            session.rollback()
            logger.error("Błąd podczas oznaczania chunk %s jako unresolvable: %s", chunk_id, exc)
        return {"status": "unresolvable"}
    return _unresolvable


@with_telemetry("increment_retry")
def increment_retry(state: FoundryState) -> dict:
    new_retry = state.get("retry_count", 0) + 1
    logger.info(
        "↺ PONOWIENIE (%d/%d) dla chunk=%s, perspektywa=%s (ostatnia ocena=%.2f)",
        new_retry, settings.max_retries_per_chunk, state["chunk"]["id"][:8], 
        state.get("perspective", "?"), state.get("quality_score", 0.0)
    )
    return {"retry_count": new_retry, "rejected_answer": state.get("answer", "")}


# ---------------------------------------------------------------------------
# Routing edges logic (Przełączniki Grafu)
# ---------------------------------------------------------------------------

def route_after_judge(state: FoundryState) -> str:
    score = state.get("quality_score", 0.0)
    retries = state.get("retry_count", 0)
    turn = state.get("turn_count", 0)
    chunk_str = state.get("chunk", {}).get("id", "UNKNOWN")[:8]

    if score >= settings.quality_threshold:
        logger.debug(f"[Router] Ocena {score:.2f} >= {settings.quality_threshold} -> Przejście do 'append_turn'")
        return "append_turn"
        
    if turn > 0:
        logger.info(f"[Router] Słaba jakość ({score:.2f}) na etapie turn={turn}. Zakończenie konwersacji ucinając słaby wkład (chunk={chunk_str}) -> Przejście do 'write_output'")
        return "write_output"
        
    if retries < settings.max_retries_per_chunk:
        logger.debug(f"[Router] Ocena {score:.2f} (wymagane {settings.quality_threshold}). Retry {retries+1}/{settings.max_retries_per_chunk} -> Przejście do 'increment_retry'")
        return "retry"
        
    logger.warning(f"[Router] Wyczerpano limity retry dla chunk={chunk_str} -> Przejście do 'mark_unresolvable'")
    return "mark_unresolvable"


def route_after_append(state: FoundryState) -> str:
    if state.get("turn_count", 0) < settings.max_turns:
        return "simulate_followup"
    return "write_output"


# ---------------------------------------------------------------------------
# Kompilacja Głównego Grafu
# ---------------------------------------------------------------------------

def build_graph(
    session: Session,
    writer: JSONLWriter,
    dpo_writer: Optional[DPOWriter] = None,
    orpo_writer: Optional[ORPOWriter] = None,
    kto_writer: Optional[KTOWriter] = None,
    deduplicator: Optional[MinHashDeduplicator] = None,
) -> Any:
    """
    Kompiluje i zwraca instancję StateGraph LangGraph.
    Reprezentuje główny potok rozumowania i weryfikacji danych.
    """
    graph = StateGraph(FoundryState)

    # Rejestracja węzłów (Nodes) — każdy chroniony nową telemetrią
    # Uwaga: Agenty z importu (np. simulate_question) również owijamy w locie dla spójności
    graph.add_node("simulate_question",       with_telemetry("simulate_question")(simulate_question))
    graph.add_node("simulate_followup",       with_telemetry("simulate_followup")(simulate_followup))
    graph.add_node("retrieve_context",        make_retrieve_node(session))
    graph.add_node("generate_answer",         with_telemetry("generate_answer")(generate_answer))
    graph.add_node("constitutional_revision", with_telemetry("constitutional_revision")(constitutional_revision))
    graph.add_node("judge_answer",            with_telemetry("judge_answer")(judge_answer))
    graph.add_node("append_turn",             append_turn)
    graph.add_node("write_output",            make_write_node(session, writer, dpo_writer, orpo_writer, kto_writer, deduplicator))
    graph.add_node("mark_unresolvable",       make_unresolvable_node(session))
    graph.add_node("increment_retry",         increment_retry)

    # Rejestracja krawędzi (Edges)
    graph.add_edge(START,                       "simulate_question")
    graph.add_edge("simulate_question",         "retrieve_context")
    graph.add_edge("retrieve_context",          "generate_answer")
    graph.add_edge("generate_answer",           "constitutional_revision")
    graph.add_edge("constitutional_revision",   "judge_answer")

    # Krawędzie decyzyjne z Judge
    graph.add_conditional_edges(
        "judge_answer", route_after_judge,
        {
            "append_turn": "append_turn", 
            "retry": "increment_retry",
            "write_output": "write_output", 
            "mark_unresolvable": "mark_unresolvable"
        },
    )

    graph.add_edge("increment_retry",   "retrieve_context")
    
    # Krawędzie decyzyjne z dołączania tur (Multi-turn dialog)
    graph.add_conditional_edges(
        "append_turn", route_after_append,
        {
            "simulate_followup": "simulate_followup", 
            "write_output": "write_output"
        },
    )
    
    graph.add_edge("simulate_followup",   "retrieve_context")
    graph.add_edge("write_output",        END)
    graph.add_edge("mark_unresolvable",   END)

    return graph.compile()
