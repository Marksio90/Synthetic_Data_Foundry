"""
agents/cross_doc.py — Cross-document synthesis Q&A generation.

After individual chunk processing, samples pairs of chunks from DIFFERENT
source documents and generates questions that require synthesising information
from multiple EU directives (CSRD ↔ SFDR ↔ Taksonomia ↔ CSDDD).

These are the highest-value samples in the dataset:
  - They test cross-regulatory knowledge integration
  - They cannot be answered from a single directive alone
  - They represent real-world compliance questions that span frameworks

The node is run as a post-processing pass, not inside the LangGraph loop.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.expert import _call_vllm
from agents.judge import judge_answer as run_judge
from config.settings import settings
from db import repository as repo
from db.models import DirectiveChunk, SourceDocument
from pipeline.state import FoundryState
from utils.classifier import classify_question
from utils.dedup import MinHashDeduplicator
from utils.output import JSONLWriter

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

if TYPE_CHECKING:
    from utils.output import DPOWriter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapa kompatybilnych par dokumentów — tylko powiązane akty prawne UE
# ---------------------------------------------------------------------------
# Klucze to fragmenty nazwy dyrektywy (case-insensitive substring match).
# Wartość: lista fragmentów nazw, z którymi dana dyrektywa jest kompatybilna.
_COMPATIBLE_DIRECTIVES: dict[str, list[str]] = {
    "csrd":       ["sfdr", "taxonomy", "taksonomia", "csddd"],
    "sfdr":       ["csrd", "taxonomy", "taksonomia"],
    "taxonomy":   ["csrd", "sfdr", "csddd"],
    "taksonomia": ["csrd", "sfdr", "csddd"],
    "csddd":      ["csrd", "taxonomy", "taksonomia"],
}


def _are_compatible(name_a: str, name_b: str) -> bool:
    """
    Sprawdza, czy dwa dokumenty tworzą sensowną parę cross-doc.
    Domyślnie akceptuje wszystkie pary, jeśli żaden dokument nie jest w mapie.
    """
    a_lower = name_a.lower()
    b_lower = name_b.lower()

    for key, compat_list in _COMPATIBLE_DIRECTIVES.items():
        if key in a_lower:
            return any(c in b_lower for c in compat_list)
        if key in b_lower:
            return any(c in a_lower for c in compat_list)

    # Nieznana para — akceptuj (zachowanie wsteczne)
    return True


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_CROSS = (
    "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE. "
    "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
    "Gdy pytanie wymaga syntezy z wielu aktów prawnych, powiąż je precyzyjnie, "
    "wskazując numery artykułów i ich wzajemne relacje. "
    "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie.\""
)

_QUESTION_SYSTEM = (
    "Jesteś ekspertem ESG zadającym pytania wymagające syntezy z WIELU dyrektyw UE.\n\n"
    "Przeczytaj poniższe fragmenty z RÓŻNYCH dyrektyw. Zadaj JEDNO pytanie, które:\n"
    "1. Wymaga informacji z CO NAJMNIEJ DWÓCH podanych fragmentów\n"
    "2. Dotyczy zależności, różnic lub spójności między aktami prawnymi\n"
    "3. Jest praktycznie istotne dla compliance ESG\n"
    "4. Brzmi profesjonalnie i precyzyjnie\n\n"
    "Przykłady dobrych pytań:\n"
    "- \"Jak wymogi ujawnień SFDR art. 8 odnoszą się do kryteriów Taksonomii UE?\"\n"
    "- \"W jaki sposób CSRD rozszerza obowiązki raportowe wcześniej nałożone przez SFDR?\"\n"
    "- \"Jakie są różnice między zakresem podmiotowym CSRD a CSDDD?\"\n\n"
    "Odpowiedz TYLKO pytaniem — bez wstępu ani komentarza."
)

_ANSWER_SYSTEM = (
    "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE odpowiadającym na pytanie "
    "wymagające syntezy wiedzy z wielu dyrektyw.\n\n"
    "ZASADY:\n"
    "1. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów podanych w KONTEKST.\n"
    "2. Jeśli pytanie wykracza poza kontekst, odpowiedz DOKŁADNIE: "
    "\"Brak danych w dyrektywie.\"\n"
    "3. Cytuj numery artykułów i ustępów z odpowiednich dyrektyw.\n"
    "4. Wyraźnie wskazuj, z której dyrektywy pochodzi każda informacja.\n"
    "5. Odpowiadaj po polsku, precyzyjnie (3–10 zdań).\n\n"
    "FORMAT ODPOWIEDZI:\n"
    "Przed właściwą odpowiedzią krótko przemyśl w tagach:\n"
    "<reasoning>\n"
    "Dyrektywy: [które akty są relevantne]\n"
    "Powiązanie: [jak fragmenty łączą się z pytaniem]\n"
    "Wniosek: [co wynika z syntezy]\n"
    "</reasoning>\n\n"
    "[Właściwa odpowiedź po polsku, 3–10 zdań]"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_centroid(chunks: list[DirectiveChunk]) -> "list[float] | None":
    """Mean-normalised embedding centroid for a document's chunks."""
    if not _HAS_NUMPY:
        return None
    embeddings = [c.embedding for c in chunks if c.embedding is not None]
    if not embeddings:
        return None
    arr = np.array(embeddings, dtype=np.float32)
    centroid = arr.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    return (centroid / norm).tolist() if norm > 1e-9 else centroid.tolist()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not _HAS_NUMPY:
        return 0.5
    return float(np.dot(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)))


def _pick_doc_pair(
    by_doc: dict[str, list[DirectiveChunk]],
    doc_names: dict[str, str],
) -> tuple[str, str] | tuple[None, None]:
    """
    Pick two document IDs biased toward topically similar pairs via softmax sampling.
    Respects _are_compatible constraint. Falls back to random if numpy is absent.
    """
    doc_ids = list(by_doc.keys())
    if len(doc_ids) < 2:
        return None, None

    doc_a = random.choice(doc_ids)
    name_a = doc_names.get(doc_a, "")

    if not _HAS_NUMPY:
        candidates = [d for d in doc_ids if d != doc_a and _are_compatible(name_a, doc_names.get(d, ""))]
        return (doc_a, random.choice(candidates)) if candidates else (None, None)

    centroid_a = _doc_centroid(by_doc[doc_a])

    scored: list[tuple[float, str]] = []
    for doc_b in doc_ids:
        if doc_b == doc_a:
            continue
        name_b = doc_names.get(doc_b, "")
        if not _are_compatible(name_a, name_b):
            continue
        c_b = _doc_centroid(by_doc[doc_b])
        sim = _cosine_sim(centroid_a, c_b) if (centroid_a and c_b) else 0.5
        scored.append((sim, doc_b))

    if not scored:
        return None, None

    scores = np.array([s for s, _ in scored], dtype=np.float32)
    # Tempered softmax (T=0.4): focuses sampling on similar docs but keeps diversity
    shifted = (scores - scores.max()) / 0.4
    probs = np.exp(shifted)
    probs /= probs.sum()
    idx = int(np.random.choice(len(scored), p=probs))
    return doc_a, scored[idx][1]


def _get_chunks_by_document(session: Session) -> dict[str, list[DirectiveChunk]]:
    """Return ready, non-superseded chunks grouped by source_doc_id string."""
    chunks = session.scalars(
        select(DirectiveChunk)
        .where(DirectiveChunk.status == "ready")
        .where(DirectiveChunk.is_superseded == False)  # noqa: E712
        .where(DirectiveChunk.embedding.isnot(None))
    ).all()

    by_doc: dict[str, list[DirectiveChunk]] = {}
    for chunk in chunks:
        key = str(chunk.source_doc_id)
        by_doc.setdefault(key, []).append(chunk)
    return by_doc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_cross_doc_samples(
    session: Session,
    writer: JSONLWriter,
    dpo_writer: Optional["DPOWriter"],
    deduplicator: MinHashDeduplicator,
    batch_id: str,
    n_samples: int = 50,
) -> int:
    """
    Generate *n_samples* cross-document Q&A records and write them to *writer*.

    Returns the number of records successfully written.
    """
    if n_samples <= 0:
        return 0

    by_doc = _get_chunks_by_document(session)
    doc_ids = list(by_doc.keys())

    if len(doc_ids) < 2:
        logger.warning(
            "Cross-doc: need ≥2 source documents, found %d — skipping pass",
            len(doc_ids),
        )
        return 0

    # Precompute document names once (avoids repeated DB queries per attempt)
    doc_names: dict[str, str] = {}
    for did in doc_ids:
        first_chunk = by_doc[did][0]
        doc_obj = session.get(SourceDocument, first_chunk.source_doc_id)
        if doc_obj:
            doc_names[did] = doc_obj.directive_name or doc_obj.filename or "Document"

    logger.info(
        "Cross-doc pass: %d source documents, targeting %d samples%s",
        len(doc_ids), n_samples,
        " (similarity-biased pairs)" if _HAS_NUMPY else " (random pairs)",
    )

    written = 0
    attempts = 0
    max_attempts = n_samples * 5  # allow generous retries

    while written < n_samples and attempts < max_attempts:
        attempts += 1

        # Pick similarity-biased compatible document pair
        id_a, id_b = _pick_doc_pair(by_doc, doc_names)
        if id_a is None or id_b is None:
            logger.warning("Cross-doc: no compatible document pair found — stopping")
            break

        chunk_a = random.choice(by_doc[id_a])
        chunk_b = random.choice(by_doc[id_b])
        name_a = doc_names.get(id_a, "Dyrektywa A")
        name_b = doc_names.get(id_b, "Dyrektywa B")

        context = (
            f"[Fragment z {name_a}]\n{chunk_a.content}\n\n"
            f"[Fragment z {name_b}]\n{chunk_b.content}"
        )

        # ── Step 1: Generate cross-doc question ───────────────────────────
        try:
            question = _call_vllm(_QUESTION_SYSTEM, context)
        except Exception as exc:
            logger.debug("Cross-doc question generation failed: %s", exc)
            continue

        # Near-duplicate check
        if deduplicator.is_duplicate(question):
            logger.debug("Cross-doc: duplicate question — skipping")
            continue

        # ── Step 2: Generate answer ───────────────────────────────────────
        user_prompt = (
            f"KONTEKST:\n{context}\n\n"
            f"PYTANIE: {question}\n\n"
            f"ODPOWIEDŹ:"
        )
        try:
            answer = _call_vllm(_ANSWER_SYSTEM, user_prompt)
        except Exception as exc:
            logger.debug("Cross-doc answer generation failed: %s", exc)
            continue

        # ── Step 3: Judge quality ─────────────────────────────────────────
        # Build a minimal FoundryState for the judge
        fake_state: FoundryState = {
            "chunk": {
                "id": str(chunk_a.id),
                "content": chunk_a.content,
                "content_md": chunk_a.content_md or chunk_a.content,
                "source_document": name_a,
                "chunk_index": chunk_a.chunk_index,
                "section_heading": chunk_a.section_heading or "",
                "valid_from_date": "",
            },
            "question": question,
            "answer": answer,
            "is_adversarial": False,
            "retrieved_context": [chunk_a.content, chunk_b.content],
            "retrieved_ids": [str(chunk_a.id), str(chunk_b.id)],
            "quality_score": 0.0,
            "judge_model": "",
            "judge_reasoning": "",
            "retry_count": 0,
            "status": "in_progress",
            "error_message": None,
            "conversation_history": [],
            "turn_count": 0,
            "perspective": "cross_doc",
            "rejected_answer": None,
            "question_type": "comparative",
            "difficulty": "hard",
            "sample_id": None,
            "batch_id": batch_id,
            "record_index": writer.record_count,
        }

        judge_result = run_judge(fake_state)
        score: float = judge_result.get("quality_score", 0.0)

        if score < settings.quality_threshold:
            logger.debug(
                "Cross-doc: score %.2f < %.2f — skipping",
                score, settings.quality_threshold,
            )
            continue

        # ── Step 4: Classify ──────────────────────────────────────────────
        q_type, difficulty = classify_question(question)
        # Cross-doc questions are always at least "hard"
        if difficulty != "hard":
            difficulty = "hard"

        # ── Step 5: Build messages ────────────────────────────────────────
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_CROSS},
            {"role": "user",   "content": question},
            {"role": "assistant", "content": answer},
        ]

        # Refusal cap pre-check
        if writer.should_skip(messages):
            continue

        # ── Step 6: DB insert ─────────────────────────────────────────────
        try:
            sample = repo.insert_sample(
                session,
                chunk_id=chunk_a.id,
                question=question,
                answer=answer,
                system_prompt=_SYSTEM_PROMPT_CROSS,
                is_adversarial=False,
                quality_score=score,
                judge_model=judge_result.get("judge_model", ""),
                judge_reasoning=judge_result.get("judge_reasoning", ""),
                batch_id=batch_id,
                perspective="cross_doc",
                conversation_json=messages,
                question_type=q_type,
                difficulty=difficulty,
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("Cross-doc DB write failed: %s", exc)
            continue

        # ── Step 7: Write JSONL ───────────────────────────────────────────
        metadata = {
            "perspective": "cross_doc",
            "question_type": q_type,
            "difficulty": difficulty,
            "source_documents": [name_a, name_b],
            "batch_id": batch_id,
        }
        record_index, watermark_hash = writer.write_conversation(messages, metadata=metadata)
        if record_index == -1:
            continue

        try:
            repo.mark_sample_written(session, sample.id, record_index, watermark_hash)
            session.commit()
        except Exception:
            session.rollback()

        written += 1
        logger.debug(
            "Cross-doc record %d written (score=%.2f type=%s)",
            record_index, score, q_type,
        )

    logger.info(
        "Cross-doc pass complete: %d/%d samples written (%d attempts)",
        written, n_samples, attempts,
    )
    return written
