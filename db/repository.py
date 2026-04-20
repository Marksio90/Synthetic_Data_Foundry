"""
db/repository.py — Data-Access Layer (DAL) - ENTERPRISE EDITION.

Warstwa abstrakcji dla bazy danych PostgreSQL + pgvector.
Zaprojektowana pod kątem maksymalnej współbieżności i wysokiego obciążenia (High-Concurrency).

Funkcje PRO:
- @with_db_retries: Automatyczne rozwiązywanie deadlocków i przerw w połączeniu (Exponential Backoff).
- Telemetria zapytań: Monitorowanie wydajności wyszukiwania wektorowego (Slow Query Logger).
- SQLAlchemy 2.0 Compliance: Bezpieczne wiązanie parametrów zapytań surowych (bindparam).
"""

from __future__ import annotations

import logging
import time
import uuid
from functools import wraps
from typing import Optional, List, TypeVar, Callable, Any

from sqlalchemy import select, text, update, bindparam
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError, PendingRollbackError, DBAPIError

from db.models import (
    DirectiveChunk,
    GeneratedSample,
    OpenAIBatchJob,
    SourceDocument,
    WatermarkRegistry,
)

logger = logging.getLogger("foundry.db.repository")

# Typ generyczny do adnotacji funkcji zwracających dowolny typ z SQLAlchemy
T = TypeVar("T")

# ---------------------------------------------------------------------------
# Dekoratory Odpornościowe (Resilience / Auto-Recovery)
# ---------------------------------------------------------------------------
def with_db_retries(max_retries: int = 3, initial_wait: float = 0.2) -> Callable:
    """
    Dekorator wdrażający wzorzec Exponential Backoff dla operacji bazodanowych.
    Automatycznie chwyta i ponawia transakcje w przypadku deadlocków (SQLSTATE 40P01)
    lub błędów serializacji (SQLSTATE 40001), które są częste przy wysokiej współbieżności.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(session: Session, *args: Any, **kwargs: Any) -> T:
            retries = 0
            while True:
                try:
                    return func(session, *args, **kwargs)
                except (OperationalError, PendingRollbackError) as e:
                    # Rozpoznajemy tymczasowe błędy współbieżności
                    session.rollback()
                    retries += 1
                    if retries > max_retries:
                        logger.error(
                            f"KRYTYCZNY BŁĄD BAZY: Osiągnięto limit {max_retries} prób w funkcji '{func.__name__}'. "
                            f"Treść błędu: {e}"
                        )
                        raise
                    
                    sleep_time = initial_wait * (2 ** (retries - 1))
                    logger.warning(
                        f"[DB Retry] Tymczasowy błąd transakcji w '{func.__name__}' (np. Deadlock). "
                        f"Ponawiam próbę {retries}/{max_retries} za {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                except DBAPIError as e:
                    # Inne krytyczne błędy bazy wyrzucamy natychmiast po wykonaniu rollbacku
                    session.rollback()
                    logger.error(f"Błąd silnika DB w '{func.__name__}': {e}", exc_info=True)
                    raise
                except Exception as e:
                    # Błędy aplikacji również wymagają cofnięcia stanu sesji
                    session.rollback()
                    raise
        return wrapper
    return decorator


# =============================================================================
# Source Documents
# =============================================================================
@with_db_retries(max_retries=2)
def upsert_source_document(session: Session, **kwargs: Any) -> SourceDocument:
    """Wstawia nowy dokument z pliku lub zwraca istniejący (deduplikacja po file_hash)."""
    existing = session.scalar(
        select(SourceDocument).where(SourceDocument.file_hash == kwargs["file_hash"])
    )
    if existing:
        return existing
    
    doc = SourceDocument(**kwargs)
    session.add(doc)
    session.flush()
    return doc


# =============================================================================
# Chunks (Przetwarzanie i Rezerwacje)
# =============================================================================
@with_db_retries(max_retries=3)
def insert_chunk(session: Session, **kwargs: Any) -> DirectiveChunk:
    chunk = DirectiveChunk(**kwargs)
    session.add(chunk)
    session.flush()
    return chunk


@with_db_retries(max_retries=5)
def claim_chunk(session: Session, chunk_id: uuid.UUID) -> bool:
    """
    Atomowa rezerwacja fragmentu (new → in_progress).
    Chroni przed sytuacją, gdzie dwa wątki AutoPilota przetwarzają ten sam chunk.
    """
    stmt = text("SELECT claim_chunk_for_processing(:cid)").bindparams(
        bindparam("cid", value=str(chunk_id))
    )
    result = session.execute(stmt)
    claimed: bool = result.scalar()
    session.flush()
    return claimed


@with_db_retries(max_retries=3)
def finalize_chunk(
    session: Session,
    chunk_id: uuid.UUID,
    success: bool,
    error: Optional[str] = None,
) -> None:
    """Atomowo oznacza chunk jako ukończony (ready) lub podbija licznik błędów."""
    stmt = text("SELECT finalize_chunk(:cid, :ok, :err)").bindparams(
        bindparam("cid", value=str(chunk_id)),
        bindparam("ok", value=success),
        bindparam("err", value=error)
    )
    session.execute(stmt)
    session.flush()


@with_db_retries(max_retries=2)
def get_pending_chunks(session: Session, limit: int = 100) -> List[DirectiveChunk]:
    """
    Pobiera chunki oczekujące na przetworzenie (lub te zablokowane po padzie serwera).
    Filtruje chunki, które wyczerpały limit prób (retry_count >= 3).
    """
    return list(
        session.scalars(
            select(DirectiveChunk)
            .where(DirectiveChunk.status.in_(["new", "in_progress"]))
            .where(DirectiveChunk.retry_count < 3)
            .order_by(DirectiveChunk.created_at)
            .limit(limit)
        )
    )


# =============================================================================
# RAG: Hybrid Search (pgvector + BM25)
# =============================================================================
@with_db_retries(max_retries=2)
def hybrid_search(
    session: Session,
    query_embedding: List[float],
    query_text: str,
    top_k: int = 5,
    diversify_by_section: bool = False,
    vector_weight: Optional[float] = None,
    bm25_weight: Optional[float] = None,
) -> List[DirectiveChunk]:
    """
    Zaawansowane wyszukiwanie hybrydowe: Cosine Similarity (pgvector) + Full-Text Search (BM25).
    Zaimplementowana telemetria pozwala zidentyfikować spowolnienia na indeksach bazy danych.
    """
    start_time = time.perf_counter()
    from config.settings import settings as _s
    vec_w = _s.hybrid_vector_weight if vector_weight is None else vector_weight
    bm25_w = _s.hybrid_bm25_weight if bm25_weight is None else bm25_weight

    # Pobieramy szerszą pule kandydatów jeśli wymagana jest dywersyfikacja semantyczna
    fetch_k = top_k * 4 if diversify_by_section else top_k

    # Zoptymalizowane zapytanie CTE (Common Table Expression) - wymusza użycie indeksu HNSW
    sql = text(
        f"""
        WITH vec AS (
            SELECT id,
                   1 - (embedding <=> CAST(:emb AS vector)) AS vec_score
            FROM   directive_chunks
            WHERE  is_superseded = FALSE
              AND  embedding IS NOT NULL
            ORDER  BY embedding <=> CAST(:emb AS vector)
            LIMIT  :fetch_k
        ),
        fts AS (
            SELECT id,
                   ts_rank(fts_vector, plainto_tsquery('simple', :q)) AS bm25_score
            FROM   directive_chunks
            WHERE  is_superseded = FALSE
              AND  fts_vector @@ plainto_tsquery('simple', :q)
            LIMIT  :fetch_k
        ),
        combined AS (
            SELECT COALESCE(v.id, f.id)                                      AS id,
                   COALESCE(v.vec_score, 0) * {vec_w}
                   + COALESCE(f.bm25_score, 0) * {bm25_w}                    AS score
            FROM   vec v
            FULL   OUTER JOIN fts f ON v.id = f.id
        )
        SELECT directive_chunks.*
        FROM   combined
        JOIN   directive_chunks ON directive_chunks.id = combined.id
        ORDER  BY combined.score DESC, directive_chunks.id ASC
        LIMIT  :fetch_k
        """
    ).bindparams(
        bindparam("emb", value=str(query_embedding)),
        bindparam("q", value=query_text),
        bindparam("fetch_k", value=fetch_k)
    )
    
    rows = session.execute(sql).fetchall()

    ids = [row[0] for row in rows]
    if not ids:
        return []
        
    # Pobranie modeli ORM na podstawie zidentyfikowanych ID
    chunks_unordered = session.scalars(
        select(DirectiveChunk).where(DirectiveChunk.id.in_(ids))
    ).all()
    
    # Przywracanie poprawnej kolejności rankingowej
    id_order = {cid: i for i, cid in enumerate(ids)}
    chunks = sorted(chunks_unordered, key=lambda c: id_order.get(c.id, 999))

    # Algorytm dywersyfikacji (Max 2 fragmenty z jednej sekcji dokumentu)
    if diversify_by_section:
        by_section: dict[str, int] = {}
        diversified: List[DirectiveChunk] = []
        for chunk in chunks:
            section_key = f"{chunk.source_doc_id}::{chunk.section_heading or '__none__'}"
            count = by_section.get(section_key, 0)
            if count < 2:
                diversified.append(chunk)
                by_section[section_key] = count + 1
            if len(diversified) >= top_k:
                break
        chunks = diversified
    else:
        chunks = chunks[:top_k]

    # Telemetria wydajnościowa zapytania wektorowego (Slow Query Warning)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    if elapsed_ms > 500:
        logger.warning(f"[SLOW QUERY] Wyszukiwanie hybrydowe trwało {elapsed_ms:.1f}ms (Zwrócono {len(chunks)} wyników).")
    else:
        logger.debug(f"Wyszukiwanie hybrydowe udane: {elapsed_ms:.1f}ms.")

    return chunks


# =============================================================================
# Generated Samples (Zapis zbiorów uczących)
# =============================================================================
@with_db_retries(max_retries=3)
def insert_sample(session: Session, **kwargs: Any) -> GeneratedSample:
    """Zapisuje wygenerowaną przez LLM próbkę treningową Q&A."""
    sample = GeneratedSample(**kwargs)
    session.add(sample)
    session.flush()
    return sample


@with_db_retries(max_retries=3)
def mark_sample_written(
    session: Session, sample_id: uuid.UUID, record_index: int, watermark_hash: Optional[str]
) -> None:
    """Oznacza próbkę jako zapisaną fizycznie na dysku (JSONL) oraz przypisuje znak wodny."""
    session.execute(
        update(GeneratedSample)
        .where(GeneratedSample.id == sample_id)
        .values(
            written_to_file=True,
            record_index=record_index,
            watermark_hash=watermark_hash,
        )
    )
    session.flush()


# =============================================================================
# Watermark Registry (Rejestr Znaków Wodnych)
# =============================================================================
@with_db_retries(max_retries=4)
def register_watermark(session: Session, **kwargs: Any) -> WatermarkRegistry:
    """
    Bezpieczny upsert znaku wodnego — ten sam batch_id wywoływany wielokrotnie (co N rekordów).
    Rozwiązuje konflikty asynchroniczne za pomocą ON CONFLICT (baza dba o spojność).
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stmt = pg_insert(WatermarkRegistry).values(**kwargs)
    stmt = stmt.on_conflict_do_update(
        index_elements=["batch_id"],
        set_={
            # Bezpieczne łączenie list po stronie bazy (operator || w PG)
            "record_indices": text("watermark_registry.record_indices || EXCLUDED.record_indices"),
            "total_records": stmt.excluded.total_records,
            "watermark_hash": stmt.excluded.watermark_hash,
        },
    )
    session.execute(stmt)
    session.flush()
    
    # Pobranie ostatecznie uaktualnionego wiersza
    return session.scalar(
        select(WatermarkRegistry).where(WatermarkRegistry.batch_id == kwargs["batch_id"])
    )


# =============================================================================
# OpenAI Batch Jobs
# =============================================================================
@with_db_retries(max_retries=2)
def insert_batch_job(session: Session, **kwargs: Any) -> OpenAIBatchJob:
    job = OpenAIBatchJob(**kwargs)
    session.add(job)
    session.flush()
    return job


@with_db_retries(max_retries=3)
def update_batch_job(session: Session, batch_job_id: str, **kwargs: Any) -> None:
    session.execute(
        update(OpenAIBatchJob)
        .where(OpenAIBatchJob.batch_job_id == batch_job_id)
        .values(**kwargs)
    )
    session.flush()
