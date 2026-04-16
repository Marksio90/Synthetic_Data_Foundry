"""
db/repository.py — Data-access layer.

All state mutations go through explicit transactions so a power-cut mid-write
leaves the DB consistent (ACID idempotency — Self-Check patch).
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from db.models import (
    DirectiveChunk,
    GeneratedSample,
    OpenAIBatchJob,
    SourceDocument,
    WatermarkRegistry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Source documents
# =============================================================================

def upsert_source_document(session: Session, **kwargs) -> SourceDocument:
    """Insert or return existing document (dedup by file_hash)."""
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
# Chunks
# =============================================================================

def insert_chunk(session: Session, **kwargs) -> DirectiveChunk:
    chunk = DirectiveChunk(**kwargs)
    session.add(chunk)
    session.flush()
    return chunk


def claim_chunk(session: Session, chunk_id: uuid.UUID) -> bool:
    """Atomically transition chunk new → in_progress. Returns False if already claimed."""
    result = session.execute(
        text("SELECT claim_chunk_for_processing(:cid)"),
        {"cid": chunk_id},
    )
    claimed: bool = result.scalar()
    session.flush()
    return claimed


def finalize_chunk(
    session: Session,
    chunk_id: uuid.UUID,
    success: bool,
    error: Optional[str] = None,
) -> None:
    """Atomically mark chunk ready or bump retry counter (ACID)."""
    session.execute(
        text("SELECT finalize_chunk(:cid, :ok, :err)"),
        {"cid": chunk_id, "ok": success, "err": error},
    )
    session.flush()


def get_pending_chunks(session: Session, limit: int = 100) -> list[DirectiveChunk]:
    """
    Return chunks eligible for processing:
      - status='new' (normal path)
      - status='in_progress' (recovery path: picks up chunks left in_progress
        by a previous crashed run — safe in single-worker deployments)
    Both filtered by retry_count < 3 to skip permanently failed chunks.
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


def hybrid_search(
    session: Session,
    query_embedding: list[float],
    query_text: str,
    top_k: int = 5,
    diversify_by_section: bool = False,
) -> list[DirectiveChunk]:
    """
    Hybrid retrieval: vector cosine similarity (pgvector) + BM25 ts_rank.
    Self-Check 3.0: WHERE is_superseded = FALSE hard-coded.

    Wagi konfigurowane przez settings.hybrid_vector_weight / hybrid_bm25_weight.
    Opcjonalna diversyfikacja: max 2 chunki z każdej sekcji dokumentu.
    """
    from config.settings import settings as _s
    vec_w = _s.hybrid_vector_weight
    bm25_w = _s.hybrid_bm25_weight

    # Przy diversyfikacji pobieramy więcej kandydatów przed filtrowaniem
    fetch_k = top_k * 4 if diversify_by_section else top_k

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
                   + COALESCE(f.bm25_score, 0) * {bm25_w}                   AS score
            FROM   vec v
            FULL   OUTER JOIN fts f ON v.id = f.id
        )
        SELECT directive_chunks.*
        FROM   combined
        JOIN   directive_chunks ON directive_chunks.id = combined.id
        ORDER  BY combined.score DESC, directive_chunks.id ASC
        LIMIT  :fetch_k
        """
    )
    rows = session.execute(
        sql,
        {"emb": str(query_embedding), "q": query_text, "fetch_k": fetch_k},
    ).fetchall()

    ids = [row[0] for row in rows]
    if not ids:
        return []
    chunks_unordered = session.scalars(
        select(DirectiveChunk).where(DirectiveChunk.id.in_(ids))
    ).all()
    id_order = {cid: i for i, cid in enumerate(ids)}
    chunks = sorted(chunks_unordered, key=lambda c: id_order.get(c.id, 999))

    if not diversify_by_section:
        return chunks[:top_k]

    # Diversyfikacja: max 2 chunki z każdej sekcji, zachowując ranking
    by_section: dict[str, int] = {}
    diversified: list[DirectiveChunk] = []
    for chunk in chunks:
        section_key = f"{chunk.source_doc_id}::{chunk.section_heading or '__none__'}"
        count = by_section.get(section_key, 0)
        if count < 2:
            diversified.append(chunk)
            by_section[section_key] = count + 1
        if len(diversified) >= top_k:
            break
    return diversified


# =============================================================================
# Generated samples
# =============================================================================

def insert_sample(session: Session, **kwargs) -> GeneratedSample:
    sample = GeneratedSample(**kwargs)
    session.add(sample)
    session.flush()
    return sample


def mark_sample_written(
    session: Session, sample_id: uuid.UUID, record_index: int, watermark_hash: Optional[str]
) -> None:
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
# Watermark registry
# =============================================================================

def register_watermark(session: Session, **kwargs) -> WatermarkRegistry:
    """Upsert watermark — same batch_id fires multiple times (every N records).
    On conflict: append new record_indices and update total_records + watermark_hash.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stmt = pg_insert(WatermarkRegistry).values(**kwargs)
    stmt = stmt.on_conflict_do_update(
        index_elements=["batch_id"],
        set_={
            # Append new positions to existing array (|| operator in PG)
            "record_indices": text(
                "watermark_registry.record_indices || EXCLUDED.record_indices"
            ),
            "total_records": stmt.excluded.total_records,
            "watermark_hash": stmt.excluded.watermark_hash,
        },
    )
    session.execute(stmt)
    session.flush()
    # Return the (now upserted) row
    return session.scalar(
        select(WatermarkRegistry).where(WatermarkRegistry.batch_id == kwargs["batch_id"])
    )


# =============================================================================
# OpenAI Batch jobs
# =============================================================================

def insert_batch_job(session: Session, **kwargs) -> OpenAIBatchJob:
    job = OpenAIBatchJob(**kwargs)
    session.add(job)
    session.flush()
    return job


def update_batch_job(session: Session, batch_job_id: str, **kwargs) -> None:
    session.execute(
        update(OpenAIBatchJob)
        .where(OpenAIBatchJob.batch_job_id == batch_job_id)
        .values(**kwargs)
    )
    session.flush()
