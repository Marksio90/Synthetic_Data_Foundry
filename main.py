"""
main.py — ESG Data Foundry entry point.

Usage:
    python main.py --pdf data/csrd.pdf [--pdf data/sfdr.pdf ...] [--batch-id my-run-001]

Pipeline:
    1. For each PDF: Ingestor → Chunker (stores chunks in PG)
    2. For each pending chunk: run LangGraph swarm
         Simulator → Expert (RAG + CoT) → Judge → write/retry/skip
    3. Cross-document synthesis pass (chunks from different directives)
    4. Output:
         output/dataset_esg_v1.jsonl       — SFT ChatML (main dataset)
         output/dataset_esg_v1_dpo.jsonl   — DPO preference pairs
         output/dataset_esg_v1.datacard.json — statistics / quality report

Idempotent: re-running after a crash skips already-processed chunks.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path

import openai
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import sessionmaker

from agents.chunker import chunk_document
from agents.cross_doc import generate_cross_doc_samples
from agents.expert import _is_tpd_limit
from agents.ingestor import ingest_pdf
from config.settings import settings
from db.models import Base, DirectiveChunk, SourceDocument
from db.repository import claim_chunk, get_pending_chunks
from pipeline.graph import build_graph
from pipeline.state import FoundryState
from utils.datacard import generate_datacard
from utils.dedup import MinHashDeduplicator
from utils.output import DPOWriter, JSONLWriter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("foundry.main")


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def get_engine():
    return create_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )


def create_tables(engine):
    Base.metadata.create_all(engine)


def run_migrations(engine) -> None:
    """
    Idempotent schema patches applied on every startup.

    Działa zarówno w Dockerze (gdzie init/01_schema.sql był uruchamiany przez
    postgres entrypoint) jak i w lokalnym środowisku deweloperskim (gdzie
    create_all tworzy tabele, ale nie uruchamia skryptów SQL).

    Patch 0: Rozszerzenia PostgreSQL (pgvector, pg_trgm).
    Patch 1: Kolumna fts_vector + trigger do automatycznej indeksacji FTS.
    Patch 2: Stored procedure claim_chunk_for_processing (używana w repository.py).
    Patch 3: Status-guarded finalize_chunk.
    Patch 4-10: Nowe kolumny na generated_samples (IF NOT EXISTS).
    """
    with engine.connect() as conn:
        # ── Patch 0: Rozszerzenia ─────────────────────────────────────────────
        for ext in ("vector", "pg_trgm", '"uuid-ossp"'):
            conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {ext};"))

        # ── Patch 1a: kolumna fts_vector ──────────────────────────────────────
        conn.execute(text(
            "ALTER TABLE directive_chunks "
            "ADD COLUMN IF NOT EXISTS fts_vector TSVECTOR;"
        ))

        # ── Patch 1b: funkcja triggerowa update_chunk_fts ─────────────────────
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION update_chunk_fts()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.fts_vector := to_tsvector('simple', COALESCE(NEW.content, ''));
                NEW.updated_at  := NOW();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """))

        # ── Patch 1c: trigger (DROP + CREATE zamiast IF NOT EXISTS) ───────────
        conn.execute(text(
            "DROP TRIGGER IF EXISTS trg_chunk_fts ON directive_chunks;"
        ))
        conn.execute(text("""
            CREATE TRIGGER trg_chunk_fts
                BEFORE INSERT OR UPDATE OF content
                ON directive_chunks
                FOR EACH ROW EXECUTE FUNCTION update_chunk_fts();
        """))

        # ── Patch 1d: GIN index na fts_vector ────────────────────────────────
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chunks_fts "
            "ON directive_chunks USING gin (fts_vector);"
        ))

        # ── Patch 1e: IVFFlat ANN index (best-effort — może nie istnieć) ─────
        try:
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding "
                "ON directive_chunks "
                "USING ivfflat (embedding vector_cosine_ops) "
                "WITH (lists = 100);"
            ))
        except Exception as exc:
            logger.warning(
                "IVFFlat index creation skipped (insufficient data or pgvector missing): %s", exc
            )
            conn.rollback()
            # Ponownie otwórz transakcję po rollback
            conn.execute(text("SELECT 1"))

        # ── Patch 2: claim_chunk_for_processing ───────────────────────────────
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION claim_chunk_for_processing(p_chunk_id UUID)
            RETURNS BOOLEAN AS $$
            DECLARE
                rows_updated INTEGER;
            BEGIN
                UPDATE directive_chunks
                SET    status = 'in_progress', updated_at = NOW()
                WHERE  id = p_chunk_id
                  AND  status = 'new'
                  AND  retry_count < 3;
                GET DIAGNOSTICS rows_updated = ROW_COUNT;
                RETURN rows_updated = 1;
            END;
            $$ LANGUAGE plpgsql;
        """))

        # ── Patch 3: status-guarded finalize_chunk ────────────────────────────
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION finalize_chunk(
                p_chunk_id  UUID,
                p_success   BOOLEAN,
                p_error     TEXT DEFAULT NULL
            )
            RETURNS VOID AS $$
            BEGIN
                IF p_success THEN
                    UPDATE directive_chunks
                    SET status = 'ready', updated_at = NOW()
                    WHERE id = p_chunk_id
                      AND status != 'ready';
                ELSE
                    UPDATE directive_chunks
                    SET
                        retry_count = retry_count + 1,
                        status = CASE
                                    WHEN retry_count + 1 >= 3 THEN 'unresolvable'
                                    ELSE 'new'
                                 END,
                        error_log = p_error,
                        updated_at = NOW()
                    WHERE id = p_chunk_id
                      AND status NOT IN ('ready', 'unresolvable');
                END IF;
            END;
            $$ LANGUAGE plpgsql;
        """))

        # ── Patch 4–10: nowe kolumny na generated_samples ────────────────────
        for col_ddl in [
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS perspective TEXT",
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS conversation_json JSONB",
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS question_type TEXT",
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS difficulty TEXT",
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS rejected_answer TEXT",
            # Sprint 2 — Auto-Reviewer
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS human_reviewed BOOLEAN",
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS human_flag TEXT",
        ]:
            conn.execute(text(col_ddl + ";"))

        conn.commit()
    logger.info(
        "DB migrations applied: extensions + fts_vector + claim/finalize funcs + 7 schema patches"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESG Data Foundry pipeline")
    p.add_argument(
        "--pdf",
        dest="pdfs",
        action="append",
        metavar="PATH",
        default=[],
        help="Path to a directive PDF (can be specified multiple times)",
    )
    p.add_argument(
        "--data-dir",
        dest="data_dir",
        metavar="DIR",
        default=None,
        help="Process all *.pdf files in this directory (recursive)",
    )
    p.add_argument(
        "--batch-id",
        default=f"batch-{uuid.uuid4().hex[:8]}",
        help="Logical batch identifier (default: random UUID prefix)",
    )
    p.add_argument(
        "--chunk-limit",
        type=int,
        default=0,
        help="Process at most N chunks (0 = unlimited; useful for testing)",
    )
    args = p.parse_args()

    # Collect PDFs from --data-dir
    if args.data_dir:
        dir_path = Path(args.data_dir)
        if not dir_path.is_dir():
            p.error(f"--data-dir does not exist or is not a directory: {args.data_dir}")
        found = sorted(dir_path.glob("*.pdf"))
        if not found:
            p.error(f"No *.pdf files found in: {args.data_dir}")
        args.pdfs.extend(str(f) for f in found)

    if not args.pdfs:
        p.error("Provide at least one --pdf PATH or --data-dir DIR")

    return args


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    engine = get_engine()
    create_tables(engine)
    run_migrations(engine)
    Session = sessionmaker(bind=engine)

    # Writers & quality tools (shared across all chunks)
    writer = JSONLWriter(
        output_path=settings.output_file,
        client_id=settings.client_id,
        batch_id=args.batch_id,
        watermark_interval=settings.watermark_interval,
    )
    dpo_writer = DPOWriter(output_path=settings.dpo_output_file)
    deduplicator = MinHashDeduplicator(threshold=settings.dedup_threshold)
    # Pre-load existing questions so resume runs don't generate duplicates
    deduplicator.load_from_jsonl(settings.output_file)

    # ── Phase 1: Ingest & Chunk all PDFs ────────────────────────────────────
    all_pdf_paths = [Path(p) for p in args.pdfs]
    for pdf_path in all_pdf_paths:
        if not pdf_path.exists():
            logger.error("PDF not found: %s — skipping", pdf_path)
            continue

        logger.info("=== Ingesting: %s ===", pdf_path.name)
        with Session() as session:
            source_doc_id, markdown = ingest_pdf(str(pdf_path), session)

            doc = session.scalar(
                select(SourceDocument).where(SourceDocument.id == uuid.UUID(source_doc_id))
            )
            valid_from = doc.valid_from_date if doc else None

            existing_count = session.scalar(
                select(func.count(DirectiveChunk.id)).where(
                    DirectiveChunk.source_doc_id == uuid.UUID(source_doc_id)
                )
            )
            if existing_count > 0:
                logger.info(
                    "Chunks already exist for %s (%d chunks) — skipping chunking",
                    pdf_path.name, existing_count,
                )
            else:
                chunk_ids = chunk_document(
                    session,
                    source_doc_id=source_doc_id,
                    markdown=markdown,
                    valid_from_date=valid_from,
                )
                logger.info("Chunked %s → %d chunks", pdf_path.name, len(chunk_ids))

    # ── Phase 2: Run swarm on pending chunks ────────────────────────────────
    logger.info("=== Starting swarm on pending chunks (batch=%s) ===", args.batch_id)

    total_processed = 0
    total_ready = 0
    total_unresolvable = 0

    with Session() as session:
        graph = build_graph(session, writer, dpo_writer, deduplicator)

        limit = args.chunk_limit if args.chunk_limit > 0 else 10_000
        pending = get_pending_chunks(session, limit=limit)
        logger.info("Found %d pending chunks", len(pending))

        perspectives = ["cfo", "prawnik", "audytor"]

        for chunk in pending:
            chunk_meta = {
                "id": str(chunk.id),
                "content": chunk.content,
                "content_md": chunk.content_md or chunk.content,
                "source_document": (
                    chunk.source_doc.filename if chunk.source_doc else "unknown"
                ),
                "chunk_index": chunk.chunk_index,
                "section_heading": chunk.section_heading or "",
                "valid_from_date": (
                    chunk.valid_from_date.isoformat() if chunk.valid_from_date else ""
                ),
            }

            # Atomically claim chunk (new → in_progress) before processing.
            # Prevents concurrent workers from double-processing the same chunk.
            # Also safe for re-runs: if chunk is already in_progress (crashed run),
            # claim_chunk returns False but we still process it (recovery path).
            claimed = claim_chunk(session, chunk.id)
            if not claimed:
                logger.debug("Chunk %s not freshly claimed (in_progress recovery)", chunk.id)
            try:
                session.commit()
            except Exception:
                session.rollback()

            # Run pipeline once per perspective (CFO / prawnik / audytor).
            # Each run produces one multi-turn conversation record (up to max_turns turns).
            chunk_ready = False
            for perspective in perspectives:
                initial_state: FoundryState = {
                    "chunk": chunk_meta,
                    "perspective": perspective,
                    "question": "",
                    "is_adversarial": False,
                    "retrieved_context": [],
                    "retrieved_ids": [],
                    "answer": "",
                    "quality_score": 0.0,
                    "judge_model": "",
                    "judge_reasoning": "",
                    "retry_count": 0,
                    "status": "in_progress",
                    "error_message": None,
                    "conversation_history": [],
                    "turn_count": 0,
                    "rejected_answer": None,
                    "question_type": "factual",
                    "difficulty": "medium",
                    "sample_id": None,
                    "batch_id": args.batch_id,
                    "record_index": writer.record_count,
                }

                try:
                    final_state = graph.invoke(initial_state)
                    final_status = final_state.get("status", "unknown")
                    if final_status == "ready":
                        chunk_ready = True
                except openai.RateLimitError as exc:
                    if _is_tpd_limit(exc):
                        # Dzienny limit TPD wyczerpany — nie ma sensu kontynuować
                        logger.critical(
                            "⛔ Dzienny limit TPD wyczerpany! Poczekaj do północy (UTC) "
                            "lub zmień providera: SECONDARY_API_KEY / SECONDARY_MODEL. Błąd: %s", exc
                        )
                        sys.exit(1)
                    logger.error(
                        "Graph failed for chunk %s [%s]: %s", chunk.id, perspective, exc
                    )
                except Exception as exc:
                    logger.error(
                        "Graph failed for chunk %s [%s]: %s", chunk.id, perspective, exc
                    )

            total_processed += 1
            if chunk_ready:
                total_ready += 1
            else:
                total_unresolvable += 1

            # Throttle — prevents secondary/OpenAI 429 rate-limit storms.
            # Cerebras: 1M tokens/day free. Each chunk uses ~2k tokens × 3 perspectives = 6k tokens.
            # CHUNK_DELAY_SECONDS=0 for Ollama/OpenAI; set 1-3 if secondary provider rate-limits.
            if settings.chunk_delay_seconds > 0:
                time.sleep(settings.chunk_delay_seconds)

            if total_processed % 10 == 0:
                logger.info(
                    "Progress: %d chunks | %d ready | %d unresolvable | "
                    "%d records | %d DPO pairs",
                    total_processed, total_ready, total_unresolvable,
                    writer.record_count, dpo_writer.pair_count,
                )

    # ── Phase 3: Cross-document synthesis pass ───────────────────────────────
    if settings.cross_doc_samples > 0:
        logger.info(
            "=== Cross-document synthesis pass (target=%d samples) ===",
            settings.cross_doc_samples,
        )
        with Session() as session:
            cross_written = generate_cross_doc_samples(
                session=session,
                writer=writer,
                dpo_writer=dpo_writer,
                deduplicator=deduplicator,
                batch_id=args.batch_id,
                n_samples=settings.cross_doc_samples,
            )
        logger.info("Cross-doc pass: %d records written", cross_written)
    else:
        cross_written = 0

    # ── Phase 4: Generate datacard ────────────────────────────────────────────
    card = generate_datacard(
        jsonl_path=settings.output_file,
        batch_id=args.batch_id,
        extra_meta={
            "dpo_pairs_count": dpo_writer.pair_count,
            "dpo_file": settings.dpo_output_file,
            "cross_doc_records": cross_written,
            "dedup_index_size": deduplicator.size,
        },
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ESG Data Foundry — Run Complete")
    logger.info("  Batch ID           : %s", args.batch_id)
    logger.info("  Chunks processed   : %d", total_processed)
    logger.info("  Records ready      : %d", total_ready)
    logger.info("  Unresolvable       : %d", total_unresolvable)
    logger.info("  SFT records total  : %d", writer.record_count)
    logger.info("  DPO pairs total    : %d", dpo_writer.pair_count)
    logger.info("  Cross-doc records  : %d", cross_written)
    logger.info("  Watermark hits     : %s", writer.watermark_positions)
    logger.info("  Output (SFT)       : %s", settings.output_file)
    logger.info("  Output (DPO)       : %s", settings.dpo_output_file)
    logger.info("  Datacard           : %s", Path(settings.output_file).with_suffix(".datacard.json"))
    logger.info("  Refusal %%          : %.1f%%", card.get("refusal_pct", 0))
    logger.info("  Multi-turn %%       : %.1f%%", card.get("turns", {}).get("multi_turn_pct", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
