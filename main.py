"""
main.py — ESG Data Foundry entry point.

Usage:
    python main.py --pdf data/csrd.pdf [--pdf data/sfdr.pdf ...] [--batch-id my-run-001]

Pipeline:
    1. For each PDF: Ingestor → Chunker (stores chunks in PG)
    2. For each pending chunk: run LangGraph swarm
         Simulator → Expert (RAG) → Judge → write/retry/skip
    3. Output: output/dataset_esg_v1.jsonl (ChatML format)

Idempotent: re-running after a crash skips already-processed chunks.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agents.chunker import chunk_document
from agents.ingestor import ingest_pdf
from config.settings import settings
from db.models import Base, DirectiveChunk
from db.repository import get_pending_chunks
from pipeline.graph import build_graph
from pipeline.state import FoundryState
from utils.output import JSONLWriter

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
        required=True,
        help="Path to a directive PDF (can be specified multiple times)",
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
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    engine = get_engine()
    create_tables(engine)
    Session = sessionmaker(bind=engine)

    writer = JSONLWriter(
        output_path=settings.output_file,
        client_id=settings.client_id,
        batch_id=args.batch_id,
        watermark_interval=settings.watermark_interval,
    )

    # ── Phase 1: Ingest & Chunk all PDFs ────────────────────────────────────
    all_pdf_paths = [Path(p) for p in args.pdfs]
    for pdf_path in all_pdf_paths:
        if not pdf_path.exists():
            logger.error("PDF not found: %s — skipping", pdf_path)
            continue

        logger.info("=== Ingesting: %s ===", pdf_path.name)
        with Session() as session:
            source_doc_id, markdown = ingest_pdf(str(pdf_path), session)

            # Determine valid_from_date from the ingested document row
            from sqlalchemy import select
            from db.models import SourceDocument
            doc = session.scalar(
                select(SourceDocument).where(SourceDocument.id == uuid.UUID(source_doc_id))
            )
            valid_from = doc.valid_from_date if doc else None

            # Check if chunks already exist (idempotency)
            from sqlalchemy import func
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

    graph = None  # built lazily per session to inject the session
    total_processed = 0
    total_ready = 0
    total_unresolvable = 0

    with Session() as session:
        graph = build_graph(session, writer)

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

            # Run pipeline once per perspective (CFO / prawnik / audytor)
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
                    "sample_id": None,
                    "batch_id": args.batch_id,
                    "record_index": writer.record_count,
                    # Multi-turn conversation state
                    "conversation_history": [],
                    "turn_count": 0,
                }

                try:
                    final_state = graph.invoke(initial_state)
                    final_status = final_state.get("status", "unknown")
                    if final_status == "ready":
                        chunk_ready = True
                        total_ready += 1
                except Exception as exc:
                    logger.error(
                        "Graph failed for chunk %s [%s]: %s", chunk.id, perspective, exc
                    )

            total_processed += 1
            if not chunk_ready:
                total_unresolvable += 1

            if total_processed % 10 == 0:
                logger.info(
                    "Progress: %d chunks | %d ready | %d unresolvable | %d records written",
                    total_processed, total_ready, total_unresolvable, writer.record_count,
                )

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ESG Data Foundry — Run Complete")
    logger.info("  Batch ID         : %s", args.batch_id)
    logger.info("  Chunks processed : %d", total_processed)
    logger.info("  Records ready    : %d", total_ready)
    logger.info("  Unresolvable     : %d", total_unresolvable)
    logger.info("  Output file      : %s", settings.output_file)
    logger.info("  Records written  : %d", writer.record_count)
    logger.info("  Watermark hits   : %s", writer.watermark_positions)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
