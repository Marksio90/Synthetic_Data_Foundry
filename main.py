"""
main.py — ESG Data Foundry entry point (Gods Finger v2).

Usage:
    python main.py --pdf data/csrd.pdf [--pdf data/sfdr.pdf ...]
    python main.py --data-dir data/ [--batch-id my-run-001]
    python main.py --data-dir data/ --formats pdf,docx,html,mp3

Obsługiwane formaty dokumentów:
    PDF, DOCX, HTML, XML, TXT, MD, MP3, WAV, M4A, MP4 (Replicate Whisper)

Pipeline:
    1. Ingest & Chunk (PDF/DOCX/HTML/audio → chunks w DB)
    2. Auto-kalibracja parametrów pipeline'u
    3. Swarm: 8 perspektyw × każdy chunk
         Simulator → RAG Expert → Constitutional AI → Judge → write/retry/skip
    4. Cross-document synthesis (pary Q&A między dokumentami)
    5. Datacard generation
    6. HuggingFace Hub auto-upload (jeśli skonfigurowany)

Wyjście:
    output/dataset_*.jsonl       — SFT ChatML
    output/dataset_*_dpo.jsonl   — DPO preference pairs
    output/dataset_*_orpo.jsonl  — ORPO preference pairs
    output/dataset_*_kto.jsonl   — KTO labeled pairs
    output/dataset_*.datacard.json — statystyki jakości
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import sessionmaker

from agents.calibrator import AdaptiveCalibrator, calibrate
from agents.chunker import chunk_document
from agents.cross_doc import generate_cross_doc_samples
from agents.hf_uploader import upload_dataset_to_hub
from agents.ingestor import ingest_document
from agents.translator import translate_chunks_in_db
from config.settings import settings
from db.models import Base, DirectiveChunk, SourceDocument
from db.repository import claim_chunk, get_pending_chunks
from utils.datacard import generate_datacard
from utils.dedup import MinHashDeduplicator
from utils.output import DPOWriter, JSONLWriter, KTOWriter, ORPOWriter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("foundry.main")

# Wszystkie obsługiwane rozszerzenia plików
_SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc",
    ".html", ".htm", ".xml",
    ".txt", ".md", ".rst",
    ".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".webm",
}


# ---------------------------------------------------------------------------
# Swarm constants
# ---------------------------------------------------------------------------

_GRAPH_RETRIES = 3
# Cap parallel workers: gpt-4o-mini parallelises well; Ollama is GPU-serial so
# values >4 don't help throughput and just waste memory / connection pool.
_MAX_PERSPECTIVE_WORKERS = 4


# ---------------------------------------------------------------------------
# Per-perspective worker (runs in its own thread + DB session)
# ---------------------------------------------------------------------------

def _run_perspective(
    perspective: str,
    chunk_meta: dict,
    batch_id: str,
    session_factory,
    writer,
    dpo_writer,
    orpo_writer,
    kto_writer,
    deduplicator,
) -> tuple[bool, float]:
    """
    Execute one LangGraph perspective for one chunk in an isolated DB session.
    Returns (is_ready, quality_score).  Thread-safe: writers already use locks.
    """
    from pipeline.graph import build_graph
    from pipeline.state import FoundryState

    with session_factory() as _session:
        _graph = build_graph(
            _session, writer,
            dpo_writer=dpo_writer,
            orpo_writer=orpo_writer,
            kto_writer=kto_writer,
            deduplicator=deduplicator,
        )
        initial_state: FoundryState = {
            "chunk": chunk_meta,
            "perspective": perspective,
            "question": "",
            "is_adversarial": False,
            "retrieved_context": [],
            "retrieved_ids": [],
            "answer": "",
            "constitutional_critique": None,
            "quality_score": 0.0,
            "judge_model": "",
            "judge_reasoning": "",
            "judge_details": {},
            "retry_count": 0,
            "status": "in_progress",
            "error_message": None,
            "conversation_history": [],
            "turn_count": 0,
            "rejected_answer": None,
            "question_type": "factual",
            "difficulty": "medium",
            "sample_id": None,
            "batch_id": batch_id,
            "record_index": writer.record_count,
        }

        for attempt in range(_GRAPH_RETRIES):
            try:
                final = _graph.invoke(initial_state)
                return final.get("status") == "ready", float(final.get("quality_score") or 0.0)
            except openai.RateLimitError:
                if attempt < _GRAPH_RETRIES - 1:
                    wait = 2 ** attempt * 4
                    logger.warning(
                        "Rate-limit (chunk=%s, %s) — retry %d/%d za %ds",
                        chunk_meta["id"], perspective, attempt + 1, _GRAPH_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Graph failed (rate-limit, %d attempts) chunk=%s [%s]",
                        _GRAPH_RETRIES, chunk_meta["id"], perspective,
                    )
            except Exception as exc:
                logger.error("Graph failed chunk=%s [%s]: %s", chunk_meta["id"], perspective, exc)
                break

    return False, 0.0


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
    with engine.connect() as conn:
        for ext in ("vector", "pg_trgm", '"uuid-ossp"'):
            conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {ext};"))

        conn.execute(text(
            "ALTER TABLE directive_chunks ADD COLUMN IF NOT EXISTS fts_vector TSVECTOR;"
        ))

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

        conn.execute(text("DROP TRIGGER IF EXISTS trg_chunk_fts ON directive_chunks;"))
        conn.execute(text("""
            CREATE TRIGGER trg_chunk_fts
                BEFORE INSERT OR UPDATE OF content ON directive_chunks
                FOR EACH ROW EXECUTE FUNCTION update_chunk_fts();
        """))

        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chunks_fts ON directive_chunks USING gin (fts_vector);"
        ))

        try:
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding "
                "ON directive_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
            ))
        except Exception as exc:
            logger.warning("IVFFlat index skipped: %s", exc)
            conn.rollback()
            conn.execute(text("SELECT 1"))

        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chunks_claim "
            "ON directive_chunks (status, created_at) WHERE status IN ('new', 'in_progress');"
        ))

        conn.execute(text("""
            CREATE OR REPLACE FUNCTION claim_chunk_for_processing(p_chunk_id UUID)
            RETURNS BOOLEAN AS $$
            DECLARE rows_updated INTEGER;
            BEGIN
                UPDATE directive_chunks SET status = 'in_progress', updated_at = NOW()
                WHERE id = p_chunk_id AND status = 'new' AND retry_count < 3;
                GET DIAGNOSTICS rows_updated = ROW_COUNT;
                RETURN rows_updated = 1;
            END;
            $$ LANGUAGE plpgsql;
        """))

        conn.execute(text("""
            CREATE OR REPLACE FUNCTION finalize_chunk(
                p_chunk_id UUID, p_success BOOLEAN, p_error TEXT DEFAULT NULL
            )
            RETURNS VOID AS $$
            BEGIN
                IF p_success THEN
                    UPDATE directive_chunks SET status = 'ready', updated_at = NOW()
                    WHERE id = p_chunk_id AND status != 'ready';
                ELSE
                    UPDATE directive_chunks SET
                        retry_count = retry_count + 1,
                        status = CASE WHEN retry_count + 1 >= 3 THEN 'unresolvable' ELSE 'new' END,
                        error_log = p_error, updated_at = NOW()
                    WHERE id = p_chunk_id AND status NOT IN ('ready', 'unresolvable');
                END IF;
            END;
            $$ LANGUAGE plpgsql;
        """))

        for col_ddl in [
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS perspective TEXT",
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS conversation_json JSONB",
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS question_type TEXT",
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS difficulty TEXT",
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS rejected_answer TEXT",
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS human_reviewed BOOLEAN",
            "ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS human_flag TEXT",
        ]:
            conn.execute(text(col_ddl + ";"))

        conn.commit()
    logger.info("DB migrations applied.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESG Data Foundry — Gods Finger v2")
    p.add_argument("--pdf", dest="files", action="append", metavar="PATH", default=[],
                   help="Plik do przetworzenia (PDF/DOCX/HTML/audio). Można podać wielokrotnie.")
    p.add_argument("--data-dir", dest="data_dir", metavar="DIR", default=None,
                   help="Katalog z dokumentami (przetworzy wszystkie obsługiwane formaty)")
    p.add_argument("--formats", dest="formats", default=None,
                   help="Filtry rozszerzeń, np. pdf,docx,mp3 (domyślnie: wszystkie)")
    p.add_argument("--batch-id", default=None,
                   help="Identyfikator batchu (domyślnie: z .env lub losowy UUID)")
    p.add_argument("--chunk-limit", type=int, default=0,
                   help="Max chunków do przetworzenia (0 = bez limitu)")
    p.add_argument("--perspectives", dest="perspectives", default=None,
                   help="Perspektywy do użycia, np. cfo,prawnik,audytor (domyślnie: z .env)")
    p.add_argument("--no-constitutional-ai", action="store_true",
                   help="Wyłącz Constitutional AI (szybszy, niższa jakość)")
    p.add_argument("--upload-hf", action="store_true",
                   help="Wymuś upload do HuggingFace Hub po zakończeniu")
    args = p.parse_args()

    # Zbierz pliki z --data-dir
    if args.data_dir:
        dir_path = Path(args.data_dir)
        if not dir_path.is_dir():
            p.error(f"--data-dir nie istnieje: {args.data_dir}")

        allowed_exts = _SUPPORTED_EXTENSIONS
        if args.formats:
            allowed_exts = {"." + ext.strip().lstrip(".") for ext in args.formats.split(",")}

        found = sorted(f for f in dir_path.rglob("*") if f.suffix.lower() in allowed_exts)
        if not found:
            p.error(f"Brak obsługiwanych plików w: {args.data_dir}")
        args.files.extend(str(f) for f in found)

    if not args.files:
        p.error("Podaj co najmniej --pdf PATH lub --data-dir DIR")

    return args


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Batch ID: CLI → .env → random
    batch_id = args.batch_id or settings.batch_id or f"batch-{uuid.uuid4().hex[:8]}"

    # Perspectives: CLI → .env
    if args.perspectives:
        perspectives = [p.strip() for p in args.perspectives.split(",") if p.strip()]
    else:
        perspectives = list(settings.perspectives)

    # Constitutional AI override
    if args.no_constitutional_ai:
        settings.use_constitutional_ai = False

    logger.info("=" * 60)
    logger.info("ESG Data Foundry — Gods Finger v2")
    logger.info("  Batch ID       : %s", batch_id)
    logger.info("  Perspektywy    : %s", perspectives)
    logger.info("  Constitutional : %s", settings.use_constitutional_ai)
    logger.info("  Pliki          : %d", len(args.files))
    logger.info("=" * 60)

    engine = get_engine()
    create_tables(engine)
    run_migrations(engine)
    Session = sessionmaker(bind=engine)

    # Writers
    writer = JSONLWriter(
        output_path=settings.output_file,
        client_id=settings.client_id,
        batch_id=batch_id,
        watermark_interval=settings.watermark_interval,
    )
    dpo_writer = DPOWriter(output_path=settings.dpo_output_file)
    orpo_writer = ORPOWriter(output_path=settings.orpo_output_file)
    kto_writer = KTOWriter(output_path=settings.kto_output_file)
    deduplicator = MinHashDeduplicator(threshold=settings.dedup_threshold)
    deduplicator.load_from_jsonl(settings.output_file)

    # ── Phase 1: Ingest & Chunk wszystkich plików ────────────────────────────
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            logger.error("Plik nie istnieje: %s — pomijam", path)
            continue

        logger.info("=== Ingesting: %s ===", path.name)
        with Session() as session:
            try:
                source_doc_id, markdown = ingest_document(str(path), session)

                doc = session.scalar(
                    select(SourceDocument).where(SourceDocument.id == uuid.UUID(source_doc_id))
                )
                valid_from = doc.valid_from_date if doc else None

                existing_count = session.scalar(
                    select(func.count(DirectiveChunk.id)).where(
                        DirectiveChunk.source_doc_id == uuid.UUID(source_doc_id)
                    )
                )
                if existing_count and existing_count > 0:
                    logger.info("Chunks już istnieją dla %s (%d) — pomijam chunking", path.name, existing_count)
                else:
                    chunk_ids = chunk_document(
                        session,
                        source_doc_id=source_doc_id,
                        markdown=markdown,
                        valid_from_date=valid_from,
                    )
                    logger.info("Podzielono %s → %d chunków", path.name, len(chunk_ids))

                    if settings.translate_chunks and chunk_ids:
                        logger.info("=== Tłumaczenie %d chunków (%s → pl) ===",
                                    len(chunk_ids), settings.translate_source_lang)
                        translated = translate_chunks_in_db(
                            session, chunk_ids, source_lang=settings.translate_source_lang
                        )
                        session.commit()
                        logger.info("Przetłumaczono %d/%d chunków", translated, len(chunk_ids))
            except Exception as exc:
                session.rollback()
                logger.error("Ingest failed for %s: %s — pomijam plik", path.name, exc)
                continue

    # ── Phase 1.5: Auto-kalibracja ────────────────────────────────────────────
    with Session() as session:
        cal_chunks = get_pending_chunks(session, limit=settings.calibration_samples)
        if cal_chunks:
            logger.info("=== Auto-kalibracja na %d próbkach ===", len(cal_chunks))
            cal = calibrate(cal_chunks)
            logger.info("Kalibracja:\n%s", cal.summary())
            settings.quality_threshold = cal.quality_threshold
            settings.max_turns = cal.max_turns
            settings.adversarial_ratio = cal.adversarial_ratio

    adaptive_cal = AdaptiveCalibrator(settings.quality_threshold)

    # ── Phase 2: Swarm na pending chunks ─────────────────────────────────────
    logger.info(
        "=== Uruchamiam swarm (batch=%s, %d perspektyw, max_workers=%d) ===",
        batch_id, len(perspectives), min(len(perspectives), _MAX_PERSPECTIVE_WORKERS),
    )

    total_processed = total_ready = total_unresolvable = 0

    with Session() as session:
        limit = args.chunk_limit if args.chunk_limit > 0 else 10_000
        pending = get_pending_chunks(session, limit=limit)
        logger.info("Znaleziono %d pending chunków", len(pending))

        for chunk in pending:
            chunk_meta = {
                "id": str(chunk.id),
                "content": chunk.content,
                "content_md": chunk.content_md or chunk.content,
                "source_document": (chunk.source_doc.filename if chunk.source_doc else "unknown"),
                "chunk_index": chunk.chunk_index,
                "section_heading": chunk.section_heading or "",
                "valid_from_date": (chunk.valid_from_date.isoformat() if chunk.valid_from_date else ""),
            }

            claimed = claim_chunk(session, chunk.id)
            try:
                session.commit()
            except Exception:
                session.rollback()
                logger.warning("Chunk %s claim commit failed — skipping", chunk.id)
                continue
            if not claimed:
                logger.debug("Chunk %s already in_progress (recovery mode)", chunk.id)

            chunk_ready = False
            max_workers = min(len(perspectives), _MAX_PERSPECTIVE_WORKERS)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _run_perspective,
                        p, chunk_meta, batch_id,
                        Session, writer, dpo_writer, orpo_writer, kto_writer, deduplicator,
                    ): p
                    for p in perspectives
                }
                for future in as_completed(futures):
                    p = futures[future]
                    try:
                        is_ready, score = future.result()
                        if is_ready:
                            chunk_ready = True
                        if score > 0:
                            adaptive_cal.record(score)
                    except Exception as exc:
                        logger.error("Perspective %s future failed: %s", p, exc)

            # Apply adaptive threshold after each chunk batch
            settings.quality_threshold = adaptive_cal.current_threshold

            total_processed += 1
            if chunk_ready:
                total_ready += 1
            else:
                total_unresolvable += 1

            if settings.chunk_delay_seconds > 0:
                time.sleep(settings.chunk_delay_seconds)

            logger.info(
                "Progress: %d chunks | %d ready | %d unresolvable | "
                "%d SFT | %d DPO | %d ORPO | %d KTO",
                total_processed, total_ready, total_unresolvable,
                writer.record_count, dpo_writer.pair_count,
                orpo_writer.pair_count, kto_writer.sample_count,
            )

    # ── Phase 3: Cross-document synthesis ────────────────────────────────────
    cross_written = 0
    if settings.cross_doc_samples > 0:
        logger.info("=== Cross-document synthesis (target=%d) ===", settings.cross_doc_samples)
        with Session() as session:
            cross_written = generate_cross_doc_samples(
                session=session,
                writer=writer,
                dpo_writer=dpo_writer,
                deduplicator=deduplicator,
                batch_id=batch_id,
                n_samples=settings.cross_doc_samples,
            )
        logger.info("Cross-doc: %d rekordów zapisanych", cross_written)

    # ── Phase 4: Datacard ─────────────────────────────────────────────────────
    card = generate_datacard(
        jsonl_path=settings.output_file,
        batch_id=batch_id,
        extra_meta={
            "dpo_pairs_count": dpo_writer.pair_count,
            "orpo_pairs_count": orpo_writer.pair_count,
            "kto_samples_count": kto_writer.sample_count,
            "cross_doc_records": cross_written,
            "dedup_index_size": deduplicator.size,
            "perspectives_used": perspectives,
            "constitutional_ai": settings.use_constitutional_ai,
        },
    )

    # ── Phase 5: HuggingFace Hub upload ───────────────────────────────────────
    _upload_requested = settings.hf_dataset_repo or args.upload_hf
    if _upload_requested:
        _sft_count = writer.record_count
        _unresolvable_pct = (total_unresolvable / max(total_processed, 1)) * 100
        _quality_ok = _sft_count >= 10 and _unresolvable_pct <= 30.0

        if not _quality_ok:
            logger.warning(
                "⚠ Quality gate: SFT records=%d, unresolvable=%.1f%% — "
                "upload blocked. Użyj --upload-hf aby wymusić.",
                _sft_count, _unresolvable_pct,
            )

        if _quality_ok or args.upload_hf:
            logger.info("=== Uploading dataset to HuggingFace Hub ===")
            upload_dataset_to_hub(
                sft_path=settings.output_file,
                batch_id=batch_id,
                dpo_path=settings.dpo_output_file,
                orpo_path=settings.orpo_output_file,
                kto_path=settings.kto_output_file,
                datacard=card,
            )

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ESG Data Foundry — Gods Finger v2 — Run Complete")
    logger.info("  Batch ID           : %s", batch_id)
    logger.info("  Perspektywy        : %s", perspectives)
    logger.info("  Constitutional AI  : %s", settings.use_constitutional_ai)
    logger.info("  Chunks processed   : %d", total_processed)
    logger.info("  Records ready      : %d", total_ready)
    logger.info("  Unresolvable       : %d", total_unresolvable)
    logger.info("  SFT records        : %d", writer.record_count)
    logger.info("  DPO pairs          : %d", dpo_writer.pair_count)
    logger.info("  ORPO pairs         : %d", orpo_writer.pair_count)
    logger.info("  KTO samples        : %d", kto_writer.sample_count)
    logger.info("  Cross-doc records  : %d", cross_written)
    logger.info("  Watermark hits     : %s", writer.watermark_positions)
    logger.info("  Output (SFT)       : %s", settings.output_file)
    logger.info("  Output (DPO)       : %s", settings.dpo_output_file)
    logger.info("  Output (ORPO)      : %s", settings.orpo_output_file)
    logger.info("  Output (KTO)       : %s", settings.kto_output_file)
    logger.info("  Datacard           : %s", Path(settings.output_file).with_suffix(".datacard.json"))
    logger.info("  Refusal %%          : %.1f%%", card.get("refusal_pct", 0))
    logger.info("  Multi-turn %%       : %.1f%%", card.get("turns", {}).get("multi_turn_pct", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
