"""
orchestration/temporal_workflows.py — Temporal.io durable workflow definitions.

Replaces the monolithic main.py subprocess pattern with crash-safe Temporal
workflows. Each document ingestion becomes one Workflow; each pipeline step
is a durable Activity with automatic retry and state checkpointing.

Architecture:
  FoundryWorkflow (per document)
    ├── ingest_activity        — parse document → chunks
    ├── calibrate_activity     — estimate quality_threshold, adversarial_ratio
    ├── process_chunk_activity — LangGraph per chunk (1 activity per chunk)
    ├── cross_doc_activity     — cross-document Q&A synthesis
    └── finalize_activity      — quality gate → datacard → HF upload

Usage:
    from orchestration.temporal_workflows import start_document_workflow
    handle = await start_document_workflow(file_path="/data/csrd.pdf", batch_id="batch_001")

Deployment:
    # Start worker
    python -m orchestration.worker

    # Temporal server (docker-compose)
    image: temporalio/auto-setup:1.24
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Optional

logger = logging.getLogger("foundry.orchestration")

# ---------------------------------------------------------------------------
# Dataclasses for typed workflow input/output
# ---------------------------------------------------------------------------

@dataclass
class DocumentWorkflowInput:
    file_path: str
    batch_id: str
    perspectives: list[str] = field(default_factory=lambda: [
        "cfo", "prawnik", "audytor", "analityk",
        "regulator", "akademik", "dziennikarz", "inwestor",
    ])
    quality_threshold: float = 0.70
    max_turns: int = 3
    adversarial_ratio: float = 0.10
    upload_to_hub: bool = False
    client_id: str = "default"


@dataclass
class DocumentWorkflowResult:
    run_id: str
    doc_id: str
    chunks_processed: int
    samples_generated: int
    dpo_pairs: int
    quality_score_avg: float
    cost_cents: float
    dataset_path: str
    uploaded_to_hub: bool
    error: Optional[str] = None


@dataclass
class ChunkActivityInput:
    chunk_id: str
    doc_id: str
    content: str
    perspective: str
    batch_id: str
    quality_threshold: float
    max_turns: int
    adversarial_ratio: float
    client_id: str


@dataclass
class ChunkActivityResult:
    chunk_id: str
    perspective: str
    samples_written: int
    dpo_pairs: int
    quality_score: float
    cost_cents: float
    status: str  # ready | unresolvable | error


# ---------------------------------------------------------------------------
# Temporal Activity definitions
# Each activity is a plain async function decorated with @activity.defn
# ---------------------------------------------------------------------------

try:
    from temporalio import activity, workflow  # type: ignore
    from temporalio.client import Client  # type: ignore
    from temporalio.worker import Worker  # type: ignore
    _TEMPORAL_AVAILABLE = True
except ImportError:
    _TEMPORAL_AVAILABLE = False
    # Stub decorators so the module is importable without temporalio installed
    class _Stub:
        @staticmethod
        def defn(func=None, **_kw):
            return func if func else (lambda f: f)
        @staticmethod
        def run(func):
            return func

    class activity:  # type: ignore[no-redef]
        defn = _Stub.defn

    class workflow:  # type: ignore[no-redef]
        defn = _Stub.defn
        run = _Stub.run

    Client = None  # type: ignore[assignment,misc]
    Worker = None  # type: ignore[assignment,misc]


@activity.defn(name="ingest_document_activity")
async def ingest_document_activity(file_path: str, batch_id: str) -> dict[str, Any]:
    """Parse document and store chunks in PostgreSQL. Returns doc_id + chunk_ids."""
    import asyncio
    from db import repository as repo
    from agents.ingestor import ingest_document
    from agents.chunker import chunk_document

    loop = asyncio.get_event_loop()

    def _run_sync() -> dict[str, Any]:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from config.settings import settings

        engine = create_engine(settings.database_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            doc_id, markdown = ingest_document(file_path, session)
            chunk_ids = chunk_document(doc_id, markdown, session)
            session.commit()
            return {"doc_id": doc_id, "chunk_ids": chunk_ids, "markdown_len": len(markdown)}
        finally:
            session.close()
            engine.dispose()

    result = await loop.run_in_executor(None, _run_sync)
    logger.info("ingest_activity: doc=%s chunks=%d", result["doc_id"], len(result["chunk_ids"]))
    return result


@activity.defn(name="calibrate_activity")
async def calibrate_activity(doc_id: str) -> dict[str, Any]:
    """Auto-calibrate quality_threshold and adversarial_ratio from sample analysis."""
    import asyncio

    def _run_sync() -> dict[str, Any]:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from config.settings import settings
        from agents.calibrator import Calibrator

        engine = create_engine(settings.database_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            cal = Calibrator(session)
            result = cal.run(doc_id=doc_id)
            session.commit()
            return result
        finally:
            session.close()
            engine.dispose()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_sync)


@activity.defn(name="process_chunk_activity")
async def process_chunk_activity(inp: dict[str, Any]) -> dict[str, Any]:
    """
    Run full LangGraph pipeline for one chunk × one perspective.
    This is the hot path — runs up to (chunks × perspectives) times per document.
    """
    import asyncio

    chunk_input = ChunkActivityInput(**inp)

    def _run_sync() -> dict[str, Any]:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from config.settings import settings
        from pipeline.graph import build_graph
        from utils.output import JSONLWriter, DPOWriter
        import os

        engine = create_engine(settings.database_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        sft_path = f"/tmp/foundry/{chunk_input.batch_id}_sft.jsonl"
        dpo_path = f"/tmp/foundry/{chunk_input.batch_id}_dpo.jsonl"
        os.makedirs("/tmp/foundry", exist_ok=True)

        writer = JSONLWriter(sft_path, batch_id=chunk_input.batch_id, client_id=chunk_input.client_id)
        dpo_writer = DPOWriter(dpo_path, batch_id=chunk_input.batch_id, client_id=chunk_input.client_id)

        try:
            graph = build_graph(session, writer, dpo_writer=dpo_writer)
            initial_state = {
                "chunk": {
                    "id": chunk_input.chunk_id,
                    "content": chunk_input.content,
                    "source_document": chunk_input.doc_id,
                },
                "perspective": chunk_input.perspective,
                "batch_id": chunk_input.batch_id,
                "quality_threshold": chunk_input.quality_threshold,
                "turn_count": 0,
                "retry_count": 0,
                "conversation_history": [],
            }
            final_state = graph.invoke(initial_state)
            return {
                "chunk_id": chunk_input.chunk_id,
                "perspective": chunk_input.perspective,
                "samples_written": 1 if final_state.get("record_index", -1) >= 0 else 0,
                "dpo_pairs": 1 if final_state.get("rejected_answer") else 0,
                "quality_score": float(final_state.get("quality_score", 0.0)),
                "cost_cents": 0.0,
                "status": final_state.get("status", "ready"),
            }
        finally:
            writer.flush()
            dpo_writer.flush()
            session.close()
            engine.dispose()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_sync)


@activity.defn(name="cross_doc_activity")
async def cross_doc_activity(doc_id: str, batch_id: str) -> dict[str, Any]:
    """Generate cross-document synthesis Q&A pairs."""
    import asyncio

    def _run_sync() -> dict[str, Any]:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from config.settings import settings
        from agents.cross_doc import generate_cross_doc_samples

        engine = create_engine(settings.database_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            count = generate_cross_doc_samples(doc_id, batch_id, session)
            session.commit()
            return {"cross_doc_samples": count}
        finally:
            session.close()
            engine.dispose()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_sync)


@activity.defn(name="finalize_activity")
async def finalize_activity(batch_id: str, upload_to_hub: bool) -> dict[str, Any]:
    """Run quality gate, generate datacard, optionally upload to HuggingFace Hub."""
    import asyncio

    def _run_sync() -> dict[str, Any]:
        from training.quality_gate import run_quality_gate
        from training.datacard import generate_datacard
        result = {"quality_gate_passed": False, "uploaded": False, "hub_url": ""}
        gate_result = run_quality_gate(batch_id=batch_id)
        result["quality_gate_passed"] = gate_result.get("passed", False)
        result["quality_stats"] = gate_result
        generate_datacard(batch_id=batch_id)
        if upload_to_hub and result["quality_gate_passed"]:
            from agents.hf_uploader import upload_dataset
            hub_url = upload_dataset(batch_id=batch_id)
            result["uploaded"] = True
            result["hub_url"] = hub_url
        return result

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_sync)


# ---------------------------------------------------------------------------
# Workflow definition
# ---------------------------------------------------------------------------

@workflow.defn(name="FoundryDocumentWorkflow")
class FoundryDocumentWorkflow:
    """
    Durable workflow for one document → dataset pipeline.

    Steps:
      1. Ingest + chunk document
      2. Calibrate pipeline parameters
      3. Process all chunks × all perspectives (parallel activities)
      4. Cross-document synthesis
      5. Quality gate + optional HF upload
    """

    @workflow.run
    async def run(self, inp: dict[str, Any]) -> dict[str, Any]:
        doc_input = DocumentWorkflowInput(**inp)

        # --- Step 1: Ingest ---
        ingest_result = await workflow.execute_activity(  # type: ignore[attr-defined]
            "ingest_document_activity",
            args=[doc_input.file_path, doc_input.batch_id],
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=_default_retry(),
        )
        doc_id: str = ingest_result["doc_id"]
        chunk_ids: list[str] = ingest_result["chunk_ids"]

        if not chunk_ids:
            return {"error": "No chunks extracted from document", "doc_id": doc_id}

        # --- Step 2: Calibrate ---
        try:
            cal_result = await workflow.execute_activity(  # type: ignore[attr-defined]
                "calibrate_activity",
                args=[doc_id],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=_default_retry(),
            )
            quality_threshold = cal_result.get("quality_threshold", doc_input.quality_threshold)
            adversarial_ratio = cal_result.get("adversarial_ratio", doc_input.adversarial_ratio)
        except Exception:
            quality_threshold = doc_input.quality_threshold
            adversarial_ratio = doc_input.adversarial_ratio

        # --- Step 3: Process chunks × perspectives (parallel) ---
        chunk_tasks = []
        for chunk_id in chunk_ids:
            for perspective in doc_input.perspectives:
                chunk_inp = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "content": "",   # worker fetches from DB by chunk_id
                    "perspective": perspective,
                    "batch_id": doc_input.batch_id,
                    "quality_threshold": quality_threshold,
                    "max_turns": doc_input.max_turns,
                    "adversarial_ratio": adversarial_ratio,
                    "client_id": doc_input.client_id,
                }
                chunk_tasks.append(
                    workflow.execute_activity(  # type: ignore[attr-defined]
                        "process_chunk_activity",
                        args=[chunk_inp],
                        start_to_close_timeout=timedelta(minutes=10),
                        retry_policy=_default_retry(max_attempts=2),
                    )
                )

        # Run all chunk×perspective tasks concurrently (Temporal handles scheduling)
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

        total_samples = 0
        total_dpo = 0
        total_cost = 0.0
        quality_scores = []
        for r in chunk_results:
            if isinstance(r, Exception):
                continue
            total_samples += r.get("samples_written", 0)
            total_dpo += r.get("dpo_pairs", 0)
            total_cost += r.get("cost_cents", 0.0)
            if r.get("quality_score", 0) > 0:
                quality_scores.append(r["quality_score"])

        # --- Step 4: Cross-document synthesis ---
        try:
            cross_result = await workflow.execute_activity(  # type: ignore[attr-defined]
                "cross_doc_activity",
                args=[doc_id, doc_input.batch_id],
                start_to_close_timeout=timedelta(minutes=15),
                retry_policy=_default_retry(),
            )
            total_samples += cross_result.get("cross_doc_samples", 0)
        except Exception as exc:
            logger.warning("cross_doc_activity failed (non-fatal): %s", exc)

        # --- Step 5: Finalize ---
        final_result = await workflow.execute_activity(  # type: ignore[attr-defined]
            "finalize_activity",
            args=[doc_input.batch_id, doc_input.upload_to_hub],
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=_default_retry(),
        )

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        return {
            "run_id": doc_input.batch_id,
            "doc_id": doc_id,
            "chunks_processed": len(chunk_ids),
            "samples_generated": total_samples,
            "dpo_pairs": total_dpo,
            "quality_score_avg": round(avg_quality, 4),
            "cost_cents": round(total_cost, 4),
            "dataset_path": f"/tmp/foundry/{doc_input.batch_id}_sft.jsonl",
            "uploaded_to_hub": final_result.get("uploaded", False),
            "hub_url": final_result.get("hub_url", ""),
            "quality_gate_passed": final_result.get("quality_gate_passed", False),
        }


def _default_retry(max_attempts: int = 3):
    """Return Temporal RetryPolicy if available, else None."""
    if not _TEMPORAL_AVAILABLE:
        return None
    from temporalio.common import RetryPolicy  # type: ignore
    return RetryPolicy(
        initial_interval=timedelta(seconds=2),
        maximum_interval=timedelta(seconds=30),
        backoff_coefficient=2.0,
        maximum_attempts=max_attempts,
    )


# ---------------------------------------------------------------------------
# Helper: start a workflow from the API layer
# ---------------------------------------------------------------------------

async def start_document_workflow(
    file_path: str,
    batch_id: str,
    temporal_url: str = "localhost:7233",
    **kwargs: Any,
) -> str:
    """
    Kick off a FoundryDocumentWorkflow on the Temporal server.
    Returns the workflow run ID.

    Falls back to direct main.py execution if Temporal is unavailable.
    """
    if not _TEMPORAL_AVAILABLE:
        logger.warning("temporalio not installed — falling back to direct execution.")
        return await _fallback_direct_run(file_path, batch_id, **kwargs)

    try:
        client = await Client.connect(temporal_url)
        inp = {"file_path": file_path, "batch_id": batch_id, **kwargs}
        handle = await client.start_workflow(
            FoundryDocumentWorkflow.run,
            inp,
            id=f"foundry-{batch_id}",
            task_queue="foundry-tasks",
        )
        logger.info("Temporal workflow started: id=%s run_id=%s", handle.id, handle.result_run_id)
        return handle.result_run_id or handle.id
    except Exception as exc:
        logger.error("Failed to start Temporal workflow (%s) — falling back to direct run.", exc)
        return await _fallback_direct_run(file_path, batch_id, **kwargs)


async def _fallback_direct_run(file_path: str, batch_id: str, **_: Any) -> str:
    """Direct asyncio fallback when Temporal server is not reachable."""
    import subprocess, sys
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "main.py", "--file", file_path, "--batch-id", batch_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    logger.info("Fallback direct run PID=%d for batch=%s", proc.pid, batch_id)
    return f"direct-{proc.pid}"
