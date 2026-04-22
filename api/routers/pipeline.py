"""
api/routers/pipeline.py — Pipeline orchestration endpoints.

Endpoints:
  POST /api/pipeline/analyze        DocAnalyzer + Calibrator preview (fast, no pipeline)
  POST /api/pipeline/run            Start AutoPilot full run (async subprocess)
  GET  /api/pipeline/status/{id}    Run status + progress counters
  GET  /api/pipeline/log/{id}       Last N log lines (polling fallback)
  WS   /api/pipeline/ws/{id}        WebSocket: live log stream

AutoPilot flow:
  1. DocAnalyzer  → language, domain, perspectives
  2. Calibrator   → quality_threshold, max_turns, adversarial_ratio
  3. Subprocess   → python main.py --pdf ... (with calibrated env vars)
  4. WS/polling   → stream subprocess stdout to UI
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from agents.calibrator import calibrate
from agents.doc_analyzer import analyze_documents
from api.db import get_session
from api.errors import FailedDependencyError, ServiceUnavailableError
from api.security import (
    create_ws_ticket,
    require_admin_api_key,
    require_admin_api_key_ws,
    verify_ws_ticket,
)
from api.schemas import (
    AnalysisResponse,
    CalibrationInfo,
    LogLinesResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    PipelineStatusResponse,
)
from api.state import runs
from config.settings import settings
from db.models import DirectiveChunk, SourceDocument

router = APIRouter()
logger = logging.getLogger("foundry.api.pipeline")

DATA_DIR = Path(settings.data_dir)
_PYTHON = sys.executable  # same interpreter as the API
_RUN_PROCS: dict[str, asyncio.subprocess.Process] = {}
_EXECUTOR_LOCK = asyncio.Lock()
_EXECUTOR_WORKERS: list[asyncio.Task] = []
_QUEUE: asyncio.Queue["PipelineJob"] = asyncio.Queue(maxsize=settings.pipeline_queue_maxsize)
_QUEUED_RUN_IDS: set[str] = set()
_CANCELLED_QUEUED_RUN_IDS: set[str] = set()
_EXECUTOR_STARTED = False
_REDIS_CLIENT = None
_REDIS_QUEUE_KEY = "foundry:pipeline:jobs"
_REDIS_CANCELLED_KEY = "foundry:pipeline:cancelled"
_REQUIRED_TABLES = ("source_documents", "directive_chunks")


# ---------------------------------------------------------------------------
# Queue-native executor
# ---------------------------------------------------------------------------


@dataclass
class PipelineJob:
    run_id: str
    cmd: list[str]
    env: dict[str, str]
    cwd: str


def _queue_backend() -> str:
    return (settings.pipeline_queue_backend or "memory").strip().lower()


async def _ensure_redis_client():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    if _queue_backend() != "redis":
        return None
    try:
        from redis.asyncio import Redis  # type: ignore
    except Exception:
        logger.warning("Redis backend requested but 'redis' package is unavailable. Falling back to memory queue.")
        return None
    try:
        client = Redis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        _REDIS_CLIENT = client
        logger.info("Pipeline queue backend: redis (%s).", settings.redis_url)
    except Exception as exc:
        logger.warning("Redis queue unavailable (%s). Falling back to memory queue.", exc)
        _REDIS_CLIENT = None
    return _REDIS_CLIENT


def _job_to_json(job: PipelineJob) -> str:
    return json.dumps({"run_id": job.run_id, "cmd": job.cmd, "env": job.env, "cwd": job.cwd}, ensure_ascii=False)


def _job_from_json(raw: str) -> PipelineJob:
    data = json.loads(raw)
    return PipelineJob(run_id=data["run_id"], cmd=list(data["cmd"]), env=dict(data["env"]), cwd=data["cwd"])


async def _queue_size() -> int:
    client = await _ensure_redis_client()
    if client is not None and _queue_backend() == "redis":
        return int(await client.llen(_REDIS_QUEUE_KEY))
    return _QUEUE.qsize()


async def _enqueue_job(job: PipelineJob) -> None:
    client = await _ensure_redis_client()
    maxsize = int(settings.pipeline_queue_maxsize)
    if client is not None and _queue_backend() == "redis":
        size = int(await client.llen(_REDIS_QUEUE_KEY))
        if size >= maxsize:
            raise asyncio.QueueFull
        await client.rpush(_REDIS_QUEUE_KEY, _job_to_json(job))
        return
    _QUEUE.put_nowait(job)


async def _dequeue_job() -> PipelineJob:
    client = await _ensure_redis_client()
    if client is not None and _queue_backend() == "redis":
        while True:
            res = await client.blpop(_REDIS_QUEUE_KEY, timeout=2)
            if res is None:
                await asyncio.sleep(0.1)
                continue
            _, raw = res
            return _job_from_json(raw)
    return await _QUEUE.get()


async def _mark_cancelled_queued(run_id: str) -> None:
    client = await _ensure_redis_client()
    if client is not None and _queue_backend() == "redis":
        await client.sadd(_REDIS_CANCELLED_KEY, run_id)
    _CANCELLED_QUEUED_RUN_IDS.add(run_id)
    _QUEUED_RUN_IDS.discard(run_id)


async def _is_cancelled_queued(run_id: str) -> bool:
    client = await _ensure_redis_client()
    if client is not None and _queue_backend() == "redis":
        if await client.sismember(_REDIS_CANCELLED_KEY, run_id):
            return True
    return run_id in _CANCELLED_QUEUED_RUN_IDS


async def _clear_cancelled_queued(run_id: str) -> None:
    client = await _ensure_redis_client()
    if client is not None and _queue_backend() == "redis":
        await client.srem(_REDIS_CANCELLED_KEY, run_id)
    _CANCELLED_QUEUED_RUN_IDS.discard(run_id)


async def _ensure_executor_started() -> None:
    global _EXECUTOR_STARTED
    if _EXECUTOR_WORKERS and _EXECUTOR_STARTED:
        return
    async with _EXECUTOR_LOCK:
        if _EXECUTOR_WORKERS and _EXECUTOR_STARTED:
            return
        worker_count = max(1, int(settings.pipeline_worker_concurrency))
        _EXECUTOR_STARTED = True
        for i in range(worker_count):
            task = asyncio.create_task(_queue_worker(i), name=f"pipeline_queue_worker_{i}")
            _EXECUTOR_WORKERS.append(task)


async def _queue_worker(worker_idx: int) -> None:
    try:
        while True:
            job = await _dequeue_job()
            try:
                _QUEUED_RUN_IDS.discard(job.run_id)
                if await _is_cancelled_queued(job.run_id):
                    await _clear_cancelled_queued(job.run_id)
                    runs.append_log(job.run_id, "[Queue] Zadanie usunięte z kolejki (anulowane przed startem).")
                    continue

                runs.append_log(job.run_id, f"[Queue] Worker {worker_idx} uruchamia zadanie...")
                await _run_subprocess(job.run_id, job.cmd, job.env, job.cwd)
            finally:
                if _queue_backend() == "memory":
                    _QUEUE.task_done()
    except asyncio.CancelledError:
        return


async def startup_executor() -> None:
    """Start pipeline queue workers at app startup."""
    await _ensure_executor_started()


async def shutdown_executor(timeout_seconds: float = 10.0) -> None:
    """Stop queue workers gracefully during app shutdown."""
    global _EXECUTOR_STARTED
    _EXECUTOR_STARTED = False
    if not _EXECUTOR_WORKERS:
        return
    if _queue_backend() == "memory":
        try:
            await asyncio.wait_for(_QUEUE.join(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            # Continue with cancellation even if queue isn't drained.
            pass
    for task in _EXECUTOR_WORKERS:
        task.cancel()
    await asyncio.gather(*_EXECUTOR_WORKERS, return_exceptions=True)
    _EXECUTOR_WORKERS.clear()
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        await _REDIS_CLIENT.aclose()
        _REDIS_CLIENT = None


# ---------------------------------------------------------------------------
# Helper: resolve full paths for filenames
# ---------------------------------------------------------------------------

def _resolve_paths(filenames: list[str]) -> list[Path]:
    paths = []
    base = DATA_DIR.resolve()
    for fn in filenames:
        if not fn or Path(fn).name != fn:
            raise HTTPException(status_code=400, detail=f"Invalid filename: {fn}")
        p = (DATA_DIR / fn).resolve()
        if base not in p.parents:
            raise HTTPException(status_code=400, detail=f"Invalid path: {fn}")
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File not found in data/: {fn}")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Helper: pull calibration chunks from DB
# ---------------------------------------------------------------------------

def _get_calibration_chunks(session: Session, filenames: list[str]) -> list:
    """
    Return up to calibration_samples chunks for auto-calibration.
    Uses existing DB chunks if the documents have already been ingested.
    """
    try:
        chunks = session.scalars(
            select(DirectiveChunk)
            .join(SourceDocument, DirectiveChunk.source_doc_id == SourceDocument.id)
            .where(SourceDocument.filename.in_(filenames))
            .order_by(DirectiveChunk.created_at)
            .limit(settings.calibration_samples)
        ).all()
    except SQLAlchemyError as exc:
        raise ServiceUnavailableError(
            error_code="pipeline_db_query_failed",
            message="Pipeline database query failed.",
            details={
                "hint": "Verify DATABASE_URL connectivity and PostgreSQL schema initialization. "
                        "Run DB bootstrap/migrations (e.g. init/01_schema.sql) and retry.",
            },
        ) from exc
    return list(chunks)


def _ensure_pipeline_db_ready(session: Session) -> None:
    """
    Preflight: verify DB connectivity + required pipeline tables exist.
    Raises typed API errors (503/424) with actionable details.
    """
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise ServiceUnavailableError(
            error_code="pipeline_db_unavailable",
            message="Pipeline database is unavailable.",
            details={"hint": "Check DATABASE_URL, network reachability, and DB credentials."},
        ) from exc

    bind = session.get_bind()
    if bind is None:
        raise ServiceUnavailableError(
            error_code="pipeline_db_unavailable",
            message="Pipeline database bind is unavailable.",
            details={
                "hint": "Verify SQLAlchemy session/engine initialization.",
            },
        )

    try:
        db_inspector = inspect(bind)
        missing_tables = [t for t in _REQUIRED_TABLES if not db_inspector.has_table(t)]
    except SQLAlchemyError as exc:
        raise ServiceUnavailableError(
            error_code="pipeline_db_inspection_failed",
            message="Unable to inspect pipeline database schema.",
            details={"hint": "Verify DB permissions and schema availability."},
        ) from exc

    if missing_tables:
        raise FailedDependencyError(
            error_code="pipeline_schema_missing",
            message="Pipeline schema is not initialized.",
            details={
                "missing_tables": missing_tables,
                "hint": "Apply SQL bootstrap/migrations (init/01_schema.sql) and retry.",
            },
        )


# ---------------------------------------------------------------------------
# POST /api/pipeline/analyze — fast preview, no pipeline started
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=AnalysisResponse, dependencies=[Depends(require_admin_api_key)])
def analyze(
    filenames: list[str] = Body(embed=True),
    session: Session = Depends(get_session),
) -> AnalysisResponse:
    """
    Run DocAnalyzer + Calibrator on the given files and return the auto-detected
    parameters — without starting the pipeline.
    Used by the UI to show the AutoPilot preview before the user clicks Start.
    """
    _ensure_pipeline_db_ready(session)
    paths = _resolve_paths(filenames)
    _ensure_pipeline_db_ready(session)
    analysis = analyze_documents([str(p) for p in paths])
    chunks = _get_calibration_chunks(session, filenames)
    calib = calibrate(chunks)

    return AnalysisResponse(
        language=analysis.language,
        translation_required=analysis.translation_required,
        domain=analysis.domain,
        domain_label=analysis.domain_label,
        perspectives=analysis.perspectives,
        domain_confidence=analysis.domain_confidence,
        total_chars=analysis.total_chars,
        auto_decisions=analysis.auto_decisions,
        calibration=CalibrationInfo(
            quality_threshold=calib.quality_threshold,
            max_turns=calib.max_turns,
            adversarial_ratio=calib.adversarial_ratio,
            n_chunks=calib.n_chunks,
            avg_chunk_len=calib.avg_chunk_len,
            vocab_richness=calib.vocab_richness,
            info_density=calib.info_density,
            reasoning=calib.reasoning,
        ),
    )


# ---------------------------------------------------------------------------
# POST /api/pipeline/run — start AutoPilot
# ---------------------------------------------------------------------------

@router.post("/run", response_model=PipelineRunResponse, dependencies=[Depends(require_admin_api_key)])
async def run_pipeline(
    req: PipelineRunRequest,
    session: Session = Depends(get_session),
) -> PipelineRunResponse:
    """
    Start a full AutoPilot pipeline run as a background subprocess.
    Returns run_id immediately; poll /status or open /ws for progress.
    """
    _ensure_pipeline_db_ready(session)
    paths = _resolve_paths(req.filenames)
    _ensure_pipeline_db_ready(session)

    # Auto-generate batch_id if not provided
    batch_id = req.batch_id or (
        f"auto-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    )
    run_id = uuid.uuid4().hex

    # DocAnalyzer (fast, no LLM)
    analysis = analyze_documents([str(p) for p in paths])

    # Calibrator (fast, heuristic)
    chunks = _get_calibration_chunks(session, req.filenames)
    calib = calibrate(chunks)

    # Apply manual overrides if provided
    quality_threshold = req.quality_threshold or calib.quality_threshold
    max_turns = req.max_turns or calib.max_turns
    adversarial_ratio = req.adversarial_ratio or calib.adversarial_ratio

    # Create run record
    rec = runs.create(run_id, batch_id)
    rec.analysis = {
        "language": analysis.language,
        "translation_required": analysis.translation_required,
        "domain": analysis.domain,
        "domain_label": analysis.domain_label,
        "perspectives": analysis.perspectives,
        "domain_confidence": analysis.domain_confidence,
        "auto_decisions": analysis.auto_decisions,
    }
    rec.calibration = {
        "quality_threshold": quality_threshold,
        "max_turns": max_turns,
        "adversarial_ratio": adversarial_ratio,
        "n_chunks": calib.n_chunks,
        "avg_chunk_len": calib.avg_chunk_len,
        "vocab_richness": calib.vocab_richness,
        "info_density": calib.info_density,
        "reasoning": calib.reasoning,
    }

    # Build subprocess environment — calibrated values override .env
    env = os.environ.copy()
    env.update({
        "QUALITY_THRESHOLD": str(quality_threshold),
        "MAX_TURNS": str(max_turns),
        "ADVERSARIAL_RATIO": str(adversarial_ratio),
        "BATCH_ID": batch_id,
        "CHUNK_LIMIT": str(req.chunk_limit),
    })

    # Build main.py command
    _APP_ROOT = Path(__file__).parent.parent.parent  # /app/api/routers → /app
    cmd = [_PYTHON, str(_APP_ROOT / "main.py")]
    for p in paths:
        cmd += ["--pdf", str(p)]
    cmd += ["--batch-id", batch_id, "--chunk-limit", str(req.chunk_limit)]

    runs.append_log(run_id, f"[AutoPilot] Run ID: {run_id}")
    runs.append_log(run_id, f"[AutoPilot] Batch ID: {batch_id}")
    runs.append_log(run_id, f"[AutoPilot] Domena: {analysis.domain_label}")
    runs.append_log(run_id, f"[AutoPilot] Język: {analysis.language.upper()}"
                   + (" → tłumaczenie wymagane" if analysis.translation_required else " ✓"))
    runs.append_log(run_id, f"[AutoPilot] Perspektywy: {', '.join(analysis.perspectives)}")
    runs.append_log(run_id, f"[AutoPilot] quality_threshold={quality_threshold} "
                   f"max_turns={max_turns} adversarial={adversarial_ratio}")
    runs.append_log(run_id, "[AutoPilot] Dodaję zadanie do kolejki wykonawczej...")
    runs.update(run_id, status="queued")

    await _ensure_executor_started()
    try:
        await _enqueue_job(
            PipelineJob(
                run_id=run_id,
                cmd=cmd,
                env=env,
                cwd=str(Path(__file__).parent.parent.parent),  # /app/api/routers → /app
            )
        )
        _QUEUED_RUN_IDS.add(run_id)
        runs.append_log(run_id, f"[Queue] Pozycja w kolejce: {await _queue_size()}.")
    except asyncio.QueueFull as exc:
        runs.update(run_id, status="error", error="Pipeline queue is full.")
        runs.append_log(run_id, "[Queue] ❌ Kolejka pipeline jest pełna.")
        raise ServiceUnavailableError(
            error_code="pipeline_queue_full",
            message="Pipeline queue is full.",
            details={"hint": "Retry later or increase PIPELINE_QUEUE_MAXSIZE / worker concurrency."},
        ) from exc

    return PipelineRunResponse(
        run_id=run_id,
        batch_id=batch_id,
        status="queued",
        message=f"Pipeline queued. Domain: {analysis.domain_label}. "
                f"Threshold (auto): {quality_threshold}",
    )


@router.get("/queue", dependencies=[Depends(require_admin_api_key)])
async def queue_stats() -> dict[str, int | str]:
    """Queue visibility endpoint for operators."""
    queued = await _queue_size()
    return {
        "backend": _queue_backend(),
        "queued_jobs": queued,
        "queued_run_ids": len(_QUEUED_RUN_IDS),
        "cancelled_queued_runs": len(_CANCELLED_QUEUED_RUN_IDS),
        "active_workers": len(_EXECUTOR_WORKERS),
        "running_processes": len(_RUN_PROCS),
    }


async def _run_subprocess(run_id: str, cmd: list[str], env: dict[str, str], cwd: str) -> None:
    """Background task: run main.py, stream its output into the run log."""
    runs.update(run_id, status="running")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=cwd,
        )
        _RUN_PROCS[run_id] = proc
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            runs.append_log(run_id, line)

        await proc.wait()
        if proc.returncode == 0:
            runs.update(run_id, status="done", progress_pct=100)
            runs.append_log(run_id, "[AutoPilot] ✅ Pipeline zakończony pomyślnie.")
        else:
            runs.update(
                run_id,
                status="error",
                error=f"Subprocess exited with code {proc.returncode}",
            )
            runs.append_log(run_id, f"[AutoPilot] ❌ Błąd (exit code {proc.returncode})")
    except Exception as exc:
        runs.update(run_id, status="error", error=str(exc))
        runs.append_log(run_id, f"[AutoPilot] ❌ Wyjątek: {exc}")
    finally:
        _RUN_PROCS.pop(run_id, None)


@router.post("/cancel/{run_id}", dependencies=[Depends(require_admin_api_key)])
async def cancel_run(run_id: str) -> dict[str, str]:
    """Cancel a running pipeline subprocess."""
    rec = runs.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    if run_id in _QUEUED_RUN_IDS or rec.status == "queued":
        await _mark_cancelled_queued(run_id)
        runs.update(
            run_id,
            status="cancelled",
            error="Cancelled by user before execution",
            ended_at=datetime.datetime.now().timestamp(),
        )
        runs.append_log(run_id, "[AutoPilot] ⏹️ Zadanie anulowane w kolejce przed uruchomieniem.")
        return {"run_id": run_id, "status": "cancelled"}

    proc = _RUN_PROCS.get(run_id)
    if proc is None or rec.status not in ("starting", "running"):
        raise HTTPException(status_code=409, detail=f"Run is not cancellable: {run_id}")

    runs.append_log(run_id, "[AutoPilot] ⏹️ Anulowanie uruchomione przez użytkownika...")
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()

    runs.update(run_id, status="error", error="Cancelled by user", ended_at=datetime.datetime.now().timestamp())
    runs.append_log(run_id, "[AutoPilot] ⏹️ Pipeline anulowany przez użytkownika.")
    return {"run_id": run_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# GET /api/pipeline/status/{run_id}
# ---------------------------------------------------------------------------

@router.get("/status/{run_id}", response_model=PipelineStatusResponse, dependencies=[Depends(require_admin_api_key)])
def get_status(run_id: str) -> PipelineStatusResponse:
    rec = runs.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return PipelineStatusResponse(
        run_id=rec.run_id,
        batch_id=rec.batch_id,
        status=rec.status,
        progress_pct=rec.progress_pct,
        chunks_total=rec.chunks_total,
        chunks_done=rec.chunks_done,
        records_written=rec.records_written,
        dpo_pairs=rec.dpo_pairs,
        elapsed_seconds=rec.elapsed_seconds,
        analysis=rec.analysis,
        calibration=rec.calibration,
        error=rec.error,
    )


# ---------------------------------------------------------------------------
# GET /api/pipeline/log/{run_id} — REST polling fallback
# ---------------------------------------------------------------------------

@router.get("/log/{run_id}", response_model=LogLinesResponse, dependencies=[Depends(require_admin_api_key)])
def get_log(run_id: str, offset: int = 0, limit: int = 200) -> LogLinesResponse:
    """Return log lines from offset to offset+limit. Use for polling."""
    rec = runs.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    lines = rec.log_lines[offset: offset + limit]
    return LogLinesResponse(run_id=run_id, lines=lines, total_lines=len(rec.log_lines))


# ---------------------------------------------------------------------------
# GET /api/pipeline/runs — list all runs (for UI history)
# ---------------------------------------------------------------------------

@router.get("/runs", dependencies=[Depends(require_admin_api_key)])
def list_runs() -> list[dict]:
    return [
        {
            "run_id": r.run_id,
            "batch_id": r.batch_id,
            "status": r.status,
            "progress_pct": r.progress_pct,
            "chunks_done": r.chunks_done,
            "chunks_total": r.chunks_total,
            "records_written": r.records_written,
            "dpo_pairs": r.dpo_pairs,
            "elapsed_seconds": r.elapsed_seconds,
        }
        for r in runs.list_runs()
    ]


# ---------------------------------------------------------------------------
# GET /api/pipeline/ws-ticket/{run_id} — short-lived WS auth ticket
# ---------------------------------------------------------------------------

@router.get("/ws-ticket/{run_id}", dependencies=[Depends(require_admin_api_key)])
def create_ws_auth_ticket(run_id: str) -> dict[str, str]:
    rec = runs.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    ticket = create_ws_ticket(run_id)
    if not ticket:
        raise HTTPException(status_code=503, detail="ADMIN_API_KEY is not configured on the server.")
    return {"run_id": run_id, "ws_ticket": ticket}


# ---------------------------------------------------------------------------
# WebSocket /api/pipeline/ws/{run_id} — live log stream
# ---------------------------------------------------------------------------

@router.websocket("/ws/{run_id}")
async def websocket_log(websocket: WebSocket, run_id: str) -> None:
    """
    Streams new log lines to the client as they arrive.
    Sends one JSON message per line: {"line": "...", "status": "running"}
    Closes when the run is done or errors.
    """
    ws_ticket = websocket.query_params.get("ws_ticket")
    if not (require_admin_api_key_ws(websocket) or verify_ws_ticket(ws_ticket, run_id)):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    rec = runs.get(run_id)
    if rec is None:
        await websocket.send_json({"error": f"Run not found: {run_id}"})
        await websocket.close()
        return

    sent = 0  # index of last sent line
    try:
        while True:
            lines = rec.log_lines
            while sent < len(lines):
                await websocket.send_json({
                    "line": lines[sent],
                    "status": rec.status,
                    "progress_pct": rec.progress_pct,
                    "records_written": rec.records_written,
                    "chunks_done": rec.chunks_done,
                    "chunks_total": rec.chunks_total,
                })
                sent += 1

            if rec.status in ("done", "error"):
                # Flush any remaining lines then close
                await websocket.send_json({"status": rec.status, "line": "__EOF__"})
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
