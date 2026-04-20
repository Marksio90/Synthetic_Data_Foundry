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
import os
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.calibrator import calibrate
from agents.doc_analyzer import analyze_documents
from api.db import get_session
from api.security import require_admin_api_key
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

router = APIRouter(dependencies=[Depends(require_admin_api_key)])

DATA_DIR = Path(settings.data_dir)
_PYTHON = sys.executable  # same interpreter as the API


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
    chunks = session.scalars(
        select(DirectiveChunk)
        .join(SourceDocument, DirectiveChunk.source_doc_id == SourceDocument.id)
        .where(SourceDocument.filename.in_(filenames))
        .order_by(DirectiveChunk.created_at)
        .limit(settings.calibration_samples)
    ).all()
    return list(chunks)


# ---------------------------------------------------------------------------
# POST /api/pipeline/analyze — fast preview, no pipeline started
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=AnalysisResponse)
def analyze(
    filenames: list[str] = Body(embed=True),
    session: Session = Depends(get_session),
) -> AnalysisResponse:
    """
    Run DocAnalyzer + Calibrator on the given files and return the auto-detected
    parameters — without starting the pipeline.
    Used by the UI to show the AutoPilot preview before the user clicks Start.
    """
    paths = _resolve_paths(filenames)
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

@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(
    req: PipelineRunRequest,
    session: Session = Depends(get_session),
) -> PipelineRunResponse:
    """
    Start a full AutoPilot pipeline run as a background subprocess.
    Returns run_id immediately; poll /status or open /ws for progress.
    """
    paths = _resolve_paths(req.filenames)

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
    runs.append_log(run_id, "[AutoPilot] Uruchamiam pipeline...")

    # Launch background task
    asyncio.create_task(_run_subprocess(run_id, cmd, env))

    return PipelineRunResponse(
        run_id=run_id,
        batch_id=batch_id,
        status="starting",
        message=f"Pipeline started. Domain: {analysis.domain_label}. "
                f"Threshold (auto): {quality_threshold}",
    )


async def _run_subprocess(run_id: str, cmd: list[str], env: dict) -> None:
    """Background task: run main.py, stream its output into the run log."""
    runs.update(run_id, status="running")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(Path(__file__).parent.parent.parent),  # /app/api/routers → /app
        )
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


# ---------------------------------------------------------------------------
# GET /api/pipeline/status/{run_id}
# ---------------------------------------------------------------------------

@router.get("/status/{run_id}", response_model=PipelineStatusResponse)
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

@router.get("/log/{run_id}", response_model=LogLinesResponse)
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

@router.get("/runs")
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
# WebSocket /api/pipeline/ws/{run_id} — live log stream
# ---------------------------------------------------------------------------

@router.websocket("/ws/{run_id}")
async def websocket_log(websocket: WebSocket, run_id: str) -> None:
    """
    Streams new log lines to the client as they arrive.
    Sends one JSON message per line: {"line": "...", "status": "running"}
    Closes when the run is done or errors.
    """
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
