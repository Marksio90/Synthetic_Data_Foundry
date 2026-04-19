"""
api/main.py — Foundry Studio FastAPI application.

Services:
  /api/documents/*  — PDF upload, listing, deletion
  /api/pipeline/*   — AutoPilot run, status, log, WebSocket
  /api/samples/*    — Q&A dataset browsing
  /api/training/*   — Hardware inspect, quality gate, SFT/DPO training, export
  /api/chatbot/*    — Chat with Ollama model, evaluation runs
  /api/scout/*      — Gap Scout: automated knowledge-gap discovery
  /health           — liveness probe

Start locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chatbot, documents, pipeline, samples, scout, training

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# APScheduler — hourly Gap Scout
# ---------------------------------------------------------------------------

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore
    _scheduler: AsyncIOScheduler | None = AsyncIOScheduler(timezone="UTC")
except ImportError:
    _scheduler = None
    logger.warning("apscheduler not installed — hourly Gap Scout disabled")


async def _scheduled_scout() -> None:
    """Background job: run Gap Scout and persist results."""
    from agents.topic_scout import run_scout
    from api.state import scouts

    scout_id = uuid.uuid4().hex
    scouts.create(scout_id)
    scouts.append_log(scout_id, "[Scheduler] Hourly Gap Scout started")
    scouts.update(scout_id, status="running")
    try:
        async def _cb(msg: str) -> None:
            scouts.append_log(scout_id, f"[Scout] {msg}")

        topics = await run_scout(progress_callback=_cb)
        scouts.add_topics(scout_id, topics)
        scouts.update(scout_id, status="done", topics_found=len(topics))
        scouts.append_log(scout_id, f"[Scheduler] ✅ Done — {len(topics)} topics")
        logger.info("Scheduled Gap Scout done: %d topics", len(topics))
    except Exception as exc:
        scouts.update(scout_id, status="error", error=str(exc))
        logger.error("Scheduled Gap Scout failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    if _scheduler is not None:
        _scheduler.add_job(
            _scheduled_scout,
            "interval",
            hours=1,
            id="hourly_scout",
            replace_existing=True,
            max_instances=1,
        )
        _scheduler.start()
        logger.info("APScheduler started — hourly Gap Scout enabled")
        # Delay first run so the server finishes startup before the long scout task runs
        async def _delayed_scout() -> None:
            await asyncio.sleep(5.0)
            await _scheduled_scout()
        asyncio.create_task(_delayed_scout())

    yield

    # ── Shutdown ──
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
    from agents.topic_scout import _HTTP as _scout_http
    await _scout_http.aclose()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Foundry Studio API",
    version="1.1.0",
    description="AutoPilot + Gap Scout backend for Synthetic Data Foundry",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(pipeline.router,  prefix="/api/pipeline",  tags=["pipeline"])
app.include_router(samples.router,   prefix="/api/samples",   tags=["samples"])
app.include_router(training.router,  prefix="/api/training",  tags=["training"])
app.include_router(chatbot.router,   prefix="/api/chatbot",   tags=["chatbot"])
app.include_router(scout.router,     prefix="/api/scout",     tags=["scout"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "foundry-api"}
