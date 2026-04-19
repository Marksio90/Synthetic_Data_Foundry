"""
api/routers/scout.py — Gap Scout endpoints.

  POST /api/scout/run            Start a discovery run (async background)
  GET  /api/scout/status/{id}    Run status + topics_found counter
  GET  /api/scout/log/{id}       Log lines (polling fallback)
  WS   /api/scout/ws/{id}        WebSocket live log stream
  GET  /api/scout/topics         All discovered topics sorted by score
  GET  /api/scout/topic/{id}     Single topic with full source list
  POST /api/scout/ingest/{id}    Queue topic sources for pipeline ingestion
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Body, HTTPException, WebSocket, WebSocketDisconnect

from api.state import scouts
from config.settings import settings as _settings

_DATA_DIR = Path(_settings.data_dir)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /run — start a scout run
# ---------------------------------------------------------------------------


@router.post("/run")
async def run_scout(
    domains: list[str] | None = Body(None, embed=True),
) -> dict:
    """
    Start a Gap Scout background run.
    If domains is None the AI auto-selects the best domains to scan.
    Returns scout_id immediately; poll /status or open /ws for progress.
    """
    from agents.topic_scout import run_scout as _run_scout

    scout_id = uuid.uuid4().hex
    scouts.create(scout_id)
    scouts.append_log(scout_id, f"[Scout] ID: {scout_id}")
    scouts.append_log(scout_id, "[Scout] Initialising knowledge gap discovery...")

    async def _background() -> None:
        scouts.update(scout_id, status="running")
        try:
            async def _cb(msg: str) -> None:
                scouts.append_log(scout_id, f"[Scout] {msg}")

            async def _topic_cb(topic) -> None:
                scouts.add_single_topic(scout_id, topic)
                rec = scouts.get(scout_id)
                n = rec.topics_found if rec else "?"
                scouts.append_log(scout_id, f"[Scout] 🎯 Topic #{n}: {topic.title[:60]} (score={topic.score:.2f})")

            topics = await _run_scout(
                domains=domains,
                max_topics=50,
                progress_callback=_cb,
                topic_callback=_topic_cb,
            )
            # add_topics is idempotent — catches any not already added via callback
            scouts.add_topics(scout_id, topics)
            final_count = scouts.get(scout_id).topics_found if scouts.get(scout_id) else len(topics)
            scouts.update(scout_id, status="done", topics_found=final_count)
            scouts.append_log(scout_id, f"[Scout] ✅ Done — {final_count} topics found.")
        except Exception as exc:
            logger.error("Scout run failed: %s", exc)
            scouts.update(scout_id, status="error", error=str(exc))
            scouts.append_log(scout_id, f"[Scout] ❌ Error: {exc}")

    asyncio.create_task(_background())
    return {"scout_id": scout_id, "status": "starting"}


# ---------------------------------------------------------------------------
# GET /status/{scout_id}
# ---------------------------------------------------------------------------


@router.get("/status/{scout_id}")
def get_status(scout_id: str) -> dict:
    rec = scouts.get(scout_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Scout run not found: {scout_id}")
    return {
        "scout_id": rec.scout_id,
        "status": rec.status,
        "topics_found": rec.topics_found,
        "elapsed_seconds": rec.elapsed_seconds,
        "error": rec.error,
    }


# ---------------------------------------------------------------------------
# GET /log/{scout_id}
# ---------------------------------------------------------------------------


@router.get("/log/{scout_id}")
def get_log(scout_id: str, offset: int = 0, limit: int = 200) -> dict:
    rec = scouts.get(scout_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Scout run not found: {scout_id}")
    lines = rec.log_lines[offset: offset + limit]
    return {"scout_id": scout_id, "lines": lines, "total_lines": len(rec.log_lines)}


# ---------------------------------------------------------------------------
# WS /ws/{scout_id} — live log stream
# ---------------------------------------------------------------------------


@router.websocket("/ws/{scout_id}")
async def websocket_scout(websocket: WebSocket, scout_id: str) -> None:
    await websocket.accept()
    rec = scouts.get(scout_id)
    if rec is None:
        await websocket.send_json({"error": f"Scout run not found: {scout_id}"})
        await websocket.close()
        return

    sent = 0
    try:
        while True:
            lines = rec.log_lines
            while sent < len(lines):
                await websocket.send_json({
                    "line": lines[sent],
                    "status": rec.status,
                    "topics_found": rec.topics_found,
                })
                sent += 1
            if rec.status in ("done", "error"):
                await websocket.send_json({"status": rec.status, "line": "__EOF__"})
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# GET /topics — all discovered topics
# ---------------------------------------------------------------------------


@router.get("/topics")
def list_topics(limit: int = 50) -> list[dict]:
    return [t.to_dict() for t in scouts.latest_topics(limit=limit)]


# ---------------------------------------------------------------------------
# GET /topic/{topic_id}
# ---------------------------------------------------------------------------


@router.get("/topic/{topic_id}")
def get_topic(topic_id: str) -> dict:
    topic = scouts.get_topic(topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic not found: {topic_id}")
    return topic.to_dict()


# ---------------------------------------------------------------------------
# POST /ingest/{topic_id} — queue for pipeline processing
# ---------------------------------------------------------------------------


_INGEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FoundryBot/1.0; +https://github.com/marksio90)"}


@router.post("/ingest/{topic_id}")
async def ingest_topic(topic_id: str) -> dict:
    """
    Download verified sources of a topic and save them to data/scout_ingested/<topic_id[:8]>/.
    Files are ready for immediate pipeline processing via:
        python main.py --data-dir <output_dir>
    """
    topic = scouts.get_topic(topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic not found: {topic_id}")

    verified = [s for s in topic.sources if s.get("verified", False)]
    if not verified:
        raise HTTPException(status_code=422, detail="No verified sources to ingest.")

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    errors: list[str] = []
    prefix = f"scout_{topic_id[:8]}_"

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=_INGEST_HEADERS) as client:
        for src in verified[:8]:  # cap at 8 downloads per ingest call
            url = src.get("url", "")
            if not url:
                continue
            # arxiv abstract → PDF
            if "arxiv.org/abs/" in url:
                url = url.replace("/abs/", "/pdf/") + ".pdf"
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except Exception as exc:
                errors.append(f"{url[:80]}: {exc}")
                logger.warning("Ingest download failed: %s — %s", url[:80], exc)
                continue

            ct = resp.headers.get("content-type", "")
            ext = ".pdf" if "pdf" in ct else ".html"
            safe_title = re.sub(r"[^\w\-]", "_", (src.get("title") or url)[:50]).strip("_") or "source"
            filename = f"{prefix}{safe_title}{ext}"
            path = _DATA_DIR / filename
            path.write_bytes(resp.content)
            saved_paths.append(filename)
            logger.info("Ingest: saved %s (%d B)", filename, len(resp.content))

    return {
        "topic_id": topic_id,
        "title": topic.title,
        "sources_downloaded": len(saved_paths),
        "errors": len(errors),
        "output_dir": str(_DATA_DIR),
        "paths": saved_paths[:5],
        "message": (
            f"Pobrano {len(saved_paths)}/{len(verified)} źródeł do {_DATA_DIR}. "
            f"Pliki gotowe — przejdź do AutoPilota."
        ),
    }
