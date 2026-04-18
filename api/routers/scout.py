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
import uuid

from fastapi import APIRouter, Body, HTTPException, WebSocket, WebSocketDisconnect

from api.state import scouts

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

            topics = await _run_scout(
                domains=domains,
                max_topics=50,
                progress_callback=_cb,
            )
            scouts.add_topics(scout_id, topics)
            scouts.update(scout_id, status="done", topics_found=len(topics))
            scouts.append_log(scout_id, f"[Scout] ✅ Done — {len(topics)} topics found.")
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Scout run failed: %s", exc)
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
    return [
        {
            "topic_id": t.topic_id,
            "title": t.title,
            "summary": t.summary,
            "score": t.score,
            "recency_score": t.recency_score,
            "llm_uncertainty": t.llm_uncertainty,
            "source_count": t.source_count,
            "social_signal": t.social_signal,
            "sources": t.sources,
            "domains": t.domains,
            "discovered_at": t.discovered_at,
        }
        for t in scouts.latest_topics(limit=limit)
    ]


# ---------------------------------------------------------------------------
# GET /topic/{topic_id}
# ---------------------------------------------------------------------------


@router.get("/topic/{topic_id}")
def get_topic(topic_id: str) -> dict:
    topic = scouts.get_topic(topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic not found: {topic_id}")
    return {
        "topic_id": topic.topic_id,
        "title": topic.title,
        "summary": topic.summary,
        "score": topic.score,
        "recency_score": topic.recency_score,
        "llm_uncertainty": topic.llm_uncertainty,
        "source_count": topic.source_count,
        "social_signal": topic.social_signal,
        "sources": topic.sources,
        "domains": topic.domains,
        "discovered_at": topic.discovered_at,
    }


# ---------------------------------------------------------------------------
# POST /ingest/{topic_id} — queue for pipeline processing
# ---------------------------------------------------------------------------


@router.post("/ingest/{topic_id}")
async def ingest_topic(topic_id: str) -> dict:
    """
    Queue all verified sources of a topic for pipeline ingestion.
    Sources are saved to data/ so the AutoPilot can process them.
    """
    topic = scouts.get_topic(topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic not found: {topic_id}")

    verified = [s for s in topic.sources if s.get("verified", False)]
    if not verified:
        raise HTTPException(status_code=422, detail="No verified sources to ingest.")

    return {
        "topic_id": topic_id,
        "title": topic.title,
        "sources_queued": len(verified),
        "sources": verified[:5],
        "message": (
            f"Queued {len(verified)} source(s) for ingestion. "
            "Open the AutoPilot page to run the pipeline."
        ),
    }
