"""
api/routers/scout.py — Gap Scout endpoints.

  POST /api/scout/run              Start a discovery run (async background)
  GET  /api/scout/status/{id}      Run status + topics_found counter
  GET  /api/scout/log/{id}         Log lines (polling fallback)
  WS   /api/scout/ws/{id}          WebSocket live log stream
  GET  /api/scout/topics           All discovered topics sorted by score
  GET  /api/scout/topic/{id}       Single topic with full source list
  POST /api/scout/ingest/{id}      Queue topic sources for pipeline ingestion

  KROK 11 — new endpoints:
  GET  /api/scout/sources          Aggregated health of all 46+ crawlers + WebSub subs
  GET  /api/scout/gaps/models      Topics grouped / ranked by LLM cutoff targets
  POST /api/scout/run/targeted     Scout run constrained to given domains (required)
  GET  /api/scout/trending         Topics sorted by citation_velocity desc
  GET  /api/scout/live             SSE real-time topic/source stream (text/event-stream)
  POST /api/scout/feedback         Submit quality signal on a discovered topic
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import APIRouter, Body, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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

    # Permanently exclude this domain from future Gap Scout runs —
    # the user has selected this topic for work, so it must not be re-discovered.
    domain_text = topic.domains[0] if topic.domains else topic.title
    try:
        from agents.scout_history import exclude_domain_permanently
        await exclude_domain_permanently(
            topic_id=topic_id,
            domain_text=domain_text,
            topic_title=topic.title,
        )
        logger.info("Scout: domain '%s' permanently excluded from future scans.", domain_text[:60])
    except Exception as exc:
        logger.warning("Scout: could not persist domain exclusion: %s", exc)

    return {
        "topic_id": topic_id,
        "title": topic.title,
        "sources_downloaded": len(saved_paths),
        "errors": len(errors),
        "output_dir": str(_DATA_DIR),
        "paths": saved_paths[:5],
        "domain_excluded": domain_text,
        "message": (
            f"Pobrano {len(saved_paths)}/{len(verified)} źródeł do {_DATA_DIR}. "
            f"Temat wykluczony z przyszłych skanów. Pliki gotowe — przejdź do AutoPilota."
        ),
    }


# ===========================================================================
# KROK 11 — new endpoints
# ===========================================================================


# ---------------------------------------------------------------------------
# GET /sources — aggregated health of all crawlers + WebSub subscriptions
# ---------------------------------------------------------------------------


@router.get("/sources")
def get_all_sources() -> dict:
    """
    Return health status for all 46+ source crawlers across layers A–E
    plus live WebSub subscription states.
    """
    from agents.crawlers import (
        get_crawler_status,
        get_crawler_status_b,
        get_crawler_status_c,
        get_crawler_status_d,
        get_crawler_status_e,
    )
    from agents.crawlers.websub import WebSubSubscriber

    crawlers = (
        get_crawler_status()
        + get_crawler_status_b()
        + get_crawler_status_c()
        + get_crawler_status_d()
        + get_crawler_status_e()
    )

    # Update Prometheus gauge
    try:
        from api.monitoring import scout_sources_active
        for c in crawlers:
            is_active = not c.get("is_paused", False) and c.get("consecutive_errors", 0) < 5
            scout_sources_active.labels(source=c.get("source_id", "unknown")).set(1 if is_active else 0)
    except Exception:
        pass

    sub = WebSubSubscriber.instance()
    active_count = sum(
        1 for c in crawlers
        if not c.get("is_paused", False) and c.get("consecutive_errors", 0) < 5
    )
    return {
        "crawlers": crawlers,
        "total_crawlers": len(crawlers),
        "active_crawlers": active_count,
        "websub_subscriptions": sub.status(),
        "websub_stats": sub.stats(),
    }


# ---------------------------------------------------------------------------
# GET /gaps/models — topics ranked per LLM cutoff model
# ---------------------------------------------------------------------------


@router.get("/gaps/models")
def gaps_by_model(
    model: Optional[str] = Query(None, description="Filter to a single model identifier"),
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    """
    Return discovered topics grouped and ranked by their LLM cutoff targets.

    Each topic that targets specific models (e.g. gpt-4o, llama-3) appears
    under each of those model keys. Topics with no cutoff target appear under 'all'.
    """
    all_topics = scouts.latest_topics(limit=500)

    model_map: dict[str, list] = {}
    for topic in all_topics:
        targets: list[str] = topic.cutoff_model_targets or []
        if not targets:
            targets = ["all"]
        for t in targets:
            model_map.setdefault(t, []).append(topic)

    def _ranked(topics: list) -> list[dict]:
        return [
            t.to_dict()
            for t in sorted(topics, key=lambda x: x.knowledge_gap_score, reverse=True)[:limit]
        ]

    if model:
        matching = model_map.get(model, [])
        return {
            "model": model,
            "count": len(matching),
            "topics": _ranked(matching),
        }

    return {
        "models": {m: _ranked(topics) for m, topics in model_map.items()},
        "model_count": len(model_map),
        "total_topics": len(all_topics),
    }


# ---------------------------------------------------------------------------
# POST /run/targeted — targeted domain scan (domains required)
# ---------------------------------------------------------------------------


class TargetedRunRequest(BaseModel):
    domains: List[str] = Field(..., min_length=1, description="Domains to scan (required)")
    max_topics: int = Field(20, ge=1, le=100)
    min_gap_score: float = Field(0.0, ge=0.0, le=1.0)


@router.post("/run/targeted")
async def run_targeted_scout(body: TargetedRunRequest) -> dict:
    """
    Start a Gap Scout run scoped to the given domains.
    Results are filtered by min_gap_score before persisting.
    Returns scout_id immediately; poll /status or open /ws for progress.
    """
    from agents.topic_scout import run_scout as _run_scout

    if not body.domains:
        raise HTTPException(status_code=422, detail="At least one domain is required.")

    scout_id = uuid.uuid4().hex
    scouts.create(scout_id)
    scouts.append_log(scout_id, f"[Scout] Targeted run {scout_id} — domains: {body.domains}")

    async def _background() -> None:
        scouts.update(scout_id, status="running")
        try:
            async def _cb(msg: str) -> None:
                scouts.append_log(scout_id, f"[Scout] {msg}")

            async def _topic_cb(topic) -> None:
                if topic.knowledge_gap_score >= body.min_gap_score:
                    scouts.add_single_topic(scout_id, topic)

            topics = await _run_scout(
                domains=body.domains,
                max_topics=body.max_topics,
                progress_callback=_cb,
                topic_callback=_topic_cb,
            )
            # add any not yet streamed via callback
            eligible = [t for t in topics if t.knowledge_gap_score >= body.min_gap_score]
            scouts.add_topics(scout_id, eligible)
            final_count = scouts.get(scout_id).topics_found if scouts.get(scout_id) else len(eligible)
            scouts.update(scout_id, status="done", topics_found=final_count)
            scouts.append_log(scout_id, f"[Scout] Done — {final_count} topics (gap_score >= {body.min_gap_score})")
        except Exception as exc:
            logger.error("Targeted scout run failed: %s", exc)
            scouts.update(scout_id, status="error", error=str(exc))
            scouts.append_log(scout_id, f"[Scout] Error: {exc}")

    asyncio.create_task(_background())
    return {"scout_id": scout_id, "status": "starting", "domains": body.domains}


# ---------------------------------------------------------------------------
# GET /trending — topics sorted by citation_velocity desc
# ---------------------------------------------------------------------------


@router.get("/trending")
def trending_topics(
    limit: int = Query(20, ge=1, le=100),
    min_velocity: float = Query(0.0, ge=0.0, description="Minimum citation_velocity to include"),
) -> list:
    """Return topics sorted by citation velocity (papers-per-day proxy) descending."""
    all_topics = scouts.latest_topics(limit=500)
    filtered = [t for t in all_topics if t.citation_velocity >= min_velocity]
    filtered.sort(key=lambda t: t.citation_velocity, reverse=True)
    return [t.to_dict() for t in filtered[:limit]]


# ---------------------------------------------------------------------------
# GET /live — SSE real-time topic/source stream
# ---------------------------------------------------------------------------


@router.get("/live")
async def live_stream(request: Request) -> StreamingResponse:
    """
    Server-Sent Events stream of Gap Scout discoveries.

    Events:
      data: {"event": "topic", "data": <ScoutTopic dict>}
      data: {"event": "heartbeat"}

    Replays the 20 most-recent topics on connect, then streams new ones live.
    Disconnect by closing the HTTP connection.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    scouts.register_sse_subscriber(queue)

    # Update SSE subscriber gauge
    try:
        from api.monitoring import scout_sse_subscribers
        scout_sse_subscribers.inc()
    except Exception:
        pass

    async def _generator():
        try:
            # Replay recent topics for new subscribers, then signal end-of-replay
            for topic in scouts.latest_topics(limit=20):
                yield f"data: {json.dumps({'event': 'topic', 'data': topic.to_dict()})}\n\n"
            yield "data: {\"event\": \"replay_end\"}\n\n"

            # Stream live events
            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=25.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield "data: {\"event\": \"heartbeat\"}\n\n"
        finally:
            scouts.unregister_sse_subscriber(queue)
            try:
                from api.monitoring import scout_sse_subscribers
                scout_sse_subscribers.dec()
            except Exception:
                pass

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# POST /feedback — human quality signal on a topic
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    topic_id: str
    rating: int = Field(..., ge=1, le=5, description="Quality score 1 (poor) – 5 (excellent)")
    helpful: bool = Field(True, description="True if topic was useful for training data")
    comment: str = Field("", max_length=500)


@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest) -> dict:
    """
    Record human quality feedback for a discovered topic.
    Feedback is stored in-memory and exposed via Prometheus scout_feedback_total.
    """
    topic = scouts.get_topic(body.topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic not found: {body.topic_id}")

    scouts.add_feedback(
        topic_id=body.topic_id,
        rating=body.rating,
        helpful=body.helpful,
        comment=body.comment,
    )

    # Aggregate feedback for this topic
    feedback_list = scouts.list_feedback(topic_id=body.topic_id)
    avg_rating = sum(f.rating for f in feedback_list) / len(feedback_list)

    return {
        "topic_id": body.topic_id,
        "status": "received",
        "feedback_count": len(feedback_list),
        "avg_rating": round(avg_rating, 2),
    }
