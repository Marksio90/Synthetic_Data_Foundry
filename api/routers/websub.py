"""
api/routers/websub.py — WebSub/PubSubHubbub callback endpoints (Tier 1 real-time)

  GET  /api/scout/webhook  — Hub intent verification (hub.challenge response)
  POST /api/scout/webhook  — Content delivery (HMAC-verified, parsed, injected)
  GET  /api/scout/websub/status  — Active subscriptions and stats

The GET endpoint is called by WebSub hubs after we send a subscribe request.
The POST endpoint receives real-time content pushes; verified sources are injected
directly into the live Gap Scout state so SSE/WebSocket clients see them <10s after
the hub's content push.

See agents/crawlers/websub.py for the subscriber implementation.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /webhook — hub intent verification (hub sends hub.challenge)
# ---------------------------------------------------------------------------


@router.get("/webhook", response_class=PlainTextResponse)
async def websub_verify(
    hub_mode:          str           = Query(...,  alias="hub.mode"),
    hub_topic:         str           = Query(...,  alias="hub.topic"),
    hub_challenge:     str           = Query(...,  alias="hub.challenge"),
    hub_lease_seconds: Optional[int] = Query(None, alias="hub.lease_seconds"),
) -> PlainTextResponse:
    """
    Respond to a WebSub hub subscription verification request.
    Returns hub.challenge as plain text → hub completes subscription.
    Returns 404 if we deny the subscription.
    """
    from agents.crawlers.websub import WebSubSubscriber
    sub = WebSubSubscriber.instance()

    challenge_response = sub.verify_intent(
        mode=hub_mode,
        topic=hub_topic,
        challenge=hub_challenge,
        lease_seconds=hub_lease_seconds,
    )
    if challenge_response is None:
        raise HTTPException(status_code=404, detail="Subscription denied")

    return PlainTextResponse(content=challenge_response, status_code=200)


# ---------------------------------------------------------------------------
# POST /webhook — hub content delivery
# ---------------------------------------------------------------------------


@router.post("/webhook")
async def websub_delivery(
    request: Request,
    x_hub_signature:     Optional[str] = Header(None, alias="X-Hub-Signature"),
    x_hub_signature_256: Optional[str] = Header(None, alias="X-Hub-Signature-256"),
) -> Response:
    """
    Receive a real-time content delivery from a WebSub hub.

    1. Verify HMAC-SHA256 signature (X-Hub-Signature-256 takes priority)
    2. Parse Atom/RSS payload → ScoutSource list
    3. Run verification firewall (5-point, max_concurrent=4 for low latency)
    4. Inject verified sources into Gap Scout state (latest active run)
    5. Return 200 immediately so hub does not retry
    """
    from agents.crawlers.websub import WebSubSubscriber
    from agents.topic_scout import _HTTP as _scout_http

    body = await request.body()

    # Resolve topic from Link header (spec) or X-Hub-Topic (non-standard)
    link_header = request.headers.get("link", "")
    topic = _parse_link_topic(link_header)
    if not topic:
        topic = request.headers.get("x-hub-topic", "")
    if not topic:
        # Some hubs put topic in query param
        topic = request.query_params.get("hub.topic", "")

    sig = x_hub_signature_256 or x_hub_signature or ""

    sub = WebSubSubscriber.instance()
    sources = await sub.handle_delivery(topic=topic, body=body, sig_header=sig)

    if not sources:
        # Return 200 even on empty delivery — don't trigger hub retry
        return Response(status_code=200)

    # 5-point verification firewall (concurrency capped at 4 for <5s latency)
    from agents.crawlers.verifier import verify_batch
    batch = await verify_batch(sources, _scout_http, max_concurrent=4)
    verified = batch.verified

    if not verified:
        logger.debug(
            "[websub] delivery: all %d sources failed verification firewall", len(sources)
        )
        return Response(status_code=200)

    # Inject into live Gap Scout state
    _inject_realtime_sources(verified, topic)

    logger.info(
        "[websub] delivery injected: topic=%s verified=%d/%d",
        topic[:80], len(verified), len(sources),
    )
    return Response(status_code=200)


# ---------------------------------------------------------------------------
# GET /websub/status — subscription observability
# ---------------------------------------------------------------------------


@router.get("/websub/status")
def websub_status() -> dict:
    """Return active WebSub subscription states and aggregate stats."""
    from agents.crawlers.websub import WebSubSubscriber
    sub = WebSubSubscriber.instance()
    return {
        "stats":         sub.stats(),
        "subscriptions": sub.status(),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_link_topic(link_header: str) -> str:
    """
    Extract the self-topic URL from a Link header.
    Example: Link: <https://…>; rel="self", <https://hub>; rel="hub"
    Returns "" when not found.
    """
    if not link_header:
        return ""
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="self"' in part or "rel=self" in part:
            # Extract URL between < >
            start = part.find("<")
            end   = part.find(">")
            if start != -1 and end != -1:
                return part[start + 1: end].strip()
    return ""


def _inject_realtime_sources(
    sources: list,
    topic: str,
) -> None:
    """
    Append real-time verified sources to the most-recent active scout run.
    If no run is active, create a lightweight 'realtime' run to hold them.
    """
    try:
        from api.state import scouts
        from agents.topic_scout import ScoutTopicData, _topic_id
        from datetime import datetime, timezone

        # Find the most recent active or done scout run
        rec = scouts.latest_run()
        if rec is None:
            # No existing run — create a realtime-only run
            import uuid
            rt_id = f"realtime_{uuid.uuid4().hex[:8]}"
            scouts.create(rt_id)
            scouts.update(rt_id, status="realtime")
            run_id = rt_id
        else:
            run_id = rec.scout_id

        # Find or create a topic for this websub feed's domain
        topic_key = f"websub:{topic[:80]}"
        t_id = _topic_id(topic_key)
        existing = scouts.get_topic(t_id)

        if existing is None:
            # Create a stub topic to hold the new sources
            best = max(
                (s for s in sources if s.published_at),
                key=lambda s: s.published_at,
                default=sources[0],
            )
            stub = ScoutTopicData(
                topic_id=t_id,
                title=f"[Live] {sources[0].source_type}: {best.title[:100]}",
                summary=best.title[:250],
                score=0.0,
                recency_score=1.0,
                llm_uncertainty=0.5,
                source_count=len(sources),
                social_signal=0.0,
                sources=sources[:10],
                domains=[topic[:100]],
                discovered_at=datetime.now(timezone.utc).isoformat(),
                knowledge_gap_score=0.0,
                ingest_ready=True,
            )
            scouts.add_topics(run_id, [stub])
        else:
            # Merge sources into existing topic
            existing_urls = {s.get("url", "") for s in existing.sources}
            new_srcs = [s for s in sources if s.url not in existing_urls]
            if new_srcs:
                existing.sources.extend(
                    [{"url": s.url, "title": s.title, "published_at": s.published_at,
                      "source_type": s.source_type, "verified": s.verified,
                      "source_tier": s.source_tier}
                     for s in new_srcs]
                )
                existing.source_count = len(existing.sources)

    except Exception as exc:
        logger.debug("[websub] _inject_realtime_sources error: %s", exc)
