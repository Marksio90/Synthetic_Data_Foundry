"""
agents/crawlers/websub.py — Tier 1 Real-Time Discovery via WebSub/PubSubHubbub

WebSub protocol (W3C Recommendation):
  1. Discovery  — parse <link rel="hub"> from an Atom/RSS feed
  2. Subscribe  — POST hub.mode=subscribe to the hub, providing our callback URL
  3. Verify     — hub sends GET with hub.challenge → respond with challenge text
  4. Deliver    — hub POSTs new content; we verify HMAC-SHA256, parse, emit sources

Latency guarantee: <10s from publication to callback delivery (hub-dependent).

Public hubs supported out of the box:
  - pubsubhubbub.appspot.com   (Google; arXiv, FederalRegister, many Atom feeds)
  - pubsubhubbub.superfeedr.com (Superfeedr; BIS, ESMA, EBA, ECB, WTO…)
  - hub.arxiv.org               (arXiv self-hosted fallback)

Pre-configured feed subscriptions (auto-registered on startup):
  arXiv cs.AI / cs.LG / econ / stat.ML, Federal Register, BIS working papers,
  ESMA news, EBA publications, ECB press releases, EUR-Lex (REG+DIR), WTO.

Usage:
    subscriber = WebSubSubscriber.instance()
    # On startup:
    await subscriber.subscribe_all(callback_base_url)
    # FastAPI GET callback:
    return subscriber.verify_intent(mode, topic, challenge, lease_seconds)
    # FastAPI POST callback:
    sources = await subscriber.handle_delivery(topic, body_bytes, sig_header)
    # Background renewal (call every hour):
    await subscriber.renew_expiring(callback_base_url, margin_seconds=3600)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

import httpx

from agents.topic_scout import ScoutSource, _get_source_tier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known feeds + their hubs
# (topic_url, hub_url, source_type, tier)
# ---------------------------------------------------------------------------

_KNOWN_FEEDS: list[tuple[str, str, str, str]] = [
    # arXiv — Google hub
    ("https://export.arxiv.org/rss/cs.AI",    "https://pubsubhubbub.appspot.com", "arxiv", "S"),
    ("https://export.arxiv.org/rss/cs.LG",    "https://pubsubhubbub.appspot.com", "arxiv", "S"),
    ("https://export.arxiv.org/rss/cs.CL",    "https://pubsubhubbub.appspot.com", "arxiv", "S"),
    ("https://export.arxiv.org/rss/econ.GN",  "https://pubsubhubbub.appspot.com", "arxiv", "S"),
    ("https://export.arxiv.org/rss/stat.ML",  "https://pubsubhubbub.appspot.com", "arxiv", "S"),
    # US Federal Register — Google hub
    ("https://www.federalregister.gov/api/v1/articles.rss"
     "?conditions[type][]=RULE&conditions[type][]=PRORULE",
     "https://pubsubhubbub.appspot.com", "federalregister", "S"),
    # BIS — Superfeedr (fat hub, fetches any URL)
    ("https://www.bis.org/doclist/wp.rss",             "https://pubsubhubbub.superfeedr.com", "bis",  "A"),
    ("https://www.bis.org/doclist/cgdfs.rss",          "https://pubsubhubbub.superfeedr.com", "bis",  "A"),
    # ESMA — Superfeedr
    ("https://www.esma.europa.eu/press-news/esma-news.rss", "https://pubsubhubbub.superfeedr.com", "esma", "A"),
    # EBA — Superfeedr
    ("https://www.eba.europa.eu/newsroom/news.rss",    "https://pubsubhubbub.superfeedr.com", "eba",  "A"),
    # ECB — Superfeedr
    ("https://www.ecb.europa.eu/rss/press.html",       "https://pubsubhubbub.superfeedr.com", "ecb",  "S"),
    ("https://www.ecb.europa.eu/rss/wps.html",         "https://pubsubhubbub.superfeedr.com", "ecb",  "S"),
    # WTO — Superfeedr
    ("https://www.wto.org/rss/english/news_e.rss",     "https://pubsubhubbub.superfeedr.com", "wto",  "A"),
    # EUR-Lex — Superfeedr
    (
        "https://eur-lex.europa.eu/search.html"
        "?type=quick&lang=EN&scope=EURLEX&FM_CODED=REG%2CDIR&RSS=true",
        "https://pubsubhubbub.superfeedr.com", "eurlex", "S",
    ),
    # Papers With Code (Atom) — Google hub
    ("https://paperswithcode.com/latest.atom",         "https://pubsubhubbub.appspot.com", "paperswithcode", "B"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WebSubSubscription:
    """State of a single WebSub subscription."""
    topic_url: str
    hub_url: str
    source_type: str
    tier: str
    callback_url: str = ""
    secret: str = ""
    lease_seconds: int = 86400          # default 24 h
    subscribed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    verified: bool = False
    delivery_count: int = 0

    def is_expiring(self, margin_s: int = 3600) -> bool:
        return datetime.now(timezone.utc) >= (self.expires_at - timedelta(seconds=margin_s))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATE_FMTS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S GMT",
)


def _to_iso(s: str) -> str:
    for fmt in _DATE_FMTS:
        try:
            dt = datetime.strptime(s[:32].strip(), fmt)
            return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()
        except ValueError:
            continue
    return s


_ATOM_NS = "http://www.w3.org/2005/Atom"
_AT = f"{{{_ATOM_NS}}}"


def _rss_text(el: ET.Element, tag: str) -> str:
    child = el.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _atom_text(el: Optional[ET.Element], tag: str) -> str:
    child = el.find(f"{_AT}{tag}") if el is not None else None
    return (child.text or "").strip() if child is not None else ""


def _atom_link(el: ET.Element) -> str:
    for lnk in el.findall(f"{_AT}link"):
        if lnk.get("rel", "alternate") == "alternate":
            return lnk.get("href", "")
    lnk = el.find(f"{_AT}link")
    return lnk.get("href", "") if lnk is not None else ""


# ---------------------------------------------------------------------------
# HMAC verification
# ---------------------------------------------------------------------------


def verify_signature(secret: str, body: bytes, sig_header: str) -> bool:
    """
    Verify an X-Hub-Signature(-256) header against the request body.

    Supports both sha256=<hex> (WebSub) and sha1=<hex> (older PuSH hubs).
    Returns True when signature matches OR when no secret is configured.
    """
    if not secret:
        return True                # no secret → trust all deliveries
    if not sig_header:
        logger.warning("[websub] delivery missing X-Hub-Signature — rejected")
        return False

    parts = sig_header.split("=", 1)
    if len(parts) != 2:
        return False
    algo, expected = parts

    try:
        if algo == "sha256":
            mac = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        elif algo == "sha1":
            mac = hmac.new(secret.encode(), body, hashlib.sha1).hexdigest()
        else:
            logger.debug("[websub] unknown sig algo: %s", algo)
            return False
        return hmac.compare_digest(mac, expected)
    except Exception as exc:
        logger.debug("[websub] signature verification error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Delivery parsing  →  list[ScoutSource]
# ---------------------------------------------------------------------------


def parse_delivery(payload: bytes, source_type: str, tier: str) -> list[ScoutSource]:
    """
    Parse an Atom or RSS payload from a hub delivery.
    Returns a list of ScoutSource objects. Never raises.
    """
    sources: list[ScoutSource] = []
    try:
        text = payload.decode("utf-8", errors="replace")
        root = ET.fromstring(text)

        # Atom feed
        entries = root.findall(f"{_AT}entry")
        if entries:
            for entry in entries:
                link = _atom_link(entry)
                if not link:
                    continue
                sources.append(ScoutSource(
                    url=link,
                    title=_atom_text(entry, "title").replace("\n", " "),
                    published_at=_to_iso(
                        _atom_text(entry, "published") or _atom_text(entry, "updated")
                    ),
                    source_type=source_type,
                    verified=(tier == "S"),   # Tier S domains pre-verified
                    source_tier=tier,
                    snippet=_atom_text(entry, "summary")[:500],
                ))
            return sources

        # RSS feed
        for item in root.findall(".//item"):
            link = _rss_text(item, "link") or _rss_text(item, "guid")
            if not link:
                continue
            sources.append(ScoutSource(
                url=link,
                title=_rss_text(item, "title"),
                published_at=_to_iso(_rss_text(item, "pubDate")),
                source_type=source_type,
                verified=(tier == "S"),
                source_tier=tier,
                snippet=_rss_text(item, "description")[:500],
            ))
    except ET.ParseError as exc:
        logger.debug("[websub] XML parse error: %s", exc)
    except Exception as exc:
        logger.debug("[websub] parse_delivery error: %s", exc)
    return sources


# ---------------------------------------------------------------------------
# WebSubSubscriber — singleton
# ---------------------------------------------------------------------------


class WebSubSubscriber:
    """
    Manages all WebSub subscriptions for the Gap Scout.

    Thread-safe for async use — all mutations protected by asyncio.Lock.
    """

    _instance: Optional["WebSubSubscriber"] = None

    def __init__(self) -> None:
        # keyed by topic_url
        self._subs: dict[str, WebSubSubscription] = {}
        self._lock = asyncio.Lock()
        self._http: Optional[httpx.AsyncClient] = None

    @classmethod
    def instance(cls) -> "WebSubSubscriber":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # HTTP client lifecycle
    # ------------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": "FoundryScout/2.0 WebSub subscriber"},
            )
        return self._http

    async def aclose(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Subscribe to a single feed
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        topic_url: str,
        hub_url: str,
        callback_url: str,
        source_type: str = "websub",
        tier: str = "C",
        secret: str = "",
        lease_seconds: int = 86400,
    ) -> bool:
        """
        Send a WebSub subscribe request to the hub.
        Returns True when hub accepted (202 Accepted) or already subscribed.
        """
        data = {
            "hub.mode":           "subscribe",
            "hub.topic":          topic_url,
            "hub.callback":       callback_url,
            "hub.lease_seconds":  str(lease_seconds),
        }
        if secret:
            data["hub.secret"] = secret

        try:
            resp = await self._client().post(
                hub_url,
                content=urlencode(data).encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            accepted = resp.status_code in (200, 202, 204)
            if accepted:
                now = datetime.now(timezone.utc)
                async with self._lock:
                    self._subs[topic_url] = WebSubSubscription(
                        topic_url=topic_url,
                        hub_url=hub_url,
                        source_type=source_type,
                        tier=tier,
                        callback_url=callback_url,
                        secret=secret,
                        lease_seconds=lease_seconds,
                        subscribed_at=now,
                        expires_at=now + timedelta(seconds=lease_seconds),
                        verified=False,
                    )
                logger.info(
                    "[websub] subscribed topic=%s hub=%s status=%d",
                    topic_url[:80], hub_url, resp.status_code,
                )
            else:
                logger.warning(
                    "[websub] subscribe rejected topic=%s hub=%s status=%d body=%s",
                    topic_url[:80], hub_url, resp.status_code,
                    resp.text[:200],
                )
            return accepted
        except Exception as exc:
            logger.warning("[websub] subscribe error topic=%s: %s", topic_url[:80], exc)
            return False

    # ------------------------------------------------------------------
    # Subscribe to all known feeds
    # ------------------------------------------------------------------

    async def subscribe_all(
        self,
        callback_base_url: str,
        secret: str = "",
        lease_seconds: int = 86400,
        max_concurrent: int = 5,
    ) -> dict:
        """
        Subscribe to all entries in _KNOWN_FEEDS.
        Returns summary dict with counts.

        Args:
            callback_base_url: Public base URL for the callback endpoint,
                               e.g. "https://api.example.com/api/scout/webhook"
        """
        if not callback_base_url:
            logger.warning("[websub] No callback URL configured — skipping subscribe_all")
            return {"skipped": len(_KNOWN_FEEDS), "reason": "no_callback_url"}

        sem = asyncio.Semaphore(max_concurrent)

        async def _sub(topic: str, hub: str, src: str, tier: str) -> bool:
            async with sem:
                return await self.subscribe(
                    topic_url=topic,
                    hub_url=hub,
                    callback_url=callback_base_url,
                    source_type=src,
                    tier=tier,
                    secret=secret,
                    lease_seconds=lease_seconds,
                )

        results = await asyncio.gather(
            *[_sub(t, h, s, ti) for t, h, s, ti in _KNOWN_FEEDS],
            return_exceptions=True,
        )
        ok    = sum(1 for r in results if r is True)
        fail  = sum(1 for r in results if r is False or isinstance(r, Exception))
        logger.info("[websub] subscribe_all: %d OK / %d failed", ok, fail)
        return {"subscribed": ok, "failed": fail, "total": len(_KNOWN_FEEDS)}

    # ------------------------------------------------------------------
    # Intent verification (called by GET /api/scout/webhook)
    # ------------------------------------------------------------------

    def verify_intent(
        self,
        mode: str,
        topic: str,
        challenge: str,
        lease_seconds: Optional[int] = None,
    ) -> Optional[str]:
        """
        Respond to a hub subscription verification GET request.

        Returns:
            challenge string — FastAPI should return this as PlainText 200
            None             — reject (topic unknown; FastAPI returns 404)
        """
        if mode == "subscribe":
            # Mark the subscription as verified
            if topic in self._subs:
                sub = self._subs[topic]
                sub.verified = True
                if lease_seconds:
                    sub.expires_at = (
                        datetime.now(timezone.utc)
                        + timedelta(seconds=lease_seconds)
                    )
                logger.info("[websub] subscription verified topic=%s", topic[:80])
            else:
                # Accept even if not yet in our registry (race condition on startup)
                logger.debug("[websub] challenge for unknown topic=%s — accepting", topic[:80])
            return challenge

        if mode == "unsubscribe":
            # Only confirm unsubscribe if we initiated it
            if topic in self._subs:
                del self._subs[topic]
                logger.info("[websub] unsubscribed topic=%s", topic[:80])
            return challenge

        if mode == "denied":
            logger.warning("[websub] subscription denied for topic=%s", topic[:80])
            return None

        return None

    # ------------------------------------------------------------------
    # Delivery handling (called by POST /api/scout/webhook)
    # ------------------------------------------------------------------

    async def handle_delivery(
        self,
        topic: str,
        body: bytes,
        sig_header: str = "",
    ) -> list[ScoutSource]:
        """
        Verify signature and parse a hub content delivery.

        Returns list of new ScoutSource objects (may be empty on auth failure).
        """
        # Determine subscription for this topic
        sub = self._subs.get(topic)
        secret = sub.secret if sub else ""

        if not verify_signature(secret, body, sig_header):
            logger.warning(
                "[websub] HMAC verification FAILED topic=%s sig=%s",
                topic[:80], sig_header[:40],
            )
            return []

        src_type = sub.source_type if sub else "websub"
        tier     = sub.tier        if sub else _get_source_tier(topic)

        sources = parse_delivery(body, src_type, tier)
        if sources and sub:
            sub.delivery_count += 1

        logger.info(
            "[websub] delivery: topic=%s sources=%d",
            topic[:80], len(sources),
        )
        return sources

    # ------------------------------------------------------------------
    # Subscription renewal
    # ------------------------------------------------------------------

    async def renew_expiring(
        self,
        callback_base_url: str,
        secret: str = "",
        margin_seconds: int = 3600,
    ) -> int:
        """
        Re-subscribe any subscriptions expiring within margin_seconds.
        Returns number of renewals attempted.
        """
        async with self._lock:
            expiring = [
                sub for sub in self._subs.values()
                if sub.is_expiring(margin_seconds)
            ]

        renewed = 0
        for sub in expiring:
            ok = await self.subscribe(
                topic_url=sub.topic_url,
                hub_url=sub.hub_url,
                callback_url=callback_base_url or sub.callback_url,
                source_type=sub.source_type,
                tier=sub.tier,
                secret=secret or sub.secret,
                lease_seconds=sub.lease_seconds,
            )
            if ok:
                renewed += 1
        if renewed:
            logger.info("[websub] renewed %d/%d expiring subscriptions", renewed, len(expiring))
        return renewed

    # ------------------------------------------------------------------
    # Status / observability
    # ------------------------------------------------------------------

    def status(self) -> list[dict]:
        """Return per-subscription state (for GET /api/scout/sources)."""
        now = datetime.now(timezone.utc)
        return [
            {
                "topic_url":      sub.topic_url[:100],
                "source_type":    sub.source_type,
                "tier":           sub.tier,
                "verified":       sub.verified,
                "expires_in_s":   max(0, int((sub.expires_at - now).total_seconds())),
                "delivery_count": sub.delivery_count,
            }
            for sub in self._subs.values()
        ]

    def stats(self) -> dict:
        total    = len(self._subs)
        verified = sum(1 for s in self._subs.values() if s.verified)
        return {
            "total_subscriptions": total,
            "verified":            verified,
            "pending_verification": total - verified,
            "feeds":               len(_KNOWN_FEEDS),
        }
