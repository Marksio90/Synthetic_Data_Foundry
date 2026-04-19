"""
agents/crawlers/base.py — Abstract base class for all Gap Scout source crawlers.

Every concrete crawler inherits CrawlerBase and implements crawl().
Infrastructure provided out of the box:
  • circuit breaker  — 5 consecutive errors → 5-min pause → auto-resume
  • HTTP retry       — exponential backoff + jitter via tenacity (max 3 attempts)
  • adaptive polling — 3× no-new-items → doubles poll_interval (capped at max)
  • cache headers    — ETag / If-None-Match, Last-Modified / If-Modified-Since
  • rate-limit guard — logs warning when X-RateLimit-Remaining < 5
  • Prometheus hooks — fetch duration histogram, verification failure counter
  • structured logs  — JSON extra={} on every warning/error

Usage:
    class ArxivCrawler(CrawlerBase):
        source_id = "arxiv"
        default_poll_interval = 120

        async def crawl(self, query: str) -> list[ScoutSource]:
            resp = await self._fetch("https://export.arxiv.org/api/query?...")
            ...
            return sources
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, ClassVar, Optional

import httpx

if TYPE_CHECKING:
    # Import only for type annotations — avoids circular import with topic_scout.py
    from agents.topic_scout import ScoutSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional tenacity — graceful degradation if not installed
# ---------------------------------------------------------------------------

try:
    from tenacity import (
        AsyncRetrying,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
        wait_random,
    )
    _HAS_TENACITY = True
except ImportError:
    _HAS_TENACITY = False
    logger.warning("tenacity not installed — HTTP retries disabled (pip install tenacity)")


class CrawlerBase(ABC):
    """
    Abstract base for all Gap Scout crawlers.

    Class variables to override in subclasses:
        source_id              unique ID used in logs and metrics (required)
        default_poll_interval  seconds between polls (default 300 = 5 min)
        max_poll_interval      upper bound for adaptive back-off (default 3600)
    """

    source_id: ClassVar[str] = "base"
    default_poll_interval: ClassVar[int] = 300     # 5 min
    max_poll_interval: ClassVar[int] = 3_600       # 1 h

    _CB_THRESHOLD: ClassVar[int] = 5               # errors before circuit opens
    _CB_PAUSE_SECONDS: ClassVar[int] = 300         # 5-min circuit-open pause

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self.poll_interval: int = self.default_poll_interval
        self.last_seen_id: str = ""          # deduplication watermark — set by crawl()
        self._consecutive_errors: int = 0
        self._no_new_count: int = 0          # consecutive polls without new items
        self._paused_until: Optional[datetime] = None
        self._etag: str = ""
        self._last_modified: str = ""
        self._http = httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "FoundryScout/2.0 (research-tool; non-commercial)"},
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def crawl(self, query: str) -> "list[ScoutSource]":
        """
        Fetch new sources for *query* from this source.

        Implementations:
          - MUST use self._fetch() for all HTTP calls (retry + cache headers)
          - SHOULD update self.last_seen_id after a successful fetch
          - MUST NOT catch all exceptions — let safe_crawl() handle them
          - MUST return only newly-seen sources (dedup by last_seen_id)
        """
        ...

    # ------------------------------------------------------------------
    # Public entry-point — wraps crawl() with circuit breaker
    # ------------------------------------------------------------------

    async def safe_crawl(self, query: str) -> "list[ScoutSource]":
        """
        Call crawl() with circuit breaker + adaptive-polling + metrics.
        Always returns a list (empty when paused or on error) — never raises.
        """
        if self._is_paused():
            logger.debug(
                "[%s] circuit breaker active until %s — skipping",
                self.source_id, self._paused_until,
            )
            return []

        t0 = time.monotonic()
        try:
            results = await self.crawl(query)
            elapsed = time.monotonic() - t0
            self._on_success(bool(results), elapsed)
            return results
        except Exception as exc:
            elapsed = time.monotonic() - t0
            self._on_error(exc, elapsed)
            return []

    # ------------------------------------------------------------------
    # HTTP helper — use inside crawl() implementations
    # ------------------------------------------------------------------

    async def _fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        use_cache_headers: bool = True,
        **kwargs,
    ) -> httpx.Response:
        """
        Perform an HTTP request with exponential-backoff retry.

        - 4xx responses are returned immediately (not retried — client error).
        - 5xx / network errors retry up to 3 times with exp backoff + jitter.
        - Automatically stores and sends ETag / Last-Modified cache headers.
        - Checks X-RateLimit-Remaining and logs warning when < 5.
        """
        extra_headers = {
            **(self.cache_headers() if use_cache_headers else {}),
            **kwargs.pop("headers", {}),
        }

        async def _do() -> httpx.Response:
            resp = await self._http.request(method, url, headers=extra_headers, **kwargs)
            if 400 <= resp.status_code < 500:
                return resp  # client error — return, do NOT retry
            resp.raise_for_status()
            return resp

        if _HAS_TENACITY:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 2),
                retry=retry_if_exception_type(
                    (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)
                ),
                reraise=True,
            ):
                with attempt:
                    resp = await _do()
        else:
            resp = await _do()

        if use_cache_headers:
            self._update_cache(resp.headers)
        self._check_rate_limit(resp.headers)
        return resp

    # ------------------------------------------------------------------
    # ETag / Last-Modified caching
    # ------------------------------------------------------------------

    def cache_headers(self) -> dict[str, str]:
        """Build conditional request headers from stored ETag / Last-Modified."""
        h: dict[str, str] = {}
        if self._etag:
            h["If-None-Match"] = self._etag
        if self._last_modified:
            h["If-Modified-Since"] = self._last_modified
        return h

    def _update_cache(self, headers: httpx.Headers) -> None:
        if etag := headers.get("etag", ""):
            self._etag = etag
        if lm := headers.get("last-modified", ""):
            self._last_modified = lm

    # ------------------------------------------------------------------
    # Rate-limit guard
    # ------------------------------------------------------------------

    def _check_rate_limit(self, headers: httpx.Headers) -> None:
        remaining = headers.get("x-ratelimit-remaining", "")
        if remaining:
            try:
                if int(remaining) < 5:
                    logger.warning(
                        "[%s] rate limit critically low: %s request(s) remaining",
                        self.source_id, remaining,
                        extra={"crawler": self.source_id, "rl_remaining": remaining},
                    )
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Circuit breaker internals
    # ------------------------------------------------------------------

    def _is_paused(self) -> bool:
        if self._paused_until is None:
            return False
        if datetime.now(timezone.utc) >= self._paused_until:
            self._paused_until = None
            logger.info(
                "[%s] circuit breaker CLOSED — resuming",
                self.source_id,
                extra={"crawler": self.source_id, "event": "circuit_breaker_closed"},
            )
            return False
        return True

    def _on_success(self, found_new: bool, elapsed: float) -> None:
        self._consecutive_errors = 0
        if found_new:
            self._no_new_count = 0
            self.poll_interval = self.default_poll_interval
        else:
            self._no_new_count += 1
            if self._no_new_count >= 3:
                new_iv = min(self.poll_interval * 2, self.max_poll_interval)
                if new_iv != self.poll_interval:
                    logger.debug(
                        "[%s] no new items ×3 → poll_interval %ds → %ds",
                        self.source_id, self.poll_interval, new_iv,
                        extra={"crawler": self.source_id, "poll_interval": new_iv},
                    )
                self.poll_interval = new_iv
                self._no_new_count = 0
        _record_fetch(self.source_id, elapsed, success=True)

    def _on_error(self, exc: Exception, elapsed: float) -> None:
        self._consecutive_errors += 1
        logger.warning(
            "[%s] crawl error (%d/%d): %s",
            self.source_id, self._consecutive_errors, self._CB_THRESHOLD, exc,
            extra={"crawler": self.source_id, "error": str(exc)},
        )
        if self._consecutive_errors >= self._CB_THRESHOLD:
            self._paused_until = (
                datetime.now(timezone.utc)
                + timedelta(seconds=self._CB_PAUSE_SECONDS)
            )
            self._consecutive_errors = 0
            logger.warning(
                "[%s] circuit breaker OPEN — pausing %ds until %s",
                self.source_id, self._CB_PAUSE_SECONDS,
                self._paused_until.isoformat(),
                extra={
                    "crawler": self.source_id,
                    "event": "circuit_breaker_open",
                    "paused_until": self._paused_until.isoformat(),
                },
            )
        _record_fetch(self.source_id, elapsed, success=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        """True when circuit breaker is open (read-only, safe to call any time)."""
        return self._paused_until is not None and datetime.now(timezone.utc) < self._paused_until

    async def aclose(self) -> None:
        """Close the underlying HTTP client (call on shutdown)."""
        await self._http.aclose()

    def __repr__(self) -> str:
        status = "PAUSED" if self.is_paused else f"poll={self.poll_interval}s"
        return f"<{self.__class__.__name__} source_id={self.source_id!r} {status}>"


# ---------------------------------------------------------------------------
# Prometheus shim — safe no-op if prometheus_client not installed or not wired
# ---------------------------------------------------------------------------

def _record_fetch(source_id: str, elapsed: float, *, success: bool) -> None:
    """Emit Prometheus metrics for a single crawl attempt. Never raises."""
    try:
        from api.monitoring import (
            _AVAILABLE,
            scout_fetch_duration,
            scout_verification_failures,
        )
        if not _AVAILABLE:
            return
        scout_fetch_duration.labels(source=source_id).observe(elapsed)
        if not success:
            scout_verification_failures.labels(
                reason="crawl_error", source=source_id
            ).inc()
    except Exception:
        pass
