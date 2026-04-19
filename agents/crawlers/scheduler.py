"""
agents/crawlers/scheduler.py — Tier 2 Adaptive Polling Scheduler

Manages continuous background crawling across all registered crawlers.
Each crawler maintains its own adaptive poll_interval (inherited from CrawlerBase):
  - new items found      → reset to default_poll_interval
  - 3× consecutive empty → double interval (max = max_poll_interval)
  - circuit breaker open → skip until auto-reset (5 min)

The scheduler ticks every TICK_SECONDS (default 5 s), checks which crawlers
are due (now >= next_run_at), runs them concurrently, deduplicates results
via DedupPipeline, then calls on_sources() callback with unique new sources.

Usage (background task):
    scheduler = PollingScheduler.from_all_layers(max_concurrent=20)
    stop = asyncio.Event()

    async def save(sources):
        for s in sources:
            await db.upsert(s)

    task = asyncio.create_task(
        scheduler.run_forever(
            query_fn=lambda: "CSRD reporting",
            on_sources=save,
            stop_event=stop,
        )
    )
    # later:
    stop.set()
    await task

Usage (one-shot manual tick for testing):
    scheduler = PollingScheduler(crawlers=[...])
    sources = await scheduler.tick_once("EU AI Act")
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

from agents.crawlers.base import CrawlerBase
from agents.crawlers.dedup import DedupPipeline

if False:  # TYPE_CHECKING — avoid circular import
    from agents.topic_scout import ScoutSource

logger = logging.getLogger(__name__)

# Scheduler check interval — how often we check if any crawler is due
TICK_SECONDS: int = 5

SourceCallback = Callable[["list[ScoutSource]"], Awaitable[None]]
QueryProvider = Callable[[], str]


class PollingScheduler:
    """
    Adaptive polling scheduler for Tier 2 continuous discovery.

    Each crawler is independently scheduled based on its current poll_interval.
    Results are cross-layer deduplicated before being passed to on_sources().
    """

    def __init__(
        self,
        crawlers: list[CrawlerBase],
        max_concurrent: int = 20,
        dedup_simhash_threshold: int = 3,
        dedup_semantic_threshold: float = 0.92,
        enable_semantic_dedup: bool = True,
    ) -> None:
        self._crawlers = crawlers
        self._max_concurrent = max_concurrent
        self._dedup_kwargs = {
            "simhash_threshold": dedup_simhash_threshold,
            "semantic_threshold": dedup_semantic_threshold,
            "enable_semantic": enable_semantic_dedup,
        }
        # next_run_at: source_id → datetime when crawler should next fire
        now = datetime.now(timezone.utc)
        self._next_run: dict[str, datetime] = {
            c.source_id: now for c in crawlers
        }
        self._running = False
        self._total_fired: int = 0
        self._total_unique: int = 0

    # ------------------------------------------------------------------
    # Factory — assemble from all registered layers
    # ------------------------------------------------------------------

    @classmethod
    def from_all_layers(
        cls,
        max_concurrent: int = 20,
        enabled: Optional[list[str]] = None,
    ) -> "PollingScheduler":
        """
        Build a scheduler with all crawlers from layers A, B, C.
        If enabled is given, only those source_ids are loaded.
        """
        from agents.crawlers.layer_a import _CRAWLERS as A
        from agents.crawlers.layer_b import _CRAWLERS as B
        from agents.crawlers.layer_c import _CRAWLERS as C
        from agents.crawlers.layer_d import _CRAWLERS as D
        from agents.crawlers.layer_e import _CRAWLERS as E

        all_crawlers: dict[str, CrawlerBase] = {**A, **B, **C, **D, **E}

        if enabled:
            crawlers = [all_crawlers[sid] for sid in enabled if sid in all_crawlers]
        else:
            try:
                from config.settings import settings
                cfg = getattr(settings, "scout_sources_enabled", [])
                if cfg:
                    crawlers = [all_crawlers[sid] for sid in cfg if sid in all_crawlers]
                else:
                    crawlers = list(all_crawlers.values())
            except Exception:
                crawlers = list(all_crawlers.values())

        logger.info(
            "[scheduler] initialised with %d crawlers from %d available",
            len(crawlers), len(all_crawlers),
        )
        return cls(crawlers=crawlers, max_concurrent=max_concurrent)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run_forever(
        self,
        query_fn: QueryProvider,
        on_sources: SourceCallback,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """
        Run the scheduling loop indefinitely.

        Args:
            query_fn:    zero-arg callable returning current search query string.
            on_sources:  async callback receiving unique deduplicated ScoutSource list.
            stop_event:  asyncio.Event — set to request graceful shutdown.
        """
        self._running = True
        logger.info(
            "[scheduler] starting — %d crawlers, tick=%ds, max_concurrent=%d",
            len(self._crawlers), TICK_SECONDS, self._max_concurrent,
        )

        while self._running:
            if stop_event and stop_event.is_set():
                logger.info("[scheduler] stop_event set — exiting loop")
                break

            try:
                sources = await self.tick_once(query_fn())
                if sources:
                    try:
                        await on_sources(sources)
                    except Exception as exc:
                        logger.error("[scheduler] on_sources callback failed: %s", exc)
            except Exception as exc:
                logger.error("[scheduler] tick error: %s", exc)

            try:
                await asyncio.sleep(TICK_SECONDS)
            except asyncio.CancelledError:
                break

        self._running = False
        logger.info(
            "[scheduler] stopped — fired=%d total_unique=%d",
            self._total_fired, self._total_unique,
        )

    # ------------------------------------------------------------------
    # Single tick (also usable standalone for testing)
    # ------------------------------------------------------------------

    async def tick_once(self, query: str) -> "list[ScoutSource]":
        """
        Fire all crawlers that are currently due.
        Returns deduplicated list of new sources.
        """
        now = datetime.now(timezone.utc)
        due = [
            c for c in self._crawlers
            if self._next_run[c.source_id] <= now and not c.is_paused
        ]

        if not due:
            return []

        self._total_fired += len(due)
        logger.debug("[scheduler] tick: %d/%d crawlers due", len(due), len(self._crawlers))

        # Run due crawlers with bounded concurrency
        sem = asyncio.Semaphore(self._max_concurrent)

        async def _guarded(crawler: CrawlerBase) -> "list[ScoutSource]":
            async with sem:
                return await crawler.safe_crawl(query)

        raw_results = await asyncio.gather(
            *[_guarded(c) for c in due],
            return_exceptions=True,
        )

        # Merge + schedule next run
        all_sources: list = []
        for crawler, batch in zip(due, raw_results):
            if isinstance(batch, Exception):
                logger.debug("[scheduler] %s raised: %s", crawler.source_id, batch)
                batch = []
            all_sources.extend(batch)
            # Schedule next run based on current (potentially adapted) poll_interval
            self._next_run[crawler.source_id] = (
                datetime.now(timezone.utc)
                + timedelta(seconds=crawler.poll_interval)
            )

        if not all_sources:
            return []

        # Cross-layer dedup
        dedup = DedupPipeline(**self._dedup_kwargs)
        unique = await dedup.filter(all_sources)
        self._total_unique += len(unique)

        logger.info(
            "[scheduler] tick complete: %d raw → %d unique from %d crawlers",
            len(all_sources), len(unique), len(due),
        )
        return unique

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Request the run_forever loop to stop after current tick."""
        self._running = False

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def status(self) -> list[dict]:
        """Return per-crawler scheduling state for GET /api/scout/sources."""
        now = datetime.now(timezone.utc)
        return [
            {
                "source_id":         c.source_id,
                "poll_interval_s":   c.poll_interval,
                "default_interval_s": c.default_poll_interval,
                "is_paused":         c.is_paused,
                "next_run_in_s":     max(
                    0,
                    int((self._next_run[c.source_id] - now).total_seconds()),
                ),
                "consecutive_errors": c._consecutive_errors,
                "last_seen_id":      c.last_seen_id,
            }
            for c in self._crawlers
        ]

    def global_stats(self) -> dict:
        """Aggregate scheduler statistics."""
        return {
            "running":       self._running,
            "crawler_count": len(self._crawlers),
            "total_fired":   self._total_fired,
            "total_unique":  self._total_unique,
        }
