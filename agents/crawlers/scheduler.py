"""
agents/crawlers/scheduler.py — Tier 2 Adaptive Polling Scheduler (ENTERPRISE EDITION)

Manages continuous background crawling across all registered crawlers.
Each crawler maintains its own adaptive poll_interval (inherited from CrawlerBase):
  - new items found      → reset to default_poll_interval
  - 3× consecutive empty → double interval (max = max_poll_interval)
  - circuit breaker open → skip until auto-reset (5 min)

Ulepszenia PRO:
  - Non-blocking Callbacks: `on_sources` jest zrzucane do tła (Fire-and-Forget), chroniąc Ticker.
  - Ochrona przed Driftem Czasowym: Ticker synchronizuje się do rzeczywistego zegara UTC, unikając opóźnień.
  - Zabezpieczenie Zamykania (Graceful Shutdown): System elegancko doczeka na koniec trwających pobrań.
  - Wysoce Wątkowa Wydajność: Kontrola obciążenia przez scentralizowany `asyncio.Semaphore`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional, Set

from agents.crawlers.base import CrawlerBase
from agents.crawlers.dedup import DedupPipeline

if False:  # TYPE_CHECKING — avoid circular import
    from agents.topic_scout import ScoutSource

logger = logging.getLogger("foundry.agents.crawlers.scheduler")

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
            "max_history_size": 50_000 # Ochrona pamięci (Enterprise parameter)
        }
        
        # next_run_at: source_id → datetime when crawler should next fire
        now = datetime.now(timezone.utc)
        self._next_run: dict[str, datetime] = {
            c.source_id: now for c in crawlers
        }
        self._running = False
        self._total_fired: int = 0
        self._total_unique: int = 0
        
        # Słownik śledzący działające w tle zapisy do bazy
        self._background_tasks: Set[asyncio.Task] = set()

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
        Build a scheduler with all crawlers from layers A, B, C, D, E.
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
            "🚀 [Scheduler] Zainicjalizowano z %d crawlerami z puli %d dostępnych.",
            len(crawlers), len(all_crawlers),
        )
        return cls(crawlers=crawlers, max_concurrent=max_concurrent)

    # ------------------------------------------------------------------
    # Async Background Task Management (Ochrona Pętli)
    # ------------------------------------------------------------------
    
    def _fire_and_forget_callback(self, coro: Awaitable[None]) -> None:
        """Zapobiega blokowaniu Tickera przez powolne zapisy bazy danych w Callbacku."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)


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
            "[Scheduler] Start pętli głównej — %d crawlers, tick=%ds, max_concurrent=%d",
            len(self._crawlers), TICK_SECONDS, self._max_concurrent,
        )

        while self._running:
            loop_start = time.monotonic()
            
            if stop_event and stop_event.is_set():
                logger.info("[Scheduler] Otrzymano sygnał zatrzymania (stop_event) — zamykanie.")
                break

            try:
                sources = await self.tick_once(query_fn())
                if sources:
                    # Rzutowanie do tła, aby `save()` nie spowolniło kolejnego Tick-a!
                    self._fire_and_forget_callback(on_sources(sources))
            except Exception as exc:
                logger.error("[Scheduler] Błąd krytyczny na pętli tick: %s", exc, exc_info=True)

            # Ochrona przed tzw. Clock Drift (opóźnienie obliczane względem trwania Ticka)
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, TICK_SECONDS - elapsed)
            
            try:
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break

        self._running = False
        logger.info(
            "[Scheduler] Pętla Zakończona. Fired=%d, Total Unique=%d",
            self._total_fired, self._total_unique,
        )
        
        # Graceful Shutdown - czekamy na dokończenie zapisów z _fire_and_forget
        if self._background_tasks:
            logger.info(f"[Scheduler] Oczekiwanie na ukończenie {len(self._background_tasks)} zadań w tle...")
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

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
        logger.debug("[Scheduler] Tick: %d/%d crawlerów zakwalifikowanych do odpytania.", len(due), len(self._crawlers))

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
                logger.debug("[Scheduler] Crawler %s zrzucił błąd i został zatrzymany: %s", crawler.source_id, batch)
                batch = []
            all_sources.extend(batch)
            
            # Schedule next run based on current (potentially adapted) poll_interval
            self._next_run[crawler.source_id] = (
                datetime.now(timezone.utc)
                + timedelta(seconds=crawler.poll_interval)
            )

        if not all_sources:
            return []

        # Cross-layer dedup (Instancjonujemy dedup do przefiltrowania zebranych batchy)
        dedup = DedupPipeline(**self._dedup_kwargs)
        unique = await dedup.filter(all_sources)
        self._total_unique += len(unique)

        logger.info(
            "[Scheduler] Wynik Ticku: Odsiew %d zebranych → %d unikalnych. Zaangażowano %d crawlerów.",
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
                "last_seen_id":       c.last_seen_id,
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
            "pending_bg_tasks": len(self._background_tasks),
        }
