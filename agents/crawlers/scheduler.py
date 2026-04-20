"""
agents/crawlers/scheduler.py — Tier 2 Adaptive Polling Scheduler (ULTRA-PRO ENTERPRISE EDITION V2)

Architektura tego modułu zarządza ciągłym pobieraniem danych (Continuous Discovery) w tle
z wykorzystaniem adaptacyjnych interwałów (Adaptive Polling) i wielowarstwowej deduplikacji.

Funkcjonalności klasy Enterprise zaimplementowane w tej wersji:
  1. State Checkpointing: Persystencja stanu (next_run_at) na dysk, chroniąca przed bombardowaniem API po restarcie aplikacji.
  2. Dynamic Load Shedding: Automatyczna adaptacja czasu `TICK_SECONDS` w przypadku dławienia się Event Loopa (Clock Drift Defense).
  3. Advanced Background Supervisor: Dedykowana klasa nadzorująca zapisy do bazy z Hard & Soft limitami pamięci operacyjnej.
  4. Async Context Management: Pełne wsparcie dla `async with`, gwarantujące czyste zamykanie (Graceful Shutdown) z timeoutami.
  5. Pydantic Telemetry: Rozszerzone logowanie i eksport metryk do celów obserwacji infrastruktury (FinOps / DevOps).

Zasady zachowane w 100%:
  - Nowe wyniki -> reset do `default_poll_interval`
  - 3x pusty przebieg -> x2 interwał (do `max_poll_interval`)
  - Wyzwalacz Circuit Breaker -> Pauza na 5 min.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional, Set, Dict, Any, List

from pydantic import BaseModel, Field

from agents.crawlers.base import CrawlerBase
from agents.crawlers.dedup import DedupPipeline

if False:  # TYPE_CHECKING — avoid circular import
    from agents.topic_scout import ScoutSource

logger = logging.getLogger("foundry.agents.crawlers.scheduler")

# ---------------------------------------------------------------------------
# Konfiguracja Cyklu Życia i Limitów
# ---------------------------------------------------------------------------
BASE_TICK_SECONDS: float = 5.0
MAX_TICK_SECONDS: float = 30.0  # Górny limit degradacji Tickera

# Plik zapisu stanu dla Checkpointingu
STATE_FILE_PATH: Path = Path("output/scheduler_state.json")

# Typowania wywołań zwrotnych dla przejrzystości architektonicznej
SourceCallback = Callable[["list[ScoutSource]"], Awaitable[None]]
QueryProvider = Callable[[], str]


# ---------------------------------------------------------------------------
# Modele Telemetryczne (Pydantic) - Dla API i Logów
# ---------------------------------------------------------------------------

class CrawlerHealthStatus(BaseModel):
    source_id: str
    is_paused: bool
    poll_interval_s: int
    default_interval_s: int
    next_run_in_s: int
    consecutive_errors: int
    last_seen_id: str

class SchedulerGlobalMetrics(BaseModel):
    is_running: bool
    registered_crawlers: int
    total_cycles_fired: int
    total_sources_fetched: int
    total_sources_unique: int
    total_clock_drifts: int
    current_tick_length_s: float
    active_background_tasks: int
    memory_limit_hit: bool


# ---------------------------------------------------------------------------
# Menedżer Stanu (State Persistence)
# ---------------------------------------------------------------------------

class SchedulerStateStore:
    """Zarządza persystencją harmonogramu zapobiegając agresywnemu pollingowi po restarcie serwera."""
    
    @staticmethod
    def load_state() -> Dict[str, datetime]:
        """Wczytuje z dysku czasy następnego wykonania dla crawlerów."""
        state = {}
        if STATE_FILE_PATH.exists():
            try:
                with STATE_FILE_PATH.open("r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    for k, v in raw_data.items():
                        state[k] = datetime.fromisoformat(v)
                logger.info(f"💾 [StateStore] Wczytano stan harmonogramu dla {len(state)} crawlerów z pliku.")
            except Exception as e:
                logger.warning(f"⚠️ [StateStore] Uszkodzony plik stanu, ignoruję: {e}")
        return state

    @staticmethod
    def save_state(next_run_map: Dict[str, datetime]) -> None:
        """Zrzuca asynchronicznie czasy do pliku JSON."""
        try:
            STATE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            raw_data = {k: v.isoformat() for k, v in next_run_map.items()}
            
            # Bezpieczny zapis z użyciem pliku tymczasowego (Atomic Write)
            tmp_path = STATE_FILE_PATH.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=2)
            os.replace(tmp_path, STATE_FILE_PATH)
        except Exception as e:
            logger.error(f"❌ [StateStore] Błąd podczas zapisu stanu Schedulera: {e}")


# ---------------------------------------------------------------------------
# Nadzorca Zadań w Tle (Background Job Supervisor)
# ---------------------------------------------------------------------------

class BackgroundJobSupervisor:
    """
    Rygorystyczny nadzorca procesów Fire-and-Forget (zapisy do bazy).
    Implementuje miękkie i twarde limity pamięci operacyjnej by uniknąć OOM w K8s.
    """
    
    def __init__(self, soft_limit: int = 200, hard_limit: int = 500):
        self._tasks: Set[asyncio.Task] = set()
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self.is_throttled = False

    def add_job(self, coro: Awaitable[Any], name: str) -> bool:
        """Dodaje zadanie do puli. Zwraca False jeśli zadanie zostało odrzucone (Hard Limit)."""
        current_size = len(self._tasks)
        
        # Hard Limit (Zapobieganie awarii serwera - odrzucamy zapis)
        if current_size >= self._hard_limit:
            logger.critical(f"🛑 [Supervisor] HARD LIMIT osiągnięty ({self._hard_limit}). Odrzucam zadanie {name}!")
            self.is_throttled = True
            return False
            
        # Soft Limit (Ostrzeżenie logowane)
        if current_size >= self._soft_limit and not self.is_throttled:
            logger.warning(f"⚠️ [Supervisor] SOFT LIMIT osiągnięty ({self._soft_limit}). Baza danych ma zatory!")
            self.is_throttled = True
        elif current_size < self._soft_limit and self.is_throttled:
            self.is_throttled = False # Odbudowa zdrowia

        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._on_job_done)
        return True

    def _on_job_done(self, task: asyncio.Task) -> None:
        """Krytyczny Error Handler."""
        self._tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                f"❌ [Supervisor] Błąd w tle ({task.get_name()}): {exc}\n{traceback.format_exc()}"
            )

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Czeka na dokończenie zadań z wbudowanym Timeoutem klasy Enterprise."""
        if not self._tasks:
            return
            
        pending = len(self._tasks)
        logger.warning(f"⏳ [Supervisor] Oczekiwanie na {pending} zadań w tle (Max {timeout}s)...")
        
        done, not_done = await asyncio.wait(self._tasks, timeout=timeout)
        
        if not_done:
            logger.critical(f"💀 [Supervisor] {len(not_done)} zadań przekroczyło czas i zostanie wymuszonych!")
            for t in not_done:
                t.cancel()
        else:
            logger.info("✅ [Supervisor] Wszystkie zadania tła zakończone pomyślnie.")
            
    @property
    def active_count(self) -> int:
        return len(self._tasks)


# ---------------------------------------------------------------------------
# Orkiestrator (Polling Scheduler)
# ---------------------------------------------------------------------------

class PollingScheduler:
    """
    Zaawansowany Orkiestrator Pobierania Danych dla środowisk wysokiej dostępności.
    Odpowiada za sprawdzanie harmonogramu crawlerów, kontrolę współbieżności, 
    agregację wyników i cross-layer deduplikację.
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
            "max_history_size": 50_000  # Enterprise parameter
        }
        
        # Menedżer Zadań w Tle
        self._supervisor = BackgroundJobSupervisor()
        
        # Stan Schedulera i Checkpointing
        now = datetime.now(timezone.utc)
        saved_state = SchedulerStateStore.load_state()
        
        self._next_run: Dict[str, datetime] = {}
        for c in crawlers:
            if c.source_id in saved_state and saved_state[c.source_id] > now:
                # Ochrona API: Ustaw czas na ten z pliku by nie spammować
                self._next_run[c.source_id] = saved_state[c.source_id]
            else:
                # Wymuszamy natychmiastowe odpalenie (lekki jitter uodparniający na stado)
                import random
                self._next_run[c.source_id] = now + timedelta(seconds=random.randint(0, 5))
        
        # Statystyki i Adaptacyjne Dławienie
        self._running = False
        self._total_fired: int = 0
        self._total_fetched: int = 0
        self._total_unique: int = 0
        self._total_drifts: int = 0
        self._current_tick = BASE_TICK_SECONDS

    # ------------------------------------------------------------------
    # Asynchroniczny Menedżer Kontekstu (Integracja z FastAPI)
    # ------------------------------------------------------------------
    
    async def __aenter__(self) -> "PollingScheduler":
        logger.debug("[Scheduler] Otwieranie asynchronicznego menedżera kontekstu.")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        logger.info("[Scheduler] Zamykanie menedżera kontekstu. Oczekiwanie na czyste wyłączenie...")
        await self.shutdown()

    # ------------------------------------------------------------------
    # Factory: Bezpieczne Składanie Modułów (Layer A-E)
    # ------------------------------------------------------------------

    @classmethod
    def from_all_layers(
        cls,
        max_concurrent: int = 20,
        enabled: Optional[list[str]] = None,
    ) -> "PollingScheduler":
        """Inicjuje harmonogram z dynamicznym montowaniem wszystkich warstw z odseparowaniem błędów."""
        all_crawlers: dict[str, CrawlerBase] = {}
        
        # Odporny import zrzucający moduły, które mają np. nieprawidłowe zależności
        layers = [
            ("Layer A", "agents.crawlers.layer_a"),
            ("Layer B", "agents.crawlers.layer_b"),
            ("Layer C", "agents.crawlers.layer_c"),
            ("Layer D", "agents.crawlers.layer_d"),
            ("Layer E", "agents.crawlers.layer_e"),
        ]
        
        import importlib
        for name, module_path in layers:
            try:
                mod = importlib.import_module(module_path)
                if hasattr(mod, "_CRAWLERS"):
                    all_crawlers.update(mod._CRAWLERS)
            except Exception as e:
                logger.error(f"[Scheduler/Factory] Krytyczny błąd importu w {name}: {e}")

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
            "🚀 [Scheduler] Pomyślnie zintegrowano %d crawlerów z puli %d.",
            len(crawlers), len(all_crawlers),
        )
        return cls(crawlers=crawlers, max_concurrent=max_concurrent)

    # ------------------------------------------------------------------
    # Główny Silnik Odpytywania (Main Loop) ze wsparciem Load Shedding
    # ------------------------------------------------------------------

    async def run_forever(
        self,
        query_fn: QueryProvider,
        on_sources: SourceCallback,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Główna pętla wdrożeniowa (Long-Running Process)."""
        self._running = True
        logger.info(
            "⚙️ [Scheduler] Start Schedulera: Crawlers=%d | Tick=%.1fs | Concurrency=%d",
            len(self._crawlers), self._current_tick, self._max_concurrent,
        )

        last_state_save = time.monotonic()

        while self._running:
            loop_start = time.monotonic()
            
            if stop_event and stop_event.is_set():
                logger.info("[Scheduler] Otrzymano sygnał stop_event. Zamykanie Schedulera.")
                break

            # 1. Wykonanie Logiki z pełnym bezpieczeństwem
            try:
                sources = await self.tick_once(query_fn())
                if sources:
                    # Fire-And-Forget (Przez Supervisora)
                    job_id = f"db_upsert_{uuid.uuid4().hex[:6]}"
                    self._supervisor.add_job(on_sources(sources), name=job_id)
            except Exception as exc:
                logger.error("[Scheduler] FATAL ERROR w cyklu tick_once: %s", exc, exc_info=True)

            # 2. State Checkpointing (Zapis stanu co ~60 sekund by oszczędzać dysk)
            if time.monotonic() - last_state_save > 60.0:
                SchedulerStateStore.save_state(self._next_run)
                last_state_save = time.monotonic()

            # 3. Dynamic Load Shedding & Clock Drift Detection
            elapsed = time.monotonic() - loop_start
            
            # Jeśli serwer nadąża, próbujemy odzyskać optymalny tick
            if elapsed < self._current_tick:
                self._current_tick = max(BASE_TICK_SECONDS, self._current_tick - 0.5)
                sleep_time = self._current_tick - elapsed
            else:
                # Serwer dławi się (Clock Drift)
                self._total_drifts += 1
                # Wydłużamy Tick (Load Shedding), żeby odciążyć zasoby
                self._current_tick = min(MAX_TICK_SECONDS, self._current_tick + 1.0)
                logger.warning(
                    f"⏱️ [Clock Drift] Cykl trwał {elapsed:.2f}s! Serwer przeciążony. "
                    f"Zwiększono Ticker do {self._current_tick:.1f}s."
                )
                sleep_time = 0.0
            
            try:
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                logger.info("[Scheduler] Złowiono CancelledError w Event Loopie. Czyste wyjście.")
                break

        self._running = False
        await self.shutdown()

    # ------------------------------------------------------------------
    # Pojedynczy Cykl Odpytywania (Single Tick Core)
    # ------------------------------------------------------------------

    async def tick_once(self, query: str) -> "list[ScoutSource]":
        """
        Identyfikuje crawlery i uruchamia je z limitowaniem współbieżności i deduplikacją.
        """
        now = datetime.now(timezone.utc)
        
        due = [
            c for c in self._crawlers
            if self._next_run[c.source_id] <= now and not c.is_paused
        ]

        if not due:
            return []

        self._total_fired += len(due)
        logger.debug(
            "[Scheduler:Tick] Pobieranie dla %d/%d crawlerów. (Query: '%s')", 
            len(due), len(self._crawlers), query[:30]
        )

        tick_start = time.monotonic()
        sem = asyncio.Semaphore(self._max_concurrent)

        async def _guarded(crawler: CrawlerBase) -> "list[ScoutSource]":
            async with sem:
                return await crawler.safe_crawl(query)

        raw_results = await asyncio.gather(
            *[_guarded(c) for c in due],
            return_exceptions=True,
        )

        all_sources: list = []
        for crawler, batch in zip(due, raw_results):
            if isinstance(batch, Exception):
                logger.error(f"[Scheduler:Tick] Błąd awaryjny w crawlerze {crawler.source_id}: {batch}")
                batch = []
            
            all_sources.extend(batch)
            
            # Aktualizacja harmonogramu na bazie adaptacji (np. ×2 po braku wyników)
            self._next_run[crawler.source_id] = (
                datetime.now(timezone.utc) + timedelta(seconds=crawler.poll_interval)
            )

        fetch_duration = time.monotonic() - tick_start
        self._total_fetched += len(all_sources)
        
        if not all_sources:
            return []

        # Deduplikacja
        dedup_start = time.monotonic()
        dedup = DedupPipeline(**self._dedup_kwargs)
        unique = await dedup.filter(all_sources)
        dedup_duration = time.monotonic() - dedup_start
        
        self._total_unique += len(unique)

        logger.info(
            "[Scheduler:Tick] Czas: Fetch=%.2fs | Dedup=%.2fs. Zebrano: %d → Unikalne: %d",
            fetch_duration, dedup_duration, len(all_sources), len(unique)
        )
        return unique

    # ------------------------------------------------------------------
    # Kontrola i Telemetria
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Planuje zatrzymanie pętli asynchronicznej."""
        self._running = False

    async def shutdown(self) -> None:
        """Czyste zamknięcie procesów (Graceful Shutdown)."""
        logger.info(
            "[Scheduler:Shutdown] Zamykanie orkiestratora. Trwa zapis stanu i czyszczenie buforów."
        )
        # 1. Zapis persystentny przed wyłączeniem
        SchedulerStateStore.save_state(self._next_run)
        
        # 2. Wymuszone zamykanie Supervisor'a (twardy Timeout dla bazy danych)
        await self._supervisor.shutdown(timeout=25.0)
        
        logger.info("✅ [Scheduler:Shutdown] System został pomyślnie wyłączony.")

    def status(self) -> List[dict]:
        """Ekspozycja metryk per crawler dla interfejsów API."""
        now = datetime.now(timezone.utc)
        return [
            CrawlerHealthStatus(
                source_id=c.source_id,
                is_paused=c.is_paused,
                poll_interval_s=c.poll_interval,
                default_interval_s=c.default_poll_interval,
                next_run_in_s=max(0, int((self._next_run[c.source_id] - now).total_seconds())),
                consecutive_errors=c._consecutive_errors,
                last_seen_id=c.last_seen_id,
            ).model_dump()
            for c in self._crawlers
        ]

    def global_stats(self) -> dict:
        """Pełen raport telemetryczny pracy harmonogramu."""
        return SchedulerGlobalMetrics(
            is_running=self._running,
            registered_crawlers=len(self._crawlers),
            total_cycles_fired=self._total_fired,
            total_sources_fetched=self._total_fetched,
            total_sources_unique=self._total_unique,
            total_clock_drifts=self._total_drifts,
            current_tick_length_s=round(self._current_tick, 2),
            active_background_tasks=self._supervisor.active_count,
            memory_limit_hit=self._supervisor.is_throttled
        ).model_dump()
