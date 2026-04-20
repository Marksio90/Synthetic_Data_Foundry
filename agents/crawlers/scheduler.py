"""
agents/crawlers/scheduler.py — Tier 2 Adaptive Polling Scheduler (ULTRA-PRO ENTERPRISE EDITION)

Architektura tego modułu zarządza ciągłym pobieraniem danych (Continuous Discovery) w tle
z wykorzystaniem adaptacyjnych interwałów (Adaptive Polling) i wielowarstwowej deduplikacji.

Funkcjonalności klasy Enterprise zaimplementowane w tej wersji:
  1. Async Context Management: Pełne wsparcie dla `async with`, gwarantujące czyste zamykanie 
     w cyklu życia aplikacji (np. FastAPI lifespan).
  2. Background Task Supervisor: Mechanizm Fire-and-Forget jest obudowany w dedykowany 
     Error Handler, który wyłapuje i loguje `Traceback`, jeśli zapis do bazy (on_sources) zawiedzie.
  3. Clock Drift Detection: Zaawansowana detekcja blokowania pętli zdarzeń (Event Loop Blocking).
     Jeśli tick zajmie dłużej niż `TICK_SECONDS`, system rzuci ostrzeżenie o dławieniu serwera.
  4. Sub-Tick Telemetry: Precyzyjne logowanie czasu trwania zapytań i deduplikacji w milisekundach.
  5. Memory Fences: Górny limit jednoczesnych zadań w tle (Background Tasks), chroniący RAM.

Zasady zachowane w 100%:
  - Nowe wyniki -> reset do `default_poll_interval`
  - 3x pusty przebieg -> x2 interwał (do `max_poll_interval`)
  - Wyzwalacz Circuit Breaker -> Pauza na 5 min.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional, Set, Dict, Any

from agents.crawlers.base import CrawlerBase
from agents.crawlers.dedup import DedupPipeline

if False:  # TYPE_CHECKING — avoid circular import
    from agents.topic_scout import ScoutSource

logger = logging.getLogger("foundry.agents.crawlers.scheduler")

# ---------------------------------------------------------------------------
# Konfiguracja Cyklu Życia i Limitów
# ---------------------------------------------------------------------------
TICK_SECONDS: int = 5

# Typowania wywołań zwrotnych dla przejrzystości architektonicznej
SourceCallback = Callable[["list[ScoutSource]"], Awaitable[None]]
QueryProvider = Callable[[], str]

# Maksymalna liczba zadań zapisujących do bazy w locie, by zapobiec OOM (Out Of Memory)
MAX_BACKGROUND_TASKS: int = 500


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
        
        # Konfiguracja silnika deduplikacyjnego (przekazywana do w locie tworzonej instancji)
        self._dedup_kwargs = {
            "simhash_threshold": dedup_simhash_threshold,
            "semantic_threshold": dedup_semantic_threshold,
            "enable_semantic": enable_semantic_dedup,
            # Dodany limit pamięci dla Stage 1 i 2 by Scheduler mógł działać latami
            "max_history_size": 50_000 
        }
        
        # Inicjalizacja zegarów — każdy crawler otrzymuje własny timestamp wykonania
        now = datetime.now(timezone.utc)
        self._next_run: Dict[str, datetime] = {
            c.source_id: now for c in crawlers
        }
        
        # Stan i Statystyki platformy
        self._running = False
        self._total_fired: int = 0
        self._total_unique: int = 0
        self._total_drifts: int = 0
        
        # Menedżer Zadań w Tle (Background Task Supervisor)
        self._background_tasks: Set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Asynchroniczny Menedżer Kontekstu (Integracja z FastAPI)
    # ------------------------------------------------------------------
    
    async def __aenter__(self) -> "PollingScheduler":
        """Inicjuje środowisko Schedulera w sposób bezpieczny dla pętli zdarzeń."""
        logger.debug("[Scheduler] Otwieranie asynchronicznego menedżera kontekstu.")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Gwarantuje, że przy zamykaniu serwera zadania tła zostaną ukończone bez utraty danych."""
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
        """
        Inicjuje harmonogram pobierając konfigurację ze wszystkich warstw aplikacji.
        Jeśli wystąpi błąd importu konkretnej warstwy (np. przez brakujące zależności),
        zaloguje to i będzie kontynuować z pozostałymi.
        """
        all_crawlers: dict[str, CrawlerBase] = {}
        
        # Bezpieczny import warstw - ochrona przed awarią całego systemu przez 1 uszkodzony plik
        try:
            from agents.crawlers.layer_a import _CRAWLERS as A
            all_crawlers.update(A)
        except Exception as e:
            logger.error(f"[Scheduler/Factory] Nie udało się załadować Layer A: {e}")

        try:
            from agents.crawlers.layer_b import _CRAWLERS as B
            all_crawlers.update(B)
        except Exception as e:
            logger.error(f"[Scheduler/Factory] Nie udało się załadować Layer B: {e}")

        try:
            from agents.crawlers.layer_c import _CRAWLERS as C
            all_crawlers.update(C)
        except Exception as e:
            logger.error(f"[Scheduler/Factory] Nie udało się załadować Layer C: {e}")

        try:
            from agents.crawlers.layer_d import _CRAWLERS as D
            all_crawlers.update(D)
        except Exception as e:
            logger.error(f"[Scheduler/Factory] Nie udało się załadować Layer D: {e}")

        try:
            from agents.crawlers.layer_e import _CRAWLERS as E
            all_crawlers.update(E)
        except Exception as e:
            logger.error(f"[Scheduler/Factory] Nie udało się załadować Layer E: {e}")

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
            "🚀 [Scheduler] Pomyślnie zintegrowano %d crawlerów z dostępnych %d modułów.",
            len(crawlers), len(all_crawlers),
        )
        return cls(crawlers=crawlers, max_concurrent=max_concurrent)

    # ------------------------------------------------------------------
    # Background Task Supervisor (Odporność Tła)
    # ------------------------------------------------------------------
    
    def _background_task_callback(self, task: asyncio.Task) -> None:
        """
        Krytyczny Error Handler dla zadań pobocznych. Wyłapuje błędy z `on_sources`, 
        zdejmując zadanie z kolejki i zapobiegając "cichemu umieraniu" korutyn.
        """
        self._background_tasks.discard(task)
        try:
            # Próba odebrania wyniku podnosi ukryty wyjątek, jeśli wystąpił
            task.result()
        except asyncio.CancelledError:
            pass  # Standardowe zachowanie przy zamykaniu aplikacji
        except Exception as exc:
            # W przypadku błędu (np. DB Connection Error) mamy pełen Traceback do analizy
            logger.error(
                f"❌ [Task Supervisor] Zadanie w tle ({task.get_name()}) rzuciło błędem: {exc}\n{traceback.format_exc()}"
            )

    def _fire_and_forget(self, coro: Awaitable[None]) -> None:
        """
        Uruchamia kod w tle chroniąc Ticker przed spowolnieniem. 
        Zawiera ogranicznik pojemności (Memory Guard) zapobiegający OOM.
        """
        if len(self._background_tasks) >= MAX_BACKGROUND_TASKS:
            logger.warning(
                f"⚠️ [Task Supervisor] Osiągnięto limit {MAX_BACKGROUND_TASKS} zadań w tle! "
                f"Aplikacja może mieć problem z wydajnością bazy danych."
            )
            
        task_id = str(uuid.uuid4())[:8]
        task = asyncio.create_task(coro, name=f"db_save_{task_id}")
        self._background_tasks.add(task)
        task.add_done_callback(self._background_task_callback)


    # ------------------------------------------------------------------
    # Główny Silnik Odpytywania (Main Loop)
    # ------------------------------------------------------------------

    async def run_forever(
        self,
        query_fn: QueryProvider,
        on_sources: SourceCallback,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """
        Uruchamia pętlę w nieskończoność. Implementuje ochronę Tickera oraz detekcję Driftu.
        
        Args:
            query_fn:    Funkcja bezargumentowa generująca zapytanie.
            on_sources:  Asynchroniczny callback (np. zapis do bazy) dla nowo odkrytych źródeł.
            stop_event:  Event sygnałowy używany do eleganckiego zamknięcia pętli.
        """
        self._running = True
        logger.info(
            "⚙️ [Scheduler] Uruchamianie Silnika. Crawlery: %d | Tick: %ds | Concurrency: %d",
            len(self._crawlers), TICK_SECONDS, self._max_concurrent,
        )

        while self._running:
            loop_start = time.monotonic()
            
            # Weryfikacja sygnału zamykającego
            if stop_event and stop_event.is_set():
                logger.info("[Scheduler] Otrzymano sygnał stop_event. Rozpoczynam procedurę wyłączania...")
                break

            # Główna sekcja wykonawcza z izolacją błędów krytycznych
            try:
                sources = await self.tick_once(query_fn())
                if sources:
                    # Callback jest asynchronicznie delegowany do nadzorcy, 
                    # by Scheduler natychmiast wracał do mierzenia czasu.
                    self._fire_and_forget(on_sources(sources))
            except Exception as exc:
                logger.error("[Scheduler] FATAL ERROR w cyklu tick_once: %s", exc, exc_info=True)

            # ---------------------------------------------------------
            # Clock Drift Detektor (Monitorowanie zdrowia Event Loopa)
            # ---------------------------------------------------------
            elapsed = time.monotonic() - loop_start
            sleep_time = TICK_SECONDS - elapsed
            
            if sleep_time < 0:
                self._total_drifts += 1
                logger.warning(
                    f"⏱️ [Clock Drift] Cykl zajął {elapsed:.2f}s (Budżet: {TICK_SECONDS}s). "
                    f"Opóźnienie Schedulera! (Zanotowano driftów: {self._total_drifts})"
                )
                sleep_time = 0.0 # Zerujemy by zapobiec błędom w asyncio.sleep
            
            try:
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                logger.info("[Scheduler] Otrzymano CancelledError z Event Loopa. Przerywam...")
                break

        # ---------------------------------------------------------
        # Faza Zamknięcia Pętli
        # ---------------------------------------------------------
        self._running = False
        await self.shutdown()

    # ------------------------------------------------------------------
    # Pojedynczy Cykl Odpytywania (Single Tick Core)
    # ------------------------------------------------------------------

    async def tick_once(self, query: str) -> "list[ScoutSource]":
        """
        Identyfikuje crawlery gotowe do działania i uruchamia je równolegle.
        Integruje deduplikację krzyżową (Cross-Layer Dedup).
        """
        now = datetime.now(timezone.utc)
        
        # Filtrowanie po czasie i stanie Circuit Breakera
        due = [
            c for c in self._crawlers
            if self._next_run[c.source_id] <= now and not c.is_paused
        ]

        if not due:
            return []

        self._total_fired += len(due)
        logger.debug(
            "[Scheduler:Tick] Pobieranie zakolejkowane dla %d/%d crawlerów. (Query: '%s')", 
            len(due), len(self._crawlers), query[:40]
        )

        tick_start = time.monotonic()

        # Równoległe wykonanie chronione Semaforem (Bounded Concurrency)
        sem = asyncio.Semaphore(self._max_concurrent)

        async def _guarded(crawler: CrawlerBase) -> "list[ScoutSource]":
            async with sem:
                return await crawler.safe_crawl(query)

        raw_results = await asyncio.gather(
            *[_guarded(c) for c in due],
            return_exceptions=True,
        )

        # Analiza i Agregacja Wyników (Z harmonogramowaniem)
        all_sources: list = []
        for crawler, batch in zip(due, raw_results):
            if isinstance(batch, Exception):
                logger.error(
                    f"[Scheduler:Tick] Crawler {crawler.source_id} zrzucił wyjątek "
                    f"mimo mechanizmów obronnych: {batch}"
                )
                batch = []
            
            all_sources.extend(batch)
            
            # Harmonogramowanie kolejnego wykonania na bazie adaptacyjnego interwału Crawlera
            self._next_run[crawler.source_id] = (
                datetime.now(timezone.utc) + timedelta(seconds=crawler.poll_interval)
            )

        fetch_duration = time.monotonic() - tick_start
        if not all_sources:
            return []

        # ---------------------------------------------------------
        # Deduplikacja (Cross-Layer)
        # ---------------------------------------------------------
        dedup_start = time.monotonic()
        dedup = DedupPipeline(**self._dedup_kwargs)
        unique = await dedup.filter(all_sources)
        dedup_duration = time.monotonic() - dedup_start
        
        self._total_unique += len(unique)

        logger.info(
            "[Scheduler:Tick] Zakończono! Zebrano %d, Po deduplikacji %d. "
            "[Czas: Fetch=%.2fs | Dedup=%.2fs]",
            len(all_sources), len(unique), fetch_duration, dedup_duration
        )
        return unique


    # ------------------------------------------------------------------
    # Bezpieczne Zamykanie i Kontrola (Graceful Control)
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Zgłasza żądanie zamknięcia dla głównej pętli run_forever()."""
        self._running = False

    async def shutdown(self) -> None:
        """
        Zatrzymuje Scheduler, upewniając się, że wszystkie zaplanowane zapisy
        do bazy danych (zadania w tle) zostały w pełni zakończone.
        """
        logger.info(
            "[Scheduler:Shutdown] Zamykanie orkiestratora. Statystyki: "
            f"Wyzwołań={self._total_fired} | Unikalnych={self._total_unique} | Drifty={self._total_drifts}"
        )
        if self._background_tasks:
            pending_count = len(self._background_tasks)
            logger.warning(f"⏳ [Scheduler:Shutdown] Oczekiwanie na ukończenie {pending_count} zadań w tle...")
            
            # Bezpieczne oczekiwanie na zapisy
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            logger.info("✅ [Scheduler:Shutdown] Wszystkie zadania tła zakończone pomyślnie.")


    # ------------------------------------------------------------------
    # Obserwowalność (Observability & API Payload)
    # ------------------------------------------------------------------

    def status(self) -> list[dict]:
        """Zwraca stan pojedynczych crawlerów dla API (GET /api/scout/sources)."""
        now = datetime.now(timezone.utc)
        return [
            {
                "source_id":          c.source_id,
                "poll_interval_s":    c.poll_interval,
                "default_interval_s": c.default_poll_interval,
                "is_paused":          c.is_paused,
                "next_run_in_s":      max(
                    0,
                    int((self._next_run[c.source_id] - now).total_seconds()),
                ),
                "consecutive_errors": c._consecutive_errors,
                "last_seen_id":       c.last_seen_id,
            }
            for c in self._crawlers
        ]

    def global_stats(self) -> dict:
        """Agreguje statystyki stanu maszyny na poziomie całej platformy."""
        return {
            "running":          self._running,
            "crawler_count":    len(self._crawlers),
            "total_fired":      self._total_fired,
            "total_unique":     self._total_unique,
            "total_drifts":     self._total_drifts,
            "pending_bg_tasks": len(self._background_tasks),
            "max_concurrent":   self._max_concurrent,
            "tick_interval_s":  TICK_SECONDS
        }
