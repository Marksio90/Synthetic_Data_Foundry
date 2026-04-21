"""
agents/crawlers/base.py — Abstract base class for all Gap Scout source crawlers (ENTERPRISE EDITION).

Każdy dedykowany crawler dziedziczy po CrawlerBase i implementuje crawl().
Infrastruktura (Out-of-the-box):
  • Global Connection Pool — współdzielony klient HTTP z limitem gniazd TCP (wysoka wydajność).
  • Circuit Breaker      — 5 błędów z rzędu → 5-minutowa pauza systemu → automatyczne wznowienie.
  • HTTP Retry (Jitter)  — Exponential backoff dla błędów sieciowych i 5xx (via Tenacity).
  • Adaptive Polling     — 3x brak nowych elementów → podwojenie interwału (limitowane górną granicą).
  • HTTP Caching         — Pełne wsparcie ETag, If-None-Match, Last-Modified.
  • Rate-Limit Guard     — Logowanie krytyczne dla X-RateLimit-Remaining < 5.
  • Prometheus Hooks     — Eksport metryk do systemu monitoringu (Histogramy / Liczniki błędów).
"""

from __future__ import annotations

import logging
import random
import time
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, ClassVar, Optional, Dict, Any, Tuple

import httpx

if TYPE_CHECKING:
    from agents.topic_scout import ScoutSource

logger = logging.getLogger("foundry.agents.crawlers.base")

# ---------------------------------------------------------------------------
# Zarządzanie Pakietami Opcjonalnymi (Tenacity)
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
    logger.warning("Biblioteka 'tenacity' nie jest zainstalowana — HTTP Retries są WYŁĄCZONE. Zalecona instalacja dla Enterprise.")

# ---------------------------------------------------------------------------
# Globalna Pula Połączeń (Connection Pooling - Ochrona przed TCP Leak)
# ---------------------------------------------------------------------------
# Zamiast tworzyć nowego klienta per instancja, używamy globalnego singletonu,
# by oszczędzać gniazda serwera i optymalizować Handshake TLS.
_GLOBAL_HTTP_CLIENT: Optional[httpx.AsyncClient] = None

def _get_global_http_client() -> httpx.AsyncClient:
    global _GLOBAL_HTTP_CLIENT
    if _GLOBAL_HTTP_CLIENT is None:
        limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
        _GLOBAL_HTTP_CLIENT = httpx.AsyncClient(
            timeout=25.0, # Zwiększono limit dla stabilności crawlerów
            follow_redirects=True,
            limits=limits,
        )
    return _GLOBAL_HTTP_CLIENT


# ---------------------------------------------------------------------------
# Klasa Bazowa Crawlera (CrawlerBase)
# ---------------------------------------------------------------------------
class CrawlerBase(ABC):
    """
    Abstrakcyjna klasa bazowa dla wszystkich crawlerów Gap Scout.

    Zmienne do nadpisania w klasach dziedziczących:
        source_id               Unikalny ID używany w metrykach i logach (Wymagane)
        default_poll_interval   Sekundy pomiędzy zapytaniami (Domyślnie 300s = 5 min)
        max_poll_interval       Górny limit dla Adaptive Backoff (Domyślnie 3600s = 1 godz)
    """

    source_id: ClassVar[str] = "base"
    default_poll_interval: ClassVar[int] = 300       # 5 min
    max_poll_interval: ClassVar[int] = 3_600         # 1 godz

    _CB_THRESHOLD: ClassVar[int] = 5                 # Błędy wyzwalające Circuit Breaker
    _CB_PAUSE_SECONDS: ClassVar[int] = 300           # Czas otwarcia obwodu (Pauza 5 min)
    _KNOWN_BOT_BLOCKED_SOURCES: ClassVar[set[str]] = {"ssrn", "oecd", "epo", "esma"}

    # Domyślne nagłówki Anti-Ban
    _USER_AGENTS = [
        "FoundryScout/2.0 (research-tool; non-commercial)",
        "Mozilla/5.0 (compatible; FoundryScoutBot/2.0; +http://foundry.example.com/bot)"
    ]

    def __init__(self) -> None:
        self.poll_interval: int = self.default_poll_interval
        self.last_seen_id: str = ""          # Znak wodny deduplikacji (aktualizowany przez .crawl())
        
        self._consecutive_errors: int = 0
        self._no_new_count: int = 0          # Licznik pustych odpytań (dla Adaptive Polling)
        self._paused_until: Optional[datetime] = None
        
        self._etag: str = ""
        self._last_modified: str = ""
        
        # Pobieramy podpięcie pod globalną pulę
        self._http = _get_global_http_client()

    # ------------------------------------------------------------------
    # Interfejs Abstrakcyjny
    # ------------------------------------------------------------------
    @abstractmethod
    async def crawl(self, query: str) -> "list[ScoutSource]":
        """
        Pobiera nowe źródła dla zadanego zapytania (query).

        Wdrożenie w subklasie MUST:
          - Używać `await self._fetch(...)` do połączeń sieciowych (Caching + Retry).
          - Zaktualizować `self.last_seen_id` po udanym pobraniu.
          - Zwracać TYLKO nowo zobaczony kontent.
        """
        pass

    # ------------------------------------------------------------------
    # Główny Punkt Wejścia (Circuit Breaker Wrapper)
    # ------------------------------------------------------------------
    async def safe_crawl(self, query: str) -> "list[ScoutSource]":
        """
        Wywoluje crawl() owinięte w Circuit Breaker, Adaptive Polling i Telemetrię.
        Nigdy nie rzuca wyjątków – w przypadku awarii zwraca bezpieczną pustą listę [].
        """
        if self._is_paused():
            logger.debug(
                "[%s] Circuit Breaker AKTYWNY (Zablokowano do %s) — pomijam crawl.",
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
    # HTTP Helper Core (Caching, Retry, User-Agent)
    # ------------------------------------------------------------------
    async def _fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        use_cache_headers: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Główne narzędzie I/O wykonujące połączenie z zaawansowaną polityką Exponential Backoff.
        Zrzuca błędy 4xx (Client Error) bez retry'u, chroniąc przed banem na IP.
        """
        # Losowy User-Agent chroni przed agresywnym firewall'em na crawlery
        headers = {
            "User-Agent": random.choice(self._USER_AGENTS),
            **(self.cache_headers() if use_cache_headers else {}),
            **kwargs.pop("headers", {}),
        }

        async def _do() -> httpx.Response:
            resp = await self._http.request(method, url, headers=headers, **kwargs)
            if resp.status_code in (429, 503):
                retry_after = self._parse_retry_after(resp.headers.get("retry-after", ""))
                if retry_after > 0:
                    logger.info(
                        "[%s] PROVIDER_RATE_LIMITED — retry-after=%ss, adaptive pause before retry.",
                        self.source_id, retry_after,
                        extra={"crawler": self.source_id, "error_code": "PROVIDER_RATE_LIMITED", "retry_after_seconds": retry_after},
                    )
                    await asyncio.sleep(min(retry_after, 10))
                resp.raise_for_status()
            if 400 <= resp.status_code < 500:
                logger.debug(f"[{self.source_id}] Klient HTTP zablokowany ({resp.status_code}) — zaniechanie powtórzeń.")
                return resp
            resp.raise_for_status()
            return resp

        if _HAS_TENACITY:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                # Jitter: Dodanie max 2s odchyłu zapobiega potężnym pętlom wielu crawlerów na raz
                wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 2),
                retry=retry_if_exception_type(
                    (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException, httpx.ReadError)
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
    # ETag / Last-Modified Caching (Optymalizacja Transferu)
    # ------------------------------------------------------------------
    def cache_headers(self) -> Dict[str, str]:
        """Buduje nagłówki warunkowe redukujące koszt łącza (HTTP 304 Not Modified)."""
        h: Dict[str, str] = {}
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
    # Rate-Limit Guard (Bariera Ochronna API)
    # ------------------------------------------------------------------
    def _check_rate_limit(self, headers: httpx.Headers) -> None:
        remaining = headers.get("x-ratelimit-remaining", "")
        if remaining:
            try:
                if int(remaining) < 5:
                    logger.warning(
                        "[%s] PROVIDER_RATE_LIMITED (low remaining): Pozostało %s zapytań.",
                        self.source_id, remaining,
                        extra={"crawler": self.source_id, "rl_remaining": remaining, "error_code": "PROVIDER_RATE_LIMITED"},
                    )
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Mechanika Circuit Breaker i Adaptive Polling
    # ------------------------------------------------------------------
    def _is_paused(self) -> bool:
        if self._paused_until is None:
            return False
        if datetime.now(timezone.utc) >= self._paused_until:
            self._paused_until = None
            logger.info(
                "[%s] 🟢 Circuit Breaker ZAMKNIĘTY — Wznawiam skanowanie.",
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
                        "[%s] Brak nowych treści (×3). Wydłużam interwał: %ds → %ds",
                        self.source_id, self.poll_interval, new_iv,
                        extra={"crawler": self.source_id, "poll_interval": new_iv},
                    )
                self.poll_interval = new_iv
                self._no_new_count = 0
                
        _record_fetch(self.source_id, elapsed, success=True)

    def _on_error(self, exc: Exception, elapsed: float) -> None:
        error_code, severity, details = self._classify_error(exc)
        self._consecutive_errors += 1
        payload = {
            "crawler": self.source_id,
            "error": str(exc),
            "error_code": error_code,
            "error_kind": details,
        }
        if severity == "info":
            logger.info(
                "[%s] %s (%d/%d): %s",
                self.source_id, error_code, self._consecutive_errors, self._CB_THRESHOLD, details,
                extra=payload,
            )
        else:
            logger.warning(
                "[%s] %s (%d/%d): %s",
                self.source_id, error_code, self._consecutive_errors, self._CB_THRESHOLD, details,
                extra=payload,
            )
        if self._consecutive_errors >= self._CB_THRESHOLD:
            self._paused_until = datetime.now(timezone.utc) + timedelta(seconds=self._CB_PAUSE_SECONDS)
            self._consecutive_errors = 0
            logger.error(
                "[%s] 🔴 CIRCUIT BREAKER OTWARTY — Zawieszam operacje na %ds do %s.",
                self.source_id, self._CB_PAUSE_SECONDS, self._paused_until.isoformat(),
                extra={
                    "crawler": self.source_id,
                    "event": "circuit_breaker_open",
                    "paused_until": self._paused_until.isoformat(),
                },
            )
        _record_fetch(self.source_id, elapsed, success=False)

    @staticmethod
    def _parse_retry_after(raw: str) -> int:
        try:
            return max(0, int(raw.strip()))
        except (TypeError, ValueError, AttributeError):
            return 0

    def _classify_error(self, exc: Exception) -> Tuple[str, str, str]:
        if isinstance(exc, httpx.TimeoutException):
            return "PROVIDER_TIMEOUT", "warning", "request timeout"
        if isinstance(exc, httpx.ConnectError):
            return "PROVIDER_DNS_FAILURE", "warning", "network/dns connectivity failure"
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 403 and self.source_id == "ieee":
                return "PROVIDER_AUTH_FORBIDDEN", "info", "IEEE 403 — sprawdź scope IEEE API key"
            if status in (403, 404) and self.source_id in self._KNOWN_BOT_BLOCKED_SOURCES:
                return "PROVIDER_BOT_BLOCKED", "info", f"known-hostile source status={status}"
            if status == 429:
                return "PROVIDER_RATE_LIMITED", "warning", "HTTP 429 Too Many Requests"
            if status == 503:
                return "PROVIDER_RATE_LIMITED", "warning", "HTTP 503 Service Unavailable"
            return "PROVIDER_HTTP_ERROR", "warning", f"HTTP status={status}"
        return "PROVIDER_UNKNOWN_ERROR", "warning", type(exc).__name__

    # ------------------------------------------------------------------
    # Interfejsy Cyklu Życia (Async Context Managers)
    # ------------------------------------------------------------------
    @property
    def is_paused(self) -> bool:
        """Stan Circuit Breakera (Tryb Read-Only)."""
        return self._paused_until is not None and datetime.now(timezone.utc) < self._paused_until

    async def aclose(self) -> None:
        """Uwaga: W wersji PRO nie zamykamy globalnego klienta z poziomu sub-crawlera."""
        pass 

    def __repr__(self) -> str:
        status = "PAUSED" if self.is_paused else f"poll={self.poll_interval}s"
        return f"<{self.__class__.__name__} source_id={self.source_id!r} {status}>"


# ---------------------------------------------------------------------------
# Integracja Prometheus (Zabezpieczony Shim)
# ---------------------------------------------------------------------------
def _record_fetch(source_id: str, elapsed: float, *, success: bool) -> None:
    """Nigdy nie wysadzi platformy w przypadku błędu monitoringu."""
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
