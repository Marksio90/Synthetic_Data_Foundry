"""
api/main.py — Foundry Studio FastAPI application (Enterprise Grade).

Services:
  /api/documents/* — PDF upload, listing, deletion
  /api/pipeline/* — AutoPilot run, status, log, WebSocket
  /api/samples/* — Q&A dataset browsing
  /api/training/* — Hardware inspect, quality gate, SFT/DPO training, export
  /api/chatbot/* — Chat with Ollama model, evaluation runs
  /api/scout/* — Gap Scout: automated knowledge-gap discovery
  /health           — liveness probe
  /metrics          — prometheus metrics export

Start locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Callable, Set, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.routers import chatbot, documents, pipeline, samples, scout, training
from api.routers import websub as websub_router
from api.monitoring import get_metrics_payload, is_available as metrics_available
from config.settings import settings

# ---------------------------------------------------------------------------
# Konfiguracja Logowania i Obserwowalności (Observability)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO) if hasattr(settings, 'log_level') else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("foundry.api")

# ---------------------------------------------------------------------------
# Zarządzanie zadaniami w tle (Graceful Shutdown & GC Leak Prevention)
# ---------------------------------------------------------------------------
class BackgroundTaskManager:
    """
    Bezpieczny menedżer zadań asynchronicznych.
    Chroni procesy (np. WebSub, AutoPilot, Scout) przed nagłym uśmierceniem
    przez Garbage Collector i pozwala na ich poprawne zamknięcie.
    """
    def __init__(self):
        self._active_tasks: Set[asyncio.Task] = set()

    def add_task(self, coro: Any, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        logger.debug(f"Rozpoczęto zadanie w tle: {name}")
        return task

    async def cancel_all(self):
        if not self._active_tasks:
            return
        logger.info(f"Anulowanie {len(self._active_tasks)} aktywnych zadań tła...")
        for task in self._active_tasks:
            task.cancel()
        await asyncio.gather(*self._active_tasks, return_exceptions=True)
        logger.info("Zadania tła zostały bezpiecznie zamknięte.")

task_manager = BackgroundTaskManager()

# ---------------------------------------------------------------------------
# APScheduler — Hourly Gap Scout
# ---------------------------------------------------------------------------
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore
    _scheduler: AsyncIOScheduler | None = AsyncIOScheduler(timezone="UTC")
except ImportError:
    _scheduler = None
    logger.warning("Biblioteka 'apscheduler' nie jest zainstalowana — cykliczny Gap Scout wyłączony.")


async def _scheduled_scout() -> None:
    """Background job: uruchamia Gap Scout i zapisuje wyniki w bazie stanu."""
    from agents.topic_scout import run_scout
    from api.state import scouts

    scout_id = uuid.uuid4().hex
    scouts.create(scout_id)
    scouts.append_log(scout_id, "[Scheduler] Rozpoczęto cykliczny Gap Scout")
    scouts.update(scout_id, status="running")
    
    try:
        async def _cb(msg: str) -> None:
            scouts.append_log(scout_id, f"[Scout] {msg}")

        # Wykonanie głównej logiki skauta
        topics = await run_scout(progress_callback=_cb)
        scouts.add_topics(scout_id, topics)
        scouts.update(scout_id, status="done", topics_found=len(topics))
        scouts.append_log(scout_id, f"[Scheduler] ✅ Zakończono — znaleziono {len(topics)} nowych tematów.")
        logger.info("Cykliczny Gap Scout zakończony sukcesem: %d tematów.", len(topics))
    except Exception as exc:
        scouts.update(scout_id, status="error", error=str(exc))
        logger.error("Błąd podczas działania cyklicznego Gap Scout: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Zarządzanie Cyklem Życia Aplikacji (Lifespan)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Orkiestracja startu i wyłączania serwisów API."""
    logger.info("🚀 Inicjalizacja środowiska Synthetic Data Foundry...")

    # 1. Startup: Scheduler (Gap Scout)
    if _scheduler is not None:
        _scheduler.add_job(
            _scheduled_scout,
            "interval",
            hours=1,
            id="hourly_scout",
            replace_existing=True,
            max_instances=1,
        )
        _scheduler.start()
        logger.info("APScheduler aktywny — zadanie 'hourly_scout' zaplanowane.")
        
        # Opóźniony start pierwszego uruchomienia skauta, aby nie blokować startu API
        async def _delayed_scout() -> None:
            await asyncio.sleep(10.0) # Zwiększono do 10s dla stabilności bazy
            await _scheduled_scout()
        
        task_manager.add_task(_delayed_scout(), name="delayed_initial_scout")

    # 2. Startup: WebSub
    _callback_url = getattr(settings, "scout_webhook_callback_url", "")
    _ws_secret    = getattr(settings, "scout_webhook_secret", "")
    
    if _callback_url:
        from agents.crawlers.websub import WebSubSubscriber
        _websub = WebSubSubscriber.instance()
        task_manager.add_task(
            _websub.subscribe_all(
                callback_base_url=_callback_url,
                secret=_ws_secret,
                lease_seconds=86400,
            ),
            name="websub_subscription_manager"
        )
        logger.info("WebSub: Nasłuchiwanie uruchomione pod adresem %s", _callback_url)
    else:
        logger.info("WebSub Tier 1 wyłączony — ustaw SCOUT_WEBHOOK_CALLBACK_URL, aby aktywować.")

    yield  # --- Aplikacja działa tutaj ---

    logger.info("🛑 Rozpoczynam procedurę Graceful Shutdown...")

    # 1. Shutdown: Zadania w tle
    await task_manager.cancel_all()

    # 2. Shutdown: Scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("APScheduler zatrzymany.")

    # 3. Shutdown: Klienci HTTP (WebSub i TopicScout)
    from agents.crawlers.websub import WebSubSubscriber as _WS
    await _WS.instance().aclose()
    
    from agents.topic_scout import _HTTP as _scout_http
    await _scout_http.aclose()
    
    logger.info("Środowisko bezpiecznie wyłączone.")


# ---------------------------------------------------------------------------
# Inicjalizacja Aplikacji FastAPI (Core App)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Synthetic Data Foundry Studio API",
    version="2.0.0-PRO",
    description="Potężny backend orkiestrujący dla AutoPilota, Treningu DPO/SFT, oraz Gap Scouta.",
    lifespan=lifespan,
    contact={
        "name": "Data Foundry Team",
    },
    docs_url="/docs",
    redoc_url="/redoc"
)


# ---------------------------------------------------------------------------
# Middlewares (Warstwa Bezpieczeństwa, Wydajności i Obserwowalności)
# ---------------------------------------------------------------------------

# 1. Middleware: Śledzenie zapytań (Request ID) i czas wykonania
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        
        # Wstrzykiwanie do kontekstu loggera można dodać przy użyciu contextvars, 
        # tu proste logowanie wejścia/wyjścia.
        logger.debug(f"--> [{request_id}] {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        process_time = time.perf_counter() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Sec"] = f"{process_time:.3f}"
        
        logger.debug(f"<-- [{request_id}] {request.method} {request.url.path} - Status: {response.status_code} ({process_time:.3f}s)")
        return response

app.add_middleware(RequestContextMiddleware)

# 2. Middleware: Kompresja danych (niezbędna przy zwracaniu ogromnych JSONL)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 3. Middleware: CORS (Cross-Origin Resource Sharing)
# Zalecenie: W produkcji podmień ALLOWED_ORIGINS na konkretne domeny z settings.
ALLOWED_ORIGINS = getattr(settings, "cors_origins", ["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Globalne zarządzanie wyjątkami (Global Exception Handling)
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Przechwytuje wszystkie nieobsłużone wyjątki.
    Zapobiega zwracaniu wrażliwych danych (Traceback) do frontendu, dbając o bezpieczeństwo.
    """
    req_id = request.headers.get("X-Request-ID", "unknown")
    logger.error(f"[{req_id}] Nieobsłużony błąd serwera (HTTP 500) pod adresem {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Wystąpił nieoczekiwany błąd serwera. Skontaktuj się z administratorem.",
            "request_id": req_id
        },
    )


# ---------------------------------------------------------------------------
# Podłączanie Modułów (Routers)
# ---------------------------------------------------------------------------
app.include_router(documents.router, prefix="/api/documents", tags=["Documents (Ingestion & Files)"])
app.include_router(pipeline.router,  prefix="/api/pipeline",  tags=["Pipeline (AutoPilot & Orkiestracja)"])
app.include_router(samples.router,   prefix="/api/samples",   tags=["Samples (Dataset Q&A)"])
app.include_router(training.router,  prefix="/api/training",  tags=["Training (SFT/DPO & Eksport)"])
app.include_router(chatbot.router,   prefix="/api/chatbot",   tags=["Chatbot (Lokalne modele)"])
app.include_router(scout.router,     prefix="/api/scout",     tags=["Gap Scout (Wykrywanie luk)"])
app.include_router(websub_router.router, prefix="/api/scout", tags=["WebSub (Real-time feeds)"])


# ---------------------------------------------------------------------------
# Główne endpointy systemowe
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health() -> dict:
    """Prosta sonda liveness wykorzystywana przez Docker/Kubernetes."""
    return {
        "status": "ok", 
        "service": "foundry-api-pro", 
        "timestamp": time.time()
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Eksport statystyk i metryk dla systemów monitorujących (np. Prometheus)."""
    from fastapi.responses import Response
    if not metrics_available():
        from fastapi import HTTPException
        raise HTTPException(
            status_code=501,
            detail="Moduł prometheus-client nie jest zainstalowany. Wykonaj: pip install prometheus-client",
        )
    payload, content_type = get_metrics_payload()
    return Response(content=payload, media_type=content_type)
