"""
api/bootstrap.py — modular bootstrap for FastAPI application wiring.

Extracts lifecycle, middleware and router registration from api/main.py,
so main entrypoint remains focused on app composition.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional, Set

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
from starlette.middleware.base import BaseHTTPMiddleware

from api.monitoring import get_metrics_payload, is_available as metrics_available
from api.errors import ApiError
from api.routers import chatbot, documents, pipeline, samples, scout, training
from api.routers import websub as websub_router
from config.settings import settings


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO)
        if hasattr(settings, "log_level")
        else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    return logging.getLogger("foundry.api")


class BackgroundTaskManager:
    """Safe async task registry used for graceful shutdown."""

    def __init__(self, logger: logging.Logger):
        self._active_tasks: Set[asyncio.Task] = set()
        self._logger = logger

    def add_task(self, coro: Any, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        self._logger.debug("Rozpoczęto zadanie w tle: %s", name)
        return task

    async def cancel_all(self) -> None:
        if not self._active_tasks:
            return
        self._logger.info("Anulowanie %d aktywnych zadań tła...", len(self._active_tasks))
        for task in self._active_tasks:
            task.cancel()
        await asyncio.gather(*self._active_tasks, return_exceptions=True)
        self._logger.info("Zadania tła zostały bezpiecznie zamknięte.")


class LeaderLock:
    """
    Cross-replica singleton guard backed by PostgreSQL advisory lock.
    Keeps a dedicated DB connection open for the process lifetime to retain the lock.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._engine = None
        self._conn: Optional[Connection] = None
        self._lock_id = int(settings.scheduler_leader_lock_id)
        self.is_leader = False

    def acquire(self) -> bool:
        try:
            self._engine = create_engine(
                settings.database_url,
                pool_pre_ping=True,
                pool_size=1,
                max_overflow=0,
            )
            self._conn = self._engine.connect()
            acquired = bool(
                self._conn.execute(
                    text("SELECT pg_try_advisory_lock(:lock_id)"),
                    {"lock_id": self._lock_id},
                ).scalar()
            )
            self.is_leader = acquired
            if acquired:
                self._logger.info("Leader lock acquired (id=%s): singleton jobs enabled.", self._lock_id)
            else:
                self._logger.info("Leader lock busy (id=%s): singleton jobs disabled in this replica.", self._lock_id)
            return acquired
        except Exception as exc:
            self._logger.warning(
                "Could not acquire leader lock id=%s (%s). Running without singleton jobs.",
                self._lock_id,
                exc,
            )
            self.close()
            return False

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._engine is not None:
            try:
                self._engine.dispose()
            except Exception:
                pass
            self._engine = None
        self.is_leader = False


try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore
except ImportError:
    AsyncIOScheduler = None  # type: ignore[misc,assignment]


def create_scheduler(logger: logging.Logger):
    if AsyncIOScheduler is None:
        logger.warning("Biblioteka 'apscheduler' nie jest zainstalowana — cykliczny Gap Scout wyłączony.")
        return None
    return AsyncIOScheduler(timezone="UTC")


async def run_scheduled_scout(logger: logging.Logger) -> None:
    """Background job: runs Gap Scout and persists results in in-memory state."""
    from agents.topic_scout import run_scout
    from api.state import scouts

    scout_id = uuid.uuid4().hex
    scouts.create(scout_id)
    scouts.append_log(scout_id, "[Scheduler] Rozpoczęto cykliczny Gap Scout")
    scouts.update(scout_id, status="running")

    try:
        async def _cb(msg: str) -> None:
            scouts.append_log(scout_id, f"[Scout] {msg}")

        topics = await run_scout(progress_callback=_cb)
        scouts.add_topics(scout_id, topics)
        scouts.update(scout_id, status="done", topics_found=len(topics))
        scouts.append_log(scout_id, f"[Scheduler] ✅ Zakończono — znaleziono {len(topics)} nowych tematów.")
        logger.info("Cykliczny Gap Scout zakończony sukcesem: %d tematów.", len(topics))
    except Exception as exc:  # pragma: no cover - defensive runtime path
        scouts.update(scout_id, status="error", error=str(exc))
        logger.error("Błąd podczas działania cyklicznego Gap Scout: %s", exc, exc_info=True)


def create_lifespan(logger: logging.Logger, task_manager: BackgroundTaskManager, scheduler):
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info("🚀 Inicjalizacja środowiska Synthetic Data Foundry...")
        leader_lock = LeaderLock(logger)
        leader_ready = leader_lock.acquire()

        if scheduler is not None and leader_ready:
            async def _scheduled_job() -> None:
                await run_scheduled_scout(logger)

            scheduler.add_job(
                _scheduled_job,
                "interval",
                hours=1,
                id="hourly_scout",
                replace_existing=True,
                max_instances=1,
            )
            scheduler.start()
            logger.info(
                "APScheduler aktywny — zadanie 'hourly_scout' zaplanowane. "
                "Pierwszy automatyczny skan za 1 godzinę; "
                "manualny start przez przycisk na frontendzie."
            )
        elif scheduler is not None:
            logger.info("APScheduler pominięty — ta replika nie jest leaderem singleton zadań.")

        callback_url = getattr(settings, "scout_webhook_callback_url", "")
        webhook_secret = getattr(settings, "scout_webhook_secret", "")

        if callback_url and leader_ready:
            from agents.crawlers.websub import WebSubSubscriber

            websub = WebSubSubscriber.instance()
            task_manager.add_task(
                websub.subscribe_all(
                    callback_base_url=callback_url,
                    secret=webhook_secret,
                    lease_seconds=86400,
                ),
                name="websub_subscription_manager",
            )
            logger.info("WebSub: Nasłuchiwanie uruchomione pod adresem %s", callback_url)
        elif callback_url:
            logger.info("WebSub pominięty — ta replika nie jest leaderem singleton zadań.")
        else:
            logger.info("WebSub Tier 1 wyłączony — ustaw SCOUT_WEBHOOK_CALLBACK_URL, aby aktywować.")

        yield

        logger.info("🛑 Rozpoczynam procedurę Graceful Shutdown...")
        await task_manager.cancel_all()

        if scheduler is not None and scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("APScheduler zatrzymany.")

        from agents.crawlers.websub import WebSubSubscriber as _WS
        await _WS.instance().aclose()

        from agents.topic_scout import _HTTP as _scout_http
        await _scout_http.aclose()

        leader_lock.close()
        logger.info("Środowisko bezpiecznie wyłączone.")

    return lifespan


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Injects request id and request processing time into response headers."""

    def __init__(self, app: FastAPI, logger: logging.Logger):
        super().__init__(app)
        self._logger = logger

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.perf_counter()

        self._logger.debug("--> [%s] %s %s", request_id, request.method, request.url.path)
        response = await call_next(request)

        process_time = time.perf_counter() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Sec"] = f"{process_time:.3f}"

        self._logger.debug(
            "<-- [%s] %s %s - Status: %s (%.3fs)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            process_time,
        )
        return response


def configure_middlewares(app: FastAPI, logger: logging.Logger) -> None:
    app.add_middleware(RequestContextMiddleware, logger=logger)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    allowed_origins = settings.cors_origins or ["http://localhost:3000", "http://localhost:8501"]
    allow_credentials = "*" not in allowed_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


def register_exception_handlers(app: FastAPI, logger: logging.Logger) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        req_id = getattr(request.state, "request_id", request.headers.get("X-Request-ID", "unknown"))
        logger.warning(
            "[%s] API error (%s) on %s: %s",
            req_id,
            exc.error_code,
            request.url.path,
            exc.message,
        )
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload(req_id))

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", request.headers.get("X-Request-ID", "unknown"))
        logger.error(
            "[%s] Nieobsłużony błąd serwera (HTTP 500) pod adresem %s: %s",
            req_id,
            request.url.path,
            str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "Wystąpił nieoczekiwany błąd serwera. Skontaktuj się z administratorem.",
                "request_id": req_id,
            },
        )


def register_routers(app: FastAPI) -> None:
    app.include_router(documents.router, prefix="/api/documents", tags=["Documents (Ingestion & Files)"])
    app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline (AutoPilot & Orkiestracja)"])
    app.include_router(samples.router, prefix="/api/samples", tags=["Samples (Dataset Q&A)"])
    app.include_router(training.router, prefix="/api/training", tags=["Training (SFT/DPO & Eksport)"])
    app.include_router(chatbot.router, prefix="/api/chatbot", tags=["Chatbot (Lokalne modele)"])
    app.include_router(scout.router, prefix="/api/scout", tags=["Gap Scout (Wykrywanie luk)"])
    app.include_router(websub_router.router, prefix="/api/scout", tags=["WebSub (Real-time feeds)"])


def register_system_routes(app: FastAPI) -> None:
    @app.get("/health", tags=["System"])
    def health() -> dict:
        return {"status": "ok", "service": "foundry-api-pro", "timestamp": time.time()}

    @app.get("/health/live", tags=["System"])
    def health_live() -> dict:
        """Kubernetes liveness probe — confirms the process is alive."""
        return {"status": "ok", "service": "foundry-api-pro", "timestamp": time.time()}

    @app.get("/health/ready", tags=["System"])
    async def health_ready() -> JSONResponse:
        """Kubernetes readiness probe — verifies DB and optional services are reachable."""
        import sqlalchemy
        from api.db import _engine

        checks: dict = {}
        healthy = True

        # database
        try:
            with _engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            checks["database"] = {"status": "ok"}
        except Exception as exc:
            checks["database"] = {"status": "error", "detail": str(exc)}
            healthy = False

        # ollama — optional, degraded state does not block readiness
        ollama_url = getattr(settings, "ollama_url", "")
        if ollama_url:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{ollama_url}/api/tags")
                checks["ollama"] = {"status": "ok" if resp.status_code == 200 else "degraded"}
            except Exception:
                checks["ollama"] = {"status": "degraded"}

        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "ok" if healthy else "error",
                "service": "foundry-api-pro",
                "timestamp": time.time(),
                "checks": checks,
            },
        )

    @app.get("/health/dependencies", tags=["System"])
    async def health_dependencies() -> JSONResponse:
        """
        Diagnostic endpoint for external/service dependencies.
        Returns 200 when core dependencies are ready, otherwise 503/424.
        """
        import sqlalchemy
        from api.db import _engine

        checks: dict[str, dict] = {}
        missing: list[str] = []
        status_code = 200

        try:
            with _engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
                ext_rows = conn.execute(
                    sqlalchemy.text("SELECT extname FROM pg_extension WHERE extname IN ('vector')")
                ).fetchall()
            checks["database"] = {"status": "ok"}
            checks["pgvector"] = {"status": "ok" if any(r[0] == "vector" for r in ext_rows) else "missing"}
            if checks["pgvector"]["status"] != "ok":
                missing.append("pgvector")
                status_code = 424
        except Exception as exc:
            checks["database"] = {"status": "error", "detail": str(exc)}
            checks["pgvector"] = {"status": "unknown"}
            missing.extend(["database", "pgvector"])
            status_code = 503

        ollama_url = getattr(settings, "ollama_url", "")
        if ollama_url:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{ollama_url}/api/tags")
                checks["ollama"] = {"status": "ok" if resp.status_code == 200 else "degraded"}
            except Exception:
                checks["ollama"] = {"status": "degraded"}
        else:
            checks["ollama"] = {"status": "not_configured"}

        checks["api_keys"] = {
            "openai_api_key_configured": bool(getattr(settings, "openai_api_key", "").strip()),
            "admin_api_key_configured": bool(getattr(settings, "admin_api_key", "").strip()),
        }

        payload = {
            "status": "ok" if status_code == 200 else "degraded",
            "service": "foundry-api-pro",
            "timestamp": time.time(),
            "checks": checks,
            "missing_dependencies": missing,
            "error_code": "dependencies_missing" if status_code in (424, 503) else None,
        }
        return JSONResponse(status_code=status_code, content=payload)

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        from fastapi import HTTPException
        from fastapi.responses import Response

        if not metrics_available():
            raise HTTPException(
                status_code=501,
                detail="Moduł prometheus-client nie jest zainstalowany. Wykonaj: pip install prometheus-client",
            )
        payload, content_type = get_metrics_payload()
        return Response(content=payload, media_type=content_type)
