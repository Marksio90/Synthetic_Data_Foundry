"""
api/main.py — Foundry Studio FastAPI application (Enterprise Grade).

Services:
  /api/documents/* — PDF upload, listing, deletion
  /api/pipeline/* — AutoPilot run, status, log, WebSocket
  /api/samples/* — Q&A dataset browsing
  /api/training/* — Hardware inspect, quality gate, SFT/DPO training, export
  /api/chatbot/* — Chat with Ollama model, evaluation runs
  /api/scout/* — Gap Scout: automated knowledge-gap discovery
  /health           — liveness probe (alias)
  /health/live      — liveness probe
  /health/ready     — readiness probe (DB + Ollama checks)
  /metrics          — prometheus metrics export

Start locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.bootstrap import (
    BackgroundTaskManager,
    configure_middlewares,
    create_lifespan,
    create_scheduler,
    register_exception_handlers,
    register_routers,
    register_system_routes,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Inicjalizacja modułów bootstrap
# ---------------------------------------------------------------------------
logger = setup_logging()
task_manager = BackgroundTaskManager(logger)
scheduler = create_scheduler(logger)

# ---------------------------------------------------------------------------
# Rate limiter (slowapi — Redis-backed in production, in-memory fallback)
# ---------------------------------------------------------------------------
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore
    from slowapi.errors import RateLimitExceeded  # type: ignore
    from slowapi.util import get_remote_address  # type: ignore

    limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
    _SLOWAPI_AVAILABLE = True
except ImportError:
    limiter = None  # type: ignore[assignment]
    _SLOWAPI_AVAILABLE = False
    logger.warning("slowapi not installed — rate limiting disabled. Run: pip install slowapi")


# ---------------------------------------------------------------------------
# Inicjalizacja Aplikacji FastAPI (Core App)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Synthetic Data Foundry Studio API",
    version="2.0.0-PRO",
    description="Potężny backend orkiestrujący dla AutoPilota, Treningu DPO/SFT, oraz Gap Scouta.",
    lifespan=create_lifespan(logger, task_manager, scheduler),
    contact={
        "name": "Data Foundry Team",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)

if _SLOWAPI_AVAILABLE and limiter is not None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

configure_middlewares(app, logger)
register_exception_handlers(app, logger)
register_routers(app)
register_system_routes(app)
