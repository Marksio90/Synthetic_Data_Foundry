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

from fastapi import FastAPI

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

configure_middlewares(app, logger)
register_exception_handlers(app, logger)
register_routers(app)
register_system_routes(app)
