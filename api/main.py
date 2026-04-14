"""
api/main.py — Foundry Studio FastAPI application.

Services:
  /api/documents/*  — PDF upload, listing, deletion
  /api/pipeline/*   — AutoPilot run, status, log, WebSocket
  /api/samples/*    — Q&A dataset browsing
  /health           — liveness probe

Start locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import documents, pipeline, samples

app = FastAPI(
    title="Foundry Studio API",
    version="1.0.0",
    description="AutoPilot backend for Synthetic Data Foundry",
)

# Allow Streamlit UI (running on port 8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(pipeline.router,  prefix="/api/pipeline",  tags=["pipeline"])
app.include_router(samples.router,   prefix="/api/samples",   tags=["samples"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "foundry-api"}
