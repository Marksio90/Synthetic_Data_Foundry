#!/usr/bin/env python3
"""
Phase C contract smoke checks for core API behavior.

Runs lightweight assertions against FastAPI TestClient without external services.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/db")
os.environ.setdefault("ASYNC_DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ADMIN_API_KEY", "ci-admin-key")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from api.main import app
from config.settings import settings


def main() -> None:
    # Force deterministic auth behavior for contract checks.
    settings.admin_api_key = "ci-admin-key"

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, f"/health expected 200, got {health.status_code}"
        assert health.json().get("status") == "ok", "/health payload mismatch"

        with_req_id = client.get("/health", headers={"X-Request-ID": "contract-123"})
        assert with_req_id.headers.get("X-Request-ID") == "contract-123", "request id not propagated"

        unauthorized = client.get("/api/pipeline/runs")
        assert unauthorized.status_code == 401, "pipeline runs should require X-API-Key"

        authorized_not_found = client.get(
            "/api/pipeline/status/not-existing-run",
            headers={"X-API-Key": "ci-admin-key"},
        )
        assert authorized_not_found.status_code == 404, "status contract mismatch for missing run"

    print("contract_smoke: OK")


if __name__ == "__main__":
    main()
