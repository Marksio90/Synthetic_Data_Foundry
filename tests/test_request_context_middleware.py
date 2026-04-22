import logging
import os
import unittest

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/db")
os.environ.setdefault("ASYNC_DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.bootstrap import configure_middlewares, register_exception_handlers
from api.errors import ServiceUnavailableError


class RequestContextMiddlewareTests(unittest.TestCase):
    def test_adds_request_id_and_process_time_headers(self) -> None:
        app = FastAPI()
        configure_middlewares(app, logging.getLogger("test"))

        @app.get("/ping")
        def ping() -> dict:
            return {"ok": True}

        with TestClient(app) as client:
            response = client.get("/ping")

        self.assertEqual(response.status_code, 200)
        self.assertIn("X-Request-ID", response.headers)
        self.assertIn("X-Process-Time-Sec", response.headers)
        self.assertNotEqual(response.headers["X-Request-ID"], "")

    def test_api_error_handler_reuses_correlation_id(self) -> None:
        app = FastAPI()
        configure_middlewares(app, logging.getLogger("test"))
        register_exception_handlers(app, logging.getLogger("test"))

        @app.get("/boom")
        def boom() -> dict:
            raise ServiceUnavailableError(
                error_code="dependency_down",
                message="Dependency is unavailable",
            )

        with TestClient(app) as client:
            response = client.get("/boom", headers={"X-Request-ID": "req-123"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "dependency_down")
        self.assertEqual(response.json()["request_id"], "req-123")
        self.assertEqual(response.headers["X-Request-ID"], "req-123")


if __name__ == "__main__":
    unittest.main()
