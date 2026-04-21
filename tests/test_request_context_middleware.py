import logging
import os
import unittest

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/db")
os.environ.setdefault("ASYNC_DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.bootstrap import configure_middlewares


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


if __name__ == "__main__":
    unittest.main()
