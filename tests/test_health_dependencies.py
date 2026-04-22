import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/db")
os.environ.setdefault("ASYNC_DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.bootstrap import register_system_routes
from config.settings import settings


class HealthDependenciesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_ollama_url = settings.ollama_url
        settings.ollama_url = ""  # disable outbound network calls in tests

    def tearDown(self) -> None:
        settings.ollama_url = self._original_ollama_url

    def _app(self) -> FastAPI:
        app = FastAPI()
        register_system_routes(app)
        return app

    def test_returns_503_when_database_is_unavailable(self) -> None:
        engine = MagicMock()
        engine.connect.side_effect = RuntimeError("db down")

        with patch("api.db._engine", engine):
            with TestClient(self._app()) as client:
                response = client.get("/health/dependencies")

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["error_code"], "dependencies_missing")
        self.assertIn("database", payload["missing_dependencies"])

    def test_returns_424_when_pgvector_is_missing(self) -> None:
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn

        first_result = MagicMock()   # SELECT 1
        second_result = MagicMock()  # extension query
        second_result.fetchall.return_value = []
        conn.execute.side_effect = [first_result, second_result]

        with patch("api.db._engine", engine):
            with TestClient(self._app()) as client:
                response = client.get("/health/dependencies")

        self.assertEqual(response.status_code, 424)
        payload = response.json()
        self.assertEqual(payload["error_code"], "dependencies_missing")
        self.assertIn("pgvector", payload["missing_dependencies"])


if __name__ == "__main__":
    unittest.main()
