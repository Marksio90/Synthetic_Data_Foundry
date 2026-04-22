import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/db")
os.environ.setdefault("ASYNC_DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import training
from config.settings import settings


class TrainingCancelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_admin_key = settings.admin_api_key
        settings.admin_api_key = "test-admin-key"
        training._TRAIN_PROCS.clear()
        training._TRAIN_TASKS.clear()

    def tearDown(self) -> None:
        settings.admin_api_key = self._original_admin_key
        training._TRAIN_PROCS.clear()
        training._TRAIN_TASKS.clear()

    def _client(self) -> TestClient:
        app = FastAPI()
        app.include_router(training.router, prefix="/api/training")
        return TestClient(app)

    def test_cancel_training_returns_cancelled(self) -> None:
        run_id = "train-abc123"
        training.runs.create(run_id, "train:test")
        training.runs.update(run_id, status="running")

        proc = MagicMock()
        proc.wait = AsyncMock(return_value=0)
        training._TRAIN_PROCS[run_id] = proc
        task = MagicMock()
        task.done.return_value = False
        training._TRAIN_TASKS[run_id] = task

        with self._client() as client:
            response = client.post(
                f"/api/training/cancel/{run_id}",
                headers={"X-API-Key": "test-admin-key"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "cancelled")
        proc.terminate.assert_called_once()
        task.cancel.assert_called_once()

    def test_cancel_training_returns_409_when_not_running(self) -> None:
        run_id = "train-not-running"
        training.runs.create(run_id, "train:test")
        training.runs.update(run_id, status="done")

        with self._client() as client:
            response = client.post(
                f"/api/training/cancel/{run_id}",
                headers={"X-API-Key": "test-admin-key"},
            )

        self.assertEqual(response.status_code, 409)

    def test_cancel_training_kills_process_after_timeout(self) -> None:
        run_id = "train-timeout"
        training.runs.create(run_id, "train:test")
        training.runs.update(run_id, status="running")

        proc = MagicMock()
        proc.wait = AsyncMock(side_effect=[asyncio.TimeoutError(), 0])
        training._TRAIN_PROCS[run_id] = proc
        task = MagicMock()
        task.done.return_value = False
        training._TRAIN_TASKS[run_id] = task

        with self._client() as client:
            response = client.post(
                f"/api/training/cancel/{run_id}",
                headers={"X-API-Key": "test-admin-key"},
            )

        self.assertEqual(response.status_code, 200)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        self.assertEqual(proc.wait.await_count, 2)

    def test_cancel_training_is_idempotent_for_second_call(self) -> None:
        run_id = "train-double-cancel"
        training.runs.create(run_id, "train:test")
        training.runs.update(run_id, status="running")

        proc = MagicMock()
        proc.wait = AsyncMock(return_value=0)
        training._TRAIN_PROCS[run_id] = proc
        task = MagicMock()
        task.done.return_value = False
        training._TRAIN_TASKS[run_id] = task

        with self._client() as client:
            first = client.post(
                f"/api/training/cancel/{run_id}",
                headers={"X-API-Key": "test-admin-key"},
            )
            second = client.post(
                f"/api/training/cancel/{run_id}",
                headers={"X-API-Key": "test-admin-key"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)


if __name__ == "__main__":
    unittest.main()
