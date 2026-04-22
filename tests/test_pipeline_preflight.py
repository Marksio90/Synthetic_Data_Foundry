import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/db")
os.environ.setdefault("ASYNC_DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError

from api.routers.pipeline import _ensure_pipeline_db_ready, _get_calibration_chunks


class PipelinePreflightTests(unittest.TestCase):
    def test_preflight_passes_when_required_tables_exist(self) -> None:
        session = MagicMock()
        session.bind = object()

        inspector = MagicMock()
        inspector.has_table.side_effect = lambda table: table in {"source_documents", "directive_chunks"}

        with patch("api.routers.pipeline.inspect", return_value=inspector):
            _ensure_pipeline_db_ready(session)

        session.execute.assert_called_once()

    def test_preflight_returns_503_when_tables_are_missing(self) -> None:
        session = MagicMock()
        session.bind = object()

        inspector = MagicMock()
        inspector.has_table.return_value = False

        with patch("api.routers.pipeline.inspect", return_value=inspector):
            with self.assertRaises(HTTPException) as ctx:
                _ensure_pipeline_db_ready(session)

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail["error_code"], "pipeline_schema_missing")
        self.assertIn("source_documents", ctx.exception.detail["missing_tables"])
        self.assertIn("directive_chunks", ctx.exception.detail["missing_tables"])

    def test_get_calibration_chunks_returns_503_when_query_fails(self) -> None:
        session = MagicMock()
        session.scalars.side_effect = SQLAlchemyError("boom")

        with self.assertRaises(HTTPException) as ctx:
            _get_calibration_chunks(session, ["doc.pdf"])

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail["error_code"], "pipeline_db_query_failed")


if __name__ == "__main__":
    unittest.main()
