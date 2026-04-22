import os
import unittest

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/db")
os.environ.setdefault("ASYNC_DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi import HTTPException

from api.security import create_ws_ticket, require_admin_api_key, verify_ws_ticket
from config.settings import settings


class RequireAdminApiKeyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_admin_key = settings.admin_api_key

    def tearDown(self) -> None:
        settings.admin_api_key = self._original_admin_key

    def test_returns_503_when_admin_key_not_configured(self) -> None:
        settings.admin_api_key = ""

        with self.assertRaises(HTTPException) as ctx:
            require_admin_api_key("any")

        self.assertEqual(ctx.exception.status_code, 503)

    def test_returns_401_when_missing_or_invalid_key(self) -> None:
        settings.admin_api_key = "super-secret"

        with self.assertRaises(HTTPException) as missing_ctx:
            require_admin_api_key(None)
        self.assertEqual(missing_ctx.exception.status_code, 401)

        with self.assertRaises(HTTPException) as invalid_ctx:
            require_admin_api_key("wrong")
        self.assertEqual(invalid_ctx.exception.status_code, 401)

    def test_passes_when_key_matches(self) -> None:
        settings.admin_api_key = "super-secret"

        # should not raise
        require_admin_api_key("super-secret")

    def test_ws_ticket_roundtrip_and_scope(self) -> None:
        settings.admin_api_key = "super-secret"

        ticket = create_ws_ticket("run-123", ttl_seconds=60)
        self.assertTrue(ticket)
        self.assertTrue(verify_ws_ticket(ticket, "run-123"))
        self.assertFalse(verify_ws_ticket(ticket, "run-xyz"))

    def test_ws_ticket_rejects_expired_or_malformed(self) -> None:
        settings.admin_api_key = "super-secret"

        ticket = create_ws_ticket("run-123", ttl_seconds=60)
        expired = f"1:{ticket.split(':', 1)[1]}"
        self.assertFalse(verify_ws_ticket(expired, "run-123"))
        self.assertFalse(verify_ws_ticket("not-a-ticket", "run-123"))


if __name__ == "__main__":
    unittest.main()
