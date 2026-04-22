import os
import unittest

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/db")
os.environ.setdefault("ASYNC_DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from pydantic import ValidationError

from config.settings import Settings


class ServiceRoleSettingsTests(unittest.TestCase):
    def test_service_role_normalizes_case_and_whitespace(self) -> None:
        cfg = Settings(service_role=" Worker ")
        self.assertEqual(cfg.service_role, "worker")

    def test_service_role_rejects_unknown_value(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(service_role="scheduler")


if __name__ == "__main__":
    unittest.main()
