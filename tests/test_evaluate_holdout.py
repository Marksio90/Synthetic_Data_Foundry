import unittest
import os

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/db")
os.environ.setdefault("ASYNC_DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from training.evaluate import _build_holdout_pool


class EvaluateHoldoutSplitTests(unittest.TestCase):
    def test_build_holdout_pool_uses_every_fifth_record(self) -> None:
        records = [{"id": i} for i in range(12)]

        holdout = _build_holdout_pool(records, modulo=5)

        self.assertEqual([r["id"] for r in holdout], [0, 5, 10])

    def test_build_holdout_pool_falls_back_to_full_dataset_for_non_positive_modulo(self) -> None:
        records = [{"id": i} for i in range(3)]

        holdout = _build_holdout_pool(records, modulo=0)

        self.assertEqual([r["id"] for r in holdout], [0, 1, 2])

    def test_build_holdout_pool_returns_empty_for_empty_input(self) -> None:
        self.assertEqual(_build_holdout_pool([], modulo=5), [])


if __name__ == "__main__":
    unittest.main()
