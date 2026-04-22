import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/db")
os.environ.setdefault("ASYNC_DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("SCOUT_STATE_DB_ENABLED", "false")

from api.state import ScoutManager


class ScoutStateTests(unittest.TestCase):
    def test_add_single_topic_updates_run_and_index(self) -> None:
        manager = ScoutManager()
        scout = manager.create("scout-1")

        topic = SimpleNamespace(
            topic_id="topic-1",
            title="Topic",
            summary="Summary",
            score=0.7,
            recency_score=0.8,
            llm_uncertainty=0.2,
            source_count=3,
            social_signal=0.4,
            sources=[
                SimpleNamespace(
                    url="https://example.com",
                    title="Example",
                    published_at="2026-04-22T00:00:00+00:00",
                    source_type="arxiv",
                    verified=True,
                )
            ],
            domains=["ai"],
            discovered_at="2026-04-22T00:00:00+00:00",
        )

        manager.add_single_topic("scout-1", topic)

        self.assertEqual(scout.topics_found, 1)
        self.assertIsNotNone(manager.get_topic("topic-1"))
        self.assertEqual(len(manager.latest_topics()), 1)


if __name__ == "__main__":
    unittest.main()
