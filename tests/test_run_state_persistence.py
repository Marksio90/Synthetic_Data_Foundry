import json
import tempfile
import unittest
from pathlib import Path

from api.state import RunManager


class RunStatePersistenceTests(unittest.TestCase):
    def test_run_manager_persists_and_restores_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "runs.json"

            manager = RunManager(snapshot_path=snapshot)
            rec = manager.create("run-1", "batch-1")
            rec.analysis = {"domain": "test"}
            manager.append_log("run-1", "hello")
            manager.update("run-1", status="done", progress_pct=100)

            self.assertTrue(snapshot.exists())
            payload = json.loads(snapshot.read_text(encoding="utf-8"))
            self.assertEqual(payload["runs"][0]["run_id"], "run-1")

            restored = RunManager(snapshot_path=snapshot)
            rec2 = restored.get("run-1")
            self.assertIsNotNone(rec2)
            assert rec2 is not None
            self.assertEqual(rec2.batch_id, "batch-1")
            self.assertEqual(rec2.status, "done")
            self.assertIn("hello", rec2.log_lines)


if __name__ == "__main__":
    unittest.main()
