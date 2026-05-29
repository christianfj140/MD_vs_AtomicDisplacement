import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_budget import BudgetTracker, record_gpu_hours, write_budget_summary  # noqa: E402


class Graph2MatDeepHBudgetTests(unittest.TestCase):
    def test_equal_n_trials_reserves_and_exhausts_per_model(self) -> None:
        tracker = BudgetTracker({"mode": "equal_n_trials", "n_trials_per_model": 2})
        record = {"model": "graph2mat", "config_id": "g2m_a"}

        self.assertTrue(tracker.can_schedule(record))
        tracker.reserve(record)
        self.assertTrue(tracker.can_schedule(record))
        tracker.reserve(record)
        self.assertFalse(tracker.can_schedule(record))
        self.assertIn("n_trials_per_model=2", tracker.summary()["budget_exhaustion_reason"]["graph2mat"])

    def test_equal_gpu_hours_accounting_and_exhaustion(self) -> None:
        tracker = BudgetTracker({"mode": "equal_gpu_hours_per_model", "gpu_hours_per_model": 0.15})
        first = {"model": "graph2mat", "config_id": "g2m_a", "status": "completed", "telemetry": {"gpu_hours_total": 0.1}}
        second = {"model": "graph2mat", "config_id": "g2m_b", "status": "completed", "telemetry": {"gpu_hours_total": 0.06}}

        tracker.reserve(first)
        tracker.add_completed(first)
        self.assertTrue(tracker.can_schedule(second))
        tracker.reserve(second)
        tracker.add_completed(second)

        summary = tracker.summary()
        self.assertAlmostEqual(summary["consumed_gpu_hours_by_model"]["graph2mat"], 0.16)
        self.assertFalse(tracker.can_schedule({"model": "graph2mat", "config_id": "g2m_c"}))
        skipped = tracker.skip_for_budget({"model": "graph2mat", "config_id": "g2m_c"})
        self.assertEqual(skipped["status"], "skipped_budget_exhausted")
        self.assertEqual(tracker.summary()["skipped_trials_by_model"]["graph2mat"], 1)

    def test_missing_gpu_hours_fails_equal_gpu_hours(self) -> None:
        tracker = BudgetTracker({"mode": "equal_gpu_hours_per_model", "gpu_hours_per_model": 1.0})

        with self.assertRaisesRegex(RuntimeError, "Missing gpu_hours_total"):
            tracker.add_completed({"model": "deeph", "config_id": "dh_a", "status": "completed"})

        self.assertEqual(tracker.summary()["budget_accounting_status"], "failed")

    def test_record_gpu_hours_can_read_telemetry_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            telemetry_path = Path(tmp) / "telemetry.json"
            telemetry_path.write_text(json.dumps({"gpu_hours_total": 2.5}), encoding="utf-8")

            self.assertEqual(record_gpu_hours({"telemetry_path": str(telemetry_path)}), 2.5)

    def test_resume_completed_values_are_not_double_counted_when_deduped_by_caller(self) -> None:
        completed_by_key = {
            "graph2mat|d|a": {"model": "graph2mat", "config_id": "a", "status": "completed", "telemetry": {"gpu_hours_total": 0.2}},
            "graph2mat|d|b": {"model": "graph2mat", "config_id": "b", "status": "completed", "telemetry": {"gpu_hours_total": 0.3}},
        }
        tracker = BudgetTracker({"mode": "equal_gpu_hours_per_model", "gpu_hours_per_model": 1.0})

        tracker.add_completed_many(list(completed_by_key.values()), source="resume_manifest")

        summary = tracker.summary()
        self.assertEqual(summary["completed_trials_by_model"]["graph2mat"], 2)
        self.assertAlmostEqual(summary["consumed_gpu_hours_by_model"]["graph2mat"], 0.5)

    def test_budget_summary_serializes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = BudgetTracker({"mode": "equal_n_trials", "n_trials_per_model": 1})
            path = Path(tmp) / "budget_summary.json"

            write_budget_summary(path, tracker.summary())

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "graph2mat_deeph_search_budget_v1")
            self.assertEqual(payload["budget_policy"]["mode"], "equal_n_trials")


if __name__ == "__main__":
    unittest.main()
