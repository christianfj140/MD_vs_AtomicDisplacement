import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_test_blindness import (  # noqa: E402
    FINAL_TEST_STAGE,
    SEARCH_STAGE,
    assert_no_test_metrics_for_search,
    build_final_test_stage_manifest,
    build_search_stage_manifest,
    is_final_benchmark_mode,
    search_stage_record_fields,
    select_top_k_validation_only,
    validate_final_evaluation_inputs,
)


class Graph2MatDeepHTestBlindnessTests(unittest.TestCase):
    def test_final_benchmark_mode_detection(self) -> None:
        self.assertTrue(is_final_benchmark_mode({"benchmark_mode": "final_publication"}))
        self.assertTrue(is_final_benchmark_mode({"paper_ready": True}))
        self.assertTrue(
            is_final_benchmark_mode(
                {
                    "protocol": {
                        "final_test_policy": {
                            "policy": "locked_until_final",
                            "locked_during_search": True,
                        }
                    }
                }
            )
        )
        self.assertFalse(is_final_benchmark_mode({"benchmark_mode": "exploratory"}))

    def test_search_artifacts_do_not_require_test_metrics(self) -> None:
        validate_final_evaluation_inputs(
            selected_runs=[],
            metric_rows=[],
            stage=SEARCH_STAGE,
            metric="low_energy_rmse_eV",
        )

    def test_search_rejects_test_metric_rows(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Test metrics are locked"):
            assert_no_test_metrics_for_search(
                [{"model": "graph2mat", "config_id": "g2m_a", "metric_split": "test"}],
                stage=SEARCH_STAGE,
            )

    def test_top_k_selection_uses_validation_only(self) -> None:
        rows = [
            {
                "model": "graph2mat",
                "config_id": "g2m_a",
                "metric_split": "validation",
                "low_energy_rmse_eV_mean": 0.2,
            },
            {
                "model": "graph2mat",
                "config_id": "g2m_b",
                "metric_split": "validation",
                "low_energy_rmse_eV_mean": 0.1,
            },
            {
                "model": "deeph",
                "config_id": "dh_a",
                "metric_split": "validation",
                "low_energy_rmse_eV_mean": 0.3,
            },
        ]

        selected = select_top_k_validation_only(
            rows,
            metric="low_energy_rmse_eV",
            mode="min",
            k_per_model=1,
        )

        self.assertEqual(
            {(row["model"], row["config_id"]) for row in selected},
            {("graph2mat", "g2m_b"), ("deeph", "dh_a")},
        )

    def test_top_k_rejects_test_metrics_even_when_validation_exists(self) -> None:
        rows = [
            {
                "model": "graph2mat",
                "config_id": "g2m_a",
                "metric_split": "validation",
                "low_energy_rmse_eV_mean": 0.2,
            },
            {
                "model": "graph2mat",
                "config_id": "g2m_a",
                "metric_split": "test",
                "low_energy_rmse_eV_mean": 0.05,
            },
        ]

        with self.assertRaisesRegex(RuntimeError, "Test metrics are locked"):
            select_top_k_validation_only(
                rows,
                metric="low_energy_rmse_eV",
                mode="min",
                k_per_model=1,
            )

    def test_final_evaluation_requires_selected_runs_and_test_metrics(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "selected final runs"):
            validate_final_evaluation_inputs(
                selected_runs=[],
                metric_rows=[],
                stage=FINAL_TEST_STAGE,
                metric="low_energy_rmse_eV",
            )

        selected = [{"model": "graph2mat", "config_id": "g2m_a"}]
        with self.assertRaisesRegex(RuntimeError, "requires test metrics"):
            validate_final_evaluation_inputs(
                selected_runs=selected,
                metric_rows=[],
                stage=FINAL_TEST_STAGE,
                metric="low_energy_rmse_eV",
            )

        with self.assertRaisesRegex(RuntimeError, "missing test metrics"):
            validate_final_evaluation_inputs(
                selected_runs=selected,
                metric_rows=[
                    {
                        "model": "deeph",
                        "config_id": "dh_a",
                        "metric_split": "test",
                        "low_energy_rmse_eV": 0.1,
                    }
                ],
                stage=FINAL_TEST_STAGE,
                metric="low_energy_rmse_eV",
            )

        validate_final_evaluation_inputs(
            selected_runs=selected,
            metric_rows=[
                {
                    "model": "graph2mat",
                    "config_id": "g2m_a",
                    "metric_split": "test",
                    "low_energy_rmse_eV": 0.1,
                }
            ],
            stage=FINAL_TEST_STAGE,
            metric="low_energy_rmse_eV",
        )

    def test_search_stage_manifest_marks_final_test_locked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_search_stage_manifest(
                run_root=Path(tmp),
                summary={"status": "completed", "runs": [{"status": "completed"}], "failed_runs": []},
                payload={"benchmark_mode": "final_publication"},
            )

            self.assertEqual(manifest["protocol_stage"], SEARCH_STAGE)
            self.assertTrue(manifest["final_test_locked"])
            self.assertEqual(manifest["final_test_status"], "pending_selection")
            self.assertTrue(Path(manifest["path"]).exists())

    def test_final_test_manifest_requires_selected_test_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_final_test_stage_manifest(
                run_root=Path(tmp),
                selected_runs=[{"model": "graph2mat", "config_id": "g2m_a"}],
                metric_rows=[
                    {
                        "model": "graph2mat",
                        "config_id": "g2m_a",
                        "metric_split": "test",
                        "low_energy_rmse_eV_mean": 0.1,
                    }
                ],
                metric="low_energy_rmse_eV",
            )

            self.assertEqual(manifest["protocol_stage"], FINAL_TEST_STAGE)
            self.assertFalse(manifest["final_test_locked"])
            self.assertEqual(manifest["final_test_metric_rows"], 1)
            self.assertTrue(Path(manifest["path"]).exists())

    def test_search_stage_record_fields_are_explicit(self) -> None:
        fields = search_stage_record_fields()

        self.assertEqual(fields["protocol_stage"], SEARCH_STAGE)
        self.assertTrue(fields["test_metrics_locked"])
        self.assertEqual(fields["test_metrics_status"], "locked_until_final")


if __name__ == "__main__":
    unittest.main()
