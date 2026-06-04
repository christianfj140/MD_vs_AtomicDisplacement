import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from deeph_prediction_adapter import EQUIVALENCE_PROVEN_RAW_GLOBAL, EQUIVALENCE_STATUS_UNPROVEN  # noqa: E402
from g2m_deeph_final_stats import (  # noqa: E402
    aggregate_final_seed_metrics,
    bootstrap_ci,
    decide_winners,
    final_statistics_report,
    protocol_violations,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def final_row(
    model: str,
    seed: int,
    value: float,
    *,
    gpu_hours: float = 1.0,
    peak_memory: float = 1000.0,
    proven_deeph: bool = True,
    stage: str = "final_test",
    split: str = "test",
    **overrides,
) -> dict:
    row = {
        "status": "completed",
        "model": model,
        "dataset_id": "dataset_a",
        "config_id": f"{model}_cfg",
        "seed": seed,
        "protocol_stage": stage,
        "metric_split": split,
        "low_energy_rmse_eV_mean": value,
        "telemetry": {
            "gpu_hours_total": gpu_hours,
            "peak_gpu_memory_mb": peak_memory,
            "samples_per_second": 10.0 / gpu_hours,
            "matrix_blocks_per_second": 100.0 / gpu_hours,
        },
        "per_system_metrics": [
            {"sample_id": "s1", "low_energy_rmse_eV_mean": value * 0.9},
            {"sample_id": "s2", "low_energy_rmse_eV_mean": value * 1.1},
        ],
    }
    if model == "deeph":
        row["adapter_equivalence_status"] = (
            EQUIVALENCE_PROVEN_RAW_GLOBAL if proven_deeph else "diagnostic_local_frame_only"
        )
        row["comparability_status"] = "valid" if proven_deeph else "diagnostic_only"
        row["diagnostic_only"] = not proven_deeph
    row.update(overrides)
    return row


class Graph2MatDeepHFinalStatsTests(unittest.TestCase):
    def test_mean_std_aggregation_and_compute_summary(self) -> None:
        rows = [
            final_row("graph2mat", 0, 0.10, gpu_hours=2.0),
            final_row("graph2mat", 1, 0.20, gpu_hours=4.0),
            final_row("graph2mat", 2, 0.15, gpu_hours=3.0),
        ]

        summary = aggregate_final_seed_metrics(rows, metric="low_energy_rmse_eV", expected_seeds=[0, 1, 2])

        self.assertEqual(summary[0]["n_seeds_completed"], 3)
        self.assertAlmostEqual(summary[0]["mean"], 0.15)
        self.assertAlmostEqual(summary[0]["gpu_hours_mean"], 3.0)
        self.assertEqual(summary[0]["missing_seeds"], [])

    def test_incomplete_seeds_block_robust_winner(self) -> None:
        summaries = aggregate_final_seed_metrics(
            [
                final_row("graph2mat", 0, 0.10),
                final_row("graph2mat", 1, 0.11),
                final_row("deeph", 0, 0.09),
                final_row("deeph", 1, 0.10),
            ],
            metric="low_energy_rmse_eV",
            expected_seeds=[0, 1, 2],
        )

        decision = decide_winners(
            summaries,
            expected_seeds=[0, 1, 2],
            min_final_seeds=3,
            mode="min",
        )

        self.assertFalse(decision["robust_claim_allowed"])
        self.assertIn("incomplete_final_seeds:graph2mat/dataset_a/graph2mat_cfg", decision["gates_failed"])
        self.assertIn("incomplete_final_seeds:deeph/dataset_a/deeph_cfg", decision["gates_failed"])

    def test_bootstrap_ci_when_per_system_metrics_exist(self) -> None:
        ci = bootstrap_ci([0.1, 0.2, 0.3, 0.4], iterations=200, seed=7)

        self.assertEqual(ci["method"], "bootstrap_per_system_mean")
        self.assertIsNotNone(ci["low"])
        self.assertIsNotNone(ci["high"])
        self.assertLessEqual(ci["low"], ci["high"])

    def test_missing_per_system_metrics_is_explicit(self) -> None:
        row = final_row("graph2mat", 0, 0.10)
        row.pop("per_system_metrics")

        summary = aggregate_final_seed_metrics([row], metric="low_energy_rmse_eV")

        self.assertEqual(summary[0]["bootstrap_ci"]["method"], "unavailable")
        self.assertIn("per-system metrics unavailable", summary[0]["bootstrap_ci"]["reason"])

    def test_deeph_diagnostic_only_gate_blocks_robust_claim(self) -> None:
        summaries = aggregate_final_seed_metrics(
            [
                final_row("graph2mat", 0, 0.20),
                final_row("graph2mat", 1, 0.21),
                final_row("graph2mat", 2, 0.22),
                final_row("deeph", 0, 0.10, proven_deeph=False),
                final_row("deeph", 1, 0.11, proven_deeph=False),
                final_row("deeph", 2, 0.12, proven_deeph=False),
            ],
            metric="low_energy_rmse_eV",
            expected_seeds=[0, 1, 2],
        )

        decision = decide_winners(summaries, expected_seeds=[0, 1, 2], mode="min")

        self.assertFalse(decision["robust_claim_allowed"])
        self.assertIn("diagnostic_only:deeph/dataset_a/deeph_cfg", decision["gates_failed"])
        self.assertIn("deeph adapter equivalence not proven", decision["diagnostic_only_reason"])

    def test_unproven_formal_equivalence_status_blocks_robust_claim(self) -> None:
        rows = [
            final_row("graph2mat", 0, 0.20),
            final_row("graph2mat", 1, 0.21),
            final_row("graph2mat", 2, 0.22),
            final_row("deeph", 0, 0.10, proven_deeph=True, equivalence_status=EQUIVALENCE_STATUS_UNPROVEN),
            final_row("deeph", 1, 0.11, proven_deeph=True, equivalence_status=EQUIVALENCE_STATUS_UNPROVEN),
            final_row("deeph", 2, 0.12, proven_deeph=True, equivalence_status=EQUIVALENCE_STATUS_UNPROVEN),
        ]

        summaries = aggregate_final_seed_metrics(rows, metric="low_energy_rmse_eV", expected_seeds=[0, 1, 2])
        decision = decide_winners(summaries, expected_seeds=[0, 1, 2], mode="min")

        self.assertFalse(decision["robust_claim_allowed"])
        self.assertIn("diagnostic_only:deeph/dataset_a/deeph_cfg", decision["gates_failed"])
        self.assertIn("equivalence=unproven", decision["diagnostic_only_reason"])

    def test_top_k_configs_are_not_mixed_in_final_seed_summary(self) -> None:
        rows = [
            final_row("graph2mat", 0, 0.10, selected_config_id="g_a", config_id="g_a_seed0"),
            final_row("graph2mat", 1, 0.10, selected_config_id="g_a", config_id="g_a_seed1"),
            final_row("graph2mat", 2, 0.10, selected_config_id="g_a", config_id="g_a_seed2"),
            final_row("graph2mat", 0, 0.40, selected_config_id="g_b", config_id="g_b_seed0"),
            final_row("graph2mat", 1, 0.40, selected_config_id="g_b", config_id="g_b_seed1"),
            final_row("graph2mat", 2, 0.40, selected_config_id="g_b", config_id="g_b_seed2"),
            final_row("deeph", 0, 0.30, selected_config_id="d_a", config_id="d_a_seed0"),
            final_row("deeph", 1, 0.30, selected_config_id="d_a", config_id="d_a_seed1"),
            final_row("deeph", 2, 0.30, selected_config_id="d_a", config_id="d_a_seed2"),
        ]

        summaries = aggregate_final_seed_metrics(rows, metric="low_energy_rmse_eV", expected_seeds=[0, 1, 2])
        by_config = {row["selected_config_id"]: row for row in summaries}
        decision = decide_winners(summaries, expected_seeds=[0, 1, 2], mode="min")

        self.assertEqual(set(by_config), {"g_a", "g_b", "d_a"})
        self.assertAlmostEqual(by_config["g_a"]["mean"], 0.10)
        self.assertAlmostEqual(by_config["g_b"]["mean"], 0.40)
        self.assertTrue(decision["robust_claim_allowed"])
        self.assertEqual(decision["precision_winner"], "graph2mat")
        self.assertEqual(decision["dataset_decisions"][0]["winner_config_id"], "g_a")

    def test_winner_tolerance_and_ci_rules(self) -> None:
        summaries = aggregate_final_seed_metrics(
            [
                final_row("graph2mat", 0, 0.10),
                final_row("graph2mat", 1, 0.10),
                final_row("graph2mat", 2, 0.10),
                final_row("deeph", 0, 0.30),
                final_row("deeph", 1, 0.30),
                final_row("deeph", 2, 0.30),
            ],
            metric="low_energy_rmse_eV",
            expected_seeds=[0, 1, 2],
        )

        decision = decide_winners(summaries, expected_seeds=[0, 1, 2], mode="min", tolerance=0.01)

        self.assertTrue(decision["robust_claim_allowed"])
        self.assertEqual(decision["precision_winner"], "graph2mat")
        self.assertTrue(decision["ci_rule_passed"])

    def test_compute_winner_threshold_logic(self) -> None:
        summaries = aggregate_final_seed_metrics(
            [
                final_row("graph2mat", 0, 0.10, gpu_hours=5.0),
                final_row("graph2mat", 1, 0.10, gpu_hours=5.0),
                final_row("graph2mat", 2, 0.10, gpu_hours=5.0),
                final_row("deeph", 0, 0.20, gpu_hours=1.0),
                final_row("deeph", 1, 0.20, gpu_hours=1.0),
                final_row("deeph", 2, 0.20, gpu_hours=1.0),
            ],
            metric="low_energy_rmse_eV",
            expected_seeds=[0, 1, 2],
        )

        decision = decide_winners(
            summaries,
            expected_seeds=[0, 1, 2],
            mode="min",
            compute_accuracy_threshold=0.25,
        )

        self.assertEqual(decision["compute_winner"], "deeph")

    def test_protocol_violation_when_test_metrics_appear_in_wrong_stage(self) -> None:
        rows = [final_row("graph2mat", 0, 0.10, stage="search", split="test")]

        violations = protocol_violations(rows)

        self.assertTrue(violations)
        self.assertIn("outside final_test", violations[0])

    def test_final_statistics_report_serializes_outputs_and_protocol_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                final_row("graph2mat", 0, 0.10, stage="search", split="test"),
                final_row("deeph", 0, 0.20),
            ]
            write_json(root / "summary" / "ranking" / "normalized_run_metrics.json", {"rows": rows})

            report = final_statistics_report(
                run_root=root,
                metric="low_energy_rmse_eV",
                expected_seeds=[0],
                min_final_seeds=1,
            )

            self.assertFalse(report["winner_decision"]["robust_claim_allowed"])
            self.assertIn("protocol_violation_test_metrics_outside_final_stage", report["winner_decision"]["gates_failed"])
            self.assertTrue((root / "summary" / "final_statistics" / "winner_decision.json").exists())


if __name__ == "__main__":
    unittest.main()
