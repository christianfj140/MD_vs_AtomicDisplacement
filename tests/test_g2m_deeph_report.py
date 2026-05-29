import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_report import (  # noqa: E402
    best_validation_summary,
    final_claim_report,
    generate_report,
    learning_curve_rows,
    pareto_report_rows,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_record(
    *,
    model: str = "graph2mat",
    config_id: str = "cfg",
    seed: int = 0,
    values: list[float] | None = None,
    gpu_hours: float | None = 1.0,
    peak_memory: float | None = 1024.0,
    status: str = "completed",
) -> dict:
    values = values or [0.4, 0.2, 0.3]
    return {
        "status": status,
        "model": model,
        "dataset_id": "dataset_a",
        "config_id": config_id,
        "common": {"seed": seed},
        "protocol_stage": "search",
        "early_stopping": {
            "validation_metric_name": "val_loss",
            "best_validation_value": min(values),
            "best_epoch": values.index(min(values)) + 1,
            "epochs_trained": len(values),
        },
        "learning_curve": [
            {
                "split": "validation",
                "epoch": index,
                "validation_metric": value,
                "wall_clock_seconds_cumulative": index * 10.0,
                "gpu_hours_cumulative": index * 0.1,
            }
            for index, value in enumerate(values, start=1)
        ],
        "telemetry": {
            "telemetry_status": "complete" if gpu_hours is not None and peak_memory is not None else "partial",
            "wall_clock_seconds_total": 30.0,
            "gpu_hours_total": gpu_hours,
            "gpu_hours_to_best_validation": 0.2,
            "wall_clock_seconds_to_best_validation": 20.0,
            "peak_gpu_memory_mb": peak_memory,
            "training_sample_count": 5,
            "matrix_block_count": 50,
            "epochs_trained": len(values),
            "best_validation_epoch": values.index(min(values)) + 1,
            "best_validation_value": min(values),
            "validation_metric": "val_loss",
            "telemetry_warnings": [] if gpu_hours is not None and peak_memory is not None else ["gpu telemetry partial"],
        },
    }


def final_statistics_payload(
    *,
    robust: bool = True,
    precision_winner: str = "graph2mat",
    compute_winner: str | None = "deeph",
    pareto_winner: str | None = "graph2mat",
) -> dict:
    return {
        "schema": "graph2mat_deeph_final_statistics_v1",
        "metric": "low_energy_rmse_eV",
        "mode": "min",
        "winner_decision": {
            "robust_claim_allowed": robust,
            "precision_winner": precision_winner if robust else None,
            "compute_winner": compute_winner if robust else None,
            "pareto_winner": pareto_winner if robust else None,
            "practical_pareto_winner": pareto_winner if robust else None,
            "gates_failed": [] if robust else ["diagnostic_only:deeph"],
            "diagnostic_only_reason": "" if robust else "deeph adapter equivalence not proven",
        },
    }


def gate_status_payload(*, robust: bool = True, claim_status: str | None = None) -> dict:
    return {
        "schema": "graph2mat_deeph_gate_status_v1",
        "claim_status": claim_status or ("robust_allowed" if robust else "invalid_equivalence"),
        "robust_claim_allowed": robust,
        "blockers": [] if robust else ["deeph_equivalence_proven: raw/global equivalence is not proven"],
        "warnings": [],
        "gates": [
            {
                "id": "telemetry_complete",
                "status": "pass" if robust else "fail",
                "message": "GPU-hours and peak GPU memory are present." if robust else "Missing telemetry fields",
            }
        ],
    }


class Graph2MatDeepHReportTests(unittest.TestCase):
    def test_parses_synthetic_per_epoch_metrics(self) -> None:
        rows = learning_curve_rows([run_record()], metric="val_loss")

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["epoch"], 2)
        self.assertEqual(rows[1]["validation_metric"], 0.2)
        self.assertEqual(rows[1]["samples_seen_cumulative"], 10)
        self.assertEqual(rows[1]["matrix_blocks_seen_cumulative"], 100)

    def test_computes_best_validation_point(self) -> None:
        records = [run_record(values=[0.5, 0.25, 0.35])]
        curves = learning_curve_rows(records, metric="val_loss")
        summary = best_validation_summary(records, curves, metric="val_loss", mode="min")

        self.assertEqual(summary[0]["best_epoch"], 2)
        self.assertEqual(summary[0]["best_validation_value"], 0.25)
        self.assertEqual(summary[0]["gpu_hours_to_best_validation"], 0.2)

    def test_computes_pareto_dominance(self) -> None:
        rows = [
            {
                "model": "graph2mat",
                "dataset_id": "d",
                "config_id": "fast_good",
                "best_validation_value": 0.1,
                "gpu_hours_total": 1.0,
            },
            {
                "model": "deeph",
                "dataset_id": "d",
                "config_id": "slow_bad",
                "best_validation_value": 0.2,
                "gpu_hours_total": 2.0,
            },
        ]

        pareto = pareto_report_rows(rows, mode="min")

        by_config = {row["config_id"]: row for row in pareto}
        self.assertFalse(by_config["fast_good"]["pareto_dominated"])
        self.assertTrue(by_config["slow_bad"]["pareto_dominated"])

    def test_missing_telemetry_is_explicit(self) -> None:
        record = run_record()
        record.pop("telemetry")
        curves = learning_curve_rows([record], metric="val_loss")
        summary = best_validation_summary([record], curves, metric="val_loss")

        self.assertEqual(summary[0]["telemetry_status"], "unavailable")
        self.assertIn("telemetry unavailable", summary[0]["telemetry_warnings"])

    def test_report_generation_from_old_artifact_without_learning_curve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = run_record()
            record.pop("learning_curve")
            write_json(root / "sweep" / "training_sweep_manifest.json", {"runs": [record]})

            manifest = generate_report(run_root=root, metric="val_loss")

            self.assertEqual(manifest["records_count"], 1)
            self.assertEqual(manifest["learning_curve_rows"], 0)
            self.assertEqual(manifest["best_validation_rows"], 1)
            self.assertTrue((root / "summary" / "report" / "best_validation_summary.csv").exists())
            self.assertIn("per-epoch learning curve unavailable", manifest["warnings"])

    def test_search_stage_report_ignores_test_metrics(self) -> None:
        record = run_record(values=[0.4, 0.2])
        record["learning_curve"] = [
            {"split": "test", "epoch": 1, "validation_metric": 0.001},
            {"split": "validation", "epoch": 1, "validation_metric": 0.4},
        ]

        rows = learning_curve_rows([record], metric="val_loss")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["validation_metric"], 0.4)

    def test_final_comparison_separates_accuracy_cost_and_pareto_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                run_record(model="graph2mat", config_id="g2m", values=[0.2, 0.1], gpu_hours=3.0),
                run_record(model="deeph", config_id="dh", values=[0.3, 0.25], gpu_hours=1.0),
            ]
            write_json(root / "sweep" / "training_sweep_manifest.json", {"runs": records})

            manifest = generate_report(run_root=root, metric="val_loss", compute_threshold=0.3)
            claims = {row["claim"]: row for row in manifest["final_comparison"]}

            self.assertEqual(claims["accuracy_winner"]["winner"], "graph2mat")
            self.assertEqual(claims["compute_winner"]["winner"], "deeph")
            self.assertIn("practical_pareto_winner", claims)

    def test_failed_gate_produces_no_winner_in_final_report(self) -> None:
        report = final_claim_report(
            metric="low_energy_rmse_eV",
            mode="min",
            final_statistics=final_statistics_payload(robust=True),
            gate_status=gate_status_payload(robust=False),
            final_statistics_path=Path("final_statistics.json"),
            gate_status_path=Path("gate_status.json"),
        )

        self.assertFalse(report["robust_claim_allowed"])
        self.assertIsNone(report["precision_winner"])
        self.assertIsNone(report["cost_winner"])
        self.assertIn("gate_check blocked robust claims", report["diagnostic_only_reason"])

    def test_passed_gate_and_final_stats_can_report_winner(self) -> None:
        report = final_claim_report(
            metric="low_energy_rmse_eV",
            mode="min",
            final_statistics=final_statistics_payload(robust=True),
            gate_status=gate_status_payload(robust=True),
            final_statistics_path=Path("final_statistics.json"),
            gate_status_path=Path("gate_status.json"),
        )

        self.assertTrue(report["robust_claim_allowed"])
        self.assertEqual(report["precision_winner"], "graph2mat")
        self.assertEqual(report["cost_winner"], "deeph")
        self.assertEqual(report["practical_pareto_winner"], "graph2mat")

    def test_common_h_mae_recommendation_cannot_override_failed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = run_record(model="deeph", config_id="dh", values=[0.05], gpu_hours=1.0)
            write_json(root / "sweep" / "training_sweep_manifest.json", {"runs": [record]})
            final_stats_path = root / "summary" / "final_statistics" / "final_statistics.json"
            gate_path = root / "summary" / "gate_status.json"
            write_json(final_stats_path, final_statistics_payload(robust=True, precision_winner="deeph"))
            write_json(gate_path, gate_status_payload(robust=False))

            manifest = generate_report(
                run_root=root,
                metric="low_energy_rmse_eV",
                final_statistics_path=final_stats_path,
                gate_status_path=gate_path,
            )

            self.assertFalse(manifest["robust_claim_allowed"])
            self.assertIsNone(manifest["final_report"]["precision_winner"])
            self.assertTrue((root / "summary" / "report" / "final_report.json").exists())
            self.assertTrue((root / "summary" / "report" / "final_report.md").exists())

    def test_missing_cost_telemetry_blocks_cost_winner(self) -> None:
        gate = gate_status_payload(robust=False, claim_status="invalid_telemetry")
        gate["gates"] = [{"id": "telemetry_complete", "status": "fail", "message": "Missing telemetry fields"}]
        report = final_claim_report(
            metric="low_energy_rmse_eV",
            mode="min",
            final_statistics=final_statistics_payload(robust=True, compute_winner="deeph"),
            gate_status=gate,
            final_statistics_path=Path("final_statistics.json"),
            gate_status_path=Path("gate_status.json"),
        )

        self.assertFalse(report["cost_claim_allowed"])
        self.assertIsNone(report["cost_winner"])
        self.assertEqual(report["claim_status"], "invalid_telemetry")

    def test_final_report_includes_required_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "sweep" / "training_sweep_manifest.json", {"runs": [run_record()]})
            final_stats_path = root / "summary" / "final_statistics" / "final_statistics.json"
            gate_path = root / "summary" / "gate_status.json"
            write_json(final_stats_path, final_statistics_payload())
            write_json(gate_path, gate_status_payload())

            manifest = generate_report(
                run_root=root,
                metric="low_energy_rmse_eV",
                final_statistics_path=final_stats_path,
                gate_status_path=gate_path,
            )
            final_report = json.loads((root / "summary" / "report" / "final_report.json").read_text(encoding="utf-8"))
            markdown = (root / "summary" / "report" / "final_report.md").read_text(encoding="utf-8")

            self.assertEqual(final_report["primary_final_metric"], "low_energy_rmse_eV")
            self.assertIn("supporting_metric_policy", final_report)
            self.assertIn("H-MAE/common metric summaries are supporting", markdown)
            self.assertIn("final_report_json", manifest["outputs"])


if __name__ == "__main__":
    unittest.main()
