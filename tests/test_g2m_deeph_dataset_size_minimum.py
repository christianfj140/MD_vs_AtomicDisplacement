import csv
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

minimum = importlib.import_module("g2m_deeph_dataset_size_minimum")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class DatasetSizeMinimumTests(unittest.TestCase):
    def test_n_min_abs_with_synthetic_rows(self) -> None:
        rows = [
            {"method": "graph2mat", "dataset_size_x": 10, "primary_metric_mev_mean": 25.0},
            {"method": "graph2mat", "dataset_size_x": 20, "primary_metric_mev_mean": 12.0},
            {"method": "graph2mat", "dataset_size_x": 40, "primary_metric_mev_mean": 8.0},
        ]

        self.assertEqual(minimum.n_min_abs(rows, 10.0), 40)

    def test_n_min_rel95_uses_best_observed_with_tolerance(self) -> None:
        rows = [
            {"method": "deeph", "dataset_size_x": 10, "primary_metric_mev_mean": 20.0},
            {"method": "deeph", "dataset_size_x": 30, "primary_metric_mev_mean": 10.4},
            {"method": "deeph", "dataset_size_x": 50, "primary_metric_mev_mean": 10.0},
        ]

        self.assertEqual(minimum.n_min_rel95(rows, 0.05), 30)

    def test_plateau_gain_uses_future_gain_fraction(self) -> None:
        rows = [
            {"method": "deeph", "dataset_size_x": 10, "primary_metric_mev_mean": 100.0},
            {"method": "deeph", "dataset_size_x": 20, "primary_metric_mev_mean": 60.0},
            {"method": "deeph", "dataset_size_x": 40, "primary_metric_mev_mean": 52.0},
            {"method": "deeph", "dataset_size_x": 80, "primary_metric_mev_mean": 50.0},
        ]

        self.assertEqual(minimum.n_min_plateau(rows, 0.05), 40)

    def test_ev_to_mev_conversion(self) -> None:
        raw = [
            {
                "model": "graph2mat",
                "dataset_size": 10,
                "config_id": "cfg",
                "epoch_label": "10 epochs",
                "h_mae_eV_mean": 0.012,
            }
        ]

        rows, warnings = minimum.normalize_rows(raw, primary_metric="h_mae_eV_mean", x_axis="n_total")

        self.assertEqual(warnings, [])
        self.assertEqual(rows[0]["primary_metric_unit"], "meV")
        self.assertAlmostEqual(rows[0]["primary_metric_mev"], 12.0)

    def test_fail_safe_if_metrics_missing(self) -> None:
        raw = [{"model": "deeph", "dataset_size": 10, "config_id": "cfg"}]

        rows, warnings = minimum.normalize_rows(raw, primary_metric="h_mae_eV_mean", x_axis="n_total")

        self.assertEqual(rows, [])
        self.assertTrue(any("missing_primary_metric:h_mae_eV_mean:deeph:cfg" in item for item in warnings))

    def test_analyze_reads_metric_scaling_rows_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "run_id": "r",
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "g10",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.020,
                    },
                    {
                        "run_id": "r",
                        "method": "graph2mat",
                        "dataset_size": 20,
                        "config_id": "g20",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.009,
                    },
                    {
                        "run_id": "r",
                        "method": "deeph",
                        "dataset_size": 10,
                        "config_id": "d10",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.030,
                    },
                ]
            }
            (root / "summary" / "ranking").mkdir(parents=True)
            (root / "summary" / "ranking" / "normalized_run_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            output_dir = base / "out"
            args = type(
                "Args",
                (),
                {
                    "run_root": [str(root)],
                    "output_dir": str(output_dir),
                    "primary_metric": "h_mae_eV_mean",
                    "threshold_mev": 10.0,
                    "relative_tolerance": 0.05,
                    "plateau_gain": 0.05,
                    "x_axis": "n_total",
                    "fit_models": "linear,inverse,power_law",
                },
            )()

            summary = minimum.analyze(args)

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["thresholds"]["graph2mat"]["N_min_abs"], 20)
            self.assertTrue((output_dir / "dataset_size_minimum_results.csv").exists())
            self.assertTrue((output_dir / "dataset_size_minimum_summary.json").exists())
            self.assertTrue((output_dir / "dataset_size_minimum_report.md").exists())

    def test_script_does_not_import_subprocess_or_build_compute_commands(self) -> None:
        source = (SCRIPTS_DIR / "g2m_deeph_dataset_size_minimum.py").read_text(encoding="utf-8")

        self.assertNotIn("import subprocess", source)
        self.assertNotIn("os.system", source)
        self.assertNotIn("Popen", source)
        self.assertIn("FORBIDDEN_COMPUTE_COMMANDS", source)


if __name__ == "__main__":
    unittest.main()
