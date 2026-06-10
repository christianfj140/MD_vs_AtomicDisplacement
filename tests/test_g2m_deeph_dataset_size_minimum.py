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
    def test_mean_by_method_size_averages_same_x_across_sources(self) -> None:
        rows = [
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev_mean": 20.0,
                "source_run_root": "/sweep_a",
                "sweep_label": "sweep_a",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev_mean": 30.0,
                "source_run_root": "/sweep_b",
                "sweep_label": "sweep_b",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 20,
                "primary_metric_mev_mean": 12.0,
                "source_run_root": "/sweep_a",
                "sweep_label": "sweep_a",
            },
        ]

        aggregated = minimum.mean_by_method_size(rows)

        self.assertEqual(len(aggregated), 2)
        by_size = {int(row["dataset_size_x"]): row for row in aggregated}
        self.assertAlmostEqual(by_size[10]["primary_metric_mev_mean"], 25.0)
        self.assertAlmostEqual(by_size[20]["primary_metric_mev_mean"], 12.0)
        self.assertTrue(by_size[10]["is_aggregated_mean"])

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


class DatasetSizeMinimumUiApiTests(unittest.TestCase):
    def test_iter_metrics_paths_avoids_recursive_results_scan(self) -> None:
        import importlib.util
        import time

        spec = importlib.util.spec_from_file_location(
            "pipeline_ui_iter_metrics_test",
            SCRIPTS_DIR / "pipeline_ui.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        started = time.time()
        paths = module.iter_dataset_size_minimum_metrics_paths()
        elapsed = time.time() - started

        self.assertLess(elapsed, 2.0)
        self.assertGreaterEqual(len(paths), 1)
        for path in paths:
            self.assertTrue(path.name == "normalized_run_metrics.json")
            self.assertIn("summary", path.parts)
            self.assertIn("ranking", path.parts)

    @classmethod
    def setUpClass(cls) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "pipeline_ui_dataset_minimum_test",
            SCRIPTS_DIR / "pipeline_ui.py",
        )
        assert spec and spec.loader
        cls.pipeline_ui = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.pipeline_ui
        spec.loader.exec_module(cls.pipeline_ui)

    def test_discover_run_roots_marks_active_root_as_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_root = base / "graphene_w90_snapshot_scaling_test" / "finished_sweep"
            metrics_path = run_root / "summary" / "ranking" / "normalized_run_metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                json.dumps(
                    {
                        "metric_scaling_rows": [
                            {
                                "method": "graph2mat",
                                "dataset_size": 10,
                                "metric_key": "h_mae_eV_mean",
                                "metric_value": 0.02,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            original_results_root = self.pipeline_ui.RESULTS_ROOT
            original_status = self.pipeline_ui.G2M_DEEPH_RUNNER.status
            try:
                self.pipeline_ui.RESULTS_ROOT = base
                self.pipeline_ui.G2M_DEEPH_RUNNER.status = lambda: {
                    "running": True,
                    "run_root": str(run_root),
                }
                items = self.pipeline_ui.discover_dataset_size_minimum_run_roots()
            finally:
                self.pipeline_ui.RESULTS_ROOT = original_results_root
                self.pipeline_ui.G2M_DEEPH_RUNNER.status = original_status
            self.assertEqual(len(items), 1)
            self.assertFalse(items[0]["selectable"])
            self.assertEqual(items[0]["blocked_reason"], "sweep_en_curso")

    def test_run_analysis_rejects_active_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_root = base / "active_sweep"
            metrics_path = run_root / "summary" / "ranking" / "normalized_run_metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps({"metric_scaling_rows": []}), encoding="utf-8")
            original_status = self.pipeline_ui.G2M_DEEPH_RUNNER.status
            try:
                self.pipeline_ui.G2M_DEEPH_RUNNER.status = lambda: {
                    "running": True,
                    "run_root": str(run_root),
                }
                with self.assertRaises(RuntimeError):
                    self.pipeline_ui.run_dataset_size_minimum_analysis({"run_roots": [str(run_root)]})
            finally:
                self.pipeline_ui.G2M_DEEPH_RUNNER.status = original_status

    def test_preview_builds_best_rows_for_completed_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_root = base / "finished_sweep"
            metrics_path = run_root / "summary" / "ranking" / "normalized_run_metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                json.dumps(
                    {
                        "metric_scaling_rows": [
                            {
                                "method": "graph2mat",
                                "dataset_size": 10,
                                "metric_key": "h_mae_eV_mean",
                                "metric_value": 0.02,
                            },
                            {
                                "method": "deeph",
                                "dataset_size": 10,
                                "metric_key": "h_mae_eV_mean",
                                "metric_value": 0.03,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            original_results_root = self.pipeline_ui.RESULTS_ROOT
            try:
                self.pipeline_ui.RESULTS_ROOT = base
                preview = self.pipeline_ui.dataset_size_minimum_preview(
                    {
                        "run_roots": [str(run_root)],
                        "primary_metric": "h_mae_eV_mean",
                        "x_axis": "n_train",
                    }
                )
            finally:
                self.pipeline_ui.RESULTS_ROOT = original_results_root

            self.assertEqual(preview["status"], "ok")
            self.assertEqual(len(preview["best_rows"]), 2)
            self.assertEqual(
                sorted(row["method"] for row in preview["best_rows"]),
                ["deeph", "graph2mat"],
            )
            self.assertTrue(all(row.get("sweep_label") for row in preview["best_rows"]))

    def test_preview_aggregates_multiple_run_roots_by_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            def write_sweep(name: str, graph2mat_value: float) -> Path:
                run_root = base / name
                metrics_path = run_root / "summary" / "ranking" / "normalized_run_metrics.json"
                metrics_path.parent.mkdir(parents=True)
                metrics_path.write_text(
                    json.dumps(
                        {
                            "metric_scaling_rows": [
                                {
                                    "method": "graph2mat",
                                    "dataset_size": 10,
                                    "metric_key": "h_mae_eV_mean",
                                    "metric_value": graph2mat_value,
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                return run_root

            sweep_a = write_sweep("sweep_a", 0.020)
            sweep_b = write_sweep("sweep_b", 0.040)
            original_results_root = self.pipeline_ui.RESULTS_ROOT
            try:
                self.pipeline_ui.RESULTS_ROOT = base
                preview = self.pipeline_ui.dataset_size_minimum_preview(
                    {
                        "run_roots": [str(sweep_a), str(sweep_b)],
                        "primary_metric": "h_mae_eV_mean",
                        "x_axis": "n_train",
                    }
                )
            finally:
                self.pipeline_ui.RESULTS_ROOT = original_results_root

            self.assertTrue(preview.get("aggregated"))
            self.assertEqual(len(preview["best_rows"]), 1)
            self.assertAlmostEqual(preview["best_rows"][0]["primary_metric_mev_mean"], 30.0)


if __name__ == "__main__":
    unittest.main()
