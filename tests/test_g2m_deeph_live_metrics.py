import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_live_metrics import live_metric_scaling_rows  # noqa: E402


def _write_dataset(root: Path, count: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = [{"sample_id": f"md_{index}", "split": "test"} for index in range(count)]
    (root / "frozen_split_manifest.json").write_text(
        json.dumps({"rows": rows, "split_counts": {"test": count}}),
        encoding="utf-8",
    )


def _write_manifest(run_root: Path, records: list[dict]) -> None:
    sweep = run_root / "sweep"
    sweep.mkdir(parents=True, exist_ok=True)
    (sweep / "training_sweep_manifest.json").write_text(json.dumps({"runs": records}), encoding="utf-8")


def _base_record(run_root: Path, dataset: Path, model: str, config_id: str = "cfg") -> dict:
    return {
        "status": "completed",
        "model": model,
        "config_id": config_id,
        "config_hash": config_id,
        "dataset_id": dataset.name,
        "dataset_root": str(dataset),
        "run_root": str(run_root / "sweep" / model / dataset.name / config_id),
        "common": {"seed": 0},
        "overrides": {"max_epochs": 100, "epochs": 50},
        "metric_fail_policy": "fail_closed",
        "fail_open_metric_outputs": False,
        "metrics_run": {"returncode": 0},
    }


def _write_graph2mat_metrics(record: dict) -> None:
    metrics = Path(record["run_root"]) / "metrics" / "graph2mat" / "eval_input" / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / "manifest.json").write_text(json.dumps({"summary": {}}), encoding="utf-8")
    (metrics / "kpoint_matrix_metrics.csv").write_text(
        "sample,row_type,h_mae_eV,h_rmse_eV,h_mse_eV2,relative_frobenius\n"
        "md_0,weighted_sample,0.1,0.2,0.04,0.3\n"
        "md_1,weighted_sample,0.3,0.4,0.16,0.5\n",
        encoding="utf-8",
    )
    (metrics / "kpoint_spectral_metrics.csv").write_text(
        "sample,global_rmse_eV,low_energy_rmse_eV,fermi_window_rmse_eV,frontier_window_rmse_eV\n"
        "md_0,1.0,2.0,3.0,4.0\n"
        "md_1,3.0,4.0,5.0,6.0\n",
        encoding="utf-8",
    )
    (metrics / "kpoint_dos_metrics.csv").write_text(
        "sample,dos_mae_500_fermi_window,dos_wasserstein_eV\n"
        "md_0,0.01,0.02\n"
        "md_1,0.03,0.04\n",
        encoding="utf-8",
    )


def _write_deeph_metrics(record: dict) -> None:
    metrics = Path(record["run_root"]) / "metrics" / "deeph" / "eval" / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / "manifest.json").write_text(
        json.dumps({"prediction_adapter": {"adapter_equivalence_status": "invalid_orbital_order_unknown"}}),
        encoding="utf-8",
    )
    (metrics / "kpoint_matrix_metrics.csv").write_text(
        "sample,row_type,h_mae_eV,h_rmse_eV,h_mse_eV2,relative_frobenius,deeph_diagnostic_only\n"
        "md_0,weighted_sample,0.05,0.06,0.0036,0.07,True\n"
        "md_1,weighted_sample,0.15,0.16,0.0256,0.17,True\n",
        encoding="utf-8",
    )
    (metrics / "kpoint_spectral_metrics.csv").write_text(
        "sample,global_rmse_eV,fermi_window_rmse_eV,frontier_window_rmse_eV\n"
        "md_0,0.5,0.6,0.7\n"
        "md_1,1.5,1.6,1.7\n",
        encoding="utf-8",
    )
    (metrics / "kpoint_dos_metrics.csv").write_text(
        "sample,dos_mae_500_fermi_window,dos_wasserstein_eV\n"
        "md_0,0.005,0.006\n"
        "md_1,0.015,0.016\n",
        encoding="utf-8",
    )


class Graph2MatDeepHLiveMetricsTests(unittest.TestCase):
    def test_graph2mat_completed_run_produces_metric_scaling_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset)
            record = _base_record(root, dataset, "graph2mat", "g2m_cfg")
            _write_graph2mat_metrics(record)
            _write_manifest(root, [record])

            rows = live_metric_scaling_rows(root)

        by_metric = {row["metric_key"]: row for row in rows}
        self.assertAlmostEqual(by_metric["h_mae_eV_mean"]["metric_value"], 0.2)
        self.assertAlmostEqual(by_metric["relative_frobenius_mean"]["metric_value"], 0.4)
        self.assertAlmostEqual(by_metric["low_energy_rmse_eV_mean"]["metric_value"], 3.0)
        self.assertEqual(by_metric["h_mae_eV_mean"]["dataset_size"], 3)
        self.assertEqual(by_metric["h_mae_eV_mean"]["epoch_label"], "100 epochs")

    def test_deeph_completed_run_produces_diagnostic_metric_scaling_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset)
            record = _base_record(root, dataset, "deeph", "deeph_cfg")
            _write_deeph_metrics(record)
            _write_manifest(root, [record])

            rows = live_metric_scaling_rows(root)

        by_metric = {row["metric_key"]: row for row in rows}
        self.assertAlmostEqual(by_metric["h_mae_eV_mean"]["metric_value"], 0.1)
        self.assertAlmostEqual(by_metric["dos_mae_500_fermi_window_mean"]["metric_value"], 0.01)
        self.assertTrue(by_metric["h_mae_eV_mean"]["diagnostic_only"])
        self.assertEqual(by_metric["h_mae_eV_mean"]["scientific_status"], "diagnostic_only")
        self.assertEqual(by_metric["h_mae_eV_mean"]["epoch_label"], "50 epochs")

    def test_incomplete_or_failed_metrics_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset)
            failed = _base_record(root, dataset, "graph2mat", "failed")
            failed["metrics_run"] = {"returncode": 1}
            missing = _base_record(root, dataset, "graph2mat", "missing")
            missing.pop("metrics_run")
            _write_manifest(root, [failed, missing])

            self.assertEqual(live_metric_scaling_rows(root), [])

    def test_dedupe_keeps_one_row_per_config_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset)
            record = _base_record(root, dataset, "graph2mat", "g2m_cfg")
            _write_graph2mat_metrics(record)
            _write_manifest(root, [record, dict(record)])

            rows = live_metric_scaling_rows(root)

        keys = [(row["config_id"], row["metric_key"]) for row in rows]
        self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
