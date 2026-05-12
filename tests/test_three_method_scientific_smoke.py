from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
METHODS = ("md", "siesta_fc_cartesian", "random_cartesian")
TEST_SETS = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian")
PRIMARY_METRIC = "low_energy_rmse_eV"


def load_pipeline_ui():
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "pipeline_ui_three_method_scientific_smoke",
        SCRIPTS_DIR / "pipeline_ui.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(relative_path: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / relative_path), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def synthetic_metric(method: str) -> str:
    return {
        "md": "0.30",
        "siesta_fc_cartesian": "0.25",
        "random_cartesian": "0.20",
    }[method]


def write_cross_metric_cell(cross_root: Path, method: str, test_set: str) -> None:
    result_dir = cross_root / f"{method}__on__{test_set}"
    metrics_dir = result_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    write_csv(
        metrics_dir / "sparse_metrics.csv",
        [{"sample": "sample_1", "relative_frobenius_union": "0.40"}],
    )
    write_csv(
        metrics_dir / "spectral_metrics.csv",
        [
            {
                "sample": "sample_1",
                "low_energy_n_states": "3",
                "low_energy_mae_eV": "0.10",
                PRIMARY_METRIC: synthetic_metric(method),
                "low_energy_max_abs_error_eV": "0.35",
            }
        ],
    )
    context_sizes = {"md": 190, "siesta_fc_cartesian": 190, "random_cartesian": 570}
    result_dirs = {
        item: f"/synthetic/results/{item}_{context_sizes[item]}"
        for item in METHODS
    }
    (result_dir / "cross_evaluation_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "three_method_scientific_smoke",
                "train_method": method,
                "test_set": test_set,
                "test_method": test_set.removeprefix("test_"),
                "dataset_size": context_sizes[method],
                "train_dataset_size": context_sizes[method],
                "dataset_size_by_method": context_sizes,
                "dataset_label_by_method": {
                    "md": "MD_190",
                    "siesta_fc_cartesian": "FC_190",
                    "random_cartesian": "RC_570",
                },
                "recipe_set_hash_by_method": {
                    "md": "md_hash_190",
                    "siesta_fc_cartesian": "fc_hash_190",
                    "random_cartesian": "rc_hash_570",
                },
                "result_dir_by_method": result_dirs,
                "md_dataset_size": context_sizes["md"],
                "atom_dataset_size": context_sizes["siesta_fc_cartesian"],
                "random_dataset_size": context_sizes["random_cartesian"],
                "seed": 1,
                "model_checkpoint": f"{method}_seed1.ckpt",
                "model_checkpoint_sha256": f"{method}_checkpoint_hash",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


class ThreeMethodScientificSmokeTests(unittest.TestCase):
    def test_three_method_control_flow_with_one_seed_is_exploratory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="three_method_scientific_smoke_") as tmp:
            root = Path(tmp)
            summary_dir = root / "summary"
            cross_root = root / "cross_evaluations"

            pipeline_ui = load_pipeline_ui()
            expected_grid = pipeline_ui.build_cross_evaluation_expected_grid(
                METHODS,
                TEST_SETS,
                experiment_id="three_method_scientific_smoke",
            )
            expected_grid_path = summary_dir / "cross_evaluation_expected_grid.json"
            expected_grid_path.parent.mkdir(parents=True, exist_ok=True)
            expected_grid_path.write_text(
                json.dumps(expected_grid, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(expected_grid["selected_methods"], list(METHODS))
            self.assertEqual(expected_grid["selected_frozen_test_sets"], list(TEST_SETS))
            self.assertEqual(expected_grid["expected_cell_count"], len(METHODS) * len(TEST_SETS))

            for method in METHODS:
                for test_set in TEST_SETS:
                    write_cross_metric_cell(cross_root, method, test_set)

            aggregate = run_script(
                "Comparison/scripts/aggregate_cross_metrics.py",
                "--experiment-id",
                "three_method_scientific_smoke",
                "--cross-root",
                str(cross_root),
                "--output-dir",
                str(summary_dir),
                "--expected-grid",
                str(expected_grid_path),
                "--primary-metric",
                PRIMARY_METRIC,
            )
            self.assertEqual(aggregate.returncode, 0, aggregate.stderr + aggregate.stdout)

            completeness = json.loads(
                (summary_dir / "cross_evaluation_completeness.json").read_text(encoding="utf-8")
            )
            self.assertTrue(completeness["complete"])
            self.assertEqual(completeness["scientific_status"], "valid_grid")
            self.assertEqual(completeness["missing_cells"], [])
            self.assertEqual(completeness["missing_primary_metric_cells"], [])

            aggregated_rows = read_csv(summary_dir / "cross_evaluation_metrics.csv")
            aggregate_summary = json.loads(
                (summary_dir / "cross_evaluation_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(aggregate_summary["train_methods"]), set(METHODS))
            self.assertEqual(set(aggregate_summary["test_sets"]), set(TEST_SETS))
            self.assertEqual(len(aggregated_rows), len(METHODS) * len(TEST_SETS))
            self.assertEqual({row["train_method"] for row in aggregated_rows}, set(METHODS))
            self.assertEqual({row["test_set"] for row in aggregated_rows}, set(TEST_SETS))
            self.assertTrue(all(row[PRIMARY_METRIC] for row in aggregated_rows))

            analysis = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(summary_dir / "cross_evaluation_metrics.csv"),
                "--output-dir",
                str(summary_dir),
                "--primary-metric",
                PRIMARY_METRIC,
            )
            self.assertEqual(analysis.returncode, 0, analysis.stderr + analysis.stdout)

            recommendation = json.loads((summary_dir / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["baseline"], "md")
            self.assertEqual(recommendation["challengers"], ["siesta_fc_cartesian", "random_cartesian"])
            self.assertEqual(recommendation["primary_metric"], PRIMARY_METRIC)
            self.assertEqual(recommendation["scientific_status"], "exploratory_only")
            self.assertNotEqual(recommendation["scientific_status"], "robust_comparison")
            self.assertIsNone(recommendation["winner"])
            self.assertIn("Fewer than 3 valid seeds", recommendation["reason"])

            nway_rows = read_csv(summary_dir / "nway_method_summary.csv")
            pairwise_rows = read_csv(summary_dir / "pairwise_vs_baseline.csv")
            self.assertEqual({row["train_method"] for row in nway_rows}, set(METHODS))
            self.assertEqual({row["challenger_method"] for row in pairwise_rows}, {"siesta_fc_cartesian", "random_cartesian"})


if __name__ == "__main__":
    unittest.main()
