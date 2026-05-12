from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("md", "siesta_fc_cartesian", "random_cartesian")
TEST_SETS = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed")
PRIMARY_METRIC = "low_energy_rmse_eV"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_completeness_report(
    path: Path,
    *,
    methods: tuple[str, ...] = METHODS,
    test_sets: tuple[str, ...] = TEST_SETS,
    missing_cells: tuple[str, ...] = (),
    missing_primary_metric_cells: tuple[str, ...] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_cells = [f"{method} on {test_set}" for method in methods for test_set in test_sets]
    missing = set(missing_cells)
    actual_cells = sorted(set(expected_cells) - missing)
    complete = not missing_cells and not missing_primary_metric_cells
    path.write_text(
        json.dumps(
            {
                "experiment_id": "synthetic_three_method",
                "primary_metric": PRIMARY_METRIC,
                "expected_cell_count": len(expected_cells),
                "actual_cell_count": len(actual_cells),
                "expected_cells": expected_cells,
                "actual_cells": actual_cells,
                "missing_cells": list(missing_cells),
                "extra_unexpected_cells": [],
                "missing_primary_metric_cells": list(missing_primary_metric_cells),
                "complete": complete,
                "scientific_status": "valid_grid" if complete else "invalid_incomplete_grid",
            }
        ),
        encoding="utf-8",
    )


def context_payload(
    *,
    md_size: int = 750,
    fc_size: int = 750,
    rc_size: int = 3500,
    md_hash: str = "md_hash_750",
    fc_hash: str = "fc_hash_750",
    rc_hash: str = "rc_hash_3500",
    suffix: str = "main",
) -> dict[str, dict[str, Any]]:
    sizes = {"md": md_size, "siesta_fc_cartesian": fc_size, "random_cartesian": rc_size}
    return {
        "dataset_size": sizes,
        "dataset_label": {
            "md": f"MD_{md_size}",
            "siesta_fc_cartesian": f"FC_{fc_size}",
            "random_cartesian": f"RC_{rc_size}_{suffix}",
        },
        "recipe_hash": {
            "md": md_hash,
            "siesta_fc_cartesian": fc_hash,
            "random_cartesian": rc_hash,
        },
        "result_dir": {
            method: f"/synthetic/{suffix}/{method}_{sizes[method]}"
            for method in METHODS
        },
    }


def make_grid_rows(
    *,
    experiment_id: str,
    values_for: Any,
    seeds: tuple[str, ...] = ("1", "2", "3"),
    methods: tuple[str, ...] = METHODS,
    test_sets: tuple[str, ...] = TEST_SETS,
    context: dict[str, dict[str, Any]] | None = None,
    omit_cells: set[tuple[str, str]] | None = None,
    missing_metric_cells: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    context = context or context_payload()
    omit_cells = omit_cells or set()
    missing_metric_cells = missing_metric_cells or set()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for method in methods:
            for test_set in test_sets:
                if (method, test_set) in omit_cells:
                    continue
                row = {
                    "experiment_id": experiment_id,
                    "sample_id": f"{experiment_id}_{seed}_{method}_{test_set}",
                    "train_method": method,
                    "test_set": test_set,
                    "test_method": test_set.removeprefix("test_"),
                    "seed": seed,
                    "model_checkpoint": f"{method}_{seed}.ckpt",
                    "dataset_size_by_method": json.dumps(context["dataset_size"], sort_keys=True),
                    "dataset_label_by_method": json.dumps(context["dataset_label"], sort_keys=True),
                    "recipe_hash_by_method": json.dumps(context["recipe_hash"], sort_keys=True),
                    "result_dir_by_method": json.dumps(context["result_dir"], sort_keys=True),
                    "train_dataset_size": context["dataset_size"][method],
                }
                if (method, test_set) not in missing_metric_cells:
                    row[PRIMARY_METRIC] = values_for(method, test_set, seed)
                rows.append(row)
    return rows


class TempWorkspace:
    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="analyze_winners_three_methods_"))
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


class AnalyzeWinnersThreeMethodsTests(unittest.TestCase):
    def run_analysis(
        self,
        rows: list[dict[str, Any]],
        root: Path,
        *,
        missing_cells: tuple[str, ...] = (),
        missing_primary_metric_cells: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        metrics_csv = root / "cross_evaluation_metrics.csv"
        output_dir = root / "summary"
        write_csv(metrics_csv, rows)
        write_completeness_report(
            output_dir / "cross_evaluation_completeness.json",
            missing_cells=missing_cells,
            missing_primary_metric_cells=missing_primary_metric_cells,
        )
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "Comparison" / "scripts" / "analyze_winners.py"),
                "--metrics-csv",
                str(metrics_csv),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                PRIMARY_METRIC,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return {
            "output_dir": output_dir,
            "recommendation": json.loads((output_dir / "recommendation.json").read_text(encoding="utf-8")),
            "nway_summary": json.loads((output_dir / "nway_ranking_summary.json").read_text(encoding="utf-8")),
            "dataset_thresholds": json.loads((output_dir / "dataset_size_thresholds_vs_md.json").read_text(encoding="utf-8")),
            "pairwise": read_csv(output_dir / "pairwise_vs_baseline.csv"),
            "nway_ranking": read_csv(output_dir / "nway_ranking.csv"),
            "nway_method_summary": read_csv(output_dir / "nway_method_summary.csv"),
        }

    def test_md_wins_across_all_test_sets_and_seeds(self) -> None:
        with TempWorkspace() as root:
            outputs = self.run_analysis(
                make_grid_rows(
                    experiment_id="md_wins",
                    values_for=lambda method, _test_set, _seed: {
                        "md": "0.20",
                        "siesta_fc_cartesian": "0.45",
                        "random_cartesian": "0.65",
                    }[method],
                ),
                root,
            )
        rec = outputs["recommendation"]
        self.assertEqual(rec["status"], "md_conservative_win")
        self.assertEqual(rec["scientific_status"], "robust_comparison")
        self.assertEqual(rec["winner"], "md")
        self.assertEqual(outputs["dataset_thresholds"]["thresholds"], [])
        self.assertTrue(all(row["winner"] == "md" for row in outputs["pairwise"]))

    def test_siesta_fc_cartesian_beats_md_robustly(self) -> None:
        with TempWorkspace() as root:
            outputs = self.run_analysis(
                make_grid_rows(
                    experiment_id="fc_wins",
                    values_for=lambda method, _test_set, _seed: {
                        "md": "0.50",
                        "siesta_fc_cartesian": "0.20",
                        "random_cartesian": "0.70",
                    }[method],
                ),
                root,
            )
        rec = outputs["recommendation"]
        self.assertEqual(rec["scientific_status"], "robust_comparison")
        self.assertEqual(rec["winner"], "siesta_fc_cartesian")
        self.assertEqual(rec["first_dataset_size_surpassing_md"], 750)
        thresholds = outputs["dataset_thresholds"]["thresholds"]
        fc_thresholds = [row for row in thresholds if row["challenger_method"] == "siesta_fc_cartesian"]
        self.assertTrue(fc_thresholds)
        self.assertEqual(fc_thresholds[0]["stability_status"], "robust_candidate")

    def test_random_cartesian_beats_md_robustly(self) -> None:
        with TempWorkspace() as root:
            outputs = self.run_analysis(
                make_grid_rows(
                    experiment_id="rc_wins",
                    values_for=lambda method, _test_set, _seed: {
                        "md": "0.50",
                        "siesta_fc_cartesian": "0.70",
                        "random_cartesian": "0.20",
                    }[method],
                ),
                root,
            )
        rec = outputs["recommendation"]
        self.assertEqual(rec["scientific_status"], "robust_comparison")
        self.assertEqual(rec["winner"], "random_cartesian")
        self.assertEqual(rec["first_dataset_size_surpassing_md"], 3500)
        self.assertEqual(rec["winning_threshold"]["challenger_recipe_hash"], "rc_hash_3500")

    def test_random_cartesian_own_distribution_only_win_is_not_general_md_defeat(self) -> None:
        def value(method: str, test_set: str, _seed: str) -> str:
            if method == "random_cartesian" and test_set == "test_random_cartesian":
                return "0.10"
            if method == "md":
                return "0.20"
            if method == "siesta_fc_cartesian":
                return "0.45"
            return "0.60"

        with TempWorkspace() as root:
            outputs = self.run_analysis(make_grid_rows(experiment_id="rc_own_only", values_for=value), root)
        rec = outputs["recommendation"]
        self.assertNotEqual(rec["winner"], "random_cartesian")
        self.assertTrue(rec["distribution_specific_wins"])
        self.assertEqual(rec["distribution_specific_wins"][0]["challenger_method"], "random_cartesian")
        self.assertEqual(rec["distribution_specific_wins"][0]["winning_test_set"], "test_random_cartesian")
        rc_thresholds = [
            row
            for row in outputs["dataset_thresholds"]["thresholds"]
            if row["challenger_method"] == "random_cartesian"
        ]
        self.assertEqual(rc_thresholds, [])

    def test_ranking_unstable_across_seeds_blocks_robust_recommendation(self) -> None:
        def value(method: str, _test_set: str, seed: str) -> str:
            winners = {
                "1": {"md": "0.10", "siesta_fc_cartesian": "0.40", "random_cartesian": "0.60"},
                "2": {"md": "0.50", "siesta_fc_cartesian": "0.10", "random_cartesian": "0.60"},
                "3": {"md": "0.50", "siesta_fc_cartesian": "0.60", "random_cartesian": "0.10"},
            }
            return winners[seed][method]

        with TempWorkspace() as root:
            outputs = self.run_analysis(make_grid_rows(experiment_id="unstable", values_for=value), root)
        rec = outputs["recommendation"]
        self.assertEqual(rec["status"], "unstable_seed_winner")
        self.assertNotEqual(rec["scientific_status"], "robust_comparison")
        self.assertTrue(rec["seed_stability"]["unstable_groups"])
        self.assertGreater(outputs["nway_summary"]["unstable_group_count"], 0)
        self.assertIn("unstable", {row["ranking_stability_status"] for row in outputs["nway_ranking"]})

    def test_one_seed_is_exploratory_only(self) -> None:
        with TempWorkspace() as root:
            outputs = self.run_analysis(
                make_grid_rows(
                    experiment_id="one_seed",
                    seeds=("1",),
                    values_for=lambda method, _test_set, _seed: {
                        "md": "0.50",
                        "siesta_fc_cartesian": "0.70",
                        "random_cartesian": "0.20",
                    }[method],
                ),
                root,
            )
        rec = outputs["recommendation"]
        self.assertEqual(rec["status"], "insufficient_seeds")
        self.assertEqual(rec["scientific_status"], "exploratory_only")
        self.assertIsNone(rec["winner"])
        self.assertEqual(rec["seed_stability"]["status"], "insufficient_seeds")
        self.assertEqual({row["seed_stability_status"] for row in outputs["pairwise"]}, {"exploratory_only"})
        self.assertGreater(outputs["nway_summary"]["exploratory_single_seed_group_count"], 0)

    def test_incomplete_grid_invalidates_recommendation(self) -> None:
        missing = ("random_cartesian on test_md",)
        rows = make_grid_rows(
            experiment_id="incomplete_grid",
            values_for=lambda method, _test_set, _seed: {
                "md": "0.20",
                "siesta_fc_cartesian": "0.45",
                "random_cartesian": "0.65",
            }[method],
            omit_cells={("random_cartesian", "test_md")},
        )
        with TempWorkspace() as root:
            outputs = self.run_analysis(rows, root, missing_cells=missing)
        rec = outputs["recommendation"]
        self.assertEqual(rec["status"], "invalid_incomplete_grid")
        self.assertEqual(rec["scientific_status"], "not_scientifically_valid")
        self.assertIsNone(rec["winner"])
        self.assertIn("random_cartesian on test_md", rec["missing_cells"])
        self.assertEqual(outputs["dataset_thresholds"]["thresholds"], [])

    def test_primary_metric_missing_in_one_cell_invalidates_recommendation(self) -> None:
        missing = ("md on test_random_cartesian",)
        rows = make_grid_rows(
            experiment_id="missing_primary",
            values_for=lambda method, _test_set, _seed: {
                "md": "0.20",
                "siesta_fc_cartesian": "0.45",
                "random_cartesian": "0.65",
            }[method],
            missing_metric_cells={("md", "test_random_cartesian")},
        )
        with TempWorkspace() as root:
            outputs = self.run_analysis(rows, root, missing_primary_metric_cells=missing)
        rec = outputs["recommendation"]
        self.assertEqual(rec["status"], "insufficient_primary_metric")
        self.assertEqual(rec["scientific_status"], "not_scientifically_valid")
        self.assertIsNone(rec["winner"])
        self.assertIn("md on test_random_cartesian", rec["missing_primary_metric_cells"])

    def test_two_random_cartesian_sizes_with_same_md_fc_sizes_do_not_collapse(self) -> None:
        rows = []
        rows.extend(
            make_grid_rows(
                experiment_id="two_rc_sizes",
                context=context_payload(rc_size=1500, rc_hash="rc_hash_1500", suffix="rc1500"),
                values_for=lambda method, _test_set, _seed: {
                    "md": "0.30",
                    "siesta_fc_cartesian": "0.55",
                    "random_cartesian": "0.60",
                }[method],
            )
        )
        rows.extend(
            make_grid_rows(
                experiment_id="two_rc_sizes",
                context=context_payload(rc_size=3500, rc_hash="rc_hash_3500", suffix="rc3500"),
                values_for=lambda method, _test_set, _seed: {
                    "md": "0.50",
                    "siesta_fc_cartesian": "0.70",
                    "random_cartesian": "0.20",
                }[method],
            )
        )
        with TempWorkspace() as root:
            outputs = self.run_analysis(rows, root)
        rc_pair_rows = [row for row in outputs["pairwise"] if row["challenger_method"] == "random_cartesian"]
        rc_sizes = {
            json.loads(row["dataset_size_by_method"])["random_cartesian"]
            for row in rc_pair_rows
        }
        self.assertEqual(rc_sizes, {1500, 3500})
        thresholds = [
            row
            for row in outputs["dataset_thresholds"]["thresholds"]
            if row["challenger_method"] == "random_cartesian"
        ]
        self.assertTrue(thresholds)
        self.assertEqual(thresholds[0]["first_stable_dataset_size"], 3500)

    def test_same_random_cartesian_size_different_recipe_hashes_do_not_collapse(self) -> None:
        rows = []
        rows.extend(
            make_grid_rows(
                experiment_id="same_rc_size_different_hash",
                context=context_payload(rc_size=3500, rc_hash="rc_hash_a_loses", suffix="rc_a"),
                values_for=lambda method, _test_set, _seed: {
                    "md": "0.30",
                    "siesta_fc_cartesian": "0.55",
                    "random_cartesian": "0.60",
                }[method],
            )
        )
        rows.extend(
            make_grid_rows(
                experiment_id="same_rc_size_different_hash",
                context=context_payload(rc_size=3500, rc_hash="rc_hash_b_wins", suffix="rc_b"),
                values_for=lambda method, _test_set, _seed: {
                    "md": "0.50",
                    "siesta_fc_cartesian": "0.70",
                    "random_cartesian": "0.20",
                }[method],
            )
        )
        with TempWorkspace() as root:
            outputs = self.run_analysis(rows, root)
        rc_pair_rows = [row for row in outputs["pairwise"] if row["challenger_method"] == "random_cartesian"]
        rc_hashes = {
            json.loads(row["recipe_hash_by_method"])["random_cartesian"]
            for row in rc_pair_rows
        }
        self.assertEqual(rc_hashes, {"rc_hash_a_loses", "rc_hash_b_wins"})
        rc_summary_hashes = {
            row["recipe_hash"]
            for row in outputs["nway_method_summary"]
            if row["method"] == "random_cartesian"
        }
        self.assertEqual(rc_summary_hashes, {"rc_hash_a_loses", "rc_hash_b_wins"})
        thresholds = [
            row
            for row in outputs["dataset_thresholds"]["thresholds"]
            if row["challenger_method"] == "random_cartesian"
        ]
        self.assertTrue(thresholds)
        self.assertEqual(thresholds[0]["challenger_recipe_hash"], "rc_hash_b_wins")


if __name__ == "__main__":
    unittest.main()
