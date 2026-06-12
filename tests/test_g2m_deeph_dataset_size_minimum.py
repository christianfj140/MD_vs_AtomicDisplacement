import argparse
import csv
import importlib
import json
import math
import random
import re
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

minimum = importlib.import_module("g2m_deeph_dataset_size_minimum")


def make_analyze_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "run_root": [],
        "output_dir": "",
        "primary_metric": "h_mae_eV_mean",
        "threshold_mev": 10.0,
        "threshold_preset_key": "h_mae_relaxed_10",
        "threshold_is_user_defined": False,
        "relative_tolerance": 0.05,
        "plateau_gain": 0.05,
        "x_axis": "n_total",
        "fit_models": "linear,inverse,power_law",
        "n_min_source": "observed",
        "n_min_fit_model": "linear",
        "moving_average_window": 3,
        "aggregation_mode": None,
        "cost_basis": "per_seed_mean",
        "bootstrap_replicates": 0,
        "bootstrap_seed": 12345,
        "ci_level": 0.95,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_normalized_replicate_row(
    *,
    method: str,
    size: int,
    mev: float,
    seed: str,
    config_id: str = "cfg",
    gpu_hours_total: float | None = None,
) -> dict:
    return {
        "method": method,
        "dataset_size_x": size,
        "dataset_size_total": size,
        "dataset_size_train": size,
        "config_id": config_id,
        "seed": seed,
        "primary_metric": "h_mae_eV_mean",
        "primary_metric_mev": mev,
        "primary_metric_mev_mean": mev,
        "gpu_hours_total": gpu_hours_total,
    }


def write_synthetic_temporal_dataset(
    root: Path,
    *,
    n_train: int = 8,
    strategy: str = "blocked_with_gap",
    temporal_gap: int = 2,
    scalar_key: str = "total_energy_eV",
    scalar_builder=None,
) -> Path:
    split_root = root / "splits"
    train_dir = split_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    frozen_rows: list[dict] = []

    for index in range(n_train):
        sample_dir = train_dir / str(index)
        sample_dir.mkdir(parents=True, exist_ok=True)
        value = scalar_builder(index) if scalar_builder is not None else float(index)
        metadata = {
            "frame_index": str(index),
            "source_frame_index": str(index),
            "block_id": "block_a",
            "temperature_K": "300",
            scalar_key: value,
        }
        (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        manifest_rows.append(
            {
                "sample_id": f"md_{index}",
                "sample_dir": str(sample_dir),
                "split": "train",
                "split_strategy": strategy,
                "temporal_gap": str(temporal_gap),
                "source_frame_index": str(index),
                "block_id": "block_a",
                "temperature_K": "300",
            }
        )
        frozen_rows.append(
            {
                "sample_id": f"md_{index}",
                "sample_dir": str(sample_dir),
                "split": "train",
            }
        )

    write_csv(split_root / "train_manifest.csv", manifest_rows)
    split_summary = {
        "strategy": strategy,
        "temporal_gap": temporal_gap,
        "counts": {"train": n_train},
        "warnings": [],
        "scientific_status": "temporal_gap_split",
    }
    (split_root / "split_summary.json").write_text(json.dumps(split_summary), encoding="utf-8")
    frozen = {
        "schema": "joint_graph2mat_deeph_frozen_split_manifest_v1",
        "dataset_root": str(root),
        "split_root": str(split_root),
        "split_counts": {"train": n_train},
        "rows": frozen_rows,
    }
    (root / "frozen_split_manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
    return root


def write_temporal_dataset_from_specs(
    root: Path,
    *,
    specs: list[dict[str, object]],
    strategy: str = "blocked_with_gap",
    temporal_gap: int = 2,
) -> Path:
    split_root = root / "splits"
    train_dir = split_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    frozen_rows: list[dict] = []

    for index, spec in enumerate(specs):
        sample_dir = train_dir / str(index)
        sample_dir.mkdir(parents=True, exist_ok=True)
        metadata = dict(spec.get("metadata") or {})
        metadata.setdefault("source_frame_index", metadata.get("frame_index", index))
        metadata.setdefault("frame_index", index)
        (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        manifest_row = {
            "sample_id": f"md_{index}",
            "sample_dir": str(sample_dir),
            "split": str(spec.get("split") or "train"),
            "split_strategy": strategy,
            "temporal_gap": str(spec.get("temporal_gap", temporal_gap)),
        }
        for key in ("source_frame_index", "frame_index", "block_id", "temperature_K", "trajectory_id"):
            value = metadata.get(key)
            if value is not None:
                manifest_row[key] = str(value)
        manifest_rows.append(manifest_row)
        frozen_rows.append(
            {
                "sample_id": f"md_{index}",
                "sample_dir": str(sample_dir),
                "split": str(spec.get("split") or "train"),
            }
        )

    write_csv(split_root / "train_manifest.csv", manifest_rows)
    split_summary = {
        "strategy": strategy,
        "temporal_gap": temporal_gap,
        "counts": {"train": sum(1 for spec in specs if str(spec.get("split") or "train") == "train")},
        "warnings": [],
        "scientific_status": "temporal_gap_split",
    }
    (split_root / "split_summary.json").write_text(json.dumps(split_summary), encoding="utf-8")
    frozen = {
        "schema": "joint_graph2mat_deeph_frozen_split_manifest_v1",
        "dataset_root": str(root),
        "split_root": str(split_root),
        "split_counts": split_summary["counts"],
        "rows": frozen_rows,
    }
    (root / "frozen_split_manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
    return root


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


def dataset_minimum_visible_fit_options() -> list[str]:
    html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
    match = re.search(
        r'<select id="dataset-minimum-fit">(.*?)</select>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    return re.findall(r'<option value="([^"]+)"', match.group(1))


class DatasetMinimumThresholdProtocolTests(unittest.TestCase):
    def test_threshold_metadata_for_metric_specific_preset(self) -> None:
        metadata = minimum.resolve_threshold_metadata(
            primary_metric="h_mae_eV_mean",
            threshold_mev=10.0,
            threshold_preset_key="h_mae_relaxed_10",
            threshold_is_user_defined=False,
        )

        self.assertEqual(metadata["threshold_basis"], minimum.THRESHOLD_BASIS_EXPLORATORY_PRESET)
        self.assertEqual(metadata["threshold_metric_family"], "hamiltonian_element_error_mev")
        self.assertFalse(metadata["threshold_is_user_defined"])
        self.assertEqual(metadata["threshold_preset_key"], "h_mae_relaxed_10")

    def test_threshold_metadata_for_manual_threshold_is_user_defined_exploratory(self) -> None:
        metadata = minimum.resolve_threshold_metadata(
            primary_metric="fermi_window_rmse_eV",
            threshold_mev=17.0,
            threshold_preset_key=None,
            threshold_is_user_defined=True,
        )

        self.assertEqual(metadata["threshold_basis"], minimum.THRESHOLD_BASIS_USER_DEFINED)
        self.assertTrue(metadata["threshold_is_user_defined"])
        self.assertEqual(metadata["threshold_preset_key"], minimum.THRESHOLD_MANUAL_PRESET_KEY)

    def test_threshold_metadata_for_explicit_protocol_can_be_paper_justified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            protocol = Path(tmp) / "threshold_protocol.json"
            protocol.write_text(
                json.dumps(
                    {
                        "metric": "h_mae_eV_mean",
                        "threshold_mev": 10.0,
                        "physical_rationale": "Test-only documented H-MAE publication threshold.",
                        "reference": "internal_protocol_v1",
                        "applies_to_metrics": ["h_mae_eV_mean"],
                        "recommended_sensitivity_thresholds_mev": [8.0, 10.0, 12.0],
                        "sensitivity_recommendation": "Audit nearby thresholds before paper-level use.",
                    }
                ),
                encoding="utf-8",
            )
            metadata = minimum.resolve_threshold_metadata(
                primary_metric="h_mae_eV_mean",
                threshold_mev=10.0,
                threshold_preset_key="h_mae_relaxed_10",
                threshold_is_user_defined=False,
                threshold_protocol_file=str(protocol),
            )

        self.assertEqual(metadata["threshold_basis"], minimum.THRESHOLD_BASIS_EXPLICIT_PROTOCOL)
        self.assertTrue(metadata["threshold_paper_justified"])
        self.assertEqual(metadata["threshold_protocol_reference"], "internal_protocol_v1")
        self.assertEqual(metadata["threshold_protocol_sensitivity_thresholds_mev"], [8.0, 10.0, 12.0])

    def test_scientific_status_blocks_exploratory_threshold_basis(self) -> None:
        status = minimum.scientific_claim_status_payload(
            temporal_diagnostics={
                "autocorrelation_available": True,
                "nominal_n_train": 100,
                "estimated_n_eff_train": 80.0,
                "warnings": [],
            },
            thresholds={"graph2mat": {"N_min_abs": 20}},
            threshold_metadata={
                "threshold_basis": minimum.THRESHOLD_BASIS_EXPLORATORY_PRESET,
                "threshold_reference": "doc",
                "threshold_interpretation": "exploratory",
                "threshold_metric_family": "spectral_error_mev",
                "threshold_is_user_defined": False,
                "threshold_paper_justified": False,
            },
            aggregation_mode="mean_seeds_per_config",
            requested_n_min_source="fit",
            actual_n_min_source="fit",
            requested_fit_model="power_law_floor",
            actual_fit_model="power_law_floor",
            fit_threshold_details={
                "graph2mat": {
                    "fit_policy": "paper_candidate",
                    "paper_candidate": True,
                    "fit_model": "power_law_floor",
                    "status": "ok",
                    "enough_points_for_paper_candidate": True,
                }
            },
            fallback_used=False,
            fallback_reason=None,
        )

        self.assertEqual(status["scientific_claim_status"], "diagnostic_only")
        self.assertIn("paper_blocked_if_threshold_basis_not_paper_justified", status["paper_level_blockers"])

    def test_threshold_sensitivity_marks_unstable_threshold_protocol(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=50.0, seed="1", config_id="a"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=18.0, seed="1", config_id="b"),
            make_normalized_replicate_row(method="graph2mat", size=30, mev=9.0, seed="1", config_id="c"),
        ]
        sensitivity = minimum.threshold_sensitivity_summary(
            rows,
            threshold_values_mev=[10.0, 60.0],
            main_threshold_mev=10.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            cost_basis="per_seed_mean",
            n_min_source="observed",
            fit_model="power_law_floor",
            moving_average_window=3,
        )
        self.assertTrue(sensitivity["enabled"])
        self.assertTrue(sensitivity["by_method"]["graph2mat"]["unstable"])
        self.assertIn(
            "paper_blocked_if_threshold_sensitivity_unstable:graph2mat",
            sensitivity["paper_level_blockers"],
        )

    def test_threshold_sensitivity_blocks_missing_threshold_crossings(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=8.0, seed="1", config_id="a"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=7.0, seed="1", config_id="b"),
            make_normalized_replicate_row(method="graph2mat", size=30, mev=6.0, seed="1", config_id="c"),
        ]
        sensitivity = minimum.threshold_sensitivity_summary(
            rows,
            threshold_values_mev=[5.0, 7.0, 9.0],
            main_threshold_mev=7.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            cost_basis="per_seed_mean",
            n_min_source="observed",
            fit_model="power_law_floor",
            moving_average_window=3,
            claim_mode="paper_candidate",
        )
        method_payload = sensitivity["by_method"]["graph2mat"]
        self.assertEqual(method_payload["missing_threshold_crossings"], [5.0])
        self.assertIn(
            "paper_blocked_if_threshold_sensitivity_missing_n_min_abs:graph2mat",
            method_payload["paper_level_blockers"],
        )

    def test_threshold_sensitivity_requires_real_range_for_paper_candidate(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=25.0, seed="1", config_id="a"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=15.0, seed="1", config_id="b"),
            make_normalized_replicate_row(method="graph2mat", size=30, mev=9.0, seed="1", config_id="c"),
        ]
        sensitivity = minimum.threshold_sensitivity_summary(
            rows,
            threshold_values_mev=[10.0],
            main_threshold_mev=10.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            cost_basis="per_seed_mean",
            n_min_source="observed",
            fit_model="power_law_floor",
            moving_average_window=3,
            claim_mode="paper_candidate",
        )
        self.assertFalse(sensitivity["sufficient_range_for_paper_candidate"])
        self.assertIn(
            "paper_blocked_if_threshold_sensitivity_insufficient_range",
            sensitivity["paper_level_blockers"],
        )

    def test_threshold_sensitivity_reports_abs_rel_tol_and_plateau(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=18.0, seed="1", config_id="a"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=12.0, seed="1", config_id="b"),
            make_normalized_replicate_row(method="graph2mat", size=30, mev=9.0, seed="1", config_id="c"),
        ]
        sensitivity = minimum.threshold_sensitivity_summary(
            rows,
            threshold_values_mev=[10.0, 12.0, 15.0],
            main_threshold_mev=12.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            cost_basis="per_seed_mean",
            n_min_source="observed",
            fit_model="power_law_floor",
            moving_average_window=3,
        )
        series = sensitivity["by_method"]["graph2mat"]["threshold_series"]
        self.assertIn("N_min_abs", series[0])
        self.assertIn(minimum.N_MIN_REL_TOL_KEY, series[0])
        self.assertIn("N_min_plateau", series[0])


class DatasetSizeMinimumTests(unittest.TestCase):
    def test_canonical_fit_model_maps_power_law_to_power_law_floor(self) -> None:
        self.assertEqual(minimum.canonical_fit_model("power_law"), "power_law_floor")
        self.assertEqual(minimum.canonical_fit_model("power_law_floor"), "power_law_floor")
        self.assertTrue(minimum.fit_models_equivalent("power_law", "power_law_floor"))

    def test_parse_fit_models_accepts_visible_ui_fit_options(self) -> None:
        visible = dataset_minimum_visible_fit_options()

        parsed = minimum.parse_fit_models(",".join(visible))

        self.assertEqual(parsed, visible)

    def test_invalid_fit_models_are_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            minimum.parse_fit_models("linear,definitely_not_a_fit")

    def test_parse_single_fit_model_accepts_power_law_floor(self) -> None:
        self.assertEqual(minimum.parse_single_fit_model("power_law_floor"), "power_law_floor")

    def test_parse_single_fit_model_accepts_power_law_alias(self) -> None:
        parsed = minimum.parse_single_fit_model("power_law")
        self.assertEqual(parsed, "power_law")
        self.assertEqual(minimum.canonical_fit_model(parsed), "power_law_floor")

    def test_parse_single_fit_model_accepts_none(self) -> None:
        self.assertEqual(minimum.parse_single_fit_model("none"), "none")

    def test_parse_single_fit_model_rejects_multiple_models(self) -> None:
        with self.assertRaises(SystemExit) as exc:
            minimum.parse_single_fit_model("linear,inverse")
        self.assertIn("exactly one fit model", str(exc.exception))

    def test_parse_single_fit_model_rejects_unknown_model(self) -> None:
        with self.assertRaises(SystemExit) as exc:
            minimum.parse_single_fit_model("not_a_model")
        self.assertIn("Unknown fit model", str(exc.exception))

    def test_power_law_floor_fit_has_valid_constraints(self) -> None:
        n_values = [10.0, 20.0, 40.0, 80.0]
        y_values = [30.0, 22.0, 18.0, 15.0]
        fit = minimum.fit_power_law_floor(n_values, y_values)
        self.assertEqual(fit["status"], "ok")
        self.assertEqual(fit["model"], "power_law_floor")
        e_inf, amplitude, alpha = fit["coefficients"]
        self.assertGreaterEqual(e_inf, 0.0)
        self.assertGreaterEqual(amplitude, 0.0)
        self.assertGreater(alpha, 0.0)
        predictions = minimum.power_law_floor_predictions(fit["coefficients"], n_values)
        self.assertTrue(all(value >= -1e-9 for value in predictions))
        self.assertEqual(fit["alpha_search_method"], "coarse_grid_plus_golden_section")
        self.assertEqual(fit["alpha_bounds"]["min"], minimum.POWER_LAW_ALPHA_MIN)
        self.assertEqual(fit["alpha_bounds"]["max"], minimum.POWER_LAW_ALPHA_MAX)
        self.assertGreater(fit["objective_evaluations"], minimum.POWER_LAW_ALPHA_GRID_POINTS)
        self.assertIn("sse", fit)
        self.assertTrue(fit["nonnegative_constraints_active"])

    def test_power_law_floor_recovers_known_alpha_with_refinement(self) -> None:
        n_values = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
        expected_alpha = 1.2
        e_inf = 4.0
        amplitude = 80.0
        y_values = [e_inf + amplitude * (n ** (-expected_alpha)) for n in n_values]

        fit = minimum.fit_power_law_floor(n_values, y_values)

        self.assertEqual(fit["status"], "ok")
        self.assertAlmostEqual(fit["coefficients_named"]["alpha"], expected_alpha, delta=0.08)
        self.assertAlmostEqual(fit["coefficients_named"]["e_inf"], e_inf, delta=0.5)
        self.assertAlmostEqual(fit["coefficients_named"]["amplitude"], amplitude, delta=5.0)
        self.assertLess(fit["rmse_mev"], 0.1)

    def test_power_law_alias_matches_floor_fit(self) -> None:
        n_values = [10.0, 20.0, 40.0, 80.0]
        y_values = [30.0, 22.0, 18.0, 15.0]
        floor = minimum.fit_power_law_floor(n_values, y_values)
        legacy = minimum.fit_power_law(n_values, y_values)
        self.assertEqual(floor["coefficients"], legacy["coefficients"])

    def test_none_n_min_fit_model_uses_observed_thresholds_without_failed_fit_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "a",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.030,
                        "seed": "1",
                    },
                    {
                        "method": "graph2mat",
                        "dataset_size": 20,
                        "config_id": "b",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.008,
                        "seed": "1",
                    },
                ]
            }
            (root / "summary" / "ranking").mkdir(parents=True)
            (root / "summary" / "ranking" / "normalized_run_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            summary = minimum.analyze(
                make_analyze_args(
                    run_root=[str(root)],
                    output_dir=str(base / "out"),
                    fit_models="none,cumulative_best,power_law",
                    n_min_source="fit",
                    n_min_fit_model="none",
                    threshold_mev=10.0,
                )
            )

            warnings = " | ".join(summary["warnings"])
            self.assertFalse(summary["fallback_used"])
            self.assertEqual(summary["requested_n_min_source"], "fit")
            self.assertEqual(summary["actual_n_min_source"], "observed")
            self.assertEqual(summary["n_min_source"], "observed")
            self.assertEqual(summary["requested_fit_model"], "none")
            self.assertEqual(summary["actual_fit_model"], "none")
            self.assertEqual(summary["canonical_fit_model"], "none")
            self.assertEqual(summary["thresholds"]["graph2mat"]["N_min_abs"], 20)
            self.assertEqual(summary["fit_threshold_details"]["graph2mat"]["status"], "not_used")
            self.assertNotIn("fit_thresholds_unavailable", warnings)
            self.assertNotIn("n_min_explicit_fallback_to_observed", warnings)

    def test_cumulative_best_fit_is_monotone_observed_only_and_drives_thresholds(self) -> None:
        rows = [
            {"method": "graph2mat", "dataset_size_x": 10, "primary_metric_mev_mean": 30.0},
            {"method": "graph2mat", "dataset_size_x": 20, "primary_metric_mev_mean": 35.0},
            {"method": "graph2mat", "dataset_size_x": 40, "primary_metric_mev_mean": 15.0},
            {"method": "graph2mat", "dataset_size_x": 80, "primary_metric_mev_mean": 18.0},
        ]

        curve_rows, fit = minimum.fitted_curve_rows(rows, fit_model="cumulative_best")
        thresholds, details, warnings = minimum.thresholds_by_method_from_fit(
            rows,
            threshold_mev=20.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            fit_model="cumulative_best",
        )

        y_values = [row["primary_metric_mev_mean"] for row in curve_rows]
        x_values = [int(row["dataset_size_x"]) for row in curve_rows]
        self.assertEqual(fit["status"], "ok")
        self.assertTrue(fit["diagnostic_only"])
        self.assertEqual(x_values, [10, 20, 40, 80])
        self.assertEqual(y_values, [30.0, 30.0, 15.0, 15.0])
        self.assertTrue(all(left >= right for left, right in zip(y_values, y_values[1:])))
        self.assertEqual(thresholds["graph2mat"]["N_min_abs"], 40)
        self.assertEqual(thresholds["graph2mat"]["fit_model"], "cumulative_best")
        self.assertEqual(details["graph2mat"]["status"], "ok")
        self.assertEqual(warnings, [])

    def test_fit_failure_sets_explicit_observed_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "a",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.010,
                        "seed": "1",
                    }
                ]
            }
            (root / "summary" / "ranking").mkdir(parents=True)
            (root / "summary" / "ranking" / "normalized_run_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            summary = minimum.analyze(
                make_analyze_args(
                    run_root=[str(root)],
                    output_dir=str(base / "out"),
                    n_min_source="fit",
                    n_min_fit_model="power_law_floor",
                )
            )
            self.assertTrue(summary["fallback_used"])
            self.assertEqual(summary["requested_n_min_source"], "fit")
            self.assertEqual(summary["actual_n_min_source"], "observed")
            self.assertEqual(summary["canonical_fit_model"], "power_law_floor")
            self.assertIn("n_min_explicit_fallback_to_observed", " | ".join(summary["warnings"]))

    def test_fit_predictive_stability_stable_curve_has_no_blocker(self) -> None:
        rows = [
            {"method": "graph2mat", "dataset_size_x": 10, "primary_metric_mev_mean": 30.0},
            {"method": "graph2mat", "dataset_size_x": 20, "primary_metric_mev_mean": 18.0},
            {"method": "graph2mat", "dataset_size_x": 40, "primary_metric_mev_mean": 12.0},
            {"method": "graph2mat", "dataset_size_x": 80, "primary_metric_mev_mean": 9.0},
            {"method": "graph2mat", "dataset_size_x": 160, "primary_metric_mev_mean": 7.0},
        ]

        stability = minimum.fit_predictive_stability_by_left_out_N(
            rows,
            threshold_mev=10.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            fit_model="power_law_floor",
            n_min_source="fit",
        )

        self.assertEqual(stability["status"], "ok")
        self.assertFalse(stability["paper_level_blockers"])
        method = stability["methods"]["graph2mat"]
        self.assertEqual(method["n_leave_one_out_trials"], 5)
        self.assertEqual(method["n_failed"], 0)
        self.assertFalse(method["unstable_criteria"])
        self.assertFalse(method["paper_level_blockers"])

    def test_fit_predictive_stability_unstable_curve_adds_blocker(self) -> None:
        rows = [
            {"method": "graph2mat", "dataset_size_x": 10, "primary_metric_mev_mean": 30.0},
            {"method": "graph2mat", "dataset_size_x": 20, "primary_metric_mev_mean": 14.0},
            {"method": "graph2mat", "dataset_size_x": 40, "primary_metric_mev_mean": 13.0},
            {"method": "graph2mat", "dataset_size_x": 80, "primary_metric_mev_mean": 12.0},
            {"method": "graph2mat", "dataset_size_x": 160, "primary_metric_mev_mean": 8.0},
        ]

        stability = minimum.fit_predictive_stability_by_left_out_N(
            rows,
            threshold_mev=10.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            fit_model="power_law_floor",
            n_min_source="fit",
        )

        self.assertEqual(stability["status"], "ok")
        method = stability["methods"]["graph2mat"]
        self.assertTrue(method["paper_level_blockers"])
        self.assertIn(
            "paper_blocked_if_fit_predictive_stability_unstable:graph2mat",
            " ".join(method["paper_level_blockers"]),
        )

    def test_fit_predictive_stability_records_leave_one_out_failures(self) -> None:
        rows = [
            {"method": "graph2mat", "dataset_size_x": 10, "primary_metric_mev_mean": 30.0},
            {"method": "graph2mat", "dataset_size_x": 20, "primary_metric_mev_mean": 20.0},
            {"method": "graph2mat", "dataset_size_x": 40, "primary_metric_mev_mean": 12.0},
        ]

        stability = minimum.fit_predictive_stability_by_left_out_N(
            rows,
            threshold_mev=12.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            fit_model="quadratic",
            n_min_source="fit",
        )

        method = stability["methods"]["graph2mat"]
        self.assertEqual(method["n_leave_one_out_trials"], 3)
        self.assertGreater(method["n_failed"], 0)
        self.assertTrue(
            any(str(trial["fit_status"]).startswith("skipped_") for trial in method["trials"])
        )

    def test_fit_predictive_stability_not_applicable_for_observed_mode(self) -> None:
        rows = [
            {"method": "graph2mat", "dataset_size_x": 10, "primary_metric_mev_mean": 30.0},
            {"method": "graph2mat", "dataset_size_x": 20, "primary_metric_mev_mean": 20.0},
            {"method": "graph2mat", "dataset_size_x": 40, "primary_metric_mev_mean": 12.0},
        ]

        stability = minimum.fit_predictive_stability_by_left_out_N(
            rows,
            threshold_mev=12.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            fit_model="power_law_floor",
            n_min_source="observed",
        )

        self.assertEqual(stability["status"], "not_applicable")
        self.assertEqual(stability["reason"], "observed_only_mode")

    def test_mean_replicates_averages_same_method_and_size(self) -> None:
        rows = [
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 10.0,
                "config_id": "cfg_a",
                "seed": "1",
                "source_run_root": "/run_a",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 14.0,
                "config_id": "cfg_b",
                "seed": "2",
                "source_run_root": "/run_b",
            },
        ]

        aggregated = minimum.aggregate_rows_mean_replicates(rows)

        self.assertEqual(len(aggregated), 1)
        row = aggregated[0]
        self.assertAlmostEqual(row["primary_metric_mev_mean"], 12.0)
        self.assertEqual(row["replicate_count"], 2)
        self.assertAlmostEqual(row["y_min"], 10.0)
        self.assertAlmostEqual(row["y_max"], 14.0)

    def test_best_config_picks_lowest_metric_for_same_method_and_size(self) -> None:
        rows = [
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev_mean": 10.0,
                "config_id": "cfg_a",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev_mean": 14.0,
                "config_id": "cfg_b",
            },
        ]

        best = minimum.best_by_method_size(rows)

        self.assertEqual(len(best), 1)
        self.assertAlmostEqual(best[0]["primary_metric_mev_mean"], 10.0)
        self.assertEqual(best[0]["config_id"], "cfg_a")

    def test_extract_base_config_id_strips_seed_suffix(self) -> None:
        self.assertEqual(minimum.extract_base_config_id("policy_a-seed1"), "policy_a")
        self.assertEqual(minimum.extract_base_config_id("policy_a-seed4"), "policy_a")
        self.assertEqual(minimum.extract_base_config_id("policy_a-seed42"), "policy_a")
        self.assertEqual(minimum.extract_base_config_id("policy_a-seed123"), "policy_a")
        self.assertEqual(minimum.extract_base_config_id("policy-seed-study-seed42"), "policy-seed-study")
        self.assertEqual(minimum.extract_base_config_id("policy-seed-study"), "policy-seed-study")
        self.assertEqual(minimum.extract_base_config_id("policy_a"), "policy_a")

    def test_extract_base_config_id_prefers_explicit_fields(self) -> None:
        self.assertEqual(
            minimum.extract_base_config_id(
                {
                    "base_config_id": "explicit_base",
                    "config_family_id": "family",
                    "config_id": "policy_a-seed42",
                }
            ),
            "explicit_base",
        )
        self.assertEqual(
            minimum.extract_base_config_id(
                {
                    "config_family_id": "family_base",
                    "config_id": "policy_a-seed42",
                }
            ),
            "family_base",
        )

    def test_mean_seeds_per_config_averages_seeds_within_config(self) -> None:
        rows = [
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 10.0,
                "config_id": "policy_a-seed1",
                "seed": "1",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 14.0,
                "config_id": "policy_a-seed2",
                "seed": "2",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 16.0,
                "config_id": "policy_a-seed42",
                "seed": "42",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 20.0,
                "config_id": "policy_a-seed123",
                "seed": "123",
            },
        ]
        aggregated = minimum.aggregate_rows_mean_seeds_per_config(rows)
        self.assertEqual(len(aggregated), 1)
        row = aggregated[0]
        self.assertAlmostEqual(row["primary_metric_mev_mean"], 15.0)
        self.assertEqual(row["base_config_id"], "policy_a")
        self.assertEqual(row["seed_count"], 4)
        self.assertEqual(row["seeds"], ["1", "123", "2", "42"])
        self.assertEqual(
            row["config_ids"],
            [
                "policy_a-seed1",
                "policy_a-seed123",
                "policy_a-seed2",
                "policy_a-seed42",
            ],
        )
        self.assertEqual(row["aggregation_mode"], "mean_seeds_per_config")

    def test_mean_seeds_per_config_prefers_explicit_base_config_id(self) -> None:
        rows = [
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 10.0,
                "config_id": "policy_a_variant-seed42",
                "base_config_id": "policy_a",
                "seed": "42",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 14.0,
                "config_id": "policy_a_other_name-seed123",
                "base_config_id": "policy_a",
                "seed": "123",
            },
        ]

        aggregated = minimum.aggregate_rows_mean_seeds_per_config(rows)

        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["base_config_id"], "policy_a")
        self.assertAlmostEqual(aggregated[0]["primary_metric_mev_mean"], 12.0)

    def test_mean_seeds_per_config_exposes_protocol_cost_fields(self) -> None:
        rows = [
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 10.0,
                "config_id": "policy_a-seed1",
                "seed": "1",
                "gpu_hours_total": 2.0,
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 14.0,
                "config_id": "policy_a-seed2",
                "seed": "2",
                "gpu_hours_total": 6.0,
            },
        ]

        aggregated = minimum.aggregate_rows_mean_seeds_per_config(rows)

        self.assertEqual(len(aggregated), 1)
        row = aggregated[0]
        self.assertAlmostEqual(row["gpu_hours_per_seed_mean"], 4.0)
        self.assertAlmostEqual(row["gpu_hours_protocol_total"], 8.0)
        self.assertAlmostEqual(row["gpu_hours_protocol_sem"], 2.0)

    def test_mean_seeds_per_config_does_not_mix_distinct_configs(self) -> None:
        rows = [
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 10.0,
                "config_id": "policy_a-seed1",
                "seed": "1",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 14.0,
                "config_id": "policy_a-seed2",
                "seed": "2",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 8.0,
                "config_id": "policy_b-seed1",
                "seed": "1",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 16.0,
                "config_id": "policy_b-seed2",
                "seed": "2",
            },
        ]
        aggregated = minimum.aggregate_rows_mean_seeds_per_config(rows)
        self.assertEqual(len(aggregated), 2)
        by_config = {row["base_config_id"]: row["primary_metric_mev_mean"] for row in aggregated}
        self.assertAlmostEqual(by_config["policy_a"], 12.0)
        self.assertAlmostEqual(by_config["policy_b"], 12.0)

    def test_best_config_mean_picks_config_by_seed_mean_not_best_seed(self) -> None:
        rows = [
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 5.0,
                "config_id": "policy_a-seed1",
                "seed": "1",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 15.0,
                "config_id": "policy_a-seed42",
                "seed": "42",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 3.0,
                "config_id": "policy_b-seed1",
                "seed": "1",
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev": 20.0,
                "config_id": "policy_b-seed123",
                "seed": "123",
            },
        ]
        best = minimum.aggregate_rows_best_config_mean(rows)
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0]["base_config_id"], "policy_a")
        self.assertAlmostEqual(best[0]["primary_metric_mev_mean"], 10.0)
        self.assertEqual(best[0]["aggregation_mode"], "best_config_mean")
        self.assertEqual(best[0]["selection_basis"], "mean_over_seeds")

        diagnostic_best = min(rows, key=lambda row: row["primary_metric_mev"])
        self.assertEqual(diagnostic_best["config_id"], "policy_b-seed1")

    def test_mean_replicates_does_not_pick_best_value(self) -> None:
        rows = [
            {"method": "deeph", "dataset_size_x": 5, "primary_metric_mev": 10.0},
            {"method": "deeph", "dataset_size_x": 5, "primary_metric_mev": 14.0},
        ]

        aggregated = minimum.aggregate_rows_mean_replicates(rows)

        self.assertAlmostEqual(aggregated[0]["primary_metric_mev_mean"], 12.0)

    def test_n_min_cost_eff_differs_by_cost_basis(self) -> None:
        rows = [
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev_mean": 10.0,
                "gpu_hours_per_seed_mean": 3.0,
                "gpu_hours_protocol_total": 12.0,
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 20,
                "primary_metric_mev_mean": 10.4,
                "gpu_hours_per_seed_mean": 4.0,
                "gpu_hours_protocol_total": 8.0,
            },
        ]

        self.assertEqual(
            minimum.n_min_cost_eff(rows, 0.05, cost_basis="per_seed_mean"),
            10,
        )
        self.assertEqual(
            minimum.n_min_cost_eff(rows, 0.05, cost_basis="protocol_total"),
            20,
        )

    def test_n_min_cost_eff_cost_basis_falls_back_safely_for_legacy_rows(self) -> None:
        rows = [
            {
                "method": "graph2mat",
                "dataset_size_x": 10,
                "primary_metric_mev_mean": 10.0,
                "gpu_hours_total_mean": 3.0,
            },
            {
                "method": "graph2mat",
                "dataset_size_x": 20,
                "primary_metric_mev_mean": 10.4,
                "gpu_hours_total_mean": 4.0,
            },
        ]

        self.assertEqual(
            minimum.n_min_cost_eff(rows, 0.05, cost_basis="protocol_total"),
            10,
        )

    def test_observed_n_min_uses_aggregated_rows(self) -> None:
        aggregated = [
            {"method": "graph2mat", "dataset_size_x": 10, "primary_metric_mev_mean": 20.0},
            {"method": "graph2mat", "dataset_size_x": 20, "primary_metric_mev_mean": 8.0},
        ]
        raw_best = [
            {"method": "graph2mat", "dataset_size_x": 10, "primary_metric_mev_mean": 10.0},
            {"method": "graph2mat", "dataset_size_x": 10, "primary_metric_mev_mean": 30.0},
            {"method": "graph2mat", "dataset_size_x": 20, "primary_metric_mev_mean": 8.0},
        ]

        self.assertEqual(minimum.n_min_abs(aggregated, 12.0), 20)
        self.assertEqual(minimum.n_min_abs(raw_best, 12.0), 10)

    def test_fit_based_n_min_uses_aggregated_rows(self) -> None:
        aggregated = [
            {"method": "graph2mat", "dataset_size_x": 10, "primary_metric_mev_mean": 20.0},
            {"method": "graph2mat", "dataset_size_x": 20, "primary_metric_mev_mean": 8.0},
        ]
        best_rows = [
            {"method": "graph2mat", "dataset_size_x": 10, "primary_metric_mev_mean": 10.0},
            {"method": "graph2mat", "dataset_size_x": 20, "primary_metric_mev_mean": 8.0},
        ]

        agg_thresholds, _, _ = minimum.thresholds_by_method_from_fit(
            aggregated,
            threshold_mev=12.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            fit_model="linear",
        )
        best_thresholds, _, _ = minimum.thresholds_by_method_from_fit(
            best_rows,
            threshold_mev=12.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            fit_model="linear",
        )

        self.assertEqual(agg_thresholds["graph2mat"]["N_min_abs"], 17)
        self.assertEqual(best_thresholds["graph2mat"]["N_min_abs"], 10)

    def test_invalid_aggregation_mode_rejected_by_argparse(self) -> None:
        with self.assertRaises(SystemExit):
            minimum.build_parser().parse_args(
                [
                    "--run-root",
                    "/tmp/run",
                    "--output-dir",
                    "/tmp/out",
                    "--threshold-mev",
                    "10",
                    "--aggregation-mode",
                    "median",
                ]
            )

    def test_default_aggregation_mode_is_best_config_for_single_root(self) -> None:
        self.assertEqual(
            minimum.resolve_aggregation_mode(None, run_root_count=1),
            "best_config",
        )

    def test_default_aggregation_mode_is_mean_replicates_for_multiple_roots(self) -> None:
        self.assertEqual(
            minimum.resolve_aggregation_mode(None, run_root_count=2),
            "mean_replicates",
        )

    def test_aggregation_mode_metadata_marks_omitted_mode_as_legacy_inferred(self) -> None:
        metadata = minimum.resolve_aggregation_mode_metadata(None, run_root_count=1)

        self.assertIsNone(metadata["requested_aggregation_mode"])
        self.assertEqual(metadata["actual_aggregation_mode"], "best_config")
        self.assertTrue(metadata["aggregation_mode_legacy_inferred"])
        self.assertEqual(metadata["aggregation_mode_classification"], "diagnostic_only")
        self.assertIn("aggregation_mode_not_explicit", metadata["aggregation_mode_warning"])

    def test_aggregation_mode_classification_for_paper_ready_modes(self) -> None:
        self.assertEqual(
            minimum.aggregation_mode_classification("mean_seeds_per_config"),
            ("paper_candidate", "paper_ready_seed_mean_per_config"),
        )
        self.assertEqual(
            minimum.aggregation_mode_classification("best_config_mean"),
            ("paper_candidate", "paper_candidate_only_if_config_selection_policy_is_locked"),
        )

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

    def test_n_min_rel_tol_matches_legacy_rel95_alias(self) -> None:
        rows = [
            {"method": "deeph", "dataset_size_x": 10, "primary_metric_mev_mean": 20.0},
            {"method": "deeph", "dataset_size_x": 30, "primary_metric_mev_mean": 10.4},
            {"method": "deeph", "dataset_size_x": 50, "primary_metric_mev_mean": 10.0},
        ]

        self.assertEqual(minimum.n_min_rel_tol(rows, 0.05), 30)
        self.assertEqual(minimum.n_min_rel95(rows, 0.05), 30)

    def test_thresholds_include_canonical_rel_tol_and_deprecated_alias(self) -> None:
        rows = [
            {"method": "deeph", "dataset_size_x": 10, "primary_metric_mev_mean": 20.0},
            {"method": "deeph", "dataset_size_x": 30, "primary_metric_mev_mean": 10.4},
            {"method": "deeph", "dataset_size_x": 50, "primary_metric_mev_mean": 10.0},
        ]

        thresholds = minimum.thresholds_by_method(
            rows,
            threshold_mev=12.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
        )
        row = thresholds["deeph"]

        self.assertEqual(row["N_min_rel_tol"], 30)
        self.assertEqual(row["N_min_rel95"], 30)
        self.assertEqual(row["N_min_rel95_deprecated_alias_for"], "N_min_rel_tol")

    def test_primary_ui_and_report_labels_use_rel_tol_not_rel95(self) -> None:
        html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn("N_min_rel_tol", html)
        self.assertNotIn("rel95", html)

        spec = importlib.util.spec_from_file_location(
            "pipeline_ui_rel_tol_labels_test",
            SCRIPTS_DIR / "pipeline_ui.py",
        )
        assert spec and spec.loader
        pipeline_ui = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pipeline_ui
        spec.loader.exec_module(pipeline_ui)
        criteria = {item["id"]: item["label"] for item in pipeline_ui.DATASET_SIZE_MINIMUM_CRITERIA}
        self.assertIn("N_min_rel_tol", criteria)
        self.assertNotIn("N_min_rel95", criteria)

        report = minimum.build_report(
            output_dir=Path("/tmp/out"),
            run_roots=[],
            grouped_rows=[],
            best_rows=[],
            thresholds={"deeph": {"N_min_rel_tol": 30}},
            fits={},
            warnings=[],
            primary_metric="h_mae_eV_mean",
            threshold_mev=10.0,
            x_axis="n_train",
            temporal_diagnostics={},
        )
        self.assertIn("N_min_rel_tol", report)
        self.assertNotIn("N_min_rel95", report)

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

    def test_normalize_reads_referenced_metric_manifest_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics" / "manifest.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                json.dumps({"summary": {"kpoint_matrix": {"h_mae_eV": {"mean": 0.021}}}}),
                encoding="utf-8",
            )
            raw = [
                {
                    "model": "graph2mat",
                    "dataset_size": 10,
                    "config_id": "cfg",
                    "validation_metrics_path": str(metrics_path),
                }
            ]

            rows, warnings = minimum.normalize_rows(raw, primary_metric="h_mae_eV_mean", x_axis="n_total")

            self.assertEqual(warnings, [])
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["primary_metric_mev"], 21.0)

    def test_normalize_reads_referenced_kpoint_matrix_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics" / "manifest.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps({"stage": "metrics"}), encoding="utf-8")
            (metrics_path.parent / "kpoint_matrix_metrics.csv").write_text(
                "sample,h_mae_eV,h_rmse_eV\n"
                "a,0.010,0.020\n"
                "b,0.014,0.030\n",
                encoding="utf-8",
            )
            raw = [
                {
                    "model": "deeph",
                    "dataset_size": 10,
                    "config_id": "cfg",
                    "validation_metrics_path": str(metrics_path),
                }
            ]

            rows, warnings = minimum.normalize_rows(raw, primary_metric="h_mae_eV_mean", x_axis="n_total")

            self.assertEqual(warnings, [])
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["primary_metric_mev"], 12.0)

    def test_explicit_run_root_with_corrupt_preferred_json_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            metrics_path = run_root / "summary" / "ranking" / "normalized_run_metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text('{"metric_scaling_rows": [', encoding="utf-8")

            with self.assertRaises(minimum.MetricFileLoadError) as exc:
                minimum.load_run_root_rows(run_root, explicit_run_root_mode=True)

            message = str(exc.exception)
            self.assertIn(str(metrics_path), message)
            self.assertIn("invalid_json", message)
            self.assertIn("JSONDecodeError", message)
            self.assertIn("mode=explicit_run_root", message)

    def test_explicit_run_root_with_missing_preferred_file_and_valid_fallback_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            csv_path = run_root / "sweep" / "training_sweep_metrics.csv"
            write_csv(
                csv_path,
                [
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "cfg-seed1",
                        "epoch_label": "10 epochs",
                        "h_mae_eV_mean": 0.012,
                    }
                ],
            )

            rows, sources, warnings = minimum.load_run_root_rows(
                run_root,
                explicit_run_root_mode=True,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(sources, [str(csv_path)])
            self.assertEqual(warnings, [])

    def test_discovery_mode_skips_invalid_candidate_when_valid_candidate_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            bad_path = run_root / "summary" / "ranking" / "normalized_run_metrics.json"
            bad_path.parent.mkdir(parents=True)
            bad_path.write_text('{"metric_scaling_rows": [', encoding="utf-8")
            good_path = run_root / "sweep" / "training_sweep_metrics.csv"
            write_csv(
                good_path,
                [
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "cfg-seed1",
                        "epoch_label": "10 epochs",
                        "h_mae_eV_mean": 0.015,
                    }
                ],
            )

            discovered = minimum.discover_metric_files(run_root)
            rows, sources, warnings = minimum.load_run_root_rows(
                run_root,
                explicit_run_root_mode=False,
            )

            self.assertEqual(discovered, [good_path])
            self.assertEqual(len(rows), 1)
            self.assertEqual(sources, [str(good_path)])
            self.assertEqual(warnings, [])

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
            args = make_analyze_args(
                run_root=[str(root)],
                output_dir=str(output_dir),
                aggregation_mode="best_config",
            )

            summary = minimum.analyze(args)

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["aggregation_mode"], "best_config")
            self.assertEqual(summary["thresholds"]["graph2mat"]["N_min_abs"], 20)
            self.assertIn("threshold_sensitivity", summary)
            self.assertEqual(summary["threshold_sensitivity"]["status"], "ok")
            self.assertTrue((output_dir / "dataset_size_minimum_results.csv").exists())
            self.assertTrue((output_dir / "dataset_size_minimum_summary.json").exists())
            self.assertTrue((output_dir / "dataset_size_minimum_report.md").exists())

    def test_analyze_summary_records_threshold_protocol_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "g10",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.009,
                    }
                ]
            }
            (root / "summary" / "ranking").mkdir(parents=True)
            (root / "summary" / "ranking" / "normalized_run_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            summary = minimum.analyze(
                make_analyze_args(
                    run_root=[str(root)],
                    output_dir=str(base / "out"),
                    primary_metric="h_mae_eV_mean",
                    threshold_mev=10.0,
                    aggregation_mode="mean_seeds_per_config",
                    n_min_source="fit",
                    n_min_fit_model="power_law_floor",
                    threshold_preset_key="h_mae_relaxed_10",
                    threshold_is_user_defined=False,
                )
            )

            self.assertEqual(summary["threshold_basis"], minimum.THRESHOLD_BASIS_EXPLORATORY_PRESET)
            self.assertEqual(summary["threshold_reference"], minimum.DEFAULT_THRESHOLD_REFERENCE)
            self.assertEqual(summary["threshold_metric_family"], "hamiltonian_element_error_mev")
            self.assertFalse(summary["threshold_is_user_defined"])

    def test_analyze_mean_replicates_across_duplicate_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "good",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.010,
                    },
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "bad",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.030,
                    },
                    {
                        "method": "graph2mat",
                        "dataset_size": 20,
                        "config_id": "good",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.005,
                    },
                ]
            }
            (root / "summary" / "ranking").mkdir(parents=True)
            (root / "summary" / "ranking" / "normalized_run_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            output_dir = base / "out"
            args = make_analyze_args(
                run_root=[str(root)],
                output_dir=str(output_dir),
                aggregation_mode="mean_replicates",
            )

            summary = minimum.analyze(args)

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["aggregation_mode"], "mean_replicates")
            self.assertEqual(summary["raw_rows_count"], 3)
            self.assertEqual(len(summary["aggregated_rows"]), 2)
            by_size = {int(row["dataset_size_x"]): row for row in summary["aggregated_rows"]}
            self.assertAlmostEqual(by_size[10]["primary_metric_mev_mean"], 20.0)
            self.assertEqual(by_size[10]["replicate_count"], 2)
            self.assertEqual(summary["thresholds"]["graph2mat"]["N_min_abs"], 20)
            self.assertIn("N_min_rel_tol", summary["thresholds"]["graph2mat"])
            self.assertEqual(
                summary["thresholds"]["graph2mat"]["N_min_rel95_deprecated_alias_for"],
                "N_min_rel_tol",
            )
            self.assertEqual(
                summary["deprecated_threshold_aliases"],
                {"N_min_rel95": "N_min_rel_tol"},
            )

    def test_analyze_without_aggregation_mode_warns_and_records_legacy_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "good",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.010,
                    },
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "bad",
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
            args = make_analyze_args(
                run_root=[str(root)],
                output_dir=str(output_dir),
            )

            summary = minimum.analyze(args)

            self.assertEqual(summary["aggregation_mode"], "best_config")
            self.assertIsNone(summary["requested_aggregation_mode"])
            self.assertEqual(summary["actual_aggregation_mode"], "best_config")
            self.assertTrue(summary["aggregation_mode_legacy_inferred"])
            self.assertEqual(summary["aggregation_mode_classification"], "diagnostic_only")
            self.assertIn("aggregation_mode_not_explicit", " ".join(summary["warnings"]))
            self.assertEqual(len(summary["aggregated_rows"]), 1)
            self.assertAlmostEqual(summary["aggregated_rows"][0]["primary_metric_mev_mean"], 10.0)

    def test_bootstrap_disabled_leaves_enabled_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "g10",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.020,
                    },
                ]
            }
            (root / "summary" / "ranking").mkdir(parents=True)
            (root / "summary" / "ranking" / "normalized_run_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            summary = minimum.analyze(
                make_analyze_args(
                    run_root=[str(root)],
                    output_dir=str(base / "out"),
                    aggregation_mode="best_config",
                    bootstrap_replicates=0,
                )
            )

            self.assertFalse(summary["bootstrap"]["enabled"])
            self.assertFalse(summary["replicate_bootstrap"]["enabled"])
            self.assertEqual(summary["replicate_bootstrap"]["display_label"], "replicate-resampling CI")
            self.assertEqual(summary["bootstrap"]["deprecated_alias_for"], "replicate_bootstrap")
            self.assertEqual(summary["bootstrap_replicates"], 0)
            self.assertIn("hierarchical_uncertainty", summary)

    def test_disabled_bootstrap_summary_uses_replicate_resampling_label(self) -> None:
        summary = minimum.disabled_bootstrap_summary()

        self.assertEqual(summary["bootstrap_type"], "replicate_resampling")
        self.assertEqual(summary["display_label"], "replicate-resampling CI")
        self.assertIn("does_not_model_temporal_autocorrelation", summary["limitations"])

    def test_bootstrap_is_deterministic_for_fixed_seed(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=25.0, seed="a1"),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=15.0, seed="a2"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=8.0, seed="b1"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=12.0, seed="b2"),
        ]
        kwargs = {
            "methods": ["graph2mat"],
            "n_replicates": 80,
            "seed": 4242,
            "ci_level": 0.95,
            "aggregation_mode": "mean_replicates",
            "threshold_mev": 10.0,
            "relative_tolerance": 0.05,
            "plateau_gain": 0.05,
            "n_min_source": "observed",
            "n_min_fit_model": "linear",
            "moving_average_window": 3,
        }
        first = minimum.compute_bootstrap_n_min(rows, **kwargs)
        second = minimum.compute_bootstrap_n_min(rows, **kwargs)

        self.assertEqual(
            first["by_method"]["graph2mat"]["N_min_abs"],
            second["by_method"]["graph2mat"]["N_min_abs"],
        )

    def test_observed_bootstrap_produces_n_min_abs_ci(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=25.0, seed="a1"),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=15.0, seed="a2"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=8.0, seed="b1"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=12.0, seed="b2"),
        ]
        result = minimum.compute_bootstrap_n_min(
            rows,
            methods=["graph2mat"],
            n_replicates=120,
            seed=7,
            ci_level=0.95,
            aggregation_mode="mean_replicates",
            threshold_mev=10.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            n_min_source="observed",
            n_min_fit_model="linear",
            moving_average_window=3,
        )

        ci = result["by_method"]["graph2mat"]["N_min_abs"]
        self.assertTrue(result["enabled"])
        self.assertGreater(ci["n_bootstrap_successful"], 0)
        self.assertIsNotNone(ci["median"])
        self.assertIsNotNone(ci["lower"])
        self.assertIsNotNone(ci["upper"])
        self.assertIn("N_min_rel_tol", result["criteria"])
        self.assertNotIn("N_min_rel95", result["criteria"])
        self.assertIn("N_min_rel_tol", result["by_method"]["graph2mat"])
        self.assertEqual(
            result["by_method"]["graph2mat"]["N_min_rel95"]["deprecated_alias_for"],
            "N_min_rel_tol",
        )

    def test_fit_bootstrap_produces_ci_for_linear_fit(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=30.0, seed="a1"),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=20.0, seed="a2"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=14.0, seed="b1"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=10.0, seed="b2"),
            make_normalized_replicate_row(method="graph2mat", size=40, mev=6.0, seed="c1"),
            make_normalized_replicate_row(method="graph2mat", size=40, mev=8.0, seed="c2"),
        ]
        result = minimum.compute_bootstrap_n_min(
            rows,
            methods=["graph2mat"],
            n_replicates=60,
            seed=99,
            ci_level=0.95,
            aggregation_mode="mean_replicates",
            threshold_mev=12.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            n_min_source="fit",
            n_min_fit_model="linear",
            moving_average_window=3,
        )

        ci = result["by_method"]["graph2mat"]["N_min_abs"]
        self.assertGreater(result["replicates_successful"], 0)
        self.assertIsNotNone(ci["median"])
        self.assertEqual(result["display_label"], "replicate-resampling CI")
        self.assertIn("does_not_model_temporal_autocorrelation", result["limitations"])

    def test_bootstrap_too_few_replicates_warns_without_crashing(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=20.0, seed="only"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=8.0, seed="only2"),
        ]
        result = minimum.compute_bootstrap_n_min(
            rows,
            methods=["graph2mat"],
            n_replicates=50,
            seed=1,
            ci_level=0.95,
            aggregation_mode="mean_replicates",
            threshold_mev=10.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            n_min_source="observed",
            n_min_fit_model="linear",
            moving_average_window=3,
        )

        self.assertIn("bootstrap_unavailable_no_replicates", result["warnings"])
        self.assertIn("replicate_bootstrap_no_multiple_seeds_or_replicates", result["warnings"])
        self.assertIn("replicate_bootstrap_no_temporal_or_block_bootstrap", result["warnings"])
        self.assertEqual(result["by_method"], {})

    def test_replicate_bootstrap_scope_warnings_for_diagnostic_aggregation(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=20.0, seed="a1"),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=18.0, seed="a2"),
        ]
        warnings = minimum.replicate_bootstrap_scope_warnings(rows, aggregation_mode="mean_replicates")

        self.assertIn("replicate_bootstrap_row_level_replicates_only", warnings)
        self.assertIn("replicate_bootstrap_no_temporal_or_block_bootstrap", warnings)
        self.assertIn("replicate_bootstrap_does_not_capture_model_selection_uncertainty", warnings)
        self.assertIn("replicate_bootstrap_selected_aggregation_is_diagnostic:mean_replicates", warnings)
        self.assertIn("replicate_bootstrap_excludes_n_min_cost_eff", warnings)

    def test_bootstrap_explicitly_excludes_cost_eff_ci_when_cost_available(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=10.0, seed="1", gpu_hours_total=1.0),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=9.0, seed="2", gpu_hours_total=1.2),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=8.0, seed="1", gpu_hours_total=2.0),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=7.7, seed="2", gpu_hours_total=2.1),
        ]

        result = minimum.compute_bootstrap_n_min(
            rows,
            methods=["graph2mat"],
            n_replicates=20,
            seed=7,
            ci_level=0.95,
            aggregation_mode="mean_seeds_per_config",
            threshold_mev=10.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            n_min_source="observed",
            n_min_fit_model="linear",
            moving_average_window=3,
            cost_basis="per_seed_mean",
        )

        self.assertFalse(result["cost_eff_ci_available"])
        self.assertEqual(result["cost_eff_ci_policy"], "excluded_no_joint_metric_cost_resampling")
        self.assertIn("replicate_bootstrap_excludes_n_min_cost_eff", result["warnings"])
        self.assertNotIn("N_min_cost_eff", result["criteria"])

    def test_bootstrap_exposes_missing_cost_warning_for_selected_basis(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=10.0, seed="1", gpu_hours_total=1.0),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=9.0, seed="2"),
        ]

        result = minimum.compute_bootstrap_n_min(
            rows,
            methods=["graph2mat"],
            n_replicates=10,
            seed=11,
            ci_level=0.95,
            aggregation_mode="mean_seeds_per_config",
            threshold_mev=10.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            n_min_source="observed",
            n_min_fit_model="linear",
            moving_average_window=3,
            cost_basis="protocol_total",
        )

        self.assertFalse(result["cost_eff_ci_available"])
        self.assertGreater(result["cost_eff_rows_missing_cost"], 0)
        self.assertIn(
            "n_min_cost_eff_missing_cost_rows_for_selected_basis:protocol_total:1",
            result["warnings"],
        )

    def test_fit_bootstrap_counts_failures(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=30.0, seed="a1"),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=20.0, seed="a2"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=12.0, seed="b1"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=10.0, seed="b2"),
        ]
        result = minimum.compute_bootstrap_n_min(
            rows,
            methods=["graph2mat"],
            n_replicates=20,
            seed=3,
            ci_level=0.95,
            aggregation_mode="mean_replicates",
            threshold_mev=12.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            n_min_source="fit",
            n_min_fit_model="quadratic",
            moving_average_window=3,
        )

        self.assertGreater(result["replicates_failed"], 0)
        self.assertTrue(result["failure_counts"])
        self.assertTrue(result["failure_reasons"])

    def test_hierarchical_uncertainty_seed_level_resampling_with_multiple_seeds(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=20.0, seed="1", config_id="cfgA-seed1"),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=22.0, seed="2", config_id="cfgA-seed2"),
        ]

        result = minimum.compute_hierarchical_uncertainty(
            rows,
            temporal_diagnostics={},
            fits={},
            fit_threshold_details={},
            fit_predictive_stability_by_left_out_N=None,
            requested_fit_model="power_law_floor",
            actual_fit_model="power_law_floor",
            actual_n_min_source="fit",
            seed=7,
            ci_level=0.9,
            n_replicates=20,
        )

        seed_level = result["levels"]["seed"]
        self.assertTrue(seed_level["available"])
        self.assertTrue(seed_level["sufficient"])
        cfg = seed_level["by_method"]["graph2mat"]["10"]["cfgA"]
        self.assertEqual(cfg["seed_count"], 2)
        self.assertIsNotNone(cfg["metric_ci_mev"])
        self.assertEqual(cfg["metric_ci_mev"]["ci_level"], 0.9)

    def test_hierarchical_uncertainty_config_level_resampling_with_multiple_configs(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=20.0, seed="1", config_id="cfgA-seed1"),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=21.0, seed="2", config_id="cfgA-seed2"),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=24.0, seed="1", config_id="cfgB-seed1"),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=25.0, seed="2", config_id="cfgB-seed2"),
        ]

        result = minimum.compute_hierarchical_uncertainty(
            rows,
            temporal_diagnostics={},
            fits={},
            fit_threshold_details={},
            fit_predictive_stability_by_left_out_N=None,
            requested_fit_model="power_law_floor",
            actual_fit_model="power_law_floor",
            actual_n_min_source="fit",
            seed=9,
            ci_level=0.9,
            n_replicates=20,
        )

        config_level = result["levels"]["config"]
        self.assertTrue(config_level["available"])
        self.assertTrue(config_level["sufficient"])
        entry = config_level["by_method"]["graph2mat"]["10"]
        self.assertEqual(entry["config_count"], 2)
        self.assertEqual(sorted(entry["config_ids"]), ["cfgA", "cfgB"])
        self.assertIsNotNone(entry["metric_ci_mev"])

    def test_hierarchical_uncertainty_block_level_resampling_with_blocks(self) -> None:
        specs = []
        for index in range(6):
            specs.append(
                {
                    "metadata": {
                        "frame_index": index,
                        "source_frame_index": index,
                        "block_id": "block_a",
                        "temperature_K": "300",
                        "total_energy_eV": float(index),
                    }
                }
            )
        for index in range(6):
            specs.append(
                {
                    "metadata": {
                        "frame_index": index,
                        "source_frame_index": index,
                        "block_id": "block_b",
                        "temperature_K": "300",
                        "total_energy_eV": float(100 + index),
                    }
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_temporal_dataset_from_specs(Path(tmp) / "dataset", specs=specs)
            temporal = minimum.summarize_temporal_diagnostics(
                [
                    {
                        "dataset_root": str(dataset_root),
                        "dataset_size_x": 10,
                        "method": "graph2mat",
                        "primary_metric_mev_mean": 20.0,
                    }
                ],
                run_roots=[],
            )
            result = minimum.compute_hierarchical_uncertainty(
                [],
                temporal_diagnostics=temporal,
                fits={},
                fit_threshold_details={},
                fit_predictive_stability_by_left_out_N=None,
                requested_fit_model="power_law_floor",
                actual_fit_model="power_law_floor",
                actual_n_min_source="fit",
                seed=11,
                ci_level=0.9,
                n_replicates=20,
            )

        block_level = result["levels"]["block"]
        self.assertTrue(block_level["available"])
        self.assertTrue(block_level["sufficient"])
        dataset_entry = next(iter(block_level["by_dataset"].values()))
        self.assertEqual(dataset_entry["block_count"], 2)
        self.assertIsNotNone(dataset_entry["n_eff_over_n_nominal_ci"])

    def test_hierarchical_uncertainty_missing_hierarchy_metadata_is_diagnostic_only(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=20.0, seed="1", config_id="cfgA-seed1"),
        ]
        result = minimum.compute_hierarchical_uncertainty(
            rows,
            temporal_diagnostics={},
            fits={},
            fit_threshold_details={},
            fit_predictive_stability_by_left_out_N=None,
            requested_fit_model="power_law_floor",
            actual_fit_model="power_law_floor",
            actual_n_min_source="fit",
            seed=5,
            ci_level=0.9,
            n_replicates=20,
        )

        self.assertEqual(result["status"], "diagnostic_only")
        self.assertIn("paper_uncertainty_block_hierarchy_unavailable", result["paper_level_blockers"])

    def test_hierarchical_uncertainty_is_deterministic_for_fixed_seed(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=20.0, seed="1", config_id="cfgA-seed1"),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=22.0, seed="2", config_id="cfgA-seed2"),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=24.0, seed="1", config_id="cfgB-seed1"),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=25.0, seed="2", config_id="cfgB-seed2"),
        ]
        kwargs = dict(
            temporal_diagnostics={},
            fits={},
            fit_threshold_details={},
            fit_predictive_stability_by_left_out_N=None,
            requested_fit_model="power_law_floor",
            actual_fit_model="power_law_floor",
            actual_n_min_source="fit",
            seed=17,
            ci_level=0.9,
            n_replicates=20,
        )

        first = minimum.compute_hierarchical_uncertainty(rows, **kwargs)
        second = minimum.compute_hierarchical_uncertainty(rows, **kwargs)
        self.assertEqual(first, second)

    def test_fit_policy_classification_metadata(self) -> None:
        n_values = [10.0, 20.0, 40.0]
        y_values = [30.0, 18.0, 12.0]

        power = minimum.fit_power_law_floor(n_values, y_values)
        linear = minimum.fit_linear_model("linear", n_values, y_values)
        none = minimum.no_fit_summary(len(n_values))
        lowess = minimum.fit_lowess_model("monotone_lowess_logx", n_values, y_values)

        self.assertFalse(power["paper_candidate"])
        self.assertTrue(power["diagnostic_only"])
        self.assertEqual(power["fit_policy"], "diagnostic_only")
        self.assertEqual(
            power["fit_policy_reason"],
            f"power_law_floor_points_lt_{minimum.MIN_FIT_POINTS_FOR_PAPER_CANDIDATE}",
        )
        self.assertFalse(linear["paper_candidate"])
        self.assertTrue(linear["diagnostic_only"])
        self.assertTrue(none["diagnostic_only"])
        self.assertTrue(lowess["diagnostic_only"])
        self.assertIn("diagnostic_fit_numerical_policy", linear)
        self.assertIn("scaled_fit_domain", linear)

    def test_linear_diagnostic_fit_reports_numerical_metadata_for_large_n(self) -> None:
        n_values = [1_000_000.0, 1_100_000.0, 1_200_000.0, 1_300_000.0]
        y_values = [20.0, 18.0, 17.0, 16.0]

        fit = minimum.fit_linear_model("linear", n_values, y_values)

        self.assertEqual(fit["status"], "ok")
        self.assertIn("fit_condition_estimate", fit)
        self.assertIn("scaled_fit_domain", fit)
        self.assertEqual(
            fit["diagnostic_fit_numerical_policy"],
            "numpy_lstsq_column_center_scale_v1" if minimum.np is not None else "normal_equations_column_center_scale_v1",
        )

    def test_quadratic_fit_becomes_diagnostic_unstable_when_rank_deficient(self) -> None:
        n_values = [100.0, 100.0, 100.0]
        y_values = [5.0, 5.1, 4.9]

        fit = minimum.fit_linear_model("quadratic", n_values, y_values)

        self.assertEqual(fit["status"], "diagnostic_unstable")
        self.assertEqual(fit["error"], "diagnostic_fit_numerically_unstable")
        self.assertIn(fit["condition_warning"], {"rank_deficient_scaled_design", "ill_conditioned_scaled_design"})
        self.assertTrue(fit["diagnostic_only"])

    def test_inverse_square_fit_detects_ill_conditioning_for_nearly_identical_sizes(self) -> None:
        n_values = [10_000.0, 10_001.0, 10_002.0, 10_003.0]
        y_values = [3.0, 2.999, 2.9985, 2.998]

        fit = minimum.fit_linear_model("inverse_square", n_values, y_values)

        self.assertIn("fit_condition_estimate", fit)
        self.assertIn("scaled_fit_domain", fit)
        self.assertTrue(fit["diagnostic_only"])

    def test_fitted_curve_rows_do_not_silently_use_unstable_quadratic_fit(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=100, mev=5.0, seed="1", config_id="a"),
            make_normalized_replicate_row(method="graph2mat", size=100, mev=5.1, seed="2", config_id="b"),
            make_normalized_replicate_row(method="graph2mat", size=100, mev=4.9, seed="3", config_id="c"),
        ]

        curve_rows, fit = minimum.fitted_curve_rows(rows, fit_model="quadratic")

        self.assertEqual(curve_rows, [])
        self.assertEqual(fit["status"], "diagnostic_unstable")
        self.assertEqual(fit["error"], "diagnostic_fit_numerically_unstable")

    def test_power_law_floor_with_five_points_can_be_paper_candidate(self) -> None:
        n_values = [10.0, 20.0, 40.0, 80.0, 160.0]
        y_values = [30.0, 18.0, 12.0, 9.0, 7.5]

        power = minimum.fit_power_law_floor(n_values, y_values)

        self.assertEqual(power["status"], "ok")
        self.assertTrue(power["paper_candidate"])
        self.assertFalse(power["diagnostic_only"])
        self.assertEqual(power["fit_policy"], "paper_candidate")
        self.assertIn("alpha_search_method", power)
        self.assertIn("objective_evaluations", power)

    def test_negative_unconstrained_fit_predictions_fall_back_to_observed_thresholds(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=0.1, seed="1", config_id="a"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=0.1, seed="1", config_id="b"),
            make_normalized_replicate_row(method="graph2mat", size=30, mev=0.1, seed="1", config_id="c"),
            make_normalized_replicate_row(method="graph2mat", size=40, mev=1.0, seed="1", config_id="d"),
        ]

        curve_rows, fit = minimum.fitted_curve_rows(rows, fit_model="quadratic")
        self.assertEqual(curve_rows, [])
        self.assertEqual(fit["status"], "invalid_negative_predictions")
        self.assertTrue(fit["invalid_for_n_min_thresholding"])

        thresholds, fit_details, warnings = minimum.thresholds_by_method_from_fit(
            rows,
            threshold_mev=0.5,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            fit_model="quadratic",
        )

        self.assertEqual(fit_details["graph2mat"]["status"], "invalid_negative_predictions")
        self.assertEqual(thresholds["graph2mat"]["N_min_abs"], 10)
        self.assertEqual(thresholds["graph2mat"]["N_min_abs_source"], "observed_invalid_fit")
        self.assertIn("fit_negative_predictions_observed_fallback:graph2mat:quadratic", warnings)

    def test_invalid_bootstrap_args_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            minimum.build_parser().parse_args(
                [
                    "--run-root",
                    "/tmp/run",
                    "--output-dir",
                    "/tmp/out",
                    "--threshold-mev",
                    "10",
                    "--bootstrap-replicates",
                    "-1",
                ]
            )

    def test_build_parser_rejects_invalid_n_min_fit_model(self) -> None:
        with self.assertRaises(SystemExit):
            minimum.build_parser().parse_args(
                [
                    "--run-root",
                    "/tmp/run",
                    "--output-dir",
                    "/tmp/out",
                    "--threshold-mev",
                    "10",
                    "--n-min-fit-model",
                    "not_a_model",
                ]
            )
        with self.assertRaises(SystemExit):
            minimum.build_parser().parse_args(
                [
                    "--run-root",
                    "/tmp/run",
                    "--output-dir",
                    "/tmp/out",
                    "--threshold-mev",
                    "10",
                    "--ci-level",
                    "1.0",
                ]
            )

    def test_build_parser_accepts_cost_basis(self) -> None:
        args = minimum.build_parser().parse_args(
            [
                "--run-root",
                "/tmp/run",
                "--output-dir",
                "/tmp/out",
                "--threshold-mev",
                "10",
                "--cost-basis",
                "protocol_total",
            ]
        )
        self.assertEqual(args.cost_basis, "protocol_total")

    def test_parse_claim_mode_defaults_and_accepts_paper_candidate(self) -> None:
        self.assertEqual(minimum.parse_claim_mode(None), "diagnostic")
        self.assertEqual(minimum.parse_claim_mode(""), "diagnostic")
        self.assertEqual(minimum.parse_claim_mode("paper_candidate"), "paper_candidate")

    def test_build_parser_accepts_claim_mode(self) -> None:
        args = minimum.build_parser().parse_args(
            [
                "--run-root",
                "/tmp/run",
                "--output-dir",
                "/tmp/out",
                "--threshold-mev",
                "10",
                "--claim-mode",
                "paper_candidate",
            ]
        )
        self.assertEqual(args.claim_mode, "paper_candidate")

    def test_build_parser_rejects_invalid_claim_mode(self) -> None:
        with self.assertRaises(SystemExit):
            minimum.build_parser().parse_args(
                [
                    "--run-root",
                    "/tmp/run",
                    "--output-dir",
                    "/tmp/out",
                    "--threshold-mev",
                    "10",
                    "--claim-mode",
                    "paper",
                ]
            )

    def test_build_parser_rejects_invalid_cost_basis(self) -> None:
        with self.assertRaises(SystemExit):
            minimum.build_parser().parse_args(
                [
                    "--run-root",
                    "/tmp/run",
                    "--output-dir",
                    "/tmp/out",
                    "--threshold-mev",
                    "10",
                    "--cost-basis",
                    "average_total",
                ]
            )

    def test_bootstrap_summary_is_json_serializable(self) -> None:
        rows = [
            make_normalized_replicate_row(method="graph2mat", size=10, mev=25.0, seed="a1"),
            make_normalized_replicate_row(method="graph2mat", size=10, mev=15.0, seed="a2"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=8.0, seed="b1"),
            make_normalized_replicate_row(method="graph2mat", size=20, mev=12.0, seed="b2"),
        ]
        result = minimum.compute_bootstrap_n_min(
            rows,
            methods=["graph2mat"],
            n_replicates=30,
            seed=11,
            ci_level=0.95,
            aggregation_mode="mean_replicates",
            threshold_mev=10.0,
            relative_tolerance=0.05,
            plateau_gain=0.05,
            n_min_source="observed",
            n_min_fit_model="linear",
            moving_average_window=3,
        )
        serialized = json.dumps(minimum.json_safe(result))
        self.assertIn("N_min_abs", serialized)

    def test_script_does_not_import_subprocess_or_build_compute_commands(self) -> None:
        source = (SCRIPTS_DIR / "g2m_deeph_dataset_size_minimum.py").read_text(encoding="utf-8")

        self.assertNotIn("import subprocess", source)
        self.assertNotIn("os.system", source)
        self.assertNotIn("Popen", source)
        self.assertIn("FORBIDDEN_COMPUTE_COMMANDS", source)


class DatasetSizeMinimumUiApiTests(unittest.TestCase):
    _analysis_counter = 0

    def _write_smoke_run_root(
        self,
        base: Path,
        *,
        dataset_root: Path | None = None,
        config_prefix: str = "policy_a",
    ) -> Path:
        run_root = base / "smoke_sweep"
        metrics_path = run_root / "summary" / "ranking" / "normalized_run_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for dataset_size, metric_value in ((10, 0.030), (20, 0.018), (40, 0.011), (80, 0.008), (160, 0.006)):
            rows.append(
                {
                    "method": "graph2mat",
                    "dataset_size": dataset_size,
                    "dataset_size_train": dataset_size,
                    "dataset_size_total": dataset_size,
                    "dataset_root": str(dataset_root) if dataset_root is not None else "",
                    "config_id": f"{config_prefix}-N{dataset_size}-seed1",
                    "epoch_label": "10 epochs",
                    "metric_key": "h_mae_eV_mean",
                    "metric_value": metric_value,
                    "seed": "1",
                }
            )
        metrics_path.write_text(json.dumps({"metric_scaling_rows": rows}), encoding="utf-8")
        return run_root

    def _run_analysis_via_helper(
        self,
        *,
        payload: dict[str, object],
    ) -> dict[str, object]:
        def recording_run(command, **kwargs):
            self.assertEqual(Path(command[1]).resolve(), self.pipeline_ui.DATASET_SIZE_MINIMUM_SCRIPT.resolve())
            joined = " ".join(command)
            for forbidden in ("siesta", "deeph-train", "deeph-preprocess", "deeph-inference", "gnubands"):
                self.assertNotIn(forbidden, joined)
            output_dir = Path(command[command.index("--output-dir") + 1])
            args = minimum.build_parser().parse_args(command[2:])
            summary = minimum.analyze(args)
            self.assertEqual(Path(summary["outputs"][-1]), output_dir / "dataset_size_minimum_summary.json")
            return MagicMock(
                returncode=0,
                stdout=json.dumps({"status": summary["status"], "output_dir": str(output_dir)}),
                stderr="",
            )

        type(self)._analysis_counter += 1
        analysis_root = (
            Path(payload["run_roots"][0]).resolve().parent
            / f"dataset_size_minimum_helper_results_{type(self)._analysis_counter:04d}"
        )
        original_results_root = self.pipeline_ui.RESULTS_ROOT
        try:
            self.pipeline_ui.RESULTS_ROOT = analysis_root
            with patch.object(self.pipeline_ui.subprocess, "run", side_effect=recording_run):
                return self.pipeline_ui.run_dataset_size_minimum_analysis(payload)
        finally:
            self.pipeline_ui.RESULTS_ROOT = original_results_root

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

    def test_resolve_run_roots_fails_clearly_for_corrupt_explicit_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            metrics_path = run_root / "summary" / "ranking" / "normalized_run_metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text('{"metric_scaling_rows": [', encoding="utf-8")

            with self.assertRaises(RuntimeError) as exc:
                self.pipeline_ui.resolve_dataset_size_minimum_run_roots([str(run_root)])

            message = str(exc.exception)
            self.assertIn(str(metrics_path.resolve()), message)
            self.assertIn("invalid_json", message)
            self.assertIn("JSONDecodeError", message)
            self.assertIn("mode=explicit_run_root", message)

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
            self.assertEqual(preview["aggregation_mode"], "mean_replicates")
            self.assertEqual(len(preview["best_rows"]), 1)
            self.assertAlmostEqual(preview["best_rows"][0]["primary_metric_mev_mean"], 30.0)

    def _write_minimum_run_root(self, base: Path, name: str = "finished_sweep") -> Path:
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
                            "metric_value": 0.02,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return run_root

    def test_run_analysis_command_includes_aggregation_mode_and_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_root = self._write_minimum_run_root(base)

            def fake_run(command, **kwargs):
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "dataset_size_minimum_summary.json").write_text(
                    json.dumps({"status": "ok"}),
                    encoding="utf-8",
                )
                return MagicMock(returncode=0, stdout="", stderr="")

            captured_command: list[str] = []

            def recording_run(command, **kwargs):
                captured_command[:] = list(command)
                return fake_run(command, **kwargs)

            with patch.object(self.pipeline_ui.subprocess, "run", side_effect=recording_run):
                self.pipeline_ui.run_dataset_size_minimum_analysis(
                    {
                        "run_roots": [str(run_root)],
                        "threshold_mev": 10.0,
                        "aggregation_mode": "mean_replicates",
                        "moving_average_window": 5,
                        "cost_basis": "protocol_total",
                        "n_min_source": "fit",
                        "n_min_fit_model": "linear",
                    }
                )

            command = captured_command
            self.assertIn("--aggregation-mode", command)
            self.assertEqual(command[command.index("--aggregation-mode") + 1], "mean_replicates")
            self.assertIn("--moving-average-window", command)
            self.assertEqual(command[command.index("--moving-average-window") + 1], "5")
            self.assertIn("--cost-basis", command)
            self.assertEqual(command[command.index("--cost-basis") + 1], "protocol_total")
            self.assertIn("--run-root", command)
            self.assertIn(str(run_root.resolve()), command)

    def test_run_analysis_command_includes_full_ui_contract_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_root = self._write_minimum_run_root(base)
            protocol_path = base / "threshold_protocol.json"
            protocol_path.write_text(
                json.dumps(
                    {
                        "metric": "h_mae_eV_mean",
                        "threshold_mev": 25.0,
                        "physical_rationale": "Test-only protocol rationale.",
                        "reference": "internal_protocol_v1",
                        "applies_to_metrics": ["h_mae_eV_mean"],
                        "recommended_sensitivity_thresholds_mev": [20.0, 25.0, 30.0],
                    }
                ),
                encoding="utf-8",
            )
            captured_command: list[str] = []

            def fake_run(command, **kwargs):
                captured_command[:] = list(command)
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "dataset_size_minimum_summary.json").write_text(
                    json.dumps({"status": "ok"}),
                    encoding="utf-8",
                )
                return MagicMock(returncode=0, stdout="", stderr="")

            with patch.object(self.pipeline_ui.subprocess, "run", side_effect=fake_run):
                self.pipeline_ui.run_dataset_size_minimum_analysis(
                    {
                        "run_roots": [str(run_root)],
                        "primary_metric": "h_mae_eV_mean",
                        "threshold_mev": 25.0,
                        "threshold_preset_key": "h_mae_relaxed_20",
                        "threshold_is_user_defined": True,
                        "x_axis": "n_train",
                        "n_min_source": "fit",
                        "n_min_fit_model": "power_law_floor",
                        "aggregation_mode": "mean_seeds_per_config",
                        "cost_basis": "protocol_total",
                        "claim_mode": "paper_candidate",
                        "threshold_protocol_file": str(protocol_path),
                        "bootstrap_replicates": 12,
                        "ci_level": 0.9,
                    }
                )

            command = captured_command
            self.assertEqual(command[command.index("--primary-metric") + 1], "h_mae_eV_mean")
            self.assertEqual(command[command.index("--threshold-mev") + 1], "25.0")
            self.assertEqual(command[command.index("--threshold-preset-key") + 1], "h_mae_relaxed_20")
            self.assertEqual(command[command.index("--threshold-is-user-defined") + 1], "true")
            self.assertEqual(command[command.index("--x-axis") + 1], "n_train")
            self.assertEqual(command[command.index("--n-min-source") + 1], "fit")
            self.assertEqual(command[command.index("--n-min-fit-model") + 1], "power_law_floor")
            self.assertEqual(command[command.index("--aggregation-mode") + 1], "mean_seeds_per_config")
            self.assertEqual(command[command.index("--cost-basis") + 1], "protocol_total")
            self.assertEqual(command[command.index("--claim-mode") + 1], "paper_candidate")
            self.assertEqual(command[command.index("--threshold-protocol-file") + 1], str(protocol_path.resolve()))
            self.assertEqual(command[command.index("--bootstrap-replicates") + 1], "12")
            self.assertEqual(command[command.index("--ci-level") + 1], "0.9")
            self.assertEqual(command[command.index("--run-root") + 1], str(run_root.resolve()))

    def test_invalid_threshold_protocol_file_rejected_by_run_analysis_before_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_root = self._write_minimum_run_root(base)
            missing_protocol = base / "missing_threshold_protocol.json"
            with patch.object(self.pipeline_ui.subprocess, "run") as mocked_run:
                with self.assertRaises(RuntimeError) as exc:
                    self.pipeline_ui.run_dataset_size_minimum_analysis(
                        {
                            "run_roots": [str(run_root)],
                            "threshold_mev": 10.0,
                            "threshold_protocol_file": str(missing_protocol),
                        }
                    )
            self.assertIn("threshold_protocol_file no existe", str(exc.exception))
            mocked_run.assert_not_called()

    def test_invalid_aggregation_mode_rejected_by_run_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_root = self._write_minimum_run_root(base)
            with self.assertRaises(RuntimeError):
                self.pipeline_ui.run_dataset_size_minimum_analysis(
                    {
                        "run_roots": [str(run_root)],
                        "threshold_mev": 10.0,
                        "aggregation_mode": "median",
                    }
                )

    def test_invalid_n_min_fit_model_rejected_by_run_analysis_before_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_root = self._write_minimum_run_root(base)
            with patch.object(self.pipeline_ui.subprocess, "run") as mocked_run:
                with self.assertRaises(RuntimeError) as exc:
                    self.pipeline_ui.run_dataset_size_minimum_analysis(
                        {
                            "run_roots": [str(run_root)],
                            "threshold_mev": 10.0,
                            "n_min_source": "fit",
                            "n_min_fit_model": "linear,inverse",
                        }
                    )
            self.assertIn("exactly one fit model", str(exc.exception))
            mocked_run.assert_not_called()

    def test_invalid_cost_basis_rejected_by_run_analysis_before_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_root = self._write_minimum_run_root(base)
            with patch.object(self.pipeline_ui.subprocess, "run") as mocked_run:
                with self.assertRaises(RuntimeError) as exc:
                    self.pipeline_ui.run_dataset_size_minimum_analysis(
                        {
                            "run_roots": [str(run_root)],
                            "threshold_mev": 10.0,
                            "cost_basis": "bad_basis",
                        }
                    )
            self.assertIn("Unknown cost_basis", str(exc.exception))
            mocked_run.assert_not_called()

    def test_invalid_claim_mode_rejected_by_run_analysis_before_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_root = self._write_minimum_run_root(base)
            with patch.object(self.pipeline_ui.subprocess, "run") as mocked_run:
                with self.assertRaises(RuntimeError) as exc:
                    self.pipeline_ui.run_dataset_size_minimum_analysis(
                        {
                            "run_roots": [str(run_root)],
                            "threshold_mev": 10.0,
                            "claim_mode": "paper",
                        }
                    )
            self.assertIn("Unknown claim_mode", str(exc.exception))
            mocked_run.assert_not_called()

    def test_visible_html_fit_options_are_accepted_by_server_side_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_root = self._write_minimum_run_root(base)
            visible = dataset_minimum_visible_fit_options()

            def fake_run(command, **kwargs):
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "dataset_size_minimum_summary.json").write_text(
                    json.dumps({"status": "ok", "outputs": [str(output_dir / "dataset_size_minimum_summary.json")]}),
                    encoding="utf-8",
                )
                return MagicMock(returncode=0, stdout="", stderr="")

            for fit_model in visible:
                with patch.object(self.pipeline_ui.subprocess, "run", side_effect=fake_run) as mocked_run:
                    self.pipeline_ui.run_dataset_size_minimum_analysis(
                        {
                            "run_roots": [str(run_root)],
                            "threshold_mev": 10.0,
                            "n_min_source": "fit",
                            "n_min_fit_model": fit_model,
                        }
                    )
                    command = mocked_run.call_args.args[0]
                    self.assertEqual(command[command.index("--n-min-fit-model") + 1], fit_model)

    def test_helper_smoke_summary_contains_core_contract_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = write_synthetic_temporal_dataset(
                base / "dataset",
                n_train=80,
                scalar_builder=lambda index: 1.0 if index % 2 else -1.0,
            )
            run_root = self._write_smoke_run_root(
                base / "graphene_w90_snapshot_scaling_smoke_campaign",
                dataset_root=dataset_root,
            )

            response = self._run_analysis_via_helper(
                payload={
                    "run_roots": [str(run_root)],
                    "threshold_mev": 20.0,
                    "primary_metric": "h_mae_eV_mean",
                    "x_axis": "n_train",
                    "n_min_source": "fit",
                    "n_min_fit_model": "power_law_floor",
                    "aggregation_mode": "mean_seeds_per_config",
                    "cost_basis": "protocol_total",
                    "bootstrap_replicates": 0,
                }
            )

            summary = response["summary"]
            self.assertEqual(summary["status"], "ok")
            self.assertIn("graph2mat", summary["thresholds"])
            self.assertIn("N_min_rel_tol", summary["thresholds"]["graph2mat"])
            self.assertIn("replicate_bootstrap", summary)
            self.assertIn("scientific_claim_status", summary)
            self.assertIn("paper_level_blockers", summary)
            self.assertEqual(summary["n_min_basis"], "nominal")
            self.assertEqual(summary["cost_basis"], "protocol_total")
            self.assertEqual(summary["thresholds"]["graph2mat"]["N_min_cost_eff_basis"], "protocol_total")
            self.assertIn("threshold_sensitivity", summary)
            report_path = next(Path(path) for path in summary["outputs"] if str(path).endswith("_report.md"))
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("N_min uses nominal N.", report_text)
            output_dir = Path(response["output_dir"])
            self.assertTrue((output_dir / "dataset_size_minimum_summary.json").exists())
            self.assertTrue((output_dir / "dataset_size_minimum_results.csv").exists())
            self.assertTrue((output_dir / "dataset_size_minimum_best_by_size.csv").exists())

    def test_claim_mode_paper_candidate_activates_strict_required_temporal_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = write_synthetic_temporal_dataset(base / "dataset", n_train=80)
            (dataset_root / "splits" / "split_summary.json").write_text('{"broken": ', encoding="utf-8")
            run_root = self._write_smoke_run_root(
                base / "graphene_w90_snapshot_scaling_smoke_campaign",
                dataset_root=dataset_root,
            )

            with self.assertRaises(minimum.JSONLoadError) as exc:
                self._run_analysis_via_helper(
                    payload={
                        "run_roots": [str(run_root)],
                        "threshold_mev": 20.0,
                        "primary_metric": "h_mae_eV_mean",
                        "x_axis": "n_train",
                        "n_min_source": "fit",
                        "n_min_fit_model": "power_law_floor",
                        "aggregation_mode": "mean_seeds_per_config",
                        "cost_basis": "protocol_total",
                        "claim_mode": "paper_candidate",
                        "bootstrap_replicates": 0,
                    }
                )
            self.assertIn("invalid_required_json", str(exc.exception))

    def test_http_endpoints_cover_get_preview_and_analyze(self) -> None:
        import http.server

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = write_synthetic_temporal_dataset(
                base / "dataset",
                n_train=80,
                scalar_builder=lambda index: 1.0 if index % 2 else -1.0,
            )
            run_root = self._write_smoke_run_root(
                base / "graphene_w90_snapshot_scaling_smoke_campaign",
                dataset_root=dataset_root,
            )

            original_results_root = self.pipeline_ui.RESULTS_ROOT
            self.pipeline_ui.RESULTS_ROOT = base

            recorded_commands: list[list[str]] = []
            real_subprocess_run = self.pipeline_ui.subprocess.run

            def recording_run(command, **kwargs):
                if not command or Path(str(command[1])).resolve() != self.pipeline_ui.DATASET_SIZE_MINIMUM_SCRIPT.resolve():
                    return real_subprocess_run(command, **kwargs)
                recorded_commands.append(list(command))
                args = minimum.build_parser().parse_args(command[2:])
                minimum.analyze(args)
                output_dir = Path(command[command.index("--output-dir") + 1])
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"status": "ok", "output_dir": str(output_dir)}),
                    stderr="",
                )

            class QuietHandler(self.pipeline_ui.ComparisonUIHandler):
                def log_message(self, format, *args):  # type: ignore[override]
                    return

            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            try:
                with patch.object(self.pipeline_ui.subprocess, "run", side_effect=recording_run):
                    thread.start()
                    base_url = f"http://127.0.0.1:{server.server_address[1]}"

                    def http_json(path: str, *, method: str = "GET", payload: dict | None = None) -> dict:
                        data = None
                        headers = {}
                        if payload is not None:
                            data = json.dumps(payload).encode("utf-8")
                            headers["Content-Type"] = "application/json"
                        request = urllib.request.Request(
                            f"{base_url}{path}",
                            data=data,
                            headers=headers,
                            method=method,
                        )
                        with urllib.request.urlopen(request, timeout=30) as response:
                            return json.loads(response.read().decode("utf-8"))

                    before = http_json("/api/g2m-deeph/dataset-size-minimum")
                    self.assertIn("outputs", before)
                    self.assertIn("run_root_sources", before)
                    self.assertTrue(
                        any(
                            Path(item["run_root"]).resolve() == run_root.resolve()
                            for item in before["run_root_sources"]
                        )
                    )

                    preview = http_json(
                        "/api/g2m-deeph/dataset-size-minimum/preview",
                        method="POST",
                        payload={
                            "run_roots": [str(run_root)],
                            "primary_metric": "h_mae_eV_mean",
                            "x_axis": "n_train",
                            "aggregation_mode": "mean_seeds_per_config",
                        },
                    )
                    self.assertEqual(preview["status"], "ok")
                    self.assertTrue(preview["best_rows"])

                    analyzed = http_json(
                        "/api/g2m-deeph/dataset-size-minimum/analyze",
                        method="POST",
                        payload={
                            "run_roots": [str(run_root)],
                            "primary_metric": "h_mae_eV_mean",
                            "threshold_mev": 20.0,
                            "threshold_preset_key": "h_mae_relaxed_20",
                            "threshold_is_user_defined": False,
                            "x_axis": "n_train",
                            "n_min_source": "fit",
                            "n_min_fit_model": "power_law_floor",
                            "aggregation_mode": "mean_seeds_per_config",
                            "cost_basis": "protocol_total",
                            "claim_mode": "paper_candidate",
                            "bootstrap_replicates": 8,
                            "ci_level": 0.9,
                        },
                    )
                    self.assertEqual(analyzed["status"], "ok")
                    self.assertTrue(recorded_commands)
                    command = recorded_commands[-1]
                    self.assertEqual(command[command.index("--primary-metric") + 1], "h_mae_eV_mean")
                    self.assertEqual(command[command.index("--threshold-mev") + 1], "20.0")
                    self.assertEqual(command[command.index("--threshold-preset-key") + 1], "h_mae_relaxed_20")
                    self.assertEqual(command[command.index("--threshold-is-user-defined") + 1], "false")
                    self.assertEqual(command[command.index("--x-axis") + 1], "n_train")
                    self.assertEqual(command[command.index("--n-min-fit-model") + 1], "power_law_floor")
                    self.assertEqual(command[command.index("--aggregation-mode") + 1], "mean_seeds_per_config")
                    self.assertEqual(command[command.index("--claim-mode") + 1], "paper_candidate")
                    self.assertEqual(command[command.index("--bootstrap-replicates") + 1], "8")
                    self.assertEqual(command[command.index("--ci-level") + 1], "0.9")
                    self.assertEqual(command[command.index("--run-root") + 1], str(run_root.resolve()))

                    output_dir = Path(analyzed["output_dir"])
                    self.assertTrue((output_dir / "dataset_size_minimum_summary.json").exists())
                    self.assertTrue((output_dir / "dataset_size_minimum_results.csv").exists())
                    self.assertTrue((output_dir / "dataset_size_minimum_best_by_size.csv").exists())

                    after = http_json("/api/g2m-deeph/dataset-size-minimum")
                    self.assertTrue(after["available"])
                    self.assertTrue(after["outputs"])
                    latest = next(
                        item for item in after["outputs"]
                        if Path(item["output_dir"]).resolve() == output_dir.resolve()
                    )
                    self.assertIn("scientific_claim_status", latest)
                    self.assertIn("paper_level_blockers", latest)
                    self.assertIn("n_min_basis", latest)
                    self.assertEqual(latest["threshold_preset_key"], "h_mae_relaxed_20")
                    self.assertEqual(latest["claim_mode_requested"], "paper_candidate")
                    self.assertIn(latest["claim_mode_actual"], {"diagnostic", "paper_candidate"})
                    self.assertTrue(Path(latest["report_path"]).exists())
                    self.assertTrue(latest["best_rows"])
                    self.assertTrue(latest["aggregated_rows"])
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()
                self.pipeline_ui.RESULTS_ROOT = original_results_root

    def test_helper_smoke_scientific_status_by_fit_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = write_synthetic_temporal_dataset(
                base / "dataset",
                n_train=80,
                scalar_builder=lambda index: 1.0 if index % 2 else -1.0,
            )
            run_root = self._write_smoke_run_root(base, dataset_root=dataset_root)

            expected = {
                "linear": "diagnostic_only",
                "none": "diagnostic_only",
                "cumulative_best": "diagnostic_only",
                "power_law_floor": "diagnostic_only",
            }
            for fit_model, expected_status in expected.items():
                response = self._run_analysis_via_helper(
                    payload={
                        "run_roots": [str(run_root)],
                        "threshold_mev": 20.0,
                        "primary_metric": "h_mae_eV_mean",
                        "x_axis": "n_train",
                        "n_min_source": "fit",
                        "n_min_fit_model": fit_model,
                        "aggregation_mode": "mean_seeds_per_config",
                        "bootstrap_replicates": 0,
                    }
                )
                summary = response["summary"]
                self.assertEqual(summary["scientific_claim_status"], expected_status)
                self.assertIn("paper_level_blockers", summary)
                self.assertTrue(summary["paper_level_blockers"])
                if fit_model == "power_law_floor":
                    self.assertIn("paper_uncertainty_block_hierarchy_incomplete", summary["paper_level_blockers"])

    def test_helper_smoke_summary_exposes_forbidden_compute_commands_without_invoking_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_root = self._write_smoke_run_root(base)
            response = self._run_analysis_via_helper(
                payload={
                    "run_roots": [str(run_root)],
                    "threshold_mev": 20.0,
                    "primary_metric": "h_mae_eV_mean",
                    "x_axis": "n_train",
                    "n_min_source": "fit",
                    "n_min_fit_model": "none",
                    "aggregation_mode": "mean_seeds_per_config",
                }
            )
            forbidden = response["summary"]["forbidden_compute_commands"]
            self.assertIn("siesta", forbidden)
            self.assertIn("deeph-train", forbidden)
            self.assertIn("deeph-preprocess", forbidden)
            self.assertIn("deeph-inference", forbidden)
            self.assertIn("gnubands", forbidden)

    def test_dataset_size_minimum_payload_exposes_aggregation_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output_dir = base / "dataset_size_minimum_ui_test" / "threshold_10meV"
            output_dir.mkdir(parents=True)
            summary = {
                "status": "ok",
                "primary_metric": "h_mae_eV_mean",
                "threshold_mev": 10.0,
                "x_axis": "n_train",
                "run_roots": [str(base / "run_a")],
                "aggregation_mode": "mean_replicates",
                "claim_mode_requested": "paper_candidate",
                "claim_mode_actual": "diagnostic",
                "moving_average_window": 5,
                "n_min_source": "fit",
                "n_min_fit_model": "linear",
                "aggregated_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size_x": 10,
                        "primary_metric_mev_mean": 20.0,
                    }
                ],
                "thresholds": {},
                "fits": {},
            }
            summary_path = output_dir / "dataset_size_minimum_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            (output_dir / "dataset_size_minimum_best_by_size.csv").write_text(
                "method,dataset_size_x,primary_metric_mev_mean\ngraph2mat,10,20.0\n",
                encoding="utf-8",
            )
            (output_dir / "dataset_size_minimum_results.csv").write_text("method\n", encoding="utf-8")

            original_paths = self.pipeline_ui._dataset_size_minimum_summary_paths
            try:
                self.pipeline_ui._dataset_size_minimum_summary_paths = lambda: [summary_path]
                payload = self.pipeline_ui.dataset_size_minimum_payload()
            finally:
                self.pipeline_ui._dataset_size_minimum_summary_paths = original_paths

            self.assertTrue(payload["outputs"])
            item = payload["outputs"][0]
            self.assertEqual(item["aggregation_mode"], "mean_replicates")
            self.assertEqual(item["actual_aggregation_mode"], "mean_replicates")
            self.assertEqual(item["aggregation_mode_classification"], "diagnostic_only")
            self.assertFalse(item["aggregation_mode_legacy_inferred"])
            self.assertEqual(item["claim_mode_requested"], "paper_candidate")
            self.assertEqual(item["claim_mode_actual"], "diagnostic")
            self.assertIn("threshold_basis", item)
            self.assertIn("threshold_reference", item)
            self.assertIn("threshold_metric_family", item)
            self.assertEqual(item["moving_average_window"], 5)
            self.assertEqual(item["x_axis"], "n_train")
            self.assertEqual(item["n_min_source"], "fit")
            self.assertEqual(item["n_min_fit_model"], "linear")
            self.assertEqual(item["run_roots"], [str(base / "run_a")])
            self.assertEqual(len(item["aggregated_rows"]), 1)
            self.assertIn("bootstrap", item)
            self.assertIn("replicate_bootstrap", item)
            self.assertEqual(item["bootstrap_replicates"], summary.get("bootstrap_replicates"))

    def test_payload_marks_old_summary_without_aggregation_mode_as_legacy_inferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output_dir = base / "dataset_size_minimum_ui_test" / "threshold_10meV"
            output_dir.mkdir(parents=True)
            summary = {
                "status": "ok",
                "primary_metric": "h_mae_eV_mean",
                "threshold_mev": 10.0,
                "x_axis": "n_train",
                "run_roots": [str(base / "run_a")],
                "thresholds": {},
                "fits": {},
            }
            summary_path = output_dir / "dataset_size_minimum_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            (output_dir / "dataset_size_minimum_best_by_size.csv").write_text("method\n", encoding="utf-8")
            (output_dir / "dataset_size_minimum_results.csv").write_text("method\n", encoding="utf-8")

            original_paths = self.pipeline_ui._dataset_size_minimum_summary_paths
            try:
                self.pipeline_ui._dataset_size_minimum_summary_paths = lambda: [summary_path]
                payload = self.pipeline_ui.dataset_size_minimum_payload()
            finally:
                self.pipeline_ui._dataset_size_minimum_summary_paths = original_paths

            item = payload["outputs"][0]
            self.assertEqual(item["aggregation_mode"], "best_config")
            self.assertEqual(item["actual_aggregation_mode"], "best_config")
            self.assertTrue(item["aggregation_mode_legacy_inferred"])
            self.assertEqual(item["aggregation_mode_classification"], "diagnostic_only")

    def test_payload_exposes_normalized_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output_dir = base / "dataset_size_minimum_ui_test" / "threshold_10meV"
            output_dir.mkdir(parents=True)
            summary = {
                "status": "ok",
                "primary_metric": "h_mae_eV_mean",
                "threshold_mev": 10.0,
                "x_axis": "n_train",
                "run_roots": [str(base / "run_a")],
                "aggregation_mode": "mean_replicates",
                "normalized_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size_x": 10,
                        "primary_metric_mev": 20.0,
                        "seed": "1",
                    }
                ],
                "aggregated_rows": [],
                "thresholds": {},
                "fits": {},
                "bootstrap": {"enabled": False},
                "bootstrap_replicates": 0,
                "ci_level": 0.95,
            }
            summary_path = output_dir / "dataset_size_minimum_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            (output_dir / "dataset_size_minimum_best_by_size.csv").write_text("method\n", encoding="utf-8")
            (output_dir / "dataset_size_minimum_results.csv").write_text("method\n", encoding="utf-8")

            original_paths = self.pipeline_ui._dataset_size_minimum_summary_paths
            try:
                self.pipeline_ui._dataset_size_minimum_summary_paths = lambda: [summary_path]
                payload = self.pipeline_ui.dataset_size_minimum_payload()
            finally:
                self.pipeline_ui._dataset_size_minimum_summary_paths = original_paths

            item = payload["outputs"][0]
            self.assertEqual(len(item["normalized_rows"]), 1)
            self.assertEqual(item["normalized_rows"][0]["method"], "graph2mat")

    def test_output_matching_rejects_aggregation_mode_mismatch(self) -> None:
        output = {
            "primary_metric": "h_mae_eV_mean",
            "threshold_mev": 10.0,
            "x_axis": "n_train",
            "n_min_source": "observed",
            "n_min_fit_model": "linear",
            "aggregation_mode": "best_config",
            "bootstrap_replicates": 0,
            "ci_level": 0.95,
            "run_roots": ["/tmp/run_a"],
        }
        controls = {
            "primary_metric": "h_mae_eV_mean",
            "threshold_mev": 10.0,
            "x_axis": "n_train",
            "n_min_source": "observed",
            "n_min_fit_model": "linear",
            "aggregation_mode": "mean_replicates",
            "bootstrap_replicates": 0,
            "ci_level": 0.95,
            "run_roots": ["/tmp/run_a"],
        }
        self.assertFalse(
            self.pipeline_ui.dataset_size_minimum_output_matches_controls(output, controls)
        )

    def test_output_matching_rejects_bootstrap_replicates_mismatch(self) -> None:
        base = {
            "primary_metric": "h_mae_eV_mean",
            "threshold_mev": 10.0,
            "x_axis": "n_train",
            "n_min_source": "observed",
            "n_min_fit_model": "linear",
            "aggregation_mode": "mean_replicates",
            "ci_level": 0.95,
            "run_roots": [],
        }
        output = {**base, "bootstrap_replicates": 100}
        controls = {**base, "bootstrap_replicates": 0}
        self.assertFalse(
            self.pipeline_ui.dataset_size_minimum_output_matches_controls(output, controls)
        )

    def test_output_matching_rejects_ci_level_mismatch(self) -> None:
        base = {
            "primary_metric": "h_mae_eV_mean",
            "threshold_mev": 10.0,
            "x_axis": "n_train",
            "n_min_source": "observed",
            "n_min_fit_model": "linear",
            "aggregation_mode": "mean_replicates",
            "bootstrap_replicates": 50,
            "run_roots": [],
        }
        output = {**base, "ci_level": 0.95}
        controls = {**base, "ci_level": 0.90}
        self.assertFalse(
            self.pipeline_ui.dataset_size_minimum_output_matches_controls(output, controls)
        )

    def test_analyze_summary_includes_normalized_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "a",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.010,
                        "seed": "1",
                    },
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "b",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.030,
                        "seed": "2",
                    },
                ]
            }
            (root / "summary" / "ranking").mkdir(parents=True)
            (root / "summary" / "ranking" / "normalized_run_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            summary = minimum.analyze(
                make_analyze_args(
                    run_root=[str(root)],
                    output_dir=str(base / "out"),
                    aggregation_mode="mean_replicates",
                    bootstrap_replicates=20,
                )
            )
            self.assertEqual(len(summary["normalized_rows"]), 2)
            self.assertTrue(summary["bootstrap"]["enabled"])

    def test_temporal_diagnostics_missing_metadata_warns_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "config_id": "a",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.010,
                        "seed": "1",
                    }
                ]
            }
            (root / "summary" / "ranking").mkdir(parents=True)
            (root / "summary" / "ranking" / "normalized_run_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            summary = minimum.analyze(
                make_analyze_args(run_root=[str(root)], output_dir=str(base / "out"))
            )
            temporal = summary["temporal_diagnostics"]
            self.assertFalse(summary["autocorrelation_available"])
            self.assertIsNone(summary["estimated_n_eff_train"])
            self.assertIn("N is nominal; N_eff not estimated", temporal["status_message"])
            self.assertEqual(summary["n_min_basis"], "nominal")
            self.assertIn("graph2mat", summary["N_min_nominal"])
            self.assertFalse(summary["N_eff_diagnostic_available"])
            self.assertIsNone(summary["N_eff_over_N_nominal"])
            self.assertEqual(summary["N_eff_by_dataset_size"], {})
            self.assertEqual(summary["autocorrelation_available_by_dataset_size"], {})
            self.assertEqual(summary["scientific_claim_status"], "diagnostic_only")
            self.assertIn(
                "paper_blocked_if_autocorrelation_unavailable",
                summary["paper_level_blockers"],
            )
            self.assertIn(
                "N_min uses nominal N. If MD snapshots are autocorrelated",
                " ".join(summary["paper_level_warnings"]),
            )
            joined = " | ".join(temporal["warnings"])
            self.assertTrue(
                "temporal_diagnostics_no_dataset_roots_detected" in joined
                or "temporal_metadata_missing" in joined
            )

    def test_read_json_optional_corrupt_json_warns_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"broken": ', encoding="utf-8")
            warnings: list[str] = []

            payload = minimum.read_json_optional(
                path,
                warnings=warnings,
                context="test.optional_json",
            )

            self.assertIsNone(payload)
            self.assertTrue(
                any("invalid_optional_json" in item and "test.optional_json" in item for item in warnings)
            )

    def test_read_json_required_corrupt_json_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"broken": ', encoding="utf-8")

            with self.assertRaises(minimum.JSONLoadError) as exc:
                minimum.read_json_required(path, context="test.required_json")

            message = str(exc.exception)
            self.assertIn("invalid_required_json", message)
            self.assertIn(str(path), message)
            self.assertIn("test.required_json", message)
            self.assertIn("JSONDecodeError", message)

    def test_temporal_diagnostics_detects_blocks_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_synthetic_temporal_dataset(Path(tmp) / "dataset", n_train=6)
            diag = minimum.diagnose_dataset_temporal_metadata(dataset_root)
            self.assertEqual(diag["n_temporal_blocks"], 1)
            self.assertEqual(diag["block_sizes"]["block_a"], 6)
            self.assertTrue(diag["blocked_split"])
            self.assertTrue(diag["temporal_order_detected"])
            self.assertEqual(diag["nominal_n_train"], 6)

    def test_temporal_gap_one_emits_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_synthetic_temporal_dataset(
                Path(tmp) / "dataset",
                temporal_gap=1,
            )
            diag = minimum.diagnose_dataset_temporal_metadata(dataset_root)
            self.assertTrue(
                any("temporal_gap_le_1" in warning for warning in diag["warnings"])
            )

    def test_analyze_temporal_gap_le_one_adds_paper_level_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = write_synthetic_temporal_dataset(
                base / "dataset",
                n_train=20,
                temporal_gap=1,
            )
            run_root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size": 20,
                        "dataset_root": str(dataset_root),
                        "config_id": "a",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.010,
                        "seed": "1",
                    }
                ]
            }
            (run_root / "summary" / "ranking").mkdir(parents=True)
            (run_root / "summary" / "ranking" / "normalized_run_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            summary = minimum.analyze(
                make_analyze_args(run_root=[str(run_root)], output_dir=str(base / "out"))
            )

            self.assertIn("paper_blocked_if_temporal_gap_le_1", summary["paper_level_blockers"])

    def test_autocorrelation_convention_fields_on_iid_series(self) -> None:
        rng = random.Random(7)
        values = [rng.gauss(0.0, 1.0) for _ in range(300)]
        diag = minimum.compute_scalar_autocorrelation_diagnostics(values)
        self.assertEqual(diag["status"], "ok")
        self.assertEqual(
            diag["autocorrelation_convention"],
            minimum.AUTOCORRELATION_CONVENTION,
        )
        self.assertAlmostEqual(diag["statistical_inefficiency"], diag["tau_int"])
        self.assertGreater(float(diag["n_eff"]), 300 * 0.5)

    def test_constant_scalar_series_does_not_crash(self) -> None:
        diag = minimum.compute_scalar_autocorrelation_diagnostics([5.0] * 20)
        self.assertIn(diag["status"], {"ok", "insufficient_samples"})

    def test_fewer_than_three_samples_does_not_crash(self) -> None:
        diag = minimum.compute_scalar_autocorrelation_diagnostics([1.0, 2.0])
        self.assertEqual(diag["status"], "insufficient_samples")
        self.assertFalse(diag["autocorrelation_available"])

    def test_autocorrelated_scalar_series_yields_n_eff_below_n(self) -> None:
        rho = 0.95
        noise = math.sqrt(1.0 - rho * rho)
        ar1_values = [0.0]
        for index in range(1, 80):
            ar1_values.append(rho * ar1_values[-1] + noise * math.sin(index))

        def builder(index: int) -> float:
            return ar1_values[index]

        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_synthetic_temporal_dataset(
                Path(tmp) / "dataset",
                n_train=80,
                scalar_builder=builder,
            )
            diag = minimum.diagnose_dataset_temporal_metadata(dataset_root)
            self.assertTrue(diag["autocorrelation_available"])
            n_eff = diag["estimated_n_eff_train"]
            self.assertIsNotNone(n_eff)
            self.assertLess(float(n_eff), 80 * 0.5)

    def test_temporal_diagnostics_keep_independent_blocks_separate(self) -> None:
        specs = []
        for index in range(6):
            specs.append(
                {
                    "metadata": {
                        "frame_index": index,
                        "source_frame_index": index,
                        "block_id": "block_a",
                        "temperature_K": "300",
                        "total_energy_eV": float(index),
                    }
                }
            )
        for index in range(6):
            specs.append(
                {
                    "metadata": {
                        "frame_index": index,
                        "source_frame_index": index,
                        "block_id": "block_b",
                        "temperature_K": "300",
                        "total_energy_eV": float(100 + index),
                    }
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_temporal_dataset_from_specs(Path(tmp) / "dataset", specs=specs)
            diag = minimum.diagnose_dataset_temporal_metadata(dataset_root)

            self.assertTrue(diag["autocorrelation_available"])
            self.assertEqual(diag["train_group_count"], 2)
            self.assertEqual(diag["autocorrelation"]["train"]["n_groups"], 2)
            self.assertEqual(len(diag["autocorrelation"]["by_block"]), 2)
            self.assertNotIn("autocorrelation_unavailable_mixed_temperatures", diag["warnings"])

    def test_temporal_diagnostics_corrupt_optional_split_summary_warns_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_synthetic_temporal_dataset(Path(tmp) / "dataset", n_train=6)
            (dataset_root / "splits" / "split_summary.json").write_text('{"broken": ', encoding="utf-8")

            diag = minimum.diagnose_dataset_temporal_metadata(dataset_root)

            self.assertIn("invalid_optional_json", " | ".join(diag["warnings"]))
            self.assertEqual(diag["nominal_n_train"], 6)

    def test_temporal_diagnostics_strict_mode_corrupt_split_summary_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_synthetic_temporal_dataset(Path(tmp) / "dataset", n_train=6)
            (dataset_root / "splits" / "split_summary.json").write_text('{"broken": ', encoding="utf-8")

            with self.assertRaises(minimum.JSONLoadError) as exc:
                minimum.diagnose_dataset_temporal_metadata(
                    dataset_root,
                    strict_required_json=True,
                )

            message = str(exc.exception)
            self.assertIn("invalid_required_json", message)
            self.assertIn("temporal_diagnostics.split_summary", message)

    def test_temporal_diagnostics_strict_mode_corrupt_frozen_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_synthetic_temporal_dataset(Path(tmp) / "dataset", n_train=6)
            (dataset_root / "frozen_split_manifest.json").write_text('{"broken": ', encoding="utf-8")

            with self.assertRaises(minimum.JSONLoadError) as exc:
                minimum.diagnose_dataset_temporal_metadata(
                    dataset_root,
                    strict_required_json=True,
                )

            message = str(exc.exception)
            self.assertIn("invalid_required_json", message)
            self.assertIn("temporal_diagnostics.frozen_split_manifest", message)

    def test_temporal_diagnostics_corrupt_frozen_manifest_degrades_in_exploratory_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_synthetic_temporal_dataset(Path(tmp) / "dataset", n_train=6)
            (dataset_root / "frozen_split_manifest.json").write_text('{"broken": ', encoding="utf-8")

            diag = minimum.diagnose_dataset_temporal_metadata(dataset_root)

            self.assertIn("invalid_optional_json", " | ".join(diag["warnings"]))
            self.assertEqual(diag["nominal_n_train"], 6)

    def test_summarize_temporal_diagnostics_exposes_n_eff_by_dataset_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_10 = write_synthetic_temporal_dataset(base / "dataset_10", n_train=10)
            dataset_20 = write_synthetic_temporal_dataset(base / "dataset_20", n_train=20)

            summary = minimum.summarize_temporal_diagnostics(
                [
                    {"dataset_root": str(dataset_10), "dataset_id": "d10"},
                    {"dataset_root": str(dataset_20), "dataset_id": "d20"},
                ],
                run_roots=[],
            )

            self.assertIn("10", summary["N_eff_by_dataset_size"])
            self.assertIn("20", summary["N_eff_by_dataset_size"])
            self.assertTrue(summary["autocorrelation_available_by_dataset_size"]["10"])
            self.assertTrue(summary["autocorrelation_available_by_dataset_size"]["20"])
            self.assertIsNotNone(summary["N_eff_over_N_by_dataset_size"]["10"])
            self.assertIsNotNone(summary["temporal_block_diagnostics_by_dataset_size"]["10"])
            self.assertEqual(
                summary["temporal_block_diagnostics_by_dataset_size"]["10"]["dataset_ids"],
                ["d10"],
            )

    def test_temporal_diagnostics_mixed_temperatures_are_not_aggregated(self) -> None:
        specs = []
        for index in range(4):
            specs.append(
                {
                    "metadata": {
                        "frame_index": index,
                        "source_frame_index": index,
                        "block_id": "block_a",
                        "temperature_K": "300",
                        "total_energy_eV": float(index),
                    }
                }
            )
        for index in range(4):
            specs.append(
                {
                    "metadata": {
                        "frame_index": index,
                        "source_frame_index": index,
                        "block_id": "block_b",
                        "temperature_K": "500",
                        "total_energy_eV": float(10 + index),
                    }
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_temporal_dataset_from_specs(Path(tmp) / "dataset", specs=specs)
            diag = minimum.diagnose_dataset_temporal_metadata(dataset_root)

            self.assertFalse(diag["autocorrelation_available"])
            self.assertIsNone(diag["estimated_n_eff_train"])
            self.assertIn("autocorrelation_unavailable_mixed_temperatures", diag["warnings"])

    def test_summarize_temporal_diagnostics_marks_partial_n_eff_by_dataset_size(self) -> None:
        specs = []
        for index in range(6):
            specs.append(
                {
                    "metadata": {
                        "frame_index": index,
                        "source_frame_index": index,
                        "block_id": "block_a",
                        "temperature_K": "300",
                    }
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_ok = write_synthetic_temporal_dataset(base / "dataset_ok", n_train=10)
            dataset_missing = write_temporal_dataset_from_specs(base / "dataset_missing", specs=specs)
            summary = minimum.summarize_temporal_diagnostics(
                [
                    {"dataset_root": str(dataset_ok), "dataset_id": "ok"},
                    {"dataset_root": str(dataset_missing), "dataset_id": "missing"},
                ],
                run_roots=[],
            )

            self.assertIn("10", summary["N_eff_by_dataset_size"])
            self.assertIn("6", summary["N_eff_by_dataset_size"])
            self.assertIsNone(summary["N_eff_by_dataset_size"]["6"])
            self.assertFalse(summary["autocorrelation_available_by_dataset_size"]["6"])
            self.assertEqual(
                summary["temporal_block_diagnostics_by_dataset_size"]["6"]["n_with_n_eff"],
                0,
            )

    def test_temporal_diagnostics_ambiguous_grouping_metadata_blocks_autocorrelation(self) -> None:
        specs = []
        for index in range(6):
            specs.append(
                {
                    "metadata": {
                        "trajectory_id": "traj_a" if index < 3 else None,
                        "frame_index": index,
                        "source_frame_index": index,
                        "temperature_K": "300",
                        "total_energy_eV": float(index),
                    }
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_temporal_dataset_from_specs(Path(tmp) / "dataset", specs=specs)
            diag = minimum.diagnose_dataset_temporal_metadata(dataset_root)

            self.assertFalse(diag["autocorrelation_available"])
            self.assertIn(
                "autocorrelation_unavailable_missing_or_ambiguous_grouping_metadata",
                diag["warnings"],
            )

    def test_temporal_diagnostics_missing_scalar_series_marks_autocorrelation_unavailable(self) -> None:
        specs = []
        for index in range(6):
            specs.append(
                {
                    "metadata": {
                        "frame_index": index,
                        "source_frame_index": index,
                        "block_id": "block_a",
                        "temperature_K": "300",
                    }
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_temporal_dataset_from_specs(Path(tmp) / "dataset", specs=specs)
            diag = minimum.diagnose_dataset_temporal_metadata(dataset_root)

            self.assertFalse(diag["autocorrelation_available"])
            self.assertIn("autocorrelation_unavailable_no_cheap_scalar_series", diag["warnings"])

    def test_summarize_temporal_diagnostics_preserves_block_details_by_dataset_size(self) -> None:
        specs = []
        for index in range(4):
            specs.append(
                {
                    "metadata": {
                        "frame_index": index,
                        "source_frame_index": index,
                        "block_id": "block_a",
                        "temperature_K": "300",
                        "total_energy_eV": float(index),
                    }
                }
            )
        for index in range(4):
            specs.append(
                {
                    "metadata": {
                        "frame_index": index,
                        "source_frame_index": index,
                        "block_id": "block_b",
                        "temperature_K": "500",
                        "total_energy_eV": float(10 + index),
                    }
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_temporal_dataset_from_specs(Path(tmp) / "dataset", specs=specs)
            summary = minimum.summarize_temporal_diagnostics(
                [{"dataset_root": str(dataset_root), "dataset_id": "mixed"}],
                run_roots=[],
            )

            diag = summary["temporal_block_diagnostics_by_dataset_size"]["8"]
            self.assertFalse(summary["autocorrelation_available_by_dataset_size"]["8"])
            self.assertEqual(diag["n_datasets"], 1)
            self.assertEqual(len(diag["datasets"][0]["block_diagnostics"]), 2)

    def test_analyze_blocks_paper_claim_when_n_eff_much_smaller_than_nominal(self) -> None:
        def smooth_builder(index: int) -> float:
            return math.sin(index / 30.0)

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = write_synthetic_temporal_dataset(
                base / "dataset",
                n_train=120,
                scalar_builder=smooth_builder,
            )
            run_root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size": 120,
                        "dataset_root": str(dataset_root),
                        "config_id": "a",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.010,
                        "seed": "1",
                    }
                ]
            }
            (run_root / "summary" / "ranking").mkdir(parents=True)
            (run_root / "summary" / "ranking" / "normalized_run_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            summary = minimum.analyze(
                make_analyze_args(run_root=[str(run_root)], output_dir=str(base / "out"))
            )

            self.assertTrue(summary["N_eff_diagnostic_available"])
            self.assertIsNotNone(summary["N_eff_over_N_nominal"])
            self.assertLess(summary["N_eff_over_N_nominal"], minimum.N_EFF_MUCH_SMALLER_THAN_NOMINAL_RATIO)
            self.assertEqual(summary["scientific_claim_status"], "diagnostic_only")
            self.assertIn(
                "paper_blocked_if_n_eff_much_smaller_than_nominal",
                summary["paper_level_blockers"],
            )
            self.assertIn("graph2mat", summary["effective_samples_at_N_min_nominal"])
            self.assertIn(
                "graph2mat",
                summary["effective_samples_at_nominal_N_min_diagnostic"],
            )
            self.assertIsNotNone(
                summary["effective_samples_at_N_min_nominal"]["graph2mat"]["N_min_abs"]
            )
            self.assertEqual(
                summary["effective_samples_at_nominal_N_min_diagnostic"],
                summary["effective_samples_at_N_min_nominal"],
            )
            self.assertEqual(
                summary["N_min_eff_diagnostic"],
                summary["effective_samples_at_nominal_N_min_diagnostic"],
            )
            self.assertEqual(
                summary["N_min_eff_diagnostic_deprecated_alias_for"],
                "effective_samples_at_nominal_N_min_diagnostic",
            )

    def test_scientific_status_blocks_fit_when_n_eff_by_dataset_size_incomplete(self) -> None:
        status = minimum.scientific_claim_status_payload(
            temporal_diagnostics={
                "autocorrelation_available": True,
                "nominal_n_train": 20,
                "estimated_n_eff_train": 16.0,
                "warnings": [],
                "N_eff_by_dataset_size": {"10": 8.0, "20": None},
                "autocorrelation_available_by_dataset_size": {"10": True, "20": False},
            },
            thresholds={"graph2mat": {"N_min_abs": 12, "available_sizes": [10, 20]}},
            threshold_metadata={
                "threshold_basis": "paper_protocol_locked_threshold",
                "threshold_reference": "test_reference",
                "threshold_interpretation": "test-only locked threshold protocol",
                "threshold_metric_family": "hamiltonian_element_error_mev",
                "threshold_is_user_defined": False,
                "threshold_paper_justified": True,
            },
            aggregation_mode="mean_seeds_per_config",
            requested_n_min_source="fit",
            actual_n_min_source="fit",
            requested_fit_model="power_law_floor",
            actual_fit_model="power_law_floor",
            fit_threshold_details={
                "graph2mat": {
                    "fit_policy": "paper_candidate",
                    "paper_candidate": True,
                    "fit_model": "power_law_floor",
                    "status": "ok",
                    "enough_points_for_paper_candidate": True,
                }
            },
            fallback_used=False,
            fallback_reason=None,
        )

        self.assertEqual(status["scientific_claim_status"], "diagnostic_only")
        self.assertIn(
            "paper_blocked_if_n_eff_by_dataset_size_incomplete",
            status["paper_level_blockers"],
        )

    def test_analyze_exposes_n_min_effective_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = write_synthetic_temporal_dataset(base / "dataset", n_train=10)
            run_root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "dataset_root": str(dataset_root),
                        "config_id": "a",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.010,
                        "seed": "1",
                    }
                ]
            }
            (run_root / "summary" / "ranking").mkdir(parents=True)
            (run_root / "summary" / "ranking" / "normalized_run_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            summary = minimum.analyze(
                make_analyze_args(run_root=[str(run_root)], output_dir=str(base / "out"))
            )

            self.assertIn("N_min_effective_diagnostic", summary)
            self.assertEqual(
                summary["N_min_effective_diagnostic"]["graph2mat"]["N_min_abs"],
                summary["N_eff_by_dataset_size"]["10"],
            )
            self.assertEqual(summary["n_min_basis"], "nominal")

    def test_iid_scalar_series_yields_n_eff_near_nominal(self) -> None:
        rng = random.Random(123)

        def iid_builder(_index: int) -> float:
            return rng.gauss(0.0, 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = write_synthetic_temporal_dataset(
                Path(tmp) / "dataset",
                n_train=200,
                scalar_builder=iid_builder,
            )
            diag = minimum.diagnose_dataset_temporal_metadata(dataset_root)
            self.assertTrue(diag["autocorrelation_available"])
            n_eff = diag["estimated_n_eff_train"]
            self.assertIsNotNone(n_eff)
            self.assertGreater(float(n_eff), 200 * 0.5)

    def test_available_autocorrelation_with_acceptable_ratio_does_not_block_for_missing_acf(self) -> None:
        def alternating_builder(index: int) -> float:
            return 1.0 if index % 2 else -1.0

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = write_synthetic_temporal_dataset(
                base / "dataset",
                n_train=40,
                scalar_builder=alternating_builder,
            )
            run_root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size": 40,
                        "dataset_root": str(dataset_root),
                        "config_id": "a",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.010,
                        "seed": "1",
                    }
                ]
            }
            (run_root / "summary" / "ranking").mkdir(parents=True)
            (run_root / "summary" / "ranking" / "normalized_run_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            summary = minimum.analyze(
                make_analyze_args(run_root=[str(run_root)], output_dir=str(base / "out"))
            )

            self.assertTrue(summary["autocorrelation_available"])
            self.assertTrue(summary["N_eff_diagnostic_available"])
            self.assertGreaterEqual(
                summary["N_eff_over_N_nominal"],
                minimum.N_EFF_MUCH_SMALLER_THAN_NOMINAL_RATIO,
            )
            self.assertNotIn(
                "paper_blocked_if_autocorrelation_unavailable",
                summary["paper_level_blockers"],
            )
            self.assertNotIn(
                "paper_blocked_if_n_eff_much_smaller_than_nominal",
                summary["paper_level_blockers"],
            )

    def test_scientific_status_allows_power_law_fit_protocol_when_temporal_policy_allows(self) -> None:
        status = minimum.scientific_claim_status_payload(
            temporal_diagnostics={
                "autocorrelation_available": True,
                "nominal_n_train": 100,
                "estimated_n_eff_train": 80.0,
                "warnings": [],
            },
            thresholds={"graph2mat": {"N_min_abs": 20}},
            threshold_metadata={
                "threshold_basis": "paper_protocol_locked_threshold",
                "threshold_reference": "test_reference",
                "threshold_interpretation": "test-only locked threshold protocol",
                "threshold_metric_family": "hamiltonian_element_error_mev",
                "threshold_is_user_defined": False,
                "threshold_paper_justified": True,
            },
            aggregation_mode="mean_seeds_per_config",
            requested_n_min_source="fit",
            actual_n_min_source="fit",
            requested_fit_model="power_law_floor",
            actual_fit_model="power_law_floor",
            fit_threshold_details={
                "graph2mat": {
                    "fit_policy": "paper_candidate",
                    "paper_candidate": True,
                    "fit_model": "power_law_floor",
                    "status": "ok",
                    "enough_points_for_paper_candidate": True,
                }
            },
            fallback_used=False,
            fallback_reason=None,
        )

        self.assertEqual(
            status["scientific_claim_status"],
            "paper_candidate_nominal_with_n_eff_diagnostic",
        )
        self.assertEqual(status["claim_mode_requested"], "diagnostic")
        self.assertEqual(status["claim_mode_actual"], "diagnostic")
        self.assertEqual(status["n_min_fit_policy"], "paper_candidate")
        self.assertFalse(status["paper_level_blockers"])

    def test_scientific_status_claim_mode_blocks_non_explicit_paper_candidate_requirements(self) -> None:
        status = minimum.scientific_claim_status_payload(
            temporal_diagnostics={
                "autocorrelation_available": True,
                "nominal_n_train": 100,
                "estimated_n_eff_train": 80.0,
                "warnings": [],
                "N_eff_by_dataset_size": {"10": 8.0, "20": 16.0},
                "autocorrelation_available_by_dataset_size": {"10": True, "20": True},
            },
            thresholds={"graph2mat": {"N_min_abs": 20, "available_sizes": [10, 20]}},
            threshold_metadata={
                "threshold_basis": "paper_protocol_locked_threshold",
                "threshold_reference": "test_reference",
                "threshold_interpretation": "test-only locked threshold protocol",
                "threshold_metric_family": "hamiltonian_element_error_mev",
                "threshold_is_user_defined": False,
                "threshold_paper_justified": True,
            },
            claim_mode_requested="paper_candidate",
            aggregation_mode="best_config_mean",
            requested_aggregation_mode="best_config_mean",
            actual_aggregation_mode="best_config_mean",
            requested_n_min_source="observed",
            actual_n_min_source="observed",
            requested_fit_model="linear",
            actual_fit_model="linear",
            fit_threshold_details={
                "graph2mat": {
                    "fit_policy": "diagnostic_only",
                    "paper_candidate": False,
                    "fit_model": "linear",
                    "status": "ok",
                }
            },
            hierarchical_uncertainty={
                "status": "diagnostic_only",
                "paper_ready": False,
                "paper_level_blockers": ["paper_uncertainty_block_hierarchy_incomplete"],
            },
            fallback_used=False,
            fallback_reason=None,
        )

        self.assertEqual(status["claim_mode_requested"], "paper_candidate")
        self.assertEqual(status["claim_mode_actual"], "diagnostic")
        self.assertIn(
            "paper_blocked_if_best_config_mean_policy_not_documented",
            status["paper_level_blockers"],
        )
        self.assertIn(
            "paper_blocked_if_claim_mode_requires_fit_n_min_source",
            status["paper_level_blockers"],
        )
        self.assertIn(
            "paper_blocked_if_claim_mode_requires_power_law_floor",
            status["paper_level_blockers"],
        )

    def test_scientific_status_claim_mode_can_pass_when_all_gates_clear(self) -> None:
        status = minimum.scientific_claim_status_payload(
            temporal_diagnostics={
                "autocorrelation_available": True,
                "nominal_n_train": 100,
                "estimated_n_eff_train": 80.0,
                "warnings": [],
                "N_eff_by_dataset_size": {"10": 8.0, "20": 16.0, "30": 24.0, "40": 32.0, "50": 40.0},
                "autocorrelation_available_by_dataset_size": {
                    "10": True,
                    "20": True,
                    "30": True,
                    "40": True,
                    "50": True,
                },
            },
            thresholds={"graph2mat": {"N_min_abs": 20, "available_sizes": [10, 20, 30, 40, 50]}},
            threshold_metadata={
                "threshold_basis": "paper_protocol_locked_threshold",
                "threshold_reference": "test_reference",
                "threshold_interpretation": "test-only locked threshold protocol",
                "threshold_metric_family": "hamiltonian_element_error_mev",
                "threshold_is_user_defined": False,
                "threshold_paper_justified": True,
            },
            claim_mode_requested="paper_candidate",
            aggregation_mode="mean_seeds_per_config",
            requested_aggregation_mode="mean_seeds_per_config",
            actual_aggregation_mode="mean_seeds_per_config",
            requested_n_min_source="fit",
            actual_n_min_source="fit",
            requested_fit_model="power_law_floor",
            actual_fit_model="power_law_floor",
            fit_threshold_details={
                "graph2mat": {
                    "fit_policy": "paper_candidate",
                    "paper_candidate": True,
                    "fit_model": "power_law_floor",
                    "status": "ok",
                    "minimum_fit_points_for_paper_candidate": minimum.MIN_FIT_POINTS_FOR_PAPER_CANDIDATE,
                    "enough_points_for_paper_candidate": True,
                }
            },
            fit_predictive_stability_by_left_out_N={
                "status": "ok",
                "methods": {"graph2mat": {"paper_level_blockers": []}},
            },
            hierarchical_uncertainty={
                "status": "paper_ready_supporting_uncertainty_available",
                "paper_ready": True,
                "paper_level_blockers": [],
            },
            fallback_used=False,
            fallback_reason=None,
        )

        self.assertEqual(
            status["scientific_claim_status"],
            "paper_candidate_nominal_with_n_eff_diagnostic",
        )
        self.assertEqual(status["claim_mode_requested"], "paper_candidate")
        self.assertEqual(status["claim_mode_actual"], "paper_candidate")

    def test_scientific_status_blocks_power_law_floor_with_lt_five_points(self) -> None:
        status = minimum.scientific_claim_status_payload(
            temporal_diagnostics={
                "autocorrelation_available": True,
                "nominal_n_train": 100,
                "estimated_n_eff_train": 80.0,
                "warnings": [],
            },
            thresholds={"graph2mat": {"N_min_abs": 20}},
            aggregation_mode="mean_seeds_per_config",
            requested_n_min_source="fit",
            actual_n_min_source="fit",
            requested_fit_model="power_law_floor",
            actual_fit_model="power_law_floor",
            fit_threshold_details={
                "graph2mat": {
                    "fit_policy": "diagnostic_only",
                    "paper_candidate": False,
                    "fit_model": "power_law_floor",
                    "status": "ok",
                    "enough_points_for_paper_candidate": False,
                }
            },
            fallback_used=False,
            fallback_reason=None,
        )

        self.assertEqual(status["scientific_claim_status"], "diagnostic_only")
        self.assertIn(
            f"paper_blocked_if_power_law_floor_points_lt_{minimum.MIN_FIT_POINTS_FOR_PAPER_CANDIDATE}:graph2mat",
            status["paper_level_blockers"],
        )

    def test_scientific_status_includes_fit_stability_blocker(self) -> None:
        status = minimum.scientific_claim_status_payload(
            temporal_diagnostics={
                "autocorrelation_available": True,
                "nominal_n_train": 100,
                "estimated_n_eff_train": 80.0,
                "warnings": [],
            },
            thresholds={"graph2mat": {"N_min_abs": 20}},
            aggregation_mode="mean_seeds_per_config",
            requested_n_min_source="fit",
            actual_n_min_source="fit",
            requested_fit_model="power_law_floor",
            actual_fit_model="power_law_floor",
            fit_threshold_details={
                "graph2mat": {
                    "fit_policy": "paper_candidate",
                    "paper_candidate": True,
                    "fit_model": "power_law_floor",
                    "status": "ok",
                    "enough_points_for_paper_candidate": True,
                }
            },
            fit_predictive_stability_by_left_out_N={
                "status": "ok",
                "methods": {
                    "graph2mat": {
                        "paper_level_blockers": [
                            "paper_blocked_if_fit_predictive_stability_unstable:graph2mat:N_min_plateau"
                        ]
                    }
                },
            },
            fallback_used=False,
            fallback_reason=None,
        )

        self.assertEqual(status["scientific_claim_status"], "diagnostic_only")
        self.assertIn(
            "paper_blocked_if_fit_predictive_stability_unstable:graph2mat:N_min_plateau",
            status["paper_level_blockers"],
        )

    def test_scientific_status_blocks_linear_fit_even_if_temporal_policy_allows(self) -> None:
        status = minimum.scientific_claim_status_payload(
            temporal_diagnostics={
                "autocorrelation_available": True,
                "nominal_n_train": 100,
                "estimated_n_eff_train": 80.0,
                "warnings": [],
            },
            thresholds={"graph2mat": {"N_min_abs": 20}},
            aggregation_mode="mean_seeds_per_config",
            requested_n_min_source="fit",
            actual_n_min_source="fit",
            requested_fit_model="linear",
            actual_fit_model="linear",
            fit_threshold_details={
                "graph2mat": {
                    "fit_policy": "diagnostic_only",
                    "paper_candidate": False,
                    "fit_model": "linear",
                    "status": "ok",
                }
            },
            fallback_used=False,
            fallback_reason=None,
        )

        self.assertEqual(status["scientific_claim_status"], "diagnostic_only")
        self.assertIn(
            "paper_blocked_if_n_min_fit_policy_diagnostic_only:linear",
            status["paper_level_blockers"],
        )

    def test_scientific_status_blocks_cumulative_best_fit(self) -> None:
        status = minimum.scientific_claim_status_payload(
            temporal_diagnostics={
                "autocorrelation_available": True,
                "nominal_n_train": 100,
                "estimated_n_eff_train": 80.0,
                "warnings": [],
            },
            thresholds={"graph2mat": {"N_min_abs": 20}},
            aggregation_mode="best_config_mean",
            requested_n_min_source="fit",
            actual_n_min_source="fit",
            requested_fit_model="cumulative_best",
            actual_fit_model="cumulative_best",
            fit_threshold_details={
                "graph2mat": {
                    "fit_policy": "diagnostic_only",
                    "paper_candidate": False,
                    "fit_model": "cumulative_best",
                    "status": "ok",
                }
            },
            fallback_used=False,
            fallback_reason=None,
        )

        self.assertEqual(status["scientific_claim_status"], "diagnostic_only")
        self.assertIn(
            "paper_blocked_if_n_min_fit_policy_diagnostic_only:cumulative_best",
            status["paper_level_blockers"],
        )

    def test_scientific_status_blocks_observed_or_none_protocol(self) -> None:
        status = minimum.scientific_claim_status_payload(
            temporal_diagnostics={
                "autocorrelation_available": True,
                "nominal_n_train": 100,
                "estimated_n_eff_train": 80.0,
                "warnings": [],
            },
            thresholds={"graph2mat": {"N_min_abs": 20}},
            aggregation_mode="mean_seeds_per_config",
            requested_n_min_source="fit",
            actual_n_min_source="observed",
            requested_fit_model="none",
            actual_fit_model="none",
            fit_threshold_details={
                "graph2mat": {
                    "fit_policy": "diagnostic_only",
                    "paper_candidate": False,
                    "fit_model": "none",
                    "status": "not_used",
                }
            },
            fallback_used=False,
            fallback_reason=None,
        )

        self.assertEqual(status["scientific_claim_status"], "diagnostic_only")
        self.assertIn(
            "paper_blocked_if_n_min_source_observed_without_locked_protocol",
            status["paper_level_blockers"],
        )

    def test_scientific_status_blocks_fit_fallback(self) -> None:
        status = minimum.scientific_claim_status_payload(
            temporal_diagnostics={
                "autocorrelation_available": True,
                "nominal_n_train": 100,
                "estimated_n_eff_train": 80.0,
                "warnings": [],
            },
            thresholds={"graph2mat": {"N_min_abs": 20}},
            aggregation_mode="mean_seeds_per_config",
            requested_n_min_source="fit",
            actual_n_min_source="observed",
            requested_fit_model="quadratic",
            actual_fit_model=None,
            fit_threshold_details={},
            fallback_used=True,
            fallback_reason="canonical_fit_failed:quadratic",
        )

        self.assertEqual(status["scientific_claim_status"], "diagnostic_only")
        self.assertIn(
            "paper_blocked_if_fit_failed_or_fallback_used",
            status["paper_level_blockers"],
        )
        self.assertIn(
            "paper_blocked_if_actual_fit_model_missing",
            status["paper_level_blockers"],
        )

    def test_scientific_status_blocks_missing_fit_policy_by_method(self) -> None:
        status = minimum.scientific_claim_status_payload(
            temporal_diagnostics={
                "autocorrelation_available": True,
                "nominal_n_train": 100,
                "estimated_n_eff_train": 80.0,
                "warnings": [],
            },
            thresholds={"graph2mat": {"N_min_abs": 20}},
            aggregation_mode="mean_seeds_per_config",
            requested_n_min_source="fit",
            actual_n_min_source="fit",
            requested_fit_model="power_law_floor",
            actual_fit_model="power_law_floor",
            fit_threshold_details={},
            fallback_used=False,
            fallback_reason=None,
        )

        self.assertEqual(status["scientific_claim_status"], "diagnostic_only")
        self.assertIn(
            "paper_blocked_if_fit_policy_missing:graph2mat",
            status["paper_level_blockers"],
        )

    def test_analyze_includes_temporal_diagnostics_with_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = write_synthetic_temporal_dataset(base / "dataset", n_train=10)
            run_root = base / "run"
            payload = {
                "metric_scaling_rows": [
                    {
                        "method": "graph2mat",
                        "dataset_size": 10,
                        "dataset_root": str(dataset_root),
                        "config_id": "a",
                        "epoch_label": "10 epochs",
                        "metric_key": "h_mae_eV_mean",
                        "metric_value": 0.010,
                        "seed": "1",
                    }
                ]
            }
            (run_root / "summary" / "ranking").mkdir(parents=True)
            (run_root / "summary" / "ranking" / "normalized_run_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            summary = minimum.analyze(
                make_analyze_args(run_root=[str(run_root)], output_dir=str(base / "out"))
            )
            self.assertEqual(summary["nominal_n_train"], 10)
            self.assertIn("N_min_nominal", summary)
            self.assertIn("effective_samples_at_N_min_nominal", summary)
            self.assertIn("effective_samples_at_nominal_N_min_diagnostic", summary)
            self.assertIn("scientific_claim_status", summary)
            self.assertIn("datasets", summary["temporal_diagnostics"])
            self.assertEqual(len(summary["temporal_diagnostics"]["datasets"]), 1)

    def test_report_includes_nominal_vs_effective_warning(self) -> None:
        report = minimum.build_report(
            output_dir=Path("/tmp/out"),
            run_roots=[Path("/tmp/run")],
            grouped_rows=[],
            best_rows=[],
            thresholds={"graph2mat": {"N_min_abs": 10}},
            fits={},
            warnings=[],
            primary_metric="h_mae_eV_mean",
            threshold_mev=10.0,
            x_axis="n_train",
            temporal_diagnostics={
                "autocorrelation_available": False,
                "N_eff_by_dataset_size": {"10": 8.0},
                "N_eff_over_N_by_dataset_size": {"10": 0.8},
                "autocorrelation_available_by_dataset_size": {"10": True},
                "temporal_block_diagnostics_by_dataset_size": {
                    "10": {
                        "n_datasets": 1,
                        "datasets": [{"block_diagnostics": {"block_a": {}}}],
                    }
                },
            },
            scientific_status={
                "n_min_basis": "nominal",
                "scientific_claim_status": "diagnostic_only",
                "paper_level_blockers": ["paper_blocked_if_autocorrelation_unavailable"],
                "N_eff_over_N_nominal": None,
                "N_min_effective_diagnostic": {"graph2mat": {"N_min_abs": 8.0}},
            },
            replicate_bootstrap={
                "enabled": True,
                "display_label": "replicate-resampling CI",
                "warnings": ["replicate_bootstrap_no_temporal_or_block_bootstrap"],
            },
            cost_basis="protocol_total",
        )

        self.assertIn(
            "N_min uses nominal N. If MD snapshots are autocorrelated, independent sample count can be lower.",
            report,
        )
        self.assertIn("Scientific claim status", report)
        self.assertIn("replicate-resampling CI", report)
        self.assertIn("Hierarchical uncertainty", report)
        self.assertIn("hierarchical uncertainty (paper-readiness audit)", report)
        self.assertIn("Fit stability (leave-one-size-out)", report)
        self.assertIn("not temporal/block bootstrap", report)
        self.assertIn("does not model temporal autocorrelation", report)
        self.assertIn("does not model model-selection uncertainty", report)
        self.assertIn("does not model hyperparameter-selection uncertainty", report)
        self.assertIn("does not model dependence between dataset sizes", report)
        self.assertIn("N_min_cost_eff CI available: False", report)
        self.assertIn("N_min_cost_eff has no replicate-resampling CI", report)
        self.assertIn("true effective-N thresholding is not implemented", report)
        self.assertIn("Dataset size (nominal N_train)", report)
        self.assertIn("N_min_effective_diagnostic", report)
        self.assertNotIn("N_min_eff_diagnostic", report)
        self.assertIn("Cost basis for `N_min_cost_eff`: `protocol_total`", report)

    def test_report_includes_power_law_fit_search_diagnostics(self) -> None:
        fit = minimum.fit_power_law_floor(
            [10.0, 20.0, 40.0, 80.0, 160.0],
            [30.0, 18.0, 12.0, 9.0, 7.5],
        )
        report = minimum.build_report(
            output_dir=Path("/tmp/out"),
            run_roots=[Path("/tmp/run")],
            grouped_rows=[],
            best_rows=[],
            thresholds={"graph2mat": {"N_min_abs": 10}},
            fits={"graph2mat": {"power_law_floor": fit}},
            warnings=[],
            primary_metric="h_mae_eV_mean",
            threshold_mev=10.0,
            x_axis="n_train",
            temporal_diagnostics={"autocorrelation_available": False},
            scientific_status={
                "n_min_basis": "nominal",
                "scientific_claim_status": "diagnostic_only",
                "paper_level_blockers": ["paper_blocked_if_autocorrelation_unavailable"],
                "N_eff_over_N_nominal": None,
            },
            replicate_bootstrap=minimum.disabled_bootstrap_summary(),
            cost_basis="protocol_total",
        )

        self.assertIn("alpha search", report)
        self.assertIn("SSE", report)
        self.assertIn("coarse_grid_plus_golden_section", report)

    def test_report_includes_threshold_protocol_and_sensitivity(self) -> None:
        report = minimum.build_report(
            output_dir=Path("/tmp/out"),
            run_roots=[Path("/tmp/run")],
            grouped_rows=[],
            best_rows=[],
            thresholds={"graph2mat": {"N_min_abs": 10}},
            fits={},
            warnings=[],
            primary_metric="h_mae_eV_mean",
            threshold_mev=10.0,
            x_axis="n_train",
            temporal_diagnostics={"autocorrelation_available": False},
            scientific_status={
                "threshold_policy": {
                    "threshold_basis": minimum.THRESHOLD_BASIS_EXPLICIT_PROTOCOL,
                    "threshold_reference": "internal_protocol_v1",
                    "threshold_metric_family": "hamiltonian_element_error_mev",
                    "threshold_is_user_defined": False,
                    "threshold_interpretation": "Protocol-backed threshold.",
                    "threshold_protocol_file": "/tmp/threshold_protocol.json",
                    "threshold_protocol_physical_rationale": "Test-only documented rationale.",
                    "threshold_protocol_applies_to_metrics": ["h_mae_eV_mean"],
                    "threshold_protocol_sensitivity_recommendation": "Audit nearby thresholds before paper-level use.",
                },
                "n_min_basis": "nominal",
                "scientific_claim_status": "diagnostic_only",
                "paper_level_blockers": ["paper_blocked_if_threshold_sensitivity_unstable:graph2mat"],
                "N_eff_over_N_nominal": None,
            },
            replicate_bootstrap=minimum.disabled_bootstrap_summary(),
            cost_basis="protocol_total",
            threshold_sensitivity={
                "status": "ok",
                "thresholds_mev": [8.0, 10.0, 12.0],
                "by_method": {
                    "graph2mat": {
                        "n_min_abs_span": 20,
                        "allowed_n_min_abs_delta": 10,
                        "unstable": True,
                        "threshold_series": [
                            {"threshold_mev": 8.0, "N_min_abs": 30},
                            {"threshold_mev": 10.0, "N_min_abs": 20},
                            {"threshold_mev": 12.0, "N_min_abs": 10},
                        ],
                        "paper_level_blockers": ["paper_blocked_if_threshold_sensitivity_unstable:graph2mat"],
                    }
                },
            },
        )
        self.assertIn("Threshold protocol file", report)
        self.assertIn("Threshold physical rationale", report)
        self.assertIn("Threshold sensitivity", report)
        self.assertIn("paper_blocked_if_threshold_sensitivity_unstable:graph2mat", report)

    def test_documentation_clarifies_effective_samples_at_nominal_n_min(self) -> None:
        doc = (REPO_ROOT / "Comparison" / "scripts" / "DATASET_SIZE_MINIMUM.md").read_text(encoding="utf-8")
        self.assertIn("effective_samples_at_nominal_N_min_diagnostic", doc)
        self.assertIn("effective-N threshold is not implemented", doc)
        self.assertIn("N_min_eff_diagnostic_deprecated_alias_for", doc)
        self.assertIn("alpha_search_method", doc)
        self.assertIn("golden-section refinement", doc)
        self.assertIn("hierarchical_uncertainty", doc)
        self.assertIn("paper_uncertainty_block_hierarchy_unavailable", doc)
        self.assertIn("--threshold-protocol-file", doc)
        self.assertIn("threshold_sensitivity", doc)
        self.assertIn("paper_blocked_if_threshold_sensitivity_unstable", doc)

    def test_documentation_and_ui_use_replicate_resampling_ci_wording(self) -> None:
        doc = (REPO_ROOT / "Comparison" / "scripts" / "DATASET_SIZE_MINIMUM.md").read_text(encoding="utf-8")
        ui_js = (REPO_ROOT / "Comparison" / "ui" / "app.js").read_text(encoding="utf-8")

        self.assertIn("replicate-resampling CI", doc)
        self.assertIn("not** a temporal/block bootstrap", doc)
        self.assertIn("replicate-resampling CI", ui_js)
        self.assertIn("not temporal/block bootstrap", ui_js)
        self.assertIn("Hierarchical uncertainty:", ui_js)
        self.assertIn("hierarchical uncertainty", ui_js)
        self.assertNotIn("Bootstrap CI", ui_js)

    def test_summarize_temporal_diagnostics_includes_convention(self) -> None:
        summary = minimum.summarize_temporal_diagnostics([], run_roots=[])
        self.assertEqual(
            summary["autocorrelation_convention"],
            minimum.AUTOCORRELATION_CONVENTION,
        )
        self.assertEqual(summary["n_eff_convention"], minimum.N_EFF_CONVENTION)
        self.assertIn("N is nominal; N_eff not estimated", summary["status_message"])


if __name__ == "__main__":
    unittest.main()
