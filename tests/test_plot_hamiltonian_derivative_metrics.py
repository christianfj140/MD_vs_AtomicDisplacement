from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "plot_hamiltonian_derivative_metrics.py"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["sample"])
        writer.writeheader()
        writer.writerows(rows)


class PlotHamiltonianDerivativeMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def derivative_root(self, name: str) -> Path:
        return self.root / name / "derivative_metrics"

    def write_dataset_root(self, name: str, *, train: int, validation: int = 1, test: int = 1) -> Path:
        dataset_root = self.root / "datasets" / name
        write_json(
            dataset_root / "frozen_split_manifest.json",
            {
                "schema": "joint_graph2mat_deeph_frozen_split_manifest_v1",
                "split_counts": {"train": train, "validation": validation, "test": test},
            },
        )
        return dataset_root

    def write_derivative_fixture(
        self,
        name: str,
        *,
        source_model: str,
        dataset_root: Path | None = None,
        dataset_id: str = "",
        rows: list[dict[str, object]] | None = None,
        quantile_rows: list[dict[str, object]] | None = None,
        group_metrics: dict[str, object] | None = None,
        onsite_offsite_rows: list[dict[str, object]] | None = None,
        onsite_offsite_payload: dict[str, object] | None = None,
        hermiticity_rows: list[dict[str, object]] | None = None,
        scientific_status: str = "diagnostic_only",
        stencils_failed: int = 0,
        fatal_errors: list[dict[str, object]] | None = None,
    ) -> Path:
        derivative_root = self.derivative_root(name)
        write_json(
            derivative_root / "manifest.json",
            {
                "schema_version": "hamiltonian_derivative_metrics_v1",
                "scientific_status": scientific_status,
                "force_constants_used": False,
                "stencils_total": len(rows or []),
                "stencils_ok": len(rows or []),
                "stencils_failed": stencils_failed,
                "fatal_errors": fatal_errors or [],
                "dataset_root": str(dataset_root) if dataset_root is not None else "",
                "dataset_id": dataset_id,
            },
        )
        write_csv(derivative_root / "derivative_matrix_metrics.csv", rows or [])
        if quantile_rows is not None:
            write_csv(derivative_root / "derivative_ref_abs_quantile_metrics.csv", quantile_rows)
        if group_metrics is not None:
            write_json(derivative_root / "derivative_group_metrics.json", group_metrics)
        if onsite_offsite_rows is not None:
            write_csv(derivative_root / "derivative_onsite_offsite_metrics.csv", onsite_offsite_rows)
        if onsite_offsite_payload is not None:
            write_json(derivative_root / "derivative_onsite_offsite_metrics.json", onsite_offsite_payload)
        write_csv(derivative_root / "derivative_hermiticity.csv", hermiticity_rows or [])
        write_csv(derivative_root / "stencil_status.csv", [])
        return derivative_root

    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_plot_script_handles_empty_metrics_gracefully(self) -> None:
        derivative_root = self.write_derivative_fixture("empty", source_model="graph2mat")
        output_dir = self.root / "plots_empty"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["available"])
        self.assertTrue(payload["diagnostic_only"])
        self.assertTrue(payload["scientific_warnings"])

    def test_plot_script_writes_manifest(self) -> None:
        derivative_root = self.write_derivative_fixture(
            "single",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                    "dh_false_zero_rate": 0.1,
                    "dh_false_nonzero_rate": 0.05,
                }
            ],
            hermiticity_rows=[
                {
                    "sample": "s0",
                    "finite_difference_method": "central",
                    "dH_ref_hermiticity_defect": 0.0,
                    "dH_pred_hermiticity_defect": 0.02,
                    "dH_hermiticity_error_delta": 0.02,
                }
            ],
        )
        output_dir = self.root / "plots_manifest"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        manifest = json.loads((output_dir / "derivative_plot_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "hamiltonian_derivative_plot_manifest_v1")
        self.assertEqual(manifest["title"], "Hamiltonian derivative diagnostics")
        self.assertTrue((output_dir / "derivative_plot_payload.json").exists())

    def test_plot_metadata_contains_scientific_warnings(self) -> None:
        derivative_root = self.write_derivative_fixture(
            "warnings",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "forward",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                    "dh_false_zero_rate": 0.1,
                    "dh_false_nonzero_rate": 0.05,
                }
            ],
            scientific_status="diagnostic_only",
            stencils_failed=1,
            fatal_errors=[{"kind": "incomplete_derivative_stencil"}],
        )
        output_dir = self.root / "plots_warnings"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        codes = {warning["code"] for warning in payload["scientific_warnings"]}
        self.assertIn("scientific_status_diagnostic", codes)
        self.assertIn("failed_stencils_present", codes)
        self.assertIn("fatal_errors_present", codes)
        self.assertIn("robust_derivative_metrics_missing", codes)
        self.assertIn("derivative_correlation_or_residual_metrics_missing", codes)
        self.assertIn("derivative_ref_abs_quantile_metrics_missing", codes)
        self.assertIn("derivative_group_metrics_missing", codes)
        self.assertIn("derivative_onsite_offsite_metrics_missing", codes)
        plot_ids = {plot["id"] for plot in payload["plots"]}
        self.assertIn("dh_mae_by_model", plot_ids)
        self.assertIn("relative_frobenius_by_model", plot_ids)
        self.assertIn("error_by_atom_index_zero_based", plot_ids)
        self.assertIn("error_by_axis", plot_ids)
        plots = {plot["id"]: plot for plot in payload["plots"]}
        required_semantic_fields = {
            "data_grain",
            "aggregation",
            "x_semantics",
            "y_semantics",
            "lower_is_better",
            "higher_is_better",
            "units",
            "diagnostic_only",
        }
        for plot in payload["plots"]:
            self.assertTrue(required_semantic_fields.issubset(plot), plot["id"])
        self.assertEqual(plots["error_vs_delta"]["data_grain"], "sample_stencil_delta")
        self.assertEqual(plots["error_vs_delta"]["aggregation"], "none")
        self.assertEqual(plots["error_vs_delta"]["x_semantics"], "delta Ang")
        self.assertEqual(plots["error_vs_delta"]["units"]["x"], "Ang")
        self.assertTrue(plots["error_vs_delta"]["lower_is_better"])
        self.assertEqual(plots["dh_mae_by_model"]["data_grain"], "model_mean")
        self.assertEqual(plots["dh_mae_by_model"]["aggregation"], "mean over rows grouped by model")
        self.assertEqual(plots["dh_mae_by_model"]["units"]["dh_mae_union_eV_per_Ang"], "eV/Ang")
        self.assertNotIn("relative_frobenius_union_robust_by_model", plot_ids)
        self.assertNotIn("derivative_correlation_by_model", plot_ids)
        self.assertNotIn("derivative_error_by_abs_ref_quantile", plot_ids)
        self.assertNotIn("robust_error_by_displaced_atom", plot_ids)
        self.assertNotIn("onsite_offsite_derivative_error", plot_ids)

    def test_onsite_offsite_metrics_are_exposed_when_csv_exists(self) -> None:
        derivative_root = self.write_derivative_fixture(
            "onsite_offsite",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                }
            ],
            onsite_offsite_rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_onsite_relative_frobenius_robust": 0.2,
                    "dh_onsite_mae_eV_per_Ang": 0.03,
                    "dh_onsite_rmse_eV_per_Ang": 0.04,
                    "dh_offsite_relative_frobenius_robust": 0.5,
                    "dh_offsite_mae_eV_per_Ang": 0.06,
                    "dh_offsite_rmse_eV_per_Ang": 0.08,
                },
                {
                    "sample": "s1",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 1,
                    "axis": "y",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_onsite_relative_frobenius_robust": 0.4,
                    "dh_onsite_mae_eV_per_Ang": 0.05,
                    "dh_onsite_rmse_eV_per_Ang": 0.06,
                    "dh_offsite_relative_frobenius_robust": 0.7,
                    "dh_offsite_mae_eV_per_Ang": 0.1,
                    "dh_offsite_rmse_eV_per_Ang": 0.12,
                },
            ],
        )
        output_dir = self.root / "plots_onsite_offsite"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        plots = {plot["id"]: plot for plot in payload["plots"]}
        self.assertIn("onsite_offsite_derivative_error", plots)
        self.assertIn("onsite_offsite_derivative_error", payload["diagnostic_plot_ids"])
        plot = plots["onsite_offsite_derivative_error"]
        self.assertEqual(plot["kind"], "grouped_bar")
        self.assertEqual(
            {metric["key"] for metric in plot["metrics"]},
            {
                "dh_onsite_relative_frobenius_robust",
                "dh_offsite_relative_frobenius_robust",
                "dh_onsite_mae_eV_per_Ang",
                "dh_offsite_mae_eV_per_Ang",
            },
        )
        self.assertAlmostEqual(plot["rows"][0]["dh_onsite_relative_frobenius_robust"], 0.3)
        self.assertAlmostEqual(plot["rows"][0]["dh_offsite_relative_frobenius_robust"], 0.6)
        self.assertAlmostEqual(plot["rows"][0]["dh_onsite_mae_eV_per_Ang"], 0.04)
        self.assertAlmostEqual(plot["rows"][0]["dh_offsite_mae_eV_per_Ang"], 0.08)
        self.assertNotIn("derivative_onsite_offsite_metrics_missing", {warning["code"] for warning in payload["scientific_warnings"]})

    def test_invalid_geometry_rows_are_marked_and_excluded_from_model_aggregates(self) -> None:
        derivative_root = self.write_derivative_fixture(
            "invalid_geometry",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "valid",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                },
                {
                    "sample": "invalid",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.02,
                    "finite_difference_method": "central",
                    "invalid_geometry": True,
                    "geometry_validation_failed": True,
                    "geometry_issue_codes": "displacement_vector_mismatch",
                    "dh_mae_union_eV_per_Ang": 9.9,
                    "dh_rmse_union_eV_per_Ang": 9.9,
                    "dh_relative_frobenius_ref": 9.9,
                },
            ],
        )
        output_dir = self.root / "plots_invalid_geometry"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        warnings = {warning["code"]: warning for warning in payload["scientific_warnings"]}
        self.assertIn("invalid_geometry_rows_excluded_from_aggregates", warnings)
        self.assertEqual(warnings["invalid_geometry_rows_excluded_from_aggregates"]["severity"], "severe")
        self.assertEqual(warnings["invalid_geometry_rows_excluded_from_aggregates"]["details"]["invalid_rows"], 1)
        plots = {plot["id"]: plot for plot in payload["plots"]}
        self.assertAlmostEqual(plots["dh_mae_by_model"]["rows"][0]["dh_mae_union_eV_per_Ang"], 0.2)
        invalid_points = [row for row in plots["error_vs_delta"]["rows"] if row["sample"] == "invalid"]
        self.assertEqual(len(invalid_points), 1)
        self.assertTrue(invalid_points[0]["invalid_geometry"])

    def test_onsite_offsite_unavailable_json_warns_without_plot(self) -> None:
        derivative_root = self.write_derivative_fixture(
            "onsite_offsite_unavailable",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                }
            ],
            onsite_offsite_payload={"available": False, "reason": "orbital_to_atom_mapping_unavailable"},
        )
        output_dir = self.root / "plots_onsite_offsite_unavailable"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        self.assertNotIn("onsite_offsite_derivative_error", {plot["id"] for plot in payload["plots"]})
        warnings = {warning["code"]: warning for warning in payload["scientific_warnings"]}
        self.assertIn("derivative_onsite_offsite_metrics_unavailable", warnings)
        self.assertEqual(
            warnings["derivative_onsite_offsite_metrics_unavailable"]["details"]["reason"],
            "orbital_to_atom_mapping_unavailable",
        )

    def test_ref_abs_quantile_metrics_are_exposed_when_csv_exists(self) -> None:
        derivative_root = self.write_derivative_fixture(
            "quantiles",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                }
            ],
            quantile_rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "reference_source": "siesta",
                    "base_sample_id": "base",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "support_threshold": 1e-12,
                    "quantile_bin": 1,
                    "n_entries": 2,
                    "abs_ref_min_eV_per_Ang": 0.0,
                    "abs_ref_max_eV_per_Ang": 0.5,
                    "abs_ref_mean_eV_per_Ang": 0.25,
                    "dh_error_mae_eV_per_Ang": 0.1,
                    "dh_error_rmse_eV_per_Ang": 0.2,
                    "dh_error_relative_l1_robust": 0.3,
                },
                {
                    "sample": "s1",
                    "source_model": "graph2mat",
                    "reference_source": "siesta",
                    "base_sample_id": "base",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "support_threshold": 1e-12,
                    "quantile_bin": 1,
                    "n_entries": 3,
                    "abs_ref_min_eV_per_Ang": 0.1,
                    "abs_ref_max_eV_per_Ang": 0.6,
                    "abs_ref_mean_eV_per_Ang": 0.35,
                    "dh_error_mae_eV_per_Ang": 0.3,
                    "dh_error_rmse_eV_per_Ang": 0.4,
                    "dh_error_relative_l1_robust": 0.5,
                },
                {
                    "sample": "s2",
                    "source_model": "graph2mat",
                    "reference_source": "siesta",
                    "base_sample_id": "base",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "support_threshold": 1e-12,
                    "quantile_bin": 2,
                    "n_entries": 4,
                    "abs_ref_min_eV_per_Ang": 0.6,
                    "abs_ref_max_eV_per_Ang": 1.0,
                    "abs_ref_mean_eV_per_Ang": 0.8,
                    "dh_error_mae_eV_per_Ang": 0.6,
                    "dh_error_rmse_eV_per_Ang": 0.7,
                    "dh_error_relative_l1_robust": 0.8,
                },
            ],
        )
        output_dir = self.root / "plots_quantiles"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        plot_ids = {plot["id"] for plot in payload["plots"]}
        self.assertIn("derivative_error_by_abs_ref_quantile", plot_ids)
        self.assertIn("derivative_relative_l1_by_abs_ref_quantile", plot_ids)
        self.assertIn("derivative_error_by_abs_ref_quantile", payload["diagnostic_plot_ids"])
        plots = {plot["id"]: plot for plot in payload["plots"]}
        error_plot = plots["derivative_error_by_abs_ref_quantile"]
        self.assertEqual(error_plot["x_key"], "quantile_bin")
        self.assertEqual(error_plot["series_key"], "model_label")
        first_row = error_plot["rows"][0]
        self.assertEqual(first_row["quantile_bin"], 1)
        self.assertEqual(first_row["n_entries_total"], 5)
        self.assertAlmostEqual(first_row["dh_error_mae_eV_per_Ang"], 0.2)
        self.assertAlmostEqual(first_row["dh_error_rmse_eV_per_Ang"], 0.3)
        relative_plot = plots["derivative_relative_l1_by_abs_ref_quantile"]
        self.assertAlmostEqual(relative_plot["rows"][0]["dh_error_relative_l1_robust"], 0.4)
        self.assertNotIn("derivative_ref_abs_quantile_metrics_missing", {warning["code"] for warning in payload["scientific_warnings"]})

    def test_group_metrics_are_exposed_when_json_exists(self) -> None:
        derivative_root = self.write_derivative_fixture(
            "group_metrics",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                }
            ],
            group_metrics={
                "schema": "hamiltonian_derivative_group_metrics_v1",
                "by_atom": [
                    {
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "n_stencils": 2,
                        "dh_relative_frobenius_union_robust_mean": 0.4,
                        "dh_relative_frobenius_union_robust_pooled": 0.25,
                        "dh_relative_l1_union_robust_mean": 0.6,
                    }
                ],
                "by_axis": [
                    {
                        "source_model": "graph2mat",
                        "axis": "x",
                        "n_stencils": 2,
                        "dh_relative_frobenius_union_robust_mean": 0.4,
                        "dh_relative_frobenius_union_robust_pooled": 0.25,
                        "dh_relative_l1_union_robust_mean": 0.6,
                        "dh_relative_l1_union_robust_pooled": 0.5,
                    }
                ],
                "by_atom_axis": [
                    {
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "n_stencils": 2,
                        "dh_relative_frobenius_union_robust_mean": 0.4,
                        "dh_relative_frobenius_union_robust_pooled": 0.25,
                        "dh_relative_l1_union_robust_mean": 0.6,
                        "dh_relative_l1_union_robust_pooled": 0.5,
                    }
                ],
            },
        )
        output_dir = self.root / "plots_group_metrics"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        plot_ids = {plot["id"] for plot in payload["plots"]}
        self.assertIn("error_by_atom_index_zero_based", plot_ids)
        self.assertIn("error_by_axis", plot_ids)
        self.assertIn("robust_error_by_displaced_atom", plot_ids)
        self.assertIn("robust_error_by_axis", plot_ids)
        self.assertIn("robust_error_by_atom_axis", plot_ids)
        plots = {plot["id"]: plot for plot in payload["plots"]}
        atom_plot = plots["robust_error_by_displaced_atom"]
        self.assertEqual(atom_plot["x_key"], "atom_index_zero_based")
        self.assertEqual(atom_plot["metrics"][0]["key"], "dh_relative_frobenius_union_robust_pooled")
        self.assertAlmostEqual(atom_plot["rows"][0]["dh_relative_frobenius_union_robust_pooled"], 0.25)
        axis_plot = plots["robust_error_by_axis"]
        self.assertEqual(axis_plot["x_key"], "axis")
        self.assertEqual(
            {metric["key"] for metric in axis_plot["metrics"]},
            {"dh_relative_frobenius_union_robust_pooled", "dh_relative_l1_union_robust_pooled"},
        )
        atom_axis_plot = plots["robust_error_by_atom_axis"]
        self.assertEqual(atom_axis_plot["x_key"], "atom_axis")
        self.assertEqual(atom_axis_plot["rows"][0]["atom_axis"], "0:x")
        self.assertNotIn("derivative_group_metrics_missing", {warning["code"] for warning in payload["scientific_warnings"]})

    def test_robust_derivative_metric_plots_are_exposed_when_columns_exist(self) -> None:
        derivative_root = self.write_derivative_fixture(
            "robust",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                    "dh_relative_frobenius_union_robust": 0.25,
                    "dh_relative_l1_union_robust": 0.5,
                    "dh_norm_ref_union_fro": 2.0,
                    "dh_norm_error_union_fro": 0.5,
                    "dh_norm_ref_l1_union": 4.0,
                    "dh_norm_error_l1_union": 2.0,
                    "dh_false_zero_rate": 0.1,
                    "dh_false_nonzero_rate": 0.05,
                }
            ],
        )
        output_dir = self.root / "plots_robust"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        plot_ids = {plot["id"] for plot in payload["plots"]}
        self.assertIn("dh_mae_by_model", plot_ids)
        self.assertIn("relative_frobenius_by_model", plot_ids)
        self.assertIn("relative_frobenius_union_robust_by_model", plot_ids)
        self.assertIn("relative_l1_union_robust_by_model", plot_ids)
        self.assertIn("robust_primary_metrics_by_model", plot_ids)
        self.assertIn("relative_frobenius_union_robust_by_model", payload["diagnostic_plot_ids"])
        plots = {plot["id"]: plot for plot in payload["plots"]}
        frob_rows = plots["relative_frobenius_union_robust_by_model"]["rows"]
        self.assertAlmostEqual(frob_rows[0]["dh_relative_frobenius_union_robust"], 0.25)
        l1_rows = plots["relative_l1_union_robust_by_model"]["rows"]
        self.assertAlmostEqual(l1_rows[0]["dh_relative_l1_union_robust"], 0.5)
        combined = plots["robust_primary_metrics_by_model"]
        combined_keys = {metric["key"] for metric in combined["metrics"]}
        self.assertEqual(
            combined_keys,
            {
                "dh_relative_frobenius_union_robust",
                "dh_relative_l1_union_robust",
                "dh_mae_union_eV_per_Ang",
                "dh_rmse_union_eV_per_Ang",
            },
        )
        self.assertAlmostEqual(combined["rows"][0]["dh_relative_frobenius_union_robust"], 0.25)
        self.assertAlmostEqual(combined["rows"][0]["dh_relative_l1_union_robust"], 0.5)
        self.assertNotIn("robust_derivative_metrics_missing", {warning["code"] for warning in payload["scientific_warnings"]})

    def test_correlation_and_residual_metric_plots_are_exposed_when_columns_exist(self) -> None:
        derivative_root = self.write_derivative_fixture(
            "correlation_residual",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                    "dh_pearson_union": 0.91,
                    "dh_spearman_union": 0.87,
                    "dh_residual_mean_union_eV_per_Ang": -0.01,
                    "dh_residual_std_union_eV_per_Ang": 0.06,
                    "dh_residual_median_union_eV_per_Ang": -0.02,
                    "dh_residual_abs_p90_union_eV_per_Ang": 0.1,
                    "dh_residual_abs_p95_union_eV_per_Ang": 0.12,
                    "dh_residual_abs_p99_union_eV_per_Ang": 0.2,
                    "dh_residual_bias_over_mae_union": 0.05,
                    "dh_false_zero_rate": 0.1,
                    "dh_false_nonzero_rate": 0.05,
                }
            ],
        )
        output_dir = self.root / "plots_correlation_residual"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        plot_ids = {plot["id"] for plot in payload["plots"]}
        self.assertIn("dh_mae_by_model", plot_ids)
        self.assertIn("relative_frobenius_by_model", plot_ids)
        self.assertIn("derivative_correlation_by_model", plot_ids)
        self.assertIn("derivative_residual_summary_by_model", plot_ids)
        self.assertIn("derivative_residual_tail_by_model", plot_ids)
        self.assertIn("derivative_correlation_by_model", payload["diagnostic_plot_ids"])
        plots = {plot["id"]: plot for plot in payload["plots"]}
        correlation = plots["derivative_correlation_by_model"]
        self.assertAlmostEqual(correlation["rows"][0]["dh_pearson_union"], 0.91)
        self.assertAlmostEqual(correlation["rows"][0]["dh_spearman_union"], 0.87)
        residual = plots["derivative_residual_summary_by_model"]
        self.assertAlmostEqual(residual["rows"][0]["dh_residual_mean_union_eV_per_Ang"], -0.01)
        self.assertAlmostEqual(residual["rows"][0]["dh_residual_bias_over_mae_union"], 0.05)
        tail = plots["derivative_residual_tail_by_model"]
        self.assertAlmostEqual(tail["rows"][0]["dh_residual_abs_p99_union_eV_per_Ang"], 0.2)
        self.assertNotIn(
            "derivative_correlation_or_residual_metrics_missing",
            {warning["code"] for warning in payload["scientific_warnings"]},
        )

    def test_no_paper_level_wording_appears(self) -> None:
        graph2mat_root = self.write_derivative_fixture(
            "graph2mat",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "shared",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                    "dh_false_zero_rate": 0.1,
                    "dh_false_nonzero_rate": 0.05,
                }
            ],
            hermiticity_rows=[
                {
                    "sample": "shared",
                    "finite_difference_method": "central",
                    "dH_ref_hermiticity_defect": 0.0,
                    "dH_pred_hermiticity_defect": 0.02,
                    "dH_hermiticity_error_delta": 0.02,
                }
            ],
        )
        deeph_root = self.write_derivative_fixture(
            "deeph",
            source_model="deeph",
            rows=[
                {
                    "sample": "shared",
                    "source_model": "deeph",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.15,
                    "dh_rmse_union_eV_per_Ang": 0.25,
                    "dh_relative_frobenius_ref": 0.35,
                    "dh_false_zero_rate": 0.05,
                    "dh_false_nonzero_rate": 0.02,
                }
            ],
            hermiticity_rows=[
                {
                    "sample": "shared",
                    "finite_difference_method": "central",
                    "dH_ref_hermiticity_defect": 0.0,
                    "dH_pred_hermiticity_defect": 0.01,
                    "dH_hermiticity_error_delta": 0.01,
                }
            ],
        )
        output_dir = self.root / "plots_labels"

        completed = self.run_script(
            "--graph2mat-root",
            str(graph2mat_root),
            "--deeph-root",
            str(deeph_root),
            "--output-dir",
            str(output_dir),
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload_text = (output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8").lower()
        self.assertNotIn("paper-level", payload_text)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["title"], "Hamiltonian derivative diagnostics")
        self.assertEqual(payload["reference_label"], "Reference: finite differences of SIESTA Hamiltonians")
        self.assertEqual(payload["force_constants_label"], "SIESTA force constants are not treated as dH/dR")
        paired_plot = next(plot for plot in payload["plots"] if plot["id"] == "graph2mat_vs_deeph_paired_comparison")
        self.assertEqual(len(paired_plot["rows"]), 1)

    def test_dataset_size_plots_aggregate_by_model_and_n_train(self) -> None:
        dataset_12 = self.write_dataset_root("dataset_12", train=12)
        dataset_50 = self.write_dataset_root("dataset_50", train=50)
        roots = [
            self.write_derivative_fixture(
                "graph2mat_n12",
                source_model="graph2mat",
                dataset_root=dataset_12,
                dataset_id="n12",
                rows=[
                    {
                        "sample": "s0",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.2,
                        "dh_rmse_union_eV_per_Ang": 0.3,
                        "dh_relative_frobenius_ref": 0.4,
                        "dh_support_f1": 0.8,
                        "dh_false_zero_rate": 0.1,
                        "dh_false_nonzero_rate": 0.05,
                    },
                    {
                        "sample": "s1",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 1,
                        "axis": "y",
                        "delta_ang": 0.02,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.4,
                        "dh_rmse_union_eV_per_Ang": 0.5,
                        "dh_relative_frobenius_ref": 0.6,
                        "dh_support_f1": 0.6,
                        "dh_false_zero_rate": 0.2,
                        "dh_false_nonzero_rate": 0.15,
                    },
                ],
            ),
            self.write_derivative_fixture(
                "deeph_n12",
                source_model="deeph",
                dataset_root=dataset_12,
                dataset_id="n12",
                rows=[
                    {
                        "sample": "s0",
                        "source_model": "deeph",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.1,
                        "dh_rmse_union_eV_per_Ang": 0.2,
                        "dh_relative_frobenius_ref": 0.3,
                        "dh_support_f1": 0.9,
                        "dh_false_zero_rate": 0.05,
                        "dh_false_nonzero_rate": 0.02,
                    }
                ],
            ),
            self.write_derivative_fixture(
                "graph2mat_n50",
                source_model="graph2mat",
                dataset_root=dataset_50,
                dataset_id="n50",
                rows=[
                    {
                        "sample": "s0",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.08,
                        "dh_rmse_union_eV_per_Ang": 0.18,
                        "dh_relative_frobenius_ref": 0.28,
                        "dh_support_f1": 0.95,
                        "dh_false_zero_rate": 0.03,
                        "dh_false_nonzero_rate": 0.01,
                    }
                ],
            ),
            self.write_derivative_fixture(
                "deeph_n50",
                source_model="deeph",
                dataset_root=dataset_50,
                dataset_id="n50",
                rows=[
                    {
                        "sample": "s0",
                        "source_model": "deeph",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.07,
                        "dh_rmse_union_eV_per_Ang": 0.17,
                        "dh_relative_frobenius_ref": 0.27,
                        "dh_support_f1": 0.96,
                        "dh_false_zero_rate": 0.02,
                        "dh_false_nonzero_rate": 0.01,
                    }
                ],
            ),
        ]
        output_dir = self.root / "plots_dataset_size"
        args: list[str] = []
        for root in roots:
            args.extend(["--derivative-root", str(root)])
        completed = self.run_script(*args, "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        dataset_size_plot_ids = [
            "dh_mae_vs_dataset_size",
            "dh_rmse_vs_dataset_size",
            "relative_frobenius_vs_dataset_size",
            "signal_to_noise_vs_dataset_size",
            "support_f1_vs_dataset_size",
            "support_error_rates_vs_dataset_size",
        ]
        self.assertEqual([plot["id"] for plot in payload["plots"][:6]], dataset_size_plot_ids)
        self.assertEqual(payload["primary_plot_ids"], payload["dataset_size_plot_ids"])
        self.assertEqual(payload["dataset_size_plot_ids"][:6], dataset_size_plot_ids)
        self.assertIn("dh_mae_by_model", payload["diagnostic_plot_ids"])
        plots = {plot["id"]: plot for plot in payload["plots"]}
        self.assertIn("dh_mae_vs_dataset_size", plots)
        self.assertIn("dh_mae_by_model", plots)
        self.assertEqual(plots["dh_mae_vs_dataset_size"]["x_key"], "x_dataset_size")
        self.assertEqual(plots["dh_mae_vs_dataset_size"]["series_key"], "model_label")
        self.assertEqual(plots["dh_mae_vs_dataset_size"]["data_grain"], "model_dataset_size_mean")
        self.assertEqual(
            plots["dh_mae_vs_dataset_size"]["aggregation"],
            "mean over rows grouped by model and dataset size",
        )
        self.assertIn("N_train when available, otherwise N_total", plots["dh_mae_vs_dataset_size"]["x_semantics"])
        self.assertEqual(plots["dh_mae_vs_dataset_size"]["units"]["x_dataset_size"], "snapshots")
        self.assertIn("frozen_split_manifest", plots["dh_mae_vs_dataset_size"]["dataset_size_sources"])
        rows = plots["dh_mae_vs_dataset_size"]["rows"]
        self.assertEqual({row["x_dataset_size"] for row in rows}, {12, 50})
        self.assertEqual({row["model_label"] for row in rows}, {"Graph2Mat", "DeepH"})
        graph2mat_12 = next(row for row in rows if row["model"] == "graph2mat" and row["x_dataset_size"] == 12)
        self.assertAlmostEqual(graph2mat_12["dh_mae_union_eV_per_Ang"], 0.3)
        self.assertEqual(graph2mat_12["n_rows"], 2)
        self.assertEqual(graph2mat_12["n_stencils"], 2)
        self.assertEqual(graph2mat_12["dataset_ids"], ["n12"])
        self.assertEqual(graph2mat_12["x_dataset_size_kind"], "N_train")
        self.assertEqual(graph2mat_12["dataset_size_source"], "frozen_split_manifest")

    def test_delta_conditioned_dataset_size_plots_aggregate_by_model_size_and_delta(self) -> None:
        dataset_12 = self.write_dataset_root("delta_dataset_12", train=12)
        dataset_50 = self.write_dataset_root("delta_dataset_50", train=50)
        roots = [
            self.write_derivative_fixture(
                "delta_graph2mat_n12",
                source_model="graph2mat",
                dataset_root=dataset_12,
                dataset_id="n12",
                rows=[
                    {
                        "sample": "s0",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.005,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.2,
                        "dh_rmse_union_eV_per_Ang": 0.3,
                        "dh_relative_frobenius_ref": 0.4,
                    },
                    {
                        "sample": "s1",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 1,
                        "axis": "y",
                        "delta_ang": 0.005,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.4,
                        "dh_rmse_union_eV_per_Ang": 0.5,
                        "dh_relative_frobenius_ref": 0.6,
                    },
                    {
                        "sample": "s2",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 2,
                        "axis": "z",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.6,
                        "dh_rmse_union_eV_per_Ang": 0.7,
                        "dh_relative_frobenius_ref": 0.8,
                    },
                ],
            ),
            self.write_derivative_fixture(
                "delta_graph2mat_n50",
                source_model="graph2mat",
                dataset_root=dataset_50,
                dataset_id="n50",
                rows=[
                    {
                        "sample": "s0",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.005,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.1,
                        "dh_rmse_union_eV_per_Ang": 0.2,
                        "dh_relative_frobenius_ref": 0.3,
                    },
                    {
                        "sample": "s2",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 2,
                        "axis": "z",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.3,
                        "dh_rmse_union_eV_per_Ang": 0.4,
                        "dh_relative_frobenius_ref": 0.5,
                    },
                ],
            ),
        ]
        output_dir = self.root / "plots_delta_dataset_size"
        args: list[str] = []
        for root in roots:
            args.extend(["--derivative-root", str(root)])

        completed = self.run_script(*args, "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        plots = {plot["id"]: plot for plot in payload["plots"]}
        delta_plot_ids = [
            "dh_mae_vs_dataset_size_by_delta",
            "dh_rmse_vs_dataset_size_by_delta",
            "relative_frobenius_vs_dataset_size_by_delta",
        ]
        for plot_id in delta_plot_ids:
            self.assertIn(plot_id, plots)
            self.assertEqual(plots[plot_id]["kind"], "scatter")
            self.assertTrue(plots[plot_id]["dataset_size_plot"])
            self.assertEqual(plots[plot_id]["x_key"], "x_dataset_size")
            self.assertEqual(plots[plot_id]["series_key"], "series_label")
            self.assertIn(plot_id, payload["dataset_size_plot_ids"])
        self.assertIn("error_vs_delta", plots)

        rows = plots["dh_mae_vs_dataset_size_by_delta"]["rows"]
        row = next(row for row in rows if row["x_dataset_size"] == 12 and row["delta_ang"] == 0.005)
        self.assertEqual(row["model"], "graph2mat")
        self.assertEqual(row["model_label"], "Graph2Mat")
        self.assertEqual(row["finite_difference_method"], "central")
        self.assertEqual(row["x_dataset_size_kind"], "N_train")
        self.assertEqual(row["dataset_size_source"], "frozen_split_manifest")
        self.assertEqual(row["n_rows"], 2)
        self.assertEqual(row["n_stencils"], 2)
        self.assertEqual(row["series_label"], "Graph2Mat Δ=0.005 Å")
        self.assertAlmostEqual(row["dh_mae_union_eV_per_Ang"], 0.3)
        self.assertAlmostEqual(row["dh_rmse_union_eV_per_Ang"], 0.4)
        self.assertAlmostEqual(row["dh_relative_frobenius_ref"], 0.5)

    def test_axis_and_small_atom_dataset_size_plots_aggregate_from_metric_rows(self) -> None:
        dataset_12 = self.write_dataset_root("axis_atom_dataset_12", train=12)
        dataset_50 = self.write_dataset_root("axis_atom_dataset_50", train=50)
        roots = [
            self.write_derivative_fixture(
                "axis_atom_graph2mat_n12",
                source_model="graph2mat",
                dataset_root=dataset_12,
                dataset_id="n12",
                rows=[
                    {
                        "sample": "s0",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.2,
                        "dh_rmse_union_eV_per_Ang": 0.3,
                        "dh_relative_frobenius_ref": 0.4,
                        "dh_relative_frobenius_union_robust": 0.5,
                    },
                    {
                        "sample": "s1",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 1,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.4,
                        "dh_rmse_union_eV_per_Ang": 0.5,
                        "dh_relative_frobenius_ref": 0.6,
                        "dh_relative_frobenius_union_robust": 0.7,
                    },
                    {
                        "sample": "s2",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "y",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.8,
                        "dh_rmse_union_eV_per_Ang": 0.9,
                        "dh_relative_frobenius_ref": 1.0,
                        "dh_relative_frobenius_union_robust": 1.1,
                    },
                ],
            ),
            self.write_derivative_fixture(
                "axis_atom_graph2mat_n50",
                source_model="graph2mat",
                dataset_root=dataset_50,
                dataset_id="n50",
                rows=[
                    {
                        "sample": "s0",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.1,
                        "dh_rmse_union_eV_per_Ang": 0.2,
                        "dh_relative_frobenius_ref": 0.3,
                        "dh_relative_frobenius_union_robust": 0.4,
                    },
                    {
                        "sample": "s1",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 1,
                        "axis": "y",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.3,
                        "dh_rmse_union_eV_per_Ang": 0.4,
                        "dh_relative_frobenius_ref": 0.5,
                        "dh_relative_frobenius_union_robust": 0.6,
                    },
                ],
            ),
        ]
        output_dir = self.root / "plots_axis_atom_dataset_size"
        args: list[str] = []
        for root in roots:
            args.extend(["--derivative-root", str(root)])

        completed = self.run_script(*args, "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        plots = {plot["id"]: plot for plot in payload["plots"]}
        for plot_id in (
            "dh_mae_vs_dataset_size_by_axis",
            "robust_frobenius_vs_dataset_size_by_axis",
            "dh_mae_vs_dataset_size_by_displaced_atom",
        ):
            self.assertIn(plot_id, plots)
            self.assertEqual(plots[plot_id]["kind"], "scatter")
            self.assertTrue(plots[plot_id]["dataset_size_plot"])
            self.assertEqual(plots[plot_id]["x_key"], "x_dataset_size")
            self.assertEqual(plots[plot_id]["series_key"], "series_label")
            self.assertIn(plot_id, payload["dataset_size_plot_ids"])
        axis_row = next(row for row in plots["dh_mae_vs_dataset_size_by_axis"]["rows"] if row["axis"] == "x" and row["x_dataset_size"] == 12)
        self.assertEqual(axis_row["series_label"], "Graph2Mat axis=x")
        self.assertEqual(axis_row["n_rows"], 2)
        self.assertEqual(axis_row["n_stencils"], 2)
        self.assertAlmostEqual(axis_row["dh_mae_union_eV_per_Ang"], 0.3)
        robust_axis_row = next(row for row in plots["robust_frobenius_vs_dataset_size_by_axis"]["rows"] if row["axis"] == "x" and row["x_dataset_size"] == 12)
        self.assertAlmostEqual(robust_axis_row["dh_relative_frobenius_union_robust"], 0.6)
        atom_row = next(
            row
            for row in plots["dh_mae_vs_dataset_size_by_displaced_atom"]["rows"]
            if row["atom_index_zero_based"] == "0" and row["x_dataset_size"] == 12
        )
        self.assertEqual(atom_row["series_label"], "Graph2Mat atom=0")
        self.assertEqual(atom_row["n_rows"], 2)
        self.assertAlmostEqual(atom_row["dh_mae_union_eV_per_Ang"], 0.5)

    def test_dataset_size_atom_plot_skips_with_warning_when_too_many_atoms(self) -> None:
        dataset_12 = self.write_dataset_root("many_atom_dataset_12", train=12)
        dataset_50 = self.write_dataset_root("many_atom_dataset_50", train=50)
        roots = []
        for size, dataset_root in ((12, dataset_12), (50, dataset_50)):
            roots.append(
                self.write_derivative_fixture(
                    f"many_atom_graph2mat_n{size}",
                    source_model="graph2mat",
                    dataset_root=dataset_root,
                    dataset_id=f"n{size}",
                    rows=[
                        {
                            "sample": f"s{atom}",
                            "source_model": "graph2mat",
                            "atom_index_zero_based": atom,
                            "axis": "x",
                            "delta_ang": 0.01,
                            "finite_difference_method": "central",
                            "dh_mae_union_eV_per_Ang": atom / 100 + size / 1000,
                            "dh_rmse_union_eV_per_Ang": 0.2,
                            "dh_relative_frobenius_ref": 0.3,
                        }
                        for atom in range(13)
                    ],
                )
            )
        output_dir = self.root / "plots_many_atom_dataset_size"
        args: list[str] = []
        for root in roots:
            args.extend(["--derivative-root", str(root)])

        completed = self.run_script(*args, "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        plot_ids = {plot["id"] for plot in payload["plots"]}
        self.assertIn("dh_mae_vs_dataset_size_by_axis", plot_ids)
        self.assertNotIn("dh_mae_vs_dataset_size_by_displaced_atom", plot_ids)
        warnings = {warning["code"]: warning for warning in payload["scientific_warnings"]}
        self.assertIn("dataset_size_atom_plot_too_many_series", warnings)
        self.assertEqual(warnings["dataset_size_atom_plot_too_many_series"]["details"]["unique_displaced_atoms"], 13)

    def test_hermiticity_and_onsite_offsite_dataset_size_plots_use_metadata(self) -> None:
        dataset_12 = self.write_dataset_root("herm_onsite_dataset_12", train=12)
        dataset_50 = self.write_dataset_root("herm_onsite_dataset_50", train=50)
        roots = [
            self.write_derivative_fixture(
                "herm_onsite_graph2mat_n12",
                source_model="graph2mat",
                dataset_root=dataset_12,
                dataset_id="n12",
                rows=[
                    {
                        "sample": "s0",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.2,
                        "dh_rmse_union_eV_per_Ang": 0.3,
                        "dh_relative_frobenius_ref": 0.4,
                    }
                ],
                hermiticity_rows=[
                    {
                        "sample": "s0",
                        "finite_difference_method": "central",
                        "dH_ref_hermiticity_defect": 0.0,
                        "dH_pred_hermiticity_defect": 0.02,
                        "dH_hermiticity_error_delta": 0.03,
                    },
                    {
                        "sample": "s1",
                        "finite_difference_method": "central",
                        "dH_ref_hermiticity_defect": 0.0,
                        "dH_pred_hermiticity_defect": 0.04,
                        "dH_hermiticity_error_delta": 0.05,
                    },
                ],
                onsite_offsite_rows=[
                    {
                        "sample": "s0",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_onsite_relative_frobenius_robust": 0.2,
                        "dh_offsite_relative_frobenius_robust": 0.4,
                        "dh_onsite_mae_eV_per_Ang": 0.02,
                        "dh_offsite_mae_eV_per_Ang": 0.04,
                    },
                    {
                        "sample": "s1",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 1,
                        "axis": "y",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_onsite_relative_frobenius_robust": 0.4,
                        "dh_offsite_relative_frobenius_robust": 0.6,
                        "dh_onsite_mae_eV_per_Ang": 0.04,
                        "dh_offsite_mae_eV_per_Ang": 0.06,
                    },
                ],
            ),
            self.write_derivative_fixture(
                "herm_onsite_graph2mat_n50",
                source_model="graph2mat",
                dataset_root=dataset_50,
                dataset_id="n50",
                rows=[
                    {
                        "sample": "s0",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.1,
                        "dh_rmse_union_eV_per_Ang": 0.2,
                        "dh_relative_frobenius_ref": 0.3,
                    }
                ],
                hermiticity_rows=[
                    {
                        "sample": "s0",
                        "finite_difference_method": "central",
                        "dH_ref_hermiticity_defect": 0.0,
                        "dH_pred_hermiticity_defect": 0.01,
                        "dH_hermiticity_error_delta": 0.02,
                    }
                ],
                onsite_offsite_rows=[
                    {
                        "sample": "s0",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_onsite_relative_frobenius_robust": 0.1,
                        "dh_offsite_relative_frobenius_robust": 0.3,
                        "dh_onsite_mae_eV_per_Ang": 0.01,
                        "dh_offsite_mae_eV_per_Ang": 0.03,
                    }
                ],
            ),
        ]
        output_dir = self.root / "plots_herm_onsite_dataset_size"
        args: list[str] = []
        for root in roots:
            args.extend(["--derivative-root", str(root)])

        completed = self.run_script(*args, "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        plots = {plot["id"]: plot for plot in payload["plots"]}
        self.assertIn("hermiticity_defect_by_model", plots)
        self.assertIn("onsite_offsite_derivative_error", plots)
        for plot_id in ("derivative_hermiticity_vs_dataset_size", "onsite_offsite_derivative_error_vs_dataset_size"):
            self.assertIn(plot_id, plots)
            self.assertEqual(plots[plot_id]["kind"], "scatter")
            self.assertTrue(plots[plot_id]["dataset_size_plot"])
            self.assertEqual(plots[plot_id]["x_key"], "x_dataset_size")
            self.assertEqual(plots[plot_id]["series_key"], "model_label")
            self.assertIn(plot_id, payload["dataset_size_plot_ids"])
        herm_row = next(row for row in plots["derivative_hermiticity_vs_dataset_size"]["rows"] if row["x_dataset_size"] == 12)
        self.assertEqual(herm_row["dataset_ids"], ["n12"])
        self.assertEqual(herm_row["x_dataset_size_kind"], "N_train")
        self.assertAlmostEqual(herm_row["dH_pred_hermiticity_defect"], 0.03)
        self.assertAlmostEqual(herm_row["dH_hermiticity_error_delta"], 0.04)
        onsite_row = next(row for row in plots["onsite_offsite_derivative_error_vs_dataset_size"]["rows"] if row["x_dataset_size"] == 12)
        self.assertEqual(onsite_row["dataset_ids"], ["n12"])
        self.assertAlmostEqual(onsite_row["dh_onsite_relative_frobenius_robust"], 0.3)
        self.assertAlmostEqual(onsite_row["dh_offsite_relative_frobenius_robust"], 0.5)
        self.assertAlmostEqual(onsite_row["dh_onsite_mae_eV_per_Ang"], 0.03)
        self.assertAlmostEqual(onsite_row["dh_offsite_mae_eV_per_Ang"], 0.05)

    def test_extended_dataset_size_plots_aggregate_optional_derivative_metrics(self) -> None:
        dataset_12 = self.write_dataset_root("extended_dataset_12", train=12)
        dataset_50 = self.write_dataset_root("extended_dataset_50", train=50)
        roots = [
            self.write_derivative_fixture(
                "extended_graph2mat_n12",
                source_model="graph2mat",
                dataset_root=dataset_12,
                dataset_id="n12",
                rows=[
                    {
                        "sample": "s0",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.2,
                        "dh_rmse_union_eV_per_Ang": 0.3,
                        "dh_relative_frobenius_ref": 0.4,
                        "dh_relative_frobenius_union_robust": 0.2,
                        "dh_relative_l1_union_robust": 0.4,
                        "dh_pearson_union": 0.8,
                        "dh_spearman_union": 0.7,
                        "dh_residual_mean_union_eV_per_Ang": -0.02,
                        "dh_residual_std_union_eV_per_Ang": 0.10,
                        "dh_residual_median_union_eV_per_Ang": -0.01,
                        "dh_residual_bias_over_mae_union": 0.20,
                        "dh_residual_abs_p90_union_eV_per_Ang": 0.30,
                        "dh_residual_abs_p95_union_eV_per_Ang": 0.40,
                        "dh_residual_abs_p99_union_eV_per_Ang": 0.50,
                    },
                    {
                        "sample": "s1",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 1,
                        "axis": "y",
                        "delta_ang": 0.02,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.4,
                        "dh_rmse_union_eV_per_Ang": 0.5,
                        "dh_relative_frobenius_ref": 0.6,
                        "dh_relative_frobenius_union_robust": 0.4,
                        "dh_relative_l1_union_robust": 0.6,
                        "dh_pearson_union": 0.6,
                        "dh_spearman_union": 0.5,
                        "dh_residual_mean_union_eV_per_Ang": 0.02,
                        "dh_residual_std_union_eV_per_Ang": 0.20,
                        "dh_residual_median_union_eV_per_Ang": 0.03,
                        "dh_residual_bias_over_mae_union": 0.40,
                        "dh_residual_abs_p90_union_eV_per_Ang": 0.50,
                        "dh_residual_abs_p95_union_eV_per_Ang": 0.60,
                        "dh_residual_abs_p99_union_eV_per_Ang": 0.70,
                    },
                ],
            ),
            self.write_derivative_fixture(
                "extended_graph2mat_n50",
                source_model="graph2mat",
                dataset_root=dataset_50,
                dataset_id="n50",
                rows=[
                    {
                        "sample": "s0",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": 0.1,
                        "dh_rmse_union_eV_per_Ang": 0.2,
                        "dh_relative_frobenius_ref": 0.3,
                        "dh_relative_frobenius_union_robust": 0.1,
                        "dh_relative_l1_union_robust": 0.3,
                        "dh_pearson_union": 0.9,
                        "dh_spearman_union": 0.85,
                        "dh_residual_mean_union_eV_per_Ang": -0.01,
                        "dh_residual_std_union_eV_per_Ang": 0.08,
                        "dh_residual_median_union_eV_per_Ang": 0.0,
                        "dh_residual_bias_over_mae_union": 0.1,
                        "dh_residual_abs_p90_union_eV_per_Ang": 0.20,
                        "dh_residual_abs_p95_union_eV_per_Ang": 0.25,
                        "dh_residual_abs_p99_union_eV_per_Ang": 0.30,
                    }
                ],
            ),
        ]
        output_dir = self.root / "plots_extended_dataset_size"
        args: list[str] = []
        for root in roots:
            args.extend(["--derivative-root", str(root)])

        completed = self.run_script(*args, "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        new_plot_ids = [
            "robust_relative_frobenius_vs_dataset_size",
            "robust_relative_l1_vs_dataset_size",
            "derivative_correlation_vs_dataset_size",
            "derivative_residual_summary_vs_dataset_size",
            "derivative_residual_tail_vs_dataset_size",
        ]
        plots = {plot["id"]: plot for plot in payload["plots"]}
        for plot_id in new_plot_ids:
            self.assertIn(plot_id, plots)
            self.assertEqual(plots[plot_id]["kind"], "scatter")
            self.assertTrue(plots[plot_id]["dataset_size_plot"])
            self.assertEqual(plots[plot_id]["x_key"], "x_dataset_size")
            self.assertEqual(plots[plot_id]["series_key"], "model_label")
        for plot_id in new_plot_ids:
            self.assertIn(plot_id, payload["dataset_size_plot_ids"])
        self.assertEqual(payload["primary_plot_ids"], payload["dataset_size_plot_ids"])

        robust_row = next(row for row in plots["robust_relative_frobenius_vs_dataset_size"]["rows"] if row["x_dataset_size"] == 12)
        self.assertAlmostEqual(robust_row["dh_relative_frobenius_union_robust"], 0.3)
        self.assertAlmostEqual(robust_row["dh_relative_l1_union_robust"], 0.5)
        self.assertAlmostEqual(robust_row["dh_pearson_union"], 0.7)
        self.assertAlmostEqual(robust_row["dh_spearman_union"], 0.6)
        self.assertAlmostEqual(robust_row["dh_residual_mean_union_eV_per_Ang"], 0.0)
        self.assertAlmostEqual(robust_row["dh_residual_std_union_eV_per_Ang"], 0.15)
        self.assertAlmostEqual(robust_row["dh_residual_median_union_eV_per_Ang"], 0.01)
        self.assertAlmostEqual(robust_row["dh_residual_bias_over_mae_union"], 0.3)
        self.assertAlmostEqual(robust_row["dh_residual_abs_p90_union_eV_per_Ang"], 0.4)
        self.assertAlmostEqual(robust_row["dh_residual_abs_p95_union_eV_per_Ang"], 0.5)
        self.assertAlmostEqual(robust_row["dh_residual_abs_p99_union_eV_per_Ang"], 0.6)

    def test_dataset_size_plots_infer_iid_workflow_size_from_real_sweep_shape(self) -> None:
        roots: list[Path] = []
        for size, mae in ((20, 0.2), (40, 0.1)):
            derivative_root = (
                self.root
                / "derivative_workflows"
                / f"graphene_w90_scale_iid{size}"
                / "derivative_metrics"
                / "graph2mat"
            )
            write_json(
                derivative_root / "manifest.json",
                {
                    "schema_version": "hamiltonian_derivative_metrics_v1",
                    "scientific_status": "diagnostic_only",
                    "force_constants_used": False,
                    "stencils_total": 1,
                    "stencils_ok": 1,
                    "stencils_failed": 0,
                    "fatal_errors": [],
                    "dataset_root": "",
                    "dataset_id": "",
                },
            )
            write_csv(
                derivative_root / "derivative_matrix_metrics.csv",
                [
                    {
                        "source_model": "graph2mat",
                        "sample": "s0",
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": mae,
                        "dh_rmse_union_eV_per_Ang": mae + 0.1,
                        "dh_relative_frobenius_ref": mae + 0.2,
                    }
                ],
            )
            write_csv(derivative_root / "derivative_hermiticity.csv", [])
            write_csv(derivative_root / "stencil_status.csv", [])
            roots.append(derivative_root)

        output_dir = self.root / "plots_iid_size"
        args: list[str] = []
        for root in roots:
            args.extend(["--derivative-root", str(root)])
        completed = self.run_script(*args, "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        plots = {plot["id"]: plot for plot in payload["plots"]}
        self.assertIn("dh_mae_vs_dataset_size", plots)
        rows = plots["dh_mae_vs_dataset_size"]["rows"]
        self.assertEqual({row["x_dataset_size"] for row in rows}, {20, 40})
        self.assertEqual(payload["summary"]["dataset_size_rows"], 2)
        self.assertEqual({row["x_dataset_size_kind"] for row in rows}, {"N_total"})

    def test_single_dataset_size_warns_and_keeps_existing_plots(self) -> None:
        dataset_12 = self.write_dataset_root("dataset_12", train=12)
        graph2mat_root = self.write_derivative_fixture(
            "single_size_graph2mat",
            source_model="graph2mat",
            dataset_root=dataset_12,
            rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                    "dh_support_f1": 0.8,
                    "dh_false_zero_rate": 0.1,
                    "dh_false_nonzero_rate": 0.05,
                }
            ],
        )
        deeph_root = self.write_derivative_fixture(
            "single_size_deeph",
            source_model="deeph",
            dataset_root=dataset_12,
            rows=[
                {
                    "sample": "s0",
                    "source_model": "deeph",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.1,
                    "dh_rmse_union_eV_per_Ang": 0.2,
                    "dh_relative_frobenius_ref": 0.3,
                    "dh_support_f1": 0.9,
                    "dh_false_zero_rate": 0.05,
                    "dh_false_nonzero_rate": 0.02,
                }
            ],
        )
        output_dir = self.root / "plots_single_size"

        completed = self.run_script(
            "--derivative-root",
            str(graph2mat_root),
            "--derivative-root",
            str(deeph_root),
            "--output-dir",
            str(output_dir),
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        plot_ids = {plot["id"] for plot in payload["plots"]}
        self.assertIn("dh_mae_by_model", plot_ids)
        self.assertIn("graph2mat_vs_deeph_paired_comparison", plot_ids)
        self.assertNotIn("dh_mae_vs_dataset_size", plot_ids)
        self.assertEqual(payload["dataset_size_plot_ids"], [])
        self.assertEqual(payload["primary_plot_ids"], payload["diagnostic_plot_ids"])
        codes = {warning["code"] for warning in payload["scientific_warnings"]}
        self.assertIn("dataset_size_plots_unavailable_single_dataset_size", codes)

    def test_missing_dataset_size_metadata_does_not_fabricate_plot(self) -> None:
        derivative_root = self.write_derivative_fixture(
            "missing_size",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                    "dh_support_f1": 0.8,
                    "dh_false_zero_rate": 0.1,
                    "dh_false_nonzero_rate": 0.05,
                }
            ],
        )
        output_dir = self.root / "plots_missing_size"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        self.assertNotIn("dh_mae_vs_dataset_size", {plot["id"] for plot in payload["plots"]})
        codes = {warning["code"] for warning in payload["scientific_warnings"]}
        self.assertIn("dataset_size_metadata_missing", codes)


if __name__ == "__main__":
    unittest.main()
