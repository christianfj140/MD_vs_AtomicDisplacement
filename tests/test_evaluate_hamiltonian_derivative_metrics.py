from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "evaluate_hamiltonian_derivative_metrics.py"
SPEC = importlib.util.spec_from_file_location("evaluate_hamiltonian_derivative_metrics", SCRIPT)
metrics_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = metrics_module
SPEC.loader.exec_module(metrics_module)


class EvaluateHamiltonianDerivativeMetricsCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.result_dir = self.root / "result"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_sparse_payload(self, path: Path, matrix: sparse.spmatrix) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            sparse.save_npz(handle, matrix.tocsr())

    def write_sample(
        self,
        sample_id: str,
        *,
        sign: int,
        reference: sparse.spmatrix | None = None,
        prediction: sparse.spmatrix | None = None,
        atom_index_zero_based: int = 0,
        axis: str = "x",
        amplitude_ang: float = 0.5,
        include_metadata: bool = True,
        include_matrix_shape: bool = True,
        claim_status: str = "diagnostic_only",
        base_sample_id: str | None = None,
    ) -> None:
        structure_dir = self.result_dir / "structures" / sample_id
        structure_dir.mkdir(parents=True, exist_ok=True)
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        coordinates = [0.0, 0.0, 0.0]
        if sign:
            coordinates[axis_index] = sign * amplitude_ang
        (structure_dir / "RUN.fdf").write_text(
            "\n".join(
                [
                    "SystemName fixture",
                    "SystemLabel fixture",
                    "NumberOfSpecies 1",
                    "NumberOfAtoms 2",
                    "%block ChemicalSpeciesLabel",
                    " 1 6 C",
                    "%endblock ChemicalSpeciesLabel",
                    "LatticeConstant 1.0 Ang",
                    "%block LatticeVectors",
                    " 8.0 0.0 0.0",
                    " 0.0 8.0 0.0",
                    " 0.0 0.0 8.0",
                    "%endblock LatticeVectors",
                    "AtomicCoordinatesFormat Ang",
                    "%block AtomicCoordinatesAndAtomicSpecies",
                    f" {coordinates[0]:.12g} {coordinates[1]:.12g} {coordinates[2]:.12g} 1",
                    " 1.0 0.0 0.0 1",
                    "%endblock AtomicCoordinatesAndAtomicSpecies",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        if include_metadata:
            metadata = {
                "sample_id": sample_id,
                "material_label": "graphene",
                "is_reference": sign == 0,
                "atom_index": atom_index_zero_based + 1 if sign else None,
                "atom_index_zero_based": atom_index_zero_based if sign else None,
                "axis": axis if sign else None,
                "axis_index": axis_index if sign else None,
                "sign": sign,
                "sign_label": "+" if sign > 0 else "-" if sign < 0 else None,
                "amplitude_ang": amplitude_ang if sign else 0.0,
                "displacement_ang": coordinates,
                "split_group_id": "generic_cartesian_displacement:graphene:reference"
                if sign == 0
                else f"generic_cartesian_displacement:graphene:atom_0001:{axis}",
                "claim_status": claim_status,
                "material_compatibility_hash": "material-hash",
                "orbital_ordering_hash": "orbital-hash",
                "neighbor_list_hash": "neighbor-hash",
                "sparsity_pattern_hash": "sparsity-hash",
                "basis_hash": "basis-hash",
                "pseudopotential_hash": "pseudo-hash",
            }
            if include_matrix_shape:
                metadata["matrix_shape"] = [2, 2]
            if base_sample_id is not None:
                metadata["base_sample_id"] = base_sample_id
            (structure_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        if reference is not None:
            self.write_sparse_payload(
                self.result_dir / "siesta_hamiltonians" / sample_id / "siesta.TSHS",
                reference,
            )
        if prediction is not None:
            self.write_sparse_payload(
                self.result_dir / "predicted_hamiltonians" / sample_id / "ML_prediction.HSX",
                prediction,
            )

    def run_cli(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.result_dir), *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check and completed.returncode != 0:
            self.fail(f"CLI failed with {completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
        return completed

    def write_central_fixture(self, *, claim_status: str = "diagnostic_only") -> None:
        zero = sparse.csr_matrix((2, 2))
        ref_plus = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        pred_plus = sparse.csr_matrix([[3.0, 0.0], [0.0, 3.0]])
        self.write_sample("base", sign=0, reference=zero, prediction=zero, claim_status=claim_status)
        self.write_sample("plus", sign=1, reference=ref_plus, prediction=pred_plus, claim_status=claim_status)
        self.write_sample("minus", sign=-1, reference=zero, prediction=zero, claim_status=claim_status)

    def write_central_fixture_without_matrix_shape(
        self,
        *,
        minus_prediction: sparse.spmatrix | None = None,
        include_minus_prediction: bool = True,
    ) -> None:
        zero = sparse.csr_matrix((2, 2))
        ref_plus = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        pred_plus = sparse.csr_matrix([[3.0, 0.0], [0.0, 3.0]])
        minus_prediction = zero if minus_prediction is None and include_minus_prediction else minus_prediction
        self.write_sample("base", sign=0, reference=zero, prediction=zero, include_matrix_shape=False)
        self.write_sample("plus", sign=1, reference=ref_plus, prediction=pred_plus, include_matrix_shape=False)
        self.write_sample("minus", sign=-1, reference=zero, prediction=minus_prediction, include_matrix_shape=False)

    def test_cli_happy_path_with_synthetic_sparse_matrices(self) -> None:
        self.write_central_fixture()

        self.run_cli("--method", "central", "--overwrite")

        metrics_root = self.result_dir / "derivative_metrics"
        manifest = json.loads((metrics_root / "manifest.json").read_text(encoding="utf-8"))
        with (metrics_root / "derivative_matrix_metrics.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with (metrics_root / "derivative_ref_abs_quantile_metrics.csv").open(encoding="utf-8") as handle:
            quantile_rows = list(csv.DictReader(handle))

        self.assertEqual(manifest["schema_version"], "hamiltonian_derivative_metrics_v1")
        self.assertFalse(manifest["force_constants_used"])
        self.assertEqual(manifest["reference_definition"], "siesta_hamiltonian_finite_difference")
        self.assertEqual(manifest["stencils_ok"], 1)
        self.assertEqual(
            manifest["outputs"]["derivative_ref_abs_quantile_metrics"],
            str(metrics_root / "derivative_ref_abs_quantile_metrics.csv"),
        )
        self.assertEqual(
            manifest["outputs"]["derivative_group_metrics"],
            str(metrics_root / "derivative_group_metrics.json"),
        )
        self.assertEqual(
            manifest["outputs"]["derivative_onsite_offsite_metrics"],
            str(metrics_root / "derivative_onsite_offsite_metrics.json"),
        )
        onsite_offsite = json.loads((metrics_root / "derivative_onsite_offsite_metrics.json").read_text(encoding="utf-8"))
        self.assertFalse(onsite_offsite["available"])
        self.assertEqual(onsite_offsite["reason"], "orbital_to_atom_mapping_unavailable")
        self.assertFalse((metrics_root / "derivative_onsite_offsite_metrics.csv").exists())
        self.assertTrue(
            any(
                warning["kind"] == "derivative_onsite_offsite_metrics_unavailable"
                and warning["reason"] == "orbital_to_atom_mapping_unavailable"
                for warning in manifest["warnings"]
            )
        )
        group_metrics = json.loads((metrics_root / "derivative_group_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(group_metrics["schema"], "hamiltonian_derivative_group_metrics_v1")
        self.assertEqual(len(group_metrics["by_axis"]), 1)
        self.assertEqual(group_metrics["by_axis"][0]["axis"], "x")
        self.assertEqual(group_metrics["by_axis"][0]["delta_ang"], 0.5)
        self.assertEqual(group_metrics["by_axis"][0]["support_threshold"], 1e-12)
        self.assertEqual(rows[0]["derivative_units"], "eV/Ang")
        self.assertAlmostEqual(float(rows[0]["dh_mae_union_eV_per_Ang"]), 1.0)
        self.assertEqual(len(quantile_rows), 2)
        self.assertEqual(sum(int(row["n_entries"]) for row in quantile_rows), 2)
        self.assertTrue(all(row["quantile_domain"] == "union_support" for row in quantile_rows))
        self.assertAlmostEqual(float(quantile_rows[0]["abs_ref_mean_eV_per_Ang"]), 2.0)
        self.assertAlmostEqual(float(quantile_rows[0]["dh_error_mae_eV_per_Ang"]), 1.0)
        self.assertAlmostEqual(float(quantile_rows[0]["dh_error_rmse_eV_per_Ang"]), 1.0)
        self.assertAlmostEqual(float(quantile_rows[0]["dh_error_relative_l1_robust"]), 0.5)
        self.assertTrue((metrics_root / "derivative_delta_stability.csv").exists())
        self.assertTrue((metrics_root / "derivative_delta_stability.json").exists())
        self.assertEqual(manifest["delta_stability"]["status"], "single_delta_only")
        self.assertEqual(manifest["delta_stability"]["pairwise_metric_rows"], [])
        self.assertIn("delta_sensitivity_study_available", manifest)
        self.assertFalse(manifest["delta_sensitivity_study_available"])
        self.assertIsNone(manifest["delta_stability_converged"])
        self.assertEqual(manifest["delta_stability_convergence_status"], "not_evaluated_without_thresholds")
        self.assertEqual(manifest["reference_noise"]["status"], "reference_noise_unavailable")

    def test_cli_empty_quantile_support_warns_without_rows(self) -> None:
        zero = sparse.csr_matrix((2, 2))
        self.write_sample("base", sign=0, reference=zero, prediction=zero)
        self.write_sample("plus", sign=1, reference=zero, prediction=zero)
        self.write_sample("minus", sign=-1, reference=zero, prediction=zero)

        self.run_cli("--method", "central", "--overwrite")

        metrics_root = self.result_dir / "derivative_metrics"
        manifest = json.loads((metrics_root / "manifest.json").read_text(encoding="utf-8"))
        with (metrics_root / "derivative_ref_abs_quantile_metrics.csv").open(encoding="utf-8") as handle:
            quantile_rows = list(csv.DictReader(handle))

        self.assertEqual(quantile_rows, [])
        self.assertTrue(
            any(
                warning["kind"] == "derivative_ref_abs_quantile_metrics_empty_union_support"
                for warning in manifest["warnings"]
            )
        )

    def test_derivative_group_metrics_preserve_conditions_and_pool_norms(self) -> None:
        rows = [
            {
                "source_model": "graph2mat",
                "reference_source": "siesta",
                "dataset_size": 10,
                "seed": 1,
                "split": "test",
                "delta_ang": 0.01,
                "finite_difference_method": "central",
                "support_threshold": 1e-12,
                "atom_index_zero_based": 0,
                "axis": "x",
                "dh_relative_frobenius_union_robust": 0.5,
                "dh_mae_union_eV_per_Ang": 1.0,
                "dh_rmse_union_eV_per_Ang": 1.5,
                "dh_relative_l1_union_robust": 0.25,
                "dh_norm_error_union_fro": 3.0,
                "dh_norm_ref_union_fro": 4.0,
                "dh_norm_error_l1_union": 5.0,
                "dh_norm_ref_l1_union": 10.0,
            },
            {
                "source_model": "graph2mat",
                "reference_source": "siesta",
                "dataset_size": 10,
                "seed": 1,
                "split": "test",
                "delta_ang": 0.01,
                "finite_difference_method": "central",
                "support_threshold": 1e-12,
                "atom_index_zero_based": 0,
                "axis": "x",
                "dh_relative_frobenius_union_robust": 1.0,
                "dh_mae_union_eV_per_Ang": 3.0,
                "dh_rmse_union_eV_per_Ang": 2.5,
                "dh_relative_l1_union_robust": 0.75,
                "dh_norm_error_union_fro": 4.0,
                "dh_norm_ref_union_fro": 3.0,
                "dh_norm_error_l1_union": 1.0,
                "dh_norm_ref_l1_union": 2.0,
            },
            {
                "source_model": "graph2mat",
                "reference_source": "siesta",
                "dataset_size": 10,
                "seed": 2,
                "split": "validation",
                "delta_ang": 0.02,
                "finite_difference_method": "central",
                "support_threshold": 1e-10,
                "atom_index_zero_based": 0,
                "axis": "x",
                "dh_relative_frobenius_union_robust": float("nan"),
                "dh_mae_union_eV_per_Ang": float("inf"),
                "dh_rmse_union_eV_per_Ang": 9.0,
                "dh_relative_l1_union_robust": 2.0,
            },
        ]

        groups = metrics_module._derivative_group_metrics(rows)

        self.assertEqual(groups["grouping_preserves"], metrics_module.GROUPING_PRESERVES)
        self.assertEqual(len(groups["by_atom"]), 2)
        self.assertEqual(len(groups["by_axis"]), 2)
        first_axis = [row for row in groups["by_axis"] if row["seed"] == 1 and row["split"] == "test"][0]
        self.assertEqual(first_axis["delta_ang"], 0.01)
        self.assertEqual(first_axis["support_threshold"], 1e-12)
        self.assertEqual(first_axis["n_stencils"], 2)
        self.assertAlmostEqual(first_axis["dh_mae_union_eV_per_Ang_mean"], 2.0)
        self.assertAlmostEqual(first_axis["dh_mae_union_eV_per_Ang_median"], 2.0)
        self.assertAlmostEqual(first_axis["dh_relative_frobenius_union_robust_mean"], 0.75)
        self.assertAlmostEqual(first_axis["dh_relative_frobenius_union_robust_pooled"], 1.0)
        self.assertAlmostEqual(first_axis["dh_relative_l1_union_robust_pooled"], 6.0 / 12.0)
        second_axis = [row for row in groups["by_axis"] if row["seed"] == 2][0]
        self.assertEqual(second_axis["split"], "validation")
        self.assertEqual(second_axis["support_threshold"], 1e-10)
        self.assertIsNone(second_axis["dh_mae_union_eV_per_Ang_mean"])

    def test_derivative_group_metrics_empty_rows(self) -> None:
        groups = metrics_module._derivative_group_metrics([])

        self.assertEqual(groups["schema"], "hamiltonian_derivative_group_metrics_v1")
        self.assertEqual(groups["by_atom"], [])
        self.assertEqual(groups["by_axis"], [])
        self.assertEqual(groups["by_atom_axis"], [])

    def test_cli_shared_base_multi_axis_has_no_false_operand_metadata_mismatch(self) -> None:
        zero = sparse.csr_matrix((2, 2))
        ref_plus = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        pred_plus = sparse.csr_matrix([[3.0, 0.0], [0.0, 3.0]])
        self.write_sample("base", sign=0, reference=zero, prediction=zero)
        for axis in ("x", "y"):
            self.write_sample(
                f"{axis}_plus",
                sign=1,
                axis=axis,
                reference=ref_plus,
                prediction=pred_plus,
                base_sample_id="base",
            )
            self.write_sample(
                f"{axis}_minus",
                sign=-1,
                axis=axis,
                reference=zero,
                prediction=zero,
                base_sample_id="base",
            )

        completed = self.run_cli("--method", "central", "--require-central", "--overwrite", check=False)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        manifest = json.loads((self.result_dir / "derivative_metrics" / "manifest.json").read_text(encoding="utf-8"))
        issue_codes = {
            code
            for error in manifest["fatal_errors"]
            for code in error.get("issue_codes", [])
        }
        self.assertEqual(manifest["stencils_ok"], 2)
        self.assertNotIn("axis_mismatch", issue_codes)
        self.assertNotIn("atom_index_mismatch", issue_codes)

    def test_delta_stability_summary_aggregates_matching_delta_sweep_rows(self) -> None:
        rows = [
            {
                "source_model": "graph2mat",
                "reference_source": "siesta",
                "split": "test",
                "base_sample_id": "base_0",
                "atom_index_zero_based": 0,
                "axis": "x",
                "finite_difference_method": "central",
                "support_threshold": 1e-12,
                "delta_ang": 0.005,
                "dh_mae_union_eV_per_Ang": 0.4,
                "dh_rmse_union_eV_per_Ang": 0.5,
                "dh_relative_frobenius_ref": 0.1,
                "dh_relative_frobenius_union_robust": 0.2,
                "dh_relative_l1_union_robust": 0.3,
            },
            {
                "source_model": "graph2mat",
                "reference_source": "siesta",
                "split": "test",
                "base_sample_id": "base_0",
                "atom_index_zero_based": 0,
                "axis": "x",
                "finite_difference_method": "central",
                "support_threshold": 1e-12,
                "delta_ang": 0.01,
                "dh_mae_union_eV_per_Ang": 0.6,
                "dh_rmse_union_eV_per_Ang": 0.7,
                "dh_relative_frobenius_ref": 0.15,
                "dh_relative_frobenius_union_robust": 0.5,
                "dh_relative_l1_union_robust": 0.9,
            },
        ]

        summary = metrics_module._delta_stability_summary(rows)

        self.assertEqual(summary["status"], "available")
        self.assertEqual(summary["groups_total"], 1)
        self.assertEqual(summary["rows"][0]["delta_count"], 2)
        self.assertAlmostEqual(summary["rows"][0]["dh_mae_union_eV_per_Ang_range"], 0.2)
        self.assertEqual(len(summary["pairwise_metric_rows"]), 4)
        pairwise = {
            row["metric_name"]: row
            for row in summary["pairwise_metric_rows"]
        }
        self.assertAlmostEqual(pairwise["dh_relative_frobenius_union_robust"]["abs_change"], 0.3)
        self.assertAlmostEqual(pairwise["dh_relative_frobenius_union_robust"]["relative_change"], 0.3 / 0.5)
        self.assertEqual(
            pairwise["dh_mae_union_eV_per_Ang"]["stability_definition"],
            "scalar_error_metric_pairwise_delta_change_not_matrix_delta_stability",
        )
        self.assertEqual(pairwise["dh_mae_union_eV_per_Ang"]["split"], "test")
        self.assertEqual(pairwise["dh_mae_union_eV_per_Ang"]["support_threshold"], 1e-12)

    def test_delta_stability_summary_reports_single_delta_unavailable(self) -> None:
        summary = metrics_module._delta_stability_summary(
            [
                {
                    "source_model": "graph2mat",
                    "base_sample_id": "base_0",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "finite_difference_method": "central",
                    "delta_ang": 0.01,
                    "dh_mae_union_eV_per_Ang": 0.6,
                }
            ]
        )

        self.assertEqual(summary["status"], "single_delta_only")
        self.assertEqual(summary["pairwise_metric_rows"], [])
        self.assertIn("Fewer than two", summary["reason"])

    def test_delta_stability_pairwise_preserves_split_and_threshold_and_skips_nonfinite(self) -> None:
        base = {
            "source_model": "graph2mat",
            "reference_source": "siesta",
            "base_sample_id": "base_0",
            "atom_index_zero_based": 0,
            "axis": "x",
            "finite_difference_method": "central",
            "dh_relative_frobenius_ref": 0.1,
            "dh_mae_union_eV_per_Ang": 1.0,
            "dh_rmse_union_eV_per_Ang": 2.0,
        }
        rows = [
            {**base, "split": "test", "support_threshold": 1e-12, "delta_ang": 0.005},
            {
                **base,
                "split": "test",
                "support_threshold": 1e-12,
                "delta_ang": 0.01,
                "dh_relative_frobenius_ref": 0.4,
                "dh_mae_union_eV_per_Ang": float("nan"),
                "dh_rmse_union_eV_per_Ang": float("inf"),
            },
            {**base, "split": "validation", "support_threshold": 1e-12, "delta_ang": 0.02},
            {**base, "split": "test", "support_threshold": 1e-10, "delta_ang": 0.02},
        ]

        summary = metrics_module._delta_stability_summary(rows)

        pairwise = summary["pairwise_metric_rows"]
        self.assertEqual(len(pairwise), 1)
        self.assertEqual(pairwise[0]["metric_name"], "dh_relative_frobenius_ref")
        self.assertEqual(pairwise[0]["split"], "test")
        self.assertEqual(pairwise[0]["support_threshold"], 1e-12)
        json.dumps(metrics_module.json_safe(summary), allow_nan=False)

    def test_delta_stability_convergence_summary_marks_availability_without_convergence_claim(self) -> None:
        summary = metrics_module._delta_stability_summary(
            [
                {
                    "source_model": "graph2mat",
                    "base_sample_id": "base_0",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "finite_difference_method": "central",
                    "delta_ang": 0.005,
                    "dh_mae_union_eV_per_Ang": 0.4,
                    "dh_rmse_union_eV_per_Ang": 0.5,
                    "dh_relative_frobenius_ref": 0.1,
                },
                {
                    "source_model": "graph2mat",
                    "base_sample_id": "base_0",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "finite_difference_method": "central",
                    "delta_ang": 0.01,
                    "dh_mae_union_eV_per_Ang": 0.6,
                    "dh_rmse_union_eV_per_Ang": 0.7,
                    "dh_relative_frobenius_ref": 0.15,
                },
            ]
        )
        convergence = metrics_module._delta_stability_convergence_summary(summary)

        self.assertEqual(summary["status"], "available")
        self.assertTrue(convergence["delta_sensitivity_study_available"])
        self.assertTrue(convergence["delta_sensitivity_study_passed"])
        self.assertIsNone(convergence["delta_stability_converged"])
        self.assertEqual(convergence["delta_stability_convergence_status"], "not_evaluated_without_thresholds")

    def test_cli_infers_matrix_shape_from_sparse_files_when_metadata_omits_shape(self) -> None:
        self.write_central_fixture_without_matrix_shape()

        self.run_cli("--method", "central", "--overwrite")

        metrics_root = self.result_dir / "derivative_metrics"
        manifest = json.loads((metrics_root / "manifest.json").read_text(encoding="utf-8"))
        geometry = json.loads((metrics_root / "derivative_geometry_validation.json").read_text(encoding="utf-8"))
        with (metrics_root / "stencil_status.csv").open(encoding="utf-8") as handle:
            status_rows = list(csv.DictReader(handle))

        self.assertEqual(manifest["stencils_ok"], 1)
        self.assertEqual(manifest["stencils_failed"], 0)
        self.assertEqual(manifest["geometry_validation"]["ok"], 1)
        self.assertEqual(geometry["ok"], 1)
        self.assertEqual(status_rows[0]["status"], "ok")

    def test_cli_mismatched_loaded_matrix_shapes_fail_clearly(self) -> None:
        self.write_central_fixture_without_matrix_shape(
            minus_prediction=sparse.csr_matrix((3, 3)),
        )

        completed = self.run_cli("--method", "central", "--overwrite", check=False)

        self.assertEqual(completed.returncode, 2)
        manifest = json.loads((self.result_dir / "derivative_metrics" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["stencils_ok"], 0)
        self.assertEqual(manifest["stencils_failed"], 1)
        self.assertEqual(manifest["fatal_errors"][0]["kind"], "stencil_validation_failed")
        self.assertIn("matrix_shape_mismatch", manifest["fatal_errors"][0]["issue_codes"])

    def test_cli_missing_matrix_still_fails_closed_without_metadata_shape(self) -> None:
        self.write_central_fixture_without_matrix_shape(include_minus_prediction=False)

        completed = self.run_cli("--method", "central", "--overwrite", check=False)

        self.assertEqual(completed.returncode, 2)
        manifest = json.loads((self.result_dir / "derivative_metrics" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["stencils_ok"], 0)
        self.assertEqual(manifest["stencils_failed"], 1)
        self.assertTrue(manifest["fatal_errors"])
        self.assertIn("missing", json.dumps(manifest["fatal_errors"]))

    def test_cli_refuses_overwrite_without_flag(self) -> None:
        self.write_central_fixture()
        self.run_cli("--method", "central", "--overwrite")

        completed = self.run_cli("--method", "central", check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Refusing to overwrite", completed.stderr)

    def test_cli_overwrite_works(self) -> None:
        self.write_central_fixture()
        self.run_cli("--method", "central", "--overwrite")

        completed = self.run_cli("--method", "central", "--overwrite", check=False)

        self.assertEqual(completed.returncode, 0)
        manifest = json.loads((self.result_dir / "derivative_metrics" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["stencils_ok"], 1)

    def test_cli_missing_stencil_produces_manifest_with_failures(self) -> None:
        self.write_sample(
            "plus",
            sign=1,
            reference=sparse.eye(2, format="csr"),
            prediction=sparse.eye(2, format="csr"),
        )

        completed = self.run_cli("--method", "central", "--require-central", "--overwrite", check=False)

        self.assertEqual(completed.returncode, 2)
        manifest = json.loads((self.result_dir / "derivative_metrics" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["stencils_ok"], 0)
        self.assertEqual(manifest["stencils_failed"], 1)
        self.assertTrue(manifest["fatal_errors"])

    def test_cli_central_only_required_behavior(self) -> None:
        zero = sparse.csr_matrix((2, 2))
        self.write_sample("base", sign=0, reference=zero, prediction=zero)
        self.write_sample(
            "plus",
            sign=1,
            reference=sparse.eye(2, format="csr"),
            prediction=sparse.eye(2, format="csr"),
        )

        forward = self.run_cli("--method", "forward", "--overwrite", check=False)
        self.assertEqual(forward.returncode, 0)
        central = self.run_cli("--method", "central", "--require-central", "--overwrite", check=False)
        self.assertEqual(central.returncode, 2)
        manifest = json.loads((self.result_dir / "derivative_metrics" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["stencils_failed"], 1)

    def test_cli_output_csv_columns(self) -> None:
        self.write_central_fixture()

        self.run_cli("--method", "central", "--overwrite")

        metrics_root = self.result_dir / "derivative_metrics"
        with (metrics_root / "derivative_matrix_metrics.csv").open(encoding="utf-8") as handle:
            matrix_fields = next(csv.reader(handle))
        with (metrics_root / "stencil_status.csv").open(encoding="utf-8") as handle:
            status_fields = next(csv.reader(handle))
        with (metrics_root / "derivative_support_sweep.csv").open(encoding="utf-8") as handle:
            sweep_fields = next(csv.reader(handle))
        with (metrics_root / "derivative_ref_abs_quantile_metrics.csv").open(encoding="utf-8") as handle:
            quantile_fields = next(csv.reader(handle))
        with (metrics_root / "derivative_geometry_validation.csv").open(encoding="utf-8") as handle:
            geometry_fields = next(csv.reader(handle))

        self.assertIn("dh_mae_ref_eV_per_Ang", matrix_fields)
        self.assertIn("comparison_status", matrix_fields)
        self.assertIn("issue_codes", status_fields)
        self.assertIn("dh_support_f1", sweep_fields)
        self.assertIn("quantile_domain", quantile_fields)
        self.assertIn("dh_error_relative_l1_robust", quantile_fields)
        self.assertIn("status", geometry_fields)
        self.assertTrue((metrics_root / "derivative_geometry_validation.json").exists())

    def test_cli_manifest_status_rules(self) -> None:
        self.write_central_fixture(claim_status="robust")

        self.run_cli("--method", "central", "--overwrite")

        manifest = json.loads((self.result_dir / "derivative_metrics" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["scientific_status"], "presentation_ready")
        self.assertFalse(manifest["paper_level"])


if __name__ == "__main__":
    unittest.main()
