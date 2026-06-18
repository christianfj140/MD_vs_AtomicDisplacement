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
        amplitude_ang: float = 0.5,
        include_metadata: bool = True,
        include_matrix_shape: bool = True,
        claim_status: str = "diagnostic_only",
    ) -> None:
        structure_dir = self.result_dir / "structures" / sample_id
        structure_dir.mkdir(parents=True, exist_ok=True)
        displacement = sign * amplitude_ang if sign else 0.0
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
                    f" {displacement:.12g} 0.0 0.0 1",
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
                "axis": "x" if sign else None,
                "axis_index": 0 if sign else None,
                "sign": sign,
                "sign_label": "+" if sign > 0 else "-" if sign < 0 else None,
                "amplitude_ang": amplitude_ang if sign else 0.0,
                "displacement_ang": [sign * amplitude_ang, 0.0, 0.0],
                "split_group_id": "generic_cartesian_displacement:graphene:reference"
                if sign == 0
                else "generic_cartesian_displacement:graphene:atom_0001",
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

        self.assertEqual(manifest["schema_version"], "hamiltonian_derivative_metrics_v1")
        self.assertFalse(manifest["force_constants_used"])
        self.assertEqual(manifest["reference_definition"], "siesta_hamiltonian_finite_difference")
        self.assertEqual(manifest["stencils_ok"], 1)
        self.assertEqual(rows[0]["derivative_units"], "eV/Ang")
        self.assertAlmostEqual(float(rows[0]["dh_mae_union_eV_per_Ang"]), 1.0)
        self.assertTrue((metrics_root / "derivative_delta_stability.csv").exists())
        self.assertTrue((metrics_root / "derivative_delta_stability.json").exists())
        self.assertEqual(manifest["delta_stability"]["status"], "unavailable_single_delta")
        self.assertEqual(manifest["reference_noise"]["status"], "reference_noise_unavailable")

    def test_delta_stability_summary_aggregates_matching_delta_sweep_rows(self) -> None:
        rows = [
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

        summary = metrics_module._delta_stability_summary(rows)

        self.assertEqual(summary["status"], "available")
        self.assertEqual(summary["groups_total"], 1)
        self.assertEqual(summary["rows"][0]["delta_count"], 2)
        self.assertAlmostEqual(summary["rows"][0]["dh_mae_union_eV_per_Ang_range"], 0.2)

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

        self.assertEqual(summary["status"], "unavailable_single_delta")
        self.assertIn("Fewer than two", summary["reason"])

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
        with (metrics_root / "derivative_geometry_validation.csv").open(encoding="utf-8") as handle:
            geometry_fields = next(csv.reader(handle))

        self.assertIn("dh_mae_ref_eV_per_Ang", matrix_fields)
        self.assertIn("comparison_status", matrix_fields)
        self.assertIn("issue_codes", status_fields)
        self.assertIn("dh_support_f1", sweep_fields)
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
