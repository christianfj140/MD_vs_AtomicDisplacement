from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SCRIPT = SCRIPTS_DIR / "evaluate_hamiltonian_derivative_metrics.py"
SPEC = importlib.util.spec_from_file_location("evaluate_hamiltonian_derivative_metrics_direct", SCRIPT)
metrics_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = metrics_module
SPEC.loader.exec_module(metrics_module)

from hamiltonian_derivative_stencil import (  # noqa: E402
    DerivativeMetadata,
    PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH,
    HamiltonianDerivativeError,
    PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
    REFERENCE_DERIVATIVE_METHOD_SIESTA,
    direct_derivative_prediction_paths,
    direct_predicted_derivative_pair,
    find_direct_derivative_prediction,
    load_direct_sparse_derivative,
)


DELTA_ANG = 0.5


class RunnerWiringTests(unittest.TestCase):
    """The derivative.graph2mat_prediction_method option wires through the runner."""

    @classmethod
    def setUpClass(cls) -> None:
        import g2m_deeph_runner

        cls.runner = g2m_deeph_runner

    def test_config_defaults_to_finite_difference(self) -> None:
        config = self.runner._normalize_derivative_workflow_config({})
        self.assertEqual(config["graph2mat_prediction_method"], "finite_difference")
        self.assertEqual(config["deeph_prediction_method"], "finite_difference")

    def test_config_accepts_autograd_vectorized(self) -> None:
        config = self.runner._normalize_derivative_workflow_config(
            {
                "derivative": {
                    "graph2mat_prediction_method": "autograd_vectorized",
                    "deeph_prediction_method": "autograd_vectorized",
                }
            }
        )
        self.assertEqual(config["graph2mat_prediction_method"], "autograd_vectorized")
        self.assertEqual(config["deeph_prediction_method"], "autograd_vectorized")

    def test_validation_rejects_unknown_method(self) -> None:
        config = self.runner._normalize_derivative_workflow_config(
            {
                "derivative": {
                    "enabled": True,
                    "method": "central",
                    "result_dir": "/tmp/derivatives",
                    "graph2mat_prediction_method": "bogus",
                }
            }
        )
        stages = {"derivative_metrics_graph2mat": True}
        with self.assertRaises(RuntimeError):
            self.runner._validate_derivative_workflow_config(stages, config)

    def test_validation_rejects_unknown_deeph_method(self) -> None:
        config = self.runner._normalize_derivative_workflow_config(
            {
                "derivative": {
                    "enabled": True,
                    "method": "central",
                    "result_dir": "/tmp/derivatives",
                    "deeph_prediction_method": "bogus",
                }
            }
        )
        stages = {"derivative_metrics_deeph": True}
        with self.assertRaises(RuntimeError):
            self.runner._validate_derivative_workflow_config(stages, config)

    def test_metric_command_includes_autograd_flag(self) -> None:
        command = self.runner._derivative_metric_command_args(
            python_executable="python",
            result_dir=Path("/tmp/result"),
            output_dir=Path("/tmp/out"),
            source_model="graph2mat",
            settings={
                "finite_difference_method": "central",
                "split": "test",
                "support_threshold": 1e-12,
                "graph2mat_prediction_method": "autograd_vectorized",
            },
        )
        self.assertIn("--graph2mat-prediction-method", command)
        self.assertEqual(
            command[command.index("--graph2mat-prediction-method") + 1],
            "autograd_vectorized",
        )

    def test_metric_command_omits_flag_by_default_and_for_deeph(self) -> None:
        base_settings = {
            "finite_difference_method": "central",
            "split": "test",
            "support_threshold": 1e-12,
        }
        legacy = self.runner._derivative_metric_command_args(
            python_executable="python",
            result_dir=Path("/tmp/result"),
            output_dir=Path("/tmp/out"),
            source_model="graph2mat",
            settings=base_settings,
        )
        self.assertNotIn("--graph2mat-prediction-method", legacy)
        deeph = self.runner._derivative_metric_command_args(
            python_executable="python",
            result_dir=Path("/tmp/result"),
            output_dir=Path("/tmp/out"),
            source_model="deeph",
            settings={**base_settings, "graph2mat_prediction_method": "autograd_vectorized"},
        )
        self.assertNotIn("--graph2mat-prediction-method", deeph)

        deeph_direct = self.runner._derivative_metric_command_args(
            python_executable="python",
            result_dir=Path("/tmp/result"),
            output_dir=Path("/tmp/out"),
            source_model="deeph",
            settings={**base_settings, "deeph_prediction_method": "autograd_vectorized"},
        )
        self.assertIn("--deeph-prediction-method", deeph_direct)
        self.assertEqual(
            deeph_direct[deeph_direct.index("--deeph-prediction-method") + 1],
            "autograd_vectorized",
        )


class DirectDerivativePairTests(unittest.TestCase):
    def metadata(self) -> DerivativeMetadata:
        return DerivativeMetadata(
            sample_id="dH_central_base_plus_minus",
            base_sample_id="base",
            plus_sample_id="plus",
            minus_sample_id="minus",
            atom_index_zero_based=0,
            axis="x",
            axis_index=0,
            delta_ang=DELTA_ANG,
            method="central",
        )

    def test_pair_metrics_match_expected_values(self) -> None:
        ref_plus = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        ref_minus = sparse.csr_matrix((2, 2))
        predicted = sparse.csr_matrix([[3.0, 0.0], [0.0, 3.0]])

        pair = direct_predicted_derivative_pair(
            method="central",
            delta_ang=DELTA_ANG,
            reference_plus=ref_plus,
            reference_minus=ref_minus,
            predicted_matrix=predicted,
            metadata=self.metadata(),
        )

        # Reference: (ref_plus - ref_minus) / (2 * 0.5) = ref_plus.
        np.testing.assert_allclose(pair.reference.matrix.toarray(), ref_plus.toarray())
        np.testing.assert_allclose(pair.predicted.matrix.toarray(), predicted.toarray())
        self.assertEqual(
            pair.diagnostics["predicted_derivative_method"],
            PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
        )
        self.assertEqual(
            pair.diagnostics["reference_derivative_method"],
            REFERENCE_DERIVATIVE_METHOD_SIESTA,
        )
        self.assertEqual(pair.diagnostics["reference_delta_ang"], DELTA_ANG)
        self.assertIsNone(pair.diagnostics["predicted_delta_ang"])
        self.assertIsNone(pair.predicted.metadata["predicted_delta_ang"])
        self.assertEqual(
            pair.diagnostics["dh_signal_to_noise_unavailable_reason"],
            "missing_predicted_operands",
        )

        row = metrics_module.derivative_sparse_metrics(
            pair.reference.matrix,
            pair.predicted.matrix,
            sample="synthetic",
            metadata=self.metadata(),
            source_model="graph2mat",
        )
        self.assertAlmostEqual(row["dh_mae_union_eV_per_Ang"], 1.0)
        self.assertAlmostEqual(row["dh_rmse_union_eV_per_Ang"], 1.0)
        self.assertAlmostEqual(row["dh_cosine_similarity_union"], 1.0)

    def test_pair_rejects_shape_mismatch(self) -> None:
        ref_plus = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        ref_minus = sparse.csr_matrix((2, 2))
        predicted = sparse.csr_matrix(np.ones((3, 3)))
        with self.assertRaises(HamiltonianDerivativeError):
            direct_predicted_derivative_pair(
                method="central",
                delta_ang=DELTA_ANG,
                reference_plus=ref_plus,
                reference_minus=ref_minus,
                predicted_matrix=predicted,
                metadata=self.metadata(),
            )


class DirectDerivativeIOTests(unittest.TestCase):
    def test_load_direct_sparse_derivative_reads_matrix_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path, json_path = direct_derivative_prediction_paths(
                tmp, base_sample_id="base", atom_index_zero_based=0, axis_index=0
            )
            npz_path.parent.mkdir(parents=True)
            matrix = sparse.csr_matrix([[1.5, 0.0], [0.0, -2.5]])
            with npz_path.open("wb") as handle:
                sparse.save_npz(handle, matrix)
            json_path.write_text(json.dumps({"predicted_delta_ang": None, "axis_name": "x"}))

            loaded, metadata = load_direct_sparse_derivative(npz_path)
            np.testing.assert_allclose(loaded.toarray(), matrix.toarray())
            self.assertIsNone(metadata["predicted_delta_ang"])
            self.assertEqual(metadata["axis_name"], "x")

            found = find_direct_derivative_prediction(
                tmp,
                candidate_base_sample_ids=["missing", "base"],
                atom_index_zero_based=0,
                axis_index=0,
            )
            self.assertEqual(found, npz_path)
            self.assertIsNone(
                find_direct_derivative_prediction(
                    tmp,
                    candidate_base_sample_ids=["missing"],
                    atom_index_zero_based=0,
                    axis_index=0,
                )
            )


class DirectPredictionEvaluatorCliTests(unittest.TestCase):
    """End-to-end evaluator runs against a synthetic direct-prediction layout."""

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
    ) -> None:
        structure_dir = self.result_dir / "structures" / sample_id
        structure_dir.mkdir(parents=True, exist_ok=True)
        coordinates = [sign * DELTA_ANG, 0.0, 0.0]
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
        metadata = {
            "sample_id": sample_id,
            "material_label": "graphene",
            "is_reference": sign == 0,
            "atom_index": 1 if sign else None,
            "atom_index_zero_based": 0 if sign else None,
            "axis": "x" if sign else None,
            "axis_index": 0 if sign else None,
            "sign": sign,
            "sign_label": "+" if sign > 0 else "-" if sign < 0 else None,
            "amplitude_ang": DELTA_ANG if sign else 0.0,
            "displacement_ang": coordinates,
            "split_group_id": "fixture:graphene:reference"
            if sign == 0
            else "fixture:graphene:atom_0001:x",
            "claim_status": "diagnostic_only",
            "matrix_shape": [2, 2],
            "hamiltonian_units": "eV",
            "displacement_units": "Ang",
            "derivative_units": "eV/Ang",
            "material_compatibility_hash": "material-hash",
            "orbital_ordering_hash": "orbital-hash",
            "neighbor_list_hash": "neighbor-hash",
            "sparsity_pattern_hash": "sparsity-hash",
            "basis_hash": "basis-hash",
            "pseudopotential_hash": "pseudo-hash",
        }
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

    def write_direct_prediction(
        self,
        matrix: sparse.spmatrix,
        *,
        base_sample_id: str = "base",
        predicted_derivative_method: str = PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
        prediction_method_key: str = "graph2mat_prediction_method",
        metadata_overrides: dict | None = None,
    ) -> Path:
        npz_path, json_path = direct_derivative_prediction_paths(
            self.result_dir,
            base_sample_id=base_sample_id,
            atom_index_zero_based=0,
            axis_index=0,
        )
        self.write_sparse_payload(npz_path, matrix)
        payload = {
            "reference_derivative_method": REFERENCE_DERIVATIVE_METHOD_SIESTA,
            "predicted_derivative_method": predicted_derivative_method,
            "reference_delta_ang": DELTA_ANG,
            "predicted_delta_ang": None,
            "atom_index_zero_based": 0,
            "axis_index": 0,
            "axis_name": "x",
            "units": "eV/Angstrom",
            prediction_method_key: "autograd_vectorized",
            "jacobian_method": "vmap_vjp_chunked",
            "jacobian_chunk_size": 64,
            "topology_fixed": True,
        }
        payload.update(metadata_overrides or {})
        json_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return npz_path

    def write_direct_fixture(
        self,
        *,
        include_direct_prediction: bool = True,
        predicted_derivative_method: str = PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
        prediction_method_key: str = "graph2mat_prediction_method",
        direct_metadata_overrides: dict | None = None,
    ) -> None:
        zero = sparse.csr_matrix((2, 2))
        ref_plus = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        # No predicted_hamiltonians are written anywhere: the direct route
        # must not look for displaced ML_prediction.HSX pairs.
        self.write_sample("base", sign=0, reference=zero)
        self.write_sample("plus", sign=1, reference=ref_plus)
        self.write_sample("minus", sign=-1, reference=zero)
        if include_direct_prediction:
            self.write_direct_prediction(
                sparse.csr_matrix([[3.0, 0.0], [0.0, 3.0]]),
                predicted_derivative_method=predicted_derivative_method,
                prediction_method_key=prediction_method_key,
                metadata_overrides=direct_metadata_overrides,
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
            self.fail(
                f"CLI failed with {completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        return completed

    def test_direct_prediction_metrics_match_expected_values(self) -> None:
        self.write_direct_fixture()

        self.run_cli(
            "--method",
            "central",
            "--overwrite",
            "--graph2mat-prediction-method",
            "autograd_vectorized",
        )

        metrics_root = self.result_dir / "derivative_metrics"
        manifest = json.loads((metrics_root / "manifest.json").read_text(encoding="utf-8"))
        with (metrics_root / "derivative_matrix_metrics.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(manifest["stencils_ok"], 1)
        self.assertEqual(manifest["stencils_failed"], 0)
        self.assertEqual(
            manifest["reference_derivative_method"], REFERENCE_DERIVATIVE_METHOD_SIESTA
        )
        self.assertEqual(
            manifest["predicted_derivative_method"],
            PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
        )
        self.assertEqual(manifest["graph2mat_prediction_method"], "autograd_vectorized")
        self.assertIsNone(manifest["predicted_delta_ang"])

        self.assertEqual(len(rows), 1)
        row = rows[0]
        # dH_ref = (2 - 0) / (2 * 0.5) = 2 on the diagonal; dH_pred = 3 -> MAE 1.
        self.assertAlmostEqual(float(row["dh_mae_union_eV_per_Ang"]), 1.0)
        self.assertAlmostEqual(float(row["dh_rmse_union_eV_per_Ang"]), 1.0)
        self.assertEqual(
            row["predicted_derivative_method"], PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT
        )
        self.assertEqual(row["reference_derivative_method"], REFERENCE_DERIVATIVE_METHOD_SIESTA)
        self.assertEqual(row["graph2mat_prediction_method"], "autograd_vectorized")
        self.assertAlmostEqual(float(row["reference_delta_ang"]), DELTA_ANG)
        self.assertEqual(row["predicted_delta_ang"], "")
        self.assertTrue(row["direct_prediction_path"].endswith("dH_pred_atom0_axis0.npz"))
        self.assertEqual(row["derivative_units"], "eV/Ang")

    def test_deeph_direct_prediction_metrics_match_expected_values(self) -> None:
        self.write_direct_fixture(
            predicted_derivative_method=PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH,
            prediction_method_key="deeph_prediction_method",
        )

        self.run_cli(
            "--method",
            "central",
            "--overwrite",
            "--source-model",
            "deeph",
            "--deeph-prediction-method",
            "autograd_vectorized",
        )

        metrics_root = self.result_dir / "derivative_metrics"
        manifest = json.loads((metrics_root / "manifest.json").read_text(encoding="utf-8"))
        with (metrics_root / "derivative_matrix_metrics.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(manifest["stencils_ok"], 1)
        self.assertEqual(manifest["predicted_derivative_method"], PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH)
        self.assertIsNone(manifest["graph2mat_prediction_method"])
        self.assertEqual(manifest["deeph_prediction_method"], "autograd_vectorized")
        self.assertIsNone(manifest["predicted_delta_ang"])
        self.assertEqual(rows[0]["predicted_derivative_method"], PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH)
        self.assertEqual(rows[0]["deeph_prediction_method"], "autograd_vectorized")
        self.assertEqual(rows[0]["source_model"], "deeph")

    def test_deeph_direct_prediction_equivalence_diagnostic_only_stays_diagnostic(self) -> None:
        self.write_direct_fixture(
            predicted_derivative_method=PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH,
            prediction_method_key="deeph_prediction_method",
            direct_metadata_overrides={
                "claim_status": "diagnostic_only",
                "deeph_diagnostic_only": True,
                "deeph_diagnostic_reason": "basis_equivalence_to_graph2mat_raw_hsx_not_proven",
                "deeph_raw_global_equivalence_proven": False,
                "deeph_equivalence_status": "unproven",
                "deeph_equivalence_scope": "deeph_processed_blockwise_global_hdf5",
            },
        )

        self.run_cli(
            "--method",
            "central",
            "--overwrite",
            "--source-model",
            "deeph",
            "--deeph-prediction-method",
            "autograd_vectorized",
        )

        metrics_root = self.result_dir / "derivative_metrics"
        manifest = json.loads((metrics_root / "manifest.json").read_text(encoding="utf-8"))
        with (metrics_root / "derivative_matrix_metrics.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(manifest["scientific_status"], "diagnostic_only")
        self.assertTrue(manifest["deeph_diagnostic_only"])
        self.assertFalse(manifest["deeph_all_raw_global_equivalence_proven"])
        self.assertEqual(manifest["deeph_raw_global_equivalence_proven_count"], 0)
        self.assertEqual(rows[0]["deeph_diagnostic_only"], "True")
        self.assertEqual(rows[0]["deeph_raw_global_equivalence_proven"], "False")

    def test_deeph_direct_prediction_equivalence_proven_is_not_forced_diagnostic(self) -> None:
        self.write_direct_fixture(
            predicted_derivative_method=PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH,
            prediction_method_key="deeph_prediction_method",
            direct_metadata_overrides={
                "claim_status": "raw_global_equivalence_proven",
                "deeph_diagnostic_only": False,
                "deeph_raw_global_equivalence_proven": True,
                "deeph_equivalence_status": "proven",
                "deeph_equivalence_scope": "raw_global",
                "deeph_equivalence_evidence_paths": ["raw_global_equivalence_evidence.json"],
            },
        )

        self.run_cli(
            "--method",
            "central",
            "--overwrite",
            "--source-model",
            "deeph",
            "--deeph-prediction-method",
            "autograd_vectorized",
        )

        manifest = json.loads(
            (self.result_dir / "derivative_metrics" / "manifest.json").read_text(encoding="utf-8")
        )

        self.assertFalse(manifest["deeph_diagnostic_only"])
        self.assertTrue(manifest["deeph_all_raw_global_equivalence_proven"])
        self.assertEqual(manifest["deeph_raw_global_equivalence_proven_count"], 1)

    def test_direct_mode_fails_closed_without_direct_matrix(self) -> None:
        self.write_direct_fixture(include_direct_prediction=False)

        completed = self.run_cli(
            "--method",
            "central",
            "--overwrite",
            "--graph2mat-prediction-method",
            "autograd_vectorized",
            check=False,
        )

        self.assertEqual(completed.returncode, 2)
        manifest = json.loads(
            (self.result_dir / "derivative_metrics" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertTrue(
            any(
                error["kind"] == "missing_direct_derivative_prediction"
                for error in manifest["fatal_errors"]
            )
        )

    def test_legacy_mode_still_requires_displaced_ml_predictions(self) -> None:
        # Same fixture (no ML_prediction.HSX): the legacy route keeps failing
        # closed, proving the direct route did not relax it.
        self.write_direct_fixture()

        completed = self.run_cli("--method", "central", "--overwrite", check=False)

        self.assertEqual(completed.returncode, 2)
        manifest = json.loads(
            (self.result_dir / "derivative_metrics" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(len(manifest["fatal_errors"]), 1)
        self.assertEqual(manifest["stencils_ok"], 0)

    def test_deeph_source_model_rejects_autograd_method(self) -> None:
        self.write_direct_fixture()
        with self.assertRaises(metrics_module.DerivativeMetricEvaluationError):
            metrics_module.evaluate_derivative_metrics(
                self.result_dir,
                method="central",
                overwrite=True,
                source_model="deeph",
                graph2mat_prediction_method="autograd_vectorized",
            )


if __name__ == "__main__":
    unittest.main()
