from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import h5py  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    H5PY_AVAILABLE = True
except ImportError:
    h5py = None
    np = None
    H5PY_AVAILABLE = False

import deeph_raw_global_equivalence_preflight as preflight  # noqa: E402
from deeph_prediction_adapter import adapt_deeph_prediction_sample  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_h5(path: Path, blocks: dict[str, object]) -> None:
    assert h5py is not None
    assert np is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in blocks.items():
            handle[key] = np.asarray(value, dtype=float)


class DeepHRawGlobalEquivalencePreflightMissingMappingTests(unittest.TestCase):
    def test_missing_sample_mapping_writes_failed_evidence_without_numeric_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample0"
            sample.mkdir()
            (sample / "reference.HSX").write_text("reference-placeholder\n", encoding="utf-8")
            frozen = root / "frozen_split_manifest.json"
            write_json(
                frozen,
                {
                    "rows": [
                        {
                            "sample_id": "sample0",
                            "split": "test",
                            "sample_dir": str(sample),
                            "artifact_paths": {"reference_hsx": str(sample / "reference.HSX")},
                        }
                    ]
                },
            )

            manifest = preflight.build_preflight_manifest(
                frozen_split_manifest=frozen,
                graph2mat_result_dir=root / "g2m",
                deeph_processed_dir=root / "processed",
                deeph_predictions_dir=root / "predictions",
                output_dir=root / "equivalence",
                sample_limit=5,
                command=["unit"],
            )

            self.assertEqual(manifest["status"], "failed")
            evidence = json.loads(
                (root / "equivalence" / "sample0" / "raw_global_equivalence_evidence.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(evidence["equivalence_status"], "failed")
            self.assertIn("missing DeepH processed sample mapping", evidence["failure_reason"])


@unittest.skipUnless(H5PY_AVAILABLE, "h5py/numpy are required for numeric preflight tests")
class DeepHRawGlobalEquivalencePreflightNumericTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sample = self.root / "sample0"
        self.processed = self.root / "processed" / "sample0"
        self.predictions = self.root / "predictions" / "sample0"
        self.output = self.root / "equivalence"
        self.sample.mkdir(parents=True)
        self.processed.mkdir(parents=True)
        self.predictions.mkdir(parents=True)
        self.reference = self.sample / "reference.HSX"
        self.reference.write_text("reference-placeholder\n", encoding="utf-8")
        self.run_fdf = self.sample / "RUN.fdf"
        self.run_fdf.write_text("SystemLabel graphene\n", encoding="utf-8")
        self.frozen = self.root / "frozen_split_manifest.json"
        write_json(
            self.frozen,
            {
                "rows": [
                    {
                        "sample_id": "sample0",
                        "split": "test",
                        "sample_dir": str(self.sample),
                        "artifact_paths": {
                            "reference_hsx": str(self.reference),
                            "run_fdf": str(self.run_fdf),
                        },
                    }
                ]
            },
        )
        self._old_raw_reference_matrices = preflight.raw_reference_matrices
        self._old_runtime_helpers = preflight._runtime_helpers
        self._old_kpoints_from_fdf = preflight.kpoints_from_fdf
        preflight.raw_reference_matrices = self.fake_raw_reference_matrices  # type: ignore[method-assign]
        preflight._runtime_helpers = self.fake_runtime_helpers  # type: ignore[method-assign]
        preflight.kpoints_from_fdf = lambda _path: ([(0.0, 0.0, 0.0)], [])  # type: ignore[assignment]

    def tearDown(self) -> None:
        preflight.raw_reference_matrices = self._old_raw_reference_matrices  # type: ignore[assignment]
        preflight._runtime_helpers = self._old_runtime_helpers  # type: ignore[assignment]
        preflight.kpoints_from_fdf = self._old_kpoints_from_fdf  # type: ignore[assignment]
        self.tmp.cleanup()

    def fake_raw_reference_matrices(self, _reference_path, _kpoint):
        assert np is not None
        return {
            "hamiltonian": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
            "overlap": np.eye(2, dtype=np.complex128),
            "spin": "",
            "orthogonal": False,
        }

    def fake_runtime_helpers(self):
        assert np is not None

        def eigenvalues(hamiltonian, _overlap):
            return np.linalg.eigvalsh(np.asarray(hamiltonian, dtype=np.complex128))

        return {"np": np, "complex_generalized_eigenvalues": eigenvalues}

    def write_processed(self, *, scale: float = 1.0, shape_mismatch: bool = False, include_overlap: bool = True) -> None:
        (self.processed / "orbital_types.dat").write_text("0\n0\n", encoding="utf-8")
        write_json(self.processed / "info.json", {"isspinful": False, "isorthogonal": False})
        h_blocks = {
            "[0, 0, 0, 1, 1]": [[0.0]],
            "[0, 0, 0, 1, 2]": [[scale, scale]] if shape_mismatch else [[scale]],
            "[0, 0, 0, 2, 1]": [[scale]],
            "[0, 0, 0, 2, 2]": [[0.0]],
        }
        write_h5(self.processed / "hamiltonians.h5", h_blocks)
        if include_overlap:
            write_h5(
                self.processed / "overlaps.h5",
                {
                    "[0, 0, 0, 1, 1]": [[1.0]],
                    "[0, 0, 0, 1, 2]": [[0.0]],
                    "[0, 0, 0, 2, 1]": [[0.0]],
                    "[0, 0, 0, 2, 2]": [[1.0]],
                },
            )
        pred_blocks = {
            "[0, 0, 0, 1, 1]": [[0.0]],
            "[0, 0, 0, 1, 2]": [[scale]],
            "[0, 0, 0, 2, 1]": [[scale]],
            "[0, 0, 0, 2, 2]": [[0.0]],
        }
        write_h5(self.predictions / "hamiltonians_pred.h5", pred_blocks)

    def run_preflight(self) -> dict:
        return preflight.build_preflight_manifest(
            frozen_split_manifest=self.frozen,
            graph2mat_result_dir=self.root / "g2m",
            deeph_processed_dir=self.root / "processed",
            deeph_predictions_dir=self.root / "predictions",
            output_dir=self.output,
            sample_limit=5,
            command=["unit"],
        )

    def test_exact_synthetic_equivalence_passes_and_adapter_accepts_evidence(self) -> None:
        self.write_processed()

        manifest = self.run_preflight()
        result = adapt_deeph_prediction_sample(
            work_dir=self.predictions,
            processed_sample_dir=self.processed,
            sample_id="sample0",
        )

        self.assertEqual(manifest["status"], "proven")
        self.assertFalse(result.diagnostic_only)
        self.assertTrue(result.metric_fields()["deeph_raw_global_equivalence_proven"])
        evidence = json.loads(
            (self.predictions / "raw_global_equivalence_evidence.json").read_text(
                encoding="utf-8"
            )
        )
        diagnostics = evidence["kpoint_diagnostics"][0]["raw_reference"]
        self.assertTrue(diagnostics["s_positive_definite"])
        self.assertAlmostEqual(diagnostics["s_condition_number"], 1.0)
        self.assertLessEqual(diagnostics["max_normalized_residual"], 1e-8)
        self.assertFalse(diagnostics["regularization"]["applied"])

    def test_audit_only_preflight_does_not_mutate_prediction_directory(self) -> None:
        self.write_processed()

        manifest = preflight.build_preflight_manifest(
            frozen_split_manifest=self.frozen,
            graph2mat_result_dir=self.root / "g2m",
            deeph_processed_dir=self.root / "processed",
            deeph_predictions_dir=self.root / "predictions",
            output_dir=self.output,
            sample_limit=5,
            command=["unit"],
            install_adapter_evidence=False,
        )

        self.assertEqual(manifest["status"], "proven")
        self.assertFalse(manifest["install_adapter_evidence"])
        self.assertFalse((self.predictions / "raw_global_equivalence_evidence.json").exists())

    def test_indefinite_overlap_cannot_be_proven(self) -> None:
        assert np is not None
        diagnostics = preflight.generalized_eigenproblem_diagnostics(
            np.eye(2),
            np.diag([1.0, -0.1]),
            hermiticity_tolerance=1e-10,
            min_overlap_eigenvalue=1e-10,
            max_overlap_condition=1e12,
            residual_tolerance=1e-8,
            normalization_tolerance=1e-8,
        )

        self.assertFalse(diagnostics["valid"])
        self.assertFalse(diagnostics["s_positive_definite"])
        self.assertFalse(diagnostics["regularization"]["applied"])

    def test_excessively_ill_conditioned_overlap_cannot_be_proven(self) -> None:
        assert np is not None
        diagnostics = preflight.generalized_eigenproblem_diagnostics(
            np.eye(2),
            np.diag([1.0, 1e-13]),
            hermiticity_tolerance=1e-10,
            min_overlap_eigenvalue=1e-15,
            max_overlap_condition=1e10,
            residual_tolerance=1e-8,
            normalization_tolerance=1e-8,
        )

        self.assertFalse(diagnostics["valid"])
        self.assertTrue(diagnostics["s_positive_definite"])
        self.assertFalse(diagnostics["s_condition_acceptable"])

    def test_siesta_orbital_mapping_signs_and_energy_shift_pass(self) -> None:
        assert np is not None
        shift_eV = 2.5
        raw_h = np.asarray(
            [
                [1.2, 0.4, -0.3, 0.2],
                [0.4, -0.7, 0.5, -0.6],
                [-0.3, 0.5, 0.9, 0.1],
                [0.2, -0.6, 0.1, -1.1],
            ],
            dtype=float,
        )
        raw_s = np.asarray(
            [
                [1.0, 0.03, -0.02, 0.04],
                [0.03, 1.1, 0.05, -0.01],
                [-0.02, 0.05, 0.9, 0.02],
                [0.04, -0.01, 0.02, 1.2],
            ],
            dtype=float,
        )

        def raw_reference(_reference_path, _kpoint):
            return {
                "hamiltonian": raw_h,
                "overlap": raw_s,
                "spin": "",
                "orthogonal": False,
            }

        preflight.raw_reference_matrices = raw_reference  # type: ignore[assignment]
        orb_indx = self.sample / "graphene.ORB_INDX"
        orb_indx.write_text(
            "\n".join(
                [
                    "1 1 1 C 1 0 0 0 0 0 s 0.0 0 0 0 1",
                    "2 1 1 C 1 0 1 -1 0 0 py 0.0 0 0 0 2",
                    "3 1 1 C 1 0 1 0 0 0 pz 0.0 0 0 0 3",
                    "4 1 1 C 1 0 1 1 0 0 px 0.0 0 0 0 4",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        write_json(
            self.frozen,
            {
                "rows": [
                    {
                        "sample_id": "sample0",
                        "split": "test",
                        "sample_dir": str(self.sample),
                        "artifact_paths": {
                            "reference_hsx": str(self.reference),
                            "run_fdf": str(self.run_fdf),
                            "orb_indx": str(orb_indx),
                        },
                    }
                ]
            },
        )
        (self.processed / "orbital_types.dat").write_text("0 1\n", encoding="utf-8")
        write_json(self.processed / "info.json", {"isspinful": False, "isorthogonal": False})

        # SIESTA order is s,py,pz,px. DeepH processed blocks are s,px,py,pz.
        # The processed Hamiltonian also uses an energy zero shifted by c*S.
        permutation = [0, 2, 3, 1]
        signs = np.asarray([1.0, -1.0, 1.0, -1.0])
        converted_h = raw_h - shift_eV * raw_s
        converted_s = raw_s
        deeph_h = np.zeros_like(converted_h)
        deeph_s = np.zeros_like(converted_s)
        for siesta_i, deeph_i in enumerate(permutation):
            for siesta_j, deeph_j in enumerate(permutation):
                deeph_h[deeph_i, deeph_j] = signs[siesta_i] * converted_h[siesta_i, siesta_j] * signs[siesta_j]
                deeph_s[deeph_i, deeph_j] = signs[siesta_i] * converted_s[siesta_i, siesta_j] * signs[siesta_j]
        blocks = {"[0, 0, 0, 1, 1]": deeph_h}
        write_h5(self.processed / "hamiltonians.h5", blocks)
        write_h5(self.processed / "overlaps.h5", {"[0, 0, 0, 1, 1]": deeph_s})
        write_h5(self.predictions / "hamiltonians_pred.h5", blocks)

        manifest = self.run_preflight()
        evidence = json.loads((self.predictions / "raw_global_equivalence_evidence.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["status"], "proven")
        self.assertEqual(evidence["equivalence_status"], "proven")
        self.assertEqual(evidence["basis_transform"]["status"], "applied")
        self.assertEqual(evidence["basis_transform"]["permutation"], permutation)
        self.assertEqual(evidence["basis_transform"]["signs"], signs.tolist())
        self.assertAlmostEqual(evidence["energy_reference_alignment"]["shift_eV"], shift_eV, places=12)

    def test_shape_mismatch_fails(self) -> None:
        self.write_processed(shape_mismatch=True)

        manifest = self.run_preflight()

        self.assertEqual(manifest["status"], "failed")
        evidence = json.loads((self.predictions / "raw_global_equivalence_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(evidence["equivalence_status"], "failed")

    def test_unit_scale_mismatch_fails_and_adapter_remains_diagnostic(self) -> None:
        self.write_processed(scale=2.0)

        manifest = self.run_preflight()
        result = adapt_deeph_prediction_sample(
            work_dir=self.predictions,
            processed_sample_dir=self.processed,
            sample_id="sample0",
        )

        self.assertEqual(manifest["status"], "failed")
        self.assertTrue(result.diagnostic_only)
        self.assertFalse(result.metric_fields()["deeph_raw_global_equivalence_proven"])

    def test_missing_overlap_fails(self) -> None:
        self.write_processed(include_overlap=False)

        manifest = self.run_preflight()

        self.assertEqual(manifest["status"], "failed")
        evidence = json.loads((self.predictions / "raw_global_equivalence_evidence.json").read_text(encoding="utf-8"))
        self.assertIn("overlaps.h5", evidence["failure_reason"])


if __name__ == "__main__":
    unittest.main()
