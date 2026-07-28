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

from deeph_prediction_adapter import (  # noqa: E402
    DeepHPredictionAdapterError,
    DeepHPredictionAdapterResult,
    EQUIVALENCE_DIAGNOSTIC_LOCAL_FRAME,
    EQUIVALENCE_INVALID_ORBITAL_ORDER,
    EQUIVALENCE_INVALID_SHAPE,
    EQUIVALENCE_INVALID_UNITS,
    EQUIVALENCE_INVALID_EVIDENCE,
    EQUIVALENCE_PROVEN_RAW_GLOBAL,
    RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME,
    RAW_GLOBAL_EQUIVALENCE_REQUIRED_CHECKS,
    EQUIVALENCE_SCOPE_DEEPH_PROCESSED_BLOCKWISE,
    EQUIVALENCE_SCOPE_LOCAL_FRAME,
    EQUIVALENCE_SCOPE_RAW_GLOBAL,
    EQUIVALENCE_STATUS_FAILED,
    EQUIVALENCE_STATUS_NOT_APPLICABLE,
    EQUIVALENCE_STATUS_PROVEN,
    EQUIVALENCE_STATUS_UNPROVEN,
    adapt_deeph_prediction_sample,
    write_adapter_manifest,
)


def write_h5(path: Path, blocks: dict[str, object]) -> None:
    assert h5py is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in blocks.items():
            handle[key] = np.asarray(value, dtype=float)


def write_processed_sample(path: Path) -> None:
    assert np is not None
    path.mkdir(parents=True, exist_ok=True)
    (path / "orbital_types.dat").write_text("0\n0\n", encoding="utf-8")
    (path / "info.json").write_text(json.dumps({"isspinful": False, "isorthogonal": False}) + "\n", encoding="utf-8")
    blocks = {
        "[0, 0, 0, 1, 1]": [[0.0]],
        "[0, 0, 0, 1, 2]": [[1.0]],
        "[0, 0, 0, 2, 1]": [[1.0]],
        "[0, 0, 0, 2, 2]": [[0.0]],
    }
    write_h5(path / "hamiltonians.h5", blocks)
    write_h5(
        path / "overlaps.h5",
        {
            "[0, 0, 0, 1, 1]": [[1.0]],
            "[0, 0, 0, 1, 2]": [[0.0]],
            "[0, 0, 0, 2, 1]": [[0.0]],
            "[0, 0, 0, 2, 2]": [[1.0]],
        },
    )


def minimal_result(*, status: str, diagnostic_only: bool) -> DeepHPredictionAdapterResult:
    return DeepHPredictionAdapterResult(
        sample_id="sample0",
        status="ok",
        metrics_ready=not diagnostic_only,
        diagnostic_only=diagnostic_only,
        diagnostic_reason="" if not diagnostic_only else status,
        prediction_path="/tmp/pred.h5",
        processed_sample_dir="/tmp/processed/sample0",
        reference_hamiltonian_path="/tmp/processed/sample0/hamiltonians.h5",
        reference_overlap_path="/tmp/processed/sample0/overlaps.h5",
        orbital_types_path="/tmp/processed/sample0/orbital_types.dat",
        n_orbitals=2,
        block_count=1,
        prediction_key_count=1,
        reference_key_count=1,
        comparability_status="valid" if not diagnostic_only else "diagnostic_only",
        adapter_equivalence_status=status,
    )


class DeepHPredictionAdapterMetadataTests(unittest.TestCase):
    def test_proven_fixture_enables_robust_metric_eligibility(self) -> None:
        result = minimal_result(status=EQUIVALENCE_PROVEN_RAW_GLOBAL, diagnostic_only=False)

        fields = result.metric_fields()

        self.assertEqual(fields["deeph_adapter_equivalence_status"], EQUIVALENCE_PROVEN_RAW_GLOBAL)
        self.assertEqual(fields["deeph_equivalence_status"], EQUIVALENCE_STATUS_PROVEN)
        self.assertEqual(fields["deeph_equivalence_scope"], EQUIVALENCE_SCOPE_RAW_GLOBAL)
        self.assertTrue(fields["deeph_raw_global_equivalence_proven"])
        self.assertFalse(fields["deeph_diagnostic_only"])
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_adapter_manifest(Path(tmp) / "adapter_manifest.json", [result])
        self.assertTrue(manifest["equivalence_gate"]["robust_claim_allowed"])
        self.assertEqual(manifest["equivalence_statuses"], [EQUIVALENCE_STATUS_PROVEN])

    def test_unit_uncertainty_forces_diagnostic_only(self) -> None:
        result = minimal_result(status=EQUIVALENCE_INVALID_UNITS, diagnostic_only=True)

        fields = result.metric_fields()

        self.assertEqual(fields["deeph_adapter_equivalence_status"], EQUIVALENCE_INVALID_UNITS)
        self.assertEqual(fields["deeph_equivalence_status"], EQUIVALENCE_STATUS_UNPROVEN)
        self.assertFalse(fields["deeph_raw_global_equivalence_proven"])
        self.assertTrue(fields["deeph_diagnostic_only"])

    def test_manifest_records_equivalence_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = minimal_result(status=EQUIVALENCE_INVALID_ORBITAL_ORDER, diagnostic_only=True)

            manifest = write_adapter_manifest(root / "adapter_manifest.json", [result])

            self.assertEqual(manifest["adapter_equivalence_statuses"], [EQUIVALENCE_INVALID_ORBITAL_ORDER])
            self.assertEqual(manifest["equivalence_statuses"], [EQUIVALENCE_STATUS_UNPROVEN])
            self.assertFalse(manifest["equivalence_gate"]["robust_claim_allowed"])
            self.assertTrue(manifest["equivalence_gate"]["diagnostic_only"])
            self.assertEqual(manifest["raw_global_equivalence_proven_count"], 0)
            self.assertFalse(manifest["robust_matrix_metrics_allowed"])

    def test_manifest_records_failed_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = minimal_result(status=EQUIVALENCE_INVALID_SHAPE, diagnostic_only=True)

            manifest = write_adapter_manifest(root / "adapter_manifest.json", [result])

            self.assertEqual(manifest["equivalence_statuses"], [EQUIVALENCE_STATUS_FAILED])
            self.assertFalse(manifest["equivalence_gate"]["robust_claim_allowed"])
            self.assertIn("invalid_shape_mismatch", manifest["equivalence_gate"]["diagnostic_only_reason"])


@unittest.skipUnless(H5PY_AVAILABLE, "h5py/numpy are required for DeepH adapter tests")
class DeepHPredictionAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.processed = self.root / "processed" / "sample0"
        self.work_dir = self.root / "predictions" / "sample0"
        write_processed_sample(self.processed)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_prediction(self, *, incompatible_shape: bool = False) -> None:
        blocks = {
            "[0, 0, 0, 1, 1]": [[0.0]],
            "[0, 0, 0, 1, 2]": [[1.1, 1.2]] if incompatible_shape else [[1.1]],
            "[0, 0, 0, 2, 1]": [[1.1]],
            "[0, 0, 0, 2, 2]": [[0.0]],
        }
        write_h5(self.work_dir / "hamiltonians_pred.h5", blocks)

    def test_missing_deeph_prediction_fails_clearly(self) -> None:
        with self.assertRaisesRegex(DeepHPredictionAdapterError, "Missing DeepH prediction HDF5"):
            adapt_deeph_prediction_sample(
                work_dir=self.work_dir,
                processed_sample_dir=self.processed,
                sample_id="sample0",
            )

    def test_incompatible_shape_fails_clearly(self) -> None:
        self.write_prediction(incompatible_shape=True)

        with self.assertRaisesRegex(DeepHPredictionAdapterError, EQUIVALENCE_INVALID_SHAPE):
            adapt_deeph_prediction_sample(
                work_dir=self.work_dir,
                processed_sample_dir=self.processed,
                sample_id="sample0",
            )

    def test_adapter_output_includes_provenance(self) -> None:
        self.write_prediction()

        result = adapt_deeph_prediction_sample(
            work_dir=self.work_dir,
            processed_sample_dir=self.processed,
            sample_id="sample0",
        )

        payload = result.to_dict()
        self.assertEqual(payload["adapter_version"], "deeph_hdf5_prediction_adapter_v1")
        self.assertEqual(payload["provenance"]["files"]["prediction"]["path"], str(self.work_dir / "hamiltonians_pred.h5"))
        self.assertTrue(payload["provenance"]["files"]["prediction"]["sha256"])
        self.assertTrue(result.metrics_ready)

    def test_diagnostic_only_status_is_emitted_when_equivalence_is_not_proven(self) -> None:
        self.write_prediction()

        result = adapt_deeph_prediction_sample(
            work_dir=self.work_dir,
            processed_sample_dir=self.processed,
            sample_id="sample0",
        )

        self.assertTrue(result.diagnostic_only)
        self.assertEqual(result.comparability_status, "diagnostic_deeph_processed_global_hdf5_blocks_shape_validated")
        self.assertEqual(result.adapter_equivalence_status, EQUIVALENCE_INVALID_ORBITAL_ORDER)
        self.assertEqual(result.equivalence_status, EQUIVALENCE_STATUS_UNPROVEN)
        self.assertEqual(result.equivalence_scope, EQUIVALENCE_SCOPE_DEEPH_PROCESSED_BLOCKWISE)
        self.assertTrue(result.equivalence_evidence_paths)
        self.assertIn("not_proven", result.diagnostic_reason)
        self.assertEqual(result.support_semantics_status, "prediction_and_processed_reference_key_sets_match")
        self.assertFalse(result.metric_fields()["deeph_raw_global_equivalence_proven"])

    def write_raw_global_evidence(self, *, checks: dict[str, object] | None = None) -> Path:
        payload = {
            "schema": "deeph_raw_global_equivalence_evidence_v1",
            "sample_id": "sample0",
            "equivalence_status": "proven",
            "equivalence_scope": "raw_global",
            "checks": {key: True for key in RAW_GLOBAL_EQUIVALENCE_REQUIRED_CHECKS},
            "errors": {"max_abs_hk_error_eV": 0.0, "max_abs_eigenvalue_error_eV": 0.0},
            "tolerances": {"max_abs_hk_error_eV": 1e-8, "max_abs_eigenvalue_error_eV": 1e-8},
            "kpoint_diagnostics": [
                {
                    "kx": 0.0,
                    "ky": 0.0,
                    "kz": 0.0,
                    "raw_reference": {
                        "h_hermiticity_relative": 0.0,
                        "s_hermiticity_relative": 0.0,
                        "s_eigenvalue_min": 1.0,
                        "s_eigenvalue_max": 1.0,
                        "s_condition_number": 1.0,
                        "s_positive_definite": True,
                        "max_normalized_residual": 0.0,
                        "max_s_normalization_error": 0.0,
                        "regularization": {"applied": False, "method": "none"},
                        "valid": True,
                    },
                    "deeph_processed": {
                        "h_hermiticity_relative": 0.0,
                        "s_hermiticity_relative": 0.0,
                        "s_eigenvalue_min": 1.0,
                        "s_eigenvalue_max": 1.0,
                        "s_condition_number": 1.0,
                        "s_positive_definite": True,
                        "max_normalized_residual": 0.0,
                        "max_s_normalization_error": 0.0,
                        "regularization": {"applied": False, "method": "none"},
                        "valid": True,
                    },
                }
            ],
        }
        if checks:
            payload["checks"].update(checks)
        path = self.work_dir / RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def test_raw_global_equivalence_evidence_marks_deeph_as_robust_ready(self) -> None:
        self.write_prediction()
        evidence_path = self.write_raw_global_evidence()

        result = adapt_deeph_prediction_sample(
            work_dir=self.work_dir,
            processed_sample_dir=self.processed,
            sample_id="sample0",
        )

        self.assertFalse(result.diagnostic_only)
        self.assertEqual(result.adapter_equivalence_status, EQUIVALENCE_PROVEN_RAW_GLOBAL)
        self.assertEqual(result.equivalence_status, EQUIVALENCE_STATUS_PROVEN)
        self.assertEqual(result.equivalence_scope, EQUIVALENCE_SCOPE_RAW_GLOBAL)
        self.assertEqual(result.equivalence_evidence_paths, [str(evidence_path)])
        self.assertTrue(result.metric_fields()["deeph_raw_global_equivalence_proven"])

    def test_failed_raw_global_equivalence_evidence_blocks_robust_claim(self) -> None:
        self.write_prediction()
        evidence_path = self.write_raw_global_evidence(checks={"orbital_order": False})

        result = adapt_deeph_prediction_sample(
            work_dir=self.work_dir,
            processed_sample_dir=self.processed,
            sample_id="sample0",
        )

        self.assertTrue(result.diagnostic_only)
        self.assertEqual(result.adapter_equivalence_status, EQUIVALENCE_INVALID_EVIDENCE)
        self.assertEqual(result.equivalence_status, EQUIVALENCE_STATUS_FAILED)
        self.assertEqual(result.equivalence_evidence_paths, [str(evidence_path)])
        self.assertIn("orbital_order", result.equivalence_reason)

    def test_local_frame_prediction_is_not_common_metric_ready(self) -> None:
        write_h5(self.work_dir / "rh_pred.h5", {"[0, 0, 0, 1, 1]": [[0.0]]})

        result = adapt_deeph_prediction_sample(
            work_dir=self.work_dir,
            processed_sample_dir=self.processed,
            sample_id="sample0",
        )

        self.assertFalse(result.metrics_ready)
        self.assertTrue(result.diagnostic_only)
        self.assertEqual(result.target_space, "deeph_local_coordinate_hprime")
        self.assertEqual(result.adapter_equivalence_status, EQUIVALENCE_DIAGNOSTIC_LOCAL_FRAME)
        self.assertEqual(result.equivalence_status, EQUIVALENCE_STATUS_NOT_APPLICABLE)
        self.assertEqual(result.equivalence_scope, EQUIVALENCE_SCOPE_LOCAL_FRAME)

    def test_adapter_manifest_preserves_sample_pairing(self) -> None:
        self.write_prediction()
        result = adapt_deeph_prediction_sample(
            work_dir=self.work_dir,
            processed_sample_dir=self.processed,
            sample_id="frozen_sample_id",
        )

        manifest = write_adapter_manifest(self.root / "adapter_manifest.json", [result])

        self.assertEqual(manifest["sample_count"], 1)
        self.assertEqual(manifest["metrics_ready_count"], 1)
        self.assertEqual(manifest["adapter_equivalence_statuses"], [EQUIVALENCE_INVALID_ORBITAL_ORDER])
        self.assertEqual(manifest["equivalence_statuses"], [EQUIVALENCE_STATUS_UNPROVEN])
        self.assertEqual(manifest["equivalence_scopes"], [EQUIVALENCE_SCOPE_DEEPH_PROCESSED_BLOCKWISE])
        self.assertFalse(manifest["robust_matrix_metrics_allowed"])
        self.assertFalse(manifest["equivalence_gate"]["robust_claim_allowed"])
        self.assertEqual(manifest["samples"][0]["sample_id"], "frozen_sample_id")
        self.assertEqual(manifest["samples"][0]["processed_sample_dir"], str(self.processed))


if __name__ == "__main__":
    unittest.main()
