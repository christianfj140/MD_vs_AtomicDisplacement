from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_hamiltonian_derivative_stencils as stencil_builder  # noqa: E402
from hamiltonian_derivative_stencil import (  # noqa: E402
    DERIVATIVE_MATRIX_METRIC_TARGET_SPACE,
    DerivativeMatrixInput,
    DerivativeMetadata,
    DerivativeStencil,
    HamiltonianDerivativeError,
    derivative_sparse_metrics,
    discover_derivative_stencils,
    finite_difference_derivative,
    finite_difference_derivative_pair,
    sparse_hermiticity_defect,
    stencil_is_valid,
    validate_derivative_stencil,
    validation_errors,
    validation_warnings,
)


class HamiltonianDerivativeStencilTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_matrix(self, name: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"matrix {name}\n".encode("utf-8"))
        return path

    def write_result_sample(
        self,
        sample_id: str,
        *,
        sign: int | None,
        atom_index_zero_based: int | None = 0,
        axis: str | None = "x",
        amplitude_ang: float = 0.03,
        material_label: str = "graphene",
        split_group_id: str = "generic_cartesian_displacement:graphene:atom_0001",
        is_reference: bool = False,
        include_metadata: bool = True,
        include_reference: bool = True,
        forbidden_reference: bool = False,
        include_prediction: bool = True,
        metadata_split: str | None = None,
        base_sample_id: str | None = None,
        source_base_sample_id: str | None = None,
        reference_sample_id: str | None = None,
    ) -> None:
        structures = self.root / "result" / "structures" / sample_id
        structures.mkdir(parents=True, exist_ok=True)
        (structures / "RUN.fdf").write_text("SystemLabel test\n", encoding="utf-8")
        if include_metadata:
            axis_index = {"x": 0, "y": 1, "z": 2}.get(axis or "", None)
            displacement = [0.0, 0.0, 0.0]
            if sign is not None and axis_index is not None:
                displacement[axis_index] = sign * amplitude_ang
            metadata = {
                "sample_id": sample_id,
                "material_label": material_label,
                "is_reference": is_reference,
                "atom_index": None if atom_index_zero_based is None else atom_index_zero_based + 1,
                "atom_index_zero_based": atom_index_zero_based,
                "axis": axis,
                "axis_index": axis_index,
                "sign": sign,
                "sign_label": "+" if sign == 1 else "-" if sign == -1 else None,
                "amplitude_ang": 0.0 if is_reference else amplitude_ang,
                "displacement_ang": displacement,
                "split_group_id": "generic_cartesian_displacement:graphene:reference"
                if is_reference
                else split_group_id,
                "matrix_shape": [2, 2],
                "material_compatibility_hash": "material-hash",
                "orbital_ordering_hash": "orbital-hash",
                "neighbor_list_hash": "neighbor-hash",
                "sparsity_pattern_hash": "sparsity-hash",
                "basis_hash": "basis-hash",
                "pseudopotential_hash": "pseudo-hash",
            }
            if metadata_split is not None:
                metadata["split"] = metadata_split
            if base_sample_id is not None:
                metadata["base_sample_id"] = base_sample_id
            if source_base_sample_id is not None:
                metadata["source_base_sample_id"] = source_base_sample_id
            if reference_sample_id is not None:
                metadata["reference_sample_id"] = reference_sample_id
            (structures / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        if include_reference or forbidden_reference:
            ref_dir = self.root / "result" / "siesta_hamiltonians" / sample_id
            ref_dir.mkdir(parents=True, exist_ok=True)
            filename = "ML_prediction.HSX" if forbidden_reference else "siesta.TSHS"
            (ref_dir / filename).write_bytes(b"reference\n")
        if include_prediction:
            pred_dir = self.root / "result" / "predicted_hamiltonians" / sample_id
            pred_dir.mkdir(parents=True, exist_ok=True)
            (pred_dir / "ML_prediction.HSX").write_bytes(b"prediction\n")

    def metadata(self, **overrides) -> DerivativeMetadata:
        values = {
            "sample_id": "stencil_0001",
            "base_sample_id": "base",
            "plus_sample_id": "plus",
            "minus_sample_id": "minus",
            "atom_index_zero_based": 0,
            "atom_index_one_based": 1,
            "axis": "x",
            "axis_index": 0,
            "delta_ang": 0.03,
            "method": "central",
            "material_compatibility_hash": "material-hash",
            "orbital_ordering_hash": "orbital-hash",
            "neighbor_list_hash": "neighbor-hash",
            "sparsity_pattern_hash": "sparsity-hash",
            "basis_hash": "basis-hash",
            "pseudopotential_hash": "pseudo-hash",
        }
        values.update(overrides)
        return DerivativeMetadata(**values)

    def matrix(self, role: str, *, source: str, **overrides) -> DerivativeMatrixInput:
        sample_id = "plus" if "plus" in role else "minus"
        filename = f"{role}.TSHS" if source == "siesta" else f"{role}.HSX"
        values = {
            "sample_id": sample_id,
            "source": source,
            "matrix_path": self.write_matrix(filename),
            "matrix_shape": (2, 2),
            "atom_index_zero_based": 0,
            "atom_index_one_based": 1,
            "axis": "x",
            "axis_index": 0,
            "delta_ang": 0.03,
            "material_compatibility_hash": "material-hash",
            "orbital_ordering_hash": "orbital-hash",
            "neighbor_list_hash": "neighbor-hash",
            "sparsity_pattern_hash": "sparsity-hash",
            "basis_hash": "basis-hash",
            "pseudopotential_hash": "pseudo-hash",
        }
        values.update(overrides)
        return DerivativeMatrixInput(**values)

    def stencil(self, **overrides) -> DerivativeStencil:
        values = {
            "metadata": self.metadata(),
            "siesta_plus": self.matrix("siesta_plus", source="siesta"),
            "siesta_minus": self.matrix("siesta_minus", source="siesta"),
            "ml_plus": self.matrix("ml_plus", source="graph2mat"),
            "ml_minus": self.matrix("ml_minus", source="graph2mat"),
            "base_structure_path": self.write_matrix("base/RUN.fdf"),
            "plus_structure_path": self.write_matrix("plus/RUN.fdf"),
            "minus_structure_path": self.write_matrix("minus/RUN.fdf"),
        }
        values.update(overrides)
        return DerivativeStencil(**values)

    def issue_codes(self, stencil: DerivativeStencil) -> set[str]:
        return {issue.code for issue in validate_derivative_stencil(stencil)}

    def test_valid_central_stencil(self) -> None:
        issues = validate_derivative_stencil(self.stencil())

        self.assertTrue(stencil_is_valid(issues))
        self.assertEqual(validation_errors(issues), [])
        self.assertEqual(validation_warnings(issues), [])

    def test_invalid_delta_fails_closed(self) -> None:
        stencil = self.stencil(metadata=self.metadata(delta_ang=0.0))

        self.assertIn("invalid_delta", self.issue_codes(stencil))

    def test_missing_plus_or_minus_for_central_difference_fails(self) -> None:
        stencil = self.stencil(siesta_minus=None)

        self.assertIn("missing_derivative_operand", self.issue_codes(stencil))

    def test_shape_mismatch_fails(self) -> None:
        stencil = self.stencil(ml_minus=self.matrix("ml_minus", source="graph2mat", matrix_shape=(3, 3)))

        self.assertIn("matrix_shape_mismatch", self.issue_codes(stencil))

    def test_unit_mismatch_fails(self) -> None:
        stencil = self.stencil(ml_plus=self.matrix("ml_plus", source="graph2mat", hamiltonian_units="Ry"))

        self.assertIn("unit_mismatch", self.issue_codes(stencil))

    def test_metadata_hash_mismatch_fails(self) -> None:
        stencil = self.stencil(siesta_plus=self.matrix("siesta_plus", source="siesta", material_compatibility_hash="other"))

        self.assertIn("metadata_hash_mismatch", self.issue_codes(stencil))

    def test_forbidden_ml_prediction_as_siesta_reference_fails(self) -> None:
        forbidden = self.write_matrix("ML_prediction.HSX")
        stencil = self.stencil(
            siesta_plus=self.matrix("siesta_plus", source="siesta", matrix_path=forbidden)
        )

        self.assertIn("forbidden_siesta_reference", self.issue_codes(stencil))

    def test_diagnostic_warning_when_ordering_and_comparability_metadata_missing(self) -> None:
        metadata = self.metadata(
            material_compatibility_hash=None,
            orbital_ordering_hash=None,
            neighbor_list_hash=None,
            sparsity_pattern_hash=None,
            basis_hash=None,
            pseudopotential_hash=None,
        )
        stencil = self.stencil(
            metadata=metadata,
            siesta_plus=self.matrix(
                "siesta_plus",
                source="siesta",
                material_compatibility_hash=None,
                orbital_ordering_hash=None,
                neighbor_list_hash=None,
                sparsity_pattern_hash=None,
                basis_hash=None,
                pseudopotential_hash=None,
            ),
            siesta_minus=self.matrix(
                "siesta_minus",
                source="siesta",
                material_compatibility_hash=None,
                orbital_ordering_hash=None,
                neighbor_list_hash=None,
                sparsity_pattern_hash=None,
                basis_hash=None,
                pseudopotential_hash=None,
            ),
            ml_plus=self.matrix(
                "ml_plus",
                source="deeph",
                material_compatibility_hash=None,
                orbital_ordering_hash=None,
                neighbor_list_hash=None,
                sparsity_pattern_hash=None,
                basis_hash=None,
                pseudopotential_hash=None,
            ),
            ml_minus=self.matrix(
                "ml_minus",
                source="deeph",
                material_compatibility_hash=None,
                orbital_ordering_hash=None,
                neighbor_list_hash=None,
                sparsity_pattern_hash=None,
                basis_hash=None,
                pseudopotential_hash=None,
            ),
        )
        issues = validate_derivative_stencil(stencil)
        warning_codes = {issue.code for issue in validation_warnings(issues)}

        self.assertTrue(stencil_is_valid(issues))
        self.assertIn("missing_orbital_ordering_hash", warning_codes)
        self.assertIn("missing_material_compatibility_hash", warning_codes)
        self.assertIn("missing_neighbor_list_hash", warning_codes)
        self.assertIn("missing_sparsity_pattern_hash", warning_codes)

    def test_missing_required_metadata_fails_for_non_diagnostic_claim(self) -> None:
        metadata = self.metadata(
            claim_status="robust",
            material_compatibility_hash=None,
            orbital_ordering_hash=None,
        )
        stencil = self.stencil(metadata=metadata)

        self.assertIn("missing_required_metadata", self.issue_codes(stencil))

    def test_central_difference_exact_known_result(self) -> None:
        plus = sparse.csr_matrix([[3.0, 1.0], [1.0, 5.0]])
        minus = sparse.csr_matrix([[1.0, -1.0], [-1.0, 1.0]])

        result = finite_difference_derivative(
            method="central",
            delta_ang=0.5,
            plus=plus,
            minus=minus,
            source="siesta",
            matrix_hashes={"plus": "p", "minus": "m"},
            metadata=self.metadata(delta_ang=0.5),
        )

        self.assertTrue(sparse.isspmatrix_csr(result.matrix))
        np.testing.assert_allclose(result.matrix.toarray(), [[2.0, 2.0], [2.0, 4.0]])
        self.assertEqual(result.metadata["method"], "central")
        self.assertEqual(result.metadata["source"], "siesta")
        self.assertEqual(result.metadata["derivative_units"], "eV/Ang")
        self.assertEqual(result.metadata["matrix_hashes"], {"plus": "p", "minus": "m"})
        self.assertEqual(result.metadata["validation_status"], "valid")
        self.assertTrue(result.metadata["finite_values"])

    def test_forward_and_backward_difference_exact_known_result(self) -> None:
        base = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
        plus = sparse.csr_matrix([[2.0, 2.0], [0.0, 4.0]])
        minus = sparse.csr_matrix([[0.0, -2.0], [0.0, -2.0]])

        forward = finite_difference_derivative(
            method="forward",
            delta_ang=0.5,
            base=base,
            plus=plus,
            source="graph2mat",
        )
        backward = finite_difference_derivative(
            method="backward",
            delta_ang=0.5,
            base=base,
            minus=minus,
            source="graph2mat",
        )

        np.testing.assert_allclose(forward.matrix.toarray(), [[2.0, 4.0], [0.0, 6.0]])
        np.testing.assert_allclose(backward.matrix.toarray(), [[2.0, 4.0], [0.0, 6.0]])

    def test_zero_or_negative_delta_error(self) -> None:
        plus = sparse.eye(2, format="csr")
        minus = sparse.eye(2, format="csr")

        with self.assertRaisesRegex(HamiltonianDerivativeError, "delta_ang must be positive"):
            finite_difference_derivative(method="central", delta_ang=0.0, plus=plus, minus=minus, source="siesta")
        with self.assertRaisesRegex(HamiltonianDerivativeError, "delta_ang must be positive"):
            finite_difference_derivative(method="central", delta_ang=-0.1, plus=plus, minus=minus, source="siesta")

    def test_derivative_shape_mismatch_error(self) -> None:
        with self.assertRaisesRegex(HamiltonianDerivativeError, "shape mismatch"):
            finite_difference_derivative(
                method="central",
                delta_ang=0.1,
                plus=sparse.eye(2, format="csr"),
                minus=sparse.eye(3, format="csr"),
                source="siesta",
            )

    def test_input_matrices_are_not_mutated(self) -> None:
        plus = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        minus = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
        plus_before = plus.copy()
        minus_before = minus.copy()

        finite_difference_derivative(method="central", delta_ang=1.0, plus=plus, minus=minus, source="siesta")

        np.testing.assert_allclose(plus.toarray(), plus_before.toarray())
        np.testing.assert_allclose(minus.toarray(), minus_before.toarray())

    def test_hermiticity_defect_for_hermitian_and_non_hermitian_derivatives(self) -> None:
        hermitian = finite_difference_derivative(
            method="central",
            delta_ang=1.0,
            plus=sparse.csr_matrix([[2.0, 1.0], [1.0, 2.0]]),
            minus=sparse.csr_matrix([[0.0, 0.0], [0.0, 0.0]]),
            source="siesta",
        )
        non_hermitian = finite_difference_derivative(
            method="central",
            delta_ang=1.0,
            plus=sparse.csr_matrix([[0.0, 2.0], [0.0, 0.0]]),
            minus=sparse.csr_matrix([[0.0, 0.0], [0.0, 0.0]]),
            source="siesta",
        )

        self.assertEqual(hermitian.metadata["dH_hermiticity_defect"], 0.0)
        self.assertGreater(non_hermitian.metadata["dH_hermiticity_defect"], 0.0)
        self.assertEqual(sparse_hermiticity_defect(hermitian.matrix), 0.0)

    def test_sparse_hermiticity_defect_returns_nan_for_rectangular_matrix(self) -> None:
        defect = sparse_hermiticity_defect(sparse.csr_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        self.assertTrue(np.isnan(defect))

    def test_support_change_diagnostic(self) -> None:
        result = finite_difference_derivative(
            method="central",
            delta_ang=1.0,
            plus=sparse.csr_matrix([[1.0, 0.0], [0.0, 0.0]]),
            minus=sparse.csr_matrix([[0.0, 1.0], [0.0, 0.0]]),
            source="siesta",
        )

        self.assertTrue(result.metadata["plus_minus_support_changed"])
        self.assertEqual(result.metadata["derivative_nnz"], 2)
        self.assertAlmostEqual(result.metadata["derivative_density"], 0.5)

    def test_pair_diagnostics_include_ref_and_pred_hermiticity(self) -> None:
        reference_plus = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        reference_minus = sparse.csr_matrix([[0.0, 0.0], [0.0, 0.0]])
        predicted_plus = sparse.csr_matrix([[0.0, 2.0], [0.0, 0.0]])
        predicted_minus = sparse.csr_matrix([[0.0, 0.0], [0.0, 0.0]])

        result = finite_difference_derivative_pair(
            method="central",
            delta_ang=1.0,
            reference_plus=reference_plus,
            reference_minus=reference_minus,
            predicted_plus=predicted_plus,
            predicted_minus=predicted_minus,
            predicted_source="deeph",
        )

        self.assertEqual(result.diagnostics["dH_ref_hermiticity_defect"], 0.0)
        self.assertGreater(result.diagnostics["dH_pred_hermiticity_defect"], 0.0)
        self.assertTrue(result.diagnostics["finite_values"])

    def test_derivative_sparse_metrics_exact_match_has_zero_absolute_errors(self) -> None:
        reference = sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
        predicted = reference.copy()

        row = derivative_sparse_metrics(
            reference,
            predicted,
            sample="s0",
            metadata=self.metadata(claim_status="robust"),
            source_model="graph2mat",
        )

        self.assertEqual(row["dh_mae_ref_eV_per_Ang"], 0.0)
        self.assertEqual(row["dh_rmse_union_eV_per_Ang"], 0.0)
        self.assertEqual(row["dh_mse_ref_eV2_per_Ang2"], 0.0)
        self.assertEqual(row["dh_relative_frobenius_ref"], 0.0)
        self.assertEqual(row["dh_relative_frobenius_union"], 0.0)
        self.assertEqual(row["dh_support_precision"], 1.0)
        self.assertEqual(row["dh_support_recall"], 1.0)
        self.assertEqual(row["comparison_status"], "robust")

    def test_derivative_sparse_metrics_known_small_values(self) -> None:
        reference = sparse.csr_matrix([[1.0, 0.0], [0.0, 3.0]])
        predicted = sparse.csr_matrix([[2.0, 0.0], [0.0, 1.0]])

        row = derivative_sparse_metrics(reference, predicted, sample="s0", metadata=self.metadata())

        self.assertAlmostEqual(row["dh_mae_ref_eV_per_Ang"], 1.5)
        self.assertAlmostEqual(row["dh_rmse_ref_eV_per_Ang"], (2.5) ** 0.5)
        self.assertAlmostEqual(row["dh_mse_ref_eV2_per_Ang2"], 2.5)
        self.assertAlmostEqual(row["dh_max_abs_error_union_eV_per_Ang"], 2.0)
        self.assertAlmostEqual(row["dh_relative_frobenius_ref"], (5.0 / 10.0) ** 0.5)
        self.assertAlmostEqual(row["dh_relative_l1_union"], 3.0 / 4.0)
        self.assertAlmostEqual(row["dh_cosine_similarity_union"], 5.0 / ((10.0) ** 0.5 * (5.0) ** 0.5))

    def test_derivative_sparse_metrics_zero_reference_norm_behavior(self) -> None:
        reference = sparse.csr_matrix((2, 2))
        predicted = sparse.csr_matrix([[1.0, 0.0], [0.0, 0.0]])

        row = derivative_sparse_metrics(reference, predicted, sample="s0", metadata=self.metadata())

        self.assertTrue(np.isnan(row["dh_relative_frobenius_ref"]))
        self.assertTrue(np.isnan(row["dh_relative_frobenius_union"]))
        self.assertTrue(np.isnan(row["dh_relative_l1_union"]))
        self.assertEqual(row["dh_relative_unavailable_reason"], "reference_derivative_norm_zero")
        self.assertTrue(np.isnan(row["dh_cosine_similarity_union"]))
        self.assertEqual(row["dh_cosine_unavailable_reason"], "reference_derivative_norm_zero")

    def test_derivative_sparse_metrics_support_mismatch_behavior(self) -> None:
        reference = sparse.csr_matrix([[1.0, 0.0], [0.0, 0.0]])
        predicted = sparse.csr_matrix([[0.0, 0.0], [0.0, 2.0]])

        row = derivative_sparse_metrics(reference, predicted, sample="s0", metadata=self.metadata())

        self.assertEqual(row["dh_support_precision"], 0.0)
        self.assertEqual(row["dh_support_recall"], 0.0)
        self.assertEqual(row["dh_support_f1"], 0.0)
        self.assertEqual(row["dh_false_zero_rate"], 1.0)
        self.assertEqual(row["dh_false_nonzero_rate"], 1.0)
        self.assertEqual(row["dh_union_nnz"], 2)

    def test_derivative_sparse_metrics_cosine_similarity_edge_cases(self) -> None:
        zero = sparse.csr_matrix((2, 2))
        nonzero = sparse.csr_matrix([[1.0, 0.0], [0.0, 0.0]])

        both_zero = derivative_sparse_metrics(zero, zero, sample="s0", metadata=self.metadata())
        pred_zero = derivative_sparse_metrics(nonzero, zero, sample="s1", metadata=self.metadata())

        self.assertTrue(np.isnan(both_zero["dh_cosine_similarity_union"]))
        self.assertEqual(both_zero["dh_cosine_unavailable_reason"], "reference_and_prediction_derivative_norm_zero")
        self.assertTrue(np.isnan(pred_zero["dh_cosine_similarity_union"]))
        self.assertEqual(pred_zero["dh_cosine_unavailable_reason"], "prediction_derivative_norm_zero")

    def test_derivative_sparse_metrics_metadata_fields_and_units_present(self) -> None:
        metadata = self.metadata(delta_ang=0.04, method="central", claim_status="diagnostic_only")
        row = derivative_sparse_metrics(
            sparse.eye(2, format="csr"),
            sparse.eye(2, format="csr"),
            sample="sample-x",
            metadata=metadata,
            source_model="deeph",
            reference_source="siesta",
        )

        self.assertEqual(row["sample"], "sample-x")
        self.assertEqual(row["atom_index_zero_based"], 0)
        self.assertEqual(row["axis"], "x")
        self.assertEqual(row["axis_index"], 0)
        self.assertEqual(row["delta_ang"], 0.04)
        self.assertEqual(row["finite_difference_method"], "central")
        self.assertEqual(row["source_model"], "deeph")
        self.assertEqual(row["reference_source"], "siesta")
        self.assertEqual(row["derivative_units"], "eV/Ang")
        self.assertEqual(row["hamiltonian_units"], "eV")
        self.assertEqual(row["displacement_units"], "Ang")
        self.assertEqual(row["matrix_metric_target_space"], DERIVATIVE_MATRIX_METRIC_TARGET_SPACE)
        self.assertEqual(row["comparison_status"], "diagnostic_only")

    def test_discovery_groups_central_stencil_from_plus_minus_metadata(self) -> None:
        self.write_result_sample("base", sign=0, is_reference=True)
        self.write_result_sample("plus", sign=1)
        self.write_result_sample("minus", sign=-1)

        discoveries = discover_derivative_stencils(
            self.root / "result",
            method="graph2mat",
            finite_difference_method="central",
        )

        self.assertEqual(len(discoveries), 1)
        discovery = discoveries[0]
        self.assertEqual(discovery.status, "valid")
        self.assertEqual(discovery.method, "central")
        self.assertIsNotNone(discovery.stencil)
        self.assertEqual(discovery.stencil.metadata.plus_sample_id, "plus")
        self.assertEqual(discovery.stencil.metadata.minus_sample_id, "minus")
        self.assertEqual(discovery.stencil.siesta_plus.matrix_path.name, "siesta.TSHS")
        self.assertEqual(discovery.stencil.ml_plus.matrix_path.name, "ML_prediction.HSX")

    def test_discovery_groups_forward_from_base_and_plus(self) -> None:
        self.write_result_sample("base", sign=0, is_reference=True)
        self.write_result_sample("plus", sign=1)

        discoveries = discover_derivative_stencils(
            self.root / "result",
            method="deeph",
            finite_difference_method="forward",
        )

        self.assertEqual(len(discoveries), 1)
        discovery = discoveries[0]
        self.assertEqual(discovery.status, "valid")
        self.assertEqual(discovery.method, "forward")
        self.assertIsNotNone(discovery.stencil.siesta_base)
        self.assertIsNotNone(discovery.stencil.ml_base)
        self.assertEqual(discovery.stencil.ml_plus.source, "deeph")

    def test_discovery_forward_with_ambiguous_base_is_reported(self) -> None:
        self.write_result_sample("base_a", sign=0, is_reference=True)
        self.write_result_sample("base_b", sign=0, is_reference=True)
        self.write_result_sample("plus", sign=1)

        discoveries = discover_derivative_stencils(
            self.root / "result",
            finite_difference_method="forward",
        )

        self.assertEqual(len(discoveries), 1)
        discovery = discoveries[0]
        self.assertEqual(discovery.status, "ambiguous")
        self.assertIsNone(discovery.stencil)
        self.assertIn("ambiguous_base_sample", {issue.code for issue in discovery.issues})

    def test_discovery_forward_without_base_is_incomplete(self) -> None:
        self.write_result_sample("plus", sign=1)

        discoveries = discover_derivative_stencils(
            self.root / "result",
            finite_difference_method="forward",
        )

        self.assertEqual(len(discoveries), 1)
        self.assertEqual(discoveries[0].status, "incomplete")
        self.assertIn("incomplete_derivative_stencil", {issue.code for issue in discoveries[0].issues})

    def test_discovery_missing_minus_with_require_central_is_incomplete(self) -> None:
        self.write_result_sample("base", sign=0, is_reference=True)
        self.write_result_sample("plus", sign=1)

        discoveries = discover_derivative_stencils(
            self.root / "result",
            finite_difference_method="central",
            require_central=True,
        )

        self.assertEqual(len(discoveries), 1)
        self.assertEqual(discoveries[0].status, "incomplete")
        self.assertIn("incomplete_derivative_stencil", {issue.code for issue in discoveries[0].issues})
        self.assertEqual(discoveries[0].issues[0].details["missing"], ["minus"])

    def test_discovery_mismatched_delta_does_not_pair_as_central(self) -> None:
        self.write_result_sample("plus", sign=1, amplitude_ang=0.03)
        self.write_result_sample("minus", sign=-1, amplitude_ang=0.04)

        discoveries = discover_derivative_stencils(
            self.root / "result",
            finite_difference_method="central",
            require_central=True,
        )

        self.assertEqual(len(discoveries), 2)
        self.assertEqual({item.status for item in discoveries}, {"incomplete"})
        self.assertTrue(all(item.stencil is None for item in discoveries))

    def test_discovery_mismatched_atom_does_not_pair_as_central(self) -> None:
        self.write_result_sample("plus", sign=1, atom_index_zero_based=0)
        self.write_result_sample("minus", sign=-1, atom_index_zero_based=1)

        discoveries = discover_derivative_stencils(
            self.root / "result",
            finite_difference_method="central",
            require_central=True,
        )

        self.assertEqual(len(discoveries), 2)
        self.assertEqual({item.status for item in discoveries}, {"incomplete"})

    def test_discovery_ambiguous_duplicate_plus_sample(self) -> None:
        self.write_result_sample("plus_a", sign=1)
        self.write_result_sample("plus_b", sign=1)
        self.write_result_sample("minus", sign=-1)

        discoveries = discover_derivative_stencils(
            self.root / "result",
            finite_difference_method="central",
        )

        self.assertEqual(len(discoveries), 1)
        self.assertEqual(discoveries[0].status, "ambiguous")
        self.assertIn("ambiguous_derivative_pairing", {issue.code for issue in discoveries[0].issues})

    def test_discovery_central_with_ambiguous_base_remains_valid(self) -> None:
        self.write_result_sample("base_a", sign=0, is_reference=True)
        self.write_result_sample("base_b", sign=0, is_reference=True)
        self.write_result_sample("plus", sign=1)
        self.write_result_sample("minus", sign=-1)

        discoveries = discover_derivative_stencils(
            self.root / "result",
            finite_difference_method="central",
        )

        self.assertEqual(len(discoveries), 1)
        self.assertIn(discoveries[0].status, {"valid", "diagnostic_only"})
        self.assertIsNotNone(discoveries[0].stencil)
        self.assertNotIn("ambiguous_base_sample", {issue.code for issue in discoveries[0].issues})

    def test_discovery_chooses_base_by_displaced_base_sample_id(self) -> None:
        self.write_result_sample("base_a", sign=0, is_reference=True)
        self.write_result_sample("base_b", sign=0, is_reference=True)
        self.write_result_sample("plus", sign=1, base_sample_id="base_b")
        self.write_result_sample("minus", sign=-1, base_sample_id="base_b")

        discoveries = discover_derivative_stencils(
            self.root / "result",
            finite_difference_method="central",
        )

        self.assertEqual(len(discoveries), 1)
        discovery = discoveries[0]
        self.assertIsNotNone(discovery.stencil)
        self.assertEqual(discovery.stencil.base_structure_path.parent.name, "base_b")
        self.assertNotIn("ambiguous_base_sample", {issue.code for issue in discovery.issues})

    def test_discovery_forbidden_reference_candidate_is_reported(self) -> None:
        self.write_result_sample("plus", sign=1, include_reference=False, forbidden_reference=True)
        self.write_result_sample("minus", sign=-1)

        discoveries = discover_derivative_stencils(
            self.root / "result",
            finite_difference_method="central",
        )

        self.assertEqual(len(discoveries), 1)
        self.assertEqual(discoveries[0].status, "incomplete")
        self.assertIn("forbidden_siesta_reference", {issue.code for issue in discoveries[0].issues})

    def test_discovery_missing_metadata_fallback_is_diagnostic_incomplete(self) -> None:
        self.write_result_sample("unknown", sign=1, include_metadata=False)

        discoveries = discover_derivative_stencils(self.root / "result")

        self.assertEqual(len(discoveries), 1)
        self.assertEqual(discoveries[0].status, "incomplete")
        self.assertIsNone(discoveries[0].stencil)
        self.assertEqual(discoveries[0].details["comparison_status"], "diagnostic_only")
        self.assertIn("missing_metadata", {issue.code for issue in discoveries[0].issues})
        self.assertIn("insufficient_metadata_for_pairing", {issue.code for issue in discoveries[0].issues})

    def test_split_specific_discovery_accepts_matching_split_metadata(self) -> None:
        self.write_result_sample("plus", sign=1, metadata_split="test")
        self.write_result_sample("minus", sign=-1, metadata_split="test")

        discoveries = discover_derivative_stencils(
            self.root / "result",
            split="test",
            finite_difference_method="central",
            require_central=True,
        )

        self.assertEqual(len(discoveries), 1)
        self.assertEqual(discoveries[0].status, "valid")
        self.assertIsNotNone(discoveries[0].stencil)

    def test_split_specific_discovery_excludes_other_split_metadata(self) -> None:
        self.write_result_sample("plus", sign=1, metadata_split="test")
        self.write_result_sample("minus", sign=-1, metadata_split="train")

        discoveries = discover_derivative_stencils(
            self.root / "result",
            split="test",
            finite_difference_method="central",
            require_central=True,
        )

        self.assertEqual(len(discoveries), 1)
        self.assertEqual(discoveries[0].status, "incomplete")
        self.assertIsNone(discoveries[0].stencil)
        self.assertEqual(discoveries[0].issues[0].code, "incomplete_derivative_stencil")
        self.assertEqual(discoveries[0].issues[0].details["missing"], ["minus"])

    def test_split_specific_discovery_rejects_missing_split_metadata_fail_closed(self) -> None:
        self.write_result_sample("plus", sign=1, metadata_split="test")
        self.write_result_sample("minus", sign=-1)

        discoveries = discover_derivative_stencils(
            self.root / "result",
            split="test",
            finite_difference_method="central",
            require_central=True,
        )

        self.assertEqual(len(discoveries), 2)
        missing_split = next(item for item in discoveries if any(issue.code == "missing_split_metadata" for issue in item.issues))
        self.assertEqual(missing_split.status, "incomplete")
        self.assertIsNone(missing_split.stencil)
        self.assertEqual(missing_split.details["comparison_status"], "diagnostic_only")
        self.assertEqual(missing_split.issues[-1].details["requested_split"], "test")
        incomplete = next(item for item in discoveries if any(issue.code == "incomplete_derivative_stencil" for issue in item.issues))
        self.assertIsNone(incomplete.stencil)

    def test_split_all_keeps_missing_split_metadata_inclusive(self) -> None:
        self.write_result_sample("plus", sign=1)
        self.write_result_sample("minus", sign=-1)

        discoveries = discover_derivative_stencils(
            self.root / "result",
            split="all",
            finite_difference_method="central",
            require_central=True,
        )

        self.assertEqual(len(discoveries), 1)
        self.assertIsNotNone(discoveries[0].stencil)
        self.assertNotIn("missing_split_metadata", {issue.code for issue in discoveries[0].issues})

    def test_builder_assigns_derivative_metadata_to_shared_base_sample(self) -> None:
        source_root = self.root / "source_dataset"
        sample_dir = source_root / "splits" / "test" / "md_270"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "RUN.fdf").write_text("SystemLabel test\n", encoding="utf-8")
        (sample_dir / "metadata.json").write_text(
            json.dumps({"structure_type": "cartesian", "material_label": "graphene"}),
            encoding="utf-8",
        )
        output_root = self.root / "derivative_result"
        captured_calls: list[dict[str, object]] = []

        def fake_write_structure_sample(**kwargs):
            captured_calls.append(kwargs)
            output_sample_dir = Path(kwargs["output_sample_dir"])
            return {
                "sample_id": kwargs["sample_id"],
                "sample_dir": str(output_sample_dir),
                "run_fdf": str(output_sample_dir / "RUN.fdf"),
                "metadata_path": str(output_sample_dir / "metadata.json"),
                "sign": kwargs["metadata"].get("sign"),
                "sign_label": kwargs["metadata"].get("sign_label"),
                "atom_index_zero_based": kwargs["metadata"].get("atom_index_zero_based"),
                "axis": kwargs["metadata"].get("axis"),
                "delta_ang": kwargs["metadata"].get("delta_ang"),
            }

        with (
            mock.patch.object(
                stencil_builder,
                "load_base_rows",
                return_value=[{"sample_id": "md_270", "sample_dir": str(sample_dir)}],
            ),
            mock.patch.object(
                stencil_builder,
                "extract_fdf_structure",
                return_value=SimpleNamespace(
                    positions_ang=[[0.0, 0.0, 0.0]],
                    atom_species=["C"],
                    lattice_vectors_ang=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    structure_type="cartesian",
                ),
            ),
            mock.patch.object(stencil_builder, "write_structure_sample", side_effect=fake_write_structure_sample),
        ):
            manifest = stencil_builder.build_derivative_stencils(
                source_dataset_root=source_root,
                output_stencil_root=output_root,
                split="test",
                method="central",
                delta_ang_values=[0.005, 0.01],
                atom_indices_zero_based=[0],
                axes=["x"],
                base_sample_ids=["md_270"],
                max_base_snapshots=1,
                include_base=True,
                overwrite=True,
            )

        base_call = next(call for call in captured_calls if call["sample_id"] == "md_270_base")
        self.assertEqual(base_call["metadata"]["atom_index_zero_based"], 0)
        self.assertEqual(base_call["metadata"]["axis"], "x")
        self.assertEqual(base_call["metadata"]["axis_index"], 0)
        self.assertEqual(base_call["metadata"]["sign"], 0)
        self.assertEqual(base_call["metadata"]["base_sample_id"], "md_270_base")
        self.assertEqual(base_call["metadata"]["source_base_sample_id"], "md_270")
        self.assertEqual(manifest["stencils"][0]["base_sample_id"], "md_270_base")
        self.assertEqual(manifest["stencils"][0]["plus_sample_id"], "md_270_atom0000_x_d0.005_plus")
        self.assertEqual(manifest["stencils"][0]["minus_sample_id"], "md_270_atom0000_x_d0.005_minus")


if __name__ == "__main__":
    unittest.main()
