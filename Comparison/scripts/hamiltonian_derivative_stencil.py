#!/usr/bin/env python3
"""Data contracts for finite-difference Hamiltonian derivative stencils."""

from __future__ import annotations

import math
import json
import sys
from dataclasses import asdict, dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from fdf_materialization import extract_fdf_structure  # noqa: E402
from reference_selection import choose_reference_matrix, file_sha256


VALID_AXES = {"x": 0, "y": 1, "z": 2}
VALID_METHODS = {"central", "forward", "backward"}
VALID_SOURCES = {"siesta", "graph2mat", "deeph"}
EXPECTED_HAMILTONIAN_UNITS = "eV"
EXPECTED_DISPLACEMENT_UNITS = "Ang"
EXPECTED_DERIVATIVE_UNITS = "eV/Ang"
DERIVATIVE_SUPPORT_THRESHOLD = 1e-12
DERIVATIVE_MATRIX_METRIC_TARGET_SPACE = "raw_global_hamiltonian_derivative"
FORBIDDEN_SIESTA_REFERENCE_NAMES = {"ML_prediction.HSX"}
DEFAULT_GEOMETRY_TOLERANCE_ANG = 1e-8
DIAGNOSTIC_STATUSES = {"", "diagnostic", "diagnostic_only", "exploratory"}
PAPER_LEVEL_STATUSES = {"paper_ready", "publication", "publicable", "final_publication"}
REQUIRED_NON_DIAGNOSTIC_HASHES = ("material_compatibility_hash", "orbital_ordering_hash")
OPTIONAL_COMPARABILITY_HASHES = (
    "material_compatibility_hash",
    "orbital_ordering_hash",
    "neighbor_list_hash",
    "sparsity_pattern_hash",
)
COMPARABILITY_HASH_FIELDS = (
    "material_compatibility_hash",
    "orbital_ordering_hash",
    "neighbor_list_hash",
    "sparsity_pattern_hash",
    "basis_hash",
    "pseudopotential_hash",
)


class HamiltonianDerivativeError(RuntimeError):
    """Raised when a finite-difference Hamiltonian derivative is not well-defined."""


@dataclass(frozen=True)
class DerivativeValidationIssue:
    severity: str
    code: str
    message: str
    field: str | None = None
    sample_id: str | None = None
    matrix_role: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DerivativeMetadata:
    sample_id: str
    plus_sample_id: str | None
    minus_sample_id: str | None
    atom_index_zero_based: int | None
    axis: str | None
    axis_index: int | None
    delta_ang: float | None
    base_sample_id: str | None = None
    atom_index_one_based: int | None = None
    hamiltonian_units: str = EXPECTED_HAMILTONIAN_UNITS
    displacement_units: str = EXPECTED_DISPLACEMENT_UNITS
    derivative_units: str = EXPECTED_DERIVATIVE_UNITS
    method: str = "central"
    claim_status: str = "diagnostic_only"
    material_compatibility_hash: str | None = None
    orbital_ordering_hash: str | None = None
    neighbor_list_hash: str | None = None
    sparsity_pattern_hash: str | None = None
    basis_hash: str | None = None
    pseudopotential_hash: str | None = None
    structure_hash: str | None = None
    metadata_hash: str | None = None
    extra_hashes: dict[str, str] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DerivativeMatrixInput:
    sample_id: str
    source: str | None
    matrix_path: Path | str | None
    matrix_shape: tuple[int, int] | list[int] | None
    matrix_sha256: str | None = None
    hamiltonian_units: str = EXPECTED_HAMILTONIAN_UNITS
    displacement_units: str = EXPECTED_DISPLACEMENT_UNITS
    derivative_units: str = EXPECTED_DERIVATIVE_UNITS
    atom_index_zero_based: int | None = None
    atom_index_one_based: int | None = None
    axis: str | None = None
    axis_index: int | None = None
    delta_ang: float | None = None
    material_compatibility_hash: str | None = None
    orbital_ordering_hash: str | None = None
    neighbor_list_hash: str | None = None
    sparsity_pattern_hash: str | None = None
    basis_hash: str | None = None
    pseudopotential_hash: str | None = None
    metadata_hash: str | None = None
    extra_hashes: dict[str, str] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.matrix_path is not None and not isinstance(self.matrix_path, Path):
            object.__setattr__(self, "matrix_path", Path(self.matrix_path))
        if self.matrix_shape is not None and not isinstance(self.matrix_shape, tuple):
            object.__setattr__(self, "matrix_shape", tuple(int(value) for value in self.matrix_shape))
        if self.matrix_sha256 is None:
            object.__setattr__(self, "matrix_sha256", file_sha256(self.matrix_path if isinstance(self.matrix_path, Path) else None))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["matrix_path"] = str(self.matrix_path) if self.matrix_path is not None else None
        return data


@dataclass(frozen=True)
class DerivativeStencil:
    metadata: DerivativeMetadata
    siesta_plus: DerivativeMatrixInput | None
    siesta_minus: DerivativeMatrixInput | None
    ml_plus: DerivativeMatrixInput | None
    ml_minus: DerivativeMatrixInput | None
    siesta_base: DerivativeMatrixInput | None = None
    ml_base: DerivativeMatrixInput | None = None
    base_structure_path: Path | str | None = None
    plus_structure_path: Path | str | None = None
    minus_structure_path: Path | str | None = None

    def __post_init__(self) -> None:
        for field_name in ("base_structure_path", "plus_structure_path", "minus_structure_path"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, field_name, Path(value))

    def matrix_inputs(self) -> dict[str, DerivativeMatrixInput | None]:
        return {
            "siesta_plus": self.siesta_plus,
            "siesta_minus": self.siesta_minus,
            "siesta_base": self.siesta_base,
            "ml_plus": self.ml_plus,
            "ml_minus": self.ml_minus,
            "ml_base": self.ml_base,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "siesta_plus": self.siesta_plus.to_dict() if self.siesta_plus else None,
            "siesta_minus": self.siesta_minus.to_dict() if self.siesta_minus else None,
            "siesta_base": self.siesta_base.to_dict() if self.siesta_base else None,
            "ml_plus": self.ml_plus.to_dict() if self.ml_plus else None,
            "ml_minus": self.ml_minus.to_dict() if self.ml_minus else None,
            "ml_base": self.ml_base.to_dict() if self.ml_base else None,
            "base_structure_path": str(self.base_structure_path) if self.base_structure_path else None,
            "plus_structure_path": str(self.plus_structure_path) if self.plus_structure_path else None,
            "minus_structure_path": str(self.minus_structure_path) if self.minus_structure_path else None,
        }


@dataclass(frozen=True)
class DerivativeMatrixResult:
    matrix: sparse.csr_matrix
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "matrix_shape": list(self.matrix.shape),
            "derivative_nnz": int(self.matrix.nnz),
        }


@dataclass(frozen=True)
class DerivativeComparisonResult:
    reference: DerivativeMatrixResult
    predicted: DerivativeMatrixResult
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference.to_dict(),
            "predicted": self.predicted.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class DerivativeSparseMetrics:
    rows: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.rows)


@dataclass(frozen=True)
class DerivativeStencilDiscovery:
    status: str
    method: str | None
    group_key: tuple[Any, ...]
    stencil: DerivativeStencil | None
    issues: tuple[DerivativeValidationIssue, ...] = ()
    sample_ids: tuple[str, ...] = ()
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "group_key": list(self.group_key),
            "stencil": self.stencil.to_dict() if self.stencil else None,
            "issues": [issue.to_dict() for issue in self.issues],
            "sample_ids": list(self.sample_ids),
            "details": dict(self.details),
        }


def validation_errors(issues: list[DerivativeValidationIssue]) -> list[DerivativeValidationIssue]:
    return [issue for issue in issues if issue.is_error]


def validation_warnings(issues: list[DerivativeValidationIssue]) -> list[DerivativeValidationIssue]:
    return [issue for issue in issues if not issue.is_error]


def stencil_is_valid(issues: list[DerivativeValidationIssue]) -> bool:
    return not validation_errors(issues)


def validate_derivative_geometry(
    discovery: DerivativeStencilDiscovery,
    *,
    tolerance_ang: float = DEFAULT_GEOMETRY_TOLERANCE_ANG,
) -> list[DerivativeValidationIssue]:
    """Validate that derivative stencil structures match the requested displacement."""

    issues: list[DerivativeValidationIssue] = []
    stencil = discovery.stencil
    if stencil is None:
        return [
            DerivativeValidationIssue(
                severity="error",
                code="missing_geometry_stencil",
                message="Derivative discovery did not produce a stencil to validate geometrically.",
                details={"group_key": list(discovery.group_key)},
            )
        ]
    metadata = stencil.metadata
    method = str(metadata.method or discovery.method or "").strip().lower()
    if method not in VALID_METHODS:
        issues.append(
            DerivativeValidationIssue(
                severity="error",
                code="invalid_geometry_method",
                message="Geometry validation requires a supported finite-difference method.",
                field="method",
                sample_id=metadata.sample_id,
                details={"method": method},
            )
        )
        return issues
    if metadata.delta_ang is None or metadata.delta_ang <= 0:
        issues.append(
            DerivativeValidationIssue(
                severity="error",
                code="invalid_geometry_delta",
                message="Geometry validation requires a positive delta_ang.",
                field="delta_ang",
                sample_id=metadata.sample_id,
            )
        )
        return issues
    if metadata.atom_index_zero_based is None:
        issues.append(
            DerivativeValidationIssue(
                severity="error",
                code="missing_geometry_atom_index",
                message="Geometry validation requires atom_index_zero_based.",
                field="atom_index_zero_based",
                sample_id=metadata.sample_id,
            )
        )
        return issues
    if metadata.axis_index is None or metadata.axis_index not in VALID_AXES.values():
        issues.append(
            DerivativeValidationIssue(
                severity="error",
                code="missing_geometry_axis_index",
                message="Geometry validation requires a valid axis_index.",
                field="axis_index",
                sample_id=metadata.sample_id,
            )
        )
        return issues

    structures = _load_geometry_structures(stencil)
    issues.extend(structures.pop("issues"))
    base = structures.get("base")
    if base is None:
        issues.append(
            DerivativeValidationIssue(
                severity="error",
                code="missing_base_structure",
                message="Geometry validation requires the base R0 structure for finite-displacement stencils.",
                field="base_structure_path",
                sample_id=metadata.sample_id,
            )
        )
        return issues

    roles = _required_geometry_roles(method)
    for role in roles:
        if structures.get(role) is None:
            issues.append(
                DerivativeValidationIssue(
                    severity="error",
                    code=f"missing_{role}_structure",
                    message=f"Geometry validation requires the {role} displaced structure.",
                    field=f"{role}_structure_path",
                    sample_id=metadata.sample_id,
                )
            )
    if validation_errors(issues):
        return issues

    for role in roles:
        structure = structures.get(role)
        if structure is None:
            continue
        issues.extend(_validate_structure_identity(base, structure, role=role, metadata=metadata, tolerance_ang=tolerance_ang))
    if validation_errors(issues):
        return issues

    if "plus" in roles and structures.get("plus") is not None:
        issues.extend(
            _validate_displacement(
                base,
                structures["plus"],
                role="plus",
                sign=1,
                metadata=metadata,
                tolerance_ang=tolerance_ang,
            )
        )
    if "minus" in roles and structures.get("minus") is not None:
        issues.extend(
            _validate_displacement(
                base,
                structures["minus"],
                role="minus",
                sign=-1,
                metadata=metadata,
                tolerance_ang=tolerance_ang,
            )
        )
    issues.extend(_validate_geometry_metadata_family(discovery))
    return issues


def finite_difference_derivative(
    *,
    method: str,
    delta_ang: float,
    plus: sparse.spmatrix | None = None,
    minus: sparse.spmatrix | None = None,
    base: sparse.spmatrix | None = None,
    source: str,
    matrix_hashes: dict[str, str | None] | None = None,
    hamiltonian_units: str = EXPECTED_HAMILTONIAN_UNITS,
    displacement_units: str = EXPECTED_DISPLACEMENT_UNITS,
    derivative_units: str = EXPECTED_DERIVATIVE_UNITS,
    validation_status: str = "valid",
    metadata: DerivativeMetadata | None = None,
) -> DerivativeMatrixResult:
    """Return dH/dR from already-loaded sparse Hamiltonian matrices."""

    method = str(method or "").strip().lower()
    source = str(source or "").strip().lower()
    if method not in VALID_METHODS:
        raise HamiltonianDerivativeError(f"Unsupported finite-difference method: {method!r}.")
    if source not in VALID_SOURCES:
        raise HamiltonianDerivativeError(f"Unsupported derivative source: {source!r}.")
    if delta_ang <= 0:
        raise HamiltonianDerivativeError("delta_ang must be positive.")
    if hamiltonian_units != EXPECTED_HAMILTONIAN_UNITS:
        raise HamiltonianDerivativeError(f"hamiltonian_units must be {EXPECTED_HAMILTONIAN_UNITS!r}.")
    if displacement_units != EXPECTED_DISPLACEMENT_UNITS:
        raise HamiltonianDerivativeError(f"displacement_units must be {EXPECTED_DISPLACEMENT_UNITS!r}.")
    if derivative_units != EXPECTED_DERIVATIVE_UNITS:
        raise HamiltonianDerivativeError(f"derivative_units must be {EXPECTED_DERIVATIVE_UNITS!r}.")

    left, right, denominator, operand_names = _finite_difference_operands(
        method=method,
        plus=plus,
        minus=minus,
        base=base,
        delta_ang=float(delta_ang),
    )
    _require_matching_shapes(operand_names, left, right)
    left_csr = _csr_copy(left)
    right_csr = _csr_copy(right)
    derivative = ((left_csr - right_csr) / denominator).tocsr()
    derivative.eliminate_zeros()

    finite_values = _sparse_finite_values(derivative)
    result_metadata = {
        "method": method,
        "delta_ang": float(delta_ang),
        "hamiltonian_units": hamiltonian_units,
        "displacement_units": displacement_units,
        "derivative_units": derivative_units,
        "source": source,
        "matrix_hashes": dict(matrix_hashes or {}),
        "validation_status": validation_status,
        "operand_roles": list(operand_names),
        "plus_minus_support_changed": sparse_support_changed(left_csr, right_csr),
        "derivative_nnz": int(derivative.nnz),
        "derivative_density": sparse_density(derivative),
        "finite_values": finite_values,
        "dH_hermiticity_defect": sparse_hermiticity_defect(derivative),
    }
    if metadata is not None:
        result_metadata.update(
            {
                "sample_id": metadata.sample_id,
                "base_sample_id": metadata.base_sample_id,
                "plus_sample_id": metadata.plus_sample_id,
                "minus_sample_id": metadata.minus_sample_id,
                "atom_index_zero_based": metadata.atom_index_zero_based,
                "atom_index_one_based": metadata.atom_index_one_based,
                "axis": metadata.axis,
                "axis_index": metadata.axis_index,
                "claim_status": metadata.claim_status,
            }
        )
    if not finite_values:
        result_metadata["validation_status"] = "invalid_nonfinite_derivative"
    return DerivativeMatrixResult(matrix=derivative, metadata=result_metadata)


def finite_difference_derivative_pair(
    *,
    method: str,
    delta_ang: float,
    reference_plus: sparse.spmatrix | None = None,
    reference_minus: sparse.spmatrix | None = None,
    reference_base: sparse.spmatrix | None = None,
    predicted_plus: sparse.spmatrix | None = None,
    predicted_minus: sparse.spmatrix | None = None,
    predicted_base: sparse.spmatrix | None = None,
    predicted_source: str = "graph2mat",
    reference_hashes: dict[str, str | None] | None = None,
    predicted_hashes: dict[str, str | None] | None = None,
    metadata: DerivativeMetadata | None = None,
) -> DerivativeComparisonResult:
    """Compute paired SIESTA and ML Hamiltonian derivatives plus diagnostics."""

    reference = finite_difference_derivative(
        method=method,
        delta_ang=delta_ang,
        plus=reference_plus,
        minus=reference_minus,
        base=reference_base,
        source="siesta",
        matrix_hashes=reference_hashes,
        metadata=metadata,
    )
    predicted = finite_difference_derivative(
        method=method,
        delta_ang=delta_ang,
        plus=predicted_plus,
        minus=predicted_minus,
        base=predicted_base,
        source=predicted_source,
        matrix_hashes=predicted_hashes,
        metadata=metadata,
    )
    _require_matching_shapes(("reference_derivative", "predicted_derivative"), reference.matrix, predicted.matrix)
    diagnostics = {
        "dH_ref_hermiticity_defect": reference.metadata["dH_hermiticity_defect"],
        "dH_pred_hermiticity_defect": predicted.metadata["dH_hermiticity_defect"],
        "plus_minus_support_changed": bool(
            reference.metadata["plus_minus_support_changed"]
            or predicted.metadata["plus_minus_support_changed"]
        ),
        "reference_plus_minus_support_changed": reference.metadata["plus_minus_support_changed"],
        "predicted_plus_minus_support_changed": predicted.metadata["plus_minus_support_changed"],
        "derivative_nnz": int((reference.matrix != 0).maximum(predicted.matrix != 0).nnz),
        "reference_derivative_nnz": reference.metadata["derivative_nnz"],
        "predicted_derivative_nnz": predicted.metadata["derivative_nnz"],
        "derivative_density": sparse_density((reference.matrix != 0).maximum(predicted.matrix != 0).tocsr()),
        "reference_derivative_density": reference.metadata["derivative_density"],
        "predicted_derivative_density": predicted.metadata["derivative_density"],
        "finite_values": bool(reference.metadata["finite_values"] and predicted.metadata["finite_values"]),
        "reference_validation_status": reference.metadata["validation_status"],
        "predicted_validation_status": predicted.metadata["validation_status"],
    }
    return DerivativeComparisonResult(reference=reference, predicted=predicted, diagnostics=diagnostics)


def derivative_sparse_metrics(
    reference: sparse.spmatrix,
    predicted: sparse.spmatrix,
    *,
    sample: str,
    metadata: DerivativeMetadata | None = None,
    source_model: str = "",
    reference_source: str = "siesta",
    support_threshold: float = DERIVATIVE_SUPPORT_THRESHOLD,
) -> dict[str, Any]:
    """Compare dH_pred/dR against dH_ref/dR on sparse derivative supports."""

    reference = reference.tocsr(copy=True)
    predicted = predicted.tocsr(copy=True)
    _require_matching_shapes(("reference_derivative", "predicted_derivative"), reference, predicted)
    ref_values = sparse_value_dict(reference, threshold=support_threshold)
    pred_values = sparse_value_dict(predicted, threshold=support_threshold)
    ref_support = set(ref_values)
    pred_support = set(pred_values)
    union_support = ref_support | pred_support
    intersection = ref_support & pred_support
    ref_errors = _errors_on_support(ref_values, pred_values, ref_support)
    pred_errors = _errors_on_support(ref_values, pred_values, pred_support)
    union_errors = _errors_on_support(ref_values, pred_values, union_support)
    ref_norm = sparse_frobenius_norm(reference)
    ref_error_norm = _frobenius_from_values(ref_errors)
    ref_union_norm = _frobenius_from_values([ref_values.get(index, 0.0) for index in union_support])
    union_error_norm = _frobenius_from_values(union_errors)
    ref_l1_union = _l1_from_values([ref_values.get(index, 0.0) for index in union_support])
    error_l1_union = _l1_from_values(union_errors)
    zero_ref_reason = "reference_derivative_norm_zero" if ref_norm == 0.0 else ""
    cosine, cosine_reason = _cosine_similarity_from_values(ref_values, pred_values, union_support)
    metadata_row = _derivative_metric_metadata(
        sample=sample,
        metadata=metadata,
        source_model=source_model,
        reference_source=reference_source,
    )
    row = {
        **metadata_row,
        "support_threshold": support_threshold,
        "dh_ref_nnz": len(ref_support),
        "dh_pred_nnz": len(pred_support),
        "dh_union_nnz": len(union_support),
        "dh_ref_density": sparse_density(reference),
        "dh_pred_density": sparse_density(predicted),
        "dh_union_density": len(union_support) / (reference.shape[0] * reference.shape[1])
        if reference.shape[0] and reference.shape[1]
        else math.nan,
        "dh_mae_ref_eV_per_Ang": _mean_abs(ref_errors),
        "dh_rmse_ref_eV_per_Ang": _rmse(ref_errors),
        "dh_mse_ref_eV2_per_Ang2": _mse(ref_errors),
        "dh_mae_pred_eV_per_Ang": _mean_abs(pred_errors),
        "dh_rmse_pred_eV_per_Ang": _rmse(pred_errors),
        "dh_mae_union_eV_per_Ang": _mean_abs(union_errors),
        "dh_rmse_union_eV_per_Ang": _rmse(union_errors),
        "dh_max_abs_error_union_eV_per_Ang": _max_abs(union_errors),
        "dh_relative_frobenius_ref": ref_error_norm / ref_norm if ref_norm else math.nan,
        "dh_relative_frobenius_union": union_error_norm / ref_union_norm if ref_union_norm else math.nan,
        "dh_relative_l1_union": error_l1_union / ref_l1_union if ref_l1_union else math.nan,
        "dh_cosine_similarity_union": cosine,
        "dh_support_precision": len(intersection) / len(pred_support) if pred_support else math.nan,
        "dh_support_recall": len(intersection) / len(ref_support) if ref_support else math.nan,
        "dh_support_f1": _f1(len(intersection), len(pred_support), len(ref_support)),
        "dh_false_zero_rate": len(ref_support - pred_support) / len(ref_support) if ref_support else math.nan,
        "dh_false_nonzero_rate": len(pred_support - ref_support) / len(pred_support) if pred_support else math.nan,
        "dh_hermiticity_ref": sparse_hermiticity_defect(reference),
        "dh_hermiticity_pred": sparse_hermiticity_defect(predicted),
        "dh_hermiticity_error_delta": abs(sparse_hermiticity_defect(predicted) - sparse_hermiticity_defect(reference)),
        "dh_finite_values": bool(_sparse_finite_values(reference) and _sparse_finite_values(predicted)),
        "dh_relative_unavailable_reason": zero_ref_reason,
        "dh_cosine_unavailable_reason": cosine_reason,
    }
    if ref_union_norm == 0.0 and not row["dh_relative_unavailable_reason"]:
        row["dh_relative_unavailable_reason"] = "reference_derivative_union_norm_zero"
    if ref_l1_union == 0.0 and not row["dh_relative_unavailable_reason"]:
        row["dh_relative_unavailable_reason"] = "reference_derivative_l1_norm_zero"
    return row


def _required_geometry_roles(method: str) -> tuple[str, ...]:
    if method == "central":
        return ("plus", "minus")
    if method == "forward":
        return ("plus",)
    if method == "backward":
        return ("minus",)
    return ()


def _load_geometry_structures(stencil: DerivativeStencil) -> dict[str, Any]:
    structures: dict[str, Any] = {"issues": []}
    for role, path in (
        ("base", stencil.base_structure_path),
        ("plus", stencil.plus_structure_path),
        ("minus", stencil.minus_structure_path),
    ):
        if path is None:
            structures[role] = None
            continue
        try:
            structures[role] = extract_fdf_structure(Path(path))
        except Exception as exc:
            structures[role] = None
            structures["issues"].append(
                DerivativeValidationIssue(
                    severity="error",
                    code=f"unreadable_{role}_structure",
                    message=f"Could not read {role} RUN.fdf for derivative geometry validation: {exc}",
                    field=f"{role}_structure_path",
                    sample_id=stencil.metadata.sample_id,
                    details={"path": str(path)},
                )
            )
    return structures


def _validate_structure_identity(
    base: Any,
    structure: Any,
    *,
    role: str,
    metadata: DerivativeMetadata,
    tolerance_ang: float,
) -> list[DerivativeValidationIssue]:
    issues: list[DerivativeValidationIssue] = []
    if base.atom_count != structure.atom_count:
        issues.append(
            DerivativeValidationIssue(
                severity="error",
                code="atom_count_mismatch",
                message=f"{role} structure atom count differs from base R0.",
                sample_id=metadata.sample_id,
                details={"role": role, "base_atom_count": base.atom_count, "role_atom_count": structure.atom_count},
            )
        )
        return issues
    if base.atom_species != structure.atom_species:
        issues.append(
            DerivativeValidationIssue(
                severity="error",
                code="atom_ordering_or_species_mismatch",
                message=f"{role} structure species/order differs from base R0.",
                sample_id=metadata.sample_id,
                details={"role": role},
            )
        )
    if [species.to_dict() for species in base.species] != [species.to_dict() for species in structure.species]:
        issues.append(
            DerivativeValidationIssue(
                severity="error",
                code="species_metadata_mismatch",
                message=f"{role} ChemicalSpeciesLabel metadata differs from base R0.",
                sample_id=metadata.sample_id,
                details={"role": role},
            )
        )
    if len(base.lattice_vectors_ang) != len(structure.lattice_vectors_ang):
        issues.append(
            DerivativeValidationIssue(
                severity="error",
                code="cell_mismatch",
                message=f"{role} lattice vector count differs from base R0.",
                sample_id=metadata.sample_id,
                details={"role": role},
            )
        )
        return issues
    for vector_index, (base_vector, role_vector) in enumerate(zip(base.lattice_vectors_ang, structure.lattice_vectors_ang)):
        for component_index, (base_value, role_value) in enumerate(zip(base_vector, role_vector)):
            if abs(float(role_value) - float(base_value)) > tolerance_ang:
                issues.append(
                    DerivativeValidationIssue(
                        severity="error",
                        code="cell_mismatch",
                        message=f"{role} cell differs from base R0.",
                        sample_id=metadata.sample_id,
                        details={
                            "role": role,
                            "vector_index": vector_index,
                            "component_index": component_index,
                            "base_value_ang": float(base_value),
                            "role_value_ang": float(role_value),
                            "tolerance_ang": tolerance_ang,
                        },
                    )
                )
                return issues
    return issues


def _validate_displacement(
    base: Any,
    structure: Any,
    *,
    role: str,
    sign: int,
    metadata: DerivativeMetadata,
    tolerance_ang: float,
) -> list[DerivativeValidationIssue]:
    issues: list[DerivativeValidationIssue] = []
    target_atom = int(metadata.atom_index_zero_based)
    axis_index = int(metadata.axis_index)
    delta_ang = float(metadata.delta_ang)
    if target_atom < 0 or target_atom >= base.atom_count:
        return [
            DerivativeValidationIssue(
                severity="error",
                code="atom_index_out_of_range",
                message="Requested derivative atom index is outside the base R0 structure.",
                field="atom_index_zero_based",
                sample_id=metadata.sample_id,
                details={"atom_index_zero_based": target_atom, "atom_count": base.atom_count},
            )
        ]
    for atom_index, (base_position, role_position) in enumerate(zip(base.positions_ang, structure.positions_ang)):
        for component_index, (base_value, role_value) in enumerate(zip(base_position, role_position)):
            expected = sign * delta_ang if atom_index == target_atom and component_index == axis_index else 0.0
            actual = float(role_value) - float(base_value)
            if abs(actual - expected) <= tolerance_ang:
                continue
            code = "displacement_component_mismatch" if atom_index == target_atom and component_index == axis_index else "unexpected_coordinate_drift"
            issues.append(
                DerivativeValidationIssue(
                    severity="error",
                    code=code,
                    message=f"{role} displacement does not match the requested finite-displacement stencil.",
                    sample_id=metadata.sample_id,
                    details={
                        "role": role,
                        "atom_index_zero_based": atom_index,
                        "component_index": component_index,
                        "expected_delta_ang": expected,
                        "actual_delta_ang": actual,
                        "tolerance_ang": tolerance_ang,
                    },
                )
            )
    return issues


def _validate_geometry_metadata_family(discovery: DerivativeStencilDiscovery) -> list[DerivativeValidationIssue]:
    issues: list[DerivativeValidationIssue] = []
    metadata_by_role = discovery.details.get("sample_metadata_by_role")
    if not isinstance(metadata_by_role, dict):
        return issues
    splits = {
        role: _metadata_text(metadata, "split", "split_name", "dataset_split")
        for role, metadata in metadata_by_role.items()
        if isinstance(metadata, dict)
    }
    nonempty_splits = {role: split for role, split in splits.items() if split}
    if len(set(nonempty_splits.values())) > 1:
        issues.append(
            DerivativeValidationIssue(
                severity="error",
                code="split_mismatch",
                message="Derivative stencil operands belong to different dataset splits.",
                field="split",
                details={"splits": nonempty_splits},
            )
        )
    families = {
        role: _metadata_text(metadata, "base_sample_id", "reference_sample_id", "reference_group_id")
        for role, metadata in metadata_by_role.items()
        if isinstance(metadata, dict)
    }
    nonempty_families = {role: family for role, family in families.items() if family}
    if len(set(nonempty_families.values())) > 1:
        issues.append(
            DerivativeValidationIssue(
                severity="error",
                code="family_mismatch",
                message="Derivative stencil operands belong to different base/family metadata groups.",
                details={"families": nonempty_families},
            )
        )
    for field_name in ("basis_hash", "pseudopotential_hash", "orbital_ordering_hash", "material_compatibility_hash"):
        values = {
            role: str(metadata[field_name])
            for role, metadata in metadata_by_role.items()
            if isinstance(metadata, dict) and str(metadata.get(field_name) or "").strip()
        }
        if len(set(values.values())) > 1:
            issues.append(
                DerivativeValidationIssue(
                    severity="error",
                    code="geometry_metadata_hash_mismatch",
                    message=f"{field_name} differs across derivative stencil structures.",
                    field=field_name,
                    details={"values": values},
                )
            )
    return issues


def discover_derivative_stencils(
    result_dir: Path | str,
    *,
    method: str | None = None,
    split: str = "all",
    finite_difference_method: str | None = None,
    require_central: bool = False,
) -> list[DerivativeStencilDiscovery]:
    """Group existing result directories into finite-difference derivative stencils.

    The expected layout is the staged comparison layout:
    structures/<sample>/metadata.json, siesta_hamiltonians/<sample>/*.HSX|*.TSHS,
    and predicted_hamiltonians/<sample>/ML_prediction.HSX.
    """

    result_dir = Path(result_dir)
    source_model = _normalize_source_model(method)
    requested_method = _normalize_discovery_method(finite_difference_method)
    structures_root = result_dir / "structures"
    if not structures_root.exists():
        return [
            DerivativeStencilDiscovery(
                status="incomplete",
                method=requested_method,
                group_key=("missing_structures_root", str(structures_root)),
                stencil=None,
                issues=(
                    DerivativeValidationIssue(
                        severity="error",
                        code="missing_structures_root",
                        message="Derivative discovery requires result-dir/structures.",
                        field="result_dir",
                    ),
                ),
                details={"result_dir": str(result_dir)},
            )
        ]

    samples = [
        sample
        for sample in (_discover_sample(sample_dir, result_dir=result_dir, source_model=source_model) for sample_dir in sorted(structures_root.iterdir()))
        if sample is not None
    ]
    discoveries: list[DerivativeStencilDiscovery] = []
    selected_samples: list[_DiscoveredDerivativeSample] = []
    requested_split = str(split or "all").strip().lower()
    for sample in samples:
        split_issue = _sample_split_issue(sample, requested_split)
        if split_issue is not None:
            discoveries.append(_split_filtered_discovery_for_sample(sample, requested_split, split_issue))
            continue
        if _sample_in_split(sample, requested_split):
            selected_samples.append(sample)
    samples = selected_samples
    base_samples = [sample for sample in samples if sample.is_base]
    displaced_samples = [sample for sample in samples if not sample.is_base]
    ungroupable = [sample for sample in displaced_samples if not sample.can_group]
    for sample in ungroupable:
        discoveries.append(_incomplete_discovery_for_sample(sample, requested_method))

    groups: dict[tuple[Any, ...], dict[int, list[_DiscoveredDerivativeSample]]] = {}
    for sample in displaced_samples:
        if not sample.can_group:
            continue
        groups.setdefault(sample.group_key, {}).setdefault(int(sample.sign or 0), []).append(sample)

    for group_key, by_sign in sorted(groups.items(), key=lambda item: str(item[0])):
        plus_samples = by_sign.get(1, [])
        minus_samples = by_sign.get(-1, [])
        if len(plus_samples) > 1 or len(minus_samples) > 1:
            discoveries.append(_ambiguous_discovery(group_key, plus_samples, minus_samples, requested_method))
            continue

        plus = plus_samples[0] if plus_samples else None
        minus = minus_samples[0] if minus_samples else None
        base_matches = _matching_base_samples(group_key, base_samples, plus=plus, minus=minus)
        base_match = base_matches[0] if len(base_matches) == 1 else None
        if _base_ambiguity_blocks_discovery(
            requested_method=requested_method,
            require_central=require_central,
            plus=plus,
            minus=minus,
            base_matches=base_matches,
        ):
            discoveries.append(_ambiguous_base_discovery(group_key, plus, minus, base_matches, requested_method))
            continue
        methods = _methods_to_emit(
            requested_method=requested_method,
            require_central=require_central,
            plus=plus,
            minus=minus,
            base=base_match,
        )
        if not methods:
            discoveries.append(_incomplete_discovery_for_group(group_key, plus, minus, base_match, requested_method or "central"))
            continue
        for method_name in methods:
            discoveries.append(_build_discovered_stencil(group_key, method_name, plus, minus, base_match, source_model))
    return discoveries


@dataclass(frozen=True)
class _DiscoveredDerivativeSample:
    sample_id: str
    structure_dir: Path
    structure_path: Path | None
    metadata_path: Path
    metadata: dict[str, Any] | None
    metadata_issue: DerivativeValidationIssue | None
    is_base: bool
    sign: int | None
    group_key: tuple[Any, ...]
    base_key: tuple[Any, ...]
    material_label: str | None
    atom_index_zero_based: int | None
    atom_index_one_based: int | None
    axis: str | None
    axis_index: int | None
    delta_ang: float | None
    siesta_matrix: DerivativeMatrixInput | None
    ml_matrix: DerivativeMatrixInput | None
    siesta_issue: DerivativeValidationIssue | None
    ml_issue: DerivativeValidationIssue | None

    @property
    def can_group(self) -> bool:
        return (
            self.metadata is not None
            and self.sign in {-1, 1}
            and self.atom_index_zero_based is not None
            and self.axis in VALID_AXES
            and self.delta_ang is not None
            and self.delta_ang > 0
            and self.material_label is not None
        )


def _discover_sample(sample_dir: Path, *, result_dir: Path, source_model: str) -> _DiscoveredDerivativeSample | None:
    if not sample_dir.is_dir():
        return None
    sample_id = sample_dir.name
    metadata_path = sample_dir / "metadata.json"
    metadata, metadata_issue = _read_derivative_metadata(metadata_path, sample_id)
    if metadata:
        sample_id = str(metadata.get("sample_id") or metadata.get("id") or sample_id)
    is_base = _is_base_metadata(metadata)
    sign = _metadata_sign(metadata)
    axis = _metadata_axis(metadata)
    axis_index = _metadata_axis_index(metadata, axis)
    atom_zero = _metadata_int(metadata, "atom_index_zero_based")
    atom_one = _metadata_int(metadata, "atom_index")
    delta_ang = _metadata_delta_ang(metadata, axis_index)
    material_label = _metadata_text(metadata, "material_label", "material_id", "base_material_label")
    system_label = _metadata_system_group_label(metadata, sample_id)
    split_group_id = _metadata_text(metadata, "split_group_id")
    base_group = _metadata_text(metadata, "base_sample_id", "reference_sample_id", "reference_group_id") or split_group_id
    matrix_shape = _metadata_shape(metadata)
    hash_values = _metadata_hashes(metadata)
    group_key = (
        base_group,
        material_label,
        system_label,
        atom_zero,
        axis,
        _rounded_delta(delta_ang),
        split_group_id,
        hash_values.get("material_compatibility_hash"),
        hash_values.get("basis_hash"),
        hash_values.get("pseudopotential_hash"),
        hash_values.get("orbital_ordering_hash"),
        hash_values.get("neighbor_list_hash"),
        hash_values.get("sparsity_pattern_hash"),
    )
    base_key = (
        material_label,
        system_label,
        hash_values.get("material_compatibility_hash"),
        hash_values.get("basis_hash"),
        hash_values.get("pseudopotential_hash"),
        hash_values.get("orbital_ordering_hash"),
        hash_values.get("neighbor_list_hash"),
        hash_values.get("sparsity_pattern_hash"),
    )
    structure_path = sample_dir / "RUN.fdf" if (sample_dir / "RUN.fdf").exists() else None
    siesta_matrix, siesta_issue = _discover_siesta_matrix(
        result_dir / "siesta_hamiltonians" / sample_id,
        sample_id=sample_id,
        metadata=metadata,
        matrix_shape=matrix_shape,
        hash_values=hash_values,
        atom_zero=atom_zero,
        atom_one=atom_one,
        axis=axis,
        axis_index=axis_index,
        delta_ang=delta_ang,
    )
    ml_matrix, ml_issue = _discover_prediction_matrix(
        result_dir / "predicted_hamiltonians" / sample_id,
        sample_id=sample_id,
        source_model=source_model,
        metadata=metadata,
        matrix_shape=matrix_shape,
        hash_values=hash_values,
        atom_zero=atom_zero,
        atom_one=atom_one,
        axis=axis,
        axis_index=axis_index,
        delta_ang=delta_ang,
    )
    return _DiscoveredDerivativeSample(
        sample_id=sample_id,
        structure_dir=sample_dir,
        structure_path=structure_path,
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_issue=metadata_issue,
        is_base=is_base,
        sign=sign,
        group_key=group_key,
        base_key=base_key,
        material_label=material_label,
        atom_index_zero_based=atom_zero,
        atom_index_one_based=atom_one,
        axis=axis,
        axis_index=axis_index,
        delta_ang=delta_ang,
        siesta_matrix=siesta_matrix,
        ml_matrix=ml_matrix,
        siesta_issue=siesta_issue,
        ml_issue=ml_issue,
    )


def _build_discovered_stencil(
    group_key: tuple[Any, ...],
    method: str,
    plus: _DiscoveredDerivativeSample | None,
    minus: _DiscoveredDerivativeSample | None,
    base: _DiscoveredDerivativeSample | None,
    source_model: str,
) -> DerivativeStencilDiscovery:
    representative = plus or minus or base
    if representative is None:
        return _incomplete_discovery_for_group(group_key, plus, minus, base, method)
    metadata = DerivativeMetadata(
        sample_id=_stencil_sample_id(method, plus, minus, base),
        base_sample_id=base.sample_id if base else None,
        plus_sample_id=plus.sample_id if plus else None,
        minus_sample_id=minus.sample_id if minus else None,
        atom_index_zero_based=representative.atom_index_zero_based,
        atom_index_one_based=representative.atom_index_one_based,
        axis=representative.axis,
        axis_index=representative.axis_index,
        delta_ang=representative.delta_ang,
        method=method,
        claim_status=_metadata_text(representative.metadata, "claim_status", "comparison_status") or "diagnostic_only",
        material_compatibility_hash=_group_hash(group_key, 7),
        basis_hash=_group_hash(group_key, 8),
        pseudopotential_hash=_group_hash(group_key, 9),
        orbital_ordering_hash=_group_hash(group_key, 10),
        neighbor_list_hash=_group_hash(group_key, 11),
        sparsity_pattern_hash=_group_hash(group_key, 12),
    )
    stencil = DerivativeStencil(
        metadata=metadata,
        siesta_plus=plus.siesta_matrix if plus else None,
        siesta_minus=minus.siesta_matrix if minus else None,
        siesta_base=base.siesta_matrix if base else None,
        ml_plus=plus.ml_matrix if plus else None,
        ml_minus=minus.ml_matrix if minus else None,
        ml_base=base.ml_matrix if base else None,
        base_structure_path=base.structure_path if base else None,
        plus_structure_path=plus.structure_path if plus else None,
        minus_structure_path=minus.structure_path if minus else None,
    )
    issues = list(validate_derivative_stencil(stencil))
    issues.extend(_sample_matrix_issues(plus, minus, base))
    status = _discovery_status(issues)
    return DerivativeStencilDiscovery(
        status=status,
        method=method,
        group_key=group_key,
        stencil=stencil,
        issues=tuple(issues),
        sample_ids=tuple(sample.sample_id for sample in (base, plus, minus) if sample is not None),
        details={
            "source_model": source_model,
            "sample_metadata_by_role": {
                role: sample.metadata
                for role, sample in (("base", base), ("plus", plus), ("minus", minus))
                if sample is not None and sample.metadata is not None
            },
        },
    )


def _discover_siesta_matrix(
    sample_dir: Path,
    *,
    sample_id: str,
    metadata: dict[str, Any] | None,
    matrix_shape: tuple[int, int] | None,
    hash_values: dict[str, str],
    atom_zero: int | None,
    atom_one: int | None,
    axis: str | None,
    axis_index: int | None,
    delta_ang: float | None,
) -> tuple[DerivativeMatrixInput | None, DerivativeValidationIssue | None]:
    forbidden = sorted(path.name for path in sample_dir.glob("ML_prediction.HSX")) if sample_dir.exists() else []
    selection = choose_reference_matrix(sample_dir)
    if not selection.ok:
        code = "forbidden_siesta_reference" if forbidden else selection.reason
        message = "ML_prediction.HSX cannot be used as a SIESTA derivative reference." if forbidden else selection.reason
        return None, DerivativeValidationIssue(
            severity="error",
            code=code,
            message=message,
            sample_id=sample_id,
            matrix_role="siesta",
            details={"candidates": list(selection.candidates), "forbidden": forbidden},
        )
    return _matrix_input_from_path(
        sample_id=sample_id,
        source="siesta",
        path=selection.path,
        metadata=metadata,
        matrix_shape=matrix_shape,
        hash_values=hash_values,
        atom_zero=atom_zero,
        atom_one=atom_one,
        axis=axis,
        axis_index=axis_index,
        delta_ang=delta_ang,
    ), None


def _discover_prediction_matrix(
    sample_dir: Path,
    *,
    sample_id: str,
    source_model: str,
    metadata: dict[str, Any] | None,
    matrix_shape: tuple[int, int] | None,
    hash_values: dict[str, str],
    atom_zero: int | None,
    atom_one: int | None,
    axis: str | None,
    axis_index: int | None,
    delta_ang: float | None,
) -> tuple[DerivativeMatrixInput | None, DerivativeValidationIssue | None]:
    prediction = sample_dir / "ML_prediction.HSX"
    if not prediction.exists() or not prediction.is_file():
        return None, DerivativeValidationIssue(
            severity="error",
            code="missing_prediction_matrix",
            message="Missing ML_prediction.HSX for derivative prediction.",
            sample_id=sample_id,
            matrix_role="ml",
            details={"prediction_dir": str(sample_dir)},
        )
    return _matrix_input_from_path(
        sample_id=sample_id,
        source=source_model,
        path=prediction,
        metadata=metadata,
        matrix_shape=matrix_shape,
        hash_values=hash_values,
        atom_zero=atom_zero,
        atom_one=atom_one,
        axis=axis,
        axis_index=axis_index,
        delta_ang=delta_ang,
    ), None


def _matrix_input_from_path(
    *,
    sample_id: str,
    source: str,
    path: Path | None,
    metadata: dict[str, Any] | None,
    matrix_shape: tuple[int, int] | None,
    hash_values: dict[str, str],
    atom_zero: int | None,
    atom_one: int | None,
    axis: str | None,
    axis_index: int | None,
    delta_ang: float | None,
) -> DerivativeMatrixInput:
    return DerivativeMatrixInput(
        sample_id=sample_id,
        source=source,
        matrix_path=path,
        matrix_shape=matrix_shape,
        hamiltonian_units=str((metadata or {}).get("hamiltonian_units") or EXPECTED_HAMILTONIAN_UNITS),
        displacement_units=str((metadata or {}).get("displacement_units") or EXPECTED_DISPLACEMENT_UNITS),
        derivative_units=str((metadata or {}).get("derivative_units") or EXPECTED_DERIVATIVE_UNITS),
        atom_index_zero_based=atom_zero,
        atom_index_one_based=atom_one,
        axis=axis,
        axis_index=axis_index,
        delta_ang=delta_ang,
        material_compatibility_hash=hash_values.get("material_compatibility_hash"),
        orbital_ordering_hash=hash_values.get("orbital_ordering_hash"),
        neighbor_list_hash=hash_values.get("neighbor_list_hash"),
        sparsity_pattern_hash=hash_values.get("sparsity_pattern_hash"),
        basis_hash=hash_values.get("basis_hash"),
        pseudopotential_hash=hash_values.get("pseudopotential_hash"),
        metadata_hash=file_sha256((Path(str((metadata or {}).get("metadata_path"))) if (metadata or {}).get("metadata_path") else None)),
    )


def _read_derivative_metadata(path: Path, sample_id: str) -> tuple[dict[str, Any] | None, DerivativeValidationIssue | None]:
    if not path.exists():
        return None, DerivativeValidationIssue(
            severity="warning",
            code="missing_metadata",
            message="Missing metadata.json; derivative discovery will not infer stencil pairing from filenames.",
            sample_id=sample_id,
            field="metadata.json",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, DerivativeValidationIssue(
            severity="warning",
            code="unreadable_metadata",
            message=f"Could not read metadata.json: {exc}",
            sample_id=sample_id,
            field="metadata.json",
        )
    except json.JSONDecodeError as exc:
        return None, DerivativeValidationIssue(
            severity="warning",
            code="invalid_metadata_json",
            message=f"Could not parse metadata.json: {exc}",
            sample_id=sample_id,
            field="metadata.json",
        )
    if not isinstance(payload, dict):
        return None, DerivativeValidationIssue(
            severity="warning",
            code="invalid_metadata_json",
            message="metadata.json must contain a JSON object.",
            sample_id=sample_id,
            field="metadata.json",
        )
    payload = dict(payload)
    payload["metadata_path"] = str(path)
    return payload, None


def _normalize_source_model(method: str | None) -> str:
    source = str(method or "graph2mat").strip().lower()
    return source if source in {"graph2mat", "deeph"} else source


def _normalize_discovery_method(method: str | None) -> str | None:
    if method is None or str(method).strip().lower() in {"", "all"}:
        return None
    method = str(method).strip().lower()
    if method not in VALID_METHODS:
        raise HamiltonianDerivativeError(f"Unsupported finite-difference discovery method: {method!r}.")
    return method


def _sample_in_split(sample: _DiscoveredDerivativeSample, split: str) -> bool:
    split = str(split or "all").strip().lower()
    if split == "all":
        return True
    metadata_split = _metadata_text(sample.metadata, "split", "split_name", "dataset_split")
    return metadata_split is not None and metadata_split.lower() == split


def _sample_split_issue(sample: _DiscoveredDerivativeSample, split: str) -> DerivativeValidationIssue | None:
    split = str(split or "all").strip().lower()
    if split == "all":
        return None
    metadata_split = _metadata_text(sample.metadata, "split", "split_name", "dataset_split")
    if metadata_split is not None:
        return None
    return DerivativeValidationIssue(
        severity="error",
        code="missing_split_metadata",
        message=(
            "Requested split-specific derivative discovery requires metadata split information; "
            "samples without split metadata are excluded fail-closed."
        ),
        field="split",
        sample_id=sample.sample_id,
        details={"requested_split": split, "metadata_path": str(sample.metadata_path)},
    )


def _is_base_metadata(metadata: dict[str, Any] | None) -> bool:
    if not metadata:
        return False
    if bool(metadata.get("is_reference")):
        return True
    sign = _metadata_sign(metadata)
    amplitude = _metadata_delta_ang(metadata, _metadata_axis_index(metadata, _metadata_axis(metadata)))
    return sign == 0 or amplitude == 0.0


def _metadata_sign(metadata: dict[str, Any] | None) -> int | None:
    if not metadata:
        return None
    value = metadata.get("sign")
    if isinstance(value, (int, float)) and int(value) in {-1, 0, 1}:
        return int(value)
    label = str(metadata.get("sign_label") or "").strip()
    if label == "+":
        return 1
    if label == "-":
        return -1
    return None


def _metadata_axis(metadata: dict[str, Any] | None) -> str | None:
    value = _metadata_text(metadata, "axis")
    if value is None:
        return None
    value = value.lower()
    return value if value in VALID_AXES else value


def _metadata_axis_index(metadata: dict[str, Any] | None, axis: str | None = None) -> int | None:
    value = _metadata_int(metadata, "axis_index")
    if value is not None:
        return value
    if axis in VALID_AXES:
        return VALID_AXES[axis]
    return None


def _metadata_delta_ang(metadata: dict[str, Any] | None, axis_index: int | None = None) -> float | None:
    value = _metadata_float(metadata, "amplitude_ang", "delta_ang")
    if value is not None:
        return abs(value)
    displacement = (metadata or {}).get("displacement_ang")
    if isinstance(displacement, list) and axis_index is not None and 0 <= axis_index < len(displacement):
        try:
            return abs(float(displacement[axis_index]))
        except (TypeError, ValueError):
            return None
    return None


def _metadata_shape(metadata: dict[str, Any] | None) -> tuple[int, int] | None:
    for key in ("matrix_shape", "hamiltonian_shape", "sparse_shape"):
        value = (metadata or {}).get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return (int(value[0]), int(value[1]))
            except (TypeError, ValueError):
                return None
    return None


def _metadata_hashes(metadata: dict[str, Any] | None) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for field_name in COMPARABILITY_HASH_FIELDS:
        value = _metadata_text(metadata, field_name)
        if value is not None:
            hashes[field_name] = value
    return hashes


def _metadata_text(metadata: dict[str, Any] | None, *keys: str) -> str | None:
    for key in keys:
        value = (metadata or {}).get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _metadata_system_group_label(metadata: dict[str, Any] | None, sample_id: str) -> str | None:
    stable = _metadata_text(
        metadata,
        "base_system_label",
        "reference_system_label",
        "system_group_label",
        "material_system_label",
    )
    if stable is not None:
        return stable
    system_label = _metadata_text(metadata, "system_label")
    if system_label is not None and system_label != sample_id:
        return system_label
    return None


def _metadata_int(metadata: dict[str, Any] | None, *keys: str) -> int | None:
    for key in keys:
        value = (metadata or {}).get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _metadata_float(metadata: dict[str, Any] | None, *keys: str) -> float | None:
    for key in keys:
        value = (metadata or {}).get(key)
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return None


def _rounded_delta(delta_ang: float | None) -> float | None:
    return round(float(delta_ang), 12) if delta_ang is not None else None


def _matching_base_samples(
    group_key: tuple[Any, ...],
    bases: list[_DiscoveredDerivativeSample],
    *,
    plus: _DiscoveredDerivativeSample | None,
    minus: _DiscoveredDerivativeSample | None,
) -> list[_DiscoveredDerivativeSample]:
    requested_ids = {
        base_id
        for sample in (plus, minus)
        for base_id in _requested_base_identity(sample)
    }
    if requested_ids:
        return [sample for sample in bases if requested_ids.intersection(_base_sample_identity(sample))]
    group_base_key = (group_key[1], group_key[2], group_key[7], group_key[8], group_key[9], group_key[10], group_key[11], group_key[12])
    return [sample for sample in bases if sample.base_key == group_base_key]


def _requested_base_identity(sample: _DiscoveredDerivativeSample | None) -> set[str]:
    if sample is None:
        return set()
    return _metadata_identity_values(sample.metadata, "base_sample_id", "source_base_sample_id", "reference_sample_id")


def _base_sample_identity(sample: _DiscoveredDerivativeSample) -> set[str]:
    identities = set(
        _metadata_identity_values(sample.metadata, "sample_id", "id")
    )
    identities.update(
        _metadata_identity_values(sample.metadata, "base_sample_id", "source_base_sample_id", "reference_sample_id")
    )
    identities.add(sample.sample_id)
    return {identity for identity in identities if identity}


def _metadata_identity_values(metadata: dict[str, Any] | None, *keys: str) -> set[str]:
    return {
        str((metadata or {}).get(key)).strip()
        for key in keys
        if str((metadata or {}).get(key) or "").strip()
    }


def _base_ambiguity_blocks_discovery(
    *,
    requested_method: str | None,
    require_central: bool,
    plus: _DiscoveredDerivativeSample | None,
    minus: _DiscoveredDerivativeSample | None,
    base_matches: list[_DiscoveredDerivativeSample],
) -> bool:
    if len(base_matches) <= 1:
        return False
    if requested_method in {"forward", "backward"}:
        return True
    if requested_method == "central" or require_central:
        return False
    if plus is not None and minus is not None:
        return False
    return (plus is not None) or (minus is not None)


def _methods_to_emit(
    *,
    requested_method: str | None,
    require_central: bool,
    plus: _DiscoveredDerivativeSample | None,
    minus: _DiscoveredDerivativeSample | None,
    base: _DiscoveredDerivativeSample | None,
) -> list[str]:
    if requested_method is not None:
        return [requested_method] if _method_available(requested_method, plus, minus, base) else []
    if require_central:
        return ["central"] if plus and minus else []
    if plus and minus:
        return ["central"]
    methods: list[str] = []
    if base and plus:
        methods.append("forward")
    if base and minus:
        methods.append("backward")
    return methods


def _method_available(
    method: str,
    plus: _DiscoveredDerivativeSample | None,
    minus: _DiscoveredDerivativeSample | None,
    base: _DiscoveredDerivativeSample | None,
) -> bool:
    if method == "central":
        return plus is not None and minus is not None
    if method == "forward":
        return plus is not None and base is not None
    if method == "backward":
        return minus is not None and base is not None
    return False


def _sample_matrix_issues(*samples: _DiscoveredDerivativeSample | None) -> list[DerivativeValidationIssue]:
    issues: list[DerivativeValidationIssue] = []
    for sample in samples:
        if sample is None:
            continue
        for issue in (sample.metadata_issue, sample.siesta_issue, sample.ml_issue):
            if issue is not None:
                issues.append(issue)
    return issues


def _discovery_status(issues: list[DerivativeValidationIssue]) -> str:
    codes = {issue.code for issue in issues}
    if "ambiguous_derivative_pairing" in codes:
        return "ambiguous"
    if any(issue.is_error for issue in issues):
        if any(code.startswith("missing_") for code in codes):
            return "incomplete"
        return "invalid"
    non_optional_warnings = [
        issue for issue in issues
        if issue.code not in {"missing_optional_base_structure"}
    ]
    if not non_optional_warnings:
        return "valid"
    return "diagnostic_only" if issues else "valid"


def _stencil_sample_id(
    method: str,
    plus: _DiscoveredDerivativeSample | None,
    minus: _DiscoveredDerivativeSample | None,
    base: _DiscoveredDerivativeSample | None,
) -> str:
    pieces = [method]
    pieces.extend(sample.sample_id for sample in (base, plus, minus) if sample is not None)
    return "dH_" + "_".join(pieces)


def _group_hash(group_key: tuple[Any, ...], index: int) -> str | None:
    value = group_key[index] if len(group_key) > index else None
    return str(value) if value else None


def _incomplete_discovery_for_sample(
    sample: _DiscoveredDerivativeSample,
    method: str | None,
) -> DerivativeStencilDiscovery:
    issues = [issue for issue in (sample.metadata_issue, sample.siesta_issue, sample.ml_issue) if issue is not None]
    issues.append(
        DerivativeValidationIssue(
            severity="error",
            code="insufficient_metadata_for_pairing",
            message="Derivative discovery did not infer stencil pairing because required metadata is unavailable.",
            sample_id=sample.sample_id,
            details={"metadata_path": str(sample.metadata_path)},
        )
    )
    return DerivativeStencilDiscovery(
        status="incomplete",
        method=method,
        group_key=("unpaired", sample.sample_id),
        stencil=None,
        issues=tuple(issues),
        sample_ids=(sample.sample_id,),
        details={"comparison_status": "diagnostic_only"},
    )


def _split_filtered_discovery_for_sample(
    sample: _DiscoveredDerivativeSample,
    split: str,
    split_issue: DerivativeValidationIssue,
) -> DerivativeStencilDiscovery:
    issues = [issue for issue in (sample.metadata_issue, sample.siesta_issue, sample.ml_issue) if issue is not None]
    issues.append(split_issue)
    return DerivativeStencilDiscovery(
        status="incomplete",
        method=None,
        group_key=("split_filtered", split, sample.sample_id),
        stencil=None,
        issues=tuple(issues),
        sample_ids=(sample.sample_id,),
        details={"comparison_status": "diagnostic_only", "requested_split": split},
    )


def _incomplete_discovery_for_group(
    group_key: tuple[Any, ...],
    plus: _DiscoveredDerivativeSample | None,
    minus: _DiscoveredDerivativeSample | None,
    base: _DiscoveredDerivativeSample | None,
    method: str,
) -> DerivativeStencilDiscovery:
    missing = []
    if method in {"central", "forward"} and plus is None:
        missing.append("plus")
    if method in {"central", "backward"} and minus is None:
        missing.append("minus")
    if method in {"forward", "backward"} and base is None:
        missing.append("base")
    issue = DerivativeValidationIssue(
        severity="error",
        code="incomplete_derivative_stencil",
        message="Derivative stencil group is missing required operands.",
        field="finite_difference_method",
        details={"missing": missing, "method": method},
    )
    return DerivativeStencilDiscovery(
        status="incomplete",
        method=method,
        group_key=group_key,
        stencil=None,
        issues=(issue,),
        sample_ids=tuple(sample.sample_id for sample in (base, plus, minus) if sample is not None),
        details={"comparison_status": "diagnostic_only"},
    )


def _ambiguous_discovery(
    group_key: tuple[Any, ...],
    plus_samples: list[_DiscoveredDerivativeSample],
    minus_samples: list[_DiscoveredDerivativeSample],
    method: str | None,
) -> DerivativeStencilDiscovery:
    issue = DerivativeValidationIssue(
        severity="error",
        code="ambiguous_derivative_pairing",
        message="Multiple samples share the same derivative grouping key and sign.",
        details={
            "plus_samples": [sample.sample_id for sample in plus_samples],
            "minus_samples": [sample.sample_id for sample in minus_samples],
        },
    )
    return DerivativeStencilDiscovery(
        status="ambiguous",
        method=method,
        group_key=group_key,
        stencil=None,
        issues=(issue,),
        sample_ids=tuple(sample.sample_id for sample in (*plus_samples, *minus_samples)),
        details={"comparison_status": "diagnostic_only"},
    )


def _ambiguous_base_discovery(
    group_key: tuple[Any, ...],
    plus: _DiscoveredDerivativeSample | None,
    minus: _DiscoveredDerivativeSample | None,
    base_samples: list[_DiscoveredDerivativeSample],
    method: str | None,
) -> DerivativeStencilDiscovery:
    issue = DerivativeValidationIssue(
        severity="error",
        code="ambiguous_base_sample",
        message="Multiple base/reference samples match this derivative grouping key for a base-dependent stencil.",
        details={
            "base_samples": [sample.sample_id for sample in base_samples],
            "method": method,
        },
    )
    return DerivativeStencilDiscovery(
        status="ambiguous",
        method=method,
        group_key=group_key,
        stencil=None,
        issues=(issue,),
        sample_ids=tuple(sample.sample_id for sample in (*base_samples, *(item for item in (plus, minus) if item is not None))),
        details={"comparison_status": "diagnostic_only"},
    )


def sparse_frobenius_norm(matrix: sparse.spmatrix) -> float:
    return float(np.sqrt(np.abs(matrix).power(2).sum()))


def sparse_hermiticity_defect(matrix: sparse.spmatrix) -> float:
    matrix = matrix.tocsr()
    rows, cols = matrix.shape
    if rows != cols:
        return math.nan
    denominator = sparse_frobenius_norm(matrix)
    if denominator == 0.0:
        return math.nan
    return sparse_frobenius_norm(matrix - matrix.getH()) / denominator


def sparse_density(matrix: sparse.spmatrix) -> float:
    rows, cols = matrix.shape
    total = int(rows) * int(cols)
    return float(matrix.nnz / total) if total else math.nan


def sparse_support_changed(first: sparse.spmatrix, second: sparse.spmatrix) -> bool:
    return _support_set(first) != _support_set(second)


def sparse_value_dict(matrix: sparse.spmatrix, *, threshold: float = DERIVATIVE_SUPPORT_THRESHOLD) -> dict[tuple[int, int], complex]:
    coo = matrix.tocoo(copy=True)
    return {
        (int(row), int(col)): complex(value)
        for row, col, value in zip(coo.row, coo.col, coo.data, strict=False)
        if abs(value) > threshold
    }


def validate_derivative_stencil(stencil: DerivativeStencil) -> list[DerivativeValidationIssue]:
    issues: list[DerivativeValidationIssue] = []
    _validate_metadata(stencil, issues)
    _validate_operands(stencil, issues)
    _validate_matrix_shapes(stencil, issues)
    _validate_operand_metadata(stencil, issues)
    _validate_comparability_hashes(stencil, issues)
    return issues


def _finite_difference_operands(
    *,
    method: str,
    plus: sparse.spmatrix | None,
    minus: sparse.spmatrix | None,
    base: sparse.spmatrix | None,
    delta_ang: float,
) -> tuple[sparse.spmatrix, sparse.spmatrix, float, tuple[str, str]]:
    if method == "central":
        if plus is None or minus is None:
            raise HamiltonianDerivativeError("central difference requires plus and minus matrices.")
        return plus, minus, 2.0 * delta_ang, ("plus", "minus")
    if method == "forward":
        if plus is None or base is None:
            raise HamiltonianDerivativeError("forward difference requires base and plus matrices.")
        return plus, base, delta_ang, ("plus", "base")
    if method == "backward":
        if base is None or minus is None:
            raise HamiltonianDerivativeError("backward difference requires base and minus matrices.")
        return base, minus, delta_ang, ("base", "minus")
    raise HamiltonianDerivativeError(f"Unsupported finite-difference method: {method!r}.")


def _require_matching_shapes(names: tuple[str, str], first: sparse.spmatrix, second: sparse.spmatrix) -> None:
    if first.shape != second.shape:
        raise HamiltonianDerivativeError(
            f"Matrix shape mismatch for {names[0]} and {names[1]}: {first.shape} vs {second.shape}."
        )


def _csr_copy(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    return matrix.tocsr(copy=True)


def _support_set(matrix: sparse.spmatrix) -> set[tuple[int, int]]:
    coo = matrix.tocoo(copy=True)
    return {
        (int(row), int(col))
        for row, col, value in zip(coo.row, coo.col, coo.data, strict=False)
        if value != 0
    }


def _sparse_finite_values(matrix: sparse.spmatrix) -> bool:
    data = matrix.data
    if data.size == 0:
        return True
    return bool(np.all(np.isfinite(data)))


def _derivative_metric_metadata(
    *,
    sample: str,
    metadata: DerivativeMetadata | None,
    source_model: str,
    reference_source: str,
) -> dict[str, Any]:
    return {
        "sample": sample,
        "atom_index_zero_based": metadata.atom_index_zero_based if metadata else None,
        "axis": metadata.axis if metadata else None,
        "axis_index": metadata.axis_index if metadata else None,
        "delta_ang": metadata.delta_ang if metadata else None,
        "finite_difference_method": metadata.method if metadata else None,
        "source_model": source_model,
        "reference_source": reference_source,
        "derivative_units": metadata.derivative_units if metadata else EXPECTED_DERIVATIVE_UNITS,
        "hamiltonian_units": metadata.hamiltonian_units if metadata else EXPECTED_HAMILTONIAN_UNITS,
        "displacement_units": metadata.displacement_units if metadata else EXPECTED_DISPLACEMENT_UNITS,
        "matrix_metric_target_space": DERIVATIVE_MATRIX_METRIC_TARGET_SPACE,
        "comparison_status": _derivative_comparison_status(metadata),
    }


def _derivative_comparison_status(metadata: DerivativeMetadata | None) -> str:
    if metadata is None:
        return "diagnostic_only"
    claim_status = str(metadata.claim_status or "").strip().lower()
    required = (
        metadata.material_compatibility_hash,
        metadata.orbital_ordering_hash,
        metadata.neighbor_list_hash,
        metadata.sparsity_pattern_hash,
    )
    if claim_status not in DIAGNOSTIC_STATUSES and all(required):
        return claim_status
    return "diagnostic_only"


def _errors_on_support(
    ref_values: dict[tuple[int, int], complex],
    pred_values: dict[tuple[int, int], complex],
    support: set[tuple[int, int]],
) -> list[complex]:
    return [pred_values.get(index, 0.0) - ref_values.get(index, 0.0) for index in support]


def _mean_abs(values: list[complex]) -> float:
    return float(np.mean(np.abs(values))) if values else math.nan


def _mse(values: list[complex]) -> float:
    return float(np.mean(np.abs(values) ** 2)) if values else math.nan


def _rmse(values: list[complex]) -> float:
    value = _mse(values)
    return float(math.sqrt(value)) if math.isfinite(value) else math.nan


def _max_abs(values: list[complex]) -> float:
    return float(max((abs(value) for value in values), default=math.nan))


def _frobenius_from_values(values: list[complex]) -> float:
    return float(math.sqrt(sum(abs(value) ** 2 for value in values)))


def _l1_from_values(values: list[complex]) -> float:
    return float(sum(abs(value) for value in values))


def _f1(intersection_count: int, pred_count: int, ref_count: int) -> float:
    if pred_count == 0 or ref_count == 0:
        return math.nan
    precision = intersection_count / pred_count
    recall = intersection_count / ref_count
    denominator = precision + recall
    return 2.0 * precision * recall / denominator if denominator else 0.0


def _cosine_similarity_from_values(
    ref_values: dict[tuple[int, int], complex],
    pred_values: dict[tuple[int, int], complex],
    support: set[tuple[int, int]],
) -> tuple[float, str]:
    ref_norm = _frobenius_from_values([ref_values.get(index, 0.0) for index in support])
    pred_norm = _frobenius_from_values([pred_values.get(index, 0.0) for index in support])
    if ref_norm == 0.0 and pred_norm == 0.0:
        return math.nan, "reference_and_prediction_derivative_norm_zero"
    if ref_norm == 0.0:
        return math.nan, "reference_derivative_norm_zero"
    if pred_norm == 0.0:
        return math.nan, "prediction_derivative_norm_zero"
    dot = sum(
        np.conjugate(ref_values.get(index, 0.0)) * pred_values.get(index, 0.0)
        for index in support
    )
    return float(np.real(dot) / (ref_norm * pred_norm)), ""


def _issue(
    issues: list[DerivativeValidationIssue],
    severity: str,
    code: str,
    message: str,
    *,
    field: str | None = None,
    sample_id: str | None = None,
    matrix_role: str | None = None,
    **details: Any,
) -> None:
    issues.append(
        DerivativeValidationIssue(
            severity=severity,
            code=code,
            message=message,
            field=field,
            sample_id=sample_id,
            matrix_role=matrix_role,
            details=details,
        )
    )


def _validate_metadata(stencil: DerivativeStencil, issues: list[DerivativeValidationIssue]) -> None:
    metadata = stencil.metadata
    if not str(metadata.sample_id or "").strip():
        _issue(issues, "error", "missing_sample_id", "Derivative stencil sample_id is required.", field="sample_id")
    method = str(metadata.method or "").strip().lower()
    if method not in VALID_METHODS:
        _issue(issues, "error", "unsupported_difference_method", f"Unsupported derivative method: {metadata.method!r}.", field="method")
    if metadata.delta_ang is None or float(metadata.delta_ang) <= 0:
        _issue(issues, "error", "invalid_delta", "Derivative stencil delta_ang must be positive.", field="delta_ang")
    if metadata.axis not in VALID_AXES:
        _issue(issues, "error", "invalid_axis", "Derivative axis must be one of x/y/z.", field="axis")
    elif metadata.axis_index != VALID_AXES[metadata.axis]:
        _issue(
            issues,
            "error",
            "axis_index_mismatch",
            "Derivative axis and axis_index disagree.",
            field="axis_index",
            axis=metadata.axis,
            axis_index=metadata.axis_index,
        )
    if metadata.atom_index_zero_based is None or int(metadata.atom_index_zero_based) < 0:
        _issue(issues, "error", "invalid_atom_index", "atom_index_zero_based must be a non-negative integer.", field="atom_index_zero_based")
    if metadata.atom_index_one_based is not None and metadata.atom_index_zero_based is not None:
        if int(metadata.atom_index_one_based) != int(metadata.atom_index_zero_based) + 1:
            _issue(issues, "error", "atom_index_mismatch", "Zero-based and one-based atom indices disagree.", field="atom_index_one_based")
    _validate_units(
        metadata.hamiltonian_units,
        metadata.displacement_units,
        metadata.derivative_units,
        issues,
        role="stencil",
        sample_id=metadata.sample_id,
    )
    claim_status = str(metadata.claim_status or "").strip().lower()
    if claim_status in PAPER_LEVEL_STATUSES:
        _issue(
            issues,
            "warning",
            "unsupported_paper_level_status",
            "Derivative stencil validation is internal/diagnostic; paper-level derivative status is not implemented.",
            field="claim_status",
            sample_id=metadata.sample_id,
            claim_status=metadata.claim_status,
        )
    if claim_status not in DIAGNOSTIC_STATUSES:
        for field_name in REQUIRED_NON_DIAGNOSTIC_HASHES:
            if not getattr(metadata, field_name):
                _issue(
                    issues,
                    "error",
                    "missing_required_metadata",
                    f"{field_name} is required when derivative comparison claims more than diagnostic status.",
                    field=field_name,
                    sample_id=metadata.sample_id,
                    claim_status=metadata.claim_status,
                )
    if method == "central" and not metadata.base_sample_id and stencil.base_structure_path is None:
        _issue(
            issues,
            "warning",
            "missing_optional_base_structure",
            "Central derivative stencil has no optional base structure/sample metadata.",
            field="base_sample_id",
            sample_id=metadata.sample_id,
        )


def _validate_units(
    hamiltonian_units: str,
    displacement_units: str,
    derivative_units: str,
    issues: list[DerivativeValidationIssue],
    *,
    role: str,
    sample_id: str | None,
    matrix_role: str | None = None,
) -> None:
    checks = (
        ("hamiltonian_units", hamiltonian_units, EXPECTED_HAMILTONIAN_UNITS),
        ("displacement_units", displacement_units, EXPECTED_DISPLACEMENT_UNITS),
        ("derivative_units", derivative_units, EXPECTED_DERIVATIVE_UNITS),
    )
    for field_name, value, expected in checks:
        if value != expected:
            _issue(
                issues,
                "error",
                "unit_mismatch",
                f"{role} {field_name} must be {expected!r}, got {value!r}.",
                field=field_name,
                sample_id=sample_id,
                matrix_role=matrix_role,
                expected=expected,
                actual=value,
            )


def _validate_operands(stencil: DerivativeStencil, issues: list[DerivativeValidationIssue]) -> None:
    method = str(stencil.metadata.method or "").strip().lower()
    required_roles = {
        "central": ("siesta_plus", "siesta_minus", "ml_plus", "ml_minus"),
        "forward": ("siesta_base", "siesta_plus", "ml_base", "ml_plus"),
        "backward": ("siesta_base", "siesta_minus", "ml_base", "ml_minus"),
    }.get(method, ())
    for role in required_roles:
        if stencil.matrix_inputs()[role] is None:
            _issue(issues, "error", "missing_derivative_operand", f"Missing required {method} derivative operand: {role}.", matrix_role=role)
    for role, matrix in stencil.matrix_inputs().items():
        if matrix is None:
            continue
        source = str(matrix.source or "").strip().lower()
        if not source:
            _issue(issues, "error", "missing_source_label", "Derivative matrix source is required.", matrix_role=role, sample_id=matrix.sample_id)
        elif source not in VALID_SOURCES:
            _issue(
                issues,
                "error",
                "unsupported_source_label",
                f"Unsupported derivative matrix source: {matrix.source!r}.",
                matrix_role=role,
                sample_id=matrix.sample_id,
            )
        if matrix.matrix_path is None:
            _issue(issues, "error", "missing_matrix_path", "Derivative matrix path is required.", matrix_role=role, sample_id=matrix.sample_id)
        elif source == "siesta" and matrix.matrix_path.name in FORBIDDEN_SIESTA_REFERENCE_NAMES:
            _issue(
                issues,
                "error",
                "forbidden_siesta_reference",
                "ML_prediction.HSX cannot be used as a SIESTA derivative reference.",
                matrix_role=role,
                sample_id=matrix.sample_id,
                matrix_path=str(matrix.matrix_path),
            )
        if matrix.matrix_sha256 is None:
            _issue(issues, "error", "missing_matrix_sha256", "Derivative matrix sha256 is required.", matrix_role=role, sample_id=matrix.sample_id)
        if matrix.matrix_shape is None:
            _issue(issues, "error", "missing_matrix_shape", "Derivative matrix shape is required.", matrix_role=role, sample_id=matrix.sample_id)
        _validate_units(
            matrix.hamiltonian_units,
            matrix.displacement_units,
            matrix.derivative_units,
            issues,
            role="matrix",
            sample_id=matrix.sample_id,
            matrix_role=role,
        )


def _validate_matrix_shapes(stencil: DerivativeStencil, issues: list[DerivativeValidationIssue]) -> None:
    shapes = {
        role: matrix.matrix_shape
        for role, matrix in stencil.matrix_inputs().items()
        if matrix is not None and matrix.matrix_shape is not None
    }
    unique_shapes = {tuple(shape) for shape in shapes.values()}
    if len(unique_shapes) > 1:
        _issue(
            issues,
            "error",
            "matrix_shape_mismatch",
            "Derivative stencil operands must have matching Hamiltonian matrix shapes.",
            field="matrix_shape",
            sample_id=stencil.metadata.sample_id,
            shapes={role: list(shape) for role, shape in shapes.items()},
        )


def _validate_operand_metadata(stencil: DerivativeStencil, issues: list[DerivativeValidationIssue]) -> None:
    metadata = stencil.metadata
    for role, matrix in stencil.matrix_inputs().items():
        if matrix is None:
            continue
        if matrix.atom_index_zero_based is not None and metadata.atom_index_zero_based is not None:
            if int(matrix.atom_index_zero_based) != int(metadata.atom_index_zero_based):
                _issue(
                    issues,
                    "error",
                    "atom_index_mismatch",
                    "Operand atom_index_zero_based does not match stencil metadata.",
                    field="atom_index_zero_based",
                    matrix_role=role,
                    sample_id=matrix.sample_id,
                    expected=metadata.atom_index_zero_based,
                    actual=matrix.atom_index_zero_based,
                )
        if matrix.atom_index_one_based is not None and metadata.atom_index_one_based is not None:
            if int(matrix.atom_index_one_based) != int(metadata.atom_index_one_based):
                _issue(
                    issues,
                    "error",
                    "atom_index_mismatch",
                    "Operand atom_index_one_based does not match stencil metadata.",
                    field="atom_index_one_based",
                    matrix_role=role,
                    sample_id=matrix.sample_id,
                    expected=metadata.atom_index_one_based,
                    actual=matrix.atom_index_one_based,
                )
        if matrix.axis is not None and metadata.axis is not None and matrix.axis != metadata.axis:
            _issue(
                issues,
                "error",
                "axis_mismatch",
                "Operand axis does not match stencil metadata.",
                field="axis",
                matrix_role=role,
                sample_id=matrix.sample_id,
                expected=metadata.axis,
                actual=matrix.axis,
            )
        if matrix.axis_index is not None and metadata.axis_index is not None and int(matrix.axis_index) != int(metadata.axis_index):
            _issue(
                issues,
                "error",
                "axis_mismatch",
                "Operand axis_index does not match stencil metadata.",
                field="axis_index",
                matrix_role=role,
                sample_id=matrix.sample_id,
                expected=metadata.axis_index,
                actual=matrix.axis_index,
            )
        if role.endswith("_base"):
            continue
        if matrix.delta_ang is not None and metadata.delta_ang is not None:
            if not math.isclose(float(matrix.delta_ang), float(metadata.delta_ang), rel_tol=0.0, abs_tol=1e-12):
                _issue(
                    issues,
                    "error",
                    "delta_mismatch",
                    "Operand delta_ang does not match stencil metadata.",
                    field="delta_ang",
                    matrix_role=role,
                    sample_id=matrix.sample_id,
                    expected=metadata.delta_ang,
                    actual=matrix.delta_ang,
                )


def _validate_comparability_hashes(stencil: DerivativeStencil, issues: list[DerivativeValidationIssue]) -> None:
    metadata = stencil.metadata
    for field_name in OPTIONAL_COMPARABILITY_HASHES:
        values = _hash_values(stencil, field_name)
        if not values:
            _issue(
                issues,
                "warning",
                f"missing_{field_name}",
                f"{field_name} is unavailable for derivative comparability validation.",
                field=field_name,
                sample_id=metadata.sample_id,
            )
    for field_name in COMPARABILITY_HASH_FIELDS:
        values_by_source = _hash_values_by_source(stencil, field_name)
        unique = sorted(set(values_by_source.values()))
        if len(unique) > 1:
            _issue(
                issues,
                "error",
                "metadata_hash_mismatch",
                f"{field_name} differs across derivative stencil operands.",
                field=field_name,
                sample_id=metadata.sample_id,
                values=values_by_source,
            )


def _hash_values(stencil: DerivativeStencil, field_name: str) -> list[str]:
    values = [str(value) for value in [getattr(stencil.metadata, field_name, None)] if value]
    for matrix in stencil.matrix_inputs().values():
        if matrix is None:
            continue
        value = getattr(matrix, field_name, None)
        if value:
            values.append(str(value))
    return values


def _hash_values_by_source(stencil: DerivativeStencil, field_name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    metadata_value = getattr(stencil.metadata, field_name, None)
    if metadata_value:
        values["stencil"] = str(metadata_value)
    for role, matrix in stencil.matrix_inputs().items():
        if matrix is None:
            continue
        value = getattr(matrix, field_name, None)
        if value:
            values[role] = str(value)
    return values
