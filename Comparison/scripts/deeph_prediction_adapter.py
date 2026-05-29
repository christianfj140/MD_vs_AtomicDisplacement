#!/usr/bin/env python3
"""Validate DeepH HDF5 predictions before common metric evaluation.

This adapter is intentionally narrow. It does not fabricate SIESTA HSX files,
and it does not claim that DeepH's processed HDF5 block convention is identical
to Graph2Mat's raw HSX convention. It only verifies the DeepH HDF5 prediction
against the DeepH processed SIESTA reference produced from the same snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ADAPTER_NAME = "deeph_hdf5_prediction_adapter"
ADAPTER_VERSION = "deeph_hdf5_prediction_adapter_v1"
GLOBAL_PREDICTION_FILENAME = "hamiltonians_pred.h5"
LOCAL_FRAME_PREDICTION_FILENAME = "rh_pred.h5"
EQUIVALENCE_PROVEN_RAW_GLOBAL = "proven_raw_global_hamiltonian_equivalent"
EQUIVALENCE_DIAGNOSTIC_LOCAL_FRAME = "diagnostic_local_frame_only"
EQUIVALENCE_INVALID_SHAPE = "invalid_shape_mismatch"
EQUIVALENCE_INVALID_ORBITAL_ORDER = "invalid_orbital_order_unknown"
EQUIVALENCE_INVALID_UNITS = "invalid_units_unknown"
EQUIVALENCE_INVALID_R_VECTOR = "invalid_r_vector_convention_unknown"
EQUIVALENCE_INVALID_MISSING_REFERENCE = "invalid_missing_reference_mapping"
EQUIVALENCE_INVALID_EVIDENCE = "invalid_raw_global_equivalence_evidence"
PROVEN_ADAPTER_EQUIVALENCE_STATUSES = {EQUIVALENCE_PROVEN_RAW_GLOBAL}
EQUIVALENCE_STATUS_PROVEN = "proven"
EQUIVALENCE_STATUS_FAILED = "failed"
EQUIVALENCE_STATUS_UNPROVEN = "unproven"
EQUIVALENCE_STATUS_NOT_APPLICABLE = "not_applicable"
EQUIVALENCE_SCOPE_RAW_GLOBAL = "raw_global"
EQUIVALENCE_SCOPE_DEEPH_PROCESSED_BLOCKWISE = "deeph_processed_blockwise_global_hdf5"
EQUIVALENCE_SCOPE_LOCAL_FRAME = "local_frame_hprime"
EQUIVALENCE_SCOPE_UNKNOWN = "unknown"
RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME = "raw_global_equivalence_evidence.json"
RAW_GLOBAL_EQUIVALENCE_EVIDENCE_SCHEMA = "deeph_raw_global_equivalence_evidence_v1"
RAW_GLOBAL_EQUIVALENCE_REQUIRED_CHECKS = (
    "shape",
    "units",
    "orbital_order",
    "atom_order",
    "r_vectors",
    "spin",
    "sparse_support",
    "hk",
    "s_ref",
    "eigenvalues",
)


class DeepHPredictionAdapterError(RuntimeError):
    """Raised when a DeepH prediction cannot be safely adapted."""


def equivalence_status_from_adapter_status(adapter_status: str) -> str:
    status = str(adapter_status or "").strip()
    if status in PROVEN_ADAPTER_EQUIVALENCE_STATUSES:
        return EQUIVALENCE_STATUS_PROVEN
    if status == EQUIVALENCE_DIAGNOSTIC_LOCAL_FRAME:
        return EQUIVALENCE_STATUS_NOT_APPLICABLE
    if status in {EQUIVALENCE_INVALID_SHAPE, EQUIVALENCE_INVALID_MISSING_REFERENCE, EQUIVALENCE_INVALID_EVIDENCE}:
        return EQUIVALENCE_STATUS_FAILED
    return EQUIVALENCE_STATUS_UNPROVEN


def equivalence_scope_from_adapter_status(adapter_status: str, target_space: str = "") -> str:
    status = str(adapter_status or "").strip()
    target = str(target_space or "").strip()
    if status in PROVEN_ADAPTER_EQUIVALENCE_STATUSES:
        return EQUIVALENCE_SCOPE_RAW_GLOBAL
    if status == EQUIVALENCE_DIAGNOSTIC_LOCAL_FRAME or "local_coordinate" in target:
        return EQUIVALENCE_SCOPE_LOCAL_FRAME
    if "global_hamiltonian_h5_blocks" in target:
        return EQUIVALENCE_SCOPE_DEEPH_PROCESSED_BLOCKWISE
    return EQUIVALENCE_SCOPE_UNKNOWN


@dataclass
class DeepHPredictionAdapterResult:
    sample_id: str
    status: str
    metrics_ready: bool
    diagnostic_only: bool
    diagnostic_reason: str
    prediction_path: str | None
    processed_sample_dir: str
    reference_hamiltonian_path: str | None
    reference_overlap_path: str | None
    orbital_types_path: str | None
    n_orbitals: int | None
    block_count: int
    prediction_key_count: int
    reference_key_count: int
    missing_reference_keys: list[str] = field(default_factory=list)
    extra_prediction_keys: list[str] = field(default_factory=list)
    gamma_hermiticity_defect: float | None = None
    reference_gamma_hermiticity_defect: float | None = None
    comparability_status: str = "unknown"
    adapter_equivalence_status: str = EQUIVALENCE_INVALID_MISSING_REFERENCE
    target_space: str = "unknown"
    units_status: str = "unknown"
    orbital_order_status: str = "unknown"
    r_vector_convention_status: str = "unknown"
    support_semantics_status: str = "unknown"
    equivalence_status: str = ""
    equivalence_scope: str = ""
    equivalence_evidence_paths: list[str] = field(default_factory=list)
    equivalence_reason: str = ""
    adapter_name: str = ADAPTER_NAME
    adapter_version: str = ADAPTER_VERSION
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.equivalence_status:
            self.equivalence_status = equivalence_status_from_adapter_status(self.adapter_equivalence_status)
        if not self.equivalence_scope:
            self.equivalence_scope = equivalence_scope_from_adapter_status(
                self.adapter_equivalence_status,
                self.target_space,
            )
        if not self.equivalence_reason:
            if self.equivalence_status == EQUIVALENCE_STATUS_PROVEN:
                self.equivalence_reason = "raw/global Hamiltonian equivalence evidence is recorded."
            elif self.diagnostic_reason:
                self.equivalence_reason = self.diagnostic_reason
            else:
                self.equivalence_reason = "DeepH raw/global equivalence evidence is unavailable."
        if not self.equivalence_evidence_paths:
            self.equivalence_evidence_paths = [
                str(path)
                for path in (
                    self.prediction_path,
                    self.reference_hamiltonian_path,
                    self.reference_overlap_path,
                    self.orbital_types_path,
                )
                if path
            ]
        if self.equivalence_status != EQUIVALENCE_STATUS_PROVEN:
            self.diagnostic_only = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def metric_fields(self) -> dict[str, Any]:
        return {
            "prediction_adapter": self.adapter_name,
            "prediction_adapter_version": self.adapter_version,
            "deeph_comparability_status": self.comparability_status,
            "deeph_adapter_equivalence_status": self.adapter_equivalence_status,
            "deeph_equivalence_status": self.equivalence_status,
            "deeph_equivalence_scope": self.equivalence_scope,
            "deeph_equivalence_evidence_paths": list(self.equivalence_evidence_paths),
            "deeph_equivalence_reason": self.equivalence_reason,
            "deeph_raw_global_equivalence_proven": self.adapter_equivalence_status
            in PROVEN_ADAPTER_EQUIVALENCE_STATUSES
            and self.equivalence_status == EQUIVALENCE_STATUS_PROVEN,
            "deeph_diagnostic_only": self.diagnostic_only,
            "deeph_diagnostic_reason": self.diagnostic_reason,
            "deeph_prediction_target_space": self.target_space,
            "deeph_units_status": self.units_status,
            "deeph_orbital_order_status": self.orbital_order_status,
            "deeph_r_vector_convention_status": self.r_vector_convention_status,
            "deeph_support_semantics_status": self.support_semantics_status,
            "deeph_prediction_metrics_ready": self.metrics_ready,
        }


def _require_h5py_numpy():
    try:
        import h5py  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency availability varies by env.
        raise DeepHPredictionAdapterError(
            "DeepH HDF5 prediction adaptation requires h5py and numpy."
        ) from exc
    return h5py, np


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeepHPredictionAdapterError(f"{EQUIVALENCE_INVALID_EVIDENCE}: cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DeepHPredictionAdapterError(f"{EQUIVALENCE_INVALID_EVIDENCE}: evidence must be a JSON object: {path}")
    return payload


def find_raw_global_equivalence_evidence(
    *,
    work_dir: Path,
    processed_sample_dir: Path,
    sample_id: str,
) -> Path | None:
    candidates = [
        work_dir / RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME,
        processed_sample_dir / RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME,
        work_dir / f"{sample_id}_{RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME}",
        processed_sample_dir / f"{sample_id}_{RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME}",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _bool_check(checks: dict[str, Any], key: str) -> bool:
    value = checks.get(key)
    if isinstance(value, dict):
        return value.get("status") in {True, "pass", "passed", "ok", "proven"}
    return value is True or str(value).strip().lower() in {"true", "pass", "passed", "ok", "proven"}


def validate_raw_global_equivalence_evidence(
    path: Path,
    *,
    sample_id: str,
) -> dict[str, Any]:
    payload = read_json(path)
    status = str(payload.get("equivalence_status") or payload.get("status") or "").strip().lower()
    scope = str(payload.get("equivalence_scope") or payload.get("scope") or "").strip().lower()
    if payload.get("sample_id") not in (None, "", sample_id):
        return {
            "status": EQUIVALENCE_STATUS_FAILED,
            "reason": (
                f"{EQUIVALENCE_INVALID_EVIDENCE}: sample_id mismatch in {path}: "
                f"{payload.get('sample_id')!r} != {sample_id!r}"
            ),
            "payload": payload,
        }
    if status != EQUIVALENCE_STATUS_PROVEN or scope != EQUIVALENCE_SCOPE_RAW_GLOBAL:
        return {
            "status": EQUIVALENCE_STATUS_FAILED,
            "reason": (
                f"{EQUIVALENCE_INVALID_EVIDENCE}: evidence must declare "
                f"status=proven and scope=raw_global."
            ),
            "payload": payload,
        }
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        return {
            "status": EQUIVALENCE_STATUS_FAILED,
            "reason": f"{EQUIVALENCE_INVALID_EVIDENCE}: missing checks object.",
            "payload": payload,
        }
    missing_or_failed = [key for key in RAW_GLOBAL_EQUIVALENCE_REQUIRED_CHECKS if not _bool_check(checks, key)]
    if missing_or_failed:
        return {
            "status": EQUIVALENCE_STATUS_FAILED,
            "reason": (
                f"{EQUIVALENCE_INVALID_EVIDENCE}: required checks failed or missing: "
                + ", ".join(missing_or_failed)
            ),
            "payload": payload,
        }
    errors = payload.get("errors") or {}
    tolerances = payload.get("tolerances") or {}
    if isinstance(errors, dict) and isinstance(tolerances, dict):
        for key, raw_error in errors.items():
            if key not in tolerances:
                continue
            try:
                error = abs(float(raw_error))
                tolerance = abs(float(tolerances[key]))
            except (TypeError, ValueError):
                return {
                    "status": EQUIVALENCE_STATUS_FAILED,
                    "reason": f"{EQUIVALENCE_INVALID_EVIDENCE}: non-numeric error/tolerance for {key}.",
                    "payload": payload,
                }
            if not math.isfinite(error) or not math.isfinite(tolerance) or error > tolerance:
                return {
                    "status": EQUIVALENCE_STATUS_FAILED,
                    "reason": (
                        f"{EQUIVALENCE_INVALID_EVIDENCE}: {key}={error} exceeds tolerance {tolerance}."
                    ),
                    "payload": payload,
                }
    return {
        "status": EQUIVALENCE_STATUS_PROVEN,
        "reason": "raw/global Hamiltonian equivalence evidence passed all required checks.",
        "payload": payload,
    }


def count_orbitals_from_orbital_types(path: Path) -> list[int]:
    if not path.exists():
        raise DeepHPredictionAdapterError(f"Missing DeepH orbital_types.dat: {path}")
    counts: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = [token for token in line.split() if token.strip()]
        if tokens:
            counts.append(sum(2 * int(token) + 1 for token in tokens))
    if not counts:
        raise DeepHPredictionAdapterError(f"No orbital counts found in {path}")
    return counts


def parse_block_key(key: str) -> tuple[tuple[int, int, int], int, int]:
    try:
        values = json.loads(key)
    except json.JSONDecodeError as exc:
        raise DeepHPredictionAdapterError(f"Invalid DeepH block key JSON: {key}") from exc
    if not isinstance(values, list) or len(values) != 5:
        raise DeepHPredictionAdapterError(f"Invalid DeepH block key: {key}")
    return (int(values[0]), int(values[1]), int(values[2])), int(values[3]) - 1, int(values[4]) - 1


def expected_block_shape(key: str, orbital_counts: list[int]) -> tuple[int, int]:
    _, atom_i, atom_j = parse_block_key(key)
    if atom_i < 0 or atom_j < 0 or atom_i >= len(orbital_counts) or atom_j >= len(orbital_counts):
        raise DeepHPredictionAdapterError(f"DeepH block key atom index out of range: {key}")
    return int(orbital_counts[atom_i]), int(orbital_counts[atom_j])


def h5_block_shapes(path: Path, orbital_counts: list[int], *, label: str) -> dict[str, tuple[int, int]]:
    h5py, _ = _require_h5py_numpy()
    if not path.exists():
        raise DeepHPredictionAdapterError(f"Missing {label} HDF5 file: {path}")
    shapes: dict[str, tuple[int, int]] = {}
    with h5py.File(path, "r") as handle:
        for key in sorted(handle.keys()):
            expected = expected_block_shape(key, orbital_counts)
            shape = tuple(int(value) for value in handle[key].shape)
            if shape != expected:
                raise DeepHPredictionAdapterError(
                    f"{EQUIVALENCE_INVALID_SHAPE}: block shape mismatch in {path}: "
                    f"{key} has {shape}, expected {expected}"
                )
            shapes[key] = expected
    if not shapes:
        raise DeepHPredictionAdapterError(f"{label} HDF5 file contains no blocks: {path}")
    return shapes


def assemble_hk(block_h5: Path, orbital_types: Path, kpoint: tuple[float, float, float]) -> Any:
    h5py, np = _require_h5py_numpy()
    orbital_counts = count_orbitals_from_orbital_types(orbital_types)
    offsets = np.cumsum([0, *orbital_counts])
    matrix = np.zeros((int(offsets[-1]), int(offsets[-1])), dtype=np.complex128)
    with h5py.File(block_h5, "r") as handle:
        for key in handle.keys():
            lattice_r, atom_i, atom_j = parse_block_key(key)
            r0, r1 = int(offsets[atom_i]), int(offsets[atom_i + 1])
            c0, c1 = int(offsets[atom_j]), int(offsets[atom_j + 1])
            block = np.asarray(handle[key][()])
            if block.shape != (r1 - r0, c1 - c0):
                raise DeepHPredictionAdapterError(
                    f"{EQUIVALENCE_INVALID_SHAPE}: block shape mismatch in {block_h5}: "
                    f"{key} has {block.shape}, expected {(r1-r0, c1-c0)}"
                )
            phase = np.exp(
                2j
                * np.pi
                * float(np.dot(np.asarray(kpoint, dtype=float), np.asarray(lattice_r, dtype=float)))
            )
            matrix[r0:r1, c0:c1] += block * phase
    return matrix


def hermiticity_defect(matrix: Any) -> float:
    _, np = _require_h5py_numpy()
    if matrix.size == 0:
        return math.nan
    denominator = float(np.linalg.norm(matrix))
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(matrix - matrix.conj().T) / denominator)


def find_prediction_file(work_dir: Path, prediction_filename: str = GLOBAL_PREDICTION_FILENAME) -> tuple[Path, str]:
    requested = work_dir / prediction_filename
    if requested.exists():
        return requested, "global"
    global_path = work_dir / GLOBAL_PREDICTION_FILENAME
    if global_path.exists():
        return global_path, "global"
    local_path = work_dir / LOCAL_FRAME_PREDICTION_FILENAME
    if local_path.exists():
        return local_path, "local_frame"
    raise DeepHPredictionAdapterError(
        f"Missing DeepH prediction HDF5 under {work_dir}: expected {prediction_filename}, "
        f"{GLOBAL_PREDICTION_FILENAME}, or {LOCAL_FRAME_PREDICTION_FILENAME}"
    )


def _provenance(
    *,
    prediction_path: Path | None,
    reference_hamiltonian: Path | None,
    reference_overlap: Path | None,
    orbital_types: Path | None,
) -> dict[str, Any]:
    files = {
        "prediction": prediction_path,
        "reference_hamiltonian": reference_hamiltonian,
        "reference_overlap": reference_overlap,
        "orbital_types": orbital_types,
    }
    return {
        "adapter_name": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "files": {
            key: {
                "path": str(path) if path is not None else None,
                "sha256": file_sha256(path),
            }
            for key, path in files.items()
        },
    }


def adapt_deeph_prediction_sample(
    *,
    work_dir: Path,
    processed_sample_dir: Path,
    sample_id: str | None = None,
    prediction_filename: str = GLOBAL_PREDICTION_FILENAME,
) -> DeepHPredictionAdapterResult:
    work_dir = Path(work_dir)
    processed_sample_dir = Path(processed_sample_dir)
    sample_id = str(sample_id or work_dir.name)
    prediction_path, prediction_kind = find_prediction_file(work_dir, prediction_filename)
    orbital_types = processed_sample_dir / "orbital_types.dat"
    reference_hamiltonian = processed_sample_dir / "hamiltonians.h5"
    reference_overlap = processed_sample_dir / "overlaps.h5"

    if prediction_kind == "local_frame":
        return DeepHPredictionAdapterResult(
            sample_id=sample_id,
            status="local_frame_prediction_only",
            metrics_ready=False,
            diagnostic_only=True,
            diagnostic_reason=(
                "DeepH produced rh_pred.h5 only. This is the local-coordinate H' "
                "representation and is not a raw/global Hamiltonian for common metrics."
            ),
            prediction_path=str(prediction_path),
            processed_sample_dir=str(processed_sample_dir),
            reference_hamiltonian_path=str(reference_hamiltonian) if reference_hamiltonian.exists() else None,
            reference_overlap_path=str(reference_overlap) if reference_overlap.exists() else None,
            orbital_types_path=str(orbital_types) if orbital_types.exists() else None,
            n_orbitals=None,
            block_count=0,
            prediction_key_count=0,
            reference_key_count=0,
            comparability_status="diagnostic_only_local_frame_hprime",
            adapter_equivalence_status=EQUIVALENCE_DIAGNOSTIC_LOCAL_FRAME,
            target_space="deeph_local_coordinate_hprime",
            units_status="not_applicable_to_common_raw_global_metrics",
            orbital_order_status="not_validated",
            r_vector_convention_status="not_validated",
            support_semantics_status="not_validated",
            equivalence_status=EQUIVALENCE_STATUS_NOT_APPLICABLE,
            equivalence_scope=EQUIVALENCE_SCOPE_LOCAL_FRAME,
            equivalence_reason="DeepH local-coordinate H' output is not a raw/global Hamiltonian.",
            provenance=_provenance(
                prediction_path=prediction_path,
                reference_hamiltonian=reference_hamiltonian if reference_hamiltonian.exists() else None,
                reference_overlap=reference_overlap if reference_overlap.exists() else None,
                orbital_types=orbital_types if orbital_types.exists() else None,
            ),
        )

    orbital_counts = count_orbitals_from_orbital_types(orbital_types)
    n_orbitals = int(sum(orbital_counts))
    pred_shapes = h5_block_shapes(prediction_path, orbital_counts, label="DeepH prediction")
    ref_shapes = h5_block_shapes(reference_hamiltonian, orbital_counts, label="DeepH processed reference")
    h5_block_shapes(reference_overlap, orbital_counts, label="DeepH processed overlap")
    pred_keys = set(pred_shapes)
    ref_keys = set(ref_shapes)
    missing_reference_keys = sorted(ref_keys - pred_keys)
    extra_prediction_keys = sorted(pred_keys - ref_keys)
    if missing_reference_keys or extra_prediction_keys:
        raise DeepHPredictionAdapterError(
            f"{EQUIVALENCE_INVALID_MISSING_REFERENCE}: DeepH prediction/reference block support mismatch: "
            f"missing_prediction_keys={missing_reference_keys[:10]} "
            f"extra_prediction_keys={extra_prediction_keys[:10]}"
        )
    for key in sorted(pred_keys):
        if pred_shapes[key] != ref_shapes[key]:
            raise DeepHPredictionAdapterError(
                f"{EQUIVALENCE_INVALID_SHAPE}: DeepH prediction/reference block shape mismatch for {key}: "
                f"{pred_shapes[key]} vs {ref_shapes[key]}"
            )

    pred_gamma = assemble_hk(prediction_path, orbital_types, (0.0, 0.0, 0.0))
    ref_gamma = assemble_hk(reference_hamiltonian, orbital_types, (0.0, 0.0, 0.0))
    evidence_path = find_raw_global_equivalence_evidence(
        work_dir=work_dir,
        processed_sample_dir=processed_sample_dir,
        sample_id=sample_id,
    )
    evidence: dict[str, Any] | None = None
    evidence_reason = ""
    evidence_status = ""
    if evidence_path is not None:
        evidence = validate_raw_global_equivalence_evidence(evidence_path, sample_id=sample_id)
        evidence_status = str(evidence.get("status") or "")
        evidence_reason = str(evidence.get("reason") or "")
        if evidence_status == EQUIVALENCE_STATUS_PROVEN:
            provenance = _provenance(
                prediction_path=prediction_path,
                reference_hamiltonian=reference_hamiltonian,
                reference_overlap=reference_overlap,
                orbital_types=orbital_types,
            )
            provenance["raw_global_equivalence_evidence"] = {
                "path": str(evidence_path),
                "sha256": file_sha256(evidence_path),
                "schema": (evidence.get("payload") or {}).get("schema"),
            }
            return DeepHPredictionAdapterResult(
                sample_id=sample_id,
                status="ok",
                metrics_ready=True,
                diagnostic_only=False,
                diagnostic_reason="",
                prediction_path=str(prediction_path),
                processed_sample_dir=str(processed_sample_dir),
                reference_hamiltonian_path=str(reference_hamiltonian),
                reference_overlap_path=str(reference_overlap),
                orbital_types_path=str(orbital_types),
                n_orbitals=n_orbitals,
                block_count=len(pred_shapes),
                prediction_key_count=len(pred_shapes),
                reference_key_count=len(ref_shapes),
                missing_reference_keys=missing_reference_keys,
                extra_prediction_keys=extra_prediction_keys,
                gamma_hermiticity_defect=hermiticity_defect(pred_gamma),
                reference_gamma_hermiticity_defect=hermiticity_defect(ref_gamma),
                comparability_status="raw_global_equivalence_proven",
                adapter_equivalence_status=EQUIVALENCE_PROVEN_RAW_GLOBAL,
                target_space="deeph_rotate_back_global_hamiltonian_h5_blocks_verified_raw_global",
                units_status="verified_by_raw_global_equivalence_evidence",
                orbital_order_status="verified_by_raw_global_equivalence_evidence",
                r_vector_convention_status="verified_by_raw_global_equivalence_evidence",
                support_semantics_status="verified_by_raw_global_equivalence_evidence",
                equivalence_status=EQUIVALENCE_STATUS_PROVEN,
                equivalence_scope=EQUIVALENCE_SCOPE_RAW_GLOBAL,
                equivalence_evidence_paths=[str(evidence_path)],
                equivalence_reason=evidence_reason,
                provenance=provenance,
            )
    warnings = [
        "DeepH HDF5 blocks were validated against DeepH processed SIESTA HDF5 artifacts only.",
        "Equivalence to Graph2Mat raw HSX orbital order/sign convention is not independently proven.",
        f"{EQUIVALENCE_INVALID_UNITS}: DeepH processed energy units were not independently checked against Graph2Mat HSX.",
        f"{EQUIVALENCE_INVALID_R_VECTOR}: DeepH HDF5 R-vector convention was not independently checked against Graph2Mat HSX.",
    ]
    provenance = _provenance(
        prediction_path=prediction_path,
        reference_hamiltonian=reference_hamiltonian,
        reference_overlap=reference_overlap,
        orbital_types=orbital_types,
    )
    if evidence_path is not None:
        provenance["raw_global_equivalence_evidence"] = {
            "path": str(evidence_path),
            "sha256": file_sha256(evidence_path),
            "schema": (evidence or {}).get("payload", {}).get("schema") if isinstance(evidence, dict) else None,
        }
    return DeepHPredictionAdapterResult(
        sample_id=sample_id,
        status="ok",
        metrics_ready=True,
        diagnostic_only=True,
        diagnostic_reason="basis_equivalence_to_graph2mat_raw_hsx_not_proven",
        prediction_path=str(prediction_path),
        processed_sample_dir=str(processed_sample_dir),
        reference_hamiltonian_path=str(reference_hamiltonian),
        reference_overlap_path=str(reference_overlap),
        orbital_types_path=str(orbital_types),
        n_orbitals=n_orbitals,
        block_count=len(pred_shapes),
        prediction_key_count=len(pred_shapes),
        reference_key_count=len(ref_shapes),
        missing_reference_keys=missing_reference_keys,
        extra_prediction_keys=extra_prediction_keys,
        gamma_hermiticity_defect=hermiticity_defect(pred_gamma),
        reference_gamma_hermiticity_defect=hermiticity_defect(ref_gamma),
        comparability_status="diagnostic_deeph_processed_global_hdf5_blocks_shape_validated",
        adapter_equivalence_status=EQUIVALENCE_INVALID_EVIDENCE if evidence_path is not None else EQUIVALENCE_INVALID_ORBITAL_ORDER,
        target_space="deeph_rotate_back_global_hamiltonian_h5_blocks",
        units_status="deeph_siesta_preprocess_internal_energy_units_unverified_against_graph2mat_hsx",
        orbital_order_status="validated_against_deeph_processed_reference_only",
        r_vector_convention_status="validated_against_deeph_processed_reference_only",
        support_semantics_status="prediction_and_processed_reference_key_sets_match",
        equivalence_status=EQUIVALENCE_STATUS_FAILED if evidence_path is not None else EQUIVALENCE_STATUS_UNPROVEN,
        equivalence_scope=EQUIVALENCE_SCOPE_DEEPH_PROCESSED_BLOCKWISE,
        equivalence_evidence_paths=[str(evidence_path)] if evidence_path is not None else [],
        equivalence_reason=evidence_reason
        or (
            "DeepH prediction was validated only against DeepH processed SIESTA HDF5 blocks; "
            "raw/global HSX units, orbital order, and R-vector convention are not proven."
        ),
        provenance=provenance,
        warnings=warnings,
    )


def write_adapter_manifest(path: Path, results: list[DeepHPredictionAdapterResult]) -> dict[str, Any]:
    adapter_equivalence_statuses = sorted({result.adapter_equivalence_status for result in results})
    equivalence_statuses = sorted({result.equivalence_status for result in results})
    equivalence_scopes = sorted({result.equivalence_scope for result in results})
    equivalence_evidence_paths = sorted(
        {path for result in results for path in result.equivalence_evidence_paths if path}
    )
    proven_count = sum(
        1
        for result in results
        if result.adapter_equivalence_status in PROVEN_ADAPTER_EQUIVALENCE_STATUSES
        and result.equivalence_status == EQUIVALENCE_STATUS_PROVEN
    )
    robust_allowed = bool(results) and proven_count == len(results)
    blocked_reasons = sorted(
        {
            result.equivalence_reason
            for result in results
            if result.equivalence_status != EQUIVALENCE_STATUS_PROVEN and result.equivalence_reason
        }
    )
    payload = {
        "schema": ADAPTER_VERSION,
        "adapter_name": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "sample_count": len(results),
        "metrics_ready_count": sum(1 for result in results if result.metrics_ready),
        "diagnostic_only_count": sum(1 for result in results if result.diagnostic_only),
        "adapter_equivalence_statuses": adapter_equivalence_statuses,
        "equivalence_statuses": equivalence_statuses,
        "equivalence_scopes": equivalence_scopes,
        "equivalence_evidence_paths": equivalence_evidence_paths,
        "raw_global_equivalence_proven_count": proven_count,
        "robust_matrix_metrics_allowed": robust_allowed,
        "equivalence_gate": {
            "robust_claim_allowed": robust_allowed,
            "diagnostic_only": not robust_allowed,
            "required_status": EQUIVALENCE_STATUS_PROVEN,
            "required_scope": EQUIVALENCE_SCOPE_RAW_GLOBAL,
            "diagnostic_only_reason": "; ".join(blocked_reasons)
            if blocked_reasons
            else ("" if robust_allowed else "DeepH equivalence evidence is missing."),
        },
        "samples": [result.to_dict() for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
