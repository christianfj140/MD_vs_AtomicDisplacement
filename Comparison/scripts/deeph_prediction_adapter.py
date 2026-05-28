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
PROVEN_ADAPTER_EQUIVALENCE_STATUSES = {EQUIVALENCE_PROVEN_RAW_GLOBAL}


class DeepHPredictionAdapterError(RuntimeError):
    """Raised when a DeepH prediction cannot be safely adapted."""


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
    adapter_name: str = ADAPTER_NAME
    adapter_version: str = ADAPTER_VERSION
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def metric_fields(self) -> dict[str, Any]:
        return {
            "prediction_adapter": self.adapter_name,
            "prediction_adapter_version": self.adapter_version,
            "deeph_comparability_status": self.comparability_status,
            "deeph_adapter_equivalence_status": self.adapter_equivalence_status,
            "deeph_raw_global_equivalence_proven": self.adapter_equivalence_status
            in PROVEN_ADAPTER_EQUIVALENCE_STATUSES,
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
    warnings = [
        "DeepH HDF5 blocks were validated against DeepH processed SIESTA HDF5 artifacts only.",
        "Equivalence to Graph2Mat raw HSX orbital order/sign convention is not independently proven.",
        f"{EQUIVALENCE_INVALID_UNITS}: DeepH processed energy units were not independently checked against Graph2Mat HSX.",
        f"{EQUIVALENCE_INVALID_R_VECTOR}: DeepH HDF5 R-vector convention was not independently checked against Graph2Mat HSX.",
    ]
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
        adapter_equivalence_status=EQUIVALENCE_INVALID_ORBITAL_ORDER,
        target_space="deeph_rotate_back_global_hamiltonian_h5_blocks",
        units_status="deeph_siesta_preprocess_internal_energy_units_unverified_against_graph2mat_hsx",
        orbital_order_status="validated_against_deeph_processed_reference_only",
        r_vector_convention_status="validated_against_deeph_processed_reference_only",
        support_semantics_status="prediction_and_processed_reference_key_sets_match",
        provenance=_provenance(
            prediction_path=prediction_path,
            reference_hamiltonian=reference_hamiltonian,
            reference_overlap=reference_overlap,
            orbital_types=orbital_types,
        ),
        warnings=warnings,
    )


def write_adapter_manifest(path: Path, results: list[DeepHPredictionAdapterResult]) -> dict[str, Any]:
    equivalence_statuses = sorted({result.adapter_equivalence_status for result in results})
    proven_count = sum(
        1 for result in results if result.adapter_equivalence_status in PROVEN_ADAPTER_EQUIVALENCE_STATUSES
    )
    payload = {
        "schema": ADAPTER_VERSION,
        "adapter_name": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "sample_count": len(results),
        "metrics_ready_count": sum(1 for result in results if result.metrics_ready),
        "diagnostic_only_count": sum(1 for result in results if result.diagnostic_only),
        "adapter_equivalence_statuses": equivalence_statuses,
        "raw_global_equivalence_proven_count": proven_count,
        "robust_matrix_metrics_allowed": bool(results) and proven_count == len(results),
        "samples": [result.to_dict() for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
