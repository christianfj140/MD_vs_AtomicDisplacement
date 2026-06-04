#!/usr/bin/env python3
"""Generate numeric DeepH raw/global equivalence evidence.

The preflight compares DeepH processed global HDF5 reference blocks against the
raw SIESTA/Graph2Mat reference matrix for selected frozen samples. It writes
``raw_global_equivalence_evidence.json`` files in the DeepH prediction work
directories so the existing DeepH adapter can consume them fail-closed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from deeph_prediction_adapter import (
    RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME,
    RAW_GLOBAL_EQUIVALENCE_EVIDENCE_SCHEMA,
    RAW_GLOBAL_EQUIVALENCE_REQUIRED_CHECKS,
    assemble_hk,
    count_orbitals_from_orbital_types,
    file_sha256,
    h5_block_shapes,
    validate_raw_global_equivalence_evidence,
)


PREFLIGHT_SCHEMA = "deeph_raw_global_equivalence_preflight_v1"
DEFAULT_MATRIX_TOLERANCE = 1e-6
DEFAULT_EIGENVALUE_TOLERANCE = 3e-6
SUPPORT_THRESHOLD = 1e-12
FORBIDDEN_REFERENCE_NAME = "ML_prediction.HSX"
SUPPORTED_ORBITAL_LABELS = {"s", "px", "py", "pz"}
SIESTA_TO_DEEPH_ORBITAL_SIGNS = {
    "s": 1.0,
    "px": -1.0,
    "py": -1.0,
    "pz": 1.0,
}


class DeepHEquivalencePreflightError(RuntimeError):
    """Raised when numeric equivalence evidence cannot be generated."""


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DeepHEquivalencePreflightError(f"JSON file must contain an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(value, np.generic):
            return json_safe(value.item())
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
    except ImportError:
        pass
    return value


def safe_sample_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._") or "sample"


def artifact_path(row: dict[str, Any], *keys: str, manifest_dir: Path) -> Path | None:
    paths = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    for key in keys:
        raw = paths.get(key) or row.get(f"{key}_path")
        if raw in (None, ""):
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = manifest_dir / path
        return path
    return None


def normalize_orbital_label(*, sym: str, angular_l: int, angular_m: int) -> str:
    label = str(sym or "").strip().lower()
    if label in SUPPORTED_ORBITAL_LABELS:
        return label
    if angular_l == 0:
        return "s"
    if angular_l == 1:
        # SIESTA ORB_INDX commonly stores the real p orbitals as m=-1,0,+1
        # with labels py,pz,px. Prefer the explicit label above when present.
        return {-1: "py", 0: "pz", 1: "px"}.get(int(angular_m), f"p{angular_m}")
    return label or f"l{angular_l}_m{angular_m}"


def siesta_orbital_labels_from_orb_indx(path: Path, *, expected_count: int) -> list[str]:
    if not path.exists():
        raise DeepHEquivalencePreflightError(f"missing SIESTA ORB_INDX for orbital mapping: {path}")
    orbitals: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = [token for token in line.split() if token.strip()]
        if len(tokens) < 16:
            continue
        try:
            int(tokens[0])
            angular_l = int(tokens[6])
            angular_m = int(tokens[7])
            isc = tuple(int(value) for value in tokens[-4:-1])
            unit_cell_index = int(tokens[-1])
        except ValueError:
            continue
        if isc != (0, 0, 0):
            continue
        if unit_cell_index < 1 or unit_cell_index > expected_count:
            continue
        label = normalize_orbital_label(sym=tokens[10], angular_l=angular_l, angular_m=angular_m)
        orbitals.append((unit_cell_index - 1, label))
    orbitals = sorted(dict(orbitals).items())
    labels = [label for _index, label in orbitals]
    if len(labels) != expected_count:
        raise DeepHEquivalencePreflightError(
            f"SIESTA ORB_INDX unit-cell orbital count mismatch in {path}: "
            f"{len(labels)} != {expected_count}"
        )
    unsupported = sorted({label for label in labels if label not in SUPPORTED_ORBITAL_LABELS})
    if unsupported:
        raise DeepHEquivalencePreflightError(
            "unsupported SIESTA orbital label(s) for DeepH raw/global mapping: " + ", ".join(unsupported)
        )
    return labels


def deeph_orbital_labels_from_orbital_types(path: Path) -> list[str]:
    if not path.exists():
        raise DeepHEquivalencePreflightError(f"missing DeepH orbital_types.dat for orbital mapping: {path}")
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = [token for token in line.split() if token.strip()]
        for token in tokens:
            angular_l = int(token)
            if angular_l == 0:
                labels.append("s")
            elif angular_l == 1:
                # DeepH rotate_back writes p blocks in OpenMX xyz order.
                labels.extend(["px", "py", "pz"])
            else:
                raise DeepHEquivalencePreflightError(
                    f"unsupported DeepH angular momentum l={angular_l} in {path}; "
                    "raw/global mapping is currently verified only for s/p orbitals"
                )
    if not labels:
        raise DeepHEquivalencePreflightError(f"no DeepH orbital labels found in {path}")
    return labels


def derive_deeph_to_siesta_basis_transform(
    *,
    row: dict[str, Any],
    manifest_dir: Path,
    orbital_types: Path,
    n_orbitals: int,
) -> dict[str, Any]:
    orb_indx = artifact_path(row, "orb_indx", manifest_dir=manifest_dir)
    if orb_indx is None or not orb_indx.exists():
        return {
            "status": "identity_missing_orb_indx",
            "orb_indx_path": str(orb_indx or ""),
            "permutation": list(range(n_orbitals)),
            "signs": [1.0] * n_orbitals,
            "siesta_orbital_labels": [],
            "deeph_orbital_labels": [],
            "warnings": ["missing ORB_INDX; DeepH/SIESTA orbital mapping was left as identity"],
        }
    siesta_labels = siesta_orbital_labels_from_orb_indx(orb_indx, expected_count=n_orbitals)
    deeph_labels = deeph_orbital_labels_from_orbital_types(orbital_types)
    if len(deeph_labels) != n_orbitals:
        raise DeepHEquivalencePreflightError(
            f"DeepH orbital_types count mismatch in {orbital_types}: {len(deeph_labels)} != {n_orbitals}"
        )

    used: set[int] = set()
    permutation: list[int] = []
    for label in siesta_labels:
        try:
            source_index = next(
                index for index, source_label in enumerate(deeph_labels) if index not in used and source_label == label
            )
        except StopIteration as exc:
            raise DeepHEquivalencePreflightError(
                "cannot map DeepH orbital order to SIESTA ORB_INDX order; missing label " + label
            ) from exc
        used.add(source_index)
        permutation.append(source_index)
    signs = [float(SIESTA_TO_DEEPH_ORBITAL_SIGNS[label]) for label in siesta_labels]
    return {
        "status": "applied",
        "orb_indx_path": str(orb_indx),
        "permutation": permutation,
        "signs": signs,
        "siesta_orbital_labels": siesta_labels,
        "deeph_orbital_labels": deeph_labels,
        "sign_policy": "SIESTA real-orbital signs matched to DeepH/OpenMX rotate_back convention",
        "warnings": [],
    }


def apply_basis_transform(matrix: Any, transform: dict[str, Any]) -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - guarded by runtime helper in numeric path.
        raise DeepHEquivalencePreflightError("numpy is required for basis transforms") from exc
    permutation = [int(index) for index in transform.get("permutation") or []]
    signs = np.asarray([float(value) for value in transform.get("signs") or []], dtype=float)
    if not permutation:
        return matrix
    transformed = np.asarray(matrix)[np.ix_(permutation, permutation)]
    if signs.size != transformed.shape[0]:
        raise DeepHEquivalencePreflightError("basis transform sign count does not match matrix shape")
    return (signs[:, None] * transformed) * signs[None, :]


def fit_energy_reference_shift(*, deeph_h: Any, raw_h: Any, raw_s: Any) -> float:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - guarded by runtime helper in numeric path.
        raise DeepHEquivalencePreflightError("numpy is required for energy reference fitting") from exc
    residual = (np.asarray(raw_h) - np.asarray(deeph_h)).reshape(-1)
    overlap = np.asarray(raw_s).reshape(-1)
    denominator = np.vdot(overlap, overlap)
    if abs(denominator) == 0.0:
        raise DeepHEquivalencePreflightError("cannot fit DeepH energy reference shift: zero overlap norm")
    shift = np.vdot(overlap, residual) / denominator
    if abs(float(np.imag(shift))) > 1e-10:
        raise DeepHEquivalencePreflightError(
            f"DeepH energy reference shift has non-negligible imaginary component: {shift}"
        )
    return float(np.real(shift))


def sample_id_from_row(row: dict[str, Any]) -> str:
    sample_dir = Path(str(row.get("sample_dir") or ""))
    return str(row.get("sample_id") or row.get("graph2mat_sample_id") or row.get("deeph_sample_id") or sample_dir.name)


def select_frozen_rows(
    rows: list[dict[str, Any]],
    *,
    sample_ids: list[str],
    sample_limit: int,
) -> list[dict[str, Any]]:
    if sample_limit <= 0:
        raise DeepHEquivalencePreflightError("--sample-limit must be positive")
    by_id = {sample_id_from_row(row): row for row in rows}
    if sample_ids:
        missing = [sample for sample in sample_ids if sample not in by_id]
        if missing:
            raise DeepHEquivalencePreflightError("Frozen split manifest is missing requested samples: " + ", ".join(missing))
        return [by_id[sample] for sample in sample_ids]

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        for row in rows:
            if row.get("split") == split:
                sample_id = sample_id_from_row(row)
                selected.append(row)
                seen.add(sample_id)
                break
    for row in rows:
        sample_id = sample_id_from_row(row)
        if sample_id not in seen:
            selected.append(row)
            seen.add(sample_id)
        if len(selected) >= sample_limit:
            break
    return selected[:sample_limit]


def existing_child_for_sample(root: Path, row: dict[str, Any]) -> Path | None:
    sample_aliases = {
        sample_id_from_row(row),
        str(row.get("graph2mat_sample_id") or ""),
        str(row.get("deeph_sample_id") or ""),
    }
    sample_aliases = {sample for sample in sample_aliases if sample}
    for alias in sorted(sample_aliases):
        direct = root / alias
        if direct.exists():
            return direct
    if not root.exists():
        return None
    safe_aliases = {safe_sample_id(alias) for alias in sample_aliases}
    matches = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and any(path.name == alias or path.name.endswith(f"_{alias}") for alias in safe_aliases)
    ]
    return sorted(matches, key=lambda path: path.name)[0] if matches else None


def graph2mat_reference_path(
    *,
    row: dict[str, Any],
    manifest_dir: Path,
    graph2mat_result_dir: Path,
) -> Path | None:
    for path in (
        artifact_path(row, "reference_tshs", "tshs", manifest_dir=manifest_dir),
        artifact_path(row, "reference_hsx", "hsx", manifest_dir=manifest_dir),
    ):
        if path is not None and path.exists():
            if path.name == FORBIDDEN_REFERENCE_NAME:
                raise DeepHEquivalencePreflightError(f"Forbidden Graph2Mat prediction cannot be reference: {path}")
            return path
    sample_id = sample_id_from_row(row)
    for root in (
        graph2mat_result_dir / "siesta_hamiltonians" / sample_id,
        Path(str(row.get("sample_dir") or "")),
    ):
        if not root.exists():
            continue
        for suffix in (".TSHS", ".HSX"):
            matches = sorted(path for path in root.glob(f"*{suffix}") if path.name != FORBIDDEN_REFERENCE_NAME)
            if matches:
                return matches[0]
    return None


def run_fdf_path(
    *,
    row: dict[str, Any],
    manifest_dir: Path,
    graph2mat_result_dir: Path,
) -> Path | None:
    path = artifact_path(row, "run_fdf", manifest_dir=manifest_dir)
    if path is not None and path.exists():
        return path
    sample_id = sample_id_from_row(row)
    for candidate in (
        graph2mat_result_dir / "structures" / sample_id / "RUN.fdf",
        Path(str(row.get("sample_dir") or "")) / "RUN.fdf",
    ):
        if candidate.exists():
            return candidate
    return None


def _runtime_helpers() -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore[import-not-found]
        import sisl  # type: ignore[import-not-found]
        from evaluate_hamiltonian_metrics import (  # noqa: PLC0415
            complex_generalized_eigenvalues,
            kpoint_hamiltonian_matrix,
            kpoint_overlap_matrix,
            parse_monkhorst_pack_kgrid,
        )
    except ImportError as exc:  # pragma: no cover - depends on scientific env.
        raise DeepHEquivalencePreflightError(
            "DeepH raw/global equivalence preflight requires numpy, scipy, sisl and h5py."
        ) from exc
    return {
        "np": np,
        "sisl": sisl,
        "complex_generalized_eigenvalues": complex_generalized_eigenvalues,
        "kpoint_hamiltonian_matrix": kpoint_hamiltonian_matrix,
        "kpoint_overlap_matrix": kpoint_overlap_matrix,
        "parse_monkhorst_pack_kgrid": parse_monkhorst_pack_kgrid,
    }


def raw_reference_matrices(reference_path: Path, kpoint: tuple[float, float, float]) -> dict[str, Any]:
    helpers = _runtime_helpers()
    np = helpers["np"]
    sile = helpers["sisl"].get_sile(str(reference_path))
    hamiltonian = sile.read_hamiltonian()
    h_k = helpers["kpoint_hamiltonian_matrix"](hamiltonian, kpoint)
    s_k = helpers["kpoint_overlap_matrix"](hamiltonian, kpoint)
    if s_k is None:
        raise DeepHEquivalencePreflightError(f"Reference overlap S(k) is unavailable for {reference_path}")
    return {
        "hamiltonian": np.asarray(h_k, dtype=np.complex128),
        "overlap": np.asarray(s_k, dtype=np.complex128),
        "spin": str(getattr(hamiltonian, "spin", "")) or "",
        "orthogonal": bool(getattr(hamiltonian, "orthogonal", False)),
    }


def kpoints_from_fdf(path: Path | None) -> tuple[list[tuple[float, float, float]], list[str]]:
    warnings: list[str] = []
    if path is None or not path.exists():
        warnings.append("missing RUN.fdf; only Gamma was checked.")
        return [(0.0, 0.0, 0.0)], warnings
    helpers = _runtime_helpers()
    kgrid = helpers["parse_monkhorst_pack_kgrid"](path)
    kpoints: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    if kgrid is None:
        warnings.append("RUN.fdf has no Monkhorst-Pack grid; Gamma was checked.")
        return kpoints, warnings
    if not kgrid.ok:
        warnings.append(f"RUN.fdf k-grid could not be parsed ({kgrid.error}); Gamma was checked.")
        return kpoints, warnings
    for kpoint in kgrid.fractional_kpoints:
        item = tuple(float(value) for value in kpoint)
        if item not in kpoints:
            kpoints.append(item)
    return kpoints, warnings


def _pass_check(status: bool, message: str = "") -> dict[str, Any]:
    return {"status": "pass" if status else "fail", "message": message}


def failed_evidence(
    *,
    sample_id: str,
    frozen_sample_id: str,
    reason: str,
    warnings: list[str] | None = None,
    source_files: dict[str, Any] | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": RAW_GLOBAL_EQUIVALENCE_EVIDENCE_SCHEMA,
        "sample_id": sample_id,
        "frozen_sample_id": frozen_sample_id,
        "equivalence_status": "failed",
        "equivalence_scope": "raw_global",
        "checks": {key: _pass_check(False, reason) for key in RAW_GLOBAL_EQUIVALENCE_REQUIRED_CHECKS},
        "errors": {},
        "tolerances": {},
        "source_files": source_files or {},
        "kpoints_checked": [],
        "generator": {
            "script": Path(__file__).name,
            "command": command or sys.argv,
        },
        "warnings": list(warnings or []),
        "failure_reason": reason,
    }


def file_entry(path: Path | None) -> dict[str, Any]:
    return {
        "path": str(path) if path is not None else "",
        "sha256": file_sha256(path) if path is not None else None,
    }


def numeric_evidence_for_sample(
    *,
    row: dict[str, Any],
    manifest_dir: Path,
    graph2mat_result_dir: Path,
    processed_dir: Path,
    predictions_dir: Path,
    matrix_tolerance: float,
    eigenvalue_tolerance: float,
    command: list[str] | None,
) -> tuple[dict[str, Any], Path | None, Path]:
    frozen_sample_id = sample_id_from_row(row)
    processed_sample = existing_child_for_sample(processed_dir, row)
    prediction_sample = existing_child_for_sample(predictions_dir, row)
    adapter_sample_id = prediction_sample.name if prediction_sample is not None else frozen_sample_id
    output_name = safe_sample_id(adapter_sample_id)
    source_files: dict[str, Any] = {}
    warnings: list[str] = []

    reference_path = graph2mat_reference_path(
        row=row,
        manifest_dir=manifest_dir,
        graph2mat_result_dir=graph2mat_result_dir,
    )
    fdf_path = run_fdf_path(row=row, manifest_dir=manifest_dir, graph2mat_result_dir=graph2mat_result_dir)
    source_files["raw_reference"] = file_entry(reference_path)
    source_files["run_fdf"] = file_entry(fdf_path)
    if reference_path is None:
        return (
            failed_evidence(
                sample_id=adapter_sample_id,
                frozen_sample_id=frozen_sample_id,
                reason="missing raw SIESTA/Graph2Mat reference HSX/TSHS",
                source_files=source_files,
                command=command,
            ),
            prediction_sample,
            Path(output_name),
        )
    if processed_sample is None:
        return (
            failed_evidence(
                sample_id=adapter_sample_id,
                frozen_sample_id=frozen_sample_id,
                reason="missing DeepH processed sample mapping",
                source_files=source_files,
                command=command,
            ),
            prediction_sample,
            Path(output_name),
        )
    if prediction_sample is None:
        return (
            failed_evidence(
                sample_id=adapter_sample_id,
                frozen_sample_id=frozen_sample_id,
                reason="missing DeepH prediction sample mapping",
                source_files=source_files,
                command=command,
            ),
            prediction_sample,
            Path(output_name),
        )

    ref_h5 = processed_sample / "hamiltonians.h5"
    overlap_h5 = processed_sample / "overlaps.h5"
    orbital_types = processed_sample / "orbital_types.dat"
    prediction_h5 = prediction_sample / "hamiltonians_pred.h5"
    info_json = processed_sample / "info.json"
    orb_indx_path = artifact_path(row, "orb_indx", manifest_dir=manifest_dir)
    source_files.update(
        {
            "deeph_processed_hamiltonian": file_entry(ref_h5),
            "deeph_processed_overlap": file_entry(overlap_h5),
            "deeph_orbital_types": file_entry(orbital_types),
            "deeph_prediction": file_entry(prediction_h5),
            "deeph_info": file_entry(info_json),
            "siesta_orb_indx": file_entry(orb_indx_path),
        }
    )
    missing = [
        str(path)
        for path in (ref_h5, overlap_h5, orbital_types, prediction_h5, info_json)
        if not path.exists()
    ]
    if missing:
        return (
            failed_evidence(
                sample_id=adapter_sample_id,
                frozen_sample_id=frozen_sample_id,
                reason="missing required DeepH artifact(s): " + ", ".join(missing),
                source_files=source_files,
                command=command,
            ),
            prediction_sample,
            Path(output_name),
        )

    try:
        helpers = _runtime_helpers()
        np = helpers["np"]
        eigenvalues = helpers["complex_generalized_eigenvalues"]
        orbital_counts = count_orbitals_from_orbital_types(orbital_types)
        n_orbitals = int(sum(orbital_counts))
        basis_transform = derive_deeph_to_siesta_basis_transform(
            row=row,
            manifest_dir=manifest_dir,
            orbital_types=orbital_types,
            n_orbitals=n_orbitals,
        )
        warnings.extend(str(item) for item in basis_transform.get("warnings") or [])
        ref_shapes = h5_block_shapes(ref_h5, orbital_counts, label="DeepH processed reference")
        pred_shapes = h5_block_shapes(prediction_h5, orbital_counts, label="DeepH prediction")
        overlap_shapes = h5_block_shapes(overlap_h5, orbital_counts, label="DeepH processed overlap")
        support_keys_match = set(ref_shapes) == set(pred_shapes) == set(overlap_shapes)
        kpoints, k_warnings = kpoints_from_fdf(fdf_path)
        warnings.extend(k_warnings)
        gamma_kpoint = kpoints[0]
        gamma_raw = raw_reference_matrices(reference_path, gamma_kpoint)
        gamma_deeph_h = apply_basis_transform(assemble_hk(ref_h5, orbital_types, gamma_kpoint), basis_transform)
        energy_reference_shift_eV = fit_energy_reference_shift(
            deeph_h=gamma_deeph_h,
            raw_h=gamma_raw["hamiltonian"],
            raw_s=gamma_raw["overlap"],
        )
        max_hk_error = 0.0
        max_s_error = 0.0
        max_eigen_error = 0.0
        support_match = True
        shape_match = True
        kpoint_rows: list[dict[str, float]] = []
        for kpoint in kpoints:
            raw = raw_reference_matrices(reference_path, kpoint)
            raw_h = raw["hamiltonian"]
            raw_s = raw["overlap"]
            deeph_h = apply_basis_transform(assemble_hk(ref_h5, orbital_types, kpoint), basis_transform)
            deeph_s = apply_basis_transform(assemble_hk(overlap_h5, orbital_types, kpoint), basis_transform)
            shape_match = shape_match and raw_h.shape == deeph_h.shape and raw_s.shape == deeph_s.shape
            if not shape_match:
                continue
            deeph_h_aligned = np.asarray(deeph_h + energy_reference_shift_eV * raw_s)
            h_delta = np.asarray(deeph_h_aligned - raw_h)
            s_delta = np.asarray(deeph_s - raw_s)
            max_hk_error = max(max_hk_error, float(np.max(np.abs(h_delta))) if h_delta.size else 0.0)
            max_s_error = max(max_s_error, float(np.max(np.abs(s_delta))) if s_delta.size else 0.0)
            raw_eig = eigenvalues(raw_h, raw_s)
            deeph_eig = eigenvalues(deeph_h_aligned, raw_s)
            if raw_eig.shape != deeph_eig.shape:
                shape_match = False
                continue
            eig_delta = np.asarray(deeph_eig - raw_eig)
            max_eigen_error = max(max_eigen_error, float(np.max(np.abs(eig_delta))) if eig_delta.size else 0.0)
            support_match = support_match and bool(
                np.array_equal(np.abs(raw_h) > SUPPORT_THRESHOLD, np.abs(deeph_h_aligned) > SUPPORT_THRESHOLD)
            )
            kpoint_rows.append({"kx": float(kpoint[0]), "ky": float(kpoint[1]), "kz": float(kpoint[2])})
        hk_pass = shape_match and max_hk_error <= matrix_tolerance
        s_pass = shape_match and max_s_error <= matrix_tolerance
        eig_pass = shape_match and max_eigen_error <= eigenvalue_tolerance
        support_pass = bool(shape_match and support_keys_match and support_match)
        info_payload = read_json(info_json)
        spin_pass = "isspinful" in info_payload
        if not spin_pass:
            warnings.append(f"DeepH info.json does not expose isspinful: {info_json}")
        checks = {
            "shape": _pass_check(shape_match, "raw and DeepH H(k)/S(k) shapes match" if shape_match else "matrix shape mismatch"),
            "units": _pass_check(hk_pass, "DeepH processed Hamiltonian numerically matches raw reference units"),
            "orbital_order": _pass_check(hk_pass and s_pass, "numeric H(k)/S(k) equality proves orbital ordering for checked k-points"),
            "atom_order": _pass_check(hk_pass and s_pass, "block assembly numerically matches raw reference for checked k-points"),
            "r_vectors": _pass_check(hk_pass and s_pass, "DeepH R-vector phases reproduce raw reference H(k)/S(k)"),
            "spin": _pass_check(spin_pass, "DeepH spin metadata is present"),
            "sparse_support": _pass_check(support_pass, "DeepH block and dense support match raw reference"),
            "hk": _pass_check(hk_pass, "DeepH processed H(k) matches raw reference"),
            "s_ref": _pass_check(s_pass, "DeepH processed S(k) matches raw reference overlap"),
            "eigenvalues": _pass_check(eig_pass, "generalized eigenvalues match with S_ref(k)"),
        }
        errors = {
            "max_abs_hk_error_eV": max_hk_error,
            "max_abs_s_ref_error": max_s_error,
            "max_abs_eigenvalue_error_eV": max_eigen_error,
            "energy_reference_shift_eV": energy_reference_shift_eV,
        }
        tolerances = {
            "max_abs_hk_error_eV": matrix_tolerance,
            "max_abs_s_ref_error": matrix_tolerance,
            "max_abs_eigenvalue_error_eV": eigenvalue_tolerance,
        }
        proven = all(str(check["status"]) == "pass" for check in checks.values())
    except Exception as exc:
        return (
            failed_evidence(
                sample_id=adapter_sample_id,
                frozen_sample_id=frozen_sample_id,
                reason=str(exc),
                warnings=warnings,
                source_files=source_files,
                command=command,
            ),
            prediction_sample,
            Path(output_name),
        )

    payload = {
        "schema": RAW_GLOBAL_EQUIVALENCE_EVIDENCE_SCHEMA,
        "sample_id": adapter_sample_id,
        "frozen_sample_id": frozen_sample_id,
        "equivalence_status": "proven" if proven else "failed",
        "equivalence_scope": "raw_global",
        "checks": checks,
        "errors": errors,
        "tolerances": tolerances,
        "source_files": source_files,
        "kpoints_checked": kpoint_rows,
        "basis_transform": basis_transform,
        "energy_reference_alignment": {
            "policy": "least_squares_shift_from_first_kpoint: H_raw ~= H_deeph_converted + c*S_ref",
            "shift_eV": energy_reference_shift_eV,
            "fit_kpoint": {
                "kx": float(gamma_kpoint[0]),
                "ky": float(gamma_kpoint[1]),
                "kz": float(gamma_kpoint[2]),
            },
        },
        "generator": {
            "script": Path(__file__).name,
            "command": command or sys.argv,
            "comparison_policy": "DeepH processed hamiltonians.h5/overlaps.h5 versus raw SIESTA HSX/TSHS reference",
        },
        "warnings": warnings,
    }
    return payload, prediction_sample, Path(output_name)


def install_evidence(payload: dict[str, Any], output_path: Path, prediction_sample: Path | None) -> dict[str, str]:
    write_json(output_path, payload)
    installed = {"output": str(output_path)}
    if prediction_sample is not None:
        adapter_path = prediction_sample / RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME
        if adapter_path.resolve(strict=False) != output_path.resolve(strict=False):
            write_json(adapter_path, payload)
        installed["adapter_discoverable"] = str(adapter_path)
    return installed


def build_preflight_manifest(
    *,
    frozen_split_manifest: Path,
    graph2mat_result_dir: Path,
    deeph_processed_dir: Path,
    deeph_predictions_dir: Path,
    output_dir: Path,
    sample_limit: int,
    sample_ids: list[str] | None = None,
    matrix_tolerance: float = DEFAULT_MATRIX_TOLERANCE,
    eigenvalue_tolerance: float = DEFAULT_EIGENVALUE_TOLERANCE,
    command: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = read_json(frozen_split_manifest)
    rows = [row for row in frozen.get("rows") or [] if isinstance(row, dict)]
    selected = select_frozen_rows(rows, sample_ids=list(sample_ids or []), sample_limit=sample_limit)
    sample_results: list[dict[str, Any]] = []
    for row in selected:
        evidence, prediction_sample, output_name = numeric_evidence_for_sample(
            row=row,
            manifest_dir=frozen_split_manifest.parent,
            graph2mat_result_dir=graph2mat_result_dir,
            processed_dir=deeph_processed_dir,
            predictions_dir=deeph_predictions_dir,
            matrix_tolerance=matrix_tolerance,
            eigenvalue_tolerance=eigenvalue_tolerance,
            command=command,
        )
        evidence_dir = output_dir / output_name.name
        evidence_path = evidence_dir / RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME
        paths = install_evidence(evidence, evidence_path, prediction_sample)
        adapter_validation = validate_raw_global_equivalence_evidence(
            Path(paths["adapter_discoverable"] if "adapter_discoverable" in paths else paths["output"]),
            sample_id=str(evidence.get("sample_id") or ""),
        )
        sample_results.append(
            {
                "sample_id": evidence.get("sample_id"),
                "frozen_sample_id": evidence.get("frozen_sample_id"),
                "status": evidence.get("equivalence_status"),
                "equivalence_scope": evidence.get("equivalence_scope"),
                "evidence_paths": paths,
                "adapter_validation_status": adapter_validation.get("status"),
                "adapter_validation_reason": adapter_validation.get("reason"),
                "warnings": evidence.get("warnings") or [],
            }
        )
    proven_count = sum(1 for row in sample_results if row.get("status") == "proven")
    failed_count = len(sample_results) - proven_count
    aggregate = {
        "schema": PREFLIGHT_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frozen_split_manifest": str(frozen_split_manifest),
        "graph2mat_result_dir": str(graph2mat_result_dir),
        "deeph_processed_dir": str(deeph_processed_dir),
        "deeph_predictions_dir": str(deeph_predictions_dir),
        "output_dir": str(output_dir),
        "sample_limit": sample_limit,
        "requested_sample_ids": list(sample_ids or []),
        "matrix_tolerance": matrix_tolerance,
        "eigenvalue_tolerance": eigenvalue_tolerance,
        "samples_seen": len(selected),
        "samples_proven": proven_count,
        "samples_failed": failed_count,
        "status": "proven" if sample_results and failed_count == 0 else "failed",
        "samples": sample_results,
    }
    write_json(output_dir / "deeph_raw_global_equivalence_preflight.json", aggregate)
    return aggregate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-split-manifest", type=Path, required=True)
    parser.add_argument("--graph2mat-result-dir", type=Path, required=True)
    parser.add_argument("--deeph-processed-dir", type=Path, required=True)
    parser.add_argument("--deeph-predictions-dir", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-tolerance", type=float, default=DEFAULT_MATRIX_TOLERANCE)
    parser.add_argument("--eigenvalue-tolerance", type=float, default=DEFAULT_EIGENVALUE_TOLERANCE)
    parser.add_argument("--fail-closed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_preflight_manifest(
        frozen_split_manifest=args.frozen_split_manifest,
        graph2mat_result_dir=args.graph2mat_result_dir,
        deeph_processed_dir=args.deeph_processed_dir,
        deeph_predictions_dir=args.deeph_predictions_dir,
        output_dir=args.output_dir,
        sample_limit=int(args.sample_limit),
        sample_ids=list(args.sample_id or []),
        matrix_tolerance=float(args.matrix_tolerance),
        eigenvalue_tolerance=float(args.eigenvalue_tolerance),
        command=[Path(__file__).name, *(argv or sys.argv[1:])],
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True, ensure_ascii=False))
    if args.fail_closed and manifest.get("status") != "proven":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
