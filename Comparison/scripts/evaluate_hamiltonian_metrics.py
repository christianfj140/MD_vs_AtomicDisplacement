#!/usr/bin/env python3
"""Evaluate sparse, spectral and total-DOS metrics for archived Hamiltonians."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg
import scipy.stats
from scipy import sparse
import sisl
import yaml

from reference_selection import REFERENCE_SELECTION_POLICY
from reference_selection import choose_reference_matrix
from reference_selection import file_sha256


SUPPORT_THRESHOLD = 1e-12
SUPPORT_THRESHOLDS_SWEEP = [1e-12, 1e-10, 1e-8, 1e-6]
FERMI_WINDOW_EV = 2.0
DOS_SIGMA_EV = 0.10
DOS_SIGMA_SWEEP_EV = [0.05, 0.10, 0.20, 0.40]
DOS_POINTS = 1000
DOS_FERMI_WINDOW_POINTS = 500
DOS_FERMI_WINDOW_MIN_EV = -6.0
DOS_FERMI_WINDOW_MAX_EV = 6.0
DOS_FERMI_WINDOW_ALIGNMENT = "reference_fermi_level"
LOW_ENERGY_N_STATES = 10
LOW_ENERGY_ALIGNMENT = "none"
COMPLEX_IMAG_TOLERANCE = 1e-12
OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD = 1e-6
MATRIX_METRIC_TARGET_SPACE = "raw_global_hamiltonian"
ORBITAL_PAIR_METRIC_TARGET_SPACE = "raw_global_hamiltonian_orbital_basis"
ORBITAL_PAIR_BASIS_SOURCE = "ion_xml_pao_degeneracy_generated_labels"
MATRIX_SEMANTIC_FIELDS = [
    "target_component_policy",
    "reference_component_count",
    "prediction_component_count",
    "reference_spin_kind",
    "prediction_spin_kind",
    "overlap_source",
    "prediction_own_overlap_used",
    "prediction_overlap_relative_frobenius_vs_reference",
    "prediction_overlap_check_threshold",
    "graph2mat_auxiliary_component_ignored",
    "prediction_self_contained_hsx_safe",
    "prediction_self_contained_hsx_unsafe_reason",
]
DEEPH_COMPARABILITY_STATUS = {
    "implemented_repo_compatible_metrics": [
        "hamiltonian_mae_rmse_mse_r2_on_repository_supports",
        "hamiltonian_mev_aliases",
        "dos_mae_500_fermi_window_when_reference_fermi_exists",
        "orbital_pair_metrics_csv_when_basis_mapping_exists",
    ],
    "caveats": [
        "matrix_metrics_use_raw_global_hamiltonian_not_deeph_hprime",
        "orbital_pair_metrics_use_repository_orbital_basis_not_deeph_local_hprime_blocks",
        "dos_units_and_system_dimensionality_may_not_match_deeph_2d_examples",
        "fermi_dependent_metrics_are_unavailable_when_siesta_fermi_is_missing",
    ],
    "future_work_not_implemented": {
        "high_symmetry_kpath_band_structure": (
            "requires explicit k-path input, k-resolved reference/predicted bands, "
            "and validation against SIESTA band-structure outputs"
        ),
        "soc_complex_hamiltonians": (
            "current compatibility gates reject complex Hamiltonians and unvalidated "
            "spin-orbit or multi-component matrix semantics"
        ),
        "optical_berry_susceptibility_shift_current": (
            "requires optical/Berry-response infrastructure, validated wavefunction "
            "or velocity/dipole data, and material-specific scientific checks"
        ),
        "ensemble_uncertainty": (
            "requires an explicit ensemble protocol and calibrated uncertainty "
            "validation across independent model instances"
        ),
        "deeph_vs_dft_system_size_scaling": (
            "requires controlled system-size series, DFT/DeepH timing protocol, "
            "and hardware-normalized scaling analysis"
        ),
    },
}
PERIODIC_STRUCTURE_TYPES = {"bulk", "crystal", "periodic", "solid", "surface", "slab"}
UNSUPPORTED_KPOINT_DIRECTIVES = {
    "kgrid_cutoff",
    "kgridcutoff",
    "kgrid_monkhorst_pack",
    "kgrid.monkhorstpack",
}
KGRID_MONKHORST_PACK_DIRECTIVES = {"kgrid_monkhorst_pack", "kgrid.monkhorstpack"}
RECOMMENDATION_PRIMARY_METRIC_PRIORITY = [
    "low_energy_rmse_eV",
    "frontier_window_rmse_eV",
    "occupied_rmse_eV",
    "relative_frobenius_union",
    "dos_wasserstein_eV",
]
DIAGNOSTIC_ONLY_RECOMMENDATION_METRICS = [
    "global_rmse_eV",
    "global_mae_eV",
    "support_precision",
    "support_recall",
    "false_zeros",
    "false_nonzeros",
    "hermiticity",
]


@dataclass
class MatrixData:
    path: Path
    hamiltonian: sparse.csr_matrix
    overlap: sparse.csr_matrix | None
    own_eigenvalues: np.ndarray
    fermi_level: float | None
    fermi_level_source: str | None
    orthogonal: bool
    has_overlap: bool
    overlap_error: str | None
    sha256: str | None = None
    component_count: int = 1
    spin_kind: str | None = None
    components: tuple[sparse.csr_matrix, ...] = ()


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_dirs(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}
    return {
        path.name: path
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_dir()
    }


def matrix_sort_key(path: Path) -> tuple[int, str]:
    numbers: list[int] = []
    for chunk in path.stem.replace("-", ".").replace("_", ".").split("."):
        if chunk.isdigit():
            numbers.append(int(chunk))
    return (numbers[-1] if numbers else 10**9, path.name)


def find_prediction(sample_dir: Path) -> Path | None:
    direct = sample_dir / "ML_prediction.HSX"
    if direct.exists():
        return direct
    matches = sorted(sample_dir.glob("*ML_prediction*.HSX"), key=matrix_sort_key)
    return matches[0] if matches else None


def sample_status_row(
    sample: str,
    *,
    prediction_path: Path | None,
    reference_path: Path | None,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    status = "failed" if errors else "warning" if warnings else "ok"
    return {
        "sample": sample,
        "status": status,
        "prediction_path": str(prediction_path) if prediction_path else None,
        "prediction_sha256": file_sha256(prediction_path),
        "reference_path": str(reference_path) if reference_path else None,
        "reference_sha256": file_sha256(reference_path),
        "reference_kind": reference_path.suffix if reference_path else None,
        "errors": errors,
        "warnings": warnings,
    }


def append_issue(
    rows: dict[str, list[dict[str, Any]]],
    target: str,
    *,
    sample: str,
    kind: str,
    message: str,
    **extra: Any,
) -> dict[str, Any]:
    issue = {"sample": sample, "kind": kind, "error": message, **extra}
    rows[target].append(issue)
    return issue


def result_method_id(result_dir: Path) -> str:
    fallback = result_method_id_from_path(result_dir)
    manifest_path = result_dir / "manifest.json"
    if not manifest_path.exists():
        return fallback
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    if not isinstance(payload, dict):
        return fallback
    method_id = str(payload.get("method_id") or payload.get("pipeline") or "").strip().lower()
    return method_id or fallback


def result_method_id_from_path(result_dir: Path) -> str:
    parts = {part.lower() for part in result_dir.parts}
    if "results_md" in parts:
        return "md"
    if "results_random_cartesian" in parts:
        return "random_cartesian"
    if "results_atomdisp" in parts:
        return "siesta_fc_cartesian"
    return ""


def _nested_mapping_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def target_component_policy_from_result(result_dir: Path) -> str:
    for path in (
        result_dir / "pipeline_config.yaml",
        result_dir / "manifest.json",
        result_dir / "metrics" / "manifest.json",
    ):
        if not path.exists():
            continue
        try:
            if path.suffix.lower() in {".yaml", ".yml"}:
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, yaml.YAMLError):
            continue
        if not isinstance(payload, dict):
            continue
        for nested_path in (
            ("training", "data", "matrix_component_policy"),
            ("prediction", "data", "matrix_component_policy"),
            ("testing", "data", "matrix_component_policy"),
            ("graph2mat_config_provenance", "graph2mat", "matrix_component_policy"),
            ("metric_compatibility", "target_component_policy"),
        ):
            value = _nested_mapping_value(payload, nested_path)
            if value not in (None, ""):
                return str(value)
    return "unknown"


def sample_structure_metadata(structure_path: Path) -> dict[str, Any]:
    metadata_path = structure_path.parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def structure_type_from_metadata(structure_path: Path) -> str:
    metadata = sample_structure_metadata(structure_path)
    material = metadata.get("material") if isinstance(metadata.get("material"), dict) else {}
    value = (
        metadata.get("material_structure_type")
        or metadata.get("structure_type")
        or material.get("structure_type")
        or ""
    )
    return str(value or "").strip().lower()


def _strip_fdf_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _unsupported_kpoint_issue(sample: str, structure_path: Path, directive: str) -> dict[str, Any]:
    return {
        "sample": sample,
        "kind": "unsupported_kpoint_sampling",
        "severity": "fatal",
        "error": (
            "K-point sampled Hamiltonian metrics are not implemented by this "
            "evaluator; only gamma/single-matrix comparisons are supported."
        ),
        "structure_path": str(structure_path),
        "directive": directive,
    }


def _float_token(value: str) -> float | None:
    try:
        return float(value.replace("d", "e").replace("D", "E"))
    except ValueError:
        return None


def _is_gamma_monkhorst_pack_rows(rows: list[str]) -> bool:
    matrix: list[list[float]] = []
    for row in rows:
        values = [_float_token(token) for token in row.split()]
        if len(values) < 4 or any(value is None for value in values[:4]):
            return False
        matrix.append([float(value) for value in values[:4] if value is not None])
    if len(matrix) != 3:
        return False
    expected = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    return all(
        math.isclose(matrix[row][col], expected[row][col], rel_tol=0.0, abs_tol=1e-12)
        for row in range(3)
        for col in range(4)
    )


def _is_gamma_monkhorst_pack_inline(tokens: list[str]) -> bool:
    values = [_float_token(token) for token in tokens]
    if not values or any(value is None for value in values):
        return False
    numeric = [float(value) for value in values if value is not None]
    if len(numeric) == 3:
        return all(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12) for value in numeric)
    if len(numeric) >= 6:
        return all(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12) for value in numeric[:3]) and all(
            math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12) for value in numeric[3:6]
        )
    return False


def unsupported_kpoint_issues(sample: str, structure_path: Path) -> list[dict[str, Any]]:
    """Detect k-point sampling forms this evaluator does not validate."""
    if not structure_path.exists():
        return []
    issues: list[dict[str, Any]] = []
    kgrid_block_name: str | None = None
    kgrid_block_rows: list[str] = []
    try:
        lines = structure_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        return [
            {
                "sample": sample,
                "kind": "structure_kpoint_check",
                "severity": "fatal",
                "error": str(exc),
                "structure_path": str(structure_path),
            }
        ]
    for raw_line in lines:
        clean = _strip_fdf_comment(raw_line)
        if not clean:
            continue
        lower = clean.lower()
        parts = lower.split()
        key = parts[0] if parts else ""
        if lower.startswith("%block"):
            block_name = parts[1] if len(parts) > 1 else ""
            if block_name in KGRID_MONKHORST_PACK_DIRECTIVES:
                kgrid_block_name = block_name
                kgrid_block_rows = []
            elif block_name in UNSUPPORTED_KPOINT_DIRECTIVES:
                issues.append(_unsupported_kpoint_issue(sample, structure_path, block_name))
            continue
        if lower.startswith("%endblock"):
            if kgrid_block_name is not None and not _is_gamma_monkhorst_pack_rows(kgrid_block_rows):
                issues.append(_unsupported_kpoint_issue(sample, structure_path, kgrid_block_name))
            kgrid_block_name = None
            kgrid_block_rows = []
            continue
        if kgrid_block_name is not None:
            kgrid_block_rows.append(clean)
            continue
        if key in KGRID_MONKHORST_PACK_DIRECTIVES:
            if not _is_gamma_monkhorst_pack_inline(parts[1:]):
                issues.append(_unsupported_kpoint_issue(sample, structure_path, key))
            break
        if key in UNSUPPORTED_KPOINT_DIRECTIVES:
            issues.append(_unsupported_kpoint_issue(sample, structure_path, key))
            break
    if kgrid_block_name is not None and not _is_gamma_monkhorst_pack_rows(kgrid_block_rows):
        issues.append(_unsupported_kpoint_issue(sample, structure_path, kgrid_block_name))
    return issues


def sample_method_id(structure_path: Path, fallback: str = "") -> str:
    metadata = sample_structure_metadata(structure_path)
    method_id = str(
        metadata.get("method")
        or metadata.get("method_id")
        or metadata.get("source_method")
        or metadata.get("pipeline")
        or ""
    ).strip().lower()
    if method_id == "atom_displacement":
        method_id = "siesta_fc_cartesian"
    return method_id or fallback.strip().lower()


def md_run_fdf_geometry_materialized(structure_path: Path) -> bool:
    metadata = sample_structure_metadata(structure_path)
    if metadata and (
        metadata.get("run_fdf_rewritten_from_xv") is True
        or str(metadata.get("run_fdf_geometry_source") or "").lower() == "siesta.xv"
    ):
        return True
    try:
        text = structure_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "Graph2Mat MD geometry materialized from siesta.XV" in text


def stale_reference_issue(
    sample: str,
    reference_path: Path | None,
    structure_path: Path,
    *,
    method_id: str = "",
) -> dict[str, Any] | None:
    if reference_path is None or not structure_path.exists():
        return None
    try:
        if reference_path.stat().st_mtime < structure_path.stat().st_mtime:
            normalized_method = sample_method_id(structure_path, method_id)
            materialized_from_xv = md_run_fdf_geometry_materialized(structure_path)
            if normalized_method == "md" or materialized_from_xv:
                return {
                    "sample": sample,
                    "kind": "md_post_siesta_run_fdf_mtime",
                    "severity": "warning",
                    "error": (
                        "SIESTA reference matrix is older than RUN.fdf, but this MD "
                        "workflow materializes per-frame RUN.fdf geometries from siesta.XV "
                        "after SIESTA. Timestamp ordering is treated as nonfatal for MD."
                    ),
                    "reference_path": str(reference_path),
                    "structure_path": str(structure_path),
                    "method_id": normalized_method or method_id,
                    "run_fdf_geometry_source": "siesta.XV" if materialized_from_xv else "md_pipeline_assumed",
                }
            return {
                "sample": sample,
                "kind": "stale_reference_matrix",
                "severity": "fatal",
                "error": "SIESTA reference matrix is older than the archived RUN.fdf structure.",
                "reference_path": str(reference_path),
                "structure_path": str(structure_path),
                "method_id": normalized_method or method_id,
            }
    except OSError as exc:
        return {
            "sample": sample,
            "kind": "reference_staleness_check",
            "severity": "fatal",
            "error": str(exc),
            "reference_path": str(reference_path) if reference_path else None,
            "structure_path": str(structure_path),
            "method_id": method_id,
        }
    return None


def evaluate_sample(
    sample: str,
    prediction_dir: Path | None,
    reference_dir: Path | None,
    result_dir: Path,
    basis_counts: dict[str, int],
    *,
    method_id: str = "",
    target_component_policy: str = "unknown",
    low_energy_enabled: bool,
    low_energy_n_states: int,
    low_energy_alignment: str,
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {
        "sparse": [],
        "spectral": [],
        "dos": [],
        "overlap": [],
        "sparse_sweep": [],
        "dos_sweep": [],
        "component": [],
        "block": [],
        "species_pair": [],
        "distance_bin": [],
        "orbital_pair": [],
        "structural_unavailable": [],
        "errors": [],
        "fatal_errors": [],
        "warnings": [],
        "sample_status": [],
    }
    sample_errors: list[dict[str, Any]] = []
    sample_warnings: list[dict[str, Any]] = []
    predicted_path = find_prediction(prediction_dir) if prediction_dir is not None else None
    reference_selection = choose_reference_matrix(reference_dir) if reference_dir is not None else None
    reference_path = reference_selection.path if reference_selection and reference_selection.ok else None
    if predicted_path is None:
        sample_errors.append(
            append_issue(
                rows,
                "fatal_errors",
                sample=sample,
                kind="missing_prediction",
                message="Missing predicted Hamiltonian.",
            )
        )
    if reference_selection is None:
        sample_errors.append(
            append_issue(
                rows,
                "fatal_errors",
                sample=sample,
                kind="missing_reference_dir",
                message="Missing SIESTA reference directory.",
            )
        )
    elif not reference_selection.ok:
        sample_errors.append(
            append_issue(
                rows,
                "fatal_errors",
                sample=sample,
                kind="reference_selection",
                message=reference_selection.reason,
                candidate_count=reference_selection.candidate_count,
                candidates=list(reference_selection.candidates),
            )
        )
    stale_issue = stale_reference_issue(
        sample,
        reference_path,
        result_dir / "structures" / sample / "RUN.fdf",
        method_id=method_id,
    )
    if stale_issue is not None:
        if stale_issue.get("severity") == "warning":
            rows["warnings"].append(stale_issue)
            sample_warnings.append(stale_issue)
        else:
            rows["fatal_errors"].append(stale_issue)
            sample_errors.append(stale_issue)
    structure_path = result_dir / "structures" / sample / "RUN.fdf"
    for issue in unsupported_kpoint_issues(sample, structure_path):
        rows["fatal_errors"].append(issue)
        sample_errors.append(issue)
    if sample_errors:
        rows["errors"].extend(sample_errors)
        rows["sample_status"].append(
            sample_status_row(
                sample,
                prediction_path=predicted_path,
                reference_path=reference_path,
                errors=sample_errors,
                warnings=sample_warnings,
            )
        )
        return rows

    try:
        assert reference_path is not None and predicted_path is not None
        reference = read_matrix(reference_path)
        predicted = read_matrix(predicted_path)
    except Exception as exc:
        sample_errors.append(
            append_issue(rows, "fatal_errors", sample=sample, kind="read_matrix", message=str(exc))
        )
        rows["errors"].extend(sample_errors)
        rows["sample_status"].append(
            sample_status_row(
                sample,
                prediction_path=predicted_path,
                reference_path=reference_path,
                errors=sample_errors,
                warnings=sample_warnings,
            )
        )
        return rows

    semantics = matrix_semantics_fields(
        reference,
        predicted,
        target_component_policy=target_component_policy,
    )

    for kind, data in (("siesta", reference), ("predicted", predicted)):
        rows["overlap"].append(
            {
                "sample": sample,
                "kind": kind,
                "matrix_path": str(data.path),
                "n_bands": int(data.hamiltonian.shape[0]),
                "hamiltonian_components": int(data.component_count),
                "spin_kind": data.spin_kind,
                "orthogonal": data.orthogonal,
                "has_overlap": data.has_overlap,
                "overlap_error": data.overlap_error,
                "fermi_level_eV": data.fermi_level,
                **semantics,
            }
        )

    rows["component"].extend(component_channel_metrics(sample, reference, predicted, semantics))

    compatibility_errors = matrix_compatibility_errors(
        sample,
        reference,
        predicted,
        target_component_policy=target_component_policy,
    )
    compatibility_warnings = matrix_compatibility_warnings(sample, reference, predicted)
    if compatibility_warnings:
        rows["warnings"].extend(compatibility_warnings)
        sample_warnings.extend(compatibility_warnings)
    if compatibility_errors:
        rows["fatal_errors"].extend(compatibility_errors)
        rows["errors"].extend(compatibility_errors)
        sample_errors.extend(compatibility_errors)
        rows["sample_status"].append(
            sample_status_row(
                sample,
                prediction_path=predicted_path,
                reference_path=reference_path,
                errors=sample_errors,
                warnings=sample_warnings,
            )
        )
        return rows

    try:
        rows["sparse"].append({**sparse_metrics(sample, reference, predicted), **semantics})
        rows["sparse_sweep"].extend(sparse_threshold_sweep_metrics(sample, reference, predicted))
    except Exception as exc:
        issue = append_issue(rows, "fatal_errors", sample=sample, kind="sparse_metrics", message=str(exc))
        rows["errors"].append(issue)
        sample_errors.append(issue)

    try:
        structural = structural_sparse_metrics(
            sample,
            reference,
            predicted,
            result_dir / "structures" / sample / "RUN.fdf",
            basis_counts,
        )
        structural_warnings = structural.get("warnings", []) or []
        if structural_warnings:
            rows["warnings"].extend(structural_warnings)
            sample_warnings.extend(structural_warnings)
        if structural["available"]:
            rows["block"].extend(structural["block_rows"])
            rows["species_pair"].extend(structural["species_pair_rows"])
            rows["distance_bin"].extend(structural["distance_bin_rows"])
            rows["orbital_pair"].extend(structural["orbital_pair_rows"])
        else:
            rows["structural_unavailable"].append({"sample": sample, "reason": structural["reason"]})
        if structural.get("distance_unavailable_reason"):
            rows["structural_unavailable"].append(
                {"sample": sample, "reason": str(structural["distance_unavailable_reason"])}
            )
    except Exception as exc:
        rows["structural_unavailable"].append({"sample": sample, "reason": str(exc)})
        sample_warnings.append(
            append_issue(
                rows,
                "warnings",
                sample=sample,
                kind="structural_metrics",
                severity="severe",
                message=str(exc),
            )
        )

    eigen_root = result_dir / "eigenvalues"
    dos_root = result_dir / "dos"
    try:
        ref_eig = generalized_eigenvalues(reference.hamiltonian, reference.overlap)
        pred_eig = generalized_eigenvalues(predicted.hamiltonian, reference.overlap)
        write_csv(eigen_root / "siesta" / f"{sample}.csv", ["band", "eigenvalue_eV"], eigenvalue_rows(ref_eig))
        write_csv(eigen_root / "predicted" / f"{sample}.csv", ["band", "eigenvalue_eV"], eigenvalue_rows(pred_eig))
        fermi_level = reference.fermi_level
        fermi_source = reference.fermi_level_source or "unavailable"
        same_band_count = ref_eig.size == pred_eig.size
        spectral_comparable = bool(
            (reference.orthogonal or reference.has_overlap)
            and same_band_count
            and fermi_level is not None
        )
        if fermi_level is None or not math.isfinite(fermi_level):
            sample_warnings.append(
                append_issue(
                    rows,
                    "warnings",
                    sample=sample,
                    kind="missing_fermi_level",
                    message=(
                        "SIESTA reference does not provide a Fermi level; "
                        "near-Fermi, occupied-band, frontier, gap, and fixed-window DOS metrics were left unavailable."
                    ),
                )
            )
            fermi_level = None
            fermi_source = "unavailable"
            spectral_comparable = False
        band_rows, spectral_metrics = eigen_error_metrics(
            ref_eig,
            pred_eig,
            fermi_level,
            fermi_source,
        )
        if low_energy_enabled:
            low_metrics = low_energy_metrics(
                reference,
                predicted,
                n_states=low_energy_n_states,
                alignment=low_energy_alignment,
            )
            spectral_metrics.update(low_metrics)
            if low_metrics.get("low_energy_warning"):
                sample_warnings.append(
                    append_issue(
                        rows,
                        "warnings",
                        sample=sample,
                        kind="low_energy_metrics",
                        message=str(low_metrics["low_energy_warning"]),
                    )
                )
        write_csv(
            eigen_root / "band_errors" / f"{sample}.csv",
            ["band", "siesta_eV", "predicted_eV", "error_eV", "abs_error_eV", "siesta_minus_fermi_eV"],
            band_rows,
        )
        rows["spectral"].append(
            {
                "sample": sample,
                "siesta_bands": int(ref_eig.size),
                "predicted_bands": int(pred_eig.size),
                "spectral_comparable": spectral_comparable,
                "same_band_count": same_band_count,
                "reference_has_overlap": reference.has_overlap,
                "hamiltonian_symmetrized_for_spectrum": True,
                **semantics,
                **spectral_metrics,
            }
        )
        dos_grid_rows, dos_metrics = dos_for_sample(ref_eig, pred_eig)
        _dos_window_grid, dos_window_metrics = dos_fermi_window_metrics(ref_eig, pred_eig, fermi_level)
        write_csv(
            dos_root / f"{sample}.csv",
            ["energy_eV", "siesta_dos", "predicted_dos", "siesta_dos_normalized", "predicted_dos_normalized"],
            dos_grid_rows,
        )
        rows["dos"].append({"sample": sample, **dos_metrics, **dos_window_metrics})
        for sigma in DOS_SIGMA_SWEEP_EV:
            _grid_rows, sweep_metrics = dos_for_sample(ref_eig, pred_eig, sigma_ev=sigma)
            rows["dos_sweep"].append({"sample": sample, **sweep_metrics})
    except Exception as exc:
        issue = append_issue(rows, "fatal_errors", sample=sample, kind="spectral_or_dos_metrics", message=str(exc))
        rows["errors"].append(issue)
        sample_errors.append(issue)
    rows["sample_status"].append(
        sample_status_row(
            sample,
            prediction_path=predicted_path,
            reference_path=reference_path,
            errors=sample_errors,
            warnings=sample_warnings,
        )
    )
    return rows


def find_reference(sample_dir: Path) -> Path | None:
    selection = choose_reference_matrix(sample_dir)
    return selection.path if selection.ok else None


def infer_component_count(hamiltonian_obj: Any) -> int:
    value = getattr(hamiltonian_obj, "dim", None)
    if callable(value):
        try:
            value = value()
        except TypeError:
            value = None
    if isinstance(value, (int, np.integer)) and int(value) > 0:
        raw_dim = int(value)
        if not bool(getattr(hamiltonian_obj, "orthogonal", False)) and raw_dim > 1:
            return raw_dim - 1
        return raw_dim
    return 1


def read_matrix(path: Path) -> MatrixData:
    sile = sisl.get_sile(str(path))
    hamiltonian_obj = sile.read_hamiltonian()
    component_count = infer_component_count(hamiltonian_obj)
    hamiltonian = hamiltonian_obj.tocsr(0)
    components = [hamiltonian]
    for component_index in range(1, component_count):
        components.append(hamiltonian_obj.tocsr(component_index))
    overlap = None
    has_overlap = False
    overlap_error = None
    try:
        overlap_obj = sile.read_overlap()
        overlap = overlap_obj.tocsr()
        has_overlap = overlap is not None
    except Exception as exc:  # pragma: no cover - backend dependent.
        overlap_error = str(exc)
    try:
        own_eigenvalues = np.asarray(hamiltonian_obj.eigh(), dtype=float)
    except Exception:
        own_eigenvalues = np.asarray([], dtype=float)
    try:
        fermi_level = float(sile.read_fermi_level())
        fermi_level_source = "siesta_file"
    except Exception:
        fermi_level = None
        fermi_level_source = "unavailable"
    return MatrixData(
        path=path,
        hamiltonian=hamiltonian,
        overlap=overlap,
        own_eigenvalues=own_eigenvalues,
        fermi_level=fermi_level,
        fermi_level_source=fermi_level_source,
        orthogonal=bool(getattr(hamiltonian_obj, "orthogonal", False)),
        has_overlap=has_overlap,
        overlap_error=overlap_error,
        sha256=file_sha256(path),
        component_count=component_count,
        spin_kind=str(getattr(hamiltonian_obj, "spin", "")) or None,
        components=tuple(components),
    )


def matrix_compatibility_errors(
    sample: str,
    reference: MatrixData,
    predicted: MatrixData,
    *,
    target_component_policy: str = "unknown",
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if reference.hamiltonian.shape != predicted.hamiltonian.shape:
        errors.append(
            {
                "sample": sample,
                "kind": "matrix_shape_mismatch",
                "error": (
                    "Reference and prediction Hamiltonian shapes differ: "
                    f"{reference.hamiltonian.shape} vs {predicted.hamiltonian.shape}."
                ),
                "reference_shape": list(reference.hamiltonian.shape),
                "predicted_shape": list(predicted.hamiltonian.shape),
            }
        )
    graph2mat_auxiliary_prediction = is_graph2mat_auxiliary_prediction(reference, predicted)
    for role, data in (("reference", reference), ("prediction", predicted)):
        if matrix_has_complex_values(data.hamiltonian):
            errors.append(
                {
                    "sample": sample,
                    "kind": "unsupported_complex_hamiltonian",
                    "error": (
                        f"Unsupported complex-valued {role} Hamiltonian. The current benchmark "
                        "does not validate spin-orbit/k-point complex matrix semantics."
                    ),
                    "matrix_role": role,
                    "matrix_path": str(data.path),
                }
            )
        if data.overlap is not None and matrix_has_complex_values(data.overlap):
            errors.append(
                {
                    "sample": sample,
                    "kind": "unsupported_complex_overlap",
                    "error": (
                        f"Unsupported complex-valued {role} overlap. The current benchmark "
                        "does not validate complex generalized eigenproblems."
                    ),
                    "matrix_role": role,
                    "matrix_path": str(data.path),
                }
            )
    if reference.component_count != 1:
        errors.append(
            {
                "sample": sample,
                "kind": "unsupported_matrix_components",
                "error": f"Unsupported reference matrix component count: {reference.component_count}.",
                "component_count": reference.component_count,
            }
        )
    if (
        target_component_policy == "h_only"
        and predicted.component_count != 1
        and not graph2mat_auxiliary_prediction
    ):
        errors.append(
            {
                "sample": sample,
                "kind": "target_component_policy_mismatch",
                "error": (
                    "Expected H-only prediction semantics, but the prediction "
                    f"container reports {predicted.component_count} Hamiltonian components."
                ),
                "target_component_policy": target_component_policy,
                "prediction_component_count": predicted.component_count,
            }
        )
    if predicted.component_count != 1 and not graph2mat_auxiliary_prediction:
        errors.append(
            {
                "sample": sample,
                "kind": "unsupported_matrix_components",
                "error": f"Unsupported prediction matrix component count: {predicted.component_count}.",
                "component_count": predicted.component_count,
            }
        )
    if (
        reference.spin_kind
        and predicted.spin_kind
        and reference.spin_kind != predicted.spin_kind
        and not graph2mat_auxiliary_prediction
    ):
        errors.append(
            {
                "sample": sample,
                "kind": "spin_state_mismatch",
                "error": (
                    "Reference and prediction spin metadata differ: "
                    f"{reference.spin_kind} vs {predicted.spin_kind}."
                ),
                "reference_spin": reference.spin_kind,
                "predicted_spin": predicted.spin_kind,
            }
        )
    if unsupported_spin_kind(reference.spin_kind):
        errors.append(
            {
                "sample": sample,
                "kind": "unsupported_spin_kind",
                "error": f"Unsupported reference spin metadata: {reference.spin_kind}.",
                "matrix_role": "reference",
                "spin_kind": reference.spin_kind,
            }
        )
    if unsupported_spin_kind(predicted.spin_kind) and not graph2mat_auxiliary_prediction:
        errors.append(
            {
                "sample": sample,
                "kind": "unsupported_spin_kind",
                "error": f"Unsupported prediction spin metadata: {predicted.spin_kind}.",
                "matrix_role": "prediction",
                "spin_kind": predicted.spin_kind,
            }
        )
    if reference.overlap is not None and reference.overlap.shape != reference.hamiltonian.shape:
        errors.append(
            {
                "sample": sample,
                "kind": "invalid_overlap_shape",
                "error": (
                    "Reference overlap shape does not match reference Hamiltonian shape: "
                    f"{reference.overlap.shape} vs {reference.hamiltonian.shape}."
                ),
                "overlap_shape": list(reference.overlap.shape),
                "reference_shape": list(reference.hamiltonian.shape),
            }
        )
    if not reference.orthogonal and reference.overlap is None:
        errors.append(
            {
                "sample": sample,
                "kind": "missing_required_overlap",
                "error": (
                    "Reference Hamiltonian is non-orthogonal but no overlap matrix was readable; "
                    "generalized spectral/DOS metrics are invalid."
                ),
                "overlap_error": reference.overlap_error,
            }
        )
    return errors


def is_graph2mat_auxiliary_prediction(reference: MatrixData, predicted: MatrixData) -> bool:
    reference_spin = str(reference.spin_kind or "").lower()
    predicted_spin = str(predicted.spin_kind or "").lower()
    return (
        reference.component_count == 1
        and predicted.component_count == 2
        and not reference.orthogonal
        and not predicted.orthogonal
        and "unpolarized" in reference_spin
        and "polarized" in predicted_spin
    )


def matrix_has_complex_values(matrix: sparse.spmatrix) -> bool:
    if not np.iscomplexobj(matrix.data):
        return False
    if matrix.data.size == 0:
        return False
    return bool(np.max(np.abs(np.imag(matrix.data))) > COMPLEX_IMAG_TOLERANCE)


def unsupported_spin_kind(spin_kind: str | None) -> bool:
    text = str(spin_kind or "").strip().lower()
    if not text:
        return False
    return "unpolarized" not in text and text not in {"none", "false", "0"}


def matrix_compatibility_warnings(sample: str, reference: MatrixData, predicted: MatrixData) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if is_graph2mat_auxiliary_prediction(reference, predicted):
        warnings.append(
            {
                "sample": sample,
                "kind": "graph2mat_auxiliary_component_ignored",
                "severity": "severe",
                "error": (
                    "Graph2Mat wrote a non-orthogonal prediction container with two matrix components. "
                    "Metrics compare Hamiltonian component 0 only; the auxiliary predicted overlap/spin-like "
                    "component is ignored, and spectral metrics use the SIESTA reference overlap."
                ),
                "reference_components": reference.component_count,
                "predicted_components": predicted.component_count,
                "reference_spin": reference.spin_kind,
                "predicted_spin": predicted.spin_kind,
            }
        )
    diagnostics = overlap_diagnostics(reference, predicted)
    value = diagnostics["prediction_overlap_relative_frobenius_vs_reference"]
    if isinstance(value, float) and math.isfinite(value) and value > OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD:
        warnings.append(
            {
                "sample": sample,
                "kind": "prediction_overlap_mismatch",
                "severity": "severe",
                "error": (
                    "Prediction-owned overlap differs from the SIESTA reference overlap. "
                    "Spectral metrics use S_ref; the prediction HSX is not safe as a "
                    "standalone generalized-eigenproblem input."
                ),
                "prediction_overlap_relative_frobenius_vs_reference": value,
                "overlap_source": diagnostics["overlap_source"],
                "prediction_own_overlap_used": diagnostics["prediction_own_overlap_used"],
                "prediction_self_contained_hsx_safe": diagnostics["prediction_self_contained_hsx_safe"],
            }
        )
    return warnings


def sparse_norm(matrix: sparse.spmatrix) -> float:
    return float(np.sqrt(np.abs(matrix).power(2).sum()))


def relative_sparse_frobenius(delta: sparse.spmatrix, reference: sparse.spmatrix) -> float:
    denominator = sparse_norm(reference)
    if denominator == 0.0:
        return math.nan
    return sparse_norm(delta) / denominator


def overlap_diagnostics(reference: MatrixData, predicted: MatrixData) -> dict[str, Any]:
    overlap_source = "siesta_reference" if reference.overlap is not None else "none_standard_eigenproblem"
    rel_diff = math.nan
    unavailable_reason = ""
    if reference.overlap is None:
        unavailable_reason = "reference_overlap_unavailable"
    elif predicted.overlap is None:
        unavailable_reason = "missing_prediction_overlap"
    elif predicted.overlap.shape != reference.overlap.shape:
        unavailable_reason = "prediction_overlap_shape_mismatch"
    else:
        rel_diff = relative_sparse_frobenius(predicted.overlap - reference.overlap, reference.overlap)
        if math.isfinite(rel_diff) and rel_diff > OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD:
            unavailable_reason = "prediction_overlap_mismatch"

    auxiliary_ignored = is_graph2mat_auxiliary_prediction(reference, predicted)
    prediction_safe = True
    if reference.overlap is not None:
        prediction_safe = (
            predicted.overlap is not None
            and predicted.overlap.shape == reference.overlap.shape
            and math.isfinite(rel_diff)
            and rel_diff <= OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD
            and not auxiliary_ignored
        )
    return {
        "overlap_source": overlap_source,
        "prediction_own_overlap_used": False,
        "prediction_overlap_relative_frobenius_vs_reference": rel_diff,
        "prediction_overlap_check_threshold": OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD,
        "prediction_self_contained_hsx_safe": prediction_safe,
        "prediction_self_contained_hsx_unsafe_reason": (
            "graph2mat_auxiliary_component_ignored" if auxiliary_ignored else unavailable_reason
        ),
    }


def matrix_semantics_fields(
    reference: MatrixData,
    predicted: MatrixData,
    *,
    target_component_policy: str,
) -> dict[str, Any]:
    auxiliary_ignored = is_graph2mat_auxiliary_prediction(reference, predicted)
    return {
        "target_component_policy": target_component_policy,
        "reference_component_count": int(reference.component_count),
        "prediction_component_count": int(predicted.component_count),
        "reference_spin_kind": reference.spin_kind,
        "prediction_spin_kind": predicted.spin_kind,
        "graph2mat_auxiliary_component_ignored": auxiliary_ignored,
        **overlap_diagnostics(reference, predicted),
    }


def matrix_components(data: MatrixData) -> tuple[sparse.csr_matrix, ...]:
    return data.components or (data.hamiltonian,)


def component_channel_metrics(
    sample: str,
    reference: MatrixData,
    predicted: MatrixData,
    semantics: dict[str, Any],
) -> list[dict[str, Any]]:
    reference_components = matrix_components(reference)
    prediction_components = matrix_components(predicted)
    rows: list[dict[str, Any]] = []
    for index in range(max(len(reference_components), len(prediction_components))):
        ref_available = index < len(reference_components)
        pred_available = index < len(prediction_components)
        channel_role = "hamiltonian" if index == 0 else "auxiliary"
        row: dict[str, Any] = {
            "sample": sample,
            "component_index": index,
            "component_role": channel_role,
            "reference_component_available": ref_available,
            "prediction_component_available": pred_available,
            **semantics,
        }
        if ref_available and pred_available:
            ref_matrix = reference_components[index]
            pred_matrix = prediction_components[index]
            if ref_matrix.shape == pred_matrix.shape:
                ref_values = csr_value_dict(ref_matrix, SUPPORT_THRESHOLD)
                pred_values = csr_value_dict(pred_matrix, SUPPORT_THRESHOLD)
                union = sorted(set(ref_values) | set(pred_values))
                deltas = [pred_values.get(key, 0.0) - ref_values.get(key, 0.0) for key in union]
                row.update(
                    {
                        "component_shape": list(ref_matrix.shape),
                        "component_mae_eV": mean_abs(deltas),
                        "component_rmse_eV": rmse(deltas),
                        "component_mse_eV2": mse(deltas),
                        "component_max_abs_error_eV": float(
                            max((abs(value) for value in deltas), default=math.nan)
                        ),
                        "component_n_entries": len(union),
                        "component_metric_available": True,
                        "component_unavailable_reason": "",
                    }
                )
            else:
                row.update(
                    {
                        "component_shape": None,
                        "component_mae_eV": math.nan,
                        "component_rmse_eV": math.nan,
                        "component_mse_eV2": math.nan,
                        "component_max_abs_error_eV": math.nan,
                        "component_n_entries": 0,
                        "component_metric_available": False,
                        "component_unavailable_reason": "component_shape_mismatch",
                    }
                )
        else:
            row.update(
                {
                    "component_shape": None,
                    "component_mae_eV": math.nan,
                    "component_rmse_eV": math.nan,
                    "component_mse_eV2": math.nan,
                    "component_max_abs_error_eV": math.nan,
                    "component_n_entries": 0,
                    "component_metric_available": False,
                    "component_unavailable_reason": (
                        "missing_reference_component"
                        if pred_available
                        else "missing_prediction_component"
                    ),
                }
            )
        rows.append(row)
    return rows


def hermiticity_defect(matrix: sparse.csr_matrix) -> float:
    denominator = sparse_norm(matrix)
    if denominator == 0:
        return math.nan
    return sparse_norm(matrix - matrix.getH()) / denominator


def csr_value_dict(matrix: sparse.csr_matrix, threshold: float) -> dict[tuple[int, int], complex]:
    coo = matrix.tocoo()
    values: dict[tuple[int, int], complex] = {}
    for row, col, value in zip(coo.row, coo.col, coo.data, strict=False):
        if abs(value) > threshold:
            values[(int(row), int(col))] = value
    return values


def mean_abs(values: list[complex]) -> float:
    return float(np.mean(np.abs(values))) if values else math.nan


def mse(values: list[complex]) -> float:
    return float(np.mean(np.abs(values) ** 2)) if values else math.nan


def rmse(values: list[complex]) -> float:
    return float(np.sqrt(np.mean(np.abs(values) ** 2))) if values else math.nan


def ev_to_mev(value: float) -> float:
    return float(value) * 1000.0


def r2_score(reference_values: list[complex], predicted_values: list[complex]) -> float:
    if not reference_values:
        return math.nan
    reference_array = np.asarray(reference_values, dtype=complex)
    predicted_array = np.asarray(predicted_values, dtype=complex)
    if reference_array.size != predicted_array.size:
        raise ValueError("R2 reference and prediction arrays must have the same size.")
    numerator = float(np.sum(np.abs(predicted_array - reference_array) ** 2))
    denominator = float(np.sum(np.abs(reference_array - np.mean(reference_array)) ** 2))
    if denominator == 0.0:
        return math.nan
    return float(1.0 - numerator / denominator)


def spearman_correlation(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        try:
            x = float(row[x_key])
            y = float(row[y_key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if len(pairs) < 2:
        return math.nan
    x_values = np.asarray([pair[0] for pair in pairs], dtype=float)
    y_values = np.asarray([pair[1] for pair in pairs], dtype=float)
    if float(np.std(x_values)) == 0.0 or float(np.std(y_values)) == 0.0:
        return math.nan
    xr = scipy.stats.rankdata(x_values)
    yr = scipy.stats.rankdata(y_values)
    return float(np.corrcoef(xr, yr)[0, 1])


def sparse_metrics(sample: str, reference: MatrixData, predicted: MatrixData) -> dict[str, Any]:
    ref_values = csr_value_dict(reference.hamiltonian, SUPPORT_THRESHOLD)
    pred_values = csr_value_dict(predicted.hamiltonian, SUPPORT_THRESHOLD)
    ref_support = set(ref_values)
    pred_support = set(pred_values)
    union_support = ref_support | pred_support
    intersection = ref_support & pred_support
    ref_indices = sorted(ref_support)
    pred_indices = sorted(pred_support)
    union_indices = sorted(union_support)

    false_zeros = ref_support - pred_support
    false_nonzeros = pred_support - ref_support
    deltas_ref = [pred_values.get(index, 0.0) - ref_values[index] for index in ref_indices]
    deltas_pred = [pred_values[index] - ref_values.get(index, 0.0) for index in pred_indices]
    deltas_union = [
        pred_values.get(index, 0.0) - ref_values.get(index, 0.0)
        for index in union_indices
    ]
    ref_targets_ref = [ref_values[index] for index in ref_indices]
    pred_targets_ref = [pred_values.get(index, 0.0) for index in ref_indices]
    ref_targets_pred = [ref_values.get(index, 0.0) for index in pred_indices]
    pred_targets_pred = [pred_values[index] for index in pred_indices]
    ref_targets_union = [ref_values.get(index, 0.0) for index in union_indices]
    pred_targets_union = [pred_values.get(index, 0.0) for index in union_indices]
    mae_ref = mean_abs(deltas_ref)
    rmse_ref = rmse(deltas_ref)
    mae_pred = mean_abs(deltas_pred)
    rmse_pred = rmse(deltas_pred)
    mae_union = mean_abs(deltas_union)
    rmse_union = rmse(deltas_union)
    ref_fro = float(np.sqrt(sum(abs(value) ** 2 for value in ref_values.values())))
    ref_pattern_fro = float(np.sqrt(sum(abs(value) ** 2 for value in deltas_ref)))
    union_fro = float(np.sqrt(sum(abs(value) ** 2 for value in deltas_union)))
    ref_l1 = float(sum(abs(value) for value in ref_values.values()))
    union_l1 = float(sum(abs(value) for value in deltas_union))
    precision = len(intersection) / len(pred_support) if pred_support else math.nan
    recall = len(intersection) / len(ref_support) if ref_support else math.nan
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else math.nan
    )
    n_entries = reference.hamiltonian.shape[0] * reference.hamiltonian.shape[1]
    return {
        "sample": sample,
        "n_orbitals": reference.hamiltonian.shape[0],
        "n_entries": n_entries,
        "ref_nnz": len(ref_support),
        "pred_nnz": len(pred_support),
        "union_nnz": len(union_support),
        "ref_density": len(ref_support) / n_entries if n_entries else math.nan,
        "pred_density": len(pred_support) / n_entries if n_entries else math.nan,
        "matrix_metric_target_space": MATRIX_METRIC_TARGET_SPACE,
        "mae_ref_eV": mae_ref,
        "rmse_ref_eV": rmse_ref,
        "mse_ref_eV2": mse(deltas_ref),
        "r2_ref": r2_score(ref_targets_ref, pred_targets_ref),
        "mae_ref_meV": ev_to_mev(mae_ref),
        "rmse_ref_meV": ev_to_mev(rmse_ref),
        "mae_pred_eV": mae_pred,
        "rmse_pred_eV": rmse_pred,
        "mse_pred_eV2": mse(deltas_pred),
        "r2_pred": r2_score(ref_targets_pred, pred_targets_pred),
        "mae_pred_meV": ev_to_mev(mae_pred),
        "rmse_pred_meV": ev_to_mev(rmse_pred),
        "mae_union_eV": mae_union,
        "rmse_union_eV": rmse_union,
        "mse_union_eV2": mse(deltas_union),
        "r2_union": r2_score(ref_targets_union, pred_targets_union),
        "mae_union_meV": ev_to_mev(mae_union),
        "rmse_union_meV": ev_to_mev(rmse_union),
        "max_abs_error_union_eV": float(max((abs(value) for value in deltas_union), default=math.nan)),
        "relative_frobenius_ref": ref_pattern_fro / ref_fro if ref_fro else math.nan,
        "relative_frobenius_union": union_fro / ref_fro if ref_fro else math.nan,
        "relative_l1_union": union_l1 / ref_l1 if ref_l1 else math.nan,
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
        "false_zeros": len(false_zeros),
        "false_nonzeros": len(false_nonzeros),
        "false_zero_rate": len(false_zeros) / len(ref_support) if ref_support else math.nan,
        "false_nonzero_rate": len(false_nonzeros) / len(pred_support) if pred_support else math.nan,
        "weighted_false_zeros_eV": float(sum(abs(ref_values[index]) for index in false_zeros)),
        "weighted_false_nonzeros_eV": float(sum(abs(pred_values[index]) for index in false_nonzeros)),
        "hermiticity_ref": hermiticity_defect(reference.hamiltonian),
        "hermiticity_pred": hermiticity_defect(predicted.hamiltonian),
    }


def sparse_threshold_sweep_metrics(sample: str, reference: MatrixData, predicted: MatrixData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in SUPPORT_THRESHOLDS_SWEEP:
        ref_values = csr_value_dict(reference.hamiltonian, threshold)
        pred_values = csr_value_dict(predicted.hamiltonian, threshold)
        support = set(ref_values) | set(pred_values)
        deltas = [pred_values.get(index, 0.0) - ref_values.get(index, 0.0) for index in support]
        rows.append(
            {
                "sample": sample,
                "support_threshold": threshold,
                "union_nnz": len(support),
                "mae_union_eV": mean_abs(deltas),
                "rmse_union_eV": rmse(deltas),
            }
        )
    return rows


def parse_structure_atoms(path: Path) -> tuple[list[str], np.ndarray]:
    """Return species labels and Cartesian coordinates from a minimal SIESTA RUN.fdf."""
    if not path.exists():
        return [], np.empty((0, 3), dtype=float)
    species_by_index: dict[str, str] = {}
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    in_species = False
    in_coords = False
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("%block"):
            block = lower.split(maxsplit=1)[1] if len(lower.split(maxsplit=1)) > 1 else ""
            in_species = "chemicalspecieslabel" in block
            in_coords = "atomiccoordinatesandatomicspecies" in block
            continue
        if lower.startswith("%endblock"):
            in_species = False
            in_coords = False
            continue
        parts = line.split()
        if in_species and len(parts) >= 3:
            species_by_index[parts[0]] = parts[-1]
            continue
        if in_coords and len(parts) >= 4:
            try:
                coords = (float(parts[0]), float(parts[1]), float(parts[2]))
            except ValueError:
                continue
            species_key = parts[3]
            atoms.append((species_by_index.get(species_key, species_key), coords))
    labels = [label for label, _coords in atoms]
    coords = np.asarray([coords for _label, coords in atoms], dtype=float) if atoms else np.empty((0, 3), dtype=float)
    return labels, coords


def basis_orbital_counts(basis_dirs: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for basis_dir in basis_dirs:
        if not basis_dir.exists():
            continue
        for path in sorted(basis_dir.glob("*.ion.xml")):
            try:
                root = ET.parse(path).getroot()
            except ET.ParseError as exc:
                raise RuntimeError(f"Could not parse basis file {path}: {exc}") from exc
            symbol = (root.findtext("symbol") or root.findtext("label") or path.stem).strip()
            paos = root.find("paos")
            if paos is None:
                raise RuntimeError(f"Basis file {path} does not contain a <paos> block.")
            total = 0
            for orbital in paos.findall("orbital"):
                try:
                    angular_momentum = int(str(orbital.attrib.get("l", "")).strip())
                except ValueError as exc:
                    raise RuntimeError(f"Basis file {path} contains an orbital without integer l.") from exc
                total += 2 * angular_momentum + 1
            if total <= 0:
                raise RuntimeError(f"Basis file {path} does not define PAO orbitals.")
            counts[symbol] = total
    return counts


def find_basis_dirs(result_dir: Path) -> list[Path]:
    candidates = [
        result_dir / "basis",
        result_dir / "structures" / "basis",
        result_dir.parent / "basis",
    ]
    return [path for path in candidates if path.exists() and list(path.glob("*.ion.xml"))]


def orbital_atom_map_from_basis(species: list[str], basis_counts: dict[str, int], n_orbitals: int) -> list[int]:
    missing = sorted({label for label in species if label not in basis_counts})
    if missing:
        raise RuntimeError(f"Missing .ion.xml basis for species: {', '.join(missing)}.")
    atom_by_orbital: list[int] = []
    for atom_index, label in enumerate(species):
        atom_by_orbital.extend([atom_index] * int(basis_counts[label]))
    if len(atom_by_orbital) != n_orbitals:
        raise RuntimeError(
            "Basis-derived orbital count does not match Hamiltonian dimension: "
            f"{len(atom_by_orbital)} != {n_orbitals}."
        )
    return atom_by_orbital


def orbital_records_from_basis(
    species: list[str],
    basis_counts: dict[str, int],
    n_orbitals: int,
) -> list[dict[str, Any]]:
    missing = sorted({label for label in species if label not in basis_counts})
    if missing:
        raise RuntimeError(f"Missing .ion.xml basis for species: {', '.join(missing)}.")
    records: list[dict[str, Any]] = []
    for atom_index, label in enumerate(species):
        for local_index in range(int(basis_counts[label])):
            records.append(
                {
                    "atom_index": atom_index,
                    "species": label,
                    "local_orbital_index": local_index,
                    "orbital_label": f"orbital_{local_index}",
                }
            )
    if len(records) != n_orbitals:
        raise RuntimeError(
            "Basis-derived orbital count does not match Hamiltonian dimension: "
            f"{len(records)} != {n_orbitals}."
        )
    return records


def distance_bin(distance_ang: float) -> str:
    if distance_ang < 1.2:
        return "0-1.2"
    if distance_ang < 2.0:
        return "1.2-2.0"
    if distance_ang < 4.0:
        return "2.0-4.0"
    return ">4.0"


def _empty_structural_metrics(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "block_rows": [],
        "species_pair_rows": [],
        "distance_bin_rows": [],
        "orbital_pair_rows": [],
        "warnings": [],
        "distance_unavailable_reason": "",
    }


def _finalize_groups(groups: dict[Any, list[dict[str, Any]]], row_builder) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, entries in sorted(groups.items(), key=lambda item: str(item[0])):
        deltas = [entry["delta"] for entry in entries]
        distances = [entry["distance_ang"] for entry in entries if entry["distance_ang"] is not None]
        row = row_builder(key, entries)
        row.update(
            {
                "n_entries": len(entries),
                "mae_union_eV": mean_abs(deltas),
                "rmse_union_eV": rmse(deltas),
                "max_abs_error_union_eV": float(max((abs(value) for value in deltas), default=math.nan)),
                "mean_distance_ang": float(np.mean(distances)) if distances else math.nan,
            }
        )
        rows.append(row)
    return rows


def _finalize_orbital_pair_groups(groups: dict[Any, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, entries in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        (
            row_species,
            col_species,
            row_orbital_index,
            col_orbital_index,
            row_orbital_label,
            col_orbital_label,
        ) = key
        species_pair = f"{row_species}-{col_species}"
        deltas = [entry["delta"] for entry in entries]
        ref_targets = [entry["ref_value"] for entry in entries]
        pred_targets = [entry["pred_value"] for entry in entries]
        rows.append(
            {
                "sample": entries[0]["sample"],
                "row_species": row_species,
                "col_species": col_species,
                "species_pair": species_pair,
                "row_orbital_index": row_orbital_index,
                "col_orbital_index": col_orbital_index,
                "row_orbital_label": row_orbital_label,
                "col_orbital_label": col_orbital_label,
                "n_entries": len(entries),
                "mae_union_eV": mean_abs(deltas),
                "mae_union_meV": ev_to_mev(mean_abs(deltas)),
                "mse_union_eV2": mse(deltas),
                "rmse_union_eV": rmse(deltas),
                "r2_union": r2_score(ref_targets, pred_targets),
                "max_abs_error_union_eV": float(max((abs(value) for value in deltas), default=math.nan)),
                "mean_abs_ref_eV": mean_abs(ref_targets),
                "mean_signed_error_eV": float(np.mean([float(np.real(value)) for value in deltas]))
                if deltas
                else math.nan,
                "metric_target_space": ORBITAL_PAIR_METRIC_TARGET_SPACE,
                "basis_source": ORBITAL_PAIR_BASIS_SOURCE,
            }
        )
    return rows


def orbital_pair_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("row_species"),
            row.get("col_species"),
            row.get("species_pair"),
            row.get("row_orbital_index"),
            row.get("col_orbital_index"),
            row.get("row_orbital_label"),
            row.get("col_orbital_label"),
            row.get("metric_target_space"),
            row.get("basis_source"),
        )
        grouped.setdefault(key, []).append(row)

    metric_names = [
        "mae_union_eV",
        "mae_union_meV",
        "mse_union_eV2",
        "rmse_union_eV",
        "r2_union",
        "max_abs_error_union_eV",
        "mean_abs_ref_eV",
        "mean_signed_error_eV",
    ]
    summary: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        (
            row_species,
            col_species,
            species_pair,
            row_orbital_index,
            col_orbital_index,
            row_orbital_label,
            col_orbital_label,
            metric_target_space,
            basis_source,
        ) = key
        row: dict[str, Any] = {
            "row_species": row_species,
            "col_species": col_species,
            "species_pair": species_pair,
            "row_orbital_index": row_orbital_index,
            "col_orbital_index": col_orbital_index,
            "row_orbital_label": row_orbital_label,
            "col_orbital_label": col_orbital_label,
            "n_samples": len({str(item.get("sample")) for item in items}),
            "n_entries": int(sum(int(item.get("n_entries") or 0) for item in items)),
            "metric_target_space": metric_target_space,
            "basis_source": basis_source,
        }
        for metric_name in metric_names:
            values = [
                float(item[metric_name])
                for item in items
                if isinstance(item.get(metric_name), (int, float)) and math.isfinite(float(item[metric_name]))
            ]
            row[f"{metric_name}_mean"] = float(np.mean(values)) if values else math.nan
            row[f"{metric_name}_std"] = float(np.std(values)) if values else math.nan
            row[f"{metric_name}_min"] = float(np.min(values)) if values else math.nan
            row[f"{metric_name}_max"] = float(np.max(values)) if values else math.nan
        summary.append(row)
    return summary


def structural_sparse_metrics(
    sample: str,
    reference: MatrixData,
    predicted: MatrixData,
    structure_path: Path,
    basis_counts: dict[str, int],
) -> dict[str, Any]:
    species, coords = parse_structure_atoms(structure_path)
    if not species:
        raise RuntimeError("Missing or unreadable structure for structural metrics.")
    if reference.hamiltonian.shape != predicted.hamiltonian.shape or reference.hamiltonian.shape[0] != reference.hamiltonian.shape[1]:
        raise RuntimeError("Matrix shape mismatch for structural metrics.")
    orbital_records = orbital_records_from_basis(species, basis_counts, reference.hamiltonian.shape[0])
    atom_by_orbital = [int(record["atom_index"]) for record in orbital_records]
    structure_type = structure_type_from_metadata(structure_path)
    periodic_distance_unsupported = structure_type in PERIODIC_STRUCTURE_TYPES
    structural_warnings: list[dict[str, Any]] = []
    distance_unavailable_reason = ""
    if periodic_distance_unsupported:
        distance_unavailable_reason = (
            "Periodic distance-bin structural metrics are unsupported unless periodic "
            "minimum-image handling is explicitly implemented and validated."
        )
        structural_warnings.append(
            {
                "sample": sample,
                "kind": "unsupported_periodic_distance_bins",
                "severity": "severe",
                "error": distance_unavailable_reason,
                "structure_type": structure_type,
                "structure_path": str(structure_path),
            }
        )

    ref_values = csr_value_dict(reference.hamiltonian, SUPPORT_THRESHOLD)
    pred_values = csr_value_dict(predicted.hamiltonian, SUPPORT_THRESHOLD)
    block_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    species_groups: dict[str, list[dict[str, Any]]] = {}
    distance_groups: dict[str, list[dict[str, Any]]] = {}
    orbital_pair_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row_index, col_index in sorted(set(ref_values) | set(pred_values)):
        row_atom = atom_by_orbital[row_index]
        col_atom = atom_by_orbital[col_index]
        ref_value = ref_values.get((row_index, col_index), 0.0)
        pred_value = pred_values.get((row_index, col_index), 0.0)
        delta = pred_value - ref_value
        distance = float(np.linalg.norm(coords[row_atom] - coords[col_atom])) if len(coords) else math.nan
        entry = {"delta": delta, "distance_ang": distance}
        block_groups.setdefault((row_atom, col_atom), []).append(entry)
        species_pair = f"{species[row_atom]}-{species[col_atom]}"
        species_groups.setdefault(species_pair, []).append(entry)
        if not periodic_distance_unsupported:
            distance_groups.setdefault(distance_bin(distance), []).append(entry)
        row_record = orbital_records[row_index]
        col_record = orbital_records[col_index]
        orbital_pair_key = (
            row_record["species"],
            col_record["species"],
            row_record["local_orbital_index"],
            col_record["local_orbital_index"],
            row_record["orbital_label"],
            col_record["orbital_label"],
        )
        orbital_pair_groups.setdefault(orbital_pair_key, []).append(
            {
                "sample": sample,
                "delta": delta,
                "ref_value": ref_value,
                "pred_value": pred_value,
            }
        )

    block_rows = _finalize_groups(
        block_groups,
        lambda key, _entries: {
            "sample": sample,
            "row_atom": key[0],
            "col_atom": key[1],
            "row_species": species[key[0]],
            "col_species": species[key[1]],
        },
    )
    species_rows = _finalize_groups(
        species_groups,
        lambda key, _entries: {"sample": sample, "species_pair": key},
    )
    distance_rows = (
        []
        if periodic_distance_unsupported
        else _finalize_groups(
            distance_groups,
            lambda key, _entries: {"sample": sample, "distance_bin": key},
        )
    )
    return {
        "available": True,
        "reason": "",
        "block_rows": block_rows,
        "species_pair_rows": species_rows,
        "distance_bin_rows": distance_rows,
        "orbital_pair_rows": _finalize_orbital_pair_groups(orbital_pair_groups),
        "warnings": structural_warnings,
        "distance_unavailable_reason": distance_unavailable_reason,
    }


def symmetrized_dense(matrix: sparse.csr_matrix) -> np.ndarray:
    dense = matrix.toarray()
    return np.asarray((dense + dense.conj().T) / 2.0)


def generalized_eigenvalues(
    hamiltonian: sparse.csr_matrix,
    overlap: sparse.csr_matrix | None,
) -> np.ndarray:
    dense_h = symmetrized_dense(hamiltonian)
    if overlap is None:
        return np.linalg.eigvalsh(dense_h)
    dense_s = symmetrized_dense(overlap)
    return np.asarray(
        scipy.linalg.eigh(dense_h, dense_s, eigvals_only=True, check_finite=False),
        dtype=float,
    )


def validate_low_energy_config(n_states: int, alignment: str) -> tuple[int, str]:
    try:
        n_states = int(n_states)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("evaluation.spectral.low_energy.n_states must be a positive integer.") from exc
    if n_states <= 0:
        raise RuntimeError("evaluation.spectral.low_energy.n_states must be a positive integer.")
    if alignment not in {"none", "global_shift"}:
        raise RuntimeError("evaluation.spectral.low_energy.alignment must be 'none' or 'global_shift'.")
    return n_states, alignment


def low_energy_eigenvalues(
    matrix: sparse.csr_matrix,
    overlap: sparse.csr_matrix | None,
    *,
    overlap_required: bool,
) -> tuple[np.ndarray | None, str]:
    if overlap_required and overlap is None:
        return None, (
            "low-energy eigenvalues unavailable: reference overlap is required "
            "for the generalized eigenproblem but was not found."
        )
    try:
        values = generalized_eigenvalues(matrix, overlap)
    except Exception as exc:
        return None, f"low-energy eigenvalues unavailable: eigenvalue solver failed: {exc}"
    if values.size == 0:
        return None, "low-energy eigenvalues unavailable: no eigenvalues returned by solver."
    return np.sort(np.asarray(values, dtype=float)), ""


def low_energy_metrics(
    reference: MatrixData,
    predicted: MatrixData,
    *,
    n_states: int = LOW_ENERGY_N_STATES,
    alignment: str = LOW_ENERGY_ALIGNMENT,
) -> dict[str, Any]:
    n_states, alignment = validate_low_energy_config(n_states, alignment)
    overlap_required = not bool(reference.orthogonal)
    overlap = reference.overlap
    overlap_used = overlap is not None
    metadata = {
        "low_energy_requested_states": n_states,
        "low_energy_alignment": alignment,
        "low_energy_overlap_used": overlap_used,
        "low_energy_overlap_required": overlap_required,
        "low_energy_solver": "scipy.linalg.eigh_generalized" if overlap_used else "numpy.linalg.eigvalsh_standard",
        "low_energy_warning": "",
    }
    ref_eig, ref_warning = low_energy_eigenvalues(
        reference.hamiltonian,
        overlap,
        overlap_required=overlap_required,
    )
    pred_eig, pred_warning = low_energy_eigenvalues(
        predicted.hamiltonian,
        overlap,
        overlap_required=overlap_required,
    )
    warning = ref_warning or pred_warning
    if warning:
        return {
            **metadata,
            "low_energy_n_states": None,
            "low_energy_mae_eV": math.nan,
            "low_energy_rmse_eV": math.nan,
            "low_energy_max_abs_error_eV": math.nan,
            "low_energy_aligned_rmse_eV": math.nan,
            "low_energy_warning": warning,
        }
    assert ref_eig is not None and pred_eig is not None
    count = min(n_states, ref_eig.size, pred_eig.size)
    if count <= 0:
        return {
            **metadata,
            "low_energy_n_states": None,
            "low_energy_mae_eV": math.nan,
            "low_energy_rmse_eV": math.nan,
            "low_energy_max_abs_error_eV": math.nan,
            "low_energy_aligned_rmse_eV": math.nan,
            "low_energy_warning": "low-energy eigenvalues unavailable: no common states to compare.",
        }
    ref_low = ref_eig[:count]
    pred_low = pred_eig[:count]
    delta = pred_low - ref_low
    aligned_rmse = math.nan
    if alignment == "global_shift":
        shift = float(np.mean(ref_low - pred_low))
        aligned_delta = (pred_low + shift) - ref_low
        aligned_rmse = float(np.sqrt(np.mean(aligned_delta**2)))
    return {
        **metadata,
        "low_energy_n_states": int(count),
        "low_energy_mae_eV": float(np.mean(np.abs(delta))),
        "low_energy_rmse_eV": float(np.sqrt(np.mean(delta**2))),
        "low_energy_max_abs_error_eV": float(np.max(np.abs(delta))),
        "low_energy_aligned_rmse_eV": aligned_rmse,
    }


def eigenvalue_rows(values: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"band": index, "eigenvalue_eV": float(value)}
        for index, value in enumerate(values)
    ]


def band_gap(values: np.ndarray, fermi_level: float | None) -> float:
    if fermi_level is None or values.size == 0:
        return math.nan
    occupied = values[values <= fermi_level]
    unoccupied = values[values > fermi_level]
    if occupied.size == 0 or unoccupied.size == 0:
        return math.nan
    return float(unoccupied[0] - occupied[-1])


def eigen_error_metrics(
    reference: np.ndarray,
    predicted: np.ndarray,
    fermi_level: float | None,
    fermi_level_source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_bands = min(reference.size, predicted.size)
    reference = reference[:n_bands]
    predicted = predicted[:n_bands]
    errors = predicted - reference
    band_rows = [
        {
            "band": index,
            "siesta_eV": float(reference[index]),
            "predicted_eV": float(predicted[index]),
            "error_eV": float(errors[index]),
            "abs_error_eV": float(abs(errors[index])),
            "siesta_minus_fermi_eV": None
            if fermi_level is None
            else float(reference[index] - fermi_level),
        }
        for index in range(n_bands)
    ]

    occupied_mask = np.zeros(n_bands, dtype=bool)
    fermi_mask = np.zeros(n_bands, dtype=bool)
    frontier_mask = np.zeros(n_bands, dtype=bool)
    homo_index = None
    lumo_index = None
    if fermi_level is not None:
        occupied_mask = reference <= fermi_level
        fermi_mask = np.abs(reference - fermi_level) <= FERMI_WINDOW_EV
        occ_indices = np.where(occupied_mask)[0]
        virt_indices = np.where(~occupied_mask)[0]
        if occ_indices.size and virt_indices.size:
            homo_index = int(occ_indices[-1])
            lumo_index = int(virt_indices[0])
    if homo_index is not None:
        frontier_mask[homo_index] = True
    if lumo_index is not None:
        frontier_mask[lumo_index] = True

    def masked_mae(mask: np.ndarray) -> float:
        return float(np.mean(np.abs(errors[mask]))) if np.any(mask) else math.nan

    def masked_rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(errors[mask] ** 2))) if np.any(mask) else math.nan

    gap_ref = band_gap(reference, fermi_level)
    gap_pred = band_gap(predicted, fermi_level)
    if (gap_ref != gap_ref or gap_pred != gap_pred) and homo_index is not None and lumo_index is not None:
        gap_ref = float(reference[lumo_index] - reference[homo_index])
        gap_pred = float(predicted[lumo_index] - predicted[homo_index])

    def aligned_errors(mask: np.ndarray | None = None) -> tuple[float, float, float]:
        if n_bands == 0:
            return (math.nan, math.nan, math.nan)
        use_ref = reference if mask is None else reference[mask]
        use_pred = predicted if mask is None else predicted[mask]
        if use_ref.size == 0:
            return (math.nan, math.nan, math.nan)
        shift = float(np.mean(use_ref - use_pred))
        delta = (use_pred + shift) - use_ref
        return shift, float(np.mean(np.abs(delta))), float(np.sqrt(np.mean(delta**2)))

    global_shift, global_aligned_mae, global_aligned_rmse = aligned_errors(None)
    fermi_shift, fermi_aligned_mae, fermi_aligned_rmse = aligned_errors(fermi_mask)
    homo_shift, homo_aligned_mae, homo_aligned_rmse = (
        aligned_errors(np.array([i == homo_index for i in range(n_bands)], dtype=bool))
        if homo_index is not None
        else (math.nan, math.nan, math.nan)
    )
    metrics = {
        "n_compared_bands": n_bands,
        "fermi_ref_eV": fermi_level,
        "fermi_level_source": fermi_level_source,
        "fermi_metric_available": fermi_level is not None and math.isfinite(float(fermi_level)),
        "global_mae_eV": float(np.mean(np.abs(errors))) if n_bands else math.nan,
        "global_rmse_eV": float(np.sqrt(np.mean(errors**2))) if n_bands else math.nan,
        "global_max_abs_error_eV": float(np.max(np.abs(errors))) if n_bands else math.nan,
        "global_mean_signed_error_eV": float(np.mean(errors)) if n_bands else math.nan,
        "occupied_bands": int(np.count_nonzero(occupied_mask)),
        "occupied_metric_available": bool(np.any(occupied_mask)),
        "occupied_mae_eV": masked_mae(occupied_mask),
        "occupied_rmse_eV": masked_rmse(occupied_mask),
        "fermi_window_eV": FERMI_WINDOW_EV,
        "fermi_window_bands": int(np.count_nonzero(fermi_mask)),
        "fermi_window_metric_available": bool(np.any(fermi_mask)),
        "fermi_window_mae_eV": masked_mae(fermi_mask),
        "fermi_window_rmse_eV": masked_rmse(fermi_mask),
        "homo_index": homo_index,
        "lumo_index": lumo_index,
        "homo_error_eV": float(errors[homo_index]) if homo_index is not None else math.nan,
        "lumo_error_eV": float(errors[lumo_index]) if lumo_index is not None else math.nan,
        "frontier_window_bands": int(np.count_nonzero(frontier_mask)),
        "frontier_metric_available": bool(np.any(frontier_mask)),
        "frontier_window_mae_eV": masked_mae(frontier_mask),
        "frontier_window_rmse_eV": masked_rmse(frontier_mask),
        "align_global_shift_eV": global_shift,
        "align_global_mae_eV": global_aligned_mae,
        "align_global_rmse_eV": global_aligned_rmse,
        "align_fermi_shift_eV": fermi_shift,
        "align_fermi_mae_eV": fermi_aligned_mae,
        "align_fermi_rmse_eV": fermi_aligned_rmse,
        "align_homo_shift_eV": homo_shift,
        "align_homo_mae_eV": homo_aligned_mae,
        "align_homo_rmse_eV": homo_aligned_rmse,
        "gap_ref_eV": gap_ref,
        "gap_pred_eV": gap_pred,
        "gap_abs_error_eV": abs(gap_pred - gap_ref)
        if gap_ref == gap_ref and gap_pred == gap_pred
        else math.nan,
    }
    return band_rows, metrics


def gaussian_dos(values: np.ndarray, grid: np.ndarray, sigma: float) -> np.ndarray:
    prefactor = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    dos = np.zeros_like(grid, dtype=float)
    for value in values:
        dos += prefactor * np.exp(-0.5 * ((grid - value) / sigma) ** 2)
    return dos


def normalized_density(density: np.ndarray, dx: float) -> np.ndarray:
    area = float(np.sum(density) * dx)
    if area <= 0:
        return density
    return density / area


def wasserstein_from_grid(a_density: np.ndarray, b_density: np.ndarray, dx: float) -> float:
    a_norm = normalized_density(a_density, dx)
    b_norm = normalized_density(b_density, dx)
    cdf_delta = np.cumsum(a_norm - b_norm) * dx
    return float(np.sum(np.abs(cdf_delta)) * dx)


def dos_for_sample(reference: np.ndarray, predicted: np.ndarray, sigma_ev: float = DOS_SIGMA_EV) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    combined = np.concatenate([reference, predicted])
    if combined.size == 0:
        return [], {
            "dos_wasserstein_eV": math.nan,
            "dos_l1": math.nan,
            "dos_l2": math.nan,
            "energy_min_eV": math.nan,
            "energy_max_eV": math.nan,
        }
    margin = max(5.0 * sigma_ev, 0.05 * float(np.ptp(combined) if combined.size > 1 else 1.0))
    energy_min = float(np.min(combined) - margin)
    energy_max = float(np.max(combined) + margin)
    grid = np.linspace(energy_min, energy_max, DOS_POINTS)
    dx = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
    ref_dos = gaussian_dos(reference, grid, sigma_ev)
    pred_dos = gaussian_dos(predicted, grid, sigma_ev)
    ref_norm = normalized_density(ref_dos, dx)
    pred_norm = normalized_density(pred_dos, dx)
    rows = [
        {
            "energy_eV": float(grid[index]),
            "siesta_dos": float(ref_dos[index]),
            "predicted_dos": float(pred_dos[index]),
            "siesta_dos_normalized": float(ref_norm[index]),
            "predicted_dos_normalized": float(pred_norm[index]),
        }
        for index in range(grid.size)
    ]
    metrics = {
        "dos_sigma_eV": sigma_ev,
        "dos_grid_points": DOS_POINTS,
        "energy_min_eV": energy_min,
        "energy_max_eV": energy_max,
        "dos_wasserstein_eV": wasserstein_from_grid(ref_dos, pred_dos, dx),
        "dos_l1": float(np.sum(np.abs(ref_norm - pred_norm)) * dx),
        "dos_l2": float(np.sqrt(np.sum((ref_norm - pred_norm) ** 2) * dx)),
    }
    return rows, metrics


def dos_fermi_window_metrics(
    reference: np.ndarray,
    predicted: np.ndarray,
    fermi_level: float | None,
    sigma_ev: float = DOS_SIGMA_EV,
) -> tuple[np.ndarray, dict[str, Any]]:
    metrics: dict[str, Any] = {
        "dos_window_min_eV": DOS_FERMI_WINDOW_MIN_EV,
        "dos_window_max_eV": DOS_FERMI_WINDOW_MAX_EV,
        "dos_window_points": DOS_FERMI_WINDOW_POINTS,
        "dos_window_sigma_eV": sigma_ev,
        "dos_window_alignment": DOS_FERMI_WINDOW_ALIGNMENT,
    }
    try:
        fermi_value = float(fermi_level)
    except (TypeError, ValueError):
        metrics.update(
            {
                "dos_mae_500_fermi_window": math.nan,
                "dos_window_metric_available": False,
                "dos_window_unavailable_reason": "missing_fermi_level",
            }
        )
        return np.asarray([], dtype=float), metrics
    if not math.isfinite(fermi_value):
        metrics.update(
            {
                "dos_mae_500_fermi_window": math.nan,
                "dos_window_metric_available": False,
                "dos_window_unavailable_reason": "missing_fermi_level",
            }
        )
        return np.asarray([], dtype=float), metrics
    if np.concatenate([reference, predicted]).size == 0:
        metrics.update(
            {
                "dos_mae_500_fermi_window": math.nan,
                "dos_window_metric_available": False,
                "dos_window_unavailable_reason": "missing_eigenvalues",
            }
        )
        return np.asarray([], dtype=float), metrics

    relative_grid = np.linspace(DOS_FERMI_WINDOW_MIN_EV, DOS_FERMI_WINDOW_MAX_EV, DOS_FERMI_WINDOW_POINTS)
    grid = fermi_value + relative_grid
    reference_dos = gaussian_dos(reference, grid, sigma_ev)
    predicted_dos = gaussian_dos(predicted, grid, sigma_ev)
    metrics.update(
        {
            "dos_mae_500_fermi_window": mean_abs((predicted_dos - reference_dos).tolist()),
            "dos_window_metric_available": True,
            "dos_window_unavailable_reason": "",
        }
    )
    return grid, metrics


def summarize_numeric(rows: list[dict[str, Any]], skip: set[str]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key in skip or value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values.setdefault(key, []).append(number)
    return {
        key: {
            "mean": float(np.mean(column)),
            "std": float(np.std(column)),
            "min": float(np.min(column)),
            "max": float(np.max(column)),
        }
        for key, column in values.items()
        if column
    }


def metric_availability(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, dict[str, Any]]:
    availability: dict[str, dict[str, Any]] = {}
    total = len(rows)
    for metric in metrics:
        available = 0
        for row in rows:
            try:
                value = float(row.get(metric))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                available += 1
        missing = max(0, total - available)
        if available == total and total:
            reason = ""
        elif total == 0:
            reason = "no_samples"
        elif metric.startswith("fermi_window"):
            reason = "fermi_window_unavailable_or_empty"
        elif metric.startswith("frontier_window"):
            reason = "frontier_levels_unavailable"
        elif metric == "dos_mae_500_fermi_window":
            reason = "missing_fermi_level_or_eigenvalues"
        else:
            reason = "metric_unavailable_for_some_samples"
        availability[metric] = {
            "metric_available": available > 0,
            "n_samples_with_metric": available,
            "n_samples_without_metric": missing,
            "metric_unavailable_reason": reason,
        }
    return availability


def pearson_correlation(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        try:
            x = float(row[x_key])
            y = float(row[y_key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if len(pairs) < 2:
        return math.nan
    x_values = np.asarray([pair[0] for pair in pairs], dtype=float)
    y_values = np.asarray([pair[1] for pair in pairs], dtype=float)
    if float(np.std(x_values)) == 0.0 or float(np.std(y_values)) == 0.0:
        return math.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def matrix_spectrum_rows(
    sparse_rows: list[dict[str, Any]],
    spectral_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    spectral_by_sample = {str(row["sample"]): row for row in spectral_rows}
    rows: list[dict[str, Any]] = []
    for sparse_row in sparse_rows:
        spectral_row = spectral_by_sample.get(str(sparse_row["sample"]))
        if spectral_row is None:
            continue
        rows.append(
            {
                "sample": sparse_row["sample"],
                "matrix_metric_target_space": sparse_row.get("matrix_metric_target_space"),
                **{field: sparse_row.get(field) for field in MATRIX_SEMANTIC_FIELDS},
                "mae_ref_eV": sparse_row.get("mae_ref_eV"),
                "rmse_ref_eV": sparse_row.get("rmse_ref_eV"),
                "mse_ref_eV2": sparse_row.get("mse_ref_eV2"),
                "r2_ref": sparse_row.get("r2_ref"),
                "rmse_union_eV": sparse_row.get("rmse_union_eV"),
                "mse_union_eV2": sparse_row.get("mse_union_eV2"),
                "r2_union": sparse_row.get("r2_union"),
                "relative_frobenius_union": sparse_row.get("relative_frobenius_union"),
                "support_f1": sparse_row.get("support_f1"),
                "global_rmse_eV": spectral_row.get("global_rmse_eV"),
                "low_energy_rmse_eV": spectral_row.get("low_energy_rmse_eV"),
                "fermi_window_rmse_eV": spectral_row.get("fermi_window_rmse_eV"),
                "frontier_window_rmse_eV": spectral_row.get("frontier_window_rmse_eV"),
                "gap_abs_error_eV": spectral_row.get("gap_abs_error_eV"),
                "fermi_level_source": spectral_row.get("fermi_level_source"),
                "fermi_metric_available": spectral_row.get("fermi_level_source") == "siesta_file",
            }
        )
    return rows


def matrix_spectrum_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fermi_rows = [
        row
        for row in rows
        if row.get("fermi_level_source") == "siesta_file"
    ]
    return {
        "samples": len(rows),
        "fermi_samples": len(fermi_rows),
        "corr_mae_ref_vs_global_rmse": pearson_correlation(rows, "mae_ref_eV", "global_rmse_eV"),
        "corr_rmse_ref_vs_global_rmse": pearson_correlation(rows, "rmse_ref_eV", "global_rmse_eV"),
        "corr_frobenius_vs_fermi_rmse": pearson_correlation(
            fermi_rows,
            "relative_frobenius_union",
            "fermi_window_rmse_eV",
        ),
        "corr_frobenius_vs_frontier_rmse": pearson_correlation(
            rows,
            "relative_frobenius_union",
            "frontier_window_rmse_eV",
        ),
        "corr_support_f1_vs_fermi_rmse": pearson_correlation(fermi_rows, "support_f1", "fermi_window_rmse_eV"),
        "spearman_corr_mae_ref_vs_global_rmse": spearman_correlation(rows, "mae_ref_eV", "global_rmse_eV"),
        "spearman_corr_frobenius_vs_fermi_rmse": spearman_correlation(
            fermi_rows, "relative_frobenius_union", "fermi_window_rmse_eV"
        ),
        "spearman_corr_frobenius_vs_frontier_rmse": spearman_correlation(
            rows, "relative_frobenius_union", "frontier_window_rmse_eV"
        ),
    }


def extract(
    result_dir: Path,
    *,
    low_energy_enabled: bool = True,
    low_energy_n_states: int = LOW_ENERGY_N_STATES,
    low_energy_alignment: str = LOW_ENERGY_ALIGNMENT,
    workers: int = 1,
) -> dict[str, Any]:
    low_energy_n_states, low_energy_alignment = validate_low_energy_config(
        low_energy_n_states,
        low_energy_alignment,
    )
    prediction_root = result_dir / "predicted_hamiltonians"
    reference_root = result_dir / "siesta_hamiltonians"
    eigen_root = result_dir / "eigenvalues"
    metrics_root = result_dir / "metrics"
    dos_root = result_dir / "dos"
    prediction_dirs = sample_dirs(prediction_root)
    reference_dirs = sample_dirs(reference_root)
    sample_names = sorted(set(prediction_dirs) | set(reference_dirs))
    basis_dirs = find_basis_dirs(result_dir)
    method_id = result_method_id(result_dir)
    target_component_policy = target_component_policy_from_result(result_dir)

    sparse_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    dos_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    sparse_sweep_rows: list[dict[str, Any]] = []
    dos_sweep_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    species_pair_rows: list[dict[str, Any]] = []
    distance_bin_rows: list[dict[str, Any]] = []
    orbital_pair_rows: list[dict[str, Any]] = []
    structural_unavailable: list[dict[str, str]] = []
    structural_basis_error = ""
    errors: list[dict[str, Any]] = []
    fatal_errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    sample_status: list[dict[str, Any]] = []
    try:
        basis_counts = basis_orbital_counts(basis_dirs)
        if not basis_counts:
            raise RuntimeError(
                "No .ion.xml basis files were found. Basis files are required for structural sparse metrics."
            )
    except Exception as exc:
        basis_counts = {}
        structural_basis_error = str(exc)
        warnings.append(
            {
                "sample": "*",
                "kind": "structural_basis",
                "severity": "severe",
                "error": structural_basis_error,
            }
        )

    def merge_sample_rows(sample_rows: dict[str, list[dict[str, Any]]]) -> None:
        sparse_rows.extend(sample_rows["sparse"])
        spectral_rows.extend(sample_rows["spectral"])
        dos_rows.extend(sample_rows["dos"])
        overlap_rows.extend(sample_rows["overlap"])
        sparse_sweep_rows.extend(sample_rows["sparse_sweep"])
        dos_sweep_rows.extend(sample_rows["dos_sweep"])
        component_rows.extend(sample_rows["component"])
        block_rows.extend(sample_rows["block"])
        species_pair_rows.extend(sample_rows["species_pair"])
        distance_bin_rows.extend(sample_rows["distance_bin"])
        orbital_pair_rows.extend(sample_rows["orbital_pair"])
        structural_unavailable.extend(sample_rows["structural_unavailable"])
        errors.extend(sample_rows["errors"])
        fatal_errors.extend(sample_rows.get("fatal_errors", []))
        warnings.extend(sample_rows.get("warnings", []))
        sample_status.extend(sample_rows.get("sample_status", []))

    worker_count = max(1, int(workers))
    if worker_count > 1 and len(sample_names) > 1:
        results_by_index: dict[int, dict[str, list[dict[str, Any]]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(
                    evaluate_sample,
                    sample,
                    prediction_dirs.get(sample),
                    reference_dirs.get(sample),
                    result_dir,
                    basis_counts,
                    method_id=method_id,
                    target_component_policy=target_component_policy,
                    low_energy_enabled=low_energy_enabled,
                    low_energy_n_states=low_energy_n_states,
                    low_energy_alignment=low_energy_alignment,
                ): index
                for index, sample in enumerate(sample_names)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                results_by_index[future_to_index[future]] = future.result()
        for index in sorted(results_by_index):
            merge_sample_rows(results_by_index[index])
    else:
        for sample in sample_names:
            merge_sample_rows(
                evaluate_sample(
                    sample,
                    prediction_dirs.get(sample),
                    reference_dirs.get(sample),
                    result_dir,
                    basis_counts,
                    method_id=method_id,
                    target_component_policy=target_component_policy,
                    low_energy_enabled=low_energy_enabled,
                    low_energy_n_states=low_energy_n_states,
                    low_energy_alignment=low_energy_alignment,
                )
            )

    orbital_pair_summary = orbital_pair_summary_rows(orbital_pair_rows)

    semantic_fields = MATRIX_SEMANTIC_FIELDS
    sparse_fields = [
        "sample",
        "n_orbitals",
        "n_entries",
        "ref_nnz",
        "pred_nnz",
        "union_nnz",
        "ref_density",
        "pred_density",
        "matrix_metric_target_space",
        *semantic_fields,
        "mae_ref_eV",
        "rmse_ref_eV",
        "mse_ref_eV2",
        "r2_ref",
        "mae_ref_meV",
        "rmse_ref_meV",
        "mae_pred_eV",
        "rmse_pred_eV",
        "mse_pred_eV2",
        "r2_pred",
        "mae_pred_meV",
        "rmse_pred_meV",
        "mae_union_eV",
        "rmse_union_eV",
        "mse_union_eV2",
        "r2_union",
        "mae_union_meV",
        "rmse_union_meV",
        "max_abs_error_union_eV",
        "relative_frobenius_ref",
        "relative_frobenius_union",
        "relative_l1_union",
        "support_precision",
        "support_recall",
        "support_f1",
        "false_zeros",
        "false_nonzeros",
        "false_zero_rate",
        "false_nonzero_rate",
        "weighted_false_zeros_eV",
        "weighted_false_nonzeros_eV",
        "hermiticity_ref",
        "hermiticity_pred",
    ]
    spectral_fields = [
        "sample",
        "siesta_bands",
        "predicted_bands",
        *semantic_fields,
        "spectral_comparable",
        "same_band_count",
        "reference_has_overlap",
        "hamiltonian_symmetrized_for_spectrum",
        "n_compared_bands",
        "fermi_ref_eV",
        "fermi_level_source",
        "fermi_metric_available",
        "global_mae_eV",
        "global_rmse_eV",
        "global_max_abs_error_eV",
        "global_mean_signed_error_eV",
        "occupied_bands",
        "occupied_metric_available",
        "occupied_mae_eV",
        "occupied_rmse_eV",
        "fermi_window_eV",
        "fermi_window_bands",
        "fermi_window_metric_available",
        "fermi_window_mae_eV",
        "fermi_window_rmse_eV",
        "gap_ref_eV",
        "gap_pred_eV",
        "gap_abs_error_eV",
        "homo_index",
        "lumo_index",
        "homo_error_eV",
        "lumo_error_eV",
        "frontier_window_bands",
        "frontier_metric_available",
        "frontier_window_mae_eV",
        "frontier_window_rmse_eV",
        "align_global_shift_eV",
        "align_global_mae_eV",
        "align_global_rmse_eV",
        "align_fermi_shift_eV",
        "align_fermi_mae_eV",
        "align_fermi_rmse_eV",
        "align_homo_shift_eV",
        "align_homo_mae_eV",
        "align_homo_rmse_eV",
        "low_energy_requested_states",
        "low_energy_n_states",
        "low_energy_mae_eV",
        "low_energy_rmse_eV",
        "low_energy_max_abs_error_eV",
        "low_energy_alignment",
        "low_energy_aligned_rmse_eV",
        "low_energy_overlap_used",
        "low_energy_overlap_required",
        "low_energy_solver",
        "low_energy_warning",
    ]
    relationship_rows = matrix_spectrum_rows(sparse_rows, spectral_rows)
    relationship_fields = [
        "sample",
        "matrix_metric_target_space",
        *semantic_fields,
        "mae_ref_eV",
        "rmse_ref_eV",
        "mse_ref_eV2",
        "r2_ref",
        "rmse_union_eV",
        "mse_union_eV2",
        "r2_union",
        "relative_frobenius_union",
        "support_f1",
        "global_rmse_eV",
        "low_energy_rmse_eV",
        "fermi_window_rmse_eV",
        "frontier_window_rmse_eV",
        "gap_abs_error_eV",
        "fermi_level_source",
        "fermi_metric_available",
    ]
    dos_fields = [
        "sample",
        "dos_sigma_eV",
        "dos_grid_points",
        "energy_min_eV",
        "energy_max_eV",
        "dos_wasserstein_eV",
        "dos_l1",
        "dos_l2",
        "dos_mae_500_fermi_window",
        "dos_window_min_eV",
        "dos_window_max_eV",
        "dos_window_points",
        "dos_window_sigma_eV",
        "dos_window_alignment",
        "dos_window_metric_available",
        "dos_window_unavailable_reason",
    ]
    dos_sweep_fields = [
        "sample",
        "dos_sigma_eV",
        "dos_grid_points",
        "energy_min_eV",
        "energy_max_eV",
        "dos_wasserstein_eV",
        "dos_l1",
        "dos_l2",
    ]
    sparse_sweep_fields = ["sample", "support_threshold", "union_nnz", "mae_union_eV", "rmse_union_eV"]
    overlap_fields = [
        "sample",
        "kind",
        "matrix_path",
        "n_bands",
        "hamiltonian_components",
        "spin_kind",
        *semantic_fields,
        "orthogonal",
        "has_overlap",
        "overlap_error",
        "fermi_level_eV",
    ]
    component_fields = [
        "sample",
        "component_index",
        "component_role",
        "reference_component_available",
        "prediction_component_available",
        "component_shape",
        "component_metric_available",
        "component_unavailable_reason",
        "component_n_entries",
        "component_mae_eV",
        "component_rmse_eV",
        "component_mse_eV2",
        "component_max_abs_error_eV",
        *semantic_fields,
    ]
    block_fields = [
        "sample",
        "row_atom",
        "col_atom",
        "row_species",
        "col_species",
        "n_entries",
        "mae_union_eV",
        "rmse_union_eV",
        "max_abs_error_union_eV",
        "mean_distance_ang",
    ]
    species_pair_fields = [
        "sample",
        "species_pair",
        "n_entries",
        "mae_union_eV",
        "rmse_union_eV",
        "max_abs_error_union_eV",
        "mean_distance_ang",
    ]
    distance_bin_fields = [
        "sample",
        "distance_bin",
        "n_entries",
        "mae_union_eV",
        "rmse_union_eV",
        "max_abs_error_union_eV",
        "mean_distance_ang",
    ]
    orbital_pair_fields = [
        "sample",
        "row_species",
        "col_species",
        "species_pair",
        "row_orbital_index",
        "col_orbital_index",
        "row_orbital_label",
        "col_orbital_label",
        "n_entries",
        "mae_union_eV",
        "mae_union_meV",
        "mse_union_eV2",
        "rmse_union_eV",
        "r2_union",
        "max_abs_error_union_eV",
        "mean_abs_ref_eV",
        "mean_signed_error_eV",
        "metric_target_space",
        "basis_source",
    ]
    orbital_pair_summary_metric_fields = [
        f"{metric_name}_{statistic}"
        for metric_name in [
            "mae_union_eV",
            "mae_union_meV",
            "mse_union_eV2",
            "rmse_union_eV",
            "r2_union",
            "max_abs_error_union_eV",
            "mean_abs_ref_eV",
            "mean_signed_error_eV",
        ]
        for statistic in ["mean", "std", "min", "max"]
    ]
    orbital_pair_summary_fields = [
        "row_species",
        "col_species",
        "species_pair",
        "row_orbital_index",
        "col_orbital_index",
        "row_orbital_label",
        "col_orbital_label",
        "n_samples",
        "n_entries",
        "metric_target_space",
        "basis_source",
        *orbital_pair_summary_metric_fields,
    ]

    write_csv(metrics_root / "sparse_metrics.csv", sparse_fields, sparse_rows)
    write_csv(metrics_root / "spectral_metrics.csv", spectral_fields, spectral_rows)
    write_csv(metrics_root / "dos_metrics.csv", dos_fields, dos_rows)
    write_csv(metrics_root / "matrix_spectrum_relationship.csv", relationship_fields, relationship_rows)
    write_csv(metrics_root / "sparse_threshold_sweep.csv", sparse_sweep_fields, sparse_sweep_rows)
    write_csv(metrics_root / "component_channel_metrics.csv", component_fields, component_rows)
    write_csv(metrics_root / "block_metrics.csv", block_fields, block_rows)
    write_csv(metrics_root / "species_pair_metrics.csv", species_pair_fields, species_pair_rows)
    write_csv(metrics_root / "distance_bin_metrics.csv", distance_bin_fields, distance_bin_rows)
    write_csv(metrics_root / "orbital_pair_metrics.csv", orbital_pair_fields, orbital_pair_rows)
    write_csv(metrics_root / "orbital_pair_summary.csv", orbital_pair_summary_fields, orbital_pair_summary)
    write_csv(eigen_root / "eigenvalue_metrics.csv", spectral_fields, spectral_rows)
    write_csv(eigen_root / "overlap_summary.csv", overlap_fields, overlap_rows)
    write_csv(metrics_root / "dos_sigma_sweep.csv", dos_sweep_fields, dos_sweep_rows)

    summary = {
        "sparse": summarize_numeric(sparse_rows, {"sample", *semantic_fields}),
        "spectral": summarize_numeric(
            spectral_rows,
            {"sample", "hamiltonian_symmetrized_for_spectrum", *semantic_fields},
        ),
        "dos": summarize_numeric(dos_rows, {"sample"}),
        "component_channel": summarize_numeric(
            component_rows,
            {
                "sample",
                "component_role",
                "component_shape",
                "component_unavailable_reason",
                *semantic_fields,
            },
        ),
        "orbital_pair": summarize_numeric(
            orbital_pair_rows,
            {
                "sample",
                "row_species",
                "col_species",
                "species_pair",
                "row_orbital_index",
                "col_orbital_index",
                "row_orbital_label",
                "col_orbital_label",
                "metric_target_space",
                "basis_source",
            },
        ),
        "orbital_pair_summary": summarize_numeric(
            orbital_pair_summary,
            {
                "row_species",
                "col_species",
                "species_pair",
                "row_orbital_index",
                "col_orbital_index",
                "row_orbital_label",
                "col_orbital_label",
                "metric_target_space",
                "basis_source",
            },
        ),
        "matrix_spectrum": matrix_spectrum_summary(relationship_rows),
        "metric_availability": {
            **metric_availability(
                spectral_rows,
                [
                    "global_rmse_eV",
                    "occupied_rmse_eV",
                    "fermi_window_rmse_eV",
                    "frontier_window_rmse_eV",
                    "low_energy_rmse_eV",
                    "gap_abs_error_eV",
                ],
            ),
            **metric_availability(
                sparse_rows,
                [
                    "relative_frobenius_union",
                    "mae_ref_eV",
                    "mse_ref_eV2",
                    "r2_ref",
                    "mse_union_eV2",
                    "r2_union",
                    "support_f1",
                ],
            ),
            **metric_availability(
                dos_rows,
                ["dos_wasserstein_eV", "dos_mae_500_fermi_window"],
            ),
        },
        "orbital_pair_metric_availability": metric_availability(
            orbital_pair_rows,
            ["mae_union_eV", "mse_union_eV2", "rmse_union_eV", "r2_union"],
        ),
    }
    severe_warnings = [
        issue
        for issue in warnings
        if str(issue.get("severity") or "").lower() in {"severe", "fatal"}
    ]
    manifest = {
        "result_dir": str(result_dir),
        "samples_seen": len(sample_names),
        "samples_compared": len(spectral_rows),
        "samples_failed": len([row for row in sample_status if row.get("status") == "failed"]),
        "sparse_samples": len(sparse_rows),
        "dos_samples": len(dos_rows),
        "overlap_entries": len(overlap_rows),
        "component_channel_entries": len(component_rows),
        "target_component_policy": target_component_policy,
        "reference_selection_policy": REFERENCE_SELECTION_POLICY,
        "sample_status": sample_status,
        "structural_metrics_basis_required": True,
        "structural_basis_dirs": [str(path) for path in basis_dirs],
        "structural_basis_orbital_counts": basis_counts,
        "structural_basis_error": structural_basis_error,
        "structural_metrics_error": bool(structural_basis_error or structural_unavailable),
        "structural_metrics_available": bool(block_rows or species_pair_rows or distance_bin_rows or orbital_pair_rows),
        "structural_metrics_samples": len(
            {
                str(row.get("sample"))
                for row in [*block_rows, *species_pair_rows, *distance_bin_rows, *orbital_pair_rows]
            }
        ),
        "orbital_pair_metrics_available": bool(orbital_pair_rows),
        "orbital_pair_metrics_samples": len({str(row.get("sample")) for row in orbital_pair_rows}),
        "orbital_pair_metric_target_space": ORBITAL_PAIR_METRIC_TARGET_SPACE,
        "orbital_pair_basis_source": ORBITAL_PAIR_BASIS_SOURCE,
        "structural_metrics_unavailable": structural_unavailable,
        "support_threshold": SUPPORT_THRESHOLD,
        "fermi_window_eV": FERMI_WINDOW_EV,
        "dos_sigma_eV": DOS_SIGMA_EV,
        "dos_sigma_sweep_eV": DOS_SIGMA_SWEEP_EV,
        "dos_fermi_window": {
            "points": DOS_FERMI_WINDOW_POINTS,
            "relative_energy_min_eV": DOS_FERMI_WINDOW_MIN_EV,
            "relative_energy_max_eV": DOS_FERMI_WINDOW_MAX_EV,
            "sigma_eV": DOS_SIGMA_EV,
            "alignment": DOS_FERMI_WINDOW_ALIGNMENT,
            "requires_real_siesta_fermi_level": True,
        },
        "low_energy": {
            "enabled": low_energy_enabled,
            "n_states": low_energy_n_states,
            "alignment": low_energy_alignment,
            "required_primary_metric_blocks_robustness": True,
        },
        "support_threshold_sweep": SUPPORT_THRESHOLDS_SWEEP,
        "dos_points": DOS_POINTS,
        "metric_workers": max(1, int(workers)),
        "fatal_errors": fatal_errors,
        "warnings": warnings,
        "severe_warnings": severe_warnings,
        "errors": errors,
        "metric_compatibility": {
            "sparse_matrix_metrics_material_agnostic": True,
            "matrix_metric_target_space": MATRIX_METRIC_TARGET_SPACE,
            "target_component_policy": target_component_policy,
            "target_semantics_fields": semantic_fields,
            "prediction_own_overlap_used_for_spectra": False,
            "nonorthogonal_spectral_overlap_source": "siesta_reference",
            "prediction_self_contained_hsx_safe_required": True,
            "orbital_pair_metric_target_space": ORBITAL_PAIR_METRIC_TARGET_SPACE,
            "orbital_pair_basis_source": ORBITAL_PAIR_BASIS_SOURCE,
            "deeph_hprime_transform_applied": False,
            "deeph_orbital_hprime_transform_applied": False,
            "complex_hamiltonians_supported": False,
            "spin_polarized_supported": False,
            "kpoint_sampled_supported": False,
            "periodic_distance_bins_supported": False,
            "nonorthogonal_spectral_requires_reference_overlap": True,
            "structural_metrics_require_basis_species_coverage": True,
        },
        "deeph_comparability_status": DEEPH_COMPARABILITY_STATUS,
        "summary": summary,
        "recommendation_metric_policy": {
            "primary_metric_priority": RECOMMENDATION_PRIMARY_METRIC_PRIORITY,
            "diagnostic_only_metrics": DIAGNOSTIC_ONLY_RECOMMENDATION_METRICS,
            "missing_fermi_metrics_are_unavailable_not_zero": True,
            "global_rmse_recommendation_role": "diagnostic_only",
        },
        "outputs": {
            "metrics_root": str(metrics_root),
            "sparse_metrics": str(metrics_root / "sparse_metrics.csv"),
            "spectral_metrics": str(metrics_root / "spectral_metrics.csv"),
            "dos_metrics": str(metrics_root / "dos_metrics.csv"),
            "matrix_spectrum_relationship": str(metrics_root / "matrix_spectrum_relationship.csv"),
            "component_channel_metrics": str(metrics_root / "component_channel_metrics.csv"),
            "block_metrics": str(metrics_root / "block_metrics.csv"),
            "species_pair_metrics": str(metrics_root / "species_pair_metrics.csv"),
            "distance_bin_metrics": str(metrics_root / "distance_bin_metrics.csv"),
            "orbital_pair_metrics": str(metrics_root / "orbital_pair_metrics.csv"),
            "orbital_pair_summary": str(metrics_root / "orbital_pair_summary.csv"),
            "eigenvalues_siesta": str(eigen_root / "siesta"),
            "eigenvalues_predicted": str(eigen_root / "predicted"),
            "band_errors": str(eigen_root / "band_errors"),
            "dos": str(dos_root),
            "overlap_summary": str(eigen_root / "overlap_summary.csv"),
        },
    }
    (eigen_root / "manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (metrics_root / "manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--disable-low-energy", action="store_true")
    parser.add_argument("--low-energy-n-states", type=int, default=LOW_ENERGY_N_STATES)
    parser.add_argument("--low-energy-alignment", default=LOW_ENERGY_ALIGNMENT, choices=["none", "global_shift"])
    parser.add_argument("--workers", type=int, default=1, help="Parallel sample workers for metric extraction.")
    args = parser.parse_args()
    manifest = extract(
        args.result_dir,
        low_energy_enabled=not args.disable_low_energy,
        low_energy_n_states=args.low_energy_n_states,
        low_energy_alignment=args.low_energy_alignment,
        workers=args.workers,
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, allow_nan=False))
    return 0 if not manifest["fatal_errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
