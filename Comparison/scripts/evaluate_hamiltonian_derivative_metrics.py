#!/usr/bin/env python3
"""Evaluate finite-difference Hamiltonian derivative metrics from archived matrices."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
from scipy import sparse
import sisl

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hamiltonian_derivative_stencil import (  # noqa: E402
    DERIVATIVE_SUPPORT_THRESHOLD,
    EXPECTED_DERIVATIVE_UNITS,
    DerivativeMatrixInput,
    DerivativeMetadata,
    DerivativeStencil,
    DerivativeStencilDiscovery,
    derivative_sparse_metrics,
    discover_derivative_stencils,
    finite_difference_derivative_pair,
    validate_derivative_geometry,
    validate_derivative_stencil,
    validation_errors,
)


SCHEMA_VERSION = "hamiltonian_derivative_metrics_v1"
REFERENCE_DEFINITION = "siesta_hamiltonian_finite_difference"
SUPPORT_THRESHOLDS_SWEEP = (1e-12, 1e-10, 1e-8, 1e-6)
STATUS_FIELDS = [
    "sample",
    "status",
    "finite_difference_method",
    "base_sample_id",
    "plus_sample_id",
    "minus_sample_id",
    "atom_index_zero_based",
    "axis",
    "axis_index",
    "delta_ang",
    "issue_codes",
    "issue_messages",
]
HERMITICITY_FIELDS = [
    "sample",
    "finite_difference_method",
    "source_model",
    "reference_source",
    "dH_ref_hermiticity_defect",
    "dH_pred_hermiticity_defect",
    "dH_hermiticity_error_delta",
    "finite_values",
]
GEOMETRY_VALIDATION_FIELDS = [
    "sample",
    "status",
    "finite_difference_method",
    "base_sample_id",
    "plus_sample_id",
    "minus_sample_id",
    "atom_index_zero_based",
    "axis",
    "axis_index",
    "delta_ang",
    "issue_codes",
    "issue_messages",
]
DELTA_STABILITY_FIELDS = [
    "source_model",
    "base_sample_id",
    "atom_index_zero_based",
    "axis",
    "finite_difference_method",
    "delta_count",
    "delta_min_ang",
    "delta_max_ang",
    "dh_mae_union_eV_per_Ang_min",
    "dh_mae_union_eV_per_Ang_max",
    "dh_mae_union_eV_per_Ang_range",
    "dh_rmse_union_eV_per_Ang_min",
    "dh_rmse_union_eV_per_Ang_max",
    "dh_rmse_union_eV_per_Ang_range",
    "dh_relative_frobenius_ref_min",
    "dh_relative_frobenius_ref_max",
    "dh_relative_frobenius_ref_range",
    "status",
]


class DerivativeMetricEvaluationError(RuntimeError):
    """Raised when derivative metric outputs cannot be written or evaluated."""


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted(
        {
            str(key)
            for row in rows
            for key in row
            if str(key) not in fieldnames
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra_fields])
        writer.writeheader()
        writer.writerows([{key: json_safe(value) for key, value in row.items()} for row in rows])


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise DerivativeMetricEvaluationError(
                f"Refusing to overwrite existing derivative metric outputs: {output_dir}. Pass --overwrite to replace them."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def load_hamiltonian_matrix(path: Path) -> sparse.csr_matrix:
    """Load a sparse Hamiltonian matrix.

    Production archives are read through sisl. The scipy .npz attempt keeps unit
    tests lightweight and is harmless for real HSX/TSHS files because invalid
    zip payloads fall through to sisl.
    """

    try:
        return sparse.load_npz(path).tocsr()
    except (OSError, ValueError, BadZipFile):
        pass
    sile = sisl.get_sile(str(path))
    hamiltonian_obj = sile.read_hamiltonian()
    return hamiltonian_obj.tocsr(0).tocsr()


def evaluate_derivative_metrics(
    result_dir: Path,
    *,
    method: str,
    split: str = "all",
    require_central: bool = False,
    overwrite: bool = False,
    diagnostic_only: bool = False,
    support_threshold: float = DERIVATIVE_SUPPORT_THRESHOLD,
    max_stencils: int | None = None,
    output_dir: Path | None = None,
    source_model: str = "graph2mat",
) -> dict[str, Any]:
    result_dir = Path(result_dir)
    output_dir = Path(output_dir) if output_dir is not None else result_dir / "derivative_metrics"
    ensure_output_dir(output_dir, overwrite=overwrite)

    discoveries = discover_derivative_stencils(
        result_dir,
        method=source_model,
        split=split,
        finite_difference_method=method,
        require_central=require_central,
    )
    if max_stencils is not None:
        discoveries = discoveries[: max(0, int(max_stencils))]

    stencil_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    hermiticity_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    fatal_errors: list[dict[str, Any]] = []

    for discovery in discoveries:
        row, metrics, sweep, hermiticity, geometry, warning_rows, error_rows = _evaluate_discovery(
            discovery,
            method=method,
            source_model=source_model,
            support_threshold=support_threshold,
            diagnostic_only=diagnostic_only,
        )
        stencil_rows.append(row)
        metric_rows.extend(metrics)
        sweep_rows.extend(sweep)
        hermiticity_rows.extend(hermiticity)
        geometry_rows.append(geometry)
        warnings.extend(warning_rows)
        fatal_errors.extend(error_rows)

    stencils_total = len(discoveries)
    stencils_ok = len(metric_rows)
    stencils_failed = stencils_total - stencils_ok
    scientific_status = _scientific_status(
        method=method,
        diagnostic_only=diagnostic_only,
        stencils_total=stencils_total,
        stencils_ok=stencils_ok,
        stencils_failed=stencils_failed,
        metric_rows=metric_rows,
        fatal_errors=fatal_errors,
    )
    delta_stability = _delta_stability_summary(metric_rows)
    reference_noise = _reference_noise_summary(metric_rows)
    summary = _summary(metric_rows, stencil_rows, hermiticity_rows)
    summary["delta_stability"] = delta_stability
    summary["reference_noise"] = reference_noise
    outputs = {
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "stencil_status": str(output_dir / "stencil_status.csv"),
        "derivative_matrix_metrics": str(output_dir / "derivative_matrix_metrics.csv"),
        "derivative_support_sweep": str(output_dir / "derivative_support_sweep.csv"),
        "derivative_hermiticity": str(output_dir / "derivative_hermiticity.csv"),
        "derivative_delta_stability": str(output_dir / "derivative_delta_stability.csv"),
        "derivative_delta_stability_json": str(output_dir / "derivative_delta_stability.json"),
        "derivative_geometry_validation": str(output_dir / "derivative_geometry_validation.csv"),
        "derivative_geometry_validation_json": str(output_dir / "derivative_geometry_validation.json"),
        "derivative_summary": str(output_dir / "derivative_summary.json"),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scientific_status": scientific_status,
        "paper_level": False,
        "finite_difference_method": method,
        "force_constants_used": False,
        "reference_definition": REFERENCE_DEFINITION,
        "derivative_units": EXPECTED_DERIVATIVE_UNITS,
        "result_dir": str(result_dir),
        "split": split,
        "require_central": bool(require_central),
        "diagnostic_only_requested": bool(diagnostic_only),
        "support_threshold": float(support_threshold),
        "support_threshold_sweep": list(SUPPORT_THRESHOLDS_SWEEP),
        "stencils_total": stencils_total,
        "stencils_ok": stencils_ok,
        "stencils_failed": stencils_failed,
        "geometry_validation": _geometry_validation_summary(geometry_rows),
        "delta_stability": delta_stability,
        "delta_sensitivity_study_passed": delta_stability["status"] == "available",
        "reference_noise": reference_noise,
        "reference_noise_status": reference_noise["status"],
        "warnings": warnings,
        "fatal_errors": fatal_errors,
        "outputs": outputs,
    }

    write_csv(output_dir / "stencil_status.csv", STATUS_FIELDS, stencil_rows)
    write_csv(output_dir / "derivative_matrix_metrics.csv", _metric_fieldnames(metric_rows), metric_rows)
    write_csv(output_dir / "derivative_support_sweep.csv", _metric_fieldnames(sweep_rows), sweep_rows)
    write_csv(output_dir / "derivative_hermiticity.csv", HERMITICITY_FIELDS, hermiticity_rows)
    write_csv(output_dir / "derivative_delta_stability.csv", DELTA_STABILITY_FIELDS, delta_stability["rows"])
    write_json(output_dir / "derivative_delta_stability.json", delta_stability)
    write_csv(output_dir / "derivative_geometry_validation.csv", GEOMETRY_VALIDATION_FIELDS, geometry_rows)
    write_json(output_dir / "derivative_geometry_validation.json", _geometry_validation_summary(geometry_rows, include_rows=True))
    write_json(output_dir / "derivative_summary.json", summary)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _evaluate_discovery(
    discovery: DerivativeStencilDiscovery,
    *,
    method: str,
    source_model: str,
    support_threshold: float,
    diagnostic_only: bool,
) -> tuple[
    list[Any] | dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    warnings: list[dict[str, Any]] = []
    fatal_errors: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    hermiticity_rows: list[dict[str, Any]] = []
    status_row = _stencil_status_row(discovery, status=discovery.status)
    geometry_issues = validate_derivative_geometry(discovery)
    geometry_row = _geometry_validation_row(discovery, geometry_issues)
    geometry_errors = validation_errors(geometry_issues)
    if discovery.stencil is None:
        fatal_errors.append(_discovery_error(discovery, "missing_stencil", "Discovery did not produce a complete stencil."))
        return status_row, metric_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors
    if geometry_errors and not diagnostic_only:
        status_row = _stencil_status_row(
            replace(discovery, issues=tuple([*discovery.issues, *geometry_issues])),
            status="failed",
        )
        fatal_errors.append(
            _discovery_error(
                discovery,
                "derivative_geometry_validation_failed",
                "Derivative geometry validation failed before metric evaluation.",
                issue_codes=[issue.code for issue in geometry_errors],
            )
        )
        return status_row, metric_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors
    if geometry_errors:
        warnings.append(
            _discovery_error(
                discovery,
                "derivative_geometry_validation_diagnostic_only",
                "Derivative geometry validation failed, but diagnostic-only mode allowed metric evaluation to continue.",
                issue_codes=[issue.code for issue in geometry_errors],
            )
        )

    try:
        loaded = _load_stencil_matrices(discovery.stencil)
        stencil = _stencil_with_loaded_shapes(discovery.stencil, loaded)
        validation = validate_derivative_stencil(stencil)
        errors = validation_errors(validation)
        if errors:
            status_row = _stencil_status_row(
                replace(discovery, stencil=stencil, issues=tuple([*discovery.issues, *validation])),
                status="failed",
            )
            fatal_errors.append(
                _discovery_error(
                    discovery,
                    "stencil_validation_failed",
                    "Derivative stencil validation failed after loading matrices.",
                    issue_codes=[issue.code for issue in errors],
                )
            )
            return status_row, metric_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors
        metadata = _metadata_for_status(stencil.metadata, diagnostic_only=diagnostic_only)
        pair = finite_difference_derivative_pair(
            method=method,
            delta_ang=float(metadata.delta_ang),
            reference_plus=loaded.get("siesta_plus"),
            reference_minus=loaded.get("siesta_minus"),
            reference_base=loaded.get("siesta_base"),
            predicted_plus=loaded.get("ml_plus"),
            predicted_minus=loaded.get("ml_minus"),
            predicted_base=loaded.get("ml_base"),
            predicted_source=source_model,
            reference_hashes=_matrix_hashes(stencil, prefix="siesta"),
            predicted_hashes=_matrix_hashes(stencil, prefix="ml"),
            metadata=metadata,
        )
        row = derivative_sparse_metrics(
            pair.reference.matrix,
            pair.predicted.matrix,
            sample=metadata.sample_id,
            metadata=metadata,
            source_model=source_model,
            reference_source="siesta",
            support_threshold=support_threshold,
        )
        row.update(
            {
                "dh_support_changed": bool(pair.diagnostics.get("plus_minus_support_changed")),
                "reference_plus_minus_support_changed": bool(pair.diagnostics.get("reference_plus_minus_support_changed")),
                "predicted_plus_minus_support_changed": bool(pair.diagnostics.get("predicted_plus_minus_support_changed")),
            }
        )
        metric_rows.append(row)
        for threshold in SUPPORT_THRESHOLDS_SWEEP:
            sweep = derivative_sparse_metrics(
                pair.reference.matrix,
                pair.predicted.matrix,
                sample=metadata.sample_id,
                metadata=metadata,
                source_model=source_model,
                reference_source="siesta",
                support_threshold=threshold,
            )
            sweep_rows.append(
                {
                    "sample": metadata.sample_id,
                    "support_threshold": threshold,
                    "dh_union_nnz": sweep["dh_union_nnz"],
                    "dh_mae_union_eV_per_Ang": sweep["dh_mae_union_eV_per_Ang"],
                    "dh_rmse_union_eV_per_Ang": sweep["dh_rmse_union_eV_per_Ang"],
                    "dh_support_precision": sweep["dh_support_precision"],
                    "dh_support_recall": sweep["dh_support_recall"],
                    "dh_support_f1": sweep["dh_support_f1"],
                }
            )
        hermiticity_rows.append(
            {
                "sample": metadata.sample_id,
                "finite_difference_method": method,
                "source_model": source_model,
                "reference_source": "siesta",
                "dH_ref_hermiticity_defect": pair.diagnostics["dH_ref_hermiticity_defect"],
                "dH_pred_hermiticity_defect": pair.diagnostics["dH_pred_hermiticity_defect"],
                "dH_hermiticity_error_delta": row["dh_hermiticity_error_delta"],
                "finite_values": pair.diagnostics["finite_values"],
            }
        )
        status_row = _stencil_status_row(replace(discovery, stencil=stencil, issues=tuple(validation)), status="ok")
    except Exception as exc:  # Backend-specific sisl readers raise heterogeneous exceptions.
        status_row = _stencil_status_row(discovery, status="failed")
        fatal_errors.append(_discovery_error(discovery, "derivative_metric_evaluation_failed", str(exc)))
    return status_row, metric_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors


def _load_stencil_matrices(stencil: DerivativeStencil) -> dict[str, sparse.csr_matrix]:
    loaded: dict[str, sparse.csr_matrix] = {}
    for role, matrix_input in stencil.matrix_inputs().items():
        if matrix_input is None:
            continue
        if matrix_input.matrix_path is None:
            continue
        loaded[role] = load_hamiltonian_matrix(Path(matrix_input.matrix_path))
    return loaded


def _stencil_with_loaded_shapes(stencil: DerivativeStencil, loaded: dict[str, sparse.csr_matrix]) -> DerivativeStencil:
    updates: dict[str, DerivativeMatrixInput | None] = {}
    for role, matrix_input in stencil.matrix_inputs().items():
        if matrix_input is None:
            updates[role] = None
            continue
        matrix = loaded.get(role)
        updates[role] = replace(matrix_input, matrix_shape=tuple(matrix.shape)) if matrix is not None else matrix_input
    return replace(stencil, **updates)


def _metadata_for_status(metadata: DerivativeMetadata, *, diagnostic_only: bool) -> DerivativeMetadata:
    if diagnostic_only:
        return replace(metadata, claim_status="diagnostic_only")
    return metadata


def _matrix_hashes(stencil: DerivativeStencil, *, prefix: str) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for role, matrix_input in stencil.matrix_inputs().items():
        if not role.startswith(prefix) or matrix_input is None:
            continue
        hashes[role.removeprefix(prefix + "_")] = matrix_input.matrix_sha256
    return hashes


def _stencil_status_row(discovery: DerivativeStencilDiscovery, *, status: str) -> dict[str, Any]:
    metadata = discovery.stencil.metadata if discovery.stencil else None
    issue_codes = [issue.code for issue in discovery.issues]
    issue_messages = [issue.message for issue in discovery.issues]
    return {
        "sample": metadata.sample_id if metadata else "|".join(discovery.sample_ids),
        "status": status,
        "finite_difference_method": discovery.method,
        "base_sample_id": metadata.base_sample_id if metadata else None,
        "plus_sample_id": metadata.plus_sample_id if metadata else None,
        "minus_sample_id": metadata.minus_sample_id if metadata else None,
        "atom_index_zero_based": metadata.atom_index_zero_based if metadata else None,
        "axis": metadata.axis if metadata else None,
        "axis_index": metadata.axis_index if metadata else None,
        "delta_ang": metadata.delta_ang if metadata else None,
        "issue_codes": ";".join(issue_codes),
        "issue_messages": "; ".join(issue_messages),
    }


def _geometry_validation_row(discovery: DerivativeStencilDiscovery, issues: list[Any]) -> dict[str, Any]:
    metadata = discovery.stencil.metadata if discovery.stencil else None
    errors = validation_errors(issues)
    warnings = [issue for issue in issues if not issue.is_error]
    if errors:
        status = "error"
    elif warnings:
        status = "warning"
    else:
        status = "ok"
    return {
        "sample": metadata.sample_id if metadata else "|".join(discovery.sample_ids),
        "status": status,
        "finite_difference_method": discovery.method,
        "base_sample_id": metadata.base_sample_id if metadata else None,
        "plus_sample_id": metadata.plus_sample_id if metadata else None,
        "minus_sample_id": metadata.minus_sample_id if metadata else None,
        "atom_index_zero_based": metadata.atom_index_zero_based if metadata else None,
        "axis": metadata.axis if metadata else None,
        "axis_index": metadata.axis_index if metadata else None,
        "delta_ang": metadata.delta_ang if metadata else None,
        "issue_codes": ";".join(issue.code for issue in issues),
        "issue_messages": "; ".join(issue.message for issue in issues),
    }


def _geometry_validation_summary(rows: list[dict[str, Any]], *, include_rows: bool = False) -> dict[str, Any]:
    summary = {
        "total": len(rows),
        "ok": len([row for row in rows if row.get("status") == "ok"]),
        "warnings": len([row for row in rows if row.get("status") == "warning"]),
        "errors": len([row for row in rows if row.get("status") == "error"]),
    }
    if include_rows:
        summary["rows"] = rows
    return summary


def _discovery_error(discovery: DerivativeStencilDiscovery, kind: str, message: str, **extra: Any) -> dict[str, Any]:
    metadata = discovery.stencil.metadata if discovery.stencil else None
    return {
        "sample": metadata.sample_id if metadata else "|".join(discovery.sample_ids),
        "kind": kind,
        "message": message,
        "status": discovery.status,
        **extra,
    }


def _scientific_status(
    *,
    method: str,
    diagnostic_only: bool,
    stencils_total: int,
    stencils_ok: int,
    stencils_failed: int,
    metric_rows: list[dict[str, Any]],
    fatal_errors: list[dict[str, Any]],
) -> str:
    if diagnostic_only or method != "central" or not metric_rows:
        return "diagnostic_only"
    if stencils_total == stencils_ok and stencils_failed == 0 and not fatal_errors:
        if all(
            row.get("comparison_status") != "diagnostic_only"
            and row.get("derivative_units") == EXPECTED_DERIVATIVE_UNITS
            and row.get("finite_difference_method") == "central"
            and row.get("dh_finite_values") is True
            and _finite_or_nan(row.get("dh_hermiticity_ref")) < 1e-8
            and _finite_or_nan(row.get("dh_hermiticity_pred")) < 1e-8
            for row in metric_rows
        ):
            return "presentation_ready"
    return "diagnostic_only"


def _finite_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _summary(
    metric_rows: list[dict[str, Any]],
    stencil_rows: list[dict[str, Any]],
    hermiticity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stencils_total": len(stencil_rows),
        "metric_rows": len(metric_rows),
        "failed_stencils": len([row for row in stencil_rows if row.get("status") != "ok"]),
        "mean_dh_mae_union_eV_per_Ang": _mean(row.get("dh_mae_union_eV_per_Ang") for row in metric_rows),
        "mean_dh_rmse_union_eV_per_Ang": _mean(row.get("dh_rmse_union_eV_per_Ang") for row in metric_rows),
        "mean_dh_relative_frobenius_ref": _mean(row.get("dh_relative_frobenius_ref") for row in metric_rows),
        "max_dh_hermiticity_ref": _max(row.get("dH_ref_hermiticity_defect") for row in hermiticity_rows),
        "max_dh_hermiticity_pred": _max(row.get("dH_pred_hermiticity_defect") for row in hermiticity_rows),
        "force_constants_used": False,
        "reference_definition": REFERENCE_DEFINITION,
    }


def _delta_stability_group_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("source_model") or ""),
        str(row.get("base_sample_id") or ""),
        str(row.get("atom_index_zero_based") or ""),
        str(row.get("axis") or ""),
        str(row.get("finite_difference_method") or ""),
    )


def _finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = [_finite_or_nan(row.get(field)) for row in rows]
    return [value for value in values if math.isfinite(value)]


def _range_payload(rows: list[dict[str, Any]], field: str, prefix: str) -> dict[str, float | None]:
    values = _finite_values(rows, field)
    if not values:
        return {
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_range": None,
        }
    minimum = min(values)
    maximum = max(values)
    return {
        f"{prefix}_min": minimum,
        f"{prefix}_max": maximum,
        f"{prefix}_range": maximum - minimum,
    }


def _delta_stability_summary(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in metric_rows:
        delta = _finite_or_nan(row.get("delta_ang"))
        if not math.isfinite(delta):
            continue
        groups.setdefault(_delta_stability_group_key(row), []).append(row)

    rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        deltas = sorted({float(row.get("delta_ang")) for row in group_rows if math.isfinite(_finite_or_nan(row.get("delta_ang")))})
        if len(deltas) < 2:
            continue
        source_model, base_sample_id, atom_index_zero_based, axis, method = key
        row = {
            "source_model": source_model,
            "base_sample_id": base_sample_id,
            "atom_index_zero_based": atom_index_zero_based,
            "axis": axis,
            "finite_difference_method": method,
            "delta_count": len(deltas),
            "delta_min_ang": min(deltas),
            "delta_max_ang": max(deltas),
            "status": "available",
        }
        row.update(_range_payload(group_rows, "dh_mae_union_eV_per_Ang", "dh_mae_union_eV_per_Ang"))
        row.update(_range_payload(group_rows, "dh_rmse_union_eV_per_Ang", "dh_rmse_union_eV_per_Ang"))
        row.update(_range_payload(group_rows, "dh_relative_frobenius_ref", "dh_relative_frobenius_ref"))
        rows.append(row)

    unique_deltas = sorted({float(row.get("delta_ang")) for row in metric_rows if math.isfinite(_finite_or_nan(row.get("delta_ang")))})
    if not rows:
        status = "unavailable_single_delta" if len(unique_deltas) < 2 else "unavailable_no_matched_delta_groups"
        reason = (
            "At least two delta_ang values for the same source_model/base_sample_id/atom/axis/method are required."
            if len(unique_deltas) >= 2
            else "Fewer than two delta_ang values were found in derivative metric rows."
        )
    else:
        status = "available"
        reason = ""
    return {
        "status": status,
        "reason": reason,
        "groups_total": len(rows),
        "unique_delta_ang": unique_deltas,
        "rows": rows,
    }


def _reference_noise_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    noise_rows = [
        {key: value for key, value in row.items() if str(key).startswith("reference_noise")}
        for row in rows
        if any(str(key).startswith("reference_noise") for key in row)
    ]
    if not noise_rows:
        return {
            "status": "reference_noise_unavailable",
            "reason": "No repeated SIESTA reference/noise evidence was found in derivative metric manifests.",
            "rows": [],
        }
    return {
        "status": "available",
        "reason": "",
        "rows": noise_rows,
    }


def _mean(values: Any) -> float | None:
    clean = [_finite_or_nan(value) for value in values]
    clean = [value for value in clean if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else None


def _max(values: Any) -> float | None:
    clean = [_finite_or_nan(value) for value in values]
    clean = [value for value in clean if math.isfinite(value)]
    return max(clean) if clean else None


def _metric_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "sample",
        "atom_index_zero_based",
        "axis",
        "axis_index",
        "delta_ang",
        "finite_difference_method",
        "source_model",
        "reference_source",
        "derivative_units",
        "hamiltonian_units",
        "displacement_units",
        "matrix_metric_target_space",
        "comparison_status",
        "support_threshold",
        "dh_mae_ref_eV_per_Ang",
        "dh_rmse_ref_eV_per_Ang",
        "dh_mse_ref_eV2_per_Ang2",
        "dh_mae_pred_eV_per_Ang",
        "dh_rmse_pred_eV_per_Ang",
        "dh_mae_union_eV_per_Ang",
        "dh_rmse_union_eV_per_Ang",
        "dh_max_abs_error_union_eV_per_Ang",
        "dh_relative_frobenius_ref",
        "dh_relative_frobenius_union",
        "dh_relative_l1_union",
        "dh_cosine_similarity_union",
        "dh_support_precision",
        "dh_support_recall",
        "dh_support_f1",
        "dh_false_zero_rate",
        "dh_false_nonzero_rate",
        "dh_support_changed",
        "reference_plus_minus_support_changed",
        "predicted_plus_minus_support_changed",
        "dh_hermiticity_ref",
        "dh_hermiticity_pred",
        "dh_hermiticity_error_delta",
    ]
    keys = {str(key) for row in rows for key in row}
    return [key for key in preferred if key in keys] or sorted(keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--method", choices=["central", "forward", "backward"], default="central")
    parser.add_argument("--split", choices=["test", "validation", "train", "all"], default="all")
    parser.add_argument("--require-central", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--support-threshold", type=float, default=DERIVATIVE_SUPPORT_THRESHOLD)
    parser.add_argument("--max-stencils", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-model", choices=["graph2mat", "deeph"], default="graph2mat")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = evaluate_derivative_metrics(
        args.result_dir,
        method=args.method,
        split=args.split,
        require_central=args.require_central,
        overwrite=args.overwrite,
        diagnostic_only=args.diagnostic_only,
        support_threshold=args.support_threshold,
        max_stencils=args.max_stencils,
        output_dir=args.output_dir,
        source_model=args.source_model,
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=True, allow_nan=False))
    return 0 if not manifest["fatal_errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
