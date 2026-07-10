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
_SHARED_DIR = SCRIPT_DIR.parents[1] / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from run_inventory import collect_run_inventory  # noqa: E402

from hamiltonian_derivative_stencil import (  # noqa: E402
    DERIVATIVE_SUPPORT_THRESHOLD,
    DEEPH_PREDICTION_METHOD_AUTOGRAD,
    DEEPH_PREDICTION_METHOD_FINITE_DIFFERENCE,
    EXPECTED_DERIVATIVE_UNITS,
    GRAPH2MAT_PREDICTION_METHOD_AUTOGRAD,
    GRAPH2MAT_PREDICTION_METHOD_FINITE_DIFFERENCE,
    PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH,
    PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
    REFERENCE_DERIVATIVE_METHOD_SIESTA,
    VALID_DEEPH_PREDICTION_METHODS,
    VALID_GRAPH2MAT_PREDICTION_METHODS,
    DerivativeMatrixInput,
    DerivativeMetadata,
    DerivativeStencil,
    DerivativeStencilDiscovery,
    derivative_ref_abs_quantile_metrics,
    derivative_sparse_metrics,
    direct_predicted_derivative_pair,
    discover_derivative_stencils,
    find_direct_derivative_prediction,
    finite_difference_derivative_pair,
    load_direct_sparse_derivative,
    validate_derivative_geometry,
    validate_derivative_stencil,
    validation_errors,
)


SCHEMA_VERSION = "hamiltonian_derivative_metrics_v1"
REFERENCE_DEFINITION = "siesta_hamiltonian_finite_difference"
SUPPORT_THRESHOLDS_SWEEP = (1e-12, 1e-10, 1e-8, 1e-6)
GROUPING_PRESERVES = [
    "source_model",
    "reference_source",
    "dataset_size",
    "seed",
    "split",
    "delta_ang",
    "finite_difference_method",
    "support_threshold",
]
GROUP_MEAN_MEDIAN_FIELDS = [
    "dh_relative_frobenius_union_robust",
    "dh_mae_union_eV_per_Ang",
    "dh_rmse_union_eV_per_Ang",
    "dh_relative_l1_union_robust",
]
SCALAR_DELTA_STABILITY_DEFINITION = "scalar_error_metric_pairwise_delta_change_not_matrix_delta_stability"
DELTA_STABILITY_PAIRWISE_GROUP_KEYS = [
    "source_model",
    "reference_source",
    "dataset_size",
    "seed",
    "split",
    "base_sample_id",
    "atom_index_zero_based",
    "axis",
    "finite_difference_method",
    "support_threshold",
]
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
# Diagnostic-only: contextualises dh_relative_frobenius_ref by reporting whether the physical
# derivative signal ||H_plus - H_minus|| is above the model's absolute-H prediction error.
DERIVATIVE_SIGNAL_TO_NOISE_FIELDS = [
    "dh_signal_norm_fro",
    "dh_signal_over_abs_h_ref",
    "dh_abs_h_pred_error_norm_fro",
    "dh_abs_h_pred_rel_error_ref",
    "dh_signal_to_noise_ratio",
    "dh_signal_below_noise_floor",
    "dh_signal_to_noise_unavailable_reason",
]
DEEPH_EQUIVALENCE_FIELDS = [
    "claim_status",
    "deeph_adapter_equivalence_status",
    "deeph_equivalence_status",
    "deeph_equivalence_scope",
    "deeph_equivalence_reason",
    "deeph_equivalence_evidence_paths",
    "deeph_raw_global_equivalence_proven",
    "deeph_diagnostic_only",
    "deeph_diagnostic_reason",
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
DERIVATIVE_REF_ABS_QUANTILE_FIELDS = [
    "sample",
    "source_model",
    "reference_source",
    "base_sample_id",
    "atom_index_zero_based",
    "axis",
    "delta_ang",
    "finite_difference_method",
    "support_threshold",
    "quantile_domain",
    "quantile_bin",
    "n_entries",
    "n_ref_zero_entries",
    "n_pred_nonzero_ref_zero_entries",
    "abs_ref_min_eV_per_Ang",
    "abs_ref_max_eV_per_Ang",
    "abs_ref_mean_eV_per_Ang",
    "dh_error_mae_eV_per_Ang",
    "dh_error_rmse_eV_per_Ang",
    "dh_error_relative_l1_robust",
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


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


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
    graph2mat_prediction_method: str = GRAPH2MAT_PREDICTION_METHOD_FINITE_DIFFERENCE,
    deeph_prediction_method: str = DEEPH_PREDICTION_METHOD_FINITE_DIFFERENCE,
) -> dict[str, Any]:
    result_dir = Path(result_dir)
    output_dir = Path(output_dir) if output_dir is not None else result_dir / "derivative_metrics"
    source_model = str(source_model or "").strip().lower()

    graph2mat_prediction_method = str(
        graph2mat_prediction_method or GRAPH2MAT_PREDICTION_METHOD_FINITE_DIFFERENCE
    ).strip().lower()
    deeph_prediction_method = str(
        deeph_prediction_method or DEEPH_PREDICTION_METHOD_FINITE_DIFFERENCE
    ).strip().lower()
    if graph2mat_prediction_method not in VALID_GRAPH2MAT_PREDICTION_METHODS:
        raise DerivativeMetricEvaluationError(
            f"Unsupported graph2mat_prediction_method {graph2mat_prediction_method!r}. "
            f"Use one of: {', '.join(sorted(VALID_GRAPH2MAT_PREDICTION_METHODS))}."
        )
    if deeph_prediction_method not in VALID_DEEPH_PREDICTION_METHODS:
        raise DerivativeMetricEvaluationError(
            f"Unsupported deeph_prediction_method {deeph_prediction_method!r}. "
            f"Use one of: {', '.join(sorted(VALID_DEEPH_PREDICTION_METHODS))}."
        )
    graph2mat_direct_mode = (
        source_model == "graph2mat" and graph2mat_prediction_method == GRAPH2MAT_PREDICTION_METHOD_AUTOGRAD
    )
    deeph_direct_mode = source_model == "deeph" and deeph_prediction_method == DEEPH_PREDICTION_METHOD_AUTOGRAD
    direct_prediction_mode = graph2mat_direct_mode or deeph_direct_mode
    if graph2mat_prediction_method == GRAPH2MAT_PREDICTION_METHOD_AUTOGRAD and source_model != "graph2mat":
        raise DerivativeMetricEvaluationError(
            "graph2mat_prediction_method='autograd_vectorized' only applies to "
            f"source_model='graph2mat', got source_model={source_model!r}."
        )
    if deeph_prediction_method == DEEPH_PREDICTION_METHOD_AUTOGRAD and source_model != "deeph":
        raise DerivativeMetricEvaluationError(
            "deeph_prediction_method='autograd_vectorized' only applies to "
            f"source_model='deeph', got source_model={source_model!r}."
        )
    predicted_derivative_method = (
        PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT
        if graph2mat_direct_mode
        else PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH
        if deeph_direct_mode
        else f"finite_difference_{source_model}"
    )
    ensure_output_dir(output_dir, overwrite=overwrite)

    discoveries = discover_derivative_stencils(
        result_dir,
        method=source_model,
        split=split,
        finite_difference_method=method,
        require_central=require_central,
        require_ml_predictions=not direct_prediction_mode,
    )
    if max_stencils is not None:
        discoveries = discoveries[: max(0, int(max_stencils))]

    stencil_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    hermiticity_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    fatal_errors: list[dict[str, Any]] = []

    for discovery in discoveries:
        row, metrics, quantiles, sweep, hermiticity, geometry, warning_rows, error_rows = _evaluate_discovery(
            discovery,
            method=method,
            source_model=source_model,
            support_threshold=support_threshold,
            diagnostic_only=diagnostic_only,
            result_dir=result_dir,
            direct_prediction_mode=direct_prediction_mode,
            predicted_derivative_method=predicted_derivative_method,
            graph2mat_prediction_method=graph2mat_prediction_method,
            deeph_prediction_method=deeph_prediction_method,
        )
        stencil_rows.append(row)
        metric_rows.extend(metrics)
        quantile_rows.extend(quantiles)
        sweep_rows.extend(sweep)
        hermiticity_rows.extend(hermiticity)
        geometry_rows.append(geometry)
        warnings.extend(warning_rows)
        fatal_errors.extend(error_rows)

    stencils_total = len(discoveries)
    stencils_ok = len(metric_rows)
    stencils_failed = stencils_total - stencils_ok
    deeph_equivalence = _deeph_equivalence_summary(
        source_model=source_model,
        deeph_prediction_method=deeph_prediction_method,
        metric_rows=metric_rows,
    )
    scientific_status = _scientific_status(
        method=method,
        diagnostic_only=diagnostic_only,
        stencils_total=stencils_total,
        stencils_ok=stencils_ok,
        stencils_failed=stencils_failed,
        metric_rows=metric_rows,
        fatal_errors=fatal_errors,
        deeph_equivalence=deeph_equivalence,
    )
    delta_stability = _delta_stability_summary(metric_rows)
    delta_stability_convergence = _delta_stability_convergence_summary(delta_stability)
    reference_noise = _reference_noise_summary(metric_rows)
    summary = _summary(metric_rows, stencil_rows, hermiticity_rows)
    group_metrics = _derivative_group_metrics(metric_rows, split=split)
    onsite_offsite_metrics = {"available": False, "reason": "orbital_to_atom_mapping_unavailable"}
    warnings.append(
        {
            "kind": "derivative_onsite_offsite_metrics_unavailable",
            "message": "Onsite/offsite derivative metrics were not computed because orbital-to-atom mapping was unavailable.",
            "reason": "orbital_to_atom_mapping_unavailable",
        }
    )
    summary["delta_stability"] = {**delta_stability, **delta_stability_convergence}
    summary.update(delta_stability_convergence)
    summary["reference_noise"] = reference_noise
    outputs = {
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "stencil_status": str(output_dir / "stencil_status.csv"),
        "derivative_matrix_metrics": str(output_dir / "derivative_matrix_metrics.csv"),
        "derivative_ref_abs_quantile_metrics": str(output_dir / "derivative_ref_abs_quantile_metrics.csv"),
        "derivative_support_sweep": str(output_dir / "derivative_support_sweep.csv"),
        "derivative_hermiticity": str(output_dir / "derivative_hermiticity.csv"),
        "derivative_delta_stability": str(output_dir / "derivative_delta_stability.csv"),
        "derivative_delta_stability_json": str(output_dir / "derivative_delta_stability.json"),
        "derivative_geometry_validation": str(output_dir / "derivative_geometry_validation.csv"),
        "derivative_geometry_validation_json": str(output_dir / "derivative_geometry_validation.json"),
        "derivative_summary": str(output_dir / "derivative_summary.json"),
        "derivative_group_metrics": str(output_dir / "derivative_group_metrics.json"),
        "derivative_onsite_offsite_metrics": str(output_dir / "derivative_onsite_offsite_metrics.json"),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scientific_status": scientific_status,
        "paper_level": False,
        "finite_difference_method": method,
        "force_constants_used": False,
        "reference_definition": REFERENCE_DEFINITION,
        "reference_derivative_method": REFERENCE_DERIVATIVE_METHOD_SIESTA,
        "predicted_derivative_method": predicted_derivative_method,
        "graph2mat_prediction_method": graph2mat_prediction_method
        if source_model == "graph2mat"
        else None,
        "deeph_prediction_method": deeph_prediction_method if source_model == "deeph" else None,
        "predicted_delta_ang": None if direct_prediction_mode else "per_stencil_delta_ang",
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
        "delta_sensitivity_study_available": delta_stability_convergence["delta_sensitivity_study_available"],
        "delta_sensitivity_study_passed": delta_stability_convergence["delta_sensitivity_study_passed"],
        "delta_stability_converged": delta_stability_convergence["delta_stability_converged"],
        "delta_stability_convergence_status": delta_stability_convergence["delta_stability_convergence_status"],
        "reference_noise": reference_noise,
        "reference_noise_status": reference_noise["status"],
        "warnings": warnings,
        "fatal_errors": fatal_errors,
        "run_inventory": collect_run_inventory(),
        "outputs": outputs,
        **deeph_equivalence,
    }

    write_csv(output_dir / "stencil_status.csv", STATUS_FIELDS, stencil_rows)
    write_csv(output_dir / "derivative_matrix_metrics.csv", _metric_fieldnames(metric_rows), metric_rows)
    write_csv(output_dir / "derivative_ref_abs_quantile_metrics.csv", DERIVATIVE_REF_ABS_QUANTILE_FIELDS, quantile_rows)
    write_csv(output_dir / "derivative_support_sweep.csv", _metric_fieldnames(sweep_rows), sweep_rows)
    write_csv(output_dir / "derivative_hermiticity.csv", HERMITICITY_FIELDS, hermiticity_rows)
    delta_stability_json = {**delta_stability, **delta_stability_convergence}
    write_csv(output_dir / "derivative_delta_stability.csv", DELTA_STABILITY_FIELDS, delta_stability["rows"])
    write_json(output_dir / "derivative_delta_stability.json", delta_stability_json)
    write_csv(output_dir / "derivative_geometry_validation.csv", GEOMETRY_VALIDATION_FIELDS, geometry_rows)
    write_json(output_dir / "derivative_geometry_validation.json", _geometry_validation_summary(geometry_rows, include_rows=True))
    write_json(output_dir / "derivative_summary.json", summary)
    write_json(output_dir / "derivative_group_metrics.json", group_metrics)
    write_json(output_dir / "derivative_onsite_offsite_metrics.json", onsite_offsite_metrics)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _evaluate_discovery(
    discovery: DerivativeStencilDiscovery,
    *,
    method: str,
    source_model: str,
    support_threshold: float,
    diagnostic_only: bool,
    result_dir: Path | None = None,
    direct_prediction_mode: bool = False,
    predicted_derivative_method: str = PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
    graph2mat_prediction_method: str = GRAPH2MAT_PREDICTION_METHOD_FINITE_DIFFERENCE,
    deeph_prediction_method: str = DEEPH_PREDICTION_METHOD_FINITE_DIFFERENCE,
) -> tuple[
    list[Any] | dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    warnings: list[dict[str, Any]] = []
    fatal_errors: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    hermiticity_rows: list[dict[str, Any]] = []
    status_row = _stencil_status_row(discovery, status=discovery.status)
    geometry_issues = validate_derivative_geometry(discovery)
    geometry_row = _geometry_validation_row(discovery, geometry_issues)
    geometry_errors = validation_errors(geometry_issues)
    if discovery.stencil is None:
        fatal_errors.append(_discovery_error(discovery, "missing_stencil", "Discovery did not produce a complete stencil."))
        return status_row, metric_rows, quantile_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors
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
        return status_row, metric_rows, quantile_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors
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
        if diagnostic_only:
            stencil = replace(stencil, metadata=replace(stencil.metadata, claim_status="diagnostic_only"))
        validation = validate_derivative_stencil(
            stencil, require_predicted_operands=not direct_prediction_mode
        )
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
            return status_row, metric_rows, quantile_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors
        metadata = _metadata_for_status(stencil.metadata, diagnostic_only=diagnostic_only)
        direct_prediction_path: Path | None = None
        direct_metadata: dict[str, Any] = {}
        if direct_prediction_mode:
            candidate_base_ids = [
                str(metadata.base_sample_id or ""),
                f"{_group_base_id(discovery)}_base",
                _group_base_id(discovery),
            ]
            direct_prediction_path = find_direct_derivative_prediction(
                result_dir if result_dir is not None else Path("."),
                candidate_base_sample_ids=candidate_base_ids,
                atom_index_zero_based=int(metadata.atom_index_zero_based),
                axis_index=int(metadata.axis_index),
            )
            if direct_prediction_path is None:
                status_row = _stencil_status_row(discovery, status="failed")
                fatal_errors.append(
                    _discovery_error(
                        discovery,
                        "missing_direct_derivative_prediction",
                        "No direct dH_pred/dR matrix was found for this stencil; "
                        f"run the {source_model} autograd derivative prediction stage first.",
                        candidate_base_sample_ids=[c for c in candidate_base_ids if c],
                    )
                )
                return status_row, metric_rows, quantile_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors
            predicted_matrix, direct_metadata = load_direct_sparse_derivative(direct_prediction_path)
            pair = direct_predicted_derivative_pair(
                method=method,
                delta_ang=float(metadata.delta_ang),
                reference_plus=loaded.get("siesta_plus"),
                reference_minus=loaded.get("siesta_minus"),
                reference_base=loaded.get("siesta_base"),
                predicted_matrix=predicted_matrix,
                predicted_source=source_model,
                predicted_derivative_method=predicted_derivative_method,
                reference_hashes=_matrix_hashes(stencil, prefix="siesta"),
                predicted_matrix_metadata=direct_metadata,
                metadata=metadata,
            )
        else:
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
                "invalid_geometry": bool(geometry_errors),
                "geometry_validation_failed": bool(geometry_errors),
                "geometry_issue_codes": ";".join(issue.code for issue in geometry_errors),
                "dh_support_changed": bool(pair.diagnostics.get("plus_minus_support_changed")),
                "reference_plus_minus_support_changed": bool(pair.diagnostics.get("reference_plus_minus_support_changed")),
                "predicted_plus_minus_support_changed": bool(pair.diagnostics.get("predicted_plus_minus_support_changed")),
                "reference_derivative_method": REFERENCE_DERIVATIVE_METHOD_SIESTA,
                "predicted_derivative_method": predicted_derivative_method,
                "reference_delta_ang": float(metadata.delta_ang),
                "predicted_delta_ang": None if direct_prediction_mode else float(metadata.delta_ang),
                "graph2mat_prediction_method": (
                    graph2mat_prediction_method if source_model == "graph2mat" else None
                ),
                "deeph_prediction_method": deeph_prediction_method if source_model == "deeph" else None,
                "direct_prediction_path": str(direct_prediction_path) if direct_prediction_path else "",
            }
        )
        row.update(
            _deeph_direct_equivalence_fields(
                source_model=source_model,
                direct_prediction_mode=direct_prediction_mode,
                direct_metadata=direct_metadata,
            )
        )
        row.update(
            {
                key: pair.diagnostics[key]
                for key in DERIVATIVE_SIGNAL_TO_NOISE_FIELDS
                if key in pair.diagnostics
            }
        )
        metric_rows.append(row)
        quantile_rows.extend(
            derivative_ref_abs_quantile_metrics(
                pair.reference.matrix,
                pair.predicted.matrix,
                sample=metadata.sample_id,
                metadata=metadata,
                source_model=source_model,
                reference_source="siesta",
                support_threshold=support_threshold,
            )
        )
        if row["dh_union_nnz"] == 0:
            warnings.append(
                _discovery_error(
                    discovery,
                    "derivative_ref_abs_quantile_metrics_empty_union_support",
                    "No derivative ref-abs quantile rows written because union support is empty.",
                )
            )
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
    return status_row, metric_rows, quantile_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors


def _group_base_id(discovery: DerivativeStencilDiscovery) -> str:
    """First component of the discovery group key: the displaced samples' base id."""

    if discovery.group_key:
        return str(discovery.group_key[0] or "")
    return ""


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


def _deeph_direct_equivalence_fields(
    *,
    source_model: str,
    direct_prediction_mode: bool,
    direct_metadata: dict[str, Any],
) -> dict[str, Any]:
    if source_model != "deeph" or not direct_prediction_mode:
        return {}
    proven = truthy(direct_metadata.get("deeph_raw_global_equivalence_proven"))
    diagnostic_only = truthy(direct_metadata.get("deeph_diagnostic_only")) if "deeph_diagnostic_only" in direct_metadata else not proven
    evidence_paths = direct_metadata.get("deeph_equivalence_evidence_paths") or []
    if isinstance(evidence_paths, str):
        evidence_text = evidence_paths
    else:
        evidence_text = ";".join(str(path) for path in evidence_paths if str(path))
    return {
        "claim_status": direct_metadata.get("claim_status") or ("raw_global_equivalence_proven" if proven else "diagnostic_only"),
        "deeph_adapter_equivalence_status": direct_metadata.get("deeph_adapter_equivalence_status") or "",
        "deeph_equivalence_status": direct_metadata.get("deeph_equivalence_status") or ("proven" if proven else "unproven"),
        "deeph_equivalence_scope": direct_metadata.get("deeph_equivalence_scope") or "",
        "deeph_equivalence_reason": direct_metadata.get("deeph_equivalence_reason") or "",
        "deeph_equivalence_evidence_paths": evidence_text,
        "deeph_raw_global_equivalence_proven": proven,
        "deeph_diagnostic_only": diagnostic_only,
        "deeph_diagnostic_reason": direct_metadata.get("deeph_diagnostic_reason") or ("" if proven else "deeph_raw_global_equivalence_not_proven"),
    }


def _deeph_equivalence_summary(
    *,
    source_model: str,
    deeph_prediction_method: str,
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if source_model != "deeph" or deeph_prediction_method != DEEPH_PREDICTION_METHOD_AUTOGRAD:
        return {}
    total = len(metric_rows)
    proven_count = sum(1 for row in metric_rows if truthy(row.get("deeph_raw_global_equivalence_proven")))
    all_proven = total > 0 and proven_count == total
    diagnostic_only = (not all_proven) or any(truthy(row.get("deeph_diagnostic_only")) for row in metric_rows)
    evidence_paths: list[str] = []
    for row in metric_rows:
        value = row.get("deeph_equivalence_evidence_paths")
        items = value if isinstance(value, list) else str(value or "").split(";")
        for item in items:
            text = str(item).strip()
            if text and text not in evidence_paths:
                evidence_paths.append(text)
    return {
        "deeph_raw_global_equivalence_proven_count": proven_count,
        "deeph_raw_global_equivalence_total": total,
        "deeph_all_raw_global_equivalence_proven": all_proven,
        "deeph_diagnostic_only": diagnostic_only,
        "deeph_equivalence_statuses": _unique_row_values(metric_rows, "deeph_equivalence_status"),
        "deeph_adapter_equivalence_statuses": _unique_row_values(metric_rows, "deeph_adapter_equivalence_status"),
        "deeph_equivalence_scopes": _unique_row_values(metric_rows, "deeph_equivalence_scope"),
        "deeph_equivalence_evidence_paths": evidence_paths,
    }


def _unique_row_values(rows: list[dict[str, Any]], field: str) -> list[str]:
    values: list[str] = []
    for row in rows:
        text = str(row.get(field) or "").strip()
        if text and text not in values:
            values.append(text)
    return values


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
    deeph_equivalence: dict[str, Any] | None = None,
) -> str:
    if diagnostic_only or method != "central" or not metric_rows:
        return "diagnostic_only"
    if deeph_equivalence and truthy(deeph_equivalence.get("deeph_diagnostic_only")):
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


def _derivative_group_metrics(metric_rows: list[dict[str, Any]], *, split: str | None = None) -> dict[str, Any]:
    rows = [{**row, **({"split": split} if split is not None and "split" not in row else {})} for row in metric_rows]
    return {
        "schema": "hamiltonian_derivative_group_metrics_v1",
        "grouping_preserves": GROUPING_PRESERVES,
        "by_atom": _aggregate_derivative_groups(rows, ["atom_index_zero_based"]),
        "by_axis": _aggregate_derivative_groups(rows, ["axis"]),
        "by_atom_axis": _aggregate_derivative_groups(rows, ["atom_index_zero_based", "axis"]),
    }


def _aggregate_derivative_groups(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = [*GROUPING_PRESERVES, *group_keys]
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    return [
        _aggregate_derivative_group(keys, key, group_rows)
        for key, group_rows in sorted(grouped.items(), key=lambda item: repr(item[0]))
    ]


def _aggregate_derivative_group(keys: list[str], key: tuple[Any, ...], rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {name: value for name, value in zip(keys, key, strict=False)}
    payload["n_stencils"] = len(rows)
    for field in GROUP_MEAN_MEDIAN_FIELDS:
        values = _finite_values(rows, field)
        payload[f"{field}_mean"] = sum(values) / len(values) if values else None
        payload[f"{field}_median"] = _median(values)
    fro_pairs = [
        (_finite_or_nan(row.get("dh_norm_error_union_fro")), _finite_or_nan(row.get("dh_norm_ref_union_fro")))
        for row in rows
        if "dh_norm_error_union_fro" in row and "dh_norm_ref_union_fro" in row
    ]
    fro_pairs = [(err, ref) for err, ref in fro_pairs if math.isfinite(err) and math.isfinite(ref)]
    if fro_pairs:
        payload["dh_relative_frobenius_union_robust_pooled"] = math.sqrt(sum(err**2 for err, _ in fro_pairs)) / (
            math.sqrt(sum(ref**2 for _, ref in fro_pairs)) + 1e-30
        )
    l1_pairs = [
        (_finite_or_nan(row.get("dh_norm_error_l1_union")), _finite_or_nan(row.get("dh_norm_ref_l1_union")))
        for row in rows
        if "dh_norm_error_l1_union" in row and "dh_norm_ref_l1_union" in row
    ]
    l1_pairs = [(err, ref) for err, ref in l1_pairs if math.isfinite(err) and math.isfinite(ref)]
    if l1_pairs:
        payload["dh_relative_l1_union_robust_pooled"] = sum(err for err, _ in l1_pairs) / (
            sum(ref for _, ref in l1_pairs) + 1e-30
        )
    return payload


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


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

    pairwise_metric_rows = _delta_stability_pairwise_metric_rows(metric_rows)
    unique_deltas = sorted({float(row.get("delta_ang")) for row in metric_rows if math.isfinite(_finite_or_nan(row.get("delta_ang")))})
    if not rows:
        status = "single_delta_only" if len(unique_deltas) < 2 else "unavailable_no_matched_delta_groups"
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
        "pairwise_metric_rows": pairwise_metric_rows,
    }


def _delta_stability_pairwise_metric_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in metric_rows:
        delta = _finite_or_nan(row.get("delta_ang"))
        if not math.isfinite(delta):
            continue
        groups.setdefault(tuple(row.get(key) for key in DELTA_STABILITY_PAIRWISE_GROUP_KEYS), []).append(row)

    pairwise_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        by_delta: dict[float, dict[str, Any]] = {}
        for row in group_rows:
            delta = _finite_or_nan(row.get("delta_ang"))
            if math.isfinite(delta):
                by_delta.setdefault(float(delta), row)
        deltas = sorted(by_delta)
        for delta_1, delta_2 in zip(deltas, deltas[1:], strict=False):
            first = by_delta[delta_1]
            second = by_delta[delta_2]
            for metric_name in _delta_stability_metric_names(first, second):
                value_1 = _finite_or_nan(first.get(metric_name))
                value_2 = _finite_or_nan(second.get(metric_name))
                if not math.isfinite(value_1) or not math.isfinite(value_2):
                    continue
                abs_change = abs(value_2 - value_1)
                row = {
                    name: value
                    for name, value in zip(DELTA_STABILITY_PAIRWISE_GROUP_KEYS, key, strict=False)
                    if value is not None
                }
                row.update(
                    {
                        "delta_1_ang": delta_1,
                        "delta_2_ang": delta_2,
                        "metric_name": metric_name,
                        "value_delta_1": value_1,
                        "value_delta_2": value_2,
                        "abs_change": abs_change,
                        "relative_change": abs_change / (abs(value_2) + 1e-30),
                        "stability_definition": SCALAR_DELTA_STABILITY_DEFINITION,
                    }
                )
                pairwise_rows.append(row)
    return pairwise_rows


def _delta_stability_metric_names(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    fro = (
        "dh_relative_frobenius_union_robust"
        if "dh_relative_frobenius_union_robust" in first or "dh_relative_frobenius_union_robust" in second
        else "dh_relative_frobenius_ref"
    )
    metrics = [fro, "dh_mae_union_eV_per_Ang", "dh_rmse_union_eV_per_Ang"]
    if "dh_relative_l1_union_robust" in first or "dh_relative_l1_union_robust" in second:
        metrics.append("dh_relative_l1_union_robust")
    return metrics


def _delta_stability_convergence_summary(
    delta_stability: dict[str, Any],
    *,
    convergence_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    available = str(delta_stability.get("status") or "").strip().lower() == "available"
    thresholds_present = bool(convergence_thresholds)
    if not thresholds_present:
        return {
            "delta_sensitivity_study_available": available,
            "delta_sensitivity_study_passed": available,
            "delta_stability_converged": None,
            "delta_stability_convergence_status": "not_evaluated_without_thresholds",
        }
    converged = delta_stability.get("converged")
    if converged is None:
        converged = str(delta_stability.get("convergence_status") or "").strip().lower() == "converged"
    convergence_status = str(delta_stability.get("convergence_status") or "").strip() or (
        "converged" if bool(converged) else "not_converged"
    )
    return {
        "delta_sensitivity_study_available": available,
        "delta_sensitivity_study_passed": available,
        "delta_stability_converged": bool(converged) if converged is not None else None,
        "delta_stability_convergence_status": convergence_status,
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
        "invalid_geometry",
        "geometry_validation_failed",
        "geometry_issue_codes",
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
        *DERIVATIVE_SIGNAL_TO_NOISE_FIELDS,
        "dh_support_precision",
        "dh_support_recall",
        "dh_support_f1",
        "dh_false_zero_rate",
        "dh_false_nonzero_rate",
        "dh_support_changed",
        "reference_plus_minus_support_changed",
        "predicted_plus_minus_support_changed",
        *DEEPH_EQUIVALENCE_FIELDS,
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
    parser.add_argument(
        "--graph2mat-prediction-method",
        choices=sorted(VALID_GRAPH2MAT_PREDICTION_METHODS),
        default=GRAPH2MAT_PREDICTION_METHOD_FINITE_DIFFERENCE,
        help=(
            "How the predicted Graph2Mat derivative is obtained: 'finite_difference' "
            "(legacy, from displaced ML_prediction.HSX pairs) or 'autograd_vectorized' "
            "(direct dH_pred/dR matrices from run_graph2mat_autograd_derivative_predictions.py)."
        ),
    )
    parser.add_argument(
        "--deeph-prediction-method",
        choices=sorted(VALID_DEEPH_PREDICTION_METHODS),
        default=DEEPH_PREDICTION_METHOD_FINITE_DIFFERENCE,
        help=(
            "How the predicted DeepH derivative is obtained: 'finite_difference' "
            "(legacy, from displaced ML_prediction.HSX pairs) or 'autograd_vectorized' "
            "(direct dH_pred/dR matrices from hamiltonians_grad_pred.h5)."
        ),
    )
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
        graph2mat_prediction_method=args.graph2mat_prediction_method,
        deeph_prediction_method=args.deeph_prediction_method,
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=True, allow_nan=False))
    return 0 if not manifest["fatal_errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
