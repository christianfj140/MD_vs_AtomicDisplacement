#!/usr/bin/env python3
"""Common metric aggregation for the Graph2Mat-vs-DeepH benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deeph_prediction_adapter import (
    EQUIVALENCE_INVALID_MISSING_REFERENCE,
    EQUIVALENCE_PROVEN_RAW_GLOBAL,
    EQUIVALENCE_STATUS_FAILED,
    EQUIVALENCE_STATUS_PROVEN,
    PROVEN_ADAPTER_EQUIVALENCE_STATUSES,
    equivalence_scope_from_adapter_status,
    equivalence_status_from_adapter_status,
)


SCHEMA = "graph2mat_deeph_common_metrics_v1"
FORBIDDEN_REFERENCE_NAMES = {"ML_prediction.HSX"}
STATUS_VALUES = {
    "valid_joint_one_pass_dataset",
    "valid_reused_joint_dataset",
    "valid_repaired_dataset_with_warning",
    "invalid_missing_artifacts",
    "invalid_incompatible_splits",
    "invalid_incompatible_basis_or_pseudos",
    "invalid_prediction_format",
    "diagnostic_only",
}
PRIMARY_METRIC = "h_mae_eV_mean"

H_MAE_METRIC_GROUP = {
    "id": "h_mae",
    "title": "Hamiltonian MAE",
    "y_title": "MAE eV",
    "metrics": [{"key": "h_mae_eV_mean", "label": "H MAE", "unit": "eV", "direction": "lower_is_better"}],
}
H_RMSE_METRIC_GROUP = {
    "id": "h_rmse",
    "title": "Hamiltonian RMSE",
    "y_title": "RMSE eV",
    "metrics": [{"key": "h_rmse_eV_mean", "label": "H RMSE", "unit": "eV", "direction": "lower_is_better"}],
}
H_MSE_METRIC_GROUP = {
    "id": "h_mse",
    "title": "Hamiltonian MSE",
    "y_title": "MSE eV^2",
    "metrics": [{"key": "h_mse_eV2_mean", "label": "H MSE", "unit": "eV^2", "direction": "lower_is_better"}],
}
R2_METRIC_GROUP = {
    "id": "r2",
    "title": "Sparse support R2",
    "y_title": "R2",
    "metrics": [{"key": "r2_mean", "label": "R2", "unit": "", "direction": "higher_is_better"}],
}
FROBENIUS_METRIC_GROUP = {
    "id": "frobenius",
    "title": "Relative Frobenius",
    "y_title": "relative error",
    "metrics": [
        {
            "key": "relative_frobenius_mean",
            "label": "Relative Frobenius",
            "unit": "",
            "direction": "lower_is_better",
        },
    ],
}
HERMITICITY_METRIC_GROUP = {
    "id": "hermiticity",
    "title": "Predicted Hamiltonian hermiticity",
    "y_title": "Hermiticity residual",
    "metrics": [
        {
            "key": "hermiticity_pred_mean",
            "label": "Hermiticity residual",
            "unit": "",
            "direction": "lower_is_better",
        },
    ],
}
SPECTRAL_GLOBAL_METRIC_GROUP = {
    "id": "spectral_global",
    "title": "Global spectral RMSE",
    "y_title": "RMSE eV",
    "metrics": [
        {"key": "global_rmse_eV_mean", "label": "Global RMSE", "unit": "eV", "direction": "lower_is_better"},
    ],
}
SPECTRAL_LOW_ENERGY_METRIC_GROUP = {
    "id": "spectral_low_energy",
    "title": "Low-energy spectral RMSE",
    "y_title": "RMSE eV",
    "metrics": [
        {"key": "low_energy_rmse_eV_mean", "label": "Low-energy RMSE", "unit": "eV", "direction": "lower_is_better"},
    ],
}
SPECTRAL_FERMI_METRIC_GROUP = {
    "id": "spectral_fermi",
    "title": "Fermi-window spectral RMSE",
    "y_title": "RMSE eV",
    "metrics": [
        {"key": "fermi_window_rmse_eV_mean", "label": "Fermi-window RMSE", "unit": "eV", "direction": "lower_is_better"},
    ],
}
SPECTRAL_FRONTIER_METRIC_GROUP = {
    "id": "spectral_frontier",
    "title": "Frontier-window spectral RMSE",
    "y_title": "RMSE eV",
    "metrics": [
        {"key": "frontier_window_rmse_eV_mean", "label": "Frontier RMSE", "unit": "eV", "direction": "lower_is_better"},
    ],
}
DOS_MAE_METRIC_GROUP = {
    "id": "dos_mae",
    "title": "DOS Fermi-window MAE",
    "y_title": "DOS MAE",
    "metrics": [
        {
            "key": "dos_mae_500_fermi_window_mean",
            "label": "DOS MAE 500 Fermi window",
            "unit": "",
            "direction": "lower_is_better",
        },
    ],
}
DOS_WASSERSTEIN_METRIC_GROUP = {
    "id": "dos_wasserstein",
    "title": "DOS Wasserstein distance",
    "y_title": "Wasserstein eV",
    "metrics": [
        {
            "key": "dos_wasserstein_eV_mean",
            "label": "DOS Wasserstein",
            "unit": "eV",
            "direction": "lower_is_better",
        },
    ],
}
COMMON_METRIC_GROUPS = [
    H_MAE_METRIC_GROUP,
    H_RMSE_METRIC_GROUP,
    H_MSE_METRIC_GROUP,
    R2_METRIC_GROUP,
    FROBENIUS_METRIC_GROUP,
    HERMITICITY_METRIC_GROUP,
    SPECTRAL_GLOBAL_METRIC_GROUP,
    SPECTRAL_LOW_ENERGY_METRIC_GROUP,
    SPECTRAL_FERMI_METRIC_GROUP,
    SPECTRAL_FRONTIER_METRIC_GROUP,
    DOS_MAE_METRIC_GROUP,
    DOS_WASSERSTEIN_METRIC_GROUP,
]


@dataclass(frozen=True)
class StagedGraph2MatMetrics:
    result_dir: Path
    sample_ids: list[str]


@dataclass(frozen=True)
class StagedDeepHMetrics:
    processed_dir: Path
    predictions_dir: Path
    sample_ids: list[str]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: json_safe(value) for key, value in row.items()} for row in rows])


def finite_or_none(value: Any) -> float | None:
    result = number(value)
    return result if math.isfinite(result) else None


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else math.nan


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)


def split_rows(frozen_split_manifest: dict[str, Any], split: str = "test") -> list[dict[str, Any]]:
    return [dict(row) for row in frozen_split_manifest.get("rows") or [] if row.get("split") == split]


def row_sample_id(row: dict[str, Any]) -> str:
    value = str(row.get("sample_id") or row.get("graph2mat_sample_id") or row.get("deeph_sample_id") or "").strip()
    if value:
        return value
    sample_dir = str(row.get("sample_dir") or "").strip()
    if sample_dir:
        return Path(sample_dir).name
    raise RuntimeError(f"Frozen split row is missing a stable sample id: {row}")


def forbidden_reference_paths_from_split(frozen_split_manifest: dict[str, Any]) -> list[str]:
    forbidden: list[str] = []
    for row in frozen_split_manifest.get("rows") or []:
        for key, value in row.items():
            if not isinstance(value, str) or not value:
                continue
            if Path(value).name in FORBIDDEN_REFERENCE_NAMES and (
                "reference" in key.lower() or "hamiltonian" in key.lower()
            ):
                forbidden.append(value)
    return sorted(set(forbidden))


def validate_no_forbidden_references(frozen_split_manifest: dict[str, Any]) -> None:
    forbidden = forbidden_reference_paths_from_split(frozen_split_manifest)
    if forbidden:
        raise RuntimeError("ML_prediction.HSX cannot be selected as SIESTA reference: " + ", ".join(forbidden))


def stage_graph2mat_metric_result(
    *,
    frozen_split_manifest: dict[str, Any],
    prediction_structs_dir: Path,
    output_dir: Path,
    dataset_root: Path | None = None,
) -> StagedGraph2MatMetrics:
    validate_no_forbidden_references(frozen_split_manifest)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    sample_ids: list[str] = []
    for row in split_rows(frozen_split_manifest, "test"):
        sample_id = row_sample_id(row)
        sample_dir = Path(str(row.get("sample_dir") or ""))
        if not sample_dir.exists():
            raise RuntimeError(f"Frozen split sample_dir does not exist: {sample_dir}")
        prediction_dir = prediction_structs_dir / sample_id
        prediction = prediction_dir / "ML_prediction.HSX"
        if not prediction.exists():
            raise RuntimeError(f"Missing Graph2Mat prediction for common metrics: {prediction}")
        _link_or_copy(sample_dir / "RUN.fdf", output_dir / "structures" / sample_id / "RUN.fdf")
        for artifact in sorted(path for path in sample_dir.iterdir() if path.is_file()):
            if artifact.name in FORBIDDEN_REFERENCE_NAMES:
                continue
            _link_or_copy(artifact, output_dir / "siesta_hamiltonians" / sample_id / artifact.name)
        _link_or_copy(prediction, output_dir / "predicted_hamiltonians" / sample_id / prediction.name)
        sample_ids.append(sample_id)

    if dataset_root is not None:
        split_root = dataset_root / "splits"
        if split_root.exists():
            _link_or_copy(split_root, output_dir / "splits")
        for basis_dir in (dataset_root / "basis", dataset_root / "materials" / "basis"):
            if basis_dir.exists():
                _link_or_copy(basis_dir, output_dir / "basis")
                break
    return StagedGraph2MatMetrics(result_dir=output_dir, sample_ids=sample_ids)


def stage_deeph_metric_inputs(
    *,
    raw_mirror: dict[str, Any],
    processed_dir: Path,
    inference_dir: Path,
    output_dir: Path,
) -> StagedDeepHMetrics:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staged_processed = output_dir / "processed"
    staged_predictions = output_dir / "predictions"
    sample_ids: list[str] = []
    for row in raw_mirror.get("rows") or []:
        if row.get("split") != "test":
            continue
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise RuntimeError(f"DeepH raw mirror row is missing sample_id: {row}")
        raw_name = Path(str(row.get("raw_dir") or "")).name
        source_processed = processed_dir / raw_name
        source_prediction = inference_dir / raw_name
        if not source_processed.exists():
            raise RuntimeError(f"Missing DeepH processed sample for common metrics: {source_processed}")
        if not source_prediction.exists():
            raise RuntimeError(f"Missing DeepH prediction sample for common metrics: {source_prediction}")
        _link_or_copy(source_processed, staged_processed / sample_id)
        _link_or_copy(source_prediction, staged_predictions / sample_id)
        sample_ids.append(sample_id)
    return StagedDeepHMetrics(staged_processed, staged_predictions, sample_ids)


def weighted_sample_rows(metrics_root: Path) -> list[dict[str, str]]:
    return [row for row in read_csv_rows(metrics_root / "kpoint_matrix_metrics.csv") if row.get("row_type") == "weighted_sample"]


def sample_ids_from_metrics(metrics_root: Path) -> set[str]:
    ids = {str(row.get("sample") or "").strip() for row in weighted_sample_rows(metrics_root)}
    if ids:
        return {sample for sample in ids if sample}
    return {str(row.get("sample") or "").strip() for row in read_csv_rows(metrics_root / "sample_status.csv") if row.get("sample")}


def method_has_diagnostic_only(metrics_root: Path, manifest: dict[str, Any]) -> bool:
    adapter_manifest_path = metrics_root.parent / "adapter_manifest.json"
    adapter_manifest = read_json(adapter_manifest_path)
    if int(adapter_manifest.get("diagnostic_only_count") or 0) > 0:
        return True
    for name in ("kpoint_matrix_metrics.csv", "kpoint_spectral_metrics.csv", "kpoint_dos_metrics.csv"):
        for row in read_csv_rows(metrics_root / name):
            value = str(row.get("deeph_diagnostic_only") or "").strip().lower()
            if value in {"true", "1", "yes"}:
                return True
    return bool(manifest.get("diagnostic_only"))


def adapter_equivalence_summary(metrics_root: Path) -> dict[str, Any]:
    adapter_manifest_path = metrics_root.parent / "adapter_manifest.json"
    adapter_manifest = read_json(adapter_manifest_path)
    if not adapter_manifest:
        return {
            "adapter_manifest_path": str(adapter_manifest_path),
            "adapter_equivalence_status": EQUIVALENCE_INVALID_MISSING_REFERENCE,
            "adapter_equivalence_statuses": [EQUIVALENCE_INVALID_MISSING_REFERENCE],
            "equivalence_status": EQUIVALENCE_STATUS_FAILED,
            "equivalence_statuses": [EQUIVALENCE_STATUS_FAILED],
            "equivalence_scope": "unknown",
            "equivalence_scopes": ["unknown"],
            "equivalence_evidence_paths": [],
            "equivalence_gate": {
                "robust_claim_allowed": False,
                "diagnostic_only": True,
                "diagnostic_only_reason": "DeepH adapter manifest is missing.",
            },
            "raw_global_equivalence_proven": False,
            "robust_matrix_metrics_allowed": False,
        }
    statuses = [
        str(status)
        for status in adapter_manifest.get("adapter_equivalence_statuses") or []
        if str(status).strip()
    ]
    if not statuses:
        statuses = sorted(
            {
                str(sample.get("adapter_equivalence_status"))
                for sample in adapter_manifest.get("samples") or []
                if str(sample.get("adapter_equivalence_status") or "").strip()
            }
        )
    if not statuses:
        statuses = [EQUIVALENCE_INVALID_MISSING_REFERENCE]
    equivalence_statuses = [
        str(status)
        for status in adapter_manifest.get("equivalence_statuses") or []
        if str(status).strip()
    ]
    if not equivalence_statuses:
        equivalence_statuses = sorted(
            {
                str(sample.get("equivalence_status"))
                for sample in adapter_manifest.get("samples") or []
                if str(sample.get("equivalence_status") or "").strip()
            }
        )
    if not equivalence_statuses:
        equivalence_statuses = sorted({equivalence_status_from_adapter_status(status) for status in statuses})
    equivalence_scopes = [
        str(scope)
        for scope in adapter_manifest.get("equivalence_scopes") or []
        if str(scope).strip()
    ]
    if not equivalence_scopes:
        equivalence_scopes = sorted(
            {
                str(sample.get("equivalence_scope"))
                for sample in adapter_manifest.get("samples") or []
                if str(sample.get("equivalence_scope") or "").strip()
            }
        )
    if not equivalence_scopes:
        equivalence_scopes = sorted({equivalence_scope_from_adapter_status(status) for status in statuses})
    raw_global_equivalence_proven = (
        bool(adapter_manifest.get("robust_matrix_metrics_allowed"))
        and all(status in PROVEN_ADAPTER_EQUIVALENCE_STATUSES for status in statuses)
        and all(status == EQUIVALENCE_STATUS_PROVEN for status in equivalence_statuses)
    )
    primary_status = (
        EQUIVALENCE_PROVEN_RAW_GLOBAL
        if raw_global_equivalence_proven
        else next((status for status in statuses if status not in PROVEN_ADAPTER_EQUIVALENCE_STATUSES), statuses[0])
    )
    primary_equivalence_status = (
        EQUIVALENCE_STATUS_PROVEN
        if raw_global_equivalence_proven
        else next((status for status in equivalence_statuses if status != EQUIVALENCE_STATUS_PROVEN), equivalence_statuses[0])
    )
    primary_equivalence_scope = equivalence_scopes[0] if len(equivalence_scopes) == 1 else ",".join(equivalence_scopes)
    equivalence_gate = adapter_manifest.get("equivalence_gate") if isinstance(adapter_manifest.get("equivalence_gate"), dict) else {}
    return {
        "adapter_manifest_path": str(adapter_manifest_path),
        "adapter_equivalence_status": primary_status,
        "adapter_equivalence_statuses": statuses,
        "equivalence_status": primary_equivalence_status,
        "equivalence_statuses": equivalence_statuses,
        "equivalence_scope": primary_equivalence_scope,
        "equivalence_scopes": equivalence_scopes,
        "equivalence_evidence_paths": adapter_manifest.get("equivalence_evidence_paths") or [],
        "equivalence_gate": equivalence_gate,
        "raw_global_equivalence_proven": raw_global_equivalence_proven,
        "robust_matrix_metrics_allowed": raw_global_equivalence_proven,
    }


def summarize_method(method: str, metrics_root: Path) -> dict[str, Any]:
    matrix_rows = weighted_sample_rows(metrics_root)
    sparse_rows = read_csv_rows(metrics_root / "sparse_metrics.csv")
    spectral_rows = read_csv_rows(metrics_root / "kpoint_spectral_metrics.csv") or read_csv_rows(metrics_root / "spectral_metrics.csv")
    dos_rows = read_csv_rows(metrics_root / "kpoint_dos_metrics.csv") or read_csv_rows(metrics_root / "dos_metrics.csv")
    manifest = read_json(metrics_root / "manifest.json")
    fatal_errors = manifest.get("fatal_errors") or []
    warnings = list(manifest.get("warnings") or [])
    if manifest and manifest.get("uses_reference_overlap_k") is not True:
        warnings.append({"severity": "severe", "kind": "missing_s_ref", "message": "S_ref(k) was not recorded as overlap source."})
    if manifest and manifest.get("kpoint_metrics_enabled") is False:
        warnings.append({"severity": "severe", "kind": "unsupported_kgrid", "message": "k-point metrics were not enabled."})
    status = "ok"
    if not matrix_rows and not sparse_rows:
        status = "missing_metrics"
    if fatal_errors:
        status = "fatal_errors"
    adapter_summary = adapter_equivalence_summary(metrics_root) if method == "deeph" else {}
    diagnostic_only = method_has_diagnostic_only(metrics_root, manifest)
    if method == "deeph" and not adapter_summary.get("raw_global_equivalence_proven"):
        diagnostic_only = True
        warnings.append(
            {
                "severity": "severe",
                "kind": "deeph_adapter_equivalence_not_proven",
                "adapter_equivalence_status": adapter_summary.get("adapter_equivalence_status"),
                "equivalence_status": adapter_summary.get("equivalence_status"),
                "equivalence_scope": adapter_summary.get("equivalence_scope"),
                "message": "DeepH prediction equivalence to Graph2Mat raw/global HSX is not proven.",
            }
        )
    row: dict[str, Any] = {
        "method": method,
        "metrics_root": str(metrics_root),
        "method_status": status,
        "diagnostic_only": diagnostic_only,
        "samples_compared": manifest.get("samples_compared"),
        "samples_failed": manifest.get("samples_failed"),
        "kpoint_metrics_enabled": manifest.get("kpoint_metrics_enabled"),
        "uses_reference_overlap_k": manifest.get("uses_reference_overlap_k"),
        "warning_count": len(warnings),
        "fatal_error_count": len(fatal_errors),
        **adapter_summary,
    }
    row["h_mae_eV_mean"] = mean([number(item.get("h_mae_eV")) for item in matrix_rows] or [number(item.get("mae_union_eV")) for item in sparse_rows])
    row["h_rmse_eV_mean"] = mean([number(item.get("h_rmse_eV")) for item in matrix_rows] or [number(item.get("rmse_union_eV")) for item in sparse_rows])
    row["h_mse_eV2_mean"] = mean([number(item.get("h_mse_eV2")) for item in matrix_rows] or [number(item.get("mse_union_eV2")) for item in sparse_rows])
    row["r2_mean"] = mean([number(item.get("r2_union")) for item in sparse_rows])
    row["relative_frobenius_mean"] = mean(
        [number(item.get("relative_frobenius")) for item in matrix_rows]
        or [number(item.get("relative_frobenius_union")) for item in sparse_rows]
    )
    row["support_precision_mean"] = mean([number(item.get("support_precision")) for item in sparse_rows])
    row["support_recall_mean"] = mean([number(item.get("support_recall")) for item in sparse_rows])
    row["support_f1_mean"] = mean([number(item.get("support_f1")) for item in sparse_rows])
    row["hermiticity_pred_mean"] = mean([number(item.get("hermiticity_pred")) for item in matrix_rows])
    row["global_rmse_eV_mean"] = mean([number(item.get("global_rmse_eV")) for item in spectral_rows])
    row["low_energy_rmse_eV_mean"] = mean([number(item.get("low_energy_rmse_eV")) for item in spectral_rows])
    row["fermi_window_rmse_eV_mean"] = mean([number(item.get("fermi_window_rmse_eV")) for item in spectral_rows])
    row["frontier_window_rmse_eV_mean"] = mean([number(item.get("frontier_window_rmse_eV")) for item in spectral_rows])
    row["dos_mae_500_fermi_window_mean"] = mean([number(item.get("dos_mae_500_fermi_window")) for item in dos_rows])
    row["dos_wasserstein_eV_mean"] = mean([number(item.get("dos_wasserstein_eV")) for item in dos_rows])
    return row


def sample_metric_rows(method: str, metrics_root: Path) -> list[dict[str, Any]]:
    rows = []
    for row in weighted_sample_rows(metrics_root):
        rows.append({"method": method, **row})
    return rows


def dataset_status(dataset_manifest: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    warnings = list(dataset_manifest.get("warnings") or [])
    if dataset_manifest and not dataset_manifest.get("benchmark_ready", dataset_manifest.get("valid", False)):
        return "invalid_missing_artifacts", warnings
    mode = str(dataset_manifest.get("generation_mode") or dataset_manifest.get("mode") or "").strip()
    if mode == "clean_one_pass":
        return "valid_joint_one_pass_dataset", warnings
    if mode == "reused_validated":
        return "valid_reused_joint_dataset", warnings
    if mode == "repaired_explicit":
        warnings.append({"severity": "severe", "kind": "repaired_dataset", "message": "Dataset was explicitly repaired."})
        return "valid_repaired_dataset_with_warning", warnings
    return "valid_reused_joint_dataset" if dataset_manifest else "diagnostic_only", warnings


def build_recommendation(summary_rows: list[dict[str, Any]], status: str, warnings: list[dict[str, Any]]) -> dict[str, Any]:
    severe = [warning for warning in warnings if str(warning.get("severity") or "").lower() == "severe"]
    if status != "valid_joint_one_pass_dataset" and status != "valid_reused_joint_dataset":
        return {
            "winner": None,
            "robust_recommendation": False,
            "status": status,
            "reason": "Comparison is not scientifically valid for winner selection.",
            "severe_warnings": severe,
        }
    if severe:
        return {
            "winner": None,
            "robust_recommendation": False,
            "status": "diagnostic_only",
            "reason": "Severe warnings prevent winner selection.",
            "severe_warnings": severe,
        }
    values = {row["method"]: number(row.get(PRIMARY_METRIC)) for row in summary_rows}
    if not all(math.isfinite(value) for value in values.values()) or len(values) < 2:
        return {
            "winner": None,
            "robust_recommendation": False,
            "status": "diagnostic_only",
            "reason": f"Primary metric {PRIMARY_METRIC} is unavailable for all methods.",
            "severe_warnings": severe,
        }
    winner = min(values, key=values.get)
    return {
        "winner": winner,
        "robust_recommendation": True,
        "status": "robust_candidate",
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_values": values,
        "reason": f"{winner} has the lower {PRIMARY_METRIC}.",
        "severe_warnings": severe,
    }


def _safe_recommendation_for_display(manifest: dict[str, Any]) -> dict[str, Any]:
    status = str(manifest.get("status") or "diagnostic_only")
    recommendation = dict(manifest.get("recommendation") or {})
    if status == "diagnostic_only" or status.startswith("invalid_"):
        recommendation["winner"] = None
        recommendation["robust_recommendation"] = False
        recommendation.setdefault("status", status)
        recommendation.setdefault("reason", "Comparison is not scientifically valid for winner selection.")
    return recommendation


def _plot_rows(summary_rows: list[dict[str, Any]], metric_group: dict[str, Any]) -> list[dict[str, Any]]:
    metric_keys = [metric["key"] for metric in metric_group.get("metrics") or []]
    rows = []
    for row in summary_rows:
        next_row: dict[str, Any] = {"method": row.get("method")}
        for key in metric_keys:
            next_row[key] = finite_or_none(row.get(key))
        rows.append(next_row)
    return rows


def _missing_metrics(rows: list[dict[str, Any]], metric_group: dict[str, Any]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for metric in metric_group.get("metrics") or []:
        key = metric["key"]
        for row in rows:
            if row.get(key) is None:
                missing.append({"method": row.get("method"), "metric": key})
    return missing


def build_common_plot_payload(
    common_metrics_manifest: dict[str, Any] | None,
    *,
    artifact_summary: dict[str, Any] | None = None,
    timing_rows: list[dict[str, Any]] | None = None,
    timing_scaling_rows: list[dict[str, Any]] | None = None,
    metric_scaling_rows: list[dict[str, Any]] | None = None,
    status_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a UI-safe Graph2Mat-vs-DeepH comparison payload.

    The payload is intentionally conservative: diagnostic/invalid summaries never expose
    a winner even if the raw metric rows contain lower values for one method.
    """
    status_payload = dict(status_payload or {})
    timing_scaling_rows = timing_scaling_rows or []
    metric_scaling_rows = metric_scaling_rows or []
    timing_scaling_plots = (
        [
            {
                "id": "timing_scaling",
                "kind": "timing_scaling",
                "title": "Phase time vs dataset size",
                "x_title": "Dataset size (snapshots)",
                "y_title": "Seconds",
                "rows": timing_scaling_rows,
            }
        ]
        if timing_scaling_rows
        else []
    )
    metric_scaling_plots: list[dict[str, Any]] = []
    if metric_scaling_rows:
        for metric_group in COMMON_METRIC_GROUPS:
            metric_keys = {metric["key"] for metric in metric_group.get("metrics") or []}
            rows = [row for row in metric_scaling_rows if row.get("metric_key") in metric_keys]
            if not rows:
                continue
            metric_scaling_plots.append(
                {
                    "id": f"metric_scaling_{metric_group['id']}",
                    "kind": "metric_scaling",
                    "title": f"{metric_group['title']} vs dataset size",
                    "x_title": "Dataset size (snapshots)",
                    "y_title": metric_group.get("y_title") or "Metric value",
                    "metrics": metric_group.get("metrics") or [],
                    "rows": rows,
                }
            )
    if not common_metrics_manifest:
        return {
            "available": bool(timing_scaling_plots or metric_scaling_plots),
            "plots": [*metric_scaling_plots, *timing_scaling_plots],
            "metric_groups": COMMON_METRIC_GROUPS,
            "artifact_summary": artifact_summary or {},
            "timing_rows": timing_rows or [],
            "timing_scaling_rows": timing_scaling_rows,
            "metric_scaling_rows": metric_scaling_rows,
            "message": "No common Graph2Mat/DeepH metrics are available yet.",
            "status": status_payload,
        }

    manifest = dict(common_metrics_manifest)
    summary_rows = [dict(row) for row in manifest.get("summary_rows") or []]
    scientific_status = str(manifest.get("status") or "diagnostic_only")
    recommendation = _safe_recommendation_for_display(manifest)
    plots: list[dict[str, Any]] = []
    for metric_group in COMMON_METRIC_GROUPS:
        rows = _plot_rows(summary_rows, metric_group)
        plots.append(
            {
                "id": metric_group["id"],
                "kind": "grouped_bar",
                "title": metric_group["title"],
                "y_title": metric_group["y_title"],
                "metrics": metric_group["metrics"],
                "rows": rows,
                "missing_metrics": _missing_metrics(rows, metric_group),
            }
        )
    plots.extend(metric_scaling_plots)
    plots.extend(timing_scaling_plots)
    return {
        "available": True,
        "schema": "graph2mat_deeph_plot_payload_v1",
        "scientific_status": scientific_status,
        "diagnostic_only": scientific_status == "diagnostic_only",
        "invalid": scientific_status.startswith("invalid_"),
        "common_metrics": {
            **manifest,
            "recommendation": recommendation,
        },
        "summary_rows": summary_rows,
        "metric_groups": COMMON_METRIC_GROUPS,
        "artifact_summary": artifact_summary or {},
        "timing_rows": timing_rows or [],
        "timing_scaling_rows": timing_scaling_rows,
        "metric_scaling_rows": metric_scaling_rows,
        "plots": plots,
        "warnings": list(manifest.get("warnings") or []),
        "recommendation": recommendation,
        "message": recommendation.get("reason") or "",
        "status": status_payload,
    }


def aggregate_common_metrics(
    *,
    graph2mat_metrics_root: Path,
    deeph_metrics_root: Path,
    output_dir: Path,
    frozen_split_manifest_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_split = read_json(frozen_split_manifest_path) if frozen_split_manifest_path else {}
    dataset_manifest = read_json(dataset_manifest_path) if dataset_manifest_path else {}
    warnings: list[dict[str, Any]] = []
    if frozen_split:
        validate_no_forbidden_references(frozen_split)
    status, dataset_warnings = dataset_status(dataset_manifest)
    warnings.extend(dataset_warnings)

    g2m_ids = sample_ids_from_metrics(graph2mat_metrics_root)
    deeph_ids = sample_ids_from_metrics(deeph_metrics_root)
    if g2m_ids != deeph_ids:
        status = "invalid_incompatible_splits"
        warnings.append(
            {
                "severity": "severe",
                "kind": "mismatched_sample_ids",
                "graph2mat_only": sorted(g2m_ids - deeph_ids),
                "deeph_only": sorted(deeph_ids - g2m_ids),
            }
        )

    summary_rows = [
        summarize_method("graph2mat", graph2mat_metrics_root),
        summarize_method("deeph", deeph_metrics_root),
    ]
    for row in summary_rows:
        if row["method_status"] != "ok":
            status = "invalid_prediction_format"
            warnings.append({"severity": "severe", "kind": f"{row['method']}_metrics_status", "status": row["method_status"]})
        if row["diagnostic_only"]:
            if not status.startswith("invalid_"):
                status = "diagnostic_only"
            warnings.append({"severity": "severe", "kind": f"{row['method']}_diagnostic_only"})
        if row["method"] == "deeph" and not row.get("raw_global_equivalence_proven"):
            if not status.startswith("invalid_"):
                status = "diagnostic_only"
            warnings.append(
                {
                    "severity": "severe",
                    "kind": "deeph_adapter_equivalence_not_proven",
                    "adapter_equivalence_status": row.get("adapter_equivalence_status"),
                    "message": "DeepH adapter did not prove raw/global HSX equivalence.",
                }
            )
        if row.get("uses_reference_overlap_k") is not True:
            if not status.startswith("invalid_"):
                status = "diagnostic_only"
            warnings.append({"severity": "severe", "kind": f"{row['method']}_missing_s_ref"})
        if row.get("kpoint_metrics_enabled") is False:
            if not status.startswith("invalid_"):
                status = "diagnostic_only"
            warnings.append({"severity": "severe", "kind": f"{row['method']}_unsupported_kgrid"})

    sample_rows = [*sample_metric_rows("graph2mat", graph2mat_metrics_root), *sample_metric_rows("deeph", deeph_metrics_root)]
    recommendation = build_recommendation(summary_rows, status, warnings)
    write_csv_rows(output_dir / "common_method_metrics.csv", summary_rows)
    write_csv_rows(output_dir / "common_sample_metrics.csv", sample_rows)
    write_json(output_dir / "recommendation.json", recommendation)
    manifest = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status,
        "status_values": sorted(STATUS_VALUES),
        "graph2mat_metrics_root": str(graph2mat_metrics_root),
        "deeph_metrics_root": str(deeph_metrics_root),
        "output_dir": str(output_dir),
        "sample_ids": sorted(g2m_ids | deeph_ids),
        "warnings": warnings,
        "summary_rows": summary_rows,
        "recommendation": recommendation,
    }
    write_json(output_dir / "common_summary.json", manifest)
    write_json(output_dir / "benchmark_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph2mat-metrics-root", type=Path, required=True)
    parser.add_argument("--deeph-metrics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-split-manifest", type=Path, default=None)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = aggregate_common_metrics(
        graph2mat_metrics_root=args.graph2mat_metrics_root,
        deeph_metrics_root=args.deeph_metrics_root,
        output_dir=args.output_dir,
        frozen_split_manifest_path=args.frozen_split_manifest,
        dataset_manifest_path=args.dataset_manifest,
    )
    print(json.dumps(json_safe({"status": manifest["status"], "output_dir": manifest["output_dir"]}), ensure_ascii=False))


if __name__ == "__main__":
    main()
