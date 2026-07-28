#!/usr/bin/env python3
"""Best-run ranking for Graph2Mat-vs-DeepH benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from analyze_winners import (
    mean,
    metric_lower_is_better as _legacy_metric_lower_is_better,
    metric_policy_role as _legacy_metric_policy_role,
    seed_stability_status,
    stddev,
    valid_stability_seeds,
    warning_items,
)
from deeph_prediction_adapter import (
    EQUIVALENCE_PROVEN_RAW_GLOBAL,
    EQUIVALENCE_STATUS_PROVEN,
    EQUIVALENCE_STATUS_UNPROVEN,
)
from g2m_deeph_metrics import summarize_method
from g2m_deeph_test_blindness import assert_no_test_metrics_for_search
from joint_artifact_contract import (
    material_profile_errors,
    md_temporal_evidence_errors,
    validate_recorded_snapshots,
)
from reference_selection import choose_reference_matrix


SCHEMA = "graph2mat_deeph_run_ranking_v1"
MODELS = ("graph2mat", "deeph")
METRIC_FAIL_POLICY_FAIL_CLOSED = "fail_closed"
METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY = "diagnostic_only"
VALID_DATASET_STATUSES = {"valid_joint_one_pass_dataset", "valid_reused_joint_dataset"}
VALID_ARTIFACT_CONTRACT_STATUSES = {"valid", "benchmark_ready"}
VALID_SPLIT_AUDIT_STATUSES = {"valid"}
VALID_COMPARABILITY_STATUSES = {"valid", "valid_joint_one_pass_dataset", "valid_reused_joint_dataset"}
RECOMMENDATION_STATUS_VALUES = {
    "robust_graph2mat_win",
    "robust_deeph_win",
    "exploratory_graph2mat_win",
    "exploratory_deeph_win",
    "no_robust_winner",
    "diagnostic_only",
    "invalid_incomplete_grid",
    "invalid_incompatible_splits",
    "invalid_incompatible_artifacts",
    "invalid_missing_provenance",
    "invalid_prediction_format",
    "invalid_metric_policy",
    "invalid_unverified_deeph_split",
    "unstable_across_seeds",
}
STATUS_PRIORITY = (
    "invalid_incompatible_splits",
    "invalid_incompatible_artifacts",
    "invalid_missing_provenance",
    "invalid_unverified_deeph_split",
    "invalid_prediction_format",
    "invalid_metric_policy",
    "invalid_incomplete_grid",
    "unstable_across_seeds",
    "diagnostic_only",
)
DIAGNOSTIC_GATES = {
    "diagnostic_only",
    "metric_fail_policy_diagnostic_only",
    "deeph_adapter_equivalence_not_proven",
}
PRIMARY_METRIC_PRIORITY = [
    "low_energy_rmse_eV",
    "fermi_window_rmse_eV",
    "frontier_window_rmse_eV",
    "dos_mae_500_fermi_window",
    "dos_wasserstein_eV",
    "relative_frobenius",
    "relative_frobenius_union",
    "h_mae_eV",
]
DIAGNOSTIC_ONLY_METRICS = {
    "global_rmse_eV",
    "global_mae_eV",
    "support_precision",
    "support_recall",
    "support_f1",
    "hermiticity_pred",
    "hermiticity_error",
    "antihermitian_norm",
}
RECOMMENDATION_GRADE_METRICS = {
    "low_energy_rmse_eV",
    "fermi_window_rmse_eV",
    "frontier_window_rmse_eV",
    "dos_mae_500_fermi_window",
    "dos_wasserstein_eV",
    "relative_frobenius",
    "relative_frobenius_union",
    "h_mae_eV",
}
METRIC_ALIASES = {
    "h_mae_eV_mean": "h_mae_eV",
    "h_rmse_eV_mean": "h_rmse_eV",
    "h_mse_eV2_mean": "h_mse_eV2",
    "r2_mean": "r2",
    "relative_frobenius_mean": "relative_frobenius",
    "support_precision_mean": "support_precision",
    "support_recall_mean": "support_recall",
    "support_f1_mean": "support_f1",
    "hermiticity_pred_mean": "hermiticity_pred",
    "global_rmse_eV_mean": "global_rmse_eV",
    "low_energy_rmse_eV_mean": "low_energy_rmse_eV",
    "fermi_window_rmse_eV_mean": "fermi_window_rmse_eV",
    "frontier_window_rmse_eV_mean": "frontier_window_rmse_eV",
    "dos_mae_500_fermi_window_mean": "dos_mae_500_fermi_window",
}
CANONICAL_TO_SOURCE = {value: key for key, value in METRIC_ALIASES.items()}
HIGHER_IS_BETTER = {"r2", "support_f1"}


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: json_safe(value) for key, value in row.items()} for row in rows])


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def finite(value: Any) -> bool:
    return math.isfinite(number(value))


def normalize_model(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "g2m": "graph2mat",
        "graph_2_mat": "graph2mat",
        "deep_h": "deeph",
        "deeph_pack": "deeph",
    }
    return aliases.get(text, text)


def canonical_metric(metric: str) -> str:
    text = str(metric or "").strip()
    if text in METRIC_ALIASES:
        return METRIC_ALIASES[text]
    if text.endswith("_mean") and text.removesuffix("_mean") in RECOMMENDATION_GRADE_METRICS | DIAGNOSTIC_ONLY_METRICS:
        return text.removesuffix("_mean")
    return text


def source_metric(metric: str) -> str:
    canonical = canonical_metric(metric)
    return CANONICAL_TO_SOURCE.get(canonical, metric)


def metric_lower_is_better(metric: str) -> bool:
    canonical = canonical_metric(metric)
    if canonical in HIGHER_IS_BETTER:
        return False
    return _legacy_metric_lower_is_better(canonical)


def metric_policy_role(metric: str) -> str:
    canonical = canonical_metric(metric)
    if canonical in DIAGNOSTIC_ONLY_METRICS:
        return "diagnostic_only"
    if canonical in RECOMMENDATION_GRADE_METRICS:
        return "recommendation_grade"
    legacy = _legacy_metric_policy_role(canonical)
    return legacy if legacy != "unknown_metric" else "diagnostic_only"


def severe_warning_items(*values: Any) -> list[Any]:
    severe: list[Any] = []
    for value in values:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and str(item.get("severity") or "").lower() == "severe":
                    severe.append(item)
                elif isinstance(item, str) and "severe" in item.lower():
                    severe.append(item)
        elif isinstance(value, dict):
            if str(value.get("severity") or "").lower() == "severe":
                severe.append(value)
        else:
            for item in warning_items(value):
                if "severe" in str(item).lower():
                    severe.append(item)
    return severe


def warning_status(severe_warnings: list[Any]) -> str:
    return "severe" if severe_warnings else "ok"


def metric_fail_policy_warning(policy: str) -> dict[str, str] | None:
    if policy != METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY:
        return None
    return {
        "severity": "severe",
        "kind": "metric_fail_policy_diagnostic_only",
        "message": "Metrics were produced in explicit fail-open diagnostic mode; robust ranking is disabled.",
    }


def deeph_adapter_equivalence_warning(
    status: str,
    *,
    equivalence_status: str = "",
    reason: str = "",
) -> dict[str, str]:
    return {
        "severity": "severe",
        "kind": "deeph_adapter_equivalence_not_proven",
        "adapter_equivalence_status": status or "missing",
        "equivalence_status": equivalence_status or "missing",
        "diagnostic_only_reason": reason or "DeepH raw/global equivalence evidence is not proven.",
        "message": "DeepH prediction equivalence to Graph2Mat raw/global HSX is not proven.",
    }


def deeph_adapter_equivalence_proven(row: dict[str, Any]) -> bool:
    if normalize_model(row.get("model")) != "deeph":
        return True
    adapter_status = str(row.get("adapter_equivalence_status") or "")
    equivalence_status = str(row.get("equivalence_status") or "")
    if equivalence_status:
        return adapter_status == EQUIVALENCE_PROVEN_RAW_GLOBAL and equivalence_status == EQUIVALENCE_STATUS_PROVEN
    return adapter_status == EQUIVALENCE_PROVEN_RAW_GLOBAL


def deeph_adapter_status(rows: list[dict[str, Any]]) -> str:
    statuses = sorted(
        {
            str(row.get("adapter_equivalence_status") or "")
            for row in rows
            if normalize_model(row.get("model")) == "deeph" and str(row.get("adapter_equivalence_status") or "")
        }
    )
    return statuses[0] if len(statuses) == 1 else ",".join(statuses) if statuses else "missing"


def deeph_equivalence_status(rows: list[dict[str, Any]]) -> str:
    statuses = sorted(
        {
            str(row.get("equivalence_status") or "")
            for row in rows
            if normalize_model(row.get("model")) == "deeph" and str(row.get("equivalence_status") or "")
        }
    )
    if statuses:
        return statuses[0] if len(statuses) == 1 else ",".join(statuses)
    adapter_status = deeph_adapter_status(rows)
    return EQUIVALENCE_STATUS_PROVEN if adapter_status == EQUIVALENCE_PROVEN_RAW_GLOBAL else EQUIVALENCE_STATUS_UNPROVEN


def split_audit_status(rows: list[dict[str, Any]]) -> str:
    statuses = sorted(
        {
            str(row.get("split_audit_status") or "")
            for row in rows
            if normalize_model(row.get("model")) == "deeph" and str(row.get("split_audit_status") or "")
        }
    )
    return statuses[0] if len(statuses) == 1 else ",".join(statuses) if statuses else "missing"


def dataset_metadata(dataset_root: Path | None) -> dict[str, Any]:
    if dataset_root is None:
        return {}
    split = read_json(dataset_root / "frozen_split_manifest.json")
    manifest = read_json(dataset_root / "benchmark_dataset_manifest.json")
    compatibility_hash = (
        manifest.get("dataset_compatibility_hash")
        or manifest.get("material_compatibility_hash")
        or manifest.get("benchmark_dataset_id")
        or manifest.get("siesta_input_sha256")
        or str(dataset_root)
    )
    provenance = manifest.get("provenance_status") or {}
    provenance_valid = bool(provenance.get("valid")) if isinstance(provenance, dict) and provenance else bool(manifest.get("benchmark_ready"))
    artifact_validation = read_json(dataset_root / "artifact_validation.json")
    live_results, live_errors = validate_recorded_snapshots(
        artifact_validation,
        base_dir=dataset_root,
    )
    reference_errors = [
        f"{result.snapshot_dir}: {selection.reason}"
        for result in live_results
        if not (selection := choose_reference_matrix(result.snapshot_dir)).ok
    ]
    temporal_errors = md_temporal_evidence_errors(dataset_root)
    profile_errors = material_profile_errors(dataset_root, manifest)
    artifacts_valid = (
        manifest.get("benchmark_ready") is True
        and artifact_validation.get("valid") is True
        and bool(live_results)
        and not live_errors
        and not reference_errors
        and not temporal_errors
        and not profile_errors
    )
    return {
        "dataset_id": manifest.get("benchmark_dataset_id") or dataset_root.name,
        "dataset_label": manifest.get("material_label") or dataset_root.name,
        "dataset_recipe_id": manifest.get("dataset_recipe_id") or "",
        "dataset_compatibility_hash": compatibility_hash,
        "frozen_split_hash": split.get("split_hash") or (manifest.get("frozen_split_manifest") or {}).get("split_hash") or "",
        "artifact_contract_status": "valid" if artifacts_valid else "invalid",
        "provenance_status": "valid" if provenance_valid else "invalid",
        "required_provenance_present": provenance_valid,
        "dataset_scientific_status": "valid" if artifacts_valid else "invalid",
        "material_profile": manifest.get("material_profile") or "missing",
        "live_artifact_validation_errors": (
            live_errors + reference_errors + temporal_errors + profile_errors
        ),
    }


def deeph_manifest_metadata(record: dict[str, Any]) -> dict[str, Any]:
    if normalize_model(record.get("model")) != "deeph":
        return {"split_audit_status": "not_applicable"}
    manifest_path = record.get("deeph_manifest_path")
    manifest = read_json(Path(str(manifest_path))) if manifest_path else {}
    audit = manifest.get("split_audit") or {}
    status = (
        record.get("split_audit_status")
        or manifest.get("split_audit_status")
        or audit.get("status")
        or "missing"
    )
    return {
        "split_audit_status": str(status),
        "split_audit_path": str(manifest.get("split_audit_path") or record.get("split_audit_path") or ""),
    }


def metric_root_for_record(record: dict[str, Any]) -> Path | None:
    run_root = Path(str(record.get("run_root") or ""))
    model = normalize_model(record.get("model"))
    if not str(run_root) or not run_root.exists():
        return None
    candidates = (
        [run_root / "metrics" / "graph2mat" / "eval_input" / "metrics", run_root / "common_metrics" / "graph2mat_eval" / "metrics"]
        if model == "graph2mat"
        else [run_root / "metrics" / "deeph" / "eval" / "metrics", run_root / "common_metrics" / "deeph_eval" / "metrics"]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def timing_seconds(record: dict[str, Any]) -> dict[str, float | None]:
    train = number((record.get("train_run") or {}).get("elapsed_seconds"))
    predict = number((record.get("predict_run") or {}).get("elapsed_seconds"))
    metrics = number((record.get("metrics_run") or {}).get("elapsed_seconds"))
    preprocess = number((record.get("preprocess_run") or {}).get("elapsed_seconds"))
    inference = sum(
        number(run.get("elapsed_seconds"))
        for run in record.get("inference_runs") or []
        if isinstance(run, dict) and finite(run.get("elapsed_seconds"))
    )
    values = [train, predict, metrics, preprocess, inference]
    total = sum(value for value in values if math.isfinite(value))
    return {
        "training_time_seconds": train if math.isfinite(train) else None,
        "prediction_time_seconds": predict if math.isfinite(predict) else (inference if inference else None),
        "preprocess_time_seconds": preprocess if math.isfinite(preprocess) else None,
        "evaluation_time_seconds": metrics if math.isfinite(metrics) else None,
        "total_time_seconds": total if total else None,
    }


def telemetry_fields(record: dict[str, Any]) -> dict[str, Any]:
    telemetry = record.get("telemetry") if isinstance(record.get("telemetry"), dict) else {}
    if not telemetry and record.get("telemetry_path"):
        telemetry = read_json(Path(str(record.get("telemetry_path"))))
    if not isinstance(telemetry, dict) or not telemetry:
        return {
            "telemetry_status": "unavailable",
            "wall_clock_seconds_total": None,
            "gpu_hours_total": None,
            "gpu_hours_to_best_validation": None,
            "wall_clock_seconds_to_best_validation": None,
            "peak_gpu_memory_mb": None,
            "samples_per_second": None,
            "matrix_blocks_per_second": None,
            "epochs_trained": None,
            "best_validation_epoch": None,
            "telemetry_warnings": ["telemetry unavailable"],
        }
    return {
        "telemetry_status": telemetry.get("telemetry_status") or "partial",
        "wall_clock_seconds_total": telemetry.get("wall_clock_seconds_total"),
        "gpu_hours_total": telemetry.get("gpu_hours_total"),
        "gpu_hours_to_best_validation": telemetry.get("gpu_hours_to_best_validation"),
        "wall_clock_seconds_to_best_validation": telemetry.get("wall_clock_seconds_to_best_validation"),
        "peak_gpu_memory_mb": telemetry.get("peak_gpu_memory_mb"),
        "samples_per_second": telemetry.get("samples_per_second"),
        "matrix_blocks_per_second": telemetry.get("matrix_blocks_per_second"),
        "epochs_trained": telemetry.get("epochs_trained"),
        "best_validation_epoch": telemetry.get("best_validation_epoch"),
        "telemetry_warnings": telemetry.get("telemetry_warnings") or [],
        "hardware": telemetry.get("hardware") or {},
    }


def early_stopping_fields(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("early_stopping") if isinstance(record.get("early_stopping"), dict) else {}
    if not metadata:
        return {}
    return {
        "validation_metric_name": metadata.get("validation_metric_name"),
        "metric_mode": metadata.get("metric_mode"),
        "early_stopping_patience": metadata.get("patience"),
        "early_stopping_min_delta": metadata.get("min_delta"),
        "early_stopping_max_epochs": metadata.get("max_epochs"),
        "early_stopping_best_epoch": metadata.get("best_epoch"),
        "early_stopping_best_validation_value": metadata.get("best_validation_value"),
        "early_stopping_epochs_trained": metadata.get("epochs_trained"),
        "early_stopping_stop_reason": metadata.get("stop_reason"),
    }


def row_from_training_record(record: dict[str, Any]) -> dict[str, Any]:
    model = normalize_model(record.get("model"))
    if model not in MODELS:
        raise RuntimeError(f"Unsupported Graph2Mat/DeepH ranking model: {record.get('model')}")
    if not record.get("config_id"):
        raise RuntimeError("Training sweep run is missing config_id.")
    dataset_root = Path(str(record.get("dataset_root") or "")) if record.get("dataset_root") else None
    metadata = dataset_metadata(dataset_root)
    metrics_root = metric_root_for_record(record)
    method_summary: dict[str, Any] = {}
    metric_warnings: list[Any] = []
    if metrics_root is not None:
        method_summary = summarize_method(model, metrics_root)
        metric_manifest = read_json(metrics_root / "manifest.json")
        metric_warnings.extend(metric_manifest.get("warnings") or [])
        metric_warnings.extend(metric_manifest.get("fatal_errors") or [])
    metric_fail_policy = str(record.get("metric_fail_policy") or METRIC_FAIL_POLICY_FAIL_CLOSED)
    metric_policy_warning = metric_fail_policy_warning(metric_fail_policy)
    if metric_policy_warning:
        metric_warnings.append(metric_policy_warning)
    deeph_manifest = deeph_manifest_metadata(record)
    adapter_equivalence_status = str(
        method_summary.get("adapter_equivalence_status")
        or record.get("adapter_equivalence_status")
        or ""
    )
    equivalence_status = str(
        method_summary.get("equivalence_status")
        or record.get("equivalence_status")
        or (EQUIVALENCE_STATUS_PROVEN if adapter_equivalence_status == EQUIVALENCE_PROVEN_RAW_GLOBAL else "")
    )
    equivalence_scope = str(method_summary.get("equivalence_scope") or record.get("equivalence_scope") or "")
    equivalence_gate = method_summary.get("equivalence_gate") if isinstance(method_summary.get("equivalence_gate"), dict) else {}
    diagnostic_reason = str(
        method_summary.get("diagnostic_only_reason")
        or record.get("diagnostic_only_reason")
        or equivalence_gate.get("diagnostic_only_reason")
        or ""
    )
    if model == "deeph" and not deeph_adapter_equivalence_proven(
        {
            "model": model,
            "adapter_equivalence_status": adapter_equivalence_status,
            "equivalence_status": equivalence_status,
        }
    ):
        metric_warnings.append(
            deeph_adapter_equivalence_warning(
                adapter_equivalence_status,
                equivalence_status=equivalence_status,
                reason=diagnostic_reason,
            )
        )
    severe = severe_warning_items(record.get("severe_warnings"), record.get("warnings"), metric_warnings)
    method_status = method_summary.get("method_status") or ("missing_metrics" if record.get("status") == "completed" else record.get("status"))
    diagnostic_only = bool(method_summary.get("diagnostic_only")) or metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY
    if model == "deeph" and not deeph_adapter_equivalence_proven(
        {
            "model": model,
            "adapter_equivalence_status": adapter_equivalence_status,
            "equivalence_status": equivalence_status,
        }
    ):
        diagnostic_only = True
    comparability_status = "diagnostic_only" if diagnostic_only else "valid"
    if method_status not in {"ok", "dry_run", "failed", "missing_metrics", None}:
        comparability_status = "invalid_prediction_format"
    overrides = record.get("overrides") if isinstance(record.get("overrides"), dict) else {}
    common = record.get("common") if isinstance(record.get("common"), dict) else {}
    epochs = overrides.get("max_epochs") or overrides.get("epochs") or common.get("epochs")
    row = {
        "benchmark_id": Path(str(record.get("run_root") or "")).parts[-5] if record.get("run_root") and len(Path(str(record.get("run_root"))).parts) >= 5 else "",
        **metadata,
        "model": model,
        "config_id": str(record.get("config_id")),
        "config_label": str(record.get("config_label") or record.get("config_id")),
        "config_hash": str(record.get("config_hash") or ""),
        "epochs": epochs,
        "epoch_label": f"{epochs} epochs" if epochs not in (None, "") else "",
        "seed": (record.get("common") or {}).get("seed")
        or (record.get("overrides") or {}).get("seed")
        or (record.get("overrides") or {}).get("seed_everything")
        or "unknown",
        "run_status": str(record.get("status") or "unknown"),
        "protocol_stage": str(record.get("protocol_stage") or "exploratory"),
        "metric_split": str(record.get("metric_split") or ("test" if record.get("metrics_run") else "")),
        "test_metrics_locked": bool(record.get("test_metrics_locked")),
        "test_metrics_status": str(record.get("test_metrics_status") or ""),
        "method_status": method_status,
        "prediction_dir": str(Path(str(record.get("run_root") or "")) / "graph2mat" / "prediction_structures")
        if model == "graph2mat"
        else str(Path(str(record.get("run_root") or "")) / "deeph" / "inference"),
        "reference_dir": str(dataset_root or ""),
        "run_dir": str(record.get("run_root") or ""),
        "metrics_root": str(metrics_root or ""),
        "metric_fail_policy": metric_fail_policy,
        "fail_open_metric_outputs": metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
        "comparability_status": comparability_status,
        "scientific_status": comparability_status,
        "warning_status": warning_status(severe),
        "severe_warnings": severe,
        "diagnostic_only": diagnostic_only,
        "adapter_equivalence_status": adapter_equivalence_status,
        "equivalence_status": equivalence_status,
        "equivalence_scope": equivalence_scope,
        "diagnostic_only_reason": diagnostic_reason,
        "raw_global_equivalence_proven": bool(method_summary.get("raw_global_equivalence_proven")),
        **deeph_manifest,
        **timing_seconds(record),
        **telemetry_fields(record),
        **early_stopping_fields(record),
    }
    for key, value in method_summary.items():
        if key.endswith("_mean"):
            row[key] = value
    return row


def rows_from_common_metrics(
    common_metrics_manifest: dict[str, Any],
    *,
    dataset_root: Path | None = None,
    frozen_split_manifest_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    metadata = dataset_metadata(dataset_root)
    if frozen_split_manifest_path:
        split = read_json(frozen_split_manifest_path)
        metadata["frozen_split_hash"] = split.get("split_hash") or metadata.get("frozen_split_hash") or ""
    if dataset_manifest_path:
        manifest = read_json(dataset_manifest_path)
        metadata["dataset_compatibility_hash"] = (
            manifest.get("dataset_compatibility_hash")
            or manifest.get("material_compatibility_hash")
            or manifest.get("benchmark_dataset_id")
            or metadata.get("dataset_compatibility_hash")
        )
        metadata["artifact_contract_status"] = "valid" if manifest.get("benchmark_ready") else metadata.get("artifact_contract_status", "unknown")
    metric_fail_policy = str(common_metrics_manifest.get("metric_fail_policy") or METRIC_FAIL_POLICY_FAIL_CLOSED)
    policy_warning = metric_fail_policy_warning(metric_fail_policy)
    severe_inputs = [common_metrics_manifest.get("warnings")]
    if policy_warning:
        severe_inputs.append([policy_warning])
    severe = severe_warning_items(*severe_inputs)
    manifest_status = str(common_metrics_manifest.get("status") or "diagnostic_only")
    if metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY:
        manifest_status = "diagnostic_only"
    rows: list[dict[str, Any]] = []
    for item in common_metrics_manifest.get("summary_rows") or []:
        model = normalize_model(item.get("method"))
        if model not in MODELS:
            continue
        adapter_equivalence_status = str(item.get("adapter_equivalence_status") or "")
        equivalence_status = str(
            item.get("equivalence_status")
            or (EQUIVALENCE_STATUS_PROVEN if adapter_equivalence_status == EQUIVALENCE_PROVEN_RAW_GLOBAL else "")
        )
        equivalence_scope = str(item.get("equivalence_scope") or "")
        equivalence_gate = item.get("equivalence_gate") if isinstance(item.get("equivalence_gate"), dict) else {}
        diagnostic_reason = str(item.get("diagnostic_only_reason") or equivalence_gate.get("diagnostic_only_reason") or "")
        item_severe = list(severe)
        if model == "deeph" and not deeph_adapter_equivalence_proven(
            {
                "model": model,
                "adapter_equivalence_status": adapter_equivalence_status,
                "equivalence_status": equivalence_status,
            }
        ):
            item_severe.append(
                deeph_adapter_equivalence_warning(
                    adapter_equivalence_status,
                    equivalence_status=equivalence_status,
                    reason=diagnostic_reason,
                )
            )
        row = {
            "benchmark_id": "",
            **metadata,
            "model": model,
            "config_id": f"default_{model}",
            "config_label": f"default_{model}",
            "config_hash": "",
            "seed": "unknown",
            "run_status": "completed",
            "method_status": item.get("method_status") or "ok",
            "prediction_dir": "",
            "reference_dir": str(dataset_root or ""),
            "run_dir": "",
            "metrics_root": str(item.get("metrics_root") or ""),
            "metric_fail_policy": metric_fail_policy,
            "fail_open_metric_outputs": metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
            "comparability_status": manifest_status,
            "scientific_status": manifest_status,
            "warning_status": warning_status(item_severe),
            "severe_warnings": item_severe,
            "diagnostic_only": bool(item.get("diagnostic_only"))
            or manifest_status == "diagnostic_only"
            or metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
            "adapter_equivalence_status": adapter_equivalence_status,
            "equivalence_status": equivalence_status,
            "equivalence_scope": equivalence_scope,
            "diagnostic_only_reason": diagnostic_reason,
            "raw_global_equivalence_proven": bool(item.get("raw_global_equivalence_proven")),
            "split_audit_status": str(item.get("split_audit_status") or ("missing" if model == "deeph" else "not_applicable")),
            "split_audit_path": str(item.get("split_audit_path") or ""),
            "training_time_seconds": None,
            "prediction_time_seconds": None,
            "preprocess_time_seconds": None,
            "evaluation_time_seconds": None,
            "total_time_seconds": None,
        }
        for key, value in item.items():
            if key.endswith("_mean"):
                row[key] = value
        if model == "deeph" and not deeph_adapter_equivalence_proven(row):
            row["diagnostic_only"] = True
            row["comparability_status"] = "diagnostic_only"
            row["scientific_status"] = "diagnostic_only"
        rows.append(row)
    return rows


def load_metric_rows(
    *,
    training_sweep_manifest_path: Path | None = None,
    common_metrics_manifest_path: Path | None = None,
    dataset_root: Path | None = None,
    frozen_split_manifest_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    training = read_json(training_sweep_manifest_path)
    for record in training.get("runs") or []:
        if not isinstance(record, dict):
            continue
        rows.append(row_from_training_record(record))
    common = read_json(common_metrics_manifest_path)
    if common:
        rows.extend(
            rows_from_common_metrics(
                common,
                dataset_root=dataset_root,
                frozen_split_manifest_path=frozen_split_manifest_path,
                dataset_manifest_path=dataset_manifest_path,
            )
        )
    return rows


def finite_metric(row: dict[str, Any], metric: str) -> float | None:
    value = number(row.get(source_metric(metric)))
    return value if math.isfinite(value) else None


def row_is_robust_eligible(row: dict[str, Any], metric: str) -> bool:
    return not row_gate_failures(row, metric)


def row_gate_failures(row: dict[str, Any], metric: str) -> list[str]:
    failures: list[str] = []
    model = normalize_model(row.get("model"))
    if row.get("run_status") not in {"completed", "ok"}:
        failures.append("invalid_incomplete_grid")
    if finite_metric(row, metric) is None:
        failures.append("invalid_incomplete_grid")
    if row.get("method_status") != "ok":
        failures.append("invalid_prediction_format")
    if str(row.get("artifact_contract_status") or "") not in VALID_ARTIFACT_CONTRACT_STATUSES:
        failures.append("invalid_incompatible_artifacts")
    if row.get("required_provenance_present") is not True or str(row.get("provenance_status") or "") != "valid":
        failures.append("invalid_missing_provenance")
    comparability = str(row.get("comparability_status") or "")
    if comparability not in VALID_COMPARABILITY_STATUSES:
        failures.append(comparability if comparability.startswith("invalid_") else "diagnostic_only")
    if row.get("warning_status") == "severe":
        failures.append("severe_warnings")
    if metric_policy_role(metric) == "diagnostic_only":
        failures.append("invalid_metric_policy")
    if row.get("fail_open_metric_outputs") or row.get("metric_fail_policy") == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY:
        failures.append("metric_fail_policy_diagnostic_only")
    if model == "deeph":
        if not deeph_adapter_equivalence_proven(row):
            failures.append("deeph_adapter_equivalence_not_proven")
        split_status = str(row.get("split_audit_status") or "missing")
        if split_status not in VALID_SPLIT_AUDIT_STATUSES:
            failures.append(split_status if split_status == "invalid_incompatible_splits" else "invalid_unverified_deeph_split")
    return sorted(set(failures))


def status_from_gates(gates_failed: list[str]) -> str:
    gates = set(gates_failed)
    if gates & DIAGNOSTIC_GATES:
        return "diagnostic_only"
    if "missing_model" in gates or "missing_primary_metric" in gates:
        return "invalid_incomplete_grid"
    for status in STATUS_PRIORITY:
        if status in gates:
            return status
    if "insufficient_seeds" in gates:
        return "no_robust_winner"
    if "severe_warnings" in gates:
        return "no_robust_winner"
    return "no_robust_winner"


def passed_gates(gates_failed: list[str]) -> list[str]:
    failed = set(gates_failed)
    gate_map = {
        "complete_required_grid": {"invalid_incomplete_grid", "missing_model", "missing_primary_metric"},
        "valid_artifact_contract": {"invalid_incompatible_artifacts"},
        "required_provenance": {"invalid_missing_provenance"},
        "same_frozen_split": {"invalid_incompatible_splits"},
        "deeph_split_audit_pass": {"invalid_unverified_deeph_split"},
        "adapter_equivalence_pass": {"deeph_adapter_equivalence_not_proven"},
        "production_fail_closed_metrics": {"metric_fail_policy_diagnostic_only"},
        "recommendation_grade_metric": {"invalid_metric_policy"},
        "valid_prediction_format": {"invalid_prediction_format"},
        "no_severe_warnings": {"severe_warnings"},
        "seed_stability": {"unstable_across_seeds", "insufficient_seeds"},
    }
    return [gate for gate, blockers in gate_map.items() if not (failed & blockers)]


def choose_primary_metric(rows: list[dict[str, Any]]) -> str | None:
    for metric in PRIMARY_METRIC_PRIORITY:
        values_by_model: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row.get("model") not in MODELS:
                continue
            value = finite_metric(row, metric)
            if value is not None:
                values_by_model[str(row["model"])].append(value)
        if all(values_by_model.get(model) for model in MODELS):
            return metric
    return None


def rank_metric_groups(rows: list[dict[str, Any]], metric: str, *, scope: str = "dataset") -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = finite_metric(row, metric)
        if value is None:
            continue
        dataset_key = str(row.get("dataset_id") or "unknown") if scope == "dataset" else "all"
        groups[(dataset_key, str(row.get("model")), str(row.get("config_id")))].append(row)
    summaries: list[dict[str, Any]] = []
    for (dataset_id, model, config_id), group_rows in groups.items():
        values = [finite_metric(row, metric) for row in group_rows]
        clean = [value for value in values if value is not None]
        seeds = [row.get("seed") for row in group_rows]
        severe = any(row.get("warning_status") == "severe" for row in group_rows)
        eligible = all(row_is_robust_eligible(row, metric) for row in group_rows)
        group_gate_failures = sorted({failure for row in group_rows for failure in row_gate_failures(row, metric)})
        seed_status = seed_stability_status(seeds, [model], has_severe_warning=severe)
        sample = group_rows[0]
        summaries.append(
            {
                "scope": scope,
                "dataset_id": dataset_id,
                "model": model,
                "config_id": config_id,
                "config_label": sample.get("config_label") or config_id,
                "metric": metric,
                "metric_column": source_metric(metric),
                "mean": mean(clean),
                "std": stddev(clean),
                "n_samples": len(clean),
                "n_seeds": len(set(str(seed) for seed in seeds)),
                "valid_seed_count": len(valid_stability_seeds(seeds)),
                "seed_stability_status": seed_status,
                "comparability_status": sample.get("comparability_status"),
                "scientific_status": "robust_candidate" if eligible and seed_status == "robust_candidate" else seed_status,
                "warning_status": "severe" if severe else "ok",
                "severe_warnings": [item for row in group_rows for item in row.get("severe_warnings") or []],
                "adapter_equivalence_status": sample.get("adapter_equivalence_status"),
                "raw_global_equivalence_proven": sample.get("raw_global_equivalence_proven"),
                "split_audit_status": sample.get("split_audit_status"),
                "run_dir": sample.get("run_dir"),
                "prediction_dir": sample.get("prediction_dir"),
                "dataset_compatibility_hash": sample.get("dataset_compatibility_hash"),
                "frozen_split_hash": sample.get("frozen_split_hash"),
                "total_time_seconds": mean([number(row.get("total_time_seconds")) for row in group_rows if finite(row.get("total_time_seconds"))]),
                "robust_eligible": eligible,
                "gates_failed": group_gate_failures,
                "metric_policy_role": metric_policy_role(metric),
            }
        )
    lower = metric_lower_is_better(metric)
    ranked: list[dict[str, Any]] = []
    by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in summaries:
        by_bucket[(str(item["dataset_id"]), str(item["model"]))].append(item)
    for bucket, items in sorted(by_bucket.items()):
        sorted_items = sorted(
            items,
            key=lambda item: (
                number(item["mean"]) if lower else -number(item["mean"]),
                str(item["config_id"]),
            ),
        )
        previous_value: float | None = None
        previous_rank = 0
        for index, item in enumerate(sorted_items, start=1):
            value = number(item["mean"])
            rank = previous_rank if previous_value is not None and value == previous_value else index
            previous_value = value
            previous_rank = rank
            ranked.append({**item, "rank": rank, "tie": rank != index})
    return ranked


def build_metric_rankings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = sorted({canonical_metric(key) for row in rows for key, value in row.items() if key.endswith("_mean") and finite(value)})
    rankings: list[dict[str, Any]] = []
    for metric in metrics:
        rankings.extend(rank_metric_groups(rows, metric, scope="dataset"))
    return rankings


def best_runs_by_model(rows: list[dict[str, Any]], primary_metric: str | None) -> list[dict[str, Any]]:
    if not primary_metric:
        return []
    dataset_rankings = [row for row in rank_metric_groups(rows, primary_metric, scope="dataset") if row["rank"] == 1]
    global_rankings = [row for row in rank_metric_groups(rows, primary_metric, scope="global") if row["rank"] == 1]
    return [*dataset_rankings, *global_rankings]


def pairwise_comparisons(best_rows: list[dict[str, Any]], *, baseline_model: str = "graph2mat") -> list[dict[str, Any]]:
    by_dataset_metric: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in best_rows:
        if row.get("scope") != "dataset":
            continue
        by_dataset_metric[(str(row.get("dataset_id")), str(row.get("metric")))][str(row.get("model"))] = row
    pairs: list[dict[str, Any]] = []
    challenger_model = "deeph" if baseline_model == "graph2mat" else "graph2mat"
    for (dataset_id, metric), methods in sorted(by_dataset_metric.items()):
        baseline = methods.get(baseline_model)
        challenger = methods.get(challenger_model)
        if not baseline or not challenger:
            pairs.append(
                {
                    "dataset_id": dataset_id,
                    "metric": metric,
                    "status": "non_comparative",
                    "winner": None,
                    "reason": "Both Graph2Mat and DeepH are required for pairwise comparison.",
                }
            )
            continue
        gates_failed: list[str] = []
        for key, label in (
            ("frozen_split_hash", "invalid_incompatible_splits"),
            ("dataset_compatibility_hash", "invalid_incompatible_artifacts"),
        ):
            if baseline.get(key) != challenger.get(key):
                gates_failed.append(label)
        if not baseline.get("robust_eligible"):
            gates_failed.extend(baseline.get("gates_failed") or ["invalid_prediction_format"])
        if not challenger.get("robust_eligible"):
            gates_failed.extend(challenger.get("gates_failed") or ["invalid_prediction_format"])
        if challenger_model == "deeph" and not deeph_adapter_equivalence_proven(challenger):
            gates_failed.append("deeph_adapter_equivalence_not_proven")
        if baseline_model == "deeph" and not deeph_adapter_equivalence_proven(baseline):
            gates_failed.append("deeph_adapter_equivalence_not_proven")
        lower = metric_lower_is_better(metric)
        baseline_value = number(baseline.get("mean"))
        challenger_value = number(challenger.get("mean"))
        gates_failed = sorted(set(gates_failed))
        if gates_failed:
            status = status_from_gates(gates_failed)
            winner = None
            improvement = None
        else:
            challenger_better = challenger_value < baseline_value if lower else challenger_value > baseline_value
            winner = challenger_model if challenger_better else baseline_model
            status = "comparable"
            if baseline_value:
                improvement = (
                    (baseline_value - challenger_value) / abs(baseline_value) * 100.0
                    if lower
                    else (challenger_value - baseline_value) / abs(baseline_value) * 100.0
                )
            else:
                improvement = None
        pairs.append(
            {
                "dataset_id": dataset_id,
                "metric": metric,
                "baseline_model": baseline_model,
                "challenger_model": challenger_model,
                "baseline_config_id": baseline.get("config_id"),
                "challenger_config_id": challenger.get("config_id"),
                "baseline_value": baseline_value,
                "challenger_value": challenger_value,
                "absolute_difference": challenger_value - baseline_value,
                "percent_improvement_challenger_vs_baseline": improvement,
                "lower_is_better": lower,
                "winner": winner,
                "status": status,
                "gates_failed": gates_failed,
            }
        )
    return pairs


def build_recommendation(
    *,
    rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    primary_metric: str | None,
    minimum_robust_seeds: int = 5,
) -> dict[str, Any]:
    models_seen = sorted({str(row.get("model")) for row in rows if row.get("model")})
    gates_failed: list[str] = []
    if set(models_seen) != set(MODELS):
        status = "invalid_incomplete_grid"
        return {
            "status": status,
            "scientific_status": status,
            "winner": None,
            "winning_model": None,
            "primary_metric": primary_metric,
            "reason": "Both graph2mat and deeph runs are required.",
            "gates_passed": [],
            "gates_failed": ["missing_model"],
            "models_seen": models_seen,
            "status_values": sorted(RECOMMENDATION_STATUS_VALUES),
        }
    if not primary_metric:
        status = "invalid_incomplete_grid"
        return {
            "status": status,
            "scientific_status": status,
            "winner": None,
            "winning_model": None,
            "primary_metric": None,
            "reason": "No shared finite primary metric is available.",
            "gates_passed": [],
            "gates_failed": ["missing_primary_metric"],
            "status_values": sorted(RECOMMENDATION_STATUS_VALUES),
        }
    if metric_policy_role(primary_metric) == "diagnostic_only":
        gates_failed.append("invalid_metric_policy")
    comparable_pairs = [pair for pair in pairs if pair.get("metric") == primary_metric and pair.get("status") == "comparable"]
    if not comparable_pairs:
        pair_failures = sorted({failure for pair in pairs for failure in pair.get("gates_failed") or []})
        gates_failed.extend(pair_failures or ["invalid_incomplete_grid"])
    gates_failed.extend(failure for row in rows for failure in row_gate_failures(row, primary_metric))
    severe = [item for row in rows for item in row.get("severe_warnings") or []]
    if severe:
        gates_failed.append("severe_warnings")
    if any(row.get("fail_open_metric_outputs") or row.get("metric_fail_policy") == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY for row in rows):
        gates_failed.append("metric_fail_policy_diagnostic_only")
    if any(normalize_model(row.get("model")) == "deeph" and not deeph_adapter_equivalence_proven(row) for row in rows):
        gates_failed.append("deeph_adapter_equivalence_not_proven")
    primary_best = [row for row in best_rows if row.get("scope") == "global" and row.get("metric") == primary_metric]
    valid_seed_counts = [int(row.get("valid_seed_count") or 0) for row in primary_best]
    if valid_seed_counts and min(valid_seed_counts) < minimum_robust_seeds:
        gates_failed.append("insufficient_seeds")
    if any(row.get("seed_stability_status") == "unstable" for row in primary_best):
        gates_failed.append("unstable_across_seeds")

    best_by_model = {str(row.get("model")): row for row in primary_best}
    if set(best_by_model) != set(MODELS):
        gates_failed.append("invalid_incomplete_grid")

    winner = None
    if best_by_model:
        lower = metric_lower_is_better(primary_metric)
        ordered = sorted(
            best_by_model.values(),
            key=lambda row: (
                number(row.get("mean")) if lower else -number(row.get("mean")),
                str(row.get("model")),
            ),
        )
        winner = str(ordered[0].get("model")) if ordered else None

    gates_failed = sorted(set(gates_failed))
    hard_failures = [failure for failure in gates_failed if failure not in {"insufficient_seeds"}]
    if hard_failures:
        status = status_from_gates(hard_failures)
        scientific_status = status
        winner = None
    elif "insufficient_seeds" in gates_failed:
        status = f"exploratory_{winner}_win" if winner in MODELS else "no_robust_winner"
        scientific_status = "exploratory_only"
    else:
        status = f"robust_{winner}_win" if winner in MODELS else "no_robust_winner"
        scientific_status = "robust_comparison" if winner in MODELS else "not_scientifically_valid"

    return {
        "status": status,
        "scientific_status": scientific_status,
        "winner": winner if status.startswith(("robust_", "exploratory_")) else None,
        "winning_model": winner if status.startswith(("robust_", "exploratory_")) else None,
        "winning_config_id": (best_by_model.get(winner) or {}).get("config_id") if winner else None,
        "winning_dataset_id": (best_by_model.get(winner) or {}).get("dataset_id") if winner else None,
        "primary_metric": primary_metric,
        "primary_metric_column": source_metric(primary_metric) if primary_metric else None,
        "reason": (
            f"{winner} has the best {primary_metric}, but seed count makes this exploratory."
            if status.startswith("exploratory_")
            else f"{winner} has the best {primary_metric} and all robust gates passed."
            if status.startswith("robust_")
            else "Scientific gates prevent a robust winner."
        ),
        "limitations": gates_failed,
        "gates_passed": passed_gates(gates_failed),
        "gates_failed": gates_failed,
        "best_graph2mat": best_by_model.get("graph2mat"),
        "best_deeph": best_by_model.get("deeph"),
        "pairwise_evidence": comparable_pairs,
        "seed_stability": {row.get("model"): row.get("seed_stability_status") for row in primary_best},
        "metric_policy": metric_policy_role(primary_metric) if primary_metric else "missing",
        "comparability_status": "valid" if not hard_failures else ("diagnostic_only" if status == "diagnostic_only" else status),
        "adapter_equivalence_status": deeph_adapter_status(rows),
        "equivalence_status": deeph_equivalence_status(rows),
        "split_audit_status": split_audit_status(rows),
        "status_values": sorted(RECOMMENDATION_STATUS_VALUES),
        "warnings": severe,
    }


def seed_robustness_analysis(
    rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    primary_metric: str | None,
    *,
    minimum_robust_seeds: int = 5,
) -> dict[str, Any]:
    """Paired seed comparison with leave-one-seed-out stability."""

    best = {
        str(row.get("model")): str(row.get("config_id"))
        for row in best_rows
        if row.get("scope") == "global" and row.get("metric") == primary_metric
    }
    grouped: dict[str, dict[str, list[float]]] = {
        model: defaultdict(list) for model in MODELS
    }
    if primary_metric:
        for row in rows:
            model = str(row.get("model") or "")
            seed = row.get("seed")
            value = finite_metric(row, primary_metric)
            if (
                model in MODELS
                and str(row.get("config_id") or "") == best.get(model)
                and seed not in (None, "", "unknown")
                and value is not None
            ):
                grouped[model][str(seed)].append(value)
    seed_means = {
        model: {seed: mean(values) for seed, values in values_by_seed.items()}
        for model, values_by_seed in grouped.items()
    }
    common_seeds = sorted(set(seed_means["graph2mat"]) & set(seed_means["deeph"]))
    differences = [
        number(seed_means["graph2mat"][seed]) - number(seed_means["deeph"][seed])
        for seed in common_seeds
    ]
    difference_mean = mean(differences) if differences else None
    difference_std = stddev(differences) if differences else None
    margin = (
        1.96 * number(difference_std) / math.sqrt(len(differences))
        if len(differences) >= 2
        else None
    )
    interval = (
        [number(difference_mean) - margin, number(difference_mean) + margin]
        if difference_mean is not None and margin is not None
        else None
    )
    lower_is_better = metric_lower_is_better(primary_metric or "")

    def winner(values: list[float]) -> str | None:
        value = mean(values)
        if value is None or value == 0:
            return None
        graph2mat_better = value < 0 if lower_is_better else value > 0
        return "graph2mat" if graph2mat_better else "deeph"

    full_winner = winner(differences)
    leave_one_out = [
        {
            "left_out_seed": seed,
            "winner": winner(
                [value for index, value in enumerate(differences) if index != offset]
            ),
        }
        for offset, seed in enumerate(common_seeds)
    ]
    loo_stable = bool(leave_one_out) and all(
        row["winner"] == full_winner and row["winner"] is not None
        for row in leave_one_out
    )
    interval_excludes_zero = bool(interval) and (
        interval[1] < 0 or interval[0] > 0
    )
    blockers: list[str] = []
    if len(common_seeds) < minimum_robust_seeds:
        blockers.append("insufficient_paired_seeds")
    if not loo_stable:
        blockers.append("leave_one_seed_out_unstable")
    if not interval_excludes_zero:
        blockers.append("paired_difference_indistinguishable_from_noise")
    return {
        "status": "robust" if not blockers else "no_robust_winner",
        "metric": primary_metric,
        "selected_configs": best,
        "minimum_robust_seeds": minimum_robust_seeds,
        "common_seeds": common_seeds,
        "paired_seed_count": len(common_seeds),
        "paired_differences_graph2mat_minus_deeph": differences,
        "paired_difference_mean": difference_mean,
        "paired_difference_std": difference_std,
        "paired_difference_95pct_normal_interval": interval,
        "interval_excludes_zero": interval_excludes_zero,
        "winner": full_winner if not blockers else None,
        "leave_one_seed_out": leave_one_out,
        "leave_one_seed_out_stable": loo_stable,
        "multiplicity": {
            "method": "Holm",
            "family": "predeclared_primary_metric_model_comparison",
            "comparisons": 1,
            "adjusted_alpha": 0.05,
        },
        "paper_level_blockers": blockers,
    }


def pareto_frontier(rows: list[dict[str, Any]], primary_metric: str | None) -> list[dict[str, Any]]:
    if not primary_metric:
        return []
    candidates: list[dict[str, Any]] = []
    for row in rows:
        value = finite_metric(row, primary_metric)
        time_value = number(row.get("total_time_seconds"))
        if value is None:
            continue
        if not row_is_robust_eligible(row, primary_metric):
            continue
        candidates.append(
            {
                "model": row.get("model"),
                "dataset_id": row.get("dataset_id"),
                "config_id": row.get("config_id"),
                "metric": primary_metric,
                "metric_value": value,
                "total_time_seconds": time_value if math.isfinite(time_value) else None,
                "train_time_seconds": row.get("training_time_seconds"),
                "predict_time_seconds": row.get("prediction_time_seconds"),
                "preprocess_time_seconds": row.get("preprocess_time_seconds"),
                "gpu_hours_total": row.get("gpu_hours_total"),
                "gpu_hours_to_best_validation": row.get("gpu_hours_to_best_validation"),
                "peak_gpu_memory_mb": row.get("peak_gpu_memory_mb"),
                "samples_per_second": row.get("samples_per_second"),
                "matrix_blocks_per_second": row.get("matrix_blocks_per_second"),
                "timing_reliability_status": "available" if math.isfinite(time_value) else "timing_unavailable",
            }
        )
    frontier: list[dict[str, Any]] = []
    for candidate in candidates:
        time_a = number(candidate.get("total_time_seconds"))
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            time_b = number(other.get("total_time_seconds"))
            if not math.isfinite(time_a) or not math.isfinite(time_b):
                continue
            no_worse = other["metric_value"] <= candidate["metric_value"] and time_b <= time_a
            strictly_better = other["metric_value"] < candidate["metric_value"] or time_b < time_a
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return [
        {**row, "pareto_rank": index + 1, "pareto_status": "robust_frontier"}
        for index, row in enumerate(sorted(frontier, key=lambda item: (item["metric_value"], number(item.get("total_time_seconds")), str(item.get("config_id")))))
    ]


def rank_graph2mat_deeph_runs(
    *,
    run_root: Path,
    output_dir: Path | None = None,
    training_sweep_manifest_path: Path | None = None,
    common_metrics_manifest_path: Path | None = None,
    dataset_root: Path | None = None,
    frozen_split_manifest_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
    minimum_robust_seeds: int = 5,
) -> dict[str, Any]:
    run_root = Path(run_root)
    output_dir = Path(output_dir or run_root / "summary" / "ranking")
    training_sweep_manifest_path = training_sweep_manifest_path or run_root / "sweep" / "training_sweep_manifest.json"
    common_metrics_manifest_path = common_metrics_manifest_path or run_root / "common_metrics" / "summary" / "common_summary.json"
    rows = load_metric_rows(
        training_sweep_manifest_path=training_sweep_manifest_path if training_sweep_manifest_path.exists() else None,
        common_metrics_manifest_path=common_metrics_manifest_path if common_metrics_manifest_path.exists() else None,
        dataset_root=dataset_root,
        frozen_split_manifest_path=frozen_split_manifest_path,
        dataset_manifest_path=dataset_manifest_path,
    )
    if any(
        str(row.get("protocol_stage") or "").lower()
        in {"search", "robust_validation"}
        for row in rows
    ):
        assert_no_test_metrics_for_search(rows, stage="search")
    primary_metric = choose_primary_metric(rows)
    metric_rankings = build_metric_rankings(rows)
    best_rows = best_runs_by_model(rows, primary_metric)
    model_config_rankings = [row for row in metric_rankings if row.get("metric") == primary_metric] if primary_metric else []
    pairs = pairwise_comparisons(best_rows)
    recommendation = build_recommendation(
        rows=rows,
        best_rows=best_rows,
        pairs=pairs,
        primary_metric=primary_metric,
        minimum_robust_seeds=minimum_robust_seeds,
    )
    seed_analysis = seed_robustness_analysis(
        rows,
        best_rows,
        primary_metric,
        minimum_robust_seeds=minimum_robust_seeds,
    )
    if (
        str(recommendation.get("status") or "").startswith("robust_")
        and seed_analysis["status"] != "robust"
    ):
        blockers = set(recommendation.get("gates_failed") or [])
        blockers.update(seed_analysis["paper_level_blockers"])
        recommendation.update(
            {
                "status": "no_robust_winner",
                "scientific_status": "not_scientifically_valid",
                "winner": None,
                "winning_model": None,
                "reason": (
                    "Paired seed uncertainty or leave-one-seed-out instability "
                    "prevents a robust winner."
                ),
                "limitations": sorted(blockers),
                "gates_failed": sorted(blockers),
            }
        )
    pareto = pareto_frontier(rows, primary_metric)
    manifest = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "primary_metric": primary_metric,
        "metric_rows_count": len(rows),
        "ranking_outputs": {
            "best_runs_by_model": str(output_dir / "best_runs_by_model.json"),
            "model_config_rankings": str(output_dir / "model_config_rankings.json"),
            "metric_rankings_by_model": str(output_dir / "metric_rankings_by_model.json"),
            "pairwise_graph2mat_vs_deeph": str(output_dir / "pairwise_graph2mat_vs_deeph.json"),
            "pareto_accuracy_cost": str(output_dir / "pareto_accuracy_cost.json"),
            "recommendation": str(output_dir / "recommendation.json"),
            "best_overall": str(output_dir / "best_overall.json"),
            "seed_robustness": str(output_dir / "seed_robustness.json"),
        },
        "recommendation": recommendation,
        "best_runs_by_model": best_rows,
        "pairwise_graph2mat_vs_deeph": pairs,
        "pareto_accuracy_cost": pareto,
        "seed_robustness": seed_analysis,
    }
    write_csv(output_dir / "normalized_run_metrics.csv", rows)
    write_json(output_dir / "normalized_run_metrics.json", {"rows": rows})
    write_csv(output_dir / "best_runs_by_model.csv", best_rows)
    write_json(output_dir / "best_runs_by_model.json", {"rows": best_rows})
    write_csv(output_dir / "model_config_rankings.csv", model_config_rankings)
    write_json(output_dir / "model_config_rankings.json", {"rows": model_config_rankings})
    write_csv(output_dir / "metric_rankings_by_model.csv", metric_rankings)
    write_json(output_dir / "metric_rankings_by_model.json", {"rows": metric_rankings})
    write_csv(output_dir / "pairwise_graph2mat_vs_deeph.csv", pairs)
    write_json(output_dir / "pairwise_graph2mat_vs_deeph.json", {"rows": pairs})
    write_csv(output_dir / "pareto_accuracy_cost.csv", pareto)
    write_json(output_dir / "pareto_accuracy_cost.json", {"rows": pareto})
    write_json(output_dir / "recommendation.json", recommendation)
    write_json(output_dir / "seed_robustness.json", seed_analysis)
    write_json(output_dir / "best_overall.json", {"recommendation": recommendation, "best_runs_by_model": best_rows})
    write_json(output_dir / "ranking_summary.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--training-sweep-manifest", type=Path, default=None)
    parser.add_argument("--common-metrics-manifest", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--frozen-split-manifest", type=Path, default=None)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--minimum-robust-seeds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = rank_graph2mat_deeph_runs(
        run_root=args.run_root,
        output_dir=args.output_dir,
        training_sweep_manifest_path=args.training_sweep_manifest,
        common_metrics_manifest_path=args.common_metrics_manifest,
        dataset_root=args.dataset_root,
        frozen_split_manifest_path=args.frozen_split_manifest,
        dataset_manifest_path=args.dataset_manifest,
        minimum_robust_seeds=args.minimum_robust_seeds,
    )
    print(json.dumps(json_safe({"status": manifest["recommendation"]["status"], "output_dir": manifest["output_dir"]}), ensure_ascii=False))


if __name__ == "__main__":
    main()
