#!/usr/bin/env python3
"""Protocol-level test blindness helpers for Graph2Mat-vs-DeepH final benchmarks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


FINAL_BENCHMARK_MODES = {"final", "final_publication", "paper", "paper_ready", "publicable"}
SEARCH_STAGE = "search"
ROBUST_VALIDATION_STAGE = "robust_validation"
FINAL_TEST_STAGE = "final_test"
EXPLORATORY_STAGE = "exploratory"
VALIDATION_SPLITS = {"validation", "val"}
TEST_SPLITS = {"test"}


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "si", "sí"}
    return default


def is_final_benchmark_mode(payload: dict[str, Any] | None) -> bool:
    """Return true when payload requests the strict final/publicable benchmark protocol."""

    payload = payload or {}
    for key in ("benchmark_mode", "protocol_mode", "mode"):
        value = str(payload.get(key) or "").strip().lower()
        if value in FINAL_BENCHMARK_MODES:
            return True
    if _parse_bool(payload.get("final_benchmark"), False) or _parse_bool(payload.get("paper_ready"), False):
        return True
    protocol = payload.get("protocol")
    if isinstance(protocol, dict):
        final_policy = protocol.get("final_test_policy") if isinstance(protocol.get("final_test_policy"), dict) else {}
        if final_policy.get("policy") == "locked_until_final" and final_policy.get("locked_during_search") is True:
            return True
    return False


def protocol_stage_from_payload(payload: dict[str, Any] | None, *, default: str | None = None) -> str:
    payload = payload or {}
    raw = str(payload.get("protocol_stage") or "").strip().lower()
    if raw:
        return raw
    if default:
        return default
    return SEARCH_STAGE if is_final_benchmark_mode(payload) else EXPLORATORY_STAGE


def metric_split(row: dict[str, Any]) -> str:
    for key in ("metric_split", "evaluation_split", "selection_split", "split", "dataset_split"):
        value = str(row.get(key) or "").strip().lower()
        if value:
            return value
    stage = str(row.get("protocol_stage") or row.get("stage") or "").strip().lower()
    if stage == FINAL_TEST_STAGE:
        return "test"
    return ""


def row_contains_test_metric(row: dict[str, Any]) -> bool:
    split = metric_split(row)
    if split in TEST_SPLITS:
        return True
    if row.get("uses_test_metrics") is True:
        return True
    scope = str(row.get("metric_scope") or row.get("scope") or "").strip().lower()
    return scope in TEST_SPLITS


def assert_no_test_metrics_for_search(rows: list[dict[str, Any]], *, stage: str = SEARCH_STAGE) -> None:
    """Fail closed if search/robust-validation inputs contain test metrics."""

    normalized_stage = str(stage or SEARCH_STAGE).strip().lower()
    if normalized_stage not in {SEARCH_STAGE, ROBUST_VALIDATION_STAGE}:
        return
    offenders = [
        str(row.get("config_id") or row.get("run_id") or index)
        for index, row in enumerate(rows)
        if isinstance(row, dict) and row_contains_test_metric(row)
    ]
    if offenders:
        raise RuntimeError(
            "Test metrics are locked during "
            f"{normalized_stage}; offending rows: {', '.join(offenders[:10])}"
        )


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    for key in (metric, f"{metric}_mean"):
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and number not in (float("inf"), float("-inf")):
            return number
    return None


def select_top_k_validation_only(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    mode: str,
    k_per_model: int,
    stage: str = SEARCH_STAGE,
) -> list[dict[str, Any]]:
    """Select top-k configs per model using validation metrics only."""

    if k_per_model <= 0:
        raise RuntimeError("k_per_model must be positive.")
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"min", "max"}:
        raise RuntimeError("mode must be min or max.")
    assert_no_test_metrics_for_search(rows, stage=stage)
    validation_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and metric_split(row) in VALIDATION_SPLITS
        and _metric_value(row, metric) is not None
    ]
    if not validation_rows:
        raise RuntimeError(
            "No validation metric rows are available for top-k selection; "
            "search/top-k selection must not use test metrics."
        )
    reverse = normalized_mode == "max"
    selected: list[dict[str, Any]] = []
    models = sorted({str(row.get("model") or "") for row in validation_rows if row.get("model")})
    for model in models:
        model_rows = [row for row in validation_rows if str(row.get("model") or "") == model]
        model_rows.sort(key=lambda row: (_metric_value(row, metric), str(row.get("config_id") or "")), reverse=reverse)
        selected.extend(model_rows[:k_per_model])
    return selected


def validate_final_evaluation_inputs(
    *,
    selected_runs: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    stage: str,
    metric: str,
) -> None:
    """Validate stage-specific metric availability for strict final protocols."""

    normalized_stage = str(stage or "").strip().lower()
    if normalized_stage in {SEARCH_STAGE, ROBUST_VALIDATION_STAGE}:
        assert_no_test_metrics_for_search(metric_rows, stage=normalized_stage)
        return
    if normalized_stage != FINAL_TEST_STAGE:
        raise RuntimeError(f"Unsupported final benchmark protocol stage: {stage}")
    if not selected_runs:
        raise RuntimeError("final_test requires selected final runs.")
    test_rows = [
        row
        for row in metric_rows
        if isinstance(row, dict)
        and metric_split(row) in TEST_SPLITS
        and _metric_value(row, metric) is not None
    ]
    if not test_rows:
        raise RuntimeError("final_test requires test metrics for selected final runs.")
    selected_keys = {
        (str(row.get("model") or ""), str(row.get("config_id") or ""))
        for row in selected_runs
    }
    measured_keys = {
        (str(row.get("model") or ""), str(row.get("config_id") or ""))
        for row in test_rows
    }
    missing = sorted(selected_keys - measured_keys)
    if missing:
        raise RuntimeError(
            "final_test is missing test metrics for selected runs: "
            + ", ".join(f"{model}/{config}" for model, config in missing[:10])
        )


def search_stage_record_fields() -> dict[str, Any]:
    return {
        "protocol_stage": SEARCH_STAGE,
        "test_metrics_locked": True,
        "test_metrics_status": "locked_until_final",
        "metric_split": "validation",
        "final_test_evaluation_allowed": False,
    }


def build_search_stage_manifest(
    *,
    run_root: Path,
    summary: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = payload or {}
    manifest = {
        "schema": "graph2mat_deeph_test_blindness_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_root": str(run_root),
        "protocol_stage": SEARCH_STAGE,
        "final_benchmark_mode": is_final_benchmark_mode(payload),
        "final_test_locked": True,
        "search_may_compute_splits": ["train", "validation"],
        "search_must_not_compute_splits": ["test"],
        "top_k_selection_split": "validation",
        "final_test_stage": FINAL_TEST_STAGE,
        "training_sweep_status": summary.get("status"),
        "completed_runs": len([row for row in summary.get("runs") or [] if row.get("status") == "completed"]),
        "failed_runs": len(summary.get("failed_runs") or []),
        "selected_final_runs": [],
        "final_test_status": "pending_selection",
    }
    path = run_root / "summary" / "test_blindness_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["path"] = str(path)
    return manifest


def build_final_test_stage_manifest(
    *,
    run_root: Path,
    selected_runs: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    """Write a manifest proving final-test metrics exist only for selected final runs."""

    validate_final_evaluation_inputs(
        selected_runs=selected_runs,
        metric_rows=metric_rows,
        stage=FINAL_TEST_STAGE,
        metric=metric,
    )
    selected_keys = {
        (str(row.get("model") or ""), str(row.get("config_id") or ""))
        for row in selected_runs
    }
    final_rows = [
        row
        for row in metric_rows
        if isinstance(row, dict)
        and metric_split(row) in TEST_SPLITS
        and (str(row.get("model") or ""), str(row.get("config_id") or "")) in selected_keys
    ]
    manifest = {
        "schema": "graph2mat_deeph_test_blindness_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_root": str(run_root),
        "protocol_stage": FINAL_TEST_STAGE,
        "final_test_locked": False,
        "selection_required_before_final_test": True,
        "selected_final_runs": selected_runs,
        "final_test_metric": metric,
        "final_test_metric_rows": len(final_rows),
        "final_test_status": "completed",
    }
    path = run_root / "summary" / "final_test_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["path"] = str(path)
    return manifest
