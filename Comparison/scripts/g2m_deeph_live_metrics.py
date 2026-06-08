#!/usr/bin/env python3
"""Incremental metric rows for in-progress Graph2Mat-vs-DeepH sweeps."""

from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
from typing import Any


METRIC_FAIL_POLICY_FAIL_CLOSED = "fail_closed"
LIVE_METRIC_SOURCE = "live_training_sweep_metrics"

MATRIX_METRICS = {
    "h_mae_eV": "h_mae_eV_mean",
    "h_rmse_eV": "h_rmse_eV_mean",
    "h_mse_eV2": "h_mse_eV2_mean",
    "relative_frobenius": "relative_frobenius_mean",
    "hermiticity_pred": "hermiticity_pred_mean",
}
SPECTRAL_METRICS = {
    "global_rmse_eV": "global_rmse_eV_mean",
    "low_energy_rmse_eV": "low_energy_rmse_eV_mean",
    "fermi_window_rmse_eV": "fermi_window_rmse_eV_mean",
    "frontier_window_rmse_eV": "frontier_window_rmse_eV_mean",
}
DOS_METRICS = {
    "dos_mae_500_fermi_window": "dos_mae_500_fermi_window_mean",
    "dos_wasserstein_eV": "dos_wasserstein_eV_mean",
}


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> float | None:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return None
    return sum(values) / len(values)


def parse_maybe_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value


def load_training_records(run_root: Path) -> list[dict[str, Any]]:
    manifest = read_json(run_root / "sweep" / "training_sweep_manifest.json")
    runs = manifest.get("runs")
    if isinstance(runs, list):
        return [dict(run) for run in runs if isinstance(run, dict)]

    csv_path = run_root / "sweep" / "training_sweep_metrics.csv"
    if not csv_path.exists():
        return []
    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = []
            for row in csv.DictReader(handle):
                parsed = {key: parse_maybe_literal(value) for key, value in row.items()}
                rows.append(parsed)
            return rows
    except (OSError, csv.Error):
        return []


def dataset_size_from_root(dataset_root: str | Path | None) -> int | None:
    if not dataset_root:
        return None
    root = Path(str(dataset_root))
    for path in (root / "benchmark_dataset_manifest.json", root / "artifact_validation.json"):
        payload = read_json(path)
        if not payload:
            continue
        samples = payload.get("samples")
        if isinstance(samples, list):
            return len(samples)
        total = (
            payload.get("total_snapshots")
            or payload.get("valid_snapshots")
            or (payload.get("artifact_summary") or {}).get("total_snapshots")
        )
        try:
            if total is not None:
                return int(total)
        except (TypeError, ValueError):
            pass
    split = read_json(root / "frozen_split_manifest.json")
    rows = split.get("rows")
    if isinstance(rows, list):
        return len(rows)
    counts = split.get("split_counts")
    if isinstance(counts, dict):
        try:
            return sum(int(value) for value in counts.values())
        except (TypeError, ValueError):
            return None
    return None


def _training_record_epochs(record: dict[str, Any]) -> Any:
    overrides = record.get("overrides") if isinstance(record.get("overrides"), dict) else {}
    common = record.get("common") if isinstance(record.get("common"), dict) else {}
    if record.get("model") == "graph2mat":
        return overrides.get("max_epochs") or common.get("epochs")
    return overrides.get("epochs") or common.get("epochs")


def _training_record_epoch_label(record: dict[str, Any]) -> str:
    epochs = _training_record_epochs(record)
    return f"{epochs} epochs" if epochs not in (None, "") else ""


def completed_metric_record(record: dict[str, Any]) -> bool:
    if str(record.get("status") or "") != "completed":
        return False
    metrics_run = record.get("metrics_run") if isinstance(record.get("metrics_run"), dict) else {}
    try:
        if int(metrics_run.get("returncode")) == 0:
            return True
    except (TypeError, ValueError):
        pass
    paths = _metric_paths(record)
    return paths["manifest"].exists() and any(
        paths[key].exists() for key in ("matrix", "spectral", "dos")
    )


def _metric_paths(record: dict[str, Any]) -> dict[str, Path]:
    run_root = Path(str(record.get("run_root") or ""))
    model = str(record.get("model") or "")
    if model == "graph2mat":
        root = run_root / "metrics" / "graph2mat" / "eval_input" / "metrics"
    elif model == "deeph":
        root = run_root / "metrics" / "deeph" / "eval" / "metrics"
    else:
        root = run_root / "metrics"
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "matrix": root / "kpoint_matrix_metrics.csv",
        "spectral": root / "kpoint_spectral_metrics.csv",
        "dos": root / "kpoint_dos_metrics.csv",
    }


def _summary_metrics(manifest: dict[str, Any]) -> dict[str, float]:
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    sections = {
        "kpoint_matrix": MATRIX_METRICS,
        "kpoint_spectral": SPECTRAL_METRICS,
        "kpoint_dos": DOS_METRICS,
        "dos": DOS_METRICS,
        "spectral": SPECTRAL_METRICS,
    }
    result: dict[str, float] = {}
    for section_name, mapping in sections.items():
        section = summary.get(section_name)
        if not isinstance(section, dict):
            continue
        for source_key, output_key in mapping.items():
            stats = section.get(source_key)
            if isinstance(stats, dict):
                value = finite_number(stats.get("mean"))
            else:
                value = finite_number(stats)
            if value is not None:
                result[output_key] = value
    return result


def _csv_metric_means(path: Path, mapping: dict[str, str], *, prefer_weighted_sample: bool = False) -> dict[str, float]:
    if not path.exists():
        return {}
    rows: list[dict[str, str]] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            all_rows = [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error):
        return {}
    if prefer_weighted_sample and any(row.get("row_type") == "weighted_sample" for row in all_rows):
        rows = [row for row in all_rows if row.get("row_type") == "weighted_sample"]
    else:
        rows = all_rows
    result: dict[str, float] = {}
    for source_key, output_key in mapping.items():
        value = mean([number for row in rows if (number := finite_number(row.get(source_key))) is not None])
        if value is not None:
            result[output_key] = value
    return result


def _record_diagnostic_status(record: dict[str, Any], manifest: dict[str, Any]) -> tuple[str, bool]:
    metric_fail_policy = str(record.get("metric_fail_policy") or METRIC_FAIL_POLICY_FAIL_CLOSED)
    fail_open = bool(record.get("fail_open_metric_outputs")) or metric_fail_policy != METRIC_FAIL_POLICY_FAIL_CLOSED
    if fail_open:
        return "diagnostic_only", True
    model = str(record.get("model") or "")
    if model == "deeph":
        adapter = manifest.get("prediction_adapter") if isinstance(manifest.get("prediction_adapter"), dict) else {}
        adapter_status = str(
            adapter.get("adapter_equivalence_status")
            or manifest.get("adapter_equivalence_status")
            or "invalid_orbital_order_unknown"
        )
        diagnostic = adapter_status != "proven_raw_global_hamiltonian_equivalent"
        return ("diagnostic_only" if diagnostic else "valid", diagnostic)
    return "valid", False


def metrics_for_record(record: dict[str, Any]) -> dict[str, float]:
    paths = _metric_paths(record)
    manifest = read_json(paths["manifest"])
    metrics = _summary_metrics(manifest)
    csv_metrics = {}
    csv_metrics.update(_csv_metric_means(paths["matrix"], MATRIX_METRICS, prefer_weighted_sample=True))
    csv_metrics.update(_csv_metric_means(paths["spectral"], SPECTRAL_METRICS))
    csv_metrics.update(_csv_metric_means(paths["dos"], DOS_METRICS))
    metrics.update(csv_metrics)
    return metrics


def live_metric_scaling_rows(run_root: Path | str) -> list[dict[str, Any]]:
    root = Path(str(run_root))
    rows: list[dict[str, Any]] = []
    for record in load_training_records(root):
        if not completed_metric_record(record):
            continue
        metrics = metrics_for_record(record)
        if not metrics:
            continue
        manifest = read_json(_metric_paths(record)["manifest"])
        dataset_root = str(record.get("dataset_root") or "")
        dataset_size = dataset_size_from_root(dataset_root)
        if dataset_size is None or dataset_size <= 0:
            continue
        scientific_status, diagnostic_only = _record_diagnostic_status(record, manifest)
        common = record.get("common") if isinstance(record.get("common"), dict) else {}
        for metric_key, metric_value in sorted(metrics.items()):
            rows.append(
                {
                    "run_id": root.name,
                    "dataset_id": str(record.get("dataset_id") or ""),
                    "dataset_root": dataset_root,
                    "dataset_size": int(dataset_size),
                    "method": str(record.get("model") or ""),
                    "config_id": str(record.get("config_id") or ""),
                    "config_hash": str(record.get("config_hash") or ""),
                    "seed": common.get("seed"),
                    "epochs": _training_record_epochs(record),
                    "epoch_label": _training_record_epoch_label(record),
                    "metric_key": metric_key,
                    "metric_value": metric_value,
                    "metric_fail_policy": str(record.get("metric_fail_policy") or METRIC_FAIL_POLICY_FAIL_CLOSED),
                    "scientific_status": scientific_status,
                    "diagnostic_only": diagnostic_only,
                    "source": LIVE_METRIC_SOURCE,
                }
            )
    return dedupe_metric_rows(rows)


def dedupe_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    seen_equivalent_values: set[tuple[str, ...]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        dataset_identity = str(row.get("dataset_root") or row.get("dataset_id") or "")
        dataset_size = str(row.get("dataset_size") or "")
        metric_value = finite_number(row.get("metric_value"))
        key = (
            str(row.get("run_id") or ""),
            dataset_identity,
            dataset_size,
            str(row.get("method") or ""),
            str(row.get("config_id") or ""),
            str(row.get("seed") or ""),
            str(row.get("epoch_label") or ""),
            str(row.get("metric_key") or ""),
        )
        value_key = (
            str(row.get("run_id") or ""),
            dataset_identity,
            dataset_size,
            str(row.get("method") or ""),
            str(row.get("config_id") or ""),
            str(row.get("epoch_label") or ""),
            str(row.get("metric_key") or ""),
            f"{metric_value:.14g}" if metric_value is not None else "",
        )
        if key in seen or value_key in seen_equivalent_values:
            continue
        seen.add(key)
        seen_equivalent_values.add(value_key)
        unique.append(row)
    return unique


def live_metrics_payload(run_root: Path | str) -> dict[str, Any]:
    root = Path(str(run_root))
    rows = live_metric_scaling_rows(root)
    return {
        "schema": "graph2mat_deeph_live_metrics_v1",
        "run_id": root.name,
        "run_root": str(root),
        "metric_scaling_rows": rows,
        "live_metric_rows": len(rows),
        "available": bool(rows),
        "source": LIVE_METRIC_SOURCE,
    }
