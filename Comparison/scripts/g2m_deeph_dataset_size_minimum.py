#!/usr/bin/env python3
"""Estimate minimum dataset size from existing Graph2Mat-vs-DeepH metrics.

This is a read-only post-processing script. It does not train, predict, run
SIESTA, run Graph2Mat, run DeepH, or materialize new Hamiltonians. It consumes
already-written metric tables/JSON files and writes derived CSV/JSON/plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PRIMARY_METRIC = "h_mae_eV_mean"
DEFAULT_FIT_MODELS = "linear,quadratic,inverse,inverse_square,power_law"
ENERGY_METRICS_WITHOUT_EV = {"dos_mae_500_fermi_window"}

FORBIDDEN_COMPUTE_COMMANDS = (
    "deeph-train",
    "deeph-preprocess",
    "deeph-inference",
    "graph2mat fit",
    "graph2mat test",
    "graph2mat predict",
    "siesta",
    "gnubands",
)

METHOD_COLORS = {
    "deeph": "#d62728",
    "graph2mat": "#1f77b4",
}


def configure_csv() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    configure_csv()
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_safe(row.get(key)) for key in (fieldnames or ["status"])})


def csv_safe(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else ""
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def parse_json_field(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def finite_number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def int_number(value: Any) -> int | None:
    out = finite_number(value)
    if out is None:
        return None
    return int(round(out))


def mean(values: Iterable[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else None


def std(values: Iterable[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return statistics.stdev(clean) if len(clean) > 1 else 0.0


def model_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"g2m", "graph2mat"}:
        return "graph2mat"
    if text in {"deeph", "deep_h"}:
        return "deeph"
    return text or "unknown"


def metric_aliases(metric: str) -> list[str]:
    aliases = [metric]
    if metric.endswith("_mean"):
        aliases.append(metric[: -len("_mean")])
    else:
        aliases.append(f"{metric}_mean")
    if metric == "h_mae_eV_mean":
        aliases.extend(["h_mae_eV", "mae_union_eV"])
    if metric == "h_rmse_eV":
        aliases.append("h_rmse_eV_mean")
    return list(dict.fromkeys(aliases))


def metric_scale(metric: str) -> tuple[float, str]:
    text = metric.lower()
    if "ev2" in text:
        return 1_000_000.0, "meV^2"
    if "ev" in text or metric in ENERGY_METRICS_WITHOUT_EV:
        return 1000.0, "meV"
    return 1.0, "a.u."


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    for key in metric_aliases(metric):
        value = finite_number(row.get(key))
        if value is not None:
            return value
    for field in ("final_test_metrics", "test_metrics", "validation_metrics", "metrics"):
        payload = parse_json_field(row.get(field))
        if not isinstance(payload, dict):
            continue
        for key in metric_aliases(metric):
            value = finite_number(payload.get(key))
            if value is not None:
                return value
    if metric == "low_energy_rmse_eV":
        return finite_number(row.get("final_test_metric_value"))
    return None


def cost_value(row: dict[str, Any]) -> float | None:
    for key in ("gpu_hours_total", "gpu_hours", "gpu_hours_mean"):
        value = finite_number(row.get(key))
        if value is not None:
            return value
    telemetry = parse_json_field(row.get("telemetry"))
    if isinstance(telemetry, dict):
        for key in ("gpu_hours_total", "gpu_hours", "gpu_hours_mean"):
            value = finite_number(telemetry.get(key))
            if value is not None:
                return value
    return None


def elapsed_value(row: dict[str, Any]) -> float | None:
    for key in ("elapsed_seconds", "total_time_seconds", "wall_clock_seconds_total", "training_time_seconds"):
        value = finite_number(row.get(key))
        if value is not None:
            return value
    telemetry = parse_json_field(row.get("telemetry"))
    if isinstance(telemetry, dict):
        for key in ("elapsed_seconds", "total_time_seconds", "wall_clock_seconds_total", "training_time_seconds"):
            value = finite_number(telemetry.get(key))
            if value is not None:
                return value
    return None


def split_counts_from_dataset_root(dataset_root: Any) -> dict[str, int]:
    if not dataset_root:
        return {}
    root = Path(str(dataset_root))
    split = read_json(root / "frozen_split_manifest.json")
    if isinstance(split, dict):
        counts = split.get("split_counts")
        if isinstance(counts, dict):
            out: dict[str, int] = {}
            for key, value in counts.items():
                number = int_number(value)
                if number is not None:
                    out[str(key)] = number
            if out:
                return out
        rows = split.get("rows")
        if isinstance(rows, list):
            counts = defaultdict(int)
            for item in rows:
                if isinstance(item, dict):
                    counts[str(item.get("split") or item.get("subset") or "unknown")] += 1
            if counts:
                return dict(counts)
    return {}


def dataset_size_from_root(dataset_root: Any) -> int | None:
    if not dataset_root:
        return None
    root = Path(str(dataset_root))
    for path in (root / "artifact_validation.json", root / "benchmark_dataset_manifest.json"):
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        samples = payload.get("samples")
        if isinstance(samples, list):
            return len(samples)
        for key in ("total_snapshots", "valid_snapshots", "dataset_size"):
            value = int_number(payload.get(key))
            if value is not None:
                return value
        artifact_summary = payload.get("artifact_summary")
        if isinstance(artifact_summary, dict):
            for key in ("total_snapshots", "valid_snapshots", "dataset_size"):
                value = int_number(artifact_summary.get(key))
                if value is not None:
                    return value
    counts = split_counts_from_dataset_root(root)
    return sum(counts.values()) if counts else None


def infer_size_from_text(value: Any) -> int | None:
    if not value:
        return None
    text = str(value)
    patterns = [
        r"(?:^|[_/-])iid(\d+)(?:$|[_/-])",
        r"(?:^|[_/-])n[_-]?(\d+)(?:$|[_/-])",
        r"(?:^|[_/-])size[_-]?(\d+)(?:$|[_/-])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def row_dataset_sizes(row: dict[str, Any]) -> tuple[int | None, int | None, list[str]]:
    warnings: list[str] = []
    n_total = (
        int_number(row.get("n_total"))
        or int_number(row.get("dataset_size"))
        or int_number(row.get("total_snapshots"))
        or int_number(row.get("valid_snapshots"))
    )
    n_train = int_number(row.get("n_train")) or int_number(row.get("train_dataset_size")) or int_number(row.get("train_size"))

    dataset_root = row.get("dataset_root") or row.get("reference_dir") or row.get("reference_root")
    if n_total is None:
        n_total = dataset_size_from_root(dataset_root)
    if n_train is None:
        counts = split_counts_from_dataset_root(dataset_root)
        n_train = counts.get("train") or counts.get("training")
    if n_total is None:
        for key in ("dataset_root", "reference_dir", "metrics_root", "run_dir", "prediction_dir", "dataset_id"):
            n_total = infer_size_from_text(row.get(key))
            if n_total is not None:
                break
    if n_train is None and n_total is not None:
        warnings.append(f"missing_train_split_count_for_size_{n_total}; using n_total as n_train fallback")
        n_train = n_total
    return n_total, n_train, warnings


def load_json_metric_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict):
        rows = payload.get("metric_scaling_rows") or payload.get("rows")
    else:
        rows = payload
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def pivot_metric_scaling_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows or not any("metric_key" in row and "metric_value" in row for row in rows):
        return rows
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    identity_keys = [
        "run_id",
        "dataset_id",
        "dataset_root",
        "dataset_size",
        "n_total",
        "n_train",
        "method",
        "model",
        "config_id",
        "config_hash",
        "selected_config_id",
        "seed",
        "epochs",
        "epoch_label",
    ]
    for row in rows:
        metric_key = row.get("metric_key")
        metric = finite_number(row.get("metric_value"))
        if not metric_key or metric is None:
            continue
        key = tuple(row.get(item) for item in identity_keys)
        out = grouped.setdefault(key, {item: row.get(item) for item in identity_keys if row.get(item) not in (None, "")})
        out[str(metric_key)] = metric
        for cost_key in ("gpu_hours_total", "elapsed_seconds", "total_time_seconds", "wall_clock_seconds_total"):
            if row.get(cost_key) not in (None, ""):
                out[cost_key] = row.get(cost_key)
    return list(grouped.values())


def metric_file_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        if path.suffix == ".json":
            rows = load_json_metric_rows(path)
        else:
            rows = [dict(row) for row in read_csv(path)]
        return len(pivot_metric_scaling_rows(rows))
    except (OSError, json.JSONDecodeError, csv.Error, ValueError):
        return 0


def discover_metric_files(run_root: Path) -> list[Path]:
    preferred_groups = [
        [run_root / "summary" / "ranking" / "normalized_run_metrics.json"],
        [run_root / "summary" / "ranking" / "normalized_run_metrics.csv"],
        [run_root / "sweep" / "training_sweep_metrics.csv"],
        [run_root / "final_test" / "sweep" / "training_sweep_metrics.csv"],
    ]

    existing_preferred: list[Path] = []
    for group in preferred_groups:
        existing = [path for path in group if path.exists()]
        existing_preferred.extend(existing)
        usable = [path for path in existing if metric_file_row_count(path) > 0]
        if usable:
            return usable

    found: list[Path] = []
    found.extend(sorted(run_root.glob("*/summary/ranking/normalized_run_metrics.json")))
    found.extend(sorted(run_root.glob("*/summary/ranking/normalized_run_metrics.csv")))
    found.extend(sorted(run_root.glob("runs/*/sweep/training_sweep_metrics.csv")))

    usable_fallback = [path for path in found if metric_file_row_count(path) > 0]
    if usable_fallback:
        return usable_fallback

    return existing_preferred or found


def load_run_root_rows(run_root: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    warnings: list[str] = []
    for path in discover_metric_files(run_root):
        if path.suffix == ".json":
            loaded = load_json_metric_rows(path)
        else:
            loaded = [dict(row) for row in read_csv(path)]
        loaded = pivot_metric_scaling_rows(loaded)
        for row in loaded:
            row.setdefault("source_run_root", str(run_root))
            row.setdefault("source_metric_file", str(path))
        rows.extend(loaded)
        sources.append(str(path))
    if not rows:
        warnings.append(f"no_metric_rows_found:{run_root}")
    return rows, sources, warnings


def normalize_rows(
    raw_rows: list[dict[str, Any]],
    *,
    primary_metric: str,
    x_axis: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    scale, unit = metric_scale(primary_metric)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in raw_rows:
        value_e = metric_value(row, primary_metric)
        n_total, n_train, row_warnings = row_dataset_sizes(row)
        if x_axis == "n_train":
            warnings.extend(row_warnings)
        x_value = n_train if x_axis == "n_train" else n_total
        method = model_key(row.get("method") or row.get("model"))
        config_id = str(row.get("selected_config_id") or row.get("config_id") or row.get("config_hash") or "unknown")
        epoch_label = str(row.get("epoch_label") or (f"{row.get('epochs')} epochs" if row.get("epochs") not in (None, "") else "unknown"))
        if value_e is None:
            warnings.append(f"missing_primary_metric:{primary_metric}:{method}:{config_id}")
            continue
        if x_value is None:
            warnings.append(f"missing_dataset_size:{method}:{config_id}")
            continue
        rows.append(
            {
                "source_run_root": row.get("source_run_root"),
                "source_metric_file": row.get("source_metric_file"),
                "method": method,
                "dataset_id": row.get("dataset_id") or "",
                "dataset_root": row.get("dataset_root") or row.get("reference_dir") or "",
                "dataset_size_total": n_total,
                "dataset_size_train": n_train,
                "dataset_size_x": int(x_value),
                "x_axis": x_axis,
                "config_id": config_id,
                "config_hash": row.get("config_hash") or "",
                "seed": row.get("seed") or "unknown",
                "epochs": row.get("epochs") or "",
                "epoch_label": epoch_label,
                "primary_metric": primary_metric,
                "primary_metric_raw": value_e,
                "primary_metric_mev": value_e * scale,
                "primary_metric_unit": unit,
                "gpu_hours_total": cost_value(row),
                "elapsed_seconds": elapsed_value(row),
            }
        )
    return rows, warnings


def group_config_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["method"], row["dataset_size_x"], row["config_id"], row["epoch_label"])
        grouped[key].append(row)
    out: list[dict[str, Any]] = []
    for (method, size, config_id, epoch_label), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])):
        metric_values = [float(item["primary_metric_mev"]) for item in items if finite_number(item.get("primary_metric_mev")) is not None]
        costs = [value for item in items if (value := finite_number(item.get("gpu_hours_total"))) is not None]
        elapsed = [value for item in items if (value := finite_number(item.get("elapsed_seconds"))) is not None]
        first = items[0]
        out.append(
            {
                "method": method,
                "dataset_size_x": size,
                "dataset_size_total": first.get("dataset_size_total"),
                "dataset_size_train": first.get("dataset_size_train"),
                "config_id": config_id,
                "epoch_label": epoch_label,
                "primary_metric": first["primary_metric"],
                "primary_metric_unit": first["primary_metric_unit"],
                "primary_metric_mev_mean": mean(metric_values),
                "primary_metric_mev_std": std(metric_values),
                "gpu_hours_total_mean": mean(costs),
                "gpu_hours_total_sum": sum(costs) if costs else None,
                "elapsed_seconds_mean": mean(elapsed),
                "row_count": len(items),
                "seed_count": len({str(item.get("seed") or "") for item in items}),
                "source_run_roots": sorted({str(item.get("source_run_root") or "") for item in items if item.get("source_run_root")}),
            }
        )
    return out


def best_by_method_size(grouped_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in grouped_rows:
        value = finite_number(row.get("primary_metric_mev_mean"))
        if value is None:
            continue
        grouped[(str(row["method"]), int(row["dataset_size_x"]))].append(row)
    best: list[dict[str, Any]] = []
    for (_method, _size), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        chosen = min(
            items,
            key=lambda row: (
                finite_number(row.get("primary_metric_mev_mean")) or math.inf,
                finite_number(row.get("gpu_hours_total_mean")) or math.inf,
                str(row.get("config_id")),
            ),
        )
        out = dict(chosen)
        out["is_best_for_method_size"] = True
        best.append(out)
    return best


def mean_by_method_size(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average rows across sources for each (method, dataset_size_x)."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = finite_number(row.get("primary_metric_mev_mean"))
        if value is None:
            continue
        grouped[(str(row["method"]), int(row["dataset_size_x"]))].append(row)
    out: list[dict[str, Any]] = []
    for (method, size), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        metric_values = [
            value
            for item in items
            if (value := finite_number(item.get("primary_metric_mev_mean"))) is not None
        ]
        costs = [
            value
            for item in items
            if (value := finite_number(item.get("gpu_hours_total_mean"))) is not None
        ]
        first = items[0]
        source_roots = sorted(
            {
                str(item.get("source_run_root") or "")
                for item in items
                if item.get("source_run_root")
            }
        )
        sweep_labels = sorted(
            {
                str(item.get("sweep_label") or "")
                for item in items
                if item.get("sweep_label")
            }
        )
        out.append(
            {
                "method": method,
                "dataset_size_x": size,
                "dataset_size_total": first.get("dataset_size_total"),
                "dataset_size_train": first.get("dataset_size_train"),
                "config_id": "aggregated_mean",
                "epoch_label": f"mean of {len(items)} source(s)",
                "primary_metric": first.get("primary_metric"),
                "primary_metric_unit": first.get("primary_metric_unit"),
                "primary_metric_mev_mean": mean(metric_values),
                "primary_metric_mev_std": std(metric_values),
                "gpu_hours_total_mean": mean(costs),
                "source_count": len(items),
                "source_run_roots": source_roots,
                "sweep_labels": sweep_labels,
                "sweep_label": (
                    f"mean ({len(sweep_labels)} sweeps)"
                    if sweep_labels
                    else f"mean ({len(items)} sources)"
                ),
                "source_run_root": source_roots[0] if len(source_roots) == 1 else None,
                "is_aggregated_mean": True,
            }
        )
    return out


def n_min_abs(best_rows: list[dict[str, Any]], threshold_mev: float) -> int | None:
    candidates = [
        int(row["dataset_size_x"])
        for row in best_rows
        if (value := finite_number(row.get("primary_metric_mev_mean"))) is not None and value <= threshold_mev
    ]
    return min(candidates) if candidates else None


def n_min_rel95(best_rows: list[dict[str, Any]], relative_tolerance: float) -> int | None:
    values = [
        (int(row["dataset_size_x"]), float(row["primary_metric_mev_mean"]))
        for row in best_rows
        if finite_number(row.get("primary_metric_mev_mean")) is not None
    ]
    if not values:
        return None
    best_observed = min(value for _size, value in values)
    cutoff = best_observed * (1.0 + relative_tolerance)
    candidates = [size for size, value in values if value <= cutoff]
    return min(candidates) if candidates else None


def n_min_plateau(best_rows: list[dict[str, Any]], plateau_gain: float) -> int | None:
    values = sorted(
        [
            (int(row["dataset_size_x"]), float(row["primary_metric_mev_mean"]))
            for row in best_rows
            if finite_number(row.get("primary_metric_mev_mean")) is not None
        ]
    )
    if not values:
        return None
    final_best = min(value for _size, value in values)
    best_so_far = math.inf
    for size, value in values:
        best_so_far = min(best_so_far, value)
        if best_so_far <= 0.0:
            return size
        future_gain_fraction = max(0.0, (best_so_far - final_best) / best_so_far)
        if future_gain_fraction <= plateau_gain:
            return size
    return values[-1][0]


def n_min_cost_eff(best_rows: list[dict[str, Any]], relative_tolerance: float) -> int | None:
    values = [
        row
        for row in best_rows
        if finite_number(row.get("primary_metric_mev_mean")) is not None
        and finite_number(row.get("gpu_hours_total_mean")) is not None
    ]
    if not values:
        return None
    best_observed = min(float(row["primary_metric_mev_mean"]) for row in values)
    cutoff = best_observed * (1.0 + relative_tolerance)
    passing = [row for row in values if float(row["primary_metric_mev_mean"]) <= cutoff]
    if not passing:
        return None
    chosen = min(
        passing,
        key=lambda row: (
            finite_number(row.get("gpu_hours_total_mean")) or math.inf,
            int(row["dataset_size_x"]),
            finite_number(row.get("primary_metric_mev_mean")) or math.inf,
        ),
    )
    return int(chosen["dataset_size_x"])


def thresholds_by_method(
    best_rows: list[dict[str, Any]],
    *,
    threshold_mev: float,
    relative_tolerance: float,
    plateau_gain: float,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for method in sorted({str(row["method"]) for row in best_rows}):
        rows = sorted([row for row in best_rows if row["method"] == method], key=lambda row: int(row["dataset_size_x"]))
        values = [finite_number(row.get("primary_metric_mev_mean")) for row in rows]
        clean_values = [value for value in values if value is not None]
        out[method] = {
            "available_sizes": [int(row["dataset_size_x"]) for row in rows],
            "best_observed_mev": min(clean_values) if clean_values else None,
            "N_min_abs": n_min_abs(rows, threshold_mev),
            "N_min_rel95": n_min_rel95(rows, relative_tolerance),
            "N_min_plateau": n_min_plateau(rows, plateau_gain),
            "N_min_cost_eff": n_min_cost_eff(rows, relative_tolerance),
        }
    return out

def thresholds_by_method_from_fit(
    best_rows: list[dict[str, Any]],
    *,
    threshold_mev: float,
    relative_tolerance: float,
    plateau_gain: float,
    fit_model: str,
    moving_average_window: int = 3,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    out: dict[str, dict[str, Any]] = {}
    fit_details: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for method in sorted({str(row["method"]) for row in best_rows}):
        observed_rows = sorted(
            [row for row in best_rows if row["method"] == method],
            key=lambda row: int(row["dataset_size_x"]),
        )

        curve_rows, fit = fitted_curve_rows(
            observed_rows,
            fit_model=fit_model,
            moving_average_window=moving_average_window,
        )
        fit_details[method] = fit

        if not curve_rows:
            warnings.append(
                f"fit_thresholds_unavailable:{method}:{fit_model}:{fit.get('status')}"
            )
            continue

        values = [
            finite_number(row.get("primary_metric_mev_mean"))
            for row in curve_rows
        ]
        clean_values = [value for value in values if value is not None]

        out[method] = {
            "available_sizes": [
                int(round(float(row["dataset_size_x"])))
                for row in observed_rows
            ],
            "fit_model": fit_model,
            "fit_domain": fit.get("fit_domain") or {},
            "best_observed_mev": min(
                float(row["primary_metric_mev_mean"])
                for row in observed_rows
                if finite_number(row.get("primary_metric_mev_mean")) is not None
            ),
            "best_fit_mev": min(clean_values) if clean_values else None,
            "N_min_abs": n_min_abs(curve_rows, threshold_mev),
            "N_min_rel95": n_min_rel95(curve_rows, relative_tolerance),
            "N_min_plateau": n_min_plateau(curve_rows, plateau_gain),

            # Cost_eff necesita costes reales. No tiene sentido deducirlo
            # solo desde una curva de error si no ajustas tambien coste(N).
            "N_min_cost_eff": n_min_cost_eff(observed_rows, relative_tolerance),

            "N_min_abs_source": "fit",
            "N_min_rel95_source": "fit",
            "N_min_plateau_source": "fit",
            "N_min_cost_eff_source": "observed_cost",
        }

    return out, fit_details, warnings


    
def fit_design(model: str, n_values: list[float]) -> list[list[float]]:
    if model == "linear":
        return [[1.0, n] for n in n_values]
    if model == "quadratic":
        return [[1.0, n, n * n] for n in n_values]
    if model == "inverse":
        return [[1.0, 1.0 / n] for n in n_values]
    if model == "inverse_square":
        return [[1.0, 1.0 / (n * n)] for n in n_values]
    raise ValueError(f"Unsupported linear fit model: {model}")


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [list(row) + [float(value)] for row, value in zip(matrix, vector)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-14:
            raise ValueError("singular least-squares normal matrix")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        augmented[col] = [value / pivot_value for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            augmented[row] = [
                current - factor * pivot_current
                for current, pivot_current in zip(augmented[row], augmented[col])
            ]
    return [augmented[row][-1] for row in range(n)]


def least_squares_coefficients(design: list[list[float]], y_values: list[float]) -> list[float]:
    if not design:
        raise ValueError("empty design matrix")
    n_columns = len(design[0])
    xtx = [[0.0 for _ in range(n_columns)] for _ in range(n_columns)]
    xty = [0.0 for _ in range(n_columns)]
    for row, y_value in zip(design, y_values):
        for i in range(n_columns):
            xty[i] += row[i] * y_value
            for j in range(n_columns):
                xtx[i][j] += row[i] * row[j]
    try:
        return solve_linear_system(xtx, xty)
    except ValueError:
        # A tiny ridge keeps polynomial/near-constant diagnostic fits from
        # failing while remaining visually indistinguishable at plot scale.
        for i in range(n_columns):
            xtx[i][i] += 1e-12
        return solve_linear_system(xtx, xty)


def fit_linear_model(model: str, n_values: list[float], y_values: list[float]) -> dict[str, Any]:
    design = fit_design(model, n_values)
    coefficients = least_squares_coefficients(design, y_values)
    predicted = [sum(coef * item for coef, item in zip(coefficients, row)) for row in design]
    return fit_summary(model, n_values, y_values, predicted, coefficients)


def fit_power_law(n_values: list[float], y_values: list[float]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    # Alpha grid; for each alpha the fit is linear in E_inf and A.
    for idx in range(160):
        alpha = 0.05 + (4.0 - 0.05) * idx / 159.0
        transformed = [n ** (-alpha) for n in n_values]
        design = [[1.0, value] for value in transformed]
        coefficients = least_squares_coefficients(design, y_values)
        pred = [sum(coef * item for coef, item in zip(coefficients, row)) for row in design]
        sse = sum((y - yhat) ** 2 for y, yhat in zip(y_values, pred))
        if best is None or sse < best["sse"]:
            best = {
                "alpha": float(alpha),
                "coefficients": [float(coefficients[0]), float(coefficients[1]), float(alpha)],
                "predicted": pred,
                "sse": sse,
            }
    assert best is not None
    summary = fit_summary("power_law", n_values, y_values, best["predicted"], best["coefficients"])
    summary["formula"] = "y = E_inf + A N^-alpha"
    return summary

LOWESS_FIT_MODELS = {"lowess_logx", "lowess_logx_robust", "monotone_lowess_logx"}


def tricube_weight(u: float) -> float:
    value = min(1.0, max(0.0, abs(float(u))))
    return (1.0 - value**3) ** 3


def median(values: list[float]) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def weighted_local_linear_estimate(
    points: list[dict[str, float]],
    x0: float,
    robust_weights: list[float],
    *,
    frac: float = 0.45,
) -> float | None:
    if not points:
        return None

    distances = sorted(abs(point["tx"] - x0) for point in points)
    k = max(2, math.ceil(frac * len(points)))
    bandwidth = distances[min(k - 1, len(distances) - 1)] if distances else 1.0
    if bandwidth <= 0.0:
        bandwidth = distances[-1] if distances and distances[-1] > 0.0 else 1.0

    sw = swx = swy = swxx = swxy = 0.0

    for index, point in enumerate(points):
        dx = point["tx"] - x0
        local_weight = tricube_weight(abs(dx) / bandwidth)
        robust_weight = robust_weights[index] if index < len(robust_weights) else 1.0
        weight = local_weight * robust_weight

        if not math.isfinite(weight) or weight <= 0.0:
            continue

        sw += weight
        swx += weight * dx
        swy += weight * point["y"]
        swxx += weight * dx * dx
        swxy += weight * dx * point["y"]

    if sw <= 0.0:
        return None

    denom = sw * swxx - swx * swx
    if abs(denom) < 1e-14:
        return swy / sw

    slope = (sw * swxy - swx * swy) / denom
    intercept = (swy - slope * swx) / sw
    return intercept


def dense_integer_grid(n_values: list[float], *, max_points: int = 2500) -> list[float]:
    if not n_values:
        return []
    n_min = int(math.ceil(min(n_values)))
    n_max = int(math.floor(max(n_values)))
    if n_max < n_min:
        return [float(round(n_values[0]))]
    if n_max - n_min + 1 <= max_points:
        return [float(n) for n in range(n_min, n_max + 1)]
    return [
        float(round(n_min + (n_max - n_min) * index / (max_points - 1)))
        for index in range(max_points)
    ]


def lowess_predict_curve(
    n_values: list[float],
    y_values: list[float],
    *,
    model: str,
    grid_values: list[float] | None = None,
    frac: float = 0.45,
) -> tuple[list[float], list[float], list[float]]:
    clean = sorted(
        [
            {"n": float(n), "y": float(y), "tx": math.log(float(n))}
            for n, y in zip(n_values, y_values)
            if n > 0 and math.isfinite(float(y))
        ],
        key=lambda row: row["n"],
    )

    if len(clean) < 3:
        return [], [], []

    robust_iterations = 2 if model in {"lowess_logx_robust", "monotone_lowess_logx"} else 0
    robust_weights = [1.0 for _ in clean]

    for _iteration in range(robust_iterations):
        fitted_at_observed = [
            weighted_local_linear_estimate(clean, point["tx"], robust_weights, frac=frac)
            for point in clean
        ]
        residuals = [
            abs(point["y"] - fit)
            for point, fit in zip(clean, fitted_at_observed)
            if fit is not None and math.isfinite(fit)
        ]
        mad = median(residuals)
        if mad is None or mad <= 1e-12:
            break

        next_weights: list[float] = []
        for point, fit in zip(clean, fitted_at_observed):
            if fit is None or not math.isfinite(fit):
                next_weights.append(0.0)
                continue
            u = abs(point["y"] - fit) / (6.0 * mad)
            if u >= 1.0:
                next_weights.append(0.0)
            else:
                next_weights.append((1.0 - u * u) ** 2)
        robust_weights = next_weights

    observed_predictions = [
        weighted_local_linear_estimate(clean, point["tx"], robust_weights, frac=frac)
        for point in clean
    ]

    grid = grid_values or dense_integer_grid([point["n"] for point in clean])
    curve_y: list[float] = []
    for n in grid:
        value = weighted_local_linear_estimate(clean, math.log(float(n)), robust_weights, frac=frac)
        curve_y.append(float("nan") if value is None else float(value))

    if model == "monotone_lowess_logx":
        best = math.inf
        monotone_y: list[float] = []
        for value in curve_y:
            if math.isfinite(value):
                best = min(best, value)
                monotone_y.append(best)
            else:
                monotone_y.append(value)
        curve_y = monotone_y

    observed_pred_clean = [
        float(value)
        for value in observed_predictions
        if value is not None and math.isfinite(float(value))
    ]

    return grid, curve_y, observed_pred_clean


def fit_lowess_model(model: str, n_values: list[float], y_values: list[float]) -> dict[str, Any]:
    grid, curve_y, observed_pred = lowess_predict_curve(n_values, y_values, model=model)

    if not grid or not curve_y:
        return {
            "model": model,
            "status": "failed",
            "error": "LOWESS produced no curve points.",
            "n_points": len(n_values),
        }

    # Recalcula predicciones exactamente en los N observados para métricas de ajuste.
    _obs_grid, _obs_curve, obs_pred = lowess_predict_curve(
        n_values,
        y_values,
        model=model,
        grid_values=n_values,
    )

    if len(obs_pred) != len(y_values):
        obs_pred = [
            value
            for value in _obs_curve
            if math.isfinite(value)
        ]

    if len(obs_pred) != len(y_values):
        # Fallback seguro: usa interpolación de la curva densa más cercana.
        obs_pred = []
        for n in n_values:
            nearest_index = min(range(len(grid)), key=lambda idx: abs(grid[idx] - n))
            obs_pred.append(curve_y[nearest_index])

    summary = fit_summary(model, n_values, y_values, obs_pred, [])
    summary["status"] = "ok"
    summary["formula"] = {
        "lowess_logx": "LOWESS over log(N)",
        "lowess_logx_robust": "robust LOWESS over log(N)",
        "monotone_lowess_logx": "robust LOWESS over log(N), monotone non-increasing",
    }[model]
    summary["curve_points"] = [
        {"x": float(n), "y": float(y)}
        for n, y in zip(grid, curve_y)
        if math.isfinite(float(y))
    ]
    summary["fit_domain"] = {
        "min_n": min(n_values),
        "max_n": max(n_values),
    }
    return summary


def fit_moving_average_model(
    n_values: list[float],
    y_values: list[float],
    *,
    window_size: int,
) -> dict[str, Any]:
    clean = sorted(
        [
            (float(n), float(y))
            for n, y in zip(n_values, y_values)
            if n > 0 and math.isfinite(float(y))
        ],
        key=lambda item: item[0],
    )

    if not clean:
        return {
            "model": "moving_average",
            "status": "failed",
            "error": "No finite points for moving average.",
            "n_points": 0,
        }

    window = max(1, min(int(window_size or 3), len(clean)))
    left_span = (window - 1) // 2
    right_span = window - left_span - 1

    curve_points: list[dict[str, Any]] = []
    smoothed_values: list[float] = []

    for index, (n_value, _y_value) in enumerate(clean):
        start = max(0, index - left_span)
        end = min(len(clean), index + right_span + 1)
        local_values = [value for _n, value in clean[start:end]]
        smoothed = mean(local_values) or 0.0
        smoothed_values.append(float(smoothed))
        curve_points.append(
            {
                "x": float(n_value),
                "y": float(smoothed),
                "window_count": len(local_values),
            }
        )

    sorted_n = [item[0] for item in clean]
    sorted_y = [item[1] for item in clean]

    summary = fit_summary("moving_average", sorted_n, sorted_y, smoothed_values, [])
    summary["status"] = "ok"
    summary["formula"] = f"centered moving average over observed N, window={window}"
    summary["curve_points"] = curve_points
    summary["moving_average_window"] = window
    summary["fit_domain"] = {
        "min_n": min(sorted_n),
        "max_n": max(sorted_n),
    }
    return summary



def predict_fit(model: str, coefficients: list[float], n_values: list[float]) -> list[float]:
    if model in LOWESS_FIT_MODELS or model == "moving_average":
        raise ValueError(f"{model} stores explicit curve_points; use curve_points instead of predict_fit.")
    if model in LOWESS_FIT_MODELS:
        raise ValueError(f"{model} stores explicit curve_points; use curve_points instead of predict_fit.")
    if model == "power_law":
        e_inf, amplitude, alpha = coefficients
        return [e_inf + amplitude * (n ** (-alpha)) for n in n_values]
    predictions = []
    for design_row in fit_design(model, n_values):
        predictions.append(sum(coef * value for coef, value in zip(coefficients, design_row)))
    return predictions

def dense_n_grid(rows: list[dict[str, Any]], *, max_points: int = 2000) -> list[float]:
    values = sorted(
        {
            int(row["dataset_size_x"])
            for row in rows
            if finite_number(row.get("dataset_size_x")) is not None
            and int(row["dataset_size_x"]) > 0
        }
    )
    if not values:
        return []
    n_min = values[0]
    n_max = values[-1]
    if n_max <= n_min:
        return [float(n_min)]
    if n_max - n_min + 1 <= max_points:
        return [float(n) for n in range(n_min, n_max + 1)]
    return [
        n_min + (n_max - n_min) * index / (max_points - 1)
        for index in range(max_points)
    ]


def fitted_curve_rows(
    rows: list[dict[str, Any]],
    *,
    fit_model: str,
    moving_average_window: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean_rows = sorted(
        [
            row
            for row in rows
            if finite_number(row.get("primary_metric_mev_mean")) is not None
            and finite_number(row.get("dataset_size_x")) is not None
            and int(row["dataset_size_x"]) > 0
        ],
        key=lambda row: int(row["dataset_size_x"]),
    )

    if len(clean_rows) < required_fit_points(fit_model):
        return [], {
            "status": "skipped_insufficient_points",
            "fit_model": fit_model,
            "n_points": len(clean_rows),
        }

    n_values = [float(row["dataset_size_x"]) for row in clean_rows]
    y_values = [float(row["primary_metric_mev_mean"]) for row in clean_rows]

    if fit_model == "power_law":
        fit = fit_power_law(n_values, y_values)
    elif fit_model in LOWESS_FIT_MODELS:
        fit = fit_lowess_model(fit_model, n_values, y_values)
    elif fit_model == "moving_average":
        fit = fit_moving_average_model(
            n_values,
            y_values,
            window_size=moving_average_window,
        )
    else:
        fit = fit_linear_model(fit_model, n_values, y_values)

    if fit.get("status") not in {None, "ok"}:
        return [], fit

    if fit_model in LOWESS_FIT_MODELS or fit_model == "moving_average":
        curve_points = fit.get("curve_points") or []
        curve_rows = [
            {
                "dataset_size_x": float(point["x"]),
                "primary_metric_mev_mean": float(point["y"]),
                "method": clean_rows[0]["method"],
                "config_id": f"fit:{fit_model}",
                "epoch_label": "fit",
            }
            for point in curve_points
            if finite_number(point.get("x")) is not None
            and finite_number(point.get("y")) is not None
        ]
    else:
        grid = dense_n_grid(clean_rows)
        y_grid = predict_fit(fit_model, [float(x) for x in fit["coefficients"]], grid)
        curve_rows = [
            {
                "dataset_size_x": n_value,
                "primary_metric_mev_mean": y_value,
                "method": clean_rows[0]["method"],
                "config_id": f"fit:{fit_model}",
                "epoch_label": "fit",
            }
            for n_value, y_value in zip(grid, y_grid)
            if math.isfinite(y_value)
        ]

    fit["status"] = "ok"
    fit["fit_model"] = fit_model
    fit["fit_grid_points"] = len(curve_rows)
    fit["fit_domain"] = fit.get("fit_domain") or {
        "min_n": min(n_values),
        "max_n": max(n_values),
    }
    return curve_rows, fit



def fit_summary(
    model: str,
    n_values: list[float],
    y_values: list[float],
    predicted: list[float],
    coefficients: list[float],
) -> dict[str, Any]:
    residuals = [y - yhat for y, yhat in zip(y_values, predicted)]
    mae = mean([abs(value) for value in residuals])
    rmse = math.sqrt(mean([value * value for value in residuals]) or 0.0)
    y_mean = mean(y_values) or 0.0
    ss_tot = sum((value - y_mean) ** 2 for value in y_values)
    ss_res = sum(value * value for value in residuals)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {
        "model": model,
        "n_points": len(n_values),
        "coefficients": coefficients,
        "mae_mev": mae,
        "rmse_mev": rmse,
        "r2": r2,
        "formula": {
            "linear": "y = a + bN",
            "quadratic": "y = a + bN + cN^2",
            "inverse": "y = a + b/N",
            "inverse_square": "y = a + b/N^2",
        }.get(model, "y = E_inf + A N^-alpha"),
    }


def fit_models_for_method(
    best_rows: list[dict[str, Any]],
    fit_models: list[str],
    *,
    moving_average_window: int = 3,
) -> dict[str, dict[str, Any]]:
    rows = sorted(
        [
            row
            for row in best_rows
            if finite_number(row.get("primary_metric_mev_mean")) is not None and int(row["dataset_size_x"]) > 0
        ],
        key=lambda row: int(row["dataset_size_x"]),
    )
    n_values = [float(row["dataset_size_x"]) for row in rows]
    y_values = [float(row["primary_metric_mev_mean"]) for row in rows]
    out: dict[str, dict[str, Any]] = {}
    for model in fit_models:
        if len(n_values) < required_fit_points(model):
            out[model] = {"model": model, "status": "skipped_insufficient_points", "n_points": len(n_values)}
            continue
        try:
            if model == "power_law":
                out[model] = fit_power_law(n_values, y_values)
            elif model in LOWESS_FIT_MODELS:
                out[model] = fit_lowess_model(model, n_values, y_values)
            elif model == "moving_average":
                out[model] = fit_moving_average_model(
                    n_values,
                    y_values,
                    window_size=moving_average_window,
                )
            else:
                out[model] = fit_linear_model(model, n_values, y_values)
            out[model]["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - fail-safe postprocess.
            out[model] = {"model": model, "status": "failed", "error": str(exc), "n_points": len(n_values)}
    return out


def required_fit_points(model: str) -> int:
    return {
        "moving_average": 1,
        "quadratic": 3,
        "power_law": 3,
        "lowess_logx": 3,
        "lowess_logx_robust": 3,
        "monotone_lowess_logx": 3,
    }.get(model, 2)


def parse_fit_models(value: str) -> list[str]:
    allowed = {
    "linear",
    "quadratic",
    "inverse",
    "inverse_square",
    "power_law",
    "lowess_logx",
    "lowess_logx_robust",
    "monotone_lowess_logx",
    "moving_average",
    }
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [model for model in models if model not in allowed]
    if unknown:
        raise SystemExit(f"Unknown fit model(s): {', '.join(unknown)}")
    return models or ["linear"]

def canonical_fit_model(model: str) -> str:
    model = str(model or "").strip()
    if model == "power_law_floor":
        return "power_law"
    return model
    
def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig: Any, output_dir: Path, stem: str) -> list[str]:
    outputs: list[str] = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(str(path))
    return outputs


def plot_metric_vs_size(
    best_rows: list[dict[str, Any]],
    fits: dict[str, dict[str, dict[str, Any]]],
    thresholds: dict[str, dict[str, Any]],
    output_dir: Path,
    *,
    threshold_mev: float,
    fit_models: list[str],
    x_axis: str,
    primary_metric: str,
) -> list[str]:
    if not best_rows:
        return []
    plt = import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    all_y = [float(row["primary_metric_mev_mean"]) for row in best_rows if finite_number(row.get("primary_metric_mev_mean")) is not None]
    for method in sorted({str(row["method"]) for row in best_rows}):
        method_rows = sorted([row for row in best_rows if row["method"] == method], key=lambda row: int(row["dataset_size_x"]))
        xs = [int(row["dataset_size_x"]) for row in method_rows]
        ys = [float(row["primary_metric_mev_mean"]) for row in method_rows]
        color = METHOD_COLORS.get(method, "#555555")
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=color, label=f"{method} best observed")
        if xs and method in thresholds:
            for key, linestyle in [("N_min_abs", ":"), ("N_min_rel95", "--"), ("N_min_plateau", "-.")]:
                n_value = thresholds[method].get(key)
                if n_value is not None:
                    ax.axvline(float(n_value), color=color, linestyle=linestyle, alpha=0.28, linewidth=1.2)
        for model in fit_models:
            fit = fits.get(method, {}).get(model, {})
            if fit.get("status") != "ok":
                continue
            curve_points = fit.get("curve_points") or []
            if curve_points:
                x_grid = [float(point["x"]) for point in curve_points if finite_number(point.get("x")) is not None]
                y_grid = [float(point["y"]) for point in curve_points if finite_number(point.get("y")) is not None]
            else:
                x_grid = [min(xs) + (max(xs) - min(xs)) * idx / 200.0 for idx in range(201)] if max(xs) > min(xs) else xs
                y_grid = predict_fit(model, [float(item) for item in fit.get("coefficients") or []], [float(item) for item in x_grid])

            ax.plot(x_grid, y_grid, color=color, alpha=0.35, linewidth=1.3, label=f"{method} fit {model}")
    ax.axhline(threshold_mev, color="#111111", linestyle=":", linewidth=1.4, alpha=0.8, label=f"threshold {threshold_mev:g} meV")
    ax.set_xlabel("Training snapshots" if x_axis == "n_train" else "Total snapshots")
    ax.set_ylabel(f"{primary_metric} (meV)")
    ax.set_title("Minimum dataset size from existing Graph2Mat-vs-DeepH metrics")
    if all_y:
        ymax = max(max(all_y) * 1.12, threshold_mev * 1.2, 1.0)
        ax.set_ylim(bottom=0.0, top=ymax)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    outputs = save_figure(fig, output_dir, "dataset_size_minimum_primary_metric")
    plt.close(fig)
    return outputs


def plot_cost_efficiency(best_rows: list[dict[str, Any]], output_dir: Path, *, x_axis: str, primary_metric: str) -> list[str]:
    rows = [
        row
        for row in best_rows
        if finite_number(row.get("gpu_hours_total_mean")) is not None
        and finite_number(row.get("primary_metric_mev_mean")) is not None
    ]
    if not rows:
        return []
    plt = import_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = sorted([row for row in rows if row["method"] == method], key=lambda row: int(row["dataset_size_x"]))
        color = METHOD_COLORS.get(method, "#555555")
        sizes = [int(row["dataset_size_x"]) for row in method_rows]
        costs = [float(row["gpu_hours_total_mean"]) for row in method_rows]
        metrics = [float(row["primary_metric_mev_mean"]) for row in method_rows]
        axes[0].plot(sizes, costs, marker="o", color=color, linewidth=2.0, label=method)
        axes[1].scatter(costs, metrics, color=color, s=60, label=method)
        for row, cost, metric in zip(method_rows, costs, metrics):
            axes[1].annotate(str(row["dataset_size_x"]), (cost, metric), fontsize=8)
    axes[0].set_xlabel("Training snapshots" if x_axis == "n_train" else "Total snapshots")
    axes[0].set_ylabel("GPU-hours mean")
    axes[0].set_title("Cost vs dataset size")
    axes[1].set_xlabel("GPU-hours mean")
    axes[1].set_ylabel(f"{primary_metric} (meV)")
    axes[1].set_title("Accuracy/cost Pareto")
    for ax in axes:
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(left=0.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    outputs = save_figure(fig, output_dir, "dataset_size_minimum_cost_efficiency")
    plt.close(fig)
    return outputs


def build_report(
    *,
    output_dir: Path,
    run_roots: list[Path],
    grouped_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    thresholds: dict[str, dict[str, Any]],
    fits: dict[str, dict[str, dict[str, Any]]],
    warnings: list[str],
    primary_metric: str,
    threshold_mev: float,
    x_axis: str,
) -> str:
    lines: list[str] = []
    lines.append("# Dataset Size Minimum Analysis\n")
    lines.append("Postprocesado de solo lectura. No se ha ejecutado entrenamiento, inferencia, SIESTA, Graph2Mat ni DeepH.\n")
    lines.append("## Inputs\n")
    for root in run_roots:
        lines.append(f"- `{root}`")
    lines.append("\n## Configuracion\n")
    lines.append(f"- Primary metric: `{primary_metric}` convertido a meV")
    lines.append(f"- Threshold absoluto: `{threshold_mev:g}` meV")
    lines.append(f"- Eje x: `{x_axis}`")
    lines.append("\n## Cobertura\n")
    lines.append(f"- Grupos config agregados: {len(grouped_rows)}")
    lines.append(f"- Mejores metodo/tamano: {len(best_rows)}")
    lines.append(f"- Metodos: {', '.join(sorted({str(row['method']) for row in best_rows})) or 'ninguno'}")
    lines.append(
        f"- Tamanos: {', '.join(str(size) for size in sorted({int(row['dataset_size_x']) for row in best_rows})) or 'ninguno'}"
    )
    lines.append("\n## N_min por metodo\n")
    lines.append("| Metodo | Best observado meV | N_min_abs | N_min_rel95 | N_min_plateau | N_min_cost_eff |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method, summary in thresholds.items():
        lines.append(
            "| {method} | {best} | {absn} | {rel} | {plateau} | {cost} |".format(
                method=method,
                best=format_optional(summary.get("best_observed_mev")),
                absn=format_optional(summary.get("N_min_abs"), precision=0),
                rel=format_optional(summary.get("N_min_rel95"), precision=0),
                plateau=format_optional(summary.get("N_min_plateau"), precision=0),
                cost=format_optional(summary.get("N_min_cost_eff"), precision=0),
            )
        )
    lines.append("\n## Fits\n")
    for method, method_fits in fits.items():
        lines.append(f"\n### {method}")
        lines.append("| Modelo | Estado | RMSE meV | R2 | Coeficientes |")
        lines.append("|---|---|---:|---:|---|")
        for model, fit in method_fits.items():
            lines.append(
                "| {model} | {status} | {rmse} | {r2} | `{coeffs}` |".format(
                    model=model,
                    status=fit.get("status"),
                    rmse=format_optional(fit.get("rmse_mev")),
                    r2=format_optional(fit.get("r2")),
                    coeffs=json.dumps(fit.get("coefficients") or [], ensure_ascii=False),
                )
            )
    lines.append("\n## Warnings / blockers\n")
    if warnings:
        for warning in sorted(set(warnings)):
            lines.append(f"- {warning}")
    else:
        lines.append("- Ninguno.")
    lines.append("\n## Outputs\n")
    lines.append(f"- `{output_dir / 'dataset_size_minimum_results.csv'}`")
    lines.append(f"- `{output_dir / 'dataset_size_minimum_summary.json'}`")
    lines.append(f"- `{output_dir / 'dataset_size_minimum_report.md'}`")
    return "\n".join(lines) + "\n"


def format_optional(value: Any, precision: int = 3) -> str:
    number = finite_number(value)
    if number is None:
        return "-"
    if precision == 0:
        return str(int(round(number)))
    return f"{number:.{precision}f}"


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_roots = [Path(item).resolve() for item in args.run_root]
    fit_models = parse_fit_models(args.fit_models)
    moving_average_window = max(1, int(args.moving_average_window or 3))
    warnings: list[str] = []
    sources: list[str] = []
    grouped_rows: list[dict[str, Any]] = []
    per_root_best: list[dict[str, Any]] = []
    for root in run_roots:
        loaded, root_sources, root_warnings = load_run_root_rows(root)
        normalized_rows, normalize_warnings = normalize_rows(
            loaded,
            primary_metric=args.primary_metric,
            x_axis=args.x_axis,
        )
        root_grouped = group_config_rows(normalized_rows)
        grouped_rows.extend(root_grouped)
        root_best = best_by_method_size(root_grouped)
        root_key = str(root.resolve())
        for row in root_best:
            enriched = dict(row)
            enriched["source_run_root"] = root_key
            per_root_best.append(enriched)
        sources.extend(root_sources)
        warnings.extend(root_warnings + normalize_warnings)

    aggregated = len(run_roots) > 1
    if aggregated:
        best_rows = mean_by_method_size(per_root_best)
        warnings.append(f"aggregated_mean_across_{len(run_roots)}_run_roots")
    else:
        best_rows = per_root_best

    observed_thresholds = thresholds_by_method(
    best_rows,
    threshold_mev=float(args.threshold_mev),
    relative_tolerance=float(args.relative_tolerance),
    plateau_gain=float(args.plateau_gain),
    )

    fit_thresholds: dict[str, dict[str, Any]] = {}
    fit_threshold_details: dict[str, dict[str, Any]] = {}

    if args.n_min_source == "fit":
        fit_thresholds, fit_threshold_details, fit_threshold_warnings = thresholds_by_method_from_fit(
        best_rows,
        threshold_mev=float(args.threshold_mev),
        relative_tolerance=float(args.relative_tolerance),
        plateau_gain=float(args.plateau_gain),
        fit_model=args.n_min_fit_model,
        moving_average_window=moving_average_window,
        )
        warnings.extend(fit_threshold_warnings)

    thresholds = fit_thresholds if args.n_min_source == "fit" and fit_thresholds else observed_thresholds

    if args.n_min_source == "fit" and not fit_thresholds:
        warnings.append("fit_thresholds_empty; falling back to observed thresholds")
        thresholds = observed_thresholds


    fits = {
        method: fit_models_for_method(
            [row for row in best_rows if row["method"] == method],
            fit_models,
            moving_average_window=moving_average_window,
        )
        for method in sorted({str(row["method"]) for row in best_rows})
    }

    outputs: list[str] = []
    write_csv(output_dir / "dataset_size_minimum_results.csv", grouped_rows)
    write_csv(output_dir / "dataset_size_minimum_best_by_size.csv", best_rows)
    outputs.extend(
        [
            str(output_dir / "dataset_size_minimum_results.csv"),
            str(output_dir / "dataset_size_minimum_best_by_size.csv"),
        ]
    )
    try:
        outputs.extend(
            plot_metric_vs_size(
                best_rows,
                fits,
                thresholds,
                output_dir,
                threshold_mev=float(args.threshold_mev),
                fit_models=fit_models,
                x_axis=args.x_axis,
                primary_metric=args.primary_metric,
            )
        )
        outputs.extend(plot_cost_efficiency(best_rows, output_dir, x_axis=args.x_axis, primary_metric=args.primary_metric))
    except Exception as exc:  # noqa: BLE001 - plots are derived, tables still useful.
        warnings.append(f"plot_generation_failed:{exc}")

    report = build_report(
        output_dir=output_dir,
        run_roots=run_roots,
        grouped_rows=grouped_rows,
        best_rows=best_rows,
        thresholds=thresholds,
        fits=fits,
        warnings=warnings,
        primary_metric=args.primary_metric,
        threshold_mev=float(args.threshold_mev),
        x_axis=args.x_axis,
    )
    report_path = output_dir / "dataset_size_minimum_report.md"
    report_path.write_text(report, encoding="utf-8")
    outputs.append(str(report_path))

    summary = {
        "script": "Comparison/scripts/g2m_deeph_dataset_size_minimum.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_roots": [str(root) for root in run_roots],
        "sources": sources,
        "primary_metric": args.primary_metric,
        "primary_metric_unit": metric_scale(args.primary_metric)[1],
        "threshold_mev": float(args.threshold_mev),
        "relative_tolerance": float(args.relative_tolerance),
        "plateau_gain": float(args.plateau_gain),
        "x_axis": args.x_axis,
        "fit_models": fit_models,
        "grouped_config_rows": len(grouped_rows),
        "best_by_size_rows": len(best_rows),
        "aggregated": aggregated,
        "methods": sorted({str(row["method"]) for row in best_rows}),
        "dataset_sizes": sorted({int(row["dataset_size_x"]) for row in best_rows}),
        "thresholds": thresholds,
        "fits": fits,
        "outputs": outputs,
        "warnings": sorted(set(warnings)),
        "status": "ok" if best_rows else "no_usable_metric_rows",
        "forbidden_compute_commands": FORBIDDEN_COMPUTE_COMMANDS,
        "n_min_source": args.n_min_source,
        "n_min_fit_model": args.n_min_fit_model,
        "observed_thresholds": observed_thresholds,
        "fit_thresholds": fit_thresholds,
        "fit_threshold_details": fit_threshold_details,
        "moving_average_window": moving_average_window,
    }
    summary_path = output_dir / "dataset_size_minimum_summary.json"
    write_json(summary_path, summary)
    outputs.append(str(summary_path))
    summary["outputs"] = outputs
    write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", required=True, help="Run root to read. Can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for derived outputs.")
    parser.add_argument("--primary-metric", default=DEFAULT_PRIMARY_METRIC)
    parser.add_argument("--threshold-mev", type=float, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--plateau-gain", type=float, default=0.05)
    parser.add_argument("--x-axis", choices=["n_total", "n_train"], default="n_train")
    parser.add_argument("--fit-models", default=DEFAULT_FIT_MODELS)
    parser.add_argument(
        "--n-min-source",
        choices=["observed", "fit"],
        default="observed",
        help="Use observed points or fitted curve to compute N_min thresholds.",
    )
    parser.add_argument(
        "--n-min-fit-model",
        default="power_law",
        help="Fit model used when --n-min-source=fit.",
    )
    parser.add_argument(
    "--moving-average-window",
    type=int,
    default=3,
    help="Window size in observed points for moving_average fit model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = analyze(args)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}, indent=2))
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
