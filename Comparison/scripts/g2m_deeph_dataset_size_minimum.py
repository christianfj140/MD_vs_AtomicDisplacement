#!/usr/bin/env python3
"""Estimate minimum dataset size from existing Graph2Mat-vs-DeepH metrics.

This is a read-only post-processing script. It does not train, predict, run
SIESTA, run Graph2Mat, run DeepH, or materialize new Hamiltonians. It consumes
already-written metric tables/JSON files and writes derived CSV/JSON/plots.

Nominal N vs effective N (N_eff):
    N_min is computed on nominal train counts (N_train) from manifests or metric
    rows. MD trajectory snapshots can be temporally autocorrelated, so those
    counts overstate independent samples. When temporal metadata and a cheap
    per-snapshot scalar series are available, this script also reports a
    diagnostic N_eff ≈ N / statistical_inefficiency with
    statistical_inefficiency = 1 + 2 Σ ρ(k) (positive lags only), without changing
    the main N_min fits.
    Interpret N_min cautiously when N_eff ≪ N or when autocorrelation diagnostics
    are unavailable. See Comparison/scripts/DATASET_SIZE_MINIMUM.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PRIMARY_METRIC = "h_mae_eV_mean"
DEFAULT_FIT_MODELS = "linear,quadratic,inverse,inverse_square,power_law_floor"
CANONICAL_POWER_LAW_MODEL = "power_law_floor"
POWER_LAW_LEGACY_ALIASES = frozenset({"power_law"})
UNCONSTRAINED_FIT_MODELS = {"linear", "quadratic", "inverse", "inverse_square"}
DIAGNOSTIC_ONLY_FIT_MODELS = {
    *UNCONSTRAINED_FIT_MODELS,
    "moving_average",
    "lowess_logx",
    "lowess_logx_robust",
    "monotone_lowess_logx",
    "cumulative_best",
    "none",
}
CURVE_POINT_FIT_MODELS = {
    "lowess_logx",
    "lowess_logx_robust",
    "monotone_lowess_logx",
    "moving_average",
    "cumulative_best",
}
NONNEG_PREDICTION_TOL = 1e-9
DEFAULT_AGGREGATION_MODE = "mean_replicates"
AGGREGATION_MODES = (
    "mean_replicates",
    "best_config",
    "mean_seeds_per_config",
    "best_config_mean",
)
BASE_CONFIG_SEED_SUFFIX = re.compile(r"-seed\d+$", re.IGNORECASE)
EXPLICIT_BASE_CONFIG_ID_FIELDS = (
    "base_config_id",
    "config_family_id",
    "parent_config_id",
)
BASE_CONFIG_ID_FIELDS = (*EXPLICIT_BASE_CONFIG_ID_FIELDS, "selected_config_id")
DEFAULT_BOOTSTRAP_SEED = 12345
DEFAULT_CI_LEVEL = 0.95
N_MIN_REL_TOL_KEY = "N_min_rel_tol"
LEGACY_N_MIN_REL95_KEY = "N_min_rel95"
LEGACY_THRESHOLD_ALIASES = {LEGACY_N_MIN_REL95_KEY: N_MIN_REL_TOL_KEY}
BOOTSTRAP_N_MIN_CRITERIA = ("N_min_abs", N_MIN_REL_TOL_KEY, "N_min_plateau")
MIN_BOOTSTRAP_SUCCESS_FOR_CI = 2
REPLICATE_BOOTSTRAP_LABEL = "replicate resampling CI"
ENERGY_METRICS_WITHOUT_EV = {"dos_mae_500_fermi_window"}
_REFERENCED_METRIC_CACHE: dict[tuple[str, str], float | None] = {}

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


def sem(values: Iterable[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    if len(clean) == 1:
        return 0.0
    return statistics.stdev(clean) / math.sqrt(len(clean))


def row_primary_metric_mev(row: dict[str, Any]) -> float | None:
    return finite_number(row.get("primary_metric_mev_mean")) or finite_number(row.get("primary_metric_mev"))


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
        value = finite_number(row.get("final_test_metric_value"))
        if value is not None:
            return value
    return referenced_metric_value(row, metric)


def metric_summary_value(payload: Any, metric: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    for group in ("kpoint_matrix", "matrix", "sparse", "kpoint_spectral", "spectral", "kpoint_dos", "dos"):
        group_payload = summary.get(group)
        if not isinstance(group_payload, dict):
            continue
        for key in metric_aliases(metric):
            value_payload = group_payload.get(key)
            if isinstance(value_payload, dict):
                value = finite_number(value_payload.get("mean"))
            else:
                value = finite_number(value_payload)
            if value is not None:
                return value
    return None


def mean_metric_from_csv(path: Path, metric: str) -> float | None:
    rows = read_csv(path)
    values: list[float] = []
    for row in rows:
        for key in metric_aliases(metric):
            value = finite_number(row.get(key))
            if value is not None:
                values.append(value)
                break
    return mean(values)


def referenced_metric_value(row: dict[str, Any], metric: str) -> float | None:
    raw_path = row.get("validation_metrics_path") or row.get("metrics_path") or row.get("metric_manifest_path")
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    cache_key = (str(path), metric)
    if cache_key in _REFERENCED_METRIC_CACHE:
        return _REFERENCED_METRIC_CACHE[cache_key]

    value: float | None = None
    payload = read_json(path)
    if isinstance(payload, dict):
        value = metric_summary_value(payload, metric)

    metrics_dir = path.parent if path.name == "manifest.json" else path
    if value is None and metrics_dir.exists():
        csv_candidates = (
            metrics_dir / "kpoint_matrix_metrics.csv",
            metrics_dir / "matrix_metrics.csv",
            metrics_dir / "sparse_metrics.csv",
            metrics_dir / "kpoint_spectral_metrics.csv",
            metrics_dir / "spectral_metrics.csv",
            metrics_dir / "kpoint_dos_metrics.csv",
            metrics_dir / "dos_metrics.csv",
        )
        for candidate in csv_candidates:
            if not candidate.exists():
                continue
            value = mean_metric_from_csv(candidate, metric)
            if value is not None:
                break

    _REFERENCED_METRIC_CACHE[cache_key] = value
    return value


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
        "base_config_id",
        "config_family_id",
        "parent_config_id",
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
                "base_config_id": row.get("base_config_id") or "",
                "config_family_id": row.get("config_family_id") or "",
                "parent_config_id": row.get("parent_config_id") or "",
                "selected_config_id": row.get("selected_config_id") or "",
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


def extract_base_config_id(row_or_config_id: Any) -> str:
    if isinstance(row_or_config_id, dict):
        for key in EXPLICIT_BASE_CONFIG_ID_FIELDS:
            value = str(row_or_config_id.get(key) or "").strip()
            if value:
                return value
        config_id = row_or_config_id.get("selected_config_id") or row_or_config_id.get("config_id")
    else:
        config_id = row_or_config_id

    text = str(config_id or "").strip() or "unknown"
    stripped = BASE_CONFIG_SEED_SUFFIX.sub("", text)
    return stripped or text


def resolve_aggregation_mode(value: str | None, *, run_root_count: int) -> str:
    """Pick aggregation mode; default preserves single-root best_config behavior."""
    if value is None or not str(value).strip():
        return "mean_replicates" if run_root_count > 1 else "best_config"
    mode = str(value).strip()
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"Unknown aggregation_mode: {mode}")
    return mode


def parse_aggregation_mode(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    mode = str(value).strip()
    if mode not in AGGREGATION_MODES:
        raise SystemExit(f"Unknown aggregation_mode: {mode}")
    return mode


def aggregate_rows_mean_replicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average replicate rows for each (method, dataset_size_x)."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metric = row_primary_metric_mev(row)
        size = int_number(row.get("dataset_size_x"))
        if metric is None or size is None:
            continue
        grouped[(str(row["method"]), int(size))].append(row)

    out: list[dict[str, Any]] = []
    for (method, size), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        metric_values = [
            float(metric)
            for item in items
            if (metric := row_primary_metric_mev(item)) is not None
        ]
        costs = [
            value
            for item in items
            if (value := finite_number(item.get("gpu_hours_total_mean") or item.get("gpu_hours_total"))) is not None
        ]
        elapsed = [
            value
            for item in items
            if (value := finite_number(item.get("elapsed_seconds_mean") or item.get("elapsed_seconds"))) is not None
        ]
        first = items[0]
        config_ids = sorted({str(item.get("config_id") or "") for item in items if item.get("config_id")})
        seeds = sorted({str(item.get("seed") or "") for item in items if item.get("seed") not in (None, "", "unknown")})
        source_roots = sorted(
            {str(item.get("source_run_root") or "") for item in items if item.get("source_run_root")}
        )
        out.append(
            {
                "method": method,
                "dataset_size_x": size,
                "dataset_size_total": first.get("dataset_size_total"),
                "dataset_size_train": first.get("dataset_size_train"),
                "config_id": "aggregated_mean",
                "epoch_label": f"mean of {len(items)} replicate(s)",
                "primary_metric": first.get("primary_metric"),
                "primary_metric_unit": first.get("primary_metric_unit"),
                "primary_metric_mev_mean": mean(metric_values),
                "primary_metric_mev_std": std(metric_values),
                "primary_metric_mev_sem": sem(metric_values),
                "replicate_count": len(items),
                "y_min": min(metric_values) if metric_values else None,
                "y_max": max(metric_values) if metric_values else None,
                "config_ids": config_ids,
                "seeds": seeds,
                "source_run_roots": source_roots,
                "gpu_hours_total_mean": mean(costs),
                "elapsed_seconds_mean": mean(elapsed),
                "is_aggregated_mean": True,
                "aggregation_mode": "mean_replicates",
            }
        )
    return out


def aggregate_rows_mean_seeds_per_config(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average seed replicates for each (method, dataset_size_x, base_config_id)."""
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metric = row_primary_metric_mev(row)
        size = int_number(row.get("dataset_size_x"))
        if metric is None or size is None:
            continue
        base_config_id = extract_base_config_id(row)
        grouped[(str(row["method"]), int(size), base_config_id)].append(row)

    out: list[dict[str, Any]] = []
    for (method, size, base_config_id), items in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        metric_values = [
            float(metric)
            for item in items
            if (metric := row_primary_metric_mev(item)) is not None
        ]
        costs = [
            value
            for item in items
            if (value := finite_number(item.get("gpu_hours_total_mean") or item.get("gpu_hours_total"))) is not None
        ]
        elapsed = [
            value
            for item in items
            if (value := finite_number(item.get("elapsed_seconds_mean") or item.get("elapsed_seconds"))) is not None
        ]
        first = items[0]
        config_ids = sorted({str(item.get("config_id") or "") for item in items if item.get("config_id")})
        seeds = sorted({str(item.get("seed") or "") for item in items if item.get("seed") not in (None, "", "unknown")})
        source_roots = sorted(
            {str(item.get("source_run_root") or "") for item in items if item.get("source_run_root")}
        )
        out.append(
            {
                "method": method,
                "dataset_size_x": size,
                "dataset_size_total": first.get("dataset_size_total"),
                "dataset_size_train": first.get("dataset_size_train"),
                "base_config_id": base_config_id,
                "config_id": base_config_id,
                "epoch_label": first.get("epoch_label") or f"mean of {len(items)} seed(s)",
                "primary_metric": first.get("primary_metric"),
                "primary_metric_unit": first.get("primary_metric_unit"),
                "primary_metric_mev_mean": mean(metric_values),
                "primary_metric_mev_std": std(metric_values),
                "primary_metric_mev_sem": sem(metric_values),
                "seed_count": len(seeds) if seeds else len(items),
                "seeds": seeds,
                "config_ids": config_ids,
                "replicate_count": len(items),
                "y_min": min(metric_values) if metric_values else None,
                "y_max": max(metric_values) if metric_values else None,
                "source_run_roots": source_roots,
                "gpu_hours_total_mean": mean(costs),
                "elapsed_seconds_mean": mean(elapsed),
                "is_aggregated_mean": True,
                "aggregation_mode": "mean_seeds_per_config",
                "selection_basis": "mean_over_seeds",
            }
        )
    return out


def aggregate_rows_best_config_mean(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the lowest seed-mean config for each (method, dataset_size_x)."""
    seed_means = aggregate_rows_mean_seeds_per_config(rows)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_means:
        grouped[(str(row["method"]), int(row["dataset_size_x"]))].append(row)

    best: list[dict[str, Any]] = []
    for (_method, _size), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        chosen = min(
            items,
            key=lambda row: (
                finite_number(row.get("primary_metric_mev_mean")) or math.inf,
                finite_number(row.get("gpu_hours_total_mean")) or math.inf,
                str(row.get("base_config_id") or row.get("config_id")),
            ),
        )
        out = dict(chosen)
        out["is_best_for_method_size"] = True
        out["aggregation_mode"] = "best_config_mean"
        out["selection_basis"] = "mean_over_seeds"
        out["config_id"] = out.get("base_config_id") or out.get("config_id")
        best.append(out)
    return best


def analysis_rows_for_aggregation_mode(
    normalized_rows: list[dict[str, Any]],
    grouped_rows: list[dict[str, Any]],
    *,
    aggregation_mode: str,
) -> list[dict[str, Any]]:
    if aggregation_mode == "best_config":
        return best_by_method_size(grouped_rows)
    return analysis_rows_from_normalized(normalized_rows, aggregation_mode=aggregation_mode)


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
        out["aggregation_mode"] = "best_config"
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


def n_min_rel_tol(best_rows: list[dict[str, Any]], relative_tolerance: float) -> int | None:
    """First N within relative tolerance of the best observed/fitted value."""
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


def n_min_rel95(best_rows: list[dict[str, Any]], relative_tolerance: float) -> int | None:
    """Deprecated alias for n_min_rel_tol; not a 95% confidence quantity."""
    return n_min_rel_tol(best_rows, relative_tolerance)


def with_legacy_threshold_aliases(thresholds: dict[str, Any]) -> dict[str, Any]:
    out = dict(thresholds)
    if N_MIN_REL_TOL_KEY in out:
        out[LEGACY_N_MIN_REL95_KEY] = out[N_MIN_REL_TOL_KEY]
        out[f"{LEGACY_N_MIN_REL95_KEY}_deprecated_alias_for"] = N_MIN_REL_TOL_KEY
    return out


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
        out[method] = with_legacy_threshold_aliases({
            "available_sizes": [int(row["dataset_size_x"]) for row in rows],
            "best_observed_mev": min(clean_values) if clean_values else None,
            "N_min_abs": n_min_abs(rows, threshold_mev),
            N_MIN_REL_TOL_KEY: n_min_rel_tol(rows, relative_tolerance),
            "N_min_plateau": n_min_plateau(rows, plateau_gain),
            "N_min_cost_eff": n_min_cost_eff(rows, relative_tolerance),
        })
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

        canonical_model = canonical_fit_model(fit_model)
        if canonical_model == "none":
            fit_details[method] = no_fit_summary(len(observed_rows))
            observed = thresholds_by_method(
                observed_rows,
                threshold_mev=threshold_mev,
                relative_tolerance=relative_tolerance,
                plateau_gain=plateau_gain,
            ).get(method)
            if observed:
                out[method] = {
                    **observed,
                    "fit_model": "none",
                    "requested_fit_model": fit_model,
                    "canonical_fit_model": "none",
                    "fit_domain": {},
                    "best_fit_mev": None,
                    "N_min_abs_source": "observed_no_fit",
                    f"{N_MIN_REL_TOL_KEY}_source": "observed_no_fit",
                    f"{LEGACY_N_MIN_REL95_KEY}_source": "deprecated_alias",
                    "N_min_plateau_source": "observed_no_fit",
                    "N_min_cost_eff_source": "observed_cost",
                }
            continue

        curve_rows, fit = fitted_curve_rows(
            observed_rows,
            fit_model=canonical_model,
            moving_average_window=moving_average_window,
        )
        fit_details[method] = fit

        if not curve_rows:
            warnings.append(
                f"fit_thresholds_unavailable:{method}:{canonical_model}:{fit.get('status')}"
            )
            if fit.get("status") == "invalid_negative_predictions":
                warnings.append(f"fit_negative_predictions_observed_fallback:{method}:{canonical_model}")
                observed = thresholds_by_method(
                    observed_rows,
                    threshold_mev=threshold_mev,
                    relative_tolerance=relative_tolerance,
                    plateau_gain=plateau_gain,
                ).get(method)
                if observed:
                    out[method] = {
                        **observed,
                        "fit_model": canonical_model,
                        "requested_fit_model": fit_model,
                        "canonical_fit_model": canonical_model,
                        "fit_domain": fit.get("fit_domain") or {},
                        "best_fit_mev": None,
                        "fit_invalid_for_n_min_thresholding": True,
                        "fit_invalid_reason": "negative_predictions_inside_fit_domain",
                        "N_min_abs_source": "observed_invalid_fit",
                        f"{N_MIN_REL_TOL_KEY}_source": "observed_invalid_fit",
                        f"{LEGACY_N_MIN_REL95_KEY}_source": "deprecated_alias",
                        "N_min_plateau_source": "observed_invalid_fit",
                        "N_min_cost_eff_source": "observed_cost",
                    }
            continue

        values = [
            finite_number(row.get("primary_metric_mev_mean"))
            for row in curve_rows
        ]
        clean_values = [value for value in values if value is not None]

        out[method] = with_legacy_threshold_aliases({
            "available_sizes": [
                int(round(float(row["dataset_size_x"])))
                for row in observed_rows
            ],
            "fit_model": canonical_model,
            "requested_fit_model": fit_model,
            "canonical_fit_model": canonical_model,
            "fit_domain": fit.get("fit_domain") or {},
            "best_observed_mev": min(
                float(row["primary_metric_mev_mean"])
                for row in observed_rows
                if finite_number(row.get("primary_metric_mev_mean")) is not None
            ),
            "best_fit_mev": min(clean_values) if clean_values else None,
            "N_min_abs": n_min_abs(curve_rows, threshold_mev),
            N_MIN_REL_TOL_KEY: n_min_rel_tol(curve_rows, relative_tolerance),
            "N_min_plateau": n_min_plateau(curve_rows, plateau_gain),

            # Cost_eff necesita costes reales. No tiene sentido deducirlo
            # solo desde una curva de error si no ajustas tambien coste(N).
            "N_min_cost_eff": n_min_cost_eff(observed_rows, relative_tolerance),

            "N_min_abs_source": "fit",
            f"{N_MIN_REL_TOL_KEY}_source": "fit",
            f"{LEGACY_N_MIN_REL95_KEY}_source": "deprecated_alias",
            "N_min_plateau_source": "fit",
            "N_min_cost_eff_source": "observed_cost",
        })

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


def fit_policy_metadata(model: str, *, status: str = "ok", n_points: int | None = None) -> dict[str, Any]:
    canonical = canonical_fit_model(model)
    enough_points = n_points is None or n_points >= required_fit_points(canonical)
    paper_candidate = canonical == CANONICAL_POWER_LAW_MODEL and status == "ok" and enough_points
    if canonical == CANONICAL_POWER_LAW_MODEL:
        classification = "paper_candidate" if paper_candidate else "diagnostic_only"
        reason = "constrained nonnegative power law + floor" if paper_candidate else "power_law_floor_not_valid_for_paper_candidate"
    else:
        classification = "diagnostic_only"
        reason = "diagnostic_fit_model_not_mechanistic_primary_law"
    return {
        "paper_candidate": paper_candidate,
        "diagnostic_only": not paper_candidate,
        "fit_policy": classification,
        "fit_policy_reason": reason,
    }


def canonical_fit_model(model: str) -> str:
    model = str(model or "").strip()
    if model in POWER_LAW_LEGACY_ALIASES or model == CANONICAL_POWER_LAW_MODEL:
        return CANONICAL_POWER_LAW_MODEL
    return model


def fit_models_equivalent(left: str, right: str) -> bool:
    return canonical_fit_model(left) == canonical_fit_model(right)


def is_power_law_fit_model(model: str) -> bool:
    return canonical_fit_model(model) == CANONICAL_POWER_LAW_MODEL


def power_law_floor_predictions(
    coefficients: list[float],
    n_values: list[float],
) -> list[float]:
    e_inf, amplitude, alpha = coefficients
    return [float(e_inf) + float(amplitude) * (float(n) ** (-float(alpha))) for n in n_values]


def predictions_nonnegative_on_domain(
    coefficients: list[float],
    n_values: list[float],
) -> bool:
    return all(
        value >= -NONNEG_PREDICTION_TOL
        for value in power_law_floor_predictions(coefficients, n_values)
    )


def nonnegative_floor_coefficients(
    transformed: list[float],
    y_values: list[float],
) -> tuple[float, float]:
    candidates: list[tuple[float, float]] = []
    design = [[1.0, value] for value in transformed]
    coefficients = least_squares_coefficients(design, y_values)
    candidates.append((max(0.0, float(coefficients[0])), max(0.0, float(coefficients[1]))))

    denom = sum(value * value for value in transformed)
    if denom > 0:
        amplitude = sum(t * y for t, y in zip(transformed, y_values)) / denom
        candidates.append((0.0, max(0.0, float(amplitude))))

    floor = max(0.0, float(sum(y_values) / len(y_values)))
    candidates.append((floor, 0.0))
    candidates.append((0.0, 0.0))

    unique: list[tuple[float, float]] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return min(
        unique,
        key=lambda pair: sum(
            (y - (pair[0] + pair[1] * transformed_value)) ** 2
            for transformed_value, y in zip(transformed, y_values)
        ),
    )


def fit_power_law_floor(n_values: list[float], y_values: list[float]) -> dict[str, Any]:
    """Constrained canonical model: y(N) = E_inf + A * N^(-alpha)."""
    if len(n_values) < required_fit_points(CANONICAL_POWER_LAW_MODEL):
        return {
            "model": CANONICAL_POWER_LAW_MODEL,
            "status": "skipped_insufficient_points",
            "n_points": len(n_values),
            "formula": "y = E_inf + A N^-alpha",
            **fit_policy_metadata(CANONICAL_POWER_LAW_MODEL, status="skipped_insufficient_points", n_points=len(n_values)),
        }

    n_min = min(n_values)
    n_max = max(n_values)
    if n_min <= 0:
        return {
            "model": CANONICAL_POWER_LAW_MODEL,
            "status": "failed_invalid_domain",
            "n_points": len(n_values),
            "formula": "y = E_inf + A N^-alpha",
            **fit_policy_metadata(CANONICAL_POWER_LAW_MODEL, status="failed_invalid_domain", n_points=len(n_values)),
        }

    best: dict[str, Any] | None = None
    for idx in range(160):
        alpha = 0.05 + (4.0 - 0.05) * idx / 159.0
        if alpha <= 0:
            continue
        transformed = [n ** (-alpha) for n in n_values]
        e_inf, amplitude = nonnegative_floor_coefficients(transformed, y_values)
        coefficients = [float(e_inf), float(amplitude), float(alpha)]
        if e_inf < 0 or amplitude < 0:
            continue
        if not predictions_nonnegative_on_domain(coefficients, n_values):
            continue
        predicted = power_law_floor_predictions(coefficients, n_values)
        sse = sum((y - yhat) ** 2 for y, yhat in zip(y_values, predicted))
        if best is None or sse < best["sse"]:
            best = {
                "alpha": float(alpha),
                "coefficients": coefficients,
                "predicted": predicted,
                "sse": sse,
            }

    if best is None:
        return {
            "model": CANONICAL_POWER_LAW_MODEL,
            "status": "failed_constraint_violation",
            "n_points": len(n_values),
            "formula": "y = E_inf + A N^-alpha",
            **fit_policy_metadata(CANONICAL_POWER_LAW_MODEL, status="failed_constraint_violation", n_points=len(n_values)),
        }

    summary = fit_summary(
        CANONICAL_POWER_LAW_MODEL,
        n_values,
        y_values,
        best["predicted"],
        best["coefficients"],
    )
    summary["status"] = "ok"
    summary["formula"] = "y = E_inf + A N^-alpha"
    summary.update(fit_policy_metadata(CANONICAL_POWER_LAW_MODEL, status="ok", n_points=len(n_values)))
    summary["constraints"] = {
        "e_inf_nonnegative": best["coefficients"][0] >= 0,
        "amplitude_nonnegative": best["coefficients"][1] >= 0,
        "alpha_positive": best["coefficients"][2] > 0,
        "predictions_nonnegative_on_observed_domain": True,
    }
    summary["fit_domain"] = {"min_n": float(n_min), "max_n": float(n_max)}
    summary["coefficients_named"] = {
        "e_inf": best["coefficients"][0],
        "amplitude": best["coefficients"][1],
        "alpha": best["coefficients"][2],
    }
    return summary


def fit_power_law(n_values: list[float], y_values: list[float]) -> dict[str, Any]:
    """Legacy alias for the constrained canonical power-law + floor model."""
    result = fit_power_law_floor(n_values, y_values)
    if result.get("status") == "ok":
        result = dict(result)
        result["legacy_model_alias"] = "power_law"
    return result

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
    summary["diagnostic_only"] = True
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
    summary["diagnostic_only"] = True
    summary["curve_points"] = curve_points
    summary["moving_average_window"] = window
    summary["fit_domain"] = {
        "min_n": min(sorted_n),
        "max_n": max(sorted_n),
    }
    return summary


def fit_cumulative_best_model(n_values: list[float], y_values: list[float]) -> dict[str, Any]:
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
            "model": "cumulative_best",
            "fit_model": "cumulative_best",
            "status": "failed",
            "error": "No finite points for cumulative best envelope.",
            "n_points": 0,
            "diagnostic_only": True,
            "description": "Monotone non-increasing envelope over observed N only.",
        }

    curve_points: list[dict[str, Any]] = []
    envelope_values: list[float] = []
    best_so_far = math.inf
    for n_value, y_value in clean:
        best_so_far = min(best_so_far, y_value)
        envelope_values.append(float(best_so_far))
        curve_points.append({"x": float(n_value), "y": float(best_so_far)})

    sorted_n = [item[0] for item in clean]
    sorted_y = [item[1] for item in clean]
    summary = fit_summary("cumulative_best", sorted_n, sorted_y, envelope_values, [])
    summary["status"] = "ok"
    summary["fit_model"] = "cumulative_best"
    summary["formula"] = "y_hat(N_i) = min_{j<=i} y(N_j)"
    summary["description"] = "Monotone non-increasing envelope over observed aggregated points sorted by dataset_size_x; no extrapolation."
    summary["diagnostic_only"] = True
    summary["curve_points"] = curve_points
    summary["fit_domain"] = {
        "min_n": min(sorted_n),
        "max_n": max(sorted_n),
    }
    return summary


def no_fit_summary(n_points: int) -> dict[str, Any]:
    return {
        "model": "none",
        "fit_model": "none",
        "status": "not_used",
        "n_points": n_points,
        "description": "No curve fit was used; thresholds are computed from observed aggregated points.",
        "formula": None,
        **fit_policy_metadata("none", status="not_used", n_points=n_points),
    }



def predict_fit(model: str, coefficients: list[float], n_values: list[float]) -> list[float]:
    if model in CURVE_POINT_FIT_MODELS:
        raise ValueError(f"{model} stores explicit curve_points; use curve_points instead of predict_fit.")
    if model == "none":
        raise ValueError("none does not fit a curve; use observed rows instead.")
    if is_power_law_fit_model(model):
        return power_law_floor_predictions([float(item) for item in coefficients], n_values)
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

    if fit_model == "none":
        return [], no_fit_summary(len(clean_rows))

    if len(clean_rows) < required_fit_points(fit_model):
        return [], {
            "status": "skipped_insufficient_points",
            "fit_model": fit_model,
            "n_points": len(clean_rows),
            **fit_policy_metadata(fit_model, status="skipped_insufficient_points", n_points=len(clean_rows)),
        }

    n_values = [float(row["dataset_size_x"]) for row in clean_rows]
    y_values = [float(row["primary_metric_mev_mean"]) for row in clean_rows]

    if is_power_law_fit_model(fit_model):
        fit = fit_power_law_floor(n_values, y_values)
    elif fit_model in LOWESS_FIT_MODELS:
        fit = fit_lowess_model(fit_model, n_values, y_values)
    elif fit_model == "moving_average":
        fit = fit_moving_average_model(
            n_values,
            y_values,
            window_size=moving_average_window,
        )
    elif fit_model == "cumulative_best":
        fit = fit_cumulative_best_model(n_values, y_values)
    else:
        fit = fit_linear_model(fit_model, n_values, y_values)

    if fit.get("status") not in {None, "ok"}:
        return [], fit

    if fit_model in CURVE_POINT_FIT_MODELS:
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
        negative_predictions = [float(value) for value in y_grid if math.isfinite(value) and value < -NONNEG_PREDICTION_TOL]
        if fit_model in UNCONSTRAINED_FIT_MODELS and negative_predictions:
            fit = dict(fit)
            fit["status"] = "invalid_negative_predictions"
            fit["invalid_for_n_min_thresholding"] = True
            fit["negative_prediction_min_mev"] = min(negative_predictions)
            fit["negative_prediction_count"] = len(negative_predictions)
            fit["fit_model"] = fit_model
            fit["fit_domain"] = fit.get("fit_domain") or {
                "min_n": min(n_values),
                "max_n": max(n_values),
            }
            return [], fit
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
        "fit_model": model,
        "status": "ok",
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
        **fit_policy_metadata(model, status="ok", n_points=len(n_values)),
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
        if model == "none":
            out[model] = no_fit_summary(len(n_values))
            continue
        if len(n_values) < required_fit_points(model):
            out[model] = {
                "model": model,
                "fit_model": model,
                "status": "skipped_insufficient_points",
                "n_points": len(n_values),
                **fit_policy_metadata(model, status="skipped_insufficient_points", n_points=len(n_values)),
            }
            continue
        try:
            if is_power_law_fit_model(model):
                out[model] = fit_power_law_floor(n_values, y_values)
            elif model in LOWESS_FIT_MODELS:
                out[model] = fit_lowess_model(model, n_values, y_values)
            elif model == "moving_average":
                out[model] = fit_moving_average_model(
                    n_values,
                    y_values,
                    window_size=moving_average_window,
                )
            elif model == "cumulative_best":
                out[model] = fit_cumulative_best_model(n_values, y_values)
            else:
                out[model] = fit_linear_model(model, n_values, y_values)
            status = str(out[model].get("status") or "ok")
            out[model].update(fit_policy_metadata(model, status=status, n_points=len(n_values)))
        except Exception as exc:  # noqa: BLE001 - fail-safe postprocess.
            out[model] = {
                "model": model,
                "status": "failed",
                "error": str(exc),
                "n_points": len(n_values),
                **fit_policy_metadata(model, status="failed", n_points=len(n_values)),
            }
    return out


def required_fit_points(model: str) -> int:
    return {
        "none": 1,
        "cumulative_best": 1,
        "moving_average": 1,
        "quadratic": 3,
        "power_law": 3,
        "power_law_floor": 3,
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
        "power_law_floor",
        "lowess_logx",
        "lowess_logx_robust",
        "monotone_lowess_logx",
        "moving_average",
        "cumulative_best",
        "none",
    }
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [model for model in models if model not in allowed]
    if unknown:
        raise SystemExit(f"Unknown fit model(s): {', '.join(unknown)}")
    return models or ["linear"]

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
            for key, linestyle in [("N_min_abs", ":"), (N_MIN_REL_TOL_KEY, "--"), ("N_min_plateau", "-.")]:
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


TEMPORAL_SCALAR_KEYS = (
    "total_energy",
    "total_energy_eV",
    "energy_eV",
    "siesta_total_energy",
    "kinetic_energy_eV",
    "potential_energy_eV",
    "displacement_magnitude",
    "displacement_amplitude",
    "instantaneous_temperature_K",
    "temperature_instant_K",
)
TEMPORAL_INDEX_KEYS = (
    "md_step",
    "snapshot_index",
    "source_frame_index",
    "md_source_frame_index",
    "frame_index",
    "time_index",
    "sample_index_within_block",
)
BLOCK_ID_KEYS = ("block_id", "md_block_id", "source_block_id")
TRAJECTORY_ID_KEYS = ("trajectory_id", "md_trajectory_id", "source_trajectory_id")
BLOCKED_SPLIT_STRATEGIES = {"blocked_with_gap", "blocked", "blocked_split"}
MAX_METADATA_SCALAR_READS = 5000
MAX_AUTOCORR_LAG_CAP = 200
AUTOCORRELATION_CONVENTION = "sokal_positive_lag_inefficiency_v1"
TAU_INT_CONVENTION = "statistical_inefficiency = 1 + 2 * sum_{k>0} rho(k) for rho(k) > 0"
N_EFF_CONVENTION = "N_eff = N / statistical_inefficiency"
N_EFF_MUCH_SMALLER_THAN_NOMINAL_RATIO = 0.25
N_MIN_NOMINAL_WARNING = (
    "N_min uses nominal N. If MD snapshots are autocorrelated, independent sample count "
    "can be lower. Check N_eff before using this as a paper-level claim."
)
N_MIN_CRITERIA = ("N_min_abs", N_MIN_REL_TOL_KEY, "N_min_plateau", "N_min_cost_eff")


def first_non_empty_text(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def temporal_index_value(record: dict[str, Any]) -> int | None:
    for key in TEMPORAL_INDEX_KEYS:
        value = int_number(record.get(key))
        if value is not None:
            return value
    sample_dir = str(record.get("sample_dir") or record.get("snapshot_dir") or "")
    if sample_dir:
        name = Path(sample_dir).name
        if name.isdigit():
            return int(name)
    return None


def split_name_normalized(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"training", "train"}:
        return "train"
    if text in {"validation", "val"}:
        return "validation"
    if text == "test":
        return "test"
    return text


def resolve_split_root(dataset_root: Path, frozen: dict[str, Any] | None) -> Path | None:
    if isinstance(frozen, dict):
        split_root_text = str(frozen.get("split_root") or "").strip()
        if split_root_text:
            split_root = Path(split_root_text)
            if split_root.exists():
                return split_root
    candidate = dataset_root / "splits"
    return candidate if candidate.exists() else None


def load_split_manifest_index(split_root: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for split in ("train", "validation", "test"):
        for row in read_csv(split_root / f"{split}_manifest.csv"):
            merged = dict(row)
            merged.setdefault("split", split)
            for key in (str(row.get("sample_dir") or ""), str(row.get("sample_id") or "")):
                if key:
                    index[key] = merged
    return index


def merge_manifest_row(base: dict[str, Any], manifest_row: dict[str, str] | None) -> dict[str, Any]:
    merged = dict(base)
    if not manifest_row:
        return merged
    for key, value in manifest_row.items():
        if value in (None, "") or str(value).strip() == "":
            continue
        if merged.get(key) in (None, ""):
            merged[key] = value
    return merged


def load_temporal_sample_records(dataset_root: Path) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    records: list[dict[str, Any]] = []
    frozen = read_json(root / "frozen_split_manifest.json")
    split_root = resolve_split_root(root, frozen if isinstance(frozen, dict) else None)
    manifest_index = load_split_manifest_index(split_root) if split_root else {}

    if isinstance(frozen, dict):
        for row in frozen.get("rows") or []:
            if not isinstance(row, dict):
                continue
            sample_dir = str(row.get("sample_dir") or "")
            manifest_row = manifest_index.get(sample_dir) or manifest_index.get(str(row.get("sample_id") or ""))
            records.append(merge_manifest_row(row, manifest_row))

    if not records and split_root:
        for split in ("train", "validation", "test"):
            for row in read_csv(split_root / f"{split}_manifest.csv"):
                merged = dict(row)
                merged.setdefault("split", split)
                records.append(merged)

    if not records:
        validation = read_json(root / "artifact_validation.json")
        if isinstance(validation, dict):
            for snapshot in validation.get("snapshots") or []:
                if not isinstance(snapshot, dict):
                    continue
                records.append(
                    {
                        "sample_dir": snapshot.get("snapshot_dir"),
                        "split": snapshot.get("split"),
                        "system_label": snapshot.get("system_label"),
                    }
                )

    return records


def load_split_summary(dataset_root: Path, split_root: Path | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if split_root is not None:
        candidates.append(split_root / "split_summary.json")
    candidates.append(Path(dataset_root) / "splits" / "split_summary.json")
    for path in candidates:
        payload = read_json(path)
        if isinstance(payload, dict):
            return payload
    return {}


def scalar_value_from_metadata(metadata: dict[str, Any]) -> tuple[str, float] | None:
    for key in TEMPORAL_SCALAR_KEYS:
        value = finite_number(metadata.get(key))
        if value is not None:
            return key, value
    return None


def read_metadata_scalar(sample_dir: Path) -> tuple[str, float] | None:
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    payload = read_json(metadata_path)
    if not isinstance(payload, dict):
        return None
    return scalar_value_from_metadata(payload)


def autocorrelation_function(values: list[float], max_lag: int | None = None) -> list[float]:
    n = len(values)
    if n < 2:
        return [1.0] if n == 1 else []
    mean_value = sum(values) / n
    variance = sum((value - mean_value) ** 2 for value in values) / n
    if variance <= 0:
        return [1.0] + [0.0] * (max_lag or 0)
    lag_limit = max_lag if max_lag is not None else min(n - 1, max(20, n // 4), MAX_AUTOCORR_LAG_CAP)
    lag_limit = max(0, min(lag_limit, n - 1))
    acf: list[float] = []
    for lag in range(lag_limit + 1):
        if lag == 0:
            acf.append(1.0)
            continue
        cov = sum(
            (values[index] - mean_value) * (values[index + lag] - mean_value)
            for index in range(n - lag)
        ) / (n - lag)
        acf.append(cov / variance)
    return acf


def statistical_inefficiency_from_acf(acf: list[float]) -> float | None:
    """Batch-means inefficiency g = 1 + 2 * sum of positive-lag autocorrelations."""
    if not acf:
        return None
    inefficiency = 1.0
    for lag in range(1, len(acf)):
        rho = acf[lag]
        if rho <= 0:
            break
        inefficiency += 2.0 * rho
    return max(inefficiency, 1.0)


def integrated_autocorrelation_time(acf: list[float]) -> float | None:
    """Legacy alias: equals statistical_inefficiency under this convention."""
    return statistical_inefficiency_from_acf(acf)


def effective_sample_size(n: int, statistical_inefficiency: float) -> float | None:
    if n <= 0 or not math.isfinite(statistical_inefficiency) or statistical_inefficiency <= 0:
        return None
    return n / statistical_inefficiency


def autocorrelation_convention_payload() -> dict[str, str]:
    return {
        "autocorrelation_convention": AUTOCORRELATION_CONVENTION,
        "tau_int_convention": TAU_INT_CONVENTION,
        "n_eff_convention": N_EFF_CONVENTION,
    }


def maybe_warn_n_eff_much_smaller_than_nominal(
    warnings: list[str],
    *,
    nominal_n: int | None,
    n_eff: float | None,
) -> None:
    if nominal_n is None or n_eff is None or nominal_n <= 0:
        return
    if n_eff / float(nominal_n) < N_EFF_MUCH_SMALLER_THAN_NOMINAL_RATIO:
        warnings.append("n_eff_much_smaller_than_nominal")


def compute_scalar_autocorrelation_diagnostics(
    values: list[float],
    *,
    max_lag: int | None = None,
) -> dict[str, Any]:
    clean = [float(value) for value in values if finite_number(value) is not None]
    n = len(clean)
    if n < 3:
        return {
            "n": n,
            "status": "insufficient_samples",
            "autocorrelation_available": False,
        }
    acf = autocorrelation_function(clean, max_lag=max_lag)
    statistical_inefficiency = statistical_inefficiency_from_acf(acf)
    n_eff = (
        effective_sample_size(n, statistical_inefficiency)
        if statistical_inefficiency is not None
        else None
    )
    return {
        "n": n,
        "status": "ok",
        "autocorrelation_available": True,
        "acf": acf[: min(len(acf), 25)],
        "statistical_inefficiency": statistical_inefficiency,
        "tau_int": statistical_inefficiency,
        "tau_int_convention": TAU_INT_CONVENTION,
        "n_eff": n_eff,
        "n_eff_convention": N_EFF_CONVENTION,
        "autocorrelation_convention": AUTOCORRELATION_CONVENTION,
    }


def scalar_series_for_records(
    records: list[dict[str, Any]],
    *,
    split: str | None = None,
) -> tuple[str, list[float], list[dict[str, Any]]] | None:
    selected = records
    if split is not None:
        selected = [row for row in records if split_name_normalized(row.get("split")) == split]
    if len(selected) < 3:
        return None

    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        block_key = (
            first_non_empty_text(record, *BLOCK_ID_KEYS)
            or first_non_empty_text(record, "temperature_K")
            or first_non_empty_text(record, *TRAJECTORY_ID_KEYS)
            or "default"
        )
        blocks[str(block_key)].append(record)

    ordered_records: list[dict[str, Any]] = []
    for block_key in sorted(blocks.keys()):
        block_items = sorted(
            blocks[block_key],
            key=lambda row: (
                temporal_index_value(row) if temporal_index_value(row) is not None else 10**12,
                str(row.get("sample_dir") or row.get("sample_id") or ""),
            ),
        )
        ordered_records.extend(block_items)

    if len(ordered_records) > MAX_METADATA_SCALAR_READS:
        ordered_records = ordered_records[:MAX_METADATA_SCALAR_READS]

    series_key: str | None = None
    values: list[float] = []
    used_records: list[dict[str, Any]] = []
    for record in ordered_records:
        sample_dir_text = str(record.get("sample_dir") or record.get("snapshot_dir") or "").strip()
        if not sample_dir_text:
            continue
        sample_dir = Path(sample_dir_text)
        scalar = read_metadata_scalar(sample_dir)
        if scalar is None:
            for key in TEMPORAL_SCALAR_KEYS:
                value = finite_number(record.get(key))
                if value is not None:
                    scalar = (key, value)
                    break
        if scalar is None:
            continue
        key, value = scalar
        if series_key is None:
            series_key = key
        elif key != series_key:
            continue
        values.append(value)
        used_records.append(record)

    if series_key is None or len(values) < 3:
        return None
    return series_key, values, used_records


def diagnose_dataset_temporal_metadata(
    dataset_root: Path,
    *,
    dataset_id: str = "",
) -> dict[str, Any]:
    root = Path(dataset_root)
    warnings: list[str] = []
    records = load_temporal_sample_records(root)
    frozen = read_json(root / "frozen_split_manifest.json")
    split_root = resolve_split_root(root, frozen if isinstance(frozen, dict) else None)
    split_summary = load_split_summary(root, split_root)

    strategy = str(split_summary.get("strategy") or "").strip().lower()
    if not strategy:
        for record in records:
            strategy = str(record.get("split_strategy") or "").strip().lower()
            if strategy:
                break

    temporal_gap = int_number(split_summary.get("temporal_gap"))
    if temporal_gap is None:
        for record in records:
            temporal_gap = int_number(record.get("temporal_gap"))
            if temporal_gap is not None:
                break

    blocked_split = strategy in BLOCKED_SPLIT_STRATEGIES or "blocked" in strategy
    temporal_order_detected = any(temporal_index_value(record) is not None for record in records)

    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        block_key = (
            first_non_empty_text(record, *BLOCK_ID_KEYS)
            or first_non_empty_text(record, "temperature_K")
            or first_non_empty_text(record, *TRAJECTORY_ID_KEYS)
            or "unknown"
        )
        blocks[str(block_key)].append(record)

    block_sizes = {block_id: len(items) for block_id, items in sorted(blocks.items())}
    split_counts = defaultdict(int)
    for record in records:
        split_counts[split_name_normalized(record.get("split"))] += 1

    if not records:
        warnings.append("temporal_metadata_missing_no_sample_records")
    if not temporal_order_detected:
        warnings.append("temporal_order_not_detected")
    if temporal_gap is None:
        warnings.append("temporal_gap_unknown")
    elif temporal_gap <= 1:
        warnings.append("temporal_gap_le_1_adjacent_frames_may_leak")

    train_count = split_counts.get("train", 0)
    nominal_n_train = train_count or split_counts_from_dataset_root(root).get("train")
    if nominal_n_train is None:
        nominal_n_train = split_counts_from_dataset_root(root).get("training")

    scalar_series = scalar_series_for_records(records, split="train")
    autocorrelation: dict[str, Any] = {
        "available": False,
        "scalar_series_key": None,
        "train": {"status": "no_scalar_series"},
    }
    estimated_n_eff_train: float | None = None

    if scalar_series is None:
        warnings.append("autocorrelation_unavailable_no_cheap_scalar_series")
    else:
        series_key, values, used_records = scalar_series
        autocorr = compute_scalar_autocorrelation_diagnostics(values)
        per_block: dict[str, Any] = {}
        for block_id, block_records in sorted(blocks.items()):
            block_series = scalar_series_for_records(block_records, split="train")
            if block_series is None:
                continue
            block_key, block_values, _ = block_series
            if block_key != series_key:
                continue
            block_diag = compute_scalar_autocorrelation_diagnostics(block_values)
            if block_diag.get("status") == "ok":
                per_block[block_id] = {
                    "n": block_diag.get("n"),
                    "statistical_inefficiency": block_diag.get("statistical_inefficiency"),
                    "tau_int": block_diag.get("tau_int"),
                    "n_eff": block_diag.get("n_eff"),
                }

        autocorrelation = {
            "available": bool(autocorr.get("autocorrelation_available")),
            "scalar_series_key": series_key,
            **autocorrelation_convention_payload(),
            "train": {
                **autocorr,
                "records_used": len(used_records),
            },
            "by_block": per_block,
        }
        estimated_n_eff_train = finite_number(autocorr.get("n_eff"))
        maybe_warn_n_eff_much_smaller_than_nominal(
            warnings,
            nominal_n=int_number(nominal_n_train),
            n_eff=estimated_n_eff_train,
        )

    return {
        "dataset_id": dataset_id or root.name,
        "dataset_root": str(root.resolve()),
        "dataset_size": len(records),
        "split_counts": dict(split_counts),
        "split_strategy": strategy or None,
        "blocked_split": blocked_split,
        "temporal_gap": temporal_gap,
        "temporal_order_detected": temporal_order_detected,
        "n_temporal_blocks": len(blocks),
        "block_sizes": block_sizes,
        "block_ids": sorted(blocks.keys()),
        "nominal_n_train": nominal_n_train,
        "estimated_n_eff_train": estimated_n_eff_train,
        "autocorrelation_available": bool(autocorrelation.get("available")),
        "autocorrelation": autocorrelation,
        "warnings": warnings,
    }


def collect_dataset_roots_for_diagnostics(
    normalized_rows: list[dict[str, Any]],
    run_roots: list[Path],
) -> list[tuple[str, Path]]:
    discovered: dict[str, Path] = {}
    for row in normalized_rows:
        root_text = str(row.get("dataset_root") or "").strip()
        dataset_id = str(row.get("dataset_id") or "").strip()
        if root_text:
            root = Path(root_text).resolve()
            key = str(root)
            discovered[key] = (dataset_id or root.name, root)

    for run_root in run_roots:
        candidates = [
            run_root,
            run_root / "dataset",
            run_root.parent / "dataset",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            if (candidate / "frozen_split_manifest.json").exists() or (candidate / "artifact_validation.json").exists():
                key = str(candidate.resolve())
                discovered.setdefault(key, (candidate.name, candidate.resolve()))

    return list(discovered.values())


def summarize_temporal_diagnostics(
    normalized_rows: list[dict[str, Any]],
    *,
    run_roots: list[Path],
) -> dict[str, Any]:
    warnings: list[str] = []
    datasets: list[dict[str, Any]] = []
    for dataset_id, dataset_root in collect_dataset_roots_for_diagnostics(normalized_rows, run_roots):
        try:
            datasets.append(
                diagnose_dataset_temporal_metadata(dataset_root, dataset_id=dataset_id)
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must never abort analysis.
            warnings.append(f"temporal_diagnostics_failed:{dataset_root}:{exc}")
            datasets.append(
                {
                    "dataset_id": dataset_id or dataset_root.name,
                    "dataset_root": str(dataset_root),
                    "dataset_size": 0,
                    "warnings": [f"temporal_diagnostics_failed:{exc}"],
                    "autocorrelation_available": False,
                }
            )

    if not datasets:
        warnings.append("temporal_diagnostics_no_dataset_roots_detected")

    nominal_values = [
        int_number(item.get("nominal_n_train"))
        for item in datasets
        if int_number(item.get("nominal_n_train")) is not None
    ]
    nominal_n_train = max(nominal_values) if nominal_values else None

    n_eff_values = [
        finite_number(item.get("estimated_n_eff_train"))
        for item in datasets
        if finite_number(item.get("estimated_n_eff_train")) is not None
    ]
    estimated_n_eff_train: float | dict[str, float] | None
    if not n_eff_values:
        estimated_n_eff_train = None
    elif len(n_eff_values) == 1:
        estimated_n_eff_train = n_eff_values[0]
    else:
        estimated_n_eff_train = {
            "min": min(n_eff_values),
            "median": statistics.median(n_eff_values),
            "max": max(n_eff_values),
        }

    autocorrelation_available = any(bool(item.get("autocorrelation_available")) for item in datasets)
    for item in datasets:
        warnings.extend(item.get("warnings") or [])

    maybe_warn_n_eff_much_smaller_than_nominal(
        warnings,
        nominal_n=int_number(nominal_n_train),
        n_eff=finite_number(estimated_n_eff_train)
        if not isinstance(estimated_n_eff_train, dict)
        else finite_number((estimated_n_eff_train or {}).get("median")),
    )

    status_message = (
        f"Estimated N_eff range: {json.dumps(estimated_n_eff_train, ensure_ascii=False)} "
        f"({N_EFF_CONVENTION}; N_min still uses nominal N)"
        if estimated_n_eff_train is not None
        else "N is nominal; N_eff not estimated"
    )

    return {
        "datasets": datasets,
        "nominal_n_train": nominal_n_train,
        "estimated_n_eff_train": estimated_n_eff_train,
        "autocorrelation_available": autocorrelation_available,
        "status_message": status_message,
        "warnings": sorted(set(warnings)),
        **autocorrelation_convention_payload(),
    }


def representative_n_eff_value(value: Any) -> float | None:
    if isinstance(value, dict):
        return finite_number(value.get("median")) or finite_number(value.get("min")) or finite_number(value.get("max"))
    return finite_number(value)


def n_eff_over_n_nominal(temporal_diagnostics: dict[str, Any]) -> float | None:
    nominal = int_number(temporal_diagnostics.get("nominal_n_train"))
    n_eff = representative_n_eff_value(temporal_diagnostics.get("estimated_n_eff_train"))
    if nominal is None or nominal <= 0 or n_eff is None:
        return None
    return float(n_eff) / float(nominal)


def nominal_n_min_map(thresholds: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for method, method_thresholds in thresholds.items():
        out[method] = {
            criterion: method_thresholds.get(criterion)
            for criterion in N_MIN_CRITERIA
            if criterion in method_thresholds
        }
    return out


def effective_samples_at_nominal_n_min(
    thresholds: dict[str, dict[str, Any]],
    *,
    n_eff_ratio: float | None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for method, method_thresholds in thresholds.items():
        out[method] = {}
        for criterion in N_MIN_CRITERIA:
            value = finite_number(method_thresholds.get(criterion))
            out[method][criterion] = float(value) * n_eff_ratio if value is not None and n_eff_ratio is not None else None
    return out


def scientific_claim_status_payload(
    *,
    temporal_diagnostics: dict[str, Any],
    thresholds: dict[str, dict[str, Any]],
    aggregation_mode: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = [N_MIN_NOMINAL_WARNING]
    temporal_warnings = [str(item) for item in temporal_diagnostics.get("warnings") or []]
    autocorrelation_available = bool(temporal_diagnostics.get("autocorrelation_available"))
    ratio = n_eff_over_n_nominal(temporal_diagnostics)

    if not autocorrelation_available:
        blockers.append("paper_blocked_if_autocorrelation_unavailable")
    if ratio is not None and ratio < N_EFF_MUCH_SMALLER_THAN_NOMINAL_RATIO:
        blockers.append("paper_blocked_if_n_eff_much_smaller_than_nominal")
    if "n_eff_much_smaller_than_nominal" in temporal_warnings and "paper_blocked_if_n_eff_much_smaller_than_nominal" not in blockers:
        blockers.append("paper_blocked_if_n_eff_much_smaller_than_nominal")
    if aggregation_mode in {"best_config", "mean_replicates"}:
        blockers.append(f"paper_blocked_if_aggregation_mode_{aggregation_mode}")

    status = "diagnostic_only" if blockers else "paper_candidate_nominal_with_n_eff_diagnostic"
    n_min_nominal = nominal_n_min_map(thresholds)
    effective_at_nominal = effective_samples_at_nominal_n_min(thresholds, n_eff_ratio=ratio)
    return {
        "n_min_basis": "nominal",
        "N_min_nominal": n_min_nominal,
        "N_eff_diagnostic_available": ratio is not None,
        "N_eff_over_N_nominal": ratio,
        "effective_samples_at_N_min_nominal": effective_at_nominal,
        "N_min_eff_diagnostic": effective_at_nominal,
        "scientific_claim_status": status,
        "paper_level_blockers": sorted(set(blockers)),
        "paper_level_warnings": warnings,
        "n_eff_diagnostic_note": (
            "Effective-N values are diagnostics only. They do not replace nominal N_min "
            "or constitute validated paper-level replacements without a stronger protocol."
        ),
    }


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
    temporal_diagnostics: dict[str, Any] | None = None,
    scientific_status: dict[str, Any] | None = None,
    replicate_bootstrap: dict[str, Any] | None = None,
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
    lines.append("| Metodo | Best observado meV | N_min_abs | N_min_rel_tol | N_min_plateau | N_min_cost_eff |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method, summary in thresholds.items():
        lines.append(
            "| {method} | {best} | {absn} | {rel} | {plateau} | {cost} |".format(
                method=method,
                best=format_optional(summary.get("best_observed_mev")),
                absn=format_optional(summary.get("N_min_abs"), precision=0),
                rel=format_optional(summary.get(N_MIN_REL_TOL_KEY), precision=0),
                plateau=format_optional(summary.get("N_min_plateau"), precision=0),
                cost=format_optional(summary.get("N_min_cost_eff"), precision=0),
            )
        )
    lines.append("\n## Fits\n")
    for method, method_fits in fits.items():
        lines.append(f"\n### {method}")
        lines.append("| Modelo | Estado | Politica | RMSE meV | R2 | Coeficientes |")
        lines.append("|---|---|---|---:|---:|---|")
        for model, fit in method_fits.items():
            lines.append(
                "| {model} | {status} | {policy} | {rmse} | {r2} | `{coeffs}` |".format(
                    model=model,
                    status=fit.get("status"),
                    policy=fit.get("fit_policy") or ("paper_candidate" if fit.get("paper_candidate") else "diagnostic_only"),
                    rmse=format_optional(fit.get("rmse_mev")),
                    r2=format_optional(fit.get("r2")),
                    coeffs=json.dumps(fit.get("coefficients") or [], ensure_ascii=False),
                )
            )
    lines.append("\n## Replicate bootstrap\n")
    boot = replicate_bootstrap or disabled_bootstrap_summary()
    lines.append(f"- Label: {boot.get('display_label') or REPLICATE_BOOTSTRAP_LABEL}")
    lines.append(f"- Enabled: {bool(boot.get('enabled'))}")
    lines.append(
        "- Scope: row-level replicate/seed resampling within `(method, dataset_size_x)`; "
        "not temporal/block bootstrap and not full scientific uncertainty."
    )
    if boot.get("warnings"):
        lines.append(f"- Replicate resampling warnings: `{json.dumps(boot.get('warnings'), ensure_ascii=False)}`")
    lines.append("\n## Temporal diagnostics (MD snapshot independence)\n")
    temporal = temporal_diagnostics or {}
    lines.append(
        f"- Status: {temporal.get('status_message') or 'N is nominal; N_eff not estimated'}"
    )
    lines.append(f"- Nominal N_train (metadata): {format_optional(temporal.get('nominal_n_train'), precision=0)}")
    estimated = temporal.get("estimated_n_eff_train")
    if estimated is None:
        lines.append("- Estimated N_eff_train: not available")
    else:
        lines.append(f"- Estimated N_eff_train: `{json.dumps(estimated, ensure_ascii=False)}`")
    lines.append(
        f"- Autocorrelation diagnostic available: {bool(temporal.get('autocorrelation_available'))}"
    )
    status_payload = scientific_status or {}
    lines.append(f"- N_min basis: `{status_payload.get('n_min_basis') or 'nominal'}`")
    lines.append(
        f"- Scientific claim status: `{status_payload.get('scientific_claim_status') or 'diagnostic_only'}`"
    )
    blockers = status_payload.get("paper_level_blockers") or []
    if blockers:
        lines.append(f"- Paper-level blockers: `{json.dumps(blockers, ensure_ascii=False)}`")
    else:
        lines.append("- Paper-level blockers: none from temporal diagnostics")
    ratio = status_payload.get("N_eff_over_N_nominal")
    lines.append(f"- N_eff / N_nominal: {format_optional(ratio)}")
    for item in temporal.get("datasets") or []:
        lines.append(
            f"- Dataset `{item.get('dataset_id')}`: blocks={item.get('n_temporal_blocks')}, "
            f"strategy={item.get('split_strategy') or '-'}, temporal_gap={item.get('temporal_gap')}, "
            f"blocked_split={item.get('blocked_split')}"
        )
    lines.append(
        f"\n{N_MIN_NOMINAL_WARNING} Effective-N values are diagnostics only and do not "
        "replace nominal N_min without a stronger validated protocol.\n"
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


def parse_bootstrap_replicates(value: Any) -> int:
    count = int(value or 0)
    if count < 0:
        raise SystemExit("bootstrap_replicates must be >= 0")
    return count


def parse_bootstrap_seed(value: Any) -> int:
    return int(value if value is not None else DEFAULT_BOOTSTRAP_SEED)


def parse_ci_level(value: Any) -> float:
    level = float(value if value is not None else DEFAULT_CI_LEVEL)
    if not 0.0 < level < 1.0:
        raise SystemExit("ci_level must be in (0, 1)")
    return level


def summary_normalized_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact replicate-level rows for UI raw-point overlays."""
    out: list[dict[str, Any]] = []
    for row in rows:
        metric = row_primary_metric_mev(row)
        size = int_number(row.get("dataset_size_x"))
        if metric is None or size is None:
            continue
        out.append(
            {
                "method": row.get("method"),
                "dataset_size_x": size,
                "dataset_size_total": row.get("dataset_size_total"),
                "dataset_size_train": row.get("dataset_size_train"),
                "primary_metric_mev": metric,
                "primary_metric_mev_mean": metric,
                "config_id": row.get("config_id"),
                "seed": row.get("seed"),
                "source_run_root": row.get("source_run_root"),
            }
        )
    return out


def disabled_bootstrap_summary(*, replicates_requested: int = 0, ci_level: float = DEFAULT_CI_LEVEL) -> dict[str, Any]:
    return {
        "enabled": False,
        "bootstrap_type": "replicate_resampling",
        "display_label": REPLICATE_BOOTSTRAP_LABEL,
        "replicates_requested": replicates_requested,
        "replicates_successful": 0,
        "replicates_failed": 0,
        "ci_level": ci_level if replicates_requested > 0 else None,
        "seed": None,
        "n_min_source": None,
        "n_min_fit_model": None,
        "by_method": {},
        "criteria": list(BOOTSTRAP_N_MIN_CRITERIA),
        "legacy_criterion_aliases": dict(LEGACY_THRESHOLD_ALIASES),
        "failure_counts": {},
        "failure_reasons": [],
        "warnings": [],
        "limitations": [
            "replicate_row_resampling_only",
            "does_not_model_temporal_autocorrelation",
            "does_not_model_model_selection_uncertainty",
            "does_not_model_hyperparameter_selection_uncertainty",
            "does_not_model_dependence_between_dataset_sizes",
            "not_a_temporal_or_block_bootstrap",
        ],
    }


def group_normalized_rows_by_method_size(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metric = row_primary_metric_mev(row)
        size = int_number(row.get("dataset_size_x"))
        if metric is None or size is None:
            continue
        grouped[(str(row["method"]), int(size))].append(row)
    return grouped


def bootstrap_resampling_has_variation(rows: list[dict[str, Any]]) -> bool:
    grouped = group_normalized_rows_by_method_size(rows)
    return any(len(items) > 1 for items in grouped.values())


def replicate_bootstrap_scope_warnings(rows: list[dict[str, Any]], *, aggregation_mode: str) -> list[str]:
    warnings = [
        "replicate_bootstrap_row_level_replicates_only",
        "replicate_bootstrap_no_temporal_or_block_bootstrap",
        "replicate_bootstrap_does_not_capture_model_selection_uncertainty",
        "replicate_bootstrap_does_not_capture_hyperparameter_selection_uncertainty",
        "replicate_bootstrap_does_not_capture_dependence_between_dataset_sizes",
    ]
    if not bootstrap_resampling_has_variation(rows):
        warnings.append("replicate_bootstrap_no_multiple_seeds_or_replicates")
    if aggregation_mode in {"best_config", "mean_replicates"}:
        warnings.append(f"replicate_bootstrap_selected_aggregation_is_diagnostic:{aggregation_mode}")
    return warnings


def bootstrap_resample_normalized_rows(
    rows: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    grouped = group_normalized_rows_by_method_size(rows)
    resampled: list[dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        items = grouped[key]
        sample_size = len(items)
        for _ in range(sample_size):
            resampled.append(dict(items[rng.randrange(sample_size)]))
    return resampled


def analysis_rows_from_normalized(
    normalized_rows: list[dict[str, Any]],
    *,
    aggregation_mode: str,
) -> list[dict[str, Any]]:
    if aggregation_mode == "mean_replicates":
        return aggregate_rows_mean_replicates(normalized_rows)
    if aggregation_mode == "mean_seeds_per_config":
        return aggregate_rows_mean_seeds_per_config(normalized_rows)
    if aggregation_mode == "best_config_mean":
        return aggregate_rows_best_config_mean(normalized_rows)
    grouped = group_config_rows(normalized_rows)
    return best_by_method_size(grouped)


def quantile_value(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(values[low])
    weight = position - low
    return float(values[low] * (1.0 - weight) + values[high] * weight)


def bootstrap_ci_from_samples(
    values: list[int],
    *,
    ci_level: float,
    n_requested: int,
) -> dict[str, Any]:
    clean = sorted(int(value) for value in values)
    n_success = len(clean)
    result: dict[str, Any] = {
        "median": None,
        "lower": None,
        "upper": None,
        "ci_level": ci_level,
        "n_bootstrap_requested": n_requested,
        "n_bootstrap_successful": n_success,
        "n_bootstrap_failed": max(0, n_requested - n_success),
    }
    if not clean:
        result["reason"] = "no_successful_bootstrap_values"
        return result
    result["median"] = quantile_value([float(value) for value in clean], 0.5)
    if n_success < MIN_BOOTSTRAP_SUCCESS_FOR_CI:
        result["reason"] = "too_few_successful_bootstrap_replicates"
        return result
    alpha = (1.0 - ci_level) / 2.0
    floats = [float(value) for value in clean]
    result["lower"] = quantile_value(floats, alpha)
    result["upper"] = quantile_value(floats, 1.0 - alpha)
    return result


def compute_bootstrap_n_min(
    normalized_rows: list[dict[str, Any]],
    *,
    methods: list[str],
    n_replicates: int,
    seed: int,
    ci_level: float,
    aggregation_mode: str,
    threshold_mev: float,
    relative_tolerance: float,
    plateau_gain: float,
    n_min_source: str,
    n_min_fit_model: str,
    moving_average_window: int,
) -> dict[str, Any]:
    warnings: list[str] = replicate_bootstrap_scope_warnings(
        normalized_rows,
        aggregation_mode=aggregation_mode,
    )
    if not bootstrap_resampling_has_variation(normalized_rows):
        warnings.append("bootstrap_unavailable_no_replicates")
        return {
            "enabled": True,
            "bootstrap_type": "replicate_resampling",
            "display_label": REPLICATE_BOOTSTRAP_LABEL,
            "replicates_requested": n_replicates,
            "replicates_successful": 0,
            "replicates_failed": n_replicates,
            "ci_level": ci_level,
            "seed": seed,
            "n_min_source": n_min_source,
            "n_min_fit_model": n_min_fit_model if n_min_source == "fit" else None,
            "by_method": {},
            "criteria": list(BOOTSTRAP_N_MIN_CRITERIA),
            "legacy_criterion_aliases": dict(LEGACY_THRESHOLD_ALIASES),
            "failure_counts": {},
            "failure_reasons": [],
            "warnings": warnings,
            "limitations": [
                "replicate_row_resampling_only",
                "does_not_model_temporal_autocorrelation",
                "does_not_model_model_selection_uncertainty",
                "does_not_model_hyperparameter_selection_uncertainty",
                "does_not_model_dependence_between_dataset_sizes",
                "not_a_temporal_or_block_bootstrap",
            ],
        }

    rng = random.Random(seed)
    samples: dict[str, dict[str, list[int]]] = {
        method: {criterion: [] for criterion in BOOTSTRAP_N_MIN_CRITERIA}
        for method in methods
    }
    failure_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    failure_reasons: set[str] = set()
    replicate_successes = 0
    replicate_failures = 0

    for _replicate in range(n_replicates):
        resampled = bootstrap_resample_normalized_rows(normalized_rows, rng)
        agg_rows = analysis_rows_from_normalized(resampled, aggregation_mode=aggregation_mode)
        fit_details: dict[str, dict[str, Any]] = {}

        if n_min_source == "fit":
            thresholds, fit_details, _fit_warnings = thresholds_by_method_from_fit(
                agg_rows,
                threshold_mev=threshold_mev,
                relative_tolerance=relative_tolerance,
                plateau_gain=plateau_gain,
                fit_model=n_min_fit_model,
                moving_average_window=moving_average_window,
            )
        else:
            thresholds = thresholds_by_method(
                agg_rows,
                threshold_mev=threshold_mev,
                relative_tolerance=relative_tolerance,
                plateau_gain=plateau_gain,
            )

        replicate_recorded = False
        for method in methods:
            method_thresholds = thresholds.get(method)
            if not method_thresholds:
                status = str((fit_details.get(method) or {}).get("status") or "missing_threshold")
                failure_counts[method]["fit_failed" if n_min_source == "fit" else "missing_threshold"] += 1
                failure_reasons.add(f"{method}:{status}")
                continue
            for criterion in BOOTSTRAP_N_MIN_CRITERIA:
                value = method_thresholds.get(criterion)
                if value is None:
                    failure_counts[method][f"{criterion}_missing"] += 1
                    failure_reasons.add(f"{method}:{criterion}_missing")
                    continue
                samples[method][criterion].append(int(value))
                replicate_recorded = True

        if replicate_recorded:
            replicate_successes += 1
        else:
            replicate_failures += 1

    by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        by_method[method] = {
            criterion: bootstrap_ci_from_samples(
                samples[method][criterion],
                ci_level=ci_level,
                n_requested=n_replicates,
            )
            for criterion in BOOTSTRAP_N_MIN_CRITERIA
        }
        if N_MIN_REL_TOL_KEY in by_method[method]:
            by_method[method][LEGACY_N_MIN_REL95_KEY] = {
                **by_method[method][N_MIN_REL_TOL_KEY],
                "deprecated_alias_for": N_MIN_REL_TOL_KEY,
            }

    if replicate_successes < MIN_BOOTSTRAP_SUCCESS_FOR_CI:
        warnings.append("bootstrap_too_few_successful_replicates")

    return {
        "enabled": True,
        "bootstrap_type": "replicate_resampling",
        "display_label": REPLICATE_BOOTSTRAP_LABEL,
        "replicates_requested": n_replicates,
        "replicates_successful": replicate_successes,
        "replicates_failed": replicate_failures,
        "ci_level": ci_level,
        "seed": seed,
        "n_min_source": n_min_source,
        "n_min_fit_model": n_min_fit_model if n_min_source == "fit" else None,
        "by_method": by_method,
        "criteria": list(BOOTSTRAP_N_MIN_CRITERIA),
        "legacy_criterion_aliases": dict(LEGACY_THRESHOLD_ALIASES),
        "failure_counts": {method: dict(counts) for method, counts in failure_counts.items()},
        "failure_reasons": sorted(failure_reasons),
        "warnings": warnings,
        "limitations": [
            "replicate_row_resampling_only",
            "does_not_model_temporal_autocorrelation",
            "does_not_model_model_selection_uncertainty",
            "does_not_model_hyperparameter_selection_uncertainty",
            "does_not_model_dependence_between_dataset_sizes",
            "not_a_temporal_or_block_bootstrap",
        ],
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_roots = [Path(item).resolve() for item in args.run_root]
    fit_models = parse_fit_models(args.fit_models)
    moving_average_window = max(1, int(getattr(args, "moving_average_window", None) or 3))
    aggregation_mode = resolve_aggregation_mode(
        getattr(args, "aggregation_mode", None),
        run_root_count=len(run_roots),
    )
    warnings: list[str] = []
    sources: list[str] = []
    all_normalized_rows: list[dict[str, Any]] = []
    for root in run_roots:
        loaded, root_sources, root_warnings = load_run_root_rows(root)
        normalized_rows, normalize_warnings = normalize_rows(
            loaded,
            primary_metric=args.primary_metric,
            x_axis=args.x_axis,
        )
        root_key = str(root.resolve())
        for row in normalized_rows:
            enriched = dict(row)
            enriched["source_run_root"] = enriched.get("source_run_root") or root_key
            all_normalized_rows.append(enriched)
        sources.extend(root_sources)
        warnings.extend(root_warnings + normalize_warnings)

    grouped_rows = group_config_rows(all_normalized_rows)
    raw_rows_count = len(all_normalized_rows)

    best_rows = analysis_rows_for_aggregation_mode(
        all_normalized_rows,
        grouped_rows,
        aggregation_mode=aggregation_mode,
    )
    if aggregation_mode == "mean_replicates" and len(run_roots) > 1:
        warnings.append(f"aggregated_mean_replicates_across_{len(run_roots)}_run_roots")

    aggregated = aggregation_mode in {"mean_seeds_per_config", "best_config_mean"} or (
        aggregation_mode == "mean_replicates"
        and (len(run_roots) > 1 or any(int(row.get("replicate_count") or 1) > 1 for row in best_rows))
    )

    observed_thresholds = thresholds_by_method(
    best_rows,
    threshold_mev=float(args.threshold_mev),
    relative_tolerance=float(args.relative_tolerance),
    plateau_gain=float(args.plateau_gain),
    )

    fit_thresholds: dict[str, dict[str, Any]] = {}
    fit_threshold_details: dict[str, dict[str, Any]] = {}

    requested_n_min_source = str(getattr(args, "n_min_source", None) or "observed")
    requested_fit_model = str(getattr(args, "n_min_fit_model", None) or CANONICAL_POWER_LAW_MODEL)
    canonical_fit = canonical_fit_model(requested_fit_model)
    actual_n_min_source = requested_n_min_source
    actual_fit_model = canonical_fit if requested_n_min_source == "fit" else None
    fallback_used = False
    fallback_reason: str | None = None

    if requested_n_min_source == "fit":
        fit_thresholds, fit_threshold_details, fit_threshold_warnings = thresholds_by_method_from_fit(
        best_rows,
        threshold_mev=float(args.threshold_mev),
        relative_tolerance=float(args.relative_tolerance),
        plateau_gain=float(args.plateau_gain),
        fit_model=requested_fit_model,
        moving_average_window=moving_average_window,
        )
        warnings.extend(fit_threshold_warnings)
        if canonical_fit == "none":
            actual_n_min_source = "observed"
            actual_fit_model = "none"
            thresholds = fit_thresholds or observed_thresholds
        elif fit_thresholds:
            thresholds = fit_thresholds
        else:
            fallback_used = True
            fallback_reason = f"canonical_fit_failed:{canonical_fit}"
            actual_n_min_source = "observed"
            actual_fit_model = None
            thresholds = observed_thresholds
            warnings.append(f"n_min_explicit_fallback_to_observed:{fallback_reason}")
    else:
        fit_thresholds = {}
        fit_threshold_details = {}
        thresholds = observed_thresholds

    bootstrap_replicates = parse_bootstrap_replicates(getattr(args, "bootstrap_replicates", 0))
    bootstrap_seed = parse_bootstrap_seed(getattr(args, "bootstrap_seed", None))
    ci_level = parse_ci_level(getattr(args, "ci_level", None))
    methods = sorted({str(row["method"]) for row in best_rows})
    if bootstrap_replicates > 0:
        replicate_bootstrap = compute_bootstrap_n_min(
            all_normalized_rows,
            methods=methods,
            n_replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            ci_level=ci_level,
            aggregation_mode=aggregation_mode,
            threshold_mev=float(args.threshold_mev),
            relative_tolerance=float(args.relative_tolerance),
            plateau_gain=float(args.plateau_gain),
            n_min_source=actual_n_min_source,
            n_min_fit_model=canonical_fit if requested_n_min_source == "fit" else requested_fit_model,
            moving_average_window=moving_average_window,
        )
        warnings.extend(replicate_bootstrap.get("warnings") or [])
    else:
        replicate_bootstrap = disabled_bootstrap_summary(ci_level=ci_level)

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

    temporal_diagnostics = summarize_temporal_diagnostics(
        all_normalized_rows,
        run_roots=run_roots,
    )
    warnings.extend(temporal_diagnostics.get("warnings") or [])
    scientific_status = scientific_claim_status_payload(
        temporal_diagnostics=temporal_diagnostics,
        thresholds=thresholds,
        aggregation_mode=aggregation_mode,
    )
    warnings.extend(scientific_status.get("paper_level_blockers") or [])

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
        temporal_diagnostics=temporal_diagnostics,
        scientific_status=scientific_status,
        replicate_bootstrap=replicate_bootstrap,
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
        "aggregation_mode": aggregation_mode,
        "aggregated_rows": best_rows,
        "raw_rows_count": raw_rows_count,
        "normalized_rows": summary_normalized_rows(all_normalized_rows),
        "methods": sorted({str(row["method"]) for row in best_rows}),
        "dataset_sizes": sorted({int(row["dataset_size_x"]) for row in best_rows}),
        "thresholds": thresholds,
        "fits": fits,
        "outputs": outputs,
        "warnings": sorted(set(warnings)),
        "status": "ok" if best_rows else "no_usable_metric_rows",
        "forbidden_compute_commands": FORBIDDEN_COMPUTE_COMMANDS,
        "n_min_source": actual_n_min_source,
        "n_min_fit_model": actual_fit_model or requested_fit_model,
        "requested_n_min_source": requested_n_min_source,
        "actual_n_min_source": actual_n_min_source,
        "requested_fit_model": requested_fit_model,
        "actual_fit_model": actual_fit_model,
        "canonical_fit_model": canonical_fit,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "observed_thresholds": observed_thresholds,
        "fit_thresholds": fit_thresholds,
        "fit_threshold_details": fit_threshold_details,
        "deprecated_threshold_aliases": dict(LEGACY_THRESHOLD_ALIASES),
        "moving_average_window": moving_average_window,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed if bootstrap_replicates > 0 else None,
        "ci_level": ci_level,
        "replicate_bootstrap": replicate_bootstrap,
        "bootstrap": {
            **replicate_bootstrap,
            "deprecated_alias_for": "replicate_bootstrap",
        },
        "bootstrap_deprecated_alias_for": "replicate_bootstrap",
        "temporal_diagnostics": temporal_diagnostics,
        "nominal_n_train": temporal_diagnostics.get("nominal_n_train"),
        "estimated_n_eff_train": temporal_diagnostics.get("estimated_n_eff_train"),
        "autocorrelation_available": temporal_diagnostics.get("autocorrelation_available", False),
        **scientific_status,
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
        default=CANONICAL_POWER_LAW_MODEL,
        help="Fit model used when --n-min-source=fit (power_law is a legacy alias).",
    )
    parser.add_argument(
    "--moving-average-window",
    type=int,
    default=3,
    help="Window size in observed points for moving_average fit model.",
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=list(AGGREGATION_MODES),
        default=None,
        help=(
            "How to combine replicate rows per method and dataset size. "
            "Paper-level: mean_seeds_per_config, best_config_mean. "
            "Diagnostic: mean_replicates, best_config. "
            "Default: best_config for one --run-root, mean_replicates for multiple."
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=parse_bootstrap_replicates,
        default=0,
        help="Bootstrap replicates for N_min confidence intervals; 0 disables bootstrap.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=parse_bootstrap_seed,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--ci-level",
        type=parse_ci_level,
        default=DEFAULT_CI_LEVEL,
        help="Confidence level for bootstrap intervals, in (0, 1).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = analyze(args)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}, indent=2))
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
