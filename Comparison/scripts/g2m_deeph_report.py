#!/usr/bin/env python3
"""Generate learning-curve and accuracy/cost reports for Graph2Mat-vs-DeepH."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "graph2mat_deeph_report_v1"
SEARCH_STAGE = "search"
FINAL_TEST_STAGE = "final_test"
VALIDATION_SPLITS = {"validation", "val"}
TEST_SPLITS = {"test"}


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _csv_safe(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            safe[key] = json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False)
        else:
            safe[key] = value
    return safe


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_rows = [_csv_safe(row) for row in rows]
    fieldnames: list[str] = []
    for row in safe_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        writer.writerows(safe_rows)


def load_training_records(run_root: Path | str) -> list[dict[str, Any]]:
    root = Path(str(run_root))
    manifest = read_json(root / "sweep" / "training_sweep_manifest.json")
    rows = manifest.get("runs")
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, dict)]
    ranking_rows = read_json(root / "summary" / "ranking" / "normalized_run_metrics.json").get("rows")
    if isinstance(ranking_rows, list):
        return [dict(row) for row in ranking_rows if isinstance(row, dict)]
    return []


def record_stage(record: dict[str, Any]) -> str:
    return str(record.get("protocol_stage") or record.get("stage") or SEARCH_STAGE).strip().lower() or SEARCH_STAGE


def record_seed(record: dict[str, Any]) -> Any:
    common = record.get("common") if isinstance(record.get("common"), dict) else {}
    overrides = record.get("overrides") if isinstance(record.get("overrides"), dict) else {}
    return record.get("seed", common.get("seed", overrides.get("seed_everything", overrides.get("seed"))))


def record_telemetry(record: dict[str, Any]) -> dict[str, Any]:
    telemetry = record.get("telemetry") if isinstance(record.get("telemetry"), dict) else {}
    if telemetry:
        return dict(telemetry)
    path = record.get("telemetry_path")
    if path:
        payload = read_json(Path(str(path)))
        if payload:
            return payload
    run_root = Path(str(record.get("run_root") or ""))
    model = str(record.get("model") or record.get("method") or "")
    if run_root and model:
        payload = read_json(run_root / "telemetry" / f"{model}.json")
        if payload:
            return payload
    return {
        "telemetry_status": "unavailable",
        "telemetry_warnings": ["telemetry unavailable"],
    }


def _event_metric_value(event: dict[str, Any], metric: str) -> float | None:
    for key in (
        "validation_metric",
        "validation_metric_value",
        "value",
        metric,
        f"{metric}_mean",
    ):
        if key in event:
            number = finite_number(event.get(key))
            if number is not None:
                return number
    return None


def _event_epoch(event: dict[str, Any], fallback: int) -> int | None:
    number = finite_number(event.get("epoch", event.get("best_validation_epoch")))
    return int(number) if number is not None else fallback


def _validation_events_from(record: dict[str, Any], telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    for source in (
        record.get("learning_curve"),
        record.get("validation_events"),
        telemetry.get("learning_curve"),
        telemetry.get("validation_events"),
        (record.get("early_stopping") or {}).get("events") if isinstance(record.get("early_stopping"), dict) else None,
    ):
        if isinstance(source, list):
            return [dict(item) for item in source if isinstance(item, dict)]
    return []


def _metric_name(record: dict[str, Any], telemetry: dict[str, Any], default: str) -> str:
    early = record.get("early_stopping") if isinstance(record.get("early_stopping"), dict) else {}
    return str(
        record.get("validation_metric")
        or early.get("validation_metric_name")
        or early.get("metric")
        or telemetry.get("validation_metric")
        or default
    )


def _epochs_trained(record: dict[str, Any], telemetry: dict[str, Any], events: list[dict[str, Any]]) -> int | None:
    early = record.get("early_stopping") if isinstance(record.get("early_stopping"), dict) else {}
    for value in (telemetry.get("epochs_trained"), early.get("epochs_trained"), record.get("epochs_trained")):
        number = finite_number(value)
        if number is not None:
            return max(1, int(number))
    epochs = [_event_epoch(event, 0) for event in events]
    clean = [epoch for epoch in epochs if epoch is not None]
    return max(clean) if clean else None


def learning_curve_rows(
    records: list[dict[str, Any]],
    *,
    metric: str = "val_loss",
    mode: str = "min",
) -> list[dict[str, Any]]:
    """Return per-epoch validation curves; search-stage test metrics are ignored."""

    rows: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("status") or record.get("run_status") or "completed") not in {"completed", "ok"}:
            continue
        stage = record_stage(record)
        telemetry = record_telemetry(record)
        events = _validation_events_from(record, telemetry)
        metric_name = _metric_name(record, telemetry, metric)
        epochs_trained = _epochs_trained(record, telemetry, events)
        total_wall = finite_number(telemetry.get("wall_clock_seconds_to_best_validation"))
        if total_wall is None:
            total_wall = finite_number(telemetry.get("wall_clock_seconds_total"))
        total_gpu = finite_number(telemetry.get("gpu_hours_to_best_validation"))
        if total_gpu is None:
            total_gpu = finite_number(telemetry.get("gpu_hours_total"))
        training_samples = finite_number(telemetry.get("training_sample_count"))
        matrix_blocks = finite_number(telemetry.get("matrix_block_count"))
        for index, event in enumerate(events, start=1):
            split = str(event.get("metric_split") or event.get("split") or "validation").lower()
            if stage != FINAL_TEST_STAGE and split in TEST_SPLITS:
                continue
            if split not in VALIDATION_SPLITS:
                continue
            value = _event_metric_value(event, metric_name)
            if value is None:
                continue
            epoch = _event_epoch(event, index)
            if epoch is None:
                continue
            progress = (
                min(max(float(epoch) / float(epochs_trained), 0.0), 1.0)
                if epochs_trained
                else None
            )
            rows.append(
                {
                    "model": record.get("model") or record.get("method"),
                    "dataset": record.get("dataset_id"),
                    "dataset_id": record.get("dataset_id"),
                    "config_id": record.get("config_id"),
                    "seed": record_seed(record),
                    "stage": stage,
                    "epoch": epoch,
                    "validation_metric_name": metric_name,
                    "validation_metric": value,
                    "wall_clock_seconds_cumulative": finite_number(event.get("wall_clock_seconds_cumulative"))
                    if event.get("wall_clock_seconds_cumulative") is not None
                    else (total_wall * progress if total_wall is not None and progress is not None else None),
                    "gpu_hours_cumulative": finite_number(event.get("gpu_hours_cumulative"))
                    if event.get("gpu_hours_cumulative") is not None
                    else (total_gpu * progress if total_gpu is not None and progress is not None else None),
                    "samples_seen_cumulative": finite_number(event.get("samples_seen_cumulative"))
                    if event.get("samples_seen_cumulative") is not None
                    else (training_samples * epoch if training_samples is not None else None),
                    "matrix_blocks_seen_cumulative": finite_number(event.get("matrix_blocks_seen_cumulative"))
                    if event.get("matrix_blocks_seen_cumulative") is not None
                    else (matrix_blocks * epoch if matrix_blocks is not None else None),
                    "telemetry_status": telemetry.get("telemetry_status", "partial"),
                    "source": "per_epoch_artifact",
                }
            )
    return rows


def _best_from_curve(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any] | None:
    if not rows:
        return None
    reverse = mode == "max"
    return sorted(
        rows,
        key=lambda row: (
            -(finite_number(row.get("validation_metric")) or 0.0)
            if reverse
            else (finite_number(row.get("validation_metric")) or 0.0),
            int(finite_number(row.get("epoch")) or 0),
        ),
    )[0]


def best_validation_summary(
    records: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    *,
    metric: str = "val_loss",
    mode: str = "min",
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in curve_rows:
        key = (
            str(row.get("model") or ""),
            str(row.get("dataset_id") or ""),
            str(row.get("config_id") or ""),
            str(row.get("seed") or ""),
        )
        by_key.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("status") or record.get("run_status") or "completed") not in {"completed", "ok"}:
            continue
        telemetry = record_telemetry(record)
        early = record.get("early_stopping") if isinstance(record.get("early_stopping"), dict) else {}
        key = (
            str(record.get("model") or record.get("method") or ""),
            str(record.get("dataset_id") or ""),
            str(record.get("config_id") or ""),
            str(record_seed(record) or ""),
        )
        best_curve = _best_from_curve(by_key.get(key, []), mode=mode)
        best_value = (
            finite_number(best_curve.get("validation_metric")) if best_curve else None
        )
        if best_value is None:
            best_value = finite_number(telemetry.get("best_validation_value"))
        if best_value is None:
            best_value = finite_number(early.get("best_validation_value"))
        best_epoch = (
            int(finite_number(best_curve.get("epoch")) or 0) if best_curve else None
        )
        if not best_epoch:
            epoch_number = finite_number(telemetry.get("best_validation_epoch", early.get("best_epoch")))
            best_epoch = int(epoch_number) if epoch_number is not None else None
        warnings = list(telemetry.get("telemetry_warnings") or [])
        if not by_key.get(key):
            warnings.append("per-epoch learning curve unavailable")
        summaries.append(
            {
                "model": key[0],
                "dataset": key[1],
                "dataset_id": key[1],
                "config_id": key[2],
                "seed": record_seed(record),
                "stage": record_stage(record),
                "validation_metric_name": _metric_name(record, telemetry, metric),
                "best_epoch": best_epoch,
                "best_validation_value": best_value,
                "gpu_hours_to_best_validation": finite_number(telemetry.get("gpu_hours_to_best_validation")),
                "wall_clock_to_best_validation": finite_number(telemetry.get("wall_clock_seconds_to_best_validation")),
                "gpu_hours_total": finite_number(telemetry.get("gpu_hours_total")),
                "wall_clock_seconds_total": finite_number(telemetry.get("wall_clock_seconds_total")),
                "peak_gpu_memory_mb": finite_number(telemetry.get("peak_gpu_memory_mb")),
                "telemetry_status": telemetry.get("telemetry_status", "unavailable"),
                "telemetry_warnings": warnings,
                "diagnostic_only": bool(record.get("diagnostic_only"))
                or str(record.get("scientific_status") or "").lower() == "diagnostic_only",
                "adapter_equivalence_status": record.get("adapter_equivalence_status"),
                "split_audit_status": record.get("split_audit_status"),
            }
        )
    return summaries


def _metric_for_pareto(row: dict[str, Any]) -> float | None:
    value = finite_number(row.get("final_test_metric_value"))
    if value is not None and row.get("stage") == FINAL_TEST_STAGE:
        return value
    return finite_number(row.get("best_validation_value"))


def pareto_report_rows(best_rows: list[dict[str, Any]], *, mode: str = "min") -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in best_rows:
        metric_value = _metric_for_pareto(row)
        if metric_value is None:
            continue
        cost = finite_number(row.get("gpu_hours_total"))
        candidates.append(
            {
                "model": row.get("model"),
                "dataset": row.get("dataset"),
                "dataset_id": row.get("dataset_id"),
                "config_id": row.get("config_id"),
                "seed": row.get("seed"),
                "stage": row.get("stage"),
                "metric_name": row.get("validation_metric_name"),
                "metric_value": metric_value,
                "gpu_hours": cost,
                "peak_gpu_memory_mb": row.get("peak_gpu_memory_mb"),
                "telemetry_status": row.get("telemetry_status"),
                "diagnostic_only": row.get("diagnostic_only"),
                "pareto_dominated": False,
                "pareto_status": "frontier",
            }
        )
    for row in candidates:
        cost_a = finite_number(row.get("gpu_hours"))
        value_a = finite_number(row.get("metric_value"))
        if cost_a is None or value_a is None:
            row["pareto_status"] = "cost_unavailable"
            continue
        dominated = False
        for other in candidates:
            if other is row:
                continue
            cost_b = finite_number(other.get("gpu_hours"))
            value_b = finite_number(other.get("metric_value"))
            if cost_b is None or value_b is None:
                continue
            metric_no_worse = value_b <= value_a if mode == "min" else value_b >= value_a
            metric_better = value_b < value_a if mode == "min" else value_b > value_a
            cost_no_worse = cost_b <= cost_a
            cost_better = cost_b < cost_a
            if metric_no_worse and cost_no_worse and (metric_better or cost_better):
                dominated = True
                break
        row["pareto_dominated"] = dominated
        row["pareto_status"] = "dominated" if dominated else "frontier"
    return sorted(
        candidates,
        key=lambda row: (
            bool(row.get("pareto_dominated")),
            finite_number(row.get("metric_value")) or math.inf,
            finite_number(row.get("gpu_hours")) or math.inf,
            str(row.get("model") or ""),
            str(row.get("config_id") or ""),
        ),
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def final_comparison_table(
    best_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    *,
    mode: str = "min",
    compute_threshold: float | None = None,
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in best_rows:
        value = _metric_for_pareto(row)
        if value is None:
            continue
        by_model.setdefault(str(row.get("model") or ""), []).append(row)
    model_stats: list[dict[str, Any]] = []
    for model, rows in sorted(by_model.items()):
        values = [value for row in rows if (value := _metric_for_pareto(row)) is not None]
        costs = [value for row in rows if (value := finite_number(row.get("gpu_hours_total"))) is not None]
        model_stats.append(
            {
                "model": model,
                "mean_metric": _mean(values),
                "std_metric": _std(values),
                "n": len(values),
                "mean_gpu_hours": _mean(costs),
                "diagnostic_only": any(row.get("diagnostic_only") for row in rows),
            }
        )
    clean_stats = [row for row in model_stats if row.get("mean_metric") is not None]
    if mode == "max":
        clean_stats.sort(key=lambda row: (-(row["mean_metric"] or 0.0), row["model"]))
    else:
        clean_stats.sort(key=lambda row: (row["mean_metric"] or math.inf, row["model"]))
    accuracy_winner = clean_stats[0]["model"] if clean_stats else None
    statistically_supported = False
    if len(clean_stats) >= 2 and clean_stats[0]["n"] >= 3 and clean_stats[1]["n"] >= 3:
        best = clean_stats[0]
        other = clean_stats[1]
        best_std = finite_number(best.get("std_metric")) or 0.0
        other_std = finite_number(other.get("std_metric")) or 0.0
        if mode == "min":
            statistically_supported = (best["mean_metric"] + best_std) < (other["mean_metric"] - other_std)
        else:
            statistically_supported = (best["mean_metric"] - best_std) > (other["mean_metric"] + other_std)

    comparison = [
        {
            "claim": "accuracy_winner",
            "winner": accuracy_winner,
            "statistically_supported": statistically_supported,
            "reason": "winner has non-overlapping one-std margin with at least three seeds per top model"
            if statistically_supported
            else "insufficient seeds or overlapping uncertainty; treat as exploratory",
            "model_stats": model_stats,
        }
    ]

    compute_candidates = best_rows
    if compute_threshold is not None:
        if mode == "min":
            compute_candidates = [row for row in best_rows if (_metric_for_pareto(row) or math.inf) <= compute_threshold]
        else:
            compute_candidates = [row for row in best_rows if (_metric_for_pareto(row) or -math.inf) >= compute_threshold]
    cost_sorted = sorted(
        [row for row in compute_candidates if finite_number(row.get("gpu_hours_total")) is not None],
        key=lambda row: (finite_number(row.get("gpu_hours_total")) or math.inf, str(row.get("model") or "")),
    )
    comparison.append(
        {
            "claim": "compute_winner",
            "winner": cost_sorted[0].get("model") if cost_sorted else None,
            "threshold": compute_threshold,
            "reason": "lowest GPU-hours among runs meeting configured threshold"
            if compute_threshold is not None
            else "no compute threshold configured; lowest-cost completed run reported",
            "row": cost_sorted[0] if cost_sorted else None,
        }
    )
    frontier = [row for row in pareto_rows if not row.get("pareto_dominated") and row.get("pareto_status") == "frontier"]
    comparison.append(
        {
            "claim": "practical_pareto_winner",
            "winner": frontier[0].get("model") if len(frontier) == 1 else None,
            "reason": "single non-dominated model/config" if len(frontier) == 1 else "multiple Pareto frontier points; no single practical winner",
            "frontier_count": len(frontier),
            "frontier": frontier,
        }
    )
    return comparison


def generate_report(
    *,
    run_root: Path | str,
    output_dir: Path | None = None,
    metric: str = "val_loss",
    mode: str = "min",
    compute_threshold: float | None = None,
) -> dict[str, Any]:
    root = Path(str(run_root))
    output = output_dir or root / "summary" / "report"
    records = load_training_records(root)
    curve_rows = learning_curve_rows(records, metric=metric, mode=mode)
    best_rows = best_validation_summary(records, curve_rows, metric=metric, mode=mode)
    pareto_rows = pareto_report_rows(best_rows, mode=mode)
    comparison = final_comparison_table(best_rows, pareto_rows, mode=mode, compute_threshold=compute_threshold)
    outputs = {
        "learning_curve_csv": str(output / "learning_curve.csv"),
        "learning_curve_json": str(output / "learning_curve.json"),
        "best_validation_summary_csv": str(output / "best_validation_summary.csv"),
        "best_validation_summary_json": str(output / "best_validation_summary.json"),
        "pareto_accuracy_cost_csv": str(output / "pareto_accuracy_cost.csv"),
        "pareto_accuracy_cost_json": str(output / "pareto_accuracy_cost.json"),
        "final_comparison_json": str(output / "final_comparison.json"),
        "report_summary_json": str(output / "report_summary.json"),
    }
    write_csv(output / "learning_curve.csv", curve_rows)
    write_json(output / "learning_curve.json", {"rows": curve_rows})
    write_csv(output / "best_validation_summary.csv", best_rows)
    write_json(output / "best_validation_summary.json", {"rows": best_rows})
    write_csv(output / "pareto_accuracy_cost.csv", pareto_rows)
    write_json(output / "pareto_accuracy_cost.json", {"rows": pareto_rows})
    write_json(output / "final_comparison.json", {"rows": comparison})
    manifest = {
        "schema": REPORT_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_root": str(root),
        "output_dir": str(output),
        "metric": metric,
        "mode": mode,
        "compute_threshold": compute_threshold,
        "records_count": len(records),
        "learning_curve_rows": len(curve_rows),
        "best_validation_rows": len(best_rows),
        "pareto_rows": len(pareto_rows),
        "outputs": outputs,
        "final_comparison": comparison,
        "warnings": sorted(
            {
                warning
                for row in best_rows
                for warning in (row.get("telemetry_warnings") or [])
                if warning
            }
        ),
    }
    write_json(output / "report_summary.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metric", default="val_loss")
    parser.add_argument("--mode", choices=("min", "max"), default="min")
    parser.add_argument("--compute-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_report(
        run_root=args.run_root,
        output_dir=args.output_dir,
        metric=args.metric,
        mode=args.mode,
        compute_threshold=args.compute_threshold,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
