#!/usr/bin/env python3
"""Final statistical aggregation and winner gates for G2M-vs-DeepH benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

from deeph_prediction_adapter import (
    EQUIVALENCE_PROVEN_RAW_GLOBAL,
    EQUIVALENCE_STATUS_PROVEN,
    EQUIVALENCE_STATUS_UNPROVEN,
)


FINAL_STATS_SCHEMA = "graph2mat_deeph_final_statistics_v1"
MODELS = ("graph2mat", "deeph")
FINAL_TEST_STAGE = "final_test"
TEST_SPLITS = {"test"}
VALID_MODES = {"min", "max"}
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _csv_safe(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            safe[key] = json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False)
        else:
            safe[key] = json_safe(value)
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


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stddev(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.stdev(values) if len(values) > 1 else 0.0


def stderr(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return (statistics.stdev(values) / math.sqrt(len(values)))


def confidence_interval(values: list[float], *, confidence_level: float = 0.95) -> dict[str, Any]:
    if len(values) < 2:
        return {
            "method": "unavailable",
            "confidence_level": confidence_level,
            "low": None,
            "high": None,
            "reason": "at least two seeds are required for a seed-level confidence interval",
        }
    center = mean(values)
    se = stderr(values)
    if center is None or se is None:
        return {"method": "unavailable", "confidence_level": confidence_level, "low": None, "high": None}
    if abs(confidence_level - 0.95) > 1e-9:
        critical = 1.96
        method = "normal_approximation"
    else:
        critical = T_CRITICAL_95.get(len(values) - 1, 1.96)
        method = "student_t_seed_mean"
    return {
        "method": method,
        "confidence_level": confidence_level,
        "low": center - critical * se,
        "high": center + critical * se,
    }


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    for key in (
        "final_test_metric_value",
        "test_metric_value",
        metric,
        f"{metric}_mean",
        "metric_value",
    ):
        if key in row:
            number = finite_number(row.get(key))
            if number is not None:
                return number
    metrics = row.get("final_test_metrics") or row.get("test_metrics")
    if isinstance(metrics, dict):
        for key in (metric, f"{metric}_mean"):
            number = finite_number(metrics.get(key))
            if number is not None:
                return number
    return None


def protocol_stage(row: dict[str, Any]) -> str:
    return str(row.get("protocol_stage") or row.get("stage") or "").strip().lower()


def metric_split(row: dict[str, Any]) -> str:
    return str(row.get("metric_split") or row.get("split") or row.get("evaluation_split") or "").strip().lower()


def final_test_row(row: dict[str, Any], metric: str) -> bool:
    if str(row.get("status") or row.get("run_status") or "completed") not in {"completed", "ok"}:
        return False
    if metric_value(row, metric) is None:
        return False
    stage = protocol_stage(row)
    split = metric_split(row)
    return stage == FINAL_TEST_STAGE or split in TEST_SPLITS


def protocol_violations(rows: list[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        split = metric_split(row)
        stage = protocol_stage(row)
        if split in TEST_SPLITS and stage != FINAL_TEST_STAGE:
            violations.append(
                f"row {row.get('config_id') or index} contains test metrics outside final_test stage"
            )
    return violations


def seed_value(row: dict[str, Any]) -> Any:
    common = row.get("common") if isinstance(row.get("common"), dict) else {}
    overrides = row.get("overrides") if isinstance(row.get("overrides"), dict) else {}
    return row.get("seed", common.get("seed", overrides.get("seed_everything", overrides.get("seed"))))


def compute_field(row: dict[str, Any], key: str) -> float | None:
    value = finite_number(row.get(key))
    if value is not None:
        return value
    telemetry = row.get("telemetry") if isinstance(row.get("telemetry"), dict) else {}
    return finite_number(telemetry.get(key))


def per_system_values(row: dict[str, Any], metric: str) -> list[float]:
    values: list[float] = []
    for key in ("per_system_metrics", "sample_metrics", "system_metrics"):
        raw = row.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            value = metric_value(item, metric)
            if value is not None:
                values.append(value)
    return values


def bootstrap_ci(values: list[float], *, iterations: int = 1000, confidence_level: float = 0.95, seed: int = 0) -> dict[str, Any]:
    if len(values) < 2:
        return {
            "method": "unavailable",
            "confidence_level": confidence_level,
            "low": None,
            "high": None,
            "reason": "per-system metrics unavailable or too small for bootstrap",
        }
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(max(1, int(iterations))):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    alpha = max(0.0, min(1.0, 1.0 - confidence_level))
    low_index = min(max(int((alpha / 2.0) * len(means)), 0), len(means) - 1)
    high_index = min(max(int((1.0 - alpha / 2.0) * len(means)) - 1, 0), len(means) - 1)
    return {
        "method": "bootstrap_per_system_mean",
        "confidence_level": confidence_level,
        "iterations": int(iterations),
        "low": means[low_index],
        "high": means[high_index],
    }


def aggregate_final_seed_metrics(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    expected_seeds: list[int] | None = None,
    confidence_level: float = 0.95,
    bootstrap_iterations: int = 1000,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if final_test_row(row, metric):
            groups.setdefault((str(row.get("model") or ""), str(row.get("dataset_id") or "")), []).append(row)
    expected = set(expected_seeds or [])
    summaries: list[dict[str, Any]] = []
    for (model, dataset_id), group_rows in sorted(groups.items()):
        values = [value for row in group_rows if (value := metric_value(row, metric)) is not None]
        seeds = sorted({seed_value(row) for row in group_rows if seed_value(row) not in (None, "")}, key=str)
        gpu_hours = [value for row in group_rows if (value := compute_field(row, "gpu_hours_total")) is not None]
        peak_memory = [value for row in group_rows if (value := compute_field(row, "peak_gpu_memory_mb")) is not None]
        samples_per_second = [value for row in group_rows if (value := compute_field(row, "samples_per_second")) is not None]
        matrix_blocks_per_second = [
            value for row in group_rows if (value := compute_field(row, "matrix_blocks_per_second")) is not None
        ]
        per_system = [value for row in group_rows for value in per_system_values(row, metric)]
        missing_expected = sorted(expected - {int(seed) for seed in seeds if isinstance(seed, int) or str(seed).isdigit()})
        diagnostic_reasons: list[str] = []
        if model == "deeph":
            for row in group_rows:
                status = str(row.get("adapter_equivalence_status") or "")
                equivalence_status = str(row.get("equivalence_status") or "")
                if not equivalence_status:
                    equivalence_status = (
                        EQUIVALENCE_STATUS_PROVEN
                        if status == EQUIVALENCE_PROVEN_RAW_GLOBAL
                        else EQUIVALENCE_STATUS_UNPROVEN
                    )
                if status != EQUIVALENCE_PROVEN_RAW_GLOBAL or equivalence_status != EQUIVALENCE_STATUS_PROVEN:
                    diagnostic_reasons.append(
                        "deeph adapter equivalence not proven: "
                        f"adapter={status or 'missing'} equivalence={equivalence_status or 'missing'}"
                    )
                    break
                if bool(row.get("diagnostic_only")) or str(row.get("comparability_status") or "").lower() == "diagnostic_only":
                    diagnostic_reasons.append("deeph metrics are diagnostic_only")
                    break
        summaries.append(
            {
                "model": model,
                "dataset_id": dataset_id,
                "metric": metric,
                "mean": mean(values),
                "std": stddev(values),
                "stderr": stderr(values),
                "n_seeds_completed": len(seeds),
                "expected_seed_count": len(expected) if expected else None,
                "completed_seeds": seeds,
                "missing_seeds": missing_expected,
                "confidence_interval": confidence_interval(values, confidence_level=confidence_level),
                "bootstrap_ci": bootstrap_ci(
                    per_system,
                    iterations=bootstrap_iterations,
                    confidence_level=confidence_level,
                    seed=0,
                )
                if per_system
                else {
                    "method": "unavailable",
                    "confidence_level": confidence_level,
                    "low": None,
                    "high": None,
                    "reason": "per-system metrics unavailable",
                },
                "gpu_hours_mean": mean(gpu_hours),
                "gpu_hours_std": stddev(gpu_hours),
                "peak_gpu_memory_mb_mean": mean(peak_memory),
                "peak_gpu_memory_mb_std": stddev(peak_memory),
                "samples_per_second_mean": mean(samples_per_second),
                "samples_per_second_std": stddev(samples_per_second),
                "matrix_blocks_per_second_mean": mean(matrix_blocks_per_second),
                "matrix_blocks_per_second_std": stddev(matrix_blocks_per_second),
                "robust_claim_allowed_by_comparability": not diagnostic_reasons,
                "diagnostic_only_reason": "; ".join(sorted(set(diagnostic_reasons))),
            }
        )
    return summaries


def _metric_better(a: float, b: float, *, mode: str) -> bool:
    return a < b if mode == "min" else a > b


def _ci_separated(best: dict[str, Any], other: dict[str, Any], *, mode: str) -> bool:
    best_ci = best.get("confidence_interval") if isinstance(best.get("confidence_interval"), dict) else {}
    other_ci = other.get("confidence_interval") if isinstance(other.get("confidence_interval"), dict) else {}
    best_low = finite_number(best_ci.get("low"))
    best_high = finite_number(best_ci.get("high"))
    other_low = finite_number(other_ci.get("low"))
    other_high = finite_number(other_ci.get("high"))
    if None in {best_low, best_high, other_low, other_high}:
        return False
    return best_high < other_low if mode == "min" else best_low > other_high


def pareto_frontier(summary_rows: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    rows = []
    for row in summary_rows:
        value = finite_number(row.get("mean"))
        cost = finite_number(row.get("gpu_hours_mean"))
        if value is None:
            continue
        item = {
            "model": row.get("model"),
            "dataset_id": row.get("dataset_id"),
            "metric": row.get("metric"),
            "metric_value": value,
            "gpu_hours": cost,
            "peak_gpu_memory_mb": row.get("peak_gpu_memory_mb_mean"),
            "pareto_dominated": False,
            "pareto_status": "frontier",
        }
        rows.append(item)
    for row in rows:
        cost_a = finite_number(row.get("gpu_hours"))
        value_a = finite_number(row.get("metric_value"))
        if cost_a is None or value_a is None:
            row["pareto_status"] = "cost_unavailable"
            continue
        dominated = False
        for other in rows:
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
    return rows


def decide_winners(
    summary_rows: list[dict[str, Any]],
    *,
    mode: str = "min",
    min_final_seeds: int = 3,
    expected_seeds: list[int] | None = None,
    tolerance: float = 0.0,
    compute_accuracy_threshold: float | None = None,
) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise RuntimeError("mode must be min or max.")
    gates_failed: list[str] = []
    by_model = {str(row.get("model")): row for row in summary_rows if row.get("model") in MODELS}
    if set(by_model) != set(MODELS):
        gates_failed.append("missing_model")
    expected_count = len(expected_seeds or [])
    for model, row in sorted(by_model.items()):
        completed = int(row.get("n_seeds_completed") or 0)
        required = max(min_final_seeds, expected_count)
        if completed < required:
            gates_failed.append(f"incomplete_final_seeds:{model}")
        if row.get("robust_claim_allowed_by_comparability") is not True:
            gates_failed.append(f"diagnostic_only:{model}")
    metric_rows = [row for row in by_model.values() if finite_number(row.get("mean")) is not None]
    if len(metric_rows) < 2:
        gates_failed.append("missing_metric")

    precision_winner = None
    effect_size = None
    ci_rule_passed = False
    if len(metric_rows) >= 2:
        metric_rows.sort(key=lambda row: (finite_number(row.get("mean")) or math.inf) * (1 if mode == "min" else -1))
        best, other = metric_rows[0], metric_rows[1]
        best_mean = finite_number(best.get("mean"))
        other_mean = finite_number(other.get("mean"))
        if best_mean is not None and other_mean is not None:
            effect_size = (other_mean - best_mean) if mode == "min" else (best_mean - other_mean)
            ci_rule_passed = _ci_separated(best, other, mode=mode)
            if effect_size > tolerance and ci_rule_passed and not gates_failed:
                precision_winner = best.get("model")
            elif effect_size <= tolerance:
                gates_failed.append("precision_difference_within_tolerance")
            elif not ci_rule_passed:
                gates_failed.append("confidence_intervals_overlap_or_unavailable")

    compute_winner = None
    if compute_accuracy_threshold is not None:
        eligible = []
        for row in summary_rows:
            value = finite_number(row.get("mean"))
            cost = finite_number(row.get("gpu_hours_mean"))
            if value is None or cost is None:
                continue
            meets = value <= compute_accuracy_threshold if mode == "min" else value >= compute_accuracy_threshold
            if meets:
                eligible.append(row)
        eligible.sort(key=lambda row: (finite_number(row.get("gpu_hours_mean")) or math.inf, str(row.get("model"))))
        compute_winner = eligible[0].get("model") if eligible else None

    pareto_rows = pareto_frontier(summary_rows, mode=mode)
    frontier = [row for row in pareto_rows if row.get("pareto_status") == "frontier"]
    pareto_winner = frontier[0].get("model") if len(frontier) == 1 else None

    robust_claim_allowed = not gates_failed and precision_winner is not None
    diagnostic_reasons = [row.get("diagnostic_only_reason") for row in summary_rows if row.get("diagnostic_only_reason")]
    return {
        "precision_winner": precision_winner,
        "compute_winner": compute_winner,
        "pareto_winner": pareto_winner,
        "practical_pareto_winner": pareto_winner,
        "robust_claim_allowed": robust_claim_allowed,
        "gates_failed": sorted(set(gates_failed)),
        "gates_passed": [] if gates_failed else [
            "complete_final_seeds",
            "statistical_ci_rule",
            "deeph_equivalence_or_not_needed",
            "final_test_protocol",
        ],
        "diagnostic_only_reason": "; ".join(sorted(set(diagnostic_reasons))),
        "effect_size_best_vs_second": effect_size,
        "ci_rule_passed": ci_rule_passed,
        "tolerance": tolerance,
        "compute_accuracy_threshold": compute_accuracy_threshold,
        "pareto_frontier": pareto_rows,
    }


def load_rows(run_root: Path | str) -> list[dict[str, Any]]:
    root = Path(str(run_root))
    candidates = [
        root / "summary" / "ranking" / "normalized_run_metrics.json",
        root / "summary" / "report" / "best_validation_summary.json",
        root / "sweep" / "training_sweep_manifest.json",
    ]
    for path in candidates:
        payload = read_json(path)
        rows = payload.get("rows") or payload.get("runs")
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def final_statistics_report(
    *,
    run_root: Path | str,
    output_dir: Path | None = None,
    metric: str,
    mode: str = "min",
    expected_seeds: list[int] | None = None,
    min_final_seeds: int = 3,
    tolerance: float = 0.0,
    compute_accuracy_threshold: float | None = None,
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    rows = load_rows(run_root)
    violations = protocol_violations(rows)
    summary_rows = aggregate_final_seed_metrics(
        rows,
        metric=metric,
        expected_seeds=expected_seeds,
        bootstrap_iterations=bootstrap_iterations,
    )
    winners = decide_winners(
        summary_rows,
        mode=mode,
        min_final_seeds=min_final_seeds,
        expected_seeds=expected_seeds,
        tolerance=tolerance,
        compute_accuracy_threshold=compute_accuracy_threshold,
    )
    if violations:
        winners["robust_claim_allowed"] = False
        winners["gates_failed"] = sorted(set([*winners.get("gates_failed", []), "protocol_violation_test_metrics_outside_final_stage"]))
        winners["diagnostic_only_reason"] = "; ".join([winners.get("diagnostic_only_reason", ""), *violations]).strip("; ")
    output = output_dir or Path(str(run_root)) / "summary" / "final_statistics"
    manifest = {
        "schema": FINAL_STATS_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_root": str(run_root),
        "output_dir": str(output),
        "metric": metric,
        "mode": mode,
        "expected_seeds": expected_seeds or [],
        "min_final_seeds": min_final_seeds,
        "protocol_violations": violations,
        "final_seed_summary": summary_rows,
        "winner_decision": winners,
        "outputs": {
            "final_seed_summary_csv": str(output / "final_seed_summary.csv"),
            "final_seed_summary_json": str(output / "final_seed_summary.json"),
            "pareto_frontier_csv": str(output / "pareto_frontier.csv"),
            "pareto_frontier_json": str(output / "pareto_frontier.json"),
            "winner_decision_json": str(output / "winner_decision.json"),
            "final_statistics_json": str(output / "final_statistics.json"),
        },
    }
    write_csv(output / "final_seed_summary.csv", summary_rows)
    write_json(output / "final_seed_summary.json", {"rows": summary_rows})
    write_csv(output / "pareto_frontier.csv", winners.get("pareto_frontier") or [])
    write_json(output / "pareto_frontier.json", {"rows": winners.get("pareto_frontier") or []})
    write_json(output / "winner_decision.json", winners)
    write_json(output / "final_statistics.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--mode", choices=("min", "max"), default="min")
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--min-final-seeds", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--compute-accuracy-threshold", type=float, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = [int(item) for item in args.expected_seeds.split(",") if item.strip()] if args.expected_seeds else None
    manifest = final_statistics_report(
        run_root=args.run_root,
        output_dir=args.output_dir,
        metric=args.metric,
        mode=args.mode,
        expected_seeds=expected,
        min_final_seeds=args.min_final_seeds,
        tolerance=args.tolerance,
        compute_accuracy_threshold=args.compute_accuracy_threshold,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
