#!/usr/bin/env python3
"""Analyze MD-vs-AtomDisplacement cross-evaluation winners conservatively.

Examples
--------
    python Comparison/scripts/analyze_winners.py \
      --metrics-csv Comparison/results/exp_001/summary/cross_evaluation_metrics.csv \
      --output-dir Comparison/results/exp_001/summary \
      --primary-metric global_rmse_eV
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD_MD = "md"
METHOD_ATOM = "atom_displacement"
METADATA_COLUMNS = {
    "experiment_id",
    "train_method",
    "test_set",
    "dataset_size",
    "seed",
    "epoch",
    "checkpoint",
    "model_checkpoint",
    "prediction_dir",
    "siesta_reference_dir",
    "sample",
    "sample_id",
    "kind",
    "matrix_path",
    "overlap_source",
    "hamiltonian_symmetrized_for_spectrum",
    "fermi_level_source",
    "status",
}


def parse_value(value: str | None) -> Any:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return number if math.isfinite(number) else None


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: parse_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["metric", "winner"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in METADATA_COLUMNS:
                continue
            if key.endswith("_seconds"):
                continue
            if finite(value):
                columns.add(key)
    return sorted(columns)


def mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def row_dataset_size(row: dict[str, Any]) -> int | str:
    value = row.get("dataset_size")
    if finite(value):
        return int(float(value))
    return "unknown"


def row_seed(row: dict[str, Any]) -> int | str:
    value = row.get("seed")
    if finite(value):
        return int(float(value))
    return "unknown"


def group_metric_means(
    rows: list[dict[str, Any]],
    metrics: list[str],
) -> tuple[dict[tuple[Any, ...], float], dict[tuple[Any, ...], dict[str, float | None]]]:
    values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    timings: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    timing_columns = [
        key
        for key in sorted({key for row in rows for key in row})
        if key.endswith("_seconds")
    ]
    for row in rows:
        train_method = str(row.get("train_method") or "")
        test_set = str(row.get("test_set") or "")
        dataset_size = row_dataset_size(row)
        seed = row_seed(row)
        for metric in metrics:
            value = row.get(metric)
            if finite(value):
                values[(dataset_size, seed, test_set, metric, train_method)].append(float(value))
        for timing_col in timing_columns:
            value = row.get(timing_col)
            if finite(value):
                for metric in metrics:
                    timings[(dataset_size, seed, test_set, metric, train_method)][timing_col].append(float(value))

    means = {key: mean(vals) for key, vals in values.items()}
    timing_means = {
        key: {timing_col: mean(vals) for timing_col, vals in timing_values.items()}
        for key, timing_values in timings.items()
    }
    return {key: value for key, value in means.items() if value is not None}, timing_means


def winner_for(md_mean: float, atom_mean: float, *, lower_is_better: bool) -> str:
    if math.isclose(md_mean, atom_mean, rel_tol=1e-12, abs_tol=1e-12):
        return "tie"
    if lower_is_better:
        return METHOD_ATOM if atom_mean < md_mean else METHOD_MD
    return METHOD_ATOM if atom_mean > md_mean else METHOD_MD


def percent_improvement(md_mean: float, atom_mean: float, *, lower_is_better: bool) -> float | None:
    baseline = abs(md_mean)
    if baseline <= 0:
        return None
    delta = (md_mean - atom_mean) if lower_is_better else (atom_mean - md_mean)
    return 100.0 * delta / baseline


def compare_methods(
    means: dict[tuple[Any, ...], float],
    timing_means: dict[tuple[Any, ...], dict[str, float | None]],
    *,
    lower_is_better: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparable = sorted(
        {
            key[:-1]
            for key in means
            if key[-1] in {METHOD_MD, METHOD_ATOM}
        }
    )
    for dataset_size, seed, test_set, metric in comparable:
        md_key = (dataset_size, seed, test_set, metric, METHOD_MD)
        atom_key = (dataset_size, seed, test_set, metric, METHOD_ATOM)
        if md_key not in means or atom_key not in means:
            continue
        md_mean = means[md_key]
        atom_mean = means[atom_key]
        row = {
            "dataset_size": dataset_size,
            "seed": seed,
            "test_set": test_set,
            "metric": metric,
            "md_mean": md_mean,
            "atom_displacement_mean": atom_mean,
            "absolute_difference_atom_minus_md": atom_mean - md_mean,
            "percent_improvement_atom_vs_md": percent_improvement(
                md_mean,
                atom_mean,
                lower_is_better=lower_is_better,
            ),
            "winner": winner_for(md_mean, atom_mean, lower_is_better=lower_is_better),
        }
        for timing_col, value in timing_means.get(md_key, {}).items():
            row[f"md_{timing_col}"] = value
        for timing_col, value in timing_means.get(atom_key, {}).items():
            row[f"atom_displacement_{timing_col}"] = value
        md_total = row.get("md_total_time_seconds")
        atom_total = row.get("atom_displacement_total_time_seconds")
        if finite(md_total) and finite(atom_total):
            row["compute_budget_seconds"] = max(float(md_total), float(atom_total))
        rows.append(row)
    return rows


def summarize_stability(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(row["dataset_size"], row["test_set"], row["metric"])].append(row)

    summary = []
    for (dataset_size, test_set, metric), rows in sorted(grouped.items()):
        winners = [str(row["winner"]) for row in rows if row["winner"] != "tie"]
        unique_winners = sorted(set(winners))
        stable = len(unique_winners) == 1 and len(rows) > 0
        summary.append(
            {
                "dataset_size": dataset_size,
                "test_set": test_set,
                "metric": metric,
                "seeds_compared": len({row["seed"] for row in rows}),
                "md_wins": sum(1 for winner in winners if winner == METHOD_MD),
                "atom_displacement_wins": sum(1 for winner in winners if winner == METHOD_ATOM),
                "ties": sum(1 for row in rows if row["winner"] == "tie"),
                "stable_across_seeds": stable,
                "stable_winner": unique_winners[0] if stable else "unstable",
                "mean_percent_improvement_atom_vs_md": mean(
                    [
                        float(row["percent_improvement_atom_vs_md"])
                        for row in rows
                        if finite(row.get("percent_improvement_atom_vs_md"))
                    ]
                ),
            }
        )
    return summary


def first_atom_win(summary_rows: list[dict[str, Any]], metric: str, *, allowed_test_sets: set[str]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in summary_rows
        if row.get("metric") == metric
        and row.get("test_set") in allowed_test_sets
        and row.get("stable_winner") == METHOD_ATOM
    ]
    candidates.sort(key=lambda row: (float(row["dataset_size"]) if finite(row["dataset_size"]) else math.inf, str(row["test_set"])))
    return candidates[0] if candidates else None


def first_compute_atom_win(pair_rows: list[dict[str, Any]], metric: str, *, allowed_test_sets: set[str]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in pair_rows
        if row.get("metric") == metric
        and row.get("test_set") in allowed_test_sets
        and row.get("winner") == METHOD_ATOM
        and finite(row.get("compute_budget_seconds"))
    ]
    candidates.sort(key=lambda row: (float(row["compute_budget_seconds"]), float(row["dataset_size"]) if finite(row["dataset_size"]) else math.inf))
    return candidates[0] if candidates else None


def build_recommendation(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    *,
    primary_metric: str,
) -> dict[str, Any]:
    test_sets = sorted({str(row.get("test_set")) for row in rows})
    methods = sorted({str(row.get("train_method")) for row in rows})
    required_cells = {
        (METHOD_MD, "test_md"),
        (METHOD_MD, "test_atomdisp"),
        (METHOD_MD, "test_mixed"),
        (METHOD_ATOM, "test_md"),
        (METHOD_ATOM, "test_atomdisp"),
        (METHOD_ATOM, "test_mixed"),
    }
    available_cells = {
        (str(row.get("train_method")), str(row.get("test_set")))
        for row in rows
    }
    missing_cells = sorted(f"{method} on {test_set}" for method, test_set in required_cells - available_cells)
    allowed_test_sets = {"test_md", "test_mixed"}
    first_size = first_atom_win(summary_rows, primary_metric, allowed_test_sets=allowed_test_sets)
    first_compute = first_compute_atom_win(
        [
            row
            for row in pair_rows
            if row.get("metric") == primary_metric
        ],
        primary_metric,
        allowed_test_sets=allowed_test_sets,
    )
    atom_only_wins = [
        row
        for row in summary_rows
        if row.get("metric") == primary_metric
        and row.get("test_set") == "test_atomdisp"
        and row.get("stable_winner") == METHOD_ATOM
    ]
    status = "no_conservative_winner"
    reason = "AtomDisplacement has not beaten MD stably on MD or mixed test distributions."
    if first_size:
        status = "atom_displacement_conservative_win"
        reason = (
            "AtomDisplacement beats MD on the same frozen test set across seeds "
            "and not only on the AtomDisplacement distribution."
        )
    elif atom_only_wins:
        reason = (
            "AtomDisplacement only wins on test_atomdisp; this is distribution-specific "
            "and is not enough for a conservative recommendation."
        )
    if missing_cells:
        status = "insufficient_cross_evaluation"
        reason = "The cross-evaluation grid is incomplete; no winner should be declared."

    return {
        "status": status,
        "primary_metric": primary_metric,
        "reason": reason,
        "methods_seen": methods,
        "test_sets_seen": test_sets,
        "missing_required_cells": missing_cells,
        "first_dataset_size_where_atom_displacement_beats_md": first_size,
        "first_compute_budget_where_atom_displacement_beats_md": first_compute,
        "conservative_rule": (
            "AtomDisplacement wins only if it beats MD on the same frozen test set, "
            "across seeds, and on test_md or test_mixed rather than only test_atomdisp."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-metric", default="global_rmse_eV")
    parser.add_argument("--higher-is-better", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_rows(args.metrics_csv)
    metrics = metric_columns(rows)
    if args.primary_metric not in metrics:
        metrics.insert(0, args.primary_metric)
    means, timing_means = group_metric_means(rows, metrics)
    pair_rows = compare_methods(
        means,
        timing_means,
        lower_is_better=not args.higher_is_better,
    )
    summary_rows = summarize_stability(pair_rows)
    compute_rows = sorted(
        [row for row in pair_rows if finite(row.get("compute_budget_seconds"))],
        key=lambda row: (
            str(row.get("metric")),
            float(row.get("compute_budget_seconds")),
            str(row.get("test_set")),
        ),
    )
    recommendation = build_recommendation(
        rows,
        summary_rows,
        pair_rows,
        primary_metric=args.primary_metric,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "winner_by_dataset_size.csv", pair_rows)
    write_rows(args.output_dir / "winner_summary.csv", summary_rows)
    write_rows(args.output_dir / "winner_by_compute_budget.csv", compute_rows)
    (args.output_dir / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(recommendation, ensure_ascii=False, allow_nan=False))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
