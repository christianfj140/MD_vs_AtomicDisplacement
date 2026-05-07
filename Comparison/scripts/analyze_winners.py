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
METHOD_ALIASES = {
    "siesta_fc_cartesian": METHOD_ATOM,
    "atomdisp": METHOD_ATOM,
}
METADATA_COLUMNS = {
    "experiment_id",
    "train_method",
    "test_set",
    "test_method",
    "dataset_size",
    "train_dataset_size",
    "dataset_size_by_method",
    "md_dataset_size",
    "atom_dataset_size",
    "md_dataset_label",
    "atom_dataset_label",
    "compute_budget_mode",
    "md_siesta_reference_count",
    "atomdisp_siesta_reference_count",
    "budget_ratio",
    "budget_mismatch_warning",
    "siesta_settings_hash",
    "siesta_settings_warning",
    "model_config_hash",
    "model_config_warning",
    "basis_pseudopotential_warning",
    "leakage_warning",
    "frozen_test_warning",
    "frozen_test_hash",
    "frozen_test_manifest",
    "checkpoint_manifest",
    "checkpoint_selection_warning",
    "reproducibility_warning",
    "nested_subset_warning",
    "manifest_warning",
    "artifact_hash_warning",
    "matrix_warning",
    "prediction_warning",
    "evaluation_warning",
    "warning_status",
    "severe_warning_status",
    "severe_warnings",
    "strict_comparison_mode",
    "seed",
    "epoch",
    "checkpoint",
    "model_checkpoint",
    "model_checkpoint_sha256",
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
ERROR_METRICS = {
    "global_mae_eV",
    "global_rmse_eV",
    "global_max_abs_error_eV",
    "low_energy_mae_eV",
    "low_energy_rmse_eV",
    "low_energy_max_abs_error_eV",
    "occupied_mae_eV",
    "occupied_rmse_eV",
    "fermi_window_mae_eV",
    "fermi_window_rmse_eV",
    "gap_abs_error_eV",
    "mae_pred_eV",
    "mae_ref_eV",
    "mae_union_eV",
    "rmse_pred_eV",
    "rmse_ref_eV",
    "rmse_union_eV",
    "max_abs_error_union_eV",
    "relative_frobenius_ref",
    "relative_frobenius_union",
    "relative_l1_union",
    "weighted_false_nonzeros_eV",
    "weighted_false_zeros_eV",
    "dos_wasserstein_eV",
    "dos_l1",
    "dos_l2",
}


def parse_value(value: str | None) -> Any:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return number if math.isfinite(number) else None


def canonical_method(value: Any) -> str:
    text = str(value or "")
    return METHOD_ALIASES.get(text, text)


def canonical_test_set(value: Any) -> str:
    text = str(value or "")
    if text == "test_siesta_fc_cartesian":
        return "test_atomdisp"
    return text


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            {key: parse_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    for row in rows:
        if row.get("train_method") not in (None, ""):
            row["train_method"] = canonical_method(row.get("train_method"))
        if row.get("test_set") not in (None, ""):
            row["test_set"] = canonical_test_set(row.get("test_set"))
        if row.get("test_method") == "siesta_fc_cartesian":
            row["test_method"] = METHOD_ATOM
    return rows


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
        for key in ERROR_METRICS:
            value = row.get(key)
            if finite(value):
                columns.add(key)
    return sorted(columns)


def mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def numeric_id(value: Any) -> int | str:
    if finite(value):
        return int(float(value))
    return "unknown"


def row_dataset_size(row: dict[str, Any]) -> int | str:
    return numeric_id(row.get("train_dataset_size", row.get("dataset_size")))


def row_md_dataset_size(row: dict[str, Any]) -> int | str:
    return numeric_id(row.get("md_dataset_size", row.get("dataset_size")))


def row_atom_dataset_size(row: dict[str, Any]) -> int | str:
    return numeric_id(row.get("atom_dataset_size", row.get("dataset_size")))


def row_seed(row: dict[str, Any]) -> int | str:
    value = row.get("seed")
    if finite(value):
        return int(float(value))
    return "unknown"


def row_experiment_id(row: dict[str, Any]) -> str:
    return str(row.get("experiment_id") or "unknown")


def row_epoch(row: dict[str, Any]) -> int | str:
    value = row.get("epoch")
    if finite(value):
        return int(float(value))
    return "unknown"


def row_checkpoint(row: dict[str, Any]) -> str:
    return str(row.get("model_checkpoint") or row.get("checkpoint") or "unknown")


def aggregation_ids(row: dict[str, Any], aggregation_mode: str) -> tuple[str, int | str, int | str, str]:
    if aggregation_mode == "pooled":
        return "pooled", "pooled", "pooled", "pooled"
    if aggregation_mode == "across_seeds":
        return row_experiment_id(row), "across_seeds", row_epoch(row), row_checkpoint(row)
    return row_experiment_id(row), row_seed(row), row_epoch(row), row_checkpoint(row)


def group_metric_means(
    rows: list[dict[str, Any]],
    metrics: list[str],
    *,
    aggregation_mode: str = "per_seed",
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
        experiment_id, seed, epoch, checkpoint = aggregation_ids(row, aggregation_mode)
        for metric in metrics:
            value = row.get(metric)
            if finite(value):
                md_dataset_size = row_md_dataset_size(row)
                atom_dataset_size = row_atom_dataset_size(row)
                values[
                    (
                        experiment_id,
                        md_dataset_size,
                        atom_dataset_size,
                        seed,
                        epoch,
                        test_set,
                        metric,
                        train_method,
                        checkpoint,
                    )
                ].append(float(value))
        for timing_col in timing_columns:
            value = row.get(timing_col)
            if finite(value):
                for metric in metrics:
                    md_dataset_size = row_md_dataset_size(row)
                    atom_dataset_size = row_atom_dataset_size(row)
                    timings[
                        (
                            experiment_id,
                            md_dataset_size,
                            atom_dataset_size,
                            seed,
                            epoch,
                            test_set,
                            metric,
                            train_method,
                            checkpoint,
                        )
                    ][timing_col].append(float(value))

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

    def comparable_sort_key(key: tuple[Any, ...]) -> tuple[str, str, str, str, str, str]:
        experiment_id, md_dataset_size, atom_dataset_size, seed, epoch, test_set, metric = key
        md_sort = f"{int(md_dataset_size):012d}" if isinstance(md_dataset_size, int) else str(md_dataset_size)
        atom_sort = f"{int(atom_dataset_size):012d}" if isinstance(atom_dataset_size, int) else str(atom_dataset_size)
        seed_sort = f"{int(seed):012d}" if isinstance(seed, int) else str(seed)
        return str(experiment_id), md_sort, atom_sort, seed_sort, str(epoch), str(test_set), str(metric)

    comparable = sorted(
        {
            key[:7]
            for key in means
            if key[7] in {METHOD_MD, METHOD_ATOM}
        },
        key=comparable_sort_key,
    )
    for experiment_id, md_dataset_size, atom_dataset_size, seed, epoch, test_set, metric in comparable:
        md_keys = [
            key
            for key in means
            if key[:7] == (experiment_id, md_dataset_size, atom_dataset_size, seed, epoch, test_set, metric)
            and key[7] == METHOD_MD
        ]
        atom_keys = [
            key
            for key in means
            if key[:7] == (experiment_id, md_dataset_size, atom_dataset_size, seed, epoch, test_set, metric)
            and key[7] == METHOD_ATOM
        ]
        if not md_keys or not atom_keys:
            continue
        for md_key in md_keys:
            for atom_key in atom_keys:
                md_mean = means[md_key]
                atom_mean = means[atom_key]
                row = {
                    "experiment_id": experiment_id,
                    "dataset_size": f"md_{md_dataset_size}__atom_{atom_dataset_size}",
                    "md_dataset_size": md_dataset_size,
                    "atom_dataset_size": atom_dataset_size,
                    "seed": seed,
                    "epoch": epoch,
                    "test_set": test_set,
                    "metric": metric,
                    "md_model_checkpoint": md_key[8],
                    "atom_displacement_model_checkpoint": atom_key[8],
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
    grouped: dict[tuple[str, Any, Any, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[
            (
                str(row.get("experiment_id", "unknown")),
                row["md_dataset_size"],
                row["atom_dataset_size"],
                row["test_set"],
                row["metric"],
            )
        ].append(row)

    summary = []
    for (experiment_id, md_dataset_size, atom_dataset_size, test_set, metric), rows in sorted(grouped.items()):
        winners = [str(row["winner"]) for row in rows if row["winner"] != "tie"]
        unique_winners = sorted(set(winners))
        seeds = sorted({row["seed"] for row in rows}, key=str)
        n_seeds = len(seeds)
        stable = len(unique_winners) == 1 and len(rows) > 0 and n_seeds > 1
        single_seed = n_seeds <= 1
        md_wins = sum(1 for winner in winners if winner == METHOD_MD)
        atom_wins = sum(1 for winner in winners if winner == METHOD_ATOM)
        winner_label = unique_winners[0] if (stable or (single_seed and len(unique_winners) == 1)) else "unstable"
        summary.append(
            {
                "experiment_id": experiment_id,
                "dataset_size": f"md_{md_dataset_size}__atom_{atom_dataset_size}",
                "md_dataset_size": md_dataset_size,
                "atom_dataset_size": atom_dataset_size,
                "test_set": test_set,
                "metric": metric,
                "seeds_compared": n_seeds,
                "n_seeds": n_seeds,
                "seeds": ",".join(str(seed) for seed in seeds),
                "md_wins": md_wins,
                "atom_displacement_wins": atom_wins,
                "wins_md": md_wins,
                "wins_atomdisp": atom_wins,
                "ties": sum(1 for row in rows if row["winner"] == "tie"),
                "stable_across_seeds": stable,
                "stable_winner": unique_winners[0] if stable else "unstable",
                "winner": winner_label,
                "winner_stability": "stable" if stable else ("single_seed" if single_seed else "unstable"),
                "single_seed_warning": single_seed,
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


def first_method_win(
    summary_rows: list[dict[str, Any]],
    metric: str,
    method: str,
    *,
    allowed_test_sets: set[str],
    minimum_robust_seeds: int,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in summary_rows
        if row.get("metric") == metric
        and row.get("test_set") in allowed_test_sets
        and row.get("stable_winner") == method
        and int(row.get("n_seeds") or row.get("seeds_compared") or 0) >= minimum_robust_seeds
    ]
    candidates.sort(
        key=lambda row: (
            float(row["md_dataset_size"]) if finite(row.get("md_dataset_size")) else math.inf,
            float(row["atom_dataset_size"]) if finite(row.get("atom_dataset_size")) else math.inf,
            str(row["test_set"]),
        )
    )
    return candidates[0] if candidates else None


def first_compute_method_win(
    pair_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    metric: str,
    method: str,
    *,
    allowed_test_sets: set[str],
    minimum_robust_seeds: int,
) -> dict[str, Any] | None:
    robust_groups = {
        (
            str(row.get("experiment_id", "unknown")),
            row.get("md_dataset_size"),
            row.get("atom_dataset_size"),
            row.get("test_set"),
        )
        for row in summary_rows
        if row.get("metric") == metric
        and row.get("test_set") in allowed_test_sets
        and row.get("stable_winner") == method
        and int(row.get("n_seeds") or row.get("seeds_compared") or 0) >= minimum_robust_seeds
    }
    candidates = [
        row
        for row in pair_rows
        if row.get("metric") == metric
        and row.get("test_set") in allowed_test_sets
        and row.get("winner") == method
        and finite(row.get("compute_budget_seconds"))
        and (
            str(row.get("experiment_id", "unknown")),
            row.get("md_dataset_size"),
            row.get("atom_dataset_size"),
            row.get("test_set"),
        )
        in robust_groups
    ]
    candidates.sort(
        key=lambda row: (
            float(row["compute_budget_seconds"]),
            float(row["md_dataset_size"]) if finite(row.get("md_dataset_size")) else math.inf,
            float(row["atom_dataset_size"]) if finite(row.get("atom_dataset_size")) else math.inf,
        )
    )
    return candidates[0] if candidates else None


def build_recommendation(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    *,
    primary_metric: str,
    minimum_robust_seeds: int = 3,
) -> dict[str, Any]:
    test_sets = sorted({str(row.get("test_set")) for row in rows})
    methods = sorted({str(row.get("train_method")) for row in rows})
    if len(methods) < 2:
        return {
            "status": "non_comparative",
            "scientific_status": "non_comparative",
            "primary_metric": primary_metric,
            "reason": "Only one training method is present; no scientific winner is emitted.",
            "methods_seen": methods,
            "test_sets_seen": test_sets,
            "missing_required_cells": [],
            "missing_primary_metric_cells": [],
            "severe_warnings": [],
            "n_seeds": len({row_seed(row) for row in rows}),
            "minimum_robust_seeds": minimum_robust_seeds,
        }
    frozen_test_hashes = sorted(
        {
            str(row.get("frozen_test_hash"))
            for row in rows
            if row.get("frozen_test_hash") not in (None, "", False)
        }
    )
    model_checkpoint_hashes = sorted(
        {
            str(row.get("model_checkpoint_sha256"))
            for row in rows
            if row.get("model_checkpoint_sha256") not in (None, "", False)
        }
    )
    required_test_sets = set(test_sets)
    required_cells = {
        (method, test_set)
        for method in methods
        for test_set in required_test_sets
    }
    available_cells = {
        (str(row.get("train_method")), str(row.get("test_set")))
        for row in rows
    }
    missing_cells = sorted(f"{method} on {test_set}" for method, test_set in required_cells - available_cells)
    primary_metric_cells = {
        (str(row.get("train_method")), str(row.get("test_set")))
        for row in rows
        if finite(row.get(primary_metric))
    }
    missing_primary_metric_cells = sorted(
        f"{method} on {test_set}"
        for method, test_set in required_cells - primary_metric_cells
    )
    severe_warnings = sorted(
        {
            str(value)
            for row in rows
            for value in (
                row.get("siesta_settings_warning"),
                row.get("model_config_warning"),
                row.get("basis_pseudopotential_warning"),
                row.get("budget_mismatch_warning"),
                row.get("leakage_warning"),
                row.get("frozen_test_warning"),
                row.get("checkpoint_selection_warning"),
                row.get("reproducibility_warning"),
                row.get("nested_subset_warning"),
                row.get("manifest_warning"),
                row.get("artifact_hash_warning"),
                row.get("matrix_warning"),
                row.get("prediction_warning"),
                row.get("evaluation_warning"),
                row.get("severe_warnings"),
            )
            if value not in (None, "", False)
        }
    )
    allowed_test_sets = set(test_sets)
    primary_summary = [
        row
        for row in summary_rows
        if row.get("metric") == primary_metric and row.get("test_set") in allowed_test_sets
    ]
    max_seeds = max(
        (int(row.get("n_seeds") or row.get("seeds_compared") or 0) for row in primary_summary),
        default=len({row_seed(row) for row in rows}),
    )
    single_seed_only = max_seeds <= 1
    insufficient_robust_seeds = max_seeds < minimum_robust_seeds
    first_atom_size = first_method_win(
        summary_rows,
        primary_metric,
        METHOD_ATOM,
        allowed_test_sets=allowed_test_sets,
        minimum_robust_seeds=minimum_robust_seeds,
    )
    first_md_size = first_method_win(
        summary_rows,
        primary_metric,
        METHOD_MD,
        allowed_test_sets=allowed_test_sets,
        minimum_robust_seeds=minimum_robust_seeds,
    )
    first_atom_compute = first_compute_method_win(
        [
            row
            for row in pair_rows
            if row.get("metric") == primary_metric
        ],
        summary_rows,
        primary_metric,
        METHOD_ATOM,
        allowed_test_sets=allowed_test_sets,
        minimum_robust_seeds=minimum_robust_seeds,
    )
    first_md_compute = first_compute_method_win(
        [
            row
            for row in pair_rows
            if row.get("metric") == primary_metric
        ],
        summary_rows,
        primary_metric,
        METHOD_MD,
        allowed_test_sets=allowed_test_sets,
        minimum_robust_seeds=minimum_robust_seeds,
    )
    atom_only_wins = [
        row
        for row in summary_rows
        if row.get("metric") == primary_metric
        and row.get("test_set") == "test_atomdisp"
        and row.get("stable_winner") == METHOD_ATOM
    ]
    underpowered_stable_wins = [
        row
        for row in summary_rows
        if row.get("metric") == primary_metric
        and row.get("test_set") in allowed_test_sets
        and row.get("stable_winner") in {METHOD_MD, METHOD_ATOM}
        and int(row.get("n_seeds") or row.get("seeds_compared") or 0) < minimum_robust_seeds
    ]
    available_primary_wins = [
        row
        for row in pair_rows
        if row.get("metric") == primary_metric
        and row.get("test_set") in allowed_test_sets
        and row.get("winner") in {METHOD_MD, METHOD_ATOM}
    ]
    available_md_wins = [row for row in available_primary_wins if row.get("winner") == METHOD_MD]
    available_atom_wins = [row for row in available_primary_wins if row.get("winner") == METHOD_ATOM]
    status = "no_conservative_winner"
    reason = "No method wins stably on MD or mixed test distributions for the primary metric."
    if first_md_size and not first_atom_size:
        status = "md_conservative_win"
        reason = (
            "MD beats AtomDisplacement on the same frozen test set across seeds "
            "for MD or mixed test distributions."
        )
    elif first_atom_size and not first_md_size:
        status = "atom_displacement_conservative_win"
        reason = (
            "AtomDisplacement beats MD on the same frozen test set across seeds "
            "and not only on the AtomDisplacement distribution."
        )
    elif first_md_size and first_atom_size:
        status = "mixed_conservative_result"
        reason = "Both methods win stably on at least one allowed test distribution."
    elif atom_only_wins:
        reason = (
            "AtomDisplacement only wins on test_atomdisp; this is distribution-specific "
            "and is not enough for a conservative recommendation."
        )
    nway_methods = set(methods) - {METHOD_MD, METHOD_ATOM}
    if missing_cells:
        status = "insufficient_cross_evaluation"
        reason = "The cross-evaluation grid is incomplete; no winner should be declared."
    elif missing_primary_metric_cells:
        status = "insufficient_primary_metric"
        reason = (
            f"The primary metric {primary_metric!r} is missing in required cells; "
            "no winner should be declared."
        )
    elif severe_warnings:
        status = "inconclusive"
        reason = "Validation warnings invalidate a robust winner: " + " | ".join(severe_warnings)
    elif single_seed_only and available_primary_wins:
        if available_md_wins and not available_atom_wins:
            status = "md_available_single_seed_win"
            reason = (
                "MD wins on the available frozen test set, but only one seed is available; "
                "this is not a robust conclusion."
            )
        elif available_atom_wins and not available_md_wins:
            status = "atom_displacement_available_single_seed_win"
            reason = (
                "AtomDisplacement wins on the available frozen test set, but only one seed is available; "
                "this is not a robust conclusion."
            )
        else:
            status = "mixed_available_single_seed_result"
            reason = "Single-seed winners disagree across allowed test distributions; no robust winner."
    elif nway_methods and not missing_cells and not missing_primary_metric_cells and not severe_warnings:
        status = "nway_complete_no_global_winner"
        reason = (
            "N-way comparison cells are complete, but the conservative legacy "
            "global-winner rule is not yet reduced to one robust winner."
        )

    robust_statuses = {"md_conservative_win", "atom_displacement_conservative_win", "mixed_conservative_result"}
    if status in robust_statuses and not insufficient_robust_seeds and not underpowered_stable_wins:
        scientific_status = "robust_comparison"
    elif not missing_cells and not missing_primary_metric_cells and not severe_warnings and insufficient_robust_seeds:
        scientific_status = "exploratory"
    elif (
        (insufficient_robust_seeds or bool(underpowered_stable_wins))
        and not missing_cells
        and not missing_primary_metric_cells
        and not severe_warnings
        and (status in robust_statuses or "single_seed" in status or bool(underpowered_stable_wins))
    ):
        scientific_status = "exploratory"
    else:
        scientific_status = "scientifically_inconclusive"

    rankings_by_test_set: dict[str, list[dict[str, Any]]] = {}
    for test_set in test_sets:
        ranking_rows = []
        for method in methods:
            values = [
                float(row[primary_metric])
                for row in rows
                if str(row.get("train_method")) == method
                and str(row.get("test_set")) == test_set
                and finite(row.get(primary_metric))
            ]
            if values:
                ranking_rows.append({"method": method, "mean": mean(values), "n": len(values)})
        rankings_by_test_set[test_set] = sorted(
            ranking_rows,
            key=lambda item: float(item["mean"]) if item["mean"] is not None else math.inf,
        )

    return {
        "status": status,
        "scientific_status": scientific_status,
        "primary_metric": primary_metric,
        "reason": reason,
        "n_seeds": max_seeds,
        "minimum_robust_seeds": minimum_robust_seeds,
        "single_seed_warning": single_seed_only,
        "insufficient_robust_seeds": insufficient_robust_seeds,
        "underpowered_stable_wins": underpowered_stable_wins,
        "methods_seen": methods,
        "test_sets_seen": test_sets,
        "frozen_test_hashes": frozen_test_hashes,
        "model_checkpoint_hashes": model_checkpoint_hashes,
        "missing_required_cells": missing_cells,
        "missing_primary_metric_cells": missing_primary_metric_cells,
        "severe_warnings": severe_warnings,
        "rankings_by_test_set": rankings_by_test_set,
        "first_dataset_size_where_md_beats_atom_displacement": first_md_size,
        "first_dataset_size_where_atom_displacement_beats_md": first_atom_size,
        "first_compute_budget_where_md_beats_atom_displacement": first_md_compute,
        "first_compute_budget_where_atom_displacement_beats_md": first_atom_compute,
        "conservative_rule": (
            "A method wins only if it beats the other method on the same frozen test set, "
            "across seeds, and on test_md or test_mixed rather than only its own distribution."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-metric", default="fermi_window_rmse_eV")
    parser.add_argument("--minimum-robust-seeds", type=int, default=3)
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument(
        "--aggregation-mode",
        choices=["per_seed", "across_seeds", "pooled"],
        default="per_seed",
        help="Default preserves experiment/seed/checkpoint. Use pooled only for explicit exploratory aggregation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_rows(args.metrics_csv)
    metrics = metric_columns(rows)
    if args.primary_metric not in metrics:
        metrics.insert(0, args.primary_metric)
    means, timing_means = group_metric_means(rows, metrics, aggregation_mode=args.aggregation_mode)
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
        minimum_robust_seeds=args.minimum_robust_seeds,
    )
    recommendation["aggregation_mode"] = args.aggregation_mode
    for row in pair_rows:
        row["aggregation_mode"] = args.aggregation_mode
    for row in summary_rows:
        row["aggregation_mode"] = args.aggregation_mode

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
