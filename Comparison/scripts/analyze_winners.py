#!/usr/bin/env python3
"""Analyze MD-vs-AtomDisplacement cross-evaluation winners conservatively.

Examples
--------
    python Comparison/scripts/analyze_winners.py \
      --metrics-csv Comparison/results/exp_001/summary/cross_evaluation_metrics.csv \
      --output-dir Comparison/results/exp_001/summary \
      --primary-metric low_energy_rmse_eV
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from method_registry import normalize_method_id, normalize_test_set_id


METHOD_MD = "md"
METHOD_ATOM = "siesta_fc_cartesian"
DEFAULT_BINARY_TEST_SETS = {"test_md", "test_siesta_fc_cartesian", "test_mixed"}
METHOD_CONTEXT_FIELDS = (
    "dataset_size",
    "dataset_label",
    "recipe_hash",
    "result_dir",
    "training_tag",
    "training_plan_label",
    "training_plan_settings",
)
TIMING_FIELDS = (
    "total_time_seconds",
    "siesta_time_seconds",
    "training_time_seconds",
    "prediction_time_seconds",
    "evaluation_time_seconds",
)
RELIABLE_COMPUTE_TIMING_FIELD = "total_time_seconds"
NON_STABILITY_SEED_IDS = {"", "unknown", "pooled", "across_seeds"}
REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_COLUMNS = {
    "experiment_id",
    "train_method",
    "test_set",
    "test_method",
    "dataset_size",
    "train_dataset_size",
    "dataset_size_by_method",
    "dataset_label_by_method",
    "recipe_id_by_method",
    "recipe_label_by_method",
    "recipe_set_hash_by_method",
    "training_tag_by_method",
    "training_plan_label_by_method",
    "training_plan_settings_by_method",
    "md_dataset_size",
    "atom_dataset_size",
    "md_dataset_label",
    "atom_dataset_label",
    "random_dataset_size",
    "random_dataset_label",
    "train_dataset_label",
    "train_training_tag",
    "train_training_index",
    "train_training_settings",
    "train_training_plan_index",
    "train_training_plan_label",
    "train_training_plan_settings",
    "train_training_plan_source_dataset_label",
    "training_tag",
    "training_index",
    "training_settings",
    "training_plan_index",
    "training_plan_label",
    "training_plan_settings",
    "training_plan_source_dataset_label",
    "compute_budget_mode",
    "md_siesta_reference_count",
    "atomdisp_siesta_reference_count",
    "budget_ratio",
    "budget_mismatch_warning",
    "siesta_settings_hash",
    "siesta_settings_warning",
    "model_config_hash",
    "model_config_warning",
    "training_plan_settings_warning",
    "basis_pseudopotential_warning",
    "material_compatibility_warning",
    "material_label",
    "material_bundle_path",
    "material_source",
    "material_preset",
    "material_structure_type",
    "material_species",
    "material_atom_count",
    "material_cell_summary",
    "fdf_sha256",
    "pseudopotential_sha256_by_species",
    "basis_sha256_by_species",
    "siesta_output_flags",
    "graph2mat_config_hash",
    "split_manifest_hash",
    "dataset_recipe",
    "dataset_recipe_parameters",
    "reference_matrix_sha256",
    "prediction_matrix_sha256",
    "material_identity_hash",
    "material_compatibility_hash",
    "material_label_by_method",
    "material_identity_hash_by_method",
    "material_compatibility_hash_by_method",
    "fdf_sha256_by_method",
    "pseudopotential_sha256_by_method",
    "basis_sha256_by_method",
    "leakage_warning",
    "leakage_scientific_status",
    "leakage_severe_warnings",
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
    "frontier_window_mae_eV",
    "frontier_window_rmse_eV",
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

RECOMMENDATION_PRIMARY_METRIC_PRIORITY = [
    "low_energy_rmse_eV",
    "frontier_window_rmse_eV",
    "occupied_rmse_eV",
    "relative_frobenius_union",
    "dos_wasserstein_eV",
]
RECOMMENDATION_GRADE_PRIMARY_METRICS = {
    "low_energy_rmse_eV",
    "relative_frobenius_union",
    "dos_wasserstein_eV",
}
CONDITIONAL_PRIMARY_METRICS = {
    "frontier_window_rmse_eV",
    "frontier_window_mae_eV",
    "occupied_rmse_eV",
    "occupied_mae_eV",
}
FERMI_WINDOW_PRIMARY_METRICS = {
    "fermi_window_rmse_eV",
    "fermi_window_mae_eV",
    "align_fermi_rmse_eV",
    "align_fermi_mae_eV",
}
DIAGNOSTIC_ONLY_PRIMARY_METRICS = {
    "global_rmse_eV",
    "global_mae_eV",
    "global_max_abs_error_eV",
    "global_mean_signed_error_eV",
    "support_precision",
    "support_recall",
    "support_f1",
    "false_zeros",
    "false_nonzeros",
    "weighted_false_zeros_eV",
    "weighted_false_nonzeros_eV",
    "hermiticity_error",
    "antihermitian_norm",
}
METRIC_POLICY_JUSTIFICATION_COLUMNS = (
    "primary_metric_justification",
    "metric_policy_justification",
    "fermi_window_metric_justification",
    "frontier_metric_justification",
)


def parse_value(value: str | None) -> Any:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return number if math.isfinite(number) else None


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def sanitize_reproducibility_warning(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    prefix = "User-local absolute paths detected:"
    if not value.startswith(prefix):
        return value
    paths = [Path(part.strip()) for part in value.removeprefix(prefix).split(",") if part.strip()]
    external = [
        str(path)
        for path in paths
        if path.is_absolute() and not path_is_relative_to(path, REPO_ROOT)
    ]
    return f"{prefix} {', '.join(sorted(dict.fromkeys(external)))}" if external else None


def sanitize_warning_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    parts = [
        sanitized
        for sanitized in (
            sanitize_reproducibility_warning(part.strip())
            for part in value.split(" | ")
            if part.strip()
        )
        if sanitized
    ]
    return " | ".join(parts) if parts else None


def derive_frontier_metrics(row: dict[str, Any]) -> None:
    if finite_number(row.get("frontier_window_rmse_eV")) is not None:
        return
    errors = [
        value
        for value in (
            finite_number(row.get("homo_error_eV")),
            finite_number(row.get("lumo_error_eV")),
        )
        if value is not None
    ]
    if not errors:
        return
    row["frontier_window_bands"] = len(errors)
    row["frontier_window_mae_eV"] = sum(abs(value) for value in errors) / len(errors)
    row["frontier_window_rmse_eV"] = math.sqrt(
        sum(value * value for value in errors) / len(errors)
    )


def canonical_method(value: Any) -> str:
    return normalize_method_id(value, allow_unknown=True)


def canonical_test_set(value: Any) -> str:
    return normalize_test_set_id(value)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            {key: parse_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    for row in rows:
        row["reproducibility_warning"] = sanitize_reproducibility_warning(
            row.get("reproducibility_warning")
        )
        row["severe_warnings"] = sanitize_warning_text(row.get("severe_warnings"))
        derive_frontier_metrics(row)
        if row.get("train_method") not in (None, ""):
            row["train_method"] = canonical_method(row.get("train_method"))
        if row.get("test_set") not in (None, ""):
            row["test_set"] = canonical_test_set(row.get("test_set"))
        if row.get("test_method") not in (None, ""):
            row["test_method"] = canonical_method(row.get("test_method"))
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


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_completeness_report(
    path: Path | None,
    output_dir: Path,
) -> tuple[dict[str, Any], Path | None]:
    report_path = path or output_dir / "cross_evaluation_completeness.json"
    if report_path.exists():
        return read_json(report_path), report_path
    return {}, None


def completeness_report_is_invalid(report: dict[str, Any]) -> bool:
    if not report:
        return False
    return report.get("scientific_status") == "invalid_incomplete_grid" or report.get("complete") is False


def invalid_grid_recommendation(
    report: dict[str, Any],
    report_path: Path | None,
    *,
    primary_metric: str,
    base_recommendation: dict[str, Any],
) -> dict[str, Any]:
    missing_required = [
        *(str(cell) for cell in report.get("missing_cells") or []),
        *(str(cell) for cell in report.get("missing_context_cells") or []),
    ]
    missing_primary = [
        *(str(cell) for cell in report.get("missing_primary_metric_cells") or []),
        *(str(cell) for cell in report.get("missing_primary_metric_context_cells") or []),
    ]
    missing_cells = sorted(dict.fromkeys([*missing_required, *missing_primary]))
    recommendation = {
        "status": "invalid_incomplete_grid",
        "scientific_status": "invalid_incomplete_grid",
        "winner": None,
        "primary_metric": primary_metric,
        "reason": "Incomplete cross-evaluation grid",
        "missing_cells": missing_cells,
        "missing_required_cells": missing_required,
        "missing_primary_metric_cells": missing_primary,
        "extra_unexpected_cells": [
            *(report.get("extra_unexpected_cells") or report.get("extra_cells") or []),
            *(report.get("extra_unexpected_context_cells") or report.get("extra_context_cells") or []),
        ],
        "complete": False,
        "cross_evaluation_completeness": report,
        "blocked_recommendation_status": base_recommendation.get("status"),
        "blocked_recommendation_scientific_status": base_recommendation.get("scientific_status"),
        "metric_policy": base_recommendation.get("metric_policy"),
        "primary_metric_policy_status": base_recommendation.get("primary_metric_policy_status"),
        "primary_metric_recommendation_role": base_recommendation.get("primary_metric_recommendation_role"),
        "robust_primary_metric_allowed": base_recommendation.get("robust_primary_metric_allowed"),
    }
    if report_path:
        recommendation["completeness_report"] = str(report_path)
    return recommendation


def warning_items(value: Any) -> list[str]:
    if value in (None, "", False):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "", False)]
    if isinstance(value, dict):
        return [json.dumps(value, sort_keys=True, ensure_ascii=False)]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(item) for item in parsed if item not in (None, "", False)]
    if isinstance(parsed, dict):
        return [json.dumps(parsed, sort_keys=True, ensure_ascii=False)]
    return [part.strip() for part in text.split(" | ") if part.strip()]


LEAKAGE_MARKERS = {
    "geometry leakage",
    "leakage",
    "exact_duplicate_geometry",
    "near_duplicate_geometry",
    "aligned_near_duplicate_geometry",
    "internal_distance_near_duplicate_geometry",
    "md_neighboring_frames_cross_split",
    "atom_displacement_same_family_cross_split",
    "random_cartesian_same_family_cross_split",
    "invalid_exact_geometry_leakage",
    "invalid_leakage",
    "potential_geometry_leakage",
    "scientifically_non_independent_splits",
}


def is_leakage_warning(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in LEAKAGE_MARKERS)


def leakage_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: set[str] = set()
    messages: set[str] = set()
    for row in rows:
        status = row.get("leakage_scientific_status")
        if status not in (None, "", False):
            statuses.add(str(status))
        for key in ("leakage_warning", "leakage_severe_warnings", "severe_warnings"):
            for item in warning_items(row.get(key)):
                if is_leakage_warning(item):
                    messages.add(item)
    blob = " | ".join(sorted([*statuses, *messages])).lower()
    invalid = (
        "invalid_leakage" in statuses
        or "invalid_exact_geometry_leakage" in blob
        or "exact_duplicate_geometry" in blob
    )
    inconclusive = (
        "scientifically_inconclusive" in statuses
        or "potential_geometry_leakage" in blob
        or "near_duplicate_geometry" in blob
        or "aligned_near_duplicate_geometry" in blob
        or "internal_distance_near_duplicate_geometry" in blob
        or "md_neighboring_frames_cross_split" in blob
        or "atom_displacement_same_family_cross_split" in blob
    )
    exploratory_only = (
        "exploratory_only" in statuses
        or "random_cartesian_same_family_cross_split" in blob
        or "random_cartesian_family_warnings" in blob
    )
    clean = bool(statuses) and statuses <= {"valid_independent_splits"} and not messages
    return {
        "statuses": sorted(statuses),
        "messages": sorted(messages),
        "invalid": invalid,
        "inconclusive": inconclusive,
        "exploratory_only": exploratory_only and not invalid and not inconclusive,
        "clean": clean,
    }


def truthy_policy_value(value: Any) -> bool:
    if value in (None, "", False):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "no", "none", "null", "unjustified"}


def metric_policy_justified(rows: list[dict[str, Any]]) -> bool:
    return any(
        truthy_policy_value(row.get(column))
        for row in rows
        for column in METRIC_POLICY_JUSTIFICATION_COLUMNS
    )


def metric_policy_role(metric: str) -> str:
    if metric in RECOMMENDATION_GRADE_PRIMARY_METRICS:
        return "recommendation_grade"
    if metric in CONDITIONAL_PRIMARY_METRICS:
        return "conditional_recommendation_grade"
    if metric in FERMI_WINDOW_PRIMARY_METRICS:
        return "fermi_window_conditional"
    if metric in DIAGNOSTIC_ONLY_PRIMARY_METRICS:
        return "diagnostic_only"
    if metric in ERROR_METRICS:
        return "unranked_error_metric"
    return "unknown_metric"


def metric_policy_diagnostics(
    rows: list[dict[str, Any]],
    *,
    primary_metric: str,
    required_cells: set[tuple[str, str]],
    primary_metric_cells: set[tuple[str, str]],
    missing_primary_metric_cells: list[str],
) -> dict[str, Any]:
    role = metric_policy_role(primary_metric)
    priority_rank = (
        RECOMMENDATION_PRIMARY_METRIC_PRIORITY.index(primary_metric) + 1
        if primary_metric in RECOMMENDATION_PRIMARY_METRIC_PRIORITY
        else None
    )
    cells_with_metric = sorted(
        f"{method} on {test_set}"
        for method, test_set in primary_metric_cells
    )
    fermi_sources = sorted(
        {
            str(row.get("fermi_level_source"))
            for row in rows
            if row.get("fermi_level_source") not in (None, "", False)
        }
    )
    metric_rows = [row for row in rows if finite(row.get(primary_metric))]
    occupied_band_counts = [
        int(float(row.get("occupied_bands")))
        for row in metric_rows
        if finite(row.get("occupied_bands"))
    ]
    occupied_well_defined = (
        bool(metric_rows)
        and len(occupied_band_counts) == len(metric_rows)
        and all(count > 0 for count in occupied_band_counts)
    )
    justified = metric_policy_justified(rows)

    if missing_primary_metric_cells:
        status = "missing_required_metric"
        robust_allowed = False
        reason = (
            f"The selected primary metric {primary_metric!r} is unavailable in at least one "
            "required train-method/test-set cell."
        )
    elif role == "diagnostic_only":
        status = "diagnostic_only_metric"
        robust_allowed = False
        reason = (
            f"{primary_metric!r} is a diagnostic metric. It can be reported, but it cannot "
            "support a robust scientific recommendation by itself."
        )
    elif role == "fermi_window_conditional" and not justified:
        status = "fermi_window_metric_unjustified"
        robust_allowed = False
        reason = (
            "Fermi-window metrics are conditional for isolated or fragile spectra; complete "
            "availability and an explicit justification are required before robustness can be claimed."
        )
    elif primary_metric.startswith("occupied_") and not occupied_well_defined:
        status = "occupied_metric_not_well_defined"
        robust_allowed = False
        reason = (
            "Occupied-state metrics require occupied bands to be well-defined in every metric row."
        )
    elif role == "unranked_error_metric":
        status = "unranked_error_metric"
        robust_allowed = False
        reason = (
            f"{primary_metric!r} is an error metric, but it is not in the recommendation primary "
            "metric policy; treat any conclusion as exploratory."
        )
    elif role == "unknown_metric":
        status = "unknown_metric"
        robust_allowed = False
        reason = (
            f"{primary_metric!r} is not recognized by the recommendation metric policy."
        )
    else:
        status = "recommendation_metric"
        robust_allowed = True
        reason = "Primary metric satisfies the recommendation metric policy."

    return {
        "primary_metric": primary_metric,
        "policy_status": status,
        "recommendation_role": role,
        "priority": RECOMMENDATION_PRIMARY_METRIC_PRIORITY,
        "priority_rank": priority_rank,
        "robust_recommendation_allowed": robust_allowed,
        "required_cell_count": len(required_cells),
        "cells_with_primary_metric": cells_with_metric,
        "missing_required_cells": missing_primary_metric_cells,
        "missing_required_cell_count": len(missing_primary_metric_cells),
        "fermi_level_sources": fermi_sources,
        "fermi_window_complete_and_justified": (
            primary_metric in FERMI_WINDOW_PRIMARY_METRICS
            and not missing_primary_metric_cells
            and justified
        ),
        "occupied_well_defined": occupied_well_defined if primary_metric.startswith("occupied_") else None,
        "justification_present": justified,
        "reason": reason,
    }


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


def stddev(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) < 2:
        return None
    avg = sum(clean) / len(clean)
    return math.sqrt(sum((value - avg) ** 2 for value in clean) / (len(clean) - 1))


def normalized_seed_label(seed: Any) -> str:
    if seed in (None, ""):
        return "unknown"
    if finite(seed):
        number = float(seed)
        return str(int(number)) if number.is_integer() else str(number)
    return str(seed)


def valid_stability_seeds(seeds: list[Any] | set[Any]) -> list[str]:
    return sorted(
        {
            label
            for seed in seeds
            if (label := normalized_seed_label(seed)) not in NON_STABILITY_SEED_IDS
        },
        key=str,
    )


def seed_stability_status(
    seeds: list[Any] | set[Any],
    outcomes: list[str],
    *,
    has_severe_warning: bool = False,
    minimum_robust_seeds: int = 3,
) -> str:
    valid_seeds = valid_stability_seeds(seeds)
    unique_outcomes = sorted({outcome for outcome in outcomes if outcome})
    if has_severe_warning:
        return "severe_warning_not_robust"
    if len(unique_outcomes) > 1 and len(valid_seeds) > 1:
        return "unstable"
    if len(valid_seeds) < minimum_robust_seeds:
        return "exploratory_only"
    if len(unique_outcomes) != 1:
        return "unstable"
    return "robust_candidate"


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


def parse_method_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        raw = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        raw = parsed if isinstance(parsed, dict) else {}
    else:
        raw = {}
    result: dict[str, Any] = {}
    for method, mapped_value in raw.items():
        result[canonical_method(method)] = mapped_value
    return result


def method_mapping_json(row: dict[str, Any], column: str) -> str | None:
    mapping = parse_method_mapping(row.get(column))
    if not mapping:
        return None
    return json.dumps(mapping, ensure_ascii=False, sort_keys=True)


def normalize_context_value(value: Any) -> Any:
    if value in (None, "", False):
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if finite(value):
        number = float(value)
        return int(number) if number.is_integer() else number
    return str(value)


def legacy_method_columns(field: str, method: str) -> list[str]:
    if method == METHOD_MD:
        prefixes = ["md"]
    elif method == METHOD_ATOM:
        prefixes = ["atom", "atom_displacement", "atomdisp", "siesta_fc_cartesian"]
    elif method == "random_cartesian":
        prefixes = ["random_cartesian", "rc"]
    else:
        prefixes = [method]
    return [f"{prefix}_{field}" for prefix in prefixes]


def method_context_mapping(row: dict[str, Any], field: str) -> dict[str, Any]:
    mapping = {
        method: normalized
        for method, value in parse_method_mapping(row.get(f"{field}_by_method")).items()
        if (normalized := normalize_context_value(value)) is not None
    }
    candidate_methods = set(mapping)
    for key in row:
        if key.endswith(f"_{field}"):
            prefix = key[: -(len(field) + 1)]
            if prefix in {"md", "atom", "atom_displacement", "atomdisp", "siesta_fc_cartesian", "random_cartesian", "rc"}:
                candidate_methods.add(canonical_method(prefix))
    if row.get("train_method") not in (None, ""):
        candidate_methods.add(str(row.get("train_method")))

    for method in sorted(candidate_methods):
        if method in mapping:
            continue
        for column in legacy_method_columns(field, method):
            value = normalize_context_value(row.get(column))
            if value is not None:
                mapping[method] = value
                break

    train_method = str(row.get("train_method") or "")
    if train_method and train_method not in mapping:
        for column in (field, f"train_{field}"):
            value = normalize_context_value(row.get(column))
            if value is not None:
                mapping[train_method] = value
                break

    if field == "dataset_size":
        if METHOD_MD not in mapping and row_md_dataset_size(row) != "unknown":
            mapping[METHOD_MD] = row_md_dataset_size(row)
        if METHOD_ATOM not in mapping and row_atom_dataset_size(row) != "unknown":
            mapping[METHOD_ATOM] = row_atom_dataset_size(row)
        if train_method and train_method not in mapping and row_dataset_size(row) != "unknown":
            mapping[train_method] = row_dataset_size(row)

    return dict(sorted(mapping.items()))


def method_context_mapping_json(row: dict[str, Any], field: str) -> str | None:
    mapping = method_context_mapping(row, field)
    if not mapping:
        return None
    return json.dumps(mapping, ensure_ascii=False, sort_keys=True)


def dataset_context(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    context = {
        field: mapping
        for field in METHOD_CONTEXT_FIELDS
        if (mapping := method_context_mapping(row, field))
    }
    return dict(sorted(context.items()))


def dataset_context_key(row: dict[str, Any]) -> str:
    return json.dumps(dataset_context(row), ensure_ascii=False, sort_keys=True)


def mapped_or_legacy_method_value(row: dict[str, Any], field: str, method: str) -> Any:
    mapping = parse_method_mapping(row.get(f"{field}_by_method"))
    if method in mapping:
        return mapping[method]
    if method == METHOD_MD:
        return row.get(f"md_{field}")
    if method == METHOD_ATOM:
        return row.get(f"atom_{field}") or row.get(f"atom_displacement_{field}")
    if str(row.get("train_method")) == method:
        return row.get(field) or row.get(f"train_{field}")
    return None


def method_timing_value(row: dict[str, Any], timing_field: str, method: str) -> float | None:
    for column in legacy_method_columns(timing_field, method):
        value = row.get(column)
        if finite(value):
            return float(value)
    if str(row.get("train_method") or "") == method:
        value = row.get(timing_field)
        if finite(value):
            return float(value)
    return None


def row_method_dataset_size(row: dict[str, Any], method: str) -> int | str:
    mapping = method_context_mapping(row, "dataset_size")
    if method in mapping:
        return numeric_id(parse_value(str(mapping[method])) if isinstance(mapping[method], str) else mapping[method])
    if method == METHOD_MD:
        return row_md_dataset_size(row)
    if method == METHOD_ATOM:
        return row_atom_dataset_size(row)
    if str(row.get("train_method")) == method:
        return row_dataset_size(row)
    return "unknown"


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
        context_key = dataset_context_key(row)
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
                        context_key,
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
                            context_key,
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

    def comparable_sort_key(key: tuple[Any, ...]) -> tuple[str, str, str, str, str, str, str, str]:
        experiment_id, md_dataset_size, atom_dataset_size, context_key, seed, epoch, test_set, metric = key
        md_sort = f"{int(md_dataset_size):012d}" if isinstance(md_dataset_size, int) else str(md_dataset_size)
        atom_sort = f"{int(atom_dataset_size):012d}" if isinstance(atom_dataset_size, int) else str(atom_dataset_size)
        seed_sort = f"{int(seed):012d}" if isinstance(seed, int) else str(seed)
        return str(experiment_id), md_sort, atom_sort, str(context_key), seed_sort, str(epoch), str(test_set), str(metric)

    comparable = sorted(
        {
            key[:8]
            for key in means
            if key[8] in {METHOD_MD, METHOD_ATOM}
        },
        key=comparable_sort_key,
    )
    for experiment_id, md_dataset_size, atom_dataset_size, context_key, seed, epoch, test_set, metric in comparable:
        md_keys = [
            key
            for key in means
            if key[:8] == (
                experiment_id,
                md_dataset_size,
                atom_dataset_size,
                context_key,
                seed,
                epoch,
                test_set,
                metric,
            )
            and key[8] == METHOD_MD
        ]
        atom_keys = [
            key
            for key in means
            if key[:8] == (
                experiment_id,
                md_dataset_size,
                atom_dataset_size,
                context_key,
                seed,
                epoch,
                test_set,
                metric,
            )
            and key[8] == METHOD_ATOM
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
                    "dataset_context_key": context_key,
                    "seed": seed,
                    "epoch": epoch,
                    "test_set": test_set,
                    "metric": metric,
                    "md_model_checkpoint": md_key[9],
                    "atom_displacement_model_checkpoint": atom_key[9],
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
    grouped: dict[tuple[str, Any, Any, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[
            (
                str(row.get("experiment_id", "unknown")),
                row["md_dataset_size"],
                row["atom_dataset_size"],
                str(row.get("dataset_context_key") or ""),
                row["test_set"],
                row["metric"],
            )
        ].append(row)

    summary = []
    for (experiment_id, md_dataset_size, atom_dataset_size, context_key, test_set, metric), rows in sorted(grouped.items()):
        outcomes = [str(row["winner"]) for row in rows if row.get("winner")]
        winners = [winner for winner in outcomes if winner != "tie"]
        unique_winners = sorted(set(winners))
        seeds = sorted({row["seed"] for row in rows}, key=str)
        valid_seeds = valid_stability_seeds(set(seeds))
        n_seeds = len(seeds)
        stable = len(unique_winners) == 1 and len(rows) > 0 and n_seeds > 1
        single_seed = n_seeds <= 1
        stability_status = seed_stability_status(seeds, outcomes)
        md_wins = sum(1 for winner in winners if winner == METHOD_MD)
        atom_wins = sum(1 for winner in winners if winner == METHOD_ATOM)
        winner_label = unique_winners[0] if (stable or (single_seed and len(unique_winners) == 1)) else "unstable"
        summary.append(
            {
                "experiment_id": experiment_id,
                "dataset_size": f"md_{md_dataset_size}__atom_{atom_dataset_size}",
                "md_dataset_size": md_dataset_size,
                "atom_dataset_size": atom_dataset_size,
                "dataset_context_key": context_key,
                "test_set": test_set,
                "metric": metric,
                "seeds_compared": n_seeds,
                "n_seeds": n_seeds,
                "valid_seed_count": len(valid_seeds),
                "seeds": ",".join(str(seed) for seed in seeds),
                "valid_seeds": ",".join(valid_seeds),
                "md_wins": md_wins,
                "atom_displacement_wins": atom_wins,
                "wins_md": md_wins,
                "wins_atomdisp": atom_wins,
                "ties": sum(1 for row in rows if row["winner"] == "tie"),
                "stable_across_seeds": stable,
                "stable_winner": unique_winners[0] if stable else "unstable",
                "winner": winner_label,
                "winner_stability": "stable" if stable else ("single_seed" if single_seed else "unstable"),
                "seed_stability_status": stability_status,
                "robust_candidate": stability_status == "robust_candidate",
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


def severe_warning_values(rows: list[dict[str, Any]]) -> list[str]:
    warning_columns = [
        "siesta_settings_warning",
        "model_config_warning",
        "basis_pseudopotential_warning",
        "budget_mismatch_warning",
        "leakage_warning",
        "frozen_test_warning",
        "checkpoint_selection_warning",
        "reproducibility_warning",
        "nested_subset_warning",
        "manifest_warning",
        "artifact_hash_warning",
        "matrix_warning",
        "prediction_warning",
        "evaluation_warning",
        "severe_warnings",
    ]
    return sorted(
        {
            str(row.get(column))
            for row in rows
            for column in warning_columns
            if row.get(column) not in (None, "", False)
        }
    )


def build_nway_method_summary(
    rows: list[dict[str, Any]],
    metrics: list[str],
    *,
    aggregation_mode: str = "per_seed",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row in rows:
        method = str(row.get("train_method") or "")
        test_set = str(row.get("test_set") or "")
        if not method or not test_set:
            continue
        context_key = dataset_context_key(row)
        experiment_id, seed, epoch, checkpoint = aggregation_ids(row, aggregation_mode)
        for metric in metrics:
            value = row.get(metric)
            if finite(value):
                grouped[(experiment_id, context_key, seed, epoch, test_set, metric, method, checkpoint)].append(
                    (row, float(value))
                )

    summary_rows: list[dict[str, Any]] = []
    for (
        experiment_id,
        context_key,
        seed,
        epoch,
        test_set,
        metric,
        method,
        checkpoint,
    ), grouped_values in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        source_rows = [row for row, _value in grouped_values]
        values = [value for _row, value in grouped_values]
        first_row = source_rows[0]
        seeds = sorted({row_seed(row) for row in source_rows}, key=str)
        warnings = severe_warning_values(source_rows)
        timing_values: dict[str, list[float]] = defaultdict(list)
        for source_row in source_rows:
            for timing_field in TIMING_FIELDS:
                timing_value = method_timing_value(source_row, timing_field, str(method))
                if finite(timing_value):
                    timing_values[timing_field].append(float(timing_value))
        timing_summary = {
            timing_field: mean(values_for_field)
            for timing_field, values_for_field in timing_values.items()
            if values_for_field
        }
        timing_fields_available = sorted(timing_summary)
        n_seeds = len(seeds)
        if warnings:
            status = "severe_warning"
        elif n_seeds < 3:
            status = "exploratory_only"
        else:
            status = "diagnostic"
        summary_row = {
                "experiment_id": experiment_id,
                "dataset_context_key": context_key,
                "seed": seed,
                "seeds": ",".join(str(item) for item in seeds),
                "n_seeds": n_seeds,
                "epoch": epoch,
                "test_set": test_set,
                "metric": metric,
                "method": method,
                "train_method": method,
                "model_checkpoint": checkpoint,
                "mean": mean(values),
                "std": stddev(values),
                "n_rows": len(values),
                "n_samples": len(values),
                "dataset_size": row_method_dataset_size(first_row, method),
                "dataset_size_by_method": method_context_mapping_json(first_row, "dataset_size"),
                "dataset_label_by_method": method_context_mapping_json(first_row, "dataset_label"),
                "recipe_hash_by_method": method_context_mapping_json(first_row, "recipe_hash"),
                "result_dir_by_method": method_context_mapping_json(first_row, "result_dir"),
                "training_tag_by_method": method_context_mapping_json(first_row, "training_tag"),
                "training_plan_label_by_method": method_context_mapping_json(first_row, "training_plan_label"),
                "training_plan_settings_by_method": method_context_mapping_json(first_row, "training_plan_settings"),
                "dataset_label": mapped_or_legacy_method_value(first_row, "dataset_label", method),
                "recipe_id": mapped_or_legacy_method_value(first_row, "recipe_id", method),
                "recipe_hash": mapped_or_legacy_method_value(first_row, "recipe_hash", method),
                "result_dir": mapped_or_legacy_method_value(first_row, "result_dir", method),
                "training_tag": mapped_or_legacy_method_value(first_row, "training_tag", method),
                "training_plan_label": mapped_or_legacy_method_value(first_row, "training_plan_label", method),
                "training_plan_settings": mapped_or_legacy_method_value(first_row, "training_plan_settings", method),
                "warning_status": status,
                "severe_warnings": " | ".join(warnings),
                "timing_fields_available": ",".join(timing_fields_available),
                "timing_reliability_status": (
                    "reliable_total"
                    if finite(timing_summary.get(RELIABLE_COMPUTE_TIMING_FIELD))
                    else "missing_reliable_timing_fields"
                ),
        }
        summary_row.update(timing_summary)
        summary_rows.append(summary_row)
    return summary_rows


def rank_nway_methods(
    method_rows: list[dict[str, Any]],
    *,
    lower_is_better: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    expected_methods: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in method_rows:
        if finite(row.get("mean")):
            expected_methods[
                (
                    row.get("experiment_id"),
                    row.get("dataset_context_key"),
                    row.get("epoch"),
                    row.get("metric"),
                )
            ].add(str(row.get("method")))
            grouped[
                (
                    row.get("experiment_id"),
                    row.get("dataset_context_key"),
                    row.get("seed"),
                    row.get("epoch"),
                    row.get("test_set"),
                    row.get("metric"),
                )
            ].append(row)

    ranked_rows: list[dict[str, Any]] = []
    scope_records: list[dict[str, Any]] = []
    for key, rows_for_scope in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        experiment_id, context_key, seed, epoch, test_set, metric = key
        expected_for_scope = sorted(
            expected_methods.get((experiment_id, context_key, epoch, metric), set())
        )
        present_methods = {str(row.get("method")) for row in rows_for_scope}
        missing_methods = sorted(set(expected_for_scope) - present_methods)
        ordered = sorted(
            rows_for_scope,
            key=lambda row: (
                float(row["mean"]) if lower_is_better else -float(row["mean"]),
                str(row.get("method")),
            ),
        )
        ordered_values = [float(row["mean"]) for row in ordered]
        previous_value: float | None = None
        previous_rank = 0
        ranked_scope_rows: list[dict[str, Any]] = []
        rank_by_method: dict[str, int] = {}
        for index, row in enumerate(ordered, start=1):
            value = float(row["mean"])
            if previous_value is not None and math.isclose(value, previous_value, rel_tol=1e-12, abs_tol=1e-12):
                rank = previous_rank
            else:
                rank = index
            previous_value = value
            previous_rank = rank
            ranked_row = dict(row)
            method = str(row.get("method"))
            ranked_row["rank"] = rank
            ranked_row["tie"] = any(
                other_index != index - 1
                and math.isclose(value, other_value, rel_tol=1e-12, abs_tol=1e-12)
                for other_index, other_value in enumerate(ordered_values)
            )
            ranked_row["n_methods_ranked"] = len(ordered)
            rank_by_method[method] = rank
            ranked_scope_rows.append(ranked_row)

        leader_methods = sorted(
            method
            for method, rank in rank_by_method.items()
            if rank == 1
        )
        rank_signature = json.dumps(
            {
                method: rank_by_method.get(method, "missing")
                for method in expected_for_scope
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for ranked_row in ranked_scope_rows:
            ranked_row["expected_methods"] = ",".join(expected_for_scope)
            ranked_row["expected_method_count"] = len(expected_for_scope)
            ranked_row["missing_methods"] = ",".join(missing_methods)
            ranked_row["missing_method_count"] = len(missing_methods)
            ranked_row["ranking_grid_status"] = "missing_method" if missing_methods else "complete_scope"
            ranked_row["rank_signature"] = rank_signature
            ranked_row["leader_methods"] = ",".join(leader_methods)
            ranked_row["leader_tie"] = len(leader_methods) > 1
            ranked_row["lower_is_better"] = lower_is_better
            ranked_row["scientific_status"] = "nway_ranking_diagnostic_only"
            ranked_rows.append(ranked_row)
        scope_records.append(
            {
                "stability_key": (experiment_id, context_key, epoch, test_set, metric),
                "seed": seed,
                "rank_signature": rank_signature,
                "missing_methods": missing_methods,
                "has_severe_warning": any(row.get("warning_status") == "severe_warning" for row in ranked_scope_rows),
                "rows": ranked_scope_rows,
            }
        )

    stability_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in scope_records:
        stability_groups[record["stability_key"]].append(record)

    for records in stability_groups.values():
        seeds = sorted({str(record["seed"]) for record in records})
        valid_seeds = valid_stability_seeds(set(seeds))
        signatures = {str(record["rank_signature"]) for record in records}
        has_missing_methods = any(record["missing_methods"] for record in records)
        has_severe_warning = any(record["has_severe_warning"] for record in records)
        if has_missing_methods:
            stability_status = "missing_method"
        elif has_severe_warning:
            stability_status = "severe_warning_not_robust"
        elif len(signatures) > 1 and len(valid_seeds) > 1:
            stability_status = "unstable"
        elif len(valid_seeds) < 3:
            stability_status = "exploratory_only"
        elif len(signatures) > 1:
            stability_status = "unstable"
        else:
            stability_status = "robust_candidate"

        for record in records:
            for row in record["rows"]:
                row["ranking_stability_status"] = stability_status
                row["legacy_ranking_stability_status"] = (
                    "stable_across_seeds"
                    if stability_status == "robust_candidate"
                    else (
                        "unstable_across_seeds"
                        if stability_status == "unstable"
                        else (
                            "exploratory_single_seed"
                            if stability_status == "exploratory_only"
                            else stability_status
                        )
                    )
                )
                row["ranking_seeds"] = ",".join(seeds)
                row["ranking_seed_count"] = len(seeds)
                row["ranking_valid_seeds"] = ",".join(valid_seeds)
                row["ranking_valid_seed_count"] = len(valid_seeds)
                row["robust_candidate"] = stability_status == "robust_candidate"
                if row.get("warning_status") == "severe_warning":
                    row["ranking_status"] = "diagnostic_only_severe_warning"
                elif stability_status == "missing_method":
                    row["ranking_status"] = "diagnostic_only_missing_method"
                elif stability_status == "severe_warning_not_robust":
                    row["ranking_status"] = "diagnostic_only_severe_warning"
                elif stability_status == "exploratory_only":
                    row["ranking_status"] = "exploratory_only"
                elif stability_status == "unstable":
                    row["ranking_status"] = "diagnostic_only_unstable"
                elif stability_status == "robust_candidate":
                    row["ranking_status"] = "robust_candidate"
                else:
                    row["ranking_status"] = "diagnostic_only"
    return ranked_rows


def summarize_nway_ranking(ranking_rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods_seen = sorted({str(row.get("method")) for row in ranking_rows if row.get("method")})
    test_sets_seen = sorted({str(row.get("test_set")) for row in ranking_rows if row.get("test_set")})
    metrics_seen = sorted({str(row.get("metric")) for row in ranking_rows if row.get("metric")})
    scope_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in ranking_rows:
        scope_rows[
            (
                row.get("experiment_id"),
                row.get("dataset_context_key"),
                row.get("epoch"),
                row.get("test_set"),
                row.get("metric"),
                row.get("seed"),
            )
        ].append(row)

    missing_method_cells = []
    tie_groups = []
    seen_missing: set[tuple[str, ...]] = set()
    seen_ties: set[tuple[str, ...]] = set()
    for (
        experiment_id,
        context_key,
        epoch,
        test_set,
        metric,
        seed,
    ), rows in sorted(scope_rows.items(), key=lambda item: tuple(str(part) for part in item[0])):
        representative = rows[0]
        missing_methods = [
            method
            for method in str(representative.get("missing_methods") or "").split(",")
            if method
        ]
        if missing_methods:
            key = (
                str(experiment_id),
                str(context_key),
                str(epoch),
                str(test_set),
                str(metric),
                str(seed),
                ",".join(missing_methods),
            )
            if key not in seen_missing:
                seen_missing.add(key)
                missing_method_cells.append(
                    {
                        "experiment_id": experiment_id,
                        "dataset_context_key": context_key,
                        "epoch": epoch,
                        "test_set": test_set,
                        "metric": metric,
                        "seed": seed,
                        "missing_methods": missing_methods,
                        "expected_methods": [
                            method
                            for method in str(representative.get("expected_methods") or "").split(",")
                            if method
                        ],
                    }
                )
        leader_methods = [
            method
            for method in str(representative.get("leader_methods") or "").split(",")
            if method
        ]
        if representative.get("leader_tie") in (True, "True", "true", "1") or len(leader_methods) > 1:
            key = (
                str(experiment_id),
                str(context_key),
                str(epoch),
                str(test_set),
                str(metric),
                str(seed),
            )
            if key not in seen_ties:
                seen_ties.add(key)
                tie_groups.append(
                    {
                        "experiment_id": experiment_id,
                        "dataset_context_key": context_key,
                        "epoch": epoch,
                        "test_set": test_set,
                        "metric": metric,
                        "seed": seed,
                        "leader_methods": leader_methods,
                    }
                )

    stability_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in ranking_rows:
        stability_rows[
            (
                row.get("experiment_id"),
                row.get("dataset_context_key"),
                row.get("epoch"),
                row.get("test_set"),
                row.get("metric"),
            )
        ].append(row)

    unstable_groups = []
    exploratory_groups = []
    exploratory_single_seed_groups = []
    robust_candidate_groups = []
    severe_warning_groups = []
    missing_method_groups = []
    seen_group_status: set[tuple[str, ...]] = set()
    for (
        experiment_id,
        context_key,
        epoch,
        test_set,
        metric,
    ), rows in sorted(stability_rows.items(), key=lambda item: tuple(str(part) for part in item[0])):
        representative = rows[0]
        status = str(representative.get("ranking_stability_status") or "")
        key = (
            str(experiment_id),
            str(context_key),
            str(epoch),
            str(test_set),
            str(metric),
            status,
        )
        if key in seen_group_status:
            continue
        seen_group_status.add(key)
        group_payload = {
            "experiment_id": experiment_id,
            "dataset_context_key": context_key,
            "epoch": epoch,
            "test_set": test_set,
            "metric": metric,
            "seeds": [
                seed
                for seed in str(representative.get("ranking_seeds") or "").split(",")
                if seed
            ],
            "valid_seeds": [
                seed
                for seed in str(representative.get("ranking_valid_seeds") or "").split(",")
                if seed
            ],
            "valid_seed_count": representative.get("ranking_valid_seed_count"),
            "rank_signatures": sorted({str(row.get("rank_signature")) for row in rows if row.get("rank_signature")}),
        }
        if status == "unstable":
            unstable_groups.append(group_payload)
        elif status == "exploratory_only":
            exploratory_groups.append(group_payload)
            exploratory_single_seed_groups.append(group_payload)
        elif status == "robust_candidate":
            robust_candidate_groups.append(group_payload)
        elif status == "severe_warning_not_robust":
            severe_warning_groups.append(group_payload)
        elif status == "missing_method":
            missing_method_groups.append(group_payload)

    return {
        "status": "diagnostic_only",
        "scientific_status": "nway_ranking_diagnostic_only",
        "n_rows": len(ranking_rows),
        "methods_seen": methods_seen,
        "test_sets_seen": test_sets_seen,
        "metrics_seen": metrics_seen,
        "missing_method_cells": missing_method_cells,
        "missing_method_cell_count": len(missing_method_cells),
        "missing_method_groups": missing_method_groups,
        "unstable_groups": unstable_groups,
        "unstable_group_count": len(unstable_groups),
        "exploratory_groups": exploratory_groups,
        "exploratory_group_count": len(exploratory_groups),
        "exploratory_single_seed_groups": exploratory_single_seed_groups,
        "exploratory_single_seed_group_count": len(exploratory_single_seed_groups),
        "robust_candidate_groups": robust_candidate_groups,
        "robust_candidate_group_count": len(robust_candidate_groups),
        "severe_warning_groups": severe_warning_groups,
        "severe_warning_group_count": len(severe_warning_groups),
        "tie_groups": tie_groups,
        "tie_group_count": len(tie_groups),
        "note": "N-way rankings are diagnostic and do not declare a robust scientific recommendation.",
    }


def winner_between_methods(
    baseline_method: str,
    challenger_method: str,
    baseline_mean: float,
    challenger_mean: float,
    *,
    lower_is_better: bool,
) -> str:
    if math.isclose(baseline_mean, challenger_mean, rel_tol=1e-12, abs_tol=1e-12):
        return "tie"
    if lower_is_better:
        return challenger_method if challenger_mean < baseline_mean else baseline_method
    return challenger_method if challenger_mean > baseline_mean else baseline_method


def percent_improvement_vs_baseline(
    baseline_mean: float,
    challenger_mean: float,
    *,
    lower_is_better: bool,
) -> float | None:
    baseline = abs(baseline_mean)
    if baseline <= 0:
        return None
    delta = (baseline_mean - challenger_mean) if lower_is_better else (challenger_mean - baseline_mean)
    return 100.0 * delta / baseline


def own_test_set_for_method(method: str) -> str:
    return f"test_{method}"


def pairwise_warning_status(baseline: dict[str, Any], challenger: dict[str, Any]) -> str:
    statuses = {str(value) for value in (baseline.get("warning_status"), challenger.get("warning_status"))}
    if "severe_warning" in statuses:
        return "severe_warning"
    if "exploratory_only" in statuses:
        return "exploratory_only"
    return "diagnostic"


def combined_warning_text(*rows: dict[str, Any]) -> str:
    warnings = sorted(
        {
            str(row.get("severe_warnings"))
            for row in rows
            if row.get("severe_warnings") not in (None, "", False)
        }
    )
    return " | ".join(warnings)


def annotate_pairwise_distribution_status(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("experiment_id"),
                row.get("dataset_context_key"),
                row.get("seed"),
                row.get("epoch"),
                row.get("metric"),
                row.get("challenger_method"),
            )
        ].append(row)

    for (_experiment_id, _context_key, _seed, _epoch, _metric, challenger_method), group_rows in grouped.items():
        own_test_set = own_test_set_for_method(str(challenger_method))
        challenger_wins = [
            row
            for row in group_rows
            if row.get("winner") == challenger_method
        ]
        if not challenger_wins:
            win_scope_status = "no_challenger_win"
        elif all(row.get("test_set") == own_test_set for row in challenger_wins):
            win_scope_status = "distribution_specific_only"
        else:
            win_scope_status = "general_or_mixed_win"

        for row in group_rows:
            test_set = str(row.get("test_set") or "")
            if test_set == own_test_set:
                role = "challenger_distribution"
            elif test_set == own_test_set_for_method(str(row.get("baseline_method"))):
                role = "baseline_distribution"
            elif test_set == "test_mixed":
                role = "mixed_distribution"
            else:
                role = "other_distribution"
            row["test_distribution_role"] = role
            row["win_scope_status"] = win_scope_status
            row["distribution_status"] = (
                "distribution_specific"
                if row.get("winner") == challenger_method and test_set == own_test_set
                else win_scope_status
            )


def annotate_pairwise_seed_stability(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("experiment_id"),
                row.get("dataset_context_key"),
                row.get("epoch"),
                row.get("test_set"),
                row.get("metric"),
                row.get("challenger_method"),
            )
        ].append(row)

    for group_rows in grouped.values():
        seeds = sorted({normalized_seed_label(row.get("seed")) for row in group_rows}, key=str)
        valid_seeds = valid_stability_seeds(set(seeds))
        outcomes = [str(row.get("winner")) for row in group_rows if row.get("winner")]
        has_severe_warning = any(row.get("warning_status") == "severe_warning" for row in group_rows)
        stability_status = seed_stability_status(
            seeds,
            outcomes,
            has_severe_warning=has_severe_warning,
        )
        stable_outcome = sorted(set(outcomes))[0] if len(set(outcomes)) == 1 and outcomes else None
        for row in group_rows:
            row["seed_stability_status"] = stability_status
            row["seed_stability_scientific_status"] = (
                "robust_candidate" if stability_status == "robust_candidate" else "not_robust"
            )
            row["seed_stability_seeds"] = ",".join(seeds)
            row["seed_stability_valid_seeds"] = ",".join(valid_seeds)
            row["seed_stability_n_seeds"] = len(seeds)
            row["seed_stability_valid_n_seeds"] = len(valid_seeds)
            row["stable_winner_across_seeds"] = stable_outcome or "unstable"
            row["robust_candidate"] = stability_status == "robust_candidate"


def build_pairwise_vs_baseline(
    method_rows: list[dict[str, Any]],
    *,
    baseline_method: str = METHOD_MD,
    lower_is_better: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in method_rows:
        if finite(row.get("mean")) and row.get("method"):
            grouped[
                (
                    row.get("experiment_id"),
                    row.get("dataset_context_key"),
                    row.get("seed"),
                    row.get("epoch"),
                    row.get("test_set"),
                    row.get("metric"),
                )
            ].append(row)

    pair_rows: list[dict[str, Any]] = []
    for key, rows_for_scope in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        experiment_id, context_key, seed, epoch, test_set, metric = key
        baseline_rows = [row for row in rows_for_scope if row.get("method") == baseline_method]
        challenger_rows = [
            row
            for row in rows_for_scope
            if row.get("method") not in (None, "", baseline_method)
        ]
        if not baseline_rows or not challenger_rows:
            continue

        for baseline in baseline_rows:
            for challenger in challenger_rows:
                baseline_mean = float(baseline["mean"])
                challenger_mean = float(challenger["mean"])
                challenger_method = str(challenger["method"])
                winner = winner_between_methods(
                    baseline_method,
                    challenger_method,
                    baseline_mean,
                    challenger_mean,
                    lower_is_better=lower_is_better,
                )
                pair_row = {
                        "experiment_id": experiment_id,
                        "dataset_context_key": context_key,
                        "dataset_size_by_method": baseline.get("dataset_size_by_method")
                        or challenger.get("dataset_size_by_method"),
                        "dataset_label_by_method": baseline.get("dataset_label_by_method")
                        or challenger.get("dataset_label_by_method"),
                        "recipe_hash_by_method": baseline.get("recipe_hash_by_method")
                        or challenger.get("recipe_hash_by_method"),
                        "result_dir_by_method": baseline.get("result_dir_by_method")
                        or challenger.get("result_dir_by_method"),
                        "training_tag_by_method": baseline.get("training_tag_by_method")
                        or challenger.get("training_tag_by_method"),
                        "training_plan_label_by_method": baseline.get("training_plan_label_by_method")
                        or challenger.get("training_plan_label_by_method"),
                        "training_plan_settings_by_method": baseline.get("training_plan_settings_by_method")
                        or challenger.get("training_plan_settings_by_method"),
                        "baseline_method": baseline_method,
                        "challenger_method": challenger_method,
                        "test_set": test_set,
                        "metric": metric,
                        "baseline_mean": baseline_mean,
                        "challenger_mean": challenger_mean,
                        "absolute_difference": challenger_mean - baseline_mean,
                        "absolute_difference_challenger_minus_baseline": challenger_mean - baseline_mean,
                        "percent_improvement_challenger_vs_baseline": percent_improvement_vs_baseline(
                            baseline_mean,
                            challenger_mean,
                            lower_is_better=lower_is_better,
                        ),
                        "winner": winner,
                        "seed": seed,
                        "epoch": epoch,
                        "baseline_model_checkpoint": baseline.get("model_checkpoint"),
                        "challenger_model_checkpoint": challenger.get("model_checkpoint"),
                        "baseline_dataset_size": baseline.get("dataset_size"),
                        "challenger_dataset_size": challenger.get("dataset_size"),
                        "baseline_dataset_label": baseline.get("dataset_label"),
                        "challenger_dataset_label": challenger.get("dataset_label"),
                        "baseline_recipe_id": baseline.get("recipe_id"),
                        "challenger_recipe_id": challenger.get("recipe_id"),
                        "baseline_recipe_hash": baseline.get("recipe_hash"),
                        "challenger_recipe_hash": challenger.get("recipe_hash"),
                        "baseline_result_dir": baseline.get("result_dir"),
                        "challenger_result_dir": challenger.get("result_dir"),
                        "baseline_training_tag": baseline.get("training_tag"),
                        "challenger_training_tag": challenger.get("training_tag"),
                        "baseline_training_plan_label": baseline.get("training_plan_label"),
                        "challenger_training_plan_label": challenger.get("training_plan_label"),
                        "baseline_training_plan_settings": baseline.get("training_plan_settings"),
                        "challenger_training_plan_settings": challenger.get("training_plan_settings"),
                        "baseline_n_rows": baseline.get("n_rows"),
                        "challenger_n_rows": challenger.get("n_rows"),
                        "baseline_n_seeds": baseline.get("n_seeds"),
                        "challenger_n_seeds": challenger.get("n_seeds"),
                        "lower_is_better": lower_is_better,
                        "baseline_warning_status": baseline.get("warning_status"),
                        "challenger_warning_status": challenger.get("warning_status"),
                        "warning_status": pairwise_warning_status(baseline, challenger),
                        "severe_warnings": combined_warning_text(baseline, challenger),
                        "status": "diagnostic_only",
                }
                for timing_field in TIMING_FIELDS:
                    baseline_timing = baseline.get(timing_field)
                    challenger_timing = challenger.get(timing_field)
                    pair_row[f"baseline_{timing_field}"] = baseline_timing
                    pair_row[f"challenger_{timing_field}"] = challenger_timing
                baseline_total = pair_row.get(f"baseline_{RELIABLE_COMPUTE_TIMING_FIELD}")
                challenger_total = pair_row.get(f"challenger_{RELIABLE_COMPUTE_TIMING_FIELD}")
                if finite(baseline_total) and finite(challenger_total):
                    pair_row["baseline_compute_budget_seconds"] = float(baseline_total)
                    pair_row["challenger_compute_budget_seconds"] = float(challenger_total)
                    pair_row["compute_budget_seconds"] = max(float(baseline_total), float(challenger_total))
                    pair_row["compute_budget_timing_source"] = RELIABLE_COMPUTE_TIMING_FIELD
                    pair_row["compute_threshold_timing_status"] = "reliable_total"
                    pair_row["compute_threshold_unavailable"] = ""
                else:
                    pair_row["compute_threshold_timing_status"] = "missing_reliable_timing_fields"
                    pair_row["compute_threshold_unavailable"] = "missing reliable timing fields"
                pair_rows.append(pair_row)

    annotate_pairwise_distribution_status(pair_rows)
    annotate_pairwise_seed_stability(pair_rows)
    return pair_rows


def summarize_pairwise_vs_baseline(
    pair_rows: list[dict[str, Any]],
    *,
    baseline_method: str = METHOD_MD,
) -> dict[str, Any]:
    challengers = sorted({str(row.get("challenger_method")) for row in pair_rows if row.get("challenger_method")})
    wins_by_challenger = {
        challenger: sum(1 for row in pair_rows if row.get("challenger_method") == challenger and row.get("winner") == challenger)
        for challenger in challengers
    }
    wins_by_baseline = {
        challenger: sum(
            1
            for row in pair_rows
            if row.get("challenger_method") == challenger and row.get("winner") == baseline_method
        )
        for challenger in challengers
    }
    ties_by_challenger = {
        challenger: sum(1 for row in pair_rows if row.get("challenger_method") == challenger and row.get("winner") == "tie")
        for challenger in challengers
    }
    stability_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        stability_groups[
            (
                row.get("experiment_id"),
                row.get("dataset_context_key"),
                row.get("epoch"),
                row.get("test_set"),
                row.get("metric"),
                row.get("challenger_method"),
            )
        ].append(row)
    robust_candidates = []
    unstable_groups = []
    exploratory_groups = []
    severe_warning_groups = []
    for (
        experiment_id,
        context_key,
        epoch,
        test_set,
        metric,
        challenger_method,
    ), rows_for_group in sorted(stability_groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        representative = rows_for_group[0]
        payload = {
            "experiment_id": experiment_id,
            "dataset_context_key": context_key,
            "epoch": epoch,
            "test_set": test_set,
            "metric": metric,
            "baseline_method": baseline_method,
            "challenger_method": challenger_method,
            "winner": representative.get("stable_winner_across_seeds"),
            "seed_stability_status": representative.get("seed_stability_status"),
            "valid_seed_count": representative.get("seed_stability_valid_n_seeds"),
            "valid_seeds": [
                seed
                for seed in str(representative.get("seed_stability_valid_seeds") or "").split(",")
                if seed
            ],
            "all_seeds": [
                seed
                for seed in str(representative.get("seed_stability_seeds") or "").split(",")
                if seed
            ],
        }
        status = representative.get("seed_stability_status")
        if status == "robust_candidate":
            robust_candidates.append(payload)
        elif status == "unstable":
            unstable_groups.append(payload)
        elif status == "exploratory_only":
            exploratory_groups.append(payload)
        elif status == "severe_warning_not_robust":
            severe_warning_groups.append(payload)

    distribution_specific_only = []
    seen_distribution_specific: set[tuple[str, str, str, str, str]] = set()
    for row in pair_rows:
        if row.get("win_scope_status") != "distribution_specific_only" or row.get("winner") != row.get("challenger_method"):
            continue
        key = (
            str(row.get("experiment_id")),
            str(row.get("dataset_context_key")),
            str(row.get("metric")),
            str(row.get("seed")),
            str(row.get("challenger_method")),
        )
        if key in seen_distribution_specific:
            continue
        seen_distribution_specific.add(key)
        distribution_specific_only.append(
            {
                "experiment_id": row.get("experiment_id"),
                "dataset_context_key": row.get("dataset_context_key"),
                "metric": row.get("metric"),
                "seed": row.get("seed"),
                "challenger_method": row.get("challenger_method"),
                "winning_test_set": row.get("test_set"),
            }
        )

    return {
        "status": "diagnostic_only",
        "scientific_status": "pairwise_diagnostic_only",
        "baseline_method": baseline_method,
        "challenger_methods": challengers,
        "n_rows": len(pair_rows),
        "wins_by_challenger": wins_by_challenger,
        "wins_by_baseline": wins_by_baseline,
        "ties_by_challenger": ties_by_challenger,
        "robust_candidates": robust_candidates,
        "robust_candidate_count": len(robust_candidates),
        "unstable_groups": unstable_groups,
        "unstable_group_count": len(unstable_groups),
        "exploratory_groups": exploratory_groups,
        "exploratory_group_count": len(exploratory_groups),
        "severe_warning_groups": severe_warning_groups,
        "severe_warning_group_count": len(severe_warning_groups),
        "distribution_specific_only": distribution_specific_only,
        "distribution_specific_only_count": len(distribution_specific_only),
        "note": "Pairwise challenger-vs-baseline comparisons only; no robust winner is declared here.",
    }


def threshold_size_sort_key(value: Any) -> tuple[int, float, str]:
    if finite(value):
        return (0, float(value), "")
    text = str(value or "").strip()
    try:
        return (0, float(text), "")
    except (TypeError, ValueError):
        return (1, math.inf, text)


def parse_seed_csv(value: Any) -> list[str]:
    return [seed for seed in str(value or "").split(",") if seed]


def pairwise_stability_groups(
    pair_rows: list[dict[str, Any]],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[
            (
                row.get("experiment_id"),
                row.get("dataset_context_key"),
                row.get("epoch"),
                row.get("test_set"),
                row.get("metric"),
                row.get("challenger_method"),
            )
        ].append(row)
    return grouped


def threshold_distribution_scope(row: dict[str, Any]) -> str:
    if row.get("test_distribution_role") == "challenger_distribution":
        return "distribution_specific"
    return "general"


def threshold_unavailable_reason(
    group_records: list[dict[str, Any]],
    *,
    challenger_method: str,
    test_set: str,
    completeness_invalid: bool,
) -> str:
    if completeness_invalid:
        return "Incomplete cross-evaluation grid"
    if test_set == own_test_set_for_method(challenger_method):
        if any(
            record.get("stable_winner_across_seeds") == challenger_method
            or record.get("has_challenger_seed_win")
            for record in group_records
        ):
            return (
                "Challenger wins on its own test distribution are distribution-specific "
                "and are not eligible for a general dataset-size threshold."
            )
        return "The challenger's own test distribution is not eligible for a general dataset-size threshold."
    challenger_wins = [record for record in group_records if record.get("stable_winner_across_seeds") == challenger_method]
    if not challenger_wins:
        seed_limited_wins = [record for record in group_records if record.get("has_challenger_seed_win")]
        if seed_limited_wins:
            statuses = sorted({str(record.get("seed_stability_status")) for record in seed_limited_wins})
            if "unstable" in statuses:
                return "Winner changes across seeds; no stable threshold exists."
            if "exploratory_only" in statuses:
                return "Fewer than 3 valid seeds are available for the challenger win."
            if "severe_warning_not_robust" in statuses:
                return "Severe warnings prevent a robust threshold."
        return "No stable challenger win over md for this metric and test set."
    statuses = sorted({str(record.get("seed_stability_status")) for record in challenger_wins})
    if "severe_warning_not_robust" in statuses:
        return "Severe warnings prevent a robust threshold."
    if "unstable" in statuses:
        return "Winner changes across seeds; no stable threshold exists."
    if "exploratory_only" in statuses:
        return "Fewer than 3 valid seeds are available for the challenger win."
    return "No eligible general challenger threshold exists."


def dataset_size_threshold_row(
    representative: dict[str, Any],
    *,
    threshold_available: bool,
    reason: str,
    scientific_status: str,
) -> dict[str, Any]:
    challenger_method = str(representative.get("challenger_method") or "")
    distribution_scope = threshold_distribution_scope(representative)
    return {
        "experiment_id": representative.get("experiment_id"),
        "dataset_context_key": representative.get("dataset_context_key"),
        "baseline_method": representative.get("baseline_method") or METHOD_MD,
        "challenger_method": challenger_method,
        "metric": representative.get("metric"),
        "test_set": representative.get("test_set"),
        "epoch": representative.get("epoch"),
        "threshold_available": threshold_available,
        "first_stable_dataset_size": representative.get("challenger_dataset_size") if threshold_available else None,
        "challenger_dataset_size": representative.get("challenger_dataset_size"),
        "corresponding_md_dataset_size": representative.get("baseline_dataset_size"),
        "baseline_dataset_size": representative.get("baseline_dataset_size"),
        "challenger_dataset_label": representative.get("challenger_dataset_label"),
        "baseline_dataset_label": representative.get("baseline_dataset_label"),
        "challenger_recipe_id": representative.get("challenger_recipe_id"),
        "baseline_recipe_id": representative.get("baseline_recipe_id"),
        "challenger_recipe_hash": representative.get("challenger_recipe_hash"),
        "baseline_recipe_hash": representative.get("baseline_recipe_hash"),
        "challenger_result_dir": representative.get("challenger_result_dir"),
        "baseline_result_dir": representative.get("baseline_result_dir"),
        "dataset_size_by_method": representative.get("dataset_size_by_method"),
        "dataset_label_by_method": representative.get("dataset_label_by_method"),
        "recipe_hash_by_method": representative.get("recipe_hash_by_method"),
        "result_dir_by_method": representative.get("result_dir_by_method"),
        "training_tag_by_method": representative.get("training_tag_by_method"),
        "training_plan_label_by_method": representative.get("training_plan_label_by_method"),
        "training_plan_settings_by_method": representative.get("training_plan_settings_by_method"),
        "challenger_training_tag": representative.get("challenger_training_tag"),
        "baseline_training_tag": representative.get("baseline_training_tag"),
        "challenger_training_plan_label": representative.get("challenger_training_plan_label"),
        "baseline_training_plan_label": representative.get("baseline_training_plan_label"),
        "challenger_training_plan_settings": representative.get("challenger_training_plan_settings"),
        "baseline_training_plan_settings": representative.get("baseline_training_plan_settings"),
        "n_seeds": representative.get("seed_stability_valid_n_seeds"),
        "valid_seed_count": representative.get("seed_stability_valid_n_seeds"),
        "seeds": representative.get("seed_stability_valid_seeds"),
        "all_seeds": representative.get("seed_stability_seeds"),
        "stability_status": representative.get("seed_stability_status"),
        "seed_stability_status": representative.get("seed_stability_status"),
        "stable_winner_across_seeds": representative.get("stable_winner_across_seeds"),
        "win_scope": distribution_scope,
        "distribution_scope": distribution_scope,
        "test_distribution_role": representative.get("test_distribution_role"),
        "warning_status": representative.get("warning_status"),
        "severe_warnings": representative.get("severe_warnings"),
        "scientific_status": scientific_status,
        "reason": reason,
    }


def build_dataset_size_thresholds_vs_md(
    pair_rows: list[dict[str, Any]],
    *,
    completeness_report: dict[str, Any] | None = None,
    baseline_method: str = METHOD_MD,
    minimum_robust_seeds: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    completeness_invalid = completeness_report_is_invalid(completeness_report)
    grouped_records: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for _stability_key, rows_for_stability in pairwise_stability_groups(pair_rows).items():
        representative = sorted(rows_for_stability, key=lambda row: str(row.get("seed") or ""))[0]
        if representative.get("baseline_method") != baseline_method:
            continue
        challenger_method = str(representative.get("challenger_method") or "")
        valid_seed_count = int(representative.get("seed_stability_valid_n_seeds") or 0)
        has_challenger_seed_win = any(row.get("winner") == challenger_method for row in rows_for_stability)
        record = dict(representative)
        record["valid_seed_count"] = valid_seed_count
        record["has_challenger_seed_win"] = has_challenger_seed_win
        grouped_records[
            (
                record.get("experiment_id"),
                challenger_method,
                record.get("metric"),
                record.get("test_set"),
            )
        ].append(record)

    threshold_rows: list[dict[str, Any]] = []
    for (
        _experiment_id,
        challenger_method,
        _metric,
        test_set,
    ), records in sorted(grouped_records.items(), key=lambda item: tuple(str(part) for part in item[0])):
        eligible = [
            record
            for record in records
            if not completeness_invalid
            and record.get("stable_winner_across_seeds") == challenger_method
            and record.get("seed_stability_status") == "robust_candidate"
            and int(record.get("valid_seed_count") or 0) >= minimum_robust_seeds
            and record.get("warning_status") != "severe_warning"
            and threshold_distribution_scope(record) == "general"
        ]
        eligible.sort(
            key=lambda record: (
                threshold_size_sort_key(record.get("challenger_dataset_size")),
                threshold_size_sort_key(record.get("baseline_dataset_size")),
                str(record.get("challenger_recipe_hash") or ""),
                str(record.get("dataset_context_key") or ""),
            )
        )
        if eligible:
            threshold_rows.append(
                dataset_size_threshold_row(
                    eligible[0],
                    threshold_available=True,
                    reason="First stable general challenger win over md.",
                    scientific_status="stable_threshold_detected",
                )
            )
            continue

        representative = sorted(
            records,
            key=lambda record: (
                threshold_size_sort_key(record.get("challenger_dataset_size")),
                str(record.get("dataset_context_key") or ""),
            ),
        )[0]
        reason = threshold_unavailable_reason(
            records,
            challenger_method=challenger_method,
            test_set=str(test_set or ""),
            completeness_invalid=completeness_invalid,
        )
        threshold_rows.append(
            dataset_size_threshold_row(
                representative,
                threshold_available=False,
                reason=reason,
                scientific_status="no_stable_threshold",
            )
        )

    thresholds = [row for row in threshold_rows if row.get("threshold_available")]
    unavailable = [row for row in threshold_rows if not row.get("threshold_available")]
    challenger_methods = sorted({str(row.get("challenger_method")) for row in threshold_rows if row.get("challenger_method")})
    challenger_summaries = []
    for challenger_method in challenger_methods:
        challenger_thresholds = [row for row in thresholds if row.get("challenger_method") == challenger_method]
        challenger_unavailable = [row for row in unavailable if row.get("challenger_method") == challenger_method]
        if challenger_thresholds:
            challenger_summaries.append(
                {
                    "challenger_method": challenger_method,
                    "threshold_available": True,
                    "threshold_count": len(challenger_thresholds),
                    "first_stable_dataset_size": challenger_thresholds[0].get("first_stable_dataset_size"),
                    "metric": challenger_thresholds[0].get("metric"),
                    "test_set": challenger_thresholds[0].get("test_set"),
                    "reason": challenger_thresholds[0].get("reason"),
                }
            )
        else:
            reasons = sorted({str(row.get("reason")) for row in challenger_unavailable if row.get("reason")})
            challenger_summaries.append(
                {
                    "challenger_method": challenger_method,
                    "threshold_available": False,
                    "threshold_count": 0,
                    "reason": reasons[0] if len(reasons) == 1 else "No eligible general threshold exists.",
                    "reasons": reasons,
                }
            )

    summary = {
        "status": "diagnostic_only",
        "scientific_status": "invalid_incomplete_grid" if completeness_invalid else "threshold_diagnostic_only",
        "baseline_method": baseline_method,
        "minimum_robust_seeds": minimum_robust_seeds,
        "completeness_report_available": bool(completeness_report),
        "complete_grid": (not completeness_invalid) if completeness_report else None,
        "threshold_count": len(thresholds),
        "unavailable_threshold_count": len(unavailable),
        "challenger_methods": challenger_methods,
        "challenger_summaries": challenger_summaries,
        "thresholds": thresholds,
        "unavailable_thresholds": unavailable,
        "all_results": threshold_rows,
        "note": (
            "Thresholds require a robust_candidate challenger win outside the challenger's own "
            "test distribution. Pairwise and ranking outputs remain diagnostic."
        ),
    }
    if completeness_invalid:
        summary["reason"] = "Incomplete cross-evaluation grid"
        summary["missing_cells"] = [
            *((completeness_report or {}).get("missing_cells") or []),
            *((completeness_report or {}).get("missing_context_cells") or []),
        ]
        summary["missing_primary_metric_cells"] = [
            *((completeness_report or {}).get("missing_primary_metric_cells") or []),
            *((completeness_report or {}).get("missing_primary_metric_context_cells") or []),
        ]
    return threshold_rows, summary


def compute_threshold_unavailable_reason(
    records: list[dict[str, Any]],
    *,
    challenger_method: str,
    test_set: str,
    completeness_invalid: bool,
) -> str:
    if completeness_invalid:
        return "Incomplete cross-evaluation grid"
    if not any(record.get("has_reliable_compute_timing") for record in records):
        return "missing reliable timing fields"
    return threshold_unavailable_reason(
        records,
        challenger_method=challenger_method,
        test_set=test_set,
        completeness_invalid=completeness_invalid,
    )


def compute_budget_threshold_row(
    representative: dict[str, Any],
    *,
    threshold_available: bool,
    reason: str,
    scientific_status: str,
) -> dict[str, Any]:
    distribution_scope = threshold_distribution_scope(representative)
    unavailable_reason = "" if threshold_available else reason
    return {
        "experiment_id": representative.get("experiment_id"),
        "dataset_context_key": representative.get("dataset_context_key"),
        "baseline_method": representative.get("baseline_method") or METHOD_MD,
        "challenger_method": representative.get("challenger_method"),
        "metric": representative.get("metric"),
        "test_set": representative.get("test_set"),
        "epoch": representative.get("epoch"),
        "threshold_available": threshold_available,
        "compute_threshold_available": threshold_available,
        "first_stable_compute_budget_seconds": representative.get("compute_budget_seconds")
        if threshold_available
        else None,
        "compute_budget_seconds": representative.get("compute_budget_seconds"),
        "mean_compute_budget_seconds": representative.get("mean_compute_budget_seconds"),
        "max_compute_budget_seconds": representative.get("max_compute_budget_seconds"),
        "baseline_compute_budget_seconds": representative.get("baseline_compute_budget_seconds"),
        "challenger_compute_budget_seconds": representative.get("challenger_compute_budget_seconds"),
        "baseline_total_time_seconds": representative.get(f"baseline_{RELIABLE_COMPUTE_TIMING_FIELD}"),
        "challenger_total_time_seconds": representative.get(f"challenger_{RELIABLE_COMPUTE_TIMING_FIELD}"),
        "compute_budget_timing_source": representative.get("compute_budget_timing_source"),
        "compute_budget_aggregation": representative.get("compute_budget_aggregation"),
        "compute_threshold_timing_status": representative.get("compute_threshold_timing_status"),
        "compute_threshold_unavailable": unavailable_reason,
        "challenger_dataset_size": representative.get("challenger_dataset_size"),
        "corresponding_md_dataset_size": representative.get("baseline_dataset_size"),
        "baseline_dataset_size": representative.get("baseline_dataset_size"),
        "challenger_dataset_label": representative.get("challenger_dataset_label"),
        "baseline_dataset_label": representative.get("baseline_dataset_label"),
        "challenger_recipe_id": representative.get("challenger_recipe_id"),
        "baseline_recipe_id": representative.get("baseline_recipe_id"),
        "challenger_recipe_hash": representative.get("challenger_recipe_hash"),
        "baseline_recipe_hash": representative.get("baseline_recipe_hash"),
        "challenger_result_dir": representative.get("challenger_result_dir"),
        "baseline_result_dir": representative.get("baseline_result_dir"),
        "dataset_size_by_method": representative.get("dataset_size_by_method"),
        "dataset_label_by_method": representative.get("dataset_label_by_method"),
        "recipe_hash_by_method": representative.get("recipe_hash_by_method"),
        "result_dir_by_method": representative.get("result_dir_by_method"),
        "training_tag_by_method": representative.get("training_tag_by_method"),
        "training_plan_label_by_method": representative.get("training_plan_label_by_method"),
        "training_plan_settings_by_method": representative.get("training_plan_settings_by_method"),
        "challenger_training_tag": representative.get("challenger_training_tag"),
        "baseline_training_tag": representative.get("baseline_training_tag"),
        "challenger_training_plan_label": representative.get("challenger_training_plan_label"),
        "baseline_training_plan_label": representative.get("baseline_training_plan_label"),
        "challenger_training_plan_settings": representative.get("challenger_training_plan_settings"),
        "baseline_training_plan_settings": representative.get("baseline_training_plan_settings"),
        "n_seeds": representative.get("seed_stability_valid_n_seeds"),
        "valid_seed_count": representative.get("seed_stability_valid_n_seeds"),
        "seeds": representative.get("seed_stability_valid_seeds"),
        "all_seeds": representative.get("seed_stability_seeds"),
        "stability_status": representative.get("seed_stability_status"),
        "seed_stability_status": representative.get("seed_stability_status"),
        "stable_winner_across_seeds": representative.get("stable_winner_across_seeds"),
        "win_scope": distribution_scope,
        "distribution_scope": distribution_scope,
        "test_distribution_role": representative.get("test_distribution_role"),
        "warning_status": representative.get("warning_status"),
        "severe_warnings": representative.get("severe_warnings"),
        "scientific_status": scientific_status,
        "reason": reason,
    }


def build_compute_budget_thresholds_vs_md(
    pair_rows: list[dict[str, Any]],
    *,
    completeness_report: dict[str, Any] | None = None,
    baseline_method: str = METHOD_MD,
    minimum_robust_seeds: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    completeness_invalid = completeness_report_is_invalid(completeness_report)
    grouped_records: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for _stability_key, rows_for_stability in pairwise_stability_groups(pair_rows).items():
        representative = sorted(rows_for_stability, key=lambda row: str(row.get("seed") or ""))[0]
        if representative.get("baseline_method") != baseline_method:
            continue
        challenger_method = str(representative.get("challenger_method") or "")
        budgets = [float(row["compute_budget_seconds"]) for row in rows_for_stability if finite(row.get("compute_budget_seconds"))]
        baseline_budgets = [
            float(row["baseline_compute_budget_seconds"])
            for row in rows_for_stability
            if finite(row.get("baseline_compute_budget_seconds"))
        ]
        challenger_budgets = [
            float(row["challenger_compute_budget_seconds"])
            for row in rows_for_stability
            if finite(row.get("challenger_compute_budget_seconds"))
        ]
        has_reliable_timing = (
            bool(budgets)
            and len(budgets) == len(rows_for_stability)
            and len(baseline_budgets) == len(rows_for_stability)
            and len(challenger_budgets) == len(rows_for_stability)
        )
        record = dict(representative)
        record["valid_seed_count"] = int(representative.get("seed_stability_valid_n_seeds") or 0)
        record["has_challenger_seed_win"] = any(row.get("winner") == challenger_method for row in rows_for_stability)
        record["has_reliable_compute_timing"] = has_reliable_timing
        if has_reliable_timing:
            record["compute_budget_seconds"] = max(budgets)
            record["mean_compute_budget_seconds"] = mean(budgets)
            record["max_compute_budget_seconds"] = max(budgets)
            record["baseline_compute_budget_seconds"] = max(baseline_budgets)
            record["challenger_compute_budget_seconds"] = max(challenger_budgets)
            record[f"baseline_{RELIABLE_COMPUTE_TIMING_FIELD}"] = max(baseline_budgets)
            record[f"challenger_{RELIABLE_COMPUTE_TIMING_FIELD}"] = max(challenger_budgets)
            record["compute_budget_timing_source"] = RELIABLE_COMPUTE_TIMING_FIELD
            record["compute_budget_aggregation"] = "max_across_valid_seed_rows"
            record["compute_threshold_timing_status"] = "reliable_total"
            record["compute_threshold_unavailable"] = ""
        else:
            record["compute_threshold_timing_status"] = "missing_reliable_timing_fields"
            record["compute_threshold_unavailable"] = "missing reliable timing fields"
        grouped_records[
            (
                record.get("experiment_id"),
                challenger_method,
                record.get("metric"),
                record.get("test_set"),
            )
        ].append(record)

    threshold_rows: list[dict[str, Any]] = []
    for (
        _experiment_id,
        challenger_method,
        _metric,
        test_set,
    ), records in sorted(grouped_records.items(), key=lambda item: tuple(str(part) for part in item[0])):
        eligible = [
            record
            for record in records
            if not completeness_invalid
            and record.get("has_reliable_compute_timing")
            and record.get("stable_winner_across_seeds") == challenger_method
            and record.get("seed_stability_status") == "robust_candidate"
            and int(record.get("valid_seed_count") or 0) >= minimum_robust_seeds
            and record.get("warning_status") != "severe_warning"
            and threshold_distribution_scope(record) == "general"
        ]
        eligible.sort(
            key=lambda record: (
                threshold_size_sort_key(record.get("compute_budget_seconds")),
                threshold_size_sort_key(record.get("challenger_compute_budget_seconds")),
                threshold_size_sort_key(record.get("challenger_dataset_size")),
                str(record.get("challenger_recipe_hash") or ""),
                str(record.get("dataset_context_key") or ""),
            )
        )
        if eligible:
            threshold_rows.append(
                compute_budget_threshold_row(
                    eligible[0],
                    threshold_available=True,
                    reason="First stable general challenger win over md with reliable total timing.",
                    scientific_status="stable_compute_threshold_detected",
                )
            )
            continue

        representative = sorted(
            records,
            key=lambda record: (
                threshold_size_sort_key(record.get("compute_budget_seconds")),
                threshold_size_sort_key(record.get("challenger_dataset_size")),
                str(record.get("dataset_context_key") or ""),
            ),
        )[0]
        reason = compute_threshold_unavailable_reason(
            records,
            challenger_method=challenger_method,
            test_set=str(test_set or ""),
            completeness_invalid=completeness_invalid,
        )
        threshold_rows.append(
            compute_budget_threshold_row(
                representative,
                threshold_available=False,
                reason=reason,
                scientific_status="no_stable_compute_threshold",
            )
        )

    thresholds = [row for row in threshold_rows if row.get("threshold_available")]
    unavailable = [row for row in threshold_rows if not row.get("threshold_available")]
    challenger_methods = sorted({str(row.get("challenger_method")) for row in threshold_rows if row.get("challenger_method")})
    challenger_summaries = []
    for challenger_method in challenger_methods:
        challenger_thresholds = [row for row in thresholds if row.get("challenger_method") == challenger_method]
        challenger_unavailable = [row for row in unavailable if row.get("challenger_method") == challenger_method]
        if challenger_thresholds:
            challenger_summaries.append(
                {
                    "challenger_method": challenger_method,
                    "compute_threshold_available": True,
                    "threshold_count": len(challenger_thresholds),
                    "first_stable_compute_budget_seconds": challenger_thresholds[0].get(
                        "first_stable_compute_budget_seconds"
                    ),
                    "metric": challenger_thresholds[0].get("metric"),
                    "test_set": challenger_thresholds[0].get("test_set"),
                    "reason": challenger_thresholds[0].get("reason"),
                }
            )
        else:
            reasons = sorted({str(row.get("reason")) for row in challenger_unavailable if row.get("reason")})
            challenger_summaries.append(
                {
                    "challenger_method": challenger_method,
                    "compute_threshold_available": False,
                    "threshold_count": 0,
                    "compute_threshold_unavailable": reasons[0] if len(reasons) == 1 else "No eligible compute threshold exists.",
                    "reasons": reasons,
                }
            )

    summary = {
        "status": "diagnostic_only",
        "scientific_status": "invalid_incomplete_grid" if completeness_invalid else "compute_threshold_diagnostic_only",
        "baseline_method": baseline_method,
        "minimum_robust_seeds": minimum_robust_seeds,
        "reliable_timing_field": RELIABLE_COMPUTE_TIMING_FIELD,
        "completeness_report_available": bool(completeness_report),
        "complete_grid": (not completeness_invalid) if completeness_report else None,
        "threshold_count": len(thresholds),
        "unavailable_threshold_count": len(unavailable),
        "challenger_methods": challenger_methods,
        "challenger_summaries": challenger_summaries,
        "thresholds": thresholds,
        "unavailable_thresholds": unavailable,
        "all_results": threshold_rows,
        "note": (
            "Compute thresholds require reliable total_time_seconds for both MD and the challenger, "
            "a robust_candidate challenger win, and a non-challenger test distribution."
        ),
    }
    if completeness_invalid:
        summary["reason"] = "Incomplete cross-evaluation grid"
        summary["missing_cells"] = [
            *((completeness_report or {}).get("missing_cells") or []),
            *((completeness_report or {}).get("missing_context_cells") or []),
        ]
        summary["missing_primary_metric_cells"] = [
            *((completeness_report or {}).get("missing_primary_metric_cells") or []),
            *((completeness_report or {}).get("missing_primary_metric_context_cells") or []),
        ]
    return threshold_rows, summary


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
        and row.get("seed_stability_status") == "robust_candidate"
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
            str(row.get("dataset_context_key") or ""),
            row.get("test_set"),
        )
        for row in summary_rows
        if row.get("metric") == metric
        and row.get("test_set") in allowed_test_sets
        and row.get("stable_winner") == method
        and row.get("seed_stability_status") == "robust_candidate"
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
            str(row.get("dataset_context_key") or ""),
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


def ranking_rows_by_test_set(
    rows: list[dict[str, Any]],
    methods: list[str],
    test_sets: list[str],
    primary_metric: str,
) -> dict[str, list[dict[str, Any]]]:
    rankings_by_test_set: dict[str, list[dict[str, Any]]] = {}
    for test_set in test_sets:
        ranking_rows = []
        for method in methods:
            context_keys = sorted(
                {
                    dataset_context_key(row)
                    for row in rows
                    if str(row.get("train_method")) == method
                    and str(row.get("test_set")) == test_set
                    and finite(row.get(primary_metric))
                }
            )
            for context_key in context_keys:
                values = [
                    float(row[primary_metric])
                    for row in rows
                    if str(row.get("train_method")) == method
                    and str(row.get("test_set")) == test_set
                    and dataset_context_key(row) == context_key
                    and finite(row.get(primary_metric))
                ]
                if values:
                    ranking_rows.append(
                        {
                            "method": method,
                            "dataset_context_key": context_key,
                            "mean": mean(values),
                            "n": len(values),
                        }
                    )
        rankings_by_test_set[test_set] = sorted(
            ranking_rows,
            key=lambda item: (
                float(item["mean"]) if item["mean"] is not None else math.inf,
                str(item.get("dataset_context_key") or ""),
                str(item.get("method") or ""),
            ),
        )
    return rankings_by_test_set


def ordered_challengers(methods: list[str]) -> list[str]:
    preferred = [METHOD_ATOM, "random_cartesian"]
    seen = [method for method in preferred if method in methods and method != METHOD_MD]
    seen.extend(sorted(method for method in methods if method not in {*preferred, METHOD_MD}))
    return seen


def json_number_or_none(value: Any) -> int | float | None:
    if not finite(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def first_available_compute_threshold(
    rows: list[dict[str, Any]],
    *,
    challenger: str,
    primary_metric: str,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("challenger_method") == challenger
        and row.get("metric") == primary_metric
        and row.get("threshold_available") in (True, "True", "true", "1", 1)
        and finite(row.get("first_stable_compute_budget_seconds"))
    ]
    candidates.sort(
        key=lambda row: (
            threshold_size_sort_key(row.get("first_stable_compute_budget_seconds")),
            threshold_size_sort_key(row.get("challenger_dataset_size")),
            str(row.get("dataset_context_key") or ""),
        )
    )
    return candidates[0] if candidates else None


def primary_pairwise_groups(
    pairwise_rows: list[dict[str, Any]],
    *,
    primary_metric: str,
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pairwise_rows:
        if row.get("metric") != primary_metric:
            continue
        grouped[
            (
                row.get("experiment_id"),
                row.get("dataset_context_key"),
                row.get("epoch"),
                row.get("test_set"),
                row.get("challenger_method"),
            )
        ].append(row)
    return grouped


def robust_md_by_challenger(
    pairwise_rows: list[dict[str, Any]],
    *,
    challengers: list[str],
    primary_metric: str,
) -> dict[str, bool]:
    result = {challenger: False for challenger in challengers}
    for (_experiment_id, _context_key, _epoch, _test_set, challenger), group_rows in primary_pairwise_groups(
        pairwise_rows,
        primary_metric=primary_metric,
    ).items():
        challenger = str(challenger)
        if challenger not in result:
            continue
        representative = group_rows[0]
        if representative.get("test_distribution_role") == "challenger_distribution":
            continue
        if (
            representative.get("seed_stability_status") == "robust_candidate"
            and representative.get("stable_winner_across_seeds") == METHOD_MD
            and representative.get("warning_status") != "severe_warning"
        ):
            result[challenger] = True
    return result


def unstable_primary_pairwise_groups(
    pairwise_rows: list[dict[str, Any]],
    *,
    primary_metric: str,
) -> list[dict[str, Any]]:
    unstable = []
    seen: set[tuple[str, ...]] = set()
    for (_experiment_id, _context_key, _epoch, _test_set, _challenger), group_rows in primary_pairwise_groups(
        pairwise_rows,
        primary_metric=primary_metric,
    ).items():
        representative = group_rows[0]
        if representative.get("seed_stability_status") != "unstable":
            continue
        key = tuple(
            str(representative.get(part) or "")
            for part in ("experiment_id", "dataset_context_key", "epoch", "test_set", "metric", "challenger_method")
        )
        if key in seen:
            continue
        seen.add(key)
        unstable.append(
            {
                "experiment_id": representative.get("experiment_id"),
                "dataset_context_key": representative.get("dataset_context_key"),
                "epoch": representative.get("epoch"),
                "test_set": representative.get("test_set"),
                "metric": representative.get("metric"),
                "challenger_method": representative.get("challenger_method"),
                "seed_stability_status": representative.get("seed_stability_status"),
                "seeds": representative.get("seed_stability_seeds"),
            }
        )
    return unstable


def distribution_specific_primary_wins(
    pairwise_rows: list[dict[str, Any]],
    *,
    primary_metric: str,
) -> list[dict[str, Any]]:
    wins = []
    seen: set[tuple[str, ...]] = set()
    for row in pairwise_rows:
        if row.get("metric") != primary_metric:
            continue
        challenger = str(row.get("challenger_method") or "")
        if row.get("winner") != challenger:
            continue
        if row.get("win_scope_status") != "distribution_specific_only":
            continue
        key = (
            str(row.get("experiment_id") or ""),
            str(row.get("dataset_context_key") or ""),
            str(row.get("metric") or ""),
            str(row.get("challenger_method") or ""),
            str(row.get("test_set") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        wins.append(
            {
                "experiment_id": row.get("experiment_id"),
                "dataset_context_key": row.get("dataset_context_key"),
                "metric": row.get("metric"),
                "challenger_method": challenger,
                "winning_test_set": row.get("test_set"),
                "reason": "Win occurs only on the challenger's own test distribution.",
            }
        )
    return wins


def final_recommendation_invalid(
    *,
    recommendation: dict[str, Any],
    status: str,
    reason: str,
    limitations: list[str],
    missing_cells: list[str] | None = None,
) -> dict[str, Any]:
    recommendation.update(
        {
            "status": status,
            "scientific_status": "not_scientifically_valid",
            "winner": None,
            "first_dataset_size_surpassing_md": None,
            "first_compute_budget_surpassing_md": None,
            "reason": reason,
            "limitations": limitations,
        }
    )
    if missing_cells is not None:
        recommendation["missing_cells"] = missing_cells
    return recommendation


def build_final_recommendation(
    rows: list[dict[str, Any]],
    legacy_recommendation: dict[str, Any],
    pairwise_vs_baseline_rows: list[dict[str, Any]],
    dataset_size_threshold_rows: list[dict[str, Any]],
    compute_budget_threshold_rows: list[dict[str, Any]],
    nway_ranking_summary: dict[str, Any],
    *,
    primary_metric: str,
    completeness_report: dict[str, Any] | None = None,
    completeness_path: Path | None = None,
    minimum_robust_seeds: int = 3,
) -> dict[str, Any]:
    methods = sorted({str(row.get("train_method")) for row in rows if row.get("train_method")})
    baseline = METHOD_MD
    challengers = ordered_challengers(methods)
    completeness_report = completeness_report or {}
    report_missing_required = [
        *(str(cell) for cell in completeness_report.get("missing_cells") or []),
        *(str(cell) for cell in completeness_report.get("missing_context_cells") or []),
    ]
    report_missing_primary = [
        *(str(cell) for cell in completeness_report.get("missing_primary_metric_cells") or []),
        *(str(cell) for cell in completeness_report.get("missing_primary_metric_context_cells") or []),
    ]
    missing_required = sorted(
        dict.fromkeys([*report_missing_required, *legacy_recommendation.get("missing_required_cells", [])])
    )
    missing_primary = sorted(
        dict.fromkeys([*report_missing_primary, *legacy_recommendation.get("missing_primary_metric_cells", [])])
    )
    missing_cells = sorted(dict.fromkeys([*missing_required, *missing_primary]))
    grid_invalid = bool(missing_required) or (
        completeness_report_is_invalid(completeness_report) and not missing_primary
    )
    complete_grid = not grid_invalid
    leakage = leakage_diagnostics(rows)
    severe_warnings = list(legacy_recommendation.get("severe_warnings") or [])
    fairness_warnings = [warning for warning in severe_warnings if not is_leakage_warning(warning)]
    fairness_status = "severe_mismatch" if fairness_warnings else "passed"
    leakage_status = (
        "invalid_leakage"
        if leakage["invalid"]
        else "severe_or_inconclusive_leakage"
        if leakage["inconclusive"]
        else "exploratory_leakage"
        if leakage["exploratory_only"]
        else "clean"
        if leakage["clean"]
        else "not_reported"
    )
    metric_policy = legacy_recommendation.get("metric_policy") or {}
    valid_seed_count = int(legacy_recommendation.get("valid_seed_count") or 0)
    unstable_groups = unstable_primary_pairwise_groups(
        pairwise_vs_baseline_rows,
        primary_metric=primary_metric,
    )
    distribution_specific_wins = distribution_specific_primary_wins(
        pairwise_vs_baseline_rows,
        primary_metric=primary_metric,
    )
    limitations = [
        "N-way ranking is diagnostic and does not declare a robust recommendation.",
    ]
    if distribution_specific_wins:
        limitations.append(
            "At least one challenger win is distribution-specific and is not treated as a general win over MD."
        )

    recommendation: dict[str, Any] = dict(legacy_recommendation)
    recommendation.update(
        {
            "baseline": baseline,
            "challengers": challengers,
            "selected_methods": methods,
            "primary_metric": primary_metric,
            "complete_grid": complete_grid,
            "grid_status": "complete" if complete_grid and not missing_primary else "incomplete",
            "seed_stability": {
                "minimum_robust_seeds": minimum_robust_seeds,
                "valid_seed_count": valid_seed_count,
                "status": (
                    "insufficient_seeds"
                    if valid_seed_count < minimum_robust_seeds
                    else "unstable"
                    if unstable_groups
                    else "candidate_stable"
                ),
                "unstable_groups": unstable_groups,
            },
            "leakage_status": leakage_status,
            "leakage_diagnostics": leakage,
            "fairness_provenance_status": fairness_status,
            "fairness_provenance_warnings": fairness_warnings,
            "distribution_specific_wins": distribution_specific_wins,
            "pairwise_vs_md_thresholds": {
                "dataset_size": dataset_size_threshold_rows,
                "compute_budget": compute_budget_threshold_rows,
            },
            "nway_ranking_diagnostic": {
                "status": nway_ranking_summary.get("status"),
                "scientific_status": nway_ranking_summary.get("scientific_status"),
                "robust_candidate_group_count": nway_ranking_summary.get("robust_candidate_group_count"),
                "unstable_group_count": nway_ranking_summary.get("unstable_group_count"),
                "missing_method_cell_count": nway_ranking_summary.get("missing_method_cell_count"),
                "tie_group_count": nway_ranking_summary.get("tie_group_count"),
            },
            "first_dataset_size_surpassing_md": None,
            "first_compute_budget_surpassing_md": None,
            "limitations": limitations,
        }
    )
    if completeness_path is not None:
        recommendation["completeness_report"] = str(completeness_path)

    if grid_invalid:
        reason = (
            "Incomplete 3-method cross-evaluation grid"
            if "random_cartesian" in challengers
            else "Incomplete cross-evaluation grid"
        )
        return final_recommendation_invalid(
            recommendation=recommendation,
            status="invalid_incomplete_grid",
            reason=reason,
            limitations=limitations,
            missing_cells=missing_cells,
        )
    if missing_primary:
        return final_recommendation_invalid(
            recommendation=recommendation,
            status="insufficient_primary_metric",
            reason=f"Primary metric {primary_metric!r} is missing in required cells",
            limitations=limitations,
            missing_cells=missing_cells,
        )
    if leakage["invalid"] or leakage["inconclusive"] or leakage["exploratory_only"]:
        leakage_block_status = (
            "invalid_leakage"
            if leakage["invalid"]
            else "leakage_exploratory_only"
            if leakage["exploratory_only"]
            else "scientifically_inconclusive_leakage"
        )
        return final_recommendation_invalid(
            recommendation=recommendation,
            status=leakage_block_status,
            reason="Leakage diagnostics prevent a robust scientific recommendation",
            limitations=limitations,
        )
    if fairness_warnings:
        return final_recommendation_invalid(
            recommendation=recommendation,
            status="fairness_provenance_mismatch",
            reason="Severe fairness or provenance warnings prevent a robust scientific recommendation",
            limitations=limitations,
        )
    if valid_seed_count < minimum_robust_seeds:
        recommendation.update(
            {
                "status": "insufficient_seeds",
                "scientific_status": "exploratory_only",
                "winner": None,
                "first_dataset_size_surpassing_md": None,
                "first_compute_budget_surpassing_md": None,
                "reason": f"Fewer than {minimum_robust_seeds} valid seeds are available",
                "limitations": limitations,
            }
        )
        return recommendation
    if unstable_groups:
        return final_recommendation_invalid(
            recommendation=recommendation,
            status="unstable_seed_winner",
            reason="Winner changes across seeds",
            limitations=limitations,
        )
    if not metric_policy.get("robust_recommendation_allowed", False):
        recommendation.update(
            {
                "status": "metric_policy_exploratory_only",
                "scientific_status": "exploratory_only",
                "winner": None,
                "reason": str(metric_policy.get("reason") or "Primary metric is not recommendation-grade."),
                "limitations": limitations,
            }
        )
        return recommendation

    threshold_candidates = [
        row
        for row in dataset_size_threshold_rows
        if row.get("metric") == primary_metric
        and row.get("baseline_method") == baseline
        and row.get("challenger_method") in challengers
        and row.get("threshold_available") in (True, "True", "true", "1", 1)
        and row.get("distribution_scope") == "general"
        and row.get("seed_stability_status") == "robust_candidate"
    ]
    threshold_candidates.sort(
        key=lambda row: (
            threshold_size_sort_key(row.get("first_stable_dataset_size")),
            str(row.get("challenger_method") or ""),
            str(row.get("dataset_context_key") or ""),
        )
    )
    if threshold_candidates:
        winning_threshold = threshold_candidates[0]
        winner = str(winning_threshold.get("challenger_method"))
        compute_threshold = first_available_compute_threshold(
            compute_budget_threshold_rows,
            challenger=winner,
            primary_metric=primary_metric,
        )
        recommendation.update(
            {
                "status": "challenger_surpasses_md",
                "scientific_status": "robust_comparison",
                "winner": winner,
                "first_dataset_size_surpassing_md": json_number_or_none(
                    winning_threshold.get("first_stable_dataset_size")
                ),
                "first_compute_budget_surpassing_md": json_number_or_none(
                    compute_threshold.get("first_stable_compute_budget_seconds")
                    if compute_threshold
                    else None
                ),
                "winning_threshold": winning_threshold,
                "reason": (
                    f"{winner} first surpasses md on {primary_metric} at dataset size "
                    f"{winning_threshold.get('first_stable_dataset_size')} with stable general "
                    f"pairwise-vs-MD evidence."
                ),
                "limitations": limitations,
            }
        )
        return recommendation

    md_robust = robust_md_by_challenger(
        pairwise_vs_baseline_rows,
        challengers=challengers,
        primary_metric=primary_metric,
    )
    if challengers and all(md_robust.get(challenger) for challenger in challengers):
        recommendation.update(
            {
                "status": "md_conservative_win",
                "scientific_status": "robust_comparison",
                "winner": baseline,
                "first_dataset_size_surpassing_md": None,
                "first_compute_budget_surpassing_md": None,
                "reason": (
                    f"No challenger has a stable general pairwise win over md on {primary_metric}; "
                    "md is stable against all selected challengers."
                ),
                "md_robust_by_challenger": md_robust,
                "limitations": limitations,
            }
        )
        return recommendation

    return final_recommendation_invalid(
        recommendation=recommendation,
        status="no_general_robust_winner",
        reason="No stable general pairwise-vs-MD winner is available",
        limitations=limitations,
    )


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
    if {METHOD_MD, METHOD_ATOM}.issubset(set(methods)) and required_test_sets.issubset(DEFAULT_BINARY_TEST_SETS):
        required_test_sets = set(DEFAULT_BINARY_TEST_SETS)
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
    metric_policy = metric_policy_diagnostics(
        rows,
        primary_metric=primary_metric,
        required_cells=required_cells,
        primary_metric_cells=primary_metric_cells,
        missing_primary_metric_cells=missing_primary_metric_cells,
    )
    severe_warnings = sorted(
        {
            str(value)
            for row in rows
            for value in (
                row.get("siesta_settings_warning"),
                row.get("model_config_warning"),
                row.get("basis_pseudopotential_warning"),
                row.get("material_compatibility_warning"),
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
    leakage = leakage_diagnostics(rows)
    non_leakage_severe_warnings = [
        warning for warning in severe_warnings if not is_leakage_warning(warning)
    ]
    allowed_test_sets = set(test_sets)
    conservative_win_test_sets = {test_set for test_set in allowed_test_sets if test_set in {"test_md", "test_mixed"}}
    if not conservative_win_test_sets:
        conservative_win_test_sets = set()
    primary_summary = [
        row
        for row in summary_rows
        if row.get("metric") == primary_metric and row.get("test_set") in allowed_test_sets
    ]
    max_seeds = max(
        (int(row.get("n_seeds") or row.get("seeds_compared") or 0) for row in primary_summary),
        default=len({row_seed(row) for row in rows}),
    )
    max_valid_seeds = max(
        (int(row.get("valid_seed_count") or 0) for row in primary_summary),
        default=len(valid_stability_seeds({row_seed(row) for row in rows})),
    )
    single_seed_only = max_valid_seeds <= 1
    insufficient_robust_seeds = max_valid_seeds < minimum_robust_seeds
    first_atom_size = first_method_win(
        summary_rows,
        primary_metric,
        METHOD_ATOM,
        allowed_test_sets=conservative_win_test_sets,
        minimum_robust_seeds=minimum_robust_seeds,
    )
    first_md_size = first_method_win(
        summary_rows,
        primary_metric,
        METHOD_MD,
        allowed_test_sets=conservative_win_test_sets,
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
        allowed_test_sets=conservative_win_test_sets,
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
        allowed_test_sets=conservative_win_test_sets,
        minimum_robust_seeds=minimum_robust_seeds,
    )
    atom_only_wins = [
        row
        for row in summary_rows
        if row.get("metric") == primary_metric
        and row.get("test_set") == "test_siesta_fc_cartesian"
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
    robust_seed_candidate_rows = [
        row
        for row in primary_summary
        if row.get("seed_stability_status") == "robust_candidate"
    ]
    unstable_seed_rows = [
        row
        for row in primary_summary
        if row.get("seed_stability_status") == "unstable"
    ]
    exploratory_seed_rows = [
        row
        for row in primary_summary
        if row.get("seed_stability_status") == "exploratory_only"
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
    rankings_by_test_set = ranking_rows_by_test_set(rows, methods, test_sets, primary_metric)
    nway_leaders_by_test_set = {
        test_set: ranking[0]["method"]
        for test_set, ranking in rankings_by_test_set.items()
        if ranking
    }
    nway_consensus_leaders = sorted(set(nway_leaders_by_test_set.values()))
    nway_consensus_leader = (
        nway_consensus_leaders[0]
        if len(nway_consensus_leaders) == 1 and len(nway_leaders_by_test_set) == len(test_sets)
        else None
    )
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
            "SIESTA FC Cartesian only wins on test_siesta_fc_cartesian; this is distribution-specific "
            "and is not enough for a conservative recommendation."
        )
    nway_methods = set(methods) - {METHOD_MD, METHOD_ATOM}
    if leakage["invalid"]:
        status = "invalid_leakage"
        reason = "Geometry leakage invalidates the scientific comparison."
    elif missing_cells:
        status = "insufficient_cross_evaluation"
        reason = "The cross-evaluation grid is incomplete; no winner should be declared."
    elif missing_primary_metric_cells:
        status = "insufficient_primary_metric"
        reason = f"{metric_policy['reason']} No winner should be declared."
    elif metric_policy["policy_status"] in {"occupied_metric_not_well_defined", "unknown_metric"}:
        status = "insufficient_primary_metric"
        reason = f"{metric_policy['reason']} No winner should be declared."
    elif non_leakage_severe_warnings:
        status = "inconclusive"
        reason = "Validation warnings invalidate a robust winner: " + " | ".join(non_leakage_severe_warnings)
    elif leakage["inconclusive"]:
        status = "inconclusive"
        reason = "Geometry leakage diagnostics prevent a robust scientific comparison."
    elif leakage["exploratory_only"]:
        status = "leakage_exploratory_only"
        reason = (
            "Random Cartesian same-family sharing across splits makes this result exploratory only; "
            "no robust winner is declared."
        )
    elif nway_methods and not missing_cells and not missing_primary_metric_cells and not severe_warnings:
        if nway_consensus_leader:
            status = "nway_consensus_win"
            reason = (
                f"{nway_consensus_leader} has the best mean {primary_metric} "
                "in every frozen test set. Treat this as exploratory until the "
                "minimum seed count is reached."
            )
        else:
            status = "nway_complete_mixed_ranking"
            reason = (
                "N-way comparison cells are complete, but different methods lead "
                "on different frozen test sets; inspect rankings_by_test_set."
            )
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

    robust_statuses = {
        "md_conservative_win",
        "atom_displacement_conservative_win",
        "mixed_conservative_result",
    }
    if status == "invalid_leakage":
        scientific_status = "invalid_leakage"
    elif status == "leakage_exploratory_only":
        scientific_status = "exploratory_only"
    elif (
        status in robust_statuses
        and not insufficient_robust_seeds
        and not underpowered_stable_wins
        and metric_policy["robust_recommendation_allowed"]
    ):
        scientific_status = "robust_comparison"
    elif status in {"nway_consensus_win", "nway_complete_mixed_ranking"} and not severe_warnings:
        scientific_status = "exploratory"
    elif (
        status in robust_statuses
        and not missing_cells
        and not missing_primary_metric_cells
        and not severe_warnings
        and not metric_policy["robust_recommendation_allowed"]
    ):
        scientific_status = "exploratory"
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

    if (
        scientific_status == "exploratory"
        and status in robust_statuses
        and not metric_policy["robust_recommendation_allowed"]
    ):
        reason = f"{reason} Metric policy downgrade: {metric_policy['reason']}"

    winner = None
    if scientific_status == "robust_comparison":
        if status == "md_conservative_win":
            winner = METHOD_MD
        elif status == "atom_displacement_conservative_win":
            winner = METHOD_ATOM

    return {
        "status": status,
        "scientific_status": scientific_status,
        "winner": winner,
        "primary_metric": primary_metric,
        "metric_policy": metric_policy,
        "primary_metric_policy_status": metric_policy["policy_status"],
        "primary_metric_recommendation_role": metric_policy["recommendation_role"],
        "robust_primary_metric_allowed": metric_policy["robust_recommendation_allowed"],
        "reason": reason,
        "n_seeds": max_seeds,
        "valid_seed_count": max_valid_seeds,
        "minimum_robust_seeds": minimum_robust_seeds,
        "single_seed_warning": single_seed_only,
        "insufficient_robust_seeds": insufficient_robust_seeds,
        "underpowered_stable_wins": underpowered_stable_wins,
        "robust_seed_candidate_rows": robust_seed_candidate_rows,
        "unstable_seed_rows": unstable_seed_rows,
        "exploratory_seed_rows": exploratory_seed_rows,
        "methods_seen": methods,
        "test_sets_seen": test_sets,
        "frozen_test_hashes": frozen_test_hashes,
        "model_checkpoint_hashes": model_checkpoint_hashes,
        "missing_required_cells": missing_cells,
        "missing_primary_metric_cells": missing_primary_metric_cells,
        "severe_warnings": severe_warnings,
        "leakage_diagnostics": leakage,
        "rankings_by_test_set": rankings_by_test_set,
        "nway_leaders_by_test_set": nway_leaders_by_test_set,
        "nway_consensus_leader": nway_consensus_leader,
        "first_dataset_size_where_md_beats_atom_displacement": first_md_size,
        "first_dataset_size_where_atom_displacement_beats_md": first_atom_size,
        "first_compute_budget_where_md_beats_atom_displacement": first_md_compute,
        "first_compute_budget_where_atom_displacement_beats_md": first_atom_compute,
        "conservative_rule": (
            "A method wins only if it beats the other method on the same frozen test set, "
            "across seeds, and on test_md or test_mixed rather than only its own distribution."
        ),
        "seed_stability_rule": (
            f"n_seeds < {minimum_robust_seeds} is exploratory_only; changing winners are unstable; "
            f"the same winner across at least {minimum_robust_seeds} valid seeds is a robust_candidate; "
            "severe warnings override seed stability."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-metric", default="low_energy_rmse_eV")
    parser.add_argument("--minimum-robust-seeds", type=int, default=3)
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument(
        "--completeness-report",
        type=Path,
        help=(
            "Optional cross_evaluation_completeness.json. If omitted, the analyzer "
            "checks the output directory for that artifact."
        ),
    )
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
    nway_method_rows = build_nway_method_summary(
        rows,
        metrics,
        aggregation_mode=args.aggregation_mode,
    )
    nway_ranking_rows = rank_nway_methods(
        nway_method_rows,
        lower_is_better=not args.higher_is_better,
    )
    nway_ranking_summary = summarize_nway_ranking(nway_ranking_rows)
    nway_ranking_summary["aggregation_mode"] = args.aggregation_mode
    pairwise_vs_baseline_rows = build_pairwise_vs_baseline(
        nway_method_rows,
        baseline_method=METHOD_MD,
        lower_is_better=not args.higher_is_better,
    )
    pairwise_vs_baseline_summary = summarize_pairwise_vs_baseline(
        pairwise_vs_baseline_rows,
        baseline_method=METHOD_MD,
    )
    pairwise_vs_baseline_summary["aggregation_mode"] = args.aggregation_mode
    compute_rows = sorted(
        [row for row in pair_rows if finite(row.get("compute_budget_seconds"))],
        key=lambda row: (
            str(row.get("metric")),
            float(row.get("compute_budget_seconds")),
            str(row.get("test_set")),
        ),
    )
    legacy_recommendation = build_recommendation(
        rows,
        summary_rows,
        pair_rows,
        primary_metric=args.primary_metric,
        minimum_robust_seeds=args.minimum_robust_seeds,
    )
    legacy_recommendation["aggregation_mode"] = args.aggregation_mode
    completeness_report, completeness_path = load_completeness_report(
        args.completeness_report,
        args.output_dir,
    )
    dataset_size_threshold_rows, dataset_size_threshold_summary = build_dataset_size_thresholds_vs_md(
        pairwise_vs_baseline_rows,
        completeness_report=completeness_report,
        baseline_method=METHOD_MD,
        minimum_robust_seeds=args.minimum_robust_seeds,
    )
    dataset_size_threshold_summary["aggregation_mode"] = args.aggregation_mode
    compute_budget_threshold_rows, compute_budget_threshold_summary = build_compute_budget_thresholds_vs_md(
        pairwise_vs_baseline_rows,
        completeness_report=completeness_report,
        baseline_method=METHOD_MD,
        minimum_robust_seeds=args.minimum_robust_seeds,
    )
    compute_budget_threshold_summary["aggregation_mode"] = args.aggregation_mode
    if completeness_path is not None:
        dataset_size_threshold_summary["completeness_report"] = str(completeness_path)
        compute_budget_threshold_summary["completeness_report"] = str(completeness_path)
    recommendation = build_final_recommendation(
        rows,
        legacy_recommendation,
        pairwise_vs_baseline_rows,
        dataset_size_threshold_rows,
        compute_budget_threshold_rows,
        nway_ranking_summary,
        primary_metric=args.primary_metric,
        completeness_report=completeness_report,
        completeness_path=completeness_path,
        minimum_robust_seeds=args.minimum_robust_seeds,
    )
    recommendation["aggregation_mode"] = args.aggregation_mode
    recommendation["legacy_recommendation"] = legacy_recommendation
    for row in pair_rows:
        row["aggregation_mode"] = args.aggregation_mode
    for row in summary_rows:
        row["aggregation_mode"] = args.aggregation_mode
    for row in nway_method_rows:
        row["aggregation_mode"] = args.aggregation_mode
    for row in nway_ranking_rows:
        row["aggregation_mode"] = args.aggregation_mode
    for row in pairwise_vs_baseline_rows:
        row["aggregation_mode"] = args.aggregation_mode
    for row in dataset_size_threshold_rows:
        row["aggregation_mode"] = args.aggregation_mode
    for row in compute_budget_threshold_rows:
        row["aggregation_mode"] = args.aggregation_mode

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "winner_by_dataset_size.csv", pair_rows)
    write_rows(args.output_dir / "winner_summary.csv", summary_rows)
    write_rows(args.output_dir / "winner_by_compute_budget.csv", compute_rows)
    write_rows(args.output_dir / "nway_method_summary.csv", nway_method_rows)
    write_rows(args.output_dir / "nway_ranking.csv", nway_ranking_rows)
    (args.output_dir / "nway_ranking_summary.json").write_text(
        json.dumps(nway_ranking_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_rows(args.output_dir / "pairwise_vs_baseline.csv", pairwise_vs_baseline_rows)
    (args.output_dir / "pairwise_vs_baseline_summary.json").write_text(
        json.dumps(pairwise_vs_baseline_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_rows(args.output_dir / "dataset_size_thresholds_vs_md.csv", dataset_size_threshold_rows)
    (args.output_dir / "dataset_size_thresholds_vs_md.json").write_text(
        json.dumps(dataset_size_threshold_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_rows(args.output_dir / "compute_budget_thresholds_vs_md.csv", compute_budget_threshold_rows)
    (args.output_dir / "compute_budget_thresholds_vs_md.json").write_text(
        json.dumps(compute_budget_threshold_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(recommendation, ensure_ascii=False, allow_nan=False))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
