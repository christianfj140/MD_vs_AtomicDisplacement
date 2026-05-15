#!/usr/bin/env python3
"""Aggregate cross-evaluation sparse/spectral/DOS metrics into one table.

Examples
--------
    python Comparison/scripts/aggregate_cross_metrics.py \
      --experiment-id exp_001 \
      --cross-root Comparison/results/exp_001/cross_evaluations \
      --output-dir Comparison/results/exp_001/summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from method_registry import normalize_method_id, normalize_method_mapping, normalize_test_set_id
from material_provenance import (
    MATERIAL_FLAT_FIELDS,
    MATERIAL_MAP_FIELDS,
    flatten_material_provenance,
    material_compatibility_warning,
    material_maps_from_manifest,
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value in ("", None):
                row[key] = None
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = value
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def json_text(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return json.dumps(normalize_method_mapping(parsed), sort_keys=True, ensure_ascii=False)
    return json.dumps(normalize_method_mapping(value if value is not None else {}), sort_keys=True, ensure_ascii=False)


def canonical_manifest_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(manifest)
    if normalized.get("train_method") not in (None, ""):
        normalized["train_method"] = normalize_method_id(normalized.get("train_method"), allow_unknown=True)
    if normalized.get("test_method") not in (None, ""):
        normalized["test_method"] = normalize_method_id(normalized.get("test_method"), allow_unknown=True)
    if normalized.get("test_set") not in (None, ""):
        normalized["test_set"] = normalize_test_set_id(normalized.get("test_set"))
        if normalized.get("test_method") in (None, ""):
            suffix = str(normalized["test_set"]).removeprefix("test_")
            normalized["test_method"] = normalize_method_id(suffix, allow_unknown=True)
    for key in (
        "dataset_size_by_method",
        "dataset_label_by_method",
        "recipe_id_by_method",
        "recipe_label_by_method",
        "recipe_set_hash_by_method",
        "training_tag_by_method",
        "training_plan_label_by_method",
        "training_plan_settings_by_method",
        *MATERIAL_MAP_FIELDS,
    ):
        normalized[key] = normalize_method_mapping(normalized.get(key))
    return normalized


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["experiment_id", "train_method", "test_set", "sample_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def cell_id(train_method: Any, test_set: Any) -> str:
    method = normalize_method_id(train_method, allow_unknown=True)
    test = normalize_test_set_id(test_set)
    return f"{method} on {test}"


def context_cell_id(pair_id: Any, train_method: Any, test_set: Any) -> str:
    pair = str(pair_id or "").strip()
    base = cell_id(train_method, test_set)
    return f"{pair} :: {base}" if pair else base


def cell_sort_key(value: str) -> tuple[str, str]:
    if " :: " in value:
        pair_id, remainder = value.split(" :: ", 1)
        method, test = cell_sort_key(remainder)
        return method, f"{test} {pair_id}"
    if " on " not in value:
        return value, ""
    train_method, test_set = value.split(" on ", 1)
    return train_method, test_set


def expected_cells_from_grid(grid: dict[str, Any]) -> set[str]:
    cells: set[str] = set()
    raw_cells = grid.get("expected_cells")
    if isinstance(raw_cells, list):
        for item in raw_cells:
            if isinstance(item, dict):
                method = item.get("train_method")
                test_set = item.get("test_set")
                if method not in (None, "") and test_set not in (None, ""):
                    cells.add(cell_id(method, test_set))
            elif isinstance(item, str) and " on " in item:
                method, test_set = item.split(" on ", 1)
                cells.add(cell_id(method, test_set))
    if cells:
        return cells

    methods = grid.get("selected_methods") or grid.get("train_methods") or []
    test_sets = (
        grid.get("selected_frozen_test_sets")
        or grid.get("selected_test_sets")
        or grid.get("test_sets")
        or []
    )
    return {
        cell_id(method, test_set)
        for method in methods
        for test_set in test_sets
        if method not in (None, "") and test_set not in (None, "")
    }


def expected_context_cells_from_grid(grid: dict[str, Any]) -> set[str]:
    cells: set[str] = set()
    raw_cells = grid.get("expected_context_cells") or grid.get("expected_cross_evaluation_cells")
    if not isinstance(raw_cells, list):
        return cells
    for item in raw_cells:
        if isinstance(item, dict):
            pair_id = item.get("pair_id")
            method = item.get("train_method")
            test_set = item.get("test_set")
            if pair_id not in (None, "") and method not in (None, "") and test_set not in (None, ""):
                cells.add(context_cell_id(pair_id, method, test_set))
        elif isinstance(item, str) and " on " in item:
            cells.add(str(item))
    return cells


def load_expected_grid(path: Path | None, output_dir: Path) -> tuple[dict[str, Any], str, Path | None]:
    grid_path = path or output_dir / "cross_evaluation_expected_grid.json"
    if grid_path.exists():
        return read_json(grid_path), "artifact", grid_path
    return {}, "observed_rows_legacy", None


def observed_expected_cells(rows: list[dict[str, Any]]) -> set[str]:
    methods = sorted({str(row.get("train_method")) for row in rows if row.get("train_method") not in (None, "")})
    test_sets = sorted({str(row.get("test_set")) for row in rows if row.get("test_set") not in (None, "")})
    return {cell_id(method, test_set) for method in methods for test_set in test_sets}


def observed_context_cells(rows: list[dict[str, Any]]) -> set[str]:
    return {
        context_cell_id(row.get("pair_id"), row.get("train_method"), row.get("test_set"))
        for row in rows
        if row.get("pair_id") not in (None, "")
        and row.get("train_method") not in (None, "")
        and row.get("test_set") not in (None, "")
    }


def build_completeness_report(
    *,
    experiment_id: str,
    rows: list[dict[str, Any]],
    expected_grid: dict[str, Any],
    expected_grid_source: str,
    expected_grid_path: Path | None,
    primary_metric: str | None = None,
) -> dict[str, Any]:
    expected_cells = expected_cells_from_grid(expected_grid) if expected_grid else observed_expected_cells(rows)
    expected_context_cells = expected_context_cells_from_grid(expected_grid) if expected_grid else observed_context_cells(rows)
    actual_cells = {
        cell_id(row.get("train_method"), row.get("test_set"))
        for row in rows
        if row.get("train_method") not in (None, "") and row.get("test_set") not in (None, "")
    }
    actual_context_cells = observed_context_cells(rows)
    missing_cells = sorted(expected_cells - actual_cells, key=cell_sort_key)
    extra_cells = sorted(actual_cells - expected_cells, key=cell_sort_key)
    missing_context_cells = sorted(expected_context_cells - actual_context_cells, key=cell_sort_key)
    extra_context_cells = sorted(actual_context_cells - expected_context_cells, key=cell_sort_key) if expected_context_cells else []
    missing_primary_metric_cells: list[str] = []
    missing_primary_metric_context_cells: list[str] = []
    metric_name = str(primary_metric or "").strip()
    if metric_name:
        cells_with_primary_metric = {
            cell_id(row.get("train_method"), row.get("test_set"))
            for row in rows
            if row.get("train_method") not in (None, "")
            and row.get("test_set") not in (None, "")
            and finite(row.get(metric_name))
        }
        missing_primary_metric_cells = sorted(expected_cells - cells_with_primary_metric, key=cell_sort_key)
        context_cells_with_primary_metric = {
            context_cell_id(row.get("pair_id"), row.get("train_method"), row.get("test_set"))
            for row in rows
            if row.get("pair_id") not in (None, "")
            and row.get("train_method") not in (None, "")
            and row.get("test_set") not in (None, "")
            and finite(row.get(metric_name))
        }
        missing_primary_metric_context_cells = sorted(
            expected_context_cells - context_cells_with_primary_metric,
            key=cell_sort_key,
        )
    complete = (
        not missing_cells
        and not extra_cells
        and not missing_primary_metric_cells
        and not missing_context_cells
        and not extra_context_cells
        and not missing_primary_metric_context_cells
    )
    report = {
        "experiment_id": experiment_id,
        "expected_grid": str(expected_grid_path) if expected_grid_path else None,
        "expected_grid_source": expected_grid_source,
        "primary_metric": metric_name or None,
        "expected_cell_count": len(expected_cells),
        "actual_cell_count": len(actual_cells),
        "expected_context_cell_count": len(expected_context_cells),
        "actual_context_cell_count": len(actual_context_cells),
        "actual_row_count": len(rows),
        "expected_cells": sorted(expected_cells, key=cell_sort_key),
        "actual_cells": sorted(actual_cells, key=cell_sort_key),
        "expected_context_cells": sorted(expected_context_cells, key=cell_sort_key),
        "actual_context_cells": sorted(actual_context_cells, key=cell_sort_key),
        "missing_cells": missing_cells,
        "extra_unexpected_cells": extra_cells,
        "extra_cells": extra_cells,
        "unexpected_cells": extra_cells,
        "missing_context_cells": missing_context_cells,
        "extra_unexpected_context_cells": extra_context_cells,
        "extra_context_cells": extra_context_cells,
        "unexpected_context_cells": extra_context_cells,
        "missing_primary_metric_cells": missing_primary_metric_cells,
        "missing_primary_metric_context_cells": missing_primary_metric_context_cells,
        "complete": complete,
        "scientific_status": "valid_grid" if complete else "invalid_incomplete_grid",
    }
    if expected_grid_source != "artifact":
        report["warning"] = "No cross_evaluation_expected_grid.json was provided; expected cells were inferred from observed methods and test sets."
    return report


def write_completeness_csv(path: Path, report: dict[str, Any]) -> None:
    issue_rows = []
    for key, issue_type in (
        ("missing_cells", "missing_cell"),
        ("extra_unexpected_cells", "unexpected_cell"),
        ("missing_primary_metric_cells", "missing_primary_metric"),
        ("missing_context_cells", "missing_context_cell"),
        ("extra_unexpected_context_cells", "unexpected_context_cell"),
        ("missing_primary_metric_context_cells", "missing_primary_metric_context"),
    ):
        for cell in report.get(key, []) or []:
            cell_text = str(cell)
            if " :: " in cell_text:
                _pair_id, cell_body = cell_text.split(" :: ", 1)
            else:
                cell_body = cell_text
            train_method, test_set = cell_body.split(" on ", 1) if " on " in cell_body else (cell_body, "")
            issue_rows.append({"issue": issue_type, "train_method": train_method, "test_set": test_set, "cell_id": cell_text})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["issue", "train_method", "test_set", "cell_id"])
        writer.writeheader()
        writer.writerows(issue_rows)


def join_by_sample(*groups: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    joined: dict[str, dict[str, Any]] = {}
    duplicate_errors: list[str] = []
    for rows in groups:
        seen_in_group: set[str] = set()
        for row in rows:
            sample = str(row.get("sample"))
            if sample in seen_in_group:
                duplicate_errors.append(f"duplicate sample row {sample!r}")
            seen_in_group.add(sample)
            joined.setdefault(sample, {})
            joined[sample].update(row)
    return joined, sorted(set(duplicate_errors))


def issue_messages(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    messages = []
    for item in items:
        if isinstance(item, dict):
            sample = item.get("sample")
            kind = item.get("kind")
            error = item.get("error")
            parts = [str(part) for part in (sample, kind, error) if part not in (None, "", False)]
            if parts:
                messages.append(":".join(parts))
        elif item not in (None, "", False):
            messages.append(str(item))
    return messages


def infer_metadata(path: Path) -> tuple[str, str]:
    name = path.name
    if "__on__" in name:
        train_method, test_set = name.split("__on__", 1)
        return train_method, test_set
    if "_on_" in name:
        train_method, test_set = name.split("_on_", 1)
        return train_method, test_set
    return "unknown", name


def aggregate_one(result_dir: Path, experiment_id: str) -> list[dict[str, Any]]:
    metrics_root = result_dir / "metrics"
    sparse = read_rows(metrics_root / "sparse_metrics.csv")
    spectral = read_rows(metrics_root / "spectral_metrics.csv")
    dos = read_rows(metrics_root / "dos_metrics.csv")
    manifest = read_json(result_dir / "cross_evaluation_manifest.json")
    if not manifest:
        train_method, test_set = infer_metadata(result_dir)
        manifest = {"train_method": train_method, "test_set": test_set}
    manifest = canonical_manifest_metadata(manifest)
    prediction_summary = read_json(result_dir / "prediction_summary.json")
    evaluation_manifest = read_json(metrics_root / "manifest.json")
    material_provenance = flatten_material_provenance(
        manifest.get("material_provenance") if isinstance(manifest.get("material_provenance"), dict) else {},
        manifest,
    )
    material_maps = material_maps_from_manifest(manifest)
    material_warning = manifest.get("material_compatibility_warning") or material_compatibility_warning(material_maps)
    joined, duplicate_errors = join_by_sample(sparse, spectral, dos)
    if duplicate_errors:
        raise RuntimeError(f"{result_dir}: duplicate sample ids in metric CSVs: {' | '.join(duplicate_errors)}")
    evaluation_warning = manifest.get("evaluation_warning")
    evaluation_messages = (
        duplicate_errors
        + issue_messages(evaluation_manifest.get("fatal_errors"))
        + issue_messages(evaluation_manifest.get("warnings"))
    )
    if evaluation_messages:
        warning_parts = [str(evaluation_warning)] if evaluation_warning else []
        warning_parts.extend(evaluation_messages)
        evaluation_warning = " | ".join(warning_parts)
    rows = []
    for sample, row in sorted(joined.items()):
        warnings = [
            manifest.get("budget_mismatch_warning"),
            manifest.get("leakage_warning"),
            " | ".join(str(item) for item in manifest.get("leakage_severe_warnings", []) or []),
            manifest.get("frozen_test_warning"),
            manifest.get("siesta_settings_warning"),
            manifest.get("model_config_warning"),
            manifest.get("training_plan_settings_warning"),
            manifest.get("basis_pseudopotential_warning"),
            manifest.get("checkpoint_selection_warning"),
            manifest.get("reproducibility_warning"),
            manifest.get("nested_subset_warning"),
            manifest.get("manifest_warning"),
            manifest.get("artifact_hash_warning"),
            manifest.get("matrix_warning"),
            manifest.get("prediction_warning"),
            material_warning,
            evaluation_warning,
        ]
        warning_text = " | ".join(str(item) for item in warnings if item not in (None, "", False))
        output_row = {
                "experiment_id": experiment_id,
                "pair_id": manifest.get("pair_id"),
                "train_method": manifest.get("train_method"),
                "test_set": manifest.get("test_set"),
                "test_method": manifest.get("test_method"),
                "dataset_size": manifest.get("dataset_size"),
                "train_dataset_size": manifest.get("train_dataset_size", manifest.get("dataset_size")),
                "dataset_size_by_method": json_text(manifest.get("dataset_size_by_method")),
                "dataset_label_by_method": json_text(manifest.get("dataset_label_by_method")),
                "recipe_id_by_method": json_text(manifest.get("recipe_id_by_method")),
                "recipe_label_by_method": json_text(manifest.get("recipe_label_by_method")),
                "recipe_set_hash_by_method": json_text(manifest.get("recipe_set_hash_by_method")),
                "training_tag_by_method": json_text(manifest.get("training_tag_by_method")),
                "training_plan_label_by_method": json_text(manifest.get("training_plan_label_by_method")),
                "training_plan_settings_by_method": json_text(manifest.get("training_plan_settings_by_method")),
                "recipe_set_hash": manifest.get("recipe_set_hash"),
                "train_dataset_label": manifest.get("train_dataset_label"),
                "train_recipe_id": manifest.get("train_recipe_id"),
                "train_recipe_label": manifest.get("train_recipe_label"),
                "train_block_id": manifest.get("train_block_id"),
                "train_block_label": manifest.get("train_block_label"),
                "train_generation_parameters_json": manifest.get("train_generation_parameters_json"),
                "train_training_tag": manifest.get("train_training_tag") or manifest.get("training_tag"),
                "train_training_index": manifest.get("train_training_index") or manifest.get("training_index"),
                "train_training_settings": json_text(
                    manifest.get("train_training_settings") or manifest.get("training_settings")
                ),
                "train_training_plan_index": manifest.get("train_training_plan_index")
                or manifest.get("training_plan_index"),
                "train_training_plan_label": manifest.get("train_training_plan_label")
                or manifest.get("training_plan_label"),
                "train_training_plan_settings": json_text(
                    manifest.get("train_training_plan_settings")
                    or manifest.get("training_plan_settings")
                ),
                "train_training_plan_source_dataset_label": manifest.get(
                    "train_training_plan_source_dataset_label"
                )
                or manifest.get("training_plan_source_dataset_label"),
                "training_tag": manifest.get("training_tag"),
                "training_index": manifest.get("training_index"),
                "training_settings": json_text(manifest.get("training_settings")),
                "training_plan_index": manifest.get("training_plan_index"),
                "training_plan_label": manifest.get("training_plan_label"),
                "training_plan_settings": json_text(manifest.get("training_plan_settings")),
                "training_plan_source_dataset_label": manifest.get("training_plan_source_dataset_label"),
                "md_dataset_size": manifest.get("md_dataset_size"),
                "atom_dataset_size": manifest.get("atom_dataset_size"),
                "random_dataset_size": manifest.get("random_dataset_size"),
                "compute_budget_mode": manifest.get("compute_budget_mode"),
                "md_siesta_reference_count": manifest.get("md_siesta_reference_count"),
                "atomdisp_siesta_reference_count": manifest.get("atomdisp_siesta_reference_count"),
                "budget_ratio": manifest.get("budget_ratio"),
                "budget_mismatch_warning": manifest.get("budget_mismatch_warning"),
                "leakage_warning": manifest.get("leakage_warning"),
                "leakage_scientific_status": manifest.get("leakage_scientific_status"),
                "leakage_severe_warnings": json_text(manifest.get("leakage_severe_warnings")),
                "frozen_test_warning": manifest.get("frozen_test_warning"),
                "frozen_test_hash": manifest.get("frozen_test_hash"),
                "frozen_test_manifest": manifest.get("frozen_test_manifest"),
                "siesta_settings_hash": manifest.get("siesta_settings_hash"),
                "siesta_settings_warning": manifest.get("siesta_settings_warning"),
                "model_config_hash": manifest.get("model_config_hash"),
                "model_config_warning": manifest.get("model_config_warning"),
                "training_plan_settings_warning": manifest.get("training_plan_settings_warning"),
                "basis_pseudopotential_warning": manifest.get("basis_pseudopotential_warning"),
                "material_compatibility_warning": material_warning,
                "strict_comparison_mode": manifest.get("strict_comparison_mode"),
                "md_dataset_label": manifest.get("md_dataset_label"),
                "atom_dataset_label": manifest.get("atom_dataset_label"),
                "random_dataset_label": manifest.get("random_dataset_label"),
                "md_recipe_set_hash": manifest.get("md_recipe_set_hash"),
                "atom_recipe_set_hash": manifest.get("atom_recipe_set_hash"),
                "random_recipe_set_hash": manifest.get("random_recipe_set_hash"),
                "seed": manifest.get("seed"),
                "epoch": manifest.get("epoch"),
                "model_checkpoint": manifest.get("model_checkpoint"),
                "model_checkpoint_sha256": manifest.get("model_checkpoint_sha256"),
                "checkpoint_manifest": manifest.get("checkpoint_manifest"),
                "checkpoint_selection_warning": manifest.get("checkpoint_selection_warning"),
                "reproducibility_warning": manifest.get("reproducibility_warning"),
                "nested_subset_warning": manifest.get("nested_subset_warning"),
                "manifest_warning": manifest.get("manifest_warning"),
                "artifact_hash_warning": manifest.get("artifact_hash_warning"),
                "matrix_warning": manifest.get("matrix_warning"),
                "prediction_warning": manifest.get("prediction_warning"),
                "evaluation_warning": evaluation_warning,
                "warning_status": "warning" if warning_text else "ok",
                "severe_warning_status": "severe" if warning_text else "ok",
                "severe_warnings": warning_text,
                "prediction_dir": manifest.get("prediction_dir"),
                "siesta_reference_dir": manifest.get("siesta_reference_dir"),
                "sample_id": sample,
                "prediction_time_seconds": prediction_summary.get("prediction_time_seconds"),
                "evaluation_time_seconds": manifest.get("evaluation_time_seconds"),
                "total_time_seconds": manifest.get("total_time_seconds"),
                "evaluation_samples_seen": evaluation_manifest.get("samples_seen"),
                **row,
        }
        output_row.update(
            {
                key: json_text(material_provenance.get(key))
                if isinstance(material_provenance.get(key), (dict, list))
                else material_provenance.get(key)
                for key in MATERIAL_FLAT_FIELDS
            }
        )
        output_row.update({key: json_text(material_maps.get(key)) for key in MATERIAL_MAP_FIELDS})
        rows.append(output_row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--cross-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-grid", type=Path, default=None)
    parser.add_argument("--primary-metric", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows: list[dict[str, Any]] = []
    if args.cross_root.exists():
        for result_dir in sorted(path for path in args.cross_root.iterdir() if path.is_dir()):
            rows.extend(aggregate_one(result_dir, args.experiment_id))
    output = args.output_dir / "cross_evaluation_metrics.csv"
    write_rows(output, rows)
    expected_grid, expected_grid_source, expected_grid_path = load_expected_grid(args.expected_grid, args.output_dir)
    completeness = build_completeness_report(
        experiment_id=args.experiment_id,
        rows=rows,
        expected_grid=expected_grid,
        expected_grid_source=expected_grid_source,
        expected_grid_path=expected_grid_path,
        primary_metric=args.primary_metric,
    )
    completeness_path = args.output_dir / "cross_evaluation_completeness.json"
    completeness_path.write_text(
        json.dumps(completeness, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_completeness_csv(args.output_dir / "missing_cross_evaluation_cells.csv", completeness)
    summary = {
        "ok": bool(rows) and bool(completeness["complete"]),
        "rows_ok": bool(rows),
        "rows": len(rows),
        "output": str(output),
        "completeness": str(completeness_path),
        "complete": completeness["complete"],
        "scientific_status": completeness["scientific_status"],
        "train_methods": sorted({str(row.get("train_method")) for row in rows}),
        "test_sets": sorted({str(row.get("test_set")) for row in rows}),
    }
    (args.output_dir / "cross_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
