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
        return value
    return json.dumps(value if value is not None else {}, sort_keys=True, ensure_ascii=False)


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
    prediction_summary = read_json(result_dir / "prediction_summary.json")
    evaluation_manifest = read_json(metrics_root / "manifest.json")
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
            manifest.get("frozen_test_warning"),
            manifest.get("siesta_settings_warning"),
            manifest.get("model_config_warning"),
            manifest.get("basis_pseudopotential_warning"),
            manifest.get("checkpoint_selection_warning"),
            manifest.get("reproducibility_warning"),
            manifest.get("nested_subset_warning"),
            manifest.get("manifest_warning"),
            manifest.get("artifact_hash_warning"),
            manifest.get("matrix_warning"),
            manifest.get("prediction_warning"),
            evaluation_warning,
        ]
        warning_text = " | ".join(str(item) for item in warnings if item not in (None, "", False))
        rows.append(
            {
                "experiment_id": experiment_id,
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
                "recipe_set_hash": manifest.get("recipe_set_hash"),
                "train_dataset_label": manifest.get("train_dataset_label"),
                "train_recipe_id": manifest.get("train_recipe_id"),
                "train_recipe_label": manifest.get("train_recipe_label"),
                "train_block_id": manifest.get("train_block_id"),
                "train_block_label": manifest.get("train_block_label"),
                "train_generation_parameters_json": manifest.get("train_generation_parameters_json"),
                "md_dataset_size": manifest.get("md_dataset_size"),
                "atom_dataset_size": manifest.get("atom_dataset_size"),
                "random_dataset_size": manifest.get("random_dataset_size"),
                "compute_budget_mode": manifest.get("compute_budget_mode"),
                "md_siesta_reference_count": manifest.get("md_siesta_reference_count"),
                "atomdisp_siesta_reference_count": manifest.get("atomdisp_siesta_reference_count"),
                "budget_ratio": manifest.get("budget_ratio"),
                "budget_mismatch_warning": manifest.get("budget_mismatch_warning"),
                "leakage_warning": manifest.get("leakage_warning"),
                "frozen_test_warning": manifest.get("frozen_test_warning"),
                "frozen_test_hash": manifest.get("frozen_test_hash"),
                "frozen_test_manifest": manifest.get("frozen_test_manifest"),
                "siesta_settings_hash": manifest.get("siesta_settings_hash"),
                "siesta_settings_warning": manifest.get("siesta_settings_warning"),
                "model_config_hash": manifest.get("model_config_hash"),
                "model_config_warning": manifest.get("model_config_warning"),
                "basis_pseudopotential_warning": manifest.get("basis_pseudopotential_warning"),
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
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--cross-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows: list[dict[str, Any]] = []
    if args.cross_root.exists():
        for result_dir in sorted(path for path in args.cross_root.iterdir() if path.is_dir()):
            rows.extend(aggregate_one(result_dir, args.experiment_id))
    output = args.output_dir / "cross_evaluation_metrics.csv"
    write_rows(output, rows)
    summary = {
        "ok": bool(rows),
        "rows": len(rows),
        "output": str(output),
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
