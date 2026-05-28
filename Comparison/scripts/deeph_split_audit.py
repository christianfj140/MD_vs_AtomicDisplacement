#!/usr/bin/env python3
"""Audit DeepH train/validation/test splits against a frozen benchmark split."""

from __future__ import annotations

import configparser
import csv
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "graph2mat_deeph_deeph_split_audit_v1"
STATUS_VALID = "valid"
STATUS_UNVERIFIED = "invalid_unverified_deeph_split"
STATUS_INCOMPATIBLE = "invalid_incompatible_splits"
SPLITS = ("train", "validation", "test")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "deeph_dataset_index",
        "processed_dir",
        "frozen_split",
        "actual_deeph_split",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_train_split_config(train_config_path: Path) -> tuple[int | None, dict[str, float], list[str]]:
    errors: list[str] = []
    if not train_config_path.exists():
        return None, {}, [f"DeepH train config is missing: {train_config_path}"]
    config = configparser.ConfigParser()
    config.read(train_config_path)
    try:
        seed = config.getint("basic", "seed")
    except Exception as exc:  # configparser exceptions vary by missing section/key.
        seed = None
        errors.append(f"DeepH train config does not define [basic] seed: {exc}")
    ratios: dict[str, float] = {}
    for key in ("train_ratio", "val_ratio", "test_ratio"):
        try:
            ratios[key] = config.getfloat("train", key)
        except Exception as exc:
            errors.append(f"DeepH train config does not define [train] {key}: {exc}")
    return seed, ratios, errors


def processed_sample_dirs(processed_dir: Path) -> list[Path]:
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        return []
    folders: list[Path] = []
    for root, _dirs, files in os.walk(processed_dir):
        if "rc.h5" in files:
            folders.append(Path(root))
    return sorted(folders, key=lambda path: str(path))


def deeph_index_split_map(dataset_size: int, ratios: dict[str, float], seed: int) -> tuple[dict[int, str], list[str]]:
    errors: list[str] = []
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return {}, ["NumPy is required to reproduce DeepH np.random.shuffle split indices."]
    sizes = {
        "train": int(float(ratios.get("train_ratio", 0.0)) * dataset_size),
        "validation": int(float(ratios.get("val_ratio", 0.0)) * dataset_size),
        "test": int(float(ratios.get("test_ratio", 0.0)) * dataset_size),
    }
    if sum(sizes.values()) > dataset_size:
        errors.append(f"DeepH split sizes exceed dataset size: {sizes} > {dataset_size}")
    if any(size <= 0 for size in sizes.values()):
        errors.append(f"DeepH split sizes must be non-empty for benchmark comparability: {sizes}")
    if errors:
        return {}, errors
    indices = list(range(dataset_size))
    np.random.seed(int(seed))
    np.random.shuffle(indices)
    actual: dict[int, str] = {}
    cursor = 0
    for split in SPLITS:
        count = sizes[split]
        for index in indices[cursor : cursor + count]:
            actual[int(index)] = split
        cursor += count
    return actual, []


def _relative_key(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.name


def _raw_mirror_by_relative_dir(raw_mirror: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_root = Path(str(raw_mirror.get("raw_dir") or ""))
    rows: dict[str, dict[str, Any]] = {}
    for row in raw_mirror.get("rows") or []:
        if not isinstance(row, dict):
            continue
        raw_dir = Path(str(row.get("raw_dir") or ""))
        rows[_relative_key(raw_dir, raw_root)] = dict(row)
    return rows


def _frozen_split_by_sample_id(frozen_split_manifest: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in frozen_split_manifest.get("rows") or []:
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id") or row.get("graph2mat_sample_id") or row.get("deeph_sample_id") or "").strip()
        split = str(row.get("split") or "").strip()
        if sample_id and split:
            result[sample_id] = split
    return result


def audit_deeph_split(
    *,
    frozen_split_manifest: dict[str, Any],
    raw_mirror: dict[str, Any],
    processed_dir: Path,
    train_config_path: Path,
    output_json: Path | None = None,
    output_csv: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    seed, ratios, config_errors = load_train_split_config(train_config_path)
    errors.extend(config_errors)
    raw_seed = raw_mirror.get("seed")
    if seed is not None and raw_seed is not None and int(raw_seed) != int(seed):
        errors.append(f"DeepH raw mirror seed {raw_seed} does not match train config seed {seed}.")

    raw_by_dir = _raw_mirror_by_relative_dir(raw_mirror)
    frozen_by_sample = _frozen_split_by_sample_id(frozen_split_manifest)
    folders = processed_sample_dirs(processed_dir)
    if not folders:
        errors.append(f"No DeepH processed sample directories with rc.h5 found under {processed_dir}.")

    processed_root = Path(processed_dir)
    unknown_processed: list[str] = []
    rows: list[dict[str, Any]] = []
    if seed is not None and ratios and folders:
        split_by_index, split_errors = deeph_index_split_map(len(folders), ratios, seed)
        errors.extend(split_errors)
        if not split_errors:
            for index, folder in enumerate(folders):
                key = _relative_key(folder, processed_root)
                mirror_row = raw_by_dir.get(key)
                if mirror_row is None:
                    unknown_processed.append(str(folder))
                    continue
                sample_id = str(mirror_row.get("sample_id") or "").strip()
                frozen_split = frozen_by_sample.get(sample_id, "")
                actual_split = split_by_index.get(index, "")
                status = "ok" if frozen_split and actual_split == frozen_split else "mismatch"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "deeph_dataset_index": index,
                        "processed_dir": str(folder),
                        "frozen_split": frozen_split,
                        "actual_deeph_split": actual_split,
                        "status": status,
                    }
                )
    if unknown_processed:
        errors.append("Processed DeepH samples cannot be mapped to raw mirror rows: " + ", ".join(unknown_processed[:10]))

    raw_sample_ids = {str(row.get("sample_id") or "") for row in raw_by_dir.values()}
    processed_sample_ids = {row["sample_id"] for row in rows if row.get("sample_id")}
    missing_processed_sample_ids = sorted(raw_sample_ids - processed_sample_ids)
    if missing_processed_sample_ids:
        errors.append(
            "DeepH processed output is missing raw mirror samples: "
            + ", ".join(missing_processed_sample_ids[:10])
        )
    if set(frozen_by_sample) != processed_sample_ids:
        errors.append(
            "DeepH processed sample IDs do not match frozen split IDs: "
            f"missing={sorted(set(frozen_by_sample) - processed_sample_ids)[:10]} "
            f"extra={sorted(processed_sample_ids - set(frozen_by_sample))[:10]}"
        )

    mismatches = [row for row in rows if row.get("status") != "ok"]
    if mismatches:
        warnings.append(f"{len(mismatches)} DeepH split assignments differ from frozen split.")

    if errors and not mismatches:
        status = STATUS_UNVERIFIED
    elif errors or mismatches:
        status = STATUS_INCOMPATIBLE
    else:
        status = STATUS_VALID
    valid = status == STATUS_VALID
    audit = {
        "schema": SCHEMA,
        "status": status,
        "valid": valid,
        "comparability_status": "valid" if valid else status,
        "scientific_status": "valid" if valid else STATUS_INCOMPATIBLE,
        "robust_winner_allowed": valid,
        "frozen_split_hash": frozen_split_manifest.get("split_hash"),
        "raw_mirror_seed": raw_seed,
        "train_config_seed": seed,
        "split_ratios": ratios,
        "processed_dir": str(processed_dir),
        "train_config_path": str(train_config_path),
        "dataset_size": len(folders),
        "rows": rows,
        "mismatched_rows": mismatches,
        "errors": errors,
        "warnings": warnings,
    }
    if output_json is not None:
        write_json(output_json, audit)
        audit["path"] = str(output_json)
    if output_csv is not None:
        write_csv(output_csv, rows)
        audit["csv_path"] = str(output_csv)
    return audit
