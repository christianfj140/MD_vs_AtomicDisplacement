#!/usr/bin/env python3
"""Build frozen common test sets for MD-vs-AtomDisplacement cross evaluation.

Examples
--------
Create the standard three test sets from method-specific test manifests::

    python Comparison/scripts/build_common_tests.py \
      --md-test-manifest md/test_manifest.csv \
      --atomdisp-test-manifest atom/test_manifest.csv \
      --train-manifest md/train_manifest.csv \
      --train-manifest atom/train_manifest.csv \
      --output-dir Comparison/results/exp_001/common_tests
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {"sample_id", "method", "structure_path", "hamiltonian_path", "status"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise RuntimeError(f"{path} missing required columns: {sorted(missing)}")
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = sorted(REQUIRED_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("method", ""), row.get("sample_id", ""))


def valid_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if str(row.get("status", "")).lower() in {"valid", "ok", "completed"}]


def copy_file_if_exists(src_value: str, destination_dir: Path) -> str:
    if not src_value:
        return ""
    src = Path(src_value)
    if not src.exists() or not src.is_file():
        return src_value
    destination_dir.mkdir(parents=True, exist_ok=True)
    dst = destination_dir / src.name
    shutil.copy2(src, dst)
    return str(dst)


def freeze_rows(rows: list[dict[str, str]], test_set: str, output_dir: Path) -> list[dict[str, Any]]:
    frozen = []
    for row in rows:
        sample_id = row["sample_id"]
        sample_dir = output_dir / test_set / "samples" / sample_id
        item = dict(row)
        item["test_set"] = test_set
        item["source_structure_path"] = row.get("structure_path", "")
        item["source_hamiltonian_path"] = row.get("hamiltonian_path", "")
        item["structure_path"] = copy_file_if_exists(row.get("structure_path", ""), sample_dir)
        item["hamiltonian_path"] = copy_file_if_exists(row.get("hamiltonian_path", ""), sample_dir)
        if row.get("run_out_path"):
            item["run_out_path"] = copy_file_if_exists(row.get("run_out_path", ""), sample_dir)
        if row.get("metadata_path"):
            item["metadata_path"] = copy_file_if_exists(row.get("metadata_path", ""), sample_dir)
        frozen.append(item)
    return frozen


def ensure_independent(test_rows: list[dict[str, str]], train_rows: list[dict[str, str]], allow_overlap: bool) -> list[str]:
    if allow_overlap:
        return []
    train_keys = {row_key(row) for row in train_rows}
    overlaps = sorted(row_key(row) for row in test_rows if row_key(row) in train_keys)
    return [f"{method}:{sample_id}" for method, sample_id in overlaps]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md-test-manifest", type=Path, required=True)
    parser.add_argument("--atomdisp-test-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--test-sets",
        default="test_md,test_atomdisp,test_mixed",
        help="Comma-separated subset of test_md,test_atomdisp,test_mixed.",
    )
    parser.add_argument("--mixed-max-per-method", type=int, default=None)
    parser.add_argument("--allow-train-test-overlap-debug", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    requested = {item.strip() for item in args.test_sets.split(",") if item.strip()}
    unknown = requested - {"test_md", "test_atomdisp", "test_mixed"}
    if unknown:
        raise RuntimeError(f"Unknown test sets: {sorted(unknown)}")

    md_rows = valid_rows(read_rows(args.md_test_manifest))
    atom_rows = valid_rows(read_rows(args.atomdisp_test_manifest))
    train_rows: list[dict[str, str]] = []
    for manifest in args.train_manifest:
        train_rows.extend(read_rows(manifest))

    test_sets: dict[str, list[dict[str, str]]] = {}
    if "test_md" in requested:
        test_sets["test_md"] = md_rows
    if "test_atomdisp" in requested:
        test_sets["test_atomdisp"] = atom_rows
    if "test_mixed" in requested:
        md_part = md_rows[: args.mixed_max_per_method] if args.mixed_max_per_method else md_rows
        atom_part = atom_rows[: args.mixed_max_per_method] if args.mixed_max_per_method else atom_rows
        test_sets["test_mixed"] = md_part + atom_part

    output_rows = []
    errors: list[str] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for test_set, rows in test_sets.items():
        overlaps = ensure_independent(rows, train_rows, args.allow_train_test_overlap_debug)
        if overlaps:
            errors.append(f"{test_set} overlaps train manifests: {overlaps}")
        frozen = freeze_rows(rows, test_set, args.output_dir)
        write_rows(args.output_dir / test_set / "test_manifest.csv", frozen)
        output_rows.extend(
            {
                "test_set": test_set,
                "sample_id": row.get("sample_id"),
                "method": row.get("method"),
                "structure_path": row.get("structure_path"),
                "hamiltonian_path": row.get("hamiltonian_path"),
                "status": row.get("status"),
            }
            for row in frozen
        )

    write_rows(args.output_dir / "common_test_sets.csv", output_rows)
    summary = {
        "ok": not errors,
        "errors": errors,
        "test_sets": {
            name: {
                "samples": len(rows),
                "manifest": str(args.output_dir / name / "test_manifest.csv"),
            }
            for name, rows in test_sets.items()
        },
        "outputs": {
            "common_test_sets": str(args.output_dir / "common_test_sets.csv"),
        },
    }
    (args.output_dir / "common_test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
