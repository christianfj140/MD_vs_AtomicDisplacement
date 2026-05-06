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
import hashlib
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


def file_sha256(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_size(path_value: str) -> int | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    return path.stat().st_size


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


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
        item["structure_sha256"] = file_sha256(item.get("structure_path", ""))
        item["hamiltonian_sha256"] = file_sha256(item.get("hamiltonian_path", ""))
        item["run_out_sha256"] = file_sha256(item.get("run_out_path", ""))
        item["metadata_sha256"] = file_sha256(item.get("metadata_path", ""))
        item["structure_size_bytes"] = file_size(item.get("structure_path", ""))
        item["hamiltonian_size_bytes"] = file_size(item.get("hamiltonian_path", ""))
        frozen.append(item)
    return frozen


def frozen_manifest(test_set: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [
        {
            "sample_id": row.get("sample_id", ""),
            "method": row.get("method", ""),
            "test_set": test_set,
            "structure_path": row.get("structure_path", ""),
            "source_structure_path": row.get("source_structure_path", ""),
            "structure_sha256": row.get("structure_sha256", ""),
            "structure_size_bytes": row.get("structure_size_bytes"),
            "hamiltonian_path": row.get("hamiltonian_path", ""),
            "source_hamiltonian_path": row.get("source_hamiltonian_path", ""),
            "hamiltonian_sha256": row.get("hamiltonian_sha256", ""),
            "hamiltonian_size_bytes": row.get("hamiltonian_size_bytes"),
            "run_out_sha256": row.get("run_out_sha256", ""),
            "metadata_sha256": row.get("metadata_sha256", ""),
            "status": row.get("status", ""),
        }
        for row in sorted(rows, key=lambda item: (str(item.get("method", "")), str(item.get("sample_id", ""))))
    ]
    methods: dict[str, int] = {}
    for sample in samples:
        method = str(sample.get("method") or "unknown")
        methods[method] = methods.get(method, 0) + 1
    payload = {
        "test_set": test_set,
        "samples": samples,
        "sample_count": len(samples),
        "method_counts": methods,
    }
    payload["frozen_test_hash"] = content_hash(samples)
    return payload


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
    parser.add_argument(
        "--mixed-selection",
        choices=["spread", "prefix"],
        default="spread",
        help="How to choose rows for test_mixed when --mixed-max-per-method is set.",
    )
    parser.add_argument("--allow-train-test-overlap-debug", action="store_true")
    return parser


def select_spread(rows: list[dict[str, str]], count: int | None) -> list[dict[str, str]]:
    if count is None or count >= len(rows):
        return list(rows)
    if count <= 0:
        return []
    used: set[int] = set()
    selected: list[int] = []
    for index in range(count):
        target = min(len(rows) - 1, int((index + 0.5) * len(rows) / count))
        if target in used:
            target = min(
                (candidate for candidate in range(len(rows)) if candidate not in used),
                key=lambda candidate: abs(candidate - target),
            )
        used.add(target)
        selected.append(target)
    return [rows[index] for index in sorted(selected)]


def select_mixed_part(rows: list[dict[str, str]], count: int | None, mode: str) -> list[dict[str, str]]:
    if mode == "prefix":
        return rows[:count] if count is not None else list(rows)
    return select_spread(rows, count)


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
        md_part = select_mixed_part(md_rows, args.mixed_max_per_method, args.mixed_selection)
        atom_part = select_mixed_part(atom_rows, args.mixed_max_per_method, args.mixed_selection)
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
        frozen_payload = frozen_manifest(test_set, frozen)
        (args.output_dir / test_set / "frozen_test_manifest.json").write_text(
            json.dumps(frozen_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        output_rows.extend(
            {
                "test_set": test_set,
                "sample_id": row.get("sample_id"),
                "method": row.get("method"),
                "structure_path": row.get("structure_path"),
                "hamiltonian_path": row.get("hamiltonian_path"),
                "structure_sha256": row.get("structure_sha256"),
                "hamiltonian_sha256": row.get("hamiltonian_sha256"),
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
                "frozen_manifest": str(args.output_dir / name / "frozen_test_manifest.json"),
                "frozen_test_hash": json.loads(
                    (args.output_dir / name / "frozen_test_manifest.json").read_text(encoding="utf-8")
                )["frozen_test_hash"],
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
