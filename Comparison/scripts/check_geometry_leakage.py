#!/usr/bin/env python3
"""Detect exact and near-duplicate geometries crossing train/test splits.

This first diagnostic intentionally uses the atom ordering present in RUN.fdf.
It does not perform alignment, so the reported RMSD is transparent and
conservative for the comparison pipeline, where samples should share a stable
ordering.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.setdefault("manifest_path", str(path))
    return rows


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def as_int(value: str | None) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def structure_path(row: dict[str, str], manifest_path: Path | None = None) -> Path | None:
    value = row.get("structure_path") or row.get("RUN.fdf") or row.get("run_fdf")
    if not value:
        sample_dir = row.get("sample_dir")
        if not sample_dir:
            return None
        value = str(Path(sample_dir) / "RUN.fdf")
    path = Path(value).expanduser()
    if not path.is_absolute() and manifest_path is not None:
        path = manifest_path.parent / path
    return path


def parse_run_fdf_coordinates(path: Path) -> list[tuple[float, float, float]]:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    inside = False
    coords: list[tuple[float, float, float]] = []
    for line in text:
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        lower = clean.lower()
        if lower.startswith("%block atomiccoordinatesandatomicspecies"):
            inside = True
            continue
        if inside and lower.startswith("%endblock atomiccoordinatesandatomicspecies"):
            break
        if not inside:
            continue
        parts = clean.split()
        if len(parts) < 3:
            continue
        try:
            coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    return coords


def compare_coords(
    left: list[tuple[float, float, float]],
    right: list[tuple[float, float, float]],
) -> tuple[float | None, float | None]:
    if not left or not right or len(left) != len(right):
        return None, None
    squared = 0.0
    max_abs = 0.0
    count = 0
    for a, b in zip(left, right):
        for av, bv in zip(a, b):
            diff = av - bv
            squared += diff * diff
            max_abs = max(max_abs, abs(diff))
            count += 1
    return math.sqrt(squared / count), max_abs


def load_manifest_rows(paths: list[Path], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_rows(path):
            row = dict(row)
            row["manifest_path"] = str(path)
            row["split"] = row.get("split") or split
            rows.append(row)
    return rows


def family_key(row: dict[str, Any]) -> tuple[str, ...]:
    keys = [
        "source_run",
        "base_sample_id",
        "displacement_magnitude",
        "displaced_atom",
        "displacement_axis",
        "displacement_sign",
    ]
    return tuple(str(row.get(key, "")) for key in keys)


def analyze(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    rmsd_threshold: float,
    max_diff_threshold: float,
    neighbor_frame_window: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coords_cache: dict[Path, list[tuple[float, float, float]]] = {}
    report: list[dict[str, Any]] = []
    parser_errors: list[str] = []

    def coords_for(row: dict[str, Any]) -> list[tuple[float, float, float]]:
        manifest_path = Path(str(row.get("manifest_path", ""))) if row.get("manifest_path") else None
        path = structure_path(row, manifest_path)
        if path is None:
            return []
        if path not in coords_cache:
            try:
                coords_cache[path] = parse_run_fdf_coordinates(path)
            except Exception as exc:
                parser_errors.append(f"{path}: {exc}")
                coords_cache[path] = []
        return coords_cache[path]

    exact = 0
    near = 0
    md_neighbors = 0
    atom_family = 0
    for train in train_rows:
        train_coords = coords_for(train)
        for test in test_rows:
            test_coords = coords_for(test)
            rmsd, max_diff = compare_coords(train_coords, test_coords)
            reasons: list[str] = []
            if rmsd is not None and max_diff is not None:
                if max_diff == 0:
                    reasons.append("exact_duplicate_geometry")
                    exact += 1
                elif rmsd <= rmsd_threshold or max_diff <= max_diff_threshold:
                    reasons.append("near_duplicate_geometry")
                    near += 1

            train_method = str(train.get("method", "")).lower()
            test_method = str(test.get("method", "")).lower()
            if train_method == "md" and test_method == "md":
                train_frame = as_int(train.get("frame_index") or train.get("time_index"))
                test_frame = as_int(test.get("frame_index") or test.get("time_index"))
                if train_frame is not None and test_frame is not None:
                    if abs(train_frame - test_frame) <= neighbor_frame_window:
                        reasons.append("md_neighboring_frames_cross_split")
                        md_neighbors += 1

            if train_method.startswith("atom") and test_method.startswith("atom"):
                key = family_key(train)
                if any(key) and key == family_key(test):
                    reasons.append("atom_displacement_same_family_cross_split")
                    atom_family += 1

            if reasons:
                report.append(
                    {
                        "train_sample_id": train.get("sample_id", ""),
                        "test_sample_id": test.get("sample_id", ""),
                        "train_split": train.get("split", "train"),
                        "test_split": test.get("split", "test"),
                        "train_method": train.get("method", ""),
                        "test_method": test.get("method", ""),
                        "rmsd": rmsd if rmsd is not None else "",
                        "max_abs_diff": max_diff if max_diff is not None else "",
                        "reasons": ";".join(sorted(set(reasons))),
                    }
                )

    summary = {
        "ok": not report,
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "pairs_checked": len(train_rows) * len(test_rows),
        "warnings": len(report),
        "exact_duplicates": exact,
        "near_duplicates": near,
        "md_neighbor_warnings": md_neighbors,
        "atom_displacement_family_warnings": atom_family,
        "parser_errors": parser_errors,
        "rmsd_threshold": rmsd_threshold,
        "max_diff_threshold": max_diff_threshold,
        "neighbor_frame_window": neighbor_frame_window,
    }
    return report, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) or [
        "train_sample_id",
        "test_sample_id",
        "reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument("--test-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rmsd-threshold", type=float, default=1e-4)
    parser.add_argument("--max-diff-threshold", type=float, default=1e-4)
    parser.add_argument("--neighbor-frame-window", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train_rows = load_manifest_rows(args.train_manifest, "train")
    test_rows = load_manifest_rows(args.test_manifest, "test")
    report, summary = analyze(
        train_rows,
        test_rows,
        rmsd_threshold=args.rmsd_threshold,
        max_diff_threshold=args.max_diff_threshold,
        neighbor_frame_window=args.neighbor_frame_window,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "geometry_leakage_report.csv", report)
    (args.output_dir / "geometry_leakage_summary.json").write_text(
        json.dumps(
            {
                **summary,
                "outputs": {
                    "geometry_leakage_report": str(args.output_dir / "geometry_leakage_report.csv"),
                    "geometry_leakage_summary": str(args.output_dir / "geometry_leakage_summary.json"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
