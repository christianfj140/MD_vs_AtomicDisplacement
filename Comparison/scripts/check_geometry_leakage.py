#!/usr/bin/env python3
"""Detect geometry leakage crossing train/test splits.

The report keeps the original raw-coordinate duplicate checks and also adds
translation/rotation-invariant diagnostics based on internal distance
signatures grouped by the atom species ordering present in RUN.fdf.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - optional diagnostic dependency.
    np = None


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


def parse_run_fdf_geometry(path: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    inside = False
    coords: list[tuple[float, float, float]] = []
    species: list[str] = []
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
            species.append(parts[3] if len(parts) >= 4 else "")
        except ValueError:
            continue
    return species, coords


def parse_run_fdf_coordinates(path: Path) -> list[tuple[float, float, float]]:
    _species, coords = parse_run_fdf_geometry(path)
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


def aligned_rmsd(
    left: list[tuple[float, float, float]],
    right: list[tuple[float, float, float]],
) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    if np is not None:
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        left_centered = left_array - left_array.mean(axis=0)
        right_centered = right_array - right_array.mean(axis=0)
        covariance = left_centered.T @ right_centered
        u_matrix, _singular_values, vt_matrix = np.linalg.svd(covariance)
        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(vt_matrix.T @ u_matrix.T)
        # Row-vector Kabsch: minimizing ||P R - Q|| over rotations gives
        # R = U diag(1,1,det) V^T. The transposed composition rotates the
        # wrong way and reports ~2*theta mismatch for a rotated duplicate.
        rotation = u_matrix @ correction @ vt_matrix
        aligned = left_centered @ rotation
        delta = aligned - right_centered
        return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    # A dependency-free rigid-shape proxy: for fixed atom ordering, equality of
    # all internal distances is invariant to translation and rotation. For the
    # small molecular systems handled by this pipeline it catches the same
    # leakage class that a full Kabsch RMSD would flag, while keeping this
    # diagnostic usable in minimal Python environments.
    distance_diff = distance_signature_max_diff([""] * len(left), left, [""] * len(right), right)
    return distance_diff


def distance_signature(
    species: list[str],
    coords: list[tuple[float, float, float]],
) -> list[tuple[str, str, float]]:
    if not coords:
        return []
    signature: list[tuple[str, str, float]] = []
    for left in range(len(coords)):
        for right in range(left + 1, len(coords)):
            label_pair = sorted([species[left] if left < len(species) else "", species[right] if right < len(species) else ""])
            distance = math.sqrt(
                sum((coords[left][axis] - coords[right][axis]) ** 2 for axis in range(3))
            )
            signature.append((label_pair[0], label_pair[1], distance))
    return sorted(signature)


def distance_signature_max_diff(
    left_species: list[str],
    left: list[tuple[float, float, float]],
    right_species: list[str],
    right: list[tuple[float, float, float]],
) -> float | None:
    left_sig = distance_signature(left_species, left)
    right_sig = distance_signature(right_species, right)
    if not left_sig or len(left_sig) != len(right_sig):
        return None
    max_diff = 0.0
    for left_item, right_item in zip(left_sig, right_sig, strict=False):
        if left_item[:2] != right_item[:2]:
            return None
        max_diff = max(max_diff, abs(left_item[2] - right_item[2]))
    return max_diff


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
        "raw_fc_run_dir",
        "raw_displacement_run_id",
        "base_sample_id",
        "matrix_label",
        "source_matrix_label",
        "displacement_magnitude",
        "displacement_ang",
        "displacement_input",
        "displaced_atom",
        "atom",
        "displacement_axis",
        "direction",
        "displacement_sign",
        "sign",
    ]
    return tuple(str(row.get(key, "")) for key in keys)


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata_path = row.get("metadata_path") or row.get("metadata")
    metadata: dict[str, Any] = {}
    if metadata_path:
        try:
            path = Path(str(metadata_path))
            if not path.is_absolute() and row.get("manifest_path"):
                path = Path(str(row["manifest_path"])).parent / path
            if path.exists():
                metadata = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    return metadata if isinstance(metadata, dict) else {}


def _canonical_family_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    text = str(value).strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, (list, dict)):
        return _canonical_family_value(parsed)
    try:
        return f"{float(text):.12g}"
    except ValueError:
        return text


def _metadata_value(row: dict[str, Any], metadata: dict[str, Any], *keys: str) -> Any:
    block_config = metadata.get("block_config") if isinstance(metadata.get("block_config"), dict) else {}
    family = metadata.get("random_cartesian_family") if isinstance(metadata.get("random_cartesian_family"), dict) else {}
    for key in keys:
        for source in (row, metadata, family, block_config):
            value = source.get(key) if isinstance(source, dict) else None
            if value not in (None, ""):
                return value
    return ""


def random_family_key(row: dict[str, Any]) -> tuple[str, ...]:
    metadata = _row_metadata(row)
    family_id = _metadata_value(row, metadata, "random_cartesian_family_id", "split_group_id")
    distribution = str(_metadata_value(row, metadata, "distribution")).strip().lower()
    sigma = _metadata_value(row, metadata, "sigma_ang")
    uniform_range = _metadata_value(row, metadata, "uniform_range_ang")
    key_values = {
        "base_geometry_hash": _metadata_value(row, metadata, "base_geometry_hash"),
        "distribution": distribution,
        "sigma_ang": sigma if distribution != "uniform" else "",
        "uniform_range_ang": uniform_range if distribution == "uniform" else "",
        "seed_family": _metadata_value(row, metadata, "seed_family", "seed"),
        "move_atoms": _metadata_value(row, metadata, "move_atoms"),
        "species_filter": _metadata_value(row, metadata, "species_filter"),
        "recipe_id": _metadata_value(row, metadata, "recipe_id"),
        "block_id": _metadata_value(row, metadata, "block_id"),
    }
    key = tuple(_canonical_family_value(value) for value in key_values.values())
    if any(key):
        return key
    if family_id not in (None, ""):
        return ("split_group_id", str(family_id))
    return ()


def random_sample_identity_key(row: dict[str, Any]) -> tuple[str, ...]:
    metadata = _row_metadata(row)
    family = random_family_key(row)
    sample_token = _metadata_value(row, metadata, "sample_index", "id", "sample_id")
    if not family or sample_token in (None, ""):
        return ()
    return family + (_canonical_family_value(sample_token),)


def analyze(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    rmsd_threshold: float,
    max_diff_threshold: float,
    neighbor_frame_window: int,
    aligned_rmsd_threshold: float,
    distance_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coords_cache: dict[Path, list[tuple[float, float, float]]] = {}
    species_cache: dict[Path, list[str]] = {}
    report: list[dict[str, Any]] = []
    parser_errors: list[str] = []

    def geometry_for(row: dict[str, Any]) -> tuple[list[str], list[tuple[float, float, float]]]:
        manifest_path = Path(str(row.get("manifest_path", ""))) if row.get("manifest_path") else None
        path = structure_path(row, manifest_path)
        if path is None:
            return [], []
        if path not in coords_cache:
            try:
                species, coords = parse_run_fdf_geometry(path)
                species_cache[path] = species
                coords_cache[path] = coords
            except Exception as exc:
                parser_errors.append(f"{path}: {exc}")
                species_cache[path] = []
                coords_cache[path] = []
        return species_cache[path], coords_cache[path]

    exact = 0
    near = 0
    aligned_near = 0
    distance_near = 0
    md_neighbors = 0
    atom_family = 0
    random_family = 0
    for train in train_rows:
        train_species, train_coords = geometry_for(train)
        for test in test_rows:
            test_species, test_coords = geometry_for(test)
            rmsd, max_diff = compare_coords(train_coords, test_coords)
            aligned_shape_rmsd = aligned_rmsd(train_coords, test_coords)
            distance_max_diff = distance_signature_max_diff(train_species, train_coords, test_species, test_coords)
            reasons: list[str] = []
            if rmsd is not None and max_diff is not None:
                if max_diff == 0:
                    reasons.append("exact_duplicate_geometry")
                    exact += 1
                elif rmsd <= rmsd_threshold or max_diff <= max_diff_threshold:
                    reasons.append("near_duplicate_geometry")
                    near += 1
            if aligned_shape_rmsd is not None and aligned_shape_rmsd <= aligned_rmsd_threshold and max_diff != 0:
                reasons.append("aligned_near_duplicate_geometry")
                aligned_near += 1
            if distance_max_diff is not None and distance_max_diff <= distance_threshold and max_diff != 0:
                reasons.append("internal_distance_near_duplicate_geometry")
                distance_near += 1

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
            if train_method == "random_cartesian" and test_method == "random_cartesian":
                key = random_family_key(train)
                if any(key) and key == random_family_key(test):
                    reasons.append("random_cartesian_same_family_cross_split")
                    random_family += 1
                sample_key = random_sample_identity_key(train)
                if any(sample_key) and sample_key == random_sample_identity_key(test):
                    reasons.append("random_cartesian_same_sample_identity_cross_split")

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
                        "aligned_rmsd": aligned_shape_rmsd if aligned_shape_rmsd is not None else "",
                        "internal_distance_max_diff": distance_max_diff if distance_max_diff is not None else "",
                        "reasons": ";".join(sorted(set(reasons))),
                        "random_cartesian_family_key": "|".join(random_family_key(train))
                        if train_method == "random_cartesian" and test_method == "random_cartesian"
                        else "",
                    }
                )

    warnings: list[str] = []
    severe_warnings: list[str] = []
    if exact:
        severe_warnings.append(f"exact_duplicate_geometry_cross_split: {exact} pair(s)")
    if near:
        warnings.append(f"near_duplicate_geometry_cross_split: {near} pair(s)")
    if aligned_near:
        warnings.append(f"aligned_near_duplicate_geometry_cross_split: {aligned_near} pair(s)")
    if distance_near:
        warnings.append(f"internal_distance_near_duplicate_geometry_cross_split: {distance_near} pair(s)")
    if md_neighbors:
        warnings.append(f"md_neighboring_frames_cross_split: {md_neighbors} pair(s)")
    if atom_family:
        warnings.append(f"atom_displacement_same_family_cross_split: {atom_family} pair(s)")
    if random_family:
        warnings.append(
            "random_cartesian_same_family_cross_split: "
            f"{random_family} pair(s); random_cartesian splits are scientifically non-independent"
        )
    leakage_status_detail = "valid_independent_splits"
    if exact:
        scientific_status = "invalid_leakage"
        leakage_status_detail = "invalid_exact_geometry_leakage"
    elif near or aligned_near or distance_near:
        scientific_status = "scientifically_inconclusive"
        leakage_status_detail = "potential_geometry_leakage"
    elif atom_family or md_neighbors:
        scientific_status = "scientifically_inconclusive"
        leakage_status_detail = "scientifically_non_independent_splits"
    elif random_family:
        scientific_status = "exploratory_only"
        leakage_status_detail = "random_cartesian_same_family_cross_split"
    else:
        scientific_status = "valid_independent_splits"
    summary = {
        "ok": not report,
        "scientific_status": scientific_status,
        "leakage_status_detail": leakage_status_detail,
        "scientifically_independent": not report,
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "pairs_checked": len(train_rows) * len(test_rows),
        "warnings": len(report),
        "exact_duplicates": exact,
        "near_duplicates": near,
        "aligned_near_duplicates": aligned_near,
        "internal_distance_near_duplicates": distance_near,
        "md_neighbor_warnings": md_neighbors,
        "atom_displacement_family_warnings": atom_family,
        "random_cartesian_family_warnings": random_family,
        "warning_messages": warnings,
        "severe_warnings": severe_warnings,
        "parser_errors": parser_errors,
        "rmsd_threshold": rmsd_threshold,
        "max_diff_threshold": max_diff_threshold,
        "aligned_rmsd_threshold": aligned_rmsd_threshold,
        "distance_threshold": distance_threshold,
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
    parser.add_argument("--aligned-rmsd-threshold", type=float, default=1e-4)
    parser.add_argument("--distance-threshold", type=float, default=1e-4)
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
        aligned_rmsd_threshold=args.aligned_rmsd_threshold,
        distance_threshold=args.distance_threshold,
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
