#!/usr/bin/env python3
"""Compare local interlayer pair environments in training FDFs and a moiré."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from fdf_materialization import extract_fdf_structure  # noqa: E402


PAIR_TYPES = (
    ("top_graphene", "bottom_graphene"),
    ("bottom_graphene", "B"),
    ("bottom_graphene", "N"),
    ("top_graphene", "B"),
    ("top_graphene", "N"),
)


def physical_layers(path: Path) -> tuple[np.ndarray, dict[str, list[int]]]:
    structure = extract_fdf_structure(path, structure_type="crystal")
    positions = np.asarray(structure.positions_ang)
    labels = {species.index: species.label for species in structure.species}
    carbon = [index for index, species in enumerate(structure.atom_species) if labels[species] == "C"]
    z_levels = sorted({round(float(positions[index, 2]), 5) for index in carbon})
    if len(z_levels) != 2:
        raise RuntimeError(f"{path}: expected two graphene z layers, found {z_levels}")
    split = 0.5 * sum(z_levels)
    layers = {
        "bottom_graphene": [index for index in carbon if positions[index, 2] < split],
        "top_graphene": [index for index in carbon if positions[index, 2] > split],
        "B": [index for index, species in enumerate(structure.atom_species) if labels[species] == "B"],
        "N": [index for index, species in enumerate(structure.atom_species) if labels[species] == "N"],
    }
    return np.asarray(structure.lattice_vectors_ang), layers | {"positions": positions.tolist()}


def descriptors(
    path: Path,
    *,
    cutoff_ang: float,
    distance_bin_ang: float,
    angular_bins: int,
) -> Counter[str]:
    lattice, payload = physical_layers(path)
    positions = np.asarray(payload.pop("positions"))
    bound = math.ceil(cutoff_ang / min(np.linalg.norm(lattice[0]), np.linalg.norm(lattice[1]))) + 1
    shifts = [
        (i, j, i * lattice[0] + j * lattice[1])
        for i in range(-bound, bound + 1)
        for j in range(-bound, bound + 1)
    ]
    counts: Counter[str] = Counter()
    for left_name, right_name in PAIR_TYPES:
        for left in payload[left_name]:
            for right in payload[right_name]:
                for _i, _j, shift in shifts:
                    vector = positions[right] + shift - positions[left]
                    distance = float(np.linalg.norm(vector))
                    if distance < 1e-8 or distance > cutoff_ang:
                        continue
                    angle = math.atan2(float(vector[1]), float(vector[0])) % (2 * math.pi)
                    d_bin = round(distance / distance_bin_ang)
                    a_bin = int(angle / (2 * math.pi) * angular_bins) % angular_bins
                    counts[f"{left_name}:{right_name}:d{d_bin}:a{a_bin}"] += 1
    return counts


def analyze(
    training_fdfs: list[Path],
    target_fdf: Path,
    *,
    cutoff_ang: float = 7.0,
    distance_bin_ang: float = 0.2,
    angular_bins: int = 12,
) -> dict:
    per_source = {
        path.parent.name: descriptors(
            path,
            cutoff_ang=cutoff_ang,
            distance_bin_ang=distance_bin_ang,
            angular_bins=angular_bins,
        )
        for path in training_fdfs
    }
    training_support = set().union(*(set(counts) for counts in per_source.values()))
    target = descriptors(
        target_fdf,
        cutoff_ang=cutoff_ang,
        distance_bin_ang=distance_bin_ang,
        angular_bins=angular_bins,
    )
    supported = {key: count for key, count in target.items() if key in training_support}
    unsupported = {key: count for key, count in target.items() if key not in training_support}
    total = sum(target.values())
    supported_count = sum(supported.values())
    per_pair = {}
    for left, right in PAIR_TYPES:
        prefix = f"{left}:{right}:"
        pair_total = sum(count for key, count in target.items() if key.startswith(prefix))
        pair_supported = sum(count for key, count in supported.items() if key.startswith(prefix))
        per_pair[f"{left}:{right}"] = {
            "target_occurrences": pair_total,
            "supported_occurrences": pair_supported,
            "weighted_coverage": pair_supported / pair_total if pair_total else None,
        }
    registry_variants_added = any(
        token in path.parent.name.lower() for path in training_fdfs for token in ("topaa", "topba")
    )
    if unsupported and registry_variants_added:
        interpretation = (
            "Rigid top-graphene AA/AB/BA registries are present. Residual unsupported exact "
            "distance/orientation bins arise from the continuous twisted target and remain "
            "explicitly marked as model extrapolation."
        )
    elif unsupported:
        interpretation = (
            "Unsupported bins are explicit target-level extrapolation. Add rigid top-graphene "
            "AA/BA registry variants before claiming interpolation-level reliability."
        )
    else:
        interpretation = "All discretized target pair environments have training support."
    return {
        "status": "covered" if not unsupported else "extrapolation_detected",
        "target": str(target_fdf),
        "training_sources": [str(path) for path in training_fdfs],
        "rigid_top_registry_variants_added": registry_variants_added,
        "cutoff_ang": cutoff_ang,
        "distance_bin_ang": distance_bin_ang,
        "angular_bins": angular_bins,
        "target_unique_bins": len(target),
        "supported_unique_bins": len(supported),
        "weighted_coverage": supported_count / total if total else None,
        "per_pair": per_pair,
        "unsupported_bins": [
            {"descriptor": key, "target_occurrences": value}
            for key, value in sorted(unsupported.items(), key=lambda item: (-item[1], item[0]))
        ],
        "scientific_interpretation": interpretation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-fdf", type=Path, action="append", required=True)
    parser.add_argument("--target-fdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff-ang", type=float, default=7.0)
    args = parser.parse_args()
    result = analyze(
        [path.resolve() for path in args.training_fdf],
        args.target_fdf.resolve(),
        cutoff_ang=args.cutoff_ang,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
