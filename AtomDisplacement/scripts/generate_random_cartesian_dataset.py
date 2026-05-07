#!/usr/bin/env python3
"""Generate random Cartesian perturbation samples for SIESTA single points."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

from atom_displacement_utils import (
    BASE_DIR,
    DATASET_DIR,
    PIPELINE_CONFIG,
    PIPELINE_PATHS,
    RANDOM_CARTESIAN_STEPS_DIR_NAME,
    RELAXED_DIR,
    Structure,
    distance,
    ensure_dir,
    load_reference_structure,
    structure_with_positions,
    write_json,
)
from pipeline_config_utils import render_single_point_fdf


DEFAULT_RANDOM_CARTESIAN_CONFIG: dict[str, Any] = {
    "enabled": False,
    "n_structures": 100,
    "seed": 1234,
    "distribution": "gaussian",
    "sigma_ang": 0.03,
    "uniform_range_ang": 0.05,
    "move_atoms": "all",
    "species_filter": [],
    "min_distance_ang": 0.65,
    "max_attempts_per_structure": 100,
    "remove_center_of_mass_translation": True,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def random_cartesian_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    root = config or PIPELINE_CONFIG
    raw = (root.get("structure", {}) or {}).get("random_cartesian", {}) or {}
    merged = copy.deepcopy(DEFAULT_RANDOM_CARTESIAN_CONFIG)
    merged.update(raw)
    merged["n_structures"] = int(merged["n_structures"])
    merged["seed"] = int(merged["seed"])
    merged["distribution"] = str(merged["distribution"]).strip().lower()
    merged["sigma_ang"] = float(merged["sigma_ang"])
    merged["uniform_range_ang"] = float(merged["uniform_range_ang"])
    merged["min_distance_ang"] = float(merged["min_distance_ang"])
    merged["max_attempts_per_structure"] = int(merged["max_attempts_per_structure"])
    if merged["distribution"] not in {"gaussian", "uniform"}:
        raise RuntimeError(
            "random_cartesian.distribution debe ser 'gaussian' o 'uniform' "
            f"(recibido: {merged['distribution']!r})."
        )
    if merged["n_structures"] <= 0:
        raise RuntimeError("random_cartesian.n_structures debe ser mayor que cero.")
    if merged["max_attempts_per_structure"] <= 0:
        raise RuntimeError("random_cartesian.max_attempts_per_structure debe ser mayor que cero.")
    if merged["sigma_ang"] < 0 or merged["uniform_range_ang"] < 0:
        raise RuntimeError("Las amplitudes random_cartesian no pueden ser negativas.")
    return merged


def moving_atom_indices(structure: Structure, config: dict[str, Any]) -> list[int]:
    species_filter = [str(item) for item in (config.get("species_filter") or [])]
    allowed_species = set(species_filter)
    move_atoms = config.get("move_atoms", "all")
    if move_atoms in (None, "", "all"):
        indices = list(range(len(structure.atom_species)))
    elif isinstance(move_atoms, list):
        indices = [int(item) - 1 for item in move_atoms]
    else:
        raise RuntimeError("random_cartesian.move_atoms debe ser 'all' o una lista de indices 1-based.")
    if any(index < 0 or index >= len(structure.atom_species) for index in indices):
        raise RuntimeError("random_cartesian.move_atoms contiene indices fuera de rango.")
    if allowed_species:
        indices = [index for index in indices if structure.symbols[index] in allowed_species]
    if not indices:
        raise RuntimeError("random_cartesian no tiene atomos movibles tras aplicar species_filter.")
    return indices


def sample_displacement_vector(rng: random.Random, config: dict[str, Any]) -> list[float]:
    if config["distribution"] == "gaussian":
        sigma = float(config["sigma_ang"])
        return [rng.gauss(0.0, sigma) for _axis in range(3)]
    half_width = float(config["uniform_range_ang"])
    return [rng.uniform(-half_width, half_width) for _axis in range(3)]


def displacement_field(
    structure: Structure,
    config: dict[str, Any],
    rng: random.Random,
) -> list[list[float]]:
    moving = moving_atom_indices(structure, config)
    displacements = [[0.0, 0.0, 0.0] for _atom in structure.atom_species]
    for index in moving:
        displacements[index] = sample_displacement_vector(rng, config)
    if bool(config.get("remove_center_of_mass_translation", True)):
        mean = [
            sum(displacements[index][axis] for index in moving) / len(moving)
            for axis in range(3)
        ]
        for index in moving:
            displacements[index] = [
                displacements[index][axis] - mean[axis]
                for axis in range(3)
            ]
    return displacements


def positions_with_displacements(
    structure: Structure,
    displacements: list[list[float]],
) -> list[list[float]]:
    return [
        [position[axis] + displacements[index][axis] for axis in range(3)]
        for index, position in enumerate(structure.positions_ang)
    ]


def minimum_pair_distance(positions_ang: list[list[float]]) -> float:
    min_distance = math.inf
    for left in range(len(positions_ang)):
        for right in range(left + 1, len(positions_ang)):
            min_distance = min(min_distance, distance(positions_ang[left], positions_ang[right]))
    return min_distance


def validate_random_structure(
    reference: Structure,
    candidate: Structure,
    *,
    min_distance_ang: float,
) -> tuple[bool, str]:
    if len(candidate.atom_species) != len(reference.atom_species):
        return False, "atom_count_changed"
    if candidate.atom_species != reference.atom_species or candidate.symbols != reference.symbols:
        return False, "species_changed"
    if candidate.lattice_vectors_ang != reference.lattice_vectors_ang:
        return False, "cell_changed"
    min_distance = minimum_pair_distance(candidate.positions_ang)
    if min_distance < min_distance_ang:
        return False, f"min_distance {min_distance:.8g} < {min_distance_ang:.8g}"
    return True, "ok"


def deterministic_split_group_id(base_geometry_hash: str, config: dict[str, Any]) -> str:
    amplitude = (
        config["sigma_ang"]
        if config["distribution"] == "gaussian"
        else config["uniform_range_ang"]
    )
    seed_family = int(config["seed"])
    return json_sha256(
        {
            "generation_method": "random_cartesian",
            "base_geometry_hash": base_geometry_hash,
            "distribution": config["distribution"],
            "amplitude_ang": amplitude,
            "seed_family": seed_family,
            "species_filter": config.get("species_filter") or [],
            "move_atoms": config.get("move_atoms", "all"),
        }
    )


def copy_required_resources(dataset_root: Path) -> dict[str, list[str]]:
    basis_sources = sorted(RELAXED_DIR.glob("*.ion.xml"))
    if not basis_sources:
        raise RuntimeError(
            f"random_cartesian requires basis .ion.xml files in {RELAXED_DIR}."
        )
    pseudo_sources = sorted(BASE_DIR.glob("*.psf"))
    if not pseudo_sources:
        raise RuntimeError(
            f"random_cartesian requires pseudopotential .psf files in {BASE_DIR}."
        )
    basis_dir = dataset_root / "basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    for src in basis_sources:
        shutil.copy2(src, basis_dir / src.name)
    return {
        "basis": [str(path) for path in basis_sources],
        "pseudopotentials": [str(path) for path in pseudo_sources],
    }


def write_split_manifests(dataset_root: Path, samples: list[dict[str, Any]]) -> None:
    splits = ("train", "validation", "test")
    for split in splits:
        rows = [sample for index, sample in enumerate(samples) if splits[index % len(splits)] == split]
        write_json(dataset_root / f"split_manifest_{split}.json", {"split": split, "samples": rows})


def artifact_hashes(dataset_root: Path, pseudo_sources: list[str]) -> dict[str, Any]:
    matrix_files = sorted(list(dataset_root.glob("sample_*/*.TSHS")) + list(dataset_root.glob("sample_*/*.HSX")))
    return {
        "run_fdf": {
            path.relative_to(dataset_root).as_posix(): file_sha256(path)
            for path in sorted(dataset_root.glob("sample_*/RUN.fdf"))
        },
        "metadata": {
            path.relative_to(dataset_root).as_posix(): file_sha256(path)
            for path in sorted(dataset_root.glob("sample_*/metadata.json"))
        },
        "basis": {
            path.name: file_sha256(path)
            for path in sorted((dataset_root / "basis").glob("*.ion.xml"))
        },
        "pseudopotentials": {
            Path(path).name: file_sha256(Path(path))
            for path in pseudo_sources
            if Path(path).exists()
        },
        "matrices": {
            path.relative_to(dataset_root).as_posix(): file_sha256(path)
            for path in matrix_files
        },
    }


def generate_dataset(config: dict[str, Any] | None = None) -> dict[str, Any]:
    rc_config = random_cartesian_config(config)
    dataset_root = DATASET_DIR / RANDOM_CARTESIAN_STEPS_DIR_NAME
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    ensure_dir(dataset_root)
    resources = copy_required_resources(dataset_root)
    reference, source_path = load_reference_structure()
    base_geometry_hash = json_sha256(reference.to_json_dict())
    split_group_id = deterministic_split_group_id(base_geometry_hash, rc_config)
    rng = random.Random(int(rc_config["seed"]))
    samples: list[dict[str, Any]] = []

    print("=== Random Cartesian dataset generation ===")
    print(f"[INFO] Geometria base: {source_path}")
    print(f"[INFO] Output root: {dataset_root}")
    print(f"[INFO] n_structures: {rc_config['n_structures']}")

    for sample_index in range(int(rc_config["n_structures"])):
        accepted: tuple[int, list[list[float]], Structure] | None = None
        last_reason = "not_attempted"
        for attempt in range(1, int(rc_config["max_attempts_per_structure"]) + 1):
            displacements = displacement_field(reference, rc_config, rng)
            positions = positions_with_displacements(reference, displacements)
            candidate = structure_with_positions(reference, positions)
            ok, reason = validate_random_structure(
                reference,
                candidate,
                min_distance_ang=float(rc_config["min_distance_ang"]),
            )
            last_reason = reason
            if ok:
                accepted = (attempt, displacements, candidate)
                break
        if accepted is None:
            raise RuntimeError(
                "random_cartesian could not generate a valid structure "
                f"for sample_index={sample_index} after "
                f"{rc_config['max_attempts_per_structure']} attempts: {last_reason}."
            )
        accepted_attempt, displacements, candidate = accepted
        sample_id = f"sample_{sample_index + 1:06d}"
        sample_dir = dataset_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        for pseudo in resources["pseudopotentials"]:
            shutil.copy2(pseudo, sample_dir / Path(pseudo).name)
        single_point_config = copy.deepcopy(PIPELINE_CONFIG)
        single_point_config.setdefault("structure", {}).setdefault("force_constants", {})["enabled"] = False
        content = render_single_point_fdf(
            single_point_config,
            positions_ang=candidate.positions_ang,
            atom_species=candidate.atom_species,
            sample_id=sample_id,
        )
        (sample_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]).write_text(content, encoding="utf-8")
        metadata = {
            "id": sample_id,
            "generation_method": "random_cartesian",
            "base_geometry_hash": base_geometry_hash,
            "base_geometry_source": source_path,
            "seed": int(rc_config["seed"]),
            "sample_index": sample_index,
            "distribution": rc_config["distribution"],
            "sigma_ang": float(rc_config["sigma_ang"]) if rc_config["distribution"] == "gaussian" else None,
            "uniform_range_ang": float(rc_config["uniform_range_ang"]) if rc_config["distribution"] == "uniform" else None,
            "displacements_ang": displacements,
            "min_distance_ang": float(rc_config["min_distance_ang"]),
            "accepted_attempt": accepted_attempt,
            "split_group_id": split_group_id,
        }
        write_json(sample_dir / "metadata.json", metadata)
        samples.append(
            {
                "sample_id": sample_id,
                "sample_dir": str(sample_dir),
                "run_fdf": str(sample_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]),
                "metadata": str(sample_dir / "metadata.json"),
                "split_group_id": split_group_id,
                "accepted_attempt": accepted_attempt,
            }
        )

    manifest = {
        "method_id": "random_cartesian",
        "generation_method": "random_cartesian",
        "dataset_root": str(dataset_root),
        "requested_structures": int(rc_config["n_structures"]),
        "generated_structures": len(samples),
        "base_geometry_hash": base_geometry_hash,
        "base_geometry_source": source_path,
        "config_snapshot": rc_config,
        "samples": samples,
        "siesta_input_hashes": {
            sample["sample_id"]: file_sha256(Path(sample["run_fdf"]))
            for sample in samples
        },
        "basis_hashes": {
            path.name: file_sha256(path)
            for path in sorted((dataset_root / "basis").glob("*.ion.xml"))
        },
        "pseudo_hashes": {
            Path(path).name: file_sha256(Path(path))
            for path in resources["pseudopotentials"]
        },
        "matrix_file_hashes": {},
        "warnings": [],
        "severe_warnings": [],
    }
    write_json(dataset_root / "dataset_manifest.json", manifest)
    write_json(PIPELINE_PATHS["samples_manifest_path"], manifest)
    write_json(dataset_root / "artifact_hashes.json", artifact_hashes(dataset_root, resources["pseudopotentials"]))
    write_split_manifests(dataset_root, samples)
    print(f"[OK] Random Cartesian dataset generado en {dataset_root}")
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-structures", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--distribution", choices=["gaussian", "uniform"], default=None)
    parser.add_argument("--sigma-ang", type=float, default=None)
    parser.add_argument("--uniform-range-ang", type=float, default=None)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    config = copy.deepcopy(PIPELINE_CONFIG)
    random_config = config.setdefault("structure", {}).setdefault("random_cartesian", {})
    if args.n_structures is not None:
        random_config["n_structures"] = args.n_structures
    if args.seed is not None:
        random_config["seed"] = args.seed
    if args.distribution is not None:
        random_config["distribution"] = args.distribution
    if args.sigma_ang is not None:
        random_config["sigma_ang"] = args.sigma_ang
    if args.uniform_range_ang is not None:
        random_config["uniform_range_ang"] = args.uniform_range_ang
    generate_dataset(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
