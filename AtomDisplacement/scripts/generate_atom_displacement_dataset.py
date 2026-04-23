#!/usr/bin/env python3
"""Generate single-point H2O samples around a relaxed reference geometry."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from atom_displacement_utils import (
    BASE_DIR,
    DATASET_DIR,
    SAMPLES_DIR,
    Structure,
    compute_water_geometry_metrics,
    copy_pseudopotentials,
    ensure_dir,
    load_reference_structure,
    structure_with_positions,
    write_json,
    write_single_point_fdf,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera muestras H2O deformadas con desplazamientos atomicos pequenos."
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.02, help="Desviacion tipica en Ang")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-displacement-norm", type=float, default=0.05)
    parser.add_argument("--min-oh", type=float, default=0.7)
    parser.add_argument("--max-oh", type=float, default=1.2)
    parser.add_argument("--min-hh", type=float, default=1.0)
    parser.add_argument("--min-angle", type=float, default=80.0)
    parser.add_argument("--max-angle", type=float, default=130.0)
    return parser


def displaced_structure(
    reference: Structure,
    rng: random.Random,
    sigma: float,
) -> tuple[Structure, list[list[float]], list[float]]:
    displacements: list[list[float]] = []
    norms: list[float] = []
    positions: list[list[float]] = []
    for position in reference.positions_ang:
        displacement = [rng.gauss(0.0, sigma) for _ in range(3)]
        norm = math.sqrt(sum(value * value for value in displacement))
        displacements.append(displacement)
        norms.append(norm)
        positions.append([coordinate + delta for coordinate, delta in zip(position, displacement)])

    return structure_with_positions(reference, positions), displacements, norms


def is_valid_structure(
    structure: Structure,
    displacement_norms: list[float],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, float]]:
    metrics = compute_water_geometry_metrics(structure)
    if any(norm > args.max_displacement_norm for norm in displacement_norms):
        return False, metrics
    if not (args.min_oh <= metrics["oh_1_ang"] <= args.max_oh):
        return False, metrics
    if not (args.min_oh <= metrics["oh_2_ang"] <= args.max_oh):
        return False, metrics
    if metrics["hh_ang"] < args.min_hh:
        return False, metrics
    if not (args.min_angle <= metrics["hoh_angle_deg"] <= args.max_angle):
        return False, metrics
    return True, metrics


def main() -> int:
    args = build_argument_parser().parse_args()
    ensure_dir(DATASET_DIR)
    ensure_dir(SAMPLES_DIR)

    reference, source_path = load_reference_structure()
    rng = random.Random(args.seed)
    accepted_samples = []
    rejected = 0
    max_attempts = max(args.num_samples * 100, 1000)

    print("=== AtomDisplacement dataset generation ===")
    print(f"[INFO] Geometria de referencia: {source_path}")
    print(f"[INFO] Numero de muestras objetivo: {args.num_samples}")
    print(f"[INFO] Sigma de desplazamiento: {args.sigma:.4f} Ang")

    sample_index = 1
    attempts = 0
    while sample_index <= args.num_samples:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"No fue posible generar {args.num_samples} muestras validas tras {attempts} intentos."
            )

        candidate, displacements, norms = displaced_structure(reference, rng, args.sigma)
        is_valid, metrics = is_valid_structure(candidate, norms, args)
        if not is_valid:
            rejected += 1
            continue

        sample_id = f"sample_{sample_index:04d}"
        sample_dir = SAMPLES_DIR / sample_id
        ensure_dir(sample_dir)
        copy_pseudopotentials(BASE_DIR, sample_dir)
        write_single_point_fdf(sample_dir / "RUN.fdf", candidate, sample_id)

        metadata = {
            "id": sample_id,
            "reference_source": source_path,
            "positions_ang": candidate.positions_ang,
            "displacements_ang": displacements,
            "displacement_norms_ang": norms,
            "geometry_metrics": metrics,
        }
        write_json(sample_dir / "metadata.json", metadata)
        accepted_samples.append(metadata)
        sample_index += 1

    manifest = {
        "reference_source": source_path,
        "num_requested": args.num_samples,
        "num_generated": len(accepted_samples),
        "num_rejected": rejected,
        "sigma_ang": args.sigma,
        "seed": args.seed,
        "filters": {
            "max_displacement_norm_ang": args.max_displacement_norm,
            "min_oh_ang": args.min_oh,
            "max_oh_ang": args.max_oh,
            "min_hh_ang": args.min_hh,
            "min_angle_deg": args.min_angle,
            "max_angle_deg": args.max_angle,
        },
        "samples": accepted_samples,
    }
    write_json(DATASET_DIR / "samples_manifest.json", manifest)
    print(f"[OK] Muestras generadas en {SAMPLES_DIR}")
    print(f"[OK] Rechazadas antes de calcular: {rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
