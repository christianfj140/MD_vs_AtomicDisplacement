#!/usr/bin/env python3
"""Compute layer-resolved primitive-graphene spectral weights for one moiré model."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from run_deeph_sparse_spectrum import (
    MIN_FREE_DISK_PERCENT,
    PARDISO_OOC_MAX_CORE_MB,
    REPO_ROOT,
    SOLVER_THREADS,
    band_kpoints,
    file_sha256,
    free_disk_percent,
    run_with_disk_guard,
    sort_bands_by_energy,
    write_json,
)
from run_graphene_hbn_moire_spectral_campaign import (
    DEFAULT_CONFIG,
    DEFAULT_ROOT,
    aggregate,
    load_config,
    read_json,
)
from validate_deeph_sparse_solver import parse_openmx_band


PRIMITIVE_PATH = [
    ("Γ", (0.0, 0.0, 0.0)),
    ("K", (1.0 / 3.0, 2.0 / 3.0, 0.0)),
    ("M", (0.5, 0.5, 0.0)),
    ("Γ", (0.0, 0.0, 0.0)),
]


def primitive_samples(points_per_segment: int) -> list[tuple[str, np.ndarray]]:
    # Reuse the repository interpolation contract while temporarily supplying its path.
    import run_deeph_sparse_spectrum as sparse

    original = sparse.K_PATH
    try:
        sparse.K_PATH = PRIMITIVE_PATH
        return band_kpoints(points_per_segment)
    finally:
        sparse.K_PATH = original


def orbital_map(input_dir: Path, layer: str, geometry: dict, destination: Path) -> None:
    positions = np.loadtxt(input_dir / "site_positions.dat").T
    shells = [
        [int(value) for value in line.split()]
        for line in (input_dir / "orbital_types.dat").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    offsets = np.cumsum([0, *[sum(2 * ell + 1 for ell in row) for row in shells]])
    z_levels = np.unique(np.round(positions[:, 2], 6))
    if len(z_levels) != 3:
        raise RuntimeError(f"Expected hBN plus two graphene z levels, found {z_levels.tolist()}")
    layer_index = {"bottom": 1, "top": 2}[layer]
    atom_indices = np.flatnonzero(np.isclose(positions[:, 2], z_levels[layer_index], atol=1e-5))
    expected = 2 * int(geometry["commensurate_cell_index"])
    if len(atom_indices) != expected:
        raise RuntimeError(f"{layer}: expected {expected} carbon atoms, found {len(atom_indices)}")

    a = float(geometry["geometry_inplane_lattice_ang"])
    primitive = np.asarray([[a, 0.0], [-a / 2.0, math.sqrt(3.0) * a / 2.0]])
    if layer == "top":
        theta = math.radians(float(geometry["materialized_twist_angle_deg"]))
        primitive = primitive @ np.asarray(
            [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
        )
        basis = np.asarray([[1 / 3, 2 / 3], [2 / 3, 1 / 3]])
    else:
        basis = np.asarray([[0.0, 0.0], [1 / 3, 2 / 3]])

    rows = []
    for atom in atom_indices:
        fractional = positions[atom, :2] @ np.linalg.inv(primitive)
        residuals = fractional[None, :] - basis
        translations = np.rint(residuals)
        errors = np.linalg.norm(residuals - translations, axis=1)
        sublattice = int(np.argmin(errors))
        if errors[sublattice] > 2e-5:
            raise RuntimeError(f"{layer}: atom {atom} does not map to a primitive graphene site")
        r1, r2 = translations[sublattice].astype(int)
        for local_orbital, orbital in enumerate(range(offsets[atom], offsets[atom + 1])):
            rows.append((orbital + 1, sublattice * 4 + local_orbital + 1, r1, r2))
    np.savetxt(destination, np.asarray(rows, dtype=int), fmt="%d")


def folded_kpoints(samples: list[tuple[str, np.ndarray]], matrix: np.ndarray) -> np.ndarray:
    primitive = np.asarray([point for _label, point in samples])
    folded = primitive[:, :2] @ matrix.T
    return np.column_stack((folded % 1.0, np.zeros(len(folded))))


def run_layer(
    *,
    root: Path,
    input_dir: Path,
    output_dir: Path,
    layer: str,
    num_bands: int,
    points_per_segment: int,
) -> dict:
    geometry = read_json(root / "target/moire_geometry.json")
    neutrality = read_json(root / "neutrality_estimate.json")
    environment = read_json(root / "solver/environment.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "unfolding_map.dat"
    primitive_path = output_dir / "primitive_kpoints.dat"
    orbital_map(input_dir, layer, geometry, mapping_path)
    samples = primitive_samples(points_per_segment)
    matrix = np.asarray(
        geometry["layer1_supercell_matrix" if layer == "bottom" else "layer2_supercell_matrix"],
        dtype=int,
    )
    folded = folded_kpoints(samples, matrix)
    np.savetxt(primitive_path, np.asarray([point for _label, point in samples]), fmt="%.18e")
    config = {
        "calc_job": "band",
        "fermi_level": float(neutrality["energy_eV"]),
        "max_iter": 1000,
        "num_band": num_bands,
        "pardiso_out_of_core": True,
        "solver_backend": "cpu_mkl_pardiso",
        "which_k": 0,
        "k_data": [
            f"1 {' '.join(map(str, point))} {' '.join(map(str, point))}" for point in folded
        ],
    }
    config_path = output_dir / "band_config.json"
    write_json(config_path, config)
    command = [
        "/usr/bin/time", "-v", "-o", str(output_dir / "time-v.txt"),
        environment["julia"], "--startup-file=no", f"--project={environment['project']}",
        str(REPO_ROOT / "Comparison/scripts/deeph_sparse_calc_unfolded.jl"),
        "--input_dir", str(input_dir), "--output_dir", str(output_dir), "--config", str(config_path),
    ]
    env = os.environ.copy()
    env.update(
        {
            "JULIA_DEPOT_PATH": environment["depot"],
            "DEEPH_SPARSE_CALC": str(REPO_ROOT.parent / "DeepH-pack/deeph/inference/sparse_calc.jl"),
            "DEEPH_UNFOLD_MAP": str(mapping_path),
            "DEEPH_UNFOLD_KPOINTS": str(primitive_path),
            "OMP_NUM_THREADS": str(SOLVER_THREADS),
            "MKL_NUM_THREADS": str(SOLVER_THREADS),
            "MKL_PARDISO_OOC_MAX_CORE_SIZE": str(PARDISO_OOC_MAX_CORE_MB),
            "MKL_PARDISO_OOC_MAX_SWAP_SIZE": "0",
            "MKL_PARDISO_OOC_KEEP_FILE": "0",
        }
    )
    returncode, reason, minimum_disk, maximum_temp, minimum_memory = run_with_disk_guard(
        command, cwd=output_dir, env=env, output_dir=output_dir
    )
    result = {
        "status": "completed" if returncode == 0 else "resource_blocked" if reason else "failed",
        "returncode": returncode,
        "reason": reason,
        "layer": layer,
        "method": "primitive_graphene_LCAO_coefficient_unfolding",
        "scientific_status": "diagnostic_nonorthogonal_basis_approximation",
        "limitation": (
            "Layer-separated coefficient Fourier weights; not a SIESTA real-space "
            "wavefunction unfolding and not a target-reference accuracy claim."
        ),
        "minimum_observed_free_disk_percent": minimum_disk,
        "maximum_observed_cpu_temperature_c": maximum_temp,
        "minimum_observed_available_memory_bytes": minimum_memory,
        "minimum_free_disk_percent": MIN_FREE_DISK_PERCENT,
        "hamiltonian_sha256": file_sha256(input_dir / "hamiltonians_pred.h5"),
        "overlap_sha256": file_sha256(input_dir / "overlaps.h5"),
    }
    if returncode == 0:
        raw = parse_openmx_band(output_dir / "openmx.Band")
        order = np.argsort(raw, axis=1)
        distances = [0.0]
        a = float(geometry["geometry_inplane_lattice_ang"])
        reciprocal = 2 * np.pi * np.linalg.inv(
            np.asarray([[a, 0.0], [-a / 2.0, math.sqrt(3.0) * a / 2.0]])
        ).T
        primitive = [point for _label, point in samples]
        for left, right in zip(primitive, primitive[1:]):
            distances.append(distances[-1] + float(np.linalg.norm((right - left) @ reciprocal)))
        points = []
        for k_index, ((label, _point), distance) in enumerate(zip(samples, distances, strict=True)):
            payload = read_json(output_dir / f"unfolding_{k_index:03d}.json")
            weights = np.asarray(payload["spectral_weights"])[order[k_index]]
            energies = sort_bands_by_energy(raw[k_index : k_index + 1])[0]
            for band_index, (energy, weight) in enumerate(zip(energies, weights, strict=True)):
                points.append(
                    {
                        "k_index": k_index,
                        "band_index": band_index,
                        "k_distance": distance,
                        "k_label": label,
                        "energy_aligned_eV": float(energy - neutrality["energy_eV"]),
                        "spectral_weight": float(weight),
                    }
                )
        result["unfolded_bands"] = points
        result["k_path"] = {
            "name": f"{layer}_graphene_primitive_Gamma-K-M-Gamma",
            "points_per_segment": points_per_segment,
            "sample_count": len(samples),
        }
    write_json(output_dir / "solver_manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--training-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-bands", type=int, default=16)
    parser.add_argument("--points-per-segment", type=int, default=8)
    args = parser.parse_args()
    input_dir = args.root / f"predictions/graph2mat/n{args.training_size}/seed{args.seed}/solver_input"
    combined_path = args.root / f"spectra/graph2mat/n{args.training_size}/seed{args.seed}/solver_manifest.json"
    combined = read_json(combined_path)
    results = {}
    for layer in ("bottom", "top"):
        output = args.root / (
            f"spectra/graph2mat/n{args.training_size}/seed{args.seed}/unfolded_{layer}"
        )
        existing = read_json(output / "solver_manifest.json")
        result = existing if existing.get("status") == "completed" else run_layer(
            root=args.root,
            input_dir=input_dir,
            output_dir=output,
            layer=layer,
            num_bands=args.num_bands,
            points_per_segment=args.points_per_segment,
        )
        results[layer] = result
        if result.get("status") != "completed":
            break
    combined["unfolding"] = {
        "status": "completed" if len(results) == 2 and all(
            result.get("status") == "completed" for result in results.values()
        ) else "incomplete",
        "layers": results,
    }
    write_json(combined_path, combined)
    aggregate(load_config(args.config), args.root)
    return 0 if combined["unfolding"]["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
