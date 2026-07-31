#!/usr/bin/env python3
"""Run overlap-tracked Graph2Mat moiré bands from N=480 downwards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_deeph_sparse_spectrum import run as run_sparse_spectrum
from validate_deeph_sparse_solver import parse_openmx_band
from run_graphene_hbn_moire_spectral_campaign import (
    DEFAULT_CONFIG,
    DEFAULT_ROOT,
    aggregate,
    load_config,
    now,
    read_json,
    write_json,
)


def sort_existing_result(output: Path, result: dict) -> dict:
    energies = np.sort(parse_openmx_band(output / "openmx.Band"), axis=1)
    legacy_k_index = (energies.shape[0] - 1) // 3
    for point in result["bands"]:
        energy = float(energies[int(point["k_index"]), int(point["band_index"])])
        point["energy_eV"] = energy
        point["energy_aligned_eV"] = energy - float(result["fermi_level_eV"])
        if output.name == "tier_b_tracked" and int(point["k_index"]) == legacy_k_index:
            point["k_label"] = "q*"
    result["band_ordering"] = {
        "method": "ascending_energy_rank_per_k",
        "band_character_continuity_claimed": False,
        "raw_solver_order_preserved_in": str(output / "openmx.Band"),
    }
    if result.get("band_tracking"):
        result["band_tracking"]["used_for_plot"] = False
    if output.name == "tier_b_tracked":
        result["k_path"] = {
            "name": "legacy_Gamma-interior-M-Gamma",
            "scientific_status": "legacy_path_requires_recalculation",
            "note": "q* was previously mislabeled K",
        }
    write_json(output / "solver_manifest.json", result)
    return result


def run_sweep(config_path: Path, root: Path, sizes: list[int]) -> dict:
    config = load_config(config_path)
    neutrality = read_json(root / "neutrality_estimate.json")
    status_path = root / "spectra/tracked_band_sweep_status.json"
    completed = []
    for size in sizes:
        corrected = size == 30
        output = root / (
            f"spectra/graph2mat/n{size}/seed0/"
            f"{'tier_b_sorted_correct_path' if corrected else 'tier_b_tracked'}"
        )
        existing = read_json(output / "solver_manifest.json")
        if existing.get("status") == "completed" and existing.get("band_tracking", {}).get("status") == "completed":
            result = sort_existing_result(output, existing)
        elif corrected and existing.get("status") == "completed":
            result = sort_existing_result(output, existing)
        else:
            write_json(
                status_path,
                {
                    "status": "running",
                    "current_training_size": size,
                    "completed_training_sizes": completed,
                    "updated_at": now(),
                },
            )
            result = run_sparse_spectrum(
                root / f"predictions/graph2mat/n{size}/seed0/solver_input",
                output,
                job="band",
                fermi_level=float(neutrality["energy_eV"]),
                num_bands=16,
                points_per_segment=16 if corrected else 8,
                kmesh=(3, 3, 1),
                environment_path=root / "solver/environment.json",
                backend="cpu_mkl_pardiso",
                track_bands=not corrected,
            )
        if result.get("status") != "completed":
            final = {
                "status": result.get("status", "failed"),
                "current_training_size": size,
                "completed_training_sizes": completed,
                "reason": result.get("reason"),
                "updated_at": now(),
            }
            write_json(status_path, final)
            return final
        combined_path = root / f"spectra/graph2mat/n{size}/seed0/solver_manifest.json"
        combined = read_json(combined_path)
        combined.update(
            {
                "status": "completed",
                "scientific_status": "prediction_only",
                "model": "graph2mat",
                "training_size": size,
                "seed": 0,
                "bands_status": "completed",
                "bands": result["bands"],
                "band_tracking": result.get("band_tracking"),
                "band_ordering": result.get("band_ordering"),
                "k_path": result.get("k_path"),
                "visible_band_tier": output.name,
                "tracked_solver_manifest": str(output / "solver_manifest.json"),
                "dense_large_cell_fallback_used": False,
                "identity_overlap_used": False,
            }
        )
        write_json(combined_path, combined)
        completed.append(size)
        aggregate(config, root)
    final = {
        "status": "completed",
        "current_training_size": None,
        "completed_training_sizes": completed,
        "updated_at": now(),
    }
    write_json(status_path, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sizes", type=int, nargs="+", default=[480, 240, 120, 60, 30])
    args = parser.parse_args()
    result = run_sweep(args.config.resolve(), args.root.resolve(), args.sizes)
    print(result)
    return 0 if result["status"] == "completed" else 2 if result["status"] == "resource_blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
