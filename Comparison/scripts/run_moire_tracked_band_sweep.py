#!/usr/bin/env python3
"""Run overlap-tracked Graph2Mat moiré bands from N=480 downwards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_deeph_sparse_spectrum import (
    DEFAULT_ENVIRONMENT,
    DEFAULT_GPU_ENVIRONMENT,
    run as run_sparse_spectrum,
    track_bands_by_overlap,
)
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
    raw = parse_openmx_band(output / "openmx.Band")
    energies = np.sort(raw, axis=1)
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
        tracked, _metadata = track_bands_by_overlap(output, raw)
    else:
        tracked = None
    def representation(values: np.ndarray) -> list[dict]:
        rows = []
        for point in result["bands"]:
            row = dict(point)
            energy = float(values[int(point["k_index"]), int(point["band_index"])])
            row["energy_eV"] = energy
            row["energy_aligned_eV"] = energy - float(result["fermi_level_eV"])
            rows.append(row)
        return rows
    result["band_representations"] = {
        "raw_arpack_order": {
            "scientific_status": "solver_order_not_a_band_identity",
            "bands": representation(raw),
        },
        "energy_sorted": {
            "scientific_status": "visible_energy_rank_per_k",
            "bands_field": "bands",
        },
    }
    if tracked is not None:
        result["band_representations"]["overlap_tracked"] = {
            "scientific_status": "diagnostic_low_confidence_nonorthogonal_coefficient_overlap",
            "bands": representation(tracked),
        }
    if output.name == "tier_b_tracked":
        result["k_path"] = {
            "name": "legacy_Gamma-interior-M-Gamma",
            "scientific_status": "legacy_path_requires_recalculation",
            "note": "q* was previously mislabeled K",
        }
    write_json(output / "solver_manifest.json", result)
    return result


def run_sweep(
    config_path: Path,
    root: Path,
    sizes: list[int],
    corrected_path: bool = False,
    backend: str = "cpu_mkl_pardiso",
    num_bands: int = 256,
    points_per_segment: int = 11,
    band_path: str = "gamma-k-m-gamma",
    project_mulliken: bool = True,
) -> dict:
    config = load_config(config_path)
    neutrality = read_json(root / "neutrality_estimate.json")
    status_path = root / "spectra/tracked_band_sweep_status.json"
    completed = []
    for size in sizes:
        low_energy_moire = band_path == "k-gamma-m"
        corrected = corrected_path or size == 30 or low_energy_moire or project_mulliken
        projected_tier = (
            "tier_projected_kprime_diagnostic"
            if band_path == "kprime-gamma-k"
            else "tier_projected_production"
            if points_per_segment >= 51
            else "tier_projected_smoke"
        )
        output = root / (
            f"spectra/graph2mat/n{size}/seed0/"
            f"{projected_tier if project_mulliken else 'tier_b_magic_angle_low_energy' if low_energy_moire else 'tier_b_sorted_correct_path' if corrected else 'tier_b_tracked'}"
        )
        existing = read_json(output / "solver_manifest.json")
        if project_mulliken and existing.get("status") == "completed" and existing.get("projection", {}).get("status") == "completed":
            result = existing
        elif existing.get("status") == "completed" and existing.get("band_tracking", {}).get("status") == "completed":
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
                num_bands=num_bands,
                points_per_segment=points_per_segment if corrected else 8,
                kmesh=(3, 3, 1),
                environment_path=(
                    DEFAULT_GPU_ENVIRONMENT if backend == "gpu_cudss" else DEFAULT_ENVIRONMENT
                ),
                backend=backend,
                track_bands=not corrected,
                band_path=band_path,
                project_mulliken=project_mulliken,
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
                "band_representations": result.get("band_representations"),
                "k_path": result.get("k_path"),
                "projection": result.get("projection"),
                "projection_mapping": result.get("projection_mapping"),
                "neutrality_reference": result.get("neutrality_reference"),
                "num_bands": result.get("num_bands"),
                "resources": result.get("resources"),
                "resource_gate": result.get("resource_gate"),
                "backend_requested": result.get("backend_requested"),
                "backend_effective": result.get("backend_effective"),
                "gpu_hardware": result.get("gpu_hardware"),
                "gpu_observations": result.get("gpu_observations"),
                "cudss_timings": result.get("cudss_timings"),
                "kprime_comparison": result.get("kprime_comparison"),
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
    parser.add_argument("--corrected-path", action="store_true")
    parser.add_argument(
        "--backend", choices=("cpu_mkl_pardiso", "gpu_cudss"), default="cpu_mkl_pardiso"
    )
    parser.add_argument("--num-bands", type=int, default=256)
    parser.add_argument("--points-per-segment", type=int, default=11)
    parser.add_argument(
        "--band-path", choices=("gamma-k-m-gamma", "k-gamma-m", "kprime-gamma-k"), default="gamma-k-m-gamma"
    )
    parser.add_argument("--legacy-no-projections", action="store_true")
    args = parser.parse_args()
    result = run_sweep(
        args.config.resolve(), args.root.resolve(), args.sizes, args.corrected_path, args.backend,
        args.num_bands, args.points_per_segment, args.band_path, not args.legacy_no_projections,
    )
    print(result)
    return 0 if result["status"] == "completed" else 2 if result["status"] == "resource_blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
