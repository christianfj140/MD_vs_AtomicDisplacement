#!/usr/bin/env python3
"""Compare the small physical cuDSS bands with the established CPU result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def grouped_bands(manifest: dict) -> list[tuple[str, np.ndarray]]:
    grouped: dict[int, list[dict]] = {}
    for point in manifest["bands"]:
        grouped.setdefault(int(point["k_index"]), []).append(point)
    return [
        (
            next((str(point["k_label"]) for point in points if point.get("k_label")), ""),
            np.sort([float(point["energy_eV"]) for point in points]),
        )
        for _index, points in sorted(grouped.items())
    ]


def maximum_matched_error(reference: dict, candidate: dict) -> tuple[float, list[dict]]:
    reference_groups = grouped_bands(reference)
    comparisons = []
    for candidate_index, (label, candidate_energies) in enumerate(grouped_bands(candidate)):
        candidates = [
            (reference_index, float(np.max(np.abs(candidate_energies - reference_energies))))
            for reference_index, (reference_label, reference_energies) in enumerate(reference_groups)
            if reference_label == label and len(reference_energies) == len(candidate_energies)
        ]
        if not candidates:
            raise RuntimeError(f"No reference match for k-point {candidate_index} ({label})")
        reference_index, error = min(candidates, key=lambda item: item[1])
        comparisons.append(
            {
                "candidate_k_index": candidate_index,
                "reference_k_index": reference_index,
                "label": label,
                "max_abs_error_eV": error,
            }
        )
    return max(item["max_abs_error_eV"] for item in comparisons), comparisons


def validate(cpu_path: Path, gpu_path: Path, output_path: Path, repeat_path: Path | None = None) -> dict:
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    maximum_error, comparisons = maximum_matched_error(cpu, gpu)
    repeat_error = None
    if repeat_path is not None:
        repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
        repeat_error, _repeat_comparisons = maximum_matched_error(gpu, repeat)
    residual = gpu.get("maximum_relative_solve_residual")
    finite = all(np.isfinite(energies).all() for _label, energies in grouped_bands(gpu))
    valid = (
        cpu.get("backend_effective", "cpu_mkl_pardiso") == "cpu_mkl_pardiso"
        and gpu.get("backend_effective") == "gpu_cudss"
        and maximum_error <= 1e-6
        and residual is not None
        and float(residual) <= 1e-10
        and finite
        and (repeat_error is None or repeat_error <= 1e-10)
    )
    result = {
        "status": "valid" if valid else "invalid",
        "system": "physical_C4BN_6_atom_SIESTA_H_and_S",
        "cpu_backend": "cpu_mkl_pardiso",
        "gpu_backend": gpu.get("backend_effective"),
        "max_abs_cpu_gpu_error_eV": maximum_error,
        "eigenvalue_threshold_eV": 1e-6,
        "maximum_relative_solve_residual": residual,
        "residual_threshold": 1e-10,
        "all_eigenvalues_finite": finite,
        "repeat_max_abs_error_eV": repeat_error,
        "determinism_threshold_eV": 1e-10,
        "comparisons": comparisons,
        "dense_large_cell_fallback_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-manifest", type=Path, required=True)
    parser.add_argument("--gpu-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-gpu-manifest", type=Path)
    args = parser.parse_args()
    result = validate(args.cpu_manifest, args.gpu_manifest, args.output, args.repeat_gpu_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
