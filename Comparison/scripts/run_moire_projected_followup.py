#!/usr/bin/env python3
"""Resume the validated N=480 projected-band and PDOS stages sequentially."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from run_deeph_sparse_spectrum import (
    DEFAULT_GPU_ENVIRONMENT,
    estimated_neutrality_reference,
    run as run_sparse_spectrum,
)
from run_graphene_hbn_moire_spectral_campaign import (
    DEFAULT_CONFIG,
    DEFAULT_ROOT,
    aggregate,
    load_config,
    now,
    read_json,
    write_json,
)
from run_moire_tracked_band_sweep import run_sweep
from validate_moire_projected_results import validate as validate_results


def validated_projection(manifest: dict) -> bool:
    diagnostics = manifest.get("projection", {}).get("diagnostics", {})
    return (
        manifest.get("status") == "completed"
        and manifest.get("projection", {}).get("status") == "completed"
        and diagnostics.get("status") == "valid"
        and float(diagnostics.get("maximum_normalization_error", 1)) < 1e-8
        and float(diagnostics.get("maximum_partition_sum_error", 1)) < 1e-6
        and float(diagnostics.get("maximum_generalized_relative_residual", 1)) < 1e-5
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--wait-minutes", type=int, default=180)
    args = parser.parse_args()
    config_path, root = args.config.resolve(), args.root.resolve()
    status_path = root / "spectra/projected_followup_status.json"
    input_dir = root / "predictions/graph2mat/n480/seed0/solver_input"
    smoke_path = root / "spectra/graph2mat/n480/seed0/tier_projected_smoke/solver_manifest.json"
    deadline = time.monotonic() + 60 * args.wait_minutes
    write_json(status_path, {"status": "waiting_for_smoke", "updated_at": now()})
    while time.monotonic() < deadline:
        smoke = read_json(smoke_path)
        sweep_status = read_json(root / "spectra/tracked_band_sweep_status.json")
        if smoke.get("status") == "completed" and sweep_status.get("status") == "completed":
            break
        if smoke.get("status") in {"failed", "resource_blocked"}:
            write_json(status_path, {"status": smoke["status"], "stage": "smoke", "updated_at": now()})
            return 1
        time.sleep(120)
    else:
        write_json(status_path, {"status": "failed", "stage": "smoke_wait_timeout", "updated_at": now()})
        return 1
    if not validated_projection(smoke):
        write_json(status_path, {"status": "failed", "stage": "smoke_validation", "updated_at": now()})
        return 1
    smoke["neutrality_reference"] = estimated_neutrality_reference(
        input_dir, float(smoke["fermi_level_eV"])
    )
    smoke["projection"]["diagnostics"]["validation_tolerances"] = {
        "normalization_absolute": 1e-8,
        "partition_absolute": 1e-6,
        "generalized_relative_residual": 1e-5,
        "energy_match_absolute_eV": 1e-8,
    }
    smoke["projection_mapping"]["group_orbital_counts"] = {
        name: len(indices)
        for name, indices in read_json(
            root / "spectra/graph2mat/n480/seed0/tier_projected_smoke/mulliken_projection_groups.json"
        ).get("groups", {}).items()
    }
    write_json(smoke_path, smoke)

    stages = (("kprime", 2, "kprime-gamma-k"), ("production", 51, "gamma-k-m-gamma"))
    for stage, points, path in stages:
        write_json(status_path, {"status": "running", "stage": stage, "updated_at": now()})
        result = run_sweep(
            config_path,
            root,
            [480],
            corrected_path=True,
            backend="gpu_cudss",
            num_bands=256,
            points_per_segment=points,
            band_path=path,
            project_mulliken=True,
        )
        if result.get("status") != "completed":
            write_json(status_path, {"status": result.get("status", "failed"), "stage": stage, "updated_at": now()})
            return 1

    write_json(status_path, {"status": "running", "stage": "dos_6x6", "updated_at": now()})
    neutrality = read_json(root / "neutrality_estimate.json")
    dos_output = root / "spectra/graph2mat/n480/seed0/tier_projected_dos_smoke"
    dos = read_json(dos_output / "solver_manifest.json")
    if not (dos.get("status") == "completed" and dos.get("projection", {}).get("status") == "completed"):
        dos = run_sparse_spectrum(
            input_dir,
            dos_output,
            job="dos",
            fermi_level=float(neutrality["energy_eV"]),
            num_bands=256,
            points_per_segment=1,
            kmesh=(6, 6, 1),
            environment_path=DEFAULT_GPU_ENVIRONMENT,
            backend="gpu_cudss",
            project_mulliken=True,
            dos_broadening_mev=0.5,
        )
    if not validated_projection(dos) or not dos.get("projected_dos"):
        write_json(status_path, {"status": dos.get("status", "failed"), "stage": "dos_6x6", "updated_at": now()})
        return 1
    aggregate(load_config(config_path), root)
    report = validate_results(root)
    write_json(root / "summary/projected_acceptance_report.json", report)
    if report.get("status") != "valid":
        write_json(status_path, {"status": "failed", "stage": "acceptance_audit", "updated_at": now()})
        return 1
    write_json(
        status_path,
        {
            "status": "completed",
            "completed_stages": ["smoke", "kprime", "production", "dos_6x6"],
            "dos_24x24_auto_launched": False,
            "updated_at": now(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
