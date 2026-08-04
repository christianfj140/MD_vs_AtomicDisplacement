#!/usr/bin/env python3
"""Audit persisted projected moiré bands/PDOS against the UI acceptance contract."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import urllib.request
from pathlib import Path

from run_graphene_hbn_moire_spectral_campaign import DEFAULT_ROOT, read_json, write_json


REQUIRED_WEIGHTS = {
    "weight_c_pz", "weight_c_total", "weight_graphene_lower", "weight_graphene_upper",
    "weight_hbn", "weight_b", "weight_n", "layer_polarization",
    "normalization_cdagger_s_c", "generalized_relative_residual", "projection_status",
}


def validate(root: Path, ui_url: str = "http://127.0.0.1:8770") -> dict:
    base = root / "spectra/graph2mat/n480/seed0"
    paths = {
        "smoke": base / "tier_projected_smoke/solver_manifest.json",
        "production": base / "tier_projected_production/solver_manifest.json",
        "kprime": base / "tier_projected_kprime_diagnostic/solver_manifest.json",
        "dos": base / "tier_projected_dos_smoke/solver_manifest.json",
    }
    manifests = {name: read_json(path) for name, path in paths.items()}
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, detail: object) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    expected_samples = {"smoke": 31, "production": 151, "kprime": 3}
    for name, expected in expected_samples.items():
        manifest = manifests[name]
        bands = manifest.get("bands") or []
        record(f"{name}_completed", manifest.get("status") == "completed", manifest.get("status"))
        record(f"{name}_sample_count", manifest.get("k_path", {}).get("sample_count") == expected, manifest.get("k_path", {}).get("sample_count"))
        record(f"{name}_256_states", manifest.get("num_bands") == 256 and len(bands) == 256 * expected, {"num_bands": manifest.get("num_bands"), "rows": len(bands)})
        projection = manifest.get("projection") or {}
        diagnostics = projection.get("diagnostics") or {}
        record(f"{name}_projection_valid", projection.get("status") == "completed" and diagnostics.get("status") == "valid", diagnostics)
        missing = sorted(REQUIRED_WEIGHTS - set(bands[0])) if bands else sorted(REQUIRED_WEIGHTS)
        finite = all(
            math.isfinite(float(point[key]))
            for point in bands
            for key in REQUIRED_WEIGHTS - {"projection_status"}
        ) if bands and not missing else False
        record(f"{name}_state_fields", not missing and finite, {"missing": missing, "finite": finite})
        record(f"{name}_no_eigenvectors", diagnostics.get("eigenvectors_persisted") is False, diagnostics.get("eigenvectors_persisted"))
        mapping = manifest.get("projection_mapping") or {}
        record(f"{name}_orbital_mapping", (
            mapping.get("carbon_pz_local_indices_zero_based") == [3]
            and mapping.get("group_orbital_counts", {}).get("carbon_pz") == 11164
            and bool(mapping.get("basis_evidence"))
        ), mapping)
        path_validation = manifest.get("k_path", {}).get("validation") or {}
        record(f"{name}_path_geometry", (
            path_validation.get("K_wigner_seitz_equidistant_points", 0) >= 3
            and path_validation.get("K_prime_wigner_seitz_equidistant_points", 0) >= 3
            and path_validation.get("M_wigner_seitz_equidistant_points", 0) >= 2
            and float(path_validation.get("Gamma_K_distance_inv_ang", 0)) > 0
            and float(path_validation.get("Gamma_M_distance_inv_ang", 0)) > 0
        ), path_validation)
        record(f"{name}_honest_ordering", (
            manifest.get("band_ordering", {}).get("band_character_continuity_claimed") is False
            and manifest.get("band_representations", {}).get("raw_arpack_order", {}).get("bands")
        ), manifest.get("band_ordering"))
        record(f"{name}_safe_resources", (
            float(manifest.get("minimum_observed_free_disk_percent", 0)) > 10
            and float(manifest.get("maximum_observed_cpu_temperature_c", 1e9)) < float(manifest.get("maximum_allowed_cpu_temperature_c", 0))
            and float((manifest.get("gpu_observations") or {}).get("maximum_temperature_c", 1e9)) < float(manifest.get("maximum_allowed_gpu_temperature_c", 0))
            and not manifest.get("error_summary")
        ), {"disk": manifest.get("minimum_observed_free_disk_percent"), "cpu": manifest.get("maximum_observed_cpu_temperature_c"), "gpu": manifest.get("gpu_observations")})

    record("kprime_comparison", manifests["kprime"].get("kprime_comparison", {}).get("status") == "completed", manifests["kprime"].get("kprime_comparison"))
    dos = manifests["dos"]
    projected_dos = dos.get("projected_dos") or []
    dos_fields = {"dos_total", "pdos_c_pz", "pdos_graphene_lower", "pdos_graphene_upper", "pdos_hbn"}
    record("dos_completed", dos.get("status") == "completed" and dos.get("projection", {}).get("status") == "completed", dos.get("status"))
    record("dos_mesh", dos.get("kmesh") == [6, 6, 1] and dos.get("projection", {}).get("diagnostics", {}).get("kpoint_count") == 36, {"kmesh": dos.get("kmesh"), "diagnostics": dos.get("projection", {}).get("diagnostics")})
    record("dos_broadening", dos.get("dos_broadening_meV") == 0.5, dos.get("dos_broadening_meV"))
    record("dos_scope_honest", dos.get("projection", {}).get("full_orbital_spectrum") is False and bool(dos.get("projection", {}).get("scope_limitation")), dos.get("projection"))
    dos_diagnostics = dos.get("projection", {}).get("diagnostics", {})
    record("dos_state_integral", abs(float(dos_diagnostics.get("integrated_total_states_in_energy_window", 0)) - 256.0) < 0.1, dos_diagnostics.get("integrated_total_states_in_energy_window"))
    record("dos_curves", bool(projected_dos) and dos_fields <= set(projected_dos[0]), len(projected_dos))
    record("dos_observables", dos.get("dos_observables", {}).get("gap_claimed") is False, dos.get("dos_observables"))
    record("dos_no_eigenvectors", dos.get("projection", {}).get("eigenvectors_persisted") is False, dos.get("projection", {}).get("eigenvectors_persisted"))
    record("dos_safe_resources", (
        float(dos.get("minimum_observed_free_disk_percent", 0)) > 10
        and float(dos.get("maximum_observed_cpu_temperature_c", 1e9)) < float(dos.get("maximum_allowed_cpu_temperature_c", 0))
        and float((dos.get("gpu_observations") or {}).get("maximum_temperature_c", 1e9)) < float(dos.get("maximum_allowed_gpu_temperature_c", 0))
        and not dos.get("error_summary")
    ), {"disk": dos.get("minimum_observed_free_disk_percent"), "cpu": dos.get("maximum_observed_cpu_temperature_c"), "gpu": dos.get("gpu_observations")})

    config = read_json(root.parents[1] / "config/graphene_hbn_magic_angle_spectral_campaign.json")
    solver = config.get("solver") or {}
    record("dos_24x24_ready_not_auto", solver.get("dos_production_kmesh") == [24, 24, 1] and solver.get("auto_launch_dos_production") is False, solver.get("dos_production_kmesh"))
    record("next_sizes_reuse_projection", solver.get("project_mulliken") is True and solver.get("band_path") == "gamma-k-m-gamma", {"project_mulliken": solver.get("project_mulliken"), "band_path": solver.get("band_path")})
    record("disk_current_above_10", 100.0 * shutil.disk_usage(root).free / shutil.disk_usage(root).total > 10, 100.0 * shutil.disk_usage(root).free / shutil.disk_usage(root).total)

    summary = read_json(root / "summary/spectral_results.json")
    row = next((item for item in summary.get("spectra") or [] if item.get("model") == "graph2mat" and item.get("training_size") == 480 and item.get("seed") == 0), {})
    record("summary_published", row.get("projection", {}).get("status") == "completed" and bool(row.get("projected_dos")), {"tier": row.get("visible_band_tier"), "dos_tier": row.get("visible_dos_tier")})
    observables = row.get("projected_low_energy_observables") or {}
    projected_observable_fields = {
        "carbon_pz_weight_mean", "carbon_pz_weight_min", "carbon_pz_weight_max",
        "hbn_weight_mean", "hbn_weight_min", "hbn_weight_max",
        "layer_polarization_mean", "layer_polarization_min", "layer_polarization_max",
    }
    record("summary_projected_observables", (
        projected_observable_fields <= set(observables)
        and all(math.isfinite(float(observables[key])) for key in projected_observable_fields)
        and observables.get("gap_and_bandwidth_claimed") is False
    ), observables)
    neutrality = manifests["production"].get("neutrality_reference") or {}
    record("neutrality_honest", (
        neutrality.get("label") == "cero visual de neutralidad estimado"
        and neutrality.get("expected_neutral_valence_electrons") == 66984
        and neutrality.get("chemical_potential_available") is False
        and neutrality.get("gap_eV") is None
    ), neutrality)
    try:
        html = urllib.request.urlopen(f"{ui_url}/", timeout=5).read().decode("utf-8")
        api = json.loads(urllib.request.urlopen(f"{ui_url}/api/cross-testing/bilayer/spectral/results", timeout=15).read())
        controls = ("ct-spectral-plot-mode", "ct-spectral-weight", "ct-spectral-artifact", "ct-spectral-diagnostic-window")
        record("ui_served_controls", all(control in html for control in controls), controls)
        api_row = next((item for item in api.get("summary", {}).get("spectra") or [] if item.get("model") == "graph2mat" and item.get("training_size") == 480 and item.get("seed") == 0), {})
        record("ui_served_real_summary", api_row.get("projection", {}).get("status") == "completed" and bool(api_row.get("projected_dos")), {"tier": api_row.get("visible_band_tier"), "dos": bool(api_row.get("projected_dos"))})
    except Exception as error:  # noqa: BLE001 - report the endpoint failure
        record("ui_served_controls", False, str(error))
        record("ui_served_real_summary", False, str(error))

    return {
        "status": "valid" if all(check["passed"] for check in checks.values()) else "invalid",
        "checks": checks,
        "artifact_paths": {name: str(path) for name, path in paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--ui-url", default="http://127.0.0.1:8770")
    args = parser.parse_args()
    report = validate(args.root.resolve(), args.ui_url)
    write_json(args.root.resolve() / "summary/projected_acceptance_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
