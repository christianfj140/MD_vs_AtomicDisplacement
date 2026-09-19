#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9 -- k=2 direct probe for the OOD_CONDITIONAL gate.

Follow-up to ``dataset_design_w90_001_s9_ood012_probe.py`` (k=1, which passed
cleanly through 0.12 Ang). That script's k=2 reading (axial_radial family)
showed a near-total cancellation: onesided Hamiltonian change ~1e-4 eV versus
~1-8 eV for k=1. Reading the sampler code explains *why*, mechanically:
``generate_axial_radial`` for k=2 selects the graphene primitive cell's only
bonded pair (the two basis atoms) and displaces BOTH by the SAME vector
(``displacements = {index: vector for index in active.indices}``) -- a near-
rigid shift of the whole 2-atom basis, which is close to a symmetry
operation, not a probe of genuine bond-level correlated response. That is a
geometric fact about this one mode, not evidence about k=2 in general.

This script tests two separate things properly:

  (a) CONFIRMS the rigid-shift explanation directly, rather than asserting
      it: measures the k=1 Jacobian for EACH atom independently (center_index
      0 and 1) and checks whether J_0 + J_1 ~ 0 along each axis -- if the sum
      of the two independent linear responses is itself near zero, the k=2
      cancellation is explained by real (near-)symmetry, not by noise or by
      an artifact of this diagnostic.

  (b) The genuinely different, physically meaningful k=2 test the expert
      flagged: ``local_pair_modes``' longitudinal (bond-stretch) and
      transverse_ip/transverse_opp (bond-bend) modes, where the two atoms
      move OPPOSITE to each other (partner_sign=-1) -- no rigid-shift
      symmetry protects these, so C(r)/N(r)/delta_nl here are a real
      anharmonicity reading for correlated motion, not a repeat of (a)'s
      cancellation artifact.

Same diagnostics as the k=1 probe throughout:
    C(r) = ||H(+r)-H(-r)|| / (||H(+r)-H0|| + ||H(-r)-H0||)   -- cancellation
    N(r) = ||H(+r)+H(-r)-2H0|| / (||H(+r)-H0|| + ||H(-r)-H0||)  -- curvature
    J_eff(r) = (H(+r)-H(-r)) / (2r); drift vs J_eff(0.01)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import w90_displacement_sampler_family as sampler  # noqa: E402
from run_epc_siesta_reference import canonical_ang_base_fdf  # noqa: E402
from run_hamiltonian_derivative_siesta_references import (  # noqa: E402
    DerivativeSiestaReferenceError,
    run_derivative_siesta_references,
)
from fdf_materialization import extract_fdf_structure, materialize_sample_fdf  # noqa: E402
import run_w90_displacement_siesta_pilot as pilot  # noqa: E402

S3_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s3"
OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_ood012_k2_probe"
AMPLITUDES_ANG = (0.01, 0.03, 0.08, 0.10, 0.12)
PAIR_MODES = ("longitudinal", "transverse_ip", "transverse_opp")
EPS_EV = 1e-9


def _amp_tag(amplitude_ang: float) -> str:
    sign = "m" if amplitude_ang < 0 else "p"
    return f"{sign}{abs(amplitude_ang):g}".replace(".", "d")


def _materialize(entries: list[tuple[str, sampler.Configuration]], *, structures_root: Path, canonical_base_fdf: Path) -> None:
    base_structure = extract_fdf_structure(canonical_base_fdf)
    base_positions = [tuple(p) for p in base_structure.positions_ang]
    for sample_id, config in entries:
        displaced = [
            tuple(base_positions[i][axis] + config.displacements_ang.get(i, (0.0, 0.0, 0.0))[axis] for axis in range(3))
            for i in range(len(base_positions))
        ]
        out_dir = structures_root / sample_id
        materialize_sample_fdf(canonical_base_fdf, out_dir / "RUN.fdf", positions_ang=displaced, single_point=True)
        metadata = dict(config.metadata)
        metadata["sample_id"] = sample_id
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Part (a): does J_0 + J_1 ~ 0 explain the axial_radial k=2 cancellation?
# --------------------------------------------------------------------------


def _atom1_jacobian_configs(geometry: sampler.Geometry) -> list[tuple[str, sampler.Configuration]]:
    """k=1 probes centered on atom index 1 (axial_radial's k=2 partner), all 5 amplitudes."""
    entries: list[tuple[str, sampler.Configuration]] = []
    for amplitude in AMPLITUDES_ANG:
        group = sampler.generate_axial_radial(geometry, k=1, dim="3D", radii_ang=(amplitude,), center_index=1)
        for index, config in enumerate(group):
            entries.append((f"ood012_k2__atom1_k1__r{_amp_tag(amplitude)}__{index:02d}", config))
    return entries


def _axis_key(direction) -> tuple[float, float, float]:
    return tuple(round(abs(float(c)), 6) for c in direction)


def _sign(direction) -> int:
    return 1 if max(direction) > 0 else -1


def rigid_shift_check(output_reference_root: Path, geometry: sampler.Geometry, h0: np.ndarray) -> list[dict[str, Any]]:
    # atom-0 Jacobian: reuse the existing S9-ood012 k=1 probe (0.01/0.10/0.12 new, 0.03/0.08 in S3 pool).
    j0_by_amp_axis: dict[float, dict[tuple, np.ndarray]] = {}
    for amplitude in AMPLITUDES_ANG:
        by_axis: dict[tuple, dict[int, np.ndarray]] = {}
        group = sampler.generate_axial_radial(geometry, k=1, dim="3D", radii_ang=(amplitude,), center_index=0)
        for index, config in enumerate(group):
            direction = tuple(round(float(c), 6) for c in config.metadata["direction"])
            if amplitude in (0.03, 0.08):
                amp_tag = f"{amplitude:g}".replace(".", "p")
                sample_id = f"axial_radial__k1__3D__r{amp_tag}__{index:03d}"
                h, _ = pilot.gamma_hamiltonian(S3_ROOT / "siesta_hamiltonians" / sample_id)
            else:
                amp_tag = f"{amplitude:g}".replace(".", "p")
                sample_id = f"ood012_probe__k1__3D__r{amp_tag}__{index:02d}"
                probe_root = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_ood012_probe/structures"
                h, _ = pilot.gamma_hamiltonian(probe_root.parent / "siesta_hamiltonians" / sample_id)
            by_axis.setdefault(_axis_key(direction), {})[_sign(direction)] = np.asarray(h)
        j0_by_amp_axis[amplitude] = {
            axis: (signed[1] - signed[-1]) / (2.0 * amplitude)
            for axis, signed in by_axis.items() if 1 in signed and -1 in signed
        }

    # atom-1 Jacobian: fresh probe, computed by this script.
    j1_by_amp_axis: dict[float, dict[tuple, np.ndarray]] = {}
    for amplitude in AMPLITUDES_ANG:
        by_axis: dict[tuple, dict[int, np.ndarray]] = {}
        group = sampler.generate_axial_radial(geometry, k=1, dim="3D", radii_ang=(amplitude,), center_index=1)
        for index, config in enumerate(group):
            direction = tuple(round(float(c), 6) for c in config.metadata["direction"])
            sample_id = f"ood012_k2__atom1_k1__r{_amp_tag(amplitude)}__{index:02d}"
            h, _ = pilot.gamma_hamiltonian(output_reference_root / sample_id)
            by_axis.setdefault(_axis_key(direction), {})[_sign(direction)] = np.asarray(h)
        j1_by_amp_axis[amplitude] = {
            axis: (signed[1] - signed[-1]) / (2.0 * amplitude)
            for axis, signed in by_axis.items() if 1 in signed and -1 in signed
        }

    # Predicted "both atoms move together" response: (J0+J1)*r*axis, compared to the
    # ACTUAL observed axial_radial k=2 onesided change at the same amplitude/axis.
    rows = []
    for amplitude in (0.03, 0.08):  # amplitudes where the real k=2 axial_radial data exists
        amp_tag = f"{amplitude:g}".replace(".", "p")
        group = sampler.generate_axial_radial(geometry, k=2, dim="3D", radii_ang=(amplitude,))
        by_axis_k2: dict[tuple, dict[int, np.ndarray]] = {}
        for index, config in enumerate(group):
            direction = tuple(round(float(c), 6) for c in config.metadata["direction"])
            sample_id = f"axial_radial__k2__3D__r{amp_tag}__{index:03d}"
            h, _ = pilot.gamma_hamiltonian(S3_ROOT / "siesta_hamiltonians" / sample_id)
            by_axis_k2.setdefault(_axis_key(direction), {})[_sign(direction)] = np.asarray(h)
        for axis_key, signed in by_axis_k2.items():
            if 1 not in signed or -1 not in signed:
                continue
            actual_onesided = np.linalg.norm(signed[1] - h0) + np.linalg.norm(signed[-1] - h0)
            j0 = j0_by_amp_axis.get(amplitude, {}).get(axis_key)
            j1 = j1_by_amp_axis.get(amplitude, {}).get(axis_key)
            if j0 is None or j1 is None:
                continue
            predicted_linear_response = np.linalg.norm((j0 + j1) * amplitude)
            rows.append({
                "amplitude_ang": amplitude, "axis": axis_key,
                "j0_norm_ev_per_ang": float(np.linalg.norm(j0)),
                "j1_norm_ev_per_ang": float(np.linalg.norm(j1)),
                "j0_plus_j1_norm_ev_per_ang": float(np.linalg.norm(j0 + j1)),
                "predicted_linear_onesided_ev": float(2 * predicted_linear_response),
                "actual_k2_onesided_ev": float(actual_onesided),
            })
    return rows


# --------------------------------------------------------------------------
# Part (b): local_pair_modes -- genuine relative (bond-stretch/bend) motion
# --------------------------------------------------------------------------


def _pair_mode_configs(geometry: sampler.Geometry) -> list[tuple[str, sampler.Configuration]]:
    entries: list[tuple[str, sampler.Configuration]] = []
    for amplitude in AMPLITUDES_ANG:
        for signed_amp in (amplitude, -amplitude):
            group = sampler.generate_local_pair_modes(geometry, dim="3D", amplitudes_ang=(signed_amp,))
            for config in group:
                mode = config.metadata["mode"]
                if mode not in PAIR_MODES:
                    continue
                sample_id = f"ood012_k2__pairmode_{mode}__r{_amp_tag(signed_amp)}"
                entries.append((sample_id, config))
    return entries


def pair_mode_metrics(output_reference_root: Path, h0: np.ndarray) -> list[dict[str, Any]]:
    by_mode_amp: dict[tuple[str, float], dict[int, np.ndarray]] = {}
    for amplitude in AMPLITUDES_ANG:
        for mode in PAIR_MODES:
            for sign, signed_amp in ((1, amplitude), (-1, -amplitude)):
                sample_id = f"ood012_k2__pairmode_{mode}__r{_amp_tag(signed_amp)}"
                h, _ = pilot.gamma_hamiltonian(output_reference_root / sample_id)
                by_mode_amp.setdefault((mode, amplitude), {})[sign] = np.asarray(h)

    rows = []
    j_eff_at_001: dict[str, np.ndarray] = {}
    for (mode, amplitude), signed in sorted(by_mode_amp.items()):
        if 1 not in signed or -1 not in signed:
            continue
        h_plus, h_minus = signed[1], signed[-1]
        e = np.linalg.norm(h_plus + h_minus - 2.0 * h0)
        o = np.linalg.norm(h_plus - h_minus)
        onesided = np.linalg.norm(h_plus - h0) + np.linalg.norm(h_minus - h0)
        j_eff = (h_plus - h_minus) / (2.0 * amplitude)
        if abs(amplitude - 0.01) < 1e-9:
            j_eff_at_001[mode] = j_eff
        rows.append({
            "mode": mode, "amplitude_ang": amplitude,
            "delta_nl": float(e / (o + EPS_EV)),
            "C_cancellation": float(o / (onesided + EPS_EV)),
            "N_curvature": float(e / (onesided + EPS_EV)),
            "onesided_change_ev": float(onesided),
            "j_eff_norm_ev_per_ang": float(np.linalg.norm(j_eff)),
        })
    for row in rows:
        ref = j_eff_at_001.get(row["mode"])
        if ref is not None and row["amplitude_ang"] != 0.01:
            signed = by_mode_amp[(row["mode"], row["amplitude_ang"])]
            j_eff = (signed[1] - signed[-1]) / (2.0 * row["amplitude_ang"])
            row["j_eff_drift_vs_0.01"] = float(np.linalg.norm(j_eff - ref) / (np.linalg.norm(ref) + EPS_EV))
    return rows


def main() -> int:
    structures_root = OUTPUT_ROOT / "structures"
    canonical_base_fdf = OUTPUT_ROOT / "_canonical_base.fdf"
    canonical_ang_base_fdf(pilot.DEFAULT_MATERIAL_FDF, canonical_base_fdf)
    geometry = sampler.load_graphene_primitive()

    entries = _atom1_jacobian_configs(geometry) + _pair_mode_configs(geometry)
    _materialize(entries, structures_root=structures_root, canonical_base_fdf=canonical_base_fdf)

    try:
        run_manifest = run_derivative_siesta_references(
            stencil_root=OUTPUT_ROOT,
            source_dataset_root=REPO_ROOT / "materials/graphene",
            siesta_command=pilot.DEFAULT_SIESTA_COMMAND,
            workers=8,
            diagnostic_only=True,
        )
    except DerivativeSiestaReferenceError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    output_reference_root = Path(run_manifest["output_reference_root"])
    h0, _ = pilot.gamma_hamiltonian(S3_ROOT / "siesta_hamiltonians" / "reference")
    h0 = np.asarray(h0)

    report = {
        "rigid_shift_check": rigid_shift_check(output_reference_root, geometry, h0),
        "pair_modes": pair_mode_metrics(output_reference_root, h0),
    }
    report_path = OUTPUT_ROOT / "k2_probe_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(report_path),
        "n_rigid_shift_rows": len(report["rigid_shift_check"]),
        "n_pair_mode_rows": len(report["pair_modes"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
