#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9 -- direct 0.10/0.12 Ang probe for the OOD_CONDITIONAL gate.

The expert consultation on record for this task (event kind
``manual_expert_consultation_answered``, request EXP-ood012-manual) found the
existing ``delta_nl`` diagnostic (historical S3 pilot, axial_radial k=1/k=2,
amplitudes 0.03/0.08 only) "encouraging but not sufficient" for k=1, and
uninterpretable for k=2 without separating cancellation from genuine
curvature. Recommendation: run a direct probe near 0.10-0.12 Ang rather than
extrapolating from 0.08, using three diagnostics that share the same one-sided
denominator (so none of them can blow up from odd-term cancellation alone):

    C(r) = ||H(+r) - H(-r)|| / (||H(+r) - H0|| + ||H(-r) - H0||)   -- cancellation check
    N(r) = ||H(+r) + H(-r) - 2*H0|| / (||H(+r) - H0|| + ||H(-r) - H0||)  -- true curvature
    J_eff(r) = (H(+r) - H(-r)) / (2r); drift(r) = ||J_eff(r) - J_eff(0.01)|| / ||J_eff(0.01)||

This script:
  (a) runs new SIESTA single points at +/-0.01, +/-0.10, +/-0.12 Ang for k=1
      axial_radial (dim=3D, x/y/z axes) -- 0.03/0.08 are reused from the
      existing S3 pool, no new DFT for those;
  (b) computes C/N/J_eff-drift at every available amplitude for k=1;
  (c) for k=2, reuses the existing 0.03/0.08 pairs (no new DFT) and applies
      the same C(r)/N(r) split to test the expert's cancellation hypothesis,
      plus the suggested L_ij = J_i*u_i + J_j*u_j comparison against the
      actual correlated response where the k=2 direction decomposes onto
      two k=1 axes.

Writes a report; does NOT flip OOD_CONDITIONAL_INCLUDED_IN_TRAINING itself --
that stays a human decision informed by this report's numbers.
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
OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_ood012_probe"
NEW_AMPLITUDES_ANG = (0.01, 0.10, 0.12)
REUSED_AMPLITUDES_ANG = (0.03, 0.08)
EPS_EV = 1e-9


def _amp_tag(amplitude_ang: float) -> str:
    return f"{amplitude_ang:g}".replace(".", "p")


def _axis_key(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(round(abs(float(c)), 6) for c in direction)


def _sign(direction: tuple[float, float, float]) -> int:
    return 1 if max(direction) > 0 else -1


def _probe_configs(geometry: sampler.Geometry) -> list[tuple[str, sampler.Configuration]]:
    entries: list[tuple[str, sampler.Configuration]] = []
    for amplitude in NEW_AMPLITUDES_ANG:
        group = sampler.generate_axial_radial(geometry, k=1, dim="3D", radii_ang=(amplitude,))
        amp_tag = _amp_tag(amplitude)
        for index, config in enumerate(group):
            entries.append((f"ood012_probe__k1__3D__r{amp_tag}__{index:02d}", config))
    return entries


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


def _load_existing_axial_pair(geometry: sampler.Geometry, k: int, amplitude: float) -> dict[tuple[float, float, float], dict[int, np.ndarray]]:
    """+/- Hamiltonians already in the S3 pool for a given (k, amplitude)."""
    from run_epc_siesta_reference import backend_preflight  # noqa: F401  (import side effect check only)

    by_axis: dict[tuple[float, float, float], dict[int, np.ndarray]] = {}
    group = sampler.generate_axial_radial(geometry, k=k, dim="3D", radii_ang=(amplitude,))
    for index, config in enumerate(group):
        amp_tag = _amp_tag(amplitude)
        sample_id = f"axial_radial__k{k}__3D__r{amp_tag}__{index:03d}"
        ref_dir = S3_ROOT / "siesta_hamiltonians" / sample_id
        h, _path = pilot.gamma_hamiltonian(ref_dir)
        direction = tuple(round(float(c), 6) for c in config.metadata["direction"])
        by_axis.setdefault(_axis_key(direction), {})[_sign(direction)] = np.asarray(h)
    return by_axis


def compute_metrics(output_reference_root: Path, geometry: sampler.Geometry) -> dict[str, Any]:
    h0, _ = pilot.gamma_hamiltonian(S3_ROOT / "siesta_hamiltonians" / "reference")
    h0 = np.asarray(h0)

    # k=1: all five amplitudes, new + reused.
    by_amp_axis_k1: dict[float, dict[tuple[float, float, float], dict[int, np.ndarray]]] = {}
    for amplitude in NEW_AMPLITUDES_ANG:
        amp_tag = _amp_tag(amplitude)
        group = sampler.generate_axial_radial(geometry, k=1, dim="3D", radii_ang=(amplitude,))
        by_axis: dict[tuple[float, float, float], dict[int, np.ndarray]] = {}
        for index, config in enumerate(group):
            sample_id = f"ood012_probe__k1__3D__r{amp_tag}__{index:02d}"
            h, _path = pilot.gamma_hamiltonian(output_reference_root / sample_id)
            direction = tuple(round(float(c), 6) for c in config.metadata["direction"])
            by_axis.setdefault(_axis_key(direction), {})[_sign(direction)] = np.asarray(h)
        by_amp_axis_k1[amplitude] = by_axis
    for amplitude in REUSED_AMPLITUDES_ANG:
        by_amp_axis_k1[amplitude] = _load_existing_axial_pair(geometry, k=1, amplitude=amplitude)

    k1_rows = []
    j_eff_at_001: dict[tuple[float, float, float], np.ndarray] = {}
    for amplitude in sorted(by_amp_axis_k1):
        for axis_key, signed in by_amp_axis_k1[amplitude].items():
            if 1 not in signed or -1 not in signed:
                continue
            h_plus, h_minus = signed[1], signed[-1]
            e = np.linalg.norm(h_plus + h_minus - 2.0 * h0)
            o = np.linalg.norm(h_plus - h_minus)
            onesided = np.linalg.norm(h_plus - h0) + np.linalg.norm(h_minus - h0)
            j_eff = (h_plus - h_minus) / (2.0 * amplitude)
            if abs(amplitude - 0.01) < 1e-9:
                j_eff_at_001[axis_key] = j_eff
            k1_rows.append({
                "amplitude_ang": amplitude, "axis": axis_key,
                "delta_nl": float(e / (o + EPS_EV)),
                "C_cancellation": float(o / (onesided + EPS_EV)),
                "N_curvature": float(e / (onesided + EPS_EV)),
                "onesided_change_ev": float(onesided),
                "j_eff_norm_ev_per_ang": float(np.linalg.norm(j_eff)),
            })
    for row in k1_rows:
        ref = j_eff_at_001.get(row["axis"])
        if ref is not None and row["amplitude_ang"] != 0.01:
            amplitude = row["amplitude_ang"]
            axis_key = row["axis"]
            signed = by_amp_axis_k1[amplitude][axis_key]
            j_eff = (signed[1] - signed[-1]) / (2.0 * amplitude)
            row["j_eff_drift_vs_0.01"] = float(np.linalg.norm(j_eff - ref) / (np.linalg.norm(ref) + EPS_EV))

    # k=2: reuse existing data only, same C/N split, no new DFT.
    k2_rows = []
    by_axis_k2 = {}
    for amplitude in REUSED_AMPLITUDES_ANG:
        by_axis_k2[amplitude] = _load_existing_axial_pair(geometry, k=2, amplitude=amplitude)
    for amplitude in sorted(by_axis_k2):
        for axis_key, signed in by_axis_k2[amplitude].items():
            if 1 not in signed or -1 not in signed:
                continue
            h_plus, h_minus = signed[1], signed[-1]
            e = np.linalg.norm(h_plus + h_minus - 2.0 * h0)
            o = np.linalg.norm(h_plus - h_minus)
            onesided = np.linalg.norm(h_plus - h0) + np.linalg.norm(h_minus - h0)
            k2_rows.append({
                "amplitude_ang": amplitude, "axis": axis_key,
                "delta_nl": float(e / (o + EPS_EV)),
                "C_cancellation": float(o / (onesided + EPS_EV)),
                "N_curvature": float(e / (onesided + EPS_EV)),
                "onesided_change_ev": float(onesided),
            })

    return {"k1": k1_rows, "k2": k2_rows}


def main() -> int:
    structures_root = OUTPUT_ROOT / "structures"
    canonical_base_fdf = OUTPUT_ROOT / "_canonical_base.fdf"
    canonical_ang_base_fdf(pilot.DEFAULT_MATERIAL_FDF, canonical_base_fdf)

    geometry = sampler.load_graphene_primitive()
    entries = _probe_configs(geometry)
    _materialize(entries, structures_root=structures_root, canonical_base_fdf=canonical_base_fdf)

    try:
        run_manifest = run_derivative_siesta_references(
            stencil_root=OUTPUT_ROOT,
            source_dataset_root=REPO_ROOT / "materials/graphene",
            siesta_command=pilot.DEFAULT_SIESTA_COMMAND,
            workers=6,
            diagnostic_only=True,
        )
    except DerivativeSiestaReferenceError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    output_reference_root = Path(run_manifest["output_reference_root"])
    metrics = compute_metrics(output_reference_root, geometry)

    report_path = OUTPUT_ROOT / "ood012_probe_report.json"
    report_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(report_path), "n_k1_rows": len(metrics["k1"]), "n_k2_rows": len(metrics["k2"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
