#!/usr/bin/env python3
"""graphene_Gamma_E2g_y_delta_convergence (direct campaign) +
graphene_Gamma_E2g_doublet_convergence.

The C3 symmetry shortcut was BLOCKED (SCF k-mesh is not C3-covariant; see
diagnose_c3_kmesh_and_field_covariance.py), so E2g_y is measured directly from
its own 12-run SIESTA campaign (e2g_y_campaign_preregistration_v1.json), using
the collective mode C1 = +Q/sqrt(2) y-hat, C2 = -Q/sqrt(2) y-hat -- the same
five-h plateau/Richardson gate already used for E2g_x, and the SAME
S_L(h)/S_R(h) connection route (certify_uniform_translation_pao_covariance.
_translation_connections, generalized to any collective Direction, not
reimplemented per direction).

graphene_Gamma_E2g_doublet_convergence reports both the worst single component
and the combined pair norm E_pair = sqrt((||E_x||^2 + ||E_y||^2)/2), labelled
explicitly as "doublet convergence by direct measurement" (E2g_x was itself
independently measured via the C2 symmetry shortcut, E2g_y via a direct
12-run campaign -- neither is derived from the other).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

import sisl  # noqa: E402

import fd_perturbation_space as fdp  # noqa: E402
from artifact_signature import file_sha256, input_signature_sha256  # noqa: E402
from certify_delta_plateau_topology import (  # noqa: E402
    D5_WEIGHTS, DECISIVE_WINDOW, H_LADDER, MAX_LIMIT_EV_PER_ANG, PROD_SZ_ROOT,
    REL_LIMIT, RMS_LIMIT_EV_PER_ANG, _compare, _d5, _k_matrix, _point,
    _support_pairs, _tshs_path,
)
from certify_full_basis_connection_semantic_v5 import solve  # noqa: E402
from certify_uniform_translation_pao_covariance import _translation_connections  # noqa: E402
from cross_basis_projection_preflight import K_POINTS, _basis_geometry  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/graphene_gamma_e2g_y_delta_convergence"
E2G_X_PATH = (
    REPO_ROOT / "Comparison/results/epc/graphene_gamma_sublattice_symmetry_reduction"
    / "graphene_gamma_sublattice_symmetry_reduction.json"
)
GAMMA = (0.0, 0.0, 0.0)


def _tag(delta_ang: float) -> str:
    magnitude = "d" + f"{abs(delta_ang):.6g}".replace(".", "p")
    return f"{magnitude}_{'plus' if delta_ang > 0 else 'minus'}"


def _e2g_y_run_dir(delta_ang: float) -> Path:
    return PROD_SZ_ROOT / "e2g_y" / _tag(delta_ang)


def main() -> int:
    e2g_x = json.loads(E2G_X_PATH.read_text())
    e2g_x_verdict = e2g_x["graphene_Gamma_E2g_x_delta_convergence"]["verdict"]

    central_label, central_h, central_fermi, central_vacuum, atom = _matrix_data(PROD_CENTRAL)
    shape = tuple(int(x) for x in sisl.get_sile(str(PROD_CENTRAL / f"{central_label}.VT")).read_grid().shape)
    base = _basis_geometry(central_h.geometry, PROD_CENTRAL / "C.ion.xml")
    dvolume = abs(np.linalg.det(base.cell)) / np.prod(shape)
    direction = fdp.collective([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], name="e2g_y")

    s0 = np.asarray(central_h.Sk(k=GAMMA, format="array"))
    k0 = _k_matrix(central_h, central_fermi, central_vacuum, GAMMA)

    # -- topology: independently re-verify {(i,j,R)} across the 12 new geometries
    reference_pairs = _support_pairs(_tshs_path(PROD_CENTRAL))
    reference_hash = input_signature_sha256({"kind": "support_pairs", "pairs": reference_pairs})
    topology_rows, topology_constant = [], True
    for h in H_LADDER:
        for step in D5_WEIGHTS:
            run_dir = _e2g_y_run_dir(step * h)
            pairs = _support_pairs(_tshs_path(run_dir))
            pair_hash = input_signature_sha256({"kind": "support_pairs", "pairs": pairs})
            matches = pair_hash == reference_hash
            topology_constant &= matches
            topology_rows.append({"run_dir": str(run_dir.relative_to(REPO_ROOT)), "matches_reference": matches})

    # -- Delta_E2g_y(h) end-to-end for every h -------------------------------
    delta_by_h, provenance_by_h = {}, {}
    for h in H_LADDER:
        points_by_step, provenance = {}, {}
        for step in D5_WEIGHTS:
            run_dir = _e2g_y_run_dir(step * h)
            hamiltonian, fermi, vacuum, tshs_path = _point(run_dir)
            points_by_step[step] = hamiltonian, fermi, vacuum
            provenance[step] = {"run_dir": str(run_dir.relative_to(REPO_ROOT)), "tshs_sha256": file_sha256(tshs_path)}
        provenance_by_h[h] = provenance
        connections = _translation_connections(direction, h, base, atom, shape, dvolume, [GAMMA])
        k_by_step = {step: _k_matrix(*points_by_step[step], GAMMA) for step in points_by_step}
        d5k = _d5(k_by_step, h)
        s_l, s_r = connections[0]
        left_term, _ = solve(s0, k0)
        right_term, _ = solve(s0, s_r)
        delta_by_h[h] = d5k - s_l @ left_term - k0 @ right_term

    s0_by_k = [s0]
    e12 = _compare([delta_by_h[0.005]], [delta_by_h[0.0025]], s0_by_k)
    e23 = _compare([delta_by_h[0.0025]], [delta_by_h[0.00125]], s0_by_k)
    r1 = (16.0 * delta_by_h[0.0025] - delta_by_h[0.005]) / 15.0
    r2 = (16.0 * delta_by_h[0.00125] - delta_by_h[0.0025]) / 15.0
    richardson_stability = _compare([r1], [r2], s0_by_k)
    production_vs_extrapolated = _compare([delta_by_h[0.005]], [r2], s0_by_k)

    e2g_y_passed = (topology_constant and e12["passed"] and e23["passed"]
                     and richardson_stability["passed"] and production_vs_extrapolated["passed"])

    # -- doublet: combine with the already-certified E2g_x worst-case numbers
    x_e12 = e2g_x["graphene_Gamma_E2g_x_delta_convergence"]["e12_0p005_to_0p0025"]["worst_rms_ev_per_ang"]
    x_e23 = e2g_x["graphene_Gamma_E2g_x_delta_convergence"]["e23_0p0025_to_0p00125"]["worst_rms_ev_per_ang"]
    y_e12 = e12["worst_rms_ev_per_ang"]
    y_e23 = e23["worst_rms_ev_per_ang"]
    pair_e12 = float(np.sqrt((x_e12**2 + y_e12**2) / 2.0))
    pair_e23 = float(np.sqrt((x_e23**2 + y_e23**2) / 2.0))
    doublet_passed = (e2g_x_verdict == "PASS" and e2g_y_passed
                       and pair_e12 < RMS_LIMIT_EV_PER_ANG and pair_e23 < RMS_LIMIT_EV_PER_ANG)

    report = {
        "schema": "graphene_gamma_e2g_y_delta_convergence_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "measurement": "direct 12-run SIESTA campaign (C3 shortcut BLOCKED, not used)",
        "h_ladder_ang": list(H_LADDER), "decisive_window_ang": list(DECISIVE_WINDOW),
        "thresholds": {"rms_ev_per_ang": RMS_LIMIT_EV_PER_ANG, "relative": REL_LIMIT,
                        "max_ev_per_ang": MAX_LIMIT_EV_PER_ANG},
        "topology": {"constant": topology_constant, "reference_sha256": reference_hash, "rows": topology_rows},
        "graphene_Gamma_E2g_y_delta_convergence": {
            "verdict": "PASS" if e2g_y_passed else "NO_GO",
            "e12_0p005_to_0p0025": e12, "e23_0p0025_to_0p00125": e23,
            "richardson": {"r1_vs_r2": richardson_stability, "production_0p005_vs_r2": production_vs_extrapolated},
            "run_provenance_by_h": {str(h): provenance_by_h[h] for h in H_LADDER},
        },
        "graphene_Gamma_E2g_doublet_convergence_by_direct_measurement": {
            "verdict": "PASS" if doublet_passed else "NO_GO",
            "e2g_x_verdict": e2g_x_verdict, "e2g_x_measurement": "C2 symmetry reduction (independent)",
            "e2g_y_verdict": "PASS" if e2g_y_passed else "NO_GO", "e2g_y_measurement": "direct 12-run campaign",
            "worst_component_rms_e12_ev_per_ang": max(x_e12, y_e12),
            "worst_component_rms_e23_ev_per_ang": max(x_e23, y_e23),
            "pair_norm_e12_ev_per_ang": pair_e12, "pair_norm_e23_ev_per_ang": pair_e23,
            "pair_norm_formula": "sqrt((||E_x||^2 + ||E_y||^2) / 2)",
        },
        "unaffected_by_this_gate": {
            "checkpoint_lineage": "NO_GO", "fine_tuning_candidate": "BLOCKED",
            "delta_out_closure": "NO_GO", "full_KS": "BLOCKED",
            "graphene_Gamma_sublattice_C3_symmetry_reduction": "BLOCKED",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "graphene_gamma_e2g_y_delta_convergence.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"graphene_Gamma_E2g_y_delta_convergence: {'PASS' if e2g_y_passed else 'NO_GO'} -> {path}")
    print(f"graphene_Gamma_E2g_doublet_convergence_by_direct_measurement: "
          f"{'PASS' if doublet_passed else 'NO_GO'}")
    return 0 if doublet_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
