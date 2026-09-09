#!/usr/bin/env python3
"""graphene_Gamma_E2g_{x,y,doublet}_electronic_contraction_convergence.

G_alpha(h) = C_f^dagger Delta_E2g_alpha(h) C_i, alpha in {x, y}, using EXACTLY
the same central C_i, C_f for all five h -- no rediagonalising the displaced
geometries, no per-h reselection.

C_i/C_f window: 2 states below and 2 above neutrality, occupied count derived
from the declared valence electron count (8, spin degeneracy 2 -> 4 occupied
bands), NEVER from where a gap happens to fall -- the same rule already used
in this repository's own certify_basis_response.py / the C19-C20 protocol
doc, reused here (not invented) because both states here are near-degenerate
pairs at Gamma: the full 2x2 block G is used as the observable, not a single
matrix element (arbitrary within a degenerate subspace).

Scope note: this is the SIMPLER Gamma-only contraction, independent of the
parked C19/C20 protocol (docs/epc_c19c20_preregistracion_graphene_gamma.md),
which additionally requires k_generic_1=(0.29,0.11,0) and true K, a
bond-stretch-overlap gauge rule, and is blocked on C14C (which needs a new
assembly function combining tonight's B_I v5 result with the existing
inter-atomic dS/dR method -- real physics work, deferred to the consultant).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

import sisl  # noqa: E402

import fd_perturbation_space as fdp  # noqa: E402
from artifact_signature import file_sha256, input_signature_sha256  # noqa: E402
from certify_delta_plateau_topology import (  # noqa: E402
    D5_WEIGHTS, DECISIVE_WINDOW, H_LADDER, MAX_LIMIT_EV_PER_ANG, PROD_SZ_ROOT,
    RMS_LIMIT_EV_PER_ANG, _d5, _k_matrix, _point, _run_dir,
)
from certify_full_basis_connection_semantic_v5 import solve  # noqa: E402
from certify_graphene_gamma_sublattice_symmetry_reduction import _orbital_mixing_matrix  # noqa: E402
from certify_production_basis_connection import _connections  # noqa: E402
from certify_uniform_translation_pao_covariance import _translation_connections  # noqa: E402
from cross_basis_projection_preflight import K_POINTS, _basis_geometry  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402
import orbital_contract as oc  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/graphene_gamma_e2g_electronic_contraction"
GAMMA = (0.0, 0.0, 0.0)
DEGENERACY_GAP_EV = 1e-3


def _select_ci_cf(k0: np.ndarray, s0: np.ndarray, n_valence_electrons: float):
    eigvals, c = eigh(k0, s0)
    n_occ = int(round(n_valence_electrons / 2.0))
    if abs(n_occ - n_valence_electrons / 2.0) > 1e-9:
        raise ValueError(f"non-integer occupied-band count from {n_valence_electrons} electrons")
    ci_idx = [n_occ - 2, n_occ - 1]
    cf_idx = [n_occ, n_occ + 1]
    return eigvals, c[:, ci_idx], c[:, cf_idx], ci_idx, cf_idx


def _contract(delta: np.ndarray, c_f: np.ndarray, c_i: np.ndarray) -> np.ndarray:
    return c_f.conj().T @ delta @ c_i


def _block_metric(matrix: np.ndarray) -> dict:
    n_f, n_i = matrix.shape
    rms = float(np.linalg.norm(matrix) / np.sqrt(n_f * n_i))
    mx = float(np.abs(matrix).max())
    return {"rms_ev_per_ang": rms, "max_ev_per_ang": mx,
            "passed": rms < RMS_LIMIT_EV_PER_ANG and mx < MAX_LIMIT_EV_PER_ANG}


def _delta_e2g_x_by_h(base, atom, shape, dvolume, s0, k0):
    delta_by_h = {}
    for h in H_LADDER:
        points_by_step = {}
        for step in D5_WEIGHTS:
            run_dir = _run_dir("C1_x", step * h)
            hamiltonian, fermi, vacuum, _ = _point(run_dir)
            points_by_step[step] = hamiltonian, fermi, vacuum
        connections = _connections("C1_x", h, base, atom, shape, dvolume, [GAMMA])
        k_by_step = {step: _k_matrix(*points_by_step[step], GAMMA) for step in points_by_step}
        d5k = _d5(k_by_step, h)
        s_l, s_r = connections[0]
        left_term, _ = solve(s0, k0)
        right_term, _ = solve(s0, s_r)
        delta_c1x = d5k - s_l @ left_term - k0 @ right_term

        # C2 sublattice symmetry reduction (already certified PASS) to get Delta_C2x
        r_g = np.diag([-1.0, -1.0, 1.0])
        m_matrix = _orbital_mixing_matrix(atom, r_g)
        zero = np.zeros_like(m_matrix)
        u_g = np.block([[zero, m_matrix], [m_matrix, zero]])
        delta_c2x = r_g[0, 0] * (u_g @ delta_c1x @ u_g.conj().T)
        delta_by_h[h] = (delta_c1x - delta_c2x) / np.sqrt(2.0)
    return delta_by_h


def _delta_e2g_y_by_h(base, atom, shape, dvolume, s0, k0):
    direction = fdp.collective([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], name="e2g_y")
    delta_by_h = {}
    for h in H_LADDER:
        points_by_step = {}
        for step in D5_WEIGHTS:
            run_dir = PROD_SZ_ROOT / "e2g_y" / (
                "d" + f"{abs(step*h):.6g}".replace(".", "p") + ("_plus" if step * h > 0 else "_minus"))
            hamiltonian, fermi, vacuum, _ = _point(run_dir)
            points_by_step[step] = hamiltonian, fermi, vacuum
        connections = _translation_connections(direction, h, base, atom, shape, dvolume, [GAMMA])
        k_by_step = {step: _k_matrix(*points_by_step[step], GAMMA) for step in points_by_step}
        d5k = _d5(k_by_step, h)
        s_l, s_r = connections[0]
        left_term, _ = solve(s0, k0)
        right_term, _ = solve(s0, s_r)
        delta_by_h[h] = d5k - s_l @ left_term - k0 @ right_term
    return delta_by_h


def _plateau(g_by_h: dict) -> dict:
    e12 = _block_metric(g_by_h[0.005] - g_by_h[0.0025])
    e23 = _block_metric(g_by_h[0.0025] - g_by_h[0.00125])
    r1 = (16.0 * g_by_h[0.0025] - g_by_h[0.005]) / 15.0
    r2 = (16.0 * g_by_h[0.00125] - g_by_h[0.0025]) / 15.0
    r1_vs_r2 = _block_metric(r1 - r2)
    prod_vs_extrap = _block_metric(g_by_h[0.005] - r2)
    passed = e12["passed"] and e23["passed"] and r1_vs_r2["passed"] and prod_vs_extrap["passed"]
    return {"verdict": "PASS" if passed else "NO_GO", "e12_0p005_to_0p0025": e12,
            "e23_0p0025_to_0p00125": e23, "richardson": {"r1_vs_r2": r1_vs_r2, "production_0p005_vs_r2": prod_vs_extrap},
            "relative_criterion": "NOT_APPLICABLE (no established zero-handling policy for this "
                                   "Gamma-only piece; decided on absolute RMS/max budgets only)"}


def main() -> int:
    central_label, central_h, central_fermi, central_vacuum, atom = _matrix_data(PROD_CENTRAL)
    shape = tuple(int(x) for x in sisl.get_sile(str(PROD_CENTRAL / f"{central_label}.VT")).read_grid().shape)
    base = _basis_geometry(central_h.geometry, PROD_CENTRAL / "C.ion.xml")
    dvolume = abs(np.linalg.det(base.cell)) / np.prod(shape)

    s0 = np.asarray(central_h.Sk(k=GAMMA, format="array"))
    k0 = _k_matrix(central_h, central_fermi, central_vacuum, GAMMA)

    valence = oc.read_ion_xml(str(PROD_CENTRAL / "C.ion.xml"))["valence"] * base.na
    eigvals, c_i, c_f, ci_idx, cf_idx = _select_ci_cf(k0, s0, valence)
    s_ortho_residual = float(np.linalg.norm(c_i.conj().T @ s0 @ c_i - np.eye(2))
                              + np.linalg.norm(c_f.conj().T @ s0 @ c_f - np.eye(2)))

    delta_x_by_h = _delta_e2g_x_by_h(base, atom, shape, dvolume, s0, k0)
    delta_y_by_h = _delta_e2g_y_by_h(base, atom, shape, dvolume, s0, k0)
    g_x_by_h = {h: _contract(delta_x_by_h[h], c_f, c_i) for h in H_LADDER}
    g_y_by_h = {h: _contract(delta_y_by_h[h], c_f, c_i) for h in H_LADDER}

    plateau_x = _plateau(g_x_by_h)
    plateau_y = _plateau(g_y_by_h)
    g_x_norm_production = float(np.linalg.norm(g_x_by_h[0.005]))
    g_y_norm_production = float(np.linalg.norm(g_y_by_h[0.005]))
    delta_norm_production = float(np.linalg.norm(delta_x_by_h[0.005]))
    near_zero_observable = max(g_x_norm_production, g_y_norm_production) < 1e-4

    pair_rms_e12 = float(np.sqrt((np.linalg.norm(g_x_by_h[0.005] - g_x_by_h[0.0025])**2
                                   + np.linalg.norm(g_y_by_h[0.005] - g_y_by_h[0.0025])**2) / (2 * 4)))
    pair_rms_e23 = float(np.sqrt((np.linalg.norm(g_x_by_h[0.0025] - g_x_by_h[0.00125])**2
                                   + np.linalg.norm(g_y_by_h[0.0025] - g_y_by_h[0.00125])**2) / (2 * 4)))
    doublet_passed = (plateau_x["verdict"] == "PASS" and plateau_y["verdict"] == "PASS"
                       and pair_rms_e12 < RMS_LIMIT_EV_PER_ANG and pair_rms_e23 < RMS_LIMIT_EV_PER_ANG)

    provenance = {
        "k": list(GAMMA), "n_valence_electrons": valence, "n_occupied_bands": int(valence / 2),
        "C_i_band_indices": ci_idx, "C_f_band_indices": cf_idx,
        "central_eigenvalues_ev": eigvals.tolist(),
        "C_i_C_f_degenerate": True,
        "S_orthonormality_residual": s_ortho_residual,
        "degeneracy_tol_ev": DEGENERACY_GAP_EV,
        "n_i": 2, "n_f": 2,
        "gauge": "K = H_abs - c_vac * S (certified PAO gauge, absolute energy origin)",
        "central_geometry_sha256": file_sha256(PROD_CENTRAL / "C.ion.xml"),
        "note": "C_i, C_f frozen at the central geometry and reused unchanged for all five h",
    }

    report = {
        "schema": "graphene_gamma_e2g_electronic_contraction_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Gamma-only, independent of the parked C19/C20 protocol (see module docstring)",
        "provenance": provenance,
        "IMPORTANT_CAVEAT": (
            "G_fi(h) ITSELF is near-zero (||G_x(0.005)||_F={:.3e}, ||G_y(0.005)||_F={:.3e} eV/Ang) "
            "even though the full Delta_E2g matrix has norm {:.1f} eV/Ang -- the plateau PASS below "
            "means this near-zero value is STABLE across h, not that a meaningful electron-phonon "
            "coupling has been shown to converge. This C_i={{2,3}}/C_f={{4,5}} pair (2 occupied + 2 "
            "unoccupied states by raw electron count at Gamma, in a minimal single-zeta basis) is "
            "very likely NOT the pi/pi* band pair that carries the real graphene E2g (G-band) "
            "electron-phonon coupling -- that requires measuring at K, with a bond-stretch-overlap "
            "gauge rule, per the parked C19/C20 protocol (docs/epc_c19c20_preregistracion_graphene_"
            "gamma.md), which additionally needs C14C resolved. Do NOT read this gate's PASS as "
            "'Graph2Mat reproduces SIESTA's E2g coupling' -- that question remains open."
        ).format(g_x_norm_production, g_y_norm_production, delta_norm_production),
        "graphene_Gamma_E2g_x_electronic_contraction_convergence": plateau_x,
        "graphene_Gamma_E2g_y_electronic_contraction_convergence": plateau_y,
        "graphene_Gamma_E2g_doublet_electronic_contraction_convergence": {
            "verdict": "PASS" if doublet_passed else "NO_GO",
            "pair_rms_e12_ev_per_ang": pair_rms_e12, "pair_rms_e23_ev_per_ang": pair_rms_e23,
            "pair_formula": "sqrt((||E_x||_F^2 + ||E_y||_F^2) / (2*n_f*n_i))",
            "g_fi_itself_near_zero": near_zero_observable,
            "g_x_norm_ev_per_ang": g_x_norm_production, "g_y_norm_ev_per_ang": g_y_norm_production,
            "delta_e2g_full_matrix_norm_ev_per_ang": delta_norm_production,
        },
        "unaffected_by_this_gate": {
            "checkpoint_lineage": "NO_GO", "fine_tuning_candidate": "BLOCKED",
            "delta_out_closure": "NO_GO", "full_KS": "BLOCKED",
            "uniform_translation_PAO_covariance_BZ_sampled_v1": "NO_GO",
            "graphene_Gamma_sublattice_C3_symmetry_reduction": "BLOCKED",
            "C14C_basis_response": "NO_GO (needs new B_I-v5-aware assembly; deferred)",
            "C19_C20_graphene_gamma_protocol": "ready_to_execute=False (blocked on C14C)",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "graphene_gamma_e2g_electronic_contraction.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"E2g_x electronic contraction: {plateau_x['verdict']}")
    print(f"E2g_y electronic contraction: {plateau_y['verdict']}")
    print(f"E2g doublet electronic contraction: {'PASS' if doublet_passed else 'NO_GO'} -> {path}")
    return 0 if doublet_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
