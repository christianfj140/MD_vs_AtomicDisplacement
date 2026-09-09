#!/usr/bin/env python3
"""graphene_Gamma_E2g_doublet_convergence, via an exact C3 symmetry reduction.

No new SIESTA. Derives, from the real PROD-SZ central geometry, the C3
rotation (120 degrees about z through a hexagon-center lattice point) and its
orbital representation U_C3, numerically via the same real-space grid
quadrature used for the C2 (sublattice-exchange) reduction -- never assuming a
p_x/p_y sign or rotation-block convention by hand.

Mandatory preflight: U_C3^dagger S U_C3 =~ S, U_C3^dagger K U_C3 =~ K,
U_C3^3 =~ I, plus a check that SIESTA's own real-space mesh is itself exactly
C3-covariant (it is: worst deviation ~1e-9, verified). The overlap check S
passes to ~1e-9 relative -- essentially exact, as expected for a purely
geometric quantity. The Hamiltonian check K does NOT: the residual
(~8.9e-5 eV, ~1.5e-6 relative to ||K||) is identical across three quadrature
spacings (0.04, 0.02, 0.01 Ang) and the real-space mesh is independently
confirmed C3-covariant to ~1e-9, so this is not a quadrature or grid artifact
of this script -- but it is ~500x larger than the analogous C2 residual
(~1.7e-7 eV), which is unexplained. Per an explicit fail-closed decision, this
gate does NOT loosen the tolerance to force a PASS: the K preflight fails,
so graphene_Gamma_sublattice_C3_symmetry_reduction and
graphene_Gamma_E2g_doublet_convergence are both reported BLOCKED, with the
real residual documented for follow-up (candidates: k-mesh C3-covariance,
SCF/DM.Tolerance-level asymmetry, or something else not yet identified).
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

from certify_delta_plateau_topology import _k_matrix  # noqa: E402
from certify_graphene_gamma_sublattice_symmetry_reduction import _grid_covariance  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/graphene_gamma_e2g_doublet_convergence"
GAMMA = (0.0, 0.0, 0.0)
QUADRATURE_SPACING_ANG = 0.04
INVOLUTION_TOL = 1e-8
PREFLIGHT_S_REL_TOL = 1e-8
PREFLIGHT_K_EV_TOL = 1e-6
GRID_COVARIANCE_TOL = 1e-6


def _derive_c3(geometry: sisl.Geometry):
    """C3 about the origin (a hexagon-center lattice point). Verified, not assumed:
    each atom must map onto SOME periodic image of itself (not the other atom)."""
    xyz = geometry.xyz
    theta = 2 * np.pi / 3
    r_g = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])
    a1, a2 = geometry.cell[0][:2], geometry.cell[1][:2]
    a2mat = np.column_stack([a1, a2])
    atom_map = {}
    for i, pos in enumerate(xyz):
        mapped = r_g @ pos
        found = None
        for j, target in enumerate(xyz):
            diff = mapped - target
            n = np.linalg.solve(a2mat, diff[:2])
            n_int = np.round(n)
            if np.abs(n - n_int).max() < 1e-6 and abs(diff[2]) < 1e-9:
                found = j
                break
        if found is None:
            raise ValueError(f"R_120 about the origin does not map atom {i} onto any atom "
                              "(mod lattice); this candidate axis is not a symmetry")
        atom_map[i] = found
    return r_g, atom_map


def _orbital_mixing_matrix(atom, r_g: np.ndarray, spacing: float = QUADRATURE_SPACING_ANG) -> np.ndarray:
    extent = max(orbital.R for orbital in atom.orbitals) + 0.1
    line = np.arange(-extent, extent + spacing / 2, spacing)
    xx, yy, zz = np.meshgrid(line, line, line, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    del xx, yy, zz
    phi = np.column_stack([orbital.psi(points) for orbital in atom.orbitals])
    r_g_inv = np.linalg.inv(r_g)
    phi_rot = np.column_stack([orbital.psi(points @ r_g_inv.T) for orbital in atom.orbitals])
    return phi.T @ phi_rot * spacing**3


def main() -> int:
    central_label, central_h, central_fermi, central_vacuum, atom = _matrix_data(PROD_CENTRAL)
    geometry = central_h.geometry

    r_g, atom_map = _derive_c3(geometry)
    if atom_map != {0: 0, 1: 1}:
        report = {
            "schema": "graphene_gamma_e2g_doublet_convergence_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "graphene_Gamma_sublattice_C3_symmetry_reduction": "BLOCKED",
            "graphene_Gamma_E2g_doublet_convergence": "BLOCKED",
            "reason": f"unexpected atom permutation under C3: {atom_map}",
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = OUTPUT / "graphene_gamma_e2g_doublet_convergence.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"graphene_Gamma_E2g_doublet_convergence: BLOCKED -> {path}")
        return 1

    m_matrix = _orbital_mixing_matrix(atom, r_g)
    zero = np.zeros_like(m_matrix)
    u_c3 = np.block([[m_matrix, zero], [zero, m_matrix]])

    involution_residual = float(np.linalg.norm(u_c3 @ u_c3 @ u_c3 - np.eye(u_c3.shape[0])))

    s0 = np.asarray(central_h.Sk(k=GAMMA, format="array"))
    k0 = _k_matrix(central_h, central_fermi, central_vacuum, GAMMA)
    s_rel = float(np.linalg.norm(u_c3.conj().T @ s0 @ u_c3 - s0) / np.linalg.norm(s0))
    k_abs = float(np.linalg.norm(u_c3.conj().T @ k0 @ u_c3 - k0))

    vt_grid = sisl.get_sile(str(PROD_CENTRAL / f"{central_label}.VT")).read_grid()
    grid_covariance_worst = _grid_covariance(vt_grid, r_g, np.zeros(3))

    # D_E2g: action of C3 on the 2D modal space spanned by e_x, e_y, derived (not
    # assumed) from how R_g mixes the x,y Cartesian components (R_g is block-diagonal
    # with the same 2x2 rotation acting on x,y for every atom, since atom_map is the
    # identity here -- no atom exchange, only a Cartesian-component rotation).
    d_e2g = r_g[:2, :2]
    d_unitary_residual = float(np.linalg.norm(d_e2g.conj().T @ d_e2g - np.eye(2)))
    d_cubed_residual = float(np.linalg.norm(np.linalg.matrix_power(d_e2g, 3) - np.eye(2)))

    preflight = {
        "R_g_120deg_about_origin": r_g.tolist(), "atom_map": atom_map,
        "orbital_mixing_matrix": m_matrix.tolist(),
        "involution_residual_U3": involution_residual, "involution_tolerance": INVOLUTION_TOL,
        "U_dagger_S_U_minus_S_relative": s_rel, "S_tolerance_relative": PREFLIGHT_S_REL_TOL,
        "U_dagger_K_U_minus_K_ev": k_abs, "K_tolerance_ev": PREFLIGHT_K_EV_TOL,
        "C3_siesta_grid_covariance": {
            "worst_fractional_index_deviation": grid_covariance_worst,
            "tolerance": GRID_COVARIANCE_TOL, "passed": grid_covariance_worst < GRID_COVARIANCE_TOL,
        },
        "D_E2g_matrix": d_e2g.tolist(),
        "D_E2g_unitary_residual": d_unitary_residual, "D_E2g_cubed_residual": d_cubed_residual,
        "k_residual_quadrature_convergence_check": {
            "spacings_ang": [0.04, 0.02, 0.01],
            "k_residual_ev_by_spacing": [8.910893650487567e-05, 8.910893650487567e-05, 8.910893650487567e-05],
            "note": "identical across 3x finer quadrature; ruled out as a quadrature artifact. "
                    "SIESTA's own real-space mesh independently confirmed C3-covariant to ~1e-9, "
                    "ruling out a grid-covariance origin too. Cause of the K residual (~500x the "
                    "analogous C2 residual) is not yet identified.",
        },
    }
    preflight_passed = (involution_residual < INVOLUTION_TOL and s_rel < PREFLIGHT_S_REL_TOL
                         and k_abs < PREFLIGHT_K_EV_TOL and grid_covariance_worst < GRID_COVARIANCE_TOL)

    report = {
        "schema": "graphene_gamma_e2g_doublet_convergence_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graphene_Gamma_sublattice_C3_symmetry_reduction": "PASS" if preflight_passed else "BLOCKED",
        "graphene_Gamma_E2g_doublet_convergence": "PASS" if preflight_passed else "BLOCKED",
        "preflight": preflight,
        "fail_closed_decision": (
            "K preflight residual (8.911e-05 eV) exceeds the tolerance (1e-6 eV) used for the "
            "analogous, much cleaner C2 case (1.67e-07 eV). Explicitly NOT loosened post-hoc to "
            "force a PASS. Delta_y was not reconstructed and no doublet-span closure checks were "
            "run, since they depend on a certified U_C3."
        ) if not preflight_passed else None,
        "next_step_if_doublet_needed": (
            "preregister and run the direct E2g_x campaign (12 new SIESTA runs, Q=+-[0.020,0.010,"
            "0.005,0.0025,0.00125,0.000625] Ang with C1=+Q/sqrt(2), C2=-Q/sqrt(2)) for a second, "
            "independent modal component -- or investigate the K-preflight residual further before "
            "reattempting this symmetry shortcut"
        ),
        "unaffected_by_this_gate": {
            "checkpoint_lineage": "NO_GO", "fine_tuning_candidate": "BLOCKED",
            "delta_out_closure": "NO_GO", "full_KS": "BLOCKED",
            "graphene_Gamma_E2g_x_delta_convergence": "PASS",
            "graphene_Gamma_sublattice_symmetry_reduction": "PASS",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "graphene_gamma_e2g_doublet_convergence.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"graphene_Gamma_E2g_doublet_convergence: "
          f"{report['graphene_Gamma_E2g_doublet_convergence']} -> {path}")
    return 0 if preflight_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
