#!/usr/bin/env python3
"""graphene_Gamma_sublattice_symmetry_reduction + graphene_Gamma_E2g_x_delta_convergence.

No new SIESTA. Derives, from the real PROD-SZ central geometry, the exact
spatial operation g that exchanges the two carbon atoms (C1 <-> C2), builds
its orbital representation U_g NUMERICALLY via real-space grid quadrature of
the actual PAO wavefunctions (never assuming a p_x/p_y/p_z sign convention by
hand), and requires the mandatory preflight U_g^dagger S U_g =~ S,
U_g^dagger K U_g =~ K at Gamma before using it for anything. If that preflight
fails, the whole gate is BLOCKED -- Delta_C2x is never inferred by hand (e.g.
never assumed equal to -Delta_C1x, and the acoustic-sum-rule identity
Delta_C1x + Delta_C2x = 0 is explicitly NOT used, since the full PAO-covariant
response matrix is not required to vanish under uniform translation).

Geometry of g (verified, not assumed): a proper 180-degree rotation about the
z axis (R_g = diag(-1,-1,1)) through the C1-C2 bond midpoint, composed with
the lattice translation a1+a2. Because R_g is diagonal, the general tensorial
transformation law Delta_{sigma(I),u} = sum_v R_g[u,v] U_g Delta_{I,v} U_g^dagger
collapses to Delta_{C2,x} = R_g[x,x] U_g Delta_{C1,x} U_g^dagger = -U_g Delta_{C1,x} U_g^dagger
-- no Delta_{C1,y} is needed for the x-component.

Delta_E2g_x(h) = (Delta_C1x(h) - Delta_C2x(h)) / sqrt(2), for the same five h
already certified by delta_plateau_topology_certification, adjudicated with
the identical plateau/Richardson gate (same thresholds, same decisive
window) -- reusing certify_delta_plateau_topology's own machinery, not a
reimplementation.

graphene_Gamma_E2g_doublet_convergence is NOT attempted here (separate step).
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

from artifact_signature import file_sha256, input_signature_sha256  # noqa: E402
from certify_delta_plateau_topology import (  # noqa: E402
    D5_WEIGHTS, DECISIVE_WINDOW, H_LADDER, MAX_LIMIT_EV_PER_ANG, REL_LIMIT,
    RMS_LIMIT_EV_PER_ANG, _compare, _connections, _d5, _k_matrix, _point, _run_dir,
)
from certify_full_basis_connection_semantic_v5 import solve  # noqa: E402
from cross_basis_projection_preflight import K_POINTS, _basis_geometry  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_nested_basis_ladder import _rms  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/graphene_gamma_sublattice_symmetry_reduction"
GATE1_PATH = (
    REPO_ROOT / "Comparison/results/epc/delta_plateau_topology_certification"
    / "delta_plateau_topology_certification.json"
)
GAMMA = (0.0, 0.0, 0.0)
QUADRATURE_SPACING_ANG = 0.04
PREFLIGHT_S_REL_TOL = 1e-8
PREFLIGHT_K_EV_TOL = 1e-6
INVOLUTION_TOL = 1e-8
GRID_COVARIANCE_TOL = 1e-6


def _derive_g(geometry: sisl.Geometry):
    """Find, and numerically verify against the real positions, the C2 operation
    exchanging the two atoms. Returns (R_g, atom_map) or raises if it cannot be
    verified to hold exactly."""
    xyz = geometry.xyz
    if geometry.na != 2:
        raise ValueError(f"expected a 2-atom cell, got na={geometry.na}")
    m = (xyz[0] + xyz[1]) / 2.0
    r_g = np.diag([-1.0, -1.0, 1.0])
    mapped0 = r_g @ (xyz[0] - m) + m
    mapped1 = r_g @ (xyz[1] - m) + m
    err0 = float(np.linalg.norm(mapped0 - xyz[1]))
    err1 = float(np.linalg.norm(mapped1 - xyz[0]))
    if err0 > 1e-6 or err1 > 1e-6:
        raise ValueError(f"C2-about-bond-midpoint does not map atom0<->atom1 exactly "
                          f"(errors {err0:.3e}, {err1:.3e} Ang)")
    t = m - r_g @ m
    a1, a2 = geometry.cell[0][:2], geometry.cell[1][:2]
    n = np.linalg.solve(np.column_stack([a1, a2]), t[:2])
    n_int = np.round(n)
    if np.abs(n - n_int).max() > 1e-6 or abs(t[2]) > 1e-9:
        raise ValueError(f"translation part is not an exact lattice vector: n={n}, t_z={t[2]}")
    return r_g, {0: 1, 1: 0}, (int(n_int[0]), int(n_int[1]))


def _grid_covariance(grid: sisl.Grid, r_g: np.ndarray, m: np.ndarray) -> float:
    """Worst deviation, in fractional-index units, between g(grid point) and the
    nearest point of the SAME real SIESTA mesh -- using the mesh's own real origin
    and dcell, not an assumed zero-offset convention (verified equal to it, but
    checked, not assumed)."""
    nx, ny, _ = grid.shape
    dcell = grid.dcell
    a1, a2 = grid.cell[0], grid.cell[1]
    icell_2d = np.linalg.inv(np.column_stack([a1[:2], a2[:2]]))
    worst = 0.0
    for i in range(nx):
        for j in range(ny):
            xyz = i * dcell[0] + j * dcell[1] + grid.origin
            transformed = r_g[:2, :2] @ (xyz[:2] - m[:2]) + m[:2]
            frac = icell_2d @ transformed
            idx_f = frac * [nx, ny]
            worst = max(worst, float(np.abs(idx_f - np.round(idx_f)).max()))
    return worst


def _orbital_mixing_matrix(atom, r_g: np.ndarray, spacing: float = QUADRATURE_SPACING_ANG) -> np.ndarray:
    """M[b,a] = <phi_b | phi_a o R_g^-1>, via real-space grid quadrature of the actual
    PAO wavefunctions -- never assumes a p_x/p_y/p_z sign convention by hand."""
    extent = max(orbital.R for orbital in atom.orbitals) + 0.1
    line = np.arange(-extent, extent + spacing / 2, spacing)
    xx, yy, zz = np.meshgrid(line, line, line, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    del xx, yy, zz
    phi = np.column_stack([orbital.psi(points) for orbital in atom.orbitals])
    phi_rot = np.column_stack([orbital.psi(points @ r_g.T) for orbital in atom.orbitals])
    return phi.T @ phi_rot * spacing**3


def main() -> int:
    central_label, central_h, central_fermi, central_vacuum, atom = _matrix_data(PROD_CENTRAL)
    geometry = central_h.geometry

    try:
        r_g, atom_map, lattice_translation = _derive_g(geometry)
    except ValueError as exc:
        report = {
            "schema": "graphene_gamma_sublattice_symmetry_reduction_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "graphene_Gamma_sublattice_symmetry_reduction": "BLOCKED",
            "graphene_Gamma_E2g_x_delta_convergence": "BLOCKED",
            "reason": f"could not derive the atom-exchange operation: {exc}",
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = OUTPUT / "graphene_gamma_sublattice_symmetry_reduction.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"graphene_Gamma_sublattice_symmetry_reduction: BLOCKED -> {path}")
        return 1

    m_matrix = _orbital_mixing_matrix(atom, r_g)
    zero = np.zeros_like(m_matrix)
    u_g = np.block([[zero, m_matrix], [m_matrix, zero]])

    involution_residual = float(np.linalg.norm(u_g @ u_g - np.eye(u_g.shape[0])))

    s0 = np.asarray(central_h.Sk(k=GAMMA, format="array"))
    k0 = _k_matrix(central_h, central_fermi, central_vacuum, GAMMA)
    s_residual = u_g.conj().T @ s0 @ u_g - s0
    k_residual = u_g.conj().T @ k0 @ u_g - k0
    s_rel = float(np.linalg.norm(s_residual) / np.linalg.norm(s0))
    k_abs = float(np.linalg.norm(k_residual))

    vt_grid = sisl.get_sile(str(PROD_CENTRAL / f"{central_label}.VT")).read_grid()
    m_cart = (geometry.xyz[0] + geometry.xyz[1]) / 2.0
    grid_covariance_worst = _grid_covariance(vt_grid, r_g, m_cart)

    preflight = {
        "R_g": r_g.tolist(), "atom_map": atom_map, "lattice_translation_n1_n2": list(lattice_translation),
        "orbital_mixing_matrix_diag": np.diag(m_matrix).tolist(),
        "involution_residual_M8x8": involution_residual, "involution_tolerance": INVOLUTION_TOL,
        "U_dagger_S_U_minus_S_relative": s_rel, "S_tolerance_relative": PREFLIGHT_S_REL_TOL,
        "U_dagger_K_U_minus_K_ev": k_abs, "K_tolerance_ev": PREFLIGHT_K_EV_TOL,
        "C2z_siesta_grid_covariance": {
            "grid_shape": list(vt_grid.shape), "grid_origin": vt_grid.origin.tolist(),
            "worst_fractional_index_deviation": grid_covariance_worst,
            "tolerance": GRID_COVARIANCE_TOL, "passed": grid_covariance_worst < GRID_COVARIANCE_TOL,
        },
    }
    preflight_passed = (involution_residual < INVOLUTION_TOL and s_rel < PREFLIGHT_S_REL_TOL
                         and k_abs < PREFLIGHT_K_EV_TOL and grid_covariance_worst < GRID_COVARIANCE_TOL)

    if not preflight_passed:
        report = {
            "schema": "graphene_gamma_sublattice_symmetry_reduction_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "graphene_Gamma_sublattice_symmetry_reduction": "BLOCKED",
            "graphene_Gamma_E2g_x_delta_convergence": "BLOCKED",
            "reason": "U_g preflight (S/K invariance or involution) did not pass",
            "preflight": preflight,
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = OUTPUT / "graphene_gamma_sublattice_symmetry_reduction.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"graphene_Gamma_sublattice_symmetry_reduction: BLOCKED (preflight failed) -> {path}")
        return 1

    # -- preflight PASS: reconstruct Delta_C1x(h) (Gate 1's own route, Gamma only)
    # and derive Delta_C2x(h) = R_g[x,x] * U_g Delta_C1x(h) U_g^dagger = -U_g ... U_g^dagger
    base = _basis_geometry(central_h.geometry, PROD_CENTRAL / "C.ion.xml")
    shape = tuple(int(x) for x in sisl.get_sile(str(PROD_CENTRAL / f"{central_label}.VT")).read_grid().shape)
    dvolume = abs(np.linalg.det(base.cell)) / np.prod(shape)
    r_g_xx = float(r_g[0, 0])

    delta_c1x_by_h, delta_e2g_by_h = {}, {}
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
        delta_c2x = r_g_xx * (u_g @ delta_c1x @ u_g.conj().T)
        delta_c1x_by_h[h] = delta_c1x
        delta_e2g_by_h[h] = (delta_c1x - delta_c2x) / np.sqrt(2.0)

    s0_by_k = [s0]
    e12 = _compare([delta_e2g_by_h[0.005]], [delta_e2g_by_h[0.0025]], s0_by_k)
    e23 = _compare([delta_e2g_by_h[0.0025]], [delta_e2g_by_h[0.00125]], s0_by_k)
    r1 = (16.0 * delta_e2g_by_h[0.0025] - delta_e2g_by_h[0.005]) / 15.0
    r2 = (16.0 * delta_e2g_by_h[0.00125] - delta_e2g_by_h[0.0025]) / 15.0
    richardson_stability = _compare([r1], [r2], s0_by_k)
    production_vs_extrapolated = _compare([delta_e2g_by_h[0.005]], [r2], s0_by_k)

    e2g_x_passed = (e12["passed"] and e23["passed"] and richardson_stability["passed"]
                     and production_vs_extrapolated["passed"])

    report = {
        "schema": "graphene_gamma_sublattice_symmetry_reduction_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graphene_Gamma_sublattice_symmetry_reduction": "PASS",
        "preflight": preflight,
        "e2g_basis": {
            "e_E2g_x": [[1 / np.sqrt(2), 0, 0], [-1 / np.sqrt(2), 0, 0]],
            "note": "geometric, Frobenius-normalized coordinate; hbar/(2 M omega) not applied here",
        },
        "graphene_Gamma_E2g_x_delta_convergence": {
            "verdict": "PASS" if e2g_x_passed else "NO_GO",
            "h_ladder_ang": list(H_LADDER), "decisive_window_ang": list(DECISIVE_WINDOW),
            "thresholds": {"rms_ev_per_ang": RMS_LIMIT_EV_PER_ANG, "relative": REL_LIMIT,
                            "max_ev_per_ang": MAX_LIMIT_EV_PER_ANG},
            "e12_0p005_to_0p0025": e12, "e23_0p0025_to_0p00125": e23,
            "richardson": {"r1_vs_r2": richardson_stability, "production_0p005_vs_r2": production_vs_extrapolated},
        },
        "graphene_Gamma_E2g_doublet_convergence": {
            "verdict": "NOT_ATTEMPTED",
            "reason": "separate step; requires demonstrating the E2g span via exact C3 symmetry",
        },
        "unaffected_by_this_gate": {
            "checkpoint_lineage": "NO_GO", "fine_tuning_candidate": "BLOCKED",
            "delta_out_closure": "NO_GO", "full_KS": "BLOCKED",
            "uniform_translation_PAO_covariance_Gamma": "PASS",
            "uniform_translation_PAO_covariance_BZ_sampled_v1": "NO_GO",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "graphene_gamma_sublattice_symmetry_reduction.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"graphene_Gamma_sublattice_symmetry_reduction: PASS -> {path}")
    print(f"graphene_Gamma_E2g_x_delta_convergence: {'PASS' if e2g_x_passed else 'NO_GO'}")
    return 0 if e2g_x_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
