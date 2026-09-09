#!/usr/bin/env python3
"""Independent PROD-SZ ``B_I`` witness from Fourier-Bessel/Slater-Koster integrals."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from certify_b_i_independent_two_center import (  # noqa: E402
    B1_LIMIT_EV_PER_ANG, B3_LIMIT_EV_PER_ANG, ONLYS_MATRICES, Basis,
    _electronic, b_i_from_connection, enumerate_images, gate_b3,
    prolate_b_i_blocks, read_radials,
)
from certify_support_exact_connection import _selector  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from onlys_semantic_preflight import _dense_k_points  # noqa: E402
from run_displaced_projection_sentinel import DIRECTIONS  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/b_i_fourier_bessel_v4"
CONFIGS = {"coarse": (600, 80.0, 3201), "fine": (800, 120.0, 4801)}
MONOCENTRIC_TOL = 1e-10
ANTIHERMITIAN_TOL = 1e-10


class FourierBesselSP:
    """The four PROD-SZ s/p Slater-Koster integrals and analytic derivatives."""

    vectors = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0],
                        [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])

    def __init__(self, basis: Basis, n_r: int, q_max: float, n_q: int, normalize=False):
        x, w = np.polynomial.legendre.leggauss(n_r)
        r = 0.5 * basis.rc_max * (x + 1.0)
        weights = 0.5 * basis.rc_max * w * r**2
        self.q = np.linspace(0.0, q_max, n_q)
        self.f = {}
        for l in (0, 1):
            radial = basis.radial(l, r)
            if normalize:
                radial = radial / np.sqrt(np.sum(weights * radial**2))
            self.f[l] = spherical_jn(l, np.outer(self.q, r)) @ (weights * radial)

    def radial_integrals(self, distance: float) -> tuple[np.ndarray, np.ndarray]:
        q, z = self.q, self.q * distance
        j = [spherical_jn(l, z) for l in range(3)]
        jd = [spherical_jn(l, z, derivative=True) for l in range(3)]
        q2, q3 = q**2, q**3
        f0, f1 = self.f[0], self.f[1]
        kernels = (j[0], j[1], j[0] - 2 * j[2], j[0] + j[2])
        kernels_d = (jd[0], jd[1], jd[0] - 2 * jd[2], jd[0] + jd[2])
        factors = (2 / np.pi, -2 * np.sqrt(3) / np.pi, 2 / np.pi, 2 / np.pi)
        products = (f0 * f0, f0 * f1, f1 * f1, f1 * f1)
        values = np.array([
            factor * simpson(q2 * product * kernel, x=q)
            for factor, product, kernel in zip(factors, products, kernels)
        ])
        derivatives = np.array([
            factor * simpson(q3 * product * kernel, x=q)
            for factor, product, kernel in zip(factors, products, kernels_d)
        ])
        return values, derivatives  # ss, sp, pp_sigma, pp_pi

    def overlap(self, separation: np.ndarray) -> np.ndarray:
        distance = float(np.linalg.norm(separation))
        values, _ = self.radial_integrals(distance)
        ss, sp, sigma, pi = values
        result = np.zeros((4, 4))
        if distance == 0.0:
            result[0, 0], result[1:, 1:] = ss, pi * np.eye(3)
            return result
        n = separation / distance
        projected = self.vectors[1:] @ n
        result[0, 0] = ss
        result[0, 1:] = sp * projected
        result[1:, 0] = -sp * projected
        result[1:, 1:] = (
            pi * (self.vectors[1:] @ self.vectors[1:].T)
            + (sigma - pi) * np.outer(projected, projected)
        )
        return result

    def derivative(self, separation: np.ndarray, axis: int) -> np.ndarray:
        distance = float(np.linalg.norm(separation))
        values, derivatives = self.radial_integrals(distance)
        _, sp, sigma, pi = values
        ss_d, sp_d, sigma_d, pi_d = derivatives
        result = np.zeros((4, 4))
        if distance == 0.0:
            components = self.vectors[1:, axis]
            result[0, 1:] = sp_d * components
            result[1:, 0] = -sp_d * components
            return result
        n, n_u = separation / distance, separation[axis] / distance
        projected = self.vectors[1:] @ n
        transverse = (self.vectors[1:, axis] - projected * n_u) / distance
        result[0, 0] = n_u * ss_d
        result[0, 1:] = transverse * sp + projected * n_u * sp_d
        result[1:, 0] = -(transverse * sp + projected * n_u * sp_d)
        uv = self.vectors[1:] @ self.vectors[1:].T
        result[1:, 1:] = (
            n_u * pi_d * uv
            + n_u * (sigma_d - pi_d) * np.outer(projected, projected)
            + (sigma - pi) * (
                np.outer(transverse, projected) + np.outer(projected, transverse)
            )
        )
        return result


def connection_blocks(engine, images, geometry, direction):
    axis, no, na = DIRECTIONS[direction], 4, geometry.na
    result = {}
    for image in images:
        matrix = np.zeros((na * no, na * no))
        matrix[:no, :no] = engine.derivative(image["vector"], axis)
        result[tuple(image["index"])] = matrix
    return result


def main() -> int:
    _, central, fermi, vacuum, _ = _matrix_data(PROD_CENTRAL)
    geometry = central.geometry
    radials, _ = read_radials(PROD_CENTRAL / "C.ion.xml")
    basis = Basis(radials, [0, 1, 1, 1])
    images = enumerate_images(basis, np.asarray(geometry.cell), geometry.xyz)
    mask, dense = _selector(geometry), _dense_k_points()
    engines = {name: FourierBesselSP(basis, *config) for name, config in CONFIGS.items()}

    identity_error = float(np.linalg.norm(engines["fine"].overlap(np.zeros(3)) - np.eye(4), 2))
    cases, saved = [], {}
    for direction in sorted(DIRECTIONS):
        blocks = {name: connection_blocks(engine, images, geometry, direction)
                  for name, engine in engines.items()}
        home = blocks["fine"][(0, 0, 0)][:4, :4]
        anti = float(np.linalg.norm(home + home.T) / max(np.linalg.norm(home), np.finfo(float).tiny))
        worst, worst_k, residuals = 0.0, None, []
        for k in dense:
            s0 = np.asarray(central.Sk(k=k, format="array"))
            k0 = np.asarray(central.Hk(k=k, format="array")) + (fermi - vacuum) * s0
            delta = (b_i_from_connection(blocks["fine"], mask, k)
                     - b_i_from_connection(blocks["coarse"], mask, k))
            error, residual = _electronic(delta, s0, k0)
            residuals.append(residual)
            if error > worst:
                worst, worst_k = error, list(map(float, k))
        a_pass = identity_error < MONOCENTRIC_TOL and anti < ANTIHERMITIAN_TOL
        internal_pass = worst < B1_LIMIT_EV_PER_ANG

        prolate = prolate_b_i_blocks(basis, images, geometry.xyz, direction, 0.00125, 400)
        prolate_worst = 0.0
        for k in dense:
            s0 = np.asarray(central.Sk(k=k, format="array"))
            k0 = np.asarray(central.Hk(k=k, format="array")) + (fermi - vacuum) * s0
            error, _ = _electronic(
                b_i_from_connection(blocks["fine"], mask, k)
                - b_i_from_connection(prolate, mask, k), s0, k0,
            )
            prolate_worst = max(prolate_worst, error)

        if a_pass and internal_pass:
            comparison, matrices = gate_b3(
                blocks["fine"], mask, direction, central, fermi, vacuum,
                np.load(ONLYS_MATRICES), dense,
            )
            saved.update(matrices)
        else:
            comparison = {
                "verdict": "BLOCKED",
                "reason": "V4-A monocentric or V4-B internal convergence failed",
            }
        cases.append({
            "direction": direction,
            "V4_A": {"monocentric_overlap_spectral_error": identity_error,
                       "antihermiticity_relative": anti,
                       "verdict": "PASS" if a_pass else "NO_GO"},
            "V4_B": {"verdict": "PASS" if internal_pass else "NO_GO",
                       "worst_400k_propagated_ev_per_ang": worst,
                       "worst_k_reduced": worst_k, "threshold_ev_per_ang": B1_LIMIT_EV_PER_ANG,
                       "max_solve_residual_relative": max(residuals)},
            "V4_C_diagnostic": {"FB_fine_vs_prolate400_worst_400k_ev_per_ang": prolate_worst,
                                  "gate": False},
            "V4_D_onlyS": comparison,
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    matrices = OUTPUT / "b_i_fourier_bessel_v4_matrices.npz"
    np.savez_compressed(matrices, **saved)
    passed = all(case[key]["verdict"] == "PASS" for case in cases
                 for key in ("V4_A", "V4_B", "V4_D_onlyS"))
    report = {
        "schema": "b_i_fourier_bessel_gaunt_v4",
        "gate": "B_I_fourier_bessel_gaunt_v4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO",
        "configs": {name: {"n_r": v[0], "q_max_inv_ang": v[1], "n_q": v[2]}
                    for name, v in CONFIGS.items()},
        "thresholds": {"monocentric": MONOCENTRIC_TOL,
                       "antihermitian": ANTIHERMITIAN_TOL,
                       "internal_ev_per_ang": B1_LIMIT_EV_PER_ANG,
                       "onlyS_ev_per_ang": B3_LIMIT_EV_PER_ANG},
        "method": "s/p-only Fourier-Bessel transforms plus analytic Slater-Koster derivatives",
        "independence": {"centre_FD": False, "prolate_used_in_construction": False,
                         "onlyS_used_in_construction": False, "D_S_used": False,
                         "TSHS_used_as_overlap_truth": False},
        "cases": cases,
        "historical_NO_GO_preserved": ["v1", "analytic_v2", "prolate_v3", "prolate_400-320"],
        "matrix_artifact": str(matrices), "matrix_artifact_sha256": file_sha256(matrices),
        "claim_scope": "PROD-SZ B_I only; no Delta_out/full-KS promotion unless this gate passes",
    }
    path = OUTPUT / "b_i_fourier_bessel_v4.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"B_I Fourier-Bessel v4: {report['verdict']} -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
