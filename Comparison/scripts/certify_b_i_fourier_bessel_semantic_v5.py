#!/usr/bin/env python3
"""Semantic closure of PROD-SZ ``B_I`` with separated q-tail/normalisation errors."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from certify_b_i_fourier_bessel_v4 import (  # noqa: E402
    ANTIHERMITIAN_TOL, FourierBesselSP, connection_blocks,
)
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

OUTPUT = REPO_ROOT / "Comparison/results/epc/b_i_fourier_bessel_semantic_v5"
ENGINES = {
    "q120_raw": (800, 120.0, 4801, False),
    "q180_raw": (1000, 180.0, 7201, False),
    "q180_normalized": (1000, 180.0, 7201, True),
}


def direct_radial_norms(basis: Basis, order=1000) -> dict[int, float]:
    x, w = np.polynomial.legendre.leggauss(order)
    r = 0.5 * basis.rc_max * (x + 1.0)
    weights = 0.5 * basis.rc_max * w * r**2
    return {l: float(np.sum(weights * basis.radial(l, r) ** 2)) for l in (0, 1)}


def dense_difference(blocks_a, blocks_b, mask, central, fermi, vacuum, dense):
    worst, worst_k, residuals = 0.0, None, []
    for k in dense:
        s0 = np.asarray(central.Sk(k=k, format="array"))
        k0 = np.asarray(central.Hk(k=k, format="array")) + (fermi - vacuum) * s0
        delta = (b_i_from_connection(blocks_a, mask, k)
                 - b_i_from_connection(blocks_b, mask, k))
        error, residual = _electronic(delta, s0, k0)
        residuals.append(residual)
        if error > worst:
            worst, worst_k = error, list(map(float, k))
    return {"worst_400k_propagated_ev_per_ang": worst, "worst_k_reduced": worst_k,
            "max_solve_residual_relative": max(residuals)}


def main() -> int:
    _, central, fermi, vacuum, _ = _matrix_data(PROD_CENTRAL)
    geometry = central.geometry
    radials, _ = read_radials(PROD_CENTRAL / "C.ion.xml")
    basis = Basis(radials, [0, 1, 1, 1])
    norms = direct_radial_norms(basis)
    normalization_factors = {l: 1.0 / np.sqrt(value) for l, value in norms.items()}
    normalized_norm_error = max(abs(value * normalization_factors[l] ** 2 - 1.0)
                                for l, value in norms.items())
    engines = {name: FourierBesselSP(basis, *config) for name, config in ENGINES.items()}
    images = enumerate_images(basis, np.asarray(geometry.cell), geometry.xyz)
    mask, dense = _selector(geometry), _dense_k_points()

    cases, saved = [], {}
    for direction in sorted(DIRECTIONS):
        blocks = {name: connection_blocks(engine, images, geometry, direction)
                  for name, engine in engines.items()}
        home = blocks["q180_normalized"][(0, 0, 0)][:4, :4]
        anti = float(np.linalg.norm(home + home.T) / max(np.linalg.norm(home), np.finfo(float).tiny))
        q_tail = dense_difference(blocks["q180_raw"], blocks["q120_raw"], mask,
                                  central, fermi, vacuum, dense)
        norm_effect = dense_difference(blocks["q180_normalized"], blocks["q180_raw"], mask,
                                       central, fermi, vacuum, dense)
        q_pass = q_tail["worst_400k_propagated_ev_per_ang"] < B1_LIMIT_EV_PER_ANG
        norm_pass = norm_effect["worst_400k_propagated_ev_per_ang"] < B1_LIMIT_EV_PER_ANG
        a_pass = normalized_norm_error < 1e-14 and anti < ANTIHERMITIAN_TOL

        prolate = prolate_b_i_blocks(basis, images, geometry.xyz, direction, 0.00125, 400)
        continuous = dense_difference(blocks["q180_normalized"], prolate, mask,
                                      central, fermi, vacuum, dense)
        if a_pass and q_pass and norm_pass:
            comparison, matrices = gate_b3(
                blocks["q180_normalized"], mask, direction, central, fermi, vacuum,
                np.load(ONLYS_MATRICES), dense,
            )
            comparison["evidence_status"] = (
                "corrected-method confirmation under thresholds fixed before v4; not blind"
            )
            saved.update(matrices)
        else:
            comparison = {"verdict": "BLOCKED", "reason": "V5-A or V5-B failed"}
        cases.append({
            "direction": direction,
            "V5_A_normalization": {"verdict": "PASS" if a_pass else "NO_GO",
                                     "antihermiticity_relative": anti},
            "V5_B_q_tail": {"verdict": "PASS" if q_pass else "NO_GO",
                              "threshold_ev_per_ang": B1_LIMIT_EV_PER_ANG, **q_tail},
            "V5_B_normalization_effect": {"verdict": "PASS" if norm_pass else "NO_GO",
                                           "threshold_ev_per_ang": B1_LIMIT_EV_PER_ANG,
                                           **norm_effect},
            "V5_C_continuous_diagnostic": {"gate": False, **continuous},
            "V5_D_onlyS": comparison,
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    matrices = OUTPUT / "b_i_fourier_bessel_semantic_v5_matrices.npz"
    np.savez_compressed(matrices, **saved)
    passed = all(case[key]["verdict"] == "PASS" for case in cases for key in (
        "V5_A_normalization", "V5_B_q_tail", "V5_B_normalization_effect", "V5_D_onlyS"
    ))
    report = {
        "schema": "b_i_fourier_bessel_semantic_closure_v5",
        "gate": "B_I_fourier_bessel_semantic_closure_v5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO",
        "engines": {name: {"n_r": cfg[0], "q_max_inv_ang": cfg[1], "n_q": cfg[2],
                             "radially_normalized": cfg[3]} for name, cfg in ENGINES.items()},
        "direct_radial_normalization": {"raw_norms": norms,
                                         "normalization_factors": normalization_factors,
                                         "normalized_norm_error": normalized_norm_error,
                                         "definition": "R_l / sqrt(integral r^2 R_l^2 dr)"},
        "thresholds": {"q_tail_ev_per_ang": B1_LIMIT_EV_PER_ANG,
                       "normalization_effect_ev_per_ang": B1_LIMIT_EV_PER_ANG,
                       "onlyS_ev_per_ang": B3_LIMIT_EV_PER_ANG,
                       "antihermiticity_relative": ANTIHERMITIAN_TOL},
        "cases": cases,
        "historical_NO_GO_preserved": ["v1", "analytic_v2", "prolate_v3",
                                        "prolate_400-320", "fourier_bessel_v4",
                                        "fourier_bessel_v4_confirmation"],
        "matrix_artifact": str(matrices), "matrix_artifact_sha256": file_sha256(matrices),
        "claim_scope": "independent PROD-SZ intra-atomic basis connection only",
    }
    path = OUTPUT / "b_i_fourier_bessel_semantic_v5.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"B_I Fourier-Bessel semantic v5: {report['verdict']} -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
