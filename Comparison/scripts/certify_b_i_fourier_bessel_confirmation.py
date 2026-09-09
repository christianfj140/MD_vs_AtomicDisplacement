#!/usr/bin/env python3
"""One-shot normalized/q-tail confirmation of Fourier-Bessel ``B_I`` v4."""

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
    ANTIHERMITIAN_TOL, CONFIGS, MONOCENTRIC_TOL, FourierBesselSP, connection_blocks,
)
from certify_b_i_independent_two_center import (  # noqa: E402
    B1_LIMIT_EV_PER_ANG, ONLYS_MATRICES, Basis, _electronic,
    b_i_from_connection, enumerate_images, gate_b3, read_radials,
)
from certify_support_exact_connection import _selector  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from onlys_semantic_preflight import _dense_k_points  # noqa: E402
from run_displaced_projection_sentinel import DIRECTIONS  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/b_i_fourier_bessel_confirmation"
CONFIRMATION_CONFIG = (1000, 180.0, 7201)


def main() -> int:
    _, central, fermi, vacuum, _ = _matrix_data(PROD_CENTRAL)
    geometry = central.geometry
    radials, _ = read_radials(PROD_CENTRAL / "C.ion.xml")
    basis = Basis(radials, [0, 1, 1, 1])
    images = enumerate_images(basis, np.asarray(geometry.cell), geometry.xyz)
    mask, dense = _selector(geometry), _dense_k_points()
    old = FourierBesselSP(basis, *CONFIGS["fine"])
    new = FourierBesselSP(basis, *CONFIRMATION_CONFIG, normalize=True)
    identity_error = float(np.linalg.norm(new.overlap(np.zeros(3)) - np.eye(4), 2))
    cases, saved = [], {}
    for direction in sorted(DIRECTIONS):
        old_blocks = connection_blocks(old, images, geometry, direction)
        new_blocks = connection_blocks(new, images, geometry, direction)
        home = new_blocks[(0, 0, 0)][:4, :4]
        anti = float(np.linalg.norm(home + home.T) / max(np.linalg.norm(home), np.finfo(float).tiny))
        worst, worst_k = 0.0, None
        for k in dense:
            s0 = np.asarray(central.Sk(k=k, format="array"))
            k0 = np.asarray(central.Hk(k=k, format="array")) + (fermi - vacuum) * s0
            error, _ = _electronic(
                b_i_from_connection(new_blocks, mask, k)
                - b_i_from_connection(old_blocks, mask, k), s0, k0,
            )
            if error > worst:
                worst, worst_k = error, list(map(float, k))
        a_pass = identity_error < MONOCENTRIC_TOL and anti < ANTIHERMITIAN_TOL
        b_pass = worst < B1_LIMIT_EV_PER_ANG
        if a_pass and b_pass:
            comparison, matrices = gate_b3(
                new_blocks, mask, direction, central, fermi, vacuum,
                np.load(ONLYS_MATRICES), dense,
            )
            saved.update(matrices)
        else:
            comparison = {"verdict": "BLOCKED", "reason": "confirmation A/B failed"}
        cases.append({
            "direction": direction,
            "A_monocentric": {"verdict": "PASS" if a_pass else "NO_GO",
                               "overlap_spectral_error": identity_error,
                               "antihermiticity_relative": anti},
            "B_tail_confirmation": {"verdict": "PASS" if b_pass else "NO_GO",
                                     "worst_400k_propagated_ev_per_ang": worst,
                                     "worst_k_reduced": worst_k,
                                     "threshold_ev_per_ang": B1_LIMIT_EV_PER_ANG},
            "D_onlyS": comparison,
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    matrices = OUTPUT / "b_i_fourier_bessel_confirmation_matrices.npz"
    np.savez_compressed(matrices, **saved)
    passed = all(case[key]["verdict"] == "PASS" for case in cases
                 for key in ("A_monocentric", "B_tail_confirmation", "D_onlyS"))
    report = {
        "schema": "b_i_fourier_bessel_confirmation_v1",
        "gate": "B_I_fourier_bessel_gaunt_v4_confirmation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO",
        "preregistered_before_run": {"config": {"n_r": 1000, "q_max_inv_ang": 180.0,
                                                   "n_q": 7201},
                                      "radial_normalization": "unit norm before transform",
                                      "comparison": "confirmation minus original v4 fine",
                                      "limit_ev_per_ang": B1_LIMIT_EV_PER_ANG},
        "cases": cases,
        "historical_v4": "NO_GO unchanged",
        "matrix_artifact": str(matrices), "matrix_artifact_sha256": file_sha256(matrices),
        "claim_scope": "PROD-SZ B_I only; downstream gates remain unchanged pending this verdict",
    }
    path = OUTPUT / "b_i_fourier_bessel_confirmation.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"B_I Fourier-Bessel confirmation: {report['verdict']} -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
