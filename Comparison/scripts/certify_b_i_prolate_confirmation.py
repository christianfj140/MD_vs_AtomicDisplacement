#!/usr/bin/env python3
"""One-shot 400-vs-320 confirmation of the converged prolate ``B_I`` route."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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

OUTPUT = REPO_ROOT / "Comparison/results/epc/b_i_prolate_confirmation"
ORDERS = (320, 400)
H_ANG = 0.00125


def main() -> int:
    _, central, fermi, vacuum, _ = _matrix_data(PROD_CENTRAL)
    geometry = central.geometry
    radials, _ = read_radials(PROD_CENTRAL / "C.ion.xml")
    basis = Basis(radials, [0, 1, 1, 1])
    images = enumerate_images(basis, np.asarray(geometry.cell), geometry.xyz)
    mask, dense = _selector(geometry), _dense_k_points()
    onlys = np.load(ONLYS_MATRICES)
    cases, saved = [], {}
    for direction in sorted(DIRECTIONS):
        blocks = {
            order: prolate_b_i_blocks(basis, images, geometry.xyz, direction, H_ANG, order)
            for order in ORDERS
        }
        worst, worst_k, residuals = 0.0, None, []
        for k in dense:
            s0 = np.asarray(central.Sk(k=k, format="array"))
            k0 = np.asarray(central.Hk(k=k, format="array")) + (fermi - vacuum) * s0
            delta = (b_i_from_connection(blocks[400], mask, k)
                     - b_i_from_connection(blocks[320], mask, k))
            error, residual = _electronic(delta, s0, k0)
            residuals.append(residual)
            if error > worst:
                worst, worst_k = error, list(map(float, k))
        convergence = {
            "verdict": "PASS" if worst < B1_LIMIT_EV_PER_ANG else "NO_GO",
            "comparison": "order 400 - 320",
            "dense_grid": [20, 20, 1],
            "worst_propagated_ev_per_ang": worst,
            "worst_k_reduced": worst_k,
            "threshold_ev_per_ang": B1_LIMIT_EV_PER_ANG,
            "max_solve_residual_relative": max(residuals),
        }
        comparison, matrices = gate_b3(
            blocks[400], mask, direction, central, fermi, vacuum, onlys, dense
        )
        saved.update(matrices)
        cases.append({"direction": direction, "dense_quadrature_confirmation": convergence,
                      "gate_B3_BI": comparison})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    matrices = OUTPUT / "b_i_prolate_confirmation_matrices.npz"
    np.savez_compressed(matrices, **saved)
    passed = all(
        case[key]["verdict"] == "PASS" for case in cases
        for key in ("dense_quadrature_confirmation", "gate_B3_BI")
    )
    report = {
        "schema": "b_i_prolate_confirmation_v1",
        "gate": "B_I_independent_prolate_confirmation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO",
        "preregistered_before_run": {
            "orders": list(ORDERS), "h_ang": H_ANG,
            "quadrature_limit_ev_per_ang": B1_LIMIT_EV_PER_ANG,
            "B3_limit_ev_per_ang": B3_LIMIT_EV_PER_ANG,
            "stop_policy": "no further quadrature refinement if C1_x remains NO_GO",
        },
        "cases": cases,
        "historical_gates": {"v1": "NO_GO unchanged", "analytic_v2": "NO_GO unchanged",
                             "prolate_v3": "NO_GO unchanged", "B0_v1": "NO_GO unchanged"},
        "matrix_artifact": str(matrices), "matrix_artifact_sha256": file_sha256(matrices),
        "claim_scope": "PROD-SZ B_I confirmation only; Delta_out and full-KS remain blocked",
    }
    path = OUTPUT / "b_i_prolate_confirmation.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"B_I prolate confirmation: {report['verdict']} -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
