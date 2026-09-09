#!/usr/bin/env python3
"""Prospective v3 gate for ``B_I`` using support-adapted prolate quadrature."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from certify_b_i_independent_two_center import (  # noqa: E402
    B1_LIMIT_EV_PER_ANG, B2_LIMIT_EV_PER_ANG, B3_LIMIT_EV_PER_ANG,
    ONLYS_MATRICES, Basis, _electronic, b_i_from_connection, enumerate_images,
    gate_b3, prolate_b_i_blocks, read_radials,
)
from certify_support_exact_connection import _selector  # noqa: E402
from cross_basis_projection_preflight import K_POINTS  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from onlys_semantic_preflight import _dense_k_points  # noqa: E402
from run_displaced_projection_sentinel import DIRECTIONS  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/b_i_prolate_two_center"
QUADRATURE_ORDERS = (180, 240, 320)
H_VALUES = (0.0025, 0.00125)
PRODUCTION_ORDER = 320
PRODUCTION_H = 0.00125


def evaluate(output: Path) -> dict:
    _, central, fermi, vacuum, _ = _matrix_data(PROD_CENTRAL)
    geometry = central.geometry
    radials, _ = read_radials(PROD_CENTRAL / "C.ion.xml")
    basis = Basis(radials, [0, 1, 1, 1])
    images = enumerate_images(basis, np.asarray(geometry.cell), geometry.xyz)
    mask, onlys = _selector(geometry), np.load(ONLYS_MATRICES)
    cases, saved = [], {}
    for direction in sorted(DIRECTIONS):
        by_order = {
            order: prolate_b_i_blocks(basis, images, geometry.xyz, direction, 0.0025, order)
            for order in QUADRATURE_ORDERS
        }
        s0 = np.asarray(central.Sk(k=K_POINTS[0], format="array"))
        k0 = np.asarray(central.Hk(k=K_POINTS[0], format="array")) + (fermi - vacuum) * s0
        q_error, q_residual = _electronic(
            b_i_from_connection(by_order[320], mask, K_POINTS[0])
            - b_i_from_connection(by_order[240], mask, K_POINTS[0]), s0, k0,
        )
        quadrature = {
            "verdict": "PASS" if q_error < B1_LIMIT_EV_PER_ANG else "NO_GO",
            "orders": list(QUADRATURE_ORDERS),
            "decisive_comparison": "320 - 240 (unseen before preregistration)",
            "propagated_ev_per_ang": q_error,
            "threshold_ev_per_ang": B1_LIMIT_EV_PER_ANG,
            "solve_residual_relative": q_residual,
        }

        fine_h = prolate_b_i_blocks(
            basis, images, geometry.xyz, direction, PRODUCTION_H, PRODUCTION_ORDER
        )
        h_error, h_residual = _electronic(
            b_i_from_connection(fine_h, mask, K_POINTS[0])
            - b_i_from_connection(by_order[320], mask, K_POINTS[0]), s0, k0,
        )
        stencil = {
            "verdict": "PASS" if h_error < B2_LIMIT_EV_PER_ANG else "NO_GO",
            "h_values_ang": list(H_VALUES),
            "production_h_ang": PRODUCTION_H,
            "propagated_ev_per_ang": h_error,
            "threshold_ev_per_ang": B2_LIMIT_EV_PER_ANG,
            "solve_residual_relative": h_residual,
        }
        comparison, matrices = gate_b3(
            fine_h, mask, direction, central, fermi, vacuum, onlys, _dense_k_points()
        )
        saved.update(matrices)
        cases.append({"direction": direction, "quadrature": quadrature,
                      "stencil": stencil, "gate_B3_BI": comparison})

    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / "b_i_prolate_two_center_matrices.npz"
    np.savez_compressed(matrix_path, **saved)
    passed = all(
        case[key]["verdict"] == "PASS"
        for case in cases for key in ("quadrature", "stencil", "gate_B3_BI")
    )
    report = {
        "schema": "b_i_prolate_two_center_v3",
        "gate": "B_I_independent_prolate_two_center_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO",
        "method": "prolate-spheroidal two-centre overlap plus five-point centre derivative",
        "preregistration": {
            "note": "orders 320-240 and h 0.00125-0.0025 were fixed before either result was evaluated",
            "quadrature_orders": list(QUADRATURE_ORDERS), "h_values_ang": list(H_VALUES),
        },
        "thresholds": {"quadrature": B1_LIMIT_EV_PER_ANG,
                       "h": B2_LIMIT_EV_PER_ANG, "B3": B3_LIMIT_EV_PER_ANG},
        "historical_gates": {"v1": "NO_GO unchanged", "analytic_v2": "NO_GO unchanged",
                             "B0_overlap_v1": "NO_GO unchanged and not superseded"},
        "independence": {"D_S_used": False, "onlyS_used_in_construction": False,
                         "sisl_orbital_psi_used": False, "VT_grid_used": False},
        "cases": cases, "matrix_artifact": str(matrix_path),
        "matrix_artifact_sha256": file_sha256(matrix_path),
        "claim_scope": "PROD-SZ B_I only; no full-basis, Delta_out, or full-KS promotion",
    }
    path = output / "b_i_prolate_two_center.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"B_I prolate two-centre validation: {report['verdict']} -> {path}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    return 0 if evaluate(args.output)["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
