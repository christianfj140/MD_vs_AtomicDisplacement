#!/usr/bin/env python3
"""Validate PROD-SZ ``B_I`` with an analytic two-centre PAO derivative.

This v2 preserves every v1 NO_GO and every preregistered threshold. It removes
the diagnosed displaced-cutoff finite-difference noise by integrating
``<phi_a|-grad_u phi_b>`` directly from the tabulated ``.ion.xml`` radials.
``.onlyS`` enters only in the final B3 comparison.
"""

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
    B1_LIMIT_EV_PER_ANG,
    B3_LIMIT_EV_PER_ANG,
    ONLYS_MATRICES,
    QUADRATURES,
    Basis,
    _electronic,
    _grid,
    analytic_b_i_blocks,
    b_i_from_connection,
    enumerate_images,
    gate_b3,
    read_radials,
    verify_orbital_ordering,
)
from certify_support_exact_connection import _selector  # noqa: E402
from cross_basis_projection_preflight import K_POINTS  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from onlys_semantic_preflight import _dense_k_points  # noqa: E402
from run_displaced_projection_sentinel import DIRECTIONS  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/b_i_analytic_two_center"
LEVELS = ("medium", "fine", "fine_ang2")


def evaluate(output: Path) -> dict:
    _, central, fermi, vacuum, _ = _matrix_data(PROD_CENTRAL)
    geometry = central.geometry
    radials, _ = read_radials(PROD_CENTRAL / "C.ion.xml")
    basis = Basis(radials, [0, 1, 1, 1])
    images = enumerate_images(basis, np.asarray(geometry.cell), geometry.xyz)
    mask = _selector(geometry)
    onlys = np.load(ONLYS_MATRICES)
    orb_indx = sorted(PROD_CENTRAL.glob("*.ORB_INDX"))[0]

    cases, saved = [], {}
    for direction in sorted(DIRECTIONS):
        computed = {}
        for level in LEVELS:
            blocks = analytic_b_i_blocks(
                basis, images, geometry.xyz,
                _grid(basis, *QUADRATURES[level]), direction,
            )
            computed[level] = (blocks, b_i_from_connection(blocks, mask, K_POINTS[0]))

        s0 = np.asarray(central.Sk(k=K_POINTS[0], format="array"))
        k0 = np.asarray(central.Hk(k=K_POINTS[0], format="array")) + (fermi - vacuum) * s0
        refinements, failures, residuals = [], [], []
        for fine, coarse in (("fine", "medium"), ("fine_ang2", "fine")):
            error, residual = _electronic(computed[fine][1] - computed[coarse][1], s0, k0)
            residuals.append(residual)
            refinements.append({
                "comparison": f"{fine} - {coarse}",
                "propagated_ev_per_ang": error,
            })
            if not error < B1_LIMIT_EV_PER_ANG:
                failures.append(f"{fine}-{coarse}: propagated={error:.6g} eV/Ang")
        quadrature = {
            "verdict": "PASS" if not failures else "NO_GO",
            "failure_reasons": failures,
            "threshold_ev_per_ang": B1_LIMIT_EV_PER_ANG,
            "refinements": refinements,
            "max_solve_residual_relative": max(residuals, default=0.0),
        }

        production_blocks = computed["fine_ang2"][0]
        comparison, matrices = gate_b3(
            production_blocks, mask, direction, central, fermi, vacuum,
            onlys, _dense_k_points(),
        )
        saved.update(matrices)
        cases.append({
            "direction": direction,
            "analytic_quadrature": quadrature,
            "gate_B3_BI": comparison,
        })

    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / "b_i_analytic_two_center_matrices.npz"
    np.savez_compressed(matrix_path, **saved)
    passed = all(
        case["analytic_quadrature"]["verdict"] == "PASS"
        and case["gate_B3_BI"]["verdict"] == "PASS"
        for case in cases
    )
    v1_path = REPO_ROOT / "Comparison/results/epc/b_i_independent_two_center/b_i_independent_two_center.json"
    report = {
        "schema": "b_i_analytic_two_center_v2",
        "gate": "B_I_independent_analytic_two_center_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO",
        "method": "analytic centre derivative <phi_row|-grad_u phi_ket>",
        "basis": "PROD-SZ",
        "thresholds": {
            "quadrature_propagated_ev_per_ang": B1_LIMIT_EV_PER_ANG,
            "B3_propagated_ev_per_ang": B3_LIMIT_EV_PER_ANG,
        },
        "orbital_ordering": verify_orbital_ordering(orb_indx, basis),
        "independence": {
            "D_S_used": False,
            "onlyS_used_in_construction": False,
            "sisl_orbital_psi_used": False,
            "finite_difference_used_for_B_I": False,
        },
        "historical_gates": {
            "B_I_independent_two_center_validation_v1": "NO_GO (unchanged)",
            "B0_overlap_v1": "NO_GO (unchanged; not superseded by this derivative gate)",
        },
        "cases": cases,
        "matrix_artifact": str(matrix_path),
        "matrix_artifact_sha256": file_sha256(matrix_path),
        "v1_report": str(v1_path),
        "v1_report_sha256": file_sha256(v1_path),
        "claim_scope": (
            "Independent analytic validation of PROD-SZ B_I only. This does not "
            "adjudicate full_basis_connection, Delta_out, or full-KS."
        ),
    }
    report_path = output / "b_i_analytic_two_center.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"B_I analytic two-centre validation: {report['verdict']} -> {report_path}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    return 0 if evaluate(args.output)["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
