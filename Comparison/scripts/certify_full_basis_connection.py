#!/usr/bin/env python3
"""Certify one-sided connections in the existing nested TZ/QZ bases."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from audit_reference_basis_dense_k import _runs  # noqa: E402
from certify_production_basis_connection import ADJOINT_REL_TOL, LIMIT_EV_PER_ANG, _connections, _propagate  # noqa: E402
from cross_basis_projection_preflight import K_POINTS, _basis_geometry  # noqa: E402
from run_displaced_projection_sentinel import H_ANG  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

ROOT = REPO_ROOT / "Comparison/results/epc"
OUTPUT = ROOT / "full_basis_connection"
CASES = (("n_tzp_d12_14", "C1_x"), ("n_tzp_d12_14", "C1_z"),
         ("n_qzp_d12_14", "C1_x"), ("n_qzp_d12_14", "C1_z"),
         ("n_qzp_d10_12", "C1_x"))
WEIGHTS = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}


def main() -> int:
    import sisl

    k_points = np.asarray(K_POINTS)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for basis, direction in CASES:
        runs = _runs(basis, direction)
        label, central, fermi, vacuum, atom = _matrix_data(runs[0])
        shape = tuple(int(x) for x in sisl.get_sile(str(runs[0] / f"{label}.VT")).read_grid().shape)
        geometry = _basis_geometry(central.geometry, runs[0] / "C.ion.xml")
        dvolume = abs(np.linalg.det(geometry.cell)) / np.prod(shape)
        connection = _connections(direction, H_ANG, geometry, atom, shape, dvolume, k_points)
        hamiltonians = {step: _matrix_data(path)[1] for step, path in runs.items()}
        rows, left_stack, right_stack, ds_stack, failures = [], [], [], [], []
        for ik, (k, (left, right)) in enumerate(zip(k_points, connection)):
            s0 = np.asarray(central.Sk(k=k, format="array"))
            k0 = np.asarray(central.Hk(k=k, format="array")) + (fermi - vacuum) * s0
            d_s = sum(weight * np.asarray(hamiltonians[step].Sk(k=k, format="array"))
                      for step, weight in WEIGHTS.items()) / (12 * H_ANG)
            adjoint = float(np.linalg.norm(left - right.conj().T) / max(np.linalg.norm(left), np.finfo(float).tiny))
            residual = d_s - left - right
            propagated = _propagate(residual / 2, residual / 2, s0, k0)
            if adjoint >= ADJOINT_REL_TOL or propagated >= LIMIT_EV_PER_ANG:
                failures.append(f"k={ik}: adjoint={adjoint:.6g}, propagated={propagated:.6g} eV/Ang")
            rows.append({"k_reduced": k.tolist(), "adjoint_relative": adjoint,
                         "D_S_connection_residual_propagated_ev_per_ang": propagated})
            left_stack.append(left); right_stack.append(right); ds_stack.append(d_s)
        artifact = OUTPUT / f"{basis}_{direction}_connections.npz"
        np.savez_compressed(artifact, k_points=k_points, S_L=np.asarray(left_stack), S_R=np.asarray(right_stack), D_S=np.asarray(ds_stack))
        cases.append({"basis": basis, "direction": direction, "verdict": "PASS" if not failures else "NO_GO",
                      "failure_reasons": failures, "k_results": rows, "artifact": str(artifact),
                      "artifact_sha256": file_sha256(artifact)})
        print(f"full basis connection {basis} {direction}: {cases[-1]['verdict']}", flush=True)
    passed = all(case["verdict"] == "PASS" for case in cases)
    report = {"schema": "full_basis_connection_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
              "verdict": "PASS" if passed else "NO_GO", "gate": "full_basis_connection",
              "k_count": len(k_points), "thresholds": {"adjoint_relative": ADJOINT_REL_TOL,
                                                          "propagated_error_ev_per_ang": LIMIT_EV_PER_ANG},
              "cases": cases, "claim_scope": "one-sided nested-basis connection validation only; no independent analytic intraatomic claim"}
    path = OUTPUT / "full_basis_connection.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"full basis connection: {report['verdict']} -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
