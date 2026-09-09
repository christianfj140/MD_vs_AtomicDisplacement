#!/usr/bin/env python3
"""Re-adjudicate v4 with a scale-stable Cholesky backward error."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from audit_reference_basis_dense_k import _runs  # noqa: E402
from certify_production_basis_connection import LIMIT_EV_PER_ANG  # noqa: E402
from certify_support_exact_connection import SOLVE_RESIDUAL_TOL  # noqa: E402
from run_nested_basis_ladder import _rms  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

V4_ROOT = REPO_ROOT / "Comparison/results/epc/full_basis_connection_semantic_v4"
OUTPUT = REPO_ROOT / "Comparison/results/epc/full_basis_connection_semantic_v5"


def solve(overlap: np.ndarray, rhs: np.ndarray):
    matrix = (overlap + overlap.conj().T) / 2
    solution = cho_solve(cho_factor(matrix, lower=True), rhs)
    residual = np.linalg.norm(matrix @ solution - rhs)
    scale = np.linalg.norm(matrix) * np.linalg.norm(solution) + np.linalg.norm(rhs)
    return solution, float(residual / max(scale, np.finfo(float).tiny))


def main() -> int:
    v4 = json.loads((V4_ROOT / "full_basis_connection_semantic_v4.json").read_text())
    if v4["verdict"] != "NO_GO":
        raise RuntimeError("v5 expects the preserved semantic v4 NO_GO")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for old in v4["cases"]:
        basis, direction = old["basis"], old["direction"]
        _, central, fermi, vacuum, _ = _matrix_data(_runs(basis, direction)[0])
        source = np.load(old["artifact"])
        rows, worst, max_residual, over = [], 0.0, 0.0, 0
        for ik, k in enumerate(source["k_points"]):
            s0 = np.asarray(central.Sk(k=k, format="array"))
            k0 = np.asarray(central.Hk(k=k, format="array")) + (fermi - vacuum) * s0
            delta = source["D_S"][ik] - source["S_L"][ik] - source["S_R"][ik]
            x, r1 = solve(s0, k0)
            y, r2 = solve(s0, delta / 2)
            propagated = _rms(-(delta / 2 @ x + k0 @ y), s0)[0]
            residual = max(r1, r2)
            adjoint = float(np.linalg.norm(source["S_L"][ik]
                                           - source["S_R"][ik].conj().T))
            passed = propagated < LIMIT_EV_PER_ANG and residual < SOLVE_RESIDUAL_TOL \
                and adjoint == 0.0
            over += int(not passed)
            worst, max_residual = max(worst, propagated), max(max_residual, residual)
            if ik < 3:
                rows.append({"k_reduced": list(map(float, k)),
                             "closure_propagated_ev_per_ang": propagated,
                             "adjoint_absolute": adjoint,
                             "solve_backward_error": residual})
        inherited_sym = old["dense_audit"]["worst_symmetrization_propagated_ev_per_ang"]
        passed = not over and inherited_sym < LIMIT_EV_PER_ANG
        artifact = OUTPUT / f"{basis}_{direction}_connections.npz"
        artifact.write_bytes(Path(old["artifact"]).read_bytes())
        cases.append({"basis": basis, "direction": direction,
                      "verdict": "PASS" if passed else "NO_GO", "diagnostic_k": rows,
                      "dense_audit": {"k_count": len(source["k_points"]),
                                      "k_over_limit": over,
                                      "worst_closure_propagated_ev_per_ang": worst,
                                      "worst_symmetrization_propagated_ev_per_ang": inherited_sym,
                                      "max_solve_backward_error": max_residual},
                      "artifact": str(artifact), "artifact_sha256": file_sha256(artifact)})
        print(basis, direction, cases[-1]["verdict"])
    passed = all(case["verdict"] == "PASS" for case in cases)
    report = {"schema": "full_basis_connection_semantic_v5",
              "gate": "full_basis_connection", "verdict": "PASS" if passed else "NO_GO",
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "method": "v4 exact-adjoint connections with scale-stable Cholesky backward error",
              "status_note": "corrected-method confirmation after observing v4 near-zero-RHS residuals",
              "thresholds": {"electronic_ev_per_ang": LIMIT_EV_PER_ANG,
                             "solve_backward_error": SOLVE_RESIDUAL_TOL},
              "historical_v1_v2_v3_v4": "NO_GO unchanged", "cases": cases,
              "claim_scope": "nested-basis one-sided connections; not Delta_out/full-KS"}
    path = OUTPUT / "full_basis_connection_semantic_v5.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"full basis connection semantic v5: {report['verdict']} -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
