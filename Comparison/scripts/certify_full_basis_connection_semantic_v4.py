#!/usr/bin/env python3
"""Close nested-basis connections by enforcing the exact adjoint identity."""

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
from certify_full_basis_connection import CASES, WEIGHTS  # noqa: E402
from certify_full_basis_connection_onlys_v2 import produce  # noqa: E402
from certify_full_basis_connection_onlys_v3 import OUTPUT as V3_ROOT, cross_derivative  # noqa: E402
from certify_production_basis_connection import LIMIT_EV_PER_ANG  # noqa: E402
from certify_support_exact_connection import SOLVE_RESIDUAL_TOL, _propagate  # noqa: E402
from onlys_semantic_preflight import _dense_k_points  # noqa: E402
from run_displaced_projection_sentinel import H_ANG  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/full_basis_connection_semantic_v4"


def enforce_adjoint(right: np.ndarray, left: np.ndarray):
    right_exact = (right + left.conj().T) / 2
    return right_exact, right_exact.conj().T


def main() -> int:
    import sisl

    v3 = json.loads((V3_ROOT / "full_basis_connection_onlys_v3.json").read_text())
    if v3["verdict"] != "NO_GO":
        raise RuntimeError("v4 expects the preserved full-basis v3 NO_GO")
    dense, cases = _dense_k_points(), []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for basis, direction in CASES:
        runs = _runs(basis, direction)
        _, central, fermi, vacuum, _ = _matrix_data(runs[0])
        overlap_runs = produce(basis, direction, runs[0], central.geometry,
                               tuple(WEIGHTS), V3_ROOT)
        overlaps = {key: sisl.get_sile(str(next(path.glob("*.onlyS")))).read_overlap()
                    for key, path in overlap_runs.items()}
        hamiltonians = {step: _matrix_data(path)[1] for step, path in runs.items()}
        lefts, rights, ds_values, rows = [], [], [], []
        over, worst_closure, worst_sym, worst_k, max_solve = 0, 0.0, 0.0, None, 0.0
        for k in dense:
            right_raw, left_raw = cross_derivative(overlaps, direction, central.no, k)
            right, left = enforce_adjoint(right_raw, left_raw)
            s0 = np.asarray(central.Sk(k=k, format="array"))
            k0 = np.asarray(central.Hk(k=k, format="array")) + (fermi - vacuum) * s0
            d_s = sum(WEIGHTS[step] * np.asarray(hamiltonians[step].Sk(k=k, format="array"))
                      for step in WEIGHTS) / (12 * H_ANG)
            closure, res1 = _propagate((d_s - left - right) / 2,
                                       (d_s - left - right) / 2, s0, k0)
            sym_effect, res2 = _propagate(left - left_raw, right - right_raw, s0, k0)
            solve = max(res1, res2)
            passed = closure < LIMIT_EV_PER_ANG and sym_effect < LIMIT_EV_PER_ANG \
                and solve < SOLVE_RESIDUAL_TOL
            over += int(not passed)
            if max(closure, sym_effect) > max(worst_closure, worst_sym):
                worst_k = list(map(float, k))
            worst_closure, worst_sym = max(worst_closure, closure), max(worst_sym, sym_effect)
            max_solve = max(max_solve, solve)
            if len(rows) < 3:
                rows.append({"k_reduced": list(map(float, k)),
                             "closure_propagated_ev_per_ang": closure,
                             "symmetrization_propagated_ev_per_ang": sym_effect,
                             "adjoint_relative": float(np.linalg.norm(left - right.conj().T)),
                             "solve_residual_relative": solve})
            lefts.append(left); rights.append(right); ds_values.append(d_s)
        artifact = OUTPUT / f"{basis}_{direction}_connections.npz"
        np.savez_compressed(artifact, k_points=np.asarray(dense), S_L=np.asarray(lefts),
                            S_R=np.asarray(rights), D_S=np.asarray(ds_values))
        cases.append({"basis": basis, "direction": direction,
                      "verdict": "PASS" if not over else "NO_GO", "diagnostic_k": rows,
                      "dense_audit": {"k_count": len(dense), "k_over_limit": over,
                                      "worst_closure_propagated_ev_per_ang": worst_closure,
                                      "worst_symmetrization_propagated_ev_per_ang": worst_sym,
                                      "worst_k_reduced": worst_k,
                                      "max_solve_residual_relative": max_solve},
                      "artifact": str(artifact), "artifact_sha256": file_sha256(artifact)})
        print(basis, direction, cases[-1]["verdict"], flush=True)
    passed = all(case["verdict"] == "PASS" for case in cases)
    report = {"schema": "full_basis_connection_semantic_v4",
              "gate": "full_basis_connection", "verdict": "PASS" if passed else "NO_GO",
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "method": "five-point onlyS connection, projected onto exact S_L=S_R^dagger identity",
              "status_note": "corrected-method confirmation after observing v3 adjoint residuals",
              "thresholds": {"closure_and_symmetrization_ev_per_ang": LIMIT_EV_PER_ANG,
                             "solve_residual_relative": SOLVE_RESIDUAL_TOL},
              "historical_v1_v2_v3": "NO_GO unchanged", "cases": cases,
              "claim_scope": "nested-basis one-sided connections; not Delta_out/full-KS"}
    path = OUTPUT / "full_basis_connection_semantic_v4.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"full basis connection semantic v4: {report['verdict']} -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
