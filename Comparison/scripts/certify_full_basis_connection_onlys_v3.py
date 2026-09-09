#!/usr/bin/env python3
"""Five-point ``.onlyS`` closure for the existing nested TZ/QZ bases."""

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
from certify_production_basis_connection import ADJOINT_REL_TOL, LIMIT_EV_PER_ANG  # noqa: E402
from certify_support_exact_connection import SOLVE_RESIDUAL_TOL, _propagate  # noqa: E402
from onlys_semantic_preflight import _dense_k_points  # noqa: E402
from run_displaced_projection_sentinel import H_ANG  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/full_basis_connection_onlys_v3"


def cross_derivative(overlaps, direction, no, k):
    right = left = 0.0
    for step, weight in WEIGHTS.items():
        matrix = np.asarray(overlaps[(direction, step)].Sk(k=k, format="array"))
        right = right + weight * matrix[:no, no:]
        left = left + weight * matrix[no:, :no]
    return right / (12 * H_ANG), left / (12 * H_ANG)


def main() -> int:
    import sisl

    semantic = json.loads((REPO_ROOT / "Comparison/results/epc/b_i_fourier_bessel_semantic_v5/b_i_fourier_bessel_semantic_v5.json").read_text())
    if semantic["verdict"] != "PASS":
        raise RuntimeError("PROD-SZ B_I semantic closure v5 is not PASS")
    dense, cases = _dense_k_points(), []
    for basis, direction in CASES:
        runs = _runs(basis, direction)
        _, central, fermi, vacuum, _ = _matrix_data(runs[0])
        overlap_runs = produce(basis, direction, runs[0], central.geometry,
                               tuple(WEIGHTS), OUTPUT)
        overlaps = {key: sisl.get_sile(str(next(path.glob("*.onlyS")))).read_overlap()
                    for key, path in overlap_runs.items()}
        hamiltonians = {step: _matrix_data(path)[1] for step, path in runs.items()}
        no, rows, saved, over, worst, worst_k = central.no, [], {}, 0, 0.0, None
        for ik, k in enumerate(dense):
            right, left = cross_derivative(overlaps, direction, no, k)
            s0 = np.asarray(central.Sk(k=k, format="array"))
            k0 = np.asarray(central.Hk(k=k, format="array")) + (fermi - vacuum) * s0
            d_s = sum(WEIGHTS[step] * np.asarray(hamiltonians[step].Sk(k=k, format="array"))
                      for step in WEIGHTS) / (12 * H_ANG)
            propagated, residual = _propagate((d_s - left - right) / 2,
                                               (d_s - left - right) / 2, s0, k0)
            adjoint = float(np.linalg.norm(left - right.conj().T)
                            / max(np.linalg.norm(left), np.finfo(float).tiny))
            passed = (propagated < LIMIT_EV_PER_ANG and adjoint < ADJOINT_REL_TOL
                      and residual < SOLVE_RESIDUAL_TOL)
            over += int(not passed)
            if propagated > worst:
                worst, worst_k = propagated, list(map(float, k))
            if ik < 3:
                rows.append({"k_reduced": list(map(float, k)),
                             "propagated_ev_per_ang": propagated,
                             "adjoint_relative": adjoint,
                             "solve_residual_relative": residual})
                saved[f"S_L_k{ik}"], saved[f"S_R_k{ik}"], saved[f"D_S_k{ik}"] = left, right, d_s
        artifact = OUTPUT / f"{basis}_{direction}_connections.npz"
        np.savez_compressed(artifact, **saved)
        cases.append({"basis": basis, "direction": direction,
                      "verdict": "PASS" if not over else "NO_GO",
                      "diagnostic_k": rows,
                      "dense_audit": {"k_count": len(dense), "k_over_limit": over,
                                      "worst_propagated_ev_per_ang": worst,
                                      "worst_k_reduced": worst_k},
                      "artifact": str(artifact), "artifact_sha256": file_sha256(artifact)})
        print(basis, direction, cases[-1]["verdict"], flush=True)
    passed = all(case["verdict"] == "PASS" for case in cases)
    report = {"schema": "full_basis_connection_onlys_v3",
              "gate": "full_basis_connection_v3", "generated_at": datetime.now(timezone.utc).isoformat(),
              "verdict": "PASS" if passed else "NO_GO", "stencil": "five-point, same as D_S",
              "thresholds": {"propagated_ev_per_ang": LIMIT_EV_PER_ANG,
                             "adjoint_relative": ADJOINT_REL_TOL,
                             "solve_residual_relative": SOLVE_RESIDUAL_TOL},
              "historical_v1_v2": "NO_GO unchanged", "cases": cases,
              "claim_scope": "nested-basis one-sided connections; not Delta_out/full-KS"}
    path = OUTPUT / "full_basis_connection_onlys_v3.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"full basis connection v3: {report['verdict']} -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
