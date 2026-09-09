#!/usr/bin/env python3
"""Close the modal identity using B's embedded PROD connection consistently."""

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
from certify_delta_out_closure import CASES, delta_out, production_connections  # noqa: E402
from certify_full_basis_connection import WEIGHTS  # noqa: E402
from certify_full_basis_connection_semantic_v5 import OUTPUT as CONNECTION_ROOT  # noqa: E402
from certify_production_basis_connection import LIMIT_EV_PER_ANG  # noqa: E402
from certify_support_exact_connection import _propagate  # noqa: E402
from diagnose_delta_out_subspace import (  # noqa: E402
    EQUIVALENCE_TOL, complement_selectors, decompose, shell,
)
from nested_basis_identity_preflight import PROD_CENTRAL, _selector  # noqa: E402
from run_displaced_projection_sentinel import H_ANG  # noqa: E402
from run_nested_basis_ladder import _rms  # noqa: E402
from run_nested_basis_sentinel import _matrix_data, _production_run  # noqa: E402

ROOT = REPO_ROOT / "Comparison/results/epc"
OUTPUT = ROOT / "captured_exterior_response_embedded_v2"


def main() -> int:
    import sisl

    old = json.loads((ROOT / "delta_out_subspace_diagnostic/delta_out_subspace_diagnostic.json").read_text())
    if old["verdict"] != "NO_GO":
        raise RuntimeError("v2 expects the preserved modal diagnostic v1 NO_GO")
    prod_atom = sisl.get_sile(str(PROD_CENTRAL / "C.ion.xml")).read_basis()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for basis, direction in CASES:
        runs = _runs(basis, direction)
        _, central, fermi, vacuum, atom = _matrix_data(runs[0])
        selector, _ = _selector(prod_atom, atom, central.geometry.na)
        complement, c_indices = complement_selectors(selector)
        groups = [shell(atom.orbitals[index % atom.no]) for index in c_indices]
        connection = np.load(CONNECTION_ROOT / f"{basis}_{direction}_connections.npz")
        prod_connection = production_connections(direction, connection["k_points"])
        _, prod_h, _, _, _ = _matrix_data(_production_run(direction, 0))
        prepared = {}
        for step, path in runs.items():
            _, h, ef, vac, _ = _matrix_data(path)
            prepared[step] = h, ef, vac
        embedded_values, modal_values, bridge_values, wd_values, retained = [], [], [], [], []
        worst_equivalence = worst_bridge = max_solve = 0.0
        worst_eq_k = worst_bridge_k = None
        for ik, k in enumerate(connection["k_points"]):
            matrices = {}
            for step, (h, ef, vac) in prepared.items():
                s = np.asarray(h.Sk(k=k, format="array"))
                matrices[step] = np.asarray(h.Hk(k=k, format="array")) + (ef - vac) * s
            d_k = sum(WEIGHTS[step] * matrices[step] for step in WEIGHTS) / (12 * H_ANG)
            s_b = np.asarray(central.Sk(k=k, format="array"))
            s_a = selector.conj().T @ s_b @ selector
            left_b, right_b = connection["S_L"][ik], connection["S_R"][ik]
            left_emb = selector.conj().T @ left_b @ selector
            right_emb = selector.conj().T @ right_b @ selector
            embedded, _, _, residual = delta_out(
                d_k, matrices[0], s_b, left_b, right_b, selector,
                s_a, left_emb, right_emb)
            modal, pieces, values, vectors, _, _, internal = decompose(
                s_b, matrices[0], right_b, selector, complement)
            equivalence = float(np.linalg.norm(modal - embedded)
                                / max(np.linalg.norm(embedded), np.finfo(float).tiny))
            left_prod, right_prod = prod_connection[ik]
            _, prod_central, prod_fermi, prod_vacuum, _ = _matrix_data(_production_run(direction, 0))
            s_prod = np.asarray(prod_central.Sk(k=k, format="array"))
            k_prod = np.asarray(prod_central.Hk(k=k, format="array")) + (prod_fermi - prod_vacuum) * s_prod
            bridge, bridge_residual = _propagate(left_prod - left_emb, right_prod - right_emb,
                                                  s_prod, k_prod)
            weights_d = np.asarray([sum(abs(vectors[j, mode]) ** 2
                                        for j, group in enumerate(groups) if group == "3d_polarization")
                                    for mode in range(len(values))])
            positive = np.asarray([_rms(piece, s_a)[0] for piece in pieces])
            wd = float(positive @ weights_d / max(positive.sum(), np.finfo(float).tiny))
            retained_response = pieces[values / values[-1] > 1e-3].sum(axis=0)
            retained_ratio = _rms(retained_response, s_a)[0] / max(_rms(modal, s_a)[0], np.finfo(float).tiny)
            embedded_values.append(embedded); modal_values.append(modal); bridge_values.append(bridge)
            wd_values.append(wd); retained.append(retained_ratio)
            max_solve = max(max_solve, residual, bridge_residual, internal)
            if equivalence > worst_equivalence:
                worst_equivalence, worst_eq_k = equivalence, list(map(float, k))
            if bridge > worst_bridge:
                worst_bridge, worst_bridge_k = bridge, list(map(float, k))
        artifact = OUTPUT / f"{basis}_{direction}.npz"
        np.savez_compressed(artifact, k_points=connection["k_points"],
                            captured_exterior_response_embedded=np.asarray(embedded_values),
                            modal_sum=np.asarray(modal_values), bridge_ev_per_ang=np.asarray(bridge_values),
                            d_weight_positive=np.asarray(wd_values), retained_above_1e_3=np.asarray(retained))
        passed_identity = worst_equivalence < EQUIVALENCE_TOL
        passed_bridge = worst_bridge < LIMIT_EV_PER_ANG
        cases.append({"basis": basis, "direction": direction,
                      "verdict": "PASS" if passed_identity and passed_bridge else "NO_GO",
                      "modal_identity": {"verdict": "PASS" if passed_identity else "NO_GO",
                                         "threshold_relative": EQUIVALENCE_TOL,
                                         "worst_relative": worst_equivalence,
                                         "worst_k_reduced": worst_eq_k},
                      "embedded_connection_bridge": {"verdict": "PASS" if passed_bridge else "NO_GO",
                                                     "historical_threshold_ev_per_ang": LIMIT_EV_PER_ANG,
                                                     "worst_ev_per_ang": worst_bridge,
                                                     "worst_k_reduced": worst_bridge_k},
                      "d_character_positive_weight": {"definition": "sum ||X_n||_RMS w_n,d / sum ||X_n||_RMS",
                                                      "min": min(wd_values), "median": float(np.median(wd_values)),
                                                      "max": max(wd_values)},
                      "near_null_diagnostic": {"cutoff_relative_lambda": 1e-3, "gate": False,
                                               "retained_response_ratio_min": min(retained),
                                               "retained_response_ratio_median": float(np.median(retained)),
                                               "retained_response_ratio_max": max(retained)},
                      "max_numerical_residual": max_solve,
                      "artifact": str(artifact), "artifact_sha256": file_sha256(artifact)})
        print(basis, direction, cases[-1]["verdict"])
    passed = all(case["verdict"] == "PASS" for case in cases)
    report = {"schema": "captured_exterior_response_embedded_v2",
              "gate": "embedded_connection_bridge_and_modal_identity_v2",
              "verdict": "PASS" if passed else "NO_GO",
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "object": "captured_exterior_response_embedded(B)",
              "bridge_identity": "X_prod = X_embedded + delta_X_bridge",
              "thresholds": {"modal_identity_relative": EQUIVALENCE_TOL,
                             "bridge_ev_per_ang_historical_connection_budget": LIMIT_EV_PER_ANG},
              "cases": cases,
              "historical_status": "delta_out_closure_v1 and modal diagnostic v1 remain NO_GO",
              "claim_scope": "embedded semantic closure and causal diagnostic only; not Delta_out/full-KS"}
    path = OUTPUT / "captured_exterior_response_embedded_v2.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"captured exterior response embedded v2: {report['verdict']} -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
