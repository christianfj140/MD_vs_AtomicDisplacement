#!/usr/bin/env python3
"""Hermitian semantic closure of the embedded captured exterior response."""

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
from certify_delta_out_closure import CASES, production_connections  # noqa: E402
from certify_full_basis_connection_semantic_v5 import OUTPUT as CONNECTION_ROOT, solve  # noqa: E402
from certify_production_basis_connection import LIMIT_EV_PER_ANG  # noqa: E402
from certify_support_exact_connection import _propagate  # noqa: E402
from diagnose_delta_out_subspace import EQUIVALENCE_TOL, complement_selectors, decompose, shell  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL, _selector  # noqa: E402
from run_nested_basis_ladder import _rms  # noqa: E402
from run_nested_basis_sentinel import _matrix_data, _production_run  # noqa: E402

ROOT = REPO_ROOT / "Comparison/results/epc"
OUTPUT = ROOT / "captured_exterior_response_embedded_v3"


def captured(s, k0, left, right, selector):
    s_a, k_a = selector.conj().T @ s @ selector, selector.conj().T @ k0 @ selector
    x_b, r1 = solve(s, k0); y_b, r2 = solve(s, right)
    left_a, right_a = selector.conj().T @ left @ selector, selector.conj().T @ right @ selector
    x_a, r3 = solve(s_a, k_a); y_a, r4 = solve(s_a, right_a)
    value = (selector.conj().T @ left @ x_b @ selector - left_a @ x_a
             + selector.conj().T @ k0 @ y_b @ selector - k_a @ y_a)
    return value, max(r1, r2, r3, r4)


def main() -> int:
    import sisl

    previous = json.loads((ROOT / "captured_exterior_response_embedded_v2/captured_exterior_response_embedded_v2.json").read_text())
    if previous["verdict"] != "NO_GO":
        raise RuntimeError("v3 expects preserved embedded v2 NO_GO")
    prod_atom = sisl.get_sile(str(PROD_CENTRAL / "C.ion.xml")).read_basis()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for basis, direction in CASES:
        _, h, fermi, vacuum, atom = _matrix_data(_runs(basis, direction)[0])
        selector, _ = _selector(prod_atom, atom, h.geometry.na)
        complement, c_indices = complement_selectors(selector)
        groups = [shell(atom.orbitals[index % atom.no]) for index in c_indices]
        connection = np.load(CONNECTION_ROOT / f"{basis}_{direction}_connections.npz")
        raw = np.load(ROOT / f"captured_exterior_response_embedded_v2/{basis}_{direction}.npz")
        prod_connection = production_connections(direction, connection["k_points"])
        _, prod_h, prod_fermi, prod_vacuum, _ = _matrix_data(_production_run(direction, 0))
        values, modal_values, wd_values, retained_values = [], [], [], []
        worst_eq = worst_herm = worst_bridge = max_residual = 0.0
        for ik, k in enumerate(connection["k_points"]):
            s_raw = np.asarray(h.Sk(k=k, format="array"))
            k_raw = np.asarray(h.Hk(k=k, format="array")) + (fermi - vacuum) * s_raw
            s, k0 = (s_raw + s_raw.conj().T) / 2, (k_raw + k_raw.conj().T) / 2
            left, right = connection["S_L"][ik], connection["S_R"][ik]
            embedded, residual = captured(s, k0, left, right, selector)
            modal, pieces, lambdas, vectors, _, _, internal = decompose(
                s, k0, right, selector, complement)
            equivalence = np.linalg.norm(modal - embedded) / max(np.linalg.norm(embedded), np.finfo(float).tiny)
            s_a = selector.conj().T @ s @ selector
            herm_effect = _rms(embedded - raw["captured_exterior_response_embedded"][ik], s_a)[0]
            left_prod, right_prod = prod_connection[ik]
            s_prod_raw = np.asarray(prod_h.Sk(k=k, format="array"))
            s_prod = (s_prod_raw + s_prod_raw.conj().T) / 2
            k_prod_raw = np.asarray(prod_h.Hk(k=k, format="array")) + (prod_fermi - prod_vacuum) * s_prod_raw
            k_prod = (k_prod_raw + k_prod_raw.conj().T) / 2
            bridge, bridge_residual = _propagate(
                left_prod - selector.conj().T @ left @ selector,
                right_prod - selector.conj().T @ right @ selector, s_prod, k_prod)
            positive = np.asarray([_rms(piece, s_a)[0] for piece in pieces])
            d_weights = np.asarray([sum(abs(vectors[j, mode]) ** 2
                                        for j, group in enumerate(groups) if group == "3d_polarization")
                                    for mode in range(len(lambdas))])
            wd_values.append(float(positive @ d_weights / max(positive.sum(), np.finfo(float).tiny)))
            retained = pieces[lambdas / lambdas[-1] > 1e-3].sum(axis=0)
            retained_values.append(_rms(retained, s_a)[0] / max(_rms(modal, s_a)[0], np.finfo(float).tiny))
            values.append(embedded); modal_values.append(modal)
            worst_eq, worst_herm, worst_bridge = max(worst_eq, equivalence), max(worst_herm, herm_effect), max(worst_bridge, bridge)
            max_residual = max(max_residual, residual, bridge_residual, internal)
        identity_pass = worst_eq < EQUIVALENCE_TOL
        bridge_pass = worst_bridge < LIMIT_EV_PER_ANG
        herm_pass = worst_herm < LIMIT_EV_PER_ANG
        artifact = OUTPUT / f"{basis}_{direction}.npz"
        np.savez_compressed(artifact, k_points=connection["k_points"],
                            captured_exterior_response_embedded=np.asarray(values),
                            modal_sum=np.asarray(modal_values), d_weight_positive=np.asarray(wd_values),
                            retained_above_1e_3=np.asarray(retained_values))
        cases.append({"basis": basis, "direction": direction,
                      "verdict": "PASS" if identity_pass and bridge_pass and herm_pass else "NO_GO",
                      "modal_identity": {"verdict": "PASS" if identity_pass else "NO_GO",
                                         "threshold_relative": EQUIVALENCE_TOL, "worst_relative": worst_eq},
                      "bridge": {"verdict": "PASS" if bridge_pass else "NO_GO",
                                 "threshold_ev_per_ang": LIMIT_EV_PER_ANG, "worst_ev_per_ang": worst_bridge},
                      "hermitianization_effect": {"verdict": "PASS" if herm_pass else "NO_GO",
                                                  "threshold_ev_per_ang": LIMIT_EV_PER_ANG,
                                                  "worst_ev_per_ang": worst_herm},
                      "d_character_positive_weight": {"min": min(wd_values),
                                                      "median": float(np.median(wd_values)), "max": max(wd_values)},
                      "retained_above_relative_lambda_1e_3": {"min": min(retained_values),
                                                              "median": float(np.median(retained_values)),
                                                              "max": max(retained_values)},
                      "max_numerical_residual": max_residual,
                      "artifact": str(artifact), "artifact_sha256": file_sha256(artifact)})
        print(basis, direction, cases[-1]["verdict"])
    passed = all(case["verdict"] == "PASS" for case in cases)
    report = {"schema": "captured_exterior_response_embedded_v3",
              "gate": "embedded_connection_bridge_and_modal_identity_v3",
              "verdict": "PASS" if passed else "NO_GO",
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "method": "Hermitian S/K plus the same embedded B connection on both sides",
              "status_note": "corrected-method confirmation after observing v2 TSHS Hermitian floors",
              "thresholds": {"identity_relative": EQUIVALENCE_TOL,
                             "bridge_and_hermitianization_ev_per_ang": LIMIT_EV_PER_ANG},
              "cases": cases,
              "historical_status": "delta_out_closure_v1 and modal diagnostics v1/v2 remain NO_GO",
              "claim_scope": "semantic modal closure and polarization hypothesis; not Delta_out/full-KS"}
    path = OUTPUT / "captured_exterior_response_embedded_v3.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"captured exterior response embedded v3: {report['verdict']} -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
