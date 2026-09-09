#!/usr/bin/env python3
"""Estimate and close PROD-SZ Delta_out inside the converged nested basis ladder."""

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
from certify_full_basis_connection import WEIGHTS  # noqa: E402
from certify_full_basis_connection_semantic_v5 import OUTPUT as CONNECTION_ROOT, solve  # noqa: E402
from certify_support_exact_connection import _reconstruct, _selector as atom_mask  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL, _selector  # noqa: E402
from onlys_semantic_preflight import DEFAULT_OUTPUT as ONLYS_ROOT  # noqa: E402
from run_displaced_projection_sentinel import H_ANG  # noqa: E402
from run_nested_basis_ladder import LIMITS, _rms  # noqa: E402
from run_nested_basis_sentinel import _matrix_data, _production_run  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/delta_out_closure"
CASES = (("n_tzp_d12_14", "C1_x"), ("n_tzp_d12_14", "C1_z"),
         ("n_qzp_d12_14", "C1_x"), ("n_qzp_d12_14", "C1_z"),
         ("n_qzp_d10_12", "C1_x"))


def covariant(d_k, k0, overlap, left, right):
    x, r1 = solve(overlap, k0)
    y, r2 = solve(overlap, right)
    return d_k - left @ x - k0 @ y, max(r1, r2)


def delta_out(d_k_b, k_b, s_b, left_b, right_b, selector,
              s_a, left_a, right_a):
    d_k_a, k_a = selector.conj().T @ d_k_b @ selector, selector.conj().T @ k_b @ selector
    delta_a, r1 = covariant(d_k_a, k_a, s_a, left_a, right_a)
    delta_b, r2 = covariant(d_k_b, k_b, s_b, left_b, right_b)
    g_a = selector.conj().T @ delta_b @ selector
    return delta_a - g_a, delta_a, g_a, max(r1, r2)


def production_connections(direction, k_points):
    onlys = np.load(ONLYS_ROOT / "onlys_semantic_preflight_matrices.npz")
    runs = {step: _matrix_data(_production_run(direction, step))[1] for step in WEIGHTS}
    _, central, _, _, _ = _matrix_data(_production_run(direction, 0))
    mask, result = atom_mask(central.geometry), []
    for ik, k in enumerate(k_points):
        d_s = sum(WEIGHTS[step] * np.asarray(runs[step].Sk(k=k, format="array"))
                  for step in WEIGHTS) / (12 * H_ANG)
        left, right = _reconstruct(d_s, mask)
        b_i = onlys[f"B_I_dense_{direction}_k{ik}"]
        result.append((left - b_i, right + b_i))
    return result


def main() -> int:
    import sisl

    connection_report = json.loads((CONNECTION_ROOT / "full_basis_connection_semantic_v5.json").read_text())
    if connection_report["verdict"] != "PASS":
        raise RuntimeError("full_basis_connection is not PASS")
    prod_atom = sisl.get_sile(str(PROD_CENTRAL / "C.ion.xml")).read_basis()
    data, case_reports, max_solve = {}, [], 0.0
    for basis, direction in CASES:
        connection_path = CONNECTION_ROOT / f"{basis}_{direction}_connections.npz"
        connection = np.load(connection_path)
        k_points = connection["k_points"]
        prod_connection = production_connections(direction, k_points)
        prepared = {}
        for step, path in _runs(basis, direction).items():
            _, h, fermi, vacuum, atom = _matrix_data(path)
            prepared[step] = h, fermi, vacuum
        central, fermi, vacuum = prepared[0]
        selector, _ = _selector(prod_atom, atom, central.geometry.na)
        _, prod_h, _, _, _ = _matrix_data(_production_run(direction, 0))
        values, deltas, references = [], [], []
        for ik, k in enumerate(k_points):
            matrices = {}
            for step, (h, ef, vac) in prepared.items():
                s = np.asarray(h.Sk(k=k, format="array"))
                matrices[step] = np.asarray(h.Hk(k=k, format="array")) + (ef - vac) * s
            d_k = sum(WEIGHTS[step] * matrices[step] for step in WEIGHTS) / (12 * H_ANG)
            s_b = np.asarray(central.Sk(k=k, format="array"))
            s_a = np.asarray(prod_h.Sk(k=k, format="array"))
            left_a, right_a = prod_connection[ik]
            out, delta_a, g_a, residual = delta_out(
                d_k, matrices[0], s_b, connection["S_L"][ik], connection["S_R"][ik],
                selector, s_a, left_a, right_a)
            max_solve = max(max_solve, residual)
            values.append(out); deltas.append(delta_a); references.append(g_a)
        artifact = OUTPUT / f"{basis}_{direction}.npz"
        OUTPUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact, k_points=k_points, delta_out=np.asarray(values),
                            delta_A=np.asarray(deltas), G_A=np.asarray(references))
        data[(basis, direction)] = (np.asarray(values), np.asarray(deltas), np.asarray(references),
                                    k_points, prod_h)
        case_reports.append({"basis": basis, "direction": direction,
                             "artifact": str(artifact), "artifact_sha256": file_sha256(artifact)})

    specs = (("cardinality_C1_x", "n_qzp_d12_14", "n_tzp_d12_14", "C1_x"),
             ("cardinality_C1_z", "n_qzp_d12_14", "n_tzp_d12_14", "C1_z"),
             ("range_C1_x", "n_qzp_d12_14", "n_qzp_d10_12", "C1_x"))
    comparisons = []
    for name, upper, lower, direction in specs:
        high, response, _, k_points, prod_h = data[(upper, direction)]
        low = data[(lower, direction)][0]
        rows = []
        for ik, k in enumerate(k_points):
            overlap = np.asarray(prod_h.Sk(k=k, format="array"))
            rms, maximum = _rms(high[ik] - low[ik], overlap)
            signal, _ = _rms(response[ik], overlap)
            relative = rms / max(signal, np.finfo(float).tiny)
            passed = rms < LIMITS["rms_ev_per_ang"] and relative < LIMITS["relative"] \
                and maximum < LIMITS["max_ev_per_ang"]
            rows.append({"k_reduced": list(map(float, k)), "rms_ev_per_ang": rms,
                         "relative_to_reference_response": relative,
                         "max_ev_per_ang": maximum, "verdict": "PASS" if passed else "NO_GO"})
        worst = max(rows, key=lambda row: max(row["rms_ev_per_ang"] / LIMITS["rms_ev_per_ang"],
                                               row["relative_to_reference_response"] / LIMITS["relative"],
                                               row["max_ev_per_ang"] / LIMITS["max_ev_per_ang"]))
        comparisons.append({"comparison": name, "verdict": "PASS" if all(
            row["verdict"] == "PASS" for row in rows) else "NO_GO", "worst_case": worst,
                            "k_results": rows})
    closure_pass = all(row["verdict"] == "PASS" for row in comparisons)

    full_connection_uncertainty = max(
        case["dense_audit"]["worst_closure_propagated_ev_per_ang"]
        + case["dense_audit"]["worst_symmetrization_propagated_ev_per_ang"]
        for case in connection_report["cases"])
    b_i_report = json.loads((REPO_ROOT / "Comparison/results/epc/b_i_fourier_bessel_semantic_v5/b_i_fourier_bessel_semantic_v5.json").read_text())
    prod_uncertainty = max(case["V5_D_onlyS"]["dense_audit"]["worst_propagated_ev_per_ang"]
                           for case in b_i_report["cases"])
    card = max(row["worst_case"]["rms_ev_per_ang"] for row in comparisons if "cardinality" in row["comparison"])
    radial = next(row["worst_case"]["rms_ev_per_ang"] for row in comparisons if "range" in row["comparison"])
    connection_uncertainty = full_connection_uncertainty + prod_uncertainty
    tail = card + radial + connection_uncertainty
    negligible_rows = []
    for direction in ("C1_x", "C1_z"):
        values, _, references, k_points, prod_h = data[("n_qzp_d12_14", direction)]
        for ik, k in enumerate(k_points):
            overlap = np.asarray(prod_h.Sk(k=k, format="array"))
            magnitude, _ = _rms(values[ik], overlap)
            reference, _ = _rms(references[ik], overlap)
            upper = magnitude + tail
            negligible_rows.append({"direction": direction, "k_reduced": list(map(float, k)),
                                     "delta_out_rms_ev_per_ang": magnitude,
                                     "operational_upper_ev_per_ang": upper,
                                     "relative_to_G_A": upper / max(reference, np.finfo(float).tiny),
                                     "verdict": "PASS" if upper < 0.005 and upper / max(reference, np.finfo(float).tiny) < 0.01 else "NO_GO"})
    negligible_pass = all(row["verdict"] == "PASS" for row in negligible_rows)
    report = {"schema": "delta_out_closure_v1", "gate": "delta_out_closure",
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "verdict": "PASS" if closure_pass else "NO_GO",
              "delta_out_negligible": "PASS" if negligible_pass else "NO_GO",
              "definition": "Delta_A^[B] - E^dagger Delta_B E, same self-consistent B operator",
              "thresholds": LIMITS, "max_solve_backward_error": max_solve,
              "comparisons": comparisons, "basis_artifacts": case_reports,
              "operational_tail": {"cardinality_ev_per_ang": card,
                                   "range_ev_per_ang": radial,
                                   "connection_ev_per_ang": connection_uncertainty,
                                   "sum_ev_per_ang": tail},
              "negligibility": {"threshold_ev_per_ang": 0.005,
                                "relative_threshold": 0.01,
                                "worst_case": max(negligible_rows, key=lambda row: max(
                                    row["operational_upper_ev_per_ang"] / 0.005,
                                    row["relative_to_G_A"] / 0.01)),
                                "k_results": negligible_rows},
              "claim_scope": "basis-converged operational Delta_out in fixed PROD-SZ; full-KS wording remains separate"}
    path = OUTPUT / "delta_out_closure.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"delta_out closure: {report['verdict']}; negligible: {report['delta_out_negligible']} -> {path}")
    return 0 if closure_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
