#!/usr/bin/env python3
"""Diagnose finite-basis captured exterior response in the exact B=A+C split."""

from __future__ import annotations

import gzip
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
from cross_basis_projection_preflight import (  # noqa: E402
    K_POINTS, _basis_geometry, _bloch_values, _orbital_grid,
)
from nested_basis_identity_preflight import PROD_CENTRAL, _selector  # noqa: E402
from run_nested_basis_ladder import _rms  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402
from certify_support_exact_connection import _propagate  # noqa: E402

ROOT = REPO_ROOT / "Comparison/results/epc"
OUTPUT = ROOT / "delta_out_subspace_diagnostic"
EQUIVALENCE_TOL = 1e-10
CUTOFFS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


def complement_selectors(selector: np.ndarray):
    a_indices = np.flatnonzero(np.any(selector != 0, axis=1))
    c_indices = np.setdiff1d(np.arange(selector.shape[0]), a_indices)
    return np.eye(selector.shape[0])[:, c_indices], c_indices


def decompose(overlap, k0, right, selector, complement):
    s_a = selector.conj().T @ overlap @ selector
    s_ac = selector.conj().T @ overlap @ complement
    projection, residual = solve(s_a, s_ac)
    w = complement - selector @ projection
    s_perp = (w.conj().T @ overlap @ w)
    s_perp = (s_perp + s_perp.conj().T) / 2
    values, vectors = np.linalg.eigh(s_perp)
    if values[0] <= 0:
        raise ValueError(f"non-positive complement metric: {values[0]:.6g}")
    modes = w @ (vectors / np.sqrt(values))
    r = modes.conj().T @ right @ selector
    h = modes.conj().T @ k0 @ selector
    pieces = np.asarray([np.outer(rn.conj(), hn) + np.outer(hn.conj(), rn)
                         for rn, hn in zip(r, h)])
    orth_a = np.linalg.norm(selector.conj().T @ overlap @ modes)
    orth_c = np.linalg.norm(modes.conj().T @ overlap @ modes - np.eye(len(values)))
    return pieces.sum(axis=0), pieces, values, vectors, r, h, max(residual, orth_a, orth_c)


def shell(orbital) -> str:
    if orbital.l == 2:
        return "3d_polarization"
    if orbital.n == 3:
        return f"3{'s' if orbital.l == 0 else 'p'}_diffuse"
    return f"2{'s' if orbital.l == 0 else 'p'}_zeta{orbital.zeta}"


def transformations(size: int):
    rng = np.random.default_rng(20260906)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(size, size)))
    scaling = np.diag(np.linspace(0.7, 1.3, size))
    triangular = np.eye(size) + 0.03 * np.triu(np.ones((size, size)), 1)
    return {"orthogonal": orthogonal, "diagonal_scaling": scaling,
            "triangular": triangular}


def principal_angles():
    import sisl

    runs = {name: _runs(name, "C1_x")[0] for name in ("n_qzp_d10_12", "n_qzp_d12_14")}
    items = {}
    shape = tuple(int(x) for x in sisl.get_sile(str(PROD_CENTRAL / "prod_central.VT")).read_grid().shape)
    dvolume = None
    prod_atom = sisl.get_sile(str(PROD_CENTRAL / "C.ion.xml")).read_basis()
    for name, run in runs.items():
        _, h, _, _, atom = _matrix_data(run)
        geometry = _basis_geometry(h.geometry, run / "C.ion.xml")
        values, offsets = _orbital_grid(geometry, shape)
        selector, _ = _selector(prod_atom, atom, geometry.na)
        complement, _ = complement_selectors(selector)
        items[name] = h, geometry, values, offsets, selector, complement
        dvolume = abs(np.linalg.det(geometry.cell)) / np.prod(shape)
    rows = []
    for k in K_POINTS:
        normalized = {}
        for name, (h, geometry, values, offsets, selector, complement) in items.items():
            s = np.asarray(h.Sk(k=k, format="array"))
            s_a = selector.T @ s @ selector
            projection, _ = solve(s_a, selector.T @ s @ complement)
            w = complement - selector @ projection
            metric = (w.conj().T @ s @ w); metric = (metric + metric.conj().T) / 2
            lam, u = np.linalg.eigh(metric)
            normalized[name] = w @ (u / np.sqrt(lam))
        x10 = _bloch_values(items["n_qzp_d10_12"][2], items["n_qzp_d10_12"][3],
                            items["n_qzp_d10_12"][1].no, k)
        x12 = _bloch_values(items["n_qzp_d12_14"][2], items["n_qzp_d12_14"][3],
                            items["n_qzp_d12_14"][1].no, k)
        cross = np.asarray((x10.conj().T @ x12).toarray()) * dvolume
        overlap = normalized["n_qzp_d10_12"].conj().T @ cross @ normalized["n_qzp_d12_14"]
        sigma = np.linalg.svd(overlap, compute_uv=False)
        rows.append({"k_reduced": list(k), "sigma_min": float(sigma.min()),
                     "sigma_max": float(sigma.max()),
                     "theta_max_deg": float(np.degrees(np.arccos(np.clip(sigma.min(), -1, 1))))})
    return rows


def main() -> int:
    import sisl

    closure = json.loads((ROOT / "delta_out_closure/delta_out_closure.json").read_text())
    if closure["verdict"] != "NO_GO":
        raise RuntimeError("diagnostic expects preserved delta_out_closure_v1 NO_GO")
    prod_atom = sisl.get_sile(str(PROD_CENTRAL / "C.ion.xml")).read_basis()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table_path = OUTPUT / "mode_table.jsonl.gz"
    cases, saved, equivalence_failures = [], {}, []
    with gzip.open(table_path, "wt", encoding="utf-8") as table:
        for basis, direction in CASES:
            run = _runs(basis, direction)[0]
            _, hamiltonian, fermi, vacuum, atom = _matrix_data(run)
            selector, _ = _selector(prod_atom, atom, hamiltonian.geometry.na)
            complement, c_indices = complement_selectors(selector)
            groups = [shell(atom.orbitals[index % atom.no]) for index in c_indices]
            connection = np.load(CONNECTION_ROOT / f"{basis}_{direction}_connections.npz")
            captured = np.load(ROOT / f"delta_out_closure/{basis}_{direction}.npz")
            prod_connection = production_connections(direction, connection["k_points"])
            from run_nested_basis_sentinel import _production_run
            _, prod_h, prod_fermi, prod_vacuum, _ = _matrix_data(_production_run(direction, 0))
            rows, rank = [], {str(tau): [] for tau in CUTOFFS}
            worst_equivalence, worst_k, max_internal = 0.0, None, 0.0
            embedded_mismatch = 0.0
            gauge = {name: 0.0 for name in transformations(len(c_indices))}
            for ik, k in enumerate(connection["k_points"]):
                s = np.asarray(hamiltonian.Sk(k=k, format="array"))
                k0 = np.asarray(hamiltonian.Hk(k=k, format="array")) + (fermi - vacuum) * s
                total, pieces, values, vectors, r, hb, internal = decompose(
                    s, k0, connection["S_R"][ik], selector, complement)
                reference = captured["delta_out"][ik]
                left_a, right_a = prod_connection[ik]
                delta_left = selector.conj().T @ connection["S_L"][ik] @ selector - left_a
                delta_right = selector.conj().T @ connection["S_R"][ik] @ selector - right_a
                s_prod = np.asarray(prod_h.Sk(k=k, format="array"))
                k_prod = np.asarray(prod_h.Hk(k=k, format="array")) + (prod_fermi - prod_vacuum) * s_prod
                mismatch, _ = _propagate(delta_left, delta_right, s_prod, k_prod)
                embedded_mismatch = max(embedded_mismatch, mismatch)
                relative = float(np.linalg.norm(total - reference)
                                 / max(np.linalg.norm(reference), np.finfo(float).tiny))
                max_internal = max(max_internal, internal)
                if relative > worst_equivalence:
                    worst_equivalence, worst_k = relative, list(map(float, k))
                if relative >= EQUIVALENCE_TOL:
                    equivalence_failures.append(f"{basis}/{direction}/k{ik}: {relative:.6g}")
                s_a = selector.conj().T @ s @ selector
                shell_names = sorted(set(groups))
                for mode, (value, piece, rn, hn) in enumerate(zip(values, pieces, r, hb)):
                    weights = {name: float(sum(abs(vectors[j, mode]) ** 2
                                               for j, group in enumerate(groups) if group == name))
                               for name in shell_names}
                    table.write(json.dumps({"basis": basis, "direction": direction,
                                            "k_index": ik, "k_reduced": list(map(float, k)),
                                            "mode": mode, "lambda": float(value),
                                            "relative_lambda": float(value / values[-1]),
                                            "r_norm": float(np.linalg.norm(rn)),
                                            "h_norm_ev": float(np.linalg.norm(hn)),
                                            "X_rms_ev_per_ang": _rms(piece, s_a)[0],
                                            "shell_weights": weights}) + "\n")
                for tau in CUTOFFS:
                    subtotal = pieces[values / values[-1] > tau].sum(axis=0)
                    rank[str(tau)].append(_rms(subtotal, s_a)[0])
                if ik < 3:
                    for name, transform in transformations(len(c_indices)).items():
                        z = selector @ selector.T + complement @ transform @ complement.T
                        transformed, *_ = decompose(z.conj().T @ s @ z,
                                                    z.conj().T @ k0 @ z,
                                                    z.conj().T @ connection["S_R"][ik] @ z,
                                                    selector, complement)
                        gauge[name] = max(gauge[name], float(np.linalg.norm(transformed - total)
                            / max(np.linalg.norm(total), np.finfo(float).tiny)))
                if ik < 3:
                    rows.append({"k_reduced": list(map(float, k)),
                                 "lambda_min": float(values[0]),
                                 "lambda_max": float(values[-1]),
                                 "condition_complement": float(values[-1] / values[0]),
                                 "equivalence_relative": relative})
            cases.append({"basis": basis, "direction": direction,
                          "verdict": "PASS" if worst_equivalence < EQUIVALENCE_TOL else "NO_GO",
                          "worst_equivalence_relative": worst_equivalence,
                          "worst_k_reduced": worst_k, "max_internal_residual": max_internal,
                          "embedded_PROD_connection_mismatch_ev_per_ang": embedded_mismatch,
                          "diagnostic_k": rows, "gauge_invariance_relative": gauge,
                          "rank_sweep": {tau: {"min_rms_ev_per_ang": min(vals),
                                                "max_rms_ev_per_ang": max(vals)}
                                         for tau, vals in rank.items()}})
            saved[f"rank_sweep_{basis}_{direction}"] = np.asarray(
                [rank[str(tau)] for tau in CUTOFFS])
    angles = principal_angles()
    passed = not equivalence_failures
    arrays = OUTPUT / "delta_out_subspace_diagnostic.npz"
    np.savez_compressed(arrays, cutoffs=np.asarray(CUTOFFS), **saved)
    report = {"schema": "delta_out_subspace_diagnostic_v1",
              "gate": "delta_out_subspace_diagnostic", "verdict": "PASS" if passed else "NO_GO",
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "object_name": "captured_exterior_response(B)",
              "definition": "sum_n (r_n^dagger h_n + h_n^dagger r_n)",
              "equivalence_threshold_relative": EQUIVALENCE_TOL,
              "cases": cases, "equivalence_failures": equivalence_failures,
              "principal_angles_QZ_D10_12_vs_D12_14": {"gate": False, "k_results": angles},
              "rank_sweep": {"gate": False, "cutoffs_relative_lambda": list(CUTOFFS),
                             "warning": "diagnostic only; no truncation defines a promoted result"},
              "mode_table": str(table_path), "mode_table_sha256": file_sha256(table_path),
              "matrix_artifact": str(arrays), "matrix_artifact_sha256": file_sha256(arrays),
              "historical_status": "delta_out_closure_v1=NO_GO and negligible=NO_GO unchanged",
              "claim_scope": "subspace diagnosis only; no regularization, new DFT, or full-KS promotion"}
    path = OUTPUT / "delta_out_subspace_diagnostic.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"delta-out subspace diagnostic: {report['verdict']} -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
