#!/usr/bin/env python3
"""Adjudicate the enriched-basis ladder after the PROD-SZ connection correction."""

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
from cross_basis_projection_preflight import K_POINTS  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_displaced_projection_sentinel import DIRECTIONS  # noqa: E402
from run_nested_basis_ladder import BASIS_ORDER, COMPARE, IDENTITY_ROOT, LIMITS, DEFAULT_OUTPUT as LADDER_ROOT, _rms  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

CONNECTION_ROOT = REPO_ROOT / "Comparison/results/epc/production_basis_connection"
DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/reference_basis_convergence_covariant"


def evaluate(output: Path) -> dict:
    connection_report = json.loads((CONNECTION_ROOT / "production_basis_connection.json").read_text(encoding="utf-8"))
    if connection_report["verdict"] != "PASS":
        raise RuntimeError("production_basis_connection is not PASS")
    connection = np.load(CONNECTION_ROOT / "production_basis_connection_matrices.npz")
    covariant, basis_reports = {}, []
    for basis in BASIS_ORDER:
        _, h_b, fermi_b, vacuum_b, atom_b = _matrix_data(IDENTITY_ROOT / basis)
        _, h_a, _, _, atom_a = _matrix_data(PROD_CENTRAL)
        from nested_basis_identity_preflight import _selector
        selector, _ = _selector(atom_a, atom_b, h_a.geometry.na)
        covariant[basis] = {}
        for direction in DIRECTIONS:
            raw = np.load(LADDER_ROOT / f"{basis}_{direction}_derivative.npz")
            saved, rows = {}, []
            covariant[basis][direction] = {}
            for ik, k in enumerate(K_POINTS):
                s_a = np.asarray(h_a.Sk(k=k, format="array"))
                s_b = np.asarray(h_b.Sk(k=k, format="array"))
                h_abs = np.asarray(h_b.Hk(k=k, format="array")) + fermi_b * s_b
                k0 = selector.T @ (h_abs - vacuum_b * s_b) @ selector
                d_k = raw[f"D5_K_prod_subblock_k{ik}"]
                left = connection[f"S_L_{direction}_k{ik}"]
                right = connection[f"S_R_{direction}_k{ik}"]
                delta = d_k - left @ np.linalg.solve(s_a, k0) - k0 @ np.linalg.solve(s_a, right)
                covariant[basis][direction][ik] = (delta, s_a)
                saved[f"Delta_covariant_k{ik}"] = delta
                saved[f"K0_prod_subblock_k{ik}"] = k0
                rows.append({"k_reduced": list(k), "covariant_rms_ev_per_ang": _rms(delta, s_a)[0]})
            artifact = output / f"{basis}_{direction}_covariant.npz"
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(artifact, **saved)
            basis_reports.append({"basis": basis, "direction": direction, "artifact": str(artifact), "artifact_sha256": file_sha256(artifact), "k_results": rows})

    comparisons, worst = [], None
    for name, (upper, lower, gated) in COMPARE.items():
        for direction in DIRECTIONS:
            for ik, k in enumerate(K_POINTS):
                high, overlap = covariant[upper][direction][ik]
                low, _ = covariant[lower][direction][ik]
                rms, maximum = _rms(high - low, overlap)
                signal, _ = _rms(high, overlap)
                relative = rms / max(signal, np.finfo(float).tiny)
                passed = rms < LIMITS["rms_ev_per_ang"] and relative < LIMITS["relative"] and maximum < LIMITS["max_ev_per_ang"]
                row = {"comparison": name, "upper": upper, "lower": lower, "direction": direction, "k_reduced": list(k), "rms_ev_per_ang": rms, "relative": relative, "max_ev_per_ang": maximum, "gated": gated, "verdict": "PASS" if passed else "NO_GO"}
                comparisons.append(row)
                if gated:
                    score = max(rms / LIMITS["rms_ev_per_ang"], relative / LIMITS["relative"], maximum / LIMITS["max_ev_per_ang"])
                    if worst is None or score > worst[0]:
                        worst = (score, row)
    passed = all(row["verdict"] == "PASS" for row in comparisons if row["gated"])
    report = {
        "schema": "reference_basis_convergence_covariant_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO", "gate": "reference_basis_convergence_covariant",
        "raw_gate": "reference_basis_convergence_raw=NO_GO (unchanged)",
        "connection_gate": "production_basis_connection=PASS",
        "observable": "D5 K_A^[B] - S_L,A S_A^-1 K_A^[B](0) - K_A^[B](0) S_A^-1 S_R,A",
        "thresholds": LIMITS, "basis_artifacts": basis_reports, "comparisons": comparisons,
        "worst_case": worst[1] if worst else None,
        "claim_scope": "fixed PROD-SZ covariant-reference convergence only; not Delta_out/full-KS",
    }
    path = output / "reference_basis_convergence_covariant.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"reference basis convergence covariant: {report['verdict']} -> {path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    return 0 if evaluate(args.output)["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
