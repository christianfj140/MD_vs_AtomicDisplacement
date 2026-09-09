#!/usr/bin/env python3
"""Recheck final QZ reference increments on the actual 20x20 TSHS k mesh."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts")]

from certify_production_basis_connection import _connections  # noqa: E402
from cross_basis_projection_preflight import _basis_geometry  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL, _selector  # noqa: E402
from qz_nested_identity_preflight import DEFAULT_OUTPUT as QZ_PREFLIGHT  # noqa: E402
from run_displaced_projection_sentinel import H_ANG, STEPS  # noqa: E402
from run_nested_basis_ladder import IDENTITY_ROOT, LIMITS, _rms  # noqa: E402
from run_nested_basis_sentinel import _matrix_data, _production_run  # noqa: E402

ROOT = REPO_ROOT / "Comparison/results/epc"
OUTPUT = ROOT / "reference_basis_convergence/dense_k_audit.json"
WEIGHTS = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}


def _runs(basis: str, direction: str) -> dict[int, Path]:
    paths = {0: (QZ_PREFLIGHT / "runs" / basis if basis == "n_qzp_d12_14" else
                 ROOT / "qz_convergence_followup/runs" / basis / direction / "central" if basis == "n_qzp_d10_12" else
                 IDENTITY_ROOT / basis)}
    for step in (-2, -1, 1, 2):
        tag = f"{'m' if step < 0 else 'p'}{abs(step)}"
        if basis == "n_qzp_d12_14" and direction == "C1_x":
            path = ROOT / "qz_cardinality_sentinel/runs/C1_x" / tag
        elif basis.startswith("n_qzp"):
            path = ROOT / "qz_convergence_followup/runs" / basis / direction / tag
        else:
            path = ROOT / "nested_basis_ladder/runs" / basis / direction / tag
        paths[step] = path
    return paths


def _covariant(basis: str, direction: str, k_points, connection) -> list[tuple[np.ndarray, np.ndarray]]:
    import sisl

    prod_atom = sisl.get_sile(str(PROD_CENTRAL / "C.ion.xml")).read_basis()
    prepared = {}
    for step, path in _runs(basis, direction).items():
        _, h, fermi, vacuum, atom = _matrix_data(path)
        selector, _ = _selector(prod_atom, atom, h.geometry.na)
        prepared[step] = h, fermi, vacuum, selector
    _, h_a, _, _, _ = _matrix_data(_production_run(direction, 0))
    result = []
    for ik, k in enumerate(k_points):
        matrices = {}
        for step, (h, fermi, vacuum, selector) in prepared.items():
            s = np.asarray(h.Sk(k=k, format="array"))
            matrices[step] = selector.T @ (np.asarray(h.Hk(k=k, format="array")) + (fermi - vacuum) * s) @ selector
        derivative = sum(weight * matrices[step] for step, weight in WEIGHTS.items()) / (12 * H_ANG)
        s_a, k0 = np.asarray(h_a.Sk(k=k, format="array")), matrices[0]
        left, right = connection[ik]
        result.append((derivative - left @ np.linalg.solve(s_a, k0) - k0 @ np.linalg.solve(s_a, right), s_a))
    return result


def main() -> int:
    import sisl

    label, h_a, _, _, atom = _matrix_data(PROD_CENTRAL)
    k_points = sisl.get_sile(str(QZ_PREFLIGHT / "runs/n_qzp_d12_14/n_qzp_d12_14.TSHS")).read_brillouinzone(trs=False).k
    shape = tuple(int(x) for x in sisl.get_sile(str(PROD_CENTRAL / f"{label}.VT")).read_grid().shape)
    base = _basis_geometry(h_a.geometry, PROD_CENTRAL / "C.ion.xml")
    dvolume = abs(np.linalg.det(base.cell)) / np.prod(shape)
    data = {}
    for direction in ("C1_x", "C1_z"):
        connection = _connections(direction, H_ANG, base, atom, shape, dvolume, k_points)
        data[("n_tzp_d12_14", direction)] = _covariant("n_tzp_d12_14", direction, k_points, connection)
        data[("n_qzp_d12_14", direction)] = _covariant("n_qzp_d12_14", direction, k_points, connection)
        if direction == "C1_x":
            data[("n_qzp_d10_12", direction)] = _covariant("n_qzp_d10_12", direction, k_points, connection)
    specs = (("cardinality_C1_x", "n_qzp_d12_14", "n_tzp_d12_14", "C1_x"),
             ("cardinality_C1_z", "n_qzp_d12_14", "n_tzp_d12_14", "C1_z"),
             ("range_C1_x", "n_qzp_d12_14", "n_qzp_d10_12", "C1_x"))
    comparisons = []
    for name, upper, lower, direction in specs:
        rows = []
        for k, (high, overlap), (low, _) in zip(k_points, data[(upper, direction)], data[(lower, direction)]):
            rms, maximum = _rms(high - low, overlap)
            signal, _ = _rms(high, overlap)
            relative = rms / max(signal, np.finfo(float).tiny)
            passed = rms < LIMITS["rms_ev_per_ang"] and relative < LIMITS["relative"] and maximum < LIMITS["max_ev_per_ang"]
            rows.append({"k_reduced": k.tolist(), "rms_ev_per_ang": rms, "relative": relative,
                         "max_ev_per_ang": maximum, "verdict": "PASS" if passed else "NO_GO"})
        worst = max(rows, key=lambda row: max(row["rms_ev_per_ang"] / LIMITS["rms_ev_per_ang"],
                                               row["relative"] / LIMITS["relative"], row["max_ev_per_ang"] / LIMITS["max_ev_per_ang"]))
        comparisons.append({"comparison": name, "upper": upper, "lower": lower, "direction": direction,
                            "verdict": "PASS" if all(row["verdict"] == "PASS" for row in rows) else "NO_GO",
                            "worst_case": worst, "k_results": rows})
    passed = all(item["verdict"] == "PASS" for item in comparisons)
    report = {"schema": "reference_basis_convergence_dense_k_audit_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
              "verdict": "PASS" if passed else "NO_GO", "gate": "reference_basis_convergence_dense_k_audit",
              "k_mesh": [20, 20, 1], "k_count": len(k_points), "k_source": "TSHS.read_brillouinzone(trs=False)",
              "thresholds": LIMITS, "comparisons": comparisons,
              "claim_scope": "dense-k audit of the existing fixed-PROD-SZ reference PASS; no new SIESTA, Delta_out, or full-KS claim"}
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"dense-k reference audit: {report['verdict']} -> {OUTPUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
