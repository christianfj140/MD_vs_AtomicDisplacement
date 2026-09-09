#!/usr/bin/env python3
"""Run the single authorized N-QZP+D12/14 C1:x cardinality sentinel."""

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
from nested_basis_identity_preflight import COND_MAX, OVERLAP_TOL, PROD_CENTRAL, _selector  # noqa: E402
from qz_nested_identity_preflight import DEFAULT_OUTPUT as QZ_PREFLIGHT, QZ_BLOCK  # noqa: E402
from run_displaced_projection_sentinel import H_ANG, STEPS  # noqa: E402
from run_nested_basis_ladder import DEFAULT_OUTPUT as LADDER, IDENTITY_ROOT, LIMITS, _rms, _run_one  # noqa: E402
from run_nested_basis_sentinel import HERM_TOL, _matrix_data, _production_run  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/qz_cardinality_sentinel"
PARENT = "n_tzp_d12_14"
QZ = "n_qzp_d12_14"


def _parent_run(step: int) -> Path:
    if step == 0:
        return IDENTITY_ROOT / PARENT
    tag = f"{'m' if step < 0 else 'p'}{abs(step)}"
    return LADDER / "runs" / PARENT / "C1_x" / tag


def produce(output: Path, siesta: str) -> dict[int, Path]:
    preflight = json.loads((QZ_PREFLIGHT / "qz_nested_identity_preflight.json").read_text(encoding="utf-8"))
    if preflight["verdict"] != "PASS":
        raise RuntimeError("QZ_nested_identity_preflight is not PASS")
    import sisl

    xyz0 = sisl.get_sile(str(PROD_CENTRAL / "prod_central.TSHS")).read_geometry().xyz
    runs = {0: QZ_PREFLIGHT / "runs" / QZ}
    for step in (-2, -1, 1, 2):
        tag = f"{'m' if step < 0 else 'p'}{abs(step)}"
        xyz = xyz0.copy()
        xyz[0, 0] += step * H_ANG
        label = f"{QZ}_c1_x_{tag}"
        runs[step] = _run_one(output / "runs" / "C1_x" / tag, label, QZ_BLOCK, xyz, siesta)
    return runs


def evaluate(output: Path, runs: dict[int, Path]) -> dict:
    import sisl

    prod_atom = sisl.get_sile(str(PROD_CENTRAL / "C.ion.xml")).read_basis()
    connection = np.load(REPO_ROOT / "Comparison/results/epc/production_basis_connection/production_basis_connection_matrices.npz")
    matrices, points, failures = {}, [], []
    weights = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
    for step in STEPS:
        qz_label, h_qz, fermi, vacuum, qz_atom = _matrix_data(runs[step])
        _, h_tz, _, _, tz_atom = _matrix_data(_parent_run(step))
        prod_selector, _ = _selector(prod_atom, qz_atom, h_qz.geometry.na)
        tz_selector, _ = _selector(tz_atom, qz_atom, h_qz.geometry.na)
        checks = []
        for ik, k in enumerate(K_POINTS):
            s_qz = np.asarray(h_qz.Sk(k=k, format="array"))
            s_tz = np.asarray(h_tz.Sk(k=k, format="array"))
            parent_error = float(np.linalg.norm(tz_selector.T @ s_qz @ tz_selector - s_tz, ord=2) / np.linalg.norm(s_tz, ord=2))
            condition = float(np.linalg.cond(s_qz))
            h_abs = np.asarray(h_qz.Hk(k=k, format="array")) + fermi * s_qz
            selected = prod_selector.T @ (h_abs - vacuum * s_qz) @ prod_selector
            hermiticity = float(np.linalg.norm(selected - selected.conj().T) / np.linalg.norm(selected))
            if parent_error >= OVERLAP_TOL:
                failures.append(f"step={step} k={ik}: parent overlap={parent_error:.6g}")
            if condition >= COND_MAX:
                failures.append(f"step={step} k={ik}: condition={condition:.6g}")
            if hermiticity >= HERM_TOL:
                failures.append(f"step={step} k={ik}: hermiticity={hermiticity:.6g}")
            matrices[(step, ik)] = selected
            checks.append({"k_reduced": list(k), "parent_overlap_relative": parent_error,
                           "condition_S_QZ": condition, "selected_K_hermiticity_relative": hermiticity})
        points.append({"step": step, "displacement_ang": step * H_ANG, "run_dir": str(runs[step]), "checks": checks,
                       "tshs_sha256": file_sha256(runs[step] / f"{qz_label}.TSHS")})

    parent = np.load(REPO_ROOT / "Comparison/results/epc/reference_basis_convergence_covariant/n_tzp_d12_14_C1_x_covariant.npz")
    central_prod = _production_run("C1_x", 0)
    _, h_a, _, _, _ = _matrix_data(central_prod)
    comparisons, saved = [], {}
    for ik, k in enumerate(K_POINTS):
        derivative = sum(weight * matrices[(step, ik)] for step, weight in weights.items()) / (12 * H_ANG)
        s_a = np.asarray(h_a.Sk(k=k, format="array"))
        k0 = matrices[(0, ik)]
        left, right = connection[f"S_L_C1_x_k{ik}"], connection[f"S_R_C1_x_k{ik}"]
        delta = derivative - left @ np.linalg.solve(s_a, k0) - k0 @ np.linalg.solve(s_a, right)
        difference = delta - parent[f"Delta_covariant_k{ik}"]
        rms, maximum = _rms(difference, s_a)
        signal, _ = _rms(delta, s_a)
        relative = rms / max(signal, np.finfo(float).tiny)
        passed = rms < LIMITS["rms_ev_per_ang"] and relative < LIMITS["relative"] and maximum < LIMITS["max_ev_per_ang"]
        comparisons.append({"k_reduced": list(k), "rms_ev_per_ang": rms, "relative": relative,
                            "max_ev_per_ang": maximum, "verdict": "PASS" if passed else "NO_GO"})
        saved[f"D5_K_prod_subblock_k{ik}"] = derivative
        saved[f"Delta_covariant_k{ik}"] = delta
        saved[f"K0_prod_subblock_k{ik}"] = k0
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "n_qzp_d12_14_C1_x_covariant.npz"
    np.savez_compressed(artifact, **saved)
    passed = not failures and all(row["verdict"] == "PASS" for row in comparisons)
    report = {
        "schema": "qz_cardinality_sentinel_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO", "gate": "QZ_cardinality_sentinel_C1_x",
        "comparison": "N-QZP+D12/14 minus N-TZP+D12/14", "thresholds": LIMITS,
        "failure_reasons": failures, "points": points, "comparisons": comparisons,
        "artifact": str(artifact), "artifact_sha256": file_sha256(artifact),
        "claim_scope": "C1:x cardinality sentinel only; no C1:z, Delta_out, or full-KS promotion",
    }
    path = output / "qz_cardinality_sentinel.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"QZ cardinality sentinel C1:x: {report['verdict']} -> {path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--siesta", default="/home/christian/bin/siesta")
    args = parser.parse_args(argv)
    return 0 if evaluate(args.output, produce(args.output, args.siesta))["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
