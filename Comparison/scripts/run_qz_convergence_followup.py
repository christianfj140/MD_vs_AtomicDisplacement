#!/usr/bin/env python3
"""Close QZ range at C1:x, then QZ cardinality at C1:z, fail-closed."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from cross_basis_projection_preflight import K_POINTS  # noqa: E402
from nested_basis_identity_preflight import COND_MAX, OVERLAP_TOL, PROD_CENTRAL, _radial_error, _selector  # noqa: E402
from qz_nested_identity_preflight import DEFAULT_OUTPUT as QZ_PREFLIGHT, QZ_BLOCK  # noqa: E402
from run_displaced_projection_sentinel import DIRECTIONS, H_ANG, STEPS  # noqa: E402
from run_nested_basis_ladder import DEFAULT_OUTPUT as LADDER, IDENTITY_ROOT, LIMITS, _rms, _run_one  # noqa: E402
from run_nested_basis_sentinel import DEFAULT_OUTPUT as NESTED_SENTINEL, HERM_TOL, _matrix_data, _production_run  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/qz_convergence_followup"
CONNECTION = REPO_ROOT / "Comparison/results/epc/production_basis_connection/production_basis_connection_matrices.npz"
COVARIANT = REPO_ROOT / "Comparison/results/epc/reference_basis_convergence_covariant"
QZ_D12_X = REPO_ROOT / "Comparison/results/epc/qz_cardinality_sentinel/n_qzp_d12_14_C1_x_covariant.npz"
QZ_D10_BLOCK = QZ_BLOCK.replace("12.000", "10.000").replace("14.000", "12.000")


def _parent_run(parent: str, direction: str, step: int) -> Path:
    if step == 0:
        return IDENTITY_ROOT / parent
    tag = f"{'m' if step < 0 else 'p'}{abs(step)}"
    if parent == "n_tzp_d10_12":
        return NESTED_SENTINEL / "runs" / direction / tag
    return LADDER / "runs" / parent / direction / tag


def _run_case(name: str, block: str, parent: str, direction: str, reference: Path, siesta: str) -> dict:
    import sisl

    axis = DIRECTIONS[direction]
    xyz0 = sisl.get_sile(str(PROD_CENTRAL / "prod_central.TSHS")).read_geometry().xyz
    runs = {}
    if name == "n_qzp_d12_14" and direction == "C1_z":
        runs[0] = QZ_PREFLIGHT / "runs" / name
    else:
        label = name
        runs[0] = _run_one(OUTPUT / "runs" / name / direction / "central", label, block, xyz0, siesta)
    for step in (-2, -1, 1, 2):
        tag = f"{'m' if step < 0 else 'p'}{abs(step)}"
        xyz = xyz0.copy()
        xyz[0, axis] += step * H_ANG
        label = f"{name}_{direction.lower()}_{tag}"
        runs[step] = _run_one(OUTPUT / "runs" / name / direction / tag, label, block, xyz, siesta)

    prod_atom = sisl.get_sile(str(PROD_CENTRAL / "C.ion.xml")).read_basis()
    parent_atom = sisl.get_sile(str(_parent_run(parent, direction, 0) / "C.ion.xml")).read_basis()
    connection, target = np.load(CONNECTION), np.load(reference)
    matrices, points, failures = {}, [], []
    weights = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
    for step in STEPS:
        label, h_qz, fermi, vacuum, qz_atom = _matrix_data(runs[step])
        _, h_parent, _, _, _ = _matrix_data(_parent_run(parent, direction, step))
        prod_selector, _ = _selector(prod_atom, qz_atom, h_qz.geometry.na)
        parent_selector, local = _selector(parent_atom, qz_atom, h_qz.geometry.na)
        checks = []
        for ik, k in enumerate(K_POINTS):
            s_qz = np.asarray(h_qz.Sk(k=k, format="array"))
            s_parent = np.asarray(h_parent.Sk(k=k, format="array"))
            overlap = float(np.linalg.norm(parent_selector.T @ s_qz @ parent_selector - s_parent, 2) / np.linalg.norm(s_parent, 2))
            condition = float(np.linalg.cond(s_qz))
            h_abs = np.asarray(h_qz.Hk(k=k, format="array")) + fermi * s_qz
            selected = prod_selector.T @ (h_abs - vacuum * s_qz) @ prod_selector
            hermiticity = float(np.linalg.norm(selected - selected.conj().T) / np.linalg.norm(selected))
            if overlap >= OVERLAP_TOL or condition >= COND_MAX or hermiticity >= HERM_TOL:
                failures.append(f"step={step} k={ik}: overlap={overlap:.6g}, condition={condition:.6g}, hermiticity={hermiticity:.6g}")
            matrices[(step, ik)] = selected
            checks.append({"k_reduced": list(k), "parent_overlap_relative": overlap, "condition_S_QZ": condition,
                           "selected_K_hermiticity_relative": hermiticity})
        points.append({"step": step, "run_dir": str(runs[step]), "checks": checks,
                       "tshs_sha256": file_sha256(runs[step] / f"{label}.TSHS")})

    qz_atom = sisl.get_sile(str(runs[0] / "C.ion.xml")).read_basis()
    _, local = _selector(parent_atom, qz_atom, 1)
    radial_max = max(_radial_error(orbital, qz_atom.orbitals[index]) for orbital, index in zip(parent_atom.orbitals, local))
    comparisons, saved = [], {}
    _, h_a, _, _, _ = _matrix_data(_production_run(direction, 0))
    for ik, k in enumerate(K_POINTS):
        derivative = sum(weight * matrices[(step, ik)] for step, weight in weights.items()) / (12 * H_ANG)
        s_a, k0 = np.asarray(h_a.Sk(k=k, format="array")), matrices[(0, ik)]
        left, right = connection[f"S_L_{direction}_k{ik}"], connection[f"S_R_{direction}_k{ik}"]
        delta = derivative - left @ np.linalg.solve(s_a, k0) - k0 @ np.linalg.solve(s_a, right)
        rms, maximum = _rms(delta - target[f"Delta_covariant_k{ik}"], s_a)
        signal, _ = _rms(delta, s_a)
        relative = rms / max(signal, np.finfo(float).tiny)
        passed = rms < LIMITS["rms_ev_per_ang"] and relative < LIMITS["relative"] and maximum < LIMITS["max_ev_per_ang"]
        comparisons.append({"k_reduced": list(k), "rms_ev_per_ang": rms, "relative": relative,
                            "max_ev_per_ang": maximum, "verdict": "PASS" if passed else "NO_GO"})
        saved[f"Delta_covariant_k{ik}"] = delta
    artifact = OUTPUT / f"{name}_{direction}_covariant.npz"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact, **saved)
    passed = not failures and radial_max < 1e-10 and all(row["verdict"] == "PASS" for row in comparisons)
    return {"basis": name, "parent": parent, "direction": direction, "verdict": "PASS" if passed else "NO_GO",
            "parent_radial_max_relative": radial_max, "failure_reasons": failures, "points": points,
            "comparisons": comparisons, "artifact": str(artifact), "artifact_sha256": file_sha256(artifact)}


def main() -> int:
    first = _run_case("n_qzp_d10_12", QZ_D10_BLOCK, "n_tzp_d10_12", "C1_x", QZ_D12_X, "/home/christian/bin/siesta")
    cases = [first]
    if first["verdict"] == "PASS":
        cases.append(_run_case("n_qzp_d12_14", QZ_BLOCK, "n_tzp_d12_14", "C1_z",
                               COVARIANT / "n_tzp_d12_14_C1_z_covariant.npz", "/home/christian/bin/siesta"))
    passed = len(cases) == 2 and all(case["verdict"] == "PASS" for case in cases)
    report = {"schema": "qz_convergence_followup_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
              "verdict": "PASS" if passed else "NO_GO", "gate": "QZ_final_cardinality_and_range",
              "thresholds": LIMITS, "cases": cases,
              "claim_scope": "QZ cardinality/range closure in PROD-SZ only; not Delta_out or full-KS"}
    path = OUTPUT / "qz_convergence_followup.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"QZ convergence follow-up: {report['verdict']} -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
