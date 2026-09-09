#!/usr/bin/env python3
"""Run and adjudicate the raw contract-preserving PAO basis ladder."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from cross_basis_projection_preflight import K_POINTS  # noqa: E402
from nested_basis_identity_preflight import BASIS_BLOCKS, COND_MAX, OVERLAP_TOL, PROD_CENTRAL, _selector, render_fdf  # noqa: E402
from run_displaced_projection_sentinel import DIRECTIONS, H_ANG, STEPS  # noqa: E402
from run_hamiltonian_derivative_siesta_references import run_siesta  # noqa: E402
from run_nested_basis_sentinel import DEFAULT_OUTPUT as NESTED_SENTINEL, HERM_TOL, _matrix_data, _production_run  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/nested_basis_ladder"
IDENTITY_ROOT = REPO_ROOT / "Comparison/results/epc/nested_basis_identity_preflight/runs"
BASIS_ORDER = ("n_dzp", "n_tzp", "n_tzp_d10_12", "n_tzp_d12_14")
COMPARE = {
    "cardinality": ("n_tzp", "n_dzp", True),
    "diffuse_importance": ("n_tzp_d10_12", "n_tzp", False),
    "diffuse_range": ("n_tzp_d12_14", "n_tzp_d10_12", True),
}
LIMITS = {"rms_ev_per_ang": 0.005, "relative": 0.01, "max_ev_per_ang": 0.015}


def _run_one(run_dir: Path, label: str, block: str, xyz: np.ndarray, siesta: str) -> Path:
    source = (PROD_CENTRAL / "RUN.fdf").read_text(encoding="utf-8")
    text = render_fdf(source, label, block)
    lines, rendered, inside = text.splitlines(), [], False
    for line in lines:
        stripped = line.strip().lower()
        if stripped == "%block atomiccoordinatesandatomicspecies":
            inside = True
            rendered.append(line)
            rendered.extend(" %.12f  %.12f  %.12f  1  # C" % tuple(position) for position in xyz)
            continue
        if inside:
            if stripped == "%endblock atomiccoordinatesandatomicspecies":
                inside = False
                rendered.append(line)
            continue
        rendered.append(line)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "RUN.fdf").write_text("\n".join(rendered) + "\n", encoding="utf-8")
    for pseudo in ("C.psf", "Ghost-H.psf"):
        shutil.copy2(PROD_CENTRAL / pseudo, run_dir / pseudo)
    if not (run_dir / f"{label}.TSHS").is_file():
        record = run_siesta(run_dir, command=siesta, use_shell=False)
        (run_dir / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if record["returncode"]:
            raise RuntimeError(f"SIESTA failed for {label}; see {run_dir / 'RUN.out'}")
    return run_dir


def produce(output: Path, siesta: str) -> dict[str, dict[str, dict[int, Path]]]:
    import sisl

    xyz0 = sisl.get_sile(str(PROD_CENTRAL / "prod_central.TSHS")).read_geometry().xyz
    runs = {}
    for basis in BASIS_ORDER:
        runs[basis] = {}
        for direction, axis in DIRECTIONS.items():
            runs[basis][direction] = {0: IDENTITY_ROOT / basis}
            for step in (-2, -1, 1, 2):
                tag = f"{'m' if step < 0 else 'p'}{abs(step)}"
                if basis == "n_tzp_d10_12":
                    runs[basis][direction][step] = NESTED_SENTINEL / "runs" / direction / tag
                    continue
                xyz = xyz0.copy()
                xyz[0, axis] += step * H_ANG
                label = f"{basis}_{direction.lower()}_{tag}"
                runs[basis][direction][step] = _run_one(
                    output / "runs" / basis / direction / tag, label, BASIS_BLOCKS[basis], xyz, siesta
                )
    return runs


def _rms(matrix: np.ndarray, overlap: np.ndarray) -> tuple[float, float]:
    values, vectors = np.linalg.eigh((overlap + overlap.conj().T) / 2)
    inv = (vectors / np.sqrt(values)) @ vectors.conj().T
    whitened = inv @ matrix @ inv
    return float(np.linalg.norm(whitened) / np.sqrt(len(overlap))), float(np.abs(whitened).max())


def evaluate(output: Path, runs: dict[str, dict[str, dict[int, Path]]]) -> dict:
    import sisl

    prod_atom = sisl.get_sile(str(PROD_CENTRAL / "C.ion.xml")).read_basis()
    derivatives, basis_reports = {}, []
    weights = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
    for basis in BASIS_ORDER:
        basis_result = {"basis": basis, "directions": []}
        derivatives[basis] = {}
        for direction in DIRECTIONS:
            points, matrices, failures = [], {}, []
            for step in STEPS:
                a_dir, b_dir = _production_run(direction, step), runs[basis][direction][step]
                a_label, h_a, _, _, _ = _matrix_data(a_dir)
                b_label, h_b, fermi_b, vacuum_b, atom_b = _matrix_data(b_dir)
                selector, local = _selector(prod_atom, atom_b, h_a.geometry.na)
                checks = []
                for ik, k in enumerate(K_POINTS):
                    s_a = np.asarray(h_a.Sk(k=k, format="array"))
                    s_b = np.asarray(h_b.Sk(k=k, format="array"))
                    selected_s = selector.T @ s_b @ selector
                    h_abs = np.asarray(h_b.Hk(k=k, format="array")) + fermi_b * s_b
                    selected_k = selector.T @ (h_abs - vacuum_b * s_b) @ selector
                    overlap_error = float(np.linalg.norm(selected_s - s_a, ord=2) / np.linalg.norm(s_a, ord=2))
                    condition = float(np.linalg.cond(s_b))
                    hermiticity = float(np.linalg.norm(selected_k - selected_k.conj().T) / np.linalg.norm(selected_k))
                    if overlap_error >= OVERLAP_TOL:
                        failures.append(f"step={step} k={ik}: overlap={overlap_error:.6g}")
                    if condition >= COND_MAX:
                        failures.append(f"step={step} k={ik}: condition={condition:.6g}")
                    if hermiticity >= HERM_TOL:
                        failures.append(f"step={step} k={ik}: hermiticity={hermiticity:.6g}")
                    matrices[(step, ik)] = selected_k
                    checks.append({"k_reduced": list(k), "selected_overlap_relative": overlap_error, "condition_S_B": condition, "selected_K_hermiticity_relative": hermiticity})
                points.append({"step": step, "displacement_ang": step * H_ANG, "checks": checks, "run_dir": str(b_dir), "tshs_sha256": file_sha256(b_dir / f"{b_label}.TSHS"), "prod_tshs_sha256": file_sha256(a_dir / f"{a_label}.TSHS")})
            derivatives[basis][direction] = {}
            saved = {"selector_E": selector}
            for ik in range(len(K_POINTS)):
                derivative = sum(weights[step] * matrices[(step, ik)] for step in weights) / (12 * H_ANG)
                derivatives[basis][direction][ik] = (derivative, np.asarray(_matrix_data(_production_run(direction, 0))[1].Sk(k=K_POINTS[ik], format="array")))
                saved[f"D5_K_prod_subblock_k{ik}"] = derivative
            artifact = output / f"{basis}_{direction}_derivative.npz"
            np.savez_compressed(artifact, **saved)
            basis_result["directions"].append({"direction": direction, "verdict": "PASS" if not failures else "NO_GO", "failure_reasons": failures, "points": points, "derivative_artifact": str(artifact), "derivative_artifact_sha256": file_sha256(artifact)})
        basis_result["verdict"] = "PASS" if all(x["verdict"] == "PASS" for x in basis_result["directions"]) else "NO_GO"
        basis_reports.append(basis_result)

    comparisons, worst = [], None
    for name, (upper, lower, gated) in COMPARE.items():
        for direction in DIRECTIONS:
            for ik, k in enumerate(K_POINTS):
                high, overlap = derivatives[upper][direction][ik]
                low, _ = derivatives[lower][direction][ik]
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
    passed = all(row["verdict"] == "PASS" for row in basis_reports) and all(row["verdict"] == "PASS" for row in comparisons if row["gated"])
    report = {
        "schema": "reference_basis_convergence_raw_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO", "gate": "reference_basis_convergence_raw",
        "observable": "D5 of E_A^T [H_B-c_vacuum*S_B] E_A", "h_ang": H_ANG,
        "directions": list(DIRECTIONS), "k_points_reduced": [list(k) for k in K_POINTS],
        "thresholds": LIMITS, "basis_results": basis_reports, "comparisons": comparisons,
        "worst_case": worst[1] if worst else None,
        "claim_scope": "fixed PROD-SZ reference convergence only; covariant PAO claim awaits intra-atomic connection; not Delta_out/full-KS",
    }
    path = output / "reference_basis_convergence_raw.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"reference basis convergence raw: {report['verdict']} -> {path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--siesta", default="/home/christian/bin/siesta")
    args = parser.parse_args(argv)
    report = evaluate(args.output, produce(args.output, args.siesta))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
