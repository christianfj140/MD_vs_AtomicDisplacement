#!/usr/bin/env python3
"""Certify nested TZ/QZ one-sided connections through validated ``.onlyS`` semantics."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from audit_reference_basis_dense_k import _runs  # noqa: E402
from certify_full_basis_connection import CASES, WEIGHTS  # noqa: E402
from certify_production_basis_connection import ADJOINT_REL_TOL, LIMIT_EV_PER_ANG  # noqa: E402
from certify_support_exact_connection import SOLVE_RESIDUAL_TOL, _propagate  # noqa: E402
from fdf_materialization import _set_fdf_directive  # noqa: E402
from onlys_semantic_preflight import (  # noqa: E402
    P0_S11_REL_TOL, _cross_derivative, _dense_k_points, _duplicated_fdf,
)
from cross_basis_projection_preflight import K_POINTS  # noqa: E402
from run_displaced_projection_sentinel import H_ANG  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/full_basis_connection_onlys_v2"
SIESTA = "/home/christian/bin/siesta"


def produce(basis: str, direction: str, central_dir: Path, geometry,
            steps=(-1, 1), output=OUTPUT) -> dict:
    source = (central_dir / "RUN.fdf").read_text(encoding="utf-8")
    result = {}
    axis = {"C1_x": 0, "C1_z": 2}[direction]
    for step in steps:
        tag = f"{'m' if step < 0 else 'p'}{abs(step)}"
        label = f"onlys_{basis}_{direction.lower()}_{tag}"
        run = output / "runs" / basis / direction / tag
        run.mkdir(parents=True, exist_ok=True)
        displaced = geometry.xyz.copy()
        displaced[0, axis] += step * H_ANG
        text = _duplicated_fdf(source, label, np.vstack([geometry.xyz, displaced]))
        text = _set_fdf_directive(text, "MaxSCFIterations", "0")
        (run / "RUN.fdf").write_text(text, encoding="utf-8")
        for pseudo in ("C.psf", "Ghost-H.psf"):
            shutil.copy2(central_dir / pseudo, run / pseudo)
        target = run / f"{label}.onlyS"
        if not target.is_file():
            started = time.time()
            with (run / "RUN.fdf").open() as stdin, (run / "RUN.out").open("w") as stdout:
                completed = subprocess.run([SIESTA], cwd=run, stdin=stdin, stdout=stdout,
                                           stderr=subprocess.STDOUT, check=False)
            (run / "run_record.json").write_text(json.dumps({
                "command": [SIESTA], "returncode": completed.returncode,
                "elapsed_seconds": time.time() - started, "overlap_only": True,
            }, indent=2) + "\n")
            if completed.returncode or not target.is_file():
                raise RuntimeError(f"overlap-only run failed: {run}")
        result[(direction, step)] = run
    return result


def main() -> int:
    import sisl

    semantic = json.loads((REPO_ROOT / "Comparison/results/epc/b_i_fourier_bessel_semantic_v5/b_i_fourier_bessel_semantic_v5.json").read_text())
    if semantic["verdict"] != "PASS":
        raise RuntimeError("PROD-SZ B_I semantic closure v5 is not PASS")
    dense = _dense_k_points()
    cases = []
    for basis, direction in CASES:
        runs = _runs(basis, direction)
        label, central, fermi, vacuum, _ = _matrix_data(runs[0])
        overlap_runs = produce(basis, direction, runs[0], central.geometry)
        overlaps = {
            key: sisl.get_sile(str(next(path.glob("*.onlyS")))).read_overlap()
            for key, path in overlap_runs.items()
        }
        no = central.no
        failures = []
        for overlap in overlaps.values():
            if overlap.no != 2 * no:
                failures.append(f"onlyS dimension {overlap.no} != {2 * no}")
                continue
            for k in K_POINTS:
                s11 = np.asarray(overlap.Sk(k=k, format="array"))[:no, :no]
                reference = np.asarray(central.Sk(k=k, format="array"))
                relative = float(np.linalg.norm(s11 - reference, 2) / np.linalg.norm(reference, 2))
                if relative >= P0_S11_REL_TOL:
                    failures.append(f"S11 vs TSHS relative={relative:.6g}")
        hamiltonians = {step: _matrix_data(path)[1] for step, path in runs.items()}
        rows, saved, worst, worst_k, over = [], {}, 0.0, None, 0
        for ik, k in enumerate(dense):
            right, left = _cross_derivative(overlaps, direction, no, k)
            s0 = np.asarray(central.Sk(k=k, format="array"))
            k0 = np.asarray(central.Hk(k=k, format="array")) + (fermi - vacuum) * s0
            d_s = sum(WEIGHTS[step] * np.asarray(hamiltonians[step].Sk(k=k, format="array"))
                      for step in WEIGHTS) / (12 * H_ANG)
            closure = d_s - left - right
            propagated, residual = _propagate(closure / 2, closure / 2, s0, k0)
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
        if over:
            failures.append(f"{over}/400 k fail; worst propagated={worst:.6g} eV/Ang")
        artifact = OUTPUT / f"{basis}_{direction}_connections.npz"
        np.savez_compressed(artifact, **saved)
        cases.append({"basis": basis, "direction": direction,
                      "verdict": "PASS" if not failures else "NO_GO",
                      "failure_reasons": failures, "diagnostic_k": rows,
                      "dense_audit": {"k_count": len(dense), "k_over_limit": over,
                                      "worst_propagated_ev_per_ang": worst,
                                      "worst_k_reduced": worst_k},
                      "artifact": str(artifact), "artifact_sha256": file_sha256(artifact),
                      "runs": [str(path) for path in overlap_runs.values()]})
        print(basis, direction, cases[-1]["verdict"], flush=True)

    passed = all(case["verdict"] == "PASS" for case in cases)
    report = {"schema": "full_basis_connection_onlys_v2", "gate": "full_basis_connection_v2",
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "verdict": "PASS" if passed else "NO_GO", "k_mesh": [20, 20, 1],
              "thresholds": {"propagated_ev_per_ang": LIMIT_EV_PER_ANG,
                             "adjoint_relative": ADJOINT_REL_TOL,
                             "solve_residual_relative": SOLVE_RESIDUAL_TOL},
              "authorization": "PROD-SZ onlyS semantics independently closed by Fourier-Bessel v5",
              "historical_full_basis_connection_v1": "NO_GO unchanged",
              "cases": cases, "claim_scope": "nested-basis one-sided connections only; not Delta_out/full-KS"}
    path = OUTPUT / "full_basis_connection_onlys_v2.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"full basis connection v2: {report['verdict']} -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
