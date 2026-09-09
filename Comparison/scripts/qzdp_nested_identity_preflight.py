#!/usr/bin/env python3
"""Central preflight: QZDP adds only a second polarization-d zeta to QZP."""

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
from nested_basis_identity_preflight import (  # noqa: E402
    COND_MAX, OVERLAP_TOL, RADIAL_TOL, _radial_error, _selector, render_fdf,
)
from qz_nested_identity_preflight import DEFAULT_OUTPUT as QZ_ROOT  # noqa: E402
from run_hamiltonian_derivative_siesta_references import run_siesta  # noqa: E402

QZ_RUN = QZ_ROOT / "runs/n_qzp_d12_14"
DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/qzdp_nested_identity_preflight"
SPLIT_NORM_ATTEMPTS = (0.15, 0.30, 0.50, 0.80, 0.99)
QZDP_BLOCK = """C 4
 n=2 0 4
   4.088 2.904883387421 3.283101463157 3.536034066719
   1.000 1.000 1.000 1.000
 n=2 1 4 P 2 S 0.99
   4.870 3.417495483358 3.885530283346 4.193131382225
   1.000 1.000 1.000 1.000
 n=3 0 1
   12.000
   1.000
 n=3 1 1
   14.000
   1.000"""


def produce(output: Path, siesta: str) -> Path:
    label, run = "n_qzdp_d12_14", output / "runs/n_qzdp_d12_14"
    run.mkdir(parents=True, exist_ok=True)
    source = (QZ_RUN / "RUN.fdf").read_text(encoding="utf-8")
    (run / "RUN.fdf").write_text(render_fdf(source, label, QZDP_BLOCK), encoding="utf-8")
    for pseudo in ("C.psf", "Ghost-H.psf"):
        shutil.copy2(QZ_RUN / pseudo, run / pseudo)
    if not (run / f"{label}.TSHS").is_file():
        record = run_siesta(run, command=siesta, use_shell=False)
        (run / "run_record.json").write_text(json.dumps(record, indent=2) + "\n")
        if record["returncode"]:
            raise RuntimeError(f"SIESTA failed; see {run / 'RUN.out'}")
    return run


def evaluate(output: Path, candidate: Path) -> dict:
    import sisl

    qz_label, dp_label = "n_qzp_d12_14", "n_qzdp_d12_14"
    qz_h = sisl.get_sile(str(QZ_RUN / f"{qz_label}.TSHS")).read_hamiltonian()
    dp_h = sisl.get_sile(str(candidate / f"{dp_label}.TSHS")).read_hamiltonian()
    qz_atom = sisl.get_sile(str(QZ_RUN / "C.ion.xml")).read_basis()
    dp_atom = sisl.get_sile(str(candidate / "C.ion.xml")).read_basis()
    selector, local = _selector(qz_atom, dp_atom, qz_h.geometry.na)
    radial = [{"orbital": old.name(), "qz_index": i, "qzdp_index": local[i],
               "relative_error": _radial_error(old, dp_atom.orbitals[local[i]]),
               "qz_cutoff_ang": float(old.R), "qzdp_cutoff_ang": float(dp_atom.orbitals[local[i]].R)}
              for i, old in enumerate(qz_atom.orbitals)]
    checks = []
    for k in K_POINTS:
        s_qz = np.asarray(qz_h.Sk(k=k, format="array"))
        s_dp = np.asarray(dp_h.Sk(k=k, format="array"))
        checks.append({"k_reduced": list(k),
                       "selected_QZP_overlap_relative": float(
                           np.linalg.norm(selector.T @ s_dp @ selector - s_qz, 2) / np.linalg.norm(s_qz, 2)),
                       "condition_S_QZDP": float(np.linalg.cond(s_dp))})
    failures = [f"{row['orbital']}: radial={row['relative_error']:.6g}"
                for row in radial if row["relative_error"] >= RADIAL_TOL]
    failures += [f"k={row['k_reduced']}: overlap={row['selected_QZP_overlap_relative']:.6g}"
                 for row in checks if row["selected_QZP_overlap_relative"] >= OVERLAP_TOL]
    failures += [f"k={row['k_reduced']}: condition={row['condition_S_QZDP']:.6g}"
                 for row in checks if row["condition_S_QZDP"] >= COND_MAX]
    d1 = next(row for row in radial if row["orbital"].startswith("3d") and "Z1" in row["orbital"])
    report = {"schema": "qzdp_nested_identity_preflight_v1",
              "gate": "QZDP_nested_identity_preflight", "generated_at": datetime.now(timezone.utc).isoformat(),
              "verdict": "PASS" if not failures else "NO_GO",
              "basis": "N-QZDP+D12/14", "parent": "N-QZP+D12/14",
              "construction": "n=2 l=1 Nzeta=4 P 2 S 0.99; certified QZ matching radii fixed explicitly",
              "failure_reasons": failures,
              "thresholds": {"radial_relative": RADIAL_TOL,
                             "selected_overlap_relative": OVERLAP_TOL,
                             "condition_S_QZDP": COND_MAX},
              "parent_selector_local_indices": local, "radial_prefix_checks": radial,
              "d_zeta1_check": d1, "k_checks": checks,
              "candidate_run_dir": str(candidate),
              "ion_sha256": file_sha256(candidate / "C.ion.xml"),
              "orb_indx_sha256": file_sha256(candidate / f"{dp_label}.ORB_INDX"),
              "claim_scope": "central QZP-prefix identity only; C1:x stencil forbidden unless PASS"}
    path = output / "qzdp_nested_identity_preflight.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"QZDP nested identity preflight: {report['verdict']} -> {path}")
    return report


def generation_failure_report(output: Path, run: Path, reason: str) -> dict:
    report = {
        "schema": "qzdp_nested_identity_preflight_v1",
        "gate": "QZDP_nested_identity_preflight",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "NO_GO",
        "basis": "N-QZDP+D12/14",
        "parent": "N-QZP+D12/14",
        "stage": "atomic_basis_generation",
        "failure_reasons": [reason],
        "split_norm_attempts": list(SPLIT_NORM_ATTEMPTS),
        "thresholds": {"radial_relative": RADIAL_TOL,
                       "selected_overlap_relative": OVERLAP_TOL,
                       "condition_S_QZDP": COND_MAX},
        "candidate_run_dir": str(run),
        "run_output": str(run / "RUN.out"),
        "claim_scope": "generation NO_GO; no TSHS, SCF result, displaced stencil, or polarization probe",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "qzdp_nested_identity_preflight.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"QZDP nested identity preflight: NO_GO -> {path}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--siesta", default="/home/christian/bin/siesta")
    args = parser.parse_args(argv)
    try:
        candidate = produce(args.output, args.siesta)
    except RuntimeError as error:
        generation_failure_report(args.output, args.output / "runs/n_qzdp_d12_14", str(error))
        return 1
    return 0 if evaluate(args.output, candidate)["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
