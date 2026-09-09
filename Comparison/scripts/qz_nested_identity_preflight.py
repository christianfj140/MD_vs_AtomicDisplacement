#!/usr/bin/env python3
"""Central preflight: does QZ add only zeta4 to N-TZP+D12/14?"""

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
from nested_basis_identity_preflight import COND_MAX, OVERLAP_TOL, RADIAL_TOL, _radial_error, _selector, render_fdf  # noqa: E402
from run_hamiltonian_derivative_siesta_references import run_siesta  # noqa: E402

TZ_ROOT = REPO_ROOT / "Comparison/results/epc/nested_basis_identity_preflight/runs/n_tzp_d12_14"
DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/qz_nested_identity_preflight"
QZ_BLOCK = """C 4
 n=2 0 4
   4.088 0.0 0.0 0.0
   1.000 1.000 1.000 1.000
 n=2 1 4 P 1
   4.870 0.0 0.0 0.0
   1.000 1.000 1.000 1.000
 n=3 0 1
   12.000
   1.000
 n=3 1 1
   14.000
   1.000"""


def produce(output: Path, siesta: str) -> Path:
    label = "n_qzp_d12_14"
    run = output / "runs" / label
    run.mkdir(parents=True, exist_ok=True)
    source = (TZ_ROOT / "RUN.fdf").read_text(encoding="utf-8")
    (run / "RUN.fdf").write_text(render_fdf(source, label, QZ_BLOCK), encoding="utf-8")
    for pseudo in ("C.psf", "Ghost-H.psf"):
        shutil.copy2(TZ_ROOT / pseudo, run / pseudo)
    if not (run / f"{label}.TSHS").is_file():
        record = run_siesta(run, command=siesta, use_shell=False)
        (run / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if record["returncode"]:
            raise RuntimeError(f"SIESTA failed; see {run / 'RUN.out'}")
    return run


def evaluate(output: Path, qz_root: Path) -> dict:
    import sisl

    tz_label, qz_label = "n_tzp_d12_14", "n_qzp_d12_14"
    tz_sile = sisl.get_sile(str(TZ_ROOT / f"{tz_label}.TSHS"))
    qz_sile = sisl.get_sile(str(qz_root / f"{qz_label}.TSHS"))
    tz_h, qz_h = tz_sile.read_hamiltonian(), qz_sile.read_hamiltonian()
    tz_atom = sisl.get_sile(str(TZ_ROOT / "C.ion.xml")).read_basis()
    qz_atom = sisl.get_sile(str(qz_root / "C.ion.xml")).read_basis()
    selector, local = _selector(tz_atom, qz_atom, tz_h.geometry.na)
    radial = []
    for tz_index, qz_index in enumerate(local):
        radial.append({
            "orbital": tz_atom.orbitals[tz_index].name(), "tz_index": tz_index, "qz_index": qz_index,
            "relative_error": _radial_error(tz_atom.orbitals[tz_index], qz_atom.orbitals[qz_index]),
            "tz_cutoff_ang": float(tz_atom.orbitals[tz_index].R), "qz_cutoff_ang": float(qz_atom.orbitals[qz_index].R),
        })
    k_checks = []
    for k in K_POINTS:
        s_tz = np.asarray(tz_h.Sk(k=k, format="array"))
        s_qz = np.asarray(qz_h.Sk(k=k, format="array"))
        selected = selector.T @ s_qz @ selector
        k_checks.append({
            "k_reduced": list(k),
            "selected_TZ_overlap_relative": float(np.linalg.norm(selected - s_tz, ord=2) / np.linalg.norm(s_tz, ord=2)),
            "condition_S_QZ": float(np.linalg.cond(s_qz)),
        })
    failures = [f"{row['orbital']}: radial error={row['relative_error']:.6g}" for row in radial if row["relative_error"] >= RADIAL_TOL]
    failures += [f"k={row['k_reduced']}: overlap error={row['selected_TZ_overlap_relative']:.6g}" for row in k_checks if row["selected_TZ_overlap_relative"] >= OVERLAP_TOL]
    failures += [f"k={row['k_reduced']}: condition={row['condition_S_QZ']:.6g}" for row in k_checks if row["condition_S_QZ"] >= COND_MAX]
    report = {
        "schema": "qz_nested_identity_preflight_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if not failures else "NO_GO", "gate": "QZ_nested_identity_preflight",
        "basis": "N-QZP+D12/14", "parent": "N-TZP+D12/14", "failure_reasons": failures,
        "thresholds": {"radial_relative": RADIAL_TOL, "selected_overlap_relative": OVERLAP_TOL, "condition_S_QZ": COND_MAX},
        "parent_selector_local_indices": local, "radial_prefix_checks": radial, "k_checks": k_checks,
        "qz_run_dir": str(qz_root), "qz_ion_sha256": file_sha256(qz_root / "C.ion.xml"),
        "qz_orb_indx_sha256": file_sha256(qz_root / f"{qz_label}.ORB_INDX"),
        "claim_scope": "central nesting check only; C1:x stencil forbidden unless PASS",
    }
    path = output / "qz_nested_identity_preflight.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"QZ nested identity preflight: {report['verdict']} -> {path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--siesta", default="/home/christian/bin/siesta")
    args = parser.parse_args(argv)
    return 0 if evaluate(args.output, produce(args.output, args.siesta))["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
