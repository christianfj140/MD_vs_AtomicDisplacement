#!/usr/bin/env python3
"""Fail-closed round trip of the certified QZP .ion package via User.Basis."""

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
from diagnose_nested_basis_central import _density, _last_total_energy  # noqa: E402
from nested_basis_identity_preflight import RADIAL_TOL, _radial_error  # noqa: E402
from onlys_semantic_preflight import _dense_k_points  # noqa: E402
from qz_nested_identity_preflight import DEFAULT_OUTPUT as QZ_ROOT  # noqa: E402
from run_hamiltonian_derivative_siesta_references import run_siesta  # noqa: E402
from run_nested_basis_ladder import _rms  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

PARENT = QZ_ROOT / "runs/n_qzp_d12_14"
DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/user_basis_roundtrip_preflight"
LABEL = "n_qzp_d12_14_user_basis"
LIMITS = {
    "radial_relative": RADIAL_TOL,
    "overlap_relative_spectral": 1e-12,
    "hamiltonian_rms_ev": 1e-5,
    "total_energy_abs_ev": 1e-6,
    "fermi_abs_ev": 1e-6,
    "density_relative_l2": 1e-6,
}


def render_fdf(source: str) -> str:
    rendered, in_basis = [], False
    for line in source.splitlines():
        stripped = line.strip().lower()
        if stripped == "%block pao.basis":
            in_basis = True
            continue
        if in_basis:
            if stripped == "%endblock pao.basis":
                in_basis = False
            continue
        if stripped.startswith("systemname"):
            line = f"SystemName {LABEL}"
        elif stripped.startswith("systemlabel"):
            line = f"SystemLabel {LABEL}"
        rendered.append(line)
    if in_basis:
        raise ValueError("unterminated PAO.Basis block")
    rendered.extend(("", "User.Basis T"))
    return "\n".join(rendered) + "\n"


def produce(output: Path, siesta: str) -> Path:
    run = output / "runs" / LABEL
    run.mkdir(parents=True, exist_ok=True)
    (run / "RUN.fdf").write_text(render_fdf((PARENT / "RUN.fdf").read_text()), encoding="utf-8")
    for name in ("C.ion", "Ghost-H.ion", "C.ion.xml", "Ghost-H.ion.xml"):
        shutil.copy2(PARENT / name, run / name)
    if not (run / f"{LABEL}.TSHS").is_file():
        record = run_siesta(run, command=siesta, use_shell=False)
        (run / "run_record.json").write_text(json.dumps(record, indent=2) + "\n")
        if record["returncode"]:
            raise RuntimeError(f"SIESTA failed; see {run / 'RUN.out'}")
    return run


def _orbital_signature(orbital) -> tuple[int, int, int, int]:
    return int(orbital.n), int(orbital.l), int(orbital.m), int(orbital.zeta)


def evaluate(output: Path, run: Path) -> dict:
    import sisl

    parent_label, parent_h, parent_fermi, _, parent_atom = _matrix_data(PARENT)
    label, current_h, current_fermi, _, current_atom = _matrix_data(run)
    radial = [_radial_error(a, b) for a, b in zip(parent_atom.orbitals, current_atom.orbitals)]
    ordering_equal = (parent_atom.no == current_atom.no and
                      [_orbital_signature(o) for o in parent_atom.orbitals] ==
                      [_orbital_signature(o) for o in current_atom.orbitals])
    k_rows, failures = [], []
    for k in _dense_k_points():
        s0 = np.asarray(parent_h.Sk(k=k, format="array"))
        s1 = np.asarray(current_h.Sk(k=k, format="array"))
        h0 = np.asarray(parent_h.Hk(k=k, format="array")) + parent_fermi * s0
        h1 = np.asarray(current_h.Hk(k=k, format="array")) + current_fermi * s1
        s_error = float(np.linalg.norm(s1 - s0, 2) / np.linalg.norm(s0, 2))
        h_rms, h_max = _rms(h1 - h0, (s0 + s0.conj().T) / 2)
        k_rows.append({"k_reduced": list(k), "overlap_relative_spectral": s_error,
                       "hamiltonian_rms_ev": h_rms, "hamiltonian_max_ev": h_max})
    worst_s = max(row["overlap_relative_spectral"] for row in k_rows)
    worst_h = max(row["hamiltonian_rms_ev"] for row in k_rows)
    shape = tuple(int(x) for x in sisl.get_sile(str(PARENT / f"{parent_label}.VT")).read_grid().shape)
    rho0 = _density(PARENT, parent_label, parent_h, parent_atom, shape)
    rho1 = _density(run, label, current_h, current_atom, shape)
    density_error = float(np.linalg.norm(rho1 - rho0) / np.linalg.norm(rho0))
    energy0 = _last_total_energy((PARENT / "RUN.out").read_text(errors="replace"))
    energy1 = _last_total_energy((run / "RUN.out").read_text(errors="replace"))
    energy_error, fermi_error = abs(energy1 - energy0), abs(current_fermi - parent_fermi)
    output_text = (run / "RUN.out").read_text(errors="replace")
    user_basis_read = "Reading PAOs and KBs from ascii files" in output_text
    ion_hashes = {name: {"parent": file_sha256(PARENT / name), "roundtrip_input": file_sha256(run / name)}
                  for name in ("C.ion", "Ghost-H.ion")}
    if not ordering_equal:
        failures.append("orbital ordering/count differs")
    if max(radial, default=0.0) >= LIMITS["radial_relative"]:
        failures.append(f"radial error={max(radial):.6g}")
    if any(row["parent"] != row["roundtrip_input"] for row in ion_hashes.values()):
        failures.append("copied .ion package hash differs from parent")
    if not user_basis_read:
        failures.append("RUN.out does not prove both declared species .ion files were read")
    for key, value in (("overlap_relative_spectral", worst_s), ("hamiltonian_rms_ev", worst_h),
                       ("total_energy_abs_ev", energy_error), ("fermi_abs_ev", fermi_error),
                       ("density_relative_l2", density_error)):
        if value >= LIMITS[key]:
            failures.append(f"{key}={value:.6g}")
    report = {
        "schema": "user_basis_roundtrip_preflight_v1", "gate": "user_basis_roundtrip_preflight",
        "generated_at": datetime.now(timezone.utc).isoformat(), "verdict": "PASS" if not failures else "NO_GO",
        "parent": str(PARENT), "roundtrip": str(run), "failure_reasons": failures, "thresholds": LIMITS,
        "user_basis_species_files_read": user_basis_read, "ion_package_hashes": ion_hashes,
        "orbital_ordering_equal": ordering_equal, "radial_max_relative": max(radial, default=0.0),
        "dense_k_count": len(k_rows), "dense_k_worst_overlap_relative": worst_s,
        "dense_k_worst_hamiltonian_rms_ev": worst_h, "total_energy_abs_difference_ev": energy_error,
        "fermi_abs_difference_ev": fermi_error, "density_relative_l2": density_error,
        "k_results": k_rows, "tshs_sha256": file_sha256(run / f"{LABEL}.TSHS"),
        "claim_scope": "byte-identical QZP .ion User.Basis round trip only; custom d2 forbidden unless PASS",
    }
    path = output / "user_basis_roundtrip_preflight.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"User.Basis round-trip preflight: {report['verdict']} -> {path}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--siesta", default="/home/christian/bin/siesta")
    args = parser.parse_args(argv)
    return 0 if evaluate(args.output, produce(args.output, args.siesta))["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
