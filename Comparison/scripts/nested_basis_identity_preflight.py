#!/usr/bin/env python3
"""N0: prove that enriched SIESTA bases contain the exact PROD-SZ orbitals."""

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
from cross_basis_projection_preflight import K_POINTS, PRODUCTION  # noqa: E402
from run_hamiltonian_derivative_siesta_references import run_siesta  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/nested_basis_identity_preflight"
PROD_CENTRAL = REPO_ROOT / "Comparison/results/epc/basis_projection_sentinel/runs/PROD-SZ/central"
RADIAL_TOL = 1e-10
OVERLAP_TOL = 1e-8
COND_MAX = 1e8

BASIS_BLOCKS = {
    "n_dzp": """C 2
 n=2 0 2
   4.088 0.0
   1.000 1.000
 n=2 1 2 P 1
   4.870 0.0
   1.000 1.000""",
    "n_tzp": """C 2
 n=2 0 3
   4.088 0.0 0.0
   1.000 1.000 1.000
 n=2 1 3 P 1
   4.870 0.0 0.0
   1.000 1.000 1.000""",
    "n_tzp_d10_12": """C 4
 n=2 0 3
   4.088 0.0 0.0
   1.000 1.000 1.000
 n=2 1 3 P 1
   4.870 0.0 0.0
   1.000 1.000 1.000
 n=3 0 1
   10.000
   1.000
 n=3 1 1
   12.000
   1.000""",
    "n_tzp_d12_14": """C 4
 n=2 0 3
   4.088 0.0 0.0
   1.000 1.000 1.000
 n=2 1 3 P 1
   4.870 0.0 0.0
   1.000 1.000 1.000
 n=3 0 1
   12.000
   1.000
 n=3 1 1
   14.000
   1.000""",
}


def render_fdf(source: str, label: str, basis_block: str) -> str:
    lines, kept, inside = source.splitlines(), [], False
    for line in lines:
        stripped = line.strip().lower()
        if stripped == "%block pao.basis":
            inside = True
            continue
        if inside and stripped == "%endblock pao.basis":
            inside = False
            continue
        if inside:
            continue
        if stripped.startswith("systemname"):
            line = f"SystemName {label}"
        elif stripped.startswith("systemlabel"):
            line = f"SystemLabel {label}"
        elif stripped.startswith("dm.tolerance"):
            line = "DM.Tolerance 1.d-6"
        kept.append(line)
    if inside:
        raise ValueError("unterminated PAO.Basis block")
    kept.extend((
        "", "PAO.BasisType split", "PAO.SplitNorm 0.15", "PAO.SplitTailNorm T",
        "PAO.OldStylePolOrbs F", "%block PAO.Polarization.Scheme", " C perturbative",
        "%endblock PAO.Polarization.Scheme", "%block PAO.Basis", basis_block,
        "%endblock PAO.Basis",
    ))
    return "\n".join(kept) + "\n"


def produce(output: Path, siesta: str) -> list[Path]:
    source = (PRODUCTION / "RUN.fdf").read_text(encoding="utf-8")
    runs = []
    for label, block in BASIS_BLOCKS.items():
        run_dir = output / "runs" / label
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "RUN.fdf").write_text(render_fdf(source, label, block), encoding="utf-8")
        for pseudo in ("C.psf", "Ghost-H.psf"):
            shutil.copy2(PRODUCTION / pseudo, run_dir / pseudo)
        if not (run_dir / f"{label}.TSHS").is_file():
            record = run_siesta(run_dir, command=siesta, use_shell=False)
            (run_dir / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            if record["returncode"]:
                raise RuntimeError(f"SIESTA failed for {label}; see {run_dir / 'RUN.out'}")
        runs.append(run_dir)
    return runs


def _signature(orbital) -> tuple[int, int, int, int]:
    return int(orbital.n), int(orbital.l), int(orbital.m), int(orbital.zeta)


def _selector(prod_atom, candidate_atom, atom_count: int) -> tuple[np.ndarray, list[int]]:
    candidate = {_signature(orbital): index for index, orbital in enumerate(candidate_atom.orbitals)}
    local = [candidate[_signature(orbital)] for orbital in prod_atom.orbitals]
    selector = np.zeros((candidate_atom.no * atom_count, prod_atom.no * atom_count))
    for atom in range(atom_count):
        for prod_index, candidate_index in enumerate(local):
            selector[atom * candidate_atom.no + candidate_index, atom * prod_atom.no + prod_index] = 1.0
    return selector, local


def _radial_error(prod_orbital, candidate_orbital) -> float:
    radius = max(float(prod_orbital.R), float(candidate_orbital.R))
    grid = np.linspace(0.0, radius, 8193)
    a, b = prod_orbital.radial(grid), candidate_orbital.radial(grid)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.finfo(float).tiny))


def evaluate(output: Path, runs: list[Path]) -> dict:
    import sisl

    prod_label = "prod_central"
    prod_sile = sisl.get_sile(str(PROD_CENTRAL / f"{prod_label}.TSHS"))
    prod_h = prod_sile.read_hamiltonian()
    prod_atom = sisl.get_sile(str(PROD_CENTRAL / "C.ion.xml")).read_basis()
    rows = []
    for run_dir in runs:
        label = run_dir.name
        sile = sisl.get_sile(str(run_dir / f"{label}.TSHS"))
        hamiltonian = sile.read_hamiltonian()
        atom = sisl.get_sile(str(run_dir / "C.ion.xml")).read_basis()
        selector, local = _selector(prod_atom, atom, prod_h.geometry.na)
        radial = []
        for prod_index, candidate_index in enumerate(local):
            radial.append({
                "prod_index": prod_index, "candidate_index": candidate_index,
                "orbital": prod_atom.orbitals[prod_index].name(),
                "relative_error": _radial_error(prod_atom.orbitals[prod_index], atom.orbitals[candidate_index]),
                "prod_cutoff_ang": float(prod_atom.orbitals[prod_index].R),
                "candidate_cutoff_ang": float(atom.orbitals[candidate_index].R),
            })
        k_checks = []
        for k in K_POINTS:
            s_a = np.asarray(prod_h.Sk(k=k, format="array"))
            s_b = np.asarray(hamiltonian.Sk(k=k, format="array"))
            selected = selector.T @ s_b @ selector
            k_checks.append({
                "k_reduced": list(k), "selected_overlap_relative": float(np.linalg.norm(selected - s_a, ord=2) / np.linalg.norm(s_a, ord=2)),
                "condition_S_B": float(np.linalg.cond(s_b)),
            })
        ordering = [_signature(orbital) for orbital in atom.orbitals]
        failures = [f"{item['orbital']}: radial error {item['relative_error']:.6g}" for item in radial if item["relative_error"] >= RADIAL_TOL]
        failures += [f"k={item['k_reduced']}: selected overlap error {item['selected_overlap_relative']:.6g}" for item in k_checks if item["selected_overlap_relative"] >= OVERLAP_TOL]
        failures += [f"k={item['k_reduced']}: condition {item['condition_S_B']:.6g}" for item in k_checks if item["condition_S_B"] >= COND_MAX]
        rows.append({
            "label": label, "verdict": "PASS" if not failures else "NO_GO",
            "orbitals_per_C": int(atom.no), "prod_selector_local_indices": local,
            "orbital_order": ordering, "radial_identity": radial, "k_checks": k_checks,
            "failure_reasons": failures, "run_dir": str(run_dir),
            "ion_xml_sha256": file_sha256(run_dir / "C.ion.xml"),
            "orb_indx_sha256": file_sha256(run_dir / f"{label}.ORB_INDX"),
            "tshs_sha256": file_sha256(run_dir / f"{label}.TSHS"),
            "run_fdf_sha256": file_sha256(run_dir / "RUN.fdf"),
            "pseudo_sha256": file_sha256(run_dir / "C.psf"),
        })
    passed = all(row["verdict"] == "PASS" for row in rows)
    report = {
        "schema": "nested_basis_identity_preflight_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO", "gate": "nested_basis_identity_preflight",
        "previous_standard_basis_gate": "displaced_projection_sentinel_v1=NO_GO (unchanged)",
        "claim_scope": "identity/nesting preflight only; no displaced stencil, Delta_out, or full-KS claim",
        "selector_policy": "constant zero-one selector matched by (n,l,m,zeta)",
        "thresholds": {"radial_relative": RADIAL_TOL, "selected_overlap_relative": OVERLAP_TOL, "condition_S_B": COND_MAX},
        "production": {"run_dir": str(PROD_CENTRAL), "ion_xml_sha256": file_sha256(PROD_CENTRAL / "C.ion.xml")},
        "bases": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "nested_basis_identity_preflight.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"nested basis identity preflight: {report['verdict']} -> {path}")
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
