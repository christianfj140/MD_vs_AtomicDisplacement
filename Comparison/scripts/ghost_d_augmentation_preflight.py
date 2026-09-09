#!/usr/bin/env python3
"""Central preflight for an exactly nested, co-located synthetic ghost-C 3d probe."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from cross_basis_projection_preflight import K_POINTS  # noqa: E402
from nested_basis_identity_preflight import COND_MAX, OVERLAP_TOL, RADIAL_TOL, _radial_error  # noqa: E402
from qz_nested_identity_preflight import DEFAULT_OUTPUT as QZ_ROOT, QZ_BLOCK  # noqa: E402
from run_hamiltonian_derivative_siesta_references import run_siesta  # noqa: E402

QZ_RUN = QZ_ROOT / "runs/n_qzp_d12_14"
DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/ghost_d_augmentation_preflight"
LABEL = "n_qzp_d12_14_gd"
GHOST_LABEL = "C-dghost"
GHOST_CUTOFF_BOHR = 4.870
SCHUR_REL_MIN = 1e-6
GHOST_BLOCK = QZ_BLOCK + f"""
{GHOST_LABEL} 1
 n=3 2 1
   {GHOST_CUTOFF_BOHR:.3f}
   1.000"""


def render_ghost_fdf(source: str, xyz: np.ndarray) -> str:
    lines, rendered, block = source.splitlines(), [], None
    for line in lines:
        stripped = line.strip().lower()
        if stripped in {"%block chemicalspecieslabel", "%block atomiccoordinatesandatomicspecies", "%block pao.basis"}:
            block = stripped.removeprefix("%block ")
            rendered.append(line)
            if block == "chemicalspecieslabel":
                rendered.extend(("  1   6  C", "  2  -1  Ghost-H", f"  3  -206  {GHOST_LABEL}  C.psf"))
            elif block == "atomiccoordinatesandatomicspecies":
                rendered.extend(" %.12f  %.12f  %.12f  1  # C" % tuple(p) for p in xyz)
                rendered.extend(" %.12f  %.12f  %.12f  3  # co-moving auxiliary d" % tuple(p) for p in xyz)
            else:
                rendered.extend(GHOST_BLOCK.splitlines())
            continue
        if block:
            if stripped == f"%endblock {block}":
                rendered.append(line)
                block = None
            continue
        if stripped.startswith("systemname"):
            line = f"SystemName {LABEL}"
        elif stripped.startswith("systemlabel"):
            line = f"SystemLabel {LABEL}"
        elif stripped.startswith("numberofatoms"):
            line = "NumberOfAtoms 4"
        elif stripped.startswith("numberofspecies"):
            line = "NumberOfSpecies 3"
        rendered.append(line)
    if block:
        raise ValueError(f"unterminated {block} block")
    rendered.extend(("", "%block SyntheticAtoms", " 3", " 2 2 3 4", " 0.0 0.0 0.0 0.0", "%endblock SyntheticAtoms"))
    return "\n".join(rendered) + "\n"


def produce(output: Path, siesta: str) -> Path:
    import sisl

    run = output / "runs" / LABEL
    run.mkdir(parents=True, exist_ok=True)
    xyz = sisl.get_sile(str(QZ_RUN / "n_qzp_d12_14.TSHS")).read_geometry().xyz
    (run / "RUN.fdf").write_text(render_ghost_fdf((QZ_RUN / "RUN.fdf").read_text(), xyz), encoding="utf-8")
    for pseudo in ("C.psf", "Ghost-H.psf"):
        shutil.copy2(QZ_RUN / pseudo, run / pseudo)
    if not (run / f"{LABEL}.TSHS").is_file():
        record = run_siesta(run, command=siesta, use_shell=False)
        (run / "run_record.json").write_text(json.dumps(record, indent=2) + "\n")
        if record["returncode"]:
            raise RuntimeError(f"SIESTA failed; see {run / 'RUN.out'}")
    return run


def _real_selector(parent, candidate) -> np.ndarray:
    selector = np.zeros((candidate.no, parent.no))
    for atom in range(2):
        old = np.arange(parent.a2o(atom), parent.a2o(atom + 1))
        new = np.arange(candidate.a2o(atom), candidate.a2o(atom + 1))
        if len(old) != len(new):
            raise ValueError("real-C orbital count changed")
        selector[new, old] = 1.0
    return selector


def evaluate(output: Path, run: Path) -> dict:
    import sisl

    qz_h = sisl.get_sile(str(QZ_RUN / "n_qzp_d12_14.TSHS")).read_hamiltonian()
    aug_h = sisl.get_sile(str(run / f"{LABEL}.TSHS")).read_hamiltonian()
    qz_atom = sisl.get_sile(str(QZ_RUN / "C.ion.xml")).read_basis()
    aug_atom = sisl.get_sile(str(run / "C.ion.xml")).read_basis()
    ghost_atom = sisl.get_sile(str(run / f"{GHOST_LABEL}.ion.xml")).read_basis()
    selector = _real_selector(qz_h.geometry, aug_h.geometry)
    radial = [_radial_error(old, new) for old, new in zip(qz_atom.orbitals, aug_atom.orbitals)]
    ghost_indices = np.concatenate([np.arange(aug_h.geometry.a2o(a), aug_h.geometry.a2o(a + 1)) for a in (2, 3)])
    real_indices = np.flatnonzero(selector.sum(axis=1))
    checks, failures = [], []
    if ghost_atom.no != 5 or any(orbital.l != 2 for orbital in ghost_atom.orbitals):
        failures.append(f"ghost contract is {ghost_atom.no} orbitals, l={[orbital.l for orbital in ghost_atom.orbitals]}")
    if max(radial, default=0.0) >= RADIAL_TOL:
        failures.append(f"real-C radial prefix error={max(radial):.6g}")
    for k in K_POINTS:
        s_qz = np.asarray(qz_h.Sk(k=k, format="array"))
        s_aug = np.asarray(aug_h.Sk(k=k, format="array"))
        prefix = np.linalg.norm(selector.T @ s_aug @ selector - s_qz, 2) / np.linalg.norm(s_qz, 2)
        condition = np.linalg.cond(s_aug)
        s_qq, s_qd = s_aug[np.ix_(real_indices, real_indices)], s_aug[np.ix_(real_indices, ghost_indices)]
        s_dd = s_aug[np.ix_(ghost_indices, ghost_indices)]
        schur = s_dd - s_qd.conj().T @ cho_solve(cho_factor((s_qq + s_qq.conj().T) / 2), s_qd)
        eigenvalues = np.linalg.eigvalsh((schur + schur.conj().T) / 2)
        schur_relative = float(eigenvalues.min() / eigenvalues.max())
        checks.append({"k_reduced": list(k), "selected_QZP_overlap_relative": float(prefix),
                       "condition_S_augmented": float(condition), "schur_eigenvalues": eigenvalues.tolist(),
                       "schur_lambda_min_over_max": schur_relative})
        if prefix >= OVERLAP_TOL:
            failures.append(f"k={list(k)}: prefix overlap={prefix:.6g}")
        if condition >= COND_MAX:
            failures.append(f"k={list(k)}: condition={condition:.6g}")
        if schur_relative <= SCHUR_REL_MIN:
            failures.append(f"k={list(k)}: Schur lambda_min/max={schur_relative:.6g}")
    colocated = bool(np.array_equal(aug_h.geometry.xyz[:2], aug_h.geometry.xyz[2:]))
    if not colocated:
        failures.append("ghost centres are not exactly co-located with their real-C parents")
    report = {
        "schema": "ghost_d_augmentation_preflight_v1", "gate": "ghost_d_augmentation_preflight",
        "generated_at": datetime.now(timezone.utc).isoformat(), "verdict": "PASS" if not failures else "NO_GO",
        "basis": "N-QZP+D12/14 plus one co-located zero-occupation synthetic ghost-C 3d centre per physical C",
        "ghost_cutoff_bohr": GHOST_CUTOFF_BOHR, "failure_reasons": failures,
        "thresholds": {"radial_relative": RADIAL_TOL, "selected_overlap_relative": OVERLAP_TOL,
                       "condition_S_augmented": COND_MAX, "schur_lambda_min_over_max": SCHUR_REL_MIN},
        "real_C_radial_prefix_max_relative": max(radial, default=0.0), "ghost_orbitals_per_parent": ghost_atom.no,
        "ghost_orbital_l": [int(o.l) for o in ghost_atom.orbitals], "ghosts_exactly_colocated": colocated,
        "k_checks": checks, "run_dir": str(run), "run_fdf_sha256": file_sha256(run / "RUN.fdf"),
        "tshs_sha256": file_sha256(run / f"{LABEL}.TSHS"),
        "claim_scope": "central auxiliary-d causal-probe suitability only; not QZDP/Delta_out closure; C1:x forbidden unless PASS",
    }
    path = output / "ghost_d_augmentation_preflight.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"ghost-d augmentation preflight: {report['verdict']} -> {path}")
    return report


def generation_failure_report(output: Path, run: Path, reason: str) -> dict:
    report = {
        "schema": "ghost_d_augmentation_preflight_v1",
        "gate": "ghost_d_augmentation_preflight",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "NO_GO",
        "stage": "auxiliary_species_generation",
        "failure_reasons": [reason],
        "attempts": [
            {"atomic_number": -6, "result": "requires occupied 2s/2p shells; not a five-d-only augmentation"},
            {"atomic_number": -100, "result": "build assigns occupied 7s/5f shells; not a five-d-only augmentation"},
            {"atomic_number": -206, "result": "zero occupations accepted, but C.psf rejected: synthetic species not detailed"},
        ],
        "thresholds_not_reached": {
            "radial_relative": RADIAL_TOL, "selected_overlap_relative": OVERLAP_TOL,
            "condition_S_augmented": COND_MAX, "schur_lambda_min_over_max": SCHUR_REL_MIN,
        },
        "run_dir": str(run), "run_output": str(run / "RUN.out"),
        "claim_scope": "generation NO_GO; no TSHS, prefix/Schur adjudication, displaced stencil, or polarization-radial probe",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "ghost_d_augmentation_preflight.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"ghost-d augmentation preflight: NO_GO -> {path}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--siesta", default="/home/christian/bin/siesta")
    args = parser.parse_args(argv)
    try:
        run = produce(args.output, args.siesta)
    except RuntimeError as error:
        generation_failure_report(args.output, args.output / "runs" / LABEL, str(error))
        return 1
    return 0 if evaluate(args.output, run)["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
