#!/usr/bin/env python3
"""Central-geometry preflight for projecting larger SIESTA bases to PROD-SZ.

This intentionally does not run displacement stencils.  It first proves that
cross-basis overlaps can be constructed on the SIESTA real-space grid and that
the resulting Galerkin map is numerically stable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from artifact_signature import file_sha256  # noqa: E402
from certify_epc_energy_zero import vacuum_level  # noqa: E402
from run_hamiltonian_derivative_siesta_references import run_siesta  # noqa: E402

PRODUCTION = REPO_ROOT / "Comparison/results/epc/convergence/gauge_delta/runs/equilibrium"
DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/basis_projection_preflight"
# A generic 3-point conditioning/convergence sample: Gamma, one BZ-corner-region
# probe, and one generic off-symmetry probe. None of these is a physical claim
# about a named high-symmetry point: for PROD-SZ's 60-degree hexagonal cell
# (a1=(2.2005,-1.27046,0), a2=(2.2005,1.27046,0)), (1/3, 1/3, 0) has a minimum
# band gap of ~1.15 eV (verified by direct diagonalisation) and is therefore
# NOT the Dirac K point; the true K/K' are at (1/3, 2/3, 0) and (2/3, 1/3, 0),
# which show ~1e-6 eV degeneracies. Consumers of K_POINTS that need Gamma/K
# specifically (e.g. certify_basis_response.py) define their own K reduced
# coordinate independently and do not rely on this tuple for that purpose.
K_POINTS = ((0.0, 0.0, 0.0), (1 / 3, 1 / 3, 0.0), (0.29, 0.11, 0.0))
K_POINTS_HISTORICAL_LABELS = ("Gamma", "generic_k_probe", "generic_k_probe")
CANDIDATES = (
    ("dzp_es0p02", "DZP", 0.020),
    ("dzp_es0p01", "DZP", 0.010),
    ("tzp_es0p01", "TZP", 0.010),
    ("tzp_es0p005", "TZP", 0.005),
)


def _replace_explicit_basis(text: str, label: str, size: str, energy_shift: float) -> str:
    """Replace the production explicit PAO block with one automatic C basis."""
    lines, kept, inside = text.splitlines(), [], False
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
        kept.append(line)
    if inside:
        raise ValueError("unterminated PAO.Basis block")
    kept.extend((
        "", "# Cross-basis central-geometry preflight",
        f"PAO.BasisSize {size}",
        f"PAO.EnergyShift {energy_shift:.6f} Ry",
        "PAO.SplitNorm 0.15", "PAO.SplitTailNorm T", "PAO.SoftDefault T",
        "PAO.OldStylePolOrbs F",
        "%block PAO.Polarization.Scheme", " C perturbative", "%endblock PAO.Polarization.Scheme",
    ))
    return "\n".join(kept) + "\n"


def prepare_and_run_central(output: Path, siesta: str) -> list[Path]:
    """Run only the four missing central geometries; reuse complete outputs."""
    source = (PRODUCTION / "RUN.fdf").read_text(encoding="utf-8")
    runs = [PRODUCTION]
    for label, size, shift in CANDIDATES:
        run_dir = output / "runs" / label
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "RUN.fdf").write_text(
            _replace_explicit_basis(source, label, size, shift), encoding="utf-8"
        )
        for pseudo in ("C.psf", "Ghost-H.psf"):
            shutil.copy2(PRODUCTION / pseudo, run_dir / pseudo)
        if not (run_dir / f"{label}.TSHS").is_file():
            record = run_siesta(run_dir, command=siesta, use_shell=False)
            (run_dir / "run_record.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            if record["returncode"]:
                raise RuntimeError(f"SIESTA failed for {label}; see {run_dir / 'RUN.out'}")
        runs.append(run_dir)
    return runs


def _basis_geometry(tshs_geometry, ion_path: Path):
    import sisl

    atom = sisl.get_sile(str(ion_path)).read_basis()
    return sisl.Geometry(tshs_geometry.xyz, atoms=[atom] * tshs_geometry.na, lattice=tshs_geometry.lattice)


def _orbital_grid(geometry, shape: tuple[int, int, int]):
    """Sparse real-space PAO values, including every periodic image in range."""
    grid = geometry._orbital_values(shape)  # sisl's native evaluator
    return grid._csr.tocsr(), grid.geometry.sc_off.copy()


def _bloch_values(values, offsets: np.ndarray, no: int, k: tuple[float, float, float]):
    phase = np.repeat(np.exp(2j * np.pi * (offsets @ np.asarray(k))), no)
    fold = sparse.csr_matrix(
        (phase, (np.arange(len(phase)), np.tile(np.arange(no), len(offsets)))),
        shape=(len(phase), no),
    )
    return values @ fold


def _inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.conj().T) / 2)
    if values.min() <= 0:
        raise ValueError(f"overlap is not positive definite: min eigenvalue={values.min():.3e}")
    return (vectors / np.sqrt(values)) @ vectors.conj().T


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(float).tiny))


def evaluate(output: Path, run_dirs: list[Path]) -> dict:
    import sisl

    shape = tuple(int(x) for x in sisl.get_sile(str(PRODUCTION / "equilibrium.VT")).read_grid().shape)
    prod_h = sisl.get_sile(str(PRODUCTION / "equilibrium.TSHS")).read_hamiltonian()
    prod_geometry = _basis_geometry(prod_h.geometry, PRODUCTION / "C.ion.xml")
    prod_values, prod_offsets = _orbital_grid(prod_geometry, shape)
    dvolume = abs(np.linalg.det(prod_geometry.cell)) / np.prod(shape)
    thresholds = {
        "self_overlap_relative_effective": 5e-6,
        "self_overlap_ideal": 1e-8,
        "identity_operator_norm": 1e-5,
        "sigma_min": 0.995,
        "overlap_condition_number": 1e8,
        "metric_reconstruction_relative": 1e-3,
        "projected_hermiticity_relative": 1e-10,
    }

    rows = []
    for run_dir in run_dirs:
        label = "equilibrium" if run_dir == PRODUCTION else run_dir.name
        tshs = run_dir / f"{label}.TSHS"
        vt = run_dir / f"{label}.VT"
        hamiltonian = sisl.get_sile(str(tshs)).read_hamiltonian()
        fermi = float(sisl.get_sile(str(tshs)).read_fermi_level())
        basis_geometry = _basis_geometry(hamiltonian.geometry, run_dir / "C.ion.xml")
        values, offsets = _orbital_grid(basis_geometry, shape)
        c_vacuum = vacuum_level(sisl.get_sile(str(vt)).read_grid().grid)[0]
        per_k = []
        for k in K_POINTS:
            xa = _bloch_values(prod_values, prod_offsets, prod_geometry.no, k)
            xb = _bloch_values(values, offsets, basis_geometry.no, k)
            c_ab = np.asarray((xa.conj().T @ xb).toarray()) * dvolume
            c_bb = np.asarray((xb.conj().T @ xb).toarray()) * dvolume
            s_a = np.asarray(prod_h.Sk(k=k, format="array"))
            s_b = np.asarray(hamiltonian.Sk(k=k, format="array"))
            mapping = np.linalg.solve(s_b, c_ab.conj().T)
            q = _inv_sqrt(s_a) @ c_ab @ _inv_sqrt(s_b)
            sigma_min = float(np.linalg.svd(q, compute_uv=False).min())
            metric = mapping.conj().T @ s_b @ mapping
            k_absolute = np.asarray(hamiltonian.Hk(k=k, format="array")) + fermi * s_b
            projected = mapping.conj().T @ (k_absolute - c_vacuum * s_b) @ mapping
            item = {
                "k_reduced": list(k),
                "candidate_self_overlap_relative": _relative(c_bb, s_b),
                "sigma_min": sigma_min,
                "condition_S_B": float(np.linalg.cond(s_b)),
                "metric_reconstruction_relative": _relative(metric, s_a),
                "projected_hermiticity_relative": _relative(projected, projected.conj().T),
            }
            if label == "equilibrium":
                item["C_AA_minus_S_A_relative"] = _relative(c_ab, s_a)
                item["M_AA_minus_I_operator_norm"] = float(
                    np.linalg.norm(mapping - np.eye(mapping.shape[0]), ord=2)
                )
            per_k.append(item)
        checks = []
        for item in per_k:
            checks.extend((
                item["candidate_self_overlap_relative"] <= thresholds["self_overlap_relative_effective"],
                item["sigma_min"] >= thresholds["sigma_min"],
                item["condition_S_B"] <= thresholds["overlap_condition_number"],
                item["metric_reconstruction_relative"] <= thresholds["metric_reconstruction_relative"],
                item["projected_hermiticity_relative"] <= thresholds["projected_hermiticity_relative"],
            ))
            if label == "equilibrium":
                checks.extend((
                    item["C_AA_minus_S_A_relative"] <= thresholds["self_overlap_relative_effective"],
                    item["M_AA_minus_I_operator_norm"] <= thresholds["identity_operator_norm"],
                ))
        rows.append({
            "label": label,
            "basis_orbitals_per_C": int(basis_geometry.atoms[0].no),
            "basis_orbitals_per_cell": int(basis_geometry.no),
            "run_dir": str(run_dir),
            "tshs_sha256": file_sha256(tshs),
            "ion_xml_sha256": file_sha256(run_dir / "C.ion.xml"),
            "orb_indx_sha256": file_sha256(run_dir / f"{label}.ORB_INDX"),
            "pseudo_sha256": file_sha256(run_dir / "C.psf"),
            "run_fdf_sha256": file_sha256(run_dir / "RUN.fdf"),
            "run_out_sha256": file_sha256(run_dir / "RUN.out"),
            "vacuum_level_ev": c_vacuum,
            "verdict": "PASS" if all(checks) else "NO_GO",
            "k_checks": per_k,
        })

    passed = all(row["verdict"] == "PASS" for row in rows)
    report = {
        "schema": "cross_basis_projection_preflight_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO",
        "claim_scope": "fixed production PAO only; Delta_out and full-KS remain phase 2",
        "method": "external PAO-grid C_AB; project H-c_vacuum*S at each geometry before future FD",
        "stencils_executed": False,
        "grid_shape": list(shape),
        "grid_dvolume_ang3": float(dvolume),
        "k_points_reduced": [list(k) for k in K_POINTS],
        "thresholds": thresholds,
        "threshold_note": "1e-8 is the analytic target; 5e-6 is the measured 600 Ry grid-quadrature gate calibrated by A=A",
        "bases": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "cross_basis_projection_preflight.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"cross-basis projection preflight: {report['verdict']} -> {path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--siesta", default="/home/christian/bin/siesta")
    args = parser.parse_args(argv)
    report = evaluate(args.output, prepare_and_run_central(args.output, args.siesta))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
