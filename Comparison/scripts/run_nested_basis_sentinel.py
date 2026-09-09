#!/usr/bin/env python3
"""N1: displaced exact-selector sentinel for N-TZP+3s(10)/3p(12)."""

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
from certify_epc_energy_zero import vacuum_level  # noqa: E402
from cross_basis_projection_preflight import K_POINTS  # noqa: E402
from nested_basis_identity_preflight import (  # noqa: E402
    BASIS_BLOCKS, COND_MAX, OVERLAP_TOL, PROD_CENTRAL, RADIAL_TOL,
    _radial_error, _selector, render_fdf,
)
from run_displaced_projection_sentinel import (  # noqa: E402
    DIRECTIONS, H_ANG, STEPS, DEFAULT_OUTPUT as OLD_SENTINEL_OUTPUT, _label,
)
from run_hamiltonian_derivative_siesta_references import run_siesta  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/nested_basis_sentinel"
HERM_TOL = 1e-10


def produce(output: Path, siesta: str) -> dict[str, dict[int, Path]]:
    import sisl

    source = (PROD_CENTRAL / "RUN.fdf").read_text(encoding="utf-8")
    xyz0 = sisl.get_sile(str(PROD_CENTRAL / "prod_central.TSHS")).read_geometry().xyz
    central = REPO_ROOT / "Comparison/results/epc/nested_basis_identity_preflight/runs/n_tzp_d10_12"
    result = {}
    for direction, axis in DIRECTIONS.items():
        result[direction] = {0: central}
        for step in (-2, -1, 1, 2):
            xyz = xyz0.copy()
            xyz[0, axis] += step * H_ANG
            tag = f"{'m' if step < 0 else 'p'}{abs(step)}"
            label = f"nested_d10_12_{direction.lower()}_{tag}"
            run_dir = output / "runs" / direction / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            text = render_fdf(source, label, BASIS_BLOCKS["n_tzp_d10_12"])
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
            (run_dir / "RUN.fdf").write_text("\n".join(rendered) + "\n", encoding="utf-8")
            for pseudo in ("C.psf", "Ghost-H.psf"):
                shutil.copy2(PROD_CENTRAL / pseudo, run_dir / pseudo)
            if not (run_dir / f"{label}.TSHS").is_file():
                record = run_siesta(run_dir, command=siesta, use_shell=False)
                (run_dir / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                if record["returncode"]:
                    raise RuntimeError(f"SIESTA failed for {label}; see {run_dir / 'RUN.out'}")
            result[direction][step] = run_dir
    return result


def _production_run(direction: str, step: int) -> Path:
    if step == 0:
        return OLD_SENTINEL_OUTPUT / "runs/PROD-SZ/central"
    tag = f"{'m' if step < 0 else 'p'}{abs(step)}"
    return OLD_SENTINEL_OUTPUT / "runs/PROD-SZ" / direction / tag


def _matrix_data(run_dir: Path):
    import sisl

    label = _label(run_dir)
    sile = sisl.get_sile(str(run_dir / f"{label}.TSHS"))
    return label, sile.read_hamiltonian(), float(sile.read_fermi_level()), vacuum_level(
        sisl.get_sile(str(run_dir / f"{label}.VT")).read_grid().grid
    )[0], sisl.get_sile(str(run_dir / "C.ion.xml")).read_basis()


def evaluate(output: Path, runs: dict[str, dict[int, Path]]) -> dict:
    import sisl

    prod_atom = sisl.get_sile(str(PROD_CENTRAL / "C.ion.xml")).read_basis()
    reports = []
    for direction in DIRECTIONS:
        points, matrices, selector0 = [], {}, None
        for step in STEPS:
            a_dir, b_dir = _production_run(direction, step), runs[direction][step]
            a_label, h_a, _, _, _ = _matrix_data(a_dir)
            b_label, h_b, fermi_b, vacuum_b, atom_b = _matrix_data(b_dir)
            selector, local = _selector(prod_atom, atom_b, h_a.geometry.na)
            selector0 = selector if selector0 is None else selector0
            failures = []
            if not np.array_equal(selector, selector0):
                failures.append("orbital selector changed with displacement")
            radial_max = max(_radial_error(prod_atom.orbitals[i], atom_b.orbitals[j]) for i, j in enumerate(local))
            if radial_max >= RADIAL_TOL:
                failures.append(f"radial identity error={radial_max:.6g}")
            k_checks = []
            for ik, k in enumerate(K_POINTS):
                s_a = np.asarray(h_a.Sk(k=k, format="array"))
                s_b = np.asarray(h_b.Sk(k=k, format="array"))
                h_abs = np.asarray(h_b.Hk(k=k, format="array")) + fermi_b * s_b
                selected_s = selector.T @ s_b @ selector
                selected_k = selector.T @ (h_abs - vacuum_b * s_b) @ selector
                overlap_error = float(np.linalg.norm(selected_s - s_a, ord=2) / np.linalg.norm(s_a, ord=2))
                condition = float(np.linalg.cond(s_b))
                hermiticity = float(np.linalg.norm(selected_k - selected_k.conj().T) / np.linalg.norm(selected_k))
                if overlap_error >= OVERLAP_TOL:
                    failures.append(f"k={ik}: selected overlap error={overlap_error:.6g}")
                if condition >= COND_MAX:
                    failures.append(f"k={ik}: condition={condition:.6g}")
                if hermiticity >= HERM_TOL:
                    failures.append(f"k={ik}: hermiticity={hermiticity:.6g}")
                matrices[(step, ik)] = selected_k
                k_checks.append({"k_reduced": list(k), "selected_overlap_relative": overlap_error, "condition_S_B": condition, "selected_K_hermiticity_relative": hermiticity})
            points.append({
                "step": step, "displacement_ang": step * H_ANG, "verdict": "PASS" if not failures else "NO_GO",
                "failure_reasons": failures, "radial_identity_max": radial_max, "selector_local_indices": local,
                "k_checks": k_checks, "prod_run_dir": str(a_dir), "nested_run_dir": str(b_dir),
                "prod_tshs_sha256": file_sha256(a_dir / f"{a_label}.TSHS"),
                "nested_tshs_sha256": file_sha256(b_dir / f"{b_label}.TSHS"),
            })
        weights = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
        saved = {"selector_E": selector0}
        for ik in range(len(K_POINTS)):
            saved[f"D5_K_prod_subblock_k{ik}"] = sum(weights[step] * matrices[(step, ik)] for step in weights) / (12 * H_ANG)
            for step in STEPS:
                saved[f"K_prod_subblock_j{step}_k{ik}"] = matrices[(step, ik)]
        artifact = output / f"{direction}_nested_selector_matrices.npz"
        np.savez_compressed(artifact, **saved)
        passed = all(point["verdict"] == "PASS" for point in points)
        reports.append({"direction": direction, "verdict": "PASS" if passed else "NO_GO", "points": points, "matrix_artifact": str(artifact), "matrix_artifact_sha256": file_sha256(artifact)})
    passed = all(row["verdict"] == "PASS" for row in reports)
    report = {
        "schema": "displaced_nested_basis_sentinel_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO", "gate": "displaced_nested_sentinel",
        "basis": "N-TZP + 3s(10 bohr) + 3p(12 bohr)", "selector": "constant E_A; no LS/ISO/C_AB",
        "gauge": "E_A^T [H-c_vacuum*S] E_A before five-point FD", "h_ang": H_ANG,
        "thresholds": {"radial_relative": RADIAL_TOL, "selected_overlap_relative": OVERLAP_TOL, "condition_S_B": COND_MAX, "hermiticity_relative": HERM_TOL},
        "claim_scope": "reference_basis_convergence_raw sentinel only; not Delta_out/full-KS",
        "direction_results": reports,
    }
    path = output / "displaced_nested_basis_sentinel.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"displaced nested basis sentinel: {report['verdict']} -> {path}")
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
