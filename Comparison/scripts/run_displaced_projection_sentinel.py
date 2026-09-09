#!/usr/bin/env python3
"""Displaced TZP(.005 Ry) projection sentinel for C1:x and C1:z."""

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

from artifact_signature import file_sha256, input_signature_sha256  # noqa: E402
from certify_epc_energy_zero import vacuum_level  # noqa: E402
from cross_basis_projection_preflight import (  # noqa: E402
    K_POINTS, PRODUCTION, _basis_geometry, _bloch_values, _inv_sqrt,
    _orbital_grid, _relative, _replace_explicit_basis,
)
from run_hamiltonian_derivative_siesta_references import run_siesta  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/basis_projection_sentinel"
H_ANG = 0.005
STEPS = (-2, -1, 0, 1, 2)
DIRECTIONS = {"C1_x": 0, "C1_z": 2}
THRESHOLDS = {
    "self_overlap_relative": 1e-5,
    "identity_operator_norm": 1e-5,
    "self_overlap_growth": 3.0,
    "sigma_min": 0.995,
    "overlap_condition_number": 1e8,
    "metric_reconstruction_relative": 1e-3,
    "projected_hermiticity_relative": 1e-10,
    "ls_iso_rms_ev_per_ang": 1e-3,
    "ls_iso_relative": 2e-3,
}


def _render_fdf(source: str, label: str, xyz: np.ndarray, *, large_basis: bool) -> str:
    text = _replace_explicit_basis(source, label, "TZP", 0.005) if large_basis else source
    lines, rendered, inside_coords = text.splitlines(), [], False
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("systemname"):
            line = f"SystemName {label}"
        elif stripped.startswith("systemlabel"):
            line = f"SystemLabel {label}"
        elif stripped.startswith("dm.tolerance"):
            line = "DM.Tolerance 1.d-6"
        if stripped == "%block atomiccoordinatesandatomicspecies":
            inside_coords = True
            rendered.append(line)
            for position in xyz:
                rendered.append(" %.12f  %.12f  %.12f  1  # C" % tuple(position))
            continue
        if inside_coords:
            if stripped == "%endblock atomiccoordinatesandatomicspecies":
                inside_coords = False
                rendered.append(line)
            continue
        rendered.append(line)
    if inside_coords:
        raise ValueError("unterminated AtomicCoordinatesAndAtomicSpecies block")
    return "\n".join(rendered) + "\n"


def _run_geometry(root: Path, label: str, xyz: np.ndarray, *, large_basis: bool, siesta: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source = (PRODUCTION / "RUN.fdf").read_text(encoding="utf-8")
    (root / "RUN.fdf").write_text(_render_fdf(source, label, xyz, large_basis=large_basis), encoding="utf-8")
    for pseudo in ("C.psf", "Ghost-H.psf"):
        shutil.copy2(PRODUCTION / pseudo, root / pseudo)
    if not (root / f"{label}.TSHS").is_file():
        record = run_siesta(root, command=siesta, use_shell=False)
        (root / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if record["returncode"]:
            raise RuntimeError(f"SIESTA failed for {label}; see {root / 'RUN.out'}")
    return root


def produce_runs(output: Path, siesta: str) -> dict[str, dict[str, dict[int, Path]]]:
    import sisl

    xyz0 = sisl.get_sile(str(PRODUCTION / "equilibrium.TSHS")).read_geometry().xyz
    result: dict[str, dict[str, dict[int, Path]]] = {"PROD-SZ": {}, "TZP-.005": {}}
    for basis_name, large in (("PROD-SZ", False), ("TZP-.005", True)):
        central_label = "prod_central" if not large else "tzp005_central"
        central = _run_geometry(output / "runs" / basis_name / "central", central_label, xyz0, large_basis=large, siesta=siesta)
        for direction, axis in DIRECTIONS.items():
            result[basis_name][direction] = {0: central}
            for step in (-2, -1, 1, 2):
                xyz = xyz0.copy()
                xyz[0, axis] += step * H_ANG
                tag = f"{'m' if step < 0 else 'p'}{abs(step)}"
                label = f"{'tzp005' if large else 'prod'}_{direction.lower()}_{tag}"
                result[basis_name][direction][step] = _run_geometry(
                    output / "runs" / basis_name / direction / tag,
                    label, xyz, large_basis=large, siesta=siesta,
                )
    return result


def _label(run_dir: Path) -> str:
    for line in (run_dir / "RUN.fdf").read_text(encoding="utf-8").splitlines():
        if line.strip().lower().startswith("systemlabel"):
            return line.split(None, 1)[1].strip()
    raise ValueError(f"SystemLabel missing in {run_dir / 'RUN.fdf'}")


def _load(run_dir: Path, shape: tuple[int, int, int]):
    import sisl

    label = _label(run_dir)
    sile = sisl.get_sile(str(run_dir / f"{label}.TSHS"))
    hamiltonian = sile.read_hamiltonian()
    geometry = _basis_geometry(hamiltonian.geometry, run_dir / "C.ion.xml")
    values, offsets = _orbital_grid(geometry, shape)
    return {
        "label": label, "run_dir": run_dir, "hamiltonian": hamiltonian,
        "fermi": float(sile.read_fermi_level()), "geometry": geometry,
        "values": values, "offsets": offsets,
        "vacuum": vacuum_level(sisl.get_sile(str(run_dir / f"{label}.VT")).read_grid().grid)[0],
    }


def _sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.conj().T) / 2)
    if values.min() <= 0:
        raise ValueError("matrix is not positive definite")
    return (vectors * np.sqrt(values)) @ vectors.conj().T


def _rms(matrix: np.ndarray, overlap: np.ndarray) -> tuple[float, float]:
    whitened = _inv_sqrt(overlap) @ matrix @ _inv_sqrt(overlap)
    return float(np.linalg.norm(whitened) / np.sqrt(len(overlap))), float(np.abs(whitened).max())


def evaluate(output: Path, runs: dict[str, dict[str, dict[int, Path]]]) -> dict:
    import sisl

    shape = tuple(int(x) for x in sisl.get_sile(str(PRODUCTION / "equilibrium.VT")).read_grid().shape)
    dvolume = abs(np.linalg.det(sisl.get_sile(str(PRODUCTION / "equilibrium.TSHS")).read_geometry().cell)) / np.prod(shape)
    direction_reports = []
    for direction in DIRECTIONS:
        points, arrays = [], {}
        loaded_a = {step: _load(path, shape) for step, path in runs["PROD-SZ"][direction].items()}
        loaded_b = {step: _load(path, shape) for step, path in runs["TZP-.005"][direction].items()}
        for step in STEPS:
            a, b = loaded_a[step], loaded_b[step]
            point = {"step": step, "displacement_ang": step * H_ANG, "k_checks": []}
            for ik, k in enumerate(K_POINTS):
                xa = _bloch_values(a["values"], a["offsets"], a["geometry"].no, k)
                xb = _bloch_values(b["values"], b["offsets"], b["geometry"].no, k)
                c_aa = np.asarray((xa.conj().T @ xa).toarray()) * dvolume
                c_ab = np.asarray((xa.conj().T @ xb).toarray()) * dvolume
                s_a = np.asarray(a["hamiltonian"].Sk(k=k, format="array"))
                s_b = np.asarray(b["hamiltonian"].Sk(k=k, format="array"))
                m_ls = np.linalg.solve(s_b, c_ab.conj().T)
                gram = m_ls.conj().T @ s_b @ m_ls
                m_iso = m_ls @ _inv_sqrt(gram) @ _sqrt(s_a)
                q = _inv_sqrt(s_a) @ c_ab @ _inv_sqrt(s_b)
                h_abs = np.asarray(b["hamiltonian"].Hk(k=k, format="array")) + b["fermi"] * s_b
                k_b = h_abs - b["vacuum"] * s_b
                projected_ls = m_ls.conj().T @ k_b @ m_ls
                projected_iso = m_iso.conj().T @ k_b @ m_iso
                m_aa = np.linalg.solve(s_a, c_aa.conj().T)
                arrays[(step, ik)] = (projected_ls, projected_iso, c_ab, m_ls, m_iso, s_a)
                point["k_checks"].append({
                    "k_reduced": list(k),
                    "C_AA_minus_S_A_relative": _relative(c_aa, s_a),
                    "M_AA_minus_I_operator_norm": float(np.linalg.norm(m_aa - np.eye(len(s_a)), ord=2)),
                    "sigma_min": float(np.linalg.svd(q, compute_uv=False).min()),
                    "condition_S_B": float(np.linalg.cond(s_b)),
                    "metric_reconstruction_relative": _relative(gram, s_a),
                    "projected_ls_hermiticity_relative": _relative(projected_ls, projected_ls.conj().T),
                    "projected_iso_metric_relative": _relative(m_iso.conj().T @ s_b @ m_iso, s_a),
                })
            point["artifacts"] = {
                basis: {
                    "run_dir": str(data[step]), "run_fdf_sha256": file_sha256(data[step] / "RUN.fdf"),
                    "tshs_sha256": file_sha256(data[step] / f"{_label(data[step])}.TSHS"),
                    "ion_xml_sha256": file_sha256(data[step] / "C.ion.xml"),
                    "pseudo_sha256": file_sha256(data[step] / "C.psf"),
                    "vacuum_level_ev": loaded_a[step]["vacuum"] if basis == "PROD-SZ" else loaded_b[step]["vacuum"],
                } for basis, data in (("PROD-SZ", runs["PROD-SZ"][direction]), ("TZP-.005", runs["TZP-.005"][direction]))
            }
            point["structure_sha256"] = input_signature_sha256({"direction": direction, "step": step, "xyz": loaded_a[step]["geometry"].xyz.tolist()})
            points.append(point)

        weights = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
        derivative_checks, saved = [], {}
        for ik, k in enumerate(K_POINTS):
            d_ls = sum(weights[step] * arrays[(step, ik)][0] for step in weights) / (12 * H_ANG)
            d_iso = sum(weights[step] * arrays[(step, ik)][1] for step in weights) / (12 * H_ANG)
            rms, maximum = _rms(d_ls - d_iso, arrays[(0, ik)][5])
            signal, _ = _rms(d_ls, arrays[(0, ik)][5])
            derivative_checks.append({
                "k_reduced": list(k), "ls_iso_rms_ev_per_ang": rms,
                "ls_iso_relative": rms / max(signal, np.finfo(float).tiny),
                "ls_iso_max_ev_per_ang": maximum, "ls_signal_rms_ev_per_ang": signal,
            })
            saved[f"D_LS_k{ik}"] = d_ls
            saved[f"D_ISO_k{ik}"] = d_iso
            for step in STEPS:
                saved[f"C_AB_j{step}_k{ik}"] = arrays[(step, ik)][2]
                saved[f"M_LS_j{step}_k{ik}"] = arrays[(step, ik)][3]
                saved[f"M_ISO_j{step}_k{ik}"] = arrays[(step, ik)][4]
        matrix_path = output / f"{direction}_projection_matrices.npz"
        np.savez_compressed(matrix_path, **saved)

        central = next(point for point in points if point["step"] == 0)
        failures = []
        for point in points:
            for ik, check in enumerate(point["k_checks"]):
                floor = central["k_checks"][ik]["C_AA_minus_S_A_relative"]
                limits = {
                    "C_AA_minus_S_A_relative": min(THRESHOLDS["self_overlap_relative"], THRESHOLDS["self_overlap_growth"] * floor),
                    "M_AA_minus_I_operator_norm": THRESHOLDS["identity_operator_norm"],
                    "condition_S_B": THRESHOLDS["overlap_condition_number"],
                    "metric_reconstruction_relative": THRESHOLDS["metric_reconstruction_relative"],
                    "projected_ls_hermiticity_relative": THRESHOLDS["projected_hermiticity_relative"],
                }
                for name, limit in limits.items():
                    if check[name] > limit:
                        failures.append(f"step={point['step']} k={ik}: {name}={check[name]:.6g} > {limit:.6g}")
                if check["sigma_min"] < THRESHOLDS["sigma_min"]:
                    failures.append(f"step={point['step']} k={ik}: sigma_min={check['sigma_min']:.6g} < {THRESHOLDS['sigma_min']:.6g}")
        for check in derivative_checks:
            if check["ls_iso_rms_ev_per_ang"] >= THRESHOLDS["ls_iso_rms_ev_per_ang"]:
                failures.append(f"k={check['k_reduced']}: ls_iso_rms={check['ls_iso_rms_ev_per_ang']:.6g} eV/Ang >= 0.001")
            if check["ls_iso_relative"] >= THRESHOLDS["ls_iso_relative"]:
                failures.append(f"k={check['k_reduced']}: ls_iso_relative={check['ls_iso_relative']:.6g} >= 0.002")
        passed = not failures
        direction_reports.append({
            "direction": direction, "verdict": "PASS" if passed else "NO_GO", "points": points,
            "five_point_derivative": derivative_checks, "matrix_artifact": str(matrix_path),
            "matrix_artifact_sha256": file_sha256(matrix_path), "failure_reasons": failures,
        })

    passed = all(row["verdict"] == "PASS" for row in direction_reports)
    report = {
        "schema": "displaced_projection_sentinel_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO", "basis": "TZP, EnergyShift=0.005 Ry",
        "directions": list(DIRECTIONS), "h_ang": H_ANG, "stencil": "five-point central",
        "scf_tolerance": 1e-6, "mesh_cutoff_ry": 600, "kgrid": [20, 20, 1],
        "gauge": "H-c_vacuum*S before projection", "map_policy": "recompute C_AB and M(R_j,k) before finite difference",
        "claim_scope": "reference_basis_convergence_raw sentinel only; not Delta_out or full-KS",
        "thresholds": THRESHOLDS, "grid_shape": list(shape), "direction_results": direction_reports,
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "displaced_projection_sentinel.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"displaced projection sentinel: {report['verdict']} -> {path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--siesta", default="/home/christian/bin/siesta")
    args = parser.parse_args(argv)
    report = evaluate(args.output, produce_runs(args.output, args.siesta))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
