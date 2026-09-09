#!/usr/bin/env python3
"""Certify one-sided PROD-SZ moving-basis connections without new DFT runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from cross_basis_projection_preflight import K_POINTS, _basis_geometry, _bloch_values, _orbital_grid  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_displaced_projection_sentinel import DIRECTIONS, _label  # noqa: E402
from run_nested_basis_ladder import _rms  # noqa: E402
from run_nested_basis_sentinel import _matrix_data, _production_run  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/production_basis_connection"
H_VALUES = (0.010, 0.005, 0.0025, 0.00125)
LIMIT_EV_PER_ANG = 0.001
ADJOINT_REL_TOL = 1e-10
INTRA_ANTIHERMITIAN_REL_TOL = 1e-6


def _geometry(base, atom, direction: str, displacement: float):
    import sisl

    xyz = base.xyz.copy()
    xyz[0, DIRECTIONS[direction]] += displacement
    return sisl.Geometry(xyz, atoms=[atom] * base.na, lattice=base.lattice)


def _connections(direction: str, h: float, base, atom, shape, dvolume, k_points=K_POINTS):
    geometries = {step: _geometry(base, atom, direction, step * h) for step in (-2, -1, 0, 1, 2)}
    grids = {step: _orbital_grid(geometry, shape) for step, geometry in geometries.items()}
    weights = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
    result = []
    for k in k_points:
        x0 = _bloch_values(*grids[0], geometries[0].no, k)
        right = 0.0
        left = 0.0
        for step, weight in weights.items():
            xj = _bloch_values(*grids[step], geometries[step].no, k)
            right = right + weight * np.asarray((x0.conj().T @ xj).toarray()) * dvolume
            left = left + weight * np.asarray((xj.conj().T @ x0).toarray()) * dvolume
        result.append((left / (12 * h), right / (12 * h)))
    return result


def _propagate(delta_left, delta_right, overlap, k0):
    correction = -(delta_left @ np.linalg.solve(overlap, k0) + k0 @ np.linalg.solve(overlap, delta_right))
    return _rms(correction, overlap)[0]


def _local_connections(atom, direction: str, h: float, spacing: float = 0.04):
    """Correlated isolated-atom checks: translated overlaps and spatial FD."""
    axis = DIRECTIONS[direction]
    extent = max(orbital.R for orbital in atom.orbitals) + 0.1
    line = np.arange(-extent, extent + spacing / 2, spacing)
    xx, yy, zz = np.meshgrid(line, line, line, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    del xx, yy, zz
    phi = np.column_stack([orbital.psi(points) for orbital in atom.orbitals])
    weights = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
    right = np.zeros((atom.no, atom.no))
    unit = np.zeros(3)
    unit[axis] = 1.0
    for step, weight in weights.items():
        shifted = np.column_stack([orbital.psi(points - step * h * unit) for orbital in atom.orbitals])
        right += weight * (phi.T @ shifted) * spacing**3
    right /= 12 * h

    eps = 1e-4
    xml = np.empty_like(right)
    for beta, orbital in enumerate(atom.orbitals):
        derivative = (orbital.psi(points + eps * unit) - orbital.psi(points - eps * unit)) / (2 * eps)
        xml[:, beta] = -(phi.T @ derivative) * spacing**3
    return right, xml


def evaluate(output: Path) -> dict:
    import sisl

    central_label, central_h, fermi, vacuum, atom = _matrix_data(PROD_CENTRAL)
    shape = tuple(int(x) for x in sisl.get_sile(str(PROD_CENTRAL / f"{central_label}.VT")).read_grid().shape)
    base = _basis_geometry(central_h.geometry, PROD_CENTRAL / "C.ion.xml")
    dvolume = abs(np.linalg.det(base.cell)) / np.prod(shape)
    reports, saved = [], {}
    all_connections = {}
    for direction in DIRECTIONS:
        per_h = {h: _connections(direction, h, base, atom, shape, dvolume) for h in H_VALUES}
        all_connections[direction] = per_h
        prod_s = {}
        for step in (-2, -1, 0, 1, 2):
            run = _production_run(direction, step)
            label = _label(run)
            prod_s[step] = sisl.get_sile(str(run / f"{label}.TSHS")).read_hamiltonian()
        k_checks, failures = [], []
        for ik, k in enumerate(K_POINTS):
            s0 = np.asarray(central_h.Sk(k=k, format="array"))
            h_abs = np.asarray(central_h.Hk(k=k, format="array")) + fermi * s0
            k0 = h_abs - vacuum * s0
            left, right = per_h[0.005][ik]
            d_s = sum(
                weight * np.asarray(prod_s[step].Sk(k=k, format="array"))
                for step, weight in {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}.items()
            ) / (12 * 0.005)
            adjoint = float(np.linalg.norm(left - right.conj().T) / max(np.linalg.norm(left), np.finfo(float).tiny))
            residual = d_s - left - right
            ds_propagated = _propagate(residual / 2, residual / 2, s0, k0)
            coarse_l, coarse_r = per_h[0.0025][ik]
            fine_l, fine_r = per_h[0.00125][ik]
            step_propagated = _propagate(fine_l - coarse_l, fine_r - coarse_r, s0, k0)
            if adjoint >= ADJOINT_REL_TOL:
                failures.append(f"k={ik}: adjoint residual={adjoint:.6g}")
            if ds_propagated >= LIMIT_EV_PER_ANG:
                failures.append(f"k={ik}: propagated D_S residual={ds_propagated:.6g} eV/Ang")
            if step_propagated >= LIMIT_EV_PER_ANG:
                failures.append(f"k={ik}: propagated step residual={step_propagated:.6g} eV/Ang")
            k_checks.append({
                "k_reduced": list(k), "adjoint_relative": adjoint,
                "D_S_connection_residual_propagated_ev_per_ang": ds_propagated,
                "h_0p0025_vs_0p00125_propagated_ev_per_ang": step_propagated,
            })
            saved[f"S_L_{direction}_k{ik}"] = left
            saved[f"S_R_{direction}_k{ik}"] = right
            saved[f"D_S_TSHS_{direction}_k{ik}"] = d_s

        local_cross, local_xml = _local_connections(atom, direction, 0.00125)
        anti_cross = float(np.linalg.norm(local_cross + local_cross.T) / max(np.linalg.norm(local_cross), np.finfo(float).tiny))
        anti_xml = float(np.linalg.norm(local_xml + local_xml.T) / max(np.linalg.norm(local_xml), np.finfo(float).tiny))
        if anti_cross >= INTRA_ANTIHERMITIAN_REL_TOL:
            failures.append(f"intra-atomic cross antihermiticity={anti_cross:.6g}")
        if anti_xml >= INTRA_ANTIHERMITIAN_REL_TOL:
            failures.append(f"intra-atomic XML antihermiticity={anti_xml:.6g}")
        local_checks = []
        for ik, k in enumerate(K_POINTS):
            s0 = np.asarray(central_h.Sk(k=k, format="array"))
            h_abs = np.asarray(central_h.Hk(k=k, format="array")) + fermi * s0
            k0 = h_abs - vacuum * s0
            delta_r = np.zeros_like(s0)
            delta_r[: atom.no, : atom.no] = local_cross - local_xml
            intra_propagated = _propagate(delta_r.conj().T, delta_r, s0, k0)
            if intra_propagated >= LIMIT_EV_PER_ANG:
                failures.append(f"k={ik}: cross-vs-XML intra-atomic residual={intra_propagated:.6g} eV/Ang")
            local_checks.append({"k_reduced": list(k), "cross_vs_XML_propagated_ev_per_ang": intra_propagated})
        saved[f"A_cross_{direction}"] = local_cross
        saved[f"A_XML_{direction}"] = local_xml
        reports.append({
            "direction": direction, "verdict": "PASS" if not failures else "NO_GO",
            "failure_reasons": failures, "k_checks": k_checks,
            "intra_atomic": {
                "antihermiticity_cross_relative": anti_cross,
                "antihermiticity_XML_relative": anti_xml,
                "checks": local_checks, "cartesian_spacing_ang": 0.04,
            },
        })
    output.mkdir(parents=True, exist_ok=True)
    matrices = output / "production_basis_connection_matrices.npz"
    np.savez_compressed(matrices, **saved)
    passed = all(row["verdict"] == "PASS" for row in reports)
    report = {
        "schema": "production_basis_connection_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO", "gate": "production_basis_connection",
        "backend": "external PROD-SZ PAO cross-geometry grid overlaps; no SCF",
        "h_values_ang": list(H_VALUES), "primary_h_ang": 0.005,
        "thresholds": {"propagated_error_ev_per_ang": LIMIT_EV_PER_ANG, "adjoint_relative": ADJOINT_REL_TOL, "intra_antihermiticity_relative": INTRA_ANTIHERMITIAN_REL_TOL},
        "directions": reports, "matrix_artifact": str(matrices), "matrix_artifact_sha256": file_sha256(matrices),
        "claim_scope": "PROD-SZ one-sided connection including intra-atomic blocks; cross/XML estimators share orbital.psi and are not fully independent; not Delta_out/full-KS",
    }
    path = output / "production_basis_connection.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"production basis connection: {report['verdict']} -> {path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    return 0 if evaluate(args.output)["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
