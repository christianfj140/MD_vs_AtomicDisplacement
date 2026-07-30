#!/usr/bin/env python3
"""Validate DeepH's Julia shift-invert solver against a dense synthetic problem."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENVIRONMENT = REPO_ROOT / "Comparison/results/graphene_hbn_magic_angle_spectral/solver/environment.json"
DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/graphene_hbn_magic_angle_spectral/solver/synthetic_validation"
EV_TO_HARTREE = 0.036749324533634074


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_openmx_band(path: Path) -> np.ndarray:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_bands = int(lines[0][0])
    n_paths = int(lines[2][0])
    cursor = 3 + n_paths
    values = []
    while cursor < len(lines):
        if int(lines[cursor][0]) != n_bands:
            raise RuntimeError(f"Invalid OpenMX band record at line {cursor + 1}")
        values.append([float(value) / EV_TO_HARTREE for value in lines[cursor + 1]])
        cursor += 2
    return np.asarray(values, dtype=float)


def validate(environment_path: Path, output: Path) -> dict:
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if environment.get("status") != "valid":
        raise RuntimeError(f"Sparse solver environment is not valid: {environment_path}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    hamiltonian = np.diag([-2.0, -0.2, 0.7, 2.0])
    overlap = np.diag([1.0, 1.1, 0.9, 1.0])
    key = "[0, 0, 0, 1, 1]"
    with h5py.File(output / "hamiltonians_pred.h5", "w") as handle:
        handle[key] = hamiltonian
    with h5py.File(output / "overlaps.h5", "w") as handle:
        handle[key] = overlap
    np.savetxt(output / "rlat.dat", np.eye(3))
    np.savetxt(output / "site_positions.dat", np.zeros((3, 1)))
    (output / "orbital_types.dat").write_text("0 0 0 0\n", encoding="utf-8")
    fermi_level = 0.25
    config = {
        "calc_job": "band",
        "which_k": 0,
        "fermi_level": fermi_level,
        "max_iter": 500,
        "num_band": 2,
        "k_data": ["2 0 0 0 0.5 0 0"],
    }
    config_path = output / "band_config.json"
    write_json(config_path, config)
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(output / "time-v.txt"),
        str(environment["julia"]),
        "--startup-file=no",
        f"--project={environment['project']}",
        str(REPO_ROOT.parent / "DeepH-pack/deeph/inference/sparse_calc.jl"),
        "--input_dir",
        str(output),
        "--output_dir",
        str(output),
        "--config",
        str(config_path),
    ]
    env = os.environ.copy()
    env["JULIA_DEPOT_PATH"] = environment["depot"]
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
    (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"DeepH sparse solver failed; see {output / 'stderr.log'}")
    sparse = np.sort(parse_openmx_band(output / "openmx.Band"), axis=1)
    dense = np.sort(np.linalg.eigvalsh(np.linalg.solve(overlap, hamiltonian)))
    expected = dense[np.argsort(np.abs(dense - fermi_level))[:2]]
    expected.sort()
    max_error = float(np.max(np.abs(sparse - expected[None, :])))
    result = {
        "status": "valid" if max_error < 1e-6 else "invalid",
        "max_abs_sparse_dense_error_eV": max_error,
        "threshold_eV": 1e-6,
        "shift_invert_center_eV": fermi_level,
        "sparse_eigenvalues_eV": sparse.tolist(),
        "dense_selected_eigenvalues_eV": expected.tolist(),
        "command": command,
        "returncode": completed.returncode,
        "dense_large_cell_fallback_used": False,
    }
    write_json(output / "validation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = validate(args.environment.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
