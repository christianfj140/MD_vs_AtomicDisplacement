#!/usr/bin/env python3
"""Install and preflight a repo-local Julia/MKL-Pardiso environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
JULIA_VERSION = "1.10.11"
JULIA_ARCHIVE = f"julia-{JULIA_VERSION}-linux-x86_64.tar.gz"
JULIA_URL = f"https://julialang-s3.julialang.org/bin/linux/x64/1.10/{JULIA_ARCHIVE}"
JULIA_SHA256 = "fb49c6b174600cd2051e37ba3f7330f8acf06dd00bce609bab6611387fdb37bf"
DEFAULT_ROOT = REPO_ROOT / "Comparison/.tools/deeph_sparse_solver"
DEFAULT_REPORT = REPO_ROOT / "Comparison/results/graphene_hbn_magic_angle_spectral/solver/environment.json"
DEFAULT_GPU_ROOT = REPO_ROOT / "Comparison/.tools/deeph_sparse_solver_gpu"
DEFAULT_GPU_REPORT = (
    REPO_ROOT / "Comparison/results/graphene_hbn_magic_angle_spectral/solver/environment_gpu_cudss.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_julia(root: Path) -> Path:
    install = root / f"julia-{JULIA_VERSION}"
    executable = install / "bin/julia"
    if executable.exists():
        return executable
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="julia-bootstrap-") as temporary:
        archive = Path(temporary) / JULIA_ARCHIVE
        urllib.request.urlretrieve(JULIA_URL, archive)
        actual = sha256(archive)
        if actual != JULIA_SHA256:
            raise RuntimeError(f"Julia archive checksum mismatch: expected {JULIA_SHA256}, got {actual}")
        with tarfile.open(archive) as handle:
            handle.extractall(root, filter="data")
    if not executable.exists():
        raise RuntimeError(f"Julia extraction did not create {executable}")
    return executable


def bootstrap(root: Path, report_path: Path, backend: str = "cpu_mkl_pardiso") -> dict:
    julia = download_julia(DEFAULT_ROOT if backend == "gpu_cudss" else root)
    project = root / "project"
    depot = root / "depot"
    project.mkdir(parents=True, exist_ok=True)
    depot.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["JULIA_DEPOT_PATH"] = str(depot)
    packages = (
        ["ArgParse", "HDF5", "Pardiso", "Arpack", "LinearMaps", "JLD", "JSON", "CUDA", "CUDSS"]
        if backend == "gpu_cudss"
        else ["ArgParse", "HDF5", "Pardiso", "Arpack", "LinearMaps", "JLD", "JSON"]
    )
    install_code = """
using Pkg
Pkg.activate(ARGS[1])
Pkg.add(split(ARGS[2], ","))
Pkg.precompile()
"""
    install = subprocess.run(
        [str(julia), "--startup-file=no", "-e", install_code, str(project), ",".join(packages)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    (root / "bootstrap.stdout.log").write_text(install.stdout, encoding="utf-8")
    (root / "bootstrap.stderr.log").write_text(install.stderr, encoding="utf-8")
    if install.returncode != 0:
        raise RuntimeError(f"Julia package bootstrap failed; see {root / 'bootstrap.stderr.log'}")
    preflight_code = """
using Pkg
Pkg.activate(ARGS[1])
using SparseArrays, LinearAlgebra, Pardiso, Arpack, LinearMaps, HDF5, JLD, JSON
A = sparse(ComplexF64[4 1; 1 3])
b = ComplexF64[1, 2]
ps = MKLPardisoSolver()
x = solve(ps, A, b)
residual = norm(A*x-b) / norm(b)
println(JSON.json(Dict(
  "status" => (residual < 1e-12 ? "valid" : "invalid"),
  "residual" => residual,
  "julia_version" => string(VERSION),
  "threads" => Threads.nthreads(),
 "pardiso_backend" => "MKLPardisoSolver",
)))
"""
    if backend == "gpu_cudss":
        preflight_code = """
using Pkg
Pkg.activate(ARGS[1])
using SparseArrays, LinearAlgebra, CUDA, CUDA.CUSPARSE, CUDSS, JSON
versions = Dict(dep.name => string(dep.version) for dep in values(Pkg.dependencies()) if dep.version !== nothing)
A = sparse(ComplexF64[2 1; 1 -1])
At = sparse(transpose(tril(A)))
rowptr_gpu = CuVector(Int64.(At.colptr))
colval_gpu = CuVector(Int64.(At.rowval))
nzval_gpu = CuVector(ComplexF64.(At.nzval))
x_gpu = CUDA.zeros(ComplexF64, 2)
b_gpu = CuVector(ComplexF64[1, 2])
solver = CudssSolver(rowptr_gpu, colval_gpu, nzval_gpu, "H", 'L')
cudss("analysis", solver, x_gpu, b_gpu; asynchronous=false)
cudss("factorization", solver, x_gpu, b_gpu; asynchronous=false)
cudss("solve", solver, x_gpu, b_gpu; asynchronous=false)
residual = norm(A * Array(x_gpu) - Array(b_gpu)) / norm(Array(b_gpu))
println(JSON.json(Dict(
  "status" => (residual < 1e-12 ? "valid" : "invalid"),
  "residual" => residual,
  "julia_version" => string(VERSION),
  "cuda_functional" => CUDA.functional(),
  "cuda_runtime" => string(CUDA.runtime_version()),
  "cuda_driver" => string(CUDA.driver_version()),
  "cuda_jl_version" => get(versions, "CUDA", "unknown"),
  "cudss_version" => string(CUDSS.version()),
  "cudss_jl_version" => get(versions, "CUDSS", "unknown"),
  "gpu_name" => CUDA.name(CUDA.device()),
  "cudss_backend" => "CUDSS.jl",
  "matrix_value_type" => "ComplexF64",
  "matrix_index_type" => string(eltype(rowptr_gpu)),
)))
"""
    preflight = subprocess.run(
        [str(julia), "--startup-file=no", f"--project={project}", "-e", preflight_code, str(project)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        result = json.loads(preflight.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        result = {"status": "invalid", "residual": None}
    result.update(
        {
            "returncode": preflight.returncode,
            "julia": str(julia),
            "julia_sha256": JULIA_SHA256,
            "project": str(project),
            "manifest": str(project / "Manifest.toml"),
            "depot": str(depot),
            "bootstrap_stdout": str(root / "bootstrap.stdout.log"),
            "bootstrap_stderr": str(root / "bootstrap.stderr.log"),
            "dense_large_cell_fallback_allowed": False,
            "backend": backend,
        }
    )
    if preflight.returncode != 0:
        result["status"] = "invalid"
        result["error"] = preflight.stderr[-4000:]
    write_json(report_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu_mkl_pardiso", "gpu_cudss"), default="cpu_mkl_pardiso")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root or (DEFAULT_GPU_ROOT if args.backend == "gpu_cudss" else DEFAULT_ROOT)
    report = args.report or (DEFAULT_GPU_REPORT if args.backend == "gpu_cudss" else DEFAULT_REPORT)
    result = bootstrap(root.resolve(), report.resolve(), args.backend)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
