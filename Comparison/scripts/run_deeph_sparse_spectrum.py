#!/usr/bin/env python3
"""Run DeepH's validated sparse solver and emit UI-ready band/DOS JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from generate_siesta_overlap_only import file_sha256, parse_time_v
from validate_deeph_sparse_solver import parse_openmx_band


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENVIRONMENT = REPO_ROOT / "Comparison/results/graphene_hbn_magic_angle_spectral/solver/environment.json"
DEFAULT_GPU_ENVIRONMENT = (
    REPO_ROOT / "Comparison/results/graphene_hbn_magic_angle_spectral/solver/environment_gpu_cudss.json"
)
K_PATH = [
    ("Γ", (0.0, 0.0, 0.0)),
    ("K", (1.0 / 3.0, 1.0 / 3.0, 0.0)),
    ("M", (0.5, 0.0, 0.0)),
    ("Γ", (0.0, 0.0, 0.0)),
]
GIB = 1024**3
MIN_FREE_DISK_PERCENT = 12.0
RESERVED_SYSTEM_MEMORY_BYTES = 12 * GIB
MIN_SOLVER_MEMORY_BYTES = 8 * GIB
MIN_AVAILABLE_MEMORY_BYTES = 20 * GIB
PARDISO_OOC_MAX_CORE_MB = 8192
SOLVER_THREADS = 8
DISK_POLL_SECONDS = 5.0
MAX_CPU_TEMPERATURE_C = 80.0
MAX_GPU_TEMPERATURE_C = 80.0
DEFAULT_GPU_MEMORY_LIMIT_GIB = 28.0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def free_disk_percent(path: Path) -> float:
    disk = shutil.disk_usage(path)
    return 100.0 * disk.free / disk.total


def cpu_package_temperature_c() -> float | None:
    temperatures = []
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            zone_type = (zone / "type").read_text(encoding="utf-8").strip()
            if zone_type not in {"x86_pkg_temp", "TCPU", "TCPU_PCI"}:
                continue
            temperatures.append(float((zone / "temp").read_text(encoding="utf-8")) / 1000.0)
        except (OSError, ValueError):
            continue
    return max(temperatures) if temperatures else None


def gpu_status() -> dict[str, float] | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        temperature, used_mib, free_mib = (
            float(value.strip()) for value in result.stdout.splitlines()[0].split(",")
        )
    except (IndexError, ValueError):
        return None
    return {
        "temperature_c": temperature,
        "used_bytes": used_mib * 1024**2,
        "free_bytes": free_mib * 1024**2,
    }


def run_with_disk_guard(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    output_dir: Path,
    poll_seconds: float = DISK_POLL_SECONDS,
    gpu_memory_limit_bytes: int | None = None,
    gpu_observations: dict | None = None,
) -> tuple[int, str | None, float, float | None, int]:
    abort_reason = None
    minimum_observed = free_disk_percent(output_dir)
    maximum_cpu_temperature = cpu_package_temperature_c()
    minimum_available_memory = available_memory_bytes()
    initial_gpu = gpu_status() if gpu_memory_limit_bytes is not None else None
    baseline_gpu_used = initial_gpu["used_bytes"] if initial_gpu else 0
    with (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
        output_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        while process.poll() is None:
            current = free_disk_percent(output_dir)
            minimum_observed = min(minimum_observed, current)
            current_available_memory = available_memory_bytes()
            minimum_available_memory = min(minimum_available_memory, current_available_memory)
            temperature = cpu_package_temperature_c()
            if temperature is not None:
                maximum_cpu_temperature = max(maximum_cpu_temperature or temperature, temperature)
            gpu = gpu_status() if gpu_memory_limit_bytes is not None else None
            if gpu is not None and gpu_observations is not None:
                gpu_observations["maximum_temperature_c"] = max(
                    gpu_observations.get("maximum_temperature_c", gpu["temperature_c"]),
                    gpu["temperature_c"],
                )
                gpu_observations["maximum_used_bytes"] = max(
                    gpu_observations.get("maximum_used_bytes", gpu["used_bytes"]),
                    gpu["used_bytes"],
                )
            if current < MIN_FREE_DISK_PERCENT:
                abort_reason = "disk_headroom_runtime"
            elif current_available_memory < MIN_AVAILABLE_MEMORY_BYTES:
                abort_reason = "memory_headroom_runtime"
            elif temperature is not None and temperature >= MAX_CPU_TEMPERATURE_C:
                abort_reason = "cpu_temperature"
            elif gpu is not None and gpu["temperature_c"] >= MAX_GPU_TEMPERATURE_C:
                abort_reason = "gpu_temperature"
            elif gpu is not None and gpu["used_bytes"] - baseline_gpu_used >= gpu_memory_limit_bytes:
                abort_reason = "gpu_memory_limit"
            if abort_reason is not None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(poll_seconds)
        returncode = process.wait()
    return returncode, abort_reason, minimum_observed, maximum_cpu_temperature, minimum_available_memory


def band_kpoints(points_per_segment: int, gamma_only: bool = False) -> list[tuple[str, np.ndarray]]:
    if gamma_only:
        return [("Γ", np.asarray(K_PATH[0][1], dtype=float))]
    if points_per_segment < 1:
        raise ValueError("points_per_segment must be positive")
    if points_per_segment == 1:
        return [(label, np.asarray(point, dtype=float)) for label, point in K_PATH[:-1]]
    points = [(K_PATH[0][0], np.asarray(K_PATH[0][1], dtype=float))]
    for (_left_label, left), (right_label, right) in zip(K_PATH[:-1], K_PATH[1:], strict=True):
        for step, point in enumerate(np.linspace(left, right, points_per_segment)[1:], start=1):
            points.append((right_label if step == points_per_segment - 1 else "", point))
    return points


def band_k_data(points_per_segment: int, gamma_only: bool = False) -> list[str]:
    return [
        f"1 {' '.join(str(value) for value in point)} {' '.join(str(value) for value in point)}"
        for _label, point in band_kpoints(points_per_segment, gamma_only)
    ]


def band_points(
    energies: np.ndarray,
    points_per_segment: int,
    reciprocal: np.ndarray,
    fermi: float,
    gamma_only: bool = False,
) -> list[dict]:
    samples = band_kpoints(points_per_segment, gamma_only)
    if len(energies) != len(samples):
        raise ValueError(f"Expected {len(samples)} k-points, received {len(energies)}")
    labels = [label for label, _point in samples]
    kpoints = [point for _label, point in samples]
    distances = [0.0]
    for left, right in zip(kpoints, kpoints[1:], strict=False):
        distances.append(distances[-1] + float(np.linalg.norm((right - left) @ reciprocal)))
    return [
        {
            "k_index": k_index,
            "band_index": band_index,
            "k_distance": distances[k_index],
            "k_label": labels[k_index],
            "energy_eV": float(energy),
            "energy_aligned_eV": float(energy - fermi),
        }
        for k_index, row in enumerate(energies)
        for band_index, energy in enumerate(row)
    ]


def run(
    input_dir: Path,
    output_dir: Path,
    *,
    job: str,
    fermi_level: float,
    num_bands: int,
    points_per_segment: int,
    kmesh: tuple[int, int, int],
    environment_path: Path,
    backend: str = "cpu_mkl_pardiso",
    gpu_hybrid_memory: bool = True,
    gpu_memory_limit_gib: float = DEFAULT_GPU_MEMORY_LIMIT_GIB,
    validate_gpu_residual: bool = False,
    gamma_only: bool = False,
) -> dict:
    if backend not in {"cpu_mkl_pardiso", "gpu_cudss"}:
        raise ValueError(f"Unsupported sparse solver backend: {backend}")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if environment.get("status") != "valid":
        raise RuntimeError(f"Invalid sparse solver environment: {environment_path}")
    for name in ("hamiltonians_pred.h5", "overlaps.h5", "rlat.dat", "site_positions.dat", "orbital_types.dat"):
        if not (input_dir / name).is_file():
            raise RuntimeError(f"Missing sparse solver input: {input_dir / name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_free_disk_percent = free_disk_percent(output_dir)
    available_memory = available_memory_bytes()
    memory_limit = max(0, available_memory - RESERVED_SYSTEM_MEMORY_BYTES)
    resource_gate = {
        "available_memory_bytes": available_memory,
        "reserved_system_memory_bytes": RESERVED_SYSTEM_MEMORY_BYTES,
        "solver_address_space_limit_bytes": memory_limit,
        "free_disk_percent": initial_free_disk_percent,
        "minimum_free_disk_percent": MIN_FREE_DISK_PERCENT,
        "minimum_available_memory_bytes": MIN_AVAILABLE_MEMORY_BYTES,
        "backend_requested": backend,
    }
    initial_gpu = gpu_status() if backend == "gpu_cudss" else None
    resource_gate["gpu_before"] = initial_gpu
    gpu_unavailable = backend == "gpu_cudss" and (
        initial_gpu is None or initial_gpu["free_bytes"] < GIB
    )
    if (
        initial_free_disk_percent < MIN_FREE_DISK_PERCENT
        or available_memory < MIN_AVAILABLE_MEMORY_BYTES
        or memory_limit < MIN_SOLVER_MEMORY_BYTES
        or gpu_unavailable
    ):
        reason = (
            "disk_headroom"
            if initial_free_disk_percent < MIN_FREE_DISK_PERCENT
            else "memory_headroom"
            if available_memory < MIN_AVAILABLE_MEMORY_BYTES or memory_limit < MIN_SOLVER_MEMORY_BYTES
            else "gpu_unavailable_or_low_memory"
        )
        manifest = {
            "status": "resource_blocked",
            "job": job,
            "returncode": None,
            "reason": reason,
            "resource_gate": resource_gate,
            "dense_large_cell_fallback_used": False,
            "identity_overlap_used": False,
            "backend_requested": backend,
            "backend_effective": None,
        }
        write_json(output_dir / "solver_manifest.json", manifest)
        return manifest
    config = {
        "calc_job": job,
        "fermi_level": fermi_level,
        "max_iter": 1000,
        "num_band": num_bands,
        "pardiso_out_of_core": backend == "cpu_mkl_pardiso",
        "solver_backend": backend,
    }
    if job == "band":
        config.update({"which_k": 0, "k_data": band_k_data(points_per_segment, gamma_only)})
    else:
        config.update(
            {
                "kmesh": list(kmesh),
                "epsilon": 0.01,
                "omegas": [-0.3, 0.3, 601],
            }
        )
    config_path = output_dir / f"{job}_config.json"
    write_json(config_path, config)
    solver_script = (
        REPO_ROOT / "Comparison/scripts/deeph_sparse_calc_gpu.jl"
        if backend == "gpu_cudss"
        else REPO_ROOT.parent / "DeepH-pack/deeph/inference/sparse_calc.jl"
    )
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(output_dir / "time-v.txt"),
        str(environment["julia"]),
        "--startup-file=no",
        f"--project={environment['project']}",
        str(solver_script),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--config",
        str(config_path),
    ]
    env = os.environ.copy()
    env["JULIA_DEPOT_PATH"] = environment["depot"]
    process_threads = 1 if backend == "gpu_cudss" else SOLVER_THREADS
    env["OMP_NUM_THREADS"] = str(process_threads)
    env["MKL_NUM_THREADS"] = str(process_threads)
    env["MKL_PARDISO_OOC_MAX_CORE_SIZE"] = str(PARDISO_OOC_MAX_CORE_MB)
    env["MKL_PARDISO_OOC_MAX_SWAP_SIZE"] = "0"
    env["MKL_PARDISO_OOC_KEEP_FILE"] = "0"
    gpu_memory_limit_bytes = int(gpu_memory_limit_gib * GIB) if backend == "gpu_cudss" else None
    if backend == "gpu_cudss":
        env["DEEPH_SPARSE_CALC"] = str(REPO_ROOT.parent / "DeepH-pack/deeph/inference/sparse_calc.jl")
        env["CUDSS_HYBRID_MEMORY"] = "1" if gpu_hybrid_memory else "0"
        env["CUDSS_DEVICE_MEMORY_LIMIT_BYTES"] = str(gpu_memory_limit_bytes)
        env["CUDSS_HOST_THREADS"] = "1"
        env["CUDSS_VALIDATE_RESIDUAL"] = "1" if validate_gpu_residual else "0"
    gpu_observations: dict = {}
    with tempfile.TemporaryDirectory(prefix="pardiso-ooc-", dir=output_dir) as ooc_dir:
        env["MKL_PARDISO_OOC_PATH"] = ooc_dir
        (
            returncode,
            runtime_abort_reason,
            minimum_free_disk_percent,
            maximum_cpu_temperature,
            minimum_available_memory,
        ) = run_with_disk_guard(
            command,
            cwd=REPO_ROOT,
            env=env,
            output_dir=output_dir,
            gpu_memory_limit_bytes=gpu_memory_limit_bytes,
            gpu_observations=gpu_observations,
        )
    stderr_text = (output_dir / "stderr.log").read_text(encoding="utf-8")
    stdout_text = (output_dir / "stdout.log").read_text(encoding="utf-8")
    stderr_lower = stderr_text.lower()
    oom_detected = any(
        marker in stderr_lower
        for marker in (
            "out of memory",
            "outofmemoryerror",
            "memoryerror",
            "failed to allocate",
            "cannot allocate memory",
            "insufficient memory",
            "insufficient_memory",
            "not enough memory",
            "cudss_status_alloc_failed",
            "cuda_error_out_of_memory",
        )
    )
    resource_blocked = runtime_abort_reason is not None or returncode in {-9, -6, 134, 137} or oom_detected
    timing_matches = re.findall(
        r"cuDSS timings: analysis_seconds=([0-9.eE+-]+) "
        r"factorization_seconds=([0-9.eE+-]+) solve_seconds=([0-9.eE+-]+) solve_count=(\d+)",
        stdout_text,
    )
    cudss_timings = (
        {
            "analysis_seconds": sum(float(item[0]) for item in timing_matches),
            "factorization_seconds": sum(float(item[1]) for item in timing_matches),
            "solve_seconds": sum(float(item[2]) for item in timing_matches),
            "solve_count": sum(int(item[3]) for item in timing_matches),
        }
        if timing_matches
        else None
    )
    residual_matches = re.findall(
        r"cuDSS maximum relative solve residual: ([0-9.eE+-]+)",
        stdout_text,
    )
    manifest = {
        "status": "resource_blocked" if resource_blocked else "completed" if returncode == 0 else "failed",
        "job": job,
        "returncode": returncode,
        "reason": (
            runtime_abort_reason
            or ("gpu_out_of_memory" if oom_detected and backend == "gpu_cudss" else None)
            or ("solver_failed" if returncode != 0 else None)
        ),
        "error_summary": stderr_text[-2000:] if returncode != 0 else None,
        "backend_requested": backend,
        "backend_effective": backend if returncode == 0 else None,
        "fermi_level_eV": fermi_level,
        "num_bands": num_bands,
        "resources": parse_time_v(output_dir / "time-v.txt"),
        "resource_gate": resource_gate,
        "minimum_observed_free_disk_percent": minimum_free_disk_percent,
        "minimum_observed_available_memory_bytes": minimum_available_memory,
        "maximum_observed_cpu_temperature_c": maximum_cpu_temperature,
        "maximum_allowed_cpu_temperature_c": MAX_CPU_TEMPERATURE_C,
        "pardiso_out_of_core": backend == "cpu_mkl_pardiso",
        "pardiso_ooc_max_core_mb": PARDISO_OOC_MAX_CORE_MB if backend == "cpu_mkl_pardiso" else None,
        "solver_threads": process_threads,
        "gpu_hybrid_memory": gpu_hybrid_memory if backend == "gpu_cudss" else None,
        "gpu_memory_limit_bytes": gpu_memory_limit_bytes,
        "gpu_observations": gpu_observations if backend == "gpu_cudss" else None,
        "gpu_hardware": (
            {
                key: environment.get(key)
                for key in (
                    "gpu_name",
                    "cuda_runtime",
                    "cuda_driver",
                    "cuda_jl_version",
                    "cudss_version",
                    "cudss_jl_version",
                )
            }
            if backend == "gpu_cudss"
            else None
        ),
        "cudss_timings": cudss_timings,
        "maximum_allowed_gpu_temperature_c": MAX_GPU_TEMPERATURE_C if backend == "gpu_cudss" else None,
        "maximum_relative_solve_residual": (
            max(float(value) for value in residual_matches)
            if residual_matches
            else None
        ),
        "dense_large_cell_fallback_used": False,
        "identity_overlap_used": False,
        "hamiltonian_sha256": file_sha256(input_dir / "hamiltonians_pred.h5"),
        "overlap_sha256": file_sha256(input_dir / "overlaps.h5"),
        "command": command,
    }
    if returncode == 0 and job == "band":
        energies = parse_openmx_band(output_dir / "openmx.Band")
        reciprocal = np.loadtxt(input_dir / "rlat.dat").T
        manifest["bands"] = band_points(
            energies,
            points_per_segment,
            reciprocal,
            fermi_level,
            gamma_only,
        )
    elif returncode == 0:
        dos = np.loadtxt(output_dir / "dos.dat")
        manifest["low_energy_dos"] = [
            {
                "energy_aligned_eV": float(row[0]),
                "energy_eV": float(row[0] + fermi_level),
                "dos": float(row[1]),
            }
            for row in dos
        ]
    write_json(output_dir / "solver_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job", choices=("band", "dos"), required=True)
    parser.add_argument("--fermi-level", type=float, required=True)
    parser.add_argument("--num-bands", type=int, default=60)
    parser.add_argument("--points-per-segment", type=int, default=4)
    parser.add_argument("--kmesh", type=int, nargs=3, default=(3, 3, 1))
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--backend", choices=("cpu_mkl_pardiso", "gpu_cudss"), default="cpu_mkl_pardiso")
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=DEFAULT_GPU_MEMORY_LIMIT_GIB)
    parser.add_argument("--no-gpu-hybrid-memory", action="store_true")
    parser.add_argument("--validate-gpu-residual", action="store_true")
    parser.add_argument("--gamma-only", action="store_true")
    args = parser.parse_args()
    environment = (
        DEFAULT_GPU_ENVIRONMENT
        if args.backend == "gpu_cudss" and args.environment == DEFAULT_ENVIRONMENT
        else args.environment
    )
    result = run(
        args.input_dir.resolve(),
        args.output_dir.resolve(),
        job=args.job,
        fermi_level=args.fermi_level,
        num_bands=args.num_bands,
        points_per_segment=args.points_per_segment,
        kmesh=tuple(args.kmesh),
        environment_path=environment.resolve(),
        backend=args.backend,
        gpu_hybrid_memory=not args.no_gpu_hybrid_memory,
        gpu_memory_limit_gib=args.gpu_memory_limit_gib,
        validate_gpu_residual=args.validate_gpu_residual,
        gamma_only=args.gamma_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2 if result["status"] == "resource_blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
