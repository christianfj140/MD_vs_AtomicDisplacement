#!/usr/bin/env python3
"""6x6 ab-initio MD label-budget campaign, deliberately specific and resumable."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import sisl


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "MD/scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dataset_design_curves_v1 as c  # noqa: E402
from generate_md_dataset import parse_xv_geometry, rewrite_run_fdf_from_xv  # noqa: E402
from siesta_output_status import parse_siesta_output  # noqa: E402


ROOT = c.OUT / "md_label_budget_6x6"
WORK = ROOT / "work"
CANONICAL = c.OUT / "coverage_6x6_r012/_canonical_base.fdf"
SOURCE_SAMPLE = c.OUT / "coverage_6x6_r012/siesta_hamiltonians/coverage_6x6_3D_R012__pt0000"
SIESTA = Path("/home/christian/bin/siesta")
GRAPH2MAT = REPO_ROOT / ".venv/bin/graph2mat"
LABEL = "graphene_6x6"
TEMPERATURES = (150, 300, 450)
SPLITS = ("train", "validation", "test")
TRAIN_SIZES = (4, 8, 16, 32, 64)
VAL_SIZES = (8, 24, 48)
TEST_SIZES = (4, 8, 16, 24, 32, 48)
SEEDS = {
    (split, temperature): 2026092200 + split_index * 10 + temperature_index + 1
    for split_index, split in enumerate(SPLITS)
    for temperature_index, temperature in enumerate(TEMPERATURES)
}
BOHR_TO_ANG = 0.529177210903
AMU_KG = 1.66053906660e-27
KB_J_K = 1.380649e-23
CARBON_MASS_AMU = 12.011
NOSE_MASS = "100.0 Ry*fs**2"
STORE_FILES = "*fdf *TSHS *TSDE *XV *HSX *STRUCT_OUT *ORB_INDX *out"
ENV = os.environ | {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_ACTIVE: dict[int, dict[str, Any]] = {}
_ACTIVE_LOCK = threading.Lock()
_COOL_SINCE: float | None = None
_LAST_RESUME = 0.0
_INCIDENT_ACTIVE = False


def log(message: str, **fields: Any) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "message": message, **fields}
    line = json.dumps(record, sort_keys=True, default=str)
    print(line, flush=True)
    with (ROOT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block(text: str, name: str) -> list[list[str]]:
    match = re.search(
        rf"(?ims)^\s*%block\s+{re.escape(name)}\s*$\n(.*?)^\s*%endblock\s+{re.escape(name)}\s*$",
        text,
    )
    if not match:
        raise RuntimeError(f"missing %{name} block in canonical FDF")
    return [line.split() for line in match.group(1).splitlines() if line.strip() and not line.lstrip().startswith("#")]


def directive(text: str, key: str) -> str | None:
    for line in text.splitlines():
        clean = line.split("#", 1)[0].strip().split(None, 1)
        if clean and clean[0].lower() == key.lower():
            return clean[1].strip() if len(clean) == 2 else ""
    return None


def directive_map(text: str) -> dict[str, str]:
    output, in_block = {}, False
    for line in text.splitlines():
        clean = line.split("#", 1)[0].strip()
        if clean.lower().startswith("%block"):
            in_block = True
            continue
        if clean.lower().startswith("%endblock"):
            in_block = False
            continue
        if clean and not in_block:
            parts = clean.split(None, 1)
            output[parts[0].lower()] = parts[1].strip() if len(parts) == 2 else ""
    return output


def canonical_geometry() -> tuple[np.ndarray, np.ndarray]:
    text = CANONICAL.read_text(encoding="utf-8")
    lattice = np.asarray([[float(value) for value in row[:3]] for row in block(text, "LatticeVectors")])
    positions = np.asarray([[float(value) for value in row[:3]]
                            for row in block(text, "AtomicCoordinatesAndAtomicSpecies")])
    return lattice, positions


def maxwell_velocities(temperature: float, seed: int, n_atoms: int = 72) -> np.ndarray:
    """Carbon velocities in Bohr/fs, with zero COM and exact 3N-3 kinetic T."""

    rng = np.random.default_rng(seed)
    sigma_m_s = math.sqrt(KB_J_K * temperature / (CARBON_MASS_AMU * AMU_KG))
    velocities = rng.normal(0.0, sigma_m_s, size=(n_atoms, 3))
    velocities -= velocities.mean(axis=0)
    kinetic_j = 0.5 * CARBON_MASS_AMU * AMU_KG * float(np.square(velocities).sum())
    measured = 2.0 * kinetic_j / ((3 * n_atoms - 3) * KB_J_K)
    velocities *= math.sqrt(temperature / measured)
    return velocities * 1e-5 / BOHR_TO_ANG


def kinetic_temperature(velocities_bohr_fs: np.ndarray) -> float:
    velocities_m_s = velocities_bohr_fs * BOHR_TO_ANG / 1e-5
    kinetic_j = 0.5 * CARBON_MASS_AMU * AMU_KG * float(np.square(velocities_m_s).sum())
    return 2.0 * kinetic_j / ((velocities_m_s.size - 3) * KB_J_K)


def write_xv(path: Path, lattice_ang: np.ndarray, positions_ang: np.ndarray,
             velocities_bohr_fs: np.ndarray) -> None:
    lines = [" ".join(f"{value / BOHR_TO_ANG:20.12f}" for value in row) + "  0.0 0.0 0.0"
             for row in lattice_ang]
    lines.append(str(len(positions_ang)))
    for position, velocity in zip(positions_ang / BOHR_TO_ANG, velocities_bohr_fs):
        values = " ".join(f"{value:20.12f}" for value in np.r_[position, velocity])
        lines.append(f"  1  6 {values}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_xv_velocities(path: Path) -> np.ndarray:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_atoms = int(lines[3].split()[0])
    return np.asarray([[float(value) for value in line.split()[5:8]] for line in lines[4:4 + n_atoms]])


def replace_directive(text: str, key: str, value: str) -> str:
    output, found = [], False
    for line in text.splitlines():
        clean = line.split("#", 1)[0].strip().split(None, 1)
        if clean and clean[0].lower() == key.lower():
            if not found:
                output.append(f"{key:<32} {value}")
                found = True
        else:
            output.append(line)
    if not found:
        output.append(f"{key:<32} {value}")
    return "\n".join(output).rstrip() + "\n"


def md_fdf(ensemble: str, temperature: int, steps: int, timestep_fs: float,
           store: bool) -> str:
    text = CANONICAL.read_text(encoding="utf-8")
    settings = {
        "DM.UseSaveDM": "T",
        "MD.UseSaveXV": "T",
        "MD.TypeOfRun": "Nose" if ensemble == "nvt" else "Verlet",
        "MD.Steps": str(steps),
        "MD.InitialTimeStep": "1",
        "MD.FinalTimeStep": str(steps),
        "MD.LengthTimeStep": f"{timestep_fs:g} fs",
        "MD.InitialTemperature": f"{temperature} K",
        "WriteMDHistory": "T",
        "SaveHS": "T" if store else "F",
        "Save.HS": "T" if store else "F",
        "TS.HS.Save": "T" if store else "F",
    }
    if ensemble == "nvt":
        settings |= {"MD.TargetTemperature": f"{temperature} K", "MD.NoseMass": NOSE_MASS}
    if store:
        settings["Lua.Script"] = "md_store.lua"
    for key, value in settings.items():
        text = replace_directive(text, key, value)
    return text


def preflight() -> dict[str, Any]:
    text = CANONICAL.read_text(encoding="utf-8")
    expected = {
        "NumberOfAtoms": "72",
        "XC.functional": "GGA",
        "XC.authors": "PBE",
        "MeshCutoff": "600.0 Ry",
        "MaxSCFIterations": "500",
        "DM.MixingWeight": "0.1",
        "DM.NumberPulay": "5",
        "DM.Tolerance": "1.d-4",
        "ElectronicTemperature": "0.075 eV",
    }
    observed = {key: directive(text, key) for key in expected}
    failures = [key for key, value in expected.items() if observed[key] != value]
    kgrid = block(text, "kgrid_Monkhorst_Pack")
    if [row[:3] for row in kgrid] != [["3", "0", "0"], ["0", "3", "0"], ["0", "0", "1"]]:
        failures.append("kgrid_Monkhorst_Pack")
    if len(block(text, "PAO.Basis")) != 7:
        failures.append("PAO.Basis")
    required = [CANONICAL, SOURCE_SAMPLE / "C.psf", SIESTA, GRAPH2MAT]
    failures += [str(path) for path in required if not path.exists()]
    nve_text, nvt_text = md_fdf("nve", 300, 3, 1.0, True), md_fdf("nvt", 300, 3, 1.0, False)
    allowed_changes = {key.lower() for key in (
        "DM.UseSaveDM", "MD.UseSaveXV", "MD.TypeOfRun", "MD.Steps", "MD.InitialTimeStep",
        "MD.FinalTimeStep", "MD.LengthTimeStep", "MD.InitialTemperature", "WriteMDHistory",
        "MD.TargetTemperature", "MD.NoseMass", "SaveHS", "Save.HS", "TS.HS.Save", "Lua.Script",
    )}
    before = directive_map(text)
    generated = (directive_map(nve_text), directive_map(nvt_text))
    changed = sorted({key for after in generated for key in before.keys() | after.keys()
                      if before.get(key) != after.get(key)})
    unexpected = sorted(set(changed) - allowed_changes)
    for name in ("PAO.Basis", "LatticeVectors", "AtomicCoordinatesAndAtomicSpecies", "kgrid_Monkhorst_Pack"):
        if any(block(text, name) != block(generated_text, name) for generated_text in (nve_text, nvt_text)):
            unexpected.append(name)
    failures += unexpected
    report = {
        "canonical_fdf": str(CANONICAL), "canonical_sha256": sha256(CANONICAL),
        "observed": observed, "kgrid": kgrid, "pao_basis": block(text, "PAO.Basis"),
        "siesta": str(SIESTA), "siesta_sha256": sha256(SIESTA) if SIESTA.exists() else None,
        "pseudopotential_sha256": sha256(SOURCE_SAMPLE / "C.psf") if (SOURCE_SAMPLE / "C.psf").exists() else None,
        "md_fdf_changed_directives": changed, "md_fdf_unexpected_changes": unexpected,
        "valid": not failures, "failures": failures,
    }
    c.write_json(ROOT / "preflight.json", report)
    if failures:
        raise RuntimeError(f"preflight failed: {failures}")
    log("preflight passed", canonical_sha256=report["canonical_sha256"])
    return report


def self_test() -> None:
    lattice, positions = canonical_geometry()
    v1 = maxwell_velocities(300, 1234)
    v2 = maxwell_velocities(300, 1234)
    v3 = maxwell_velocities(300, 1235)
    assert np.array_equal(v1, v2)
    assert not np.array_equal(v1, v3)
    assert np.linalg.norm(v1.mean(axis=0)) < 1e-15
    assert abs(kinetic_temperature(v1) / 300 - 1) < 1e-12
    target = ROOT / "self_test.XV"
    write_xv(target, lattice, positions, v1)
    parsed_lattice, parsed_atoms = parse_xv_geometry(target)
    assert np.allclose(parsed_lattice, lattice, atol=1e-9)
    assert np.allclose([atom[2:] for atom in parsed_atoms], positions, atol=1e-9)
    assert np.allclose(read_xv_velocities(target), v1, atol=1e-12)
    target.unlink()
    hashes = {hashlib.sha256(maxwell_velocities(t, SEEDS[(s, t)]).tobytes()).hexdigest()
              for s in SPLITS for t in TEMPERATURES}
    assert len(hashes) == 9
    log("velocity/XV self-test passed", unique_velocity_hashes=len(hashes))


def package_temperature() -> float | None:
    try:
        output = subprocess.run(["sensors"], capture_output=True, text=True, timeout=5).stdout
        match = re.search(r"Package id 0:\s*\+?([\d.]+)°C", output)
        return float(match.group(1)) if match else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def available_ram_gb() -> float:
    match = re.search(r"MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text())
    return int(match.group(1)) / 1024**2 if match else 0.0


def _set_paused(item: dict[str, Any], paused: bool, reason: str, temperature: float | None) -> None:
    if item["paused"] == paused or item["process"].poll() is not None:
        return
    os.killpg(item["process"].pid, signal.SIGSTOP if paused else signal.SIGCONT)
    item["paused"] = paused
    log("process paused" if paused else "process resumed", pid=item["process"].pid,
        job=item["name"], reason=reason, package_c=temperature)


def reconcile_temperature(*, pause_c: float = 72, limit_c: float = 75,
                          resume_c: float = 63) -> float | None:
    global _COOL_SINCE, _LAST_RESUME, _INCIDENT_ACTIVE
    if not resume_c < pause_c < limit_c:
        raise ValueError("thermal thresholds must satisfy resume_c < pause_c < limit_c")
    temperature = package_temperature()
    if temperature is None:
        return None
    now = time.monotonic()
    with _ACTIVE_LOCK:
        active = list(_ACTIVE.values())
    if temperature >= limit_c:
        if not _INCIDENT_ACTIVE:
            log("thermal incident", package_c=temperature, limit_c=limit_c)
        _INCIDENT_ACTIVE, _COOL_SINCE = True, None
        for item in active:
            _set_paused(item, True, f"Package id 0 >= {limit_c:g} C", temperature)
    elif temperature >= pause_c:
        _COOL_SINCE = None
        for item in active:
            _set_paused(item, True, f"Package id 0 >= {pause_c:g} C", temperature)
    elif temperature < resume_c and any(item["paused"] for item in active):
        _COOL_SINCE = _COOL_SINCE or now
        if now - _COOL_SINCE >= 120 and now - _LAST_RESUME >= 10:
            item = next(item for item in active if item["paused"])
            _set_paused(item, False, "Package id 0 cool for two minutes", temperature)
            _LAST_RESUME = now
            if not any(candidate["paused"] for candidate in active):
                _INCIDENT_ACTIVE, _COOL_SINCE = False, None
    elif temperature >= resume_c:
        _COOL_SINCE = None
    return temperature


def prepare_run(run_dir: Path, fdf: str, initial_xv: Path, interval: int = 1) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "RUN.fdf").write_text(fdf, encoding="utf-8")
    shutil.copy2(SOURCE_SAMPLE / "C.psf", run_dir / "C.psf")
    shutil.copy2(initial_xv, run_dir / f"{LABEL}.XV")
    dm = initial_xv.parent / f"{LABEL}.DM"
    if dm.exists():
        shutil.copy2(dm, run_dir / dm.name)
    if directive(fdf, "Lua.Script"):
        subprocess.run(
            [str(GRAPH2MAT), "siesta", "md", "setup-store", "--interval", str(interval),
             "--files", STORE_FILES], cwd=run_dir, check=True, env=ENV,
             stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )


def run_siesta(run_dir: Path, ranks: int, name: str) -> dict[str, Any]:
    marker = run_dir / "completed.json"
    if marker.exists():
        saved = c.read_json(marker)
        status = parse_siesta_output(run_dir / "RUN.out", run_dir / "RUN.fdf")
        if status["valid"] and status["job_completed"] and (run_dir / f"{LABEL}.XV").exists():
            return saved
        marker.unlink()
    command = (["mpirun", "-np", str(ranks), str(SIESTA)] if ranks > 1 else [str(SIESTA)])
    start_wall = time.time()
    start = time.monotonic()
    with (run_dir / "RUN.fdf").open("rb") as stdin, (run_dir / "RUN.out").open("wb") as stdout:
        process = subprocess.Popen(command, cwd=run_dir, stdin=stdin, stdout=stdout,
                                   stderr=subprocess.STDOUT, env=ENV, start_new_session=True)
        item = {"process": process, "name": name, "started": time.monotonic(), "paused": False}
        with _ACTIVE_LOCK:
            _ACTIVE[process.pid] = item
        peak_temperature = package_temperature()
        try:
            while process.poll() is None:
                temperature = reconcile_temperature()
                if temperature is not None:
                    peak_temperature = max(peak_temperature or temperature, temperature)
                time.sleep(10)
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE.pop(process.pid, None)
    seconds = time.monotonic() - start
    status = parse_siesta_output(run_dir / "RUN.out", run_dir / "RUN.fdf")
    output_text = (run_dir / "RUN.out").read_text(encoding="utf-8", errors="ignore")
    if process.returncode or not status["valid"] or not status["job_completed"] or re.search(r"\bnan\b", output_text, re.I):
        raise RuntimeError(f"SIESTA failed for {name}: rc={process.returncode}, {status['parser_status']}")
    result = {
        "name": name, "ranks": ranks, "wall_s": seconds,
        "siesta_cpu_s": c.siesta_seconds(run_dir / "RUN.out"),
        "cpu_s_estimate": ((c.siesta_seconds(run_dir / "RUN.out") or seconds) if ranks == 1 else seconds * ranks),
        "started_at_epoch_s": start_wall, "peak_package_c": peak_temperature,
        "run_out_sha256": sha256(run_dir / "RUN.out"), "final_xv_sha256": sha256(run_dir / f"{LABEL}.XV"),
    }
    c.write_json(marker, result)
    log("SIESTA completed", **result)
    return result


def parse_mde(path: Path) -> list[dict[str, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 6 and parts[0].isdigit():
            values = [float(value) for value in parts]
            rows.append(dict(zip(("step", "temperature_K", "E_KS_eV", "E_total_eV", "volume_A3", "pressure_kbar"), values)))
    if not rows:
        raise RuntimeError(f"no MD energy rows in {path}")
    return rows


def microbenchmark() -> dict[str, Any]:
    path = ROOT / "microbenchmark.json"
    if path.exists():
        return c.read_json(path)
    lattice, positions = canonical_geometry()
    base = WORK / "microbenchmark/initial"
    base.mkdir(parents=True, exist_ok=True)
    initial = base / f"{LABEL}.XV"
    write_xv(initial, lattice, positions, maxwell_velocities(300, 2026092299))
    results = []
    for ranks in (1, 2, 3):
        run_dir = WORK / f"microbenchmark/ranks_{ranks}"
        prepare_run(run_dir, md_fdf("nvt", 300, 3, 1.0, False), initial)
        result = run_siesta(run_dir, ranks, f"microbenchmark_r{ranks}")
        result["seconds_per_step"] = result["wall_s"] / 3
        result["cpu_s_per_step"] = result["cpu_s_estimate"] / 3
        results.append(result)
    one = results[0]
    eligible = [row for row in results if row["seconds_per_step"] < one["seconds_per_step"]
                and row["cpu_s_per_step"] <= 1.10 * one["cpu_s_per_step"]
                and (row["peak_package_c"] or 999) < 70]
    selected = min(eligible, key=lambda row: row["seconds_per_step"]) if eligible else one
    report = {"results": results, "selected_ranks": selected["ranks"],
              "rule": "faster, CPU-s/step <= 1.10x rank1, peak Package <70 C; otherwise rank1"}
    c.write_json(path, report)
    log("microbenchmark complete", selected_ranks=selected["ranks"], results=results)
    return report


def initial_xv(split: str, temperature: int) -> Path:
    path = WORK / "trajectories" / f"{split}_T{temperature}" / "initial" / f"{LABEL}.XV"
    if not path.exists():
        lattice, positions = canonical_geometry()
        write_xv(path, lattice, positions, maxwell_velocities(temperature, SEEDS[(split, temperature)]))
        c.write_json(path.parent / "velocity.json", {
            "seed": SEEDS[(split, temperature)], "target_temperature_K": temperature,
            "kinetic_temperature_K": kinetic_temperature(read_xv_velocities(path)),
            "velocity_sha256": hashlib.sha256(read_xv_velocities(path).tobytes()).hexdigest(),
        })
    return path


def equilibrated(rows: list[dict[str, float]], target: int) -> tuple[bool, dict[str, float]]:
    tail = rows[-25:]
    temperatures = np.asarray([row["temperature_K"] for row in tail])
    x = np.arange(len(temperatures), dtype=float)
    slope = float(np.polyfit(x, temperatures, 1)[0]) if len(temperatures) > 1 else float("inf")
    half_difference = float(abs(temperatures[:len(tail) // 2].mean() - temperatures[len(tail) // 2:].mean()))
    diagnostics = {
        "tail_mean_temperature_K": float(temperatures.mean()),
        "tail_temperature_slope_K_per_fs": slope,
        "tail_half_difference_K": half_difference,
    }
    valid = (
        len(tail) == 25
        and np.isfinite(temperatures).all()
        and abs(temperatures.mean() / target - 1) <= 0.20
        and abs(slope) * 25 <= 0.20 * target
        and half_difference <= 0.20 * target
    )
    return bool(valid), diagnostics


def minimum_periodic_distance(lattice: np.ndarray, positions: np.ndarray) -> float:
    inverse = np.linalg.inv(lattice)
    minimum = float("inf")
    for index in range(len(positions)):
        fractional = (positions[index + 1:] - positions[index]) @ inverse
        fractional -= np.round(fractional)
        distances = np.linalg.norm(fractional @ lattice, axis=1)
        if len(distances):
            minimum = min(minimum, float(distances.min()))
    return minimum


def nearest_neighbour_pairs() -> list[tuple[int, int]]:
    lattice, positions = canonical_geometry()
    pairs = []
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            fractional = (positions[right] - positions[left]) @ np.linalg.inv(lattice)
            fractional -= np.round(fractional)
            distance = np.linalg.norm(fractional @ lattice)
            if distance < 1.8:
                pairs.append((left, right))
    if len(pairs) != 108:
        raise RuntimeError(f"expected 108 graphene nearest-neighbour bonds, found {len(pairs)}")
    return pairs


def bond_observable(xv: Path, pairs: list[tuple[int, int]]) -> tuple[float, float, str]:
    lattice_rows, atoms = parse_xv_geometry(xv)
    lattice = np.asarray(lattice_rows)
    positions = np.asarray([atom[2:] for atom in atoms])
    inverse = np.linalg.inv(lattice)
    distances = []
    for left, right in pairs:
        fractional = (positions[right] - positions[left]) @ inverse
        fractional -= np.round(fractional)
        distances.append(float(np.linalg.norm(fractional @ lattice)))
    return float(np.mean(distances)), minimum_periodic_distance(lattice, positions), c.s5.geometry_hash(positions)


def statistical_inefficiency(values: list[float]) -> tuple[float, int, list[float]]:
    data = np.asarray(values, dtype=float)
    centered = data - data.mean()
    variance = float(np.dot(centered, centered) / len(centered))
    if len(data) < 3 or not np.isfinite(variance) or variance <= 0:
        raise RuntimeError("autocorrelation is not calculable")
    correlations = [1.0]
    cutoff = 0
    for lag in range(1, len(data)):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / ((len(data) - lag) * variance))
        correlations.append(rho)
        if rho <= 0:
            cutoff = lag
            break
    if not cutoff:
        raise RuntimeError("autocorrelation did not reach its first non-positive value")
    g = 1.0 + 2.0 * sum(correlations[1:cutoff])
    return max(1.0, float(g)), cutoff, correlations


def nve_diagnostics(run_dirs: list[Path], timestep_fs: float) -> dict[str, Any]:
    energies, temperatures, times = [], [], []
    offset = 0
    for run_dir in run_dirs:
        rows = parse_mde(run_dir / f"{LABEL}.MDE")
        energies += [row["E_total_eV"] for row in rows]
        temperatures += [row["temperature_K"] for row in rows]
        times += [(offset + index) * timestep_fs for index in range(len(rows))]
        offset += len(rows)
    slope_eV_fs = float(np.polyfit(times, energies, 1)[0])
    drift = abs(slope_eV_fs) * 1000 / 72 * 1000
    return {
        "n_steps": len(energies), "mean_temperature_K": float(np.mean(temperatures)),
        "energy_drift_meV_atom_ps": drift,
        "stable": bool(np.isfinite(energies).all() and drift <= 1.0),
    }


def stored_frames(run_dirs: list[Path], timestep_fs: float) -> list[dict[str, Any]]:
    pairs = nearest_neighbour_pairs()
    rows, seen, offset = [], set(), 0
    for run_dir in run_dirs:
        steps_dir = run_dir / "MD_steps"
        frame_dirs = sorted((path for path in steps_dir.iterdir() if path.is_dir() and path.name.isdigit()),
                            key=lambda path: int(path.name))
        local_steps = len(parse_mde(run_dir / f"{LABEL}.MDE"))
        for frame_dir in frame_dirs:
            xv = frame_dir / f"{LABEL}.XV"
            mean_bond, min_distance, geometry_hash = bond_observable(xv, pairs)
            if geometry_hash in seen:
                continue
            seen.add(geometry_hash)
            local_step = int(frame_dir.name)
            rows.append({
                "frame_dir": str(frame_dir), "global_step": offset + local_step,
                "time_fs": (offset + local_step) * timestep_fs,
                "mean_bond_A": mean_bond, "minimum_distance_A": min_distance,
                "geometry_hash": geometry_hash,
            })
        offset += local_steps
    return rows


def run_pilot(temperature: int, ranks: int) -> dict[str, Any]:
    trajectory_id = f"train_T{temperature}"
    root = WORK / "trajectories" / trajectory_id
    summary_path = root / "pilot.json"
    if summary_path.exists():
        summary = c.read_json(summary_path)
        if summary.get("valid"):
            return summary
    source_xv = initial_xv("train", temperature)
    nvt_rows: list[dict[str, float]] = []
    nvt_runs: list[Path] = []
    total = 0
    equilibrium: dict[str, float] = {}
    while total < 300:
        steps = 50 if total == 0 else 25
        run_dir = root / "nvt" / f"segment_{total:03d}_{total + steps:03d}"
        prepare_run(run_dir, md_fdf("nvt", temperature, steps, 1.0, False), source_xv)
        run_siesta(run_dir, ranks, f"{trajectory_id}_nvt_{total}_{total + steps}")
        rows = parse_mde(run_dir / f"{LABEL}.MDE")
        if total == 0 and abs(rows[0]["temperature_K"] / temperature - 1) >= 0.01:
            raise RuntimeError(f"{trajectory_id}: initial SIESTA temperature differs by >=1%")
        nvt_rows += rows
        nvt_runs.append(run_dir)
        total += steps
        source_xv = run_dir / f"{LABEL}.XV"
        is_equilibrated, equilibrium = equilibrated(nvt_rows, temperature)
        log("NVT segment assessed", trajectory=trajectory_id, total_fs=total,
            equilibrated=is_equilibrated, **equilibrium)
        if is_equilibrated:
            break
    if not is_equilibrated:
        raise RuntimeError(f"{trajectory_id}: NVT did not equilibrate within 300 fs")

    chosen_dt, nve_runs, nve = None, [], {}
    for timestep_fs in (1.0, 0.5):
        candidate_runs, candidate_source = [], source_xv
        total_steps = 0
        while total_steps < round(600 / timestep_fs):
            physical_fs = 120 if total_steps == 0 else 120
            steps = round(physical_fs / timestep_fs)
            dt_label = str(timestep_fs).replace(".", "p")
            run_dir = root / f"nve_dt{dt_label}" / f"block_{total_steps:04d}_{total_steps + steps:04d}"
            prepare_run(run_dir, md_fdf("nve", temperature, steps, timestep_fs, True), candidate_source)
            run_siesta(run_dir, ranks, f"{trajectory_id}_nve_dt{dt_label}_{total_steps}_{total_steps + steps}")
            candidate_runs.append(run_dir)
            candidate_source = run_dir / f"{LABEL}.XV"
            total_steps += steps
            nve = nve_diagnostics(candidate_runs, timestep_fs)
            frames = stored_frames(candidate_runs, timestep_fs)
            if min(row["minimum_distance_A"] for row in frames) <= 1.0:
                raise RuntimeError(f"{trajectory_id}: non-physical geometry")
            try:
                g, cutoff, correlations = statistical_inefficiency([row["mean_bond_A"] for row in frames])
                nve |= {"g": g, "acf_cutoff": cutoff, "acf": correlations, "n_unique_frames": len(frames)}
                acf_ok = True
            except RuntimeError:
                acf_ok = False
            log("NVE pilot assessed", trajectory=trajectory_id, timestep_fs=timestep_fs,
                total_steps=total_steps, acf_ok=acf_ok, **{key: value for key, value in nve.items() if key != "acf"})
            if nve["stable"] and acf_ok and total_steps * timestep_fs >= 120:
                chosen_dt, nve_runs = timestep_fs, candidate_runs
                break
            if not nve["stable"]:
                break
        if chosen_dt is not None:
            break
    if chosen_dt is None:
        raise RuntimeError(f"{trajectory_id}: no stable NVE protocol at 1 or 0.5 fs")
    summary = {
        "trajectory_id": trajectory_id, "split": "train", "temperature_K": temperature,
        "velocity_seed": SEEDS[("train", temperature)],
        "velocity_sha256": c.read_json(initial_xv("train", temperature).parent / "velocity.json")["velocity_sha256"],
        "nvt_steps": total, "nvt_runs": [str(path) for path in nvt_runs], "equilibrium": equilibrium,
        "timestep_fs": chosen_dt, "nve_runs": [str(path) for path in nve_runs], "nve": nve,
        "final_xv": str(nve_runs[-1] / f"{LABEL}.XV"), "valid": True,
    }
    c.write_json(summary_path, summary)
    return summary


def run_dynamic(tasks: list[tuple[str, Any]], worker: Any, provisional_max: int = 6) -> dict[str, Any]:
    """Start two tasks; add one every five cool minutes, never above the fixed cap."""

    # The campaign's sustained three-trajectory pilot reached 75 C; never retry that envelope.
    if (ROOT / "run.log").exists() and '"message": "thermal incident"' in (ROOT / "run.log").read_text():
        provisional_max = min(provisional_max, 2)
    pending = list(tasks)
    futures: dict[concurrent.futures.Future[Any], str] = {}
    results: dict[str, Any] = {}
    started = 0
    last_add = time.monotonic()
    cool_since = time.monotonic() if (package_temperature() or 999) < 65 else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=provisional_max) as pool:
        while pending or futures:
            desired = min(2, provisional_max)
            while pending and len(futures) < desired:
                name, payload = pending.pop(0)
                futures[pool.submit(worker, payload)] = name
                started += 1
            done = [future for future in futures if future.done()]
            for future in done:
                name = futures.pop(future)
                results[name] = future.result()
            temperature = reconcile_temperature()
            with _ACTIVE_LOCK:
                has_paused_process = any(item["paused"] for item in _ACTIVE.values())
            safe = (temperature is not None and temperature < 65 and not has_paused_process and os.getloadavg()[0] < 20
                    and available_ram_gb() > 8)
            cool_since = (cool_since or time.monotonic()) if safe else None
            if (pending and len(futures) < provisional_max and cool_since is not None
                    and time.monotonic() - cool_since >= 300 and time.monotonic() - last_add >= 300):
                name, payload = pending.pop(0)
                futures[pool.submit(worker, payload)] = name
                started += 1
                last_add = time.monotonic()
                cool_since = time.monotonic()
                log("dynamic parallelism increased", running=len(futures), started=started,
                    package_c=temperature, load=os.getloadavg()[0], available_ram_gb=available_ram_gb())
            if not done:
                time.sleep(10)
    return results


def pilots() -> dict[str, Any]:
    path = ROOT / "pilot_summary.json"
    saved = c.read_json(path) if path.exists() else {}
    if saved.get("stride") and len(saved.get("temperatures", {})) == 3:
        return saved
    existing = saved.get("temperatures", saved)
    ranks = int(microbenchmark()["selected_ranks"])
    pending = [(f"train_T{temperature}", temperature) for temperature in TEMPERATURES
               if not existing.get(str(temperature), {}).get("valid")]
    if pending:
        completed = run_dynamic(pending, lambda temperature: run_pilot(temperature, ranks), provisional_max=3)
        existing |= {str(result["temperature_K"]): result for result in completed.values()}
    if len(existing) != 3:
        raise RuntimeError(f"expected three valid pilots, found {list(existing)}")
    stride = math.ceil(2 * max(float(row["nve"]["g"]) for row in existing.values()))
    output = {"temperatures": existing, "stride": stride,
              "stride_rule": "ceil(2 * max(g150, g300, g450)); g stops before first non-positive rho"}
    c.write_json(path, output)
    log("pilots complete", stride=stride, g={key: row["nve"]["g"] for key, row in existing.items()})
    return output


def md_history_positions(path: Path, n_atoms: int = 72) -> list[np.ndarray]:
    """Read SIESTA's simple Fortran-unformatted .MD records; positions are Bohr."""

    positions = []
    with path.open("rb") as handle:
        while marker := handle.read(4):
            if len(marker) != 4:
                raise RuntimeError(f"truncated Fortran record marker in {path}")
            size = struct.unpack("<i", marker)[0]
            payload = handle.read(size)
            end = handle.read(4)
            if len(payload) != size or len(end) != 4 or struct.unpack("<i", end)[0] != size:
                raise RuntimeError(f"invalid Fortran record in {path}")
            if size != 4 + 2 * n_atoms * 3 * 8:
                raise RuntimeError(f"unexpected .MD record size {size} in {path}")
            values = np.frombuffer(payload, dtype="<f8", offset=4)
            positions.append(values[:3 * n_atoms].reshape(n_atoms, 3) * BOHR_TO_ANG)
    if not positions:
        raise RuntimeError(f"empty MD history: {path}")
    return positions


def trajectory_acf(run_dirs: list[Path]) -> dict[str, Any]:
    lattice, _ = canonical_geometry()
    inverse, pairs = np.linalg.inv(lattice), nearest_neighbour_pairs()
    values = []
    for run_dir in run_dirs:
        for positions in md_history_positions(run_dir / f"{LABEL}.MD"):
            distances = []
            for left, right in pairs:
                fractional = (positions[right] - positions[left]) @ inverse
                fractional -= np.round(fractional)
                distances.append(np.linalg.norm(fractional @ lattice))
            values.append(float(np.mean(distances)))
    g, cutoff, correlations = statistical_inefficiency(values)
    return {"g": g, "acf_cutoff": cutoff, "acf": correlations, "n_observations": len(values)}


def run_equilibration(split: str, temperature: int, ranks: int) -> tuple[list[Path], Path, dict[str, Any]]:
    trajectory_id = f"{split}_T{temperature}"
    root = WORK / "trajectories" / trajectory_id
    source_xv = initial_xv(split, temperature)
    rows: list[dict[str, float]] = []
    runs: list[Path] = []
    total = 0
    diagnostics: dict[str, Any] = {}
    while total < 300:
        steps = 50 if total == 0 else 25
        run_dir = root / "nvt" / f"segment_{total:03d}_{total + steps:03d}"
        prepare_run(run_dir, md_fdf("nvt", temperature, steps, 1.0, False), source_xv)
        run_siesta(run_dir, ranks, f"{trajectory_id}_nvt_{total}_{total + steps}")
        segment = parse_mde(run_dir / f"{LABEL}.MDE")
        if total == 0 and abs(segment[0]["temperature_K"] / temperature - 1) >= 0.01:
            raise RuntimeError(f"{trajectory_id}: initial SIESTA temperature differs by >=1%")
        rows += segment
        runs.append(run_dir)
        total += steps
        source_xv = run_dir / f"{LABEL}.XV"
        passed, diagnostics = equilibrated(rows, temperature)
        log("NVT segment assessed", trajectory=trajectory_id, total_fs=total,
            equilibrated=passed, **diagnostics)
        if passed:
            return runs, source_xv, {"nvt_steps": total, **diagnostics}
    raise RuntimeError(f"{trajectory_id}: NVT did not equilibrate within 300 fs")


def decorrelated_frames(run_dirs: list[Path], stride: int, timestep_fs: float) -> list[dict[str, Any]]:
    selected = []
    minimum_gap = stride * timestep_fs - 1e-9
    for row in stored_frames(run_dirs, timestep_fs):
        if not selected or row["time_fs"] - selected[-1]["time_fs"] >= minimum_gap:
            selected.append(row)
    return selected


def extend_train_pilot(temperature: int, pilot: dict[str, Any], stride: int,
                       target_candidates: int, ranks: int) -> dict[str, Any]:
    root = WORK / "trajectories" / f"train_T{temperature}"
    timestep_fs = float(pilot["timestep_fs"])
    run_dirs = [Path(path) for path in pilot["nve_runs"]]
    source_xv = Path(pilot["final_xv"])
    frames = decorrelated_frames(run_dirs, stride, timestep_fs)
    total_steps = sum(len(parse_mde(path / f"{LABEL}.MDE")) for path in run_dirs)
    while len(frames) < target_candidates:
        missing = target_candidates - len(frames)
        steps = max(round(120 / timestep_fs), missing * stride + 1)
        run_dir = root / f"nve_dt{str(timestep_fs).replace('.', 'p')}" / f"block_{total_steps:04d}_{total_steps + steps:04d}"
        prepare_run(run_dir, md_fdf("nve", temperature, steps, timestep_fs, True), source_xv, interval=stride)
        run_siesta(run_dir, ranks, f"train_T{temperature}_nve_extension_{total_steps}_{total_steps + steps}")
        run_dirs.append(run_dir)
        source_xv = run_dir / f"{LABEL}.XV"
        total_steps += steps
        frames = decorrelated_frames(run_dirs, stride, timestep_fs)
    diagnostics = nve_diagnostics(run_dirs, timestep_fs) | trajectory_acf(run_dirs)
    if not diagnostics["stable"]:
        raise RuntimeError(f"train_T{temperature}: extended NVE became unstable")
    return pilot | {"nve_runs": [str(path) for path in run_dirs], "final_xv": str(source_xv),
                    "nve": diagnostics, "candidate_frames": frames}


def run_production_trajectory(payload: tuple[str, int, int, int, float]) -> dict[str, Any]:
    split, temperature, stride, ranks, timestep_fs = payload
    trajectory_id = f"{split}_T{temperature}"
    root = WORK / "trajectories" / trajectory_id
    summary_path = root / "production.json"
    if summary_path.exists():
        summary = c.read_json(summary_path)
        if summary.get("valid"):
            return summary
    nvt_runs, source_xv, equilibrium = run_equilibration(split, temperature, ranks)
    retained, target_candidates = 16, math.ceil(16 * 1.25)
    steps = 1 + (target_candidates - 1) * stride
    run_dir = root / "nve" / f"production_{steps:04d}"
    prepare_run(run_dir, md_fdf("nve", temperature, steps, timestep_fs, True), source_xv, interval=stride)
    run_siesta(run_dir, ranks, f"{trajectory_id}_nve_production")
    run_dirs = [run_dir]
    frames = stored_frames(run_dirs, timestep_fs)
    if len(frames) < target_candidates:
        raise RuntimeError(f"{trajectory_id}: only {len(frames)}/{target_candidates} stored candidates")
    diagnostics = nve_diagnostics(run_dirs, timestep_fs) | trajectory_acf(run_dirs)
    if not diagnostics["stable"] or min(row["minimum_distance_A"] for row in frames) <= 1.0:
        raise RuntimeError(f"{trajectory_id}: unstable or non-physical production")
    summary = {
        "trajectory_id": trajectory_id, "split": split, "temperature_K": temperature,
        "velocity_seed": SEEDS[(split, temperature)],
        "velocity_sha256": c.read_json(initial_xv(split, temperature).parent / "velocity.json")["velocity_sha256"],
        "nvt_runs": [str(path) for path in nvt_runs], "equilibrium": equilibrium,
        "timestep_fs": timestep_fs, "nve_runs": [str(path) for path in run_dirs],
        "nve": diagnostics, "candidate_frames": frames, "retained_frames": retained,
        "valid": True,
    }
    c.write_json(summary_path, summary)
    return summary


def production() -> dict[str, Any]:
    path = ROOT / "production_summary.json"
    if path.exists():
        summary = c.read_json(path)
        if summary.get("valid"):
            return summary
    pilot_summary = pilots()
    stride, ranks = int(pilot_summary["stride"]), int(microbenchmark()["selected_ranks"])
    timesteps = {float(row["timestep_fs"]) for row in pilot_summary["temperatures"].values()}
    if len(timesteps) != 1:
        raise RuntimeError(f"pilots selected inconsistent timesteps: {timesteps}")
    timestep_fs = timesteps.pop()
    train_targets = {150: math.ceil(22 * 1.25), 300: math.ceil(21 * 1.25), 450: math.ceil(21 * 1.25)}
    train_tasks = [(f"train_T{temperature}", (
        temperature, pilot_summary["temperatures"][str(temperature)], stride,
        train_targets[temperature], ranks,
    )) for temperature in TEMPERATURES]
    trains = run_dynamic(train_tasks, lambda payload: extend_train_pilot(*payload), provisional_max=2)
    tasks = [(f"{split}_T{temperature}", (split, temperature, stride, ranks, timestep_fs))
             for split in ("validation", "test") for temperature in TEMPERATURES]
    additional = run_dynamic(tasks, run_production_trajectory, provisional_max=6)
    trajectories = {row["trajectory_id"]: row for row in trains.values()}
    trajectories |= additional
    hashes = [row["velocity_sha256"] for row in trajectories.values()]
    if len(trajectories) != 9 or len(set(hashes)) != 9:
        raise RuntimeError("production must contain nine trajectories with unique initial velocities")
    summary = {"valid": True, "stride": stride, "timestep_fs": timestep_fs,
               "trajectories": trajectories}
    c.write_json(path, summary)
    log("MD production complete", trajectories=len(trajectories), stride=stride)
    return summary


def round_robin(by_temperature: dict[int, list[dict[str, Any]]], counts: dict[int, int]) -> list[dict[str, Any]]:
    output = []
    for index in range(max(counts.values())):
        for temperature in TEMPERATURES:
            if index < counts[temperature]:
                output.append(by_temperature[temperature][index])
    return output


def tshs_matches_fdf(frame_dir: Path) -> bool:
    try:
        reference = np.asarray(c.s5._positions_from_run_fdf(frame_dir / "RUN.fdf"))
        geometry = sisl.get_sile(frame_dir / f"{LABEL}.TSHS").read_geometry()
        predicted = np.asarray(geometry.xyz)
        lattice = np.asarray(geometry.cell)
        fractional = (predicted - reference) @ np.linalg.inv(lattice)
        fractional -= np.round(fractional)
        return float(np.abs(fractional @ lattice).max()) < 1e-6
    except Exception:
        return False


def materialize_frame(row: dict[str, Any], split: str, temperature: int,
                      trajectory_id: str, index: int) -> dict[str, Any]:
    frame_dir = Path(row["frame_dir"])
    run_fdf = frame_dir / "RUN.fdf"
    xv = frame_dir / f"{LABEL}.XV"
    rewrite_run_fdf_from_xv(run_fdf, xv)
    if not tshs_matches_fdf(frame_dir):
        raise RuntimeError(f"TSHS/RUN.fdf geometry mismatch in {frame_dir}")
    geometry = c.geometry_record("6x6", run_fdf)
    sample_id = f"md6x6_{split}_T{temperature}_{trajectory_id}_s{int(row['global_step']):05d}"
    metadata = {
        "sample_id": sample_id, "split": split, "temperature_K": temperature,
        "trajectory_id": trajectory_id, "md_step": int(row["global_step"]),
        "time_fs": float(row["time_fs"]), "geometry_hash": geometry["hash"],
        "run_fdf_rewritten_from_xv": True, "run_fdf_geometry_source": f"{LABEL}.XV",
    }
    c.write_json(frame_dir / "metadata.json", metadata)
    return {
        "sample_id": sample_id, "family": "MD", "dim": "3D", "k": 3,
        "amplitude_ang": geometry["real_amplitude_ang"], "real_amplitude_ang": geometry["real_amplitude_ang"],
        "run_fdf": str(run_fdf), "reference_dir": str(frame_dir),
        "reference_matrix": str(frame_dir / f"{LABEL}.TSHS"), "hash": geometry["hash"],
        "siesta_cpu_s": 0.0, **metadata,
    }


def freeze_dataset() -> dict[str, Any]:
    # The requested MD benchmark mirrors the completed Cartesian label-budget
    # study; it deliberately uses every valid snapshot and no ACF thinning.
    return benchmark_freeze_dataset()


def strict_freeze_dataset() -> dict[str, Any]:
    """Original independent-trajectory/ACF protocol, retained for provenance."""

    path = ROOT / "manifest.json"
    if path.exists():
        manifest = c.read_json(path)
        if manifest.get("datasets_frozen"):
            return manifest
    campaign = production()
    allocations = {"train": {150: 22, 300: 21, 450: 21},
                   "validation": {150: 16, 300: 16, 450: 16},
                   "test": {150: 16, 300: 16, 450: 16}}
    datasets: dict[str, list[dict[str, Any]]] = {}
    composition = []
    for split in SPLITS:
        by_temperature = {}
        for temperature in TEMPERATURES:
            trajectory_id = f"{split}_T{temperature}"
            trajectory = campaign["trajectories"][trajectory_id]
            candidates = trajectory["candidate_frames"][:allocations[split][temperature]]
            if len(candidates) != allocations[split][temperature]:
                raise RuntimeError(f"{trajectory_id}: insufficient decorrelated candidates")
            by_temperature[temperature] = [materialize_frame(row, split, temperature, trajectory_id, index)
                                           for index, row in enumerate(candidates)]
            composition.append({"split": split, "temperature_K": temperature,
                                "trajectory_id": trajectory_id, "n_frames": len(candidates)})
        datasets[split] = round_robin(by_temperature, allocations[split])
    all_rows = sum(datasets.values(), [])
    if len(all_rows) != 160 or len({row["hash"] for row in all_rows}) != 160:
        raise RuntimeError("expected 160 unique MD geometries")
    if any(set(row["trajectory_id"] for row in datasets[a]) & set(row["trajectory_id"] for row in datasets[b])
           for index, a in enumerate(SPLITS) for b in SPLITS[index + 1:]):
        raise RuntimeError("trajectory overlap between splits")
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "datasets_frozen": True,
        "canonical_fdf": str(CANONICAL), "canonical_sha256": sha256(CANONICAL),
        "train_sizes": list(TRAIN_SIZES), "validation_sizes": list(VAL_SIZES),
        "test_sizes": list(TEST_SIZES), "stride": campaign["stride"],
        "training": {"batch_size": 16, "optimizer_updates": 8000, "validation_checks": 2000,
                     "loss": "graph2mat.metrics.elementwise_mse", "scheduler": "cosine"},
        "train_prefixes": {str(size): [row["sample_id"] for row in datasets["train"][:size]]
                           for size in TRAIN_SIZES},
        "validation_prefixes": {str(size): [row["sample_id"] for row in datasets["validation"][:size]]
                                for size in VAL_SIZES},
        "test_pool": [row["sample_id"] for row in datasets["test"]], "datasets": datasets,
        "frozen_sha256": {split: c.ids_hash([row["sample_id"] + ":" + row["hash"] for row in rows])
                          for split, rows in datasets.items()},
    }
    c.write_csv(ROOT / "dataset_composition.csv", composition)
    c.write_json(path, manifest)
    log("MD datasets frozen", counts={key: len(value) for key, value in datasets.items()},
        hashes=manifest["frozen_sha256"])
    return manifest


VRAM_MIB = {
    (4, 8): 2458, (4, 24): 4098, (4, 48): 5518,
    (8, 8): 4206, (8, 24): 4100, (8, 48): 5520,
    (16, 8): 7576, (16, 24): 7560, (16, 48): 8502,
    (32, 8): 8454, (32, 24): 7520, (32, 48): 8468,
    (64, 8): 7580, (64, 24): 7578, (64, 48): 8536,
}


def gpu_free_mib() -> int:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.splitlines()[0]
        return int(output.strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def train_process(config: Path, run_dir: Path, name: str) -> tuple[Path, float, int]:
    env = ENV | {"PYTHONPATH": os.pathsep.join(filter(None, [str(c.s4.TORCH_COMPAT_DIR), ENV.get("PYTHONPATH", "")]))}
    peak, start = 0, time.monotonic()
    with (run_dir / "train.log").open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [str(c.s4.GRAPH2MAT_BIN), "models", "mace", "main", "fit", "-c", config.name],
            cwd=run_dir, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
        item = {"process": process, "name": name, "started": time.monotonic(), "paused": False}
        with _ACTIVE_LOCK:
            _ACTIVE[process.pid] = item
        try:
            while process.poll() is None:
                peak = max(peak, c.gpu_mib(process.pid))
                reconcile_temperature()
                time.sleep(5)
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE.pop(process.pid, None)
    seconds = time.monotonic() - start
    if process.returncode:
        raise RuntimeError(f"Graph2Mat failed for {name}; see {run_dir / 'train.log'}")
    checkpoints = sorted(run_dir.rglob("best-*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"{name}: expected one best checkpoint, found {checkpoints}")
    return checkpoints[0], seconds, peak


def selected_rows(manifest: dict[str, Any], source: str, n_train: int,
                  n_val: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if source == "md":
        train = manifest["datasets"]["train"][:n_train]
        validation = manifest["datasets"]["validation"][:n_val]
    else:
        synthetic = c.read_json(c.OUT / "label_budget_6x6/manifest.json")
        train = c.read_json(c.OUT / "coverage_6x6_winner.json")["samples"][:n_train]
        val_all = c.read_json(c.OUT / "common_dev_6x6_v1.json")["samples"]
        by_id = {row["sample_id"]: row for row in val_all}
        validation = [by_id[sample_id] for sample_id in synthetic["validation_sets"][str(n_val)]["sample_ids"]]
    return train, validation


def train_model(source: str, n_train: int, n_val: int, seed: int,
                concurrency: int) -> dict[str, Any]:
    manifest = freeze_dataset()
    job_id = f"{source}__N{n_train}__V{n_val}__seed{seed}"
    run_dir = ROOT / "runs" / source / job_id
    result_path, done_path = run_dir / "result.json", run_dir / "train_done.json"
    if result_path.exists():
        result = c.read_json(result_path)
        checkpoint = Path(result["checkpoint"])
        if checkpoint.exists() and sha256(checkpoint) == result["checkpoint_sha256"]:
            return result
    train_rows, val_rows = selected_rows(manifest, source, n_train, n_val)
    train = [c.sample_from_dict(row) for row in train_rows]
    validation = [c.sample_from_dict(row) for row in val_rows]
    if done_path.exists():
        done = c.read_json(done_path)
        checkpoint = Path(done["checkpoint"])
        if not checkpoint.exists() or sha256(checkpoint) != done["checkpoint_sha256"]:
            raise RuntimeError(f"invalid completed checkpoint marker: {done_path}")
        seconds, peak = float(done["seconds"]), int(done["peak_gpu_mib"])
    else:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        config = c.s4.build_graph2mat_config(
            run_dir, train, validation, run_name=job_id, accelerator="gpu", training_seed=seed, **c.TRAIN_KW,
        )
        c.apply_config_patch(config, c.fixed_update_elemmse_patch("6x6", n_train))
        checkpoint, seconds, peak = train_process(config, run_dir, job_id)
        c.write_json(done_path, {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                                 "seconds": seconds, "peak_gpu_mib": peak})
    epochs = c.tensorboard_epochs(run_dir, checkpoint)
    val_metrics = c.summarize(c.evaluate(checkpoint, validation, run_dir / "val_eval", "gpu"))
    step = re.search(r"step=?(\d+)", checkpoint.name)
    result = {
        "model_id": job_id, "training_dataset": source, "N_train": n_train, "N_val": n_val,
        "training_seed": seed, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_policy": "best_val_loss (save_top_k=1)",
        "best_update": int(step.group(1)) if step else None, **epochs,
        "val_metrics": val_metrics, "train_seconds": seconds, "gpu_h": seconds / 3600,
        "peak_gpu_mib": peak, "concurrent_jobs_at_launch": concurrency,
        "train_ids_sha256": c.ids_hash([row["sample_id"] for row in train_rows]),
        "validation_ids_sha256": c.ids_hash([row["sample_id"] for row in val_rows]),
    }
    c.write_json(result_path, result)
    return result


def gpu_schedule(jobs: list[tuple[str, int, int, int]]) -> list[dict[str, Any]]:
    pending, futures, output = list(jobs), {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        while pending or futures:
            launched = False
            while pending:
                source, n_train, n_val, seed = pending[0]
                estimate = VRAM_MIB[(n_train, n_val)]
                temperature = package_temperature()
                running = len(futures)
                cap = 3 if n_train >= 32 else (6 if n_train <= 4 and all(job[1] <= 4 for job in pending[:1]) else 4)
                if (temperature is None or temperature >= 68 or running >= cap
                        or estimate * 1.15 + 4096 > gpu_free_mib()):
                    break
                pending.pop(0)
                concurrency = running + 1
                future = pool.submit(train_model, source, n_train, n_val, seed, concurrency)
                futures[future] = (source, n_train, n_val, seed)
                launched = True
                log("training launched", source=source, N_train=n_train, N_val=n_val, seed=seed,
                    concurrency=concurrency, estimated_vram_mib=estimate, free_vram_mib=gpu_free_mib())
            done = [future for future in futures if future.done()]
            for future in done:
                job = futures.pop(future)
                result = future.result()
                output.append(result)
                log("training completed", job=job, checkpoint_sha256=result["checkpoint_sha256"],
                    gpu_h=result["gpu_h"], peak_gpu_mib=result["peak_gpu_mib"])
            if not launched and not done:
                if not futures and pending and gpu_free_mib() == 0:
                    raise RuntimeError("GPU unavailable for pending training jobs")
                time.sleep(10)
    return output


def initial_training() -> list[dict[str, Any]]:
    freeze_dataset()
    if _ACTIVE:
        raise RuntimeError("refusing to overlap Graph2Mat training with active MD")
    jobs = [("md", n_train, n_val, 0) for n_train in TRAIN_SIZES for n_val in VAL_SIZES]
    # Small jobs first make early scientific checkpoints and use otherwise stranded VRAM.
    jobs.sort(key=lambda row: (VRAM_MIB[(row[1], row[2])], row[1], row[2], row[3]))
    results = gpu_schedule(jobs)
    paths = sorted((ROOT / "runs").glob("*/*/result.json"))
    all_results = [c.read_json(path) for path in paths]
    c.write_csv(ROOT / "model_results.csv", all_results)
    return results


METRICS = ("H_MAE_meV", "band_rmse_meV", "dos_rel_L1")


def synthetic_models() -> list[dict[str, Any]]:
    models = []
    for row in c.read_csv(c.OUT / "label_budget_6x6/model_results.csv"):
        models.append({
            "model_id": f"synthetic__N{row['N_train']}__V{row['N_val']}__seed0",
            "training_dataset": "synthetic", "N_train": int(row["N_train"]),
            "N_val": int(row["N_val"]), "training_seed": 0,
            "checkpoint": row["checkpoint"], "checkpoint_sha256": row["checkpoint_sha256"],
            "gpu_h": float(row["train_seconds"]) / 3600,
            "synthetic_test_csv": row["final_test_csv"],
        })
    return models


def md_models() -> list[dict[str, Any]]:
    return [c.read_json(path) for path in sorted((ROOT / "runs/md").glob("*/result.json"))]


def frozen_models() -> list[dict[str, Any]]:
    models = md_models() + synthetic_models()
    for model in models:
        checkpoint = Path(model["checkpoint"])
        if not checkpoint.exists() or sha256(checkpoint) != model["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint invalid: {model['model_id']}")
    return models


def freeze_checkpoints() -> dict[str, Any]:
    path = ROOT / "checkpoints_frozen.json"
    if path.exists():
        return c.read_json(path)
    models = frozen_models()
    md_seed0 = [row for row in models if row["training_dataset"] == "md" and row["training_seed"] == 0]
    required = {(n_train, n_val) for n_train in TRAIN_SIZES for n_val in VAL_SIZES}
    if {(row["N_train"], row["N_val"]) for row in md_seed0} != required:
        raise RuntimeError("cannot open MD-Test48 before all 15 MD seed0 checkpoints exist")
    synthetic_seed0 = [row for row in models if row["training_dataset"] == "synthetic"
                       and row["training_seed"] == 0]
    if {(row["N_train"], row["N_val"]) for row in synthetic_seed0} != required:
        raise RuntimeError("the comparable 15-model synthetic Cartesian sweep is incomplete")
    manifest = freeze_dataset()
    frozen = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_hashes": manifest["frozen_sha256"],
        "models": [{key: row[key] for key in ("model_id", "training_dataset", "N_train", "N_val",
                                                "training_seed", "checkpoint", "checkpoint_sha256")}
                   for row in models],
    }
    c.write_json(path, frozen)
    log("checkpoints frozen; blind MD-Test48 may now be opened", n_models=len(models))
    return frozen


def test_rows(source: str) -> list[dict[str, Any]]:
    if source == "md":
        return freeze_dataset()["datasets"]["test"]
    return c.read_json(c.OUT / "final_test_manifest.json")["systems"]["6x6"]["samples"]


def evaluate_model(model: dict[str, Any], evaluation_dataset: str) -> Path:
    out_dir = ROOT / "evaluations" / model["model_id"] / f"on_{evaluation_dataset}"
    path = out_dir / "per_structure_metrics.csv"
    if path.exists() and len(c.read_csv(path)) == 48:
        return path
    rows = test_rows(evaluation_dataset)
    samples = [c.sample_from_dict(row) for row in rows]
    predicted_root = c.predict(Path(model["checkpoint"]), samples, out_dir, "gpu")
    metadata = {row["sample_id"]: row for row in rows}
    metrics = []
    for sample in samples:
        row = c.structure_metrics(sample, predicted_root)
        if row is None:
            raise RuntimeError(f"missing prediction for {model['model_id']} / {sample.sample_id}")
        row |= c.band_dos_metrics(sample, predicted_root / sample.sample_id / "ML_prediction.HSX", "6x6")
        meta = metadata[sample.sample_id]
        row |= {
            "model_id": model["model_id"], "training_dataset": model["training_dataset"],
            "evaluation_dataset": evaluation_dataset, "N_train": model["N_train"],
            "N_val": model["N_val"], "training_seed": model["training_seed"],
            "temperature_K": meta.get("temperature_K", ""),
            "trajectory_id": meta.get("trajectory_id", ""), "md_step": meta.get("md_step", ""),
        }
        metrics.append(row)
    c.write_csv(path, metrics)
    shutil.rmtree(predicted_root, ignore_errors=True)
    log("model evaluated", model=model["model_id"], evaluation_dataset=evaluation_dataset)
    return path


def evaluate_initial() -> list[dict[str, Any]]:
    freeze_checkpoints()
    models = frozen_models()
    for model in models:
        evaluate_model(model, "md")
        if model["training_dataset"] == "md":
            evaluate_model(model, "synthetic")
        elif not Path(model.get("synthetic_test_csv", "")).exists():
            model["synthetic_test_csv"] = str(evaluate_model(model, "synthetic"))
    all_rows = []
    for model in models:
        all_rows += c.read_csv(ROOT / "evaluations" / model["model_id"] / "on_md/per_structure_metrics.csv")
        if model["training_dataset"] == "md":
            all_rows += c.read_csv(ROOT / "evaluations" / model["model_id"] / "on_synthetic/per_structure_metrics.csv")
    c.write_csv(ROOT / "per_structure_results.csv", all_rows)
    return all_rows


def md_test_orders(rows: list[dict[str, Any]], repeats: int = 10_000,
                   seed: int = 20260926) -> np.ndarray:
    rng = np.random.default_rng(seed)
    by_temperature = {temperature: np.asarray(sorted(row["sample_id"] for row in rows
                                                     if int(row["temperature_K"]) == temperature), dtype=object)
                      for temperature in TEMPERATURES}
    if any(len(ids) != 16 for ids in by_temperature.values()):
        raise RuntimeError("MD-Test48 must contain 16 structures per temperature")
    orders = np.empty((repeats, 48), dtype=object)
    for repeat in range(repeats):
        shuffled = {temperature: rng.permutation(ids) for temperature, ids in by_temperature.items()}
        rotation = int(rng.integers(0, 3))
        temperature_order = TEMPERATURES[rotation:] + TEMPERATURES[:rotation]
        orders[repeat] = [shuffled[temperature][index] for index in range(16)
                          for temperature in temperature_order]
    return orders


def seed0_analysis() -> dict[str, Any]:
    evaluate_initial()
    models = [row for row in md_models() if row["training_seed"] == 0]
    test = freeze_dataset()["datasets"]["test"]
    orders = md_test_orders(test)
    summaries, stats, draws = [], [], {}
    for model in models:
        rows = c.read_csv(ROOT / "evaluations" / model["model_id"] / "on_md/per_structure_metrics.csv")
        summary = model | {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in METRICS}
        summaries.append(summary)
        for metric in METRICS:
            values = {row["sample_id"]: float(row[metric]) for row in rows}
            full = summary[metric]
            for n_test in TEST_SIZES:
                sampled = np.asarray([[values[sample_id] for sample_id in order[:n_test]]
                                      for order in orders], dtype=float).mean(axis=1)
                delta = (sampled - full) / full
                draws[(model["model_id"], n_test, metric)] = sampled
                stats.append({
                    "model_id": model["model_id"], "N_train": model["N_train"], "N_val": model["N_val"],
                    "N_test": n_test, "metric": metric, "full_test_value": full,
                    "delta_median": float(np.median(delta)), "delta_p05": float(np.quantile(delta, .05)),
                    "delta_p95": float(np.quantile(delta, .95)),
                    "q90_abs_delta": float(np.quantile(np.abs(delta), .90)),
                    "P_abs_delta_le_10pct": float(np.mean(np.abs(delta) <= .10)),
                    "interval_type": "empirical subsampling interval",
                })
    reference = next(row for row in summaries if row["N_train"] == 64 and row["N_val"] == 48)
    ranking = []
    for left_index, left in enumerate(summaries):
        for right in summaries[left_index + 1:]:
            full_difference = left["H_MAE_meV"] - right["H_MAE_meV"]
            relevant = abs(full_difference) >= .05 * reference["H_MAE_meV"]
            for n_test in TEST_SIZES:
                sampled = draws[(left["model_id"], n_test, "H_MAE_meV")] - draws[(right["model_id"], n_test, "H_MAE_meV")]
                preserved = float(np.mean(np.sign(sampled) == np.sign(full_difference))) if full_difference else 1.0
                ranking.append({"model_A": left["model_id"], "model_B": right["model_id"],
                                "N_test": n_test, "difference_relevant": relevant,
                                "rank_preservation_probability": preserved,
                                "rank_inversion_frequency": 1 - preserved})
    indexed_stats = {(row["model_id"], row["N_test"], row["metric"]): row for row in stats}
    cartesian = []
    for model in summaries:
        quality = {
            "H_guard": model["H_MAE_meV"] <= 1.05 * reference["H_MAE_meV"],
            "band_guard": model["band_rmse_meV"] <= 1.10 * reference["band_rmse_meV"],
            "DOS_guard": model["dos_rel_L1"] <= 1.10 * reference["dos_rel_L1"],
        }
        for n_test in TEST_SIZES:
            h_stats = indexed_stats[(model["model_id"], n_test, "H_MAE_meV")]
            relevant_pairs = [row for row in ranking if row["N_test"] == n_test and row["difference_relevant"]
                              and model["model_id"] in (row["model_A"], row["model_B"])]
            worst_preservation = min((row["rank_preservation_probability"] for row in relevant_pairs), default=1.0)
            reliability = h_stats["q90_abs_delta"] <= .10
            ranking_guard = worst_preservation >= .50
            passed = all(quality.values()) and reliability and ranking_guard
            cartesian.append({
                "model_id": model["model_id"], "N_train": model["N_train"], "N_val": model["N_val"],
                "N_test": n_test, "N_total": model["N_train"] + model["N_val"] + n_test,
                **{metric: model[metric] for metric in METRICS},
                "H_ratio_to_reference": model["H_MAE_meV"] / reference["H_MAE_meV"],
                "band_ratio_to_reference": model["band_rmse_meV"] / reference["band_rmse_meV"],
                "DOS_ratio_to_reference": model["dos_rel_L1"] / reference["dos_rel_L1"],
                "H_q90_abs_delta": h_stats["q90_abs_delta"],
                "H_P_abs_delta_le_10pct": h_stats["P_abs_delta_le_10pct"],
                **quality, "test_reliability": reliability, "worst_rank_preservation": worst_preservation,
                "ranking_guard": ranking_guard, "pass": passed,
            })
    passed = [row for row in cartesian if row["pass"]]
    minimum = min(passed, key=lambda row: (row["N_total"], row["H_MAE_meV"], row["N_train"],
                                            row["N_val"], row["N_test"])) if passed else None
    output = {
        "status": "minimum found" if minimum else "minimum not reached", "minimum": minimum,
        "reference": {key: reference[key] for key in ("model_id", "N_train", "N_val", *METRICS)},
        "criteria": {"H_ratio_max": 1.05, "band_ratio_max": 1.10, "DOS_ratio_max": 1.10,
                     "H_q90_abs_delta_max": .10, "rank_preservation_min": .50},
    }
    c.write_csv(ROOT / "test_subsampling.csv", stats)
    c.write_csv(ROOT / "ranking_stability.csv", ranking)
    c.write_csv(ROOT / "cartesian_results.csv", cartesian)
    c.write_json(ROOT / "minimum_md.json", output)
    return output


def conclusion_seed_jobs(minimum_result: dict[str, Any]) -> list[tuple[str, int, int, int]]:
    minimum = minimum_result.get("minimum")
    if not minimum:
        return []
    cartesian = c.read_csv(ROOT / "cartesian_results.csv")
    build_budget = int(minimum["N_train"]) + int(minimum["N_val"])
    cheaper = [row for row in cartesian if int(row["N_train"]) + int(row["N_val"]) < build_budget
               and int(row["N_test"]) == 48]
    neighbour = min(cheaper, key=lambda row: (build_budget - int(row["N_train"]) - int(row["N_val"]),
                                               float(row["H_MAE_meV"]))) if cheaper else None
    pairs = {(4, 8), (64, 48), (int(minimum["N_train"]), int(minimum["N_val"]))}
    if neighbour:
        pairs.add((int(neighbour["N_train"]), int(neighbour["N_val"])))
    existing = {(row["training_dataset"], row["N_train"], row["N_val"], row["training_seed"])
                for row in frozen_models()}
    jobs = []
    for source in ("md", "synthetic"):
        for n_train, n_val in pairs:
            item = (source, n_train, n_val, 1)
            if item not in existing:
                jobs.append(item)
    c.write_json(ROOT / "seed1_selection.json", {"minimum_pair": [int(minimum["N_train"]), int(minimum["N_val"])],
                                                  "pairs": sorted([list(pair) for pair in pairs]),
                                                  "cheaper_neighbour": neighbour, "jobs": jobs})
    return jobs


def confirm_conclusions() -> dict[str, Any]:
    minimum = seed0_analysis()
    jobs = conclusion_seed_jobs(minimum)
    if jobs:
        gpu_schedule(sorted(jobs, key=lambda row: VRAM_MIB[(row[1], row[2])]))
        c.write_json(ROOT / "confirmation_checkpoints.json", {
            "selected_without_retraining_or_hyperparameter_changes": True,
            "models": [{"model_id": row["model_id"], "checkpoint_sha256": row["checkpoint_sha256"]}
                       for row in frozen_models() if row["training_seed"] == 1],
        })
    # Confirmation checkpoints are a prospective repeat of already-fixed cells.
    for model in frozen_models():
        if model["training_seed"] == 1:
            evaluate_model(model, "md")
            evaluate_model(model, "synthetic")
    rows = []
    for model in frozen_models():
        path = ROOT / "evaluations" / model["model_id"] / "on_md/per_structure_metrics.csv"
        if path.exists():
            values = c.read_csv(path)
            rows.append({**{key: model[key] for key in ("model_id", "training_dataset", "N_train", "N_val", "training_seed")},
                         **{metric: float(np.mean([float(row[metric]) for row in values])) for metric in METRICS}})
    c.write_csv(ROOT / "model_results.csv", rows)
    minimum["seed1_models"] = [row for row in rows if row["training_seed"] == 1]
    c.write_json(ROOT / "minimum_md.json", minimum)
    return minimum


def cross_results() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_structure, summary = [], []
    synthetic_meta = {row["sample_id"]: row for row in test_rows("synthetic")}
    for model in frozen_models():
        datasets = ("md", "synthetic") if model["training_dataset"] == "md" else ("md",)
        for evaluation_dataset in datasets:
            path = ROOT / "evaluations" / model["model_id"] / f"on_{evaluation_dataset}/per_structure_metrics.csv"
            if not path.exists():
                continue
            rows = c.read_csv(path)
            per_structure += rows
            summary.append({
                "model_id": model["model_id"], "training_dataset": model["training_dataset"],
                "evaluation_dataset": evaluation_dataset, "N_train": model["N_train"],
                "N_val": model["N_val"], "training_seed": model["training_seed"],
                "gpu_h": model.get("gpu_h", ""),
                **{metric: float(np.mean([float(row[metric]) for row in rows])) for metric in METRICS},
            })
        if model["training_dataset"] == "synthetic":
            path = Path(model.get("synthetic_test_csv", ""))
            if path.exists():
                rows = c.read_csv(path)
                enriched = []
                for row in rows:
                    meta = synthetic_meta[row["sample_id"]]
                    enriched.append(row | {
                        "model_id": model["model_id"], "training_dataset": "synthetic",
                        "evaluation_dataset": "synthetic", "N_train": model["N_train"],
                        "N_val": model["N_val"], "training_seed": model["training_seed"],
                        "temperature_K": "", "trajectory_id": "", "md_step": "",
                        "dim": meta.get("dim", ""), "amplitude_ang": meta.get("amplitude_ang", ""),
                    })
                per_structure += enriched
                summary.append({
                    "model_id": model["model_id"], "training_dataset": "synthetic",
                    "evaluation_dataset": "synthetic", "N_train": model["N_train"],
                    "N_val": model["N_val"], "training_seed": model["training_seed"],
                    "gpu_h": model.get("gpu_h", ""),
                    **{metric: float(np.mean([float(row[metric]) for row in enriched])) for metric in METRICS},
                })
    c.write_csv(ROOT / "per_structure_results.csv", per_structure)
    return per_structure, summary


def paired_bootstrap(values: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(axis=1)
    return {"difference_mean": float(values.mean()), "difference_p05": float(np.quantile(draws, .05)),
            "difference_p95": float(np.quantile(draws, .95)), "bootstrap_repetitions": 10_000}


def matched_comparisons(per_structure: list[dict[str, Any]]) -> list[dict[str, Any]]:
    md_rows = [row for row in per_structure if row["evaluation_dataset"] == "md"]
    groups: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = {}
    for row in md_rows:
        key = (row["training_dataset"], int(row["N_train"]), int(row["N_val"]), int(row["training_seed"]))
        groups.setdefault(key, {})[row["sample_id"]] = row
    output = []
    pairs = sorted({(int(row["N_train"]), int(row["N_val"]), int(row["training_seed"])) for row in md_rows})
    for n_train, n_val, seed in pairs:
        designed = groups.get(("synthetic", n_train, n_val, seed))
        md = groups.get(("md", n_train, n_val, seed))
        if not designed or not md or designed.keys() != md.keys():
            continue
        for temperature in ("all", *TEMPERATURES):
            ids = [sample_id for sample_id in designed
                   if temperature == "all" or int(designed[sample_id]["temperature_K"]) == temperature]
            for metric_index, metric in enumerate(METRICS):
                differences = np.asarray([float(designed[sample_id][metric]) - float(md[sample_id][metric])
                                          for sample_id in ids])
                ratios = np.asarray([float(designed[sample_id][metric]) / float(md[sample_id][metric])
                                     for sample_id in ids])
                output.append({
                    "comparison": "individual_seed", "N_train": n_train, "N_val": n_val,
                    "training_seed": seed, "temperature_K": temperature, "metric": metric,
                    "n_structures": len(ids), "ratio_mean": float(ratios.mean()),
                    **paired_bootstrap(differences, 20260927 + metric_index + seed * 10),
                })
    # Average all available seeds within each training source before the paired comparison.
    configurations = sorted({(int(row["N_train"]), int(row["N_val"])) for row in md_rows})
    for n_train, n_val in configurations:
        by_source = {}
        for source in ("synthetic", "md"):
            rows = [row for row in md_rows if row["training_dataset"] == source
                    and int(row["N_train"]) == n_train and int(row["N_val"]) == n_val]
            sample_ids = sorted({row["sample_id"] for row in rows})
            by_source[source] = {sample_id: {metric: float(np.mean([float(row[metric]) for row in rows
                                                                    if row["sample_id"] == sample_id]))
                                                    for metric in METRICS}
                                 for sample_id in sample_ids}
        if not by_source["synthetic"] or by_source["synthetic"].keys() != by_source["md"].keys():
            continue
        metadata = next(groups[key] for key in groups if key[0] == "md" and key[1:3] == (n_train, n_val))
        for temperature in ("all", *TEMPERATURES):
            ids = [sample_id for sample_id in by_source["md"]
                   if temperature == "all" or int(metadata[sample_id]["temperature_K"]) == temperature]
            for metric_index, metric in enumerate(METRICS):
                differences = np.asarray([by_source["synthetic"][sample_id][metric] - by_source["md"][sample_id][metric]
                                          for sample_id in ids])
                ratios = np.asarray([by_source["synthetic"][sample_id][metric] / by_source["md"][sample_id][metric]
                                     for sample_id in ids])
                output.append({
                    "comparison": "seed_mean", "N_train": n_train, "N_val": n_val,
                    "training_seed": "mean_available", "temperature_K": temperature, "metric": metric,
                    "n_structures": len(ids), "ratio_mean": float(ratios.mean()),
                    **paired_bootstrap(differences, 20261027 + metric_index),
                })
    c.write_csv(ROOT / "matched_comparisons.csv", output)
    return output


def run_cost(run_dir: Path) -> dict[str, float]:
    marker_path = run_dir / "completed.json"
    marker = c.read_json(marker_path) if marker_path.exists() else {}
    steps = len(parse_mde(run_dir / f"{LABEL}.MDE"))
    measured = c.siesta_seconds(run_dir / "RUN.out")
    fallback_per_step = float(microbenchmark()["results"][0]["cpu_s_per_step"])
    cpu_s = float(measured) if measured is not None else float(marker.get("cpu_s_estimate", steps * fallback_per_step))
    return {"steps": steps, "cpu_s": cpu_s, "wall_s": float(marker.get("wall_s", cpu_s))}


def trajectory_cost(trajectory: dict[str, Any], last_step: int | None = None,
                    include_retries: bool = True) -> dict[str, float]:
    equilibration = [run_cost(Path(path)) for path in trajectory["nvt_runs"]]
    production_runs = [run_cost(Path(path)) for path in trajectory["nve_runs"]]
    used_steps = sum(run["steps"] for run in production_runs) if last_step is None else last_step + 1
    production_cpu, production_wall, consumed = 0.0, 0.0, 0
    for run in production_runs:
        use = min(run["steps"], max(0, used_steps - consumed))
        fraction = use / run["steps"]
        production_cpu += run["cpu_s"] * fraction
        production_wall += run["wall_s"] * fraction
        consumed += run["steps"]
    selected_dirs = {Path(path).resolve() for path in trajectory["nvt_runs"] + trajectory["nve_runs"]}
    root = WORK / "trajectories" / trajectory["trajectory_id"]
    retry_runs = [run_cost(path.parent) for path in root.rglob("completed.json")
                  if path.parent.resolve() not in selected_dirs]
    retry_cpu = sum(run["cpu_s"] for run in retry_runs) / 3600 if include_retries else 0.0
    retry_wall = sum(run["wall_s"] for run in retry_runs) / 3600 if include_retries else 0.0
    return {
        "equilibration_steps": sum(run["steps"] for run in equilibration),
        "production_steps": used_steps, "equilibration_cpu_h": sum(run["cpu_s"] for run in equilibration) / 3600,
        "production_cpu_h": production_cpu / 3600,
        "equilibration_wall_h": sum(run["wall_s"] for run in equilibration) / 3600,
        "production_wall_h": production_wall / 3600,
        "repeated_physics_steps": sum(run["steps"] for run in retry_runs),
        "repeated_physics_cpu_h": retry_cpu, "repeated_physics_wall_h": retry_wall,
    }


def cost_outputs(model_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    campaign, manifest = benchmark_production(), freeze_dataset()
    diagnostics = []
    for trajectory_id, trajectory in campaign["trajectories"].items():
        cost = trajectory_cost(trajectory)
        retained = sum(row["trajectory_id"] == trajectory_id for row in sum(manifest["datasets"].values(), []))
        diagnostics.append({
            "trajectory_id": trajectory_id, "split": "master_MD_pool",
            "temperature_K": trajectory["temperature_K"], "timestep_fs": trajectory["timestep_fs"],
            "stride": 1, "retained_frames": retained,
            "discarded_steps": int(cost["equilibration_steps"] + cost["production_steps"]
                                   + cost["repeated_physics_steps"] - retained),
            "energy_drift_meV_atom_ps": trajectory["nve"]["energy_drift_meV_atom_ps"],
            "decorrelation_thinning": False, **cost,
        })
    c.write_csv(ROOT / "trajectory_diagnostics.csv", diagnostics)

    h_lookup = {(row["training_dataset"], int(row["N_train"]), int(row["N_val"]), int(row["training_seed"])):
                float(row["H_MAE_meV"]) for row in model_summary if row["evaluation_dataset"] == "md"}
    gpu_lookup = {(row["training_dataset"], int(row["N_train"]), int(row["N_val"]), int(row["training_seed"])):
                  row.get("gpu_h", "") for row in model_summary if row["evaluation_dataset"] == "md"}
    frontier = []
    for source in ("md", "synthetic"):
        for n_train in TRAIN_SIZES:
            for n_val in VAL_SIZES:
                if source == "md":
                    selected = manifest["datasets"]["train"][:n_train] + manifest["datasets"]["validation"][:n_val]
                    grouped: dict[str, list[dict[str, Any]]] = {}
                    for row in selected:
                        grouped.setdefault(row["trajectory_id"], []).append(row)
                    costs = [trajectory_cost(campaign["trajectories"][trajectory_id],
                                             max(int(row["md_step"]) for row in rows))
                             for trajectory_id, rows in grouped.items()]
                    cpu_h = sum(row["equilibration_cpu_h"] + row["production_cpu_h"]
                                + row["repeated_physics_cpu_h"] for row in costs)
                    equilibration_h = sum(row["equilibration_cpu_h"] for row in costs)
                    production_h = sum(row["production_cpu_h"] for row in costs)
                    repeated_h = sum(row["repeated_physics_cpu_h"] for row in costs)
                else:
                    train_rows, val_rows = selected_rows(manifest, "synthetic", n_train, n_val)
                    seconds = [c.siesta_seconds(Path(row["reference_dir"]) / "RUN.out") or 0.0
                               for row in train_rows + val_rows]
                    cpu_h, equilibration_h, production_h, repeated_h = sum(seconds) / 3600, 0.0, sum(seconds) / 3600, 0.0
                frontier.append({
                    "training_dataset": source, "N_train": n_train, "N_val": n_val,
                    "construction_labels": n_train + n_val, "siesta_cpu_h_construction": cpu_h,
                    "equilibration_cpu_h": equilibration_h, "production_cpu_h": production_h,
                    "repeated_physics_cpu_h": repeated_h,
                    "gpu_h_seed0": gpu_lookup.get((source, n_train, n_val, 0), ""),
                    "H_MAE_meV_on_MD_Test48_seed0": h_lookup.get((source, n_train, n_val, 0), ""),
                })
    c.write_csv(ROOT / "cost_frontier.csv", frontier)
    return frontier


def final_outputs() -> dict[str, Any]:
    minimum = seed0_analysis()
    per_structure, cross = cross_results()
    c.write_csv(ROOT / "model_results.csv", cross)
    matched = matched_comparisons(per_structure)
    frontier = cost_outputs(cross)
    minimum["seed1_confirmation"] = "not run; exact analogue of the seed-0 Cartesian study"
    c.write_json(ROOT / "minimum_md.json", minimum)
    summary = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mode": "md_cartesian_benchmark",
        "minimum_md": minimum,
        "n_models": len({row["model_id"] for row in cross}), "n_cross_evaluations": len(cross),
        "matched_comparisons": len(matched), "cost_points": len(frontier),
        "protocol_microbenchmark_cpu_h": sum(float(row["cpu_s_estimate"])
                                               for row in microbenchmark()["results"]) / 3600,
        "thermal_incident_at_75C": '"message": "thermal incident"' in (ROOT / "run.log").read_text(),
    }
    c.write_json(ROOT / "final_summary.json", summary)
    write_report(summary, cross, frontier)
    return summary


def write_report(summary: dict[str, Any], cross: list[dict[str, Any]],
                 frontier: list[dict[str, Any]]) -> None:
    minimum = summary["minimum_md"].get("minimum")
    reference = summary["minimum_md"]["reference"]
    selected = (f"Ntrain={minimum['N_train']}, Nval={minimum['N_val']}, Ntest={minimum['N_test']}"
                if minimum else "no alcanzado")
    model_rows = [row for row in cross if row["evaluation_dataset"] == "md" and row["training_seed"] == 0]
    table = "\n".join(
        ["| Dataset | Ntrain | Nval | H-MAE (meV) | Band RMSE (meV) | DOS L1 |",
         "|---|---:|---:|---:|---:|---:|"]
        + [f"| {row['training_dataset']} | {row['N_train']} | {row['N_val']} | {row['H_MAE_meV']:.3f} | "
           f"{row['band_rmse_meV']:.2f} | {row['dos_rel_L1']:.4f} |"
           for row in sorted(model_rows, key=lambda item: (item["training_dataset"], item["N_train"], item["N_val"]))]
    )
    report = f"""# Baseline MD ab initio 6×6 para el presupuesto de etiquetas

Generado el {time.strftime('%Y-%m-%d %H:%M %Z')}.

## Resultado ejecutivo

- Mínimo exploratorio MD: **{selected}**.
- Referencia MD N64/V48/Test48: H-MAE **{reference['H_MAE_meV']:.3f} meV**, bandas **{reference['band_rmse_meV']:.2f} meV**, DOS L1 **{reference['dos_rel_L1']:.4f}**.
- Seed de entrenamiento: **0**, igual que el producto cartesiano sintético comparado.
- Se compararon modelos sintéticos y MD sobre exactamente el mismo MD-Test48 congelado.

## Diseño físico

Se reutilizaron tres trayectorias post-NVT de grafeno 6×6 (72 C), una por 150/300/450 K. Los snapshots consecutivos se repartieron en bloques temporales sin geometrías repetidas: Train64 (22/21/21), Validation48 (16 por temperatura) y Test48 (16 por temperatura). No se aplicó thinning por autocorrelación: MD actúa aquí como método de construcción de dataset comparable con Sobol/LHS, igual que en el estudio MD 5×5 previo.

Los splits comparten trayectoria dentro de cada temperatura, pero nunca el mismo snapshot. Por ello el resultado es un **benchmark exploratorio de muestreo MD**, no una estimación de generalización a trayectorias independientes.

La prueba sostenida con tres trayectorias alcanzó 75 °C y el watchdog las pausó; a partir de esa evidencia la producción quedó limitada a dos procesos. No se operó deliberadamente por encima de 75 °C.

## Resultados sobre MD-Test48

{table}

## Coste

La frontera de coste usa CPU·h SIESTA de construcción (train+validación) y separa equilibrado de producción. Los RUN.out copiados dentro de frames no se vuelven a contabilizar. `cost_frontier.csv` contiene {len(frontier)} puntos; test, pasos intermedios y descartados se informan aparte en `trajectory_diagnostics.csv`.

## Interpretación

Los intervalos asociados a Ntest son **intervalos empíricos de submuestreo**, no intervalos de confianza de generalización. Los 15 modelos MD usan exactamente batch 16, `elementwise_mse`, 8000 actualizaciones, 2000 validaciones, scheduler cosine y mejor checkpoint por `val_loss`, como el producto cartesiano sintético.

La clasificación frente a N64/V48 es un **criterio operativo de equivalencia**, no un veredicto de validez. Un resultado fuera de los márgenes H≤1.05×, bandas≤1.10× o DOS≤1.10× puede seguir siendo científicamente útil; simplemente no se usa como sustituto equivalente de la referencia al buscar el presupuesto mínimo.

## Artefactos

Todos los artefactos están en `Comparison/results/dataset_design_curves_v1/md_label_budget_6x6/`; la UI Dataset Design consume directamente esos CSV/JSON.
"""
    report_path = REPO_ROOT / "docs/dataset_design_md_baseline_6x6.md"
    report_path.write_text(report, encoding="utf-8")


def run_benchmark_test_trajectory(payload: tuple[int, int, float]) -> dict[str, Any]:
    """Create the only new MD labels needed by the benchmark: blind Test48."""

    temperature, ranks, timestep_fs = payload
    trajectory_id = f"test_T{temperature}"
    root = WORK / "trajectories" / trajectory_id
    summary_path = root / "benchmark_test.json"
    if summary_path.exists():
        summary = c.read_json(summary_path)
        if summary.get("valid"):
            return summary
    nvt_runs, source_xv, equilibrium = run_equilibration("test", temperature, ranks)
    run_dir = root / "nve" / "benchmark_0016"
    prepare_run(run_dir, md_fdf("nve", temperature, 16, timestep_fs, True), source_xv, interval=1)
    run_siesta(run_dir, ranks, f"{trajectory_id}_benchmark_nve")
    frames = stored_frames([run_dir], timestep_fs)
    if len(frames) != 16 or min(row["minimum_distance_A"] for row in frames) <= 1.0:
        raise RuntimeError(f"{trajectory_id}: invalid 16-frame benchmark trajectory")
    summary = {
        "trajectory_id": trajectory_id, "split": "test", "temperature_K": temperature,
        "velocity_seed": SEEDS[("test", temperature)],
        "velocity_sha256": c.read_json(initial_xv("test", temperature).parent / "velocity.json")["velocity_sha256"],
        "nvt_runs": [str(path) for path in nvt_runs], "equilibrium": equilibrium,
        "timestep_fs": timestep_fs, "nve_runs": [str(run_dir)],
        "nve": nve_diagnostics([run_dir], timestep_fs), "candidate_frames": frames,
        "retained_frames": 16, "valid": True,
    }
    c.write_json(summary_path, summary)
    return summary


def benchmark_production() -> dict[str, Any]:
    path = ROOT / "benchmark_production_summary.json"
    if path.exists():
        summary = c.read_json(path)
        if summary.get("valid") and summary.get("mode") == "md_cartesian_benchmark":
            return summary
    ranks = int(microbenchmark()["selected_ranks"])
    trajectories = {}
    for temperature in TEMPERATURES:
        root = WORK / "trajectories" / f"train_T{temperature}"
        if temperature != 300:
            trajectory = c.read_json(root / "pilot.json")
            if not trajectory.get("valid"):
                raise RuntimeError(f"missing valid reusable T={temperature} K pilot")
        else:
            trajectory_path = root / "benchmark_cartesian.json"
            trajectory = c.read_json(trajectory_path) if trajectory_path.exists() else {}
            if not trajectory.get("valid"):
                nvt_runs = sorted(path for path in (root / "nvt").glob("segment_*")
                                  if (path / "completed.json").exists())
                if not nvt_runs:
                    raise RuntimeError("missing equilibrated T=300 K NVT trajectory")
                partial = root / "nve_dt1p0/block_0000_0120"
                nve_runs = [partial]
                frames = stored_frames(nve_runs, 1.0)
                missing = max(0, 54 - len(frames))
                if missing:
                    continuation = root / f"nve_cartesian/continuation_{len(frames):04d}_{len(frames) + missing:04d}"
                    if not (continuation / "completed.json").exists() and continuation.exists():
                        shutil.rmtree(continuation)
                    prepare_run(continuation, md_fdf("nve", temperature, missing, 1.0, True),
                                partial / f"{LABEL}.XV", interval=1)
                    run_siesta(continuation, ranks, "train_T300_cartesian_continuation")
                    nve_runs.append(continuation)
                frames = stored_frames(nve_runs, 1.0)
                if len(frames) < 54 or min(row["minimum_distance_A"] for row in frames[:54]) <= 1.0:
                    raise RuntimeError(f"T=300 K provides only {len(frames)}/54 valid MD snapshots")
                trajectory = {
                    "trajectory_id": "train_T300", "split": "master", "temperature_K": 300,
                    "nvt_runs": [str(value) for value in nvt_runs], "timestep_fs": 1.0,
                    "nve_runs": [str(value) for value in nve_runs],
                    "nve": nve_diagnostics(nve_runs, 1.0), "valid": True,
                }
                c.write_json(trajectory_path, trajectory)
        run_dirs = [Path(value) for value in trajectory["nve_runs"]]
        frames = stored_frames(run_dirs, float(trajectory["timestep_fs"]))
        if len(frames) < 54:
            raise RuntimeError(f"T={temperature} K provides only {len(frames)}/54 valid MD snapshots")
        trajectories[f"md_T{temperature}"] = trajectory | {"candidate_frames": frames[:54]}
    summary = {
        "valid": True, "mode": "md_cartesian_benchmark", "label_stride": 1,
        "decorrelation_thinning": False, "timestep_fs": 1.0, "trajectories": trajectories,
    }
    c.write_json(path, summary)
    log("MD Cartesian pool complete", trajectories=len(trajectories), label_stride=1,
        decorrelation_thinning=False)
    return summary


def evenly_spaced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    indices = np.rint(np.linspace(0, len(rows) - 1, count)).astype(int)
    if len(rows) < count or len(set(indices.tolist())) != count:
        raise RuntimeError(f"cannot select {count} unique frames from {len(rows)} candidates")
    return [rows[index] for index in indices]


def benchmark_freeze_dataset() -> dict[str, Any]:
    path = ROOT / "manifest.json"
    if path.exists():
        manifest = c.read_json(path)
        if manifest.get("datasets_frozen") and manifest.get("mode") == "md_cartesian_benchmark":
            return manifest
    campaign = benchmark_production()
    allocations = {150: 22, 300: 21, 450: 21}
    train_by_temperature: dict[int, list[dict[str, Any]]] = {}
    validation_by_temperature: dict[int, list[dict[str, Any]]] = {}
    test_by_temperature: dict[int, list[dict[str, Any]]] = {}
    composition = []
    for temperature in TEMPERATURES:
        trajectory_id = f"md_T{temperature}"
        frames = campaign["trajectories"][trajectory_id]["candidate_frames"]
        n_train = allocations[temperature]
        train_frames = frames[:n_train]
        validation_frames = frames[n_train:n_train + 16]
        test_frames = frames[n_train + 16:n_train + 32]
        train_by_temperature[temperature] = [
            materialize_frame(row, "train", temperature, trajectory_id, index)
            for index, row in enumerate(train_frames)
        ]
        validation_by_temperature[temperature] = [
            materialize_frame(row, "validation", temperature, trajectory_id, index)
            for index, row in enumerate(validation_frames)
        ]
        test_by_temperature[temperature] = [
            materialize_frame(row, "test", temperature, trajectory_id, index)
            for index, row in enumerate(test_frames)
        ]
        composition += [
            {"split": "train", "temperature_K": temperature, "trajectory_id": trajectory_id,
             "first_md_step": train_frames[0]["global_step"], "last_md_step": train_frames[-1]["global_step"],
             "n_frames": len(train_frames)},
            {"split": "validation", "temperature_K": temperature, "trajectory_id": trajectory_id,
             "first_md_step": validation_frames[0]["global_step"],
             "last_md_step": validation_frames[-1]["global_step"], "n_frames": len(validation_frames)},
            {"split": "test", "temperature_K": temperature, "trajectory_id": trajectory_id,
             "first_md_step": test_frames[0]["global_step"], "last_md_step": test_frames[-1]["global_step"],
             "n_frames": len(test_frames)},
        ]
    datasets = {
        "train": round_robin(train_by_temperature, allocations),
        "validation": round_robin(validation_by_temperature, {temperature: 16 for temperature in TEMPERATURES}),
        "test": round_robin(test_by_temperature, {temperature: 16 for temperature in TEMPERATURES}),
    }
    all_rows = sum(datasets.values(), [])
    if [len(datasets[split]) for split in SPLITS] != [64, 48, 48] or len({row["hash"] for row in all_rows}) != 160:
        raise RuntimeError("benchmark must contain 64/48/48 unique geometries")
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "datasets_frozen": True,
        "mode": "md_cartesian_benchmark", "purpose": "same Cartesian label-budget study with MD snapshots",
        "canonical_fdf": str(CANONICAL), "canonical_sha256": sha256(CANONICAL),
        "N_train": 64, "N_val": 48, "N_test": 48, "label_stride": 1,
        "train_sizes": list(TRAIN_SIZES), "validation_sizes": list(VAL_SIZES),
        "test_sizes": list(TEST_SIZES), "decorrelation_thinning": False,
        "selection": "consecutive valid post-NVT snapshots; disjoint temporal blocks; round-robin temperatures",
        "trajectory_overlap_across_splits": True, "exact_geometry_overlap": False,
        "training": {"batch_size": 16, "optimizer_updates": 8000, "validation_checks": 2000,
                     "loss": "graph2mat.metrics.elementwise_mse", "scheduler": "cosine",
                     "checkpoint": "best_val_loss"},
        "train_prefixes": {str(size): [row["sample_id"] for row in datasets["train"][:size]]
                           for size in TRAIN_SIZES},
        "validation_prefixes": {str(size): [row["sample_id"] for row in datasets["validation"][:size]]
                                for size in VAL_SIZES},
        "test_pool": [row["sample_id"] for row in datasets["test"]], "datasets": datasets,
        "frozen_sha256": {split: c.ids_hash([row["sample_id"] + ":" + row["hash"] for row in rows])
                          for split, rows in datasets.items()},
    }
    c.write_csv(ROOT / "dataset_composition.csv", composition)
    c.write_json(path, manifest)
    log("MD benchmark datasets frozen", counts={key: len(value) for key, value in datasets.items()},
        decorrelation_thinning=False)
    return manifest


def benchmark_train_model(profile: str) -> dict[str, Any]:
    manifest = benchmark_freeze_dataset()
    job_id = f"md__N64__V48__seed0__{profile}"
    run_dir = ROOT / "runs" / "md" / job_id
    result_path, done_path = run_dir / "result.json", run_dir / "train_done.json"
    if result_path.exists():
        result = c.read_json(result_path)
        if Path(result["checkpoint"]).exists() and sha256(Path(result["checkpoint"])) == result["checkpoint_sha256"]:
            return result
    train_rows, val_rows = manifest["datasets"]["train"], manifest["datasets"]["validation"]
    train = [c.sample_from_dict(row) for row in train_rows]
    validation = [c.sample_from_dict(row) for row in val_rows]
    if done_path.exists():
        done = c.read_json(done_path)
        checkpoint = Path(done["checkpoint"])
        seconds, peak = float(done["seconds"]), int(done["peak_gpu_mib"])
    else:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        config = c.s4.build_graph2mat_config(
            run_dir, train, validation, run_name=job_id, accelerator="gpu", training_seed=0, **c.TRAIN_KW,
        )
        if profile == "production8k":
            c.apply_config_patch(config, c.fixed_update_elemmse_patch("6x6", 64))
        checkpoint, seconds, peak = train_process(config, run_dir, job_id)
        c.write_json(done_path, {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                                 "seconds": seconds, "peak_gpu_mib": peak})
    result = {
        "model_id": job_id, "training_dataset": "md", "family": "md", "profile": profile,
        "N_train": 64, "N_val": 48, "training_seed": 0,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_policy": "best_val_loss (save_top_k=1)", "train_seconds": seconds,
        "gpu_h": seconds / 3600, "peak_gpu_mib": peak,
        "val_metrics": c.summarize(c.evaluate(checkpoint, validation, run_dir / "val_eval", "gpu")),
        "train_ids_sha256": manifest["frozen_sha256"]["train"],
        "validation_ids_sha256": manifest["frozen_sha256"]["validation"],
    }
    c.write_json(result_path, result)
    return result


def benchmark_training() -> list[dict[str, Any]]:
    benchmark_freeze_dataset()
    if _ACTIVE:
        raise RuntimeError("refusing to overlap Graph2Mat training with active MD")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(benchmark_train_model, ("legacy600", "production8k")))
    return results


def benchmark_synthetic_models() -> list[dict[str, Any]]:
    rows = c.load_results("6x6")
    selected = []
    for family, stage in (("sobol_sparse", "curve"), ("random_cartesian", "curve"),
                          ("latin_hypercube", "lhs")):
        candidates = [row for row in rows if row.get("family") == family and row.get("stage") == stage
                      and int(row.get("N", 0)) == 64 and int(row.get("training_seed", -1)) == 0
                      and Path(row.get("checkpoint", "")).exists()]
        if not candidates:
            raise RuntimeError(f"no comparable N64 legacy model for {family}")
        row = min(candidates, key=lambda item: float(item["val_metrics"]["H_MAE_meV"]))
        selected.append({
            "model_id": f"{family}__N64__V48__seed0__legacy600", "training_dataset": family,
            "family": family, "profile": "legacy600", "N_train": 64, "N_val": 48,
            "training_seed": 0, "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"], "recipe_id": row["recipe_id"],
            "gpu_h": float(row.get("gpu_h", 0)), "siesta_cpu_h_train": float(row.get("siesta_cpu_h_train", 0)),
            "selection_rule": "lowest common-dev H-MAE among pre-existing N64 seed0 base-profile models",
        })
    production = next(row for row in c.read_csv(c.OUT / "label_budget_6x6/model_results.csv")
                      if int(row["N_train"]) == 64 and int(row["N_val"]) == 48)
    selected.append({
        "model_id": "sobol_sparse__N64__V48__seed0__production8k", "training_dataset": "sobol_sparse",
        "family": "sobol_sparse", "profile": "production8k", "N_train": 64, "N_val": 48,
        "training_seed": 0, "checkpoint": production["checkpoint"],
        "checkpoint_sha256": production["checkpoint_sha256"],
        "recipe_id": "sobol_sparse__3D__R0.12__d64__scaled",
        "gpu_h": float(production["train_seconds"]) / 3600,
        "siesta_cpu_h_train": float(production["siesta_cpu_h_train"]),
        "selection_rule": "pre-existing production winner fixed before opening MD-Test48",
    })
    return selected


def benchmark_models() -> list[dict[str, Any]]:
    md = [c.read_json(path) for path in sorted((ROOT / "runs/md").glob("*__legacy600/result.json"))]
    md += [c.read_json(path) for path in sorted((ROOT / "runs/md").glob("*__production8k/result.json"))]
    return md + benchmark_synthetic_models()


def benchmark_evaluation() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    benchmark_training()
    models = benchmark_models()
    frozen = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_hashes": benchmark_freeze_dataset()["frozen_sha256"],
        "models": [{key: row[key] for key in ("model_id", "training_dataset", "profile", "checkpoint",
                                                 "checkpoint_sha256")} for row in models],
    }
    for row in models:
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.exists() or sha256(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError(f"invalid benchmark checkpoint: {row['model_id']}")
    c.write_json(ROOT / "benchmark_checkpoints_frozen.json", frozen)
    per_structure, summaries = [], []
    for model in models:
        for evaluation_dataset in ("md", "synthetic"):
            path = evaluate_model(model, evaluation_dataset)
            rows = c.read_csv(path)
            per_structure += rows
            summaries.append({
                "model_id": model["model_id"], "training_dataset": model["training_dataset"],
                "profile": model["profile"], "recipe_id": model.get("recipe_id", "md"),
                "evaluation_dataset": evaluation_dataset, "N_train": 64, "N_val": 48,
                "training_seed": 0, "gpu_h": model.get("gpu_h", ""),
                "siesta_cpu_h_train": model.get("siesta_cpu_h_train", ""),
                **{metric: float(np.mean([float(row[metric]) for row in rows])) for metric in METRICS},
            })
    c.write_csv(ROOT / "per_structure_results.csv", per_structure)
    c.write_csv(ROOT / "model_results.csv", summaries)
    return per_structure, summaries


def benchmark_final_outputs() -> dict[str, Any]:
    per_structure, summaries = benchmark_evaluation()
    by_model_dataset = {(row["model_id"], row["evaluation_dataset"]): row for row in summaries}
    model_by_id = {row["model_id"]: row for row in benchmark_models()}
    md_reference = {row["profile"]: row["model_id"] for row in benchmark_models()
                    if row["training_dataset"] == "md"}
    matched = []
    for model in benchmark_models():
        if model["training_dataset"] == "md":
            continue
        reference_id = md_reference[model["profile"]]
        for dataset in ("md", "synthetic"):
            designed = {row["sample_id"]: row for row in per_structure
                        if row["model_id"] == model["model_id"] and row["evaluation_dataset"] == dataset}
            reference = {row["sample_id"]: row for row in per_structure
                         if row["model_id"] == reference_id and row["evaluation_dataset"] == dataset}
            ids = sorted(designed.keys() & reference.keys())
            for metric_index, metric in enumerate(METRICS):
                differences = np.asarray([float(designed[key][metric]) - float(reference[key][metric]) for key in ids])
                row = {
                    "model_id": model["model_id"], "reference_model_id": reference_id,
                    "profile": model["profile"], "evaluation_dataset": dataset, "metric": metric,
                    "ratio_of_means": by_model_dataset[(model["model_id"], dataset)][metric]
                                      / by_model_dataset[(reference_id, dataset)][metric],
                    **paired_bootstrap(differences, 20260927 + metric_index),
                }
                matched.append(row)
    c.write_csv(ROOT / "matched_comparisons.csv", matched)
    pilot_summary = pilots()
    md_build_dirs = {Path(run) for row in pilot_summary["temperatures"].values()
                     for run in row["nvt_runs"] + row["nve_runs"]}
    md_build_cpu_h = sum(run_cost(path)["cpu_s"] for path in md_build_dirs) / 3600
    costs = []
    for row in summaries:
        if row["evaluation_dataset"] != "md":
            continue
        model = model_by_id[row["model_id"]]
        costs.append({
            "model_id": row["model_id"], "method": row["training_dataset"], "profile": row["profile"],
            "N_train": 64, "N_val": 48,
            "siesta_cpu_h_train_val": md_build_cpu_h if row["training_dataset"] == "md"
                                      else float(model.get("siesta_cpu_h_train", 0)),
            "gpu_h": float(model.get("gpu_h", 0)), "H_MAE_meV_on_MD_Test48": row["H_MAE_meV"],
        })
    c.write_csv(ROOT / "cost_frontier.csv", costs)
    c.write_csv(ROOT / "cartesian_results.csv", summaries)
    c.write_json(ROOT / "minimum_md.json", {
        "mode": "benchmark_only", "minimum_not_estimated": True,
        "reference_models": md_reference, "reason": "MD is a benchmark, not a label-budget sweep",
    })
    summary = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mode": "md_sampling_benchmark",
        "n_models": len(benchmark_models()), "n_evaluations": len(summaries),
        "g_role": "diagnostic only", "label_stride": 1, "reference_models": md_reference,
    }
    c.write_json(ROOT / "final_summary.json", summary)
    lines = ["# Benchmark MD ab initio 6×6", "", "MD se usa únicamente como referencia frente a los métodos de muestreo implementados.", "",
             "| Perfil | Método de entrenamiento | Test | H-MAE (meV) | Band RMSE (meV) | DOS L1 |",
             "|---|---|---|---:|---:|---:|"]
    lines += [f"| {row['profile']} | {row['training_dataset']} | {row['evaluation_dataset']} | "
              f"{row['H_MAE_meV']:.3f} | {row['band_rmse_meV']:.2f} | {row['dos_rel_L1']:.4f} |"
              for row in summaries]
    lines += ["", "`g` se conserva como diagnóstico de autocorrelación; no se usa para descartar etiquetas. ",
              "El test MD usa tres trayectorias independientes, una por temperatura. Train y validación usan bloques temporales disjuntos de los pilotos."]
    (REPO_ROOT / "docs/dataset_design_md_baseline_6x6.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("MD SAMPLING BENCHMARK COMPLETE", models=len(benchmark_models()))
    return summary


def run_all() -> dict[str, Any]:
    benchmark_freeze_dataset()
    initial_training()
    return final_outputs()


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "self-test", "microbenchmark", "pilots",
                                          "production", "freeze", "train", "evaluate", "analyze", "final", "all"))
    return parser.parse_args()


def main() -> None:
    args = cli()
    if args.phase == "preflight":
        preflight()
    elif args.phase == "self-test":
        preflight()
        self_test()
    elif args.phase == "microbenchmark":
        preflight()
        self_test()
        microbenchmark()
    elif args.phase == "pilots":
        preflight()
        self_test()
        pilots()
    elif args.phase == "production":
        preflight()
        self_test()
        benchmark_production()
    elif args.phase == "freeze":
        preflight()
        self_test()
        freeze_dataset()
    elif args.phase == "train":
        preflight()
        self_test()
        initial_training()
    elif args.phase == "evaluate":
        preflight()
        self_test()
        evaluate_initial()
    elif args.phase == "analyze":
        preflight()
        self_test()
        seed0_analysis()
    elif args.phase == "final":
        preflight()
        self_test()
        final_outputs()
    elif args.phase == "all":
        preflight()
        self_test()
        run_all()


if __name__ == "__main__":
    main()
