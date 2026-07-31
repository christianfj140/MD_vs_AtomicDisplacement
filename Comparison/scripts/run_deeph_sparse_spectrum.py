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
    ("K", (1.0 / 3.0, 2.0 / 3.0, 0.0)),
    ("M", (0.5, 0.5, 0.0)),
    ("Γ", (0.0, 0.0, 0.0)),
]
BAND_PATH_INDICES = {
    "gamma-k-m-gamma": (0, 1, 2, 3),
    "k-gamma-m": (1, 0, 2),
    "kprime-gamma-k": (),
}
GIB = 1024**3
MIN_FREE_DISK_PERCENT = 12.0
RESERVED_SYSTEM_MEMORY_BYTES = 12 * GIB
MIN_SOLVER_MEMORY_BYTES = 8 * GIB
MIN_AVAILABLE_MEMORY_BYTES = 20 * GIB
PARDISO_OOC_MAX_CORE_MB = 8192
SOLVER_THREADS = 8
DISK_POLL_SECONDS = 300.0
MAX_CPU_TEMPERATURE_C = 85.0
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
            try:
                process.wait(timeout=poll_seconds)
            except subprocess.TimeoutExpired:
                pass
        returncode = process.wait()
    return returncode, abort_reason, minimum_observed, maximum_cpu_temperature, minimum_available_memory


def reciprocal_angle_deg(reciprocal: np.ndarray) -> float:
    b1, b2 = reciprocal[0], reciprocal[1]
    cosine = float(np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _wigner_seitz_multiplicity(point: tuple[float, float, float], reciprocal: np.ndarray) -> int:
    dimensions = reciprocal.shape[0]
    cartesian = np.asarray(point[:dimensions]) @ reciprocal
    distances = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            fractional = np.zeros(dimensions)
            fractional[:2] = (i, j)
            lattice_point = fractional @ reciprocal
            distances.append(float(np.linalg.norm(cartesian - lattice_point)))
    minimum = min(distances)
    return sum(np.isclose(value, minimum, rtol=1e-7, atol=1e-10) for value in distances)


def validated_hexagonal_k_path(reciprocal: np.ndarray) -> tuple[list, dict]:
    b1, b2 = reciprocal[0], reciprocal[1]
    if not np.isclose(np.linalg.norm(b1), np.linalg.norm(b2), rtol=1e-5):
        raise RuntimeError("The in-plane reciprocal lattice is not hexagonal")
    angle = reciprocal_angle_deg(reciprocal)
    if np.isclose(angle, 120.0, atol=1e-4):
        k_point, m_point = (1 / 3, 2 / 3, 0.0), (0.5, 0.5, 0.0)
        k_prime = (2 / 3, 1 / 3, 0.0)
    elif np.isclose(angle, 60.0, atol=1e-4):
        k_point, m_point = (1 / 3, 1 / 3, 0.0), (0.5, 0.0, 0.0)
        k_prime = (2 / 3, 2 / 3, 0.0)
    else:
        raise RuntimeError(f"Expected a 60 or 120 degree reciprocal basis, found {angle:.8f}")
    k_multiplicity = _wigner_seitz_multiplicity(k_point, reciprocal)
    k_prime_multiplicity = _wigner_seitz_multiplicity(k_prime, reciprocal)
    m_multiplicity = _wigner_seitz_multiplicity(m_point, reciprocal)
    if k_multiplicity < 3 or k_prime_multiplicity < 3 or m_multiplicity < 2:
        raise RuntimeError("Selected K/M points failed the Wigner-Seitz geometry check")
    path = [("Γ", (0.0, 0.0, 0.0)), ("K", k_point), ("M", m_point), ("Γ", (0.0, 0.0, 0.0))]
    return path, {
        "reciprocal_angle_deg": angle,
        "K_wigner_seitz_equidistant_points": int(k_multiplicity),
        "K_prime_fractional": list(k_prime),
        "K_prime_wigner_seitz_equidistant_points": int(k_prime_multiplicity),
        "M_wigner_seitz_equidistant_points": int(m_multiplicity),
        "K_cartesian_inv_ang": (np.asarray(k_point) @ reciprocal).tolist(),
        "K_prime_cartesian_inv_ang": (np.asarray(k_prime) @ reciprocal).tolist(),
        "M_cartesian_inv_ang": (np.asarray(m_point) @ reciprocal).tolist(),
        "Gamma_K_distance_inv_ang": float(np.linalg.norm(np.asarray(k_point) @ reciprocal)),
        "Gamma_M_distance_inv_ang": float(np.linalg.norm(np.asarray(m_point) @ reciprocal)),
    }


def select_band_path(reciprocal: np.ndarray, name: str) -> tuple[list, dict]:
    path, validation = validated_hexagonal_k_path(reciprocal)
    if name == "kprime-gamma-k":
        k_prime = tuple(validation["K_prime_fractional"])
        return [("K′", k_prime), path[0], path[1]], validation
    try:
        return [path[index] for index in BAND_PATH_INDICES[name]], validation
    except KeyError as error:
        raise ValueError(f"Unsupported band path: {name}") from error


def mulliken_projection_groups(input_dir: Path) -> dict:
    """Build one-based orbital masks from the exact DeepH/OpenMX ordering."""
    elements = np.loadtxt(input_dir / "element.dat", dtype=int).reshape(-1)
    positions = np.loadtxt(input_dir / "site_positions.dat", dtype=float).T
    shells = [
        [int(value) for value in line.split()]
        for line in (input_dir / "orbital_types.dat").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(elements) != len(shells) or positions.shape != (len(elements), 3):
        raise RuntimeError("Element, position and orbital metadata disagree")
    orbital_counts = [sum(2 * ell + 1 for ell in atom_shells) for atom_shells in shells]
    offsets = np.cumsum([0, *orbital_counts])
    carbon_atoms = np.flatnonzero(elements == 6)
    if not len(carbon_atoms):
        raise RuntimeError("No carbon atoms available for the moire projection")
    carbon_shells = {tuple(shells[index]) for index in carbon_atoms}
    if carbon_shells != {(0, 1)}:
        raise RuntimeError(f"Carbon orbital basis is not uniformly [s,p]: {carbon_shells}")
    carbon_z = np.unique(np.round(positions[carbon_atoms, 2], 6))
    if len(carbon_z) != 2:
        raise RuntimeError(f"Expected two graphene z layers, found {carbon_z.tolist()}")
    layer_boundary = float(carbon_z.mean())

    def atom_orbitals(atom_indices: np.ndarray) -> list[int]:
        return [
            orbital + 1
            for atom in atom_indices
            for orbital in range(int(offsets[atom]), int(offsets[atom + 1]))
        ]

    lower_atoms = carbon_atoms[positions[carbon_atoms, 2] < layer_boundary]
    upper_atoms = carbon_atoms[positions[carbon_atoms, 2] > layer_boundary]
    pz = []
    pz_local_indices = set()
    for atom in carbon_atoms:
        local_offset = 0
        for ell in shells[atom]:
            if ell == 1:
                # DeepH expands m=-l..l. Graph2Mat's documented
                # siesta_spherical convention is (-py, pz, -px), so m=0 is pz.
                pz_local_indices.add(local_offset + ell)
                pz.append(int(offsets[atom] + local_offset + ell + 1))
            local_offset += 2 * ell + 1
    groups = {
        "carbon_pz": pz,
        "carbon": atom_orbitals(carbon_atoms),
        "graphene_lower": atom_orbitals(lower_atoms),
        "graphene_upper": atom_orbitals(upper_atoms),
        "boron": atom_orbitals(np.flatnonzero(elements == 5)),
        "nitrogen": atom_orbitals(np.flatnonzero(elements == 7)),
    }
    groups["hbn"] = sorted(groups["boron"] + groups["nitrogen"])
    return {
        "status": "validated",
        "index_base": 1,
        "groups": groups,
        "mapping": {
            "atom_count": int(len(elements)),
            "total_orbitals": int(offsets[-1]),
            "species_atom_counts": {
                "B": int(np.sum(elements == 5)),
                "C": int(np.sum(elements == 6)),
                "N": int(np.sum(elements == 7)),
            },
            "species_orbitals_per_atom": {
                str(int(element)): sorted({
                    orbital_counts[index]
                    for index, value in enumerate(elements)
                    if value == element
                })
                for element in np.unique(elements)
            },
            "carbon_shells": [list(value) for value in sorted(carbon_shells)],
            "carbon_pz_local_indices_zero_based": sorted(pz_local_indices),
            "orbital_expansion": "shell order, then m=-l..l",
            "basis_convention": "siesta_spherical=(-py,pz,-px) for l=1",
            "basis_evidence": (
                "graph2mat/core/data/basis.py maps siesta_spherical to -yz-x; "
                "PointBasis expands m=-l..l"
            ),
            "graphene_z_levels": carbon_z.tolist(),
            "layer_boundary_z": layer_boundary,
            "lower_carbon_atoms": int(len(lower_atoms)),
            "upper_carbon_atoms": int(len(upper_atoms)),
            "group_orbital_counts": {name: len(indices) for name, indices in groups.items()},
        },
    }


def estimated_neutrality_reference(input_dir: Path, energy_eV: float) -> dict:
    """Record electron-count evidence without claiming a target chemical potential."""
    if not (input_dir / "element.dat").is_file():
        return {
            "label": "cero visual de neutralidad estimado",
            "method": "median_small_cell_SIESTA_Fermi_levels",
            "energy_eV": energy_eV,
            "expected_neutral_valence_electrons": None,
            "chemical_potential_available": False,
            "vbm_eV": None,
            "cbm_eV": None,
            "gap_eV": None,
            "limitation": "element.dat is unavailable; target electron count was not inferred.",
        }
    elements = np.loadtxt(input_dir / "element.dat", dtype=int).reshape(-1)
    pseudo_root = REPO_ROOT / "Comparison/datasets/graphene_hbn_bilayer_md_nested/n480"
    evidence = {}
    valence = {}
    for atomic_number, symbol in ((5, "B"), (7, "N")):
        path = pseudo_root / f"{symbol}.psml"
        match = re.search(r'z-pseudo="([0-9.]+)"', path.read_text(encoding="utf-8")) if path.is_file() else None
        if match:
            valence[atomic_number] = float(match.group(1))
            evidence[symbol] = {"path": str(path), "field": "pseudo-atom-spec z-pseudo"}
    carbon_path = pseudo_root / "C.psf"
    if carbon_path.is_file():
        header = carbon_path.read_text(encoding="utf-8", errors="replace").splitlines()[:6]
        candidates = [line.split()[-1] for line in header if len(line.split()) >= 6]
        try:
            valence[6] = float(candidates[-1])
            evidence["C"] = {"path": str(carbon_path), "field": "PSF header valence charge"}
        except (IndexError, ValueError):
            pass
    counts = {int(element): int(np.sum(elements == element)) for element in np.unique(elements)}
    electron_count = (
        int(round(sum(count * valence[element] for element, count in counts.items())))
        if set(counts) <= set(valence)
        else None
    )
    source_fdf = REPO_ROOT / "materials/bilayer_graphene_hBN_AA/RUN.fdf"
    return {
        "label": "cero visual de neutralidad estimado",
        "method": "median_small_cell_SIESTA_Fermi_levels",
        "energy_eV": energy_eV,
        "expected_neutral_valence_electrons": electron_count,
        "valence_electrons_per_species": {str(key): value for key, value in valence.items()},
        "species_atom_counts": {str(key): value for key, value in counts.items()},
        "pseudopotential_evidence": evidence,
        "net_charge_electrons": 0.0,
        "spin_degeneracy": 2,
        "spin_assumption": "non-spin-polarized source calculations",
        "source_electronic_temperature_eV": 0.075,
        "source_fdf": str(source_fdf),
        "chemical_potential_available": False,
        "vbm_eV": None,
        "cbm_eV": None,
        "gap_eV": None,
        "limitation": (
            "Electron count is known, but only a partial shift-invert spectrum is available; "
            "no target SCF occupations or converged 2-D chemical potential exist."
        ),
    }


def projected_band_data(output_dir: Path, raw_energies: np.ndarray) -> tuple[list[list[dict]], dict]:
    rows = []
    normalization_errors = []
    residuals = []
    partition_errors = []
    carbon_layer_errors = []
    hbn_species_errors = []
    hermiticity_h = []
    hermiticity_s = []
    for k_index, energy_row in enumerate(raw_energies):
        payload = json.loads(
            (output_dir / f"mulliken_{k_index:03d}.json").read_text(encoding="utf-8")
        )
        if payload.get("k_index") != k_index:
            raise RuntimeError("Mulliken k-point sequence is incomplete")
        recorded = np.asarray(payload["energies_eV"], dtype=float)
        if recorded.shape != energy_row.shape or not np.allclose(recorded, energy_row, atol=1e-8):
            raise RuntimeError("Mulliken weights do not match raw solver eigenvalues")
        weights = {key: np.asarray(value, dtype=float) for key, value in payload["mulliken_weights"].items()}
        normalizations = np.asarray(payload["normalization_cdagger_s_c"], dtype=float)
        row_residuals = np.asarray(payload["generalized_relative_residual"], dtype=float)
        if not all(np.isfinite(value).all() for value in [*weights.values(), normalizations, row_residuals]):
            raise RuntimeError("Non-finite Mulliken projection output")
        partition = weights["carbon"] + weights["boron"] + weights["nitrogen"]
        carbon_layers = weights["graphene_lower"] + weights["graphene_upper"]
        hbn_species = weights["boron"] + weights["nitrogen"]
        normalization_errors.extend(np.abs(normalizations - 1).tolist())
        residuals.extend(row_residuals.tolist())
        partition_errors.extend(np.abs(partition - 1).tolist())
        carbon_layer_errors.extend(np.abs(weights["carbon"] - carbon_layers).tolist())
        hbn_species_errors.extend(np.abs(weights["hbn"] - hbn_species).tolist())
        rows.append([
            {
                "solver_band_index": int(band),
                "weight_c_pz": float(weights["carbon_pz"][band]),
                "weight_c_total": float(weights["carbon"][band]),
                "weight_graphene_lower": float(weights["graphene_lower"][band]),
                "weight_graphene_upper": float(weights["graphene_upper"][band]),
                "weight_hbn": float(weights["hbn"][band]),
                "weight_b": float(weights["boron"][band]),
                "weight_n": float(weights["nitrogen"][band]),
                "layer_polarization": float(
                    (weights["graphene_upper"][band] - weights["graphene_lower"][band])
                    / (
                        carbon_layers[band]
                        if abs(carbon_layers[band]) > 1e-15
                        else np.copysign(1e-15, carbon_layers[band] or 1.0)
                    )
                ),
                "normalization_cdagger_s_c": float(normalizations[band]),
                "generalized_relative_residual": float(row_residuals[band]),
                "projection_status": "valid_s_aware_mulliken",
            }
            for band in np.argsort(energy_row)
        ])
        quality = json.loads(
            (output_dir / f"matrix_quality_{k_index:03d}.json").read_text(encoding="utf-8")
        )
        hermiticity_h.append(float(quality["h_relative_hermiticity_before_solver_symmetrization"]))
        hermiticity_s.append(float(quality["s_relative_hermiticity_before_solver_symmetrization"]))
    valid = (
        max(normalization_errors, default=0.0) < 1e-8
        and max(partition_errors, default=0.0) < 1e-6
        and max(carbon_layer_errors, default=0.0) < 1e-6
        and max(hbn_species_errors, default=0.0) < 1e-6
        and max(residuals, default=0.0) < 1e-5
    )
    diagnostics = {
        "status": "valid" if valid else "failed",
        "method": "s_aware_mulliken_streaming",
        "maximum_normalization_error": max(normalization_errors, default=0.0),
        "maximum_partition_sum_error": max(partition_errors, default=0.0),
        "maximum_carbon_layer_sum_error": max(carbon_layer_errors, default=0.0),
        "maximum_hbn_species_sum_error": max(hbn_species_errors, default=0.0),
        "maximum_generalized_relative_residual": max(residuals, default=0.0),
        "median_generalized_relative_residual": float(np.median(residuals)) if residuals else None,
        "p95_generalized_relative_residual": float(np.percentile(residuals, 95)) if residuals else None,
        "maximum_h_relative_hermiticity_before_solver_symmetrization": max(hermiticity_h, default=0.0),
        "maximum_s_relative_hermiticity_before_solver_symmetrization": max(hermiticity_s, default=0.0),
        "overlap_positive_definite_full_space": "not_evaluated_large_sparse_system",
        "sampled_eigenstate_s_norm_positive": True,
        "eigenvectors_persisted": False,
        "validation_tolerances": {
            "normalization_absolute": 1e-8,
            "partition_absolute": 1e-6,
            "generalized_relative_residual": 1e-5,
            "energy_match_absolute_eV": 1e-8,
        },
    }
    if diagnostics["status"] != "valid":
        raise RuntimeError(f"Mulliken partition validation failed: {diagnostics}")
    return rows, diagnostics


def projected_dos_data(
    output_dir: Path,
    raw_energies: np.ndarray,
    fermi_level: float,
    broadening_mev: float,
    energy_window_mev: float,
) -> tuple[list[dict], dict]:
    """Build uniformly weighted DOS/PDOS from the actual 2-D solver mesh."""
    if broadening_mev <= 0 or energy_window_mev <= 0:
        raise ValueError("DOS broadening and energy window must be positive")
    sigma = broadening_mev / 1000.0
    grid = np.linspace(-energy_window_mev / 1000.0, energy_window_mev / 1000.0, 801)
    fields = {
        "pdos_c_pz": "carbon_pz",
        "pdos_graphene_lower": "graphene_lower",
        "pdos_graphene_upper": "graphene_upper",
        "pdos_hbn": "hbn",
    }
    curves = {"dos_total": np.zeros_like(grid), **{name: np.zeros_like(grid) for name in fields}}
    normalization_errors, partition_errors, residuals = [], [], []
    nk = raw_energies.shape[0]
    prefactor = 1.0 / (nk * sigma * np.sqrt(2.0 * np.pi))
    for k_index, energy_row in enumerate(raw_energies):
        payload = json.loads(
            (output_dir / f"mulliken_{k_index:03d}.json").read_text(encoding="utf-8")
        )
        recorded = np.asarray(payload["energies_eV"], dtype=float)
        weights = {
            key: np.asarray(value, dtype=float)
            for key, value in payload["mulliken_weights"].items()
        }
        norms = np.asarray(payload["normalization_cdagger_s_c"], dtype=float)
        row_residuals = np.asarray(payload["generalized_relative_residual"], dtype=float)
        values = [recorded, norms, row_residuals, *weights.values()]
        if recorded.shape != energy_row.shape or not np.allclose(recorded, energy_row, atol=1e-8):
            raise RuntimeError("Projected DOS weights do not match solver eigenvalues")
        if not all(np.isfinite(value).all() for value in values):
            raise RuntimeError("Non-finite projected DOS input")
        normalization_errors.extend(np.abs(norms - 1.0).tolist())
        partition_errors.extend(
            np.abs(weights["carbon"] + weights["boron"] + weights["nitrogen"] - 1.0).tolist()
        )
        residuals.extend(row_residuals.tolist())
        kernels = prefactor * np.exp(
            -0.5 * ((grid[:, None] - (recorded - fermi_level)[None, :]) / sigma) ** 2
        )
        curves["dos_total"] += kernels.sum(axis=1)
        for output_name, group_name in fields.items():
            curves[output_name] += kernels @ weights[group_name]
    rows = [
        {
            "energy_aligned_eV": float(energy),
            "energy_eV": float(energy + fermi_level),
            **{name: float(values[index]) for name, values in curves.items()},
        }
        for index, energy in enumerate(grid)
    ]
    valid = (
        max(normalization_errors, default=0.0) < 1e-8
        and max(partition_errors, default=0.0) < 1e-6
        and max(residuals, default=0.0) < 1e-5
    )
    diagnostics = {
        "status": "valid" if valid else "failed",
        "method": "uniform_2d_mesh_gaussian_s_aware_mulliken",
        "kpoint_count": nk,
        "uniform_k_weight": 1.0 / nk,
        "broadening_meV": broadening_mev,
        "energy_window_meV": energy_window_mev,
        "state_count_per_k": int(raw_energies.shape[1]),
        "integrated_total_states_in_energy_window": float(np.trapezoid(curves["dos_total"], grid)),
        "full_orbital_spectrum": False,
        "scope_limitation": (
            "DOS/PDOS contains only the requested shift-invert states near the visual "
            "neutrality reference, not all 117222 orbital eigenstates."
        ),
        "maximum_normalization_error": max(normalization_errors, default=0.0),
        "maximum_partition_sum_error": max(partition_errors, default=0.0),
        "maximum_generalized_relative_residual": max(residuals, default=0.0),
        "validation_tolerances": {
            "normalization_absolute": 1e-8,
            "partition_absolute": 1e-6,
            "generalized_relative_residual": 1e-5,
            "energy_match_absolute_eV": 1e-8,
        },
    }
    if diagnostics["status"] != "valid":
        raise RuntimeError(f"Projected DOS validation failed: {diagnostics}")
    return rows, diagnostics


def projected_dos_observables(rows: list[dict]) -> dict:
    energies = np.asarray([row["energy_aligned_eV"] for row in rows], dtype=float)
    total = np.asarray([row["dos_total"] for row in rows], dtype=float)
    peak_indices = np.flatnonzero((total[1:-1] > total[:-2]) & (total[1:-1] >= total[2:])) + 1
    strongest = sorted(peak_indices, key=lambda index: total[index], reverse=True)[:8]
    return {
        "scientific_status": "broadening_and_mesh_dependent_diagnostic",
        "dos_at_visual_neutrality_states_per_eV_cell": float(np.interp(0.0, energies, total)),
        "van_hove_peak_candidates": [
            {
                "energy_aligned_meV": 1000.0 * float(energies[index]),
                "dos_states_per_eV_cell": float(total[index]),
            }
            for index in strongest
        ],
        "gap_claimed": False,
    }


def band_kpoints(
    points_per_segment: int,
    gamma_only: bool = False,
    k_path: list | None = None,
) -> list[tuple[str, np.ndarray]]:
    path = K_PATH if k_path is None else k_path
    if gamma_only:
        return [("Γ", np.asarray(path[0][1], dtype=float))]
    if points_per_segment < 1:
        raise ValueError("points_per_segment must be positive")
    if points_per_segment == 1:
        return [(label, np.asarray(point, dtype=float)) for label, point in path[:-1]]
    points = [(path[0][0], np.asarray(path[0][1], dtype=float))]
    for (_left_label, left), (right_label, right) in zip(path[:-1], path[1:], strict=True):
        for step, point in enumerate(np.linspace(left, right, points_per_segment)[1:], start=1):
            points.append((right_label if step == points_per_segment - 1 else "", point))
    return points


def band_k_data(points_per_segment: int, gamma_only: bool = False, k_path: list | None = None) -> list[str]:
    return [
        f"1 {' '.join(str(value) for value in point)} {' '.join(str(value) for value in point)}"
        for _label, point in band_kpoints(points_per_segment, gamma_only, k_path)
    ]


def band_points(
    energies: np.ndarray,
    points_per_segment: int,
    reciprocal: np.ndarray,
    fermi: float,
    gamma_only: bool = False,
    k_path: list | None = None,
) -> list[dict]:
    samples = band_kpoints(points_per_segment, gamma_only, k_path)
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


def sort_bands_by_energy(energies: np.ndarray) -> np.ndarray:
    """Return the conventional ascending-energy band rank at every k-point."""
    if energies.ndim != 2 or not np.isfinite(energies).all():
        raise ValueError("Band energies must be a finite 2-D array")
    return np.sort(energies, axis=1)


def kprime_comparison(bands: list[dict]) -> dict:
    """Compare endpoint state sets without asserting one-to-one band identity."""
    last_k = max(int(point["k_index"]) for point in bands)
    left = sorted(
        (point for point in bands if int(point["k_index"]) == 0),
        key=lambda point: int(point["band_index"]),
    )
    right = sorted(
        (point for point in bands if int(point["k_index"]) == last_k),
        key=lambda point: int(point["band_index"]),
    )
    if not left or len(left) != len(right):
        raise RuntimeError("K'/K endpoint spectra are incomplete")
    energy_differences = 1000.0 * (
        np.asarray([point["energy_aligned_eV"] for point in left])
        - np.asarray([point["energy_aligned_eV"] for point in right])
    )
    result = {
        "status": "completed",
        "pairing": "ascending_energy_rank_only_not_band_identity",
        "state_count": len(left),
        "energy_rank_rmse_meV": float(np.sqrt(np.mean(energy_differences**2))),
        "energy_rank_max_abs_difference_meV": float(np.max(np.abs(energy_differences))),
        "symmetry_equivalence_required": False,
        "limitation": "hBN and the rigid geometry may break K/K' equivalence.",
    }
    for key in ("weight_c_pz", "weight_graphene_lower", "weight_graphene_upper", "weight_hbn"):
        if all(point.get(key) is not None for point in [*left, *right]):
            differences = np.asarray([point[key] for point in left]) - np.asarray(
                [point[key] for point in right]
            )
            result[f"{key}_rank_mean_abs_difference"] = float(np.mean(np.abs(differences)))
    return result


def track_bands_by_overlap(output_dir: Path, energies: np.ndarray) -> tuple[np.ndarray, dict]:
    from scipy.optimize import linear_sum_assignment

    steps = [
        json.loads((output_dir / f"band_tracking_{k_index:03d}.json").read_text(encoding="utf-8"))
        for k_index in range(len(energies))
    ]
    if any(int(step["k_index"]) != index for index, step in enumerate(steps)):
        raise RuntimeError("Band-tracking k-point sequence is incomplete")
    orders = [np.argsort(energies[0])]
    assigned_overlaps = []
    for step in steps[1:]:
        overlap = np.asarray(step["overlap_from_previous"], dtype=float)
        if overlap.shape != (energies.shape[1], energies.shape[1]) or not np.isfinite(overlap).all():
            raise RuntimeError("Invalid adjacent-k eigenvector overlap matrix")
        rows, columns = linear_sum_assignment(-overlap)
        mapping = np.empty(energies.shape[1], dtype=int)
        mapping[rows] = columns
        orders.append(mapping[orders[-1]])
        assigned_overlaps.extend(overlap[rows, columns].tolist())
    tracked = np.asarray([row[order] for row, order in zip(energies, orders, strict=True)])
    return tracked, {
        "method": "maximum_adjacent_eigenvector_overlap_hungarian",
        "status": "completed",
        "minimum_assigned_overlap": min(assigned_overlaps) if assigned_overlaps else 1.0,
        "mean_assigned_overlap": float(np.mean(assigned_overlaps)) if assigned_overlaps else 1.0,
        "raw_eigenvalues_preserved_in": str(output_dir / "openmx.Band"),
        "eigenvectors_persisted": False,
        "working_storage": "previous_and_current_k_only",
    }


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
    track_bands: bool = False,
    band_path: str = "gamma-k-m-gamma",
    project_mulliken: bool = False,
    dos_broadening_mev: float = 0.5,
    dos_energy_window_mev: float = 100.0,
) -> dict:
    if backend not in {"cpu_mkl_pardiso", "gpu_cudss"}:
        raise ValueError(f"Unsupported sparse solver backend: {backend}")
    if track_bands and backend != "cpu_mkl_pardiso":
        raise ValueError("Eigenvector-overlap tracking currently requires cpu_mkl_pardiso")
    if track_bands and project_mulliken:
        raise ValueError("Approximate band tracking and physical projections are separate modes")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if environment.get("status") != "valid":
        raise RuntimeError(f"Invalid sparse solver environment: {environment_path}")
    required_inputs = ["hamiltonians_pred.h5", "overlaps.h5", "rlat.dat", "site_positions.dat", "orbital_types.dat"]
    if project_mulliken:
        required_inputs.append("element.dat")
    for name in required_inputs:
        if not (input_dir / name).is_file():
            raise RuntimeError(f"Missing sparse solver input: {input_dir / name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_free_disk_percent = free_disk_percent(output_dir)
    disk_usage = shutil.disk_usage(output_dir)
    if job == "band":
        segments = 3 if band_path == "gamma-k-m-gamma" else 2
        estimated_kpoints = 1 if gamma_only else 1 + segments * max(0, points_per_segment - 1)
    else:
        estimated_kpoints = int(np.prod(kmesh))
    estimated_output_bytes = 10 * 1024**2 + estimated_kpoints * num_bands * (
        1200 if project_mulliken else 400
    )
    estimated_free_disk_percent_after = 100.0 * max(
        0, disk_usage.free - estimated_output_bytes
    ) / disk_usage.total
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
        "estimated_kpoint_count": estimated_kpoints,
        "estimated_output_bytes": estimated_output_bytes,
        "estimated_free_disk_percent_after": estimated_free_disk_percent_after,
    }
    initial_gpu = gpu_status() if backend == "gpu_cudss" else None
    resource_gate["gpu_before"] = initial_gpu
    gpu_unavailable = backend == "gpu_cudss" and (
        initial_gpu is None or initial_gpu["free_bytes"] < GIB
    )
    if (
        initial_free_disk_percent < MIN_FREE_DISK_PERCENT
        or estimated_free_disk_percent_after < MIN_FREE_DISK_PERCENT
        or available_memory < MIN_AVAILABLE_MEMORY_BYTES
        or memory_limit < MIN_SOLVER_MEMORY_BYTES
        or gpu_unavailable
    ):
        reason = (
            "disk_headroom"
            if initial_free_disk_percent < MIN_FREE_DISK_PERCENT
            or estimated_free_disk_percent_after < MIN_FREE_DISK_PERCENT
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
            "band_tracking_requested": track_bands,
        }
        write_json(output_dir / "solver_manifest.json", manifest)
        return manifest
    reciprocal = np.loadtxt(input_dir / "rlat.dat").T
    k_path = None
    k_path_validation = None
    if job == "band":
        k_path, k_path_validation = select_band_path(reciprocal, band_path)
        direct = np.loadtxt(input_dir / "lat.dat").T if (input_dir / "lat.dat").is_file() else None
        write_json(
            output_dir / "k_path_validation.json",
            {
                "status": "valid",
                "direct_lattice_vectors_ang": direct.tolist() if direct is not None else None,
                "reciprocal_lattice_vectors_inv_ang": reciprocal.tolist(),
                "high_symmetry_points_fractional": {
                    label: list(point) for label, point in k_path[:-1]
                },
                **k_path_validation,
            },
        )
    config = {
        "calc_job": job,
        "fermi_level": fermi_level,
        "max_iter": 1000,
        "num_band": num_bands,
        "pardiso_out_of_core": backend == "cpu_mkl_pardiso",
        "solver_backend": backend,
        "track_bands_by_eigenvector_overlap": track_bands,
    }
    if job == "band":
        config.update({
            "which_k": 0,
            "k_data": band_k_data(points_per_segment, gamma_only, k_path),
        })
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
    projection_mapping = None
    projection_groups_path = output_dir / "mulliken_projection_groups.json"
    if project_mulliken:
        projection_mapping = mulliken_projection_groups(input_dir)
        write_json(projection_groups_path, projection_mapping)
    solver_script = (
        REPO_ROOT / "Comparison/scripts/deeph_sparse_calc_gpu.jl"
        if backend == "gpu_cudss"
        else REPO_ROOT / "Comparison/scripts/deeph_sparse_calc_projected.jl"
        if project_mulliken
        else REPO_ROOT / "Comparison/scripts/deeph_sparse_calc_tracked.jl"
        if track_bands
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
    process_threads = SOLVER_THREADS
    env["OMP_NUM_THREADS"] = str(process_threads)
    env["MKL_NUM_THREADS"] = str(process_threads)
    env["OPENBLAS_NUM_THREADS"] = str(process_threads)
    env["MKL_PARDISO_OOC_MAX_CORE_SIZE"] = str(PARDISO_OOC_MAX_CORE_MB)
    env["MKL_PARDISO_OOC_MAX_SWAP_SIZE"] = "0"
    env["MKL_PARDISO_OOC_KEEP_FILE"] = "0"
    if project_mulliken:
        env["DEEPH_MULLIKEN_GROUPS"] = str(projection_groups_path)
        env["DEEPH_PROJECTION_CHUNK_BANDS"] = "16"
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
        "band_tracking_requested": track_bands,
        "mulliken_projection_requested": project_mulliken,
        "fermi_level_eV": fermi_level,
        "neutrality_reference": estimated_neutrality_reference(input_dir, fermi_level),
        "num_bands": num_bands,
        "kmesh": list(kmesh) if job == "dos" else None,
        "dos_broadening_meV": dos_broadening_mev if job == "dos" else None,
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
        "projection_mapping": projection_mapping.get("mapping") if projection_mapping else None,
    }
    if returncode == 0 and job == "band":
        raw_energies = parse_openmx_band(output_dir / "openmx.Band")
        energies = sort_bands_by_energy(raw_energies)
        manifest["band_ordering"] = {
            "method": "ascending_energy_rank_per_k",
            "band_character_continuity_claimed": False,
            "raw_solver_order_preserved_in": str(output_dir / "openmx.Band"),
        }
        tracked_energies = None
        if track_bands:
            tracked_energies, manifest["band_tracking"] = track_bands_by_overlap(output_dir, raw_energies)
            manifest["band_tracking"]["used_for_plot"] = False
            manifest["band_tracking"]["reason"] = (
                "diagnostic only: low-confidence coefficient overlaps must not replace "
                "the conventional ascending-energy spectrum"
            )
        direct = np.loadtxt(input_dir / "lat.dat").T if (input_dir / "lat.dat").is_file() else None
        manifest["k_path"] = {
            "name": f"moire_{band_path}",
            "coordinates": [{"label": label, "k": list(point)} for label, point in k_path],
            "coordinate_system": "reciprocal_lattice_vectors",
            "points_per_segment": points_per_segment,
            "sample_count": len(band_kpoints(points_per_segment, gamma_only, k_path)),
            "direct_lattice_vectors_ang": direct.tolist() if direct is not None else None,
            "reciprocal_lattice_vectors_inv_ang": reciprocal.tolist(),
            "validation": k_path_validation,
        }
        manifest["bands"] = band_points(
            energies,
            points_per_segment,
            reciprocal,
            fermi_level,
            gamma_only,
            k_path,
        )
        if project_mulliken:
            projected, projection_diagnostics = projected_band_data(output_dir, raw_energies)
            for point in manifest["bands"]:
                point.update(projected[int(point["k_index"])][int(point["band_index"])])
            manifest["projection"] = {
                "status": "completed",
                "scientific_status": "s_aware_mulliken_population",
                "diagnostics": projection_diagnostics,
                "mapping_path": str(projection_groups_path),
                "full_space_lowdin_used": False,
                "lowdin_reason": "dense S^(1/2) is unsafe for 117222 orbitals",
            }
        if band_path == "kprime-gamma-k":
            manifest["kprime_comparison"] = kprime_comparison(manifest["bands"])
        manifest["band_representations"] = {
            "raw_arpack_order": {
                "scientific_status": "solver_order_not_a_band_identity",
                "bands": band_points(
                    raw_energies, points_per_segment, reciprocal, fermi_level, gamma_only, k_path
                ),
            },
            "energy_sorted": {
                "scientific_status": "visible_energy_rank_per_k",
                "bands_field": "bands",
            },
        }
        if tracked_energies is not None:
            manifest["band_representations"]["overlap_tracked"] = {
                "scientific_status": "diagnostic_low_confidence_nonorthogonal_coefficient_overlap",
                "bands": band_points(
                    tracked_energies,
                    points_per_segment,
                    reciprocal,
                    fermi_level,
                    gamma_only,
                    k_path,
                ),
            }
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
        if project_mulliken:
            raw_energies = np.loadtxt(output_dir / "egvals.dat", dtype=float).T
            projected_dos, projection_diagnostics = projected_dos_data(
                output_dir,
                raw_energies,
                fermi_level,
                dos_broadening_mev,
                dos_energy_window_mev,
            )
            manifest["projected_dos"] = projected_dos
            manifest["dos_observables"] = projected_dos_observables(projected_dos)
            manifest["projection"] = {
                "status": "completed",
                "scientific_status": "s_aware_mulliken_pdos_on_uniform_2d_mesh",
                "diagnostics": projection_diagnostics,
                "mapping_path": str(projection_groups_path),
                "eigenvectors_persisted": False,
                "full_orbital_spectrum": False,
                "scope_limitation": (
                    "Low-energy 256-state shift-invert subspace; not a full-spectrum DOS."
                ),
            }
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
    parser.add_argument("--track-bands", action="store_true")
    parser.add_argument("--project-mulliken", action="store_true")
    parser.add_argument("--band-path", choices=tuple(BAND_PATH_INDICES), default="gamma-k-m-gamma")
    parser.add_argument(
        "--dos-broadening-mev", type=float, choices=(0.25, 0.5, 1.0, 2.0), default=0.5
    )
    parser.add_argument("--dos-energy-window-mev", type=float, default=100.0)
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
        track_bands=args.track_bands,
        band_path=args.band_path,
        project_mulliken=args.project_mulliken,
        dos_broadening_mev=args.dos_broadening_mev,
        dos_energy_window_mev=args.dos_energy_window_mev,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2 if result["status"] == "resource_blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
