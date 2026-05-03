#!/usr/bin/env python3
"""Shared utilities for the H2O atom-displacement dataset pipeline."""

from __future__ import annotations

import json
import math
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline_config_utils import (
    command,
    load_pipeline_config,
    paths,
    render_single_point_fdf,
)

BOHR_TO_ANG = 0.529177210903
RY_TO_EV = 13.605693009
ATDIS_STEPS_DIR_NAME = "AtDis_steps"
FC_STEPS_DIR_NAME = "FC_steps"
FC_RUNS_DIR_NAME = "FC_runs"
ATOM_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_CONFIG = load_pipeline_config()
PIPELINE_PATHS = paths(PIPELINE_CONFIG)
DEFAULT_VENV_ACTIVATE = PIPELINE_PATHS["venv_activate"]

BASE_DIR = PIPELINE_PATHS["base_dir"]
RELAXED_DIR = PIPELINE_PATHS["relaxed_dir"]
DATASET_DIR = PIPELINE_PATHS["dataset_dir"]
SAMPLES_DIR = PIPELINE_PATHS["samples_dir"]
COLLECTED_DIR = PIPELINE_PATHS["collected_dir"]
TRAINING_DIR = PIPELINE_PATHS["training_dir"]


@dataclass
class Structure:
    lattice_vectors_ang: list[list[float]]
    species_labels: dict[int, tuple[int, str]]
    atom_species: list[int]
    positions_ang: list[list[float]]

    @property
    def symbols(self) -> list[str]:
        return [self.species_labels[index][1] for index in self.atom_species]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "lattice_vectors_ang": self.lattice_vectors_ang,
            "species_labels": {
                str(key): {"atomic_number": value[0], "symbol": value[1]}
                for key, value in self.species_labels.items()
            },
            "atom_species": self.atom_species,
            "positions_ang": self.positions_ang,
            "symbols": self.symbols,
        }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_max_fc_structures(n_atoms: int, include_reference: bool = True) -> int:
    """Return the finite-difference FC structure limit for ``n_atoms``.

    SIESTA FC displaces every selected atom along the three Cartesian axes with
    both signs. This gives ``2 * 3 * n_atoms == 6N`` displaced structures. Some
    workflows also keep the undisplaced reference geometry, represented by the
    optional ``+1``.
    """

    n_atoms = int(n_atoms)
    if n_atoms <= 0:
        raise ValueError(f"n_atoms must be positive for FC generation: {n_atoms}")
    return 6 * n_atoms + (1 if include_reference else 0)


def fc_displaced_atom_count(
    total_atoms: int,
    first_atom: int = 1,
    last_atom: int | None = None,
) -> int:
    """Validate an FC atom range and return the number of displaced atoms."""

    total_atoms = int(total_atoms)
    first_atom = int(first_atom)
    last_atom = total_atoms if last_atom is None else int(last_atom)
    if first_atom < 1 or last_atom < first_atom or last_atom > total_atoms:
        raise ValueError(
            "Invalid FC atom range: "
            f"FC.First={first_atom}, FC.Last={last_atom}, NumberOfAtoms={total_atoms}"
        )
    return last_atom - first_atom + 1


def require_command(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"No se encontro '{command_name}' en PATH. Activa el entorno correcto."
        )


def run_command(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"El comando fallo con codigo {result.returncode}: {' '.join(cmd)}"
        )


def run_command_printing(cmd: list[str], cwd: Path) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    run_command(cmd, cwd)


def run_command_in_venv(
    cmd: list[str],
    cwd: Path,
    activate_path: Path = DEFAULT_VENV_ACTIVATE,
) -> None:
    if not activate_path.exists():
        raise RuntimeError(
            f"No se encontro el script de activacion esperado en {activate_path}"
        )

    quoted_cmd = " ".join(shlex_quote(token) for token in cmd)
    shell = command(PIPELINE_CONFIG, "shell")
    bash_cmd = f"source {shlex.quote(str(activate_path))} && {quoted_cmd}"
    print(f"\n[RUN] {shell} -lc \"{bash_cmd}\"")
    result = subprocess.run([shell, "-lc", bash_cmd], cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"El comando fallo con codigo {result.returncode}: {' '.join(cmd)}"
        )


def run_siesta_in_dir(
    cwd: Path,
    run_out_path: Path,
    activate_path: Path = DEFAULT_VENV_ACTIVATE,
) -> None:
    if not activate_path.exists():
        raise RuntimeError(
            f"No se encontro el script de activacion esperado en {activate_path}"
        )

    shell = command(PIPELINE_CONFIG, "shell")
    siesta = command(PIPELINE_CONFIG, "siesta")
    run_fdf_name = PIPELINE_CONFIG["paths"]["run_fdf_name"]
    bash_cmd = (
        f"source {shlex.quote(str(activate_path))} "
        f"&& {shlex.quote(siesta)} < {shlex.quote(str(run_fdf_name))}"
    )
    with run_out_path.open("w", encoding="utf-8") as run_out:
        process = subprocess.Popen(
            [shell, "-lc", bash_cmd],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            run_out.write(line)

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"siesta termino con codigo {return_code} en {cwd}")


def copy_pseudopotentials(src_dir: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    psf_files = sorted(src_dir.glob("*.psf"))
    if not psf_files:
        raise RuntimeError(f"No se encontraron pseudopotenciales .psf en {src_dir}")

    for psf in psf_files:
        shutil.copy2(psf, dst_dir / psf.name)


def _read_named_block(lines: list[str], block_name: str) -> list[str]:
    start = None
    lower_name = block_name.lower()
    for index, raw_line in enumerate(lines):
        if raw_line.strip().lower() == f"%block {lower_name}":
            start = index + 1
            break

    if start is None:
        raise RuntimeError(f"No se encontro el bloque '{block_name}'")

    block: list[str] = []
    end_marker = f"%endblock {lower_name}"
    for raw_line in lines[start:]:
        if raw_line.strip().lower() == end_marker:
            return block
        block.append(raw_line)

    raise RuntimeError(f"El bloque '{block_name}' no tiene cierre")


def parse_fdf_structure(fdf_path: Path) -> Structure:
    lines = fdf_path.read_text(encoding="utf-8").splitlines()
    species_block = _read_named_block(lines, "ChemicalSpeciesLabel")
    lattice_block = _read_named_block(lines, "LatticeVectors")
    coords_block = _read_named_block(lines, "AtomicCoordinatesAndAtomicSpecies")

    lattice_constant = 1.0
    for line in lines:
        clean = line.split("#", 1)[0].strip()
        if clean.lower().startswith("latticeconstant"):
            parts = clean.split()
            lattice_constant = float(parts[1])
            break

    species_labels: dict[int, tuple[int, str]] = {}
    for line in species_block:
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        index_str, atomic_number_str, symbol = clean.split()[:3]
        species_labels[int(index_str)] = (int(atomic_number_str), symbol)

    lattice_vectors_ang: list[list[float]] = []
    for line in lattice_block:
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        vector = [float(value) * lattice_constant for value in clean.split()[:3]]
        lattice_vectors_ang.append(vector)

    atom_species: list[int] = []
    positions_ang: list[list[float]] = []
    for line in coords_block:
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        parts = clean.split()
        positions_ang.append([float(value) for value in parts[:3]])
        atom_species.append(int(parts[3]))

    return Structure(
        lattice_vectors_ang=lattice_vectors_ang,
        species_labels=species_labels,
        atom_species=atom_species,
        positions_ang=positions_ang,
    )


def parse_xv_structure(xv_path: Path, species_labels: dict[int, tuple[int, str]]) -> Structure:
    lines = [line.strip() for line in xv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lattice_vectors_ang = [
        [float(value) * BOHR_TO_ANG for value in lines[index].split()[:3]]
        for index in range(3)
    ]
    n_atoms = int(lines[3].split()[0])

    atom_species: list[int] = []
    positions_ang: list[list[float]] = []
    for raw_line in lines[4 : 4 + n_atoms]:
        parts = raw_line.split()
        atom_species.append(int(parts[0]))
        positions_ang.append([float(value) * BOHR_TO_ANG for value in parts[2:5]])

    return Structure(
        lattice_vectors_ang=lattice_vectors_ang,
        species_labels=species_labels,
        atom_species=atom_species,
        positions_ang=positions_ang,
    )


def format_single_point_fdf(structure: Structure, system_label: str, system_name: str) -> str:
    return render_single_point_fdf(
        PIPELINE_CONFIG,
        positions_ang=structure.positions_ang,
        atom_species=structure.atom_species,
        sample_id=system_label,
    )


def write_single_point_fdf(path: Path, structure: Structure, sample_id: str) -> None:
    content = format_single_point_fdf(
        structure=structure,
        system_label=sample_id,
        system_name=f"H2O {sample_id}",
    )
    path.write_text(content, encoding="utf-8")


def load_reference_structure() -> tuple[Structure, str]:
    base_structure = parse_fdf_structure(PIPELINE_PATHS["base_run_fdf_path"])
    xv_files = sorted(RELAXED_DIR.glob("*.XV"))
    if xv_files:
        return parse_xv_structure(xv_files[0], base_structure.species_labels), str(xv_files[0])
    return base_structure, str(PIPELINE_PATHS["base_run_fdf_path"])


def distance(point_a: list[float], point_b: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))


def angle_degrees(point_a: list[float], point_b: list[float], point_c: list[float]) -> float:
    ba = [a - b for a, b in zip(point_a, point_b)]
    bc = [c - b for c, b in zip(point_c, point_b)]
    dot = sum(a * c for a, c in zip(ba, bc))
    norm_ba = math.sqrt(sum(a * a for a in ba))
    norm_bc = math.sqrt(sum(c * c for c in bc))
    cosine = dot / (norm_ba * norm_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def compute_water_geometry_metrics(structure: Structure) -> dict[str, float]:
    symbols = structure.symbols
    oxygen_indices = [index for index, symbol in enumerate(symbols) if symbol == "O"]
    hydrogen_indices = [index for index, symbol in enumerate(symbols) if symbol == "H"]
    if len(oxygen_indices) != 1 or len(hydrogen_indices) != 2:
        raise RuntimeError("La estructura de referencia no corresponde a una molecula H2O")

    oxygen = oxygen_indices[0]
    h1, h2 = hydrogen_indices
    positions = structure.positions_ang
    oh_1 = distance(positions[oxygen], positions[h1])
    oh_2 = distance(positions[oxygen], positions[h2])
    hh = distance(positions[h1], positions[h2])
    hoh_angle = angle_degrees(positions[h1], positions[oxygen], positions[h2])
    return {
        "oh_1_ang": oh_1,
        "oh_2_ang": oh_2,
        "hh_ang": hh,
        "hoh_angle_deg": hoh_angle,
    }


def structure_with_positions(structure: Structure, positions_ang: list[list[float]]) -> Structure:
    return Structure(
        lattice_vectors_ang=structure.lattice_vectors_ang,
        species_labels=structure.species_labels,
        atom_species=structure.atom_species,
        positions_ang=positions_ang,
    )


def parse_fa_file(fa_path: Path) -> list[list[float]]:
    lines = [line.strip() for line in fa_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_atoms = int(lines[0].split()[0])
    forces = []
    for raw_line in lines[1 : 1 + n_atoms]:
        parts = raw_line.split()
        forces.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return forces


def parse_total_energy_ev(sample_dir: Path) -> float | None:
    outvars_candidates = sorted(sample_dir.glob("OUTVARS*.yml")) + sorted(sample_dir.glob("*.yml"))
    for outvars_path in outvars_candidates:
        text = outvars_path.read_text(encoding="utf-8")
        match = re.search(r"^\s*Etot:\s*([-+0-9.Ee]+)", text, re.MULTILINE)
        if match:
            return float(match.group(1)) * RY_TO_EV

    run_out = sample_dir / "RUN.out"
    if not run_out.exists():
        return None

    text = run_out.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"siesta:\s+Total\s*=\s*([-+0-9.]+)", text)
    if match:
        return float(match.group(1))
    return None


def sample_run_status(sample_dir: Path) -> dict[str, Any]:
    run_out = sample_dir / "RUN.out"
    if not run_out.exists():
        return {"job_completed": False, "scf_converged": False}
    run_fdf = sample_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]
    if run_fdf.exists() and run_out.stat().st_mtime < run_fdf.stat().st_mtime:
        return {"job_completed": False, "scf_converged": False}

    text = run_out.read_text(encoding="utf-8", errors="ignore")
    return {
        "job_completed": "Job completed" in text,
        "scf_converged": "SCF cycle converged" in text,
    }


def generated_sample_dirs() -> list[Path]:
    configured_sample_dirs = sorted(
        path
        for path in SAMPLES_DIR.iterdir()
        if path.is_dir() and (path / PIPELINE_CONFIG["paths"]["run_fdf_name"]).exists()
    ) if SAMPLES_DIR.exists() else []
    if configured_sample_dirs:
        return sorted(configured_sample_dirs, key=lambda path: path.name)

    fc_steps_dir = DATASET_DIR / FC_STEPS_DIR_NAME
    if fc_steps_dir.exists():
        return sorted(
            (
                path
                for path in fc_steps_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            key=lambda path: int(path.name),
        )

    manifest_path = PIPELINE_PATHS["samples_manifest_path"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("generation_mode") == "siesta_fc_multi_run":
            run_dirs = [
                Path(run["run_dir"])
                for run in manifest.get("runs", [])
                if isinstance(run, dict) and run.get("run_dir")
            ]
            run_dirs = [
                path if path.is_absolute() else DATASET_DIR / path
                for path in run_dirs
            ]
            existing_runs = [
                path
                for path in run_dirs
                if path.is_dir() and (path / PIPELINE_CONFIG["paths"]["run_fdf_name"]).exists()
            ]
            if existing_runs:
                return sorted(existing_runs, key=lambda path: path.name)
        if manifest.get("generation_mode") == "siesta_fc_single_run":
            atdis_steps_dir = DATASET_DIR / ATDIS_STEPS_DIR_NAME
            if atdis_steps_dir.exists():
                return sorted(
                    (
                        path
                        for path in atdis_steps_dir.iterdir()
                        if path.is_dir() and path.name.isdigit()
                    ),
                    key=lambda path: int(path.name),
                )
            return [DATASET_DIR]
        sample_ids = [
            sample["id"]
            for sample in manifest.get("samples", [])
            if isinstance(sample, dict) and sample.get("id")
        ]
        if sample_ids:
            return [
                SAMPLES_DIR / sample_id
                for sample_id in sample_ids
                if (SAMPLES_DIR / sample_id).is_dir()
            ]
    sample_dirs = sorted(path for path in SAMPLES_DIR.glob("sample_*") if path.is_dir())
    if sample_dirs:
        return sample_dirs
    return sorted(
        path
        for path in SAMPLES_DIR.iterdir()
        if path.is_dir() and (path / PIPELINE_CONFIG["paths"]["run_fdf_name"]).exists()
    )


def completed_sample_dirs() -> list[Path]:
    completed = []
    for sample_dir in generated_sample_dirs():
        hsx_path = find_first_output(sample_dir, ".HSX")
        tshs_path = find_first_output(sample_dir, ".TSHS")
        run_fdf = sample_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]
        if run_fdf.exists() and tshs_path:
            completed.append(sample_dir)
            continue
        status = sample_run_status(sample_dir)
        if status["job_completed"] and status["scf_converged"] and (hsx_path or tshs_path):
            completed.append(sample_dir)
    return completed


def relaxed_basis_files() -> list[Path]:
    basis_files = sorted(RELAXED_DIR.glob("*.ion.xml"))
    if not basis_files:
        raise RuntimeError(
            f"No se encontraron ficheros .ion.xml en {RELAXED_DIR}. "
            "Ejecuta primero la relajacion de referencia."
        )
    return basis_files


def resolve_ckpt_rel_path(training_dir: Path, default_rel_path: str) -> str:
    from pipeline_config_utils import resolve_checkpoint

    return resolve_checkpoint(PIPELINE_CONFIG)


def find_first_output(sample_dir: Path, suffix: str) -> Path | None:
    matches = sorted(sample_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def shlex_quote(value: str) -> str:
    return shlex.quote(value)
