#!/usr/bin/env python3
"""Normalize a SIESTA FC run into independent Graph2Mat-ready steps."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atom_displacement_utils import (
    BOHR_TO_ANG,
    DATASET_DIR,
    PIPELINE_CONFIG,
    Structure,
    compute_water_geometry_metrics,
    ensure_dir,
    parse_fdf_structure,
    write_json,
)
from pipeline_config_utils import render_single_point_fdf


FC_STEPS_DIR_NAME = "FC_steps"


@dataclass(frozen=True)
class FcStep:
    index: int
    matrix_label: str
    source_matrix_label: str
    amplitude_index: int
    atom: int | None
    direction: int | None
    sign: int | None
    displacement_ang: float
    displacement_input: str
    positions_ang: list[list[float]]


def parse_displacement(value: str) -> tuple[float, str]:
    match = re.fullmatch(r"\s*([-+0-9.Ee]+)\s*([A-Za-z/]+)?\s*", str(value))
    if not match:
        raise RuntimeError(f"No pude interpretar FC.Displacement: {value!r}")
    magnitude = float(match.group(1))
    unit = (match.group(2) or "Ang").lower()
    if unit in {"ang", "angstrom", "angstroms"}:
        return magnitude, "Ang"
    if unit in {"bohr", "bohrs"}:
        return magnitude * BOHR_TO_ANG, "Bohr"
    raise RuntimeError(f"Unidad no soportada en FC.Displacement: {value!r}")


def step_positions(
    reference: Structure,
    atom: int | None,
    direction: int | None,
    sign: int | None,
    displacement_ang: float,
) -> list[list[float]]:
    positions = [list(position) for position in reference.positions_ang]
    if atom is None or direction is None or sign is None:
        return positions
    positions[atom - 1][direction - 1] += sign * displacement_ang
    return positions


def fc_sequence(
    reference: Structure,
    system_label: str,
    first_atom: int,
    last_atom: int,
    displacement_ang: float,
    displacement_input: str,
    include_reference: bool,
    target_count: int | None = None,
) -> list[FcStep]:
    if first_atom < 1 or last_atom < first_atom or last_atom > len(reference.atom_species):
        raise RuntimeError(
            "Rango FC invalido: "
            f"first_atom={first_atom}, last_atom={last_atom}, atoms={len(reference.atom_species)}"
        )

    displaced_atoms = last_atom - first_atom + 1
    canonical_count = (1 if include_reference else 0) + 6 * displaced_atoms
    target_count = canonical_count if target_count is None else int(target_count)
    if target_count < 1:
        raise RuntimeError(f"target_count debe ser positivo: {target_count}")

    steps: list[FcStep] = []
    if include_reference:
        steps.append(
            FcStep(
                index=0,
                matrix_label=f"{system_label}.00000",
                source_matrix_label=f"{system_label}.00000",
                amplitude_index=0,
                atom=None,
                direction=None,
                sign=None,
                displacement_ang=0.0,
                displacement_input="0 Ang",
                positions_ang=step_positions(reference, None, None, None, displacement_ang),
            )
        )

    index = len(steps)
    amplitude_index = 1
    while index < target_count:
        current_displacement_ang = displacement_ang * amplitude_index
        current_displacement_input = (
            displacement_input if amplitude_index == 1 else f"{amplitude_index} * ({displacement_input})"
        )
        for atom in range(first_atom, last_atom + 1):
            displacement_index = 1
            for direction in (1, 2, 3):
                for sign in (-1, 1):
                    source_matrix_label = f"{system_label}.{atom:05d}-{displacement_index}"
                    matrix_label = (
                        source_matrix_label
                        if amplitude_index == 1
                        else f"{system_label}.amp{amplitude_index:02d}.{atom:05d}-{displacement_index}"
                    )
                    steps.append(
                        FcStep(
                            index=index,
                            matrix_label=matrix_label,
                            source_matrix_label=source_matrix_label,
                            amplitude_index=amplitude_index,
                            atom=atom,
                            direction=direction,
                            sign=sign,
                            displacement_ang=current_displacement_ang,
                            displacement_input=current_displacement_input,
                            positions_ang=step_positions(
                                reference,
                                atom,
                                direction,
                                sign,
                                current_displacement_ang,
                            ),
                        )
                    )
                    index += 1
                    displacement_index += 1
                    if index >= target_count:
                        break
                if index >= target_count:
                    break
            if index >= target_count:
                break
        amplitude_index += 1
    return steps


def read_system_label(fdf_path: Path) -> str:
    for line in fdf_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        parts = clean.split()
        if len(parts) >= 2 and parts[0].lower() == "systemlabel":
            return parts[1]
    return "siesta"


def single_point_config() -> dict[str, Any]:
    config = copy.deepcopy(PIPELINE_CONFIG)
    config["structure"].setdefault("force_constants", {})["enabled"] = False
    return config


def write_step_run_fdf(step_dir: Path, reference: Structure, step: FcStep) -> None:
    config = single_point_config()
    content = render_single_point_fdf(
        config,
        positions_ang=step.positions_ang,
        atom_species=reference.atom_species,
        sample_id=step.matrix_label,
    )
    (step_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]).write_text(content, encoding="utf-8")


def copy_first_existing(candidates: list[Path], destination: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, destination / candidate.name)
            return destination / candidate.name
    return None


def copy_pseudos(fc_run_dir: Path, base_fdf: Path, step_dir: Path) -> list[str]:
    copied: list[str] = []
    sources = sorted(fc_run_dir.glob("*.psf")) or sorted(base_fdf.parent.glob("*.psf"))
    if not sources:
        raise RuntimeError(f"No se encontraron pseudopotenciales .psf para {step_dir}")
    for source in sources:
        shutil.copy2(source, step_dir / source.name)
        copied.append(source.name)
    return copied


def copy_basis(fc_run_dir: Path, output_dir: Path) -> int:
    basis_dir = output_dir / "basis"
    if basis_dir.exists():
        shutil.rmtree(basis_dir)
    basis_sources = sorted(fc_run_dir.glob("*.ion*"))
    if not basis_sources:
        return 0
    basis_dir.mkdir(parents=True, exist_ok=True)
    for source in basis_sources:
        shutil.copy2(source, basis_dir / source.name)
    return len(basis_sources)


def normalize_fc_steps(args: argparse.Namespace) -> dict[str, Any]:
    if not args.base_fdf.exists():
        raise RuntimeError(f"No existe RUN.fdf base: {args.base_fdf}")
    if not args.fc_run_dir.exists():
        raise RuntimeError(f"No existe el directorio FC raw: {args.fc_run_dir}")

    reference = parse_fdf_structure(args.base_fdf)
    system_label = args.system_label or read_system_label(args.base_fdf)
    displacement_value = args.fc_displ or PIPELINE_CONFIG["structure"]["force_constants"]["displacement"]
    displacement_ang, displacement_unit = parse_displacement(displacement_value)
    first_atom = args.fc_first or int(PIPELINE_CONFIG["structure"]["force_constants"].get("first_atom", 1))
    last_atom = args.fc_last or int(
        PIPELINE_CONFIG["structure"]["force_constants"].get("last_atom") or len(reference.atom_species)
    )
    steps = fc_sequence(
        reference,
        system_label,
        first_atom,
        last_atom,
        displacement_ang,
        displacement_value,
        include_reference=args.include_reference,
        target_count=args.target_count,
    )

    if args.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "fc_run_dir": str(args.fc_run_dir),
            "output_dir": str(args.output_dir),
            "expected_steps": len(steps),
            "steps": [step.matrix_label for step in steps],
        }

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basis_count = copy_basis(args.fc_run_dir, args.output_dir)

    rows: list[dict[str, Any]] = []
    missing_matrices: list[str] = []
    for step in steps:
        step_dir = args.output_dir / f"{step.index:03d}"
        ensure_dir(step_dir)
        write_step_run_fdf(step_dir, reference, step)
        copied_matrix = copy_first_existing(
            [
                args.fc_run_dir / f"{step.matrix_label}.TSHS",
                args.fc_run_dir / f"{step.matrix_label}.HSX",
            ],
            step_dir,
        )
        if copied_matrix is None:
            missing_matrices.append(step.matrix_label)
        pseudos = copy_pseudos(args.fc_run_dir, args.base_fdf, step_dir)
        metadata = {
            "id": f"{step.index:03d}",
            "generation_mode": "siesta_fc_normalized_step",
            "source": "SIESTA MD.TypeOfRun FC",
            "raw_fc_run_dir": str(args.fc_run_dir),
            "matrix_label": step.matrix_label,
            "source_matrix_label": step.source_matrix_label,
            "amplitude_index": step.amplitude_index,
            "matrix_file": str(copied_matrix) if copied_matrix else None,
            "atom": step.atom,
            "direction": step.direction,
            "sign": step.sign,
            "displacement_ang": step.displacement_ang,
            "displacement_input": step.displacement_input,
            "positions_ang": step.positions_ang,
            "geometry_metrics": compute_water_geometry_metrics(
                Structure(
                    lattice_vectors_ang=reference.lattice_vectors_ang,
                    species_labels=reference.species_labels,
                    atom_species=reference.atom_species,
                    positions_ang=step.positions_ang,
                )
            ),
            "pseudopotentials": pseudos,
        }
        write_json(step_dir / "metadata.json", metadata)
        rows.append(metadata)

    if missing_matrices and not args.allow_missing_matrix:
        raise RuntimeError(
            "Faltan Hamiltonianos FC para estos steps: "
            f"{missing_matrices}. Usa --allow-missing-matrix solo si ejecutarás SIESTA despues."
        )

    fc_outputs = sorted(args.fc_run_dir.glob("*.FC")) + sorted(args.fc_run_dir.glob("FC"))
    manifest = {
        "ok": not missing_matrices or bool(args.allow_missing_matrix),
        "needs_single_points": bool(missing_matrices),
        "generation_mode": "siesta_fc_normalized_steps",
        "raw_fc_run_dir": str(args.fc_run_dir),
        "base_fdf": str(args.base_fdf),
        "output_dir": str(args.output_dir),
        "system_label": system_label,
        "fc_first": first_atom,
        "fc_last": last_atom,
        "include_reference": args.include_reference,
        "displacement_input": displacement_value,
        "displacement_ang": displacement_ang,
        "displacement_unit": displacement_unit,
        "expected_steps": len(steps),
        "target_count": args.target_count,
        "generated_steps": len(rows),
        "basis_files": basis_count,
        "fc_outputs": [str(path) for path in fc_outputs],
        "missing_matrices": missing_matrices,
        "samples": rows,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    force_constants = PIPELINE_CONFIG["structure"]["force_constants"]
    parser = argparse.ArgumentParser(
        description="Create FC_steps/000/RUN.fdf ... from a SIESTA MD.TypeOfRun FC run."
    )
    parser.add_argument("--base-fdf", type=Path, default=DATASET_DIR / PIPELINE_CONFIG["paths"]["run_fdf_name"])
    parser.add_argument("--fc-run-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR / FC_STEPS_DIR_NAME)
    parser.add_argument("--system-label", default=None)
    parser.add_argument("--fc-displ", default=force_constants["displacement"])
    parser.add_argument("--fc-first", type=int, default=int(force_constants.get("first_atom", 1)))
    parser.add_argument("--fc-last", type=int, default=force_constants.get("last_atom"))
    parser.add_argument("--target-count", type=int, default=force_constants.get("target_count"))
    parser.add_argument("--include-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-missing-matrix",
        action="store_true",
        default=bool(force_constants.get("allow_missing_matrix", False)),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = normalize_fc_steps(args)
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        return 0
    print(f"[OK] FC_steps normalizado en {args.output_dir}")
    return 0 if manifest.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
