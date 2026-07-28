#!/usr/bin/env python3
"""Normalize a SIESTA FC run into independent Graph2Mat-ready steps."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atom_displacement_utils import (
    BOHR_TO_ANG,
    DATASET_DIR,
    PIPELINE_CONFIG,
    PIPELINE_PATHS,
    Structure,
    compute_max_fc_structures,
    compute_water_geometry_metrics,
    fc_displaced_atom_count,
    ensure_dir,
    parse_fdf_structure,
    write_json,
)
from pipeline_config_utils import render_single_point_fdf


FC_STEPS_DIR_NAME = "FC_steps"


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def geometry_sha256(reference: Structure, positions_ang: list[list[float]]) -> str:
    return canonical_sha256(
        {
            "lattice_vectors_ang": reference.lattice_vectors_ang,
            "atom_species": reference.atom_species,
            "species_labels": reference.species_labels,
            "positions_ang": positions_ang,
        }
    )


def subsampling_semantics(method: str, seed: int) -> dict[str, Any]:
    random_method = method.lower() == "random"
    return {
        "method": method,
        "seed": seed,
        "seed_effective": random_method,
        "seed_role": (
            "algorithmic_subsampling"
            if random_method
            else "recorded_but_ineffective_for_deterministic_subsampling"
        ),
        "independent_replica_claim_allowed": False,
        "experimental_unit": "fc_geometry_family",
    }


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
    canonical_count = compute_max_fc_structures(displaced_atoms, include_reference)
    target_count = canonical_count if target_count is None else int(target_count)
    if target_count < 1:
        raise RuntimeError(f"target_count debe ser positivo: {target_count}")
    if target_count > canonical_count:
        raise ValueError(
            "SIESTA FC cannot generate more than 6N displaced structures "
            f"({canonical_count} including reference={include_reference}) for "
            f"{displaced_atoms} selected atoms; requested {target_count}."
        )

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
    for atom in range(first_atom, last_atom + 1):
        displacement_index = 1
        for direction in (1, 2, 3):
            for sign in (-1, 1):
                matrix_label = f"{system_label}.{atom:05d}-{displacement_index}"
                steps.append(
                    FcStep(
                        index=index,
                        matrix_label=matrix_label,
                        source_matrix_label=matrix_label,
                        amplitude_index=1,
                        atom=atom,
                        direction=direction,
                        sign=sign,
                        displacement_ang=displacement_ang,
                        displacement_input=displacement_input,
                        positions_ang=step_positions(
                            reference,
                            atom,
                            direction,
                            sign,
                            displacement_ang,
                        ),
                    )
                )
                index += 1
                displacement_index += 1
    return steps[:target_count]


def select_spread(items: list[FcStep], count: int) -> list[FcStep]:
    if count >= len(items):
        return list(items)
    used: set[int] = set()
    selected: list[int] = []
    for index in range(count):
        target = min(len(items) - 1, int((index + 0.5) * len(items) / count))
        if target in used:
            target = min(
                (candidate for candidate in range(len(items)) if candidate not in used),
                key=lambda candidate: abs(candidate - target),
            )
        used.add(target)
        selected.append(target)
    return [items[index] for index in sorted(selected)]


def select_fc_steps(
    steps: list[FcStep],
    requested_count: int,
    *,
    method: str,
    seed: int,
) -> list[FcStep]:
    """Subsample one FC run reproducibly.

    ``spread`` is deterministic and preserves coverage across the canonical FC
    order. ``random`` uses a local ``Random(seed)`` and then restores canonical
    ordering, so repeated runs produce the same merged dataset.
    """

    requested_count = int(requested_count)
    if requested_count > len(steps):
        raise ValueError(
            f"Requested {requested_count} structures, but FC generated only {len(steps)}."
        )
    if requested_count <= 0:
        raise ValueError(f"requested_count must be positive: {requested_count}")
    if requested_count == len(steps):
        return list(steps)
    method = method.lower()
    if method == "spread":
        return select_spread(steps, requested_count)
    if method == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(steps, requested_count), key=lambda step: step.index)
    if method in {"first", "prefix"}:
        return steps[:requested_count]
    raise ValueError(f"Unsupported FC subsampling method: {method!r}")


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


def copy_reference_matrices(candidates: list[Path], destination: Path) -> Path | None:
    copied_files: list[Path] = []

    for candidate in candidates:
        if candidate.exists():
            copied = destination / candidate.name
            shutil.copyfile(candidate, copied)
            copied.touch(exist_ok=True)
            copied_files.append(copied)

    # Prefer TSHS as the selected reference, but keep HSX too if available.
    for copied in copied_files:
        if copied.suffix == ".TSHS":
            return copied
    for copied in copied_files:
        if copied.suffix == ".HSX":
            return copied

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


def copy_basis_into(fc_run_dir: Path, output_dir: Path) -> int:
    """Merge basis files from one raw FC run into ``output_dir/basis``."""

    basis_dir = output_dir / "basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in sorted(fc_run_dir.glob("*.ion*")):
        destination = basis_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
            count += 1
    return count


def load_generation_manifest() -> dict[str, Any] | None:
    manifest_path = PIPELINE_PATHS["samples_manifest_path"]
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def normalize_multi_fc_steps(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    """Normalize and merge all configured FC magnitudes into one FC_steps tree."""

    output_dir = args.output_dir
    if args.dry_run:
        output_dir_repr = str(output_dir)
    else:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_repr = str(output_dir)

    subsampling = manifest.get("subsampling") or {}
    method = str(subsampling.get("method", "spread"))
    seed = int(subsampling.get("seed", 0))
    include_reference = bool(manifest.get("force_constants", {}).get("include_reference", True))

    rows: list[dict[str, Any]] = []
    missing_matrices: list[str] = []
    displacement_summaries: list[dict[str, Any]] = []
    basis_count = 0
    global_index = 0

    for run in manifest.get("runs", []):
        run_dir = Path(run["run_dir"])
        if not run_dir.is_absolute():
            run_dir = DATASET_DIR / run_dir
        base_fdf = run_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]
        if not base_fdf.exists():
            raise RuntimeError(f"No existe RUN.fdf para FC run: {base_fdf}")

        reference = parse_fdf_structure(base_fdf)
        system_label = run.get("id") or read_system_label(base_fdf)
        displacement_value = run["displacement"]
        displacement_ang, displacement_unit = parse_displacement(displacement_value)
        first_atom = int(run.get("first_atom", 1))
        last_atom = int(run.get("last_atom") or len(reference.atom_species))
        displaced_atoms = fc_displaced_atom_count(len(reference.atom_species), first_atom, last_atom)
        max_structures = compute_max_fc_structures(displaced_atoms, include_reference)
        requested = int(run.get("requested_structures") or max_structures)
        if requested > max_structures:
            raise ValueError(
                f"Requested {requested} FC structures for {displacement_value}, "
                f"but the FC limit is {max_structures}."
            )

        all_steps = fc_sequence(
            reference,
            system_label,
            first_atom,
            last_atom,
            displacement_ang,
            displacement_value,
            include_reference=include_reference,
            target_count=None,
        )
        selected_steps = select_fc_steps(
            all_steps,
            requested,
            method=method,
            seed=seed + int(run.get("index", 0)),
        )
        displacement_summaries.append(
            {
                "id": system_label,
                "run_dir": str(run_dir),
                "displacement_input": displacement_value,
                "displacement_ang": displacement_ang,
                "displacement_unit": displacement_unit,
                "requested_structures": requested,
                "available_structures": len(all_steps),
                "selected_indices": [step.index for step in selected_steps],
                "selected_geometry_sha256": [
                    geometry_sha256(reference, step.positions_ang)
                    for step in selected_steps
                ],
            }
        )

        if not args.dry_run:
            basis_count += copy_basis_into(run_dir, output_dir)

        for step in selected_steps:
            step_id = f"{global_index:03d}"
            step_dir = output_dir / step_id
            copied_matrix = None
            pseudos: list[str] = []
            if not args.dry_run:
                ensure_dir(step_dir)
                write_step_run_fdf(step_dir, reference, step)
                copied_matrix = copy_reference_matrices(
                    [
                        run_dir / f"{step.matrix_label}.TSHS",
                        run_dir / f"{step.matrix_label}.HSX",
                    ],
                    step_dir,
                )
                if copied_matrix is None:
                    missing_matrices.append(step.matrix_label)
                pseudos = copy_pseudos(run_dir, base_fdf, step_dir)

            metadata = {
                "id": step_id,
                "generation_mode": "siesta_fc_multi_normalized_step",
                "source": "SIESTA MD.TypeOfRun FC",
                "raw_fc_run_dir": str(run_dir),
                "raw_displacement_run_id": system_label,
                "matrix_label": step.matrix_label,
                "source_matrix_label": step.source_matrix_label,
                "source_step_index": step.index,
                "atom": step.atom,
                "direction": step.direction,
                "sign": step.sign,
                "displacement_ang": step.displacement_ang,
                "displacement_input": step.displacement_input,
                "positions_ang": step.positions_ang,
                "geometry_sha256": geometry_sha256(reference, step.positions_ang),
                "matrix_file": str(copied_matrix) if copied_matrix else None,
                "pseudopotentials": pseudos,
                "subsampling": subsampling_semantics(
                    method,
                    seed + int(run.get("index", 0)),
                ),
                "geometry_metrics": compute_water_geometry_metrics(
                    Structure(
                        lattice_vectors_ang=reference.lattice_vectors_ang,
                        species_labels=reference.species_labels,
                        atom_species=reference.atom_species,
                        positions_ang=step.positions_ang,
                    )
                ),
            }
            if not args.dry_run:
                write_json(step_dir / "metadata.json", metadata)
            rows.append(metadata)
            global_index += 1

    if missing_matrices and not args.allow_missing_matrix:
        raise RuntimeError(
            "Faltan Hamiltonianos FC para estos steps: "
            f"{missing_matrices}. Usa --allow-missing-matrix solo si ejecutarás SIESTA despues."
        )

    payload = {
        "ok": not missing_matrices or bool(args.allow_missing_matrix),
        "needs_single_points": bool(missing_matrices),
        "generation_mode": "siesta_fc_multi_normalized_steps",
        "output_dir": output_dir_repr,
        "include_reference": include_reference,
        "subsampling": subsampling_semantics(method, seed),
        "displacements": displacement_summaries,
        "generated_steps": len(rows),
        "basis_files": basis_count,
        "missing_matrices": missing_matrices,
        "samples": rows,
    }
    if not args.dry_run:
        write_json(output_dir / "manifest.json", payload)
    return payload


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
        copied_matrix = copy_reference_matrices(
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
    parser.add_argument(
        "--include-reference",
        action=argparse.BooleanOptionalAction,
        default=bool(force_constants.get("include_reference", True)),
    )
    parser.add_argument(
        "--allow-missing-matrix",
        action="store_true",
        default=bool(force_constants.get("allow_missing_matrix", False)),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    generation_manifest = load_generation_manifest()
    if (
        generation_manifest
        and generation_manifest.get("generation_mode") == "siesta_fc_multi_run"
        and args.fc_run_dir == DATASET_DIR
    ):
        manifest = normalize_multi_fc_steps(args, generation_manifest)
    else:
        manifest = normalize_fc_steps(args)
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        return 0
    print(f"[OK] FC_steps normalizado en {args.output_dir}")
    return 0 if manifest.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
