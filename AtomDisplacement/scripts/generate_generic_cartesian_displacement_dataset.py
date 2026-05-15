#!/usr/bin/env python3
"""Generate material-aware Cartesian atom-displacement samples.

This generator is intentionally separate from the historical SIESTA FC/H2O
flow. It starts from a validated material bundle and writes explicit +/-x/y/z
single-point FDF inputs for the selected atoms.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ATOM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ATOM_ROOT.parent
SHARED_DIR = REPO_ROOT / "shared"
for candidate in (ATOM_ROOT / "scripts", SHARED_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from fdf_materialization import extract_bundle_structure, materialize_sample_fdf  # noqa: E402
from material_bundle import BASIS_EXTENSIONS, ValidatedMaterialBundle, file_sha256  # noqa: E402
from material_presets import resolve_material_bundle  # noqa: E402
from pipeline_config_utils import load_pipeline_config, paths  # noqa: E402


AXES: tuple[tuple[str, int], ...] = (("x", 0), ("y", 1), ("z", 2))
SIGNS: tuple[int, ...] = (1, -1)
GENERATION_MODE = "generic_cartesian_displacement"
METHOD_ID = "siesta_fc_cartesian"


class GenericCartesianDisplacementError(RuntimeError):
    """Raised when the generic Cartesian recipe cannot be generated safely."""


@dataclass(frozen=True)
class GenericCartesianSettings:
    recipe: str
    amplitude_ang: float
    selected_species: set[str] | None
    include_base: bool
    overwrite: bool


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _parse_amplitude_ang(value: Any) -> float:
    if isinstance(value, (int, float)):
        amplitude = float(value)
    else:
        parts = str(value).strip().split()
        if not parts:
            raise GenericCartesianDisplacementError("atomic_displacement.amplitude_ang cannot be empty.")
        try:
            amplitude = float(parts[0])
        except ValueError as exc:
            raise GenericCartesianDisplacementError(
                f"atomic_displacement.amplitude_ang must be numeric in Ang: {value!r}"
            ) from exc
        if len(parts) > 1 and parts[1].lower() not in {"ang", "angstrom", "angstroms"}:
            raise GenericCartesianDisplacementError(
                "atomic_displacement.amplitude_ang only supports Ang units."
            )
    if amplitude <= 0:
        raise GenericCartesianDisplacementError(
            "atomic_displacement.amplitude_ang must be positive."
        )
    return amplitude


def _parse_selected_species(value: Any) -> set[str] | None:
    if value in (None, "", "all"):
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value]
    else:
        raise GenericCartesianDisplacementError(
            "atomic_displacement.selected_species must be null, a string, or a list."
        )
    selected = {item for item in items if item}
    if not selected:
        raise GenericCartesianDisplacementError(
            "atomic_displacement.selected_species cannot be empty; use null to select all species."
        )
    return selected


def _recipe_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("atomic_displacement")
    if raw is None:
        raw = config.get("structure", {}).get("atomic_displacement")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise GenericCartesianDisplacementError("atomic_displacement config must be a mapping.")
    return raw


def generic_cartesian_settings(config: dict[str, Any]) -> GenericCartesianSettings:
    raw = _recipe_config(config)
    recipe = str(raw.get("recipe", "generic_cartesian")).strip()
    if recipe != "generic_cartesian":
        raise GenericCartesianDisplacementError(
            "generate_generic_cartesian_displacement_dataset.py only supports "
            "atomic_displacement.recipe='generic_cartesian'. H2O-specific "
            "bond/angle or SIESTA FC recipes remain in their existing generators."
        )
    return GenericCartesianSettings(
        recipe=recipe,
        amplitude_ang=_parse_amplitude_ang(raw.get("amplitude_ang", 0.03)),
        selected_species=_parse_selected_species(raw.get("selected_species")),
        include_base=bool(raw.get("include_base", True)),
        overwrite=bool(raw.get("overwrite", False)),
    )


def _sample_id(config: dict[str, Any], index: int) -> str:
    template = str(config.get("generation", {}).get("sample_id_format", "sample_{index:04d}"))
    try:
        sample_id = template.format(index=index)
    except (KeyError, IndexError, ValueError) as exc:
        raise GenericCartesianDisplacementError(
            f"Invalid generation.sample_id_format for generic Cartesian samples: {template!r}"
        ) from exc
    if not sample_id or "/" in sample_id or "\\" in sample_id or sample_id in {".", ".."}:
        raise GenericCartesianDisplacementError(
            f"generation.sample_id_format produced an unsafe sample id: {sample_id!r}"
        )
    return sample_id


def _prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists():
        existing = list(output_root.iterdir())
        if existing and not overwrite:
            raise GenericCartesianDisplacementError(
                f"Output sample directory already exists and is not empty: {output_root}. "
                "Set atomic_displacement.overwrite=true or pass --overwrite to regenerate it."
            )
        if overwrite:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _copy_material_inputs(validated: ValidatedMaterialBundle, sample_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    for label, pseudo_path in sorted(validated.pseudopotentials.items()):
        target = sample_dir / pseudo_path.name
        shutil.copy2(pseudo_path, target)
        copied[label] = target.name
    return copied


def _copy_basis_files(validated: ValidatedMaterialBundle, output_root: Path) -> dict[str, str]:
    if validated.bundle.basis_dir is None:
        return {}
    target_dir = output_root / "basis"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for path in sorted(item for item in validated.bundle.basis_dir.iterdir() if item.is_file()):
        if any(path.name.endswith(extension) for extension in BASIS_EXTENSIONS):
            target = target_dir / path.name
            shutil.copy2(path, target)
            copied[path.name] = file_sha256(target)
    return copied


def _species_by_index(structure: Any) -> dict[int, str]:
    return {int(species.index): str(species.label) for species in structure.species}


def _selected_atom_records(structure: Any, selected_species: set[str] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    species_by_index = _species_by_index(structure)
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, atom in enumerate(structure.atoms, start=1):
        label = species_by_index[int(atom.species_index)]
        record = {
            "atom_index": index,
            "atom_index_zero_based": index - 1,
            "species": label,
            "species_index": int(atom.species_index),
        }
        if selected_species is None or label in selected_species:
            selected.append(record)
        else:
            skipped.append(record)
    if not selected:
        expected = ", ".join(sorted(selected_species or []))
        raise GenericCartesianDisplacementError(
            f"atomic_displacement.selected_species matched no atoms: {expected}"
        )
    return selected, skipped


def _with_displacement(
    positions: list[tuple[float, float, float]],
    *,
    atom_index_zero_based: int,
    axis_index: int,
    delta_ang: float,
) -> list[list[float]]:
    updated = [list(position) for position in positions]
    updated[atom_index_zero_based][axis_index] += delta_ang
    return updated


def _base_metadata(
    *,
    sample_id: str,
    material_label: str,
    settings: GenericCartesianSettings,
    materialized_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "id": sample_id,
        "sample_id": sample_id,
        "generation_mode": GENERATION_MODE,
        "generation_method": "generic_cartesian",
        "method": METHOD_ID,
        "recipe": settings.recipe,
        "material_label": material_label,
        "is_reference": True,
        "atom_index": None,
        "atom_index_zero_based": None,
        "species": None,
        "axis": None,
        "axis_index": None,
        "sign": None,
        "sign_label": None,
        "amplitude_ang": 0.0,
        "displacement_ang": [0.0, 0.0, 0.0],
        "split_group_id": f"{GENERATION_MODE}:{material_label}:reference",
    }
    metadata.update(materialized_metadata)
    return metadata


def _displacement_metadata(
    *,
    sample_id: str,
    material_label: str,
    settings: GenericCartesianSettings,
    atom_record: dict[str, Any],
    axis: str,
    axis_index: int,
    sign: int,
    materialized_metadata: dict[str, Any],
) -> dict[str, Any]:
    displacement = [0.0, 0.0, 0.0]
    displacement[axis_index] = sign * settings.amplitude_ang
    metadata = {
        "id": sample_id,
        "sample_id": sample_id,
        "generation_mode": GENERATION_MODE,
        "generation_method": "generic_cartesian",
        "method": METHOD_ID,
        "recipe": settings.recipe,
        "material_label": material_label,
        "is_reference": False,
        "atom_index": atom_record["atom_index"],
        "atom_index_zero_based": atom_record["atom_index_zero_based"],
        "species": atom_record["species"],
        "species_index": atom_record["species_index"],
        "axis": axis,
        "axis_index": axis_index,
        "sign": sign,
        "sign_label": "+" if sign > 0 else "-",
        "amplitude_ang": settings.amplitude_ang,
        "displacement_ang": displacement,
        "split_group_id": (
            f"{GENERATION_MODE}:{material_label}:atom_{atom_record['atom_index']:04d}"
        ),
    }
    metadata.update(materialized_metadata)
    return metadata


def _write_sample(
    *,
    sample_dir: Path,
    sample_id: str,
    positions_ang: list[list[float]] | list[tuple[float, float, float]],
    validated: ValidatedMaterialBundle,
    structure: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=False)
    copied_pseudos = _copy_material_inputs(validated, sample_dir)
    materialized = materialize_sample_fdf(
        validated.bundle.fdf,
        sample_dir / "RUN.fdf",
        positions_ang=positions_ang,
        atom_species=structure.atom_species,
        lattice_vectors_ang=structure.lattice_vectors_ang,
        system_label=sample_id,
        system_name=f"{validated.bundle.label} {sample_id}",
        structure_type=validated.bundle.structure_type,
    )
    materialized_metadata = dict(materialized.metadata)
    materialized_metadata["structure_species"] = materialized_metadata.pop("species", [])
    metadata = {**materialized_metadata, **metadata}
    metadata["pseudopotentials_copied"] = copied_pseudos
    _write_json(sample_dir / "metadata.json", metadata)
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "sample_dir": str(sample_dir),
        "run_fdf": str(sample_dir / "RUN.fdf"),
        "metadata_path": str(sample_dir / "metadata.json"),
        "is_reference": metadata["is_reference"],
        "atom_index": metadata["atom_index"],
        "atom_index_zero_based": metadata["atom_index_zero_based"],
        "species": metadata["species"],
        "axis": metadata["axis"],
        "axis_index": metadata["axis_index"],
        "sign": metadata["sign"],
        "sign_label": metadata["sign_label"],
        "amplitude_ang": metadata["amplitude_ang"],
        "displacement_ang": metadata["displacement_ang"],
        "split_group_id": metadata["split_group_id"],
        "materialized_fdf_sha256": metadata["materialized_fdf_sha256"],
    }


def generate_dataset(
    config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    base_dir: str | Path = REPO_ROOT,
    overwrite: bool | None = None,
) -> dict[str, Any]:
    """Generate generic +/- Cartesian displacement samples from a material FDF."""

    settings = generic_cartesian_settings(config)
    if overwrite is not None:
        settings = GenericCartesianSettings(
            recipe=settings.recipe,
            amplitude_ang=settings.amplitude_ang,
            selected_species=settings.selected_species,
            include_base=settings.include_base,
            overwrite=bool(overwrite),
        )
    if output_dir is None:
        pipeline_paths = paths(config)
        output_root = pipeline_paths["samples_dir"]
        manifest_path = pipeline_paths["samples_manifest_path"]
    else:
        output_root = Path(output_dir)
        manifest_path = output_root.parent / "samples_manifest.json"

    resolved = resolve_material_bundle(config, base_dir=base_dir)
    validated = resolved.validated
    structure = extract_bundle_structure(validated)
    selected_atoms, skipped_atoms = _selected_atom_records(structure, settings.selected_species)

    _prepare_output_root(output_root, overwrite=settings.overwrite)
    copied_basis_hashes = _copy_basis_files(validated, output_root)

    material_label = validated.bundle.label
    sample_records: list[dict[str, Any]] = []
    sample_index = 0
    used_ids: set[str] = set()

    def next_sample_id() -> str:
        nonlocal sample_index
        sample_id = _sample_id(config, sample_index)
        sample_index += 1
        if sample_id in used_ids:
            raise GenericCartesianDisplacementError(f"Duplicate generated sample id: {sample_id}")
        used_ids.add(sample_id)
        return sample_id

    if settings.include_base:
        sample_id = next_sample_id()
        materialized_preview = {
            "structure_type": validated.bundle.structure_type,
        }
        metadata = _base_metadata(
            sample_id=sample_id,
            material_label=material_label,
            settings=settings,
            materialized_metadata=materialized_preview,
        )
        sample_records.append(
            _write_sample(
                sample_dir=output_root / sample_id,
                sample_id=sample_id,
                positions_ang=structure.positions_ang,
                validated=validated,
                structure=structure,
                metadata=metadata,
            )
        )

    for atom_record in selected_atoms:
        for axis, axis_index in AXES:
            for sign in SIGNS:
                sample_id = next_sample_id()
                delta = sign * settings.amplitude_ang
                positions = _with_displacement(
                    structure.positions_ang,
                    atom_index_zero_based=atom_record["atom_index_zero_based"],
                    axis_index=axis_index,
                    delta_ang=delta,
                )
                metadata = _displacement_metadata(
                    sample_id=sample_id,
                    material_label=material_label,
                    settings=settings,
                    atom_record=atom_record,
                    axis=axis,
                    axis_index=axis_index,
                    sign=sign,
                    materialized_metadata={"structure_type": validated.bundle.structure_type},
                )
                sample_records.append(
                    _write_sample(
                        sample_dir=output_root / sample_id,
                        sample_id=sample_id,
                        positions_ang=positions,
                        validated=validated,
                        structure=structure,
                        metadata=metadata,
                    )
                )

    manifest = {
        "generation_mode": GENERATION_MODE,
        "generation_method": "generic_cartesian",
        "method": METHOD_ID,
        "recipe": settings.recipe,
        "material": resolved.to_manifest_dict(),
        "reference_source": str(validated.bundle.fdf),
        "sample_root": str(output_root),
        "requested_structures": len(sample_records),
        "generated_structures": len(sample_records),
        "include_base": settings.include_base,
        "amplitude_ang": settings.amplitude_ang,
        "selected_species": sorted(settings.selected_species) if settings.selected_species else None,
        "selected_atoms": selected_atoms,
        "skipped_atoms": skipped_atoms,
        "axis_order": [axis for axis, _axis_index in AXES],
        "sign_order": list(SIGNS),
        "basis_file_sha256": copied_basis_hashes,
        "samples": sample_records,
    }
    _write_json(manifest_path, manifest)
    _write_json(output_root / "dataset_manifest.json", manifest)
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate generic Cartesian AtomicDisplacement samples from a material bundle."
    )
    parser.add_argument("--config", type=Path, default=None, help="Pipeline YAML config path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output sample directory.")
    parser.add_argument("--material-base-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--amplitude-ang", type=float, default=None)
    parser.add_argument("--selected-species", nargs="*", default=None)
    parser.add_argument("--include-base", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    config = load_pipeline_config(args.config)
    recipe = dict(_recipe_config(config))
    if args.amplitude_ang is not None:
        recipe["amplitude_ang"] = args.amplitude_ang
    if args.selected_species is not None:
        recipe["selected_species"] = args.selected_species
    if args.include_base is not None:
        recipe["include_base"] = args.include_base
    if args.overwrite:
        recipe["overwrite"] = True
    config["atomic_displacement"] = recipe

    manifest = generate_dataset(
        config,
        output_dir=args.output_dir,
        base_dir=args.material_base_dir,
        overwrite=args.overwrite or None,
    )
    print(
        "[OK] Generic Cartesian AtomicDisplacement samples generated: "
        f"{manifest['generated_structures']} in {manifest['sample_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
