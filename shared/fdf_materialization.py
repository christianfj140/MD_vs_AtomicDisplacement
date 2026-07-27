"""Minimal SIESTA FDF extraction and per-sample materialization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from material_bundle import (
    MaterialSpecies,
    ValidatedMaterialBundle,
    extract_chemical_species,
    file_sha256,
    read_fdf_block,
)


SUPPORTED_COORDINATE_FORMATS = {"ang", "angstrom", "angstroms"}
DEFAULT_REQUIRED_OUTPUT_FLAGS = {
    "SaveHS": "true",
    "Save.HS": "T",
    "TS.HS.Save": "T",
    "TS.DE.Save": "T",
    "XML.Write": "T",
    "Write.OrbitalIndex": "T",
}
BOHR_TO_ANG = 0.529177210903


class FdfMaterializationError(RuntimeError):
    """Raised when an FDF cannot be safely materialized."""


@dataclass(frozen=True)
class FdfAtom:
    position_ang: tuple[float, float, float]
    species_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_ang": list(self.position_ang),
            "species_index": self.species_index,
        }


@dataclass(frozen=True)
class FdfStructure:
    fdf_path: Path
    species: list[MaterialSpecies]
    atoms: list[FdfAtom]
    lattice_vectors_ang: list[tuple[float, float, float]]
    coordinate_format: str
    number_of_atoms_declared: int | None
    structure_type: str | None = None

    @property
    def atom_count(self) -> int:
        return len(self.atoms)

    @property
    def atom_species(self) -> list[int]:
        return [atom.species_index for atom in self.atoms]

    @property
    def positions_ang(self) -> list[tuple[float, float, float]]:
        return [atom.position_ang for atom in self.atoms]

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "fdf": str(self.fdf_path),
            "structure_type": self.structure_type,
            "coordinate_format": self.coordinate_format,
            "atom_count": self.atom_count,
            "number_of_atoms_declared": self.number_of_atoms_declared,
            "species": [species.to_dict() for species in self.species],
            "lattice_vectors_ang": [list(vector) for vector in self.lattice_vectors_ang],
        }


@dataclass(frozen=True)
class MaterializedFdf:
    path: Path
    metadata: dict[str, Any]


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _first_directive_value(text: str, key: str) -> str | None:
    lower_key = key.lower()
    for raw_line in text.splitlines():
        clean = _strip_comment(raw_line)
        if not clean:
            continue
        parts = clean.split(None, 1)
        if parts and parts[0].lower() == lower_key:
            return parts[1].strip() if len(parts) > 1 else ""
    return None


def _parse_optional_int_directive(text: str, key: str, path: Path) -> int | None:
    value = _first_directive_value(text, key)
    if value in (None, ""):
        return None
    try:
        return int(value.split()[0])
    except ValueError as exc:
        raise FdfMaterializationError(f"{path}: invalid integer directive {key}: {value!r}") from exc


def _coordinate_format(text: str, path: Path) -> str:
    value = _first_directive_value(text, "AtomicCoordinatesFormat")
    if value in (None, ""):
        raise FdfMaterializationError(
            f"{path}: missing AtomicCoordinatesFormat; only explicit Ang coordinates are supported."
        )
    token = value.split()[0].strip().lower()
    if token not in SUPPORTED_COORDINATE_FORMATS:
        raise FdfMaterializationError(
            f"{path}: unsupported AtomicCoordinatesFormat {value!r}; only Ang is supported."
        )
    return token


def _parse_float_triplet(parts: list[str], path: Path, row: str) -> tuple[float, float, float]:
    if len(parts) < 3:
        raise FdfMaterializationError(f"{path}: invalid coordinate/lattice row: {row!r}")
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise FdfMaterializationError(f"{path}: invalid numeric row: {row!r}") from exc


def _lattice_constant_scale_ang(text: str, path: Path) -> float:
    value = _first_directive_value(text, "LatticeConstant")
    if value in (None, ""):
        return 1.0
    parts = value.split()
    try:
        magnitude = float(parts[0])
    except (IndexError, ValueError) as exc:
        raise FdfMaterializationError(f"{path}: invalid LatticeConstant: {value!r}") from exc
    unit = parts[1].lower() if len(parts) > 1 else "ang"
    if unit in {"ang", "angstrom", "angstroms"}:
        return magnitude
    if unit in {"bohr", "bohrs"}:
        return magnitude * BOHR_TO_ANG
    raise FdfMaterializationError(
        f"{path}: unsupported LatticeConstant unit {unit!r}; only Ang and Bohr are supported."
    )


def _parse_lattice_vectors(fdf_path: Path, text: str) -> list[tuple[float, float, float]]:
    scale = _lattice_constant_scale_ang(text, fdf_path)
    vectors = []
    for row in read_fdf_block(fdf_path, "LatticeVectors"):
        vector = _parse_float_triplet(row.split(), fdf_path, row)
        vectors.append(tuple(component * scale for component in vector))
    return vectors


def _parse_atoms(fdf_path: Path, declared_species: set[int]) -> list[FdfAtom]:
    rows = read_fdf_block(fdf_path, "AtomicCoordinatesAndAtomicSpecies")
    if not rows:
        raise FdfMaterializationError(
            f"{fdf_path}: missing AtomicCoordinatesAndAtomicSpecies block."
        )
    atoms = []
    for row in rows:
        parts = row.split()
        if len(parts) < 4:
            raise FdfMaterializationError(
                f"{fdf_path}: invalid AtomicCoordinatesAndAtomicSpecies row: {row!r}"
            )
        position = _parse_float_triplet(parts, fdf_path, row)
        try:
            species_index = int(parts[3])
        except ValueError as exc:
            raise FdfMaterializationError(
                f"{fdf_path}: invalid atomic species index in row: {row!r}"
            ) from exc
        if species_index not in declared_species:
            raise FdfMaterializationError(
                f"{fdf_path}: atom row uses undeclared species index {species_index}."
            )
        atoms.append(FdfAtom(position_ang=position, species_index=species_index))
    return atoms


def extract_fdf_structure(fdf_path: Path, *, structure_type: str | None = None) -> FdfStructure:
    if not fdf_path.is_file():
        raise FdfMaterializationError(f"FDF does not exist or is not a file: {fdf_path}")
    text = fdf_path.read_text(encoding="utf-8", errors="ignore")
    coordinate_format = _coordinate_format(text, fdf_path)
    try:
        species = extract_chemical_species(fdf_path)
    except RuntimeError as exc:
        raise FdfMaterializationError(str(exc)) from exc
    declared_species = {item.index for item in species}
    atoms = _parse_atoms(fdf_path, declared_species)
    declared_atom_count = _parse_optional_int_directive(text, "NumberOfAtoms", fdf_path)
    if declared_atom_count is not None and declared_atom_count != len(atoms):
        raise FdfMaterializationError(
            f"{fdf_path}: NumberOfAtoms={declared_atom_count} does not match "
            f"AtomicCoordinatesAndAtomicSpecies rows={len(atoms)}."
        )
    return FdfStructure(
        fdf_path=fdf_path,
        species=species,
        atoms=atoms,
        lattice_vectors_ang=_parse_lattice_vectors(fdf_path, text),
        coordinate_format=coordinate_format,
        number_of_atoms_declared=declared_atom_count,
        structure_type=structure_type,
    )


def extract_bundle_structure(validated: ValidatedMaterialBundle) -> FdfStructure:
    return extract_fdf_structure(
        validated.bundle.fdf,
        structure_type=validated.bundle.structure_type,
    )


def _format_float(value: float) -> str:
    return f"{float(value):.12f}"


def _replace_or_append_block(text: str, block_name: str, block_lines: list[str]) -> str:
    lines = text.splitlines()
    lower_name = block_name.lower()
    output: list[str] = []
    index = 0
    replaced = False
    while index < len(lines):
        clean = _strip_comment(lines[index]).lower()
        if clean == f"%block {lower_name}":
            output.append(f"%block {block_name}")
            output.extend(block_lines)
            output.append(f"%endblock {block_name}")
            index += 1
            while index < len(lines):
                end_clean = _strip_comment(lines[index]).lower()
                index += 1
                if end_clean == f"%endblock {lower_name}":
                    break
            replaced = True
            continue
        output.append(lines[index])
        index += 1
    if not replaced:
        if output and output[-1].strip():
            output.append("")
        output.append(f"%block {block_name}")
        output.extend(block_lines)
        output.append(f"%endblock {block_name}")
    return "\n".join(output).rstrip() + "\n"


def _set_fdf_directive(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    inserted = False
    lower_key = key.lower()
    for line in lines:
        clean = _strip_comment(line)
        first = clean.split(None, 1)[0].lower() if clean else ""
        if first == lower_key:
            if not inserted:
                output.append(f"{key:<32} {value}")
                inserted = True
            continue
        output.append(line)
    if not inserted:
        if output and output[-1].strip():
            output.append("")
        output.append(f"{key:<32} {value}")
    return "\n".join(output).rstrip() + "\n"


def ensure_required_output_flags(
    text: str,
    required_flags: dict[str, str] | None = None,
) -> str:
    updated = text
    for key, value in (required_flags or DEFAULT_REQUIRED_OUTPUT_FLAGS).items():
        updated = _set_fdf_directive(updated, key, value)
    return updated


def _normalized_positions(positions_ang: list[list[float]] | list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    positions = []
    for position in positions_ang:
        if len(position) != 3:
            raise FdfMaterializationError(f"Each position must contain exactly three values: {position!r}")
        positions.append((float(position[0]), float(position[1]), float(position[2])))
    return positions


def _coordinate_block_lines(
    positions: list[tuple[float, float, float]],
    atom_species: list[int],
    species_by_index: dict[int, MaterialSpecies],
) -> list[str]:
    lines = []
    for position, species_index in zip(positions, atom_species):
        species = species_by_index[int(species_index)]
        lines.append(
            f" {_format_float(position[0])}  {_format_float(position[1])}  "
            f"{_format_float(position[2])}  {int(species_index)}  # {species.label}"
        )
    return lines


_SINGLE_POINT_STRIPPED_KEYS = {"lua.script", "writemdhistory"}


def strip_to_single_point(text: str) -> str:
    """Drop MD/Lua directives so SIESTA runs one SCF at the given geometry.

    Derivative stencils need the Hamiltonian at exactly the written positions.
    A base fdf inherited from an MD dataset carries ``MD.TypeOfRun Verlet`` and
    ``Lua.Script``: SIESTA then evolves the structure for MD.Steps before the
    stored TSHS, silently invalidating the finite-difference reference.
    """
    lines: list[str] = []
    for line in text.splitlines():
        clean = _strip_comment(line)
        first = clean.split(None, 1)[0].lower() if clean else ""
        if first.startswith("md.") or first in _SINGLE_POINT_STRIPPED_KEYS:
            continue
        lines.append(line)
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    text = _set_fdf_directive(text, "MD.TypeOfRun", "CG")
    return _set_fdf_directive(text, "MD.NumCGsteps", "0")


def materialize_fdf_text(
    base_text: str,
    structure: FdfStructure,
    *,
    positions_ang: list[list[float]] | list[tuple[float, float, float]],
    atom_species: list[int] | None = None,
    lattice_vectors_ang: list[list[float]] | list[tuple[float, float, float]] | None = None,
    system_label: str | None = None,
    system_name: str | None = None,
    required_output_flags: dict[str, str] | None = None,
    single_point: bool = False,
) -> str:
    positions = _normalized_positions(positions_ang)
    species_indices = list(atom_species or structure.atom_species)
    if len(positions) != len(species_indices):
        raise FdfMaterializationError(
            f"positions_ang length {len(positions)} does not match atom_species length {len(species_indices)}."
        )
    species_by_index = {item.index: item for item in structure.species}
    missing_species = sorted({int(index) for index in species_indices} - set(species_by_index))
    if missing_species:
        raise FdfMaterializationError(f"Cannot materialize atoms with undeclared species: {missing_species}")

    text = base_text
    if system_label is not None:
        text = _set_fdf_directive(text, "SystemLabel", str(system_label))
    if system_name is not None:
        text = _set_fdf_directive(text, "SystemName", str(system_name))
    text = _set_fdf_directive(text, "NumberOfAtoms", str(len(positions)))
    text = _set_fdf_directive(text, "AtomicCoordinatesFormat", "Ang")
    if lattice_vectors_ang is not None:
        vectors = _normalized_positions(lattice_vectors_ang)
        lattice_lines = [
            f" {_format_float(vector[0])}  {_format_float(vector[1])}  {_format_float(vector[2])}"
            for vector in vectors
        ]
        text = _set_fdf_directive(text, "LatticeConstant", "1.0 Ang")
        text = _replace_or_append_block(text, "LatticeVectors", lattice_lines)
    text = _replace_or_append_block(
        text,
        "AtomicCoordinatesAndAtomicSpecies",
        _coordinate_block_lines(positions, species_indices, species_by_index),
    )
    if single_point:
        text = strip_to_single_point(text)
    return ensure_required_output_flags(text, required_output_flags)


def materialize_sample_fdf(
    base_fdf: Path,
    output_fdf: Path,
    *,
    positions_ang: list[list[float]] | list[tuple[float, float, float]],
    atom_species: list[int] | None = None,
    lattice_vectors_ang: list[list[float]] | list[tuple[float, float, float]] | None = None,
    system_label: str | None = None,
    system_name: str | None = None,
    structure_type: str | None = None,
    required_output_flags: dict[str, str] | None = None,
    single_point: bool = False,
) -> MaterializedFdf:
    structure = extract_fdf_structure(base_fdf, structure_type=structure_type)
    base_text = base_fdf.read_text(encoding="utf-8", errors="ignore")
    materialized_text = materialize_fdf_text(
        base_text,
        structure,
        positions_ang=positions_ang,
        atom_species=atom_species,
        lattice_vectors_ang=lattice_vectors_ang,
        system_label=system_label,
        system_name=system_name,
        required_output_flags=required_output_flags,
        single_point=single_point,
    )
    output_fdf.parent.mkdir(parents=True, exist_ok=True)
    output_fdf.write_text(materialized_text, encoding="utf-8")
    output_structure = extract_fdf_structure(output_fdf, structure_type=structure_type)
    metadata = output_structure.to_manifest_dict()
    metadata.update(
        {
            "base_fdf": str(base_fdf),
            "base_fdf_sha256": file_sha256(base_fdf),
            "materialized_fdf_sha256": file_sha256(output_fdf),
            "required_output_flags": required_output_flags or DEFAULT_REQUIRED_OUTPUT_FLAGS,
            "single_point": bool(single_point),
        }
    )
    return MaterializedFdf(path=output_fdf, metadata=metadata)
