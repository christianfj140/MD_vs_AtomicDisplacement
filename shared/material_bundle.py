"""Validation helpers for user-provided SIESTA material input bundles.

The bundle validator is intentionally small: it checks paths, extracts species
from the ``ChemicalSpeciesLabel`` block, verifies pseudopotential coverage, and
returns hashes/provenance. It does not try to be a complete FDF parser.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PSEUDOPOTENTIAL_EXTENSIONS = (".psf", ".psml")
BASIS_EXTENSIONS = (".ion.xml", ".ion")
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class MaterialBundleError(RuntimeError):
    """Raised when a material bundle is incomplete or inconsistent."""


@dataclass(frozen=True)
class MaterialSpecies:
    index: int
    atomic_number: int
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "atomic_number": self.atomic_number,
            "label": self.label,
        }


@dataclass(frozen=True)
class MaterialBundle:
    label: str
    fdf: Path
    pseudopotential_dir: Path
    basis_dir: Path | None = None
    structure_type: str | None = None
    source_paths_absolute: bool = False


@dataclass(frozen=True)
class ValidatedMaterialBundle:
    bundle: MaterialBundle
    species: list[MaterialSpecies]
    pseudopotentials: dict[str, Path]
    fdf_sha256: str
    pseudopotential_sha256: dict[str, str]
    basis_file_sha256: dict[str, str]
    absolute_paths_used: bool

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "label": self.bundle.label,
            "structure_type": self.bundle.structure_type,
            "fdf": str(self.bundle.fdf),
            "fdf_sha256": self.fdf_sha256,
            "pseudopotential_dir": str(self.bundle.pseudopotential_dir),
            "basis_dir": str(self.bundle.basis_dir) if self.bundle.basis_dir else None,
            "species": [species.to_dict() for species in self.species],
            "pseudopotentials": {
                label: str(path)
                for label, path in sorted(self.pseudopotentials.items())
            },
            "pseudopotential_sha256": dict(sorted(self.pseudopotential_sha256.items())),
            "basis_file_sha256": dict(sorted(self.basis_file_sha256.items())),
            "absolute_paths_used": self.absolute_paths_used,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_bundle_path(value: str | Path, root_dir: Path, *, allow_absolute: bool = True) -> tuple[Path, bool]:
    if value in (None, ""):
        raise MaterialBundleError("Material bundle path cannot be empty.")
    raw_path = Path(value).expanduser()
    if raw_path.is_absolute():
        if not allow_absolute:
            raise MaterialBundleError(f"Absolute material bundle paths are not allowed: {raw_path}")
        return raw_path.resolve(), True

    root = root_dir.expanduser().resolve()
    resolved = (root / raw_path).resolve()
    if not _is_relative_to(resolved, root):
        raise MaterialBundleError(
            f"Material bundle path escapes its root: {value!r} resolved under {root}"
        )
    return resolved, False


def _validate_label(label: Any) -> str:
    text = str(label or "").strip()
    if not text:
        raise MaterialBundleError("material.label must be non-empty.")
    if not LABEL_PATTERN.fullmatch(text):
        raise MaterialBundleError(
            "material.label must contain only letters, numbers, '.', '_' or '-' "
            "and must start with a letter or number."
        )
    return text


def material_bundle_from_config(
    config: dict[str, Any],
    *,
    base_dir: str | Path,
    allow_absolute_paths: bool = True,
) -> MaterialBundle:
    raw_material = config.get("material", config)
    if not isinstance(raw_material, dict):
        raise MaterialBundleError("material config must be a mapping.")

    label = _validate_label(raw_material.get("label"))
    root_dir = Path(base_dir)
    source_paths_absolute = False
    if raw_material.get("root_dir") not in (None, ""):
        root_dir, root_absolute = _resolve_bundle_path(
            raw_material["root_dir"],
            Path(base_dir),
            allow_absolute=allow_absolute_paths,
        )
        source_paths_absolute = source_paths_absolute or root_absolute

    fdf, fdf_absolute = _resolve_bundle_path(
        raw_material.get("fdf", ""),
        root_dir,
        allow_absolute=allow_absolute_paths,
    )
    source_paths_absolute = source_paths_absolute or fdf_absolute
    pseudo_dir, pseudo_absolute = _resolve_bundle_path(
        raw_material.get("pseudopotential_dir", ""),
        root_dir,
        allow_absolute=allow_absolute_paths,
    )
    source_paths_absolute = source_paths_absolute or pseudo_absolute
    basis_dir = None
    if raw_material.get("basis_dir") not in (None, ""):
        basis_dir, basis_absolute = _resolve_bundle_path(
            raw_material["basis_dir"],
            root_dir,
            allow_absolute=allow_absolute_paths,
        )
        source_paths_absolute = source_paths_absolute or basis_absolute

    structure_type = raw_material.get("structure_type")
    structure_type = str(structure_type).strip() if structure_type not in (None, "") else None
    return MaterialBundle(
        label=label,
        fdf=fdf,
        pseudopotential_dir=pseudo_dir,
        basis_dir=basis_dir,
        structure_type=structure_type,
        source_paths_absolute=source_paths_absolute,
    )


def _strip_fdf_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def read_fdf_block(fdf_path: Path, block_name: str) -> list[str]:
    lower_name = block_name.lower()
    lines = fdf_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start: int | None = None
    for index, raw_line in enumerate(lines):
        clean = _strip_fdf_comment(raw_line).lower()
        if clean == f"%block {lower_name}":
            start = index + 1
            break
    if start is None:
        return []

    block: list[str] = []
    end_marker = f"%endblock {lower_name}"
    for raw_line in lines[start:]:
        clean = _strip_fdf_comment(raw_line)
        if clean.lower() == end_marker:
            return block
        if clean:
            block.append(clean)
    raise MaterialBundleError(f"FDF block {block_name!r} is not closed in {fdf_path}.")


def extract_chemical_species(fdf_path: Path) -> list[MaterialSpecies]:
    rows = read_fdf_block(fdf_path, "ChemicalSpeciesLabel")
    if not rows:
        raise MaterialBundleError(
            f"{fdf_path} does not define a ChemicalSpeciesLabel block; "
            "cannot validate pseudopotential coverage."
        )

    species: list[MaterialSpecies] = []
    seen_indices: set[int] = set()
    for line in rows:
        parts = line.split()
        if len(parts) < 3:
            raise MaterialBundleError(f"Invalid ChemicalSpeciesLabel row in {fdf_path}: {line!r}")
        try:
            index = int(parts[0])
            atomic_number = int(parts[1])
        except ValueError as exc:
            raise MaterialBundleError(
                f"Invalid ChemicalSpeciesLabel numeric fields in {fdf_path}: {line!r}"
            ) from exc
        label = str(parts[2]).strip()
        if not label:
            raise MaterialBundleError(f"Empty species label in {fdf_path}: {line!r}")
        if index in seen_indices:
            raise MaterialBundleError(f"Duplicate species index {index} in {fdf_path}.")
        seen_indices.add(index)
        species.append(MaterialSpecies(index=index, atomic_number=atomic_number, label=label))
    return species


def extract_coordinate_species_indices(fdf_path: Path) -> list[int]:
    rows = read_fdf_block(fdf_path, "AtomicCoordinatesAndAtomicSpecies")
    indices: list[int] = []
    for line in rows:
        parts = line.split()
        if len(parts) < 4:
            raise MaterialBundleError(
                f"Invalid AtomicCoordinatesAndAtomicSpecies row in {fdf_path}: {line!r}"
            )
        try:
            indices.append(int(parts[3]))
        except ValueError as exc:
            raise MaterialBundleError(
                f"Invalid atomic species index in {fdf_path}: {line!r}"
            ) from exc
    return indices


def validate_fdf_species_consistency(fdf_path: Path, species: list[MaterialSpecies]) -> None:
    declared = {item.index for item in species}
    used = set(extract_coordinate_species_indices(fdf_path))
    missing = sorted(used - declared)
    if missing:
        raise MaterialBundleError(
            f"{fdf_path} uses undeclared species indices in AtomicCoordinatesAndAtomicSpecies: {missing}"
        )


def resolve_pseudopotentials(
    pseudopotential_dir: Path,
    species: list[MaterialSpecies],
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for item in species:
        candidates = [
            pseudopotential_dir / f"{item.label}{extension}"
            for extension in PSEUDOPOTENTIAL_EXTENSIONS
            if (pseudopotential_dir / f"{item.label}{extension}").is_file()
        ]
        if item.atomic_number < 0 and not candidates:
            continue
        if not candidates:
            extensions = ", ".join(PSEUDOPOTENTIAL_EXTENSIONS)
            raise MaterialBundleError(
                f"Missing pseudopotential for species {item.label!r} in "
                f"{pseudopotential_dir}; expected {item.label} with one of: {extensions}."
            )
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise MaterialBundleError(
                f"Ambiguous pseudopotential for species {item.label!r} in "
                f"{pseudopotential_dir}: {names}"
            )
        resolved[item.label] = candidates[0].resolve()
    return resolved


def _basis_hashes(basis_dir: Path | None) -> dict[str, str]:
    if basis_dir is None:
        return {}
    hashes: dict[str, str] = {}
    for path in sorted(path for path in basis_dir.iterdir() if path.is_file()):
        if any(path.name.endswith(extension) for extension in BASIS_EXTENSIONS):
            hashes[path.name] = file_sha256(path)
    return hashes


def validate_material_bundle(bundle: MaterialBundle) -> ValidatedMaterialBundle:
    if not bundle.fdf.is_file():
        raise MaterialBundleError(f"Material FDF does not exist or is not a file: {bundle.fdf}")
    if not bundle.pseudopotential_dir.is_dir():
        raise MaterialBundleError(
            f"Material pseudopotential directory does not exist: {bundle.pseudopotential_dir}"
        )
    if bundle.basis_dir is not None and not bundle.basis_dir.is_dir():
        raise MaterialBundleError(f"Material basis directory does not exist: {bundle.basis_dir}")

    species = extract_chemical_species(bundle.fdf)
    validate_fdf_species_consistency(bundle.fdf, species)
    pseudos = resolve_pseudopotentials(bundle.pseudopotential_dir, species)
    pseudo_hashes = {
        label: file_sha256(path)
        for label, path in sorted(pseudos.items())
    }
    return ValidatedMaterialBundle(
        bundle=bundle,
        species=species,
        pseudopotentials=pseudos,
        fdf_sha256=file_sha256(bundle.fdf),
        pseudopotential_sha256=pseudo_hashes,
        basis_file_sha256=_basis_hashes(bundle.basis_dir),
        absolute_paths_used=bundle.source_paths_absolute,
    )


def validate_material_config(
    config: dict[str, Any],
    *,
    base_dir: str | Path,
    allow_absolute_paths: bool = True,
) -> ValidatedMaterialBundle:
    bundle = material_bundle_from_config(
        config,
        base_dir=base_dir,
        allow_absolute_paths=allow_absolute_paths,
    )
    return validate_material_bundle(bundle)


def manifest_json(validated: ValidatedMaterialBundle) -> str:
    return json.dumps(
        validated.to_manifest_dict(),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
