#!/usr/bin/env python3
"""Small helpers for material provenance in manifests and aggregate rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


INCOMPATIBLE_MATERIAL_WARNING_CODE = "INCOMPATIBLE_MATERIAL_PROVENANCE"

MATERIAL_FLAT_FIELDS = (
    "material_label",
    "material_bundle_path",
    "material_source",
    "material_preset",
    "material_structure_type",
    "material_species",
    "material_atom_count",
    "material_cell_summary",
    "fdf_sha256",
    "pseudopotential_sha256_by_species",
    "basis_sha256_by_species",
    "siesta_settings_hash",
    "siesta_output_flags",
    "graph2mat_config_hash",
    "split_manifest_hash",
    "dataset_recipe",
    "dataset_recipe_parameters",
    "reference_matrix_sha256",
    "prediction_matrix_sha256",
    "material_identity_hash",
    "material_compatibility_hash",
)

MATERIAL_MAP_FIELDS = (
    "material_label_by_method",
    "material_identity_hash_by_method",
    "material_compatibility_hash_by_method",
    "fdf_sha256_by_method",
    "pseudopotential_sha256_by_method",
    "basis_sha256_by_method",
)


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def stable_json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_text(value).encode("utf-8")).hexdigest()


def file_collection_hash(paths: list[Path]) -> str:
    entries = []
    for path in sorted({item.resolve() for item in paths if item.exists() and item.is_file()}):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append((path.name, digest.hexdigest()))
    return stable_json_hash(entries) if entries else ""


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", {}, [], False):
            return value
    return None


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalise_species(species: Any) -> list[dict[str, Any]] | list[str]:
    if not isinstance(species, list):
        return []
    if all(isinstance(item, dict) for item in species):
        return sorted(
            (
                {
                    key: item[key]
                    for key in ("index", "atomic_number", "label")
                    if key in item
                }
                for item in species
            ),
            key=lambda item: (int(item.get("index") or 0), str(item.get("label") or "")),
        )
    return sorted(str(item) for item in species if item not in (None, ""))


def _atom_count_from_material(material: dict[str, Any]) -> int | None:
    for key in ("atom_count", "number_of_atoms", "n_atoms"):
        try:
            value = material.get(key)
            if value not in (None, ""):
                return int(value)
        except (TypeError, ValueError):
            pass
    atoms = material.get("atoms")
    if isinstance(atoms, list):
        return len(atoms)
    return None


def _basis_hashes_from_graph2mat(graph2mat: dict[str, Any]) -> dict[str, str]:
    by_species = graph2mat.get("basis_files_by_species")
    if not isinstance(by_species, dict):
        return {}
    hashes: dict[str, str] = {}
    for species, item in by_species.items():
        if isinstance(item, dict) and item.get("sha256"):
            hashes[str(species)] = str(item["sha256"])
    return dict(sorted(hashes.items()))


def _first_material_mapping(sources: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    for source in sources:
        material = source.get("material")
        if isinstance(material, dict):
            return material
    for source in sources:
        if source.get("label") or source.get("material_label"):
            return source
    return {}


def flatten_material_provenance(*sources: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic, JSON-safe material provenance summary.

    The current repository has material metadata in several sidecars. This
    helper accepts any combination of those dictionaries and extracts the common
    fields without requiring legacy archives to contain all of them.
    """

    valid_sources = tuple(source for source in sources if isinstance(source, dict) and source)
    if not valid_sources:
        return {}

    material = _first_material_mapping(valid_sources)
    graph2mat = next(
        (_as_mapping(source.get("graph2mat")) for source in valid_sources if isinstance(source.get("graph2mat"), dict)),
        {},
    )
    reference = next(
        (_as_mapping(source.get("reference_matrix")) for source in valid_sources if isinstance(source.get("reference_matrix"), dict)),
        {},
    )
    prediction = next(
        (_as_mapping(source.get("prediction_matrix")) for source in valid_sources if isinstance(source.get("prediction_matrix"), dict)),
        {},
    )

    material_species = _normalise_species(
        _first_present(
            material.get("species"),
            *(source.get("material_species") for source in valid_sources),
        )
    )
    basis_hashes = _first_present(
        *(source.get("basis_sha256_by_species") for source in valid_sources),
        material.get("basis_sha256_by_species"),
        material.get("basis_file_sha256"),
        _basis_hashes_from_graph2mat(graph2mat),
    ) or {}
    pseudo_hashes = _first_present(
        *(source.get("pseudopotential_sha256_by_species") for source in valid_sources),
        material.get("pseudopotential_sha256_by_species"),
        material.get("pseudopotential_sha256"),
    ) or {}
    split_hash = _first_present(
        *(source.get("split_manifest_hash") for source in valid_sources),
        graph2mat.get("split_manifest_hash"),
        stable_json_hash(graph2mat.get("split_file_sha256"))
        if isinstance(graph2mat.get("split_file_sha256"), dict) and graph2mat.get("split_file_sha256")
        else None,
    )
    dataset_recipe = _first_present(*(source.get("dataset_recipe") for source in valid_sources))
    dataset_recipe_parameters = _first_present(
        *(source.get("dataset_recipe_parameters") for source in valid_sources),
        _as_mapping(dataset_recipe).get("parameters") if isinstance(dataset_recipe, dict) else None,
        _as_mapping(dataset_recipe).get("generation_parameters") if isinstance(dataset_recipe, dict) else None,
    )

    provenance = {
        "material_label": _first_present(
            *(source.get("material_label") for source in valid_sources),
            material.get("label"),
        ),
        "material_bundle_path": _first_present(
            *(source.get("material_bundle_path") for source in valid_sources),
            material.get("material_bundle_path"),
            material.get("material_yaml"),
            material.get("fdf"),
        ),
        "material_source": _first_present(
            *(source.get("material_source") for source in valid_sources),
            material.get("material_source"),
        ),
        "material_preset": _first_present(
            *(source.get("material_preset") for source in valid_sources),
            material.get("preset"),
        ),
        "material_structure_type": _first_present(
            *(source.get("material_structure_type") for source in valid_sources),
            material.get("structure_type"),
        ),
        "material_species": material_species,
        "material_atom_count": _first_present(
            *(source.get("material_atom_count") for source in valid_sources),
            _atom_count_from_material(material),
        ),
        "material_cell_summary": _first_present(
            *(source.get("material_cell_summary") for source in valid_sources),
            material.get("cell_summary"),
            material.get("lattice_vectors"),
        ),
        "fdf_sha256": _first_present(
            *(source.get("fdf_sha256") for source in valid_sources),
            material.get("fdf_sha256"),
            material.get("base_fdf_sha256"),
        ),
        "pseudopotential_sha256_by_species": dict(sorted(_as_mapping(pseudo_hashes).items())),
        "basis_sha256_by_species": dict(sorted(_as_mapping(basis_hashes).items())),
        "siesta_settings_hash": _first_present(*(source.get("siesta_settings_hash") for source in valid_sources)),
        "siesta_output_flags": _first_present(
            *(source.get("siesta_output_flags") for source in valid_sources),
            material.get("siesta_output_flags"),
            material.get("required_output_flags"),
            *(source.get("required_output_flags") for source in valid_sources),
        ),
        "graph2mat_config_hash": _first_present(
            *(source.get("graph2mat_config_hash") for source in valid_sources),
            graph2mat.get("config_sha256"),
        ),
        "split_manifest_hash": split_hash,
        "dataset_recipe": dataset_recipe,
        "dataset_recipe_parameters": dataset_recipe_parameters,
        "reference_matrix_sha256": _first_present(
            *(source.get("reference_matrix_sha256") for source in valid_sources),
            reference.get("sha256"),
        ),
        "prediction_matrix_sha256": _first_present(
            *(source.get("prediction_matrix_sha256") for source in valid_sources),
            prediction.get("sha256"),
        ),
    }
    identity_payload = {
        "label": provenance["material_label"],
        "structure_type": provenance["material_structure_type"],
        "species": provenance["material_species"],
        "fdf_sha256": provenance["fdf_sha256"],
        "pseudopotential_sha256_by_species": provenance["pseudopotential_sha256_by_species"],
        "basis_sha256_by_species": provenance["basis_sha256_by_species"],
    }
    compatibility_payload = {
        **identity_payload,
        "siesta_settings_hash": provenance["siesta_settings_hash"],
        "siesta_output_flags": provenance["siesta_output_flags"],
    }
    identity_present = any(value not in (None, "", {}, [], False) for value in identity_payload.values())
    if identity_present:
        provenance["material_identity_hash"] = stable_json_hash(identity_payload)
        provenance["material_compatibility_hash"] = stable_json_hash(compatibility_payload)
    return {key: value for key, value in provenance.items() if value not in (None, "", {}, [], False)}


def material_maps_from_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {
        key: _as_mapping(manifest.get(key)).copy()
        for key in MATERIAL_MAP_FIELDS
    }
    method_provenance = _as_mapping(manifest.get("method_provenance"))
    for method, entry in method_provenance.items():
        if not isinstance(entry, dict):
            continue
        flat = flatten_material_provenance(entry.get("material_provenance") or entry)
        if flat.get("material_label"):
            maps["material_label_by_method"].setdefault(str(method), flat["material_label"])
        if flat.get("material_identity_hash"):
            maps["material_identity_hash_by_method"].setdefault(str(method), flat["material_identity_hash"])
        if flat.get("material_compatibility_hash"):
            maps["material_compatibility_hash_by_method"].setdefault(
                str(method), flat["material_compatibility_hash"]
            )
        if flat.get("fdf_sha256"):
            maps["fdf_sha256_by_method"].setdefault(str(method), flat["fdf_sha256"])
        if flat.get("pseudopotential_sha256_by_species"):
            maps["pseudopotential_sha256_by_method"].setdefault(
                str(method), flat["pseudopotential_sha256_by_species"]
            )
        if flat.get("basis_sha256_by_species"):
            maps["basis_sha256_by_method"].setdefault(str(method), flat["basis_sha256_by_species"])
    return {key: dict(sorted(value.items())) for key, value in maps.items()}


def material_compatibility_warning(material_maps: dict[str, Any]) -> str:
    hashes = _as_mapping(material_maps.get("material_compatibility_hash_by_method"))
    known = {str(method): str(value) for method, value in hashes.items() if value not in (None, "", False)}
    if len(set(known.values())) <= 1:
        return ""
    detail = ", ".join(f"{method}={value[:12]}" for method, value in sorted(known.items()))
    return (
        f"{INCOMPATIBLE_MATERIAL_WARNING_CODE}: material compatibility hashes differ across "
        f"methods ({detail}); do not pool these runs as one benchmark."
    )
