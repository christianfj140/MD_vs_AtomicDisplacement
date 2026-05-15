"""Material-aware Graph2Mat configuration helpers.

These helpers keep the Graph2Mat YAML schema unchanged. Material provenance is
written to a sidecar JSON file so unknown keys are not injected into Graph2Mat
configs.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from material_bundle import file_sha256
from material_presets import resolve_material_bundle


GRAPH2MAT_BASIS_EXTENSION = ".ion.xml"
PROVENANCE_FILE_NAME = "config_provenance.json"


def _relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path, start).replace("\\", "/")


def _split_file_hashes(dataset_dir: Path, training_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    split_dir = dataset_dir / "splits"
    if split_dir.exists():
        for path in sorted(item for item in split_dir.rglob("*") if item.is_file()):
            hashes[_relpath(path, training_dir)] = file_sha256(path)
    runs_json = training_dir / "runs.json"
    if runs_json.exists():
        hashes[_relpath(runs_json, training_dir)] = file_sha256(runs_json)
    return hashes


def resolve_graph2mat_basis_files(validated: Any) -> dict[str, Path]:
    basis_dir = validated.bundle.basis_dir
    if basis_dir is None:
        raise RuntimeError(
            "Graph2Mat config generation requires material.basis_dir with "
            f"{GRAPH2MAT_BASIS_EXTENSION} files."
        )
    resolved: dict[str, Path] = {}
    for species in validated.species:
        expected = basis_dir / f"{species.label}{GRAPH2MAT_BASIS_EXTENSION}"
        if not expected.is_file():
            fallback = basis_dir / f"{species.label}.ion"
            if fallback.is_file():
                raise RuntimeError(
                    "Graph2Mat config generation requires .ion.xml basis files; "
                    f"found only {fallback} for species {species.label!r}."
                )
            raise RuntimeError(
                f"Missing Graph2Mat basis file for species {species.label!r}: {expected}"
            )
        resolved[species.label] = expected.resolve()
    return resolved


def copy_graph2mat_basis_files(
    basis_by_species: dict[str, Path],
    target_dir: Path,
) -> dict[str, dict[str, str]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {path.name for path in basis_by_species.values()}
    extras = sorted(
        path.name
        for path in target_dir.glob(f"*{GRAPH2MAT_BASIS_EXTENSION}")
        if path.name not in expected_names
    )
    if extras:
        raise RuntimeError(
            "Graph2Mat basis target contains files not declared by the material "
            f"bundle: {', '.join(extras)}. Remove the stale files or use a fresh dataset directory."
        )

    copied: dict[str, dict[str, str]] = {}
    for label, source in sorted(basis_by_species.items()):
        target = target_dir / source.name
        source_hash = file_sha256(source)
        if target.exists():
            if not target.is_file():
                raise RuntimeError(f"Graph2Mat basis target is not a file: {target}")
            target_hash = file_sha256(target)
            if target_hash != source_hash:
                raise RuntimeError(
                    f"Graph2Mat basis file for species {label!r} differs from "
                    f"the material bundle copy: {target}"
                )
            action = "verified"
        else:
            shutil.copy2(source, target)
            target_hash = file_sha256(target)
            action = "copied"
        copied[label] = {
            "source": str(source),
            "target": str(target),
            "file_name": target.name,
            "sha256": target_hash,
            "action": action,
        }
    return copied


def apply_material_graph2mat_config(
    config: dict[str, Any],
    *,
    base_dir: str | Path,
    dataset_dir: str | Path,
    training_dir: str | Path,
    basis_subdir: str = "material_basis",
) -> dict[str, Any]:
    training_dir = Path(training_dir)
    dataset_dir = Path(dataset_dir)
    resolved = resolve_material_bundle(config, base_dir=base_dir)
    validated = resolved.validated
    basis_by_species = resolve_graph2mat_basis_files(validated)
    target_dir = dataset_dir / basis_subdir
    copied_basis = copy_graph2mat_basis_files(basis_by_species, target_dir)
    basis_glob = _relpath(target_dir / f"*{GRAPH2MAT_BASIS_EXTENSION}", training_dir)

    updated_sections: list[str] = []
    for section_name in ("training", "testing", "prediction"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        data = section.get("data")
        if isinstance(data, dict):
            data["basis_files"] = basis_glob
            updated_sections.append(f"{section_name}.data")
    training_data = (config.get("training", {}) or {}).get("data", {}) or {}
    if isinstance(training_data, dict):
        training_data["basis_files"] = basis_glob
        updated_sections.append("training.data")
        matrix_target = str(training_data.get("out_matrix", "hamiltonian"))
        if matrix_target != "hamiltonian":
            raise RuntimeError(
                "Only Graph2Mat data.out_matrix='hamiltonian' is supported by this "
                f"benchmark material-aware config path; got {matrix_target!r}."
            )
    else:
        matrix_target = "hamiltonian"

    return {
        "material": resolved.to_manifest_dict(),
        "graph2mat": {
            "matrix_target": matrix_target,
            "basis_files": basis_glob,
            "basis_files_target_dir": str(target_dir),
            "basis_files_by_species": copied_basis,
            "split_file_sha256": _split_file_hashes(dataset_dir, training_dir),
            "updated_sections": sorted(set(updated_sections)),
        },
    }


def write_graph2mat_config_provenance(
    config_path: Path,
    provenance: dict[str, Any],
    *,
    validation_metadata: dict[str, Any] | None = None,
) -> Path:
    payload = dict(provenance)
    graph2mat = dict(payload.get("graph2mat") or {})
    graph2mat["config_path"] = str(config_path)
    graph2mat["config_sha256"] = file_sha256(config_path) if config_path.exists() else None
    if validation_metadata:
        graph2mat["validation"] = validation_metadata
    payload["graph2mat"] = graph2mat
    output_path = config_path.with_name(PROVENANCE_FILE_NAME)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_graph2mat_config_provenance(training_dir: Path) -> dict[str, Any]:
    path = training_dir / PROVENANCE_FILE_NAME
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}
