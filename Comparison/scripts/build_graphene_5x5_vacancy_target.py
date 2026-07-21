#!/usr/bin/env python3
"""Build a test-only 49-carbon target from pristine graphene 5x5 snapshots."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = REPO_ROOT / "shared"
for path in (SCRIPT_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_manifest import file_sha256, write_benchmark_manifests  # noqa: E402
from fdf_materialization import (  # noqa: E402
    extract_fdf_structure,
    materialize_fdf_text,
)
from joint_artifact_contract import (  # noqa: E402
    G2M_DEEPH_BENCHMARK_PROFILE,
    validate_dataset,
    validate_snapshot,
)
from ml_vs_siesta import read_dataset_samples  # noqa: E402
from run_hamiltonian_derivative_siesta_references import (  # noqa: E402
    run_siesta,
    stage_required_pseudopotentials,
)


SYSTEM_LABEL = "graphene_5x5_vacancy"
DEFAULT_ATOM_INDEX = 24
IDEAL_VACANCY_FRACTIONAL = (0.5, 0.5, 0.0)
VACANCY_SITE_TOLERANCE = 0.05
STATIC_DROP_PREFIXES = ("md.",)
STATIC_DROP_KEYS = {"writemdhistory", "lua.script"}


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    allowed = ((REPO_ROOT / "Comparison" / "datasets").resolve(), Path(tempfile.gettempdir()).resolve())
    if not any(root in resolved.parents for root in allowed):
        raise RuntimeError(
            f"Refusing output root {resolved}; use a child of Comparison/datasets or {tempfile.gettempdir()}."
        )
    return resolved


def _static_fdf(text: str) -> str:
    lines = []
    for line in text.splitlines():
        clean = line.split("#", 1)[0].strip()
        key = clean.split(None, 1)[0].lower() if clean else ""
        if key in STATIC_DROP_KEYS or any(key.startswith(prefix) for prefix in STATIC_DROP_PREFIXES):
            continue
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def _species_label(structure, species_index: int) -> str:
    by_index = {item.index: item.label for item in structure.species}
    return str(by_index.get(species_index) or species_index)


def vacancy_fdf(source_fdf: Path, atom_index: int) -> tuple[str, dict[str, Any]]:
    """Return a static 49-atom FDF plus removal metadata."""
    structure = extract_fdf_structure(source_fdf, structure_type="crystal")
    if structure.atom_count != 50:
        raise RuntimeError(f"Expected 50 atoms in {source_fdf}, found {structure.atom_count}.")
    if atom_index < 0 or atom_index >= structure.atom_count:
        raise RuntimeError(f"atom-index must be in [0, 49], got {atom_index}.")
    removed = structure.atoms[atom_index]
    removed_label = _species_label(structure, removed.species_index)
    if removed_label != "C":
        raise RuntimeError(f"Selected vacancy atom must be C, got {removed_label!r} at index {atom_index}.")

    positions = [atom.position_ang for i, atom in enumerate(structure.atoms) if i != atom_index]
    species = [atom.species_index for i, atom in enumerate(structure.atoms) if i != atom_index]
    text = materialize_fdf_text(
        source_fdf.read_text(encoding="utf-8", errors="ignore"),
        structure,
        positions_ang=positions,
        atom_species=species,
        lattice_vectors_ang=structure.lattice_vectors_ang,
        system_label=SYSTEM_LABEL,
        system_name=SYSTEM_LABEL,
    )
    lattice = np.asarray(structure.lattice_vectors_ang, dtype=float)
    actual_fractional = (
        np.asarray(removed.position_ang) @ np.linalg.inv(lattice)
        if lattice.shape == (3, 3)
        else np.full(3, np.nan)
    )
    periodic_delta = (actual_fractional - np.asarray(IDEAL_VACANCY_FRACTIONAL) + 0.5) % 1.0 - 0.5
    if not np.isfinite(periodic_delta).all() or np.linalg.norm(periodic_delta) > VACANCY_SITE_TOLERANCE:
        raise RuntimeError(
            f"Atom index {atom_index} is not the central vacancy site: fractional position "
            f"{actual_fractional.tolist()}, expected near {list(IDEAL_VACANCY_FRACTIONAL)}."
        )
    metadata = {
        "defect": "monovacancy",
        "source_num_atoms": 50,
        "defect_num_atoms": 49,
        "removed_atom_index": atom_index,
        "removed_atom_species": removed_label,
        "removed_atom_position_fractional": list(IDEAL_VACANCY_FRACTIONAL),
        "removed_atom_actual_position_fractional": actual_fractional.tolist(),
        "removed_atom_position_cartesian_ang": list(removed.position_ang),
        "relaxed": False,
        "spin_polarized": False,
        "system_label": SYSTEM_LABEL,
    }
    return _static_fdf(text), metadata


def _selected_samples(source_dataset: Path, source_split: str, limit: int) -> list[Any]:
    if limit < 1:
        raise RuntimeError("--limit must be a positive integer.")
    samples = [sample for sample in read_dataset_samples(source_dataset) if sample.split == source_split]
    if not samples:
        raise RuntimeError(f"No {source_split!r} samples in {source_dataset}.")
    return samples[:limit]


def _copy_basis(source_dataset: Path, sample_dir: Path, output_root: Path) -> None:
    source_basis = source_dataset / "material_basis"
    if source_basis.is_dir():
        shutil.copytree(source_basis, output_root / "material_basis", dirs_exist_ok=True)
    for basis in sample_dir.glob("*.ion.xml"):
        target = output_root / "material_basis" / basis.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(basis, target)
    for pseudo in source_dataset.glob("*.psf"):
        shutil.copy2(pseudo, output_root / pseudo.name)


def _write_test_manifest(output_root: Path, rows: list[dict[str, Any]]) -> None:
    path = output_root / "splits" / "test_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "method", "source_run", "source_sample_id", "structure_path",
        "hamiltonian_path", "run_out_path", "metadata_path", "valid", "status", "sample_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _material_provenance(
    source_dataset: Path,
    output_root: Path,
    first_sample: Path,
    siesta_command: str,
) -> dict[str, Any]:
    source_path = source_dataset / "material_provenance.json"
    source = _json(source_path) if source_path.is_file() else {}
    material_fdf = REPO_ROOT / "materials" / SYSTEM_LABEL / "RUN.fdf"
    source.update(
        {
            "label": SYSTEM_LABEL,
            "preset": SYSTEM_LABEL,
            "material_source": "derived_monovacancy_from_pristine_test_snapshots",
            "source_dataset_root": str(source_dataset),
            "fdf": str(material_fdf),
            "fdf_sha256": file_sha256(material_fdf),
            "run_out_path": str(first_sample / "RUN.out"),
            "siesta_stdout_path": str(first_sample / "RUN.out"),
            "siesta_command_line": siesta_command,
            "environment": {
                **(source.get("environment") if isinstance(source.get("environment"), dict) else {}),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "executable": sys.executable,
            },
            "structure_type": "crystal",
            "defect": {
                "type": "monovacancy",
                "source_num_atoms": 50,
                "target_num_atoms": 49,
                "relaxed": False,
                "spin_polarized": False,
            },
        }
    )
    return source


def _prepare_geometry(source_sample: Any, destination: Path, atom_index: int) -> dict[str, Any]:
    text, metadata = vacancy_fdf(source_sample.sample_dir / "RUN.fdf", atom_index)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "RUN.fdf").write_text(text, encoding="utf-8")
    metadata.update(
        {
            "source_sample_id": source_sample.sample_id,
            "source_sample_dir": str(source_sample.sample_dir),
        }
    )
    _write_json(destination / "metadata.json", metadata)
    parsed = extract_fdf_structure(destination / "RUN.fdf", structure_type="crystal")
    if parsed.atom_count != 49:
        raise RuntimeError(f"Vacancy geometry validation failed for {destination}: {parsed.atom_count} atoms.")
    return metadata


def dry_run_plan(source_dataset: Path, source_split: str, limit: int, atom_index: int) -> dict[str, Any]:
    samples = _selected_samples(source_dataset, source_split, limit)
    transformed = []
    with tempfile.TemporaryDirectory(prefix="graphene_vacancy_dry_run_") as tmp:
        for index, sample in enumerate(samples):
            destination = Path(tmp) / str(index)
            metadata = _prepare_geometry(sample, destination, atom_index)
            transformed.append(
                {
                    "source_sample_id": sample.sample_id,
                    "removed_atom_position_cartesian_ang": metadata["removed_atom_position_cartesian_ang"],
                    "defect_num_atoms": metadata["defect_num_atoms"],
                }
            )
    return {
        "dry_run": True,
        "source_dataset": str(source_dataset),
        "source_split": source_split,
        "atom_index": atom_index,
        "n_selected": len(samples),
        "siesta_invoked": False,
        "samples": transformed,
    }


def build_target(
    source_dataset: Path,
    output_root: Path,
    *,
    source_split: str,
    limit: int,
    atom_index: int,
    siesta_command: str,
    overwrite: bool,
) -> dict[str, Any]:
    source_dataset = source_dataset.resolve()
    output_root = _safe_output_root(output_root)
    samples = _selected_samples(source_dataset, source_split, limit)
    backup: Path | None = None
    if output_root.exists():
        if not overwrite:
            raise RuntimeError(f"Output exists: {output_root}; pass --overwrite to replace it.")
        backup = output_root.with_name(f".{output_root.name}.backup.{uuid.uuid4().hex}")
        output_root.rename(backup)
    output_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    sample_dirs: list[Path] = []
    try:
        for index, sample in enumerate(samples):
            sample_id = f"vacancy_{sample.sample_id}"
            destination = output_root / "splits" / "test" / str(index)
            _prepare_geometry(sample, destination, atom_index)
            _copy_basis(source_dataset, sample.sample_dir, output_root)
            stage_required_pseudopotentials(reference_dir=destination, source_dataset_root=source_dataset)
            run = run_siesta(destination, command=siesta_command, use_shell=False)
            validation = validate_snapshot(destination)
            if run["returncode"] != 0 or not validation.valid:
                raise RuntimeError(
                    f"SIESTA reference failed for {sample_id}: returncode={run['returncode']}, "
                    f"errors={validation.errors}, missing={validation.missing_required}"
                )
            sample_dirs.append(destination)
            rows.append(
                {
                    "sample_id": sample_id,
                    "method": "md_vacancy",
                    "source_run": str(source_dataset),
                    "source_sample_id": sample.sample_id,
                    "structure_path": str(destination / "RUN.fdf"),
                    "hamiltonian_path": validation.present_artifacts.get("tshs") or validation.present_artifacts["hsx"],
                    "run_out_path": str(destination / "RUN.out"),
                    "metadata_path": str(destination / "metadata.json"),
                    "valid": True,
                    "status": "completed",
                    "sample_dir": str(destination),
                }
            )

        _write_test_manifest(output_root, rows)
        provenance = _material_provenance(source_dataset, output_root, sample_dirs[0], siesta_command)
        _write_json(output_root / "material_provenance.json", provenance)
        validation = validate_dataset(
            output_root,
            snapshot_dirs=sample_dirs,
            basis_dirs=[output_root / "material_basis"],
            pseudopotential_provenance_paths=[output_root / "material_provenance.json"],
            material_identity_paths=[output_root / "material_provenance.json"],
            siesta_input_paths=[sample_dirs[0] / "RUN.fdf", output_root / "material_provenance.json"],
            validation_profile=G2M_DEEPH_BENCHMARK_PROFILE,
        )
        _write_json(output_root / "artifact_validation.json", validation.to_dict())
        if not validation.valid:
            raise RuntimeError(f"Vacancy target failed joint artifact validation: {validation.errors}")
        dataset_manifest, frozen = write_benchmark_manifests(
            dataset_root=output_root,
            split_root=output_root / "splits",
            generation_mode="derived_pristine_monovacancy_static_siesta",
            strict_paper_ready_provenance=True,
        )
        if backup is not None:
            shutil.rmtree(backup)
        return {
            "dry_run": False,
            "output_root": str(output_root),
            "n_samples": len(rows),
            "split_counts": frozen["split_counts"],
            "benchmark_dataset_id": dataset_manifest["benchmark_dataset_id"],
            "benchmark_ready": dataset_manifest["benchmark_ready"],
        }
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        if backup is not None and backup.exists():
            backup.rename(output_root)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--atom-index", type=int, default=DEFAULT_ATOM_INDEX)
    parser.add_argument("--siesta-command", default="siesta")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dataset = args.source_dataset.expanduser()
    if not source_dataset.is_absolute():
        source_dataset = REPO_ROOT / source_dataset
    if args.dry_run:
        result = dry_run_plan(source_dataset.resolve(), args.source_split, args.limit, args.atom_index)
    else:
        result = build_target(
            source_dataset,
            args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root,
            source_split=args.source_split,
            limit=args.limit,
            atom_index=args.atom_index,
            siesta_command=args.siesta_command,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
