#!/usr/bin/env python3
"""Fuse the AA/AB1/AB2 graphene-hBN MD datasets into one train+validation pool.

This is a pure merge: it never launches SIESTA and never trains. It reads the
train/validation splits of the three stacking datasets, copies each snapshot
directory (with its SIESTA artifacts) into a single ``dataset_root`` with
re-indexed, stacking-prefixed sample ids, writes merged split manifests, and
regenerates the joint benchmark manifests with the existing shared builders.

Precondition (fail-closed): the three sources must share identical basis and
pseudopotential hashes -- a heterogeneous train pool over different bases would
be physically meaningless.
"""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = REPO_ROOT / "shared"
for path in (SCRIPT_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_manifest import write_benchmark_manifests  # noqa: E402
from joint_artifact_contract import (  # noqa: E402
    G2M_DEEPH_BENCHMARK_PROFILE,
    validate_dataset,
)
from ml_vs_siesta import read_dataset_samples  # noqa: E402


SYSTEM_LABEL = "graphene_hBN_bilayer"
STACKING_PRESETS = ("graphene_hBN_AA", "graphene_hBN_AB1", "graphene_hBN_AB2")
MERGE_SPLITS = ("train", "validation")


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


def _base_hashes(source_dataset: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return (basis_file_sha256, pseudopotential_sha256) from provenance."""
    provenance = _json(source_dataset / "material_provenance.json")
    basis = provenance.get("basis_file_sha256") or {}
    pseudo = provenance.get("pseudopotential_sha256") or {}
    if not basis or not pseudo:
        raise RuntimeError(
            f"{source_dataset} provenance lacks basis/pseudo hashes; cannot verify merge safety."
        )
    return dict(basis), dict(pseudo)


def _assert_shared_base(sources: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    ref_basis, ref_pseudo = _base_hashes(sources[0])
    for source in sources[1:]:
        basis, pseudo = _base_hashes(source)
        if basis != ref_basis:
            raise RuntimeError(
                f"Basis hashes differ between {sources[0]} and {source}; refusing to merge."
            )
        if pseudo != ref_pseudo:
            raise RuntimeError(
                f"Pseudopotential hashes differ between {sources[0]} and {source}; refusing to merge."
            )
    return ref_basis, ref_pseudo


def _copy_dataset_material(source_dataset: Path, output_root: Path) -> None:
    """Copy basis + pseudopotentials once (all sources share them)."""
    source_basis = source_dataset / "material_basis"
    if source_basis.is_dir():
        shutil.copytree(source_basis, output_root / "material_basis", dirs_exist_ok=True)
    for pattern in ("*.psf", "*.psml"):
        for pseudo in source_dataset.glob(pattern):
            target = output_root / pseudo.name
            if not target.exists():
                shutil.copy2(pseudo, target)


def _write_split_manifest(output_root: Path, split: str, rows: list[dict[str, Any]]) -> None:
    path = output_root / "splits" / f"{split}_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "method", "source_run", "source_stacking", "source_sample_id",
        "structure_path", "hamiltonian_path", "run_out_path", "metadata_path",
        "valid", "split", "status", "sample_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _material_provenance(
    sources: list[Path],
    basis_hashes: dict[str, str],
    pseudo_hashes: dict[str, str],
) -> dict[str, Any]:
    base = _json(sources[0] / "material_provenance.json")
    base.update(
        {
            "label": SYSTEM_LABEL,
            "preset": SYSTEM_LABEL,
            "material_source": "merged_bilayer_train_from_three_stackings",
            "stacking_mixture": [
                {
                    "preset": path.name,
                    "source_dataset_root": str(path),
                    "material_provenance_sha256_of_basis": _base_hashes(path)[0],
                }
                for path in sources
            ],
            "basis_file_sha256": basis_hashes,
            "pseudopotential_sha256": pseudo_hashes,
            "structure_type": "crystal",
            "environment": {
                **(base.get("environment") if isinstance(base.get("environment"), dict) else {}),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "executable": sys.executable,
            },
            "provenance_note": (
                "Train+validation pool fuses AA/AB1/AB2 graphene-hBN stackings. "
                "Basis and pseudopotentials are identical across all three (asserted "
                "before merge). Test split is intentionally empty; cross testing uses "
                "an independent moire target."
            ),
        }
    )
    return base


def _merge_source(
    source_dataset: Path,
    stacking: str,
    output_root: Path,
    split_rows: dict[str, list[dict[str, Any]]],
    sample_dirs: list[Path],
    split_counters: dict[str, int],
) -> None:
    samples = read_dataset_samples(source_dataset)
    for sample in samples:
        split = sample.split
        if split not in MERGE_SPLITS:
            continue
        index = split_counters[split]
        split_counters[split] += 1
        destination = output_root / "splits" / split / str(index)
        if not sample.sample_dir.is_dir():
            raise RuntimeError(f"Missing source sample dir: {sample.sample_dir}")
        shutil.copytree(sample.sample_dir, destination, dirs_exist_ok=True)
        sample_dirs.append(destination)
        merged_id = f"{stacking}__{sample.sample_id}"
        struct = destination / "RUN.fdf"
        run_out = destination / "RUN.out"
        meta = destination / "metadata.json"
        tshs = next(destination.glob("*.TSHS"), None)
        hsx = next(destination.glob("*.HSX"), None)
        split_rows[split].append(
            {
                "sample_id": merged_id,
                "method": "md_bilayer_merge",
                "source_run": str(source_dataset),
                "source_stacking": stacking,
                "source_sample_id": sample.sample_id,
                "structure_path": str(struct),
                "hamiltonian_path": str(tshs or hsx or ""),
                "run_out_path": str(run_out),
                "metadata_path": str(meta),
                "valid": True,
                "split": split,
                "status": "completed",
                "sample_dir": str(destination),
            }
        )


def build_dataset(sources: list[Path], output_root: Path, *, overwrite: bool) -> dict[str, Any]:
    sources = [source.resolve() for source in sources]
    for source in sources:
        if not (source / "frozen_split_manifest.json").is_file():
            raise RuntimeError(f"Not a materialized dataset (no frozen_split_manifest.json): {source}")
    basis_hashes, pseudo_hashes = _assert_shared_base(sources)

    output_root = _safe_output_root(output_root)
    backup: Path | None = None
    if output_root.exists():
        if not overwrite:
            raise RuntimeError(f"Output exists: {output_root}; pass --overwrite to replace it.")
        backup = output_root.with_name(f".{output_root.name}.backup.{uuid.uuid4().hex}")
        output_root.rename(backup)
    output_root.mkdir(parents=True)

    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in MERGE_SPLITS}
    split_counters: dict[str, int] = {split: 0 for split in MERGE_SPLITS}
    sample_dirs: list[Path] = []
    try:
        _copy_dataset_material(sources[0], output_root)
        for source in sources:
            _merge_source(source, source.name, output_root, split_rows, sample_dirs, split_counters)

        merged_ids = [row["sample_id"] for rows in split_rows.values() for row in rows]
        if len(merged_ids) != len(set(merged_ids)):
            raise RuntimeError("Sample id collision after merge; ids must be unique across stackings.")
        for split in MERGE_SPLITS:
            if not split_rows[split]:
                raise RuntimeError(f"Merged {split!r} split is empty; sources have no {split} samples.")
            _write_split_manifest(output_root, split, split_rows[split])

        provenance = _material_provenance(sources, basis_hashes, pseudo_hashes)
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
            raise RuntimeError(f"Merged bilayer dataset failed joint artifact validation: {validation.errors}")
        dataset_manifest, frozen = write_benchmark_manifests(
            dataset_root=output_root,
            split_root=output_root / "splits",
            generation_mode="merged_bilayer_stackings_md",
            strict_paper_ready_provenance=True,
        )
        if backup is not None:
            shutil.rmtree(backup)
        return {
            "output_root": str(output_root),
            "n_samples": len(sample_dirs),
            "split_counts": frozen["split_counts"],
            "benchmark_dataset_id": dataset_manifest["benchmark_dataset_id"],
            "benchmark_ready": dataset_manifest["benchmark_ready"],
            "sources": [str(source) for source in sources],
        }
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        if backup is not None and backup.exists():
            backup.rename(output_root)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dataset",
        type=Path,
        action="append",
        required=True,
        help="Repeat 3x: the AA/AB1/AB2 MD dataset roots to fuse.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "Comparison" / "datasets" / "graphene_hBN_bilayer_train",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources = [s if s.is_absolute() else REPO_ROOT / s for s in args.source_dataset]
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    result = build_dataset(sources, output_root, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
