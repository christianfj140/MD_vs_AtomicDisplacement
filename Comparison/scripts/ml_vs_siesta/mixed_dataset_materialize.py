"""Bridge: turn a small/large mixing selection into a real merged dataset_root.

This is the missing link between the (dry-run only) mixing manifests in
:mod:`dataset_mixing` and the real train/predict/evaluate engine
(``g2m_deeph_runner``). It copies/links selected snapshots from two source
datasets into one merged ``dataset_root`` with a merged train/val/test split and
regenerates the benchmark manifests using the existing shared builders.

It never launches SIESTA and never trains. Basis/species compatibility between
the two source systems is validated up front (both graphene systems must share
the carbon PAO basis for a heterogeneous train pool to be meaningful).
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# shared/ is placed on sys.path by the package __init__.
from benchmark_manifest import write_benchmark_manifests  # noqa: E402
from joint_artifact_contract import validate_snapshot  # noqa: E402

DEFAULT_SPLIT_FRACTIONS = (0.7, 0.15, 0.15)
_SPLITS = ("train", "validation", "test")


class DatasetCompatibilityError(RuntimeError):
    """Raised when two source datasets cannot be mixed (basis/species differ)."""


class DatasetMaterializeError(RuntimeError):
    """Raised when a merged dataset cannot be materialized."""


@dataclass
class DatasetSample:
    """One snapshot from a source dataset, discovered from its frozen split."""

    sample_id: str
    sample_dir: Path
    system_label: str | None
    source_root: Path
    n_atoms: int | None = None

    def to_manifest_entry(self) -> dict[str, Any]:
        return {"id": self.sample_id, "n_atoms": self.n_atoms}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _snapshot_atom_count(sample_dir: Path) -> int | None:
    meta = sample_dir / "metadata.json"
    if meta.is_file():
        try:
            payload = _load_json(meta)
        except (ValueError, OSError):
            payload = {}
        for key in ("n_atoms", "num_atoms", "atom_count", "natoms"):
            if payload.get(key) is not None:
                try:
                    return int(payload[key])
                except (TypeError, ValueError):
                    pass
    return None


def read_dataset_samples(dataset_root: str | Path) -> list[DatasetSample]:
    """Read a dataset's ``frozen_split_manifest.json`` into a list of samples."""
    root = Path(dataset_root)
    manifest_path = root / "frozen_split_manifest.json"
    if not manifest_path.is_file():
        raise DatasetMaterializeError(
            f"No frozen_split_manifest.json in {root}; not a materialized dataset_root."
        )
    manifest = _load_json(manifest_path)
    rows = manifest.get("rows") or []
    samples: list[DatasetSample] = []
    for row in rows:
        sample_dir = Path(str(row.get("sample_dir") or ""))
        sample_id = str(row.get("sample_id") or sample_dir.name)
        samples.append(
            DatasetSample(
                sample_id=sample_id,
                sample_dir=sample_dir,
                system_label=row.get("system_label"),
                source_root=root,
                n_atoms=_snapshot_atom_count(sample_dir),
            )
        )
    if not samples:
        raise DatasetMaterializeError(f"Frozen split manifest has no rows: {manifest_path}")
    return samples


def dataset_atom_count(dataset_root: str | Path) -> int | None:
    """Best-effort atom count for a dataset (from its first snapshot)."""
    for sample in read_dataset_samples(dataset_root):
        if sample.n_atoms is not None:
            return sample.n_atoms
    return None


def validate_datasets_compatible(
    small_root: str | Path,
    large_root: str | Path,
) -> dict[str, Any]:
    """Validate that two datasets share species + basis so they can be mixed.

    Compares ``material_provenance.json`` species labels and basis file hashes.
    Raises :class:`DatasetCompatibilityError` on any mismatch.
    """
    small = _load_json(Path(small_root) / "material_provenance.json")
    large = _load_json(Path(large_root) / "material_provenance.json")

    small_species = sorted(str(s.get("label")) for s in (small.get("species") or []))
    large_species = sorted(str(s.get("label")) for s in (large.get("species") or []))
    if small_species != large_species:
        raise DatasetCompatibilityError(
            f"Species differ between datasets: {small_species} vs {large_species}. "
            "Mixing requires the same species set."
        )

    small_basis = small.get("basis_file_sha256") or {}
    large_basis = large.get("basis_file_sha256") or {}
    if small_basis and large_basis and small_basis != large_basis:
        raise DatasetCompatibilityError(
            "Orbital basis differs between datasets (basis_file_sha256 mismatch); "
            "Graph2Mat/DeepH cannot train on a heterogeneous basis pool. "
            f"small={small_basis} large={large_basis}."
        )
    return {
        "species": small_species,
        "basis_file_sha256": small_basis or large_basis,
        "compatible": True,
    }


def _split_pool(
    n: int,
    fractions: tuple[float, float, float],
    rng: random.Random,
) -> list[str]:
    """Assign each of ``n`` items to a split, guaranteeing ≥1 per split when n≥3."""
    order = list(range(n))
    rng.shuffle(order)
    if n >= 3:
        n_train = max(1, round(fractions[0] * n))
        n_val = max(1, round(fractions[1] * n))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - 1 - n_train)
    else:
        # Degenerate pools: fill splits in priority order.
        n_train = min(1, n)
        n_val = 1 if n >= 2 else 0
    assignment = [""] * n
    for rank, item in enumerate(order):
        if rank < n_train:
            assignment[item] = "train"
        elif rank < n_train + n_val:
            assignment[item] = "validation"
        else:
            assignment[item] = "test"
    return assignment


def _link_or_copy(src: Path, dst: Path, *, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        try:
            os.symlink(src.resolve(), dst)
            return
        except (OSError, NotImplementedError):
            pass
    shutil.copy2(src, dst)


def _copy_dataset_level_files(small_root: Path, output_root: Path) -> None:
    """Copy provenance / basis / pseudos / RUN.fdf so provenance validates."""
    for name in ("material_provenance.json", "RUN.fdf", "RUN.out"):
        src = small_root / name
        if src.is_file():
            shutil.copy2(src, output_root / name)
    for psf in small_root.glob("*.psf"):
        shutil.copy2(psf, output_root / psf.name)
    basis_src = small_root / "material_basis"
    if basis_src.is_dir():
        shutil.copytree(basis_src, output_root / "material_basis", dirs_exist_ok=True)


def materialize_mixed_dataset(
    small_root: str | Path,
    large_root: str | Path,
    selected_small_ids: list[str],
    selected_large_ids: list[str],
    output_root: str | Path,
    *,
    seed: int = 0,
    split_fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS,
    link: bool = True,
) -> dict[str, Any]:
    """Materialize a merged, runner-ready ``dataset_root`` from selected samples.

    The merged pool (selected small + large snapshots) is re-split into
    train/validation/test deterministically by ``seed`` (the "combined test").
    Snapshot artifacts are symlinked (or copied) into ``splits/<split>/<idx>/``
    and the benchmark manifests are regenerated via ``write_benchmark_manifests``.
    """
    small_root = Path(small_root)
    large_root = Path(large_root)
    output_root = Path(output_root)

    compat = validate_datasets_compatible(small_root, large_root)

    small_by_id = {s.sample_id: s for s in read_dataset_samples(small_root)}
    large_by_id = {s.sample_id: s for s in read_dataset_samples(large_root)}

    selected: list[tuple[str, DatasetSample]] = []
    for sid in selected_small_ids:
        if sid not in small_by_id:
            raise DatasetMaterializeError(f"small sample id not found: {sid}")
        selected.append((f"small__{sid}", small_by_id[sid]))
    for sid in selected_large_ids:
        if sid not in large_by_id:
            raise DatasetMaterializeError(f"large sample id not found: {sid}")
        selected.append((f"large__{sid}", large_by_id[sid]))

    if not selected:
        raise DatasetMaterializeError("No samples selected for the merged dataset.")

    rng = random.Random(seed)
    split_assignment = _split_pool(len(selected), split_fractions, rng)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, list[dict[str, str]]] = {split: [] for split in _SPLITS}
    validation_snapshots: list[dict[str, Any]] = []
    per_split_index: dict[str, int] = {split: 0 for split in _SPLITS}

    for (merged_id, sample), split in zip(selected, split_assignment):
        idx = per_split_index[split]
        per_split_index[split] += 1
        dest = output_root / "splits" / split / str(idx)
        dest.mkdir(parents=True, exist_ok=True)
        for artifact in sorted(p for p in sample.sample_dir.iterdir() if p.is_file()):
            if artifact.name == "ML_prediction.HSX":
                continue
            _link_or_copy(artifact, dest / artifact.name, link=link)
        result = validate_snapshot(dest)
        validation_snapshots.append(
            {
                "snapshot_dir": str(dest),
                "system_label": result.system_label,
                "valid": bool(result.valid),
                "repair_required": False,
                "missing_required": list(result.missing_required),
                "errors": list(result.errors),
                "warnings": list(result.warnings),
                "present_artifacts": dict(result.present_artifacts),
            }
        )
        split_rows[split].append(
            {
                "sample_id": merged_id,
                "sample_dir": str(dest),
                "split": split,
                "system_label": result.system_label or "",
                "source_root": str(sample.source_root),
                "origin": "small" if merged_id.startswith("small__") else "large",
            }
        )

    _write_split_csvs(output_root / "splits", split_rows)
    _copy_dataset_level_files(small_root, output_root)

    all_valid = all(s["valid"] for s in validation_snapshots)
    artifact_validation = {
        "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
        "valid": all_valid,
        "warnings": [] if all_valid else ["some merged snapshots failed validation"],
        "snapshots": validation_snapshots,
    }
    (output_root / "artifact_validation.json").write_text(
        json.dumps(artifact_validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    write_benchmark_manifests(
        dataset_root=output_root,
        split_root=output_root / "splits",
        generation_mode="mixed_dataset",
        strict_paper_ready_provenance=False,
    )

    split_counts = {split: len(rows) for split, rows in split_rows.items()}
    return {
        "output_root": str(output_root),
        "n_small_selected": len(selected_small_ids),
        "n_large_selected": len(selected_large_ids),
        "total": len(selected),
        "split_counts": split_counts,
        "compatibility": compat,
        "seed": seed,
    }


def _write_split_csvs(split_root: Path, split_rows: dict[str, list[dict[str, str]]]) -> None:
    import csv

    split_root.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "sample_dir", "split", "system_label", "source_root", "origin"]
    for split in _SPLITS:
        path = split_root / f"{split}_manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in split_rows[split]:
                writer.writerow(row)
