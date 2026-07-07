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
import tempfile
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


def _atom_count_from_fdf(fdf_path: Path) -> int | None:
    """Read ``NumberOfAtoms`` from a SIESTA ``.fdf`` (real datasets omit it from
    metadata.json, so this is the reliable fallback)."""
    if not fdf_path.is_file():
        return None
    try:
        text = fdf_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() == "numberofatoms":
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


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
    # Real datasets don't store an atom count in metadata.json; fall back to the
    # snapshot's RUN.fdf.
    return _atom_count_from_fdf(sample_dir / "RUN.fdf")


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


def _real_species_labels(provenance: dict[str, Any]) -> list[str]:
    """Real (non-ghost) species labels. Ghost atoms have atomic_number < 0."""
    return sorted(
        str(s.get("label"))
        for s in (provenance.get("species") or [])
        if int(s.get("atomic_number", 0)) >= 0
    )


def _ghost_species_labels(provenance: dict[str, Any]) -> list[str]:
    return sorted(
        str(s.get("label"))
        for s in (provenance.get("species") or [])
        if int(s.get("atomic_number", 0)) < 0
    )


def _basis_for_real_species(
    basis: dict[str, Any],
    real_labels: set[str],
) -> dict[str, Any]:
    """Keep only basis entries whose file belongs to a real species.

    Basis keys look like ``"C.ion.xml"`` / ``"Ghost-H.ion.xml"``; the species
    label is the stem before the first dot.
    """
    kept: dict[str, Any] = {}
    for key, value in (basis or {}).items():
        stem = str(key).split(".", 1)[0]
        if stem in real_labels:
            kept[key] = value
    return kept


def validate_datasets_compatible(
    small_root: str | Path,
    large_root: str | Path,
) -> dict[str, Any]:
    """Validate that two datasets share species + basis so they can be mixed.

    Compares only the *real* species and their basis files. Ghost atoms
    (``atomic_number < 0``, e.g. Wannier projection centers like ``Ghost-H``)
    are not part of the Graph2Mat/DeepH representation, so they are ignored:
    a 2-atom ``["C", "Ghost-H"]`` cell is compatible with a ``["C"]`` supercell.
    Raises :class:`DatasetCompatibilityError` on a real mismatch.
    """
    small = _load_json(Path(small_root) / "material_provenance.json")
    large = _load_json(Path(large_root) / "material_provenance.json")

    small_species = _real_species_labels(small)
    large_species = _real_species_labels(large)
    if small_species != large_species:
        raise DatasetCompatibilityError(
            f"Real species differ between datasets: {small_species} vs {large_species}. "
            "Mixing requires the same real (non-ghost) species set."
        )

    real_set = set(small_species)
    small_basis = _basis_for_real_species(small.get("basis_file_sha256") or {}, real_set)
    large_basis = _basis_for_real_species(large.get("basis_file_sha256") or {}, real_set)
    if small_basis and large_basis and small_basis != large_basis:
        raise DatasetCompatibilityError(
            "Orbital basis differs between datasets for the real species "
            "(basis_file_sha256 mismatch); Graph2Mat/DeepH cannot train on a "
            f"heterogeneous basis pool. small={small_basis} large={large_basis}."
        )
    ghost_ignored = sorted(set(_ghost_species_labels(small)) | set(_ghost_species_labels(large)))
    return {
        "species": small_species,
        "basis_file_sha256": small_basis or large_basis,
        "ghost_species_ignored": ghost_ignored,
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


def _validate_safe_output_root(output_root: Path) -> Path:
    """Refuse to materialize (and later rmtree) outside a safe working root.

    Safe roots are ``Comparison/results`` and the system temp dir; the output
    must be a strict subdirectory of one of them, which by construction rules
    out ``/``, ``$HOME``, the repo root and ``Comparison/results`` itself.
    """
    resolved = Path(output_root).resolve()
    repo_root = Path(__file__).resolve().parents[3]
    safe_roots = (
        repo_root / "Comparison" / "results",
        Path(tempfile.gettempdir()).resolve(),
    )
    if not any(root in resolved.parents for root in safe_roots):
        raise DatasetMaterializeError(
            f"Refusing to materialize into {resolved}: output_root must live "
            f"inside one of the safe working roots {[str(r) for r in safe_roots]} "
            "(it will be recursively deleted on overwrite)."
        )
    return resolved


def _fixed_common_test_split(
    selected: list[tuple[str, "DatasetSample"]],
    small_pool_ids: list[str],
    fractions: tuple[float, float, float],
    seed: int,
) -> list[str]:
    """Split with a test set derived only from the small pool + seed.

    A fixed fraction (``fractions[2]``) of the *whole* small pool is reserved
    for test; any selected small snapshot in that reservation goes to "test".
    The remaining selected snapshots are shuffled into train/validation with
    the train:validation proportion of ``fractions``.
    """
    rng = random.Random(seed)
    test_ids = fixed_common_test_ids(small_pool_ids, fractions, seed)

    assignment = [""] * len(selected)
    rest: list[int] = []
    for i, (merged_id, _sample) in enumerate(selected):
        if merged_id.startswith("small__") and merged_id[len("small__"):] in test_ids:
            assignment[i] = "test"
        else:
            rest.append(i)
    rng.shuffle(rest)
    n_rest = len(rest)
    train_share = fractions[0] / (fractions[0] + fractions[1])
    n_train = min(max(1, round(train_share * n_rest)), max(n_rest - 1, 1)) if n_rest else 0
    for rank, idx in enumerate(rest):
        assignment[idx] = "train" if rank < n_train else "validation"
    return assignment


def fixed_common_test_ids(
    small_pool_ids: list[str],
    fractions: tuple[float, float, float],
    seed: int,
) -> set[str]:
    rng = random.Random(seed)
    n_test = max(1, round(fractions[2] * len(small_pool_ids)))
    return set(rng.sample(small_pool_ids, min(n_test, len(small_pool_ids))))


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
    """Copy dataset-level provenance / basis / pseudos.

    RUN.fdf/RUN.out stay inside snapshot dirs; a root RUN.fdf makes the shared
    validator treat the dataset root itself as a snapshot.
    """
    for name in ("material_provenance.json",):
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
    mode: str | None = None,
    ratio: float | None = None,
    split_policy: str = "resplit_combined",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize a merged, runner-ready ``dataset_root`` from selected samples.

    Split policies:

    - ``"resplit_combined"`` (default): the merged pool (selected small + large
      snapshots) is re-split into train/validation/test deterministically by
      ``seed`` (the "combined test"). The test set therefore changes with the
      selection, i.e. between ratios of the same sweep.
    - ``"fixed_common_test"``: the test set is a fixed fraction
      (``split_fractions[2]``) of the *small pool*, derived only from
      ``small_root`` + ``seed`` and independent of the ratio, so permutations
      of the same size/seed share exactly the same test snapshots (recommended
      for scientific MAE-vs-composition analysis). ``run_mixing_sweep`` keeps
      those reserved small samples selected even in ``mode="replace"``. The
      remaining selected snapshots are split into train/validation.

    ``mode``/``ratio`` are optional metadata recorded in the provenance file
    (``mixed_dataset_provenance.json``); they do not affect the selection.
    Snapshot artifacts are symlinked (or copied) into ``splits/<split>/<idx>/``
    and the benchmark manifests are regenerated via ``write_benchmark_manifests``.

    ``output_root`` must live inside a safe working root (``Comparison/results``
    or the system temp dir) and, if it already exists, is only recursively
    replaced when ``overwrite=True``.
    """
    small_root = Path(small_root)
    large_root = Path(large_root)
    output_root = Path(output_root)
    _validate_safe_output_root(output_root)
    if output_root.exists() and not overwrite:
        raise DatasetMaterializeError(
            f"output_root already exists: {output_root}. Pass overwrite=True to replace it."
        )

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

    if split_policy == "resplit_combined":
        rng = random.Random(seed)
        split_assignment = _split_pool(len(selected), split_fractions, rng)
    elif split_policy == "fixed_common_test":
        split_assignment = _fixed_common_test_split(
            selected, sorted(small_by_id), split_fractions, seed
        )
    else:
        raise DatasetMaterializeError(
            f"Unknown split_policy {split_policy!r}; "
            "use 'resplit_combined' or 'fixed_common_test'."
        )

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

    # Self-contained provenance of the mixture (reproducibility contract).
    provenance = {
        "schema": "ml_vs_siesta_mixed_dataset_provenance_v1",
        "mode": mode,
        "ratio": ratio,
        "ratio_semantics": "fraction_of_large_pool",
        "seed": seed,
        "small_root": str(small_root),
        "large_root": str(large_root),
        "selected_small_ids": list(selected_small_ids),
        "selected_large_ids": list(selected_large_ids),
        "split_policy": split_policy,
        "split_fractions": list(split_fractions),
        "compatibility": compat,
    }
    (output_root / "mixed_dataset_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
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
