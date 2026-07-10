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
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# shared/ is placed on sys.path by the package __init__.
from benchmark_manifest import write_benchmark_manifests  # noqa: E402
from joint_artifact_contract import validate_snapshot  # noqa: E402
from run_inventory import collect_run_inventory  # noqa: E402

from .dataset_mixing import ratio_semantics_for_mode
from .dataset_compatibility import (
    GHOST_INCOMPATIBLE,
    GHOST_STATUSES_OK,
    build_dataset_compatibility_report,
    write_report as write_compatibility_report,
)

DEFAULT_SPLIT_FRACTIONS = (0.7, 0.15, 0.15)
_SPLITS = ("train", "validation", "test")

# Split policies (audit Fase 7). ``fixed_common_test`` keeps its historical
# name but its evaluation scope is small-only; ``fixed_stratified_test`` is the
# recommended policy with a fixed small+large test.
SPLIT_POLICY_EVALUATION_SCOPE = {
    "fixed_stratified_test": "small_and_large",
    "fixed_common_test": "small_only",
    "fixed_common_test_small_only": "small_only",
    "resplit_combined": "combined_resplit_legacy",
}


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
    split: str | None = None

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
                split=(str(row.get("split")) or None) if row.get("split") else None,
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


GHOST_EXEMPTION_NOTE = (
    "UNVERIFIED exemption: the claim that ghost species (atomic_number < 0, "
    "e.g. Wannier projection centers like Ghost-H) are not part of the "
    "Graph2Mat/DeepH representation is not verified anywhere in this "
    "repository's training code. Ghost atoms carry basis orbitals in SIESTA, "
    "so Hamiltonians of snapshots WITH ghosts contain orbital blocks absent "
    "from snapshots WITHOUT them. Mixing such pools requires an explicit "
    "confirm_ghost_species_exemption=True and is recorded in provenance."
)


def validate_datasets_compatible(
    small_root: str | Path,
    large_root: str | Path,
    *,
    confirm_ghost_species_exemption: bool = False,
) -> dict[str, Any]:
    """Validate that two datasets share species + basis so they can be mixed.

    Compares the *real* species and their basis files; a mismatch raises
    :class:`DatasetCompatibilityError`.

    Ghost atoms (``atomic_number < 0``) are exempted from the species check
    only when the caller explicitly confirms the exemption: the assertion that
    ghosts play no role in the Graph2Mat/DeepH representation is NOT verified
    in code (see ``GHOST_EXEMPTION_NOTE``), so datasets whose ghost species
    sets differ refuse to mix unless ``confirm_ghost_species_exemption=True``.
    The returned report records whether the exemption was required/confirmed.
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

    small_ghosts = _ghost_species_labels(small)
    large_ghosts = _ghost_species_labels(large)

    # Physical compatibility from real artifacts (audit Fase 6): declared vs
    # ACTIVE species, DFT fingerprints, k-point density.
    compatibility_report: dict[str, Any] | None = None
    try:
        small_sample = read_dataset_samples(small_root)[0].sample_dir
        large_sample = read_dataset_samples(large_root)[0].sample_dir
        compatibility_report = build_dataset_compatibility_report(
            small_sample, large_sample, small, large
        )
    except Exception as exc:  # noqa: BLE001 - fall back to provenance-only logic
        compatibility_report = {
            "schema": "ml_vs_siesta_dataset_compatibility_report_v1",
            "compatible": None,
            "ghost_compatibility_status": "unproven",
            "blocking_errors": [],
            "warnings": [f"compatibility report unavailable: {exc}"],
            "sampling_differences": [],
        }

    ghost_status = compatibility_report.get("ghost_compatibility_status", "unproven")
    if ghost_status == GHOST_INCOMPATIBLE:
        raise DatasetCompatibilityError(
            "Ghost species are ACTIVE in one dataset but not the other (target "
            f"spaces differ): {compatibility_report.get('ghost_compatibility', {}).get('reason')}. "
            "This cannot be overridden."
        )
    blocking = compatibility_report.get("blocking_errors") or []
    if blocking:
        raise DatasetCompatibilityError(
            "Datasets are physically incompatible: " + "; ".join(blocking)
        )
    ghost_exemption_required = (
        small_ghosts != large_ghosts and ghost_status not in GHOST_STATUSES_OK
    )
    if ghost_exemption_required and not confirm_ghost_species_exemption:
        raise DatasetCompatibilityError(
            f"Ghost species differ between datasets (small={small_ghosts}, "
            f"large={large_ghosts}) and their activity could not be proven from "
            f"artifacts (status={ghost_status}). {GHOST_EXEMPTION_NOTE} Pass "
            "confirm_ghost_species_exemption=True to mix them anyway."
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
    ghost_ignored = sorted(set(small_ghosts) | set(large_ghosts))
    return {
        "species": small_species,
        "basis_file_sha256": small_basis or large_basis,
        "ghost_species_ignored": ghost_ignored,
        "ghost_compatibility_status": ghost_status,
        "compatibility_report": compatibility_report,
        "ghost_species_exemption": {
            "required": ghost_exemption_required,
            "confirmed": bool(confirm_ghost_species_exemption),
            "verified_in_code": ghost_status in GHOST_STATUSES_OK,
            "small_ghost_species": small_ghosts,
            "large_ghost_species": large_ghosts,
            "note": GHOST_EXEMPTION_NOTE if ghost_exemption_required else "",
        },
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
    small_pool: list["DatasetSample | str"],
    fractions: tuple[float, float, float],
    seed: int,
) -> list[str]:
    """Split with a fixed, temporally blocked test set from the small pool.

    The test ids come from :func:`fixed_common_test_ids` (the source dataset's
    own test split when available, otherwise the temporal tail of the pool) so
    the test set is identical across permutations and never randomly
    interleaved inside the source MD trajectories. The remaining selected
    snapshots are shuffled into train/validation with the train:validation
    proportion of ``fractions``.
    """
    rng = random.Random(seed)
    test_ids = fixed_common_test_ids(small_pool, fractions, seed)

    assignment = [""] * len(selected)
    rest: list[int] = []
    for i, (merged_id, _sample) in enumerate(selected):
        if merged_id.startswith("small__") and merged_id[len("small__"):] in test_ids:
            assignment[i] = "test"
        else:
            rest.append(i)
    rng.shuffle(rest)
    n_rest = len(rest)
    if n_rest < 2:
        raise DatasetMaterializeError(
            "split_policy='fixed_common_test' cannot populate both train and "
            f"validation: only {n_rest} selected snapshot(s) remain outside the "
            "reserved test set. Select more snapshots."
        )
    train_share = fractions[0] / (fractions[0] + fractions[1])
    n_train = min(max(1, round(train_share * n_rest)), n_rest - 1)
    for rank, idx in enumerate(rest):
        assignment[idx] = "train" if rank < n_train else "validation"
    return assignment


def _fixed_stratified_test_split(
    selected: list[tuple[str, "DatasetSample"]],
    small_pool: list["DatasetSample | str"],
    large_pool: list["DatasetSample | str"],
    fractions: tuple[float, float, float],
    seed: int,
) -> list[str]:
    """Fixed, stratified (small AND large) test set; rest -> train/validation.

    Test ids come from :func:`fixed_common_test_ids` applied independently to
    each pool (source test split when available, temporal tail otherwise), so
    the test set is identical across ratios, modes and selection seeds and
    contains both domains (audit Fase 7 policy ``fixed_stratified_test``).
    """
    rng = random.Random(seed)
    small_test = fixed_common_test_ids(small_pool, fractions, seed)
    large_test = fixed_common_test_ids(large_pool, fractions, seed)

    assignment = [""] * len(selected)
    rest: list[int] = []
    for i, (merged_id, _sample) in enumerate(selected):
        if merged_id.startswith("small__") and merged_id[len("small__"):] in small_test:
            assignment[i] = "test"
        elif merged_id.startswith("large__") and merged_id[len("large__"):] in large_test:
            assignment[i] = "test"
        else:
            rest.append(i)
    if "test" not in assignment:
        raise DatasetMaterializeError(
            "split_policy='fixed_stratified_test' produced an empty test split; "
            "selected ids must include the reserved small/large test snapshots."
        )
    rng.shuffle(rest)
    n_rest = len(rest)
    if n_rest < 2:
        raise DatasetMaterializeError(
            "split_policy='fixed_stratified_test' cannot populate both train and "
            f"validation: only {n_rest} selected snapshot(s) remain outside the "
            "reserved test set. Select more snapshots."
        )
    train_share = fractions[0] / (fractions[0] + fractions[1])
    n_train = min(max(1, round(train_share * n_rest)), n_rest - 1)
    for rank, idx in enumerate(rest):
        assignment[idx] = "train" if rank < n_train else "validation"
    return assignment


def _temporal_sort_key(sample_id: str) -> tuple[str, int]:
    """Sort key approximating MD temporal order (numeric suffix, e.g. md_17)."""
    match = re.search(r"(\d+)$", sample_id)
    if match:
        return (sample_id[: match.start()], int(match.group(1)))
    return (sample_id, -1)


def fixed_common_test_ids(
    small_pool: list["DatasetSample | str"],
    fractions: tuple[float, float, float],
    seed: int = 0,
) -> set[str]:
    """Fixed test ids for the small pool, preserving temporal structure.

    Priority (audit C2: the small pool is MD trajectory frames 1 fs apart, so
    a RANDOM test subset interleaves test frames between train frames of the
    same trajectory — temporal leakage):

    1. The source dataset's own frozen test split (``DatasetSample.split ==
       "test"``), when available: this reuses exactly the temporally blocked
       test set the source benchmark used.
    2. Otherwise the temporal *tail* of the pool (last ``fractions[2]`` ids in
       numeric-suffix order): a single temporal boundary instead of random
       interleaving.

    ``seed`` is accepted for API compatibility but the selection is fully
    deterministic (selection-independent, so all permutations of one sweep
    share the same test snapshots).
    """
    ids: list[str] = []
    source_test: set[str] = set()
    for item in small_pool:
        if isinstance(item, DatasetSample):
            ids.append(item.sample_id)
            if item.split == "test":
                source_test.add(item.sample_id)
        else:
            ids.append(str(item))
    if source_test:
        return source_test
    ordered = sorted(ids, key=_temporal_sort_key)
    n_test = min(max(1, round(fractions[2] * len(ordered))), len(ordered))
    return set(ordered[len(ordered) - n_test :])


_ORB_INDX_HEADER_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*=\s*orbitals in unit cell")


def _snapshot_matrix_elements(sample_dir: Path) -> int | None:
    """H matrix size (n_unit_orbitals x n_supercell_orbitals) from ORB_INDX."""
    matches = sorted(Path(sample_dir).glob("*.ORB_INDX"))
    if not matches:
        return None
    try:
        for line in matches[0].read_text(encoding="utf-8", errors="ignore").splitlines()[:5]:
            found = _ORB_INDX_HEADER_RE.match(line)
            if found:
                return int(found.group(1)) * int(found.group(2))
    except OSError:
        return None
    return None


def _composition_metrics(
    selected: list[tuple[str, "DatasetSample"]],
    split_assignment: list[str],
    split_policy: str,
) -> dict[str, Any]:
    """Effective composition (audit Fase 8): counts, atoms and matrix elements.

    Fractions are computed over the TRAIN split, which is what the model
    actually learns from; the "dataset size" axis must mean actual_train_size.
    Edge blocks are not reported (they require neighbour lists); node blocks
    equal the atom count, matrix elements come from each snapshot's ORB_INDX.
    """
    per_split_origin: dict[str, dict[str, int]] = {
        split: {"small": 0, "large": 0} for split in _SPLITS
    }
    atoms = {"small": 0, "large": 0}
    elements = {"small": 0, "large": 0}
    elements_known = True
    for (merged_id, sample), split in zip(selected, split_assignment):
        origin = "small" if merged_id.startswith("small__") else "large"
        per_split_origin[split][origin] += 1
        if split == "train":
            atoms[origin] += int(sample.n_atoms or 0)
            n_elements = _snapshot_matrix_elements(sample.sample_dir)
            if n_elements is None:
                elements_known = False
            else:
                elements[origin] += n_elements

    n_small_train = per_split_origin["train"]["small"]
    n_large_train = per_split_origin["train"]["large"]
    actual_train_size = n_small_train + n_large_train

    def _fraction(large_part: float, small_part: float) -> float | None:
        total = large_part + small_part
        return (large_part / total) if total else None

    return {
        "split_policy": split_policy,
        "evaluation_scope": SPLIT_POLICY_EVALUATION_SCOPE.get(split_policy, "unknown"),
        "actual_train_size": actual_train_size,
        "n_small_train": n_small_train,
        "n_large_train": n_large_train,
        "validation_size": sum(per_split_origin["validation"].values()),
        "test_small_size": per_split_origin["test"]["small"],
        "test_large_size": per_split_origin["test"]["large"],
        "materialized_total_size": len(selected),
        "actual_large_fraction_by_snapshots": _fraction(n_large_train, n_small_train),
        "small_atoms_total": atoms["small"],
        "large_atoms_total": atoms["large"],
        "actual_large_fraction_by_atoms": _fraction(atoms["large"], atoms["small"]),
        "small_node_blocks": atoms["small"],
        "large_node_blocks": atoms["large"],
        "actual_large_fraction_by_blocks": _fraction(atoms["large"], atoms["small"]),
        "small_matrix_elements": elements["small"] if elements_known else None,
        "large_matrix_elements": elements["large"] if elements_known else None,
        "actual_large_fraction_by_matrix_elements": (
            _fraction(elements["large"], elements["small"]) if elements_known else None
        ),
    }


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


def _copy_dataset_level_files(
    small_root: Path,
    large_root: Path,
    output_root: Path,
    compat: dict[str, Any],
) -> None:
    """Write dataset-level provenance / basis / pseudos for the merged pool.

    RUN.fdf/RUN.out stay inside snapshot dirs; a root RUN.fdf makes the shared
    validator treat the dataset root itself as a snapshot. The merged
    ``material_provenance.json`` must not masquerade as the small dataset's
    provenance (audit I3): when the two sources differ it is written as an
    explicitly mixed provenance pointing at ``mixed_dataset_provenance.json``.
    """
    _write_merged_material_provenance(small_root, large_root, output_root, compat)
    for psf in small_root.glob("*.psf"):
        shutil.copy2(psf, output_root / psf.name)
    basis_src = small_root / "material_basis"
    if basis_src.is_dir():
        shutil.copytree(basis_src, output_root / "material_basis", dirs_exist_ok=True)


def _write_merged_material_provenance(
    small_root: Path,
    large_root: Path,
    output_root: Path,
    compat: dict[str, Any],
) -> None:
    small_path = small_root / "material_provenance.json"
    large_path = large_root / "material_provenance.json"
    small = _load_json(small_path) if small_path.is_file() else {}
    large = _load_json(large_path) if large_path.is_file() else {}
    if not small and not large:
        return
    if small == large:
        # Truly homogeneous sources: the shared provenance is the pool's.
        shutil.copy2(small_path, output_root / "material_provenance.json")
        return

    real_set = set(compat.get("species") or [])
    small_pseudo = small.get("pseudopotential_sha256") or {}
    large_pseudo = large.get("pseudopotential_sha256") or {}
    small_label = str(small.get("label") or small_root.name)
    large_label = str(large.get("label") or large_root.name)
    merged: dict[str, Any] = {
        "schema": "ml_vs_siesta_mixed_material_provenance_v1",
        "material_source": "mixed_dataset",
        "heterogeneous_material_pool": True,
        "provenance_source_of_truth": "mixed_dataset_provenance.json",
        "label": small_label if small_label == large_label else f"mixed({small_label}+{large_label})",
        # Real species are validated identical between the sources.
        "species": [
            entry
            for entry in (small.get("species") or [])
            if str(entry.get("label")) in real_set
        ],
        "ghost_species_by_source": {
            "small": _ghost_species_labels(small),
            "large": _ghost_species_labels(large),
        },
        "basis_file_sha256": compat.get("basis_file_sha256") or {},
        "mixed_from": {
            "small_root": str(small_root),
            "small_label": small_label,
            "small_fdf_sha256": small.get("fdf_sha256"),
            "large_root": str(large_root),
            "large_label": large_label,
            "large_fdf_sha256": large.get("fdf_sha256"),
        },
        "warnings": [],
    }
    if small_pseudo == large_pseudo:
        merged["pseudopotential_sha256"] = small_pseudo
    else:
        merged["pseudopotential_sha256_by_source"] = {
            "small": small_pseudo,
            "large": large_pseudo,
        }
        merged["warnings"].append("pseudopotential hashes differ between sources")
    # Dataset-level FDF lineage: two source templates. Hash the pair so the
    # provenance gate has a real, traceable token instead of the small's hash.
    fdf_pair = {"small": small.get("fdf_sha256"), "large": large.get("fdf_sha256")}
    if any(fdf_pair.values()):
        import hashlib

        merged["fdf_sha256"] = hashlib.sha256(
            json.dumps(fdf_pair, sort_keys=True).encode("utf-8")
        ).hexdigest()
        merged["fdf_sha256_semantics"] = "sha256_of_source_fdf_sha256_pair"
        merged["fdf_sha256_by_source"] = fdf_pair
    exemption = (compat.get("ghost_species_exemption") or {})
    if exemption.get("required"):
        merged["warnings"].append(GHOST_EXEMPTION_NOTE)
    (output_root / "material_provenance.json").write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
    split_policy: str = "fixed_common_test",
    overwrite: bool = False,
    confirm_ghost_species_exemption: bool = False,
    allow_source_test_in_train: bool = False,
) -> dict[str, Any]:
    """Materialize a merged, runner-ready ``dataset_root`` from selected samples.

    Split policies:

    - ``"resplit_combined"`` (legacy, opt-in): the merged pool (selected small +
      large snapshots) is re-split into train/validation/test deterministically
      by ``seed`` (the "combined test"). The test set therefore changes with the
      selection, i.e. between ratios of the same sweep — MAE differences can
      come from the test set changing, not from the composition.
    - ``"fixed_common_test"`` (default): the test set is fixed and temporally
      blocked — the small source dataset's own frozen test split when it has
      one, otherwise the temporal tail (``split_fractions[2]``) of the small
      pool. It is independent of the ratio/selection, so permutations of the
      same size/seed share exactly the same test snapshots (recommended for
      scientific MAE-vs-composition analysis), and test frames are never
      randomly interleaved inside the source MD trajectories (audit C2).
      ``run_mixing_sweep`` keeps those reserved small samples selected even in
      ``mode="replace"``. The remaining selected snapshots are split into
      train/validation.

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

    compat = validate_datasets_compatible(
        small_root,
        large_root,
        confirm_ghost_species_exemption=confirm_ghost_species_exemption,
    )

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

    if split_policy == "fixed_common_test_small_only":
        split_policy = "fixed_common_test"  # historical name, same semantics
    if split_policy == "resplit_combined":
        rng = random.Random(seed)
        split_assignment = _split_pool(len(selected), split_fractions, rng)
    elif split_policy == "fixed_common_test":
        split_assignment = _fixed_common_test_split(
            selected, list(small_by_id.values()), split_fractions, seed
        )
        if "test" not in split_assignment:
            raise DatasetMaterializeError(
                "split_policy='fixed_common_test' produced an empty test split: "
                "none of the selected small snapshots fall in the reserved common "
                "test set. Include the reserved small ids in selected_small_ids "
                "(run_mixing_sweep does this automatically via reserved_small_ids)."
            )
    elif split_policy == "fixed_stratified_test":
        split_assignment = _fixed_stratified_test_split(
            selected,
            list(small_by_id.values()),
            list(large_by_id.values()),
            split_fractions,
            seed,
        )
    else:
        raise DatasetMaterializeError(
            f"Unknown split_policy {split_policy!r}; use 'fixed_stratified_test', "
            "'fixed_common_test' (small-only test) or 'resplit_combined' (legacy)."
        )

    # No-leakage guard (audit Fase 7): a snapshot the SOURCE dataset held out
    # as test must never train or validate the mixed model.
    if not allow_source_test_in_train:
        for (merged_id, sample), split in zip(selected, split_assignment):
            if sample.split == "test" and split in ("train", "validation"):
                raise DatasetMaterializeError(
                    f"Source-test snapshot {merged_id} would be assigned to "
                    f"'{split}' under split_policy={split_policy!r} — temporal "
                    "leakage. Use a fixed test policy, drop the snapshot, or pass "
                    "allow_source_test_in_train=True (NOT scientifically valid)."
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
                "source_sample_id": sample.sample_id,
                "source_split": sample.split or "",
                "n_atoms": sample.n_atoms if sample.n_atoms is not None else "",
            }
        )

    _write_split_csvs(output_root / "splits", split_rows)
    _copy_dataset_level_files(small_root, large_root, output_root, compat)
    if compat.get("compatibility_report"):
        write_compatibility_report(compat["compatibility_report"], output_root)

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

    composition = _composition_metrics(selected, split_assignment, split_policy)

    # Self-contained provenance of the mixture (reproducibility contract).
    provenance = {
        "schema": "ml_vs_siesta_mixed_dataset_provenance_v2",
        "mode": mode,
        "ratio": ratio,
        "requested_ratio": ratio,
        "ratio_semantics": ratio_semantics_for_mode(mode),
        "seed": seed,
        "selection_seed": seed,
        "small_root": str(small_root),
        "large_root": str(large_root),
        "selected_small_ids": list(selected_small_ids),
        "selected_large_ids": list(selected_large_ids),
        "split_policy": split_policy,
        "evaluation_scope": composition["evaluation_scope"],
        "split_fractions": list(split_fractions),
        "composition": composition,
        "compatibility": compat,
        "run_inventory": collect_run_inventory(),
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
        "split_policy": split_policy,
        "evaluation_scope": composition["evaluation_scope"],
        "composition": composition,
        "compatibility": compat,
        "seed": seed,
        "selection_seed": seed,
    }


def _write_split_csvs(split_root: Path, split_rows: dict[str, list[dict[str, str]]]) -> None:
    import csv

    split_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "sample_dir", "split", "system_label", "source_root", "origin",
        "source_sample_id", "source_split", "n_atoms",
    ]
    for split in _SPLITS:
        path = split_root / f"{split}_manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in split_rows[split]:
                writer.writerow(row)
