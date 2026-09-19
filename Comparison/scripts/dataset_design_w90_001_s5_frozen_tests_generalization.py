#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S5 -- frozen tests + generalization matrix.

Builds three independent frozen-test families for the W90 graphene pilot and
evaluates every S4 (``dataset_design_w90_001_s4_train_learning_curves.py``)
model on all three, producing a (training family x frozen-test family)
generalization matrix. Frozen-test results are never fed back into sampler,
amplitude or hyperparameter choice (S4 is already finished and frozen before
this script runs).

Cites and reuses, rather than reimplements (S1 section 4):
  - ``docs/dataset_design_w90_001_s1_convention_manifest.md`` (S1).
  - ``Comparison/scripts/w90_displacement_sampler_family.py`` (S2): six
    training-time generators plus the new ``generate_latin_hypercube`` (added
    in this step) used only for the frozen ``common_displacement`` test.
  - ``Comparison/scripts/run_w90_displacement_siesta_pilot.py`` (S3):
    ``materialize_structures``, ``verify_hermiticity``,
    ``verify_orbital_dimension``, ``gamma_hamiltonian`` and the
    ``canonical_ang_base_fdf`` / ``run_derivative_siesta_references`` idempotent
    SIESTA-submission path -- reused verbatim for ``common_displacement`` and
    ``ood_displacement``.
  - ``Comparison/scripts/dataset_design_w90_001_s4_train_learning_curves.py``
    (S4): ``Sample``, ``write_val_manifest``, ``run_prediction``,
    ``compute_validation_metrics`` -- reused verbatim for every frozen test so
    the metric definitions are identical to the ones already reported per
    training group.
  - ``Comparison/scripts/reference_selection.py::choose_reference_matrix``.
  - ``shared/fdf_materialization.py::materialize_sample_fdf`` to write a
    RUN.fdf for each selected MD frame (only a TSHS is persisted for those
    frames; a RUN.fdf is required by ``predict_model_on_dataset.py``'s
    manifest schema).

Three frozen-test families:
  - ``common_displacement``: ``generate_latin_hypercube`` (a Latin Hypercube
    QMC construction, distinct from every training-time generator), mixing
    amplitudes continuously across 0.03-0.08 Ang, k in {1, 2}, all four
    dimensionalities (in-plane and out-of-plane).
  - ``md_frozen``: real MD frames for the same graphene primitive cell,
    reused from ``Comparison/results/results_md/MD_dataset*_*`` (already
    SIESTA-computed, never used by S3/S4 training -- S2's
    ``md_family_contract()`` reserved MD for a separate step that never fed
    the S3/S4 pilot). Selected by measuring the statistical inefficiency of
    three configurational observables (nearest-neighbor bond length, mean
    bond angle, out-of-plane pyramidalization angle) over the frame-index
    series, then keeping only frames spaced at least that many MD steps
    apart -- not a fixed/arbitrary stride.
  - ``ood_displacement``: R = 0.12 Ang (``OOD_AMPLITUDE_ANG``, outside the
    S3 pilot's {0.03, 0.08} grid) *and*, for k=2, a (family, amplitude)
    combination S3 never generated -- ``generate_local_pair_modes`` (the
    bond-frame-correlated longitudinal/transverse displacement family) at
    amplitude 0.12, whereas S3's ``build_pilot_configurations`` only ever
    calls that generator at amplitude in {0.03, 0.08}. (Note:
    ``pair_mode="arbitrary"`` is not usable for this novelty -- the W90
    system is a 2-atom primitive cell, so ``select_active_atoms``'s
    min-image convention makes "farthest atom" resolve to the same single
    neighbor as "bonded"; that pair_mode distinction only matters for
    N>2 supercells.)

Known limitation (documented rather than hidden): only 20 real MD frames
across the whole 0..59 trajectory were persisted with a SIESTA-computed
Hamiltonian by the pre-existing MD comparison campaign (the rest lived only
in a since-cleaned-up workspace directory) and those 20 are themselves
sparse/irregularly spaced along the trajectory. The statistical-inefficiency
estimate in ``estimate_statistical_inefficiency`` is therefore a rough,
small-sample estimate -- reported transparently in
``md_frozen_selection_report.json``, not a paper-grade autocorrelation
analysis. Regenerating a dedicated, densely-sampled MD trajectory for this
pilot (via ``MD/scripts/generate_md_dataset.py``'s temperature-block
orchestrator) is a separate, much larger integration project, matching the
scope boundary S4 already drew around the same generator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import w90_displacement_sampler_family as sampler  # noqa: E402
import run_w90_displacement_siesta_pilot as pilot  # noqa: E402
import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402
from reference_selection import choose_reference_matrix  # noqa: E402
from fdf_materialization import materialize_sample_fdf  # noqa: E402
from run_epc_siesta_reference import canonical_ang_base_fdf  # noqa: E402
from run_hamiltonian_derivative_siesta_references import (  # noqa: E402
    DerivativeSiestaReferenceError,
    run_derivative_siesta_references,
)

DEFAULT_S4_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s4"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s5"
DEFAULT_MD_RESULTS_ROOT = REPO_ROOT / "Comparison/results/results_md"
DEFAULT_SIESTA_COMMAND = pilot.DEFAULT_SIESTA_COMMAND

COMMON_SEED = 42  # distinct from S2/S3's PILOT_SEED = 0
COMMON_VALIDATION_SEED = 41  # distinct from COMMON_SEED: a *validation* pool, not a frozen test
COMMON_N_PER_GROUP = 3  # k in {1,2} x 4 dims x 3 = 24 samples
OOD_N_PER_GROUP = 4  # k in {1,2} x 4 dims x 4 = 32 samples
OOD_AMPLITUDE_ANG = sampler.OOD_AMPLITUDE_ANG

FROZEN_TEST_NAMES = ("common_displacement", "md_frozen", "ood_displacement")
# NOT a frozen test: this is the single validation distribution every
# sampler family is scored on to answer "best sampler at fixed N/cost"
# (S7 conclusions 1, 2, 9). Task rule (S1 section 12): frozen tests must
# never be used to select a sampler/radius/hyperparameter, so ranking must
# come from a set disjoint from FROZEN_TEST_NAMES -- reusing
# common_displacement itself for ranking would consume the one untouched
# final test for the reported winner (audit finding: "no untouched final
# test remains for the reported winner").
COMMON_VALIDATION_NAME = "common_validation"

GENERALIZATION_MATRIX_FIELDNAMES = [
    "training_family",
    "frozen_test",
    "n_models",
    "H_MAE_mean",
    "H_RMSE_mean",
    "rel_Frob_mean",
    "spectral_err_mean",
    "hermiticity_mean",
]

PER_MODEL_FIELDNAMES = [
    "training_family",
    "training_dim",
    "training_k",
    "training_n_train",
    "training_seed",
    "frozen_test",
    "H_MAE",
    "H_RMSE",
    "rel_Frob",
    "spectral_err",
    "hermiticity",
    "n_evaluated",
    "status",
    "checkpoint",
    "split_id",
]

# Numeric fields that come back as strings when a cached CSV row is reused
# instead of re-evaluated; must be coerced back before feeding a mean/np.mean.
_PER_MODEL_NUMERIC_FIELDS = ("H_MAE", "H_RMSE", "rel_Frob", "spectral_err", "hermiticity", "n_evaluated")


# --------------------------------------------------------------------------
# common_displacement (S2's new latin_hypercube sampler; never used by S3)
# --------------------------------------------------------------------------


def build_common_displacement_configs(
    geometry: sampler.Geometry,
    *,
    n_per_group: int = COMMON_N_PER_GROUP,
    seed: int = COMMON_SEED,
) -> list[tuple[str, sampler.Configuration]]:
    entries: list[tuple[str, sampler.Configuration]] = []
    for k in (1, 2):
        for dim in sampler.DIMENSIONALITIES:
            configs = sampler.generate_latin_hypercube(
                geometry, k, dim, n_structures=n_per_group, seed=seed
            )
            for index, config in enumerate(configs):
                sample_id = f"common_displacement__k{k}__{dim}__{index:03d}"
                entries.append((sample_id, config))
    return entries


def build_common_validation_configs(
    geometry: sampler.Geometry,
    *,
    n_per_group: int = COMMON_N_PER_GROUP,
    seed: int = COMMON_VALIDATION_SEED,
) -> list[tuple[str, sampler.Configuration]]:
    """Same Latin-Hypercube construction as ``common_displacement``, disjoint seed.

    Used only to rank samplers at fixed N/cost (S7 conclusions 1, 2, 9) --
    never reported as a generalization/frozen-test result, so it is not
    subject to the "frozen tests forbidden for selection" rule and does not
    consume ``common_displacement`` for that purpose.
    """

    entries: list[tuple[str, sampler.Configuration]] = []
    for k in (1, 2):
        for dim in sampler.DIMENSIONALITIES:
            configs = sampler.generate_latin_hypercube(
                geometry, k, dim, n_structures=n_per_group, seed=seed
            )
            for index, config in enumerate(configs):
                sample_id = f"{COMMON_VALIDATION_NAME}__k{k}__{dim}__{index:03d}"
                entries.append((sample_id, config))
    return entries


# --------------------------------------------------------------------------
# ood_displacement (R=0.12 Ang; k=2 uses generate_local_pair_modes, a
# (family, amplitude) combination S3's build_pilot_configurations never runs
# -- it only calls that generator at amplitude in {0.03, 0.08})
# --------------------------------------------------------------------------


def build_ood_displacement_configs(
    geometry: sampler.Geometry,
    *,
    n_per_group: int = OOD_N_PER_GROUP,
    seed: int = COMMON_SEED,
    amplitude_ang: float = OOD_AMPLITUDE_ANG,
) -> list[tuple[str, sampler.Configuration]]:
    entries: list[tuple[str, sampler.Configuration]] = []
    for dim in sampler.DIMENSIONALITIES:
        k1_configs = sampler.generate_sobol_sparse(
            geometry, 1, dim, amplitude_ang, max_n=n_per_group, seed=seed
        )
        for index, config in enumerate(k1_configs):
            sample_id = f"ood_displacement__k1__{dim}__{index:03d}"
            entries.append((sample_id, config))

        k2_configs = sampler.generate_local_pair_modes(geometry, dim, amplitudes_ang=(amplitude_ang,))
        for index, config in enumerate(k2_configs):
            sample_id = f"ood_displacement__k2__{dim}__{index:03d}"
            entries.append((sample_id, config))
    return entries


# --------------------------------------------------------------------------
# Materialize + run SIESTA for a generated frozen-test entry set (reuses S3)
# --------------------------------------------------------------------------


def materialize_and_run(
    entries: list[tuple[str, sampler.Configuration]],
    *,
    output_root: Path,
    siesta_command: str,
    workers: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    structures_root = output_root / "structures"
    canonical_base_fdf = output_root / "_canonical_base.fdf"
    canonical_ang_base_fdf(pilot.DEFAULT_MATERIAL_FDF, canonical_base_fdf)

    pilot.materialize_structures(
        entries, structures_root=structures_root, canonical_base_fdf=canonical_base_fdf
    )

    try:
        run_manifest = run_derivative_siesta_references(
            stencil_root=output_root,
            source_dataset_root=pilot.DEFAULT_PSEUDO_SOURCE_ROOT,
            siesta_command=siesta_command,
            workers=workers,
            diagnostic_only=True,
        )
    except DerivativeSiestaReferenceError as exc:
        raise pilot.W90PilotError(str(exc)) from exc

    output_reference_root = Path(run_manifest["output_reference_root"])
    usable_ids = [
        row["sample_id"] for row in run_manifest["rows"] if row["status"] in {"ok", "staged", "skipped_existing"}
    ]
    hermiticity_rows = {sid: pilot.verify_hermiticity(output_reference_root / sid) for sid in usable_ids}
    orbital_rows = {sid: pilot.verify_orbital_dimension(output_reference_root / sid) for sid in usable_ids}
    return {
        "run_manifest": run_manifest,
        "output_reference_root": output_reference_root,
        "usable_ids": usable_ids,
        "hermiticity_rows": hermiticity_rows,
        "orbital_rows": orbital_rows,
    }


def samples_from_run(
    entries: list[tuple[str, sampler.Configuration]],
    run_result: dict[str, Any],
) -> list[s4.Sample]:
    """Build S4-compatible ``Sample`` objects for every usable, non-reference entry."""

    by_id = dict(entries)
    output_reference_root = run_result["output_reference_root"]
    samples: list[s4.Sample] = []
    for sample_id in run_result["usable_ids"]:
        config = by_id.get(sample_id)
        if config is None:
            continue
        reference_dir = output_reference_root / sample_id
        selection = choose_reference_matrix(reference_dir, require_positive_provenance=False)
        if not selection.ok or selection.path is None:
            continue
        samples.append(
            s4.Sample(
                sample_id=sample_id,
                family=str(config.metadata.get("family", "unknown")),
                dim=str(config.metadata.get("dimensionality", "unknown")),
                k=int(config.metadata.get("k", 0)),
                amplitude_ang=float(config.metadata.get("amplitude_ang", 0.0)),
                run_fdf=reference_dir / "RUN.fdf",
                reference_dir=reference_dir,
                reference_matrix=selection.path,
                max_displacement_ang=(
                    float(config.metadata["max_displacement_ang"])
                    if "max_displacement_ang" in config.metadata else None
                ),
            )
        )
    return samples


# --------------------------------------------------------------------------
# md_frozen: reuse already SIESTA-computed graphene MD frames, select by
# measured statistical inefficiency of configurational observables
# --------------------------------------------------------------------------


def gather_md_candidate_frames(md_results_root: Path = DEFAULT_MD_RESULTS_ROOT) -> dict[int, Path]:
    """frame_index -> graphene.TSHS path, deduplicated (first dataset seen wins)."""

    frames: dict[int, Path] = {}
    for dataset_dir in sorted(md_results_root.glob("MD_dataset*_*")):
        for run_dir in sorted(dataset_dir.glob("run_*")):
            siesta_root = run_dir / "siesta_hamiltonians"
            if not siesta_root.is_dir():
                continue
            for frame_dir in sorted(siesta_root.iterdir()):
                if not frame_dir.name.isdigit():
                    continue
                tshs_candidates = sorted(frame_dir.glob("*.TSHS"))
                if not tshs_candidates:
                    continue
                frames.setdefault(int(frame_dir.name), tshs_candidates[0])
    return frames


def _three_neighbor_vectors(positions_ang: np.ndarray, lattice_ang: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 3 nearest periodic images of atom 1 around atom 0 (sp2 coordination)."""

    shifts = sampler._SHIFTS @ lattice_ang
    candidates = positions_ang[1] - positions_ang[0] + shifts
    distances = np.linalg.norm(candidates, axis=1)
    order = np.argsort(distances)[:3]
    return candidates[order], distances[order]


def configurational_observables(tshs_path: Path) -> dict[str, float]:
    import sisl

    geometry = sisl.get_sile(str(tshs_path)).read_geometry()
    positions_ang = np.asarray(geometry.xyz, dtype=float)
    lattice_ang = np.asarray(geometry.lattice.cell, dtype=float)
    neighbor_vectors, distances = _three_neighbor_vectors(positions_ang, lattice_ang)

    bond_length_ang = float(np.mean(distances))
    unit_vectors = neighbor_vectors / distances[:, None]
    pairs = ((0, 1), (0, 2), (1, 2))
    angles_deg = [
        float(np.degrees(np.arccos(np.clip(np.dot(unit_vectors[a], unit_vectors[b]), -1.0, 1.0))))
        for a, b in pairs
    ]
    mean_bond_angle_deg = float(np.mean(angles_deg))

    normal = np.cross(neighbor_vectors[0], neighbor_vectors[1])
    normal_norm = float(np.linalg.norm(normal))
    z_hat = np.array([0.0, 0.0, 1.0])
    if normal_norm > 1e-12:
        pyramidalization_deg = float(
            np.degrees(np.arccos(np.clip(abs(float(np.dot(normal, z_hat))) / normal_norm, -1.0, 1.0)))
        )
    else:
        pyramidalization_deg = 0.0

    return {
        "bond_length_ang": bond_length_ang,
        "mean_bond_angle_deg": mean_bond_angle_deg,
        "pyramidalization_deg": pyramidalization_deg,
    }


def estimate_statistical_inefficiency(indices: list[int], values: list[float]) -> dict[str, Any]:
    """Empirical lag-binned autocorrelation over an irregularly-spaced index series.

    Returns the smallest index-lag at which the normalized autocorrelation
    first drops to <= 1/e (a standard correlation-length definition); if the
    series never decays below that threshold within the observed lag range,
    the largest available lag is returned instead (the conservative estimate
    the data can actually support -- never an assumed/default stride).
    """

    x = np.asarray(values, dtype=float) - np.mean(values)
    n = len(x)
    variance = float(np.mean(x**2)) if n else 0.0
    lag_products: dict[int, list[float]] = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            lag = indices[j] - indices[i]
            lag_products[lag].append(float(x[i] * x[j]))

    rho_by_lag: dict[int, float] = {}
    if variance > 0.0:
        for lag, products in lag_products.items():
            rho_by_lag[lag] = float(np.mean(products)) / variance

    threshold = 1.0 / np.e
    g: int | None = None
    for lag in sorted(rho_by_lag):
        if rho_by_lag[lag] <= threshold:
            g = lag
            break
    if g is None:
        max_lag = (max(indices) - min(indices)) if indices else 1
        g = max(1, max_lag)

    return {
        "g": int(max(1, g)),
        "variance": variance,
        "rho_by_lag": {str(k): v for k, v in sorted(rho_by_lag.items())},
        "n_points": n,
    }


def select_decorrelated_frames(
    frame_indices: list[int],
    *,
    g: int,
) -> list[int]:
    """Greedy selection keeping only frames spaced >= g apart (measured, not arbitrary)."""

    ordered = sorted(frame_indices)
    if not ordered:
        return []
    selected = [ordered[0]]
    for idx in ordered[1:]:
        if idx - selected[-1] >= g:
            selected.append(idx)
    return selected


def build_md_frozen_samples(
    *,
    output_root: Path,
    md_results_root: Path = DEFAULT_MD_RESULTS_ROOT,
) -> tuple[list[s4.Sample], dict[str, Any]]:
    candidate_frames = gather_md_candidate_frames(md_results_root)
    if not candidate_frames:
        return [], {"status": "no_md_frames_found", "md_results_root": str(md_results_root)}

    sorted_indices = sorted(candidate_frames)
    observables_by_frame = {idx: configurational_observables(candidate_frames[idx]) for idx in sorted_indices}

    inefficiencies = {
        key: estimate_statistical_inefficiency(
            sorted_indices, [observables_by_frame[idx][key] for idx in sorted_indices]
        )
        for key in ("bond_length_ang", "mean_bond_angle_deg", "pyramidalization_deg")
    }
    # Most conservative (slowest-decorrelating) observable sets the selection gap.
    g = max(diag["g"] for diag in inefficiencies.values())
    selected_indices = select_decorrelated_frames(sorted_indices, g=g)

    structures_root = output_root / "structures"
    canonical_base_fdf = output_root / "_canonical_base.fdf"
    canonical_ang_base_fdf(pilot.DEFAULT_MATERIAL_FDF, canonical_base_fdf)

    import sisl

    samples: list[s4.Sample] = []
    for idx in selected_indices:
        sample_id = f"md_frozen__{idx:03d}"
        tshs_path = candidate_frames[idx]
        geometry = sisl.get_sile(str(tshs_path)).read_geometry()
        positions_ang = [tuple(row) for row in np.asarray(geometry.xyz, dtype=float)]

        out_dir = structures_root / sample_id
        materialize_sample_fdf(
            canonical_base_fdf,
            out_dir / "RUN.fdf",
            positions_ang=positions_ang,
            single_point=True,
        )
        dest_matrix = out_dir / tshs_path.name
        if not dest_matrix.exists():
            dest_matrix.write_bytes(tshs_path.read_bytes())

        selection = choose_reference_matrix(out_dir, require_positive_provenance=False)
        if not selection.ok or selection.path is None:
            continue
        samples.append(
            s4.Sample(
                sample_id=sample_id,
                family="md_frozen",
                dim="md",
                k=0,
                amplitude_ang=float("nan"),
                run_fdf=out_dir / "RUN.fdf",
                reference_dir=out_dir,
                reference_matrix=selection.path,
            )
        )

    report = {
        "candidate_frame_count": len(candidate_frames),
        "candidate_frame_indices": sorted_indices,
        "statistical_inefficiency_by_observable": inefficiencies,
        "g_used": g,
        "selected_frame_indices": selected_indices,
        "selected_count": len(samples),
        "limitation": (
            "Only frames persisted with a SIESTA-computed Hamiltonian by the "
            "pre-existing MD comparison campaign are available (the rest of "
            "the underlying trajectory lived only in a since-cleaned-up "
            "workspace); g is estimated from these sparse, irregularly-spaced "
            "points -- a rough estimate, documented rather than hidden."
        ),
    }
    return samples, report


# --------------------------------------------------------------------------
# Leakage check: zero overlap between S3 training geometries and any frozen
# test's geometries (by rounded absolute-position hash, not just sample_id)
# --------------------------------------------------------------------------


def geometry_hash(positions_ang: list[tuple[float, float, float]] | np.ndarray, *, decimals: int = 6) -> str:
    array = np.round(np.asarray(positions_ang, dtype=float), decimals=decimals)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _positions_from_run_fdf(run_fdf: Path) -> np.ndarray:
    import sisl

    geometry = sisl.get_sile(str(run_fdf)).read_geometry()
    return np.asarray(geometry.xyz, dtype=float)


def check_no_leakage(
    training_samples: list[s4.Sample],
    frozen_samples_by_test: dict[str, list[s4.Sample]],
) -> dict[str, Any]:
    training_hashes = {
        geometry_hash(_positions_from_run_fdf(sample.run_fdf)): sample.sample_id for sample in training_samples
    }
    overlaps: dict[str, list[str]] = {}
    for test_name, samples in frozen_samples_by_test.items():
        hits = []
        for sample in samples:
            digest = geometry_hash(_positions_from_run_fdf(sample.run_fdf))
            if digest in training_hashes:
                hits.append((sample.sample_id, training_hashes[digest]))
        if hits:
            overlaps[test_name] = hits
    return {
        "training_geometry_count": len(training_hashes),
        "overlaps": overlaps,
        "leakage_free": not overlaps,
    }


# --------------------------------------------------------------------------
# Evaluation: every S4 model x every frozen test
# --------------------------------------------------------------------------


def load_s4_models(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("status") == "ok" and row.get("checkpoint")]


def frozen_test_split_id(samples: list[s4.Sample]) -> str:
    """Stable identity for *which* geometries make up a frozen test's sample set.

    Distinct from ``checkpoint``/``frozen_test_name``: if the frozen test is
    ever regenerated (different sample_ids), this changes, invalidating any
    cached prediction keyed on the old set even though the checkpoint and
    test *name* are unchanged.
    """

    ids = sorted(sample.sample_id for sample in samples)
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:16]


def cached_prediction_row(
    existing_by_key: dict[tuple[str, str, str, str, str, str], dict[str, str]],
    model_row: dict[str, str],
    frozen_test_name: str,
    checkpoint: Path,
    split_id: str,
) -> dict[str, Any] | None:
    """Reuse a prior ``per_model_frozen_test_metrics.csv`` row iff (checkpoint, split, test) all match.

    Avoids re-running the (expensive) ``predict_model_on_dataset.py`` subprocess
    when nothing that could change its output -- the checkpoint identity, the
    frozen-test's geometry set, or the test name -- has changed since the row
    was written.
    """

    key = (
        model_row["family"], model_row["dim"], model_row["k"],
        model_row["N_train"], model_row["seed"], frozen_test_name,
    )
    cached = existing_by_key.get(key)
    if cached is None or cached.get("status") != "ok":
        return None
    if cached.get("checkpoint") != str(checkpoint) or cached.get("split_id") != split_id:
        return None
    coerced = dict(cached)
    for field in _PER_MODEL_NUMERIC_FIELDS:
        value = coerced.get(field)
        if value not in (None, ""):
            try:
                coerced[field] = float(value)
            except ValueError:
                pass
    return coerced


def read_existing_eval_rows(csv_path: Path) -> dict[tuple[str, str, str, str, str, str], dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (r["training_family"], r["training_dim"], r["training_k"], r["training_n_train"], r["training_seed"], r["frozen_test"]): r
        for r in rows
    }


def evaluate_model_on_frozen_test(
    model_row: dict[str, str],
    frozen_test_name: str,
    frozen_samples: list[s4.Sample],
    *,
    output_root: Path,
    existing_by_key: dict[tuple[str, str, str, str, str, str]] | None = None,
    accelerator: str | None = None,
) -> dict[str, Any]:
    checkpoint = Path(model_row["checkpoint"])
    split_id = frozen_test_split_id(frozen_samples)
    if existing_by_key is not None:
        cached = cached_prediction_row(existing_by_key, model_row, frozen_test_name, checkpoint, split_id)
        if cached is not None:
            return cached
    run_tag = f"{model_row['family']}__{model_row['dim']}__k{model_row['k']}__n{model_row['N_train']}__seed{model_row['seed']}"
    run_dir = output_root / "evaluations" / frozen_test_name / run_tag
    row: dict[str, Any] = {
        "training_family": model_row["family"],
        "training_dim": model_row["dim"],
        "training_k": model_row["k"],
        "training_n_train": model_row["N_train"],
        "training_seed": model_row["seed"],
        "frozen_test": frozen_test_name,
        "checkpoint": str(checkpoint),
        "split_id": split_id,
    }
    if not checkpoint.exists() or not frozen_samples:
        row.update(status="skipped_missing_checkpoint_or_samples", n_evaluated=0)
        return row
    try:
        manifest_path = run_dir / "manifest.csv"
        s4.write_val_manifest(manifest_path, frozen_samples)
        # Default compute policy: prefer GPU when available (same preflight every
        # other training/inference call site in this pipeline already uses) rather
        # than a silent, undocumented CPU fallback for frozen-test inference.
        effective_accelerator = accelerator or s4.torch_backend_preflight()["effective_backend"]
        predicted_root = s4.run_prediction(checkpoint, manifest_path, run_dir / "prediction", effective_accelerator)
        row["backend"] = effective_accelerator
        metrics = s4.compute_validation_metrics(frozen_samples, predicted_root)
        row.update(metrics)
        row["status"] = "ok" if metrics["n_evaluated"] > 0 else "prediction_incomplete"
    except Exception as exc:  # noqa: BLE001 - a failed eval is a recorded row, not a crash
        row["status"] = f"error: {type(exc).__name__}: {exc}"[:500]
        row["n_evaluated"] = 0
    return row


def build_generalization_matrix(per_model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_model_rows:
        if row.get("status") == "ok":
            buckets[(row["training_family"], row["frozen_test"])].append(row)

    matrix_rows: list[dict[str, Any]] = []
    for (family, test_name), rows in sorted(buckets.items()):
        def _mean(key: str) -> float:
            values = [row[key] for row in rows if row.get(key) is not None]
            return float(np.mean(values)) if values else float("nan")

        matrix_rows.append(
            {
                "training_family": family,
                "frozen_test": test_name,
                "n_models": len(rows),
                "H_MAE_mean": _mean("H_MAE"),
                "H_RMSE_mean": _mean("H_RMSE"),
                "rel_Frob_mean": _mean("rel_Frob"),
                "spectral_err_mean": _mean("spectral_err"),
                "hermiticity_mean": _mean("hermiticity"),
            }
        )
    return matrix_rows


def write_csv(csv_path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s4-root", type=Path, default=DEFAULT_S4_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--siesta-command", default=DEFAULT_SIESTA_COMMAND)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-models", type=int, default=None, help="Debug/testing cap on models evaluated.")
    parser.add_argument("--skip-siesta", action="store_true", help="Debug: assume common/ood SIESTA already ran.")
    args = parser.parse_args()

    output_root: Path = args.output_root
    geometry = sampler.load_graphene_primitive()

    common_entries = build_common_displacement_configs(geometry)
    validation_entries = build_common_validation_configs(geometry)
    ood_entries = build_ood_displacement_configs(geometry)

    workers = args.workers if not args.skip_siesta else 0
    common_run = materialize_and_run(
        common_entries,
        output_root=output_root / "common_displacement",
        siesta_command=args.siesta_command,
        workers=workers,
    )
    validation_run = materialize_and_run(
        validation_entries,
        output_root=output_root / COMMON_VALIDATION_NAME,
        siesta_command=args.siesta_command,
        workers=workers,
    )
    ood_run = materialize_and_run(
        ood_entries,
        output_root=output_root / "ood_displacement",
        siesta_command=args.siesta_command,
        workers=workers,
    )

    common_samples = samples_from_run(common_entries, common_run)
    validation_samples = samples_from_run(validation_entries, validation_run)
    ood_samples = samples_from_run(ood_entries, ood_run)
    md_samples, md_report = build_md_frozen_samples(output_root=output_root / "md_frozen")
    write_json(output_root / "md_frozen_selection_report.json", md_report)

    frozen_samples_by_test = {
        "common_displacement": common_samples,
        "md_frozen": md_samples,
        "ood_displacement": ood_samples,
    }

    training_samples = s4.load_s3_samples()
    # Leakage check covers common_validation too, even though it is not a
    # frozen test -- it must still never overlap the training pool.
    leakage_report = check_no_leakage(
        training_samples, {**frozen_samples_by_test, COMMON_VALIDATION_NAME: validation_samples}
    )
    write_json(output_root / "leakage_check_report.json", leakage_report)
    if not leakage_report["leakage_free"]:
        print(f"[S5][ERROR] leakage detected: {leakage_report['overlaps']}", file=sys.stderr)
        return 2

    models = load_s4_models(args.s4_root / "learning_curves.csv")
    if args.max_models is not None:
        models = models[: max(0, int(args.max_models))]

    per_model_existing = read_existing_eval_rows(output_root / "per_model_frozen_test_metrics.csv")
    per_model_rows: list[dict[str, Any]] = []
    for model_row in models:
        for test_name in FROZEN_TEST_NAMES:
            eval_row = evaluate_model_on_frozen_test(
                model_row, test_name, frozen_samples_by_test[test_name], output_root=output_root,
                existing_by_key=per_model_existing,
            )
            per_model_rows.append(eval_row)
            write_csv(output_root / "per_model_frozen_test_metrics.csv", per_model_rows, PER_MODEL_FIELDNAMES)

    matrix_rows = build_generalization_matrix(per_model_rows)
    write_csv(output_root / "generalization_matrix.csv", matrix_rows, GENERALIZATION_MATRIX_FIELDNAMES)

    # common_validation: same evaluation machinery, written to its own files
    # so it can never be mistaken for (or accidentally folded into) the
    # frozen-test generalization matrix above.
    validation_existing = read_existing_eval_rows(output_root / "per_model_common_validation_metrics.csv")
    validation_rows: list[dict[str, Any]] = []
    for model_row in models:
        eval_row = evaluate_model_on_frozen_test(
            model_row, COMMON_VALIDATION_NAME, validation_samples, output_root=output_root,
            existing_by_key=validation_existing,
        )
        validation_rows.append(eval_row)
        write_csv(output_root / "per_model_common_validation_metrics.csv", validation_rows, PER_MODEL_FIELDNAMES)
    validation_matrix_rows = build_generalization_matrix(validation_rows)
    write_csv(output_root / "common_validation_matrix.csv", validation_matrix_rows, GENERALIZATION_MATRIX_FIELDNAMES)

    summary = {
        "task": "DATASET-DESIGN-W90-001-S5",
        "common_displacement_samples": len(common_samples),
        "common_validation_samples": len(validation_samples),
        "ood_displacement_samples": len(ood_samples),
        "md_frozen_samples": len(md_samples),
        "models_evaluated": len(models),
        "leakage_free": leakage_report["leakage_free"],
        "generalization_matrix_rows": len(matrix_rows),
        "common_validation_matrix_rows": len(validation_matrix_rows),
    }
    write_json(output_root / "s5_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
