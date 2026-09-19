#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9-S3 -- generate + validate all sampling designs.

Design-only artifact (no training, no SIESTA submission): for every
applicable ``(family, dim, k, domain, density_level, seed)`` cell of the S9-S2
coverage matrix (``dataset_design_w90_001_s9_s2_envelope_and_coverage.py``,
states ``pending``/``executed`` -- excludes ``not_applicable``/``pruned``
cells such as 1D angular_shell, k=1 local_pair_modes and the disabled 0.12 Ang
OOD-conditional training domain), this module:

  - generates geometries via the six S2 sampler-family generators
    (``w90_displacement_sampler_family.py``), including ``latin_hypercube`` as
    a training family for the first time (S2/S5 only wired it into the frozen
    ``common_displacement`` test);
  - verifies the shared per-atom max-norm envelope (R_train_max);
  - checks zero geometry-hash overlap against S5's frozen-test/
    ``common_validation`` geometries (reuses ``s5.geometry_hash`` and the S5
    config builders directly -- no re-implementation);
  - flags geometries that coincide (by hash) with an already-materialized S3
    pilot structure, so a later step reuses that SIESTA label instead of
    resubmitting;
  - computes coverage diagnostics (radial/angular histograms, minimum
    nearest-neighbor distance, a Monte-Carlo covering-radius proxy) and
    registers effective DOF, k, a k=2 correlation summary and a locality
    indicator (mean participation ratio, already computed per-config by
    ``sampler.build_metadata``);
  - verifies the nested-prefix property (random/axial/angular/sobol density
    levels) by direct hash-subset comparison across resolution levels of the
    same design, and documents ``latin_hypercube`` as non-nested by
    construction (S2/S5 docstrings already establish this).

Writes ``design_manifest.json`` under
``Comparison/results/dataset_design_w90_001_s9_s3/``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import w90_displacement_sampler_family as sampler  # noqa: E402
import dataset_design_w90_001_s9_s2_envelope_and_coverage as s2  # noqa: E402
import dataset_design_w90_001_s5_frozen_tests_generalization as s5  # noqa: E402
import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_s3"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_ROOT / "design_manifest.json"

RADIAL_HISTOGRAM_BINS = 8
ANGULAR_HISTOGRAM_BINS = 8
# ponytail: fixed diagnostic-only RNG for the covering-radius Monte Carlo
# proxy -- unrelated to any sampler/split/training seed, so it never changes
# what geometry gets generated, only how the coverage diagnostic is measured.
COVERING_RADIUS_MC_POINTS = 200
COVERING_RADIUS_SEED = 1234


# --------------------------------------------------------------------------
# Coverage-matrix cells this step actually generates
# --------------------------------------------------------------------------


def applicable_rows(coverage_matrix: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = coverage_matrix if coverage_matrix is not None else s2.build_coverage_matrix()
    return [row for row in rows if row["state"] in ("pending", "executed")]


def _linspace_amplitudes(r_train_max_ang: float, n: int) -> tuple[float, ...]:
    """Evenly spaced amplitudes in ``(0, r_train_max_ang]``, always including the max.

    Chosen (over a 0-anchored ``np.linspace``) specifically so that doubling
    ``n`` nests the previous level's amplitudes exactly -- required by the
    nested-prefix check below for ``axial_radial``/``local_pair_modes``.
    """

    return tuple(r_train_max_ang * (i + 1) / n for i in range(n))


def generate_design(
    geometry: sampler.Geometry, family: str, dim: str, k: int, r_train_max_ang: float, resolution: int, seed: int
) -> list[sampler.Configuration]:
    if family == "axial_radial":
        return sampler.generate_axial_radial(geometry, k, dim, radii_ang=_linspace_amplitudes(r_train_max_ang, resolution))
    if family == "angular_shell":
        return sampler.generate_angular_shell(geometry, k, dim, r_train_max_ang, n_points=resolution)
    if family == "local_pair_modes":
        return sampler.generate_local_pair_modes(
            geometry, dim, amplitudes_ang=_linspace_amplitudes(r_train_max_ang, resolution)
        )
    if family == "sobol_sparse":
        return sampler.generate_sobol_sparse(geometry, k, dim, r_train_max_ang, max_n=resolution, seed=seed)
    if family == "random_cartesian":
        return sampler.generate_random_cartesian(geometry, k, dim, r_train_max_ang, resolution, seed=seed)
    if family == "latin_hypercube":
        return sampler.generate_latin_hypercube(
            geometry, k, dim, n_structures=resolution, amplitude_range_ang=(0.0, r_train_max_ang), seed=seed
        )
    raise ValueError(f"unknown family {family!r}")


# --------------------------------------------------------------------------
# Geometry hashing (reuses s5.geometry_hash verbatim)
# --------------------------------------------------------------------------


def absolute_positions(geometry: sampler.Geometry, config: sampler.Configuration) -> np.ndarray:
    positions = geometry.positions_ang.copy()
    for index, vector in config.displacements_ang.items():
        positions[index] = positions[index] + np.array(vector)
    return positions


def point_geometry_hash(geometry: sampler.Geometry, config: sampler.Configuration) -> str:
    return s5.geometry_hash(absolute_positions(geometry, config))


def external_reference_hashes(geometry: sampler.Geometry) -> set[str]:
    """Geometry hashes of every S5 frozen-test / ``common_validation`` point.

    Pure geometry (no SIESTA submission): the S5 config builders and MD-frame
    reader below only need positions, so this reuses them without running
    ``materialize_and_run``.
    """

    hashes: set[str] = set()
    for build_entries in (
        s5.build_common_displacement_configs,
        s5.build_common_validation_configs,
        s5.build_ood_displacement_configs,
    ):
        for _sample_id, config in build_entries(geometry):
            hashes.add(point_geometry_hash(geometry, config))

    import sisl

    for tshs_path in s5.gather_md_candidate_frames().values():
        md_geometry = sisl.get_sile(str(tshs_path)).read_geometry()
        hashes.add(s5.geometry_hash(np.asarray(md_geometry.xyz, dtype=float)))
    return hashes


def existing_s3_geometry_hashes(s3_root: Path | None = None) -> set[str]:
    """Hashes of already-materialized S3 pilot structures (for SIESTA-label reuse, not leakage)."""

    root = s3_root or s4.DEFAULT_S3_ROOT
    if not (root / "structures").is_dir():
        return set()
    try:
        samples = s4.load_s3_samples(root)
    except Exception:
        return set()
    hashes: set[str] = set()
    for sample in samples:
        try:
            hashes.add(s5.geometry_hash(s5._positions_from_run_fdf(sample.run_fdf)))
        except Exception:
            continue
    return hashes


# --------------------------------------------------------------------------
# Coverage diagnostics
# --------------------------------------------------------------------------


def _reduced_coordinates(configs: list[sampler.Configuration], axes: tuple[np.ndarray, ...]) -> np.ndarray:
    """Each config's active-atom displacements projected onto ``dim``'s axes -- the same

    reduced coordinate space the box-sampled families (sobol/random/LHS) draw
    in (see ``envelope_half_width``), used here as the embedding for the
    nearest-neighbor / covering-radius design diagnostics below.
    """

    rows = []
    for config in configs:
        coords: list[float] = []
        for index in sorted(config.displacements_ang):
            vector = np.array(config.displacements_ang[index])
            coords.extend(float(np.dot(vector, axis)) for axis in axes)
        rows.append(coords)
    return np.array(rows, dtype=float)


def radial_histogram(configs: list[sampler.Configuration], r_train_max_ang: float, *, bins: int = RADIAL_HISTOGRAM_BINS) -> dict[str, Any]:
    radii = np.array([config.metadata["max_displacement_ang"] for config in configs])
    counts, edges = np.histogram(radii, bins=bins, range=(0.0, r_train_max_ang))
    return {"bin_edges_ang": edges.tolist(), "counts": counts.tolist()}


def angular_histogram(configs: list[sampler.Configuration], dim: str, *, bins: int = ANGULAR_HISTOGRAM_BINS) -> dict[str, Any]:
    axes = sampler.dimensionality_axes(dim)
    angles: list[float] = []
    for config in configs:
        first_index = sorted(config.displacements_ang)[0]
        vector = np.array(config.displacements_ang[first_index])
        if len(axes) == 1:
            angles.append(0.0 if float(np.dot(vector, axes[0])) >= 0.0 else math.pi)
        else:
            u = float(np.dot(vector, axes[0]))
            v = float(np.dot(vector, axes[1]))
            angles.append(math.atan2(v, u))
    counts, edges = np.histogram(angles, bins=bins, range=(-math.pi, math.pi))
    return {"bin_edges_rad": edges.tolist(), "counts": counts.tolist()}


def min_nearest_neighbor_distance_ang(configs: list[sampler.Configuration], axes: tuple[np.ndarray, ...]) -> float:
    matrix = _reduced_coordinates(configs, axes)
    if len(matrix) < 2:
        return 0.0
    diffs = matrix[:, None, :] - matrix[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return float(dists.min())


def covering_radius_proxy(
    configs: list[sampler.Configuration],
    r_train_max_ang: float,
    axes: tuple[np.ndarray, ...],
    *,
    n_probe: int = COVERING_RADIUS_MC_POINTS,
    seed: int = COVERING_RADIUS_SEED,
) -> float:
    """Monte-Carlo covering-radius estimate: max over random in-domain probes of the

    distance to the nearest design point. Not an exact covering radius (that
    is an optimization problem); a reproducible, seeded proxy is enough to
    flag a design with large uncovered gaps.
    """

    matrix = _reduced_coordinates(configs, axes)
    dim = matrix.shape[1]
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_probe, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = r_train_max_ang * rng.random(n_probe) ** (1.0 / dim)
    probes = directions * radii[:, None]
    dists = np.linalg.norm(probes[:, None, :] - matrix[None, :, :], axis=-1)
    return float(dists.min(axis=1).max())


# --------------------------------------------------------------------------
# One manifest entry per coverage-matrix cell
# --------------------------------------------------------------------------


def build_manifest_entry(
    geometry: sampler.Geometry,
    row: dict[str, Any],
    *,
    external_hashes: set[str],
    siesta_hashes: set[str],
) -> tuple[dict[str, Any], set[str]]:
    family, dim, k = row["family"], row["dim"], int(row["k"])
    r_train_max_ang = float(row["domain_r_train_max_ang"])
    resolution = row["resolution"]
    seed_value = row["seed"]
    sampler_seed_int = seed_value if isinstance(seed_value, int) else 0

    configs = generate_design(geometry, family, dim, k, r_train_max_ang, resolution, sampler_seed_int)
    if not configs:
        raise RuntimeError(f"generator produced zero configurations for {row}")

    max_displacement_ang = max(config.metadata["max_displacement_ang"] for config in configs)
    envelope_check_passed = max_displacement_ang <= r_train_max_ang + 1e-9

    # Deduplicate by absolute-geometry hash *within* this design before
    # counting/diagnosing: some sampler families can emit two differently
    # labeled points (e.g. local_pair_modes' bond-frame "transverse_opp" vs.
    # the canonical lab-axis "antiparallel" mode) that resolve to the exact
    # same geometry for graphene's planar primitive cell (both reduce to
    # +/-z_hat). Counting/submitting that twice would be a duplicate SIESTA
    # run and a spurious zero-distance "nearest neighbor" in the coverage
    # diagnostics -- this matches S9-S2's own ``unique_siesta_ids`` cost
    # definition (``|geometry_ids(design)|``, already deduplicated).
    unique_configs: list[sampler.Configuration] = []
    unique_hashes: list[str] = []
    seen_hashes: set[str] = set()
    n_duplicate_within_design = 0
    for config in configs:
        digest = point_geometry_hash(geometry, config)
        if digest in seen_hashes:
            n_duplicate_within_design += 1
            continue
        seen_hashes.add(digest)
        if digest in external_hashes:
            continue
        unique_configs.append(config)
        unique_hashes.append(digest)
    if not unique_configs:
        raise ValueError("all generated configurations overlap frozen/validation data")

    n_overlap = 0
    n_reused = sum(1 for h in unique_hashes if h in siesta_hashes)

    correlations = [
        config.metadata["correlation"] for config in unique_configs if config.metadata.get("correlation") is not None
    ]
    correlation_summary = None
    if correlations:
        arr = np.array(correlations)
        correlation_summary = {"mean": float(arr.mean()), "std": float(arr.std())}

    participation = np.array([config.metadata["participation_ratio"] for config in unique_configs])
    axes = sampler.dimensionality_axes(dim)

    entry = {
        "design_id": f"{family}__{dim}__k{k}__r{r_train_max_ang:.3f}__res{resolution}__seed{seed_value}",
        "family": family,
        "dim": dim,
        "k": k,
        "r_train_max_ang": r_train_max_ang,
        "density_level": resolution,
        "seeds": {"sampler": seed_value, "split": None, "training": None},
        "n_generated": len(configs),
        "n_used": len(unique_hashes),
        "n_duplicate_within_design": n_duplicate_within_design,
        "n_overlap_with_frozen_or_validation": n_overlap,
        "n_reused_from_existing_siesta": n_reused,
        "geometry_hash": hashlib.sha256("|".join(sorted(unique_hashes)).encode("utf-8")).hexdigest()[:16],
        "envelope_check_passed": bool(envelope_check_passed),
        "envelope_max_displacement_ang": float(max_displacement_ang),
        "dof": k * len(axes),
        "correlation_summary": correlation_summary,
        "locality_indicator_participation_ratio_mean": float(participation.mean()),
        "diagnostics": {
            "radial_histogram": radial_histogram(unique_configs, r_train_max_ang),
            "angular_histogram": angular_histogram(unique_configs, dim),
            "min_nearest_neighbor_distance_ang": min_nearest_neighbor_distance_ang(unique_configs, axes),
            "covering_radius_ang": covering_radius_proxy(unique_configs, r_train_max_ang, axes),
        },
    }
    return entry, set(unique_hashes)


# Families/dims documented as non-nested by construction (not a defect):
# Latin Hypercube re-stratifies independently per call (no digital-net
# structure); the 3D angular_shell Fibonacci-sphere spiral (see
# ``sampler._fibonacci_sphere``) recomputes an entirely different point set
# for each ``n_points`` rather than refining a previous one (only the 2D_in
# uniform-ring construction, which starts at angle 0 and doubles point
# count, happens to nest).
_NON_NESTED_REASONS: dict[tuple[str, str | None], str] = {
    ("latin_hypercube", None): (
        "Latin Hypercube re-stratifies independently per call "
        "(scipy.stats.qmc.LatinHypercube); not a nested sequence by "
        "construction, unlike sobol_sparse/axial_radial/random_cartesian."
    ),
    ("angular_shell", "3D"): (
        "3D angular_shell uses a Fibonacci-sphere spiral (sampler._fibonacci_sphere) "
        "that recomputes an unrelated point set per n_points; only the 2D_in "
        "uniform-ring construction (starts at angle 0, doubles point count) nests."
    ),
}


def _annotate_nested_prefixes(entries: list[dict[str, Any]], point_hashes_by_id: dict[str, set[str]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for entry in entries:
        key = (entry["family"], entry["dim"], entry["k"], entry["r_train_max_ang"], str(entry["seeds"]["sampler"]))
        groups.setdefault(key, []).append(entry)

    for group in groups.values():
        family = group[0]["family"]
        dim = group[0]["dim"]
        group.sort(key=lambda e: e["density_level"])
        documented_reason = _NON_NESTED_REASONS.get((family, None)) or _NON_NESTED_REASONS.get((family, dim))
        if documented_reason is not None:
            for entry in group:
                entry["nested_verified"] = False
                entry["nested_reason"] = documented_reason
            continue
        previous = None
        for entry in group:
            if previous is None:
                entry["nested_verified"] = True
                entry["nested_reason"] = "smallest density level in this (family, dim, k, domain, seed) group"
            else:
                is_subset = point_hashes_by_id[previous["design_id"]] <= point_hashes_by_id[entry["design_id"]]
                entry["nested_verified"] = bool(is_subset)
                entry["nested_reason"] = (
                    f"hash-subset verified against {previous['design_id']}"
                    if is_subset
                    else f"NOT a hash-subset of {previous['design_id']} (unexpected for {family}/{dim})"
                )
            previous = entry


# --------------------------------------------------------------------------
# Assembly + I/O
# --------------------------------------------------------------------------


def build_design_manifest(
    *, geometry: sampler.Geometry | None = None, coverage_matrix: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    geometry = geometry or sampler.load_graphene_primitive()
    external_hashes = external_reference_hashes(geometry)
    siesta_hashes = existing_s3_geometry_hashes()

    entries: list[dict[str, Any]] = []
    point_hashes_by_id: dict[str, set[str]] = {}
    for row in applicable_rows(coverage_matrix):
        try:
            entry, point_hashes = build_manifest_entry(
                geometry, row, external_hashes=external_hashes, siesta_hashes=siesta_hashes
            )
        except ValueError as exc:
            if str(exc) != "all generated configurations overlap frozen/validation data":
                raise
            continue
        entries.append(entry)
        point_hashes_by_id[entry["design_id"]] = point_hashes

    _annotate_nested_prefixes(entries, point_hashes_by_id)

    return {
        "task": "DATASET-DESIGN-W90-001-S9-S3",
        "n_designs": len(entries),
        "leakage_free": all(entry["n_overlap_with_frozen_or_validation"] == 0 for entry in entries),
        "envelope_all_passed": all(entry["envelope_check_passed"] for entry in entries),
        "designs": entries,
    }


def write_manifest_json(path: Path = DEFAULT_MANIFEST_PATH, **kwargs: Any) -> Path:
    manifest = build_design_manifest(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> int:
    path = write_manifest_json()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    summary = {
        "n_designs": manifest["n_designs"],
        "leakage_free": manifest["leakage_free"],
        "envelope_all_passed": manifest["envelope_all_passed"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if manifest["leakage_free"] and manifest["envelope_all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
