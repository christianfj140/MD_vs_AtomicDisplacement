#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9-S4 -- pilot screening (N in {8,16,32,64}, 2 seeds).

Trains Graph2Mat on every S9-S3 design-manifest entry (
``dataset_design_w90_001_s9_s3_design_generation.py``) that is (a) in the
pilot density range N in {8, 16, 32, 64} and (b) *fully* backed by real
SIESTA references already materialized by the S3 pilot campaign
(``run_w90_displacement_siesta_pilot.py``) -- matched to the manifest's own
geometries by geometry hash, reusing ``s3.generate_design``/
``s3.point_geometry_hash`` verbatim rather than re-deriving them.

S9-S3 is a design-only step (no SIESTA submission for newly-generated
geometries); most of the 792-cell manifest therefore has zero real DFT
reference and cannot be trained on without a new DFT campaign, which is out
of scope here. Those cells are recorded in ``pilot_pruned.json`` with reason
``budget`` (no SIESTA reference data materialized for this pilot's compute
budget) rather than silently skipped -- this is exactly the situation
S9-S3's ``n_reused_from_existing_siesta`` field was added to flag.

Two seeds per surviving design vary Graph2Mat's own training seed (weight
init / optimizer stochasticity) -- a surviving design's training pool
already equals its full generated point set (N_train = design's N), so
there is no held-out split to vary. Reuses S4's ``build_graph2mat_config``/
``run_graph2mat_training``/``run_prediction``/``compute_validation_metrics``
verbatim (same architecture/hyperparameters, S4 fairness requirement).

Evaluation set 1 ("development set"): S5's ``common_validation`` frozen
pool, reused verbatim (``s5.build_common_validation_configs`` +
``s5.materialize_and_run`` + ``s5.samples_from_run``; already SIESTA-computed,
the call is a fast resubmission-skip). Also used as the Graph2Mat trainer's
own validation split during training, so every design is monitored against
the same distribution.

Evaluation set 2: the 6 pre-reserved test amplitudes from S9-S2
(``s2.TEST_AMPLITUDES_ANG`` = 0.01/0.03/0.05/0.08/0.10/0.12 Ang). None of
S5's existing frozen tests cover all six at a geometry disjoint from the S3
training pool, so this step builds one small (k in {1,2} x 4 dims, 1 point
each) real-SIESTA sample set per amplitude, using ``generate_sobol_sparse``
at both k=1 and k=2 with a seed reserved for this step
(``TEST_AMPLITUDE_SEED``, disjoint from S2/S3's ``PILOT_SEED``, S5's
``COMMON_SEED``/``COMMON_VALIDATION_SEED``) -- new SIESTA submissions via
``s5.materialize_and_run`` (idempotent/resumable, same as every other step).
S5's own ``ood_displacement`` uses ``generate_local_pair_modes`` for k=2,
which is deliberately *not* reused here: that generator is deterministic
given (dim, amplitude), so at the training-domain amplitudes (0.03/0.08,
already used by the S3 pool's own local_pair_modes family) it reproduces
training geometries exactly -- a real leakage hit caught during development
of this step (see ``build_test_amplitude_configs``).

Per (design, seed, evaluation-set) row is written to ``run_metrics.csv`` via
``s5.evaluate_model_on_frozen_test`` (identical metric definitions and
resume-by-cache behavior as S5). Pareto dominance (H-MAE vs the design's
``unique_siesta_ids`` cost, S9-S2's cost schema) and physical gates
(envelope, convergence) are checked afterward and written to
``pilot_pruned.json`` with a reason from the controlled vocabulary
``{dominated, envelope_fail, convergence_fail, budget}``. This step never
uses S5's frozen ``common_displacement``/``ood_displacement``/``md_frozen``
tests (the reserved final tests) for pruning -- only the development set and
the six pre-reserved test amplitudes built above.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import sys
import threading
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import w90_displacement_sampler_family as sampler  # noqa: E402
import dataset_design_w90_001_s9_s2_envelope_and_coverage as s2  # noqa: E402
import dataset_design_w90_001_s9_s3_design_generation as s3  # noqa: E402
import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402
import dataset_design_w90_001_s5_frozen_tests_generalization as s5  # noqa: E402
import run_w90_displacement_siesta_pilot as pilot  # noqa: E402

DEFAULT_MANIFEST_PATH = s3.DEFAULT_MANIFEST_PATH
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_s4"
DEFAULT_S5_ROOT = s5.DEFAULT_OUTPUT_ROOT

PILOT_DENSITY_LEVELS: tuple[int, ...] = (8, 16, 32, 64)
# Screening budget ceiling on the REALIZED training-pool size (`entry["n_used"]`),
# not on `entry["density_level"]`. `density_level` is each family's own
# resolution-axis value (e.g. axial_radial's n_radial_shells in {1,2,4},
# sobol_sparse's max_n in {16,64,256}) -- scales that are not comparable
# across families. Filtering pilot eligibility on `density_level in
# PILOT_DENSITY_LEVELS` (the pre-fix behavior) silently zeroed out every
# axial_radial/local_pair_modes cell (whose density_level never equals 8, 16,
# 32, or 64) while admitting angular_shell at all 3 of its own levels -- an
# accidental numeric coincidence, not a deliberate per-family screening tier.
# `n_used` is family-agnostic (always "points actually available to train
# on"), so bounding it decouples the pilot's N budget from each family's
# resolution axis and lets every applicable family contribute multiple
# resolution levels within the same N ceiling.
PILOT_MAX_N: int = max(PILOT_DENSITY_LEVELS)
TRAINING_SEEDS: tuple[int, ...] = (0, 1)
TEST_AMPLITUDES_ANG: tuple[float, ...] = s2.TEST_AMPLITUDES_ANG
# Disjoint from S2/S3's PILOT_SEED=0 and S5's COMMON_SEED=42/COMMON_VALIDATION_SEED=41.
TEST_AMPLITUDE_SEED = 44
TEST_AMPLITUDE_N_PER_GROUP = 1
DEVELOPMENT_SET_NAME = "development_set"

PRUNE_REASONS = ("dominated", "envelope_fail", "convergence_fail", "budget")

RUN_METRICS_FIELDNAMES = [
    "design_id", "family", "dim", "k", "N_train", "domain", "density", "seed",
    "test_amplitude", "ood_status", "H_MAE", "H_RMSE", "rel_Frob", "spectral_err",
    "hermiticity", "n_evaluated", "status", "split_id", "checkpoint", "backend",
    "siesta_cost",
]


# --------------------------------------------------------------------------
# Candidate selection: manifest entries in the pilot density range that are
# fully backed by real, already-materialized S3 SIESTA references.
# --------------------------------------------------------------------------


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return s3.build_design_manifest()


def build_s3_hash_index(s3_pool: list[s4.Sample]) -> dict[str, s4.Sample]:
    return {s5.geometry_hash(s5._positions_from_run_fdf(sample.run_fdf)): sample for sample in s3_pool}


def matched_training_samples(
    geometry: sampler.Geometry, entry: dict[str, Any], hash_to_sample: dict[str, s4.Sample],
    external_hashes: set[str] | None = None,
) -> list[s4.Sample]:
    """Real S3 samples matching ``entry``'s own generated geometries, by hash.

    Recomputes the same unique, deduplicated hash set S9-S3's
    ``build_manifest_entry`` computed (not just trusting the cached
    ``n_reused_from_existing_siesta`` count), so a mismatch would show up as
    a shorter list rather than silently fabricated training data.
    """

    family, dim, k = entry["family"], entry["dim"], int(entry["k"])
    r_train_max_ang = float(entry["r_train_max_ang"])
    resolution = entry["density_level"]
    seed_value = entry["seeds"]["sampler"]
    sampler_seed_int = seed_value if isinstance(seed_value, int) else 0

    configs = s3.generate_design(geometry, family, dim, k, r_train_max_ang, resolution, sampler_seed_int)
    external_hashes = external_hashes if external_hashes is not None else s3.external_reference_hashes(geometry)
    seen: set[str] = set()
    samples: list[s4.Sample] = []
    for config in configs:
        digest = s3.point_geometry_hash(geometry, config)
        if digest in seen:
            continue
        seen.add(digest)
        if digest in external_hashes:
            continue
        sample = hash_to_sample.get(digest)
        if sample is not None:
            samples.append(sample)
    return samples


def select_candidates(
    manifest: dict[str, Any], geometry: sampler.Geometry, hash_to_sample: dict[str, s4.Sample]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (trainable, budget_pruned) -- pilot-density entries only."""

    trainable: list[dict[str, Any]] = []
    pruned: list[dict[str, Any]] = []
    external_hashes = s3.external_reference_hashes(geometry)
    for entry in manifest["designs"]:
        if not (1 <= int(entry["n_used"]) <= PILOT_MAX_N):
            continue
        if not entry.get("envelope_check_passed", True):
            pruned.append(_prune_record(entry, "envelope_fail", "manifest envelope_check_passed is False"))
            continue
        if entry["n_reused_from_existing_siesta"] != entry["n_used"]:
            pruned.append(_prune_record(
                entry, "budget",
                f"only {entry['n_reused_from_existing_siesta']}/{entry['n_used']} points were SIESTA-backed in the manifest",
            ))
            continue
        matched = matched_training_samples(geometry, entry, hash_to_sample, external_hashes)
        if len(matched) < entry["n_used"]:
            pruned.append(
                _prune_record(
                    entry, "budget",
                    f"only {len(matched)}/{entry['n_used']} points have real SIESTA references",
                )
            )
            continue
        entry = dict(entry)
        entry["_matched_samples"] = matched
        trainable.append(entry)
    return trainable, pruned


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def filter_designs(
    entries: list[dict[str, Any]], *, families: list[str] | None = None, dims: list[str] | None = None,
    k_values: list[int] | None = None, r_train_max: list[float] | None = None,
    density_levels: list[int] | None = None,
) -> list[dict[str, Any]]:
    """DATASET-DESIGN-W90-001-S9-S6: UI-driven subset of the pilot-density manifest
    cells, applied before training so Pilot only trains what the user selected."""

    def keep(entry: dict[str, Any]) -> bool:
        if families and entry["family"] not in families:
            return False
        if dims and entry["dim"] not in dims:
            return False
        if k_values and int(entry["k"]) not in k_values:
            return False
        if r_train_max and not any(abs(float(entry["r_train_max_ang"]) - r) < 1e-9 for r in r_train_max):
            return False
        if density_levels and int(entry["density_level"]) not in density_levels:
            return False
        return True

    return [entry for entry in entries if keep(entry)]


def _prune_record(entry: dict[str, Any], reason: str, detail: str) -> dict[str, Any]:
    assert reason in PRUNE_REASONS
    return {
        "design_id": entry["design_id"],
        "family": entry["family"],
        "dim": entry["dim"],
        "k": entry["k"],
        "domain_r_train_max_ang": entry["r_train_max_ang"],
        "density_level": entry["density_level"],
        "reason": reason,
        "detail": detail,
    }


# --------------------------------------------------------------------------
# Development set + test-amplitude sweep (real SIESTA, reuses S5 machinery)
# --------------------------------------------------------------------------


def build_test_amplitude_configs(
    geometry: sampler.Geometry, amplitude_ang: float, *, seed: int, n_per_group: int = TEST_AMPLITUDE_N_PER_GROUP
) -> list[tuple[str, sampler.Configuration]]:
    """sobol_sparse at k in {1,2}, seeded -- *not* S5 ood_displacement's local_pair_modes(k=2)

    construction: that generator is deterministic given (dim, amplitude), so
    at the training-domain amplitudes (0.03/0.08, already used by the S3
    pilot pool's own local_pair_modes family) it reproduces the exact same
    geometries as training data -- a real leakage hit caught by
    ``check_no_leakage`` during development of this step. sobol_sparse is
    seed-controlled at any k, so a seed reserved for this step
    (``TEST_AMPLITUDE_SEED``) keeps every amplitude's test points disjoint
    from the training pool.
    """

    entries: list[tuple[str, sampler.Configuration]] = []
    for dim in sampler.DIMENSIONALITIES:
        for k in (1, 2):
            configs = sampler.generate_sobol_sparse(geometry, k, dim, amplitude_ang, max_n=n_per_group, seed=seed)
            for index, config in enumerate(configs):
                entries.append((f"test_amp{amplitude_ang:.2f}__k{k}__{dim}__{index:03d}", config))
    return entries


def build_evaluation_sets(
    geometry: sampler.Geometry, *, output_root: Path, siesta_command: str, workers: int
) -> dict[str, list[s4.Sample]]:
    sets: dict[str, list[s4.Sample]] = {}

    validation_entries = s5.build_common_validation_configs(geometry)
    validation_run = s5.materialize_and_run(
        validation_entries, output_root=DEFAULT_S5_ROOT / s5.COMMON_VALIDATION_NAME,
        siesta_command=siesta_command, workers=workers,
    )
    sets[DEVELOPMENT_SET_NAME] = s5.samples_from_run(validation_entries, validation_run)

    for amplitude in TEST_AMPLITUDES_ANG:
        entries = build_test_amplitude_configs(geometry, amplitude, seed=TEST_AMPLITUDE_SEED)
        run_result = s5.materialize_and_run(
            entries, output_root=output_root / f"test_amplitude_{amplitude:.2f}",
            siesta_command=siesta_command, workers=workers,
        )
        sets[f"{amplitude:.2f}"] = s5.samples_from_run(entries, run_result)
    return sets


def effective_ood_label(r_train_max_ang: float, samples: list[s4.Sample]) -> str:
    """Batch-level OOD label from each sample's own realized displacement.

    Reduces ``s2.effective_ood_status`` to a single CSV-safe label:
    "in_domain"/"ood" when every sample in the batch agrees, "mixed" when a
    nominally-single-amplitude batch actually straddles the domain boundary
    (falls back to "unknown" if no sample carries ``max_displacement_ang``,
    e.g. stale cached data from before this field existed).
    """

    values = [s.max_displacement_ang for s in samples if s.max_displacement_ang is not None]
    return s2.effective_ood_status(r_train_max_ang, values)["status"]


# --------------------------------------------------------------------------
# Training (reuses S4 verbatim)
# --------------------------------------------------------------------------


def train_one_design(entry: dict[str, Any], seed: int, dev_samples: list[s4.Sample], *, output_root: Path) -> dict[str, Any]:
    matched: list[s4.Sample] = entry["_matched_samples"]
    run_name = f"{entry['design_id']}__seed{seed}"
    run_dir = output_root / "runs" / run_name

    backend_info = s4.torch_backend_preflight()
    accelerator = backend_info["effective_backend"]

    model_row: dict[str, Any] = {
        "family": entry["family"], "dim": entry["dim"], "k": str(entry["k"]),
        "N_train": str(entry["n_used"]), "seed": str(seed),
    }
    existing_checkpoints = sorted(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime) if run_dir.exists() else []
    try:
        if existing_checkpoints:
            checkpoint = existing_checkpoints[-1]
        else:
            config_path = s4.build_graph2mat_config(
                run_dir, matched, dev_samples, run_name=run_name, accelerator=accelerator, training_seed=seed,
            )
            checkpoint = s4.run_graph2mat_training(config_path, run_dir)
        model_row["checkpoint"] = str(checkpoint)
        model_row["status"] = "ok"
        model_row["backend"] = accelerator
    except Exception as exc:  # noqa: BLE001 - a failed run is a recorded prune, not a crash
        model_row["checkpoint"] = ""
        model_row["status"] = f"convergence_fail: {type(exc).__name__}: {exc}"[:500]
        model_row["backend"] = accelerator
    return model_row


def _train_and_evaluate_one(
    entry: dict[str, Any],
    seed: int,
    dev_samples: list[s4.Sample],
    evaluation_sets: dict[str, list[s4.Sample]],
    *,
    output_root: Path,
    per_test_existing: dict,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """One (design, seed) unit of work: train, then evaluate on every distribution.

    Pure with respect to shared state -- everything it touches on disk
    (``run_dir``, each frozen test's ``run_tag`` directory) is keyed by
    ``design_id`` and ``seed``, so distinct units never write the same path.
    Returns rows to merge in and, on a training failure, a prune record --
    the caller owns all shared bookkeeping (the CSV, the in-memory dict), so
    this function can run from any thread without a lock.
    """
    model_row = train_one_design(entry, seed, dev_samples, output_root=output_root)
    if model_row["status"] != "ok":
        return [], _prune_record(entry, "convergence_fail", model_row["status"])

    rows: list[dict[str, Any]] = []
    for test_name, samples in evaluation_sets.items():
        eval_row = s5.evaluate_model_on_frozen_test(
            model_row, test_name, samples, output_root=output_root, existing_by_key=per_test_existing,
            accelerator=model_row["backend"],
        )
        rows.append({
            "design_id": entry["design_id"], "family": entry["family"], "dim": entry["dim"],
            "k": entry["k"], "N_train": entry["n_used"], "domain": entry["r_train_max_ang"],
            "density": entry["density_level"], "seed": seed,
            "test_amplitude": test_name,
            "ood_status": (
                "n/a" if test_name == DEVELOPMENT_SET_NAME
                else effective_ood_label(float(entry["r_train_max_ang"]), samples)
            ),
            "H_MAE": eval_row.get("H_MAE"), "H_RMSE": eval_row.get("H_RMSE"),
            "rel_Frob": eval_row.get("rel_Frob"), "spectral_err": eval_row.get("spectral_err"),
            "hermiticity": eval_row.get("hermiticity"), "n_evaluated": eval_row.get("n_evaluated"),
            "status": eval_row.get("status"), "split_id": eval_row.get("split_id"),
            "checkpoint": model_row["checkpoint"], "backend": model_row["backend"],
            "siesta_cost": entry["n_used"],
        })
    return rows, None


# --------------------------------------------------------------------------
# Pareto dominance (H-MAE on the development set vs unique_siesta_ids cost)
# --------------------------------------------------------------------------


def flag_pareto_dominated(design_scores: dict[str, tuple[float, int]]) -> set[str]:
    """design_id -> dominated iff another design has cost<= and H_MAE<=, one strict."""

    dominated: set[str] = set()
    for design_id, (mae, cost) in design_scores.items():
        for other_id, (other_mae, other_cost) in design_scores.items():
            if other_id == design_id:
                continue
            not_worse = other_mae <= mae and other_cost <= cost
            strictly_better = other_mae < mae or other_cost < cost
            if not_worse and strictly_better:
                dominated.add(design_id)
                break
    return dominated


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--siesta-command", default=pilot.DEFAULT_SIESTA_COMMAND)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--parallel-designs", type=int, default=7,
        help="Concurrent (design, seed) training jobs on the GPU (default: 7, matching this repo's "
             "validated max_parallel_graph2mat_training_jobs for models this size).",
    )
    parser.add_argument("--max-designs", type=int, default=None, help="Debug/testing cap on trainable designs.")
    parser.add_argument("--families", default=None, help="Comma-separated sampler families to train (default: all).")
    parser.add_argument("--dims", default=None, help="Comma-separated dimensionalities to train (default: all).")
    parser.add_argument("--k-values", default=None, help="Comma-separated k (active-atom count) values (default: all).")
    parser.add_argument("--r-train-max", default=None, help="Comma-separated R_train_max (Ang) domains (default: all).")
    parser.add_argument("--density-levels", default=None, help="Comma-separated density/resolution levels (default: all).")
    parser.add_argument("--seeds", default=None, help=f"Comma-separated training seeds (default: {TRAINING_SEEDS}).")
    args = parser.parse_args()

    siesta_command = args.siesta_command
    training_seeds: tuple[int, ...] = (
        tuple(int(v) for v in _parse_csv_list(args.seeds)) if args.seeds else TRAINING_SEEDS
    )

    backend_info = s4.torch_backend_preflight()
    print(json.dumps({"gpu_preflight": backend_info}, sort_keys=True))

    output_root: Path = args.output_root
    geometry = sampler.load_graphene_primitive()

    manifest = load_manifest(args.manifest_path)
    s3_pool = s4.load_s3_samples()
    hash_to_sample = build_s3_hash_index(s3_pool)

    trainable, budget_pruned = select_candidates(manifest, geometry, hash_to_sample)
    trainable = filter_designs(
        trainable,
        families=_parse_csv_list(args.families) or None,
        dims=_parse_csv_list(args.dims) or None,
        k_values=[int(v) for v in _parse_csv_list(args.k_values)] or None,
        r_train_max=[float(v) for v in _parse_csv_list(args.r_train_max)] or None,
        density_levels=[int(v) for v in _parse_csv_list(args.density_levels)] or None,
    )
    if args.max_designs is not None:
        trainable = trainable[: max(0, int(args.max_designs))]

    evaluation_sets = build_evaluation_sets(
        geometry, output_root=output_root, siesta_command=siesta_command, workers=args.workers,
    )
    dev_samples = evaluation_sets[DEVELOPMENT_SET_NAME]

    leakage_report = s5.check_no_leakage(s3_pool, evaluation_sets)
    write_json(output_root / "leakage_check_report.json", leakage_report)
    if not leakage_report["leakage_free"]:
        print(f"[S9-S4][ERROR] leakage detected: {leakage_report['overlaps']}", file=sys.stderr)
        return 2

    run_metrics_path = output_root / "run_metrics.csv"
    existing_rows = read_existing_run_metrics(run_metrics_path)
    # Keyed upsert (not a plain list) so re-running this resumable step never
    # duplicates a row for a (design, seed, test) combination already on disk --
    # every iteration below still runs (to reach un-cached combinations), but a
    # cache-hit re-processing of an already-recorded combination overwrites its
    # existing entry in place instead of appending a second copy.
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = dict(existing_rows)
    per_test_existing = _index_for_evaluate(list(rows_by_key.values()))

    convergence_failed: list[dict[str, Any]] = []
    dev_mae_by_design: dict[str, list[float]] = {}

    # Each (design, seed) unit trains and evaluates independently (see
    # _train_and_evaluate_one's docstring for why that's safe); only merging
    # results back into the shared CSV/dicts needs a lock. GPU headroom is
    # large for this model size (~23K params/graphene primitive cell), and
    # `args.parallel_designs` defaults to 7 -- the same concurrent-Graph2Mat-
    # job figure already validated elsewhere in this repo (g2m_deeph_runner's
    # max_parallel_graph2mat_training_jobs).
    units = [(entry, seed) for entry in trainable for seed in training_seeds]
    write_lock = threading.Lock()

    def _run_unit(entry: dict[str, Any], seed: int) -> tuple[dict[str, Any], int, list[dict[str, Any]], dict[str, Any] | None]:
        rows, prune_record = _train_and_evaluate_one(
            entry, seed, dev_samples, evaluation_sets, output_root=output_root, per_test_existing=per_test_existing,
        )
        return entry, seed, rows, prune_record

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.parallel_designs), thread_name_prefix="s9-s4-train",
    ) as executor:
        futures = [executor.submit(_run_unit, entry, seed) for entry, seed in units]
        for future in concurrent.futures.as_completed(futures):
            entry, seed, rows, prune_record = future.result()
            with write_lock:
                if prune_record is not None:
                    convergence_failed.append(prune_record)
                    continue
                for row in rows:
                    rows_by_key[(entry["design_id"], str(seed), row["test_amplitude"])] = row
                    if (
                        row["test_amplitude"] == DEVELOPMENT_SET_NAME
                        and row["status"] == "ok" and isinstance(row["H_MAE"], (int, float))
                    ):
                        dev_mae_by_design.setdefault(entry["design_id"], []).append(float(row["H_MAE"]))
                write_csv(run_metrics_path, list(rows_by_key.values()))

    design_scores = {
        design_id: (sum(values) / len(values), next(e["n_used"] for e in trainable if e["design_id"] == design_id))
        for design_id, values in dev_mae_by_design.items()
        if values
    }
    dominated_ids = flag_pareto_dominated(design_scores)
    dominated_pruned = [
        _prune_record(next(e for e in trainable if e["design_id"] == design_id), "dominated",
                       f"dev H_MAE={design_scores[design_id][0]:.6g} at cost={design_scores[design_id][1]} "
                       "is Pareto-dominated by another surviving design")
        for design_id in sorted(dominated_ids)
    ]

    all_pruned = budget_pruned + convergence_failed + dominated_pruned
    write_json(output_root / "pilot_pruned.json", {"task": "DATASET-DESIGN-W90-001-S9-S4", "pruned": all_pruned})

    summary = {
        "task": "DATASET-DESIGN-W90-001-S9-S4",
        "gpu_preflight": backend_info,
        "n_candidate_pilot_designs": sum(
            1 for e in manifest["designs"] if 1 <= int(e["n_used"]) <= PILOT_MAX_N
        ),
        "n_trainable": len(trainable),
        "n_budget_pruned": len(budget_pruned),
        "n_convergence_failed": len(convergence_failed),
        "n_dominated": len(dominated_pruned),
        "run_metrics_rows": len(rows_by_key),
        "leakage_free": leakage_report["leakage_free"],
    }
    write_json(output_root / "s9_s4_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def read_existing_run_metrics(path: Path) -> dict[tuple, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["design_id"], row["seed"], row["test_amplitude"]): row for row in csv.DictReader(handle)
        }


def _index_for_evaluate(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, str, str], dict[str, str]]:
    index: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            str(row["family"]), str(row["dim"]), str(row["k"]), str(row["N_train"]), str(row["seed"]),
            str(row["test_amplitude"]),
        )
        index[key] = {
            "training_family": str(row["family"]), "training_dim": str(row["dim"]), "training_k": str(row["k"]),
            "training_n_train": str(row["N_train"]), "training_seed": str(row["seed"]),
            "frozen_test": str(row["test_amplitude"]), "checkpoint": str(row.get("checkpoint", "")),
            "split_id": str(row.get("split_id", "")), "status": str(row.get("status", "")),
            "H_MAE": row.get("H_MAE", ""), "H_RMSE": row.get("H_RMSE", ""), "rel_Frob": row.get("rel_Frob", ""),
            "spectral_err": row.get("spectral_err", ""), "hermiticity": row.get("hermiticity", ""),
            "n_evaluated": row.get("n_evaluated", ""),
        }
    return index


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_METRICS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in RUN_METRICS_FIELDNAMES})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
