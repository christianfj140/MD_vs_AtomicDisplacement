#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9-S5 -- full Pareto-set training on pilot survivors.

S9-S4's pilot (N in {8,16,32,64}, 2 seeds) screened 468 candidate manifest
cells down to 44 real-SIESTA-backed ``trainable`` designs, then Pareto-pruned
those against each other on the development set's H-MAE vs cost -- 42 were
``dominated``, leaving exactly 2 survivors
(``Comparison/results/dataset_design_w90_001_s9_s4/pilot_pruned.json``):
``angular_shell__2D_in__k1__r0.080__res8__seeddeterministic`` and
``sobol_sparse__1D_in__k2__r0.080__res16__seed0``. This step scales only
those 2 designs to N in {128, 256}, 5 training seeds each, reusing S4's
trainer/predictor and S5's frozen-development-set/evaluation machinery
verbatim (S4 fairness requirement, same as S9-S4).

N=128/256 is above every level S9-S2 defined for either family
(``angular_shell`` tops out at 32; ``sobol_sparse`` at 256) -- S9-S2's level
*list* is frozen, but ``s3.generate_design`` itself takes an arbitrary
``resolution`` int, so the same generator is reused at a resolution beyond
the pre-registered list rather than inventing a new family/generator. This
is recorded explicitly (``density_level_source: "extrapolated_beyond_s2_levels"``)
rather than silently presented as an S2-registered level.

Nesting: whether the smaller N's geometries are an exact ORDERED hash-prefix
of the larger N's is checked empirically per design (``_is_nested_prefix``),
not assumed by family name. ``sobol_sparse``'s scrambled net satisfies this
(its own family doc: "any power-of-two prefix is itself a valid nested
net") -- verified by ``test_s9_s5_sobol_sparse_n128_is_a_hash_prefix_of_n256``
-- so this step submits ONE SIESTA batch at N=256 and reuses its first 128
(by sample-id order, zero-padded so lexical order == generation order) as
the N=128 pool. ``angular_shell`` nests only as a *set*, not as a prefix
(``np.linspace(0, 2pi, n, endpoint=False)``'s angle at index ``i`` for N
equals index ``2i`` for 2N -- an even/odd interleave, not a first-half
prefix), so the empirical order check correctly declines to reuse it and
submits two independent SIESTA batches instead. Guessing this by family name
would have either silently truncated angular_shell's N=128 pool to the wrong
points, or (in the other direction) needlessly re-submitted SIESTA jobs
sobol_sparse had already paid for.

Cost accounting follows S9-S2's ``COST_SCHEMA`` literally: every design/N
pool's geometry-hash set (``s5.geometry_hash`` on materialized positions, the
same identity used by S9-S4's own leakage/dedup checks) is compared against a
running campaign union seeded with the existing S3 pilot pool's hashes, so
``reproducible_cost``/``incremental_cost``/``campaign_union_cost`` are
recomputable from the hash sets alone (cost_per_siesta_point = 1 "unique
SIESTA calculation", matching S9-S4's own ``siesta_cost`` convention of
using point counts, not a currency).

Domain coverage: both survivors share R_train_max = 0.08 Ang (the only
domain that happened to survive pilot pruning) -- S9-S2 defines 3 domain
levels (0.03/0.05/0.08), so the "3 domains x 6 amplitudes" grid this task's
acceptance criteria describes cannot be filled from real trained models; this
step honestly reports a 1-domain x 6-amplitude grid per design (documented in
``s9_s5_summary.json["domain_coverage_note"]``) rather than fabricating rows
for domains with no surviving, trained model.

Minimum-N: per design, N_min is the smallest tested N (pooling S9-S4's low-N
dev H-MAE with this step's high-N dev H-MAE) within ``PLATEAU_REL_TOL`` of
the best H-MAE observed at any tested N for that design -- a plateau
criterion, not a physically-derived accuracy target (no such target was
specified for this task). Non-monotonic curves are handled by scanning N in
ascending order and taking the first one crossing the threshold, and a
"bracket" is always the (last-insufficient-N, N_min) pair actually observed
-- never a continuous-N interpolation, since only 6 discrete N were ever
trained.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
import dataset_design_w90_001_s9_s4_pilot_screening as s94  # noqa: E402
import run_w90_displacement_siesta_pilot as pilot  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_s5"
S94_ROOT = s94.DEFAULT_OUTPUT_ROOT

SCALE_N_LEVELS: tuple[int, ...] = (128, 256)
# Disjoint from S9-S4's TRAINING_SEEDS = (0, 1); paired across both N levels
# for the same design (same seed value trains both N=128 and N=256), so a
# scaling comparison isn't confounded by a different weight-init draw.
TRAINING_SEEDS: tuple[int, ...] = (2, 3, 4, 5, 6)
PLATEAU_REL_TOL = 0.05

PARETO_TABLE_FIELDNAMES = [
    "design_id", "family", "dim", "k", "domain_r_train_max_ang", "N_train", "seed",
    "split_id", "test_amplitude", "ood_status", "H_MAE", "H_RMSE", "status",
    "cost_siesta_unique", "reproducible_cost", "incremental_cost", "dominance",
    "checkpoint", "backend",
]
DOMAIN_GENERALIZATION_FIELDNAMES = [
    "design_id", "family", "N_train", "seed", "R_train_max_ang", "test_amplitude", "H_MAE", "ood_flag", "split_id",
]


# --------------------------------------------------------------------------
# Survivor selection: S9-S4's trainable set minus its own "dominated" prunes
# --------------------------------------------------------------------------


def load_survivors() -> tuple[list[dict[str, Any]], sampler.Geometry, dict[str, s4.Sample]]:
    manifest = s94.load_manifest()
    geometry = sampler.load_graphene_primitive()
    s3_pool = s4.load_s3_samples()
    hash_to_sample = s94.build_s3_hash_index(s3_pool)
    trainable, _pruned = s94.select_candidates(manifest, geometry, hash_to_sample)

    pruned_path = S94_ROOT / "pilot_pruned.json"
    pruned = json.loads(pruned_path.read_text(encoding="utf-8"))["pruned"]
    dominated_ids = {p["design_id"] for p in pruned if p["reason"] == "dominated"}
    survivors = [e for e in trainable if e["design_id"] not in dominated_ids]
    return survivors, geometry, hash_to_sample


# --------------------------------------------------------------------------
# Training-pool materialization at N in {128, 256} (new SIESTA where needed)
# --------------------------------------------------------------------------


def unique_configs_at_resolution(
    geometry: sampler.Geometry, entry: dict[str, Any], resolution: int, seed: int
) -> list[sampler.Configuration]:
    configs = s3.generate_design(
        geometry, entry["family"], entry["dim"], int(entry["k"]), float(entry["r_train_max_ang"]), resolution, seed,
    )
    seen: set[str] = set()
    unique: list[sampler.Configuration] = []
    for config in configs:
        digest = s3.point_geometry_hash(geometry, config)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(config)
    return unique


def _is_nested_prefix(geometry: sampler.Geometry, entry: dict[str, Any], seed: int, small_n: int, big_n_hashes: list[str]) -> bool:
    small_configs = unique_configs_at_resolution(geometry, entry, small_n, seed)
    small_hashes = [s3.point_geometry_hash(geometry, c) for c in small_configs]
    return small_hashes == big_n_hashes[:small_n]


def build_training_pools(
    entry: dict[str, Any], geometry: sampler.Geometry, *, output_root: Path, siesta_command: str, workers: int,
    n_levels: tuple[int, ...] = SCALE_N_LEVELS,
) -> tuple[dict[int, list[s4.Sample]], dict[str, float]]:
    """Returns ({N: samples}, {phase_timer_name: seconds}).

    Whether smaller N is an exact ORDERED hash-prefix of the largest N (so
    one SIESTA batch covers every level) is detected empirically rather than
    assumed per family: ``sobol_sparse``'s scrambled net satisfies this by
    construction. ``angular_shell``'s point *set* at N is a subset of its set
    at 2N (``np.linspace(0, 2pi, n, endpoint=False)``'s angle at index ``i``
    equals index ``2i`` at 2N), but that's an even/odd interleave, not a
    prefix -- ``samples[:n]`` would silently pick the wrong points, so the
    empirical check correctly falls through to independent submission for
    it.
    """

    seed_value = entry["seeds"]["sampler"]
    sampler_seed_int = seed_value if isinstance(seed_value, int) else 0
    timers: dict[str, float] = {}
    pools: dict[int, list[s4.Sample]] = {}

    sorted_levels = sorted(n_levels)
    max_n = sorted_levels[-1]
    max_configs = unique_configs_at_resolution(geometry, entry, max_n, sampler_seed_int)
    max_hashes = [s3.point_geometry_hash(geometry, c) for c in max_configs]
    nested = all(
        _is_nested_prefix(geometry, entry, sampler_seed_int, n, max_hashes) for n in sorted_levels[:-1]
    )

    if nested:
        tagged = [(f"{entry['design_id']}__N{max_n}__{i:04d}", c) for i, c in enumerate(max_configs)]
        pool_dir = output_root / "training_pools" / f"{entry['design_id']}__N{max_n}"
        start = time.perf_counter()
        run_result = s5.materialize_and_run(tagged, output_root=pool_dir, siesta_command=siesta_command, workers=workers)
        timers[f"siesta_wall_time_seconds__N{max_n}"] = time.perf_counter() - start
        samples = sorted(s5.samples_from_run(tagged, run_result), key=lambda s: s.sample_id)
        for n in sorted_levels:
            pools[n] = samples[:n]
    else:
        for n in sorted_levels:
            unique = max_configs if n == max_n else unique_configs_at_resolution(geometry, entry, n, sampler_seed_int)
            tagged = [(f"{entry['design_id']}__N{n}__{i:04d}", c) for i, c in enumerate(unique)]
            pool_dir = output_root / "training_pools" / f"{entry['design_id']}__N{n}"
            start = time.perf_counter()
            run_result = s5.materialize_and_run(tagged, output_root=pool_dir, siesta_command=siesta_command, workers=workers)
            timers[f"siesta_wall_time_seconds__N{n}"] = time.perf_counter() - start
            pools[n] = sorted(s5.samples_from_run(tagged, run_result), key=lambda s: s.sample_id)

    return pools, timers


# --------------------------------------------------------------------------
# Cost accounting (S9-S2 COST_SCHEMA, computed from geometry-hash sets)
# --------------------------------------------------------------------------


def pool_hashes(samples: list[s4.Sample]) -> set[str]:
    return {s5.geometry_hash(s5._positions_from_run_fdf(s.run_fdf)) for s in samples}


def cost_for_pool(samples: list[s4.Sample], campaign_union: set[str]) -> dict[str, int]:
    hashes = pool_hashes(samples)
    return {
        "cost_siesta_unique": len(hashes),
        "reproducible_cost": len(hashes),
        "incremental_cost": len(hashes - campaign_union),
    }


# --------------------------------------------------------------------------
# Training (reuses S4 verbatim, same pattern as S9-S4's train_one_design)
# --------------------------------------------------------------------------


def train_one(
    entry: dict[str, Any], n_train: int, seed: int, train_samples: list[s4.Sample], dev_samples: list[s4.Sample],
    *, output_root: Path, accelerator: str,
) -> tuple[dict[str, Any], float]:
    run_name = f"{entry['design_id']}__N{n_train}__seed{seed}"
    run_dir = output_root / "runs" / run_name
    model_row: dict[str, Any] = {
        "family": entry["family"], "dim": entry["dim"], "k": str(entry["k"]),
        "N_train": str(n_train), "seed": str(seed),
    }
    existing_checkpoints = sorted(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime) if run_dir.exists() else []
    start = time.perf_counter()
    try:
        if existing_checkpoints:
            checkpoint = existing_checkpoints[-1]
        else:
            config_path = s4.build_graph2mat_config(
                run_dir, train_samples, dev_samples, run_name=run_name, accelerator=accelerator, training_seed=seed,
            )
            checkpoint = s4.run_graph2mat_training(config_path, run_dir)
        model_row["checkpoint"] = str(checkpoint)
        model_row["status"] = "ok"
        model_row["backend"] = accelerator
    except Exception as exc:  # noqa: BLE001 - a failed run is a recorded row, not a crash
        model_row["checkpoint"] = ""
        model_row["status"] = f"convergence_fail: {type(exc).__name__}: {exc}"[:500]
        model_row["backend"] = accelerator
    return model_row, time.perf_counter() - start


# --------------------------------------------------------------------------
# Minimum-N determination (plateau criterion, pools S9-S4 low-N + this step's high-N)
# --------------------------------------------------------------------------


def determine_minimum_n(dev_mae_by_n: dict[int, float]) -> dict[str, Any]:
    """dev_mae_by_n: {N: mean H_MAE on development set}, any subset of tested N."""

    if not dev_mae_by_n:
        return {"status": "not_reached", "reason": "no dev-set H_MAE observations for this design"}

    ordered_n = sorted(dev_mae_by_n)
    best_mae = min(dev_mae_by_n.values())
    threshold = best_mae * (1.0 + PLATEAU_REL_TOL)

    n_min = None
    last_insufficient = None
    for n in ordered_n:
        if dev_mae_by_n[n] <= threshold:
            n_min = n
            break
        last_insufficient = n

    if n_min is None:
        return {
            "status": "not_reached",
            "threshold_H_MAE_eV": threshold,
            "threshold_definition": f"best observed H_MAE * (1 + {PLATEAU_REL_TOL})",
            "n_values_explored": ordered_n,
            "best_observed_H_MAE_eV": best_mae,
        }

    return {
        "status": "reached",
        "threshold_H_MAE_eV": threshold,
        "threshold_definition": f"best observed H_MAE * (1 + {PLATEAU_REL_TOL})",
        "N_min_observed": n_min,
        "N_min_bracket": [last_insufficient, n_min],
        "n_values_explored": ordered_n,
        "best_observed_H_MAE_eV": best_mae,
        "uncertainty": (
            "only the discrete N values actually trained were compared; the true minimum "
            "may lie anywhere inside N_min_bracket (or below the smallest N explored)"
        ),
    }


# --------------------------------------------------------------------------
# Minimum-N determination (absolute selectable threshold, items 16-17 of the
# second-audit closure) -- replaces the plateau-relative rule above as the
# PRIMARY summary statistic. ``determine_minimum_n`` (plateau) is kept as a
# secondary diagnostic since existing tests/consumers already read it.
# --------------------------------------------------------------------------

MIN_BUNDLES_CONFIRMATORY = 5
# One-sided 95% Student-t critical values by degrees of freedom (n_bundles-1),
# for n_bundles in [2, 10]; n_bundles > 10 falls back to the normal approximation.
_T95_TABLE: dict[int, float] = {
    2: 6.314, 3: 2.920, 4: 2.353, 5: 2.132, 6: 2.015, 7: 1.943, 8: 1.895, 9: 1.860, 10: 1.833,
}


def _t95(n_bundles: int) -> float:
    if n_bundles in _T95_TABLE:
        return _T95_TABLE[n_bundles]
    return 1.645 if n_bundles > 10 else _T95_TABLE[2]


def _bundle_u95(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"n_bundles": n, "mean_H_MAE_eV": mean, "sample_std_H_MAE_eV": float("nan"), "U95_H_MAE_eV": float("inf")}
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    u95 = mean + _t95(n) * std / math.sqrt(n)
    return {"n_bundles": n, "mean_H_MAE_eV": mean, "sample_std_H_MAE_eV": std, "U95_H_MAE_eV": u95}


def determine_minimum_n_for_target(
    dev_mae_by_n_bundles: dict[int, list[float]], target_h_mae_ev: float,
    *, min_bundles_confirmatory: int = MIN_BUNDLES_CONFIRMATORY,
) -> dict[str, Any]:
    """Selectable-threshold minimum N: smallest tested N with >=5 valid bundles

    and U95 = mean + t_(0.95,n-1)*s/sqrt(n) <= target_h_mae_ev on the common
    development set. Two-bundle pilot screening can never confirm (n_bundles
    < min_bundles_confirmatory always fails ``meets_rule``, however low its
    mean is) -- it can only produce ``not_reached``/``insufficient_evidence``,
    never ``reached``. Does not assume monotonicity: scans all tested N in
    ascending order and reports every crossing plus the bracket around the
    first one that meets the rule.
    """

    if not dev_mae_by_n_bundles:
        return {"status": "insufficient_evidence", "target_H_MAE_eV": target_h_mae_ev, "reason": "no dev-set H_MAE bundles for this design"}

    ordered_n = sorted(dev_mae_by_n_bundles)
    per_n: dict[int, dict[str, Any]] = {}
    for n in ordered_n:
        values = [float(v) for v in dev_mae_by_n_bundles[n] if isinstance(v, (int, float)) and math.isfinite(v)]
        if not values:
            continue
        stat = _bundle_u95(values)
        stat["raw_values"] = values
        stat["confirmatory_eligible"] = stat["n_bundles"] >= min_bundles_confirmatory
        stat["meets_rule"] = stat["confirmatory_eligible"] and stat["U95_H_MAE_eV"] <= target_h_mae_ev
        per_n[n] = stat

    if not per_n:
        return {"status": "insufficient_evidence", "target_H_MAE_eV": target_h_mae_ev, "reason": "no finite dev-set H_MAE observations"}

    crossings = [{"N": n, **{k: v for k, v in stat.items() if k != "raw_values"}} for n, stat in per_n.items()]
    n_min = next((n for n in ordered_n if n in per_n and per_n[n]["meets_rule"]), None)

    if n_min is None:
        any_confirmatory = any(stat["confirmatory_eligible"] for stat in per_n.values())
        status = "not_reached" if any_confirmatory else "insufficient_evidence"
        reason = (
            "no N reached U95<=target with >=5 valid bundles on the development set" if any_confirmatory else
            f"fewer than {min_bundles_confirmatory} bundles were trained at any N; "
            "pilot-only screening (2 bundles) can never confirm a minimum N"
        )
        return {
            "status": status, "target_H_MAE_eV": target_h_mae_ev, "n_values_explored": ordered_n,
            "crossings": crossings, "reason": reason,
        }

    last_insufficient = next((n for n in reversed(ordered_n) if n < n_min and n in per_n and not per_n[n]["meets_rule"]), None)
    values_at_min = per_n[n_min]["raw_values"]
    loo_meets: list[bool] = []
    for i in range(len(values_at_min)):
        held_out = values_at_min[:i] + values_at_min[i + 1:]
        if len(held_out) >= 2:
            loo_stat = _bundle_u95(held_out)
            loo_meets.append(len(held_out) >= min_bundles_confirmatory and loo_stat["U95_H_MAE_eV"] <= target_h_mae_ev)
    sensitivity_fragile = bool(loo_meets) and not all(loo_meets)

    return {
        "status": "reached",
        "target_H_MAE_eV": target_h_mae_ev,
        "N_min_observed": n_min,
        "N_min_bracket": [last_insufficient, n_min],
        "n_values_explored": ordered_n,
        "crossings": crossings,
        "sensitivity_fragile": sensitivity_fragile,
        "rule": (
            f"N meets iff >={min_bundles_confirmatory} valid bundles (one bundle = one training-seed "
            "instance evaluated on the common development set) and U95=mean+t(0.95,n-1)*s/sqrt(n) "
            "<= target_H_MAE_eV"
        ),
        "uncertainty": "only discrete N values actually trained were compared; true minimum may lie inside N_min_bracket",
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--siesta-command", default=pilot.DEFAULT_SIESTA_COMMAND)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--n-levels", type=int, nargs="+", default=None,
        help="Debug/smoke-test override for SCALE_N_LEVELS (e.g. --n-levels 4 8).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Debug/smoke-test override for TRAINING_SEEDS.",
    )
    parser.add_argument("--max-designs", type=int, default=None, help="Debug/testing cap on survivor designs.")
    parser.add_argument("--families", default=None, help="Comma-separated sampler families to train (default: all survivors).")
    parser.add_argument("--dims", default=None, help="Comma-separated dimensionalities to train (default: all survivors).")
    parser.add_argument("--k-values", default=None, help="Comma-separated k values to train (default: all survivors).")
    parser.add_argument("--r-train-max", default=None, help="Comma-separated R_train_max (Ang) domains (default: all survivors).")
    parser.add_argument(
        "--target-h-mae-ev", type=float, default=None,
        help="Selectable absolute H-MAE accuracy target (eV) for determine_minimum_n_for_target; "
        "omitted means no target selected (reported as insufficient_evidence, never fabricated).",
    )
    args = parser.parse_args()

    n_levels: tuple[int, ...] = tuple(args.n_levels) if args.n_levels else SCALE_N_LEVELS
    training_seeds: tuple[int, ...] = tuple(args.seeds) if args.seeds else TRAINING_SEEDS

    campaign_start = time.perf_counter()
    output_root: Path = args.output_root
    siesta_command = args.siesta_command

    backend_info = s4.torch_backend_preflight()
    print(json.dumps({"gpu_preflight": backend_info}, sort_keys=True))
    accelerator = backend_info["effective_backend"]

    survivors, geometry, hash_to_sample = load_survivors()
    survivors = s94.filter_designs(
        survivors,
        families=s94._parse_csv_list(args.families) or None,
        dims=s94._parse_csv_list(args.dims) or None,
        k_values=[int(v) for v in s94._parse_csv_list(args.k_values)] or None,
        r_train_max=[float(v) for v in s94._parse_csv_list(args.r_train_max)] or None,
    )
    if args.max_designs is not None:
        survivors = survivors[: max(0, int(args.max_designs))]
    campaign_union: set[str] = set(hash_to_sample.keys())

    evaluation_sets = s94.build_evaluation_sets(
        geometry, output_root=S94_ROOT, siesta_command=siesta_command, workers=args.workers,
    )
    dev_samples = evaluation_sets.pop(s94.DEVELOPMENT_SET_NAME)
    test_amplitude_sets = evaluation_sets

    pareto_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    cost_ledger: list[dict[str, Any]] = []
    timing_by_design: dict[str, Any] = {}
    dev_mae_by_design_n: dict[tuple[str, int], list[float]] = {}

    for entry in survivors:
        design_id = entry["design_id"]
        pools, siesta_timers = build_training_pools(
            entry, geometry, output_root=output_root, siesta_command=siesta_command, workers=args.workers,
            n_levels=n_levels,
        )
        design_timing: dict[str, Any] = {"siesta": siesta_timers, "training_seconds": {}, "inference_seconds": {}}

        for n_train, train_samples in pools.items():
            costs = cost_for_pool(train_samples, campaign_union)
            campaign_union |= pool_hashes(train_samples)
            cost_ledger.append({"design_id": design_id, "N_train": n_train, **costs})

            for seed in training_seeds:
                model_row, train_seconds = train_one(
                    entry, n_train, seed, train_samples, dev_samples, output_root=output_root, accelerator=accelerator,
                )
                design_timing["training_seconds"][f"N{n_train}_seed{seed}"] = train_seconds
                if model_row["status"] != "ok":
                    continue

                eval_targets = {s94.DEVELOPMENT_SET_NAME: dev_samples, **test_amplitude_sets}
                for test_name, samples in eval_targets.items():
                    infer_start = time.perf_counter()
                    eval_row = s5.evaluate_model_on_frozen_test(
                        model_row, test_name, samples, output_root=output_root, accelerator=model_row["backend"],
                    )
                    design_timing["inference_seconds"][f"N{n_train}_seed{seed}_{test_name}"] = (
                        time.perf_counter() - infer_start
                    )
                    h_mae = eval_row.get("H_MAE")
                    is_dev = test_name == s94.DEVELOPMENT_SET_NAME
                    ood_status = (
                        "n/a" if is_dev else s94.effective_ood_label(float(entry["r_train_max_ang"]), samples)
                    )
                    pareto_rows.append({
                        "design_id": design_id, "family": entry["family"], "dim": entry["dim"], "k": entry["k"],
                        "domain_r_train_max_ang": entry["r_train_max_ang"], "N_train": n_train, "seed": seed,
                        "split_id": eval_row.get("split_id"), "test_amplitude": test_name, "ood_status": ood_status,
                        "H_MAE": h_mae, "H_RMSE": eval_row.get("H_RMSE"), "status": eval_row.get("status"),
                        "cost_siesta_unique": costs["cost_siesta_unique"],
                        "reproducible_cost": costs["reproducible_cost"], "incremental_cost": costs["incremental_cost"],
                        "dominance": "",
                        "checkpoint": model_row["checkpoint"], "backend": model_row["backend"],
                    })
                    if not is_dev and isinstance(h_mae, (int, float)):
                        domain_rows.append({
                            "design_id": design_id, "family": entry["family"], "N_train": n_train, "seed": seed,
                            "R_train_max_ang": entry["r_train_max_ang"], "test_amplitude": test_name, "H_MAE": h_mae,
                            "ood_flag": ood_status in ("ood", "mixed"), "split_id": eval_row.get("split_id"),
                        })
                    if is_dev and isinstance(h_mae, (int, float)) and eval_row.get("status") == "ok":
                        dev_mae_by_design_n.setdefault((design_id, n_train), []).append(float(h_mae))

        timing_by_design[design_id] = design_timing

    # Pareto dominance over (design, N) points, scored by mean dev-set H_MAE at that cost.
    point_scores: dict[str, tuple[float, int]] = {}
    for (design_id, n_train), values in dev_mae_by_design_n.items():
        cost = next(c["cost_siesta_unique"] for c in cost_ledger if c["design_id"] == design_id and c["N_train"] == n_train)
        point_scores[f"{design_id}__N{n_train}"] = (sum(values) / len(values), cost)
    dominated_points = s94.flag_pareto_dominated(point_scores)
    for row in pareto_rows:
        key = f"{row['design_id']}__N{row['N_train']}"
        if key in point_scores:
            row["dominance"] = "dominated" if key in dominated_points else "pareto_optimal"

    # Minimum-N: pool S9-S4's own low-N dev H_MAE with this step's high-N dev H_MAE.
    s94_rows = s94.read_existing_run_metrics(S94_ROOT / "run_metrics.csv")
    min_n_report: dict[str, Any] = {}
    min_n_target_report: dict[str, Any] = {}
    for entry in survivors:
        design_id = entry["design_id"]
        pilot_dev_mae_by_n: dict[int, list[float]] = {}
        for (row_design_id, _seed_str, test_name), row in s94_rows.items():
            if row_design_id != design_id or test_name != s94.DEVELOPMENT_SET_NAME or row.get("status") != "ok":
                continue
            try:
                n = int(row["N_train"])
                mae = float(row["H_MAE"])
            except (KeyError, ValueError, TypeError):
                continue
            pilot_dev_mae_by_n.setdefault(n, []).append(mae)
        pilot_means = {n: sum(v) / len(v) for n, v in pilot_dev_mae_by_n.items()}
        bundles_by_n: dict[int, list[float]] = dict(pilot_dev_mae_by_n)
        for n_train in n_levels:
            values = dev_mae_by_design_n.get((design_id, n_train))
            if values:
                pilot_means[n_train] = sum(values) / len(values)
                bundles_by_n[n_train] = values
        min_n_report[design_id] = determine_minimum_n(pilot_means)
        if args.target_h_mae_ev is None:
            min_n_target_report[design_id] = {
                "status": "insufficient_evidence",
                "target_H_MAE_eV": None,
                "reason": "no absolute H-MAE target selected for this run (--target-h-mae-ev not passed)",
            }
        else:
            min_n_target_report[design_id] = determine_minimum_n_for_target(bundles_by_n, args.target_h_mae_ev)

    write_csv(output_root / "pareto_table.csv", pareto_rows, PARETO_TABLE_FIELDNAMES)
    write_csv(output_root / "domain_generalization.csv", domain_rows, DOMAIN_GENERALIZATION_FIELDNAMES)
    write_json(output_root / "cost_ledger.json", cost_ledger)
    write_json(output_root / "timing.json", {
        "total_wall_time_seconds": time.perf_counter() - campaign_start,
        "gpu_preflight": backend_info,
        "workers": args.workers,
        "by_design": timing_by_design,
        "note": (
            "siesta_core_time_seconds is not separately measured by the SIESTA runner; approximate "
            "core-seconds as siesta_wall_time_seconds * workers (one subprocess per worker)"
        ),
    })

    summary = {
        "task": "DATASET-DESIGN-W90-001-S9-S5",
        "gpu_preflight": backend_info,
        "n_survivor_designs": len(survivors),
        "survivor_design_ids": [e["design_id"] for e in survivors],
        "n_levels": list(n_levels),
        "training_seeds": list(training_seeds),
        "density_level_source": "extrapolated_beyond_s2_levels",
        "domain_coverage_note": (
            "both pilot survivors share R_train_max=0.08 Ang; S9-S2 defines 3 domain levels "
            "(0.03/0.05/0.08) but only 1 is represented among real Pareto-surviving, trained designs, "
            "so domain_generalization.csv is a 1-domain x 6-amplitude grid per design, not 3x6"
        ),
        "pareto_table_rows": len(pareto_rows),
        "domain_generalization_rows": len(domain_rows),
        "minimum_n_by_design": min_n_report,
        "minimum_n_target_by_design": min_n_target_report,
        "target_h_mae_ev": args.target_h_mae_ev,
        "independent_confirmation": {
            "status": "provisional",
            "reason": (
                "frozen final tests (common_displacement/ood_displacement/md_frozen) were not opened in this "
                "step; only the development set and the 6 pre-reserved test amplitudes were used, per this "
                "task's instruction not to open frozen final test metrics until finalists are fixed"
            ),
        },
    }
    write_json(output_root / "s9_s5_summary.json", summary)
    print(json.dumps(summary, sort_keys=True, default=str))
    return 0


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
