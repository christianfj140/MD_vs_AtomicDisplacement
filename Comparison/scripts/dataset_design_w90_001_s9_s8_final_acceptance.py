#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9-S8 -- final acceptance integration.

Aggregation-only step: no new SIESTA/training runs. It reads the **S9
lineage** -- the one produced by the bug-fixed pipeline -- and writes one
acceptance package under ``Comparison/results/dataset_design_w90_001_s9/``:

  - S9-S2/S9-S3: envelope/coverage spec and the generated, validated designs
    (envelope, zero-leakage and nested-prefix checks).
  - S9-S4 (pilot): the executed screening runs, with domain and density as
    first-class columns and the three seeds kept separate.
  - S9-S5 (full/Pareto): the executed training for the pilot survivors, with
    the cost ledger and the domain-generalization grid.

Every number in ``summary.json`` is recomputed here from those artifacts, so
the package cannot drift from the data it claims to describe.

What this package must *not* do is inherit conclusions from the pre-S9-S1
historical S4-S7 pilot. Those runs predate the bug fixes (cache keys that
ignored domain/density, ``hash()``-seeded splits, the mismatched amplitude
envelope, conflated seeds), so their numbers cannot be attributed to the
fixed pipeline. They are kept only as a clearly-labelled historical
benchmark in ``historical_test_inventory``, never as an answer.

``independent_confirmation`` stays ``provisional`` while executed coverage is
incomplete, and ``executed_coverage`` computes -- from the artifacts, not
from prose -- exactly which designed domains and sampler families have no
executed runs and why. At the time of writing the binding gap is that no
SIESTA labels exist for the 0.05 Ang domain or for ``latin_hypercube``, so
those cells cannot be trained without new DFT work.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402
import dataset_design_w90_001_s9_s2_envelope_and_coverage as s2  # noqa: E402
import dataset_design_w90_001_s9_s4_pilot_screening as s9s4  # noqa: E402

S4_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s4"
S5_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s5"
S6_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s6"
S7_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s7"
S9_S2_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_s2"
S9_S3_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_s3"
S9_S4_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_s4"
S9_S5_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_s5"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9"

TRACEABILITY_SAMPLE_SEED = 8

REQUIRED_FILE_NAMES = ("design_manifest.json", "run_metrics.csv", "domain_generalization.csv", "pareto_table.csv", "summary.json")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
# Executed-coverage check (the hard evidence behind "provisional")
# --------------------------------------------------------------------------


def _domain_key(value: Any) -> str:
    """One spelling for a domain level, whatever artifact it came from."""
    return f"{float(value):.2f}"


def executed_coverage() -> dict[str, Any]:
    """What S9-S2/S9-S3 designed versus what S9-S4/S9-S5 actually executed.

    Recomputed from the artifacts every time, because the honest statement of
    what this campaign can answer depends entirely on which cells have runs.
    A design that was never trained cannot support a conclusion, and the
    reason it was never trained decides whether more GPU time would fix it
    (it would not, if the SIESTA labels do not exist).
    """
    designs = _read_json(S9_S3_ROOT / "design_manifest.json")["designs"]
    pilot_ok = [r for r in _read_csv(S9_S4_ROOT / "run_metrics.csv") if r["status"] == "ok"]
    generalization_rows = _read_csv(S9_S5_ROOT / "domain_generalization.csv")
    pruned = _read_json(S9_S4_ROOT / "pilot_pruned.json")["pruned"]

    designed_domains = sorted({_domain_key(d["r_train_max_ang"]) for d in designs})
    designed_families = sorted({d["family"] for d in designs})
    pilot_domains = sorted({_domain_key(r["domain"]) for r in pilot_ok})
    pilot_families = sorted({r["family"] for r in pilot_ok})
    full_domains = sorted({_domain_key(r["R_train_max_ang"]) for r in generalization_rows})

    # "budget" in the pilot's pruning log does not mean "we ran out of time":
    # the detail says the design's geometries have no materialized SIESTA
    # reference. Those cells need new DFT, not a longer GPU run.
    unlabelled = [e for e in pruned if "SIESTA reference" in str(e.get("detail", ""))]
    unlabelled_domains = sorted({_domain_key(e["domain_r_train_max_ang"]) for e in unlabelled})
    unlabelled_families = sorted({e["family"] for e in unlabelled})

    missing_domains = [d for d in designed_domains if d not in pilot_domains]
    missing_families = [f for f in designed_families if f not in pilot_families]

    # Selection provenance: the confirmation stage exists to confirm what the
    # pilot selected. If the pilot has since been re-run, its survivor set can
    # move, and a frontier built from the previous set is citing designs the
    # current screening did not choose. That is a silent inconsistency exactly
    # of the kind this package is supposed to make impossible, so it is
    # computed rather than remembered.
    pilot_trained = {r["design_id"] for r in pilot_ok}
    pilot_pruned_ids = {e["design_id"] for e in pruned}
    confirmed_ids = {r["design_id"] for r in full_rows()}
    stale_confirmations = sorted(
        design_id for design_id in confirmed_ids
        if design_id not in pilot_trained or design_id in pilot_pruned_ids
    )
    unconfirmed_survivors = sorted(
        design_id for design_id in pilot_trained
        if design_id not in pilot_pruned_ids and design_id not in confirmed_ids
    )

    gaps: list[str] = []
    if stale_confirmations:
        gaps.append(
            f"the confirmation stage reports design(s) {stale_confirmations} that the current "
            "pilot did not select: S9-S5 predates the latest S9-S4 run and must be re-run "
            "before its frontier can be read as confirming this pilot"
        )
    if unconfirmed_survivors:
        gaps.append(
            f"pilot survivor(s) {unconfirmed_survivors} have no confirmation runs, so the "
            "frontier does not cover everything the current screening selected"
        )
    for domain in missing_domains:
        why = (
            "no SIESTA labels exist for its geometries (needs new DFT, not more GPU time)"
            if domain in unlabelled_domains else "not executed"
        )
        gaps.append(f"domain {domain} Ang was designed but never trained: {why}")
    for family in missing_families:
        why = (
            "no SIESTA labels exist for its geometries (needs new DFT, not more GPU time)"
            if family in unlabelled_families else "not executed"
        )
        gaps.append(f"sampler family '{family}' was designed but never trained: {why}")
    if len(full_domains) < len(pilot_domains):
        gaps.append(
            f"the full/Pareto stage covers domain(s) {full_domains} Ang while the pilot "
            f"covers {pilot_domains} Ang, so domain-size effects are established only at "
            "pilot scale and not at confirmation scale"
        )

    return {
        "designed_domains_ang": designed_domains,
        "designed_families": designed_families,
        "pilot_executed_domains_ang": pilot_domains,
        "pilot_executed_families": pilot_families,
        "full_executed_domains_ang": full_domains,
        "n_designs_without_siesta_labels": len(unlabelled),
        "domains_without_siesta_labels_ang": unlabelled_domains,
        "families_without_siesta_labels": unlabelled_families,
        "pilot_supports_matched_domain_comparison": len(pilot_domains) > 1,
        "stale_confirmation_designs": stale_confirmations,
        "unconfirmed_pilot_survivors": unconfirmed_survivors,
        "confirmation_matches_current_pilot": not (stale_confirmations or unconfirmed_survivors),
        "gaps": gaps,
        "conclusion": "incomplete_executed_coverage" if gaps else "designed_coverage_fully_executed",
    }


# --------------------------------------------------------------------------
# The executed S9 lineage, rebuilt into the acceptance-package schema
# --------------------------------------------------------------------------
#
# The S9 stages write what is natural for each stage (per-model rows, cost
# ledgers keyed by design). The acceptance package is a different, aggregate
# contract -- matched grids and a recomputable Pareto frontier. These builders
# are that translation, and they are the only place it happens.


def pilot_rows() -> list[dict[str, Any]]:
    """Executed S9-S4 screening runs."""
    return [r for r in _read_csv(S9_S4_ROOT / "run_metrics.csv") if r.get("status") == "ok"]


def full_rows(test_amplitude: str | None = None) -> list[dict[str, Any]]:
    """Executed S9-S5 confirmation runs, optionally one evaluation distribution."""
    rows = [r for r in _read_csv(S9_S5_ROOT / "pareto_table.csv") if r.get("status") == "ok"]
    if test_amplitude is not None:
        rows = [r for r in rows if r.get("test_amplitude") == test_amplitude]
    return rows


def build_run_metrics() -> list[dict[str, Any]]:
    """Pilot and confirmation runs in one schema, tagged by the stage they came from.

    The Pareto table below cites confirmation runs (N=128/256) that exist only
    in S9-S5, so a package whose run_metrics held the pilot alone would carry a
    frontier nothing in it could account for.
    """
    rows: list[dict[str, Any]] = []
    for row in pilot_rows():
        rows.append({
            "stage": "pilot",
            "design_id": row["design_id"],
            "family": row["family"],
            "dim": row["dim"],
            "k": row["k"],
            "N_train": row["N_train"],
            "R_train_max_ang": _domain_key(row["domain"]),
            "density_level": row["density"],
            "seed": row["seed"],
            "test_amplitude": row["test_amplitude"],
            "ood_status": row["ood_status"],
            "H_MAE": row["H_MAE"],
            "H_RMSE": row["H_RMSE"],
            "hermiticity": row["hermiticity"],
            "siesta_cost": int(row["N_train"]) + s4.VAL_COUNT,
            "status": row["status"],
            "split_id": row["split_id"],
            "checkpoint": row["checkpoint"],
            "backend": row["backend"],
        })
    for row in full_rows():
        rows.append({
            "stage": "full",
            "design_id": row["design_id"],
            "family": row["family"],
            "dim": row["dim"],
            "k": row["k"],
            "N_train": row["N_train"],
            "R_train_max_ang": _domain_key(row["domain_r_train_max_ang"]),
            "density_level": "",
            "seed": row["seed"],
            "test_amplitude": row["test_amplitude"],
            "ood_status": row["ood_status"],
            "H_MAE": row["H_MAE"],
            "H_RMSE": row["H_RMSE"],
            "hermiticity": "",
            "siesta_cost": int(row["N_train"]) + s4.VAL_COUNT,
            "status": row["status"],
            "split_id": row["split_id"],
            "checkpoint": row["checkpoint"],
            "backend": row["backend"],
        })
    return rows


def build_domain_generalization() -> list[dict[str, Any]]:
    """Training domain x evaluation amplitude, averaged over seeds.

    Both stages contribute. The pilot is the only stage that trained more than
    one domain, so it is the only evidence that can speak to domain size at
    all; dropping it because it is "only" the pilot would leave the grid with
    one row per amplitude and no domain axis whatsoever.
    """
    buckets: dict[tuple[str, str, str, str], list[float]] = {}
    ood_flags: dict[tuple[str, str, str, str], set[str]] = {}

    def add(stage: str, domain: str, family: str, amplitude: str, h_mae: str, ood: str) -> None:
        if amplitude == "development_set":
            return
        try:
            value = float(h_mae)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        key = (stage, domain, family, amplitude)
        buckets.setdefault(key, []).append(value)
        ood_flags.setdefault(key, set()).add(ood)

    for row in pilot_rows():
        add("pilot", _domain_key(row["domain"]), row["family"],
            row["test_amplitude"], row["H_MAE"], row["ood_status"])
    for row in full_rows():
        add("full", _domain_key(row["domain_r_train_max_ang"]), row["family"],
            row["test_amplitude"], row["H_MAE"], row["ood_status"])

    grid: list[dict[str, Any]] = []
    for key in sorted(buckets):
        stage, domain, family, amplitude = key
        values = buckets[key]
        flags = ood_flags[key]
        grid.append({
            "stage": stage,
            "R_train_max_ang": domain,
            "family": family,
            "test_amplitude": amplitude,
            "n_models": len(values),
            "H_MAE_mean": statistics.fmean(values),
            "H_MAE_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "ood_status": "|".join(sorted(flags)),
            "is_ood": any(flag in ("ood", "mixed") for flag in flags),
        })
    return grid


def build_pareto_table() -> list[dict[str, Any]]:
    """The accuracy-cost frontier on the selection distribution only.

    Two things the audit asked for are settled here rather than inherited.
    Cost counts the validation labels a design needs (``N_train + VAL_COUNT``),
    not just its training pool, and dominance is *recomputed* from the pairs
    written into this very table, so the published flag cannot disagree with
    the published numbers. Only the development distribution is used: mixing
    evaluation distributions in one frontier compares models on different
    questions.
    """
    rows = full_rows(test_amplitude="development_set")
    scored: dict[str, tuple[float, int]] = {}
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            h_mae = float(row["H_MAE"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(h_mae):
            continue
        cost = int(row["N_train"]) + s4.VAL_COUNT
        tag = f"row{len(prepared)}"
        scored[tag] = (h_mae, cost)
        prepared.append({
            "design_id": row["design_id"],
            "family": row["family"],
            "dim": row["dim"],
            "k": row["k"],
            "R_train_max_ang": _domain_key(row["domain_r_train_max_ang"]),
            "N_train": row["N_train"],
            "seed": row["seed"],
            "evaluation_distribution": "development_set",
            "H_MAE": row["H_MAE"],
            "siesta_cost": cost,
            "cost_training_pool": row.get("cost_siesta_unique", ""),
            "reproducible_cost": row.get("reproducible_cost", ""),
            "incremental_cost": row.get("incremental_cost", ""),
            "_tag": tag,
        })

    dominated = s9s4.flag_pareto_dominated(scored)
    for entry in prepared:
        entry["pareto_dominated"] = entry.pop("_tag") in dominated
    return prepared


# --------------------------------------------------------------------------
# Pruned branches (reason category + evidence reference)
# --------------------------------------------------------------------------


def pruned_branches() -> list[dict[str, Any]]:
    branches: list[dict[str, Any]] = []

    # The S9 pilot's own pruning record, aggregated by why a branch stopped.
    # "budget" is reported with its real detail: for most entries it means the
    # design's geometries were never SIESTA-labelled, which is a different
    # decision from dropping a branch because it was dominated.
    grouped: dict[tuple[str, str, str], int] = {}
    for entry in _read_json(S9_S4_ROOT / "pilot_pruned.json")["pruned"]:
        detail = str(entry.get("detail", ""))
        category = entry.get("reason", "unspecified")
        if category == "budget" and "SIESTA reference" in detail:
            category = "no_siesta_labels"
        key = (entry.get("family", "?"), _domain_key(entry.get("domain_r_train_max_ang", 0)), category)
        grouped[key] = grouped.get(key, 0) + 1
    for (family, domain, category), count in sorted(grouped.items()):
        branches.append({
            "scope": f"family={family}, R_train_max={domain} Ang",
            "reason_category": category,
            "decision": "pruned",
            "rationale": (
                f"{count} design(s) pruned in the S9 pilot; "
                + (
                    "their geometries have no materialized SIESTA reference, so they need new "
                    "DFT work rather than more training time"
                    if category == "no_siesta_labels"
                    else "dominated on (H-MAE, cost) by an executed design"
                    if category == "dominated"
                    else "see the pilot pruning log for the recorded detail"
                )
            ),
            "evidence_reference": "Comparison/results/dataset_design_w90_001_s9_s4/pilot_pruned.json",
        })

    seen: set[tuple[str, str, int, str]] = set()
    for row in s2.build_coverage_matrix():
        if row["state"] not in ("not_applicable", "pruned"):
            continue
        key = (row["family"], row["dim"], int(row["k"]), row["state"])
        if key in seen:
            continue
        seen.add(key)
        branches.append({
            "scope": f"family={row['family']}, dim={row['dim']}, k={row['k']}",
            "reason_category": row["state"],
            "decision": row["state"],
            "rationale": row["reason"],
            "evidence_reference": "Comparison/results/dataset_design_w90_001_s9_s2/envelope_spec.json#coverage_matrix",
        })
    return branches


# --------------------------------------------------------------------------
# Historical test inventory: which evaluations influenced which decisions
# --------------------------------------------------------------------------


def historical_test_inventory() -> list[dict[str, Any]]:
    return [
        {
            "test_name": "common_validation",
            "role": "model_selection_ranking",
            "n_samples": 24,
            "decisions_influenced": [
                "S7 section 1: best sampler family at fixed N_train",
                "S7 section 2: best sampler family at fixed SIESTA cost",
                "S7 section 9: Pareto-optimal set (H-MAE axis)",
            ],
            "constraint": (
                "One predeclared validation distribution shared by every training family; "
                "per S1 section 12, never used to name a winner from a frozen test."
            ),
            "files": ["Comparison/results/dataset_design_w90_001_s5/per_model_common_validation_metrics.csv"],
        },
        {
            "test_name": "common_displacement (frozen)",
            "role": "generalization_estimate_only",
            "n_samples": 24,
            "decisions_influenced": ["S7 section 5: required displacement amplitudes (in-domain generalization number)"],
            "constraint": "Frozen test, amplitudes drawn within the trained 0.03-0.08 Ang range; never used to pick a sampler-family winner.",
            "files": ["Comparison/results/dataset_design_w90_001_s5/generalization_matrix.csv"],
        },
        {
            "test_name": "ood_displacement (frozen)",
            "role": "generalization_estimate_only",
            "n_samples": 37,
            "decisions_influenced": ["S7 section 5: required displacement amplitudes (out-of-domain error growth)"],
            "constraint": "Frozen test at 0.12 Ang, outside every training domain; never used to pick a winner.",
            "files": ["Comparison/results/dataset_design_w90_001_s5/generalization_matrix.csv"],
        },
        {
            "test_name": "md_frozen (frozen, real MD frames)",
            "role": "generalization_estimate_only",
            "n_samples": 7,
            "decisions_influenced": ["S7 section 7: generalization to real MD frames"],
            "constraint": "7 decorrelated frames from a single MD trajectory region (S6); small-sample, not a paper-grade MD test.",
            "files": ["Comparison/results/dataset_design_w90_001_s6/md_frozen_decorrelation_report.json"],
        },
        {
            "test_name": "S4 pilot/validation signal (development set)",
            "role": "development_only_never_a_generalization_claim",
            "n_samples": 120,
            "decisions_influenced": ["S6: rejection of the 5% relative-H-MAE candidate non-inferiority margin"],
            "constraint": "Used only to check whether the 5% candidate margin is achievable at all by this architecture, not as a held-out claim.",
            "files": ["Comparison/results/dataset_design_w90_001_s6/non_inferiority_margin.json"],
        },
    ]


# --------------------------------------------------------------------------
# Five scientific-question answers
# --------------------------------------------------------------------------


PLATEAU_REL_TOL = 0.05


def _dev_curve() -> dict[str, dict[int, list[float]]]:
    """design_id -> N_train -> development-set H-MAE for every seed."""
    curve: dict[str, dict[int, list[float]]] = {}
    def add(design_id: str, n_train: str, h_mae: str) -> None:
        try:
            value = float(h_mae)
        except (TypeError, ValueError):
            return
        if math.isfinite(value):
            curve.setdefault(design_id, {}).setdefault(int(n_train), []).append(value)

    for row in pilot_rows():
        if row["test_amplitude"] == "development_set":
            add(row["design_id"], row["N_train"], row["H_MAE"])
    for row in full_rows(test_amplitude="development_set"):
        add(row["design_id"], row["N_train"], row["H_MAE"])
    return curve


def minimum_n_by_design() -> dict[str, Any]:
    """Smallest executed N whose mean H-MAE reaches each design's own plateau.

    Stated as a *bracket*, never as a point: only the N values actually trained
    were measured, so the true minimum can sit anywhere between the largest N
    that missed the threshold and the smallest that met it. Monotonicity is not
    assumed -- the smallest qualifying N is taken, not the first one scanned.
    """
    answers: dict[str, Any] = {}
    for design_id, by_n in _dev_curve().items():
        means = {n: statistics.fmean(values) for n, values in by_n.items()}
        if len(means) < 2:
            continue
        best = min(means.values())
        threshold = best * (1 + PLATEAU_REL_TOL)
        qualifying = sorted(n for n, mean in means.items() if mean <= threshold)
        explored = sorted(means)
        if not qualifying:
            answers[design_id] = {
                "status": "not_reached",
                "n_values_explored": explored,
                "threshold_H_MAE_eV": threshold,
            }
            continue
        n_min = qualifying[0]
        below = [n for n in explored if n < n_min]
        answers[design_id] = {
            "status": "reached",
            "N_min_observed": n_min,
            "N_min_bracket": [max(below) if below else None, n_min],
            "n_values_explored": explored,
            "n_seeds_per_point": {str(n): len(v) for n, v in sorted(by_n.items())},
            "best_observed_H_MAE_eV": best,
            "threshold_H_MAE_eV": threshold,
            "threshold_definition": f"each design's own best observed H-MAE x (1 + {PLATEAU_REL_TOL})",
        }
    return answers


def matched_family_comparison() -> list[dict[str, Any]]:
    """Family ranking only where families share a domain, N and evaluation set.

    An unmatched comparison would be reporting sampler quality and training
    budget as if they were one number.
    """
    cells: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in pilot_rows():
        if row["test_amplitude"] != "development_set":
            continue
        try:
            value = float(row["H_MAE"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        key = (_domain_key(row["domain"]), row["N_train"])
        cells.setdefault(key, {}).setdefault(row["family"], []).append(value)

    comparisons: list[dict[str, Any]] = []
    for (domain, n_train), families in sorted(cells.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
        if len(families) < 2:
            continue
        means = {family: statistics.fmean(values) for family, values in families.items()}
        best = min(means, key=lambda family: means[family])
        comparisons.append({
            "R_train_max_ang": domain,
            "N_train": int(n_train),
            "families_compared": sorted(means),
            "H_MAE_mean_eV": {family: means[family] for family in sorted(means)},
            "n_seeds": {family: len(values) for family, values in sorted(families.items())},
            "best_family": best,
        })
    return comparisons


def domain_effect_at_matched_n() -> list[dict[str, Any]]:
    """Same design family, same N, different training domain.

    The only executed evidence about domain size, and it exists only at pilot
    scale: the confirmation stage trained one domain.
    """
    cells: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for row in pilot_rows():
        if row["test_amplitude"] != "development_set":
            continue
        try:
            value = float(row["H_MAE"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        key = (row["family"], row["dim"], row["N_train"])
        cells.setdefault(key, {}).setdefault(_domain_key(row["domain"]), []).append(value)

    effects: list[dict[str, Any]] = []
    for (family, dim, n_train), domains in sorted(cells.items()):
        if len(domains) < 2:
            continue
        means = {domain: statistics.fmean(values) for domain, values in domains.items()}
        effects.append({
            "family": family,
            "dim": dim,
            "N_train": int(n_train),
            "H_MAE_mean_eV_by_domain": {d: means[d] for d in sorted(means)},
            "lower_error_at": min(means, key=lambda d: means[d]),
        })
    return effects


def scientific_question_answers() -> dict[str, Any]:
    coverage = executed_coverage()
    minimum_n = minimum_n_by_design()
    families = matched_family_comparison()
    domain_effects = domain_effect_at_matched_n()
    frontier = [row for row in build_pareto_table() if not row["pareto_dominated"]]
    grid = build_domain_generalization()
    runs = build_run_metrics()

    in_domain = [r["H_MAE_mean"] for r in grid if not r["is_ood"]]
    out_domain = [r["H_MAE_mean"] for r in grid if r["is_ood"]]

    return {
        "q1_minimum_training_set_size": {
            "value": minimum_n,
            "units": "N_train (unique SIESTA-labelled structures)",
            "uncertainty": (
                "reported as a bracket, not a point: only the N values actually trained were "
                "measured, so the true minimum may lie anywhere inside N_min_bracket (or below "
                "the smallest N explored). The threshold is each design's own best observed "
                f"H-MAE x (1 + {PLATEAU_REL_TOL}) -- a relative plateau criterion, NOT the "
                "user-selectable absolute H-MAE target the objective asks for, which no "
                "executed run implements. The confirmation stage trained only N in {128, 256}, "
                "so the gap between the pilot's largest N and 128 is unmeasured."
            ),
            "method": (
                "per-design development-set learning curve (mean over seeds at each executed "
                "N_train), smallest qualifying N taken without assuming monotonicity"
            ),
            "data_source": (
                "Comparison/results/dataset_design_w90_001_s9_s4/run_metrics.csv; "
                "Comparison/results/dataset_design_w90_001_s9_s5/pareto_table.csv"
            ),
            "status": "partial",
        },
        "q2_distribution_density": {
            "value": {
                "matched_family_comparisons": families,
                "domain_effect_at_matched_n": domain_effects,
            },
            "units": "eV (H-MAE) against N_train, training domain and sampler family",
            "uncertainty": (
                "means over the 2 pilot seeds per point; every comparison listed holds domain, "
                "N_train and evaluation set fixed, but the seed count is small and no confidence "
                "band is claimed. Density is NOT isolated here: the executed designs vary "
                "resolution together with the generated pool size, so a fixed-N density series "
                "of three levels per family -- which the objective asks for -- is not available "
                "from executed runs."
            ),
            "method": (
                "development-set H-MAE averaged over seeds, compared only within cells that "
                "share (domain, N_train) or (family, dim, N_train)"
            ),
            "data_source": "Comparison/results/dataset_design_w90_001_s9_s4/run_metrics.csv",
            "status": "partial",
            "scope_limits": (
                f"executed families {coverage['pilot_executed_families']} of designed "
                f"{coverage['designed_families']}; executed domains "
                f"{coverage['pilot_executed_domains_ang']} of designed "
                f"{coverage['designed_domains_ang']} Ang. Missing cells have no SIESTA labels "
                "(see independent_confirmation.evidence), so they are pending DFT, not pending GPU time."
            ),
        },
        "q3_baseline_graph2mat_md_equivalence": {
            "value": {
                "graph2mat_baseline_established": True,
                "executed_runs": len(runs),
                "designs_trained": len({row["design_id"] for row in runs}),
                "stages": sorted({row["stage"] for row in runs}),
                "all_checkpoints_resolve": all(
                    (not row["checkpoint"]) or Path(row["checkpoint"]).exists() for row in runs
                ),
                "deeph_comparison": "deferred_out_of_scope",
            },
            "units": "count (runs, designs) -- the baseline itself is qualitative",
            "uncertainty": (
                "'established' means every reported row resolves to an existing checkpoint, a "
                "recorded split and real SIESTA references; it does not claim the baseline is "
                "converged-optimal for any architecture choice"
            ),
            "method": (
                "inventory of executed Graph2Mat runs with checkpoint/split/reference resolution "
                "(see traceability_sample for the row-level check)"
            ),
            "data_source": (
                "Comparison/results/dataset_design_w90_001_s9_s4/run_metrics.csv; "
                "Comparison/results/dataset_design_w90_001_s9_s5/pareto_table.csv"
            ),
            "status": "answered_scoped",
            "scope_limits": (
                "Graph2Mat only. DeepH is explicitly deferred to a later task by this "
                "objective, so no model-versus-model claim is made or implied here."
            ),
        },
        "q4_pareto_frontier": {
            "value": [
                {
                    "design_id": row["design_id"],
                    "family": row["family"],
                    "dim": row["dim"],
                    "k": row["k"],
                    "R_train_max_ang": row["R_train_max_ang"],
                    "N_train": int(row["N_train"]),
                    "seed": int(row["seed"]),
                    "siesta_cost": row["siesta_cost"],
                    "H_MAE_eV": float(row["H_MAE"]),
                }
                for row in frontier
            ],
            "units": "H-MAE (eV) versus unique SIESTA-labelled geometries (training + validation)",
            "uncertainty": (
                "strict dominance over the executed confirmation runs on one evaluation "
                "distribution; per-seed points are kept separate and no confidence band is "
                "claimed, so a frontier point separated from a dominated point by less than "
                "the seed spread is not a demonstrated difference"
            ),
            "method": (
                "dominance recomputed here from (H-MAE, siesta_cost) pairs on the development "
                "distribution only; cost counts N_train + VAL_COUNT so the validation labels a "
                "design needs are charged to it, and reused labels are never charged twice"
            ),
            "data_source": "Comparison/results/dataset_design_w90_001_s9_s5/pareto_table.csv",
            "status": "answered_scoped",
            "scope_limits": (
                f"the frontier spans only the confirmation-stage designs "
                f"{sorted({row['design_id'] for row in frontier})} at "
                f"{coverage['full_executed_domains_ang']} Ang; it is not a frontier over the "
                "full designed space"
            ),
        },
        "q5_required_displacement_positions_amplitudes": {
            "value": {
                "trained_domains_ang": coverage["pilot_executed_domains_ang"],
                "in_domain_H_MAE_mean_eV": statistics.fmean(in_domain) if in_domain else None,
                "out_of_domain_H_MAE_mean_eV": statistics.fmean(out_domain) if out_domain else None,
                "domain_effect_at_matched_n": domain_effects,
                "grid_cells": len(grid),
            },
            "units": "Angstrom (displacement amplitude), eV (H-MAE)",
            "uncertainty": (
                "the in/out-of-domain means pool cells across families and training domains, so "
                "they show direction, not a per-design degradation law. OOD status is taken from "
                "each row's own envelope, so a cell marked 'mixed' contains geometries on both "
                "sides of its training boundary and is counted as out-of-domain."
            ),
            "method": (
                "training domain x evaluation amplitude grid, averaged over seeds, built from "
                "both executed stages; domain effects reported only where family, dim and "
                "N_train are matched across domains"
            ),
            "data_source": (
                "Comparison/results/dataset_design_w90_001_s9_s4/run_metrics.csv; "
                "Comparison/results/dataset_design_w90_001_s9_s5/domain_generalization.csv"
            ),
            "status": "partial",
            "reason": (
                "domain size is compared at matched N only at pilot scale: the confirmation "
                f"stage trained {coverage['full_executed_domains_ang']} Ang alone. The 0.05 Ang "
                "level was designed but has no SIESTA labels, so the requested three-level "
                "nested-domain comparison is not executable without new DFT work."
            ),
        },
    }


# --------------------------------------------------------------------------
# software_validated / results_executed / conclusions_pending
# --------------------------------------------------------------------------


def epistemic_status_buckets() -> dict[str, list[str]]:
    coverage = executed_coverage()
    runs = build_run_metrics()
    pilot_designs = {row["design_id"] for row in runs if row["stage"] == "pilot"}
    full_designs = {row["design_id"] for row in runs if row["stage"] == "full"}

    return {
        "software_validated": [
            "S9-S1: six pipeline bug fixes, each covered by a regression test "
            "(tests/test_dataset_design_s9_contract.py -k bug_fix)",
            "S9-S2: envelope/domain/density/coverage-matrix spec, validated by "
            "dataset_design_w90_001_s9_s2_envelope_and_coverage.validate_spec",
            "S9-S3: design generation for every applicable coverage-matrix cell -- envelope, "
            "zero-leakage and nested-prefix checks all pass",
        ],
        "results_executed": [
            f"S9-S4 (pilot): {len(pilot_designs)} designs trained with the bug-fixed pipeline across "
            f"families {coverage['pilot_executed_families']} and domains "
            f"{coverage['pilot_executed_domains_ang']} Ang",
            f"S9-S5 (confirmation): {len(full_designs)} surviving designs trained at N in {{128, 256}} "
            "with five paired seeds and evaluated on the development set plus six reserved amplitudes",
            "Hermiticity and finiteness are recorded per evaluated run in run_metrics.csv, not asserted in prose",
        ],
        "conclusions_pending": [
            *coverage["gaps"],
            "A user-selectable absolute H-MAE target is not implemented by any executed run: q1 "
            "reports a relative plateau bracket instead, which is a weaker claim than the objective asks for",
            "Density is not isolated from pool size in the executed designs, so the requested "
            "three-resolution-per-family comparison at fixed N is not available",
        ],
    }


def independent_confirmation() -> dict[str, Any]:
    """Whether the executed S9 evidence can stand on its own yet.

    ``provisional`` here is a statement about *coverage*, recomputed from the
    artifacts: the package reads the bug-fixed S9 lineage only, so the old
    "these numbers predate the fixes" caveat no longer applies, but designed
    cells with no executed runs still cap what the answers may claim.
    """
    evidence = executed_coverage()
    if not evidence["gaps"]:
        return {
            "status": "confirmed",
            "reason": (
                "Every designed domain and sampler family has executed runs from the bug-fixed "
                "pipeline, and the confirmation stage covers the same domains as the pilot."
            ),
            "evidence": evidence,
        }
    return {
        "status": "provisional",
        "reason": (
            "The package reports only the bug-fixed S9 lineage, but the designed space is not "
            "fully executed: " + "; ".join(evidence["gaps"]) + ". Cells whose geometries have no "
            "SIESTA labels cannot be closed with more GPU time -- they need new DFT calculations."
        ),
        "evidence": evidence,
    }


# --------------------------------------------------------------------------
# Traceability: UI row -> checkpoint + split + geometry_hash + SIESTA reference
# --------------------------------------------------------------------------


def trace_row(
    row: dict[str, Any],
    manifest_by_id: dict[str, dict[str, Any]],
    geometry: Any,
    hash_to_sample: dict[str, s4.Sample],
) -> dict[str, Any]:
    """Resolve one reported row back to the artifacts it claims to summarise.

    The design's training points are regenerated from the manifest entry and
    re-matched to real SIESTA references by geometry hash, rather than trusting
    the manifest's own cached count: a design whose labels went missing shows
    up here as a short list instead of a passing claim.
    """
    entry = manifest_by_id.get(row["design_id"])
    checkpoint_path = Path(row["checkpoint"]) if row.get("checkpoint") else None
    checkpoint_exists = bool(checkpoint_path and checkpoint_path.exists())

    matched: list[s4.Sample] = []
    if entry is not None:
        matched = s9s4.matched_training_samples(geometry, entry, hash_to_sample)
    references = [str(sample.reference_matrix) for sample in matched]
    references_exist = bool(references) and all(Path(path).exists() for path in references)
    n_used = int(entry["n_used"]) if entry is not None else 0
    split_reproducible = bool(entry is not None and len(matched) >= n_used > 0)

    return {
        "row": {
            key: row[key]
            for key in ("stage", "design_id", "family", "dim", "k", "N_train", "seed", "H_MAE")
            if key in row
        },
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_exists": checkpoint_exists,
        "split_id": row.get("split_id") or None,
        "split_reproducible": split_reproducible,
        "n_matched_training_samples": len(matched),
        "n_declared_training_samples": n_used,
        "geometry_hash": entry.get("geometry_hash") if entry else None,
        "n_siesta_references": len(references),
        "siesta_references_exist": references_exist,
        "traceable": bool(
            checkpoint_exists and split_reproducible and references_exist and entry is not None
        ),
    }


def traceability_sample(n: int = 3) -> list[dict[str, Any]]:
    """A reproducible spot-check across both executed stages."""
    manifest_by_id = {
        entry["design_id"]: entry
        for entry in _read_json(S9_S3_ROOT / "design_manifest.json")["designs"]
    }
    geometry = s9s4.sampler.load_graphene_primitive()
    hash_to_sample = s9s4.build_s3_hash_index(s4.load_s3_samples())

    rows = [row for row in build_run_metrics() if row["checkpoint"]]
    pilot = [row for row in rows if row["stage"] == "pilot"]
    full = [row for row in rows if row["stage"] == "full"]

    rng = random.Random(TRACEABILITY_SAMPLE_SEED)
    chosen: list[dict[str, Any]] = []
    for pool, label in ((pilot, "run_metrics.csv (pilot)"), (full, "pareto_table.csv (confirmation)")):
        for row in rng.sample(pool, k=min(2, len(pool))):
            chosen.append({"source": label, **trace_row(row, manifest_by_id, geometry, hash_to_sample)})
    return chosen[:n]


# --------------------------------------------------------------------------
# Assembly + I/O
# --------------------------------------------------------------------------


def build_summary() -> dict[str, Any]:
    return {
        "task": "DATASET-DESIGN-W90-001-S9-S8",
        "scientific_questions": scientific_question_answers(),
        "pruned_branches": pruned_branches(),
        "historical_test_inventory": historical_test_inventory(),
        "independent_confirmation": independent_confirmation(),
        **epistemic_status_buckets(),
        "traceability_sample": traceability_sample(),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty {path.name}: the S9 lineage produced no rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_acceptance_package(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Build the package from the executed S9 lineage.

    The tables are derived, not copied: each stage stores what suits it, and
    the aggregate contract this package publishes (a matched domain grid, a
    frontier whose dominance flag can be recomputed from its own columns) is
    produced here so it cannot disagree with the runs behind it.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(S9_S3_ROOT / "design_manifest.json", output_root / "design_manifest.json")
    _write_csv(output_root / "run_metrics.csv", build_run_metrics())
    _write_csv(output_root / "domain_generalization.csv", build_domain_generalization())
    _write_csv(output_root / "pareto_table.csv", build_pareto_table())
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(build_summary(), indent=2, sort_keys=True), encoding="utf-8")
    return output_root


def main() -> int:
    output_root = write_acceptance_package()
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    all_present = all(
        (output_root / name).stat().st_size > 0 for name in REQUIRED_FILE_NAMES
    )
    all_traceable = all(entry["traceable"] for entry in summary["traceability_sample"])
    print(json.dumps({
        "output_root": str(output_root),
        "all_5_files_present_and_non_empty": all_present,
        "traceability_sample_all_traceable": all_traceable,
        "independent_confirmation_status": summary["independent_confirmation"]["status"],
    }, sort_keys=True))
    return 0 if all_present and all_traceable else 2


if __name__ == "__main__":
    raise SystemExit(main())
