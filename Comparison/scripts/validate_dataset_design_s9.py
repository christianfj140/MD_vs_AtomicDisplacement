#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9 semantic validator -- the "S9 validator" referenced by
S9-S8's task contract, run as one of its four required validation commands
(regression tests, S9 contract tests, this validator, artifact-presence
check).

Standalone, no arguments: re-validates the S9-S2 envelope/coverage spec
in-memory (``s2.validate_spec``, raises on any violation), rebuilds the cheap
S9-S8 acceptance package (aggregation-only, reads existing CSV/JSON verbatim,
does not launch a training campaign) into
``Comparison/results/dataset_design_w90_001_s9/``, and then runs semantic
checks over those five real, on-disk artifacts: manifest schema and finite
values, split-leakage flags, domain/resolution sweep coverage, checkpoint
reference existence, recomputed SIESTA cost + Pareto dominance, and the
summary's scientific-question status/limitations contract.

Every check below is a *validity* check (schema violation, missing evidence,
an inconsistency between two artifacts that should agree, or an
implementation defect such as double-counted cost or a fabricated dominance
flag) -- never an *outcome* check. A scientific question honestly reporting
``not_reached``/``insufficient_evidence``/``partial`` is a valid result and
must never make this validator fail; only an unrecognized status, or a status
claiming success without the evidence to back it, is rejected.

Exit code 0 iff every sub-check passes; always prints one JSON summary line.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402
import dataset_design_w90_001_s9_s2_envelope_and_coverage as s2  # noqa: E402
import dataset_design_w90_001_s9_s4_pilot_screening as s94  # noqa: E402
import dataset_design_w90_001_s9_s8_final_acceptance as s8  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "Comparison" / "results" / "dataset_design_w90_001_s9"
REQUIRED_FILES = ("design_manifest.json", "run_metrics.csv", "domain_generalization.csv", "pareto_table.csv", "summary.json")
REQUIRED_QUESTION_FIELDS = ("value", "units", "uncertainty", "method", "data_source", "status")
EXPECTED_QUESTIONS = {
    "q1_minimum_training_set_size",
    "q2_distribution_density",
    "q3_baseline_graph2mat_md_equivalence",
    "q4_pareto_frontier",
    "q5_required_displacement_positions_amplitudes",
}
REQUIRED_DESIGN_FIELDS = (
    "design_id", "family", "dim", "k", "r_train_max_ang", "density_level",
    "n_used", "n_generated", "geometry_hash", "envelope_check_passed",
    "envelope_max_displacement_ang", "n_overlap_with_frozen_or_validation",
    "nested_verified", "diagnostics",
)

# Precision-selector states: the confidence/completeness level a scientific-question
# answer claims for itself. An honest not-met/partial/insufficient status is a valid
# *outcome*, never rejected on its own; only an unrecognized status, or "answered_scoped"
# with no supporting value, is a validity defect (an unmet threshold dressed as success).
OUTCOME_NOT_MET_STATUSES = {"not_reached", "insufficient_evidence", "partial"}
VALID_QUESTION_STATUSES = OUTCOME_NOT_MET_STATUSES | {"answered_scoped"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# --------------------------------------------------------------------------
# Cost-accounting utilities (S9-S2's COST_SCHEMA, made concrete/checkable)
# --------------------------------------------------------------------------


def unique_siesta_cost(geometry_hashes: Iterable[str]) -> int:
    """COST_SCHEMA.unique_siesta_ids: distinct displaced-geometry ids for one design."""

    return len(set(geometry_hashes))


def campaign_union_cost(designs_geometry_hashes: Iterable[Iterable[str]]) -> int:
    """COST_SCHEMA.campaign_union_cost: dedup once across the whole campaign.

    A naive ``sum(unique_siesta_cost(d) for d in designs)`` (COST_SCHEMA's
    ``per_design_cost``) double-counts any geometry shared by more than one
    design (e.g. a training pool reusing a frozen test's displacement); this
    is the correct total.
    """

    union: set[str] = set()
    for hashes in designs_geometry_hashes:
        union.update(hashes)
    return len(union)


def detect_split_leakage(train_hashes: Iterable[str], test_hashes: Iterable[str]) -> set[str]:
    return set(train_hashes) & set(test_hashes)


# --------------------------------------------------------------------------
# S2 envelope/coverage spec (in-memory, no artifacts on disk)
# --------------------------------------------------------------------------


def validate_s2_envelope_spec() -> dict[str, Any]:
    try:
        spec = s2.build_spec()
        s2.validate_spec(spec)
    except Exception as exc:  # noqa: BLE001 - report, don't hide
        return {"ok": False, "problems": [str(exc)]}
    return {"ok": True, "problems": [], "coverage_matrix_rows": len(spec["coverage_matrix"])}


# --------------------------------------------------------------------------
# design_manifest.json: schema, finite values, leakage flags, sweep coverage
# --------------------------------------------------------------------------


def check_domain_resolution_is_a_sweep(
    coverage: dict[tuple[str, str, int], set[tuple[float, Any]]],
    expected: dict[tuple[str, str, int], set[tuple[float, Any]]],
) -> list[str]:
    """Any group S2 expects to be swept across >1 domain/resolution cell must show >1

    distinct cell in the manifest too -- otherwise a single (domain, density) point is
    being presented as a sweep.
    """

    problems: list[str] = []
    for key, expected_cells in expected.items():
        if len(expected_cells) <= 1:
            continue
        actual_cells = coverage.get(key, set())
        if len(actual_cells) <= 1:
            problems.append(
                f"{key}: S2 spec expects a multi-cell domain/resolution sweep "
                f"({len(expected_cells)} cells), manifest shows only {len(actual_cells)}"
            )
    return problems


def validate_design_manifest(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"ok": False, "problems": [f"missing or empty {path}"]}
    manifest = _read_json(path)
    problems: list[str] = []

    designs = manifest.get("designs", [])
    if manifest.get("n_designs") != len(designs):
        problems.append(f"n_designs={manifest.get('n_designs')} != len(designs)={len(designs)}")
    if not manifest.get("leakage_free"):
        problems.append("manifest declares leakage_free=False")

    coverage: dict[tuple[str, str, int], set[tuple[float, Any]]] = {}
    for entry in designs:
        design_id = entry.get("design_id", "<unknown>")
        missing = [field for field in REQUIRED_DESIGN_FIELDS if field not in entry]
        if missing:
            problems.append(f"{design_id}: missing fields {missing}")
            continue

        for field in ("r_train_max_ang", "envelope_max_displacement_ang"):
            if not _finite(entry[field]):
                problems.append(f"{design_id}: non-finite {field}={entry[field]!r}")
        for field in ("n_used", "n_generated"):
            if not (isinstance(entry[field], int) and entry[field] >= 0):
                problems.append(f"{design_id}: invalid {field}={entry[field]!r}")
        if isinstance(entry.get("n_used"), int) and isinstance(entry.get("n_generated"), int):
            if entry["n_used"] > entry["n_generated"]:
                problems.append(f"{design_id}: n_used ({entry['n_used']}) > n_generated ({entry['n_generated']})")
        if not entry.get("envelope_check_passed"):
            problems.append(f"{design_id}: envelope_check_passed is False")
        if not entry.get("geometry_hash"):
            problems.append(f"{design_id}: empty geometry_hash")

        overlap = entry.get("n_overlap_with_frozen_or_validation")
        if not isinstance(overlap, int) or overlap != 0:
            problems.append(f"{design_id}: n_overlap_with_frozen_or_validation={overlap!r} (train/test leakage)")

        state = entry.get("state", "executed")
        if state in ("executed", "pending"):
            key = (entry["family"], entry["dim"], int(entry["k"]))
            coverage.setdefault(key, set()).add((float(entry["r_train_max_ang"]), entry["density_level"]))

    try:
        expected: dict[tuple[str, str, int], set[tuple[float, Any]]] = {}
        for row in s2.build_spec()["coverage_matrix"]:
            if row["state"] not in ("executed", "pending"):
                continue
            key = (row["family"], row["dim"], int(row["k"]))
            expected.setdefault(key, set()).add((float(row["domain_r_train_max_ang"]), row["resolution"]))
        problems.extend(check_domain_resolution_is_a_sweep(coverage, expected))
    except Exception as exc:  # noqa: BLE001 - report, don't hide
        problems.append(f"could not cross-check coverage against S2 spec: {exc}")

    return {"ok": not problems, "problems": problems[:30], "n_designs": manifest.get("n_designs")}


# --------------------------------------------------------------------------
# run_metrics.csv: status/finite-value schema, checkpoint reference existence
# --------------------------------------------------------------------------


def validate_run_metrics(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"ok": False, "problems": [f"missing or empty {path}"]}
    rows = _read_csv(path)
    problems: list[str] = []
    if not rows:
        problems.append("run_metrics.csv has no rows")
    for i, row in enumerate(rows):
        if row.get("status") != "ok":
            continue
        for field in ("H_MAE", "H_RMSE"):
            value = row.get(field)
            try:
                if not math.isfinite(float(value)):
                    problems.append(f"row {i}: non-finite {field}={value!r}")
            except (TypeError, ValueError):
                problems.append(f"row {i}: unparseable {field}={value!r}")
        checkpoint = row.get("checkpoint")
        if checkpoint and not Path(checkpoint).exists():
            problems.append(f"row {i}: checkpoint reference does not exist: {checkpoint}")
    return {"ok": not problems, "problems": problems[:20], "n_rows": len(rows)}


# --------------------------------------------------------------------------
# domain_generalization.csv: matched-position aggregate schema
# --------------------------------------------------------------------------


def validate_domain_generalization(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"ok": False, "problems": [f"missing or empty {path}"]}
    rows = _read_csv(path)
    problems: list[str] = []
    if not rows:
        problems.append("domain_generalization.csv has no rows")
    for i, row in enumerate(rows):
        n_models = row.get("n_models")
        try:
            if int(n_models) <= 0:
                problems.append(f"row {i}: n_models={n_models!r} must be positive")
        except (TypeError, ValueError):
            problems.append(f"row {i}: unparseable n_models={n_models!r}")
        value = row.get("H_MAE_mean")
        try:
            if not math.isfinite(float(value)):
                problems.append(f"row {i}: non-finite H_MAE_mean={value!r}")
        except (TypeError, ValueError):
            problems.append(f"row {i}: unparseable H_MAE_mean={value!r}")
    return {"ok": not problems, "problems": problems[:20], "n_rows": len(rows)}


# --------------------------------------------------------------------------
# pareto_table.csv: recomputed SIESTA cost + recomputed Pareto dominance
# --------------------------------------------------------------------------


def validate_pareto_costs_and_dominance(pareto_path: Path, run_metrics_path: Path) -> dict[str, Any]:
    if not pareto_path.exists() or pareto_path.stat().st_size == 0:
        return {"ok": False, "problems": [f"missing or empty {pareto_path}"]}
    pareto_rows = _read_csv(pareto_path)
    run_rows = _read_csv(run_metrics_path) if run_metrics_path.exists() else []
    ok_run_keys = {
        (r["family"], r["dim"], r["k"], r["seed"], r["N_train"])
        for r in run_rows if r.get("status") == "ok"
    }

    problems: list[str] = []
    scores: dict[str, tuple[float, int]] = {}
    for i, row in enumerate(pareto_rows):
        try:
            declared_cost = int(row["siesta_cost"])
        except (KeyError, ValueError):
            problems.append(f"row {i}: unparseable siesta_cost={row.get('siesta_cost')!r}")
            continue
        # Cost recomputed from the underlying trained run: siesta_cost must equal a real
        # N_train + VAL_COUNT for an actually-trained (family, dim, k, seed) combo. A
        # double-counted/inflated declared cost (e.g. summing two designs that share
        # geometries) will not match any real N_train and is caught here.
        n_train = declared_cost - s4.VAL_COUNT
        key = (row.get("family"), row.get("dim"), row.get("k"), row.get("seed"), str(n_train))
        if key not in ok_run_keys:
            problems.append(f"row {i}: cost {declared_cost} has no matching ok run_metrics row for {key}")

        try:
            h_mae = float(row["H_MAE"])
        except (KeyError, ValueError):
            problems.append(f"row {i}: unparseable H_MAE={row.get('H_MAE')!r}")
            continue
        scores[f"row{i}"] = (h_mae, declared_cost)

    recomputed_dominated = s94.flag_pareto_dominated(scores)
    for i, row in enumerate(pareto_rows):
        if f"row{i}" not in scores:
            continue
        declared = str(row.get("pareto_dominated", "")).strip().lower() == "true"
        actual = f"row{i}" in recomputed_dominated
        if declared != actual:
            problems.append(f"row {i}: declared pareto_dominated={declared} but recomputed={actual}")

    return {"ok": not problems, "problems": problems[:20], "n_rows": len(pareto_rows)}


# --------------------------------------------------------------------------
# summary.json: scientific-question precision-selector states, final-test
# provenance (traceability_sample), and declared limitations
# --------------------------------------------------------------------------


def _check_summary_contract(summary: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    questions = summary.get("scientific_questions", {})
    if set(questions) != EXPECTED_QUESTIONS:
        problems.append(f"scientific_questions keys mismatch: {sorted(questions)}")
    for name, answer in questions.items():
        missing = [field for field in REQUIRED_QUESTION_FIELDS if field not in answer]
        if missing:
            problems.append(f"{name} missing fields {missing}")
            continue
        status = answer["status"]
        if status not in VALID_QUESTION_STATUSES:
            problems.append(f"{name}: unrecognized status {status!r}")
        elif status in OUTCOME_NOT_MET_STATUSES:
            # An honest not-met/partial/insufficient outcome is valid -- but it must
            # come with a declared limitation, not a silent empty field.
            if not (answer.get("reason") or answer.get("uncertainty")):
                problems.append(f"{name}: status {status!r} declares no limitation (reason/uncertainty)")
        elif status == "answered_scoped" and answer.get("value") in (None, "", []):
            problems.append(f"{name}: status 'answered_scoped' but value is empty (unmet threshold presented as success)")

    if not summary.get("pruned_branches"):
        problems.append("pruned_branches is empty")
    else:
        for entry in summary["pruned_branches"]:
            if not entry.get("reason_category") or not entry.get("evidence_reference"):
                problems.append(f"pruned_branches entry missing reason/evidence: {entry.get('scope')}")
                break

    inventory = summary.get("historical_test_inventory", [])
    if not any("development" in entry.get("role", "") for entry in inventory):
        problems.append("historical_test_inventory does not document development-set usage")

    confirmation = summary.get("independent_confirmation", {})
    if confirmation.get("status") not in ("confirmed", "provisional"):
        problems.append(f"independent_confirmation.status invalid: {confirmation.get('status')!r}")
    if not confirmation.get("reason"):
        problems.append("independent_confirmation declares no reason (declared limitations)")

    for bucket in ("software_validated", "results_executed", "conclusions_pending"):
        if not summary.get(bucket):
            problems.append(f"{bucket} is empty")

    # Final-test provenance: every sampled result row must trace back to a real
    # checkpoint, a reproducible split, and existing SIESTA reference data.
    sample = summary.get("traceability_sample", [])
    if len(sample) != 3:
        problems.append(f"traceability_sample has {len(sample)} rows, expected 3")
    for entry in sample:
        if not entry.get("traceable"):
            problems.append(f"traceability_sample row not traceable: {entry.get('row')}")
    return problems


def validate_summary_contract(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"ok": False, "problems": [f"missing or empty {path}"]}
    problems = _check_summary_contract(_read_json(path))
    return {"ok": not problems, "problems": problems}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main() -> int:
    output_root = s8.write_acceptance_package()

    results: dict[str, Any] = {
        "s2_envelope_spec": validate_s2_envelope_spec(),
        "design_manifest": validate_design_manifest(output_root / "design_manifest.json"),
        "run_metrics": validate_run_metrics(output_root / "run_metrics.csv"),
        "domain_generalization": validate_domain_generalization(output_root / "domain_generalization.csv"),
        "pareto_costs_and_dominance": validate_pareto_costs_and_dominance(
            output_root / "pareto_table.csv", output_root / "run_metrics.csv"
        ),
        "summary_contract": validate_summary_contract(output_root / "summary.json"),
    }
    missing_files = [
        name for name in REQUIRED_FILES
        if not (output_root / name).exists() or (output_root / name).stat().st_size == 0
    ]
    if missing_files:
        results["required_files"] = {"ok": False, "problems": [f"missing or empty: {missing_files}"]}

    ok = not missing_files and all(section["ok"] for section in results.values())
    print(json.dumps({"ok": ok, "output_root": str(output_root), **results}, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
