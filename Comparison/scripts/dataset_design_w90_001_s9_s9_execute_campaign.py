#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9-S9 -- execution engine for the S9 campaign.

Orchestrates S9-S3 (design generation) -> S9-S4 (pilot screening) -> S9-S5
(Pareto-set scale training) -> S9-S8 (final acceptance package) ->
``validate_dataset_design_s9.py`` as one resumable sequence, instead of each
being invoked by hand. Every phase already does its own real work (SIESTA
submission, GPU training, aggregation) and its own resume-by-checkpoint /
resume-by-materialized-SIESTA skip (S9-S4's ``existing_checkpoints`` check,
S9-S5's same pattern, ``materialize_and_run``'s idempotent submission) -- this
script does not re-implement any of that. It adds exactly what a bare
sequence of ``python step.py`` calls doesn't give you:

  - a hard gate on PID 1355284 (stdlib ``os.kill(pid, 0)`` liveness poll) so
    this campaign never launches competing training against it;
  - one GPU preflight recorded up front (``s4.torch_backend_preflight``,
    already the shared helper every phase calls internally too), with an
    explicit stderr warning on any CPU fallback rather than a silent one;
  - each phase run as its own subprocess with a 6-hour timeout, so a stuck
    phase is killed and recorded rather than blocking past the command-length
    limit this task operates under;
  - a running wall-time budget (24h) checked before starting each phase, so a
    slow campaign aborts before a new phase rather than blowing the budget
    mid-phase;
  - a campaign-wide cost tracker (``cost_tracker.json``) built from the
    already-written ``run_metrics.csv`` / ``cost_ledger.json`` / on-disk
    structure counts -- no new SIESTA/hashing work -- distinguishing
    per-(design, N) instance cost from the deduplicated campaign union, per
    S9-S2's ``COST_SCHEMA``;
  - one aggregated manifest (``s9_s9_campaign_manifest.json``) recording
    every phase's start/end/wall-time/exit status, for resumability across
    runs of this script itself.

Adaptive pruning is not re-decided here: it is S9-S4's own
``dominated``/``envelope_fail``/``convergence_fail``/``budget`` vocabulary
(``pilot_pruned.json``), which already realizes this task's three stopping
rules (Pareto dominance, physical/data gate failure, non-improving budget
exhaustion) -- this script only aggregates and reports it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402
import dataset_design_w90_001_s5_frozen_tests_generalization as s5  # noqa: E402
import dataset_design_w90_001_s9_s4_pilot_screening as s94  # noqa: E402
import dataset_design_w90_001_s9_s5_pareto_training as s95  # noqa: E402
import run_w90_displacement_siesta_pilot as pilot  # noqa: E402
import w90_displacement_sampler_family as sampler  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_s9"
DEFAULT_WAIT_PID = 1355284
WALL_BUDGET_SECONDS = 24 * 3600
PER_COMMAND_TIMEOUT_SECONDS = 6 * 3600

# (name, script filename, needs --siesta-command/--workers)
PHASE_SPECS: tuple[tuple[str, str, bool], ...] = (
    ("s9_s3_design_generation", "dataset_design_w90_001_s9_s3_design_generation.py", False),
    ("s9_s4_pilot_screening", "dataset_design_w90_001_s9_s4_pilot_screening.py", True),
    ("s9_s5_pareto_training", "dataset_design_w90_001_s9_s5_pareto_training.py", True),
    ("s9_s8_final_acceptance", "dataset_design_w90_001_s9_s8_final_acceptance.py", False),
    ("s9_validator", "validate_dataset_design_s9.py", False),
)


# --------------------------------------------------------------------------
# Process-wait gate (stdlib liveness poll -- no new dependency)
# --------------------------------------------------------------------------


def wait_for_process(pid: int, *, poll_seconds: float = 30.0) -> dict[str, Any]:
    def _is_running() -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists, owned by another user -- still must not compete

    start = time.monotonic()
    was_running_at_start = _is_running()
    polls = 0
    while _is_running():
        polls += 1
        time.sleep(poll_seconds)
    return {
        "pid": pid,
        "was_running_at_start": was_running_at_start,
        "waited_seconds": time.monotonic() - start,
        "polls": polls,
        "method": "os.kill(pid, 0) liveness poll (stdlib)",
    }


# --------------------------------------------------------------------------
# Phase runner: each step as its own subprocess, bounded by the 6h command limit
# --------------------------------------------------------------------------


def run_phase(name: str, cmd: list[str], *, timeout_seconds: float) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    timed_out = False
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_seconds)
        returncode: int | None = result.returncode
        stdout_tail, stderr_tail = result.stdout[-4000:], result.stderr[-4000:]
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout_tail = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
    return {
        "name": name,
        "command": cmd,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - start,
        "returncode": returncode,
        "timed_out": timed_out,
        "ok": (not timed_out) and returncode == 0,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


# --------------------------------------------------------------------------
# Cost tracker: aggregated from already-written artifacts, no recomputation
# --------------------------------------------------------------------------


def _hash_structures_dir(structures_dir: Path) -> set[str]:
    """Real geometry hashes of every ``RUN.fdf`` under one materialized structures/ dir.

    Pure disk read (no new SIESTA): the same ``geometry_hash``/``_positions_from_run_fdf``
    S9-S4/S9-S5 already use for their own leakage/dedup checks, applied to files these
    phases already wrote to disk.
    """

    hashes: set[str] = set()
    if not structures_dir.exists():
        return hashes
    for struct_dir in structures_dir.iterdir():
        run_fdf = struct_dir / "RUN.fdf"
        if run_fdf.exists():
            hashes.add(s5.geometry_hash(s5._positions_from_run_fdf(run_fdf)))
    return hashes


def compute_cost_tracker() -> dict[str, Any]:
    # per-design declared cost (S9-S2 COST_SCHEMA.unique_siesta_ids: each design's own
    # count, no cross-design dedup), read verbatim from what S9-S4/S9-S5 already wrote.
    per_design_instance_ids: dict[str, int] = {}

    s94_metrics_path = s94.DEFAULT_OUTPUT_ROOT / "run_metrics.csv"
    if s94_metrics_path.exists():
        with s94_metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "ok":
                    continue
                try:
                    per_design_instance_ids[f"{row['design_id']}@N{row['N_train']}"] = int(row["siesta_cost"])
                except (KeyError, ValueError):
                    continue

    s95_ledger_path = s95.DEFAULT_OUTPUT_ROOT / "cost_ledger.json"
    if s95_ledger_path.exists():
        for entry in json.loads(s95_ledger_path.read_text(encoding="utf-8")):
            per_design_instance_ids[f"{entry['design_id']}@N{entry['N_train']}"] = int(entry["cost_siesta_unique"])

    # campaign_union_ids (S9-S2 COST_SCHEMA.campaign_union_cost: "union over designs in
    # campaign of geometry_ids(design)") -- real hash union of what was actually
    # consumed, not the size of the whole pre-existing S3 reference pool (most of which
    # no trainable design ever selected).
    manifest = s94.load_manifest()
    geometry = sampler.load_graphene_primitive()
    s3_pool = s4.load_s3_samples()
    hash_to_sample = s94.build_s3_hash_index(s3_pool)
    trainable, _budget_pruned = s94.select_candidates(manifest, geometry, hash_to_sample)
    pilot_used_hashes: set[str] = set()
    for entry in trainable:
        pilot_used_hashes.update(
            s5.geometry_hash(s5._positions_from_run_fdf(sample.run_fdf)) for sample in entry["_matched_samples"]
        )

    scale_pool_hashes: set[str] = set()
    training_pools_root = s95.DEFAULT_OUTPUT_ROOT / "training_pools"
    if training_pools_root.exists():
        for pool_dir in training_pools_root.iterdir():
            scale_pool_hashes |= _hash_structures_dir(pool_dir / "structures")

    evaluation_hashes: set[str] = set()
    evaluation_hashes |= _hash_structures_dir(s5.DEFAULT_OUTPUT_ROOT / s5.COMMON_VALIDATION_NAME / "structures")
    for amplitude_dir in sorted(s94.DEFAULT_OUTPUT_ROOT.glob("test_amplitude_*")):
        evaluation_hashes |= _hash_structures_dir(amplitude_dir / "structures")

    campaign_union_hashes = pilot_used_hashes | scale_pool_hashes | evaluation_hashes
    sum_per_design_ids = sum(per_design_instance_ids.values())
    campaign_union_ids = len(campaign_union_hashes)

    return {
        "definition": (
            "per-instance cost keyed by design_id@N (S9-S2 COST_SCHEMA.unique_siesta_ids); "
            "campaign_union_ids = real hash union of pilot-consumed + scaled-training-pool + "
            "evaluation-set geometries (S9-S2 COST_SCHEMA.campaign_union_cost, literally)"
        ),
        "pilot_used_unique_ids": len(pilot_used_hashes),
        "scale_training_pool_unique_ids": len(scale_pool_hashes),
        "evaluation_set_unique_ids": len(evaluation_hashes),
        "per_design_instance_ids": per_design_instance_ids,
        "sum_per_design_ids": sum_per_design_ids,
        "campaign_union_ids": campaign_union_ids,
        "reuse_credited": sum_per_design_ids > campaign_union_ids,
    }


def collect_pruned_branches() -> list[dict[str, Any]]:
    path = s94.DEFAULT_OUTPUT_ROOT / "pilot_pruned.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("pruned", [])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--siesta-command", default=pilot.DEFAULT_SIESTA_COMMAND)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--wait-pid", type=int, default=DEFAULT_WAIT_PID)
    parser.add_argument("--wait-poll-seconds", type=float, default=30.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-wait", action="store_true", help="Debug/testing: skip the process-wait gate.")
    args = parser.parse_args()

    campaign_start = time.monotonic()
    output_root: Path = args.output_root

    process_wait = (
        {"skipped": True, "reason": "--skip-wait"} if args.skip_wait
        else wait_for_process(args.wait_pid, poll_seconds=args.wait_poll_seconds)
    )
    print(json.dumps({"process_wait": process_wait}, sort_keys=True))

    backend_info = s4.torch_backend_preflight()
    print(json.dumps({"gpu_preflight": backend_info}, sort_keys=True))
    if backend_info["effective_backend"] != "gpu":
        print(f"[S9-S9][WARN] CPU fallback in effect: {backend_info['reason']}", file=sys.stderr)

    common_args = ["--siesta-command", args.siesta_command, "--workers", str(args.workers)]
    phases: list[dict[str, Any]] = []
    aborted_reason: str | None = None

    for name, script_name, needs_common_args in PHASE_SPECS:
        elapsed = time.monotonic() - campaign_start
        if elapsed > WALL_BUDGET_SECONDS:
            aborted_reason = f"wall budget ({WALL_BUDGET_SECONDS}s) exceeded before phase {name}"
            break
        cmd = [sys.executable, str(SCRIPT_DIR / script_name)]
        if needs_common_args:
            cmd += common_args
        phase = run_phase(name, cmd, timeout_seconds=PER_COMMAND_TIMEOUT_SECONDS)
        phases.append(phase)
        print(json.dumps({"phase_complete": name, "ok": phase["ok"], "wall_seconds": phase["wall_seconds"]}, sort_keys=True))
        if not phase["ok"]:
            aborted_reason = f"phase {name} failed (returncode={phase['returncode']}, timed_out={phase['timed_out']})"
            break

    try:
        cost_tracker = compute_cost_tracker()
    except Exception as exc:  # noqa: BLE001 - report, don't crash the manifest write
        cost_tracker = {"error": f"{type(exc).__name__}: {exc}"}
    pruned_branches = collect_pruned_branches()

    total_wall_seconds = time.monotonic() - campaign_start
    no_timeout = all(not p["timed_out"] for p in phases)
    all_ok = aborted_reason is None and all(p["ok"] for p in phases)
    summary = {
        "task": "DATASET-DESIGN-W90-001-S9-S9",
        "process_wait": process_wait,
        "gpu_preflight": backend_info,
        "wall_budget_seconds": WALL_BUDGET_SECONDS,
        "per_command_timeout_seconds": PER_COMMAND_TIMEOUT_SECONDS,
        "total_wall_seconds": total_wall_seconds,
        "within_wall_budget": total_wall_seconds <= WALL_BUDGET_SECONDS,
        "no_command_exceeded_timeout": no_timeout,
        "aborted_reason": aborted_reason,
        "phases": phases,
        "cost_tracker": cost_tracker,
        "n_pruned_branches": len(pruned_branches),
        "pruned_branches_sample": pruned_branches[:5],
        "all_phases_ok": all_ok,
    }
    write_json(output_root / "s9_s9_campaign_manifest.json", summary)
    write_json(output_root / "cost_tracker.json", cost_tracker)

    reuse_credited = bool(cost_tracker.get("reuse_credited"))
    print(json.dumps({
        "all_phases_ok": all_ok,
        "within_wall_budget": summary["within_wall_budget"],
        "no_command_exceeded_timeout": no_timeout,
        "n_pruned_branches": len(pruned_branches),
        "reuse_credited": reuse_credited,
        "sum_per_design_ids": cost_tracker.get("sum_per_design_ids"),
        "campaign_union_ids": cost_tracker.get("campaign_union_ids"),
        "aborted_reason": aborted_reason,
    }, sort_keys=True, default=str))

    ok = all_ok and summary["within_wall_budget"] and no_timeout and reuse_credited and len(pruned_branches) >= 1
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
