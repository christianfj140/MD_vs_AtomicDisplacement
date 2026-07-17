#!/usr/bin/env python3
"""Backfill null DeepH validation-cost telemetry from on-disk TensorBoard logs.

Historical DeepH runs have ``epochs_trained``/``best_validation_*`` as null in
``telemetry/deeph.json`` because the extractor only scanned for CSV logs while
DeepH-pack writes TensorBoard events (fixed in g2m_deeph_telemetry). The
training logs are still on disk, so those fields are recoverable exactly.

Only null fields are filled; nothing else in the files is touched. The embedded
telemetry copies inside ``training_sweep_manifest.json`` are updated to match,
so both sources stay consistent. Dry-run by default; pass --apply to write.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))

from g2m_deeph_telemetry import extract_deeph_validation_cost  # noqa: E402

BACKFILL_FIELDS = (
    "epochs_trained",
    "best_validation_epoch",
    "best_validation_step",
    "best_validation_value",
    "wall_clock_seconds_to_best_validation",
)
BACKFILL_MARKER = {
    "source": "deeph_tensorboard_validation_events",
    "script": "Comparison/scripts/ops/backfill_deeph_epochs_trained.py",
}


def _fill_nulls(target: dict[str, Any], cost: dict[str, Any]) -> list[str]:
    filled = []
    for key in BACKFILL_FIELDS:
        if target.get(key) is None and cost.get(key) is not None:
            target[key] = cost[key]
            filled.append(key)
    if filled:
        if target.get("validation_metric") is None:
            target["validation_metric"] = cost.get("selection_metric")
        target["telemetry_backfill"] = {
            **BACKFILL_MARKER,
            "fields": filled,
            "backfilled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    return filled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "Comparison" / "results")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run report)")
    args = parser.parse_args()

    # Pass 1: per-run telemetry files.
    updated_costs: dict[str, dict[str, Any]] = {}
    n_seen = n_filled = n_no_logs = 0
    for path in sorted(args.results_root.rglob("telemetry/deeph.json")):
        try:
            telemetry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        n_seen += 1
        if telemetry.get("epochs_trained") is not None:
            continue
        run_root = Path(str(telemetry.get("run_root") or path.parent.parent))
        train_dir = run_root / "deeph" / "train"
        if not train_dir.is_dir():
            n_no_logs += 1
            continue
        cost = extract_deeph_validation_cost(train_dir)
        filled = _fill_nulls(telemetry, cost)
        if not filled:
            n_no_logs += 1
            continue
        n_filled += 1
        updated_costs[str(run_root)] = cost
        print(f"[{'APPLY' if args.apply else 'DRY'}] {path.relative_to(args.results_root)}: {', '.join(filled)}")
        if args.apply:
            path.write_text(json.dumps(telemetry, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Pass 2: embedded copies in sweep manifests, kept consistent with pass 1.
    n_manifests = 0
    for path in sorted(args.results_root.rglob("training_sweep_manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        changed = False
        for run in manifest.get("runs") or []:
            if run.get("model") != "deeph":
                continue
            embedded = run.get("telemetry")
            cost = updated_costs.get(str(run.get("run_root")))
            if not isinstance(embedded, dict) or cost is None or embedded.get("epochs_trained") is not None:
                continue
            if _fill_nulls(embedded, cost):
                changed = True
        if changed:
            n_manifests += 1
            print(f"[{'APPLY' if args.apply else 'DRY'}] manifest {path.relative_to(args.results_root)}")
            if args.apply:
                path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"\ndeeph telemetry files: {n_seen} seen, {n_filled} backfillable, "
        f"{n_no_logs} without recoverable logs; manifests to update: {n_manifests}"
        + ("" if args.apply else "  (dry-run: nothing written, use --apply)")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
