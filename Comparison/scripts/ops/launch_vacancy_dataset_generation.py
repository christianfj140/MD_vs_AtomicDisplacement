#!/usr/bin/env python
"""Generate the graphene 5x5 monovacancy MD datasets, optionally waiting for the queue.

The vacancy has never had datasets of its own: its 20 existing snapshots are static
single points derived from pristine frames (``derived_pristine_monovacancy_static_siesta``,
``relaxed: false``), so no model has ever been trained on vacancy data. This runs the
normal dataset-generation pipeline (``run_mode: generate_datasets_only``) against the
graphene_5x5_vacancy preset with the same split and thermal configuration as the pristine
campaign, so the resulting curve is comparable.

Cost, from the pristine campaign's measured dataset_sweep_summary: ~31 s per snapshot.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))

PAYLOAD = REPO_ROOT / "Comparison/config/graphene_5x5_vacancy_snapshot_scaling_payload.json"


def wait_for(pattern: str, log) -> None:
    while subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0:
        log(f"en cola, esperando a: {pattern}")
        time.sleep(120)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=PAYLOAD)
    parser.add_argument(
        "--wait-for",
        default="regenerate_derivative_siesta_references.py",
        help="pgrep -f pattern to wait on; empty string starts immediately.",
    )
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        with args.log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    if args.wait_for:
        wait_for(args.wait_for, log)

    from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner  # noqa: PLC0415

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    sizes = [r.get("size") for r in (payload.get("dataset_sweep") or {}).get("recipes") or []]
    log(f"generando datasets de vacante, tamanos {sizes} (material {payload.get('material_preset')})")

    runner = Graph2MatDeepHBenchmarkRunner()
    runner.start(payload)
    last = ""
    while runner.status().get("running"):
        status = runner.status()
        stamp = f"{status.get('stage')} | {status.get('phase') or ''}"
        if stamp != last:
            log(f"  {stamp}")
            last = stamp
        time.sleep(30)

    status = runner.status()
    log(f"TERMINADO rc={status.get('returncode')} error={status.get('error')}")
    return 0 if not status.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
