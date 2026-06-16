#!/usr/bin/env python3
"""Run one Graph2Mat-vs-DeepH payload and persist status for long jobs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    runner = Graph2MatDeepHBenchmarkRunner()
    status = runner.start(payload)
    write_json(args.status_json, {"status": status, "updated_at": time.time()})

    while True:
        status = runner.status()
        write_json(args.status_json, {"status": status, "updated_at": time.time()})
        runner.write_incremental_manifest(args.manifest_json)
        if not status.get("running"):
            break
        time.sleep(max(1.0, float(args.poll_seconds)))

    return int(status.get("returncode") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
