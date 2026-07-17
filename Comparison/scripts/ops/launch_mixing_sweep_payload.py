#!/usr/bin/env python3
"""Launch one mixing payload through the same backend path as the UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_mixing_e2e_payload_once import _run_mixing_payload, read_json  # noqa: E402


def _log(line: str) -> None:
    print(line, end="" if line.endswith("\n") else "\n", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    payload_path = args.payload if args.payload.is_absolute() else REPO_ROOT / args.payload
    output_root = args.output_root
    if output_root is not None and not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    payload = read_json(payload_path)
    _log(f"[LAUNCH] payload={payload_path}\n")
    _log(
        "[LAUNCH] "
        f"split_policy={payload.get('split_policy')} "
        f"split_fractions={payload.get('split_fractions')} "
        f"temporal_gap={payload.get('temporal_gap')} "
        f"training_weighting_policy={payload.get('training_weighting_policy')}\n"
    )
    summary: dict[str, Any] = _run_mixing_payload(payload, output_root, log_fn=_log)
    _log("[LAUNCH] completed\n")
    _log(
        json.dumps(
            {
                key: summary.get(key)
                for key in (
                    "n_permutations",
                    "n_trained",
                    "n_partial",
                    "n_failed",
                    "parallel_run_root",
                    "training_weighting_policy",
                )
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
