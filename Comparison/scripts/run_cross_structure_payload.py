#!/usr/bin/env python3
"""Run a cross-structure Graph2Mat/DeepH payload."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ml_vs_siesta as mvs  # noqa: E402
from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner  # noqa: E402


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_train(payload: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    runner = Graph2MatDeepHBenchmarkRunner()
    result = mvs.run_cross_structure_payload(payload, launch_fn=runner.start)
    if args.status_json:
        _write_json(args.status_json, {"status": result.get("runner_result"), "updated_at": time.time()})
    while True:
        status = runner.status()
        result["runner_result"] = status
        if args.status_json:
            _write_json(args.status_json, {"status": status, "updated_at": time.time()})
        if args.manifest_json:
            runner.write_incremental_manifest(args.manifest_json)
        if not status.get("running"):
            break
        time.sleep(max(1.0, float(args.poll_seconds)))
    result["runner_results"] = runner.results()
    final_status = result.get("runner_result") if isinstance(result.get("runner_result"), dict) else {}
    return result, int(final_status.get("returncode") or 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{args.payload} must contain a JSON object.")
    action = str(payload.get("action") or "preview").strip().lower()
    if action == "train":
        result, returncode = _run_train(payload, args)
    else:
        result = mvs.run_cross_structure_payload(payload)
        returncode = 0
    text = json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n"
    if args.result_json:
        _write_json(args.result_json, result)
    print(text, end="")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
