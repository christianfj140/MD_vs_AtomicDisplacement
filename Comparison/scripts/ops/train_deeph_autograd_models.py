#!/usr/bin/env python3
"""Retrain selected DeepH checkpoints with graph.new_sp=True."""

from __future__ import annotations

import argparse
import concurrent.futures
import configparser
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OPS_DIR = Path(__file__).resolve().parent
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

from launch_ui_real_metrics_derivatives import _cross_cases, _mixing_cases, _path, _read  # noqa: E402


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _selected_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    cases = _mixing_cases(_read(_path(str(config["mixing_summary"]))))
    cases.extend(_cross_cases(_read(_path(str(config["cross_summary"])))) )
    selected_ids = {str(value) for value in config.get("include_case_ids") or []}
    if selected_ids:
        cases = [case for case in cases if str(case["id"]) in selected_ids]
    if not cases:
        state_id = str(config.get("campaign_state_id") or "").strip()
        plan = _path(str(config["output_root"])) / f"derivative_campaign_plan_{state_id}.json"
        if state_id and plan.is_file():
            cases = [
                case for case in _read(plan).get("cases") or []
                if not selected_ids or str(case["id"]) in selected_ids
            ]
    output_campaign = str(config.get("output_campaign") or "").strip()
    if output_campaign:
        for case in cases:
            case["campaign"] = output_campaign
    return cases


def _prepare_config(
    source: Path,
    destination: Path,
    graph_dir: Path,
    save_dir: Path,
    num_threads: int,
) -> None:
    parser = configparser.RawConfigParser()
    parser.optionxform = str
    parser.read(source, encoding="utf-8")
    for section in ("basic", "graph"):
        if not parser.has_section(section):
            raise RuntimeError(f"Missing [{section}] in {source}")
    parser.set("basic", "graph_dir", str(graph_dir))
    parser.set("basic", "save_dir", str(save_dir))
    parser.set("basic", "num_threads", str(num_threads))
    parser.set("graph", "new_sp", "True")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        parser.write(handle)


def _run(command: list[str], log: Path) -> None:
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = _read(_path(args.payload))
    output_root = _path(str(config["output_root"]))
    model_subdir = str(config.get("deeph_autograd_model_subdir") or "deeph_autograd_model")
    max_parallel = max(1, int(config.get("max_parallel_deeph_autograd_jobs") or 3))
    num_threads = max(1, int(config.get("deeph_autograd_num_threads") or 2))
    selected = _selected_cases(config)
    requested = set(args.case_id)
    if requested:
        selected = [case for case in selected if str(case["id"]) in requested]
    if not selected:
        raise RuntimeError("No DeepH autograd training cases selected.")
    print(json.dumps({"planned": len(selected), "cases": [case["id"] for case in selected]}, ensure_ascii=False), flush=True)
    if args.plan_only:
        return 0
    deeph_root = REPO_ROOT.parent / "DeepH-pack" / ".venv" / "bin"

    def train(case: dict[str, Any]) -> dict[str, Any]:
        case_root = output_root / str(case["campaign"]) / str(case["id"]) / model_subdir
        train_dir = case_root / "train"
        config_path = train_dir / "config.ini"
        log_path = case_root / "train.log"
        completion_path = case_root / "training_complete.json"
        source_model = Path(case["deeph_model_dir"])
        source_config = source_model.parent / "config" / "train.ini"
        if not source_config.is_file():
            source_config = source_model / "config.ini"
        if not source_config.is_file():
            raise RuntimeError(f"Missing source DeepH config: {source_config}")
        started = time.time()
        if completion_path.is_file() and (train_dir / "best_state_dict.pkl").is_file() and config_path.is_file():
            status = "skipped_existing"
        else:
            _prepare_config(
                source_config,
                config_path,
                case_root / "graph",
                train_dir,
                num_threads,
            )
            _run([str(deeph_root / "deeph-train"), "--config", str(config_path)], log_path)
            if not (train_dir / "best_state_dict.pkl").is_file():
                raise RuntimeError(f"DeepH autograd checkpoint missing after training: {train_dir}")
            status = "completed"
            _write(completion_path, {"status": status, "model_dir": str(train_dir), "completed_at_epoch_seconds": time.time()})
        return {"id": case["id"], "status": status, "model_dir": str(train_dir), "elapsed_seconds": time.time() - started}

    completed: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        results = executor.map(train, selected)
        for index, result in enumerate(results, start=1):
            completed.append(result)
            _write(output_root / "deeph_autograd_training_status_cross_w90_to_5x5_2delta.json", {"completed": completed})
            print(json.dumps({"index": index, "total": len(selected), **completed[-1]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
