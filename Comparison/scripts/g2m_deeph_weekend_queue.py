#!/usr/bin/env python3
"""Queue paper-ready Graph2Mat-vs-DeepH weekend sweeps through the UI API.

This script is intentionally a small control-plane helper: it does not train
models directly. It prepares final-workflow runner payloads, waits for the
current UI run to finish, POSTs the next payload to /api/g2m-deeph/run, then
waits for that run to finish before launching the next one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / "Comparison" / "scripts" / "g2m_deeph_final_workflow.py"


@dataclass(frozen=True)
class QueueEntry:
    key: str
    protocol: Path
    workflow_root: Path
    dataset_id: str
    run_id_prefix: str


QUEUE_ENTRIES = {
    "iid600-fast": QueueEntry(
        key="iid600-fast",
        protocol=REPO_ROOT / "Comparison" / "config" / "g2m_deeph_iid600_weekend_fast_graph2mat_full_deeph_anchors_v1.json",
        workflow_root=REPO_ROOT / "Comparison" / "results" / "g2m_deeph_iid600_weekend_fast_graph2mat_full_deeph_anchors_v1",
        dataset_id="graphene_w90_phase1_iid600",
        run_id_prefix="g2m_deeph_iid600_weekend_fast",
    ),
    "iid1000-fast": QueueEntry(
        key="iid1000-fast",
        protocol=REPO_ROOT / "Comparison" / "config" / "g2m_deeph_iid1000_weekend_fast_transfer_saturation_v1.json",
        workflow_root=REPO_ROOT / "Comparison" / "results" / "g2m_deeph_iid1000_weekend_fast_transfer_saturation_v1",
        dataset_id="graphene_w90_phase1_iid1000",
        run_id_prefix="g2m_deeph_iid1000_weekend_fast",
    ),
    "iid600": QueueEntry(
        key="iid600",
        protocol=REPO_ROOT / "Comparison" / "config" / "g2m_deeph_iid600_phaseB_intermediate_spectral_refine_v1.json",
        workflow_root=REPO_ROOT / "Comparison" / "results" / "g2m_deeph_iid600_phaseB_intermediate_spectral_refine_v1",
        dataset_id="graphene_w90_phase1_iid600",
        run_id_prefix="g2m_deeph_iid600_phaseB_intermediate",
    ),
    "iid1000": QueueEntry(
        key="iid1000",
        protocol=REPO_ROOT / "Comparison" / "config" / "g2m_deeph_iid1000_phaseB_transfer_spectral_refine_v1.json",
        workflow_root=REPO_ROOT / "Comparison" / "results" / "g2m_deeph_iid1000_phaseB_transfer_spectral_refine_v1",
        dataset_id="graphene_w90_phase1_iid1000",
        run_id_prefix="g2m_deeph_iid1000_phaseB_transfer",
    ),
    "iid300-expanded": QueueEntry(
        key="iid300-expanded",
        protocol=REPO_ROOT / "Comparison" / "config" / "g2m_deeph_iid300_phaseB_expanded_spectral_refine_v1.json",
        workflow_root=REPO_ROOT / "Comparison" / "results" / "g2m_deeph_iid300_phaseB_expanded_spectral_refine_v1",
        dataset_id="graphene_w90_phase1_iid300",
        run_id_prefix="g2m_deeph_iid300_phaseB_expanded",
    ),
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return payload


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    allow_error: bool = False,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"UI API request failed: {url}: {exc}") from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    if parsed.get("error") and not allow_error:
        raise RuntimeError(str(parsed["error"]))
    return parsed


def run_workflow_command(args: list[str]) -> None:
    command = [sys.executable, str(WORKFLOW), *args]
    print("[QUEUE][CMD]", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def prepare_payload(entry: QueueEntry, *, queue_root: Path) -> Path:
    run_id = f"{entry.run_id_prefix}_queued_{now_stamp()}"
    run_workflow_command(
        [
            "--stage",
            "validate-protocol",
            "--protocol",
            str(entry.protocol),
            "--workflow-root",
            str(entry.workflow_root),
        ]
    )
    run_workflow_command(
        [
            "--stage",
            "generate-search-plan",
            "--protocol",
            str(entry.protocol),
            "--workflow-root",
            str(entry.workflow_root),
        ]
    )
    run_workflow_command(
        [
            "--stage",
            "run-search",
            "--dry-run",
            "--protocol",
            str(entry.protocol),
            "--workflow-root",
            str(entry.workflow_root),
            "--dataset-id",
            entry.dataset_id,
            "--run-id",
            run_id,
        ]
    )
    payload = read_json(entry.workflow_root / "search" / "run_search_payload.json")
    payload["dry_run"] = False
    payload["run_id"] = run_id
    payload_path = queue_root / "payloads" / f"{entry.key}_{run_id}.json"
    write_json(payload_path, payload)
    return payload_path


def wait_until_idle(ui_url: str, *, poll_seconds: float) -> dict[str, Any]:
    while True:
        status = request_json(f"{ui_url.rstrip('/')}/api/g2m-deeph/status", allow_error=True)
        sweep = status.get("training_sweep") if isinstance(status.get("training_sweep"), dict) else {}
        print(
            "[QUEUE][WAIT]",
            f"running={status.get('running')}",
            f"run_id={status.get('run_id')}",
            f"active={sweep.get('active_config_id')}",
            f"completed={sweep.get('completed')}",
            f"failed={sweep.get('failed')}",
            f"total={sweep.get('total')}",
            flush=True,
        )
        if not status.get("running"):
            return status
        time.sleep(poll_seconds)


def wait_for_run(ui_url: str, *, run_id: str, poll_seconds: float) -> dict[str, Any]:
    while True:
        status = request_json(f"{ui_url.rstrip('/')}/api/g2m-deeph/status", allow_error=True)
        sweep = status.get("training_sweep") if isinstance(status.get("training_sweep"), dict) else {}
        print(
            "[QUEUE][RUN]",
            f"run_id={status.get('run_id')}",
            f"running={status.get('running')}",
            f"active={sweep.get('active_config_id')}",
            f"completed={sweep.get('completed')}",
            f"failed={sweep.get('failed')}",
            f"total={sweep.get('total')}",
            flush=True,
        )
        if status.get("run_id") == run_id and not status.get("running"):
            return status
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui-url", default="http://127.0.0.1:8770")
    parser.add_argument("--queue", default="iid600-fast,iid1000-fast", help="Comma-separated queue keys.")
    parser.add_argument("--state-root", type=Path, default=REPO_ROOT / "Comparison" / "results" / "weekend_queue")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue_keys = [item.strip() for item in str(args.queue).split(",") if item.strip()]
    unknown = sorted(set(queue_keys) - set(QUEUE_ENTRIES))
    if unknown:
        raise RuntimeError("Unknown queue entries: " + ", ".join(unknown))
    queue_root = args.state_root / f"weekend_queue_{now_stamp()}"
    queue_root.mkdir(parents=True, exist_ok=True)
    print("[QUEUE] state_root", queue_root, flush=True)

    payloads: list[tuple[QueueEntry, Path]] = []
    for key in queue_keys:
        entry = QUEUE_ENTRIES[key]
        payload_path = prepare_payload(entry, queue_root=queue_root)
        payloads.append((entry, payload_path))
        print("[QUEUE][PREPARED]", key, payload_path, flush=True)

    if args.prepare_only:
        print("[QUEUE] prepare-only complete", flush=True)
        return

    results: list[dict[str, Any]] = []
    for entry, payload_path in payloads:
        wait_until_idle(args.ui_url, poll_seconds=args.poll_seconds)
        payload = read_json(payload_path)
        print("[QUEUE][START]", entry.key, payload.get("run_id"), flush=True)
        started = request_json(f"{args.ui_url.rstrip('/')}/api/g2m-deeph/run", method="POST", payload=payload)
        write_json(queue_root / "started" / f"{entry.key}.json", started)
        final_status = wait_for_run(args.ui_url, run_id=str(payload["run_id"]), poll_seconds=args.poll_seconds)
        write_json(queue_root / "completed" / f"{entry.key}.json", final_status)
        results.append({"entry": entry.key, "run_id": payload.get("run_id"), "returncode": final_status.get("returncode")})
        if final_status.get("returncode") not in (0, None) and not args.continue_on_failure:
            print("[QUEUE][STOP] run failed; not launching remaining entries", flush=True)
            break

    write_json(queue_root / "queue_summary.json", {"queue": queue_keys, "results": results})
    print("[QUEUE] complete", queue_root / "queue_summary.json", flush=True)


if __name__ == "__main__":
    main()
