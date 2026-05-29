#!/usr/bin/env python3
"""Read-only sidecar API for live Graph2Mat-vs-DeepH sweep metrics."""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from g2m_deeph_live_metrics import (
    _training_record_epoch_label,
    _training_record_epochs,
    dataset_size_from_root,
    dedupe_metric_rows,
    live_metric_scaling_rows,
    load_training_records,
)
from g2m_deeph_metrics import COMMON_METRIC_GROUPS, build_common_plot_payload


def _finite_elapsed(value: object) -> float | None:
    try:
        elapsed = float(value)
    except (TypeError, ValueError):
        return None
    return elapsed if elapsed > 0 else None


def _append_timing_row(
    rows: list[dict[str, object]],
    *,
    record: dict[str, object],
    phase: str,
    label: str,
    elapsed_seconds: object,
    source: str,
) -> None:
    elapsed = _finite_elapsed(elapsed_seconds)
    if elapsed is None:
        return
    dataset_root = str(record.get("dataset_root") or "")
    dataset_size = dataset_size_from_root(dataset_root)
    if dataset_size is None or dataset_size <= 0:
        return
    rows.append(
        {
            "run_id": str(record.get("parent_run_id") or ""),
            "dataset_id": str(record.get("dataset_id") or ""),
            "dataset_root": dataset_root,
            "dataset_size": int(dataset_size),
            "phase": phase,
            "label": label,
            "model": str(record.get("model") or ""),
            "config_id": str(record.get("config_id") or ""),
            "epochs": _training_record_epochs(record),
            "epoch_label": _training_record_epoch_label(record),
            "elapsed_seconds": elapsed,
            "seconds_per_snapshot": elapsed / float(dataset_size),
            "source": source,
            "status": str(record.get("status") or ""),
        }
    )


def _process_elapsed_seconds(pid: int) -> float | None:
    try:
        uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        start_ticks = float(stat.split()[21])
        clock_ticks = float(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    except (OSError, IndexError, KeyError, TypeError, ValueError):
        return None
    elapsed = uptime - (start_ticks / clock_ticks)
    return elapsed if elapsed > 0 else None


def _active_deeph_phase(command_line: str) -> tuple[str, str] | None:
    if "deeph-preprocess" in command_line:
        return "deeph_preprocess", "DeepH preprocess (active)"
    if "deeph-train" in command_line:
        return "deeph_train", "DeepH train (active)"
    if "deeph-inference" in command_line:
        return "deeph_predict", "DeepH predict (active)"
    return None


def _planned_records_by_id(run_root: Path) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads((run_root / "sweep" / "search_plan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(row.get("config_id") or ""): row
        for row in payload.get("planned_runs") or []
        if isinstance(row, dict) and row.get("config_id")
    }


def _active_timing_rows(run_roots: list[Path]) -> list[dict[str, object]]:
    if not Path("/proc").exists():
        return []
    planned_by_root = {root: _planned_records_by_id(root) for root in run_roots}
    rows: list[dict[str, object]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command_line = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except OSError:
            continue
        phase = _active_deeph_phase(command_line)
        if phase is None:
            continue
        for root in run_roots:
            root_text = str(root)
            if root_text not in command_line:
                continue
            config_path = ""
            if "--config" in command_line:
                config_path = command_line.split("--config", 1)[-1].strip().split()[0]
            parts = Path(config_path).parts
            config_id = ""
            dataset_id = ""
            if "sweep" in parts:
                index = parts.index("sweep")
                if len(parts) > index + 3:
                    dataset_id = parts[index + 2]
                    config_id = parts[index + 3]
            planned = planned_by_root.get(root, {}).get(config_id, {})
            record = dict(planned) if isinstance(planned, dict) else {}
            record.setdefault("model", "deeph")
            record.setdefault("dataset_id", dataset_id)
            record.setdefault("config_id", config_id)
            record.setdefault("dataset_root", "")
            record.setdefault("status", "running")
            _append_timing_row(
                rows,
                record={**record, "parent_run_id": root.name},
                phase=phase[0],
                label=phase[1],
                elapsed_seconds=_process_elapsed_seconds(int(proc.name)),
                source="active_deeph_process_sidecar",
            )
    return rows


def live_timing_scaling_rows(run_roots: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in run_roots:
        for record in load_training_records(root):
            record = {**record, "parent_run_id": root.name}
            model = str(record.get("model") or "")
            if model == "graph2mat":
                phase_specs = [
                    ("graph2mat_train", "Graph2Mat train", (record.get("train_run") or {}).get("elapsed_seconds")),
                    ("graph2mat_predict", "Graph2Mat predict", (record.get("predict_run") or {}).get("elapsed_seconds")),
                ]
            elif model == "deeph":
                phase_specs = [
                    ("deeph_preprocess", "DeepH preprocess", (record.get("preprocess_run") or {}).get("elapsed_seconds")),
                    ("deeph_train", "DeepH train", (record.get("train_run") or {}).get("elapsed_seconds")),
                    (
                        "deeph_predict",
                        "DeepH predict",
                        sum(
                            float(run.get("elapsed_seconds") or 0)
                            for run in record.get("inference_runs") or []
                            if isinstance(run, dict)
                        ),
                    ),
                ]
            else:
                phase_specs = []
            phase_specs.append(("metrics", "Metrics", (record.get("metrics_run") or {}).get("elapsed_seconds")))
            for phase, label, elapsed in phase_specs:
                _append_timing_row(
                    rows,
                    record=record,
                    phase=phase,
                    label=label,
                    elapsed_seconds=elapsed,
                    source="live_training_sweep_timing_sidecar",
                )
    rows.extend(_active_timing_rows(run_roots))
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


class LiveMetricsHandler(BaseHTTPRequestHandler):
    run_root: Path
    run_roots: list[Path]

    def _run_roots(self) -> list[Path]:
        roots = getattr(self, "run_roots", None) or []
        if roots:
            return list(roots)
        return [self.run_root]

    def _headers(self, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/api/g2m-deeph/live-plots", "/health"}:
            self._headers(HTTPStatus.NOT_FOUND)
            self.wfile.write(json.dumps({"error": "not_found"}).encode("utf-8"))
            return
        if path == "/health":
            self._headers()
            self.wfile.write(
                json.dumps(
                    {"ok": True, "run_root": str(self.run_root), "run_roots": [str(root) for root in self._run_roots()]}
                ).encode("utf-8")
            )
            return
        try:
            rows = dedupe_metric_rows(
                [
                    row
                    for root in self._run_roots()
                    for row in live_metric_scaling_rows(root)
                ]
            )
            timing_rows = live_timing_scaling_rows(self._run_roots())
            payload = build_common_plot_payload(
                None,
                metric_scaling_rows=rows,
                timing_scaling_rows=timing_rows,
                status_payload={"run_root": str(self.run_root), "run_roots": [str(root) for root in self._run_roots()], "running": True},
            )
            payload.update(
                {
                    "schema": "graph2mat_deeph_live_plot_payload_v1",
                    "run_id": self.run_root.name,
                    "run_root": str(self.run_root),
                    "run_roots": [str(root) for root in self._run_roots()],
                    "metric_scaling_rows": rows,
                    "timing_scaling_rows": timing_rows,
                    "live_metric_rows": len(rows),
                    "live_timing_rows": len(timing_rows),
                    "metric_groups": COMMON_METRIC_GROUPS,
                    "source": "live_training_sweep_metrics_sidecar",
                }
            )
            self._headers()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        except Exception as exc:  # Sidecar errors should be visible, not fatal to the main run.
            self._headers(HTTPStatus.INTERNAL_SERVER_ERROR)
            self.wfile.write(
                json.dumps(
                    {"error": str(exc), "run_root": str(self.run_root), "run_roots": [str(root) for root in self._run_roots()]}
                ).encode("utf-8")
            )

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_roots = [path.resolve(strict=False) for path in args.run_root]
    handler = type(
        "ConfiguredLiveMetricsHandler",
        (LiveMetricsHandler,),
        {"run_root": run_roots[-1], "run_roots": run_roots},
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        json.dumps(
            {
                "status": "serving",
                "url": f"http://{args.host}:{args.port}/api/g2m-deeph/live-plots",
                "run_root": str(run_roots[-1]),
                "run_roots": [str(root) for root in run_roots],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
