#!/usr/bin/env python3
"""Low-overhead cost telemetry helpers for Graph2Mat-vs-DeepH runs."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


TELEMETRY_SCHEMA = "graph2mat_deeph_cost_telemetry_v1"
REQUIRED_TELEMETRY_FIELDS = (
    "wall_clock_seconds_total",
    "gpu_hours_total",
    "peak_gpu_memory_mb",
    "samples_per_second",
    "matrix_blocks_per_second",
    "best_validation_epoch",
)


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def compute_gpu_hours(gpu_active_seconds: float | int | None, gpu_count: int | None) -> float | None:
    seconds = finite_number(gpu_active_seconds)
    if seconds is None or seconds < 0 or not gpu_count or gpu_count <= 0:
        return None
    return seconds * int(gpu_count) / 3600.0


def compute_throughput(
    *,
    samples: int | None,
    matrix_blocks: int | None,
    elapsed_seconds: float | int | None,
) -> dict[str, float | None]:
    elapsed = finite_number(elapsed_seconds)
    if elapsed is None or elapsed <= 0:
        return {"samples_per_second": None, "matrix_blocks_per_second": None}
    samples_per_second = None if samples is None or samples < 0 else float(samples) / elapsed
    matrix_blocks_per_second = None if matrix_blocks is None or matrix_blocks < 0 else float(matrix_blocks) / elapsed
    return {
        "samples_per_second": samples_per_second,
        "matrix_blocks_per_second": matrix_blocks_per_second,
    }


def optimizer_update_accounting(
    *,
    train_samples: int | None,
    batch_size: int | None,
    max_epochs: int | None,
    gradient_accumulation: int | None = 1,
    drop_last: bool = False,
    low_update_threshold: int = 200,
) -> dict[str, Any]:
    """Return optimizer-step accounting without inspecting model internals."""

    warnings: list[str] = []
    if train_samples is None or train_samples <= 0:
        warnings.append("train_samples unavailable")
    if batch_size is None or batch_size <= 0:
        warnings.append("batch_size unavailable")
    if max_epochs is None or max_epochs <= 0:
        warnings.append("max_epochs unavailable")
    accumulation = int(gradient_accumulation or 1)
    if accumulation <= 0:
        warnings.append("gradient_accumulation invalid; treated as 1")
        accumulation = 1
    if warnings:
        return {
            "status": "unavailable",
            "train_samples": train_samples,
            "batch_size": batch_size,
            "gradient_accumulation": accumulation,
            "max_epochs": max_epochs,
            "drop_last": bool(drop_last),
            "steps_per_epoch": None,
            "optimizer_updates_per_epoch": None,
            "total_optimizer_updates": None,
            "possible_undertraining": True,
            "warnings": warnings,
        }
    if drop_last:
        steps_per_epoch = int(train_samples) // int(batch_size)
    else:
        steps_per_epoch = math.ceil(int(train_samples) / int(batch_size))
    updates_per_epoch = math.ceil(max(0, steps_per_epoch) / accumulation)
    total_updates = updates_per_epoch * int(max_epochs)
    undertrained = total_updates < int(low_update_threshold)
    if undertrained:
        warnings.append(
            f"total_optimizer_updates={total_updates} below paper-ready threshold {low_update_threshold}; "
            "Graph2Mat may be undertrained."
        )
    return {
        "status": "available",
        "train_samples": int(train_samples),
        "batch_size": int(batch_size),
        "gradient_accumulation": accumulation,
        "max_epochs": int(max_epochs),
        "drop_last": bool(drop_last),
        "steps_per_epoch": steps_per_epoch,
        "optimizer_updates_per_epoch": updates_per_epoch,
        "total_optimizer_updates": total_updates,
        "possible_undertraining": undertrained,
        "warnings": warnings,
    }


def best_validation_cost_from_events(
    events: list[dict[str, Any]],
    *,
    mode: str = "min",
    run_started_at: float | None = None,
) -> dict[str, Any]:
    clean = []
    for item in events:
        value = finite_number(item.get("value"))
        if value is None:
            continue
        clean.append({**item, "value": value})
    if not clean:
        return {
            "best_validation_value": None,
            "best_validation_epoch": None,
            "best_validation_step": None,
            "wall_clock_seconds_to_best_validation": None,
            "status": "unavailable",
            "warning": "no validation events found",
        }
    reverse = mode == "max"
    best = sorted(clean, key=lambda item: item["value"], reverse=reverse)[0]
    first_wall_time = finite_number(run_started_at)
    if first_wall_time is None:
        first_wall_time = min(
            (finite_number(item.get("wall_time")) for item in clean if finite_number(item.get("wall_time")) is not None),
            default=None,
        )
    best_wall_time = finite_number(best.get("wall_time"))
    seconds_to_best = (
        best_wall_time - first_wall_time
        if best_wall_time is not None and first_wall_time is not None and best_wall_time >= first_wall_time
        else None
    )
    return {
        "best_validation_value": best["value"],
        "best_validation_epoch": best.get("epoch"),
        "best_validation_step": best.get("step"),
        "wall_clock_seconds_to_best_validation": seconds_to_best,
        "status": "available",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def split_sample_count(frozen_split_manifest_path: Path | None, *, splits: set[str] | None = None) -> int | None:
    if frozen_split_manifest_path is None or not frozen_split_manifest_path.exists():
        return None
    payload = _read_json(frozen_split_manifest_path)
    splits = splits or {"train", "validation"}
    rows = payload.get("rows")
    if isinstance(rows, list):
        return sum(1 for row in rows if isinstance(row, dict) and str(row.get("split")) in splits)
    counts = payload.get("split_counts")
    if isinstance(counts, dict):
        total = 0
        seen = False
        for split in splits:
            value = finite_number(counts.get(split))
            if value is not None:
                total += int(value)
                seen = True
        return total if seen else None
    return None


def _parse_nvidia_smi_rows(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        memory = re.sub(r"[^0-9.]", "", parts[1])
        memory_mb = finite_number(memory)
        if memory_mb is None:
            continue
        rows.append(
            {
                "pid": int(parts[0]),
                "used_memory_mb": memory_mb,
                "gpu_uuid": parts[2] if len(parts) > 2 and parts[2] else "",
            }
        )
    return rows


def query_nvidia_compute_processes() -> tuple[list[dict[str, Any]], str | None]:
    if shutil.which("nvidia-smi") is None:
        return [], "nvidia-smi unavailable"
    commands = [
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
    ]
    last_error = None
    for command in commands:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            return _parse_nvidia_smi_rows(result.stdout), None
        last_error = (result.stderr or result.stdout or "").strip() or f"{command[0]} failed"
    return [], last_error


def descendant_pids(root_pid: int) -> set[int]:
    seen = {int(root_pid)}
    queue = [int(root_pid)]
    while queue:
        pid = queue.pop()
        children_file = Path("/proc") / str(pid) / "task" / str(pid) / "children"
        try:
            children = [int(item) for item in children_file.read_text(encoding="utf-8").split() if item.isdigit()]
        except OSError:
            children = []
        for child in children:
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return seen


@dataclass
class GpuTelemetryMonitor:
    """Poll per-process GPU memory for one subprocess tree."""

    poll_interval_seconds: float = 1.0
    query_processes: Callable[[], tuple[list[dict[str, Any]], str | None]] = query_nvidia_compute_processes
    pid_tree: Callable[[int], set[int]] = descendant_pids
    _root_pid: int | None = None
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _peak_gpu_memory_mb: float | None = None
    _gpu_active_seconds: float = 0.0
    _observed_gpu_ids: set[str] = field(default_factory=set)
    _observed_pids: set[int] = field(default_factory=set)
    _warnings: list[str] = field(default_factory=list)

    def start(self, root_pid: int) -> None:
        self._root_pid = int(root_pid)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="g2m-deeph-gpu-telemetry", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last_poll = time.time()
        while not self._stop.wait(self.poll_interval_seconds):
            now = time.time()
            self._poll(now - last_poll)
            last_poll = now
        self._poll(max(0.0, time.time() - last_poll))

    def _poll(self, elapsed_since_last_poll: float) -> None:
        root_pid = self._root_pid
        if root_pid is None:
            return
        tracked = self.pid_tree(root_pid)
        rows, warning = self.query_processes()
        if warning:
            with self._lock:
                if warning not in self._warnings:
                    self._warnings.append(warning)
            return
        matched = [row for row in rows if int(row.get("pid", -1)) in tracked]
        if not matched:
            return
        memory_sum = sum(float(row.get("used_memory_mb") or 0.0) for row in matched)
        gpu_ids = {str(row.get("gpu_uuid") or "gpu") for row in matched}
        pids = {int(row.get("pid")) for row in matched if str(row.get("pid", "")).isdigit()}
        with self._lock:
            self._peak_gpu_memory_mb = max(self._peak_gpu_memory_mb or 0.0, memory_sum)
            self._gpu_active_seconds += max(0.0, float(elapsed_since_last_poll))
            self._observed_gpu_ids.update(gpu_ids)
            self._observed_pids.update(pids)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.poll_interval_seconds * 2.0))
        with self._lock:
            gpu_count = len(self._observed_gpu_ids) if self._observed_gpu_ids else None
            return {
                "peak_gpu_memory_mb": self._peak_gpu_memory_mb,
                "gpu_active_seconds": self._gpu_active_seconds if self._gpu_active_seconds > 0 else None,
                "observed_gpu_count": gpu_count,
                "observed_gpu_process_count": len(self._observed_pids),
                "warnings": list(self._warnings),
            }


def hardware_metadata(*, performance: dict[str, Any] | None = None) -> dict[str, Any]:
    performance = performance or {}
    metadata: dict[str, Any] = {
        "compute_accelerator": performance.get("compute_accelerator"),
        "dtype_policy": performance.get("torch_mixed_precision") or performance.get("graph2mat_precision"),
        "torch_float32_matmul_precision": performance.get("torch_float32_matmul_precision"),
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": None,
        "gpu_names": [],
        "source": "unavailable",
    }
    try:
        import torch  # type: ignore

        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["cuda_version"] = getattr(torch.version, "cuda", None)
        if metadata["cuda_available"]:
            count = int(torch.cuda.device_count())
            metadata["gpu_count"] = count
            metadata["gpu_names"] = [torch.cuda.get_device_name(index) for index in range(count)]
            metadata["source"] = "torch"
            return metadata
        metadata["source"] = "torch"
    except Exception as exc:
        metadata["torch_error"] = str(exc)

    if shutil.which("nvidia-smi") is not None:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            metadata["gpu_names"] = names
            metadata["gpu_count"] = len(names) or None
            metadata["cuda_available"] = bool(names)
            metadata["source"] = "nvidia-smi"
    return metadata


def _tensorboard_validation_events(training_dir: Path, metric: str = "val_loss") -> list[dict[str, Any]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return []
    event_files = sorted(
        (training_dir / "lightning_logs").rglob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not event_files:
        return []
    events: list[dict[str, Any]] = []
    for path in event_files[-3:]:
        try:
            accumulator = event_accumulator.EventAccumulator(str(path), size_guidance={"scalars": 0})
            accumulator.Reload()
            tags = set(accumulator.Tags().get("scalars", []))
            if metric not in tags:
                continue
            epoch_by_step = {}
            if "epoch" in tags:
                epoch_by_step = {
                    int(item.step): int(float(item.value))
                    for item in accumulator.Scalars("epoch")
                }
            for item in accumulator.Scalars(metric):
                events.append(
                    {
                        "step": int(item.step),
                        "epoch": epoch_by_step.get(int(item.step), int(item.step)),
                        "value": float(item.value),
                        "wall_time": float(item.wall_time),
                    }
                )
        except Exception:
            continue
    return events


def extract_graph2mat_validation_cost(training_dir: Path, *, metric: str = "val_loss") -> dict[str, Any]:
    events = _tensorboard_validation_events(training_dir, metric=metric)
    result = best_validation_cost_from_events(events, mode="min")
    result["selection_metric"] = metric
    if events:
        result["epochs_trained"] = max(
            (int(item["epoch"]) for item in events if item.get("epoch") is not None),
            default=None,
        )
    else:
        result["epochs_trained"] = None
    return result


def extract_deeph_validation_cost(save_dir: Path) -> dict[str, Any]:
    warnings: list[str] = []
    events: list[dict[str, Any]] = []
    for path in sorted(save_dir.glob("*.csv")):
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    metric_key = next((key for key in row if key.lower() in {"val_loss", "validation_loss"}), None)
                    if metric_key is None:
                        continue
                    events.append(
                        {
                            "epoch": int(float(row.get("epoch") or row.get("epochs") or len(events))),
                            "step": int(float(row.get("step") or row.get("epoch") or len(events))),
                            "value": float(row[metric_key]),
                            "wall_time": finite_number(row.get("wall_time")),
                        }
                    )
        except (OSError, ValueError):
            continue
    if not events:
        warnings.append("DeepH validation event log not found in save_dir; best validation cost unavailable.")
    result = best_validation_cost_from_events(events, mode="min")
    result["selection_metric"] = "validation_loss"
    result["epochs_trained"] = max((int(item["epoch"]) for item in events), default=None) if events else None
    if warnings:
        result["warnings"] = warnings
    return result


def _command_gpu_hours(command_run: dict[str, Any]) -> float | None:
    telemetry = command_run.get("telemetry") if isinstance(command_run.get("telemetry"), dict) else {}
    return finite_number(telemetry.get("gpu_hours"))


def _command_peak_memory(command_run: dict[str, Any]) -> float | None:
    telemetry = command_run.get("telemetry") if isinstance(command_run.get("telemetry"), dict) else {}
    return finite_number(telemetry.get("peak_gpu_memory_mb"))


def _phase_seconds(command_run: dict[str, Any] | None) -> float | None:
    return finite_number((command_run or {}).get("elapsed_seconds"))


def summarize_run_telemetry(
    *,
    model: str,
    run_root: Path,
    training_dir: Path | None = None,
    deeph_save_dir: Path | None = None,
    frozen_split_manifest_path: Path | None = None,
    train_run: dict[str, Any] | None = None,
    predict_run: dict[str, Any] | None = None,
    preprocess_run: dict[str, Any] | None = None,
    inference_runs: list[dict[str, Any]] | None = None,
    metrics_run: dict[str, Any] | None = None,
    performance: dict[str, Any] | None = None,
    matrix_block_count: int | None = None,
    optimizer_accounting: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inference_runs = inference_runs or []
    phase_seconds = {
        "preprocess": _phase_seconds(preprocess_run),
        "train": _phase_seconds(train_run),
        "predict": _phase_seconds(predict_run),
        "inference": sum(
            seconds for seconds in (_phase_seconds(run) for run in inference_runs) if seconds is not None
        )
        if inference_runs
        else None,
        "metrics": _phase_seconds(metrics_run),
    }
    wall_clock_total = sum(seconds for seconds in phase_seconds.values() if seconds is not None)
    wall_clock_total = wall_clock_total if wall_clock_total > 0 else None
    gpu_hours_values = [
        value
        for value in (
            _command_gpu_hours(run)
            for run in [preprocess_run, train_run, predict_run, metrics_run, *inference_runs]
            if isinstance(run, dict)
        )
        if value is not None
    ]
    peak_values = [
        value
        for value in (
            _command_peak_memory(run)
            for run in [preprocess_run, train_run, predict_run, metrics_run, *inference_runs]
            if isinstance(run, dict)
        )
        if value is not None
    ]
    gpu_hours_total = sum(gpu_hours_values) if gpu_hours_values else None
    peak_gpu_memory_mb = max(peak_values) if peak_values else None
    training_samples = split_sample_count(frozen_split_manifest_path, splits={"train", "validation"})
    throughput = compute_throughput(
        samples=training_samples,
        matrix_blocks=matrix_block_count,
        elapsed_seconds=phase_seconds["train"],
    )
    validation_cost = (
        extract_graph2mat_validation_cost(training_dir)
        if model == "graph2mat" and training_dir is not None
        else extract_deeph_validation_cost(deeph_save_dir)
        if model == "deeph" and deeph_save_dir is not None
        else {"status": "unavailable", "warning": "validation cost source unavailable"}
    )
    gpu_count = None
    if train_run and isinstance(train_run.get("telemetry"), dict):
        gpu_count = train_run["telemetry"].get("observed_gpu_count")
    seconds_to_best = finite_number(validation_cost.get("wall_clock_seconds_to_best_validation"))
    gpu_hours_to_best = compute_gpu_hours(seconds_to_best, int(gpu_count)) if gpu_count else None
    telemetry = {
        "schema": TELEMETRY_SCHEMA,
        "model": model,
        "run_root": str(run_root),
        "wall_clock_seconds_total": wall_clock_total,
        "wall_clock_seconds_by_phase": phase_seconds,
        "gpu_hours_total": gpu_hours_total,
        "gpu_hours_to_best_validation": gpu_hours_to_best,
        "wall_clock_seconds_to_best_validation": seconds_to_best,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "samples_per_second": throughput["samples_per_second"],
        "matrix_blocks_per_second": throughput["matrix_blocks_per_second"],
        "training_sample_count": training_samples,
        "matrix_block_count": matrix_block_count,
        "epochs_trained": validation_cost.get("epochs_trained"),
        "best_validation_epoch": validation_cost.get("best_validation_epoch"),
        "best_validation_step": validation_cost.get("best_validation_step"),
        "best_validation_value": validation_cost.get("best_validation_value"),
        "validation_metric": validation_cost.get("selection_metric"),
        "optimizer_update_accounting": optimizer_accounting or {},
        "steps_per_epoch": (optimizer_accounting or {}).get("steps_per_epoch"),
        "total_optimizer_updates": (optimizer_accounting or {}).get("total_optimizer_updates"),
        "hardware": hardware_metadata(performance=performance),
        "telemetry_warnings": [],
    }
    warnings = telemetry["telemetry_warnings"]
    for key in REQUIRED_TELEMETRY_FIELDS:
        if telemetry.get(key) is None:
            warnings.append(f"{key} unavailable")
    for command_name, command_run in (
        ("preprocess", preprocess_run),
        ("train", train_run),
        ("predict", predict_run),
        ("metrics", metrics_run),
    ):
        command_telemetry = command_run.get("telemetry") if isinstance(command_run, dict) else {}
        for warning in (command_telemetry or {}).get("warnings") or []:
            message = f"{command_name}: {warning}"
            if message not in warnings:
                warnings.append(message)
    for warning in validation_cost.get("warnings") or []:
        if warning not in warnings:
            warnings.append(warning)
    for warning in (optimizer_accounting or {}).get("warnings") or []:
        message = f"optimizer_updates: {warning}"
        if message not in warnings:
            warnings.append(message)
    if validation_cost.get("warning") and validation_cost["warning"] not in warnings:
        warnings.append(str(validation_cost["warning"]))
    telemetry["telemetry_status"] = "complete" if not warnings else ("partial" if wall_clock_total is not None else "unavailable")
    return telemetry


def write_telemetry(path: Path, telemetry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(telemetry, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
