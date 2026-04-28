#!/usr/bin/env python3
"""Local web UI and API for running MD and AtomDisplacement together."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pty
import json
import select
import shutil
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).resolve().parents[1] / "ui"
COMPARISON_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = COMPARISON_ROOT / "results"
WORKSPACES_ROOT = COMPARISON_ROOT / "workspaces"
LOG_HEARTBEAT_SECONDS = 30.0


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "sin estimacion"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


@dataclass(frozen=True)
class PipelineSpec:
    key: str
    label: str
    root: Path
    main_script: Path

    @property
    def config_path(self) -> Path:
        return self.root / "pipeline_config.yaml"


PIPELINES = {
    "md": PipelineSpec(
        key="md",
        label="MD",
        root=REPO_ROOT / "MD",
        main_script=REPO_ROOT / "MD" / "scripts" / "main_md.py",
    ),
    "atom_displacement": PipelineSpec(
        key="atom_displacement",
        label="AtomDisplacement",
        root=REPO_ROOT / "AtomDisplacement",
        main_script=REPO_ROOT / "AtomDisplacement" / "scripts" / "main_atdisp.py",
    ),
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise RuntimeError(f"La configuracion debe ser un diccionario YAML: {path}")
    return config


def resolve_pipeline_path(spec: PipelineSpec, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return spec.root / path


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def stream_process_output(
    process: subprocess.Popen[str],
    append: Any,
    *,
    label: str,
    master_fd: int | None = None,
    eta_provider: Any | None = None,
) -> int:
    if master_fd is None:
        assert process.stdout is not None
        fd = process.stdout.fileno()
    else:
        fd = master_fd

    pending = ""
    started_at = time.time()
    last_output = started_at
    last_heartbeat = last_output
    while True:
        ready, _, _ = select.select([fd], [], [], 1.0)
        if ready:
            try:
                chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
            except OSError:
                chunk = ""
            if chunk:
                last_output = time.time()
                pending += chunk.replace("\r", "\n")
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    append(line + "\n")
            elif process.poll() is not None:
                break

        now = time.time()
        if process.poll() is not None:
            break
        if now - last_output >= LOG_HEARTBEAT_SECONDS and now - last_heartbeat >= LOG_HEARTBEAT_SECONDS:
            eta_seconds = eta_provider() if eta_provider is not None else None
            append(
                "[UI] "
                f"{label} sigue ejecutandose | PID {process.pid} | "
                f"elapsed {format_duration(now - started_at)} | "
                f"sin nueva salida {format_duration(now - last_output)} | "
                f"ETA {format_duration(eta_seconds)}\n"
            )
            last_heartbeat = now

    if pending:
        append(pending)
    return process.wait()


class PipelineRunner:
    def __init__(self, spec: PipelineSpec) -> None:
        self.spec = spec
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._logs: list[str] = []
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._returncode: int | None = None
        self._command: list[str] | None = None

    def start(self) -> dict[str, Any]:
        config = load_config(self.spec.config_path)
        venv_activate = resolve_pipeline_path(self.spec, config["paths"]["venv_activate"])
        if not venv_activate.exists():
            raise RuntimeError(
                f"{self.spec.label}: no se encontro el entorno virtual: {venv_activate}"
            )

        shell = str(config.get("commands", {}).get("shell", "bash"))
        python = str(config.get("commands", {}).get("python", "python"))
        shell_command = (
            f"source {shlex.quote(str(venv_activate))} "
            f"&& {shlex.quote(python)} {shlex.quote(str(self.spec.main_script))}"
        )

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"{self.spec.label}: el pipeline ya se esta ejecutando.")
            self._logs = [
                f"[UI] Ejecutando {self.spec.label}: {self.spec.main_script}\n",
                f"[UI] Root: {self.spec.root}\n",
                f"[UI] Config: {self.spec.config_path}\n",
                "[UI] ETA: sin estimacion inicial para ejecuciones individuales.\n",
            ]
            self._started_at = time.time()
            self._finished_at = None
            self._returncode = None
            self._command = [shell, "-lc", shell_command]
            master_fd, slave_fd = pty.openpty()
            self._process = subprocess.Popen(
                self._command,
                cwd=self.spec.root,
                stdin=subprocess.DEVNULL,
                stdout=slave_fd,
                stderr=slave_fd,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            os.close(slave_fd)
            process = self._process
            self._logs.append(f"[UI] PID: {process.pid}\n")
            self._logs.append(f"[RUN] {' '.join(self._command)}\n")

        threading.Thread(target=self._collect_output, args=(process, master_fd), daemon=True).start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return self.status()
            process.terminate()
            self._logs.append("\n[UI] Solicitud de parada enviada.\n")
        return self.status()

    def _collect_output(self, process: subprocess.Popen[str], master_fd: int) -> None:
        try:
            returncode = stream_process_output(
                process,
                lambda line: self._append_log(line),
                label=self.spec.label,
                master_fd=master_fd,
            )
        finally:
            os.close(master_fd)
        with self._lock:
            self._returncode = returncode
            self._finished_at = time.time()
            elapsed = self._finished_at - (self._started_at or self._finished_at)
            if self._process is process:
                self._process = None
            self._logs.append(
                f"\n[UI] {self.spec.label} finalizado con codigo {returncode} "
                f"en {format_duration(elapsed)}.\n"
            )

    def _append_log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return {
                "key": self.spec.key,
                "label": self.spec.label,
                "running": running,
                "returncode": None if running else self._returncode,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "command": self._command,
                "elapsed_seconds": None
                if self._started_at is None
                else (time.time() if running else self._finished_at or time.time()) - self._started_at,
                "eta_seconds": None,
                "log_size": len(self._logs),
            }

    def logs(self, since: int = 0) -> dict[str, Any]:
        with self._lock:
            since = max(0, since)
            return {
                "offset": len(self._logs),
                "lines": self._logs[since:],
                "status": self.status(),
            }


RUNNERS = {key: PipelineRunner(spec) for key, spec in PIPELINES.items()}


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(length).decode("utf-8")
    if not body:
        return {}
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError("El cuerpo JSON debe ser un objeto.")
    return parsed


def parse_sizes(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, list):
        raw_items = value
    else:
        raise RuntimeError("Los tamaños deben enviarse como lista o texto separado por comas.")

    sizes = []
    for item in raw_items:
        if item == "":
            continue
        size = int(item)
        if size <= 0:
            raise RuntimeError("Los tamaños de dataset deben ser mayores que cero.")
        sizes.append(size)
    if not sizes:
        raise RuntimeError("Define al menos un tamaño de dataset.")
    return sizes


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False, allow_unicode=False),
        encoding="utf-8",
    )


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_matching_files(source_root: Path, pattern: str, destination_root: Path) -> int:
    count = 0
    for src in sorted(source_root.glob(pattern)):
        if not src.is_file():
            continue
        dst = destination_root / src.relative_to(source_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
    return count


def split_counts(size: int) -> dict[str, int]:
    train = int(size * 0.8)
    validation = int(size * 0.1)
    test = size - train - validation
    if size >= 2 and test == 0:
        test = 1
        train = max(1, train - 1)
    return {"train": train, "validation": validation, "test": test}


def select_spread(items: list[Path], count: int) -> list[Path]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)

    used: set[int] = set()
    selected: list[int] = []
    for index in range(count):
        target = min(len(items) - 1, int((index + 0.5) * len(items) / count))
        if target in used:
            target = min(
                (candidate for candidate in range(len(items)) if candidate not in used),
                key=lambda candidate: abs(candidate - target),
            )
        used.add(target)
        selected.append(target)
    return [items[index] for index in sorted(selected)]


def split_spread(items: list[Path], counts: dict[str, int]) -> dict[str, list[Path]]:
    selected = list(items)
    test = select_spread(selected, counts["test"])
    remaining = [item for item in selected if item not in set(test)]
    validation = select_spread(remaining, counts["validation"])
    train = [item for item in remaining if item not in set(validation)]
    if len(train) > counts["train"]:
        train = select_spread(train, counts["train"])
    return {"train": train, "validation": validation, "test": test}


def sample_names(samples: list[Path]) -> str:
    return ", ".join(path.name for path in samples) if samples else "-"


def sample_sort_key(path: Path) -> tuple[int, str]:
    if path.name == "dataset":
        return (-1, path.name)
    if path.name.isdigit():
        return (int(path.name), path.name)
    suffix = path.name.rsplit("_", 1)[-1]
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (10**9, path.name)


def natural_matrix_sort_key(path: Path) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in path.stem.replace("-", ".").split("."):
        if chunk.isdigit():
            parts.append(int(chunk))
    return tuple(parts) if parts else (10**9,)


def read_system_label(run_fdf: Path) -> str:
    for line in run_fdf.read_text(encoding="utf-8", errors="ignore").splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        parts = clean.split()
        if len(parts) >= 2 and parts[0].lower() == "systemlabel":
            return parts[1]
    return "siesta"


def is_completed_atom_sample(sample_dir: Path) -> bool:
    if (sample_dir / "RUN.fdf").exists() and any(sample_dir.glob("*.TSHS")):
        return True
    run_out = sample_dir / "RUN.out"
    if not run_out.exists():
        return False
    text = run_out.read_text(encoding="utf-8", errors="ignore")
    has_reference = any(
        path.suffix in {".HSX", ".TSHS"} and path.name != "ML_prediction.HSX"
        for path in sample_dir.iterdir()
        if path.is_file()
    )
    return "Job completed" in text and "SCF cycle converged" in text and has_reference


def atom_source_samples_dir(spec: PipelineSpec, config: dict[str, Any]) -> Path:
    dataset_dir = resolve_pipeline_path(spec, config["paths"]["dataset_dir"])
    atdis_steps_dir = dataset_dir / "AtDis_steps"
    if atdis_steps_dir.exists():
        return atdis_steps_dir
    configured_samples = resolve_pipeline_path(spec, config["paths"]["samples_dir"])
    if configured_samples.exists():
        return configured_samples
    raise RuntimeError(
        "AtomDisplacement: no se encontro un dataset valido. "
        f"Busque {atdis_steps_dir} y {configured_samples}."
    )


def completed_atom_samples(source_samples_dir: Path) -> list[Path]:
    if source_samples_dir.name == "AtDis_steps":
        base_sample = source_samples_dir.parent
        base_samples = [base_sample] if is_completed_atom_sample(base_sample) else []
        return sorted(
            base_samples
            + [
                path
                for path in source_samples_dir.iterdir()
                if path.is_dir() and path.name.isdigit() and is_completed_atom_sample(path)
            ],
            key=sample_sort_key,
        )
    return sorted(
        (
            path
            for path in source_samples_dir.glob("sample_*")
            if path.is_dir() and is_completed_atom_sample(path)
        ),
        key=sample_sort_key,
    )


def generated_atom_samples(source_samples_dir: Path) -> list[Path]:
    if source_samples_dir.name == "AtDis_steps":
        base_sample = source_samples_dir.parent
        base_samples = [base_sample] if (base_sample / "RUN.fdf").exists() else []
        return sorted(
            base_samples
            + [
                path
                for path in source_samples_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ],
            key=sample_sort_key,
        )
    return sorted(
        (path for path in source_samples_dir.glob("sample_*") if path.is_dir()),
        key=sample_sort_key,
    )


def copy_sample_dirs(sample_dirs: list[Path], destination_root: Path) -> int:
    destination_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for sample_dir in sample_dirs:
        destination = destination_root / sample_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        for src in sorted(path for path in sample_dir.iterdir() if path.is_file()):
            shutil.copy2(src, destination / src.name)
        stale_prediction = destination / "ML_prediction.HSX"
        if stale_prediction.exists():
            stale_prediction.unlink()
        normalize_siesta_matrix_name(destination)
        count += 1
    return count


def normalize_siesta_matrix_name(sample_dir: Path) -> None:
    run_fdf = sample_dir / "RUN.fdf"
    if not run_fdf.exists():
        return
    system_label = read_system_label(run_fdf)
    canonical_hsx = sample_dir / f"{system_label}.HSX"
    canonical_tshs = sample_dir / f"{system_label}.TSHS"
    if canonical_hsx.exists() or canonical_tshs.exists():
        return
    candidates = sorted(sample_dir.glob(f"{system_label}*.TSHS"), key=natural_matrix_sort_key)
    if not candidates:
        return
    shutil.copy2(candidates[-1], canonical_tshs)


def copy_basis_files(source_samples_dir: Path, destination_dir: Path) -> int:
    source_basis_dir = source_samples_dir / "basis"
    if not source_basis_dir.exists():
        source_basis_dir = source_samples_dir.parent / "basis"
    if not source_basis_dir.exists():
        return 0
    destination_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(source_basis_dir.glob("*.ion.xml")):
        shutil.copy2(src, destination_dir / src.name)
        count += 1
    return count


def copy_pseudopotentials(source_dir: Path, destination_dir: Path) -> int:
    psf_files = sorted(source_dir.glob("*.psf"))
    if not psf_files:
        raise RuntimeError(f"No se encontraron pseudopotenciales .psf en {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for psf_file in psf_files:
        shutil.copy2(psf_file, destination_dir / psf_file.name)
    return len(psf_files)


def copy_relaxed_basis(source_dir: Path, destination_dir: Path) -> dict[str, int]:
    basis_files = sorted(source_dir.glob("*.ion.xml"))
    if not basis_files:
        raise RuntimeError(f"No se encontraron basis files .ion.xml en {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for basis_file in basis_files:
        shutil.copy2(basis_file, destination_dir / basis_file.name)
    xv_count = 0
    for xv_file in sorted(source_dir.glob("*.XV")):
        shutil.copy2(xv_file, destination_dir / xv_file.name)
        xv_count += 1
    return {"basis_files": len(basis_files), "xv_files": xv_count}


def read_metrics_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        return {"exists": True, "error": str(exc)}

    numeric_columns: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            try:
                numeric_columns.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                continue
    means = {
        key: sum(values) / len(values)
        for key, values in numeric_columns.items()
        if values
    }
    return {"exists": True, "rows": len(rows), "means": means}


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value in (None, ""):
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def numeric_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    columns: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key == "sample" or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if math.isfinite(number):
                columns.setdefault(key, []).append(number)
    return {
        key: sum(values) / len(values)
        for key, values in columns.items()
        if values
    }


def plot_data_summary() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    groups = {
        "md": RESULTS_ROOT / "results_md",
        "atom_displacement": RESULTS_ROOT / "results_atomdisp",
    }
    for key, root in groups.items():
        if not root.exists():
            continue
        for manifest_path in sorted(root.glob("dataset_*/run_*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            result_dir = Path(manifest.get("result_dir", manifest_path.parent))
            if not result_dir.is_absolute():
                result_dir = manifest_path.parent
            sparse_rows = read_csv_rows(result_dir / "metrics" / "sparse_metrics.csv")
            spectral_rows = read_csv_rows(result_dir / "metrics" / "spectral_metrics.csv")
            dos_rows = read_csv_rows(result_dir / "metrics" / "dos_metrics.csv")
            if not sparse_rows and not spectral_rows and not dos_rows:
                continue
            runs.append(
                {
                    "pipeline": key,
                    "label": PIPELINES[key].label,
                    "dataset_size": int(manifest.get("dataset_size", 0)),
                    "run_id": str(manifest.get("run_id", manifest_path.parent.name.removeprefix("run_"))),
                    "result_dir": str(result_dir),
                    "means": {
                        "sparse": numeric_means(sparse_rows),
                        "spectral": numeric_means(spectral_rows),
                        "dos": numeric_means(dos_rows),
                    },
                    "samples": {
                        "sparse": sparse_rows,
                        "spectral": spectral_rows,
                        "dos": dos_rows,
                    },
                }
            )
    runs.sort(key=lambda item: (item["pipeline"], item["dataset_size"], item["run_id"]))
    return {"runs": runs}


class ExperimentRunner:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._logs: list[str] = []
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._returncode: int | None = None
        self._current: dict[str, Any] | None = None
        self._results: list[dict[str, Any]] = []
        self._stop_requested = False
        self._run_id: str | None = None
        self._rate_seconds_per_structure: dict[str, float] = {}

    def start(self, md_sizes: list[int], atom_sizes: list[int]) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Ya hay una comparacion experimental en ejecucion.")
            self._logs = []
            self._started_at = time.time()
            self._finished_at = None
            self._returncode = None
            self._current = None
            self._results = []
            self._stop_requested = False
            self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._rate_seconds_per_structure = {}
            self._thread = threading.Thread(
                target=self._run,
                args=(md_sizes, atom_sizes, self._run_id),
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._stop_requested = True
            process = self._process
            self._logs.append("\n[UI] Solicitud de parada enviada al experimento.\n")
        if process is not None and process.poll() is None:
            process.terminate()
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            current = dict(self._current) if self._current is not None else None
            if current and running:
                started_at = current.get("started_at")
                if started_at is not None:
                    elapsed = time.time() - float(started_at)
                    current["elapsed_seconds"] = elapsed
                    current["eta_seconds"] = self._estimated_seconds(
                        str(current["pipeline"]),
                        int(current["size"]),
                        elapsed,
                    )
            return {
                "running": running,
                "returncode": None if running else self._returncode,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "current": current,
                "results": self._results,
                "log_size": len(self._logs),
                "run_id": self._run_id,
                "results_root": str(RESULTS_ROOT),
            }

    def logs(self, since: int = 0) -> dict[str, Any]:
        with self._lock:
            since = max(0, since)
            return {
                "offset": len(self._logs),
                "lines": self._logs[since:],
                "status": self.status(),
            }

    def _append(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)

    def _set_current(
        self,
        pipeline: str,
        size: int,
        *,
        started_at: float | None = None,
        eta_seconds: float | None = None,
    ) -> None:
        with self._lock:
            self._current = {
                "pipeline": pipeline,
                "size": size,
                "started_at": started_at,
                "eta_seconds": eta_seconds,
            }

    def _run(self, md_sizes: list[int], atom_sizes: list[int], run_id: str) -> None:
        original_configs = {
            key: spec.config_path.read_text(encoding="utf-8")
            for key, spec in PIPELINES.items()
        }
        returncode = 0
        try:
            self._append(f"[UI] Comparacion {run_id} iniciada.\n")
            self._append(f"[UI] MD sizes: {md_sizes}\n")
            self._append(f"[UI] AtomDisplacement sizes: {atom_sizes}\n")
            self._append(f"[UI] Workspaces: {WORKSPACES_ROOT / run_id}\n")
            self._append(f"[UI] Results root: {RESULTS_ROOT}\n")
            self._append(
                "[UI] ETA: el primer dataset de cada pipeline no tiene historico; "
                "los siguientes usan segundos/estructura de runs ya completados.\n"
            )
            for size in md_sizes:
                self._ensure_not_stopped()
                self._restore_original_config("md", original_configs)
                result = self._run_one("md", size, run_id)
                with self._lock:
                    self._results.append(result)
            for size in atom_sizes:
                self._ensure_not_stopped()
                self._restore_original_config("atom_displacement", original_configs)
                result = self._run_one("atom_displacement", size, run_id)
                with self._lock:
                    self._results.append(result)
            self._append("\n[UI] Comparacion experimental finalizada correctamente.\n")
        except Exception as exc:
            returncode = 1
            self._append(f"\n[ERROR] {exc}\n")
        finally:
            for key, raw in original_configs.items():
                PIPELINES[key].config_path.write_text(raw, encoding="utf-8")
            with self._lock:
                self._process = None
                self._current = None
                self._returncode = returncode
                self._finished_at = time.time()
            self._append("[UI] Configuraciones originales restauradas.\n")

    def _restore_original_config(self, key: str, original_configs: dict[str, str]) -> None:
        PIPELINES[key].config_path.write_text(original_configs[key], encoding="utf-8")

    def _ensure_not_stopped(self) -> None:
        with self._lock:
            if self._stop_requested:
                raise RuntimeError("Experimento detenido por el usuario.")

    def _run_one(self, key: str, size: int, run_id: str) -> dict[str, Any]:
        spec = PIPELINES[key]
        started_at = time.time()
        eta_seconds = self._estimated_seconds(key, size)
        self._set_current(key, size, started_at=started_at, eta_seconds=eta_seconds)
        self._append(f"\n[UI] === {spec.label} dataset_{size} ===\n")
        self._append(f"[UI] ETA inicial: {format_duration(eta_seconds)}\n")
        config = load_config(spec.config_path)
        workspace = WORKSPACES_ROOT / run_id / key / f"dataset_{size}"
        result_group = "results_md" if key == "md" else "results_atomdisp"
        result_dir = RESULTS_ROOT / result_group / f"dataset_{size}" / f"run_{run_id}"
        workspace.mkdir(parents=True, exist_ok=True)
        self._append(f"[UI] Workspace: {workspace}\n")
        self._append(f"[UI] Result dir previsto: {result_dir}\n")
        self._append(f"[UI] Config temporal: {spec.config_path}\n")
        with self._lock:
            log_start = len(self._logs)
        prepare_metadata: dict[str, Any] = {}
        if key == "md":
            self._prepare_md_config(config, workspace, size)
        else:
            prepare_metadata = self._prepare_atom_config(config, workspace, size)
        write_yaml(spec.config_path, config)
        self._append("[UI] Config temporal escrita; se restaurara al finalizar el experimento.\n")
        if key == "atom_displacement" and prepare_metadata.get("test_needs_siesta"):
            self._run_atom_test_single_points(spec, config, Path(prepare_metadata["test_samples_dir"]))
            write_yaml(spec.config_path, config)
            self._append("[UI] Config de entrenamiento restaurada tras SIESTA del test.\n")
        returncode = self._run_pipeline_process(spec, key=key, size=size, started_at=started_at)
        with self._lock:
            run_log = "".join(self._logs[log_start:])
        archive = self._archive_outputs(key, size, run_id, workspace, config, returncode, run_log)
        elapsed = time.time() - started_at
        self._update_rate(key, size, elapsed, returncode)
        if returncode != 0:
            raise RuntimeError(f"{spec.label} dataset_{size} fallo con codigo {returncode}.")
        return archive

    def _estimated_seconds(self, key: str, size: int, elapsed: float = 0.0) -> float | None:
        rate = self._rate_seconds_per_structure.get(key)
        if rate is None:
            return None
        return max(0.0, rate * size - elapsed)

    def _update_rate(self, key: str, size: int, elapsed: float, returncode: int) -> None:
        if returncode != 0 or size <= 0:
            return
        new_rate = elapsed / size
        old_rate = self._rate_seconds_per_structure.get(key)
        if old_rate is None:
            self._rate_seconds_per_structure[key] = new_rate
        else:
            self._rate_seconds_per_structure[key] = (old_rate * 0.6) + (new_rate * 0.4)
        self._append(
            f"[UI] ETA actualizado para {PIPELINES[key].label}: "
            f"{self._rate_seconds_per_structure[key]:.2f}s/estructura.\n"
        )

    def _prepare_md_config(self, config: dict[str, Any], workspace: Path, size: int) -> None:
        dataset_dir = workspace / "dataset"
        pseudo_count = copy_pseudopotentials(PIPELINES["md"].root / "dataset", dataset_dir)
        counts = split_counts(size)
        config["paths"]["dataset_dir"] = str(dataset_dir)
        config["paths"]["training_dir"] = str(workspace / "training")
        config["md"]["steps"] = size
        config["splits"] = {
            "enabled": True,
            "strategy": "spread",
            "train": counts["train"],
            "validation": counts["validation"],
            "test": counts["test"],
        }
        config["training"]["data"]["train_runs"] = "../dataset/splits/train/*/RUN.fdf"
        config["testing"]["test_runs"] = "../dataset/splits/test/*/RUN.fdf"
        config["prediction"]["predict_structs"] = "../dataset/splits/test/*/RUN.fdf"
        config["checkpoint"]["path"] = None
        self._append(f"[UI] MD dataset_dir: {dataset_dir}\n")
        self._append(f"[UI] MD training_dir: {workspace / 'training'}\n")
        self._append(f"[UI] MD pseudopotenciales copiados: {pseudo_count}\n")
        self._append(f"[UI] MD steps configurados: {size}\n")
        self._append(
            "[UI] MD split: "
            f"{counts['train']} train, {counts['test']} test, "
            f"{counts['validation']} validation; seleccion espaciada.\n"
        )
        self._append(f"[UI] MD train_runs: {config['training']['data']['train_runs']}\n")
        self._append(f"[UI] MD test_runs: {config['testing']['test_runs']}\n")

    def _prepare_atom_config(self, config: dict[str, Any], workspace: Path, size: int) -> dict[str, Any]:
        dataset_dir = workspace / "dataset"
        training_dir = workspace / "training"
        relaxed_dir = workspace / "relaxed"
        train_samples_dir = dataset_dir / "train_samples"
        validation_samples_dir = dataset_dir / "validation_samples"
        test_samples_dir = dataset_dir / "test_samples"
        basis_dir = dataset_dir / "basis"
        relaxed_counts = copy_relaxed_basis(PIPELINES["atom_displacement"].root / "relaxed", relaxed_dir)
        source_samples_dir = atom_source_samples_dir(PIPELINES["atom_displacement"], config)
        basis_count = copy_basis_files(source_samples_dir, basis_dir)
        all_samples = generated_atom_samples(source_samples_dir)
        completed_samples = completed_atom_samples(source_samples_dir)
        if len(all_samples) < size:
            raise RuntimeError(
                f"AtomDisplacement: se pidieron {size} estructuras, pero solo hay "
                f"{len(all_samples)} muestras generadas."
            )
        counts = split_counts(size)
        train_needed = counts["train"]
        validation_needed = counts["validation"]
        test_needed = counts["test"]
        selected_samples = select_spread(all_samples, size)
        split_samples = split_spread(selected_samples, counts)
        train_samples = split_samples["train"]
        validation_samples = split_samples["validation"]
        test_samples = split_samples["test"]
        incomplete_train = [
            path for path in train_samples if not is_completed_atom_sample(path)
        ]
        if incomplete_train:
            raise RuntimeError(
                "AtomDisplacement: no puedo entrenar ese tamano sin recalcular SIESTA "
                f"en train. Para dataset_{size}, el split 80/10/10 necesita "
                f"{train_needed} muestras de train con Hamiltoniano SIESTA; faltan "
                f"{len(incomplete_train)} en el split espaciado. Puedo limitar nuevos "
                "SIESTA al 10% de test, pero train tambien necesita referencias ya existentes."
            )
        if len(test_samples) < test_needed or len(validation_samples) < validation_needed:
            raise RuntimeError(
                "AtomDisplacement: no hay suficientes muestras para completar el split "
                f"80/10/10 ({len(train_samples)} train, {len(test_samples)} test, "
                f"{len(validation_samples)} validation)."
            )
        copy_sample_dirs(train_samples, train_samples_dir)
        copy_sample_dirs(validation_samples, validation_samples_dir)
        copy_sample_dirs(test_samples, test_samples_dir)
        test_needs_siesta = any(
            not is_completed_atom_sample(path)
            for path in test_samples_dir.iterdir()
            if path.is_dir() and (path / "RUN.fdf").exists()
        )
        config["paths"]["relaxed_dir"] = str(relaxed_dir)
        config["paths"]["dataset_dir"] = str(dataset_dir)
        config["paths"]["samples_dir"] = str(train_samples_dir)
        config["paths"]["validation_samples_dir"] = str(validation_samples_dir)
        config["paths"]["test_samples_dir"] = str(test_samples_dir)
        config["paths"]["collected_dir"] = str(dataset_dir / "collected")
        config["paths"]["training_dir"] = str(training_dir)
        config["single_points"]["rerun"] = False
        basis_files_pattern = "../dataset/basis/*.ion.xml" if basis_count else "../relaxed/*.ion.xml"
        config["training"]["data"]["basis_files"] = basis_files_pattern
        config["prediction"]["data"]["basis_files"] = basis_files_pattern
        config["testing"]["data"]["basis_files"] = basis_files_pattern
        config["testing"]["test_runs"] = "../dataset/test_samples/*/RUN.fdf"
        config["prediction"]["data"]["predict_structs"] = "../dataset/test_samples/*/RUN.fdf"
        config["checkpoint"]["path"] = None
        config["pipeline"]["steps"] = [
            "render_inputs",
            "run_atdisp_training",
            "run_atdisp_testing",
            "run_atdisp_prediction",
        ]
        self._append(f"[UI] AtomDisplacement dataset_dir: {dataset_dir}\n")
        self._append(f"[UI] AtomDisplacement train_samples_dir: {train_samples_dir}\n")
        self._append(f"[UI] AtomDisplacement validation_samples_dir: {validation_samples_dir}\n")
        self._append(f"[UI] AtomDisplacement test_samples_dir: {test_samples_dir}\n")
        self._append(f"[UI] AtomDisplacement training_dir: {training_dir}\n")
        self._append(f"[UI] AtomDisplacement relaxed_dir: {relaxed_dir}\n")
        self._append(f"[UI] AtomDisplacement source_samples_dir: {source_samples_dir}\n")
        self._append(
            "[UI] AtomDisplacement relaxed copiado: "
            f"{relaxed_counts['basis_files']} basis .ion.xml, {relaxed_counts['xv_files']} XV.\n"
        )
        self._append(f"[UI] AtomDisplacement basis dataset copiados: {basis_count}\n")
        self._append(
            f"[UI] AtomDisplacement disponibles: {len(all_samples)} generadas, "
            f"{len(completed_samples)} con referencia SIESTA.\n"
        )
        self._append(
            "[UI] AtomDisplacement split: "
            f"{len(train_samples)} train, {len(test_samples)} test, "
            f"{len(validation_samples)} validation; seleccion espaciada.\n"
        )
        self._append(f"[UI] AtomDisplacement train samples: {sample_names(train_samples)}\n")
        self._append(f"[UI] AtomDisplacement test samples: {sample_names(test_samples)}\n")
        self._append(f"[UI] AtomDisplacement validation samples: {sample_names(validation_samples)}\n")
        if test_needs_siesta:
            self._append("[UI] AtomDisplacement ejecutara SIESTA solo en el split de test.\n")
        else:
            self._append("[UI] AtomDisplacement reutiliza referencias SIESTA ya existentes para test.\n")
        self._append(f"[UI] AtomDisplacement test_runs: {config['testing']['test_runs']}\n")
        self._append(f"[UI] AtomDisplacement steps: {', '.join(config['pipeline']['steps'])}\n")
        return {"test_samples_dir": str(test_samples_dir), "test_needs_siesta": test_needs_siesta}

    def _run_pipeline_process(
        self,
        spec: PipelineSpec,
        *,
        key: str,
        size: int,
        started_at: float,
    ) -> int:
        config = load_config(spec.config_path)
        venv_activate = resolve_pipeline_path(spec, config["paths"]["venv_activate"])
        if not venv_activate.exists():
            raise RuntimeError(f"{spec.label}: no se encontro el entorno virtual: {venv_activate}")
        shell = str(config.get("commands", {}).get("shell", "bash"))
        python = str(config.get("commands", {}).get("python", "python"))
        shell_command = (
            f"source {shlex.quote(str(venv_activate))} "
            f"&& {shlex.quote(python)} {shlex.quote(str(spec.main_script))}"
        )
        command = [shell, "-lc", shell_command]
        self._append(f"[RUN] {' '.join(command)}\n")
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            command,
            cwd=spec.root,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        os.close(slave_fd)
        with self._lock:
            self._process = process
        self._append(f"[UI] PID: {process.pid}\n")
        self._append(f"[UI] CWD: {spec.root}\n")
        self._append(f"[UI] ETA al arrancar proceso: {format_duration(self._estimated_seconds(key, size))}\n")
        try:
            returncode = stream_process_output(
                process,
                self._append,
                label=spec.label,
                master_fd=master_fd,
                eta_provider=lambda: self._estimated_seconds(key, size, time.time() - started_at),
            )
        finally:
            os.close(master_fd)
        with self._lock:
            if self._process is process:
                self._process = None
        elapsed = time.time() - started_at
        self._append(
            f"[UI] {spec.label} finalizo con codigo {returncode} "
            f"en {format_duration(elapsed)}.\n"
        )
        return returncode

    def _run_atom_test_single_points(
        self,
        spec: PipelineSpec,
        config: dict[str, Any],
        test_samples_dir: Path,
    ) -> None:
        self._append(f"[UI] Preparando SIESTA solo para test: {test_samples_dir}\n")
        original_samples_dir = config["paths"]["samples_dir"]
        original_limit = config["single_points"].get("limit")
        original_steps = list(config["pipeline"]["steps"])
        config["paths"]["samples_dir"] = str(test_samples_dir)
        config["single_points"]["limit"] = None
        config["pipeline"]["steps"] = ["run_single_points"]
        write_yaml(spec.config_path, config)
        returncode = self._run_pipeline_process(
            spec,
            key=spec.key,
            size=max(
                1,
                len(
                    [
                        path
                        for path in test_samples_dir.iterdir()
                        if path.is_dir() and (path / "RUN.fdf").exists()
                    ]
                ),
            ),
            started_at=time.time(),
        )
        config["paths"]["samples_dir"] = original_samples_dir
        config["single_points"]["limit"] = original_limit
        config["pipeline"]["steps"] = original_steps
        if returncode != 0:
            raise RuntimeError(f"{spec.label}: fallo SIESTA del split de test con codigo {returncode}.")

    def _archive_outputs(
        self,
        key: str,
        size: int,
        run_id: str,
        workspace: Path,
        config: dict[str, Any],
        returncode: int,
        run_log: str,
    ) -> dict[str, Any]:
        result_group = "results_md" if key == "md" else "results_atomdisp"
        result_dir = RESULTS_ROOT / result_group / f"dataset_{size}" / f"run_{run_id}"
        result_dir.mkdir(parents=True, exist_ok=True)
        self._append(f"[UI] Archivando salidas en {result_dir}\n")
        write_yaml(result_dir / "pipeline_config.yaml", config)
        (result_dir / "run.log").write_text(run_log, encoding="utf-8")
        self._append(f"[UI] run.log guardado: {result_dir / 'run.log'}\n")

        dataset_dir = Path(config["paths"]["dataset_dir"])
        training_dir = Path(config["paths"]["training_dir"])
        prediction_count = 0
        reference_count = 0
        if key == "md":
            md_outputs_root = dataset_dir / "splits" / "test"
            if not md_outputs_root.exists():
                md_outputs_root = dataset_dir / "MD_steps"
            copy_matching_files(
                md_outputs_root,
                "*/RUN.fdf",
                result_dir / "structures",
            )
            prediction_count = copy_matching_files(
                md_outputs_root,
                "*/ML_prediction.HSX",
                result_dir / "predicted_hamiltonians",
            )
            reference_count += copy_matching_files(
                md_outputs_root,
                "*/siesta.TSHS",
                result_dir / "siesta_hamiltonians",
            )
            reference_count += copy_matching_files(
                md_outputs_root,
                "*/siesta.HSX",
                result_dir / "siesta_hamiltonians",
            )
        else:
            samples_dir = Path(config["paths"].get("test_samples_dir", config["paths"]["samples_dir"]))
            copy_matching_files(
                samples_dir,
                "*/RUN.fdf",
                result_dir / "structures",
            )
            copy_matching_files(
                samples_dir,
                "*/metadata.json",
                result_dir / "structures",
            )
            prediction_count = copy_matching_files(
                samples_dir,
                "*/ML_prediction.HSX",
                result_dir / "predicted_hamiltonians",
            )
            for sample_dir in sorted(
                path
                for path in samples_dir.iterdir()
                if path.is_dir() and (path / "RUN.fdf").exists()
            ):
                system_label = read_system_label(sample_dir / "RUN.fdf")
                canonical = [
                    path
                    for path in (
                        sample_dir / f"{system_label}.HSX",
                        sample_dir / f"{system_label}.TSHS",
                    )
                    if path.exists()
                ]
                sources = canonical or [
                    src
                    for src in sorted(
                        list(sample_dir.glob("*.HSX")) + list(sample_dir.glob("*.TSHS")),
                        key=natural_matrix_sort_key,
                    )
                    if src.name != "ML_prediction.HSX"
                ]
                for src in sources:
                    dst = result_dir / "siesta_hamiltonians" / sample_dir.name / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    reference_count += 1

        copy_if_exists(training_dir / "sample_metrics.csv", result_dir / "sample_metrics.csv")
        copy_if_exists(dataset_dir / "run_summary.json", result_dir / "run_summary.json")
        copy_if_exists(dataset_dir / "samples_manifest.json", result_dir / "samples_manifest.json")
        copy_if_exists(dataset_dir / "collected" / "water_atom_displacement_dataset.json", result_dir / "water_atom_displacement_dataset.json")
        copy_if_exists(dataset_dir / "collected" / "water_atom_displacement_summary.csv", result_dir / "water_atom_displacement_summary.csv")
        evaluation_metrics = self._evaluate_hamiltonian_metrics(key, config, result_dir)
        manifest = {
            "pipeline": key,
            "dataset_size": size,
            "run_id": run_id,
            "returncode": returncode,
            "workspace": str(workspace),
            "result_dir": str(result_dir),
            "predicted_hamiltonians": prediction_count,
            "siesta_hamiltonians": reference_count,
            "metrics": read_metrics_summary(result_dir / "sample_metrics.csv"),
            "hamiltonian_evaluation": evaluation_metrics,
        }
        (result_dir / "manifest.json").write_text(
            json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._append(
            f"[UI] Resultados archivados en {result_dir} | "
            f"predichos: {prediction_count} | siesta: {reference_count}\n"
        )
        return manifest

    def _evaluate_hamiltonian_metrics(
        self,
        key: str,
        config: dict[str, Any],
        result_dir: Path,
    ) -> dict[str, Any]:
        spec = PIPELINES[key]
        venv_activate = resolve_pipeline_path(spec, config["paths"]["venv_activate"])
        python = venv_activate.parent / "python"
        if not python.exists():
            python = Path(sys.executable)
        script = COMPARISON_ROOT / "scripts" / "evaluate_hamiltonian_metrics.py"
        command = [str(python), str(script), str(result_dir)]
        self._append(f"[UI] Calculando metricas sparse/espectro/DOS: {' '.join(command)}\n")
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            self._append(result.stdout.strip() + "\n")
        if result.stderr.strip():
            self._append(result.stderr.strip() + "\n")

        manifest_path = result_dir / "eigenvalues" / "manifest.json"
        if manifest_path.exists():
            try:
                summary = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                summary = {"exists": True, "error": str(exc)}
        else:
            summary = {"exists": False}
        summary["returncode"] = result.returncode
        if result.returncode == 0:
            self._append(
                "[UI] Metricas Hamiltonianas archivadas: "
                f"{summary.get('samples_compared', 0)} muestras comparadas.\n"
            )
        else:
            self._append(
                "[WARN] Evaluacion Hamiltoniana termino con codigo "
                f"{result.returncode}. Revisa {manifest_path}.\n"
            )
        return summary


EXPERIMENT_RUNNER = ExperimentRunner()


def all_status() -> dict[str, Any]:
    statuses = {key: runner.status() for key, runner in RUNNERS.items()}
    return {
        "running": any(status["running"] for status in statuses.values()),
        "pipelines": statuses,
    }


def run_all() -> dict[str, Any]:
    started: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key, runner in RUNNERS.items():
        try:
            started[key] = runner.start()
        except Exception as exc:
            errors[key] = str(exc)
    payload = all_status()
    payload["started"] = started
    payload["errors"] = errors
    if errors and not started:
        raise RuntimeError("; ".join(errors.values()))
    return payload


def stop_all() -> dict[str, Any]:
    for runner in RUNNERS.values():
        runner.stop()
    return all_status()


def result_summary() -> dict[str, Any]:
    md_predictions = sorted((REPO_ROOT / "MD" / "dataset" / "MD_steps").glob("*/ML_prediction.HSX"))
    atdisp_predictions = sorted(
        (REPO_ROOT / "AtomDisplacement" / "dataset" / "samples").glob("*/ML_prediction.HSX")
    )
    archived = archived_results_summary()
    return {
        "md": {
            "root": str(REPO_ROOT / "MD"),
            "metrics": str(REPO_ROOT / "MD" / "training" / "sample_metrics.csv"),
            "metrics_exists": (REPO_ROOT / "MD" / "training" / "sample_metrics.csv").exists(),
            "predictions": len(md_predictions),
            "prediction_glob": "MD/dataset/MD_steps/*/ML_prediction.HSX",
        },
        "atom_displacement": {
            "root": str(REPO_ROOT / "AtomDisplacement"),
            "metrics": str(REPO_ROOT / "AtomDisplacement" / "training" / "sample_metrics.csv"),
            "metrics_exists": (REPO_ROOT / "AtomDisplacement" / "training" / "sample_metrics.csv").exists(),
            "predictions": len(atdisp_predictions),
            "prediction_glob": "AtomDisplacement/dataset/samples/*/ML_prediction.HSX",
        },
        "archived": archived,
    }


def archived_results_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {"md": [], "atom_displacement": []}
    groups = {
        "md": RESULTS_ROOT / "results_md",
        "atom_displacement": RESULTS_ROOT / "results_atomdisp",
    }
    for key, root in groups.items():
        if not root.exists():
            continue
        for manifest_path in sorted(root.glob("dataset_*/run_*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary[key].append(manifest)
    return summary


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(
        json_safe(payload),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error_response(handler: BaseHTTPRequestHandler, exc: Exception, status: int = 400) -> None:
    json_response(handler, {"error": str(exc)}, status=status)


class ComparisonUIHandler(BaseHTTPRequestHandler):
    server_version = "ComparisonPipelineUI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[comparison-ui] {self.address_string()} - {fmt % args}\n")

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        try:
            if path == "/api/health":
                json_response(
                    self,
                    {
                        "ok": True,
                        "repo_root": str(REPO_ROOT),
                        "pipelines": {
                            key: {
                                "root": str(spec.root),
                                "config_path": str(spec.config_path),
                                "main_script": str(spec.main_script),
                            }
                            for key, spec in PIPELINES.items()
                        },
                    },
                )
            elif path == "/api/run/status":
                json_response(self, all_status())
            elif path == "/api/run/logs":
                query = parse_qs(parsed_url.query)
                key = query.get("pipeline", [""])[0]
                since = int(query.get("since", ["0"])[0])
                if key not in RUNNERS:
                    raise RuntimeError("Pipeline no reconocido.")
                json_response(self, RUNNERS[key].logs(since=since))
            elif path == "/api/results":
                json_response(self, result_summary())
            elif path == "/api/plots":
                json_response(self, plot_data_summary())
            elif path == "/api/experiment/status":
                json_response(self, EXPERIMENT_RUNNER.status())
            elif path == "/api/experiment/logs":
                query = parse_qs(parsed_url.query)
                since = int(query.get("since", ["0"])[0])
                json_response(self, EXPERIMENT_RUNNER.logs(since=since))
            elif path == "/":
                self._serve_file(UI_DIR / "index.html")
            else:
                requested = (UI_DIR / path.lstrip("/")).resolve()
                if UI_DIR.resolve() not in requested.parents:
                    raise FileNotFoundError(path)
                self._serve_file(requested)
        except FileNotFoundError:
            error_response(self, RuntimeError("No encontrado."), HTTPStatus.NOT_FOUND)
        except Exception as exc:
            error_response(self, exc)

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/api/run":
                json_response(self, run_all(), status=HTTPStatus.ACCEPTED)
            elif path == "/api/run/stop":
                json_response(self, stop_all(), status=HTTPStatus.ACCEPTED)
            elif path == "/api/experiment":
                payload = read_json_body(self)
                md_sizes = parse_sizes(payload.get("md_sizes"), [50, 100, 200, 500])
                atom_sizes = parse_sizes(payload.get("atom_sizes"), [100, 1000, 10000])
                json_response(
                    self,
                    EXPERIMENT_RUNNER.start(md_sizes, atom_sizes),
                    status=HTTPStatus.ACCEPTED,
                )
            elif path == "/api/experiment/stop":
                json_response(
                    self,
                    EXPERIMENT_RUNNER.stop(),
                    status=HTTPStatus.ACCEPTED,
                )
            else:
                raise FileNotFoundError(self.path)
        except FileNotFoundError:
            error_response(self, RuntimeError("No encontrado."), HTTPStatus.NOT_FOUND)
        except Exception as exc:
            error_response(self, exc)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }.get(path.suffix, "application/octet-stream")
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the combined comparison pipeline UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ComparisonUIHandler)
    print(f"Comparison Pipeline UI listening on http://{args.host}:{args.port}")
    print(f"Repo root: {REPO_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Comparison Pipeline UI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
