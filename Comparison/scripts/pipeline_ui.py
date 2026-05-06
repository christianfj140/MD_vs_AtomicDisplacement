#!/usr/bin/env python3
"""Local web UI and API for running MD and AtomDisplacement together."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import itertools
import math
import os
import json
import platform
import select
import shutil
import shlex
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml
from siesta_settings import DEFAULT_SHARED, compare_settings, file_digest
from model_settings import compare_model_settings

try:
    import pty
except Exception:  # pragma: no cover - Windows fallback.
    pty = None

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).resolve().parents[1] / "ui"
COMPARISON_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = COMPARISON_ROOT / "results"
WORKSPACES_ROOT = COMPARISON_ROOT / "workspaces"
LOG_HEARTBEAT_SECONDS = 30.0
METRIC_VERSION = "2026-05-04.strict-validation-v1"


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
    text = str(value)
    graph2mat_venv = os.environ.get("GRAPH2MAT_VENV", "")
    text = text.replace("${REPO_ROOT}", str(REPO_ROOT))
    if graph2mat_venv:
        text = text.replace("${GRAPH2MAT_VENV}", graph2mat_venv)
    text = os.path.expandvars(text)
    path = Path(text).expanduser()
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
        started_at = time.time()
        last_output = started_at
        last_heartbeat = last_output
        while True:
            line = process.stdout.readline()
            if line:
                last_output = time.time()
                append(line)
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
        return process.wait()
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
            master_fd: int | None = None
            if pty is not None:
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
            else:
                self._process = subprocess.Popen(
                    self._command,
                    cwd=self.spec.root,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
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

    def _collect_output(self, process: subprocess.Popen[str], master_fd: int | None) -> None:
        try:
            returncode = stream_process_output(
                process,
                lambda line: self._append_log(line),
                label=self.spec.label,
                master_fd=master_fd,
            )
        finally:
            if master_fd is not None:
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


def write_csv_dicts(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["sample_id", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_digest(paths_to_hash: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in paths_to_hash if path.exists() and path.is_file()}):
        digest.update(str(path).encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update((file_sha256(path) or "").encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def files_content_digest(paths_to_hash: list[Path]) -> str:
    entries = [
        (path.name, file_sha256(path) or "")
        for path in paths_to_hash
        if path.exists() and path.is_file()
    ]
    if not entries:
        return ""
    digest = hashlib.sha256()
    for name, sha in sorted(entries):
        digest.update(name.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty_warning() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "Could not determine git dirty state."
    return "Working tree has uncommitted changes." if result.stdout.strip() else ""


def command_version(command_name: str) -> str:
    candidates = ([command_name, "--version"], [command_name, "-V"])
    for command in candidates:
        try:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except Exception:
            continue
        output = (result.stdout or result.stderr or "").strip()
        if result.returncode == 0 and output:
            return output.splitlines()[0]
    return "unavailable"


def python_module_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:
        return "unavailable"
    return str(getattr(module, "__version__", "unavailable"))


def environment_versions(configs: list[dict[str, Any]] | None = None) -> dict[str, str]:
    configs = configs or []
    siesta = "siesta"
    graph2mat = "graph2mat"
    for config in configs:
        commands = config.get("commands", {}) if isinstance(config, dict) else {}
        siesta = str(commands.get("siesta", siesta))
        graph2mat = str(commands.get("graph2mat", graph2mat))
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit() or "unavailable",
        "siesta_version": command_version(siesta),
        "graph2mat_version": command_version(graph2mat),
        "sisl_version": python_module_version("sisl"),
    }


def absolute_path_warning(paths_to_check: list[Path]) -> str:
    home = str(Path.home())
    matches = [
        str(path)
        for path in paths_to_check
        if path.is_absolute() and (str(path).startswith(home) or str(path).startswith("/mnt/"))
    ]
    return "User-local absolute paths detected: " + ", ".join(matches) if matches else ""


def sample_set_hash(sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(set(sample_ids)):
        digest.update(sample_id.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()


def experiment_root(run_id: str) -> Path:
    return RESULTS_ROOT / run_id


def experiment_manifest_path(run_id: str) -> Path:
    return experiment_root(run_id) / "experiment_manifest.yaml"


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


DEFAULT_SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
DEFAULT_COMMON_TEST_SETS = ["test_md", "test_atomdisp", "test_mixed"]
DEFAULT_PRIMARY_METRIC = "fermi_window_rmse_eV"
MINIMUM_ROBUST_SEEDS = 3
STRICT_COMPARISON_MODE = True
SPLIT_MANIFEST_FIELDS = [
    "sample_id",
    "method",
    "source_run",
    "frame_index",
    "time_index",
    "displacement_amplitude",
    "displacement_magnitude",
    "displaced_atom",
    "displacement_axis",
    "displacement_sign",
    "displacement_family",
    "structure_path",
    "hamiltonian_path",
    "output_path",
    "run_out_path",
    "metadata_path",
    "valid",
    "validation_reason",
    "split",
    "split_group_id",
    "split_group_fields",
    "split_strategy",
    "seed",
    "status",
    "sample_dir",
]


def split_ratios_from_config(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("split_ratios") or config.get("splits") or {}
    ratios = {
        "train": float(raw.get("train", DEFAULT_SPLIT_RATIOS["train"])),
        "validation": float(
            raw.get("validation", raw.get("val", DEFAULT_SPLIT_RATIOS["validation"]))
        ),
        "test": float(raw.get("test", DEFAULT_SPLIT_RATIOS["test"])),
    }
    if any(value <= 0 or value >= 1 for value in ratios.values()):
        return dict(DEFAULT_SPLIT_RATIOS)
    return ratios


def parse_split_ratios(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError("Los splits deben enviarse como un objeto.")
    ratios = {
        "train": float(value.get("train", DEFAULT_SPLIT_RATIOS["train"])),
        "validation": float(
            value.get("validation", value.get("val", DEFAULT_SPLIT_RATIOS["validation"]))
        ),
        "test": float(value.get("test", DEFAULT_SPLIT_RATIOS["test"])),
    }
    return ratios


def validate_split_sizes(
    dataset_size: int,
    splits: dict[str, float],
    *,
    label: str = "dataset",
) -> dict[str, int]:
    if dataset_size < 3:
        raise RuntimeError(
            f"{label}: el dataset debe tener al menos 3 estructuras para que train, "
            "validation y test no queden vacios."
        )
    ratios = {
        "train": float(splits["train"]),
        "validation": float(splits.get("validation", splits.get("val", 0.0))),
        "test": float(splits["test"]),
    }
    if any(value <= 0 for value in ratios.values()):
        raise RuntimeError("Los ratios de split deben ser positivos.")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(
            "Los ratios de split deben sumar 1.0 "
            f"(recibido: {sum(ratios.values()):.6g})."
        )

    raw = {key: dataset_size * ratio for key, ratio in ratios.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = dataset_size - sum(counts.values())
    order = sorted(
        counts,
        key=lambda key: (raw[key] - counts[key], ratios[key]),
        reverse=True,
    )
    for key in order[:remainder]:
        counts[key] += 1

    empty = [key for key, count in counts.items() if count < 1]
    if empty:
        raise RuntimeError(
            f"{label}: split invalido; cada particion debe tener al menos 1 estructura. "
            f"dataset_size={dataset_size}, ratios={ratios}, counts={counts}, "
            f"vacios={empty}."
        )
    return counts


def atom_fc_sample_limit(config: dict[str, Any]) -> int | None:
    structure = config.get("structure", {})
    force_constants = structure.get("force_constants") or {}
    if not bool(force_constants.get("enabled", False)):
        return None

    atoms = structure.get("atoms") or []
    first_atom = int(force_constants.get("first_atom", 1))
    last_atom = force_constants.get("last_atom")
    last_atom = len(atoms) if last_atom is None else int(last_atom)
    if first_atom < 1 or last_atom < first_atom or last_atom > len(atoms):
        raise RuntimeError(
            "AtomDisplacement: rango FC invalido. "
            f"FC.First={first_atom}, FC.Last={last_atom}, NumberOfAtoms={len(atoms)}."
        )
    displaced_atoms = last_atom - first_atom + 1
    include_reference = bool(force_constants.get("include_reference", True))
    return (1 if include_reference else 0) + 6 * displaced_atoms


def atom_fc_displacement_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    force_constants = config.get("structure", {}).get("force_constants", {}) or {}
    entries = force_constants.get("displacements")
    if isinstance(entries, dict):
        return [
            {"value": key, "structure_options": value}
            for key, value in sorted(entries.items(), key=lambda item: _sort_displacement_key(item[0]))
        ]
    if entries:
        return [entry if isinstance(entry, dict) else {"value": entry} for entry in entries]
    return [{"value": force_constants.get("displacement", "0.05 Ang")}]


def parse_fc_displacements(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise RuntimeError("Las magnitudes FC deben enviarse como una lista.")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, dict):
            displacement = str(item.get("value", "")).strip()
            entry = {"value": displacement}
            if item.get("label"):
                entry["label"] = str(item["label"])
            if item.get("n_structures") not in (None, ""):
                entry["n_structures"] = int(item["n_structures"])
        else:
            displacement = str(item).strip()
            entry = {"value": displacement}
        if not displacement:
            raise RuntimeError(f"La magnitud FC #{index} no tiene valor.")
        entries.append(entry)
    return entries or None


def parse_structures_per_displacement(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, list):
        raw_items = value
    else:
        raise RuntimeError("structures_per_displacement debe ser lista o texto separado por comas.")
    counts: list[int] = []
    for item in raw_items:
        if item == "":
            continue
        count = int(item)
        if count <= 0:
            raise RuntimeError("structures_per_displacement solo acepta enteros positivos.")
        counts.append(count)
    return counts or None


def _sort_displacement_key(value: str) -> tuple[float, str]:
    text = str(value).strip()
    number = ""
    for char in text:
        if char.isdigit() or char in ".-+Ee":
            number += char
        elif number:
            break
    try:
        return (float(number), text)
    except ValueError:
        return (math.inf, text)


def _displacement_slug(value: str) -> str:
    text = str(value).strip()
    number = ""
    for char in text:
        if char.isdigit() or char in ".-+Ee":
            number += char
        elif number:
            break
    text = number or text
    slug = (
        text.replace("+", "")
        .replace("-", "m")
        .replace(".", "p")
        .replace(" ", "")
        .replace("/", "_")
    )
    return "".join(char if char.isalnum() else "_" for char in slug).strip("_") or "disp"


def parse_fc_displacement_options(value: Any) -> dict[str, list[int]] | None:
    """Parse the FC mapping: displacement -> user-defined counts.

    The API accepts a JSON object such as ``{"0.02 Ang": [5, 7], "0.05 Ang": [2]}``.
    Keeping the mapping separate from ``structures_per_displacement`` preserves
    backward compatibility with the older uniform grid. The mapping can be
    consumed in strict index-based ``aligned`` mode or opt-in ``cartesian`` mode.
    """

    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError("displacement_options debe ser un objeto displacement -> lista.")
    parsed: dict[str, list[int]] = {}
    for raw_key, raw_counts in value.items():
        displacement = str(raw_key).strip()
        if not displacement:
            raise RuntimeError("displacement_options contiene una magnitud vacia.")
        counts = parse_structures_per_displacement(raw_counts)
        if not counts:
            raise RuntimeError(f"{displacement}: define al menos un numero de estructuras.")
        parsed[displacement] = list(counts)
    return dict(sorted(parsed.items(), key=lambda item: _sort_displacement_key(item[0])))


def parse_max_datasets(value: Any, default: int = 100) -> int:
    if value in (None, ""):
        return default
    limit = int(value)
    if limit <= 0:
        raise RuntimeError("max_datasets debe ser mayor que cero.")
    return limit


def parse_combination_mode(value: Any) -> str:
    mode = "aligned" if value in (None, "") else str(value).strip().lower()
    if mode not in {"aligned", "cartesian"}:
        raise RuntimeError(
            "combination_mode debe ser 'aligned' o 'cartesian' "
            f"(recibido: {value!r})."
        )
    return mode


def parse_split_mode(value: Any) -> str:
    mode = "block" if value in (None, "") else str(value).strip().lower()
    if mode not in {"block", "spread", "blocked_with_gap"}:
        raise RuntimeError(
            "split_mode debe ser 'block', 'spread' o 'blocked_with_gap' "
            f"(recibido: {value!r})."
        )
    return mode


def parse_test_sets(value: Any) -> list[str]:
    if value in (None, ""):
        return list(DEFAULT_COMMON_TEST_SETS)
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, list):
        raw_items = [str(item).strip() for item in value]
    else:
        raise RuntimeError("test_sets debe ser lista o texto separado por comas.")
    selected = [item for item in raw_items if item]
    unknown = sorted(set(selected) - set(DEFAULT_COMMON_TEST_SETS))
    if unknown:
        raise RuntimeError(f"test_sets contiene valores no soportados: {unknown}.")
    return selected or list(DEFAULT_COMMON_TEST_SETS)


def parse_compute_budget_mode(value: Any) -> str:
    mode = "both" if value in (None, "") else str(value).strip().lower()
    allowed = {"equal_sample_count", "equal_siesta_budget", "both"}
    if mode not in allowed:
        raise RuntimeError(
            "compute_budget_mode debe ser equal_sample_count, equal_siesta_budget o both "
            f"(recibido: {value!r})."
        )
    return mode


def reference_budget_for_run(run: dict[str, Any]) -> int:
    for key in ("completed_samples", "effective_dataset_size", "dataset_size"):
        value = run.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return int(float(value))
        if isinstance(value, str) and value.strip().isdigit():
            return int(value)
    return 0


def budget_ratio(md_budget: int, atom_budget: int) -> float | None:
    if md_budget <= 0 or atom_budget <= 0:
        return None
    return max(md_budget, atom_budget) / min(md_budget, atom_budget)


def budget_warning(md_budget: int, atom_budget: int, *, tolerance: float = 1.25) -> str:
    ratio = budget_ratio(md_budget, atom_budget)
    if ratio is None:
        return "Budget could not be computed."
    if ratio > tolerance:
        return f"Budgets differ by ratio {ratio:.3g}; equal-budget comparison is approximate."
    return ""


def should_compare_budget_pair(
    md_run: dict[str, Any],
    atom_run: dict[str, Any],
    all_atom_runs: list[dict[str, Any]],
    mode: str,
) -> bool:
    md_size = int(md_run.get("dataset_size", 0))
    atom_size = int(atom_run.get("dataset_size", 0))
    if mode == "both":
        return True
    if mode == "equal_sample_count":
        return md_size == atom_size
    if mode == "equal_siesta_budget":
        md_budget = reference_budget_for_run(md_run)
        atom_budget = reference_budget_for_run(atom_run)
        if md_budget <= 0 or atom_budget <= 0:
            return False
        deltas = [
            abs(reference_budget_for_run(candidate) - md_budget)
            for candidate in all_atom_runs
            if reference_budget_for_run(candidate) > 0
        ]
        return bool(deltas) and abs(atom_budget - md_budget) == min(deltas)
    return True


def unique_ints_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    unique: list[int] = []
    for value in values:
        number = int(value)
        if number in seen:
            continue
        seen.add(number)
        unique.append(number)
    return unique


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


def distribute_fc_counts(total: int, n_displacements: int, per_displacement_limit: int) -> list[int]:
    if n_displacements <= 0:
        raise RuntimeError("AtomDisplacement: define al menos una magnitud FC.")
    capacity = n_displacements * per_displacement_limit
    if total > capacity:
        raise RuntimeError(
            "AtomDisplacement: el tamano pedido excede la capacidad FC configurada. "
            f"Pedido={total}, capacidad={capacity} ({n_displacements} magnitudes x "
            f"{per_displacement_limit} estructuras/magnitud)."
        )
    counts = [0 for _ in range(n_displacements)]
    remaining = total
    index = 0
    while remaining:
        if counts[index] < per_displacement_limit:
            counts[index] += 1
            remaining -= 1
        index = (index + 1) % n_displacements
    return counts


def make_aligned_dataset_label(entries: list[dict[str, Any]]) -> str:
    parts = [
        f"d{_displacement_slug(str(entry['value']))}_{int(entry['n_structures'])}"
        for entry in entries
    ]
    return "dataset_" + "__".join(parts)


def build_fc_aligned_dataset_specs(
    displacement_options: dict[str, list[int]],
    *,
    per_displacement_limit: int,
    split_ratios: dict[str, float],
    max_datasets: int,
) -> list[dict[str, Any]]:
    """Build one independent dataset spec per index-aligned combination.

    Counts are zipped across sorted displacement keys. With
    ``{"0.03": [5, 7], "0.04": [6, 8]}``, only two datasets are created:
    ``(5, 6)`` and ``(7, 8)``. This deliberately avoids Cartesian products so
    the user controls exactly which combinations are generated.
    """

    keys = list(displacement_options)
    lengths = {key: len(displacement_options[key]) for key in keys}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            "Todas las listas de displacement_options deben tener la misma longitud. "
            f"Longitudes recibidas: {lengths}."
        )
    n_datasets = next(iter(lengths.values()), 0)
    if n_datasets > max_datasets:
        raise RuntimeError(
            "La configuracion FC genera demasiados datasets: "
            f"{n_datasets} datasets alineados > max_datasets={max_datasets}. "
            "Reduce filas o aumenta max_datasets."
        )
    specs: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for dataset_index, combo in enumerate(zip(*(displacement_options[key] for key in keys))):
        entries = []
        for displacement, count in zip(keys, combo):
            if count > per_displacement_limit:
                raise RuntimeError(
                    f"{displacement}: pide {count} estructuras, pero el limite FC "
                    f"por magnitud es {per_displacement_limit}."
                )
            entries.append({"value": displacement, "n_structures": int(count)})
        size = sum(int(entry["n_structures"]) for entry in entries)
        label = f"dataset_{dataset_index}_" + make_aligned_dataset_label(entries).removeprefix("dataset_")
        validate_split_sizes(size, split_ratios, label=label)
        if label in seen_labels:
            raise RuntimeError(f"Nombre de dataset duplicado: {label}.")
        seen_labels.add(label)
        specs.append({"label": label, "size": size, "displacements": entries})
    return specs


def build_fc_cartesian_dataset_specs(
    displacement_options: dict[str, list[int]],
    *,
    per_displacement_limit: int,
    split_ratios: dict[str, float],
    max_datasets: int,
) -> list[dict[str, Any]]:
    """Build one dataset per Cartesian product of displacement-count options.

    Counts are expanded across sorted displacement keys. With
    ``{"0.03": [5, 7], "0.04": [6, 8]}``, four datasets are created:
    ``(5, 6)``, ``(5, 8)``, ``(7, 6)``, and ``(7, 8)``. This mode is explicit
    and opt-in because the number of datasets grows multiplicatively.
    """

    keys = list(displacement_options)
    if not keys:
        raise RuntimeError("Cartesian FC requiere al menos una magnitud.")
    n_datasets = math.prod(len(displacement_options[key]) for key in keys)
    if n_datasets > max_datasets:
        raise RuntimeError(
            "La configuracion FC genera demasiados datasets: "
            f"{n_datasets} datasets cartesianos > max_datasets={max_datasets}. "
            "Reduce opciones o aumenta max_datasets."
        )

    specs: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for dataset_index, combo in enumerate(
        itertools.product(*(displacement_options[key] for key in keys))
    ):
        entries = []
        for displacement, count in zip(keys, combo):
            if count > per_displacement_limit:
                raise RuntimeError(
                    f"{displacement}: pide {count} estructuras, pero el limite FC "
                    f"por magnitud es {per_displacement_limit}."
                )
            entries.append({"value": displacement, "n_structures": int(count)})
        size = sum(int(entry["n_structures"]) for entry in entries)
        label = f"dataset_{dataset_index}_" + make_aligned_dataset_label(entries).removeprefix("dataset_")
        validate_split_sizes(size, split_ratios, label=label)
        if label in seen_labels:
            raise RuntimeError(f"Nombre de dataset duplicado: {label}.")
        seen_labels.add(label)
        specs.append({"label": label, "size": size, "displacements": entries})
    return specs


def build_fc_dataset_specs_from_options(
    displacement_options: dict[str, list[int]],
    *,
    combination_mode: str,
    per_displacement_limit: int,
    split_ratios: dict[str, float],
    max_datasets: int,
) -> list[dict[str, Any]]:
    if combination_mode == "aligned":
        return build_fc_aligned_dataset_specs(
            displacement_options,
            per_displacement_limit=per_displacement_limit,
            split_ratios=split_ratios,
            max_datasets=max_datasets,
        )
    if combination_mode == "cartesian":
        return build_fc_cartesian_dataset_specs(
            displacement_options,
            per_displacement_limit=per_displacement_limit,
            split_ratios=split_ratios,
            max_datasets=max_datasets,
        )
    raise RuntimeError(f"combination_mode no soportado: {combination_mode!r}.")


def build_fc_dataset_specs(
    atom_sizes: list[int],
    fc_displacements: list[dict[str, Any]] | None,
    structures_per_displacement: list[int] | None,
    *,
    per_displacement_limit: int,
    split_ratios: dict[str, float],
) -> tuple[list[int], dict[int, list[dict[str, Any]]] | None]:
    if structures_per_displacement:
        if not fc_displacements:
            raise RuntimeError(
                "Define magnitudes FC para usar structures_per_displacement."
            )
        specs: dict[int, list[dict[str, Any]]] = {}
        generated_sizes: list[int] = []
        for count in structures_per_displacement:
            if count > per_displacement_limit:
                raise RuntimeError(
                    "AtomDisplacement: structures_per_displacement excede el limite FC. "
                    f"Pedido={count}, maximo por magnitud={per_displacement_limit}."
                )
            size = count * len(fc_displacements)
            validate_split_sizes(size, split_ratios, label=f"AtomDisplacement dataset_{size}")
            if size in specs:
                raise RuntimeError(
                    "Dos entradas de structures_per_displacement producen el mismo "
                    f"dataset_size={size}."
                )
            generated_sizes.append(size)
            specs[size] = [
                {**entry, "n_structures": count}
                for entry in fc_displacements
            ]
        return generated_sizes, specs

    if fc_displacements and all("n_structures" in entry for entry in fc_displacements):
        explicit_total = sum(int(entry["n_structures"]) for entry in fc_displacements)
        for entry in fc_displacements:
            requested = int(entry["n_structures"])
            if requested > per_displacement_limit:
                raise RuntimeError(
                    "AtomDisplacement: una magnitud FC pide mas estructuras de las que "
                    f"permite FC. {entry.get('value')} pide {requested}, "
                    f"maximo {per_displacement_limit}."
                )
        validate_split_sizes(
            explicit_total,
            split_ratios,
            label=f"AtomDisplacement dataset_{explicit_total}",
        )
        return [explicit_total], {explicit_total: fc_displacements}

    return atom_sizes, None


def validate_atom_sizes_for_fc(
    atom_sizes: list[int],
    fc_dataset_specs: dict[int, list[dict[str, Any]]] | None = None,
    split_ratios: dict[str, float] | None = None,
) -> None:
    config = load_config(PIPELINES["atom_displacement"].config_path)
    limit = atom_fc_sample_limit(config)
    if limit is None:
        return
    ratios = split_ratios or split_ratios_from_config(config)
    displacement_entries = (
        next(iter(fc_dataset_specs.values()))
        if fc_dataset_specs
        else atom_fc_displacement_entries(config)
    )
    displacement_count = len(displacement_entries)
    if fc_dataset_specs:
        for size, entries in fc_dataset_specs.items():
            validate_split_sizes(size, ratios, label=f"AtomDisplacement dataset_{size}")
            for entry in entries:
                requested = int(entry.get("n_structures") or 0)
                if requested > limit:
                    raise RuntimeError(
                        "AtomDisplacement: una magnitud FC pide mas estructuras de "
                        f"las que permite FC. {entry.get('value')} pide {requested}, "
                        f"maximo {limit}."
                    )
    else:
        for size in atom_sizes:
            validate_split_sizes(size, ratios, label=f"AtomDisplacement dataset_{size}")
    total_limit = limit * displacement_count
    too_large = [size for size in atom_sizes if size > total_limit]
    if not too_large:
        return
    raise RuntimeError(
        "AtomDisplacement usa SIESTA MD.TypeOfRun FC, que no genera un numero "
        "arbitrario de estructuras. Con la configuracion actual "
        f"({len(config['structure']['atoms'])} atomos, FC.First="
        f"{config['structure']['force_constants'].get('first_atom', 1)}, "
        f"FC.Last={config['structure']['force_constants'].get('last_atom') or len(config['structure']['atoms'])}) "
        f"el maximo por magnitud es {limit} estructuras. Con "
        f"{displacement_count} magnitudes, la capacidad total es {total_limit}. "
        f"Tamanos invalidos: {too_large}. Anade magnitudes FC o reduce el tamano."
    )


def validate_atom_dataset_specs_for_fc(
    specs: list[dict[str, Any]],
    split_ratios: dict[str, float],
) -> None:
    config = load_config(PIPELINES["atom_displacement"].config_path)
    limit = atom_fc_sample_limit(config)
    if limit is None:
        return
    for spec in specs:
        label = str(spec.get("label") or f"dataset_{spec.get('size')}")
        size = int(spec["size"])
        validate_split_sizes(size, split_ratios, label=f"AtomDisplacement {label}")
        entries = spec.get("displacements") or []
        if not entries:
            raise RuntimeError(f"AtomDisplacement {label}: no tiene magnitudes FC.")
        for entry in entries:
            requested = int(entry.get("n_structures") or 0)
            if requested <= 0:
                raise RuntimeError(
                    f"AtomDisplacement {label}: {entry.get('value')} pide "
                    f"{requested} estructuras; debe ser un entero positivo."
                )
            if requested > limit:
                raise RuntimeError(
                    "AtomDisplacement: una magnitud FC pide mas estructuras de "
                    f"las que permite FC. {label}, {entry.get('value')} pide "
                    f"{requested}, maximo {limit}."
                )


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


ATOM_SPLIT_GROUP_FIELDS = [
    "raw_displacement_run_id",
    "raw_fc_run_dir",
    "displacement_input",
    "displacement_ang",
    "atom",
    "direction",
    "sign",
]


def atom_split_group_id(sample_dir: Path) -> str:
    metadata = load_sample_metadata(sample_dir)
    values = {field: metadata.get(field, "") for field in ATOM_SPLIT_GROUP_FIELDS}
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    readable = "|".join(str(values[field]) for field in ATOM_SPLIT_GROUP_FIELDS if values[field] not in (None, ""))
    return readable or digest


def split_grouped_exact(items: list[Path], counts: dict[str, int]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for item in items:
        groups[atom_split_group_id(item)].append(item)
    ordered_groups = [(key, sorted(value, key=sample_sort_key)) for key, value in sorted(groups.items())]
    result = {"train": [], "validation": [], "test": []}
    for split_name in ("test", "validation", "train"):
        target = counts[split_name]
        for key, samples in list(ordered_groups):
            if len(result[split_name]) + len(samples) > target:
                continue
            result[split_name].extend(samples)
            ordered_groups.remove((key, samples))
            if len(result[split_name]) == target:
                break
        if len(result[split_name]) != target:
            raise RuntimeError(
                "AtomDisplacement grouped split no puede satisfacer los tamanos exactos "
                f"sin partir familias: {split_name} necesita {target}, obtuvo {len(result[split_name])}."
            )
    return result


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


def find_reference_matrix(sample_dir: Path) -> Path | None:
    run_fdf = sample_dir / "RUN.fdf"
    if run_fdf.exists():
        system_label = read_system_label(run_fdf)
        for candidate in (sample_dir / f"{system_label}.TSHS", sample_dir / f"{system_label}.HSX"):
            if candidate.exists():
                return candidate
    for candidate in (sample_dir / "siesta.TSHS", sample_dir / "siesta.HSX"):
        if candidate.exists():
            return candidate
    candidates = sorted(
        [
            path
            for path in list(sample_dir.glob("*.TSHS")) + list(sample_dir.glob("*.HSX"))
            if path.name != "ML_prediction.HSX"
        ],
        key=natural_matrix_sort_key,
    )
    return candidates[0] if candidates else None


def reference_matrices(sample_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in list(sample_dir.glob("*.TSHS")) + list(sample_dir.glob("*.HSX"))
            if path.name != "ML_prediction.HSX"
        ],
        key=natural_matrix_sort_key,
    )


def sample_output_status(run_out: Path) -> tuple[bool, str]:
    if not run_out.exists():
        return False, "missing_output"
    try:
        text = run_out.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False, "parser_error"
    reasons = []
    if "Job completed" not in text:
        reasons.append("job_not_completed")
    if "SCF cycle converged" not in text:
        reasons.append("scf_not_converged")
    return not reasons, "ok" if not reasons else ";".join(reasons)


def validated_reference_for_sample(sample_dir: Path) -> tuple[Path | None, bool, str]:
    reasons = []
    if not (sample_dir / "RUN.fdf").exists():
        reasons.append("missing_run_fdf")
    matrices = reference_matrices(sample_dir)
    if not matrices:
        reasons.append("missing_matrix")
    if len(matrices) > 1:
        reasons.append("ambiguous_reference_matrix")
    output_ok, output_reason = sample_output_status(sample_dir / "RUN.out")
    if not output_ok:
        reasons.extend(output_reason.split(";"))
    valid = not reasons
    return (matrices[0] if len(matrices) == 1 else None), valid, "ok" if valid else ";".join(reasons)


def load_sample_metadata(sample_dir: Path) -> dict[str, Any]:
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def atom_displacement_family(metadata: dict[str, Any]) -> str:
    parts = [
        metadata.get("generation_mode"),
        metadata.get("raw_displacement_run_id"),
        metadata.get("matrix_label"),
        metadata.get("direction"),
        metadata.get("sign"),
    ]
    return "|".join(str(part) for part in parts if part not in (None, ""))


def write_atom_split_manifests(
    dataset_dir: Path,
    split_samples: dict[str, list[Path]],
) -> dict[str, Path]:
    split_root = dataset_dir / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    paths_by_split: dict[str, Path] = {}
    for split_name, sample_dirs in split_samples.items():
        rows: list[dict[str, Any]] = []
        for sample_dir in sample_dirs:
            copied_dir = {
                "train": dataset_dir / "train_samples",
                "validation": dataset_dir / "validation_samples",
                "test": dataset_dir / "test_samples",
            }[split_name] / sample_dir.name
            metadata = load_sample_metadata(copied_dir)
            structure_path = copied_dir / "RUN.fdf"
            hamiltonian_path = find_reference_matrix(copied_dir)
            run_out_path = copied_dir / "RUN.out"
            if not run_out_path.exists() and metadata.get("raw_fc_run_dir"):
                raw_run_out = Path(str(metadata["raw_fc_run_dir"])) / "RUN.out"
                if raw_run_out.exists():
                    run_out_path = raw_run_out
            metadata_path = copied_dir / "metadata.json"
            hamiltonian_path, valid, validation_reason = validated_reference_for_sample(copied_dir)
            displacement = metadata.get("displacement_ang", "")
            group_id = atom_split_group_id(copied_dir)
            rows.append(
                {
                    "sample_id": f"atomdisp_{sample_dir.name}",
                    "method": "atom_displacement",
                    "source_run": str(metadata.get("raw_fc_run_dir") or sample_dir.parent),
                    "frame_index": "",
                    "time_index": "",
                    "displacement_amplitude": displacement,
                    "displacement_magnitude": displacement,
                    "displaced_atom": metadata.get("atom", ""),
                    "displacement_axis": metadata.get("direction", ""),
                    "displacement_sign": metadata.get("sign", ""),
                    "displacement_family": atom_displacement_family(metadata),
                    "structure_path": str(structure_path),
                    "hamiltonian_path": str(hamiltonian_path or ""),
                    "output_path": str(run_out_path) if run_out_path.exists() else "",
                    "run_out_path": str(run_out_path) if run_out_path.exists() else "",
                    "metadata_path": str(metadata_path) if metadata_path.exists() else "",
                    "valid": valid,
                    "validation_reason": validation_reason,
                    "split": split_name,
                    "split_group_id": group_id,
                    "split_group_fields": ",".join(ATOM_SPLIT_GROUP_FIELDS),
                    "split_strategy": "grouped_exact",
                    "seed": metadata.get("subsampling", {}).get("seed", ""),
                    "status": "completed" if valid else "incomplete",
                    "sample_dir": str(copied_dir),
                }
            )
        manifest_path = split_root / f"{split_name}_manifest.csv"
        write_csv_dicts(manifest_path, rows, SPLIT_MANIFEST_FIELDS)
        paths_by_split[split_name] = manifest_path
    return paths_by_split


def is_completed_atom_sample(sample_dir: Path) -> bool:
    _matrix, valid, _reason = validated_reference_for_sample(sample_dir)
    return valid


def atom_source_samples_dir(spec: PipelineSpec, config: dict[str, Any]) -> Path:
    dataset_dir = resolve_pipeline_path(spec, config["paths"]["dataset_dir"])
    fc_steps_dir = dataset_dir / "FC_steps"
    if fc_steps_dir.exists():
        return fc_steps_dir
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
    if source_samples_dir.name in {"AtDis_steps", "FC_steps"}:
        return sorted(
            [
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
    if source_samples_dir.name in {"AtDis_steps", "FC_steps"}:
        return sorted(
            [
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


def find_latest_checkpoint(training_dir: Path, config: dict[str, Any]) -> Path | None:
    manifest_path = training_dir / "checkpoint_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint_value = manifest.get("checkpoint_path") or manifest.get("path")
            if checkpoint_value:
                candidate = Path(str(checkpoint_value))
                if not candidate.is_absolute():
                    candidate = training_dir / candidate
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    configured = config.get("checkpoint", {}).get("path")
    if configured:
        candidate = Path(str(configured))
        if not candidate.is_absolute():
            candidate = training_dir / candidate
        if candidate.exists():
            return candidate
    search_glob = str(config.get("checkpoint", {}).get("search_glob", "lightning_logs/**/checkpoints/*.ckpt"))
    candidates = sorted(
        [path for path in training_dir.glob(search_glob) if path.is_file()],
        key=lambda path: path.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]
    fallback = sorted(training_dir.rglob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    return fallback[-1] if fallback else None


def checkpoint_selection_warning(training_dir: Path, checkpoint_path: Path | None) -> str:
    manifest_path = training_dir / "checkpoint_manifest.json"
    if checkpoint_path is None:
        return "No checkpoint was selected."
    if not manifest_path.exists():
        return "Checkpoint selected by latest-version fallback; strict scientific comparison should use checkpoint_manifest.json."
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Checkpoint manifest could not be parsed: {exc}"
    expected = manifest.get("checkpoint_path") or manifest.get("path")
    if not expected:
        return "Checkpoint manifest does not define checkpoint_path."
    expected_path = Path(str(expected))
    if not expected_path.is_absolute():
        expected_path = training_dir / expected_path
    if expected_path.resolve() != checkpoint_path.resolve():
        return "Selected checkpoint does not match checkpoint_manifest.json."
    expected_hash = manifest.get("checkpoint_sha256") or manifest.get("sha256")
    actual_hash = file_sha256(checkpoint_path)
    if expected_hash and actual_hash and str(expected_hash) != str(actual_hash):
        return "Selected checkpoint hash does not match checkpoint_manifest.json."
    return ""


def checkpoint_metadata(checkpoint_path: Path | None, training_dir: Path) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"path": None, "sha256": None, "relative_path": None, "selection": None}
    relative_path = None
    try:
        relative_path = checkpoint_path.relative_to(training_dir).as_posix()
    except ValueError:
        relative_path = str(checkpoint_path)
    version = None
    for part in checkpoint_path.parts:
        if part.startswith("version_") and part.removeprefix("version_").isdigit():
            version = int(part.removeprefix("version_"))
    return {
        "path": str(checkpoint_path),
        "relative_path": relative_path,
        "sha256": file_sha256(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else None,
        "mtime": checkpoint_path.stat().st_mtime if checkpoint_path.exists() else None,
        "version": version,
        "selection": "manifest_signed_checkpoint",
    }


def write_checkpoint_manifest(training_dir: Path, metadata: dict[str, Any], warning: str) -> Path:
    path = training_dir / "checkpoint_manifest.json"
    payload = {
        "checkpoint_path": metadata.get("path"),
        "checkpoint_sha256": metadata.get("sha256"),
        "relative_path": metadata.get("relative_path"),
        "selection_reason": "manifest" if not warning else "latest_version_fallback",
        "checkpoint_selection_warning": warning,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "training_dir": str(training_dir),
    }
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return path


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


def finite_metric_count(rows: list[dict[str, Any]], metric: str) -> int:
    count = 0
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            count += 1
    return count


def metric_diagnostics(
    spectral_rows: list[dict[str, Any]],
    relationship_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    missing_fermi_errors = [
        error
        for error in errors
        if error.get("kind") == "missing_fermi_level"
    ]
    unavailable_fermi_rows = [
        row
        for row in spectral_rows
        if row.get("fermi_level_source") == "unavailable"
    ]
    return {
        "spectral_samples": len(spectral_rows),
        "fermi_window_samples": finite_metric_count(spectral_rows, "fermi_window_rmse_eV"),
        "matrix_spectrum_samples": len(relationship_rows),
        "matrix_spectrum_fermi_samples": finite_metric_count(
            relationship_rows,
            "fermi_window_rmse_eV",
        ),
        "missing_fermi_level_samples": len(missing_fermi_errors),
        "unavailable_fermi_source_samples": len(unavailable_fermi_rows),
        "errors": errors,
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
                result_dir = manifest_path.parent / result_dir
            if not result_dir.exists():
                result_dir = manifest_path.parent
            sparse_rows = read_csv_rows(result_dir / "metrics" / "sparse_metrics.csv")
            spectral_rows = read_csv_rows(result_dir / "metrics" / "spectral_metrics.csv")
            dos_rows = read_csv_rows(result_dir / "metrics" / "dos_metrics.csv")
            sparse_sweep_rows = read_csv_rows(result_dir / "metrics" / "sparse_threshold_sweep.csv")
            dos_sweep_rows = read_csv_rows(result_dir / "metrics" / "dos_sigma_sweep.csv")
            relationship_rows = read_csv_rows(
                result_dir / "metrics" / "matrix_spectrum_relationship.csv"
            )
            if not sparse_rows and not spectral_rows and not dos_rows and not relationship_rows and not sparse_sweep_rows and not dos_sweep_rows:
                continue
            errors = manifest.get("errors", [])
            if not isinstance(errors, list):
                errors = []
            runs.append(
                {
                    "pipeline": key,
                    "label": PIPELINES[key].label,
                    "dataset_size": int(
                        manifest.get(
                            "requested_dataset_size",
                            manifest.get("dataset_size", 0),
                        )
                    ),
                    "effective_dataset_size": int(
                        manifest.get(
                            "effective_dataset_size",
                            manifest.get("dataset_size", 0),
                        )
                    ),
                    "requested_dataset_size": int(
                        manifest.get(
                            "requested_dataset_size",
                            manifest.get("dataset_size", 0),
                        )
                    ),
                    "run_id": str(manifest.get("run_id", manifest_path.parent.name.removeprefix("run_"))),
                    "result_dir": str(result_dir),
                    "pipeline_elapsed_seconds": manifest.get("pipeline_elapsed_seconds"),
                    "means": {
                        "run": {
                            "pipeline_elapsed_seconds": manifest.get("pipeline_elapsed_seconds"),
                        }
                        if isinstance(manifest.get("pipeline_elapsed_seconds"), (int, float))
                        else {},
                        "sparse": numeric_means(sparse_rows),
                        "spectral": numeric_means(spectral_rows),
                        "dos": numeric_means(dos_rows),
                        "sparse_sweep": numeric_means(sparse_sweep_rows),
                        "dos_sweep": numeric_means(dos_sweep_rows),
                        "matrix_spectrum": numeric_means(relationship_rows),
                    },
                    "samples": {
                        "sparse": sparse_rows,
                        "spectral": spectral_rows,
                        "dos": dos_rows,
                        "sparse_sweep": sparse_sweep_rows,
                        "dos_sweep": dos_sweep_rows,
                        "matrix_spectrum": relationship_rows,
                    },
                    "diagnostics": metric_diagnostics(
                        spectral_rows,
                        relationship_rows,
                        errors,
                    ),
                    "summary": manifest.get("summary", {}),
                }
            )
    runs.sort(key=lambda item: (item["pipeline"], item["dataset_size"], item["run_id"]))
    cross_experiments: list[dict[str, Any]] = []
    for metrics_path in sorted(RESULTS_ROOT.glob("*/summary/cross_evaluation_metrics.csv")):
        experiment_dir = metrics_path.parents[1]
        rows = read_csv_rows(metrics_path)
        recommendation_path = experiment_dir / "summary" / "recommendation.json"
        manifest_path = experiment_dir / "experiment_manifest.yaml"
        recommendation: dict[str, Any] = {}
        if recommendation_path.exists():
            try:
                recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
            except Exception as exc:
                recommendation = {"error": str(exc)}
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = load_config(manifest_path)
            except Exception as exc:
                manifest = {"error": str(exc)}
        cross_experiments.append(
            {
                "experiment_id": experiment_dir.name,
                "metrics": rows,
                "recommendation": recommendation,
                "manifest": manifest,
                "outputs": {
                    "cross_evaluation_metrics": str(metrics_path),
                    "winner_summary": str(experiment_dir / "summary" / "winner_summary.csv"),
                    "winner_by_dataset_size": str(experiment_dir / "summary" / "winner_by_dataset_size.csv"),
                    "winner_by_compute_budget": str(experiment_dir / "summary" / "winner_by_compute_budget.csv"),
                },
            }
        )
    return {"runs": runs, "cross_experiments": cross_experiments}


def atom_fc_ui_config() -> dict[str, Any]:
    config = load_config(PIPELINES["atom_displacement"].config_path)
    limit = atom_fc_sample_limit(config)
    force_constants = config.get("structure", {}).get("force_constants", {}) or {}
    entries = atom_fc_displacement_entries(config)
    normalized = []
    for entry in entries:
        normalized.append(
            {
                "value": entry.get("value", entry.get("displacement", "")),
                "n_structures": entry.get("n_structures") or "",
                "label": entry.get("label", ""),
            }
        )
    structures_per_displacement = force_constants.get("structures_per_displacement")
    if not structures_per_displacement:
        structures_per_displacement = [
            entry.get("n_structures")
            for entry in entries
            if entry.get("n_structures") not in (None, "")
        ]
    displacement_options = force_constants.get("displacement_options")
    if displacement_options is None and isinstance(force_constants.get("displacements"), dict):
        displacement_options = force_constants.get("displacements")
    if displacement_options is None:
        option_counts = structures_per_displacement or [2, 4, 6]
        displacement_options = {
            str(entry["value"]): option_counts
            for entry in entries
            if entry.get("value")
        }
    return {
        "max_per_displacement": limit,
        "include_reference": bool(force_constants.get("include_reference", True)),
        "subsampling": force_constants.get("subsampling", {}),
        "random_seed": force_constants.get(
            "random_seed",
            (force_constants.get("subsampling") or {}).get("seed", 0),
        ),
        "structures_per_displacement": structures_per_displacement or [2, 4, 6],
        "displacement_options": displacement_options,
        "combination_mode": parse_combination_mode(force_constants.get("combination_mode", "aligned")),
        "max_datasets": force_constants.get("max_datasets", 100),
        "splits": split_ratios_from_config(config),
        "displacements": normalized,
    }


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

    def start(
        self,
        md_sizes: list[int],
        atom_sizes: list[int],
        fc_dataset_specs: dict[int, list[dict[str, Any]]] | None = None,
        atom_dataset_specs: list[dict[str, Any]] | None = None,
        split_ratios: dict[str, float] | None = None,
        random_seed: int | None = None,
        split_mode: str = "block",
        test_sets: list[str] | None = None,
        primary_metric: str = DEFAULT_PRIMARY_METRIC,
        compute_budget_mode: str = "both",
    ) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Ya hay una comparacion experimental en ejecucion.")
            md_config = load_config(PIPELINES["md"].config_path)
            ratios = split_ratios or split_ratios_from_config(md_config)
            for size in md_sizes:
                validate_split_sizes(size, ratios, label=f"MD dataset_{size}")
            if atom_dataset_specs:
                atom_sizes = [int(spec["size"]) for spec in atom_dataset_specs]
                validate_atom_dataset_specs_for_fc(atom_dataset_specs, ratios)
            else:
                validate_atom_sizes_for_fc(atom_sizes, fc_dataset_specs, ratios)
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
                args=(
                    md_sizes,
                    atom_sizes,
                    self._run_id,
                    fc_dataset_specs,
                    atom_dataset_specs,
                    ratios,
                    random_seed,
                    split_mode,
                    test_sets or list(DEFAULT_COMMON_TEST_SETS),
                    primary_metric,
                    compute_budget_mode,
                ),
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

    def _initial_experiment_manifest(
        self,
        run_id: str,
        md_sizes: list[int],
        atom_sizes: list[int],
        split_ratios: dict[str, float],
        random_seed: int | None,
        split_mode: str,
        atom_dataset_specs: list[dict[str, Any]] | None,
        test_sets: list[str],
        primary_metric: str,
        compute_budget_mode: str,
    ) -> dict[str, Any]:
        md_config = load_config(PIPELINES["md"].config_path)
        atom_config = load_config(PIPELINES["atom_displacement"].config_path)
        shared_settings = load_config(DEFAULT_SHARED) if DEFAULT_SHARED.exists() else {}
        siesta_report = compare_settings(md_config, atom_config, shared_settings)
        model_report = compare_model_settings(md_config, atom_config)
        files_to_hash = [
            PIPELINES["md"].config_path,
            PIPELINES["atom_displacement"].config_path,
            PIPELINES["md"].root / "dataset" / "RUN.fdf",
            PIPELINES["atom_displacement"].root / "base" / "RUN.fdf",
            *sorted((PIPELINES["md"].root / "dataset").glob("*.psf")),
            *sorted((PIPELINES["atom_displacement"].root / "base").glob("*.psf")),
            *sorted((PIPELINES["atom_displacement"].root / "relaxed").glob("*.ion.xml")),
        ]
        md_basis_files = sorted((PIPELINES["md"].root / "dataset" / "MD_steps" / "basis").glob("*.ion.xml"))
        atom_basis_files = (
            sorted((PIPELINES["atom_displacement"].root / "dataset" / "FC_steps" / "basis").glob("*.ion.xml"))
            or sorted((PIPELINES["atom_displacement"].root / "dataset" / "AtDis_steps" / "basis").glob("*.ion.xml"))
            or sorted((PIPELINES["atom_displacement"].root / "relaxed").glob("*.ion.xml"))
        )
        md_pseudo_files = sorted((PIPELINES["md"].root / "dataset").glob("*.psf"))
        atom_pseudo_files = sorted((PIPELINES["atom_displacement"].root / "base").glob("*.psf"))
        md_basis_content_hash = files_content_digest(md_basis_files)
        atom_basis_content_hash = files_content_digest(atom_basis_files)
        md_pseudo_content_hash = files_content_digest(md_pseudo_files)
        atom_pseudo_content_hash = files_content_digest(atom_pseudo_files)
        basis_pseudopotential_warning = ""
        if md_basis_content_hash and atom_basis_content_hash and md_basis_content_hash != atom_basis_content_hash:
            basis_pseudopotential_warning = "MD and AtomDisplacement basis .ion.xml content hashes differ."
        if md_pseudo_content_hash and atom_pseudo_content_hash and md_pseudo_content_hash != atom_pseudo_content_hash:
            suffix = "MD and AtomDisplacement pseudopotential .psf content hashes differ."
            basis_pseudopotential_warning = (
                f"{basis_pseudopotential_warning} | {suffix}" if basis_pseudopotential_warning else suffix
            )
        basis_and_pseudos = [
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for path in files_to_hash
            if path.exists() and path.suffix in {".psf", ".xml", ".fdf"}
        ]
        system_name = (
            md_config.get("md", {}).get("system_name")
            or atom_config.get("structure", {}).get("system_name")
            or "unknown"
        )
        resolved_paths = [
            resolve_pipeline_path(PIPELINES["md"], value)
            for value in (md_config.get("paths", {}) or {}).values()
            if isinstance(value, (str, Path))
        ] + [
            resolve_pipeline_path(PIPELINES["atom_displacement"], value)
            for value in (atom_config.get("paths", {}) or {}).values()
            if isinstance(value, (str, Path))
        ]
        artifact_hashes = {
            "configs": files_digest([PIPELINES["md"].config_path, PIPELINES["atom_displacement"].config_path]),
            "md_pseudopotentials": files_content_digest(md_pseudo_files),
            "atom_displacement_pseudopotentials": files_content_digest(atom_pseudo_files),
            "md_basis": files_content_digest(md_basis_files),
            "atom_displacement_basis": files_content_digest(atom_basis_files),
            "rendered_inputs": file_digest([
                PIPELINES["md"].root / "dataset" / "RUN.fdf",
                PIPELINES["atom_displacement"].root / "base" / "RUN.fdf",
            ]),
        }
        return {
            "experiment_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "dirty_tree_warning": git_dirty_warning(),
            "environment_versions": environment_versions([md_config, atom_config]),
            "artifact_hashes": artifact_hashes,
            "reproducibility_warning": absolute_path_warning(resolved_paths),
            "basis_pseudopotential_warning": basis_pseudopotential_warning,
            "metric_version": METRIC_VERSION,
            "molecule_system_name": system_name,
            "config_hash": files_digest([PIPELINES["md"].config_path, PIPELINES["atom_displacement"].config_path]),
            "siesta_settings_hash": siesta_report["siesta_settings_hash"],
            "md_siesta_settings_hash": siesta_report["md_siesta_settings_hash"],
            "atom_displacement_siesta_settings_hash": siesta_report["atom_displacement_siesta_settings_hash"],
            "shared_siesta_settings_hash": siesta_report["shared_siesta_settings_hash"],
            "siesta_settings_warning": siesta_report["warning"],
            "siesta_settings_mismatches": siesta_report["mismatches"],
            "model_config_hash": model_report["model_config_hash"],
            "md_model_config_hash": model_report["md_model_config_hash"],
            "atom_displacement_model_config_hash": model_report["atom_displacement_model_config_hash"],
            "model_config_warning": model_report["warning"],
            "model_config_mismatches": model_report["mismatches"],
            "basis_hash": files_content_digest([*md_basis_files, *atom_basis_files]),
            "pseudopotential_hash": files_content_digest([*md_pseudo_files, *atom_pseudo_files]),
            "basis_pseudopotential_info": basis_and_pseudos,
            "train_methods": ["md", "atom_displacement"],
            "dataset_sizes": {
                "md": md_sizes,
                "atom_displacement": atom_sizes,
            },
            "atom_displacement_dataset_specs": atom_dataset_specs or [],
            "seeds": [random_seed] if random_seed is not None else [],
            "split_ratios": split_ratios,
            "split_mode": split_mode,
            "minimum_robust_seeds": MINIMUM_ROBUST_SEEDS,
            "test_sets": list(test_sets),
            "training_hyperparameters": {
                "md": md_config.get("training", {}),
                "atom_displacement": atom_config.get("training", {}),
            },
            "selected_metrics": {
                "sparse": True,
                "spectral": True,
                "dos": True,
                "primary_metric": primary_metric,
            },
            "compute_budget_mode": compute_budget_mode,
            "strict_comparison_mode": STRICT_COMPARISON_MODE,
            "output_directories": {
                "experiment_root": str(experiment_root(run_id)),
                "manifest": str(experiment_manifest_path(run_id)),
                "common_tests": str(experiment_root(run_id) / "common_tests"),
                "cross_evaluations": str(experiment_root(run_id) / "cross_evaluations"),
                "summary": str(experiment_root(run_id) / "summary"),
            },
            "timing": {
                "md_siesta_generation_seconds": None,
                "atom_displacement_siesta_generation_seconds": None,
                "dataset_preparation_seconds": None,
                "normalization_seconds": None,
                "training_seconds": {},
                "prediction_seconds": {},
                "evaluation_seconds": {},
                "winner_analysis_seconds": None,
                "total_experiment_seconds": None,
                "total_seconds": None,
                "timing_incomplete_warning": (
                    "Per-phase generation/training/prediction timings are partly unavailable "
                    "for legacy Graph2Mat entrypoints; run manifests keep measured totals."
                ),
            },
            "runs": [],
            "warnings": [],
            "cross_evaluation": {},
        }

    def _write_experiment_manifest(self, manifest: dict[str, Any]) -> None:
        path = experiment_manifest_path(str(manifest["experiment_id"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(path, json_safe(manifest))

    def _set_current(
        self,
        pipeline: str,
        size: int,
        *,
        dataset_label: str | None = None,
        started_at: float | None = None,
        eta_seconds: float | None = None,
    ) -> None:
        with self._lock:
            self._current = {
                "pipeline": pipeline,
                "size": size,
                "dataset_label": dataset_label or f"dataset_{size}",
                "started_at": started_at,
                "eta_seconds": eta_seconds,
            }

    def _run(
        self,
        md_sizes: list[int],
        atom_sizes: list[int],
        run_id: str,
        fc_dataset_specs: dict[int, list[dict[str, Any]]] | None = None,
        atom_dataset_specs: list[dict[str, Any]] | None = None,
        split_ratios: dict[str, float] | None = None,
        random_seed: int | None = None,
        split_mode: str = "block",
        test_sets: list[str] | None = None,
        primary_metric: str = DEFAULT_PRIMARY_METRIC,
        compute_budget_mode: str = "both",
    ) -> None:
        original_configs = {
            key: spec.config_path.read_text(encoding="utf-8")
            for key, spec in PIPELINES.items()
        }
        split_ratios = split_ratios or dict(DEFAULT_SPLIT_RATIOS)
        manifest = self._initial_experiment_manifest(
            run_id,
            md_sizes,
            atom_sizes,
            split_ratios,
            random_seed,
            split_mode,
            atom_dataset_specs,
            test_sets or list(DEFAULT_COMMON_TEST_SETS),
            primary_metric,
            compute_budget_mode,
        )
        experiment_root(run_id).mkdir(parents=True, exist_ok=True)
        if manifest.get("siesta_settings_warning"):
            manifest.setdefault("warnings", []).append(str(manifest["siesta_settings_warning"]))
        if manifest.get("model_config_warning"):
            manifest.setdefault("warnings", []).append(str(manifest["model_config_warning"]))
        if manifest.get("basis_pseudopotential_warning"):
            manifest.setdefault("warnings", []).append(str(manifest["basis_pseudopotential_warning"]))
        if manifest.get("reproducibility_warning"):
            manifest.setdefault("warnings", []).append(str(manifest["reproducibility_warning"]))
        self._write_experiment_manifest(manifest)
        returncode = 0
        try:
            self._append(f"[UI] Comparacion {run_id} iniciada.\n")
            self._append(f"[UI] MD sizes: {md_sizes}\n")
            self._append(f"[UI] AtomDisplacement sizes: {atom_sizes}\n")
            self._append(
                "[UI] Split ratios: "
                f"{split_ratios['train']} train, {split_ratios['validation']} validation, "
                f"{split_ratios['test']} test.\n"
            )
            self._append(f"[UI] Split mode MD: {split_mode}\n")
            self._append(f"[UI] Common test sets: {', '.join(test_sets or DEFAULT_COMMON_TEST_SETS)}\n")
            self._append(f"[UI] Primary metric: {primary_metric}; compute mode: {compute_budget_mode}\n")
            if manifest.get("siesta_settings_warning"):
                self._append(f"[WARN] {manifest['siesta_settings_warning']}\n")
            if manifest.get("model_config_warning"):
                self._append(f"[WARN] {manifest['model_config_warning']}\n")
            if STRICT_COMPARISON_MODE and manifest.get("siesta_settings_warning"):
                raise RuntimeError(
                    "Strict comparison aborted: MD y AtomDisplacement tienen settings SIESTA distintas. "
                    "Revisa experiment_manifest.yaml: siesta_settings_mismatches."
                )
            if STRICT_COMPARISON_MODE and manifest.get("model_config_warning"):
                raise RuntimeError(
                    "Strict comparison aborted: MD y AtomDisplacement tienen hiperparametros Graph2Mat distintos. "
                    "Revisa experiment_manifest.yaml: model_config_mismatches."
                )
            if atom_dataset_specs:
                self._append(
                    "[UI] AtomDisplacement FC plan: "
                    + "; ".join(
                        f"{spec['label']}: "
                        + ", ".join(
                            f"{entry['value']} -> {entry['n_structures']}"
                            for entry in spec["displacements"]
                        )
                        for spec in atom_dataset_specs
                    )
                    + "\n"
                )
            elif fc_dataset_specs:
                self._append(
                    "[UI] AtomDisplacement FC plan: "
                    + "; ".join(
                        f"dataset_{size}: "
                        + ", ".join(
                            f"{entry['value']} -> {entry['n_structures']}"
                            for entry in entries
                        )
                        for size, entries in fc_dataset_specs.items()
                    )
                    + "\n"
                )
            self._append(f"[UI] Workspaces: {WORKSPACES_ROOT / run_id}\n")
            self._append(f"[UI] Results root: {RESULTS_ROOT}\n")
            self._append(
                "[UI] ETA: el primer dataset de cada pipeline no tiene historico; "
                "los siguientes usan segundos/estructura de runs ya completados.\n"
            )
            previous_by_method: dict[str, dict[str, Any]] = {}
            for size in md_sizes:
                self._ensure_not_stopped()
                self._restore_original_config("md", original_configs)
                result = self._run_one("md", size, run_id, split_ratios=split_ratios, split_mode=split_mode)
                self._annotate_nested_subset(result, previous_by_method.get("md"))
                previous_by_method["md"] = result
                with self._lock:
                    self._results.append(result)
                manifest["runs"].append(result)
                self._write_experiment_manifest(manifest)
            atom_runs = atom_dataset_specs or [
                {
                    "label": f"dataset_{size}",
                    "size": size,
                    "displacements": fc_dataset_specs.get(size) if fc_dataset_specs else None,
                }
                for size in atom_sizes
            ]
            for atom_spec in atom_runs:
                self._ensure_not_stopped()
                self._restore_original_config("atom_displacement", original_configs)
                size = int(atom_spec["size"])
                result = self._run_one(
                    "atom_displacement",
                    size,
                    run_id,
                    dataset_label=str(atom_spec["label"]),
                    fc_displacements=atom_spec.get("displacements"),
                    split_ratios=split_ratios,
                    random_seed=random_seed,
                    split_mode=split_mode,
                )
                self._annotate_nested_subset(result, previous_by_method.get("atom_displacement"))
                previous_by_method["atom_displacement"] = result
                with self._lock:
                    self._results.append(result)
                manifest["runs"].append(result)
                self._write_experiment_manifest(manifest)
            manifest["cross_evaluation"] = self._run_cross_evaluation(run_id, manifest)
            self._write_experiment_manifest(manifest)
            self._append("\n[UI] Comparacion experimental finalizada correctamente.\n")
        except Exception as exc:
            returncode = 1
            self._append(f"\n[ERROR] {exc}\n")
            manifest.setdefault("warnings", []).append(str(exc))
            self._write_experiment_manifest(manifest)
        finally:
            for key, raw in original_configs.items():
                PIPELINES[key].config_path.write_text(raw, encoding="utf-8")
            with self._lock:
                self._process = None
                self._current = None
                self._returncode = returncode
                self._finished_at = time.time()
            manifest["timing"]["total_seconds"] = (
                self._finished_at - self._started_at
                if self._finished_at is not None and self._started_at is not None
                else None
            )
            manifest["timing"]["total_experiment_seconds"] = manifest["timing"]["total_seconds"]
            self._write_experiment_manifest(manifest)
            self._append("[UI] Configuraciones originales restauradas.\n")

    def _restore_original_config(self, key: str, original_configs: dict[str, str]) -> None:
        PIPELINES[key].config_path.write_text(original_configs[key], encoding="utf-8")

    def _ensure_not_stopped(self) -> None:
        with self._lock:
            if self._stop_requested:
                raise RuntimeError("Experimento detenido por el usuario.")

    def _annotate_nested_subset(self, result: dict[str, Any], parent: dict[str, Any] | None) -> None:
        result["parent_dataset_size"] = parent.get("dataset_size") if parent else None
        result["parent_dataset_hash"] = parent.get("dataset_sample_hash") if parent else ""
        result["nested_subset_hash"] = result.get("dataset_sample_hash", "")
        result["nested_subset_of_parent"] = True
        result["nested_subset_warning"] = ""
        if parent:
            parent_samples = set(str(item) for item in parent.get("dataset_sample_ids", []))
            current_samples = set(str(item) for item in result.get("dataset_sample_ids", []))
            if not parent_samples.issubset(current_samples):
                result["nested_subset_of_parent"] = False
                result["nested_subset_warning"] = (
                    "Dataset-size sweep is not nested: previous dataset samples "
                    "are not all present in the current dataset."
                )
        manifest_path = Path(str(result.get("result_dir", ""))) / "manifest.json"
        if manifest_path.exists():
            manifest_path.write_text(
                json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    def _run_one(
        self,
        key: str,
        size: int,
        run_id: str,
        dataset_label: str | None = None,
        fc_displacements: list[dict[str, Any]] | None = None,
        split_ratios: dict[str, float] | None = None,
        random_seed: int | None = None,
        split_mode: str = "block",
    ) -> dict[str, Any]:
        spec = PIPELINES[key]
        dataset_label = dataset_label or f"dataset_{size}"
        started_at = time.time()
        eta_seconds = self._estimated_seconds(key, size)
        self._set_current(
            key,
            size,
            dataset_label=dataset_label,
            started_at=started_at,
            eta_seconds=eta_seconds,
        )
        self._append(f"\n[UI] === {spec.label} {dataset_label} ===\n")
        self._append(f"[UI] ETA inicial: {format_duration(eta_seconds)}\n")
        config = load_config(spec.config_path)
        workspace = WORKSPACES_ROOT / run_id / key / dataset_label
        result_group = "results_md" if key == "md" else "results_atomdisp"
        result_dir = RESULTS_ROOT / result_group / dataset_label / f"run_{run_id}"
        workspace.mkdir(parents=True, exist_ok=True)
        self._append(f"[UI] Workspace: {workspace}\n")
        self._append(f"[UI] Result dir previsto: {result_dir}\n")
        self._append(f"[UI] Config temporal: {spec.config_path}\n")
        with self._lock:
            log_start = len(self._logs)
        prepare_metadata: dict[str, Any] = {}
        if key == "md":
            self._prepare_md_config(config, workspace, size, split_ratios, split_mode=split_mode)
            original_steps = list(config.get("pipeline", {}).get("steps", []))
            config.setdefault("pipeline", {})["steps"] = ["generate_md_dataset"]
            write_yaml(spec.config_path, config)
            self._append("[UI] Config temporal MD escrita para generar dataset y manifests.\n")
            generation_returncode = self._run_pipeline_process(
                spec,
                key=key,
                size=size,
                started_at=started_at,
            )
            if generation_returncode != 0:
                raise RuntimeError(
                    f"{spec.label} dataset_{size} fallo generando dataset con codigo "
                    f"{generation_returncode}."
                )
            self._validate_split_manifests(key, config, size)
            config["pipeline"]["steps"] = [
                step
                for step in original_steps
                if step in {"run_md_training", "run_md_testing", "run_md_prediction"}
            ] or ["run_md_training", "run_md_testing", "run_md_prediction"]
            write_yaml(spec.config_path, config)
            self._append("[UI] Config temporal MD escrita para entrenar/evaluar tras validacion.\n")
            returncode = self._run_pipeline_process(spec, key=key, size=size, started_at=started_at)
        else:
            self._prepare_atom_generation_config(
                config,
                workspace,
                size,
                fc_displacements,
                random_seed=random_seed,
            )
            write_yaml(spec.config_path, config)
            self._append(
                "[UI] Config temporal FC escrita; SIESTA generara AtDis_steps en el workspace.\n"
            )
            generation_returncode = self._run_pipeline_process(
                spec,
                key=key,
                size=size,
                started_at=started_at,
            )
            if generation_returncode != 0:
                raise RuntimeError(
                    f"{spec.label} dataset_{size} fallo generando FC con codigo "
                    f"{generation_returncode}."
                )
            prepare_metadata = self._prepare_atom_config(config, workspace, size, split_ratios)
            write_yaml(spec.config_path, config)
            self._append("[UI] Config temporal de entrenamiento escrita tras FC.\n")
            if prepare_metadata.get("test_needs_siesta"):
                self._run_atom_test_single_points(spec, config, Path(prepare_metadata["test_samples_dir"]))
                self._refresh_atom_split_manifests(config)
                write_yaml(spec.config_path, config)
                self._append("[UI] Config de entrenamiento restaurada tras SIESTA del test.\n")
            self._validate_split_manifests(key, config, size)
            returncode = self._run_pipeline_process(spec, key=key, size=size, started_at=started_at)
        with self._lock:
            run_log = "".join(self._logs[log_start:])
        pipeline_elapsed = time.time() - started_at
        archive = self._archive_outputs(
            key,
            size,
            run_id,
            workspace,
            config,
            returncode,
            run_log,
            prepare_metadata,
            dataset_label=dataset_label,
            pipeline_elapsed_seconds=pipeline_elapsed,
        )
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

    def _prepare_md_config(
        self,
        config: dict[str, Any],
        workspace: Path,
        size: int,
        split_ratios: dict[str, float] | None = None,
        split_mode: str = "block",
    ) -> None:
        dataset_dir = workspace / "dataset"
        pseudo_count = copy_pseudopotentials(PIPELINES["md"].root / "dataset", dataset_dir)
        ratios = split_ratios or split_ratios_from_config(config)
        counts = validate_split_sizes(size, ratios, label=f"MD dataset_{size}")
        config["paths"]["dataset_dir"] = str(dataset_dir)
        config["paths"]["training_dir"] = str(workspace / "training")
        config["md"]["steps"] = size
        config["splits"] = {
            "enabled": True,
            "strategy": split_mode,
            "train": counts["train"],
            "validation": counts["validation"],
            "test": counts["test"],
        }
        config["training"]["data"]["train_runs"] = "../dataset/splits/train/*/RUN.fdf"
        config["testing"]["test_runs"] = "../dataset/splits/test/*/RUN.fdf"
        config["testing"].setdefault("callbacks", {})
        config["testing"]["callbacks"]["plot_matrix_error"] = False
        config["testing"]["callbacks"]["show_plot"] = False
        config["testing"]["callbacks"]["samplewise_metrics_logger"] = False
        config["prediction"]["predict_structs"] = "../dataset/splits/test/*/RUN.fdf"
        config["checkpoint"]["path"] = None
        self._append(f"[UI] MD dataset_dir: {dataset_dir}\n")
        self._append(f"[UI] MD training_dir: {workspace / 'training'}\n")
        self._append(f"[UI] MD pseudopotenciales copiados: {pseudo_count}\n")
        self._append(f"[UI] MD steps configurados: {size}\n")
        self._append(
            "[UI] MD split: "
            f"{counts['train']} train, {counts['test']} test, "
            f"{counts['validation']} validation; ratios "
            f"{ratios['train']}/{ratios['validation']}/{ratios['test']}.\n"
        )
        self._append(f"[UI] MD train_runs: {config['training']['data']['train_runs']}\n")
        self._append(f"[UI] MD test_runs: {config['testing']['test_runs']}\n")

    def _prepare_atom_generation_config(
        self,
        config: dict[str, Any],
        workspace: Path,
        size: int,
        fc_displacements: list[dict[str, Any]] | None = None,
        random_seed: int | None = None,
    ) -> None:
        dataset_dir = workspace / "dataset"
        training_dir = workspace / "training"
        base_dir = workspace / "base"
        relaxed_dir = workspace / "relaxed"
        pseudo_count = copy_pseudopotentials(PIPELINES["atom_displacement"].root / "base", base_dir)
        relaxed_counts = copy_relaxed_basis(PIPELINES["atom_displacement"].root / "relaxed", relaxed_dir)

        config["paths"]["base_dir"] = str(base_dir)
        config["paths"]["relaxed_dir"] = str(relaxed_dir)
        config["paths"]["dataset_dir"] = str(dataset_dir)
        config["paths"]["samples_dir"] = str(dataset_dir / "samples")
        config["paths"]["collected_dir"] = str(dataset_dir / "collected")
        config["paths"]["training_dir"] = str(training_dir)
        config["single_points"]["limit"] = None
        config["single_points"]["rerun"] = False
        force_constants = config["structure"]["force_constants"]
        limit = atom_fc_sample_limit(config)
        if limit is None:
            raise RuntimeError("AtomDisplacement: FC no esta habilitado en la configuracion.")
        if fc_displacements:
            requested_total = sum(int(entry["n_structures"]) for entry in fc_displacements)
            if requested_total != size:
                raise RuntimeError(
                    "AtomDisplacement: la suma de estructuras FC "
                    f"({requested_total}) no coincide con dataset_{size}."
                )
            force_constants["displacements"] = list(fc_displacements)
        else:
            displacement_entries = atom_fc_displacement_entries(config)
            counts = distribute_fc_counts(size, len(displacement_entries), limit)
            force_constants["displacements"] = [
                {**entry, "n_structures": count}
                for entry, count in zip(displacement_entries, counts)
                if count > 0
            ]
        if random_seed is not None:
            force_constants["random_seed"] = int(random_seed)
            force_constants.setdefault("subsampling", {})["seed"] = int(random_seed)
        force_constants["target_count"] = None
        config["structure"]["force_constants"]["allow_missing_matrix"] = False
        config["pipeline"]["steps"] = [
            "render_inputs",
            "generate_atom_displacement_dataset",
            "run_single_points",
            "normalize_fc_steps",
            "run_single_points",
            "collect_atom_displacement_dataset",
        ]
        self._append(f"[UI] AtomDisplacement FC dataset_dir: {dataset_dir}\n")
        self._append(f"[UI] AtomDisplacement FC base_dir: {base_dir}\n")
        self._append(f"[UI] AtomDisplacement FC relaxed_dir: {relaxed_dir}\n")
        self._append(f"[UI] AtomDisplacement training_dir: {training_dir}\n")
        self._append(f"[UI] AtomDisplacement pseudopotenciales copiados: {pseudo_count}\n")
        self._append(
            "[UI] AtomDisplacement relaxed copiado: "
            f"{relaxed_counts['basis_files']} basis .ion.xml, {relaxed_counts['xv_files']} XV.\n"
        )
        self._append(
            "[UI] AtomDisplacement generara FC raw y normalizara FC_steps antes del split.\n"
        )
        self._append(
            "[UI] AtomDisplacement FC magnitudes: "
            + ", ".join(
                f"{entry.get('value')} -> {entry.get('n_structures')}"
                for entry in force_constants["displacements"]
            )
            + f" (limite {limit} por magnitud).\n"
        )
        self._append(f"[UI] AtomDisplacement FC steps: {', '.join(config['pipeline']['steps'])}\n")

    def _prepare_atom_config(
        self,
        config: dict[str, Any],
        workspace: Path,
        size: int,
        split_ratios: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        dataset_dir = workspace / "dataset"
        training_dir = workspace / "training"
        relaxed_dir = workspace / "relaxed"
        train_samples_dir = dataset_dir / "train_samples"
        validation_samples_dir = dataset_dir / "validation_samples"
        test_samples_dir = dataset_dir / "test_samples"
        basis_dir = dataset_dir / "basis"
        source_samples_dir = dataset_dir / "FC_steps"
        if not source_samples_dir.exists():
            source_samples_dir = atom_source_samples_dir(PIPELINES["atom_displacement"], config)
        basis_count = copy_basis_files(source_samples_dir, basis_dir)
        all_samples = generated_atom_samples(source_samples_dir)
        completed_samples = completed_atom_samples(source_samples_dir)
        if not all_samples:
            raise RuntimeError(
                "AtomDisplacement: SIESTA FC no genero muestras normalizadas en FC_steps. "
                f"Revisa {dataset_dir / 'run_summary.json'}."
            )
        if len(all_samples) < size:
            raise RuntimeError(
                "AtomDisplacement: SIESTA FC genero menos estructuras de las pedidas "
                f"({len(all_samples)} < {size}). Revisa FC.First/FC.Last y {dataset_dir / 'run_summary.json'}."
            )
        ratios = split_ratios or split_ratios_from_config(config)
        counts = validate_split_sizes(size, ratios, label=f"AtomDisplacement dataset_{size}")
        train_needed = counts["train"]
        validation_needed = counts["validation"]
        test_needed = counts["test"]
        selected_samples = select_spread(all_samples, size)
        split_samples = split_grouped_exact(selected_samples, counts)
        train_samples = split_samples["train"]
        validation_samples = split_samples["validation"]
        test_samples = split_samples["test"]
        incomplete_train = [
            path for path in train_samples if not is_completed_atom_sample(path)
        ]
        if incomplete_train:
            raise RuntimeError(
                "AtomDisplacement: no puedo entrenar ese tamano sin recalcular SIESTA "
                f"en train. Para dataset_{size}, el split configurado necesita "
                f"{train_needed} muestras de train con Hamiltoniano SIESTA; faltan "
                f"{len(incomplete_train)} en el split espaciado."
            )
        if len(test_samples) < test_needed or len(validation_samples) < validation_needed:
            raise RuntimeError(
                "AtomDisplacement: no hay suficientes muestras para completar el split "
                f"configurado ({len(train_samples)} train, {len(test_samples)} test, "
                f"{len(validation_samples)} validation)."
            )
        copy_sample_dirs(train_samples, train_samples_dir)
        copy_sample_dirs(validation_samples, validation_samples_dir)
        copy_sample_dirs(test_samples, test_samples_dir)
        split_manifest_paths = write_atom_split_manifests(
            dataset_dir,
            {
                "train": train_samples,
                "validation": validation_samples,
                "test": test_samples,
            },
        )
        test_needs_siesta = any(
            not is_completed_atom_sample(path)
            for path in test_samples_dir.iterdir()
            if path.is_dir() and (path / "RUN.fdf").exists()
        )
        config["paths"]["dataset_dir"] = str(dataset_dir)
        config["paths"]["samples_dir"] = str(train_samples_dir)
        config["paths"]["validation_samples_dir"] = str(validation_samples_dir)
        config["paths"]["test_samples_dir"] = str(test_samples_dir)
        config["paths"]["collected_dir"] = str(dataset_dir / "collected")
        config["paths"]["training_dir"] = str(training_dir)
        config["paths"]["relaxed_dir"] = str(relaxed_dir)
        config["single_points"]["rerun"] = False
        basis_files_pattern = "../dataset/basis/*.ion.xml" if basis_count else "../relaxed/*.ion.xml"
        config["training"]["data"]["basis_files"] = basis_files_pattern
        config["training"]["data"].pop("n_matrix_components", None)
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
        self._append(f"[UI] AtomDisplacement basis dataset copiados: {basis_count}\n")
        self._append(
            f"[UI] AtomDisplacement FC raw disponibles: {len(all_samples)} generadas, "
            f"{len(completed_samples)} con referencia SIESTA; "
            f"seleccionadas para dataset_{size}: {len(selected_samples)}.\n"
        )
        self._append(
            "[UI] AtomDisplacement split: "
            f"{len(train_samples)} train, {len(test_samples)} test, "
            f"{len(validation_samples)} validation; ratios "
            f"{ratios['train']}/{ratios['validation']}/{ratios['test']}.\n"
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
        return {
            "test_samples_dir": str(test_samples_dir),
            "test_needs_siesta": test_needs_siesta,
            "requested_size": size,
            "effective_size": size,
            "generated_samples": len(selected_samples),
            "completed_samples": sum(1 for path in selected_samples if is_completed_atom_sample(path)),
            "fc_generated_samples": len(all_samples),
            "fc_completed_samples": len(completed_samples),
            "split_manifest_paths": {key: str(value) for key, value in split_manifest_paths.items()},
            "seed": (config.get("structure", {}).get("force_constants", {}) or {}).get("random_seed"),
        }

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
        master_fd: int | None = None
        if pty is not None:
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
        else:
            process = subprocess.Popen(
                command,
                cwd=spec.root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
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
            if master_fd is not None:
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

    def _refresh_atom_split_manifests(self, config: dict[str, Any]) -> None:
        dataset_dir = Path(config["paths"]["dataset_dir"])
        split_samples = {
            "train": sorted(path for path in (dataset_dir / "train_samples").iterdir() if path.is_dir()),
            "validation": sorted(path for path in (dataset_dir / "validation_samples").iterdir() if path.is_dir()),
            "test": sorted(path for path in (dataset_dir / "test_samples").iterdir() if path.is_dir()),
        }
        write_atom_split_manifests(dataset_dir, split_samples)
        self._append("[UI] AtomDisplacement split manifests refrescados tras SIESTA de test.\n")

    def _validate_split_manifests(
        self,
        key: str,
        config: dict[str, Any],
        size: int,
    ) -> dict[str, Any]:
        dataset_dir = Path(config["paths"]["dataset_dir"])
        split_root = dataset_dir / "splits"
        script = COMPARISON_ROOT / "scripts" / "validate_sample_bundle.py"
        summaries: dict[str, Any] = {}
        for split_name in ("train", "validation", "test"):
            manifest_path = split_root / f"{split_name}_manifest.csv"
            if not manifest_path.exists():
                raise RuntimeError(f"{PIPELINES[key].label}: falta split manifest: {manifest_path}")
            rows = read_csv_rows(manifest_path)
            min_valid = len(rows)
            output_dir = dataset_dir / "validation" / split_name
            command = [
                sys.executable,
                str(script),
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
                "--min-valid",
                str(min_valid),
            ]
            self._append(f"[UI] Validando {PIPELINES[key].label} {split_name}: {' '.join(command)}\n")
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
            summary_path = output_dir / "validation_summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    summary = {"ok": False, "error": str(exc)}
            else:
                summary = {"ok": False, "error": "validation_summary_missing"}
            summary["returncode"] = result.returncode
            summaries[split_name] = summary
            valid_manifest = output_dir / "valid_samples.csv"
            if valid_manifest.exists():
                shutil.copy2(valid_manifest, split_root / f"{split_name}_valid_manifest.csv")
            if result.returncode != 0 or not summary.get("ok"):
                raise RuntimeError(
                    f"{PIPELINES[key].label}: validacion {split_name} fallo; "
                    f"revisa {summary_path}."
                )
        self._append(f"[UI] Validacion de muestras completada para {PIPELINES[key].label} dataset_{size}.\n")
        return summaries

    def _archive_outputs(
        self,
        key: str,
        size: int,
        run_id: str,
        workspace: Path,
        config: dict[str, Any],
        returncode: int,
        run_log: str,
        prepare_metadata: dict[str, Any] | None = None,
        dataset_label: str | None = None,
        pipeline_elapsed_seconds: float | None = None,
    ) -> dict[str, Any]:
        prepare_metadata = prepare_metadata or {}
        dataset_label = dataset_label or f"dataset_{size}"
        result_group = "results_md" if key == "md" else "results_atomdisp"
        result_dir = RESULTS_ROOT / result_group / dataset_label / f"run_{run_id}"
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
            copy_basis_files(dataset_dir / "MD_steps", result_dir / "basis")
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
            copy_basis_files(samples_dir, result_dir / "basis")
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
        for manifest_file in sorted((dataset_dir / "splits").glob("*_manifest.csv")):
            copy_if_exists(manifest_file, result_dir / "splits" / manifest_file.name)
        for validation_file in sorted((dataset_dir / "validation").glob("*/*.csv")):
            copy_if_exists(validation_file, result_dir / "validation" / validation_file.parent.name / validation_file.name)
        evaluation_metrics = self._evaluate_hamiltonian_metrics(key, config, result_dir)
        timing_breakdown = {
            "md_siesta_generation_seconds": None,
            "atomdisp_siesta_generation_seconds": None,
            "dataset_preparation_seconds": None,
            "normalization_seconds": None,
            "training_seconds": None,
            "prediction_seconds": None,
            "evaluation_seconds": evaluation_metrics.get("evaluation_time_seconds"),
            "winner_analysis_seconds": None,
            "total_experiment_seconds": pipeline_elapsed_seconds,
            "timing_incomplete_warning": (
                "This per-run manifest contains measured total time and Hamiltonian evaluation time; "
                "legacy training/testing scripts do not expose separate training and prediction timings."
            ),
        }
        if key == "md":
            timing_breakdown["md_siesta_generation_seconds"] = prepare_metadata.get("generation_seconds")
        else:
            timing_breakdown["atomdisp_siesta_generation_seconds"] = prepare_metadata.get("generation_seconds")
            timing_breakdown["dataset_preparation_seconds"] = prepare_metadata.get("dataset_preparation_seconds")
        (result_dir / "timing_breakdown.json").write_text(
            json.dumps(json_safe(timing_breakdown), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        effective_size = int(prepare_metadata.get("effective_size", size))
        checkpoint_path = find_latest_checkpoint(training_dir, config)
        signed_checkpoint = checkpoint_metadata(checkpoint_path, training_dir)
        checkpoint_warning = checkpoint_selection_warning(training_dir, checkpoint_path)
        checkpoint_manifest_path = write_checkpoint_manifest(training_dir, signed_checkpoint, checkpoint_warning)
        source_pseudos = (
            sorted((PIPELINES["md"].root / "dataset").glob("*.psf"))
            if key == "md"
            else sorted((PIPELINES["atom_displacement"].root / "base").glob("*.psf"))
        )
        run_artifact_hashes = {
            "basis": files_content_digest(sorted((result_dir / "basis").glob("*.ion.xml"))),
            "pseudopotentials": files_content_digest(source_pseudos),
            "rendered_run_fdf": files_content_digest(sorted((result_dir / "structures").glob("*/RUN.fdf"))),
            "pipeline_config": files_content_digest([result_dir / "pipeline_config.yaml"]),
            "checkpoint_manifest": files_content_digest([checkpoint_manifest_path]),
        }
        dataset_sample_ids: list[str] = []
        for split_manifest in sorted((result_dir / "splits").glob("*_manifest.csv")):
            for row in read_csv_rows(split_manifest):
                sample_id = str(row.get("sample_id") or row.get("sample") or "")
                if sample_id:
                    dataset_sample_ids.append(sample_id)
        manifest = {
            "pipeline": key,
            "dataset_label": dataset_label,
            "dataset_size": size,
            "requested_dataset_size": size,
            "effective_dataset_size": effective_size,
            "run_id": run_id,
            "returncode": returncode,
            "pipeline_elapsed_seconds": pipeline_elapsed_seconds,
            "workspace": str(workspace),
            "dataset_dir": str(dataset_dir),
            "training_dir": str(training_dir),
            "result_dir": str(result_dir),
            "model_checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "model_checkpoint_metadata": signed_checkpoint,
            "model_checkpoint_sha256": signed_checkpoint.get("sha256"),
            "checkpoint_manifest": str(checkpoint_manifest_path),
            "checkpoint_selection_warning": checkpoint_warning,
            "artifact_hashes": run_artifact_hashes,
            "dataset_sample_ids": sorted(set(dataset_sample_ids)),
            "dataset_sample_hash": sample_set_hash(dataset_sample_ids),
            "seed": prepare_metadata.get("seed"),
            "predicted_hamiltonians": prediction_count,
            "siesta_hamiltonians": reference_count,
            "timing_breakdown": timing_breakdown,
            "generated_samples": prepare_metadata.get("generated_samples"),
            "completed_samples": prepare_metadata.get("completed_samples"),
            "fc_generated_samples": prepare_metadata.get("fc_generated_samples"),
            "fc_completed_samples": prepare_metadata.get("fc_completed_samples"),
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
        started_at = time.time()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.time() - started_at
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
        summary["evaluation_time_seconds"] = elapsed
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
        if summary.get("structural_metrics_error"):
            raise RuntimeError(
                "Evaluacion Hamiltoniana sin basis orbital valida: "
                f"{summary.get('structural_basis_error') or summary.get('structural_metrics_unavailable')}"
            )
        return summary

    def _run_local_script(
        self,
        command: list[str],
        *,
        label: str,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self._append(f"[UI] {label}: {' '.join(command)}\n")
        result = subprocess.run(
            command,
            cwd=cwd or REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            self._append(result.stdout.strip() + "\n")
        if result.stderr.strip():
            self._append(result.stderr.strip() + "\n")
        return result

    def _split_manifest_for_result(self, result: dict[str, Any], split_name: str) -> Path:
        result_dir = Path(str(result["result_dir"]))
        split_root = result_dir / "splits"
        for name in (f"{split_name}_valid_manifest.csv", f"{split_name}_manifest.csv"):
            candidate = split_root / name
            if candidate.exists():
                return candidate
        raise RuntimeError(f"Missing {split_name} manifest for {result.get('pipeline')}: {result_dir}")

    def _basis_files_glob_for_result(self, result: dict[str, Any]) -> str:
        config = load_config(Path(str(result["result_dir"])) / "pipeline_config.yaml")
        dataset_dir = Path(config["paths"]["dataset_dir"])
        candidates = [
            dataset_dir / "basis",
            dataset_dir / "MD_steps" / "basis",
            Path(str(config["paths"].get("relaxed_dir", ""))),
        ]
        for directory in candidates:
            if directory.exists() and list(directory.glob("*.ion.xml")):
                return str(directory / "*.ion.xml")
        configured = (
            config.get("prediction", {}).get("data", {}).get("basis_files")
            or config.get("training", {}).get("data", {}).get("basis_files")
        )
        if configured:
            return str(configured)
        raise RuntimeError(f"No basis .ion.xml files found for {result.get('pipeline')} result.")

    def _python_for_result(self, result: dict[str, Any], config: dict[str, Any]) -> str:
        pipeline = str(result.get("pipeline"))
        spec = PIPELINES.get(pipeline)
        venv_activate = config.get("paths", {}).get("venv_activate")
        if spec is not None and venv_activate:
            python = resolve_pipeline_path(spec, str(venv_activate)).parent / "python"
            if python.exists():
                return str(python)
        return sys.executable

    def _n_matrix_components_for_result(self, config: dict[str, Any]) -> int | None:
        for section in ("prediction", "testing", "training"):
            data = config.get(section, {}).get("data", {})
            value = data.get("n_matrix_components")
            if value not in (None, ""):
                return int(value)
        return None

    def _prepare_cross_result_dir(
        self,
        cross_result_dir: Path,
        prediction_dir: Path,
        test_manifest: Path,
        basis_files_glob: str | None = None,
    ) -> dict[str, int]:
        if cross_result_dir.exists():
            shutil.rmtree(cross_result_dir)
        cross_result_dir.mkdir(parents=True, exist_ok=True)
        prediction_root = prediction_dir / "predicted_hamiltonians"
        if not prediction_root.exists():
            raise RuntimeError(f"Prediction output missing: {prediction_root}")
        shutil.copytree(prediction_root, cross_result_dir / "predicted_hamiltonians")
        copy_if_exists(prediction_dir / "prediction_summary.json", cross_result_dir / "prediction_summary.json")
        copy_if_exists(prediction_dir / "prediction_manifest.csv", cross_result_dir / "prediction_manifest.csv")
        copy_if_exists(test_manifest.parent / "frozen_test_manifest.json", cross_result_dir / "frozen_test_manifest.json")
        if basis_files_glob:
            basis_dir = cross_result_dir / "basis"
            basis_dir.mkdir(parents=True, exist_ok=True)
            for src in sorted(Path(path) for path in glob.glob(basis_files_glob)):
                if src.is_file():
                    shutil.copy2(src, basis_dir / src.name)

        rows = read_csv_rows(test_manifest)
        references = 0
        structures = 0
        for row in rows:
            sample_id = str(row.get("sample_id") or row.get("sample") or "")
            if not sample_id:
                continue
            hamiltonian_path = Path(str(row.get("hamiltonian_path") or ""))
            if hamiltonian_path.exists() and hamiltonian_path.is_file():
                dst = cross_result_dir / "siesta_hamiltonians" / sample_id / hamiltonian_path.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(hamiltonian_path, dst)
                references += 1
            structure_path = Path(str(row.get("structure_path") or ""))
            if structure_path.exists() and structure_path.is_file():
                dst = cross_result_dir / "structures" / sample_id / "RUN.fdf"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(structure_path, dst)
                structures += 1
            metadata_path = Path(str(row.get("metadata_path") or ""))
            if metadata_path.exists() and metadata_path.is_file():
                dst = cross_result_dir / "structures" / sample_id / metadata_path.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(metadata_path, dst)
        return {"references": references, "structures": structures}

    def _common_test_pair_id(self, md_result: dict[str, Any], atom_result: dict[str, Any]) -> str:
        md_label = str(md_result.get("dataset_label", f"dataset_{md_result.get('dataset_size')}"))
        atom_label = str(atom_result.get("dataset_label", f"dataset_{atom_result.get('dataset_size')}"))
        return f"{md_label}__{atom_label}".replace("/", "_").replace("\\", "_")

    def _run_cross_evaluation(self, run_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
        runs = [run for run in manifest.get("runs", []) if run.get("returncode") == 0]
        md_runs = [run for run in runs if run.get("pipeline") == "md"]
        atom_runs = [run for run in runs if run.get("pipeline") == "atom_displacement"]
        summary: dict[str, Any] = {
            "ok": False,
            "warnings": [],
            "common_tests": [],
            "cross_evaluations": [],
            "outputs": {},
        }
        if not md_runs or not atom_runs:
            summary["warnings"].append("Both MD and AtomDisplacement successful runs are required for cross evaluation.")
            self._append("[WARN] No hay runs exitosos de ambos metodos; se omite cross-evaluation.\n")
            return summary

        common_root = experiment_root(run_id) / "common_tests"
        cross_root = experiment_root(run_id) / "cross_evaluations"
        prediction_root = experiment_root(run_id) / "cross_predictions"
        summary_root = experiment_root(run_id) / "summary"
        test_sets = list(manifest.get("test_sets") or DEFAULT_COMMON_TEST_SETS)
        common_root.mkdir(parents=True, exist_ok=True)
        cross_root.mkdir(parents=True, exist_ok=True)
        prediction_root.mkdir(parents=True, exist_ok=True)

        md_runs = sorted(md_runs, key=lambda run: (int(run.get("dataset_size", 0)), str(run.get("dataset_label", ""))))
        atom_runs = sorted(atom_runs, key=lambda run: (int(run.get("dataset_size", 0)), str(run.get("dataset_label", ""))))
        for md_result in md_runs:
            md_dataset_size = int(md_result.get("dataset_size", 0))
            md_budget = reference_budget_for_run(md_result)
            for atom_result in atom_runs:
                compute_mode = str(manifest.get("compute_budget_mode", "both"))
                if not should_compare_budget_pair(md_result, atom_result, atom_runs, compute_mode):
                    continue
                atom_dataset_size = int(atom_result.get("dataset_size", 0))
                atom_budget = reference_budget_for_run(atom_result)
                ratio = budget_ratio(md_budget, atom_budget)
                mismatch_warning = budget_warning(md_budget, atom_budget)
                pair_id = self._common_test_pair_id(md_result, atom_result)
                pair_common_dir = common_root / pair_id
                build_command = [
                    sys.executable,
                    str(COMPARISON_ROOT / "scripts" / "build_common_tests.py"),
                    "--md-test-manifest",
                    str(self._split_manifest_for_result(md_result, "test")),
                    "--atomdisp-test-manifest",
                    str(self._split_manifest_for_result(atom_result, "test")),
                    "--train-manifest",
                    str(self._split_manifest_for_result(md_result, "train")),
                    "--train-manifest",
                    str(self._split_manifest_for_result(atom_result, "train")),
                    "--output-dir",
                    str(pair_common_dir),
                    "--test-sets",
                    ",".join(test_sets),
                ]
                build_result = self._run_local_script(build_command, label=f"Construyendo common tests {pair_id}")
                if build_result.returncode != 0:
                    raise RuntimeError(f"Common test builder failed for {pair_id}.")
                summary["common_tests"].append(str(pair_common_dir))
                train_manifests = [
                    self._split_manifest_for_result(md_result, "train"),
                    self._split_manifest_for_result(atom_result, "train"),
                ]
                leakage_warnings_by_test_set: dict[str, str] = {}
                leakage_summary_by_test_set: dict[str, str] = {}
                for test_set in test_sets:
                    test_manifest = pair_common_dir / test_set / "test_manifest.csv"
                    if not test_manifest.exists():
                        continue
                    leakage_dir = pair_common_dir / "geometry_leakage" / test_set
                    leakage_command = [
                        sys.executable,
                        str(COMPARISON_ROOT / "scripts" / "check_geometry_leakage.py"),
                        "--train-manifest",
                        str(train_manifests[0]),
                        "--train-manifest",
                        str(train_manifests[1]),
                        "--test-manifest",
                        str(test_manifest),
                        "--output-dir",
                        str(leakage_dir),
                    ]
                    leakage_result = self._run_local_script(
                        leakage_command,
                        label=f"Chequeando leakage geometrico {pair_id} {test_set}",
                    )
                    leakage_summary_path = leakage_dir / "geometry_leakage_summary.json"
                    if leakage_summary_path.exists():
                        leakage_summary_by_test_set[test_set] = str(leakage_summary_path)
                    if leakage_result.returncode != 0:
                        warning = f"Geometry leakage detected for {pair_id} {test_set}; see {leakage_dir}."
                        summary["warnings"].append(warning)
                        leakage_warnings_by_test_set[test_set] = warning
                        if STRICT_COMPARISON_MODE:
                            raise RuntimeError(warning)

                for train_result in (md_result, atom_result):
                    train_method = str(train_result["pipeline"])
                    checkpoint = train_result.get("model_checkpoint")
                    if not checkpoint or not Path(str(checkpoint)).exists():
                        warning = f"Missing checkpoint for {train_method} {train_result.get('dataset_label')}; skipping cross prediction."
                        summary["warnings"].append(warning)
                        self._append(f"[WARN] {warning}\n")
                        continue
                    basis_files = self._basis_files_glob_for_result(train_result)
                    train_config = load_config(Path(str(train_result["result_dir"])) / "pipeline_config.yaml")
                    train_python = self._python_for_result(train_result, train_config)
                    n_matrix_components = self._n_matrix_components_for_result(train_config)
                    for test_set in test_sets:
                        test_manifest = pair_common_dir / test_set / "test_manifest.csv"
                        if not test_manifest.exists():
                            warning = f"Missing common test manifest {test_manifest}; skipping."
                            summary["warnings"].append(warning)
                            self._append(f"[WARN] {warning}\n")
                            continue
                        frozen_manifest_path = test_manifest.parent / "frozen_test_manifest.json"
                        frozen_test_hash = None
                        frozen_test_warning = ""
                        if frozen_manifest_path.exists():
                            frozen_payload = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
                            frozen_test_hash = frozen_payload.get("frozen_test_hash")
                        else:
                            frozen_test_warning = f"Missing frozen test manifest for {pair_id} {test_set}."
                            if STRICT_COMPARISON_MODE:
                                raise RuntimeError(frozen_test_warning)
                        cross_name = f"{pair_id}__{train_method}__on__{test_set}"
                        prediction_dir = prediction_root / cross_name
                        predict_command = [
                            train_python,
                            str(COMPARISON_ROOT / "scripts" / "predict_model_on_dataset.py"),
                            "--checkpoint",
                            str(checkpoint),
                            "--train-method",
                            train_method,
                            "--test-set",
                            test_set,
                            "--test-manifest",
                            str(test_manifest),
                            "--basis-files",
                            basis_files,
                            "--output-dir",
                            str(prediction_dir),
                        ]
                        if n_matrix_components is not None:
                            predict_command.extend(["--n-matrix-components", str(n_matrix_components)])
                        predict_command.append("--patch-graph2mat-basis-loading")
                        predict_start = time.time()
                        predict_result = self._run_local_script(
                            predict_command,
                            label=f"Prediccion cruzada {cross_name}",
                        )
                        prediction_time = time.time() - predict_start
                        if predict_result.returncode != 0:
                            raise RuntimeError(f"Cross prediction failed for {cross_name}.")

                        cross_result_dir = cross_root / cross_name
                        copy_counts = self._prepare_cross_result_dir(
                            cross_result_dir,
                            prediction_dir,
                            test_manifest,
                            basis_files,
                        )
                        evaluation_start = time.time()
                        evaluation_summary = self._evaluate_hamiltonian_metrics(
                            train_method,
                            train_config,
                            cross_result_dir,
                        )
                        evaluation_time = time.time() - evaluation_start
                        cross_manifest = {
                            "experiment_id": run_id,
                            "pair_id": pair_id,
                            "train_method": train_method,
                            "test_set": test_set,
                            "dataset_size": int(train_result.get("dataset_size", 0)),
                            "train_dataset_size": int(train_result.get("dataset_size", 0)),
                            "md_dataset_size": md_dataset_size,
                            "atom_dataset_size": atom_dataset_size,
                            "compute_budget_mode": compute_mode,
                            "md_siesta_reference_count": md_budget,
                            "atomdisp_siesta_reference_count": atom_budget,
                            "budget_ratio": ratio,
                            "budget_mismatch_warning": mismatch_warning,
                            "leakage_warning": leakage_warnings_by_test_set.get(test_set, ""),
                            "leakage_summary": leakage_summary_by_test_set.get(test_set, ""),
                            "frozen_test_warning": frozen_test_warning,
                            "frozen_test_hash": frozen_test_hash,
                            "frozen_test_manifest": str(frozen_manifest_path) if frozen_manifest_path.exists() else "",
                            "siesta_settings_hash": manifest.get("siesta_settings_hash"),
                            "siesta_settings_warning": manifest.get("siesta_settings_warning", ""),
                            "model_config_hash": manifest.get("model_config_hash"),
                            "model_config_warning": manifest.get("model_config_warning", ""),
                            "basis_pseudopotential_warning": manifest.get("basis_pseudopotential_warning", ""),
                            "strict_comparison_mode": manifest.get("strict_comparison_mode", STRICT_COMPARISON_MODE),
                            "md_dataset_label": str(md_result.get("dataset_label", f"dataset_{md_dataset_size}")),
                            "atom_dataset_label": str(atom_result.get("dataset_label", f"dataset_{atom_dataset_size}")),
                            "seed": train_result.get("seed"),
                            "epoch": None,
                            "model_checkpoint": str(checkpoint),
                            "model_checkpoint_sha256": train_result.get("model_checkpoint_sha256"),
                            "checkpoint_manifest": train_result.get("checkpoint_manifest", ""),
                            "checkpoint_selection_warning": train_result.get("checkpoint_selection_warning", ""),
                            "reproducibility_warning": manifest.get("reproducibility_warning", ""),
                            "nested_subset_warning": train_result.get("nested_subset_warning", ""),
                            "prediction_dir": str(prediction_dir),
                            "siesta_reference_dir": str(cross_result_dir / "siesta_hamiltonians"),
                            "prediction_time_seconds": prediction_time,
                            "evaluation_time_seconds": evaluation_time,
                            "total_time_seconds": (
                                float(train_result.get("pipeline_elapsed_seconds") or 0.0)
                                + prediction_time
                                + evaluation_time
                            ),
                            "references": copy_counts["references"],
                            "structures": copy_counts["structures"],
                            "evaluation": evaluation_summary,
                        }
                        (cross_result_dir / "cross_evaluation_manifest.json").write_text(
                            json.dumps(json_safe(cross_manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                            encoding="utf-8",
                        )
                        summary["cross_evaluations"].append(cross_manifest)

        aggregate_command = [
            sys.executable,
            str(COMPARISON_ROOT / "scripts" / "aggregate_cross_metrics.py"),
            "--experiment-id",
            run_id,
            "--cross-root",
            str(cross_root),
            "--output-dir",
            str(summary_root),
        ]
        aggregate_result = self._run_local_script(aggregate_command, label="Agregando metricas cruzadas")
        if aggregate_result.returncode != 0:
            raise RuntimeError("Cross metric aggregation failed.")

        winner_command = [
            sys.executable,
            str(COMPARISON_ROOT / "scripts" / "analyze_winners.py"),
            "--metrics-csv",
            str(summary_root / "cross_evaluation_metrics.csv"),
            "--output-dir",
            str(summary_root),
            "--primary-metric",
            str((manifest.get("selected_metrics") or {}).get("primary_metric", DEFAULT_PRIMARY_METRIC)),
            "--minimum-robust-seeds",
            str(manifest.get("minimum_robust_seeds", MINIMUM_ROBUST_SEEDS)),
        ]
        winner_result = self._run_local_script(winner_command, label="Analizando winners")
        if winner_result.returncode != 0:
            raise RuntimeError("Winner analysis failed.")

        summary["ok"] = True
        summary["outputs"] = {
            "common_tests": str(common_root),
            "cross_evaluations": str(cross_root),
            "cross_evaluation_metrics": str(summary_root / "cross_evaluation_metrics.csv"),
            "winner_summary": str(summary_root / "winner_summary.csv"),
            "recommendation": str(summary_root / "recommendation.json"),
        }
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
            elif path == "/api/atom-fc-config":
                json_response(self, atom_fc_ui_config())
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
                raw_md_sizes = payload.get("md_sizes")
                displacement_options = parse_fc_displacement_options(
                    payload.get("fc_displacement_options")
                )
                fc_displacements = parse_fc_displacements(payload.get("fc_displacements"))
                structures_per_displacement = parse_structures_per_displacement(
                    payload.get("structures_per_displacement")
                )
                split_ratios = parse_split_ratios(payload.get("splits")) or dict(DEFAULT_SPLIT_RATIOS)
                random_seed = payload.get("random_seed")
                random_seed = None if random_seed in (None, "") else int(random_seed)
                max_datasets = parse_max_datasets(payload.get("max_datasets"))
                combination_mode = parse_combination_mode(payload.get("combination_mode"))
                split_mode = parse_split_mode(payload.get("split_mode"))
                test_sets = parse_test_sets(payload.get("test_sets"))
                primary_metric = str(payload.get("primary_metric") or DEFAULT_PRIMARY_METRIC).strip()
                compute_budget_mode = parse_compute_budget_mode(payload.get("compute_budget_mode"))
                raw_atom_sizes = payload.get("atom_sizes")
                atom_config = load_config(PIPELINES["atom_displacement"].config_path)
                force_constants = atom_config.get("structure", {}).get("force_constants", {}) or {}
                if payload.get("max_datasets") in (None, ""):
                    max_datasets = parse_max_datasets(force_constants.get("max_datasets"), 100)
                if payload.get("combination_mode") in (None, ""):
                    combination_mode = parse_combination_mode(
                        force_constants.get("combination_mode", "aligned")
                    )
                limit = atom_fc_sample_limit(atom_config)
                if limit is None:
                    raise RuntimeError("AtomDisplacement: FC no esta habilitado en la configuracion.")
                atom_dataset_specs = None
                if displacement_options:
                    atom_dataset_specs = build_fc_dataset_specs_from_options(
                        displacement_options,
                        combination_mode=combination_mode,
                        per_displacement_limit=limit,
                        split_ratios=split_ratios,
                        max_datasets=max_datasets,
                    )
                    atom_sizes = [int(spec["size"]) for spec in atom_dataset_specs]
                    if parse_bool(payload.get("sync_md_sizes"), True):
                        md_sizes = unique_ints_preserve_order(atom_sizes)
                    else:
                        md_sizes = parse_sizes(raw_md_sizes, atom_sizes)
                    fc_dataset_specs = None
                else:
                    md_sizes = parse_sizes(raw_md_sizes, [10, 20])
                    requested_atom_sizes = parse_sizes(raw_atom_sizes, [10])
                    atom_sizes, fc_dataset_specs = build_fc_dataset_specs(
                        requested_atom_sizes,
                        fc_displacements,
                        structures_per_displacement,
                        per_displacement_limit=limit,
                        split_ratios=split_ratios,
                    )
                json_response(
                    self,
                    EXPERIMENT_RUNNER.start(
                        md_sizes,
                        atom_sizes,
                        fc_dataset_specs,
                        atom_dataset_specs,
                        split_ratios,
                        random_seed,
                        split_mode,
                        test_sets,
                        primary_metric,
                        compute_budget_mode,
                    ),
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
