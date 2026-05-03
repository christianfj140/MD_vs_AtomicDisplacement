#!/usr/bin/env python3
"""Local web UI and API for running MD and AtomDisplacement together."""

from __future__ import annotations

import argparse
import csv
import itertools
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


DEFAULT_SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


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
                raise RuntimeError(
                    "Manifest points to a missing result_dir. "
                    f"manifest={manifest_path}, result_dir={result_dir}"
                )
            sparse_rows = read_csv_rows(result_dir / "metrics" / "sparse_metrics.csv")
            spectral_rows = read_csv_rows(result_dir / "metrics" / "spectral_metrics.csv")
            dos_rows = read_csv_rows(result_dir / "metrics" / "dos_metrics.csv")
            relationship_rows = read_csv_rows(
                result_dir / "metrics" / "matrix_spectrum_relationship.csv"
            )
            if not sparse_rows and not spectral_rows and not dos_rows and not relationship_rows:
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
                        "matrix_spectrum": numeric_means(relationship_rows),
                    },
                    "samples": {
                        "sparse": sparse_rows,
                        "spectral": spectral_rows,
                        "dos": dos_rows,
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
    return {"runs": runs}


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
    ) -> None:
        original_configs = {
            key: spec.config_path.read_text(encoding="utf-8")
            for key, spec in PIPELINES.items()
        }
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
            for size in md_sizes:
                self._ensure_not_stopped()
                self._restore_original_config("md", original_configs)
                result = self._run_one("md", size, run_id, split_ratios=split_ratios)
                with self._lock:
                    self._results.append(result)
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
                )
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

    def _run_one(
        self,
        key: str,
        size: int,
        run_id: str,
        dataset_label: str | None = None,
        fc_displacements: list[dict[str, Any]] | None = None,
        split_ratios: dict[str, float] | None = None,
        random_seed: int | None = None,
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
            self._prepare_md_config(config, workspace, size, split_ratios)
            write_yaml(spec.config_path, config)
            self._append("[UI] Config temporal escrita; se restaurara al finalizar el experimento.\n")
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
                write_yaml(spec.config_path, config)
                self._append("[UI] Config de entrenamiento restaurada tras SIESTA del test.\n")
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
            "strategy": "spread",
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
        config["structure"]["force_constants"]["allow_missing_matrix"] = True
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
        effective_size = int(prepare_metadata.get("effective_size", size))
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
            "result_dir": str(result_dir),
            "predicted_hamiltonians": prediction_count,
            "siesta_hamiltonians": reference_count,
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
