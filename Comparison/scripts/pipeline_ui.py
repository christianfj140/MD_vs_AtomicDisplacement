#!/usr/bin/env python3
"""Local web UI and API for running MD and AtomDisplacement together."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).resolve().parents[1] / "ui"


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
            self._logs = [f"[UI] Ejecutando {self.spec.label}: {self.spec.main_script}\n"]
            self._started_at = time.time()
            self._finished_at = None
            self._returncode = None
            self._command = [shell, "-lc", shell_command]
            self._process = subprocess.Popen(
                self._command,
                cwd=self.spec.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            process = self._process

        threading.Thread(target=self._collect_output, args=(process,), daemon=True).start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return self.status()
            process.terminate()
            self._logs.append("\n[UI] Solicitud de parada enviada.\n")
        return self.status()

    def _collect_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            with self._lock:
                self._logs.append(line)
        returncode = process.wait()
        with self._lock:
            self._returncode = returncode
            self._finished_at = time.time()
            if self._process is process:
                self._process = None
            self._logs.append(f"\n[UI] {self.spec.label} finalizado con codigo {returncode}.\n")

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
    }


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
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
