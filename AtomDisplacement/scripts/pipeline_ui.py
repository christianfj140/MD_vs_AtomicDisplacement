#!/usr/bin/env python3
"""Local web UI and API for editing pipeline_config.yaml and running main_atdisp.py."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from pipeline_config_utils import command, load_pipeline_config, paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "pipeline_config.yaml"
MAIN_SCRIPT = PROJECT_ROOT / "scripts" / "main_atdisp.py"
UI_DIR = PROJECT_ROOT / "ui"


class PipelineRunner:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._logs: list[str] = []
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._returncode: int | None = None
        self._command: list[str] | None = None

    def start(self) -> dict[str, Any]:
        config = load_pipeline_config()
        pipeline_paths = paths(config)
        venv_activate = pipeline_paths["venv_activate"]
        if not venv_activate.exists():
            raise RuntimeError(f"No se encontro el entorno virtual: {venv_activate}")

        shell = command(config, "shell")
        python = command(config, "python")
        shell_command = (
            f"source {shlex.quote(str(venv_activate))} "
            f"&& {shlex.quote(python)} {shlex.quote(str(MAIN_SCRIPT))}"
        )

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("El pipeline ya se esta ejecutando.")
            self._logs = []
            self._started_at = time.time()
            self._finished_at = None
            self._returncode = None
            self._command = [shell, "-lc", shell_command]
            self._process = subprocess.Popen(
                self._command,
                cwd=PROJECT_ROOT,
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
            self._logs.append(f"\n[UI] Proceso finalizado con codigo {returncode}.\n")

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return {
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


RUNNER = PipelineRunner()


def read_config() -> dict[str, Any]:
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("pipeline_config.yaml debe contener un objeto YAML.")
    return {"raw": raw, "parsed": parsed}


def write_config(raw: str) -> dict[str, Any]:
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("pipeline_config.yaml debe contener un objeto YAML.")
    CONFIG_PATH.write_text(raw, encoding="utf-8")
    load_pipeline_config()
    return {"raw": raw, "parsed": parsed}


def set_nested(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target: Any = config
    index = 0
    while index < len(parts) - 1:
        if not isinstance(target, dict):
            raise RuntimeError(f"No se puede escribir la ruta de configuracion: {path}")
        match = None
        for end in range(len(parts) - 1, index, -1):
            candidate = ".".join(parts[index:end])
            if candidate in target and isinstance(target[candidate], dict):
                match = candidate
                index = end
                break
        if match is None:
            raise RuntimeError(f"No existe la ruta de configuracion: {path}")
        target = target[match]

    leaf = ".".join(parts[index:])
    if leaf not in target:
        for end in range(len(parts), index, -1):
            candidate = ".".join(parts[index:end])
            if candidate in target:
                leaf = candidate
                break
    target[leaf] = value


def patch_config(updates: dict[str, Any]) -> dict[str, Any]:
    config_data = read_config()
    parsed = config_data["parsed"]
    for key, value in updates.items():
        set_nested(parsed, key, value)
    raw = yaml.safe_dump(parsed, sort_keys=False, default_flow_style=False, allow_unicode=False)
    CONFIG_PATH.write_text(raw, encoding="utf-8")
    load_pipeline_config()
    return {"raw": raw, "parsed": parsed}


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error_response(handler: BaseHTTPRequestHandler, exc: Exception, status: int = 400) -> None:
    json_response(handler, {"error": str(exc)}, status=status)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(length).decode("utf-8")
    if not body:
        return {}
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError("El cuerpo JSON debe ser un objeto.")
    return parsed


class PipelineUIHandler(BaseHTTPRequestHandler):
    server_version = "AtomDisplacementPipelineUI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[pipeline-ui] {self.address_string()} - {fmt % args}\n")

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        try:
            if parsed_url.path == "/api/health":
                config = load_pipeline_config()
                json_response(
                    self,
                    {
                        "ok": True,
                        "project_root": str(PROJECT_ROOT),
                        "config_path": str(CONFIG_PATH),
                        "main_script": str(MAIN_SCRIPT),
                        "venv_activate": str(paths(config)["venv_activate"]),
                    },
                )
            elif parsed_url.path == "/api/config":
                json_response(self, read_config())
            elif parsed_url.path == "/api/run/status":
                json_response(self, RUNNER.status())
            elif parsed_url.path == "/api/run/logs":
                query = parse_qs(parsed_url.query)
                since = int(query.get("since", ["0"])[0])
                json_response(self, RUNNER.logs(since=since))
            elif parsed_url.path == "/":
                self._serve_file(UI_DIR / "index.html")
            else:
                requested = (UI_DIR / parsed_url.path.lstrip("/")).resolve()
                if UI_DIR.resolve() not in requested.parents:
                    raise FileNotFoundError(parsed_url.path)
                self._serve_file(requested)
        except FileNotFoundError:
            error_response(self, RuntimeError("No encontrado."), HTTPStatus.NOT_FOUND)
        except Exception as exc:
            error_response(self, exc)

    def do_PUT(self) -> None:
        try:
            if urlparse(self.path).path != "/api/config":
                raise FileNotFoundError(self.path)
            payload = read_json_body(self)
            content = payload.get("content")
            if not isinstance(content, str):
                raise RuntimeError("Falta el campo string 'content'.")
            json_response(self, write_config(content))
        except FileNotFoundError:
            error_response(self, RuntimeError("No encontrado."), HTTPStatus.NOT_FOUND)
        except Exception as exc:
            error_response(self, exc)

    def do_PATCH(self) -> None:
        try:
            if urlparse(self.path).path != "/api/config":
                raise FileNotFoundError(self.path)
            payload = read_json_body(self)
            updates = payload.get("updates")
            if not isinstance(updates, dict):
                raise RuntimeError("Falta el campo objeto 'updates'.")
            json_response(self, patch_config(updates))
        except FileNotFoundError:
            error_response(self, RuntimeError("No encontrado."), HTTPStatus.NOT_FOUND)
        except Exception as exc:
            error_response(self, exc)

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/api/run":
                json_response(self, RUNNER.start(), status=HTTPStatus.ACCEPTED)
            elif path == "/api/run/stop":
                json_response(self, RUNNER.stop(), status=HTTPStatus.ACCEPTED)
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
    parser = argparse.ArgumentParser(description="Serve the AtomDisplacement pipeline UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), PipelineUIHandler)
    print(f"AtomDisplacement Pipeline UI listening on http://{args.host}:{args.port}")
    print(f"Project root: {PROJECT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Pipeline UI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
