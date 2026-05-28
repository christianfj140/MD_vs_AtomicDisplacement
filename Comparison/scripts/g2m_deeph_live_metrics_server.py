#!/usr/bin/env python3
"""Read-only sidecar API for live Graph2Mat-vs-DeepH sweep metrics."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from g2m_deeph_live_metrics import live_metric_scaling_rows
from g2m_deeph_metrics import COMMON_METRIC_GROUPS, build_common_plot_payload


class LiveMetricsHandler(BaseHTTPRequestHandler):
    run_root: Path

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
            self.wfile.write(json.dumps({"ok": True, "run_root": str(self.run_root)}).encode("utf-8"))
            return
        try:
            rows = live_metric_scaling_rows(self.run_root)
            payload = build_common_plot_payload(
                None,
                metric_scaling_rows=rows,
                timing_scaling_rows=[],
                status_payload={"run_root": str(self.run_root), "running": True},
            )
            payload.update(
                {
                    "schema": "graph2mat_deeph_live_plot_payload_v1",
                    "run_id": self.run_root.name,
                    "run_root": str(self.run_root),
                    "metric_scaling_rows": rows,
                    "live_metric_rows": len(rows),
                    "metric_groups": COMMON_METRIC_GROUPS,
                    "source": "live_training_sweep_metrics_sidecar",
                }
            )
            self._headers()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        except Exception as exc:  # Sidecar errors should be visible, not fatal to the main run.
            self._headers(HTTPStatus.INTERNAL_SERVER_ERROR)
            self.wfile.write(json.dumps({"error": str(exc), "run_root": str(self.run_root)}).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handler = type("ConfiguredLiveMetricsHandler", (LiveMetricsHandler,), {"run_root": args.run_root.resolve(strict=False)})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        json.dumps(
            {
                "status": "serving",
                "url": f"http://{args.host}:{args.port}/api/g2m-deeph/live-plots",
                "run_root": str(args.run_root.resolve(strict=False)),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
