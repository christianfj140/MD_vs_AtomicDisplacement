#!/usr/bin/env python3
"""Static audit: does any compute path fall back from GPU to CPU silently?

The roadmap's GPU-first / explicit-CPU-fallback gate has no auditor. This
script is a static scanner, not a runtime one: it does not import torch or run
anything, so it works even in an environment with no GPU. It walks the
repository's own compute scripts and flags every place that looks like a
CUDA-to-CPU fallback (a ``cuda`` reference near a literal ``"cpu"``/``'cpu'``
device string) and checks whether that fallback is *explicit* -- meaning
something nearby records why the fallback happened (a named record/log call,
a ``reason=``/``*_reason`` field, or the ``effective_backend``/
``requested_backend`` provenance pair this repository already uses in
``Comparison/results/epc/preregistration/matbg/*_protocol.json``).

This is deliberately a fail-closed *reviewer's* heuristic, not a compiler: a
flagged site is a candidate for human review, not proof of a bug, and an
unflagged repository is not proof there is no silent fallback anywhere (a
fallback expressed through indirection the scanner cannot see would be
missed). What it guarantees is the inverse: every textual pattern this script
recognises as "device picked from a CUDA-availability check, no nearby record
of why" is reported, never swallowed.

Two known-good patterns already live in this codebase and are used as the
compliance bar:

* ``graph2mat_autograd_derivatives.resolve_jvp_backend`` returns a
  ``JvpBackendRecord(requested, effective, status, reason)`` on every CPU
  fallback -- "a CPU fallback is never silent" (its own docstring).
* The EPC preregistration protocols for S53/S55 declare
  ``backend: {requested_backend, effective_backend, backend_fallback_reason}``.

Usage::

    audit_epc_backends.py --fail-on-silent-cpu
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

SCHEMA = "epc_backend_provenance_audit_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "Comparison" / "results" / "epc_repair" / "backend_provenance_audit.json"
DEFAULT_SCAN_ROOTS = (
    REPO_ROOT / "Comparison" / "scripts",
    REPO_ROOT / "shared",
)
DEFAULT_RUNTIME_MANIFESTS = (
    REPO_ROOT / "Comparison/results/epc/preflight/graph2mat_jvp_runtime.json",
)
SKIP_DIR_NAMES = {"__pycache__", ".git"}

CUDA_TOKEN = re.compile(r"cuda", re.IGNORECASE)
CPU_LITERAL = re.compile(r"""(['"])cpu\1""")
PROVENANCE_MARKERS = re.compile(
    r"record|logger\.|log\.|print\(|warn|raise |reason\s*=|provenance|telemetry"
    r"|fallback_reason|effective_backend|requested_backend|never silent|never_silent",
    re.IGNORECASE,
)
FALLBACK_HINT = re.compile(
    r"\b(?:if|else|available|fallback)\b|requested_backend|effective_backend",
    re.IGNORECASE,
)
DOCSTRING = re.compile(r"(?P<quote>\"\"\"|''')[\s\S]*?(?P=quote)")
# How far past a 'cuda' line to look for the paired 'cpu' fallback literal.
FALLBACK_WINDOW = 8
# How far past the fallback site to look for an explicit provenance marker.
PROVENANCE_WINDOW = 25


def iter_python_files(root: Path):
    if not root.exists():
        return
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        yield path


def find_candidates(text: str) -> list[dict[str, Any]]:
    # Prose can mention both devices near real code; preserve line numbers but ignore it.
    text = DOCSTRING.sub(lambda match: "\n" * match.group(0).count("\n"), text)
    lines = text.splitlines()
    candidates: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not CUDA_TOKEN.search(line):
            continue
        fallback_window = lines[index : index + FALLBACK_WINDOW + 1]
        cpu_offset = next(
            (offset for offset, wline in enumerate(fallback_window) if CPU_LITERAL.search(wline)),
            None,
        )
        if cpu_offset is None:
            continue
        # A backend enum or prose mentioning both devices is not a fallback.
        # Require control-flow/device-selection language in the paired span.
        candidate_span = "\n".join(fallback_window[: cpu_offset + 1])
        if not FALLBACK_HINT.search(candidate_span):
            continue
        provenance_window = lines[index : index + PROVENANCE_WINDOW + 1]
        explicit = any(PROVENANCE_MARKERS.search(wline) for wline in provenance_window)
        candidates.append(
            {
                "cuda_line": index + 1,
                "cpu_line": index + cpu_offset + 1,
                "snippet": line.strip()[:200],
                "explicit": explicit,
            }
        )
    return candidates


def scan(roots: list[Path]) -> tuple[list[dict[str, Any]], int]:
    findings: list[dict[str, Any]] = []
    files_scanned = 0
    for root in roots:
        for path in iter_python_files(root):
            files_scanned += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                findings.append({"path": str(path), "error": str(exc), "candidates": []})
                continue
            candidates = find_candidates(text)
            if candidates:
                findings.append({"path": str(path), "candidates": candidates})
    return findings, files_scanned


def _backend_records(value: Any):
    if isinstance(value, dict):
        if {"requested_backend", "effective_backend"} & value.keys():
            yield value
        for child in value.values():
            yield from _backend_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _backend_records(child)


def audit_runtime_manifests(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    required = {"requested_backend", "effective_backend", "backend_preflight", "device", "precision", "backend_fallback_reason"}
    for path in paths:
        if not path.is_file():
            rows.append({"path": str(path), "valid": False, "reason": "missing"})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("manifest must be an object")
        except (OSError, ValueError) as exc:
            rows.append({"path": str(path), "valid": False, "reason": str(exc)})
            continue
        records = list(_backend_records(payload))
        complete = [record for record in records if required <= record.keys()]
        fallback_documented = all(
            record.get("effective_backend") == record.get("requested_backend")
            or bool(record.get("backend_fallback_reason"))
            for record in complete
        )
        rows.append({
            "path": str(path),
            "valid": bool(records) and len(complete) == len(records) and fallback_documented
            and bool(payload.get("generated_at"))
            and payload.get("schema") == "graph2mat_jvp_runtime_preflight_v1"
            and all(str(record.get("backend_preflight")).upper() in {"PASS", "PASSED"}
                    and bool(record.get("device")) and bool(record.get("precision"))
                    and str(record.get("effective_backend", "")).lower() in {"cuda", "gpu"}
                    for record in complete),
            "records_found": len(records),
            "complete_records": len(complete),
            "fallbacks_documented": fallback_documented,
        })
    return rows


def build_report(*, roots: list[Path] | None = None, runtime_manifests: list[Path] | None = None) -> dict[str, Any]:
    roots = roots if roots is not None else list(DEFAULT_SCAN_ROOTS)
    findings, files_scanned = scan(roots)
    silent: list[dict[str, Any]] = []
    explicit: list[dict[str, Any]] = []
    for entry in findings:
        for candidate in entry.get("candidates", []):
            row = {"path": entry["path"], **candidate}
            (explicit if candidate["explicit"] else silent).append(row)
    runtime = audit_runtime_manifests(
        list(DEFAULT_RUNTIME_MANIFESTS) if runtime_manifests is None else runtime_manifests
    )
    return {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scanned_roots": [str(root) for root in roots],
        "files_scanned": files_scanned,
        "files_with_candidates": len(findings),
        "silent_cpu_fallback_candidates": silent,
        "explicit_cpu_fallback_sites": explicit,
        "runtime_manifests": runtime,
        "summary": {
            "silent_count": len(silent),
            "explicit_count": len(explicit),
            "no_silent_fallbacks_found": len(silent) == 0,
            "runtime_evidence_valid": bool(runtime) and all(row["valid"] for row in runtime),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        dest="scan_roots",
        default=None,
        help="Directory to scan for *.py files (repeatable). Defaults to Comparison/scripts and shared.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-manifest", type=Path, action="append", default=None)
    parser.add_argument(
        "--fail-on-silent-cpu",
        action="store_true",
        help="Exit non-zero if any silent CPU-fallback candidate is found.",
    )
    parser.add_argument("--require-runtime", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = args.scan_roots if args.scan_roots else list(DEFAULT_SCAN_ROOTS)
    report = build_report(roots=roots, runtime_manifests=args.runtime_manifest)
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True, ensure_ascii=False))
    if args.fail_on_silent_cpu and report["summary"]["silent_count"] > 0:
        return 1
    if args.require_runtime and not report["summary"]["runtime_evidence_valid"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
