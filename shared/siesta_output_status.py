"""Canonical, fail-closed status parser for archived SIESTA text output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


JOB_COMPLETION_MARKERS = ("Job completed",)
SCF_CONVERGENCE_MARKERS = ("SCF cycle converged", "PostSCF", "FINAL_HF")
SCF_STARTED_MARKERS = (
    "SCF cycle converged",
    "iscf     Eharris",
    "SCF Convergence by",
    "PostSCF",
    "FINAL_HF",
)
EXPLICIT_NONCONVERGENCE_MARKERS = (
    "SCF cycle NOT converged",
    "SCF cycle did not converge",
    "SCF_NOT_CONV",
)
ABORT_MARKERS = ("Job aborted", "MPI_ABORT", "ERROR STOP", "FATAL ERROR")
MD_RUN_FDF_XV_MARKER = "Graph2Mat MD geometry materialized from siesta.XV"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_rewritten_from_xv(input_path: Path | None) -> bool:
    if input_path is None or not input_path.exists():
        return False
    metadata_path = input_path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        if isinstance(metadata, dict) and (
            metadata.get("run_fdf_rewritten_from_xv") is True
            or str(metadata.get("run_fdf_geometry_source") or "").lower() == "siesta.xv"
        ):
            return True
    try:
        return MD_RUN_FDF_XV_MARKER in input_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def _detected(text: str, markers: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [marker for marker in markers if marker.lower() in lowered]


def parse_siesta_output(
    output_path: Path | None,
    input_path: Path | None = None,
) -> dict[str, Any]:
    """Return machine-readable execution evidence; validity is fail-closed."""

    output_path = Path(output_path) if output_path is not None else None
    input_path = Path(input_path) if input_path is not None else None
    status: dict[str, Any] = {
        "output_path": str(output_path) if output_path is not None else "",
        "input_path": str(input_path) if input_path is not None else "",
        "output_exists": False,
        "output_readable": False,
        "output_fresh": False,
        "stale_output": False,
        "input_rewritten_after_run": False,
        "job_completed": False,
        "scf_started": False,
        "scf_converged": False,
        "explicit_nonconvergence": False,
        "aborted": False,
        "parser_status": "missing_run_out",
        "markers_detected": [],
        "run_out_sha256": "",
        "valid": False,
        "warnings": [],
    }
    if output_path is None or not output_path.exists() or not output_path.is_file():
        return status

    status["output_exists"] = True
    try:
        raw = output_path.read_bytes()
        text = raw.decode("utf-8", errors="ignore")
    except OSError as exc:
        status["parser_status"] = f"unreadable_run_out:{exc}"
        return status
    status["output_readable"] = True
    status["run_out_sha256"] = hashlib.sha256(raw).hexdigest()
    if not text.strip():
        status["parser_status"] = "empty_run_out"
        return status

    rewritten = input_rewritten_from_xv(input_path)
    stale = bool(
        input_path is not None
        and input_path.exists()
        and output_path.stat().st_mtime < input_path.stat().st_mtime
        and not rewritten
    )
    status["input_rewritten_after_run"] = rewritten
    status["stale_output"] = stale
    status["output_fresh"] = not stale
    if rewritten and output_path.stat().st_mtime < input_path.stat().st_mtime:
        status["warnings"].append("md_post_siesta_run_fdf_mtime")

    groups = {
        "job_completed": _detected(text, JOB_COMPLETION_MARKERS),
        "scf_started": _detected(text, SCF_STARTED_MARKERS),
        "scf_converged": _detected(text, SCF_CONVERGENCE_MARKERS),
        "explicit_nonconvergence": _detected(text, EXPLICIT_NONCONVERGENCE_MARKERS),
        "aborted": _detected(text, ABORT_MARKERS),
    }
    status["markers_detected"] = sorted({marker for values in groups.values() for marker in values})
    for key, values in groups.items():
        status[key] = bool(values)

    if stale:
        parser_status = "stale_output"
    elif status["explicit_nonconvergence"]:
        parser_status = "explicit_scf_nonconvergence"
    elif status["aborted"]:
        parser_status = "aborted"
    elif not status["scf_started"]:
        parser_status = "scf_not_started"
    elif not status["scf_converged"]:
        parser_status = "scf_not_converged"
    elif not status["job_completed"]:
        parser_status = "job_not_completed"
    else:
        parser_status = "valid"
    status["parser_status"] = parser_status
    status["valid"] = parser_status == "valid"
    return status
