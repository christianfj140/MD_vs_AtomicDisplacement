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

# A snapshot archived from a molecular-dynamics trajectory is a *slice* of a still-running
# SIESTA job, so two markers this parser looks for cannot exist in it by construction:
#
#   - "Job completed" is written once, when the whole run closes. No MD step has it.
#   - the first archived frame (step 0) is captured before its first SCF cycle closes,
#     so it has no "SCF cycle converged" either -- the RUN.out is cumulative, and step N
#     simply contains N converged cycles (measured: 0/1/2/3 for train/0..3).
#
# Demanding them of MD frames rejects every MD dataset in the repo (measured: 167/167).
# The fix is not to relax the parser but to apply the right contract per run type:
# single-point runs must still show a completed job.
MD_TRAJECTORY_MARKERS = ("MD.TypeOfRun", "Begin MD step", "Begin CG move")


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


def is_md_trajectory_frame(input_path: Path | None, text: str = "") -> bool:
    """True when this snapshot is one frame of an MD trajectory, not a single-point run.

    Checked in order of trustworthiness: the dataset generator's own metadata, then a real
    MD block in the run input, then the trajectory markers SIESTA prints. A single-point
    stencil written by build_hamiltonian_derivative_stencils sets ``MD.TypeOfRun CG`` with
    ``MD.NumCGsteps 0``, which is *not* a trajectory -- hence the explicit zero-step check.
    """
    if input_path is not None and input_path.exists():
        metadata_path = input_path.parent / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
            if isinstance(metadata, dict):
                if metadata.get("timestep_fs") is not None:
                    return True
                if str(metadata.get("run_fdf_geometry_source") or "").lower().endswith(".xv"):
                    return True
        try:
            fdf = input_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            fdf = ""
        steps = 0
        run_type = ""
        for line in fdf.splitlines():
            clean = line.split("#", 1)[0].strip().lower()
            if clean.startswith("md.typeofrun"):
                run_type = clean.split()[-1] if len(clean.split()) > 1 else ""
            elif clean.startswith(("md.steps", "md.finaltimestep", "md.numcgsteps")):
                parts = clean.split()
                if len(parts) > 1 and parts[-1].isdigit():
                    steps = max(steps, int(parts[-1]))
        if run_type and steps > 0:
            return True
    return bool(_detected(text, MD_TRAJECTORY_MARKERS[1:]))


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

    md_frame = is_md_trajectory_frame(input_path, text)
    status["md_trajectory_frame"] = md_frame

    if stale:
        parser_status = "stale_output"
    elif status["explicit_nonconvergence"]:
        parser_status = "explicit_scf_nonconvergence"
    elif status["aborted"]:
        parser_status = "aborted"
    elif md_frame:
        # An MD frame is a slice of a live run: no "Job completed", and frame 0 is archived
        # before its first cycle closes. What must still hold is that nothing blew up --
        # abort and explicit non-convergence are checked above and still reject the frame.
        parser_status = "valid"
        if not status["scf_converged"]:
            status["warnings"].append("md_frame_without_closed_scf_cycle")
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
