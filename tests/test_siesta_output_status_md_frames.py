"""MD trajectory frames validate; single-point runs still must prove they completed.

Regression guard for the 2026-07-28 breakage: hardening validate_snapshot to parse
RUN.out rejected every MD dataset in the repo (measured 167/167), because an MD frame is
a slice of a live SIESTA job and can never contain "Job completed" -- nor, for frame 0,
even a closed SCF cycle. The fix must not weaken the single-point contract, which is what
catches the derivative-stencil contamination.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "shared"))

from siesta_output_status import is_md_trajectory_frame, parse_siesta_output  # noqa: E402

SCF_BLOCK = "iscf     Eharris\nSCF cycle converged after 16 iterations\n"


def _write(dirpath: Path, *, fdf: str, out: str, metadata: str | None = None) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "RUN.fdf").write_text(fdf, encoding="utf-8")
    (dirpath / "RUN.out").write_text(out, encoding="utf-8")
    if metadata is not None:
        (dirpath / "metadata.json").write_text(metadata, encoding="utf-8")
    return dirpath


def test_md_frame_zero_is_valid_without_any_closed_scf_cycle(tmp_path):
    """Frame 0 is archived before its first cycle closes; RUN.out is cumulative."""
    d = _write(
        tmp_path / "frame0",
        fdf="MD.TypeOfRun Verlet\nMD.Steps 300\n",
        out="Authorization required, but no authorization protocol specified\nredata: ...\n",
    )
    status = parse_siesta_output(d / "RUN.out", d / "RUN.fdf")
    assert status["md_trajectory_frame"] is True
    assert status["valid"], status["parser_status"]
    assert "md_frame_without_closed_scf_cycle" in status["warnings"]


def test_md_frame_detected_from_generator_metadata(tmp_path):
    """The dataset generator's metadata is authoritative even if the fdf looks static."""
    d = _write(
        tmp_path / "meta",
        fdf="SystemLabel graphene_5x5\n",
        out=SCF_BLOCK,
        metadata='{"timestep_fs": 1.0, "run_fdf_geometry_source": "graphene_5x5.XV"}',
    )
    assert is_md_trajectory_frame(d / "RUN.fdf") is True
    assert parse_siesta_output(d / "RUN.out", d / "RUN.fdf")["valid"]


def test_single_point_stencil_is_not_an_md_frame_and_needs_job_completed(tmp_path):
    """`MD.TypeOfRun CG` with zero steps is a single point, not a trajectory."""
    d = _write(
        tmp_path / "stencil",
        fdf="MD.TypeOfRun CG\nMD.NumCGsteps 0\n",
        out=SCF_BLOCK,
    )
    status = parse_siesta_output(d / "RUN.out", d / "RUN.fdf")
    assert status["md_trajectory_frame"] is False
    assert not status["valid"]
    assert status["parser_status"] == "job_not_completed"

    (d / "RUN.out").write_text(SCF_BLOCK + "Job completed\n", encoding="utf-8")
    assert parse_siesta_output(d / "RUN.out", d / "RUN.fdf")["valid"]


def test_md_frame_still_rejected_when_the_run_actually_broke(tmp_path):
    """Being an MD frame is not a free pass: aborts and non-convergence still fail."""
    fdf = "MD.TypeOfRun Verlet\nMD.Steps 300\n"
    aborted = _write(tmp_path / "aborted", fdf=fdf, out=SCF_BLOCK + "MPI_ABORT\n")
    assert parse_siesta_output(aborted / "RUN.out", aborted / "RUN.fdf")["parser_status"] == "aborted"

    diverged = _write(
        tmp_path / "diverged", fdf=fdf, out="iscf     Eharris\nSCF cycle NOT converged\n"
    )
    status = parse_siesta_output(diverged / "RUN.out", diverged / "RUN.fdf")
    assert status["parser_status"] == "explicit_scf_nonconvergence"
    assert not status["valid"]
