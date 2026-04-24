#!/usr/bin/env python3
"""Run the reference H2O relaxation in AtomDisplacement/relaxed."""

from __future__ import annotations

from pathlib import Path

from atom_displacement_utils import (
    BASE_DIR,
    PIPELINE_CONFIG,
    PIPELINE_PATHS,
    RELAXED_DIR,
    copy_pseudopotentials,
    ensure_dir,
    require_command,
    run_siesta_in_dir,
)
from pipeline_config_utils import command, render_relaxation_fdf


def prepare_relaxed_dir() -> Path:
    ensure_dir(RELAXED_DIR)
    copy_pseudopotentials(BASE_DIR, RELAXED_DIR)
    run_fdf_dst = PIPELINE_PATHS["relaxed_run_fdf_path"]
    run_fdf_dst.write_text(render_relaxation_fdf(PIPELINE_CONFIG), encoding="utf-8")
    return run_fdf_dst


def main() -> int:
    print("=== Relaxation pipeline for H2O ===")
    require_command(command(PIPELINE_CONFIG, "shell"))
    require_command(command(PIPELINE_CONFIG, "siesta"))

    run_fdf_path = prepare_relaxed_dir()
    print(f"[OK] Input preparado en {run_fdf_path}")
    run_siesta_in_dir(RELAXED_DIR, PIPELINE_PATHS["relaxed_run_out_path"])
    print(f"[OK] Relajacion completada en {RELAXED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
