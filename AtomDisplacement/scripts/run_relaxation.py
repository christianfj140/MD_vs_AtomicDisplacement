#!/usr/bin/env python3
"""Run the reference H2O relaxation in AtomDisplacement/relaxed."""

from __future__ import annotations

from pathlib import Path

from atom_displacement_utils import (
    BASE_DIR,
    RELAXED_DIR,
    copy_pseudopotentials,
    ensure_dir,
    require_command,
    run_siesta_in_dir,
)


def prepare_relaxed_dir() -> Path:
    ensure_dir(RELAXED_DIR)
    copy_pseudopotentials(BASE_DIR, RELAXED_DIR)
    run_fdf_src = BASE_DIR / "RUN.fdf"
    run_fdf_dst = RELAXED_DIR / "RUN.fdf"
    run_fdf_dst.write_text(run_fdf_src.read_text(encoding="utf-8"), encoding="utf-8")
    return run_fdf_dst


def main() -> int:
    print("=== Relaxation pipeline for H2O ===")
    require_command("bash")
    require_command("siesta")

    run_fdf_path = prepare_relaxed_dir()
    print(f"[OK] Input preparado en {run_fdf_path}")
    run_siesta_in_dir(RELAXED_DIR, RELAXED_DIR / "RUN.out")
    print(f"[OK] Relajacion completada en {RELAXED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
