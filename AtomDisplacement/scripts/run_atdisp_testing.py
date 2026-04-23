#!/usr/bin/env python3
"""Test the trained Graph2Mat/MACE model on one AtomDisplacement sample."""

from __future__ import annotations

import os

from atom_displacement_utils import (
    TRAINING_DIR,
    completed_sample_dirs,
    resolve_ckpt_rel_path,
    run_command_in_venv,
)

DEFAULT_CKPT_REL_PATH = (
    "lightning_logs/atom_displacement_model/version_0/checkpoints/best-0.ckpt"
)


def main() -> int:
    print("=== AtomDisplacement (test) ===")
    sample_dirs = completed_sample_dirs()
    if not sample_dirs:
        raise RuntimeError(
            "No hay muestras completadas para testear. Ejecuta primero run_single_points.py."
        )

    ckpt_path = resolve_ckpt_rel_path(TRAINING_DIR, DEFAULT_CKPT_REL_PATH)
    test_run = os.path.relpath(sample_dirs[0] / "RUN.fdf", TRAINING_DIR).replace("\\", "/")
    cmd = [
        "graph2mat",
        "models",
        "mace",
        "main",
        "test",
        "--ckpt_path",
        ckpt_path,
        "--data.test_runs",
        test_run,
        "--trainer.callbacks+",
        "PlotMatrixError",
        "--trainer.callbacks.show",
        "False",
        "--trainer.callbacks+",
        "SamplewiseMetricsLogger",
    ]
    run_command_in_venv(cmd, cwd=TRAINING_DIR)
    print("\n=== Testeo completado correctamente ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
