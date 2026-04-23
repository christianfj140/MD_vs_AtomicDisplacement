#!/usr/bin/env python3
"""Run prediction on the AtomDisplacement structures using a trained model."""

from __future__ import annotations

from atom_displacement_utils import TRAINING_DIR, resolve_ckpt_rel_path, run_command_in_venv

DEFAULT_CKPT_REL_PATH = (
    "lightning_logs/atom_displacement_model/version_0/checkpoints/best-0.ckpt"
)


def main() -> int:
    print("=== AtomDisplacement (predict) ===")
    ckpt_path = resolve_ckpt_rel_path(TRAINING_DIR, DEFAULT_CKPT_REL_PATH)
    cmd = [
        "graph2mat",
        "models",
        "mace",
        "main",
        "predict",
        "--ckpt_path",
        ckpt_path,
        "--data.predict_structs",
        "../dataset/samples/*/RUN.fdf",
        "--trainer.callbacks+",
        "MatrixWriter",
        "--trainer.callbacks.output_file",
        "ML_prediction.HSX",
    ]
    run_command_in_venv(cmd, cwd=TRAINING_DIR)
    print("\n=== Prediccion completada correctamente ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
