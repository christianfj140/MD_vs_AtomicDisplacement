#!/usr/bin/env python3
"""Train a Graph2Mat/MACE model on the AtomDisplacement dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path

from atom_displacement_utils import (
    TRAINING_DIR,
    completed_sample_dirs,
    ensure_dir,
    generated_sample_dirs,
    relaxed_basis_files,
    run_command_in_venv,
)

CONFIG_PATH = TRAINING_DIR / "config.yaml"
RUNS_JSON_PATH = TRAINING_DIR / "runs.json"
MODEL_NAME = "atom_displacement_model"
FIT_COMMAND = ["graph2mat", "models", "mace", "main", "fit", "-c", "config.yaml"]


def relpath_from_training(path: Path) -> str:
    return os.path.relpath(path, TRAINING_DIR).replace("\\", "/")


def write_runs_json(sample_dirs: list[Path]) -> str:
    runs = [relpath_from_training(sample_dir / "RUN.fdf") for sample_dir in sample_dirs]
    RUNS_JSON_PATH.write_text(
        json.dumps({"train": runs}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return RUNS_JSON_PATH.name


def build_config_yaml() -> str:
    sample_dirs = completed_sample_dirs()
    all_sample_dirs = generated_sample_dirs()
    if len(sample_dirs) < 2:
        raise RuntimeError(
            "Se necesitan al menos 2 muestras completadas para entrenar. "
            "Ejecuta primero run_single_points.py para generar mas datos validos."
        )
    if not relaxed_basis_files():
        raise RuntimeError("No se encontraron ficheros .ion.xml en relaxed/")

    basis_glob = relpath_from_training(TRAINING_DIR.parent / "relaxed" / "*.ion.xml")
    runs_json = write_runs_json(sample_dirs)
    batch_size = min(10, len(sample_dirs))

    if len(sample_dirs) != len(all_sample_dirs):
        print(
            "[WARN] No todas las muestras generadas estan completadas. "
            f"Se entrenara solo con {len(sample_dirs)} de {len(all_sample_dirs)}."
        )

    return f"""# pytorch_lightning==2.6.1
data:
    out_matrix: hamiltonian
    symmetric_matrix: True
    basis_files: {basis_glob}
    runs_json: {runs_json}
    batch_size: {batch_size}
    store_in_memory: True
model:
    num_interactions: 1
    correlation: 1
    max_ell: 2
    hidden_irreps: 10x0e + 10x1o + 10x2e
    loss: graph2mat.metrics.block_type_mae
    optim_lr: 0.005
trainer:
    accelerator: cpu
    logger:
        class_path: TensorBoardLogger
        init_args:
            name: {MODEL_NAME}
            save_dir: lightning_logs
    max_epochs: 200
"""


def main() -> int:
    print("=== AtomDisplacement (train) ===")
    ensure_dir(TRAINING_DIR)
    CONFIG_PATH.write_text(build_config_yaml(), encoding="utf-8")
    print(f"[OK] Config escrito en {CONFIG_PATH}")
    run_command_in_venv(FIT_COMMAND, cwd=TRAINING_DIR)
    print("[INFO] Si quieres ver metricas: tensorboard --logdir lightning_logs")
    print("\n=== Entrenamiento completado correctamente ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
