#!/usr/bin/env python3
"""Train a Graph2Mat/MACE model on the AtomDisplacement dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path

from atom_displacement_utils import (
    PIPELINE_CONFIG,
    PIPELINE_PATHS,
    TRAINING_DIR,
    completed_sample_dirs,
    ensure_dir,
    generated_sample_dirs,
    relaxed_basis_files,
    run_command_in_venv,
)
from pipeline_config_utils import command, render_training_config

CONFIG_PATH = PIPELINE_PATHS["training_config_path"]
RUNS_JSON_PATH = PIPELINE_PATHS["runs_json_path"]


def relpath_from_training(path: Path) -> str:
    return os.path.relpath(path, TRAINING_DIR).replace("\\", "/")


def write_runs_json(sample_dirs: list[Path]) -> str:
    run_fdf_name = PIPELINE_CONFIG["paths"]["run_fdf_name"]
    runs = [relpath_from_training(sample_dir / run_fdf_name) for sample_dir in sample_dirs]
    RUNS_JSON_PATH.write_text(
        json.dumps({"train": runs}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return RUNS_JSON_PATH.name


def build_config_yaml() -> str:
    sample_dirs = completed_sample_dirs()
    all_sample_dirs = generated_sample_dirs()
    min_completed = int(PIPELINE_CONFIG["training"]["min_completed_samples"])
    if len(sample_dirs) < min_completed:
        raise RuntimeError(
            f"Se necesitan al menos {min_completed} muestras completadas para entrenar. "
            "Ejecuta primero run_single_points.py para generar mas datos validos."
        )
    if not relaxed_basis_files():
        raise RuntimeError("No se encontraron ficheros .ion.xml en relaxed/")

    runs_json = write_runs_json(sample_dirs)
    configured_batch_size = int(PIPELINE_CONFIG["training"]["data"]["batch_size"])
    PIPELINE_CONFIG["training"]["data"]["batch_size"] = min(
        configured_batch_size,
        len(sample_dirs),
    )
    PIPELINE_CONFIG["training"]["data"]["runs_json"] = runs_json

    if len(sample_dirs) != len(all_sample_dirs):
        print(
            "[WARN] No todas las muestras generadas estan completadas. "
            f"Se entrenara solo con {len(sample_dirs)} de {len(all_sample_dirs)}."
        )

    return render_training_config(PIPELINE_CONFIG)


def main() -> int:
    print("=== AtomDisplacement (train) ===")
    ensure_dir(TRAINING_DIR)
    CONFIG_PATH.write_text(build_config_yaml(), encoding="utf-8")
    print(f"[OK] Config escrito en {CONFIG_PATH}")
    fit_command = [command(PIPELINE_CONFIG, "graph2mat"), *PIPELINE_CONFIG["training"]["fit_args"]]
    run_command_in_venv(fit_command, cwd=TRAINING_DIR)
    print(f"[INFO] Si quieres ver metricas: {PIPELINE_CONFIG['training']['tensorboard_hint']}")
    print("\n=== Entrenamiento completado correctamente ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
