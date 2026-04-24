#!/usr/bin/env python3
"""Run MACE predictions from pipeline_config.yaml."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from md_pipeline_config import command, load_pipeline_config, paths, resolve_checkpoint


def require_command(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"No se encontró '{command_name}' en PATH. "
            "Activa tu entorno antes de ejecutar este script."
        )


def run_command(cmd: list[str], cwd: Path) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"El comando falló con código {result.returncode}: {' '.join(cmd)}"
        )


def main() -> int:
    config = load_pipeline_config()
    pipeline_paths = paths(config)

    print("=== Pipeline MD (fase predicción): inferencia del modelo ===")
    print(f"Repositorio: {pipeline_paths['training_dir'].parent}")
    print(f"Training dir: {pipeline_paths['training_dir']}")

    if not pipeline_paths["training_dir"].exists():
        raise RuntimeError(
            f"No existe el directorio de entrenamiento: {pipeline_paths['training_dir']}"
        )

    require_command(command(config, "graph2mat"))

    ckpt_path = resolve_checkpoint(config)
    cmd = [
        command(config, "graph2mat"),
        "models",
        "mace",
        "main",
        "predict",
        "--ckpt_path",
        ckpt_path,
        "--data.predict_structs",
        str(config["prediction"]["predict_structs"]),
    ]
    if config["prediction"]["callbacks"].get("matrix_writer", True):
        cmd.extend(
            [
                "--trainer.callbacks+",
                "MatrixWriter",
                "--trainer.callbacks.output_file",
                str(config["prediction"]["output_file"]),
            ]
        )

    run_command(cmd, cwd=pipeline_paths["training_dir"])
    print("\n=== Predicción completada correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
