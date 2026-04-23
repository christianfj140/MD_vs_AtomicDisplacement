#!/usr/bin/env python3
"""Automatiza el testeo del modelo MACE según MD/command_history.txt.

Flujo replicado (hardcodeado):

graph2mat models mace main test \
   --ckpt_path lightning_logs/my_first_model/version_0/checkpoints/best-2040.ckpt  \
   --data.test_runs ../dataset/MD_steps/25/RUN.fdf \
   --trainer.callbacks+ PlotMatrixError --trainer.callbacks.show True \
   --trainer.callbacks+ SamplewiseMetricsLogger
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "MD" / "training"

TEST_COMMAND = [
    "graph2mat",
    "models",
    "mace",
    "main",
    "test",
    "--ckpt_path",
    "lightning_logs/my_first_model/version_0/checkpoints/best-2040.ckpt",
    "--data.test_runs",
    "../dataset/MD_steps/25/RUN.fdf",
    "--trainer.callbacks+",
    "PlotMatrixError",
    "--trainer.callbacks.show",
    "True",
    "--trainer.callbacks+",
    "SamplewiseMetricsLogger",
]


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
    print("=== Pipeline MD (fase test): evaluación del modelo ===")
    print(f"Repositorio: {REPO_ROOT}")
    print(f"Training dir: {TRAINING_DIR}")

    if not TRAINING_DIR.exists():
        raise RuntimeError(f"No existe el directorio de entrenamiento: {TRAINING_DIR}")

    require_command("graph2mat")
    run_command(TEST_COMMAND, cwd=TRAINING_DIR)

    print("\n=== Testeo completado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
