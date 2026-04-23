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
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "MD" / "training"
DEFAULT_CKPT_REL_PATH = "lightning_logs/my_first_model/version_0/checkpoints/best-2040.ckpt"


def _best_step(path: Path) -> int:
    match = re.search(r"best-(\d+)\.ckpt$", path.name)
    return int(match.group(1)) if match else -1


def resolve_ckpt_rel_path() -> str:
    """Resuelve el checkpoint a usar para test/predict.

    Prioridad:
    1) Ruta fija original de command_history.txt (best-2040.ckpt).
    2) Si no existe, elegir automáticamente el best-<step>.ckpt más alto dentro de
       lightning_logs/*/checkpoints/, siguiendo la pista de command_history_variables.
    """
    default_abs = TRAINING_DIR / DEFAULT_CKPT_REL_PATH
    if default_abs.exists():
        return DEFAULT_CKPT_REL_PATH

    candidates = sorted(
        TRAINING_DIR.glob("lightning_logs/**/checkpoints/best-*.ckpt"),
        key=_best_step,
    )

    if candidates:
        selected = candidates[-1]
        rel = selected.relative_to(TRAINING_DIR).as_posix()
        print(
            "[WARN] No existe el checkpoint hardcodeado best-2040.ckpt. "
            f"Se usará automáticamente: {rel}"
        )
        return rel

    raise RuntimeError(
        "No se encontró ningún checkpoint best-*.ckpt en MD/training/lightning_logs. "
        "Entrena el modelo primero o ajusta la ruta del checkpoint."
    )


TEST_COMMAND = [
    "graph2mat",
    "models",
    "mace",
    "main",
    "test",
    "--ckpt_path",
    "__CKPT_PATH__",
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
    ckpt_path = resolve_ckpt_rel_path()
    cmd = [ckpt_path if token == "__CKPT_PATH__" else token for token in TEST_COMMAND]
    run_command(cmd, cwd=TRAINING_DIR)
    print("\n=== Testeo completado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
