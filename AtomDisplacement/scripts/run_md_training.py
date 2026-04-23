#!/usr/bin/env python3
"""Automatiza el entrenamiento MACE según MD/command_history.txt.

Flujo replicado (hardcodeado):
1) cd ..
2) mkdir training
3) cd training
4) crear config.yaml
5) graph2mat models mace main fit -c config.yaml
6) (opcional) tensorboard --logdir lightning_logs
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "MD" / "training"
CONFIG_PATH = TRAINING_DIR / "config.yaml"

# Config hardcodeado para esta fase inicial (basado en el config actual del repo).
CONFIG_YAML_CONTENT = """# pytorch_lightning==2.6.1
data:
    # We want to fit the density matrix, change to hamiltonian or energy_density_matrix
    # if you want to fit those.
    out_matrix: hamiltonian
    # Specify that it is a symmetric matrix (will save operations and predictions will be
    # strictly symmetric)
    symmetric_matrix: True
    # Where to find the basis files. In this dataset they are stored as XML.
    basis_files: ../dataset/MD_steps/basis/*.ion.xml
    # Where to find the run files. Sisl will attempt to read the matrix from these files.
    train_runs: ../dataset/MD_steps/*/RUN.fdf
    # Data will be split in batches during the training process. Specify how big these
    # batches should be
    batch_size: 10
    # Keep the matrices loaded in memory so that we don't need to read them each time.
    # (This might not be possible for very big datasets)
    store_in_memory: True
model:
    # We could leave this empty and just use the defaults, but for the sake
    # of learning, we will mention some of the model's most important parameters.
    # FIRST, MACE PARAMETERS
    # Number of times that messages are sent through the graph.
    num_interactions: 1
    # Number that determines how you take into account many-body interactions
    # The higher, the more complex the interactions. 1 means just interact through pairs.
    correlation: 1
    # Maximum order of spherical harmonics used internally by mace.
    # This should at least be as high as your highest order orbital.
    max_ell: 2
    # Size of MACE's internal representation. Here 10 scalars, 10 vectors, and
    # 10 order 2 spherical harmonics. Increasing the number of features will most
    # likely increase the performance if you have enough data.
    hidden_irreps: 10x0e + 10x1o + 10x2e
    # The loss function to use for the optimizer. You can use any of the functions
    # in graph2mat.data.metrics. This is part of the training process, but
    # LightningCLI requires it here for some strange reason.
    loss: graph2mat.metrics.block_type_mae
    # The learning rate for the optimizer. Increasing this might make the learning
    # faster and/or increase performance, but increasing it too much might make
    # the optimizer diverge. It can also make the learning more noisy.
    optim_lr: 0.005
trainer:
    # Run training on cpu (change to gpu if you have a GPU).
    accelerator: cpu
    # Define how the results of the training process will be logged.
    # Everything will be stored in a lightning_logs/my_first_model directory.
    # Change the name for other models that you train.
    logger:
        class_path: TensorBoardLogger
        init_args:
            name: my_first_model
            save_dir: lightning_logs
    # Number of times the training process goes over the whole dataset (one epoch)
    # We could set it to something very high if we want to stop it manually when we
    # are satisfied.
    max_epochs: 200
"""

FIT_COMMAND = ["graph2mat", "models", "mace", "main", "fit", "-c", "config.yaml"]


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


def ensure_training_dir() -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)


def write_config_yaml() -> None:
    # Para reproducibilidad en esta fase, sobreescribimos config.yaml.
    CONFIG_PATH.write_text(CONFIG_YAML_CONTENT, encoding="utf-8")
    print(f"[OK] Config escrito en {CONFIG_PATH}")


def main() -> int:
    print("=== Pipeline MD (fase entrenamiento): fit del modelo ===")
    print(f"Repositorio: {REPO_ROOT}")
    print(f"Training dir: {TRAINING_DIR}")

    require_command("graph2mat")

    ensure_training_dir()
    write_config_yaml()
    run_command(FIT_COMMAND, cwd=TRAINING_DIR)

    print("[INFO] Si quieres ver métricas: tensorboard --logdir lightning_logs")
    print("\n=== Entrenamiento completado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
