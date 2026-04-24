#!/usr/bin/env python3
"""Test the MACE model from pipeline_config.yaml."""

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

    print("=== Pipeline MD (fase test): evaluación del modelo ===")
    print(f"Repositorio: {pipeline_paths['training_dir'].parent}")
    print(f"Training dir: {pipeline_paths['training_dir']}")

    if not pipeline_paths["training_dir"].exists():
        raise RuntimeError(
            f"No existe el directorio de entrenamiento: {pipeline_paths['training_dir']}"
        )

    require_command(command(config, "graph2mat"))
    ckpt_path = resolve_checkpoint(config)
    callbacks = config["testing"]["callbacks"]
    cmd = [
        command(config, "graph2mat"),
        "models",
        "mace",
        "main",
        "test",
        "--ckpt_path",
        ckpt_path,
        "--data.test_runs",
        str(config["testing"]["test_runs"]),
    ]
    if callbacks.get("plot_matrix_error", True):
        cmd.extend(["--trainer.callbacks+", "PlotMatrixError"])
        cmd.extend(["--trainer.callbacks.show", str(callbacks.get("show_plot", True))])
    if callbacks.get("samplewise_metrics_logger", True):
        cmd.extend(["--trainer.callbacks+", "SamplewiseMetricsLogger"])

    run_command(cmd, cwd=pipeline_paths["training_dir"])
    print("\n=== Testeo completado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
