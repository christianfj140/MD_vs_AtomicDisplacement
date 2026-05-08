#!/usr/bin/env python3
"""Train the MACE model from pipeline_config.yaml."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from md_pipeline_config import (
    command,
    load_pipeline_config,
    paths,
    render_training_config,
    resolve_checkpoint,
    write_checkpoint_manifest,
)

TORCH_COMPAT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "torch_serialization_compat"
sys.path.insert(0, str(TORCH_COMPAT_DIR))
from torch_safe_globals import env_with_torch_compat


def require_command(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"No se encontró '{command_name}' en PATH. "
            "Activa tu entorno antes de ejecutar este script."
        )


def run_command(cmd: list[str], cwd: Path, config: dict) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        env=env_with_torch_compat(
            matmul_precision=config.get("training", {}).get("torch_float32_matmul_precision"),
            performance=config.get("performance", {}),
        ),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"El comando falló con código {result.returncode}: {' '.join(cmd)}"
        )


def write_config_yaml(config: dict) -> None:
    pipeline_paths = paths(config)
    # Para reproducibilidad en esta fase, sobreescribimos config.yaml.
    pipeline_paths["training_config_path"].write_text(
        render_training_config(config),
        encoding="utf-8",
    )
    print(f"[OK] Config escrito en {pipeline_paths['training_config_path']}")


def main() -> int:
    config = load_pipeline_config()
    pipeline_paths = paths(config)
    fit_command = [
        command(config, "graph2mat"),
        "models",
        "mace",
        "main",
        "fit",
        "-c",
        pipeline_paths["training_config_path"].name,
    ]

    print("=== Pipeline MD (fase entrenamiento): fit del modelo ===")
    print(f"Repositorio: {pipeline_paths['training_dir'].parent}")
    print(f"Training dir: {pipeline_paths['training_dir']}")

    require_command(command(config, "graph2mat"))

    pipeline_paths["training_dir"].mkdir(parents=True, exist_ok=True)
    write_config_yaml(config)
    run_command(fit_command, cwd=pipeline_paths["training_dir"], config=config)
    ckpt_path = resolve_checkpoint(config)
    selection_mode = "configured_path" if config.get("checkpoint", {}).get("path") else str(config.get("checkpoint", {}).get("selection", "latest_version"))
    manifest_path = write_checkpoint_manifest(
        config,
        ckpt_path,
        selection_mode=selection_mode,
        selection_metric="best_checkpoint",
    )
    print(f"[OK] Checkpoint manifest escrito en {manifest_path}")

    log_dir = config["training"]["trainer"]["logger"]["init_args"]["save_dir"]
    print(f"[INFO] Si quieres ver métricas: tensorboard --logdir {log_dir}")
    print("\n=== Entrenamiento completado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
