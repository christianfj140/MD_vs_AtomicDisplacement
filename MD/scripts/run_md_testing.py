#!/usr/bin/env python3
"""Test the MACE model from pipeline_config.yaml."""

from __future__ import annotations

import os
import inspect
import shutil
import sys
from pathlib import Path

TORCH_COMPAT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "torch_serialization_compat"
sys.path.insert(0, str(TORCH_COMPAT_DIR))
from torch_safe_globals import (
    allow_graph2mat_checkpoint_globals,
    apply_torch_float32_matmul_precision,
    apply_torch_num_threads,
)

allow_graph2mat_checkpoint_globals()

from md_pipeline_config import command, config_dir, load_pipeline_config, paths, resolve_checkpoint, write_checkpoint_manifest
from graph2mat_material_config import apply_material_graph2mat_config

_PIPELINE_CONFIG = load_pipeline_config()
apply_torch_float32_matmul_precision(
    _PIPELINE_CONFIG.get("training", {}).get("torch_float32_matmul_precision")
)
apply_torch_num_threads((_PIPELINE_CONFIG.get("performance") or {}).get("torch_num_threads"))

import pytorch_lightning as pl
from graph2mat.tools.lightning import (
    MatrixDataModule,
    PlotMatrixError,
    SamplewiseMetricsLogger,
)
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel


def require_command(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"No se encontro '{command_name}' en PATH. "
            "Activa tu entorno antes de ejecutar este script."
        )


def main() -> int:
    config = load_pipeline_config()
    pipeline_paths = paths(config)

    print("=== Pipeline MD (fase test): evaluacion del modelo ===")
    print(f"Repositorio: {pipeline_paths['training_dir'].parent}")
    print(f"Training dir: {pipeline_paths['training_dir']}")

    if not pipeline_paths["training_dir"].exists():
        raise RuntimeError(
            f"No existe el directorio de entrenamiento: {pipeline_paths['training_dir']}"
        )

    require_command(command(config, "graph2mat"))
    apply_material_graph2mat_config(
        config,
        base_dir=config_dir(config),
        dataset_dir=pipeline_paths["dataset_dir"],
        training_dir=pipeline_paths["training_dir"],
    )
    ckpt_path = resolve_checkpoint(config)
    selection_mode = "configured_path" if config.get("checkpoint", {}).get("path") else str(config.get("checkpoint", {}).get("selection", "latest_version"))
    manifest_path = write_checkpoint_manifest(
        config,
        ckpt_path,
        selection_mode=selection_mode,
        selection_metric="tested_checkpoint",
    )
    print(f"[OK] Checkpoint manifest escrito en {manifest_path}")
    os.chdir(pipeline_paths["training_dir"])

    training_data = config["training"]["data"]
    callbacks_config = config["testing"]["callbacks"]
    allow_graph2mat_checkpoint_globals()
    model = LitMACEMatrixModel.load_from_checkpoint(
        str(pipeline_paths["training_dir"] / ckpt_path)
    )
    datamodule_kwargs = {
        "out_matrix": training_data["out_matrix"],
        "symmetric_matrix": bool(training_data["symmetric_matrix"]),
        "sub_point_matrix": bool(training_data.get("sub_point_matrix", False)),
        "basis_files": training_data["basis_files"],
        "test_runs": str(config["testing"]["test_runs"]),
        "store_in_memory": bool(training_data.get("store_in_memory", True)),
    }
    if "n_matrix_components" in inspect.signature(MatrixDataModule).parameters:
        datamodule_kwargs["n_matrix_components"] = int(training_data.get("n_matrix_components", 2))
    if "loader_threads" in inspect.signature(MatrixDataModule).parameters and training_data.get("loader_threads") is not None:
        datamodule_kwargs["loader_threads"] = int(training_data["loader_threads"])
    datamodule = MatrixDataModule(**datamodule_kwargs)

    callbacks = []
    if callbacks_config.get("plot_matrix_error", True):
        callbacks.append(
            PlotMatrixError(
                split="test",
                show=bool(callbacks_config.get("show_plot", True)),
            )
        )
    if callbacks_config.get("samplewise_metrics_logger", True):
        callbacks.append(SamplewiseMetricsLogger(splits=["test"]))

    trainer = pl.Trainer(
        accelerator=config["training"]["trainer"]["accelerator"],
        logger=False,
        callbacks=callbacks,
    )
    trainer.test(
        model,
        datamodule=datamodule,
        ckpt_path=str(pipeline_paths["training_dir"] / ckpt_path),
    )
    print("\n=== Testeo completado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
