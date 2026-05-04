#!/usr/bin/env python3
"""Run MACE predictions from pipeline_config.yaml."""

from __future__ import annotations

import os
import inspect
import shutil
import sys

import pytorch_lightning as pl
from graph2mat.tools.lightning import MatrixDataModule, MatrixWriter
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

from md_pipeline_config import command, load_pipeline_config, paths, resolve_checkpoint


def require_command(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"No se encontro '{command_name}' en PATH. "
            "Activa tu entorno antes de ejecutar este script."
        )


def main() -> int:
    config = load_pipeline_config()
    pipeline_paths = paths(config)

    print("=== Pipeline MD (fase prediccion): inferencia del modelo ===")
    print(f"Repositorio: {pipeline_paths['training_dir'].parent}")
    print(f"Training dir: {pipeline_paths['training_dir']}")

    if not pipeline_paths["training_dir"].exists():
        raise RuntimeError(
            f"No existe el directorio de entrenamiento: {pipeline_paths['training_dir']}"
        )

    require_command(command(config, "graph2mat"))
    ckpt_path = resolve_checkpoint(config)
    os.chdir(pipeline_paths["training_dir"])

    training_data = config["training"]["data"]
    prediction = config["prediction"]
    model = LitMACEMatrixModel.load_from_checkpoint(
        str(pipeline_paths["training_dir"] / ckpt_path)
    )
    datamodule_kwargs = {
        "out_matrix": training_data["out_matrix"],
        "symmetric_matrix": bool(training_data["symmetric_matrix"]),
        "sub_point_matrix": bool(training_data.get("sub_point_matrix", False)),
        "basis_files": training_data["basis_files"],
        "predict_structs": str(prediction["predict_structs"]),
        "store_in_memory": bool(training_data.get("store_in_memory", True)),
    }
    if "n_matrix_components" in inspect.signature(MatrixDataModule).parameters:
        datamodule_kwargs["n_matrix_components"] = int(training_data.get("n_matrix_components", 2))
    datamodule = MatrixDataModule(**datamodule_kwargs)

    callbacks = []
    if prediction["callbacks"].get("matrix_writer", True):
        callbacks.append(
            MatrixWriter(
                output_file=str(prediction["output_file"]),
                splits=["predict"],
            )
        )

    trainer = pl.Trainer(
        accelerator=config["training"]["trainer"]["accelerator"],
        logger=False,
        callbacks=callbacks,
    )
    trainer.predict(
        model,
        datamodule=datamodule,
        ckpt_path=str(pipeline_paths["training_dir"] / ckpt_path),
    )
    print("\n=== Prediccion completada correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
