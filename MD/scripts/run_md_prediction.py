#!/usr/bin/env python3
"""Run MACE predictions from pipeline_config.yaml."""

from __future__ import annotations

import os
import inspect
import shutil
import sys
import glob
from pathlib import Path

TORCH_COMPAT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "torch_serialization_compat"
sys.path.insert(0, str(TORCH_COMPAT_DIR))
from torch_safe_globals import (
    allow_graph2mat_checkpoint_globals,
    apply_torch_float32_matmul_precision,
    apply_torch_num_threads,
)

allow_graph2mat_checkpoint_globals()

from md_pipeline_config import command, load_pipeline_config, paths, resolve_checkpoint

_PIPELINE_CONFIG = load_pipeline_config()
apply_torch_float32_matmul_precision(
    _PIPELINE_CONFIG.get("training", {}).get("torch_float32_matmul_precision")
)
apply_torch_num_threads((_PIPELINE_CONFIG.get("performance") or {}).get("torch_num_threads"))

import pytorch_lightning as pl
from graph2mat.tools.lightning import MatrixDataModule, MatrixWriter
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

EDGE_LABEL_CONSUMPTION_ERROR = "Predicted edge labels were not fully consumed by yield_from_batch"


class SafeMatrixWriter(MatrixWriter):
    """MatrixWriter variant that keeps valid single-sample predictions."""

    def _on_batch_end(self, split, trainer, pl_module, prediction, batch, batch_idx, dataloader_idx):
        try:
            return super()._on_batch_end(split, trainer, pl_module, prediction, batch, batch_idx, dataloader_idx)
        except ValueError as exc:
            if EDGE_LABEL_CONSUMPTION_ERROR not in str(exc):
                raise
            outputs = [
                self._get_out_file(batch.get_example(index), trainer)
                for index in range(getattr(batch, "num_graphs", 0))
            ]
            if not outputs or any(not output.exists() for output in outputs):
                raise
            print(
                "[WARN] MatrixWriter reporto etiquetas de arista sobrantes tras "
                "escribir las predicciones del batch; se continua."
            )


def require_command(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"No se encontro '{command_name}' en PATH. "
            "Activa tu entorno antes de ejecutar este script."
        )


def expected_prediction_outputs(predict_structs: str, output_file: str) -> list[Path]:
    outputs: list[Path] = []
    for structure in sorted(glob.glob(predict_structs)):
        structure_path = Path(structure)
        destination = Path(output_file)
        if not destination.is_absolute():
            destination = structure_path.parent / destination
        outputs.append(destination)
    return outputs


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
    allow_graph2mat_checkpoint_globals()
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
    if "batch_size" in inspect.signature(MatrixDataModule).parameters:
        datamodule_kwargs["batch_size"] = 1
    if "n_matrix_components" in inspect.signature(MatrixDataModule).parameters:
        datamodule_kwargs["n_matrix_components"] = int(training_data.get("n_matrix_components", 2))
    datamodule = MatrixDataModule(**datamodule_kwargs)

    callbacks = []
    if prediction["callbacks"].get("matrix_writer", True):
        callbacks.append(
            SafeMatrixWriter(
                output_file=str(prediction["output_file"]),
                splits=["predict"],
            )
        )

    trainer = pl.Trainer(
        accelerator=config["training"]["trainer"]["accelerator"],
        logger=False,
        callbacks=callbacks,
    )
    try:
        trainer.predict(
            model,
            datamodule=datamodule,
            ckpt_path=str(pipeline_paths["training_dir"] / ckpt_path),
        )
    except ValueError as exc:
        if EDGE_LABEL_CONSUMPTION_ERROR not in str(exc):
            raise
        outputs = expected_prediction_outputs(
            str(prediction["predict_structs"]),
            str(prediction["output_file"]),
        )
        if not outputs or any(not output.exists() for output in outputs):
            raise
        print(
            "[WARN] MatrixWriter reporto etiquetas de arista sobrantes tras "
            "escribir todas las predicciones esperadas; se continua."
        )
    print("\n=== Prediccion completada correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
