#!/usr/bin/env python3
"""Run prediction on the AtomDisplacement structures using a trained model."""

from __future__ import annotations

import os
import inspect

import pytorch_lightning as pl
from graph2mat.core.data.processing import MatrixDataProcessor
from graph2mat.tools.lightning import MatrixDataModule, MatrixWriter
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

from atom_displacement_utils import PIPELINE_CONFIG, TRAINING_DIR, resolve_ckpt_rel_path


def patch_graph2mat_run_loading() -> None:
    """Avoid passing ``basis=...`` when Graph2Mat reads SIESTA RUN.fdf files."""

    original = MatrixDataProcessor.get_config_kwargs

    def patched(self: MatrixDataProcessor, obj: object) -> dict[str, object]:
        kwargs = original(self, obj)
        kwargs.pop("basis", None)
        return kwargs

    MatrixDataProcessor.get_config_kwargs = patched


def main() -> int:
    print("=== AtomDisplacement (predict) ===")
    patch_graph2mat_run_loading()

    ckpt_path = resolve_ckpt_rel_path(TRAINING_DIR, "")
    os.chdir(TRAINING_DIR)

    prediction = PIPELINE_CONFIG["prediction"]
    data = prediction["data"]
    callbacks_config = prediction["callbacks"]
    model = LitMACEMatrixModel.load_from_checkpoint(str(TRAINING_DIR / ckpt_path))
    datamodule_kwargs = {
        "out_matrix": data["out_matrix"],
        "symmetric_matrix": bool(data["symmetric_matrix"]),
        "sub_point_matrix": bool(data.get("sub_point_matrix", False)),
        "basis_files": data["basis_files"],
        "predict_structs": data["predict_structs"],
        "store_in_memory": bool(data["store_in_memory"]),
    }
    if "n_matrix_components" in inspect.signature(MatrixDataModule).parameters:
        datamodule_kwargs["n_matrix_components"] = int(data["n_matrix_components"])
    datamodule = MatrixDataModule(**datamodule_kwargs)
    callbacks = []
    if bool(callbacks_config["matrix_writer"]):
        callbacks.append(MatrixWriter(output_file=callbacks_config["output_file"], splits=["predict"]))
    trainer = pl.Trainer(
        accelerator=PIPELINE_CONFIG["training"]["trainer"]["accelerator"],
        logger=False,
        callbacks=callbacks,
    )

    trainer.predict(model, datamodule=datamodule, ckpt_path=str(TRAINING_DIR / ckpt_path))
    print("\n=== Prediccion completada correctamente ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
