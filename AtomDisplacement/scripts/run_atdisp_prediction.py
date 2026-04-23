#!/usr/bin/env python3
"""Run prediction on the AtomDisplacement structures using a trained model."""

from __future__ import annotations

import os

import pytorch_lightning as pl
from graph2mat.core.data.processing import MatrixDataProcessor
from graph2mat.tools.lightning import MatrixDataModule, MatrixWriter
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

from atom_displacement_utils import TRAINING_DIR, resolve_ckpt_rel_path

DEFAULT_CKPT_REL_PATH = (
    "lightning_logs/atom_displacement_model/version_0/checkpoints/best-0.ckpt"
)


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

    ckpt_path = resolve_ckpt_rel_path(TRAINING_DIR, DEFAULT_CKPT_REL_PATH)
    os.chdir(TRAINING_DIR)

    model = LitMACEMatrixModel.load_from_checkpoint(str(TRAINING_DIR / ckpt_path))
    datamodule = MatrixDataModule(
        out_matrix="hamiltonian",
        symmetric_matrix=True,
        basis_files="../relaxed/*.ion.xml",
        predict_structs="../dataset/samples/*/RUN.fdf",
        store_in_memory=True,
        n_matrix_components=2,
    )
    callbacks = [MatrixWriter(output_file="ML_prediction.HSX", splits=["predict"])]
    trainer = pl.Trainer(accelerator="cpu", logger=False, callbacks=callbacks)

    trainer.predict(model, datamodule=datamodule, ckpt_path=str(TRAINING_DIR / ckpt_path))
    print("\n=== Prediccion completada correctamente ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
