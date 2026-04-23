#!/usr/bin/env python3
"""Test the trained Graph2Mat/MACE model on one AtomDisplacement sample."""

from __future__ import annotations

import os

import pytorch_lightning as pl
from graph2mat.core.data.processing import MatrixDataProcessor
from graph2mat.tools.lightning import MatrixDataModule, PlotMatrixError, SamplewiseMetricsLogger
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

from atom_displacement_utils import TRAINING_DIR, completed_sample_dirs, resolve_ckpt_rel_path

DEFAULT_CKPT_REL_PATH = (
    "lightning_logs/atom_displacement_model/version_0/checkpoints/best-0.ckpt"
)


def patch_graph2mat_run_loading() -> None:
    """Avoid passing ``basis=...`` when Graph2Mat reads SIESTA RUN.fdf files.

    In our AtomDisplacement samples, the CLI path ``RUN.fdf -> read_hamiltonian(geometry=...)``
    can return ``None`` inside Graph2Mat. Reading the same run without injecting ``basis``
    works correctly, so we apply the workaround locally in this script.
    """

    original = MatrixDataProcessor.get_config_kwargs

    def patched(self: MatrixDataProcessor, obj: object) -> dict[str, object]:
        kwargs = original(self, obj)
        kwargs.pop("basis", None)
        return kwargs

    MatrixDataProcessor.get_config_kwargs = patched


def main() -> int:
    print("=== AtomDisplacement (test) ===")
    sample_dirs = completed_sample_dirs()
    if not sample_dirs:
        raise RuntimeError(
            "No hay muestras completadas para testear. Ejecuta primero run_single_points.py."
        )

    patch_graph2mat_run_loading()

    ckpt_path = resolve_ckpt_rel_path(TRAINING_DIR, DEFAULT_CKPT_REL_PATH)
    test_run = os.path.relpath(sample_dirs[0] / "RUN.fdf", TRAINING_DIR).replace("\\", "/")
    os.chdir(TRAINING_DIR)

    model = LitMACEMatrixModel.load_from_checkpoint(str(TRAINING_DIR / ckpt_path))
    datamodule = MatrixDataModule(
        out_matrix="hamiltonian",
        symmetric_matrix=True,
        basis_files="../relaxed/*.ion.xml",
        test_runs=test_run,
        store_in_memory=True,
        n_matrix_components=2,
    )
    callbacks = [
        PlotMatrixError(split="test", show=False, store_in_logger=False),
        SamplewiseMetricsLogger(splits=["test"], output_file="sample_metrics.csv"),
    ]
    trainer = pl.Trainer(accelerator="cpu", logger=False, callbacks=callbacks)

    print(f"[INFO] Test sample: {test_run}")
    trainer.test(model, datamodule=datamodule, ckpt_path=str(TRAINING_DIR / ckpt_path))
    print("\n=== Testeo completado correctamente ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
