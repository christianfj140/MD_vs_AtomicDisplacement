#!/usr/bin/env python3
"""Test the trained Graph2Mat/MACE model on one AtomDisplacement sample."""

from __future__ import annotations

import os
import inspect
import csv
import sys
from pathlib import Path

TORCH_COMPAT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "torch_serialization_compat"
sys.path.insert(0, str(TORCH_COMPAT_DIR))
from torch_safe_globals import allow_graph2mat_checkpoint_globals

allow_graph2mat_checkpoint_globals()

import pytorch_lightning as pl
from graph2mat.core.data.processing import MatrixDataProcessor
from graph2mat.tools.lightning import MatrixDataModule, PlotMatrixError, SamplewiseMetricsLogger
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

from atom_displacement_utils import DATASET_DIR, PIPELINE_CONFIG, TRAINING_DIR, completed_sample_dirs, resolve_ckpt_rel_path
from pipeline_config_utils import write_checkpoint_manifest


SPLITS_DIR = DATASET_DIR / "splits"


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


def relpath_from_training(path: Path) -> str:
    return os.path.relpath(path, TRAINING_DIR).replace("\\", "/")


def manifest_structures(path: Path) -> list[Path]:
    runs: list[Path] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            status = str(row.get("status") or "").lower()
            valid = str(row.get("valid") or "").lower()
            if status and status not in {"valid", "completed"}:
                continue
            if valid and valid not in {"true", "1", "yes"}:
                continue
            structure = row.get("structure_path")
            if structure and Path(structure).exists():
                runs.append(Path(structure))
    return runs


def runs_pattern_from_structures(structures: list[Path]) -> str:
    if len(structures) == 1:
        return relpath_from_training(structures[0])
    parents = {path.parent.parent for path in structures if path.name == PIPELINE_CONFIG["paths"]["run_fdf_name"]}
    if len(parents) == 1:
        return relpath_from_training(next(iter(parents)) / "*" / PIPELINE_CONFIG["paths"]["run_fdf_name"])
    raise RuntimeError("El manifest de test contiene estructuras en carpetas no agrupables por un glob seguro.")


def strict_test_runs() -> str:
    configured_test_runs = PIPELINE_CONFIG["testing"].get("test_runs")
    if configured_test_runs:
        return str(configured_test_runs)
    for name in ("test_valid_manifest.csv", "test_manifest.csv"):
        manifest = SPLITS_DIR / name
        if manifest.exists():
            runs = manifest_structures(manifest)
            if not runs:
                raise RuntimeError(f"El manifest de test no contiene muestras validas: {manifest}")
            return runs_pattern_from_structures(runs)
    if bool(PIPELINE_CONFIG.get("testing", {}).get("allow_sample_index_debug", False)):
        sample_dirs = completed_sample_dirs()
        if not sample_dirs:
            raise RuntimeError("No hay muestras completadas para testear. Ejecuta primero run_single_points.py.")
        sample_index = int(PIPELINE_CONFIG["testing"]["sample_index"])
        return relpath_from_training(sample_dirs[sample_index] / PIPELINE_CONFIG["paths"]["run_fdf_name"])
    raise RuntimeError(
        "AtomDisplacement strict testing requiere testing.test_runs o "
        "dataset/splits/test_manifest.csv. No se usara sample_index salvo con "
        "testing.allow_sample_index_debug=true."
    )


def main() -> int:
    print("=== AtomDisplacement (test) ===")
    patch_graph2mat_run_loading()

    ckpt_path = resolve_ckpt_rel_path(TRAINING_DIR, "")
    selection_mode = "configured_path" if PIPELINE_CONFIG.get("checkpoint", {}).get("path") else str(PIPELINE_CONFIG.get("checkpoint", {}).get("selection", "latest_version"))
    manifest_path = write_checkpoint_manifest(
        PIPELINE_CONFIG,
        ckpt_path,
        selection_mode=selection_mode,
        selection_metric="tested_checkpoint",
    )
    print(f"[OK] Checkpoint manifest escrito en {manifest_path}")
    test_run = strict_test_runs()
    os.chdir(TRAINING_DIR)

    testing = PIPELINE_CONFIG["testing"]
    data = testing["data"]
    callbacks_config = testing["callbacks"]
    allow_graph2mat_checkpoint_globals()
    model = LitMACEMatrixModel.load_from_checkpoint(str(TRAINING_DIR / ckpt_path))
    datamodule_kwargs = {
        "out_matrix": data["out_matrix"],
        "symmetric_matrix": bool(data["symmetric_matrix"]),
        "sub_point_matrix": bool(data.get("sub_point_matrix", False)),
        "basis_files": data["basis_files"],
        "test_runs": test_run,
        "store_in_memory": bool(data["store_in_memory"]),
    }
    if "n_matrix_components" in inspect.signature(MatrixDataModule).parameters:
        datamodule_kwargs["n_matrix_components"] = int(data["n_matrix_components"])
    datamodule = MatrixDataModule(**datamodule_kwargs)
    callbacks = []
    if bool(callbacks_config["plot_matrix_error"]):
        callbacks.append(
            PlotMatrixError(
                split="test",
                show=bool(callbacks_config["show_plot"]),
                store_in_logger=bool(callbacks_config["store_plot_in_logger"]),
            )
        )
    if bool(callbacks_config["samplewise_metrics_logger"]):
        callbacks.append(
            SamplewiseMetricsLogger(
                splits=["test"],
                output_file=callbacks_config["output_file"],
            )
        )
    trainer = pl.Trainer(
        accelerator=PIPELINE_CONFIG["training"]["trainer"]["accelerator"],
        logger=False,
        callbacks=callbacks,
    )

    print(f"[INFO] Test sample: {test_run}")
    trainer.test(model, datamodule=datamodule, ckpt_path=str(TRAINING_DIR / ckpt_path))
    print("\n=== Testeo completado correctamente ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
