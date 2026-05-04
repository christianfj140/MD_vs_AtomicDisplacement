#!/usr/bin/env python3
"""Run prediction on the AtomDisplacement structures using a trained model."""

from __future__ import annotations

import os
import inspect
import csv
from pathlib import Path

import pytorch_lightning as pl
from graph2mat.core.data.processing import MatrixDataProcessor
from graph2mat.tools.lightning import MatrixDataModule, MatrixWriter
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

from atom_displacement_utils import DATASET_DIR, PIPELINE_CONFIG, TRAINING_DIR, resolve_ckpt_rel_path


SPLITS_DIR = DATASET_DIR / "splits"


def patch_graph2mat_run_loading() -> None:
    """Avoid passing ``basis=...`` when Graph2Mat reads SIESTA RUN.fdf files."""

    original = MatrixDataProcessor.get_config_kwargs

    def patched(self: MatrixDataProcessor, obj: object) -> dict[str, object]:
        kwargs = original(self, obj)
        kwargs.pop("basis", None)
        return kwargs

    MatrixDataProcessor.get_config_kwargs = patched


def relpath_from_training(path: Path) -> str:
    return os.path.relpath(path, TRAINING_DIR).replace("\\", "/")


def manifest_structures(path: Path) -> list[Path]:
    structures: list[Path] = []
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
                structures.append(Path(structure))
    return structures


def pattern_from_structures(structures: list[Path]) -> str:
    if len(structures) == 1:
        return relpath_from_training(structures[0])
    parents = {path.parent.parent for path in structures if path.name == PIPELINE_CONFIG["paths"]["run_fdf_name"]}
    if len(parents) == 1:
        return relpath_from_training(next(iter(parents)) / "*" / PIPELINE_CONFIG["paths"]["run_fdf_name"])
    raise RuntimeError("El manifest de predict contiene estructuras en carpetas no agrupables por un glob seguro.")


def strict_predict_structs() -> str:
    configured = str(PIPELINE_CONFIG["prediction"]["data"].get("predict_structs") or "")
    dangerous_full_dataset = "FC_steps" in configured or "samples/*" in configured
    if configured and not dangerous_full_dataset:
        return configured
    for name in ("test_valid_manifest.csv", "test_manifest.csv"):
        manifest = SPLITS_DIR / name
        if manifest.exists():
            structures = manifest_structures(manifest)
            if not structures:
                raise RuntimeError(f"El manifest de predict no contiene muestras validas: {manifest}")
            if dangerous_full_dataset:
                print("[WARN] prediction.data.predict_structs apuntaba al dataset completo; se usara el split test.")
            return pattern_from_structures(structures)
    if configured and bool(PIPELINE_CONFIG.get("prediction", {}).get("allow_full_dataset_debug", False)):
        print("[WARN] prediction.allow_full_dataset_debug=true; prediciendo sobre el glob configurado.")
        return configured
    raise RuntimeError(
        "AtomDisplacement strict prediction requiere predict_structs de test o "
        "dataset/splits/test_manifest.csv. No se predice sobre FC_steps completo "
        "salvo con prediction.allow_full_dataset_debug=true."
    )


def main() -> int:
    print("=== AtomDisplacement (predict) ===")
    patch_graph2mat_run_loading()

    ckpt_path = resolve_ckpt_rel_path(TRAINING_DIR, "")
    os.chdir(TRAINING_DIR)

    prediction = PIPELINE_CONFIG["prediction"]
    data = prediction["data"]
    callbacks_config = prediction["callbacks"]
    predict_structs = strict_predict_structs()
    model = LitMACEMatrixModel.load_from_checkpoint(str(TRAINING_DIR / ckpt_path))
    datamodule_kwargs = {
        "out_matrix": data["out_matrix"],
        "symmetric_matrix": bool(data["symmetric_matrix"]),
        "sub_point_matrix": bool(data.get("sub_point_matrix", False)),
        "basis_files": data["basis_files"],
        "predict_structs": predict_structs,
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
