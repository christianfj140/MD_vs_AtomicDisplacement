#!/usr/bin/env python3
"""Run MACE predictions from pipeline_config.yaml."""

from __future__ import annotations

import os
import inspect
import re
import shutil
import sys
import glob
import traceback
import json
from pathlib import Path

TORCH_COMPAT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "torch_serialization_compat"
sys.path.insert(0, str(TORCH_COMPAT_DIR))
from torch_safe_globals import (
    allow_graph2mat_checkpoint_globals,
    apply_torch_float32_matmul_precision,
    apply_torch_num_threads,
)

allow_graph2mat_checkpoint_globals()

from md_pipeline_config import command, config_dir, load_pipeline_config, paths, resolve_checkpoint
from graph2mat_material_config import (
    apply_material_graph2mat_config,
    resolve_matrix_component_policy,
    validate_model_matrix_component_policy,
)

_PIPELINE_CONFIG = load_pipeline_config()
apply_torch_float32_matmul_precision(
    _PIPELINE_CONFIG.get("training", {}).get("torch_float32_matmul_precision")
)
apply_torch_num_threads((_PIPELINE_CONFIG.get("performance") or {}).get("torch_num_threads"))

import pytorch_lightning as pl
from graph2mat.tools.lightning import MatrixDataModule, MatrixWriter
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

EDGE_LABEL_CONSUMPTION_ERROR = "Predicted edge labels were not fully consumed by yield_from_batch"
EDGE_LABEL_CONSUMPTION_PATTERN = re.compile(r"consumed=(\d+), total=(\d+)")


def incomplete_prediction_error(exc: ValueError) -> RuntimeError:
    return RuntimeError(
        "Graph2Mat MatrixWriter reported incomplete prediction export: "
        f"{exc}. Existing output files are not accepted as proof of complete predictions."
    )


def edge_label_consumption_counts(exc: ValueError) -> tuple[int, int] | None:
    match = EDGE_LABEL_CONSUMPTION_PATTERN.search(str(exc))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def recoverable_edge_label_consumption_error(exc: ValueError) -> bool:
    """Detect Graph2Mat's symmetric same-species edge accounting mismatch."""

    if EDGE_LABEL_CONSUMPTION_ERROR not in str(exc):
        return False
    counts = edge_label_consumption_counts(exc)
    if counts is None:
        return False
    consumed, total = counts
    return consumed > total


def warn_recovered_edge_label_consumption(exc: ValueError, *, batch_idx: int | None = None) -> None:
    suffix = f" batch={batch_idx}" if batch_idx is not None else ""
    print(
        "[WARN] Graph2Mat MatrixWriter reported an edge-label consumption "
        f"mismatch{suffix}: {exc}. Continuing and validating all prediction "
        "outputs after export.",
        file=sys.stderr,
    )


class StrictMatrixWriter(MatrixWriter):
    """MatrixWriter variant that fails closed unless output validation can decide."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warned_edge_label_consumption = False

    def _on_batch_end(self, split, trainer, pl_module, prediction, batch, batch_idx, dataloader_idx):
        try:
            return super()._on_batch_end(split, trainer, pl_module, prediction, batch, batch_idx, dataloader_idx)
        except ValueError as exc:
            if EDGE_LABEL_CONSUMPTION_ERROR not in str(exc):
                raise
            if recoverable_edge_label_consumption_error(exc):
                if not self._warned_edge_label_consumption:
                    warn_recovered_edge_label_consumption(exc, batch_idx=batch_idx)
                    self._warned_edge_label_consumption = True
                return None
            raise incomplete_prediction_error(exc) from exc


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


def clear_prediction_outputs(predict_structs: str, output_file: str) -> None:
    for output in expected_prediction_outputs(predict_structs, output_file):
        try:
            output.unlink()
        except FileNotFoundError:
            pass


def validate_prediction_outputs(predict_structs: str, output_file: str) -> None:
    outputs = expected_prediction_outputs(predict_structs, output_file)
    if not outputs:
        raise RuntimeError(f"No prediction structures matched: {predict_structs}")
    missing = [str(output) for output in outputs if not output.exists()]
    empty = [str(output) for output in outputs if output.exists() and output.stat().st_size <= 0]
    if missing:
        raise RuntimeError("Missing prediction outputs: " + ", ".join(missing[:10]))
    if empty:
        raise RuntimeError("Empty prediction outputs: " + ", ".join(empty[:10]))


def _basis_labels(pattern: str) -> set[str]:
    return {Path(path).name.removesuffix(".ion.xml") for path in glob.glob(pattern)}


def _sanitize_prediction_fdf(path: Path, labels: set[str]) -> None:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        lower = stripped.lower()
        if lower.startswith("numberofspecies"):
            out.append(f"NumberOfSpecies        {len(labels)}")
            i += 1
            continue
        if lower == "%block chemicalspecieslabel":
            out.append(lines[i])
            i += 1
            while i < len(lines) and not lines[i].strip().lower().startswith("%endblock"):
                parts = lines[i].split()
                if len(parts) >= 3 and parts[2] in labels:
                    out.append(lines[i])
                i += 1
            if i < len(lines):
                out.append(lines[i])
                i += 1
            continue
        if lower == "%block pao.basis":
            out.append(lines[i])
            i += 1
            keep = True
            while i < len(lines) and not lines[i].strip().lower().startswith("%endblock"):
                parts = lines[i].split()
                if lines[i] and not lines[i][0].isspace() and parts:
                    keep = parts[0] in labels
                if keep:
                    out.append(lines[i])
                i += 1
            if i < len(lines):
                out.append(lines[i])
                i += 1
            continue
        out.append(lines[i])
        i += 1
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def sanitize_prediction_structures_for_basis(predict_structs: str, basis_files: str) -> None:
    labels = _basis_labels(basis_files)
    if not labels:
        return
    for fdf in glob.glob(predict_structs):
        _sanitize_prediction_fdf(Path(fdf), labels)


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
    apply_material_graph2mat_config(
        config,
        base_dir=config_dir(config),
        dataset_dir=pipeline_paths["dataset_dir"],
        training_dir=pipeline_paths["training_dir"],
    )
    checkpoint_manifest_path = pipeline_paths["training_dir"] / "checkpoint_manifest.json"
    if checkpoint_manifest_path.is_file():
        checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        basis_files = checkpoint_manifest.get("predict_metrics_basis_files")
        if checkpoint_manifest.get("predict_metrics_only") and basis_files:
            config["training"]["data"]["basis_files"] = str(basis_files)
    ckpt_path = resolve_checkpoint(config)
    os.chdir(pipeline_paths["training_dir"])

    training_data = config["training"]["data"]
    prediction = config["prediction"]
    if checkpoint_manifest_path.is_file():
        checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        if checkpoint_manifest.get("predict_metrics_only") and checkpoint_manifest.get("predict_metrics_basis_files"):
            sanitize_prediction_structures_for_basis(
                str(prediction["predict_structs"]),
                str(training_data["basis_files"]),
            )
    allow_graph2mat_checkpoint_globals()
    model = LitMACEMatrixModel.load_from_checkpoint(
        str(pipeline_paths["training_dir"] / ckpt_path),
        root_dir=str(training_data.get("root_dir") or "."),
        basis_files=str(training_data["basis_files"]),
        weights_only=False,
    )
    datamodule_kwargs = {
        "out_matrix": training_data["out_matrix"],
        "symmetric_matrix": bool(training_data["symmetric_matrix"]),
        "sub_point_matrix": bool(training_data.get("sub_point_matrix", False)),
        "basis_files": training_data["basis_files"],
        "predict_structs": str(prediction["predict_structs"]),
        "store_in_memory": bool(training_data.get("store_in_memory", True)),
    }
    matrix_component_policy, n_matrix_components = resolve_matrix_component_policy(
        training_data,
        context="training.data",
    )
    validate_model_matrix_component_policy(
        model,
        matrix_component_policy=matrix_component_policy,
        n_matrix_components=n_matrix_components,
        context="MD prediction",
    )
    if "batch_size" in inspect.signature(MatrixDataModule).parameters:
        datamodule_kwargs["batch_size"] = 1
    if "n_matrix_components" in inspect.signature(MatrixDataModule).parameters:
        datamodule_kwargs["n_matrix_components"] = n_matrix_components
    if "matrix_component_policy" in inspect.signature(MatrixDataModule).parameters:
        datamodule_kwargs["matrix_component_policy"] = matrix_component_policy
    if "loader_threads" in inspect.signature(MatrixDataModule).parameters and training_data.get("loader_threads") is not None:
        datamodule_kwargs["loader_threads"] = int(training_data["loader_threads"])
    datamodule = MatrixDataModule(**datamodule_kwargs)

    callbacks = []
    if prediction["callbacks"].get("matrix_writer", True):
        callbacks.append(
            StrictMatrixWriter(
                output_file=str(prediction["output_file"]),
                splits=["predict"],
            )
        )

    trainer = pl.Trainer(
        accelerator=config["training"]["trainer"]["accelerator"],
        logger=False,
        callbacks=callbacks,
    )
    clear_prediction_outputs(
        str(prediction["predict_structs"]),
        str(prediction["output_file"]),
    )
    try:
        trainer.predict(
            model,
            datamodule=datamodule,
            ckpt_path=None,
        )
    except ValueError as exc:
        if EDGE_LABEL_CONSUMPTION_ERROR not in str(exc):
            raise
        if recoverable_edge_label_consumption_error(exc):
            warn_recovered_edge_label_consumption(exc)
        else:
            raise incomplete_prediction_error(exc) from exc
    validate_prediction_outputs(
        str(prediction["predict_structs"]),
        str(prediction["output_file"]),
    )
    print("\n=== Prediccion completada correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
