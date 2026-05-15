#!/usr/bin/env python3
"""Train a Graph2Mat/MACE model on the AtomDisplacement dataset."""

from __future__ import annotations

import json
import os
import csv
from glob import glob
from pathlib import Path

from atom_displacement_utils import (
    PIPELINE_CONFIG,
    PIPELINE_PATHS,
    DATASET_DIR,
    TRAINING_DIR,
    completed_sample_dirs,
    ensure_dir,
    generated_sample_dirs,
    relaxed_basis_files,
    run_command_in_venv,
)
from pipeline_config_utils import (
    command,
    config_dir,
    render_training_config,
    require_explicit_validation_split,
    resolve_checkpoint,
    write_checkpoint_manifest,
)
from graph2mat_material_config import (
    apply_material_graph2mat_config,
    write_graph2mat_config_provenance,
)

CONFIG_PATH = PIPELINE_PATHS["training_config_path"]
RUNS_JSON_PATH = PIPELINE_PATHS["runs_json_path"]
SPLITS_DIR = DATASET_DIR / "splits"


def relpath_from_training(path: Path) -> str:
    return os.path.relpath(path, TRAINING_DIR).replace("\\", "/")


def write_runs_json(train_sample_dirs: list[Path], val_sample_dirs: list[Path]) -> str:
    run_fdf_name = PIPELINE_CONFIG["paths"]["run_fdf_name"]
    train_runs = [
        relpath_from_training(sample_dir / run_fdf_name)
        for sample_dir in train_sample_dirs
    ]
    val_runs = [
        relpath_from_training(sample_dir / run_fdf_name)
        for sample_dir in val_sample_dirs
    ]
    RUNS_JSON_PATH.write_text(
        json.dumps({"train": train_runs, "val": val_runs}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return RUNS_JSON_PATH.name


def configured_val_runs_match_samples() -> bool:
    value = PIPELINE_CONFIG["training"]["data"].get("val_runs")
    if value in (None, ""):
        return False
    patterns = value if isinstance(value, list) else [value]
    matches: list[str] = []
    for pattern in patterns:
        raw_path = Path(str(pattern))
        search = raw_path if raw_path.is_absolute() else TRAINING_DIR / raw_path
        matches.extend(glob(str(search)))
    return bool(matches)


def read_manifest_runs(path: Path) -> list[Path]:
    run_fdf_name = PIPELINE_CONFIG["paths"]["run_fdf_name"]
    sample_dirs: list[Path] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            status = str(row.get("status") or "").lower()
            valid = str(row.get("valid") or "").lower()
            if status and status not in {"valid", "completed"}:
                continue
            if valid and valid not in {"true", "1", "yes"}:
                continue
            sample_dir_value = row.get("sample_dir")
            structure_value = row.get("structure_path")
            if sample_dir_value:
                sample_dir = Path(sample_dir_value)
            elif structure_value:
                sample_dir = Path(structure_value).parent
            else:
                continue
            if (sample_dir / run_fdf_name).exists():
                sample_dirs.append(sample_dir)
    return sorted(dict.fromkeys(sample_dirs))


def strict_train_sample_dirs() -> list[Path]:
    for name in ("train_valid_manifest.csv", "train_manifest.csv"):
        manifest = SPLITS_DIR / name
        if manifest.exists():
            samples = read_manifest_runs(manifest)
            if samples:
                return samples
            raise RuntimeError(f"El manifest de train no contiene muestras validas: {manifest}")
    if bool(PIPELINE_CONFIG.get("training", {}).get("allow_all_completed_debug", False)):
        print("[WARN] training.allow_all_completed_debug=true; usando todas las muestras completadas.")
        return completed_sample_dirs()
    raise RuntimeError(
        "AtomDisplacement strict training requiere dataset/splits/train_manifest.csv "
        "o train_valid_manifest.csv. Usa la UI/prepare splits o activa "
        "training.allow_all_completed_debug solo para depuracion."
    )


def strict_validation_sample_dirs() -> list[Path]:
    for name in ("validation_valid_manifest.csv", "validation_manifest.csv"):
        manifest = SPLITS_DIR / name
        if manifest.exists():
            samples = read_manifest_runs(manifest)
            if samples:
                return samples
            raise RuntimeError(f"El manifest de validation no contiene muestras validas: {manifest}")
    raise RuntimeError(
        "AtomDisplacement strict training requiere dataset/splits/validation_manifest.csv "
        "o validation_valid_manifest.csv para seleccionar checkpoints con validacion."
    )


def build_config_yaml() -> str:
    sample_dirs = strict_train_sample_dirs()
    validation_sample_dirs = strict_validation_sample_dirs()
    all_sample_dirs = generated_sample_dirs()
    min_completed = int(PIPELINE_CONFIG["training"]["min_completed_samples"])
    if len(sample_dirs) < min_completed:
        raise RuntimeError(
            f"Se necesitan al menos {min_completed} muestras completadas para entrenar. "
            "Ejecuta primero run_single_points.py para generar mas datos validos."
        )
    if not relaxed_basis_files():
        raise RuntimeError("No se encontraron ficheros .ion.xml en relaxed/")

    runs_json = write_runs_json(sample_dirs, validation_sample_dirs)
    configured_batch_size = int(PIPELINE_CONFIG["training"]["data"]["batch_size"])
    PIPELINE_CONFIG["training"]["data"]["batch_size"] = min(
        configured_batch_size,
        len(sample_dirs),
    )
    PIPELINE_CONFIG["training"]["data"]["runs_json"] = runs_json
    if PIPELINE_CONFIG["training"]["data"].get("val_runs") and not configured_val_runs_match_samples():
        PIPELINE_CONFIG["training"]["data"].pop("val_runs", None)
    material_provenance = apply_material_graph2mat_config(
        PIPELINE_CONFIG,
        base_dir=config_dir(PIPELINE_CONFIG),
        dataset_dir=DATASET_DIR,
        training_dir=TRAINING_DIR,
    )
    PIPELINE_CONFIG["_graph2mat_material_provenance"] = material_provenance
    validation_metadata = require_explicit_validation_split(PIPELINE_CONFIG)
    PIPELINE_CONFIG["_graph2mat_validation_metadata"] = validation_metadata

    if len(sample_dirs) != len(all_sample_dirs):
        print(
            "[WARN] No todas las muestras generadas estan completadas. "
            f"Se entrenara solo con {len(sample_dirs)} de {len(all_sample_dirs)}."
        )
    print(
        "[OK] Validation split conectado a Graph2Mat: "
        f"{validation_metadata['validation_source']} "
        f"({validation_metadata['validation_sample_count']} muestras)."
    )

    return render_training_config(PIPELINE_CONFIG)


def main() -> int:
    print("=== AtomDisplacement (train) ===")
    ensure_dir(TRAINING_DIR)
    CONFIG_PATH.write_text(build_config_yaml(), encoding="utf-8")
    print(f"[OK] Config escrito en {CONFIG_PATH}")
    provenance_path = write_graph2mat_config_provenance(
        CONFIG_PATH,
        PIPELINE_CONFIG.get("_graph2mat_material_provenance", {}),
        validation_metadata=PIPELINE_CONFIG.get("_graph2mat_validation_metadata"),
    )
    print(f"[OK] Provenance Graph2Mat escrito en {provenance_path}")
    fit_command = [command(PIPELINE_CONFIG, "graph2mat"), *PIPELINE_CONFIG["training"]["fit_args"]]
    run_command_in_venv(fit_command, cwd=TRAINING_DIR)
    ckpt_path = resolve_checkpoint(PIPELINE_CONFIG)
    selection_mode = "configured_path" if PIPELINE_CONFIG.get("checkpoint", {}).get("path") else str(PIPELINE_CONFIG.get("checkpoint", {}).get("selection", "latest_version"))
    manifest_path = write_checkpoint_manifest(
        PIPELINE_CONFIG,
        ckpt_path,
        selection_mode=selection_mode,
        selection_metric="val_loss",
    )
    print(f"[OK] Checkpoint manifest escrito en {manifest_path}")
    print(f"[INFO] Si quieres ver metricas: {PIPELINE_CONFIG['training']['tensorboard_hint']}")
    print("\n=== Entrenamiento completado correctamente ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
