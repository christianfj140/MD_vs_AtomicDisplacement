"""Shared configuration helpers for the MD pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from glob import glob
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on runtime environment.
    raise RuntimeError(
        "PyYAML is required to read pipeline_config.yaml. Install pyyaml in the "
        "environment used to run these scripts."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "pipeline_config.yaml"
SHARED_DIR = PROJECT_ROOT.parent / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from siesta_run_fdf import md_common_settings, render_common_run_fdf, render_md_layer
from graph2mat_material_config import (
    load_graph2mat_config_provenance,
    resolve_matrix_component_policy,
)


def load_pipeline_config(config_path: Path | None = None) -> dict[str, Any]:
    env_path = os.environ.get("PIPELINE_CONFIG_PATH")
    path = Path(env_path).expanduser() if config_path is None and env_path else config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise RuntimeError(f"No existe el archivo de configuración: {path}")

    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise RuntimeError(f"La configuración debe ser un diccionario YAML: {path}")

    config["_config_path"] = path
    config["_config_dir"] = path.parent
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = (
        "paths",
        "commands",
        "md",
        "training",
        "checkpoint",
        "testing",
        "prediction",
        "pipeline",
    )
    for section in required_sections:
        if section not in config:
            raise RuntimeError(f"Falta la sección '{section}' en pipeline_config.yaml.")

    md = config["md"]
    if md_temperature_blocks(config):
        pass
    elif int(md["steps"]) <= 0:
        raise RuntimeError("md.steps debe ser mayor que cero.")
    if len(md["lattice_vectors"]) != 3:
        raise RuntimeError("md.lattice_vectors debe contener exactamente 3 vectores.")
    if not md["species"]:
        raise RuntimeError("md.species debe contener al menos una especie.")
    if not md["atoms"]:
        raise RuntimeError("md.atoms debe contener al menos un átomo.")
    training_data = config.get("training", {}).get("data", {})
    if isinstance(training_data, dict):
        resolve_matrix_component_policy(training_data, context="training.data")


def config_dir(config: dict[str, Any]) -> Path:
    return Path(config["_config_dir"])


def expand_path_tokens(value: str | Path) -> str:
    text = str(value)
    repo_root = PROJECT_ROOT.parent
    replacements = {
        "${REPO_ROOT}": str(repo_root),
        "${GRAPH2MAT_VENV}": os.environ.get("GRAPH2MAT_VENV", ""),
    }
    for token, replacement in replacements.items():
        if replacement:
            text = text.replace(token, replacement)
    return os.path.expandvars(text)


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(expand_path_tokens(value)).expanduser()
    if path.is_absolute():
        return path
    return config_dir(config) / path


def paths(config: dict[str, Any]) -> dict[str, Path]:
    raw_paths = config["paths"]
    dataset_dir = resolve_path(config, raw_paths["dataset_dir"])
    training_dir = resolve_path(config, raw_paths["training_dir"])
    return {
        "dataset_dir": dataset_dir,
        "training_dir": training_dir,
        "run_fdf_path": dataset_dir / raw_paths["run_fdf_name"],
        "run_out_path": dataset_dir / raw_paths["run_out_name"],
        "training_config_path": training_dir / raw_paths["training_config_name"],
        "venv_activate": resolve_path(config, raw_paths["venv_activate"]),
    }


def command(config: dict[str, Any], name: str) -> str:
    return str(config["commands"][name])


def _fdf_bool(value: bool) -> str:
    return "t" if value else "f"


def _format_float(value: float) -> str:
    return f"{float(value):.8f}"


def md_temperature_blocks(config: dict[str, Any]) -> list[dict[str, Any]]:
    md = config.get("md", {}) or {}
    raw_blocks = md.get("temperature_blocks") or md.get("blocks") or []
    if raw_blocks in (None, "", []):
        return []
    if not isinstance(raw_blocks, list):
        raise RuntimeError("md.temperature_blocks debe ser una lista.")
    blocks: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_blocks, start=1):
        if not isinstance(raw, dict):
            raise RuntimeError(f"md.temperature_blocks[{index}] debe ser un objeto.")
        n_snapshots = int(raw.get("n_snapshots") or raw.get("steps") or 0)
        if n_snapshots <= 0:
            raise RuntimeError(f"md.temperature_blocks[{index}].n_snapshots debe ser > 0.")
        block = dict(raw)
        block["n_snapshots"] = n_snapshots
        if block.get("temperature_K") not in (None, ""):
            temperature = float(block["temperature_K"])
            if temperature < 0:
                raise RuntimeError(f"md.temperature_blocks[{index}].temperature_K no puede ser negativa.")
            block["temperature_K"] = temperature
        if block.get("timestep_fs") not in (None, ""):
            timestep = float(block["timestep_fs"])
            if timestep <= 0:
                raise RuntimeError(f"md.temperature_blocks[{index}].timestep_fs debe ser > 0.")
            block["timestep_fs"] = timestep
        block.setdefault("block_id", f"md_block_{index}")
        block.setdefault("label", f"{n_snapshots} snapshots")
        blocks.append(block)
    return blocks


def md_total_steps(config: dict[str, Any]) -> int:
    blocks = md_temperature_blocks(config)
    if blocks:
        return sum(int(block["n_snapshots"]) for block in blocks)
    return int(config["md"]["steps"])


def render_run_fdf(config: dict[str, Any], block: dict[str, Any] | None = None) -> str:
    md = config["md"]
    base = render_common_run_fdf(
        system_name=str(md.get("system_name", "MD dataset")),
        system_label=str((block or {}).get("system_label", md.get("system_label", "siesta"))),
        lattice_constant=md["lattice_constant"],
        lattice_vectors=md["lattice_vectors"],
        species=md["species"],
        coordinates_format=md["coordinates_format"],
        atoms=md["atoms"],
        kgrid_monkhorst_pack=md.get("kgrid_monkhorst_pack"),
        siesta_settings=md_common_settings(md),
        header="common MD/FC/Random Cartesian electronic and structural base",
    )
    return base + render_md_layer(md, block)


def render_training_config(config: dict[str, Any]) -> str:
    graph2mat_keys = (
        "data",
        "model",
        "trainer",
        "optimizer",
        "lr_scheduler",
        "seed_everything",
        "multiprocessing_sharing_strategy",
        "ckpt_path",
        "weights_only",
    )
    training_config = {
        key: config["training"][key]
        for key in graph2mat_keys
        if key in config["training"]
    }
    return "# Generated from ../pipeline_config.yaml\n" + yaml.safe_dump(
        training_config,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )


def checkpoint_version(path: Path) -> int:
    for part in path.parts:
        if part.startswith("version_"):
            suffix = part.removeprefix("version_")
            if suffix.isdigit():
                return int(suffix)
    return -1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_values(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _matched_training_paths(training_dir: Path, patterns: list[str]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        raw_path = Path(pattern)
        search = raw_path if raw_path.is_absolute() else training_dir / pattern
        matches.extend(str(path) for path in sorted(glob(str(search))))
    return matches


def validation_source_metadata(config: dict[str, Any]) -> dict[str, Any]:
    training_dir = paths(config)["training_dir"]
    data = (config.get("training", {}) or {}).get("data", {}) or {}
    val_patterns = _path_values(data.get("val_runs"))
    val_matches = _matched_training_paths(training_dir, val_patterns)
    if val_patterns:
        return {
            "validation_source": "training.data.val_runs",
            "validation_runs": val_patterns[0] if len(val_patterns) == 1 else val_patterns,
            "validation_sample_count": len(val_matches),
            "checkpoint_selection_validation_backed": bool(val_matches),
            "checkpoint_selection_metric": "val_loss",
            "checkpoint_selection_criterion": "Graph2Mat ModelCheckpoint monitor=val_loss mode=min",
        }

    runs_json = data.get("runs_json")
    if runs_json:
        json_path = Path(str(runs_json))
        if not json_path.is_absolute():
            json_path = training_dir / json_path
        runs_dict: dict[str, Any] = {}
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            runs_dict = payload if isinstance(payload, dict) else {}
        val_entries = _path_values(runs_dict.get("val"))
        return {
            "validation_source": "training.data.runs_json:val",
            "validation_runs_json": str(runs_json),
            "validation_sample_count": len(val_entries),
            "checkpoint_selection_validation_backed": bool(val_entries),
            "checkpoint_selection_metric": "val_loss",
            "checkpoint_selection_criterion": "Graph2Mat ModelCheckpoint monitor=val_loss mode=min",
        }

    return {
        "validation_source": None,
        "validation_sample_count": 0,
        "checkpoint_selection_validation_backed": False,
        "checkpoint_selection_metric": None,
        "checkpoint_selection_criterion": None,
        "checkpoint_selection_warning": "No explicit Graph2Mat validation split was configured.",
    }


def require_explicit_validation_split(config: dict[str, Any]) -> dict[str, Any]:
    metadata = validation_source_metadata(config)
    if not metadata["checkpoint_selection_validation_backed"]:
        raise RuntimeError(
            "MD strict training requires an explicit validation split for Graph2Mat "
            "checkpoint selection. Set training.data.val_runs to the generated "
            "validation split and ensure it matches at least one RUN.fdf."
        )
    return metadata


def write_checkpoint_manifest(
    config: dict[str, Any],
    checkpoint_rel_path: str,
    *,
    selection_mode: str,
    selection_metric: str | None = None,
    epoch: int | None = None,
    run_id: str | None = None,
) -> Path:
    training_dir = paths(config)["training_dir"]
    checkpoint_path = Path(checkpoint_rel_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = training_dir / checkpoint_path
    validation_metadata = validation_source_metadata(config)
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "relative_path": checkpoint_rel_path,
        "checkpoint_sha256": file_sha256(checkpoint_path) if checkpoint_path.exists() else None,
        "epoch": epoch,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "training_dir": str(training_dir),
        **validation_metadata,
    }
    provenance = load_graph2mat_config_provenance(training_dir)
    if provenance:
        payload["graph2mat_config_provenance"] = provenance
    manifest_path = training_dir / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def resolve_checkpoint(config: dict[str, Any]) -> str:
    pipeline_paths = paths(config)
    training_dir = pipeline_paths["training_dir"]
    checkpoint_config = config["checkpoint"]
    configured_path = checkpoint_config.get("path")

    if configured_path:
        configured_path = str(configured_path)
        configured_abs = training_dir / configured_path
        if configured_abs.exists():
            return configured_path

    if bool(checkpoint_config.get("auto_best", True)):
        candidates = sorted(training_dir.glob(str(checkpoint_config["search_glob"])))
        if candidates:
            selection = str(checkpoint_config.get("selection", "latest_version"))
            if selection != "latest_version":
                raise RuntimeError(
                    "checkpoint.selection solo soporta actualmente "
                    "el valor 'latest_version'."
                )

            latest_version = max(checkpoint_version(path) for path in candidates)
            latest_candidates = [
                path for path in candidates if checkpoint_version(path) == latest_version
            ]
            if len(latest_candidates) != 1:
                rel_candidates = "\n".join(
                    f"  - {path.relative_to(training_dir).as_posix()}"
                    for path in latest_candidates
                )
                raise RuntimeError(
                    "Se encontró más de un checkpoint best-*.ckpt dentro de la "
                    f"última versión version_{latest_version}. Define "
                    "checkpoint.path en pipeline_config.yaml con uno de estos "
                    f"valores:\n{rel_candidates}"
                )

            selected = latest_candidates[0]
            rel_path = selected.relative_to(training_dir).as_posix()
            print(
                "[INFO] Se usará automáticamente el checkpoint de la versión "
                f"más nueva: {rel_path}"
            )
            return rel_path

    raise RuntimeError(
        "No se encontró ningún checkpoint best-*.ckpt válido. Ajusta "
        "checkpoint.path o checkpoint.search_glob en pipeline_config.yaml."
    )
