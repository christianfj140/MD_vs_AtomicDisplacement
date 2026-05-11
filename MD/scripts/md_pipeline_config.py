"""Shared configuration helpers for the MD pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import sys
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
    }
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
