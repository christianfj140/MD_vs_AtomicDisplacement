"""Shared configuration helpers for the AtomDisplacement pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
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
GENERATED_HEADER = "# Generated from ../pipeline_config.yaml\n"
SHARED_DIR = PROJECT_ROOT.parent / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from siesta_run_fdf import render_common_run_fdf, render_fc_layer
from graph2mat_material_config import load_graph2mat_config_provenance


def load_pipeline_config(config_path: Path | None = None) -> dict[str, Any]:
    env_path = os.environ.get("PIPELINE_CONFIG_PATH")
    path = Path(env_path).expanduser() if config_path is None and env_path else config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise RuntimeError(f"No existe el archivo de configuracion: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise RuntimeError(f"La configuracion debe ser un diccionario YAML: {path}")
    config["_config_path"] = path
    config["_config_dir"] = path.parent
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = (
        "paths",
        "commands",
        "generation",
        "structure",
        "training",
        "checkpoint",
        "testing",
        "prediction",
        "pipeline",
    )
    for section in required:
        if section not in config:
            raise RuntimeError(f"Falta la seccion '{section}' en pipeline_config.yaml.")
    if len(config["structure"]["lattice_vectors"]) != 3:
        raise RuntimeError("structure.lattice_vectors debe contener exactamente 3 vectores.")
    if not config["structure"]["species"]:
        raise RuntimeError("structure.species debe contener al menos una especie.")
    if not config["structure"]["atoms"]:
        raise RuntimeError("structure.atoms debe contener al menos un atomo.")
    if not config["generation"].get("sample_id_format"):
        raise RuntimeError("generation.sample_id_format no puede estar vacio.")
    force_constants = config["structure"].get("force_constants", {})
    if force_constants and int(force_constants.get("first_atom", 1)) <= 0:
        raise RuntimeError("structure.force_constants.first_atom debe ser mayor que cero.")


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
    raw = config["paths"]
    base_dir = resolve_path(config, raw["base_dir"])
    relaxed_dir = resolve_path(config, raw["relaxed_dir"])
    dataset_dir = resolve_path(config, raw["dataset_dir"])
    samples_dir = resolve_path(config, raw["samples_dir"])
    collected_dir = resolve_path(config, raw["collected_dir"])
    training_dir = resolve_path(config, raw["training_dir"])
    return {
        "base_dir": base_dir,
        "relaxed_dir": relaxed_dir,
        "dataset_dir": dataset_dir,
        "samples_dir": samples_dir,
        "collected_dir": collected_dir,
        "training_dir": training_dir,
        "base_run_fdf_path": base_dir / raw["run_fdf_name"],
        "relaxed_run_fdf_path": relaxed_dir / raw["run_fdf_name"],
        "relaxed_run_out_path": relaxed_dir / raw["run_out_name"],
        "training_config_path": training_dir / raw["training_config_name"],
        "runs_json_path": training_dir / raw["runs_json_name"],
        "samples_manifest_path": dataset_dir / raw["samples_manifest_name"],
        "run_summary_path": dataset_dir / raw["run_summary_name"],
        "collected_json_path": collected_dir / raw["collected_json_name"],
        "collected_csv_path": collected_dir / raw["collected_csv_name"],
        "venv_activate": resolve_path(config, raw["venv_activate"]),
    }


def command(config: dict[str, Any], name: str) -> str:
    return str(config["commands"][name])


def _format_float(value: Any) -> str:
    return f"{float(value):.8f}"


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    return str(value)


def _species_map(config: dict[str, Any]) -> dict[int, tuple[int, str]]:
    return {
        int(item["index"]): (int(item["atomic_number"]), str(item["symbol"]))
        for item in config["structure"]["species"]
    }


def render_fdf(
    config: dict[str, Any],
    *,
    positions_ang: list[list[float]] | None = None,
    atom_species: list[int] | None = None,
    system_label: str | None = None,
    system_name: str | None = None,
    include_relaxation: bool,
    header: str,
) -> str:
    structure = config["structure"]
    species = _species_map(config)
    atoms = structure["atoms"]
    positions = positions_ang or [atom["position"] for atom in atoms]
    atom_species = atom_species or [int(atom["species_index"]) for atom in atoms]
    system_label = system_label or structure["relaxation"]["system_label"]
    system_name = system_name or structure["relaxation"]["system_name"]

    siesta = dict(structure["siesta"])
    if not include_relaxation:
        siesta.update(structure.get("single_point_overrides", {}))
    text = render_common_run_fdf(
        system_name=system_name,
        system_label=system_label,
        lattice_constant=structure["lattice_constant"],
        lattice_vectors=structure["lattice_vectors"],
        species=structure["species"],
        coordinates_format=structure["coordinates_format"],
        positions=positions,
        atom_species=atom_species,
        kgrid_monkhorst_pack=structure.get("kgrid_monkhorst_pack"),
        siesta_settings=siesta,
        header=header,
    )
    lines = text.rstrip().splitlines()
    if include_relaxation:
        lines.append("")
        for key, value in structure["relaxation_md"].items():
            lines.append(f"{key:<32} {_format_value(value)}")
    elif bool(structure.get("force_constants", {}).get("enabled", False)):
        lines.extend(render_fc_layer(dict(structure["force_constants"]), len(atom_species)).splitlines())
    return "\n".join(lines) + "\n"


def render_relaxation_fdf(config: dict[str, Any]) -> str:
    return render_fdf(
        config,
        include_relaxation=True,
        header="Relaxation of an isolated water molecule with SIESTA",
    )


def render_single_point_fdf(
    config: dict[str, Any],
    positions_ang: list[list[float]],
    atom_species: list[int],
    sample_id: str,
) -> str:
    single_point = config["structure"]["single_point"]
    return render_fdf(
        config,
        positions_ang=positions_ang,
        atom_species=atom_species,
        system_label=sample_id,
        system_name=single_point["system_name_template"].format(sample_id=sample_id),
        include_relaxation=False,
        header=single_point["title"],
    )


def render_training_config(config: dict[str, Any]) -> str:
    training_config = {
        "data": config["training"]["data"],
        "model": config["training"]["model"],
        "trainer": config["training"]["trainer"],
    }
    return GENERATED_HEADER + yaml.safe_dump(
        training_config,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )


def write_generated_inputs(config: dict[str, Any]) -> None:
    pipeline_paths = paths(config)
    pipeline_paths["base_dir"].mkdir(parents=True, exist_ok=True)
    pipeline_paths["relaxed_dir"].mkdir(parents=True, exist_ok=True)
    pipeline_paths["training_dir"].mkdir(parents=True, exist_ok=True)
    content = render_relaxation_fdf(config)
    pipeline_paths["base_run_fdf_path"].write_text(content, encoding="utf-8")
    pipeline_paths["relaxed_run_fdf_path"].write_text(content, encoding="utf-8")
    pipeline_paths["training_config_path"].write_text(
        render_training_config(config),
        encoding="utf-8",
    )


def checkpoint_version(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"version_(\d+)", part)
        if match:
            return int(match.group(1))
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
    if val_patterns and val_matches:
        return {
            "validation_source": "training.data.val_runs",
            "validation_runs": val_patterns[0] if len(val_patterns) == 1 else val_patterns,
            "validation_sample_count": len(val_matches),
            "checkpoint_selection_validation_backed": True,
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
        if val_entries:
            return {
                "validation_source": "training.data.runs_json:val",
                "validation_runs_json": str(runs_json),
                "validation_sample_count": len(val_entries),
                "checkpoint_selection_validation_backed": True,
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
            "AtomDisplacement strict training requires an explicit validation split "
            "for Graph2Mat checkpoint selection. Provide training.data.val_runs or "
            "a runs_json file with a non-empty 'val' split."
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
    training_dir = paths(config)["training_dir"]
    checkpoint_config = config["checkpoint"]
    configured_path = checkpoint_config.get("path")
    if configured_path:
        configured_path = str(configured_path)
        configured_abs = Path(configured_path)
        if not configured_abs.is_absolute():
            configured_abs = training_dir / configured_abs
        if configured_abs.exists():
            return configured_path
        raise RuntimeError(f"checkpoint.path no existe: {configured_abs}")

    if bool(checkpoint_config.get("auto_best", True)):
        candidates = sorted(training_dir.glob(str(checkpoint_config["search_glob"])))
        if candidates:
            if str(checkpoint_config.get("selection", "latest_version")) != "latest_version":
                raise RuntimeError("checkpoint.selection solo soporta 'latest_version'.")
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
                    "Se encontro mas de un checkpoint best-*.ckpt dentro de "
                    f"version_{latest_version}. Define checkpoint.path con uno "
                    f"de estos valores:\n{rel_candidates}"
                )
            selected = latest_candidates[0].relative_to(training_dir).as_posix()
            print(f"[INFO] Checkpoint seleccionado automaticamente: {selected}")
            return selected

    raise RuntimeError(
        "No se encontro ningun checkpoint best-*.ckpt valido. Ajusta "
        "checkpoint.path o checkpoint.search_glob en pipeline_config.yaml."
    )
