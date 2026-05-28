#!/usr/bin/env python3
"""Training sweep expansion for the Graph2Mat-vs-DeepH benchmark."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graph2mat_sweep_config import normalize_graph2mat_overrides


COMMON_KEYS = {"seeds", "seed", "epochs", "learning_rate", "batch_size"}
GRAPH2MAT_KEYS = {
    "enabled",
    "max_epochs",
    "optim_lr",
    "batch_size",
    "loader_threads",
    "seed_everything",
    "loss",
    "loss_kwargs",
    "num_interactions",
    "correlation",
    "max_ell",
    "hidden_irreps",
    "hidden_irreps_channels",
}
DEEPH_KEYS = {
    "enabled",
    "epochs",
    "learning_rate",
    "batch_size",
    "seed",
    "optimizer",
    "weight_decay",
    "criterion",
    "atom_fea_len",
    "edge_fea_len",
    "gauss_stop",
    "num_l",
    "if_edge_update",
    "if_lcmp",
    "normalization",
    "atom_update_net",
    "retain_edge_fea",
}
FORBIDDEN_DEEPH_KEYS = {
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "local_coordinate",
    "radius",
    "orbital",
    "O_component",
    "target",
    "interface",
    "MeshCutoff",
    "basis",
    "pseudopotential",
    "k_grid",
    "spin",
}


@dataclass(frozen=True)
class SweepDataset:
    dataset_id: str
    dataset_root: Path


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _as_list(value: Any, *, field: str) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, tuple):
        raw = list(value)
    else:
        raw = [value]
    values: list[Any] = []
    seen: set[str] = set()
    for item in raw:
        marker = json.dumps(json_safe(item), sort_keys=True, ensure_ascii=False)
        if marker in seen:
            continue
        seen.add(marker)
        values.append(item)
    if not values:
        raise RuntimeError(f"training_sweep.{field} has no usable values.")
    return values


def _grid(section: dict[str, Any], *, allowed: set[str], section_name: str) -> list[dict[str, Any]]:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise RuntimeError(f"Unsupported training_sweep.{section_name} keys: {', '.join(unknown)}.")
    keys = [key for key in sorted(section) if key != "enabled" and _as_list(section.get(key), field=f"{section_name}.{key}")]
    if not keys:
        return [{}]
    values = [_as_list(section[key], field=f"{section_name}.{key}") for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _common_grid(common: dict[str, Any]) -> list[dict[str, Any]]:
    unknown = sorted(set(common) - COMMON_KEYS)
    if unknown:
        raise RuntimeError(f"Unsupported training_sweep.common keys: {', '.join(unknown)}.")
    normalized: dict[str, Any] = {}
    for key, value in common.items():
        target = "seed" if key == "seeds" else key
        if target in normalized:
            raise RuntimeError("Use either training_sweep.common.seed or seeds, not both.")
        normalized[target] = value
    return _grid(normalized, allowed={"seed", "epochs", "learning_rate", "batch_size"}, section_name="common")


def _graph2mat_overrides(common: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    overrides = dict(model)
    for source, target in (
        ("epochs", "max_epochs"),
        ("learning_rate", "optim_lr"),
        ("batch_size", "batch_size"),
        ("seed", "seed_everything"),
    ):
        if source in common and target not in overrides:
            overrides[target] = common[source]
    return normalize_graph2mat_overrides(overrides)


def _deeph_overrides(common: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    forbidden = sorted(set(model) & FORBIDDEN_DEEPH_KEYS)
    if forbidden:
        raise RuntimeError(
            "DeepH training sweep cannot change split/preprocess/physics keys: "
            + ", ".join(forbidden)
        )
    overrides = dict(model)
    for key in ("epochs", "learning_rate", "batch_size", "seed"):
        if key in common and key not in overrides:
            overrides[key] = common[key]
    return overrides


def _enabled(section: dict[str, Any], default: bool = True) -> bool:
    raw = section.get("enabled", default)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on", "si", "sí"}
    return bool(raw)


def parse_datasets(datasets: list[dict[str, Any]]) -> list[SweepDataset]:
    parsed: list[SweepDataset] = []
    seen: set[str] = set()
    for index, item in enumerate(datasets):
        dataset_id = str(item.get("dataset_id") or item.get("id") or f"dataset_{index + 1}").strip()
        if not dataset_id:
            raise RuntimeError("training_sweep dataset_id cannot be empty.")
        if dataset_id in seen:
            raise RuntimeError(f"Duplicate training_sweep dataset_id: {dataset_id}")
        seen.add(dataset_id)
        root = Path(str(item.get("dataset_root") or item.get("root") or ""))
        if not str(root):
            raise RuntimeError(f"training_sweep dataset {dataset_id} is missing dataset_root.")
        parsed.append(SweepDataset(dataset_id=dataset_id, dataset_root=root))
    return parsed


def expand_training_sweep(value: Any, *, datasets: list[dict[str, Any]]) -> dict[str, Any]:
    if value in (None, "", False):
        return {"enabled": False, "planned_runs": []}
    if not isinstance(value, dict):
        raise RuntimeError("training_sweep must be an object.")
    if not _enabled(value, False):
        return {"enabled": False, "planned_runs": []}
    parsed_datasets = parse_datasets(datasets)
    if not parsed_datasets:
        raise RuntimeError("training_sweep requires at least one validated dataset.")
    apply_to = value.get("apply_to_datasets") or ["all"]
    if apply_to != ["all"]:
        wanted = {str(item) for item in _as_list(apply_to, field="apply_to_datasets")}
        parsed_datasets = [dataset for dataset in parsed_datasets if dataset.dataset_id in wanted]
        missing = sorted(wanted - {dataset.dataset_id for dataset in parsed_datasets})
        if missing:
            raise RuntimeError(f"training_sweep.apply_to_datasets contains unknown datasets: {', '.join(missing)}")
    common = value.get("common") or {}
    graph2mat = value.get("graph2mat") or {}
    deeph = value.get("deeph") or {}
    if not isinstance(common, dict) or not isinstance(graph2mat, dict) or not isinstance(deeph, dict):
        raise RuntimeError("training_sweep.common, graph2mat and deeph must be objects.")
    unknown_top = sorted(set(value) - {"enabled", "max_runs", "apply_to_datasets", "error_policy", "common", "graph2mat", "deeph"})
    if unknown_top:
        raise RuntimeError(f"Unsupported training_sweep keys: {', '.join(unknown_top)}.")
    graph2mat_unknown = sorted(set(graph2mat) - GRAPH2MAT_KEYS)
    if graph2mat_unknown:
        raise RuntimeError(f"Unsupported training_sweep.graph2mat keys: {', '.join(graph2mat_unknown)}.")
    deeph_forbidden = sorted(set(deeph) & FORBIDDEN_DEEPH_KEYS)
    if deeph_forbidden:
        raise RuntimeError(
            "DeepH training sweep cannot change split/preprocess/physics keys: "
            + ", ".join(deeph_forbidden)
        )
    deeph_unknown = sorted(set(deeph) - DEEPH_KEYS - FORBIDDEN_DEEPH_KEYS)
    if deeph_unknown:
        raise RuntimeError(f"Unsupported training_sweep.deeph keys: {', '.join(deeph_unknown)}.")

    common_combos = _common_grid(common)
    graph2mat_combos = _grid(
        {key: value for key, value in graph2mat.items() if key != "enabled"},
        allowed=GRAPH2MAT_KEYS - {"enabled"},
        section_name="graph2mat",
    ) if _enabled(graph2mat, True) else []
    deeph_combos = _grid(
        {key: value for key, value in deeph.items() if key != "enabled"},
        allowed=DEEPH_KEYS - {"enabled"},
        section_name="deeph",
    ) if _enabled(deeph, True) else []
    if not graph2mat_combos and not deeph_combos:
        raise RuntimeError("training_sweep must enable Graph2Mat or DeepH.")

    planned: list[dict[str, Any]] = []
    seen_configs: set[str] = set()
    for dataset in parsed_datasets:
        for common_combo in common_combos:
            for model_name, combos in (("graph2mat", graph2mat_combos), ("deeph", deeph_combos)):
                for model_combo in combos:
                    overrides = (
                        _graph2mat_overrides(common_combo, model_combo)
                        if model_name == "graph2mat"
                        else _deeph_overrides(common_combo, model_combo)
                    )
                    identity = {
                        "model": model_name,
                        "dataset_id": dataset.dataset_id,
                        "common": common_combo,
                        "overrides": overrides,
                    }
                    config_hash = stable_hash(identity)
                    if config_hash in seen_configs:
                        continue
                    seen_configs.add(config_hash)
                    config_id = f"{model_name}_{config_hash}"
                    planned.append(
                        {
                            "index": len(planned) + 1,
                            "model": model_name,
                            "dataset_id": dataset.dataset_id,
                            "dataset_root": str(dataset.dataset_root),
                            "config_id": config_id,
                            "config_label": config_id,
                            "config_hash": config_hash,
                            "common": common_combo,
                            "overrides": overrides,
                            "status": "planned",
                        }
                    )
    max_runs = int(value.get("max_runs") or 128)
    if max_runs <= 0:
        raise RuntimeError("training_sweep.max_runs must be positive.")
    if len(planned) > max_runs:
        raise RuntimeError(f"training_sweep planned {len(planned)} runs, above max_runs={max_runs}.")
    return {
        "enabled": True,
        "max_runs": max_runs,
        "error_policy": str(value.get("error_policy") or "continue_on_error"),
        "planned_runs": planned,
    }
