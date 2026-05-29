#!/usr/bin/env python3
"""Training sweep expansion for the Graph2Mat-vs-DeepH benchmark."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graph2mat_sweep_config import GRAPH2MAT_SWEEP_KEYS, normalize_graph2mat_overrides


COMMON_KEYS = {"seeds", "seed", "epochs", "learning_rate", "batch_size"}
GRAPH2MAT_KEYS = {"enabled", *GRAPH2MAT_SWEEP_KEYS}
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
SEARCH_PLAN_SCHEMA = "graph2mat_deeph_training_search_plan_v1"
ALLOWED_SEARCH_STRATEGIES = {"grid", "random", "latin_hypercube"}


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


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(
        json.dumps(json_safe(parts), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


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


def _distribution_spec(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, dict):
        if "choices" in value:
            choices = _as_list(value.get("choices"), field=f"{field}.choices")
            return {"kind": "categorical", "choices": choices}
        if "value" in value:
            return {"kind": "fixed", "value": value["value"]}
        if "fixed" in value:
            return {"kind": "fixed", "value": value["fixed"]}
        dist = str(value.get("distribution") or value.get("type") or "").strip().lower()
        if dist in {"loguniform", "log_uniform"}:
            low = value.get("min", value.get("low"))
            high = value.get("max", value.get("high"))
            if isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise RuntimeError(f"{field} loguniform requires numeric min and max.")
            if float(low) <= 0 or float(high) <= 0 or float(low) >= float(high):
                raise RuntimeError(f"{field} loguniform requires 0 < min < max.")
            return {"kind": "loguniform", "min": float(low), "max": float(high)}
        if dist in {"uniform", "float"}:
            low = value.get("min", value.get("low"))
            high = value.get("max", value.get("high"))
            if isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise RuntimeError(f"{field} uniform requires numeric min and max.")
            if float(low) >= float(high):
                raise RuntimeError(f"{field} uniform requires min < max.")
            return {"kind": "uniform", "min": float(low), "max": float(high)}
        if dist in {"int", "integer", "randint"}:
            low = value.get("min", value.get("low"))
            high = value.get("max", value.get("high"))
            if isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, int) or not isinstance(high, int):
                raise RuntimeError(f"{field} integer range requires integer min and max.")
            if low > high:
                raise RuntimeError(f"{field} integer range requires min <= max.")
            return {"kind": "integer", "min": int(low), "max": int(high)}
        raise RuntimeError(
            f"{field} has unsupported distribution. Use choices, value/fixed, "
            "loguniform, uniform, or integer."
        )
    if isinstance(value, list):
        return {"kind": "categorical", "choices": _as_list(value, field=field)}
    return {"kind": "fixed", "value": value}


def _sample_from_spec(spec: dict[str, Any], *, rng: random.Random, u: float | None = None) -> Any:
    kind = spec["kind"]
    if kind == "fixed":
        return spec["value"]
    if u is None:
        u = rng.random()
    u = min(max(float(u), 0.0), math.nextafter(1.0, 0.0))
    if kind == "categorical":
        choices = list(spec["choices"])
        return choices[min(int(u * len(choices)), len(choices) - 1)]
    if kind == "integer":
        low = int(spec["min"])
        high = int(spec["max"])
        return low + min(int(u * (high - low + 1)), high - low)
    if kind == "uniform":
        return float(spec["min"]) + u * (float(spec["max"]) - float(spec["min"]))
    if kind == "loguniform":
        log_low = math.log(float(spec["min"]))
        log_high = math.log(float(spec["max"]))
        return math.exp(log_low + u * (log_high - log_low))
    raise RuntimeError(f"Unsupported distribution kind: {kind}")


def _sample_space(
    section: dict[str, Any],
    *,
    allowed: set[str],
    section_name: str,
    strategy: str,
    n_trials: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise RuntimeError(f"Unsupported training_sweep.{section_name} keys: {', '.join(unknown)}.")
    dimensions = [key for key in sorted(section) if key != "enabled"]
    specs = {key: _distribution_spec(section[key], field=f"training_sweep.{section_name}.{key}") for key in dimensions}
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    if not dimensions:
        return ([{} for _ in range(n_trials)], [])
    if strategy == "random":
        for _ in range(n_trials):
            rows.append({key: _sample_from_spec(specs[key], rng=rng) for key in dimensions})
    elif strategy == "latin_hypercube":
        strata_by_key: dict[str, list[float]] = {}
        for key in dimensions:
            values = [(index + rng.random()) / n_trials for index in range(n_trials)]
            rng.shuffle(values)
            strata_by_key[key] = values
        for trial in range(n_trials):
            rows.append(
                {
                    key: _sample_from_spec(specs[key], rng=rng, u=strata_by_key[key][trial])
                    for key in dimensions
                }
            )
    else:
        raise RuntimeError(f"Unsupported sampled search strategy: {strategy}")
    spec_report = [{"name": key, **json_safe(specs[key])} for key in dimensions]
    return rows, spec_report


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


def _normalize_common_keys(common: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(common) - COMMON_KEYS)
    if unknown:
        raise RuntimeError(f"Unsupported training_sweep.common keys: {', '.join(unknown)}.")
    normalized: dict[str, Any] = {}
    for key, value in common.items():
        target = "seed" if key == "seeds" else key
        if target in normalized:
            raise RuntimeError("Use either training_sweep.common.seed or seeds, not both.")
        normalized[target] = value
    return normalized


def _sample_common(common: dict[str, Any], *, strategy: str, n_trials: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = _normalize_common_keys(common)
    return _sample_space(
        normalized,
        allowed={"seed", "epochs", "learning_rate", "batch_size"},
        section_name="common",
        strategy=strategy,
        n_trials=n_trials,
        seed=seed,
    )


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


def _search_policy(value: dict[str, Any]) -> dict[str, Any]:
    policy = value.get("search_policy") or {}
    if policy in (None, ""):
        policy = {}
    if not isinstance(policy, dict):
        raise RuntimeError("training_sweep.search_policy must be an object.")
    strategy = str(policy.get("strategy") or "grid").strip().lower()
    if strategy not in ALLOWED_SEARCH_STRATEGIES:
        raise RuntimeError(
            "training_sweep.search_policy.strategy must be one of: "
            + ", ".join(sorted(ALLOWED_SEARCH_STRATEGIES))
            + "."
        )
    seed = int(policy.get("random_seed", policy.get("search_seed", value.get("search_seed", 0))) or 0)
    n_trials = policy.get("n_trials_per_model") or value.get("n_trials_per_model")
    if strategy in {"random", "latin_hypercube"}:
        if n_trials in (None, ""):
            raise RuntimeError("training_sweep.search_policy.n_trials_per_model is required for sampled search.")
        n_trials = int(n_trials)
        if n_trials <= 0:
            raise RuntimeError("training_sweep.search_policy.n_trials_per_model must be positive.")
    return {
        "strategy": strategy,
        "random_seed": seed,
        "n_trials_per_model": int(n_trials) if n_trials not in (None, "") else None,
    }


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


def _planned_record(
    *,
    index: int,
    model_name: str,
    dataset: SweepDataset,
    common_combo: dict[str, Any],
    model_combo: dict[str, Any],
    config_id: str,
    config_hash: str,
    search_strategy: str,
    search_seed: int | None = None,
    search_trial_index: int | None = None,
    protocol_id: str | None = None,
    protocol_hash: str | None = None,
    duplicate_config: bool = False,
    duplicate_of_config_id: str = "",
) -> dict[str, Any]:
    overrides = (
        _graph2mat_overrides(common_combo, model_combo)
        if model_name == "graph2mat"
        else _deeph_overrides(common_combo, model_combo)
    )
    row = {
        "index": index,
        "model": model_name,
        "dataset_id": dataset.dataset_id,
        "dataset_root": str(dataset.dataset_root),
        "config_id": config_id,
        "config_label": config_id,
        "config_hash": config_hash,
        "common": common_combo,
        "overrides": overrides,
        "search_strategy": search_strategy,
        "status": "planned",
    }
    if search_seed is not None:
        row["search_seed"] = search_seed
    if search_trial_index is not None:
        row["search_trial_index"] = search_trial_index
    if protocol_id:
        row["protocol_id"] = protocol_id
    if protocol_hash:
        row["protocol_hash"] = protocol_hash
    if duplicate_config:
        row["duplicate_config"] = True
        row["duplicate_of_config_id"] = duplicate_of_config_id
    return row


def _search_plan(
    *,
    strategy: str,
    random_seed: int | None,
    n_trials_per_model: int | None,
    protocol_id: str,
    protocol_hash: str,
    planned: list[dict[str, Any]],
    duplicate_configs: list[dict[str, Any]],
    sampled_dimensions: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SEARCH_PLAN_SCHEMA,
        "strategy": strategy,
        "random_seed": random_seed,
        "n_trials_per_model": n_trials_per_model,
        "protocol_id": protocol_id,
        "protocol_hash": protocol_hash,
        "planned_run_count": len(planned),
        "duplicate_config_count": len(duplicate_configs),
        "duplicate_configs": duplicate_configs,
        "sampled_dimensions": sampled_dimensions or {},
        "planned_runs": planned,
    }


def training_sweep_from_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Convert a validated final benchmark protocol into a training_sweep payload."""

    models = protocol.get("models") if isinstance(protocol.get("models"), dict) else {}
    search_policy = dict(protocol.get("search_policy") or {})
    return {
        "enabled": True,
        "protocol_id": str(protocol.get("protocol_id") or ""),
        "protocol_hash": str(protocol.get("protocol_hash") or ""),
        "search_policy": search_policy,
        "budget_policy": dict(protocol.get("budget_policy") or {}),
        "common": {},
        "graph2mat": {
            "enabled": True,
            **dict((models.get("graph2mat") or {}).get("search_space") or {}),
        },
        "deeph": {
            "enabled": True,
            **dict((models.get("deeph") or {}).get("search_space") or {}),
        },
        "max_runs": int(search_policy.get("max_runs") or 100000),
    }


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
    unknown_top = sorted(
        set(value)
        - {
            "enabled",
            "max_runs",
            "n_trials_per_model",
            "search_seed",
            "search_policy",
            "budget_policy",
            "protocol_id",
            "protocol_hash",
            "apply_to_datasets",
            "error_policy",
            "common",
            "graph2mat",
            "deeph",
        }
    )
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

    policy = _search_policy(value)
    budget_policy = value.get("budget_policy") if isinstance(value.get("budget_policy"), dict) else {}
    strategy = str(policy["strategy"])
    search_seed = int(policy["random_seed"])
    n_trials_per_model = policy.get("n_trials_per_model")
    protocol_id = str(value.get("protocol_id") or "")
    protocol_hash = str(value.get("protocol_hash") or "")

    common_combos = _common_grid(common) if strategy == "grid" else []
    graph2mat_combos = (
        _grid(
            {key: value for key, value in graph2mat.items() if key != "enabled"},
            allowed=GRAPH2MAT_KEYS - {"enabled"},
            section_name="graph2mat",
        )
        if strategy == "grid" and _enabled(graph2mat, True)
        else []
    )
    deeph_combos = (
        _grid(
            {key: value for key, value in deeph.items() if key != "enabled"},
            allowed=DEEPH_KEYS - {"enabled"},
            section_name="deeph",
        )
        if strategy == "grid" and _enabled(deeph, True)
        else []
    )
    if not graph2mat_combos and not deeph_combos:
        if strategy == "grid":
            raise RuntimeError("training_sweep must enable Graph2Mat or DeepH.")

    planned: list[dict[str, Any]] = []
    seen_configs: set[str] = set()
    duplicate_configs: list[dict[str, Any]] = []
    sampled_dimensions: dict[str, list[dict[str, Any]]] = {}
    if strategy == "grid":
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
                            _planned_record(
                                index=len(planned) + 1,
                                model_name=model_name,
                                dataset=dataset,
                                common_combo=common_combo,
                                model_combo=model_combo,
                                config_id=config_id,
                                config_hash=config_hash,
                                search_strategy=strategy,
                                protocol_id=protocol_id,
                                protocol_hash=protocol_hash,
                            )
                        )
    else:
        if n_trials_per_model is None:
            raise RuntimeError("n_trials_per_model is required for sampled search.")
        for model_name, model_section, allowed in (
            ("graph2mat", graph2mat, GRAPH2MAT_KEYS - {"enabled"}),
            ("deeph", deeph, DEEPH_KEYS - {"enabled"}),
        ):
            if not _enabled(model_section, True):
                continue
            model_seed = _stable_seed(search_seed, model_name, protocol_id, protocol_hash)
            common_samples, common_report = _sample_common(
                common,
                strategy=strategy,
                n_trials=n_trials_per_model,
                seed=_stable_seed(model_seed, "common"),
            )
            model_samples, model_report = _sample_space(
                {key: item for key, item in model_section.items() if key != "enabled"},
                allowed=allowed,
                section_name=model_name,
                strategy=strategy,
                n_trials=n_trials_per_model,
                seed=_stable_seed(model_seed, "model"),
            )
            sampled_dimensions[f"{model_name}.common"] = common_report
            sampled_dimensions[model_name] = model_report
            config_hash_seen_for_model: dict[str, str] = {}
            for trial_index in range(n_trials_per_model):
                common_combo = common_samples[trial_index]
                model_combo = model_samples[trial_index]
                for dataset in parsed_datasets:
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
                    duplicate_of = config_hash_seen_for_model.get(config_hash, "")
                    duplicate = bool(duplicate_of)
                    config_id = f"{model_name}_{config_hash}"
                    if duplicate:
                        config_id = f"{config_id}_dup{trial_index + 1}"
                        duplicate_configs.append(
                            {
                                "model": model_name,
                                "dataset_id": dataset.dataset_id,
                                "trial_index": trial_index + 1,
                                "config_hash": config_hash,
                                "config_id": config_id,
                                "duplicate_of_config_id": duplicate_of,
                            }
                        )
                    else:
                        config_hash_seen_for_model[config_hash] = config_id
                    planned.append(
                        _planned_record(
                            index=len(planned) + 1,
                            model_name=model_name,
                            dataset=dataset,
                            common_combo=common_combo,
                            model_combo=model_combo,
                            config_id=config_id,
                            config_hash=config_hash,
                            search_strategy=strategy,
                            search_seed=search_seed,
                            search_trial_index=trial_index + 1,
                            protocol_id=protocol_id,
                            protocol_hash=protocol_hash,
                            duplicate_config=duplicate,
                            duplicate_of_config_id=duplicate_of,
                        )
                    )
    if strategy != "grid" and not planned:
        raise RuntimeError("training_sweep must enable Graph2Mat or DeepH.")
    max_runs = int(value.get("max_runs") or 128)
    if max_runs <= 0:
        raise RuntimeError("training_sweep.max_runs must be positive.")
    if len(planned) > max_runs:
        raise RuntimeError(f"training_sweep planned {len(planned)} runs, above max_runs={max_runs}.")
    return {
        "enabled": True,
        "max_runs": max_runs,
        "error_policy": str(value.get("error_policy") or "continue_on_error"),
        "search_policy": policy,
        "budget_policy": budget_policy,
        "search_plan": _search_plan(
            strategy=strategy,
            random_seed=search_seed,
            n_trials_per_model=n_trials_per_model,
            protocol_id=protocol_id,
            protocol_hash=protocol_hash,
            planned=planned,
            duplicate_configs=duplicate_configs,
            sampled_dimensions=sampled_dimensions,
        ),
        "planned_runs": planned,
    }
