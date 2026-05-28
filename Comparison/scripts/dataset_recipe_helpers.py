"""Shared dataset recipe helpers for Comparison workflows.

The helpers here intentionally cover only recipe parsing/planning. They do not
run SIESTA and they do not know about the web server state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


FORBIDDEN_MD_SWEEP_FIELDS = {
    "meshcutoff",
    "mesh_cutoff",
    "basis",
    "basis_size",
    "basis_type",
    "pseudopotential",
    "pseudopotentials",
    "pseudo",
    "pseudos",
    "xc",
    "xc_functional",
    "xc_authors",
    "kgrid",
    "kgrid_monkhorst_pack",
    "kgrid.monkhorstpack",
    "spin",
    "spin_polarized",
    "fix_spin",
    "non_collinear_spin",
    "dm_tolerance",
    "dm_mixing_weight",
    "dm_number_pulay",
    "max_scf_iterations",
    "solution_method",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def stable_payload_hash(payload: Any, *, length: int = 12) -> str:
    text = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def recipe_set_hash(recipes: dict[str, Any] | list[Any] | None) -> str:
    return stable_payload_hash(recipes or {}, length=16)


def slugify_label(value: Any, default: str = "dataset") -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
        ".": "p",
        "+": "plus",
        "-": "m",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    slug = "".join(char if char.isalnum() else "_" for char in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or default


def optional_seed(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, float) and not value.is_integer():
        raise RuntimeError("seed debe ser un entero >= 0.")
    seed = int(value)
    if seed < 0:
        raise RuntimeError("seed debe ser un entero >= 0.")
    return seed


def validate_split_sizes(
    dataset_size: int,
    splits: dict[str, float],
    *,
    label: str = "dataset",
) -> dict[str, int]:
    if dataset_size < 3:
        raise RuntimeError(
            f"{label}: el dataset debe tener al menos 3 estructuras para que train, "
            "validation y test no queden vacios."
        )
    ratios = {
        "train": float(splits["train"]),
        "validation": float(splits.get("validation", splits.get("val", 0.0))),
        "test": float(splits["test"]),
    }
    if any(value <= 0 for value in ratios.values()):
        raise RuntimeError("Los ratios de split deben ser positivos.")
    total_ratio = sum(ratios.values())
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(f"Los ratios de split deben sumar 1.0 (recibido: {total_ratio:.6g}).")

    raw = {key: dataset_size * ratio for key, ratio in ratios.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = dataset_size - sum(counts.values())
    order = sorted(
        counts,
        key=lambda key: (raw[key] - counts[key], ratios[key]),
        reverse=True,
    )
    for key in order[:remainder]:
        counts[key] += 1

    empty = [key for key, count in counts.items() if count < 1]
    for key in empty:
        donor = max(counts, key=lambda item: counts[item])
        if counts[donor] <= 1:
            break
        counts[donor] -= 1
        counts[key] += 1
    empty = [key for key, count in counts.items() if count < 1]
    if empty:
        raise RuntimeError(f"{label}: split vacio para {empty}; aumenta dataset_size o ajusta ratios.")
    return counts


def _recipe_list(raw: Any, method: str) -> list[dict[str, Any]]:
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise RuntimeError(f"dataset_recipes.{method} debe ser una lista.")
    recipes: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"dataset_recipes.{method}[{index}] debe ser un objeto.")
        blocks = item.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise RuntimeError(f"dataset_recipes.{method}[{index}].blocks debe ser una lista no vacia.")
        recipes.append(dict(item))
    return recipes


def _check_forbidden_md_fields(payload: dict[str, Any], *, context: str) -> None:
    for key in payload:
        normalized = str(key).strip().lower()
        if normalized in FORBIDDEN_MD_SWEEP_FIELDS:
            raise RuntimeError(
                f"{context}.{key} no puede variar en dataset_sweep v1; "
                "crea un grupo de compatibilidad separado para cambios de fisica SIESTA."
            )


def _md_recipe_metadata(
    *,
    recipe: dict[str, Any],
    block: dict[str, Any],
    size: int,
    recipe_index: int,
    block_index: int,
) -> dict[str, Any]:
    recipe_id = str(recipe.get("recipe_id") or f"md_recipe_{recipe_index + 1}")
    recipe_label = str(recipe.get("label") or recipe_id)
    block_id = str(block.get("block_id") or f"{recipe_id}_block_{block_index + 1}")
    block_label = str(block.get("label") or block_id)
    generation_parameters = {
        key: value
        for key, value in block.items()
        if key not in {"block_id", "label"}
    }
    return {
        "method": "md",
        "recipe_id": recipe_id,
        "recipe_label": recipe_label,
        "block_id": block_id,
        "block_label": block_label,
        "dataset_size": int(size),
        "recipe_index": recipe_index,
        "block_index": block_index,
        "generation_parameters": generation_parameters,
        "generation_parameters_json": json.dumps(
            json_safe(generation_parameters),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "seed": optional_seed(block.get("seed", recipe.get("seed"))),
    }


def md_dataset_recipes_to_specs(
    raw_recipes: Any,
    *,
    split_ratios: dict[str, float],
    max_datasets: int | None = None,
) -> dict[str, Any] | None:
    """Convert Experiment-style MD dataset recipes into executable specs."""

    if raw_recipes in (None, "", {}):
        return None
    if isinstance(raw_recipes, dict) and "md" in raw_recipes:
        recipes = _recipe_list(raw_recipes.get("md"), "md")
        normalized = {"md": recipes}
    elif isinstance(raw_recipes, list):
        recipes = _recipe_list(raw_recipes, "md")
        normalized = {"md": recipes}
    else:
        raise RuntimeError("dataset_recipes debe ser un objeto con clave md o una lista de recetas MD.")

    if not recipes:
        return None
    if max_datasets is not None and len(recipes) > int(max_datasets):
        raise RuntimeError(f"dataset_sweep pide {len(recipes)} datasets, max_datasets={int(max_datasets)}.")

    seen_ids: set[str] = set()
    specs: list[dict[str, Any]] = []
    for recipe_index, recipe in enumerate(recipes):
        _check_forbidden_md_fields(recipe, context=f"dataset_recipes.md[{recipe_index}]")
        recipe_id = str(recipe.get("recipe_id") or f"md_recipe_{recipe_index + 1}")
        if recipe_id in seen_ids:
            raise RuntimeError(f"recipe_id duplicado en dataset_sweep: {recipe_id}")
        seen_ids.add(recipe_id)

        block_metadata: list[dict[str, Any]] = []
        temperature_blocks: list[dict[str, Any]] = []
        for block_index, block in enumerate(recipe["blocks"]):
            if not isinstance(block, dict):
                raise RuntimeError("Cada bloque MD debe ser un objeto.")
            _check_forbidden_md_fields(
                block,
                context=f"dataset_recipes.md[{recipe_index}].blocks[{block_index}]",
            )
            size = int(block.get("n_snapshots") or block.get("n_structures") or 0)
            if size <= 0:
                raise RuntimeError("Cada bloque MD necesita n_snapshots positivo.")
            temperature = float(block.get("temperature_K", recipe.get("temperature_K", 300.0)))
            if temperature < 0:
                raise RuntimeError("temperature_K no puede ser negativa.")
            timestep = block.get("timestep_fs")
            if timestep not in (None, "") and float(timestep) <= 0:
                raise RuntimeError("timestep_fs debe ser > 0.")
            metadata = _md_recipe_metadata(
                recipe=recipe,
                block=block,
                size=size,
                recipe_index=recipe_index,
                block_index=block_index,
            )
            block_metadata.append(metadata)
            temperature_blocks.append(
                {
                    "block_id": metadata["block_id"],
                    "label": metadata["block_label"],
                    "n_snapshots": size,
                    "temperature_K": temperature,
                    "seed": metadata["seed"],
                    **({"timestep_fs": float(timestep)} if timestep not in (None, "") else {}),
                    **({"ensemble": str(block["ensemble"])} if block.get("ensemble") not in (None, "") else {}),
                    **({"thermostat": str(block["thermostat"])} if block.get("thermostat") not in (None, "") else {}),
                }
            )
        total_size = sum(int(block["n_snapshots"]) for block in temperature_blocks)
        split_counts = validate_split_sizes(total_size, split_ratios, label=f"MD recipe {recipe_id}")
        recipe_metadata = {
            "method": "md",
            "recipe_id": recipe_id,
            "recipe_label": str(recipe.get("label") or recipe_id),
            "block_id": "__".join(meta["block_id"] for meta in block_metadata),
            "block_label": "__".join(meta["block_label"] for meta in block_metadata),
            "dataset_size": total_size,
            "blocks": block_metadata,
            "md_temperature_blocks": temperature_blocks,
            "generation_parameters": {"temperature_blocks": temperature_blocks},
            "generation_parameters_json": json.dumps(
                {"temperature_blocks": temperature_blocks},
                sort_keys=True,
                ensure_ascii=False,
            ),
            "seed": optional_seed(recipe.get("seed")),
            "comparison_role": recipe.get("comparison_role"),
            "scientific_note": recipe.get("scientific_note"),
        }
        specs.append(
            {
                "recipe_id": recipe_id,
                "label": str(recipe.get("label") or recipe_id),
                "dataset_slug": slugify_label(recipe_id, f"md_dataset_{recipe_index + 1}"),
                "size": total_size,
                "split_counts": split_counts,
                "temperature_blocks": temperature_blocks,
                "recipe_metadata": recipe_metadata,
            }
        )

    return {
        "recipes": normalized,
        "recipe_set_hash": recipe_set_hash(normalized),
        "md_dataset_specs": specs,
    }


def dataset_sweep_recipes_from_payload(payload: dict[str, Any]) -> tuple[bool, list[dict[str, Any]], int]:
    sweep = payload.get("dataset_sweep") or {}
    if not isinstance(sweep, dict):
        raise RuntimeError("dataset_sweep debe ser un objeto.")
    enabled = bool(sweep.get("enabled"))
    max_datasets = int(sweep.get("max_datasets") or payload.get("max_datasets") or 20)
    if max_datasets <= 0:
        raise RuntimeError("dataset_sweep.max_datasets debe ser > 0.")
    recipes: list[dict[str, Any]] = []
    raw_sweep_recipes = sweep.get("recipes")
    if raw_sweep_recipes not in (None, "") and (enabled or raw_sweep_recipes):
        if not isinstance(raw_sweep_recipes, list):
            raise RuntimeError("dataset_sweep.recipes debe ser una lista.")
        recipes = [dict(item) for item in raw_sweep_recipes]
    elif payload.get("dataset_recipes") not in (None, "", {}):
        raw = payload.get("dataset_recipes")
        if not isinstance(raw, dict):
            raise RuntimeError("dataset_recipes debe ser un objeto.")
        raw_md = raw.get("md") or []
        if not isinstance(raw_md, list):
            raise RuntimeError("dataset_recipes.md debe ser una lista.")
        recipes = [dict(item) for item in raw_md]
        enabled = enabled or bool(recipes)
    elif str(payload.get("dataset_mode") or "").strip() == "generate_new":
        snapshot_count = int(payload.get("snapshot_count") or 0)
        if snapshot_count <= 0:
            raise RuntimeError("dataset_mode='generate_new' necesita snapshot_count positivo.")
        temperature = float(payload.get("temperature_K") or 300.0)
        recipes = [
            {
                "recipe_id": f"md_single_{snapshot_count}_300K",
                "label": f"MD single dataset: {snapshot_count} snapshots @ {temperature:g} K",
                "blocks": [
                    {
                        "block_id": "md_300K",
                        "n_snapshots": snapshot_count,
                        "temperature_K": temperature,
                    }
                ],
            }
        ]
        enabled = True
    if enabled and not recipes:
        raise RuntimeError("dataset_sweep esta activado pero no contiene recetas MD.")
    return enabled, recipes, max_datasets
