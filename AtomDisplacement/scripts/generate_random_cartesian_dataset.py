#!/usr/bin/env python3
"""Generate random Cartesian perturbation samples for SIESTA single points."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Any

from atom_displacement_utils import (
    BASE_DIR,
    DATASET_DIR,
    PIPELINE_CONFIG,
    PIPELINE_PATHS,
    RANDOM_CARTESIAN_STEPS_DIR_NAME,
    RELAXED_DIR,
    Structure,
    angle_degrees,
    compute_water_geometry_metrics,
    distance,
    ensure_dir,
    load_reference_structure,
    structure_with_positions,
    write_json,
)
from pipeline_config_utils import render_single_point_fdf


DEFAULT_RANDOM_CARTESIAN_CONFIG: dict[str, Any] = {
    "enabled": False,
    "n_structures": 100,
    "seed": 1234,
    "components": None,
    "distribution": "gaussian",
    "sigma_ang": 0.03,
    "uniform_range_ang": 0.05,
    "move_atoms": "all",
    "species_filter": [],
    "min_distance_ang": 0.65,
    "max_rmsd_from_reference_ang": None,
    "max_attempts_per_structure": 100,
    "remove_center_of_mass_translation": True,
    "validation": {},
}

BOHR_TO_ANG = 0.529177210903
SCIENTIFIC_WARNING = "This is a constrained local non-MD perturbation method, not a thermodynamic ensemble."
COMPONENT_NAMES = ("atom_displacement", "bond_displacement", "angle_displacement")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def numeric_value_with_unit(value: Any, *, default_unit: str = "ang") -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    match = text and re.search(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", text)
    if not match:
        raise RuntimeError(f"No se pudo leer una amplitud numerica: {value!r}.")
    amount = float(match.group(0))
    unit_text = text[match.end() :].strip().lower() or default_unit
    if "bohr" in unit_text:
        return amount * BOHR_TO_ANG
    if unit_text in {"ang", "angs", "angstrom", "angstroms", "a", "å"} or "ang" in unit_text:
        return amount
    return amount


def apply_random_cartesian_amplitude(config: dict[str, Any]) -> None:
    raw = config.get("max_displacement", None)
    if raw in (None, ""):
        raw = config.get("amplitude_ang", None)
    if raw in (None, ""):
        return
    amplitude = numeric_value_with_unit(raw)
    config["amplitude_ang"] = amplitude
    if str(config.get("distribution", "gaussian")).strip().lower() == "uniform":
        config["uniform_range_ang"] = amplitude
    else:
        config["sigma_ang"] = amplitude


def parse_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    raise RuntimeError(f"No se pudo interpretar como booleano: {value!r}.")


def validate_distribution(value: Any, *, label: str) -> str:
    distribution = str(value or "gaussian").strip().lower()
    if distribution not in {"gaussian", "uniform"}:
        raise RuntimeError(f"{label}.distribution debe ser 'gaussian' o 'uniform' (recibido: {value!r}).")
    return distribution


def component_source(config: dict[str, Any], name: str) -> dict[str, Any]:
    components = config.get("components")
    if isinstance(components, dict) and isinstance(components.get(name), dict):
        return copy.deepcopy(components[name])
    if isinstance(config.get(name), dict):
        return copy.deepcopy(config[name])
    return {}


def has_explicit_component_config(config: dict[str, Any]) -> bool:
    components = config.get("components")
    if isinstance(components, dict) and any(name in components for name in COMPONENT_NAMES):
        return True
    return any(isinstance(config.get(name), dict) for name in COMPONENT_NAMES)


def normalized_delta_bounds(component: dict[str, Any], *, range_key: str, min_key: str, max_key: str) -> tuple[float, float]:
    half_width = float(component.get(range_key, 0.0))
    min_delta = component.get(min_key)
    max_delta = component.get(max_key)
    lower = -half_width if min_delta in (None, "") else float(min_delta)
    upper = half_width if max_delta in (None, "") else float(max_delta)
    if lower > upper:
        raise RuntimeError(f"{min_key} no puede ser mayor que {max_key}.")
    return lower, upper


def normalize_atom_component(config: dict[str, Any], source: dict[str, Any], *, default_enabled: bool) -> dict[str, Any]:
    atom_defaults = {
        "distribution": "gaussian",
        "sigma_ang": 0.03,
        "uniform_range_ang": 0.05,
        "move_atoms": "all",
        "species_filter": [],
        "remove_center_of_mass_translation": True,
    }

    def legacy_aware_value(key: str, default: Any) -> Any:
        if key in source and source.get(key) != atom_defaults.get(key):
            return source.get(key)
        return config.get(key, source.get(key, default))

    atom = {
        "enabled": parse_bool(source.get("enabled"), default_enabled),
        "distribution": validate_distribution(
            legacy_aware_value("distribution", "gaussian"),
            label="random_cartesian.components.atom_displacement",
        ),
        "sigma_ang": float(legacy_aware_value("sigma_ang", 0.03)),
        "uniform_range_ang": float(legacy_aware_value("uniform_range_ang", 0.05)),
        "move_atoms": copy.deepcopy(legacy_aware_value("move_atoms", "all")),
        "species_filter": copy.deepcopy(legacy_aware_value("species_filter", []) or []),
        "remove_center_of_mass_translation": parse_bool(
            legacy_aware_value("remove_center_of_mass_translation", True),
            True,
        ),
    }
    for amplitude_key in ("max_displacement", "amplitude_ang"):
        if amplitude_key in source:
            atom[amplitude_key] = source[amplitude_key]
        elif amplitude_key in config:
            atom[amplitude_key] = config[amplitude_key]
    apply_random_cartesian_amplitude(atom)
    atom["distribution"] = validate_distribution(atom["distribution"], label="random_cartesian.components.atom_displacement")
    atom["sigma_ang"] = float(atom["sigma_ang"])
    atom["uniform_range_ang"] = float(atom["uniform_range_ang"])
    if atom["sigma_ang"] < 0 or atom["uniform_range_ang"] < 0:
        raise RuntimeError("Las amplitudes de atom_displacement no pueden ser negativas.")
    return atom


def normalize_bond_component(source: dict[str, Any], *, default_enabled: bool) -> dict[str, Any]:
    bond = {
        "enabled": parse_bool(source.get("enabled"), default_enabled),
        "distribution": validate_distribution(
            source.get("distribution", "gaussian"),
            label="random_cartesian.components.bond_displacement",
        ),
        "sigma_ang": float(source.get("sigma_ang", 0.01)),
        "uniform_range_ang": float(source.get("uniform_range_ang", 0.02)),
        "min_delta_ang": source.get("min_delta_ang"),
        "max_delta_ang": source.get("max_delta_ang"),
        "min_bond_ang": float(source.get("min_bond_ang", 0.70)),
        "max_bond_ang": float(source.get("max_bond_ang", 1.30)),
        "bonds": source.get("bonds", "h2o_oh"),
    }
    lower, upper = normalized_delta_bounds(
        bond,
        range_key="uniform_range_ang",
        min_key="min_delta_ang",
        max_key="max_delta_ang",
    )
    bond["min_delta_ang"] = lower
    bond["max_delta_ang"] = upper
    if bond["sigma_ang"] < 0 or bond["uniform_range_ang"] < 0:
        raise RuntimeError("Las amplitudes de bond_displacement no pueden ser negativas.")
    if bond["min_bond_ang"] <= 0 or bond["max_bond_ang"] <= 0 or bond["min_bond_ang"] > bond["max_bond_ang"]:
        raise RuntimeError("Los limites de O-H de bond_displacement son invalidos.")
    if str(bond["bonds"]) != "h2o_oh":
        raise RuntimeError("bond_displacement solo soporta bonds='h2o_oh' por ahora.")
    return bond


def normalize_angle_component(source: dict[str, Any], *, default_enabled: bool) -> dict[str, Any]:
    angle = {
        "enabled": parse_bool(source.get("enabled"), default_enabled),
        "distribution": validate_distribution(
            source.get("distribution", "gaussian"),
            label="random_cartesian.components.angle_displacement",
        ),
        "sigma_deg": float(source.get("sigma_deg", 3.0)),
        "uniform_range_deg": float(source.get("uniform_range_deg", 5.0)),
        "min_delta_deg": source.get("min_delta_deg"),
        "max_delta_deg": source.get("max_delta_deg"),
        "min_angle_deg": float(source.get("min_angle_deg", 80.0)),
        "max_angle_deg": float(source.get("max_angle_deg", 130.0)),
        "angles": source.get("angles", "h2o_hoh"),
    }
    lower, upper = normalized_delta_bounds(
        angle,
        range_key="uniform_range_deg",
        min_key="min_delta_deg",
        max_key="max_delta_deg",
    )
    angle["min_delta_deg"] = lower
    angle["max_delta_deg"] = upper
    if angle["sigma_deg"] < 0 or angle["uniform_range_deg"] < 0:
        raise RuntimeError("Las amplitudes de angle_displacement no pueden ser negativas.")
    if angle["min_angle_deg"] <= 0 or angle["max_angle_deg"] > 180 or angle["min_angle_deg"] > angle["max_angle_deg"]:
        raise RuntimeError("Los limites H-O-H de angle_displacement son invalidos.")
    if str(angle["angles"]) != "h2o_hoh":
        raise RuntimeError("angle_displacement solo soporta angles='h2o_hoh' por ahora.")
    return angle


def normalize_validation_config(config: dict[str, Any]) -> dict[str, Any]:
    raw_validation = config.get("validation") if isinstance(config.get("validation"), dict) else {}
    validation = {
        "min_distance_ang": float(raw_validation.get("min_distance_ang", config.get("min_distance_ang", 0.65))),
        "max_rmsd_from_reference_ang": raw_validation.get(
            "max_rmsd_from_reference_ang",
            config.get("max_rmsd_from_reference_ang"),
        ),
        "max_attempts_per_structure": int(
            raw_validation.get("max_attempts_per_structure", config.get("max_attempts_per_structure", 100))
        ),
    }
    if validation["min_distance_ang"] < 0:
        raise RuntimeError("random_cartesian.validation.min_distance_ang no puede ser negativo.")
    if validation["max_attempts_per_structure"] <= 0:
        raise RuntimeError("random_cartesian.validation.max_attempts_per_structure debe ser mayor que cero.")
    if validation["max_rmsd_from_reference_ang"] not in (None, ""):
        validation["max_rmsd_from_reference_ang"] = float(validation["max_rmsd_from_reference_ang"])
        if validation["max_rmsd_from_reference_ang"] < 0:
            raise RuntimeError("random_cartesian.validation.max_rmsd_from_reference_ang no puede ser negativo.")
    else:
        validation["max_rmsd_from_reference_ang"] = None
    return validation


def normalize_component_config(config: dict[str, Any], *, explicit_components: bool | None = None) -> dict[str, dict[str, Any]]:
    explicit = has_explicit_component_config(config) if explicit_components is None else explicit_components
    atom_source = component_source(config, "atom_displacement")
    bond_source = component_source(config, "bond_displacement")
    angle_source = component_source(config, "angle_displacement")
    components = {
        "atom_displacement": normalize_atom_component(
            config,
            atom_source,
            default_enabled=(not explicit) or (bool(atom_source) and "enabled" not in atom_source),
        ),
        "bond_displacement": normalize_bond_component(
            bond_source,
            default_enabled=False,
        ),
        "angle_displacement": normalize_angle_component(
            angle_source,
            default_enabled=False,
        ),
    }
    if not enabled_component_names(components):
        raise RuntimeError("random_cartesian necesita al menos un componente habilitado.")
    return components


def enabled_component_names(components: dict[str, dict[str, Any]]) -> list[str]:
    return [name for name in COMPONENT_NAMES if parse_bool(components.get(name, {}).get("enabled"), False)]


def random_cartesian_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    root = config or PIPELINE_CONFIG
    raw = (root.get("structure", {}) or {}).get("random_cartesian", {}) or {}
    merged = copy.deepcopy(DEFAULT_RANDOM_CARTESIAN_CONFIG)
    merged.update(raw)
    merged["n_structures"] = int(merged["n_structures"])
    merged["seed"] = int(merged["seed"])
    merged["distribution"] = str(merged["distribution"]).strip().lower()
    apply_random_cartesian_amplitude(merged)
    merged["sigma_ang"] = float(merged["sigma_ang"])
    merged["uniform_range_ang"] = float(merged["uniform_range_ang"])
    merged["distribution"] = validate_distribution(merged["distribution"], label="random_cartesian")
    merged["components"] = normalize_component_config(merged)
    merged["validation"] = normalize_validation_config(merged)
    merged["min_distance_ang"] = float(merged["validation"]["min_distance_ang"])
    merged["max_rmsd_from_reference_ang"] = merged["validation"]["max_rmsd_from_reference_ang"]
    merged["max_attempts_per_structure"] = int(merged["validation"]["max_attempts_per_structure"])
    if merged["n_structures"] <= 0:
        raise RuntimeError("random_cartesian.n_structures debe ser mayor que cero.")
    if merged["sigma_ang"] < 0 or merged["uniform_range_ang"] < 0:
        raise RuntimeError("Las amplitudes random_cartesian no pueden ser negativas.")
    return merged


def public_random_cartesian_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def random_cartesian_block_configs(rc_config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_blocks = rc_config.get("blocks") or []
    if not raw_blocks:
        block = copy.deepcopy(rc_config)
        block.pop("blocks", None)
        block.setdefault("block_id", "rc_block_1")
        block.setdefault("label", f"{block['n_structures']} random structures")
        block["_seed_explicit"] = True
        return [block]
    if not isinstance(raw_blocks, list):
        raise RuntimeError("random_cartesian.blocks debe ser una lista.")
    blocks: list[dict[str, Any]] = []
    for index, raw_block in enumerate(raw_blocks):
        if not isinstance(raw_block, dict):
            raise RuntimeError("Cada bloque random_cartesian debe ser un objeto.")
        block = copy.deepcopy(rc_config)
        block.pop("blocks", None)
        block.update(raw_block)
        merged_components = copy.deepcopy(rc_config.get("components", {}))
        raw_components = raw_block.get("components") if isinstance(raw_block.get("components"), dict) else {}
        for name in COMPONENT_NAMES:
            component_override = {}
            if isinstance(raw_components, dict) and isinstance(raw_components.get(name), dict):
                component_override.update(raw_components[name])
            if isinstance(raw_block.get(name), dict):
                component_override.update(raw_block[name])
            if component_override:
                merged_components.setdefault(name, {}).update(component_override)
        legacy_atom_keys = {
            "distribution",
            "sigma_ang",
            "uniform_range_ang",
            "move_atoms",
            "species_filter",
            "remove_center_of_mass_translation",
            "max_displacement",
            "amplitude_ang",
        }
        legacy_atom_override = {
            key: raw_block[key]
            for key in legacy_atom_keys
            if key in raw_block
        }
        if legacy_atom_override:
            merged_components.setdefault("atom_displacement", {}).update(legacy_atom_override)
        block["components"] = merged_components
        block["n_structures"] = int(block.get("n_structures") or 0)
        if block["n_structures"] <= 0:
            raise RuntimeError("Cada bloque random_cartesian necesita n_structures positivo.")
        block["seed"] = int(block.get("seed", rc_config["seed"]))
        block["_seed_explicit"] = "seed" in raw_block
        block["distribution"] = str(block.get("distribution", rc_config["distribution"])).strip().lower()
        apply_random_cartesian_amplitude(block)
        block["sigma_ang"] = float(block["sigma_ang"])
        block["uniform_range_ang"] = float(block["uniform_range_ang"])
        block["distribution"] = validate_distribution(block["distribution"], label="random_cartesian")
        block["components"] = normalize_component_config(block)
        block["validation"] = normalize_validation_config(block)
        block["min_distance_ang"] = float(block["validation"]["min_distance_ang"])
        block["max_rmsd_from_reference_ang"] = block["validation"]["max_rmsd_from_reference_ang"]
        block["max_attempts_per_structure"] = int(block["validation"]["max_attempts_per_structure"])
        if block["sigma_ang"] < 0 or block["uniform_range_ang"] < 0:
            raise RuntimeError("Las amplitudes random_cartesian no pueden ser negativas.")
        block.setdefault("block_id", f"rc_block_{index + 1}")
        block.setdefault("label", f"RC block {index + 1}")
        blocks.append(block)
    total = sum(int(block["n_structures"]) for block in blocks)
    if total != int(rc_config["n_structures"]):
        raise RuntimeError(
            f"random_cartesian.blocks suman {total}, pero n_structures={rc_config['n_structures']}."
        )
    return blocks


def moving_atom_indices(structure: Structure, config: dict[str, Any]) -> list[int]:
    species_filter = [str(item) for item in (config.get("species_filter") or [])]
    allowed_species = set(species_filter)
    move_atoms = config.get("move_atoms", "all")
    if move_atoms in (None, "", "all"):
        indices = list(range(len(structure.atom_species)))
    elif isinstance(move_atoms, list):
        indices = [int(item) - 1 for item in move_atoms]
    else:
        raise RuntimeError("random_cartesian.move_atoms debe ser 'all' o una lista de indices 1-based.")
    if any(index < 0 or index >= len(structure.atom_species) for index in indices):
        raise RuntimeError("random_cartesian.move_atoms contiene indices fuera de rango.")
    if allowed_species:
        indices = [index for index in indices if structure.symbols[index] in allowed_species]
    if not indices:
        raise RuntimeError("random_cartesian no tiene atomos movibles tras aplicar species_filter.")
    return indices


def sample_displacement_vector(rng: random.Random, config: dict[str, Any]) -> list[float]:
    if config["distribution"] == "gaussian":
        sigma = float(config["sigma_ang"])
        return [rng.gauss(0.0, sigma) for _axis in range(3)]
    half_width = float(config["uniform_range_ang"])
    return [rng.uniform(-half_width, half_width) for _axis in range(3)]


def displacement_field(
    structure: Structure,
    config: dict[str, Any],
    rng: random.Random,
) -> list[list[float]]:
    moving = moving_atom_indices(structure, config)
    displacements = [[0.0, 0.0, 0.0] for _atom in structure.atom_species]
    for index in moving:
        displacements[index] = sample_displacement_vector(rng, config)
    if bool(config.get("remove_center_of_mass_translation", True)):
        mean = [
            sum(displacements[index][axis] for index in moving) / len(moving)
            for axis in range(3)
        ]
        for index in moving:
            displacements[index] = [
                displacements[index][axis] - mean[axis]
                for axis in range(3)
            ]
    return displacements


def positions_with_displacements(
    structure: Structure,
    displacements: list[list[float]],
) -> list[list[float]]:
    return [
        [position[axis] + displacements[index][axis] for axis in range(3)]
        for index, position in enumerate(structure.positions_ang)
    ]


def minimum_pair_distance(positions_ang: list[list[float]]) -> float:
    min_distance = math.inf
    for left in range(len(positions_ang)):
        for right in range(left + 1, len(positions_ang)):
            min_distance = min(min_distance, distance(positions_ang[left], positions_ang[right]))
    return min_distance


def vector_sub(left: list[float], right: list[float]) -> list[float]:
    return [left[axis] - right[axis] for axis in range(3)]


def vector_add(left: list[float], right: list[float]) -> list[float]:
    return [left[axis] + right[axis] for axis in range(3)]


def vector_scale(vector: list[float], scale: float) -> list[float]:
    return [value * scale for value in vector]


def vector_dot(left: list[float], right: list[float]) -> float:
    return sum(left[axis] * right[axis] for axis in range(3))


def vector_cross(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(vector_dot(vector, vector))


def vector_unit(vector: list[float]) -> list[float]:
    norm = vector_norm(vector)
    if norm <= 1e-14:
        raise RuntimeError("No se puede normalizar un vector de longitud cero.")
    return [value / norm for value in vector]


def rotate_vector(vector: list[float], axis: list[float], angle_rad: float) -> list[float]:
    unit_axis = vector_unit(axis)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    cross = vector_cross(unit_axis, vector)
    dot = vector_dot(unit_axis, vector)
    return [
        vector[axis_index] * cos_angle
        + cross[axis_index] * sin_angle
        + unit_axis[axis_index] * dot * (1.0 - cos_angle)
        for axis_index in range(3)
    ]


def water_atom_indices(structure: Structure) -> tuple[int, int, int]:
    oxygen_indices = [index for index, symbol in enumerate(structure.symbols) if symbol == "O"]
    hydrogen_indices = [index for index, symbol in enumerate(structure.symbols) if symbol == "H"]
    if len(oxygen_indices) != 1 or len(hydrogen_indices) != 2 or len(structure.symbols) != 3:
        raise RuntimeError(
            "random_cartesian bond_displacement/angle_displacement solo soportan H2O "
            "(exactamente 1 O y 2 H)."
        )
    return oxygen_indices[0], hydrogen_indices[0], hydrogen_indices[1]


def require_h2o_topology(structure: Structure) -> tuple[int, int, int]:
    oxygen, h1, h2 = water_atom_indices(structure)
    oh_1 = distance(structure.positions_ang[oxygen], structure.positions_ang[h1])
    oh_2 = distance(structure.positions_ang[oxygen], structure.positions_ang[h2])
    if oh_1 <= 1e-12 or oh_2 <= 1e-12:
        raise RuntimeError("random_cartesian H2O requiere dos enlaces O-H con longitud positiva.")
    angle_degrees(structure.positions_ang[h1], structure.positions_ang[oxygen], structure.positions_ang[h2])
    return oxygen, h1, h2


def is_h2o_structure(structure: Structure) -> bool:
    try:
        require_h2o_topology(structure)
    except RuntimeError:
        return False
    return True


def water_geometry_metrics_or_none(structure: Structure) -> dict[str, float]:
    if not is_h2o_structure(structure):
        return {}
    try:
        return compute_water_geometry_metrics(structure)
    except Exception:
        return {}


def sample_bounded_scalar(
    rng: random.Random,
    component: dict[str, Any],
    *,
    sigma_key: str,
    min_key: str,
    max_key: str,
) -> float:
    lower = float(component[min_key])
    upper = float(component[max_key])
    if lower > upper:
        raise RuntimeError(f"{min_key} no puede ser mayor que {max_key}.")
    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-15):
        return lower
    if component["distribution"] == "uniform":
        return rng.uniform(lower, upper)
    sigma = float(component[sigma_key])
    if sigma == 0:
        return max(lower, min(upper, 0.0))
    for _attempt in range(100):
        value = rng.gauss(0.0, sigma)
        if lower <= value <= upper:
            return value
    return max(lower, min(upper, value))


def apply_bond_displacement(
    reference: Structure,
    positions: list[list[float]],
    rng: random.Random,
    component: dict[str, Any],
) -> tuple[list[list[float]], list[dict[str, float]]]:
    oxygen, h1, h2 = water_atom_indices(reference)
    updated = copy.deepcopy(positions)
    deltas: list[dict[str, float]] = []
    for label, hydrogen in (("oh_1", h1), ("oh_2", h2)):
        ref_length = distance(reference.positions_ang[oxygen], reference.positions_ang[hydrogen])
        delta = sample_bounded_scalar(
            rng,
            component,
            sigma_key="sigma_ang",
            min_key="min_delta_ang",
            max_key="max_delta_ang",
        )
        target_length = ref_length + delta
        direction = vector_sub(updated[hydrogen], updated[oxygen])
        if vector_norm(direction) <= 1e-14:
            direction = vector_sub(reference.positions_ang[hydrogen], reference.positions_ang[oxygen])
        updated[hydrogen] = vector_add(updated[oxygen], vector_scale(vector_unit(direction), target_length))
        deltas.append(
            {
                "bond": label,
                "atom_indices_0_based": [oxygen, hydrogen],
                "reference_length_ang": ref_length,
                "delta_ang": target_length - ref_length,
                "target_length_ang": target_length,
            }
        )
    return updated, deltas


def perpendicular_axis(vector: list[float]) -> list[float]:
    unit = vector_unit(vector)
    candidates = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    base = min(candidates, key=lambda candidate: abs(vector_dot(unit, candidate)))
    return vector_unit(vector_cross(unit, base))


def apply_angle_displacement(
    reference: Structure,
    positions: list[list[float]],
    rng: random.Random,
    component: dict[str, Any],
) -> tuple[list[list[float]], dict[str, float]]:
    oxygen, h1, h2 = water_atom_indices(reference)
    updated = copy.deepcopy(positions)
    ref_angle = angle_degrees(
        reference.positions_ang[h1],
        reference.positions_ang[oxygen],
        reference.positions_ang[h2],
    )
    delta = sample_bounded_scalar(
        rng,
        component,
        sigma_key="sigma_deg",
        min_key="min_delta_deg",
        max_key="max_delta_deg",
    )
    target_angle = ref_angle + delta
    pivot = updated[oxygen]
    v1 = vector_sub(updated[h1], pivot)
    v2 = vector_sub(updated[h2], pivot)
    r1 = vector_norm(v1)
    r2 = vector_norm(v2)
    if r1 <= 1e-14 or r2 <= 1e-14:
        raise RuntimeError("No se puede aplicar angle_displacement con un enlace O-H de longitud cero.")
    current_angle = angle_degrees(updated[h1], updated[oxygen], updated[h2])
    axis = vector_cross(v1, v2)
    if vector_norm(axis) <= 1e-12:
        axis = perpendicular_axis(v1)
    delta_rad = math.radians(target_angle - current_angle)
    new_v1 = rotate_vector(vector_scale(vector_unit(v1), r1), axis, -0.5 * delta_rad)
    new_v2 = rotate_vector(vector_scale(vector_unit(v2), r2), axis, 0.5 * delta_rad)
    updated[h1] = vector_add(pivot, new_v1)
    updated[h2] = vector_add(pivot, new_v2)
    final_angle = angle_degrees(updated[h1], updated[oxygen], updated[h2])
    return updated, {
        "angle": "h2o_hoh",
        "atom_indices_0_based": [h1, oxygen, h2],
        "reference_angle_deg": ref_angle,
        "current_angle_before_deg": current_angle,
        "delta_deg": target_angle - ref_angle,
        "target_angle_deg": target_angle,
        "final_angle_deg": final_angle,
    }


def apply_atom_displacement(
    reference: Structure,
    positions: list[list[float]],
    rng: random.Random,
    component: dict[str, Any],
) -> tuple[list[list[float]], list[list[float]]]:
    component_config = {
        "distribution": component["distribution"],
        "sigma_ang": component["sigma_ang"],
        "uniform_range_ang": component["uniform_range_ang"],
        "move_atoms": component.get("move_atoms", "all"),
        "species_filter": component.get("species_filter") or [],
        "remove_center_of_mass_translation": component.get("remove_center_of_mass_translation", True),
    }
    displacements = displacement_field(reference, component_config, rng)
    return [
        [position[axis] + displacements[index][axis] for axis in range(3)]
        for index, position in enumerate(positions)
    ], displacements


def remove_mean_translation_from_reference(
    reference: Structure,
    positions: list[list[float]],
) -> tuple[list[list[float]], list[float]]:
    if len(reference.positions_ang) != len(positions):
        return positions, [0.0, 0.0, 0.0]
    translation = [
        sum(positions[index][axis] - reference.positions_ang[index][axis] for index in range(len(positions)))
        / len(positions)
        for axis in range(3)
    ]
    return [
        [position[axis] - translation[axis] for axis in range(3)]
        for position in positions
    ], translation


def rmsd_from_reference(reference: Structure, candidate: Structure) -> float:
    if len(reference.positions_ang) != len(candidate.positions_ang):
        return math.inf
    total = 0.0
    for ref_position, candidate_position in zip(reference.positions_ang, candidate.positions_ang):
        total += sum((candidate_position[axis] - ref_position[axis]) ** 2 for axis in range(3))
    return math.sqrt(total / max(1, len(reference.positions_ang)))


def build_geometry_metrics(reference: Structure, candidate: Structure) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "minimum_pair_distance_ang": minimum_pair_distance(candidate.positions_ang),
        "rmsd_from_reference_ang": rmsd_from_reference(reference, candidate),
    }
    metrics.update(water_geometry_metrics_or_none(candidate))
    return metrics


def generate_candidate(
    reference: Structure,
    block_config: dict[str, Any],
    rng: random.Random,
) -> tuple[Structure, dict[str, Any]]:
    components = block_config["components"]
    positions = copy.deepcopy(reference.positions_ang)
    deltas: dict[str, Any] = {
        "atom_displacements_ang": None,
        "bond_deltas": [],
        "angle_delta": None,
        "center_of_mass_translation_removed_ang": [0.0, 0.0, 0.0],
    }
    if components["bond_displacement"]["enabled"]:
        positions, deltas["bond_deltas"] = apply_bond_displacement(
            reference,
            positions,
            rng,
            components["bond_displacement"],
        )
    if components["angle_displacement"]["enabled"]:
        positions, deltas["angle_delta"] = apply_angle_displacement(
            reference,
            positions,
            rng,
            components["angle_displacement"],
        )
    if components["atom_displacement"]["enabled"]:
        positions, deltas["atom_displacements_ang"] = apply_atom_displacement(
            reference,
            positions,
            rng,
            components["atom_displacement"],
        )
        if (
            components["atom_displacement"].get("remove_center_of_mass_translation", True)
            and (
                components["bond_displacement"]["enabled"]
                or components["angle_displacement"]["enabled"]
            )
        ):
            positions, deltas["center_of_mass_translation_removed_ang"] = remove_mean_translation_from_reference(
                reference,
                positions,
            )
    return structure_with_positions(reference, positions), deltas


def validate_random_structure(
    reference: Structure,
    candidate: Structure,
    *,
    block_config: dict[str, Any],
    base_geometry_hash: str | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    metrics = build_geometry_metrics(reference, candidate)
    if len(candidate.atom_species) != len(reference.atom_species):
        return False, "atom_count_changed", metrics
    if candidate.atom_species != reference.atom_species or candidate.symbols != reference.symbols:
        return False, "species_changed", metrics
    if candidate.lattice_vectors_ang != reference.lattice_vectors_ang:
        return False, "cell_changed", metrics
    if any(
        not math.isfinite(value)
        for position in candidate.positions_ang
        for value in position
    ):
        return False, "non_finite_coordinate", metrics
    if base_geometry_hash and json_sha256(reference.to_json_dict()) != base_geometry_hash:
        return False, "reference_mutated", metrics
    validation = block_config["validation"]
    min_distance_ang = float(validation["min_distance_ang"])
    if metrics["minimum_pair_distance_ang"] < min_distance_ang:
        return False, "min_distance_below_threshold", metrics
    max_rmsd = validation.get("max_rmsd_from_reference_ang")
    if max_rmsd is not None and metrics["rmsd_from_reference_ang"] > float(max_rmsd):
        return False, "rmsd_above_threshold", metrics
    components = block_config["components"]
    if components["bond_displacement"]["enabled"] or components["angle_displacement"]["enabled"]:
        if not {"oh_1_ang", "oh_2_ang", "hoh_angle_deg"}.issubset(metrics):
            return False, "not_h2o_topology", metrics
        bond_component = components["bond_displacement"]
        min_bond = float(bond_component["min_bond_ang"])
        max_bond = float(bond_component["max_bond_ang"])
        if metrics["oh_1_ang"] < min_bond or metrics["oh_1_ang"] > max_bond:
            return False, "oh_bond_out_of_range", metrics
        if metrics["oh_2_ang"] < min_bond or metrics["oh_2_ang"] > max_bond:
            return False, "oh_bond_out_of_range", metrics
    if components["angle_displacement"]["enabled"]:
        angle_component = components["angle_displacement"]
        min_angle = float(angle_component["min_angle_deg"])
        max_angle = float(angle_component["max_angle_deg"])
        if metrics["hoh_angle_deg"] < min_angle or metrics["hoh_angle_deg"] > max_angle:
            return False, "hoh_angle_out_of_range", metrics
    return True, "ok", metrics


def random_cartesian_family_payload(base_geometry_hash: str, config: dict[str, Any]) -> dict[str, Any]:
    amplitude = (
        config["sigma_ang"]
        if config["distribution"] == "gaussian"
        else config["uniform_range_ang"]
    )
    seed_family = int(config["seed"])
    return {
        "generation_method": "random_cartesian",
        "base_geometry_hash": base_geometry_hash,
        "distribution": config["distribution"],
        "amplitude_ang": amplitude,
        "sigma_ang": float(config["sigma_ang"]) if config["distribution"] == "gaussian" else None,
        "uniform_range_ang": float(config["uniform_range_ang"]) if config["distribution"] == "uniform" else None,
        "seed_family": seed_family,
        "species_filter": config.get("species_filter") or [],
        "move_atoms": config.get("move_atoms", "all"),
        "enabled_components": enabled_component_names(config["components"]),
        "component_configuration": config["components"],
        "validation": config["validation"],
        "recipe_id": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("recipe_id"),
        "block_id": config.get("block_id") or (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("block_id"),
    }


def deterministic_split_group_id(base_geometry_hash: str, config: dict[str, Any]) -> str:
    return json_sha256(random_cartesian_family_payload(base_geometry_hash, config))


def copy_required_resources(dataset_root: Path) -> dict[str, list[str]]:
    basis_sources = sorted(RELAXED_DIR.glob("*.ion.xml"))
    if not basis_sources:
        raise RuntimeError(
            f"random_cartesian requires basis .ion.xml files in {RELAXED_DIR}."
        )
    pseudo_sources = sorted(BASE_DIR.glob("*.psf"))
    if not pseudo_sources:
        raise RuntimeError(
            f"random_cartesian requires pseudopotential .psf files in {BASE_DIR}."
        )
    basis_dir = dataset_root / "basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    for src in basis_sources:
        shutil.copy2(src, basis_dir / src.name)
    return {
        "basis": [str(path) for path in basis_sources],
        "pseudopotentials": [str(path) for path in pseudo_sources],
    }


def write_split_manifests(dataset_root: Path, samples: list[dict[str, Any]]) -> None:
    splits = ("train", "validation", "test")
    for split in splits:
        rows = [sample for index, sample in enumerate(samples) if splits[index % len(splits)] == split]
        write_json(dataset_root / f"split_manifest_{split}.json", {"split": split, "samples": rows})


def artifact_hashes(dataset_root: Path, pseudo_sources: list[str]) -> dict[str, Any]:
    matrix_files = sorted(list(dataset_root.glob("sample_*/*.TSHS")) + list(dataset_root.glob("sample_*/*.HSX")))
    return {
        "run_fdf": {
            path.relative_to(dataset_root).as_posix(): file_sha256(path)
            for path in sorted(dataset_root.glob("sample_*/RUN.fdf"))
        },
        "metadata": {
            path.relative_to(dataset_root).as_posix(): file_sha256(path)
            for path in sorted(dataset_root.glob("sample_*/metadata.json"))
        },
        "basis": {
            path.name: file_sha256(path)
            for path in sorted((dataset_root / "basis").glob("*.ion.xml"))
        },
        "pseudopotentials": {
            Path(path).name: file_sha256(Path(path))
            for path in pseudo_sources
            if Path(path).exists()
        },
        "matrices": {
            path.relative_to(dataset_root).as_posix(): file_sha256(path)
            for path in matrix_files
        },
    }


def generate_dataset(config: dict[str, Any] | None = None) -> dict[str, Any]:
    rc_config = random_cartesian_config(config)
    dataset_root = DATASET_DIR / RANDOM_CARTESIAN_STEPS_DIR_NAME
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    ensure_dir(dataset_root)
    resources = copy_required_resources(dataset_root)
    reference, source_path = load_reference_structure()
    base_geometry_hash = json_sha256(reference.to_json_dict())
    block_configs = random_cartesian_block_configs(rc_config)
    dataset_rng = random.Random(int(rc_config["seed"]))
    samples: list[dict[str, Any]] = []
    total_attempts = 0
    rejection_counts: dict[str, int] = {}

    print("=== Random Cartesian dataset generation ===")
    print(f"[INFO] Geometria base: {source_path}")
    print(f"[INFO] Output root: {dataset_root}")
    print(f"[INFO] n_structures: {rc_config['n_structures']}")
    print(f"[INFO] blocks: {len(block_configs)}")

    sample_index = 0
    for block_index, block_config in enumerate(block_configs):
        block_public_config = public_random_cartesian_config(block_config)
        block_enabled_components = enabled_component_names(block_config["components"])
        if (
            "bond_displacement" in block_enabled_components
            or "angle_displacement" in block_enabled_components
        ) and not is_h2o_structure(reference):
            raise RuntimeError(
                "random_cartesian bond_displacement/angle_displacement requieren H2O "
                "(exactamente 1 O y 2 H). Usa solo atom_displacement para otras moleculas."
            )
        family_payload = random_cartesian_family_payload(base_geometry_hash, block_config)
        split_group_id = json_sha256(family_payload)
        rng = random.Random(int(block_config["seed"])) if block_config.get("_seed_explicit") else dataset_rng
        print(
            "[INFO] block "
            f"{block_index + 1}/{len(block_configs)}: "
            f"{block_config.get('label')} · {block_config['n_structures']} structures"
        )
        for sample_index_within_block in range(int(block_config["n_structures"])):
            accepted: tuple[int, dict[str, Any], Structure, dict[str, Any], str | None] | None = None
            last_reason = "not_attempted"
            last_rejected_reason: str | None = None
            for attempt in range(1, int(block_config["max_attempts_per_structure"]) + 1):
                total_attempts += 1
                candidate, sampled_deltas = generate_candidate(reference, block_config, rng)
                ok, reason, geometry_metrics = validate_random_structure(
                    reference,
                    candidate,
                    block_config=block_config,
                    base_geometry_hash=base_geometry_hash,
                )
                last_reason = reason
                if ok:
                    accepted = (attempt, sampled_deltas, candidate, geometry_metrics, last_rejected_reason)
                    break
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                last_rejected_reason = reason
            if accepted is None:
                raise RuntimeError(
                    "random_cartesian could not generate a valid structure "
                    f"for block={block_config.get('block_id')} sample_index={sample_index_within_block} after "
                    f"{block_config['max_attempts_per_structure']} attempts: {last_reason}."
                )
            accepted_attempt, sampled_deltas, candidate, geometry_metrics, last_rejected_reason = accepted
            sample_id = f"sample_{sample_index + 1:06d}"
            sample_dir = dataset_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            for pseudo in resources["pseudopotentials"]:
                shutil.copy2(pseudo, sample_dir / Path(pseudo).name)
            single_point_config = copy.deepcopy(PIPELINE_CONFIG)
            single_point_config.setdefault("structure", {}).setdefault("force_constants", {})["enabled"] = False
            content = render_single_point_fdf(
                single_point_config,
                positions_ang=candidate.positions_ang,
                atom_species=candidate.atom_species,
                sample_id=sample_id,
            )
            (sample_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]).write_text(content, encoding="utf-8")
            metadata = {
                "id": sample_id,
                "generation_method": "random_cartesian",
                "method": "random_cartesian",
                "enabled_components": block_enabled_components,
                "component_config": block_config["components"],
                "recipe_id": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("recipe_id"),
                "recipe_label": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("recipe_label"),
                "block_id": block_config.get("block_id") or (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("block_id"),
                "block_label": block_config.get("label") or (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("block_label"),
                "generation_parameters_json": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("generation_parameters_json"),
                "base_geometry_hash": base_geometry_hash,
                "base_geometry_source": source_path,
                "seed": int(block_config["seed"]),
                "seed_family": int(block_config["seed"]),
                "sample_index": sample_index,
                "sample_index_within_block": sample_index_within_block,
                "global_sample_id": sample_id,
                "distribution": block_config["distribution"],
                "sigma_ang": float(block_config["sigma_ang"]) if block_config["distribution"] == "gaussian" else None,
                "uniform_range_ang": float(block_config["uniform_range_ang"]) if block_config["distribution"] == "uniform" else None,
                "amplitude_ang": block_public_config.get("amplitude_ang"),
                "move_atoms": block_config.get("move_atoms", "all"),
                "species_filter": block_config.get("species_filter") or [],
                "block_n_structures": int(block_config["n_structures"]),
                "block_config": block_public_config,
                "displacements_ang": sampled_deltas.get("atom_displacements_ang"),
                "atom_displacements_ang": sampled_deltas.get("atom_displacements_ang"),
                "bond_displacements_ang": sampled_deltas.get("bond_deltas") or [],
                "angle_displacement_deg": sampled_deltas.get("angle_delta"),
                "sampled_deltas": sampled_deltas,
                "final_geometry_metrics": geometry_metrics,
                "minimum_pair_distance_ang": geometry_metrics.get("minimum_pair_distance_ang"),
                "rmsd_from_reference_ang": geometry_metrics.get("rmsd_from_reference_ang"),
                "validation_thresholds": block_config["validation"],
                "min_distance_ang": float(block_config["min_distance_ang"]),
                "accepted_attempt": accepted_attempt,
                "acceptance_status": "accepted",
                "rejection_reason": None,
                "last_rejection_reason": last_rejected_reason,
                "split_group_id": split_group_id,
                "random_cartesian_family": family_payload,
                "random_cartesian_family_id": split_group_id,
            }
            write_json(sample_dir / "metadata.json", metadata)
            samples.append(
                {
                    "sample_id": sample_id,
                    "sample_dir": str(sample_dir),
                    "run_fdf": str(sample_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]),
                    "metadata": str(sample_dir / "metadata.json"),
                    "split_group_id": split_group_id,
                    "random_cartesian_family_id": split_group_id,
                    "base_geometry_hash": base_geometry_hash,
                    "distribution": metadata.get("distribution"),
                    "sigma_ang": metadata.get("sigma_ang"),
                    "uniform_range_ang": metadata.get("uniform_range_ang"),
                    "seed_family": metadata.get("seed_family"),
                    "move_atoms": json.dumps(metadata.get("move_atoms", "all"), sort_keys=True),
                    "species_filter": json.dumps(metadata.get("species_filter", []), sort_keys=True),
                    "accepted_attempt": accepted_attempt,
                    "enabled_components": json.dumps(block_enabled_components, sort_keys=True),
                    "minimum_pair_distance_ang": metadata.get("minimum_pair_distance_ang"),
                    "rmsd_from_reference_ang": metadata.get("rmsd_from_reference_ang"),
                    "method": "random_cartesian",
                    "recipe_id": metadata.get("recipe_id"),
                    "recipe_label": metadata.get("recipe_label"),
                    "block_id": metadata.get("block_id"),
                    "block_label": metadata.get("block_label"),
                    "generation_parameters_json": metadata.get("generation_parameters_json"),
                    "sample_index_within_block": sample_index_within_block,
                    "global_sample_id": sample_id,
                }
            )
            sample_index += 1

    manifest = {
        "method_id": "random_cartesian",
        "generation_method": "random_cartesian",
        "dataset_root": str(dataset_root),
        "requested_structures": int(rc_config["n_structures"]),
        "generated_structures": len(samples),
        "total_attempts": total_attempts,
        "acceptance_ratio": (len(samples) / total_attempts) if total_attempts else 0.0,
        "rejection_counts_by_reason": dict(sorted(rejection_counts.items())),
        "seed": int(rc_config["seed"]),
        "base_geometry_hash": base_geometry_hash,
        "base_geometry_source": source_path,
        "component_config": rc_config["components"],
        "validation": rc_config["validation"],
        "config_snapshot": public_random_cartesian_config(rc_config),
        "blocks": [public_random_cartesian_config(block) for block in block_configs],
        "dataset_recipe": PIPELINE_CONFIG.get("dataset_recipe") or {},
        "samples": samples,
        "siesta_input_hashes": {
            sample["sample_id"]: file_sha256(Path(sample["run_fdf"]))
            for sample in samples
        },
        "basis_hashes": {
            path.name: file_sha256(path)
            for path in sorted((dataset_root / "basis").glob("*.ion.xml"))
        },
        "pseudo_hashes": {
            Path(path).name: file_sha256(Path(path))
            for path in resources["pseudopotentials"]
        },
        "matrix_file_hashes": {},
        "deterministic_hashes": {
            "base_geometry_hash": base_geometry_hash,
            "config_hash": json_sha256(public_random_cartesian_config(rc_config)),
            "block_config_hashes": {
                str(block.get("block_id") or index): json_sha256(public_random_cartesian_config(block))
                for index, block in enumerate(block_configs, start=1)
            },
            "sample_family_hashes": {
                sample["sample_id"]: sample["random_cartesian_family_id"]
                for sample in samples
            },
        },
        "scientific_warning": SCIENTIFIC_WARNING,
        "warnings": [SCIENTIFIC_WARNING],
        "severe_warnings": [],
    }
    write_json(dataset_root / "dataset_manifest.json", manifest)
    write_json(PIPELINE_PATHS["samples_manifest_path"], manifest)
    write_json(dataset_root / "artifact_hashes.json", artifact_hashes(dataset_root, resources["pseudopotentials"]))
    write_split_manifests(dataset_root, samples)
    print(f"[OK] Random Cartesian dataset generado en {dataset_root}")
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-structures", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--distribution", choices=["gaussian", "uniform"], default=None)
    parser.add_argument("--sigma-ang", type=float, default=None)
    parser.add_argument("--uniform-range-ang", type=float, default=None)
    parser.add_argument("--random-cartesian-config-json", default=None)
    parser.add_argument("--enable-atom-displacement", dest="atom_displacement_enabled", action="store_true", default=None)
    parser.add_argument("--disable-atom-displacement", dest="atom_displacement_enabled", action="store_false")
    parser.add_argument("--enable-bond-displacement", dest="bond_displacement_enabled", action="store_true", default=None)
    parser.add_argument("--disable-bond-displacement", dest="bond_displacement_enabled", action="store_false")
    parser.add_argument("--enable-angle-displacement", dest="angle_displacement_enabled", action="store_true", default=None)
    parser.add_argument("--disable-angle-displacement", dest="angle_displacement_enabled", action="store_false")
    parser.add_argument("--atom-distribution", choices=["gaussian", "uniform"], default=None)
    parser.add_argument("--atom-sigma-ang", type=float, default=None)
    parser.add_argument("--atom-uniform-range-ang", type=float, default=None)
    parser.add_argument("--bond-distribution", choices=["gaussian", "uniform"], default=None)
    parser.add_argument("--bond-sigma-ang", type=float, default=None)
    parser.add_argument("--bond-uniform-range-ang", type=float, default=None)
    parser.add_argument("--min-bond-ang", type=float, default=None)
    parser.add_argument("--max-bond-ang", type=float, default=None)
    parser.add_argument("--bond-min-delta-ang", type=float, default=None)
    parser.add_argument("--bond-max-delta-ang", type=float, default=None)
    parser.add_argument("--angle-distribution", choices=["gaussian", "uniform"], default=None)
    parser.add_argument("--angle-sigma-deg", type=float, default=None)
    parser.add_argument("--angle-uniform-range-deg", type=float, default=None)
    parser.add_argument("--min-angle-deg", type=float, default=None)
    parser.add_argument("--max-angle-deg", type=float, default=None)
    parser.add_argument("--angle-min-delta-deg", type=float, default=None)
    parser.add_argument("--angle-max-delta-deg", type=float, default=None)
    parser.add_argument("--min-distance-ang", type=float, default=None)
    parser.add_argument("--max-rmsd-from-reference-ang", type=float, default=None)
    parser.add_argument("--max-attempts-per-structure", type=int, default=None)
    return parser


def deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def cli_component(random_config: dict[str, Any], name: str) -> dict[str, Any]:
    return random_config.setdefault("components", {}).setdefault(name, {})


def main() -> int:
    args = build_argument_parser().parse_args()
    config = copy.deepcopy(PIPELINE_CONFIG)
    random_config = config.setdefault("structure", {}).setdefault("random_cartesian", {})
    if args.random_cartesian_config_json:
        try:
            override = json.loads(args.random_cartesian_config_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"--random-cartesian-config-json no es JSON valido: {exc}") from exc
        if not isinstance(override, dict):
            raise RuntimeError("--random-cartesian-config-json debe contener un objeto JSON.")
        deep_update(random_config, override)
    if args.n_structures is not None:
        random_config["n_structures"] = args.n_structures
    if args.seed is not None:
        random_config["seed"] = args.seed
    if args.distribution is not None:
        random_config["distribution"] = args.distribution
    if args.sigma_ang is not None:
        random_config["sigma_ang"] = args.sigma_ang
    if args.uniform_range_ang is not None:
        random_config["uniform_range_ang"] = args.uniform_range_ang
    component_flag_values = {
        "atom_displacement": args.atom_displacement_enabled,
        "bond_displacement": args.bond_displacement_enabled,
        "angle_displacement": args.angle_displacement_enabled,
    }
    for name, enabled in component_flag_values.items():
        if enabled is not None:
            cli_component(random_config, name)["enabled"] = enabled
    if args.atom_distribution is not None:
        cli_component(random_config, "atom_displacement")["distribution"] = args.atom_distribution
    if args.atom_sigma_ang is not None:
        cli_component(random_config, "atom_displacement")["sigma_ang"] = args.atom_sigma_ang
    if args.atom_uniform_range_ang is not None:
        cli_component(random_config, "atom_displacement")["uniform_range_ang"] = args.atom_uniform_range_ang
    if args.bond_distribution is not None:
        cli_component(random_config, "bond_displacement")["distribution"] = args.bond_distribution
    if args.bond_sigma_ang is not None:
        cli_component(random_config, "bond_displacement")["sigma_ang"] = args.bond_sigma_ang
    if args.bond_uniform_range_ang is not None:
        cli_component(random_config, "bond_displacement")["uniform_range_ang"] = args.bond_uniform_range_ang
    if args.min_bond_ang is not None:
        cli_component(random_config, "bond_displacement")["min_bond_ang"] = args.min_bond_ang
    if args.max_bond_ang is not None:
        cli_component(random_config, "bond_displacement")["max_bond_ang"] = args.max_bond_ang
    if args.bond_min_delta_ang is not None:
        cli_component(random_config, "bond_displacement")["min_delta_ang"] = args.bond_min_delta_ang
    if args.bond_max_delta_ang is not None:
        cli_component(random_config, "bond_displacement")["max_delta_ang"] = args.bond_max_delta_ang
    if args.angle_distribution is not None:
        cli_component(random_config, "angle_displacement")["distribution"] = args.angle_distribution
    if args.angle_sigma_deg is not None:
        cli_component(random_config, "angle_displacement")["sigma_deg"] = args.angle_sigma_deg
    if args.angle_uniform_range_deg is not None:
        cli_component(random_config, "angle_displacement")["uniform_range_deg"] = args.angle_uniform_range_deg
    if args.min_angle_deg is not None:
        cli_component(random_config, "angle_displacement")["min_angle_deg"] = args.min_angle_deg
    if args.max_angle_deg is not None:
        cli_component(random_config, "angle_displacement")["max_angle_deg"] = args.max_angle_deg
    if args.angle_min_delta_deg is not None:
        cli_component(random_config, "angle_displacement")["min_delta_deg"] = args.angle_min_delta_deg
    if args.angle_max_delta_deg is not None:
        cli_component(random_config, "angle_displacement")["max_delta_deg"] = args.angle_max_delta_deg
    validation = random_config.setdefault("validation", {})
    if args.min_distance_ang is not None:
        validation["min_distance_ang"] = args.min_distance_ang
    if args.max_rmsd_from_reference_ang is not None:
        validation["max_rmsd_from_reference_ang"] = args.max_rmsd_from_reference_ang
    if args.max_attempts_per_structure is not None:
        validation["max_attempts_per_structure"] = args.max_attempts_per_structure
    generate_dataset(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
