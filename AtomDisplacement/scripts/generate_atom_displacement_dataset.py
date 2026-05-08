#!/usr/bin/env python3
"""Generate one or more SIESTA FC inputs for atom-displacement datasets."""

from __future__ import annotations

import copy
import re
import shutil
from pathlib import Path
from typing import Any

from atom_displacement_utils import (
    BASE_DIR,
    DATASET_DIR,
    FC_RUNS_DIR_NAME,
    PIPELINE_CONFIG,
    PIPELINE_PATHS,
    compute_max_fc_structures,
    compute_water_geometry_metrics,
    copy_pseudopotentials,
    ensure_dir,
    fc_displaced_atom_count,
    load_reference_structure,
    run_command_in_venv,
    write_json,
)
from pipeline_config_utils import command, render_single_point_fdf


STORE_DIR_NAME = "AtDis_steps"


def _unit_from_displacement(value: str) -> str:
    match = re.fullmatch(r"\s*[-+0-9.Ee]+\s*([A-Za-z/]+)?\s*", str(value))
    return (match.group(1) if match and match.group(1) else "Ang")


def _format_displacement_value(value: Any, default_unit: str) -> str:
    if isinstance(value, (int, float)):
        return f"{value} {default_unit}"
    text = str(value).strip()
    if re.fullmatch(r"[-+0-9.Ee]+", text):
        return f"{text} {default_unit}"
    return text


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "displacement"


def setup_lua_store(run_dir: Path, lua_script_name: str) -> None:
    """Prepare Graph2Mat's SIESTA store script inside one raw FC run directory."""

    run_command_in_venv(
        [command(PIPELINE_CONFIG, "graph2mat"), "siesta", "md", "setup-store"],
        cwd=run_dir,
    )
    lua_script = run_dir / lua_script_name
    text = lua_script.read_text(encoding="utf-8")
    text = text.replace('local store_dir = "MD_steps"', f'local store_dir = "{STORE_DIR_NAME}"')
    lua_script.write_text(text, encoding="utf-8")


def configured_displacements(force_constants: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized displacement entries while preserving legacy config.

    New configs can define ``structure.force_constants.displacements`` as a list
    of ``{value, n_structures}`` objects. If that list is absent, the legacy
    single ``displacement``/``target_count`` pair is converted into one entry.
    """

    default_unit = _unit_from_displacement(force_constants.get("displacement", "0.05 Ang"))
    raw_entries = force_constants.get("displacements")
    if not raw_entries:
        raw_entries = [
            {
                "value": force_constants.get("displacement", "0.05 Ang"),
                "n_structures": force_constants.get("target_count"),
            }
        ]
    elif isinstance(raw_entries, dict):
        def max_requested_count(value: Any) -> int | None:
            if value in (None, ""):
                return None
            if isinstance(value, (int, float)):
                return int(value)
            return max(int(count) for count in value) if value else None

        raw_entries = [
            {
                "value": displacement,
                # Standalone generation prepares enough selected FC steps for
                # the largest requested option. The comparison UI expands exact
                # per-dataset aligned/cartesian combinations before this script
                # is called.
                "n_structures": max_requested_count(counts),
            }
            for displacement, counts in sorted(
                raw_entries.items(),
                key=lambda item: str(item[0]),
            )
        ]

    entries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_entries):
        if isinstance(raw_entry, dict):
            value = raw_entry.get("value", force_constants.get("displacement", "0.05 Ang"))
            n_structures = raw_entry.get("n_structures", force_constants.get("target_count"))
            label = raw_entry.get("label")
            unit = raw_entry.get("unit", default_unit)
        else:
            value = raw_entry
            n_structures = force_constants.get("target_count")
            label = None
            unit = default_unit
        displacement = _format_displacement_value(value, unit)
        entries.append(
            {
                "index": index,
                "label": str(label) if label else f"disp_{index:03d}_{_safe_slug(displacement)}",
                "value": displacement,
                "n_structures": None if n_structures is None else int(n_structures),
            }
        )
    return entries


def build_run_config(displacement_value: str, system_label: str) -> dict[str, Any]:
    config = copy.deepcopy(PIPELINE_CONFIG)
    molecule_name = config["structure"].get("molecule_name", "system")
    config["structure"]["force_constants"]["displacement"] = displacement_value
    config["structure"]["single_point"]["system_name_template"] = f"{molecule_name} {{sample_id}}"
    config["structure"]["single_point"]["title"] = (
        f"Force-constant calculation for {system_label}"
    )
    return config


def write_fc_run(
    *,
    run_dir: Path,
    reference: Any,
    system_label: str,
    displacement_value: str,
    force_constants: dict[str, Any],
) -> None:
    ensure_dir(run_dir)
    setup_lua_store(run_dir, force_constants["lua_script"])
    copy_pseudopotentials(BASE_DIR, run_dir)
    run_config = build_run_config(displacement_value, system_label)
    content = render_single_point_fdf(
        run_config,
        positions_ang=reference.positions_ang,
        atom_species=reference.atom_species,
        sample_id=system_label,
    )
    (run_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]).write_text(content, encoding="utf-8")


def main() -> int:
    ensure_dir(DATASET_DIR)
    fc_runs_dir = DATASET_DIR / FC_RUNS_DIR_NAME
    if fc_runs_dir.exists():
        shutil.rmtree(fc_runs_dir)
    ensure_dir(fc_runs_dir)

    reference, source_path = load_reference_structure()
    metrics = compute_water_geometry_metrics(reference)
    force_constants = PIPELINE_CONFIG["structure"]["force_constants"]
    include_reference = bool(force_constants.get("include_reference", True))
    first_atom = int(force_constants.get("first_atom", 1))
    last_atom = force_constants.get("last_atom")
    last_atom = len(reference.atom_species) if last_atom is None else int(last_atom)
    displaced_atoms = fc_displaced_atom_count(
        len(reference.atom_species),
        first_atom,
        last_atom,
    )
    max_structures = compute_max_fc_structures(displaced_atoms, include_reference)
    displacement_entries = configured_displacements(force_constants)

    print("=== AtomDisplacement dataset generation ===")
    print(f"[INFO] Geometria de referencia: {source_path}")
    print("[INFO] Modo de desplazamiento: SIESTA MD.TypeOfRun FC")
    print(f"[INFO] Rango de atomos FC: {first_atom}-{last_atom}")
    print(
        "[INFO] Limite FC por magnitud: "
        f"6N{' + 1 referencia' if include_reference else ''} = {max_structures} "
        f"(N desplazados={displaced_atoms})"
    )

    runs = []
    for entry in displacement_entries:
        requested = entry["n_structures"] if entry["n_structures"] is not None else max_structures
        if requested > max_structures:
            raise ValueError(
                "FC cannot generate the requested number of structures for one "
                f"displacement magnitude: requested={requested}, max={max_structures}. "
                "The FC method is limited to +/- displacements along 3 Cartesian "
                "directions for each selected atom."
            )

        system_label = f"fc_{entry['index']:03d}_{_safe_slug(entry['value'])}"
        run_dir = fc_runs_dir / entry["label"]
        print(
            f"[INFO] FC run {entry['index']}: displacement={entry['value']}, "
            f"requested={requested}, run_dir={run_dir}"
        )
        write_fc_run(
            run_dir=run_dir,
            reference=reference,
            system_label=system_label,
            displacement_value=entry["value"],
            force_constants=force_constants,
        )

        force_constants_metadata = {
            "md_type_of_run": "FC",
            "displacement": entry["value"],
            "requested_structures": requested,
            "max_structures": max_structures,
            "include_reference": include_reference,
            "first_atom": first_atom,
            "last_atom": last_atom,
            "lua_script": force_constants.get("lua_script"),
            "save_tshs": bool(force_constants.get("save_tshs", True)),
            "save_tsde": bool(force_constants.get("save_tsde", True)),
            "save_dhs": bool(force_constants.get("save_dhs", True)),
            "dHdR_tolerance": force_constants.get("dHdR_tolerance"),
            "dSdR_tolerance": force_constants.get("dSdR_tolerance"),
        }
        metadata = {
            "id": system_label,
            "generation_mode": "siesta_fc_run",
            "method": "siesta_fc_cartesian",
            "dataset_recipe": PIPELINE_CONFIG.get("dataset_recipe") or {},
            "recipe_id": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("recipe_id"),
            "recipe_label": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("recipe_label"),
            "block_id": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("block_id"),
            "block_label": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("block_label"),
            "generation_parameters_json": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("generation_parameters_json"),
            "reference_source": source_path,
            "positions_ang": reference.positions_ang,
            "geometry_metrics": metrics,
            "force_constants": force_constants_metadata,
            "expected_outputs": {
                "force_constants": "FC",
                "hamiltonian_derivatives": f"{system_label}.dHSdR.nc",
            },
        }
        write_json(run_dir / "metadata.json", metadata)
        runs.append(
            {
                "index": entry["index"],
                "id": system_label,
                "label": entry["label"],
                "run_dir": str(run_dir),
                "run_fdf": str(run_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]),
                "displacement": entry["value"],
                "requested_structures": requested,
                "max_structures": max_structures,
                "include_reference": include_reference,
                "first_atom": first_atom,
                "last_atom": last_atom,
                "recipe_id": metadata.get("recipe_id"),
                "recipe_label": metadata.get("recipe_label"),
                "block_id": metadata.get("block_id"),
                "block_label": metadata.get("block_label"),
                "generation_parameters_json": metadata.get("generation_parameters_json"),
            }
        )

    manifest = {
        "generation_mode": "siesta_fc_multi_run",
        "dataset_recipe": PIPELINE_CONFIG.get("dataset_recipe") or {},
        "reference_source": source_path,
        "fc_runs_dir": str(fc_runs_dir),
        "subsampling": force_constants.get("subsampling", {"method": "spread", "seed": 0}),
        "force_constants": {
            "first_atom": first_atom,
            "last_atom": last_atom,
            "include_reference": include_reference,
            "max_structures_per_displacement": max_structures,
        },
        "runs": runs,
    }
    write_json(PIPELINE_PATHS["samples_manifest_path"], manifest)
    print(f"[OK] Entradas FC generadas en {fc_runs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
