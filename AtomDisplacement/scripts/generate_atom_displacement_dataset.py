#!/usr/bin/env python3
"""Generate the SIESTA FC input used for the atom-displacement dataset."""

from __future__ import annotations

import shutil

from atom_displacement_utils import (
    BASE_DIR,
    DATASET_DIR,
    PIPELINE_CONFIG,
    PIPELINE_PATHS,
    compute_water_geometry_metrics,
    copy_pseudopotentials,
    ensure_dir,
    load_reference_structure,
    run_command_in_venv,
    write_json,
    write_single_point_fdf,
)
from pipeline_config_utils import command


STORE_DIR_NAME = "AtDis_steps"


def setup_lua_store() -> None:
    run_command_in_venv(
        [command(PIPELINE_CONFIG, "graph2mat"), "siesta", "md", "setup-store"],
        cwd=DATASET_DIR,
    )
    lua_script = DATASET_DIR / PIPELINE_CONFIG["structure"]["force_constants"]["lua_script"]
    text = lua_script.read_text(encoding="utf-8")
    text = text.replace('local store_dir = "MD_steps"', f'local store_dir = "{STORE_DIR_NAME}"')
    lua_script.write_text(text, encoding="utf-8")


def main() -> int:
    ensure_dir(DATASET_DIR)
    atdis_steps_dir = DATASET_DIR / STORE_DIR_NAME
    if atdis_steps_dir.exists():
        shutil.rmtree(atdis_steps_dir)

    reference, source_path = load_reference_structure()
    metrics = compute_water_geometry_metrics(reference)
    force_constants = PIPELINE_CONFIG["structure"]["force_constants"]
    last_atom = force_constants.get("last_atom") or len(reference.atom_species)

    print("=== AtomDisplacement dataset generation ===")
    print(f"[INFO] Geometria de referencia: {source_path}")
    print("[INFO] Modo de desplazamiento: SIESTA MD.TypeOfRun FC")
    print(f"[INFO] FC.Displacement: {force_constants['displacement']}")
    print(f"[INFO] Rango de atomos FC: {force_constants.get('first_atom', 1)}-{last_atom}")
    print(f"[INFO] Lua store: {force_constants['lua_script']}")

    force_constants_metadata = {
        "md_type_of_run": "FC",
        "displacement": force_constants["displacement"],
        "first_atom": int(force_constants.get("first_atom", 1)),
        "last_atom": int(last_atom),
        "lua_script": force_constants.get("lua_script"),
        "save_tshs": bool(force_constants.get("save_tshs", True)),
        "save_tsde": bool(force_constants.get("save_tsde", True)),
        "save_dhs": bool(force_constants.get("save_dhs", True)),
        "dHdR_tolerance": force_constants.get("dHdR_tolerance"),
        "dSdR_tolerance": force_constants.get("dSdR_tolerance"),
    }

    system_label = "fc_dataset"
    setup_lua_store()
    copy_pseudopotentials(BASE_DIR, DATASET_DIR)
    write_single_point_fdf(
        DATASET_DIR / PIPELINE_CONFIG["paths"]["run_fdf_name"],
        reference,
        system_label,
    )

    metadata = {
        "id": system_label,
        "generation_mode": "siesta_fc_single_run",
        "reference_source": source_path,
        "positions_ang": reference.positions_ang,
        "geometry_metrics": metrics,
        "force_constants": force_constants_metadata,
        "expected_outputs": {
            "force_constants": "FC",
            "hamiltonian_derivatives": f"{system_label}.dHSdR.nc",
        },
    }
    write_json(DATASET_DIR / "metadata.json", metadata)

    manifest = {
        "generation_mode": "siesta_fc_single_run",
        "reference_source": source_path,
        "force_constants": force_constants_metadata,
        "run": metadata,
    }
    write_json(PIPELINE_PATHS["samples_manifest_path"], manifest)
    print(f"[OK] Entrada FC generada en {DATASET_DIR / PIPELINE_CONFIG['paths']['run_fdf_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
