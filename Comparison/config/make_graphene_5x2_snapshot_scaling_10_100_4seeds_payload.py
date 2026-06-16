#!/usr/bin/env python3
"""Generate a graphene 5x2 snapshot-scaling payload with 4 training seeds.

This mirrors the recent graphene_w90 10-1000 4-seed protocol, but regenerates
datasets from materials/graphene_5x2/RUN.fdf and restricts the sweep to
20, 40, 60, 80 and 100 snapshots.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "Comparison" / "config"
BASE_PAYLOAD = CONFIG_DIR / "graphene_w90_snapshot_scaling_10_500_paper_ready_like_payload.json"
BASE_PIPELINE_CONFIG = (
    REPO_ROOT
    / "Comparison/workspaces/20260521_124937/md/"
    "md_dataset1_azrt5n_graphene_w90_default_huber_b0p01_real_d26e8feb74/pipeline_config.yaml"
)
GRAPHENE_5X2_FDF = REPO_ROOT / "materials" / "graphene_5x2" / "RUN.fdf"

RUN_TAG = "graphene_5x2_snapshot_scaling_20_100_4seeds"
OUT_PAYLOAD = CONFIG_DIR / f"{RUN_TAG}_payload.json"
OUT_PIPELINE_CONFIG = CONFIG_DIR / f"{RUN_TAG}_pipeline_config.yaml"

SIZES = [20, 40, 60, 80, 100]
TRAINING_SEEDS = (1, 2, 3, 4)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fdf_block(text: str, name: str) -> list[str]:
    lines = text.splitlines()
    start = None
    marker = f"%block {name}".lower()
    end_marker = f"%endblock {name}".lower()
    for index, line in enumerate(lines):
        if line.split("#", 1)[0].strip().lower() == marker:
            start = index + 1
            break
    if start is None:
        return []
    out: list[str] = []
    for line in lines[start:]:
        clean = line.split("#", 1)[0].strip()
        if clean.lower() == end_marker:
            return out
        if clean:
            out.append(clean)
    raise RuntimeError(f"Unclosed FDF block: {name}")


def fdf_directive(text: str, key: str) -> str | None:
    target = key.lower()
    for line in text.splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        parts = clean.split(None, 1)
        if parts and parts[0].lower() == target:
            return parts[1].strip() if len(parts) > 1 else ""
    return None


def parse_graphene_5x2_structure() -> dict[str, Any]:
    text = GRAPHENE_5X2_FDF.read_text(encoding="utf-8")
    lattice_constant = fdf_directive(text, "LatticeConstant") or "1.0 Ang"
    lc_parts = lattice_constant.split()
    lc_value = float(lc_parts[0])
    lc_unit = lc_parts[1] if len(lc_parts) > 1 else "Ang"
    if lc_unit.lower() not in {"ang", "angstrom", "angstroms"}:
        raise RuntimeError(f"Unsupported LatticeConstant unit in {GRAPHENE_5X2_FDF}: {lc_unit}")

    lattice_vectors = [
        [float(part) * lc_value for part in row.split()[:3]]
        for row in fdf_block(text, "LatticeVectors")
    ]
    if len(lattice_vectors) != 3:
        raise RuntimeError(f"Expected three lattice vectors in {GRAPHENE_5X2_FDF}")

    coord_format = (fdf_directive(text, "AtomicCoordinatesFormat") or "").strip().lower()
    if coord_format != "fractional":
        raise RuntimeError(f"Expected Fractional coordinates in {GRAPHENE_5X2_FDF}")

    atoms = []
    for row in fdf_block(text, "AtomicCoordinatesAndAtomicSpecies"):
        parts = row.split()
        frac = [float(parts[0]), float(parts[1]), float(parts[2])]
        position = [
            sum(frac[i] * lattice_vectors[i][axis] for i in range(3))
            for axis in range(3)
        ]
        atoms.append(
            {
                "label": "C",
                "species_index": int(parts[3]),
                "position": [round(value, 10) for value in position],
            }
        )
    if len(atoms) != 20:
        raise RuntimeError(f"Expected 20 atoms in {GRAPHENE_5X2_FDF}, found {len(atoms)}")

    return {
        "lattice_vectors": lattice_vectors,
        "atoms": atoms,
    }


def seed_for(temp_k: int, size: int) -> int:
    return int(f"{temp_k}{size:03d}")


def recipe_for_size(size: int) -> dict[str, Any]:
    n_150 = size // 5
    n_450 = size // 5
    n_300 = size - n_150 - n_450
    return {
        "recipe_id": f"graphene_5x2_scale_iid{size}",
        "label": f"Graphene 5x2 IID scale: {size} snapshots",
        "size": size,
        "thermal_regime": "iid_mixed",
        "split_intent": "IID mixed temperature train/validation/test with blocked temporal gap.",
        "blocks": [
            {
                "block_id": f"graphene_5x2_scale_iid{size}_T150_{n_150}_1",
                "label": f"{n_150} snapshots @ 150 K",
                "n_snapshots": n_150,
                "temperature_K": 150,
                "seed": seed_for(150, size),
            },
            {
                "block_id": f"graphene_5x2_scale_iid{size}_T300_{n_300}_2",
                "label": f"{n_300} snapshots @ 300 K",
                "n_snapshots": n_300,
                "temperature_K": 300,
                "seed": seed_for(300, size),
            },
            {
                "block_id": f"graphene_5x2_scale_iid{size}_T450_{n_450}_3",
                "label": f"{n_450} snapshots @ 450 K",
                "n_snapshots": n_450,
                "temperature_K": 450,
                "seed": seed_for(450, size),
            },
        ],
    }


def template_prefix(config_id: str) -> str:
    if "-N" not in config_id:
        raise RuntimeError(f"Unexpected config_id without -N suffix: {config_id}")
    return config_id.rsplit("-N", 1)[0]


def apply_training_seed(run: dict[str, Any], seed: int) -> None:
    overrides = run.setdefault("overrides", {})
    if run.get("model") == "graph2mat":
        overrides["seed_everything"] = seed
    elif run.get("model") == "deeph":
        overrides["seed"] = seed


def build_manual_runs(base_payload: dict[str, Any]) -> list[dict[str, Any]]:
    templates = copy.deepcopy(base_payload["training_sweep"]["manual_runs"][:4])
    if len(templates) != 4:
        raise RuntimeError("Expected four manual run templates in the base payload.")

    manual_runs: list[dict[str, Any]] = []
    for size in SIZES:
        dataset_id = f"graphene_5x2_scale_iid{size}"
        for template in templates:
            prefix = template_prefix(str(template["config_id"]))
            for seed in TRAINING_SEEDS:
                run = copy.deepcopy(template)
                run_id = f"{prefix}-G5x2-N{size}-seed{seed}"
                run["id"] = run_id
                run["config_id"] = run_id
                run["dataset_id"] = dataset_id
                apply_training_seed(run, seed)
                manual_runs.append(run)
    return manual_runs


def write_pipeline_config() -> None:
    config = yaml.safe_load(BASE_PIPELINE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError(f"Invalid YAML config: {BASE_PIPELINE_CONFIG}")
    structure = parse_graphene_5x2_structure()

    config["material"] = {"preset": "graphene_5x2"}
    config.setdefault("md", {})
    md = config["md"]
    md.update(
        {
            "system_name": "Graphene 5x2 MD dataset",
            "system_label": "graphene_5x2",
            "run_fdf_template": "${REPO_ROOT}/materials/graphene_5x2/RUN.fdf",
            "lattice_constant": {"value": 1.0, "unit": "Ang"},
            "lattice_vectors": structure["lattice_vectors"],
            "species": [{"index": 1, "atomic_number": 6, "symbol": "C"}],
            "coordinates_format": "Ang",
            "kgrid_monkhorst_pack": [
                [4, 0, 0, 0.0],
                [0, 10, 0, 0.0],
                [0, 0, 1, 0.0],
            ],
            "atoms": structure["atoms"],
        }
    )

    for section in (
        config.get("training", {}).get("benchmark_metadata"),
        config.get("training", {}).get("ui_training_settings", {}).get("benchmark_metadata"),
    ):
        if isinstance(section, dict):
            section["benchmark_method_id"] = str(section.get("benchmark_method_id", "graphene")).replace(
                "graphene_w90",
                "graphene_5x2",
            )
            section["fdf_template"] = "materials/graphene_5x2/RUN.fdf"
            section["fdf_template_note"] = "SIESTA graphene 5x2 20-carbon supercell template."
            section["kpoint_mesh"] = "4x10x1"

    OUT_PIPELINE_CONFIG.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def main() -> None:
    base_payload = load_json(BASE_PAYLOAD)
    write_pipeline_config()

    recipes = [recipe_for_size(size) for size in SIZES]
    manual_runs = build_manual_runs(base_payload)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    payload = copy.deepcopy(base_payload)
    payload["description"] = (
        "Graphene 5x2 snapshot-scaling payload derived from the recent graphene_w90 "
        "10-1000 4-seed benchmark. It regenerates datasets from "
        "materials/graphene_5x2/RUN.fdf and keeps only the original N<=100 sizes."
    )
    payload["source_reference_payloads"] = [
        "Comparison/config/graphene_w90_snapshot_scaling_reuse_10_1000_4seeds_payload.json",
        str(BASE_PAYLOAD.relative_to(REPO_ROOT)),
    ]
    payload["pipeline_config"] = str(OUT_PIPELINE_CONFIG.relative_to(REPO_ROOT))
    payload["material_preset"] = "graphene_5x2"
    payload["material"] = {"mode": "preset", "preset": "graphene_5x2"}
    payload["dataset_mode"] = "full_strict_pipeline"
    payload["run_mode"] = "full_strict_pipeline"
    payload["dataset_root"] = f"${{REPO_ROOT}}/Comparison/datasets/{RUN_TAG}"
    payload["output_root"] = f"${{REPO_ROOT}}/Comparison/results/{RUN_TAG}"
    payload["system_label"] = "graphene_5x2"
    payload["run_id"] = f"{RUN_TAG}_{stamp}"
    payload.pop("reuse_dataset_sweep_from_run_root", None)
    payload["dataset_sweep"] = {
        "enabled": True,
        "max_datasets": len(recipes),
        "recipes": recipes,
    }
    payload["training_sweep"]["manual_runs"] = manual_runs
    payload["training_sweep"]["max_runs"] = len(manual_runs)
    payload["training_sweep"]["apply_to_datasets"] = ["all"]
    payload.setdefault("notes", {})
    payload["notes"].update(
        {
            "intended_use": "Fresh graphene_5x2 dataset generation plus 4-seed Graph2Mat/DeepH snapshot scaling.",
            "source_protocol": "graphene_w90_snapshot_scaling_reuse_10_1000_4seeds, restricted to N<=100",
            "expected_dataset_sizes": SIZES,
            "expected_dataset_count": len(SIZES),
            "configs_per_dataset": 4,
            "training_seeds": list(TRAINING_SEEDS),
            "expected_training_runs": len(manual_runs),
            "supercell": "5x2 graphene, 20 carbon atoms",
            "fdf_template": "materials/graphene_5x2/RUN.fdf",
            "force_constants_used": False,
        }
    )

    if len(manual_runs) != len(SIZES) * 4 * len(TRAINING_SEEDS):
        raise RuntimeError("Unexpected manual run count.")
    ids = [str(run["config_id"]) for run in manual_runs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate manual run ids detected.")
    for recipe in recipes:
        if sum(int(block["n_snapshots"]) for block in recipe["blocks"]) != int(recipe["size"]):
            raise RuntimeError(f"Recipe block count mismatch: {recipe['recipe_id']}")

    OUT_PAYLOAD.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote payload: {OUT_PAYLOAD}")
    print(f"Wrote pipeline config: {OUT_PIPELINE_CONFIG}")
    print(f"run_id: {payload['run_id']}")
    print(f"datasets: {len(SIZES)} {SIZES}")
    print(f"training runs: {len(manual_runs)}")


if __name__ == "__main__":
    main()
