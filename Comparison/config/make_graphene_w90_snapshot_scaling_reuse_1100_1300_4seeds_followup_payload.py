#!/usr/bin/env python3
"""Generate a full follow-up Graph2Mat-vs-DeepH payload for N=1100..1300.

This is a separated follow-up clone of the completed
`graphene_w90_snapshot_scaling_reuse_10_1000_4seeds_*` sweep:

- same four anchor configs
- same four training seeds
- only dataset sizes 1100, 1200, 1300

The high-N recipes come from the existing 10..1500 dense payload. This follow-up
generates the required datasets and then runs the full Graph2Mat/DeepH workflow
(training, inference, metrics and ranking) on those newly materialized datasets.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "Comparison" / "config"
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"

BASE_PAYLOAD = CONFIG_DIR / "graphene_w90_snapshot_scaling_reuse_10_1000_4seeds_payload.json"
RECIPE_SOURCE_PAYLOAD = CONFIG_DIR / "graphene_w90_snapshot_scaling_10_1500_dense_10_30_unit_payload.json"

PARENT_RUN_ID = "graphene_w90_snapshot_scaling_reuse_10_1000_4seeds_20260610_122311"
SOURCE_RUN_ID = "graphene_w90_snapshot_scaling_10_1500_dense_10_30_unit_20260610_111930"
RUN_TAG = "graphene_w90_snapshot_scaling_1100_1300_4seeds_followup"

OUT_PAYLOAD = CONFIG_DIR / f"{RUN_TAG}_payload.json"
OUT_CATALOG_ROOT = CONFIG_DIR / f"{RUN_TAG}_dataset_catalog"
OUT_CATALOG_SUMMARY = OUT_CATALOG_ROOT / "summary" / "dataset_sweep_summary.json"

FOLLOWUP_SIZES = (1100, 1200, 1300)
TRAINING_SEEDS = (1, 2, 3, 4)
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recipe_by_size(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for recipe in payload.get("dataset_sweep", {}).get("recipes") or []:
        out[int(recipe["size"])] = recipe
    return out


def template_prefix(config_id: str) -> str:
    if "-N" not in config_id:
        raise RuntimeError(f"Unexpected config_id without -N suffix: {config_id}")
    return config_id.rsplit("-N", 1)[0]


def seed1_anchor_templates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    seen_prefixes: set[str] = set()
    for row in payload.get("training_sweep", {}).get("manual_runs") or []:
        config_id = str(row.get("config_id") or "")
        if not config_id.endswith("-seed1"):
            continue
        prefix = template_prefix(config_id)
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        templates.append(copy.deepcopy(row))
    if len(templates) != 4:
        raise RuntimeError(f"Expected 4 seed1 anchor templates, found {len(templates)}")
    return templates


def apply_training_seed(run: dict[str, Any], seed: int) -> None:
    overrides = run.setdefault("overrides", {})
    if run.get("model") == "graph2mat":
        overrides["seed_everything"] = seed
    elif run.get("model") == "deeph":
        overrides["seed"] = seed


def build_manual_runs(base_payload: dict[str, Any], sizes: list[int]) -> list[dict[str, Any]]:
    templates = seed1_anchor_templates(base_payload)
    manual_runs: list[dict[str, Any]] = []
    for size in sizes:
        dataset_id = f"graphene_w90_scale_iid{size}"
        for template in templates:
            prefix = template_prefix(str(template["config_id"]))
            for seed in TRAINING_SEEDS:
                run = copy.deepcopy(template)
                run_id = f"{prefix}-N{size}-seed{seed}"
                run["id"] = run_id
                run["config_id"] = run_id
                run["dataset_id"] = dataset_id
                apply_training_seed(run, seed)
                manual_runs.append(run)
    return manual_runs


def build_catalog_summary(
    recipes: list[dict[str, Any]],
    *,
    split_ratios: dict[str, float],
    payload_dataset_root: str,
) -> tuple[dict[str, Any], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    base_dataset_root = Path(payload_dataset_root.replace("${REPO_ROOT}", str(REPO_ROOT)))
    for recipe in recipes:
        dataset_slug = str(recipe["recipe_id"])
        dataset_root = base_dataset_root / dataset_slug
        rows.append(
            {
                "artifact_contract_version": "joint_graph2mat_deeph_artifact_contract_v1",
                "artifact_validation_path": str(dataset_root / "artifact_validation.json"),
                "artifact_validation_status": "pending_generation",
                "benchmark_manifest_path": str(dataset_root / "benchmark_dataset_manifest.json"),
                "blocks": json.dumps(recipe.get("blocks") or [], ensure_ascii=False),
                "compatibility_hash": "6cfe5f16cfed73ba",
                "config_path": "",
                "dataset_root": str(dataset_root),
                "dataset_size": int(recipe["size"]),
                "dataset_slug": dataset_slug,
                "elapsed_seconds": 0.0,
                "frozen_split_manifest_path": str(dataset_root / "frozen_split_manifest.json"),
                "generation_seconds": 0.0,
                "recipe_id": dataset_slug,
                "recipe_label": str(recipe.get("label") or dataset_slug),
                "reserved_gap_frames": 2,
                "split_counts": "",
                "split_mode": "blocked_with_gap",
                "status": "planned_generation",
                "temporal_gap": 1,
                "source": "generated_by_followup_full_payload",
            }
        )
    summary = {
        "schema": "graph2mat_deeph_dataset_sweep_summary_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_mode": "full_strict_pipeline_followup",
        "scientific_rule": (
            "Full follow-up generation plus benchmark workflow for higher N. "
            "This payload is a separated continuation of the completed 10..1000 reuse sweep."
        ),
        "artifact_contract_version": "joint_graph2mat_deeph_artifact_contract_v1",
        "dataset_root": f"${{REPO_ROOT}}/Comparison/datasets/{RUN_TAG}",
        "max_datasets": len(rows),
        "total_datasets": len(rows),
        "total_snapshots": sum(int(row.get("dataset_size") or 0) for row in rows),
        "split_ratios": split_ratios,
        "recipe_set_hash": f"followup_full_{'_'.join(str(int(row['dataset_size'])) for row in rows)}",
        "rows": rows,
    }
    return summary, warnings


def validate_training_plan(payload: dict[str, Any], catalog_summary: dict[str, Any]) -> dict[str, Any]:
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from g2m_deeph_training_sweep import expand_training_sweep

    datasets = [
        {
            "dataset_id": str(row.get("recipe_id") or row.get("dataset_slug") or Path(str(row.get("dataset_root"))).name),
            "dataset_root": str(row.get("dataset_root")),
        }
        for row in catalog_summary.get("rows") or []
    ]
    return expand_training_sweep(payload.get("training_sweep"), datasets=datasets)


def main() -> None:
    base_payload = load_json(BASE_PAYLOAD)
    recipe_payload = load_json(RECIPE_SOURCE_PAYLOAD)
    recipes_by_size = recipe_by_size(recipe_payload)
    recipes = [copy.deepcopy(recipes_by_size[size]) for size in FOLLOWUP_SIZES]
    missing_sizes = [size for size in FOLLOWUP_SIZES if size not in recipes_by_size]
    if missing_sizes:
        raise RuntimeError(f"Missing follow-up recipes for sizes: {missing_sizes}")

    payload = copy.deepcopy(base_payload)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{RUN_TAG}_{stamp}"
    payload_dataset_root = f"${{REPO_ROOT}}/Comparison/datasets/{RUN_TAG}"
    payload_output_root = f"${{REPO_ROOT}}/Comparison/results/{RUN_TAG}"
    manual_runs = build_manual_runs(base_payload, list(FOLLOWUP_SIZES))
    catalog_summary, dataset_warnings = build_catalog_summary(
        recipes,
        split_ratios=base_payload.get("splits") or {"train": 0.8, "validation": 0.1, "test": 0.1},
        payload_dataset_root=payload_dataset_root,
    )

    payload["description"] = (
        "Graphene W90 snapshot-scaling follow-up payload cloned from the completed "
        f"{PARENT_RUN_ID} sweep. It keeps the same four anchor configs, the same "
        f"{len(TRAINING_SEEDS)} training seeds per config, and extends the study with "
        "a separated high-N follow-up at N=1100,1200,1300. "
        "This payload generates the datasets with the original full-strict pipeline and then runs "
        "Graph2Mat/DeepH training, inference, metrics and ranking."
    )
    payload["source_reference_payloads"] = [
        str(BASE_PAYLOAD.relative_to(REPO_ROOT)),
        str(RECIPE_SOURCE_PAYLOAD.relative_to(REPO_ROOT)),
    ]
    payload["source_reference_runs"] = [
        PARENT_RUN_ID,
        SOURCE_RUN_ID,
    ]
    payload["dataset_mode"] = "full_strict_pipeline"
    payload["run_mode"] = "full_strict_pipeline"
    payload.pop("reuse_dataset_sweep_from_run_root", None)
    payload.pop("resume_training_sweep_from_run_root", None)
    payload.pop("resume_from_run_root", None)
    payload["dataset_root"] = payload_dataset_root
    payload["output_root"] = payload_output_root
    payload["run_id"] = run_id

    payload["dataset_sweep"] = {
        "enabled": True,
        "max_datasets": len(FOLLOWUP_SIZES),
        "recipes": recipes,
        "notes": (
            "Follow-up clone of the completed 10..1000 reuse sweep. "
            "Datasets for N=1100,1200,1300 are generated with the original full-strict pipeline recipe "
            "before launching the training sweep."
        ),
    }
    payload["dataset_sweep"].pop("reuse_only", None)

    payload["training_sweep"]["manual_runs"] = manual_runs
    payload["training_sweep"]["max_runs"] = len(manual_runs)
    payload["training_sweep"]["apply_to_datasets"] = ["all"]

    payload["notes"] = {
        **(payload.get("notes") or {}),
        "intended_use": "Separated follow-up clone for high-N continuation of the 10..1000 reuse sweep.",
        "follow_up_parent_run": PARENT_RUN_ID,
        "follow_up_sizes_only": list(FOLLOWUP_SIZES),
        "dataset_generation": "enabled_full_strict_pipeline",
        "reuse_mode": "disabled",
        "expected_dataset_sizes": list(FOLLOWUP_SIZES),
        "expected_dataset_count": len(FOLLOWUP_SIZES),
        "expected_training_runs": len(manual_runs),
        "training_seeds": list(TRAINING_SEEDS),
        "configs_per_dataset": 4,
        "config_clone_policy": "Same four anchor configs and same four seeds as the completed 10..1000 reuse sweep.",
        "dataset_source_by_size": {
            str(recipe["size"]): {
                "source": "generated_from_followup_full_payload_recipe",
                "dataset_root": str(Path(payload_dataset_root.replace("${REPO_ROOT}", str(REPO_ROOT))) / str(recipe["recipe_id"])),
                "dataset_id": str(recipe["recipe_id"]),
            }
            for recipe in recipes
        },
        "warnings": dataset_warnings,
    }

    plan = validate_training_plan(payload, catalog_summary)
    planned = plan.get("planned_runs") or []
    if len(planned) != len(manual_runs):
        raise RuntimeError(
            f"Training plan mismatch: manual_runs={len(manual_runs)} planned_runs={len(planned)}"
        )
    ids = [str(row.get("config_id") or "") for row in manual_runs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate manual run ids detected.")

    OUT_CATALOG_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_CATALOG_SUMMARY.write_text(
        json.dumps(catalog_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    OUT_PAYLOAD.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote payload: {OUT_PAYLOAD}")
    print(f"Wrote dataset catalog: {OUT_CATALOG_SUMMARY}")
    print(f"run_id: {run_id}")
    print(f"parent_run: {PARENT_RUN_ID}")
    print(f"followup_sizes: {list(FOLLOWUP_SIZES)}")
    print(f"total_training_runs: {len(manual_runs)}")
    print(f"planned_runs: {len(planned)}")
    if dataset_warnings:
        print("warnings:")
        for item in dataset_warnings:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
