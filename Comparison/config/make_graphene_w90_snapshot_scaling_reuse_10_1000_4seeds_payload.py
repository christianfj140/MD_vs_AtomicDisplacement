#!/usr/bin/env python3
"""Generate a Graph2Mat-vs-DeepH payload that reuses existing MD datasets (no SIESTA/MD).

Reads completed runs:
  - graphene_w90_snapshot_scaling_10_500_paper_ready_like_20260608_163955
  - graphene_w90_snapshot_scaling_150_1000_dense_20260609_132720

Writes:
  - Comparison/config/graphene_w90_snapshot_scaling_reuse_10_1000_4seeds_payload.json
  - Comparison/config/graphene_w90_snapshot_scaling_reuse_10_1000_4seeds_dataset_catalog/summary/dataset_sweep_summary.json
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

BASE_PAYLOAD = CONFIG_DIR / "graphene_w90_snapshot_scaling_10_500_paper_ready_like_payload.json"
DENSE_PAYLOAD = CONFIG_DIR / "graphene_w90_snapshot_scaling_150_1000_dense_payload.json"

PAPER_RUN_SUMMARY = (
    REPO_ROOT
    / "Comparison/results/graphene_w90_snapshot_scaling_10_500_paper_ready_like"
    / "graphene_w90_snapshot_scaling_10_500_paper_ready_like_20260608_163955"
    / "summary/dataset_sweep_summary.json"
)
DENSE_RUN_SUMMARY = (
    REPO_ROOT
    / "Comparison/results/graphene_w90_snapshot_scaling_150_1000_dense"
    / "graphene_w90_snapshot_scaling_150_1000_dense_20260609_132720"
    / "summary/dataset_sweep_summary.json"
)

PAPER_DATASET_ROOT = REPO_ROOT / "Comparison/datasets/graphene_w90_snapshot_scaling_paper_ready_like"
DENSE_DATASET_ROOT = REPO_ROOT / "Comparison/datasets/graphene_w90_snapshot_scaling_150_1000_dense"

RUN_TAG = "graphene_w90_snapshot_scaling_reuse_10_1000_4seeds"
OUT_PAYLOAD = CONFIG_DIR / f"{RUN_TAG}_payload.json"
OUT_CATALOG_ROOT = CONFIG_DIR / f"{RUN_TAG}_dataset_catalog"
OUT_CATALOG_SUMMARY = OUT_CATALOG_ROOT / "summary" / "dataset_sweep_summary.json"

REQUIRED_DATASET_FILES = (
    "benchmark_dataset_manifest.json",
    "frozen_split_manifest.json",
    "artifact_validation.json",
)

TRAINING_SEEDS = (1, 2, 3, 4)
OVERLAP_SIZES = {150, 200, 250, 300, 500}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recipe_by_size(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for recipe in payload.get("dataset_sweep", {}).get("recipes") or []:
        out[int(recipe["size"])] = recipe
    return out


def rows_by_size(summary_path: Path) -> dict[int, dict[str, Any]]:
    summary = load_json(summary_path)
    out: dict[int, dict[str, Any]] = {}
    for row in summary.get("rows") or []:
        out[int(row["dataset_size"])] = dict(row)
    return out

def keep_size_for_reduced_sweep(size: int) -> bool:
    """Keep dense sampling around the N_min region and sparse sampling at high N."""
    if size < 100:
        return size in {10, 20, 25, 30, 40, 50, 55, 60, 70, 75, 80, 90, 95}

    if size <= 200:
        return size % 10 == 0 or size == 150 or size == 175 or size == 180

    if size <= 300:
        return size % 20 == 0 or size in {250, 300}

    if size <= 500:
        return size % 50 == 0

    return size % 100 == 0

    
def canonical_source_for_size(
    size: int,
    *,
    paper_rows: dict[int, dict[str, Any]],
    dense_rows: dict[int, dict[str, Any]],
) -> str:
    """Prefer paper_ready_like for N<=500 when available; otherwise dense."""
    if size in paper_rows and size in dense_rows:
        return "paper" if size <= 500 else "dense"
    if size in paper_rows:
        return "paper"
    if size in dense_rows:
        return "dense"
    raise RuntimeError(f"No dataset source found for N={size}")


def resolve_dataset_mapping(
    paper_rows: dict[int, dict[str, Any]],
    dense_rows: dict[int, dict[str, Any]],
    paper_recipes: dict[int, dict[str, Any]],
    dense_recipes: dict[int, dict[str, Any]],
) -> tuple[list[int], dict[int, dict[str, Any]], list[str]]:
    all_sizes = sorted(set(paper_rows) | set(dense_rows))
    sizes = [size for size in all_sizes if keep_size_for_reduced_sweep(size)]
    dropped_sizes = [size for size in all_sizes if size not in sizes]

    mapping: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []

    if dropped_sizes:
        warnings.append(
            "Reduced sweep density: dropped sizes "
            f"{dropped_sizes}. Policy: dense below N=300, multiples of 50 for 300<N<=500, "
            "multiples of 100 for N>500."
        )
    mapping: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []

    for size in sizes:
        source = canonical_source_for_size(size, paper_rows=paper_rows, dense_rows=dense_rows)
        row = paper_rows.get(size) if source == "paper" else dense_rows.get(size)
        recipe = paper_recipes.get(size) if source == "paper" else dense_recipes.get(size)
        if row is None or recipe is None:
            raise RuntimeError(
                f"Missing dataset row/recipe for N={size} from source={source}. "
                "Payload generation is blocked."
            )
        dataset_root = Path(str(row["dataset_root"]))
        missing = [name for name in REQUIRED_DATASET_FILES if not (dataset_root / name).exists()]
        if missing:
            raise RuntimeError(
                f"Dataset N={size} at {dataset_root} is missing required files: {', '.join(missing)}"
            )
        if size in OVERLAP_SIZES and size in paper_rows and size in dense_rows:
            warnings.append(
                f"N={size}: duplicate in both runs; using paper_ready_like source "
                f"({paper_rows[size]['dataset_root']}); dense copy ignored "
                f"({dense_rows[size]['dataset_root']})."
            )
        mapping[size] = {
            "size": size,
            "source": source,
            "dataset_id": str(recipe["recipe_id"]),
            "dataset_root": str(dataset_root),
            "recipe": recipe,
            "catalog_row": {
                **row,
                "dataset_root": str(dataset_root),
                "status": "benchmark_ready",
                "generation_seconds": 0.0,
                "elapsed_seconds": 0.0,
                "source": f"reused_from_{source}_run",
            },
        }
    return sizes, mapping, warnings


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


def build_manual_runs(base_payload: dict[str, Any], sizes: list[int]) -> list[dict[str, Any]]:
    templates = copy.deepcopy(base_payload["training_sweep"]["manual_runs"][:4])
    if len(templates) != 4:
        raise RuntimeError("Expected four manual run templates in the base payload.")

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
    sizes: list[int],
    mapping: dict[int, dict[str, Any]],
    *,
    split_ratios: dict[str, float],
) -> dict[str, Any]:
    rows = [mapping[size]["catalog_row"] for size in sizes]
    return {
        "schema": "graph2mat_deeph_dataset_sweep_summary_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_mode": "reuse_existing_datasets",
        "scientific_rule": "Train-only reuse of validated MD datasets; SIESTA/MD generation skipped.",
        "artifact_contract_version": rows[0].get("artifact_contract_version", "joint_graph2mat_deeph_artifact_contract_v1"),
        "dataset_root": f"${{REPO_ROOT}}/Comparison/datasets/{RUN_TAG}",
        "max_datasets": len(rows),
        "total_datasets": len(rows),
        "total_snapshots": sum(int(row.get("dataset_size") or 0) for row in rows),
        "split_ratios": split_ratios,
        "recipe_set_hash": f"reuse_union_{len(rows)}_sizes",
        "rows": rows,
    }


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
    dense_payload = load_json(DENSE_PAYLOAD)

    paper_rows = rows_by_size(PAPER_RUN_SUMMARY)
    dense_rows = rows_by_size(DENSE_RUN_SUMMARY)
    paper_recipes = recipe_by_size(base_payload)
    dense_recipes = recipe_by_size(dense_payload)

    sizes, mapping, warnings = resolve_dataset_mapping(
        paper_rows,
        dense_rows,
        paper_recipes,
        dense_recipes,
    )

    manual_runs = build_manual_runs(base_payload, sizes)
    recipes = [mapping[size]["recipe"] for size in sizes]
    catalog_summary = build_catalog_summary(
        sizes,
        mapping,
        split_ratios=base_payload.get("splits") or {"train": 0.8, "validation": 0.1, "test": 0.1},
    )

    payload = copy.deepcopy(base_payload)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{RUN_TAG}_{stamp}"

    payload["description"] = (
        "Graphene W90 snapshot-scaling train-only payload reusing validated MD datasets from "
        "graphene_w90_snapshot_scaling_10_500_paper_ready_like_20260608_163955 and "
        "graphene_w90_snapshot_scaling_150_1000_dense_20260609_132720. "
        "No MD/SIESTA/dataset generation is executed; only Graph2Mat/DeepH training, inference, "
        "metrics and ranking run on existing frozen-split datasets. "
        f"Each method/config/N uses {len(TRAINING_SEEDS)} independent training seeds."
    )
    payload["source_reference_payloads"] = [
        str(BASE_PAYLOAD.relative_to(REPO_ROOT)),
        str(DENSE_PAYLOAD.relative_to(REPO_ROOT)),
    ]
    payload["source_reference_runs"] = [
        "graphene_w90_snapshot_scaling_10_500_paper_ready_like_20260608_163955",
        "graphene_w90_snapshot_scaling_150_1000_dense_20260609_132720",
    ]
    payload["dataset_mode"] = "full_strict_pipeline"
    payload["run_mode"] = "full_strict_pipeline"
    payload["reuse_dataset_sweep_from_run_root"] = (
        f"${{REPO_ROOT}}/Comparison/config/{RUN_TAG}_dataset_catalog"
    )
    payload["dataset_root"] = f"${{REPO_ROOT}}/Comparison/datasets/{RUN_TAG}"
    payload["output_root"] = f"${{REPO_ROOT}}/Comparison/results/{RUN_TAG}"
    payload["run_id"] = run_id

    payload["dataset_sweep"] = {
        "enabled": True,
        "max_datasets": len(sizes),
        "recipes": recipes,
        "reuse_only": True,
        "notes": (
            "Recipes are metadata/fallback only. Actual dataset paths come from "
            f"{RUN_TAG}_dataset_catalog/summary/dataset_sweep_summary.json via "
            "reuse_dataset_sweep_from_run_root. MD/SIESTA generation is never launched."
        ),
    }

    payload["training_sweep"]["manual_runs"] = manual_runs
    payload["training_sweep"]["max_runs"] = len(manual_runs)
    payload["training_sweep"]["apply_to_datasets"] = ["all"]

    payload["notes"] = {
        **(payload.get("notes") or {}),
        "intended_use": "Train-only reuse benchmark for combined 10-1000 snapshot scaling with 4 seeds per config.",
        "dataset_generation": "disabled",
        "reuse_mode": "reuse_dataset_sweep_from_run_root + prebuilt dataset_sweep_summary.json",
        "expected_dataset_sizes": sizes,
        "expected_dataset_count": len(sizes),
        "expected_training_runs": len(manual_runs),
        "training_seeds": list(TRAINING_SEEDS),
        "configs_per_dataset": 4,
        "duplicate_resolution_policy": {
            "rule": (
                "Union of both source runs. When N exists in both, prefer paper_ready_like for N<=500; "
                "otherwise use the only available source."
            ),
            "overlapping_sizes": sorted(OVERLAP_SIZES),
            "exceptions": [],
        },
        "dataset_source_by_size": {
            str(size): {
                "source": mapping[size]["source"],
                "dataset_root": mapping[size]["dataset_root"],
                "dataset_id": mapping[size]["dataset_id"],
            }
            for size in sizes
        },
        "warnings": warnings,
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
    print(f"dataset_mode: {payload['dataset_mode']}")
    print(f"run_mode: {payload['run_mode']}")
    print(f"reuse_dataset_sweep_from_run_root: {payload['reuse_dataset_sweep_from_run_root']}")
    print(f"dataset_root (logical): {payload['dataset_root']}")
    print(f"output_root: {payload['output_root']}")
    print(f"N sizes: {len(sizes)}")
    print(f"N list: {sizes}")
    print(f"configs per N: 4")
    print(f"seeds per config: {len(TRAINING_SEEDS)} ({list(TRAINING_SEEDS)})")
    print(f"total training runs: {len(manual_runs)}")
    print(f"planned_runs (validated): {len(planned)}")
    if warnings:
        print("warnings:")
        for item in warnings:
            print(f"  - {item}")
    print("dataset_root by N:")
    for size in sizes:
        item = mapping[size]
        print(f"  N={size:4d} [{item['source']:5s}] {item['dataset_root']}")


if __name__ == "__main__":
    main()
