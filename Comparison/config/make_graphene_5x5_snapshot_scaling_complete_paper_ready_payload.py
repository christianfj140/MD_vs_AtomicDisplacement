#!/usr/bin/env python3
"""Build graphene 5x5 complete paper-ready payloads without launching runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "Comparison" / "config"

ALL_SIZES = [20, 30, 50, 60, 80, 90, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
MISSING_SIZES = [30, 60, 90, 600, 800, 1000]
EXISTING_DATASET_ROOTS = {
    20: REPO_ROOT / "Comparison/datasets/graphene_5x5_snapshot_scaling_20_50_80/graphene_5x5_scale_iid20",
    50: REPO_ROOT / "Comparison/datasets/graphene_5x5_snapshot_scaling_20_50_80/graphene_5x5_scale_iid50",
    80: REPO_ROOT / "Comparison/datasets/graphene_5x5_snapshot_scaling_20_50_80/graphene_5x5_scale_iid80",
    100: REPO_ROOT / "Comparison/datasets/graphene_5x5_snapshot_scaling_100_500_mixing/graphene_5x5_scale_iid100",
    150: REPO_ROOT / "Comparison/datasets/graphene_5x5_snapshot_scaling_100_500_mixing/graphene_5x5_scale_iid150",
    200: REPO_ROOT / "Comparison/datasets/graphene_5x5_snapshot_scaling_100_500_mixing/graphene_5x5_scale_iid200",
    300: REPO_ROOT / "Comparison/datasets/graphene_5x5_snapshot_scaling_100_500_mixing/graphene_5x5_scale_iid300",
    400: REPO_ROOT / "Comparison/datasets/graphene_5x5_snapshot_scaling_100_500_mixing/graphene_5x5_scale_iid400",
    500: REPO_ROOT / "Comparison/datasets/graphene_5x5_snapshot_scaling_100_500_mixing/graphene_5x5_scale_iid500",
}
MISSING_DATASET_ROOT = REPO_ROOT / "Comparison/datasets/graphene_5x5_snapshot_scaling_complete_paper_ready_missing"
CATALOG_ROOT = CONFIG_ROOT / "graphene_5x5_snapshot_scaling_complete_paper_ready_dataset_catalog"


def repo_token(path: str) -> str:
    return path.replace(str(REPO_ROOT), "${REPO_ROOT}")


def blocks(size: int) -> list[dict[str, object]]:
    low = size // 5
    mid = size - 2 * low
    return [
        {
            "block_id": f"graphene_5x5_scale_iid{size}_T150_{low}_1",
            "label": f"{low} snapshots @ 150 K",
            "n_snapshots": low,
            "temperature_K": 150,
            "seed": 150000 + size,
        },
        {
            "block_id": f"graphene_5x5_scale_iid{size}_T300_{mid}_2",
            "label": f"{mid} snapshots @ 300 K",
            "n_snapshots": mid,
            "temperature_K": 300,
            "seed": 300000 + size,
        },
        {
            "block_id": f"graphene_5x5_scale_iid{size}_T450_{low}_3",
            "label": f"{low} snapshots @ 450 K",
            "n_snapshots": low,
            "temperature_K": 450,
            "seed": 450000 + size,
        },
    ]


def recipe(size: int) -> dict[str, object]:
    return {
        "recipe_id": f"graphene_5x5_scale_iid{size}",
        "label": f"Graphene 5x5 IID scale: {size} snapshots",
        "size": size,
        "thermal_regime": "iid_mixed",
        "split_intent": "IID mixed temperature train/validation/test with blocked temporal gap.",
        "blocks": blocks(size),
    }


def split_counts(size: int) -> dict[str, int]:
    usable = size - 2
    train = round(usable * 0.8)
    validation = max(1, round(usable * 0.1))
    test = usable - train - validation
    if test < 1:
        train -= 1
        test = 1
    return {"train": train, "validation": validation, "test": test}


G2M_ANCHOR = {
    "max_epochs": 600,
    "optim_lr": 0.0018,
    "batch_size": 32,
    "loader_threads": 4,
    "num_interactions": 3,
    "correlation": 3,
    "max_ell": 3,
    "hidden_irreps": "48x0e + 48x1o + 48x2e + 48x3o",
    "loss": "graph2mat.metrics.block_type_huber",
    "loss_kwargs": {"beta": 0.01},
    "readout": "default",
}

DH_ANCHOR = {
    "epochs": 600,
    "batch_size": 8,
    "learning_rate": 0.0003,
    "retain_edge_fea": True,
    "criterion": "MaskMSELoss",
    "optimizer": "adamW",
    "weight_decay": 0.0001,
    "normalization": "LayerNorm",
    "gauss_stop": 6,
    "num_l": 4,
    "edge_fea_len": 128,
    "atom_update_net": "CGConv",
    "if_lcmp": True,
    "atom_fea_len": 128,
    "if_edge_update": True,
}

PERFORMANCE = {
    "preset": "paper_ready_parallel_trains_snapshot_scaling_no_global_batch",
    "compute_accelerator": "gpu",
    "store_in_memory": True,
    "reuse_validated_siesta_outputs": True,
    "enable_experiment_cache": False,
    "max_parallel_dataset_jobs": 1,
    "max_parallel_siesta_jobs": 1,
    "max_parallel_prediction_jobs": 1,
    "max_parallel_evaluation_jobs": 3,
    "max_parallel_metric_jobs": 8,
    "max_parallel_graph2mat_training_jobs": 7,
    "max_parallel_deeph_training_jobs": 5,
    "model_batch_schedule": "alternating",
    "model_batch_start": "deeph",
    "omp_num_threads": 2,
    "mkl_num_threads": 2,
    "openblas_num_threads": 2,
    "numexpr_num_threads": 2,
    "torch_num_threads": 2,
    "torch_float32_matmul_precision": "high",
    "torch_mixed_precision": "bf16-mixed",
    "graph2mat_log_every_n_steps": 1,
    "graph2mat_check_val_every_n_epoch": 1,
    "graph2mat_checkpoint_every_n_epochs": 1,
    "graph2mat_require_cuequivariance": True,
    "error_policy": "continue_on_error",
}

EARLY_STOPPING = {
    "metric": "val_loss",
    "mode": "min",
    "patience": 80,
    "min_delta": 0.0,
    "max_epochs": 600,
}


def common_payload() -> dict[str, object]:
    return {
        "pipeline_config": "Comparison/config/graphene_5x5_snapshot_scaling_20_50_pipeline_config.yaml",
        "material_preset": "graphene_5x5",
        "material": {"mode": "preset", "preset": "graphene_5x5"},
        "system_label": "graphene_5x5",
        "require_tshs": True,
        "require_tsde": True,
        "require_run_output": True,
        "allow_repair": False,
        "repair_mode": "disabled",
        "allow_diagnostic_metrics": False,
        "metric_fail_policy": "fail_closed",
        "split_mode": "blocked_with_gap",
        "temporal_gap": 1,
        "block_order": "train,validation,test",
        "splits": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "performance": PERFORMANCE,
        "early_stopping": EARLY_STOPPING,
        "graph2mat_overrides": {
            "max_epochs": 750,
            "optim_lr": 0.0018,
            "batch_size": 40,
            "loader_threads": 4,
            "num_interactions": 3,
            "correlation": 3,
            "max_ell": 3,
            "hidden_irreps": "64x0e + 64x1o + 64x2e + 64x3o",
            "loss": "graph2mat.metrics.block_type_huber",
            "loss_kwargs": {"beta": 0.006},
            "readout": "edge_node_mix",
            "seed_everything": 0,
        },
        "deeph": {
            "device": "cuda:0",
            "disable_cuda": False,
            "num_threads": 2,
            "multiprocessing": 0,
            "epochs": 750,
            "batch_size": 4,
            "learning_rate": 0.0003,
            "criterion": "MaskMSELoss",
            "retain_edge_fea": True,
            "weight_decay": 0.0,
            "optimizer": "adamW",
            "edge_fea_len": 192,
            "normalization": "LayerNorm",
            "atom_update_net": "CGConv",
            "if_lcmp": True,
            "gauss_stop": 6,
            "atom_fea_len": 192,
            "num_l": 4,
            "if_edge_update": True,
        },
    }


def manual_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for size in ALL_SIZES:
        dataset_id = f"graphene_5x5_scale_iid{size}"
        runs.append(
            {
                "id": f"G2M-T1000-03-anchor-G5x5-N{size}",
                "config_id": f"G2M-T1000-03-anchor-G5x5-N{size}",
                "model": "graph2mat",
                "dataset_id": dataset_id,
                "overrides": G2M_ANCHOR,
            }
        )
        runs.append(
            {
                "id": f"DH-T1000-04-anchor-G5x5-N{size}",
                "config_id": f"DH-T1000-04-anchor-G5x5-N{size}",
                "model": "deeph",
                "dataset_id": dataset_id,
                "overrides": DH_ANCHOR,
            }
        )
    return runs


def generation_payload() -> dict[str, object]:
    payload = {
        **common_payload(),
        "description": (
            "Graphene 5x5 missing dataset-generation payload for iid30/iid60/iid90/"
            "iid600/iid800/iid1000. Generates SIESTA MD datasets only; no model training."
        ),
        "source_reference_payloads": [
            "Comparison/config/graphene_5x5_snapshot_scaling_20_50_80_payload.json",
            "Comparison/config/graphene_5x5_snapshot_scaling_100_500_mixing_payload.json",
            "Comparison/config/graphene_5x2_snapshot_scaling_12_1300_600epochs_1train_payload.json",
        ],
        "dataset_mode": "generate_new",
        "run_mode": "generate_datasets_only",
        "dataset_root": repo_token(str(MISSING_DATASET_ROOT)),
        "output_root": "${REPO_ROOT}/Comparison/results/graphene_5x5_snapshot_scaling_complete_paper_ready_missing",
        "dataset_sweep": {
            "enabled": True,
            "max_datasets": len(MISSING_SIZES),
            "recipes": [recipe(size) for size in MISSING_SIZES],
        },
        "training_sweep": {"enabled": False, "manual_runs": [], "max_runs": 0, "apply_to_datasets": []},
        "overwrite_datasets": False,
        "notes": {
            "intended_use": "Create only the missing graphene_5x5 datasets needed by the complete paper-ready training payload.",
            "missing_dataset_sizes": MISSING_SIZES,
            "temperature_distribution": "20% at 150 K, 60% at 300 K, 20% at 450 K.",
            "training": "disabled",
        },
        "run_id": "graphene_5x5_snapshot_scaling_complete_paper_ready_missing",
    }
    return payload


def training_payload() -> dict[str, object]:
    payload = {
        **common_payload(),
        "description": (
            "Graphene 5x5 complete paper-ready train payload. Reuses existing validated "
            "iid20/iid50/iid80/iid100/iid150/iid200/iid300/iid400/iid500 datasets and "
            "expects iid30/iid60/iid90/iid600/iid800/iid1000 from the missing-datasets payload."
        ),
        "source_reference_payloads": [
            "Comparison/config/graphene_5x5_snapshot_scaling_20_50_80_payload.json",
            "Comparison/config/graphene_5x5_snapshot_scaling_100_500_mixing_payload.json",
            "Comparison/config/graphene_5x2_snapshot_scaling_12_1300_600epochs_1train_payload.json",
        ],
        "dataset_mode": "full_strict_pipeline",
        "run_mode": "full_strict_pipeline",
        "dataset_root": "${REPO_ROOT}/Comparison/datasets/graphene_5x5_snapshot_scaling_complete_paper_ready",
        "output_root": "${REPO_ROOT}/Comparison/results/graphene_5x5_snapshot_scaling_complete_paper_ready",
        "dataset_sweep": {
            "enabled": True,
            "max_datasets": len(ALL_SIZES),
            "recipes": [recipe(size) for size in ALL_SIZES],
        },
        "training_sweep": {
            "enabled": True,
            "max_runs": len(ALL_SIZES) * 2,
            "apply_to_datasets": ["all"],
            "error_policy": "continue_on_error",
            "search_policy": {"strategy": "manual", "random_seed": 0},
            "manual_runs": manual_runs(),
        },
        "overwrite_datasets": False,
        "notes": {
            "intended_use": "Train Graph2Mat/DeepH on the complete graphene_5x5 snapshot-scaling grid.",
            "expected_dataset_sizes": ALL_SIZES,
            "existing_dataset_sizes": sorted(EXISTING_DATASET_ROOTS),
            "missing_dataset_sizes": MISSING_SIZES,
            "required_before_training": "Run graphene_5x5_snapshot_scaling_complete_paper_ready_missing_payload.json first.",
            "paper_ready_similarity": "Uses the same fixed 600-epoch anchors as graphene_5x2: G2M-T1000-03 and DH-T1000-04.",
            "manual_anchor_configs_per_dataset": 2,
            "configs_per_dataset": 2,
            "training_seeds": [],
            "single_training_per_model_dataset": True,
            "supercell": "5x5 graphene, 50 carbon atoms",
            "fdf_template": "materials/graphene_5x5/RUN.fdf",
        },
        "run_id": "graphene_5x5_snapshot_scaling_complete_paper_ready",
        "reuse_dataset_sweep_from_run_root": repo_token(str(CATALOG_ROOT)),
    }
    return payload


def catalog_row(size: int) -> dict[str, object]:
    dataset_root = EXISTING_DATASET_ROOTS.get(size) or MISSING_DATASET_ROOT / f"graphene_5x5_scale_iid{size}"
    return {
        "artifact_contract_version": "joint_graph2mat_deeph_artifact_contract_v1",
        "artifact_validation_path": str(dataset_root / "artifact_validation.json"),
        "artifact_validation_status": "valid",
        "benchmark_manifest_path": str(dataset_root / "benchmark_dataset_manifest.json"),
        "blocks": json.dumps(blocks(size), sort_keys=True),
        "compatibility_hash": f"graphene_5x5_iid{size}",
        "config_path": "",
        "dataset_root": str(dataset_root),
        "dataset_size": size,
        "dataset_slug": f"graphene_5x5_scale_iid{size}",
        "elapsed_seconds": 0.0,
        "frozen_split_manifest_path": str(dataset_root / "frozen_split_manifest.json"),
        "generation_seconds": 0.0,
        "recipe_id": f"graphene_5x5_scale_iid{size}",
        "recipe_label": f"Graphene 5x5 IID scale: {size} snapshots",
        "reserved_gap_frames": 2,
        "split_counts": json.dumps(split_counts(size), sort_keys=True),
        "split_mode": "blocked_with_gap",
        "status": "benchmark_ready",
        "temporal_gap": 1,
        "source": "existing_5x5_dataset" if size in EXISTING_DATASET_ROOTS else "generated_by_missing_payload",
    }


def catalog() -> dict[str, object]:
    rows = [catalog_row(size) for size in ALL_SIZES]
    return {
        "schema": "graph2mat_deeph_dataset_sweep_summary_v1",
        "created_at": "2026-07-10T00:00:00",
        "run_mode": "reuse_existing_and_generated_missing_datasets",
        "scientific_rule": "Train-only union of validated graphene_5x5 MD datasets; SIESTA/MD generation skipped by the training payload.",
        "artifact_contract_version": "joint_graph2mat_deeph_artifact_contract_v1",
        "dataset_root": "${REPO_ROOT}/Comparison/datasets/graphene_5x5_snapshot_scaling_complete_paper_ready",
        "max_datasets": len(rows),
        "total_datasets": len(rows),
        "total_snapshots": sum(ALL_SIZES),
        "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "recipe_set_hash": "graphene_5x5_complete_paper_ready_15_sizes",
        "rows": rows,
        "notes": {
            "existing_dataset_sizes": sorted(EXISTING_DATASET_ROOTS),
            "missing_dataset_sizes": MISSING_SIZES,
            "missing_payload": "Comparison/config/graphene_5x5_snapshot_scaling_complete_paper_ready_missing_payload.json",
        },
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    write_json(CONFIG_ROOT / "graphene_5x5_snapshot_scaling_complete_paper_ready_missing_payload.json", generation_payload())
    write_json(CONFIG_ROOT / "graphene_5x5_snapshot_scaling_complete_paper_ready_payload.json", training_payload())
    catalog_payload = catalog()
    write_json(CATALOG_ROOT / "summary" / "dataset_sweep_summary.json", catalog_payload)
    write_csv(CATALOG_ROOT / "summary" / "dataset_sweep_summary.csv", list(catalog_payload["rows"]))


if __name__ == "__main__":
    main()
