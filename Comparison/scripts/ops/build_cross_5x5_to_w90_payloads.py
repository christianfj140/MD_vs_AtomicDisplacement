#!/usr/bin/env python
"""Emit per-seed predict_metrics payloads for graphene 5x5 (50 atoms) -> w90 (2 atoms).

One payload per seed, because run_cross_structure_sweep_payload.py passes the same
``existing_artifacts`` to every seed of a ``seeds: [...]`` loop -- under
``predict_metrics`` nothing else depends on the seed, so a single multi-seed payload
would reuse one checkpoint and emit identical MAEs for all seeds (the bug the
graphene_5x5_vacancy campaign had to fix by retraining).

Seed 0 checkpoints come from the 5x5 paper-ready scaling campaign; seeds 1 and 2 come
from the vacancy campaign's seed_1/seed_2 retrains, which trained on the same 5x5
source datasets (train/val are source-only, so the checkpoints are target-agnostic).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "Comparison/config"

# Sizes where all three seeds have a summary-attested, successfully trained checkpoint.
# The vacancy seed_1/seed_2 retrains only covered 60..600 and several of those failed,
# so this is the intersection: seed1 trained {60,80,100,150,400,500,600} and seed2
# trained {60,80,100,300,400,500}.
SIZES = [60, 80, 100, 400, 500]

# Source dataset per size, matching what each checkpoint was actually trained on
# (cross-checked against source_root in the vacancy campaign summaries).
SOURCE_DIR = {
    60: "graphene_5x5_snapshot_scaling_complete_paper_ready_missing",
    80: "graphene_5x5_snapshot_scaling_20_50_80",
    100: "graphene_5x5_snapshot_scaling_100_500_mixing",
    400: "graphene_5x5_snapshot_scaling_100_500_mixing",
    500: "graphene_5x5_snapshot_scaling_100_500_mixing",
}

# Frozen test set: the OOD curve must vary only with source size, never the test set.
TARGET = (
    "Comparison/datasets/graphene_w90_snapshot_scaling_20_1100_600epochs_1train_derivatives"
    "/graphene_w90_scale_iid500"
)

PAPER_READY = (
    "Comparison/results/graphene_5x5_snapshot_scaling_complete_paper_ready"
    "/graphene_5x5_snapshot_scaling_complete_paper_ready/sweep"
)
VACANCY = "Comparison/results/ml_vs_siesta_cross_structure_vacancy"

HYPERPARAMS = {
    "graph2mat": {
        "batch_size": 32,
        "correlation": 3,
        "hidden_irreps": "48x0e + 48x1o + 48x2e + 48x3o",
        "loader_threads": 4,
        "loss": "graph2mat.metrics.block_type_huber",
        "loss_kwargs": {"beta": 0.01},
        "max_ell": 3,
        "max_epochs": 600,
        "num_interactions": 3,
        "optim_lr": 0.0018,
        "readout": "default",
    },
    "deeph": {
        "atom_fea_len": 128,
        "atom_update_net": "CGConv",
        "batch_size": 8,
        "criterion": "MaskMSELoss",
        "edge_fea_len": 128,
        "epochs": 600,
        "gauss_stop": 6,
        "if_edge_update": True,
        "if_lcmp": True,
        "learning_rate": 0.0003,
        "normalization": "LayerNorm",
        "num_l": 4,
        "optimizer": "adamW",
        "retain_edge_fea": True,
        "weight_decay": 0.0001,
    },
}

PERFORMANCE = {
    "preset": "paper_ready_parallel_trains_snapshot_scaling_no_global_batch",
    "compute_accelerator": "gpu",
    "store_in_memory": True,
    "reuse_validated_siesta_outputs": True,
    "enable_experiment_cache": False,
    # Pairs are fanned out per model stage: run_cross_structure_sweep sees
    # cross_model_schedule=deeph_then_graph2mat, runs the DeepH stage for every pair with
    # max_parallel_deeph_training_jobs workers, then the Graph2Mat stage with
    # max_parallel_graph2mat_training_jobs (it copies each into
    # max_parallel_prediction_jobs, which is what actually sizes the pair thread pool).
    "cross_model_schedule": "deeph_then_graph2mat",
    "max_parallel_deeph_training_jobs": 4,
    "max_parallel_graph2mat_training_jobs": 6,
    "max_parallel_dataset_jobs": 1,
    "max_parallel_siesta_jobs": 1,
    "max_parallel_prediction_jobs": 1,
    "max_parallel_evaluation_jobs": 3,
    "max_parallel_metric_jobs": 8,
    "omp_num_threads": 2,
    "mkl_num_threads": 2,
    "openblas_num_threads": 2,
    "numexpr_num_threads": 2,
    "torch_num_threads": 2,
    "torch_float32_matmul_precision": "high",
    "torch_mixed_precision": "bf16-mixed",
    "graph2mat_require_cuequivariance": True,
    "error_policy": "continue_on_error",
}


def _one(pattern: str, label: str) -> str:
    """Resolve a glob to exactly one existing path, relative to the repo root."""
    hits = sorted(glob.glob(str(REPO_ROOT / pattern)))
    if len(hits) != 1:
        raise SystemExit(f"{label}: expected 1 match, got {len(hits)} for {pattern}")
    return str(Path(hits[0]).relative_to(REPO_ROOT))


def _vacancy_run_roots(seed: int) -> dict[str, str]:
    """source_id -> run dir actually used by the vacancy retrain.

    Some pairs have several ``g2m_deeph_*`` dirs (aborted attempts plus reruns), so the
    campaign summary is the authority on which run produced the published seed MAE.
    """
    summary = json.loads(
        (REPO_ROOT / VACANCY / f"seed_{seed}" / "cross_structure_sweep_summary.json").read_text(encoding="utf-8")
    )
    roots: dict[str, str] = {}
    for perm in summary.get("permutations") or []:
        # Failed pairs still leave a run dir behind (sometimes with a usable-looking
        # checkpoint from the model that did succeed); only trained pairs are eligible.
        if perm.get("status") != "trained":
            continue
        run_root = ((perm.get("launch") or {}).get("runner_status") or {}).get("run_root")
        if run_root:
            roots[str(perm.get("source_id"))] = str(Path(run_root).relative_to(REPO_ROOT))
    return roots


def artifacts_for(seed: int, size: int) -> dict[str, str]:
    if seed == 0:
        g2m = _one(f"{PAPER_READY}/graph2mat/graphene_5x5_scale_iid{size}/*/graph2mat/training", f"g2m s0 N{size}")
        deeph = _one(f"{PAPER_READY}/deeph/graphene_5x5_scale_iid{size}/*/deeph/train", f"deeph s0 N{size}")
    else:
        source_id = f"graphene_5x5__graphene_5x5_scale_iid{size}"
        base = _vacancy_run_roots(seed).get(source_id)
        if base is None:
            raise SystemExit(f"seed {seed} N{size}: no run_root in the vacancy summary")
        g2m = _one(f"{base}/graph2mat/training", f"g2m s{seed} N{size}")
        deeph = _one(f"{base}/deeph/train", f"deeph s{seed} N{size}")

    # predict_metrics fails closed on these; verify now rather than mid-campaign.
    if not sorted((REPO_ROOT / g2m).rglob("best-*.ckpt")):
        raise SystemExit(f"no best-*.ckpt under {g2m}")
    for name in ("config.ini", "best_state_dict.pkl"):
        if not (REPO_ROOT / deeph / name).exists():
            raise SystemExit(f"missing {name} under {deeph}")
    return {"graph2mat_training_dir": g2m, "deeph_save_dir": deeph}


def build(seed: int) -> dict:
    pairs, existing = [], {}
    for size in SIZES:
        source = f"Comparison/datasets/{SOURCE_DIR[size]}/graphene_5x5_scale_iid{size}"
        pairs.append({"size": size, "direction": "5x5_to_w90", "source": source, "target": TARGET})
        existing[f"graphene_5x5_scale_iid{size}"] = artifacts_for(seed, size)
    return {
        "schema": "g2m_deeph_cross_structure_predict_metrics_payload_v1",
        "description": (
            f"Cross-testing 5x5 (50 atoms, train) -> graphene w90 (2 atoms, test), seed {seed}. "
            "Predict+metrics only: reuses checkpoints already trained on the 5x5 sources, no retraining. "
            "Frozen target graphene_w90_scale_iid500 so the OOD curve varies only with source size."
        ),
        "action": "predict_metrics",
        "sizes": SIZES,
        "models": ["graph2mat", "deeph"],
        "epochs": 600,
        "seed": seed,
        "seeds": [seed],
        "confirm_ghost_species_exemption": True,
        "confirm_incomplete_hamiltonian_semantics": True,
        "strict_dataset_validation": False,
        "sources": {},
        "targets": {},
        "pairs": pairs,
        "existing_artifacts": existing,
        "early_stopping": {"max_epochs": 600, "metric": "val_loss", "min_delta": 0.0, "mode": "min", "patience": 80},
        "hyperparams": HYPERPARAMS,
        "performance": PERFORMANCE,
        "notes": {
            "seed_policy": (
                "One payload per seed. run_cross_structure_sweep_payload.py reuses the same "
                "existing_artifacts for every seed of a seeds:[...] loop, and predict_metrics "
                "does not train, so a multi-seed payload would emit identical MAEs per seed."
            ),
            "checkpoint_provenance": (
                "seed 0: graphene_5x5_snapshot_scaling_complete_paper_ready. "
                "seeds 1-2: ml_vs_siesta_cross_structure_vacancy seed_1/seed_2 retrains, which "
                "trained on these same 5x5 sources (train/val are source-only, so target-agnostic)."
            ),
            "size_policy": "N in {20,30,50,60,80,100,400,500}: the sizes with complete artifacts for all 3 seeds.",
        },
    }


def main() -> None:
    for seed in (0, 1, 2):
        path = CONFIG_DIR / f"cross_5x5_to_w90_predict_metrics_seed{seed}_payload.json"
        path.write_text(json.dumps(build(seed), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
