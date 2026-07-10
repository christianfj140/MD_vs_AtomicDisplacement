#!/usr/bin/env python
"""Relaunch ONLY the paper-ready mixing permutations that OOM'd in the 17h sweep.

The original sweep (Comparison/config/ml_vs_siesta_mixing_sweep_20_500_paper_ready_train_payload.json)
ran 48 permutations x 2 models. 20 Graph2Mat runs died on CUDA OOM (7 replicas
of the 64x0e+64x1o+64x2e+64x3o net at batch_size=40 did not fit in ~31.4GiB);
the same 4 also killed DeepH. This reruns exactly those failures -- no cartesian
product, no re-training of the 28 G2M + 44 DeepH that already succeeded.

It reuses the EXACT mechanism of pipeline_ui._run_mixing_sweep_parallel Phase 2:
a hand-built datasets_list ({dataset_id: dataset_root}) plus manual_runs that
reference those ids, injected into Graph2MatDeepHBenchmarkRunner via the same
_training_sweep_datasets monkey-patch. Datasets are reused in place
(dataset_mode=reuse_validated, strict_dataset_validation=false) -- no
regeneration. Only max_parallel_graph2mat_training_jobs is lowered (7 -> 2) so
even the worst-case size500 replicas (~14GB each) fit with margin.

Dry-run by default (validates payload + that every dataset_root exists and
passes the manifest gate). Pass --apply to actually train.

    .venv/bin/python Comparison/scripts/refill_mixing_paper_ready_oom.py           # dry-run
    .venv/bin/python Comparison/scripts/refill_mixing_paper_ready_oom.py --apply   # train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner  # noqa: E402

# Source of the already-materialized datasets (the original 17h sweep output).
SWEEP_ROOT = REPO_ROOT / "Comparison" / "results" / "ml_vs_siesta_mixing_sweep_20_500_paper_ready"
# Fresh output so we never overwrite the 28 G2M + 44 DeepH successes.
OUTPUT_ROOT = REPO_ROOT / "Comparison" / "results" / "ml_vs_siesta_mixing_sweep_20_500_paper_ready_refill"

# 20 Graph2Mat permutations to relaunch (dataset_root = SWEEP_ROOT/<name>).
G2M_PERMUTATIONS = [
    "size50_add_r0p500", "size50_add_r1p000", "size50_replace_r1p000",
    "size100_add_r0p500", "size100_add_r1p000", "size100_replace_r0p500", "size100_replace_r1p000",
    "size150_add_r0p500", "size150_replace_r0p500", "size150_replace_r1p000",
    "size200_replace_r0p500", "size200_replace_r1p000",
    "size300_add_r1p000", "size300_replace_r1p000",
    "size400_add_r1p000", "size400_replace_r0p500", "size400_replace_r1p000",
    "size500_add_r0p500", "size500_add_r1p000", "size500_replace_r1p000",
]
# 4 DeepH permutations to relaunch (subset of the above).
DEEPH_PERMUTATIONS = [
    "size100_add_r1p000", "size100_replace_r1p000",
    "size200_add_r1p000", "size500_add_r1p000",
]

# Verbatim from the original paper-ready sweep payload (do NOT change).
G2M_OVERRIDES: dict[str, Any] = {
    "max_epochs": 750, "optim_lr": 0.0018, "batch_size": 40, "loader_threads": 4,
    "num_interactions": 3, "correlation": 3, "max_ell": 3,
    "hidden_irreps": "64x0e + 64x1o + 64x2e + 64x3o",
    "loss": "graph2mat.metrics.block_type_huber", "loss_kwargs": {"beta": 0.006},
    "edge_block_readout": "graph2mat.bindings.e3nn.E3nnEdgeBlockNodeMix",
    "node_block_readout": "graph2mat.bindings.e3nn.E3nnSimpleNodeBlock",
    "preprocessing_edges": "graph2mat.bindings.e3nn.E3nnEdgeMessageBlock",
    "preprocessing_edges_reuse_nodes": True,
    "readout": "edge_node_mix", "seed_everything": 1,
}
DEEPH_OVERRIDES: dict[str, Any] = {
    "epochs": 750, "batch_size": 4, "learning_rate": 0.0003, "criterion": "MaskMSELoss",
    "retain_edge_fea": True, "weight_decay": 0.0, "optimizer": "adamW",
    "edge_fea_len": 192, "normalization": "LayerNorm", "atom_update_net": "CGConv",
    "if_lcmp": True, "gauss_stop": 6, "atom_fea_len": 192, "num_l": 4,
    "if_edge_update": True, "seed": 1, "num_threads": 2,
}

# Original performance block, but G2M parallelism lowered 7 -> 2 to avoid OOM.
PERFORMANCE: dict[str, Any] = {
    "compute_accelerator": "gpu",
    "store_in_memory": True,
    "reuse_validated_siesta_outputs": True,
    "enable_experiment_cache": False,
    "max_parallel_prediction_jobs": 1,
    "max_parallel_evaluation_jobs": 3,
    "max_parallel_metric_jobs": 8,
    "max_parallel_graph2mat_training_jobs": 1,
    "max_parallel_deeph_training_jobs": 2,
    "model_batch_schedule": "alternating",
    "model_batch_start": "deeph",
    "omp_num_threads": 2, "mkl_num_threads": 2, "openblas_num_threads": 2,
    "numexpr_num_threads": 2, "torch_num_threads": 2,
    "torch_float32_matmul_precision": "high",
    "torch_mixed_precision": "bf16-mixed",
    "graph2mat_log_every_n_steps": 1,
    "graph2mat_check_val_every_n_epoch": 1,
    "graph2mat_checkpoint_every_n_epochs": 1,
    "graph2mat_require_cuequivariance": True,
    "error_policy": "continue_on_error",
}
DEEPH_BLOCK: dict[str, Any] = {
    "repo_path": "/home/christian/repositorios/DeepH-pack",
    "device": "cuda:0", "disable_cuda": False, "num_threads": 2, "multiprocessing": 0,
}

_POLL_SECONDS = 5.0


def _build_plan() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (datasets_list, manual_runs) with one dataset_id per permutation.

    A permutation that trains both models shares a single dataset_id (so the
    root is validated once), same as the anchor payloads.
    """
    all_perms = list(dict.fromkeys(G2M_PERMUTATIONS + DEEPH_PERMUTATIONS))
    datasets_list = [
        {"dataset_id": name, "dataset_root": str(SWEEP_ROOT / name)}
        for name in all_perms
    ]
    manual_runs: list[dict[str, Any]] = []
    for name in G2M_PERMUTATIONS:
        manual_runs.append({
            "model": "graph2mat", "dataset_id": name,
            "config_id": f"graph2mat_{name}", "seed": 0,
            "overrides": dict(G2M_OVERRIDES),
        })
    for name in DEEPH_PERMUTATIONS:
        manual_runs.append({
            "model": "deeph", "dataset_id": name,
            "config_id": f"deeph_{name}", "seed": 0,
            "overrides": dict(DEEPH_OVERRIDES),
        })
    return datasets_list, manual_runs


def _build_payload(datasets_list: list[dict[str, Any]], manual_runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_mode": "reuse_validated",
        # Runner validates this top-level root; per-run roots come from the
        # injected datasets_list via the _training_sweep_datasets patch.
        "dataset_root": datasets_list[0]["dataset_root"],
        "output_root": str(OUTPUT_ROOT / "parallel_run"),
        "allow_regenerate_siesta": False,
        # Mixed datasets never carry strict-only single-SIESTA provenance.
        "strict_dataset_validation": False,
        "performance": PERFORMANCE,
        "deeph": DEEPH_BLOCK,
        "training_sweep": {
            "enabled": True,
            "error_policy": "continue_on_error",
            "search_policy": {"strategy": "manual", "random_seed": 0},
            "manual_runs": manual_runs,
        },
        "_mixing_datasets_list": datasets_list,
    }


def _validate_datasets(datasets_list: list[dict[str, Any]]) -> list[str]:
    """Return a list of human-readable problems; empty means all roots are usable."""
    problems: list[str] = []
    for entry in datasets_list:
        root = Path(entry["dataset_root"])
        if not root.is_dir():
            problems.append(f"MISSING dir: {root}")
            continue
        manifest = root / "benchmark_dataset_manifest.json"
        if not manifest.is_file():
            problems.append(f"MISSING manifest: {manifest}")
            continue
        try:
            data = json.loads(manifest.read_text())
        except Exception as exc:  # noqa: BLE001
            problems.append(f"UNREADABLE manifest {manifest}: {exc}")
            continue
        if not data.get("benchmark_ready", False):
            problems.append(f"NOT benchmark_ready: {root}")
        if not (root / "frozen_split_manifest.json").is_file():
            problems.append(f"MISSING frozen_split_manifest: {root}")
        if not (root / "splits").is_dir():
            problems.append(f"MISSING splits/: {root}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually train (default: dry-run validation only).")
    args = parser.parse_args()

    datasets_list, manual_runs = _build_plan()
    payload = _build_payload(datasets_list, manual_runs)

    print(f"[refill] output_root: {OUTPUT_ROOT}")
    print(f"[refill] datasets: {len(datasets_list)} | manual_runs: {len(manual_runs)} "
          f"({len(G2M_PERMUTATIONS)} G2M + {len(DEEPH_PERMUTATIONS)} DeepH)")
    print(f"[refill] max_parallel_graph2mat_training_jobs="
          f"{PERFORMANCE['max_parallel_graph2mat_training_jobs']} "
          f"max_parallel_deeph_training_jobs={PERFORMANCE['max_parallel_deeph_training_jobs']}")

    problems = _validate_datasets(datasets_list)
    if problems:
        print(f"[refill] DATASET VALIDATION FAILED ({len(problems)} problems):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"[refill] All {len(datasets_list)} dataset roots exist and pass the manifest gate.")

    # Payload must be JSON-serializable (parseable end to end).
    json.dumps(payload)
    print("[refill] Payload parses (json.dumps OK).")

    if not args.apply:
        print("[refill] DRY-RUN OK. Re-run with --apply to launch the ~hours-long training.")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    runner = Graph2MatDeepHBenchmarkRunner()

    def _patched_training_sweep_datasets(_validation: dict[str, Any]) -> list[dict[str, Any]]:
        return datasets_list

    runner._training_sweep_datasets = _patched_training_sweep_datasets  # type: ignore[method-assign]

    print("[refill] Launching runner...")
    runner.start(payload)
    log_offset = 0
    while True:
        try:
            log_payload = runner.logs(since=log_offset, limit=None)
            for line in log_payload.get("lines") or []:
                sys.stdout.write(str(line))
            log_offset = int(log_payload.get("offset") or log_offset)
        except Exception as exc:  # noqa: BLE001
            print(f"[refill][WARN] log read failed: {exc}")
        if not runner.status().get("running"):
            break
        time.sleep(_POLL_SECONDS)

    # Drain remaining logs.
    try:
        log_payload = runner.logs(since=log_offset, limit=None)
        for line in log_payload.get("lines") or []:
            sys.stdout.write(str(line))
    except Exception:  # noqa: BLE001
        pass

    results = runner.results()
    status = results.get("status") or {}
    run_root = status.get("run_root") or str(OUTPUT_ROOT / "parallel_run")
    print(f"[refill] Runner finished. run_root: {run_root}")
    print(f"[refill] status: {json.dumps({k: status.get(k) for k in ('running', 'error', 'returncode') if k in status})}")
    print(f"[refill] Consolidation: copy each {run_root}/sweep/graph2mat/<name>/ and "
          f"{run_root}/sweep/deeph/<name>/ into the matching perm_N/ dir of the original sweep manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
