#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9 -- single-design probe with cross-testing hyperparameters.

S9's pilot/confirmation trained every design with a deliberately tiny, fast
architecture (num_interactions=1, correlation=1, max_ell=2,
hidden_irreps=10x0e+10x1o+10x2e, lr=0.005, batch_size<=10, 60 epochs --
MODEL_OVERRIDES/PILOT_MAX_EPOCHS in dataset_design_w90_001_s4_train_learning_curves.py),
so it could screen hundreds of designs quickly. That is ~4-7x worse H-MAE than the
repo's separate MD-trajectory study (Graph2Mat plateaus at ~63-69 meV for any N_train).

That MD study's Graph2Mat runs used the repo's "cross testing" hyperparameters
(Comparison/results/ml_vs_siesta_cross_structure_sweep/*/graph2mat/pipeline_config.yaml,
consistent across every run inspected): num_interactions=3, correlation=3, max_ell=3,
hidden_irreps=48x0e+48x1o+48x2e+48x3o, optim_lr=0.0018, batch_size=32, max_epochs=600
(cap), EarlyStopping(monitor=val_loss, mode=min, patience=80, min_delta=0.0).

This script retrains ONE already-picked-winner S9 design (default: the cheapest
pareto_optimal design in S9-S5's confirmation results,
axial_radial__1D_in__k1__r0.080__res2__seeddeterministic, N_train=256) with those
matched hyperparameters, evaluates on the same development set S9 always uses, and
prints a direct H-MAE comparison against both the tiny-pilot number and the ~63-69 meV
MD figure -- to test whether the training-budget/capacity gap (not the displacement
sampling method itself) explains most of S9's worse accuracy.

Also records peak GPU VRAM during training (sampled via nvidia-smi in a background
thread) since the cross-testing architecture is ~5x more parameters and this number
is needed to size --parallel-designs if this hyperparameter set is later rolled out
to more S9 designs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402
import dataset_design_w90_001_s5_frozen_tests_generalization as s5  # noqa: E402
import dataset_design_w90_001_s9_s4_pilot_screening as s94  # noqa: E402
import dataset_design_w90_001_s9_s5_pareto_training as s95  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_cross_testing_hparam_probe"
DEFAULT_DESIGN_ID = "axial_radial__1D_in__k1__r0.080__res2__seeddeterministic"
DEFAULT_N_TRAIN = 256

# Verbatim from Comparison/results/ml_vs_siesta_cross_structure_sweep/*/graph2mat/
# pipeline_config.yaml (identical across every run inspected).
CROSS_TESTING_MODEL: dict[str, Any] = {
    "num_interactions": 3,
    "correlation": 3,
    "max_ell": 3,
    "hidden_irreps": "48x0e + 48x1o + 48x2e + 48x3o",
    "loss": "graph2mat.metrics.block_type_mae",
    "optim_lr": 0.0018,
}
CROSS_TESTING_MAX_EPOCHS = 600
CROSS_TESTING_BATCH_SIZE = 32
CROSS_TESTING_EARLY_STOPPING_PATIENCE = 80  # EarlyStopping(monitor=val_loss, mode=min, patience=80, min_delta=0.0)
CROSS_TESTING_PRECISION = "bf16-mixed"
CROSS_TESTING_TORCH_FLOAT32_MATMUL_PRECISION = "high"  # not actually applied, see build_graph2mat_config
CROSS_TESTING_LOADER_THREADS = 4

# Reference numbers this probe's result gets compared against (see module docstring).
S9_TINY_PILOT_MEDIAN_H_MAE_MEV = 413.7
MD_STUDY_GRAPH2MAT_H_MAE_MEV_RANGE = (62.6, 69.1)


class VramSampler:
    """Background nvidia-smi poller; records peak MiB used by the current process tree."""

    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10, check=True,
                ).stdout.strip()
                used = max(int(line.strip()) for line in out.splitlines() if line.strip())
                self.peak_mib = max(self.peak_mib, used)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "VramSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def load_cached_development_set(geometry: Any) -> list[s4.Sample]:
    entries = s5.build_common_validation_configs(geometry)
    reference_root = s94.DEFAULT_S5_ROOT / s5.COMMON_VALIDATION_NAME / "siesta_hamiltonians"
    samples = s5.samples_from_run(
        entries,
        {"output_reference_root": reference_root, "usable_ids": [sample_id for sample_id, _ in entries]},
    )
    if len(samples) != len(entries):
        raise RuntimeError(
            f"cached w90 development set is incomplete: {len(samples)}/{len(entries)} usable references"
        )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-id", default=DEFAULT_DESIGN_ID)
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-tag", default="crosstesting_hparams", help="Distinguishes repeated probes (e.g. a bf16/tf32 speed re-measurement) in run_dir naming.")
    parser.add_argument(
        "--source", choices=("confirmation", "pilot"), default="confirmation",
        help="'confirmation': design must be an S9-S5 Pareto survivor (N=128/256, extrapolated resolution). "
             "'pilot': any S9-S3/S9-S4 trainable candidate at its own registered density -- covers the full "
             "80-combo cross-testing campaign, not just the handful that reached S9-S5.",
    )
    args = parser.parse_args()

    backend_info = s4.torch_backend_preflight()
    accelerator = backend_info["effective_backend"]
    print(json.dumps({"gpu_preflight": backend_info}, sort_keys=True))

    geometry = s94.sampler.load_graphene_primitive()
    if args.source == "confirmation":
        survivors, geometry, _hash_to_sample = s95.load_survivors()
        entry = next((e for e in survivors if e["design_id"] == args.design_id), None)
        if entry is None:
            print(f"design_id {args.design_id!r} not found among S9-S5 survivors", file=sys.stderr)
            return 2
        # output_root=s95.DEFAULT_OUTPUT_ROOT (not this script's own OUTPUT_ROOT): this
        # design was already trained at N=256 by S9-S5, so its training pool's SIESTA
        # references already exist under that root's training_pools/ dir --
        # materialize_and_run's skip-if-exists check only fires if we point at the SAME
        # directory it originally wrote to, otherwise it silently re-submits real SIESTA
        # jobs under a fresh path (caught empirically: first run of this script spawned
        # 8 new siesta processes instead of reusing the cache).
        pools, siesta_timers = s95.build_training_pools(
            entry, geometry, output_root=s95.DEFAULT_OUTPUT_ROOT, siesta_command=s94.pilot.DEFAULT_SIESTA_COMMAND,
            workers=args.workers, n_levels=(args.n_train,),
        )
        train_samples = pools[args.n_train]
    else:
        manifest = s94.load_manifest()
        s3_pool = s94.s4.load_s3_samples()
        hash_to_sample = s94.build_s3_hash_index(s3_pool)
        trainable, _pruned = s94.select_candidates(manifest, geometry, hash_to_sample)
        entry = next((e for e in trainable if e["design_id"] == args.design_id), None)
        if entry is None:
            print(f"design_id {args.design_id!r} not found among S9-S4 trainable pilot candidates", file=sys.stderr)
            return 2
        # entry["_matched_samples"] (set by select_candidates itself) is exactly this
        # design's own real-SIESTA-backed pool -- already labeled during the pilot, no
        # new SIESTA whatsoever, matched by geometry hash like the pilot's own trainer.
        train_samples = entry["_matched_samples"]
        siesta_timers = {}

    dev_samples = load_cached_development_set(geometry)

    run_name = f"{entry['design_id']}__N{args.n_train}__{args.run_tag}"
    run_dir = OUTPUT_ROOT / "runs" / run_name

    with VramSampler() as vram:
        config_path = s4.build_graph2mat_config(
            run_dir, train_samples, dev_samples, run_name=run_name,
            max_epochs=CROSS_TESTING_MAX_EPOCHS, accelerator=accelerator, training_seed=0,
            model_overrides=CROSS_TESTING_MODEL, batch_size=CROSS_TESTING_BATCH_SIZE,
            early_stopping_patience=CROSS_TESTING_EARLY_STOPPING_PATIENCE,
            precision=CROSS_TESTING_PRECISION,
            torch_float32_matmul_precision=CROSS_TESTING_TORCH_FLOAT32_MATMUL_PRECISION,
            loader_threads=CROSS_TESTING_LOADER_THREADS,
        )
        train_start = time.perf_counter()
        checkpoint = s4.run_graph2mat_training(config_path, run_dir)
        train_seconds = time.perf_counter() - train_start

        val_manifest_path = run_dir / "dev_manifest.csv"
        s4.write_val_manifest(val_manifest_path, dev_samples)
        predicted_root = s4.run_prediction(checkpoint, val_manifest_path, run_dir / "dev_predictions", accelerator)

    metrics = s4.compute_validation_metrics(dev_samples, predicted_root)

    result = {
        "design_id": entry["design_id"],
        "n_train": args.n_train,
        "n_train_actual": len(train_samples),
        "hyperparameters": {
            **CROSS_TESTING_MODEL, "max_epochs": CROSS_TESTING_MAX_EPOCHS, "batch_size": CROSS_TESTING_BATCH_SIZE,
            "early_stopping_patience": CROSS_TESTING_EARLY_STOPPING_PATIENCE,
            "precision": CROSS_TESTING_PRECISION,
            "torch_float32_matmul_precision": "not_applied (graph2mat CLI rejects this key directly; see build_graph2mat_config note)",
            "loader_threads": CROSS_TESTING_LOADER_THREADS,
        },
        "accelerator": accelerator,
        "train_seconds": train_seconds,
        "siesta_timers": siesta_timers,
        "peak_gpu_vram_mib": vram.peak_mib,
        "metrics_eV": metrics,
        "H_MAE_meV": metrics["H_MAE"] * 1000 if metrics["H_MAE"] == metrics["H_MAE"] else None,
        "comparison": {
            "s9_tiny_pilot_median_H_MAE_meV": S9_TINY_PILOT_MEDIAN_H_MAE_MEV,
            "md_study_graph2mat_H_MAE_meV_range": MD_STUDY_GRAPH2MAT_H_MAE_MEV_RANGE,
        },
    }
    # Includes design_id (not just run_tag): concurrent probes for different designs
    # sharing one --run-tag would otherwise race on the same report_path and clobber
    # each other's results (caught empirically running 3 candidates in parallel).
    report_path = OUTPUT_ROOT / f"probe_report__{entry['design_id']}__{args.run_tag}.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
