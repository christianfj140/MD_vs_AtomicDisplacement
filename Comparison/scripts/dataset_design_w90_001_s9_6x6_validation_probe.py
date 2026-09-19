#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9 -- single-design 6x6 (72-atom, all-atom displacement) validation probe.

Purpose: get REAL SIESTA-per-structure and Graph2Mat-per-epoch timing at 6x6
scale (materials/graphene_6x6/RUN.fdf, 72 atoms) before committing to a top-9/
top-20 6x6 campaign -- every number used so far for 6x6 was extrapolated from
w90 (2 atoms, 1.25s/calc) and 5x5 (50 atoms, ~31s/MD-step) real measurements,
not measured at 72 atoms itself.

Replicates the #1-ranked w90 pilot design (sobol_sparse, 2D_in, k=1,
R_train_max=0.05, density=64, dev H-MAE=279.0 meV) but with k=72 (every atom
independently displaced -- sobol_sparse gives each active atom its own draw,
see w90_displacement_sampler_family.generate_sobol_sparse; this only works
since select_active_atoms() was just extended to support k=n_atoms_total).

Trains with the same cross-testing hyperparameters already validated on w90
today (num_interactions=3, correlation=3, max_ell=3, hidden_irreps=48x...,
lr=0.0018, batch_size=32, max_epochs=600 cap, early stopping patience=80,
precision=bf16-mixed, loader_threads=4).
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
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import w90_displacement_sampler_family as sampler  # noqa: E402
from run_epc_siesta_reference import canonical_ang_base_fdf  # noqa: E402
from run_hamiltonian_derivative_siesta_references import (  # noqa: E402
    DerivativeSiestaReferenceError,
    run_derivative_siesta_references,
)
from fdf_materialization import extract_fdf_structure, materialize_sample_fdf  # noqa: E402
import run_w90_displacement_siesta_pilot as pilot  # noqa: E402
import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_6x6_validation_probe"

TRAIN_SEED, TEST_SEED = 0, 999

CROSS_TESTING_MODEL: dict[str, Any] = {
    "num_interactions": 3, "correlation": 3, "max_ell": 3,
    "hidden_irreps": "48x0e + 48x1o + 48x2e + 48x3o",
    "loss": "graph2mat.metrics.block_type_mae", "optim_lr": 0.0018,
}
CROSS_TESTING_MAX_EPOCHS = 600
CROSS_TESTING_BATCH_SIZE = 32
CROSS_TESTING_EARLY_STOPPING_PATIENCE = 80
CROSS_TESTING_PRECISION = "bf16-mixed"
CROSS_TESTING_LOADER_THREADS = 4


class SensorSampler:
    """Background `sensors` poller; records peak x86_pkg_temp (°C) and warns above threshold."""

    def __init__(self, warn_above_c: float = 70.0, interval_s: float = 5.0) -> None:
        self.warn_above_c = warn_above_c
        self.interval_s = interval_s
        self.peak_c = 0.0
        self.warned = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _read_pkg_temp_c(self) -> float | None:
        try:
            out = subprocess.run(["sensors"], capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return None
        for line in out.splitlines():
            if "Package id 0" in line or "Tctl" in line or "x86_pkg_temp" in line:
                try:
                    return float(line.split("+")[1].split(".")[0])
                except Exception:
                    continue
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            temp = self._read_pkg_temp_c()
            if temp is not None:
                self.peak_c = max(self.peak_c, temp)
                if temp > self.warn_above_c and not self.warned:
                    print(f"[SensorSampler][WARN] CPU package temp {temp}C exceeds {self.warn_above_c}C", file=sys.stderr)
                    self.warned = True
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "SensorSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _configs_for(geometry: sampler.Geometry, seed: int, n: int, dim: str, amplitude_ang: float, run_tag: str) -> list[tuple[str, sampler.Configuration]]:
    k = len(geometry.positions_ang)
    configs = sampler.generate_sobol_sparse(geometry, k=k, dim=dim, amplitude_ang=amplitude_ang, max_n=n, seed=seed)
    return [(f"g6x6_val__{run_tag}__seed{seed}__{i:03d}", c) for i, c in enumerate(configs)]


def prepare_and_label(workers: int, *, dim: str, amplitude_ang: float, n_train: int, n_test: int, run_tag: str) -> Path:
    structures_root = OUTPUT_ROOT / "structures"
    canonical_base_fdf = OUTPUT_ROOT / f"_canonical_base__{run_tag}.fdf"
    canonical_ang_base_fdf(sampler.GRAPHENE_6X6_FDF, canonical_base_fdf)
    geometry = sampler.load_graphene_6x6()
    print(json.dumps({"n_atoms": len(geometry.positions_ang)}, sort_keys=True))

    base_structure = extract_fdf_structure(canonical_base_fdf)
    base_positions = [tuple(p) for p in base_structure.positions_ang]

    all_entries = (
        _configs_for(geometry, TRAIN_SEED, n_train, dim, amplitude_ang, run_tag)
        + _configs_for(geometry, TEST_SEED, n_test, dim, amplitude_ang, run_tag)
    )
    for sample_id, config in all_entries:
        displaced = [
            tuple(base_positions[i][axis] + config.displacements_ang.get(i, (0.0, 0.0, 0.0))[axis] for axis in range(3))
            for i in range(len(base_positions))
        ]
        out_dir = structures_root / sample_id
        materialize_sample_fdf(canonical_base_fdf, out_dir / "RUN.fdf", positions_ang=displaced, single_point=True)

    try:
        run_manifest = run_derivative_siesta_references(
            stencil_root=OUTPUT_ROOT,
            source_dataset_root=REPO_ROOT / "materials/graphene",
            siesta_command=pilot.DEFAULT_SIESTA_COMMAND,
            workers=workers,
            diagnostic_only=True,
        )
    except DerivativeSiestaReferenceError as exc:
        raise RuntimeError(f"SIESTA labeling failed: {exc}") from exc

    ids = {sid for sid, _ in all_entries}
    failed = [r["sample_id"] for r in run_manifest["rows"] if r["sample_id"] in ids and r["status"] not in {"ok", "staged", "skipped_existing"}]
    if failed:
        raise RuntimeError(f"{len(failed)} 6x6 geometries failed SIESTA labeling: {failed[:10]}")
    return Path(run_manifest["output_reference_root"])


def _sample_for(sample_id: str, reference_root: Path, dim: str, amplitude_ang: float) -> s4.Sample:
    structure_dir = OUTPUT_ROOT / "structures" / sample_id
    reference_dir = reference_root / sample_id
    selection_path = None
    for candidate in ("graphene_6x6.TSHS", "graphene_6x6.HSX"):
        if (reference_dir / candidate).exists():
            selection_path = reference_dir / candidate
            break
    if selection_path is None:
        matches = list(reference_dir.glob("*.TSHS")) + list(reference_dir.glob("*.HSX"))
        selection_path = matches[0] if matches else reference_dir / "MISSING"
    return s4.Sample(
        sample_id=sample_id, family="sobol_sparse", dim=dim, k=72, amplitude_ang=amplitude_ang,
        run_fdf=structure_dir / "RUN.fdf", reference_dir=reference_dir, reference_matrix=selection_path,
    )


class VramSampler:
    """Background nvidia-smi poller; records peak MiB used on the GPU (whole-device, not per-process)."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6, help="SIESTA labeling workers (kept modest: CPU thermal caution).")
    parser.add_argument("--dim", default="2D_in")
    parser.add_argument("--amplitude-ang", type=float, default=0.05)
    parser.add_argument("--n-train", type=int, default=40)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--run-tag", default="default", help="Distinguishes concurrent probes (paths, report filename).")
    parser.add_argument(
        "--skip-labeling", action="store_true",
        help="Reuse an already-completed run's structures/siesta_hamiltonians for this run_tag verbatim -- "
             "no materialize_sample_fdf, no SIESTA subprocess at all. For re-testing the training phase "
             "(e.g. GPU-parallelism) without re-touching the CPU/thermal-sensitive labeling path.",
    )
    args = parser.parse_args()

    backend_info = s4.torch_backend_preflight()
    accelerator = backend_info["effective_backend"]
    print(json.dumps({"gpu_preflight": backend_info}, sort_keys=True))

    if args.skip_labeling:
        reference_root = OUTPUT_ROOT / "siesta_hamiltonians"
        siesta_seconds = 0.0
        cpu_temp = SensorSampler()  # never entered; peak_c stays 0, reported as "not measured this run"
    else:
        with SensorSampler() as cpu_temp:
            t0 = time.perf_counter()
            reference_root = prepare_and_label(
                workers=args.workers, dim=args.dim, amplitude_ang=args.amplitude_ang,
                n_train=args.n_train, n_test=args.n_test, run_tag=args.run_tag,
            )
            siesta_seconds = time.perf_counter() - t0

    geometry = sampler.load_graphene_6x6()
    train_samples = [
        _sample_for(sid, reference_root, args.dim, args.amplitude_ang)
        for sid, _ in _configs_for(geometry, TRAIN_SEED, args.n_train, args.dim, args.amplitude_ang, args.run_tag)
    ]
    test_samples = [
        _sample_for(sid, reference_root, args.dim, args.amplitude_ang)
        for sid, _ in _configs_for(geometry, TEST_SEED, args.n_test, args.dim, args.amplitude_ang, args.run_tag)
    ]

    run_name = f"6x6__{args.run_tag}__sobol_sparse__{args.dim}__k72__r{args.amplitude_ang}__N{args.n_train}"
    run_dir = OUTPUT_ROOT / "runs" / run_name
    with SensorSampler() as cpu_temp_train, VramSampler() as vram:
        config_path = s4.build_graph2mat_config(
            run_dir, train_samples, test_samples, run_name=run_name,
            max_epochs=CROSS_TESTING_MAX_EPOCHS, accelerator=accelerator, training_seed=0,
            model_overrides=CROSS_TESTING_MODEL, batch_size=CROSS_TESTING_BATCH_SIZE,
            early_stopping_patience=CROSS_TESTING_EARLY_STOPPING_PATIENCE,
            precision=CROSS_TESTING_PRECISION, loader_threads=CROSS_TESTING_LOADER_THREADS,
        )
        train_start = time.perf_counter()
        checkpoint = s4.run_graph2mat_training(config_path, run_dir)
        train_seconds = time.perf_counter() - train_start

        val_manifest_path = run_dir / "test_manifest.csv"
        s4.write_val_manifest(val_manifest_path, test_samples)
        predicted_root = s4.run_prediction(checkpoint, val_manifest_path, run_dir / "test_predictions", accelerator)

    metrics = s4.compute_validation_metrics(test_samples, predicted_root)

    result = {
        "n_atoms": 72, "family": "sobol_sparse", "dim": args.dim, "amplitude_ang": args.amplitude_ang,
        "n_train": args.n_train, "n_test": args.n_test, "run_tag": args.run_tag,
        "siesta_seconds_total": siesta_seconds,
        "siesta_seconds_per_structure": siesta_seconds / (args.n_train + args.n_test),
        "train_seconds": train_seconds,
        "peak_cpu_pkg_temp_c_siesta_phase": cpu_temp.peak_c,
        "peak_cpu_pkg_temp_c_train_phase": cpu_temp_train.peak_c,
        "peak_gpu_vram_mib_whole_device": vram.peak_mib,
        "metrics_eV": metrics,
        "H_MAE_meV": metrics["H_MAE"] * 1000 if metrics["H_MAE"] == metrics["H_MAE"] else None,
    }
    report_path = OUTPUT_ROOT / f"validation_report__{args.run_tag}.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
