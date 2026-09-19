#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9 -- the 80 distinct w90 combos, replicated on the
6x6 supercell (72 atoms, k=n_atoms -- every atom independently displaced),
with cross-testing hyperparameters. Paired with
dataset_design_w90_001_s9_crosstesting_campaign_w90.py's w90-side run.

Uses dataset_design_w90_001_s9_s3_design_generation.generate_design(...) --
the SAME dispatcher the S9 pilot used to build these 80 recipes on w90 -- so
each combo's family/dim/R_train_max/density semantics carry over exactly,
just with k=72 instead of k=1/2 (only valid for sobol_sparse/random_cartesian/
latin_hypercube; the 80-combo list is already restricted to those).

Two phases, run once for the whole campaign (not per-design):
  1. Generate + label every combo's train+test structures in ONE combined
     SIESTA batch (modest workers -- this repo's CPU hit 86 C at 6 workers
     for a single 6x6 design; --workers defaults to 5).
  2. Train all 80 with the cross-testing hyperparameters, bounded parallel
     (measured 2x14.4GB fits in 32GB with ~1.5GB to spare; default --parallel 2).
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
import dataset_design_w90_001_s9_s3_design_generation as s3  # noqa: E402
from run_epc_siesta_reference import canonical_ang_base_fdf  # noqa: E402
from run_hamiltonian_derivative_siesta_references import (  # noqa: E402
    DerivativeSiestaReferenceError,
    run_derivative_siesta_references,
)
from fdf_materialization import extract_fdf_structure, materialize_sample_fdf  # noqa: E402
import run_w90_displacement_siesta_pilot as pilot  # noqa: E402
import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402

COMBOS_PATH = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_chain_logs/crosstesting_campaign_80combos.json"
OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_6x6_campaign"
SUMMARY_PATH = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_chain_logs/6x6_campaign_summary.json"

TRAIN_SEED, TEST_SEED = 0, 999
N_TEST = 8

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
    def __init__(self, warn_above_c: float = 70.0, interval_s: float = 5.0) -> None:
        self.warn_above_c = warn_above_c
        self.interval_s = interval_s
        self.peak_c = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _read_pkg_temp_c(self) -> float | None:
        try:
            out = subprocess.run(["sensors"], capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return None
        for line in out.splitlines():
            if "Package id 0" in line:
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
                if temp > self.warn_above_c:
                    print(f"[SensorSampler][WARN] CPU pkg temp {temp}C > {self.warn_above_c}C", file=sys.stderr, flush=True)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "SensorSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _combo_tag(combo: dict) -> str:
    return f"{combo['family']}__{combo['dim']}__R{combo['domain_ang']}__d{combo['density']}"


def _configs_for(geometry: sampler.Geometry, combo: dict, seed: int, n: int) -> list[tuple[str, sampler.Configuration]]:
    k = len(geometry.positions_ang)
    configs = s3.generate_design(geometry, combo["family"], combo["dim"], k, combo["domain_ang"], n, seed)
    tag = _combo_tag(combo)
    return [(f"g6x6c__{tag}__seed{seed}__{i:03d}", c) for i, c in enumerate(configs)]


def prepare_and_label_all(combos: list[dict], *, workers: int) -> Path:
    structures_root = OUTPUT_ROOT / "structures"
    canonical_base_fdf = OUTPUT_ROOT / "_canonical_base.fdf"
    canonical_ang_base_fdf(sampler.GRAPHENE_6X6_FDF, canonical_base_fdf)
    geometry = sampler.load_graphene_6x6()
    base_structure = extract_fdf_structure(canonical_base_fdf)
    base_positions = [tuple(p) for p in base_structure.positions_ang]

    all_entries: list[tuple[str, sampler.Configuration]] = []
    for combo in combos:
        all_entries += _configs_for(geometry, combo, TRAIN_SEED, combo["density"])
        all_entries += _configs_for(geometry, combo, TEST_SEED, N_TEST)

    for sample_id, config in all_entries:
        displaced = [
            tuple(base_positions[i][axis] + config.displacements_ang.get(i, (0.0, 0.0, 0.0))[axis] for axis in range(3))
            for i in range(len(base_positions))
        ]
        out_dir = structures_root / sample_id
        materialize_sample_fdf(canonical_base_fdf, out_dir / "RUN.fdf", positions_ang=displaced, single_point=True)

    print(json.dumps({"n_structures_total": len(all_entries)}, sort_keys=True), flush=True)
    try:
        run_manifest = run_derivative_siesta_references(
            stencil_root=OUTPUT_ROOT,
            source_dataset_root=REPO_ROOT / "materials/graphene",
            siesta_command=pilot.DEFAULT_SIESTA_COMMAND,
            workers=workers,
            require_positive_provenance_for_reuse=False,
            diagnostic_only=True,
        )
    except DerivativeSiestaReferenceError as exc:
        raise RuntimeError(f"SIESTA labeling failed: {exc}") from exc

    ids = {sid for sid, _ in all_entries}
    failed = [r["sample_id"] for r in run_manifest["rows"] if r["sample_id"] in ids and r["status"] not in {"ok", "staged", "skipped_existing"}]
    if failed:
        raise RuntimeError(f"{len(failed)} 6x6 campaign geometries failed SIESTA labeling: {failed[:10]}")
    return Path(run_manifest["output_reference_root"])


def _sample_for(sample_id: str, reference_root: Path, combo: dict) -> s4.Sample:
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
        sample_id=sample_id, family=combo["family"], dim=combo["dim"], k=72, amplitude_ang=combo["domain_ang"],
        run_fdf=structure_dir / "RUN.fdf", reference_dir=reference_dir, reference_matrix=selection_path,
    )


def _looks_like_oom(run_dir: Path) -> bool:
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "out of memory" in text or "outofmemoryerror" in text or "cuda error" in text


def train_one(combo: dict, reference_root: Path, geometry: sampler.Geometry, accelerator: str) -> dict:
    tag = _combo_tag(combo)
    train_samples = [_sample_for(sid, reference_root, combo) for sid, _ in _configs_for(geometry, combo, TRAIN_SEED, combo["density"])]
    test_samples = [_sample_for(sid, reference_root, combo) for sid, _ in _configs_for(geometry, combo, TEST_SEED, N_TEST)]

    run_name = f"6x6c__{tag}"
    run_dir = OUTPUT_ROOT / "runs" / run_name
    config_path = s4.build_graph2mat_config(
        run_dir, train_samples, test_samples, run_name=run_name,
        max_epochs=CROSS_TESTING_MAX_EPOCHS, accelerator=accelerator, training_seed=0,
        model_overrides=CROSS_TESTING_MODEL, batch_size=CROSS_TESTING_BATCH_SIZE,
        early_stopping_patience=CROSS_TESTING_EARLY_STOPPING_PATIENCE,
        precision=CROSS_TESTING_PRECISION, loader_threads=CROSS_TESTING_LOADER_THREADS,
    )
    start = time.perf_counter()
    checkpoint = s4.run_graph2mat_training(config_path, run_dir)
    train_seconds = time.perf_counter() - start

    val_manifest_path = run_dir / "test_manifest.csv"
    s4.write_val_manifest(val_manifest_path, test_samples)
    predicted_root = s4.run_prediction(checkpoint, val_manifest_path, run_dir / "test_predictions", accelerator)
    metrics = s4.compute_validation_metrics(test_samples, predicted_root)
    return {
        "rank": combo["rank"], "combo_tag": tag, "family": combo["family"], "dim": combo["dim"],
        "domain_ang": combo["domain_ang"], "density": combo["density"],
        "w90_H_MAE_meV": combo["w90_H_MAE_meV"],
        "H_MAE_meV": metrics["H_MAE"] * 1000 if metrics["H_MAE"] == metrics["H_MAE"] else None,
        "metrics_eV": metrics, "train_seconds": train_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5, help="SIESTA labeling workers (6 previously hit 86C on this box).")
    parser.add_argument("--parallel", type=int, default=2, help="Concurrent GPU training jobs (measured: 2x14.4GB fits in 32GB, 3 would not).")
    parser.add_argument("--limit", type=int, default=None, help="Debug: only the first N combos.")
    args = parser.parse_args()

    combos = json.loads(COMBOS_PATH.read_text(encoding="utf-8"))["combos"]
    if args.limit:
        combos = combos[: args.limit]
    total = len(combos)
    selected_tags = {_combo_tag(combo) for combo in combos}

    results: list[dict] = []
    if SUMMARY_PATH.exists():
        try:
            results = [
                row for row in json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
                if "error" not in row and row.get("combo_tag") in selected_tags
            ]
        except (json.JSONDecodeError, OSError):
            results = []
    completed_tags = {row["combo_tag"] for row in results}
    pending_combos = [combo for combo in combos if _combo_tag(combo) not in completed_tags]

    backend_info = s4.torch_backend_preflight()
    accelerator = backend_info["effective_backend"]
    print(json.dumps({"gpu_preflight": backend_info, "n_combos": total, "n_completed": len(results)}, sort_keys=True), flush=True)

    if not pending_combos:
        print(json.dumps({"finished": True, "n_ok": len(results), "n_total": total, "resumed": True}, sort_keys=True))
        return 0

    with SensorSampler() as cpu_temp:
        t0 = time.perf_counter()
        reference_root = prepare_and_label_all(combos, workers=args.workers)
        siesta_seconds = time.perf_counter() - t0
    print(json.dumps({"phase": "siesta_done", "seconds": siesta_seconds, "peak_cpu_pkg_temp_c": cpu_temp.peak_c}, sort_keys=True), flush=True)

    geometry = sampler.load_graphene_6x6()

    def _run_batch(batch: list[dict], parallel: int, already: list[dict]) -> list[dict]:
        # Writes SUMMARY_PATH after EVERY completion (not just at batch end):
        # this is both the live-progress signal the watchdog reads and the
        # resume checkpoint on next startup -- if this only wrote once at the
        # end, a crash mid-batch would silently lose the resume benefit for
        # every design finished since the last full-batch completion.
        batch_results: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="6x6-campaign") as executor:
            futures = {executor.submit(train_one, combo, reference_root, geometry, accelerator): combo for combo in batch}
            for future in concurrent.futures.as_completed(futures):
                combo = futures[future]
                run_dir = OUTPUT_ROOT / "runs" / f"6x6c__{_combo_tag(combo)}"
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - record, keep going
                    oom = _looks_like_oom(run_dir)
                    result = {
                        "rank": combo["rank"], "combo_tag": _combo_tag(combo),
                        "error": f"{type(exc).__name__}: {exc}", "oom_suspected": oom,
                    }
                batch_results.append(result)
                print(json.dumps({"done": len(already) + len(batch_results), "of": total, "parallel": parallel,
                                   **{k: result.get(k) for k in ("combo_tag", "H_MAE_meV", "w90_H_MAE_meV", "error", "oom_suspected") if k in result}},
                                  sort_keys=True), flush=True)
                SUMMARY_PATH.write_text(
                    json.dumps(sorted(already + batch_results, key=lambda r: r["rank"]), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        return batch_results

    results.extend(_run_batch(pending_combos, args.parallel, results))

    # Fallback: any failure whose train.log looks like a CUDA OOM gets one
    # retry pass at parallel=1 (never at the same concurrency that produced
    # it) -- VRAM usage is uniform across these 80 designs (same architecture/
    # envelope), so an OOM at args.parallel is a concurrency problem, not a
    # one-off, and args.parallel-1>=1 concurrent jobs is the one setting we
    # haven't already tried for the failed ones.
    oom_tags = {r["combo_tag"] for r in results if r.get("oom_suspected")}
    if oom_tags and args.parallel > 1:
        retry_combos = [c for c in pending_combos if _combo_tag(c) in oom_tags]
        print(json.dumps({"retrying_at_parallel_1": [c["rank"] for c in retry_combos]}, sort_keys=True), flush=True)
        results = [r for r in results if r.get("combo_tag") not in oom_tags]
        results.extend(_run_batch(retry_combos, 1, results))

    n_ok = sum(1 for r in results if "error" not in r)
    print(json.dumps({"finished": True, "n_ok": n_ok, "n_total": len(results)}, sort_keys=True))
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
