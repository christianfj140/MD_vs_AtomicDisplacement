#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9 -- k=2 (local_pair_modes) 0.12 Ang ablation study.

The expert's DECISIVE experiment (task event log, request EXP-ood012-k2-tradeoff,
"Net recommendation" section): train matched Graph2Mat models that differ only in
how much 0.08-0.12 Ang boundary data they see, then compare per-mode, per-amplitude
error curves on an independent held-out grid -- not one aggregate MAE. If denser
boundary data monotonically improves 0.10-0.12 Ang error without degrading r<=0.08,
that is evidence FOR promoting 0.12 Ang k=2 (local_pair_modes) from "trained-on" to
"trusted." If it does not, or degrades the core domain, that is evidence for keeping
0.12 Ang k=2 flagged provisional (OOD_CONDITIONAL_K2_PROVISIONAL in S9-S2).

Three matched models, identical architecture/hyperparameters (reuses
dataset_design_w90_001_s4_train_learning_curves.build_graph2mat_config /
run_graph2mat_training verbatim -- same MODEL_OVERRIDES, same fairness requirement
as every other S9 comparison):

  M_BASE   -- amplitudes {0.01, 0.03, 0.05, 0.08}                (never sees > 0.08)
  M_SPARSE -- M_BASE + {0.10, 0.12}                               (2 boundary points)
  M_DENSE  -- M_BASE + {0.09, 0.10, 0.11, 0.12}                   (4 boundary points)

Each amplitude is realized for all three local_pair_modes bond-frame modes
(longitudinal, transverse_ip, transverse_opp -- see w90_displacement_sampler_family.
generate_local_pair_modes), so a "model" here trains on 3x its amplitude count.

HELD-OUT TEST SET: local_pair_modes configurations at amplitudes {0.02, 0.06, 0.075,
0.095, 0.115} Ang. These are deliberately offset from every training amplitude above
(0.01/0.03/0.05/0.08/0.09/0.10/0.11/0.12) -- local_pair_modes' bond-frame directions
are fixed by geometry, not randomized per seed, so testing at an amplitude that any
model also trained on would be literal train/test leakage (the identical geometry),
not a fair held-out point. 0.075/0.095/0.115 stand in for "near the 0.08/0.10/0.12
boundary" without colliding with it.

This script, by default, ONLY does the cheap/safe part: generates the training and
held-out geometries and labels them via SIESTA (a few dozen single-point
calculations, seconds each -- same infrastructure as
dataset_design_w90_001_s9_s3_siesta_labeling.py). It prints the three
``dataset_design_w90_001_s4_train_learning_curves``-style training commands it WOULD
run and exits. Pass --train to actually run the three Graph2Mat trainings and the
comparison (real GPU time, unattended-unsafe until a human has reviewed this script).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
import dataset_design_w90_001_s9_s3_design_generation as s3  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_ood012_k2_ablation"
MODES: tuple[str, ...] = ("longitudinal", "transverse_ip", "transverse_opp")

BASE_AMPLITUDES_ANG: tuple[float, ...] = (0.01, 0.03, 0.05, 0.08)
SPARSE_EXTRA_ANG: tuple[float, ...] = (0.10, 0.12)
DENSE_EXTRA_ANG: tuple[float, ...] = (0.09, 0.10, 0.11, 0.12)
TEST_AMPLITUDES_ANG: tuple[float, ...] = (0.02, 0.06, 0.075, 0.095, 0.115)

MODEL_TRAIN_AMPLITUDES: dict[str, tuple[float, ...]] = {
    "M_BASE": BASE_AMPLITUDES_ANG,
    "M_SPARSE": tuple(sorted(BASE_AMPLITUDES_ANG + SPARSE_EXTRA_ANG)),
    "M_DENSE": tuple(sorted(BASE_AMPLITUDES_ANG + DENSE_EXTRA_ANG)),
}
TRAINING_SEED = 7  # disjoint from S9-S4's (0,1) and S9-S5's (2..6)


def _amp_tag(amplitude_ang: float) -> str:
    return f"{amplitude_ang:g}".replace(".", "d")


def _all_amplitudes() -> list[float]:
    seen: list[float] = []
    for amps in (*MODEL_TRAIN_AMPLITUDES.values(), TEST_AMPLITUDES_ANG):
        for a in amps:
            if a not in seen:
                seen.append(a)
    return sorted(seen)


def _configs_for_amplitude(geometry: sampler.Geometry, amplitude: float) -> list[tuple[str, str, sampler.Configuration]]:
    """(sample_id, mode, Configuration) for every mode at one amplitude."""
    group = sampler.generate_local_pair_modes(geometry, dim="3D", amplitudes_ang=(amplitude,))
    out = []
    for config in group:
        mode = config.metadata["mode"]
        if mode not in MODES:
            continue
        out.append((f"ood012_k2ab__{mode}__r{_amp_tag(amplitude)}", mode, config))
    return out


def _materialize(entries: list[tuple[str, sampler.Configuration]], *, structures_root: Path, canonical_base_fdf: Path) -> None:
    base_structure = extract_fdf_structure(canonical_base_fdf)
    base_positions = [tuple(p) for p in base_structure.positions_ang]
    for sample_id, config in entries:
        displaced = [
            tuple(base_positions[i][axis] + config.displacements_ang.get(i, (0.0, 0.0, 0.0))[axis] for axis in range(3))
            for i in range(len(base_positions))
        ]
        out_dir = structures_root / sample_id
        materialize_sample_fdf(canonical_base_fdf, out_dir / "RUN.fdf", positions_ang=displaced, single_point=True)
        metadata = dict(config.metadata)
        metadata["sample_id"] = sample_id
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_geometries(workers: int = 12) -> Path:
    """Generate + label (SIESTA) every geometry every model or the test set needs.

    Cheap: len(_all_amplitudes()) x 3 modes single-point calculations, same
    engine as the other S9 labeling scripts (idempotent, skip-if-exists).
    """
    structures_root = OUTPUT_ROOT / "structures"
    canonical_base_fdf = OUTPUT_ROOT / "_canonical_base.fdf"
    canonical_ang_base_fdf(pilot.DEFAULT_MATERIAL_FDF, canonical_base_fdf)
    geometry = sampler.load_graphene_primitive()

    entries: list[tuple[str, sampler.Configuration]] = []
    id_to_mode_amp: dict[str, tuple[str, float]] = {}
    for amplitude in _all_amplitudes():
        for sample_id, mode, config in _configs_for_amplitude(geometry, amplitude):
            entries.append((sample_id, config))
            id_to_mode_amp[sample_id] = (mode, amplitude)

    _materialize(entries, structures_root=structures_root, canonical_base_fdf=canonical_base_fdf)
    (OUTPUT_ROOT / "sample_index.json").write_text(
        json.dumps({sid: {"mode": m, "amplitude_ang": a} for sid, (m, a) in id_to_mode_amp.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

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

    failed = [r["sample_id"] for r in run_manifest["rows"] if r["sample_id"] in id_to_mode_amp and r["status"] not in {"ok", "staged", "skipped_existing"}]
    if failed:
        raise RuntimeError(f"{len(failed)} ablation geometries failed SIESTA labeling: {failed[:10]}")

    return Path(run_manifest["output_reference_root"])


def _sample_for(sample_id: str, mode: str, amplitude: float, reference_root: Path) -> s4.Sample:
    structure_dir = OUTPUT_ROOT / "structures" / sample_id
    reference_dir = reference_root / sample_id
    selection_path = None
    for candidate in ("graphene.TSHS", "graphene.HSX"):
        if (reference_dir / candidate).exists():
            selection_path = reference_dir / candidate
            break
    if selection_path is None:
        matches = list(reference_dir.glob("*.TSHS")) + list(reference_dir.glob("*.HSX"))
        selection_path = matches[0] if matches else reference_dir / "MISSING"
    return s4.Sample(
        sample_id=sample_id, family="local_pair_modes", dim="3D", k=2, amplitude_ang=amplitude,
        run_fdf=structure_dir / "RUN.fdf", reference_dir=reference_dir, reference_matrix=selection_path,
    )


def build_pools(reference_root: Path) -> dict[str, list[s4.Sample]]:
    """Training pool per model + the held-out test pool, as real s4.Sample objects."""
    geometry = sampler.load_graphene_primitive()
    pools: dict[str, list[s4.Sample]] = {name: [] for name in MODEL_TRAIN_AMPLITUDES}
    test_pool: list[s4.Sample] = []
    for amplitude in _all_amplitudes():
        for sample_id, mode, _config in _configs_for_amplitude(geometry, amplitude):
            sample = _sample_for(sample_id, mode, amplitude, reference_root)
            for model_name, amps in MODEL_TRAIN_AMPLITUDES.items():
                if amplitude in amps:
                    pools[model_name].append(sample)
            if amplitude in TEST_AMPLITUDES_ANG:
                test_pool.append(sample)
    pools["TEST"] = test_pool
    return pools


def per_sample_errors(val_samples: list[s4.Sample], predicted_root: Path) -> list[dict[str, Any]]:
    rows = []
    for sample in val_samples:
        predicted_path = predicted_root / sample.sample_id / "ML_prediction.HSX"
        if not predicted_path.exists():
            continue
        h_pred = s4.gamma_hk(predicted_path)
        h_ref = s4.gamma_hk(sample.reference_matrix)
        error = h_pred - h_ref
        rows.append({
            "sample_id": sample.sample_id, "mode": sample.sample_id.split("__")[1], "amplitude_ang": sample.amplitude_ang,
            "H_MAE": float(np.mean(np.abs(error))), "H_RMSE": float(np.sqrt(np.mean(np.abs(error) ** 2))),
            "rel_Frob": float(np.linalg.norm(error) / np.linalg.norm(h_ref)),
        })
    return rows


def _train_one_model(model_name: str, pools: dict[str, list[s4.Sample]], accelerator: str, test_manifest_path: Path) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / "runs" / model_name
    config_path = s4.build_graph2mat_config(
        run_dir, pools[model_name], pools["TEST"], run_name=model_name,
        accelerator=accelerator, training_seed=TRAINING_SEED,
    )
    checkpoint = s4.run_graph2mat_training(config_path, run_dir)
    predicted_root = s4.run_prediction(checkpoint, test_manifest_path, run_dir / "test_predictions", accelerator)
    rows = per_sample_errors(pools["TEST"], predicted_root)
    return {"n_train": len(pools[model_name]), "checkpoint": str(checkpoint), "per_sample": rows}


def run_ablation_training(pools: dict[str, list[s4.Sample]]) -> dict[str, Any]:
    """Train the three matched models and evaluate all three on the SAME held-out pool.

    Real GPU time (three from-scratch Graph2Mat trainings). Not called by
    ``main()`` unless --train is passed. Only 3 independent models exist here
    (unlike S9-S4's hundreds of pilot designs), so they run concurrently via
    a 3-worker pool -- same ThreadPoolExecutor-per-independent-job pattern as
    S9-S4's ``--parallel-designs``, just capped at the number of real jobs
    rather than an arbitrary higher number (this repo's tiny Graph2Mat models
    use ~700MB VRAM each, so 3-way concurrency has large GPU headroom).
    """
    import concurrent.futures

    accelerator = s4.torch_backend_preflight()["effective_backend"]
    test_manifest_path = OUTPUT_ROOT / "test_manifest.csv"
    s4.write_val_manifest(test_manifest_path, pools["TEST"])

    results: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODEL_TRAIN_AMPLITUDES), thread_name_prefix="k2ab-train") as executor:
        futures = {
            executor.submit(_train_one_model, model_name, pools, accelerator, test_manifest_path): model_name
            for model_name in MODEL_TRAIN_AMPLITUDES
        }
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--train", action="store_true", help="Actually train the 3 models (real GPU time). Default: prepare geometries only.")
    args = parser.parse_args()

    reference_root = prepare_geometries(workers=args.workers)
    pools = build_pools(reference_root)
    summary = {name: len(samples) for name, samples in pools.items()}
    print(json.dumps({"pool_sizes": summary, "trained": args.train}, sort_keys=True))

    if not args.train:
        print(
            "\nGeometries prepared and labeled. Not training (pass --train to run the "
            "3-model ablation for real -- three from-scratch Graph2Mat trainings, GPU time).",
            file=sys.stderr,
        )
        return 0

    results = run_ablation_training(pools)
    report_path = OUTPUT_ROOT / "ablation_report.json"
    report_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"report": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
