#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S6 -- MD baseline + non-inferiority margin.

Verifies the MD train/frozen split, attempts a Graph2Mat MD baseline at the
same N_train in {16, 32, 64} used by S4/S5, and fixes (or documents the
impossibility of fixing) the non-inferiority margin Delta_NI for H-MAE, using
only pilot/validation results -- never the (not-yet-run) full 5-seed
comparison, and no Pareto frontier is built here.

Cites and reuses, rather than reimplements (S1 section 4):
  - ``docs/dataset_design_w90_001_s1_convention_manifest.md`` (S1).
  - ``Comparison/scripts/dataset_design_w90_001_s4_train_learning_curves.py``
    (S4): ``Sample``, ``select_split``, ``build_graph2mat_config``,
    ``run_graph2mat_training``, ``write_val_manifest``, ``run_prediction``,
    ``compute_validation_metrics``, ``torch_backend_preflight`` -- reused
    verbatim so an MD baseline is trained/evaluated identically to every
    displacement family in S4, and ``learning_curves.csv`` supplies the
    pilot/validation signal for the margin.
  - ``Comparison/scripts/dataset_design_w90_001_s5_frozen_tests_generalization.py``
    (S5): ``gather_md_candidate_frames``, ``build_md_frozen_samples``
    (statistical-inefficiency-gated MD frozen-test selection),
    ``materialize_and_run`` / ``build_common_displacement_configs`` /
    ``build_ood_displacement_configs`` for the other two frozen tests --
    reused verbatim, only invoked lazily if an MD baseline actually trains.

Key finding (S6, this step -- see ``md_split_integrity_report.json``): of the
whole repo, only 20 MD frames anywhere still have a persisted
SIESTA-computed Hamiltonian (``Comparison/results/results_md/MD_dataset*/run_*/
siesta_hamiltonians/<frame_index>/``), and every one of those 20 falls inside
the *test* partition of its dataset's ``splits/split_summary.json``. The
*train*/*validation*-partition Hamiltonians were written under
``Comparison/workspaces/<run_id>/...``, which has since been cleaned up (the
same loss S5's docstring already flagged for its own ``md_frozen`` frame
pool). S5 already claims 7 of those 20 frames (the ones spaced far enough
apart to pass the measured statistical-inefficiency gap ``g``) as the
``md_frozen`` frozen test. That leaves at most 13 frames anywhere in the repo
that could serve as an "MD training pool" distinct from ``md_frozen`` --
short of even N_train=16. Regenerating a real, densely-sampled MD trajectory
is the same "separate, much larger integration project" S4 and S5 already
scoped out; this step does not attempt it. The MD-baseline training/eval code
path below is real and will run correctly the moment that pool exists (no
rewrite needed), but with today's data every (N_train, seed) combo is
recorded as ``insufficient_md_training_data``, never fabricated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402
import dataset_design_w90_001_s5_frozen_tests_generalization as s5  # noqa: E402
import run_w90_displacement_siesta_pilot as pilot  # noqa: E402
from reference_selection import choose_reference_matrix  # noqa: E402
from fdf_materialization import materialize_sample_fdf  # noqa: E402
from run_epc_siesta_reference import canonical_ang_base_fdf  # noqa: E402

DEFAULT_MD_RESULTS_ROOT = s5.DEFAULT_MD_RESULTS_ROOT
DEFAULT_S4_ROOT = s4.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s6"

# The task's own suggested candidate ("e.g. 5% relative H-MAE"); examined
# against pilot evidence below, never assumed to hold.
CANDIDATE_MARGIN_RELATIVE_H_MAE = 0.05

MD_BASELINE_CSV_FIELDNAMES = [
    "n_train", "seed", "frozen_test", "status", "available_n", "val_n",
    "backend", "checkpoint", "H_MAE", "H_RMSE", "rel_Frob", "spectral_err",
    "hermiticity", "n_evaluated",
]


# --------------------------------------------------------------------------
# (a) MD train/frozen split verification + frozen decorrelation check
# --------------------------------------------------------------------------


def load_md_dataset_run_dirs(md_results_root: Path = DEFAULT_MD_RESULTS_ROOT) -> list[Path]:
    run_dirs = []
    for dataset_dir in sorted(md_results_root.glob("MD_dataset*_*")):
        candidates = sorted(dataset_dir.glob("run_*"))
        if candidates:
            run_dirs.append(candidates[-1])
    return run_dirs


def _frame_indices_from_manifest(manifest_path: Path) -> set[int]:
    if not manifest_path.exists():
        return set()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indices: set[int] = set()
    for row in rows:
        raw = row.get("frame_index") or row.get("sample_id", "").rsplit("_", 1)[-1]
        try:
            indices.add(int(raw))
        except (TypeError, ValueError):
            continue
    return indices


def verify_md_split_integrity(md_results_root: Path = DEFAULT_MD_RESULTS_ROOT) -> dict[str, Any]:
    """Per-dataset train/validation/test partition sanity + persisted-Hamiltonian audit."""

    persisted_frames = s5.gather_md_candidate_frames(md_results_root)
    per_dataset: list[dict[str, Any]] = []
    for run_dir in load_md_dataset_run_dirs(md_results_root):
        dataset_name = run_dir.parent.name
        splits_dir = run_dir / "splits"
        summary_path = splits_dir / "split_summary.json"
        if not summary_path.exists():
            per_dataset.append({"dataset": dataset_name, "status": "missing_split_summary"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        train_idx = _frame_indices_from_manifest(splits_dir / "train_manifest.csv")
        val_idx = _frame_indices_from_manifest(splits_dir / "validation_manifest.csv")
        test_idx = _frame_indices_from_manifest(splits_dir / "test_manifest.csv")
        overlap = sorted((train_idx & val_idx) | (train_idx & test_idx) | (val_idx & test_idx))
        local_persisted = {idx for idx, path in persisted_frames.items() if dataset_name in str(path)}
        persisted_outside_test = sorted(local_persisted & (train_idx | val_idx))
        per_dataset.append(
            {
                "dataset": dataset_name,
                "status": "ok",
                "temporal_gap": summary.get("temporal_gap"),
                "strategy": summary.get("strategy"),
                "train_count": len(train_idx),
                "validation_count": len(val_idx),
                "test_count": len(test_idx),
                "overlap_between_partitions": overlap,
                "persisted_frame_indices": sorted(local_persisted),
                "persisted_outside_test_partition": persisted_outside_test,
            }
        )

    evaluated = [d for d in per_dataset if d.get("status") == "ok"]
    no_partition_overlap = all(not d["overlap_between_partitions"] for d in evaluated)
    train_block_hamiltonians_available = any(d["persisted_outside_test_partition"] for d in evaluated)
    return {
        "per_dataset": per_dataset,
        "total_persisted_frames": len(persisted_frames),
        "no_partition_overlap": no_partition_overlap,
        "train_block_hamiltonians_available": train_block_hamiltonians_available,
        "split_verified": bool(evaluated) and no_partition_overlap,
        "finding": (
            "Only the *test*-partition frames of each MD_dataset run still have a persisted "
            "SIESTA-computed Hamiltonian on disk; the train/validation-partition Hamiltonians "
            "were written under Comparison/workspaces/<run_id>/... which has since been "
            "cleaned up. There is currently no on-disk MD training pool distinct from the "
            "frozen-test frame pool."
            if not train_block_hamiltonians_available
            else "Train/validation-partition Hamiltonians are present on disk for at least one dataset."
        ),
    }


def check_md_frozen_decorrelation(output_root: Path) -> tuple[list[s4.Sample], dict[str, Any], bool]:
    """Reuses S5's statistical-inefficiency-gated selection; verifies the selection it made."""

    samples, report = s5.build_md_frozen_samples(output_root=output_root / "md_frozen")
    selected = report.get("selected_frame_indices", [])
    g = report.get("g_used", 0)
    gaps_ok = all(b - a >= g for a, b in zip(selected, selected[1:]))
    decorrelation_verified = bool(selected) and g > 0 and gaps_ok
    report["decorrelation_verified"] = decorrelation_verified
    return samples, report, decorrelation_verified


# --------------------------------------------------------------------------
# (b) MD training pool + baseline training/eval (real code path; will be
# exercised the moment enough MD frames exist -- currently none do)
# --------------------------------------------------------------------------


def md_training_pool_indices(md_results_root: Path, frozen_report: dict[str, Any]) -> list[int]:
    persisted = s5.gather_md_candidate_frames(md_results_root)
    frozen = set(frozen_report.get("selected_frame_indices", []))
    return sorted(idx for idx in persisted if idx not in frozen)


def materialize_md_samples(
    indices: list[int],
    persisted_frames: dict[int, Path],
    *,
    output_root: Path,
    family: str,
) -> list[s4.Sample]:
    import sisl

    structures_root = output_root / "structures"
    canonical_base_fdf = output_root / "_canonical_base.fdf"
    canonical_ang_base_fdf(pilot.DEFAULT_MATERIAL_FDF, canonical_base_fdf)

    samples: list[s4.Sample] = []
    for idx in indices:
        sample_id = f"{family}__{idx:03d}"
        tshs_path = persisted_frames[idx]
        geometry = sisl.get_sile(str(tshs_path)).read_geometry()
        positions_ang = [tuple(row) for row in np.asarray(geometry.xyz, dtype=float)]

        out_dir = structures_root / sample_id
        materialize_sample_fdf(canonical_base_fdf, out_dir / "RUN.fdf", positions_ang=positions_ang, single_point=True)
        dest_matrix = out_dir / tshs_path.name
        if not dest_matrix.exists():
            dest_matrix.write_bytes(tshs_path.read_bytes())

        selection = choose_reference_matrix(out_dir, require_positive_provenance=False)
        if not selection.ok or selection.path is None:
            continue
        samples.append(
            s4.Sample(
                sample_id=sample_id, family=family, dim="md", k=0, amplitude_ang=float("nan"),
                run_fdf=out_dir / "RUN.fdf", reference_dir=out_dir, reference_matrix=selection.path,
            )
        )
    return samples


def run_md_baseline_combo(
    training_samples: list[s4.Sample],
    n_train: int,
    seed: int,
    *,
    output_root: Path,
    get_frozen_samples_by_test: Callable[[], dict[str, list[s4.Sample]]],
) -> list[dict[str, Any]]:
    row: dict[str, Any] = {"n_train": n_train, "seed": seed, "available_n": len(training_samples)}
    split = s4.select_split(training_samples, n_train, seed)
    if split is None:
        row.update(status="insufficient_md_training_data", val_n=0, frozen_test="", backend="", checkpoint="")
        return [row]

    train_samples, val_samples = split
    row["val_n"] = len(val_samples)
    run_name = f"md_baseline__n{n_train}__seed{seed}"
    run_dir = output_root / "runs" / run_name
    accelerator = s4.torch_backend_preflight()["effective_backend"]
    row["backend"] = accelerator

    try:
        config_path = s4.build_graph2mat_config(
            run_dir, train_samples, val_samples, run_name=run_name, accelerator=accelerator
        )
        checkpoint = s4.run_graph2mat_training(config_path, run_dir)
    except Exception as exc:  # noqa: BLE001 - a failed run is a recorded row, not a crash
        row.update(status=f"error: {type(exc).__name__}: {exc}"[:500], frozen_test="", checkpoint="")
        return [row]
    row["checkpoint"] = str(checkpoint)

    rows: list[dict[str, Any]] = []
    for test_name, samples in get_frozen_samples_by_test().items():
        eval_row = dict(row)
        eval_row["frozen_test"] = test_name
        if not samples:
            eval_row.update(status="no_frozen_samples", n_evaluated=0)
            rows.append(eval_row)
            continue
        try:
            eval_dir = run_dir / "eval" / test_name
            manifest_path = eval_dir / "manifest.csv"
            s4.write_val_manifest(manifest_path, samples)
            predicted_root = s4.run_prediction(checkpoint, manifest_path, eval_dir / "prediction", accelerator)
            metrics = s4.compute_validation_metrics(samples, predicted_root)
            eval_row.update(metrics)
            eval_row["status"] = "ok" if metrics["n_evaluated"] > 0 else "prediction_incomplete"
        except Exception as exc:  # noqa: BLE001
            eval_row["status"] = f"error: {type(exc).__name__}: {exc}"[:500]
        rows.append(eval_row)
    return rows


# --------------------------------------------------------------------------
# (c)/(d) Non-inferiority margin from pilot/validation results only
# --------------------------------------------------------------------------


def summarize_pilot_signal(s4_csv: Path = DEFAULT_S4_ROOT / "learning_curves.csv") -> dict[str, Any]:
    if not s4_csv.exists():
        return {"n_models": 0, "rel_Frob_min": None, "rel_Frob_median": None, "H_MAE_min": None, "H_MAE_median": None}
    with s4_csv.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("status") == "ok"]
    rel_frob = [float(r["rel_Frob"]) for r in rows]
    h_mae = [float(r["H_MAE"]) for r in rows]
    return {
        "n_models": len(rows),
        "rel_Frob_min": min(rel_frob) if rel_frob else None,
        "rel_Frob_median": float(np.median(rel_frob)) if rel_frob else None,
        "rel_Frob_max": max(rel_frob) if rel_frob else None,
        "H_MAE_min": min(h_mae) if h_mae else None,
        "H_MAE_median": float(np.median(h_mae)) if h_mae else None,
        "H_MAE_max": max(h_mae) if h_mae else None,
    }


def fix_non_inferiority_margin(pilot_signal: dict[str, Any], *, md_training_available: bool) -> dict[str, Any]:
    reasons: list[str] = []
    if not md_training_available:
        reasons.append(
            "No genuine MD training baseline could be trained (every N_train in {16,32,64} is "
            "insufficient_md_training_data -- see md_baseline_metrics.csv). A non-inferiority "
            "margin for an 'equivalent to MD' claim requires an actual MD baseline to be "
            "non-inferior to."
        )
    best_rel_frob = pilot_signal.get("rel_Frob_min")
    if best_rel_frob is not None and best_rel_frob > CANDIDATE_MARGIN_RELATIVE_H_MAE:
        reasons.append(
            f"The best relative-Frobenius H error achieved by any S4 pilot model "
            f"({best_rel_frob:.1%}) is already {best_rel_frob / CANDIDATE_MARGIN_RELATIVE_H_MAE:.1f}x "
            f"the candidate margin ({CANDIDATE_MARGIN_RELATIVE_H_MAE:.0%} relative H-MAE); fixing "
            "that candidate now would not be grounded in anything the pilot has demonstrated as "
            "achievable -- the S4 pilot architecture/epoch budget is deliberately small/fast, "
            "not production-representative."
        )
    justified = not reasons
    return {
        "candidate_margin_relative_h_mae": CANDIDATE_MARGIN_RELATIVE_H_MAE,
        "delta_ni_fixed": justified,
        "delta_ni_relative_h_mae": CANDIDATE_MARGIN_RELATIVE_H_MAE if justified else None,
        "pilot_signal": pilot_signal,
        "reasons_not_justified": reasons,
        "conclusion": (
            "Delta_NI fixed at the candidate value."
            if justified
            else (
                "No defensible Delta_NI can be fixed from pilot/validation results alone. Any "
                "S6+ conclusion must be limited to observed error/cost and a Pareto front, "
                "without an 'equivalent to MD' non-inferiority claim, until both (i) a real MD "
                "training pool is regenerated and (ii) a production-representative (non-pilot) "
                "model accuracy ceiling exists to ground a practically-relevant margin."
            )
        ),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(csv_path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md-results-root", type=Path, default=DEFAULT_MD_RESULTS_ROOT)
    parser.add_argument("--s4-root", type=Path, default=DEFAULT_S4_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    output_root: Path = args.output_root

    split_report = verify_md_split_integrity(args.md_results_root)
    write_json(output_root / "md_split_integrity_report.json", split_report)

    frozen_samples, frozen_report, decorrelation_verified = check_md_frozen_decorrelation(output_root)
    write_json(output_root / "md_frozen_decorrelation_report.json", frozen_report)

    training_indices = md_training_pool_indices(args.md_results_root, frozen_report)
    persisted_frames = s5.gather_md_candidate_frames(args.md_results_root)
    training_samples = materialize_md_samples(
        training_indices, persisted_frames, output_root=output_root / "md_train_pool", family="md_train_pool"
    )

    frozen_cache: dict[str, list[s4.Sample]] = {}

    def get_frozen_samples_by_test() -> dict[str, list[s4.Sample]]:
        if not frozen_cache:
            geometry = s5.sampler.load_graphene_primitive()
            common_entries = s5.build_common_displacement_configs(geometry)
            ood_entries = s5.build_ood_displacement_configs(geometry)
            common_run = s5.materialize_and_run(
                common_entries, output_root=output_root / "common_displacement",
                siesta_command=s5.DEFAULT_SIESTA_COMMAND, workers=8,
            )
            ood_run = s5.materialize_and_run(
                ood_entries, output_root=output_root / "ood_displacement",
                siesta_command=s5.DEFAULT_SIESTA_COMMAND, workers=8,
            )
            frozen_cache["common_displacement"] = s5.samples_from_run(common_entries, common_run)
            frozen_cache["ood_displacement"] = s5.samples_from_run(ood_entries, ood_run)
            frozen_cache["md_frozen"] = frozen_samples
        return frozen_cache

    baseline_rows: list[dict[str, Any]] = []
    for n_train in s4.NESTED_N_TRAIN:
        for seed in s4.SEEDS:
            baseline_rows.extend(
                run_md_baseline_combo(
                    training_samples, n_train, seed, output_root=output_root,
                    get_frozen_samples_by_test=get_frozen_samples_by_test,
                )
            )
    write_csv(output_root / "md_baseline_metrics.csv", baseline_rows, MD_BASELINE_CSV_FIELDNAMES)

    md_training_available = any(row.get("status") == "ok" for row in baseline_rows)
    pilot_signal = summarize_pilot_signal(args.s4_root / "learning_curves.csv")
    margin_report = fix_non_inferiority_margin(pilot_signal, md_training_available=md_training_available)
    write_json(output_root / "non_inferiority_margin.json", margin_report)

    summary = {
        "task": "DATASET-DESIGN-W90-001-S6",
        "split_verified": split_report["split_verified"],
        "train_block_hamiltonians_available": split_report["train_block_hamiltonians_available"],
        "md_frozen_decorrelation_verified": decorrelation_verified,
        "md_frozen_g_used": frozen_report.get("g_used"),
        "md_frozen_selected_count": len(frozen_samples),
        "md_training_pool_available_n": len(training_samples),
        "md_training_available": md_training_available,
        "delta_ni_fixed": margin_report["delta_ni_fixed"],
        "delta_ni_relative_h_mae": margin_report["delta_ni_relative_h_mae"],
    }
    write_json(output_root / "s6_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
