#!/usr/bin/env python3
"""DATASET-DESIGN-CURVES-V1 -- learning curves, cost-accuracy frontier, designed-vs-MD.

Implements docs/dataset_design_curves_v1_preregistration.md (the plan in
docs/gola_para_claude.md). One subcommand per phase, run in this order:

  inventory      Phase 1: pools N=64 (hash-verified prefixes), checkpoint audit,
                 w90 dev check, MD inventory.
  dev6x6         Phase 2: build + label common_dev_6x6_v1 (48 structures).
  reeval         Phase 2.5: 80+80 existing checkpoints on the common dev sets;
                 clean table of the 160 existing results.
  select         Phase 3: lock 6 curve recipes per system (+ best LHS).
  md-split       Phase 8.1-8.2: trajectory-level MD split for w90.
  train --stage  curves | lhs | confirm | md  (Phases 4, 5, 7, 8).
  finalists      Phase 6 (and Phase 7 N* confirmation/escalation on re-run).
  final-test     Phase 9: freeze models, build + label the final test.
  evaluate-final Phase 10: one prediction pass of every frozen model.
  analyze        Phases 10-12: paired differences, Pareto, figures, table.

Everything is written under OUT and every step is resumable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402
import dataset_design_w90_001_s5_frozen_tests_generalization as s5  # noqa: E402
import dataset_design_w90_001_s9_crosstesting_campaign_6x6 as c6  # noqa: E402

sampler = c6.sampler
RESULTS = REPO_ROOT / "Comparison/results"
OUT = RESULTS / "dataset_design_curves_v1"
CHAIN = RESULTS / "dataset_design_w90_001_s9_chain_logs"
COMBOS_PATH = CHAIN / "crosstesting_campaign_80combos.json"
W90_RUNS = RESULTS / "dataset_design_w90_001_s9_cross_testing_hparam_probe/runs"
W90_DEV_ROOT = RESULTS / "dataset_design_w90_001_s5/common_validation"
G6_ROOT = c6.OUTPUT_ROOT
MD_DATASETS = REPO_ROOT / "Comparison/datasets"

SYSTEMS = ("w90", "6x6")
NS = (4, 8, 16, 32, 64)
CURVE_FAMILIES = ("sobol_sparse", "random_cartesian")
DIMS = ("1D_in", "1D_z", "2D_in", "3D")
DIM_AXES = {"1D_in": {"x"}, "1D_z": {"z"}, "2D_in": {"x", "y"}, "3D": {"x", "y", "z"}}
DIM_TIE_ORDER = ("1D_z", "1D_in", "3D", "2D_in")
STRATA = (0.03, 0.05, 0.08, 0.12)  # upper edges of the real-amplitude strata
SATURATION_TOL = 0.05
MD_TEMPERATURES = (150.0, 300.0, 450.0)
MD_MIN_T_FS = 6  # autocorrelation zero crossing of the interatomic vector (inventory)
MD_TEST_PER_T = {150.0: 5, 300.0: 6, 450.0: 5}
MD_VAL_PER_T = 8
FINAL_AMPLITUDES = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
FINAL_SEEDS = {"w90": 20260919, "6x6": 20260920}
BOOTSTRAP_B, BOOTSTRAP_SEED = 10_000, 12345
PARALLEL = {"w90": 4, "6x6": 2}

TRAIN_KW: dict[str, Any] = {
    "max_epochs": c6.CROSS_TESTING_MAX_EPOCHS, "model_overrides": c6.CROSS_TESTING_MODEL,
    "batch_size": c6.CROSS_TESTING_BATCH_SIZE, "early_stopping_patience": c6.CROSS_TESTING_EARLY_STOPPING_PATIENCE,
    "precision": c6.CROSS_TESTING_PRECISION, "loader_threads": c6.CROSS_TESTING_LOADER_THREADS,
}


# --------------------------------------------------------------------------
# Small I/O helpers
# --------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def sample_to_dict(sample: s4.Sample, **extra: Any) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id, "family": sample.family, "dim": sample.dim, "k": sample.k,
        "amplitude_ang": sample.amplitude_ang, "run_fdf": str(sample.run_fdf),
        "reference_dir": str(sample.reference_dir), "reference_matrix": str(sample.reference_matrix), **extra,
    }


def sample_from_dict(row: dict[str, Any]) -> s4.Sample:
    return s4.Sample(
        sample_id=row["sample_id"], family=row["family"], dim=row["dim"], k=int(row["k"]),
        amplitude_ang=float(row["amplitude_ang"]), run_fdf=Path(row["run_fdf"]),
        reference_dir=Path(row["reference_dir"]), reference_matrix=Path(row["reference_matrix"]),
        max_displacement_ang=row.get("real_amplitude_ang"),
    )


# --------------------------------------------------------------------------
# Geometry: hash, real amplitude, SIESTA cost
# --------------------------------------------------------------------------


_BASE_CACHE: dict[str, np.ndarray] = {}


def base_positions(system: str) -> np.ndarray:
    if system not in _BASE_CACHE:
        fdf = W90_DEV_ROOT / "_canonical_base.fdf" if system == "w90" else G6_ROOT / "_canonical_base.fdf"
        _BASE_CACHE[system] = np.asarray(s5._positions_from_run_fdf(fdf))
    return _BASE_CACHE[system]


def geometry_record(system: str, run_fdf: Path) -> dict[str, Any]:
    """Geometry hash (S5 convention) + real amplitude = max_a |r_a - r_a^base| (Ang)."""

    positions = np.asarray(s5._positions_from_run_fdf(run_fdf))
    amplitude = float(np.linalg.norm(positions - base_positions(system), axis=1).max())
    return {"hash": s5.geometry_hash(positions), "real_amplitude_ang": amplitude}


_SIESTA_TIMER = re.compile(r"^siesta\s+1\s+[\d.]+\s+([\d.]+)", re.MULTILINE)


def siesta_seconds(run_out: Path) -> float | None:
    """Total ``siesta`` timer (s). SIESTA here runs serial (1 MPI rank): CPU s == wall s."""

    try:
        match = _SIESTA_TIMER.search(run_out.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return None
    return float(match.group(1)) if match else None


def stratum_of(amplitude: float) -> int | None:
    for index, edge in enumerate(STRATA):
        if amplitude <= edge + 1e-12:
            return index
    return None


# --------------------------------------------------------------------------
# Metrics (per structure; identical formulas to s4.compute_validation_metrics)
# --------------------------------------------------------------------------


def structure_metrics(sample: s4.Sample, predicted_root: Path, *, window_ev: float = 2.0) -> dict[str, Any] | None:
    predicted = predicted_root / sample.sample_id / "ML_prediction.HSX"
    if not predicted.exists():
        return None
    h_pred = s4.gamma_hk(predicted)
    h_ref = s4.gamma_hk(sample.reference_matrix)
    error = h_pred - h_ref
    ref_norm = float(np.linalg.norm(h_ref))
    eig_pred = s4.gamma_eigenvalues(predicted, sample.reference_matrix)
    eig_ref = s4.gamma_eigenvalues(sample.reference_matrix)
    mask = np.abs(eig_ref - s4.fermi_level_ev(sample.reference_matrix)) <= window_ev
    return {
        "sample_id": sample.sample_id,
        "H_MAE_meV": 1e3 * float(np.mean(np.abs(error))),
        "H_RMSE_meV": 1e3 * float(np.sqrt(np.mean(np.abs(error) ** 2))),
        "rel_Frob": float(np.linalg.norm(error) / ref_norm) if ref_norm else float("nan"),
        "hermiticity_eV": float(np.max(np.abs(h_pred - h_pred.conj().T))),
        "spectral_err_meV": 1e3 * float(np.sqrt(np.mean((eig_pred[mask] - eig_ref[mask]) ** 2))) if mask.any() else float("nan"),
    }


def predict(checkpoint: Path, samples: list[s4.Sample], out_dir: Path, accelerator: str) -> Path:
    """s4.run_prediction; retried on CPU when the shared GPU is out of memory."""

    manifest = out_dir / "manifest.csv"
    s4.write_val_manifest(manifest, samples)
    try:
        return s4.run_prediction(checkpoint, manifest, out_dir, accelerator)
    except RuntimeError:
        if accelerator == "cpu":
            raise
        shutil.rmtree(out_dir / "predicted_hamiltonians", ignore_errors=True)
        return s4.run_prediction(checkpoint, manifest, out_dir, "cpu")


def evaluate(checkpoint: Path, samples: list[s4.Sample], out_dir: Path, accelerator: str) -> list[dict[str, Any]]:
    predicted_root = predict(checkpoint, samples, out_dir, accelerator)
    rows = [row for row in (structure_metrics(sample, predicted_root) for sample in samples) if row]
    if len(rows) != len(samples):
        raise RuntimeError(f"only {len(rows)}/{len(samples)} predictions under {predicted_root}")
    write_csv(out_dir / "per_structure_metrics.csv", rows)
    # Predicted HSX files are large and fully reproducible from the checkpoint.
    shutil.rmtree(predicted_root, ignore_errors=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("H_MAE_meV", "H_RMSE_meV", "rel_Frob", "hermiticity_eV", "spectral_err_meV")
    return {key: float(np.nanmean([float(row[key]) for row in rows])) for key in keys} | {"n_structures": len(rows)}


# --------------------------------------------------------------------------
# TensorBoard: best / final epoch
# --------------------------------------------------------------------------


def tensorboard_epochs(run_dir: Path, checkpoint: Path | None = None) -> dict[str, Any]:
    """Best-val_loss epoch and final epoch of the TB log that wrote ``checkpoint`` (else newest)."""

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    version_dir = None
    if checkpoint is not None and checkpoint.parent.name == "checkpoints":
        version_dir = checkpoint.parent.parent
    events = sorted((version_dir or run_dir).rglob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    if not events:
        return {}
    accumulator = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    if "val_loss" not in accumulator.Tags()["scalars"]:
        return {}
    epoch_at = {event.step: int(event.value) for event in accumulator.Scalars("epoch")}
    val = accumulator.Scalars("val_loss")
    best = min(val, key=lambda event: event.value)
    return {
        "best_epoch": epoch_at.get(best.step), "best_val_loss": float(best.value),
        "final_epoch": epoch_at.get(val[-1].step), "final_val_loss": float(val[-1].value),
        "n_val_epochs": len(val),
    }


def checkpoint_epoch(checkpoint: Path) -> int | None:
    match = re.search(r"epoch=?(\d+)", checkpoint.name)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------
# Existing campaigns (the 80 + 80 runs)
# --------------------------------------------------------------------------


def load_combos() -> list[dict[str, Any]]:
    return read_json(COMBOS_PATH)["combos"]


def parse_design_id(design_id: str) -> dict[str, Any]:
    family, dim, k, radius, res, seed = design_id.split("__")
    return {"family": family, "dim": dim, "k": int(k[1:]), "R": float(radius[1:]), "density": int(res[3:]),
            "sampler_seed": int(seed[4:])}


def recipe_id(system: str, combo: dict[str, Any]) -> str:
    return combo["w90_design_id"] if system == "w90" else c6._combo_tag(combo)


def existing_run_dir(system: str, combo: dict[str, Any]) -> Path:
    if system == "w90":
        return W90_RUNS / f"{combo['w90_design_id']}__N{combo['density']}__campaign80"
    return G6_ROOT / "runs" / f"6x6c__{c6._combo_tag(combo)}"


def existing_checkpoint(run_dir: Path) -> Path:
    """The checkpoint the campaign actually predicted with (s4 used the newest by mtime)."""

    return sorted(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)[-1]


def recipe_meta(system: str, combo: dict[str, Any]) -> dict[str, Any]:
    if system == "w90":
        meta = parse_design_id(combo["w90_design_id"])
    else:
        meta = {"family": combo["family"], "dim": combo["dim"], "k": 72, "R": float(combo["domain_ang"]),
                "density": int(combo["density"]), "sampler_seed": c6.TRAIN_SEED}
    return {"system": system, "recipe_id": recipe_id(system, combo), **meta}


def load_w90_matched_pools(combos: list[dict[str, Any]]) -> dict[str, list[s4.Sample]]:
    """Real SIESTA-backed training set of every w90 design, in generation order (S9-S4 matching)."""

    import dataset_design_w90_001_s9_s4_pilot_screening as s94

    geometry = s94.sampler.load_graphene_primitive()
    hash_to_sample = s94.build_s3_hash_index(s94.s4.load_s3_samples())
    trainable, _pruned = s94.select_candidates(s94.load_manifest(), geometry, hash_to_sample)
    by_id = {entry["design_id"]: entry["_matched_samples"] for entry in trainable}
    return {combo["w90_design_id"]: by_id[combo["w90_design_id"]] for combo in combos}


def g6_samples(combo: dict[str, Any], seed: int, n: int) -> list[s4.Sample]:
    tag = c6._combo_tag(combo)
    reference_root = G6_ROOT / "siesta_hamiltonians"
    return [c6._sample_for(f"g6x6c__{tag}__seed{seed}__{i:03d}", reference_root, combo) for i in range(n)]


def w90_dev_samples() -> list[s4.Sample]:
    entries = s5.build_common_validation_configs(sampler.load_graphene_primitive())
    reference_root = W90_DEV_ROOT / "siesta_hamiltonians"
    return s5.samples_from_run(entries, {"output_reference_root": reference_root, "usable_ids": [sid for sid, _ in entries]})


def dev_samples(system: str) -> list[s4.Sample]:
    if system == "w90":
        return w90_dev_samples()
    return [sample_from_dict(row) for row in read_json(OUT / "common_dev_6x6_v1.json")["samples"]]


# --------------------------------------------------------------------------
# Phase 1 -- inventory
# --------------------------------------------------------------------------


def cmd_inventory(args: argparse.Namespace) -> int:
    combos = load_combos()
    w90_pools = load_w90_matched_pools(combos)
    summary: dict[str, Any] = {}

    for system in SYSTEMS:
        pools: dict[str, Any] = {}
        all_training_hashes: set[str] = set()
        for combo in combos:
            meta = recipe_meta(system, combo)
            samples = w90_pools[meta["recipe_id"]] if system == "w90" else g6_samples(combo, c6.TRAIN_SEED, meta["density"])
            rows = []
            for sample in samples:
                rows.append(sample_to_dict(
                    sample, **geometry_record(system, sample.run_fdf),
                    siesta_cpu_s=siesta_seconds(sample.reference_dir / "RUN.out"),
                    label_ok=sample.reference_matrix.exists(),
                ))
            hashes = [row["hash"] for row in rows]
            all_training_hashes |= set(hashes)
            pools[meta["recipe_id"]] = {**meta, "n": len(rows), "n_unique_hashes": len(set(hashes)),
                                        "all_labeled": all(row["label_ok"] for row in rows), "samples": rows}

        # Prefix verification + curve compatibility of the existing N=16 results.
        for rid, pool in pools.items():
            hashes = [row["hash"] for row in pool["samples"]]
            if pool["density"] == 64:
                prefixes = [set(hashes[:n]) for n in NS]
                pool["prefix_chain_verified"] = (
                    len(hashes) == 64 and len(set(hashes)) == 64 and all(a < b for a, b in zip(prefixes, prefixes[1:]))
                )
                pool["pool_hash"] = ids_hash(hashes)
            else:
                parents = [other for other, p in pools.items() if p["density"] == 64 and p["family"] == pool["family"]
                           and [r["hash"] for r in p["samples"][:len(hashes)]] == hashes]
                pool["prefix_of_n64_pool"] = parents[0] if parents else None

        # Checkpoint audit (Phase 1.4).
        audit = []
        for combo in combos:
            run_dir = existing_run_dir(system, combo)
            checkpoint = existing_checkpoint(run_dir)
            audit.append({"system": system, "recipe_id": recipe_id(system, combo), "checkpoint_used": str(checkpoint),
                          "checkpoint_epoch": checkpoint_epoch(checkpoint), **tensorboard_epochs(run_dir, checkpoint)})
        for row in audit:
            row["used_checkpoint_is_best_val"] = row.get("best_epoch") == row["checkpoint_epoch"]
        write_csv(OUT / f"checkpoint_audit_{system}.csv", audit)

        write_json(OUT / f"pools_{system}.json", pools)
        n64 = [p for p in pools.values() if p["density"] == 64]
        summary[system] = {
            "n_recipes": len(pools),
            "n64_pools": len(n64),
            "n64_prefix_chain_verified": sum(bool(p["prefix_chain_verified"]) for p in n64),
            "n64_all_labeled": sum(p["all_labeled"] for p in n64),
            "n16_prefix_compatible": sorted(rid for rid, p in pools.items() if p.get("prefix_of_n64_pool")),
            "n16_incompatible_historical": sorted(rid for rid, p in pools.items()
                                                  if p["density"] == 16 and not p.get("prefix_of_n64_pool")),
            "checkpoints_using_best_val": sum(row["used_checkpoint_is_best_val"] for row in audit),
            "median_epochs_past_best": float(np.median([row["checkpoint_epoch"] - row["best_epoch"]
                                                        for row in audit if row.get("best_epoch") is not None])),
            "median_val_loss_ratio_used_over_best": float(np.median([row["final_val_loss"] / row["best_val_loss"]
                                                                     for row in audit if row.get("best_val_loss")])),
            "n_distinct_training_geometries": len(all_training_hashes),
        }
        summary[system]["_training_hashes"] = sorted(all_training_hashes)

    # w90 common dev (Phase 2.1): identical for all, disjoint, labelled.
    dev = w90_dev_samples()
    dev_ids = sorted(sample.sample_id for sample in dev)
    dev_hashes = {geometry_record("w90", sample.run_fdf)["hash"] for sample in dev}
    used_ids = {tuple(sorted(row["sample_id"] for row in read_csv(existing_run_dir("w90", combo) / "dev_manifest.csv")))
                for combo in combos}
    summary["w90_dev"] = {
        "n": len(dev), "identical_for_all_80_models": used_ids == {tuple(dev_ids)},
        "overlap_with_training_hashes": len(dev_hashes & set(summary["w90"]["_training_hashes"])),
        "all_scf_converged": all((s.reference_dir / "0_NORMAL_EXIT").exists() and s.reference_matrix.exists() for s in dev),
        "sample_ids": dev_ids, "hashes": sorted(dev_hashes),
    }
    summary["md_inventory"] = md_inventory()
    for system in SYSTEMS:
        write_json(OUT / f"training_hashes_{system}.json", summary[system].pop("_training_hashes"))
    write_json(OUT / "phase1_inventory.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "md_inventory"} | {
        "md_inventory": {k: v for k, v in summary["md_inventory"].items() if k != "trajectories"}}, indent=1, default=str))
    return 0


def md_inventory() -> dict[str, Any]:
    """Distinct w90 MD trajectories with their labelled frames.

    Identity is the frame-1 geometry, not the manifest seed: SIESTA's
    MD.InitialTemperature draws the same velocities whatever the block seed,
    so every block at one temperature is a copy (verified frame by frame at
    t = 6, 50, 150, 199). The longest copy of each trajectory is kept.
    """

    trajectories: dict[str, dict[str, Any]] = {}
    n_blocks = 0
    for manifest in sorted(MD_DATASETS.glob("graphene_w90_*/*/md_temperature_blocks_manifest.json")):
        dataset = manifest.parent
        for block in read_json(manifest)["blocks"]:
            steps = dataset / "md_temperature_blocks" / block["block_id"] / "MD_steps"
            if not (steps / "1" / "graphene.TSHS").exists():
                continue
            n_blocks += 1
            n_frames = sum(1 for d in steps.iterdir() if (d / "graphene.TSHS").exists())
            key = f"T{float(block['temperature_K'])}|{geometry_record('w90', steps / '1' / 'RUN.fdf')['hash'][:16]}"
            copy = {"seed": block["seed"], "n_frames": n_frames, "steps_dir": str(steps),
                    "dataset": str(dataset.relative_to(MD_DATASETS))}
            entry = trajectories.setdefault(key, {"key": key, "temperature_K": float(block["temperature_K"]),
                                                  "copies": [], "timestep_fs": 1.0,
                                                  "ensemble": "NVE (verlet, no thermostat)"})
            entry["copies"].append(copy)
    for entry in trajectories.values():
        longest = max(entry["copies"], key=lambda c: c["n_frames"])
        entry |= {"n_frames": longest["n_frames"], "steps_dir": longest["steps_dir"], "dataset": longest["dataset"],
                  "n_copies": len(entry.pop("copies")),
                  "per_step_siesta_s": siesta_seconds(Path(longest["steps_dir"]).parent / "RUN.out") / longest["n_frames"]}
    return {
        "n_md_blocks": n_blocks, "n_distinct_trajectories": len(trajectories),
        "trajectories": sorted(trajectories.values(), key=lambda t: t["temperature_K"]), "6x6_md_frames": 0,
        "note": "block seeds do not change SIESTA initial velocities: all blocks at one T are the same trajectory",
    }


# --------------------------------------------------------------------------
# SIESTA labelling of new structures (dev 6x6, final test)
# --------------------------------------------------------------------------


def label_structures(entries: list[tuple[str, Any]], root: Path, material_fdf: Path, workers: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base_fdf = root / "_canonical_base.fdf"
    c6.canonical_ang_base_fdf(material_fdf, base_fdf)
    c6.pilot.materialize_structures(entries, structures_root=root / "structures", canonical_base_fdf=base_fdf)
    manifest = c6.run_derivative_siesta_references(
        stencil_root=root, source_dataset_root=REPO_ROOT / "materials/graphene",
        siesta_command=c6.pilot.DEFAULT_SIESTA_COMMAND, workers=workers,
        require_positive_provenance_for_reuse=False, diagnostic_only=True,
    )
    wanted = {sample_id for sample_id, _ in entries}
    failed = [row["sample_id"] for row in manifest["rows"]
              if row["sample_id"] in wanted and row["status"] not in {"ok", "staged", "skipped_existing"}]
    if failed:
        raise RuntimeError(f"SIESTA failed for {failed}")
    return Path(manifest["output_reference_root"])


def labelled_sample(system: str, sample_id: str, reference_root: Path, meta: dict[str, Any]) -> dict[str, Any]:
    reference_dir = reference_root / sample_id
    matrix = reference_dir / f"{'graphene' if system == 'w90' else 'graphene_6x6'}.TSHS"
    if not matrix.exists():
        raise RuntimeError(f"missing SIESTA reference {matrix}")
    sample = s4.Sample(sample_id=sample_id, family=meta["family"], dim=meta["dim"], k=meta["k"],
                       amplitude_ang=meta["amplitude_ang"], run_fdf=reference_dir / "RUN.fdf",
                       reference_dir=reference_dir, reference_matrix=matrix)
    return sample_to_dict(sample, **geometry_record(system, sample.run_fdf),
                          siesta_cpu_s=siesta_seconds(reference_dir / "RUN.out"), **{
                              k: v for k, v in meta.items() if k not in ("family", "dim", "k", "amplitude_ang")})


# --------------------------------------------------------------------------
# Phase 2 -- common_dev_6x6_v1
# --------------------------------------------------------------------------


def cmd_dev6x6(args: argparse.Namespace) -> int:
    training_hashes = set(read_json(OUT / "training_hashes_6x6.json"))
    combos = load_combos()
    # Candidates: every labelled seed-999 (per-recipe test) structure, dedup by hash.
    by_hash: dict[str, dict[str, Any]] = {}
    for combo in sorted(combos, key=lambda c: -c["density"]):  # prefer the d64 copy of duplicated geometries
        for index, sample in enumerate(g6_samples(combo, c6.TEST_SEED, c6.N_TEST)):
            record = geometry_record("6x6", sample.run_fdf)
            if record["hash"] in training_hashes or record["hash"] in by_hash or not sample.reference_matrix.exists():
                continue
            by_hash[record["hash"]] = sample_to_dict(sample, **record, index=index,
                                                     siesta_cpu_s=siesta_seconds(sample.reference_dir / "RUN.out"))

    chosen: list[dict[str, Any]] = []
    missing: list[tuple[str, str, int]] = []
    for dim in DIMS:
        for stratum in range(len(STRATA)):
            for family in ("sobol_sparse", "random_cartesian", "latin_hypercube"):
                candidates = sorted(
                    (row for row in by_hash.values() if row["dim"] == dim and row["family"] == family
                     and stratum_of(row["real_amplitude_ang"]) == stratum),
                    key=lambda row: (row["index"], row["sample_id"]),
                )
                if candidates:
                    chosen.append(candidates[0] | {"stratum": stratum, "origin": "existing_seed999_test"})
                else:
                    missing.append((dim, family, stratum))

    new_rows: list[dict[str, Any]] = []
    if missing:
        geometry = sampler.load_graphene_6x6()
        entries = []
        for dim, family, stratum in missing:
            # Same rule as the existing candidates: lowest index whose real amplitude is in the
            # stratum (LHS draws a per-structure amplitude, so index 0 is not always in it).
            configs = c6.s3.generate_design(geometry, family, dim, 72, STRATA[stratum], c6.N_TEST, c6.TEST_SEED)
            index = next(i for i, config in enumerate(configs)
                         if stratum_of(max(np.linalg.norm(v) for v in config.displacements_ang.values())) == stratum)
            entries.append((f"cdev6x6__{family}__{dim}__R{STRATA[stratum]}__seed{c6.TEST_SEED}__{index:03d}", configs[index]))
        reference_root = label_structures(entries, OUT / "common_dev_6x6_v1_new", sampler.GRAPHENE_6X6_FDF, args.workers)
        for (sample_id, config), (dim, family, stratum) in zip(entries, missing):
            row = labelled_sample("6x6", sample_id, reference_root, {
                "family": family, "dim": dim, "k": 72, "amplitude_ang": STRATA[stratum]})
            if stratum_of(row["real_amplitude_ang"]) != stratum:
                raise RuntimeError(f"{sample_id}: real amplitude {row['real_amplitude_ang']} not in stratum {stratum}")
            new_rows.append(row | {"stratum": stratum, "origin": "new_siesta"})

    samples = sorted(chosen + new_rows, key=lambda r: (DIMS.index(r["dim"]), r["stratum"], r["family"]))
    assert len(samples) == 48 and len({r["hash"] for r in samples}) == 48
    assert not {r["hash"] for r in samples} & training_hashes
    write_json(OUT / "common_dev_6x6_v1.json", {
        "name": "common_dev_6x6_v1", "n": len(samples), "n_new_siesta": len(new_rows),
        "strata_upper_edges_ang": STRATA, "selection_rule": "one per family per (dim, real-amplitude stratum); lowest test index",
        "dev_hash": ids_hash(sorted(r["hash"] for r in samples)), "samples": samples,
    })
    print(json.dumps({"n": len(samples), "n_new_siesta": len(new_rows), "missing_strata_filled": missing}, default=str))
    return 0


# --------------------------------------------------------------------------
# Phase 2.5 -- re-evaluate the 160 existing checkpoints on the common dev sets
# --------------------------------------------------------------------------


def cmd_reeval(args: argparse.Namespace) -> int:
    combos = load_combos()
    accelerator = s4.torch_backend_preflight()["effective_backend"]
    w90_reports = {row["design_id"]: row["report"] for row in read_json(CHAIN / "w90_campaign_summary.json")}
    g6_reports = {row["combo_tag"]: row for row in read_json(CHAIN / "6x6_campaign_summary.json")}
    table: list[dict[str, Any]] = []
    for system in SYSTEMS:
        dev = dev_samples(system)
        pools = read_json(OUT / f"pools_{system}.json")
        audit = {row["recipe_id"]: row for row in read_csv(OUT / f"checkpoint_audit_{system}.csv")}
        per_structure: list[dict[str, Any]] = []

        def _one(combo: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            rid = recipe_id(system, combo)
            out_dir = OUT / "reeval" / system / rid
            cache = out_dir / "per_structure_metrics.csv"
            if cache.exists():
                return combo, read_csv(cache)
            if system == "w90":
                # Already predicted on this exact dev set by the campaign: re-score, no inference.
                predicted_root = existing_run_dir(system, combo) / "dev_predictions/predicted_hamiltonians"
                rows = [structure_metrics(sample, predicted_root) for sample in dev]
                if None in rows:
                    raise RuntimeError(f"missing dev predictions under {predicted_root}")
                write_csv(cache, rows)
                return combo, rows
            return combo, evaluate(existing_checkpoint(existing_run_dir(system, combo)), dev, out_dir, accelerator)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for combo, rows in pool.map(_one, combos):
                rid = recipe_id(system, combo)
                meta = recipe_meta(system, combo)
                summ = summarize(rows)
                per_structure += [{"recipe_id": rid, **row} for row in rows]
                if system == "w90":
                    reported, reported_on = w90_reports[rid]["H_MAE_meV"], "w90 common_validation (24, also early-stopping val)"
                else:
                    reported, reported_on = g6_reports[rid]["H_MAE_meV"], "own seed999 test (8, also early-stopping val)"
                table.append({
                    **{k: meta[k] for k in ("system", "recipe_id", "family", "dim", "k")},
                    "R_train": meta["R"], "N_train": meta["density"], "sampler_seed": meta["sampler_seed"],
                    "training_seed": 0,
                    "pilot_H_MAE_meV": combo["w90_H_MAE_meV"] if system == "w90" else "",
                    "production_H_MAE_meV_as_reported": reported, "production_reported_on": reported_on,
                    "dev_H_MAE_meV": summ["H_MAE_meV"], "dev_H_RMSE_meV": summ["H_RMSE_meV"],
                    "dev_rel_Frob": summ["rel_Frob"], "dev_hermiticity_eV": summ["hermiticity_eV"],
                    "evaluation_role": "development",
                    "evaluation_manifest": "w90_common_validation_s5" if system == "w90" else "common_dev_6x6_v1",
                    "checkpoint_policy": "last_epoch (pre-fix)",
                    "checkpoint_epoch": audit[rid]["checkpoint_epoch"], "best_val_epoch": audit[rid].get("best_epoch", ""),
                    "curve_role": ("n64_pool" if meta["density"] == 64 else
                                   ("prefix_of:" + pools[rid]["prefix_of_n64_pool"]) if pools[rid].get("prefix_of_n64_pool")
                                   else "historical_only (other sampler seed / not a prefix)"),
                })
        write_csv(OUT / f"reeval_per_structure_{system}.csv", per_structure)
    table.sort(key=lambda r: (r["system"], r["dev_H_MAE_meV"]))
    for system in SYSTEMS:
        rows = [r for r in table if r["system"] == system]
        for rank, row in enumerate(rows, 1):
            row["dev_rank_within_system"] = rank
    write_csv(OUT / "existing_results_160.csv", table)
    print(f"wrote {len(table)} rows")
    return 0


# --------------------------------------------------------------------------
# Phase 3 -- lock the curve recipes
# --------------------------------------------------------------------------


def dim_distance(a: str, b: str) -> int:
    return len(DIM_AXES[a] ^ DIM_AXES[b])


def choose_control(a: dict[str, Any], b: dict[str, Any], available: list[dict[str, Any]]) -> dict[str, Any]:
    """Pre-registered coverage control: uses only A, B and what exists -- never C's metric."""

    dims = [d for d in DIMS if d not in (a["dim"], b["dim"]) and any(r["dim"] == d for r in available)]
    dim_c = max(dims, key=lambda d: (min(dim_distance(d, a["dim"]), dim_distance(d, b["dim"])),
                                     dim_distance(d, a["dim"]) + dim_distance(d, b["dim"]),
                                     -DIM_TIE_ORDER.index(d)))
    options = [r for r in available if r["dim"] == dim_c]
    return max(options, key=lambda r: (min(abs(math.log(r["R"] / a["R"])), abs(math.log(r["R"] / b["R"]))), r["R"]))


def select_recipes(ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``ranking``: N=64 rows (recipe_id, family, dim, R, dev_H_MAE_meV) of one system."""

    selected = []
    for family in CURVE_FAMILIES:
        rows = sorted((r for r in ranking if r["family"] == family), key=lambda r: r["dev_H_MAE_meV"])
        a = rows[0]
        b = next(r for r in rows if r["dim"] != a["dim"])
        c = choose_control(a, b, [r for r in rows if r not in (a, b)])
        for role, row, reason in (
            ("A", a, "best dev H-MAE of the family"),
            ("B", b, "best dev H-MAE with dim != A"),
            ("C", c, "coverage control from A/B only (dim distance, then log-R distance)"),
        ):
            selected.append({**row, "role": role, "reason": reason, "family_rank": rows.index(row) + 1})
    return selected


def cmd_select(args: argparse.Namespace) -> int:
    path = OUT / "selection_manifest.json"
    manifest: dict[str, Any] = read_json(path) if path.exists() else {"systems": {}}
    for system in args.systems:
        if system in manifest["systems"]:
            print(f"{system} already locked in {path}; not overwriting (pre-registration)")
            continue
        pools = read_json(OUT / f"pools_{system}.json")
        ranking = []
        for rid, pool in pools.items():
            if pool["density"] != 64:
                continue
            rows = read_csv(OUT / "reeval" / system / rid / "per_structure_metrics.csv")
            ranking.append({"recipe_id": rid, "family": pool["family"], "dim": pool["dim"], "k": pool["k"],
                            "R": pool["R"], "dev_H_MAE_meV": float(np.mean([float(r["H_MAE_meV"]) for r in rows]))})
        curves = select_recipes(ranking)
        lhs = min((r for r in ranking if r["family"] == "latin_hypercube"), key=lambda r: r["dev_H_MAE_meV"])
        for row in curves + [lhs]:
            pool = pools[row["recipe_id"]]
            row.update({"sampler_seed": pool["sampler_seed"], "pool_hash": pool["pool_hash"],
                        "prefix_chain_verified": pool["prefix_chain_verified"]})
        dev_hashes = (read_json(OUT / "phase1_inventory.json")["w90_dev"]["hashes"] if system == "w90"
                      else [r["hash"] for r in read_json(OUT / "common_dev_6x6_v1.json")["samples"]])
        manifest["systems"][system] = {"curve_recipes": curves, "lhs_recipe": lhs, "ranking": ranking,
                                       "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                       "dev_hash": ids_hash(sorted(dev_hashes)), "n_dev": len(dev_hashes)}
    write_json(path, manifest)
    for system, block in manifest["systems"].items():
        for row in block["curve_recipes"]:
            print(system, row["role"], row["recipe_id"], round(row["dev_H_MAE_meV"], 2), "rank", row["family_rank"])
        print(system, "LHS", block["lhs_recipe"]["recipe_id"], round(block["lhs_recipe"]["dev_H_MAE_meV"], 2))
    return 0


# --------------------------------------------------------------------------
# Phase 8.1-8.2 -- trajectory-level MD split (w90 only)
# --------------------------------------------------------------------------


def md_frame_row(trajectory: dict[str, Any], t: int, steps_cost: int) -> dict[str, Any]:
    frame_dir = Path(trajectory["steps_dir"]) / str(t)
    sample = s4.Sample(
        sample_id=f"md__T{int(trajectory['temperature_K'])}__t{t:03d}", family="MD",
        dim="3D", k=2, amplitude_ang=0.0, run_fdf=frame_dir / "RUN.fdf", reference_dir=frame_dir,
        reference_matrix=frame_dir / "graphene.TSHS",
    )
    return sample_to_dict(sample, **geometry_record("w90", sample.run_fdf), temperature_K=trajectory["temperature_K"],
                          trajectory=trajectory["key"], t_fs=t,
                          # MD steps this frame adds to its trajectory: summed over a time-ordered
                          # prefix it gives exactly the (t_max + 1) steps that had to be run.
                          siesta_cpu_s=trajectory["per_step_siesta_s"] * steps_cost)


def md_blocks(n_frames: int, n_test: int, n_val: int) -> dict[str, list[int]]:
    """Temporal blocks of one trajectory, stride MD_MIN_T_FS, one stride of gap between blocks."""

    strided = list(range(MD_MIN_T_FS, n_frames, MD_MIN_T_FS))
    test = strided[-n_test:]
    val = strided[-(n_test + 1 + n_val):-(n_test + 1)]
    train = strided[:-(n_test + 1 + n_val + 1)]
    return {"train": train, "validation": val, "test": test}


def cmd_md_split(args: argparse.Namespace) -> int:
    inventory = read_json(OUT / "phase1_inventory.json")["md_inventory"]["trajectories"]
    split: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train_pool": []}
    train_by_t: dict[float, list[dict[str, Any]]] = {}
    for temperature in MD_TEMPERATURES:
        (trajectory,) = [t for t in inventory if t["temperature_K"] == temperature]
        blocks = md_blocks(trajectory["n_frames"], MD_TEST_PER_T[temperature], MD_VAL_PER_T)
        split["test"] += [md_frame_row(trajectory, t, 0) for t in blocks["test"]]
        split["validation"] += [md_frame_row(trajectory, t, 0) for t in blocks["validation"]]
        train_by_t[temperature] = [md_frame_row(trajectory, t, t + 1 if i == 0 else MD_MIN_T_FS)
                                   for i, t in enumerate(blocks["train"])]
    # Round-robin 150 -> 300 -> 450 K, time-ordered within each: every prefix mixes
    # temperatures and corresponds to running each trajectory up to its last used frame.
    for index in range(64):
        split["train_pool"].append(train_by_t[MD_TEMPERATURES[index % 3]][index // 3])

    all_rows = [row for rows in split.values() for row in rows]
    assert len({row["hash"] for row in all_rows}) == len(all_rows)
    for temperature in MD_TEMPERATURES:  # blocks are disjoint in time, train < validation < test
        times = {name: [r["t_fs"] for r in rows if r["temperature_K"] == temperature] for name, rows in split.items()}
        assert max(times["train_pool"]) < min(times["validation"]) and max(times["validation"]) < min(times["test"])
    designed = set(read_json(OUT / "training_hashes_w90.json")) | set(read_json(OUT / "phase1_inventory.json")["w90_dev"]["hashes"])
    assert not {row["hash"] for row in all_rows} & designed
    amplitudes = {name: [row["real_amplitude_ang"] for row in rows] for name, rows in split.items()}
    write_json(OUT / "md_split_w90.json", {
        "rule": "3 distinct trajectories (150/300/450 K); stride 6 fs; per trajectory train block first, "
                "then validation, then test, one 6-fs stride of gap between blocks",
        "t_fs_ranges": {name: {str(T): [min(r["t_fs"] for r in rows if r["temperature_K"] == T),
                                        max(r["t_fs"] for r in rows if r["temperature_K"] == T)]
                               for T in MD_TEMPERATURES} for name, rows in split.items()},
        "counts": {name: len(rows) for name, rows in split.items()},
        "real_amplitude_ang": {name: {"min": min(a), "median": float(np.median(a)), "max": max(a)} for name, a in amplitudes.items()},
        "R_train_md": max(amplitudes["train_pool"]),
        "pool_hash": ids_hash([row["hash"] for row in split["train_pool"]]), **split,
    })
    print(json.dumps({"counts": {k: len(v) for k, v in split.items()},
                      "amplitude": {k: [round(min(a), 4), round(max(a), 4)] for k, a in amplitudes.items()}}))
    return 0


# --------------------------------------------------------------------------
# Training (Phases 4, 5, 7, 8)
# --------------------------------------------------------------------------


def gpu_mib(pid: int) -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001 - memory telemetry must never kill a training
        return 0
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid) and parts[1].isdigit():
            return int(parts[1])
    return 0


def run_training(config_path: Path, run_dir: Path) -> tuple[Path, float, int]:
    """``s4.run_graph2mat_training`` with Popen, to sample this process's peak GPU memory."""

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(s4.TORCH_COMPAT_DIR), env.get("PYTHONPATH", "")]))
    peak = 0
    start = time.perf_counter()
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(s4.GRAPH2MAT_BIN), "models", "mace", "main", "fit", "-c", config_path.name],
                                   cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, env=env)
        while process.poll() is None:
            peak = max(peak, gpu_mib(process.pid))
            time.sleep(5)
    seconds = time.perf_counter() - start
    if process.returncode != 0:
        raise RuntimeError(f"graph2mat training failed (rc={process.returncode}); see {run_dir / 'train.log'}")
    best = sorted(run_dir.rglob("best-*.ckpt"))
    if len(best) != 1:
        raise RuntimeError(f"expected exactly one best-val_loss checkpoint under {run_dir}, found {best}")
    return best[0], seconds, peak


def make_job(system: str, recipe: dict[str, Any], train_rows: list[dict[str, Any]], n: int, seed: int,
             stage: str, val: str = "dev") -> dict[str, Any]:
    return {
        "system": system, "stage": stage, "recipe_id": recipe["recipe_id"], "family": recipe["family"],
        "dim": recipe["dim"], "k": recipe["k"], "R_train": recipe["R"], "sampler_seed": recipe.get("sampler_seed"),
        "pool_hash": recipe.get("pool_hash"), "N": n, "training_seed": seed, "val": val,
        "job_id": f"{recipe['recipe_id']}__N{n}__ts{seed}", "train": train_rows[:n],
    }


def val_samples(job: dict[str, Any]) -> list[s4.Sample]:
    if job["val"] == "md_validation":
        return [sample_from_dict(row) for row in read_json(OUT / "md_split_w90.json")["validation"]]
    return dev_samples(job["system"])


def result_path(job: dict[str, Any]) -> Path:
    return OUT / "runs" / job["system"] / job["job_id"] / "result.json"


def run_job(job: dict[str, Any], accelerator: str, concurrency: int) -> dict[str, Any]:
    run_dir = OUT / "runs" / job["system"] / job["job_id"]
    done_marker = run_dir / "train_done.json"
    if result_path(job).exists():
        return read_json(result_path(job))
    train = [sample_from_dict(row) for row in job["train"]]
    val = val_samples(job)
    assert len(train) == job["N"] and len({row["hash"] for row in job["train"]}) == job["N"]
    if done_marker.exists():  # training finished, only the evaluation failed: never retrain
        done = read_json(done_marker)
        checkpoint, seconds, peak = Path(done["checkpoint"]), done["seconds"], done["peak"]
    else:
        if run_dir.exists():  # interrupted training: start clean, never mix checkpoints
            shutil.rmtree(run_dir)
        config = s4.build_graph2mat_config(run_dir, train, val, run_name=job["job_id"], accelerator=accelerator,
                                           training_seed=job["training_seed"], **TRAIN_KW)
        checkpoint, seconds, peak = run_training(config, run_dir)
        write_json(done_marker, {"checkpoint": str(checkpoint), "seconds": seconds, "peak": peak})
    epochs = tensorboard_epochs(run_dir, checkpoint)
    rows = evaluate(checkpoint, val, run_dir / "val_eval", accelerator)
    result = {key: value for key, value in job.items() if key != "train"} | {
        "train_ids_hash": ids_hash([row["sample_id"] for row in job["train"]]),
        "train_geometry_hash": ids_hash([row["hash"] for row in job["train"]]),
        "train_max_real_amplitude_ang": max(row["real_amplitude_ang"] for row in job["train"]),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_policy": "best_val_loss (ModelCheckpoint save_top_k=1)",
        "checkpoint_epoch": checkpoint_epoch(checkpoint), **epochs,
        "train_seconds": seconds, "gpu_h": seconds / 3600, "concurrent_jobs": concurrency, "peak_gpu_mib": peak,
        "siesta_cpu_h_train": sum(float(row["siesta_cpu_s"]) for row in job["train"]) / 3600,
        "val_metrics": summarize(rows), "spectral_ref_S": True,
    }
    write_json(result_path(job), result)
    return result


def cmd_rescore_val(args: argparse.Namespace) -> int:
    """Fix the Gamma spectral column of results scored before predictions used the reference S.

    Only ``spectral_err_meV`` is replaced; H-MAE and the other H metrics (the
    decision inputs) are left exactly as first computed.
    """

    accelerator = s4.torch_backend_preflight()["effective_backend"]
    for system in SYSTEMS:
        for path in sorted((OUT / "runs" / system).glob("*/result.json")):
            result = read_json(path)
            if result.get("spectral_ref_S"):
                continue
            samples = val_samples(result)
            out_dir = path.parent / "val_eval_rescore"
            predicted_root = predict(Path(result["checkpoint"]), samples, out_dir, accelerator)
            fixed = {sample.sample_id: structure_metrics(sample, predicted_root)["spectral_err_meV"] for sample in samples}
            rows = read_csv(path.parent / "val_eval" / "per_structure_metrics.csv")
            for row in rows:
                row["spectral_err_meV"] = fixed[row["sample_id"]]
            write_csv(path.parent / "val_eval" / "per_structure_metrics.csv", rows)
            shutil.rmtree(out_dir, ignore_errors=True)
            result["val_metrics"]["spectral_err_meV"] = float(np.nanmean(list(fixed.values())))
            result["spectral_ref_S"] = True
            write_json(path, result)
            print("rescored", result["job_id"], round(result["val_metrics"]["spectral_err_meV"], 1), flush=True)
    return 0


def curve_jobs(system: str) -> list[dict[str, Any]]:
    selection = read_json(OUT / "selection_manifest.json")["systems"][system]
    pools = read_json(OUT / f"pools_{system}.json")
    return [make_job(system, recipe, pools[recipe["recipe_id"]]["samples"], n, 0, "curve")
            for recipe in selection["curve_recipes"] for n in NS]


def lhs_jobs(system: str) -> list[dict[str, Any]]:
    recipe = read_json(OUT / "selection_manifest.json")["systems"][system]["lhs_recipe"]
    pool = read_json(OUT / f"pools_{system}.json")[recipe["recipe_id"]]["samples"]
    return [make_job(system, recipe, pool, 64, seed, "lhs") for seed in (0, 1, 2)]


def confirm_jobs(system: str) -> list[dict[str, Any]]:
    finalists = read_json(OUT / "finalists.json")["systems"][system]
    pools = read_json(OUT / f"pools_{system}.json")
    jobs = []
    for finalist in finalists["finalists"]:
        pool = pools[finalist["recipe_id"]]["samples"]
        for n in sorted({finalist["N_star_candidate"], 64}):
            for seed in (0, 1, 2):
                jobs.append(make_job(system, finalist, pool, n, seed, "confirm"))
    return jobs


def md_jobs(system: str) -> list[dict[str, Any]]:
    if system != "w90":
        return []
    split = read_json(OUT / "md_split_w90.json")
    n_star = read_json(OUT / "finalists.json")["systems"]["w90"]["md_sizes"]
    recipe = {"recipe_id": "md_w90", "family": "MD", "dim": "3D", "k": 2, "R": split["R_train_md"],
              "sampler_seed": None, "pool_hash": split["pool_hash"]}
    return [make_job("w90", recipe, split["train_pool"], n, seed, "md", val="md_validation")
            for n in n_star for seed in (0, 1, 2)]


STAGES = {"curves": curve_jobs, "lhs": lhs_jobs, "confirm": confirm_jobs, "md": md_jobs}


def cmd_train(args: argparse.Namespace) -> int:
    accelerator = s4.torch_backend_preflight()["effective_backend"]
    failures = 0
    for system in args.systems:
        jobs = [job for job in STAGES[args.stage](system) if not result_path(job).exists()]
        parallel = args.parallel or PARALLEL[system]
        print(json.dumps({"stage": args.stage, "system": system, "pending": len(jobs), "parallel": parallel}), flush=True)
        # Largest N first: the long jobs start early and the tail packs better.
        jobs.sort(key=lambda job: -job["N"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(run_job, job, accelerator, parallel): job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    print(json.dumps({"done": job["job_id"], "H_MAE_meV": round(result["val_metrics"]["H_MAE_meV"], 3),
                                      "best_epoch": result.get("best_epoch"), "final_epoch": result.get("final_epoch"),
                                      "min": round(result["train_seconds"] / 60, 1)}), flush=True)
                except Exception as exc:  # noqa: BLE001 - record and continue; rerun resumes
                    failures += 1
                    print(json.dumps({"failed": job["job_id"], "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    return 1 if failures else 0


# --------------------------------------------------------------------------
# Phase 6 (finalists) and Phase 7 (N* confirmation)
# --------------------------------------------------------------------------


def load_results(system: str) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted((OUT / "runs" / system).glob("*/result.json"))]


def n_star(curve: dict[int, float], tol: float = SATURATION_TOL) -> int:
    """min{N : E(N) <= (1 + tol) E(64)}."""

    return min(n for n, e in curve.items() if e <= (1 + tol) * curve[64])


def gains(curve: dict[int, float]) -> dict[str, float]:
    return {f"{n}->{2 * n}": (curve[n] - curve[2 * n]) / curve[n] for n in NS[:-1] if n in curve and 2 * n in curve}


def needs_n12(curve: dict[int, float], tol: float = SATURATION_TOL) -> bool:
    g = gains(curve)
    return (g["8->16"] > 2 * max(g["4->8"], g["16->32"]) and curve[8] > (1 + tol) * curve[64]
            and curve[16] <= (1 + tol) * curve[64])


def pareto_front(points: list[dict[str, Any]], cost: str = "cost", error: str = "E") -> list[dict[str, Any]]:
    """Non-dominated points: nobody has cost <= and error <= with one strict."""

    def dominated(p: dict[str, Any]) -> bool:
        return any(q[cost] <= p[cost] and q[error] <= p[error] and (q[cost] < p[cost] or q[error] < p[error])
                   for q in points)

    return sorted((p for p in points if not dominated(p)), key=lambda p: p[cost])


def choose_finalists(curves: dict[str, dict[str, Any]], lhs: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Pre-registered Phase 6 rules. ``curves[rid]`` = recipe meta + E{N} + cost{N} (seed 0).

    ``lhs`` (only if it passed Phase 5) = meta + E64 (3-seed mean) + cost64.
    """

    candidates = dict(curves)
    if lhs is not None:
        candidates[lhs["recipe_id"]] = lhs
    precision = min(candidates.values(), key=lambda r: r["E"][64])
    threshold = (1 + SATURATION_TOL) * precision["E"][64]
    points = [{"recipe_id": rid, "N": n, "E": e, "cost": r["cost"][n]}
              for rid, r in candidates.items() for n, e in r["E"].items()]
    eligible = sorted((p for p in points if p["E"] <= threshold), key=lambda p: (p["cost"], p["E"]))
    efficient = eligible[0]
    finalists = [{**precision, "role": "precision", "N_star_candidate": n_star(precision["E"]),
                  "reason": "lowest dev E(64)"}]
    if efficient["recipe_id"] == precision["recipe_id"]:
        finalists[0]["N_star_candidate"] = efficient["N"]
        finalists[0]["reason"] += "; also the cheapest point within 5% of it"
        others = [p for p in eligible if p["recipe_id"] != precision["recipe_id"]]
        if others:
            efficient, reason = others[0], "cheapest point of another recipe within 5% (precision == efficient)"
        else:
            # "Next" frontier point: the neighbour just below the precision point in cost.
            anchor = next(p["cost"] for p in points if p["recipe_id"] == precision["recipe_id"]
                          and p["N"] == finalists[0]["N_star_candidate"])
            front = [p for p in pareto_front(points) if p["recipe_id"] != precision["recipe_id"]]
            below = [p for p in front if p["cost"] <= anchor]
            efficient = max(below, key=lambda p: p["cost"]) if below else front[0]
            reason = "next Pareto (cost, E) point below the precision point, other recipe (none within 5%)"
    else:
        reason = "cheapest (recipe, N) with E <= 1.05 E_precision(64)"
    finalists.append({**candidates[efficient["recipe_id"]], "role": "efficient", "N_star_candidate": efficient["N"],
                      "reason": reason})

    chosen = {f["recipe_id"] for f in finalists}
    families = {f["family"] for f in finalists}
    remaining = sorted((r for rid, r in candidates.items() if rid not in chosen), key=lambda r: r["E"][64])
    alternative = next((r for r in remaining if r["family"] in CURVE_FAMILIES and r["family"] not in families), None)
    reason = "best remaining recipe of an unrepresented family"
    if alternative is None and lhs is not None and lhs["recipe_id"] not in chosen:
        alternative, reason = lhs, "LHS N=64 passed Phase 5"
    if alternative is None:
        dims = {f["dim"] for f in finalists}
        alternative = next(r for r in remaining if r["dim"] not in dims)
        reason = "best remaining recipe with a dimensionality not in the other finalists"
    finalists.append({**alternative, "role": "alternative", "reason": reason,
                      "N_star_candidate": 64 if alternative["family"] == "latin_hypercube" else n_star(alternative["E"])})
    return finalists


def cmd_finalists(args: argparse.Namespace) -> int:
    path = OUT / "finalists.json"
    selection = read_json(OUT / "selection_manifest.json")["systems"]
    payload: dict[str, Any] = read_json(path) if path.exists() else {"systems": {}}
    for system in args.systems:
        if system in payload["systems"]:
            print(f"{system} finalists already locked; use `confirm` for Phase 7")
            continue
        results = load_results(system)
        curves: dict[str, dict[str, Any]] = {}
        for recipe in selection[system]["curve_recipes"]:
            rows = {r["N"]: r for r in results if r["recipe_id"] == recipe["recipe_id"] and r["stage"] == "curve"}
            if set(rows) != set(NS):
                raise RuntimeError(f"{system}/{recipe['recipe_id']}: curve incomplete, have N={sorted(rows)}")
            curve = {n: rows[n]["val_metrics"]["H_MAE_meV"] for n in NS}
            curves[recipe["recipe_id"]] = {
                **{k: recipe[k] for k in ("recipe_id", "family", "dim", "k", "R", "sampler_seed", "pool_hash")},
                "curve_role": recipe["role"], "E": curve, "cost": {n: rows[n]["siesta_cpu_h_train"] for n in NS},
                "N_star": n_star(curve), "gains": gains(curve), "needs_N12": needs_n12(curve),
            }
        lhs_recipe = selection[system]["lhs_recipe"]
        lhs_rows = [r for r in results if r["stage"] == "lhs"]
        best_e64 = min(c["E"][64] for c in curves.values())
        lhs_e = [r["val_metrics"]["H_MAE_meV"] for r in lhs_rows]
        lhs_valid = len(lhs_rows) == 3 and all(math.isfinite(e) and r["val_metrics"]["hermiticity_eV"] < 1e-6
                                               for e, r in zip(lhs_e, lhs_rows))
        lhs_pass = lhs_valid and float(np.mean(lhs_e)) <= (1 + SATURATION_TOL) * best_e64
        lhs = None
        if lhs_pass:
            lhs = {**{k: lhs_recipe[k] for k in ("recipe_id", "family", "dim", "k", "R", "sampler_seed", "pool_hash")},
                   "E": {64: float(np.mean(lhs_e))}, "cost": {64: lhs_rows[0]["siesta_cpu_h_train"]}}
        finalists = choose_finalists(curves, lhs)
        for finalist in finalists:
            finalist["N_star_history"] = [finalist["N_star_candidate"]]
        payload["systems"][system] = {
            "curves": curves, "finalists": finalists, "confirmed": False,
            "lhs": {"recipe_id": lhs_recipe["recipe_id"], "E_seeds": lhs_e, "mean": float(np.mean(lhs_e)) if lhs_e else None,
                    "best_sobol_random_E64": best_e64, "valid": lhs_valid, "passed": lhs_pass},
            "n12_needed_for": sorted(rid for rid, c in curves.items() if c["needs_N12"]),
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    write_json(path, payload)
    for system, block in payload["systems"].items():
        for rid, c in block["curves"].items():
            print(system, c["curve_role"], rid, {n: round(e, 2) for n, e in c["E"].items()}, "N*", c["N_star"])
        print(system, "LHS", block["lhs"])
        for f in block["finalists"]:
            print(system, "FINALIST", f["role"], f["recipe_id"], "N*cand", f["N_star_candidate"], "-", f["reason"])
        print(system, "N=12 needed for:", block["n12_needed_for"])
    return 0


def bootstrap_mean_ci(values: np.ndarray, *, level: float = 0.95, one_sided_upper: bool = False) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = values[rng.integers(0, len(values), size=(BOOTSTRAP_B, len(values)))].mean(axis=1)
    if one_sided_upper:
        return float("-inf"), float(np.percentile(means, 100 * level))
    alpha = (1 - level) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def seed_stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    low, high = bootstrap_mean_ci(array)
    return {"values": values, "mean": float(array.mean()), "median": float(np.median(array)),
            "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0, "min": float(array.min()),
            "max": float(array.max()), "boot95": [low, high]}


def confirm_n_star(e_n: list[float], e_64: list[float], valid: bool) -> dict[str, Any]:
    """Pre-registered Phase 7 rules (3 seeds at N* and at 64)."""

    limit = 1 + SATURATION_TOL
    mean_n, mean_64 = float(np.mean(e_n)), float(np.mean(e_64))
    rule1 = mean_n <= limit * mean_64 and float(np.median(e_n)) <= limit * float(np.median(e_64))
    rule2 = all(np.mean(np.delete(e_n, i)) <= limit * np.mean(np.delete(e_64, i)) for i in range(len(e_n)))
    rule3 = float(np.std(e_n, ddof=1)) <= max(abs(mean_n - mean_64), SATURATION_TOL * mean_64)
    return {"rule1_mean_and_median": bool(rule1), "rule2_leave_one_seed_out": bool(rule2),
            "rule3_seed_spread": bool(rule3), "rule4_valid": bool(valid),
            "confirmed": bool(rule1 and rule2 and rule3 and valid)}


def cmd_confirm(args: argparse.Namespace) -> int:
    payload = read_json(OUT / "finalists.json")
    all_done = True
    for system in payload["systems"]:
        block = payload["systems"][system]
        results = load_results(system)
        for finalist in block["finalists"]:
            by_n: dict[int, list[dict[str, Any]]] = {}
            for r in results:
                if r["recipe_id"] == finalist["recipe_id"] and r["stage"] in ("curve", "confirm", "lhs"):
                    by_n.setdefault(r["N"], {})[r["training_seed"]] = r
            n = finalist["N_star_candidate"]
            if any(len(by_n.get(size, {})) < 3 for size in {n, 64}):
                all_done = False
                finalist["confirmation"] = {"status": "pending seeds", "have": {s: sorted(v) for s, v in by_n.items()}}
                continue
            e_n = [by_n[n][s]["val_metrics"]["H_MAE_meV"] for s in (0, 1, 2)]
            e_64 = [by_n[64][s]["val_metrics"]["H_MAE_meV"] for s in (0, 1, 2)]
            valid = all(math.isfinite(by_n[size][s]["val_metrics"]["H_MAE_meV"])
                        and by_n[size][s]["val_metrics"]["hermiticity_eV"] < 1e-6 for size in {n, 64} for s in (0, 1, 2))
            check = confirm_n_star(e_n, e_64, valid) if n != 64 else {"confirmed": valid, "rule4_valid": valid,
                                                                     "note": "N*=64: nothing smaller to confirm"}
            finalist["confirmation"] = {"N": n, "E_N": seed_stats(e_n), "E_64": seed_stats(e_64), **check}
            if not check["confirmed"] and n == 64:
                all_done = False
                print(f"{system} {finalist['recipe_id']}: N=64 runs invalid -- inspect before continuing")
            elif not check["confirmed"]:
                next_n = NS[NS.index(n) + 1]
                finalist["N_star_candidate"] = next_n
                finalist["N_star_history"].append(next_n)
                all_done = False
                print(f"{system} {finalist['recipe_id']}: N*={n} not confirmed -> escalate to {next_n}")
        if all(f.get("confirmation", {}).get("confirmed") for f in block["finalists"]):
            block["confirmed"] = True
            # MD at the confirmed N* of the precision and efficient finalists, and at 64.
            block["md_sizes"] = sorted({f["N_star_candidate"] for f in block["finalists"]
                                        if f["role"] in ("precision", "efficient")} | {64})
    write_json(OUT / "finalists.json", payload)
    for system in payload["systems"]:
        for f in payload["systems"][system]["finalists"]:
            conf = f.get("confirmation", {})
            print(system, f["role"], f["recipe_id"], "N*", f["N_star_candidate"], "confirmed", conf.get("confirmed"),
                  {k: v for k, v in conf.items() if k.startswith("rule")})
    print("ALL CONFIRMED" if all_done else "pending: run `train --stage confirm` then `confirm` again")
    return 0 if all_done else 2


# --------------------------------------------------------------------------
# Phase 9 -- frozen final test
# --------------------------------------------------------------------------


def final_test_configs(system: str) -> list[tuple[str, Any, dict[str, Any]]]:
    """Box sample in the dim axes, rescaled so max_a |dr_a| == A exactly (pre-registered generator)."""

    geometry = sampler.load_graphene_primitive() if system == "w90" else sampler.load_graphene_6x6()
    ks, replicas = ((1, 2), (0,)) if system == "w90" else ((72,), (0, 1))
    entries = []
    for k in ks:
        active = sampler.select_active_atoms(geometry, k)
        for dim_index, dim in enumerate(DIMS):
            axes = sampler.dimensionality_axes(dim)
            for amp_index, amplitude in enumerate(FINAL_AMPLITUDES):
                for replica in replicas:
                    rng = np.random.default_rng([FINAL_SEEDS[system], k, dim_index, amp_index, replica])
                    vectors = {i: sum(rng.uniform(-1, 1) * axis for axis in axes) for i in active.indices}
                    scale = amplitude / max(np.linalg.norm(v) for v in vectors.values())
                    displacements = {i: tuple((scale * v).tolist()) for i, v in vectors.items()}
                    metadata = sampler.build_metadata(geometry, active, displacements, family="final_test",
                                                      dim=dim, amplitude_ang=amplitude)
                    sample_id = f"final_{system}__k{k}__{dim}__A{amplitude:.2f}__r{replica}"
                    entries.append((sample_id, sampler.Configuration("final_test", displacements, metadata),
                                    {"k": k, "dim": dim, "A": amplitude, "replica": replica}))
    return entries


def frozen_models(system: str) -> list[dict[str, Any]]:
    return [{key: result.get(key) for key in (
        "system", "stage", "recipe_id", "family", "dim", "k", "R_train", "N", "training_seed", "job_id",
        "checkpoint", "checkpoint_sha256", "siesta_cpu_h_train", "gpu_h", "peak_gpu_mib", "best_epoch",
        "final_epoch", "train_max_real_amplitude_ang", "concurrent_jobs")} for result in load_results(system)]


def all_frozen_models() -> list[dict[str, Any]]:
    return [m for block in read_json(OUT / "frozen_models.json")["systems"].values() for m in block["models"]]


def md_relabel_check(md_split: dict[str, Any], workers: int) -> list[dict[str, Any]]:
    """Re-run 4 MD test frames as SIESTA single points: is the MD label the synthetic label?"""

    geometry = sampler.load_graphene_primitive()
    base = base_positions("w90")
    entries = []
    for row in md_split["test"][:4]:
        positions = np.asarray(s5._positions_from_run_fdf(Path(row["run_fdf"])))
        displacements = {i: tuple((positions[i] - base[i]).tolist()) for i in range(len(base))}
        metadata = sampler.build_metadata(geometry, sampler.select_active_atoms(geometry, 2), displacements,
                                          family="md_relabel", dim="3D", amplitude_ang=row["real_amplitude_ang"])
        entries.append((f"relabel__{row['sample_id']}", sampler.Configuration("md_relabel", displacements, metadata)))
    root = label_structures(entries, OUT / "md_relabel_check", sampler.GRAPHENE_PRIMITIVE_FDF, workers)
    checks = []
    for (sample_id, _), row in zip(entries, md_split["test"][:4]):
        h_md = s4.gamma_hk(Path(row["reference_matrix"]))
        h_sp = s4.gamma_hk(root / sample_id / "graphene.TSHS")
        checks.append({"frame": row["sample_id"], "max_abs_dH_meV": 1e3 * float(np.abs(h_md - h_sp).max()),
                       "mean_abs_dH_meV": 1e3 * float(np.abs(h_md - h_sp).mean())})
    return checks


def cmd_final_test(args: argparse.Namespace) -> int:
    finalists = read_json(OUT / "finalists.json")["systems"]
    frozen_path, manifest_path = OUT / "frozen_models.json", OUT / "final_test_manifest.json"
    frozen = read_json(frozen_path) if frozen_path.exists() else {"systems": {}}
    manifest = read_json(manifest_path) if manifest_path.exists() else {"systems": {}}
    md_split = read_json(OUT / "md_split_w90.json")
    for system in args.systems:
        if system in manifest["systems"]:
            print(f"{system}: final test already frozen")
            continue
        if not finalists.get(system, {}).get("confirmed"):
            raise SystemExit(f"{system}: Phase 7 not confirmed; the final test comes after freezing N* and seeds")
        if system == "w90" and len([r for r in load_results("w90") if r["stage"] == "md"]) != 3 * len(finalists["w90"]["md_sizes"]):
            raise SystemExit("w90: MD baseline not trained yet; freeze requires it")
        # 1. Freeze the models (checkpoint hashes) before the test exists.
        if system not in frozen["systems"]:
            frozen["systems"][system] = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "models": frozen_models(system)}
            write_json(frozen_path, frozen)
        for model in frozen["systems"][system]["models"]:
            if sha256_file(Path(model["checkpoint"])) != model["checkpoint_sha256"]:
                raise SystemExit(f"checkpoint changed after freeze: {model['checkpoint']}")
        # 2. Build + label the pre-registered test.
        entries = final_test_configs(system)
        material = sampler.GRAPHENE_PRIMITIVE_FDF if system == "w90" else sampler.GRAPHENE_6X6_FDF
        reference_root = label_structures([(sid, cfg) for sid, cfg, _ in entries], OUT / f"final_test_{system}",
                                          material, args.workers)
        rows = []
        for sample_id, _config, meta in entries:
            row = labelled_sample(system, sample_id, reference_root, {
                "family": "final_test", "dim": meta["dim"], "k": meta["k"], "amplitude_ang": meta["A"],
                "replica": meta["replica"], "origin": "synthetic"})
            if abs(row["real_amplitude_ang"] - meta["A"]) > 1e-5:
                raise RuntimeError(f"{sample_id}: real amplitude {row['real_amplitude_ang']} != {meta['A']}")
            rows.append(row)
        if system == "w90":
            rows += [row | {"origin": "md"} for row in md_split["test"]]
        taken = set(read_json(OUT / f"training_hashes_{system}.json"))
        taken |= set(read_json(OUT / "phase1_inventory.json")["w90_dev"]["hashes"]) if system == "w90" else {
            r["hash"] for r in read_json(OUT / "common_dev_6x6_v1.json")["samples"]}
        if system == "w90":
            taken |= {r["hash"] for name in ("validation", "train_pool") for r in md_split[name]}
        hashes = [row["hash"] for row in rows]
        assert len(set(hashes)) == len(hashes) and not set(hashes) & taken
        block = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n": len(rows), "n_new_siesta": len(entries),
                 "test_hash": ids_hash(sorted(hashes)), "samples": rows}
        if system == "w90":
            block["md_label_equivalence_check"] = md_relabel_check(md_split, args.workers)
        manifest["systems"][system] = block
        write_json(manifest_path, manifest)
        print(json.dumps({system: {"n": block["n"], "new_siesta": block["n_new_siesta"],
                                   "md_relabel": block.get("md_label_equivalence_check")}}, indent=1))
    return 0


# --------------------------------------------------------------------------
# Phase 10 -- single evaluation pass on the final test
# --------------------------------------------------------------------------


def band_dos_metrics(sample: s4.Sample, predicted: Path, system: str) -> dict[str, float]:
    """Band RMSE on the G-K-M-G path (+-2 eV of E_F) and relative L1 DOS difference (E_F +-3 eV)."""

    import sisl

    h_pred = sisl.get_sile(str(predicted)).read_hamiltonian()
    h_ref = sisl.get_sile(str(sample.reference_matrix)).read_hamiltonian()
    e_fermi = s4.fermi_level_ev(sample.reference_matrix)
    import scipy.linalg

    def eig_pred_at(k: Any) -> np.ndarray:  # predicted H with the reference S (predictions carry S = 1)
        return scipy.linalg.eigh(h_pred.Hk(k=k, format="array"), h_ref.Sk(k=k, format="array"), eigvals_only=True)

    path = sisl.BandStructure(h_ref.geometry, [[0, 0, 0], [1 / 3, 2 / 3, 0], [0.5, 0.5, 0], [0, 0, 0]], 60)
    eig_ref = np.array([h_ref.eigh(k=k) for k in path.k])
    eig_pred = np.array([eig_pred_at(k) for k in path.k])
    mask = np.abs(eig_ref - e_fermi) <= 2.0
    grid = 24 if system == "w90" else 4
    mp = sisl.MonkhorstPack(h_ref.geometry, [grid, grid, 1])
    energies = np.linspace(e_fermi - 3, e_fermi + 3, 601)

    def dos(eig_at: Any) -> np.ndarray:
        eig = np.concatenate([eig_at(k) for k in mp.k])
        return np.exp(-((energies[:, None] - eig[None, :]) ** 2) / (2 * 0.1**2)).sum(axis=1)

    d_ref, d_pred = dos(lambda k: h_ref.eigh(k=k)), dos(eig_pred_at)
    return {"band_rmse_meV": 1e3 * float(np.sqrt(np.mean((eig_pred[mask] - eig_ref[mask]) ** 2))),
            "dos_rel_L1": float(np.abs(d_pred - d_ref).sum() / d_ref.sum())}


def cmd_evaluate_final(args: argparse.Namespace) -> int:
    accelerator = s4.torch_backend_preflight()["effective_backend"]
    manifest = read_json(OUT / "final_test_manifest.json")
    finalists = read_json(OUT / "finalists.json")["systems"]
    models = [m for m in all_frozen_models() if m["system"] in args.systems]

    def _one(model: dict[str, Any]) -> str:
        system = model["system"]
        out_dir = OUT / "runs" / system / model["job_id"] / "final_test"
        if (out_dir / "per_structure_metrics.csv").exists():
            return model["job_id"]
        samples = [sample_from_dict(row) for row in manifest["systems"][system]["samples"]]
        finalist_ids = {f["recipe_id"] for f in finalists[system]["finalists"]}
        spectral = model["recipe_id"] in finalist_ids or model["stage"] == "md"
        predicted_root = predict(Path(model["checkpoint"]), samples, out_dir, accelerator)
        rows = []
        for sample in samples:
            row = structure_metrics(sample, predicted_root)
            if row is None:
                raise RuntimeError(f"{model['job_id']}: missing prediction for {sample.sample_id}")
            if spectral:
                row |= band_dos_metrics(sample, predicted_root / sample.sample_id / "ML_prediction.HSX", system)
            rows.append(row)
        write_csv(out_dir / "per_structure_metrics.csv", rows)
        shutil.rmtree(predicted_root, ignore_errors=True)
        return model["job_id"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for done in pool.map(_one, models):
            print("evaluated", done, flush=True)
    return 0


# --------------------------------------------------------------------------
# Phases 10-12 -- analysis, figures, final table
# --------------------------------------------------------------------------

COLORS = {"sobol_sparse": "#2a78d6", "random_cartesian": "#eb6834", "latin_hypercube": "#1baf7a", "MD": "#52514e"}
MARKERS = {"sobol_sparse": "o", "random_cartesian": "s", "latin_hypercube": "^", "MD": "D"}
LABELS = {"sobol_sparse": "Sobol", "random_cartesian": "random", "latin_hypercube": "LHS", "MD": "MD"}
ROLE_STYLE = {"A": "-", "B": "--", "C": ":"}


def final_long_table() -> Any:
    """One row per (frozen model, final-test structure)."""

    import pandas as pd

    manifest = read_json(OUT / "final_test_manifest.json")
    test_meta = {row["sample_id"]: {"origin": row.get("origin", "synthetic"), "A": row["amplitude_ang"],
                                    "real_A": row["real_amplitude_ang"], "test_dim": row["dim"], "test_k": row["k"]}
                 for block in manifest["systems"].values() for row in block["samples"]}
    frames = []
    for model in all_frozen_models():
        rows = pd.read_csv(OUT / "runs" / model["system"] / model["job_id"] / "final_test" / "per_structure_metrics.csv")
        for key in ("system", "stage", "recipe_id", "family", "dim", "R_train", "N", "training_seed", "job_id",
                    "siesta_cpu_h_train", "gpu_h", "peak_gpu_mib", "best_epoch", "final_epoch",
                    "train_max_real_amplitude_ang"):
            rows[key] = model[key]
        frames.append(rows)
    table = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame.from_dict(test_meta, orient="index").rename_axis("sample_id").reset_index()
    table = table.merge(meta, on="sample_id", how="left")
    # MD models: domain = the largest real amplitude they were trained on.
    r_domain = np.where(table["family"] == "MD", table["train_max_real_amplitude_ang"], table["R_train"])
    table["region"] = np.where(table["origin"] == "md", "md_frame",
                               np.where(table["real_A"] <= r_domain + 1e-9, "in_domain", "ood"))
    return table


def config_key(row: Any) -> str:
    return f"{row['recipe_id']}__N{row['N']}"


def paired(table: Any, key_a: tuple[str, int], key_b: tuple[str, int], system: str, subset: Any = None) -> dict[str, Any]:
    """Seed-averaged per-structure errors of two configs, paired over the same structures."""

    sub = table[table["system"] == system] if subset is None else subset

    def seed_by_structure(recipe: str, n: int) -> Any:
        rows = sub[(sub["recipe_id"] == recipe) & (sub["N"] == n)]
        return rows.pivot_table(index="training_seed", columns="sample_id", values="H_MAE_meV")

    ma, mb = seed_by_structure(*key_a), seed_by_structure(*key_b)
    common = ma.columns.intersection(mb.columns)
    a, b = ma[common].to_numpy(), mb[common].to_numpy()
    d = a.mean(axis=0) - b.mean(axis=0)
    low, high = bootstrap_mean_ci(d)
    # Seeds + structures: resample the seeds of each arm and the structures (training noise included).
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pick_a = a[rng.integers(0, len(a), (BOOTSTRAP_B, len(a)))].mean(axis=1)
    pick_b = b[rng.integers(0, len(b), (BOOTSTRAP_B, len(b)))].mean(axis=1)
    cols = rng.integers(0, len(common), (BOOTSTRAP_B, len(common)))
    boot = np.take_along_axis(pick_a - pick_b, cols, axis=1).mean(axis=1)
    return {"system": system, "A": f"{key_a[0]}@N{key_a[1]}", "B": f"{key_b[0]}@N{key_b[1]}",
            "seeds_A": len(a), "seeds_B": len(b), "n_structures": int(len(d)),
            "E_A": float(a.mean()), "E_B": float(b.mean()), "mean_d": float(d.mean()),
            "ci95_low": low, "ci95_high": high, "upper95_one_sided": bootstrap_mean_ci(d, one_sided_upper=True)[1],
            "ci95_seeds_structs_low": float(np.percentile(boot, 2.5)),
            "ci95_seeds_structs_high": float(np.percentile(boot, 97.5)),
            "upper95_seeds_structs": float(np.percentile(boot, 95)),
            "rel_diff": float(d.mean() / b.mean()), "_d": d, "_ids": list(common)}


def markdown_table(frame: Any) -> str:
    cells = [[("" if value is None or value != value else str(value)) for value in row] for row in frame.itertuples(index=False)]
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in cells]) + "\n"


def cmd_analyze(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import pandas as pd

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160})
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table = final_long_table()
    systems = [s for s in SYSTEMS if s in set(table["system"])]
    table.to_csv(OUT / "final_per_structure_long.csv", index=False)
    finalists = read_json(OUT / "finalists.json")["systems"]
    selection = read_json(OUT / "selection_manifest.json")["systems"]

    # ---- per-model and per-config summaries on the final test --------------------------------
    agg = {"H_MAE_meV": "mean", "H_RMSE_meV": "mean", "rel_Frob": "mean", "hermiticity_eV": "max"}
    models = table.groupby(["system", "stage", "recipe_id", "family", "dim", "R_train", "N", "training_seed",
                            "siesta_cpu_h_train", "gpu_h", "peak_gpu_mib", "best_epoch", "final_epoch"],
                           dropna=False).agg(agg).reset_index()
    for origin in ("synthetic", "md"):
        part = table[table["origin"] == origin].groupby("job_id")["H_MAE_meV"].mean()
        models[f"H_MAE_{origin}_meV"] = models.apply(
            lambda r, part=part: part.get(f"{r['recipe_id']}__N{r['N']}__ts{r['training_seed']}", np.nan), axis=1)
    models.to_csv(OUT / "final_models.csv", index=False)

    configs = []
    for (system, recipe, n), rows in table.groupby(["system", "recipe_id", "N"]):
        per_structure = rows.groupby("sample_id")["H_MAE_meV"].mean().to_numpy()
        per_seed = rows.groupby("training_seed")["H_MAE_meV"].mean()
        low, high = bootstrap_mean_ci(per_structure)
        first = rows.iloc[0]
        configs.append({
            "system": system, "recipe_id": recipe, "family": first["family"], "dim": first["dim"],
            "R_train": first["R_train"], "N": n, "n_seeds": len(per_seed), "H_MAE_meV": float(per_structure.mean()),
            "ci95": [low, high], "seed_sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
            "seed_values": per_seed.round(4).tolist(),
            "H_MAE_synthetic_meV": float(rows[rows["origin"] == "synthetic"]["H_MAE_meV"].mean()),
            "H_MAE_md_frames_meV": float(rows[rows["origin"] == "md"]["H_MAE_meV"].mean()) if (rows["origin"] == "md").any() else None,
            "rel_Frob": float(rows["rel_Frob"].mean()), "H_RMSE_meV": float(rows["H_RMSE_meV"].mean()),
            "band_rmse_meV": float(rows["band_rmse_meV"].mean()) if "band_rmse_meV" in rows and rows["band_rmse_meV"].notna().any() else None,
            "dos_rel_L1": float(rows["dos_rel_L1"].mean()) if "dos_rel_L1" in rows and rows["dos_rel_L1"].notna().any() else None,
            "siesta_cpu_h": float(first["siesta_cpu_h_train"]), "gpu_h_mean": float(rows.groupby("training_seed")["gpu_h"].first().mean()),
            "best_epoch_mean": float(rows.groupby("training_seed")["best_epoch"].first().mean()),
            "peak_gpu_mib_max": float(rows["peak_gpu_mib"].max()),
        })
    configs_df = pd.DataFrame(configs)
    configs_df.to_csv(OUT / "final_configs.csv", index=False)
    by_config = {(c["system"], c["recipe_id"], c["N"]): c for c in configs}

    # ---- learning curves on the final test (seed 0) ----------------------------------------------
    curve_rows = []
    for system in systems:
        for recipe in selection[system]["curve_recipes"]:
            seed0 = table[(table["system"] == system) & (table["recipe_id"] == recipe["recipe_id"]) & (table["training_seed"] == 0)]
            curve = seed0.groupby("N")["H_MAE_meV"].mean().to_dict()
            dev = finalists[system]["curves"][recipe["recipe_id"]]["E"]
            for n in NS:
                curve_rows.append({"system": system, "recipe_id": recipe["recipe_id"], "role": recipe["role"],
                                   "family": recipe["family"], "dim": recipe["dim"], "R": recipe["R"], "N": n,
                                   "E_dev_meV": dev[str(n)], "E_final_meV": curve.get(n)})
            final_curve = {n: curve[n] for n in NS}
            curve_rows[-1]["N_star_dev"] = finalists[system]["curves"][recipe["recipe_id"]]["N_star"]
            curve_rows[-1]["N_star_final_descriptive"] = n_star(final_curve)
            curve_rows[-1]["gains_final"] = json.dumps({k: round(v, 3) for k, v in gains(final_curve).items()})
    pd.DataFrame(curve_rows).to_csv(OUT / "learning_curves.csv", index=False)

    # ---- paired comparisons (Phase 10) --------------------------------------------------------
    comparisons = []
    for system in systems:
        fins = finalists[system]["finalists"]
        for f in fins:
            if f["N_star_candidate"] != 64:
                comparisons.append({"comparison": f"designed N* vs designed 64 ({f['role']})",
                                    **paired(table, (f["recipe_id"], f["N_star_candidate"]), (f["recipe_id"], 64), system)})
        if system == "w90":
            for n in finalists["w90"]["md_sizes"]:
                for f in fins:
                    if f["N_star_candidate"] == n or n == 64:
                        comparisons.append({"comparison": f"designed ({f['role']}) vs MD, N={n}",
                                            **paired(table, (f["recipe_id"], n), ("md_w90", n), system)})
                        for origin in ("synthetic", "md"):
                            sub = table[(table["system"] == system) & (table["origin"] == origin)]
                            comparisons.append({"comparison": f"designed ({f['role']}) vs MD, N={n}, {origin} test only",
                                                **paired(table, (f["recipe_id"], n), ("md_w90", n), system, sub)})
        # Sobol vs random: per N, family mean over its 3 curve recipes (seed 0), paired per structure.
        seed0 = table[(table["system"] == system) & (table["stage"] == "curve") & (table["training_seed"] == 0)]
        for n in NS:
            fam = seed0[seed0["N"] == n].groupby(["family", "sample_id"])["H_MAE_meV"].mean().unstack(0)
            d = (fam["sobol_sparse"] - fam["random_cartesian"]).to_numpy()
            low, high = bootstrap_mean_ci(d)
            comparisons.append({"comparison": f"Sobol - random (mean of 3 recipes each), N={n}", "system": system,
                                "n_structures": len(d), "E_A": float(fam["sobol_sparse"].mean()),
                                "E_B": float(fam["random_cartesian"].mean()), "mean_d": float(d.mean()),
                                "ci95_low": low, "ci95_high": high, "rel_diff": float(d.mean() / fam["random_cartesian"].mean())})
    comparisons_df = pd.DataFrame([{k: v for k, v in c.items() if not k.startswith("_")} for c in comparisons])
    comparisons_df.to_csv(OUT / "paired_comparisons.csv", index=False)

    # In-domain vs OOD vs MD frames, per model config (synthetic amplitudes relative to R_train).
    regions = (table.groupby(["system", "recipe_id", "family", "N", "region"])["H_MAE_meV"].mean()
               .unstack("region").reset_index())
    regions.to_csv(OUT / "in_domain_vs_ood.csv", index=False)

    # ---- dev: H-MAE vs Gamma spectral error (input for a future non-inferiority margin) --------
    dev_rows = []
    for system in systems:
        for path in (OUT / "runs" / system).glob("*/val_eval/per_structure_metrics.csv"):
            if read_json(path.parents[1] / "result.json")["val"] == "dev":
                dev_rows.append(pd.read_csv(path).assign(system=system))
    dev_all = pd.concat(dev_rows)
    margin_input = {system: {
        "spearman_H_MAE_vs_spectral": float(g["H_MAE_meV"].corr(g["spectral_err_meV"], method="spearman")),
        "spectral_err_meV_per_meV_H_MAE_median": float((g["spectral_err_meV"] / g["H_MAE_meV"]).median()),
    } for system, g in dev_all.groupby("system")}

    # ---- Pareto (Phase 11) ---------------------------------------------------------------------
    pareto_points = []
    for c in configs:
        new_siesta = 0.0  # every training label already existed (curves, LHS, confirmation, MD)
        pareto_points.append({**{k: c[k] for k in ("system", "recipe_id", "family", "N", "H_MAE_meV", "siesta_cpu_h")},
                              "cost_reproducible_h": c["siesta_cpu_h"] + c["gpu_h_mean"],
                              "cost_incremental_h": new_siesta + c["gpu_h_mean"],
                              "E": c["H_MAE_meV"]})
    for cost in ("siesta_cpu_h", "cost_reproducible_h", "cost_incremental_h"):
        front = {(p["system"], p["recipe_id"], p["N"]) for system in systems
                 for p in pareto_front([q for q in pareto_points if q["system"] == system], cost=cost)}
        for point in pareto_points:
            point[f"pareto_{cost}"] = (point["system"], point["recipe_id"], point["N"]) in front
    pareto_df = pd.DataFrame(pareto_points)
    pareto_df.to_csv(OUT / "pareto_points.csv", index=False)

    # ---- Figure 1: learning curves ------------------------------------------------------------
    fig, axes = plt.subplots(1, len(systems), figsize=(5.5 * len(systems), 4.2), squeeze=False)
    for ax, system in zip(axes[0], systems):
        for recipe in selection[system]["curve_recipes"]:
            rows = [r for r in curve_rows if r["system"] == system and r["recipe_id"] == recipe["recipe_id"]]
            ax.plot([r["N"] for r in rows], [r["E_final_meV"] for r in rows], ROLE_STYLE[recipe["role"]],
                    marker=MARKERS[recipe["family"]], ms=5, lw=2, color=COLORS[recipe["family"]],
                    label=f"{LABELS[recipe['family']]} {recipe['role']}: {recipe['dim']} R={recipe['R']}")
        for f in finalists[system]["finalists"]:
            for n in sorted({f["N_star_candidate"], 64}):
                c = by_config.get((system, f["recipe_id"], n))
                if c and c["n_seeds"] == 3:
                    ax.errorbar(n * 1.06, c["H_MAE_meV"], yerr=[[c["H_MAE_meV"] - min(c["seed_values"])],
                                                                [max(c["seed_values"]) - c["H_MAE_meV"]]],
                                fmt="none", ecolor=COLORS.get(f["family"], "#52514e"), capsize=3, lw=1.2)
        lhs = by_config.get((system, selection[system]["lhs_recipe"]["recipe_id"], 64))
        if lhs:
            ax.errorbar(64 * 0.94, lhs["H_MAE_meV"], yerr=[[lhs["H_MAE_meV"] - min(lhs["seed_values"])],
                                                           [max(lhs["seed_values"]) - lhs["H_MAE_meV"]]],
                        fmt="^", color=COLORS["latin_hypercube"], ms=8, capsize=3, label="LHS N=64 (3 seeds)")
        md = [by_config[k] for k in by_config if k[0] == system and k[1] == "md_w90"]
        if md:
            ax.errorbar([c["N"] for c in md], [c["H_MAE_meV"] for c in md],
                        yerr=[[c["H_MAE_meV"] - min(c["seed_values"]) for c in md],
                              [max(c["seed_values"]) - c["H_MAE_meV"] for c in md]],
                        fmt="D", color=COLORS["MD"], ms=7, capsize=3, label="MD (3 seeds)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        ax.tick_params(axis="y", which="minor", labelsize=7)
        ax.set_xticks(NS, [str(n) for n in NS])
        ax.set_xlabel("N estructuras de entrenamiento")
        ax.set_ylabel("H-MAE test final (meV)")
        ax.set_title(f"{system}" + ("  (48 sintéticas + 16 MD)" if system == "w90" else "  (48 sintéticas)"))
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.suptitle("Fig. 1 — Curvas de aprendizaje (test final; semilla 0, barras = rango de 3 semillas)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_learning_curves.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: Pareto ---------------------------------------------------------------------
    fig, axes = plt.subplots(2, len(systems), figsize=(5.5 * len(systems), 8), squeeze=False)
    for col, system in enumerate(systems):
        for row, (cost, label) in enumerate((("siesta_cpu_h", "CPU·h SIESTA de las etiquetas de entrenamiento (reproducible)"),
                                             ("cost_incremental_h", "coste incremental: GPU·h por modelo (SIESTA nuevo = 0)"))):
            ax = axes[row][col]
            pts = pareto_df[pareto_df["system"] == system]
            for family, group in pts.groupby("family"):
                ax.scatter(group[cost], group["E"], s=36, marker=MARKERS[family], color=COLORS[family],
                           edgecolor="#fcfcfb", linewidth=1.5, label=LABELS[family], zorder=3)
            front = pts[pts[f"pareto_{cost}"]].sort_values(cost)
            ax.step(front[cost], front["E"], where="post", color="#0b0b0b", lw=1.2, zorder=2, label="frontera de Pareto")
            for _, p in front.iterrows():
                ax.annotate(f"N={p['N']}", (p[cost], p["E"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
            ax.set_xscale("log")
            ax.set_xlabel(label)
            ax.set_ylabel("H-MAE test final (meV)")
            ax.set_title(system)
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Fig. 2 — Frontera coste–precisión (media de semillas cuando hay 3)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_pareto.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: designed vs MD, paired per structure (w90) ----------------------------------
    md_pairs = [c for c in comparisons if c["comparison"].startswith("designed") and "vs MD" in c["comparison"]
                and "test only" not in c["comparison"]]
    if md_pairs:
        fig, ax = plt.subplots(figsize=(9, 0.9 + 0.8 * len(md_pairs)))
        origin_of = dict(zip(table["sample_id"], table["origin"]))
        for i, c in enumerate(md_pairs):
            is_md = np.array([origin_of[s] == "md" for s in c["_ids"]])
            jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(c["_d"]))
            ax.scatter(c["_d"][~is_md], i + jitter[~is_md], s=14, color="#2a78d6", alpha=0.6,
                       label="estructura sintética" if i == 0 else None)
            ax.scatter(c["_d"][is_md], i + jitter[is_md], s=18, marker="D", color="#52514e", alpha=0.7,
                       label="frame MD" if i == 0 else None)
            ax.errorbar(c["mean_d"], i, xerr=[[c["mean_d"] - c["ci95_seeds_structs_low"]],
                                              [c["ci95_seeds_structs_high"] - c["mean_d"]]],
                        fmt="none", ecolor="#0b0b0b", capsize=3, lw=1)
            ax.errorbar(c["mean_d"], i, xerr=[[c["mean_d"] - c["ci95_low"]], [c["ci95_high"] - c["mean_d"]]],
                        fmt="o", color="#0b0b0b", ms=7, capsize=0, lw=3)
            ax.text(1.02, i, f"Δ = {c['mean_d']:+.1f} meV\nestr. [{c['ci95_low']:+.1f}, {c['ci95_high']:+.1f}]\n"
                    f"sem.+estr. [{c['ci95_seeds_structs_low']:+.1f}, {c['ci95_seeds_structs_high']:+.1f}]",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        ax.axvline(0, color="#8a8984", lw=1)
        ax.set_yticks(range(len(md_pairs)), [c["comparison"].replace("designed ", "diseñado ") for c in md_pairs], fontsize=8)
        ax.set_xlabel("d_i = E_diseñado − E_MD por estructura (meV; media de 3 semillas)")
        ax.set_title("Fig. 3 — Diseñado − MD por estructura (w90); negativo = diseñado mejor\n"
                     "barra gruesa: IC95 sobre estructuras; fina: IC95 sobre semillas y estructuras", fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig3_designed_vs_md.png", bbox_inches="tight")
        plt.close(fig)

    # ---- Figure 4: generalisation heatmap -------------------------------------------------------
    fig, axes = plt.subplots(1, len(systems), figsize=(6 * len(systems), 4.6), squeeze=False)
    for ax, system in zip(axes[0], systems):
        keep = [(r["recipe_id"], 64) for r in selection[system]["curve_recipes"]]
        keep.append((selection[system]["lhs_recipe"]["recipe_id"], 64))
        if system == "w90":
            keep.append(("md_w90", 64))
        sub = table[(table["system"] == system) & (table["origin"] == "synthetic")]
        rows, labels = [], []
        for recipe, n in sorted(keep, key=lambda k: float(sub[sub["recipe_id"] == k[0]]["R_train"].iloc[0])):
            cell = sub[(sub["recipe_id"] == recipe) & (sub["N"] == n)].groupby("A")["H_MAE_meV"].mean()
            rows.append([cell.get(a, np.nan) for a in FINAL_AMPLITUDES])
            first = sub[sub["recipe_id"] == recipe].iloc[0]
            domain = first["train_max_real_amplitude_ang"] if first["family"] == "MD" else first["R_train"]
            labels.append(f"{LABELS[first['family']]} {first['dim']} R≤{domain:.3f}")
        grid = np.array(rows)
        image = ax.imshow(grid, cmap="Blues", aspect="auto")
        for (i, j), value in np.ndenumerate(grid):
            ax.text(j, i, f"{value:.0f}" if value >= 10 else f"{value:.1f}", ha="center", va="center", fontsize=7,
                    color="#fcfcfb" if value > np.nanpercentile(grid, 60) else "#0b0b0b")
        ax.set_xticks(range(len(FINAL_AMPLITUDES)), [f"{a:.2f}" for a in FINAL_AMPLITUDES])
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("amplitud del test A (Å)")
        ax.set_title(f"{system} (N=64; media de semillas disponibles)")
        fig.colorbar(image, ax=ax, label="H-MAE (meV)")
    fig.suptitle("Fig. 4 — Generalización: dominio de entrenamiento × amplitud del test")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_generalization_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 5: seed stability of the finalists ----------------------------------------------
    fig, axes = plt.subplots(1, len(systems), figsize=(5.5 * len(systems), 4), squeeze=False)
    for ax, system in zip(axes[0], systems):
        entries = []
        for f in finalists[system]["finalists"]:
            for n in sorted({f["N_star_candidate"], 64}):
                entries.append((f"{f['role']}\n{LABELS[f['family']]} N={n}", f["family"], by_config[(system, f["recipe_id"], n)]))
        if system == "w90":
            for n in finalists["w90"]["md_sizes"]:
                entries.append((f"MD\nN={n}", "MD", by_config[("w90", "md_w90", n)]))
        for i, (label, family, c) in enumerate(entries):
            ax.scatter([i] * len(c["seed_values"]), c["seed_values"], color=COLORS[family], marker=MARKERS[family], s=30, zorder=3)
            ax.errorbar(i + 0.2, c["H_MAE_meV"], yerr=[[c["H_MAE_meV"] - c["ci95"][0]], [c["ci95"][1] - c["H_MAE_meV"]]],
                        fmt="_", color="#0b0b0b", ms=12, capsize=3)
        ax.set_xticks(range(len(entries)), [e[0] for e in entries], fontsize=7)
        ax.set_ylabel("H-MAE test final (meV)")
        ax.set_title(system)
    fig.suptitle("Fig. 5 — Estabilidad entre semillas (puntos = semillas; barra = media e IC95 sobre estructuras)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_seed_stability.png", bbox_inches="tight")
    plt.close(fig)

    # ---- final table ---------------------------------------------------------------------------
    md_diff = {c["comparison"]: c for c in comparisons}
    table_rows = []
    for system in systems:
        wanted = []
        for f in finalists[system]["finalists"]:
            wanted += [(f["recipe_id"], n, f["role"]) for n in sorted({f["N_star_candidate"], 64})]
        wanted.append((selection[system]["lhs_recipe"]["recipe_id"], 64, "LHS"))
        if system == "w90":
            wanted += [("md_w90", n, "MD baseline") for n in finalists["w90"]["md_sizes"]]
        for recipe, n, role in dict.fromkeys(wanted):
            c = by_config[(system, recipe, n)]
            diff = next((v for k, v in md_diff.items() if system == "w90" and f"N={n}" in k and "test only" not in k
                         and "vs MD" in k and v["A"] == f"{recipe}@N{n}"), None)
            table_rows.append({
                "Sistema": system, "Dataset": f"{role}: {recipe}", "Familia": LABELS[c["family"]], "N": n,
                "semillas": c["n_seeds"], "H-MAE (meV)": round(c["H_MAE_meV"], 2),
                "IC95": f"[{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]", "sd semillas": round(c["seed_sd"], 2),
                "Rel. Frob": round(c["rel_Frob"], 4), "CPU·h SIESTA": round(c["siesta_cpu_h"], 3),
                "GPU·h (por modelo)": round(c["gpu_h_mean"], 3),
                "Bandas RMSE (meV)": None if c["band_rmse_meV"] is None else round(c["band_rmse_meV"], 1),
                "Diferencia vs MD (meV, IC95)": "" if diff is None else
                f"{diff['mean_d']:+.2f} [{diff['ci95_low']:+.2f}, {diff['ci95_high']:+.2f}]",
            })
    final_table = pd.DataFrame(table_rows)
    final_table.to_csv(OUT / "final_table.csv", index=False)
    (OUT / "final_table.md").write_text(markdown_table(final_table), encoding="utf-8")
    write_json(OUT / "analysis_summary.json", {
        "configs": configs, "comparisons": [{k: v for k, v in c.items() if not k.startswith("_")} for c in comparisons],
        "margin_input_dev": margin_input, "final_test": {s: b["n"] for s, b in read_json(OUT / "final_test_manifest.json")["systems"].items()},
        "md_label_equivalence_check": read_json(OUT / "final_test_manifest.json")["systems"]["w90"]["md_label_equivalence_check"],
    })
    print(markdown_table(final_table))
    print(comparisons_df.to_string())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory")
    p = sub.add_parser("dev6x6")
    p.add_argument("--workers", type=int, default=5)
    p = sub.add_parser("reeval")
    p.add_argument("--parallel", type=int, default=2)
    p = sub.add_parser("select")
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    sub.add_parser("md-split")
    p = sub.add_parser("train")
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    p.add_argument("--parallel", type=int, default=None)
    p = sub.add_parser("finalists")
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    sub.add_parser("confirm")
    p = sub.add_parser("final-test")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    p = sub.add_parser("evaluate-final")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    sub.add_parser("analyze")
    sub.add_parser("rescore-val")
    args = parser.parse_args()
    return {"inventory": cmd_inventory, "dev6x6": cmd_dev6x6, "reeval": cmd_reeval, "select": cmd_select,
            "md-split": cmd_md_split, "train": cmd_train, "finalists": cmd_finalists,
            "confirm": cmd_confirm, "final-test": cmd_final_test,
            "evaluate-final": cmd_evaluate_final, "analyze": cmd_analyze,
            "rescore-val": cmd_rescore_val}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
