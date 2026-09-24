#!/usr/bin/env python3
"""Matched 6x6 sweep of the number of displaced atoms; single-purpose and resumable."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dataset_design_curves_v1 as c  # noqa: E402
import dataset_design_w90_001_s9_s3_design_generation as s3  # noqa: E402
import run_6x6_method_amplitude_crosstest as cross  # noqa: E402
import run_6x6_md_label_budget as thermal  # noqa: E402
import run_hamiltonian_derivative_siesta_references as siesta_refs  # noqa: E402
import w90_displacement_sampler_family as sampler  # noqa: E402


ROOT = c.OUT / "active_atom_sweep_6x6"
WORK = c.OUT / "active_atom_sweep_6x6_work"
RUNS = c.OUT / "runs/6x6"
REPORT = REPO_ROOT / "docs/dataset_design_active_atom_sweep_6x6.md"
K_VALUES = (2, 12, 22, 32, 42, 52, 62, 72)
TRAIN_SEEDS = (0, 1)
N_TRAIN, N_VAL, N_TEST = 64, 48, 32
AMPLITUDE = 0.12
SAMPLER_SEED = 0
MAX_TRAININGS = 3
SIESTA_WORKERS = 6
THERMAL_PAUSE_C, THERMAL_LIMIT_C, THERMAL_RESUME_C = 78.0, 80.0, 70.0
ENV = os.environ | {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


def log(message: str, **fields: Any) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "message": message, **fields}
    line = json.dumps(row, sort_keys=True, default=str)
    print(line, flush=True)
    with (ROOT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def reconcile_temperature() -> float | None:
    return thermal.reconcile_temperature(
        pause_c=THERMAL_PAUSE_C, limit_c=THERMAL_LIMIT_C, resume_c=THERMAL_RESUME_C,
    )


def wait_for_thermal_headroom() -> None:
    while (temperature := reconcile_temperature()) is not None and temperature >= THERMAL_PAUSE_C:
        time.sleep(10)


def thermal_run_siesta(reference_dir: Path, *, command: str, use_shell: bool = False) -> dict[str, Any]:
    wait_for_thermal_headroom()
    run_fdf, run_out = reference_dir / "RUN.fdf", reference_dir / "RUN.out"
    command_args: str | list[str] = command if use_shell else shlex.split(command)
    with run_fdf.open("r", encoding="utf-8") as stdin, run_out.open("w", encoding="utf-8") as stdout:
        process = subprocess.Popen(command_args, cwd=reference_dir, stdin=stdin, stdout=stdout,
                                   stderr=subprocess.STDOUT, shell=use_shell, text=True,
                                   env=ENV, start_new_session=True)
        item = {"process": process, "name": reference_dir.name, "started": time.monotonic(), "paused": False}
        with thermal._ACTIVE_LOCK:
            thermal._ACTIVE[process.pid] = item
        try:
            while process.poll() is None:
                reconcile_temperature()
                time.sleep(10)
        finally:
            with thermal._ACTIVE_LOCK:
                thermal._ACTIVE.pop(process.pid, None)
    return {"command": command if use_shell else command_args, "shell": use_shell,
            "returncode": process.returncode, "stdout_path": str(run_out)}


def thermal_run_training(config_path: Path, run_dir: Path) -> tuple[Path, float, int]:
    wait_for_thermal_headroom()
    env = ENV | {
        "PYTHONPATH": os.pathsep.join(filter(None, [str(c.s4.TORCH_COMPAT_DIR), ENV.get("PYTHONPATH", "")]))
    }
    peak, start = 0, time.perf_counter()
    with (run_dir / "train.log").open("w", encoding="utf-8") as handle:
        process = subprocess.Popen([str(c.s4.GRAPH2MAT_BIN), "models", "mace", "main", "fit", "-c", config_path.name],
                                   cwd=run_dir, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)
        item = {"process": process, "name": run_dir.name, "started": time.monotonic(), "paused": False}
        with thermal._ACTIVE_LOCK:
            thermal._ACTIVE[process.pid] = item
        try:
            while process.poll() is None:
                peak = max(peak, c.gpu_mib(process.pid))
                reconcile_temperature()
                time.sleep(10)
        finally:
            with thermal._ACTIVE_LOCK:
                thermal._ACTIVE.pop(process.pid, None)
    seconds = time.perf_counter() - start
    if process.returncode:
        raise RuntimeError(f"Graph2Mat training failed (rc={process.returncode}); see {run_dir / 'train.log'}")
    checkpoints = sorted(run_dir.rglob("best-*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected exactly one best-val_loss checkpoint under {run_dir}, found {checkpoints}")
    return checkpoints[0], seconds, peak


def thermal_run_prediction(checkpoint: Path, manifest_path: Path, output_dir: Path,
                           accelerator: str) -> Path:
    wait_for_thermal_headroom()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(c.s4.PYTHON), str(c.s4.SCRIPT_DIR / "predict_model_on_dataset.py"),
        "--checkpoint", str(checkpoint), "--train-method", "md",
        "--test-set", "w90_s4_validation", "--test-manifest", str(manifest_path),
        "--output-dir", str(output_dir), "--basis-files", c.s4.DEFAULT_BASIS_GLOB,
        "--matrix-component-policy", "h_only", "--n-matrix-components", "1",
        "--accelerator", accelerator, "--no-store-in-memory",
    ]
    with (output_dir / "predict.log").open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   text=True, env=ENV, start_new_session=True)
        item = {"process": process, "name": output_dir.name, "started": time.monotonic(), "paused": False}
        with thermal._ACTIVE_LOCK:
            thermal._ACTIVE[process.pid] = item
        try:
            while process.poll() is None:
                reconcile_temperature()
                time.sleep(10)
        finally:
            with thermal._ACTIVE_LOCK:
                thermal._ACTIVE.pop(process.pid, None)
    if process.returncode:
        raise RuntimeError(f"predict_model_on_dataset.py failed (rc={process.returncode}); see {output_dir / 'predict.log'}")
    return output_dir / "predicted_hamiltonians"


def paired_configs() -> dict[int, list[sampler.Configuration]]:
    """Use one 216-D Sobol design and mask atoms, keeping shared motions identical."""

    geometry = sampler.load_graphene_6x6()
    full = s3.generate_design(geometry, "sobol_sparse", "3D", 72, AMPLITUDE, N_TRAIN, SAMPLER_SEED)
    output: dict[int, list[sampler.Configuration]] = {}
    for k in K_VALUES:
        active = sampler.select_active_atoms(geometry, k)
        output[k] = []
        for index, parent in enumerate(full):
            displacements = {atom: parent.displacements_ang[atom] for atom in active.indices}
            metadata = sampler.build_metadata(
                geometry, active, displacements, family="sobol_sparse", dim="3D", amplitude_ang=AMPLITUDE,
                extra={"paired_k72_index": index, "active_selection": "nested_periodic_bond_bfs"},
            )
            output[k].append(sampler.Configuration("sobol_sparse", displacements, metadata))
    return output


def positions_and_hash(config: sampler.Configuration) -> tuple[np.ndarray, str]:
    positions = c.base_positions("6x6").copy()
    for atom, displacement in config.displacements_ang.items():
        positions[atom] += displacement
    return positions, c.s5.geometry_hash(positions)


def validation_rows() -> list[dict[str, Any]]:
    rows = c.read_json(c.OUT / "common_dev_6x6_v1.json")["samples"]
    if len(rows) != N_VAL:
        raise RuntimeError(f"expected Validation{N_VAL}, found {len(rows)}")
    return rows


def test_rows() -> list[dict[str, Any]]:
    manifest = c.read_json(cross.ROOT / "manifest.json")
    rows = manifest.get("test", {}).get("sobol_sparse__R0.12", [])
    if manifest.get("datasets_frozen") is not True or len(rows) != N_TEST:
        raise RuntimeError("the frozen Sobol R=0.12 Test32 from the method/amplitude campaign is unavailable")
    return rows


def self_test() -> dict[str, Any]:
    geometry, pools = sampler.load_graphene_6x6(), paired_configs()
    selections = [sampler.select_active_atoms(geometry, k) for k in K_VALUES]
    assert all(set(left.indices) < set(right.indices) for left, right in zip(selections, selections[1:]))
    assert all(selection.connected for selection in selections)
    hashes: dict[int, list[str]] = {}
    for k, configs in pools.items():
        assert len(configs) == N_TRAIN
        assert all(len(config.displacements_ang) == k for config in configs)
        hashes[k] = [positions_and_hash(config)[1] for config in configs]
        assert len(set(hashes[k])) == N_TRAIN
    for left, right in zip(K_VALUES, K_VALUES[1:]):
        shared = sampler.select_active_atoms(geometry, left).indices
        for low, high in zip(pools[left], pools[right]):
            assert all(np.allclose(low.displacements_ang[atom], high.displacements_ang[atom], atol=0)
                       for atom in shared)
    existing = c.read_json(c.OUT / "coverage_pool_6x6_r012.json")["samples"]
    assert hashes[72] == [row["hash"] for row in existing]
    validation, test = validation_rows(), test_rows()
    train_hashes = {value for values in hashes.values() for value in values}
    assert len(train_hashes) == len(K_VALUES) * N_TRAIN
    assert not train_hashes & {row["hash"] for row in validation}
    assert not train_hashes & {row["hash"] for row in test}
    plan = {"k_values": K_VALUES, "new_labels": (len(K_VALUES) - 1) * N_TRAIN,
            "new_models": (len(K_VALUES) - 1) * len(TRAIN_SEEDS), "reused_models": 2,
            "validation": len(validation), "test": len(test)}
    print(json.dumps(plan, sort_keys=True), flush=True)
    return plan


def _new_rows(configs: dict[int, list[sampler.Configuration]]) -> tuple[list[tuple[str, sampler.Configuration]], dict[int, list[str]]]:
    entries, ids = [], {}
    for k in K_VALUES[:-1]:
        ids[k] = []
        for index, config in enumerate(configs[k]):
            sample_id = f"k6x6__train__k{k:03d}__{index:03d}"
            entries.append((sample_id, config))
            ids[k].append(sample_id)
    return entries, ids


def _manifest_is_frozen(manifest: dict[str, Any]) -> bool:
    return (manifest.get("datasets_frozen") is True
            and set(map(int, manifest.get("train", {}))) == set(K_VALUES)
            and all(len(rows) == N_TRAIN for rows in manifest["train"].values())
            and len(manifest.get("validation", [])) == N_VAL
            and len(manifest.get("test", [])) == N_TEST)


def prepare_datasets() -> dict[str, Any]:
    thermal.log = log
    path = ROOT / "manifest.json"
    if path.exists() and _manifest_is_frozen(manifest := c.read_json(path)):
        return manifest
    plan, configs = self_test(), paired_configs()
    entries, ids = _new_rows(configs)
    preliminary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "labels_pending",
        "k_values": K_VALUES, "family": "sobol_sparse", "dim": "3D", "amplitude_ang": AMPLITUDE,
        "N_train": N_TRAIN, "N_val": N_VAL, "N_test": N_TEST, "training_seeds": TRAIN_SEEDS,
        "active_selection": "nested periodic bond-graph breadth-first prefixes",
        "paired_design": "one k=72 Sobol design; inactive atoms masked for each smaller k",
        "primary_test": "fixed Sobol 3D R=0.12 Test32 with k=72",
        "training": {"loss": "graph2mat.metrics.elementwise_mse", "batch_size": 16,
                     "optimizer_updates": 8000, "validation_evaluations": 2000,
                     "scheduler": "cosine", "checkpoint": "best val_loss"},
        "audit": plan, "siesta_workers": SIESTA_WORKERS,
    }
    c.write_json(path, preliminary)
    siesta_refs.run_siesta = thermal_run_siesta
    references = c.label_structures(entries, WORK, cross.CANONICAL, workers=SIESTA_WORKERS)
    train: dict[str, list[dict[str, Any]]] = {}
    for k in K_VALUES[:-1]:
        train[str(k)] = [c.labelled_sample(
            "6x6", sample_id, references,
            {"family": "sobol_sparse", "dim": "3D", "k": k, "amplitude_ang": AMPLITUDE,
             "paired_k72_index": index, "active_selection": "nested_periodic_bond_bfs"},
        ) for index, sample_id in enumerate(ids[k])]
    train["72"] = [dict(row) | {"paired_k72_index": index, "active_selection": "all_atoms"}
                   for index, row in enumerate(c.read_json(c.OUT / "coverage_pool_6x6_r012.json")["samples"])]
    validation, test = validation_rows(), test_rows()
    groups = [rows for rows in train.values()] + [validation, test]
    hash_sets = [{row["hash"] for row in rows} for rows in groups]
    if any(hash_sets[i] & hash_sets[j] for i in range(len(hash_sets)) for j in range(i + 1, len(hash_sets))):
        raise RuntimeError("train/validation/test geometry overlap")
    physics = cross.fdf_physics_hash(cross.REFERENCE_FDF)
    for row in [item for rows in groups for item in rows]:
        if cross.fdf_physics_hash(Path(row["run_fdf"])) != physics:
            raise RuntimeError(f"SIESTA settings differ for {row['sample_id']}")
    manifest = preliminary | {
        "status": "datasets_frozen", "datasets_frozen": True,
        "train": train, "validation": validation, "test": test,
        "hashes": {"train": {k: c.ids_hash([row["hash"] for row in rows]) for k, rows in train.items()},
                   "validation": c.ids_hash([row["hash"] for row in validation]),
                   "test": c.ids_hash([row["hash"] for row in test])},
    }
    c.write_json(path, manifest)
    log("datasets frozen", train=sum(map(len, train.values())), validation=N_VAL, test=N_TEST)
    return manifest


def matrix_metrics(sample: c.s4.Sample, predicted_root: Path) -> dict[str, float]:
    predicted = predicted_root / sample.sample_id / "ML_prediction.HSX"
    h_pred, h_ref = c.s4.gamma_hk(predicted), c.s4.gamma_hk(sample.reference_matrix)
    error = h_pred - h_ref
    return {"H_MAE_meV": 1e3 * float(np.mean(np.abs(error))),
            "rel_Frob": float(np.linalg.norm(error) / np.linalg.norm(h_ref))}


def evaluate_samples(checkpoint: Path, rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    thermal.log = log
    c.s4.run_prediction = thermal_run_prediction
    samples = [c.sample_from_dict(row) for row in rows]
    predicted_root = c.predict(checkpoint, samples, out_dir, "gpu")
    output = [{"sample_id": sample.sample_id, **matrix_metrics(sample, predicted_root)} for sample in samples]
    shutil.rmtree(predicted_root, ignore_errors=True)
    return output


def existing_k72(seed: int) -> dict[str, Any]:
    row = cross.existing_model("sobol_sparse", AMPLITUDE, seed)
    if row is None:
        raise RuntimeError(f"missing exact current-protocol k=72 checkpoint for seed {seed}")
    return row | {"model_id": f"k6x6__k072__seed{seed}", "k": 72,
                  "gpu_h": float(row["train_seconds"]) / 3600 if row.get("train_seconds") else None}


def train_one(k: int, seed: int, manifest: dict[str, Any]) -> dict[str, Any]:
    model_id = f"k6x6__k{k:03d}__seed{seed}"
    run_dir, result_path, done_path = RUNS / model_id, RUNS / model_id / "result.json", RUNS / model_id / "train_done.json"
    if result_path.exists():
        saved = c.read_json(result_path)
        checkpoint = Path(saved["checkpoint"])
        if checkpoint.exists() and c.sha256_file(checkpoint) == saved["checkpoint_sha256"]:
            return saved
    if done_path.exists():
        done = c.read_json(done_path)
        checkpoint, seconds, peak = Path(done["checkpoint"]), float(done["seconds"]), int(done["peak_gpu_mib"])
        if not checkpoint.exists() or c.sha256_file(checkpoint) != done["checkpoint_sha256"]:
            raise RuntimeError(f"invalid completion marker: {done_path}")
    else:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        train = [c.sample_from_dict(row) for row in manifest["train"][str(k)]]
        validation = [c.sample_from_dict(row) for row in manifest["validation"]]
        config = c.s4.build_graph2mat_config(run_dir, train, validation, run_name=model_id,
                                             accelerator="gpu", training_seed=seed, **c.TRAIN_KW)
        c.apply_config_patch(config, c.fixed_update_elemmse_patch("6x6", N_TRAIN))
        checkpoint, seconds, peak = thermal_run_training(config, run_dir)
        c.write_json(done_path, {"checkpoint": str(checkpoint), "checkpoint_sha256": c.sha256_file(checkpoint),
                                 "seconds": seconds, "peak_gpu_mib": peak})
    validation_metrics = evaluate_samples(checkpoint, manifest["validation"], run_dir / "validation_eval")
    step = re.search(r"step=?([0-9]+)", checkpoint.name)
    result = {
        "model_id": model_id, "k": k, "training_seed": seed, "source": "new",
        "checkpoint": str(checkpoint), "checkpoint_sha256": c.sha256_file(checkpoint),
        "best_update": int(step.group(1)) if step else None, "train_seconds": seconds,
        "gpu_h": seconds / 3600, "peak_gpu_mib": peak,
        "val_H_MAE_meV": float(np.mean([row["H_MAE_meV"] for row in validation_metrics])),
        "val_rel_Frob": float(np.mean([row["rel_Frob"] for row in validation_metrics])),
        "train_ids_sha256": c.ids_hash([row["sample_id"] for row in manifest["train"][str(k)]]),
        "validation_ids_sha256": c.ids_hash([row["sample_id"] for row in manifest["validation"]]),
        "result_path": str(result_path),
    }
    c.write_json(result_path, result)
    return result


def train_models(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    output = [existing_k72(seed) for seed in TRAIN_SEEDS]
    jobs = [(k, seed) for k in K_VALUES[:-1] for seed in TRAIN_SEEDS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_TRAININGS) as pool:
        futures = {pool.submit(train_one, k, seed, manifest): (k, seed) for k, seed in jobs}
        for future in concurrent.futures.as_completed(futures):
            k, seed = futures[future]
            output.append(future.result())
            output.sort(key=lambda row: (int(row["k"]), int(row["training_seed"])))
            c.write_csv(ROOT / "model_results.csv", output)
            log("training completed", k=k, seed=seed, completed=len(output), total=len(K_VALUES) * 2)
    if len(output) != len(K_VALUES) * len(TRAIN_SEEDS):
        raise RuntimeError(f"expected 16 models, found {len(output)}")
    return output


def _reused_k72_metrics(seed: int, expected_ids: set[str]) -> list[dict[str, Any]] | None:
    rows = [row for row in c.read_csv(cross.ROOT / "per_structure_results.csv")
            if row.get("train_method") == "sobol_sparse" and float(row.get("train_R", -1)) == AMPLITUDE
            and int(row.get("training_seed", -1)) == seed and row.get("test_method") == "sobol_sparse"
            and float(row.get("test_R", -1)) == AMPLITUDE]
    return rows if len(rows) == N_TEST and {row["sample_id"] for row in rows} == expected_ids else None


def evaluate_models(manifest: dict[str, Any], models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = ROOT / "per_structure_results.csv"
    stored = c.read_csv(path) if path.exists() else []
    expected_ids = {row["sample_id"] for row in manifest["test"]}
    for model in models:
        previous = [row for row in stored if row["model_id"] == model["model_id"]]
        if len(previous) == N_TEST and {row["sample_id"] for row in previous} == expected_ids:
            continue
        stored = [row for row in stored if row["model_id"] != model["model_id"]]
        metrics = _reused_k72_metrics(int(model["training_seed"]), expected_ids) if int(model["k"]) == 72 else None
        if metrics is None:
            metrics = evaluate_samples(Path(model["checkpoint"]), manifest["test"],
                                       Path(model["result_path"]).parent / "active_atom_test")
        stored += [{"model_id": model["model_id"], "k": model["k"],
                    "training_seed": model["training_seed"], "sample_id": row["sample_id"],
                    "H_MAE_meV": row["H_MAE_meV"], "rel_Frob": row["rel_Frob"]} for row in metrics]
        stored.sort(key=lambda row: (int(row["k"]), int(row["training_seed"]), row["sample_id"]))
        c.write_csv(path, stored)
        log("model evaluated", model_id=model["model_id"], rows=len(stored))
    if len(stored) != len(K_VALUES) * len(TRAIN_SEEDS) * N_TEST:
        raise RuntimeError(f"expected 512 per-structure rows, found {len(stored)}")
    return stored


def analyze(manifest: dict[str, Any], models: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    summaries = []
    for k in K_VALUES:
        seed_values = []
        for seed in TRAIN_SEEDS:
            selected = [row for row in rows if int(row["k"]) == k and int(row["training_seed"]) == seed]
            if len(selected) != N_TEST:
                raise RuntimeError(f"incomplete Test32 for k={k}, seed={seed}")
            seed_values.append({"seed": seed,
                                "H_MAE_meV": float(np.mean([float(row["H_MAE_meV"]) for row in selected])),
                                "rel_Frob": float(np.mean([float(row["rel_Frob"]) for row in selected]))})
        train_cost = sum(float(row.get("siesta_cpu_s") or 0) for row in manifest["train"][str(k)]) / 3600
        summaries.append({
            "k": k, "k_over_72": k / 72, "n_seeds": 2,
            "H_MAE_seed0_meV": seed_values[0]["H_MAE_meV"], "H_MAE_seed1_meV": seed_values[1]["H_MAE_meV"],
            "H_MAE_meV": float(np.mean([row["H_MAE_meV"] for row in seed_values])),
            "H_MAE_sd_meV": float(np.std([row["H_MAE_meV"] for row in seed_values], ddof=1)),
            "rel_Frob_seed0": seed_values[0]["rel_Frob"], "rel_Frob_seed1": seed_values[1]["rel_Frob"],
            "rel_Frob": float(np.mean([row["rel_Frob"] for row in seed_values])),
            "rel_Frob_sd": float(np.std([row["rel_Frob"] for row in seed_values], ddof=1)),
            "siesta_cpu_h_train": train_cost,
        })
    reference = next(row for row in summaries if row["k"] == 72)
    for row in summaries:
        row["H_ratio_vs_k72"] = row["H_MAE_meV"] / reference["H_MAE_meV"]
        row["Frob_ratio_vs_k72"] = row["rel_Frob"] / reference["rel_Frob"]
        row["within_5pct_H_and_10pct_Frob"] = (row["H_ratio_vs_k72"] <= 1.05
                                                and row["Frob_ratio_vs_k72"] <= 1.10)
    minimum = next((row for row in summaries if row["within_5pct_H_and_10pct_Frob"]), None)
    c.write_csv(ROOT / "summary.csv", summaries)
    table = [
        "# Influencia del número de átomos desplazados en grafeno 6×6", "",
        "## Diseño", "",
        "Sobol 3D, R=0.12 Å, Train64, Validation48 común, Test32 fijo con k=72 y dos semillas. "
        "Los conjuntos activos son prefijos conectados y anidados; los desplazamientos de los átomos compartidos "
        "son idénticos entre valores de k. Todos los modelos usan `elementwise_mse`, batch 16, 8000 actualizaciones, "
        "2000 evaluaciones de validación, scheduler coseno y el mejor checkpoint por `val_loss`.", "",
        "El test común k=72 evita que los k pequeños parezcan mejores únicamente por contener más átomos en equilibrio.", "",
        "## Resultados", "",
        "| k | k/72 | H-MAE seed 0 | H-MAE seed 1 | H-MAE media ± sd (meV) | Frobenius media ± sd (%) | vs k=72 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table += [f"| {row['k']} | {row['k_over_72']:.3f} | {row['H_MAE_seed0_meV']:.3f} | "
              f"{row['H_MAE_seed1_meV']:.3f} | {row['H_MAE_meV']:.3f} ± {row['H_MAE_sd_meV']:.3f} | "
              f"{100 * row['rel_Frob']:.3f} ± {100 * row['rel_Frob_sd']:.3f} | {row['H_ratio_vs_k72']:.3f}× |"
              for row in summaries]
    table += ["", (f"Mínimo exploratorio dentro de 5% en H-MAE y 10% en Frobenius respecto a k=72: **k={minimum['k']}**."
                       if minimum else "Ningún k<72 satisface conjuntamente los márgenes preregistrados."), "",
              "Con dos semillas, la dispersión es descriptiva y no constituye evidencia inferencial fuerte."]
    REPORT.write_text("\n".join(table) + "\n", encoding="utf-8")
    manifest["status"] = "complete"
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["summary"] = {"reference_k": 72, "minimum_k": minimum["k"] if minimum else None,
                           "new_siesta_cpu_h": sum(row["siesta_cpu_h_train"] for row in summaries if row["k"] != 72)}
    c.write_json(ROOT / "manifest.json", manifest)
    log("campaign complete", **manifest["summary"])


def run_all() -> None:
    manifest = prepare_datasets()
    models = train_models(manifest)
    rows = evaluate_models(manifest, models)
    analyze(manifest, models, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("self-test", "prepare", "train", "evaluate", "analyze", "all"),
                        default="all", nargs="?")
    args = parser.parse_args()
    if args.phase == "self-test":
        self_test()
    elif args.phase == "prepare":
        prepare_datasets()
    elif args.phase == "train":
        train_models(prepare_datasets())
    elif args.phase == "evaluate":
        manifest = prepare_datasets()
        evaluate_models(manifest, train_models(manifest))
    elif args.phase == "analyze":
        manifest = c.read_json(ROOT / "manifest.json")
        analyze(manifest, c.read_csv(ROOT / "model_results.csv"), c.read_csv(ROOT / "per_structure_results.csv"))
    else:
        run_all()


if __name__ == "__main__":
    main()
