#!/usr/bin/env python3
"""Matched 6x6 method/amplitude cross-test; deliberately single-purpose and resumable."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dataset_design_curves_v1 as c  # noqa: E402
import dataset_design_w90_001_s9_s3_design_generation as s3  # noqa: E402
import run_6x6_md_label_budget as thermal  # noqa: E402
import run_hamiltonian_derivative_siesta_references as siesta_refs  # noqa: E402
import run_w90_displacement_siesta_pilot as pilot  # noqa: E402
import w90_displacement_sampler_family as sampler  # noqa: E402
from siesta_output_status import parse_siesta_output  # noqa: E402


ROOT = c.OUT / "method_amplitude_crosstest_6x6"
WORK = c.OUT / "method_amplitude_crosstest_6x6_work"
RUNS = c.OUT / "runs/6x6"
CANONICAL = c.OUT / "coverage_6x6_r012/_canonical_base.fdf"
REFERENCE_FDF = c.OUT / "coverage_6x6_r012/siesta_hamiltonians/coverage_6x6_3D_R012__pt0000/RUN.fdf"
METHODS = ("sobol_sparse", "latin_hypercube", "random_cartesian", "angular_shell_collective")
AMPLITUDES = (0.03, 0.08, 0.12)
TRAIN_SEEDS = (0, 1)
SAMPLER_TRAIN_SEED = 0
SAMPLER_TEST_SEED = 999
N_TRAIN, N_VAL, N_TEST = 64, 48, 32
SIERRA_WORKERS = 2  # Conservative by design; a third worker is optional, not required.
VRAM_PER_MODEL_MIB = 8600
GPU_RESERVE_MIB = 4096
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


def domain(method: str, amplitude: float) -> str:
    return f"{method}__R{amplitude:.2f}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fdf_physics_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(
        r"(?ims)^\s*%block\s+AtomicCoordinatesAndAtomicSpecies\s*$.*?^\s*%endblock\s+AtomicCoordinatesAndAtomicSpecies\s*$",
        "%block AtomicCoordinatesAndAtomicSpecies\n<geometry>\n%endblock AtomicCoordinatesAndAtomicSpecies",
        text,
    )
    return hashlib.sha256(text.encode()).hexdigest()


def configs(method: str, amplitude: float, count: int, seed: int) -> list[sampler.Configuration]:
    geometry = sampler.load_graphene_6x6()
    if method == "angular_shell_collective":
        return sampler.generate_angular_shell_collective(geometry, amplitude, count, seed=seed)
    return s3.generate_design(geometry, method, "3D", 72, amplitude, count, seed)


def positions_and_hash(config: sampler.Configuration) -> tuple[np.ndarray, str, float]:
    positions = c.base_positions("6x6").copy()
    for atom, displacement in config.displacements_ang.items():
        positions[atom] += displacement
    displacement = positions - c.base_positions("6x6")
    return positions, c.s5.geometry_hash(positions), float(np.linalg.norm(displacement, axis=1).max())


def self_test() -> None:
    geometry = sampler.load_graphene_6x6()
    first = sampler.generate_angular_shell_collective(geometry, 0.03, 4, seed=17)
    same = sampler.generate_angular_shell_collective(geometry, 0.03, 4, seed=17)
    other = sampler.generate_angular_shell_collective(geometry, 0.03, 4, seed=18)
    scaled = sampler.generate_angular_shell_collective(geometry, 0.12, 4, seed=17)
    assert first == same and first != other
    fingerprints = set()
    for small, large in zip(first, scaled):
        a = np.asarray([small.displacements_ang[index] for index in range(72)])
        b = np.asarray([large.displacements_ang[index] for index in range(72)])
        assert np.allclose(np.linalg.norm(a, axis=1), 0.03, rtol=0, atol=1e-14)
        assert np.count_nonzero(np.linalg.norm(a, axis=1) > 0) == 72
        assert np.linalg.norm(a.sum(axis=0)) < 1e-13
        assert np.allclose(a / 0.03, b / 0.12, atol=1e-14)
        fingerprints.add(a.tobytes())
    assert len(fingerprints) == len(first)
    assert CANONICAL.exists() and geometry.n_atoms == 72
    log("angular_shell_collective self-test passed", structures=len(first))


def valid_reference(reference_dir: Path) -> bool:
    matrix = reference_dir / "graphene_6x6.TSHS"
    run_fdf, run_out = reference_dir / "RUN.fdf", reference_dir / "RUN.out"
    if not (matrix.exists() and run_fdf.exists() and run_out.exists()):
        return False
    status = parse_siesta_output(run_out, run_fdf)
    return bool(status["valid"] and status["job_completed"] and not re.search(r"\bnan\b", run_out.read_text(errors="ignore"), re.I))


def existing_label_index() -> dict[str, Path]:
    roots = [
        REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_6x6_campaign/siesta_hamiltonians",
        c.OUT / "common_dev_6x6_v1_new/siesta_hamiltonians",
        WORK / "siesta_hamiltonians",
    ]
    indexed: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for reference_dir in root.iterdir():
            name = reference_dir.name
            relevant = (root == WORK / "siesta_hamiltonians" or
                        ("__3D__" in name and "__seed999__" in name))
            if not relevant or not reference_dir.is_dir() or not valid_reference(reference_dir):
                continue
            geometry_hash = c.geometry_record("6x6", reference_dir / "RUN.fdf")["hash"]
            indexed.setdefault(geometry_hash, reference_dir)
    return indexed


def normalize_existing(row: dict[str, Any], method: str, amplitude: float, split: str) -> dict[str, Any]:
    output = dict(row)
    reference_dir = Path(output["reference_dir"])
    output |= {
        "method": method, "family": method, "dim": "3D", "k": 72,
        "R": amplitude, "amplitude_ang": amplitude, "split": split,
        "run_fdf": str(reference_dir / "RUN.fdf"),
        "reference_matrix": str(reference_dir / "graphene_6x6.TSHS"),
        "label_reused": True,
    }
    if not valid_reference(reference_dir):
        raise RuntimeError(f"invalid reusable SIESTA label: {reference_dir}")
    return output


def descriptor(method: str, amplitude: float, split: str, index: int,
               config: sampler.Configuration) -> dict[str, Any]:
    _positions, geometry_hash, real_amplitude = positions_and_hash(config)
    sample_id = f"cross6x6__{split}__{method}__R{amplitude:.2f}__seed{SAMPLER_TRAIN_SEED if split == 'train' else SAMPLER_TEST_SEED}__{index:03d}"
    return {
        "sample_id": sample_id, "method": method, "family": method, "dim": "3D", "k": 72,
        "R": amplitude, "amplitude_ang": amplitude, "split": split, "index": index,
        "hash": geometry_hash, "real_amplitude_ang": real_amplitude, "config": config,
    }


def reusable_train() -> dict[tuple[str, float], list[dict[str, Any]]]:
    pools = c.read_json(c.OUT / "pools_6x6.json")
    expanded = c.read_json(c.OUT / "coverage_pool_6x6_r012.json")
    output: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for method in METHODS[:3]:
        for amplitude in AMPLITUDES:
            if method == "sobol_sparse" and amplitude == 0.12:
                pool = expanded
            elif amplitude in (0.03, 0.08):
                pool = pools[f"{method}__3D__R{amplitude:.2f}__d64"]
            else:
                continue
            rows = [normalize_existing(row, method, amplitude, "train") for row in pool["samples"]]
            expected = configs(method, amplitude, N_TRAIN, SAMPLER_TRAIN_SEED)
            expected_hashes = [positions_and_hash(config)[1] for config in expected]
            if [row["hash"] for row in rows] != expected_hashes:
                raise RuntimeError(f"reusable pool does not match generator: {domain(method, amplitude)}")
            output[(method, amplitude)] = rows
    return output


def make_plan() -> dict[str, Any]:
    trains = reusable_train()
    generated: list[dict[str, Any]] = []
    for method in METHODS:
        for amplitude in AMPLITUDES:
            if (method, amplitude) in trains:
                continue
            rows = [descriptor(method, amplitude, "train", index, config)
                    for index, config in enumerate(configs(method, amplitude, N_TRAIN, SAMPLER_TRAIN_SEED))]
            trains[(method, amplitude)] = rows
            generated += rows

    validation = c.read_json(c.OUT / "common_dev_6x6_v1.json")["samples"]
    train_hashes = {row["hash"] for rows in trains.values() for row in rows}
    val_hashes = {row["hash"] for row in validation}
    if train_hashes & val_hashes:
        raise RuntimeError("training and Validation48 overlap")

    tests: dict[tuple[str, float], list[dict[str, Any]]] = {(method, amplitude): []
                                                            for method in METHODS for amplitude in AMPLITUDES}
    used = set(train_hashes) | val_hashes
    for method in METHODS:
        by_amplitude = {amplitude: configs(method, amplitude, 64, SAMPLER_TEST_SEED)
                        for amplitude in AMPLITUDES}
        for index in range(64):
            candidates = {amplitude: descriptor(method, amplitude, "test", index, by_amplitude[amplitude][index])
                          for amplitude in AMPLITUDES}
            hashes = {row["hash"] for row in candidates.values()}
            if len(hashes) != len(AMPLITUDES) or hashes & used:
                continue
            for amplitude, row in candidates.items():
                tests[(method, amplitude)].append(row)
                used.add(row["hash"])
            if all(len(tests[(method, amplitude)]) == N_TEST for amplitude in AMPLITUDES):
                break
        if not all(len(tests[(method, amplitude)]) == N_TEST for amplitude in AMPLITUDES):
            raise RuntimeError(f"not enough disjoint paired test structures for {method}")
    generated += [row for rows in tests.values() for row in rows]

    labels = existing_label_index()
    for row in generated:
        row["reference_dir"] = str(labels[row["hash"]]) if row["hash"] in labels else ""
    missing = [row for row in generated if not row["reference_dir"]]
    return {"trains": trains, "validation": validation, "tests": tests, "generated": generated, "missing": missing}


def serializable(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "config"}


def thermal_run_siesta(reference_dir: Path, *, command: str, use_shell: bool = False) -> dict[str, Any]:
    while True:
        temperature = thermal.reconcile_temperature()
        if temperature is None or temperature < 68:
            break
        time.sleep(10)
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
                thermal.reconcile_temperature()
                time.sleep(10)
        finally:
            with thermal._ACTIVE_LOCK:
                thermal._ACTIVE.pop(process.pid, None)
    return {"command": command if use_shell else command_args, "shell": use_shell,
            "returncode": process.returncode, "stdout_path": str(run_out)}


def materialize(rows: list[dict[str, Any]]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    if not (WORK / "_canonical_base.fdf").exists():
        shutil.copy2(CANONICAL, WORK / "_canonical_base.fdf")
    entries = [(row["sample_id"], row["config"]) for row in rows]
    pilot.materialize_structures(entries, structures_root=WORK / "structures",
                                 canonical_base_fdf=WORK / "_canonical_base.fdf")


def label_materialized() -> dict[str, Any]:
    thermal.log = log
    siesta_refs.run_siesta = thermal_run_siesta
    return siesta_refs.run_derivative_siesta_references(
        stencil_root=WORK,
        source_dataset_root=REPO_ROOT / "materials/graphene",
        siesta_command=pilot.DEFAULT_SIESTA_COMMAND,
        workers=SIERRA_WORKERS,
        require_positive_provenance_for_reuse=False,
        diagnostic_only=True,
    )


def record_for(row: dict[str, Any]) -> dict[str, Any]:
    reference_dir = Path(row["reference_dir"]) if row.get("reference_dir") else WORK / "siesta_hamiltonians" / row["sample_id"]
    if not valid_reference(reference_dir):
        raise RuntimeError(f"missing or invalid label for {row['sample_id']}: {reference_dir}")
    actual = c.geometry_record("6x6", reference_dir / "RUN.fdf")
    if actual["hash"] != row["hash"]:
        raise RuntimeError(f"geometry mismatch for {row['sample_id']}")
    return serializable(row) | {
        "run_fdf": str(reference_dir / "RUN.fdf"), "reference_dir": str(reference_dir),
        "reference_matrix": str(reference_dir / "graphene_6x6.TSHS"),
        "siesta_cpu_s": c.siesta_seconds(reference_dir / "RUN.out") or 0.0,
        "label_reused": not reference_dir.is_relative_to(WORK / "siesta_hamiltonians"),
    }


def prepare_datasets() -> dict[str, Any]:
    self_test()
    plan = make_plan()
    preliminary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "labels_pending",
        "methods": METHODS, "amplitudes_ang": AMPLITUDES, "N_train": N_TRAIN,
        "N_val": N_VAL, "N_test": N_TEST, "training_seeds": TRAIN_SEEDS,
        "training": {"loss": "graph2mat.metrics.elementwise_mse", "batch_size": 16,
                     "optimizer_updates": 8000, "validation_evaluations": 2000,
                     "scheduler": "cosine", "checkpoint": "best val_loss"},
        "siesta": {"canonical_fdf": str(CANONICAL), "canonical_sha256": sha256(CANONICAL),
                   "physics_hash": fdf_physics_hash(REFERENCE_FDF), "workers": SIERRA_WORKERS},
        "audit": {"reused_train_labels": sum("config" not in row for rows in plan["trains"].values() for row in rows),
                  "reused_generated_labels": len(plan["generated"]) - len(plan["missing"]),
                  "new_labels": len(plan["missing"]), "expected_new_models": 20},
    }
    c.write_json(ROOT / "manifest.json", preliminary)

    if plan["missing"]:
        preflight_path = WORK / "preflight.json"
        if not preflight_path.exists():
            angular = [row for row in plan["missing"] if row["method"] == "angular_shell_collective"]
            preflight_rows = [angular[0], angular[-1]]
            materialize(preflight_rows)
            result = label_materialized()
            if result["samples_failed"]:
                raise RuntimeError("SIESTA angular preflight failed")
            for row in preflight_rows:
                record_for(row)
            c.write_json(preflight_path, {"valid": True, "sample_ids": [row["sample_id"] for row in preflight_rows]})
            log("two-structure SIESTA preflight passed", sample_ids=[row["sample_id"] for row in preflight_rows])
        materialize(plan["missing"])
        result = label_materialized()
        if result["samples_failed"]:
            raise RuntimeError(f"SIESTA labeling failed for {result['samples_failed']} structures")

    plan = make_plan()
    if plan["missing"]:
        raise RuntimeError(f"{len(plan['missing'])} labels remain missing")
    train = {domain(method, amplitude): [record_for(row) if "config" in row else row
                                         for row in plan["trains"][(method, amplitude)]]
             for method in METHODS for amplitude in AMPLITUDES}
    test = {domain(method, amplitude): [record_for(row) for row in plan["tests"][(method, amplitude)]]
            for method in METHODS for amplitude in AMPLITUDES}
    validation = [normalize_existing(row, row["family"], float(row["amplitude_ang"]), "validation")
                  for row in plan["validation"]]
    all_train_hashes = {row["hash"] for rows in train.values() for row in rows}
    all_test_hashes = {row["hash"] for rows in test.values() for row in rows}
    val_hashes = {row["hash"] for row in validation}
    if len(all_test_hashes) != 12 * N_TEST or all_train_hashes & val_hashes or all_train_hashes & all_test_hashes or val_hashes & all_test_hashes:
        raise RuntimeError("dataset uniqueness/disjointness gate failed")
    physics_hash = fdf_physics_hash(REFERENCE_FDF)
    for row in [item for rows in train.values() for item in rows] + validation + [item for rows in test.values() for item in rows]:
        if fdf_physics_hash(Path(row["run_fdf"])) != physics_hash:
            raise RuntimeError(f"SIESTA settings differ for {row['sample_id']}")
    manifest = preliminary | {
        "status": "datasets_frozen", "datasets_frozen": True,
        "train": train, "validation": validation, "test": test,
        "hashes": {"train": c.ids_hash(sorted(all_train_hashes)), "validation": c.ids_hash(sorted(val_hashes)),
                   "test": c.ids_hash(sorted(all_test_hashes))},
    }
    c.write_json(ROOT / "manifest.json", manifest)
    log("datasets frozen", train=sum(map(len, train.values())), validation=len(validation),
        test=sum(map(len, test.values())), new_labels=preliminary["audit"]["new_labels"])
    return manifest


def existing_model(method: str, amplitude: float, seed: int) -> dict[str, Any] | None:
    if method != "sobol_sparse" or amplitude not in (0.08, 0.12):
        return None
    recipe = (f"sobol_sparse__3D__R0.08__d64" if amplitude == 0.08 else
              "sobol_sparse__3D__R0.12__d64__scaled")
    path = RUNS / f"{recipe}__N64__ts{seed}__coverage8k_full/result.json"
    if not path.exists():
        return None
    result = c.read_json(path)
    checkpoint = Path(result["checkpoint"])
    config_path = path.parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    valid = (
        checkpoint.exists() and sha256(checkpoint) == result["checkpoint_sha256"] and
        config["model"]["loss"] == "graph2mat.metrics.elementwise_mse" and
        int(config["data"]["batch_size"]) == 16 and int(config["trainer"]["max_epochs"]) == 2000 and
        int(config["trainer"]["check_val_every_n_epoch"]) == 1 and
        config["lr_scheduler"]["class_path"].endswith("CosineAnnealingLR") and
        int(config["lr_scheduler"]["init_args"]["T_max"]) == 2000 and
        len(config["trainer"].get("callbacks", [])) == 1 and
        config["trainer"]["callbacks"][0]["init_args"]["monitor"] == "val_loss"
    )
    if not valid:
        raise RuntimeError(f"reusable checkpoint failed current-protocol audit: {path}")
    return {
        "model_id": f"cross6x6__{domain(method, amplitude)}__seed{seed}",
        "train_method": method, "train_R": amplitude, "training_seed": seed,
        "source": "reused coverage8k_full", "checkpoint": str(checkpoint),
        "checkpoint_sha256": result["checkpoint_sha256"], "best_update": result.get("best_update"),
        "train_seconds": result.get("train_seconds"), "peak_gpu_mib": result.get("peak_gpu_mib"),
        "val_H_MAE_meV": result["val_metrics"]["H_MAE_meV"],
        "val_rel_Frob": result["val_metrics"]["rel_Frob"], "result_path": str(path),
    }


def matrix_metrics(sample: c.s4.Sample, predicted_root: Path) -> dict[str, float]:
    predicted = predicted_root / sample.sample_id / "ML_prediction.HSX"
    h_pred, h_ref = c.s4.gamma_hk(predicted), c.s4.gamma_hk(sample.reference_matrix)
    error = h_pred - h_ref
    return {"H_MAE_meV": 1e3 * float(np.mean(np.abs(error))),
            "rel_Frob": float(np.linalg.norm(error) / np.linalg.norm(h_ref))}


def evaluate_samples(checkpoint: Path, rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    samples = [c.sample_from_dict(row) for row in rows]
    predicted_root = c.predict(checkpoint, samples, out_dir, "gpu")
    output = [{"sample_id": sample.sample_id, **matrix_metrics(sample, predicted_root)} for sample in samples]
    shutil.rmtree(predicted_root, ignore_errors=True)
    return output


def train_process(config: Path, run_dir: Path, name: str) -> tuple[Path, float, int]:
    env = ENV | {"PYTHONPATH": os.pathsep.join(filter(None, [str(c.s4.TORCH_COMPAT_DIR), ENV.get("PYTHONPATH", "")]))}
    start, peak = time.monotonic(), 0
    with (run_dir / "train.log").open("w", encoding="utf-8") as handle:
        process = subprocess.Popen([str(c.s4.GRAPH2MAT_BIN), "models", "mace", "main", "fit", "-c", config.name],
                                   cwd=run_dir, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)
        item = {"process": process, "name": name, "started": time.monotonic(), "paused": False}
        with thermal._ACTIVE_LOCK:
            thermal._ACTIVE[process.pid] = item
        try:
            while process.poll() is None:
                peak = max(peak, c.gpu_mib(process.pid))
                time.sleep(10)
        finally:
            with thermal._ACTIVE_LOCK:
                thermal._ACTIVE.pop(process.pid, None)
    if process.returncode:
        raise RuntimeError(f"Graph2Mat failed for {name}; see {run_dir / 'train.log'}")
    checkpoints = sorted(run_dir.rglob("best-*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"{name}: expected one best checkpoint, found {len(checkpoints)}")
    return checkpoints[0], time.monotonic() - start, peak


def train_one(method: str, amplitude: float, seed: int, manifest: dict[str, Any]) -> dict[str, Any]:
    reused = existing_model(method, amplitude, seed)
    if reused:
        return reused
    model_id = f"cross6x6__{domain(method, amplitude)}__seed{seed}"
    run_dir, result_path, done_path = RUNS / model_id, RUNS / model_id / "result.json", RUNS / model_id / "train_done.json"
    if result_path.exists():
        saved = c.read_json(result_path)
        checkpoint = Path(saved["checkpoint"])
        if checkpoint.exists() and sha256(checkpoint) == saved["checkpoint_sha256"]:
            return saved
    train_rows, val_rows = manifest["train"][domain(method, amplitude)], manifest["validation"]
    if done_path.exists():
        done = c.read_json(done_path)
        checkpoint = Path(done["checkpoint"])
        seconds, peak = float(done["seconds"]), int(done["peak_gpu_mib"])
        if not checkpoint.exists() or sha256(checkpoint) != done["checkpoint_sha256"]:
            raise RuntimeError(f"invalid completion marker: {done_path}")
    else:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        config = c.s4.build_graph2mat_config(run_dir, [c.sample_from_dict(row) for row in train_rows],
                                             [c.sample_from_dict(row) for row in val_rows], run_name=model_id,
                                             accelerator="gpu", training_seed=seed, **c.TRAIN_KW)
        c.apply_config_patch(config, c.fixed_update_elemmse_patch("6x6", N_TRAIN))
        checkpoint, seconds, peak = train_process(config, run_dir, model_id)
        c.write_json(done_path, {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                                 "seconds": seconds, "peak_gpu_mib": peak})
    val_metrics = evaluate_samples(checkpoint, val_rows, run_dir / "validation_eval")
    step = re.search(r"step=?([0-9]+)", checkpoint.name)
    result = {
        "model_id": model_id, "train_method": method, "train_R": amplitude, "training_seed": seed,
        "source": "new", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "best_update": int(step.group(1)) if step else None, "train_seconds": seconds,
        "gpu_h": seconds / 3600, "peak_gpu_mib": peak,
        "val_H_MAE_meV": float(np.mean([row["H_MAE_meV"] for row in val_metrics])),
        "val_rel_Frob": float(np.mean([row["rel_Frob"] for row in val_metrics])),
        "train_ids_sha256": c.ids_hash([row["sample_id"] for row in train_rows]),
        "validation_ids_sha256": c.ids_hash([row["sample_id"] for row in val_rows]),
        "result_path": str(result_path),
    }
    c.write_json(result_path, result)
    return result


def gpu_free_mib() -> int:
    try:
        value = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True, timeout=10).stdout.splitlines()[0]
        return int(value.strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def train_models(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    thermal.log = log
    jobs = [(method, amplitude, seed) for method in METHODS for amplitude in AMPLITUDES for seed in TRAIN_SEEDS
            if existing_model(method, amplitude, seed) is None]
    completed = [existing_model(method, amplitude, seed) for method in METHODS for amplitude in AMPLITUDES
                 for seed in TRAIN_SEEDS]
    output = [row for row in completed if row]
    pending, futures = list(jobs), {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        while pending or futures:
            cap = 3
            while pending and len(futures) < cap:
                if (len(futures) + 1) * VRAM_PER_MODEL_MIB + GPU_RESERVE_MIB > 32607:
                    break
                if gpu_free_mib() < VRAM_PER_MODEL_MIB + GPU_RESERVE_MIB:
                    break
                job = pending.pop(0)
                futures[pool.submit(train_one, *job, manifest)] = job
                log("training launched", method=job[0], R=job[1], seed=job[2], concurrency=len(futures))
            done = [future for future in futures if future.done()]
            for future in done:
                job = futures.pop(future)
                result = future.result()
                output.append(result)
                c.write_csv(ROOT / "model_results.csv", sorted(output, key=lambda row: (METHODS.index(row["train_method"]), row["train_R"], row["training_seed"])))
                log("training completed", method=job[0], R=job[1], seed=job[2], checkpoint=result["checkpoint_sha256"])
            if not done:
                time.sleep(10)
    if len(output) != 24:
        raise RuntimeError(f"expected 24 models, found {len(output)}")
    c.write_csv(ROOT / "model_results.csv", sorted(output, key=lambda row: (METHODS.index(row["train_method"]), row["train_R"], row["training_seed"])))
    return output


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_cross_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["model_id"]), str(row["test_method"]), str(row["test_R"])), []).append(row)
    output = []
    for (model_id, test_method, test_r), values in groups.items():
        if len(values) != N_TEST:
            continue
        first = values[0]
        output.append({
            "model_id": model_id, "train_method": first["train_method"], "train_R": first["train_R"],
            "training_seed": first["training_seed"], "test_method": test_method, "test_R": test_r,
            "n_structures": len(values), "H_MAE_meV": float(np.mean([float(row["H_MAE_meV"]) for row in values])),
            "rel_Frob": float(np.mean([float(row["rel_Frob"]) for row in values])),
        })
    return sorted(output, key=lambda row: (METHODS.index(row["train_method"]), float(row["train_R"]),
                                            int(row["training_seed"]), METHODS.index(row["test_method"]), float(row["test_R"])))


def evaluate_models(manifest: dict[str, Any], models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = ROOT / "per_structure_results.csv"
    stored: list[dict[str, Any]] = read_csv(path)
    tests = [(method, amplitude, row) for method in METHODS for amplitude in AMPLITUDES
             for row in manifest["test"][domain(method, amplitude)]]
    expected_keys = {(method, f"{amplitude:.2f}", row["sample_id"]) for method, amplitude, row in tests}
    for model in models:
        previous = [row for row in stored if row["model_id"] == model["model_id"]]
        found = {(row["test_method"], f"{float(row['test_R']):.2f}", row["sample_id"]) for row in previous}
        if found == expected_keys:
            continue
        stored = [row for row in stored if row["model_id"] != model["model_id"]]
        test_rows = [row for _method, _amplitude, row in tests]
        run_dir = Path(model["result_path"]).parent
        metrics = {row["sample_id"]: row for row in evaluate_samples(Path(model["checkpoint"]), test_rows,
                                                                       run_dir / "method_amplitude_crosstest")}
        for test_method, test_r, sample in tests:
            stored.append({
                "model_id": model["model_id"], "train_method": model["train_method"],
                "train_R": model["train_R"], "training_seed": model["training_seed"],
                "test_method": test_method, "test_R": test_r, "sample_id": sample["sample_id"],
                **metrics[sample["sample_id"]],
            })
        stored.sort(key=lambda row: (row["model_id"], METHODS.index(row["test_method"]), float(row["test_R"]), row["sample_id"]))
        c.write_csv(path, stored)
        matrix = build_cross_matrix(stored)
        c.write_csv(ROOT / "cross_matrix.csv", matrix)
        log("model cross-evaluated", model_id=model["model_id"], cells=len(matrix))
    if len(stored) != 24 * 12 * N_TEST:
        raise RuntimeError(f"expected {24 * 12 * N_TEST} per-structure rows, found {len(stored)}")
    matrix = build_cross_matrix(stored)
    if len(matrix) != 24 * 12:
        raise RuntimeError(f"expected 288 seed-specific cells, found {len(matrix)}")
    c.write_csv(ROOT / "cross_matrix.csv", matrix)
    return matrix


def analyze(matrix: list[dict[str, Any]]) -> None:
    averaged: dict[tuple[str, float, str, float], dict[str, float]] = {}
    for train_method in METHODS:
        for train_r in AMPLITUDES:
            for test_method in METHODS:
                for test_r in AMPLITUDES:
                    selected = [row for row in matrix if row["train_method"] == train_method and
                                float(row["train_R"]) == train_r and row["test_method"] == test_method and
                                float(row["test_R"]) == test_r]
                    if len(selected) != 2:
                        raise RuntimeError("each cross-test cell must contain exactly two seeds")
                    averaged[(train_method, train_r, test_method, test_r)] = {
                        metric: float(np.mean([float(row[metric]) for row in selected]))
                        for metric in ("H_MAE_meV", "rel_Frob")
                    }
    domain_scores = []
    for train_method in METHODS:
        for train_r in AMPLITUDES:
            cells = [averaged[(train_method, train_r, test_method, test_r)]["H_MAE_meV"]
                     for test_method in METHODS for test_r in AMPLITUDES]
            diagonal = averaged[(train_method, train_r, train_method, train_r)]["H_MAE_meV"]
            domain_scores.append({"method": train_method, "R": train_r, "mean": float(np.mean(cells)),
                                  "worst": max(cells), "diagonal": diagonal})
    method_scores = [{"method": method,
                      "mean": float(np.mean([row["mean"] for row in domain_scores if row["method"] == method])),
                      "worst": max(row["worst"] for row in domain_scores if row["method"] == method)}
                     for method in METHODS]
    best_mean = min(method_scores, key=lambda row: row["mean"])
    best_worst = min(domain_scores, key=lambda row: row["worst"])
    extrapolation = [value["H_MAE_meV"] for (tm, tr, _xm, xr), value in averaged.items() if xr > tr]
    interpolation = [value["H_MAE_meV"] for (tm, tr, _xm, xr), value in averaged.items() if xr < tr]
    diagonal = [value["H_MAE_meV"] for (tm, tr, xm, xr), value in averaged.items() if tm == xm and tr == xr]
    off_diagonal = [value["H_MAE_meV"] for (tm, tr, xm, xr), value in averaged.items() if tm != xm or tr != xr]
    penalties = []
    domains = [(method, amplitude) for method in METHODS for amplitude in AMPLITUDES]
    for train_method, train_r in domains:
        baseline = averaged[(train_method, train_r, train_method, train_r)]["H_MAE_meV"]
        for test_method, test_r in domains:
            if (train_method, train_r) != (test_method, test_r):
                value = averaged[(train_method, train_r, test_method, test_r)]["H_MAE_meV"]
                penalties.append({"train": domain(train_method, train_r), "test": domain(test_method, test_r),
                                  "ratio": value / baseline, "delta": value - baseline})
    asymmetries = []
    for left_index, (left_method, left_r) in enumerate(domains):
        for right_method, right_r in domains[left_index + 1:]:
            left_to_right = averaged[(left_method, left_r, right_method, right_r)]["H_MAE_meV"]
            right_to_left = averaged[(right_method, right_r, left_method, left_r)]["H_MAE_meV"]
            asymmetries.append({"A": domain(left_method, left_r), "B": domain(right_method, right_r),
                                "A_to_B": left_to_right, "B_to_A": right_to_left,
                                "delta": left_to_right - right_to_left})
    worst_penalty = max(penalties, key=lambda row: row["ratio"])
    largest_asymmetries = sorted(asymmetries, key=lambda row: abs(row["delta"]), reverse=True)[:10]
    report = [
        "# Cross-testing de método y amplitud en grafeno 6×6", "",
        "## Protocolo", "",
        "Doce dominios (cuatro métodos × tres amplitudes), Train64, Validation48 común, Test32 por dominio y dos semillas. "
        "Todos los modelos usan `elementwise_mse`, batch 16, 8000 actualizaciones, 2000 evaluaciones de validación, "
        "scheduler coseno y el mejor checkpoint por `val_loss`. Los intervalos no se interpretan inferencialmente con n=2.", "",
        "## Resultados principales", "",
        f"- Mejor método medio: **{best_mean['method']}**, H-MAE media {best_mean['mean']:.3f} meV.",
        f"- Mejor peor caso: **{best_worst['method']} R={best_worst['R']:.2f} Å**, peor H-MAE {best_worst['worst']:.3f} meV.",
        f"- Media diagonal (mismo método y amplitud): {np.mean(diagonal):.3f} meV.",
        f"- Media fuera de diagonal: {np.mean(off_diagonal):.3f} meV.",
        f"- Penalización media fuera de diagonal: {np.mean([row['ratio'] for row in penalties]):.2f}× respecto a la diagonal del modelo.",
        f"- Mayor penalización: {worst_penalty['train']} → {worst_penalty['test']}, {worst_penalty['ratio']:.2f}× ({worst_penalty['delta']:+.3f} meV).",
        f"- Transferencia hacia amplitudes mayores: {np.mean(extrapolation):.3f} meV; hacia menores: {np.mean(interpolation):.3f} meV.", "",
        "## Resultado por dominio de entrenamiento", "",
        "| Método | Rtrain (Å) | Diagonal seed 0 | Diagonal seed 1 | Media 12 tests | Peor test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in domain_scores:
        seeds = sorted((cell for cell in matrix if cell["train_method"] == row["method"] and
                        float(cell["train_R"]) == row["R"] and cell["test_method"] == row["method"] and
                        float(cell["test_R"]) == row["R"]), key=lambda cell: int(cell["training_seed"]))
        report.append(f"| {row['method']} | {row['R']:.2f} | {float(seeds[0]['H_MAE_meV']):.3f} | "
                      f"{float(seeds[1]['H_MAE_meV']):.3f} | {row['mean']:.3f} | {row['worst']:.3f} |")
    report += ["", "## Mayores asimetrías de transferencia", "",
               "Δ = H-MAE(A→B) − H-MAE(B→A); el signo indica qué dirección transfiere peor.", "",
               "| A | B | A→B (meV) | B→A (meV) | Δ (meV) |", "|---|---|---:|---:|---:|"]
    report += [f"| {row['A']} | {row['B']} | {row['A_to_B']:.3f} | {row['B_to_A']:.3f} | {row['delta']:+.3f} |"
               for row in largest_asymmetries]
    report += ["", "Los CSV conservan ambas semillas y las 32 estructuras de cada test. Menor H-MAE y menor Frobenius relativo son mejores."]
    (REPO_ROOT / "docs/dataset_design_method_amplitude_crosstest_6x6.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = c.read_json(ROOT / "manifest.json")
    manifest["status"] = "complete"
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["summary"] = {"best_mean": best_mean, "best_worst_case": best_worst,
                           "mean_diagonal_H_MAE_meV": float(np.mean(diagonal)),
                           "mean_off_diagonal_H_MAE_meV": float(np.mean(off_diagonal)),
                           "mean_transfer_penalty_ratio": float(np.mean([row["ratio"] for row in penalties])),
                           "worst_transfer_penalty": worst_penalty,
                           "largest_asymmetry": largest_asymmetries[0]}
    c.write_json(ROOT / "manifest.json", manifest)
    log("campaign complete", **manifest["summary"])


def run_all() -> None:
    manifest = prepare_datasets()
    models = train_models(manifest)
    matrix = evaluate_models(manifest, models)
    analyze(matrix)
    with thermal._ACTIVE_LOCK:
        if thermal._ACTIVE:
            raise RuntimeError(f"residual managed processes: {list(thermal._ACTIVE)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("self-test", "prepare", "train", "evaluate", "analyze", "all"), default="all", nargs="?")
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
        analyze(read_csv(ROOT / "cross_matrix.csv"))
    else:
        run_all()


if __name__ == "__main__":
    main()
