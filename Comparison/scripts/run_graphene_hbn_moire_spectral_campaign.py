#!/usr/bin/env python3
"""Resumable control plane for the bilayer-graphene/hBN magic-angle spectra."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEEPH_HUGE_STRUCTURE_CHUNK_ROWS = 500_000
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_graphene_hbn_moire_target import build_geometry_only  # noqa: E402
from build_graphene_hbn_bilayer_train_dataset import build_dataset as build_training_dataset  # noqa: E402
from dataset_recipe_helpers import slugify_label  # noqa: E402
from deeph_config import render_inference_config  # noqa: E402
from export_siesta_hamiltonian_to_deeph import export as export_siesta_hamiltonian  # noqa: E402
from generate_siesta_overlap_only import generate as generate_overlap, parse_time_v  # noqa: E402
from g2m_deeph_runner import _exclusive_gpu_training, _wait_for_free_gpu_memory  # noqa: E402
from run_deeph_sparse_spectrum import run as run_sparse_spectrum  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "Comparison/config/graphene_hbn_magic_angle_spectral_campaign.json"
DEFAULT_ROOT = REPO_ROOT / "Comparison/results/graphene_hbn_magic_angle_spectral"
MUTATING_ACTIONS = {
    "generate-training-data",
    "train",
    "build-target",
    "build-overlap",
    "predict",
    "solve-bands",
    "solve-dos",
    "aggregate",
    "run",
    "resume",
}
MIN_FREE_DISK_PERCENT = 12.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], cwd: Path = REPO_ROOT) -> str | None:
    try:
        completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or completed.stderr).strip()
    return text if completed.returncode == 0 and text else None


def process_resource_blocked(completed: subprocess.CompletedProcess[str]) -> bool:
    text = f"{completed.stdout or ''}\n{completed.stderr or ''}".lower()
    return completed.returncode in {-9, 137} or "out of memory" in text


def disk_headroom(root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(root)
    free_percent = 100.0 * usage.free / usage.total
    return {
        "status": "safe" if free_percent >= MIN_FREE_DISK_PERCENT else "resource_blocked",
        "free_bytes": usage.free,
        "free_percent": free_percent,
        "minimum_free_percent": MIN_FREE_DISK_PERCENT,
    }


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("campaign_kind") != "moire_spectral_prediction":
        raise RuntimeError(f"{path}: not a moire_spectral_prediction campaign")
    quotas = config.get("training_quota_matrix_n30")
    if (
        not isinstance(quotas, list)
        or len(quotas) != len(config.get("training_presets") or [])
        or any(not isinstance(row, list) or len(row) != len(config.get("temperatures_K") or []) for row in quotas)
        or sum(sum(int(value) for value in row) for row in quotas) != 30
    ):
        raise RuntimeError("training_quota_matrix_n30 must match presets/temperatures and sum to 30")
    if not active_model_seeds(config) or not set(active_model_seeds(config)) <= {
        int(value) for value in config["model_seeds"]
    }:
        raise RuntimeError("active_model_seeds must be a non-empty subset of model_seeds")
    return config


def active_model_seeds(config: dict[str, Any]) -> list[int]:
    return [int(value) for value in config.get("active_model_seeds", config["model_seeds"])]


def campaign_root(config: dict[str, Any]) -> Path:
    return resolve_path(config.get("output_root") or DEFAULT_ROOT)


def status_path(root: Path) -> Path:
    return root / "status.json"


def set_status(root: Path, **updates: Any) -> dict[str, Any]:
    payload = read_json(status_path(root))
    payload.update(updates)
    payload["updated_at"] = now()
    write_json(status_path(root), payload)
    return payload


@contextmanager
def campaign_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "campaign.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Campaign already running; lock held at {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def stage_state(root: Path, stage: str, status: str, **extra: Any) -> None:
    path = root / "stages" / f"{stage}.json"
    payload = read_json(path)
    payload.update({"stage": stage, "status": status, "updated_at": now(), **extra})
    payload.setdefault("started_at", now())
    write_json(path, payload)
    set_status(root, current_stage=stage, stage_status=status)


def environment_inventory(config: dict[str, Any]) -> dict[str, Any]:
    commit = command_output(["git", "rev-parse", "HEAD"])
    dirty = command_output(["git", "status", "--short"]) or ""
    nvidia = command_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    mem_kib = None
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            mem_kib = int(line.split()[1])
            break
    presets = [str(item) for item in config["training_presets"]]
    material_hashes = {}
    for preset in presets:
        root = REPO_ROOT / "materials" / preset
        material_hashes[preset] = {
            "RUN.fdf": sha256(root / "RUN.fdf"),
            "material.yaml": sha256(root / "material.yaml"),
        }
    return {
        "created_at": now(),
        "repo_commit": commit,
        "dirty_worktree": bool(dirty),
        "git_status_short": dirty.splitlines(),
        "python": sys.version,
        "siesta": (command_output([str(config["overlap"]["siesta_command"]), "--version"]) or "").splitlines()[:8],
        "julia": command_output(["julia", "--version"]),
        "graph2mat": command_output([str(REPO_ROOT / ".venv/bin/python"), "-c", "import graph2mat; print(graph2mat.__version__)"]),
        "torch": command_output([str(REPO_ROOT / ".venv/bin/python"), "-c", "import torch; print(torch.__version__)"]),
        "sisl": command_output([str(REPO_ROOT / ".venv/bin/python"), "-c", "import sisl; print(sisl.__version__)"]),
        "cpu_count": os.cpu_count(),
        "ram_gib": mem_kib / 1024**2 if mem_kib else None,
        "gpu": nvidia,
        "disk_free_gib": shutil.disk_usage(REPO_ROOT).free / 1024**3,
        "material_hashes": material_hashes,
    }


def legacy_inventory(root: Path) -> dict[str, Any]:
    legacy_root = REPO_ROOT / "Comparison/results/ml_vs_siesta_cross_structure_bilayer_moire"
    payload = {
        "scientific_status": "legacy_invalid_geometry",
        "reason": "4-atom source and hBN-twist target; not bilayer graphene/hBN magic-angle physics",
        "preserved": legacy_root.exists(),
        "path": str(legacy_root),
        "included_in_spectral_results": False,
    }
    write_json(root / "legacy_results_inventory.json", {"runs": [payload]})
    return payload


def plan(config: dict[str, Any], root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    target = config["target"]
    stages = [
        "generate-training-data",
        "train",
        "build-target",
        "build-overlap",
        "predict",
        "solve-bands",
        "solve-dos",
        "aggregate",
    ]
    inventory = environment_inventory(config)
    write_json(root / "environment_inventory.json", inventory)
    legacy = legacy_inventory(root)
    payload = {
        "schema": config["schema"],
        "campaign_kind": config["campaign_kind"],
        "target_contract": config["target_contract"],
        "scientific_status": "prediction_only",
        "reference_hamiltonian_available": False,
        "target": target,
        "training_sizes": config["training_sizes"],
        "model_seeds": config["model_seeds"],
        "active_model_seeds": active_model_seeds(config),
        "models": ["graph2mat", "deeph"],
        "stages": stages,
        "badges": [
            "No target H reference",
            "Prediction only",
            "Exact PAO overlap",
            "Sparse partial eigensolver",
            "Estimated neutrality energy",
            "Rigid/unrelaxed structure",
            "hBN strain recorded",
        ],
        "forbidden": ["S=I", "dense large-cell solve", "target MAE", "target Frobenius"],
        "legacy": legacy,
        "environment_inventory": str(root / "environment_inventory.json"),
        "config": str(DEFAULT_CONFIG),
        "config_sha256": sha256(DEFAULT_CONFIG),
        "created_at": now(),
    }
    write_json(root / "campaign_manifest.json", payload)
    set_status(
        root,
        running=False,
        state="planned",
        current_stage=None,
        stage_status=None,
        stop_requested=False,
        error=None,
        pid=None,
    )
    return payload


def build_target(config: dict[str, Any], root: Path, *, resume: bool) -> dict[str, Any]:
    target_root = root / "target"
    geometry_manifest = target_root / "moire_geometry.json"
    if resume and geometry_manifest.exists():
        geometry = read_json(geometry_manifest)
        if geometry.get("num_atoms") == config["target"]["expected_atoms"]:
            return {"status": "reused", "geometry": geometry}
    m, n = (int(value) for value in config["target"]["commensurate_index"])
    source = REPO_ROOT / "materials" / config["source_preset"] / "RUN.fdf"
    result = build_geometry_only(
        source,
        target_root,
        approximant=2,
        m=m,
        n=n,
        overwrite=True,
    )
    geometry = read_json(geometry_manifest)
    if geometry.get("num_atoms") != config["target"]["expected_atoms"]:
        raise RuntimeError(f"Magic target atom count mismatch: {geometry.get('num_atoms')}")
    return {"status": "completed", **result, "geometry": geometry}


def build_overlap(config: dict[str, Any], root: Path, *, resume: bool) -> dict[str, Any]:
    output = root / "overlap"
    manifest = output / "overlap_manifest.json"
    if resume and manifest.exists():
        existing = read_json(manifest)
        overlap_h5 = Path(str(existing.get("export", {}).get("overlaps_h5") or ""))
        if existing.get("status") == "completed" and overlap_h5.exists():
            sparse_path = output / "diagnostics_sparse.json"
            sparse = read_json(sparse_path)
            kpoints = sparse.get("kpoints") if isinstance(sparse.get("kpoints"), list) else []
            if (
                sparse.get("status") == "valid"
                and sparse.get("no_identity_overlap") is True
                and len(kpoints) == 3
                and all(
                    row.get("positivity_status") == "validated_sparse_arpack"
                    and float(row.get("minimum_eigenvalue") or 0) > 0
                    for row in kpoints
                )
            ):
                existing["diagnostics"] = sparse
                existing["sparse_positivity_gate"] = {
                    "status": "validated",
                    "path": str(sparse_path),
                    "sha256": sha256(sparse_path),
                }
                write_json(output / "diagnostics.json", sparse)
                write_json(manifest, existing)
            return {"status": "reused", "manifest": existing}
    fdf = root / "target/splits/test/0/RUN.fdf"
    if not fdf.exists():
        raise RuntimeError("Build the magic-angle target before its overlap")
    settings = config["overlap"]
    generated = generate_overlap(
        fdf,
        output,
        preset=config["source_preset"],
        siesta_command=str(settings["siesta_command"]),
        kgrid=int(settings["kgrid"][0]),
        overwrite=True,
    )
    return {"status": generated["status"], "manifest": generated}


def training_source_payload(
    config: dict[str, Any],
    root: Path,
    preset: str,
    temperature: int,
) -> dict[str, Any]:
    suffix = preset.rsplit("_", 1)[-1]
    template_suffix = suffix if suffix in {"AA", "AB1", "AB2"} else "AA"
    template_path = REPO_ROOT / f"Comparison/config/bilayer_graphene_hbn_{template_suffix}_md30_payload.json"
    payload = read_json(template_path)
    max_train = 64 if temperature == 300 else 32
    usable = int(round(max_train / 0.8))
    total = usable + 20  # two ten-frame temporal gaps
    dataset_root = (
        REPO_ROOT / "Comparison/datasets/graphene_hbn_bilayer_md_nested/master_sources"
        / preset / f"T{temperature}"
    )
    result_root = root / "training_data" / preset / f"T{temperature}"
    recipe_id = f"{preset}_T{temperature}_master{total}"
    payload.update(
        {
            "description": (
                f"Master C4BN Verlet/NVE MD source for {preset}, initialized at {temperature} K: "
                f"{total} frames -> {max_train} frozen train frames."
            ),
            "material_preset": preset,
            "material": {"mode": "preset", "preset": preset},
            "system_label": preset,
            "dataset_root": str(dataset_root),
            "output_root": str(result_root),
            "overwrite_datasets": False,
            "run_id": recipe_id,
            "temporal_gap": 10,
        }
    )
    payload["dataset_sweep"] = {
        "enabled": True,
        "max_datasets": 1,
        "recipes": [
            {
                "recipe_id": recipe_id,
                "label": f"{preset} master MD at {temperature} K",
                "size": total,
                "thermal_regime": "single_temperature_trajectory",
                "split_intent": (
                    f"{max_train} train + frozen validation/test after two 10-frame temporal gaps."
                ),
                "blocks": [
                    {
                        "block_id": f"{suffix}_T{temperature}_{total}",
                        "label": f"{total} snapshots, T_initial={temperature} K",
                        "n_snapshots": total,
                        "temperature_K": temperature,
                        "seed": 10000 + temperature * 10 + sum(ord(char) for char in preset),
                    }
                ],
            }
        ],
    }
    return payload


def training_source_dataset(payload: dict[str, Any]) -> Path:
    recipe_id = str(payload["dataset_sweep"]["recipes"][0]["recipe_id"])
    return Path(payload["dataset_root"]) / slugify_label(recipe_id)


def generate_training_data(config: dict[str, Any], root: Path, *, resume: bool) -> dict[str, Any]:
    rows = []
    for preset in config["training_presets"]:
        for temperature in config["temperatures_K"]:
            payload = training_source_payload(config, root, str(preset), int(temperature))
            payload_path = root / "training_data" / str(preset) / f"T{temperature}" / "payload.json"
            status_json = payload_path.parent / "status.json"
            runner_manifest = payload_path.parent / "runner_manifest.json"
            write_json(payload_path, payload)
            if resume and read_json(status_json).get("status", {}).get("returncode") == 0:
                rows.append(
                    {
                        "preset": preset,
                        "temperature_K": temperature,
                        "status": "reused",
                        "payload": str(payload_path),
                    }
                )
                continue
            command = [
                str(REPO_ROOT / ".venv/bin/python"),
                str(SCRIPT_DIR / "run_g2m_deeph_payload_once.py"),
                str(payload_path),
                "--status-json",
                str(status_json),
                "--manifest-json",
                str(runner_manifest),
            ]
            completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
            rows.append(
                {
                    "preset": preset,
                    "temperature_K": temperature,
                    "status": "completed" if completed.returncode == 0 else "failed",
                    "returncode": completed.returncode,
                    "payload": str(payload_path),
                    "runner_manifest": str(runner_manifest),
                }
            )
            if completed.returncode != 0:
                break
        if rows and rows[-1]["status"] == "failed":
            break
    sources = []
    for preset in config["training_presets"]:
        for temperature in config["temperatures_K"]:
            payload = training_source_payload(config, root, str(preset), int(temperature))
            sources.append(training_source_dataset(payload))
    ready = rows and all(row["status"] in {"completed", "reused"} for row in rows)
    nested = []
    if ready:
        for size in config["training_sizes"]:
            destination = (
                REPO_ROOT / "Comparison/datasets/graphene_hbn_bilayer_md_nested" / f"n{int(size)}"
            )
            if resume and read_json(destination / "frozen_split_manifest.json").get("split_counts", {}).get("train") == int(size):
                result = {"output_root": str(destination), "requested_train_size": int(size), "status": "reused"}
            else:
                result = build_training_dataset(
                    sources,
                    destination,
                    overwrite=destination.exists(),
                    train_size=int(size),
                    train_quotas=[
                        int(quota) * (int(size) // 30)
                        for registry_quotas in config["training_quota_matrix_n30"]
                        for quota in registry_quotas
                    ],
                )
                result["status"] = "completed"
            nested.append(result)
        for smaller, larger in zip(nested, nested[1:], strict=False):
            small_rows = read_json(Path(smaller["output_root"]) / "frozen_split_manifest.json").get("rows", [])
            large_rows = read_json(Path(larger["output_root"]) / "frozen_split_manifest.json").get("rows", [])
            small_ids = {row["sample_id"] for row in small_rows if row.get("split") == "train"}
            large_ids = {row["sample_id"] for row in large_rows if row.get("split") == "train"}
            if not small_ids <= large_ids:
                raise RuntimeError(f"Nested dataset invariant failed: n{len(small_ids)} is not a subset of n{len(large_ids)}")
    result = {
        "status": "completed" if ready and len(nested) == len(config["training_sizes"]) else "failed",
        "source_runs": rows,
        "nested_datasets": nested,
        "shared_validation": True,
    }
    write_json(root / "training_data/nested_dataset_manifest.json", result)
    return result


def training_payload(config: dict[str, Any], root: Path, size: int) -> dict[str, Any]:
    seeds = active_model_seeds(config)
    payload = read_json(REPO_ROOT / "Comparison/config/bilayer_graphene_hbn_AA_md30_payload.json")
    payload.pop("dataset_sweep", None)
    payload.pop("overwrite_datasets", None)
    # Mixed registry datasets keep each source SystemLabel; paths come from the frozen manifest.
    payload.pop("system_label", None)
    payload.update(
        {
            "description": f"Graph2Mat+DeepH C4BN campaign, n_train={size}, seeds={seeds}.",
            "dataset_mode": "reuse_validated",
            "run_mode": "benchmark",
            "dataset_root": str(REPO_ROOT / f"Comparison/datasets/graphene_hbn_bilayer_md_nested/n{size}"),
            "output_root": str(root / "training" / f"n{size}"),
            "run_id": f"n{size}",
            "final_benchmark": True,
            "protocol_stage": "search",
            "metric_fail_policy": "fail_closed",
            "early_stopping": dict(config["early_stopping"]),
            "reuse_run_root": True,
            "resume_training_sweep": True,
            "resume_training_sweep_from_run_root": str(root / "training" / f"n{size}" / f"n{size}"),
        }
    )
    performance = dict(payload.get("performance") or {})
    performance.update(
        {
            "max_parallel_graph2mat_training_jobs": 1,
            "max_parallel_deeph_training_jobs": 1,
            "model_batch_schedule": "alternating",
            "model_batch_start": "deeph",
            "graph2mat_min_free_gpu_memory_mb": 18000,
            "deeph_min_free_gpu_memory_mb": 24000,
            "gpu_memory_wait_timeout_seconds": 3600,
            "gpu_memory_poll_seconds": 15,
        }
    )
    payload["performance"] = performance
    graph = dict(payload["graph2mat_overrides"])
    deeph = dict(payload["deeph"])
    manual_runs = []
    for seed in seeds:
        graph_seed = {**graph, "seed_everything": int(seed)}
        deeph_seed = {**deeph, "seed": int(seed)}
        manual_runs.extend(
            [
                {
                    "id": f"g2m_n{size}_seed{seed}",
                    "config_id": f"g2m_n{size}_seed{seed}",
                    "model": "graph2mat",
                    "overrides": graph_seed,
                },
                {
                    "id": f"deeph_n{size}_seed{seed}",
                    "config_id": f"deeph_n{size}_seed{seed}",
                    "model": "deeph",
                    "overrides": deeph_seed,
                },
            ]
        )
    payload["training_sweep"] = {
        "enabled": True,
        "max_runs": len(manual_runs),
        "apply_to_datasets": ["all"],
        "error_policy": "continue_on_error",
        "search_policy": {"strategy": "manual", "random_seed": 0},
        "manual_runs": manual_runs,
    }
    return payload


def train_models(config: dict[str, Any], root: Path, *, resume: bool) -> dict[str, Any]:
    nested = read_json(root / "training_data/nested_dataset_manifest.json")
    if nested.get("status") != "completed":
        raise RuntimeError("Generate and validate all nested MD datasets before training")
    rows = []
    for size in config["training_sizes"]:
        disk = disk_headroom(root)
        if disk["status"] != "safe":
            rows.append({"size": int(size), "status": "resource_blocked", "resource": "disk", "disk": disk})
            break
        payload = training_payload(config, root, int(size))
        control = root / "training" / f"n{int(size)}" / "control"
        payload_path = control / "payload.json"
        status_json = control / "status.json"
        runner_manifest = control / "runner_manifest.json"
        write_json(payload_path, payload)
        previous = read_json(status_json).get("status", {})
        if resume and previous.get("returncode") == 0:
            rows.append({"size": size, "status": "reused", "payload": str(payload_path)})
            continue
        command = [
            str(REPO_ROOT / ".venv/bin/python"),
            str(SCRIPT_DIR / "run_g2m_deeph_payload_once.py"),
            str(payload_path),
            "--status-json",
            str(status_json),
            "--manifest-json",
            str(runner_manifest),
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        rows.append(
            {
                "size": int(size),
                "status": "completed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "payload": str(payload_path),
                "runner_manifest": str(runner_manifest),
            }
        )
        if completed.returncode != 0:
            break
    result = {
        "status": (
            "resource_blocked"
            if rows and rows[-1]["status"] == "resource_blocked"
            else "completed"
            if len(rows) == len(config["training_sizes"]) and all(
                row["status"] in {"completed", "reused"} for row in rows
            )
            else "failed"
        ),
        "rows": rows,
    }
    write_json(root / "training/training_campaign_manifest.json", result)
    return result


def trained_models(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    jobs = []
    for size in config["training_sizes"]:
        manifest = read_json(
            root / "training" / f"n{int(size)}" / f"n{int(size)}"
            / "sweep/training_sweep_manifest.json"
        )
        for row in manifest.get("runs") or []:
            if not isinstance(row, dict) or row.get("status") != "completed":
                continue
            match = re.search(r"seed(\d+)", str(row.get("config_id") or ""))
            if row.get("model") not in {"graph2mat", "deeph"} or not match:
                continue
            jobs.append(
                {
                    "training_size": int(size),
                    "seed": int(match.group(1)),
                    "model": row["model"],
                    "run_root": str(row["run_root"]),
                    "training_record": row,
                }
            )
    expected = len(config["training_sizes"]) * len(active_model_seeds(config)) * 2
    if len(jobs) != expected:
        raise RuntimeError(f"Expected {expected} completed trained models, found {len(jobs)}")
    return jobs


def _link_exact_overlap_inputs(overlap_root: Path, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "overlaps.h5",
        "lat.dat",
        "rlat.dat",
        "site_positions.dat",
        "element.dat",
        "orbital_types.dat",
        "R_list.dat",
        "info.json",
    ):
        source = overlap_root / name
        destination = work_dir / name
        if not source.is_file():
            raise RuntimeError(f"Missing exact-overlap artifact: {source}")
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise RuntimeError(f"Prediction input collision: {destination}")
            continue
        os.symlink(os.path.relpath(source, work_dir), destination)


def _graph2mat_checkpoint(run_root: Path) -> Path:
    training = run_root / "graph2mat/training"
    manifest = read_json(training / "checkpoint_manifest.json")
    path = Path(str(manifest.get("checkpoint_path") or manifest.get("source_checkpoint_path") or ""))
    if path and not path.is_absolute():
        path = training / path
    if not path.is_file():
        raise RuntimeError(f"Missing Graph2Mat checkpoint under {training}")
    return path


def _prepare_deeph_local_coordinates(root: Path, overlap_root: Path, *, resume: bool) -> Path:
    common = root / "predictions/deeph_common"
    rc = common / "rc.h5"
    if resume and rc.is_file():
        return rc
    _link_exact_overlap_inputs(overlap_root, common)
    config_path = common / "local_coordinates.ini"
    render_inference_config(
        config_path,
        work_dir=common,
        trained_model_dir=common,
        python_interpreter=str(REPO_ROOT.parent / "DeepH-pack/.venv/bin/python"),
        interface="openmx",
        task=[2],
        disable_cuda=True,
        device="cpu",
        huge_structure=True,
        restore_blocks_py=True,
        radius=-1.0,
        # DeepH calls this "from DFT", but task 2 reads only exact overlaps.h5
        # as the neighbour list; it never reads or requires a target Hamiltonian.
        create_from_dft=True,
    )
    completed = subprocess.run(
        [
            str(REPO_ROOT.parent / "DeepH-pack/.venv/bin/deeph-inference"),
            "--config",
            str(config_path),
        ],
        cwd=REPO_ROOT.parent / "DeepH-pack",
        text=True,
        capture_output=True,
        check=False,
    )
    (common / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (common / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not rc.is_file():
        raise RuntimeError(f"DeepH local-coordinate generation failed; see {common / 'stderr.log'}")
    return rc


def predict_models(config: dict[str, Any], root: Path, *, resume: bool) -> dict[str, Any]:
    overlap_root = root / "overlap"
    target_manifest = root / "target/splits/test_manifest.csv"
    orb_indx = next(overlap_root.glob("*.ORB_INDX"), None)
    if orb_indx is None:
        raise RuntimeError("Magic-angle ORB_INDX is unavailable")
    rows = []
    deeph_rc: Path | None = None
    for job in trained_models(config, root):
        disk = disk_headroom(root)
        if disk["status"] != "safe":
            rows.append({**job, "status": "resource_blocked", "resource": "disk", "disk": disk})
            break
        model = str(job["model"])
        size = int(job["training_size"])
        seed = int(job["seed"])
        # ponytail: dense DeepH Hamiltonian reconstruction OOMs on the 11k-atom
        # moire target regardless of training size (needs ~40GB, GPU has 32GB).
        # Skip DeepH predict jobs so graph2mat can still advance; switch to the
        # sparse solver path (run_deeph_sparse_spectrum.py) to lift this.
        if model == "deeph" and os.environ.get("MOIRE_CAMPAIGN_SKIP_DEEPH") == "1":
            rows.append({**job, "status": "skipped_dense_oom"})
            continue
        output = root / "predictions" / model / f"n{size}" / f"seed{seed}"
        work = output / "solver_input"
        prediction_h5 = work / "hamiltonians_pred.h5"
        manifest_path = output / "prediction_manifest.json"
        if resume and read_json(manifest_path).get("status") == "completed" and prediction_h5.is_file():
            rows.append(read_json(manifest_path))
            continue
        _link_exact_overlap_inputs(overlap_root, work)
        run_root = Path(job["run_root"])
        started = time.time()
        guard_payload = {
            "performance": {
                "compute_accelerator": "gpu",
                f"{model}_min_free_gpu_memory_mb": 18000 if model == "graph2mat" else 24000,
                "gpu_memory_wait_timeout_seconds": 3600,
                "gpu_memory_poll_seconds": 15,
            }
        }
        if model == "graph2mat":
            checkpoint = _graph2mat_checkpoint(run_root)
            raw = output / "graph2mat_raw"
            command = [
                "/usr/bin/time",
                "-v",
                "-o",
                str(output / "time-v.txt"),
                str(REPO_ROOT / ".venv/bin/python"),
                str(SCRIPT_DIR / "predict_model_on_dataset.py"),
                "--checkpoint",
                str(checkpoint),
                "--train-method",
                "md",
                "--test-set",
                "magic_angle_31_30",
                "--test-manifest",
                str(target_manifest),
                "--output-dir",
                str(raw),
                "--basis-files",
                str(root / "target/material_basis/*.ion.xml"),
                "--matrix-component-policy",
                "h_only",
                "--n-matrix-components",
                "1",
                "--accelerator",
                "gpu",
                "--precision",
                "bf16-mixed",
                "--mace-node-chunk-size",
                "512",
                "--mace-edge-chunk-size",
                "8192",
                "--graph2mat-edge-chunk-size",
                "8192",
                "--graph2mat-node-chunk-size",
                "512",
                "--loader-threads",
                "1",
                "--no-store-in-memory",
                "--torch-float32-matmul-precision",
                "high",
            ]
            with _exclusive_gpu_training(guard_payload) as gpu_lock:
                gpu_guard = _wait_for_free_gpu_memory(guard_payload, model)
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env={
                        **os.environ,
                        "NVIDIA_TF32_OVERRIDE": "0",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    },
                    text=True,
                    capture_output=True,
                    check=False,
                )
            (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode == 0:
                predicted_hsx = raw / "predicted_hamiltonians/moire_0/ML_prediction.HSX"
                export_siesta_hamiltonian(predicted_hsx, orb_indx, prediction_h5)
            model_source = str(checkpoint)
        else:
            if deeph_rc is None:
                deeph_rc = _prepare_deeph_local_coordinates(root, overlap_root, resume=resume)
            rc_link = work / "rc.h5"
            if not rc_link.exists():
                os.symlink(os.path.relpath(deeph_rc, work), rc_link)
            common_graph = deeph_rc.parent / "graph.pkl"
            graph_link = work / "graph.pkl"
            if common_graph.is_file() and not graph_link.exists():
                os.symlink(os.path.relpath(common_graph, work), graph_link)
            config_path = output / "deeph_inference.ini"
            render_inference_config(
                config_path,
                work_dir=work,
                trained_model_dir=run_root / "deeph/train",
                python_interpreter=str(REPO_ROOT.parent / "DeepH-pack/.venv/bin/python"),
                interface="openmx",
                task=[3, 4],
                disable_cuda=False,
                device="cuda:0",
                huge_structure=True,
                restore_blocks_py=True,
                radius=-1.0,
                create_from_dft=True,
            )
            command = [
                "/usr/bin/time",
                "-v",
                "-o",
                str(output / "time-v.txt"),
                str(REPO_ROOT.parent / "DeepH-pack/.venv/bin/deeph-inference"),
                "--config",
                str(config_path),
            ]
            with _exclusive_gpu_training(guard_payload) as gpu_lock:
                gpu_guard = _wait_for_free_gpu_memory(guard_payload, model)
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT.parent / "DeepH-pack",
                    env={
                        **os.environ,
                        "DEEPH_HUGE_STRUCTURE_CHUNK_ROWS": str(DEEPH_HUGE_STRUCTURE_CHUNK_ROWS),
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    },
                    text=True,
                    capture_output=True,
                    check=False,
                )
            (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode == 0 and graph_link.is_file() and not common_graph.exists():
                graph_link.replace(common_graph)
            model_source = str(run_root / "deeph/train")
        resource_blocked = process_resource_blocked(completed)
        row = {
            **job,
            "status": (
                "completed"
                if completed.returncode == 0 and prediction_h5.is_file()
                else "resource_blocked"
                if resource_blocked
                else "failed"
            ),
            "returncode": completed.returncode,
            "model_source": model_source,
            "solver_input": str(work),
            "hamiltonian_pred": str(prediction_h5),
            "elapsed_seconds": time.time() - started,
            "resources": parse_time_v(output / "time-v.txt"),
            "gpu_training_lock": gpu_lock,
            "gpu_memory_guard": gpu_guard,
            "target_reference_hamiltonian_used": False,
            "neighbour_topology_source": "SIESTA_exact_PAO_overlap_blocks",
        }
        if model == "graph2mat":
            row["inference"] = {
                "backend": "e3nn_original_checkpoint",
                "mace_symmetric_contraction_node_chunk_size": 512,
                "mace_message_passing_edge_chunk_size": 8192,
                "graph2mat_readout_edge_chunk_size": 8192,
                "graph2mat_readout_node_chunk_size": 512,
                "precision": "bf16-mixed",
                "tf32_enabled": False,
                "chunking_validation": {
                    "system_atoms": 6,
                    "precision": "fp32_no_tf32",
                    "relative_frobenius": 2.8971418190068316e-7,
                    "maximum_absolute_difference_eV": 5.722045899325678e-6,
                },
            }
        else:
            row["deeph_create_from_DFT_semantics"] = (
                "DeepH upstream name for reading neighbour keys from overlaps.h5; "
                "no target DFT Hamiltonian is read or required"
            )
            row["inference"] = {
                "backend": "deeph_huge_structure_streaming",
                "subgraph_chunk_rows": DEEPH_HUGE_STRUCTURE_CHUNK_ROWS,
                "subgraph_storage": "cpu_with_per_chunk_gpu_transfer",
                "precision": "fp32",
            }
        write_json(manifest_path, row)
        rows.append(row)
        if row["status"] != "completed":
            break
    expected = len(config["training_sizes"]) * len(active_model_seeds(config)) * 2
    result = {
        "status": (
            "resource_blocked"
            if rows and rows[-1]["status"] == "resource_blocked"
            else "completed"
            if len(rows) == expected
            and all(row["status"] in {"completed", "skipped_dense_oom"} for row in rows)
            else "failed"
        ),
        "rows": rows,
    }
    write_json(root / "predictions/prediction_campaign_manifest.json", result)
    return result


def neutrality_estimate(root: Path) -> dict[str, Any]:
    path = root / "neutrality_estimate.json"
    existing = read_json(path)
    if existing.get("status") == "estimated":
        return existing
    dataset = REPO_ROOT / "Comparison/datasets/graphene_hbn_bilayer_md_nested/n480"
    frozen = read_json(dataset / "frozen_split_manifest.json")
    values = []
    sources = []
    import sisl

    for row in frozen.get("rows") or []:
        if row.get("split") not in {"validation", "test"}:
            continue
        matrix = Path(str(row.get("reference_tshs_path") or row.get("hamiltonian_path") or ""))
        if not matrix.is_file():
            continue
        try:
            value = float(sisl.get_sile(str(matrix)).read_fermi_level())
        except Exception:
            continue
        values.append(value)
        sources.append(str(matrix))
    if not values:
        raise RuntimeError("Cannot estimate neutrality: no small-cell SIESTA Fermi levels are readable")
    import numpy as np

    array = np.asarray(values)
    median = float(np.median(array))
    result = {
        "status": "estimated",
        "energy_eV": median,
        "source": "median_small_cell_SIESTA_Fermi_levels",
        "target_SCF_used": False,
        "sample_count": len(values),
        "median_absolute_deviation_eV": float(np.median(np.abs(array - median))),
        "minimum_eV": float(array.min()),
        "maximum_eV": float(array.max()),
        "source_files": sources,
        "limitation": (
            "Energy gauge estimated from training cells; no magic-angle chemical potential "
            "or target Hamiltonian reference is claimed."
        ),
    }
    write_json(path, result)
    return result


def spectrum_tiers(config: dict[str, Any], prediction: dict[str, Any], job: str) -> list[str]:
    representative = (
        int(prediction["training_size"]) == max(int(value) for value in config["training_sizes"])
        and int(prediction["seed"]) == min(active_model_seeds(config))
    )
    if job == "bands":
        return ["tier_b"] if representative else ["tier_a"]
    if job == "dos":
        return ["tier_c"] if representative else []
    raise RuntimeError(f"Unknown spectrum job: {job}")


def solve_spectra(
    config: dict[str, Any],
    root: Path,
    *,
    resume: bool,
    job: str,
) -> dict[str, Any]:
    predictions = read_json(root / "predictions/prediction_campaign_manifest.json")
    if predictions.get("status") != "completed":
        raise RuntimeError("Complete all magic-angle Hamiltonian predictions before sparse solving")
    neutrality = neutrality_estimate(root)
    settings = config["solver"]
    active_models = set(settings.get("active_models") or ("graph2mat", "deeph"))
    active_training_sizes = {
        int(value) for value in settings.get("active_training_sizes") or config["training_sizes"]
    }
    rows = []
    for prediction in predictions["rows"]:
        model = str(prediction["model"])
        size = int(prediction["training_size"])
        seed = int(prediction["seed"])
        if model not in active_models or size not in active_training_sizes:
            continue
        spectrum_root = root / "spectra" / model / f"n{size}" / f"seed{seed}"
        combined_path = spectrum_root / "solver_manifest.json"
        combined = read_json(combined_path)
        if resume and combined.get(f"{job}_status") == "completed":
            rows.append(combined)
            continue
        if prediction.get("status") == "skipped_dense_oom":
            rows.append({**prediction, f"{job}_status": "skipped_dense_oom"})
            continue
        tiers = spectrum_tiers(config, prediction, job)
        if not tiers:
            combined.update(
                {
                    "status": "completed",
                    "scientific_status": "prediction_only",
                    "model": model,
                    "training_size": size,
                    "seed": seed,
                    "neutrality_estimate": neutrality,
                    f"{job}_status": "not_scheduled_by_tier_policy",
                    f"{job}_policy": "tier_c_only_nmax_representative_seed",
                    "dense_large_cell_fallback_used": False,
                    "identity_overlap_used": False,
                }
            )
            write_json(combined_path, combined)
            aggregate(config, root)
            rows.append(combined)
            continue
        tier_results = {}
        for tier in tiers:
            existing_tier = (combined.get("tier_results") or {}).get(tier)
            persisted_tier = read_json(spectrum_root / tier / "solver_manifest.json")
            if persisted_tier.get("status") == "completed":
                existing_tier = {**persisted_tier, "tier": tier}
            if resume and isinstance(existing_tier, dict) and existing_tier.get("status") == "completed":
                tier_results[tier] = existing_tier
                continue
            result = run_sparse_spectrum(
                Path(prediction["solver_input"]),
                spectrum_root / tier,
                job="band" if job == "bands" else "dos",
                fermi_level=float(neutrality["energy_eV"]),
                num_bands=int(
                    settings["tier_a_bands"]
                    if tier == "tier_a"
                    else settings["tier_b_bands"]
                    if tier == "tier_b"
                    else settings["tier_c_bands"]
                ),
                points_per_segment=(
                    1 if tier == "tier_a" else int(settings["tier_b_points_per_segment"])
                ),
                kmesh=tuple(int(value) for value in settings["tier_c_kmesh"]),
                environment_path=root
                / "solver"
                / (
                    "environment_gpu_cudss.json"
                    if settings.get("compute_backend") == "gpu_cudss"
                    else "environment.json"
                ),
                backend=settings.get("compute_backend", "cpu_mkl_pardiso"),
                gpu_hybrid_memory=bool(settings.get("gpu_hybrid_memory", True)),
                gpu_memory_limit_gib=float(settings.get("gpu_memory_limit_gib", 28)),
            )
            result["tier"] = tier
            write_json(spectrum_root / tier / "solver_manifest.json", result)
            tier_results[tier] = result
            if result["status"] != "completed":
                break
        result = tier_results[tiers[-1]] if tiers[-1] in tier_results else tier_results[tiers[0]]
        completed_tiers = all(tier_results.get(tier, {}).get("status") == "completed" for tier in tiers)
        job_status = (
            "completed"
            if completed_tiers
            else "resource_blocked"
            if any(item.get("status") == "resource_blocked" for item in tier_results.values())
            else "failed"
        )
        visible_result = tier_results.get("tier_c") or result
        if job == "bands":
            visible_result = tier_results.get("tier_b") or tier_results.get("tier_a") or result
            if not visible_result.get("bands"):
                visible_result = tier_results.get("tier_a") or visible_result
        combined.update(
            {
                "status": job_status,
                "scientific_status": "prediction_only",
                "model": model,
                "training_size": size,
                "seed": seed,
                "neutrality_estimate": neutrality,
                "tier_results": {**(combined.get("tier_results") or {}), **tier_results},
                f"{job}_status": job_status,
                f"{job}_resources": {
                    tier: item.get("resources") for tier, item in tier_results.items()
                },
                "backend_requested": visible_result.get("backend_requested"),
                "backend_effective": visible_result.get("backend_effective"),
                "gpu_hardware": visible_result.get("gpu_hardware"),
                "cudss_timings": visible_result.get("cudss_timings"),
                "gpu_observations": visible_result.get("gpu_observations"),
                "dense_large_cell_fallback_used": False,
                "identity_overlap_used": False,
            }
        )
        if job == "bands" and visible_result.get("bands"):
            combined["bands"] = visible_result["bands"]
            combined["visible_band_tier"] = visible_result["tier"]
        if job == "dos" and visible_result.get("low_energy_dos"):
            combined["low_energy_dos"] = visible_result["low_energy_dos"]
            combined["visible_dos_tier"] = visible_result["tier"]
        write_json(combined_path, combined)
        aggregate(config, root)
        rows.append(combined)
        if job_status == "resource_blocked":
            break
        if job_status != "completed":
            break
    expected = len(predictions["rows"])
    if rows and rows[-1].get(f"{job}_status") == "resource_blocked":
        status = "resource_blocked"
    else:
        status = "completed" if len(rows) == expected and all(
            row.get(f"{job}_status") in {"completed", "not_scheduled_by_tier_policy", "skipped_dense_oom"}
            for row in rows
        ) else "failed"
    result = {"status": status, "job": job, "rows": rows}
    write_json(root / f"spectra/{job}_campaign_manifest.json", result)
    return result


def aggregate(config: dict[str, Any], root: Path) -> dict[str, Any]:
    overlap_rows = [
        read_json(path)
        for path in sorted((root / "overlap_benchmarks").glob("atoms_*/overlap_manifest.json"))
    ]
    spectra = []
    for manifest in sorted((root / "spectra").glob("*/*/*/solver_manifest.json")):
        row = read_json(manifest)
        row["manifest_path"] = str(manifest)
        if not row.get("bands"):
            for tier in ("tier_b", "tier_a"):
                tier_manifest = manifest.parent / tier / "solver_manifest.json"
                tier_result = read_json(tier_manifest)
                if tier_result.get("status") == "completed" and tier_result.get("bands"):
                    row["bands"] = tier_result["bands"]
                    row["bands_status"] = "completed"
                    row["visible_band_tier"] = tier
                    row["manifest_path"] = str(tier_manifest)
                    row["status"] = "completed"
                    break
        if not row.get("low_energy_dos"):
            tier_manifest = manifest.parent / "tier_c/solver_manifest.json"
            tier_result = read_json(tier_manifest)
            if tier_result.get("status") == "completed" and tier_result.get("low_energy_dos"):
                row["low_energy_dos"] = tier_result["low_energy_dos"]
                row["dos_status"] = "completed"
                row["visible_dos_tier"] = "tier_c"
                row["manifest_path"] = str(tier_manifest)
                row["status"] = "completed"
        band_rows = row.get("bands") if isinstance(row.get("bands"), list) else []
        energies = [float(item["energy_aligned_eV"]) for item in band_rows if item.get("energy_aligned_eV") is not None]
        if energies:
            occupied = [value for value in energies if value <= 0]
            empty = [value for value in energies if value > 0]
            if occupied and empty:
                row["estimated_gap_meV"] = 1000.0 * max(0.0, min(empty) - max(occupied))
            by_k: dict[int, list[float]] = {}
            for item in band_rows:
                by_k.setdefault(int(item["k_index"]), []).append(float(item["energy_aligned_eV"]))
            central = [
                value
                for values in by_k.values()
                for value in sorted(values, key=abs)[:4]
            ]
            if central:
                row["flat_band_width_meV"] = 1000.0 * (max(central) - min(central))
                row["flat_band_width_policy"] = "four_shift_invert_states_closest_to_estimated_neutrality_per_k"
        spectra.append(row)
    by_size_seed: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in spectra:
        by_size_seed.setdefault(
            (int(row.get("training_size") or 0), int(row.get("seed") or 0)),
            {},
        )[str(row.get("model"))] = row
    for models in by_size_seed.values():
        if set(models) != {"graph2mat", "deeph"}:
            continue
        left = models["graph2mat"].get("bands") or []
        right = models["deeph"].get("bands") or []
        left_values = sorted(
            (int(item["k_index"]), float(item["energy_aligned_eV"])) for item in left
        )
        right_values = sorted(
            (int(item["k_index"]), float(item["energy_aligned_eV"])) for item in right
        )
        if len(left_values) == len(right_values) and left_values:
            import numpy as np

            difference = np.asarray([item[1] for item in left_values]) - np.asarray(
                [item[1] for item in right_values]
            )
            consistency = 1000.0 * float(np.sqrt(np.mean(difference**2)))
            for row in models.values():
                row["g2m_deeph_consistency_meV"] = consistency
                row["consistency_interpretation"] = "model_to_model_difference_not_target_error"
    summary = {
        "campaign_kind": config["campaign_kind"],
        "target_contract": config["target_contract"],
        "scientific_status": "prediction_only",
        "thermal_protocol": "Verlet_NVE_initialized_at_150_300_450_K_not_thermostatted",
        "target_reference_metrics_available": False,
        "mae": None,
        "relative_frobenius": None,
        "reference_validation": {
            "overlap_only_vs_full": read_json(root / "reference_validation/overlap_only_vs_full.json"),
            "synthetic_sparse_dense": read_json(root / "solver/synthetic_validation/validation.json"),
            "physical_atoms6_sparse_dense": read_json(
                root / "solver/physical_validation_atoms6/validation.json"
            ),
            "physical_atoms6_cpu_gpu": read_json(
                root / "solver/physical_validation_atoms6/gpu_validation.json"
            ),
        },
        "overlap_benchmarks": overlap_rows,
        "spectra": spectra,
        "coverage": read_json(root / "local_environment_coverage.json"),
        "neutrality_estimate": read_json(root / "neutrality_estimate.json"),
        "target": read_json(root / "target/moire_geometry.json"),
        "nested_datasets": read_json(root / "training_data/nested_dataset_manifest.json"),
        "training": read_json(root / "training/training_campaign_manifest.json"),
        "predictions": read_json(root / "predictions/prediction_campaign_manifest.json"),
        "updated_at": now(),
    }
    write_json(root / "summary/spectral_results.json", summary)
    return summary


def stop_requested(root: Path) -> bool:
    return bool(read_json(status_path(root)).get("stop_requested"))


def run_action(action: str, config: dict[str, Any], root: Path, *, resume: bool) -> dict[str, Any]:
    functions = {
        "generate-training-data": generate_training_data,
        "train": train_models,
        "build-target": build_target,
        "build-overlap": build_overlap,
        "predict": predict_models,
    }
    if action in functions:
        stage_state(root, action, "running")
        result = functions[action](config, root, resume=resume)
        stage_state(root, action, result.get("status", "completed"), result=result)
        return result
    if action == "aggregate":
        stage_state(root, action, "running")
        result = aggregate(config, root)
        stage_state(root, action, "completed", result=result)
        return result
    if action in {"solve-bands", "solve-dos"}:
        stage_state(root, action, "running")
        result = solve_spectra(
            config,
            root,
            resume=resume,
            job="bands" if action == "solve-bands" else "dos",
        )
        stage_state(root, action, result["status"], result=result)
        return result
    raise RuntimeError(f"Unknown action: {action}")


def run_pipeline(config: dict[str, Any], root: Path, *, resume: bool) -> dict[str, Any]:
    results = {}
    for stage in (
        "build-target",
        "build-overlap",
        "generate-training-data",
        "train",
        "predict",
        "solve-bands",
        "solve-dos",
    ):
        if stop_requested(root):
            set_status(root, running=False, state="stopped_after_stage", current_stage=stage)
            return {"status": "stopped_after_stage", "results": results}
        results[stage] = run_action(stage, config, root, resume=resume)
        if results[stage].get("status") == "failed":
            return {"status": "failed", "results": results}
        if results[stage].get("status") == "resource_blocked":
            results["aggregate"] = run_action("aggregate", config, root, resume=resume)
            return {"status": "resource_blocked", "results": results}
    results["aggregate"] = run_action("aggregate", config, root, resume=resume)
    return {"status": "completed", "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "plan",
            "generate-training-data",
            "train",
            "build-target",
            "build-overlap",
            "predict",
            "solve-bands",
            "solve-dos",
            "aggregate",
            "run",
            "resume",
            "status",
            "stop",
        ],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    root = campaign_root(config)

    if args.action == "status":
        print(json.dumps(read_json(status_path(root)), indent=2, sort_keys=True))
        return 0
    if args.action == "stop":
        result = set_status(root, stop_requested=True, state="stop_requested")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.action == "plan":
        result = plan(config, root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    with campaign_lock(root):
        set_status(
            root,
            running=True,
            state="running",
            error=None,
            stop_requested=False,
            pid=os.getpid(),
            command=[str(item) for item in sys.argv],
        )
        try:
            if args.action in {"run", "resume"}:
                result = run_pipeline(config, root, resume=args.action == "resume")
            else:
                result = run_action(args.action, config, root, resume=False)
            set_status(root, running=False, state=result.get("status", "completed"), pid=None)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result.get("status") not in {"failed", "invalid"} else 1
        except Exception as exc:
            set_status(root, running=False, state="failed", error=str(exc), pid=None)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
