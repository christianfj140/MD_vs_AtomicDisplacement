#!/usr/bin/env python3
"""Backend runner for the Graph2Mat vs DeepH benchmark workflow."""

from __future__ import annotations

import json
import os
import csv
import hashlib
import select
import shutil
import subprocess
import sys
import threading
import time
import math
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from joint_artifact_contract import (  # noqa: E402
    CONTRACT_NAME,
    G2M_DEEPH_BENCHMARK_PROFILE,
    DatasetValidationResult,
    validate_dataset,
)
from deeph_config import (  # noqa: E402
    DEEPh_DEFAULT_DATASET_NAME,
    build_deeph_raw_mirror,
    default_deeph_paths,
    deeph_orbital_json_from_raw_mirror,
    ensure_path_inside,
    load_json as load_deeph_json,
    render_inference_config,
    render_preprocess_config,
    render_train_config,
    write_json as write_deeph_json,
)
from deeph_prediction_adapter import adapt_deeph_prediction_sample, write_adapter_manifest  # noqa: E402
from deeph_split_audit import audit_deeph_split  # noqa: E402
from g2m_deeph_metrics import (  # noqa: E402
    aggregate_common_metrics,
    build_common_plot_payload,
    stage_deeph_metric_inputs,
    stage_graph2mat_metric_result,
)
from g2m_deeph_rank_runs import rank_graph2mat_deeph_runs  # noqa: E402
from g2m_deeph_live_metrics import dedupe_metric_rows, live_metric_scaling_rows  # noqa: E402
from g2m_deeph_telemetry import (  # noqa: E402
    GpuTelemetryMonitor,
    compute_gpu_hours,
    optimizer_update_accounting,
    summarize_run_telemetry,
    write_telemetry,
)
from g2m_deeph_early_stopping import (  # noqa: E402
    DeepHEarlyStoppingObserver,
    graph2mat_early_stopping_callbacks,
    parse_early_stopping_policy,
    tensorboard_policy_metadata,
)
from g2m_deeph_budget import BudgetTracker, write_budget_summary  # noqa: E402
from g2m_deeph_test_blindness import (  # noqa: E402
    SEARCH_STAGE,
    build_search_stage_manifest,
    is_final_benchmark_mode,
    protocol_stage_from_payload,
    search_stage_record_fields,
)
from g2m_deeph_topk import (  # noqa: E402
    generate_robust_rerun_plan,
    select_top_configs,
    selection_policy_from_payload,
    write_selection_artifacts,
)
from g2m_deeph_training_sweep import expand_training_sweep  # noqa: E402
from dataset_recipe_helpers import (  # noqa: E402
    dataset_sweep_recipes_from_payload,
    md_dataset_recipes_to_specs,
    recipe_set_hash,
    stable_payload_hash,
    slugify_label,
    validate_split_sizes,
)


DEFAULT_LOG_RESPONSE_LIMIT = 2000
MAX_LOG_RESPONSE_LIMIT = 20000
LOG_HEARTBEAT_SECONDS = 30.0
DEFAULT_DATASETS_ROOT = REPO_ROOT / "Comparison" / "datasets"
DEFAULT_DATASET_ROOT = DEFAULT_DATASETS_ROOT / "graphene_w90_joint"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison" / "results" / "graphene_w90_g2m_deeph_benchmark"
DEFAULT_MD_PIPELINE_CONFIG = REPO_ROOT / "MD" / "pipeline_config.yaml"
DEFAULT_MD_TRAINING_SCRIPT = REPO_ROOT / "MD" / "scripts" / "run_md_training.py"
DEFAULT_MD_PREDICTION_SCRIPT = REPO_ROOT / "MD" / "scripts" / "run_md_prediction.py"
DEFAULT_HAMILTONIAN_METRICS_SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "evaluate_hamiltonian_metrics.py"
DEFAULT_DEEPH_KPOINT_METRICS_SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "evaluate_deeph_kpoint_metrics.py"
DEEPH_PACK_ROOT_ENV = "DEEPH_PACK_ROOT"
DEEPH_CLI_NAMES = ("deeph-preprocess", "deeph-train", "deeph-inference")
DEEPH_SIBLING_REPO_NAME = "DeepH-pack"
DATASET_SWEEP_RUN_MODE = "generate_datasets_only"
FULL_STRICT_PIPELINE_RUN_MODE = "full_strict_pipeline"
METRIC_FAIL_POLICY_FAIL_CLOSED = "fail_closed"
METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY = "diagnostic_only"
TEST_METRICS_LOCKED_MESSAGE = (
    "Final/publicable benchmark mode keeps test predictions and test metrics locked "
    "during search. Run the final_test stage only after validation-based top-k selection."
)
DEEPH_TRAIN_OVERRIDE_KEYS = {
    "epochs",
    "batch_size",
    "learning_rate",
    "seed",
    "optimizer",
    "weight_decay",
    "criterion",
    "atom_fea_len",
    "edge_fea_len",
    "gauss_stop",
    "num_l",
    "if_edge_update",
    "if_lcmp",
    "normalization",
    "atom_update_net",
    "retain_edge_fea",
}

RUNNER_PHASES = (
    "validate_inputs",
    "generate_or_validate_joint_dataset",
    "freeze_splits",
    "training_sweep",
    "graph2mat_train",
    "graph2mat_predict",
    "deeph_preprocess",
    "deeph_train",
    "deeph_predict",
    "common_metrics",
    "ranking",
    "plots_and_summary",
    "complete",
)


@dataclass
class G2MDeepHRunState:
    running: bool = False
    stage: str = "idle"
    returncode: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    stop_requested: bool = False
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    run_id: str | None = None
    dataset_root: str | None = None
    run_root: str | None = None
    graph2mat_config_path: str | None = None
    graph2mat_training_dir: str | None = None
    deeph_manifest_path: str | None = None
    deeph_processed_dir: str | None = None
    deeph_save_dir: str | None = None
    benchmark_manifest_path: str | None = None
    frozen_split_manifest_path: str | None = None
    training_sweep_status: dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph2MatBenchmarkContext:
    dataset_root: Path
    run_root: Path
    graph2mat_root: Path
    training_dir: Path
    prediction_structs_dir: Path
    config_path: Path
    graph2mat_config_path: Path
    graph2mat_manifest_path: Path
    frozen_split_manifest_path: Path
    benchmark_dataset_manifest_path: Path
    runs_json_path: Path
    runs_json_counts: dict[str, int]
    train_glob: str
    validation_glob: str
    predict_glob: str
    output_file: str
    test_sample_ids: list[str]
    split_hash: str | None
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "run_root": str(self.run_root),
            "graph2mat_root": str(self.graph2mat_root),
            "training_dir": str(self.training_dir),
            "prediction_structs_dir": str(self.prediction_structs_dir),
            "config_path": str(self.config_path),
            "graph2mat_config_path": str(self.graph2mat_config_path),
            "graph2mat_manifest_path": str(self.graph2mat_manifest_path),
            "frozen_split_manifest_path": str(self.frozen_split_manifest_path),
            "benchmark_dataset_manifest_path": str(self.benchmark_dataset_manifest_path),
            "runs_json_path": str(self.runs_json_path),
            "runs_json_counts": dict(self.runs_json_counts),
            "train_glob": self.train_glob,
            "validation_glob": self.validation_glob,
            "predict_glob": self.predict_glob,
            "output_file": self.output_file,
            "test_sample_ids": list(self.test_sample_ids),
            "split_hash": self.split_hash,
            "dry_run": self.dry_run,
        }


@dataclass
class DeepHBenchmarkContext:
    root: Path
    raw_dir: Path
    processed_dir: Path
    graph_dir: Path
    save_dir: Path
    inference_dir: Path
    preprocess_config: Path
    train_config: Path
    inference_configs: list[Path]
    inference_work_dirs: list[Path]
    manifest_path: Path
    deeph_discovery: dict[str, Any]
    split_audit_path: Path
    split_audit_csv_path: Path
    split_hash: str | None
    raw_mirror: dict[str, Any]
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "raw_dir": str(self.raw_dir),
            "processed_dir": str(self.processed_dir),
            "graph_dir": str(self.graph_dir),
            "save_dir": str(self.save_dir),
            "inference_dir": str(self.inference_dir),
            "preprocess_config": str(self.preprocess_config),
            "train_config": str(self.train_config),
            "inference_configs": [str(path) for path in self.inference_configs],
            "inference_work_dirs": [str(path) for path in self.inference_work_dirs],
            "manifest_path": str(self.manifest_path),
            "deeph_discovery": self.deeph_discovery,
            "split_audit_path": str(self.split_audit_path),
            "split_audit_csv_path": str(self.split_audit_csv_path),
            "split_hash": self.split_hash,
            "raw_mirror": self.raw_mirror,
            "dry_run": self.dry_run,
        }


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "si", "sí"}
    return default


def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise RuntimeError(f"{field_name} must be a positive integer.")
    return parsed


def _optional_int_value(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "sin estimacion"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_progress_value(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 1:
        return f"{number:.4f}"
    return f"{number:.6f}"


def _lightning_training_progress(training_dir: Path) -> str | None:
    """Return compact Graph2Mat/Lightning progress from TensorBoard events."""

    try:
        event_files = sorted(
            (training_dir / "lightning_logs").rglob("events.out.tfevents.*"),
            key=lambda path: path.stat().st_mtime,
        )
        if not event_files:
            return None
        config = {}
        config_path = training_dir / "config.yaml"
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config = loaded
        max_epochs = (config.get("trainer") or {}).get("max_epochs")
        from tensorboard.backend.event_processing import event_accumulator

        accumulator = event_accumulator.EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags().get("scalars", []))

        def latest_scalar(tag: str) -> tuple[int | None, float | None]:
            if tag not in scalar_tags:
                return None, None
            values = accumulator.Scalars(tag)
            if not values:
                return None, None
            item = values[-1]
            return int(item.step), float(item.value)

        epoch_step, epoch_value = latest_scalar("epoch")
        train_epoch_step, train_epoch_loss = latest_scalar("train_loss_epoch")
        _, train_step_loss = latest_scalar("train_loss_step")
        val_step, val_loss = latest_scalar("val_loss")
        _, val_node = latest_scalar("val_node_smooth_l1")
        _, val_edge = latest_scalar("val_edge_smooth_l1")
        _, beta = latest_scalar("val_smooth_l1_beta")
        step = max(
            [item for item in (epoch_step, train_epoch_step, val_step) if item is not None],
            default=None,
        )
        pieces: list[str] = []
        if epoch_value is not None:
            epoch_display = str(int(epoch_value))
            if max_epochs not in (None, ""):
                epoch_display += f"/{max_epochs}"
            pieces.append(f"epoch {epoch_display}")
        if step is not None:
            pieces.append(f"step {step}")
        if train_epoch_loss is not None:
            pieces.append(f"train_epoch {_format_progress_value(train_epoch_loss)}")
        if train_step_loss is not None:
            pieces.append(f"train_step {_format_progress_value(train_step_loss)}")
        if val_loss is not None:
            pieces.append(f"val {_format_progress_value(val_loss)}")
        if val_node is not None:
            pieces.append(f"val_node {_format_progress_value(val_node)}")
        if val_edge is not None:
            pieces.append(f"val_edge {_format_progress_value(val_edge)}")
        if beta is not None:
            pieces.append(f"beta {_format_progress_value(beta)}")
        checkpoints = sorted(
            (training_dir / "lightning_logs").rglob("checkpoints/*.ckpt"),
            key=lambda path: path.stat().st_mtime,
        )
        if checkpoints:
            pieces.append(f"ckpt {checkpoints[-1].name}")
        return " | ".join(pieces) if pieces else None
    except Exception:
        return None


def _performance_settings_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("performance") if isinstance(payload.get("performance"), dict) else {}
    accelerator = str(
        raw.get("compute_accelerator")
        or payload.get("compute_accelerator")
        or "cpu"
    ).strip().lower()
    if accelerator not in {"cpu", "gpu", "auto"}:
        raise RuntimeError(f"performance.compute_accelerator must be cpu, gpu or auto, got {accelerator!r}.")
    settings: dict[str, Any] = dict(raw)
    settings["compute_accelerator"] = accelerator
    for key in (
        "batch_size",
        "graph2mat_log_every_n_steps",
        "graph2mat_check_val_every_n_epoch",
        "graph2mat_checkpoint_every_n_epochs",
        "max_parallel_graph2mat_training_jobs",
        "max_parallel_deeph_training_jobs",
        "torch_num_threads",
        "omp_num_threads",
        "mkl_num_threads",
        "openblas_num_threads",
        "numexpr_num_threads",
    ):
        if key in settings:
            settings[key] = _optional_positive_int(settings.get(key), field_name=f"performance.{key}")
    if "store_in_memory" in settings:
        settings["store_in_memory"] = _parse_bool(settings.get("store_in_memory"), False)
    precision = settings.get("torch_float32_matmul_precision")
    if precision in (None, "", "null"):
        settings["torch_float32_matmul_precision"] = None
    elif str(precision).strip().lower() not in {"high", "medium"}:
        raise RuntimeError("performance.torch_float32_matmul_precision must be high, medium or null.")
    elif precision is not None:
        settings["torch_float32_matmul_precision"] = str(precision).strip().lower()
    mixed_precision = settings.get("torch_mixed_precision") or settings.get("graph2mat_precision")
    if mixed_precision in (None, "", "null"):
        settings["torch_mixed_precision"] = None
    else:
        mixed_precision = str(mixed_precision).strip().lower()
        if mixed_precision not in {"32-true", "16-mixed", "bf16-mixed"}:
            raise RuntimeError("performance.torch_mixed_precision must be 32-true, 16-mixed, bf16-mixed or null.")
        settings["torch_mixed_precision"] = mixed_precision
    if "graph2mat_require_cuequivariance" in settings:
        settings["graph2mat_require_cuequivariance"] = _parse_bool(
            settings.get("graph2mat_require_cuequivariance"),
            False,
        )
    return settings


def _graph2mat_checkpoint_callbacks(every_n_epochs: int) -> list[dict[str, Any]]:
    every_n_epochs = max(1, int(every_n_epochs))
    return [
        {
            "class_path": "ModelCheckpoint",
            "init_args": {
                "monitor": "step",
                "mode": "max",
                "filename": "last-{step:02d}",
                "save_last": True,
                "auto_insert_metric_name": False,
                "every_n_epochs": every_n_epochs,
            },
        },
        {
            "class_path": "ModelCheckpoint",
            "init_args": {
                "monitor": "val_loss",
                "mode": "min",
                "filename": "best-{step:02d}",
                "auto_insert_metric_name": False,
                "every_n_epochs": every_n_epochs,
            },
        },
    ]


def _apply_common_early_stopping_to_graph2mat_config(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    policy = parse_early_stopping_policy(payload)
    if policy is None:
        return None
    training = config.setdefault("training", {})
    trainer = training.setdefault("trainer", {})
    trainer["max_epochs"] = policy.max_epochs
    callbacks = list(trainer.get("callbacks") or [])
    callbacks.extend(graph2mat_early_stopping_callbacks(policy))
    trainer["callbacks"] = callbacks
    config["common_early_stopping"] = policy.to_dict()
    return policy.to_dict()


def _apply_performance_to_graph2mat_config(config: dict[str, Any], performance: dict[str, Any]) -> None:
    if not performance:
        return
    config["performance"] = dict(performance)
    training = config.setdefault("training", {})
    trainer = training.setdefault("trainer", {})
    data = training.setdefault("data", {})
    if performance.get("compute_accelerator"):
        trainer["accelerator"] = str(performance["compute_accelerator"])
    if performance.get("torch_mixed_precision"):
        trainer["precision"] = str(performance["torch_mixed_precision"])
    if performance.get("graph2mat_log_every_n_steps") is not None:
        trainer["log_every_n_steps"] = int(performance["graph2mat_log_every_n_steps"])
    if performance.get("graph2mat_check_val_every_n_epoch") is not None:
        trainer["check_val_every_n_epoch"] = int(performance["graph2mat_check_val_every_n_epoch"])
    if performance.get("graph2mat_checkpoint_every_n_epochs") is not None:
        trainer["callbacks"] = _graph2mat_checkpoint_callbacks(
            int(performance["graph2mat_checkpoint_every_n_epochs"])
        )
    if performance.get("batch_size") is not None:
        data["batch_size"] = int(performance["batch_size"])
    if performance.get("store_in_memory") is not None:
        data["store_in_memory"] = bool(performance["store_in_memory"])
        config.setdefault("prediction", {}).setdefault("data", {})["store_in_memory"] = bool(
            performance["store_in_memory"]
        )
    if performance.get("torch_float32_matmul_precision"):
        training["torch_float32_matmul_precision"] = performance["torch_float32_matmul_precision"]


def _performance_env(settings: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, env_name in (
        ("omp_num_threads", "OMP_NUM_THREADS"),
        ("mkl_num_threads", "MKL_NUM_THREADS"),
        ("openblas_num_threads", "OPENBLAS_NUM_THREADS"),
        ("numexpr_num_threads", "NUMEXPR_NUM_THREADS"),
        ("torch_num_threads", "TORCH_NUM_THREADS"),
    ):
        value = settings.get(key)
        if value not in (None, "", "null"):
            env[env_name] = str(int(value))
    if settings.get("torch_float32_matmul_precision"):
        env["TORCH_FLOAT32_MATMUL_PRECISION"] = str(settings["torch_float32_matmul_precision"])
    return env


def _graph2mat_training_parallelism(payload: dict[str, Any]) -> int:
    settings = _performance_settings_from_payload(payload)
    raw = (
        settings.get("max_parallel_graph2mat_training_jobs")
        or settings.get("max_parallel_training_jobs")
        or 1
    )
    try:
        workers = int(raw)
    except (TypeError, ValueError):
        workers = 1
    return max(1, workers)


def _deeph_training_parallelism(payload: dict[str, Any]) -> int:
    settings = _performance_settings_from_payload(payload)
    raw = settings.get("max_parallel_deeph_training_jobs") or 1
    try:
        workers = int(raw)
    except (TypeError, ValueError):
        workers = 1
    return max(1, workers)


def _training_record_epochs(record: dict[str, Any]) -> Any:
    overrides = record.get("overrides") if isinstance(record.get("overrides"), dict) else {}
    common = record.get("common") if isinstance(record.get("common"), dict) else {}
    return overrides.get("max_epochs") or overrides.get("epochs") or common.get("epochs")


def _training_record_epoch_label(record: dict[str, Any]) -> str:
    epochs = _training_record_epochs(record)
    return f"{epochs} epochs" if epochs not in (None, "") else ""


def _training_record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("model") or ""),
        str(record.get("dataset_id") or ""),
        str(record.get("config_id") or ""),
    )


def _expand_repo_tokens(value: Any) -> str:
    text = str(value)
    return os.path.expandvars(text.replace("${REPO_ROOT}", str(REPO_ROOT)))


def _resolve_repo_path(value: Any, *, field_name: str) -> Path:
    if value in (None, ""):
        raise RuntimeError(f"{field_name} is required.")
    path = Path(_expand_repo_tokens(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def _resolve_optional_repo_path(value: Any, default: Path) -> Path:
    raw = default if value in (None, "") else value
    path = Path(_expand_repo_tokens(raw)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Required JSON file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON file must contain an object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Required YAML file is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML file must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, allow_unicode=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if src.resolve(strict=True) == dst.resolve(strict=True):
                return
        except OSError:
            pass
        try:
            if src.is_file() and dst.exists() and dst.is_file() and _file_sha256(src) == _file_sha256(dst):
                return
        except OSError:
            pass
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except OSError:
        shutil.copy2(src, dst)


def _graph2mat_basis_files_for_dataset(dataset_root: Path) -> list[Path]:
    by_name: dict[str, Path] = {}
    for basis_dir in (
        dataset_root / "material_basis",
        dataset_root / "basis",
        dataset_root / "MD_steps" / "basis",
    ):
        if not basis_dir.exists():
            continue
        for basis_file in sorted(basis_dir.glob("*.ion.xml")):
            by_name.setdefault(basis_file.name, basis_file)
    return [by_name[name] for name in sorted(by_name)]


def _materialize_graph2mat_basis_files(dataset_root: Path, sample_dirs: list[Path]) -> int:
    basis_files = _graph2mat_basis_files_for_dataset(dataset_root)
    if not basis_files:
        return 0
    materialized = 0
    for sample_dir in sample_dirs:
        if not sample_dir.exists() or not sample_dir.is_dir():
            continue
        for basis_file in basis_files:
            _link_or_copy_file(basis_file, sample_dir / basis_file.name)
            materialized += 1
    return materialized


def _glob_matches(pattern: str) -> list[Path]:
    if Path(pattern).is_absolute():
        return sorted(Path("/").glob(pattern[1:]))
    return sorted(Path().glob(pattern))


def _relative_pattern(pattern: str | Path, base_dir: Path) -> str:
    return os.path.relpath(str(pattern), str(base_dir))


def _bounded_log_payload(
    logs: list[str],
    *,
    since: int = 0,
    limit: int | None = DEFAULT_LOG_RESPONSE_LIMIT,
) -> dict[str, Any]:
    total = len(logs)
    requested_since = max(0, int(since or 0))
    start = min(requested_since, total)
    end = total
    effective_limit = None if limit is None or int(limit) <= 0 else min(int(limit), MAX_LOG_RESPONSE_LIMIT)
    dropped_lines = 0
    truncated = False
    if effective_limit is not None and end - start > effective_limit:
        dropped_lines = end - start - effective_limit
        start = end - effective_limit
        truncated = True
    lines = list(logs[start:end])
    if truncated:
        lines.insert(
            0,
            (
                "[G2M-DEEPH] Log truncado en la respuesta web: "
                f"se omitieron {dropped_lines} lineas antiguas.\n"
            ),
        )
    return {
        "offset": total,
        "lines": lines,
        "truncated": truncated,
        "dropped_lines": dropped_lines,
        "returned_from": start,
        "requested_since": requested_since,
        "limit": effective_limit,
    }


def _manifest_path_status(dataset_root: Path) -> dict[str, Any]:
    paths = {
        "artifact_validation": dataset_root / "artifact_validation.json",
        "benchmark_dataset_manifest": dataset_root / "benchmark_dataset_manifest.json",
        "frozen_split_manifest": dataset_root / "frozen_split_manifest.json",
    }
    return {
        key: {
            "path": str(path),
            "exists": path.exists(),
        }
        for key, path in paths.items()
    }


def _missing_key_counts(result: DatasetValidationResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for snapshot in result.snapshots:
        for key in snapshot.missing_required:
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _validation_payload(
    result: DatasetValidationResult,
    *,
    snapshot_root: Path,
    max_snapshots: int = 20,
) -> dict[str, Any]:
    invalid = [snapshot for snapshot in result.snapshots if not snapshot.valid]
    preview = invalid[: max(0, int(max_snapshots))]
    repair_required = bool(result.errors or result.invalid_snapshots or result.repair_required_snapshots)
    return {
        "contract_name": CONTRACT_NAME,
        "benchmark_ready": bool(result.valid),
        "repair_required": repair_required,
        "dataset_root": str(result.dataset_root),
        "snapshot_root": str(snapshot_root),
        "artifact_summary": {
            "total_snapshots": result.total_snapshots,
            "valid_snapshots": result.valid_snapshots,
            "invalid_snapshots": result.invalid_snapshots,
            "repair_required_snapshots": result.repair_required_snapshots,
            "missing_required_counts": _missing_key_counts(result),
            "basis_present": result.basis_present,
            "pseudopotential_provenance_present": result.pseudopotential_provenance_present,
            "material_identity_present": result.material_identity_present,
            "siesta_input_provenance_present": result.siesta_input_provenance_present,
        },
        "errors": list(result.errors),
        "warnings": list(result.warnings),
        "invalid_snapshots_preview": [snapshot.to_dict() for snapshot in preview],
        "invalid_snapshots_preview_count": len(preview),
        "manifest_paths": _manifest_path_status(result.dataset_root),
    }


def _format_validation_not_ready_message(
    validation: dict[str, Any],
    *,
    dataset_mode: str,
    allow_repair: bool,
) -> str:
    summary = validation.get("artifact_summary") or {}
    errors = [str(item) for item in validation.get("errors") or [] if str(item).strip()]
    warnings = [str(item) for item in validation.get("warnings") or [] if str(item).strip()]
    missing = summary.get("missing_required_counts") or {}
    total = int(summary.get("total_snapshots") or 0)
    invalid = int(summary.get("invalid_snapshots") or 0)
    dataset_root = validation.get("dataset_root") or "<unknown>"
    snapshot_root = validation.get("snapshot_root") or dataset_root

    if dataset_mode == "generate_new":
        return (
            "Generate new joint dataset is not implemented in this runner yet. "
            "Create or validate a joint dataset first, then run with "
            f"dataset_mode='reuse_validated'. Target dataset root: {dataset_root}"
        )

    reasons: list[str] = []
    if total <= 0:
        reasons.append(f"no snapshot directories were found under {snapshot_root}")
    if invalid:
        reasons.append(f"{invalid} snapshot(s) are invalid")
    if missing:
        missing_text = ", ".join(f"{key}={value}" for key, value in sorted(missing.items()))
        reasons.append(f"missing required artifacts: {missing_text}")
    for error in errors:
        if total <= 0 and error.startswith("no snapshot directories found under "):
            continue
        if error not in reasons:
            reasons.append(error)
    if not reasons and warnings:
        reasons.extend(warnings)
    if not reasons:
        reasons.append("dataset validation did not satisfy the joint artifact contract")

    suffix = (
        "Repair mode was explicitly requested, but repair execution is not implemented yet."
        if allow_repair
        else "Use a validated joint dataset, or select explicit repair mode only if you accept the expensive repair path."
    )
    return (
        "Graph2Mat/DeepH dataset is not benchmark-ready. "
        + "; ".join(reasons)
        + f". Dataset root: {dataset_root}. {suffix}"
    )


def _metric_fail_policy(payload: dict[str, Any]) -> str:
    raw = str(payload.get("metric_fail_policy") or "").strip().lower().replace("-", "_")
    if not raw:
        raw = (
            METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY
            if _parse_bool(payload.get("allow_diagnostic_metrics"), False)
            else METRIC_FAIL_POLICY_FAIL_CLOSED
        )
    aliases = {
        "fail_closed": METRIC_FAIL_POLICY_FAIL_CLOSED,
        "production": METRIC_FAIL_POLICY_FAIL_CLOSED,
        "robust": METRIC_FAIL_POLICY_FAIL_CLOSED,
        "diagnostic": METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
        "diagnostic_only": METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
        "fail_open": METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
        "development": METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
    }
    policy = aliases.get(raw)
    if policy is None:
        raise RuntimeError(
            "Unsupported metric_fail_policy. Use 'fail_closed' for production or "
            "'diagnostic_only' for explicit development diagnostics."
        )
    return policy


def _metric_allowed_returncodes(policy: str) -> tuple[int, ...]:
    return (0, 2) if policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY else (0,)


def _metric_fail_policy_warning(policy: str) -> dict[str, str] | None:
    if policy != METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY:
        return None
    return {
        "severity": "severe",
        "kind": "metric_fail_policy_diagnostic_only",
        "message": "Metrics were produced in explicit fail-open diagnostic mode; robust winners are disabled.",
    }


def _force_diagnostic_metric_manifest(
    manifest: dict[str, Any],
    *,
    metric_fail_policy: str,
) -> dict[str, Any]:
    manifest["metric_fail_policy"] = metric_fail_policy
    manifest["fail_open_metric_outputs"] = metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY
    manifest["robust_winner_allowed"] = metric_fail_policy == METRIC_FAIL_POLICY_FAIL_CLOSED
    if metric_fail_policy != METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY:
        return manifest
    warning = _metric_fail_policy_warning(metric_fail_policy)
    warnings = list(manifest.get("warnings") or [])
    if warning and warning not in warnings:
        warnings.append(warning)
    manifest["warnings"] = warnings
    manifest["status"] = "diagnostic_only"
    manifest["scientific_status"] = "diagnostic_only"
    manifest["comparability_status"] = "diagnostic_only"
    manifest["robust_winner_allowed"] = False
    recommendation = dict(manifest.get("recommendation") or {})
    recommendation.update(
        {
            "winner": None,
            "robust_recommendation": False,
            "diagnostic_only": True,
            "reason": "Metrics were run in explicit fail-open diagnostic mode.",
            "metric_fail_policy": metric_fail_policy,
        }
    )
    manifest["recommendation"] = recommendation
    return manifest


def _deeph_metric_command_args(
    *,
    python_executable: str,
    graph2mat_result_dir: Path,
    processed_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    metric_fail_policy: str,
) -> list[str]:
    command = [
        python_executable,
        str(DEFAULT_DEEPH_KPOINT_METRICS_SCRIPT),
        "--graph2mat-result-dir",
        str(graph2mat_result_dir),
        "--processed-dir",
        str(processed_dir),
        "--predictions-dir",
        str(predictions_dir),
        "--output-dir",
        str(output_dir),
        "--prediction-filename",
        "hamiltonians_pred.h5",
        "--split",
        "test",
    ]
    if metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY:
        command.append("--no-fail-closed")
    return command


def _strict_dataset_validation_kwargs(
    payload: dict[str, Any],
    *,
    dataset_root: Path,
    snapshot_root: Path,
) -> dict[str, Any]:
    material_provenance = dataset_root / "material_provenance.json"
    return {
        "system_label": payload.get("system_label") or None,
        "require_tshs": _parse_bool(payload.get("require_tshs"), True),
        "require_tsde": _parse_bool(payload.get("require_tsde"), True),
        "require_run_output": _parse_bool(payload.get("require_run_output"), True),
        "basis_dirs": [Path(path) for path in payload.get("basis_dirs", [])]
        if isinstance(payload.get("basis_dirs"), list)
        else [
            dataset_root / "basis",
            snapshot_root / "basis",
        ],
        "pseudopotential_provenance_paths": [
            Path(path) for path in payload.get("pseudopotential_provenance_paths", [])
        ]
        if isinstance(payload.get("pseudopotential_provenance_paths"), list)
        else [material_provenance],
        "material_identity_paths": [Path(path) for path in payload.get("material_identity_paths", [])]
        if isinstance(payload.get("material_identity_paths"), list)
        else [material_provenance],
        "siesta_input_paths": [Path(path) for path in payload.get("siesta_input_paths", [])]
        if isinstance(payload.get("siesta_input_paths"), list)
        else [dataset_root / "RUN.fdf", material_provenance],
        "validation_profile": G2M_DEEPH_BENCHMARK_PROFILE,
    }


class Graph2MatDeepHBenchmarkRunner:
    """Dedicated backend runner for the joint Graph2Mat/DeepH workflow."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._processes: list[subprocess.Popen[Any]] = []
        self._logs: list[str] = []
        self._state = G2MDeepHRunState()
        self._dataset_validation: dict[str, Any] | None = None
        self._last_results: dict[str, Any] | None = None
        self._phase_timings: list[dict[str, Any]] = []
        self._graph2mat_acceleration_cache: dict[str, dict[str, Any]] = {}
        self._graph2mat_acceleration_logged = False

    def validate_dataset_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        dataset_root = _resolve_optional_repo_path(payload.get("dataset_root"), DEFAULT_DATASET_ROOT)
        snapshot_root = Path(str(payload.get("snapshot_root") or payload.get("steps_dir") or ""))
        if str(snapshot_root) in {"", "."}:
            snapshot_root = dataset_root / "MD_steps" if (dataset_root / "MD_steps").exists() else dataset_root
        elif not snapshot_root.is_absolute():
            snapshot_root = dataset_root / snapshot_root
        snapshot_root = snapshot_root.resolve(strict=False)
        validate_kwargs = _strict_dataset_validation_kwargs(
            payload,
            dataset_root=dataset_root,
            snapshot_root=snapshot_root,
        )
        validation_root = snapshot_root if snapshot_root != dataset_root else dataset_root
        result = validate_dataset(
            validation_root,
            snapshot_dirs=None if snapshot_root == dataset_root else None,
            **validate_kwargs,
        )

        if snapshot_root != dataset_root:
            result.dataset_root = dataset_root

        validation = _validation_payload(
            result,
            snapshot_root=snapshot_root,
            max_snapshots=int(payload.get("max_invalid_snapshot_preview", 20) or 20),
        )
        if validation["benchmark_ready"] or snapshot_root != dataset_root or not dataset_root.exists():
            return validation

        child_payloads: list[dict[str, Any]] = []
        for child in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            if not (child / "benchmark_dataset_manifest.json").exists():
                continue
            if not (child / "frozen_split_manifest.json").exists():
                continue
            child_snapshot_root = child / "MD_steps" if (child / "MD_steps").exists() else child
            child_validate_kwargs = _strict_dataset_validation_kwargs(
                payload,
                dataset_root=child,
                snapshot_root=child_snapshot_root,
            )
            child_result = validate_dataset(
                child_snapshot_root,
                snapshot_dirs=None,
                **child_validate_kwargs,
            )
            if child_snapshot_root != child:
                child_result.dataset_root = child
            child_payload = _validation_payload(
                child_result,
                snapshot_root=child_snapshot_root,
                max_snapshots=int(payload.get("max_invalid_snapshot_preview", 20) or 20),
            )
            child_payload["dataset_collection_root"] = str(dataset_root)
            if child_payload["benchmark_ready"]:
                child_payloads.append(child_payload)

        if len(child_payloads) == 1:
            selected = child_payloads[0]
            selected["auto_selected_child_dataset"] = True
            selected.setdefault("warnings", []).append(
                "dataset_root points to a dataset collection; auto-selected the only valid joint child dataset"
            )
            return selected
        if child_payloads:
            validation["child_datasets"] = [
                {
                    "dataset_root": child["dataset_root"],
                    "snapshot_root": child["snapshot_root"],
                    "artifact_summary": child.get("artifact_summary"),
                    "manifest_paths": child.get("manifest_paths"),
                }
                for child in child_payloads
            ]
            validation.setdefault("warnings", []).append(
                "dataset_root points to a dataset collection with multiple valid joint datasets; "
                "select one child dataset explicitly"
            )
        return validation

    def available_datasets_payload(self, root: str | Path | None = None) -> dict[str, Any]:
        search_root = _resolve_optional_repo_path(root, DEFAULT_DATASETS_ROOT)
        datasets: list[dict[str, Any]] = []
        if search_root.exists():
            manifest_paths = sorted(search_root.rglob("benchmark_dataset_manifest.json"))
            for manifest_path in manifest_paths:
                dataset_root = manifest_path.parent
                if not (dataset_root / "frozen_split_manifest.json").exists():
                    continue
                try:
                    manifest = _load_json(manifest_path)
                except Exception as exc:
                    datasets.append(
                        {
                            "id": stable_payload_hash(str(dataset_root), length=12),
                            "dataset_root": str(dataset_root),
                            "relative_path": str(dataset_root.relative_to(REPO_ROOT))
                            if dataset_root.is_relative_to(REPO_ROOT)
                            else str(dataset_root),
                            "label": dataset_root.name,
                            "benchmark_ready": False,
                            "error": str(exc),
                        }
                    )
                    continue
                try:
                    validation = self.validate_dataset_payload(
                        {
                            "dataset_root": str(dataset_root),
                            "max_invalid_snapshot_preview": 3,
                        }
                    )
                    summary = validation.get("artifact_summary") or {}
                    ready = bool(validation.get("benchmark_ready"))
                    errors = list(validation.get("errors") or [])
                except Exception as exc:
                    summary = {}
                    ready = False
                    errors = [str(exc)]
                label = (
                    manifest.get("dataset_id")
                    or manifest.get("label")
                    or manifest.get("recipe_label")
                    or dataset_root.name
                )
                try:
                    relative_path = str(dataset_root.relative_to(REPO_ROOT))
                except ValueError:
                    relative_path = str(dataset_root)
                datasets.append(
                    {
                        "id": stable_payload_hash(str(dataset_root), length=12),
                        "label": str(label),
                        "dataset_root": str(dataset_root),
                        "relative_path": relative_path,
                        "benchmark_ready": ready,
                        "total_snapshots": int(summary.get("total_snapshots") or 0),
                        "valid_snapshots": int(summary.get("valid_snapshots") or 0),
                        "invalid_snapshots": int(summary.get("invalid_snapshots") or 0),
                        "missing_required_counts": summary.get("missing_required_counts") or {},
                        "split_counts": manifest.get("split_counts") or manifest.get("splits") or {},
                        "split_hash": manifest.get("split_hash") or manifest.get("frozen_split_hash"),
                        "artifact_contract_version": manifest.get("artifact_contract_version"),
                        "errors": errors,
                    }
                )
        datasets.sort(key=lambda item: (not item.get("benchmark_ready"), item.get("relative_path") or ""))
        return {
            "datasets_root": str(search_root),
            "datasets": datasets,
            "count": len(datasets),
            "ready_count": sum(1 for item in datasets if item.get("benchmark_ready")),
        }

    def _split_ratios_from_payload(self, payload: dict[str, Any]) -> dict[str, float]:
        raw = payload.get("splits") or {}
        if not isinstance(raw, dict):
            raise RuntimeError("splits debe ser un objeto.")
        return {
            "train": float(raw.get("train", 0.8)),
            "validation": float(raw.get("validation", raw.get("val", 0.1))),
            "test": float(raw.get("test", 0.1)),
        }

    def dataset_sweep_info_from_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        enabled, recipes, max_datasets = dataset_sweep_recipes_from_payload(payload)
        if not enabled:
            return None
        split_ratios = self._split_ratios_from_payload(payload)
        info = md_dataset_recipes_to_specs(
            {"md": recipes},
            split_ratios=split_ratios,
            max_datasets=max_datasets,
        )
        if not info or not info.get("md_dataset_specs"):
            raise RuntimeError("dataset_sweep esta activado pero no contiene datasets MD validos.")
        info["enabled"] = True
        info["max_datasets"] = max_datasets
        info["split_ratios"] = split_ratios
        info["total_snapshots"] = sum(int(spec.get("size") or 0) for spec in info["md_dataset_specs"])
        return info

    def _benchmark_run_root(self, payload: dict[str, Any], run_id: str) -> Path:
        output_root = _resolve_optional_repo_path(payload.get("output_root"), DEFAULT_OUTPUT_ROOT)
        run_root = output_root / run_id
        if run_root.exists() and not _parse_bool(payload.get("reuse_run_root"), False):
            raise RuntimeError(f"Benchmark run_root already exists: {run_root}")
        return run_root

    def _load_frozen_split_manifest(self, dataset_root: Path) -> dict[str, Any]:
        path = dataset_root / "frozen_split_manifest.json"
        manifest = _load_json(path)
        if not manifest.get("valid"):
            raise RuntimeError(f"Frozen split manifest is not valid: {path}")
        rows = manifest.get("rows")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"Frozen split manifest has no rows: {path}")
        split_counts = manifest.get("split_counts") or {}
        for split in ("train", "validation", "test"):
            if int(split_counts.get(split) or 0) <= 0:
                raise RuntimeError(
                    f"Frozen split manifest requires a non-empty {split!r} split for Graph2Mat."
                )
        return manifest

    def _ensure_graph2mat_basis_visible(self, dataset_root: Path) -> int:
        sample_dirs: list[Path] = []
        md_steps = dataset_root / "MD_steps"
        if md_steps.exists():
            sample_dirs.extend(
                sorted(
                    (path for path in md_steps.iterdir() if path.is_dir() and path.name.isdigit()),
                    key=lambda path: int(path.name),
                )
            )
        split_root = dataset_root / "splits"
        if split_root.exists():
            sample_dirs.extend(sorted(path for path in split_root.glob("*/*") if path.is_dir()))
        return _materialize_graph2mat_basis_files(dataset_root, sample_dirs)

    def _copy_prediction_structures(
        self,
        frozen_split: dict[str, Any],
        destination_root: Path,
        *,
        dataset_root: Path,
    ) -> list[str]:
        test_rows = [row for row in frozen_split.get("rows", []) if row.get("split") == "test"]
        if not test_rows:
            raise RuntimeError("Frozen split manifest has no test rows for prediction.")
        if destination_root.exists():
            shutil.rmtree(destination_root)
        sample_ids: list[str] = []
        for row in test_rows:
            sample_id = str(row.get("sample_id") or row.get("deeph_sample_id") or "").strip()
            sample_dir = Path(str(row.get("sample_dir") or ""))
            if not sample_id:
                raise RuntimeError(f"Frozen split test row is missing sample_id: {row}")
            if not sample_dir.exists():
                raise RuntimeError(f"Frozen split test sample_dir does not exist: {sample_dir}")
            target = destination_root / sample_id
            target.mkdir(parents=True, exist_ok=True)
            for artifact in sorted(path for path in sample_dir.iterdir() if path.is_file()):
                if artifact.name == "ML_prediction.HSX":
                    continue
                _link_or_copy_file(artifact, target / artifact.name)
            _materialize_graph2mat_basis_files(dataset_root, [target])
            if not (target / "RUN.fdf").exists():
                raise RuntimeError(f"Prediction sample is missing RUN.fdf after staging: {target}")
            sample_ids.append(sample_id)
        return sample_ids

    def _write_graph2mat_runs_json(
        self,
        *,
        dataset_root: Path,
        training_dir: Path,
        prediction_structs_dir: Path,
    ) -> tuple[Path, dict[str, int]]:
        """Write explicit Graph2Mat split paths to avoid repeated absolute glob expansion."""

        training_dir.mkdir(parents=True, exist_ok=True)
        split_specs = (
            ("train", dataset_root / "splits" / "train"),
            ("val", dataset_root / "splits" / "validation"),
            ("test", dataset_root / "splits" / "test"),
        )
        runs: dict[str, list[str]] = {}
        counts: dict[str, int] = {}
        for split_name, split_dir in split_specs:
            paths = sorted(split_dir.glob("*/RUN.fdf"))
            if not paths:
                raise RuntimeError(f"Graph2Mat runs_json split {split_name!r} has no RUN.fdf files under {split_dir}")
            runs[split_name] = [os.path.relpath(str(path), str(training_dir)) for path in paths]
            counts[split_name] = len(paths)
        predict_paths = sorted(prediction_structs_dir.glob("*/RUN.fdf"))
        if predict_paths:
            runs["predict"] = [os.path.relpath(str(path), str(training_dir)) for path in predict_paths]
            counts["predict"] = len(predict_paths)
        runs_json_path = training_dir / "runs.json"
        _write_json(runs_json_path, runs)
        return runs_json_path, counts

    def _graph2mat_acceleration_status(self, python_executable: str) -> dict[str, Any]:
        key = str(Path(python_executable).expanduser())
        with self._lock:
            cached = self._graph2mat_acceleration_cache.get(key)
        if cached is not None:
            return dict(cached)
        code = (
            "import importlib.util,json;"
            "mods=['cuequivariance','cuequivariance_torch'];"
            "print(json.dumps({m: importlib.util.find_spec(m) is not None for m in mods}))"
        )
        status: dict[str, Any] = {
            "source": "python_import_probe",
            "python": key,
            "cuequivariance_available": False,
            "cuequivariance_torch_available": False,
            "acceleration_available": False,
        }
        try:
            completed = subprocess.run(
                [python_executable, "-c", code],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            status["returncode"] = completed.returncode
            if completed.returncode == 0:
                payload = json.loads(completed.stdout.strip() or "{}")
                status["cuequivariance_available"] = bool(payload.get("cuequivariance"))
                status["cuequivariance_torch_available"] = bool(payload.get("cuequivariance_torch"))
                status["acceleration_available"] = bool(
                    status["cuequivariance_available"] or status["cuequivariance_torch_available"]
                )
            else:
                status["error"] = completed.stderr.strip() or completed.stdout.strip() or "probe_failed"
        except Exception as exc:
            status["error"] = str(exc)
        with self._lock:
            self._graph2mat_acceleration_cache[key] = dict(status)
            if not self._graph2mat_acceleration_logged:
                if status.get("acceleration_available"):
                    self._logs.append(
                        "[G2M-DEEPH][PERF] cuequivariance acceleration detected for Graph2Mat.\n"
                    )
                else:
                    self._logs.append(
                        "[G2M-DEEPH][PERF][WARN] cuequivariance/cuequivariance_torch no detectado; "
                        "Graph2Mat usara kernels PyTorch/e3nn sin esa aceleracion.\n"
                    )
                self._graph2mat_acceleration_logged = True
        return dict(status)

    def _enforce_graph2mat_acceleration_policy(self, payload: dict[str, Any], python_executable: str) -> dict[str, Any]:
        status = self._graph2mat_acceleration_status(python_executable)
        performance = _performance_settings_from_payload(payload)
        if performance.get("graph2mat_require_cuequivariance") and not status.get("acceleration_available"):
            raise RuntimeError(
                "performance.graph2mat_require_cuequivariance=true but cuequivariance/cuequivariance_torch "
                f"is not importable by {python_executable}. Install cuequivariance and cuequivariance-torch "
                "in the Graph2Mat Python environment or disable the requirement."
            )
        return status

    def _graph2mat_python(self, payload: dict[str, Any]) -> str:
        raw = payload.get("python") or payload.get("graph2mat_python")
        if raw not in (None, ""):
            return str(_resolve_optional_repo_path(raw, Path(str(raw))))
        repo_venv_python = REPO_ROOT / ".venv" / "bin" / "python"
        if repo_venv_python.exists():
            return str(repo_venv_python)
        return sys.executable

    def _prepare_graph2mat_context(
        self,
        payload: dict[str, Any],
        validation: dict[str, Any],
    ) -> Graph2MatBenchmarkContext:
        dataset_root = Path(validation["dataset_root"])
        run_id = str(payload.get("run_id") or self._state.run_id or time.strftime("g2m_deeph_%Y%m%d_%H%M%S"))
        run_root = self._benchmark_run_root(payload, run_id)
        graph2mat_root = run_root / "graph2mat"
        training_dir = graph2mat_root / "training"
        prediction_structs_dir = graph2mat_root / "prediction_structures" / "test"
        config_path = graph2mat_root / "pipeline_config.yaml"
        graph2mat_config_path = training_dir / "config.yaml"
        graph2mat_manifest_path = graph2mat_root / "graph2mat_manifest.json"
        frozen_split_manifest_path = dataset_root / "frozen_split_manifest.json"
        benchmark_dataset_manifest_path = dataset_root / "benchmark_dataset_manifest.json"
        frozen_split = self._load_frozen_split_manifest(dataset_root)
        benchmark_manifest = _load_json(benchmark_dataset_manifest_path)
        if not benchmark_manifest.get("benchmark_ready"):
            raise RuntimeError(f"Benchmark dataset manifest is not ready: {benchmark_dataset_manifest_path}")

        materialized_basis = self._ensure_graph2mat_basis_visible(dataset_root)
        if materialized_basis:
            self._logs.append(
                "[G2M-DEEPH] Basis Graph2Mat materializada en snapshots/splits: "
                f"{materialized_basis} enlaces/copias.\n"
            )
        test_sample_ids = self._copy_prediction_structures(
            frozen_split,
            prediction_structs_dir,
            dataset_root=dataset_root,
        )
        runs_json_path, runs_json_counts = self._write_graph2mat_runs_json(
            dataset_root=dataset_root,
            training_dir=training_dir,
            prediction_structs_dir=prediction_structs_dir,
        )
        train_glob = str(dataset_root / "splits" / "train" / "*" / "RUN.fdf")
        validation_glob = str(dataset_root / "splits" / "validation" / "*" / "RUN.fdf")
        predict_glob = str(prediction_structs_dir / "*" / "RUN.fdf")
        for label, pattern in (
            ("training.data.train_runs", train_glob),
            ("training.data.val_runs", validation_glob),
            ("prediction.predict_structs", predict_glob),
        ):
            if not _glob_matches(pattern):
                raise RuntimeError(f"{label} did not match any RUN.fdf files: {pattern}")

        template_path = _resolve_optional_repo_path(
            payload.get("pipeline_config") or payload.get("md_pipeline_config"),
            DEFAULT_MD_PIPELINE_CONFIG,
        )
        config = _load_yaml(template_path)
        config.setdefault("paths", {})
        config["paths"]["dataset_dir"] = str(dataset_root)
        config["paths"]["training_dir"] = str(training_dir)
        config["paths"]["training_config_name"] = graph2mat_config_path.name
        config.setdefault("training", {}).setdefault("data", {})
        config["training"]["data"].pop("train_runs", None)
        config["training"]["data"].pop("val_runs", None)
        config["training"]["data"]["runs_json"] = runs_json_path.name
        config["training"]["data"]["root_dir"] = "."
        config["training"]["data"]["out_matrix"] = "hamiltonian"
        config["training"]["data"]["matrix_component_policy"] = "h_only"
        config["training"]["data"]["n_matrix_components"] = 1
        config["training"].setdefault("trainer", {}).setdefault("log_every_n_steps", 1)
        config.setdefault("prediction", {})
        config["prediction"]["predict_structs"] = _relative_pattern(predict_glob, training_dir)
        config["prediction"]["output_file"] = "ML_prediction.HSX"
        config.setdefault("checkpoint", {})
        config["checkpoint"]["path"] = None
        config["checkpoint"]["auto_best"] = True
        _apply_performance_to_graph2mat_config(config, _performance_settings_from_payload(payload))
        overrides = payload.get("graph2mat_overrides") or {}
        if isinstance(overrides, dict):
            self._apply_graph2mat_overrides(config, overrides)
        early_stopping_policy = _apply_common_early_stopping_to_graph2mat_config(config, payload)

        _write_yaml(config_path, config)
        context = Graph2MatBenchmarkContext(
            dataset_root=dataset_root,
            run_root=run_root,
            graph2mat_root=graph2mat_root,
            training_dir=training_dir,
            prediction_structs_dir=prediction_structs_dir,
            config_path=config_path,
            graph2mat_config_path=graph2mat_config_path,
            graph2mat_manifest_path=graph2mat_manifest_path,
            frozen_split_manifest_path=frozen_split_manifest_path,
            benchmark_dataset_manifest_path=benchmark_dataset_manifest_path,
            runs_json_path=runs_json_path,
            runs_json_counts=runs_json_counts,
            train_glob=train_glob,
            validation_glob=validation_glob,
            predict_glob=predict_glob,
            output_file="ML_prediction.HSX",
            test_sample_ids=test_sample_ids,
            split_hash=frozen_split.get("split_hash"),
            dry_run=_parse_bool(payload.get("dry_run"), False),
        )
        acceleration = self._enforce_graph2mat_acceleration_policy(
            payload,
            self._graph2mat_python(payload),
        )
        optimizer_accounting = self._graph2mat_optimizer_accounting(context, config)
        self._write_graph2mat_manifest(
            context,
            extra={
                "prepared": True,
                "template_config": str(template_path),
                "runs_json": str(runs_json_path),
                "runs_json_counts": runs_json_counts,
                "graph2mat_acceleration": acceleration,
                "early_stopping_policy": early_stopping_policy,
                "optimizer_update_accounting": optimizer_accounting,
            },
        )
        return context

    def _apply_graph2mat_overrides(self, config: dict[str, Any], overrides: dict[str, Any]) -> None:
        training = config.setdefault("training", {})
        model = training.setdefault("model", {})
        trainer = training.setdefault("trainer", {})
        data = training.setdefault("data", {})
        for key in ("max_epochs", "accelerator", "log_every_n_steps"):
            if key in overrides:
                trainer[key] = overrides[key]
        for key in ("optim_lr", "hidden_irreps", "num_interactions", "correlation", "max_ell", "loss"):
            if key in overrides:
                model[key] = overrides[key]
        if "loss_kwargs" in overrides:
            model["loss_kwargs"] = dict(overrides["loss_kwargs"])
        if "batch_size" in overrides:
            data["batch_size"] = overrides["batch_size"]
        if "loader_threads" in overrides:
            data["loader_threads"] = overrides["loader_threads"]
        if "seed_everything" in overrides:
            training["seed_everything"] = overrides["seed_everything"]
        if "store_in_memory" in overrides:
            data["store_in_memory"] = overrides["store_in_memory"]

    def _graph2mat_command_env(self, context: Graph2MatBenchmarkContext, payload: dict[str, Any]) -> dict[str, str]:
        return self._python_command_env(
            self._graph2mat_python(payload),
            PIPELINE_CONFIG_PATH=str(context.config_path),
            PYTHONUNBUFFERED="1",
            **_performance_env(_performance_settings_from_payload(payload)),
        )

    def _deeph_command_env(self, payload: dict[str, Any]) -> dict[str, str]:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        discovery = self._deeph_discovery(payload)
        python_path = str(discovery.get("python") or "")
        if python_path:
            python_bin = str(Path(python_path).expanduser().resolve().parent)
            current_path = env.get("PATH", "")
            env["PATH"] = python_bin if not current_path else f"{python_bin}{os.pathsep}{current_path}"
        repo_path = discovery.get("repo_path")
        if repo_path:
            repo = Path(str(repo_path)).expanduser().resolve()
            if (repo / "deeph").exists():
                current_pythonpath = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = str(repo) if not current_pythonpath else f"{repo}{os.pathsep}{current_pythonpath}"
        return env

    def _python_command_env(self, python_executable: str, **extra: str) -> dict[str, str]:
        env = {**os.environ, **extra}
        python_path = Path(python_executable).expanduser()
        if not python_path.is_absolute():
            python_path = REPO_ROOT / python_path
        python_bin = str(python_path.parent.resolve())
        current_path = env.get("PATH", "")
        env["PATH"] = python_bin if not current_path else f"{python_bin}{os.pathsep}{current_path}"
        return env

    def _run_command(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        label: str,
        allowed_returncodes: tuple[int, ...] = (0,),
        progress_provider: Any | None = None,
        line_observer: Any | None = None,
    ) -> dict[str, Any]:
        started_at = time.time()
        controlled_stop_reason: str | None = None
        with self._lock:
            self._logs.append(f"[G2M-DEEPH][RUN] {label}: {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        telemetry_monitor = GpuTelemetryMonitor()
        telemetry_monitor.start(process.pid)
        with self._lock:
            self._processes.append(process)
        try:
            assert process.stdout is not None
            fd = process.stdout.fileno()
            os.set_blocking(fd, False)
            pending = ""
            last_progress = ""
            last_heartbeat = started_at
            while True:
                select_timeout = min(1.0, max(0.05, LOG_HEARTBEAT_SECONDS))
                ready, _, _ = select.select([fd], [], [], select_timeout)
                if ready:
                    try:
                        chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
                    except BlockingIOError:
                        chunk = ""
                    if chunk:
                        pending += chunk.replace("\r", "\n")
                        while "\n" in pending:
                            line, pending = pending.split("\n", 1)
                            with self._lock:
                                self._logs.append(line + "\n")
                            if line_observer is not None and process.poll() is None:
                                reason = line_observer(line)
                                if reason and controlled_stop_reason is None:
                                    controlled_stop_reason = str(reason)
                                    with self._lock:
                                        self._logs.append(
                                            f"[G2M-DEEPH][EARLY-STOP] {label}: {controlled_stop_reason}\n"
                                        )
                                    process.terminate()

                if self._state.stop_requested and process.poll() is None:
                    process.terminate()

                now = time.time()
                if now - last_heartbeat >= LOG_HEARTBEAT_SECONDS and process.poll() is None:
                    progress_text = progress_provider() if progress_provider is not None else None
                    if progress_text and progress_text != last_progress:
                        with self._lock:
                            self._logs.append(
                                "[G2M-DEEPH][PROGRESS] "
                                f"{label} | elapsed {_format_duration(now - started_at)} | "
                                f"{progress_text}\n"
                            )
                        last_progress = progress_text
                    elif progress_provider is not None:
                        with self._lock:
                            self._logs.append(
                                "[G2M-DEEPH][PROGRESS] "
                                f"{label} | elapsed {_format_duration(now - started_at)} | "
                                "esperando eventos de entrenamiento...\n"
                            )
                    last_heartbeat = now

                if process.poll() is not None:
                    while True:
                        try:
                            chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
                        except BlockingIOError:
                            break
                        if not chunk:
                            break
                        pending += chunk.replace("\r", "\n")
                        while "\n" in pending:
                            line, pending = pending.split("\n", 1)
                            with self._lock:
                                self._logs.append(line + "\n")
                            if line_observer is not None and controlled_stop_reason is None:
                                reason = line_observer(line)
                                if reason:
                                    controlled_stop_reason = str(reason)
                    break
            if pending:
                with self._lock:
                    self._logs.append(pending)
            returncode = process.wait()
        finally:
            command_telemetry = telemetry_monitor.stop()
            if process.stdout is not None:
                process.stdout.close()
            with self._lock:
                self._processes = [item for item in self._processes if item is not process]
        finished_at = time.time()
        if returncode not in allowed_returncodes and controlled_stop_reason is None:
            raise RuntimeError(f"{label} failed with exit code {returncode}")
        return {
            "label": label,
            "command": command,
            "cwd": str(cwd),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": finished_at - started_at,
            "returncode": returncode,
            "controlled_stop_reason": controlled_stop_reason,
            "telemetry": {
                **command_telemetry,
                "gpu_hours": compute_gpu_hours(
                    command_telemetry.get("gpu_active_seconds"),
                    command_telemetry.get("observed_gpu_count"),
                ),
            },
        }

    def _md_generation_python(self, payload: dict[str, Any]) -> str:
        raw = payload.get("python") or payload.get("graph2mat_python")
        if raw not in (None, ""):
            return str(_resolve_optional_repo_path(raw, Path(str(raw))))
        repo_venv_python = REPO_ROOT / ".venv" / "bin" / "python"
        if repo_venv_python.exists():
            return str(repo_venv_python)
        return sys.executable

    def _md_split_counts_for_sweep(
        self,
        dataset_size: int,
        split_ratios: dict[str, float],
        *,
        split_mode: str,
        temporal_gap: int,
        label: str,
    ) -> tuple[dict[str, int], int]:
        if split_mode != "blocked_with_gap":
            return validate_split_sizes(dataset_size, split_ratios, label=label), 0
        reserved = 2 * max(0, int(temporal_gap))
        usable_size = dataset_size - reserved
        if usable_size < 3:
            raise RuntimeError(
                f"{label}: blocked_with_gap con temporal_gap={temporal_gap} necesita al menos "
                f"{3 + reserved} frames MD."
            )
        return validate_split_sizes(usable_size, split_ratios, label=label), reserved

    def _dataset_sweep_compatibility_hash(self, config: dict[str, Any]) -> str:
        md = dict(config.get("md") or {})
        for key in ("steps", "temperature_K", "temperature_blocks", "timestep_fs", "ensemble", "thermostat"):
            md.pop(key, None)
        return stable_payload_hash(
            {
                "material": config.get("material"),
                "md_reference_settings": md,
                "training_target": (config.get("training") or {}).get("data"),
            },
            length=16,
        )

    def _prepare_md_dataset_sweep_config(
        self,
        payload: dict[str, Any],
        spec: dict[str, Any],
        *,
        run_root: Path,
        dataset_root: Path,
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        split_mode = str(payload.get("split_mode") or "blocked_with_gap").strip().lower()
        temporal_gap = int(payload.get("temporal_gap", 1 if split_mode == "blocked_with_gap" else 0) or 0)
        split_ratios = self._split_ratios_from_payload(payload)
        counts, reserved_gap_frames = self._md_split_counts_for_sweep(
            int(spec["size"]),
            split_ratios,
            split_mode=split_mode,
            temporal_gap=temporal_gap,
            label=f"MD recipe {spec['recipe_id']}",
        )
        template_path = _resolve_optional_repo_path(
            payload.get("pipeline_config") or payload.get("md_pipeline_config"),
            DEFAULT_MD_PIPELINE_CONFIG,
        )
        config = _load_yaml(template_path)
        workspace = run_root / "workspaces" / "datasets" / str(spec["dataset_slug"])
        training_dir = workspace / "training"
        config_path = workspace / "pipeline_config.yaml"
        config.setdefault("paths", {})
        config["paths"]["dataset_dir"] = str(dataset_root)
        config["paths"]["training_dir"] = str(training_dir)
        config.setdefault("md", {})
        config["md"]["steps"] = int(spec["size"])
        config["md"]["temperature_blocks"] = list(spec["temperature_blocks"])
        config["dataset_recipe"] = dict(spec["recipe_metadata"])
        config["splits"] = {
            "enabled": True,
            "strategy": split_mode,
            "train": counts["train"],
            "validation": counts["validation"],
            "test": counts["test"],
        }
        if split_mode == "blocked_with_gap":
            config["splits"]["temporal_gap"] = temporal_gap
            config["splits"]["block_order"] = str(payload.get("block_order") or "train,validation,test")
        config.setdefault("pipeline", {})["steps"] = ["generate_md_dataset"]
        _write_yaml(config_path, config)
        return config_path, config, {
            "split_mode": split_mode,
            "temporal_gap": temporal_gap,
            "reserved_gap_frames": reserved_gap_frames,
            "split_counts": counts,
            "template_config": str(template_path),
            "workspace": str(workspace),
        }

    def _validate_swept_dataset(self, dataset_root: Path, system_label: str | None) -> dict[str, Any]:
        snapshot_root = dataset_root / "MD_steps" if (dataset_root / "MD_steps").exists() else dataset_root
        material_provenance = dataset_root / "material_provenance.json"
        validation = validate_dataset(
            snapshot_root,
            system_label=system_label,
            require_tshs=True,
            require_tsde=True,
            require_run_output=True,
            basis_dirs=[dataset_root / "basis", snapshot_root / "basis"],
            pseudopotential_provenance_paths=[material_provenance],
            material_identity_paths=[material_provenance],
            siesta_input_paths=[dataset_root / "RUN.fdf", material_provenance],
            validation_profile=G2M_DEEPH_BENCHMARK_PROFILE,
        )
        if not validation.valid:
            missing = _missing_key_counts(validation)
            raise RuntimeError(f"Swept dataset failed joint artifact validation: {dataset_root} missing={missing}")
        manifest_path = dataset_root / "benchmark_dataset_manifest.json"
        split_path = dataset_root / "frozen_split_manifest.json"
        if not manifest_path.exists() or not split_path.exists():
            raise RuntimeError(
                "Swept dataset did not produce benchmark manifests: "
                f"{manifest_path}, {split_path}"
            )
        return {
            "artifact_validation": validation.to_dict(),
            "benchmark_dataset_manifest": str(manifest_path),
            "frozen_split_manifest": str(split_path),
            "benchmark_ready": True,
        }

    def _write_dataset_sweep_summary(
        self,
        run_root: Path,
        *,
        payload: dict[str, Any],
        sweep_info: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        summary_root = run_root / "summary"
        manifest = {
            "schema": "graph2mat_deeph_dataset_sweep_summary_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "artifact_contract_version": CONTRACT_NAME,
            "run_mode": str(payload.get("run_mode") or DATASET_SWEEP_RUN_MODE),
            "dataset_root": str(_resolve_optional_repo_path(payload.get("dataset_root"), DEFAULT_DATASET_ROOT)),
            "recipe_set_hash": sweep_info.get("recipe_set_hash"),
            "split_ratios": sweep_info.get("split_ratios"),
            "max_datasets": sweep_info.get("max_datasets"),
            "total_datasets": len(rows),
            "total_snapshots": sum(int(row.get("dataset_size") or 0) for row in rows),
            "rows": rows,
            "scientific_rule": (
                "Dataset sweep v1 varies only MD size/temperature/timestep/ensemble/"
                "thermostat/seed; SIESTA physics settings remain fixed."
            ),
        }
        _write_json(summary_root / "dataset_sweep_summary.json", manifest)
        _write_csv(summary_root / "dataset_sweep_summary.csv", rows)
        _write_yaml(run_root / "benchmark_manifest.yaml", manifest)
        return manifest

    def _run_dataset_sweep_generation(
        self,
        payload: dict[str, Any],
        sweep_info: dict[str, Any],
        *,
        run_root: Path,
    ) -> dict[str, Any]:
        base_dataset_root = _resolve_optional_repo_path(payload.get("dataset_root"), DEFAULT_DATASET_ROOT)
        system_label = str(payload.get("system_label") or "graphene")
        overwrite = _parse_bool(payload.get("overwrite_datasets"), False)
        dry_run = _parse_bool(payload.get("dry_run"), False)
        run_root.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for spec in sweep_info["md_dataset_specs"]:
            if self._state.stop_requested:
                raise RuntimeError("Stop requested during dataset sweep generation.")
            self._set_stage("generate_or_validate_joint_dataset")
            dataset_root = base_dataset_root / str(spec["dataset_slug"])
            if dataset_root.exists() and any(dataset_root.iterdir()) and not overwrite:
                raise RuntimeError(
                    f"Dataset sweep target already exists: {dataset_root}. "
                    "Set overwrite_datasets=true only for explicit regeneration."
                )
            if overwrite and dataset_root.exists() and not dry_run:
                shutil.rmtree(dataset_root)
            started_at = time.time()
            config_path, config, extra = self._prepare_md_dataset_sweep_config(
                payload,
                spec,
                run_root=run_root,
                dataset_root=dataset_root,
            )
            compatibility_hash = self._dataset_sweep_compatibility_hash(config)
            row = {
                "recipe_id": spec["recipe_id"],
                "recipe_label": spec["label"],
                "dataset_slug": spec["dataset_slug"],
                "dataset_root": str(dataset_root),
                "dataset_size": int(spec["size"]),
                "blocks": json.dumps(spec["temperature_blocks"], sort_keys=True, ensure_ascii=False),
                "split_counts": json.dumps(extra["split_counts"], sort_keys=True),
                "split_mode": extra["split_mode"],
                "temporal_gap": extra["temporal_gap"],
                "reserved_gap_frames": extra["reserved_gap_frames"],
                "config_path": str(config_path),
                "compatibility_hash": compatibility_hash,
                "artifact_contract_version": CONTRACT_NAME,
                "status": "dry_run" if dry_run else "pending",
            }
            self._logs.append(
                f"[G2M-DEEPH] Dataset sweep {spec['recipe_id']}: "
                f"{spec['size']} snapshots -> {dataset_root}\n"
            )
            if dry_run:
                row.update(
                    {
                        "generation_seconds": 0.0,
                        "artifact_validation_status": "not_run_dry_run",
                        "benchmark_manifest_path": "",
                        "frozen_split_manifest_path": "",
                        "status": "dry_run",
                    }
                )
            else:
                command = [self._md_generation_python(payload), str(REPO_ROOT / "MD" / "scripts" / "generate_md_dataset.py")]
                run = self._run_command(
                    command,
                    cwd=REPO_ROOT,
                    env=self._python_command_env(
                        command[0],
                        PIPELINE_CONFIG_PATH=str(config_path),
                        PYTHONUNBUFFERED="1",
                    ),
                    label=f"generate_md_dataset[{spec['recipe_id']}]",
                )
                validation = self._validate_swept_dataset(dataset_root, system_label)
                row.update(
                    {
                        "generation_seconds": run["elapsed_seconds"],
                        "artifact_validation_status": "valid",
                        "artifact_validation_path": str(dataset_root / "artifact_validation.json"),
                        "benchmark_manifest_path": validation["benchmark_dataset_manifest"],
                        "frozen_split_manifest_path": validation["frozen_split_manifest"],
                        "status": "benchmark_ready",
                    }
                )
            row["elapsed_seconds"] = time.time() - started_at
            rows.append(row)
        return self._write_dataset_sweep_summary(
            run_root,
            payload=payload,
            sweep_info=sweep_info,
            rows=rows,
        )

    def _run_dataset_sweep_only(self, payload: dict[str, Any], sweep_info: dict[str, Any]) -> None:
        try:
            run_root = Path(str(self._state.run_root or self._benchmark_run_root(payload, self._state.run_id or "dataset_sweep")))
            dry_run = _parse_bool(payload.get("dry_run"), False)
            manifest = self._run_dataset_sweep_generation(payload, sweep_info, run_root=run_root)
            with self._lock:
                self._last_results = {
                    "dry_run": dry_run,
                    "contract_name": CONTRACT_NAME,
                    "dataset_sweep": manifest,
                    "phase_timings": list(self._phase_timings),
                    "message": "Dataset sweep completed; no Graph2Mat or DeepH training was launched.",
                }
            self._set_stage("complete")
            self._finish(returncode=0)
        except Exception as exc:
            self._finish(returncode=1, error=str(exc))

    def _validate_graph2mat_prediction_outputs(self, context: Graph2MatBenchmarkContext) -> dict[str, Any]:
        structures = _glob_matches(context.predict_glob)
        if not structures:
            raise RuntimeError(f"No prediction RUN.fdf files matched: {context.predict_glob}")
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        empty: list[str] = []
        for structure in structures:
            output = structure.parent / context.output_file
            row = {
                "sample_id": structure.parent.name,
                "structure_path": str(structure),
                "prediction_path": str(output),
                "exists": output.exists(),
                "sha256": _file_sha256(output) if output.exists() and output.is_file() else None,
            }
            if not output.exists():
                missing.append(str(output))
            elif output.stat().st_size <= 0:
                empty.append(str(output))
            rows.append(row)
        if missing:
            raise RuntimeError("Missing Graph2Mat prediction outputs: " + ", ".join(missing[:10]))
        if empty:
            raise RuntimeError("Empty Graph2Mat prediction outputs: " + ", ".join(empty[:10]))
        return {
            "count": len(rows),
            "rows": rows,
            "prediction_root": str(context.prediction_structs_dir),
        }

    def _write_graph2mat_manifest(
        self,
        context: Graph2MatBenchmarkContext,
        *,
        checkpoint_manifest: dict[str, Any] | None = None,
        prediction_outputs: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "schema": "graph2mat_deeph_graph2mat_stage_manifest_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "artifact_contract_version": CONTRACT_NAME,
            "context": context.to_dict(),
            "config_sha256": _file_sha256(context.config_path) if context.config_path.exists() else None,
            "graph2mat_config_sha256": (
                _file_sha256(context.graph2mat_config_path)
                if context.graph2mat_config_path.exists()
                else None
            ),
            "checkpoint_manifest": checkpoint_manifest,
            "prediction_outputs": prediction_outputs,
            "extra": extra or {},
        }
        _write_json(context.graph2mat_manifest_path, payload)
        return payload

    def _deeph_options(self, payload: dict[str, Any]) -> dict[str, Any]:
        options = payload.get("deeph") or payload.get("deeph_options") or {}
        return options if isinstance(options, dict) else {}

    def _deeph_configured_repo(self, payload: dict[str, Any]) -> tuple[Path | None, str, str | None]:
        options = self._deeph_options(payload)
        raw = (
            options.get("repo_path")
            or options.get("deeph_repo_path")
            or options.get("repo")
            or payload.get("deeph_repo_path")
            or payload.get("deeph_repo")
        )
        if raw not in (None, ""):
            return _resolve_optional_repo_path(raw, Path(str(raw))), "config", None
        env_value = os.environ.get(DEEPH_PACK_ROOT_ENV)
        if env_value:
            return _resolve_optional_repo_path(env_value, Path(env_value)), "env", DEEPH_PACK_ROOT_ENV
        return None, "unavailable", None

    def _deeph_configured_bin_dir(self, payload: dict[str, Any]) -> Path | None:
        options = self._deeph_options(payload)
        raw = options.get("bin_dir") or payload.get("deeph_bin_dir")
        if raw in (None, ""):
            return None
        return _resolve_optional_repo_path(raw, Path(str(raw)))

    def _deeph_explicit_command(self, payload: dict[str, Any], command_name: str) -> tuple[str | None, str]:
        options = self._deeph_options(payload)
        commands = options.get("commands") if isinstance(options.get("commands"), dict) else {}
        explicit = commands.get(command_name) or options.get(command_name) or payload.get(command_name.replace("-", "_"))
        if explicit in (None, ""):
            return None, ""
        explicit_text = str(explicit)
        if os.sep not in explicit_text and "/" not in explicit_text:
            found_explicit = shutil.which(explicit_text)
            if found_explicit:
                return found_explicit, "config"
        command = _resolve_optional_repo_path(explicit, Path(explicit_text))
        if command.exists():
            return str(command), "config"
        raise RuntimeError(f"DeepH CLI {command_name!r} was configured but does not exist: {command}")

    def _deeph_discovery(self, payload: dict[str, Any]) -> dict[str, Any]:
        repo, repo_source, env_var = self._deeph_configured_repo(payload)
        bin_dir = self._deeph_configured_bin_dir(payload)
        sibling_repo = (REPO_ROOT.parent / DEEPH_SIBLING_REPO_NAME).resolve()
        sibling_repo_available = sibling_repo.exists()
        errors: list[str] = []
        if repo is not None and not repo.exists():
            errors.append(f"Configured DeepH repo from {repo_source} does not exist: {repo}")
        if bin_dir is not None and not bin_dir.exists():
            errors.append(f"Configured DeepH bin_dir does not exist: {bin_dir}")

        commands: dict[str, dict[str, str | None]] = {}
        for command_name in DEEPH_CLI_NAMES:
            try:
                explicit, explicit_source = self._deeph_explicit_command(payload, command_name)
            except RuntimeError as exc:
                commands[command_name] = {"path": None, "source": "config", "error": str(exc)}
                errors.append(str(exc))
                continue
            if explicit:
                commands[command_name] = {"path": explicit, "source": explicit_source}
                continue
            if bin_dir is not None and not bin_dir.exists():
                message = f"Configured DeepH bin_dir does not exist: {bin_dir}"
                commands[command_name] = {"path": None, "source": "config", "error": message}
                continue
            if repo is not None and not repo.exists() and bin_dir is None:
                message = f"Configured DeepH repo from {repo_source} does not exist: {repo}"
                commands[command_name] = {"path": None, "source": repo_source, "error": message}
                continue

            candidate: Path | None = None
            candidate_source = "unavailable"
            if bin_dir is not None:
                candidate = bin_dir / command_name
                candidate_source = "config"
            elif repo is not None and repo.exists():
                candidate = repo / ".venv" / "bin" / command_name
                candidate_source = repo_source
            if candidate is not None and candidate.exists():
                commands[command_name] = {"path": str(candidate), "source": candidate_source}
                continue

            found = shutil.which(command_name)
            if found:
                commands[command_name] = {"path": found, "source": "PATH"}
            elif sibling_repo_available:
                candidate = sibling_repo / ".venv" / "bin" / command_name
                if candidate.exists():
                    commands[command_name] = {"path": str(candidate), "source": "sibling_repo"}
                else:
                    missing_hint = (
                        f"DeepH CLI {command_name!r} not found in sibling repository {sibling_repo}. "
                        f"Configure deeph.repo_path/deeph_repo_path, set {DEEPH_PACK_ROOT_ENV}, "
                        f"provide deeph.commands.{command_name}, or install it in PATH."
                    )
                    commands[command_name] = {"path": None, "source": "sibling_repo", "error": missing_hint}
                    errors.append(missing_hint)
            else:
                missing_hint = (
                    f"DeepH CLI {command_name!r} not found. Configure deeph.repo_path/deeph_repo_path, "
                    f"set {DEEPH_PACK_ROOT_ENV}, provide deeph.commands.{command_name}, install it in PATH, "
                    f"or place {DEEPH_SIBLING_REPO_NAME} next to this repository."
                )
                commands[command_name] = {"path": None, "source": "unavailable", "error": missing_hint}
                errors.append(missing_hint)

        python_path: str | None = None
        python_source = "runtime"
        options = self._deeph_options(payload)
        raw_python = options.get("python") or payload.get("deeph_python")
        if raw_python not in (None, ""):
            candidate_python = _resolve_optional_repo_path(raw_python, Path(str(raw_python)))
            if not candidate_python.exists():
                errors.append(f"Configured DeepH Python does not exist: {candidate_python}")
            python_path = str(candidate_python)
            python_source = "config"
        elif repo is not None and repo.exists() and (repo / ".venv" / "bin" / "python").exists():
            python_path = str(repo / ".venv" / "bin" / "python")
            python_source = repo_source
        elif sibling_repo_available and any(
            (commands.get(name) or {}).get("source") == "sibling_repo" for name in DEEPH_CLI_NAMES
        ) and (sibling_repo / ".venv" / "bin" / "python").exists():
            python_path = str(sibling_repo / ".venv" / "bin" / "python")
            python_source = "sibling_repo"
        else:
            python_path = sys.executable

        available_sources = {str(item.get("source")) for item in commands.values() if item.get("path")}
        source = (
            "config"
            if "config" in available_sources or repo_source == "config" and repo is not None
            else "env"
            if "env" in available_sources or repo_source == "env" and repo is not None
            else "PATH"
            if "PATH" in available_sources
            else "sibling_repo"
            if "sibling_repo" in available_sources or repo_source == "sibling_repo" and repo is not None
            else "unavailable"
        )
        return {
            "source": source,
            "repo_path": (
                str(repo)
                if repo is not None
                else str(sibling_repo)
                if "sibling_repo" in available_sources
                else None
            ),
            "repo_source": (
                repo_source
                if repo is not None
                else "sibling_repo"
                if "sibling_repo" in available_sources
                else None
            ),
            "env_var": env_var,
            "bin_dir": str(bin_dir) if bin_dir is not None else None,
            "commands": commands,
            "python": python_path,
            "python_source": python_source,
            "available": not errors and all(commands[name].get("path") for name in DEEPH_CLI_NAMES),
            "errors": errors,
        }

    def _deeph_python(self, payload: dict[str, Any]) -> str:
        discovery = self._deeph_discovery(payload)
        if discovery.get("python_source") == "config" and discovery.get("errors"):
            python_errors = [error for error in discovery["errors"] if "DeepH Python" in error]
            if python_errors:
                raise RuntimeError(python_errors[0])
        return str(discovery["python"])

    def _deeph_command(self, payload: dict[str, Any], command_name: str) -> str:
        discovery = self._deeph_discovery(payload)
        command_info = (discovery.get("commands") or {}).get(command_name) or {}
        command_path = command_info.get("path")
        if command_path:
            return str(command_path)
        error = command_info.get("error") or "; ".join(discovery.get("errors") or [])
        raise RuntimeError(error or f"DeepH CLI {command_name!r} is unavailable.")

    def _prepare_deeph_context(
        self,
        payload: dict[str, Any],
        graph2mat_context: Graph2MatBenchmarkContext,
    ) -> DeepHBenchmarkContext:
        options = self._deeph_options(payload)
        paths = default_deeph_paths(graph2mat_context.run_root)
        ensure_path_inside(paths.root, graph2mat_context.run_root, label="DeepH root")
        frozen_split = load_deeph_json(graph2mat_context.frozen_split_manifest_path)
        seed_value = options.get("seed", payload.get("random_seed", 42))
        seed = 42 if seed_value in (None, "") else int(seed_value)
        raw_mirror = build_deeph_raw_mirror(
            frozen_split,
            raw_dir=paths.raw_dir,
            workspace_root=paths.root,
            seed=seed,
        )
        orbital_json = str(options.get("orbital") or deeph_orbital_json_from_raw_mirror(raw_mirror))
        dataset_name = str(options.get("dataset_name") or DEEPh_DEFAULT_DATASET_NAME)
        early_stopping_policy = parse_early_stopping_policy(payload)
        epochs = int(options.get("epochs", payload.get("deeph_epochs", 100)) or 100)
        if early_stopping_policy is not None:
            epochs = early_stopping_policy.max_epochs
            options = dict(options)
            options["epochs"] = early_stopping_policy.max_epochs
        batch_size = int(options.get("batch_size", 3) or 3)
        learning_rate = float(options.get("learning_rate", 0.001) or 0.001)
        disable_cuda = _parse_bool(options.get("disable_cuda"), True)
        device = str(options.get("device") or ("cpu" if disable_cuda else "cuda:0"))
        multiprocessing = int(options.get("multiprocessing", 0) or 0)
        radius = float(options.get("radius", -1.0) or -1.0)
        num_threads = int(options.get("num_threads", -1) or -1)
        render_preprocess_config(
            paths.preprocess_config,
            raw_dir=paths.raw_dir,
            processed_dir=paths.processed_dir,
            multiprocessing=multiprocessing,
            local_coordinate=_parse_bool(options.get("local_coordinate"), True),
            get_s=_parse_bool(options.get("get_S", options.get("get_s")), True),
            radius=radius,
            julia_interpreter=str(options.get("julia_interpreter") or "julia"),
        )
        render_train_config(
            paths.train_config,
            processed_dir=paths.processed_dir,
            graph_dir=paths.graph_dir,
            save_dir=paths.save_dir,
            dataset_name=dataset_name,
            split_ratios=raw_mirror["split_ratios"],
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            disable_cuda=disable_cuda,
            device=device,
            num_threads=num_threads,
            multiprocessing=multiprocessing,
            radius=radius,
            orbital=orbital_json,
            early_stopping_loss=-1.0 if early_stopping_policy is not None else None,
            early_stopping_loss_epoch=[-1.0, early_stopping_policy.max_epochs + 1]
            if early_stopping_policy is not None
            else None,
            overrides={key: options[key] for key in DEEPH_TRAIN_OVERRIDE_KEYS if key in options},
        )
        inference_configs: list[Path] = []
        inference_work_dirs: list[Path] = []
        deeph_discovery = self._deeph_discovery(payload)
        python_interpreter = self._deeph_python(payload)
        for row in raw_mirror["rows"]:
            if row.get("split") != "test":
                continue
            raw_sample = Path(str(row["raw_dir"]))
            work_dir = paths.inference_dir / raw_sample.name
            inference_config = paths.config_dir / "inference" / f"{raw_sample.name}.ini"
            render_inference_config(
                inference_config,
                work_dir=work_dir,
                trained_model_dir=paths.save_dir,
                python_interpreter=python_interpreter,
                interface=str(options.get("inference_interface") or "openmx"),
                task=list(options.get("inference_task") or [3, 4]),
                disable_cuda=disable_cuda,
                device=device,
                huge_structure=_parse_bool(options.get("huge_structure"), True),
                restore_blocks_py=_parse_bool(options.get("restore_blocks_py"), True),
                radius=radius,
            )
            work_dir.mkdir(parents=True, exist_ok=True)
            inference_configs.append(inference_config)
            inference_work_dirs.append(work_dir)
        context = DeepHBenchmarkContext(
            root=paths.root,
            raw_dir=paths.raw_dir,
            processed_dir=paths.processed_dir,
            graph_dir=paths.graph_dir,
            save_dir=paths.save_dir,
            inference_dir=paths.inference_dir,
            preprocess_config=paths.preprocess_config,
            train_config=paths.train_config,
            inference_configs=inference_configs,
            inference_work_dirs=inference_work_dirs,
            manifest_path=paths.manifest_path,
            deeph_discovery=deeph_discovery,
            split_audit_path=paths.root / "deeph_split_audit.json",
            split_audit_csv_path=paths.root / "deeph_split_audit.csv",
            split_hash=graph2mat_context.split_hash,
            raw_mirror=raw_mirror,
            dry_run=_parse_bool(payload.get("dry_run"), False),
        )
        self._write_deeph_manifest(context, extra={"prepared": True})
        return context

    def _stage_deeph_inference_inputs(self, context: DeepHBenchmarkContext) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for row in context.raw_mirror.get("rows", []):
            if row.get("split") != "test":
                continue
            raw_sample = Path(str(row["raw_dir"]))
            processed_sample = context.processed_dir / raw_sample.name
            work_dir = context.inference_dir / raw_sample.name
            if not processed_sample.exists():
                missing.append(str(processed_sample))
                continue
            work_dir.mkdir(parents=True, exist_ok=True)
            staged_files: list[str] = []
            for item in sorted(processed_sample.iterdir()):
                if item.is_file():
                    _link_or_copy_file(item, work_dir / item.name)
                    staged_files.append(str(work_dir / item.name))
            rows.append(
                {
                    "sample_id": row.get("sample_id"),
                    "processed_sample": str(processed_sample),
                    "work_dir": str(work_dir),
                    "staged_files": staged_files,
                }
            )
        if missing:
            raise RuntimeError(
                "Missing DeepH processed test samples before inference: "
                + ", ".join(missing[:10])
            )
        return {"count": len(rows), "rows": rows}

    def _validate_deeph_training_outputs(self, context: DeepHBenchmarkContext) -> dict[str, Any]:
        expected = [context.save_dir / "config.ini", context.save_dir / "best_state_dict.pkl"]
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise RuntimeError("Missing DeepH training outputs: " + ", ".join(missing))
        return {
            "save_dir": str(context.save_dir),
            "best_state_dict": str(context.save_dir / "best_state_dict.pkl"),
            "best_model": str(context.save_dir / "best_model.pt")
            if (context.save_dir / "best_model.pt").exists()
            else "",
        }

    def _validate_deeph_prediction_outputs(self, context: DeepHBenchmarkContext) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        adapter_results = []
        for work_dir in context.inference_work_dirs:
            candidates = [work_dir / "hamiltonians_pred.h5", work_dir / "rh_pred.h5"]
            present = [path for path in candidates if path.exists() and path.stat().st_size > 0]
            if not present:
                missing.append(str(work_dir))
                adapter = None
            else:
                adapter_result = adapt_deeph_prediction_sample(
                    work_dir=work_dir,
                    processed_sample_dir=context.processed_dir / work_dir.name,
                    sample_id=work_dir.name,
                )
                adapter_results.append(adapter_result)
                adapter = adapter_result.to_dict()
            rows.append(
                {
                    "work_dir": str(work_dir),
                    "prediction_files": [str(path) for path in present],
                    "prediction_sha256": {path.name: _file_sha256(path) for path in present},
                    "adapter": adapter,
                }
            )
        if missing:
            raise RuntimeError("Missing DeepH inference outputs under: " + ", ".join(missing[:10]))
        adapter_manifest = write_adapter_manifest(context.inference_dir / "adapter_manifest.json", adapter_results)
        return {
            "count": len(rows),
            "rows": rows,
            "inference_dir": str(context.inference_dir),
            "adapter_manifest": str(context.inference_dir / "adapter_manifest.json"),
            "adapter_summary": {
                "metrics_ready_count": adapter_manifest["metrics_ready_count"],
                "diagnostic_only_count": adapter_manifest["diagnostic_only_count"],
            },
        }

    def _audit_deeph_split(
        self,
        context: DeepHBenchmarkContext,
        graph2mat_context: Graph2MatBenchmarkContext,
    ) -> dict[str, Any]:
        audit = audit_deeph_split(
            frozen_split_manifest=_load_json(graph2mat_context.frozen_split_manifest_path),
            raw_mirror=context.raw_mirror,
            processed_dir=context.processed_dir,
            train_config_path=context.train_config,
            output_json=context.split_audit_path,
            output_csv=context.split_audit_csv_path,
        )
        if not audit.get("valid"):
            self._write_deeph_manifest(context, split_audit=audit)
            raise RuntimeError(
                "DeepH split audit failed: "
                f"{audit.get('status')}; see {context.split_audit_path}"
            )
        return audit

    def _write_deeph_manifest(
        self,
        context: DeepHBenchmarkContext,
        *,
        preprocess_run: dict[str, Any] | None = None,
        train_run: dict[str, Any] | None = None,
        inference_runs: list[dict[str, Any]] | None = None,
        training_outputs: dict[str, Any] | None = None,
        prediction_outputs: dict[str, Any] | None = None,
        split_audit: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous = _load_json(context.manifest_path) if context.manifest_path.exists() else {}
        payload = {
            "schema": "graph2mat_deeph_deeph_stage_manifest_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "artifact_contract_version": CONTRACT_NAME,
            "context": context.to_dict(),
            "deeph_discovery": context.deeph_discovery,
            "deeph_discovery_source": context.deeph_discovery.get("source"),
            "config_sha256": {
                "preprocess": _file_sha256(context.preprocess_config) if context.preprocess_config.exists() else None,
                "train": _file_sha256(context.train_config) if context.train_config.exists() else None,
                "inference": {
                    str(path): _file_sha256(path) for path in context.inference_configs if path.exists()
                },
            },
            "preprocess_run": preprocess_run if preprocess_run is not None else previous.get("preprocess_run"),
            "train_run": train_run if train_run is not None else previous.get("train_run"),
            "inference_runs": inference_runs if inference_runs is not None else previous.get("inference_runs", []),
            "training_outputs": training_outputs
            if training_outputs is not None
            else previous.get("training_outputs"),
            "prediction_outputs": prediction_outputs
            if prediction_outputs is not None
            else previous.get("prediction_outputs"),
            "split_audit": split_audit if split_audit is not None else previous.get("split_audit"),
            "split_audit_path": str(context.split_audit_path),
            "split_audit_status": (
                (split_audit or previous.get("split_audit") or {}).get("status")
                if (split_audit or previous.get("split_audit"))
                else "pending"
            ),
            "extra": {**(previous.get("extra") or {}), **(extra or {})},
        }
        write_deeph_json(context.manifest_path, payload)
        return payload

    def _training_sweep_datasets(self, validation: dict[str, Any]) -> list[dict[str, Any]]:
        dataset_root = Path(str(validation.get("dataset_root") or ""))
        if not dataset_root:
            raise RuntimeError("training_sweep requires a validated dataset_root.")
        split_path = dataset_root / "frozen_split_manifest.json"
        manifest_path = dataset_root / "benchmark_dataset_manifest.json"
        if not split_path.exists() or not manifest_path.exists():
            raise RuntimeError(
                "training_sweep requires benchmark_dataset_manifest.json and frozen_split_manifest.json "
                f"under {dataset_root}."
            )
        dataset_id = str(
            validation.get("dataset_id")
            or dataset_root.name
            or "dataset_1"
        )
        return [{"dataset_id": dataset_id, "dataset_root": str(dataset_root)}]

    def _training_sweep_datasets_from_dataset_sweep(
        self,
        manifest: dict[str, Any],
        *,
        allow_dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        datasets: list[dict[str, Any]] = []
        for row in manifest.get("rows") or []:
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "")
            if status != "benchmark_ready" and not (allow_dry_run and status == "dry_run"):
                raise RuntimeError(
                    "Full strict pipeline generated a dataset that is not benchmark-ready: "
                    f"{row.get('recipe_id') or row.get('dataset_slug') or row.get('dataset_root')} "
                    f"status={status or 'unknown'}."
                )
            dataset_root = str(row.get("dataset_root") or "").strip()
            if not dataset_root:
                raise RuntimeError(f"Full strict pipeline dataset row has no dataset_root: {row}")
            datasets.append(
                {
                    "dataset_id": str(row.get("recipe_id") or row.get("dataset_slug") or Path(dataset_root).name),
                    "dataset_root": dataset_root,
                }
            )
        if not datasets:
            raise RuntimeError("Full strict pipeline did not produce any datasets for training_sweep.")
        return datasets

    def _child_training_payload(self, payload: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        benchmark_id = str(self._state.run_id or payload.get("run_id") or time.strftime("g2m_deeph_%Y%m%d_%H%M%S"))
        child = dict(payload)
        child["dataset_root"] = record["dataset_root"]
        child["dataset_mode"] = "reuse_validated"
        child["run_id"] = "/".join(
            [
                benchmark_id,
                "sweep",
                str(record["model"]),
                str(record["dataset_id"]),
                str(record["config_id"]),
            ]
        )
        child.pop("dataset_sweep", None)
        child.pop("training_sweep", None)
        if record["model"] == "graph2mat":
            child["graph2mat_overrides"] = dict(record.get("overrides") or {})
        else:
            deeph_options = dict(child.get("deeph") or {})
            deeph_options.update(dict(record.get("overrides") or {}))
            child["deeph"] = deeph_options
        return child

    def _clean_resume_training_run_root(self, child_payload: dict[str, Any], record: dict[str, Any]) -> None:
        if not (
            child_payload.get("resume_from_run_root")
            or child_payload.get("resume_training_sweep_from_run_root")
            or child_payload.get("reuse_dataset_sweep_from_run_root")
            or _parse_bool(child_payload.get("resume_training_sweep"), False)
        ):
            return
        run_root = self._benchmark_run_root(child_payload, str(child_payload["run_id"]))
        if run_root.exists():
            shutil.rmtree(run_root)
            self._logs.append(
                f"[G2M-DEEPH] Cleaned stale incomplete run directory before retry: "
                f"{record.get('dataset_id')} {record.get('config_id')}\n"
            )

    def _stage_reference_metric_result(
        self,
        *,
        frozen_split_manifest: dict[str, Any],
        output_dir: Path,
        dataset_root: Path,
    ) -> Path:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        for row in frozen_split_manifest.get("rows") or []:
            if row.get("split") != "test":
                continue
            sample_id = str(row.get("sample_id") or row.get("graph2mat_sample_id") or row.get("deeph_sample_id") or "").strip()
            sample_dir = Path(str(row.get("sample_dir") or ""))
            if not sample_id or not sample_dir.exists():
                raise RuntimeError(f"Cannot stage DeepH metrics reference row: {row}")
            _link_or_copy_file(sample_dir / "RUN.fdf", output_dir / "structures" / sample_id / "RUN.fdf")
            for artifact in sorted(path for path in sample_dir.iterdir() if path.is_file()):
                if artifact.name == "ML_prediction.HSX":
                    continue
                _link_or_copy_file(artifact, output_dir / "siesta_hamiltonians" / sample_id / artifact.name)
            split_root = dataset_root / "splits"
        if split_root.exists():
            target = output_dir / "splits"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(split_root, target, symlinks=True)
        return output_dir

    def _write_run_cost_telemetry(
        self,
        *,
        model: str,
        run_root: Path,
        frozen_split_manifest_path: Path,
        train_run: dict[str, Any] | None = None,
        predict_run: dict[str, Any] | None = None,
        preprocess_run: dict[str, Any] | None = None,
        inference_runs: list[dict[str, Any]] | None = None,
        metrics_run: dict[str, Any] | None = None,
        training_dir: Path | None = None,
        deeph_save_dir: Path | None = None,
        payload: dict[str, Any] | None = None,
        optimizer_accounting: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        telemetry = summarize_run_telemetry(
            model=model,
            run_root=run_root,
            training_dir=training_dir,
            deeph_save_dir=deeph_save_dir,
            frozen_split_manifest_path=frozen_split_manifest_path,
            train_run=train_run,
            predict_run=predict_run,
            preprocess_run=preprocess_run,
            inference_runs=inference_runs or [],
            metrics_run=metrics_run,
            performance=_performance_settings_from_payload(payload or {}),
            optimizer_accounting=optimizer_accounting,
        )
        telemetry_path = run_root / "telemetry" / f"{model}.json"
        write_telemetry(telemetry_path, telemetry)
        telemetry["telemetry_path"] = str(telemetry_path)
        return telemetry

    def _graph2mat_optimizer_accounting(
        self,
        context: Graph2MatBenchmarkContext,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        training = config.get("training") if isinstance(config.get("training"), dict) else {}
        data = training.get("data") if isinstance(training.get("data"), dict) else {}
        trainer = training.get("trainer") if isinstance(training.get("trainer"), dict) else {}
        return optimizer_update_accounting(
            train_samples=context.runs_json_counts.get("train"),
            batch_size=_optional_int_value(data.get("batch_size")),
            max_epochs=_optional_int_value(trainer.get("max_epochs")),
            gradient_accumulation=_optional_int_value(trainer.get("accumulate_grad_batches")) or 1,
            drop_last=_parse_bool(data.get("drop_last"), False),
        )

    def _run_training_sweep_graph2mat_job(
        self,
        payload: dict[str, Any],
        validation: dict[str, Any],
        record: dict[str, Any],
    ) -> dict[str, Any]:
        child = self._child_training_payload(payload, record)
        self._clean_resume_training_run_root(child, record)
        context = self._prepare_graph2mat_context(child, {**validation, "dataset_root": record["dataset_root"]})
        prepared_manifest = _load_json(context.graph2mat_manifest_path)
        prepared_extra = prepared_manifest.get("extra") if isinstance(prepared_manifest.get("extra"), dict) else {}
        optimizer_accounting = (
            prepared_extra.get("optimizer_update_accounting")
            if isinstance(prepared_extra.get("optimizer_update_accounting"), dict)
            else {}
        )
        result: dict[str, Any] = {
            **record,
            "run_root": str(context.run_root),
            "graph2mat_manifest_path": str(context.graph2mat_manifest_path),
            "optimizer_update_accounting": optimizer_accounting,
        }
        metric_fail_policy = _metric_fail_policy(payload)
        final_mode = is_final_benchmark_mode(payload)
        result["metric_fail_policy"] = metric_fail_policy
        result["fail_open_metric_outputs"] = metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY
        if final_mode:
            result.update(search_stage_record_fields())
        warning = _metric_fail_policy_warning(metric_fail_policy)
        if warning:
            result["severe_warnings"] = [warning]
        if context.dry_run:
            result["status"] = "dry_run"
            return result
        train_run = self._run_command(
            [self._graph2mat_python(child), str(DEFAULT_MD_TRAINING_SCRIPT)],
            cwd=REPO_ROOT,
            env=self._graph2mat_command_env(context, child),
            label=f"Graph2Mat sweep train {record['config_id']}",
            progress_provider=lambda: _lightning_training_progress(context.training_dir),
        )
        early_stopping_policy = parse_early_stopping_policy(child)
        early_stopping_metadata = (
            tensorboard_policy_metadata(context.training_dir, early_stopping_policy)
            if early_stopping_policy is not None
            else None
        )
        checkpoint_manifest = _load_json(context.training_dir / "checkpoint_manifest.json")
        self._write_graph2mat_manifest(
            context,
            checkpoint_manifest=checkpoint_manifest,
            extra={
                "training_completed": True,
                "training_run": train_run,
                "sweep_record": record,
                "early_stopping": early_stopping_metadata,
                "optimizer_update_accounting": optimizer_accounting,
                **(search_stage_record_fields() if final_mode else {}),
            },
        )
        if final_mode:
            telemetry = self._write_run_cost_telemetry(
                model="graph2mat",
                run_root=context.run_root,
                frozen_split_manifest_path=context.frozen_split_manifest_path,
                train_run=train_run,
                training_dir=context.training_dir,
                payload=child,
                optimizer_accounting=optimizer_accounting,
            )
            self._write_graph2mat_manifest(
                context,
                checkpoint_manifest=checkpoint_manifest,
                extra={
                    "training_completed": True,
                    "training_run": train_run,
                    "sweep_record": record,
                    "telemetry": telemetry,
                    "early_stopping": early_stopping_metadata,
                    "optimizer_update_accounting": optimizer_accounting,
                    **search_stage_record_fields(),
                },
            )
            self._logs.append(
                f"[G2M-DEEPH] Graph2Mat sweep {record['config_id']}: {TEST_METRICS_LOCKED_MESSAGE}\n"
            )
            result.update(
                {
                    "status": "completed",
                    "train_run": train_run,
                    "telemetry": telemetry,
                    "telemetry_path": telemetry["telemetry_path"],
                    "early_stopping": early_stopping_metadata,
                    "optimizer_update_accounting": optimizer_accounting,
                    **search_stage_record_fields(),
                }
            )
            return result
        predict_run = self._run_command(
            [self._graph2mat_python(child), str(DEFAULT_MD_PREDICTION_SCRIPT)],
            cwd=REPO_ROOT,
            env=self._graph2mat_command_env(context, child),
            label=f"Graph2Mat sweep predict {record['config_id']}",
        )
        prediction_outputs = self._validate_graph2mat_prediction_outputs(context)
        self._write_graph2mat_manifest(
            context,
            checkpoint_manifest=checkpoint_manifest,
            prediction_outputs=prediction_outputs,
            extra={"prediction_completed": True, "prediction_run": predict_run, "sweep_record": record},
        )
        metrics_root = context.run_root / "metrics" / "graph2mat"
        staged = stage_graph2mat_metric_result(
            frozen_split_manifest=_load_json(context.frozen_split_manifest_path),
            prediction_structs_dir=context.prediction_structs_dir,
            output_dir=metrics_root / "eval_input",
            dataset_root=context.dataset_root,
        )
        metrics_run = self._run_command(
            [
                self._graph2mat_python(child),
                str(DEFAULT_HAMILTONIAN_METRICS_SCRIPT),
                str(staged.result_dir),
                "--workers",
                "1",
                "--enable-kpoint-metrics",
                "--overwrite",
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            label=f"Graph2Mat sweep metrics {record['config_id']}",
            allowed_returncodes=_metric_allowed_returncodes(metric_fail_policy),
        )
        telemetry = self._write_run_cost_telemetry(
            model="graph2mat",
            run_root=context.run_root,
            frozen_split_manifest_path=context.frozen_split_manifest_path,
            train_run=train_run,
            predict_run=predict_run,
            metrics_run=metrics_run,
            training_dir=context.training_dir,
            payload=child,
            optimizer_accounting=optimizer_accounting,
        )
        self._write_graph2mat_manifest(
            context,
            checkpoint_manifest=checkpoint_manifest,
            prediction_outputs=prediction_outputs,
                            extra={
                                "prediction_completed": True,
                                "prediction_run": predict_run,
                                "sweep_record": record,
                                "telemetry": telemetry,
                                "early_stopping": early_stopping_metadata,
                                "optimizer_update_accounting": optimizer_accounting,
                            },
                        )
        result.update(
            {
                "status": "completed",
                "train_run": train_run,
                "predict_run": predict_run,
                "metrics_run": metrics_run,
                "telemetry": telemetry,
                "telemetry_path": telemetry["telemetry_path"],
                "early_stopping": early_stopping_metadata,
                "optimizer_update_accounting": optimizer_accounting,
            }
        )
        return result

    def _run_training_sweep_deeph_job(
        self,
        payload: dict[str, Any],
        validation: dict[str, Any],
        record: dict[str, Any],
    ) -> dict[str, Any]:
        child = self._child_training_payload(payload, record)
        self._clean_resume_training_run_root(child, record)
        graph_context = self._prepare_graph2mat_context(child, {**validation, "dataset_root": record["dataset_root"]})
        deeph_context = self._prepare_deeph_context(child, graph_context)
        result: dict[str, Any] = {
            **record,
            "run_root": str(graph_context.run_root),
            "deeph_manifest_path": str(deeph_context.manifest_path),
        }
        metric_fail_policy = _metric_fail_policy(payload)
        final_mode = is_final_benchmark_mode(payload)
        result["metric_fail_policy"] = metric_fail_policy
        result["fail_open_metric_outputs"] = metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY
        if final_mode:
            result.update(search_stage_record_fields())
        warning = _metric_fail_policy_warning(metric_fail_policy)
        if warning:
            result["severe_warnings"] = [warning]
        if deeph_context.dry_run:
            result["status"] = "dry_run"
            return result
        preprocess_run = self._run_command(
            [self._deeph_command(child, "deeph-preprocess"), "--config", str(deeph_context.preprocess_config)],
            cwd=deeph_context.root,
            env=self._deeph_command_env(child),
            label=f"DeepH sweep preprocess {record['config_id']}",
        )
        split_audit = self._audit_deeph_split(deeph_context, graph_context)
        self._write_deeph_manifest(
            deeph_context,
            preprocess_run=preprocess_run,
            split_audit=split_audit,
            extra={"sweep_record": record},
        )
        early_stopping_policy = parse_early_stopping_policy(child)
        deeph_early_stopping = DeepHEarlyStoppingObserver(early_stopping_policy) if early_stopping_policy else None
        train_run = self._run_command(
            [self._deeph_command(child, "deeph-train"), "--config", str(deeph_context.train_config)],
            cwd=deeph_context.root,
            env=self._deeph_command_env(child),
            label=f"DeepH sweep train {record['config_id']}",
            line_observer=deeph_early_stopping,
        )
        early_stopping_metadata = deeph_early_stopping.metadata() if deeph_early_stopping is not None else None
        training_outputs = self._validate_deeph_training_outputs(deeph_context)
        self._write_deeph_manifest(
            deeph_context,
            train_run=train_run,
            training_outputs=training_outputs,
            extra={
                "sweep_record": record,
                "early_stopping": early_stopping_metadata,
                **(search_stage_record_fields() if final_mode else {}),
            },
        )
        if final_mode:
            telemetry = self._write_run_cost_telemetry(
                model="deeph",
                run_root=graph_context.run_root,
                frozen_split_manifest_path=graph_context.frozen_split_manifest_path,
                preprocess_run=preprocess_run,
                train_run=train_run,
                deeph_save_dir=deeph_context.save_dir,
                payload=child,
            )
            self._write_deeph_manifest(
                deeph_context,
                preprocess_run=preprocess_run,
                train_run=train_run,
                training_outputs=training_outputs,
                split_audit=split_audit,
                extra={
                    "sweep_record": record,
                    "telemetry": telemetry,
                    "early_stopping": early_stopping_metadata,
                    **search_stage_record_fields(),
                },
            )
            self._logs.append(
                f"[G2M-DEEPH] DeepH sweep {record['config_id']}: {TEST_METRICS_LOCKED_MESSAGE}\n"
            )
            result.update(
                {
                    "status": "completed",
                    "preprocess_run": preprocess_run,
                    "train_run": train_run,
                    "telemetry": telemetry,
                    "telemetry_path": telemetry["telemetry_path"],
                    "early_stopping": early_stopping_metadata,
                    **search_stage_record_fields(),
                }
            )
            return result
        staged_inputs = self._stage_deeph_inference_inputs(deeph_context)
        inference_runs: list[dict[str, Any]] = []
        for inference_config in deeph_context.inference_configs:
            inference_runs.append(
                self._run_command(
                    [self._deeph_command(child, "deeph-inference"), "--config", str(inference_config)],
                    cwd=deeph_context.root,
                    env=self._deeph_command_env(child),
                    label=f"DeepH sweep inference {record['config_id']} {inference_config.stem}",
                )
            )
        prediction_outputs = self._validate_deeph_prediction_outputs(deeph_context)
        self._write_deeph_manifest(
            deeph_context,
            inference_runs=inference_runs,
            training_outputs=training_outputs,
            prediction_outputs=prediction_outputs,
            extra={"inference_inputs": staged_inputs, "sweep_record": record},
        )
        metrics_root = graph_context.run_root / "metrics" / "deeph"
        reference_dir = self._stage_reference_metric_result(
            frozen_split_manifest=_load_json(graph_context.frozen_split_manifest_path),
            output_dir=metrics_root / "reference_input",
            dataset_root=graph_context.dataset_root,
        )
        staged_deeph = stage_deeph_metric_inputs(
            raw_mirror=deeph_context.raw_mirror,
            processed_dir=deeph_context.processed_dir,
            inference_dir=deeph_context.inference_dir,
            output_dir=metrics_root / "deeph_inputs",
        )
        deeph_metric_command = _deeph_metric_command_args(
            python_executable=self._graph2mat_python(child),
            graph2mat_result_dir=reference_dir,
            processed_dir=staged_deeph.processed_dir,
            predictions_dir=staged_deeph.predictions_dir,
            output_dir=metrics_root / "eval",
            metric_fail_policy=metric_fail_policy,
        )
        metrics_run = self._run_command(
            deeph_metric_command,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            label=f"DeepH sweep metrics {record['config_id']}",
            allowed_returncodes=_metric_allowed_returncodes(metric_fail_policy),
        )
        telemetry = self._write_run_cost_telemetry(
            model="deeph",
            run_root=graph_context.run_root,
            frozen_split_manifest_path=graph_context.frozen_split_manifest_path,
            preprocess_run=preprocess_run,
            train_run=train_run,
            inference_runs=inference_runs,
            metrics_run=metrics_run,
            deeph_save_dir=deeph_context.save_dir,
            payload=child,
        )
        self._write_deeph_manifest(
            deeph_context,
            preprocess_run=preprocess_run,
            train_run=train_run,
            inference_runs=inference_runs,
            training_outputs=training_outputs,
            prediction_outputs=prediction_outputs,
            split_audit=split_audit,
            extra={
                "inference_inputs": staged_inputs,
                "sweep_record": record,
                "telemetry": telemetry,
                "early_stopping": early_stopping_metadata,
            },
        )
        result.update(
            {
                "status": "completed",
                "preprocess_run": preprocess_run,
                "train_run": train_run,
                "inference_runs": inference_runs,
                "metrics_run": metrics_run,
                "telemetry": telemetry,
                "telemetry_path": telemetry["telemetry_path"],
                "early_stopping": early_stopping_metadata,
            }
        )
        return result

    def _write_training_sweep_summary(self, run_root: Path, summary: dict[str, Any]) -> None:
        sweep_root = run_root / "sweep"
        _write_json(sweep_root / "training_sweep_manifest.json", summary)
        rows = summary.get("runs") or []
        csv_path = sweep_root / "training_sweep_metrics.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows if isinstance(row, dict) for key in row}) or ["status"]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _write_training_search_plan(self, run_root: Path, plan: dict[str, Any]) -> None:
        search_plan = plan.get("search_plan") if isinstance(plan.get("search_plan"), dict) else None
        if search_plan is None:
            search_policy = plan.get("search_policy") if isinstance(plan.get("search_policy"), dict) else {}
            search_plan = {
                "schema": "graph2mat_deeph_training_search_plan_v1",
                "strategy": str(search_policy.get("strategy") or "grid"),
                "planned_run_count": len(plan.get("planned_runs") or []),
                "planned_runs": list(plan.get("planned_runs") or []),
            }
        _write_json(run_root / "sweep" / "search_plan.json", search_plan)

    def _write_budget_summary(self, run_root: Path, tracker: BudgetTracker) -> dict[str, Any]:
        summary = tracker.summary()
        write_budget_summary(run_root / "sweep" / "budget_summary.json", summary)
        return summary

    def _resume_run_root(self, payload: dict[str, Any], *, default_run_root: Path | None = None) -> Path | None:
        raw = (
            payload.get("resume_from_run_root")
            or payload.get("resume_training_sweep_from_run_root")
            or payload.get("reuse_dataset_sweep_from_run_root")
        )
        if raw in (None, "") and _parse_bool(payload.get("resume_training_sweep"), False):
            raw = default_run_root
        if raw in (None, ""):
            return None
        path = Path(_expand_repo_tokens(raw)).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve(strict=False)

    def _resume_training_run_root(self, payload: dict[str, Any], *, default_run_root: Path | None = None) -> Path | None:
        raw = payload.get("resume_from_run_root") or payload.get("resume_training_sweep_from_run_root")
        if raw in (None, "") and _parse_bool(payload.get("resume_training_sweep"), False):
            raw = default_run_root
        if raw in (None, ""):
            return None
        path = Path(_expand_repo_tokens(raw)).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve(strict=False)

    def _load_resume_training_sweep(
        self,
        payload: dict[str, Any],
        *,
        run_root: Path,
    ) -> tuple[Path | None, dict[str, dict[str, Any]]]:
        resume_root = self._resume_training_run_root(payload, default_run_root=run_root)
        if resume_root is None:
            return None, {}
        manifest_path = resume_root / "sweep" / "training_sweep_manifest.json"
        if not manifest_path.exists():
            self._logs.append(f"[G2M-DEEPH][WARN] Resume requested but no training sweep manifest exists: {manifest_path}\n")
            return resume_root, {}
        previous = _load_json(manifest_path)
        completed: dict[str, dict[str, Any]] = {}
        for row in previous.get("runs") or []:
            if not isinstance(row, dict) or row.get("status") != "completed":
                continue
            completed["|".join(_training_record_key(row))] = row
        self._logs.append(
            f"[G2M-DEEPH] Resuming training sweep from {manifest_path}; "
            f"{len(completed)} completed runs will be reused.\n"
        )
        return resume_root, completed

    def _dataset_sweep_summary_from_existing_datasets(
        self,
        payload: dict[str, Any],
        sweep_info: dict[str, Any],
        *,
        run_root: Path,
    ) -> dict[str, Any]:
        base_dataset_root = _resolve_optional_repo_path(payload.get("dataset_root"), DEFAULT_DATASET_ROOT)
        system_label = str(payload.get("system_label") or "graphene")
        split_mode = str(payload.get("split_mode") or "blocked_with_gap").strip().lower()
        rows: list[dict[str, Any]] = []
        for spec in sweep_info.get("md_dataset_specs") or []:
            dataset_slug = str(spec.get("dataset_slug") or spec.get("recipe_id") or "").strip()
            if not dataset_slug:
                raise RuntimeError(f"Cannot reuse dataset sweep row without dataset_slug/recipe_id: {spec}")
            dataset_root = base_dataset_root / dataset_slug
            if not dataset_root.exists():
                raise RuntimeError(
                    "Cannot reuse dataset sweep without rerunning SIESTA; missing dataset directory: "
                    f"{dataset_root}"
                )
            validation = self._validate_swept_dataset(dataset_root, system_label)
            rows.append(
                {
                    "recipe_id": str(spec.get("recipe_id") or dataset_slug),
                    "recipe_label": str(spec.get("label") or spec.get("recipe_id") or dataset_slug),
                    "dataset_slug": dataset_slug,
                    "dataset_root": str(dataset_root),
                    "dataset_size": int(spec.get("size") or 0),
                    "blocks": json.dumps(spec.get("temperature_blocks") or [], sort_keys=True, ensure_ascii=False),
                    "split_counts": json.dumps(spec.get("split_counts") or {}, sort_keys=True),
                    "split_mode": split_mode,
                    "temporal_gap": int(payload.get("temporal_gap", 1 if split_mode == "blocked_with_gap" else 0) or 0),
                    "reserved_gap_frames": "",
                    "config_path": "",
                    "compatibility_hash": stable_payload_hash(
                        {
                            "dataset_slug": dataset_slug,
                            "recipe": spec.get("recipe_metadata") or spec,
                            "artifact_contract_version": CONTRACT_NAME,
                        },
                        length=16,
                    ),
                    "artifact_contract_version": CONTRACT_NAME,
                    "status": "benchmark_ready",
                    "generation_seconds": 0.0,
                    "elapsed_seconds": 0.0,
                    "artifact_validation_status": "valid",
                    "artifact_validation_path": str(dataset_root / "artifact_validation.json"),
                    "benchmark_manifest_path": validation["benchmark_dataset_manifest"],
                    "frozen_split_manifest_path": validation["frozen_split_manifest"],
                    "source": "reused_existing_dataset_sweep_without_summary",
                }
            )
        if not rows:
            raise RuntimeError("Cannot reuse dataset sweep; payload did not resolve to any dataset specs.")
        manifest = self._write_dataset_sweep_summary(
            run_root,
            payload=payload,
            sweep_info=sweep_info,
            rows=rows,
        )
        self._logs.append(
            "[G2M-DEEPH] Reconstructed dataset sweep summary from existing validated datasets; "
            "SIESTA generation is skipped.\n"
        )
        return manifest

    def _load_resume_dataset_sweep(self, payload: dict[str, Any], *, run_root: Path) -> dict[str, Any] | None:
        resume_root = self._resume_run_root(payload, default_run_root=run_root)
        if resume_root is None:
            return None
        summary_path = resume_root / "summary" / "dataset_sweep_summary.json"
        if not summary_path.exists():
            sweep_info = self.dataset_sweep_info_from_payload(payload)
            if sweep_info is not None:
                self._logs.append(
                    f"[G2M-DEEPH][WARN] Resume dataset sweep summary not found: {summary_path}. "
                    "Trying to reuse existing validated dataset directories from the payload recipes.\n"
                )
                return self._dataset_sweep_summary_from_existing_datasets(
                    payload,
                    sweep_info,
                    run_root=run_root,
                )
            raise RuntimeError(f"resume_from_run_root has no dataset sweep summary: {summary_path}")
        manifest = _load_json(summary_path)
        self._logs.append(f"[G2M-DEEPH] Reusing dataset sweep summary from {summary_path}; SIESTA generation is skipped.\n")
        return manifest

    def _run_ranking(
        self,
        run_root: Path,
        *,
        validation: dict[str, Any] | None = None,
        training_sweep_manifest_path: Path | None = None,
        common_metrics_manifest_path: Path | None = None,
        dataset_root: Path | None = None,
        frozen_split_manifest_path: Path | None = None,
        dataset_manifest_path: Path | None = None,
    ) -> dict[str, Any]:
        validation = validation or {}
        dataset_root = dataset_root or Path(str(validation.get("dataset_root") or ""))
        if dataset_root and not frozen_split_manifest_path:
            frozen_split_manifest_path = dataset_root / "frozen_split_manifest.json"
        if dataset_root and not dataset_manifest_path:
            dataset_manifest_path = dataset_root / "benchmark_dataset_manifest.json"
        self._logs.append(f"[G2M-DEEPH] Ranking best runs: {run_root / 'summary' / 'ranking'}\n")
        ranking = rank_graph2mat_deeph_runs(
            run_root=run_root,
            output_dir=run_root / "summary" / "ranking",
            training_sweep_manifest_path=training_sweep_manifest_path,
            common_metrics_manifest_path=common_metrics_manifest_path,
            dataset_root=dataset_root if dataset_root and dataset_root.exists() else None,
            frozen_split_manifest_path=frozen_split_manifest_path,
            dataset_manifest_path=dataset_manifest_path,
        )
        _write_json(run_root / "summary" / "ranking_manifest.json", ranking)
        return ranking

    def _run_training_sweep(
        self,
        payload: dict[str, Any],
        validation: dict[str, Any],
        plan: dict[str, Any],
    ) -> None:
        planned = list(plan.get("planned_runs") or [])
        error_policy = str(plan.get("error_policy") or "continue_on_error")
        metric_fail_policy = _metric_fail_policy(payload)
        final_mode = is_final_benchmark_mode(payload)
        protocol_stage = protocol_stage_from_payload(payload, default=SEARCH_STAGE if final_mode else "exploratory")
        run_root = Path(str(self._state.run_root or self._benchmark_run_root(payload, str(self._state.run_id))))
        resume_root, completed_by_key = self._load_resume_training_sweep(payload, run_root=run_root)
        budget_tracker = BudgetTracker(plan.get("budget_policy"))
        budget_tracker.add_completed_many(list(completed_by_key.values()), source="resume_manifest")
        summary: dict[str, Any] = {
            "schema": "graph2mat_deeph_training_sweep_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "running",
            "protocol_stage": protocol_stage,
            "final_benchmark_mode": final_mode,
            "final_test_locked": final_mode,
            "error_policy": error_policy,
            "metric_fail_policy": metric_fail_policy,
            "fail_open_metric_outputs": metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
            "resumed_from_run_root": str(resume_root) if resume_root else "",
            "planned_runs": planned,
            "search_policy": plan.get("search_policy"),
            "budget_policy": plan.get("budget_policy"),
            "search_plan_path": str(run_root / "sweep" / "search_plan.json"),
            "budget_summary_path": str(run_root / "sweep" / "budget_summary.json"),
            "runs": list(completed_by_key.values()),
            "failed_runs": [],
            "skipped_runs": [],
        }
        self._write_training_search_plan(run_root, plan)
        summary["budget"] = self._write_budget_summary(run_root, budget_tracker)
        self._set_stage("training_sweep")
        with self._lock:
            self._state.training_sweep_status = {
                "enabled": True,
                "total": len(planned),
                "completed": len(completed_by_key),
                "failed": 0,
                "graph2mat_parallelism": _graph2mat_training_parallelism(payload),
                "deeph_parallelism": _deeph_training_parallelism(payload),
                "active_model": None,
                "active_dataset": None,
                "active_config_id": None,
                "active_runs": [],
            }
        graph2mat_parallelism = _graph2mat_training_parallelism(payload)
        deeph_parallelism = _deeph_training_parallelism(payload)
        summary["graph2mat_parallelism"] = graph2mat_parallelism
        summary["deeph_parallelism"] = deeph_parallelism
        self._logs.append(
            f"[G2M-DEEPH] Training sweep: {len(planned)} planned runs; "
            f"Graph2Mat parallel jobs={graph2mat_parallelism}; "
            f"DeepH parallel jobs={deeph_parallelism}.\n"
        )
        if final_mode:
            self._logs.append(f"[G2M-DEEPH] {TEST_METRICS_LOCKED_MESSAGE}\n")
        deeph_options = self._deeph_options(payload)
        deeph_inner_workers = int(deeph_options.get("multiprocessing", 0) or 0)
        if deeph_parallelism > 1 and deeph_inner_workers > 1:
            self._logs.append(
                "[G2M-DEEPH][WARN] DeepH parallel sweep jobs share CPU/GPU resources while "
                f"deeph.multiprocessing={deeph_inner_workers}; consider lowering one of them.\n"
            )

        def record_success(result: dict[str, Any]) -> None:
            if result.get("status") == "completed":
                budget_tracker.add_completed(result)
            else:
                budget_tracker.release(result)
            summary["runs"].append(result)
            summary["budget"] = self._write_budget_summary(run_root, budget_tracker)
            with self._lock:
                self._state.training_sweep_status["completed"] += 1

        def record_failure(record: dict[str, Any], exc: Exception) -> None:
            budget_tracker.release(record)
            failed = {**record, "status": "failed", "error": str(exc)}
            summary["runs"].append(failed)
            summary["failed_runs"].append(failed)
            with self._lock:
                self._state.training_sweep_status["failed"] += 1
            self._logs.append(f"[G2M-DEEPH][WARN] Training sweep run failed: {record.get('config_id')}: {exc}\n")

        def record_budget_skip(record: dict[str, Any]) -> None:
            skipped = budget_tracker.skip_for_budget(record)
            summary["runs"].append(skipped)
            summary["skipped_runs"].append(skipped)
            summary["budget"] = self._write_budget_summary(run_root, budget_tracker)
            self._logs.append(
                f"[G2M-DEEPH][BUDGET] Skipping {record.get('model')} {record.get('config_id')}: "
                f"{skipped['budget_skip_reason']}\n"
            )

        index = 0
        while index < len(planned):
            if self._state.stop_requested:
                summary["status"] = "stopped"
                break
            record = planned[index]
            resume_key = "|".join(_training_record_key(record))
            if resume_key in completed_by_key:
                self._logs.append(
                    f"[G2M-DEEPH] Skipping completed run from resume manifest: "
                    f"{record.get('dataset_id')} {record.get('config_id')}\n"
                )
                index += 1
                continue
            if not budget_tracker.can_schedule(record):
                record_budget_skip(record)
                index += 1
                self._write_training_sweep_summary(run_root, summary)
                continue
            if record.get("model") == "graph2mat" and graph2mat_parallelism > 1:
                batch: list[dict[str, Any]] = []
                while index < len(planned) and len(batch) < graph2mat_parallelism:
                    candidate = planned[index]
                    candidate_key = "|".join(_training_record_key(candidate))
                    if candidate.get("model") != "graph2mat" or candidate_key in completed_by_key:
                        break
                    if not budget_tracker.can_schedule(candidate):
                        record_budget_skip(candidate)
                        index += 1
                        continue
                    budget_tracker.reserve(candidate)
                    batch.append(candidate)
                    index += 1
                if not batch:
                    continue
                with self._lock:
                    self._state.training_sweep_status.update(
                        {
                            "active_model": "graph2mat_parallel" if len(batch) > 1 else "graph2mat",
                            "active_dataset": ",".join(str(item.get("dataset_id")) for item in batch),
                            "active_config_id": ",".join(str(item.get("config_id")) for item in batch),
                            "active_runs": [
                                {
                                    "model": item.get("model"),
                                    "dataset_id": item.get("dataset_id"),
                                    "config_id": item.get("config_id"),
                                }
                                for item in batch
                            ],
                        }
                    )
                self._logs.append(
                    "[G2M-DEEPH][PERF] Running Graph2Mat sweep batch: "
                    + ", ".join(str(item.get("config_id")) for item in batch)
                    + "\n"
                )
                first_error: Exception | None = None
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(batch),
                    thread_name_prefix="g2m-deeph-graph2mat",
                ) as executor:
                    futures = {
                        executor.submit(self._run_training_sweep_graph2mat_job, payload, validation, item): item
                        for item in batch
                    }
                    for future in concurrent.futures.as_completed(futures):
                        item = futures[future]
                        try:
                            record_success(future.result())
                        except Exception as exc:
                            record_failure(item, exc)
                            if first_error is None:
                                first_error = exc
                self._write_training_sweep_summary(run_root, summary)
                if first_error is not None and error_policy == "fail_fast":
                    raise first_error
                continue
            if record.get("model") == "deeph" and deeph_parallelism > 1:
                batch = []
                while index < len(planned) and len(batch) < deeph_parallelism:
                    candidate = planned[index]
                    candidate_key = "|".join(_training_record_key(candidate))
                    if candidate.get("model") != "deeph" or candidate_key in completed_by_key:
                        break
                    if not budget_tracker.can_schedule(candidate):
                        record_budget_skip(candidate)
                        index += 1
                        continue
                    budget_tracker.reserve(candidate)
                    batch.append(candidate)
                    index += 1
                if not batch:
                    continue
                with self._lock:
                    self._state.training_sweep_status.update(
                        {
                            "active_model": "deeph_parallel" if len(batch) > 1 else "deeph",
                            "active_dataset": ",".join(str(item.get("dataset_id")) for item in batch),
                            "active_config_id": ",".join(str(item.get("config_id")) for item in batch),
                            "active_runs": [
                                {
                                    "model": item.get("model"),
                                    "dataset_id": item.get("dataset_id"),
                                    "config_id": item.get("config_id"),
                                }
                                for item in batch
                            ],
                        }
                    )
                self._logs.append(
                    "[G2M-DEEPH][PERF] Running DeepH sweep batch: "
                    + ", ".join(str(item.get("config_id")) for item in batch)
                    + "\n"
                )
                first_error = None
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(batch),
                    thread_name_prefix="g2m-deeph-deeph",
                ) as executor:
                    futures = {
                        executor.submit(self._run_training_sweep_deeph_job, payload, validation, item): item
                        for item in batch
                    }
                    for future in concurrent.futures.as_completed(futures):
                        item = futures[future]
                        try:
                            record_success(future.result())
                        except Exception as exc:
                            record_failure(item, exc)
                            if first_error is None:
                                first_error = exc
                self._write_training_sweep_summary(run_root, summary)
                if first_error is not None and error_policy == "fail_fast":
                    raise first_error
                continue
            index += 1
            with self._lock:
                self._state.training_sweep_status.update(
                    {
                        "active_model": record.get("model"),
                        "active_dataset": record.get("dataset_id"),
                        "active_config_id": record.get("config_id"),
                        "active_runs": [
                            {
                                "model": record.get("model"),
                                "dataset_id": record.get("dataset_id"),
                                "config_id": record.get("config_id"),
                            }
                        ],
                    }
                )
            try:
                budget_tracker.reserve(record)
                if record["model"] == "graph2mat":
                    result = self._run_training_sweep_graph2mat_job(payload, validation, record)
                elif record["model"] == "deeph":
                    result = self._run_training_sweep_deeph_job(payload, validation, record)
                else:
                    raise RuntimeError(f"Unsupported training sweep model: {record.get('model')}")
                record_success(result)
            except Exception as exc:
                record_failure(record, exc)
                if error_policy == "fail_fast":
                    raise
            self._write_training_sweep_summary(run_root, summary)
        if summary.get("status") == "stopped":
            self._write_training_sweep_summary(run_root, summary)
            with self._lock:
                self._state.training_sweep_status.update(
                    {
                        "active_model": None,
                        "active_dataset": None,
                        "active_config_id": None,
                        "active_runs": [],
                        "status": "stopped",
                    }
                )
            self._set_stage("complete")
            self._finish(returncode=130, error="Stop requested.")
            return
        summary["status"] = "completed" if not summary["failed_runs"] else "completed_with_failures"
        summary["budget"] = self._write_budget_summary(run_root, budget_tracker)
        self._write_training_sweep_summary(run_root, summary)
        if final_mode:
            test_blindness = build_search_stage_manifest(run_root=run_root, summary=summary, payload=payload)
            dry_run = _parse_bool(payload.get("dry_run"), False)
            selection: dict[str, Any] = {
                "status": "skipped_dry_run" if dry_run else "not_started",
                "reason": "dry_run does not execute training jobs, so top-k selection is pending."
                if dry_run
                else "",
            }
            if not dry_run:
                policy = selection_policy_from_payload(payload)
                if not policy.get("final_seeds"):
                    selection = {
                        "status": "pending_missing_final_seeds",
                        "reason": "No final_seeds were provided; top-k robust rerun planning is pending.",
                        "policy": policy,
                    }
                else:
                    selected_configs = select_top_configs(
                        list(summary.get("runs") or []),
                        metric=str(policy["metric"]),
                        mode=str(policy["mode"]),
                        k_per_model=int(policy["k_per_model"]),
                        grouping=str(policy["grouping"]),
                        allow_diagnostic=False,
                    )
                    robust_plan = generate_robust_rerun_plan(
                        selected_configs,
                        final_seeds=list(policy.get("final_seeds") or []),
                    )
                    paths = write_selection_artifacts(run_root / "summary" / "selection", selected_configs, robust_plan)
                    selection = {
                        "status": "planned",
                        "selected_configs": selected_configs,
                        "robust_rerun_plan": robust_plan,
                        "paths": paths,
                    }
                    test_blindness["selected_final_runs"] = robust_plan.get("planned_runs") or []
                    test_blindness["final_test_status"] = "pending_final_test"
                    test_blindness["selected_configs_path"] = paths["selected_configs_json"]
                    test_blindness["robust_rerun_plan_path"] = paths["robust_rerun_plan_json"]
                    _write_json(run_root / "summary" / "test_blindness_manifest.json", test_blindness)
                summary["selection"] = selection
                self._write_training_sweep_summary(run_root, summary)
            message = (
                "Final/publicable search completed test-blind. Test prediction, test metrics, "
                "ranking and winner claims are locked until validation-based top-k selection "
                "and the explicit final_test stage."
            )
            with self._lock:
                self._state.training_sweep_status.update(
                    {
                        "active_model": None,
                        "active_dataset": None,
                        "active_config_id": None,
                        "active_runs": [],
                        "status": summary["status"],
                        "protocol_stage": SEARCH_STAGE,
                        "final_test_locked": True,
                    }
                )
                self._last_results = {
                    "dry_run": _parse_bool(payload.get("dry_run"), False),
                    "contract_name": CONTRACT_NAME,
                    "dataset_validation": validation,
                    "dataset_sweep": payload.get("_dataset_sweep_manifest"),
                    "training_sweep": summary,
                    "test_blindness": test_blindness,
                    "selection": selection,
                    "ranking": None,
                    "message": message,
                }
            self._finish(returncode=0 if not summary["failed_runs"] else 1)
            return
        self._set_stage("ranking")
        ranking = self._run_ranking(
            run_root,
            validation=validation,
            training_sweep_manifest_path=run_root / "sweep" / "training_sweep_manifest.json",
        )
        dataset_sweep_manifest = payload.get("_dataset_sweep_manifest")
        message = (
            "Full strict pipeline completed: dataset sweep generation, validation, training, "
            "prediction, metrics and ranking were chained."
            if isinstance(dataset_sweep_manifest, dict)
            else "Training sweep completed. SIESTA was not invoked by the sweep runner."
        )
        with self._lock:
            self._state.training_sweep_status.update(
                {
                    "active_model": None,
                    "active_dataset": None,
                    "active_config_id": None,
                    "active_runs": [],
                    "status": summary["status"],
                }
            )
            self._last_results = {
                "dry_run": _parse_bool(payload.get("dry_run"), False),
                "contract_name": CONTRACT_NAME,
                "dataset_validation": validation,
                "dataset_sweep": dataset_sweep_manifest,
                "training_sweep": summary,
                "ranking": ranking,
                "message": message,
            }
        self._finish(returncode=0 if not summary["failed_runs"] else 1)

    def _run_full_strict_pipeline(self, payload: dict[str, Any], sweep_info: dict[str, Any]) -> None:
        training_sweep = payload.get("training_sweep") if isinstance(payload.get("training_sweep"), dict) else {}
        if not _parse_bool(training_sweep.get("enabled"), False):
            raise RuntimeError("full_strict_pipeline requires training_sweep.enabled=true.")
        run_root = Path(str(self._state.run_root or self._benchmark_run_root(payload, self._state.run_id or "full_strict_pipeline")))
        dry_run = _parse_bool(payload.get("dry_run"), False)
        self._logs.append(
            "[G2M-DEEPH] Full strict pipeline: dataset generation + training sweep + metrics/ranking.\n"
        )
        dataset_sweep_manifest = self._load_resume_dataset_sweep(payload, run_root=run_root)
        if dataset_sweep_manifest is None:
            dataset_sweep_manifest = self._run_dataset_sweep_generation(payload, sweep_info, run_root=run_root)
        datasets = self._training_sweep_datasets_from_dataset_sweep(
            dataset_sweep_manifest,
            allow_dry_run=dry_run,
        )
        plan = expand_training_sweep(training_sweep, datasets=datasets)
        payload["_dataset_sweep_manifest"] = dataset_sweep_manifest

        if dry_run:
            summary: dict[str, Any] = {
                "schema": "graph2mat_deeph_training_sweep_v1",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": "planned_dry_run",
                "error_policy": str(plan.get("error_policy") or "continue_on_error"),
                "metric_fail_policy": _metric_fail_policy(payload),
                "fail_open_metric_outputs": _metric_fail_policy(payload) == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
                "planned_runs": list(plan.get("planned_runs") or []),
                "runs": [],
                "failed_runs": [],
                "graph2mat_parallelism": _graph2mat_training_parallelism(payload),
                "deeph_parallelism": _deeph_training_parallelism(payload),
            }
            self._set_stage("training_sweep")
            with self._lock:
                self._state.training_sweep_status = {
                    "enabled": True,
                    "total": len(summary["planned_runs"]),
                    "completed": 0,
                "failed": 0,
                "status": "planned_dry_run",
                "graph2mat_parallelism": _graph2mat_training_parallelism(payload),
                "deeph_parallelism": _deeph_training_parallelism(payload),
            }
            self._logs.append(
                f"[G2M-DEEPH] Full strict dry-run: {len(datasets)} datasets and "
                f"{len(summary['planned_runs'])} training runs planned.\n"
            )
            self._write_training_sweep_summary(run_root, summary)
            with self._lock:
                self._last_results = {
                    "dry_run": True,
                    "contract_name": CONTRACT_NAME,
                    "dataset_sweep": dataset_sweep_manifest,
                    "training_sweep": summary,
                    "message": (
                        "Full strict pipeline dry-run planned dataset generation and training sweep; "
                        "no SIESTA, Graph2Mat or DeepH subprocess was launched."
                    ),
                }
            self._set_stage("complete")
            self._finish(returncode=0)
            return

        validation = {
            "contract_name": CONTRACT_NAME,
            "benchmark_ready": True,
            "repair_required": False,
            "dataset_root": datasets[0]["dataset_root"],
            "snapshot_root": datasets[0]["dataset_root"],
            "artifact_summary": {
                "total_snapshots": sum(int(row.get("dataset_size") or 0) for row in dataset_sweep_manifest.get("rows") or []),
                "valid_snapshots": sum(int(row.get("dataset_size") or 0) for row in dataset_sweep_manifest.get("rows") or []),
                "invalid_snapshots": 0,
                "repair_required_snapshots": 0,
                "missing_required_counts": {},
            },
            "errors": [],
            "warnings": [],
            "manifest_paths": {},
            "dataset_sweep": dataset_sweep_manifest,
        }
        self._run_training_sweep(payload, validation, plan)

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload or {})
        dataset_mode = str(payload.get("dataset_mode") or "reuse_validated").strip() or "reuse_validated"
        run_mode = str(payload.get("run_mode") or "").strip()
        training_sweep_requested = _parse_bool((payload.get("training_sweep") or {}).get("enabled"), False) if isinstance(payload.get("training_sweep"), dict) else False
        dataset_sweep_info = self.dataset_sweep_info_from_payload(payload)
        full_strict_pipeline = run_mode == FULL_STRICT_PIPELINE_RUN_MODE or dataset_mode == FULL_STRICT_PIPELINE_RUN_MODE
        generate_datasets_only = (
            not full_strict_pipeline
            and (
                run_mode == DATASET_SWEEP_RUN_MODE
                or (dataset_mode == "generate_new" and dataset_sweep_info is not None)
            )
        )
        if full_strict_pipeline:
            if dataset_sweep_info is None:
                raise RuntimeError("full_strict_pipeline requires an enabled dataset_sweep with at least one MD dataset.")
            if not training_sweep_requested:
                raise RuntimeError("full_strict_pipeline requires training_sweep.enabled=true.")
            payload["run_mode"] = FULL_STRICT_PIPELINE_RUN_MODE
            payload["dataset_mode"] = FULL_STRICT_PIPELINE_RUN_MODE
        if training_sweep_requested and not full_strict_pipeline and (generate_datasets_only or dataset_mode == "generate_new"):
            raise RuntimeError(
                "training_sweep only runs on already validated/reused joint datasets. "
                "Generate and validate datasets first, then launch the training sweep."
            )
        if generate_datasets_only:
            payload["run_mode"] = DATASET_SWEEP_RUN_MODE
        if dataset_sweep_info is not None and not generate_datasets_only and not full_strict_pipeline:
            raise RuntimeError(
                "dataset_sweep v1 solo se puede ejecutar con run_mode='generate_datasets_only' "
                "o 'full_strict_pipeline'."
            )
        validation = (
            {
                "contract_name": CONTRACT_NAME,
                "benchmark_ready": True,
                "repair_required": False,
                "dataset_root": str(_resolve_optional_repo_path(payload.get("dataset_root"), DEFAULT_DATASET_ROOT)),
                "snapshot_root": str(_resolve_optional_repo_path(payload.get("dataset_root"), DEFAULT_DATASET_ROOT)),
                "artifact_summary": {
                    "total_snapshots": 0,
                    "valid_snapshots": 0,
                    "invalid_snapshots": 0,
                    "repair_required_snapshots": 0,
                    "missing_required_counts": {},
                },
                "errors": [],
                "warnings": [],
                "manifest_paths": {},
                "dataset_sweep": dataset_sweep_info,
            }
            if generate_datasets_only or full_strict_pipeline
            else self.validate_dataset_payload(payload)
        )
        allow_repair = _parse_bool(payload.get("allow_repair"), False) or _parse_bool(
            payload.get("repair_mode"),
            False,
        )
        if not generate_datasets_only and not full_strict_pipeline and not validation["benchmark_ready"] and not allow_repair:
            raise RuntimeError(
                _format_validation_not_ready_message(
                    validation,
                    dataset_mode=dataset_mode,
                    allow_repair=allow_repair,
                )
            )
        if training_sweep_requested and allow_repair:
            raise RuntimeError("training_sweep cannot run in repair mode.")
        if training_sweep_requested and not full_strict_pipeline:
            payload["_training_sweep_plan"] = expand_training_sweep(
                payload.get("training_sweep"),
                datasets=self._training_sweep_datasets(validation),
            )

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Graph2Mat/DeepH benchmark runner is already active.")
            now = time.time()
            run_id = str(payload.get("run_id") or time.strftime("g2m_deeph_%Y%m%d_%H%M%S"))
            self._logs = [
                "[G2M-DEEPH] Benchmark runner iniciado.\n",
                f"[G2M-DEEPH] Run id: {run_id}\n",
                f"[G2M-DEEPH] Dataset root: {validation['dataset_root']}\n",
                (
                    "[G2M-DEEPH] Full strict pipeline mode: dataset sweep + training/test chain.\n"
                    if full_strict_pipeline
                    else
                    "[G2M-DEEPH] Dataset sweep mode: generate_datasets_only.\n"
                    if generate_datasets_only
                    else "[G2M-DEEPH] Training sweep enabled.\n"
                    if training_sweep_requested
                    else "[G2M-DEEPH] Phase 7: Graph2Mat and DeepH command chain enabled.\n"
                ),
            ]
            self._phase_timings = []
            run_root = self._benchmark_run_root(payload, run_id)
            self._state = G2MDeepHRunState(
                running=True,
                stage="validate_inputs",
                started_at=now,
                stop_requested=False,
                run_id=run_id,
                dataset_root=validation["dataset_root"],
                run_root=str(run_root),
                benchmark_manifest_path=(
                    str(run_root / "benchmark_manifest.yaml")
                    if generate_datasets_only or full_strict_pipeline
                    else validation["manifest_paths"]["benchmark_dataset_manifest"]["path"]
                ),
                frozen_split_manifest_path=(
                    None
                    if generate_datasets_only or full_strict_pipeline
                    else validation["manifest_paths"]["frozen_split_manifest"]["path"]
                ),
                training_sweep_status=(
                    {
                        "enabled": True,
                        "total": len((payload.get("_training_sweep_plan") or {}).get("planned_runs") or []),
                        "completed": 0,
                        "failed": 0,
                        "graph2mat_parallelism": _graph2mat_training_parallelism(payload),
                        "deeph_parallelism": _deeph_training_parallelism(payload),
                    }
                    if training_sweep_requested
                    else {}
                ),
            )
            if allow_repair:
                self._state.warnings.append(
                    "repair_mode_requested: repair is explicit but not implemented in this phase"
                )
            self._dataset_validation = validation
            self._last_results = None
            self._thread = threading.Thread(
                target=self._run_workflow,
                args=(payload, validation, allow_repair),
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._state.stop_requested = True
            for process in list(self._processes):
                if process.poll() is None:
                    process.terminate()
            if self._thread is None or not self._thread.is_alive():
                self._state.running = False
                self._state.stage = "stopped"
                self._state.finished_at = self._state.finished_at or time.time()
            self._logs.append("[G2M-DEEPH] Solicitud de parada recibida.\n")
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            self._state.running = running
            elapsed = None
            if self._state.started_at is not None:
                end = time.time() if running else self._state.finished_at or time.time()
                elapsed = end - self._state.started_at
            return {
                "running": running,
                "stage": self._state.stage,
                "returncode": None if running else self._state.returncode,
                "started_at": self._state.started_at,
                "finished_at": self._state.finished_at,
                "elapsed_seconds": elapsed,
                "stop_requested": self._state.stop_requested,
                "error": self._state.error,
                "warnings": list(self._state.warnings),
                "run_id": self._state.run_id,
                "dataset_root": self._state.dataset_root,
                "run_root": self._state.run_root,
                "graph2mat_config_path": self._state.graph2mat_config_path,
                "graph2mat_training_dir": self._state.graph2mat_training_dir,
                "deeph_manifest_path": self._state.deeph_manifest_path,
                "deeph_processed_dir": self._state.deeph_processed_dir,
                "deeph_save_dir": self._state.deeph_save_dir,
                "benchmark_manifest_path": self._state.benchmark_manifest_path,
                "frozen_split_manifest_path": self._state.frozen_split_manifest_path,
                "contract_name": CONTRACT_NAME,
                "phases": list(RUNNER_PHASES),
                "log_size": len(self._logs),
                "active_processes": len([process for process in self._processes if process.poll() is None]),
                "dataset_validation": self._dataset_validation,
                "training_sweep": dict(self._state.training_sweep_status),
            }

    def logs(self, since: int = 0, limit: int | None = DEFAULT_LOG_RESPONSE_LIMIT) -> dict[str, Any]:
        with self._lock:
            payload = _bounded_log_payload(self._logs, since=since, limit=limit)
            payload["status"] = self.status()
            return payload

    def results(self) -> dict[str, Any]:
        with self._lock:
            plot_payload = self._build_plot_payload_locked()
            return {
                "available": self._last_results is not None,
                "results": self._last_results,
                "plot_payload": plot_payload,
                "status": self.status(),
            }

    def plots(self) -> dict[str, Any]:
        with self._lock:
            return self._build_plot_payload_locked()

    def _optional_json(self, path_value: Any) -> dict[str, Any]:
        if not path_value:
            return {}
        path = Path(str(path_value))
        if not path.exists():
            return {}
        try:
            return _load_json(path)
        except Exception:
            return {}

    def _append_timing_row(
        self,
        rows: list[dict[str, Any]],
        *,
        phase: str,
        label: str,
        elapsed_seconds: Any,
        source: str,
        status: str = "available",
    ) -> None:
        try:
            elapsed = float(elapsed_seconds)
        except (TypeError, ValueError):
            elapsed = None
            status = "missing"
        rows.append(
            {
                "phase": phase,
                "label": label,
                "elapsed_seconds": elapsed,
                "source": source,
                "status": status,
            }
        )

    def _timing_rows_locked(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        results = self._last_results or {}
        graph2mat = results.get("graph2mat") or {}
        deeph = results.get("deeph") or {}
        common = results.get("common_metrics") or {}
        graph2mat_manifest = self._optional_json(graph2mat.get("graph2mat_manifest_path"))
        deeph_manifest = self._optional_json(deeph.get("manifest_path"))
        graph_extra = graph2mat_manifest.get("extra") or {}

        phase_elapsed = {
            str(row.get("phase")): row.get("elapsed_seconds")
            for row in self._phase_timings
            if row.get("phase")
        }
        self._append_timing_row(
            rows,
            phase="dataset_generation",
            label="Dataset generation / validation",
            elapsed_seconds=phase_elapsed.get("generate_or_validate_joint_dataset"),
            source="runner_phase",
        )
        self._append_timing_row(
            rows,
            phase="graph2mat_train",
            label="Graph2Mat train",
            elapsed_seconds=(graph_extra.get("training_run") or {}).get("elapsed_seconds")
            or phase_elapsed.get("graph2mat_train"),
            source="graph2mat_manifest",
        )
        self._append_timing_row(
            rows,
            phase="graph2mat_predict",
            label="Graph2Mat predict",
            elapsed_seconds=(graph_extra.get("prediction_run") or {}).get("elapsed_seconds")
            or phase_elapsed.get("graph2mat_predict"),
            source="graph2mat_manifest",
        )
        self._append_timing_row(
            rows,
            phase="deeph_preprocess",
            label="DeepH preprocess",
            elapsed_seconds=(deeph_manifest.get("preprocess_run") or {}).get("elapsed_seconds")
            or phase_elapsed.get("deeph_preprocess"),
            source="deeph_manifest",
        )
        self._append_timing_row(
            rows,
            phase="deeph_train",
            label="DeepH train",
            elapsed_seconds=(deeph_manifest.get("train_run") or {}).get("elapsed_seconds")
            or phase_elapsed.get("deeph_train"),
            source="deeph_manifest",
        )
        inference_runs = deeph_manifest.get("inference_runs") or []
        inference_elapsed = sum(
            float(run.get("elapsed_seconds") or 0)
            for run in inference_runs
            if isinstance(run, dict)
        )
        self._append_timing_row(
            rows,
            phase="deeph_predict",
            label="DeepH predict",
            elapsed_seconds=inference_elapsed or phase_elapsed.get("deeph_predict"),
            source="deeph_manifest",
        )
        metric_runs = common.get("runs") or {}
        metrics_elapsed = sum(
            float(run.get("elapsed_seconds") or 0)
            for run in metric_runs.values()
            if isinstance(run, dict)
        )
        self._append_timing_row(
            rows,
            phase="metrics",
            label="Common metrics",
            elapsed_seconds=metrics_elapsed or phase_elapsed.get("common_metrics"),
            source="common_metrics_manifest",
        )
        return rows

    def _dataset_size_from_root(self, dataset_root: str | Path | None) -> int | None:
        if not dataset_root:
            return None
        root = Path(str(dataset_root))
        for path in (root / "benchmark_dataset_manifest.json", root / "artifact_validation.json"):
            try:
                payload = self._optional_json(str(path))
            except Exception:
                payload = {}
            if not payload:
                continue
            samples = payload.get("samples")
            if isinstance(samples, list):
                return len(samples)
            total = (
                payload.get("total_snapshots")
                or payload.get("valid_snapshots")
                or (payload.get("artifact_summary") or {}).get("total_snapshots")
            )
            try:
                if total is not None:
                    return int(total)
            except (TypeError, ValueError):
                pass
        split = self._optional_json(str(root / "frozen_split_manifest.json"))
        rows = split.get("rows")
        if isinstance(rows, list):
            return len(rows)
        counts = split.get("split_counts")
        if isinstance(counts, dict):
            try:
                return sum(int(value) for value in counts.values())
            except (TypeError, ValueError):
                return None
        return None

    def _dataset_sweep_size_map(self, dataset_sweep: dict[str, Any]) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for row in dataset_sweep.get("rows") or []:
            if not isinstance(row, dict):
                continue
            try:
                size = int(row.get("dataset_size"))
            except (TypeError, ValueError):
                continue
            for key in ("recipe_id", "dataset_slug", "dataset_root"):
                value = str(row.get(key) or "").strip()
                if value:
                    sizes[value] = size
        return sizes

    def _append_timing_scaling_row(
        self,
        rows: list[dict[str, Any]],
        *,
        dataset_id: str,
        dataset_root: str,
        dataset_size: int | None,
        phase: str,
        label: str,
        elapsed_seconds: Any,
        source: str,
        model: str = "",
        config_id: str = "",
        epochs: Any = None,
        epoch_label: str = "",
        status: str = "available",
    ) -> None:
        try:
            elapsed = float(elapsed_seconds)
        except (TypeError, ValueError):
            return
        if elapsed < 0:
            return
        try:
            size_value = int(dataset_size) if dataset_size is not None else None
        except (TypeError, ValueError):
            size_value = None
        if size_value is None or size_value <= 0:
            return
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_root": dataset_root,
                "dataset_size": size_value,
                "phase": phase,
                "label": label,
                "model": model,
                "config_id": config_id,
                "epochs": epochs,
                "epoch_label": epoch_label,
                "elapsed_seconds": elapsed,
                "seconds_per_snapshot": elapsed / size_value if size_value else None,
                "source": source,
                "status": status,
            }
        )

    def _run_dataset_root_from_archive(self, run_root: Path) -> Path | None:
        graph_manifest = self._optional_json(str(run_root / "graph2mat" / "graph2mat_manifest.json"))
        context = graph_manifest.get("context") or {}
        dataset_root = context.get("dataset_root")
        if dataset_root:
            return Path(str(dataset_root))
        ranking = self._optional_json(str(run_root / "summary" / "ranking" / "normalized_run_metrics.json"))
        ranking_rows = ranking.get("rows") if isinstance(ranking.get("rows"), list) else []
        for row in ranking_rows:
            if isinstance(row, dict) and row.get("reference_dir"):
                return Path(str(row["reference_dir"]))
        return None

    def _append_metric_scaling_rows_from_summary(
        self,
        rows: list[dict[str, Any]],
        *,
        run_root: Path,
        summary: dict[str, Any],
        dataset_root: Path | None,
        dataset_size: int | None,
        source: str,
    ) -> None:
        if dataset_size is None or dataset_size <= 0:
            return
        run_id = run_root.name
        for source_row in summary.get("summary_rows") or []:
            if not isinstance(source_row, dict):
                continue
            method = str(source_row.get("method") or source_row.get("model") or "")
            for key, value in source_row.items():
                if not key.endswith("_mean"):
                    continue
                try:
                    metric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(metric_value):
                    continue
                rows.append(
                    {
                        "run_id": run_id,
                        "dataset_id": Path(str(dataset_root)).name if dataset_root else "",
                        "dataset_root": str(dataset_root or ""),
                        "dataset_size": int(dataset_size),
                        "method": method,
                        "metric_key": key,
                        "metric_value": metric_value,
                        "scientific_status": summary.get("status") or source_row.get("scientific_status") or "",
                        "diagnostic_only": bool(source_row.get("diagnostic_only"))
                        or str(summary.get("status") or "") == "diagnostic_only",
                        "source": source,
                    }
                )

    def _append_metric_scaling_rows_from_normalized(
        self,
        rows: list[dict[str, Any]],
        *,
        run_root: Path,
        normalized: dict[str, Any],
    ) -> None:
        run_id = run_root.name
        for source_row in normalized.get("rows") or []:
            if not isinstance(source_row, dict):
                continue
            dataset_root = Path(str(source_row.get("reference_dir") or "")) if source_row.get("reference_dir") else None
            dataset_size = self._dataset_size_from_root(dataset_root)
            if dataset_size is None or dataset_size <= 0:
                continue
            for key, value in source_row.items():
                if not key.endswith("_mean"):
                    continue
                try:
                    metric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(metric_value):
                    continue
                rows.append(
                    {
                        "run_id": run_id,
                        "dataset_id": str(source_row.get("dataset_id") or (dataset_root.name if dataset_root else "")),
                        "dataset_root": str(dataset_root or ""),
                        "dataset_size": int(dataset_size),
                        "method": str(source_row.get("model") or source_row.get("method") or ""),
                        "config_id": str(source_row.get("config_id") or ""),
                        "epochs": source_row.get("epochs"),
                        "epoch_label": source_row.get("epoch_label") or (
                            f"{source_row.get('epochs')} epochs"
                            if source_row.get("epochs") not in (None, "")
                            else ""
                        ),
                        "metric_key": key,
                        "metric_value": metric_value,
                        "scientific_status": source_row.get("scientific_status") or "",
                        "diagnostic_only": bool(source_row.get("diagnostic_only")),
                        "source": "ranking_normalized_run_metrics",
                    }
                )

    def _archive_plot_rows_locked(self) -> dict[str, list[dict[str, Any]]]:
        timing_scaling_rows: list[dict[str, Any]] = []
        metric_scaling_rows: list[dict[str, Any]] = []
        if not DEFAULT_OUTPUT_ROOT.exists():
            return {"timing_scaling_rows": [], "metric_scaling_rows": []}
        for run_root in sorted(path for path in DEFAULT_OUTPUT_ROOT.iterdir() if path.is_dir()):
            dataset_sweep = self._optional_json(str(run_root / "summary" / "dataset_sweep_summary.json"))
            size_map = self._dataset_sweep_size_map(dataset_sweep)
            for row in dataset_sweep.get("rows") or []:
                if not isinstance(row, dict):
                    continue
                self._append_timing_scaling_row(
                    timing_scaling_rows,
                    dataset_id=str(row.get("recipe_id") or row.get("dataset_slug") or ""),
                    dataset_root=str(row.get("dataset_root") or ""),
                    dataset_size=row.get("dataset_size"),
                    phase="dataset_generation",
                    label="Dataset generation",
                    elapsed_seconds=row.get("generation_seconds") or row.get("elapsed_seconds"),
                    source="archived_dataset_sweep_summary",
                    model="dataset",
                    config_id=str(row.get("recipe_id") or ""),
                    status=str(row.get("status") or ""),
                )

            training_sweep = self._optional_json(str(run_root / "sweep" / "training_sweep_manifest.json"))
            for record in training_sweep.get("runs") or []:
                if not isinstance(record, dict):
                    continue
                dataset_id = str(record.get("dataset_id") or "")
                dataset_root = str(record.get("dataset_root") or "")
                dataset_size = size_map.get(dataset_id) or size_map.get(dataset_root) or self._dataset_size_from_root(dataset_root)
                model = str(record.get("model") or "")
                config_id = str(record.get("config_id") or "")
                phase_specs = (
                    [
                        ("graph2mat_train", "Graph2Mat train", (record.get("train_run") or {}).get("elapsed_seconds")),
                        ("graph2mat_predict", "Graph2Mat predict", (record.get("predict_run") or {}).get("elapsed_seconds")),
                    ]
                    if model == "graph2mat"
                    else [
                        ("deeph_preprocess", "DeepH preprocess", (record.get("preprocess_run") or {}).get("elapsed_seconds")),
                        ("deeph_train", "DeepH train", (record.get("train_run") or {}).get("elapsed_seconds")),
                        (
                            "deeph_predict",
                            "DeepH predict",
                            sum(
                                float(run.get("elapsed_seconds") or 0)
                                for run in record.get("inference_runs") or []
                                if isinstance(run, dict)
                            ),
                        ),
                    ]
                    if model == "deeph"
                    else []
                )
                phase_specs.append(("metrics", "Metrics", (record.get("metrics_run") or {}).get("elapsed_seconds")))
                for phase, label, elapsed in phase_specs:
                    self._append_timing_scaling_row(
                        timing_scaling_rows,
                        dataset_id=dataset_id,
                        dataset_root=dataset_root,
                        dataset_size=dataset_size,
                        phase=phase,
                        label=label,
                        elapsed_seconds=elapsed,
                        source="archived_training_sweep_manifest",
                        model=model,
                        config_id=config_id,
                        epochs=_training_record_epochs(record),
                        epoch_label=_training_record_epoch_label(record),
                        status=str(record.get("status") or ""),
                    )

            dataset_root = self._run_dataset_root_from_archive(run_root)
            dataset_size = self._dataset_size_from_root(dataset_root)
            graph_manifest = self._optional_json(str(run_root / "graph2mat" / "graph2mat_manifest.json"))
            deeph_manifest = self._optional_json(str(run_root / "deeph" / "deeph_manifest.json"))
            graph_extra = graph_manifest.get("extra") or {}
            self._append_timing_scaling_row(
                timing_scaling_rows,
                dataset_id=dataset_root.name if dataset_root else "",
                dataset_root=str(dataset_root or ""),
                dataset_size=dataset_size,
                phase="graph2mat_train",
                label="Graph2Mat train",
                elapsed_seconds=(graph_extra.get("training_run") or {}).get("elapsed_seconds"),
                source="archived_graph2mat_manifest",
                model="graph2mat",
                config_id="default_graph2mat",
            )
            self._append_timing_scaling_row(
                timing_scaling_rows,
                dataset_id=dataset_root.name if dataset_root else "",
                dataset_root=str(dataset_root or ""),
                dataset_size=dataset_size,
                phase="graph2mat_predict",
                label="Graph2Mat predict",
                elapsed_seconds=(graph_extra.get("prediction_run") or {}).get("elapsed_seconds"),
                source="archived_graph2mat_manifest",
                model="graph2mat",
                config_id="default_graph2mat",
            )
            self._append_timing_scaling_row(
                timing_scaling_rows,
                dataset_id=dataset_root.name if dataset_root else "",
                dataset_root=str(dataset_root or ""),
                dataset_size=dataset_size,
                phase="deeph_preprocess",
                label="DeepH preprocess",
                elapsed_seconds=(deeph_manifest.get("preprocess_run") or {}).get("elapsed_seconds"),
                source="archived_deeph_manifest",
                model="deeph",
                config_id="default_deeph",
            )
            self._append_timing_scaling_row(
                timing_scaling_rows,
                dataset_id=dataset_root.name if dataset_root else "",
                dataset_root=str(dataset_root or ""),
                dataset_size=dataset_size,
                phase="deeph_train",
                label="DeepH train",
                elapsed_seconds=(deeph_manifest.get("train_run") or {}).get("elapsed_seconds"),
                source="archived_deeph_manifest",
                model="deeph",
                config_id="default_deeph",
            )
            inference_elapsed = sum(
                float(run.get("elapsed_seconds") or 0)
                for run in deeph_manifest.get("inference_runs") or []
                if isinstance(run, dict)
            )
            self._append_timing_scaling_row(
                timing_scaling_rows,
                dataset_id=dataset_root.name if dataset_root else "",
                dataset_root=str(dataset_root or ""),
                dataset_size=dataset_size,
                phase="deeph_predict",
                label="DeepH predict",
                elapsed_seconds=inference_elapsed,
                source="archived_deeph_manifest",
                model="deeph",
                config_id="default_deeph",
            )

            normalized = self._optional_json(str(run_root / "summary" / "ranking" / "normalized_run_metrics.json"))
            if normalized.get("rows"):
                self._append_metric_scaling_rows_from_normalized(
                    metric_scaling_rows,
                    run_root=run_root,
                    normalized=normalized,
                )
            else:
                common_summary = self._optional_json(str(run_root / "common_metrics" / "summary" / "common_summary.json"))
                self._append_metric_scaling_rows_from_summary(
                    metric_scaling_rows,
                    run_root=run_root,
                    summary=common_summary,
                    dataset_root=dataset_root,
                    dataset_size=dataset_size,
                    source="archived_common_summary",
                )

        def dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            seen: set[str] = set()
            unique: list[dict[str, Any]] = []
            for item in items:
                key = json.dumps(item, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(item)
            return unique

        return {
            "timing_scaling_rows": dedupe(timing_scaling_rows),
            "metric_scaling_rows": dedupe(metric_scaling_rows),
        }

    def _timing_scaling_rows_locked(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        results = self._last_results or {}
        dataset_sweep = results.get("dataset_sweep") if isinstance(results.get("dataset_sweep"), dict) else {}
        size_map = self._dataset_sweep_size_map(dataset_sweep)
        for row in dataset_sweep.get("rows") or []:
            if not isinstance(row, dict):
                continue
            self._append_timing_scaling_row(
                rows,
                dataset_id=str(row.get("recipe_id") or row.get("dataset_slug") or ""),
                dataset_root=str(row.get("dataset_root") or ""),
                dataset_size=row.get("dataset_size"),
                phase="dataset_generation",
                label="Dataset generation",
                elapsed_seconds=row.get("generation_seconds") or row.get("elapsed_seconds"),
                source="dataset_sweep_summary",
                model="dataset",
                config_id=str(row.get("recipe_id") or ""),
                status=str(row.get("status") or ""),
            )
        training_sweep = results.get("training_sweep") if isinstance(results.get("training_sweep"), dict) else {}
        for record in training_sweep.get("runs") or []:
            if not isinstance(record, dict):
                continue
            dataset_id = str(record.get("dataset_id") or "")
            dataset_root = str(record.get("dataset_root") or "")
            dataset_size = size_map.get(dataset_id) or size_map.get(dataset_root) or self._dataset_size_from_root(dataset_root)
            model = str(record.get("model") or "")
            config_id = str(record.get("config_id") or "")
            if model == "graph2mat":
                self._append_timing_scaling_row(
                    rows,
                    dataset_id=dataset_id,
                    dataset_root=dataset_root,
                    dataset_size=dataset_size,
                    phase="graph2mat_train",
                    label="Graph2Mat train",
                    elapsed_seconds=(record.get("train_run") or {}).get("elapsed_seconds"),
                    source="training_sweep_manifest",
                    model=model,
                    config_id=config_id,
                    epochs=_training_record_epochs(record),
                    epoch_label=_training_record_epoch_label(record),
                    status=str(record.get("status") or ""),
                )
                self._append_timing_scaling_row(
                    rows,
                    dataset_id=dataset_id,
                    dataset_root=dataset_root,
                    dataset_size=dataset_size,
                    phase="graph2mat_predict",
                    label="Graph2Mat predict",
                    elapsed_seconds=(record.get("predict_run") or {}).get("elapsed_seconds"),
                    source="training_sweep_manifest",
                    model=model,
                    config_id=config_id,
                    epochs=_training_record_epochs(record),
                    epoch_label=_training_record_epoch_label(record),
                    status=str(record.get("status") or ""),
                )
            elif model == "deeph":
                self._append_timing_scaling_row(
                    rows,
                    dataset_id=dataset_id,
                    dataset_root=dataset_root,
                    dataset_size=dataset_size,
                    phase="deeph_preprocess",
                    label="DeepH preprocess",
                    elapsed_seconds=(record.get("preprocess_run") or {}).get("elapsed_seconds"),
                    source="training_sweep_manifest",
                    model=model,
                    config_id=config_id,
                    epochs=_training_record_epochs(record),
                    epoch_label=_training_record_epoch_label(record),
                    status=str(record.get("status") or ""),
                )
                self._append_timing_scaling_row(
                    rows,
                    dataset_id=dataset_id,
                    dataset_root=dataset_root,
                    dataset_size=dataset_size,
                    phase="deeph_train",
                    label="DeepH train",
                    elapsed_seconds=(record.get("train_run") or {}).get("elapsed_seconds"),
                    source="training_sweep_manifest",
                    model=model,
                    config_id=config_id,
                    epochs=_training_record_epochs(record),
                    epoch_label=_training_record_epoch_label(record),
                    status=str(record.get("status") or ""),
                )
                inference_elapsed = sum(
                    float(run.get("elapsed_seconds") or 0)
                    for run in record.get("inference_runs") or []
                    if isinstance(run, dict)
                )
                self._append_timing_scaling_row(
                    rows,
                    dataset_id=dataset_id,
                    dataset_root=dataset_root,
                    dataset_size=dataset_size,
                    phase="deeph_predict",
                    label="DeepH predict",
                    elapsed_seconds=inference_elapsed,
                    source="training_sweep_manifest",
                    model=model,
                    config_id=config_id,
                    epochs=_training_record_epochs(record),
                    epoch_label=_training_record_epoch_label(record),
                    status=str(record.get("status") or ""),
                )
            self._append_timing_scaling_row(
                rows,
                dataset_id=dataset_id,
                dataset_root=dataset_root,
                dataset_size=dataset_size,
                phase="metrics",
                label="Metrics",
                elapsed_seconds=(record.get("metrics_run") or {}).get("elapsed_seconds"),
                source="training_sweep_manifest",
                model=model,
                config_id=config_id,
                epochs=_training_record_epochs(record),
                epoch_label=_training_record_epoch_label(record),
                status=str(record.get("status") or ""),
            )
        return rows

    def _build_plot_payload_locked(self) -> dict[str, Any]:
        results = self._last_results or {}
        common_metrics = results.get("common_metrics") if isinstance(results, dict) else None
        artifact_summary = {}
        if self._dataset_validation:
            artifact_summary = dict(self._dataset_validation.get("artifact_summary") or {})
        archive_rows = self._archive_plot_rows_locked()
        current_timing_scaling = self._timing_scaling_rows_locked()
        timing_scaling_rows = [
            *current_timing_scaling,
            *(archive_rows.get("timing_scaling_rows") or []),
        ]
        current_metric_scaling: list[dict[str, Any]] = []
        if self._state.run_root:
            current_metric_scaling = live_metric_scaling_rows(Path(str(self._state.run_root)))
        metric_scaling_rows = dedupe_metric_rows(
            [
                *current_metric_scaling,
                *(archive_rows.get("metric_scaling_rows") or []),
            ]
        )
        payload = build_common_plot_payload(
            common_metrics if isinstance(common_metrics, dict) else None,
            artifact_summary=artifact_summary,
            timing_rows=self._timing_rows_locked(),
            timing_scaling_rows=timing_scaling_rows,
            metric_scaling_rows=metric_scaling_rows,
            status_payload=self.status(),
        )
        payload["live_metric_rows"] = len(current_metric_scaling)
        payload["archived_runs"] = len(
            {
                str(row.get("run_id"))
                for row in archive_rows.get("metric_scaling_rows") or []
                if row.get("run_id")
            }
        )
        payload["archived_timing_runs"] = len(
            {
                str(row.get("dataset_root") or row.get("run_id") or row.get("config_id"))
                for row in archive_rows.get("timing_scaling_rows") or []
                if row.get("dataset_root") or row.get("run_id") or row.get("config_id")
            }
        )
        ranking = results.get("ranking") if isinstance(results, dict) else None
        if isinstance(ranking, dict):
            payload["ranking"] = ranking
            payload["available"] = bool(payload.get("available")) or bool(ranking.get("metric_rows_count"))
        return payload

    def _set_stage(self, stage: str) -> None:
        with self._lock:
            self._state.stage = stage
            self._logs.append(f"[G2M-DEEPH] Stage: {stage}\n")

    def _finish(self, *, returncode: int, error: str | None = None) -> None:
        with self._lock:
            self._state.running = False
            self._state.returncode = returncode
            self._state.error = error
            self._state.finished_at = time.time()
            if error:
                self._logs.append(f"[G2M-DEEPH][ERROR] {error}\n")
            self._logs.append(f"[G2M-DEEPH] Finalizado con codigo {returncode}.\n")

    def _run_workflow(
        self,
        payload: dict[str, Any],
        validation: dict[str, Any],
        allow_repair: bool,
    ) -> None:
        context: Graph2MatBenchmarkContext | None = None
        deeph_context: DeepHBenchmarkContext | None = None
        common_metrics_manifest: dict[str, Any] | None = None
        ranking_manifest: dict[str, Any] | None = None
        graph2mat_training_run: dict[str, Any] | None = None
        graph2mat_prediction_run: dict[str, Any] | None = None
        deeph_preprocess_run: dict[str, Any] | None = None
        deeph_training_run: dict[str, Any] | None = None
        deeph_inference_runs: list[dict[str, Any]] = []
        graph2mat_eval_run: dict[str, Any] | None = None
        deeph_eval_run: dict[str, Any] | None = None
        graph2mat_early_stopping: dict[str, Any] | None = None
        deeph_early_stopping: dict[str, Any] | None = None
        test_blindness_manifest: dict[str, Any] | None = None
        try:
            final_mode = is_final_benchmark_mode(payload)
            sweep_info = self.dataset_sweep_info_from_payload(payload)
            if sweep_info is not None and str(payload.get("run_mode") or "").strip() == FULL_STRICT_PIPELINE_RUN_MODE:
                self._run_full_strict_pipeline(payload, sweep_info)
                return
            if sweep_info is not None and str(payload.get("run_mode") or "").strip() == DATASET_SWEEP_RUN_MODE:
                self._run_dataset_sweep_only(payload, sweep_info)
                return
            training_sweep_plan = payload.get("_training_sweep_plan")
            if isinstance(training_sweep_plan, dict) and training_sweep_plan.get("enabled"):
                self._run_training_sweep(payload, validation, training_sweep_plan)
                return
            for phase in RUNNER_PHASES:
                if self._state.stop_requested:
                    self._set_stage("stopped")
                    self._finish(returncode=-15)
                    return
                phase_started_at = time.time()
                self._set_stage(phase)
                if phase == "generate_or_validate_joint_dataset":
                    if not validation["benchmark_ready"]:
                        if allow_repair:
                            self._finish(
                                returncode=2,
                                error=(
                                    "repair mode was explicitly requested, but repair execution is "
                                    "not implemented in Phase 7"
                                ),
                            )
                            return
                    self._logs.append(
                        "[G2M-DEEPH] Joint artifact contract validated.\n"
                    )
                    context = self._prepare_graph2mat_context(payload, validation)
                    with self._lock:
                        self._state.run_root = str(context.run_root)
                        self._state.graph2mat_config_path = str(context.config_path)
                        self._state.graph2mat_training_dir = str(context.training_dir)
                    self._logs.append(f"[G2M-DEEPH] Graph2Mat config: {context.config_path}\n")
                elif phase == "freeze_splits":
                    if context is None:
                        raise RuntimeError("Graph2Mat context was not prepared before freeze_splits.")
                    self._logs.append(
                        "[G2M-DEEPH] Frozen split conectado a Graph2Mat: "
                        f"split_hash={context.split_hash}\n"
                    )
                elif phase == "graph2mat_train":
                    if context is None:
                        raise RuntimeError("Graph2Mat context was not prepared before graph2mat_train.")
                    if context.dry_run:
                        self._logs.append("[G2M-DEEPH] graph2mat_train: dry-run, no subprocess launched.\n")
                    else:
                        command = [self._graph2mat_python(payload), str(DEFAULT_MD_TRAINING_SCRIPT)]
                        graph2mat_training_run = self._run_command(
                            command,
                            cwd=REPO_ROOT,
                            env=self._graph2mat_command_env(context, payload),
                            label="Graph2Mat training",
                            progress_provider=lambda: _lightning_training_progress(context.training_dir),
                        )
                        checkpoint_manifest_path = context.training_dir / "checkpoint_manifest.json"
                        checkpoint_manifest = _load_json(checkpoint_manifest_path)
                        early_stopping_policy = parse_early_stopping_policy(payload)
                        graph2mat_early_stopping = (
                            tensorboard_policy_metadata(context.training_dir, early_stopping_policy)
                            if early_stopping_policy is not None
                            else None
                        )
                        self._write_graph2mat_manifest(
                            context,
                            checkpoint_manifest=checkpoint_manifest,
                            extra={
                                "training_completed": True,
                                "training_run": graph2mat_training_run,
                                "early_stopping": graph2mat_early_stopping,
                            },
                        )
                elif phase == "graph2mat_predict":
                    if context is None:
                        raise RuntimeError("Graph2Mat context was not prepared before graph2mat_predict.")
                    if final_mode:
                        self._logs.append(f"[G2M-DEEPH] graph2mat_predict: {TEST_METRICS_LOCKED_MESSAGE}\n")
                    elif context.dry_run:
                        self._logs.append("[G2M-DEEPH] graph2mat_predict: dry-run, no subprocess launched.\n")
                    else:
                        command = [self._graph2mat_python(payload), str(DEFAULT_MD_PREDICTION_SCRIPT)]
                        graph2mat_prediction_run = self._run_command(
                            command,
                            cwd=REPO_ROOT,
                            env=self._graph2mat_command_env(context, payload),
                            label="Graph2Mat prediction",
                        )
                        checkpoint_manifest = _load_json(context.training_dir / "checkpoint_manifest.json")
                        prediction_outputs = self._validate_graph2mat_prediction_outputs(context)
                        self._write_graph2mat_manifest(
                            context,
                            checkpoint_manifest=checkpoint_manifest,
                            prediction_outputs=prediction_outputs,
                            extra={
                                "prediction_completed": True,
                                "prediction_run": graph2mat_prediction_run,
                                "early_stopping": graph2mat_early_stopping,
                            },
                        )
                elif phase == "deeph_preprocess":
                    if context is None:
                        raise RuntimeError("Graph2Mat context was not prepared before deeph_preprocess.")
                    deeph_context = self._prepare_deeph_context(payload, context)
                    with self._lock:
                        self._state.deeph_manifest_path = str(deeph_context.manifest_path)
                        self._state.deeph_processed_dir = str(deeph_context.processed_dir)
                        self._state.deeph_save_dir = str(deeph_context.save_dir)
                    self._logs.append(f"[G2M-DEEPH] DeepH preprocess config: {deeph_context.preprocess_config}\n")
                    if deeph_context.dry_run:
                        self._logs.append("[G2M-DEEPH] deeph_preprocess: dry-run, no subprocess launched.\n")
                    else:
                        command = [
                            self._deeph_command(payload, "deeph-preprocess"),
                            "--config",
                            str(deeph_context.preprocess_config),
                        ]
                        deeph_preprocess_run = self._run_command(
                            command,
                            cwd=deeph_context.root,
                            env=self._deeph_command_env(payload),
                            label="DeepH preprocess",
                        )
                        split_audit = self._audit_deeph_split(deeph_context, context)
                        self._write_deeph_manifest(
                            deeph_context,
                            preprocess_run=deeph_preprocess_run,
                            split_audit=split_audit,
                        )
                elif phase == "deeph_train":
                    if deeph_context is None:
                        raise RuntimeError("DeepH context was not prepared before deeph_train.")
                    if deeph_context.dry_run:
                        self._logs.append("[G2M-DEEPH] deeph_train: dry-run, no subprocess launched.\n")
                    else:
                        command = [
                            self._deeph_command(payload, "deeph-train"),
                            "--config",
                            str(deeph_context.train_config),
                        ]
                        early_stopping_policy = parse_early_stopping_policy(payload)
                        deeph_early_observer = DeepHEarlyStoppingObserver(early_stopping_policy) if early_stopping_policy else None
                        deeph_training_run = self._run_command(
                            command,
                            cwd=deeph_context.root,
                            env=self._deeph_command_env(payload),
                            label="DeepH train",
                            line_observer=deeph_early_observer,
                        )
                        deeph_early_stopping = deeph_early_observer.metadata() if deeph_early_observer is not None else None
                        training_outputs = self._validate_deeph_training_outputs(deeph_context)
                        self._write_deeph_manifest(
                            deeph_context,
                            train_run=deeph_training_run,
                            training_outputs=training_outputs,
                            extra={"early_stopping": deeph_early_stopping},
                        )
                elif phase == "deeph_predict":
                    if deeph_context is None:
                        raise RuntimeError("DeepH context was not prepared before deeph_predict.")
                    if final_mode:
                        self._logs.append(f"[G2M-DEEPH] deeph_predict: {TEST_METRICS_LOCKED_MESSAGE}\n")
                    elif deeph_context.dry_run:
                        self._logs.append("[G2M-DEEPH] deeph_predict: dry-run, no subprocess launched.\n")
                    else:
                        staged_inputs = self._stage_deeph_inference_inputs(deeph_context)
                        deeph_inference_runs = []
                        for inference_config in deeph_context.inference_configs:
                            command = [
                                self._deeph_command(payload, "deeph-inference"),
                                "--config",
                                str(inference_config),
                            ]
                            deeph_inference_runs.append(
                                self._run_command(
                                    command,
                                    cwd=deeph_context.root,
                                    env=self._deeph_command_env(payload),
                                    label=f"DeepH inference {inference_config.stem}",
                                )
                            )
                        prediction_outputs = self._validate_deeph_prediction_outputs(deeph_context)
                        training_outputs = (
                            self._validate_deeph_training_outputs(deeph_context)
                            if (deeph_context.save_dir / "best_state_dict.pkl").exists()
                            else None
                        )
                        self._write_deeph_manifest(
                            deeph_context,
                            inference_runs=deeph_inference_runs,
                            training_outputs=training_outputs,
                            prediction_outputs=prediction_outputs,
                            extra={"inference_inputs": staged_inputs, "early_stopping": deeph_early_stopping},
                        )
                elif phase == "common_metrics":
                    if context is None:
                        raise RuntimeError("Graph2Mat context was not prepared before common_metrics.")
                    if deeph_context is None:
                        raise RuntimeError("DeepH context was not prepared before common_metrics.")
                    metric_fail_policy = _metric_fail_policy(payload)
                    if final_mode:
                        self._logs.append(f"[G2M-DEEPH] common_metrics: {TEST_METRICS_LOCKED_MESSAGE}\n")
                    elif context.dry_run or deeph_context.dry_run:
                        self._logs.append("[G2M-DEEPH] common_metrics: dry-run, no evaluator subprocess launched.\n")
                    else:
                        common_root = context.run_root / "common_metrics"
                        frozen_split = _load_json(context.frozen_split_manifest_path)
                        staged_graph2mat = stage_graph2mat_metric_result(
                            frozen_split_manifest=frozen_split,
                            prediction_structs_dir=context.prediction_structs_dir,
                            output_dir=common_root / "graph2mat_eval",
                            dataset_root=context.dataset_root,
                        )
                        graph2mat_eval_run = self._run_command(
                            [
                                self._graph2mat_python(payload),
                                str(DEFAULT_HAMILTONIAN_METRICS_SCRIPT),
                                str(staged_graph2mat.result_dir),
                                "--workers",
                                "1",
                                "--enable-kpoint-metrics",
                                "--overwrite",
                            ],
                            cwd=REPO_ROOT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"},
                            label="Graph2Mat common metrics",
                            allowed_returncodes=_metric_allowed_returncodes(metric_fail_policy),
                        )
                        staged_deeph = stage_deeph_metric_inputs(
                            raw_mirror=deeph_context.raw_mirror,
                            processed_dir=deeph_context.processed_dir,
                            inference_dir=deeph_context.inference_dir,
                            output_dir=common_root / "deeph_inputs",
                        )
                        deeph_metric_command = _deeph_metric_command_args(
                            python_executable=self._graph2mat_python(payload),
                            graph2mat_result_dir=staged_graph2mat.result_dir,
                            processed_dir=staged_deeph.processed_dir,
                            predictions_dir=staged_deeph.predictions_dir,
                            output_dir=common_root / "deeph_eval",
                            metric_fail_policy=metric_fail_policy,
                        )
                        deeph_eval_run = self._run_command(
                            deeph_metric_command,
                            cwd=REPO_ROOT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"},
                            label="DeepH common metrics",
                            allowed_returncodes=_metric_allowed_returncodes(metric_fail_policy),
                        )
                        common_metrics_manifest = aggregate_common_metrics(
                            graph2mat_metrics_root=staged_graph2mat.result_dir / "metrics",
                            deeph_metrics_root=common_root / "deeph_eval" / "metrics",
                            output_dir=common_root / "summary",
                            frozen_split_manifest_path=context.frozen_split_manifest_path,
                            dataset_manifest_path=context.benchmark_dataset_manifest_path,
                        )
                        common_metrics_manifest["runs"] = {
                            "graph2mat_eval": graph2mat_eval_run,
                            "deeph_eval": deeph_eval_run,
                        }
                        graph2mat_telemetry = self._write_run_cost_telemetry(
                            model="graph2mat",
                            run_root=context.run_root,
                            frozen_split_manifest_path=context.frozen_split_manifest_path,
                            train_run=graph2mat_training_run,
                            predict_run=graph2mat_prediction_run,
                            metrics_run=graph2mat_eval_run,
                            training_dir=context.training_dir,
                            payload=payload,
                        )
                        deeph_telemetry = self._write_run_cost_telemetry(
                            model="deeph",
                            run_root=context.run_root,
                            frozen_split_manifest_path=context.frozen_split_manifest_path,
                            preprocess_run=deeph_preprocess_run,
                            train_run=deeph_training_run,
                            inference_runs=deeph_inference_runs,
                            metrics_run=deeph_eval_run,
                            deeph_save_dir=deeph_context.save_dir,
                            payload=payload,
                        )
                        common_metrics_manifest["telemetry"] = {
                            "graph2mat": graph2mat_telemetry,
                            "deeph": deeph_telemetry,
                        }
                        checkpoint_manifest = _load_json(context.training_dir / "checkpoint_manifest.json")
                        prediction_outputs = self._validate_graph2mat_prediction_outputs(context)
                        self._write_graph2mat_manifest(
                            context,
                            checkpoint_manifest=checkpoint_manifest,
                            prediction_outputs=prediction_outputs,
                            extra={
                                "prediction_completed": True,
                                "prediction_run": graph2mat_prediction_run,
                                "telemetry": graph2mat_telemetry,
                                "early_stopping": graph2mat_early_stopping,
                            },
                        )
                        training_outputs = (
                            self._validate_deeph_training_outputs(deeph_context)
                            if (deeph_context.save_dir / "best_state_dict.pkl").exists()
                            else None
                        )
                        prediction_outputs_deeph = self._validate_deeph_prediction_outputs(deeph_context)
                        self._write_deeph_manifest(
                            deeph_context,
                            preprocess_run=deeph_preprocess_run,
                            train_run=deeph_training_run,
                            inference_runs=deeph_inference_runs,
                            training_outputs=training_outputs,
                            prediction_outputs=prediction_outputs_deeph,
                            split_audit=_load_json(deeph_context.split_audit_path) if deeph_context.split_audit_path.exists() else None,
                            extra={"telemetry": deeph_telemetry, "early_stopping": deeph_early_stopping},
                        )
                        _force_diagnostic_metric_manifest(
                            common_metrics_manifest,
                            metric_fail_policy=metric_fail_policy,
                        )
                        _write_json(common_root / "summary" / "common_summary.json", common_metrics_manifest)
                        _write_json(common_root / "summary" / "benchmark_manifest.json", common_metrics_manifest)
                        self._logs.append(
                            "[G2M-DEEPH] Common metrics status: "
                            f"{common_metrics_manifest['status']} "
                            f"({common_root / 'summary' / 'common_summary.json'})\n"
                        )
                elif phase == "ranking":
                    if context is None:
                        raise RuntimeError("Graph2Mat context was not prepared before ranking.")
                    if final_mode:
                        test_blindness_manifest = build_search_stage_manifest(
                            run_root=context.run_root,
                            summary={
                                "status": "completed",
                                "runs": [],
                                "failed_runs": [],
                            },
                            payload=payload,
                        )
                        self._logs.append(
                            "[G2M-DEEPH] ranking: skipped in final/publicable search; "
                            "top-k selection must use validation metrics only.\n"
                        )
                    elif common_metrics_manifest is None:
                        self._logs.append("[G2M-DEEPH] ranking: no common metrics summary in dry-run.\n")
                    else:
                        ranking_manifest = self._run_ranking(
                            context.run_root,
                            validation=validation,
                            common_metrics_manifest_path=context.run_root / "common_metrics" / "summary" / "common_summary.json",
                            dataset_root=context.dataset_root,
                            frozen_split_manifest_path=context.frozen_split_manifest_path,
                            dataset_manifest_path=context.benchmark_dataset_manifest_path,
                        )
                        self._logs.append(
                            "[G2M-DEEPH] Ranking status: "
                            f"{ranking_manifest['recommendation']['status']} "
                            f"({context.run_root / 'summary' / 'ranking' / 'ranking_summary.json'})\n"
                        )
                elif phase == "plots_and_summary":
                    if common_metrics_manifest is None and ranking_manifest is None:
                        self._logs.append("[G2M-DEEPH] plots_and_summary: no metrics/ranking summary in dry-run.\n")
                    else:
                        self._logs.append(
                            "[G2M-DEEPH] plots_and_summary: summary/ranking JSON/CSV written; plot rendering remains external.\n"
                        )
                with self._lock:
                    self._phase_timings.append(
                        {
                            "phase": phase,
                            "started_at": phase_started_at,
                            "finished_at": time.time(),
                            "elapsed_seconds": time.time() - phase_started_at,
                        }
                    )
                time.sleep(float(payload.get("phase_delay_seconds", 0) or 0))
            with self._lock:
                self._last_results = {
                    "dry_run": bool(context.dry_run) if context is not None else None,
                    "contract_name": CONTRACT_NAME,
                    "dataset_validation": validation,
                    "graph2mat": context.to_dict() if context is not None else None,
                    "deeph": deeph_context.to_dict() if deeph_context is not None else None,
                    "common_metrics": common_metrics_manifest,
                    "ranking": ranking_manifest,
                    "test_blindness": test_blindness_manifest,
                    "phase_timings": list(self._phase_timings),
                    "message": (
                        "Final/publicable search completed test-blind; test metrics are locked "
                        "until validation-based top-k selection and final_test."
                        if final_mode
                        else "Graph2Mat and DeepH command chain completed; common metrics were "
                        "computed when prediction outputs were available."
                    ),
                }
            self._finish(returncode=0)
        except Exception as exc:
            self._finish(returncode=1, error=str(exc))

    def write_incremental_manifest(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "graph2mat_deeph_benchmark_runner_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": self.status(),
            "results": self._last_results,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
