#!/usr/bin/env python3
"""Backend runner for the Graph2Mat vs DeepH benchmark workflow."""

from __future__ import annotations

import argparse
import json
import os
import csv
import hashlib
import re
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
from g2m_deeph_derivative_gate_check import build_derivative_gate_report  # noqa: E402
from g2m_deeph_rank_runs import rank_graph2mat_deeph_runs  # noqa: E402
from g2m_deeph_live_metrics import completed_metric_record, dedupe_metric_rows, live_metric_scaling_rows  # noqa: E402
from g2m_deeph_telemetry import (  # noqa: E402
    GpuTelemetryMonitor,
    ProcResourceMonitor,
    classify_failure,
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
    ROBUST_VALIDATION_STAGE,
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
from plot_hamiltonian_derivative_metrics import write_derivative_plot_outputs  # noqa: E402


DEFAULT_LOG_RESPONSE_LIMIT = 2000
MAX_LOG_RESPONSE_LIMIT = 20000
LOG_HEARTBEAT_SECONDS = 30.0
DETACHED_SEARCH_ROOT_ENV = "G2M_DEEPH_DETACHED_SEARCH_ROOT"
EXTERNAL_FINAL_LOG_ROOT = REPO_ROOT / "Comparison" / "results" / "paper_ready_final_70_logs"
EXTERNAL_FINAL_RUN_GLOB = "paper_ready_final70_*"
DEFAULT_DATASETS_ROOT = REPO_ROOT / "Comparison" / "datasets"
DEFAULT_DATASET_ROOT = DEFAULT_DATASETS_ROOT / "graphene_w90_joint"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison" / "results" / "graphene_w90_g2m_deeph_benchmark"
DEFAULT_MD_PIPELINE_CONFIG = REPO_ROOT / "MD" / "pipeline_config.yaml"
DEFAULT_MD_TRAINING_SCRIPT = REPO_ROOT / "MD" / "scripts" / "run_md_training.py"
DEFAULT_MD_PREDICTION_SCRIPT = REPO_ROOT / "MD" / "scripts" / "run_md_prediction.py"
DEFAULT_HAMILTONIAN_METRICS_SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "evaluate_hamiltonian_metrics.py"
DEFAULT_DERIVATIVE_METRICS_SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "evaluate_hamiltonian_derivative_metrics.py"
DEFAULT_DERIVATIVE_STENCIL_SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "build_hamiltonian_derivative_stencils.py"
DEFAULT_DERIVATIVE_GEOMETRY_VALIDATION_SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "validate_hamiltonian_derivative_geometry.py"
DEFAULT_DERIVATIVE_SIESTA_REFERENCE_SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "run_hamiltonian_derivative_siesta_references.py"
DEFAULT_DERIVATIVE_PREDICTION_SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "run_hamiltonian_derivative_predictions.py"
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
MODULAR_WORKFLOW_STAGE_NAMES = (
    "generate_or_validate_dataset",
    "freeze_splits",
    "train_graph2mat",
    "predict_graph2mat",
    "train_deeph",
    "predict_deeph",
    "hamiltonian_metrics",
    "build_derivative_stencils",
    "validate_derivative_stencils",
    "run_derivative_siesta_reference",
    "predict_derivative_graph2mat",
    "predict_derivative_deeph",
    "derivative_metrics_graph2mat",
    "derivative_metrics_deeph",
    "derivative_gate_check",
    "derivative_plots",
    "derivative_model_comparison",
)
MODULAR_WORKFLOW_DEFAULT_STAGES = {
    "generate_or_validate_dataset": True,
    "freeze_splits": True,
    "train_graph2mat": True,
    "predict_graph2mat": True,
    "train_deeph": True,
    "predict_deeph": True,
    "hamiltonian_metrics": True,
    "build_derivative_stencils": False,
    "validate_derivative_stencils": False,
    "run_derivative_siesta_reference": False,
    "predict_derivative_graph2mat": False,
    "predict_derivative_deeph": False,
    "derivative_metrics_graph2mat": False,
    "derivative_metrics_deeph": False,
    "derivative_gate_check": False,
    "derivative_plots": False,
    "derivative_model_comparison": True,
}
MODULAR_WORKFLOW_MODE_ALIASES = {
    "h_only": "hamiltonian_only",
    "h-only": "hamiltonian_only",
    "hamiltonian": "hamiltonian_only",
    "hamiltonian-only": "hamiltonian_only",
    "hamiltonian_only": "hamiltonian_only",
    "derivative-stencils-only": "derivative_stencils_only",
    "derivative_stencils_only": "derivative_stencils_only",
    "derivative-reference-only": "derivative_reference_only",
    "derivative_reference_only": "derivative_reference_only",
    "derivative-predictions-only": "derivative_predictions_only",
    "derivative_predictions_only": "derivative_predictions_only",
    "derivative-metrics-only": "derivative_metrics_only",
    "derivative_metrics_only": "derivative_metrics_only",
    "h_then_derivative_postprocess": "h_then_derivative_postprocess",
    "h-then-derivative-postprocess": "h_then_derivative_postprocess",
    "h_then_derivative_full": "h_then_derivative_full",
    "h-then-derivative-full": "h_then_derivative_full",
    "full-end-to-end": "full_end_to_end",
    "full_end_to_end": "full_end_to_end",
}
MODULAR_WORKFLOW_MODE_OVERRIDES = {
    "hamiltonian_only": {},
    "derivative_stencils_only": {
        "generate_or_validate_dataset": False,
        "freeze_splits": False,
        "train_graph2mat": False,
        "predict_graph2mat": False,
        "train_deeph": False,
        "predict_deeph": False,
        "hamiltonian_metrics": False,
        "build_derivative_stencils": True,
        "validate_derivative_stencils": True,
    },
    "derivative_reference_only": {
        "generate_or_validate_dataset": False,
        "freeze_splits": False,
        "train_graph2mat": False,
        "predict_graph2mat": False,
        "train_deeph": False,
        "predict_deeph": False,
        "hamiltonian_metrics": False,
        "validate_derivative_stencils": True,
        "run_derivative_siesta_reference": True,
    },
    "derivative_predictions_only": {
        "generate_or_validate_dataset": False,
        "freeze_splits": False,
        "train_graph2mat": False,
        "predict_graph2mat": False,
        "train_deeph": False,
        "predict_deeph": False,
        "hamiltonian_metrics": False,
        "validate_derivative_stencils": True,
        "predict_derivative_graph2mat": True,
        "predict_derivative_deeph": True,
    },
    "derivative_metrics_only": {
        "generate_or_validate_dataset": False,
        "freeze_splits": False,
        "train_graph2mat": False,
        "predict_graph2mat": False,
        "train_deeph": False,
        "predict_deeph": False,
        "hamiltonian_metrics": False,
        "derivative_metrics_graph2mat": True,
        "derivative_metrics_deeph": True,
        "derivative_gate_check": True,
        "derivative_plots": True,
    },
    "h_then_derivative_postprocess": {
        "derivative_metrics_graph2mat": True,
        "derivative_metrics_deeph": True,
        "derivative_gate_check": True,
        "derivative_plots": True,
    },
    "h_then_derivative_full": {
        "build_derivative_stencils": True,
        "validate_derivative_stencils": True,
        "run_derivative_siesta_reference": True,
        "predict_derivative_graph2mat": True,
        "predict_derivative_deeph": True,
        "derivative_metrics_graph2mat": True,
        "derivative_metrics_deeph": True,
        "derivative_gate_check": True,
        "derivative_plots": True,
    },
    "full_end_to_end": {
        "build_derivative_stencils": True,
        "validate_derivative_stencils": True,
        "run_derivative_siesta_reference": True,
        "predict_derivative_graph2mat": True,
        "predict_derivative_deeph": True,
        "derivative_metrics_graph2mat": True,
        "derivative_metrics_deeph": True,
        "derivative_gate_check": True,
        "derivative_plots": True,
    },
}
DERIVATIVE_STAGE_NAMES = {
    "build_derivative_stencils",
    "validate_derivative_stencils",
    "run_derivative_siesta_reference",
    "predict_derivative_graph2mat",
    "predict_derivative_deeph",
    "derivative_metrics_graph2mat",
    "derivative_metrics_deeph",
    "derivative_gate_check",
    "derivative_plots",
}
HAMILTONIAN_STAGE_NAMES = tuple(
    stage
    for stage in MODULAR_WORKFLOW_STAGE_NAMES
    if stage not in DERIVATIVE_STAGE_NAMES and stage != "derivative_model_comparison"
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


class CommandRunError(RuntimeError):
    """Subprocess failure that preserves the structured command record."""

    def __init__(self, message: str, run_record: dict[str, Any]) -> None:
        super().__init__(message)
        self.run_record = run_record


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
    prediction_split: str = "test"
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
            "prediction_split": self.prediction_split,
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
    inference_split: str = "test"
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
            "inference_split": self.inference_split,
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
        "max_parallel_derivative_workflows",
        "max_parallel_derivative_reference_jobs",
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


def _derivative_reference_workers(payload: dict[str, Any], config: dict[str, Any]) -> int:
    configured = config.get("reference_workers")
    if configured not in (None, "", "null"):
        return _optional_positive_int(
            configured,
            field_name="modular_workflow.derivative.reference_workers",
        ) or 1
    settings = _performance_settings_from_payload(payload)
    fallback = settings.get("max_parallel_derivative_reference_jobs")
    if fallback not in (None, "", "null"):
        return _optional_positive_int(
            fallback,
            field_name="performance.max_parallel_derivative_reference_jobs",
        ) or 1
    return 1


def _derivative_workflow_parallelism(payload: dict[str, Any]) -> int:
    settings = _performance_settings_from_payload(payload)
    raw = settings.get("max_parallel_derivative_workflows") or 1
    try:
        workers = int(raw)
    except (TypeError, ValueError):
        workers = 1
    return max(1, workers)


def _mixed_model_training_batches(payload: dict[str, Any]) -> bool:
    return _model_batch_schedule(payload) == "mixed"


def _model_batch_schedule(payload: dict[str, Any]) -> str:
    settings = _performance_settings_from_payload(payload)
    training_sweep = payload.get("training_sweep") if isinstance(payload.get("training_sweep"), dict) else {}
    raw = (
        settings.get("model_batch_schedule")
        or training_sweep.get("model_batch_schedule")
        or payload.get("model_batch_schedule")
    )
    if raw not in (None, "", "null"):
        schedule = str(raw).strip().lower()
        if schedule not in {"contiguous", "mixed", "alternating"}:
            raise RuntimeError("model_batch_schedule must be contiguous, mixed, or alternating.")
        return schedule
    mixed_raw = (
        settings.get("mixed_model_training_batches")
        if settings.get("mixed_model_training_batches") is not None
        else training_sweep.get("mixed_model_training_batches")
    )
    if _parse_bool(mixed_raw, False):
        return "mixed"
    alternating_raw = (
        settings.get("alternating_model_training_batches")
        if settings.get("alternating_model_training_batches") is not None
        else training_sweep.get("alternating_model_training_batches")
    )
    if _parse_bool(alternating_raw, False):
        return "alternating"
    return "contiguous"


def _alternating_model_start(payload: dict[str, Any]) -> str:
    settings = _performance_settings_from_payload(payload)
    training_sweep = payload.get("training_sweep") if isinstance(payload.get("training_sweep"), dict) else {}
    raw = (
        settings.get("model_batch_start")
        or settings.get("alternating_model_start")
        or training_sweep.get("model_batch_start")
        or training_sweep.get("alternating_model_start")
        or payload.get("model_batch_start")
        or payload.get("alternating_model_start")
        or "graph2mat"
    )
    start = str(raw).strip().lower()
    if start not in {"graph2mat", "deeph"}:
        raise RuntimeError("model_batch_start must be graph2mat or deeph.")
    return start


def _alternating_model_batch_order(
    records: list[dict[str, Any]],
    *,
    graph2mat_parallelism: int,
    deeph_parallelism: int,
    start_model: str = "graph2mat",
) -> list[dict[str, Any]]:
    """Order runs as same-model batches, alternating between G2M and DeepH."""

    queues = {
        "graph2mat": [record for record in records if record.get("model") == "graph2mat"],
        "deeph": [record for record in records if record.get("model") == "deeph"],
    }
    other = [record for record in records if record.get("model") not in queues]
    limits = {
        "graph2mat": max(1, int(graph2mat_parallelism)),
        "deeph": max(1, int(deeph_parallelism)),
    }
    model = start_model if start_model in queues else "graph2mat"
    ordered: list[dict[str, Any]] = []
    while queues["graph2mat"] or queues["deeph"]:
        if not queues[model]:
            model = "deeph" if model == "graph2mat" else "graph2mat"
            if not queues[model]:
                break
        count = min(limits[model], len(queues[model]))
        ordered.extend(queues[model][:count])
        del queues[model][:count]
        model = "deeph" if model == "graph2mat" else "graph2mat"
    ordered.extend(other)
    return ordered


def _deeph_device_settings(payload: dict[str, Any], options: dict[str, Any]) -> tuple[bool, str]:
    settings = _performance_settings_from_payload(payload)
    accelerator = str(settings.get("compute_accelerator") or "").strip().lower()
    default_disable_cuda = accelerator not in {"gpu", "cuda"}
    disable_cuda = _parse_bool(options.get("disable_cuda"), default_disable_cuda)
    device = str(options.get("device") or ("cpu" if disable_cuda else "cuda:0"))
    return disable_cuda, device


def _budget_accounting_fail_closed(payload: dict[str, Any], *, final_mode: bool) -> bool:
    """Return true when incomplete cost telemetry must fail the sweep run."""

    if final_mode:
        return True
    if payload.get("budget_accounting_fail_closed") is not None:
        return _parse_bool(payload.get("budget_accounting_fail_closed"), False)
    training_sweep = payload.get("training_sweep") if isinstance(payload.get("training_sweep"), dict) else {}
    budget_policy = training_sweep.get("budget_policy") if isinstance(training_sweep.get("budget_policy"), dict) else {}
    if budget_policy.get("fail_closed_on_missing_telemetry") is not None:
        return _parse_bool(budget_policy.get("fail_closed_on_missing_telemetry"), False)
    return False


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


def _process_table_lines_for_text(*needles: str) -> list[str]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return []
    lines = []
    for line in result.stdout.splitlines():
        if all(needle in line for needle in needles):
            lines.append(line)
    return lines


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


def _search_validation_metrics_requested(payload: dict[str, Any]) -> bool:
    stage = protocol_stage_from_payload(payload)
    if stage not in {SEARCH_STAGE, ROBUST_VALIDATION_STAGE}:
        return False
    raw = payload.get("search_validation_metrics")
    if raw is not None:
        return _parse_bool(raw, False)
    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    search_eval = protocol.get("search_evaluation") if isinstance(protocol.get("search_evaluation"), dict) else {}
    if "run_validation_metrics" in search_eval:
        return _parse_bool(search_eval.get("run_validation_metrics"), False)
    policy = selection_policy_from_payload(payload)
    return str(policy.get("metric") or "").strip() == "val_spectral_composite"


def _metric_evaluation_split(payload: dict[str, Any]) -> str:
    if _search_validation_metrics_requested(payload):
        split = "validation"
    else:
        split = "test"
    raw = payload.get("metric_evaluation_split") or payload.get("prediction_split")
    if raw not in (None, ""):
        split = str(raw).strip()
    if split not in {"train", "validation", "test"}:
        raise RuntimeError("metric_evaluation_split must be train, validation, or test.")
    stage = protocol_stage_from_payload(payload)
    if stage in {SEARCH_STAGE, ROBUST_VALIDATION_STAGE} and _search_validation_metrics_requested(payload) and split == "test":
        raise RuntimeError("Validation metric evaluation cannot use the locked test split.")
    return split


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _csv_metric_mean(path: Path, metric: str, *, row_type: str | None = None) -> float | None:
    if not path.exists():
        return None
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row_type is not None and str(row.get("row_type") or "") != row_type:
                continue
            number = _finite_float(row.get(metric))
            if number is not None:
                values.append(number)
    if not values:
        return None
    return float(sum(values) / len(values))


def _summary_metric_mean(manifest: dict[str, Any], group: str, metric: str) -> float | None:
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    group_summary = summary.get(group) if isinstance(summary.get(group), dict) else {}
    item = group_summary.get(metric) if isinstance(group_summary.get(metric), dict) else {}
    return _finite_float(item.get("mean"))


def _first_not_none(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _extract_validation_metrics(metrics_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    manifest_path = metrics_dir / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}

    def set_if_available(key: str, value: float | None) -> None:
        if value is not None:
            metrics[key] = value

    for metric in ("low_energy_rmse_eV", "fermi_window_rmse_eV", "frontier_window_rmse_eV"):
        set_if_available(
            metric,
            _first_not_none(
                _summary_metric_mean(manifest, "kpoint_spectral", metric),
                _summary_metric_mean(manifest, "spectral", metric),
                _csv_metric_mean(metrics_dir / "kpoint_spectral_metrics.csv", metric),
                _csv_metric_mean(metrics_dir / "spectral_metrics.csv", metric),
            ),
        )
    global_rmse = _first_not_none(
        _summary_metric_mean(manifest, "kpoint_spectral", "global_rmse_eV"),
        _summary_metric_mean(manifest, "spectral", "global_rmse_eV"),
        _csv_metric_mean(metrics_dir / "kpoint_spectral_metrics.csv", "global_rmse_eV"),
        _csv_metric_mean(metrics_dir / "spectral_metrics.csv", "global_rmse_eV"),
    )
    set_if_available("global_band_rmse", global_rmse)
    set_if_available("global_rmse_eV", global_rmse)
    dos_wasserstein = _first_not_none(
        _summary_metric_mean(manifest, "kpoint_dos", "dos_wasserstein_eV"),
        _summary_metric_mean(manifest, "dos", "dos_wasserstein_eV"),
        _csv_metric_mean(metrics_dir / "kpoint_dos_metrics.csv", "dos_wasserstein_eV"),
        _csv_metric_mean(metrics_dir / "dos_metrics.csv", "dos_wasserstein_eV"),
    )
    set_if_available("dos_wasserstein", dos_wasserstein)
    set_if_available("dos_wasserstein_eV", dos_wasserstein)
    dos_fermi_mae = _first_not_none(
        _summary_metric_mean(manifest, "kpoint_dos", "dos_mae_500_fermi_window"),
        _summary_metric_mean(manifest, "dos", "dos_mae_500_fermi_window"),
        _csv_metric_mean(metrics_dir / "kpoint_dos_metrics.csv", "dos_mae_500_fermi_window"),
        _csv_metric_mean(metrics_dir / "dos_metrics.csv", "dos_mae_500_fermi_window"),
    )
    set_if_available("dos_mae_near_fermi", dos_fermi_mae)
    set_if_available("dos_mae_500_fermi_window", dos_fermi_mae)
    h_mae = _first_not_none(
        _summary_metric_mean(manifest, "kpoint_matrix", "h_mae_eV"),
        _csv_metric_mean(metrics_dir / "kpoint_matrix_metrics.csv", "h_mae_eV", row_type="weighted_sample"),
        _summary_metric_mean(manifest, "sparse", "h_matrix_mae_eV"),
        _summary_metric_mean(manifest, "sparse", "mae_ref_eV"),
        _csv_metric_mean(metrics_dir / "sparse_metrics.csv", "h_matrix_mae_eV"),
        _csv_metric_mean(metrics_dir / "sparse_metrics.csv", "mae_ref_eV"),
    )
    set_if_available("h_mae_eV", h_mae)
    return metrics


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
    split: str = "test",
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
        split,
    ]
    if metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY:
        command.append("--no-fail-closed")
    return command


def _derivative_metrics_settings(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("derivative_metrics") if isinstance(payload.get("derivative_metrics"), dict) else {}
    enabled = _parse_bool(raw.get("enabled"), False)
    split = str(raw.get("split") or (_metric_evaluation_split(payload) if enabled else "test"))
    return {
        "enabled": enabled,
        "finite_difference_method": str(raw.get("finite_difference_method") or raw.get("method") or "central"),
        "split": split,
        "require_central": _parse_bool(raw.get("require_central"), True),
        "diagnostic_only": _parse_bool(raw.get("diagnostic_only"), True),
        "support_threshold": float(raw.get("support_threshold", 1e-12) or 1e-12),
        "max_stencils": _optional_int_value(raw.get("max_stencils")),
    }


def _normalized_derivative_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("derivative_metrics") if isinstance(payload.get("derivative_metrics"), dict) else {}
    normalized = dict(raw)
    normalized["enabled"] = _parse_bool(raw.get("enabled"), False)
    return normalized


def _normalize_modular_workflow_mode(payload: dict[str, Any]) -> str:
    raw = str(payload.get("workflow_mode") or payload.get("benchmark_workflow") or "").strip().lower()
    if not raw:
        return ""
    mode = MODULAR_WORKFLOW_MODE_ALIASES.get(raw)
    if mode is None:
        allowed = ", ".join(sorted(set(MODULAR_WORKFLOW_MODE_ALIASES.values())))
        raise RuntimeError(f"Unsupported workflow_mode {raw!r}. Use one of: {allowed}.")
    return mode


def _normalize_modular_workflow_stages(payload: dict[str, Any]) -> dict[str, bool]:
    stages = dict(MODULAR_WORKFLOW_DEFAULT_STAGES)
    mode = _normalize_modular_workflow_mode(payload)
    if mode:
        stages.update(MODULAR_WORKFLOW_MODE_OVERRIDES[mode])
    raw = payload.get("stages")
    if raw is None:
        raw = payload.get("workflow_stages")
    if raw is not None:
        if not isinstance(raw, dict):
            raise RuntimeError("stages must be an object mapping stage names to booleans.")
        for key, value in raw.items():
            stage = str(key).strip()
            if stage not in stages:
                allowed = ", ".join(MODULAR_WORKFLOW_STAGE_NAMES)
                raise RuntimeError(f"Unsupported workflow stage {stage!r}. Use one of: {allowed}.")
            stages[stage] = _parse_bool(value, False)
    return stages


def _normalize_derivative_workflow_config(payload: dict[str, Any]) -> dict[str, Any]:
    raw = _raw_derivative_workflow_config(payload)
    config = dict(raw)
    config["enabled"] = _parse_bool(raw.get("enabled"), False)
    config["method"] = str(raw.get("method") or raw.get("finite_difference_method") or "central").strip().lower()
    config["base_split"] = str(raw.get("base_split") or raw.get("split") or "test").strip().lower()
    config["overwrite"] = _parse_bool(raw.get("overwrite"), False)
    config["skip_if_exists"] = _parse_bool(raw.get("skip_if_exists"), True)
    config["delta_ang_values"] = _normalize_derivative_delta_values(raw.get("delta_ang"))
    config["delta_ang"] = config["delta_ang_values"][0] if config["delta_ang_values"] else None
    if raw.get("max_base_snapshots") not in (None, ""):
        config["max_base_snapshots"] = _optional_int_value(raw.get("max_base_snapshots"))
    else:
        config["max_base_snapshots"] = None
    has_min_base_snapshots = raw.get("min_base_snapshots") not in (None, "", "null")
    has_base_fraction = raw.get("base_fraction") not in (None, "", "null")
    raw_policy = raw.get("base_selection_policy")
    if raw_policy not in (None, "", "null"):
        config["base_selection_policy"] = str(raw_policy).strip().lower()
    elif has_min_base_snapshots or has_base_fraction:
        config["base_selection_policy"] = "adaptive_min_fraction"
    elif config["max_base_snapshots"] is not None:
        config["base_selection_policy"] = "first"
    else:
        config["base_selection_policy"] = "all"
    config["min_base_snapshots"] = _optional_int_value(raw.get("min_base_snapshots")) if has_min_base_snapshots else None
    if has_base_fraction:
        try:
            config["base_fraction"] = float(raw.get("base_fraction"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("derivative.base_fraction must be numeric.") from exc
    else:
        config["base_fraction"] = None
    config["base_selection_seed"] = (
        _optional_int_value(raw.get("base_selection_seed"))
        if raw.get("base_selection_seed") not in (None, "", "null")
        else None
    )
    config["atoms"] = _normalize_optional_string_list(raw.get("atoms"), field="derivative.atoms")
    config["axes"] = _normalize_optional_string_list(raw.get("axes"), field="derivative.axes")
    return config


def _raw_derivative_workflow_config(payload: dict[str, Any]) -> dict[str, Any]:
    derivative_present = "derivative" in payload and payload.get("derivative") is not None
    derivatives_present = "derivatives" in payload and payload.get("derivatives") is not None
    derivative = payload.get("derivative")
    derivatives = payload.get("derivatives")
    if derivative_present and not isinstance(derivative, dict):
        raise RuntimeError("derivative must be an object.")
    if derivatives_present and not isinstance(derivatives, dict):
        raise RuntimeError("derivatives must be an object.")
    if derivative_present and derivatives_present:
        derivative_payload = dict(derivative)
        derivatives_payload = dict(derivatives)
        if _canonical_derivative_alias_payload(derivative_payload) != _canonical_derivative_alias_payload(derivatives_payload):
            raise RuntimeError("derivative and derivatives configs conflict; use one canonical derivative object.")
        return derivative_payload
    if derivative_present:
        return dict(derivative)
    if derivatives_present:
        return dict(derivatives)
    return {}


def _canonical_derivative_alias_payload(config: dict[str, Any]) -> Any:
    canonical = dict(config)
    if "delta_ang" in canonical:
        canonical["delta_ang"] = _normalize_derivative_delta_values(canonical.get("delta_ang"))
    return _json_safe_payload(canonical)


def _normalize_derivative_delta_values(value: Any) -> list[float]:
    if value in (None, ""):
        return []
    raw_values: list[Any]
    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        raw_values = []
        for item in value:
            if isinstance(item, str) and "," in item:
                raw_values.extend(part.strip() for part in item.split(","))
            else:
                raw_values.append(item)
    else:
        raw_values = [value]
    parsed: list[float] = []
    for raw in raw_values:
        if raw in (None, ""):
            continue
        try:
            delta = float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("derivative.delta_ang must contain positive numeric Ang values.") from exc
        if delta <= 0:
            raise RuntimeError("derivative.delta_ang must contain positive numeric Ang values.")
        parsed.append(delta)
    return parsed


def _normalize_optional_string_list(value: Any, *, field: str) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value]
    else:
        raise RuntimeError(f"{field} must be a list or comma-separated string.")
    return [item for item in items if item]


def _require_derivative_field(config: dict[str, Any], field: str, *, stage: str) -> None:
    value = config.get(field)
    if value in (None, "", []):
        display_field = "delta_ang" if field == "delta_ang_values" else field
        raise RuntimeError(f"derivative stage {stage!r} requires derivative.{display_field}.")


def _validate_derivative_workflow_config(stages: dict[str, bool], config: dict[str, Any]) -> None:
    enabled_derivative_stages = [stage for stage in MODULAR_WORKFLOW_STAGE_NAMES if stage in DERIVATIVE_STAGE_NAMES and stages.get(stage)]
    if not enabled_derivative_stages:
        return
    if not config.get("enabled"):
        raise RuntimeError(
            "derivative.enabled must be true when derivative stages are enabled: "
            + ", ".join(enabled_derivative_stages)
        )
    if config["method"] not in {"central", "forward", "backward"}:
        raise RuntimeError("derivative.method must be one of: central, forward, backward.")
    if config["base_split"] not in {"train", "validation", "test", "all"}:
        raise RuntimeError("derivative.base_split must be one of: train, validation, test, all.")
    if config.get("max_base_snapshots") is not None and int(config["max_base_snapshots"]) <= 0:
        raise RuntimeError("derivative.max_base_snapshots must be positive when provided.")
    policy = str(config.get("base_selection_policy") or "all")
    if policy not in {"all", "first", "adaptive_min_fraction"}:
        raise RuntimeError("derivative.base_selection_policy must be one of: all, first, adaptive_min_fraction.")
    if policy == "adaptive_min_fraction":
        if config.get("max_base_snapshots") is not None:
            raise RuntimeError(
                "derivative.max_base_snapshots cannot be combined with "
                "derivative.base_selection_policy adaptive_min_fraction."
            )
        if config.get("min_base_snapshots") is None:
            raise RuntimeError(
                "derivative.min_base_snapshots is required when "
                "derivative.base_selection_policy is adaptive_min_fraction."
            )
        if int(config["min_base_snapshots"]) <= 0:
            raise RuntimeError("derivative.min_base_snapshots must be positive when adaptive base selection is used.")
        if config.get("base_fraction") is None:
            raise RuntimeError(
                "derivative.base_fraction is required when "
                "derivative.base_selection_policy is adaptive_min_fraction."
            )
        if not (0 < float(config["base_fraction"]) <= 1):
            raise RuntimeError("derivative.base_fraction must be in (0, 1] when adaptive base selection is used.")
    invalid_axes = [axis for axis in config.get("axes", []) if axis not in {"x", "y", "z"}]
    if invalid_axes:
        raise RuntimeError(f"derivative.axes contains unsupported axes: {', '.join(invalid_axes)}.")

    if stages.get("build_derivative_stencils"):
        _require_derivative_field(config, "source_dataset_root", stage="build_derivative_stencils")
        _require_derivative_field(config, "delta_ang_values", stage="build_derivative_stencils")
        _require_derivative_field(config, "atoms", stage="build_derivative_stencils")
        _require_derivative_field(config, "axes", stage="build_derivative_stencils")

    result_consuming_stages = [
        "validate_derivative_stencils",
        "run_derivative_siesta_reference",
        "predict_derivative_graph2mat",
        "predict_derivative_deeph",
        "derivative_metrics_graph2mat",
        "derivative_metrics_deeph",
        "derivative_gate_check",
        "derivative_plots",
    ]
    h_benchmark_produces_metric_inputs = bool(stages.get("hamiltonian_metrics"))
    derivative_workflow_produces_inputs = bool(stages.get("build_derivative_stencils"))
    for stage in result_consuming_stages:
        if not stages.get(stage):
            continue
        if derivative_workflow_produces_inputs:
            continue
        if h_benchmark_produces_metric_inputs and stage in {
            "derivative_metrics_graph2mat",
            "derivative_metrics_deeph",
            "derivative_gate_check",
            "derivative_plots",
        }:
            continue
        if config.get("result_dir") in (None, "") and config.get("output_root") in (None, ""):
            raise RuntimeError(
                f"derivative stage {stage!r} requires derivative.result_dir or derivative.output_root."
            )


def _normalized_modular_workflow_payload(payload: dict[str, Any]) -> dict[str, Any]:
    mode = _normalize_modular_workflow_mode(payload)
    stages = _normalize_modular_workflow_stages(payload)
    derivative = _normalize_derivative_workflow_config(payload)
    _validate_derivative_workflow_config(stages, derivative)
    return {
        "schema": "graph2mat_deeph_modular_workflow_config_v1",
        "workflow_mode": mode or "default",
        "stages": stages,
        "derivative": derivative,
    }


def _has_derivative_stencils(
    *,
    result_dir: Path,
    source_model: str,
    settings: dict[str, Any],
) -> bool:
    from hamiltonian_derivative_stencil import discover_derivative_stencils

    discoveries = discover_derivative_stencils(
        result_dir,
        method=source_model,
        split=str(settings.get("split") or "all"),
        finite_difference_method=str(settings.get("finite_difference_method") or "central"),
        require_central=bool(settings.get("require_central")),
    )
    return any(discovery.stencil is not None for discovery in discoveries)


def _derivative_metric_command_args(
    *,
    python_executable: str,
    result_dir: Path,
    output_dir: Path,
    source_model: str,
    settings: dict[str, Any],
) -> list[str]:
    command = [
        python_executable,
        str(DEFAULT_DERIVATIVE_METRICS_SCRIPT),
        str(result_dir),
        "--method",
        str(settings["finite_difference_method"]),
        "--split",
        str(settings["split"]),
        "--support-threshold",
        str(settings["support_threshold"]),
        "--output-dir",
        str(output_dir),
        "--source-model",
        source_model,
        "--overwrite",
    ]
    if settings.get("require_central"):
        command.append("--require-central")
    if settings.get("diagnostic_only"):
        command.append("--diagnostic-only")
    if settings.get("max_stencils") is not None:
        command.extend(["--max-stencils", str(settings["max_stencils"])])
    return command


def _derivative_cost_dataset_size(source_dataset_root: str) -> int | None:
    if not source_dataset_root:
        return None
    path = Path(source_dataset_root)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return None
    frozen_split = path / "frozen_split_manifest.json"
    if not frozen_split.exists():
        return None
    try:
        payload = _load_json(frozen_split)
    except Exception:
        return None
    split_counts = payload.get("split_counts") if isinstance(payload.get("split_counts"), dict) else {}
    if split_counts:
        try:
            return sum(int(value) for value in split_counts.values())
        except (TypeError, ValueError):
            return None
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    return len(rows) if rows else None


def _derivative_cost_summary(
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
    reference_workers: int,
) -> dict[str, Any]:
    source_dataset_root = str(manifest.get("source_dataset_root") or config.get("source_dataset_root") or "")
    dataset_id = str(config.get("dataset_id") or (Path(source_dataset_root).name if source_dataset_root else ""))
    structure_count = manifest.get("expected_total_structure_samples")
    if structure_count is None:
        structure_count = manifest.get("sample_count")
    return {
        "dataset_id": dataset_id,
        "dataset_size": config.get("dataset_size") or _derivative_cost_dataset_size(source_dataset_root),
        "n_test_available": manifest.get("available_base_snapshot_count"),
        "derivative_k_selected": manifest.get("selected_base_snapshot_count"),
        "derivative_structure_count": structure_count,
        "derivative_reference_workers": reference_workers,
    }


def _logged_subprocess_run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    label: str,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    started_at = time.time()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    finished_at = time.time()
    record = {
        "label": label,
        "command": list(command),
        "cwd": str(cwd),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": finished_at - started_at,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode not in allowed_returncodes:
        raise CommandRunError(f"{label} failed with exit code {completed.returncode}", record)
    return record


def _derivative_output_root(run_root: Path, method: str) -> Path:
    return run_root / "common_metrics" / f"{method}_eval" / "derivative_metrics"


def _existing_derivative_root(run_root: Path, method: str) -> Path | None:
    root = _derivative_output_root(run_root, method)
    return root if (root / "manifest.json").exists() else None


def _json_safe_payload(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_payload(item) for item in value]
    return value


def _run_status_label(record: dict[str, Any]) -> str:
    status = str(record.get("status") or "").strip()
    if status:
        return status
    return "completed" if int(record.get("returncode") or 0) in {0, 2} else "failed"


DERIVATIVE_PAIR_KEY_FIELDS = (
    "base_sample_id",
    "atom_index_zero_based",
    "axis",
    "delta_ang",
    "finite_difference_method",
)
DERIVATIVE_PAIR_METRICS = (
    "dh_mae_union_eV_per_Ang",
    "dh_rmse_union_eV_per_Ang",
    "dh_max_abs_error_union_eV_per_Ang",
    "dh_relative_frobenius_ref",
    "dh_relative_l1_union",
    "dh_cosine_similarity_union",
    "dh_support_precision",
    "dh_support_recall",
    "dh_support_f1",
    "dh_false_zero_rate",
    "dh_false_nonzero_rate",
    "dh_hermiticity_error_delta",
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _derivative_pair_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field) or "").strip() for field in DERIVATIVE_PAIR_KEY_FIELDS)


def _derivative_comparison_claim_status(gate_report: dict[str, Any] | None) -> str:
    status = str((gate_report or {}).get("scientific_status") or "").strip()
    if status == "blocked":
        return "blocked"
    if status in {"technical_presentation", "paper_level_candidate"}:
        return "presentation_ready"
    return "diagnostic_only"


def _derivative_winner_claim_allowed(gate_report: dict[str, Any] | None) -> bool:
    status = str((gate_report or {}).get("scientific_status") or "").strip()
    if status not in {"technical_presentation", "paper_level_candidate"}:
        return False
    blocked_claims = " ".join(str(item) for item in (gate_report or {}).get("blocked_claims") or [])
    return "winner" not in blocked_claims.lower()


def _winner_from_delta(delta: float | None) -> str | None:
    if delta is None:
        return None
    if abs(delta) <= 1e-15:
        return "tie"
    return "graph2mat" if delta < 0 else "deeph"


def _mean_delta(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_finite_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def build_derivative_model_comparison_summary(
    *,
    graph2mat_root: Path | None,
    deeph_root: Path | None,
    output_dir: Path,
    gate_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    graph2mat_metric_path = Path(graph2mat_root) / "derivative_matrix_metrics.csv" if graph2mat_root is not None else None
    deeph_metric_path = Path(deeph_root) / "derivative_matrix_metrics.csv" if deeph_root is not None else None
    graph_rows = _read_csv_rows(graph2mat_metric_path) if graph2mat_metric_path is not None else []
    deeph_rows = _read_csv_rows(deeph_metric_path) if deeph_metric_path is not None else []
    graph_by_key = {_derivative_pair_key(row): row for row in graph_rows if any(_derivative_pair_key(row))}
    deeph_by_key = {_derivative_pair_key(row): row for row in deeph_rows if any(_derivative_pair_key(row))}
    paired_keys = sorted(set(graph_by_key) & set(deeph_by_key))
    paired_rows: list[dict[str, Any]] = []
    for key in paired_keys:
        graph_row = graph_by_key[key]
        deeph_row = deeph_by_key[key]
        row: dict[str, Any] = {field: key[index] for index, field in enumerate(DERIVATIVE_PAIR_KEY_FIELDS)}
        row.update(
            {
                "graph2mat_sample": graph_row.get("sample"),
                "deeph_sample": deeph_row.get("sample"),
                "graph2mat_comparison_status": graph_row.get("comparison_status"),
                "deeph_comparison_status": deeph_row.get("comparison_status"),
            }
        )
        for metric in DERIVATIVE_PAIR_METRICS:
            graph_value = _finite_float(graph_row.get(metric))
            deeph_value = _finite_float(deeph_row.get(metric))
            row[f"graph2mat_{metric}"] = graph_value
            row[f"deeph_{metric}"] = deeph_value
            row[f"delta_graph2mat_minus_deeph_{metric}"] = (
                None if graph_value is None or deeph_value is None else graph_value - deeph_value
            )
        paired_rows.append(row)

    missing_graph2mat = sorted(set(deeph_by_key) - set(graph_by_key))
    missing_deeph = sorted(set(graph_by_key) - set(deeph_by_key))
    claim_status = _derivative_comparison_claim_status(gate_report)
    winner_allowed = _derivative_winner_claim_allowed(gate_report)
    mean_mae_delta = _mean_delta(paired_rows, "delta_graph2mat_minus_deeph_dh_mae_union_eV_per_Ang")
    winner = _winner_from_delta(mean_mae_delta) if winner_allowed and paired_rows else None
    block_metrics = {
        "status": "block_metrics_unavailable",
        "reason": "No verified orbital-to-atom block mapping is present in derivative metric rows.",
    }
    summary = {
        "schema": "graph2mat_deeph_derivative_model_comparison_v1",
        "claim_status": claim_status,
        "winner_claim_allowed": winner_allowed,
        "winner": winner,
        "winner_metric": "mean_delta_graph2mat_minus_deeph_dh_mae_union_eV_per_Ang" if winner_allowed else None,
        "paired_count": len(paired_rows),
        "graph2mat_rows": len(graph_rows),
        "deeph_rows": len(deeph_rows),
        "missing_graph2mat_count": len(missing_graph2mat),
        "missing_deeph_count": len(missing_deeph),
        "mean_delta_graph2mat_minus_deeph_dh_mae_union_eV_per_Ang": mean_mae_delta,
        "block_metrics": block_metrics,
        "outputs": {
            "paired_comparison_csv": str(output_dir / "derivative_model_paired_comparison.csv"),
            "summary_json": str(output_dir / "derivative_model_comparison_summary.json"),
        },
        "inputs": {
            "graph2mat_root": str(graph2mat_root) if graph2mat_root is not None else "",
            "deeph_root": str(deeph_root) if deeph_root is not None else "",
            "gate_scientific_status": str((gate_report or {}).get("scientific_status") or ""),
        },
        "missing_pairs": {
            "graph2mat": [dict(zip(DERIVATIVE_PAIR_KEY_FIELDS, key)) for key in missing_graph2mat],
            "deeph": [dict(zip(DERIVATIVE_PAIR_KEY_FIELDS, key)) for key in missing_deeph],
        },
    }
    _write_csv(output_dir / "derivative_model_paired_comparison.csv", paired_rows)
    _write_json(output_dir / "derivative_model_comparison_summary.json", summary)
    return summary


def run_derivative_postprocess(
    *,
    run_root: Path,
    graph2mat_result_dir: Path | None,
    deeph_result_dir: Path | None,
    settings: dict[str, Any],
    overwrite: bool = True,
    diagnostic_only: bool = True,
    python_executable: str,
    command_runner: Any | None = None,
    plot_writer: Any | None = None,
    gate_report_builder: Any | None = None,
    json_writer: Any | None = None,
    log: Any | None = None,
) -> dict[str, Any]:
    run_root = Path(run_root)
    common_summary_root = run_root / "common_metrics" / "summary"
    normalized_settings = {
        "enabled": bool(settings.get("enabled", True)),
        "finite_difference_method": str(settings.get("finite_difference_method") or "central"),
        "split": str(settings.get("split") or "test"),
        "require_central": bool(settings.get("require_central", True)),
        "diagnostic_only": bool(settings.get("diagnostic_only", diagnostic_only)),
        "support_threshold": float(settings.get("support_threshold", 1e-12) or 1e-12),
        "max_stencils": _optional_int_value(settings.get("max_stencils")),
        "overwrite": bool(settings.get("overwrite", overwrite)),
    }
    summary: dict[str, Any] = {
        "enabled": normalized_settings["enabled"],
        "settings": dict(normalized_settings),
        "execution": {},
        "plot_outputs": {"status": "skipped_disabled"},
        "gate_report": {"status": "skipped_disabled"},
        "model_comparison": {"status": "skipped_disabled"},
        "roots": {"graph2mat": "", "deeph": ""},
    }
    if not normalized_settings["enabled"]:
        return summary

    runner = command_runner or _logged_subprocess_run
    plot_writer = plot_writer or write_derivative_plot_outputs
    gate_report_builder = gate_report_builder or build_derivative_gate_report
    json_writer = json_writer or _write_json
    logger = log or (lambda _message: None)
    method_inputs = (
        ("graph2mat", graph2mat_result_dir),
        ("deeph", deeph_result_dir),
    )
    derivative_roots: list[Path] = []
    for method_name, raw_result_dir in method_inputs:
        output_dir = _derivative_output_root(run_root, method_name)
        record: dict[str, Any] = {
            "label": f"{method_name} derivative metrics",
            "status": "skipped_missing_input",
            "result_dir": str(raw_result_dir) if raw_result_dir is not None else "",
            "output_dir": str(output_dir),
            "enabled": True,
        }
        result_dir = Path(raw_result_dir) if raw_result_dir is not None else None
        if result_dir is None or not result_dir.exists():
            logger(
                "[G2M-DEEPH] derivative_metrics: skipped for "
                f"{method_name} because result_dir is missing.\n"
            )
            summary["execution"][method_name] = record
            continue
        if not _has_derivative_stencils(
            result_dir=result_dir,
            source_model=method_name,
            settings=normalized_settings,
        ):
            record["status"] = "skipped_no_stencils"
            logger(
                "[G2M-DEEPH] derivative_metrics: skipped for "
                f"{method_name} because no derivative stencils were discovered.\n"
            )
            summary["execution"][method_name] = record
            continue
        command = _derivative_metric_command_args(
            python_executable=python_executable,
            result_dir=result_dir,
            output_dir=output_dir,
            source_model=method_name,
            settings=normalized_settings,
        )
        try:
            command_record = runner(
                command,
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                label=f"{method_name} derivative metrics",
                allowed_returncodes=(0, 2),
            )
            record = {
                **record,
                **_json_safe_payload(command_record),
                "status": "completed",
            }
            derivative_roots.append(output_dir)
            summary["roots"][method_name] = str(output_dir)
            logger(
                "[G2M-DEEPH] derivative_metrics: completed for "
                f"{method_name} -> {output_dir}\n"
            )
        except CommandRunError as exc:
            record = {
                **record,
                **_json_safe_payload(exc.run_record),
                "status": "failed",
            }
            logger(
                "[G2M-DEEPH][WARN] Derivative metric postprocess failed "
                f"for {method_name}; existing H metrics are preserved.\n"
            )
        summary["execution"][method_name] = record

    graph2mat_root = _existing_derivative_root(run_root, "graph2mat")
    deeph_root = _existing_derivative_root(run_root, "deeph")
    if graph2mat_root is None and summary["roots"]["graph2mat"]:
        graph2mat_root = Path(summary["roots"]["graph2mat"])
    if deeph_root is None and summary["roots"]["deeph"]:
        deeph_root = Path(summary["roots"]["deeph"])

    if graph2mat_root is None and deeph_root is None:
        summary["plot_outputs"] = {"status": "skipped_no_stencils", "output_dir": str(common_summary_root / "derivative_plots")}
        summary["gate_report"] = {"status": "skipped_no_stencils", "output_path": str(common_summary_root / "derivative_gate_report.json")}
        return summary

    try:
        plot_result = plot_writer(
            derivative_roots=[],
            graph2mat_root=graph2mat_root,
            deeph_root=deeph_root,
            output_dir=common_summary_root / "derivative_plots",
        )
        summary["plot_outputs"] = {
            "status": "completed",
            "output_dir": str(common_summary_root / "derivative_plots"),
            "payload_path": str(plot_result["payload_path"]),
            "manifest_path": str(plot_result["manifest_path"]),
            "available": bool(plot_result["payload"].get("available")),
        }
        logger(
            "[G2M-DEEPH] derivative_metrics: plots updated -> "
            f"{common_summary_root / 'derivative_plots'}\n"
        )
    except Exception as exc:
        summary["plot_outputs"] = {
            "status": "failed",
            "output_dir": str(common_summary_root / "derivative_plots"),
            "error": str(exc),
        }
        logger(f"[G2M-DEEPH][WARN] Derivative plot postprocess failed: {exc}\n")

    try:
        roots_for_gate = [root for root in (graph2mat_root, deeph_root) if root is not None]
        report = gate_report_builder(
            derivative_roots=roots_for_gate,
            run_root=run_root,
        )
        gate_report_path = common_summary_root / "derivative_gate_report.json"
        json_writer(gate_report_path, report)
        summary["gate_report"] = {
            "status": "completed",
            "output_path": str(gate_report_path),
            "scientific_status": str(report.get("scientific_status") or ""),
            "blockers": len(report.get("blockers") or []),
            "warnings": len(report.get("warnings") or []),
        }
        logger(
            "[G2M-DEEPH] derivative_metrics: gate report updated -> "
            f"{gate_report_path}\n"
        )
    except Exception as exc:
        summary["gate_report"] = {
            "status": "failed",
            "output_path": str(common_summary_root / "derivative_gate_report.json"),
            "error": str(exc),
        }
        logger(f"[G2M-DEEPH][WARN] Derivative gate report generation failed: {exc}\n")
        report = None

    try:
        comparison = build_derivative_model_comparison_summary(
            graph2mat_root=graph2mat_root,
            deeph_root=deeph_root,
            output_dir=common_summary_root / "derivative_model_comparison",
            gate_report=report,
        )
        summary["model_comparison"] = {
            "status": "completed",
            "output_dir": str(common_summary_root / "derivative_model_comparison"),
            "paired_count": comparison["paired_count"],
            "claim_status": comparison["claim_status"],
            "winner_claim_allowed": comparison["winner_claim_allowed"],
            "winner": comparison["winner"],
        }
        logger(
            "[G2M-DEEPH] derivative_metrics: model comparison updated -> "
            f"{common_summary_root / 'derivative_model_comparison'}\n"
        )
    except Exception as exc:
        summary["model_comparison"] = {
            "status": "failed",
            "output_dir": str(common_summary_root / "derivative_model_comparison"),
            "error": str(exc),
        }
        logger(f"[G2M-DEEPH][WARN] Derivative model comparison failed: {exc}\n")
    return summary


def _backfill_result_dir(run_root: Path, method: str) -> Path | None:
    candidates = {
        "graph2mat": (
            run_root / "common_metrics" / "graph2mat_eval",
            run_root / "metrics" / "graph2mat" / "eval_input",
        ),
        "deeph": (
            run_root / "common_metrics" / "deeph_eval",
            run_root / "metrics" / "deeph" / "eval",
        ),
    }
    for candidate in candidates[method]:
        if candidate.exists():
            return candidate
    return None


def backfill_derivative_postprocess_from_training_sweep(
    *,
    training_sweep_manifest_path: Path,
    settings: dict[str, Any],
    python_executable: str,
    command_runner: Any | None = None,
    plot_writer: Any | None = None,
    gate_report_builder: Any | None = None,
    json_writer: Any | None = None,
    log: Any | None = None,
) -> dict[str, Any]:
    manifest_path = Path(training_sweep_manifest_path)
    manifest = _load_json(manifest_path)
    benchmark_run_root = manifest_path.parent.parent
    logger = log or (lambda _message: None)
    plot_writer = plot_writer or write_derivative_plot_outputs
    gate_report_builder = gate_report_builder or build_derivative_gate_report
    json_writer = json_writer or _write_json
    runs = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
    summary_runs: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for entry in runs:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status") or "").strip().lower() != "completed":
            continue
        raw_run_root = str(entry.get("run_root") or "").strip()
        if not raw_run_root:
            continue
        child_run_root = Path(raw_run_root).expanduser().resolve(strict=False)
        if not child_run_root.exists():
            continue
        logger(f"[G2M-DEEPH][backfill] Processing derivative postprocess for {child_run_root}\n")
        derivative_summary = run_derivative_postprocess(
            run_root=child_run_root,
            graph2mat_result_dir=_backfill_result_dir(child_run_root, "graph2mat"),
            deeph_result_dir=_backfill_result_dir(child_run_root, "deeph"),
            settings=settings,
            overwrite=bool(settings.get("overwrite", True)),
            diagnostic_only=bool(settings.get("diagnostic_only", True)),
            python_executable=python_executable,
            command_runner=command_runner,
            plot_writer=plot_writer,
            gate_report_builder=gate_report_builder,
            json_writer=json_writer,
            log=logger,
        )
        method_statuses = {
            method: _run_status_label(record)
            for method, record in derivative_summary.get("execution", {}).items()
            if isinstance(record, dict)
        }
        overall_status = "completed" if any(status == "completed" for status in method_statuses.values()) else "skipped_no_stencils"
        if any(status == "failed" for status in method_statuses.values()):
            overall_status = "failed"
        counts[overall_status] = counts.get(overall_status, 0) + 1
        plot_status = str((derivative_summary.get("plot_outputs") or {}).get("status") or "")
        gate_status = str((derivative_summary.get("gate_report") or {}).get("status") or "")
        model_comparison_status = str((derivative_summary.get("model_comparison") or {}).get("status") or "")
        summary_runs.append(
            {
                "run_root": str(child_run_root),
                "run_id": child_run_root.name,
                "overall_status": overall_status,
                "method_statuses": method_statuses,
                "plot_status": plot_status,
                "gate_status": gate_status,
                "model_comparison_status": model_comparison_status,
                "summary": derivative_summary,
            }
        )
    postprocess_stage_failed = any(
        run.get("plot_status") == "failed"
        or run.get("gate_status") == "failed"
        or run.get("model_comparison_status") == "failed"
        for run in summary_runs
    )
    backfill_summary = {
        "schema": "graph2mat_deeph_derivative_backfill_summary_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "training_sweep_manifest_path": str(manifest_path),
        "benchmark_run_root": str(benchmark_run_root),
        "settings": _json_safe_payload(settings),
        "processed_runs": len(summary_runs),
        "counts": counts,
        "runs": summary_runs,
        "status": "completed_with_failures" if postprocess_stage_failed else "completed",
    }
    output_path = benchmark_run_root / "summary" / "derivative_backfill_summary.json"
    json_writer(output_path, backfill_summary)
    return backfill_summary


def _strict_dataset_validation_kwargs(
    payload: dict[str, Any],
    *,
    dataset_root: Path,
    snapshot_root: Path,
) -> dict[str, Any]:
    material_provenance = dataset_root / "material_provenance.json"
    validation_profile = (
        G2M_DEEPH_BENCHMARK_PROFILE
        if _parse_bool(payload.get("strict_dataset_validation"), True)
        else None
    )
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
        "validation_profile": validation_profile,
    }


def _frozen_split_snapshot_dirs(dataset_root: Path) -> list[Path] | None:
    manifest_path = dataset_root / "frozen_split_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        rows = _load_json(manifest_path).get("rows") or []
    except Exception:
        return None
    dirs: list[Path] = []
    for row in rows:
        sample_dir = row.get("sample_dir") if isinstance(row, dict) else None
        if not sample_dir:
            continue
        path = Path(str(sample_dir))
        if not path.is_absolute():
            path = dataset_root / path
        if path.is_dir():
            dirs.append(path)
    return dirs or None


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
        self._external_final_log_offsets: dict[str, int] = {}
        self._external_final_last_sync = 0.0
        self._external_final_run_root: Path | None = None
        self._external_detached_log_offsets: dict[str, int] = {}
        self._external_detached_last_sync = 0.0
        self._external_detached_run_root: Path | None = None
        self._external_detached_status: dict[str, Any] = {}
        self._external_detached_status_signature = ""

    def _external_final_process_lines(self) -> list[str]:
        workflow_lines = _process_table_lines_for_text("paper_ready_final70")
        script_lines = _process_table_lines_for_text("run_paper_ready_final_70.sh")
        seen: set[str] = set()
        lines: list[str] = []
        for line in script_lines + workflow_lines:
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
        return lines

    def _external_final_active(self) -> bool:
        return bool(self._external_final_process_lines())

    def _latest_external_final_run_root(self) -> Path | None:
        candidates = sorted(
            (REPO_ROOT / "Comparison" / "results").glob(f"g2m_deeph_*/runs/{EXTERNAL_FINAL_RUN_GLOB}"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _sync_external_final_run_locked(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._external_final_last_sync < 5.0:
            return
        self._external_final_last_sync = now
        process_lines = self._external_final_process_lines()
        if not process_lines:
            return
        run_root = self._latest_external_final_run_root()
        if run_root is not None:
            self._external_final_run_root = run_root
        if run_root is None:
            self._logs.append(
                "[G2M-DEEPH][external] Paper-ready final runner activo, esperando run root materializado.\n"
            )
            return
        active_children = [
            line for line in process_lines
            if any(token in line for token in ("deeph-train", "deeph-preprocess", "run_md_training", "g2m_deeph_final_workflow"))
        ]
        self._logs.append(
            "[G2M-DEEPH][external] Watching detached paper-ready final run: "
            f"{run_root.name} | active_processes={len(active_children)} | "
            f"root={run_root}\n"
        )
        for line in active_children[:20]:
            self._logs.append(f"[G2M-DEEPH][external][ps] {line}\n")
        recent_files = sorted(
            [
                path
                for pattern in ("sweep/**/deeph/train/result.txt", "sweep/**/deeph/train/stderr.txt")
                for path in run_root.glob(pattern)
                if path.is_file()
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:12]
        for path in sorted(recent_files):
            key = str(path)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            start = self._external_final_log_offsets.get(key)
            if start is None:
                start = max(0, size - 4000)
                if start > 0:
                    self._logs.append(f"[G2M-DEEPH][external][tail] ... {path.relative_to(run_root)}\n")
            if size <= start:
                self._external_final_log_offsets[key] = size
                continue
            try:
                with path.open("rb") as handle:
                    handle.seek(start)
                    chunk = handle.read(min(size - start, 8000)).decode("utf-8", errors="replace")
            except OSError:
                continue
            self._external_final_log_offsets[key] = size
            rel = path.relative_to(run_root)
            for line in chunk.splitlines():
                if line.strip():
                    self._logs.append(f"[G2M-DEEPH][external][{rel}] {line}\n")

    def _latest_detached_running_run_root(self) -> Path | None:
        candidates: list[tuple[float, Path]] = []
        search_root = _resolve_optional_repo_path(
            os.environ.get(DETACHED_SEARCH_ROOT_ENV),
            REPO_ROOT / "Comparison" / "results",
        )
        patterns = (
            "*/runner_status.json",
            "*/*/runner_status.json",
            "*/runs/*/runner_status.json",
            "metric_archives_by_run/*/core/runner_status.json",
        )
        seen: set[str] = set()
        current_root = self._external_detached_run_root
        if current_root is not None:
            status_path = current_root / "runner_status.json"
            if status_path.exists():
                seen.add(str(status_path.resolve(strict=False)))
                try:
                    payload = _load_json(status_path)
                except Exception:
                    payload = {}
                status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
                if status and bool(status.get("running")):
                    started_at = status.get("started_at")
                    try:
                        sort_key = float(started_at)
                    except (TypeError, ValueError):
                        try:
                            sort_key = status_path.stat().st_mtime
                        except OSError:
                            sort_key = 0.0
                    candidates.append((sort_key, current_root))
        for pattern in patterns:
            for status_path in search_root.glob(pattern):
                resolved_key = str(status_path.resolve(strict=False))
                if resolved_key in seen:
                    continue
                seen.add(resolved_key)
                run_root = status_path.parent
                try:
                    payload = _load_json(status_path)
                except Exception:
                    continue
                status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
                if not status or not bool(status.get("running")):
                    continue
                if self._state.run_root and Path(str(self._state.run_root)).resolve(strict=False) == run_root.resolve(strict=False):
                    continue
                started_at = status.get("started_at")
                try:
                    sort_key = float(started_at)
                except (TypeError, ValueError):
                    try:
                        sort_key = status_path.stat().st_mtime
                    except OSError:
                        sort_key = 0.0
                candidates.append((sort_key, run_root))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _sync_detached_external_run_locked(self, *, force: bool = False) -> bool:
        now = time.time()
        if not force and now - self._external_detached_last_sync < 5.0:
            return self._external_detached_run_root is not None and bool(self._external_detached_status.get("running"))
        self._external_detached_last_sync = now
        run_root = self._latest_detached_running_run_root()
        if run_root is None:
            self._external_detached_run_root = None
            self._external_detached_status = {}
            self._external_detached_status_signature = ""
            return False
        try:
            payload = _load_json(run_root / "runner_status.json")
        except Exception:
            return False
        status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
        if not status or not bool(status.get("running")):
            self._external_detached_run_root = None
            self._external_detached_status = {}
            self._external_detached_status_signature = ""
            return False
        self._external_detached_run_root = run_root
        self._external_detached_status = dict(status)
        training_sweep = status.get("training_sweep") if isinstance(status.get("training_sweep"), dict) else {}
        signature = json.dumps(
            {
                "run_id": status.get("run_id") or run_root.name,
                "stage": status.get("stage") or "",
                "completed": training_sweep.get("completed"),
                "failed": training_sweep.get("failed"),
                "total": training_sweep.get("total"),
                "active_model": training_sweep.get("active_model"),
                "active_dataset": training_sweep.get("active_dataset"),
            },
            sort_keys=True,
            default=str,
        )
        if signature != self._external_detached_status_signature:
            self._external_detached_status_signature = signature
            self._logs.append(
                "[G2M-DEEPH][external] Watching detached benchmark run: "
                f"{status.get('run_id') or run_root.name} | stage={status.get('stage') or 'unknown'} | "
                f"progress={training_sweep.get('completed', 0)}/{training_sweep.get('total', 0)} | "
                f"root={run_root}\n"
            )
        recent_files = sorted(
            [
                path
                for pattern in (
                    "sweep/**/train/result.txt",
                    "sweep/**/train/stderr.txt",
                    "sweep/**/inference/**/result.txt",
                    "sweep/**/inference/**/stderr.txt",
                    "sweep/**/stdout.txt",
                    "sweep/**/stderr.txt",
                )
                for path in run_root.glob(pattern)
                if path.is_file()
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:16]
        for path in sorted(recent_files):
            key = str(path)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            start = self._external_detached_log_offsets.get(key)
            if start is None:
                start = max(0, size - 4000)
                if start > 0:
                    self._logs.append(f"[G2M-DEEPH][external][tail] ... {path.relative_to(run_root)}\n")
            if size <= start:
                self._external_detached_log_offsets[key] = size
                continue
            try:
                with path.open("rb") as handle:
                    handle.seek(start)
                    chunk = handle.read(min(size - start, 8000)).decode("utf-8", errors="replace")
            except OSError:
                continue
            self._external_detached_log_offsets[key] = size
            rel = path.relative_to(run_root)
            for line in chunk.splitlines():
                if line.strip():
                    self._logs.append(f"[G2M-DEEPH][external][{rel}] {line}\n")
        return True

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
        snapshot_dirs = _frozen_split_snapshot_dirs(dataset_root) if snapshot_root == dataset_root else None
        validation_root = snapshot_root if snapshot_root != dataset_root else dataset_root
        result = validate_dataset(
            validation_root,
            snapshot_dirs=snapshot_dirs,
            **validate_kwargs,
        )

        if snapshot_root != dataset_root:
            result.dataset_root = dataset_root

        validation = _validation_payload(
            result,
            snapshot_root=snapshot_root,
            max_snapshots=int(payload.get("max_invalid_snapshot_preview", 20) or 20),
        )
        if not _parse_bool(payload.get("strict_dataset_validation"), True):
            validation.setdefault("warnings", []).append(
                "strict_dataset_validation=false: dataset provenance gates are exploratory-only; "
                "do not use this run for robust scientific claims."
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
        split: str = "test",
    ) -> list[str]:
        selected_rows = [row for row in frozen_split.get("rows", []) if row.get("split") == split]
        if not selected_rows:
            raise RuntimeError(f"Frozen split manifest has no {split} rows for prediction.")
        if destination_root.exists():
            shutil.rmtree(destination_root)
        sample_ids: list[str] = []
        for row in selected_rows:
            sample_id = str(row.get("sample_id") or row.get("deeph_sample_id") or "").strip()
            sample_dir = Path(str(row.get("sample_dir") or ""))
            if not sample_id:
                raise RuntimeError(f"Frozen split {split} row is missing sample_id: {row}")
            if not sample_dir.exists():
                raise RuntimeError(f"Frozen split {split} sample_dir does not exist: {sample_dir}")
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
        prediction_split = _metric_evaluation_split(payload)
        prediction_structs_dir = graph2mat_root / "prediction_structures" / prediction_split
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
            split=prediction_split,
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
            prediction_split=prediction_split,
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
        for key in (
            "optim_lr",
            "hidden_irreps",
            "num_interactions",
            "correlation",
            "max_ell",
            "loss",
            "node_block_readout",
            "edge_block_readout",
            "preprocessing_edges",
            "preprocessing_edges_reuse_nodes",
        ):
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
        try:
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
        except OSError as exc:
            finished_at = time.time()
            failure = classify_failure(returncode=127, output_excerpt=str(exc))
            run_record = {
                "label": label,
                "command": command,
                "cwd": str(cwd),
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_seconds": finished_at - started_at,
                "returncode": 127,
                "controlled_stop_reason": None,
                "telemetry": {
                    "telemetry_status": "unavailable",
                    "warnings": ["subprocess failed to start"],
                    **failure,
                },
            }
            raise CommandRunError(f"{label} failed to start: {exc}", run_record) from exc
        telemetry_monitor = GpuTelemetryMonitor()
        resource_monitor = ProcResourceMonitor()
        telemetry_monitor.start(process.pid)
        resource_monitor.start(process.pid)
        with self._lock:
            self._processes.append(process)
        output_tail: list[str] = []

        def remember_output(line: str) -> None:
            output_tail.append(str(line).strip())
            del output_tail[:-40]

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
                            remember_output(line)
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
                            remember_output(line)
                            with self._lock:
                                self._logs.append(line + "\n")
                            if line_observer is not None and controlled_stop_reason is None:
                                reason = line_observer(line)
                                if reason:
                                    controlled_stop_reason = str(reason)
                    break
            if pending:
                remember_output(pending)
                with self._lock:
                    self._logs.append(pending)
            returncode = process.wait()
        finally:
            gpu_telemetry = telemetry_monitor.stop()
            proc_telemetry = resource_monitor.stop()
            if process.stdout is not None:
                process.stdout.close()
            with self._lock:
                self._processes = [item for item in self._processes if item is not process]
        finished_at = time.time()
        output_excerpt = "\n".join(line for line in output_tail if line)
        failure = classify_failure(
            returncode=returncode,
            output_excerpt=output_excerpt,
            controlled_stop_reason=controlled_stop_reason,
            stop_requested=bool(self._state.stop_requested),
        )
        command_warnings: list[str] = []
        for warning in [*(gpu_telemetry.get("warnings") or []), *(proc_telemetry.get("warnings") or [])]:
            if warning and warning not in command_warnings:
                command_warnings.append(str(warning))
        observed_resource = any(
            proc_telemetry.get(key) is not None
            for key in ("cpu_time_seconds", "cpu_peak_percent", "peak_rss_mb")
        ) or any(
            gpu_telemetry.get(key) is not None
            for key in ("peak_gpu_memory_mb", "gpu_active_seconds", "observed_gpu_count")
        )
        telemetry_status = (
            "complete"
            if observed_resource and not command_warnings
            else "partial"
            if observed_resource or command_warnings
            else "unavailable"
        )
        run_record = {
            "label": label,
            "command": command,
            "cwd": str(cwd),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": finished_at - started_at,
            "returncode": returncode,
            "controlled_stop_reason": controlled_stop_reason,
            "telemetry": {
                **gpu_telemetry,
                **{key: value for key, value in proc_telemetry.items() if key != "warnings"},
                "gpu_hours": compute_gpu_hours(
                    gpu_telemetry.get("gpu_active_seconds"),
                    gpu_telemetry.get("observed_gpu_count"),
                ),
                "telemetry_status": telemetry_status,
                "warnings": command_warnings,
                **failure,
            },
        }
        if returncode not in allowed_returncodes and controlled_stop_reason is None:
            raise CommandRunError(f"{label} failed with exit code {returncode}", run_record)
        return run_record

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
        disable_cuda, device = _deeph_device_settings(payload, options)
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
        inference_split = _metric_evaluation_split(payload)
        for row in raw_mirror["rows"]:
            if row.get("split") != inference_split:
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
            inference_split=inference_split,
            dry_run=_parse_bool(payload.get("dry_run"), False),
        )
        self._write_deeph_manifest(context, extra={"prepared": True})
        return context

    def _stage_deeph_inference_inputs(self, context: DeepHBenchmarkContext) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for row in context.raw_mirror.get("rows", []):
            if row.get("split") != context.inference_split:
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
                f"Missing DeepH processed {context.inference_split} samples before inference: "
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
        split: str = "test",
    ) -> Path:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        split_root = dataset_root / "splits"
        for row in frozen_split_manifest.get("rows") or []:
            if row.get("split") != split:
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
        run_validation_metrics = final_mode and _search_validation_metrics_requested(child)
        if final_mode and not run_validation_metrics:
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
        if run_validation_metrics:
            self._logs.append(
                f"[G2M-DEEPH] Graph2Mat sweep {record['config_id']}: "
                "evaluating validation-only spectral/DOS metrics; test split remains locked.\n"
            )
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
            split=context.prediction_split,
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
        validation_metrics = _extract_validation_metrics(staged.result_dir / "metrics") if run_validation_metrics else {}
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
                                "validation_metrics": validation_metrics,
                                **(search_stage_record_fields() if final_mode else {}),
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
                "metric_split": context.prediction_split,
                "validation_metrics": validation_metrics,
                "validation_metrics_path": str(staged.result_dir / "metrics" / "manifest.json"),
                **(search_stage_record_fields() if final_mode else {}),
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
        run_validation_metrics = final_mode and _search_validation_metrics_requested(child)
        if final_mode and not run_validation_metrics:
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
        if run_validation_metrics:
            self._logs.append(
                f"[G2M-DEEPH] DeepH sweep {record['config_id']}: "
                "evaluating validation-only spectral/DOS metrics; test split remains locked.\n"
            )
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
            split=deeph_context.inference_split,
        )
        staged_deeph = stage_deeph_metric_inputs(
            raw_mirror=deeph_context.raw_mirror,
            processed_dir=deeph_context.processed_dir,
            inference_dir=deeph_context.inference_dir,
            output_dir=metrics_root / "deeph_inputs",
            split=deeph_context.inference_split,
        )
        deeph_metric_command = _deeph_metric_command_args(
            python_executable=self._graph2mat_python(child),
            graph2mat_result_dir=reference_dir,
            processed_dir=staged_deeph.processed_dir,
            predictions_dir=staged_deeph.predictions_dir,
            output_dir=metrics_root / "eval",
            metric_fail_policy=metric_fail_policy,
            split=deeph_context.inference_split,
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
        validation_metrics = _extract_validation_metrics(metrics_root / "eval" / "metrics") if run_validation_metrics else {}
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
                "validation_metrics": validation_metrics,
                **(search_stage_record_fields() if final_mode else {}),
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
                "metric_split": deeph_context.inference_split,
                "validation_metrics": validation_metrics,
                "validation_metrics_path": str(metrics_root / "eval" / "metrics" / "manifest.json"),
                **(search_stage_record_fields() if final_mode else {}),
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

    def _training_sweep_derivative_child_payload(
        self,
        payload: dict[str, Any],
        *,
        child_run_root: Path,
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        workflow = payload["modular_workflow"]
        stages = dict(workflow["stages"])
        derivative = dict(workflow["derivative"])
        if stages.get("build_derivative_stencils"):
            dataset_root = entry.get("dataset_root")
            if dataset_root in (None, ""):
                raise RuntimeError(
                    "missing dataset_root for completed training sweep child run with "
                    "build_derivative_stencils enabled."
                )
            child_dataset_root = _resolve_optional_repo_path(dataset_root, Path(str(dataset_root)))
            derivative["source_dataset_root"] = str(child_dataset_root)
        if not any(
            stages.get(stage)
            for stage in (
                "build_derivative_stencils",
                "validate_derivative_stencils",
                "run_derivative_siesta_reference",
                "predict_derivative_graph2mat",
                "predict_derivative_deeph",
            )
        ):
            graph2mat_result_dir = _backfill_result_dir(child_run_root, "graph2mat")
            deeph_result_dir = _backfill_result_dir(child_run_root, "deeph")
            if graph2mat_result_dir is not None and derivative.get("graph2mat_result_dir") in (None, ""):
                derivative["graph2mat_result_dir"] = str(graph2mat_result_dir)
            if deeph_result_dir is not None and derivative.get("deeph_result_dir") in (None, ""):
                derivative["deeph_result_dir"] = str(deeph_result_dir)
        child = dict(payload)
        child["modular_workflow"] = {
            **workflow,
            "stages": stages,
            "derivative": derivative,
        }
        return child

    def _training_sweep_derivative_contexts(
        self,
        record: dict[str, Any],
    ) -> tuple[Graph2MatBenchmarkContext | None, DeepHBenchmarkContext | None]:
        graph_manifest = self._manifest_path_value(record.get("graph2mat_manifest_path"), base_dir=REPO_ROOT)
        deeph_manifest = self._manifest_path_value(record.get("deeph_manifest_path"), base_dir=REPO_ROOT)
        return (
            self._graph2mat_context_from_manifest_path(graph_manifest),
            self._deeph_context_from_manifest_path(deeph_manifest),
        )

    def _training_sweep_derivative_enabled_models(
        self,
        stages: dict[str, bool],
    ) -> list[str]:
        return [
            model
            for model in ("graph2mat", "deeph")
            if stages.get(f"predict_derivative_{model}") or stages.get(f"derivative_metrics_{model}")
        ]

    def _training_sweep_derivative_group_label(
        self,
        entries: list[dict[str, Any]],
        *,
        dataset_root: Path | None,
    ) -> str:
        for key in ("dataset_id", "recipe_id", "config_id"):
            for entry in entries:
                value = str(entry.get(key) or "").strip()
                if value:
                    return slugify_label(value, "dataset")
        if dataset_root is not None:
            return slugify_label(dataset_root.name, "dataset")
        return stable_payload_hash(entries, length=12)

    def _select_training_sweep_derivative_entry(
        self,
        entries: list[dict[str, Any]],
        *,
        dataset_root: Path | None,
        model: str,
        required: bool,
    ) -> dict[str, Any] | None:
        matches = [entry for entry in entries if str(entry.get("model") or "").strip() == model]
        if not matches:
            if required:
                dataset_label = str(dataset_root) if dataset_root is not None else "<missing dataset_root>"
                raise RuntimeError(
                    f"training sweep derivative handoff requires a completed {model} child run for dataset_root "
                    f"{dataset_label}, but none were found."
                )
            return None
        if len(matches) > 1:
            dataset_label = str(dataset_root) if dataset_root is not None else "<missing dataset_root>"
            inspected = ", ".join(str(entry.get("run_root") or "<missing run_root>") for entry in matches)
            raise RuntimeError(
                f"training sweep derivative handoff is ambiguous for dataset_root {dataset_label}: "
                f"found {len(matches)} completed {model} child runs ({inspected})."
            )
        return matches[0]

    def _training_sweep_derivative_launch_payload(
        self,
        payload: dict[str, Any],
        *,
        derivative_run_root: Path,
        dataset_root: Path | None,
        graph2mat_context: Graph2MatBenchmarkContext | None,
        deeph_context: DeepHBenchmarkContext | None,
        multiple_groups: bool,
        group_label: str,
    ) -> dict[str, Any]:
        workflow = payload["modular_workflow"]
        stages = dict(workflow["stages"])
        derivative = dict(workflow["derivative"])
        if stages.get("build_derivative_stencils"):
            if dataset_root is None:
                raise RuntimeError(
                    "missing dataset_root for completed training sweep dataset group with "
                    "build_derivative_stencils enabled."
                )
            derivative["source_dataset_root"] = str(dataset_root)
        explicit_root = derivative.get("result_dir") or derivative.get("output_root")
        if explicit_root not in (None, ""):
            base_root = _resolve_optional_repo_path(explicit_root, Path(str(explicit_root)))
            derivative["result_dir"] = str(base_root / group_label) if multiple_groups else str(base_root)
            derivative.pop("output_root", None)
        else:
            derivative["result_dir"] = str(derivative_run_root)
        for model, field, context in (
            ("graph2mat", "graph2mat_checkpoint", graph2mat_context),
            ("deeph", "deeph_model_dir", deeph_context),
        ):
            if derivative.get(field) not in (None, "") or context is None:
                continue
            derivative[field] = str(
                self._require_inferred_derivative_model_artifact(
                    model,
                    graph2mat_context=graph2mat_context,
                    deeph_context=deeph_context,
                )
            )
        child = dict(payload)
        child["modular_workflow"] = {
            **workflow,
            "stages": stages,
            "derivative": derivative,
        }
        return child

    def _run_training_sweep_derivative_workflows(
        self,
        payload: dict[str, Any],
        *,
        run_root: Path,
        summary: dict[str, Any],
    ) -> list[dict[str, Any]]:
        workflow = payload.get("modular_workflow") if isinstance(payload.get("modular_workflow"), dict) else {}
        stages = workflow.get("stages") if isinstance(workflow.get("stages"), dict) else {}
        if not any(stages.get(stage) for stage in DERIVATIVE_STAGE_NAMES):
            return []
        derivative_parallelism = _derivative_workflow_parallelism(payload)

        def set_derivative_status(*, total: int, queued: int, active: int, completed: int, failed: int) -> None:
            with self._lock:
                derivatives = dict(self._state.training_sweep_status.get("derivatives") or {})
                derivatives.update(
                    {
                        "enabled": True,
                        "total": total,
                        "queued": queued,
                        "active": active,
                        "completed": completed,
                        "failed": failed,
                        "max_parallel_derivative_workflows": derivative_parallelism,
                    }
                )
                self._state.training_sweep_status["derivatives"] = derivatives

        workflow_mode = str(workflow.get("workflow_mode") or "")
        records: list[dict[str, Any]] = []
        jobs: list[dict[str, Any]] = []

        def register_job(
            record: dict[str, Any],
            *,
            child_payload: dict[str, Any],
            derivative_run_root: Path,
            graph2mat_context: Graph2MatBenchmarkContext | None,
            deeph_context: DeepHBenchmarkContext | None,
            success_fields: dict[str, Any] | None = None,
        ) -> None:
            records.append(record)
            jobs.append(
                {
                    "index": len(records) - 1,
                    "run_id": record["run_id"],
                    "payload": child_payload,
                    "derivative_run_root": derivative_run_root,
                    "graph2mat_context": graph2mat_context,
                    "deeph_context": deeph_context,
                    "output_root": self._derivative_root(child_payload, run_root=derivative_run_root),
                    "success_fields": dict(success_fields or {}),
                }
            )

        if workflow_mode not in {"h_then_derivative_full", "full_end_to_end"}:
            for entry in list(summary.get("runs") or []):
                if not isinstance(entry, dict) or str(entry.get("status") or "") != "completed":
                    continue
                raw_run_root = str(entry.get("run_root") or "").strip()
                if not raw_run_root:
                    continue
                child_run_root = Path(raw_run_root).expanduser().resolve(strict=False)
                record = {
                    "run_id": str(entry.get("config_id") or child_run_root.name),
                    "child_result_dir": str(child_run_root),
                    "derivative_workflow_status": "pending",
                    "derivative_workflow_manifest_path": "",
                    "failure_reason": "",
                }
                try:
                    graph2mat_context, deeph_context = self._training_sweep_derivative_contexts(entry)
                    child_payload = self._training_sweep_derivative_child_payload(
                        payload,
                        child_run_root=child_run_root,
                        entry=entry,
                    )
                    register_job(
                        record,
                        child_payload=child_payload,
                        derivative_run_root=child_run_root,
                        graph2mat_context=graph2mat_context,
                        deeph_context=deeph_context,
                    )
                except Exception as exc:
                    record.update(
                        {
                            "derivative_workflow_status": "failed",
                            "failure_reason": str(exc),
                        }
                    )
                    records.append(record)
        else:
            enabled_models = self._training_sweep_derivative_enabled_models(stages)
            completed_entries = [
                entry
                for entry in list(summary.get("runs") or [])
                if isinstance(entry, dict) and str(entry.get("status") or "") == "completed"
            ]
            grouped_entries: dict[str, dict[str, Any]] = {}
            for entry in completed_entries:
                dataset_root_value = str(entry.get("dataset_root") or "").strip()
                key = dataset_root_value or f"__missing__::{entry.get('config_id') or entry.get('run_root') or len(grouped_entries)}"
                grouped_entries.setdefault(key, {"dataset_root": dataset_root_value, "entries": []})["entries"].append(entry)
            multiple_groups = len(grouped_entries) > 1
            for group in grouped_entries.values():
                entries = list(group.get("entries") or [])
                dataset_root_value = str(group.get("dataset_root") or "").strip()
                dataset_root = (
                    _resolve_optional_repo_path(dataset_root_value, Path(dataset_root_value))
                    if dataset_root_value
                    else None
                )
                group_label = self._training_sweep_derivative_group_label(entries, dataset_root=dataset_root)
                derivative_run_root = run_root / "sweep" / "derivative_workflows" / group_label
                record = {
                    "run_id": group_label,
                    "dataset_root": str(dataset_root) if dataset_root is not None else "",
                    "graph2mat_child_result_dir": "",
                    "deeph_child_result_dir": "",
                    "derivative_workflow_status": "pending",
                    "derivative_workflow_manifest_path": "",
                    "failure_reason": "",
                }
                try:
                    graph2mat_entry = self._select_training_sweep_derivative_entry(
                        entries,
                        dataset_root=dataset_root,
                        model="graph2mat",
                        required="graph2mat" in enabled_models,
                    )
                    deeph_entry = self._select_training_sweep_derivative_entry(
                        entries,
                        dataset_root=dataset_root,
                        model="deeph",
                        required="deeph" in enabled_models,
                    )
                    graph2mat_context, _ = self._training_sweep_derivative_contexts(graph2mat_entry or {})
                    _, deeph_context = self._training_sweep_derivative_contexts(deeph_entry or {})
                    child_payload = self._training_sweep_derivative_launch_payload(
                        payload,
                        derivative_run_root=derivative_run_root,
                        dataset_root=dataset_root,
                        graph2mat_context=graph2mat_context,
                        deeph_context=deeph_context,
                        multiple_groups=multiple_groups,
                        group_label=group_label,
                    )
                    register_job(
                        record,
                        child_payload=child_payload,
                        derivative_run_root=run_root,
                        graph2mat_context=graph2mat_context,
                        deeph_context=deeph_context,
                        success_fields={
                            "graph2mat_child_result_dir": str(graph2mat_entry.get("run_root") or "") if graph2mat_entry else "",
                            "deeph_child_result_dir": str(deeph_entry.get("run_root") or "") if deeph_entry else "",
                        },
                    )
                except Exception as exc:
                    record.update(
                        {
                            "derivative_workflow_status": "failed",
                            "failure_reason": str(exc),
                        }
                    )
                    records.append(record)

        seen_output_roots: dict[str, str] = {}
        for job in jobs:
            output_root = Path(job["output_root"]).resolve(strict=False)
            output_key = str(output_root)
            run_id = str(job["run_id"])
            if output_key in seen_output_roots:
                raise RuntimeError(
                    "training sweep derivative workflow output root collision: "
                    f"{output_root} is shared by {seen_output_roots[output_key]} and {run_id}."
                )
            seen_output_roots[output_key] = run_id

        initial_failed = len([record for record in records if record.get("derivative_workflow_status") == "failed"])
        set_derivative_status(
            total=len(records),
            queued=len(jobs),
            active=0,
            completed=0,
            failed=initial_failed,
        )

        def run_derivative_job(job: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            with self._lock:
                derivatives = dict(self._state.training_sweep_status.get("derivatives") or {})
                derivatives["queued"] = max(0, int(derivatives.get("queued") or 0) - 1)
                derivatives["active"] = int(derivatives.get("active") or 0) + 1
                self._state.training_sweep_status["derivatives"] = derivatives
            success = False
            try:
                derivative_summary = self._run_modular_derivative_workflow(
                    job["payload"],
                    run_root=job["derivative_run_root"],
                    graph2mat_context=job["graph2mat_context"],
                    deeph_context=job["deeph_context"],
                )
                manifest_path = Path(str(derivative_summary["result_dir"])) / "derivative_workflow_manifest.json"
                cost = derivative_summary.get("derivative_cost") if isinstance(derivative_summary.get("derivative_cost"), dict) else None
                success = True
                update = {
                    **dict(job.get("success_fields") or {}),
                    "derivative_workflow_status": "completed",
                    "derivative_workflow_manifest_path": str(manifest_path),
                    "failure_reason": "",
                }
                if cost is not None:
                    update["derivative_cost"] = cost
                    update.update(cost)
                return (int(job["index"]), update)
            except Exception as exc:
                return (
                    int(job["index"]),
                    {
                        "derivative_workflow_status": "failed",
                        "failure_reason": str(exc),
                    },
                )
            finally:
                with self._lock:
                    derivatives = dict(self._state.training_sweep_status.get("derivatives") or {})
                    derivatives["active"] = max(0, int(derivatives.get("active") or 0) - 1)
                    counter = "completed" if success else "failed"
                    derivatives[counter] = int(derivatives.get(counter) or 0) + 1
                    self._state.training_sweep_status["derivatives"] = derivatives

        if jobs and derivative_parallelism > 1 and len(jobs) > 1:
            self._logs.append(
                f"[G2M-DEEPH][PERF] Running {len(jobs)} derivative workflows with max_parallel_derivative_workflows={derivative_parallelism}.\n"
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(derivative_parallelism, len(jobs)),
                thread_name_prefix="g2m-deeph-derivatives",
            ) as executor:
                futures = {executor.submit(run_derivative_job, job): job for job in jobs}
                for future in concurrent.futures.as_completed(futures):
                    index, update = future.result()
                    records[index].update(update)
        else:
            for job in jobs:
                index, update = run_derivative_job(job)
                records[index].update(update)

        summary["derivative_workflows"] = records
        self._write_training_sweep_summary(run_root, summary)
        return records

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
        budget_fail_closed = _budget_accounting_fail_closed(payload, final_mode=final_mode)
        protocol_stage = protocol_stage_from_payload(payload, default=SEARCH_STAGE if final_mode else "exploratory")
        workflow = payload.get("modular_workflow") if isinstance(payload.get("modular_workflow"), dict) else {}
        stages = workflow.get("stages") if isinstance(workflow.get("stages"), dict) else {}
        run_root = Path(str(self._state.run_root or self._benchmark_run_root(payload, str(self._state.run_id))))
        resume_root, completed_by_key = self._load_resume_training_sweep(payload, run_root=run_root)
        budget_tracker = BudgetTracker(plan.get("budget_policy"))
        budget_warnings: list[str] = []
        graph2mat_parallelism = _graph2mat_training_parallelism(payload)
        deeph_parallelism = _deeph_training_parallelism(payload)
        model_batch_schedule = _model_batch_schedule(payload)
        if model_batch_schedule == "alternating":
            model_batch_start = _alternating_model_start(payload)
            planned = _alternating_model_batch_order(
                planned,
                graph2mat_parallelism=graph2mat_parallelism,
                deeph_parallelism=deeph_parallelism,
                start_model=model_batch_start,
            )
            plan = {**plan, "planned_runs": planned, "planned_run_count": len(planned)}

        def add_budget_completed(result: dict[str, Any], *, source: str = "completed") -> None:
            try:
                budget_tracker.add_completed(result, source=source)
            except RuntimeError as exc:
                if budget_fail_closed:
                    raise
                warning = str(exc)
                budget_warnings.append(warning)
                result["budget_accounting_status"] = "incomplete"
                result["budget_accounting_warning"] = warning
                warnings = result.setdefault("severe_warnings", [])
                if isinstance(warnings, list) and warning not in warnings:
                    warnings.append(warning)
                self._logs.append(
                    "[G2M-DEEPH][WARN] Budget accounting incomplete but non-final run "
                    f"continues: {warning}\n"
                )

        for completed in completed_by_key.values():
            add_budget_completed(completed, source="resume_manifest")
        summary: dict[str, Any] = {
            "schema": "graph2mat_deeph_training_sweep_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "running",
            "protocol_stage": protocol_stage,
            "final_benchmark_mode": final_mode,
            "budget_accounting_fail_closed": budget_fail_closed,
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
            "warnings": budget_warnings,
        }
        self._write_training_search_plan(run_root, plan)
        summary["budget"] = self._write_budget_summary(run_root, budget_tracker)
        self._set_stage("training_sweep")
        derivative_workflow_enabled = any(stages.get(stage) for stage in DERIVATIVE_STAGE_NAMES)
        derivative_workflow_parallelism = _derivative_workflow_parallelism(payload)
        with self._lock:
            self._state.training_sweep_status = {
                "enabled": True,
                "total": len(planned),
                "completed": len(completed_by_key),
                "failed": 0,
                "graph2mat_parallelism": graph2mat_parallelism,
                "deeph_parallelism": deeph_parallelism,
                "model_batch_schedule": model_batch_schedule,
                "active_model": None,
                "active_dataset": None,
                "active_config_id": None,
                "active_runs": [],
                "active_started_at": None,
                "derivatives": {
                    "enabled": derivative_workflow_enabled,
                    "total": 0,
                    "queued": 0,
                    "active": 0,
                    "completed": 0,
                    "failed": 0,
                    "max_parallel_derivative_workflows": derivative_workflow_parallelism,
                },
            }
        mixed_model_batches = model_batch_schedule == "mixed"
        summary["graph2mat_parallelism"] = graph2mat_parallelism
        summary["deeph_parallelism"] = deeph_parallelism
        summary["model_batch_schedule"] = model_batch_schedule
        summary["mixed_model_training_batches"] = mixed_model_batches
        self._logs.append(
            f"[G2M-DEEPH] Training sweep: {len(planned)} planned runs; "
            f"Graph2Mat parallel jobs={graph2mat_parallelism}; "
            f"DeepH parallel jobs={deeph_parallelism}; "
            f"model batch schedule={model_batch_schedule}.\n"
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
                add_budget_completed(result)
            else:
                budget_tracker.release(result)
            summary["runs"].append(result)
            summary["warnings"] = list(dict.fromkeys(budget_warnings))
            summary["budget"] = self._write_budget_summary(run_root, budget_tracker)
            with self._lock:
                self._state.training_sweep_status["completed"] += 1
            self._write_training_sweep_summary(run_root, summary)

        def record_failure(record: dict[str, Any], exc: Exception) -> None:
            budget_tracker.release(record)
            failed = {**record, "status": "failed", "error": str(exc)}
            command_record = getattr(exc, "run_record", None)
            if isinstance(command_record, dict):
                telemetry = command_record.get("telemetry") if isinstance(command_record.get("telemetry"), dict) else {}
                failed["failed_command"] = command_record
                failed["failure_category"] = telemetry.get("failure_category") or "unknown_failure"
                failed["failure_evidence_excerpt"] = telemetry.get("failure_evidence_excerpt") or ""
                failed["telemetry"] = telemetry
            else:
                failed["failure_category"] = "unknown_failure"
                failed["failure_evidence_excerpt"] = str(exc)
            summary["runs"].append(failed)
            summary["failed_runs"].append(failed)
            with self._lock:
                self._state.training_sweep_status["failed"] += 1
            self._logs.append(f"[G2M-DEEPH][WARN] Training sweep run failed: {record.get('config_id')}: {exc}\n")
            self._write_training_sweep_summary(run_root, summary)

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
            if mixed_model_batches and (graph2mat_parallelism > 1 or deeph_parallelism > 1):
                batch: list[dict[str, Any]] = []
                counts = {"graph2mat": 0, "deeph": 0}
                limits = {"graph2mat": graph2mat_parallelism, "deeph": deeph_parallelism}
                while index < len(planned):
                    candidate = planned[index]
                    candidate_key = "|".join(_training_record_key(candidate))
                    model = str(candidate.get("model") or "")
                    if model not in limits:
                        break
                    if candidate_key in completed_by_key:
                        self._logs.append(
                            f"[G2M-DEEPH] Skipping completed run from resume manifest: "
                            f"{candidate.get('dataset_id')} {candidate.get('config_id')}\n"
                        )
                        index += 1
                        continue
                    if counts[model] >= limits[model]:
                        break
                    if not budget_tracker.can_schedule(candidate):
                        record_budget_skip(candidate)
                        index += 1
                        continue
                    budget_tracker.reserve(candidate)
                    batch.append(candidate)
                    counts[model] += 1
                    index += 1
                    if counts["graph2mat"] >= graph2mat_parallelism and counts["deeph"] >= deeph_parallelism:
                        break
                if not batch:
                    continue
                with self._lock:
                    active_started_at = time.time()
                    self._state.training_sweep_status.update(
                        {
                            "active_model": "mixed_parallel" if len(batch) > 1 else batch[0].get("model"),
                            "active_dataset": ",".join(str(item.get("dataset_id")) for item in batch),
                            "active_config_id": ",".join(str(item.get("config_id")) for item in batch),
                            "active_started_at": active_started_at,
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
                    "[G2M-DEEPH][PERF] Running mixed Graph2Mat/DeepH sweep batch: "
                    + ", ".join(f"{item.get('model')}:{item.get('config_id')}" for item in batch)
                    + "\n"
                )
                first_error: Exception | None = None
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(batch),
                    thread_name_prefix="g2m-deeph-mixed",
                ) as executor:
                    futures = {}
                    for item in batch:
                        if item.get("model") == "graph2mat":
                            future = executor.submit(self._run_training_sweep_graph2mat_job, payload, validation, item)
                        elif item.get("model") == "deeph":
                            future = executor.submit(self._run_training_sweep_deeph_job, payload, validation, item)
                        else:
                            raise RuntimeError(f"Unsupported training sweep model: {item.get('model')}")
                        futures[future] = item
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
                    active_started_at = time.time()
                    self._state.training_sweep_status.update(
                        {
                            "active_model": "graph2mat_parallel" if len(batch) > 1 else "graph2mat",
                            "active_dataset": ",".join(str(item.get("dataset_id")) for item in batch),
                            "active_config_id": ",".join(str(item.get("config_id")) for item in batch),
                            "active_started_at": active_started_at,
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
                    active_started_at = time.time()
                    self._state.training_sweep_status.update(
                        {
                            "active_model": "deeph_parallel" if len(batch) > 1 else "deeph",
                            "active_dataset": ",".join(str(item.get("dataset_id")) for item in batch),
                            "active_config_id": ",".join(str(item.get("config_id")) for item in batch),
                            "active_started_at": active_started_at,
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
                active_started_at = time.time()
                self._state.training_sweep_status.update(
                    {
                        "active_model": record.get("model"),
                        "active_dataset": record.get("dataset_id"),
                        "active_config_id": record.get("config_id"),
                        "active_started_at": active_started_at,
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
                        "active_started_at": None,
                        "status": "stopped",
                    }
                )
            self._set_stage("complete")
            self._finish(returncode=130, error="Stop requested.")
            return
        summary["status"] = "completed" if not summary["failed_runs"] else "completed_with_failures"
        summary["budget"] = self._write_budget_summary(run_root, budget_tracker)
        self._write_training_sweep_summary(run_root, summary)
        derivative_workflows = self._run_training_sweep_derivative_workflows(
            payload,
            run_root=run_root,
            summary=summary,
        )
        derivative_failed = any(
            record.get("derivative_workflow_status") == "failed"
            for record in derivative_workflows
        )
        if derivative_failed:
            summary["status"] = "completed_with_derivative_failures"
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
                        "active_started_at": None,
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
            self._finish(returncode=0 if not summary["failed_runs"] and not derivative_failed else 1)
            return
        if derivative_failed:
            message = (
                "Training sweep completed, but one or more requested derivative workflows failed. "
                "See sweep/training_sweep_manifest.json derivative_workflows."
            )
            with self._lock:
                self._state.training_sweep_status.update(
                    {
                        "active_model": None,
                        "active_dataset": None,
                        "active_config_id": None,
                        "active_runs": [],
                        "active_started_at": None,
                        "status": summary["status"],
                    }
                )
                self._last_results = {
                    "dry_run": _parse_bool(payload.get("dry_run"), False),
                    "contract_name": CONTRACT_NAME,
                    "dataset_validation": validation,
                    "dataset_sweep": payload.get("_dataset_sweep_manifest"),
                    "training_sweep": summary,
                    "ranking": None,
                    "message": message,
                }
            self._finish(returncode=1, error=message)
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
                    "active_started_at": None,
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
            workflow = payload.get("modular_workflow") if isinstance(payload.get("modular_workflow"), dict) else {}
            stages = workflow.get("stages") if isinstance(workflow.get("stages"), dict) else {}
            derivative_workflow_enabled = any(stages.get(stage) for stage in DERIVATIVE_STAGE_NAMES)
            derivative_workflow_parallelism = _derivative_workflow_parallelism(payload)
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
                    "derivatives": {
                        "enabled": derivative_workflow_enabled,
                        "total": 0,
                        "queued": 0,
                        "active": 0,
                        "completed": 0,
                        "failed": 0,
                        "max_parallel_derivative_workflows": derivative_workflow_parallelism,
                    },
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
        payload["derivative_metrics"] = _normalized_derivative_metrics_payload(payload)
        payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)
        workflow = payload["modular_workflow"]
        workflow_stages = workflow["stages"]
        derivative_stages_requested = any(workflow_stages.get(stage) for stage in DERIVATIVE_STAGE_NAMES)
        hamiltonian_stages_requested = any(workflow_stages.get(stage) for stage in HAMILTONIAN_STAGE_NAMES)
        dataset_mode = str(payload.get("dataset_mode") or "reuse_validated").strip() or "reuse_validated"
        run_mode = str(payload.get("run_mode") or "").strip()
        precomputed_training_sweep_plan = payload.get("_training_sweep_plan")
        precomputed_training_sweep_requested = (
            isinstance(precomputed_training_sweep_plan, dict)
            and _parse_bool(precomputed_training_sweep_plan.get("enabled"), False)
        )
        training_sweep_requested = (
            _parse_bool((payload.get("training_sweep") or {}).get("enabled"), False)
            if isinstance(payload.get("training_sweep"), dict)
            else False
        ) or precomputed_training_sweep_requested
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
        derivative_only = derivative_stages_requested and not hamiltonian_stages_requested
        derivative_result_root = _resolve_optional_repo_path(
            workflow["derivative"].get("result_dir") or workflow["derivative"].get("output_root"),
            DEFAULT_OUTPUT_ROOT / "derivative_workflows",
        )
        validation = (
            {
                "contract_name": CONTRACT_NAME,
                "benchmark_ready": True,
                "repair_required": False,
                "dataset_root": str(
                    _resolve_optional_repo_path(
                        workflow["derivative"].get("source_dataset_root")
                        or workflow["derivative"].get("result_dir")
                        or workflow["derivative"].get("output_root"),
                        derivative_result_root,
                    )
                ),
                "snapshot_root": str(derivative_result_root),
                "artifact_summary": {
                    "total_snapshots": 0,
                    "valid_snapshots": 0,
                    "invalid_snapshots": 0,
                    "repair_required_snapshots": 0,
                    "missing_required_counts": {},
                },
                "errors": [],
                "warnings": ["derivative_only_workflow: skipped H benchmark dataset validation"],
                "manifest_paths": {},
                "dataset_sweep": dataset_sweep_info,
            }
            if derivative_only
            else
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
            if precomputed_training_sweep_requested:
                payload["_training_sweep_plan"] = precomputed_training_sweep_plan
            else:
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
                    else None
                    if derivative_only
                    else validation["manifest_paths"]["benchmark_dataset_manifest"]["path"]
                ),
                frozen_split_manifest_path=(
                    None
                    if generate_datasets_only or full_strict_pipeline or derivative_only
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
                target=self._run_derivative_only_workflow if derivative_only else self._run_workflow,
                args=(payload,) if derivative_only else (payload, validation, allow_repair),
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
            external_running = False
            external_run_root = None
            detached_running = False
            detached_run_root = None
            detached_status: dict[str, Any] = {}
            if not running and self._external_final_active():
                self._sync_external_final_run_locked()
                external_running = True
                external_run_root = self._external_final_run_root
            if not running and not external_running:
                detached_running = self._sync_detached_external_run_locked()
                detached_run_root = self._external_detached_run_root
                detached_status = dict(self._external_detached_status)
            self._state.running = running
            elapsed = None
            if self._state.started_at is not None:
                end = time.time() if running or external_running or detached_running else self._state.finished_at or time.time()
                elapsed = end - self._state.started_at
            if external_running:
                status_files = sorted(
                    path
                    for path in EXTERNAL_FINAL_LOG_ROOT.glob("paper_ready_final_70_*.status")
                    if ".postprocess." not in path.name and ".winner_selection." not in path.name
                )
                started_at = self._state.started_at
                status_values: dict[str, str] = {}
                if status_files:
                    try:
                        status_file = status_files[-1]
                        started_at = status_file.stat().st_mtime
                        for line in status_file.read_text(encoding="utf-8").splitlines():
                            if "=" in line:
                                key, value = line.split("=", 1)
                                status_values[key.strip()] = value.strip()
                    except OSError:
                        started_at = self._state.started_at
                graph2mat_parallelism = int(status_values.get("graph2mat_parallelism") or 8)
                deeph_parallelism = int(status_values.get("deeph_parallelism") or 6)
                return {
                    "running": True,
                    "stage": "external_paper_ready_final",
                    "returncode": None,
                    "started_at": started_at,
                    "finished_at": None,
                    "elapsed_seconds": None if started_at is None else time.time() - float(started_at),
                    "stop_requested": False,
                    "error": None,
                    "warnings": [
                        "attached_to_detached_paper_ready_final_run: stop desde UI no matara este proceso externo"
                    ],
                    "run_id": external_run_root.name if external_run_root is not None else "paper_ready_final70_external",
                    "dataset_root": self._state.dataset_root,
                    "run_root": str(external_run_root) if external_run_root is not None else "",
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
                    "active_processes": len(self._external_final_process_lines()),
                    "dataset_validation": self._dataset_validation,
                    "training_sweep": {
                        "enabled": True,
                        "status": "external_running",
                        "graph2mat_parallelism": graph2mat_parallelism,
                        "deeph_parallelism": deeph_parallelism,
                        "model_batch_schedule": status_values.get("model_batch_schedule", "alternating"),
                        "model_batch_start": status_values.get("model_batch_start", "deeph"),
                    },
                }
            if detached_running:
                detached_payload = dict(detached_status)
                detached_warnings = list(detached_payload.get("warnings") or [])
                detached_warnings.append(
                    "attached_to_detached_g2m_deeph_run: stop desde UI no matara este proceso externo"
                )
                detached_payload["warnings"] = detached_warnings
                detached_payload["run_id"] = detached_payload.get("run_id") or (
                    detached_run_root.name if detached_run_root is not None else "detached_g2m_deeph_run"
                )
                detached_payload["run_root"] = detached_payload.get("run_root") or (
                    str(detached_run_root) if detached_run_root is not None else ""
                )
                detached_payload["log_size"] = len(self._logs)
                return detached_payload
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
            if self._thread is None or not self._thread.is_alive():
                self._sync_external_final_run_locked()
                self._sync_detached_external_run_locked()
            payload = _bounded_log_payload(self._logs, since=since, limit=limit)
            payload["status"] = self.status()
            return payload

    def results(self) -> dict[str, Any]:
        with self._lock:
            # `results()` is used heavily by dry-run and summary tests; avoid
            # rescanning the whole archived results tree unless callers ask for
            # the broader plots view explicitly. Reuse the plot-entry ID so the
            # current run still matches the archive-row filter.
            selected_run_ids = (
                {self._plot_run_entry_locked(Path(str(self._state.run_root)))["id"]}
                if self._state.run_root
                else set()
            )
            plot_payload = self._build_plot_payload_locked(
                selected_run_ids=selected_run_ids if selected_run_ids else None,
                include_archive_roots=False,
            )
            return {
                "available": self._last_results is not None,
                "results": self._last_results,
                "plot_payload": plot_payload,
                "status": self.status(),
            }

    def plots(self, selected_run_ids: set[str] | None = None) -> dict[str, Any]:
        with self._lock:
            return self._build_plot_payload_locked(selected_run_ids=selected_run_ids)

    def plot_runs(self) -> dict[str, Any]:
        with self._lock:
            return self._plot_runs_payload_locked()

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
        run_id: str = "",
        run_root: str = "",
    ) -> None:
        try:
            elapsed = float(elapsed_seconds)
        except (TypeError, ValueError):
            return
        if elapsed < 0:
            return
        if elapsed == 0 and str(status).lower() != "running":
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
                "run_id": run_id,
                "run_root": run_root,
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
                        "run_root": str(run_root),
                    }
                )

    def _append_training_telemetry_metric_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        run_root: Path,
        record: dict[str, Any],
        dataset_size: int | None,
        source: str,
    ) -> None:
        try:
            size_value = int(dataset_size) if dataset_size is not None else None
        except (TypeError, ValueError):
            size_value = None
        if size_value is None or size_value <= 0:
            return
        telemetry = record.get("telemetry") if isinstance(record.get("telemetry"), dict) else {}
        common = record.get("common") if isinstance(record.get("common"), dict) else {}
        metric_values = {
            "validation_metric_value": record.get("validation_metric_value"),
            "gpu_hours_total": telemetry.get("gpu_hours_total"),
            "peak_gpu_memory_mb": telemetry.get("peak_gpu_memory_mb"),
            "peak_rss_mb": telemetry.get("peak_rss_mb"),
            "cpu_time_seconds_total": telemetry.get("cpu_time_seconds_total"),
            "samples_per_second": telemetry.get("samples_per_second"),
        }
        for metric_key, raw_value in metric_values.items():
            try:
                metric_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(metric_value):
                continue
            rows.append(
                {
                    "run_id": run_root.name,
                    "dataset_id": str(record.get("dataset_id") or ""),
                    "dataset_root": str(record.get("dataset_root") or ""),
                    "dataset_size": size_value,
                    "method": str(record.get("model") or ""),
                    "config_id": str(record.get("config_id") or ""),
                    "selected_config_id": str(record.get("selected_config_id") or ""),
                    "config_hash": str(record.get("config_hash") or ""),
                    "seed": record.get("final_seed") or record.get("seed") or common.get("seed"),
                    "epochs": _training_record_epochs(record),
                    "epoch_label": _training_record_epoch_label(record),
                    "metric_key": metric_key,
                    "metric_value": metric_value,
                    "metric_fail_policy": str(record.get("metric_fail_policy") or ""),
                    "scientific_status": str(record.get("protocol_stage") or "training_stage"),
                    "diagnostic_only": True,
                    "source": source,
                    "run_root": str(run_root),
                }
            )

    def _discover_plot_run_roots_locked(self) -> list[Path]:
        roots: dict[str, Path] = {}
        patterns = (
            "*/runs/*/runner_status.json",
            "*/runs/*/sweep/training_sweep_manifest.json",
            "*/runs/*/summary/ranking/normalized_run_metrics.json",
            "*/runs/*/common_metrics/summary/common_summary.json",
            "*/runner_status.json",
            "*/sweep/training_sweep_manifest.json",
            "*/summary/ranking/normalized_run_metrics.json",
            "*/common_metrics/summary/common_summary.json",
            "*/*/runner_status.json",
            "*/*/sweep/training_sweep_manifest.json",
            "*/*/summary/ranking/normalized_run_metrics.json",
            "*/*/common_metrics/summary/common_summary.json",
            "metric_archives_by_run/*/core/sweep/training_sweep_manifest.json",
            "metric_archives_by_run/*/core/summary/ranking/normalized_run_metrics.json",
            "metric_archives_by_run/*/core/common_metrics/summary/common_summary.json",
        )
        search_roots: list[Path] = []
        if self._state.run_root:
            # Prefer the active output tree when a run is in progress or has
            # just completed; it is the only archive the current session needs.
            search_roots.append(Path(str(self._state.run_root)).parent)
        else:
            search_roots.append(DEFAULT_OUTPUT_ROOT)
            # DEFAULT_OUTPUT_ROOT is a single fixed subfolder; completed sweeps
            # are commonly written directly under Comparison/results/<run_name>,
            # so also scan there to discover those when no run is active.
            results_root = REPO_ROOT / "Comparison" / "results"
            if results_root != DEFAULT_OUTPUT_ROOT:
                search_roots.append(results_root)
        if self._state.run_root:
            run_root = Path(str(self._state.run_root))
            roots[str(run_root.resolve())] = run_root
        if self._external_final_active():
            self._sync_external_final_run_locked()
        self._sync_detached_external_run_locked()
        if self._external_final_run_root is not None:
            run_root = self._external_final_run_root
            roots[str(run_root.resolve())] = run_root
            search_roots.append(run_root.parent)
        if self._external_detached_run_root is not None:
            run_root = self._external_detached_run_root
            roots[str(run_root.resolve())] = run_root
            search_roots.append(run_root.parent)
        for base in search_roots:
            if not base.exists():
                continue
            for pattern in patterns:
                for artifact in base.glob(pattern):
                    if artifact.name == "runner_status.json":
                        run_root = artifact.parent
                    elif artifact.name == "training_sweep_manifest.json":
                        run_root = artifact.parent.parent
                    else:
                        run_root = artifact.parents[2]
                    roots[str(run_root.resolve())] = run_root
        return sorted(roots.values(), key=lambda path: (path.stat().st_mtime if path.exists() else 0), reverse=True)

    def _plot_run_entry_locked(self, run_root: Path) -> dict[str, Any]:
        training_sweep = self._optional_json(str(run_root / "sweep" / "training_sweep_manifest.json"))
        normalized = self._optional_json(str(run_root / "summary" / "ranking" / "normalized_run_metrics.json"))
        common_summary = self._optional_json(str(run_root / "common_metrics" / "summary" / "common_summary.json"))
        runner_status_payload = self._optional_json(str(run_root / "runner_status.json"))
        runner_status = (
            runner_status_payload.get("status")
            if isinstance(runner_status_payload.get("status"), dict)
            else {}
        )
        runner_training_sweep = (
            runner_status.get("training_sweep")
            if isinstance(runner_status.get("training_sweep"), dict)
            else {}
        )
        rows = training_sweep.get("runs") if isinstance(training_sweep.get("runs"), list) else []
        planned = training_sweep.get("planned_runs") if isinstance(training_sweep.get("planned_runs"), list) else []
        partial_records: list[dict[str, Any]] = []
        row_keys = {
            (
                str(row.get("model") or ""),
                str(row.get("dataset_id") or ""),
                str(row.get("config_id") or ""),
            )
            for row in rows
            if isinstance(row, dict)
        }
        partial_manifest_patterns = (
            ("deeph", "sweep/deeph/*/*/deeph/deeph_manifest.json"),
            ("graph2mat", "sweep/graph2mat/*/*/graph2mat/graph2mat_manifest.json"),
        )
        for default_model, pattern in partial_manifest_patterns:
            for manifest_path in run_root.glob(pattern):
                manifest = self._optional_json(str(manifest_path))
                extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
                record = extra.get("sweep_record") if isinstance(extra.get("sweep_record"), dict) else {}
                if not record:
                    continue
                record = dict(record)
                record.setdefault("model", default_model)
                key = (
                    str(record.get("model") or ""),
                    str(record.get("dataset_id") or ""),
                    str(record.get("config_id") or ""),
                )
                if key in row_keys:
                    continue
                training_run = (
                    manifest.get("train_run")
                    if isinstance(manifest.get("train_run"), dict)
                    else extra.get("training_run")
                    if isinstance(extra.get("training_run"), dict)
                    else {}
                )
                if training_run.get("returncode") == 0:
                    record["status"] = "completed"
                    record["train_run"] = training_run
                elif manifest_path.parent.joinpath("train", "result.txt").exists():
                    record["status"] = "running"
                partial_records.append(record)
        active_runs = runner_training_sweep.get("active_runs") if isinstance(runner_training_sweep.get("active_runs"), list) else []
        dataset_ids = sorted(
            {
                str(row.get("dataset_id") or "")
                for row in [*rows, *planned, *partial_records, *active_runs]
                if isinstance(row, dict) and row.get("dataset_id")
            }
        )
        if not dataset_ids:
            dataset_ids = sorted(
                {
                    item.strip()
                    for item in str(runner_training_sweep.get("active_dataset") or "").split(",")
                    if item.strip()
                }
            )
        models = sorted(
            {
                str(row.get("model") or "")
                for row in [*rows, *planned, *partial_records, *active_runs]
                if isinstance(row, dict) and row.get("model")
            }
        )
        if not models:
            models = sorted(
                {
                    item.strip().removesuffix("_parallel")
                    for item in str(runner_training_sweep.get("active_model") or "").split(",")
                    if item.strip()
                }
            )
        completed = sum(1 for row in rows if isinstance(row, dict) and row.get("status") == "completed") + sum(
            1 for row in partial_records if isinstance(row, dict) and row.get("status") == "completed"
        )
        failed = sum(1 for row in rows if isinstance(row, dict) and row.get("status") == "failed") + sum(
            1 for row in partial_records if isinstance(row, dict) and row.get("status") == "failed"
        )
        if not completed:
            completed = int(runner_training_sweep.get("completed") or 0)
        if not failed:
            failed = int(runner_training_sweep.get("failed") or 0)
        planned_runs = len(planned) or len(partial_records) or int(runner_training_sweep.get("total") or 0)
        status = str(
            training_sweep.get("status")
            or common_summary.get("status")
            or normalized.get("status")
            or (runner_status.get("stage") if runner_status.get("running") else "")
            or ""
        )
        live_deeph_result_files = list(run_root.glob("sweep/deeph/*/*/deeph/train/result.txt"))
        if live_deeph_result_files and not status:
            status = "running"
        if live_deeph_result_files and "deeph" not in models:
            models.append("deeph")
        try:
            modified_at = run_root.stat().st_mtime
        except OSError:
            modified_at = None
        has_metric_rows = bool(normalized.get("rows") or common_summary.get("summary_rows")) or any(
            completed_metric_record(row) for row in rows if isinstance(row, dict)
        ) or any(
            isinstance(row, dict)
            and (
                row.get("validation_metric_value") is not None
                or isinstance(row.get("telemetry"), dict)
            )
            for row in rows
        ) or bool(live_deeph_result_files)
        return {
            "id": hashlib.sha256(str(run_root.resolve()).encode("utf-8")).hexdigest()[:16],
            "run_id": run_root.name,
            "run_root": str(run_root),
            "label": run_root.name,
            "status": status,
            "dataset_ids": dataset_ids,
            "models": models,
            "completed_runs": completed,
            "failed_runs": failed,
            "planned_runs": planned_runs,
            "has_training_sweep": bool(training_sweep),
            "has_metric_rows": has_metric_rows,
            "modified_at": modified_at,
        }

    def _append_live_deeph_training_loss_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        run_root: Path,
    ) -> None:
        number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        pattern = re.compile(
            r"Epoch\s+#(?P<epoch>\d+).*?"
            rf"Train loss:\s*(?P<train>{number_pattern}).*?"
            rf"Val loss:\s*(?P<val>{number_pattern}).*?"
            rf"Best val loss:\s*(?P<best>{number_pattern})"
        )
        for result_path in run_root.glob("sweep/deeph/*/*/deeph/train/result.txt"):
            try:
                text = result_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            match = matches[-1]
            manifest = self._optional_json(str(result_path.parents[1] / "deeph_manifest.json"))
            extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
            record = extra.get("sweep_record") if isinstance(extra.get("sweep_record"), dict) else {}
            context = manifest.get("context") if isinstance(manifest.get("context"), dict) else {}
            dataset_root = str(record.get("dataset_root") or context.get("dataset_root") or "")
            dataset_size = self._dataset_size_from_root(dataset_root)
            if dataset_size is None or dataset_size <= 0:
                continue
            metric_values = {
                "deeph_live_train_loss": match.group("train"),
                "deeph_live_val_loss": match.group("val"),
                "deeph_live_best_val_loss": match.group("best"),
            }
            for key, raw_value in metric_values.items():
                try:
                    metric_value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(metric_value):
                    continue
                rows.append(
                    {
                        "run_id": run_root.name,
                        "dataset_id": str(record.get("dataset_id") or ""),
                        "dataset_root": dataset_root,
                        "dataset_size": int(dataset_size),
                        "method": "deeph",
                        "config_id": str(record.get("config_id") or result_path.parents[2].name),
                        "selected_config_id": str(record.get("selected_config_id") or ""),
                        "config_hash": str(record.get("config_hash") or ""),
                        "seed": record.get("final_seed") or record.get("seed") or (record.get("common") or {}).get("seed"),
                        "epochs": int(match.group("epoch")),
                        "epoch_label": f"epoch {match.group('epoch')}",
                        "metric_key": key,
                        "metric_value": metric_value,
                        "scientific_status": "live_training_diagnostic",
                        "diagnostic_only": True,
                        "source": "deeph_live_result_txt",
                        "run_root": str(run_root),
                    }
                )

    def _plot_runs_payload_locked(self) -> dict[str, Any]:
        runs = [self._plot_run_entry_locked(run_root) for run_root in self._discover_plot_run_roots_locked()]
        return {
            "schema": "graph2mat_deeph_plot_runs_v1",
            "runs": runs,
            "default_selected_run_ids": [],
        }

    def _archive_plot_rows_locked(
        self,
        selected_run_ids: set[str] | None = None,
        *,
        include_archive_roots: bool = True,
    ) -> dict[str, list[dict[str, Any]]]:
        timing_scaling_rows: list[dict[str, Any]] = []
        metric_scaling_rows: list[dict[str, Any]] = []
        if selected_run_ids is not None and not selected_run_ids:
            return {
                "timing_scaling_rows": timing_scaling_rows,
                "metric_scaling_rows": metric_scaling_rows,
            }
        if include_archive_roots:
            run_roots = self._discover_plot_run_roots_locked()
        else:
            # Keep summary calls fast by limiting the scan to the active run and
            # the detached final run, if present.
            run_roots = []
            if self._state.run_root:
                run_roots.append(Path(str(self._state.run_root)))
            if self._external_final_run_root is not None:
                run_roots.append(self._external_final_run_root)
        for run_root in run_roots:
            entry = self._plot_run_entry_locked(run_root)
            if selected_run_ids is not None and entry["id"] not in selected_run_ids:
                continue
            run_id = run_root.name
            for metric_row in live_metric_scaling_rows(run_root):
                enriched_metric_row = dict(metric_row)
                enriched_metric_row.setdefault("run_id", run_id)
                enriched_metric_row.setdefault("run_root", str(run_root))
                metric_scaling_rows.append(enriched_metric_row)
            self._append_live_deeph_training_loss_rows(metric_scaling_rows, run_root=run_root)
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
                    run_id=run_id,
                    run_root=str(run_root),
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
                        run_id=run_id,
                        run_root=str(run_root),
                    )
                self._append_training_telemetry_metric_rows(
                    metric_scaling_rows,
                    run_root=run_root,
                    record=record,
                    dataset_size=dataset_size,
                    source="archived_training_sweep_telemetry",
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
                run_id=run_id,
                run_root=str(run_root),
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
                run_id=run_id,
                run_root=str(run_root),
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
                run_id=run_id,
                run_root=str(run_root),
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
                run_id=run_id,
                run_root=str(run_root),
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
                run_id=run_id,
                run_root=str(run_root),
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
        if not training_sweep.get("runs") and self._state.run_root:
            training_sweep = self._optional_json(
                str(Path(str(self._state.run_root)) / "sweep" / "training_sweep_manifest.json")
            )
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

    def _live_training_sweep_timing_rows_locked(self) -> list[dict[str, Any]]:
        """Expose currently running sweep jobs as timing rows for live plots."""

        status = self._state.training_sweep_status if isinstance(self._state.training_sweep_status, dict) else {}
        active_started_at = status.get("active_started_at")
        active_runs = status.get("active_runs") if isinstance(status.get("active_runs"), list) else []
        if not active_started_at or not active_runs:
            return []
        try:
            elapsed = max(0.0, time.time() - float(active_started_at))
        except (TypeError, ValueError):
            return []

        training_sweep = self._last_results.get("training_sweep") if isinstance(self._last_results, dict) else {}
        if not isinstance(training_sweep, dict):
            training_sweep = {}
        if not training_sweep.get("planned_runs") and self._state.run_root:
            training_sweep = self._optional_json(
                str(Path(str(self._state.run_root)) / "sweep" / "training_sweep_manifest.json")
            )
        planned_by_key = {
            "|".join(_training_record_key(record)): record
            for record in training_sweep.get("planned_runs") or []
            if isinstance(record, dict)
        }

        rows: list[dict[str, Any]] = []
        for active in active_runs:
            if not isinstance(active, dict):
                continue
            key = "|".join(
                (
                    str(active.get("model") or ""),
                    str(active.get("dataset_id") or ""),
                    str(active.get("config_id") or ""),
                )
            )
            record = dict(planned_by_key.get(key) or active)
            model = str(record.get("model") or active.get("model") or "")
            dataset_id = str(record.get("dataset_id") or active.get("dataset_id") or "")
            dataset_root = str(record.get("dataset_root") or self._state.dataset_root or "")
            dataset_size = self._dataset_size_from_root(dataset_root)
            if dataset_size is None:
                continue
            label_model = "DeepH" if model == "deeph" else "Graph2Mat" if model == "graph2mat" else model or "Sweep"
            self._append_timing_scaling_row(
                rows,
                dataset_id=dataset_id,
                dataset_root=dataset_root,
                dataset_size=dataset_size,
                phase=f"{model or 'sweep'}_active_job",
                label=f"{label_model} active job",
                elapsed_seconds=elapsed,
                source="live_training_sweep_status",
                model=model,
                config_id=str(record.get("config_id") or active.get("config_id") or ""),
                epochs=_training_record_epochs(record),
                epoch_label=_training_record_epoch_label(record),
                status="running",
            )
        return rows

    def _build_plot_payload_locked(
        self,
        selected_run_ids: set[str] | None = None,
        *,
        include_archive_roots: bool = True,
    ) -> dict[str, Any]:
        results = self._last_results or {}
        common_metrics = results.get("common_metrics") if isinstance(results, dict) else None
        artifact_summary = {}
        if self._dataset_validation:
            artifact_summary = dict(self._dataset_validation.get("artifact_summary") or {})
        archive_rows = self._archive_plot_rows_locked(
            selected_run_ids=selected_run_ids,
            include_archive_roots=include_archive_roots,
        )
        current_timing_scaling = [] if selected_run_ids is not None else self._timing_scaling_rows_locked()
        timing_scaling_rows = [
            *current_timing_scaling,
            *(archive_rows.get("timing_scaling_rows") or []),
        ]
        current_metric_scaling: list[dict[str, Any]] = []
        if selected_run_ids is None and self._state.run_root:
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
        payload["selected_run_ids"] = sorted(selected_run_ids) if selected_run_ids is not None else None
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
        payload["live_timing_rows"] = 0
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

    def _derivative_path(self, config: dict[str, Any], key: str) -> Path | None:
        value = config.get(key)
        if value in (None, ""):
            return None
        return _resolve_optional_repo_path(value, Path(str(value)))

    def _manifest_path_value(self, value: Any, *, base_dir: Path) -> Path | None:
        if value in (None, ""):
            return None
        path = Path(str(value))
        return path if path.is_absolute() else base_dir / path

    def _infer_graph2mat_derivative_checkpoint(self, context: Graph2MatBenchmarkContext | None) -> Path | None:
        if context is None:
            return None
        manifest_path = context.training_dir / "checkpoint_manifest.json"
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            for key in ("checkpoint_path", "path"):
                checkpoint = self._manifest_path_value(manifest.get(key), base_dir=context.training_dir)
                if checkpoint is not None and checkpoint.exists():
                    return checkpoint
        checkpoints = sorted(
            context.training_dir.rglob("*.ckpt"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
        )
        return checkpoints[-1] if checkpoints else None

    def _infer_deeph_derivative_model_dir(self, context: DeepHBenchmarkContext | None) -> Path | None:
        if context is None:
            return None
        return context.save_dir if context.save_dir.exists() else None

    def _graph2mat_derivative_artifact_inspection_paths(
        self,
        context: Graph2MatBenchmarkContext | None,
    ) -> list[str]:
        if context is None:
            return ["Graph2Mat H workflow context: <missing>"]
        manifest_path = context.training_dir / "checkpoint_manifest.json"
        inspected = [
            f"Graph2Mat checkpoint manifest: {manifest_path}",
            f"Graph2Mat checkpoint search root: {context.training_dir}",
        ]
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            for key in ("checkpoint_path", "path"):
                checkpoint = self._manifest_path_value(manifest.get(key), base_dir=context.training_dir)
                if checkpoint is not None:
                    inspected.append(f"Graph2Mat manifest field {key}: {checkpoint}")
        return inspected

    def _deeph_derivative_artifact_inspection_paths(
        self,
        context: DeepHBenchmarkContext | None,
    ) -> list[str]:
        if context is None:
            return ["DeepH H workflow context: <missing>"]
        return [
            f"DeepH manifest: {context.manifest_path}",
            f"DeepH save_dir: {context.save_dir}",
        ]

    def _format_derivative_artifact_inspection(
        self,
        model: str,
        *,
        graph2mat_context: Graph2MatBenchmarkContext | None,
        deeph_context: DeepHBenchmarkContext | None,
    ) -> str:
        if model == "graph2mat":
            inspected = self._graph2mat_derivative_artifact_inspection_paths(graph2mat_context)
        elif model == "deeph":
            inspected = self._deeph_derivative_artifact_inspection_paths(deeph_context)
        else:
            inspected = [f"Unsupported derivative model: {model}"]
        return "; inspected " + "; ".join(str(item) for item in inspected)

    def _graph2mat_context_from_manifest_path(self, manifest_path: Path | None) -> Graph2MatBenchmarkContext | None:
        if manifest_path is None or not manifest_path.exists():
            return None
        manifest = _load_json(manifest_path)
        raw = manifest.get("context") if isinstance(manifest.get("context"), dict) else {}
        if not raw:
            return None
        return Graph2MatBenchmarkContext(
            dataset_root=Path(str(raw["dataset_root"])),
            run_root=Path(str(raw["run_root"])),
            graph2mat_root=Path(str(raw["graph2mat_root"])),
            training_dir=Path(str(raw["training_dir"])),
            prediction_structs_dir=Path(str(raw["prediction_structs_dir"])),
            config_path=Path(str(raw["config_path"])),
            graph2mat_config_path=Path(str(raw["graph2mat_config_path"])),
            graph2mat_manifest_path=Path(str(raw["graph2mat_manifest_path"])),
            frozen_split_manifest_path=Path(str(raw["frozen_split_manifest_path"])),
            benchmark_dataset_manifest_path=Path(str(raw["benchmark_dataset_manifest_path"])),
            runs_json_path=Path(str(raw["runs_json_path"])),
            runs_json_counts=dict(raw.get("runs_json_counts") or {}),
            train_glob=str(raw.get("train_glob") or ""),
            validation_glob=str(raw.get("validation_glob") or ""),
            predict_glob=str(raw.get("predict_glob") or ""),
            output_file=str(raw.get("output_file") or "ML_prediction.HSX"),
            test_sample_ids=[str(item) for item in raw.get("test_sample_ids") or []],
            split_hash=raw.get("split_hash"),
            prediction_split=str(raw.get("prediction_split") or "test"),
            dry_run=_parse_bool(raw.get("dry_run"), False),
        )

    def _deeph_context_from_manifest_path(self, manifest_path: Path | None) -> DeepHBenchmarkContext | None:
        if manifest_path is None or not manifest_path.exists():
            return None
        manifest = _load_json(manifest_path)
        raw = manifest.get("context") if isinstance(manifest.get("context"), dict) else {}
        if not raw:
            return None
        return DeepHBenchmarkContext(
            root=Path(str(raw["root"])),
            raw_dir=Path(str(raw["raw_dir"])),
            processed_dir=Path(str(raw["processed_dir"])),
            graph_dir=Path(str(raw["graph_dir"])),
            save_dir=Path(str(raw["save_dir"])),
            inference_dir=Path(str(raw["inference_dir"])),
            preprocess_config=Path(str(raw["preprocess_config"])),
            train_config=Path(str(raw["train_config"])),
            inference_configs=[Path(str(path)) for path in raw.get("inference_configs") or []],
            inference_work_dirs=[Path(str(path)) for path in raw.get("inference_work_dirs") or []],
            manifest_path=Path(str(raw["manifest_path"])),
            deeph_discovery=manifest.get("deeph_discovery") if isinstance(manifest.get("deeph_discovery"), dict) else {},
            split_audit_path=Path(str(raw["split_audit_path"])),
            split_audit_csv_path=Path(str(raw["split_audit_csv_path"])),
            split_hash=raw.get("split_hash"),
            raw_mirror=raw.get("raw_mirror") if isinstance(raw.get("raw_mirror"), dict) else {},
            inference_split=str(raw.get("inference_split") or "test"),
            dry_run=_parse_bool(raw.get("dry_run"), False),
        )

    def _require_inferred_derivative_model_artifact(
        self,
        model: str,
        *,
        graph2mat_context: Graph2MatBenchmarkContext | None,
        deeph_context: DeepHBenchmarkContext | None,
    ) -> Path:
        if model == "graph2mat":
            checkpoint = self._infer_graph2mat_derivative_checkpoint(graph2mat_context)
            if checkpoint is None:
                raise RuntimeError(
                    "derivative stage 'predict_derivative_graph2mat' requires "
                    "derivative.graph2mat_checkpoint or derivative.graph2mat_existing_prediction_root; "
                    "no Graph2Mat checkpoint could be inferred from the completed H workflow context/manifest"
                    + self._format_derivative_artifact_inspection(
                        "graph2mat",
                        graph2mat_context=graph2mat_context,
                        deeph_context=deeph_context,
                    )
                    + "."
                )
            return checkpoint
        if model == "deeph":
            model_dir = self._infer_deeph_derivative_model_dir(deeph_context)
            if model_dir is None:
                raise RuntimeError(
                    "derivative stage 'predict_derivative_deeph' requires "
                    "derivative.deeph_model_dir or derivative.deeph_existing_prediction_root; "
                    "no DeepH model directory could be inferred from the completed H workflow context"
                    + self._format_derivative_artifact_inspection(
                        "deeph",
                        graph2mat_context=graph2mat_context,
                        deeph_context=deeph_context,
                    )
                    + "."
                )
            return model_dir
        raise RuntimeError(f"Unsupported derivative prediction model: {model}")

    def _preflight_derivative_workflow(
        self,
        payload: dict[str, Any],
        *,
        stages: dict[str, bool],
        config: dict[str, Any],
        common_root: Path,
        run_root: Path | None,
        graph2mat_context: Graph2MatBenchmarkContext | None,
        deeph_context: DeepHBenchmarkContext | None,
    ) -> None:
        def fail(stage: str, message: str) -> None:
            raise RuntimeError(f"derivative preflight for stage {stage!r} failed: {message}")

        build_stencils = bool(stages.get("build_derivative_stencils"))
        predict_graph2mat = bool(stages.get("predict_derivative_graph2mat"))
        predict_deeph = bool(stages.get("predict_derivative_deeph"))
        enabled_derivative_stages = [stage for stage in DERIVATIVE_STAGE_NAMES if stages.get(stage)]
        if not enabled_derivative_stages:
            return

        if build_stencils:
            source_dataset_root = self._derivative_path(config, "source_dataset_root")
            if source_dataset_root is None or not source_dataset_root.exists():
                fail(
                    "build_derivative_stencils",
                    "missing or nonexistent derivative.source_dataset_root; provide an existing source dataset root.",
                )
            if str(config.get("method") or "central") == "central" and not config.get("delta_ang_values"):
                fail(
                    "build_derivative_stencils",
                    "missing derivative.delta_ang; central finite differences require positive delta_ang values.",
                )
            if not config.get("atoms"):
                fail("build_derivative_stencils", "missing derivative.atoms; provide one or more atom indices.")
            if not config.get("axes"):
                fail("build_derivative_stencils", "missing derivative.axes; provide one or more Cartesian axes.")

        common_root_created_by_workflow = build_stencils
        needs_existing_common_root = any(
            stages.get(stage)
            for stage in (
                "validate_derivative_stencils",
                "run_derivative_siesta_reference",
                "predict_derivative_graph2mat",
                "predict_derivative_deeph",
            )
        ) and not common_root_created_by_workflow
        if needs_existing_common_root and not common_root.exists():
            fail(
                "derivative_artifact_root",
                "missing derivative.result_dir or derivative.output_root artifact root; "
                "provide an existing root or enable build_derivative_stencils.",
            )

        if stages.get("run_derivative_siesta_reference"):
            existing_reference_root = self._derivative_path(config, "existing_reference_root")
            if existing_reference_root is not None:
                if not existing_reference_root.exists():
                    fail(
                        "run_derivative_siesta_reference",
                        "derivative.existing_reference_root does not exist; provide an existing reference root.",
                    )

        for model, enabled in (("graph2mat", predict_graph2mat), ("deeph", predict_deeph)):
            if not enabled:
                continue
            existing_prediction_root = self._derivative_path(config, f"{model}_existing_prediction_root")
            if existing_prediction_root is not None:
                if not existing_prediction_root.exists():
                    fail(
                        f"predict_derivative_{model}",
                        f"derivative.{model}_existing_prediction_root does not exist; "
                        f"provide an existing prediction root or configure model artifacts.",
                    )
                continue
            if model == "graph2mat":
                checkpoint = self._derivative_path(config, "graph2mat_checkpoint")
                if checkpoint is not None:
                    if not checkpoint.exists():
                        fail(
                            "predict_derivative_graph2mat",
                            f"derivative.graph2mat_checkpoint does not exist: {checkpoint}; "
                            "alternatively provide derivative.graph2mat_existing_prediction_root "
                            "or run after an H workflow with an inferable Graph2Mat checkpoint.",
                        )
                elif self._infer_graph2mat_derivative_checkpoint(graph2mat_context) is None:
                    fail(
                        "predict_derivative_graph2mat",
                        "missing derivative.graph2mat_checkpoint; alternatively provide "
                        "derivative.graph2mat_existing_prediction_root or run after an H workflow "
                        "with an inferable Graph2Mat checkpoint"
                        + self._format_derivative_artifact_inspection(
                            "graph2mat",
                            graph2mat_context=graph2mat_context,
                            deeph_context=deeph_context,
                        )
                        + ".",
                    )
                if config.get("basis_files") in (None, ""):
                    source_dataset_root = self._derivative_path(config, "source_dataset_root")
                    example = (
                        str(source_dataset_root / "material_basis" / "*.ion.xml")
                        if source_dataset_root is not None
                        else "Comparison/datasets/<dataset>/material_basis/*.ion.xml"
                    )
                    fail(
                        "predict_derivative_graph2mat",
                        "missing derivative.basis_files; set derivative.basis_files to the Graph2Mat basis XML glob or file list, e.g. "
                        f"{example}.",
                    )
            else:
                model_dir = self._derivative_path(config, "deeph_model_dir")
                if model_dir is not None:
                    if not model_dir.exists():
                        fail(
                            "predict_derivative_deeph",
                            f"derivative.deeph_model_dir does not exist: {model_dir}; alternatively provide "
                            "derivative.deeph_existing_prediction_root.",
                        )
                elif self._infer_deeph_derivative_model_dir(deeph_context) is None:
                    fail(
                        "predict_derivative_deeph",
                        "missing derivative.deeph_model_dir; alternatively provide "
                        "derivative.deeph_existing_prediction_root or run after an H workflow "
                        "with an inferable DeepH save_dir"
                        + self._format_derivative_artifact_inspection(
                            "deeph",
                            graph2mat_context=graph2mat_context,
                            deeph_context=deeph_context,
                        )
                        + ".",
                    )
        metrics_created_by_predictions = predict_graph2mat or predict_deeph
        h_metric_fallback = run_root is not None and bool(stages.get("hamiltonian_metrics"))
        if not build_stencils and not metrics_created_by_predictions and not h_metric_fallback:
            for model in ("graph2mat", "deeph"):
                stage = f"derivative_metrics_{model}"
                if not stages.get(stage):
                    continue
                model_result_dir = self._derivative_path(config, f"{model}_result_dir")
                if model_result_dir is not None:
                    if not model_result_dir.exists():
                        fail(
                            stage,
                            f"derivative.{model}_result_dir does not exist; alternatively provide "
                            "derivative.result_dir or derivative.output_root.",
                        )
                    continue
                if not common_root.exists():
                    fail(
                        stage,
                        "missing derivative.result_dir or derivative.output_root; provide an existing artifact root "
                        f"or derivative.{model}_result_dir.",
                    )

    def _derivative_root(self, payload: dict[str, Any], run_root: Path | None = None) -> Path:
        config = payload["modular_workflow"]["derivative"]
        raw = config.get("result_dir") or config.get("output_root")
        if raw not in (None, ""):
            return _resolve_optional_repo_path(raw, Path(str(raw)))
        if run_root is not None:
            return run_root / "derivative_workflow"
        return DEFAULT_OUTPUT_ROOT / "derivative_workflows" / str(payload.get("run_id") or time.strftime("derivative_%Y%m%d_%H%M%S"))

    def _derivative_model_root(
        self,
        common_root: Path,
        model: str,
        config: dict[str, Any],
        *,
        separate_models: bool,
    ) -> Path:
        explicit = self._derivative_path(config, f"{model}_result_dir")
        if explicit is not None:
            return explicit
        return common_root / f"{model}_derivative_result" if separate_models else common_root

    def _materialize_derivative_model_root(self, common_root: Path, model_root: Path) -> None:
        if model_root.resolve(strict=False) == common_root.resolve(strict=False):
            return
        model_root.mkdir(parents=True, exist_ok=True)
        for dirname in ("structures", "siesta_hamiltonians"):
            source = common_root / dirname
            if not source.exists():
                continue
            target = model_root / dirname
            if target.exists():
                continue
            shutil.copytree(source, target)
        manifest = common_root / "derivative_stencil_manifest.json"
        if manifest.exists():
            target = model_root / "derivative_stencil_manifest.json"
            if not target.exists():
                _link_or_copy_file(manifest, target)

    def _check_derivative_manifest(self, path: Path, *, stage: str, fail_on_samples_failed: bool = False) -> dict[str, Any]:
        if not path.exists():
            raise RuntimeError(f"derivative stage {stage!r} did not produce expected manifest: {path}")
        manifest = _load_json(path)
        if fail_on_samples_failed and int(manifest.get("samples_failed") or 0) > 0:
            raise RuntimeError(f"derivative stage {stage!r} reported failed samples in {path}")
        return manifest

    def _run_derivative_stage_command(
        self,
        command: list[str],
        *,
        payload: dict[str, Any],
        label: str,
        allowed_returncodes: tuple[int, ...] = (0,),
    ) -> dict[str, Any]:
        return self._run_command(
            command,
            cwd=REPO_ROOT,
            env=self._python_command_env(self._graph2mat_python(payload), PYTHONUNBUFFERED="1"),
            label=label,
            allowed_returncodes=allowed_returncodes,
        )

    def _run_modular_derivative_workflow(
        self,
        payload: dict[str, Any],
        *,
        run_root: Path | None = None,
        graph2mat_context: Graph2MatBenchmarkContext | None = None,
        deeph_context: DeepHBenchmarkContext | None = None,
    ) -> dict[str, Any]:
        workflow = payload["modular_workflow"]
        stages = workflow["stages"]
        config = workflow["derivative"]
        common_root = self._derivative_root(payload, run_root=run_root)
        method = str(config.get("method") or "central")
        split = str(config.get("base_split") or "test")
        overwrite = _parse_bool(config.get("overwrite"), False)
        skip_if_exists = _parse_bool(config.get("skip_if_exists"), True)
        diagnostic_only = _parse_bool(config.get("diagnostic_only"), False)
        reference_workers = _derivative_reference_workers(payload, config)
        python_executable = self._graph2mat_python(payload)
        summary: dict[str, Any] = {
            "schema": "graph2mat_deeph_modular_derivative_workflow_v1",
            "workflow_mode": workflow.get("workflow_mode"),
            "result_dir": str(common_root),
            "stages": {},
        }
        self._preflight_derivative_workflow(
            payload,
            stages=stages,
            config=config,
            common_root=common_root,
            run_root=run_root,
            graph2mat_context=graph2mat_context,
            deeph_context=deeph_context,
        )

        def mark(stage: str, record: dict[str, Any]) -> None:
            summary["stages"][stage] = _json_safe_payload(record)

        if stages.get("build_derivative_stencils"):
            manifest_path = common_root / "derivative_stencil_manifest.json"
            if skip_if_exists and not overwrite and manifest_path.exists():
                manifest = self._check_derivative_manifest(manifest_path, stage="build_derivative_stencils")
                cost = _derivative_cost_summary(
                    manifest=manifest,
                    config=config,
                    reference_workers=reference_workers,
                )
                summary["derivative_cost"] = cost
                summary["derivative_cost_by_dataset"] = [cost]
                mark("build_derivative_stencils", {"status": "skipped_existing", "manifest": str(manifest_path), "cost": cost})
            else:
                command = [
                    python_executable,
                    str(DEFAULT_DERIVATIVE_STENCIL_SCRIPT),
                    "--source-dataset-root",
                    str(self._derivative_path(config, "source_dataset_root")),
                    "--output-stencil-root",
                    str(common_root),
                    "--method",
                    method,
                    "--delta-ang",
                    *[str(delta) for delta in config.get("delta_ang_values") or []],
                    "--split",
                    split,
                    "--atoms",
                    ",".join(str(item) for item in config.get("atoms") or []),
                    "--axes",
                    ",".join(str(item) for item in config.get("axes") or []),
                ]
                frozen_split = self._derivative_path(config, "frozen_split")
                if frozen_split is not None:
                    command.extend(["--frozen-split", str(frozen_split)])
                raw_sample_ids = config.get("base_sample_ids") or config.get("base_sample_id") or []
                if isinstance(raw_sample_ids, str):
                    raw_sample_ids = [item.strip() for item in raw_sample_ids.split(",") if item.strip()]
                for sample_id in raw_sample_ids:
                    command.extend(["--base-sample-id", str(sample_id)])
                command.extend(["--base-selection-policy", str(config.get("base_selection_policy") or "all")])
                if config.get("max_base_snapshots") is not None:
                    command.extend(["--max-base-snapshots", str(config["max_base_snapshots"])])
                if config.get("min_base_snapshots") is not None:
                    command.extend(["--min-base-snapshots", str(config["min_base_snapshots"])])
                if config.get("base_fraction") is not None:
                    command.extend(["--base-fraction", str(config["base_fraction"])])
                if config.get("base_selection_seed") is not None:
                    command.extend(["--base-selection-seed", str(config["base_selection_seed"])])
                include_base = config.get("include_base")
                if include_base is not None:
                    command.append("--include-base" if _parse_bool(include_base, True) else "--no-include-base")
                elif method == "central" and stages.get("validate_derivative_stencils"):
                    command.append("--include-base")
                if overwrite:
                    command.append("--overwrite")
                record = self._run_derivative_stage_command(command, payload=payload, label="Derivative stencil builder")
                manifest = self._check_derivative_manifest(manifest_path, stage="build_derivative_stencils")
                cost = _derivative_cost_summary(
                    manifest=manifest,
                    config=config,
                    reference_workers=reference_workers,
                )
                summary["derivative_cost"] = cost
                summary["derivative_cost_by_dataset"] = [cost]
                mark("build_derivative_stencils", {**record, "status": "completed", "manifest": manifest, "cost": cost})

        if stages.get("validate_derivative_stencils"):
            output_json = common_root / "derivative_geometry_validation.json"
            if skip_if_exists and not overwrite and output_json.exists():
                validation_summary = _load_json(output_json)
                if int(validation_summary.get("errors") or 0) > 0 and not diagnostic_only:
                    raise RuntimeError(f"derivative stage 'validate_derivative_stencils' found geometry errors in {output_json}")
                mark("validate_derivative_stencils", {"status": "skipped_existing", "manifest": str(output_json)})
            else:
                command = [
                    python_executable,
                    str(DEFAULT_DERIVATIVE_GEOMETRY_VALIDATION_SCRIPT),
                    str(common_root),
                    "--output-dir",
                    str(common_root),
                    "--method",
                    method,
                    "--split",
                    split,
                ]
                if method == "central" or _parse_bool(config.get("require_central"), method == "central"):
                    command.append("--require-central")
                tolerance = config.get("tolerance_ang")
                if tolerance not in (None, ""):
                    command.extend(["--tolerance-ang", str(tolerance)])
                record = self._run_derivative_stage_command(
                    command,
                    payload=payload,
                    label="Derivative stencil geometry validation",
                    allowed_returncodes=(0, 1) if diagnostic_only else (0,),
                )
                validation_summary = self._check_derivative_manifest(output_json, stage="validate_derivative_stencils")
                if int(validation_summary.get("errors") or 0) > 0 and not diagnostic_only:
                    raise RuntimeError(f"derivative stage 'validate_derivative_stencils' found geometry errors in {output_json}")
                mark("validate_derivative_stencils", {**record, "status": "completed", "manifest": validation_summary})

        if stages.get("run_derivative_siesta_reference"):
            reference_root = self._derivative_path(config, "reference_root") or (common_root / "siesta_hamiltonians")
            manifest_path = reference_root / "derivative_siesta_reference_manifest.json"
            if skip_if_exists and not overwrite and manifest_path.exists():
                self._check_derivative_manifest(manifest_path, stage="run_derivative_siesta_reference", fail_on_samples_failed=not diagnostic_only)
                mark(
                    "run_derivative_siesta_reference",
                    {
                        "status": "skipped_existing",
                        "manifest": str(manifest_path),
                        "reference_workers": reference_workers,
                    },
                )
            else:
                command = [
                    python_executable,
                    str(DEFAULT_DERIVATIVE_SIESTA_REFERENCE_SCRIPT),
                    "--stencil-root",
                    str(common_root),
                    "--output-reference-root",
                    str(reference_root),
                    "--siesta-command",
                    str(config.get("siesta_command") or "siesta"),
                ]
                source_dataset_root = self._derivative_path(config, "source_dataset_root")
                if source_dataset_root is not None:
                    command.extend(["--source-dataset-root", str(source_dataset_root)])
                existing_reference_root = self._derivative_path(config, "existing_reference_root")
                if existing_reference_root is not None:
                    command.extend(["--existing-reference-root", str(existing_reference_root)])
                if overwrite:
                    command.append("--overwrite")
                command.append("--skip-if-exists" if skip_if_exists else "--no-skip-if-exists")
                if diagnostic_only:
                    command.append("--diagnostic-only")
                max_samples = config.get("max_samples") if config.get("max_samples") not in (None, "") else config.get("max_jobs")
                if max_samples not in (None, ""):
                    command.extend(["--max-samples", str(max_samples)])
                command.extend(["--workers", str(reference_workers)])
                if _parse_bool(config.get("siesta_shell"), False):
                    command.append("--siesta-shell")
                record = self._run_derivative_stage_command(
                    command,
                    payload=payload,
                    label="Derivative SIESTA reference Hamiltonians",
                    allowed_returncodes=(0, 2) if diagnostic_only else (0,),
                )
                manifest = self._check_derivative_manifest(
                    manifest_path,
                    stage="run_derivative_siesta_reference",
                    fail_on_samples_failed=not diagnostic_only,
                )
                mark(
                    "run_derivative_siesta_reference",
                    {**record, "status": "completed", "manifest": manifest, "reference_workers": reference_workers},
                )

        prediction_models = [model for model in ("graph2mat", "deeph") if stages.get(f"predict_derivative_{model}") or stages.get(f"derivative_metrics_{model}")]
        separate_models = len(prediction_models) > 1 and any(stages.get(f"predict_derivative_{model}") for model in prediction_models)
        model_roots = {
            model: self._derivative_model_root(common_root, model, config, separate_models=separate_models)
            for model in ("graph2mat", "deeph")
        }

        for model in ("graph2mat", "deeph"):
            if not stages.get(f"predict_derivative_{model}"):
                continue
            model_root = model_roots[model]
            self._materialize_derivative_model_root(common_root, model_root)
            output_root = self._derivative_path(config, f"{model}_prediction_root") or (model_root / "predicted_hamiltonians")
            manifest_path = output_root / f"derivative_{model}_prediction_manifest.json"
            if skip_if_exists and not overwrite and manifest_path.exists():
                self._check_derivative_manifest(manifest_path, stage=f"predict_derivative_{model}", fail_on_samples_failed=not diagnostic_only)
                mark(f"predict_derivative_{model}", {"status": "skipped_existing", "manifest": str(manifest_path)})
                continue
            command = [
                python_executable,
                str(DEFAULT_DERIVATIVE_PREDICTION_SCRIPT),
                "--stencil-root",
                str(model_root),
                "--model",
                model,
                "--output-root",
                str(output_root),
            ]
            model_artifact: dict[str, Any] = {"model": model}
            existing_prediction_root = self._derivative_path(config, f"{model}_existing_prediction_root")
            explicit_checkpoint = self._derivative_path(config, f"{model}_checkpoint")
            explicit_model_dir = self._derivative_path(config, f"{model}_model_dir")
            if existing_prediction_root is not None:
                command.extend(["--existing-prediction-root", str(existing_prediction_root)])
                model_artifact.update(
                    {
                        "source": "configured_existing_prediction_root",
                        "existing_prediction_root": str(existing_prediction_root),
                    }
                )
                if explicit_checkpoint is not None:
                    command.extend(["--checkpoint", str(explicit_checkpoint)])
                if explicit_model_dir is not None:
                    command.extend(["--model-dir", str(explicit_model_dir)])
            else:
                if model == "graph2mat":
                    checkpoint = explicit_checkpoint
                    source = "configured_graph2mat_checkpoint"
                    if checkpoint is None:
                        checkpoint = self._require_inferred_derivative_model_artifact(
                            model,
                            graph2mat_context=graph2mat_context,
                            deeph_context=deeph_context,
                        )
                        source = "inferred_h_workflow_checkpoint"
                    command.extend(["--checkpoint", str(checkpoint)])
                    model_artifact.update({"source": source, "checkpoint": str(checkpoint)})
                    if explicit_model_dir is not None:
                        command.extend(["--model-dir", str(explicit_model_dir)])
                else:
                    if explicit_checkpoint is not None:
                        command.extend(["--checkpoint", str(explicit_checkpoint)])
                    model_dir = explicit_model_dir
                    source = "configured_deeph_model_dir"
                    if model_dir is None:
                        model_dir = self._require_inferred_derivative_model_artifact(
                            model,
                            graph2mat_context=graph2mat_context,
                            deeph_context=deeph_context,
                        )
                        source = "inferred_h_workflow_model_dir"
                    command.extend(["--model-dir", str(model_dir)])
                    model_artifact.update({"source": source, "model_dir": str(model_dir)})
            if overwrite:
                command.append("--overwrite")
            command.append("--skip-if-exists" if skip_if_exists else "--no-skip-if-exists")
            if diagnostic_only:
                command.append("--diagnostic-only")
            max_samples = config.get("max_samples") if config.get("max_samples") not in (None, "") else config.get("max_jobs")
            if max_samples not in (None, ""):
                command.extend(["--max-samples", str(max_samples)])
            if model == "graph2mat":
                command.extend(["--python-executable", python_executable])
                for key, flag in (
                    ("basis_files", "--basis-files"),
                    ("accelerator", "--accelerator"),
                    ("matrix_component_policy", "--matrix-component-policy"),
                    ("n_matrix_components", "--n-matrix-components"),
                    ("loader_threads", "--loader-threads"),
                ):
                    if config.get(key) not in (None, ""):
                        command.extend([flag, str(config[key])])
            else:
                deeph_command = config.get("deeph_command")
                if deeph_command in (None, ""):
                    deeph_command = self._deeph_command(payload, "deeph-inference")
                if deeph_command not in (None, ""):
                    command.extend(["--deeph-command", str(deeph_command)])
                if _parse_bool(config.get("deeph_shell"), False):
                    command.append("--deeph-shell")
            record = self._run_derivative_stage_command(
                command,
                payload=payload,
                label=f"Derivative {model} Hamiltonian predictions",
                allowed_returncodes=(0, 2) if diagnostic_only else (0,),
            )
            manifest = self._check_derivative_manifest(
                manifest_path,
                stage=f"predict_derivative_{model}",
                fail_on_samples_failed=not diagnostic_only,
            )
            mark(
                f"predict_derivative_{model}",
                {**record, "status": "completed", "manifest": manifest, "model_artifact": model_artifact},
            )

        metric_roots: dict[str, Path | None] = {"graph2mat": None, "deeph": None}
        for model in ("graph2mat", "deeph"):
            if not stages.get(f"derivative_metrics_{model}"):
                continue
            result_dir = self._derivative_path(config, f"{model}_result_dir")
            if result_dir is None and config.get("result_dir") in (None, "") and config.get("output_root") in (None, "") and run_root is not None:
                result_dir = run_root / "common_metrics" / f"{model}_eval"
            if result_dir is None:
                result_dir = model_roots[model]
            output_dir = self._derivative_path(config, f"{model}_metrics_output_dir") or (common_root / "derivative_metrics" / model)
            manifest_path = output_dir / "manifest.json"
            settings = {
                "finite_difference_method": method,
                "split": split,
                "require_central": _parse_bool(config.get("require_central"), method == "central"),
                "diagnostic_only": diagnostic_only,
                "overwrite": overwrite,
                "support_threshold": float(config.get("support_threshold", 1e-12) or 1e-12),
                "max_stencils": _optional_int_value(config.get("max_stencils")),
            }
            if skip_if_exists and not overwrite and manifest_path.exists():
                metric_roots[model] = output_dir
                mark(f"derivative_metrics_{model}", {"status": "skipped_existing", "manifest": str(manifest_path)})
                continue
            command = _derivative_metric_command_args(
                python_executable=python_executable,
                result_dir=result_dir,
                output_dir=output_dir,
                source_model=model,
                settings=settings,
            )
            record = self._run_derivative_stage_command(
                command,
                payload=payload,
                label=f"Derivative {model} finite-difference metrics",
                allowed_returncodes=(0, 2) if diagnostic_only else (0,),
            )
            manifest = self._check_derivative_manifest(manifest_path, stage=f"derivative_metrics_{model}")
            metric_roots[model] = output_dir
            mark(f"derivative_metrics_{model}", {**record, "status": "completed", "manifest": manifest})

        gate_report: dict[str, Any] | None = None
        if stages.get("derivative_plots"):
            roots = [root for root in metric_roots.values() if root is not None]
            output_dir = self._derivative_path(config, "plots_output_dir") or (common_root / "derivative_metrics" / "summary" / "derivative_plots")
            if not roots:
                raise RuntimeError("derivative stage 'derivative_plots' requires at least one completed derivative metrics output.")
            plot_result = write_derivative_plot_outputs(
                derivative_roots=roots,
                graph2mat_root=metric_roots["graph2mat"],
                deeph_root=metric_roots["deeph"],
                output_dir=output_dir,
            )
            mark("derivative_plots", {"status": "completed", "outputs": plot_result})

        if stages.get("derivative_gate_check"):
            roots = [root for root in metric_roots.values() if root is not None]
            if not roots:
                raise RuntimeError("derivative stage 'derivative_gate_check' requires at least one completed derivative metrics output.")
            gate_report = build_derivative_gate_report(derivative_roots=roots, run_root=run_root or common_root)
            output_path = self._derivative_path(config, "gate_report_path") or (common_root / "derivative_metrics" / "summary" / "derivative_gate_report.json")
            _write_json(output_path, gate_report)
            mark("derivative_gate_check", {"status": "completed", "output_path": str(output_path), "report": gate_report})

        if stages.get("derivative_model_comparison", True) and (
            metric_roots["graph2mat"] is not None or metric_roots["deeph"] is not None
        ):
            comparison = build_derivative_model_comparison_summary(
                graph2mat_root=metric_roots["graph2mat"],
                deeph_root=metric_roots["deeph"],
                output_dir=common_root / "derivative_metrics" / "summary" / "derivative_model_comparison",
                gate_report=gate_report,
            )
            mark("derivative_model_comparison", {"status": "completed", "summary": comparison})

        _write_json(common_root / "derivative_workflow_manifest.json", summary)
        return summary

    def _run_derivative_only_workflow(self, payload: dict[str, Any]) -> None:
        try:
            self._set_stage("derivative_workflow")
            summary = self._run_modular_derivative_workflow(payload)
            with self._lock:
                self._last_results = {
                    "dry_run": False,
                    "contract_name": CONTRACT_NAME,
                    "dataset_validation": self._dataset_validation,
                    "derivative_workflow": summary,
                    "phase_timings": list(self._phase_timings),
                    "message": "Derivative workflow completed from configured stencil/Hamiltonian artifacts.",
                }
            self._finish(returncode=0)
        except Exception as exc:
            self._finish(returncode=1, error=str(exc))

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
        derivative_workflow_summary: dict[str, Any] | None = None
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
                        derivative_settings = _derivative_metrics_settings(payload)
                        derivative_summary = run_derivative_postprocess(
                            run_root=context.run_root,
                            graph2mat_result_dir=staged_graph2mat.result_dir,
                            deeph_result_dir=common_root / "deeph_eval",
                            settings=derivative_settings,
                            overwrite=True,
                            diagnostic_only=bool(derivative_settings.get("diagnostic_only", True)),
                            python_executable=self._graph2mat_python(payload),
                            command_runner=self._run_command,
                            log=self._logs.append,
                        )
                        graph2mat_derivative_root = (
                            Path(str(derivative_summary["roots"]["graph2mat"]))
                            if derivative_summary["roots"]["graph2mat"]
                            else None
                        )
                        deeph_derivative_root = (
                            Path(str(derivative_summary["roots"]["deeph"]))
                            if derivative_summary["roots"]["deeph"]
                            else None
                        )
                        common_metrics_manifest = aggregate_common_metrics(
                            graph2mat_metrics_root=staged_graph2mat.result_dir / "metrics",
                            deeph_metrics_root=common_root / "deeph_eval" / "metrics",
                            output_dir=common_root / "summary",
                            frozen_split_manifest_path=context.frozen_split_manifest_path,
                            dataset_manifest_path=context.benchmark_dataset_manifest_path,
                            graph2mat_derivative_root=graph2mat_derivative_root,
                            deeph_derivative_root=deeph_derivative_root,
                        )
                        common_metrics_manifest["runs"] = {
                            "graph2mat_eval": graph2mat_eval_run,
                            "deeph_eval": deeph_eval_run,
                            "derivative_metrics": derivative_summary.get("execution", {}),
                        }
                        common_metrics_manifest.setdefault("derivative_metrics", {})
                        common_metrics_manifest["derivative_metrics"].update(
                            {
                                "enabled": bool(derivative_summary.get("enabled")),
                                "settings": derivative_summary.get("settings") or {},
                                "execution": derivative_summary.get("execution") or {},
                                "plot_outputs": derivative_summary.get("plot_outputs") or {},
                                "gate_report": derivative_summary.get("gate_report") or {},
                                "model_comparison": derivative_summary.get("model_comparison") or {},
                                "roots": derivative_summary.get("roots") or {},
                            }
                        )
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
            workflow = payload.get("modular_workflow") or {}
            workflow_stages = workflow.get("stages") if isinstance(workflow.get("stages"), dict) else {}
            if context is not None and any(workflow_stages.get(stage) for stage in DERIVATIVE_STAGE_NAMES):
                self._set_stage("derivative_workflow")
                derivative_workflow_summary = self._run_modular_derivative_workflow(
                    payload,
                    run_root=context.run_root,
                    graph2mat_context=context,
                    deeph_context=deeph_context,
                )
            with self._lock:
                self._last_results = {
                    "dry_run": bool(context.dry_run) if context is not None else None,
                    "contract_name": CONTRACT_NAME,
                    "dataset_validation": validation,
                    "graph2mat": context.to_dict() if context is not None else None,
                    "deeph": deeph_context.to_dict() if deeph_context is not None else None,
                    "common_metrics": common_metrics_manifest,
                    "ranking": ranking_manifest,
                    "derivative_workflow": derivative_workflow_summary,
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


def _cli_derivative_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": True,
        "finite_difference_method": str(args.method or "central"),
        "split": str(args.split or "test"),
        "require_central": bool(args.require_central),
        "diagnostic_only": bool(args.diagnostic_only),
        "support_threshold": float(args.support_threshold),
        "max_stencils": None,
        "overwrite": bool(args.overwrite),
    }


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill-derivatives-from-training-sweep", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--diagnostic-only", action="store_true", default=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", default="central")
    parser.add_argument("--require-central", action="store_true", default=True)
    parser.add_argument("--support-threshold", type=float, default=1e-12)
    return parser.parse_args()


def main() -> int:
    args = _parse_cli_args()
    training_sweep_manifest_path = args.backfill_derivatives_from_training_sweep
    if training_sweep_manifest_path is None and args.run_root is not None:
        training_sweep_manifest_path = args.run_root / "sweep" / "training_sweep_manifest.json"
    if training_sweep_manifest_path is None:
        raise SystemExit(
            "Provide --backfill-derivatives-from-training-sweep <training_sweep_manifest.json> "
            "or --run-root <benchmark_run_root>."
        )
    summary = backfill_derivative_postprocess_from_training_sweep(
        training_sweep_manifest_path=training_sweep_manifest_path,
        settings=_cli_derivative_settings(args),
        python_executable=str(args.python_executable),
        log=lambda message: sys.stdout.write(message),
    )
    sys.stdout.write(json.dumps(_json_safe_payload(summary), indent=2, ensure_ascii=False) + "\n")
    return 1 if summary.get("status") == "completed_with_failures" else 0


if __name__ == "__main__":
    raise SystemExit(main())
