#!/usr/bin/env python3
"""Local web UI and API for running MD and AtomDisplacement together."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import hashlib
import itertools
import math
import os
import json
import platform
import re
import select
import secrets
import shutil
import shlex
import socket
import string
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import yaml
from siesta_settings import DEFAULT_SHARED, compare_method_settings, file_digest
from model_settings import compare_method_model_settings
from reference_selection import choose_reference_matrix as strict_choose_reference_matrix
from reference_selection import reference_candidates
from cleanup_generated_datasets import (
    cleanup_generated_datasets,
    cleanup_selected_generated_datasets,
    generated_dataset_records,
)
from method_registry import (
    METHOD_REGISTRY as SCIENTIFIC_METHOD_REGISTRY,
    normalize_method_id,
    normalize_test_set_id,
)
from material_provenance import (
    MATERIAL_FLAT_FIELDS,
    MATERIAL_MAP_FIELDS,
    flatten_material_provenance,
    material_compatibility_warning,
    material_maps_from_manifest,
    read_json_file,
)
from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner
from plot_hamiltonian_derivative_metrics import build_derivative_plot_payload

try:
    import pty
except Exception:  # pragma: no cover - Windows fallback.
    pty = None

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).resolve().parents[1] / "ui"
COMPARISON_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = COMPARISON_ROOT / "results"
DEFAULT_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DATASET_SIZE_MINIMUM_UI_ARTIFACTS = {
    "dataset_size_minimum_cost_efficiency.png": {
        "label": "Cost efficiency plot",
        "kind": "image",
        "mime_type": "image/png",
    },
    "dataset_size_minimum_cost_efficiency.pdf": {
        "label": "Cost efficiency PDF",
        "kind": "document",
        "mime_type": "application/pdf",
    },
    "dataset_size_minimum_primary_metric.pdf": {
        "label": "Primary metric PDF",
        "kind": "document",
        "mime_type": "application/pdf",
    },
}


def postprocess_python_executable() -> str:
    """Prefer the repo venv so postprocess scripts can import matplotlib and peers."""
    if DEFAULT_VENV_PYTHON.exists():
        return str(DEFAULT_VENV_PYTHON)
    return sys.executable
WORKSPACES_ROOT = COMPARISON_ROOT / "workspaces"
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
from material_bundle import MaterialBundleError, extract_coordinate_species_indices  # noqa: E402
from material_presets import (  # noqa: E402
    DEFAULT_PRESET_DIR as MATERIAL_PRESET_DIR,
    resolve_material_bundle,
)
from g2m_deeph_dataset_size_minimum import (
    AGGREGATION_MODES,
    CANONICAL_POWER_LAW_MODEL,
    CLAIM_MODES,
    COST_BASES,
    MetricFileLoadError,
    aggregation_mode_classification as dataset_minimum_aggregation_mode_classification,
    analysis_rows_for_aggregation_mode as dataset_minimum_analysis_rows_for_aggregation_mode,
    canonical_fit_model as dataset_minimum_canonical_fit_model,
    discover_metric_files as discover_dataset_minimum_metric_files,
    fit_models_equivalent as dataset_minimum_fit_models_equivalent,
    group_config_rows as dataset_minimum_group_config_rows,
    load_metric_file_rows as load_dataset_minimum_metric_file_rows,
    load_run_root_rows as dataset_minimum_load_run_root_rows,
    normalize_rows as dataset_minimum_normalize_rows,
    parse_claim_mode as dataset_minimum_parse_claim_mode,
    parse_cost_basis as dataset_minimum_parse_cost_basis,
    parse_single_fit_model as dataset_minimum_parse_single_fit_model,
    resolve_aggregation_mode as resolve_dataset_minimum_aggregation_mode,
)
LOG_HEARTBEAT_SECONDS = 30.0
DEFAULT_LOG_RESPONSE_LIMIT = 2000
MAX_LOG_RESPONSE_LIMIT = 20000
MAX_CROSS_DIAGNOSTIC_MANIFEST_BYTES = 2_000_000
MAX_PLOT_SAMPLE_ROWS_PER_GROUP = 100
METRIC_VERSION = "2026-05-08.frontier-window-v1"
DEFAULT_VENV_ACTIVATE_COMMAND = "source ${REPO_ROOT}/.venv/bin/activate"
KGRID_MONKHORST_PACK_DIRECTIVES = {"kgrid_monkhorst_pack", "kgrid.monkhorstpack"}


def format_duration(seconds: float | int | None) -> str:
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


def remove_tree_with_retries(path: Path, *, attempts: int = 5) -> bool:
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except OSError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        return False
    return True


def reset_output_directory(path: Path) -> None:
    """Create an empty output directory without deleting an active tree in place."""
    if path.exists():
        stale_root = path.parent / ".stale"
        stale_root.mkdir(parents=True, exist_ok=True)
        stale_path = stale_root / f"{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(6)}"
        try:
            path.rename(stale_path)
        except FileNotFoundError:
            pass
        except OSError:
            if not remove_tree_with_retries(path):
                raise
        else:
            remove_tree_with_retries(stale_path)
    path.mkdir(parents=True, exist_ok=False)


def _strip_fdf_comment_for_kgrid(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _float_token_for_kgrid(value: str) -> float | None:
    try:
        return float(value.replace("d", "e").replace("D", "E"))
    except ValueError:
        return None


def _monkhorst_pack_rows_are_gamma(rows: list[str]) -> bool:
    matrix: list[list[float]] = []
    for row in rows:
        values = [_float_token_for_kgrid(token) for token in row.split()]
        if len(values) < 4 or any(value is None for value in values[:4]):
            return False
        matrix.append([float(value) for value in values[:4] if value is not None])
    if len(matrix) != 3:
        return False
    expected = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    )
    return all(
        math.isclose(matrix[row][col], expected[row][col], rel_tol=0.0, abs_tol=1e-12)
        for row in range(3)
        for col in range(4)
    )


def _monkhorst_pack_inline_is_gamma(tokens: list[str]) -> bool:
    values = [_float_token_for_kgrid(token) for token in tokens]
    if not values or any(value is None for value in values):
        return False
    numeric = [float(value) for value in values if value is not None]
    if len(numeric) == 3:
        return all(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12) for value in numeric)
    if len(numeric) >= 6:
        return all(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12) for value in numeric[:3]) and all(
            math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12) for value in numeric[3:6]
        )
    return False


def structure_has_nongamma_monkhorst_pack(structure_path: Path) -> bool:
    """Lightweight UI-side check for deciding whether to opt into k-point metrics."""
    if not structure_path.exists():
        return False
    try:
        lines = structure_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return False
    kgrid_block_name: str | None = None
    kgrid_block_rows: list[str] = []
    for raw_line in lines:
        clean = _strip_fdf_comment_for_kgrid(raw_line)
        if not clean:
            continue
        lower = clean.lower()
        parts = lower.split()
        key = parts[0] if parts else ""
        if lower.startswith("%block"):
            block_name = parts[1] if len(parts) > 1 else ""
            if block_name in KGRID_MONKHORST_PACK_DIRECTIVES:
                kgrid_block_name = block_name
                kgrid_block_rows = []
            continue
        if lower.startswith("%endblock"):
            if kgrid_block_name is not None and not _monkhorst_pack_rows_are_gamma(kgrid_block_rows):
                return True
            kgrid_block_name = None
            kgrid_block_rows = []
            continue
        if kgrid_block_name is not None:
            kgrid_block_rows.append(clean)
            continue
        if key in KGRID_MONKHORST_PACK_DIRECTIVES:
            return not _monkhorst_pack_inline_is_gamma(parts[1:])
    return bool(kgrid_block_name is not None and not _monkhorst_pack_rows_are_gamma(kgrid_block_rows))


def result_dir_needs_kpoint_metrics(result_dir: Path) -> bool:
    structure_root = result_dir / "structures"
    if not structure_root.exists():
        return False
    for structure_path in itertools.islice(structure_root.glob("*/RUN.fdf"), 20):
        if structure_has_nongamma_monkhorst_pack(structure_path):
            return True
    return False


@dataclass(frozen=True)
class PipelineSpec:
    key: str
    label: str
    root: Path
    main_script: Path

    @property
    def config_path(self) -> Path:
        return self.root / "pipeline_config.yaml"


PIPELINES = {
    "md": PipelineSpec(
        key="md",
        label="MD",
        root=REPO_ROOT / "MD",
        main_script=REPO_ROOT / "MD" / "scripts" / "main_md.py",
    ),
    "atom_displacement": PipelineSpec(
        key="atom_displacement",
        label="AtomDisplacement",
        root=REPO_ROOT / "AtomDisplacement",
        main_script=REPO_ROOT / "AtomDisplacement" / "scripts" / "main_atdisp.py",
    ),
}


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    pipeline_key: str | None
    dataset_root: Path | None
    training_root: Path | None
    results_root: Path | None
    legacy_aliases: tuple[str, ...] = ()
    available: bool = True
    unavailable_reason: str = ""


METHOD_REGISTRY: dict[str, MethodSpec] = {
    "md": MethodSpec(
        method_id="md",
        display_name=SCIENTIFIC_METHOD_REGISTRY["md"].display_name,
        pipeline_key="md",
        dataset_root=PIPELINES["md"].root / "dataset",
        training_root=PIPELINES["md"].root / "training",
        results_root=RESULTS_ROOT / SCIENTIFIC_METHOD_REGISTRY["md"].results_dir,
    ),
    "siesta_fc_cartesian": MethodSpec(
        method_id="siesta_fc_cartesian",
        display_name=SCIENTIFIC_METHOD_REGISTRY["siesta_fc_cartesian"].display_name,
        pipeline_key="atom_displacement",
        dataset_root=PIPELINES["atom_displacement"].root / "dataset" / "FC_steps",
        training_root=PIPELINES["atom_displacement"].root / "training",
        results_root=RESULTS_ROOT / SCIENTIFIC_METHOD_REGISTRY["siesta_fc_cartesian"].results_dir,
        legacy_aliases=SCIENTIFIC_METHOD_REGISTRY["siesta_fc_cartesian"].legacy_aliases,
    ),
    "random_cartesian": MethodSpec(
        method_id="random_cartesian",
        display_name=SCIENTIFIC_METHOD_REGISTRY["random_cartesian"].display_name,
        pipeline_key=None,
        dataset_root=PIPELINES["atom_displacement"].root / "dataset" / "RandomCartesian_steps",
        training_root=PIPELINES["atom_displacement"].root / "training",
        results_root=RESULTS_ROOT / SCIENTIFIC_METHOD_REGISTRY["random_cartesian"].results_dir,
        available=True,
    ),
}

DATASET_ONLY_RUN_MODE = "dataset_only"
FULL_STRICT_RUN_MODE = "full_strict_pipeline"
DOWNSTREAM_ONLY_RUN_MODE = "train_test_metrics_plots_only"
DEEPH_COMPARISON_RUN_MODE = "deeph_comparison"
GRAPH2MAT_DEEPH_RUN_MODE = "graph2mat_deeph_comparison"
RUN_MODES = {
    DATASET_ONLY_RUN_MODE,
    FULL_STRICT_RUN_MODE,
    DOWNSTREAM_ONLY_RUN_MODE,
    DEEPH_COMPARISON_RUN_MODE,
    GRAPH2MAT_DEEPH_RUN_MODE,
}
MD_DOWNSTREAM_STEPS = ("run_md_training", "run_md_testing", "run_md_prediction")
ATOM_DOWNSTREAM_STEPS = ("render_inputs", "run_atdisp_training", "run_atdisp_testing", "run_atdisp_prediction")
PRESERVE_ARCHIVED_SPLITS = "preserve_archived_splits"
REBUILD_REUSABLE_SPLITS = "rebuild_splits"
REUSABLE_SPLIT_POLICIES = {PRESERVE_ARCHIVED_SPLITS, REBUILD_REUSABLE_SPLITS}
DEFAULT_DEEPH_REPO = Path(os.environ["DEEPH_PACK_ROOT"]).expanduser() if os.environ.get("DEEPH_PACK_ROOT") else None
DEFAULT_DEEPH_PYTHON = DEFAULT_DEEPH_REPO / ".venv" / "bin" / "python" if DEFAULT_DEEPH_REPO else None
DEFAULT_PIPELINE_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_DEEPH_GRAPH2MAT_RESULT = (
    RESULTS_ROOT
    / "results_md"
    / "md_dataset1_df6jp1_graphene_w90_default_mse_lr0p005_ep600"
    / "run_20260521_141037"
)


def method_registry_payload() -> list[dict[str, Any]]:
    payload = []
    for spec in METHOD_REGISTRY.values():
        payload.append(
            {
                "method_id": spec.method_id,
                "display_name": spec.display_name,
                "dataset_root": str(spec.dataset_root) if spec.dataset_root else "",
                "training_root": str(spec.training_root) if spec.training_root else "",
                "results_root": str(spec.results_root) if spec.results_root else "",
                "legacy_aliases": list(spec.legacy_aliases),
                "available": spec.available,
                "availability_status": "available" if spec.available else "not_implemented",
                "unavailable_reason": spec.unavailable_reason,
                "config_snapshot": {},
                "warnings": ([] if spec.available else [spec.unavailable_reason]),
                "severe_warnings": ([] if spec.available else [spec.unavailable_reason]),
            }
        )
    return payload


def normalize_selected_methods(value: Any, *, default_legacy: bool = True) -> list[str]:
    if value is None:
        return ["md", "siesta_fc_cartesian"] if default_legacy else []
    if isinstance(value, str):
        raw_methods = [item.strip() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, list):
        raw_methods = [str(item).strip() for item in value]
    else:
        raise RuntimeError("selected_methods debe ser una lista o texto separado por comas.")
    selected: list[str] = []
    unknown: list[str] = []
    for raw in raw_methods:
        if not raw:
            continue
        try:
            method_id = normalize_method_id(raw)
        except ValueError:
            unknown.append(raw)
            continue
        if method_id not in selected:
            selected.append(method_id)
    if unknown:
        raise RuntimeError(f"selected_methods contiene metodos no soportados: {unknown}.")
    if not selected:
        raise RuntimeError("Selecciona al menos un metodo para el experimento.")
    unavailable = [
        method_id
        for method_id in selected
        if not METHOD_REGISTRY[method_id].available
    ]
    if unavailable:
        reasons = [
            f"{method_id}: {METHOD_REGISTRY[method_id].unavailable_reason}"
            for method_id in unavailable
        ]
        raise RuntimeError("; ".join(reasons))
    return selected


def parse_run_mode(value: Any) -> str:
    mode = FULL_STRICT_RUN_MODE if value in (None, "") else str(value).strip()
    if mode not in RUN_MODES:
        raise RuntimeError(
            "run_mode debe ser 'dataset_only', 'full_strict_pipeline', "
            "'train_test_metrics_plots_only', 'deeph_comparison' o "
            "'graph2mat_deeph_comparison' "
            f"(recibido: {value!r})."
        )
    return mode


def run_mode_skips_dataset_generation(run_mode: str) -> bool:
    return run_mode in {DOWNSTREAM_ONLY_RUN_MODE, GRAPH2MAT_DEEPH_RUN_MODE}


def run_mode_uses_planned_dataset_targets(run_mode: str) -> bool:
    return run_mode == FULL_STRICT_RUN_MODE


def resolve_ui_path(value: Any, default: Path | str | None = None) -> Path:
    raw = default if value in (None, "") else value
    if raw in (None, ""):
        raise RuntimeError("Se esperaba una ruta no vacia.")
    text = str(raw).strip().replace("${REPO_ROOT}", str(REPO_ROOT))
    return Path(os.path.expandvars(text)).expanduser()


def parse_ui_path_list(value: Any) -> list[Path]:
    if value in (None, "", []):
        return []
    raw_items: list[Any]
    if isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        text = str(value)
        raw_items = []
        for line in text.replace(",", "\n").splitlines():
            item = line.strip()
            if item:
                raw_items.append(item)
    paths: list[Path] = []
    seen: set[str] = set()
    for item in raw_items:
        path = resolve_ui_path(item)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def parse_optional_positive_int(value: Any, label: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} debe ser un entero positivo.") from exc
    if parsed <= 0:
        raise RuntimeError(f"{label} debe ser un entero positivo.")
    return parsed


def parse_positive_float(value: Any, label: str, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} debe ser un numero positivo.") from exc
    if parsed <= 0:
        raise RuntimeError(f"{label} debe ser un numero positivo.")
    return parsed


def parse_non_negative_int(value: Any, label: str, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} debe ser un entero >= 0.") from exc
    if parsed < 0:
        raise RuntimeError(f"{label} debe ser un entero >= 0.")
    return parsed


def parse_deeph_comparison_options(
    value: Any,
    *,
    require_graph2mat_result: bool = True,
) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    graph2mat_result_dirs = parse_ui_path_list(payload.get("graph2mat_result_dirs"))
    candidate_summary_csv = (
        resolve_ui_path(payload.get("graph2mat_candidate_summary_csv"))
        if payload.get("graph2mat_candidate_summary_csv") not in (None, "")
        else None
    )
    single_result_required = require_graph2mat_result and not graph2mat_result_dirs and candidate_summary_csv is None
    graph2mat_result_dir = None
    if single_result_required:
        graph2mat_result_dir = resolve_ui_path(payload.get("graph2mat_result_dir"), DEFAULT_DEEPH_GRAPH2MAT_RESULT)
    elif not graph2mat_result_dirs and candidate_summary_csv is None and payload.get("graph2mat_result_dir") not in (None, ""):
        graph2mat_result_dir = resolve_ui_path(payload.get("graph2mat_result_dir"))
    if payload.get("deeph_repo") in (None, "") and DEFAULT_DEEPH_REPO is None:
        raise RuntimeError("Configura deeph_repo o define DEEPH_PACK_ROOT para ejecutar DeepH.")
    deeph_repo = resolve_ui_path(payload.get("deeph_repo"), DEFAULT_DEEPH_REPO)
    deeph_python = resolve_ui_path(payload.get("deeph_python"), DEFAULT_DEEPH_PYTHON or sys.executable)
    pipeline_python = resolve_ui_path(
        payload.get("pipeline_python"),
        DEFAULT_PIPELINE_PYTHON if DEFAULT_PIPELINE_PYTHON.exists() else sys.executable,
    )
    output_root = resolve_ui_path(
        payload.get("output_root"),
        RESULTS_ROOT / "graphene_w90_deeph_fair_benchmark",
    )
    epochs = parse_optional_positive_int(payload.get("epochs"), "DeepH epochs") or 200
    batch_size = parse_optional_positive_int(payload.get("batch_size"), "DeepH batch size") or 4
    sample_limit = parse_optional_positive_int(
        payload.get("sample_limit_per_split"),
        "Sample limit per split",
    )
    top_percent = parse_positive_float(
        payload.get("graph2mat_top_percent"),
        "Graph2Mat top percent",
        10.0,
    )
    if top_percent > 100.0:
        raise RuntimeError("Graph2Mat top percent debe estar entre 0 y 100.")
    top_count = parse_optional_positive_int(payload.get("graph2mat_top_count"), "Graph2Mat top count")
    learning_rate = parse_positive_float(payload.get("learning_rate"), "DeepH learning rate", 0.001)
    seed = parse_non_negative_int(payload.get("seed"), "DeepH seed", 0)
    siesta_command = str(payload.get("siesta_command") or "siesta").strip()
    if not siesta_command:
        raise RuntimeError("SIESTA command no puede estar vacio.")
    device = str(payload.get("device") or "cuda:0").strip()
    if not device:
        device = "cuda:0"
    if graph2mat_result_dir is not None and not graph2mat_result_dir.exists():
        raise RuntimeError(f"No existe el result dir de Graph2Mat: {graph2mat_result_dir}")
    for result_dir in graph2mat_result_dirs:
        if not result_dir.exists():
            raise RuntimeError(f"No existe el result dir de Graph2Mat: {result_dir}")
    if candidate_summary_csv is not None and not candidate_summary_csv.exists():
        raise RuntimeError(f"No existe el CSV de candidatos Graph2Mat: {candidate_summary_csv}")
    if not deeph_repo.exists():
        raise RuntimeError(f"No existe el repo de DeepH: {deeph_repo}")
    if not deeph_python.exists():
        raise RuntimeError(f"No existe el Python de DeepH: {deeph_python}")
    if not pipeline_python.exists():
        raise RuntimeError(f"No existe el Python del pipeline: {pipeline_python}")
    return {
        "graph2mat_result_dir": str(graph2mat_result_dir) if graph2mat_result_dir is not None else "",
        "graph2mat_result_dirs": [str(path) for path in graph2mat_result_dirs],
        "graph2mat_candidate_summary_csv": str(candidate_summary_csv) if candidate_summary_csv is not None else "",
        "deeph_repo": str(deeph_repo),
        "deeph_python": str(deeph_python),
        "pipeline_python": str(pipeline_python),
        "output_root": str(output_root),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "sample_limit_per_split": sample_limit,
        "graph2mat_top_percent": top_percent,
        "graph2mat_top_count": top_count,
        "allow_regenerate_siesta": parse_bool(payload.get("allow_regenerate_siesta"), True),
        "siesta_command": siesta_command,
        "device": device,
        "copy_raw": parse_bool(payload.get("copy_raw"), True),
    }


def parse_reusable_dataset_ids(value: Any) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        raw_ids = [item.strip() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, list):
        raw_ids = [str(item).strip() for item in value]
    else:
        raise RuntimeError("reusable_dataset_ids debe ser una lista o texto separado por comas.")
    ids: list[str] = []
    for item in raw_ids:
        if item and item not in ids:
            ids.append(item)
    return ids


def planned_dataset_target_id(method_id: Any, recipe_id: Any, occurrence: int = 1) -> str:
    method = normalize_method_id(method_id, allow_unknown=True)
    recipe = str(recipe_id or "").strip()
    if not recipe:
        raise RuntimeError("dataset target recipe_id vacio.")
    base = f"{method}:{recipe}"
    return base if int(occurrence) <= 1 else f"{base}#{int(occurrence)}"


def parse_dataset_targets(value: Any) -> list[dict[str, Any]]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        raw_targets: list[Any] = [item.strip() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, list):
        raw_targets = value
    else:
        raise RuntimeError("dataset_targets debe ser una lista o texto separado por comas.")
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_target in enumerate(raw_targets):
        if isinstance(raw_target, dict):
            target_id = parse_optional_text(
                raw_target.get("target_id", raw_target.get("id")),
                f"dataset_targets[{index}].target_id",
            )
            method_id = parse_optional_text(raw_target.get("method_id"), f"dataset_targets[{index}].method_id")
            recipe_id = parse_optional_text(raw_target.get("recipe_id"), f"dataset_targets[{index}].recipe_id")
            if target_id is None and method_id is not None and recipe_id is not None:
                target_id = planned_dataset_target_id(method_id, recipe_id)
            if target_id is None:
                raise RuntimeError(
                    f"dataset_targets[{index}] necesita target_id o method_id+recipe_id."
                )
            target = {
                "target_id": target_id,
                "method_id": normalize_method_id(method_id, allow_unknown=True) if method_id else None,
                "recipe_id": recipe_id,
                "dataset_label": parse_optional_text(
                    raw_target.get("dataset_label"),
                    f"dataset_targets[{index}].dataset_label",
                ),
            }
        else:
            target_id = str(raw_target).strip()
            if not target_id:
                continue
            target = {"target_id": target_id}
        if target["target_id"] not in seen:
            seen.add(str(target["target_id"]))
            targets.append(target)
    return targets


def validate_training_plan_for_run_mode(training_plan: list[dict[str, Any]], run_mode: str) -> None:
    if not training_plan:
        return
    if run_mode == DATASET_ONLY_RUN_MODE:
        raise RuntimeError("training_plan no esta disponible con dataset_only.")
    for item in training_plan:
        index = item.get("index", "?")
        if run_mode_skips_dataset_generation(run_mode):
            if not item.get("reusable_dataset_ids"):
                raise RuntimeError(
                    f"training_plan[{index}] necesita reusable_dataset_ids en train_test_metrics_plots_only."
                )
        elif run_mode_uses_planned_dataset_targets(run_mode):
            if not item.get("dataset_targets"):
                raise RuntimeError(
                    f"training_plan[{index}] necesita dataset_targets en {run_mode}."
                )


def parse_reusable_split_policy(value: Any) -> str:
    policy = PRESERVE_ARCHIVED_SPLITS if value in (None, "") else str(value).strip().lower()
    if policy not in REUSABLE_SPLIT_POLICIES:
        raise RuntimeError(
            "reusable_split_policy debe ser 'preserve_archived_splits' o "
            f"'rebuild_splits' (recibido: {value!r})."
        )
    return policy


def parse_random_cartesian_options(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise RuntimeError("random_cartesian_options debe ser un objeto.")
    return dict(value)


def random_cartesian_sizes_from_options(options: dict[str, Any]) -> list[int]:
    dataset_specs = options.get("_dataset_specs")
    if isinstance(dataset_specs, list) and dataset_specs:
        sizes = [int(spec.get("size", 0)) for spec in dataset_specs if isinstance(spec, dict)]
        if not sizes or any(size < 3 for size in sizes):
            raise RuntimeError("random_cartesian dataset_recipes debe contener tamanos enteros >= 3.")
        return sizes
    raw_sizes = options.get("n_structures")
    if raw_sizes in (None, ""):
        config = load_config(PIPELINES["atom_displacement"].config_path)
        random_config = (config.get("structure", {}) or {}).get("random_cartesian", {}) or {}
        raw_sizes = random_config.get("n_structures", 100)
    if isinstance(raw_sizes, str):
        values = [item.strip() for item in raw_sizes.replace(";", ",").split(",") if item.strip()]
    elif isinstance(raw_sizes, (list, tuple)):
        values = list(raw_sizes)
    else:
        values = [raw_sizes]
    sizes = [int(value) for value in values]
    if not sizes or any(size < 3 for size in sizes):
        raise RuntimeError("random_cartesian.n_structures debe contener tamanos enteros >= 3.")
    return sizes


def random_cartesian_size_from_options(options: dict[str, Any]) -> int:
    return random_cartesian_sizes_from_options(options)[0]


def slugify_label(value: Any, default: str = "dataset") -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
        ".": "p",
        "+": "plus",
        "-": "m",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    slug = "".join(char if char.isalnum() else "_" for char in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or default


def stable_payload_hash(payload: Any, *, length: int = 12) -> str:
    text = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def recipe_set_hash(recipes: dict[str, Any] | list[Any] | None) -> str:
    return stable_payload_hash(recipes or {}, length=16)


def _optional_seed(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, float) and not value.is_integer():
        raise RuntimeError("seed debe ser un entero >= 0.")
    seed = int(value)
    if seed < 0:
        raise RuntimeError("seed debe ser un entero >= 0.")
    return seed


def dataset_recipe_seed_values(recipes: dict[str, Any] | list[Any] | None) -> list[int]:
    values: list[int] = []

    def add(value: Any) -> None:
        seed = _optional_seed(value)
        if seed is not None and seed not in values:
            values.append(seed)

    if isinstance(recipes, dict):
        recipe_items: list[Any] = []
        for method_recipes in recipes.values():
            if isinstance(method_recipes, list):
                recipe_items.extend(method_recipes)
        for recipe in recipe_items:
            if not isinstance(recipe, dict):
                continue
            add(recipe.get("seed"))
            for block in recipe.get("blocks") or []:
                if isinstance(block, dict):
                    add(block.get("seed"))
    elif isinstance(recipes, list):
        for recipe in recipes:
            if isinstance(recipe, dict):
                add(recipe.get("seed"))
    return values


MAX_DATASET_LABEL_LENGTH = 96
DATASET_ID_LENGTH = 6
DATASET_ID_ALPHABET = string.ascii_letters + string.digits
CROSS_ID_LENGTH = 12
DATASET_LABEL_PREFIXES = {
    "md": "MD",
    "fc": "FC",
    "siesta_fc_cartesian": "FC",
    "atom_displacement": "FC",
    "random_cartesian": "RC",
    "rc": "RC",
}
METHOD_LABEL_SLUGS = {
    "md": "md",
    "atom_displacement": "fc",
    "siesta_fc_cartesian": "fc",
    "random_cartesian": "rc",
}
TEST_SET_LABEL_SLUGS = {
    "test_md": "test_md",
    "test_atomdisp": "test_ad",
    "test_siesta_fc_cartesian": "test_fc",
    "test_random_cartesian": "test_rc",
    "test_mixed": "test_mixed",
}


def compact_dataset_label(label: str, payload: Any, *, max_length: int = MAX_DATASET_LABEL_LENGTH) -> str:
    clean = slugify_label(label, "dataset")
    if len(clean) <= max_length:
        return clean
    digest = stable_payload_hash(payload, length=10)
    keep = max(16, max_length - len(digest) - 1)
    return f"{clean[:keep].rstrip('_')}_{digest}"


def _dataset_label_prefix(method: str) -> str:
    return DATASET_LABEL_PREFIXES.get(str(method).strip().lower(), slugify_label(method, "DS").upper())


def _random_dataset_id(used_ids: set[str]) -> str:
    for _attempt in range(4096):
        token = "".join(secrets.choice(DATASET_ID_ALPHABET) for _ in range(DATASET_ID_LENGTH))
        if token not in used_ids:
            used_ids.add(token)
            return token
    raise RuntimeError("No se pudo generar un id de dataset unico.")


def allocate_dataset_label(
    method: str,
    dataset_index: int,
    *,
    used_ids: set[str],
    used_labels: set[str] | None = None,
) -> tuple[str, str]:
    used_labels = used_labels if used_labels is not None else set()
    prefix = _dataset_label_prefix(method)
    for _attempt in range(4096):
        token = _random_dataset_id(used_ids)
        label = f"{prefix}_dataset{int(dataset_index) + 1}_{token}"
        if label not in used_labels:
            used_labels.add(label)
            return label, token
    raise RuntimeError("No se pudo generar un nombre de dataset unico.")


def register_dataset_label(label: Any, used_ids: set[str], used_labels: set[str]) -> None:
    text = str(label or "").strip()
    if not text:
        return
    used_labels.add(text)
    token = text.rsplit("_", 1)[-1]
    if len(token) == DATASET_ID_LENGTH and token.isascii() and token.isalnum():
        used_ids.add(token)


def method_slug_for_cross(method: Any) -> str:
    method_id = normalize_method_id(method, allow_unknown=True)
    return METHOD_LABEL_SLUGS.get(method_id, slugify_label(method_id, "method"))


def test_set_slug_for_cross(test_set: Any) -> str:
    value = normalize_test_set_id(test_set)
    return TEST_SET_LABEL_SLUGS.get(value, slugify_label(value, "test"))


def cross_pair_id(combo_by_method: dict[str, dict[str, Any]], methods: list[str]) -> str:
    payload: dict[str, Any] = {}
    for method in methods:
        run = combo_by_method.get(method) or {}
        payload[method_slug_for_cross(method)] = {
            "method": method,
            "dataset_label": run.get("dataset_label"),
            "dataset_short_id": run.get("dataset_short_id"),
            "dataset_size": run.get("dataset_size"),
            "recipe_id": run.get("recipe_id"),
            "recipe_set_hash": run.get("recipe_set_hash"),
            "result_dir": run.get("result_dir"),
            "split_manifest": run.get("split_manifest"),
        }
    return f"cross_{stable_payload_hash(payload, length=CROSS_ID_LENGTH)}"


def _existing_split_manifests_for_result(result: dict[str, Any]) -> dict[str, str]:
    result_dir_value = str(result.get("result_dir") or "").strip()
    if not result_dir_value:
        return {}
    result_dir = Path(result_dir_value)
    split_root = result_dir / "splits"
    split_manifests: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        for file_name in (f"{split_name}_valid_manifest.csv", f"{split_name}_manifest.csv"):
            candidate = split_root / file_name
            if candidate.exists():
                split_manifests[split_name] = str(candidate)
                break
    return split_manifests


def _method_mismatch_messages(
    manifest: dict[str, Any],
    method: str,
    key: str,
    *,
    severe_only: bool,
) -> list[str]:
    messages: list[str] = []
    for mismatch in manifest.get(key, []) or []:
        if not isinstance(mismatch, dict):
            continue
        methods = [normalize_method_id(item, allow_unknown=True) for item in mismatch.get("methods", []) or []]
        if method not in methods:
            continue
        severity = str(mismatch.get("severity") or "warning")
        if severe_only and severity != "severe":
            continue
        if not severe_only and severity == "severe":
            continue
        section = mismatch.get("section")
        mismatch_key = mismatch.get("key", "unknown")
        label = f"{section}.{mismatch_key}" if section else str(mismatch_key)
        messages.append(f"{severity}: {label} differs across {', '.join(methods)}")
    return messages


def _checkpoint_warnings_for_run(method: str, run: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    checkpoint_path = run.get("model_checkpoint")
    checkpoint_hash = run.get("model_checkpoint_sha256") or (
        (run.get("model_checkpoint_metadata") or {}).get("sha256")
        if isinstance(run.get("model_checkpoint_metadata"), dict)
        else None
    )
    checkpoint_selection_warning = str(run.get("checkpoint_selection_warning") or "").strip()
    if checkpoint_selection_warning:
        warnings.append(checkpoint_selection_warning)
    if checkpoint_path and not checkpoint_hash:
        warnings.append(f"Missing checkpoint hash for {method} {run.get('dataset_label', '')}.")
    if not checkpoint_path:
        warnings.append(f"Missing checkpoint path for {method} {run.get('dataset_label', '')}.")
    return [warning for warning in warnings if warning]


def _provenance_entry_for_run(run: dict[str, Any]) -> dict[str, Any]:
    artifact_hashes = run.get("artifact_hashes") if isinstance(run.get("artifact_hashes"), dict) else {}
    checkpoint_metadata = run.get("model_checkpoint_metadata")
    if not isinstance(checkpoint_metadata, dict):
        checkpoint_metadata = {}
    checkpoint_hash = run.get("model_checkpoint_sha256") or checkpoint_metadata.get("sha256")
    recipe_hash = run.get("recipe_hash") or run.get("recipe_set_hash")
    material_provenance = flatten_material_provenance(run.get("material_provenance") or run)
    entry = {
        "dataset_size": run.get("dataset_size"),
        "effective_dataset_size": run.get("effective_dataset_size"),
        "dataset_label": run.get("dataset_label"),
        "recipe_id": run.get("recipe_id"),
        "recipe_label": run.get("recipe_label"),
        "recipe_hash": recipe_hash,
        "recipe_set_hash": run.get("recipe_set_hash"),
        "result_directory": run.get("result_dir"),
        "split_manifests": _existing_split_manifests_for_result(run),
        "checkpoint_path": run.get("model_checkpoint"),
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_manifest": run.get("checkpoint_manifest"),
        "checkpoint_selection_warning": run.get("checkpoint_selection_warning", ""),
        "basis_hash": artifact_hashes.get("basis"),
        "pseudopotential_hash": artifact_hashes.get("pseudopotentials"),
        "pipeline": run.get("pipeline"),
        "method_id": normalize_method_id(run.get("method_id") or run.get("pipeline") or "", allow_unknown=True),
        "seed": run.get("seed"),
        "warnings": _checkpoint_warnings_for_run(
            normalize_method_id(run.get("method_id") or run.get("pipeline") or "", allow_unknown=True),
            run,
        ),
    }
    if material_provenance:
        entry["material_provenance"] = material_provenance
        entry.update({key: material_provenance.get(key) for key in MATERIAL_FLAT_FIELDS if key in material_provenance})
    return entry


def _frozen_test_manifests_from_cross_evaluations(manifest: dict[str, Any]) -> dict[str, str]:
    cross_evaluation = manifest.get("cross_evaluation") if isinstance(manifest.get("cross_evaluation"), dict) else {}
    frozen: dict[str, str] = {}
    for item in cross_evaluation.get("cross_evaluations", []) or []:
        if not isinstance(item, dict):
            continue
        test_set = str(item.get("test_set") or "").strip()
        frozen_manifest = str(item.get("frozen_test_manifest") or "").strip()
        if test_set and frozen_manifest:
            frozen[test_set] = frozen_manifest
    return frozen


def build_method_provenance(
    manifest: dict[str, Any],
    *,
    selected_methods: list[str] | None = None,
    runs: list[dict[str, Any]] | None = None,
    frozen_test_manifests_by_test_set: dict[str, str] | None = None,
) -> dict[str, Any]:
    runs = runs if runs is not None else list(manifest.get("runs", []) or [])
    selected_methods = selected_methods or [
        normalize_method_id(method)
        for method in (manifest.get("selected_methods") or [])
    ]
    if not selected_methods:
        selected_methods = sorted(
            {
                normalize_method_id(run.get("method_id") or run.get("pipeline") or "", allow_unknown=True)
                for run in runs
                if run.get("method_id") or run.get("pipeline")
            }
        )
    frozen_test_manifests_by_test_set = (
        frozen_test_manifests_by_test_set
        if frozen_test_manifests_by_test_set is not None
        else _frozen_test_manifests_from_cross_evaluations(manifest)
    )
    runs_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        method_id = normalize_method_id(run.get("method_id") or run.get("pipeline") or "", allow_unknown=True)
        if method_id:
            runs_by_method[method_id].append(run)

    siesta_hash_by_method = manifest.get("siesta_settings_hash_by_method") or {}
    model_hash_by_method = manifest.get("model_config_hash_by_method") or {}
    basis_hash_by_method = manifest.get("basis_hash_by_method") or {}
    pseudo_hash_by_method = manifest.get("pseudopotential_hash_by_method") or {}
    provenance: dict[str, Any] = {}
    for method in selected_methods:
        method_id = normalize_method_id(method, allow_unknown=True)
        method_runs = sorted(
            runs_by_method.get(method_id, []),
            key=lambda run: (
                int(run.get("dataset_size") or 0),
                str(run.get("dataset_label") or ""),
                str(run.get("result_dir") or ""),
            ),
        )
        run_entries = [_provenance_entry_for_run(run) for run in method_runs]
        warnings: list[str] = []
        severe_warnings: list[str] = []
        if not method_runs:
            warnings.append(f"Missing successful run provenance for selected method {method_id}.")
        if not siesta_hash_by_method.get(method_id):
            warnings.append(f"Missing SIESTA settings hash for {method_id}.")
        if not model_hash_by_method.get(method_id):
            warnings.append(f"Missing model settings hash for {method_id}.")
        if not basis_hash_by_method.get(method_id):
            warnings.append(f"Missing basis hash for {method_id}.")
        if not pseudo_hash_by_method.get(method_id):
            warnings.append(f"Missing pseudopotential hash for {method_id}.")
        for run_entry in run_entries:
            warnings.extend(str(item) for item in run_entry.get("warnings", []) or [] if item)
        warnings.extend(_method_mismatch_messages(manifest, method_id, "siesta_settings_mismatches", severe_only=False))
        warnings.extend(_method_mismatch_messages(manifest, method_id, "model_config_mismatches", severe_only=False))
        severe_warnings.extend(
            _method_mismatch_messages(manifest, method_id, "siesta_settings_severe_mismatches", severe_only=True)
        )
        severe_warnings.extend(
            _method_mismatch_messages(manifest, method_id, "model_config_severe_mismatches", severe_only=True)
        )
        owned_test_set = (
            SCIENTIFIC_METHOD_REGISTRY[method_id].frozen_test_set
            if method_id in SCIENTIFIC_METHOD_REGISTRY
            else f"test_{method_id}"
        )
        frozen_for_method = {
            test_set: path
            for test_set, path in sorted((frozen_test_manifests_by_test_set or {}).items())
            if test_set in {owned_test_set, "test_mixed"} or frozen_test_manifests_by_test_set is not None
        }
        first_run = run_entries[0] if len(run_entries) == 1 else {}
        representative_run = run_entries[0] if run_entries else {}
        first_material = flatten_material_provenance(
            representative_run.get("material_provenance") or representative_run
        )
        material_hashes = sorted(
            {
                str(run.get("material_compatibility_hash"))
                for run in run_entries
                if run.get("material_compatibility_hash")
            }
        )
        entry = {
            "method_id": method_id,
            "display_name": (
                SCIENTIFIC_METHOD_REGISTRY[method_id].display_name
                if method_id in SCIENTIFIC_METHOD_REGISTRY
                else method_id
            ),
            "dataset_size": first_run.get("dataset_size"),
            "dataset_label": first_run.get("dataset_label"),
            "dataset_sizes": [run.get("dataset_size") for run in run_entries if run.get("dataset_size") is not None],
            "dataset_labels": [run.get("dataset_label") for run in run_entries if run.get("dataset_label")],
            "recipe_id": first_run.get("recipe_id"),
            "recipe_ids": sorted({str(run.get("recipe_id")) for run in run_entries if run.get("recipe_id")}),
            "recipe_hash": first_run.get("recipe_hash"),
            "recipe_hashes": sorted({str(run.get("recipe_hash")) for run in run_entries if run.get("recipe_hash")}),
            "result_directory": first_run.get("result_directory"),
            "result_directories": [run.get("result_directory") for run in run_entries if run.get("result_directory")],
            "split_manifest": first_run.get("split_manifests"),
            "split_manifests_by_dataset_label": {
                str(run.get("dataset_label")): run.get("split_manifests", {})
                for run in run_entries
                if run.get("dataset_label")
            },
            "frozen_test_manifest": frozen_for_method.get(owned_test_set),
            "frozen_test_manifests": frozen_for_method,
            "siesta_settings_hash": siesta_hash_by_method.get(method_id),
            "model_settings_hash": model_hash_by_method.get(method_id),
            "basis_hash": basis_hash_by_method.get(method_id) or first_run.get("basis_hash"),
            "pseudopotential_hash": pseudo_hash_by_method.get(method_id) or first_run.get("pseudopotential_hash"),
            "checkpoint_path": first_run.get("checkpoint_path"),
            "checkpoint_hash": first_run.get("checkpoint_hash"),
            "checkpoint_manifest": first_run.get("checkpoint_manifest"),
            "runs": run_entries,
            "warnings": sorted(dict.fromkeys(warnings)),
            "severe_warnings": sorted(dict.fromkeys(severe_warnings)),
        }
        if first_material:
            entry["material_provenance"] = first_material
            entry.update({key: first_material.get(key) for key in MATERIAL_FLAT_FIELDS if key in first_material})
        if material_hashes:
            entry["material_compatibility_hashes"] = material_hashes
        if len(material_hashes) == 1 and not entry.get("material_compatibility_hash"):
            entry["material_compatibility_hash"] = material_hashes[0]
        provenance[method_id] = entry
    return provenance


def refresh_method_provenance(
    manifest: dict[str, Any],
    *,
    selected_methods: list[str] | None = None,
    runs: list[dict[str, Any]] | None = None,
    frozen_test_manifests_by_test_set: dict[str, str] | None = None,
) -> dict[str, Any]:
    provenance = build_method_provenance(
        manifest,
        selected_methods=selected_methods,
        runs=runs,
        frozen_test_manifests_by_test_set=frozen_test_manifests_by_test_set,
    )
    manifest["method_provenance"] = provenance
    method_warnings = []
    severe_warnings = []
    for method, entry in provenance.items():
        for warning in entry.get("warnings", []) or []:
            method_warnings.append(f"{method}: {warning}")
        for warning in entry.get("severe_warnings", []) or []:
            severe_warnings.append(f"{method}: {warning}")
    material_maps = material_maps_from_manifest({"method_provenance": provenance})
    material_warning = material_compatibility_warning(material_maps)
    if material_warning:
        severe_warnings.append(material_warning)
        manifest["material_compatibility_warning"] = material_warning
    manifest.update(material_maps)
    manifest["method_provenance_warnings"] = sorted(dict.fromkeys(method_warnings))
    manifest["method_provenance_severe_warnings"] = sorted(dict.fromkeys(severe_warnings))
    return manifest


def geometry_leakage_diagnostic_fields(
    *,
    pair_id: str,
    test_set: str,
    leakage_dir: Path,
    summary_path: Path,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            summary = {"scientific_status": "geometry_leakage_summary_unreadable", "parser_errors": [str(exc)]}
    status = str(summary.get("scientific_status") or "geometry_leakage_detected")
    count_fields = (
        "exact_duplicates",
        "near_duplicates",
        "aligned_near_duplicates",
        "internal_distance_near_duplicates",
        "md_neighbor_warnings",
        "atom_displacement_family_warnings",
        "random_cartesian_family_warnings",
    )
    counts = {
        key: int(summary.get(key) or 0)
        for key in count_fields
        if int(summary.get(key) or 0) > 0
    }
    details = ", ".join(f"{key}={value}" for key, value in counts.items())
    warning = f"Geometry leakage detected for {pair_id} {test_set}; scientific_status={status}"
    if details:
        warning += f"; {details}"
    warning += f"; see {leakage_dir}."
    return {
        "warning": warning,
        "scientific_status": status,
        "summary": summary,
        "severe_warnings": summary.get("severe_warnings", []) if isinstance(summary.get("severe_warnings"), list) else [],
        "warning_messages": summary.get("warning_messages", []) if isinstance(summary.get("warning_messages"), list) else [],
    }


def cross_result_name(pair_id: str, train_method: Any, test_set: Any) -> str:
    return f"{pair_id}__{method_slug_for_cross(train_method)}__on__{test_set_slug_for_cross(test_set)}"


def deduplicate_common_test_sets(test_sets: Any) -> list[str]:
    requested = [normalize_test_set_id(item) for item in (test_sets or []) if str(item).strip()]
    selected: list[str] = []
    seen: set[str] = set()
    for test_set in requested:
        if test_set in seen:
            continue
        seen.add(test_set)
        selected.append(test_set)
    return selected


def build_cross_evaluation_expected_grid(
    selected_methods: Any,
    frozen_test_sets: Any,
    *,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    methods = []
    seen_methods: set[str] = set()
    for item in selected_methods or []:
        method_id = normalize_method_id(item)
        if method_id in seen_methods:
            continue
        seen_methods.add(method_id)
        methods.append(method_id)
    test_sets = deduplicate_common_test_sets(frozen_test_sets)
    expected_cells = [
        {
            "train_method": method,
            "test_set": test_set,
            "cell_id": f"{method} on {test_set}",
        }
        for method in methods
        for test_set in test_sets
    ]
    return {
        "experiment_id": experiment_id,
        "canonical_method_ids": list(SCIENTIFIC_METHOD_REGISTRY),
        "selected_methods": methods,
        "selected_frozen_test_sets": test_sets,
        "expected_cell_count": len(expected_cells),
        "expected_cells": expected_cells,
    }


def _recipe_metadata(
    *,
    method: str,
    recipe: dict[str, Any],
    block: dict[str, Any],
    size: int,
    recipe_index: int,
    block_index: int,
) -> dict[str, Any]:
    recipe_id = str(recipe.get("recipe_id") or f"{method}_recipe_{recipe_index + 1}")
    recipe_label = str(recipe.get("label") or recipe_id)
    block_id = str(block.get("block_id") or f"{recipe_id}_block_{block_index + 1}")
    block_label = str(block.get("label") or block_id)
    generation_parameters = {
        key: value
        for key, value in block.items()
        if key not in {"block_id", "label"}
    }
    return {
        "method": method,
        "recipe_id": recipe_id,
        "recipe_label": recipe_label,
        "block_id": block_id,
        "block_label": block_label,
        "dataset_size": int(size),
        "recipe_index": recipe_index,
        "block_index": block_index,
        "generation_parameters": generation_parameters,
        "generation_parameters_json": json.dumps(
            json_safe(generation_parameters),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "seed": _optional_seed(block.get("seed", recipe.get("seed"))),
    }


def _dataset_label_from_recipe(
    method: str,
    metadata: dict[str, Any],
    size: int,
    *,
    dataset_index: int,
    used_ids: set[str],
    used_labels: set[str],
) -> str:
    label, short_id = allocate_dataset_label(
        method,
        dataset_index,
        used_ids=used_ids,
        used_labels=used_labels,
    )
    metadata["dataset_short_id"] = short_id
    metadata["dataset_label"] = label
    metadata["dataset_label_policy"] = "short_random_alnum6_v1"
    metadata["dataset_size"] = int(size)
    return label


def legacy_payload_to_dataset_recipes(
    *,
    md_sizes: list[int],
    atom_dataset_specs: list[dict[str, Any]] | None,
    atom_sizes: list[int],
    fc_dataset_specs: dict[int, list[dict[str, Any]]] | None,
    random_cartesian_options: dict[str, Any],
    selected_methods: list[str],
) -> dict[str, list[dict[str, Any]]]:
    recipes: dict[str, list[dict[str, Any]]] = {"md": [], "siesta_fc_cartesian": [], "random_cartesian": []}
    if "md" in selected_methods:
        for size in md_sizes:
            recipes["md"].append(
                {
                    "recipe_id": f"md_dataset_{size}",
                    "label": f"MD dataset {size}",
                    "blocks": [{"block_id": f"md_snapshots_{size}", "n_snapshots": int(size)}],
                }
            )
    if "siesta_fc_cartesian" in selected_methods:
        if atom_dataset_specs:
            for spec in atom_dataset_specs:
                blocks = []
                for entry in spec.get("displacements") or []:
                    displacement = str(entry.get("value", "disp"))
                    count = int(entry.get("n_structures", 0))
                    blocks.append(
                        {
                            "block_id": f"fc_{_displacement_slug(displacement)}_{count}",
                            "displacement": displacement,
                            "n_structures": count,
                        }
                    )
                recipes["siesta_fc_cartesian"].append(
                    {
                        "recipe_id": str(spec.get("label") or f"fc_dataset_{spec.get('size')}"),
                        "label": str(spec.get("label") or f"FC dataset {spec.get('size')}"),
                        "blocks": blocks,
                    }
                )
        else:
            for size in atom_sizes:
                entries = (fc_dataset_specs or {}).get(size) or []
                blocks = [
                    {
                        "block_id": f"fc_{_displacement_slug(str(entry.get('value', 'disp')))}_{int(entry.get('n_structures', 0) or 0)}",
                        "displacement": entry.get("value"),
                        "n_structures": int(entry.get("n_structures", 0) or 0),
                    }
                    for entry in entries
                ] or [{"block_id": f"fc_dataset_{size}", "n_structures": int(size)}]
                recipes["siesta_fc_cartesian"].append(
                    {
                        "recipe_id": f"fc_dataset_{size}",
                        "label": f"FC dataset {size}",
                        "blocks": blocks,
                    }
                )
    if "random_cartesian" in selected_methods:
        for size in random_cartesian_sizes_from_options(random_cartesian_options):
            block = {**random_cartesian_options, "n_structures": int(size)}
            block.pop("_dataset_specs", None)
            recipes["random_cartesian"].append(
                {
                    "recipe_id": f"rc_dataset_{size}",
                    "label": f"Random Cartesian dataset {size}",
                    "blocks": [{"block_id": f"rc_structures_{size}", **block}],
                }
            )
    return {key: value for key, value in recipes.items() if value}


def _recipe_list(raw: Any, method: str) -> list[dict[str, Any]]:
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise RuntimeError(f"dataset_recipes.{method} debe ser una lista.")
    recipes: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"dataset_recipes.{method}[{index}] debe ser un objeto.")
        blocks = item.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise RuntimeError(f"dataset_recipes.{method}[{index}].blocks debe ser una lista no vacia.")
        recipes.append(dict(item))
    return recipes


def dataset_recipes_to_execution_specs(
    raw_recipes: Any,
    *,
    selected_methods: list[str],
    split_ratios: dict[str, float],
    random_cartesian_defaults: dict[str, Any],
) -> dict[str, Any] | None:
    if raw_recipes in (None, "", {}):
        return None
    if not isinstance(raw_recipes, dict):
        raise RuntimeError("dataset_recipes debe ser un objeto con claves md/siesta_fc_cartesian/random_cartesian.")

    normalized: dict[str, list[dict[str, Any]]] = {}
    md_specs: list[dict[str, Any]] = []
    atom_specs: list[dict[str, Any]] = []
    random_specs: list[dict[str, Any]] = []
    used_dataset_ids: set[str] = set()
    used_dataset_labels: set[str] = set()

    if "md" in selected_methods:
        md_recipes = _recipe_list(raw_recipes.get("md"), "md")
        normalized["md"] = md_recipes
        for recipe_index, recipe in enumerate(md_recipes):
            block_metadata: list[dict[str, Any]] = []
            temperature_blocks: list[dict[str, Any]] = []
            for block_index, block in enumerate(recipe["blocks"]):
                if not isinstance(block, dict):
                    raise RuntimeError("Cada bloque MD debe ser un objeto.")
                size = int(block.get("n_snapshots") or block.get("n_structures") or 0)
                if size <= 0:
                    raise RuntimeError("Cada bloque MD necesita n_snapshots positivo.")
                temperature = float(block.get("temperature_K", recipe.get("temperature_K", 300.0)))
                if temperature < 0:
                    raise RuntimeError("temperature_K no puede ser negativa.")
                timestep = block.get("timestep_fs")
                if timestep not in (None, "") and float(timestep) <= 0:
                    raise RuntimeError("timestep_fs debe ser > 0.")
                metadata = _recipe_metadata(
                    method="md",
                    recipe=recipe,
                    block=block,
                    size=size,
                    recipe_index=recipe_index,
                    block_index=block_index,
                )
                block_metadata.append(metadata)
                temperature_blocks.append(
                    {
                        "block_id": metadata["block_id"],
                        "label": metadata["block_label"],
                        "n_snapshots": size,
                        "temperature_K": temperature,
                        "seed": metadata["seed"],
                        **(
                            {"timestep_fs": float(timestep)}
                            if timestep not in (None, "")
                            else {}
                        ),
                        **(
                            {"ensemble": str(block["ensemble"])}
                            if block.get("ensemble") not in (None, "")
                            else {}
                        ),
                        **(
                            {"thermostat": str(block["thermostat"])}
                            if block.get("thermostat") not in (None, "")
                            else {}
                        ),
                    }
                )
            total_size = sum(int(block["n_snapshots"]) for block in temperature_blocks)
            validate_split_sizes(total_size, split_ratios, label=f"MD recipe {recipe.get('recipe_id', recipe_index + 1)}")
            recipe_metadata = {
                "method": "md",
                "recipe_id": str(recipe.get("recipe_id") or f"md_recipe_{recipe_index + 1}"),
                "recipe_label": str(recipe.get("label") or recipe.get("recipe_id") or f"MD recipe {recipe_index + 1}"),
                "block_id": "__".join(meta["block_id"] for meta in block_metadata),
                "block_label": "__".join(meta["block_label"] for meta in block_metadata),
                "dataset_size": total_size,
                "blocks": block_metadata,
                "md_temperature_blocks": temperature_blocks,
                "generation_parameters": {"temperature_blocks": temperature_blocks},
                "generation_parameters_json": json.dumps(
                    {"temperature_blocks": temperature_blocks},
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                "seed": _optional_seed(recipe.get("seed")),
            }
            md_specs.append(
                {
                    "label": _dataset_label_from_recipe(
                        "md",
                        recipe_metadata,
                        total_size,
                        dataset_index=recipe_index,
                        used_ids=used_dataset_ids,
                        used_labels=used_dataset_labels,
                    ),
                    "size": total_size,
                    "temperature_blocks": temperature_blocks,
                    "recipe_metadata": recipe_metadata,
                }
            )

    if "siesta_fc_cartesian" in selected_methods:
        fc_recipes = _recipe_list(raw_recipes.get("siesta_fc_cartesian"), "siesta_fc_cartesian")
        normalized["siesta_fc_cartesian"] = fc_recipes
        atom_config = load_config(PIPELINES["atom_displacement"].config_path)
        limit = atom_fc_sample_limit(atom_config)
        if limit is None:
            raise RuntimeError("AtomDisplacement: FC no esta habilitado en la configuracion.")
        for recipe_index, recipe in enumerate(fc_recipes):
            entries: list[dict[str, Any]] = []
            block_metadata: list[dict[str, Any]] = []
            for block_index, block in enumerate(recipe["blocks"]):
                if not isinstance(block, dict):
                    raise RuntimeError("Cada bloque FC debe ser un objeto.")
                displacement = str(block.get("displacement") or block.get("value") or "").strip()
                if not displacement:
                    raise RuntimeError("Cada bloque FC necesita displacement.")
                count = int(block.get("n_structures") or 0)
                if count <= 0:
                    raise RuntimeError("Cada bloque FC necesita n_structures positivo.")
                if count > limit:
                    raise RuntimeError(
                        f"FC {displacement}: pide {count} estructuras, maximo {limit}."
                    )
                entries.append({"value": displacement, "n_structures": count})
                block_metadata.append(
                    _recipe_metadata(
                        method="siesta_fc_cartesian",
                        recipe=recipe,
                        block=block,
                        size=count,
                        recipe_index=recipe_index,
                        block_index=block_index,
                    )
                )
            size = sum(int(entry["n_structures"]) for entry in entries)
            validate_split_sizes(size, split_ratios, label=f"FC recipe {recipe.get('recipe_id', recipe_index + 1)}")
            recipe_metadata = {
                "method": "siesta_fc_cartesian",
                "recipe_id": str(recipe.get("recipe_id") or f"fc_recipe_{recipe_index + 1}"),
                "recipe_label": str(recipe.get("label") or recipe.get("recipe_id") or f"FC recipe {recipe_index + 1}"),
                "block_id": "__".join(meta["block_id"] for meta in block_metadata),
                "block_label": "__".join(meta["block_label"] for meta in block_metadata),
                "dataset_size": size,
                "blocks": block_metadata,
                "generation_parameters": {"displacements": entries},
                "generation_parameters_json": json.dumps(
                    {"displacements": entries},
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                "seed": _optional_seed(recipe.get("seed")),
            }
            atom_specs.append(
                {
                    "label": _dataset_label_from_recipe(
                        "fc",
                        recipe_metadata,
                        size,
                        dataset_index=recipe_index,
                        used_ids=used_dataset_ids,
                        used_labels=used_dataset_labels,
                    ),
                    "size": size,
                    "displacements": entries,
                    "recipe_metadata": recipe_metadata,
                }
            )

    if "random_cartesian" in selected_methods:
        rc_recipes = _recipe_list(raw_recipes.get("random_cartesian"), "random_cartesian")
        normalized["random_cartesian"] = rc_recipes
        for recipe_index, recipe in enumerate(rc_recipes):
            block_metadata: list[dict[str, Any]] = []
            random_blocks: list[dict[str, Any]] = []
            recipe_seed = _optional_seed(recipe.get("seed"))
            for block_index, block in enumerate(recipe["blocks"]):
                if not isinstance(block, dict):
                    raise RuntimeError("Cada bloque Random Cartesian debe ser un objeto.")
                size = int(block.get("n_structures") or 0)
                if size <= 0:
                    raise RuntimeError("Cada bloque Random Cartesian necesita n_structures positivo.")
                metadata = _recipe_metadata(
                    method="random_cartesian",
                    recipe=recipe,
                    block=block,
                    size=size,
                    recipe_index=recipe_index,
                    block_index=block_index,
                )
                block_defaults = {
                    key: value
                    for key, value in random_cartesian_defaults.items()
                    if key != "seed"
                }
                block_options = {**block_defaults, **block, "n_structures": size}
                if "seed" in block:
                    block_seed = _optional_seed(block.get("seed"))
                    if block_seed is None:
                        block_options.pop("seed", None)
                    else:
                        block_options["seed"] = block_seed
                else:
                    block_options.pop("seed", None)
                block_options["block_id"] = metadata["block_id"]
                block_options["label"] = metadata["block_label"]
                block_metadata.append(metadata)
                random_blocks.append(block_options)
            total_size = sum(int(block["n_structures"]) for block in random_blocks)
            validate_split_sizes(total_size, split_ratios, label=f"Random Cartesian recipe {recipe.get('recipe_id', recipe_index + 1)}")
            recipe_metadata = {
                "method": "random_cartesian",
                "recipe_id": str(recipe.get("recipe_id") or f"rc_recipe_{recipe_index + 1}"),
                "recipe_label": str(recipe.get("label") or recipe.get("recipe_id") or f"Random Cartesian recipe {recipe_index + 1}"),
                "block_id": "__".join(meta["block_id"] for meta in block_metadata),
                "block_label": "__".join(meta["block_label"] for meta in block_metadata),
                "dataset_size": total_size,
                "blocks": block_metadata,
                "generation_parameters": {"random_cartesian_blocks": random_blocks},
                "generation_parameters_json": json.dumps(
                    {"random_cartesian_blocks": random_blocks},
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                "seed": recipe_seed,
            }
            first_block_defaults = {
                key: value
                for key, value in (random_blocks[0] if random_blocks else {}).items()
                if key not in {
                    "block_id",
                    "label",
                    "n_structures",
                    "components",
                    "atom_displacement",
                    "bond_displacement",
                    "angle_displacement",
                    "validation",
                }
            }
            options = {
                **random_cartesian_defaults,
                **({"seed": recipe_seed} if recipe_seed is not None else {}),
                **first_block_defaults,
                "n_structures": total_size,
                "blocks": random_blocks,
            }
            random_specs.append(
                {
                    "label": _dataset_label_from_recipe(
                        "rc",
                        recipe_metadata,
                        total_size,
                        dataset_index=recipe_index,
                        used_ids=used_dataset_ids,
                        used_labels=used_dataset_labels,
                    ),
                    "size": total_size,
                    "options": options,
                    "recipe_metadata": recipe_metadata,
                }
            )

    normalized = {key: value for key, value in normalized.items() if value}
    return {
        "recipes": normalized,
        "recipe_set_hash": recipe_set_hash(normalized),
        "md_dataset_specs": md_specs,
        "atom_dataset_specs": atom_specs,
        "random_cartesian_dataset_specs": random_specs,
    }


def pipeline_keys_for_methods(method_ids: list[str]) -> list[str]:
    keys: list[str] = []
    for method_id in method_ids:
        key = METHOD_REGISTRY[method_id].pipeline_key
        if key and key not in keys:
            keys.append(key)
    return keys


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise RuntimeError(f"La configuracion debe ser un diccionario YAML: {path}")
    return config


def resolve_pipeline_path(spec: PipelineSpec, value: str | Path) -> Path:
    text = str(value)
    graph2mat_venv = os.environ.get("GRAPH2MAT_VENV", "")
    text = text.replace("${REPO_ROOT}", str(REPO_ROOT))
    if graph2mat_venv:
        text = text.replace("${GRAPH2MAT_VENV}", graph2mat_venv)
    text = os.path.expandvars(text)
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return spec.root / path


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


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


def _resolve_config_relative_path(config_path: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    text = os.path.expandvars(str(value)).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def lightning_training_progress_from_config(config_path: Path) -> str | None:
    """Return a compact progress summary from Lightning TensorBoard events."""
    try:
        config = load_config(config_path)
        training_dir = _resolve_config_relative_path(
            config_path,
            (config.get("paths") or {}).get("training_dir"),
        )
        if training_dir is None:
            return None
        training_config = load_config(training_dir / "config.yaml")
        max_epochs = (training_config.get("trainer") or {}).get("max_epochs")
        log_root = training_dir / "lightning_logs"
        event_files = sorted(
            log_root.rglob("events.out.tfevents.*"),
            key=lambda path: path.stat().st_mtime,
        )
        if not event_files:
            return None
        from tensorboard.backend.event_processing import event_accumulator

        accumulator = event_accumulator.EventAccumulator(
            str(event_files[-1]),
            size_guidance={"scalars": 0},
        )
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
        train_step, train_loss = latest_scalar("train_loss_epoch")
        val_step, val_loss = latest_scalar("val_loss")
        _, val_node = latest_scalar("val_node_smooth_l1")
        _, val_edge = latest_scalar("val_edge_smooth_l1")
        _, beta = latest_scalar("val_smooth_l1_beta")
        step = max([item for item in (epoch_step, train_step, val_step) if item is not None], default=None)
        parts = []
        if epoch_value is not None:
            epoch_display = str(int(epoch_value))
            if max_epochs not in (None, ""):
                epoch_display += f"/{max_epochs}"
            parts.append(f"epoch {epoch_display}")
        if step is not None:
            parts.append(f"step {step}")
        if train_loss is not None:
            parts.append(f"train_loss {_format_progress_value(train_loss)}")
        if val_loss is not None:
            parts.append(f"val_loss {_format_progress_value(val_loss)}")
        if val_node is not None:
            parts.append(f"val_node {_format_progress_value(val_node)}")
        if val_edge is not None:
            parts.append(f"val_edge {_format_progress_value(val_edge)}")
        if beta is not None:
            parts.append(f"beta {_format_progress_value(beta)}")
        checkpoints = sorted(
            log_root.rglob("checkpoints/*.ckpt"),
            key=lambda path: path.stat().st_mtime,
        )
        if checkpoints:
            parts.append(f"ckpt {checkpoints[-1].name}")
        return " | ".join(parts) if parts else None
    except Exception:
        return None


def stream_process_output(
    process: subprocess.Popen[str],
    append: Any,
    *,
    label: str,
    master_fd: int | None = None,
    eta_provider: Any | None = None,
    progress_provider: Any | None = None,
) -> int:
    if master_fd is None:
        assert process.stdout is not None
        started_at = time.time()
        last_output = started_at
        last_heartbeat = last_output
        while True:
            line = process.stdout.readline()
            if line:
                last_output = time.time()
                append(line)
            elif process.poll() is not None:
                break

            now = time.time()
            if process.poll() is not None:
                break
            if now - last_output >= LOG_HEARTBEAT_SECONDS and now - last_heartbeat >= LOG_HEARTBEAT_SECONDS:
                eta_seconds = eta_provider() if eta_provider is not None else None
                progress_text = progress_provider() if progress_provider is not None else None
                progress_fragment = f" | progreso {progress_text}" if progress_text else ""
                append(
                    "[UI] "
                    f"{label} sigue ejecutandose | PID {process.pid} | "
                    f"elapsed {format_duration(now - started_at)} | "
                    f"sin nueva salida {format_duration(now - last_output)} | "
                    f"ETA {format_duration(eta_seconds)}"
                    f"{progress_fragment}\n"
                )
                last_heartbeat = now
        return process.wait()
    else:
        fd = master_fd

    pending = ""
    started_at = time.time()
    last_output = started_at
    last_heartbeat = last_output
    while True:
        ready, _, _ = select.select([fd], [], [], 1.0)
        if ready:
            try:
                chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
            except OSError:
                chunk = ""
            if chunk:
                last_output = time.time()
                pending += chunk.replace("\r", "\n")
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    append(line + "\n")
            elif process.poll() is not None:
                break

        now = time.time()
        if process.poll() is not None:
            break
        if now - last_output >= LOG_HEARTBEAT_SECONDS and now - last_heartbeat >= LOG_HEARTBEAT_SECONDS:
            eta_seconds = eta_provider() if eta_provider is not None else None
            progress_text = progress_provider() if progress_provider is not None else None
            progress_fragment = f" | progreso {progress_text}" if progress_text else ""
            append(
                "[UI] "
                f"{label} sigue ejecutandose | PID {process.pid} | "
                f"elapsed {format_duration(now - started_at)} | "
                f"sin nueva salida {format_duration(now - last_output)} | "
                f"ETA {format_duration(eta_seconds)}"
                f"{progress_fragment}\n"
            )
            last_heartbeat = now

    if pending:
        append(pending)
    return process.wait()


class PipelineRunner:
    def __init__(self, spec: PipelineSpec) -> None:
        self.spec = spec
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._logs: list[str] = []
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._returncode: int | None = None
        self._command: list[str] | None = None

    def start(self) -> dict[str, Any]:
        config = load_config(self.spec.config_path)
        venv_activate = resolve_pipeline_path(self.spec, config["paths"]["venv_activate"])
        if not venv_activate.exists():
            raise RuntimeError(
                f"{self.spec.label}: no se encontro el entorno virtual: {venv_activate}"
            )

        shell = str(config.get("commands", {}).get("shell", "bash"))
        python = str(config.get("commands", {}).get("python", "python"))
        shell_command = (
            f"source {shlex.quote(str(venv_activate))} "
            f"&& {shlex.quote(python)} {shlex.quote(str(self.spec.main_script))}"
        )

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"{self.spec.label}: el pipeline ya se esta ejecutando.")
            self._logs = [
                f"[UI] Ejecutando {self.spec.label}: {self.spec.main_script}\n",
                f"[UI] Root: {self.spec.root}\n",
                f"[UI] Config: {self.spec.config_path}\n",
                "[UI] ETA: sin estimacion inicial para ejecuciones individuales.\n",
            ]
            self._started_at = time.time()
            self._finished_at = None
            self._returncode = None
            self._command = [shell, "-lc", shell_command]
            master_fd: int | None = None
            if pty is not None:
                master_fd, slave_fd = pty.openpty()
                self._process = subprocess.Popen(
                    self._command,
                    cwd=self.spec.root,
                    stdin=subprocess.DEVNULL,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                os.close(slave_fd)
            else:
                self._process = subprocess.Popen(
                    self._command,
                    cwd=self.spec.root,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            process = self._process
            self._logs.append(f"[UI] PID: {process.pid}\n")
            self._logs.append(f"[RUN] {' '.join(self._command)}\n")

        threading.Thread(target=self._collect_output, args=(process, master_fd), daemon=True).start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return self.status()
            process.terminate()
            self._logs.append("\n[UI] Solicitud de parada enviada.\n")
        return self.status()

    def _collect_output(self, process: subprocess.Popen[str], master_fd: int | None) -> None:
        try:
            returncode = stream_process_output(
                process,
                lambda line: self._append_log(line),
                label=self.spec.label,
                master_fd=master_fd,
                progress_provider=lambda: lightning_training_progress_from_config(self.spec.config_path),
            )
        finally:
            if master_fd is not None:
                os.close(master_fd)
        with self._lock:
            self._returncode = returncode
            self._finished_at = time.time()
            elapsed = self._finished_at - (self._started_at or self._finished_at)
            if self._process is process:
                self._process = None
            self._logs.append(
                f"\n[UI] {self.spec.label} finalizado con codigo {returncode} "
                f"en {format_duration(elapsed)}.\n"
            )

    def _append_log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return {
                "key": self.spec.key,
                "label": self.spec.label,
                "running": running,
                "returncode": None if running else self._returncode,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "command": self._command,
                "elapsed_seconds": None
                if self._started_at is None
                else (time.time() if running else self._finished_at or time.time()) - self._started_at,
                "eta_seconds": None,
                "log_size": len(self._logs),
            }

    def logs(self, since: int = 0, limit: int | None = DEFAULT_LOG_RESPONSE_LIMIT) -> dict[str, Any]:
        with self._lock:
            log_payload = bounded_log_payload(self._logs, since=since, limit=limit)
            log_payload["status"] = self.status()
            return {
                **log_payload,
            }


RUNNERS = {key: PipelineRunner(spec) for key, spec in PIPELINES.items()}


@dataclass(frozen=True)
class ScriptRunnerSpec:
    key: str
    label: str
    script_path: Path
    default_payload: Path
    default_manifest: Path


MIXING_E2E_RUNNER_SPEC = ScriptRunnerSpec(
    key="mixing_e2e",
    label="MixingE2E",
    script_path=REPO_ROOT / "Comparison" / "scripts" / "run_mixing_e2e_payload_once.py",
    default_payload=REPO_ROOT / "Comparison" / "config" / "ml_vs_siesta_mixing_e2e_20_50_80_payload.json",
    default_manifest=REPO_ROOT / "Comparison" / "results" / "ml_vs_siesta_mixing_e2e_20_50_80_manifest.json",
)


class ScriptPayloadRunner:
    def __init__(self, spec: ScriptRunnerSpec) -> None:
        self.spec = spec
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._logs: list[str] = []
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._returncode: int | None = None
        self._command: list[str] | None = None
        self._payload_path: Path | None = None
        self._manifest_path: Path | None = None

    def start(
        self,
        *,
        payload_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        poll_seconds: float = 30.0,
    ) -> dict[str, Any]:
        payload = Path(payload_path).expanduser() if payload_path else self.spec.default_payload
        manifest = Path(manifest_path).expanduser() if manifest_path else self.spec.default_manifest
        if not payload.is_absolute():
            payload = REPO_ROOT / payload
        if not manifest.is_absolute():
            manifest = REPO_ROOT / manifest
        if not payload.exists():
            raise RuntimeError(f"{self.spec.label}: payload no encontrado: {payload}")
        python = DEFAULT_VENV_PYTHON if DEFAULT_VENV_PYTHON.exists() else Path(sys.executable)
        self._logs = [
            f"[UI] Ejecutando {self.spec.label}: {self.spec.script_path}\n",
            f"[UI] Payload: {payload}\n",
            f"[UI] Manifest: {manifest}\n",
            f"[UI] Poll seconds: {poll_seconds}\n",
            "[UI] ETA: sin estimacion inicial para ejecuciones individuales.\n",
        ]
        self._started_at = time.time()
        self._finished_at = None
        self._returncode = None
        self._payload_path = payload
        self._manifest_path = manifest
        self._command = [
            str(python),
            str(self.spec.script_path),
            str(payload),
            "--manifest-json",
            str(manifest),
            "--poll-seconds",
            str(float(poll_seconds)),
        ]
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"{self.spec.label}: ya se esta ejecutando.")
            master_fd: int | None = None
            if pty is not None:
                master_fd, slave_fd = pty.openpty()
                self._process = subprocess.Popen(
                    self._command,
                    cwd=REPO_ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                os.close(slave_fd)
            else:
                self._process = subprocess.Popen(
                    self._command,
                    cwd=REPO_ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            process = self._process
            self._logs.append(f"[UI] PID: {process.pid}\n")
            self._logs.append(f"[RUN] {' '.join(self._command)}\n")
        threading.Thread(target=self._collect_output, args=(process, master_fd), daemon=True).start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return self.status()
            process.terminate()
            self._logs.append("\n[UI] Solicitud de parada enviada.\n")
        return self.status()

    def _collect_output(self, process: subprocess.Popen[str], master_fd: int | None) -> None:
        try:
            returncode = stream_process_output(
                process,
                lambda line: self._append_log(line),
                label=self.spec.label,
                master_fd=master_fd,
            )
        finally:
            if master_fd is not None:
                os.close(master_fd)
        with self._lock:
            self._returncode = returncode
            self._finished_at = time.time()
            elapsed = self._finished_at - (self._started_at or self._finished_at)
            if self._process is process:
                self._process = None
            self._logs.append(
                f"\n[UI] {self.spec.label} finalizado con codigo {returncode} "
                f"en {format_duration(elapsed)}.\n"
            )

    def _append_log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return {
                "key": self.spec.key,
                "label": self.spec.label,
                "running": running,
                "returncode": None if running else self._returncode,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "command": self._command,
                "payload_path": str(self._payload_path) if self._payload_path else None,
                "manifest_path": str(self._manifest_path) if self._manifest_path else None,
                "elapsed_seconds": None
                if self._started_at is None
                else (time.time() if running else self._finished_at or time.time()) - self._started_at,
                "eta_seconds": None,
                "log_size": len(self._logs),
            }

    def logs(self, since: int = 0, limit: int | None = DEFAULT_LOG_RESPONSE_LIMIT) -> dict[str, Any]:
        with self._lock:
            log_payload = bounded_log_payload(self._logs, since=since, limit=limit)
            log_payload["status"] = self.status()
            return {
                **log_payload,
            }


MIXING_E2E_RUNNER = ScriptPayloadRunner(MIXING_E2E_RUNNER_SPEC)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(length).decode("utf-8")
    if not body:
        return {}
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError("El cuerpo JSON debe ser un objeto.")
    return parsed


def parse_sizes(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, list):
        raw_items = value
    else:
        raise RuntimeError("Los tamaños deben enviarse como lista o texto separado por comas.")

    sizes = []
    for item in raw_items:
        if item == "":
            continue
        size = int(item)
        if size <= 0:
            raise RuntimeError("Los tamaños de dataset deben ser mayores que cero.")
        sizes.append(size)
    if not sizes:
        raise RuntimeError("Define al menos un tamaño de dataset.")
    return sizes


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False, allow_unicode=False),
        encoding="utf-8",
    )


MATERIAL_BUNDLE_PAYLOAD_KEYS = (
    "preset",
    "label",
    "root_dir",
    "fdf",
    "pseudopotential_dir",
    "basis_dir",
    "structure_type",
)


def repository_display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_material_payload(value: Any, *, required: bool = False) -> dict[str, Any] | None:
    if value in (None, ""):
        if required:
            raise RuntimeError("Define material.preset o un bundle material completo.")
        return None
    if not isinstance(value, dict):
        raise RuntimeError("material debe ser un objeto JSON.")
    mode = str(value.get("mode") or "").strip()
    material: dict[str, Any] = {}
    if mode == "preset" or value.get("preset") not in (None, ""):
        preset = str(value.get("preset") or "").strip()
        if not preset:
            raise RuntimeError("material.preset no puede estar vacio.")
        material["preset"] = preset
        return material
    for key in MATERIAL_BUNDLE_PAYLOAD_KEYS:
        if key == "preset":
            continue
        item = value.get(key)
        if item not in (None, ""):
            material[key] = item
    if not material:
        if required:
            raise RuntimeError("Define material.preset o un bundle material completo.")
        return None
    return material


def validate_material_payload(
    value: Any,
    *,
    base_dir: Path = REPO_ROOT,
    required: bool = True,
) -> dict[str, Any]:
    material = parse_material_payload(value, required=required)
    if material is None:
        return {"ok": True, "material": None, "warnings": []}
    resolved = resolve_material_bundle(
        {"material": material},
        base_dir=base_dir,
        allow_legacy_default=False,
        allow_absolute_paths=True,
    )
    manifest = resolved.to_manifest_dict()
    atom_count = len(extract_coordinate_species_indices(resolved.validated.bundle.fdf))
    manifest["atom_count"] = atom_count
    warnings = [
        str(item)
        for item in (manifest.get("warning"),)
        if item not in (None, "")
    ]
    if manifest.get("absolute_paths_used"):
        warnings.append(
            "El bundle usa rutas absolutas; quedan registradas para trazabilidad, "
            "pero no son portables entre maquinas."
        )
    return {
        "ok": True,
        "material": manifest,
        "material_config": material,
        "warnings": warnings,
        "species": manifest.get("species", []),
        "atom_count": atom_count,
        "pseudopotentials": manifest.get("pseudopotentials", {}),
        "basis_files": manifest.get("basis_file_sha256", {}),
    }


def material_validation_response(value: Any, *, base_dir: Path = REPO_ROOT) -> dict[str, Any]:
    try:
        return validate_material_payload(value, base_dir=base_dir, required=True)
    except (MaterialBundleError, RuntimeError, OSError) as exc:
        return {
            "ok": False,
            "message": str(exc),
            "warnings": [],
            "species": [],
            "pseudopotentials": {},
            "basis_files": {},
        }


def material_presets_payload(*, preset_dir: Path = MATERIAL_PRESET_DIR) -> dict[str, Any]:
    presets: list[dict[str, Any]] = []
    if preset_dir.exists():
        preset_paths = sorted(preset_dir.glob("*/material.yaml"))
    else:
        preset_paths = []
    for path in preset_paths:
        name = path.parent.name
        item: dict[str, Any] = {
            "name": name,
            "path": repository_display_path(path),
            "valid": False,
        }
        try:
            validation = validate_material_payload({"preset": name}, required=True)
            material = validation.get("material") or {}
            item.update(
                {
                    "valid": True,
                    "label": material.get("label"),
                    "structure_type": material.get("structure_type"),
                    "species": material.get("species", []),
                    "warnings": validation.get("warnings", []),
                }
            )
        except (MaterialBundleError, RuntimeError, OSError) as exc:
            item["error"] = str(exc)
        presets.append(item)
    return {
        "presets": presets,
        "default_preset": "h2o",
        "preset_dir": repository_display_path(preset_dir),
    }


def apply_material_to_config(config: dict[str, Any], material: dict[str, Any] | None) -> None:
    if material is not None:
        config["material"] = dict(material)
        md = config.get("md")
        if isinstance(md, dict):
            preset = str(material.get("preset") or "").strip().lower()
            if preset == "graphene":
                md["run_fdf_template"] = "${REPO_ROOT}/materials/graphene/RUN.fdf"
            else:
                md.pop("run_fdf_template", None)


def resolve_venv_activate_from_command(command: str) -> str:
    text = str(command).strip()
    if not text:
        raise RuntimeError("El comando de activacion del entorno virtual esta vacio.")
    if text.startswith("source "):
        text = text[len("source ") :].strip()
    if text.startswith(". "):
        text = text[2:].strip()
    text = text.strip("\"'")
    if not text:
        raise RuntimeError("No se pudo extraer la ruta del entorno virtual desde el comando.")
    if "${GRAPH2MAT_VENV}" in text and not os.environ.get("GRAPH2MAT_VENV"):
        raise RuntimeError(
            "GRAPH2MAT_VENV debe estar definido para usarlo como ruta del entorno virtual."
        )
    if (
        not Path(os.path.expandvars(text.replace("${REPO_ROOT}", str(REPO_ROOT)))).expanduser().is_absolute()
        and not text.startswith("${GRAPH2MAT_VENV}")
    ):
        text = "${REPO_ROOT}/" + text
    resolved_text = os.path.expandvars(text.replace("${REPO_ROOT}", str(REPO_ROOT)))
    resolved_path = Path(resolved_text).expanduser()
    return text


def apply_venv_activate_to_pipeline_configs(command: str | None) -> str:
    effective_command = str(command).strip() if command is not None else DEFAULT_VENV_ACTIVATE_COMMAND
    if not effective_command:
        effective_command = DEFAULT_VENV_ACTIVATE_COMMAND
    venv_activate_path = resolve_venv_activate_from_command(effective_command)
    for spec in PIPELINES.values():
        config = load_config(spec.config_path)
        paths = config.setdefault("paths", {})
        if not isinstance(paths, dict):
            raise RuntimeError(f"{spec.label}: 'paths' debe ser un diccionario en {spec.config_path}.")
        paths["venv_activate"] = venv_activate_path
        write_yaml(spec.config_path, config)
    return venv_activate_path


def write_csv_dicts(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["sample_id", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_digest(paths_to_hash: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in paths_to_hash if path.exists() and path.is_file()}):
        digest.update(str(path).encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update((file_sha256(path) or "").encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def files_content_digest(paths_to_hash: list[Path]) -> str:
    entries = [
        (path.name, file_sha256(path) or "")
        for path in paths_to_hash
        if path.exists() and path.is_file()
    ]
    if not entries:
        return ""
    digest = hashlib.sha256()
    for name, sha in sorted(entries):
        digest.update(name.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty_warning() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "Could not determine git dirty state."
    return "Working tree has uncommitted changes." if result.stdout.strip() else ""


def command_version(command_name: str) -> str:
    candidates = ([command_name, "--version"], [command_name, "-V"])
    for command in candidates:
        try:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except Exception:
            continue
        output = (result.stdout or result.stderr or "").strip()
        if result.returncode == 0 and output:
            return output.splitlines()[0]
    return "unavailable"


def python_module_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:
        return "unavailable"
    return str(getattr(module, "__version__", "unavailable"))


def environment_versions(configs: list[dict[str, Any]] | None = None) -> dict[str, str]:
    configs = configs or []
    siesta = "siesta"
    graph2mat = "graph2mat"
    for config in configs:
        commands = config.get("commands", {}) if isinstance(config, dict) else {}
        siesta = str(commands.get("siesta", siesta))
        graph2mat = str(commands.get("graph2mat", graph2mat))
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit() or "unavailable",
        "siesta_version": command_version(siesta),
        "graph2mat_version": command_version(graph2mat),
        "sisl_version": python_module_version("sisl"),
    }


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def absolute_path_warning(paths_to_check: list[Path]) -> str:
    home = str(Path.home())
    portable_roots = [REPO_ROOT]
    matches = [
        str(path)
        for path in paths_to_check
        if path.is_absolute()
        and (str(path).startswith(home) or str(path).startswith("/mnt/"))
        and not any(_path_is_relative_to(path, root) for root in portable_roots)
    ]
    matches = sorted(dict.fromkeys(matches))
    return "User-local absolute paths detected: " + ", ".join(matches) if matches else ""


def sanitize_reproducibility_warning(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    prefix = "User-local absolute paths detected:"
    if not value.startswith(prefix):
        return value
    paths = [Path(part.strip()) for part in value.removeprefix(prefix).split(",") if part.strip()]
    return absolute_path_warning(paths)


def sanitize_recommendation_warnings(recommendation: dict[str, Any]) -> dict[str, Any]:
    warnings = recommendation.get("severe_warnings")
    if isinstance(warnings, list):
        sanitized = [
            warning
            for warning in (sanitize_reproducibility_warning(item) for item in warnings)
            if warning
        ]
        recommendation = {**recommendation, "severe_warnings": sanitized}
    return recommendation


def sample_set_hash(sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(set(sample_ids)):
        digest.update(sample_id.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()


def experiment_root(run_id: str) -> Path:
    return RESULTS_ROOT / run_id


def experiment_manifest_path(run_id: str) -> Path:
    return experiment_root(run_id) / "experiment_manifest.yaml"


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_matching_files(source_root: Path, pattern: str, destination_root: Path) -> int:
    count = 0
    for src in sorted(source_root.glob(pattern)):
        if not src.is_file():
            continue
        dst = destination_root / src.relative_to(source_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
    return count


def copy_selected_reference_files(source_root: Path, destination_root: Path) -> int:
    if not source_root.exists():
        return 0
    count = 0
    for sample_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        selection = strict_choose_reference_matrix(sample_dir)
        if not selection.ok or selection.path is None:
            continue
        dst = destination_root / sample_dir.name / selection.path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selection.path, dst)
        count += 1
    return count


DEFAULT_SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
DEFAULT_MD_SPLIT_MODE = "blocked_with_gap"
# 30 frames @ 1 fs/frame ≈ one carbon vibrational period (~20-40 fs). A gap of
# 1 frame (1 fs) is physically meaningless for decorrelating MD trajectory
# frames between train/validation/test; datasets generated with the old gap=1
# are documented as a known limitation in docs/known_limitations.md.
DEFAULT_MD_TEMPORAL_GAP = 30
DEFAULT_MD_BLOCK_ORDER = "train,validation,test"
MD_SPREAD_SPLIT_WARNING = (
    "MD split_mode=spread is exploratory/debug only because temporally adjacent "
    "trajectory frames can cross train/validation/test."
)
DEFAULT_COMMON_TEST_SETS = ["test_md", "test_siesta_fc_cartesian", "test_mixed"]
DEFAULT_PRIMARY_METRIC = "low_energy_rmse_eV"
MINIMUM_ROBUST_SEEDS = 3
STRICT_COMPARISON_MODE = True
SPLIT_MANIFEST_FIELDS = [
    "sample_id",
    "method",
    "source_run",
    "frame_index",
    "time_index",
    "displacement_amplitude",
    "displacement_magnitude",
    "displaced_atom",
    "displacement_axis",
    "displacement_sign",
    "displacement_family",
    "structure_path",
    "hamiltonian_path",
    "output_path",
    "run_out_path",
    "metadata_path",
    "valid",
    "validation_reason",
    "split",
    "split_group_id",
    "split_group_fields",
    "split_strategy",
    "random_cartesian_family_id",
    "base_geometry_hash",
    "distribution",
    "sigma_ang",
    "uniform_range_ang",
    "seed_family",
    "move_atoms",
    "species_filter",
    "recipe_id",
    "block_id",
    "seed",
    "status",
    "sample_dir",
]


def split_ratios_from_config(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("split_ratios") or config.get("splits") or {}
    ratios = {
        "train": float(raw.get("train", DEFAULT_SPLIT_RATIOS["train"])),
        "validation": float(
            raw.get("validation", raw.get("val", DEFAULT_SPLIT_RATIOS["validation"]))
        ),
        "test": float(raw.get("test", DEFAULT_SPLIT_RATIOS["test"])),
    }
    if any(value <= 0 or value >= 1 for value in ratios.values()):
        return dict(DEFAULT_SPLIT_RATIOS)
    return ratios


def parse_split_ratios(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError("Los splits deben enviarse como un objeto.")
    ratios = {
        "train": float(value.get("train", DEFAULT_SPLIT_RATIOS["train"])),
        "validation": float(
            value.get("validation", value.get("val", DEFAULT_SPLIT_RATIOS["validation"]))
        ),
        "test": float(value.get("test", DEFAULT_SPLIT_RATIOS["test"])),
    }
    return ratios


def validate_split_sizes(
    dataset_size: int,
    splits: dict[str, float],
    *,
    label: str = "dataset",
) -> dict[str, int]:
    if dataset_size < 3:
        raise RuntimeError(
            f"{label}: el dataset debe tener al menos 3 estructuras para que train, "
            "validation y test no queden vacios."
        )
    ratios = {
        "train": float(splits["train"]),
        "validation": float(splits.get("validation", splits.get("val", 0.0))),
        "test": float(splits["test"]),
    }
    if any(value <= 0 for value in ratios.values()):
        raise RuntimeError("Los ratios de split deben ser positivos.")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(
            "Los ratios de split deben sumar 1.0 "
            f"(recibido: {sum(ratios.values()):.6g})."
        )

    raw = {key: dataset_size * ratio for key, ratio in ratios.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = dataset_size - sum(counts.values())
    order = sorted(
        counts,
        key=lambda key: (raw[key] - counts[key], ratios[key]),
        reverse=True,
    )
    for key in order[:remainder]:
        counts[key] += 1

    empty = [key for key, count in counts.items() if count < 1]
    if empty:
        raise RuntimeError(
            f"{label}: split invalido; cada particion debe tener al menos 1 estructura. "
            f"dataset_size={dataset_size}, ratios={ratios}, counts={counts}, "
            f"vacios={empty}."
        )
    return counts


def _archived_temporal_gap_from_items(items: list[dict[str, Any]]) -> int | None:
    """Temporal gap recorded in reused split-manifest rows, if unambiguous."""
    gaps: set[int] = set()
    for item in items:
        row = item.get("row") or {}
        value = str(row.get("temporal_gap") or "").strip()
        if not value:
            continue
        try:
            gaps.add(int(float(value)))
        except ValueError:
            return None
    if len(gaps) == 1:
        return gaps.pop()
    return None


def md_split_counts_for_mode(
    dataset_size: int,
    splits: dict[str, float],
    *,
    split_mode: str,
    label: str,
    temporal_gap: int | None = None,
) -> tuple[dict[str, int], int]:
    """Return train/validation/test counts plus frames reserved as temporal gaps.

    ``temporal_gap=None`` uses the scientific default; reused archived datasets
    pass their recorded gap so a rebuild preserves the archived structure.
    """
    if split_mode != DEFAULT_MD_SPLIT_MODE:
        return validate_split_sizes(dataset_size, splits, label=label), 0

    gap = DEFAULT_MD_TEMPORAL_GAP if temporal_gap is None else int(temporal_gap)
    reserved_gap_frames = 2 * gap
    usable_size = dataset_size - reserved_gap_frames
    if usable_size < 3:
        raise RuntimeError(
            f"{label}: blocked_with_gap con temporal_gap={gap} necesita al menos "
            f"{3 + reserved_gap_frames} frames MD para mantener train, validation "
            "y test no vacios."
        )
    return validate_split_sizes(usable_size, splits, label=label), reserved_gap_frames


def atom_fc_sample_limit(config: dict[str, Any]) -> int | None:
    structure = config.get("structure", {})
    force_constants = structure.get("force_constants") or {}
    if not bool(force_constants.get("enabled", False)):
        return None

    atoms = structure.get("atoms") or []
    first_atom = int(force_constants.get("first_atom", 1))
    last_atom = force_constants.get("last_atom")
    last_atom = len(atoms) if last_atom is None else int(last_atom)
    if first_atom < 1 or last_atom < first_atom or last_atom > len(atoms):
        raise RuntimeError(
            "AtomDisplacement: rango FC invalido. "
            f"FC.First={first_atom}, FC.Last={last_atom}, NumberOfAtoms={len(atoms)}."
        )
    displaced_atoms = last_atom - first_atom + 1
    include_reference = bool(force_constants.get("include_reference", True))
    return (1 if include_reference else 0) + 6 * displaced_atoms


def atom_fc_displacement_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    force_constants = config.get("structure", {}).get("force_constants", {}) or {}
    entries = force_constants.get("displacements")
    if isinstance(entries, dict):
        return [
            {"value": key, "structure_options": value}
            for key, value in sorted(entries.items(), key=lambda item: _sort_displacement_key(item[0]))
        ]
    if entries:
        return [entry if isinstance(entry, dict) else {"value": entry} for entry in entries]
    return [{"value": force_constants.get("displacement", "0.05 Ang")}]


def parse_fc_displacements(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise RuntimeError("Las magnitudes FC deben enviarse como una lista.")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, dict):
            displacement = str(item.get("value", "")).strip()
            entry = {"value": displacement}
            if item.get("label"):
                entry["label"] = str(item["label"])
            if item.get("n_structures") not in (None, ""):
                entry["n_structures"] = int(item["n_structures"])
        else:
            displacement = str(item).strip()
            entry = {"value": displacement}
        if not displacement:
            raise RuntimeError(f"La magnitud FC #{index} no tiene valor.")
        entries.append(entry)
    return entries or None


def parse_structures_per_displacement(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, list):
        raw_items = value
    else:
        raise RuntimeError("structures_per_displacement debe ser lista o texto separado por comas.")
    counts: list[int] = []
    for item in raw_items:
        if item == "":
            continue
        count = int(item)
        if count <= 0:
            raise RuntimeError("structures_per_displacement solo acepta enteros positivos.")
        counts.append(count)
    return counts or None


def _sort_displacement_key(value: str) -> tuple[float, str]:
    text = str(value).strip()
    number = ""
    for char in text:
        if char.isdigit() or char in ".-+Ee":
            number += char
        elif number:
            break
    try:
        return (float(number), text)
    except ValueError:
        return (math.inf, text)


def _displacement_slug(value: str) -> str:
    text = str(value).strip()
    number = ""
    for char in text:
        if char.isdigit() or char in ".-+Ee":
            number += char
        elif number:
            break
    text = number or text
    slug = (
        text.replace("+", "")
        .replace("-", "m")
        .replace(".", "p")
        .replace(" ", "")
        .replace("/", "_")
    )
    return "".join(char if char.isalnum() else "_" for char in slug).strip("_") or "disp"


def parse_fc_displacement_options(value: Any) -> dict[str, list[int]] | None:
    """Parse the FC mapping: displacement -> user-defined counts.

    The API accepts a JSON object such as ``{"0.02 Ang": [5, 7], "0.05 Ang": [2]}``.
    Keeping the mapping separate from ``structures_per_displacement`` preserves
    backward compatibility with the older uniform grid. The mapping can be
    consumed in strict index-based ``aligned`` mode or opt-in ``cartesian`` mode.
    """

    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError("displacement_options debe ser un objeto displacement -> lista.")
    parsed: dict[str, list[int]] = {}
    for raw_key, raw_counts in value.items():
        displacement = str(raw_key).strip()
        if not displacement:
            raise RuntimeError("displacement_options contiene una magnitud vacia.")
        counts = parse_structures_per_displacement(raw_counts)
        if not counts:
            raise RuntimeError(f"{displacement}: define al menos un numero de estructuras.")
        parsed[displacement] = list(counts)
    return dict(sorted(parsed.items(), key=lambda item: _sort_displacement_key(item[0])))


def parse_max_datasets(value: Any, default: int = 100) -> int:
    if value in (None, ""):
        return default
    limit = int(value)
    if limit <= 0:
        raise RuntimeError("max_datasets debe ser mayor que cero.")
    return limit


def parse_combination_mode(value: Any) -> str:
    mode = "aligned" if value in (None, "") else str(value).strip().lower()
    if mode not in {"aligned", "cartesian"}:
        raise RuntimeError(
            "combination_mode debe ser 'aligned' o 'cartesian' "
            f"(recibido: {value!r})."
        )
    return mode


def parse_split_mode(value: Any) -> str:
    mode = DEFAULT_MD_SPLIT_MODE if value in (None, "") else str(value).strip().lower()
    if mode not in {"block", "spread", "blocked_with_gap"}:
        raise RuntimeError(
            "split_mode debe ser 'block', 'spread' o 'blocked_with_gap' "
            f"(recibido: {value!r})."
        )
    return mode


def parse_test_sets(value: Any) -> list[str]:
    if value in (None, ""):
        return list(DEFAULT_COMMON_TEST_SETS)
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, list):
        raw_items = [str(item).strip() for item in value]
    else:
        raise RuntimeError("test_sets debe ser lista o texto separado por comas.")
    selected = [normalize_test_set_id(item) for item in raw_items if item]
    allowed = set(DEFAULT_COMMON_TEST_SETS) | {
        "test_siesta_fc_cartesian",
        "test_random_cartesian",
    }
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise RuntimeError(f"test_sets contiene valores no soportados: {unknown}.")
    return deduplicate_common_test_sets(selected) or list(DEFAULT_COMMON_TEST_SETS)


def parse_compute_budget_mode(value: Any) -> str:
    mode = "both" if value in (None, "") else str(value).strip().lower()
    allowed = {"equal_sample_count", "equal_siesta_budget", "both"}
    if mode not in allowed:
        raise RuntimeError(
            "compute_budget_mode debe ser equal_sample_count, equal_siesta_budget o both "
            f"(recibido: {value!r})."
        )
    return mode


def parse_compute_accelerator(value: Any) -> str:
    accelerator = "cpu" if value in (None, "") else str(value).strip().lower()
    allowed = {"cpu", "gpu", "auto"}
    if accelerator not in allowed:
        raise RuntimeError(
            "compute_accelerator debe ser cpu, gpu o auto "
            f"(recibido: {value!r})."
        )
    return accelerator


def apply_training_accelerator(config: dict[str, Any], accelerator: str) -> None:
    config.setdefault("training", {}).setdefault("trainer", {})["accelerator"] = accelerator


DEFAULT_PERFORMANCE_SETTINGS: dict[str, Any] = {
    "max_parallel_siesta_jobs": 1,
    "max_parallel_dataset_jobs": 1,
    "max_parallel_prediction_jobs": 1,
    "max_parallel_evaluation_jobs": 1,
    "max_parallel_metric_jobs": 1,
    "max_parallel_graph2mat_training_jobs": 1,
    "max_parallel_deeph_training_jobs": 1,
    "omp_num_threads": None,
    "mkl_num_threads": None,
    "openblas_num_threads": None,
    "numexpr_num_threads": None,
    "torch_num_threads": None,
    "compute_accelerator": "cpu",
    "batch_size": None,
    "store_in_memory": None,
    "reuse_validated_siesta_outputs": True,
    "enable_experiment_cache": False,
    "error_policy": "fail_fast",
    "preset": None,
    "torch_float32_matmul_precision": None,
    "torch_mixed_precision": None,
    "graph2mat_log_every_n_steps": None,
    "graph2mat_check_val_every_n_epoch": None,
    "graph2mat_checkpoint_every_n_epochs": None,
    "graph2mat_require_cuequivariance": False,
}

PERFORMANCE_PRESET_IDS = {
    "soft",
    "balanced",
    "aggressive",
    "aggressive_local",
    "gpu_focused",
    "parallel_trains",
    "gpu_only",
    "cpu_only",
    "max_aggressive",
    "stress",
    "auto_detect",
    "single_run_debug",
}


def _read_meminfo_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                return round(int(parts[1]) / (1024 * 1024), 2)
    return None


def _physical_cpu_cores() -> int | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None
    cores: set[tuple[str, str]] = set()
    current: dict[str, str] = {}
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines() + [""]:
        if not line.strip():
            if "physical id" in current and "core id" in current:
                cores.add((current["physical id"], current["core id"]))
            current = {}
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()
    return len(cores) or None


def _nvidia_smi_gpu_info() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except Exception:
        return {"available": False, "source": "nvidia-smi-unavailable"}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"available": False, "source": "nvidia-smi-unavailable"}
    first = proc.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 3:
        return {"available": False, "source": "nvidia-smi-unparseable"}
    try:
        total_mb = int(float(parts[1]))
        free_mb = int(float(parts[2]))
    except ValueError:
        total_mb = None
        free_mb = None
    return {
        "available": True,
        "source": "nvidia-smi",
        "name": parts[0],
        "vram_total_mb": total_mb,
        "vram_free_mb": free_mb,
        "vram_total_gb": round(total_mb / 1024, 2) if total_mb else None,
        "vram_free_gb": round(free_mb / 1024, 2) if free_mb else None,
    }


def _torch_cuda_info() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return {"torch_available": False, "cuda_available": False, "error": type(exc).__name__}
    try:
        available = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if available else None
        total = None
        free = None
        if available:
            props = torch.cuda.get_device_properties(0)
            total = int(getattr(props, "total_memory", 0) // (1024 * 1024))
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free = int(free_bytes // (1024 * 1024))
                total = int(total_bytes // (1024 * 1024))
            except Exception:
                pass
        return {
            "torch_available": True,
            "cuda_available": available,
            "device_name": name,
            "vram_total_mb": total,
            "vram_free_mb": free,
        }
    except Exception as exc:
        return {"torch_available": True, "cuda_available": False, "error": type(exc).__name__}


def detect_hardware() -> dict[str, Any]:
    logical = int(os.cpu_count() or 1)
    physical = _physical_cpu_cores() or max(1, logical // 2)
    gpu = _nvidia_smi_gpu_info()
    torch_info = _torch_cuda_info()
    cuda_available = bool(gpu.get("available") or torch_info.get("cuda_available"))
    gpu_name = torch_info.get("device_name") or gpu.get("name")
    vram_total_mb = torch_info.get("vram_total_mb") or gpu.get("vram_total_mb")
    vram_free_mb = torch_info.get("vram_free_mb") or gpu.get("vram_free_mb")
    return {
        "platform": platform.platform(),
        "cpu_physical_cores": physical,
        "cpu_logical_cores": logical,
        "ram_total_gb": _read_meminfo_gb(),
        "gpu_available": bool(gpu.get("available")),
        "gpu_name": gpu_name,
        "gpu_vram_total_gb": round(vram_total_mb / 1024, 2) if vram_total_mb else None,
        "gpu_vram_free_gb": round(vram_free_mb / 1024, 2) if vram_free_mb else None,
        "cuda_available": cuda_available,
        "torch_available": bool(torch_info.get("torch_available")),
        "torch_cuda_available": bool(torch_info.get("cuda_available")),
        "detection_notes": [
            note
            for note in [
                None if gpu.get("available") else "nvidia-smi GPU info unavailable",
                None if torch_info.get("torch_available") else "PyTorch unavailable in the UI environment",
                None if torch_info.get("cuda_available") else "PyTorch CUDA unavailable in the UI environment",
            ]
            if note
        ],
    }


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _vram_batch_size(vram_gb: float | None, *, mode: str) -> int:
    if not vram_gb:
        return 32 if mode == "cpu" else 64
    if mode == "soft":
        return 64 if vram_gb >= 16 else 32
    if mode == "balanced":
        if vram_gb >= 24:
            return 128
        if vram_gb >= 12:
            return 96
        return 64
    if mode == "aggressive":
        if vram_gb >= 28:
            return 256
        if vram_gb >= 16:
            return 192
        return 96
    if mode == "stress":
        if vram_gb >= 28:
            return 384
        if vram_gb >= 16:
            return 256
        return 128
    if mode == "gpu_focused":
        if vram_gb >= 28:
            return 320
        if vram_gb >= 16:
            return 192
        return 96
    return 32


def _base_profile(
    *,
    preset: str,
    hardware: dict[str, Any],
    accelerator: str,
    batch_size: int,
    siesta_jobs: int,
    evaluation_jobs: int,
    metric_jobs: int,
    threads: int,
    torch_threads: int,
    store_in_memory: bool | None = True,
    graph2mat_training_jobs: int = 1,
    deeph_training_jobs: int = 1,
    torch_mixed_precision: str | None = None,
    graph2mat_log_every_n_steps: int | None = None,
    graph2mat_check_val_every_n_epoch: int | None = None,
    graph2mat_checkpoint_every_n_epochs: int | None = None,
    graph2mat_require_cuequivariance: bool = False,
) -> dict[str, Any]:
    return {
        "max_parallel_siesta_jobs": int(siesta_jobs),
        "max_parallel_dataset_jobs": 1,
        "max_parallel_prediction_jobs": 1,
        "max_parallel_evaluation_jobs": int(evaluation_jobs),
        "max_parallel_metric_jobs": int(metric_jobs),
        "max_parallel_graph2mat_training_jobs": int(graph2mat_training_jobs),
        "max_parallel_deeph_training_jobs": int(deeph_training_jobs),
        "omp_num_threads": int(threads),
        "mkl_num_threads": int(threads),
        "openblas_num_threads": int(threads),
        "numexpr_num_threads": int(max(metric_jobs, torch_threads, 1)),
        "torch_num_threads": int(torch_threads),
        "compute_accelerator": accelerator,
        "batch_size": int(batch_size),
        "store_in_memory": store_in_memory,
        "reuse_validated_siesta_outputs": True,
        "enable_experiment_cache": False,
        "error_policy": "fail_fast",
        "preset": preset,
        "torch_float32_matmul_precision": "high",
        "torch_mixed_precision": torch_mixed_precision,
        "graph2mat_log_every_n_steps": graph2mat_log_every_n_steps,
        "graph2mat_check_val_every_n_epoch": graph2mat_check_val_every_n_epoch,
        "graph2mat_checkpoint_every_n_epochs": graph2mat_checkpoint_every_n_epochs,
        "graph2mat_require_cuequivariance": bool(graph2mat_require_cuequivariance),
    }


def performance_preset_catalog(hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    hw = hardware or detect_hardware()
    logical = max(1, int(hw.get("cpu_logical_cores") or 1))
    physical = max(1, int(hw.get("cpu_physical_cores") or max(1, logical // 2)))
    ram_gb = hw.get("ram_total_gb")
    vram_gb = hw.get("gpu_vram_total_gb")
    has_cuda = bool(hw.get("cuda_available"))
    low_ram = bool(ram_gb is not None and float(ram_gb) < 32)
    strong_gpu = bool(has_cuda and vram_gb is not None and float(vram_gb) >= 20)

    soft = _base_profile(
        preset="soft",
        hardware=hw,
        accelerator="gpu" if has_cuda else "cpu",
        batch_size=_vram_batch_size(vram_gb, mode="soft") if has_cuda else 16,
        siesta_jobs=1,
        evaluation_jobs=1,
        metric_jobs=_clamp(logical // 4, 1, 4),
        threads=1,
        torch_threads=_clamp(logical // 4, 1, 4),
        store_in_memory=False if low_ram else True,
    )
    balanced = _base_profile(
        preset="balanced",
        hardware=hw,
        accelerator="gpu" if has_cuda else "cpu",
        batch_size=_vram_batch_size(vram_gb, mode="balanced") if has_cuda else 32,
        siesta_jobs=_clamp(physical // 3, 1, 4),
        evaluation_jobs=_clamp(logical // 6, 1, 4),
        metric_jobs=_clamp(logical // 2, 4, 16),
        threads=2 if logical >= 8 else 1,
        torch_threads=_clamp(logical // 3, 2, 8),
        store_in_memory=False if low_ram else True,
    )
    aggressive = _base_profile(
        preset="aggressive",
        hardware=hw,
        accelerator="gpu" if has_cuda else "cpu",
        batch_size=_vram_batch_size(vram_gb, mode="aggressive") if has_cuda else 48,
        siesta_jobs=_clamp(physical // 2, 2, 8),
        evaluation_jobs=_clamp(logical // 3, 2, 8),
        metric_jobs=_clamp(logical, 8, 32),
        threads=_clamp(logical // max(1, _clamp(physical // 2, 2, 8)), 1, 4),
        torch_threads=_clamp(logical // 2, 4, 16),
        store_in_memory=False if low_ram else True,
        graph2mat_training_jobs=1,
        torch_mixed_precision="bf16-mixed" if has_cuda else None,
    )
    gpu_focused = _base_profile(
        preset="gpu_focused",
        hardware=hw,
        accelerator="gpu" if has_cuda else "cpu",
        batch_size=1024 if strong_gpu else (_vram_batch_size(vram_gb, mode="gpu_focused") if has_cuda else 32),
        siesta_jobs=_clamp(physical // 6, 1, 3),
        evaluation_jobs=_clamp(logical // 8, 1, 3),
        metric_jobs=_clamp(logical // 3, 2, 8),
        threads=1,
        torch_threads=_clamp(logical // 2, 4, 16),
        store_in_memory=False if low_ram else True,
        graph2mat_training_jobs=2 if strong_gpu else 1,
        torch_mixed_precision="bf16-mixed" if has_cuda else None,
        graph2mat_log_every_n_steps=10 if strong_gpu else None,
        graph2mat_check_val_every_n_epoch=5 if strong_gpu else None,
        graph2mat_checkpoint_every_n_epochs=5 if strong_gpu else None,
    )
    parallel_trains = _base_profile(
        preset="parallel_trains",
        hardware=hw,
        accelerator="gpu" if has_cuda else "cpu",
        batch_size=1024 if strong_gpu else (_vram_batch_size(vram_gb, mode="gpu_focused") if has_cuda else 32),
        siesta_jobs=_clamp(physical // 6, 1, 3),
        evaluation_jobs=_clamp(logical // 8, 1, 3),
        metric_jobs=_clamp(logical // 3, 2, 8),
        threads=1,
        torch_threads=1,
        store_in_memory=False if low_ram else True,
        graph2mat_training_jobs=4 if strong_gpu else 2,
        deeph_training_jobs=2 if strong_gpu else 1,
        torch_mixed_precision="bf16-mixed" if has_cuda else None,
        graph2mat_log_every_n_steps=10 if strong_gpu else None,
        graph2mat_check_val_every_n_epoch=5 if strong_gpu else None,
        graph2mat_checkpoint_every_n_epochs=5 if strong_gpu else None,
    )
    parallel_trains["numexpr_num_threads"] = 1
    cpu_only = _base_profile(
        preset="cpu_only",
        hardware=hw,
        accelerator="cpu",
        batch_size=32 if not low_ram else 16,
        siesta_jobs=_clamp(physical // 2, 1, 6),
        evaluation_jobs=_clamp(logical // 4, 1, 6),
        metric_jobs=_clamp(logical // 2, 2, 16),
        threads=_clamp(logical // max(1, _clamp(physical // 2, 1, 6)), 1, 4),
        torch_threads=_clamp(logical // 2, 2, 12),
        store_in_memory=False if low_ram else True,
    )
    stress = _base_profile(
        preset="max_aggressive",
        hardware=hw,
        accelerator="gpu" if has_cuda else "cpu",
        batch_size=_vram_batch_size(vram_gb, mode="stress") if has_cuda else 64,
        siesta_jobs=_clamp(physical, 2, 12),
        evaluation_jobs=_clamp(logical // 2, 4, 12),
        metric_jobs=_clamp(logical * 2, 16, 64),
        threads=_clamp(logical // max(1, _clamp(physical, 2, 12)), 1, 4),
        torch_threads=_clamp(logical, 8, 24),
        store_in_memory=False if low_ram else True,
        graph2mat_training_jobs=2 if strong_gpu else 1,
        torch_mixed_precision="bf16-mixed" if has_cuda else None,
    )
    debug = _base_profile(
        preset="single_run_debug",
        hardware=hw,
        accelerator="gpu" if has_cuda else "cpu",
        batch_size=16,
        siesta_jobs=1,
        evaluation_jobs=1,
        metric_jobs=1,
        threads=1,
        torch_threads=1,
        store_in_memory=False,
    )

    recommended = "cpu_only"
    if low_ram:
        recommended = "soft"
    elif strong_gpu and logical < 16:
        recommended = "gpu_focused"
    elif strong_gpu:
        recommended = "balanced"
    elif has_cuda:
        recommended = "balanced"

    dynamic: list[dict[str, Any]] = []
    if has_cuda and hw.get("gpu_name"):
        name = str(hw["gpu_name"])
        profile = dict(gpu_focused if strong_gpu else balanced)
        profile["preset"] = "dynamic_gpu_focused"
        dynamic.append({
            "id": "dynamic_gpu_focused",
            "label": f"{name} GPU-focused",
            "description": "Auto-generated profile for the detected GPU and VRAM.",
            "settings": profile,
            "warnings": performance_warnings(profile, hw),
        })
    if ram_gb is not None and float(ram_gb) >= 64:
        profile = dict(aggressive)
        profile["preset"] = "dynamic_high_ram_aggressive"
        dynamic.append({
            "id": "dynamic_high_ram_aggressive",
            "label": "High-RAM aggressive",
            "description": "Auto-generated profile that keeps store_in_memory enabled and raises CPU-side throughput.",
            "settings": profile,
            "warnings": performance_warnings(profile, hw),
        })
    cpu_profile = dict(cpu_only)
    cpu_profile["preset"] = "dynamic_cpu_heavy_siesta"
    cpu_profile["max_parallel_siesta_jobs"] = _clamp(physical, 1, 10)
    cpu_profile["omp_num_threads"] = _clamp(logical // max(1, cpu_profile["max_parallel_siesta_jobs"]), 1, 4)
    cpu_profile["mkl_num_threads"] = cpu_profile["omp_num_threads"]
    cpu_profile["openblas_num_threads"] = cpu_profile["omp_num_threads"]
    dynamic.append({
        "id": "dynamic_cpu_heavy_siesta",
        "label": "CPU-heavy SIESTA mode",
        "description": "Auto-generated profile for throughput in SIESTA single-point generation.",
        "settings": cpu_profile,
        "warnings": performance_warnings(cpu_profile, hw),
    })
    if low_ram:
        low_mem = dict(soft)
        low_mem["preset"] = "dynamic_low_memory_safe"
        low_mem["store_in_memory"] = False
        dynamic.append({
            "id": "dynamic_low_memory_safe",
            "label": "Low-memory safe mode",
            "description": "Auto-generated fallback for limited RAM.",
            "settings": low_mem,
            "warnings": performance_warnings(low_mem, hw),
        })
    dynamic.append({
        "id": "dynamic_single_run_debug",
        "label": "Single-run debug mode",
        "description": "Auto-generated low-parallelism profile for debugging one pipeline at a time.",
        "settings": debug,
        "warnings": performance_warnings(debug, hw),
    })

    builtins = [
        ("soft", "Soft", "Conservative mode for keeping the PC responsive.", soft, []),
        ("balanced", "Balanced", "Recommended default: good GPU usage without overloading CPU/RAM/I/O.", balanced, []),
        ("aggressive", "Aggressive", "High-performance local mode with controlled CPU parallelism.", aggressive, []),
        ("gpu_focused", "GPU-focused", "Prioritizes GPU training/inference and keeps CPU-side jobs moderate.", gpu_focused, []),
        (
            "parallel_trains",
            "Parallel trains",
            "Runs several independent Graph2Mat trainings on the GPU while keeping CPU worker pressure bounded.",
            parallel_trains,
            [],
        ),
        ("cpu_only", "CPU-only", "Disables GPU assumptions and uses CPU-safe throughput settings.", cpu_only, []),
        (
            "max_aggressive",
            "Max aggressive / stress",
            "Very aggressive mode for dedicated machines.",
            stress,
            ["May make the machine unresponsive, exhaust RAM/VRAM, or lose efficiency from oversubscription."],
        ),
        ("auto_detect", "Auto detect", f"Uses detected hardware and currently resolves to {recommended}.", {}, []),
    ]
    presets = [
        {
            "id": preset_id,
            "label": label,
            "description": description,
            "settings": settings,
            "warnings": warnings + performance_warnings(settings, hw),
            "recommended": preset_id == "balanced",
        }
        for preset_id, label, description, settings, warnings in builtins
    ]
    return {
        "hardware": hw,
        "default_preset": "balanced",
        "auto_detect_choice": recommended,
        "presets": presets,
        "dynamic_profiles": dynamic,
    }


def performance_oversubscription_score(settings: dict[str, Any]) -> int:
    blas_threads = max(
        int(settings.get("omp_num_threads") or 1),
        int(settings.get("mkl_num_threads") or 1),
        int(settings.get("openblas_num_threads") or 1),
    )
    return (
        int(settings.get("max_parallel_siesta_jobs") or 1) * blas_threads
        + int(settings.get("max_parallel_evaluation_jobs") or 1)
        + int(settings.get("max_parallel_metric_jobs") or 1)
        + int(settings.get("max_parallel_graph2mat_training_jobs") or 1)
        * int(settings.get("torch_num_threads") or 1)
        + int(settings.get("max_parallel_deeph_training_jobs") or 1)
        * int(settings.get("torch_num_threads") or 1)
    )


def performance_warnings(settings: dict[str, Any], hardware: dict[str, Any] | None = None) -> list[str]:
    hw = hardware or detect_hardware()
    warnings: list[str] = []
    logical = max(1, int(hw.get("cpu_logical_cores") or 1))
    score = performance_oversubscription_score(settings)
    if score > logical * 2:
        warnings.append(
            f"Estimated CPU pressure {score} exceeds 2x logical cores ({logical}); this can reduce efficiency."
        )
    if settings.get("compute_accelerator") == "gpu" and not hw.get("cuda_available"):
        warnings.append("GPU selected, but CUDA/GPU availability could not be confirmed in the UI environment.")
    if int(settings.get("max_parallel_graph2mat_training_jobs") or 1) > 1:
        if settings.get("compute_accelerator") == "gpu":
            warnings.append("Parallel Graph2Mat training jobs share the same GPU; monitor VRAM and throughput.")
        else:
            warnings.append("Parallel Graph2Mat training jobs on CPU can oversubscribe memory bandwidth.")
    if int(settings.get("max_parallel_deeph_training_jobs") or 1) > 1:
        if settings.get("compute_accelerator") == "gpu":
            warnings.append("Parallel DeepH training jobs share the same GPU; monitor VRAM and throughput.")
        else:
            warnings.append("Parallel DeepH training jobs on CPU can oversubscribe memory bandwidth.")
    vram = hw.get("gpu_vram_total_gb")
    batch = int(settings.get("batch_size") or 0)
    if settings.get("compute_accelerator") == "gpu" and vram is not None:
        if float(vram) < 12 and batch > 96:
            warnings.append("Batch size may be too high for the detected VRAM.")
        if float(vram) >= 24 and batch < 64:
            warnings.append("Detected high VRAM; this profile may underuse the GPU.")
    ram = hw.get("ram_total_gb")
    if settings.get("store_in_memory") and ram is not None and float(ram) < 32:
        warnings.append("store_in_memory is enabled on a low-RAM machine.")
    if (
        settings.get("compute_accelerator") == "gpu"
        and not settings.get("graph2mat_require_cuequivariance")
    ):
        warnings.append("Graph2Mat equivariant CUDA acceleration is optional; install cuequivariance for faster epochs.")
    if str(settings.get("preset")) in {"max_aggressive", "stress"}:
        warnings.append("Stress mode can make the machine unresponsive or exhaust RAM/VRAM.")
    return warnings


def validate_performance_settings(settings: dict[str, Any], hardware: dict[str, Any] | None = None) -> None:
    hw = hardware or detect_hardware()
    logical = max(1, int(hw.get("cpu_logical_cores") or 1))
    hard_limit = max(16, logical * 8)
    score = performance_oversubscription_score(settings)
    if score > hard_limit:
        raise RuntimeError(
            "performance settings are dangerously oversubscribed: "
            f"estimated CPU pressure {score} > hard limit {hard_limit}. "
            "Reduce SIESTA/evaluation/metric jobs or BLAS/Torch threads."
        )
    for key in (
        "max_parallel_siesta_jobs",
        "max_parallel_dataset_jobs",
        "max_parallel_prediction_jobs",
        "max_parallel_evaluation_jobs",
        "max_parallel_metric_jobs",
        "max_parallel_graph2mat_training_jobs",
        "max_parallel_deeph_training_jobs",
    ):
        value = int(settings.get(key) or 1)
        if value > max(64, logical * 4):
            raise RuntimeError(f"performance.{key}={value} is too high for this machine.")


def preset_settings_by_id(preset: str, hardware: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    normalized = str(preset).strip().lower()
    if normalized == "gpu_only":
        normalized = "gpu_focused"
    if normalized == "stress":
        normalized = "max_aggressive"
    if normalized == "aggressive_local":
        normalized = "aggressive"
    catalog = performance_preset_catalog(hardware)
    if normalized == "auto_detect":
        normalized = str(catalog["auto_detect_choice"])
    for item in list(catalog["presets"]) + list(catalog["dynamic_profiles"]):
        if item["id"] == normalized:
            settings = dict(item.get("settings") or {})
            settings["preset"] = normalized
            return normalized, settings
    raise RuntimeError(
        "performance.preset debe ser uno de "
        f"{', '.join(sorted(PERFORMANCE_PRESET_IDS))} o un perfil dynamic_* conocido."
    )


def parse_optional_positive_int(value: Any, name: str) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} debe ser un entero positivo.") from exc
    if number <= 0:
        raise RuntimeError(f"{name} debe ser un entero positivo.")
    return number


def parse_optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} debe ser un entero >= 0.") from exc
    if number < 0:
        raise RuntimeError(f"{name} debe ser un entero >= 0.")
    return number


def parse_optional_positive_float(value: Any, name: str) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} debe ser un numero positivo.") from exc
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(f"{name} debe ser un numero positivo.")
    return number


def parse_optional_bool(value: Any, name: str, default: bool | None = None) -> bool | None:
    if value in (None, "", "null"):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "si", "sí"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} debe ser booleano o null.")


def parse_optional_text(value: Any, name: str) -> str | None:
    if value in (None, "", "null"):
        return None
    text = str(value).strip()
    if not text:
        return None
    if "\n" in text or "\r" in text:
        raise RuntimeError(f"{name} debe caber en una linea.")
    return text


def parse_optional_json_object(value: Any, name: str) -> dict[str, Any] | None:
    if value in (None, "", "null"):
        return None
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise RuntimeError(f"{name} debe ser un objeto JSON.")
    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} debe ser un objeto JSON valido.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{name} debe ser un objeto JSON.")
    return parsed


TRAINING_SETTINGS_OBJECT_KEYS = {
    "data",
    "model",
    "trainer",
    "optimizer",
    "lr_scheduler",
    "training",
    "benchmark_metadata",
}


def deep_merge_dict(target: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge_dict(target[key], value)
        else:
            target[key] = value
    return target


def validate_hamiltonian_honly_training_config(config: dict[str, Any]) -> None:
    data = config.get("training", {}).get("data", {})
    if not data:
        return
    if data.get("out_matrix") not in (None, "hamiltonian"):
        raise RuntimeError("training.data.out_matrix debe ser hamiltonian.")
    policy = str(data.get("matrix_component_policy") or "h_only").strip()
    if policy != "h_only":
        raise RuntimeError("training.data.matrix_component_policy debe ser h_only.")
    try:
        n_components = int(data.get("n_matrix_components", 1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("training.data.n_matrix_components debe ser 1.") from exc
    if n_components != 1:
        raise RuntimeError("training.data.n_matrix_components debe ser 1.")
    if bool(data.get("symmetric_matrix", True)) is not True:
        raise RuntimeError("training.data.symmetric_matrix debe ser true.")


HIDDEN_IRREPS_TERM_RE = re.compile(r"^(?:(\d+)\s*x\s*)?(\d+)\s*([eoEO])$")
DEFAULT_HYPERPARAMETER_SWEEP_MAX_CONFIGS = 256
HYPERPARAMETER_SWEEP_MODE = "cartesian"
HYPERPARAMETER_SWEEP_PARAMETER_KEYS = {
    "max_epochs",
    "optim_lr",
    "batch_size",
    "loader_threads",
    "seed_everything",
    "loss",
    "loss_kwargs",
    "num_interactions",
    "correlation",
    "max_ell",
    "hidden_irreps",
    "hidden_irreps_channels",
    "data",
    "model",
    "trainer",
    "optimizer",
    "lr_scheduler",
    "training",
    "benchmark_metadata",
}
HYPERPARAMETER_SWEEP_TRAINING_KEYS = HYPERPARAMETER_SWEEP_PARAMETER_KEYS - {"hidden_irreps_channels"}


def parse_hidden_irreps_terms(value: str, name: str = "training_settings.hidden_irreps") -> list[dict[str, Any]]:
    terms: list[dict[str, Any]] = []
    for raw_term in str(value).split("+"):
        term = raw_term.strip()
        if not term:
            raise RuntimeError(f"{name}: termino vacio en hidden_irreps.")
        match = HIDDEN_IRREPS_TERM_RE.fullmatch(term)
        if not match:
            raise RuntimeError(
                f'{name}: "{term}" no es un termino Irreps valido. '
                "Usa formato NxLe, por ejemplo 10x1o."
            )
        multiplier = int(match.group(1) or "1")
        ell = int(match.group(2))
        parity = match.group(3).lower()
        if multiplier <= 0:
            raise RuntimeError(f"{name}: la multiplicidad debe ser positiva.")
        terms.append({"multiplier": multiplier, "ell": ell, "parity": parity})
    return terms


def expected_hidden_irreps(multiplier: int, max_ell: int) -> str:
    return " + ".join(
        f"{multiplier}x{ell}{'e' if ell % 2 == 0 else 'o'}"
        for ell in range(max_ell + 1)
    )


def build_hidden_irreps(channels: int, max_ell: int) -> str:
    if int(channels) <= 0:
        raise RuntimeError("hidden_irreps_channels debe ser un entero positivo.")
    if int(max_ell) < 0:
        raise RuntimeError("max_ell debe ser un entero >= 0 para generar hidden_irreps.")
    return expected_hidden_irreps(int(channels), int(max_ell))


def hidden_irreps_dimension(hidden_irreps: str) -> int:
    terms = parse_hidden_irreps_terms(hidden_irreps)
    dimension = sum(int(term["multiplier"]) * (2 * int(term["ell"]) + 1) for term in terms)
    if dimension <= 0:
        raise RuntimeError("hidden_irreps_dimension debe ser positiva.")
    return dimension


def validate_hidden_irreps(value: str, max_ell: int | None, name: str = "training_settings.hidden_irreps") -> None:
    terms = parse_hidden_irreps_terms(value, name)
    multipliers = {int(term["multiplier"]) for term in terms}
    if len(multipliers) != 1:
        raise RuntimeError(
            f"{name}: todos los terminos deben tener la misma multiplicidad/canales."
        )
    seen_ell: set[int] = set()
    for term in terms:
        ell = int(term["ell"])
        if ell in seen_ell:
            raise RuntimeError(f"{name}: l={ell} aparece mas de una vez.")
        seen_ell.add(ell)
        expected_parity = "e" if ell % 2 == 0 else "o"
        if str(term["parity"]) != expected_parity:
            raise RuntimeError(
                f"{name}: paridad inesperada para l={ell}; usa {ell}{expected_parity}."
            )
    if max_ell is None:
        return
    lmax = max(seen_ell)
    multiplier = next(iter(multipliers))
    expected = expected_hidden_irreps(multiplier, max_ell)
    if lmax != max_ell:
        raise RuntimeError(
            f"{name}: lmax={lmax} no coincide con training_settings.max_ell={max_ell}. "
            f"Usa {expected}."
        )
    missing = [ell for ell in range(max_ell + 1) if ell not in seen_ell]
    if missing:
        missing_text = ", ".join(f"l={ell}" for ell in missing)
        raise RuntimeError(f"{name}: faltan {missing_text}. Usa {expected}.")


def parse_training_settings(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        raw: dict[str, Any] = {}
    elif isinstance(value, dict):
        raw = value
    else:
        raise RuntimeError("training_settings debe ser un objeto.")
    settings: dict[str, Any] = {}
    for key in ("max_epochs", "batch_size", "loader_threads", "num_interactions", "correlation"):
        parsed = parse_optional_positive_int(raw.get(key), f"training_settings.{key}")
        if parsed is not None:
            settings[key] = parsed
    training_seed = parse_optional_nonnegative_int(
        raw.get("seed_everything", raw.get("seed")),
        "training_settings.seed_everything",
    )
    if training_seed is not None:
        settings["seed_everything"] = training_seed
    max_ell = parse_optional_nonnegative_int(raw.get("max_ell"), "training_settings.max_ell")
    if max_ell is not None:
        settings["max_ell"] = max_ell
    optim_lr = parse_optional_positive_float(raw.get("optim_lr"), "training_settings.optim_lr")
    if optim_lr is not None:
        settings["optim_lr"] = optim_lr
    for key in ("loss", "hidden_irreps"):
        parsed_text = parse_optional_text(raw.get(key), f"training_settings.{key}")
        if parsed_text is not None:
            settings[key] = parsed_text
    loss_kwargs = parse_optional_json_object(
        raw.get("loss_kwargs"),
        "training_settings.loss_kwargs",
    )
    if loss_kwargs is not None:
        settings["loss_kwargs"] = loss_kwargs
    for key in sorted(TRAINING_SETTINGS_OBJECT_KEYS):
        parsed_object = parse_optional_json_object(raw.get(key), f"training_settings.{key}")
        if parsed_object is not None:
            settings[key] = parsed_object
    if settings.get("hidden_irreps") is not None:
        validate_hidden_irreps(
            str(settings["hidden_irreps"]),
            settings.get("max_ell") if isinstance(settings.get("max_ell"), int) else None,
        )
    return settings


def _parse_sweep_value_list(value: Any, key: str) -> list[Any]:
    if value in (None, "", []):
        return []
    if key == "loss_kwargs" or key in TRAINING_SETTINGS_OBJECT_KEYS:
        if isinstance(value, str):
            raw_values = [
                item.strip()
                for item in value.splitlines()
                if item.strip()
            ]
        elif isinstance(value, (list, tuple)):
            raw_values = list(value)
        else:
            raw_values = [value]
        values = []
        seen = set()
        for raw in raw_values:
            parsed = parse_optional_json_object(
                raw,
                f"hyperparameter_sweep.parameters.{key}",
            )
            if parsed is None:
                continue
            marker = json.dumps(json_safe(parsed), sort_keys=True, ensure_ascii=False)
            if marker not in seen:
                seen.add(marker)
                values.append(parsed)
        return values
    if isinstance(value, str):
        raw_values = [
            item.strip()
            for item in re.split(r"[\n,;]+", value)
            if item.strip()
        ]
    elif isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raw_values = [value]
    values: list[Any] = []
    seen: set[str] = set()
    for raw in raw_values:
        if raw in (None, ""):
            continue
        if key in {
            "max_epochs",
            "batch_size",
            "loader_threads",
            "num_interactions",
            "correlation",
            "hidden_irreps_channels",
        }:
            parsed = parse_optional_positive_int(raw, f"hyperparameter_sweep.parameters.{key}")
        elif key == "seed_everything":
            parsed = parse_optional_nonnegative_int(raw, f"hyperparameter_sweep.parameters.{key}")
        elif key == "max_ell":
            parsed = parse_optional_nonnegative_int(raw, f"hyperparameter_sweep.parameters.{key}")
        elif key == "optim_lr":
            parsed = parse_optional_positive_float(raw, f"hyperparameter_sweep.parameters.{key}")
        elif key in {"loss", "hidden_irreps"}:
            parsed = parse_optional_text(raw, f"hyperparameter_sweep.parameters.{key}")
        elif key == "loss_kwargs":
            parsed = parse_optional_json_object(
                raw,
                f"hyperparameter_sweep.parameters.{key}",
            )
        else:
            raise RuntimeError(f"Parametro de sweep no soportado: {key}")
        if parsed is None:
            continue
        marker = json.dumps(json_safe(parsed), sort_keys=True, ensure_ascii=False)
        if marker not in seen:
            seen.add(marker)
            values.append(parsed)
    return values


def _parse_sweep_excluded_indices(value: Any) -> list[int]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        raw_values = [
            item.strip()
            for item in re.split(r"[\n,;]+", value)
            if item.strip()
        ]
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]
    indices: list[int] = []
    seen: set[int] = set()
    for raw in raw_values:
        parsed = parse_optional_positive_int(raw, "hyperparameter_sweep.excluded_indices")
        if parsed is None:
            continue
        if parsed not in seen:
            seen.add(parsed)
            indices.append(parsed)
    return indices


def parse_hyperparameter_sweep(value: Any) -> dict[str, Any]:
    if value in (None, "", False):
        return {"enabled": False}
    if isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "off"}:
        return {"enabled": False}
    if not isinstance(value, dict):
        raise RuntimeError("hyperparameter_sweep debe ser un objeto.")
    enabled = parse_bool(value.get("enabled"), False)
    if not enabled:
        return {"enabled": False}
    mode = str(value.get("mode") or HYPERPARAMETER_SWEEP_MODE).strip().lower()
    if mode != HYPERPARAMETER_SWEEP_MODE:
        raise RuntimeError("hyperparameter_sweep.mode solo soporta 'cartesian'.")
    label_prefix = parse_optional_text(value.get("label_prefix"), "hyperparameter_sweep.label_prefix") or "sweep"
    max_configs = parse_optional_positive_int(
        value.get("max_configs"),
        "hyperparameter_sweep.max_configs",
    ) or DEFAULT_HYPERPARAMETER_SWEEP_MAX_CONFIGS
    raw_parameters = value.get("parameters") or {}
    if not isinstance(raw_parameters, dict):
        raise RuntimeError("hyperparameter_sweep.parameters debe ser un objeto.")
    unknown = sorted(set(raw_parameters) - HYPERPARAMETER_SWEEP_PARAMETER_KEYS)
    if unknown:
        raise RuntimeError(f"Parametros de sweep no soportados: {', '.join(unknown)}.")
    parameters: dict[str, list[Any]] = {}
    for key in sorted(HYPERPARAMETER_SWEEP_PARAMETER_KEYS):
        values = _parse_sweep_value_list(raw_parameters.get(key), key)
        if values:
            parameters[key] = values
    if parameters.get("hidden_irreps") and parameters.get("hidden_irreps_channels"):
        raise RuntimeError(
            "Usa hidden_irreps o hidden_irreps_channels en el sweep, no ambos."
        )
    raw_excluded = value.get("excluded_indices")
    if raw_excluded in (None, ""):
        raw_excluded = value.get("excluded_config_indices")
    excluded_indices = _parse_sweep_excluded_indices(raw_excluded)
    return {
        "enabled": True,
        "mode": mode,
        "label_prefix": label_prefix,
        "max_configs": max_configs,
        "parameters": parameters,
        "excluded_indices": excluded_indices,
    }


def _format_sweep_label_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.8g}"
    else:
        text = str(value)
    return slugify_label(text, "v")


def _sweep_label_parts(settings: dict[str, Any], sweep_parameters: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    if "max_epochs" in sweep_parameters:
        parts.append(f"ep{settings.get('max_epochs')}")
    if "optim_lr" in sweep_parameters:
        parts.append(f"lr{_format_sweep_label_value(settings.get('optim_lr'))}")
    if "max_ell" in sweep_parameters:
        parts.append(f"l{settings.get('max_ell')}")
    if "hidden_irreps_channels" in sweep_parameters:
        parts.append(f"c{sweep_parameters.get('hidden_irreps_channels')}")
    elif "hidden_irreps" in sweep_parameters and settings.get("hidden_irreps"):
        terms = parse_hidden_irreps_terms(str(settings["hidden_irreps"]))
        channels = terms[0]["multiplier"] if terms else "x"
        parts.append(f"c{channels}")
    if "num_interactions" in sweep_parameters:
        parts.append(f"i{settings.get('num_interactions')}")
    if "correlation" in sweep_parameters:
        parts.append(f"corr{settings.get('correlation')}")
    if "batch_size" in sweep_parameters:
        parts.append(f"b{settings.get('batch_size')}")
    if "loader_threads" in sweep_parameters:
        parts.append(f"w{settings.get('loader_threads')}")
    if "seed_everything" in sweep_parameters:
        parts.append(f"seed{settings.get('seed_everything')}")
    if "loss" in sweep_parameters:
        parts.append(f"loss{_format_sweep_label_value(settings.get('loss'))}")
    metadata = settings.get("benchmark_metadata")
    if "benchmark_metadata" in sweep_parameters and isinstance(metadata, dict):
        method_id = metadata.get("benchmark_method_id") or metadata.get("method_id")
        if method_id:
            parts.append(str(method_id))
    return [part for part in parts if part and "None" not in part]


def _unique_sweep_label(
    raw_label: str,
    payload: dict[str, Any],
    used_labels: set[str],
) -> str:
    label = compact_dataset_label(raw_label, payload, max_length=48)
    suffix = 2
    while label in used_labels:
        label = compact_dataset_label(f"{raw_label}_{suffix}", {**payload, "suffix": suffix}, max_length=48)
        suffix += 1
    used_labels.add(label)
    return label


def expand_hyperparameter_sweep_to_training_plan(
    sweep: Any,
    *,
    base_training_settings: dict[str, Any] | None = None,
    run_mode: str,
    reusable_dataset_ids: list[str] | None = None,
    dataset_targets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    parsed = parse_hyperparameter_sweep(sweep)
    if not parsed.get("enabled"):
        return []
    run_mode = parse_run_mode(run_mode)
    if run_mode == DATASET_ONLY_RUN_MODE:
        raise RuntimeError("hyperparameter_sweep no esta disponible con dataset_only.")
    base_settings = parse_training_settings(base_training_settings)
    parameters = dict(parsed.get("parameters") or {})
    if not parameters:
        raise RuntimeError("hyperparameter_sweep necesita al menos un parametro con valores.")
    keys = sorted(parameters)
    count = math.prod(len(parameters[key]) for key in keys)
    excluded_indices = list(parsed.get("excluded_indices") or [])
    excluded_set = set(excluded_indices)
    out_of_range_exclusions = sorted(index for index in excluded_set if index > count)
    if out_of_range_exclusions:
        raise RuntimeError(
            "hyperparameter_sweep.excluded_indices fuera de rango: "
            + ", ".join(str(index) for index in out_of_range_exclusions[:10])
        )
    active_count = count - len(excluded_set)
    if active_count <= 0:
        raise RuntimeError("hyperparameter_sweep.excluded_indices excluye todas las configuraciones.")
    max_configs = int(parsed.get("max_configs") or DEFAULT_HYPERPARAMETER_SWEEP_MAX_CONFIGS)
    if active_count > max_configs:
        raise RuntimeError(
            f"hyperparameter_sweep genera {active_count} configuraciones activas "
            f"({count} totales, {len(excluded_set)} excluidas), por encima del limite {max_configs}."
        )
    reusable_ids = parse_reusable_dataset_ids(reusable_dataset_ids)
    targets = parse_dataset_targets(dataset_targets)
    if run_mode_skips_dataset_generation(run_mode) and not reusable_ids:
        raise RuntimeError("hyperparameter_sweep necesita reusable_dataset_ids en train_test_metrics_plots_only.")
    if run_mode_uses_planned_dataset_targets(run_mode) and not targets:
        raise RuntimeError(f"hyperparameter_sweep necesita dataset_targets en {run_mode}.")

    plan: list[dict[str, Any]] = []
    used_labels: set[str] = set()
    prefix = str(parsed.get("label_prefix") or "sweep")
    for zero_index, combo in enumerate(itertools.product(*(parameters[key] for key in keys))):
        cartesian_index = zero_index + 1
        if cartesian_index in excluded_set:
            continue
        sweep_parameters = dict(zip(keys, combo))
        raw_settings = dict(base_settings)
        for key, value in sweep_parameters.items():
            if key in HYPERPARAMETER_SWEEP_TRAINING_KEYS:
                raw_settings[key] = value
        if "hidden_irreps_channels" in sweep_parameters:
            max_ell = raw_settings.get("max_ell")
            if max_ell is None:
                raise RuntimeError(
                    "hidden_irreps_channels requiere max_ell en el sweep o en los controles de training."
                )
            raw_settings["hidden_irreps"] = build_hidden_irreps(
                int(sweep_parameters["hidden_irreps_channels"]),
                int(max_ell),
            )
        settings = parse_training_settings(raw_settings)
        dimension = (
            hidden_irreps_dimension(str(settings["hidden_irreps"]))
            if settings.get("hidden_irreps") is not None
            else None
        )
        display_label = "_".join(
            [f"{prefix}{cartesian_index:03d}", *_sweep_label_parts(settings, sweep_parameters)]
        )
        label = _unique_sweep_label(
            display_label,
            {"index": cartesian_index, "settings": settings, "sweep_parameters": sweep_parameters},
            used_labels,
        )
        plan_item: dict[str, Any] = {
            "index": len(plan) + 1,
            "label": label,
            "display_label": display_label,
            "training_settings": settings,
            "reusable_dataset_ids": reusable_ids if run_mode_skips_dataset_generation(run_mode) else [],
            "dataset_targets": targets if run_mode_uses_planned_dataset_targets(run_mode) else [],
            "sweep_index": cartesian_index,
            "sweep_label": label,
            "sweep_parameters": sweep_parameters,
        }
        if dimension is not None:
            plan_item["hidden_irreps_dimension"] = dimension
        plan.append(plan_item)
    validate_training_plan_for_run_mode(plan, run_mode)
    return plan


def validate_training_plan_sweep_sources(
    training_plan: list[dict[str, Any]],
    hyperparameter_sweep: dict[str, Any],
) -> None:
    if (
        hyperparameter_sweep.get("enabled")
        and training_plan
        and not all(item.get("sweep_index") is not None for item in training_plan)
    ):
        raise RuntimeError(
            "No mezcles hyperparameter_sweep con training_plan manual en la misma ejecucion."
        )


def parse_training_plan(value: Any) -> list[dict[str, Any]]:
    if value in (None, "", []):
        return []
    if not isinstance(value, list):
        raise RuntimeError("training_plan debe ser una lista de configuraciones.")
    plan: list[dict[str, Any]] = []
    used_labels: set[str] = set()
    for index, raw_item in enumerate(value):
        if not isinstance(raw_item, dict):
            raise RuntimeError(f"training_plan[{index}] debe ser un objeto.")
        dataset_ids = parse_reusable_dataset_ids(raw_item.get("reusable_dataset_ids"))
        dataset_targets = parse_dataset_targets(raw_item.get("dataset_targets"))
        if not dataset_ids and not dataset_targets:
            raise RuntimeError(
                f"training_plan[{index}] debe contener reusable_dataset_ids o dataset_targets."
            )
        settings = parse_training_settings(
            raw_item.get("training_settings", raw_item.get("settings", {}))
        )
        raw_label = parse_optional_text(raw_item.get("label"), f"training_plan[{index}].label")
        raw_display_label = parse_optional_text(
            raw_item.get("display_label"),
            f"training_plan[{index}].display_label",
        )
        display_label = raw_display_label or raw_label or f"Config {index + 1}"
        base_label = compact_dataset_label(
            raw_label or display_label,
            {"index": index + 1, "training_settings": settings},
            max_length=48,
        )
        label = base_label
        suffix = 2
        while label in used_labels:
            label = compact_dataset_label(
                f"{base_label}_{suffix}",
                {"index": index + 1, "suffix": suffix},
                max_length=48,
            )
            suffix += 1
        used_labels.add(label)
        plan.append(
            {
                "index": index + 1,
                "label": label,
                "display_label": display_label,
                "training_settings": settings,
                "reusable_dataset_ids": dataset_ids,
                "dataset_targets": dataset_targets,
            }
        )
        for metadata_key in (
            "sweep_index",
            "sweep_label",
            "sweep_parameters",
            "hidden_irreps_dimension",
        ):
            if metadata_key in raw_item:
                plan[-1][metadata_key] = raw_item[metadata_key]
    return plan


def apply_training_settings_to_config(config: dict[str, Any], settings: dict[str, Any] | None) -> None:
    if not settings:
        return
    training = config.setdefault("training", {})
    data = training.setdefault("data", {})
    model = training.setdefault("model", {})
    trainer = training.setdefault("trainer", {})
    if settings.get("max_epochs") is not None:
        trainer["max_epochs"] = int(settings["max_epochs"])
    if settings.get("batch_size") is not None:
        data["batch_size"] = int(settings["batch_size"])
    if settings.get("loader_threads") is not None:
        data["loader_threads"] = int(settings["loader_threads"])
    if settings.get("seed_everything") is not None:
        training["seed_everything"] = int(settings["seed_everything"])
    for key in ("num_interactions", "correlation", "max_ell"):
        if settings.get(key) is not None:
            model[key] = int(settings[key])
    if settings.get("optim_lr") is not None:
        model["optim_lr"] = float(settings["optim_lr"])
    for key in ("loss", "hidden_irreps"):
        if settings.get(key) is not None:
            model[key] = str(settings[key])
    if settings.get("loss_kwargs") is not None:
        model["loss_kwargs"] = dict(settings["loss_kwargs"])
    for section_key in ("data", "model", "trainer", "optimizer", "lr_scheduler"):
        section_override = settings.get(section_key)
        if isinstance(section_override, dict):
            deep_merge_dict(training.setdefault(section_key, {}), dict(section_override))
    training_override = settings.get("training")
    if isinstance(training_override, dict):
        for key, value in training_override.items():
            if isinstance(value, dict) and isinstance(training.get(key), dict):
                deep_merge_dict(training[key], value)
            else:
                training[key] = value
    if isinstance(settings.get("benchmark_metadata"), dict):
        training["benchmark_metadata"] = dict(settings["benchmark_metadata"])
    loss_name = str(training.get("model", {}).get("loss") or "")
    if loss_name.endswith((".coefficient_space_mse", ".coefficient_space_mae")):
        training.setdefault("model", {})["return_coefficients"] = True
    data = training.setdefault("data", {})
    data.setdefault("out_matrix", "hamiltonian")
    data.setdefault("matrix_component_policy", "h_only")
    data.setdefault("n_matrix_components", 1)
    data.setdefault("symmetric_matrix", True)
    validate_hamiltonian_honly_training_config(config)
    model = training.setdefault("model", {})
    if model.get("hidden_irreps") is not None and model.get("max_ell") is not None:
        validate_hidden_irreps(
            str(model["hidden_irreps"]),
            int(model["max_ell"]),
            "training.model.hidden_irreps",
        )
    training["ui_training_settings"] = dict(settings)


def benchmark_metadata_from_config(
    config: dict[str, Any],
    recipe_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    training = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    data = training.get("data", {}) if isinstance(training.get("data"), dict) else {}
    model = training.get("model", {}) if isinstance(training.get("model"), dict) else {}
    ui_settings = (
        training.get("ui_training_settings", {})
        if isinstance(training.get("ui_training_settings"), dict)
        else {}
    )
    metadata = {}
    for source in (
        ui_settings.get("benchmark_metadata"),
        training.get("benchmark_metadata"),
    ):
        if isinstance(source, dict):
            metadata.update(source)
    recipe_metadata = recipe_metadata or {}
    loss = metadata.get("loss") or model.get("loss")
    loss_kwargs = metadata.get("loss_kwargs")
    if loss_kwargs is None:
        loss_kwargs = model.get("loss_kwargs", {})
    training_stages = (
        metadata.get("training_stages")
        or model.get("training_stages")
        or training.get("training_stages")
        or training.get("stages")
        or []
    )
    hamiltonian_context = (
        metadata.get("hamiltonian_context")
        or model.get("hamiltonian_context")
        or model.get("context")
        or {}
    )
    if isinstance(hamiltonian_context, dict):
        context_enabled = bool(
            metadata.get(
                "context_enabled",
                hamiltonian_context.get("enabled", hamiltonian_context.get("use_context", False)),
            )
        )
    else:
        context_enabled = bool(metadata.get("context_enabled", hamiltonian_context))
    readout = (
        metadata.get("readout")
        or model.get("readout")
        or model.get("readout_type")
        or model.get("matrix_readout")
        or "default"
    )
    architecture = (
        metadata.get("architecture")
        or model.get("architecture")
        or ("hamiltonian_context" if context_enabled else "default")
    )
    return {
        **metadata,
        "benchmark_method_id": metadata.get("benchmark_method_id")
        or metadata.get("method_id")
        or recipe_metadata.get("training_plan_label")
        or recipe_metadata.get("sweep_label"),
        "architecture": architecture,
        "readout": readout,
        "hamiltonian_context": hamiltonian_context,
        "context_enabled": context_enabled,
        "loss": loss,
        "loss_kwargs": loss_kwargs if isinstance(loss_kwargs, dict) else {},
        "training_stages": training_stages,
        "seed": training.get("seed_everything"),
        "coefficient_space_enabled": bool(model.get("return_coefficients"))
        or str(loss or "").endswith((".coefficient_space_mse", ".coefficient_space_mae")),
        "diagnostic_only": bool(metadata.get("diagnostic_only", metadata.get("non_production", False))),
        "target": {
            "out_matrix": data.get("out_matrix"),
            "matrix_component_policy": data.get("matrix_component_policy"),
            "n_matrix_components": data.get("n_matrix_components"),
            "symmetric_matrix": data.get("symmetric_matrix"),
        },
        "dataset_reference_policy": {
            "require_real_siesta_reference": True,
            "forbidden_reference_filenames": ["ML_prediction.HSX"],
        },
    }


def _git_output(path: Path, args: list[str], timeout: float = 5.0) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_metadata_for_path(path: Path | str | None) -> dict[str, Any]:
    if path in (None, ""):
        return {"path": None, "commit": None, "branch": None, "dirty": None}
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return {"path": str(candidate), "commit": None, "branch": None, "dirty": None, "error": "path_missing"}
    root_text = _git_output(candidate, ["rev-parse", "--show-toplevel"])
    if not root_text:
        return {"path": str(candidate), "commit": None, "branch": None, "dirty": None, "error": "not_git_repo"}
    root = Path(root_text)
    commit = _git_output(root, ["rev-parse", "HEAD"])
    branch = _git_output(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    dirty_text = _git_output(root, ["status", "--porcelain"], timeout=10.0)
    return {
        "path": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty_text),
    }


def _python_from_config(config: dict[str, Any]) -> Path | None:
    paths = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    activate = paths.get("venv_activate")
    if not activate:
        return None
    activate_text = os.path.expandvars(str(activate).replace("${REPO_ROOT}", str(REPO_ROOT)))
    activate_path = Path(activate_text).expanduser()
    python = activate_path.parent / "python"
    return python if python.exists() else None


def graph2mat_git_metadata(config: dict[str, Any]) -> dict[str, Any]:
    python = _python_from_config(config)
    if python is None:
        return {"path": None, "commit": None, "branch": None, "dirty": None, "error": "python_not_found"}
    script = "import graph2mat, pathlib; print(pathlib.Path(graph2mat.__file__).resolve())"
    try:
        completed = subprocess.run(
            [str(python), "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
        )
    except Exception as exc:
        return {"path": None, "commit": None, "branch": None, "dirty": None, "error": str(exc)}
    if completed.returncode != 0:
        return {
            "path": None,
            "commit": None,
            "branch": None,
            "dirty": None,
            "error": completed.stderr.strip() or "graph2mat_import_failed",
        }
    package_file = Path(completed.stdout.strip()).expanduser()
    metadata = git_metadata_for_path(package_file.parent)
    metadata["package_file"] = str(package_file)
    return metadata


def run_inventory_payload() -> dict[str, Any]:
    """Fase 0 (audit): repo SHAs + imported checkouts + reproducibility status."""
    from run_inventory import collect_run_inventory

    deeph_python = REPO_ROOT.parent / "DeepH-pack" / ".venv" / "bin" / "python"
    return collect_run_inventory(
        deeph_python=str(deeph_python) if deeph_python.exists() else None
    )


def deeph_capabilities_payload() -> dict[str, Any]:
    """Fase 14 (audit): real DeepH autograd capability from the effective backend."""
    from run_inventory import git_repository_state

    deeph_repo = REPO_ROOT.parent / "DeepH-pack"
    deeph_python = deeph_repo / ".venv" / "bin" / "python"
    payload: dict[str, Any] = {
        "schema": "deeph_capabilities_ui_payload_v1",
        "repository": git_repository_state(deeph_repo),
    }
    if not deeph_python.exists():
        payload["autograd"] = {"available": False, "errors": ["deeph_python_not_found"]}
        return payload
    try:
        completed = subprocess.run(
            [
                str(deeph_python),
                "-c",
                "import json; from deeph.inference.capability import autograd_capability; "
                "print(json.dumps(autograd_capability()))",
            ],
            capture_output=True,
            text=True,
            timeout=120.0,
        )
        if completed.returncode == 0:
            payload["autograd"] = json.loads(completed.stdout.strip().splitlines()[-1])
        else:
            payload["autograd"] = {
                "available": False,
                "errors": ["capability_module_unavailable", completed.stderr.strip()[-500:]],
            }
    except Exception as exc:  # noqa: BLE001 - UI payloads must not crash the server
        payload["autograd"] = {"available": False, "errors": [repr(exc)]}
    return payload


def aggressive_local_performance_defaults() -> dict[str, Any]:
    cores = max(1, os.cpu_count() or 1)
    siesta_jobs = max(1, min(max(1, cores // 2), 8))
    return {
        "max_parallel_siesta_jobs": siesta_jobs,
        "max_parallel_dataset_jobs": 1,
        "max_parallel_prediction_jobs": 1,
        "max_parallel_evaluation_jobs": min(cores, 8),
        "max_parallel_metric_jobs": cores,
        "max_parallel_graph2mat_training_jobs": 1,
        "max_parallel_deeph_training_jobs": 1,
        "omp_num_threads": max(1, cores // siesta_jobs),
        "mkl_num_threads": max(1, cores // siesta_jobs),
        "openblas_num_threads": max(1, cores // siesta_jobs),
        "numexpr_num_threads": cores,
        "torch_num_threads": min(cores, 16),
        "compute_accelerator": "gpu",
        "batch_size": 64,
        "store_in_memory": True,
        "reuse_validated_siesta_outputs": True,
        "enable_experiment_cache": False,
        "error_policy": "fail_fast",
        "torch_float32_matmul_precision": "high",
        "torch_mixed_precision": "bf16-mixed",
    }


def parse_performance_settings(value: Any, *, compute_accelerator: str | None = None) -> dict[str, Any]:
    if value in (None, ""):
        raw: dict[str, Any] = {}
    elif isinstance(value, dict):
        raw = value
    else:
        raise RuntimeError("performance debe ser un objeto.")
    settings = dict(DEFAULT_PERFORMANCE_SETTINGS)
    preset = raw.get("preset", "balanced")
    if preset in (None, "", "null"):
        settings["preset"] = None
    else:
        resolved_preset, preset_settings = preset_settings_by_id(str(preset))
        settings.update(preset_settings)
        settings["preset"] = resolved_preset
    settings["compute_accelerator"] = parse_compute_accelerator(
        raw.get(
            "compute_accelerator",
            compute_accelerator if compute_accelerator is not None else settings.get("compute_accelerator"),
        )
    )
    for key in (
        "max_parallel_siesta_jobs",
        "max_parallel_dataset_jobs",
        "max_parallel_prediction_jobs",
        "max_parallel_evaluation_jobs",
        "max_parallel_metric_jobs",
        "max_parallel_graph2mat_training_jobs",
        "max_parallel_deeph_training_jobs",
    ):
        settings[key] = parse_optional_positive_int(
            raw.get(key, settings[key]),
            f"performance.{key}",
        ) or int(settings[key] or 1)
    for key in (
        "omp_num_threads",
        "mkl_num_threads",
        "openblas_num_threads",
        "numexpr_num_threads",
        "torch_num_threads",
        "batch_size",
        "graph2mat_log_every_n_steps",
        "graph2mat_check_val_every_n_epoch",
        "graph2mat_checkpoint_every_n_epochs",
    ):
        settings[key] = parse_optional_positive_int(raw.get(key, settings.get(key)), f"performance.{key}")
    settings["store_in_memory"] = parse_optional_bool(
        raw.get("store_in_memory"),
        "performance.store_in_memory",
        settings.get("store_in_memory"),
    )
    settings["reuse_validated_siesta_outputs"] = bool(parse_optional_bool(
        raw.get("reuse_validated_siesta_outputs"),
        "performance.reuse_validated_siesta_outputs",
        settings.get("reuse_validated_siesta_outputs"),
    ))
    settings["enable_experiment_cache"] = bool(parse_optional_bool(
        raw.get("enable_experiment_cache"),
        "performance.enable_experiment_cache",
        settings.get("enable_experiment_cache"),
    ))
    settings["graph2mat_require_cuequivariance"] = bool(parse_optional_bool(
        raw.get("graph2mat_require_cuequivariance"),
        "performance.graph2mat_require_cuequivariance",
        settings.get("graph2mat_require_cuequivariance"),
    ))
    if settings["enable_experiment_cache"]:
        raise RuntimeError(
            "performance.enable_experiment_cache todavia no esta implementado con "
            "validacion completa por hashes; dejalo en false para evitar cache insegura."
        )
    error_policy = str(raw.get("error_policy", settings["error_policy"]) or "fail_fast").strip().lower()
    if error_policy not in {"fail_fast", "continue_on_error"}:
        raise RuntimeError("performance.error_policy debe ser fail_fast o continue_on_error.")
    settings["error_policy"] = error_policy
    precision = raw.get(
        "torch_float32_matmul_precision",
        settings.get("torch_float32_matmul_precision"),
    )
    if precision in (None, "", "null"):
        settings["torch_float32_matmul_precision"] = None
    else:
        precision = str(precision).strip().lower()
        if precision not in {"high", "medium"}:
            raise RuntimeError("performance.torch_float32_matmul_precision debe ser null, high o medium.")
        settings["torch_float32_matmul_precision"] = precision
    mixed_precision = raw.get("torch_mixed_precision", settings.get("torch_mixed_precision"))
    if mixed_precision in (None, "", "null"):
        settings["torch_mixed_precision"] = None
    else:
        mixed_precision = str(mixed_precision).strip().lower()
        if mixed_precision not in {"32-true", "16-mixed", "bf16-mixed"}:
            raise RuntimeError("performance.torch_mixed_precision debe ser null, 32-true, 16-mixed o bf16-mixed.")
        settings["torch_mixed_precision"] = mixed_precision
    warnings = performance_warnings(settings)
    settings["warnings"] = warnings
    validate_performance_settings(settings)
    return settings


def performance_env(settings: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, env_name in (
        ("omp_num_threads", "OMP_NUM_THREADS"),
        ("mkl_num_threads", "MKL_NUM_THREADS"),
        ("openblas_num_threads", "OPENBLAS_NUM_THREADS"),
        ("numexpr_num_threads", "NUMEXPR_NUM_THREADS"),
        ("torch_num_threads", "TORCH_NUM_THREADS"),
    ):
        value = settings.get(key)
        if value not in (None, ""):
            env[env_name] = str(int(value))
    if settings.get("torch_float32_matmul_precision"):
        env["TORCH_FLOAT32_MATMUL_PRECISION"] = str(settings["torch_float32_matmul_precision"])
    return env


def apply_performance_to_config(config: dict[str, Any], settings: dict[str, Any]) -> None:
    config["performance"] = dict(settings)
    apply_training_accelerator(config, str(settings["compute_accelerator"]))
    training_data = config.setdefault("training", {}).setdefault("data", {})
    if settings.get("batch_size") is not None:
        training_data["batch_size"] = int(settings["batch_size"])
    if settings.get("store_in_memory") is not None:
        training_data["store_in_memory"] = bool(settings["store_in_memory"])
        config.setdefault("testing", {}).setdefault("data", {})["store_in_memory"] = bool(settings["store_in_memory"])
        config.setdefault("prediction", {}).setdefault("data", {})["store_in_memory"] = bool(settings["store_in_memory"])
    single_points = config.setdefault("single_points", {})
    single_points["workers"] = int(settings["max_parallel_siesta_jobs"])
    single_points["rerun"] = not bool(settings.get("reuse_validated_siesta_outputs", True))
    config.setdefault("training", {})["torch_float32_matmul_precision"] = settings.get(
        "torch_float32_matmul_precision"
    )
    if settings.get("torch_mixed_precision"):
        config.setdefault("training", {}).setdefault("trainer", {})["precision"] = str(settings["torch_mixed_precision"])
    trainer = config.setdefault("training", {}).setdefault("trainer", {})
    if settings.get("graph2mat_log_every_n_steps") is not None:
        trainer["log_every_n_steps"] = int(settings["graph2mat_log_every_n_steps"])
    if settings.get("graph2mat_check_val_every_n_epoch") is not None:
        trainer["check_val_every_n_epoch"] = int(settings["graph2mat_check_val_every_n_epoch"])
    if settings.get("graph2mat_checkpoint_every_n_epochs") is not None:
        every_n_epochs = int(settings["graph2mat_checkpoint_every_n_epochs"])
        trainer["callbacks"] = [
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


def cuda_available(python_executable: Path | str) -> bool:
    result = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import torch; print('1' if torch.cuda.is_available() else '0')",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip().endswith("1")


def reference_budget_for_run(run: dict[str, Any]) -> int:
    for key in ("completed_samples", "effective_dataset_size", "dataset_size"):
        value = run.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return int(float(value))
        if isinstance(value, str) and value.strip().isdigit():
            return int(value)
    return 0


def budget_ratio(md_budget: int, atom_budget: int) -> float | None:
    if md_budget <= 0 or atom_budget <= 0:
        return None
    return max(md_budget, atom_budget) / min(md_budget, atom_budget)


def budget_warning(md_budget: int, atom_budget: int, *, tolerance: float = 1.25) -> str:
    ratio = budget_ratio(md_budget, atom_budget)
    if ratio is None:
        return "Budget could not be computed."
    if ratio > tolerance:
        return f"Budgets differ by ratio {ratio:.3g}; equal-budget comparison is approximate."
    return ""


def should_compare_budget_pair(
    md_run: dict[str, Any],
    atom_run: dict[str, Any],
    all_atom_runs: list[dict[str, Any]],
    mode: str,
) -> bool:
    md_size = int(md_run.get("dataset_size", 0))
    atom_size = int(atom_run.get("dataset_size", 0))
    if mode == "both":
        return True
    if mode == "equal_sample_count":
        return md_size == atom_size
    if mode == "equal_siesta_budget":
        md_budget = reference_budget_for_run(md_run)
        atom_budget = reference_budget_for_run(atom_run)
        if md_budget <= 0 or atom_budget <= 0:
            return False
        deltas = [
            abs(reference_budget_for_run(candidate) - md_budget)
            for candidate in all_atom_runs
            if reference_budget_for_run(candidate) > 0
        ]
        return bool(deltas) and abs(atom_budget - md_budget) == min(deltas)
    return True


def unique_ints_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    unique: list[int] = []
    for value in values:
        number = int(value)
        if number in seen:
            continue
        seen.add(number)
        unique.append(number)
    return unique


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


def distribute_fc_counts(total: int, n_displacements: int, per_displacement_limit: int) -> list[int]:
    if n_displacements <= 0:
        raise RuntimeError("AtomDisplacement: define al menos una magnitud FC.")
    capacity = n_displacements * per_displacement_limit
    if total > capacity:
        raise RuntimeError(
            "AtomDisplacement: el tamano pedido excede la capacidad FC configurada. "
            f"Pedido={total}, capacidad={capacity} ({n_displacements} magnitudes x "
            f"{per_displacement_limit} estructuras/magnitud)."
        )
    counts = [0 for _ in range(n_displacements)]
    remaining = total
    index = 0
    while remaining:
        if counts[index] < per_displacement_limit:
            counts[index] += 1
            remaining -= 1
        index = (index + 1) % n_displacements
    return counts


def make_aligned_dataset_label(entries: list[dict[str, Any]]) -> str:
    parts = [
        f"d{_displacement_slug(str(entry['value']))}_{int(entry['n_structures'])}"
        for entry in entries
    ]
    return "dataset_" + "__".join(parts)


def build_fc_aligned_dataset_specs(
    displacement_options: dict[str, list[int]],
    *,
    per_displacement_limit: int,
    split_ratios: dict[str, float],
    max_datasets: int,
) -> list[dict[str, Any]]:
    """Build one independent dataset spec per index-aligned combination.

    Counts are zipped across sorted displacement keys. With
    ``{"0.03": [5, 7], "0.04": [6, 8]}``, only two datasets are created:
    ``(5, 6)`` and ``(7, 8)``. This deliberately avoids Cartesian products so
    the user controls exactly which combinations are generated.
    """

    keys = list(displacement_options)
    lengths = {key: len(displacement_options[key]) for key in keys}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            "Todas las listas de displacement_options deben tener la misma longitud. "
            f"Longitudes recibidas: {lengths}."
        )
    n_datasets = next(iter(lengths.values()), 0)
    if n_datasets > max_datasets:
        raise RuntimeError(
            "La configuracion FC genera demasiados datasets: "
            f"{n_datasets} datasets alineados > max_datasets={max_datasets}. "
            "Reduce filas o aumenta max_datasets."
        )
    specs: list[dict[str, Any]] = []
    used_dataset_ids: set[str] = set()
    seen_labels: set[str] = set()
    for dataset_index, combo in enumerate(zip(*(displacement_options[key] for key in keys))):
        entries = []
        for displacement, count in zip(keys, combo):
            if count > per_displacement_limit:
                raise RuntimeError(
                    f"{displacement}: pide {count} estructuras, pero el limite FC "
                    f"por magnitud es {per_displacement_limit}."
                )
            entries.append({"value": displacement, "n_structures": int(count)})
        size = sum(int(entry["n_structures"]) for entry in entries)
        label, short_id = allocate_dataset_label(
            "fc",
            dataset_index,
            used_ids=used_dataset_ids,
            used_labels=seen_labels,
        )
        validate_split_sizes(size, split_ratios, label=label)
        specs.append({"label": label, "dataset_short_id": short_id, "size": size, "displacements": entries})
    return specs


def build_fc_cartesian_dataset_specs(
    displacement_options: dict[str, list[int]],
    *,
    per_displacement_limit: int,
    split_ratios: dict[str, float],
    max_datasets: int,
) -> list[dict[str, Any]]:
    """Build one dataset per Cartesian product of displacement-count options.

    Counts are expanded across sorted displacement keys. With
    ``{"0.03": [5, 7], "0.04": [6, 8]}``, four datasets are created:
    ``(5, 6)``, ``(5, 8)``, ``(7, 6)``, and ``(7, 8)``. This mode is explicit
    and opt-in because the number of datasets grows multiplicatively.
    """

    keys = list(displacement_options)
    if not keys:
        raise RuntimeError("Cartesian FC requiere al menos una magnitud.")
    n_datasets = math.prod(len(displacement_options[key]) for key in keys)
    if n_datasets > max_datasets:
        raise RuntimeError(
            "La configuracion FC genera demasiados datasets: "
            f"{n_datasets} datasets cartesianos > max_datasets={max_datasets}. "
            "Reduce opciones o aumenta max_datasets."
        )

    specs: list[dict[str, Any]] = []
    used_dataset_ids: set[str] = set()
    seen_labels: set[str] = set()
    for dataset_index, combo in enumerate(
        itertools.product(*(displacement_options[key] for key in keys))
    ):
        entries = []
        for displacement, count in zip(keys, combo):
            if count > per_displacement_limit:
                raise RuntimeError(
                    f"{displacement}: pide {count} estructuras, pero el limite FC "
                    f"por magnitud es {per_displacement_limit}."
                )
            entries.append({"value": displacement, "n_structures": int(count)})
        size = sum(int(entry["n_structures"]) for entry in entries)
        label, short_id = allocate_dataset_label(
            "fc",
            dataset_index,
            used_ids=used_dataset_ids,
            used_labels=seen_labels,
        )
        validate_split_sizes(size, split_ratios, label=label)
        specs.append({"label": label, "dataset_short_id": short_id, "size": size, "displacements": entries})
    return specs


def build_fc_dataset_specs_from_options(
    displacement_options: dict[str, list[int]],
    *,
    combination_mode: str,
    per_displacement_limit: int,
    split_ratios: dict[str, float],
    max_datasets: int,
) -> list[dict[str, Any]]:
    if combination_mode == "aligned":
        return build_fc_aligned_dataset_specs(
            displacement_options,
            per_displacement_limit=per_displacement_limit,
            split_ratios=split_ratios,
            max_datasets=max_datasets,
        )
    if combination_mode == "cartesian":
        return build_fc_cartesian_dataset_specs(
            displacement_options,
            per_displacement_limit=per_displacement_limit,
            split_ratios=split_ratios,
            max_datasets=max_datasets,
        )
    raise RuntimeError(f"combination_mode no soportado: {combination_mode!r}.")


def build_fc_dataset_specs(
    atom_sizes: list[int],
    fc_displacements: list[dict[str, Any]] | None,
    structures_per_displacement: list[int] | None,
    *,
    per_displacement_limit: int,
    split_ratios: dict[str, float],
) -> tuple[list[int], dict[int, list[dict[str, Any]]] | None]:
    if structures_per_displacement:
        if not fc_displacements:
            raise RuntimeError(
                "Define magnitudes FC para usar structures_per_displacement."
            )
        specs: dict[int, list[dict[str, Any]]] = {}
        generated_sizes: list[int] = []
        for count in structures_per_displacement:
            if count > per_displacement_limit:
                raise RuntimeError(
                    "AtomDisplacement: structures_per_displacement excede el limite FC. "
                    f"Pedido={count}, maximo por magnitud={per_displacement_limit}."
                )
            size = count * len(fc_displacements)
            validate_split_sizes(size, split_ratios, label=f"AtomDisplacement dataset_{size}")
            if size in specs:
                raise RuntimeError(
                    "Dos entradas de structures_per_displacement producen el mismo "
                    f"dataset_size={size}."
                )
            generated_sizes.append(size)
            specs[size] = [
                {**entry, "n_structures": count}
                for entry in fc_displacements
            ]
        return generated_sizes, specs

    if fc_displacements and all("n_structures" in entry for entry in fc_displacements):
        explicit_total = sum(int(entry["n_structures"]) for entry in fc_displacements)
        for entry in fc_displacements:
            requested = int(entry["n_structures"])
            if requested > per_displacement_limit:
                raise RuntimeError(
                    "AtomDisplacement: una magnitud FC pide mas estructuras de las que "
                    f"permite FC. {entry.get('value')} pide {requested}, "
                    f"maximo {per_displacement_limit}."
                )
        validate_split_sizes(
            explicit_total,
            split_ratios,
            label=f"AtomDisplacement dataset_{explicit_total}",
        )
        return [explicit_total], {explicit_total: fc_displacements}

    return atom_sizes, None


def validate_atom_sizes_for_fc(
    atom_sizes: list[int],
    fc_dataset_specs: dict[int, list[dict[str, Any]]] | None = None,
    split_ratios: dict[str, float] | None = None,
) -> None:
    config = load_config(PIPELINES["atom_displacement"].config_path)
    limit = atom_fc_sample_limit(config)
    if limit is None:
        return
    ratios = split_ratios or split_ratios_from_config(config)
    displacement_entries = (
        next(iter(fc_dataset_specs.values()))
        if fc_dataset_specs
        else atom_fc_displacement_entries(config)
    )
    displacement_count = len(displacement_entries)
    if fc_dataset_specs:
        for size, entries in fc_dataset_specs.items():
            validate_split_sizes(size, ratios, label=f"AtomDisplacement dataset_{size}")
            for entry in entries:
                requested = int(entry.get("n_structures") or 0)
                if requested > limit:
                    raise RuntimeError(
                        "AtomDisplacement: una magnitud FC pide mas estructuras de "
                        f"las que permite FC. {entry.get('value')} pide {requested}, "
                        f"maximo {limit}."
                    )
    else:
        for size in atom_sizes:
            validate_split_sizes(size, ratios, label=f"AtomDisplacement dataset_{size}")
    total_limit = limit * displacement_count
    too_large = [size for size in atom_sizes if size > total_limit]
    if not too_large:
        return
    raise RuntimeError(
        "AtomDisplacement usa SIESTA MD.TypeOfRun FC, que no genera un numero "
        "arbitrario de estructuras. Con la configuracion actual "
        f"({len(config['structure']['atoms'])} atomos, FC.First="
        f"{config['structure']['force_constants'].get('first_atom', 1)}, "
        f"FC.Last={config['structure']['force_constants'].get('last_atom') or len(config['structure']['atoms'])}) "
        f"el maximo por magnitud es {limit} estructuras. Con "
        f"{displacement_count} magnitudes, la capacidad total es {total_limit}. "
        f"Tamanos invalidos: {too_large}. Anade magnitudes FC o reduce el tamano."
    )


def validate_atom_dataset_specs_for_fc(
    specs: list[dict[str, Any]],
    split_ratios: dict[str, float],
) -> None:
    config = load_config(PIPELINES["atom_displacement"].config_path)
    limit = atom_fc_sample_limit(config)
    if limit is None:
        return
    for spec in specs:
        label = str(spec.get("label") or f"dataset_{spec.get('size')}")
        size = int(spec["size"])
        validate_split_sizes(size, split_ratios, label=f"AtomDisplacement {label}")
        entries = spec.get("displacements") or []
        if not entries:
            raise RuntimeError(f"AtomDisplacement {label}: no tiene magnitudes FC.")
        for entry in entries:
            requested = int(entry.get("n_structures") or 0)
            if requested <= 0:
                raise RuntimeError(
                    f"AtomDisplacement {label}: {entry.get('value')} pide "
                    f"{requested} estructuras; debe ser un entero positivo."
                )
            if requested > limit:
                raise RuntimeError(
                    "AtomDisplacement: una magnitud FC pide mas estructuras de "
                    f"las que permite FC. {label}, {entry.get('value')} pide "
                    f"{requested}, maximo {limit}."
                )


def select_spread(items: list[Path], count: int) -> list[Path]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)

    used: set[int] = set()
    selected: list[int] = []
    for index in range(count):
        target = min(len(items) - 1, int((index + 0.5) * len(items) / count))
        if target in used:
            target = min(
                (candidate for candidate in range(len(items)) if candidate not in used),
                key=lambda candidate: abs(candidate - target),
            )
        used.add(target)
        selected.append(target)
    return [items[index] for index in sorted(selected)]


def split_spread(items: list[Path], counts: dict[str, int]) -> dict[str, list[Path]]:
    selected = list(items)
    test = select_spread(selected, counts["test"])
    remaining = [item for item in selected if item not in set(test)]
    validation = select_spread(remaining, counts["validation"])
    train = [item for item in remaining if item not in set(validation)]
    if len(train) > counts["train"]:
        train = select_spread(train, counts["train"])
    return {"train": train, "validation": validation, "test": test}


def select_spread_items(items: list[Any], count: int) -> list[Any]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)

    used: set[int] = set()
    selected: list[int] = []
    for index in range(count):
        target = min(len(items) - 1, int((index + 0.5) * len(items) / count))
        if target in used:
            target = min(
                (candidate for candidate in range(len(items)) if candidate not in used),
                key=lambda candidate: abs(candidate - target),
            )
        used.add(target)
        selected.append(target)
    return [items[index] for index in sorted(selected)]


def split_spread_items(items: list[Any], counts: dict[str, int]) -> dict[str, list[Any]]:
    indexed = list(enumerate(items))
    test = select_spread_items(indexed, counts["test"])
    test_indexes = {index for index, _item in test}
    remaining = [entry for entry in indexed if entry[0] not in test_indexes]
    validation = select_spread_items(remaining, counts["validation"])
    validation_indexes = {index for index, _item in validation}
    train = [entry for entry in remaining if entry[0] not in validation_indexes]
    if len(train) > counts["train"]:
        train = select_spread_items(train, counts["train"])
    return {
        "train": [item for _index, item in train],
        "validation": [item for _index, item in validation],
        "test": [item for _index, item in test],
    }


def split_block_items(items: list[Any], counts: dict[str, int]) -> dict[str, list[Any]]:
    requested = counts["train"] + counts["validation"] + counts["test"]
    selected = select_spread_items(items, requested)
    train_end = counts["train"]
    validation_end = train_end + counts["validation"]
    return {
        "train": selected[:train_end],
        "validation": selected[train_end:validation_end],
        "test": selected[validation_end:],
    }


def split_blocked_with_gap_items(
    items: list[Any],
    counts: dict[str, int],
    *,
    temporal_gap: int,
) -> tuple[dict[str, list[Any]], list[tuple[Any, str]]]:
    required = sum(counts.values()) + 2 * temporal_gap
    if required > len(items):
        raise RuntimeError(
            "El split MD blocked_with_gap necesita mas muestras de las disponibles: "
            f"{required} > {len(items)} (gap={temporal_gap}, counts={counts})."
        )
    result: dict[str, list[Any]] = {"train": [], "validation": [], "test": []}
    excluded: list[tuple[Any, str]] = []
    cursor = 0
    order = ("train", "validation", "test")
    for index, split_name in enumerate(order):
        count = counts[split_name]
        result[split_name] = list(items[cursor : cursor + count])
        cursor += count
        if index < len(order) - 1 and temporal_gap > 0:
            next_split = order[index + 1]
            for item in items[cursor : cursor + temporal_gap]:
                excluded.append((item, f"temporal_gap_between_{split_name}_and_{next_split}"))
            cursor += temporal_gap
    return result, excluded


def split_grouped_exact_items(
    items: list[dict[str, Any]],
    counts: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        row = item.get("row") if isinstance(item, dict) else {}
        group_id = str((row or {}).get("split_group_id") or item.get("name") or item.get("source") or "")
        groups[group_id].append(item)
    ordered_groups = [
        (key, sorted(value, key=lambda entry: sample_sort_key(Path(str(entry.get("source") or "")))))
        for key, value in sorted(groups.items())
    ]
    result: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for split_name in ("test", "validation", "train"):
        target = counts[split_name]
        for key, samples in list(ordered_groups):
            if len(result[split_name]) + len(samples) > target:
                continue
            result[split_name].extend(samples)
            ordered_groups.remove((key, samples))
            if len(result[split_name]) == target:
                break
        if len(result[split_name]) != target:
            raise RuntimeError(
                "Reusable grouped split no puede satisfacer los tamanos exactos "
                f"sin partir familias: {split_name} necesita {target}, obtuvo "
                f"{len(result[split_name])}."
            )
    return result


ATOM_SPLIT_GROUP_FIELDS = [
    "raw_displacement_run_id",
    "raw_fc_run_dir",
    "displacement_input",
    "displacement_ang",
    "atom",
    "direction",
    "sign",
]


def _metadata_float(metadata: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metadata.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _positions_payload_for_group(sample_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    positions = metadata.get("positions_ang")
    if isinstance(positions, list) and positions:
        return {
            "positions_ang": positions,
            "pseudopotentials": metadata.get("pseudopotentials", []),
        }
    run_fdf = sample_dir / "RUN.fdf"
    coords: list[list[Any]] = []
    if run_fdf.exists():
        inside = False
        for line in run_fdf.read_text(encoding="utf-8", errors="ignore").splitlines():
            clean = line.split("#", 1)[0].strip()
            lower = clean.lower()
            if lower.startswith("%block atomiccoordinatesandatomicspecies"):
                inside = True
                continue
            if inside and lower.startswith("%endblock atomiccoordinatesandatomicspecies"):
                break
            if not inside or not clean:
                continue
            parts = clean.split()
            if len(parts) >= 4:
                coords.append(parts[:4])
    return {"run_fdf_coordinates": coords}


def atom_zero_reference_group_id(sample_dir: Path, metadata: dict[str, Any] | None = None) -> str | None:
    metadata = metadata if metadata is not None else load_sample_metadata(sample_dir)
    displacement = _metadata_float(metadata, "displacement_ang", "displacement_magnitude")
    if displacement is None or abs(displacement) > 1e-12:
        return None
    if any(str(metadata.get(key) or "").strip() for key in ("atom", "direction", "sign")):
        return None
    payload = _positions_payload_for_group(sample_dir, metadata)
    digest = stable_payload_hash(payload, length=16)
    return f"fc_zero_reference|{digest}"


def atom_split_group_id(sample_dir: Path) -> str:
    metadata = load_sample_metadata(sample_dir)
    zero_reference_group = atom_zero_reference_group_id(sample_dir, metadata)
    if zero_reference_group:
        return zero_reference_group
    if metadata.get("split_group_id"):
        return str(metadata["split_group_id"])
    values = {field: metadata.get(field, "") for field in ATOM_SPLIT_GROUP_FIELDS}
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    readable = "|".join(str(values[field]) for field in ATOM_SPLIT_GROUP_FIELDS if values[field] not in (None, ""))
    return readable or digest


def atom_visible_split_samples(items: list[Path]) -> list[Path]:
    return [item for item in items if atom_zero_reference_group_id(item) is None]


def split_grouped_exact(items: list[Path], counts: dict[str, int]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for item in atom_visible_split_samples(items):
        groups[atom_split_group_id(item)].append(item)
    ordered_groups = [(key, sorted(value, key=sample_sort_key)) for key, value in sorted(groups.items())]
    result = {"train": [], "validation": [], "test": []}
    for split_name in ("test", "validation", "train"):
        target = counts[split_name]
        for key, samples in list(ordered_groups):
            if len(result[split_name]) + len(samples) > target:
                continue
            result[split_name].extend(samples)
            ordered_groups.remove((key, samples))
            if len(result[split_name]) == target:
                break
        if len(result[split_name]) != target:
            raise RuntimeError(
                "AtomDisplacement grouped split no puede satisfacer los tamanos exactos "
                f"sin partir familias: {split_name} necesita {target}, obtuvo {len(result[split_name])}."
            )
    return result


def sample_names(samples: list[Path]) -> str:
    return ", ".join(path.name for path in samples) if samples else "-"


def sample_sort_key(path: Path) -> tuple[int, str]:
    if path.name == "dataset":
        return (-1, path.name)
    if path.name.isdigit():
        return (int(path.name), path.name)
    suffix = path.name.rsplit("_", 1)[-1]
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (10**9, path.name)


def natural_matrix_sort_key(path: Path) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in path.stem.replace("-", ".").split("."):
        if chunk.isdigit():
            parts.append(int(chunk))
    return tuple(parts) if parts else (10**9,)


def read_system_label(run_fdf: Path) -> str:
    for line in run_fdf.read_text(encoding="utf-8", errors="ignore").splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        parts = clean.split()
        if len(parts) >= 2 and parts[0].lower() == "systemlabel":
            return parts[1]
    return "siesta"


def reference_matrices(sample_dir: Path) -> list[Path]:
    return reference_candidates(sample_dir)


def _metadata_reference_matrix(sample_dir: Path) -> Path | None:
    """Return the matrix explicitly recorded in metadata, preferring the copy inside sample_dir.

    Normalized FC samples store an absolute matrix_file in metadata. After samples are
    copied into train/test/validation folders, the same filename usually exists inside
    the copied sample directory. Prefer that local copy so manifests are self-contained.
    """
    metadata = load_sample_metadata(sample_dir)
    raw_value = metadata.get("matrix_file") or metadata.get("hamiltonian_path")
    if not raw_value:
        return None

    raw_path = Path(str(raw_value))
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(sample_dir / raw_path.name)
        candidates.append(raw_path)
    else:
        candidates.append(sample_dir / raw_path)
        candidates.append(sample_dir / raw_path.name)

    for candidate in candidates:
        if (
            candidate.exists()
            and candidate.is_file()
            and candidate.suffix in {".TSHS", ".HSX"}
            and candidate.name != "ML_prediction.HSX"
        ):
            return candidate
    return None


def _canonical_reference_matrix(sample_dir: Path) -> Path | None:
    run_fdf = sample_dir / "RUN.fdf"
    if run_fdf.exists():
        system_label = read_system_label(run_fdf)
        for candidate in (
            sample_dir / f"{system_label}.TSHS",
            sample_dir / f"{system_label}.HSX",
        ):
            if candidate.exists() and candidate.is_file():
                return candidate

    for candidate in (
        sample_dir / "siesta.TSHS",
        sample_dir / "siesta.HSX",
    ):
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def _choose_reference_matrix(sample_dir: Path) -> tuple[Path | None, str]:
    """Choose one SIESTA reference matrix with the shared strict policy."""
    selection = strict_choose_reference_matrix(sample_dir)
    return selection.path, selection.reason


def find_reference_matrix(sample_dir: Path) -> Path | None:
    matrix, reason = _choose_reference_matrix(sample_dir)
    if reason == "ok":
        return matrix
    return None


def sample_output_status(run_out: Path) -> tuple[bool, str]:
    if not run_out.exists():
        return False, "missing_output"
    try:
        text = run_out.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False, "parser_error"
    reasons = []
    if "Job completed" not in text:
        reasons.append("job_not_completed")
    if "SCF cycle converged" not in text:
        reasons.append("scf_not_converged")
    return not reasons, "ok" if not reasons else ";".join(reasons)


def validated_reference_for_sample(sample_dir: Path) -> tuple[Path | None, bool, str]:
    reasons = []

    if not (sample_dir / "RUN.fdf").exists():
        reasons.append("missing_run_fdf")

    selection = strict_choose_reference_matrix(sample_dir)
    matrix = selection.path if selection.ok else None
    if not selection.ok:
        reasons.append(selection.reason)

    output_ok, output_reason = sample_output_status(sample_dir / "RUN.out")
    if not output_ok:
        reasons.extend(output_reason.split(";"))

    valid = not reasons
    return matrix, valid, "ok" if valid else ";".join(reasons)


def load_sample_metadata(sample_dir: Path) -> dict[str, Any]:
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def atom_displacement_family(metadata: dict[str, Any]) -> str:
    parts = [
        metadata.get("generation_mode"),
        metadata.get("raw_displacement_run_id"),
        metadata.get("matrix_label"),
        metadata.get("direction"),
        metadata.get("sign"),
    ]
    return "|".join(str(part) for part in parts if part not in (None, ""))


def write_atom_split_manifests(
    dataset_dir: Path,
    split_samples: dict[str, list[Path]],
    *,
    method_id: str = "atom_displacement",
    sample_prefix: str = "atomdisp",
    split_strategy: str = "grouped_exact",
) -> dict[str, Path]:
    split_root = dataset_dir / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    paths_by_split: dict[str, Path] = {}
    for split_name, sample_dirs in split_samples.items():
        rows: list[dict[str, Any]] = []
        for sample_dir in sample_dirs:
            copied_dir = {
                "train": dataset_dir / "train_samples",
                "validation": dataset_dir / "validation_samples",
                "test": dataset_dir / "test_samples",
            }[split_name] / sample_dir.name
            metadata = load_sample_metadata(copied_dir)
            structure_path = copied_dir / "RUN.fdf"
            hamiltonian_path = find_reference_matrix(copied_dir)
            run_out_path = copied_dir / "RUN.out"
            if not run_out_path.exists() and metadata.get("raw_fc_run_dir"):
                raw_run_out = Path(str(metadata["raw_fc_run_dir"])) / "RUN.out"
                if raw_run_out.exists():
                    run_out_path = raw_run_out
            metadata_path = copied_dir / "metadata.json"
            hamiltonian_path, valid, validation_reason = validated_reference_for_sample(copied_dir)
            displacement = metadata.get("displacement_ang", "")
            group_id = atom_split_group_id(copied_dir)
            block_config = metadata.get("block_config") if isinstance(metadata.get("block_config"), dict) else {}
            rows.append(
                {
                    "sample_id": f"{sample_prefix}_{sample_dir.name}",
                    "method": method_id,
                    "source_run": str(metadata.get("raw_fc_run_dir") or sample_dir.parent),
                    "frame_index": "",
                    "time_index": "",
                    "displacement_amplitude": displacement,
                    "displacement_magnitude": displacement,
                    "displaced_atom": metadata.get("atom", ""),
                    "displacement_axis": metadata.get("direction", ""),
                    "displacement_sign": metadata.get("sign", ""),
                    "displacement_family": atom_displacement_family(metadata),
                    "structure_path": str(structure_path),
                    "hamiltonian_path": str(hamiltonian_path or ""),
                    "output_path": str(run_out_path) if run_out_path.exists() else "",
                    "run_out_path": str(run_out_path) if run_out_path.exists() else "",
                    "metadata_path": str(metadata_path) if metadata_path.exists() else "",
                    "valid": valid,
                    "validation_reason": validation_reason,
                    "split": split_name,
                    "split_group_id": group_id,
                    "split_group_fields": ",".join(ATOM_SPLIT_GROUP_FIELDS)
                    if method_id != "random_cartesian"
                    else (
                        "base_geometry_hash,distribution,sigma_ang,uniform_range_ang,"
                        "seed_family,move_atoms,species_filter,recipe_id,block_id"
                    ),
                    "split_strategy": split_strategy,
                    "random_cartesian_family_id": metadata.get("random_cartesian_family_id", group_id)
                    if method_id == "random_cartesian"
                    else "",
                    "base_geometry_hash": metadata.get("base_geometry_hash", "")
                    if method_id == "random_cartesian"
                    else "",
                    "distribution": metadata.get("distribution", "")
                    if method_id == "random_cartesian"
                    else "",
                    "sigma_ang": metadata.get("sigma_ang", "")
                    if method_id == "random_cartesian"
                    else "",
                    "uniform_range_ang": metadata.get("uniform_range_ang", "")
                    if method_id == "random_cartesian"
                    else "",
                    "seed_family": metadata.get("seed_family", metadata.get("seed", ""))
                    if method_id == "random_cartesian"
                    else "",
                    "move_atoms": json.dumps(metadata.get("move_atoms", block_config.get("move_atoms", "all")), sort_keys=True)
                    if method_id == "random_cartesian"
                    else "",
                    "species_filter": json.dumps(metadata.get("species_filter", block_config.get("species_filter", [])), sort_keys=True)
                    if method_id == "random_cartesian"
                    else "",
                    "recipe_id": metadata.get("recipe_id", "")
                    if method_id == "random_cartesian"
                    else metadata.get("recipe_id", ""),
                    "block_id": metadata.get("block_id", "")
                    if method_id == "random_cartesian"
                    else metadata.get("block_id", ""),
                    "seed": metadata.get("seed", metadata.get("subsampling", {}).get("seed", "")),
                    "status": "completed" if valid else "incomplete",
                    "sample_dir": str(copied_dir),
                }
            )
        manifest_path = split_root / f"{split_name}_manifest.csv"
        write_csv_dicts(manifest_path, rows, SPLIT_MANIFEST_FIELDS)
        paths_by_split[split_name] = manifest_path
    return paths_by_split


def is_completed_atom_sample(sample_dir: Path) -> bool:
    _matrix, valid, _reason = validated_reference_for_sample(sample_dir)
    return valid


def atom_source_samples_dir(spec: PipelineSpec, config: dict[str, Any]) -> Path:
    dataset_dir = resolve_pipeline_path(spec, config["paths"]["dataset_dir"])
    fc_steps_dir = dataset_dir / "FC_steps"
    if fc_steps_dir.exists():
        return fc_steps_dir
    atdis_steps_dir = dataset_dir / "AtDis_steps"
    if atdis_steps_dir.exists():
        return atdis_steps_dir
    configured_samples = resolve_pipeline_path(spec, config["paths"]["samples_dir"])
    if configured_samples.exists():
        return configured_samples
    raise RuntimeError(
        "AtomDisplacement: no se encontro un dataset valido. "
        f"Busque {atdis_steps_dir} y {configured_samples}."
    )


def _csv_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "valid"}


def validated_atom_sample_paths_from_validation(dataset_dir: Path) -> set[Path]:
    """Return FC_steps sample dirs accepted by run_single_points validation.

    This prevents Comparison from re-invalidating samples that the AtomDisplacement
    pipeline has already validated and written to dataset/validation/valid_samples.csv.
    """
    valid_csv = dataset_dir / "validation" / "valid_samples.csv"
    if not valid_csv.exists():
        return set()

    paths: set[Path] = set()
    with valid_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_dir_value = row.get("sample_dir") or ""
            hamiltonian_value = row.get("hamiltonian_path") or ""
            run_fdf_value = row.get("run_fdf") or row.get("structure_path") or ""
            status_value = row.get("status") or ""
            valid_value = row.get("valid")

            sample_dir = Path(str(sample_dir_value)) if sample_dir_value else None
            hamiltonian_path = Path(str(hamiltonian_value)) if hamiltonian_value else None
            run_fdf_path = Path(str(run_fdf_value)) if run_fdf_value else None

            if sample_dir is None or not sample_dir.exists():
                continue
            if hamiltonian_path is None or not hamiltonian_path.exists():
                continue
            if run_fdf_path is None or not run_fdf_path.exists():
                continue
            if valid_value is not None and not _csv_bool(valid_value):
                continue
            if status_value and str(status_value).strip().lower() not in {"valid", "completed", "skipped_validated"}:
                continue

            paths.add(sample_dir.resolve())

    return paths


def is_completed_atom_sample_with_validation_csv(sample_dir: Path, validated_paths: set[Path]) -> bool:
    if sample_dir.resolve() in validated_paths:
        return True
    return is_completed_atom_sample(sample_dir)


def completed_atom_samples(source_samples_dir: Path) -> list[Path]:
    if source_samples_dir.name in {"AtDis_steps", "FC_steps"}:
        return sorted(
            [
                path
                for path in source_samples_dir.iterdir()
                if path.is_dir() and path.name.isdigit() and is_completed_atom_sample(path)
            ],
            key=sample_sort_key,
        )
    return sorted(
        (
            path
            for path in source_samples_dir.glob("sample_*")
            if path.is_dir() and is_completed_atom_sample(path)
        ),
        key=sample_sort_key,
    )


def generated_atom_samples(source_samples_dir: Path) -> list[Path]:
    if source_samples_dir.name in {"AtDis_steps", "FC_steps"}:
        return sorted(
            [
                path
                for path in source_samples_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ],
            key=sample_sort_key,
        )
    return sorted(
        (path for path in source_samples_dir.glob("sample_*") if path.is_dir()),
        key=sample_sort_key,
    )


def copy_sample_dirs(sample_dirs: list[Path], destination_root: Path) -> int:
    destination_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for sample_dir in sample_dirs:
        destination = destination_root / sample_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        for src in sorted(path for path in sample_dir.iterdir() if path.is_file()):
            shutil.copy2(src, destination / src.name)
        stale_prediction = destination / "ML_prediction.HSX"
        if stale_prediction.exists():
            stale_prediction.unlink()
        normalize_siesta_matrix_name(destination)
        count += 1
    return count


def normalize_siesta_matrix_name(sample_dir: Path) -> None:
    run_fdf = sample_dir / "RUN.fdf"
    if not run_fdf.exists():
        return
    system_label = read_system_label(run_fdf)
    canonical_hsx = sample_dir / f"{system_label}.HSX"
    canonical_tshs = sample_dir / f"{system_label}.TSHS"
    if canonical_hsx.exists() or canonical_tshs.exists():
        return
    candidates = sorted(sample_dir.glob(f"{system_label}*.TSHS"), key=natural_matrix_sort_key)
    if not candidates:
        return
    shutil.copy2(candidates[-1], canonical_tshs)


def copy_basis_files(source_samples_dir: Path, destination_dir: Path) -> int:
    source_basis_dir = source_samples_dir / "basis"
    if not source_basis_dir.exists():
        source_basis_dir = source_samples_dir.parent / "basis"
    if not source_basis_dir.exists():
        return 0
    destination_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(source_basis_dir.glob("*.ion.xml")):
        shutil.copy2(src, destination_dir / src.name)
        count += 1
    return count


def copy_pseudopotentials(source_dir: Path, destination_dir: Path) -> int:
    psf_files = sorted(source_dir.glob("*.psf"))
    if not psf_files:
        raise RuntimeError(f"No se encontraron pseudopotenciales .psf en {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for psf_file in psf_files:
        shutil.copy2(psf_file, destination_dir / psf_file.name)
    return len(psf_files)


def copy_material_pseudopotentials(config: dict[str, Any], destination_dir: Path) -> int:
    resolved = resolve_material_bundle(
        config,
        base_dir=REPO_ROOT,
        allow_legacy_default=True,
        allow_absolute_paths=True,
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for _label, source in sorted(resolved.validated.pseudopotentials.items()):
        shutil.copy2(source, destination_dir / source.name)
        count += 1
    return count


def copy_relaxed_basis(source_dir: Path, destination_dir: Path) -> dict[str, int]:
    basis_files = sorted(source_dir.glob("*.ion.xml"))
    if not basis_files:
        raise RuntimeError(f"No se encontraron basis files .ion.xml en {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for basis_file in basis_files:
        shutil.copy2(basis_file, destination_dir / basis_file.name)
    xv_count = 0
    for xv_file in sorted(source_dir.glob("*.XV")):
        shutil.copy2(xv_file, destination_dir / xv_file.name)
        xv_count += 1
    return {"basis_files": len(basis_files), "xv_files": xv_count}


def read_metrics_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        return {"exists": True, "error": str(exc)}

    numeric_columns: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            try:
                numeric_columns.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                continue
    means = {
        key: sum(values) / len(values)
        for key, values in numeric_columns.items()
        if values
    }
    return {"exists": True, "rows": len(rows), "means": means}


def find_latest_checkpoint(training_dir: Path, config: dict[str, Any]) -> Path | None:
    manifest_path = training_dir / "checkpoint_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint_value = manifest.get("checkpoint_path") or manifest.get("path")
            if checkpoint_value:
                candidate = Path(str(checkpoint_value))
                if not candidate.is_absolute():
                    candidate = training_dir / candidate
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    configured = config.get("checkpoint", {}).get("path")
    if configured:
        candidate = Path(str(configured))
        if not candidate.is_absolute():
            candidate = training_dir / candidate
        if candidate.exists():
            return candidate
    search_glob = str(config.get("checkpoint", {}).get("search_glob", "lightning_logs/**/checkpoints/*.ckpt"))
    candidates = sorted(
        [path for path in training_dir.glob(search_glob) if path.is_file()],
        key=lambda path: path.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]
    fallback = sorted(training_dir.rglob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    return fallback[-1] if fallback else None


def checkpoint_selection_warning(training_dir: Path, checkpoint_path: Path | None) -> str:
    manifest_path = training_dir / "checkpoint_manifest.json"
    if checkpoint_path is None:
        return "No checkpoint was selected."
    if not manifest_path.exists():
        return "Checkpoint selected by latest-version fallback; strict scientific comparison should use checkpoint_manifest.json."
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Checkpoint manifest could not be parsed: {exc}"
    expected = manifest.get("checkpoint_path") or manifest.get("path")
    if not expected:
        return "Checkpoint manifest does not define checkpoint_path."
    expected_path = Path(str(expected))
    if not expected_path.is_absolute():
        expected_path = training_dir / expected_path
    if expected_path.resolve() != checkpoint_path.resolve():
        return "Selected checkpoint does not match checkpoint_manifest.json."
    expected_hash = manifest.get("checkpoint_sha256") or manifest.get("sha256")
    actual_hash = file_sha256(checkpoint_path)
    if expected_hash and actual_hash and str(expected_hash) != str(actual_hash):
        return "Selected checkpoint hash does not match checkpoint_manifest.json."
    return ""


def checkpoint_metadata(checkpoint_path: Path | None, training_dir: Path) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"path": None, "sha256": None, "relative_path": None, "selection": None}
    relative_path = None
    try:
        relative_path = checkpoint_path.relative_to(training_dir).as_posix()
    except ValueError:
        relative_path = str(checkpoint_path)
    version = None
    for part in checkpoint_path.parts:
        if part.startswith("version_") and part.removeprefix("version_").isdigit():
            version = int(part.removeprefix("version_"))
    return {
        "path": str(checkpoint_path),
        "relative_path": relative_path,
        "sha256": file_sha256(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else None,
        "mtime": checkpoint_path.stat().st_mtime if checkpoint_path.exists() else None,
        "version": version,
        "selection": "manifest_signed_checkpoint",
    }


def write_checkpoint_manifest(training_dir: Path, metadata: dict[str, Any], warning: str) -> Path:
    path = training_dir / "checkpoint_manifest.json"
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload.update(existing)
        except Exception:
            payload = {}
    payload.update({
        "checkpoint_path": metadata.get("path"),
        "checkpoint_sha256": metadata.get("sha256"),
        "relative_path": metadata.get("relative_path"),
        "selection_reason": "manifest" if not warning else "latest_version_fallback",
        "checkpoint_selection_warning": warning,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "training_dir": str(training_dir),
    })
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value in (None, ""):
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            parsed["reproducibility_warning"] = sanitize_reproducibility_warning(
                parsed.get("reproducibility_warning")
            )
            rows.append(parsed)
    derive_frontier_metrics(rows)
    return rows


DATASET_SIZE_MINIMUM_SCRIPT = Path(__file__).resolve().parent / "g2m_deeph_dataset_size_minimum.py"
DATASET_SIZE_MINIMUM_CRITERIA: list[dict[str, Any]] = [
    {
        "id": "N_min_abs",
        "label": "N_min_abs",
        "description": "Primer N donde y(N) cruza el umbral absoluto (meV).",
        "requires_threshold": True,
        "plot_dash": "dash",
    },
    {
        "id": "N_min_rel_tol",
        "label": "N_min_rel_tol",
        "description": "Primer N dentro de la tolerancia relativa respecto al mejor valor observado/ajustado.",
        "requires_threshold": False,
        "plot_dash": "dot",
    },
    {
        "id": "N_min_plateau",
        "label": "N_min_plateau",
        "description": "Primer N donde la mejora futura restante cae por debajo del umbral de plateau.",
        "requires_threshold": False,
        "plot_dash": "dashdot",
    },
    {
        "id": "N_min_cost_eff",
        "label": "N_min_cost_eff",
        "description": "Menor N con coste minimo entre puntos ya cercanos al mejor observado.",
        "requires_threshold": False,
        "plot_dash": "longdash",
    },
]


def active_g2m_deeph_run_roots() -> set[str]:
    """Run roots that must not be post-processed while a benchmark is active."""
    roots: set[str] = set()
    try:
        status = G2M_DEEPH_RUNNER.status()
    except Exception:
        return roots
    if status.get("running") and status.get("run_root"):
        roots.add(str(Path(str(status["run_root"])).resolve()))
    return roots


def _dataset_size_minimum_summary_paths() -> list[Path]:
    patterns = (
        "dataset_size_minimum_existing_sweeps_*/dataset_size_minimum_summary.json",
        "dataset_size_minimum_existing_sweeps_*/*/dataset_size_minimum_summary.json",
        "dataset_size_minimum_ui_*/dataset_size_minimum_summary.json",
        "dataset_size_minimum_ui_*/*/dataset_size_minimum_summary.json",
        "dataset_size_minimum_*/dataset_size_minimum_summary.json",
        "dataset_size_minimum_*/**/dataset_size_minimum_summary.json",
    )
    found: list[Path] = []
    for pattern in patterns:
        found.extend(RESULTS_ROOT.glob(pattern))
    return sorted(set(found), key=lambda path: path.stat().st_mtime, reverse=True)



def _dataset_size_minimum_candidate_run_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        if not path.is_dir():
            return
        resolved = str(path.resolve())
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(path)

    # Any graphene campaign root (w90/5x2/5x5 scaling, derivative smokes...):
    # one iterdir level plus fixed-path probes, never a recursive glob.
    for campaign_root in sorted(RESULTS_ROOT.glob("graphene_*")):
        if not campaign_root.is_dir():
            continue

        if (campaign_root / "summary").exists() or (campaign_root / "sweep").exists():
            add(campaign_root)

        for child in sorted(campaign_root.iterdir()):
            if child.is_dir():
                add(child)

    for run_root in sorted(RESULTS_ROOT.glob("g2m_deeph_*/runs/*")):
        add(run_root)

    return roots


def _dataset_size_minimum_rows_from_metric_file(path: Path) -> list[dict[str, Any]]:
    return load_dataset_minimum_metric_file_rows(path, explicit_run_root_mode=False)


def iter_dataset_size_minimum_metric_sources() -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    seen: set[str] = set()

    for run_root in _dataset_size_minimum_candidate_run_roots():
        for metric_path in discover_dataset_minimum_metric_files(run_root):
            if not metric_path.is_file():
                continue
            resolved = str(metric_path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append((run_root.resolve(), metric_path.resolve()))

    return sorted(found, key=lambda item: item[1].stat().st_mtime, reverse=True)


def iter_dataset_size_minimum_metrics_paths() -> list[Path]:
    """Find normalized ranking metrics without scanning all of Comparison/results.

    A recursive ``**/summary/ranking/normalized_run_metrics.json`` glob can take
    more than a minute on large result trees and leave the UI with an empty
    sweep list. Limit discovery to known benchmark layouts instead.
    """
    found: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        resolved = str(path.resolve())
        if resolved in seen or not path.is_file():
            return
        seen.add(resolved)
        found.append(path)

    # Any graphene campaign root, same non-recursive discovery as
    # _dataset_size_minimum_candidate_run_roots.
    for campaign_root in sorted(RESULTS_ROOT.glob("graphene_*")):
        if not campaign_root.is_dir():
            continue
        add(campaign_root / "summary" / "ranking" / "normalized_run_metrics.json")
        for run_root in campaign_root.iterdir():
            if not run_root.is_dir():
                continue
            add(run_root / "summary" / "ranking" / "normalized_run_metrics.json")

    for path in RESULTS_ROOT.glob(
        "graphene_w90_snapshot_scaling_*_ui_alias/summary/ranking/normalized_run_metrics.json"
    ):
        add(path)

    for path in RESULTS_ROOT.glob(
        "g2m_deeph_*/runs/*/summary/ranking/normalized_run_metrics.json"
    ):
        add(path)

    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)



def discover_dataset_size_minimum_run_roots() -> list[dict[str, Any]]:

    active_roots = active_g2m_deeph_run_roots()
    grouped_sources: dict[str, dict[str, Any]] = {}

    for run_root, metrics_path in iter_dataset_size_minimum_metric_sources():
        key = str(run_root.resolve())
        item = grouped_sources.setdefault(
            key,
            {
                "run_root": run_root.resolve(),
                "metric_files": [],
                "rows": [],
                "errors": [],
                "modified_at": 0.0,
            },
        )
        item["metric_files"].append(metrics_path)
        item["modified_at"] = max(item["modified_at"], metrics_path.stat().st_mtime)
        try:
            item["rows"].extend(_dataset_size_minimum_rows_from_metric_file(metrics_path))
        except (MetricFileLoadError, OSError, json.JSONDecodeError, csv.Error, ValueError) as exc:
            item["errors"].append(f"{metrics_path.name}:{exc}")

    items: list[dict[str, Any]] = []
    for key, payload in grouped_sources.items():
        run_root = Path(payload["run_root"])
        rows = payload["rows"]
        metric_files = payload["metric_files"]
        dataset_sizes: set[int] = set()

        for row in rows:
            for field in (
                "dataset_size",
                "dataset_size_x",
                "n_total",
                "n_train",
                "dataset_size_total",
                "dataset_size_train",
                "train_dataset_size",
                "train_size",
            ):
                value = row.get(field)
                if value in (None, ""):
                    continue
                try:
                    dataset_sizes.add(int(round(float(value))))
                    break
                except (TypeError, ValueError):
                    continue

        is_active = key in active_roots
        incomplete = len(rows) == 0
        blocked_reason = None
        if is_active:
            blocked_reason = "sweep_en_curso"
        elif incomplete:
            blocked_reason = "sin_metricas"

        items.append(
            {
                "run_root": key,
                "label": run_root.name,
                "metrics_path": str(metric_files[0]) if metric_files else "",
                "metric_files": [str(path) for path in metric_files],
                "metric_rows": len(rows),
                "dataset_sizes": sorted(dataset_sizes),
                "selectable": not is_active and not incomplete,
                "blocked_reason": blocked_reason,
                "discovery_errors": payload["errors"],
                "modified_at": payload["modified_at"],
            }
        )

    label_counts: dict[str, int] = {}
    for item in items:
        label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1
    if any(count > 1 for count in label_counts.values()):
        # Disambiguate items that share the same run_root.name (e.g. an
        # "OUTER" campaign root and a same-named "INNER" nested run root)
        # so the UI never offers two selectable entries with identical,
        # indistinguishable labels. The run_root path itself is untouched;
        # only the human-facing label gains a depth-derived suffix.
        for item in items:
            if label_counts.get(item["label"], 0) <= 1:
                continue
            run_root_path = Path(item["run_root"])
            try:
                suffix = str(run_root_path.relative_to(RESULTS_ROOT.resolve()))
            except ValueError:
                suffix = str(run_root_path)
            item["label"] = f"{item['label']} ({suffix})" if suffix else item["label"]

    return sorted(items, key=lambda item: item.get("modified_at") or 0.0, reverse=True)


def dataset_size_minimum_available_combinations(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combinations: list[dict[str, Any]] = []
    seen: set[tuple[str, float | None]] = set()
    for output in outputs:
        if output.get("status") != "ok":
            continue
        metric = str(output.get("primary_metric") or "")
        threshold = output.get("threshold_mev")
        try:
            threshold_value = float(threshold) if threshold not in (None, "") else None
        except (TypeError, ValueError):
            threshold_value = None
        key = (metric, threshold_value)
        if key in seen:
            continue
        seen.add(key)
        combinations.append(
            {
                "primary_metric": metric,
                "threshold_mev": threshold_value,
                "label": f"{metric} · {threshold_value:g} meV" if threshold_value is not None else metric,
            }
        )
    combinations.sort(
        key=lambda item: (
            str(item.get("primary_metric") or ""),
            item.get("threshold_mev") if item.get("threshold_mev") is not None else math.inf,
        )
    )
    return combinations



def dataset_size_minimum_command_text(command: list[str] | None) -> str:
    if not command:
        return "<command_not_built>"
    return " ".join(shlex.quote(str(part)) for part in command)


def resolve_dataset_size_minimum_run_roots(run_roots_raw: list[Any]) -> list[Path]:
    if not isinstance(run_roots_raw, list) or not run_roots_raw:
        raise RuntimeError("Envia run_roots como lista no vacia de sweeps terminados.")

    run_roots: list[Path] = []
    for item in run_roots_raw:
        path = Path(str(item)).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        else:
            path = path.resolve()

        if not path.exists():
            raise RuntimeError(f"run_root no encontrado: {path}")

        metric_files = discover_dataset_minimum_metric_files(path)
        if not metric_files:
            raise RuntimeError(f"run_root sin metricas compatibles para dataset-size-minimum: {path}")

        try:
            loaded_rows, _sources, root_warnings = dataset_minimum_load_run_root_rows(
                path,
                explicit_run_root_mode=True,
            )
        except MetricFileLoadError as exc:
            raise RuntimeError(str(exc)) from exc

        usable_rows = len(loaded_rows)
        if usable_rows <= 0:
            detail = f" ({'; '.join(root_warnings)})" if root_warnings else ""
            raise RuntimeError(
                f"run_root sin filas metricas compatibles para dataset-size-minimum: {path}{detail}"
            )

        run_roots.append(path)

    if not run_roots:
        raise RuntimeError("No hay run_roots validos para dataset-size-minimum.")

    blocked_roots = active_g2m_deeph_run_roots()
    for path in run_roots:
        if str(path) in blocked_roots:
            raise RuntimeError(
                f"El sweep {path.name} esta en curso; no se ejecutara el postproceso sobre el."
            )

    return run_roots


def dataset_size_minimum_output_matches_controls(
    output: dict[str, Any],
    controls: dict[str, Any],
) -> bool:
    """Mirror of frontend output-selection compatibility checks."""
    if output.get("primary_metric") != controls.get("primary_metric"):
        return False
    if (output.get("x_axis") or "n_train") != (controls.get("x_axis") or "n_train"):
        return False

    threshold = float(controls.get("threshold_mev") or 0.0)
    out_threshold = output.get("threshold_mev")
    if out_threshold is None or abs(float(out_threshold) - threshold) >= 1e-9:
        return False
    if bool(output.get("threshold_is_user_defined")) != bool(controls.get("threshold_is_user_defined")):
        return False
    if not bool(controls.get("threshold_is_user_defined")):
        if str(output.get("threshold_preset_key") or "") != str(controls.get("threshold_preset_key") or ""):
            return False

    if str(output.get("requested_n_min_source") or output.get("n_min_source") or "observed") != str(
        controls.get("n_min_source") or "observed"
    ):
        return False

    output_fit = output.get("requested_fit_model") or output.get("n_min_fit_model") or CANONICAL_POWER_LAW_MODEL
    control_fit = controls.get("n_min_fit_model") or CANONICAL_POWER_LAW_MODEL
    if not dataset_minimum_fit_models_equivalent(str(output_fit), str(control_fit)):
        return False

    out_mode = output.get("aggregation_mode")
    if not out_mode:
        roots = output.get("run_roots") or []
        out_mode = "mean_replicates" if len(roots) > 1 else "best_config"
    if str(out_mode) != str(controls.get("aggregation_mode") or "mean_replicates"):
        return False

    selected_fit = str(controls.get("n_min_fit_model") or "power_law")
    if selected_fit == "moving_average":
        if int(output.get("moving_average_window") or 3) != int(
            controls.get("moving_average_window") or 3
        ):
            return False

    if int(output.get("bootstrap_replicates") or 0) != int(controls.get("bootstrap_replicates") or 0):
        return False

    out_ci = float(output.get("ci_level") or 0.95)
    ctrl_ci = float(controls.get("ci_level") or 0.95)
    if abs(out_ci - ctrl_ci) >= 1e-9:
        return False

    selected_roots = sorted(
        {str(Path(item).resolve()) for item in (controls.get("run_roots") or []) if item}
    )
    output_roots = sorted(
        {str(Path(item).resolve()) for item in (output.get("run_roots") or []) if item}
    )
    if selected_roots and output_roots != selected_roots:
        return False
    return True


def parse_dataset_size_minimum_aggregation_mode(
    payload: dict[str, Any],
    *,
    run_root_count: int,
) -> str:
    value = payload.get("aggregation_mode")
    if value is not None and str(value).strip():
        mode = str(value).strip()
        if mode not in AGGREGATION_MODES:
            raise RuntimeError(
                "aggregation_mode debe ser uno de: "
                + ", ".join(AGGREGATION_MODES)
            )
        return mode
    return resolve_dataset_minimum_aggregation_mode(None, run_root_count=run_root_count)


def dataset_size_minimum_output_aggregation_metadata(output: dict[str, Any]) -> dict[str, Any]:
    requested = output.get("requested_aggregation_mode")
    actual = output.get("actual_aggregation_mode") or output.get("aggregation_mode")
    legacy_inferred = bool(output.get("aggregation_mode_legacy_inferred"))

    if not actual:
        roots = output.get("run_roots") or []
        actual = "mean_replicates" if len(roots) > 1 else "best_config"
        legacy_inferred = True

    classification = output.get("aggregation_mode_classification")
    reason = output.get("aggregation_mode_classification_reason")
    if not classification or not reason:
        classification, reason = dataset_minimum_aggregation_mode_classification(str(actual))

    return {
        "requested_aggregation_mode": requested,
        "actual_aggregation_mode": str(actual),
        "aggregation_mode_legacy_inferred": legacy_inferred,
        "aggregation_mode_classification": classification,
        "aggregation_mode_classification_reason": reason,
    }


def dataset_size_minimum_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Build plot-ready best_rows from completed sweeps without writing artifacts."""
    run_roots = resolve_dataset_size_minimum_run_roots(payload.get("run_roots") or [])

    primary_metric = str(payload.get("primary_metric") or "h_mae_eV_mean").strip()
    x_axis = str(payload.get("x_axis") or "n_train").strip()
    if x_axis not in {"n_total", "n_train"}:
        raise RuntimeError("x_axis debe ser n_total o n_train.")

    sources_by_root = {
        str(item.get("run_root") or ""): item for item in discover_dataset_size_minimum_run_roots()
    }

    aggregation_mode = parse_dataset_size_minimum_aggregation_mode(
        payload,
        run_root_count=len(run_roots),
    )
    aggregation_classification, aggregation_reason = dataset_minimum_aggregation_mode_classification(
        aggregation_mode
    )
    all_normalized: list[dict[str, Any]] = []
    warnings: list[str] = []
    resolved_roots: list[str] = []

    for run_root in run_roots:
        key = str(run_root.resolve())
        source_meta = sources_by_root.get(key) or {}
        if source_meta.get("selectable") is False:
            reason = source_meta.get("blocked_reason") or "no_selectable"
            raise RuntimeError(f"El sweep {run_root.name} no es selectable: {reason}")

        loaded, _sources, root_warnings = dataset_minimum_load_run_root_rows(
            run_root,
            explicit_run_root_mode=True,
        )
        normalized, normalize_warnings = dataset_minimum_normalize_rows(
            loaded,
            primary_metric=primary_metric,
            x_axis=x_axis,
        )
        sweep_label = str(source_meta.get("label") or run_root.name)
        for row in normalized:
            enriched = dict(row)
            enriched["sweep_label"] = sweep_label
            enriched["source_run_root"] = enriched.get("source_run_root") or key
            all_normalized.append(enriched)
        warnings.extend(root_warnings + normalize_warnings)
        if normalized:
            resolved_roots.append(key)

    grouped = dataset_minimum_group_config_rows(all_normalized)
    best_rows = dataset_minimum_analysis_rows_for_aggregation_mode(
        all_normalized,
        grouped,
        aggregation_mode=aggregation_mode,
    )
    if aggregation_mode == "mean_replicates" and len(run_roots) > 1:
        warnings.append(f"aggregated_mean_replicates_across_{len(run_roots)}_run_roots")

    for row in best_rows:
        if row.get("sweep_label"):
            continue
        method = row.get("method")
        size = row.get("dataset_size_x")
        for source_row in all_normalized:
            if source_row.get("method") == method and source_row.get("dataset_size_x") == size:
                row.setdefault("sweep_label", source_row.get("sweep_label"))
                row.setdefault("source_run_root", source_row.get("source_run_root"))
                break

    aggregated = aggregation_mode in {"mean_seeds_per_config", "best_config_mean"} or (
        aggregation_mode == "mean_replicates"
        and (
            len(run_roots) > 1
            or any(int(row.get("replicate_count") or 1) > 1 for row in best_rows)
        )
    )

    status = "ok" if best_rows else "no_usable_metric_rows"
    return {
        "status": status,
        "primary_metric": primary_metric,
        "x_axis": x_axis,
        "run_roots": resolved_roots,
        "best_rows": best_rows,
        "aggregated_rows": best_rows,
        "warnings": warnings,
        "is_preview": True,
        "aggregated": aggregated,
        "aggregation_mode": aggregation_mode,
        "requested_aggregation_mode": payload.get("aggregation_mode"),
        "actual_aggregation_mode": aggregation_mode,
        "aggregation_mode_legacy_inferred": payload.get("aggregation_mode") in (None, ""),
        "aggregation_mode_classification": aggregation_classification,
        "aggregation_mode_classification_reason": aggregation_reason,
    }


def run_dataset_size_minimum_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """Run read-only dataset-size-minimum post-processing on completed sweeps."""
    if not DATASET_SIZE_MINIMUM_SCRIPT.exists():
        raise RuntimeError(f"No se encontro el script de postproceso: {DATASET_SIZE_MINIMUM_SCRIPT}")

    n_min_source = str(payload.get("n_min_source") or "observed").strip()
    try:
        n_min_fit_model = dataset_minimum_parse_single_fit_model(
            str(payload.get("n_min_fit_model") or payload.get("fit_model") or CANONICAL_POWER_LAW_MODEL)
        )
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from None

    if n_min_source not in {"observed", "fit"}:
        raise RuntimeError("n_min_source debe ser 'observed' o 'fit'.")

    threshold_mev = float(payload.get("threshold_mev") or 10.0)
    primary_metric = str(payload.get("primary_metric") or "h_mae_eV_mean").strip()
    threshold_preset_key = payload.get("threshold_preset_key")
    threshold_is_user_defined = bool(payload.get("threshold_is_user_defined"))
    x_axis = str(payload.get("x_axis") or "n_train").strip()

    if x_axis not in {"n_total", "n_train"}:
        raise RuntimeError("x_axis debe ser n_total o n_train.")
    try:
        cost_basis = dataset_minimum_parse_cost_basis(
            str(payload.get("cost_basis") or COST_BASES[0])
        )
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from None
    try:
        claim_mode = dataset_minimum_parse_claim_mode(
            str(payload.get("claim_mode") or CLAIM_MODES[0])
        )
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from None
    threshold_protocol_file_value = payload.get("threshold_protocol_file")
    threshold_protocol_file: Path | None = None
    if threshold_protocol_file_value not in (None, ""):
        protocol_text = str(threshold_protocol_file_value).strip()
        if not protocol_text:
            raise RuntimeError("threshold_protocol_file debe ser una ruta no vacia si se proporciona.")
        threshold_protocol_file = Path(protocol_text).expanduser().resolve()
        if not threshold_protocol_file.exists():
            raise RuntimeError(f"threshold_protocol_file no existe: {threshold_protocol_file}")
        if not threshold_protocol_file.is_file():
            raise RuntimeError(f"threshold_protocol_file debe apuntar a un fichero JSON: {threshold_protocol_file}")

    relative_tolerance = float(payload.get("relative_tolerance") or 0.05)
    plateau_gain = float(payload.get("plateau_gain") or 0.05)
    fit_models = str(
        payload.get("fit_models")
        or "linear,quadratic,inverse,inverse_square,power_law_floor,power_law,"
        "lowess_logx,lowess_logx_robust,monotone_lowess_logx,moving_average,"
        "cumulative_best,none"
    )

    moving_average_window = int(payload.get("moving_average_window") or 3)
    if moving_average_window <= 0:
        raise RuntimeError("moving_average_window debe ser un entero positivo.")

    bootstrap_replicates = int(payload.get("bootstrap_replicates") or 0)
    if bootstrap_replicates < 0:
        raise RuntimeError("bootstrap_replicates debe ser >= 0.")
    bootstrap_seed = int(payload.get("bootstrap_seed") or 12345)
    ci_level = float(payload.get("ci_level") or 0.95)
    if not 0.0 < ci_level < 1.0:
        raise RuntimeError("ci_level debe estar en (0, 1).")

    run_roots = resolve_dataset_size_minimum_run_roots(payload.get("run_roots") or [])
    aggregation_mode = parse_dataset_size_minimum_aggregation_mode(
        payload,
        run_root_count=len(run_roots),
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        RESULTS_ROOT / f"dataset_size_minimum_ui_{stamp}" / f"threshold_{threshold_mev:g}meV"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        postprocess_python_executable(),
        str(DATASET_SIZE_MINIMUM_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--primary-metric",
        primary_metric,
        "--threshold-mev",
        str(threshold_mev),
        "--threshold-preset-key",
        str(threshold_preset_key or ""),
        "--threshold-is-user-defined",
        "true" if threshold_is_user_defined else "false",
        "--relative-tolerance",
        str(relative_tolerance),
        "--plateau-gain",
        str(plateau_gain),
        "--x-axis",
        x_axis,
        "--fit-models",
        fit_models,
        "--n-min-source",
        n_min_source,
        "--n-min-fit-model",
        n_min_fit_model,
        "--moving-average-window",
        str(moving_average_window),
        "--aggregation-mode",
        aggregation_mode,
        "--cost-basis",
        cost_basis,
        "--claim-mode",
        claim_mode,
        "--bootstrap-replicates",
        str(bootstrap_replicates),
        "--bootstrap-seed",
        str(bootstrap_seed),
        "--ci-level",
        str(ci_level),
    ]
    if threshold_protocol_file is not None:
        command.extend(["--threshold-protocol-file", str(threshold_protocol_file)])

    for path in run_roots:
        command.extend(["--run-root", str(path)])

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    summary_path = output_dir / "dataset_size_minimum_summary.json"
    summary: dict[str, Any] | None = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = None

    if completed.returncode != 0:
        if summary and summary.get("status") == "no_usable_metric_rows":
            return {
                "status": "no_usable_metric_rows",
                "output_dir": str(output_dir),
                "summary_path": str(summary_path),
                "command": command,
                "run_roots": [str(path) for path in run_roots],
                "summary": summary,
                "warnings": summary.get("warnings") or [],
            }

        stderr = (completed.stderr or completed.stdout or "").strip()
        command_text = " ".join(shlex.quote(str(part)) for part in command)
        raise RuntimeError(
            "Postproceso dataset-size-minimum fallo"
            + (f": {stderr}" if stderr else ".")
            + f" Comando: {command_text}"
        )

    return {
        "status": (summary or {}).get("status") or "ok",
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "command": command,
        "run_roots": [str(path) for path in run_roots],
        "summary": summary,
    }


def dataset_size_minimum_payload() -> dict[str, Any]:
    """Return read-only dataset-size-minimum post-processing outputs."""
    summary_paths = _dataset_size_minimum_summary_paths()
    outputs: list[dict[str, Any]] = []
    for summary_path in sorted(set(summary_paths), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            outputs.append(
                {
                    "status": "invalid_summary",
                    "summary_path": str(summary_path),
                    "error": str(exc),
                }
            )
            continue
        output_dir = summary_path.parent
        best_rows = read_csv_rows(output_dir / "dataset_size_minimum_best_by_size.csv")
        aggregated_rows = summary.get("aggregated_rows")
        if not isinstance(aggregated_rows, list):
            aggregated_rows = best_rows
        result_rows = read_csv_rows(output_dir / "dataset_size_minimum_results.csv")
        summary_run_roots = summary.get("run_roots") or []
        aggregation_metadata = dataset_size_minimum_output_aggregation_metadata(summary)
        aggregation_mode = aggregation_metadata["actual_aggregation_mode"]
        artifact_outputs: list[dict[str, Any]] = []
        for filename, meta in DATASET_SIZE_MINIMUM_UI_ARTIFACTS.items():
            artifact_path = output_dir / filename
            if not artifact_path.exists() or not artifact_path.is_file():
                continue
            query = urlencode({"output_dir": str(output_dir), "name": filename})
            artifact_outputs.append(
                {
                    "name": filename,
                    "label": meta["label"],
                    "kind": meta["kind"],
                    "mime_type": meta["mime_type"],
                    "path": str(artifact_path),
                    "url": f"/api/g2m-deeph/dataset-size-minimum/artifact?{query}",
                }
            )
        outputs.append(
            {
                "status": summary.get("status") or "unknown",
                "output_dir": str(output_dir),
                "summary_path": str(summary_path),
                "report_path": str(output_dir / "dataset_size_minimum_report.md"),
                "artifact_outputs": artifact_outputs,
                "primary_metric": summary.get("primary_metric"),
                "threshold_mev": summary.get("threshold_mev"),
                "threshold_basis": summary.get("threshold_basis"),
                "threshold_reference": summary.get("threshold_reference"),
                "threshold_interpretation": summary.get("threshold_interpretation"),
                "threshold_metric_family": summary.get("threshold_metric_family"),
                "threshold_is_user_defined": summary.get("threshold_is_user_defined"),
                "threshold_preset_key": summary.get("threshold_preset_key"),
                "x_axis": summary.get("x_axis"),
                "thresholds": summary.get("thresholds") or {},
                "fits": summary.get("fits") or {},
                "dataset_sizes": summary.get("dataset_sizes") or [],
                "methods": summary.get("methods") or sorted((summary.get("thresholds") or {}).keys()),
                "warnings": summary.get("warnings") or [],
                "best_rows": best_rows,
                "aggregated_rows": aggregated_rows,
                "result_rows_count": len(result_rows),
                "raw_rows_count": summary.get("raw_rows_count"),
                "run_roots": summary_run_roots,
                "modified_at": summary_path.stat().st_mtime,
                "moving_average_window": summary.get("moving_average_window"),
                "cost_basis": summary.get("cost_basis") or COST_BASES[0],
                "n_min_source": summary.get("n_min_source"),
                "n_min_fit_model": summary.get("n_min_fit_model"),
                "requested_n_min_source": summary.get("requested_n_min_source"),
                "actual_n_min_source": summary.get("actual_n_min_source"),
                "requested_fit_model": summary.get("requested_fit_model"),
                "actual_fit_model": summary.get("actual_fit_model"),
                "canonical_fit_model": summary.get("canonical_fit_model"),
                "fallback_used": summary.get("fallback_used"),
                "fallback_reason": summary.get("fallback_reason"),
                "aggregation_mode": aggregation_mode,
                **aggregation_metadata,
                "aggregated": summary.get("aggregated"),
                "replicate_bootstrap": summary.get("replicate_bootstrap") or summary.get("bootstrap") or {},
                "bootstrap": summary.get("bootstrap") or {},
                "bootstrap_deprecated_alias_for": summary.get("bootstrap_deprecated_alias_for"),
                "bootstrap_replicates": summary.get("bootstrap_replicates"),
                "bootstrap_seed": summary.get("bootstrap_seed"),
                "ci_level": summary.get("ci_level"),
                "normalized_rows": summary.get("normalized_rows") or [],
                "temporal_diagnostics": summary.get("temporal_diagnostics") or {},
                "nominal_n_train": summary.get("nominal_n_train"),
                "estimated_n_eff_train": summary.get("estimated_n_eff_train"),
                "autocorrelation_available": summary.get("autocorrelation_available"),
                "n_min_basis": summary.get("n_min_basis"),
                "N_min_nominal": summary.get("N_min_nominal") or {},
                "N_eff_diagnostic_available": summary.get("N_eff_diagnostic_available"),
                "N_eff_over_N_nominal": summary.get("N_eff_over_N_nominal"),
                "effective_samples_at_N_min_nominal": summary.get("effective_samples_at_N_min_nominal") or {},
                "N_min_eff_diagnostic": summary.get("N_min_eff_diagnostic") or {},
                "claim_mode_requested": summary.get("claim_mode_requested") or CLAIM_MODES[0],
                "claim_mode_actual": summary.get("claim_mode_actual") or CLAIM_MODES[0],
                "scientific_claim_status": summary.get("scientific_claim_status"),
                "paper_level_blockers": summary.get("paper_level_blockers") or [],
                "paper_level_warnings": summary.get("paper_level_warnings") or [],
                "n_min_protocol": summary.get("n_min_protocol") or {},
                "n_min_fit_policy": summary.get("n_min_fit_policy"),
                "n_min_fit_policy_by_method": summary.get("n_min_fit_policy_by_method") or {},
                "fit_predictive_stability_by_left_out_N": summary.get("fit_predictive_stability_by_left_out_N") or {},
                "n_eff_diagnostic_note": summary.get("n_eff_diagnostic_note"),
            }
        )
    run_root_sources = discover_dataset_size_minimum_run_roots()
    active_run_roots = sorted(active_g2m_deeph_run_roots())
    return {
        "available": any(item.get("status") == "ok" for item in outputs),
        "schema": "dataset_size_minimum_ui_payload_v2",
        "outputs": outputs,
        "criteria": DATASET_SIZE_MINIMUM_CRITERIA,
        "available_combinations": dataset_size_minimum_available_combinations(outputs),
        "run_root_sources": run_root_sources,
        "active_run_roots": active_run_roots,
        "default_run_roots": [
            item["run_root"]
            for item in run_root_sources
            if item.get("selectable")
        ][:1],
        "message": (
            f"{len(outputs)} dataset-size-minimum output(s) found."
            if outputs
            else "No dataset_size_minimum summaries found under Comparison/results."
        ),
        "diagnostic_warning": (
            "Dataset-size-minimum outputs are diagnostic post-processing unless they come from "
            "a locked paper-ready protocol with repeated final seeds and full gates."
        ),
    }


PLOT_MATERIAL_FIELDS = (
    "material_label",
    "material_source",
    "material_preset",
    "material_structure_type",
    "material_species",
    "material_atom_count",
    "material_identity_hash",
    "material_compatibility_hash",
)


def nonempty_material_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def plot_material_display_label(material: dict[str, Any]) -> str:
    label = material.get("material_label") or material.get("material_preset")
    if nonempty_material_value(label):
        return str(label)
    return "unknown material"


def plot_material_metadata(*sources: Any) -> dict[str, Any]:
    material = flatten_material_provenance(
        *(source for source in sources if isinstance(source, dict))
    )
    payload = {field: material.get(field) for field in PLOT_MATERIAL_FIELDS}
    payload["material_display_label"] = plot_material_display_label(material)
    return payload


def parse_json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def append_unique_string(values: list[str], value: Any) -> None:
    if not nonempty_material_value(value):
        return
    text = str(value)
    if text not in values:
        values.append(text)


def merge_string_mapping(target: dict[str, str], value: Any) -> None:
    for key, item in parse_json_mapping(value).items():
        if nonempty_material_value(key) and nonempty_material_value(item):
            target.setdefault(str(key), str(item))


def cross_material_summary(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    labels: list[str] = []
    identity_hashes: list[str] = []
    compatibility_hashes: list[str] = []
    warnings: list[str] = []
    label_by_method: dict[str, str] = {}
    identity_hash_by_method: dict[str, str] = {}
    compatibility_hash_by_method: dict[str, str] = {}
    rows_with_material = 0

    manifest_material = flatten_material_provenance(
        manifest.get("material_provenance") if isinstance(manifest.get("material_provenance"), dict) else {},
        manifest,
    )
    append_unique_string(labels, manifest_material.get("material_label"))
    append_unique_string(identity_hashes, manifest_material.get("material_identity_hash"))
    append_unique_string(compatibility_hashes, manifest_material.get("material_compatibility_hash"))
    manifest_maps = material_maps_from_manifest(manifest)
    merge_string_mapping(label_by_method, manifest_maps.get("material_label_by_method"))
    merge_string_mapping(identity_hash_by_method, manifest_maps.get("material_identity_hash_by_method"))
    merge_string_mapping(compatibility_hash_by_method, manifest_maps.get("material_compatibility_hash_by_method"))

    for row in rows:
        row_material_known = any(
            nonempty_material_value(row.get(field))
            for field in ("material_label", "material_identity_hash", "material_compatibility_hash")
        )
        append_unique_string(labels, row.get("material_label"))
        append_unique_string(identity_hashes, row.get("material_identity_hash"))
        append_unique_string(compatibility_hashes, row.get("material_compatibility_hash"))
        append_unique_string(warnings, row.get("material_compatibility_warning"))
        for field in MATERIAL_MAP_FIELDS:
            mapping = parse_json_mapping(row.get(field))
            if mapping:
                row_material_known = True
            if field == "material_label_by_method":
                merge_string_mapping(label_by_method, mapping)
            elif field == "material_identity_hash_by_method":
                merge_string_mapping(identity_hash_by_method, mapping)
            elif field == "material_compatibility_hash_by_method":
                merge_string_mapping(compatibility_hash_by_method, mapping)
        if row_material_known:
            rows_with_material += 1

    for value in label_by_method.values():
        append_unique_string(labels, value)
    for value in identity_hash_by_method.values():
        append_unique_string(identity_hashes, value)
    for value in compatibility_hash_by_method.values():
        append_unique_string(compatibility_hashes, value)

    warning = material_compatibility_warning(manifest_maps)
    append_unique_string(warnings, warning)

    labels = sorted(labels)
    identity_hashes = sorted(identity_hashes)
    compatibility_hashes = sorted(compatibility_hashes)
    has_unknown_material = bool(rows) and rows_with_material < len(rows) and not manifest_material
    mixed_materials = (
        len(compatibility_hashes) > 1
        or bool(warnings)
        or (not compatibility_hashes and len(labels) > 1)
    )
    display_labels = labels[:] if labels else []
    if has_unknown_material:
        append_unique_string(display_labels, "unknown material")

    return {
        "material_labels": labels,
        "material_display_labels": sorted(display_labels),
        "material_identity_hashes": identity_hashes,
        "material_compatibility_hashes": compatibility_hashes,
        "material_label_by_method": dict(sorted(label_by_method.items())),
        "material_identity_hash_by_method": dict(sorted(identity_hash_by_method.items())),
        "material_compatibility_hash_by_method": dict(sorted(compatibility_hash_by_method.items())),
        "material_compatibility_warnings": sorted(warnings),
        "has_unknown_material": has_unknown_material,
        "mixed_materials": mixed_materials,
    }


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def derive_frontier_metrics(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if finite_number(row.get("frontier_window_rmse_eV")) is not None:
            continue
        homo_error = finite_number(row.get("homo_error_eV"))
        lumo_error = finite_number(row.get("lumo_error_eV"))
        errors = [value for value in (homo_error, lumo_error) if value is not None]
        if not errors:
            continue
        row["frontier_window_bands"] = len(errors)
        row["frontier_window_mae_eV"] = sum(abs(value) for value in errors) / len(errors)
        row["frontier_window_rmse_eV"] = math.sqrt(
            sum(value * value for value in errors) / len(errors)
        )


def numeric_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    columns: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key == "sample" or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if math.isfinite(number):
                columns.setdefault(key, []).append(number)
    return {
        key: sum(values) / len(values)
        for key, values in columns.items()
        if values
    }


def bounded_plot_sample_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int = MAX_PLOT_SAMPLE_ROWS_PER_GROUP,
) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[0]]
    last_index = len(rows) - 1
    indices = {
        round(index * last_index / (limit - 1))
        for index in range(limit)
    }
    return [rows[index] for index in sorted(indices)]


def plot_sample_payload(rows_by_group: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        group: bounded_plot_sample_rows(rows)
        for group, rows in rows_by_group.items()
    }


def plot_sample_row_counts(rows_by_group: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int | bool]]:
    counts: dict[str, dict[str, int | bool]] = {}
    for group, rows in rows_by_group.items():
        shown = min(len(rows), MAX_PLOT_SAMPLE_ROWS_PER_GROUP)
        counts[group] = {
            "total": len(rows),
            "shown": shown,
            "truncated": len(rows) > shown,
        }
    return counts


def finite_metric_count(rows: list[dict[str, Any]], metric: str) -> int:
    count = 0
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            count += 1
    return count


SPECTRAL_PLOT_AVAILABILITY_METRICS = [
    "fermi_window_rmse_eV",
    "frontier_window_rmse_eV",
    "low_energy_rmse_eV",
]

CROSS_PLOT_AVAILABILITY_METRICS = [
    "fermi_window_rmse_eV",
    "frontier_window_rmse_eV",
    "low_energy_rmse_eV",
    "global_rmse_eV",
    "relative_frobenius_union",
    "mae_ref_eV",
    "mse_ref_eV2",
    "mse_union_eV2",
    "r2_ref",
    "r2_union",
    "mae_ref_meV",
    "rmse_ref_meV",
    "dos_mae_500_fermi_window",
]


def metric_availability_for_rows(
    rows: list[dict[str, Any]],
    metrics: list[str],
) -> dict[str, dict[str, Any]]:
    total = len(rows)
    availability: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        finite = finite_metric_count(rows, metric)
        availability[metric] = {
            "n_total": total,
            "n_finite": finite,
            "missing_count": max(0, total - finite),
            "metric_available": finite > 0,
        }
    return availability


def metric_space_from_manifest(manifest: dict[str, Any]) -> str:
    if manifest.get("kpoint_metrics_enabled") or manifest.get("kpoint_samples_compared"):
        return "kpoint_sampled"
    return "gamma_only"


def kpoint_output_warnings(
    metric_manifest: dict[str, Any],
    metric_rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not metric_manifest.get("kpoint_metrics_enabled"):
        return []
    missing = [
        key
        for key in ("kpoint_matrix", "kpoint_spectral", "kpoint_dos")
        if not metric_rows.get(key)
    ]
    if not missing:
        return []
    return [
        {
            "kind": "missing_kpoint_metric_csv",
            "severity": "warning",
            "error": (
                "metrics/manifest.json marks k-point metrics as enabled, "
                f"but these k-point CSV groups are empty or missing: {', '.join(missing)}"
            ),
            "missing_groups": missing,
        }
    ]


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def metric_manifest_warning_entries(metric_manifest: dict[str, Any], key: str) -> list[dict[str, Any]]:
    entries = metric_manifest.get(key, [])
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def metric_manifest_safety_summary(metric_manifest: dict[str, Any]) -> dict[str, Any]:
    """Compact provenance/safety fields for UI payloads without shipping huge warning lists."""
    if not metric_manifest:
        return {
            "metrics_schema_version": "unknown",
            "target_component_policy": "unknown",
            "n_matrix_components": None,
            "reference_component_count": None,
            "prediction_component_count": None,
            "overlap_source": "unknown",
            "fermi_level_source": "unknown",
            "prediction_own_overlap_used_for_spectra": None,
            "prediction_artifacts_standalone_safe": None,
            "prediction_artifact_safety_status": "unknown",
            "prediction_self_contained_hsx_safe_samples": None,
            "prediction_self_contained_hsx_unsafe_samples": None,
            "prediction_self_contained_hsx_unsafe_reasons": {},
            "graph2mat_auxiliary_component_ignored_samples": None,
            "severe_warning_count": 0,
            "warning_count": 0,
            "severe_warning_kinds": [],
            "safety_warnings_preview": [],
        }
    severe_entries = metric_manifest_warning_entries(metric_manifest, "severe_warnings")
    warning_entries = metric_manifest_warning_entries(metric_manifest, "warnings")
    severe_kinds = sorted(
        {
            str(entry.get("kind") or entry.get("code") or entry.get("severity") or "severe_warning")
            for entry in severe_entries
        }
    )
    warning_kinds = sorted(
        {
            str(entry.get("kind") or entry.get("code") or entry.get("severity") or "warning")
            for entry in warning_entries
        }
    )
    overlap_source = (
        metric_manifest.get("overlap_source")
        or next((entry.get("overlap_source") for entry in severe_entries + warning_entries if entry.get("overlap_source")), None)
        or "unknown"
    )
    fermi_level_source = (
        metric_manifest.get("fermi_level_source")
        or next((entry.get("fermi_level_source") for entry in severe_entries + warning_entries if entry.get("fermi_level_source")), None)
        or "unknown"
    )
    prediction_own_overlap_used = metric_manifest.get("prediction_own_overlap_used_for_spectra")
    if prediction_own_overlap_used is None:
        entry_value = next(
            (
                entry.get("prediction_own_overlap_used")
                for entry in severe_entries + warning_entries
                if "prediction_own_overlap_used" in entry
            ),
            None,
        )
        prediction_own_overlap_used = entry_value
    unsafe_reasons = metric_manifest.get("prediction_self_contained_hsx_unsafe_reasons")
    if not isinstance(unsafe_reasons, dict):
        unsafe_reasons = {}
    unsafe_sample_ids = {
        str(entry.get("sample"))
        for entry in severe_entries
        if entry.get("sample") not in (None, "")
        and (
            entry.get("prediction_self_contained_hsx_safe") is False
            or str(entry.get("kind") or "") in {
                "prediction_overlap_mismatch",
                "graph2mat_auxiliary_component_ignored",
            }
        )
    }
    derived_unsafe_count = len(unsafe_sample_ids)
    unsafe_count = metric_manifest.get("prediction_self_contained_hsx_unsafe_samples")
    if unsafe_count is None and derived_unsafe_count:
        unsafe_count = derived_unsafe_count
    safe_count = metric_manifest.get("prediction_self_contained_hsx_safe_samples")
    auxiliary_count = metric_manifest.get("graph2mat_auxiliary_component_ignored_samples")
    if auxiliary_count is None:
        auxiliary_count = len(
            {
                str(entry.get("sample"))
                for entry in severe_entries + warning_entries
                if entry.get("sample") not in (None, "")
                and str(entry.get("kind") or "") == "graph2mat_auxiliary_component_ignored"
            }
        ) or None
    standalone_safe = metric_manifest.get("prediction_artifacts_standalone_safe")
    if standalone_safe is None:
        try:
            unsafe_positive = unsafe_count not in (None, "") and int(unsafe_count or 0) > 0
        except (TypeError, ValueError):
            unsafe_positive = False
        try:
            safe_positive = safe_count not in (None, "") and int(safe_count or 0) > 0
        except (TypeError, ValueError):
            safe_positive = False
        if unsafe_positive:
            standalone_safe = False
        elif safe_positive:
            standalone_safe = True
    if standalone_safe is True:
        safety_status = "safe"
    elif standalone_safe is False:
        safety_status = "unsafe"
    else:
        safety_status = "unknown"
    preview: list[str] = []
    for entry in severe_entries[:4]:
        append_unique_text(
            preview,
            entry.get("kind") or entry.get("code") or entry.get("error") or entry.get("message") or entry,
        )
    return {
        "metrics_schema_version": metric_manifest.get("metrics_schema_version") or "unknown",
        "target_component_policy": metric_manifest.get("target_component_policy") or "unknown",
        "n_matrix_components": metric_manifest.get("n_matrix_components"),
        "reference_component_count": metric_manifest.get("reference_component_count"),
        "prediction_component_count": metric_manifest.get("prediction_component_count"),
        "overlap_source": overlap_source,
        "fermi_level_source": fermi_level_source,
        "prediction_own_overlap_used_for_spectra": prediction_own_overlap_used,
        "prediction_artifacts_standalone_safe": standalone_safe,
        "prediction_artifact_safety_status": safety_status,
        "prediction_self_contained_hsx_safe_samples": safe_count,
        "prediction_self_contained_hsx_unsafe_samples": unsafe_count,
        "prediction_self_contained_hsx_unsafe_reasons": unsafe_reasons,
        "graph2mat_auxiliary_component_ignored_samples": auxiliary_count,
        "severe_warning_count": len(severe_entries),
        "warning_count": len(warning_entries),
        "severe_warning_kinds": severe_kinds,
        "warning_kinds": warning_kinds,
        "safety_warnings_preview": preview,
    }


def append_unique_text(items: list[str], value: Any) -> None:
    if value in (None, "", False):
        return
    if isinstance(value, list):
        for item in value:
            append_unique_text(items, item)
        return
    if isinstance(value, dict):
        text = json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False)
    else:
        text = str(value)
    if text and text not in items:
        items.append(text)


def run_metric_gap_diagnostics(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostics = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run.get("pipeline") or "unknown")].append(run)
    for pipeline, items in sorted(grouped.items()):
        label = PIPELINES[pipeline].label if pipeline in PIPELINES else "Random Cartesian" if pipeline == "random_cartesian" else pipeline
        metric_groups = [
            ("spectral", "fermi_window_rmse_eV"),
            ("spectral", "low_energy_rmse_eV"),
            ("spectral", "frontier_window_rmse_eV"),
            ("sparse", "relative_frobenius_union"),
            ("sparse", "mse_union_eV2"),
            ("sparse", "r2_union"),
            ("dos", "dos_wasserstein_eV"),
            ("dos", "dos_mae_500_fermi_window"),
            ("kpoint_matrix", "h_mae_eV"),
            ("kpoint_matrix", "h_rmse_eV"),
            ("kpoint_matrix", "relative_frobenius"),
            ("kpoint_spectral", "low_energy_rmse_eV"),
            ("kpoint_spectral", "fermi_window_rmse_eV"),
            ("kpoint_spectral", "frontier_window_rmse_eV"),
            ("kpoint_dos", "dos_wasserstein_eV"),
            ("kpoint_dos", "dos_mae_500_fermi_window"),
        ]
        metrics = []
        for group, metric in metric_groups:
            total = 0
            finite = 0
            for run in items:
                rows = run.get("samples", {}).get(group, []) or []
                total += len(rows)
                finite += finite_metric_count(rows, metric)
            if total > 0:
                metrics.append(
                    {
                        "group": group,
                        "metric": metric,
                        "n_total": total,
                        "n_finite": finite,
                        "missing_count": max(0, total - finite),
                        "metric_available": finite > 0,
                    }
                )
        diagnostics.append(
            {
                "pipeline": pipeline,
                "label": label,
                "runs": len(items),
                "metrics": metrics,
            }
        )
    return diagnostics


def availability_size_value(value: str) -> Any:
    if value == "":
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def cross_metric_availability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("experiment_id") or ""),
            str(row.get("train_method") or ""),
            str(row.get("test_set") or ""),
            str(row.get("md_dataset_size") or row.get("dataset_size") or ""),
            str(row.get("atom_dataset_size") or row.get("dataset_size") or ""),
            str(row.get("random_dataset_size") or ""),
            str(row.get("train_dataset_size") or row.get("dataset_size") or ""),
        )
        groups.setdefault(key, []).append(row)
    availability = []
    for (
        experiment_id,
        train_method,
        test_set,
        md_dataset_size,
        atom_dataset_size,
        random_dataset_size,
        train_dataset_size,
    ), group_rows in sorted(groups.items()):
        availability.append(
            {
                "experiment_id": experiment_id,
                "train_method": train_method,
                "test_set": test_set,
                "md_dataset_size": availability_size_value(md_dataset_size),
                "atom_dataset_size": availability_size_value(atom_dataset_size),
                "random_dataset_size": availability_size_value(random_dataset_size),
                "train_dataset_size": availability_size_value(train_dataset_size),
                "metrics": metric_availability_for_rows(
                    group_rows,
                    CROSS_PLOT_AVAILABILITY_METRICS,
                ),
            }
        )
    return availability


def cross_plot_diagnostics(
    *,
    cross_experiments: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_cross_ids = {str(experiment.get("experiment_id") or "") for experiment in cross_experiments}
    archived_by_run_id: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "by_method": defaultdict(int)}
    )
    for run in runs:
        run_id = str(run.get("run_id") or "")
        if not run_id:
            continue
        bucket = archived_by_run_id[run_id]
        bucket["total"] += 1
        method = str(run.get("method_id") or run.get("pipeline") or "unknown")
        bucket["by_method"][method] += 1

    diagnostics = []
    for manifest_path in sorted(RESULTS_ROOT.glob("*/experiment_manifest.yaml")):
        experiment_id = manifest_path.parent.name
        if experiment_id in existing_cross_ids:
            continue
        try:
            manifest_size = manifest_path.stat().st_size
        except OSError:
            manifest_size = 0
        if manifest_size > MAX_CROSS_DIAGNOSTIC_MANIFEST_BYTES:
            archived = archived_by_run_id.get(experiment_id, {"total": 0, "by_method": {}})
            diagnostics.append(
                {
                    "experiment_id": experiment_id,
                    "severity": "warning",
                    "reason": "experiment_manifest_too_large_for_plot_diagnostics",
                    "message": (
                        f"No se cargo {manifest_path.name} para diagnosticos cross porque ocupa "
                        f"{manifest_size} bytes; los plots de runs archivados siguen disponibles."
                    ),
                    "expected_csv": str(manifest_path.parent / "summary" / "cross_evaluation_metrics.csv"),
                    "manifest_runs": None,
                    "archived_runs": int(archived.get("total", 0) or 0),
                    "archived_runs_by_method": dict(archived.get("by_method", {})),
                    "warnings": [],
                }
            )
            continue
        try:
            manifest = load_config(manifest_path)
        except Exception as exc:
            diagnostics.append(
                {
                    "experiment_id": experiment_id,
                    "severity": "warning",
                    "reason": "experiment_manifest_unreadable",
                    "message": f"No se pudo leer {manifest_path}: {exc}",
                    "expected_csv": str(manifest_path.parent / "summary" / "cross_evaluation_metrics.csv"),
                }
            )
            continue
        run_mode = manifest.get("run_mode")
        cross_evaluation = manifest.get("cross_evaluation") if isinstance(manifest.get("cross_evaluation"), dict) else {}
        manifest_runs = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
        archived = archived_by_run_id.get(experiment_id, {"total": 0, "by_method": {}})
        warnings: list[str] = []
        append_unique_text(warnings, cross_evaluation.get("warnings"))
        append_unique_text(warnings, cross_evaluation.get("missing_cells"))
        append_unique_text(warnings, manifest.get("warnings"))
        skipped = bool(cross_evaluation.get("skipped"))
        if run_mode == DATASET_ONLY_RUN_MODE or skipped:
            reason = str(cross_evaluation.get("reason") or run_mode or "cross_evaluation_skipped")
            severity = "info"
            message = f"Cross-evaluation omitida para {experiment_id}: {reason}."
        elif archived.get("total", 0) or manifest_runs:
            reason = "cross_evaluation_metrics_missing"
            severity = "warning"
            message = (
                f"Falta summary/cross_evaluation_metrics.csv para {experiment_id}; "
                "los plots cross, learning, compute y winner no pueden generarse."
            )
            if not manifest_runs and archived.get("total", 0):
                message += (
                    " Hay runs archivados con ese run_id, pero el experiment_manifest no "
                    "contiene runs registrados; probablemente la ejecucion se interrumpio antes de la agregacion cross."
                )
        else:
            reason = "no_completed_runs_for_cross"
            severity = "info"
            message = f"No hay runs completos registrados para generar cross-evaluation en {experiment_id}."
        diagnostics.append(
            {
                "experiment_id": experiment_id,
                "severity": severity,
                "reason": reason,
                "message": message,
                "expected_csv": str(manifest_path.parent / "summary" / "cross_evaluation_metrics.csv"),
                "manifest_runs": len(manifest_runs),
                "archived_runs": int(archived.get("total", 0) or 0),
                "archived_runs_by_method": dict(archived.get("by_method", {})),
                "warnings": warnings[:12],
            }
        )
    diagnostics.sort(key=lambda item: str(item.get("experiment_id") or ""))
    return diagnostics[-8:]


def normalized_method_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        try:
            method = normalize_method_id(value)
        except ValueError:
            method = str(value or "").strip()
        if method and method not in normalized:
            normalized.append(method)
    return normalized


def normalized_test_set_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        try:
            test_set = normalize_test_set_id(value)
        except ValueError:
            test_set = str(value or "").strip()
        if test_set and test_set not in normalized:
            normalized.append(test_set)
    return normalized


def observed_cross_methods(rows: list[dict[str, Any]]) -> list[str]:
    methods: list[str] = []
    for row in rows:
        raw_method = row.get("train_method")
        if raw_method in (None, ""):
            continue
        try:
            method = normalize_method_id(raw_method)
        except ValueError:
            method = str(raw_method)
        if method not in methods:
            methods.append(method)
    return sorted(methods)


def observed_cross_test_sets(rows: list[dict[str, Any]]) -> list[str]:
    test_sets: list[str] = []
    for row in rows:
        raw_test_set = row.get("test_set")
        if raw_test_set in (None, ""):
            continue
        try:
            test_set = normalize_test_set_id(raw_test_set)
        except ValueError:
            test_set = str(raw_test_set)
        if test_set not in test_sets:
            test_sets.append(test_set)
    return sorted(test_sets)


def scientific_plot_status(recommendation: dict[str, Any]) -> str:
    status = str(recommendation.get("status") or "")
    scientific_status = str(recommendation.get("scientific_status") or "")
    if status in {"invalid_incomplete_grid", "invalid_leakage"}:
        return status
    if status.startswith("invalid_incomplete"):
        return "invalid_incomplete_grid"
    if status.startswith("invalid_leakage") or ("leakage" in status and "invalid" in status):
        return "invalid_leakage"
    if status in {"insufficient_seeds", "metric_policy_exploratory_only", "leakage_exploratory_only"}:
        return "exploratory_only"
    if status in {
        "insufficient_primary_metric",
        "unstable_seed_winner",
        "scientifically_inconclusive_leakage",
        "fairness_provenance_mismatch",
        "no_general_robust_winner",
    }:
        return "scientifically_inconclusive"
    if scientific_status in {"exploratory", "exploratory_only"}:
        return "exploratory_only"
    if scientific_status in {"scientifically_inconclusive", "inconclusive"}:
        return "scientifically_inconclusive"
    if scientific_status == "not_scientifically_valid":
        return "scientifically_inconclusive"
    return scientific_status or status or "unknown"


def warning_entry(
    warnings: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    severity: str = "warning",
    status: str = "",
    details: Any = None,
) -> None:
    if not message:
        return
    entry = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    if status:
        entry["scientific_status"] = status
    if details not in (None, "", [], {}):
        entry["details"] = details
    if entry not in warnings:
        warnings.append(entry)


def warning_text_blob(recommendation: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in (
        "reason",
        "severe_warnings",
        "fairness_provenance_warnings",
        "method_provenance_warnings",
        "warnings",
    ):
        append_unique_text(pieces, recommendation.get(key))
    return " | ".join(pieces).lower()


def cross_experiment_plot_warnings(
    *,
    rows: list[dict[str, Any]],
    recommendation: dict[str, Any],
    manifest: dict[str, Any],
    outputs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    plot_status = scientific_plot_status(recommendation)
    status = str(recommendation.get("status") or "")
    text_blob = warning_text_blob(recommendation)
    observed_methods = observed_cross_methods(rows)
    observed_test_sets = observed_cross_test_sets(rows)
    selected_methods = normalized_method_list(
        manifest.get("selected_methods")
        or recommendation.get("selected_methods")
        or ([*observed_methods] if observed_methods else [])
    )
    selected_test_sets = normalized_test_set_list(
        manifest.get("test_sets")
        or recommendation.get("test_sets_seen")
        or ([*observed_test_sets] if observed_test_sets else [])
    )
    missing_methods = sorted(set(selected_methods) - set(observed_methods))
    missing_test_sets = sorted(set(selected_test_sets) - set(observed_test_sets))
    missing_cells = (
        recommendation.get("missing_cells")
        or recommendation.get("missing_required_cells")
        or []
    )
    missing_primary = recommendation.get("missing_primary_metric_cells") or []

    if status == "invalid_incomplete_grid" or missing_cells:
        warning_entry(
            warnings,
            code="incomplete_grid",
            severity="error",
            status="invalid_incomplete_grid",
            message="Cross-evaluation grid is incomplete; plots are diagnostic only.",
            details=missing_cells,
        )
    if missing_primary or status == "insufficient_primary_metric":
        warning_entry(
            warnings,
            code="missing_primary_metric",
            severity="error",
            status="scientifically_inconclusive",
            message="Primary metric is missing in required cells; no winner should be inferred from plots.",
            details=missing_primary,
        )
    if missing_methods:
        warning_entry(
            warnings,
            code="missing_method",
            severity="warning",
            status=plot_status,
            message="Selected method(s) are missing from cross-evaluation rows.",
            details=missing_methods,
        )
    if missing_test_sets:
        warning_entry(
            warnings,
            code="missing_test_set",
            severity="warning",
            status=plot_status,
            message="Selected frozen test set(s) are missing from cross-evaluation rows.",
            details=missing_test_sets,
        )
    if recommendation.get("single_seed_warning") or (
        recommendation.get("valid_seed_count") not in (None, "")
        and isinstance(recommendation.get("valid_seed_count"), (int, float))
        and int(recommendation.get("valid_seed_count") or 0) < 3
    ):
        warning_entry(
            warnings,
            code="single_seed_or_insufficient_seeds",
            severity="warning",
            status="exploratory_only",
            message="Seed count is below the robust threshold; plots are exploratory.",
            details={"n_seeds": recommendation.get("n_seeds"), "valid_seed_count": recommendation.get("valid_seed_count")},
        )
    seed_stability = recommendation.get("seed_stability")
    unstable_seed_status = (
        isinstance(seed_stability, dict)
        and (
            seed_stability.get("status") == "unstable"
            or bool(seed_stability.get("unstable_groups"))
        )
    )
    if status == "unstable_seed_winner" or "unstable" in text_blob or unstable_seed_status:
        warning_entry(
            warnings,
            code="unstable_winner",
            severity="error",
            status="scientifically_inconclusive",
            message="Winner changes across seeds; plot winners are not robust.",
        )
    leakage_diagnostics = recommendation.get("leakage_diagnostics")
    leakage_problem = (
        status in {"invalid_leakage", "leakage_exploratory_only", "scientifically_inconclusive_leakage"}
        or "leakage" in text_blob
        or "duplicate_geometry" in text_blob
        or "random_cartesian_family" in text_blob
    )
    if isinstance(leakage_diagnostics, dict):
        leakage_problem = leakage_problem or any(
            bool(leakage_diagnostics.get(key))
            for key in ("invalid", "inconclusive", "exploratory_only")
        )
    if leakage_problem:
        warning_entry(
            warnings,
            code="severe_leakage",
            severity="error",
            status=("invalid_leakage" if status == "invalid_leakage" else "scientifically_inconclusive"),
            message="Leakage diagnostics affect scientific validity; plots must not be read as robust.",
            details=leakage_diagnostics,
        )
    if status == "fairness_provenance_mismatch" or "mismatch" in text_blob or "provenance" in text_blob:
        warning_entry(
            warnings,
            code="fairness_provenance_mismatch",
            severity="error",
            status="scientifically_inconclusive",
            message="Fairness/provenance warnings are present; plot comparisons are not robust.",
            details=recommendation.get("fairness_provenance_warnings") or recommendation.get("severe_warnings"),
        )
    if recommendation.get("severe_warnings"):
        warning_entry(
            warnings,
            code="severe_scientific_warnings",
            severity="error",
            status="scientifically_inconclusive",
            message="Severe scientific warnings are present; the selected plot set is not a robust benchmark.",
            details=recommendation.get("severe_warnings"),
        )
    if "target_component_policy" in text_blob or "target semantics" in text_blob or "target_component" in text_blob:
        warning_entry(
            warnings,
            code="target_semantics_warning",
            severity="error",
            status="scientifically_inconclusive",
            message="Target-component semantics warnings are present; do not compare as a robust H-only benchmark.",
            details=recommendation.get("severe_warnings"),
        )
    if "prediction_overlap" in text_blob or "prediction_self_contained" in text_blob or "overlap_source" in text_blob:
        warning_entry(
            warnings,
            code="prediction_hsx_overlap_safety_warning",
            severity="error",
            status="scientifically_inconclusive",
            message="Prediction HSX/overlap safety warnings are present; spectra must use validated reference overlap.",
            details=recommendation.get("severe_warnings"),
        )
    if "missing_fermi" in text_blob or "fermi level" in text_blob:
        warning_entry(
            warnings,
            code="missing_fermi_level_warning",
            severity="warning",
            status=plot_status,
            message="Fermi-level warnings are present; near-Fermi and DOS-window metrics may be unavailable.",
            details=recommendation.get("severe_warnings"),
        )
    if "checkpoint" in text_blob:
        warning_entry(
            warnings,
            code="checkpoint_warning",
            severity="error",
            status="scientifically_inconclusive",
            message="Checkpoint compatibility/fallback warnings are present; comparisons are not robust.",
            details=recommendation.get("severe_warnings"),
        )
    method_provenance = manifest.get("method_provenance")
    if "random_cartesian" in selected_methods and (
        not isinstance(method_provenance, dict)
        or not isinstance(method_provenance.get("random_cartesian"), dict)
        or "missing random_cartesian provenance" in text_blob
    ):
        warning_entry(
            warnings,
            code="missing_random_cartesian_provenance",
            severity="warning",
            status="scientifically_inconclusive",
            message="Random Cartesian was selected but explicit method provenance is missing or incomplete.",
        )

    timing_fields = ("total_time_seconds", "siesta_time_seconds", "training_time_seconds", "prediction_time_seconds", "evaluation_time_seconds")
    has_timing = any(
        finite_number(row.get(field)) is not None
        for row in rows
        for field in timing_fields
    )
    compute_summary: dict[str, Any] = {}
    compute_summary_path = outputs.get("compute_budget_thresholds_vs_md")
    if compute_summary_path:
        path = Path(str(compute_summary_path))
        if path.exists():
            try:
                compute_summary = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                compute_summary = {}
    compute_unavailable = (
        not has_timing
        or compute_summary.get("compute_threshold_unavailable")
        or any(
            threshold.get("reason") == "compute_threshold_unavailable: missing reliable timing fields"
            for threshold in compute_summary.get("thresholds", [])
            if isinstance(threshold, dict)
        )
    )
    if compute_unavailable:
        warning_entry(
            warnings,
            code="missing_compute_timings",
            severity="info",
            status=plot_status,
            message="Reliable compute timings are missing; compute-budget plots or thresholds are unavailable.",
        )

    status_warning_needed = plot_status in {
        "exploratory_only",
        "scientifically_inconclusive",
        "invalid_incomplete_grid",
        "invalid_leakage",
    }
    if status_warning_needed:
        warning_entry(
            warnings,
            code="plot_scientific_status",
            severity=("error" if plot_status.startswith("invalid") else "warning"),
            status=plot_status,
            message=f"Final recommendation status is {plot_status}; plots are diagnostic, not a robust recommendation.",
        )

    methods_payload = {
        "selected": selected_methods,
        "observed": observed_methods,
        "missing": missing_methods,
    }
    test_sets_payload = {
        "selected": selected_test_sets,
        "observed": observed_test_sets,
        "missing": missing_test_sets,
    }
    return warnings, {
        "scientific_status": plot_status,
        "methods": methods_payload,
        "test_sets": test_sets_payload,
    }


def metric_diagnostics(
    spectral_rows: list[dict[str, Any]],
    relationship_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    issues = errors + list(warnings or [])
    missing_fermi_errors = [
        error
        for error in issues
        if error.get("kind") == "missing_fermi_level"
    ]
    unavailable_fermi_rows = [
        row
        for row in spectral_rows
        if row.get("fermi_level_source") == "unavailable"
    ]
    return {
        "spectral_samples": len(spectral_rows),
        "fermi_window_samples": finite_metric_count(spectral_rows, "fermi_window_rmse_eV"),
        "matrix_spectrum_samples": len(relationship_rows),
        "matrix_spectrum_fermi_samples": finite_metric_count(
            relationship_rows,
            "fermi_window_rmse_eV",
        ),
        "missing_fermi_level_samples": len(missing_fermi_errors),
        "unavailable_fermi_source_samples": len(unavailable_fermi_rows),
        "metric_availability": {
            "spectral": metric_availability_for_rows(
                spectral_rows,
                SPECTRAL_PLOT_AVAILABILITY_METRICS,
            ),
            "matrix_spectrum": metric_availability_for_rows(
                relationship_rows,
                SPECTRAL_PLOT_AVAILABILITY_METRICS,
            ),
        },
        "errors": errors,
        "warnings": warnings or [],
    }


def archived_manifest_result_dir(manifest: dict[str, Any], manifest_path: Path) -> Path:
    raw_result_dir = manifest.get("result_dir")
    if raw_result_dir not in (None, ""):
        result_dir = Path(str(raw_result_dir))
        candidates = [result_dir] if result_dir.is_absolute() else [
            REPO_ROOT / result_dir,
            manifest_path.parent / result_dir,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    if manifest_path.parent.name == "metrics":
        return manifest_path.parent.parent
    return manifest_path.parent


def csv_data_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return max(0, sum(1 for _line in handle) - 1)
    except OSError:
        return None


def archived_run_diagnostic_outputs(result_dir: Path) -> dict[str, dict[str, Any]]:
    metrics_dir = result_dir / "metrics"
    eigen_dir = result_dir / "eigenvalues"
    outputs: dict[str, dict[str, Any]] = {}
    for key, path in {
        "orbital_pair_metrics": metrics_dir / "orbital_pair_metrics.csv",
        "orbital_pair_summary": metrics_dir / "orbital_pair_summary.csv",
        "kpoint_matrix_metrics": metrics_dir / "kpoint_matrix_metrics.csv",
        "kpoint_spectral_metrics": metrics_dir / "kpoint_spectral_metrics.csv",
        "kpoint_dos_metrics": metrics_dir / "kpoint_dos_metrics.csv",
        "kpoints": eigen_dir / "kpoints.csv",
    }.items():
        outputs[key] = {
            "path": str(path),
            "exists": path.exists(),
            "rows": csv_data_row_count(path),
            "diagnostic_only": key.startswith("orbital_pair"),
        }
    return outputs


G2M_DEEPH_DERIVATIVE_ARTIFACTS = {
    "graph2mat_manifest": {
        "label": "Graph2Mat derivative manifest",
        "relative_path": Path("common_metrics/graph2mat_eval/derivative_metrics/manifest.json"),
        "mime_type": "application/json; charset=utf-8",
    },
    "graph2mat_matrix_metrics": {
        "label": "Graph2Mat derivative matrix metrics CSV",
        "relative_path": Path("common_metrics/graph2mat_eval/derivative_metrics/derivative_matrix_metrics.csv"),
        "mime_type": "text/csv; charset=utf-8",
    },
    "graph2mat_hermiticity": {
        "label": "Graph2Mat derivative hermiticity CSV",
        "relative_path": Path("common_metrics/graph2mat_eval/derivative_metrics/derivative_hermiticity.csv"),
        "mime_type": "text/csv; charset=utf-8",
    },
    "graph2mat_support_sweep": {
        "label": "Graph2Mat derivative support sweep CSV",
        "relative_path": Path("common_metrics/graph2mat_eval/derivative_metrics/derivative_support_sweep.csv"),
        "mime_type": "text/csv; charset=utf-8",
    },
    "graph2mat_summary": {
        "label": "Graph2Mat derivative summary JSON",
        "relative_path": Path("common_metrics/graph2mat_eval/derivative_metrics/derivative_summary.json"),
        "mime_type": "application/json; charset=utf-8",
    },
    "graph2mat_geometry_validation": {
        "label": "Graph2Mat derivative geometry validation JSON",
        "relative_path": Path("common_metrics/graph2mat_eval/derivative_metrics/derivative_geometry_validation.json"),
        "mime_type": "application/json; charset=utf-8",
    },
    "graph2mat_delta_stability": {
        "label": "Graph2Mat derivative delta stability JSON",
        "relative_path": Path("common_metrics/graph2mat_eval/derivative_metrics/derivative_delta_stability.json"),
        "mime_type": "application/json; charset=utf-8",
    },
    "graph2mat_stencil_status": {
        "label": "Graph2Mat derivative stencil status CSV",
        "relative_path": Path("common_metrics/graph2mat_eval/derivative_metrics/stencil_status.csv"),
        "mime_type": "text/csv; charset=utf-8",
    },
    "deeph_manifest": {
        "label": "DeepH derivative manifest",
        "relative_path": Path("common_metrics/deeph_eval/derivative_metrics/manifest.json"),
        "mime_type": "application/json; charset=utf-8",
    },
    "deeph_matrix_metrics": {
        "label": "DeepH derivative matrix metrics CSV",
        "relative_path": Path("common_metrics/deeph_eval/derivative_metrics/derivative_matrix_metrics.csv"),
        "mime_type": "text/csv; charset=utf-8",
    },
    "deeph_hermiticity": {
        "label": "DeepH derivative hermiticity CSV",
        "relative_path": Path("common_metrics/deeph_eval/derivative_metrics/derivative_hermiticity.csv"),
        "mime_type": "text/csv; charset=utf-8",
    },
    "deeph_support_sweep": {
        "label": "DeepH derivative support sweep CSV",
        "relative_path": Path("common_metrics/deeph_eval/derivative_metrics/derivative_support_sweep.csv"),
        "mime_type": "text/csv; charset=utf-8",
    },
    "deeph_summary": {
        "label": "DeepH derivative summary JSON",
        "relative_path": Path("common_metrics/deeph_eval/derivative_metrics/derivative_summary.json"),
        "mime_type": "application/json; charset=utf-8",
    },
    "deeph_geometry_validation": {
        "label": "DeepH derivative geometry validation JSON",
        "relative_path": Path("common_metrics/deeph_eval/derivative_metrics/derivative_geometry_validation.json"),
        "mime_type": "application/json; charset=utf-8",
    },
    "deeph_delta_stability": {
        "label": "DeepH derivative delta stability JSON",
        "relative_path": Path("common_metrics/deeph_eval/derivative_metrics/derivative_delta_stability.json"),
        "mime_type": "application/json; charset=utf-8",
    },
    "deeph_stencil_status": {
        "label": "DeepH derivative stencil status CSV",
        "relative_path": Path("common_metrics/deeph_eval/derivative_metrics/stencil_status.csv"),
        "mime_type": "text/csv; charset=utf-8",
    },
    "plot_payload": {
        "label": "Derivative plot payload JSON",
        "relative_path": Path("common_metrics/summary/derivative_plots/derivative_plot_payload.json"),
        "mime_type": "application/json; charset=utf-8",
    },
    "plot_manifest": {
        "label": "Derivative plot manifest JSON",
        "relative_path": Path("common_metrics/summary/derivative_plots/derivative_plot_manifest.json"),
        "mime_type": "application/json; charset=utf-8",
    },
    "gate_report": {
        "label": "Derivative gate report JSON",
        "relative_path": Path("common_metrics/summary/derivative_gate_report.json"),
        "mime_type": "application/json; charset=utf-8",
    },
    "model_comparison_summary": {
        "label": "Derivative model comparison summary JSON",
        "relative_path": Path("common_metrics/summary/derivative_model_comparison/derivative_model_comparison_summary.json"),
        "mime_type": "application/json; charset=utf-8",
    },
    "model_paired_comparison": {
        "label": "Derivative model paired comparison CSV",
        "relative_path": Path("common_metrics/summary/derivative_model_comparison/derivative_model_paired_comparison.csv"),
        "mime_type": "text/csv; charset=utf-8",
    },
    "artifact_validation": {
        "label": "Derivative artifact validation JSON",
        "relative_path": Path("derivative_artifact_validation.json"),
        "mime_type": "application/json; charset=utf-8",
    },
}


def _g2m_deeph_standalone_derivative_summary_path(run_root: Path, filename: str) -> Path | None:
    candidates = [
        run_root / "derivative_metrics" / "summary" / filename,
        run_root / "derivative_metrics" / "summary" / "derivative_plots" / filename,
        run_root / "derivative_metrics" / "graph2mat" / "summary" / filename,
        run_root / "derivative_metrics" / "deeph" / "summary" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _is_standalone_derivative_run_root(run_root: Path) -> bool:
    derivative_root = run_root / "derivative_metrics"
    if not derivative_root.exists():
        return False
    if any((derivative_root / method / "manifest.json").exists() for method in ("graph2mat", "deeph")):
        return True
    return derivative_root.joinpath("manifest.json").exists() and run_root.joinpath("derivative_stencil_manifest.json").exists()


def _g2m_deeph_standalone_derivative_run_roots() -> list[Path]:
    roots: list[Path] = []
    candidates: list[Path] = []
    smoke_root = RESULTS_ROOT / "derivative_smoke"
    if smoke_root.exists():
        candidates.extend(candidate for candidate in sorted(smoke_root.iterdir()) if candidate.is_dir())
    for pattern in ("derivative_postprocess", "*/derivative_postprocess", "*/*/derivative_postprocess", "*/*/*/derivative_postprocess"):
        candidates.extend(path for path in sorted(RESULTS_ROOT.glob(pattern)) if path.is_dir())
    # ui_real_metrics_derivatives campaign: per-case dirs live under
    # <output_root>/{mixing,cross_testing}/<case_id>/ — validated by contents below.
    for pattern in ("ui_real_metrics_derivatives/*/*", "*/ui_real_metrics_derivatives/*/*"):
        candidates.extend(path for path in sorted(RESULTS_ROOT.glob(pattern)) if path.is_dir())
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if _is_standalone_derivative_run_root(candidate):
            roots.append(candidate)
    return roots


def _g2m_deeph_nested_derivative_run_root(run_root: Path) -> Path | None:
    candidates: list[Path] = []
    sweep_manifest = load_json_object(run_root / "sweep" / "training_sweep_manifest.json")
    workflows = sweep_manifest.get("derivative_workflows")
    if isinstance(workflows, list):
        for workflow in workflows:
            if not isinstance(workflow, dict):
                continue
            manifest_path = str(workflow.get("derivative_workflow_manifest_path") or "").strip()
            if manifest_path:
                candidate = Path(manifest_path).expanduser()
                if not candidate.is_absolute():
                    candidate = run_root / candidate
                candidates.append(candidate.parent)
            workflow_run_id = str(workflow.get("run_id") or "").strip()
            if workflow_run_id:
                candidates.append(run_root / "sweep" / "derivative_workflows" / workflow_run_id)
    derivative_workflow_root = run_root / "sweep" / "derivative_workflows"
    if derivative_workflow_root.exists():
        candidates.extend(candidate for candidate in sorted(derivative_workflow_root.iterdir()) if candidate.is_dir())
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if _is_standalone_derivative_run_root(candidate):
            return candidate
    return None


def _g2m_deeph_effective_derivative_run_root(run_root: Path) -> Path:
    return _g2m_deeph_nested_derivative_run_root(run_root) or run_root


def _g2m_deeph_standalone_derivative_run_entry(run_root: Path) -> dict[str, Any]:
    models = [method for method in ("graph2mat", "deeph") if (run_root / "derivative_metrics" / method / "manifest.json").exists()]
    gate_report_path = _g2m_deeph_standalone_derivative_summary_path(run_root, "derivative_gate_report.json")
    gate_report = load_json_object(gate_report_path) if gate_report_path is not None else {}
    manifest_status = ""
    for method in models:
        manifest = load_json_object(run_root / "derivative_metrics" / method / "manifest.json")
        manifest_status = str(manifest.get("scientific_status") or "").strip()
        if manifest_status:
            break
    try:
        modified_at = run_root.stat().st_mtime
    except OSError:
        modified_at = None
    return {
        "id": hashlib.sha256(str(run_root.resolve()).encode("utf-8")).hexdigest()[:16],
        "run_id": run_root.name,
        "run_root": str(run_root),
        "label": f"{run_root.name} (derivative smoke)",
        "status": str(gate_report.get("scientific_status") or manifest_status or "derivative_only"),
        "dataset_ids": [],
        "models": models,
        "completed_runs": len(models),
        "failed_runs": 0,
        "planned_runs": len(models),
        "has_training_sweep": False,
        "has_metric_rows": False,
        "modified_at": modified_at,
        "derivative_only": True,
    }


def _g2m_deeph_sweep_run_roots() -> list[Path]:
    roots: list[Path] = []
    candidates = [
        manifest.parent.parent
        for manifest in RESULTS_ROOT.glob("*/*/sweep/training_sweep_manifest.json")
        if manifest.is_file()
    ]
    candidates.extend(
        manifest.parent.parent
        for manifest in RESULTS_ROOT.glob("*/sweep/training_sweep_manifest.json")
        if manifest.is_file()
    )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            roots.append(candidate)
    return roots


def _g2m_deeph_sweep_run_entry(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "sweep" / "training_sweep_manifest.json"
    manifest = load_json_object(manifest_path)
    runner_status_payload = load_json_object(run_root / "runner_status.json")
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
    workflows = manifest.get("derivative_workflows")
    derivative_models: set[str] = set()
    derivative_status = ""
    if isinstance(workflows, list):
        for workflow in workflows:
            if not isinstance(workflow, dict):
                continue
            derivative_status = derivative_status or str(workflow.get("derivative_workflow_status") or "")
            workflow_root: Path | None = None
            workflow_manifest = str(workflow.get("derivative_workflow_manifest_path") or "").strip()
            if workflow_manifest:
                workflow_manifest_path = Path(workflow_manifest).expanduser()
                if not workflow_manifest_path.is_absolute():
                    workflow_manifest_path = run_root / workflow_manifest_path
                workflow_root = workflow_manifest_path.parent
            if workflow_root is None:
                workflow_run_id = str(workflow.get("run_id") or "").strip()
                if workflow_run_id:
                    workflow_root = run_root / "sweep" / "derivative_workflows" / workflow_run_id
            if workflow_root is not None:
                for method in ("graph2mat", "deeph"):
                    if (workflow_root / "derivative_metrics" / method / "manifest.json").exists():
                        derivative_models.add(method)
    manifest_runs = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
    manifest_planned = manifest.get("planned_runs") if isinstance(manifest.get("planned_runs"), list) else []
    manifest_completed = manifest.get("completed_runs") if isinstance(manifest.get("completed_runs"), list) else []
    manifest_failed = manifest.get("failed_runs") if isinstance(manifest.get("failed_runs"), list) else []
    completed_runs = sum(
        1
        for row in manifest_runs
        if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "completed"
    )
    failed_runs = sum(
        1
        for row in manifest_runs
        if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "failed"
    )
    if not completed_runs:
        completed_runs = len(manifest_completed)
    if not failed_runs:
        failed_runs = len(manifest_failed)
    if not completed_runs:
        completed_runs = int(runner_training_sweep.get("completed") or 0)
    if not failed_runs:
        failed_runs = int(runner_training_sweep.get("failed") or 0)
    planned_runs = len(manifest_planned)
    if not planned_runs:
        planned_runs = int(runner_training_sweep.get("total") or 0)
    try:
        modified_at = max(manifest_path.stat().st_mtime, run_root.stat().st_mtime)
    except OSError:
        modified_at = None
    return {
        "id": hashlib.sha256(str(run_root.resolve()).encode("utf-8")).hexdigest()[:16],
        "run_id": run_root.name,
        "run_root": str(run_root),
        "label": f"{run_root.name} (training sweep)",
        "status": str(manifest.get("status") or derivative_status or "completed"),
        "dataset_ids": [],
        "models": sorted(derivative_models) or ["graph2mat", "deeph"],
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "planned_runs": planned_runs,
        "has_training_sweep": True,
        "has_metric_rows": False,
        "modified_at": modified_at,
    }


def _g2m_deeph_plot_runs_payload() -> dict[str, Any]:
    payload = G2M_DEEPH_RUNNER.plot_runs()
    runs = [dict(item) for item in payload.get("runs") or [] if isinstance(item, dict)]
    known_roots = {
        str(Path(str(item.get("run_root") or "")).expanduser().resolve())
        for item in runs
        if item.get("run_root")
    }
    for run_root in _g2m_deeph_standalone_derivative_run_roots():
        root_key = str(run_root.resolve())
        if root_key in known_roots:
            continue
        runs.append(_g2m_deeph_standalone_derivative_run_entry(run_root))
        known_roots.add(root_key)
    for run_root in _g2m_deeph_sweep_run_roots():
        root_key = str(run_root.resolve())
        if root_key in known_roots:
            continue
        runs.append(_g2m_deeph_sweep_run_entry(run_root))
        known_roots.add(root_key)
    runs.sort(key=lambda item: float(item.get("modified_at") or 0.0), reverse=True)
    return {
        "schema": str(payload.get("schema") or "graph2mat_deeph_plot_runs_v1"),
        "runs": runs,
        "default_selected_run_ids": payload.get("default_selected_run_ids") or [],
    }


def _g2m_deeph_plot_run_entries() -> list[dict[str, Any]]:
    payload = _g2m_deeph_plot_runs_payload()
    runs = payload.get("runs")
    return [dict(item) for item in runs] if isinstance(runs, list) else []


def resolve_g2m_deeph_run_root(run_id: str | None = None) -> Path | None:
    requested_run_id = str(run_id or "").strip()
    status = G2M_DEEPH_RUNNER.status()
    status_run_root = Path(str(status.get("run_root") or "")).expanduser() if status.get("run_root") else None
    if requested_run_id:
        if status_run_root and status_run_root.name == requested_run_id:
            return status_run_root
        for entry in _g2m_deeph_plot_run_entries():
            if str(entry.get("run_id") or "").strip() != requested_run_id:
                continue
            run_root = Path(str(entry.get("run_root") or "")).expanduser()
            return run_root if run_root.exists() else None
        return None
    if status_run_root and status_run_root.exists():
        return status_run_root
    runs = _g2m_deeph_plot_run_entries()
    if runs:
        run_root = Path(str(runs[0].get("run_root") or "")).expanduser()
        return run_root if run_root.exists() else None
    return None


def _g2m_deeph_derivative_root(run_root: Path, method: str) -> Path | None:
    derivative_run_root = _g2m_deeph_effective_derivative_run_root(run_root)
    candidates = [
        run_root / "common_metrics" / f"{method}_eval" / "derivative_metrics",
        derivative_run_root / "common_metrics" / f"{method}_eval" / "derivative_metrics",
        derivative_run_root / "derivative_metrics" / method,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _g2m_deeph_derivative_workflow_roots(run_root: Path) -> list[Path]:
    roots: list[Path] = []
    for parent in (
        run_root / "derivative_workflows",
        run_root / "sweep" / "derivative_workflows",
        run_root.parent / "derivative_workflows",
    ):
        if parent.exists():
            roots.extend(path for path in sorted(parent.glob("graphene_*_scale_iid*")) if path.is_dir())
    return roots


def _g2m_deeph_partial_derivative_metric_roots(run_root: Path, method: str) -> list[Path]:
    roots: list[Path] = []
    for workflow_root in _g2m_deeph_derivative_workflow_roots(run_root):
        candidate = workflow_root / "derivative_metrics" / method
        if (candidate / "manifest.json").exists() and (candidate / "derivative_matrix_metrics.csv").exists():
            roots.append(candidate)
    return roots


def _g2m_deeph_derivative_workflow_status_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workflow_root in _g2m_deeph_derivative_workflow_roots(run_root):
        status = load_json_object(workflow_root / "derivative_incremental_status.json")
        graph_root = workflow_root / "derivative_metrics" / "graph2mat"
        deeph_root = workflow_root / "derivative_metrics" / "deeph"
        if (graph_root / "manifest.json").exists() and (deeph_root / "manifest.json").exists():
            state = "completed_visible"
        else:
            state = str(status.get("status") or "pending_metrics")
        rows.append(
            {
                "dataset_id": workflow_root.name,
                "workflow_root": str(workflow_root),
                "status": state,
                "graph2mat_metrics": (graph_root / "manifest.json").exists(),
                "deeph_metrics": (deeph_root / "manifest.json").exists(),
            }
        )
    return rows


def _g2m_deeph_derivative_summary_path(run_root: Path, filename: str) -> Path | None:
    derivative_run_root = _g2m_deeph_effective_derivative_run_root(run_root)
    candidates = [
        run_root / "summary" / filename,
        run_root / "summary" / "derivative_plots" / filename,
        run_root / "common_metrics" / "summary" / filename,
        run_root / "common_metrics" / "summary" / "derivative_plots" / filename,
        derivative_run_root / "common_metrics" / "summary" / filename,
        derivative_run_root / "common_metrics" / "summary" / "derivative_plots" / filename,
        _g2m_deeph_standalone_derivative_summary_path(run_root, filename),
        _g2m_deeph_standalone_derivative_summary_path(derivative_run_root, filename),
    ]
    for path in candidates:
        if path is not None and path.exists():
            return path
    return None


def _g2m_deeph_combined_derivative_plot_payload(run_root: Path) -> dict[str, Any] | None:
    roots = [
        *(_g2m_deeph_partial_derivative_metric_roots(run_root, "graph2mat")),
        *(_g2m_deeph_partial_derivative_metric_roots(run_root, "deeph")),
    ]
    if len(roots) < 2:
        return None
    payload = build_derivative_plot_payload(derivative_roots=roots)
    plots = payload.get("plots") if isinstance(payload.get("plots"), list) else []
    if any(plot.get("id") == "dh_mae_vs_dataset_size" and plot.get("rows") for plot in plots if isinstance(plot, dict)):
        return payload
    return None


def _g2m_deeph_derivative_artifact_path(run_root: Path, kind: str) -> Path | None:
    if kind.startswith("graph2mat_"):
        method_root = _g2m_deeph_derivative_root(run_root, "graph2mat")
        if method_root is None:
            return None
        name = kind.removeprefix("graph2mat_")
        filenames = {
            "manifest": "manifest.json",
            "matrix_metrics": "derivative_matrix_metrics.csv",
            "hermiticity": "derivative_hermiticity.csv",
            "support_sweep": "derivative_support_sweep.csv",
            "summary": "derivative_summary.json",
            "geometry_validation": "derivative_geometry_validation.json",
            "delta_stability": "derivative_delta_stability.json",
            "stencil_status": "stencil_status.csv",
        }
        filename = filenames.get(name)
        return method_root / filename if filename else None
    if kind.startswith("deeph_"):
        method_root = _g2m_deeph_derivative_root(run_root, "deeph")
        if method_root is None:
            return None
        name = kind.removeprefix("deeph_")
        filenames = {
            "manifest": "manifest.json",
            "matrix_metrics": "derivative_matrix_metrics.csv",
            "hermiticity": "derivative_hermiticity.csv",
            "support_sweep": "derivative_support_sweep.csv",
            "summary": "derivative_summary.json",
            "geometry_validation": "derivative_geometry_validation.json",
            "delta_stability": "derivative_delta_stability.json",
            "stencil_status": "stencil_status.csv",
        }
        filename = filenames.get(name)
        return method_root / filename if filename else None
    if kind == "plot_payload":
        return _g2m_deeph_derivative_summary_path(run_root, "derivative_plot_payload.json")
    if kind == "plot_manifest":
        return _g2m_deeph_derivative_summary_path(run_root, "derivative_plot_manifest.json")
    if kind == "gate_report":
        return _g2m_deeph_derivative_summary_path(run_root, "derivative_gate_report.json")
    if kind == "model_comparison_summary":
        derivative_run_root = _g2m_deeph_effective_derivative_run_root(run_root)
        candidates = [
            run_root / "summary" / "derivative_model_comparison" / "derivative_model_comparison_summary.json",
            run_root / "common_metrics" / "summary" / "derivative_model_comparison" / "derivative_model_comparison_summary.json",
            run_root / "derivative_metrics" / "summary" / "derivative_model_comparison" / "derivative_model_comparison_summary.json",
            derivative_run_root
            / "common_metrics"
            / "summary"
            / "derivative_model_comparison"
            / "derivative_model_comparison_summary.json",
            derivative_run_root
            / "derivative_metrics"
            / "summary"
            / "derivative_model_comparison"
            / "derivative_model_comparison_summary.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None
    if kind == "model_paired_comparison":
        derivative_run_root = _g2m_deeph_effective_derivative_run_root(run_root)
        candidates = [
            run_root / "summary" / "derivative_model_comparison" / "derivative_model_paired_comparison.csv",
            run_root / "common_metrics" / "summary" / "derivative_model_comparison" / "derivative_model_paired_comparison.csv",
            run_root / "derivative_metrics" / "summary" / "derivative_model_comparison" / "derivative_model_paired_comparison.csv",
            derivative_run_root
            / "common_metrics"
            / "summary"
            / "derivative_model_comparison"
            / "derivative_model_paired_comparison.csv",
            derivative_run_root
            / "derivative_metrics"
            / "summary"
            / "derivative_model_comparison"
            / "derivative_model_paired_comparison.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None
    if kind == "artifact_validation":
        derivative_run_root = _g2m_deeph_effective_derivative_run_root(run_root)
        for path in (run_root / "derivative_artifact_validation.json", derivative_run_root / "derivative_artifact_validation.json"):
            if path.exists():
                return path
        return None
    return None


def _g2m_deeph_derivative_artifact_url(run_id: str, kind: str) -> str:
    return "/api/g2m-deeph/derivative-metrics/artifact?" + urlencode({"run_id": run_id, "kind": kind})


def _g2m_deeph_derivative_issue_rows(
    model_label: str,
    manifest: dict[str, Any],
    stencil_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest.get("warnings") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "model": model_label,
                "severity": item.get("severity") or "warning",
                "code": item.get("code") or item.get("kind") or "warning",
                "sample": item.get("sample") or "",
                "message": item.get("message") or "",
            }
        )
    for item in manifest.get("fatal_errors") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "model": model_label,
                "severity": "severe",
                "code": item.get("code") or item.get("kind") or "fatal_error",
                "sample": item.get("sample") or "",
                "message": item.get("message") or "",
            }
        )
    for row in stencil_rows:
        issue_codes = [code.strip() for code in str(row.get("issue_codes") or "").split(";") if code.strip()]
        issue_messages = [message.strip() for message in str(row.get("issue_messages") or "").split(";") if message.strip()]
        if not issue_codes and not issue_messages:
            continue
        rows.append(
            {
                "model": model_label,
                "severity": "severe" if str(row.get("status") or "") != "ok" else "warning",
                "code": ", ".join(issue_codes) or "stencil_issue",
                "sample": row.get("sample") or "",
                "message": "; ".join(issue_messages) or "Derivative stencil diagnostic issue.",
            }
        )
    return rows


def _g2m_deeph_derivative_status_row(model_label: str, root: Path) -> dict[str, Any]:
    manifest = load_json_object(root / "manifest.json")
    metric_rows = read_csv_rows(root / "derivative_matrix_metrics.csv")
    stencil_rows = read_csv_rows(root / "stencil_status.csv")
    methods_seen = sorted(
        {
            str(row.get("finite_difference_method") or "").strip()
            for row in metric_rows
            if str(row.get("finite_difference_method") or "").strip()
        }
    )
    delta_values = sorted(
        {
            float(row.get("delta_ang"))
            for row in metric_rows
            if finite_number(row.get("delta_ang")) is not None
        }
    )
    units_seen = sorted(
        {
            str(row.get("derivative_units") or "").strip()
            for row in metric_rows
            if str(row.get("derivative_units") or "").strip()
        }
    )
    issue_rows = _g2m_deeph_derivative_issue_rows(model_label, manifest, stencil_rows)
    return {
        "method": model_label,
        "scientific_status": str(manifest.get("scientific_status") or "diagnostic_only"),
        "finite_difference_method": ", ".join(methods_seen) if methods_seen else str(manifest.get("finite_difference_method") or ""),
        "delta_ang": ", ".join(f"{value:.6g}" for value in delta_values) if delta_values else "",
        "derivative_units": ", ".join(units_seen) if units_seen else str(manifest.get("derivative_units") or ""),
        "stencils_ok": int(manifest.get("stencils_ok") or 0),
        "stencils_failed": int(manifest.get("stencils_failed") or 0),
        "diagnostic_only": str(manifest.get("scientific_status") or "diagnostic_only") == "diagnostic_only",
        "issue_rows": issue_rows,
    }


def _g2m_deeph_derivative_comparison_rows(plot_payload: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = {str(plot.get("id") or ""): plot for plot in plot_payload.get("plots") or [] if isinstance(plot, dict)}
    mae_rows = {
        str(row.get("method") or ""): row
        for row in lookup.get("dh_mae_by_model", {}).get("rows", [])
        if isinstance(row, dict)
    }
    rmse_rows = {
        str(row.get("method") or ""): row
        for row in lookup.get("dh_rmse_by_model", {}).get("rows", [])
        if isinstance(row, dict)
    }
    fro_rows = {
        str(row.get("method") or ""): row
        for row in lookup.get("relative_frobenius_by_model", {}).get("rows", [])
        if isinstance(row, dict)
    }
    herm_rows = {
        str(row.get("method") or ""): row
        for row in lookup.get("hermiticity_defect_by_model", {}).get("rows", [])
        if isinstance(row, dict)
    }
    methods = sorted(set(mae_rows) | set(rmse_rows) | set(fro_rows) | set(herm_rows))
    return [
        {
            "method": method,
            "dh_mae_union_eV_per_Ang": (mae_rows.get(method) or {}).get("dh_mae_union_eV_per_Ang"),
            "dh_rmse_union_eV_per_Ang": (rmse_rows.get(method) or {}).get("dh_rmse_union_eV_per_Ang"),
            "dh_relative_frobenius_ref": (fro_rows.get(method) or {}).get("dh_relative_frobenius_ref"),
            "dH_pred_hermiticity_defect": (herm_rows.get(method) or {}).get("dH_pred_hermiticity_defect"),
            "dH_hermiticity_error_delta": (herm_rows.get(method) or {}).get("dH_hermiticity_error_delta"),
        }
        for method in methods
    ]


def _g2m_deeph_derivative_metric_summary_rows(
    graph2mat_root: Path | None,
    deeph_root: Path | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, root in (("Graph2Mat", graph2mat_root), ("DeepH", deeph_root)):
        if root is None:
            continue
        metric_rows = read_csv_rows(root / "derivative_matrix_metrics.csv")
        hermiticity_rows = read_csv_rows(root / "derivative_hermiticity.csv")
        rows.append(
            {
                "method": method,
                "dh_mae_union_eV_per_Ang": _g2m_deeph_derivative_mean(metric_rows, "dh_mae_union_eV_per_Ang"),
                "dh_rmse_union_eV_per_Ang": _g2m_deeph_derivative_mean(metric_rows, "dh_rmse_union_eV_per_Ang"),
                "dh_relative_frobenius_ref": _g2m_deeph_derivative_mean(metric_rows, "dh_relative_frobenius_ref"),
                "dH_pred_hermiticity_defect": _g2m_deeph_derivative_mean(hermiticity_rows, "dH_pred_hermiticity_defect"),
                "dH_hermiticity_error_delta": _g2m_deeph_derivative_mean(hermiticity_rows, "dH_hermiticity_error_delta"),
            }
        )
    return rows


def _g2m_deeph_derivative_paired_comparison_rows(run_root: Path) -> list[dict[str, Any]]:
    derivative_run_root = _g2m_deeph_effective_derivative_run_root(run_root)
    candidates = [
        run_root
        / "common_metrics"
        / "summary"
        / "derivative_model_comparison"
        / "derivative_model_paired_comparison.csv",
        run_root
        / "derivative_metrics"
        / "summary"
        / "derivative_model_comparison"
        / "derivative_model_paired_comparison.csv",
        derivative_run_root
        / "common_metrics"
        / "summary"
        / "derivative_model_comparison"
        / "derivative_model_paired_comparison.csv",
        derivative_run_root
        / "derivative_metrics"
        / "summary"
        / "derivative_model_comparison"
        / "derivative_model_paired_comparison.csv",
    ]
    for path in candidates:
        rows = read_csv_rows(path)
        if rows:
            return rows[:25]
    return []


def _g2m_deeph_derivative_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else None


def _g2m_deeph_derivative_status_from_gate(gate_report: dict[str, Any], status_rows: list[dict[str, Any]]) -> str:
    gate_status = str(gate_report.get("scientific_status") or "").strip()
    if gate_status == "blocked":
        return "blocked"
    if gate_status in {"technical_presentation", "paper_level_candidate"}:
        return "presentation_ready"
    if status_rows and all(str(row.get("scientific_status") or "") == "presentation_ready" for row in status_rows):
        return "presentation_ready"
    return "diagnostic_only"


def _g2m_deeph_derivative_gate_report(run_root: Path) -> dict[str, Any]:
    persisted_path = _g2m_deeph_derivative_summary_path(run_root, "derivative_gate_report.json")
    persisted = load_json_object(persisted_path) if persisted_path is not None else {}
    if persisted:
        payload = dict(persisted)
        gate_rows = payload.get("gate_rows") if isinstance(payload.get("gate_rows"), list) else []
        if not gate_rows:
            gate_rows = []
            for item in payload.get("blockers") or []:
                if not isinstance(item, dict):
                    continue
                gate_rows.append(
                    {
                        "gate": item.get("gate_id") or item.get("id") or item.get("gate") or item.get("code") or "derivative_gate",
                        "status": item.get("status") or "blocked",
                        "severity": item.get("severity") or "blocker",
                        "message": item.get("message") or "",
                    }
                )
            for item in payload.get("warnings") or []:
                if not isinstance(item, dict):
                    continue
                gate_rows.append(
                    {
                        "gate": item.get("gate_id") or item.get("id") or item.get("gate") or item.get("code") or "derivative_warning",
                        "status": item.get("status") or "warning",
                        "severity": item.get("severity") or "warning",
                        "message": item.get("message") or "",
                    }
                )
        payload["gate_rows"] = gate_rows
        payload.setdefault("derivative_winner_claim", "none")
        payload.setdefault("message", "Derivative gate report loaded. No winner claim is shown unless the gate explicitly permits it.")
        return payload
    common_summary = load_json_object(run_root / "common_metrics" / "summary" / "common_summary.json")
    ranking_summary = load_json_object(run_root / "summary" / "ranking" / "ranking_summary.json")
    common_recommendation = common_summary.get("recommendation") if isinstance(common_summary.get("recommendation"), dict) else {}
    ranking_recommendation = ranking_summary.get("recommendation") if isinstance(ranking_summary.get("recommendation"), dict) else {}
    gate_rows = [
        *( {"gate": gate, "status": "failed"} for gate in ranking_recommendation.get("gates_failed") or [] ),
        *( {"gate": gate, "status": "passed"} for gate in ranking_recommendation.get("gates_passed") or [] ),
    ]
    return {
        "common_metrics_status": str(common_summary.get("status") or "unknown"),
        "common_recommendation_status": str(common_recommendation.get("status") or common_summary.get("status") or "unknown"),
        "common_reason": str(common_recommendation.get("reason") or ""),
        "ranking_status": str(ranking_recommendation.get("status") or "unknown"),
        "ranking_scientific_status": str(ranking_recommendation.get("scientific_status") or "unknown"),
        "ranking_reason": str(ranking_recommendation.get("reason") or ""),
        "winner": ranking_recommendation.get("winner"),
        "primary_metric": ranking_recommendation.get("primary_metric") or common_recommendation.get("primary_metric") or "",
        "gate_rows": gate_rows,
        "derivative_winner_claim": "none",
        "diagnostic_only": True,
        "message": "Hamiltonian derivative diagnostics are diagnostic-only / no winner claim and never override the gate report.",
    }


def _g2m_deeph_hamiltonian_metrics_root(run_root: Path) -> Path:
    """Resolve the root that holds H graph2mat/deeph metrics (the `sweep/` tree).

    Some completed runs nest the actual sweep one level deeper, under a
    child directory sharing the run root's own name (OUTER/OUTER.name/sweep).
    Derivative diagnostics (derivative_workflows/) live at the outer level
    regardless, so this resolution must stay independent of that lookup.
    """
    if (run_root / "sweep").exists() or (run_root / "benchmark_manifest.yaml").exists():
        return run_root
    nested_root = run_root / run_root.name
    if (nested_root / "sweep").exists() or (nested_root / "benchmark_manifest.yaml").exists():
        return nested_root
    return run_root


def g2m_deeph_derivative_metrics_payload(run_id: str | None = None) -> dict[str, Any]:
    run_root = resolve_g2m_deeph_run_root(run_id)
    requested_run_id = str(run_id or "").strip()
    if run_root is None:
        return {
            "available": False,
            "status": "not_computed",
            "run_id": requested_run_id,
            "message": (
                "Derivative diagnostics are optional post-processing outputs. "
                "If not computed, the benchmark remains valid for H-vs-H metrics."
            ),
            "not_computed": True,
        }
    effective_run_id = run_root.name
    hamiltonian_metrics_root = _g2m_deeph_hamiltonian_metrics_root(run_root)
    graph2mat_root = _g2m_deeph_derivative_root(run_root, "graph2mat")
    deeph_root = _g2m_deeph_derivative_root(run_root, "deeph")
    partial_graph2mat_roots = _g2m_deeph_partial_derivative_metric_roots(run_root, "graph2mat")
    partial_deeph_roots = _g2m_deeph_partial_derivative_metric_roots(run_root, "deeph")
    partial_workflow_rows = _g2m_deeph_derivative_workflow_status_rows(run_root)
    gate_report = _g2m_deeph_derivative_gate_report(run_root)
    if graph2mat_root is None and partial_graph2mat_roots:
        graph2mat_root = partial_graph2mat_roots[0]
    if deeph_root is None and partial_deeph_roots:
        deeph_root = partial_deeph_roots[0]
    if graph2mat_root is None and deeph_root is None:
        return {
            "available": False,
            "status": "not_computed",
            "run_id": effective_run_id,
            "run_root": str(run_root),
            "hamiltonian_metrics_root": str(hamiltonian_metrics_root),
            "message": (
                "Derivative diagnostics are optional post-processing outputs. "
                "If not computed, the benchmark remains valid for H-vs-H metrics."
            ),
            "not_computed": True,
            "title": "Hamiltonian derivative diagnostics",
            "gate_report": gate_report,
        }
    plot_payload_path = _g2m_deeph_derivative_summary_path(run_root, "derivative_plot_payload.json")
    if plot_payload_path is not None and plot_payload_path.exists():
        plot_payload = load_json_object(plot_payload_path)
    else:
        plot_payload = {"available": False, "plots": [], "status": "not_computed"}
    if not any(
        plot.get("id") == "dh_mae_vs_dataset_size" and plot.get("rows")
        for plot in (plot_payload.get("plots") or [])
        if isinstance(plot, dict)
    ):
        plot_payload = _g2m_deeph_combined_derivative_plot_payload(run_root) or plot_payload
    status_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for method_key, label, root in (
        ("graph2mat", "Graph2Mat", graph2mat_root),
        ("deeph", "DeepH", deeph_root),
    ):
        if root is None:
            continue
        row = _g2m_deeph_derivative_status_row(label, root)
        status_rows.append({key: value for key, value in row.items() if key != "issue_rows"})
        issue_rows.extend(row.get("issue_rows") or [])
        for artifact_kind, meta in G2M_DEEPH_DERIVATIVE_ARTIFACTS.items():
            if not artifact_kind.startswith(method_key):
                continue
            path = _g2m_deeph_derivative_artifact_path(run_root, artifact_kind)
            artifact_rows.append(
                {
                    "label": meta["label"],
                    "kind": artifact_kind,
                    "exists": bool(path and path.exists()),
                    "path": str(path) if path is not None else "",
                    "url": _g2m_deeph_derivative_artifact_url(effective_run_id, artifact_kind) if path and path.exists() else "",
                }
            )
    for artifact_kind in (
        "plot_payload",
        "plot_manifest",
        "gate_report",
        "model_comparison_summary",
        "model_paired_comparison",
        "artifact_validation",
    ):
        meta = G2M_DEEPH_DERIVATIVE_ARTIFACTS[artifact_kind]
        path = _g2m_deeph_derivative_artifact_path(run_root, artifact_kind)
        artifact_rows.append(
            {
                "label": meta["label"],
                "kind": artifact_kind,
                "exists": bool(path and path.exists()),
                "path": str(path) if path is not None else "",
                "url": _g2m_deeph_derivative_artifact_url(effective_run_id, artifact_kind) if path and path.exists() else "",
            }
        )
    prominent = [
        row for row in issue_rows
        if any(token in f"{row.get('code', '')} {row.get('message', '')}".lower() for token in ("gauge", "orbital", "order", "metadata", "hash"))
    ]
    paired_rows = _g2m_deeph_derivative_paired_comparison_rows(run_root)
    comparison_rows = paired_rows or _g2m_deeph_derivative_comparison_rows(plot_payload) or _g2m_deeph_derivative_metric_summary_rows(
        graph2mat_root,
        deeph_root,
    )
    overall_status = _g2m_deeph_derivative_status_from_gate(gate_report, status_rows)
    return {
        "available": True,
        "status": overall_status,
        "run_id": effective_run_id,
        "run_root": str(run_root),
        "hamiltonian_metrics_root": str(hamiltonian_metrics_root),
        "title": "Hamiltonian derivative diagnostics",
        "reference_label": "Reference: finite differences of SIESTA Hamiltonians",
        "force_constants_label": "SIESTA force constants are not treated as dH/dR",
        "default_status_text": "Default status: diagnostic-only unless all scientific gates pass",
        "not_computed": False,
        "winner": None,
        "gate_report": gate_report,
        "status_rows": status_rows,
        "workflow_status_rows": partial_workflow_rows,
        "issue_rows": issue_rows,
        "prominent_issue_rows": prominent,
        "comparison_rows": comparison_rows,
        "paired_comparison_rows": paired_rows,
        "plot_payload": plot_payload,
        "artifact_rows": artifact_rows,
        "scientific_warnings": plot_payload.get("scientific_warnings") or [],
        "message": (
            "Derivative diagnostics are available for technical internal diagnostic inspection only. "
            "They do not change the H-vs-H benchmark winner."
        ),
    }


def g2m_deeph_derivative_metrics_multi_payload(run_ids: list[str]) -> dict[str, Any]:
    ids = [str(item).strip() for item in run_ids if str(item).strip()]
    if len(ids) <= 1:
        return g2m_deeph_derivative_metrics_payload(ids[0] if ids else None)
    payloads = [g2m_deeph_derivative_metrics_payload(run_id) for run_id in ids]
    combined_plots: dict[str, dict[str, Any]] = {}
    primary_ids: list[str] = []
    diagnostic_ids: list[str] = []
    dataset_size_ids: list[str] = []
    for payload in payloads:
        run_id = str(payload.get("run_id") or "")
        plot_payload = payload.get("plot_payload") or {}
        for key, target in (
            ("primary_plot_ids", primary_ids),
            ("diagnostic_plot_ids", diagnostic_ids),
            ("dataset_size_plot_ids", dataset_size_ids),
        ):
            for plot_id in plot_payload.get(key) or []:
                if plot_id not in target:
                    target.append(plot_id)
        for plot in plot_payload.get("plots") or []:
            plot_id = str(plot.get("id") or "")
            if not plot_id:
                continue
            merged = combined_plots.setdefault(plot_id, {**plot, "rows": []})
            for row in plot.get("rows") or []:
                method = str(row.get("method") or row.get("model_label") or row.get("model") or "")
                series = str(row.get(plot.get("series_key") or "") or method or "diagnostic")
                merged["rows"].append(
                    {
                        **row,
                        "run_id": run_id,
                        "run_label": run_id,
                        "combined_series": series or method or "diagnostic",
                    }
                )
    plot_payload = {
        "available": any((payload.get("plot_payload") or {}).get("available") for payload in payloads),
        "plots": list(combined_plots.values()),
        "primary_plot_ids": primary_ids,
        "diagnostic_plot_ids": diagnostic_ids,
        "dataset_size_plot_ids": dataset_size_ids,
        "scientific_warnings": [warning for payload in payloads for warning in (payload.get("scientific_warnings") or [])],
    }
    first = payloads[0] if payloads else {}
    return {
        **first,
        "available": any(payload.get("available") for payload in payloads),
        "run_id": ", ".join(ids),
        "run_ids": ids,
        "run_root": "",
        "status_rows": [row for payload in payloads for row in (payload.get("status_rows") or [])],
        "workflow_status_rows": [row for payload in payloads for row in (payload.get("workflow_status_rows") or [])],
        "issue_rows": [row for payload in payloads for row in (payload.get("issue_rows") or [])],
        "prominent_issue_rows": [row for payload in payloads for row in (payload.get("prominent_issue_rows") or [])],
        "comparison_rows": [row for payload in payloads for row in (payload.get("comparison_rows") or [])],
        "paired_comparison_rows": [row for payload in payloads for row in (payload.get("paired_comparison_rows") or [])],
        "artifact_rows": [row for payload in payloads for row in (payload.get("artifact_rows") or [])],
        "plot_payload": plot_payload,
        "scientific_warnings": plot_payload["scientific_warnings"],
    }


def weighted_kpoint_matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("row_type") or "") == "weighted_sample"]


def archived_run_metric_rows(result_dir: Path) -> dict[str, list[dict[str, Any]]]:
    metrics_dir = result_dir / "metrics"
    kpoint_matrix_rows = read_csv_rows(metrics_dir / "kpoint_matrix_metrics.csv")
    return {
        "sparse": read_csv_rows(metrics_dir / "sparse_metrics.csv"),
        "spectral": read_csv_rows(metrics_dir / "spectral_metrics.csv"),
        "dos": read_csv_rows(metrics_dir / "dos_metrics.csv"),
        "kpoint_matrix": weighted_kpoint_matrix_rows(kpoint_matrix_rows),
        "kpoint_matrix_per_k": [
            row for row in kpoint_matrix_rows if str(row.get("row_type") or "") == "per_k"
        ],
        "kpoint_spectral": read_csv_rows(metrics_dir / "kpoint_spectral_metrics.csv"),
        "kpoint_dos": read_csv_rows(metrics_dir / "kpoint_dos_metrics.csv"),
        "sparse_sweep": read_csv_rows(metrics_dir / "sparse_threshold_sweep.csv"),
        "dos_sweep": read_csv_rows(metrics_dir / "dos_sigma_sweep.csv"),
        "matrix_spectrum": read_csv_rows(metrics_dir / "matrix_spectrum_relationship.csv"),
        "orbital_pair_summary": read_csv_rows(metrics_dir / "orbital_pair_summary.csv"),
    }


PLOT_METRIC_CSV_FILES = {
    "sparse": "sparse_metrics.csv",
    "spectral": "spectral_metrics.csv",
    "dos": "dos_metrics.csv",
    "kpoint_matrix": "kpoint_matrix_metrics.csv",
    "kpoint_spectral": "kpoint_spectral_metrics.csv",
    "kpoint_dos": "kpoint_dos_metrics.csv",
    "sparse_sweep": "sparse_threshold_sweep.csv",
    "dos_sweep": "dos_sigma_sweep.csv",
    "matrix_spectrum": "matrix_spectrum_relationship.csv",
    "orbital_pair_summary": "orbital_pair_summary.csv",
}


def archived_run_plot_metric_row_counts(result_dir: Path) -> dict[str, int]:
    metrics_dir = result_dir / "metrics"
    counts: dict[str, int] = {}
    for key, filename in PLOT_METRIC_CSV_FILES.items():
        rows = csv_data_row_count(metrics_dir / filename)
        counts[key] = int(rows or 0)
    return counts


def archived_run_has_plot_metric_outputs(result_dir: Path) -> bool:
    return any(count > 0 for count in archived_run_plot_metric_row_counts(result_dir).values())


def archived_run_has_plot_metrics(result_dir: Path) -> bool:
    rows_by_metric = archived_run_metric_rows(result_dir)
    return any(rows for rows in rows_by_metric.values())


def plot_data_summary() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    groups = {
        "md": RESULTS_ROOT / "results_md",
        "atom_displacement": RESULTS_ROOT / "results_atomdisp",
        "random_cartesian": RESULTS_ROOT / "results_random_cartesian",
    }
    for key, root in groups.items():
        if not root.exists():
            continue
        for manifest_path in archived_result_manifest_paths(root):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            result_dir = archived_manifest_result_dir(manifest, manifest_path)
            metric_rows = archived_run_metric_rows(result_dir)
            if not any(metric_rows.values()):
                continue
            sparse_rows = metric_rows["sparse"]
            spectral_rows = metric_rows["spectral"]
            dos_rows = metric_rows["dos"]
            kpoint_matrix_rows = metric_rows["kpoint_matrix"]
            kpoint_matrix_per_k_rows = metric_rows["kpoint_matrix_per_k"]
            kpoint_spectral_rows = metric_rows["kpoint_spectral"]
            kpoint_dos_rows = metric_rows["kpoint_dos"]
            sparse_sweep_rows = metric_rows["sparse_sweep"]
            dos_sweep_rows = metric_rows["dos_sweep"]
            relationship_rows = metric_rows["matrix_spectrum"]
            orbital_pair_summary_rows = metric_rows["orbital_pair_summary"]
            diagnostic_outputs = archived_run_diagnostic_outputs(result_dir)
            errors = manifest.get("errors", [])
            if not isinstance(errors, list):
                errors = []
            metric_manifest = {}
            metric_manifest_path = result_dir / "metrics" / "manifest.json"
            metric_manifest = load_json_object(metric_manifest_path)
            metric_safety = metric_manifest_safety_summary(metric_manifest)
            metric_errors = metric_manifest.get("fatal_errors", metric_manifest.get("errors", []))
            if isinstance(metric_errors, list):
                errors = errors + metric_errors
            warnings = metric_manifest.get("warnings", [])
            if not isinstance(warnings, list):
                warnings = []
            warnings = warnings + kpoint_output_warnings(metric_manifest, metric_rows)
            metric_space = metric_space_from_manifest(metric_manifest)
            inferred_dataset_size = int(
                manifest.get(
                    "requested_dataset_size",
                    manifest.get(
                        "dataset_size",
                        metric_manifest.get(
                            "samples_seen",
                            metric_manifest.get("samples_compared", 0),
                        ),
                    ),
                )
                or 0
            )
            inferred_run_id = str(manifest.get("run_id") or result_dir.name.removeprefix("run_"))
            inferred_dataset_label = manifest.get("dataset_label") or result_dir.parent.name
            material_payload = plot_material_metadata(
                manifest.get("material_provenance") if isinstance(manifest.get("material_provenance"), dict) else {},
                manifest.get("material_validation") if isinstance(manifest.get("material_validation"), dict) else {},
                manifest,
                metric_manifest.get("material_provenance") if isinstance(metric_manifest.get("material_provenance"), dict) else {},
                metric_manifest,
            )
            samples_by_group = {
                "sparse": sparse_rows,
                "spectral": spectral_rows,
                "dos": dos_rows,
                "kpoint_matrix": kpoint_matrix_rows,
                "kpoint_matrix_per_k": kpoint_matrix_per_k_rows,
                "kpoint_spectral": kpoint_spectral_rows,
                "kpoint_dos": kpoint_dos_rows,
                "sparse_sweep": sparse_sweep_rows,
                "dos_sweep": dos_sweep_rows,
                "matrix_spectrum": relationship_rows,
                "orbital_pair_summary": orbital_pair_summary_rows,
            }
            runs.append(
                {
                    "pipeline": key,
                    "method_id": manifest.get(
                        "method_id",
                        "siesta_fc_cartesian" if key == "atom_displacement" else key,
                    ),
                    "label": PIPELINES[key].label if key in PIPELINES else "Random Cartesian",
                    "dataset_size": inferred_dataset_size,
                    "effective_dataset_size": int(
                        manifest.get(
                            "effective_dataset_size",
                            inferred_dataset_size,
                        )
                    ),
                    "requested_dataset_size": inferred_dataset_size,
                    "run_id": inferred_run_id,
                    "result_dir": str(result_dir),
                    "dataset_label": inferred_dataset_label,
                    "training_tag": manifest.get("training_tag"),
                    "training_index": manifest.get("training_index"),
                    "training_base_dataset_label": manifest.get("training_base_dataset_label"),
                    "training_base_dataset_size": manifest.get("training_base_dataset_size"),
                    "training_base_dataset_dir": manifest.get("training_base_dataset_dir"),
                    "training_base_result_dir": manifest.get("training_base_result_dir"),
                    "training_base_run_id": manifest.get("training_base_run_id"),
                    "training_plan_index": manifest.get("training_plan_index"),
                    "training_plan_label": manifest.get("training_plan_label"),
                    "training_plan_display_label": manifest.get("training_plan_display_label"),
                    "training_plan_settings": manifest.get("training_plan_settings"),
                    "training_plan_source_dataset_label": manifest.get("training_plan_source_dataset_label"),
                    "sweep_index": manifest.get("sweep_index"),
                    "sweep_label": manifest.get("sweep_label"),
                    "sweep_parameters": manifest.get("sweep_parameters"),
                    "hidden_irreps_dimension": manifest.get("hidden_irreps_dimension"),
                    "recipe_id": manifest.get("recipe_id"),
                    "recipe_label": manifest.get("recipe_label"),
                    "block_id": manifest.get("block_id"),
                    "block_label": manifest.get("block_label"),
                    "recipe_set_hash": manifest.get("recipe_set_hash"),
                    "dataset_recipe": manifest.get("dataset_recipe", {}),
                    **material_payload,
                    **metric_safety,
                    "metric_manifest_path": str(metric_manifest_path),
                    "metric_space": metric_space,
                    "kpoint_metrics_enabled": bool(metric_manifest.get("kpoint_metrics_enabled")),
                    "kpoint_sampled_supported": bool(metric_manifest.get("kpoint_sampled_supported")),
                    "kpoint_mesh": metric_manifest.get("kpoint_mesh"),
                    "kpoint_count": metric_manifest.get("kpoint_count"),
                    "kpoint_source": metric_manifest.get("kpoint_source"),
                    "uses_reference_overlap_k": bool(metric_manifest.get("uses_reference_overlap_k")),
                    "complex_hamiltonians_supported_for_kpoint_metrics": bool(
                        metric_manifest.get("complex_hamiltonians_supported_for_kpoint_metrics")
                    ),
                    "pipeline_elapsed_seconds": manifest.get("pipeline_elapsed_seconds"),
                    "means": {
                        "run": {
                            "pipeline_elapsed_seconds": manifest.get("pipeline_elapsed_seconds"),
                        }
                        if isinstance(manifest.get("pipeline_elapsed_seconds"), (int, float))
                        else {},
                        "sparse": numeric_means(sparse_rows),
                        "spectral": numeric_means(spectral_rows),
                        "dos": numeric_means(dos_rows),
                        "kpoint_matrix": numeric_means(kpoint_matrix_rows),
                        "kpoint_matrix_per_k": numeric_means(kpoint_matrix_per_k_rows),
                        "kpoint_spectral": numeric_means(kpoint_spectral_rows),
                        "kpoint_dos": numeric_means(kpoint_dos_rows),
                        "sparse_sweep": numeric_means(sparse_sweep_rows),
                        "dos_sweep": numeric_means(dos_sweep_rows),
                        "matrix_spectrum": numeric_means(relationship_rows),
                        "orbital_pair_summary": numeric_means(orbital_pair_summary_rows),
                    },
                    "samples": plot_sample_payload(samples_by_group),
                    "sample_row_counts": plot_sample_row_counts(samples_by_group),
                    "diagnostics": metric_diagnostics(
                        spectral_rows,
                        relationship_rows,
                        errors,
                        warnings,
                    ),
                    "metric_availability": {
                        "spectral": metric_availability_for_rows(
                            spectral_rows,
                            SPECTRAL_PLOT_AVAILABILITY_METRICS,
                        ),
                        "kpoint_matrix": metric_availability_for_rows(
                            kpoint_matrix_rows,
                            ["h_mae_eV", "h_rmse_eV", "relative_frobenius"],
                        ),
                        "kpoint_spectral": metric_availability_for_rows(
                            kpoint_spectral_rows,
                            SPECTRAL_PLOT_AVAILABILITY_METRICS + ["global_rmse_eV", "gap_abs_error_eV"],
                        ),
                        "kpoint_dos": metric_availability_for_rows(
                            kpoint_dos_rows,
                            ["dos_wasserstein_eV", "dos_mae_500_fermi_window"],
                        ),
                    },
                    "summary": manifest.get("summary", {}),
                    "diagnostic_outputs": diagnostic_outputs,
                }
            )
    runs.sort(key=lambda item: (item["pipeline"], item["dataset_size"], item["run_id"]))
    runs.extend(deeph_comparison_plot_runs())
    runs.sort(key=lambda item: (item["pipeline"], item["dataset_size"], item["run_id"]))
    cross_experiments: list[dict[str, Any]] = []
    for metrics_path in sorted(RESULTS_ROOT.glob("*/summary/cross_evaluation_metrics.csv")):
        experiment_dir = metrics_path.parents[1]
        rows = read_csv_rows(metrics_path)
        recommendation_path = experiment_dir / "summary" / "recommendation.json"
        manifest_path = experiment_dir / "experiment_manifest.yaml"
        recommendation: dict[str, Any] = {}
        if recommendation_path.exists():
            try:
                recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
            except Exception as exc:
                recommendation = {"error": str(exc)}
        recommendation = sanitize_recommendation_warnings(recommendation)
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = load_config(manifest_path)
            except Exception as exc:
                manifest = {"error": str(exc)}
        material_summary = cross_material_summary(rows, manifest)
        compatibility = {
            "metric_version": manifest.get("metric_version"),
            "molecule_system_name": manifest.get("molecule_system_name"),
            "siesta_settings_hash": manifest.get("siesta_settings_hash"),
            "model_config_hash": manifest.get("model_config_hash"),
            "test_sets": manifest.get("test_sets"),
            "selected_methods": manifest.get("selected_methods"),
            "material_labels": material_summary.get("material_labels"),
            "material_identity_hashes": material_summary.get("material_identity_hashes"),
            "material_compatibility_hashes": material_summary.get("material_compatibility_hashes"),
            "material_label_by_method": material_summary.get("material_label_by_method"),
            "material_identity_hash_by_method": material_summary.get("material_identity_hash_by_method"),
            "material_compatibility_hash_by_method": material_summary.get("material_compatibility_hash_by_method"),
        }
        compatibility_group_id = stable_payload_hash(compatibility, length=16)
        outputs = {
            "cross_evaluation_metrics": str(metrics_path),
            "winner_summary": str(experiment_dir / "summary" / "winner_summary.csv"),
            "winner_by_dataset_size": str(experiment_dir / "summary" / "winner_by_dataset_size.csv"),
            "winner_by_compute_budget": str(experiment_dir / "summary" / "winner_by_compute_budget.csv"),
            "dataset_size_thresholds_vs_md": str(experiment_dir / "summary" / "dataset_size_thresholds_vs_md.json"),
            "compute_budget_thresholds_vs_md": str(experiment_dir / "summary" / "compute_budget_thresholds_vs_md.json"),
            "recommendation": str(recommendation_path),
        }
        plot_warnings, plot_validity = cross_experiment_plot_warnings(
            rows=rows,
            recommendation=recommendation,
            manifest=manifest,
            outputs=outputs,
        )
        if material_summary.get("mixed_materials"):
            warning_entry(
                plot_warnings,
                code="mixed_material_groups",
                severity=("error" if material_summary.get("material_compatibility_warnings") else "warning"),
                status="scientifically_inconclusive",
                message="Multiple material groups are shown; interpret plots as diagnostics, not a pooled benchmark.",
                details=material_summary,
            )
        cross_experiments.append(
            {
                "experiment_id": experiment_dir.name,
                "metrics": rows,
                "metric_availability": cross_metric_availability(rows),
                "recommendation": recommendation,
                "manifest": manifest,
                "compatibility": compatibility,
                "compatibility_group_id": compatibility_group_id,
                "material_summary": material_summary,
                "methods": plot_validity["methods"],
                "test_sets": plot_validity["test_sets"],
                "plot_scientific_status": plot_validity["scientific_status"],
                "plot_warnings": plot_warnings,
                "outputs": outputs,
            }
        )
    groups_by_id: dict[str, dict[str, Any]] = {}
    for experiment in cross_experiments:
        group_id = str(experiment.get("compatibility_group_id") or "unknown")
        group = groups_by_id.setdefault(
            group_id,
            {
                "group_id": group_id,
                "compatibility": experiment.get("compatibility", {}),
                "experiment_ids": [],
                "rows": 0,
            },
        )
        group["experiment_ids"].append(experiment["experiment_id"])
        group["rows"] += len(experiment.get("metrics") or [])
        metric_version = (experiment.get("compatibility") or {}).get("metric_version")
        if metric_version is not None:
            group.setdefault("metric_versions", [])
            if metric_version not in group["metric_versions"]:
                group["metric_versions"].append(metric_version)
    compatible_groups = sorted(
        groups_by_id.values(),
        key=lambda group: (str(group["experiment_ids"][-1]) if group["experiment_ids"] else "", group["group_id"]),
    )
    latest_group = compatible_groups[-1] if compatible_groups else None
    visualization_warnings = []
    if len(compatible_groups) > 1:
        visualization_warnings.append(
            "All experiments are shown by default for visualization; compatibility or metric_version groups differ."
        )
    plot_diagnostics = {
        "run_metric_gaps": run_metric_gap_diagnostics(runs),
        "cross": cross_plot_diagnostics(cross_experiments=cross_experiments, runs=runs),
    }
    plot_warnings: list[dict[str, Any]] = []
    for experiment in cross_experiments:
        for warning in experiment.get("plot_warnings") or []:
            entry = {"experiment_id": experiment.get("experiment_id"), **warning}
            if entry not in plot_warnings:
                plot_warnings.append(entry)
    for warning in visualization_warnings:
        plot_warnings.append(
            {
                "severity": "warning",
                "code": "visualization_compatibility",
                "message": warning,
                "scientific_status": "exploratory_only",
            }
        )
    for diagnostic in plot_diagnostics["cross"]:
        if diagnostic.get("severity") == "warning":
            plot_warnings.append(
                {
                    "experiment_id": diagnostic.get("experiment_id"),
                    "severity": "warning",
                    "code": str(diagnostic.get("reason") or "cross_plot_warning"),
                    "message": str(diagnostic.get("message") or ""),
                    "scientific_status": "scientifically_inconclusive",
                    "details": diagnostic.get("warnings") or [],
                }
            )
    return {
        "runs": runs,
        "cross_experiments": cross_experiments,
        "compatible_experiment_groups": compatible_groups,
        "visualization_warnings": visualization_warnings,
        "plot_warnings": plot_warnings,
        "plot_diagnostics": plot_diagnostics,
        "default_plot_selection": {
            "mode": "all",
            "group_id": latest_group.get("group_id") if latest_group else None,
            "experiment_ids": [experiment["experiment_id"] for experiment in cross_experiments],
        },
    }


def atom_fc_ui_config() -> dict[str, Any]:
    config = load_config(PIPELINES["atom_displacement"].config_path)
    limit = atom_fc_sample_limit(config)
    force_constants = config.get("structure", {}).get("force_constants", {}) or {}
    entries = atom_fc_displacement_entries(config)
    normalized = []
    for entry in entries:
        normalized.append(
            {
                "value": entry.get("value", entry.get("displacement", "")),
                "n_structures": entry.get("n_structures") or "",
                "label": entry.get("label", ""),
            }
        )
    structures_per_displacement = force_constants.get("structures_per_displacement")
    if not structures_per_displacement:
        structures_per_displacement = [
            entry.get("n_structures")
            for entry in entries
            if entry.get("n_structures") not in (None, "")
        ]
    displacement_options = force_constants.get("displacement_options")
    if displacement_options is None and isinstance(force_constants.get("displacements"), dict):
        displacement_options = force_constants.get("displacements")
    if displacement_options is None:
        option_counts = structures_per_displacement or [2, 4, 6]
        displacement_options = {
            str(entry["value"]): option_counts
            for entry in entries
            if entry.get("value")
        }
    return {
        "max_per_displacement": limit,
        "include_reference": bool(force_constants.get("include_reference", True)),
        "subsampling": force_constants.get("subsampling", {}),
        "random_seed": force_constants.get(
            "random_seed",
            (force_constants.get("subsampling") or {}).get("seed", 0),
        ),
        "structures_per_displacement": structures_per_displacement or [2, 4, 6],
        "displacement_options": displacement_options,
        "combination_mode": parse_combination_mode(force_constants.get("combination_mode", "aligned")),
        "max_datasets": force_constants.get("max_datasets", 100),
        "splits": split_ratios_from_config(config),
        "displacements": normalized,
    }


class ExperimentRunner:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._processes: set[subprocess.Popen[str]] = set()
        self._logs: list[str] = []
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._returncode: int | None = None
        self._current: dict[str, Any] | None = None
        self._results: list[dict[str, Any]] = []
        self._progress: dict[str, Any] = {}
        self._stop_requested = False
        self._run_id: str | None = None
        self._rate_seconds_per_structure: dict[str, float] = {}

    def start(
        self,
        md_sizes: list[int],
        atom_sizes: list[int],
        fc_dataset_specs: dict[int, list[dict[str, Any]]] | None = None,
        atom_dataset_specs: list[dict[str, Any]] | None = None,
        split_ratios: dict[str, float] | None = None,
        random_seed: int | None = None,
        split_mode: str = DEFAULT_MD_SPLIT_MODE,
        test_sets: list[str] | None = None,
        primary_metric: str = DEFAULT_PRIMARY_METRIC,
        compute_budget_mode: str = "both",
        compute_accelerator: str = "cpu",
        selected_methods: list[str] | None = None,
        run_mode: str = "full_strict_pipeline",
        random_cartesian_options: dict[str, Any] | None = None,
        performance: dict[str, Any] | None = None,
        training_settings: dict[str, Any] | None = None,
        venv_activate_path: str | None = None,
        dataset_recipes_info: dict[str, Any] | None = None,
        reusable_split_policy: str = PRESERVE_ARCHIVED_SPLITS,
        training_plan: list[dict[str, Any]] | None = None,
        material: dict[str, Any] | None = None,
        hyperparameter_sweep: dict[str, Any] | None = None,
        deeph_comparison_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Ya hay una comparacion experimental en ejecucion.")
            selected_methods = normalize_selected_methods(selected_methods)
            run_mode = parse_run_mode(run_mode)
            reusable_split_policy = parse_reusable_split_policy(reusable_split_policy)
            training_plan = parse_training_plan(training_plan)
            validate_training_plan_for_run_mode(training_plan, run_mode)
            hyperparameter_sweep = parse_hyperparameter_sweep(hyperparameter_sweep)
            validate_training_plan_sweep_sources(training_plan, hyperparameter_sweep)
            deeph_comparison_options = (
                parse_deeph_comparison_options(
                    deeph_comparison_options or {},
                    require_graph2mat_result=run_mode == DEEPH_COMPARISON_RUN_MODE,
                )
                if run_mode in {DEEPH_COMPARISON_RUN_MODE, GRAPH2MAT_DEEPH_RUN_MODE}
                else None
            )
            material_config = parse_material_payload(material, required=material is not None)
            if material_config is not None:
                validate_material_payload(material_config, required=True)
            performance_settings = parse_performance_settings(
                performance,
                compute_accelerator=compute_accelerator,
            )
            training_settings = parse_training_settings(training_settings)
            compute_accelerator = str(performance_settings["compute_accelerator"])
            pipeline_keys = pipeline_keys_for_methods(selected_methods)
            random_cartesian_options = random_cartesian_options or {}
            md_config = load_config(PIPELINES["md"].config_path)
            ratios = split_ratios or split_ratios_from_config(md_config)
            if "md" in pipeline_keys:
                for size in md_sizes:
                    validate_split_sizes(size, ratios, label=f"MD dataset_{size}")
            else:
                md_sizes = []
            if "atom_displacement" in pipeline_keys:
                if atom_dataset_specs:
                    atom_sizes = [int(spec["size"]) for spec in atom_dataset_specs]
                    validate_atom_dataset_specs_for_fc(atom_dataset_specs, ratios)
                else:
                    validate_atom_sizes_for_fc(atom_sizes, fc_dataset_specs, ratios)
            else:
                atom_sizes = []
                fc_dataset_specs = None
                atom_dataset_specs = None
            if dataset_recipes_info is None:
                legacy_recipes = legacy_payload_to_dataset_recipes(
                    md_sizes=md_sizes,
                    atom_dataset_specs=atom_dataset_specs,
                    atom_sizes=atom_sizes,
                    fc_dataset_specs=fc_dataset_specs,
                    random_cartesian_options=random_cartesian_options,
                    selected_methods=selected_methods,
                )
                dataset_recipes_info = {
                    "recipes": legacy_recipes,
                    "recipe_set_hash": recipe_set_hash(legacy_recipes),
                    "md_dataset_specs": [],
                    "atom_dataset_specs": atom_dataset_specs or [],
                    "random_cartesian_dataset_specs": random_cartesian_options.get("_dataset_specs") or [],
                }
            self._logs = []
            self._started_at = time.time()
            self._finished_at = None
            self._returncode = None
            self._current = None
            self._results = []
            self._progress = {}
            self._processes = set()
            self._stop_requested = False
            self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._rate_seconds_per_structure = {}
            self._thread = threading.Thread(
                target=self._run,
                args=(
                    md_sizes,
                    atom_sizes,
                    self._run_id,
                    fc_dataset_specs,
                    atom_dataset_specs,
                    ratios,
                    random_seed,
                    split_mode,
                    test_sets or list(DEFAULT_COMMON_TEST_SETS),
                    primary_metric,
                    compute_budget_mode,
                    compute_accelerator,
                    selected_methods,
                    run_mode,
                    random_cartesian_options,
                    performance_settings,
                    training_settings,
                    venv_activate_path,
                    dataset_recipes_info,
                    reusable_split_policy,
                    training_plan,
                    material_config,
                    hyperparameter_sweep,
                    deeph_comparison_options,
                ),
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def start_deeph_comparison(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        parsed_options = parse_deeph_comparison_options(options or {})
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Ya hay una comparacion experimental en ejecucion.")
            self._logs = []
            self._started_at = time.time()
            self._finished_at = None
            self._returncode = None
            self._current = None
            self._results = []
            self._progress = {}
            self._processes = set()
            self._stop_requested = False
            self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._rate_seconds_per_structure = {}
            self._thread = threading.Thread(
                target=self._run_deeph_comparison,
                args=(self._run_id, parsed_options),
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._stop_requested = True
            processes = list(self._processes)
            if self._process is not None and self._process not in processes:
                processes.append(self._process)
            self._logs.append("\n[UI] Solicitud de parada enviada al experimento.\n")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            current = dict(self._current) if self._current is not None else None
            if current and running:
                started_at = current.get("started_at")
                if started_at is not None:
                    elapsed = time.time() - float(started_at)
                    current["elapsed_seconds"] = elapsed
                    current["eta_seconds"] = self._estimated_seconds(
                        str(current["pipeline"]),
                        int(current["size"]),
                        elapsed,
                    )
            return {
                "running": running,
                "returncode": None if running else self._returncode,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "current": current,
                "results": self._results,
                "progress": dict(self._progress),
                "log_size": len(self._logs),
                "run_id": self._run_id,
                "results_root": str(RESULTS_ROOT),
                "active_processes": len([process for process in self._processes if process.poll() is None]),
            }

    def logs(self, since: int = 0, limit: int | None = DEFAULT_LOG_RESPONSE_LIMIT) -> dict[str, Any]:
        with self._lock:
            log_payload = bounded_log_payload(self._logs, since=since, limit=limit)
            log_payload["status"] = self.status()
            return {
                **log_payload,
            }

    def _append(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)

    def _run_deeph_stage(
        self,
        label: str,
        command: list[str],
        *,
        cwd: Path = REPO_ROOT,
    ) -> int:
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        }
        self._append(f"\n[DEEPh] {label}\n")
        self._append(f"[RUN] {' '.join(shlex.quote(part) for part in command)}\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        with self._lock:
            self._process = process
            self._processes.add(process)
        self._append(f"[UI] PID: {process.pid}\n")
        started_at = time.time()
        try:
            returncode = stream_process_output(
                process,
                self._append,
                label=f"DeepH comparison: {label}",
            )
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None
                self._processes.discard(process)
        elapsed = time.time() - started_at
        self._append(
            f"[DEEPh] {label} finalizo con codigo {returncode} "
            f"en {format_duration(elapsed)}.\n"
        )
        if self._stop_requested:
            raise RuntimeError("Comparacion Graph2Mat vs DeepH detenida por el usuario.")
        if returncode != 0:
            raise RuntimeError(f"DeepH comparison fallo en etapa '{label}' con codigo {returncode}.")
        return returncode

    def _run_deeph_comparison(self, run_id: str, options: dict[str, Any]) -> None:
        if options.get("graph2mat_result_dirs") or options.get("graph2mat_candidate_summary_csv"):
            self._run_deeph_existing_graph2mat_candidates_comparison(run_id, options)
            return
        started_at = time.time()
        graph2mat_result_dir = Path(str(options["graph2mat_result_dir"]))
        deeph_repo = Path(str(options["deeph_repo"]))
        deeph_python = Path(str(options["deeph_python"]))
        pipeline_python = Path(str(options["pipeline_python"]))
        output_root = Path(str(options["output_root"]))
        output_dir = output_root / f"ui_run_{run_id}"
        raw_prepare_dir = output_dir / "raw_prepare"
        preprocess_dir = output_dir / "preprocess"
        processed_dir = preprocess_dir / "processed"
        train_dir = output_dir / "train"
        eval_dir = output_dir / "eval"
        comparison_dir = output_dir / "comparison"
        sample_limit = options.get("sample_limit_per_split")
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            with self._lock:
                self._current = {
                    "pipeline": DEEPH_COMPARISON_RUN_MODE,
                    "size": int(sample_limit or 0),
                    "dataset_label": "Graph2Mat vs DeepH",
                    "started_at": started_at,
                }
                self._progress = {
                    "stage": "starting",
                    "output_dir": str(output_dir),
                    "graph2mat_result_dir": str(graph2mat_result_dir),
                }
            manifest = {
                "run_id": run_id,
                "run_mode": DEEPH_COMPARISON_RUN_MODE,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "graph2mat_result_dir": str(graph2mat_result_dir),
                "deeph_repo": str(deeph_repo),
                "deeph_python": str(deeph_python),
                "pipeline_python": str(pipeline_python),
                "epochs": options["epochs"],
                "batch_size": options["batch_size"],
                "learning_rate": options["learning_rate"],
                "seed": options["seed"],
                "sample_limit_per_split": sample_limit,
                "allow_regenerate_siesta": options["allow_regenerate_siesta"],
                "siesta_command": options["siesta_command"],
                "device": options["device"],
                "output_dir": str(output_dir),
            }
            (output_dir / "ui_deeph_comparison_manifest.json").write_text(
                json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            self._append(f"[UI] Comparacion Graph2Mat vs DeepH {run_id} iniciada.\n")
            self._append(f"[UI] Graph2Mat result dir: {graph2mat_result_dir}\n")
            self._append(f"[UI] Output dir: {output_dir}\n")
            common_sample_args: list[str] = []
            if sample_limit is not None:
                common_sample_args = ["--sample-limit-per-split", str(sample_limit)]

            with self._lock:
                self._progress["stage"] = "prepare_deeph_raw"
            prepare_command = [
                str(pipeline_python),
                str(COMPARISON_ROOT / "scripts" / "prepare_deeph_siesta_dataset.py"),
                "--graph2mat-result-dir",
                str(graph2mat_result_dir),
                "--output-dir",
                str(raw_prepare_dir),
                "--deeph-repo",
                str(deeph_repo),
                "--siesta-command",
                str(options["siesta_command"]),
                *common_sample_args,
            ]
            if options.get("allow_regenerate_siesta"):
                prepare_command.append("--allow-regenerate-siesta")
            if options.get("copy_raw"):
                prepare_command.append("--copy")
            self._run_deeph_stage("1/5 preparar dataset raw SIESTA para DeepH", prepare_command)

            with self._lock:
                self._progress["stage"] = "deeph_preprocess"
            preprocess_command = [
                str(pipeline_python),
                str(COMPARISON_ROOT / "scripts" / "run_deeph_preprocess.py"),
                "--raw-dir",
                str(raw_prepare_dir / "raw"),
                "--processed-dir",
                str(processed_dir),
                "--output-dir",
                str(preprocess_dir),
                "--raw-manifest",
                str(raw_prepare_dir / "deeph_raw_manifest.json"),
                "--deeph-repo",
                str(deeph_repo),
                "--python",
                str(deeph_python),
                "--multiprocessing",
                "0",
            ]
            self._run_deeph_stage("2/5 preprocess DeepH SIESTA", preprocess_command)

            with self._lock:
                self._progress["stage"] = "deeph_train"
            train_command = [
                str(pipeline_python),
                str(COMPARISON_ROOT / "scripts" / "run_deeph_benchmark.py"),
                "--graph2mat-result-dir",
                str(graph2mat_result_dir),
                "--processed-dir",
                str(processed_dir),
                "--output-dir",
                str(train_dir),
                "--deeph-repo",
                str(deeph_repo),
                "--python",
                str(deeph_python),
                "--epochs",
                str(options["epochs"]),
                "--batch-size",
                str(options["batch_size"]),
                "--learning-rate",
                str(options["learning_rate"]),
                "--seed",
                str(options["seed"]),
                "--device",
                str(options["device"]),
                *common_sample_args,
            ]
            self._run_deeph_stage("3/5 entrenar DeepH con los splits de Graph2Mat", train_command)

            with self._lock:
                self._progress["stage"] = "deeph_evaluate"
            eval_command = [
                str(pipeline_python),
                str(COMPARISON_ROOT / "scripts" / "evaluate_deeph_kpoint_metrics.py"),
                "--graph2mat-result-dir",
                str(graph2mat_result_dir),
                "--processed-dir",
                str(processed_dir),
                "--predictions-dir",
                str(eval_dir / "predictions"),
                "--output-dir",
                str(eval_dir),
                "--trained-model-dir",
                str(train_dir / "training"),
                "--deeph-repo",
                str(deeph_repo),
                "--python",
                str(deeph_python),
                "--generate-predictions",
                "--split",
                "test",
                "--device",
                str(options["device"]),
                *common_sample_args,
            ]
            self._run_deeph_stage("4/5 evaluar DeepH con metricas k-point", eval_command)

            with self._lock:
                self._progress["stage"] = "aggregate"
            compare_command = [
                str(pipeline_python),
                str(COMPARISON_ROOT / "scripts" / "compare_graph2mat_deeph.py"),
                "--graph2mat-result-dir",
                str(graph2mat_result_dir),
                "--deeph-eval-dir",
                str(eval_dir),
                "--output-dir",
                str(comparison_dir),
            ]
            self._run_deeph_stage("5/5 agregar comparacion Graph2Mat vs DeepH", compare_command)

            with self._lock:
                self._results.append(
                    {
                        "pipeline": DEEPH_COMPARISON_RUN_MODE,
                        "dataset_label": "Graph2Mat vs DeepH",
                        "dataset_size": int(sample_limit or 0),
                        "predicted_hamiltonians": "",
                        "siesta_hamiltonians": "",
                        "result_dir": str(output_dir),
                        "comparison_report": str(comparison_dir / "final_report.md"),
                        "aggregate_csv": str(comparison_dir / "aggregate_graph2mat_vs_deeph.csv"),
                    }
                )
                self._progress["stage"] = "completed"
                self._current = None
                self._returncode = 0
                self._finished_at = time.time()
            self._append(f"[UI] Comparacion Graph2Mat vs DeepH completada: {comparison_dir}\n")
        except Exception as exc:
            with self._lock:
                self._returncode = 1
                self._finished_at = time.time()
                self._current = None
            self._append(f"[ERROR] Comparacion Graph2Mat vs DeepH fallo: {exc}\n")

    def _mean_metric_from_csv(self, path: Path, metric: str) -> float | None:
        rows = read_csv_rows(path)
        values: list[float] = []
        for row in rows:
            raw = row.get(metric)
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if not values:
            return None
        return sum(values) / len(values)

    def _graph2mat_candidate_score(
        self,
        result: dict[str, Any],
        primary_metric: str,
    ) -> tuple[float, str]:
        result_dir = Path(str(result.get("result_dir") or ""))
        metric_root = result_dir / "metrics"
        preferred: list[tuple[str, str]] = [
            ("kpoint_spectral_metrics.csv", primary_metric),
            ("spectral_metrics.csv", primary_metric),
            ("matrix_spectrum_relationship.csv", primary_metric),
            ("kpoint_matrix_metrics.csv", primary_metric),
            ("sparse_metrics.csv", primary_metric),
            ("kpoint_spectral_metrics.csv", "low_energy_rmse_eV"),
            ("kpoint_spectral_metrics.csv", "fermi_window_rmse_eV"),
            ("kpoint_matrix_metrics.csv", "h_mae_eV"),
            ("kpoint_spectral_metrics.csv", "global_rmse_eV"),
            ("sparse_metrics.csv", "relative_frobenius_union"),
        ]
        seen: set[tuple[str, str]] = set()
        for filename, metric in preferred:
            key = (filename, metric)
            if key in seen:
                continue
            seen.add(key)
            value = self._mean_metric_from_csv(metric_root / filename, metric)
            if value is not None:
                return value, metric
        return math.inf, "no_finite_metric"

    def _select_graph2mat_top_candidates(
        self,
        manifest: dict[str, Any],
        primary_metric: str,
        top_percent: float,
        top_count: int | None = None,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for run in manifest.get("runs") or []:
            if str(run.get("pipeline") or "") != "md":
                continue
            returncode = run.get("returncode", 1)
            if returncode is None:
                returncode = 1
            if int(returncode) != 0:
                continue
            if not Path(str(run.get("result_dir") or "")).exists():
                continue
            evaluation = run.get("hamiltonian_evaluation") or {}
            if int(evaluation.get("samples_compared") or 0) <= 0:
                continue
            score, metric = self._graph2mat_candidate_score(run, primary_metric)
            enriched = dict(run)
            enriched["_deeph_selection_score"] = score
            enriched["_deeph_selection_metric"] = metric
            candidates.append(enriched)
        if not candidates:
            raise RuntimeError("No hay candidatos MD Graph2Mat completados para comparar con DeepH.")
        ranked = sorted(
            candidates,
            key=lambda item: (
                float(item.get("_deeph_selection_score") or math.inf),
                str(item.get("dataset_label") or ""),
            ),
        )
        finite = [item for item in ranked if math.isfinite(float(item.get("_deeph_selection_score") or math.inf))]
        pool = finite or ranked
        count = top_count if top_count is not None else max(1, math.ceil(len(pool) * (top_percent / 100.0)))
        count = min(max(1, count), len(pool))
        return pool[:count]

    def _deeph_manual_candidate(self, result_dir: Path, rank: int, source: str) -> dict[str, Any]:
        if not result_dir.exists():
            raise RuntimeError(f"No existe el result dir de Graph2Mat: {result_dir}")
        return {
            "pipeline": "md",
            "returncode": 0,
            "dataset_label": result_dir.parent.name,
            "result_dir": str(result_dir),
            "_deeph_selection_metric": source,
            "_deeph_selection_score": rank,
        }

    def _select_graph2mat_candidates_from_summary_csv(
        self,
        path: Path,
        top_count: int | None,
    ) -> list[dict[str, Any]]:
        rows = read_csv_rows(path)
        candidates: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            raw_path = row.get("run_dir") or row.get("result_dir") or row.get("graph2mat_result_dir")
            if not raw_path:
                continue
            result_dir = resolve_ui_path(raw_path)
            if not result_dir.exists():
                continue
            score_raw = row.get("score_0_10")
            try:
                score = float(score_raw) if score_raw not in (None, "") else math.nan
            except (TypeError, ValueError):
                score = math.nan
            candidates.append(
                {
                    "pipeline": "md",
                    "returncode": 0,
                    "dataset_label": row.get("method_id") or row.get("dataset_label") or result_dir.parent.name,
                    "result_dir": str(result_dir),
                    "dataset_recipe_id": row.get("dataset_recipe_id"),
                    "_deeph_selection_metric": "score_0_10" if math.isfinite(score) else "summary_csv_order",
                    "_deeph_selection_score": score if math.isfinite(score) else index,
                    "_deeph_summary_selected": row.get("selected"),
                    "_deeph_summary_source": str(path),
                }
            )
        if not candidates:
            raise RuntimeError(f"No hay candidatos Graph2Mat validos en {path}.")

        def sort_key(candidate: dict[str, Any]) -> tuple[int, float, str]:
            metric = str(candidate.get("_deeph_selection_metric") or "")
            score = candidate.get("_deeph_selection_score")
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                numeric_score = math.nan
            if metric == "score_0_10" and math.isfinite(numeric_score):
                return (0, -numeric_score, str(candidate.get("dataset_label") or ""))
            if math.isfinite(numeric_score):
                return (1, numeric_score, str(candidate.get("dataset_label") or ""))
            return (2, math.inf, str(candidate.get("dataset_label") or ""))

        ranked = sorted(candidates, key=sort_key)
        count = top_count if top_count is not None else len(ranked)
        count = min(max(1, count), len(ranked))
        return ranked[:count]

    def _deeph_existing_graph2mat_candidates_from_options(self, options: dict[str, Any]) -> list[dict[str, Any]]:
        manual_dirs = [Path(path) for path in options.get("graph2mat_result_dirs") or []]
        if manual_dirs:
            return [
                self._deeph_manual_candidate(result_dir, rank, "manual_list")
                for rank, result_dir in enumerate(manual_dirs, start=1)
            ]
        summary_csv = str(options.get("graph2mat_candidate_summary_csv") or "")
        if summary_csv:
            return self._select_graph2mat_candidates_from_summary_csv(
                Path(summary_csv),
                options.get("graph2mat_top_count"),
            )
        return []

    def _run_deeph_existing_graph2mat_candidates_comparison(self, run_id: str, options: dict[str, Any]) -> None:
        started_at = time.time()
        output_root = Path(str(options["output_root"]))
        output_dir = output_root / f"ui_existing_graph2mat_deeph_{run_id}"
        try:
            candidates = self._deeph_existing_graph2mat_candidates_from_options(options)
            if not candidates:
                raise RuntimeError("No hay candidatos Graph2Mat existentes para comparar con DeepH.")
            with self._lock:
                self._current = {
                    "pipeline": DEEPH_COMPARISON_RUN_MODE,
                    "size": int(options.get("sample_limit_per_split") or 0),
                    "dataset_label": f"Graph2Mat existing top {len(candidates)} vs DeepH",
                    "started_at": started_at,
                }
                self._progress = {
                    "stage": "starting_existing_candidates",
                    "output_dir": str(output_dir),
                    "graph2mat_candidates": len(candidates),
                }
            self._append(
                f"[UI] Comparacion DeepH para candidatos Graph2Mat existentes iniciada. "
                f"Candidatos: {len(candidates)}.\n"
            )
            result = self._run_deeph_group_for_graph2mat_candidates(
                run_id,
                options,
                candidates,
                output_dir,
            )
            with self._lock:
                self._results.append(result)
                self._progress["stage"] = "completed"
                self._current = None
                self._returncode = 0
                self._finished_at = time.time()
            self._append(f"[UI] Comparacion DeepH multi-candidato completada: {output_dir}\n")
        except Exception as exc:
            with self._lock:
                self._returncode = 1
                self._finished_at = time.time()
                self._current = None
            self._append(f"[ERROR] Comparacion DeepH multi-candidato fallo: {exc}\n")

    def _run_deeph_group_for_graph2mat_candidates(
        self,
        run_id: str,
        options: dict[str, Any],
        candidates: list[dict[str, Any]],
        output_dir: Path,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=False)
        comparison_root = output_dir / "comparison"
        comparison_root.mkdir(parents=True, exist_ok=True)
        selection_rows = [
            {
                "rank": index,
                "dataset_label": candidate.get("dataset_label"),
                "result_dir": candidate.get("result_dir"),
                "selection_metric": candidate.get("_deeph_selection_metric"),
                "selection_score": candidate.get("_deeph_selection_score"),
                "split_manifest_hash": candidate.get("split_manifest_hash"),
            }
            for index, candidate in enumerate(candidates, start=1)
        ]
        write_csv_dicts(comparison_root / "selected_graph2mat_candidates.csv", selection_rows)
        (comparison_root / "selected_graph2mat_candidates.json").write_text(
            json.dumps(json_safe(selection_rows), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._append(
            "[DEEPh] One-to-one mode: DeepH se entrenara por separado para cada candidato Graph2Mat top. "
            f"Candidatos: {len(candidates)}.\n"
        )
        comparison_outputs = []
        for index, candidate in enumerate(candidates, start=1):
            candidate_root = output_dir / f"top_{index:02d}"
            comparison_outputs.append(
                self._run_deeph_one_to_one_candidate(
                    index,
                    len(candidates),
                    options,
                    candidate,
                    candidate_root,
                )
            )
        write_csv_dicts(comparison_root / "comparison_outputs.csv", comparison_outputs)
        (comparison_root / "comparison_outputs.json").write_text(
            json.dumps(json_safe(comparison_outputs), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return {
            "pipeline": GRAPH2MAT_DEEPH_RUN_MODE,
            "dataset_label": f"Graph2Mat top {len(candidates)} vs DeepH one-to-one",
            "dataset_size": int(options.get("sample_limit_per_split") or 0),
            "predicted_hamiltonians": "",
            "siesta_hamiltonians": "",
            "result_dir": str(output_dir),
            "comparison_report": str(comparison_root / "comparison_outputs.csv"),
            "aggregate_csv": str(comparison_root / "comparison_outputs.csv"),
            "graph2mat_candidates_compared": len(candidates),
            "deeph_trainings": len(candidates),
        }

    def _run_deeph_one_to_one_candidate(
        self,
        rank: int,
        total: int,
        options: dict[str, Any],
        candidate: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        graph2mat_result_dir = Path(str(candidate["result_dir"]))
        deeph_repo = Path(str(options["deeph_repo"]))
        deeph_python = Path(str(options["deeph_python"]))
        pipeline_python = Path(str(options["pipeline_python"]))
        raw_prepare_dir = output_dir / "raw_prepare"
        preprocess_dir = output_dir / "preprocess"
        processed_dir = preprocess_dir / "processed"
        train_dir = output_dir / "train"
        eval_dir = output_dir / "eval"
        comparison_dir = output_dir / "comparison"
        sample_limit = options.get("sample_limit_per_split")
        output_dir.mkdir(parents=True, exist_ok=False)
        common_sample_args = ["--sample-limit-per-split", str(sample_limit)] if sample_limit is not None else []
        self._append(
            f"[DEEPh] Top {rank}/{total}: entrenando DeepH para "
            f"{candidate.get('dataset_label')} | {graph2mat_result_dir}\n"
        )
        with self._lock:
            self._current = {
                "pipeline": GRAPH2MAT_DEEPH_RUN_MODE,
                "size": int(sample_limit or 0),
                "dataset_label": f"DeepH top {rank}/{total}",
                "started_at": time.time(),
            }
            self._progress = {
                "stage": f"top_{rank}_prepare_deeph_raw",
                "output_dir": str(output_dir),
                "graph2mat_result_dir": str(graph2mat_result_dir),
            }
        prepare_command = [
            str(pipeline_python),
            str(COMPARISON_ROOT / "scripts" / "prepare_deeph_siesta_dataset.py"),
            "--graph2mat-result-dir",
            str(graph2mat_result_dir),
            "--output-dir",
            str(raw_prepare_dir),
            "--deeph-repo",
            str(deeph_repo),
            "--siesta-command",
            str(options["siesta_command"]),
            *common_sample_args,
        ]
        if options.get("allow_regenerate_siesta"):
            prepare_command.append("--allow-regenerate-siesta")
        if options.get("copy_raw"):
            prepare_command.append("--copy")
        self._run_deeph_stage(f"top {rank}/{total} 1/5 preparar raw SIESTA", prepare_command)

        with self._lock:
            self._progress["stage"] = f"top_{rank}_deeph_preprocess"
        preprocess_command = [
            str(pipeline_python),
            str(COMPARISON_ROOT / "scripts" / "run_deeph_preprocess.py"),
            "--raw-dir",
            str(raw_prepare_dir / "raw"),
            "--processed-dir",
            str(processed_dir),
            "--output-dir",
            str(preprocess_dir),
            "--raw-manifest",
            str(raw_prepare_dir / "deeph_raw_manifest.json"),
            "--deeph-repo",
            str(deeph_repo),
            "--python",
            str(deeph_python),
            "--multiprocessing",
            "0",
        ]
        self._run_deeph_stage(f"top {rank}/{total} 2/5 preprocess DeepH", preprocess_command)

        with self._lock:
            self._progress["stage"] = f"top_{rank}_deeph_train"
        train_command = [
            str(pipeline_python),
            str(COMPARISON_ROOT / "scripts" / "run_deeph_benchmark.py"),
            "--graph2mat-result-dir",
            str(graph2mat_result_dir),
            "--processed-dir",
            str(processed_dir),
            "--output-dir",
            str(train_dir),
            "--deeph-repo",
            str(deeph_repo),
            "--python",
            str(deeph_python),
            "--epochs",
            str(options["epochs"]),
            "--batch-size",
            str(options["batch_size"]),
            "--learning-rate",
            str(options["learning_rate"]),
            "--seed",
            str(options["seed"]),
            "--device",
            str(options["device"]),
            *common_sample_args,
        ]
        self._run_deeph_stage(f"top {rank}/{total} 3/5 entrenar DeepH", train_command)

        with self._lock:
            self._progress["stage"] = f"top_{rank}_deeph_evaluate"
        eval_command = [
            str(pipeline_python),
            str(COMPARISON_ROOT / "scripts" / "evaluate_deeph_kpoint_metrics.py"),
            "--graph2mat-result-dir",
            str(graph2mat_result_dir),
            "--processed-dir",
            str(processed_dir),
            "--predictions-dir",
            str(eval_dir / "predictions"),
            "--output-dir",
            str(eval_dir),
            "--trained-model-dir",
            str(train_dir / "training"),
            "--deeph-repo",
            str(deeph_repo),
            "--python",
            str(deeph_python),
            "--generate-predictions",
            "--split",
            "test",
            "--device",
            str(options["device"]),
            *common_sample_args,
        ]
        self._run_deeph_stage(f"top {rank}/{total} 4/5 evaluar DeepH k-point", eval_command)

        with self._lock:
            self._progress["stage"] = f"top_{rank}_aggregate"
        compare_command = [
            str(pipeline_python),
            str(COMPARISON_ROOT / "scripts" / "compare_graph2mat_deeph.py"),
            "--graph2mat-result-dir",
            str(graph2mat_result_dir),
            "--deeph-eval-dir",
            str(eval_dir),
            "--output-dir",
            str(comparison_dir),
        ]
        self._run_deeph_stage(f"top {rank}/{total} 5/5 comparar 1 a 1", compare_command)
        return {
            "rank": rank,
            "dataset_label": candidate.get("dataset_label"),
            "graph2mat_result_dir": str(graph2mat_result_dir),
            "deeph_output_dir": str(output_dir),
            "comparison_dir": str(comparison_dir),
            "report": str(comparison_dir / "final_report.md"),
            "aggregate_csv": str(comparison_dir / "aggregate_graph2mat_vs_deeph.csv"),
            "selection_metric": candidate.get("_deeph_selection_metric"),
            "selection_score": candidate.get("_deeph_selection_score"),
        }

    def _run_deeph_group_for_graph2mat_candidates_old_single_deeph(
        self,
        run_id: str,
        options: dict[str, Any],
        candidates: list[dict[str, Any]],
        output_dir: Path,
    ) -> dict[str, Any]:
        representative = candidates[0]
        graph2mat_result_dir = Path(str(representative["result_dir"]))
        deeph_repo = Path(str(options["deeph_repo"]))
        deeph_python = Path(str(options["deeph_python"]))
        pipeline_python = Path(str(options["pipeline_python"]))
        raw_prepare_dir = output_dir / "raw_prepare"
        preprocess_dir = output_dir / "preprocess"
        processed_dir = preprocess_dir / "processed"
        train_dir = output_dir / "train"
        eval_dir = output_dir / "eval"
        comparison_root = output_dir / "comparison"
        sample_limit = options.get("sample_limit_per_split")
        output_dir.mkdir(parents=True, exist_ok=False)
        common_sample_args = ["--sample-limit-per-split", str(sample_limit)] if sample_limit is not None else []
        self._append(
            "[DEEPh] Representative Graph2Mat split for DeepH: "
            f"{representative.get('dataset_label')} | {graph2mat_result_dir}\n"
        )
        self._append(
            "[DEEPh] Graph2Mat top candidates selected: "
            f"{len(candidates)} (top {options.get('graph2mat_top_percent')}%).\n"
        )
        selection_rows = []
        for index, candidate in enumerate(candidates, start=1):
            selection_rows.append(
                {
                    "rank": index,
                    "dataset_label": candidate.get("dataset_label"),
                    "result_dir": candidate.get("result_dir"),
                    "selection_metric": candidate.get("_deeph_selection_metric"),
                    "selection_score": candidate.get("_deeph_selection_score"),
                    "split_manifest_hash": candidate.get("split_manifest_hash"),
                }
            )
        comparison_root.mkdir(parents=True, exist_ok=True)
        write_csv_dicts(comparison_root / "selected_graph2mat_candidates.csv", selection_rows)
        (comparison_root / "selected_graph2mat_candidates.json").write_text(
            json.dumps(json_safe(selection_rows), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        with self._lock:
            self._current = {
                "pipeline": GRAPH2MAT_DEEPH_RUN_MODE,
                "size": int(sample_limit or 0),
                "dataset_label": "DeepH on Graph2Mat top candidates",
                "started_at": time.time(),
            }
            self._progress = {
                "stage": "prepare_deeph_raw",
                "output_dir": str(output_dir),
                "graph2mat_result_dir": str(graph2mat_result_dir),
            }
        prepare_command = [
            str(pipeline_python),
            str(COMPARISON_ROOT / "scripts" / "prepare_deeph_siesta_dataset.py"),
            "--graph2mat-result-dir",
            str(graph2mat_result_dir),
            "--output-dir",
            str(raw_prepare_dir),
            "--deeph-repo",
            str(deeph_repo),
            "--siesta-command",
            str(options["siesta_command"]),
            *common_sample_args,
        ]
        if options.get("allow_regenerate_siesta"):
            prepare_command.append("--allow-regenerate-siesta")
        if options.get("copy_raw"):
            prepare_command.append("--copy")
        self._run_deeph_stage("1/5 preparar dataset raw SIESTA para DeepH", prepare_command)

        with self._lock:
            self._progress["stage"] = "deeph_preprocess"
        preprocess_command = [
            str(pipeline_python),
            str(COMPARISON_ROOT / "scripts" / "run_deeph_preprocess.py"),
            "--raw-dir",
            str(raw_prepare_dir / "raw"),
            "--processed-dir",
            str(processed_dir),
            "--output-dir",
            str(preprocess_dir),
            "--raw-manifest",
            str(raw_prepare_dir / "deeph_raw_manifest.json"),
            "--deeph-repo",
            str(deeph_repo),
            "--python",
            str(deeph_python),
            "--multiprocessing",
            "0",
        ]
        self._run_deeph_stage("2/5 preprocess DeepH SIESTA", preprocess_command)

        with self._lock:
            self._progress["stage"] = "deeph_train"
        train_command = [
            str(pipeline_python),
            str(COMPARISON_ROOT / "scripts" / "run_deeph_benchmark.py"),
            "--graph2mat-result-dir",
            str(graph2mat_result_dir),
            "--processed-dir",
            str(processed_dir),
            "--output-dir",
            str(train_dir),
            "--deeph-repo",
            str(deeph_repo),
            "--python",
            str(deeph_python),
            "--epochs",
            str(options["epochs"]),
            "--batch-size",
            str(options["batch_size"]),
            "--learning-rate",
            str(options["learning_rate"]),
            "--seed",
            str(options["seed"]),
            "--device",
            str(options["device"]),
            *common_sample_args,
        ]
        self._run_deeph_stage("3/5 entrenar DeepH con el split Graph2Mat", train_command)

        with self._lock:
            self._progress["stage"] = "deeph_evaluate"
        eval_command = [
            str(pipeline_python),
            str(COMPARISON_ROOT / "scripts" / "evaluate_deeph_kpoint_metrics.py"),
            "--graph2mat-result-dir",
            str(graph2mat_result_dir),
            "--processed-dir",
            str(processed_dir),
            "--predictions-dir",
            str(eval_dir / "predictions"),
            "--output-dir",
            str(eval_dir),
            "--trained-model-dir",
            str(train_dir / "training"),
            "--deeph-repo",
            str(deeph_repo),
            "--python",
            str(deeph_python),
            "--generate-predictions",
            "--split",
            "test",
            "--device",
            str(options["device"]),
            *common_sample_args,
        ]
        self._run_deeph_stage("4/5 evaluar DeepH con metricas k-point", eval_command)

        comparison_outputs = []
        for index, candidate in enumerate(candidates, start=1):
            candidate_result_dir = Path(str(candidate["result_dir"]))
            safe_label = compact_dataset_label(
                f"g2m_top{index}_{candidate.get('dataset_label') or candidate_result_dir.parent.name}",
                {"rank": index, "result_dir": str(candidate_result_dir)},
                max_length=96,
            )
            candidate_comparison_dir = comparison_root / safe_label
            with self._lock:
                self._progress["stage"] = f"aggregate_top_{index}_of_{len(candidates)}"
            compare_command = [
                str(pipeline_python),
                str(COMPARISON_ROOT / "scripts" / "compare_graph2mat_deeph.py"),
                "--graph2mat-result-dir",
                str(candidate_result_dir),
                "--deeph-eval-dir",
                str(eval_dir),
                "--output-dir",
                str(candidate_comparison_dir),
            ]
            self._run_deeph_stage(
                f"5/5 agregar comparacion Graph2Mat top {index}/{len(candidates)} vs DeepH",
                compare_command,
            )
            comparison_outputs.append(
                {
                    "rank": index,
                    "dataset_label": candidate.get("dataset_label"),
                    "result_dir": str(candidate_result_dir),
                    "comparison_dir": str(candidate_comparison_dir),
                    "report": str(candidate_comparison_dir / "final_report.md"),
                    "aggregate_csv": str(candidate_comparison_dir / "aggregate_graph2mat_vs_deeph.csv"),
                    "selection_metric": candidate.get("_deeph_selection_metric"),
                    "selection_score": candidate.get("_deeph_selection_score"),
                }
            )
        write_csv_dicts(comparison_root / "comparison_outputs.csv", comparison_outputs)
        (comparison_root / "comparison_outputs.json").write_text(
            json.dumps(json_safe(comparison_outputs), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return {
            "pipeline": GRAPH2MAT_DEEPH_RUN_MODE,
            "dataset_label": f"Graph2Mat top {len(candidates)} vs DeepH",
            "dataset_size": int(sample_limit or 0),
            "predicted_hamiltonians": "",
            "siesta_hamiltonians": "",
            "result_dir": str(output_dir),
            "comparison_report": str(comparison_root / "comparison_outputs.csv"),
            "aggregate_csv": str(comparison_root / "comparison_outputs.csv"),
            "representative_graph2mat_result_dir": str(graph2mat_result_dir),
            "graph2mat_candidates_compared": len(candidates),
        }

    def _initial_experiment_manifest(
        self,
        run_id: str,
        md_sizes: list[int],
        atom_sizes: list[int],
        split_ratios: dict[str, float],
        random_seed: int | None,
        split_mode: str,
        atom_dataset_specs: list[dict[str, Any]] | None,
        test_sets: list[str],
        primary_metric: str,
        compute_budget_mode: str,
        compute_accelerator: str,
        selected_methods: list[str],
        run_mode: str,
        random_cartesian_options: dict[str, Any] | None = None,
        performance: dict[str, Any] | None = None,
        training_settings: dict[str, Any] | None = None,
        dataset_recipes_info: dict[str, Any] | None = None,
        material: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        md_config = load_config(PIPELINES["md"].config_path)
        atom_config = load_config(PIPELINES["atom_displacement"].config_path)
        material_config = parse_material_payload(material, required=material is not None)
        for config in (md_config, atom_config):
            apply_material_to_config(config, material_config)
        performance_settings = parse_performance_settings(
            performance,
            compute_accelerator=compute_accelerator,
        )
        training_settings = parse_training_settings(training_settings)
        for config in (md_config, atom_config):
            apply_performance_to_config(config, performance_settings)
            apply_training_settings_to_config(config, training_settings)
        shared_settings = load_config(DEFAULT_SHARED) if DEFAULT_SHARED.exists() else {}
        configs_by_method = {
            "md": md_config,
            "siesta_fc_cartesian": atom_config,
            "random_cartesian": atom_config,
        }
        model_report = compare_method_model_settings(
            configs_by_method,
            selected_methods=list(selected_methods),
        )
        files_to_hash = [
            PIPELINES["md"].config_path,
            PIPELINES["atom_displacement"].config_path,
            PIPELINES["md"].root / "dataset" / "RUN.fdf",
            PIPELINES["atom_displacement"].root / "base" / "RUN.fdf",
            *sorted((PIPELINES["md"].root / "dataset").glob("*.psf")),
            *sorted((PIPELINES["atom_displacement"].root / "base").glob("*.psf")),
            *sorted((PIPELINES["atom_displacement"].root / "relaxed").glob("*.ion.xml")),
        ]
        md_basis_files = sorted((PIPELINES["md"].root / "dataset" / "MD_steps" / "basis").glob("*.ion.xml"))
        atom_basis_files = (
            sorted((PIPELINES["atom_displacement"].root / "dataset" / "FC_steps" / "basis").glob("*.ion.xml"))
            or sorted((PIPELINES["atom_displacement"].root / "dataset" / "AtDis_steps" / "basis").glob("*.ion.xml"))
            or sorted((PIPELINES["atom_displacement"].root / "relaxed").glob("*.ion.xml"))
        )
        random_cartesian_basis_files = (
            sorted((PIPELINES["atom_displacement"].root / "dataset" / "RandomCartesian_steps" / "basis").glob("*.ion.xml"))
            or sorted((PIPELINES["atom_displacement"].root / "relaxed").glob("*.ion.xml"))
        )
        md_pseudo_files = sorted((PIPELINES["md"].root / "dataset").glob("*.psf"))
        atom_pseudo_files = sorted((PIPELINES["atom_displacement"].root / "base").glob("*.psf"))
        method_artifact_paths = {
            "md": {"basis_files": md_basis_files, "pseudopotential_files": md_pseudo_files},
            "siesta_fc_cartesian": {"basis_files": atom_basis_files, "pseudopotential_files": atom_pseudo_files},
            "random_cartesian": {
                "basis_files": random_cartesian_basis_files,
                "pseudopotential_files": atom_pseudo_files,
            },
        }
        artifact_hashes_by_method = {
            method: {
                "basis_hash": files_content_digest(paths["basis_files"]),
                "pseudopotential_hash": files_content_digest(paths["pseudopotential_files"]),
            }
            for method, paths in method_artifact_paths.items()
            if method in selected_methods
        }
        siesta_report = compare_method_settings(
            configs_by_method,
            shared_settings,
            artifact_hashes_by_method=artifact_hashes_by_method,
            selected_methods=list(selected_methods),
        )
        basis_pseudopotential_warning = siesta_report.get("basis_pseudopotential_warning", "")
        basis_and_pseudos = [
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for path in files_to_hash
            if path.exists() and path.suffix in {".psf", ".xml", ".fdf"}
        ]
        system_name = (
            md_config.get("md", {}).get("system_name")
            or atom_config.get("structure", {}).get("system_name")
            or "unknown"
        )
        resolved_paths = [
            resolve_pipeline_path(PIPELINES["md"], value)
            for value in (md_config.get("paths", {}) or {}).values()
            if isinstance(value, (str, Path))
        ] + [
            resolve_pipeline_path(PIPELINES["atom_displacement"], value)
            for value in (atom_config.get("paths", {}) or {}).values()
            if isinstance(value, (str, Path))
        ]
        artifact_hashes = {
            "configs": files_digest([PIPELINES["md"].config_path, PIPELINES["atom_displacement"].config_path]),
            "md_pseudopotentials": files_content_digest(md_pseudo_files),
            "atom_displacement_pseudopotentials": files_content_digest(atom_pseudo_files),
            "siesta_fc_cartesian_pseudopotentials": files_content_digest(atom_pseudo_files),
            "random_cartesian_pseudopotentials": files_content_digest(atom_pseudo_files),
            "md_basis": files_content_digest(md_basis_files),
            "atom_displacement_basis": files_content_digest(atom_basis_files),
            "siesta_fc_cartesian_basis": files_content_digest(atom_basis_files),
            "random_cartesian_basis": files_content_digest(random_cartesian_basis_files),
            "rendered_inputs": file_digest([
                PIPELINES["md"].root / "dataset" / "RUN.fdf",
                PIPELINES["atom_displacement"].root / "base" / "RUN.fdf",
            ]),
        }
        dataset_recipes_info = dataset_recipes_info or {}
        normalized_recipes = dataset_recipes_info.get("recipes") or {}
        normalized_recipe_hash = dataset_recipes_info.get("recipe_set_hash") or recipe_set_hash(normalized_recipes)
        dataset_seeds = dataset_recipe_seed_values(normalized_recipes)
        if random_seed is not None and random_seed not in dataset_seeds:
            dataset_seeds.append(random_seed)
        return {
            "experiment_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "dirty_tree_warning": git_dirty_warning(),
            "environment_versions": environment_versions([md_config, atom_config]),
            "artifact_hashes": artifact_hashes,
            "reproducibility_warning": absolute_path_warning(resolved_paths),
            "basis_pseudopotential_warning": basis_pseudopotential_warning,
            "metric_version": METRIC_VERSION,
            "molecule_system_name": system_name,
            "config_hash": files_digest([PIPELINES["md"].config_path, PIPELINES["atom_displacement"].config_path]),
            "siesta_settings_hash": siesta_report["siesta_settings_hash"],
            "md_siesta_settings_hash": siesta_report.get("md_siesta_settings_hash"),
            "atom_displacement_siesta_settings_hash": siesta_report.get("atom_displacement_siesta_settings_hash"),
            "siesta_fc_cartesian_siesta_settings_hash": siesta_report.get("siesta_fc_cartesian_siesta_settings_hash"),
            "random_cartesian_siesta_settings_hash": siesta_report.get("random_cartesian_siesta_settings_hash"),
            "siesta_settings_hash_by_method": siesta_report.get("siesta_settings_hash_by_method", {}),
            "method_siesta_settings": siesta_report.get("method_siesta_settings", {}),
            "shared_siesta_settings_hash": siesta_report["shared_siesta_settings_hash"],
            "siesta_settings_warning": siesta_report.get("severe_warning", ""),
            "siesta_settings_nonsevere_warning": (
                siesta_report.get("warning", "") if not siesta_report.get("severe_warning") else ""
            ),
            "siesta_settings_severe_warning": siesta_report.get("severe_warning", ""),
            "siesta_settings_mismatches": siesta_report.get("pairwise_mismatch_report", []),
            "siesta_settings_pairwise_mismatch_report": siesta_report.get("pairwise_mismatch_report", []),
            "siesta_settings_severe_mismatches": siesta_report.get("severe_mismatches", []),
            "model_config_hash": model_report["model_config_hash"],
            "md_model_config_hash": model_report.get("md_model_config_hash"),
            "atom_displacement_model_config_hash": model_report.get("atom_displacement_model_config_hash"),
            "siesta_fc_cartesian_model_config_hash": model_report.get("siesta_fc_cartesian_model_config_hash"),
            "random_cartesian_model_config_hash": model_report.get("random_cartesian_model_config_hash"),
            "model_config_hash_by_method": model_report.get("model_config_hash_by_method", {}),
            "method_model_settings": model_report.get("method_model_settings", {}),
            "model_config_warning": model_report.get("severe_warning", ""),
            "model_config_nonsevere_warning": (
                model_report.get("warning", "") if not model_report.get("severe_warning") else ""
            ),
            "model_config_severe_warning": model_report.get("severe_warning", ""),
            "model_config_mismatches": model_report.get("pairwise_mismatch_report", []),
            "model_config_pairwise_mismatch_report": model_report.get("pairwise_mismatch_report", []),
            "model_config_severe_mismatches": model_report.get("severe_mismatches", []),
            "basis_hash": files_content_digest([
                *md_basis_files,
                *atom_basis_files,
                *random_cartesian_basis_files,
            ]),
            "pseudopotential_hash": files_content_digest([*md_pseudo_files, *atom_pseudo_files]),
            "basis_hash_by_method": siesta_report.get("basis_hash_by_method", {}),
            "pseudopotential_hash_by_method": siesta_report.get("pseudopotential_hash_by_method", {}),
            "basis_pseudopotential_info": basis_and_pseudos,
            "run_mode": run_mode,
            "scientific_status": "dataset_only" if run_mode == DATASET_ONLY_RUN_MODE else (
                "non_comparative" if len(selected_methods) < 2 else "pending"
            ),
            "selected_methods": list(selected_methods),
            "method_registry": method_registry_payload(),
            "train_methods": pipeline_keys_for_methods(selected_methods),
            "dataset_sizes": {
                "md": md_sizes,
                "siesta_fc_cartesian": atom_sizes,
                "atom_displacement": atom_sizes,
                "random_cartesian": random_cartesian_sizes_from_options(random_cartesian_options or {})
                if "random_cartesian" in selected_methods
                else [],
            },
            "dataset_recipes": normalized_recipes,
            "dataset_recipe_set_hash": normalized_recipe_hash,
            "dataset_recipe_specs": {
                "md": dataset_recipes_info.get("md_dataset_specs") or [],
                "siesta_fc_cartesian": dataset_recipes_info.get("atom_dataset_specs") or atom_dataset_specs or [],
                "random_cartesian": dataset_recipes_info.get("random_cartesian_dataset_specs") or (
                    (random_cartesian_options or {}).get("_dataset_specs") or []
                ),
            },
            "random_cartesian_options": random_cartesian_options or {},
            "atom_displacement_dataset_specs": atom_dataset_specs or [],
            "seeds": dataset_seeds,
            "split_ratios": split_ratios,
            "split_mode": split_mode,
            "minimum_robust_seeds": MINIMUM_ROBUST_SEEDS,
            "test_sets": list(test_sets),
            "training_hyperparameters": {
                "md": md_config.get("training", {}),
                "atom_displacement": atom_config.get("training", {}),
                "siesta_fc_cartesian": atom_config.get("training", {}),
                "random_cartesian": atom_config.get("training", {}) if "random_cartesian" in selected_methods else {},
            },
            "selected_metrics": {
                "sparse": True,
                "spectral": True,
                "dos": True,
                "primary_metric": primary_metric,
            },
            "compute_budget_mode": compute_budget_mode,
            "compute_accelerator": compute_accelerator,
            "performance": dict(performance_settings),
            "optimization_settings": dict(performance_settings),
            "training_settings": dict(training_settings),
            "strict_comparison_mode": STRICT_COMPARISON_MODE,
            "output_directories": {
                "experiment_root": str(experiment_root(run_id)),
                "manifest": str(experiment_manifest_path(run_id)),
                "common_tests": str(experiment_root(run_id) / "common_tests"),
                "cross_evaluations": str(experiment_root(run_id) / "cross_evaluations"),
                "summary": str(experiment_root(run_id) / "summary"),
            },
            "timing": {
                "stages": [],
                "by_method": {},
                "by_dataset": {},
                "counters": {
                    "siesta_launched": 0,
                    "siesta_skipped_or_reused": 0,
                    "siesta_failed": 0,
                    "graph2mat_trainings": 0,
                    "predictions": 0,
                    "cross_evaluations": 0,
                },
                "md_siesta_generation_seconds": None,
                "atom_displacement_siesta_generation_seconds": None,
                "dataset_preparation_seconds": None,
                "normalization_seconds": None,
                "training_seconds": {},
                "prediction_seconds": {},
                "evaluation_seconds": {},
                "winner_analysis_seconds": None,
                "total_experiment_seconds": None,
                "total_seconds": None,
                "timing_incomplete_warning": (
                    "Per-phase generation/training/prediction timings are partly unavailable "
                    "for legacy Graph2Mat entrypoints; run manifests keep measured totals."
                ),
            },
            "runs": [],
            "warnings": [],
            "cross_evaluation": {},
        }

    def _write_experiment_manifest(self, manifest: dict[str, Any]) -> None:
        path = experiment_manifest_path(str(manifest["experiment_id"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        refresh_method_provenance(manifest)
        write_yaml(path, json_safe(manifest))

    def _write_performance_report(self, manifest: dict[str, Any]) -> None:
        root = experiment_root(str(manifest["experiment_id"]))
        report = {
            "experiment_id": manifest.get("experiment_id"),
            "performance": manifest.get("performance", {}),
            "timing": manifest.get("timing", {}),
            "scientific_status": manifest.get("scientific_status"),
            "warnings": manifest.get("warnings", []),
        }
        (root / "performance_report.json").write_text(
            json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        rows = []
        for stage in (manifest.get("timing", {}) or {}).get("stages", []) or []:
            if isinstance(stage, dict):
                rows.append(stage)
        write_csv_dicts(root / "performance_report.csv", rows or [{"stage": "none", "wall_time_seconds": ""}])

    def _record_run_result(self, manifest: dict[str, Any], result: dict[str, Any]) -> None:
        with self._lock:
            self._results.append(result)
        manifest["runs"].append(result)
        self._merge_run_timing(manifest, result)
        self._write_experiment_manifest(manifest)

    def _run_dataset_tasks(
        self,
        tasks: list[tuple[str, Any]],
        *,
        manifest: dict[str, Any],
        workers: int,
        error_policy: str,
    ) -> list[dict[str, Any]]:
        results_by_index: dict[int, dict[str, Any]] = {}
        failures: list[str] = []
        with self._lock:
            self._progress.update(
                {
                    "active_stage": "dataset_jobs",
                    "total_tasks": len(tasks),
                    "completed_tasks": 0,
                    "failed_tasks": 0,
                    "active_tasks": [],
                }
            )
        if workers <= 1 or len(tasks) <= 1:
            for index, (_label, fn) in enumerate(tasks):
                try:
                    with self._lock:
                        self._progress["active_tasks"] = [tasks[index][0]]
                    results_by_index[index] = fn()
                    with self._lock:
                        self._progress["completed_tasks"] = int(self._progress.get("completed_tasks", 0)) + 1
                except Exception as exc:
                    failures.append(f"{tasks[index][0]}: {exc}")
                    with self._lock:
                        self._progress["failed_tasks"] = int(self._progress.get("failed_tasks", 0)) + 1
                    if error_policy == "fail_fast":
                        raise
            if failures:
                manifest.setdefault("warnings", []).extend(failures)
                manifest.setdefault("partial_failures", []).extend(failures)
                manifest["scientific_status"] = "partial_failed"
                self._write_experiment_manifest(manifest)
            return [results_by_index[index] for index in sorted(results_by_index)]

        self._append(f"[PERF] Ejecutando {len(tasks)} dataset jobs con max_parallel_dataset_jobs={workers}.\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(fn): index
                for index, (_label, fn) in enumerate(tasks)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                label = tasks[index][0]
                try:
                    results_by_index[index] = future.result()
                    with self._lock:
                        self._progress["completed_tasks"] = int(self._progress.get("completed_tasks", 0)) + 1
                    self._append(f"[PROGRESS] Dataset job completado: {label}\n")
                except Exception as exc:
                    message = f"{label}: {exc}"
                    failures.append(message)
                    with self._lock:
                        self._progress["failed_tasks"] = int(self._progress.get("failed_tasks", 0)) + 1
                    self._append(f"[ERROR] Dataset job fallo: {message}\n")
                    if error_policy == "fail_fast":
                        for pending in future_to_index:
                            if pending is not future:
                                pending.cancel()
                        raise
        if failures:
            manifest.setdefault("warnings", []).extend(failures)
            manifest.setdefault("partial_failures", []).extend(failures)
            manifest["scientific_status"] = "partial_failed"
            self._write_experiment_manifest(manifest)
        return [results_by_index[index] for index in sorted(results_by_index)]

    def _run_callable_tasks(
        self,
        tasks: list[tuple[str, Any]],
        *,
        workers: int,
        error_policy: str,
        stage: str,
    ) -> tuple[list[Any], list[str]]:
        results_by_index: dict[int, Any] = {}
        failures: list[str] = []
        with self._lock:
            self._progress.update(
                {
                    "active_stage": stage,
                    "total_tasks": len(tasks),
                    "completed_tasks": 0,
                    "failed_tasks": 0,
                    "active_tasks": [],
                }
            )
        if workers <= 1 or len(tasks) <= 1:
            for index, (label, fn) in enumerate(tasks):
                try:
                    with self._lock:
                        self._progress["active_tasks"] = [label]
                    results_by_index[index] = fn()
                    with self._lock:
                        self._progress["completed_tasks"] = int(self._progress.get("completed_tasks", 0)) + 1
                except Exception as exc:
                    message = f"{label}: {exc}"
                    failures.append(message)
                    with self._lock:
                        self._progress["failed_tasks"] = int(self._progress.get("failed_tasks", 0)) + 1
                    self._append(f"[ERROR] {stage} fallo: {message}\n")
                    if error_policy == "fail_fast":
                        raise
            return [results_by_index[index] for index in sorted(results_by_index)], failures

        self._append(f"[PERF] Ejecutando {len(tasks)} tareas {stage} con workers={workers}.\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(fn): index
                for index, (_label, fn) in enumerate(tasks)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                label = tasks[index][0]
                try:
                    results_by_index[index] = future.result()
                    with self._lock:
                        self._progress["completed_tasks"] = int(self._progress.get("completed_tasks", 0)) + 1
                    self._append(f"[PROGRESS] {stage} completado: {label}\n")
                except Exception as exc:
                    message = f"{label}: {exc}"
                    failures.append(message)
                    with self._lock:
                        self._progress["failed_tasks"] = int(self._progress.get("failed_tasks", 0)) + 1
                    self._append(f"[ERROR] {stage} fallo: {message}\n")
                    if error_policy == "fail_fast":
                        for pending in future_to_index:
                            if pending is not future:
                                pending.cancel()
                        raise
        return [results_by_index[index] for index in sorted(results_by_index)], failures

    def _merge_run_timing(self, manifest: dict[str, Any], result: dict[str, Any]) -> None:
        timing = manifest.setdefault("timing", {})
        stages = timing.setdefault("stages", [])
        by_method = timing.setdefault("by_method", {})
        by_dataset = timing.setdefault("by_dataset", {})
        counters = timing.setdefault("counters", {})
        method = str(result.get("method_id") or result.get("pipeline") or "")
        dataset = str(result.get("dataset_label") or f"dataset_{result.get('dataset_size', '')}")
        elapsed = result.get("pipeline_elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            self._append(f"[TIMING] {method} {dataset} method_run={float(elapsed):.3f}s\n")
            stages.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "stage": "method_run",
                    "wall_time_seconds": float(elapsed),
                }
            )
            by_method[method] = float(by_method.get(method, 0.0)) + float(elapsed)
            by_dataset[dataset] = float(by_dataset.get(dataset, 0.0)) + float(elapsed)
        breakdown = result.get("timing_breakdown") or {}
        for stage, seconds in {
            "siesta_generation": breakdown.get("md_siesta_generation_seconds")
            or breakdown.get("atomdisp_siesta_generation_seconds"),
            "dataset_preparation": breakdown.get("dataset_preparation_seconds"),
            "test_single_points": breakdown.get("test_single_points_seconds"),
            "training_prediction": breakdown.get("training_prediction_seconds"),
            "hamiltonian_metrics": breakdown.get("evaluation_seconds"),
        }.items():
            if isinstance(seconds, (int, float)):
                stages.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "stage": stage,
                        "wall_time_seconds": float(seconds),
                    }
                )
        counts = result.get("siesta_counts") or {}
        counters["siesta_launched"] = int(counters.get("siesta_launched", 0)) + int(counts.get("launched", 0) or 0)
        counters["siesta_skipped_or_reused"] = int(counters.get("siesta_skipped_or_reused", 0)) + int(counts.get("skipped_or_reused", 0) or 0)
        counters["siesta_failed"] = int(counters.get("siesta_failed", 0)) + int(counts.get("failed", 0) or 0)
        if result.get("run_mode", manifest.get("run_mode")) != DATASET_ONLY_RUN_MODE:
            counters["graph2mat_trainings"] = int(counters.get("graph2mat_trainings", 0)) + 1
            counters["predictions"] = int(counters.get("predictions", 0)) + (1 if int(result.get("predicted_hamiltonians") or 0) > 0 else 0)

    def _set_current(
        self,
        pipeline: str,
        size: int,
        *,
        dataset_label: str | None = None,
        started_at: float | None = None,
        eta_seconds: float | None = None,
    ) -> None:
        with self._lock:
            self._current = {
                "pipeline": pipeline,
                "size": size,
                "dataset_label": dataset_label or f"dataset_{size}",
                "started_at": started_at,
                "eta_seconds": eta_seconds,
            }

    def _run(
        self,
        md_sizes: list[int],
        atom_sizes: list[int],
        run_id: str,
        fc_dataset_specs: dict[int, list[dict[str, Any]]] | None = None,
        atom_dataset_specs: list[dict[str, Any]] | None = None,
        split_ratios: dict[str, float] | None = None,
        random_seed: int | None = None,
        split_mode: str = DEFAULT_MD_SPLIT_MODE,
        test_sets: list[str] | None = None,
        primary_metric: str = DEFAULT_PRIMARY_METRIC,
        compute_budget_mode: str = "both",
        compute_accelerator: str = "cpu",
        selected_methods: list[str] | None = None,
        run_mode: str = "full_strict_pipeline",
        random_cartesian_options: dict[str, Any] | None = None,
        performance: dict[str, Any] | None = None,
        training_settings: dict[str, Any] | None = None,
        venv_activate_path: str | None = None,
        dataset_recipes_info: dict[str, Any] | None = None,
        reusable_split_policy: str = PRESERVE_ARCHIVED_SPLITS,
        training_plan: list[dict[str, Any]] | None = None,
        material: dict[str, Any] | None = None,
        hyperparameter_sweep: dict[str, Any] | None = None,
        deeph_comparison_options: dict[str, Any] | None = None,
    ) -> None:
        split_ratios = split_ratios or dict(DEFAULT_SPLIT_RATIOS)
        selected_methods = normalize_selected_methods(selected_methods)
        run_mode = parse_run_mode(run_mode)
        reusable_split_policy = parse_reusable_split_policy(reusable_split_policy)
        training_plan = parse_training_plan(training_plan)
        validate_training_plan_for_run_mode(training_plan, run_mode)
        hyperparameter_sweep = parse_hyperparameter_sweep(hyperparameter_sweep)
        validate_training_plan_sweep_sources(training_plan, hyperparameter_sweep)
        deeph_comparison_options = (
            parse_deeph_comparison_options(
                deeph_comparison_options or {},
                require_graph2mat_result=run_mode == DEEPH_COMPARISON_RUN_MODE,
            )
            if run_mode in {DEEPH_COMPARISON_RUN_MODE, GRAPH2MAT_DEEPH_RUN_MODE}
            else None
        )
        material_config = parse_material_payload(material, required=material is not None)
        material_validation = (
            validate_material_payload(material_config, required=True)
            if material_config is not None
            else None
        )
        performance_settings = parse_performance_settings(
            performance,
            compute_accelerator=compute_accelerator,
        )
        training_settings = parse_training_settings(training_settings)
        if venv_activate_path in (None, ""):
            venv_activate_path = resolve_venv_activate_from_command(DEFAULT_VENV_ACTIVATE_COMMAND)
        compute_accelerator = str(performance_settings["compute_accelerator"])
        random_cartesian_options = random_cartesian_options or {}
        dataset_recipes_info = dataset_recipes_info or {}
        pipeline_keys = pipeline_keys_for_methods(selected_methods)
        manifest = self._initial_experiment_manifest(
            run_id,
            md_sizes,
            atom_sizes,
            split_ratios,
            random_seed,
            split_mode,
            atom_dataset_specs,
            test_sets or list(DEFAULT_COMMON_TEST_SETS),
            primary_metric,
            compute_budget_mode,
            compute_accelerator,
            selected_methods,
            run_mode,
            random_cartesian_options,
            performance_settings,
            training_settings,
            dataset_recipes_info,
            material_config,
        )
        if material_validation is not None:
            manifest["material_selection"] = material_config
            manifest["material_validation"] = {
                key: value
                for key, value in material_validation.items()
                if key != "material_config"
            }
        manifest["reusable_split_policy"] = reusable_split_policy
        manifest["training_plan"] = training_plan
        manifest["hyperparameter_sweep"] = hyperparameter_sweep
        manifest["hyperparameter_sweep_expanded_count"] = (
            len(training_plan) if hyperparameter_sweep.get("enabled") else 0
        )
        experiment_root(run_id).mkdir(parents=True, exist_ok=True)
        if manifest.get("siesta_settings_warning"):
            manifest.setdefault("warnings", []).append(str(manifest["siesta_settings_warning"]))
        if manifest.get("model_config_warning"):
            manifest.setdefault("warnings", []).append(str(manifest["model_config_warning"]))
        if manifest.get("basis_pseudopotential_warning"):
            manifest.setdefault("warnings", []).append(str(manifest["basis_pseudopotential_warning"]))
        if manifest.get("reproducibility_warning"):
            manifest.setdefault("warnings", []).append(str(manifest["reproducibility_warning"]))
        self._write_experiment_manifest(manifest)
        returncode = 0
        try:
            self._append(f"[UI] Comparacion {run_id} iniciada.\n")
            self._append(f"[UI] Run mode: {run_mode}\n")
            if material_validation is not None:
                material_info = material_validation.get("material") or {}
                species = ", ".join(
                    str(item.get("label") or "")
                    for item in material_validation.get("species") or []
                    if isinstance(item, dict)
                )
                self._append(
                    "[UI] Material: "
                    f"{material_info.get('label')} ({material_info.get('material_source')}); "
                    f"species: {species or 'unknown'}.\n"
                )
                for warning in material_validation.get("warnings") or []:
                    self._append(f"[WARN] Material: {warning}\n")
            if run_mode_skips_dataset_generation(run_mode):
                self._append(f"[UI] Reusable split policy: {reusable_split_policy}\n")
            self._append(f"[UI] Selected methods: {', '.join(selected_methods)}\n")
            self._append(f"[UI] MD sizes: {md_sizes}\n")
            self._append(f"[UI] AtomDisplacement sizes: {atom_sizes}\n")
            self._append(
                "[UI] Split ratios: "
                f"{split_ratios['train']} train, {split_ratios['validation']} validation, "
                f"{split_ratios['test']} test.\n"
            )
            self._append(f"[UI] Split mode MD: {split_mode}\n")
            self._append(f"[UI] Common test sets: {', '.join(test_sets or DEFAULT_COMMON_TEST_SETS)}\n")
            self._append(
                f"[UI] Primary metric: {primary_metric}; compute mode: {compute_budget_mode}; "
                f"accelerator: {compute_accelerator}\n"
            )
            self._append(f"[PERF] Effective settings: {json.dumps(json_safe(performance_settings), sort_keys=True)}\n")
            self._append(f"[TRAIN] Effective overrides: {json.dumps(json_safe(training_settings), sort_keys=True)}\n")
            if manifest.get("siesta_settings_warning"):
                self._append(f"[WARN] {manifest['siesta_settings_warning']}\n")
            if manifest.get("model_config_warning"):
                self._append(f"[WARN] {manifest['model_config_warning']}\n")
            if (
                run_mode != DATASET_ONLY_RUN_MODE
                and STRICT_COMPARISON_MODE
                and manifest.get("siesta_settings_warning")
            ):
                raise RuntimeError(
                    "Strict comparison aborted: MD y AtomDisplacement tienen settings SIESTA distintas. "
                    "Revisa experiment_manifest.yaml: siesta_settings_mismatches."
                )
            if (
                run_mode != DATASET_ONLY_RUN_MODE
                and STRICT_COMPARISON_MODE
                and manifest.get("model_config_warning")
            ):
                raise RuntimeError(
                    "Strict comparison aborted: los metodos seleccionados tienen hiperparametros Graph2Mat distintos. "
                    "Revisa experiment_manifest.yaml: model_config_mismatches."
                )
            if atom_dataset_specs:
                self._append(
                    "[UI] AtomDisplacement FC plan: "
                    + "; ".join(
                        f"{spec['label']}: "
                        + ", ".join(
                            f"{entry['value']} -> {entry['n_structures']}"
                            for entry in spec["displacements"]
                        )
                        for spec in atom_dataset_specs
                    )
                    + "\n"
                )
            elif fc_dataset_specs:
                self._append(
                    "[UI] AtomDisplacement FC plan: "
                    + "; ".join(
                        f"dataset_{size}: "
                        + ", ".join(
                            f"{entry['value']} -> {entry['n_structures']}"
                            for entry in entries
                        )
                        for size, entries in fc_dataset_specs.items()
                    )
                    + "\n"
                )
            self._append(f"[UI] Workspaces: {WORKSPACES_ROOT / run_id}\n")
            self._append(f"[UI] Results root: {RESULTS_ROOT}\n")
            self._append(
                "[UI] ETA: el primer dataset de cada pipeline no tiene historico; "
                "los siguientes usan segundos/estructura de runs ya completados.\n"
            )
            dataset_workers = int(performance_settings.get("max_parallel_dataset_jobs") or 1)
            if compute_accelerator == "gpu" and dataset_workers > 1:
                self._append("[WARN] GPU training requested; max_parallel_dataset_jobs forced to 1 to avoid single-GPU contention.\n")
                dataset_workers = 1
            if training_plan and dataset_workers > 1:
                self._append("[UI] Training plan activo: los entrenamientos se ejecutaran secuencialmente.\n")
                dataset_workers = 1
            manifest["timing"]["counters"]["dataset_jobs"] = 0
            manifest["timing"]["counters"]["dataset_workers_used"] = dataset_workers

            dataset_tasks: list[tuple[str, Any]] = []
            used_dataset_ids: set[str] = set()
            used_dataset_labels: set[str] = set()
            for existing_specs in (
                dataset_recipes_info.get("md_dataset_specs") or [],
                dataset_recipes_info.get("atom_dataset_specs") or atom_dataset_specs or [],
                dataset_recipes_info.get("random_cartesian_dataset_specs")
                or random_cartesian_options.get("_dataset_specs")
                or [],
            ):
                for existing_spec in existing_specs:
                    register_dataset_label(existing_spec.get("label"), used_dataset_ids, used_dataset_labels)

            def fallback_dataset_label(method: str, index: int) -> str:
                label, _short_id = allocate_dataset_label(
                    method,
                    index,
                    used_ids=used_dataset_ids,
                    used_labels=used_dataset_labels,
                )
                return label

            def add_md_task(
                md_spec: dict[str, Any],
                task_training_settings: dict[str, Any],
                *,
                task_run_mode: str = run_mode,
                source_run: dict[str, Any] | None = None,
                target_id: str | None = None,
            ) -> None:
                size = int(md_spec["size"])
                label = f"md {md_spec.get('label', f'dataset_{size}')}"
                if target_id:
                    label = f"{label} [{target_id}]"

                def run_md_task() -> dict[str, Any]:
                    result = self._run_one(
                        "md",
                        size,
                        run_id,
                        dataset_label=str(md_spec.get("label") or f"dataset_{size}"),
                        recipe_metadata=md_spec.get("recipe_metadata"),
                        md_temperature_blocks=md_spec.get("temperature_blocks"),
                        split_ratios=split_ratios,
                        split_mode=split_mode,
                        run_mode=task_run_mode,
                        compute_accelerator=compute_accelerator,
                        performance=performance_settings,
                        training_settings=task_training_settings,
                        venv_activate_path=venv_activate_path,
                        reusable_split_policy=reusable_split_policy,
                        source_run=source_run,
                        material=material_config,
                    )
                    if target_id:
                        result["planned_dataset_target_id"] = target_id
                    return result

                dataset_tasks.append((
                    label,
                    run_md_task,
                ))

            def add_atom_task(
                atom_spec: dict[str, Any],
                task_training_settings: dict[str, Any],
                *,
                task_run_mode: str = run_mode,
                source_run: dict[str, Any] | None = None,
                target_id: str | None = None,
            ) -> None:
                atom_recipe_seed = (atom_spec.get("recipe_metadata") or {}).get("seed")
                atom_random_seed = atom_recipe_seed if atom_recipe_seed not in (None, "") else random_seed
                label = f"atom_displacement {atom_spec['label']}"
                if target_id:
                    label = f"{label} [{target_id}]"

                def run_atom_task() -> dict[str, Any]:
                    result = self._run_one(
                        "atom_displacement",
                        int(atom_spec["size"]),
                        run_id,
                        dataset_label=str(atom_spec["label"]),
                        fc_displacements=atom_spec.get("displacements"),
                        recipe_metadata=atom_spec.get("recipe_metadata"),
                        split_ratios=split_ratios,
                        random_seed=atom_random_seed,
                        split_mode=split_mode,
                        run_mode=task_run_mode,
                        compute_accelerator=compute_accelerator,
                        performance=performance_settings,
                        training_settings=task_training_settings,
                        venv_activate_path=venv_activate_path,
                        reusable_split_policy=reusable_split_policy,
                        source_run=source_run,
                        material=material_config,
                    )
                    if target_id:
                        result["planned_dataset_target_id"] = target_id
                    return result

                dataset_tasks.append((
                    label,
                    run_atom_task,
                ))

            def add_random_task(
                random_spec: dict[str, Any],
                task_training_settings: dict[str, Any],
                *,
                task_run_mode: str = run_mode,
                source_run: dict[str, Any] | None = None,
                target_id: str | None = None,
            ) -> None:
                random_size = int(random_spec["size"])
                size_options = {
                    **random_cartesian_options,
                    **dict(random_spec.get("options") or {}),
                    "n_structures": random_size,
                }
                size_options.pop("_dataset_specs", None)
                label = f"random_cartesian {random_spec.get('label', f'dataset_{random_size}')}"
                if target_id:
                    label = f"{label} [{target_id}]"

                def run_random_task() -> dict[str, Any]:
                    result = self._run_random_cartesian(
                        random_size,
                        run_id,
                        dataset_label=str(random_spec.get("label") or f"dataset_{random_size}"),
                        recipe_metadata=random_spec.get("recipe_metadata"),
                        split_ratios=split_ratios,
                        split_mode=split_mode,
                        random_cartesian_options=size_options,
                        run_mode=task_run_mode,
                        compute_accelerator=compute_accelerator,
                        performance=performance_settings,
                        training_settings=task_training_settings,
                        venv_activate_path=venv_activate_path,
                        reusable_split_policy=reusable_split_policy,
                        source_run=source_run,
                        material=material_config,
                    )
                    if target_id:
                        result["planned_dataset_target_id"] = target_id
                    return result

                dataset_tasks.append((
                    label,
                    run_random_task,
                ))

            md_task_specs = dataset_recipes_info.get("md_dataset_specs") or [
                {"label": fallback_dataset_label("md", index), "size": size, "recipe_metadata": None}
                for index, size in enumerate(md_sizes)
            ]
            atom_runs = dataset_recipes_info.get("atom_dataset_specs") or atom_dataset_specs or [
                {
                    "label": fallback_dataset_label("fc", index),
                    "size": size,
                    "displacements": fc_dataset_specs.get(size) if fc_dataset_specs else None,
                }
                for index, size in enumerate(atom_sizes)
            ]
            random_specs = dataset_recipes_info.get("random_cartesian_dataset_specs") or random_cartesian_options.get("_dataset_specs") or [
                {
                    "label": fallback_dataset_label("rc", index),
                    "size": random_size,
                    "options": {**random_cartesian_options, "n_structures": random_size},
                    "recipe_metadata": None,
                }
                for index, random_size in enumerate(random_cartesian_sizes_from_options(random_cartesian_options))
            ]

            def add_planned_target_task(
                target: dict[str, Any],
                task_training_settings: dict[str, Any],
                *,
                plan_item: dict[str, Any] | None,
                task_run_mode: str,
                source_run: dict[str, Any] | None = None,
            ) -> None:
                spec = dict(target["spec"])
                if plan_item is not None:
                    spec = self._spec_with_training_plan_metadata(spec, plan_item)
                pipeline_key = str(target["pipeline_key"])
                target_id = str(target.get("target_id") or "")
                if pipeline_key == "md":
                    add_md_task(
                        spec,
                        task_training_settings,
                        task_run_mode=task_run_mode,
                        source_run=source_run,
                        target_id=target_id,
                    )
                elif pipeline_key == "atom_displacement":
                    add_atom_task(
                        spec,
                        task_training_settings,
                        task_run_mode=task_run_mode,
                        source_run=source_run,
                        target_id=target_id,
                    )
                elif pipeline_key == "random_cartesian":
                    add_random_task(
                        spec,
                        task_training_settings,
                        task_run_mode=task_run_mode,
                        source_run=source_run,
                        target_id=target_id,
                    )
                else:
                    raise RuntimeError(f"Unsupported planned dataset target pipeline: {pipeline_key}")

            if training_plan and run_mode_skips_dataset_generation(run_mode):
                self._append(f"[UI] Training plan: {len(training_plan)} configuraciones.\n")
                for plan_item in training_plan:
                    plan_training_settings = parse_training_settings(plan_item.get("training_settings"))
                    reusable_info = self.dataset_recipes_info_for_reusable_dataset_ids(
                        list(plan_item.get("reusable_dataset_ids") or []),
                        selected_methods=selected_methods,
                    )
                    plan_dataset_count = sum(
                        len(reusable_info.get(key) or [])
                        for key in (
                            "md_dataset_specs",
                            "atom_dataset_specs",
                            "random_cartesian_dataset_specs",
                        )
                    )
                    self._append(
                        "[TRAIN-PLAN] "
                        f"{plan_item['label']}: {plan_dataset_count} datasets; "
                        f"overrides={json.dumps(json_safe(plan_training_settings), sort_keys=True)}\n"
                    )
                    for md_spec in (
                        reusable_info.get("md_dataset_specs") or []
                        if "md" in pipeline_keys
                        else []
                    ):
                        add_md_task(
                            self._spec_with_training_plan_metadata(md_spec, plan_item),
                            plan_training_settings,
                        )
                    for atom_spec in (
                        reusable_info.get("atom_dataset_specs") or []
                        if "atom_displacement" in pipeline_keys
                        else []
                    ):
                        add_atom_task(
                            self._spec_with_training_plan_metadata(atom_spec, plan_item),
                            plan_training_settings,
                        )
                    for random_spec in (
                        reusable_info.get("random_cartesian_dataset_specs") or []
                        if "random_cartesian" in selected_methods
                        else []
                    ):
                        add_random_task(
                            self._spec_with_training_plan_metadata(random_spec, plan_item),
                            plan_training_settings,
                        )
            elif training_plan and run_mode_uses_planned_dataset_targets(run_mode):
                planned_targets = self._planned_dataset_targets_for_specs(
                    md_specs=md_task_specs if "md" in pipeline_keys else [],
                    atom_specs=atom_runs if "atom_displacement" in pipeline_keys else [],
                    random_specs=random_specs if "random_cartesian" in selected_methods else [],
                    selected_methods=selected_methods,
                )
                planned_targets_by_id = {str(target["target_id"]): target for target in planned_targets}
                manifest["planned_dataset_targets"] = [
                    self._planned_dataset_target_public(target)
                    for target in planned_targets
                ]
                for plan_item in training_plan:
                    missing_targets = [
                        target_id
                        for target_id in self._training_plan_target_ids(plan_item)
                        if target_id not in planned_targets_by_id
                    ]
                    if missing_targets:
                        raise RuntimeError(
                            f"training_plan[{plan_item.get('index')}].dataset_targets no encontrados: {missing_targets}."
                        )

                self._append(
                    f"[UI] Training plan full strict: generando {len(planned_targets)} datasets fuente una vez.\n"
                )
                for target in planned_targets:
                    add_planned_target_task(
                        target,
                        {},
                        plan_item=None,
                        task_run_mode=DATASET_ONLY_RUN_MODE,
                    )
                source_tasks = list(dataset_tasks)
                dataset_tasks = []
                source_results = self._run_dataset_tasks(
                    source_tasks,
                    manifest=manifest,
                    workers=dataset_workers,
                    error_policy=str(performance_settings.get("error_policy", "fail_fast")),
                )
                manifest["timing"]["counters"]["dataset_jobs"] = int(
                    manifest["timing"]["counters"].get("dataset_jobs", 0)
                ) + len(source_tasks)
                source_results_by_target: dict[str, dict[str, Any]] = {}
                manifest["generated_dataset_sources"] = []
                for source_result in source_results:
                    target_id = str(source_result.get("planned_dataset_target_id") or "")
                    if not target_id:
                        raise RuntimeError("Internal error: generated dataset source missing target id.")
                    source_results_by_target[target_id] = source_result
                    source_manifest_path = Path(str(source_result.get("result_dir") or "")) / "manifest.json"
                    if source_manifest_path.exists():
                        source_manifest_path.write_text(
                            json.dumps(json_safe(source_result), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                            encoding="utf-8",
                        )
                    public_source = dict(source_result)
                    manifest["generated_dataset_sources"].append(public_source)
                    self._merge_run_timing(manifest, source_result)
                    self._write_experiment_manifest(manifest)

                required_source_targets = sorted(
                    {
                        target_id
                        for plan_item in training_plan
                        for target_id in self._training_plan_target_ids(plan_item)
                    }
                )
                missing_source_targets = [
                    target_id for target_id in required_source_targets if target_id not in source_results_by_target
                ]
                if missing_source_targets:
                    raise RuntimeError(
                        "Training plan no puede continuar: no se generaron los datasets fuente "
                        f"para targets {missing_source_targets}. Revisa los errores de la fase dataset-only."
                    )

                self._append(f"[UI] Training plan: {len(training_plan)} configuraciones.\n")
                for plan_item in training_plan:
                    plan_training_settings = parse_training_settings(plan_item.get("training_settings"))
                    target_ids = self._training_plan_target_ids(plan_item)
                    self._append(
                        "[TRAIN-PLAN] "
                        f"{plan_item['label']}: {len(target_ids)} datasets; "
                        f"overrides={json.dumps(json_safe(plan_training_settings), sort_keys=True)}\n"
                    )
                    for target_id in target_ids:
                        add_planned_target_task(
                            planned_targets_by_id[target_id],
                            plan_training_settings,
                            plan_item=plan_item,
                            task_run_mode=FULL_STRICT_RUN_MODE,
                            source_run=source_results_by_target[target_id],
                        )
            else:
                for md_spec in (md_task_specs if "md" in pipeline_keys else []):
                    add_md_task(md_spec, training_settings)

                for atom_spec in (atom_runs if "atom_displacement" in pipeline_keys else []):
                    add_atom_task(atom_spec, training_settings)

                if "random_cartesian" in selected_methods:
                    for random_spec in random_specs:
                        add_random_task(random_spec, training_settings)

            dataset_results = self._run_dataset_tasks(
                dataset_tasks,
                manifest=manifest,
                workers=dataset_workers,
                error_policy=str(performance_settings.get("error_policy", "fail_fast")),
            )
            manifest["timing"]["counters"]["dataset_jobs"] = int(
                manifest["timing"]["counters"].get("dataset_jobs", 0)
            ) + len(dataset_tasks)
            previous_by_method: dict[str, dict[str, Any]] = {}
            for result in sorted(dataset_results, key=lambda item: (str(item.get("method_id") or item.get("pipeline")), int(item.get("dataset_size") or 0), str(item.get("dataset_label") or ""))):
                method = str(result.get("method_id") or result.get("pipeline"))
                if method in {"md", "siesta_fc_cartesian", "atom_displacement"}:
                    nested_key = "atom_displacement" if method in {"siesta_fc_cartesian", "atom_displacement"} else "md"
                    self._annotate_nested_subset(result, previous_by_method.get(nested_key))
                    previous_by_method[nested_key] = result
                self._record_run_result(manifest, result)
                self._write_experiment_manifest(manifest)
            if run_mode == DATASET_ONLY_RUN_MODE:
                manifest["cross_evaluation"] = {
                    "ok": False,
                    "skipped": True,
                    "reason": DATASET_ONLY_RUN_MODE,
                    "warnings": ["dataset_only skips training, prediction, evaluation, cross-evaluation and winner analysis."],
                }
                manifest["scientific_status"] = "dataset_only"
                self._append("[UI] dataset_only: se omiten training, evaluacion cruzada y winners.\n")
            elif len(selected_methods) < 2:
                manifest["cross_evaluation"] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "single_method_selected",
                    "warnings": ["A single selected method is non-comparative; no robust winner is emitted."],
                }
                manifest["scientific_status"] = "non_comparative"
                self._append("[UI] Solo hay un metodo seleccionado; se omite winner robusto.\n")
            else:
                manifest["cross_evaluation"] = self._run_cross_evaluation(run_id, manifest)
                cross_items = manifest["cross_evaluation"].get("cross_evaluations", [])
                if isinstance(cross_items, list):
                    counters = manifest.setdefault("timing", {}).setdefault("counters", {})
                    counters["cross_evaluations"] = int(counters.get("cross_evaluations", 0)) + len(cross_items)
                    counters["predictions"] = int(counters.get("predictions", 0)) + len(cross_items)
                if manifest["cross_evaluation"].get("ok"):
                    recommendation_path = experiment_root(run_id) / "summary" / "recommendation.json"
                    if recommendation_path.exists():
                        try:
                            recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
                        except Exception:
                            recommendation = {}
                        manifest["scientific_status"] = recommendation.get(
                            "scientific_status",
                            "analysis_completed",
                        )
                    else:
                        manifest["scientific_status"] = "analysis_completed"
            if run_mode == GRAPH2MAT_DEEPH_RUN_MODE:
                if deeph_comparison_options is None:
                    deeph_comparison_options = parse_deeph_comparison_options(
                        {},
                        require_graph2mat_result=False,
                    )
                top_percent = float(deeph_comparison_options.get("graph2mat_top_percent") or 10.0)
                top_count = deeph_comparison_options.get("graph2mat_top_count")
                candidates = self._select_graph2mat_top_candidates(
                    manifest,
                    primary_metric,
                    top_percent,
                    int(top_count) if top_count not in (None, "") else None,
                )
                self._append(
                    "[DEEPh] Graph2Mat sweep terminado; "
                    f"seleccionados {len(candidates)} candidatos top para DeepH one-to-one.\n"
                )
                deeph_output_dir = (
                    Path(str(deeph_comparison_options["output_root"]))
                    / f"ui_graph2mat_deeph_{run_id}"
                )
                deeph_result = self._run_deeph_group_for_graph2mat_candidates(
                    run_id,
                    deeph_comparison_options,
                    candidates,
                    deeph_output_dir,
                )
                manifest["deeph_comparison"] = deeph_result
                with self._lock:
                    self._results.append(deeph_result)
                manifest["scientific_status"] = "graph2mat_deeph_comparison_completed"
            self._write_experiment_manifest(manifest)
            self._append("\n[UI] Comparacion experimental finalizada correctamente.\n")
        except Exception as exc:
            returncode = 1
            self._append(f"\n[ERROR] {exc}\n")
            manifest.setdefault("warnings", []).append(str(exc))
            self._write_experiment_manifest(manifest)
        finally:
            with self._lock:
                self._process = None
                self._current = None
                self._returncode = returncode
                self._finished_at = time.time()
            manifest["timing"]["total_seconds"] = (
                self._finished_at - self._started_at
                if self._finished_at is not None and self._started_at is not None
                else None
            )
            manifest["timing"]["total_experiment_seconds"] = manifest["timing"]["total_seconds"]
            self._write_experiment_manifest(manifest)
            self._write_performance_report(manifest)
            self._append("[UI] Configuraciones globales no modificadas por el experimento.\n")

    def _restore_original_config(self, key: str, original_configs: dict[str, str]) -> None:
        PIPELINES[key].config_path.write_text(original_configs[key], encoding="utf-8")

    def _ensure_not_stopped(self) -> None:
        with self._lock:
            if self._stop_requested:
                raise RuntimeError("Experimento detenido por el usuario.")

    def _annotate_nested_subset(self, result: dict[str, Any], parent: dict[str, Any] | None) -> None:
        result["parent_dataset_size"] = parent.get("dataset_size") if parent else None
        result["parent_dataset_hash"] = parent.get("dataset_sample_hash") if parent else ""
        result["nested_subset_hash"] = result.get("dataset_sample_hash", "")
        result["nested_subset_of_parent"] = True
        result["nested_subset_warning"] = ""
        if parent:
            parent_samples = set(str(item) for item in parent.get("dataset_sample_ids", []))
            current_samples = set(str(item) for item in result.get("dataset_sample_ids", []))
            if not parent_samples.issubset(current_samples):
                result["nested_subset_of_parent"] = False
                result["nested_subset_warning"] = (
                    "Dataset-size sweep is not nested: previous dataset samples "
                    "are not all present in the current dataset."
                )
        manifest_path = Path(str(result.get("result_dir", ""))) / "manifest.json"
        if manifest_path.exists():
            manifest_path.write_text(
                json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    def _method_id_for_pipeline_key(self, key: str) -> str:
        if key == "atom_displacement":
            return "siesta_fc_cartesian"
        return key

    def _configure_md_downstream_steps(self, config: dict[str, Any], original_steps: list[str]) -> None:
        config.setdefault("pipeline", {})["steps"] = [
            step for step in original_steps if step in MD_DOWNSTREAM_STEPS
        ] or list(MD_DOWNSTREAM_STEPS)

    def _configure_atom_downstream_steps(self, config: dict[str, Any]) -> None:
        config.setdefault("pipeline", {})["steps"] = list(ATOM_DOWNSTREAM_STEPS)

    def _dataset_candidate_id(self, run: dict[str, Any]) -> str:
        payload = {
            "method_id": run.get("method_id") or run.get("pipeline"),
            "dataset_label": run.get("dataset_label"),
            "dataset_size": run.get("dataset_size") or run.get("effective_dataset_size"),
            "dataset_dir": run.get("dataset_dir"),
            "result_dir": run.get("result_dir"),
            "run_id": run.get("run_id"),
        }
        return stable_payload_hash(payload, length=16)

    def _archived_dataset_run_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for group_root in (
            RESULTS_ROOT / "results_md",
            RESULTS_ROOT / "results_atomdisp",
            RESULTS_ROOT / "results_random_cartesian",
        ):
            for manifest_path in archived_result_manifest_paths(group_root):
                try:
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict):
                    result_dir = archived_manifest_result_dir(payload, manifest_path)
                    run_mode = str(payload.get("run_mode") or "")
                    dataset_dir = Path(str(payload.get("dataset_dir") or ""))
                    reusable_dataset_only = (
                        run_mode == DATASET_ONLY_RUN_MODE
                        and int(payload.get("returncode", 0) or 0) == 0
                        and dataset_dir.exists()
                    )
                    if not reusable_dataset_only and not archived_run_has_plot_metrics(result_dir):
                        continue
                    candidate = dict(payload)
                    candidate["_source_manifest_path"] = str(manifest_path)
                    candidates.append(candidate)
        return candidates

    def reusable_dataset_candidates_payload(self) -> dict[str, Any]:
        datasets: list[dict[str, Any]] = []
        seen: set[str] = set()
        for run in self._archived_dataset_run_candidates():
            candidate_id = self._dataset_candidate_id(run)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            method_id = normalize_method_id(
                run.get("method_id") or run.get("pipeline") or "",
                allow_unknown=True,
            )
            try:
                size = int(run.get("dataset_size") or run.get("effective_dataset_size") or 0)
            except (TypeError, ValueError):
                size = 0
            dataset_dir_text = str(run.get("dataset_dir") or "")
            dataset_exists = bool(dataset_dir_text) and Path(dataset_dir_text).exists()
            returncode = int(run.get("returncode", 0) or 0)
            method_spec = METHOD_REGISTRY.get(method_id)
            datasets.append(
                {
                    "id": candidate_id,
                    "method_id": method_id,
                    "method_label": method_spec.display_name if method_spec else method_id,
                    "dataset_label": str(run.get("dataset_label") or f"dataset_{size}"),
                    "dataset_size": size,
                    "training_tag": run.get("training_tag"),
                    "training_index": run.get("training_index"),
                    "training_base_dataset_label": run.get("training_base_dataset_label"),
                    "run_id": run.get("run_id"),
                    "result_dir": run.get("result_dir"),
                    "dataset_dir": dataset_dir_text,
                    "reused_from_run_id": run.get("reused_run_id"),
                    "run_mode": run.get("run_mode"),
                    "returncode": returncode,
                    "recipe_id": run.get("recipe_id"),
                    "block_id": run.get("block_id"),
                    "seed": run.get("seed"),
                    "source_manifest_path": run.get("_source_manifest_path"),
                    "mtime": self._candidate_modified_time(run),
                    "eligible": returncode == 0 and size > 0 and dataset_exists,
                    "missing_dataset": bool(dataset_dir_text) and not dataset_exists,
                }
            )
        datasets.sort(
            key=lambda item: (
                str(item.get("method_id") or ""),
                str(item.get("dataset_label") or ""),
                int(item.get("dataset_size") or 0),
                float(item.get("mtime") or 0.0),
            ),
            reverse=True,
        )
        return {
            "datasets": datasets,
            "count": len(datasets),
            "results_root": str(RESULTS_ROOT),
        }

    def dataset_recipes_info_for_reusable_dataset_ids(
        self,
        reusable_dataset_ids: list[str],
        *,
        selected_methods: list[str],
    ) -> dict[str, Any]:
        selected = set(selected_methods)
        runs_by_id = {
            self._dataset_candidate_id(run): run
            for run in self._archived_dataset_run_candidates()
        }
        missing = [item for item in reusable_dataset_ids if item not in runs_by_id]
        if missing:
            raise RuntimeError(f"Reusable dataset IDs no encontrados: {missing}. Refresca la lista de datasets.")
        recipes: dict[str, list[dict[str, Any]]] = {"md": [], "siesta_fc_cartesian": [], "random_cartesian": []}
        md_specs: list[dict[str, Any]] = []
        atom_specs: list[dict[str, Any]] = []
        random_specs: list[dict[str, Any]] = []
        used_labels_by_method: dict[str, set[str]] = defaultdict(set)
        for index, reusable_id in enumerate(reusable_dataset_ids):
            run = dict(runs_by_id[reusable_id])
            method_id = normalize_method_id(
                run.get("method_id") or run.get("pipeline") or "",
                allow_unknown=True,
            )
            if method_id not in selected:
                raise RuntimeError(
                    f"El dataset reusable {reusable_id} pertenece a {method_id}, "
                    "pero ese metodo no esta seleccionado."
                )
            try:
                size = int(run.get("dataset_size") or run.get("effective_dataset_size") or 0)
            except (TypeError, ValueError):
                size = 0
            if size <= 0:
                raise RuntimeError(f"El dataset reusable {reusable_id} no tiene dataset_size valido.")
            label = str(run.get("dataset_label") or f"dataset_{size}")
            used_labels = used_labels_by_method[method_id]
            if label in used_labels:
                label = f"{label}_{reusable_id[:8]}"
            used_labels.add(label)
            generation_parameters = {
                "reuse_source_id": reusable_id,
                "reused_dataset_dir": run.get("dataset_dir"),
                "reused_result_dir": run.get("result_dir"),
                "reused_run_id": run.get("run_id"),
            }
            metadata = {
                "method": method_id,
                "recipe_id": str(run.get("recipe_id") or f"reuse_{reusable_id}"),
                "recipe_label": str(run.get("recipe_label") or f"Reuse {label}"),
                "block_id": str(run.get("block_id") or f"reuse_{reusable_id}"),
                "block_label": str(run.get("block_label") or f"Reuse {label}"),
                "dataset_size": size,
                "dataset_label": label,
                "reuse_source_id": reusable_id,
                "reused_dataset_dir": run.get("dataset_dir"),
                "reused_result_dir": run.get("result_dir"),
                "reused_run_id": run.get("run_id"),
                "generation_parameters": generation_parameters,
                "generation_parameters_json": json.dumps(
                    json_safe(generation_parameters),
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                "seed": run.get("seed"),
            }
            block = {
                "block_id": metadata["block_id"],
                "label": metadata["block_label"],
                "n_structures": size,
                "reuse_source_id": reusable_id,
            }
            if method_id == "md":
                block["n_snapshots"] = size
                recipes["md"].append(
                    {
                        "recipe_id": metadata["recipe_id"],
                        "label": metadata["recipe_label"],
                        "blocks": [block],
                    }
                )
                md_specs.append({"label": label, "size": size, "recipe_metadata": metadata})
            elif method_id == "siesta_fc_cartesian":
                recipes["siesta_fc_cartesian"].append(
                    {
                        "recipe_id": metadata["recipe_id"],
                        "label": metadata["recipe_label"],
                        "blocks": [block],
                    }
                )
                atom_specs.append({"label": label, "size": size, "displacements": [], "recipe_metadata": metadata})
            elif method_id == "random_cartesian":
                recipes["random_cartesian"].append(
                    {
                        "recipe_id": metadata["recipe_id"],
                        "label": metadata["recipe_label"],
                        "blocks": [block],
                    }
                )
                random_specs.append(
                    {
                        "label": label,
                        "size": size,
                        "options": {"n_structures": size, "reuse_source_id": reusable_id},
                        "recipe_metadata": metadata,
                    }
                )
            else:
                raise RuntimeError(f"Metodo no soportado para dataset reusable: {method_id}")
        recipes = {key: value for key, value in recipes.items() if value}
        return {
            "recipes": recipes,
            "recipe_set_hash": recipe_set_hash(recipes),
            "md_dataset_specs": md_specs,
            "atom_dataset_specs": atom_specs,
            "random_cartesian_dataset_specs": random_specs,
        }

    def _planned_dataset_targets_for_specs(
        self,
        *,
        md_specs: list[dict[str, Any]],
        atom_specs: list[dict[str, Any]],
        random_specs: list[dict[str, Any]],
        selected_methods: list[str],
    ) -> list[dict[str, Any]]:
        selected = set(selected_methods)
        targets: list[dict[str, Any]] = []
        occurrences: dict[tuple[str, str], int] = defaultdict(int)

        def add_target(method_id: str, pipeline_key: str, spec: dict[str, Any]) -> None:
            if method_id not in selected:
                return
            metadata = dict(spec.get("recipe_metadata") or {})
            recipe_id = str(metadata.get("recipe_id") or spec.get("label") or f"{method_id}_{len(targets) + 1}")
            occurrences[(method_id, recipe_id)] += 1
            target_id = planned_dataset_target_id(method_id, recipe_id, occurrences[(method_id, recipe_id)])
            targets.append(
                {
                    "target_id": target_id,
                    "method_id": method_id,
                    "pipeline_key": pipeline_key,
                    "recipe_id": recipe_id,
                    "dataset_label": str(spec.get("label") or metadata.get("dataset_label") or recipe_id),
                    "dataset_size": int(spec.get("size") or metadata.get("dataset_size") or 0),
                    "spec": spec,
                }
            )

        for spec in md_specs:
            add_target("md", "md", spec)
        for spec in atom_specs:
            add_target("siesta_fc_cartesian", "atom_displacement", spec)
        for spec in random_specs:
            add_target("random_cartesian", "random_cartesian", spec)
        return targets

    def _planned_dataset_target_public(self, target: dict[str, Any]) -> dict[str, Any]:
        return {
            key: target.get(key)
            for key in ("target_id", "method_id", "pipeline_key", "recipe_id", "dataset_label", "dataset_size")
        }

    def _training_plan_target_ids(self, plan_item: dict[str, Any]) -> list[str]:
        ids: list[str] = []
        for target in plan_item.get("dataset_targets") or []:
            target_id = str((target or {}).get("target_id") or "").strip()
            if target_id and target_id not in ids:
                ids.append(target_id)
        return ids

    def _training_plan_dataset_label(
        self,
        dataset_label: Any,
        plan_item: dict[str, Any],
    ) -> str:
        fallback_label = f"train_config_{plan_item.get('index', 0)}"
        plan_label = plan_item.get("label") or fallback_label
        return compact_dataset_label(
            f"{dataset_label}__{plan_label}",
            {
                "dataset_label": dataset_label,
                "training_plan_index": plan_item.get("index"),
                "training_plan_label": plan_item.get("label"),
            },
            max_length=MAX_DATASET_LABEL_LENGTH,
        )

    def _spec_with_training_plan_metadata(
        self,
        spec: dict[str, Any],
        plan_item: dict[str, Any],
    ) -> dict[str, Any]:
        updated = dict(spec)
        original_label = str(updated.get("label") or f"dataset_{updated.get('size', '')}")
        metadata = dict(updated.get("recipe_metadata") or {})
        metadata.update(
            {
                "training_plan_index": plan_item.get("index"),
                "training_plan_label": plan_item.get("label"),
                "training_plan_display_label": plan_item.get("display_label"),
                "training_plan_settings": dict(plan_item.get("training_settings") or {}),
                "training_plan_reusable_dataset_ids": list(plan_item.get("reusable_dataset_ids") or []),
                "training_plan_dataset_targets": list(plan_item.get("dataset_targets") or []),
                "training_plan_source_dataset_label": original_label,
                "sweep_index": plan_item.get("sweep_index"),
                "sweep_label": plan_item.get("sweep_label"),
                "sweep_parameters": dict(plan_item.get("sweep_parameters") or {}),
                "hidden_irreps_dimension": plan_item.get("hidden_irreps_dimension"),
            }
        )
        updated["label"] = self._training_plan_dataset_label(original_label, plan_item)
        updated["recipe_metadata"] = metadata
        return updated

    def _dataset_match_score(
        self,
        run: dict[str, Any],
        *,
        key: str,
        size: int,
        dataset_label: str,
        recipe_metadata: dict[str, Any] | None,
    ) -> int | None:
        expected_method = self._method_id_for_pipeline_key(key)
        actual_method = normalize_method_id(
            run.get("method_id") or run.get("pipeline") or "",
            allow_unknown=True,
        )
        if actual_method != expected_method:
            return None
        if not run.get("dataset_dir"):
            return None
        try:
            run_size = int(run.get("dataset_size") or run.get("effective_dataset_size") or 0)
        except (TypeError, ValueError):
            return None
        if run_size != int(size):
            return None
        if int(run.get("returncode", 0) or 0) != 0:
            return None

        metadata = recipe_metadata or {}
        recipe_fields = {
            key: metadata.get(key)
            for key in ("recipe_id", "block_id", "generation_parameters_json", "seed")
            if metadata.get(key) not in (None, "")
        }
        if recipe_fields:
            for field, value in recipe_fields.items():
                if str(run.get(field) or "") != str(value):
                    return None
            return 30 + len(recipe_fields)

        if dataset_label and str(run.get("dataset_label") or "") == str(dataset_label):
            return 20
        return 10

    def _candidate_modified_time(self, run: dict[str, Any]) -> float:
        candidates = [
            Path(str(run.get("_source_manifest_path") or "")),
            Path(str(run.get("result_dir") or "")),
            Path(str(run.get("dataset_dir") or "")),
        ]
        mtimes = []
        for path in candidates:
            try:
                if path.exists():
                    mtimes.append(path.stat().st_mtime)
            except OSError:
                continue
        return max(mtimes) if mtimes else 0.0

    def _find_existing_dataset_run(
        self,
        key: str,
        size: int,
        *,
        dataset_label: str,
        recipe_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        reuse_source_id = (recipe_metadata or {}).get("reuse_source_id")
        if reuse_source_id not in (None, ""):
            expected_method = self._method_id_for_pipeline_key(key)
            for run in self._archived_dataset_run_candidates():
                if self._dataset_candidate_id(run) != str(reuse_source_id):
                    continue
                actual_method = normalize_method_id(
                    run.get("method_id") or run.get("pipeline") or "",
                    allow_unknown=True,
                )
                try:
                    run_size = int(run.get("dataset_size") or run.get("effective_dataset_size") or 0)
                except (TypeError, ValueError):
                    run_size = 0
                if actual_method != expected_method:
                    raise RuntimeError(
                        "Selected reusable dataset method mismatch: "
                        f"expected={expected_method}, got={actual_method}."
                    )
                if run_size != int(size):
                    raise RuntimeError(
                        "Selected reusable dataset size mismatch: "
                        f"expected={size}, got={run_size}."
                    )
                if int(run.get("returncode", 0) or 0) != 0:
                    raise RuntimeError(f"Selected reusable dataset failed originally: {reuse_source_id}.")
                dataset_dir = Path(str(run.get("dataset_dir") or ""))
                self._validate_existing_dataset_source(key, dataset_dir, size)
                return dict(run)
            raise RuntimeError(f"Selected reusable dataset was not found: {reuse_source_id}. Refresca la lista.")
        scored: list[tuple[int, float, dict[str, Any]]] = []
        for run in self._archived_dataset_run_candidates():
            score = self._dataset_match_score(
                run,
                key=key,
                size=size,
                dataset_label=dataset_label,
                recipe_metadata=recipe_metadata,
            )
            if score is not None:
                scored.append((score, self._candidate_modified_time(run), run))
        if not scored:
            method_id = self._method_id_for_pipeline_key(key)
            recipe_id = (recipe_metadata or {}).get("recipe_id")
            detail = f" recipe_id={recipe_id!r}" if recipe_id not in (None, "") else f" label={dataset_label!r}"
            raise RuntimeError(
                "No existing dataset found for downstream-only mode: "
                f"method={method_id}, size={size},{detail}. "
                "Run dataset_only or full_strict_pipeline first with the same dataset selection."
            )
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = dict(scored[0][2])
        dataset_dir = Path(str(selected.get("dataset_dir") or ""))
        self._validate_existing_dataset_source(key, dataset_dir, size)
        return selected

    def _manifest_path_for_split(self, dataset_dir: Path, split_name: str) -> Path:
        split_root = dataset_dir / "splits"
        for file_name in (f"{split_name}_valid_manifest.csv", f"{split_name}_manifest.csv"):
            candidate = split_root / file_name
            if candidate.exists():
                return candidate
        raise RuntimeError(f"Missing existing dataset split manifest: {split_root / (split_name + '_manifest.csv')}")

    def _read_csv_preserving_fields(self, path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
        return fieldnames, rows

    def _validate_existing_dataset_source(self, key: str, dataset_dir: Path, size: int) -> None:
        method_id = self._method_id_for_pipeline_key(key)
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise RuntimeError(f"{method_id}: existing dataset directory is missing: {dataset_dir}")
        if not (dataset_dir / "splits").exists():
            raise RuntimeError(f"{method_id}: existing dataset is missing split manifests: {dataset_dir / 'splits'}")
        if key == "md":
            basis_candidates = sorted((dataset_dir / "MD_steps" / "basis").glob("*.ion.xml"))
            sample_root = dataset_dir / "MD_steps"
        elif key == "random_cartesian":
            basis_candidates = sorted((dataset_dir / "basis").glob("*.ion.xml")) or sorted(
                (dataset_dir / "RandomCartesian_steps" / "basis").glob("*.ion.xml")
            )
            sample_root = dataset_dir / "RandomCartesian_steps"
        else:
            basis_candidates = sorted((dataset_dir / "basis").glob("*.ion.xml")) or sorted(
                (dataset_dir / "FC_steps" / "basis").glob("*.ion.xml")
            )
            sample_root = dataset_dir / "FC_steps"
        if not basis_candidates:
            raise RuntimeError(f"{method_id}: existing dataset is missing basis .ion.xml files under {dataset_dir}.")
        if key in {"md", "random_cartesian", "atom_displacement"} and not sample_root.exists():
            raise RuntimeError(f"{method_id}: existing dataset is missing sample root: {sample_root}")

        total_rows = 0
        for split_name in ("train", "validation", "test"):
            manifest_path = self._manifest_path_for_split(dataset_dir, split_name)
            _fieldnames, rows = self._read_csv_preserving_fields(manifest_path)
            if not rows:
                raise RuntimeError(f"{method_id}: split manifest is empty: {manifest_path}")
            total_rows += len(rows)
            for index, row in enumerate(rows, start=2):
                structure = Path(str(row.get("structure_path") or ""))
                sample_dir = Path(str(row.get("sample_dir") or ""))
                if not structure.exists() and sample_dir:
                    structure = sample_dir / "RUN.fdf"
                if not structure.exists():
                    raise RuntimeError(
                        f"{method_id}: missing structure in {manifest_path}:{index}: {structure}"
                    )
                hamiltonian_value = str(row.get("hamiltonian_path") or "")
                if hamiltonian_value and not Path(hamiltonian_value).exists():
                    raise RuntimeError(
                        f"{method_id}: missing reference Hamiltonian in {manifest_path}:{index}: {hamiltonian_value}"
                    )
        excluded_gap_rows = 0
        excluded_gap_manifest = dataset_dir / "splits" / "excluded_gap_manifest.csv"
        if key == "md" and excluded_gap_manifest.exists():
            _fieldnames, rows = self._read_csv_preserving_fields(excluded_gap_manifest)
            excluded_gap_rows = len(rows)
            for index, row in enumerate(rows, start=2):
                sample_dir = Path(str(row.get("sample_dir") or ""))
                structure = Path(str(row.get("structure_path") or ""))
                if not structure.exists() and sample_dir:
                    structure = sample_dir / "RUN.fdf"
                if not structure.exists():
                    raise RuntimeError(
                        f"{method_id}: missing excluded-gap structure in "
                        f"{excluded_gap_manifest}:{index}: {structure}"
                    )
        if total_rows != int(size) and total_rows + excluded_gap_rows != int(size):
            raise RuntimeError(
                f"{method_id}: existing dataset split rows ({total_rows}) "
                f"plus excluded gap rows ({excluded_gap_rows}) do not match requested size {size}."
            )

    def _rewrite_copied_manifest_value(self, value: Any, source_dataset_dir: Path, target_dataset_dir: Path) -> Any:
        if value in (None, ""):
            return value
        text = str(value)
        source = str(source_dataset_dir.resolve(strict=False))
        target = str(target_dataset_dir.resolve(strict=False))
        if text == source:
            return target
        if text.startswith(source + os.sep):
            return target + text[len(source):]
        return value

    def _rewrite_copied_dataset_manifests(self, source_dataset_dir: Path, target_dataset_dir: Path) -> None:
        split_root = target_dataset_dir / "splits"
        for valid_manifest in sorted(split_root.glob("*_valid_manifest.csv")):
            valid_manifest.unlink()
        validation_root = target_dataset_dir / "validation"
        if validation_root.exists():
            shutil.rmtree(validation_root)
        for manifest_path in sorted(split_root.glob("*_manifest.csv")):
            fieldnames, rows = self._read_csv_preserving_fields(manifest_path)
            rewritten_rows = [
                {
                    key: self._rewrite_copied_manifest_value(value, source_dataset_dir, target_dataset_dir)
                    for key, value in row.items()
                }
                for row in rows
            ]
            write_csv_dicts(manifest_path, rewritten_rows, fieldnames)

    def _remove_copied_downstream_outputs(self, dataset_dir: Path) -> None:
        for pattern in ("ML_prediction.*", "prediction_summary.json", "prediction_manifest.csv"):
            for path in sorted(dataset_dir.rglob(pattern)):
                if path.is_file():
                    path.unlink()

    def _raw_reusable_sample_root(self, key: str, dataset_dir: Path) -> Path:
        if key == "md":
            return dataset_dir / "MD_steps"
        if key == "random_cartesian":
            return dataset_dir / "RandomCartesian_steps"
        for candidate in (dataset_dir / "FC_steps", dataset_dir / "AtDis_steps"):
            if candidate.exists():
                return candidate
        return dataset_dir / "FC_steps"

    def _source_name_for_reusable_row(self, key: str, row: dict[str, str]) -> str:
        if key == "md":
            for field in ("source_frame_index", "frame_index"):
                value = str(row.get(field) or "").strip()
                if value:
                    return value
        sample_dir = str(row.get("sample_dir") or "").strip()
        if sample_dir:
            return Path(sample_dir).name
        structure_path = str(row.get("structure_path") or "").strip()
        if structure_path:
            return Path(structure_path).parent.name
        sample_id = str(row.get("sample_id") or "").strip()
        if sample_id.startswith(("md_", "atomdisp_", "random_")):
            return sample_id.split("_", 1)[1]
        return sample_id

    def _is_relative_to_path(self, path: Path, root: Path) -> bool:
        try:
            path.resolve(strict=False).relative_to(root.resolve(strict=False))
            return True
        except ValueError:
            return False

    def _reusable_split_cleanup_roots(self, key: str, dataset_dir: Path) -> list[Path]:
        roots = [dataset_dir / "splits", dataset_dir / "validation"]
        if key != "md":
            roots.extend(
                [
                    dataset_dir / "train_samples",
                    dataset_dir / "validation_samples",
                    dataset_dir / "test_samples",
                ]
            )
        return roots

    def _reusable_split_destination_root(self, key: str, dataset_dir: Path, split_name: str) -> Path:
        if key == "md":
            return dataset_dir / "splits" / split_name
        return {
            "train": dataset_dir / "train_samples",
            "validation": dataset_dir / "validation_samples",
            "test": dataset_dir / "test_samples",
        }[split_name]

    def _collect_reusable_split_items(
        self,
        key: str,
        dataset_dir: Path,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        raw_root = self._raw_reusable_sample_root(key, dataset_dir)
        fieldnames: list[str] = []
        items: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        split_names = ["train", "validation", "test"]
        if key == "md" and (dataset_dir / "splits" / "excluded_gap_manifest.csv").exists():
            split_names.append("excluded_gap")
        for split_name in split_names:
            manifest_path = self._manifest_path_for_split(dataset_dir, split_name)
            split_fieldnames, rows = self._read_csv_preserving_fields(manifest_path)
            for field in split_fieldnames:
                if field and field not in fieldnames:
                    fieldnames.append(field)
            for row in rows:
                name = self._source_name_for_reusable_row(key, row)
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                raw_source = raw_root / name
                row_source = Path(str(row.get("sample_dir") or ""))
                source = raw_source if raw_source.exists() else row_source
                if not source.exists():
                    raise RuntimeError(
                        f"{self._method_id_for_pipeline_key(key)}: no puedo reconstruir split; "
                        f"falta la muestra fuente {source}."
                    )
                items.append({"name": name, "source": source, "row": dict(row)})
        if not items:
            raise RuntimeError(f"{self._method_id_for_pipeline_key(key)}: no hay muestras para reconstruir splits.")
        return fieldnames, sorted(items, key=lambda item: sample_sort_key(Path(str(item["source"]))))

    def _stage_resplit_sources(
        self,
        items: list[dict[str, Any]],
        cleanup_roots: list[Path],
        dataset_dir: Path,
    ) -> Path | None:
        pool_dir = dataset_dir / ".reusable_split_source_pool"
        staged_any = False
        for item in items:
            source = Path(str(item["source"]))
            if not any(self._is_relative_to_path(source, root) for root in cleanup_roots if root.exists()):
                continue
            staged = pool_dir / str(item["name"])
            if staged.exists():
                shutil.rmtree(staged)
            shutil.copytree(source, staged)
            item["source"] = staged
            staged_any = True
        return pool_dir if staged_any else None

    def _copy_resplit_sample(self, source: Path, destination: Path) -> None:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        for pattern in ("ML_prediction.*", "prediction_summary.json", "prediction_manifest.csv"):
            for path in sorted(destination.glob(pattern)):
                if path.is_file():
                    path.unlink()
        normalize_siesta_matrix_name(destination)

    def _rewrite_resplit_row(
        self,
        row: dict[str, Any],
        *,
        key: str,
        dataset_dir: Path,
        split_name: str,
        destination: Path,
        split_mode: str,
        temporal_gap: int = 0,
    ) -> dict[str, Any]:
        updated = dict(row)
        structure_path = destination / "RUN.fdf"
        metadata_path = destination / "metadata.json"
        hamiltonian_path = find_reference_matrix(destination)
        run_out_path = Path(str(updated.get("run_out_path") or updated.get("output_path") or ""))
        if not run_out_path.exists():
            run_out_path = destination / "RUN.out"
        if key == "md" and not run_out_path.exists():
            run_out_path = dataset_dir / "RUN.out"
        updated["split"] = split_name
        updated["split_strategy"] = split_mode
        updated["temporal_gap"] = str(temporal_gap)
        updated.setdefault("excluded_gap_reason", "")
        updated["sample_dir"] = str(destination)
        updated["structure_path"] = str(structure_path)
        updated["metadata_path"] = str(metadata_path) if metadata_path.exists() else ""
        updated["hamiltonian_path"] = str(hamiltonian_path or "")
        updated["run_out_path"] = str(run_out_path) if run_out_path.exists() else ""
        updated["output_path"] = str(run_out_path) if run_out_path.exists() else ""
        valid = bool(structure_path.exists() and hamiltonian_path)
        updated["valid"] = "true" if valid else "false"
        updated["status"] = "completed" if valid else "incomplete"
        updated["validation_reason"] = "ok" if valid else "missing_run_fdf_or_matrix"
        if key == "md":
            name = destination.name
            updated["sample_id"] = updated.get("sample_id") or f"md_{name}"
            updated["frame_index"] = updated.get("frame_index") or name
            updated["time_index"] = updated.get("time_index") or name
            updated["source_frame_index"] = updated.get("source_frame_index") or name
            updated["source_run"] = str(dataset_dir)
        elif key == "random_cartesian":
            updated["method"] = "random_cartesian"
            updated["sample_id"] = updated.get("sample_id") or f"random_{destination.name}"
            updated["source_run"] = str(dataset_dir / "RandomCartesian_steps")
        else:
            updated["method"] = "atom_displacement"
            updated["sample_id"] = updated.get("sample_id") or f"atomdisp_{destination.name}"
        return updated

    def _split_reusable_items(
        self,
        key: str,
        items: list[dict[str, Any]],
        counts: dict[str, int],
        split_mode: str,
    ) -> dict[str, list[dict[str, Any]]]:
        if split_mode == "spread":
            return split_spread_items(items, counts)
        if key == "atom_displacement" and split_mode == "block":
            return split_grouped_exact_items(items, counts)
        return split_block_items(items, counts)

    def _rebuild_reusable_dataset_splits(
        self,
        key: str,
        dataset_dir: Path,
        size: int,
        split_ratios: dict[str, float],
        split_mode: str,
    ) -> dict[str, Any]:
        method_id = self._method_id_for_pipeline_key(key)
        fieldnames, items = self._collect_reusable_split_items(key, dataset_dir)
        # Reused archived datasets keep the temporal_gap they were generated
        # with (audit C2 decision: old datasets are documented, not rebuilt to
        # the new default); only when the rows record no gap does the current
        # scientific default apply.
        archived_gap = _archived_temporal_gap_from_items(items) if key == "md" else None
        effective_gap = (
            archived_gap
            if archived_gap is not None
            else (DEFAULT_MD_TEMPORAL_GAP if split_mode == DEFAULT_MD_SPLIT_MODE else 0)
        )
        if key == "md":
            counts, reserved_gap_frames = md_split_counts_for_mode(
                size,
                split_ratios,
                split_mode=split_mode,
                label=f"{method_id} reusable dataset",
                temporal_gap=effective_gap,
            )
        else:
            counts = validate_split_sizes(size, split_ratios, label=f"{method_id} reusable dataset")
            reserved_gap_frames = 0
        requested_items = sum(counts.values()) + reserved_gap_frames
        if requested_items > len(items):
            raise RuntimeError(
                f"{method_id}: split rebuild pide {requested_items} muestras, "
                f"pero solo hay {len(items)} disponibles."
            )
        cleanup_roots = self._reusable_split_cleanup_roots(key, dataset_dir)
        pool_dir = self._stage_resplit_sources(items, cleanup_roots, dataset_dir)
        for root in cleanup_roots:
            if root.exists():
                shutil.rmtree(root)
        excluded_gap_items: list[tuple[dict[str, Any], str]] = []
        if key == "md" and split_mode == DEFAULT_MD_SPLIT_MODE:
            split_items, excluded_gap_items = split_blocked_with_gap_items(
                items,
                counts,
                temporal_gap=effective_gap,
            )
        else:
            split_items = self._split_reusable_items(key, items, counts, split_mode)
        split_root = dataset_dir / "splits"
        split_root.mkdir(parents=True, exist_ok=True)
        for split_name, split_entries in split_items.items():
            rows: list[dict[str, Any]] = []
            destination_root = self._reusable_split_destination_root(key, dataset_dir, split_name)
            destination_root.mkdir(parents=True, exist_ok=True)
            for item in split_entries:
                destination = destination_root / str(item["name"])
                self._copy_resplit_sample(Path(str(item["source"])), destination)
                rows.append(
                    self._rewrite_resplit_row(
                        item["row"],
                        key=key,
                        dataset_dir=dataset_dir,
                        split_name=split_name,
                        destination=destination,
                        split_mode=split_mode,
                        temporal_gap=effective_gap if split_mode == DEFAULT_MD_SPLIT_MODE else 0,
                    )
                )
            manifest_path = split_root / f"{split_name}_manifest.csv"
            row_fields = list(fieldnames)
            for row in rows:
                for field in row:
                    if field not in row_fields:
                        row_fields.append(field)
            write_csv_dicts(manifest_path, rows, row_fields)
        if excluded_gap_items:
            rows = []
            for item, reason in excluded_gap_items:
                source = Path(str(item["source"]))
                row = self._rewrite_resplit_row(
                    item["row"],
                    key=key,
                    dataset_dir=dataset_dir,
                    split_name="excluded_gap",
                    destination=source,
                    split_mode=split_mode,
                    temporal_gap=effective_gap,
                )
                row["valid"] = "false"
                row["status"] = "excluded"
                row["validation_reason"] = "excluded_temporal_gap"
                row["excluded_gap_reason"] = reason
                rows.append(row)
            row_fields = list(fieldnames)
            for row in rows:
                for field in row:
                    if field not in row_fields:
                        row_fields.append(field)
            write_csv_dicts(split_root / "excluded_gap_manifest.csv", rows, row_fields)
        if pool_dir and pool_dir.exists():
            shutil.rmtree(pool_dir)
        self._remove_copied_downstream_outputs(dataset_dir)
        return {
            "reused_split_policy": REBUILD_REUSABLE_SPLITS,
            "reused_split_counts": counts,
            "reused_split_strategy": split_mode,
            "reused_temporal_gap": effective_gap if reserved_gap_frames else 0,
            "reused_excluded_gap_count": len(excluded_gap_items),
            "effective_size": sum(counts.values()),
        }

    def _copy_existing_dataset_to_workspace(
        self,
        key: str,
        source_run: dict[str, Any],
        workspace: Path,
        size: int,
    ) -> dict[str, Any]:
        source_dataset_dir = Path(str(source_run.get("dataset_dir") or ""))
        dataset_dir = workspace / "dataset"
        if dataset_dir.exists():
            raise RuntimeError(f"Refusing to overwrite existing dataset workspace: {dataset_dir}")
        shutil.copytree(source_dataset_dir, dataset_dir)
        self._rewrite_copied_dataset_manifests(source_dataset_dir, dataset_dir)
        self._remove_copied_downstream_outputs(dataset_dir)
        method_id = self._method_id_for_pipeline_key(key)
        return {
            "requested_size": size,
            "effective_size": int(source_run.get("effective_dataset_size") or source_run.get("dataset_size") or size),
            "generated_samples": source_run.get("generated_samples"),
            "completed_samples": source_run.get("completed_samples"),
            "fc_generated_samples": source_run.get("fc_generated_samples"),
            "fc_completed_samples": source_run.get("fc_completed_samples"),
            "dataset_reused": True,
            "reused_dataset_dir": str(source_dataset_dir),
            "reused_result_dir": source_run.get("result_dir"),
            "reused_run_id": source_run.get("run_id"),
            "training_base_method_id": source_run.get("training_base_method_id") or method_id,
            "training_base_dataset_label": source_run.get("training_base_dataset_label")
            or source_run.get("dataset_label")
            or f"dataset_{size}",
            "training_base_dataset_size": int(
                source_run.get("training_base_dataset_size")
                or source_run.get("effective_dataset_size")
                or source_run.get("dataset_size")
                or size
            ),
            "training_base_dataset_dir": source_run.get("training_base_dataset_dir")
            or source_run.get("reused_dataset_dir")
            or str(source_dataset_dir),
            "training_base_result_dir": source_run.get("training_base_result_dir")
            or source_run.get("reused_result_dir")
            or source_run.get("result_dir"),
            "training_base_run_id": source_run.get("training_base_run_id")
            or source_run.get("reused_run_id")
            or source_run.get("run_id"),
            "generation_seconds": None,
        }

    def _configure_existing_md_dataset(
        self,
        config: dict[str, Any],
        workspace: Path,
        size: int,
        *,
        original_steps: list[str],
    ) -> None:
        dataset_dir = workspace / "dataset"
        training_dir = workspace / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        config["paths"]["dataset_dir"] = str(dataset_dir)
        config["paths"]["training_dir"] = str(training_dir)
        config["md"]["steps"] = size
        config["training"]["data"]["basis_files"] = "../dataset/MD_steps/basis/*.ion.xml"
        config["training"]["data"]["train_runs"] = "../dataset/splits/train/*/RUN.fdf"
        config["training"]["data"]["val_runs"] = "../dataset/splits/validation/*/RUN.fdf"
        config["testing"]["test_runs"] = "../dataset/splits/test/*/RUN.fdf"
        config["prediction"]["predict_structs"] = "../dataset/splits/test/*/RUN.fdf"
        config["checkpoint"]["path"] = None
        self._configure_md_downstream_steps(config, original_steps)

    def _configure_existing_atom_dataset(
        self,
        config: dict[str, Any],
        workspace: Path,
        *,
        method_id: str,
    ) -> None:
        dataset_dir = workspace / "dataset"
        training_dir = workspace / "training"
        base_dir = workspace / "base"
        relaxed_dir = workspace / "relaxed"
        training_dir.mkdir(parents=True, exist_ok=True)
        copy_pseudopotentials(PIPELINES["atom_displacement"].root / "base", base_dir)
        copy_relaxed_basis(PIPELINES["atom_displacement"].root / "relaxed", relaxed_dir)
        config["paths"]["base_dir"] = str(base_dir)
        config["paths"]["relaxed_dir"] = str(relaxed_dir)
        config["paths"]["dataset_dir"] = str(dataset_dir)
        config["paths"]["samples_dir"] = str(dataset_dir / "train_samples")
        config["paths"]["validation_samples_dir"] = str(dataset_dir / "validation_samples")
        config["paths"]["test_samples_dir"] = str(dataset_dir / "test_samples")
        config["paths"]["collected_dir"] = str(dataset_dir / "collected")
        config["paths"]["training_dir"] = str(training_dir)
        basis_files_pattern = "../dataset/basis/*.ion.xml" if sorted((dataset_dir / "basis").glob("*.ion.xml")) else "../relaxed/*.ion.xml"
        config["training"]["data"]["basis_files"] = basis_files_pattern
        config["training"]["data"].setdefault("matrix_component_policy", "h_only")
        config["training"]["data"].setdefault("n_matrix_components", 1)
        config["training"]["data"]["val_runs"] = "../dataset/validation_samples/*/RUN.fdf"
        config["testing"]["data"]["basis_files"] = basis_files_pattern
        config["testing"]["data"].setdefault("matrix_component_policy", "h_only")
        config["testing"]["data"].setdefault("n_matrix_components", 1)
        config["testing"]["test_runs"] = "../dataset/test_samples/*/RUN.fdf"
        config["prediction"]["data"]["basis_files"] = basis_files_pattern
        config["prediction"]["data"].setdefault("matrix_component_policy", "h_only")
        config["prediction"]["data"].setdefault("n_matrix_components", 1)
        config["prediction"]["data"]["predict_structs"] = "../dataset/test_samples/*/RUN.fdf"
        config["single_points"]["rerun"] = False
        config["checkpoint"]["path"] = None
        if method_id == "random_cartesian":
            config.setdefault("structure", {}).setdefault("random_cartesian", {})["enabled"] = True
        self._configure_atom_downstream_steps(config)

    def _run_one(
        self,
        key: str,
        size: int,
        run_id: str,
        dataset_label: str | None = None,
        fc_displacements: list[dict[str, Any]] | None = None,
        recipe_metadata: dict[str, Any] | None = None,
        md_temperature_blocks: list[dict[str, Any]] | None = None,
        split_ratios: dict[str, float] | None = None,
        random_seed: int | None = None,
        split_mode: str = DEFAULT_MD_SPLIT_MODE,
        run_mode: str = "full_strict_pipeline",
        compute_accelerator: str = "cpu",
        performance: dict[str, Any] | None = None,
        training_settings: dict[str, Any] | None = None,
        venv_activate_path: str | None = None,
        reusable_split_policy: str = PRESERVE_ARCHIVED_SPLITS,
        source_run: dict[str, Any] | None = None,
        material: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = PIPELINES[key]
        if key != "md" and split_mode == DEFAULT_MD_SPLIT_MODE:
            split_mode = "block"
        dataset_label = dataset_label or f"dataset_{size}"
        started_at = time.time()
        eta_seconds = self._estimated_seconds(key, size)
        self._set_current(
            key,
            size,
            dataset_label=dataset_label,
            started_at=started_at,
            eta_seconds=eta_seconds,
        )
        self._append(f"\n[UI] === {spec.label} {dataset_label} ===\n")
        self._append(f"[UI] ETA inicial: {format_duration(eta_seconds)}\n")
        config = load_config(spec.config_path)
        apply_material_to_config(config, parse_material_payload(material, required=material is not None))
        workspace = WORKSPACES_ROOT / run_id / key / dataset_label
        if key == "md":
            result_group = "results_md"
        elif key == "random_cartesian":
            result_group = "results_random_cartesian"
        else:
            result_group = "results_atomdisp"
        result_dir = RESULTS_ROOT / result_group / dataset_label / f"run_{run_id}"
        workspace.mkdir(parents=True, exist_ok=True)
        config_snapshot_path = workspace / "pipeline_config.yaml"
        self._append(f"[UI] Workspace: {workspace}\n")
        self._append(f"[UI] Result dir previsto: {result_dir}\n")
        self._append(f"[UI] Config snapshot: {config_snapshot_path}\n")
        performance = parse_performance_settings(performance, compute_accelerator=compute_accelerator)
        apply_performance_to_config(config, performance)
        training_settings = parse_training_settings(training_settings)
        apply_training_settings_to_config(config, training_settings)
        if recipe_metadata:
            config["dataset_recipe"] = dict(recipe_metadata)
        if venv_activate_path:
            config.setdefault("paths", {})["venv_activate"] = venv_activate_path
        compute_accelerator = str(performance["compute_accelerator"])
        self._append(f"[UI] Accelerator: {compute_accelerator}\n")
        self._append(f"[PERF] {key} {dataset_label}: {json.dumps(json_safe(performance), sort_keys=True)}\n")
        self._append(f"[TRAIN] {key} {dataset_label}: {json.dumps(json_safe(training_settings), sort_keys=True)}\n")
        with self._lock:
            log_start = len(self._logs)
        prepare_metadata: dict[str, Any] = {}
        if recipe_metadata and recipe_metadata.get("seed") not in (None, ""):
            prepare_metadata["seed"] = recipe_metadata.get("seed")
        original_steps = list(config.get("pipeline", {}).get("steps", []))
        if source_run is not None or run_mode_skips_dataset_generation(run_mode):
            reusable_split_policy = parse_reusable_split_policy(reusable_split_policy)
            reuse_context = "training-plan" if source_run is not None else "downstream-only"
            if source_run is None:
                source_run = self._find_existing_dataset_run(
                    key,
                    size,
                    dataset_label=dataset_label,
                    recipe_metadata=recipe_metadata,
                )
            else:
                self._validate_existing_dataset_source(
                    key,
                    Path(str(source_run.get("dataset_dir") or "")),
                    size,
                )
            prepare_metadata.update(
                self._copy_existing_dataset_to_workspace(key, source_run, workspace, size)
            )
            prepare_metadata["reused_split_policy"] = reusable_split_policy
            if reusable_split_policy == REBUILD_REUSABLE_SPLITS:
                rebuilt = self._rebuild_reusable_dataset_splits(
                    key,
                    workspace / "dataset",
                    size,
                    split_ratios or dict(DEFAULT_SPLIT_RATIOS),
                    split_mode,
                )
                prepare_metadata.update(rebuilt)
                config["splits"] = {
                    "enabled": True,
                    "strategy": split_mode,
                    "train": rebuilt["reused_split_counts"]["train"],
                    "validation": rebuilt["reused_split_counts"]["validation"],
                    "test": rebuilt["reused_split_counts"]["test"],
                }
                if key == "md" and split_mode == DEFAULT_MD_SPLIT_MODE:
                    config["splits"]["temporal_gap"] = DEFAULT_MD_TEMPORAL_GAP
                    config["splits"]["block_order"] = DEFAULT_MD_BLOCK_ORDER
                self._append(
                    f"[UI] {reuse_context}: dataset reused; splits rebuilt from "
                    f"training controls ({split_mode}, counts={rebuilt['reused_split_counts']}).\n"
                )
            else:
                self._append(f"[UI] {reuse_context}: preserving archived dataset splits.\n")
            if key == "md":
                self._configure_existing_md_dataset(
                    config,
                    workspace,
                    size,
                    original_steps=original_steps,
                )
            else:
                self._configure_existing_atom_dataset(
                    config,
                    workspace,
                    method_id=self._method_id_for_pipeline_key(key),
                )
            write_yaml(config_snapshot_path, config)
            self._append(
                f"[UI] {reuse_context}: dataset generation skipped; "
                f"reusing {prepare_metadata['reused_dataset_dir']}.\n"
            )
            self._validate_split_manifests(key, config, size)
            write_yaml(config_snapshot_path, config)
            training_started = time.time()
            returncode = self._run_pipeline_process(
                spec,
                key=key,
                size=size,
                started_at=started_at,
                config_path=config_snapshot_path,
            )
            prepare_metadata["training_prediction_seconds"] = time.time() - training_started
        elif key == "md":
            self._prepare_md_config(
                config,
                workspace,
                size,
                split_ratios,
                split_mode=split_mode,
                temperature_blocks=md_temperature_blocks,
            )
            config.setdefault("pipeline", {})["steps"] = ["generate_md_dataset"]
            write_yaml(config_snapshot_path, config)
            self._append("[UI] Config temporal MD escrita para generar dataset y manifests.\n")
            generation_started = time.time()
            generation_returncode = self._run_pipeline_process(
                spec,
                key=key,
                size=size,
                started_at=started_at,
                config_path=config_snapshot_path,
            )
            generation_seconds = time.time() - generation_started
            prepare_metadata["generation_seconds"] = generation_seconds
            if generation_returncode != 0:
                raise RuntimeError(
                    f"{spec.label} dataset_{size} fallo generando dataset con codigo "
                    f"{generation_returncode}."
                )
            self._validate_split_manifests(key, config, size)
            if run_mode == DATASET_ONLY_RUN_MODE:
                config["pipeline"]["steps"] = []
                write_yaml(config_snapshot_path, config)
                self._append("[UI] dataset_only: MD validado; no se entrena ni predice.\n")
                returncode = 0
            else:
                self._configure_md_downstream_steps(config, original_steps)
                write_yaml(config_snapshot_path, config)
                self._append("[UI] Config temporal MD escrita para entrenar/evaluar tras validacion.\n")
                training_started = time.time()
                returncode = self._run_pipeline_process(spec, key=key, size=size, started_at=started_at, config_path=config_snapshot_path)
                prepare_metadata["training_prediction_seconds"] = time.time() - training_started
        else:
            self._prepare_atom_generation_config(
                config,
                workspace,
                size,
                fc_displacements,
                random_seed=random_seed,
            )
            write_yaml(config_snapshot_path, config)
            self._append(
                "[UI] Config temporal FC escrita; SIESTA generara AtDis_steps en el workspace.\n"
            )
            generation_started = time.time()
            generation_returncode = self._run_pipeline_process(
                spec,
                key=key,
                size=size,
                started_at=started_at,
                config_path=config_snapshot_path,
            )
            generation_seconds = time.time() - generation_started
            if generation_returncode != 0:
                raise RuntimeError(
                    f"{spec.label} dataset_{size} fallo generando FC con codigo "
                    f"{generation_returncode}."
                )
            prepare_metadata = self._prepare_atom_config(config, workspace, size, split_ratios)
            prepare_metadata["generation_seconds"] = generation_seconds
            write_yaml(config_snapshot_path, config)
            self._append("[UI] Config temporal de entrenamiento escrita tras FC.\n")
            if prepare_metadata.get("test_needs_siesta"):
                test_siesta_started = time.time()
                self._run_atom_test_single_points(spec, config, Path(prepare_metadata["test_samples_dir"]), config_snapshot_path)
                prepare_metadata["test_single_points_seconds"] = time.time() - test_siesta_started
                self._refresh_atom_split_manifests(config)
                write_yaml(config_snapshot_path, config)
                self._append("[UI] Config de entrenamiento restaurada tras SIESTA del test.\n")
            self._validate_split_manifests(key, config, size)
            if run_mode == DATASET_ONLY_RUN_MODE:
                config["pipeline"]["steps"] = []
                write_yaml(config_snapshot_path, config)
                self._append("[UI] dataset_only: SIESTA FC y splits validados; no se entrena ni predice.\n")
                returncode = 0
            else:
                training_started = time.time()
                returncode = self._run_pipeline_process(spec, key=key, size=size, started_at=started_at, config_path=config_snapshot_path)
                prepare_metadata["training_prediction_seconds"] = time.time() - training_started
        with self._lock:
            run_log = "".join(self._logs[log_start:])
        pipeline_elapsed = time.time() - started_at
        archive = self._archive_outputs(
            key,
            size,
            run_id,
            workspace,
            config,
            returncode,
            run_log,
            prepare_metadata,
            dataset_label=dataset_label,
            pipeline_elapsed_seconds=pipeline_elapsed,
            run_mode=run_mode,
            recipe_metadata=recipe_metadata,
        )
        elapsed = time.time() - started_at
        self._update_rate(key, size, elapsed, returncode)
        if returncode != 0:
            raise RuntimeError(f"{spec.label} dataset_{size} fallo con codigo {returncode}.")
        return archive

    def _run_random_cartesian(
        self,
        size: int,
        run_id: str,
        *,
        dataset_label: str | None = None,
        recipe_metadata: dict[str, Any] | None = None,
        split_ratios: dict[str, float] | None = None,
        split_mode: str = "spread",
        random_cartesian_options: dict[str, Any] | None = None,
        run_mode: str = "dataset_only",
        compute_accelerator: str = "cpu",
        performance: dict[str, Any] | None = None,
        training_settings: dict[str, Any] | None = None,
        venv_activate_path: str | None = None,
        reusable_split_policy: str = PRESERVE_ARCHIVED_SPLITS,
        source_run: dict[str, Any] | None = None,
        material: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = PIPELINES["atom_displacement"]
        if split_mode == DEFAULT_MD_SPLIT_MODE:
            split_mode = "block"
        dataset_label = dataset_label or f"dataset_{size}"
        started_at = time.time()
        self._set_current(
            "random_cartesian",
            size,
            dataset_label=dataset_label,
            started_at=started_at,
            eta_seconds=self._estimated_seconds("random_cartesian", size),
        )
        self._append(f"\n[UI] === Random Cartesian {dataset_label} ===\n")
        config = load_config(spec.config_path)
        apply_material_to_config(config, parse_material_payload(material, required=material is not None))
        performance = parse_performance_settings(performance, compute_accelerator=compute_accelerator)
        apply_performance_to_config(config, performance)
        training_settings = parse_training_settings(training_settings)
        apply_training_settings_to_config(config, training_settings)
        if recipe_metadata:
            config["dataset_recipe"] = dict(recipe_metadata)
        if venv_activate_path:
            config.setdefault("paths", {})["venv_activate"] = venv_activate_path
        compute_accelerator = str(performance["compute_accelerator"])
        self._append(f"[UI] Accelerator: {compute_accelerator}\n")
        self._append(f"[PERF] random_cartesian {dataset_label}: {json.dumps(json_safe(performance), sort_keys=True)}\n")
        self._append(f"[TRAIN] random_cartesian {dataset_label}: {json.dumps(json_safe(training_settings), sort_keys=True)}\n")
        workspace = WORKSPACES_ROOT / run_id / "random_cartesian" / dataset_label
        dataset_dir = workspace / "dataset"
        training_dir = workspace / "training"
        base_dir = workspace / "base"
        relaxed_dir = workspace / "relaxed"
        workspace.mkdir(parents=True, exist_ok=True)
        config_snapshot_path = workspace / "pipeline_config.yaml"
        random_config = config.setdefault("structure", {}).setdefault("random_cartesian", {})
        random_config.update(random_cartesian_options or {})
        random_config["enabled"] = True
        random_config["n_structures"] = int(size)
        self._append(f"[UI] Random Cartesian config: {random_config}\n")
        with self._lock:
            log_start = len(self._logs)
        if source_run is not None or run_mode_skips_dataset_generation(run_mode):
            reusable_split_policy = parse_reusable_split_policy(reusable_split_policy)
            reuse_context = "training-plan" if source_run is not None else "downstream-only"
            if source_run is None:
                source_run = self._find_existing_dataset_run(
                    "random_cartesian",
                    size,
                    dataset_label=dataset_label,
                    recipe_metadata=recipe_metadata,
                )
            else:
                self._validate_existing_dataset_source(
                    "random_cartesian",
                    Path(str(source_run.get("dataset_dir") or "")),
                    size,
                )
            prepare_metadata = self._copy_existing_dataset_to_workspace(
                "random_cartesian",
                source_run,
                workspace,
                size,
            )
            if recipe_metadata and recipe_metadata.get("seed") not in (None, ""):
                prepare_metadata["seed"] = recipe_metadata.get("seed")
            prepare_metadata["reused_split_policy"] = reusable_split_policy
            if reusable_split_policy == REBUILD_REUSABLE_SPLITS:
                rebuilt = self._rebuild_reusable_dataset_splits(
                    "random_cartesian",
                    workspace / "dataset",
                    size,
                    split_ratios or dict(DEFAULT_SPLIT_RATIOS),
                    split_mode,
                )
                prepare_metadata.update(rebuilt)
                config["splits"] = {
                    "enabled": True,
                    "strategy": split_mode,
                    "train": rebuilt["reused_split_counts"]["train"],
                    "validation": rebuilt["reused_split_counts"]["validation"],
                    "test": rebuilt["reused_split_counts"]["test"],
                }
                self._append(
                    f"[UI] {reuse_context}: Random Cartesian dataset reused; splits rebuilt "
                    f"from training controls ({split_mode}, counts={rebuilt['reused_split_counts']}).\n"
                )
            else:
                self._append(f"[UI] {reuse_context}: preserving archived Random Cartesian splits.\n")
            self._configure_existing_atom_dataset(config, workspace, method_id="random_cartesian")
            write_yaml(config_snapshot_path, config)
            self._append(
                f"[UI] {reuse_context}: Random Cartesian dataset generation skipped; "
                f"reusing {prepare_metadata['reused_dataset_dir']}.\n"
            )
            self._validate_split_manifests("atom_displacement", config, size)
            write_yaml(config_snapshot_path, config)
            training_started = time.time()
            returncode = self._run_pipeline_process(
                spec,
                key="random_cartesian",
                size=size,
                started_at=started_at,
                config_path=config_snapshot_path,
            )
            prepare_metadata["training_prediction_seconds"] = time.time() - training_started
        else:
            pseudo_count = copy_pseudopotentials(PIPELINES["atom_displacement"].root / "base", base_dir)
            relaxed_counts = copy_relaxed_basis(PIPELINES["atom_displacement"].root / "relaxed", relaxed_dir)
            ratios = split_ratios or split_ratios_from_config(config)
            validate_split_sizes(size, ratios, label=f"Random Cartesian dataset_{size}")
            config["paths"]["base_dir"] = str(base_dir)
            config["paths"]["relaxed_dir"] = str(relaxed_dir)
            config["paths"]["dataset_dir"] = str(dataset_dir)
            config["paths"]["samples_dir"] = str(dataset_dir / "RandomCartesian_steps")
            config["paths"]["collected_dir"] = str(dataset_dir / "collected")
            config["paths"]["training_dir"] = str(training_dir)
            config["single_points"]["limit"] = None
            config["single_points"]["rerun"] = False
            config["pipeline"]["steps"] = [
                "render_inputs",
                "generate_random_cartesian_dataset",
                "run_single_points",
                "collect_atom_displacement_dataset",
            ]
            write_yaml(config_snapshot_path, config)
            self._append(f"[UI] Random Cartesian dataset_dir: {dataset_dir}\n")
            self._append(f"[UI] Random Cartesian samples_dir: {config['paths']['samples_dir']}\n")
            self._append(f"[UI] Random Cartesian pseudopotenciales copiados: {pseudo_count}\n")
            self._append(
                "[UI] Random Cartesian relaxed copiado: "
                f"{relaxed_counts['basis_files']} basis .ion.xml, {relaxed_counts['xv_files']} XV.\n"
            )
            generation_returncode = self._run_pipeline_process(
                spec,
                key="random_cartesian",
                size=size,
                started_at=started_at,
                config_path=config_snapshot_path,
            )
            generation_seconds = time.time() - started_at
            if generation_returncode != 0:
                raise RuntimeError(f"Random Cartesian dataset_{size} fallo generando dataset con codigo {generation_returncode}.")
            prepare_metadata = {
                "requested_size": size,
                "effective_size": size,
                "generated_samples": size,
                "completed_samples": None,
                "seed": random_config.get("seed"),
                "generation_seconds": generation_seconds,
            }
            returncode = 0
            generation_metadata = dict(prepare_metadata)
            prepare_metadata = self._prepare_atom_config(
                config,
                workspace,
                size,
                split_ratios,
                source_samples_dir=dataset_dir / "RandomCartesian_steps",
                method_id="random_cartesian",
            )
            prepare_metadata.update(
                {
                    "generation_seconds": generation_metadata.get("generation_seconds"),
                    "generated_samples": generation_metadata.get("generated_samples"),
                    "seed": generation_metadata.get("seed"),
                }
            )
            write_yaml(config_snapshot_path, config)
            self._append("[UI] Config temporal Random Cartesian escrita para splits reutilizables.\n")
            self._validate_split_manifests("atom_displacement", config, size)
            if run_mode == DATASET_ONLY_RUN_MODE:
                config["pipeline"]["steps"] = []
                write_yaml(config_snapshot_path, config)
                self._append("[UI] dataset_only: Random Cartesian validado; no se entrena ni predice.\n")
            else:
                self._append("[UI] Config temporal Random Cartesian escrita para entrenar/evaluar.\n")
                training_started = time.time()
                returncode = self._run_pipeline_process(
                    spec,
                    key="random_cartesian",
                    size=size,
                    started_at=started_at,
                    config_path=config_snapshot_path,
                )
                prepare_metadata["training_prediction_seconds"] = time.time() - training_started
        with self._lock:
            run_log = "".join(self._logs[log_start:])
        pipeline_elapsed = time.time() - started_at
        archive = self._archive_outputs(
            "random_cartesian",
            size,
            run_id,
            workspace,
            config,
            returncode,
            run_log,
            prepare_metadata,
            dataset_label=dataset_label,
            pipeline_elapsed_seconds=pipeline_elapsed,
            run_mode=run_mode,
            recipe_metadata=recipe_metadata,
        )
        elapsed = time.time() - started_at
        self._update_rate("random_cartesian", size, elapsed, returncode)
        if returncode != 0:
            raise RuntimeError(f"Random Cartesian dataset_{size} fallo con codigo {returncode}.")
        return archive

    def _estimated_seconds(self, key: str, size: int, elapsed: float = 0.0) -> float | None:
        rate = self._rate_seconds_per_structure.get(key)
        if rate is None:
            return None
        return max(0.0, rate * size - elapsed)

    def _update_rate(self, key: str, size: int, elapsed: float, returncode: int) -> None:
        if returncode != 0 or size <= 0:
            return

        new_rate = elapsed / size
        old_rate = self._rate_seconds_per_structure.get(key)

        if old_rate is None:
            self._rate_seconds_per_structure[key] = new_rate
        else:
            self._rate_seconds_per_structure[key] = (old_rate * 0.6) + (new_rate * 0.4)

        if key in PIPELINES:
            label = PIPELINES[key].label
        elif key in METHOD_REGISTRY:
            label = METHOD_REGISTRY[key].display_name
        else:
            label = key

        self._append(
            f"[UI] ETA actualizado para {label}: "
            f"{self._rate_seconds_per_structure[key]:.2f}s/estructura.\n"
        )


    def _prepare_md_config(
        self,
        config: dict[str, Any],
        workspace: Path,
        size: int,
        split_ratios: dict[str, float] | None = None,
        split_mode: str = DEFAULT_MD_SPLIT_MODE,
        temperature_blocks: list[dict[str, Any]] | None = None,
    ) -> None:
        dataset_dir = workspace / "dataset"
        pseudo_count = copy_material_pseudopotentials(config, dataset_dir)
        ratios = split_ratios or split_ratios_from_config(config)
        counts, reserved_gap_frames = md_split_counts_for_mode(
            size,
            ratios,
            split_mode=split_mode,
            label=f"MD dataset_{size}",
        )
        config["paths"]["dataset_dir"] = str(dataset_dir)
        config["paths"]["training_dir"] = str(workspace / "training")
        config["md"]["steps"] = size
        if temperature_blocks:
            total = sum(int(block.get("n_snapshots") or 0) for block in temperature_blocks)
            if total != size:
                raise RuntimeError(
                    f"MD temperature_blocks total {total} no coincide con dataset size {size}."
                )
            config["md"]["temperature_blocks"] = list(temperature_blocks)
        else:
            config["md"]["temperature_blocks"] = []
        config["splits"] = {
            "enabled": True,
            "strategy": split_mode,
            "train": counts["train"],
            "validation": counts["validation"],
            "test": counts["test"],
        }
        if split_mode == DEFAULT_MD_SPLIT_MODE:
            config["splits"]["temporal_gap"] = DEFAULT_MD_TEMPORAL_GAP
            config["splits"]["block_order"] = DEFAULT_MD_BLOCK_ORDER
        config["training"]["data"]["train_runs"] = "../dataset/splits/train/*/RUN.fdf"
        config["training"]["data"]["val_runs"] = "../dataset/splits/validation/*/RUN.fdf"
        config["testing"]["test_runs"] = "../dataset/splits/test/*/RUN.fdf"
        config["testing"].setdefault("callbacks", {})
        config["testing"]["callbacks"]["plot_matrix_error"] = False
        config["testing"]["callbacks"]["show_plot"] = False
        config["testing"]["callbacks"]["samplewise_metrics_logger"] = False
        config["prediction"]["predict_structs"] = "../dataset/splits/test/*/RUN.fdf"
        config["checkpoint"]["path"] = None
        self._append(f"[UI] MD dataset_dir: {dataset_dir}\n")
        self._append(f"[UI] MD training_dir: {workspace / 'training'}\n")
        self._append(f"[UI] MD pseudopotenciales copiados: {pseudo_count}\n")
        self._append(f"[UI] MD steps configurados: {size}\n")
        if temperature_blocks:
            self._append(
                "[UI] MD temperature blocks: "
                + ", ".join(
                    f"{block.get('temperature_K')} K -> {block.get('n_snapshots')}"
                    for block in temperature_blocks
                )
                + "\n"
            )
        self._append(
            "[UI] MD split: "
            f"{counts['train']} train, {counts['test']} test, "
            f"{counts['validation']} validation; ratios "
            f"{ratios['train']}/{ratios['validation']}/{ratios['test']}; "
            f"strategy {split_mode}.\n"
        )
        if reserved_gap_frames:
            self._append(
                "[UI] MD temporal gap: "
                f"{DEFAULT_MD_TEMPORAL_GAP} frame entre splits; "
                f"{reserved_gap_frames} frames reservados fuera de train/validation/test.\n"
            )
        if split_mode == "spread":
            self._append(f"[WARN] {MD_SPREAD_SPLIT_WARNING}\n")
        self._append(f"[UI] MD train_runs: {config['training']['data']['train_runs']}\n")
        self._append(f"[UI] MD val_runs: {config['training']['data']['val_runs']}\n")
        self._append(f"[UI] MD test_runs: {config['testing']['test_runs']}\n")

    def _prepare_atom_generation_config(
        self,
        config: dict[str, Any],
        workspace: Path,
        size: int,
        fc_displacements: list[dict[str, Any]] | None = None,
        random_seed: int | None = None,
    ) -> None:
        dataset_dir = workspace / "dataset"
        training_dir = workspace / "training"
        base_dir = workspace / "base"
        relaxed_dir = workspace / "relaxed"
        pseudo_count = copy_pseudopotentials(PIPELINES["atom_displacement"].root / "base", base_dir)
        relaxed_counts = copy_relaxed_basis(PIPELINES["atom_displacement"].root / "relaxed", relaxed_dir)

        config["paths"]["base_dir"] = str(base_dir)
        config["paths"]["relaxed_dir"] = str(relaxed_dir)
        config["paths"]["dataset_dir"] = str(dataset_dir)
        config["paths"]["samples_dir"] = str(dataset_dir / "samples")
        config["paths"]["collected_dir"] = str(dataset_dir / "collected")
        config["paths"]["training_dir"] = str(training_dir)
        config["single_points"]["limit"] = None
        config["single_points"]["rerun"] = False
        force_constants = config["structure"]["force_constants"]
        limit = atom_fc_sample_limit(config)
        if limit is None:
            raise RuntimeError("AtomDisplacement: FC no esta habilitado en la configuracion.")
        if fc_displacements:
            requested_total = sum(int(entry["n_structures"]) for entry in fc_displacements)
            if requested_total != size:
                raise RuntimeError(
                    "AtomDisplacement: la suma de estructuras FC "
                    f"({requested_total}) no coincide con dataset_{size}."
                )
            force_constants["displacements"] = list(fc_displacements)
        else:
            displacement_entries = atom_fc_displacement_entries(config)
            counts = distribute_fc_counts(size, len(displacement_entries), limit)
            force_constants["displacements"] = [
                {**entry, "n_structures": count}
                for entry, count in zip(displacement_entries, counts)
                if count > 0
            ]
        if random_seed is not None:
            force_constants["random_seed"] = int(random_seed)
            force_constants.setdefault("subsampling", {})["seed"] = int(random_seed)
        force_constants["target_count"] = None
        config["structure"]["force_constants"]["allow_missing_matrix"] = False
        config["pipeline"]["steps"] = [
            "render_inputs",
            "generate_atom_displacement_dataset",
            "run_single_points",
            "normalize_fc_steps",
            "run_single_points",
            "collect_atom_displacement_dataset",
        ]
        self._append(f"[UI] AtomDisplacement FC dataset_dir: {dataset_dir}\n")
        self._append(f"[UI] AtomDisplacement FC base_dir: {base_dir}\n")
        self._append(f"[UI] AtomDisplacement FC relaxed_dir: {relaxed_dir}\n")
        self._append(f"[UI] AtomDisplacement training_dir: {training_dir}\n")
        self._append(f"[UI] AtomDisplacement pseudopotenciales copiados: {pseudo_count}\n")
        self._append(
            "[UI] AtomDisplacement relaxed copiado: "
            f"{relaxed_counts['basis_files']} basis .ion.xml, {relaxed_counts['xv_files']} XV.\n"
        )
        self._append(
            "[UI] AtomDisplacement generara FC raw y normalizara FC_steps antes del split.\n"
        )
        self._append(
            "[UI] AtomDisplacement FC magnitudes: "
            + ", ".join(
                f"{entry.get('value')} -> {entry.get('n_structures')}"
                for entry in force_constants["displacements"]
            )
            + f" (limite {limit} por magnitud).\n"
        )
        self._append(f"[UI] AtomDisplacement FC steps: {', '.join(config['pipeline']['steps'])}\n")

    def _prepare_atom_config(
        self,
        config: dict[str, Any],
        workspace: Path,
        size: int,
        split_ratios: dict[str, float] | None = None,
        source_samples_dir: Path | None = None,
        method_id: str = "atom_displacement",
    ) -> dict[str, Any]:
        dataset_dir = workspace / "dataset"
        training_dir = workspace / "training"
        relaxed_dir = workspace / "relaxed"
        train_samples_dir = dataset_dir / "train_samples"
        validation_samples_dir = dataset_dir / "validation_samples"
        test_samples_dir = dataset_dir / "test_samples"
        basis_dir = dataset_dir / "basis"
        source_samples_dir = source_samples_dir or dataset_dir / "FC_steps"
        if not source_samples_dir.exists():
            source_samples_dir = atom_source_samples_dir(PIPELINES["atom_displacement"], config)
        basis_count = copy_basis_files(source_samples_dir, basis_dir)
        generated_samples = generated_atom_samples(source_samples_dir)
        excluded_reference_samples = (
            []
            if method_id == "random_cartesian"
            else [path for path in generated_samples if atom_zero_reference_group_id(path) is not None]
        )
        all_samples = (
            generated_samples
            if method_id == "random_cartesian"
            else [path for path in generated_samples if atom_zero_reference_group_id(path) is None]
        )
        validated_sample_paths = validated_atom_sample_paths_from_validation(dataset_dir)
        completed_samples = [
            path
            for path in all_samples
            if is_completed_atom_sample_with_validation_csv(path, validated_sample_paths)
        ]
        if not generated_samples:
            raise RuntimeError(
                "AtomDisplacement: SIESTA FC no genero muestras normalizadas en FC_steps. "
                f"Revisa {dataset_dir / 'run_summary.json'}."
            )
        if not all_samples:
            raise RuntimeError(
                "AtomDisplacement: SIESTA FC solo genero estructuras de referencia cero, "
                "que se excluyen del benchmark. Aumenta FC.First/FC.Last o revisa "
                f"{dataset_dir / 'run_summary.json'}."
            )
        if len(all_samples) < size:
            raise RuntimeError(
                "AtomDisplacement: SIESTA FC genero menos estructuras utiles de las pedidas "
                f"tras excluir referencias cero ({len(all_samples)} < {size}; "
                f"{len(generated_samples)} generadas, {len(excluded_reference_samples)} excluidas). "
                f"Revisa FC.First/FC.Last y {dataset_dir / 'run_summary.json'}."
            )
        ratios = split_ratios or split_ratios_from_config(config)
        counts = validate_split_sizes(size, ratios, label=f"AtomDisplacement dataset_{size}")
        train_needed = counts["train"]
        validation_needed = counts["validation"]
        test_needed = counts["test"]
        selected_samples = select_spread(all_samples, size)
        split_strategy = "spread" if method_id == "random_cartesian" else "grouped_exact"
        split_samples = (
            split_spread(selected_samples, counts)
            if method_id == "random_cartesian"
            else split_grouped_exact(selected_samples, counts)
        )
        train_samples = split_samples["train"]
        validation_samples = split_samples["validation"]
        test_samples = split_samples["test"]
        incomplete_train = [
            path
            for path in train_samples
            if not is_completed_atom_sample_with_validation_csv(path, validated_sample_paths)
        ]
        if incomplete_train:
            raise RuntimeError(
                "AtomDisplacement: no puedo entrenar ese tamano sin recalcular SIESTA "
                f"en train. Para dataset_{size}, el split configurado necesita "
                f"{train_needed} muestras de train con Hamiltoniano SIESTA; faltan "
                f"{len(incomplete_train)} en el split espaciado."
            )
        if len(test_samples) < test_needed or len(validation_samples) < validation_needed:
            raise RuntimeError(
                "AtomDisplacement: no hay suficientes muestras para completar el split "
                f"configurado ({len(train_samples)} train, {len(test_samples)} test, "
                f"{len(validation_samples)} validation)."
            )
        copy_sample_dirs(train_samples, train_samples_dir)
        copy_sample_dirs(validation_samples, validation_samples_dir)
        copy_sample_dirs(test_samples, test_samples_dir)
        split_manifest_paths = write_atom_split_manifests(
            dataset_dir,
            {
                "train": train_samples,
                "validation": validation_samples,
                "test": test_samples,
            },
            method_id=method_id,
            sample_prefix="random" if method_id == "random_cartesian" else "atomdisp",
            split_strategy=split_strategy,
        )
        test_needs_siesta = any(
            not is_completed_atom_sample(path)
            for path in test_samples_dir.iterdir()
            if path.is_dir() and (path / "RUN.fdf").exists()
        )
        config["paths"]["dataset_dir"] = str(dataset_dir)
        config["paths"]["samples_dir"] = str(train_samples_dir)
        config["paths"]["validation_samples_dir"] = str(validation_samples_dir)
        config["paths"]["test_samples_dir"] = str(test_samples_dir)
        config["paths"]["collected_dir"] = str(dataset_dir / "collected")
        config["paths"]["training_dir"] = str(training_dir)
        config["paths"]["relaxed_dir"] = str(relaxed_dir)
        config["single_points"]["rerun"] = False
        basis_files_pattern = "../dataset/basis/*.ion.xml" if basis_count else "../relaxed/*.ion.xml"
        config["training"]["data"]["basis_files"] = basis_files_pattern
        config["training"]["data"].setdefault("matrix_component_policy", "h_only")
        config["training"]["data"].setdefault("n_matrix_components", 1)
        config["training"]["data"]["val_runs"] = "../dataset/validation_samples/*/RUN.fdf"
        config["prediction"]["data"]["basis_files"] = basis_files_pattern
        config["prediction"]["data"].setdefault("matrix_component_policy", "h_only")
        config["prediction"]["data"].setdefault("n_matrix_components", 1)
        config["testing"]["data"]["basis_files"] = basis_files_pattern
        config["testing"]["data"].setdefault("matrix_component_policy", "h_only")
        config["testing"]["data"].setdefault("n_matrix_components", 1)
        config["testing"]["test_runs"] = "../dataset/test_samples/*/RUN.fdf"
        config["prediction"]["data"]["predict_structs"] = "../dataset/test_samples/*/RUN.fdf"
        config["checkpoint"]["path"] = None
        config["pipeline"]["steps"] = [
            "render_inputs",
            "run_atdisp_training",
            "run_atdisp_testing",
            "run_atdisp_prediction",
        ]
        self._append(f"[UI] AtomDisplacement dataset_dir: {dataset_dir}\n")
        self._append(f"[UI] AtomDisplacement train_samples_dir: {train_samples_dir}\n")
        self._append(f"[UI] AtomDisplacement validation_samples_dir: {validation_samples_dir}\n")
        self._append(f"[UI] AtomDisplacement test_samples_dir: {test_samples_dir}\n")
        self._append(f"[UI] AtomDisplacement training_dir: {training_dir}\n")
        self._append(f"[UI] AtomDisplacement relaxed_dir: {relaxed_dir}\n")
        self._append(f"[UI] AtomDisplacement source_samples_dir: {source_samples_dir}\n")
        self._append(f"[UI] AtomDisplacement basis dataset copiados: {basis_count}\n")
        self._append(
            f"[UI] AtomDisplacement FC raw disponibles: {len(generated_samples)} generadas, "
            f"{len(excluded_reference_samples)} referencias cero excluidas, "
            f"{len(all_samples)} utiles, "
            f"{len(completed_samples)} con referencia SIESTA; "
            f"seleccionadas para dataset_{size}: {len(selected_samples)}.\n"
        )
        self._append(
            "[UI] AtomDisplacement split: "
            f"{len(train_samples)} train, {len(test_samples)} test, "
            f"{len(validation_samples)} validation; ratios "
            f"{ratios['train']}/{ratios['validation']}/{ratios['test']}.\n"
        )
        self._append(f"[UI] AtomDisplacement train samples: {sample_names(train_samples)}\n")
        self._append(f"[UI] AtomDisplacement test samples: {sample_names(test_samples)}\n")
        self._append(f"[UI] AtomDisplacement validation samples: {sample_names(validation_samples)}\n")
        if test_needs_siesta:
            self._append("[UI] AtomDisplacement ejecutara SIESTA solo en el split de test.\n")
        else:
            self._append("[UI] AtomDisplacement reutiliza referencias SIESTA ya existentes para test.\n")
        self._append(f"[UI] AtomDisplacement test_runs: {config['testing']['test_runs']}\n")
        self._append(f"[UI] AtomDisplacement steps: {', '.join(config['pipeline']['steps'])}\n")
        return {
            "test_samples_dir": str(test_samples_dir),
            "test_needs_siesta": test_needs_siesta,
            "requested_size": size,
            "effective_size": size,
            "generated_samples": len(selected_samples),
            "completed_samples": sum(
                1
                for path in selected_samples
                if is_completed_atom_sample_with_validation_csv(path, validated_sample_paths)
            ),
            "fc_generated_samples": len(generated_samples),
            "fc_excluded_reference_samples": len(excluded_reference_samples),
            "fc_usable_samples": len(all_samples),
            "fc_completed_samples": len(completed_samples),
            "split_manifest_paths": {key: str(value) for key, value in split_manifest_paths.items()},
            "seed": (config.get("structure", {}).get("force_constants", {}) or {}).get("random_seed"),
        }

    def _run_pipeline_process(
        self,
        spec: PipelineSpec,
        *,
        key: str,
        size: int,
        started_at: float,
        config_path: Path | None = None,
    ) -> int:
        config_path = config_path or spec.config_path
        config = load_config(config_path)
        shell = str(config.get("commands", {}).get("shell", "bash"))
        python = str(config.get("commands", {}).get("python", "python"))
        venv_activate = resolve_pipeline_path(spec, config["paths"]["venv_activate"])
        if not venv_activate.exists():
            raise RuntimeError(f"{spec.label}: no se encontro el entorno virtual: {venv_activate}")
        python_executable = venv_activate.parent / "python"
        if not python_executable.exists():
            python_executable = Path(sys.executable)
        accelerator = str(config.get("training", {}).get("trainer", {}).get("accelerator", "cpu"))
        if accelerator in {"gpu", "auto"}:
            has_cuda = cuda_available(python_executable)
            if accelerator == "gpu" and not has_cuda:
                raise RuntimeError(f"{spec.label}: accelerator=gpu solicitado pero CUDA no esta disponible.")
            if accelerator == "auto":
                resolved = "gpu" if has_cuda else "cpu"
                config.setdefault("training", {}).setdefault("trainer", {})["accelerator"] = resolved
                config.setdefault("performance", {})["compute_accelerator"] = resolved
                write_yaml(config_path, config)
                message = "GPU disponible; usando accelerator=gpu." if has_cuda else "CUDA no disponible; auto usa accelerator=cpu."
                self._append(f"[PERF] {spec.label}: {message}\n")
        shell_command = (
            f"source {shlex.quote(str(venv_activate))} "
            f"&& {shlex.quote(python)} {shlex.quote(str(spec.main_script))}"
        )
        command = [shell, "-lc", shell_command]
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PIPELINE_CONFIG_PATH": str(config_path),
            **performance_env(config.get("performance", {})),
        }
        self._append(f"[RUN] {' '.join(command)}\n")
        master_fd: int | None = None
        if pty is not None:
            master_fd, slave_fd = pty.openpty()
            process = subprocess.Popen(
                command,
                cwd=spec.root,
                stdin=subprocess.DEVNULL,
                stdout=slave_fd,
                stderr=slave_fd,
                env=env,
            )
            os.close(slave_fd)
        else:
            process = subprocess.Popen(
                command,
                cwd=spec.root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        with self._lock:
            self._process = process
            self._processes.add(process)
        self._append(f"[UI] PID: {process.pid}\n")
        self._append(f"[UI] CWD: {spec.root}\n")
        self._append(f"[UI] ETA al arrancar proceso: {format_duration(self._estimated_seconds(key, size))}\n")
        try:
            returncode = stream_process_output(
                process,
                self._append,
                label=spec.label,
                master_fd=master_fd,
                eta_provider=lambda: self._estimated_seconds(key, size, time.time() - started_at),
                progress_provider=lambda: lightning_training_progress_from_config(config_path),
            )
        finally:
            if master_fd is not None:
                os.close(master_fd)
        with self._lock:
            if self._process is process:
                self._process = None
            self._processes.discard(process)
        elapsed = time.time() - started_at
        self._append(
            f"[UI] {spec.label} finalizo con codigo {returncode} "
            f"en {format_duration(elapsed)}.\n"
        )
        return returncode

    def _run_atom_test_single_points(
        self,
        spec: PipelineSpec,
        config: dict[str, Any],
        test_samples_dir: Path,
        config_path: Path,
    ) -> None:
        self._append(f"[UI] Preparando SIESTA solo para test: {test_samples_dir}\n")
        original_samples_dir = config["paths"]["samples_dir"]
        original_limit = config["single_points"].get("limit")
        original_steps = list(config["pipeline"]["steps"])
        config["paths"]["samples_dir"] = str(test_samples_dir)
        config["single_points"]["limit"] = None
        config["pipeline"]["steps"] = ["run_single_points"]
        write_yaml(config_path, config)
        returncode = self._run_pipeline_process(
            spec,
            key=spec.key,
            size=max(
                1,
                len(
                    [
                        path
                        for path in test_samples_dir.iterdir()
                        if path.is_dir() and (path / "RUN.fdf").exists()
                    ]
                ),
            ),
            started_at=time.time(),
            config_path=config_path,
        )
        config["paths"]["samples_dir"] = original_samples_dir
        config["single_points"]["limit"] = original_limit
        config["pipeline"]["steps"] = original_steps
        if returncode != 0:
            raise RuntimeError(f"{spec.label}: fallo SIESTA del split de test con codigo {returncode}.")

    def _refresh_atom_split_manifests(self, config: dict[str, Any]) -> None:
        dataset_dir = Path(config["paths"]["dataset_dir"])
        split_samples = {
            "train": sorted(path for path in (dataset_dir / "train_samples").iterdir() if path.is_dir()),
            "validation": sorted(path for path in (dataset_dir / "validation_samples").iterdir() if path.is_dir()),
            "test": sorted(path for path in (dataset_dir / "test_samples").iterdir() if path.is_dir()),
        }
        write_atom_split_manifests(dataset_dir, split_samples)
        self._append("[UI] AtomDisplacement split manifests refrescados tras SIESTA de test.\n")

    def _read_csv_raw(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _write_enriched_valid_manifest(
        self,
        *,
        source_manifest: Path,
        validator_valid_manifest: Path,
        output_manifest: Path,
    ) -> None:
        """Write a training-compatible valid manifest.

        validate_sample_bundle.py writes validator-oriented columns. The training
        script expects the original split-manifest schema, especially
        structure_path/sample_dir/status/valid. Therefore, keep rows from the
        original manifest that passed validation, and normalize common aliases.
        """
        source_rows = self._read_csv_raw(source_manifest)
        validator_rows = self._read_csv_raw(validator_valid_manifest)

        valid_sample_ids = {
            str(row.get("sample_id") or "").strip()
            for row in validator_rows
            if str(row.get("sample_id") or "").strip()
        }
        valid_sample_dirs = {
            str(row.get("sample_dir") or "").strip()
            for row in validator_rows
            if str(row.get("sample_dir") or "").strip()
        }
        valid_basenames = {
            Path(item).name
            for item in valid_sample_dirs
            if item
        }

        enriched_rows: list[dict[str, Any]] = []
        for row in source_rows:
            sample_id = str(row.get("sample_id") or "").strip()
            sample_dir = str(row.get("sample_dir") or "").strip()
            sample_basename = Path(sample_dir).name if sample_dir else ""

            passed = (
                sample_id in valid_sample_ids
                or sample_dir in valid_sample_dirs
                or sample_basename in valid_basenames
            )
            if not passed:
                continue

            row = dict(row)

            # Normalize aliases expected by downstream scripts.
            if not row.get("structure_path") and row.get("run_fdf"):
                row["structure_path"] = row["run_fdf"]
            if not row.get("run_fdf") and row.get("structure_path"):
                row["run_fdf"] = row["structure_path"]

            if not row.get("run_out_path") and row.get("run_out"):
                row["run_out_path"] = row["run_out"]
            if not row.get("output_path") and row.get("run_out_path"):
                row["output_path"] = row["run_out_path"]

            if not row.get("sample_dir") and row.get("structure_path"):
                row["sample_dir"] = str(Path(row["structure_path"]).parent)

            row["valid"] = "true"
            row["status"] = "valid"
            row["validation_reason"] = "ok"

            enriched_rows.append(row)

        if not enriched_rows:
            debug = {
                "source_manifest": str(source_manifest),
                "validator_valid_manifest": str(validator_valid_manifest),
                "source_rows": len(source_rows),
                "validator_rows": len(validator_rows),
                "valid_sample_ids": sorted(valid_sample_ids)[:10],
                "valid_basenames": sorted(valid_basenames)[:10],
                "first_source_row": source_rows[0] if source_rows else {},
                "first_validator_row": validator_rows[0] if validator_rows else {},
            }
            raise RuntimeError(
                "No se pudo construir un valid manifest compatible con training: "
                + json.dumps(debug, ensure_ascii=False)
            )

        write_csv_dicts(output_manifest, enriched_rows, SPLIT_MANIFEST_FIELDS)

    def _validate_split_manifests(
        self,
        key: str,
        config: dict[str, Any],
        size: int,
    ) -> dict[str, Any]:
        dataset_dir = Path(config["paths"]["dataset_dir"])
        split_root = dataset_dir / "splits"
        script = COMPARISON_ROOT / "scripts" / "validate_sample_bundle.py"
        summaries: dict[str, Any] = {}
        for split_name in ("train", "validation", "test"):
            manifest_path = split_root / f"{split_name}_manifest.csv"
            if not manifest_path.exists():
                raise RuntimeError(f"{PIPELINES[key].label}: falta split manifest: {manifest_path}")
            rows = read_csv_rows(manifest_path)
            min_valid = len(rows)
            output_dir = dataset_dir / "validation" / split_name
            command = [
                sys.executable,
                str(script),
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
                "--min-valid",
                str(min_valid),
            ]
            self._append(f"[UI] Validando {PIPELINES[key].label} {split_name}: {' '.join(command)}\n")
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                self._append(result.stdout.strip() + "\n")
            if result.stderr.strip():
                self._append(result.stderr.strip() + "\n")
            summary_path = output_dir / "validation_summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    summary = {"ok": False, "error": str(exc)}
            else:
                summary = {"ok": False, "error": "validation_summary_missing"}
            summary["returncode"] = result.returncode
            summaries[split_name] = summary
            valid_manifest = output_dir / "valid_samples.csv"
            if valid_manifest.exists():
                self._write_enriched_valid_manifest(
                    source_manifest=manifest_path,
                    validator_valid_manifest=valid_manifest,
                    output_manifest=split_root / f"{split_name}_valid_manifest.csv",
                )
            if result.returncode != 0 or not summary.get("ok"):
                raise RuntimeError(
                    f"{PIPELINES[key].label}: validacion {split_name} fallo; "
                    f"revisa {summary_path}."
                )
        self._append(f"[UI] Validacion de muestras completada para {PIPELINES[key].label} dataset_{size}.\n")
        return summaries

    def _single_point_counts(self, summary_path: Path) -> dict[str, int]:
        counts = {
            "launched": 0,
            "skipped_or_reused": 0,
            "failed": 0,
            "valid": 0,
            "total": 0,
        }
        if not summary_path.exists():
            return counts
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return counts
        samples = payload if isinstance(payload, list) else payload.get("samples", [])
        if not isinstance(samples, list):
            return counts
        for row in samples:
            if not isinstance(row, dict):
                continue
            counts["total"] += 1
            status = str(row.get("status") or "")
            if status == "skipped_validated":
                counts["skipped_or_reused"] += 1
            elif status == "failed":
                counts["failed"] += 1
                counts["launched"] += 1
            elif status:
                counts["launched"] += 1
            if row.get("valid") is True:
                counts["valid"] += 1
        return counts

    def _identity_path(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return str(Path(text).resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            return text

    def _training_base_metadata(
        self,
        key: str,
        dataset_label: str,
        size: int,
        dataset_dir: Path,
        result_dir: Path,
        run_id: str,
        prepare_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        method_id = self._method_id_for_pipeline_key(key)
        dataset_reused = bool(prepare_metadata.get("dataset_reused"))
        base_dataset_dir = (
            prepare_metadata.get("training_base_dataset_dir")
            or prepare_metadata.get("reused_dataset_dir")
            or str(dataset_dir)
        )
        base_result_dir = (
            prepare_metadata.get("training_base_result_dir")
            or prepare_metadata.get("reused_result_dir")
            or (str(result_dir) if not dataset_reused else None)
        )
        base_run_id = (
            prepare_metadata.get("training_base_run_id")
            or prepare_metadata.get("reused_run_id")
            or (run_id if not dataset_reused else None)
        )
        return {
            "method_id": str(prepare_metadata.get("training_base_method_id") or method_id),
            "dataset_label": str(
                prepare_metadata.get("training_base_dataset_label")
                or dataset_label
                or f"dataset_{size}"
            ),
            "dataset_size": int(prepare_metadata.get("training_base_dataset_size") or size),
            "dataset_dir": self._identity_path(base_dataset_dir),
            "result_dir": self._identity_path(base_result_dir),
            "run_id": str(base_run_id or ""),
        }

    def _manifest_training_base_metadata(
        self,
        manifest: dict[str, Any],
        manifest_path: Path,
        fallback_method_id: str,
    ) -> dict[str, Any]:
        dataset_reused = bool(manifest.get("dataset_reused"))
        result_dir = archived_manifest_result_dir(manifest, manifest_path)
        dataset_size = manifest.get("training_base_dataset_size") or manifest.get(
            "effective_dataset_size",
            manifest.get("dataset_size", 0),
        )
        try:
            dataset_size = int(dataset_size or 0)
        except (TypeError, ValueError):
            dataset_size = 0
        return {
            "method_id": str(
                manifest.get("training_base_method_id")
                or manifest.get("method_id")
                or fallback_method_id
            ),
            "dataset_label": str(
                manifest.get("training_base_dataset_label")
                or manifest.get("dataset_label")
                or f"dataset_{dataset_size}"
            ),
            "dataset_size": dataset_size,
            "dataset_dir": self._identity_path(
                manifest.get("training_base_dataset_dir")
                or (manifest.get("reused_dataset_dir") if dataset_reused else None)
                or manifest.get("dataset_dir")
            ),
            "result_dir": self._identity_path(
                manifest.get("training_base_result_dir")
                or (manifest.get("reused_result_dir") if dataset_reused else None)
                or result_dir
            ),
            "run_id": str(
                manifest.get("training_base_run_id")
                or (manifest.get("reused_run_id") if dataset_reused else None)
                or manifest.get("run_id")
                or ""
            ),
        }

    def _same_training_base(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        if str(left.get("method_id") or "") != str(right.get("method_id") or ""):
            return False
        for field in ("result_dir", "dataset_dir"):
            left_value = str(left.get(field) or "")
            right_value = str(right.get(field) or "")
            if left_value and right_value and left_value == right_value:
                return True
        left_run_id = str(left.get("run_id") or "")
        if left_run_id and left_run_id == str(right.get("run_id") or ""):
            return True
        has_identity = any(
            str(left.get(field) or "") or str(right.get(field) or "")
            for field in ("result_dir", "dataset_dir", "run_id")
        )
        if has_identity:
            return False
        return (
            str(left.get("dataset_label") or "") == str(right.get("dataset_label") or "")
            and int(left.get("dataset_size") or 0) == int(right.get("dataset_size") or 0)
        )

    def _next_training_index(
        self,
        result_group: str,
        current_base: dict[str, Any],
        *,
        current_result_dir: Path,
    ) -> int:
        root = RESULTS_ROOT / result_group
        max_index = 0
        legacy_matches = 0
        for manifest_path in archived_result_manifest_paths(root):
            if self._identity_path(manifest_path.parent) == self._identity_path(current_result_dir):
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("run_mode") == DATASET_ONLY_RUN_MODE:
                continue
            if int(manifest.get("returncode", 0) or 0) != 0:
                continue
            candidate_base = self._manifest_training_base_metadata(
                manifest,
                manifest_path,
                str(current_base.get("method_id") or ""),
            )
            if not self._same_training_base(candidate_base, current_base):
                continue
            try:
                index = int(manifest.get("training_index") or 0)
            except (TypeError, ValueError):
                index = 0
            if index > 0:
                max_index = max(max_index, index)
            else:
                legacy_matches += 1
        return max_index + 1 if max_index else legacy_matches + 1

    def _training_tag(self, base_label: str, training_index: int) -> str:
        clean_label = compact_dataset_label(base_label, {"training_index": training_index}, max_length=80)
        return f"{clean_label}_train{training_index}"

    def _archive_outputs(
        self,
        key: str,
        size: int,
        run_id: str,
        workspace: Path,
        config: dict[str, Any],
        returncode: int,
        run_log: str,
        prepare_metadata: dict[str, Any] | None = None,
        dataset_label: str | None = None,
        pipeline_elapsed_seconds: float | None = None,
        run_mode: str = "full_strict_pipeline",
        recipe_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prepare_metadata = prepare_metadata or {}
        dataset_label = dataset_label or f"dataset_{size}"
        recipe_metadata = recipe_metadata or config.get("dataset_recipe") or {}
        if key == "md":
            result_group = "results_md"
        elif key == "random_cartesian":
            result_group = "results_random_cartesian"
        else:
            result_group = "results_atomdisp"
        result_dir = RESULTS_ROOT / result_group / dataset_label / f"run_{run_id}"
        result_dir.mkdir(parents=True, exist_ok=True)
        self._append(f"[UI] Archivando salidas en {result_dir}\n")
        write_yaml(result_dir / "pipeline_config.yaml", config)
        (result_dir / "run.log").write_text(run_log, encoding="utf-8")
        self._append(f"[UI] run.log guardado: {result_dir / 'run.log'}\n")

        dataset_dir = Path(config["paths"]["dataset_dir"])
        training_dir = Path(config["paths"]["training_dir"])
        prediction_count = 0
        reference_count = 0
        if key == "md":
            md_outputs_root = dataset_dir / "splits" / "test"
            if not md_outputs_root.exists():
                md_outputs_root = dataset_dir / "MD_steps"
            copy_basis_files(dataset_dir / "MD_steps", result_dir / "basis")
            copy_matching_files(
                md_outputs_root,
                "*/RUN.fdf",
                result_dir / "structures",
            )
            copy_matching_files(
                md_outputs_root,
                "*/metadata.json",
                result_dir / "structures",
            )
            prediction_count = copy_matching_files(
                md_outputs_root,
                "*/ML_prediction.HSX",
                result_dir / "predicted_hamiltonians",
            )
            reference_count += copy_selected_reference_files(
                md_outputs_root,
                result_dir / "siesta_hamiltonians",
            )
        else:
            samples_dir = Path(config["paths"].get("test_samples_dir", config["paths"]["samples_dir"]))
            copy_basis_files(samples_dir, result_dir / "basis")
            copy_matching_files(
                samples_dir,
                "*/RUN.fdf",
                result_dir / "structures",
            )
            copy_matching_files(
                samples_dir,
                "*/metadata.json",
                result_dir / "structures",
            )
            prediction_count = copy_matching_files(
                samples_dir,
                "*/ML_prediction.HSX",
                result_dir / "predicted_hamiltonians",
            )
            reference_count += copy_selected_reference_files(
                samples_dir,
                result_dir / "siesta_hamiltonians",
            )

        copy_if_exists(training_dir / "sample_metrics.csv", result_dir / "sample_metrics.csv")
        copy_if_exists(dataset_dir / "run_summary.json", result_dir / "run_summary.json")
        copy_if_exists(dataset_dir / "samples_manifest.json", result_dir / "samples_manifest.json")
        copy_if_exists(dataset_dir / "collected" / "water_atom_displacement_dataset.json", result_dir / "water_atom_displacement_dataset.json")
        copy_if_exists(dataset_dir / "collected" / "water_atom_displacement_summary.csv", result_dir / "water_atom_displacement_summary.csv")
        for manifest_file in sorted((dataset_dir / "splits").glob("*_manifest.csv")):
            copy_if_exists(manifest_file, result_dir / "splits" / manifest_file.name)
        copy_if_exists(dataset_dir / "splits" / "split_summary.json", result_dir / "splits" / "split_summary.json")
        for validation_file in sorted((dataset_dir / "validation").glob("*/*.csv")):
            copy_if_exists(validation_file, result_dir / "validation" / validation_file.parent.name / validation_file.name)
        siesta_counts = self._single_point_counts(result_dir / "run_summary.json")
        if prepare_metadata.get("dataset_reused"):
            total_reused = int(siesta_counts.get("total") or prepare_metadata.get("effective_size") or size)
            siesta_counts = {
                "launched": 0,
                "skipped_or_reused": total_reused,
                "failed": 0,
                "valid": int(siesta_counts.get("valid") or total_reused),
                "total": total_reused,
            }
        if run_mode == DATASET_ONLY_RUN_MODE:
            evaluation_metrics = {
                "skipped": True,
                "reason": DATASET_ONLY_RUN_MODE,
                "evaluation_time_seconds": None,
            }
        else:
            evaluation_metrics = self._evaluate_hamiltonian_metrics(key, config, result_dir)
        timing_breakdown = {
            "md_siesta_generation_seconds": None,
            "atomdisp_siesta_generation_seconds": None,
            "dataset_preparation_seconds": None,
            "normalization_seconds": None,
            "training_seconds": None,
            "prediction_seconds": None,
            "evaluation_seconds": evaluation_metrics.get("evaluation_time_seconds"),
            "winner_analysis_seconds": None,
            "total_experiment_seconds": pipeline_elapsed_seconds,
            "timing_incomplete_warning": (
                "This per-run manifest contains measured total time and Hamiltonian evaluation time; "
                "legacy training/testing scripts do not expose separate training and prediction timings."
            ),
        }
        if key == "md":
            timing_breakdown["md_siesta_generation_seconds"] = prepare_metadata.get("generation_seconds")
        else:
            timing_breakdown["atomdisp_siesta_generation_seconds"] = prepare_metadata.get("generation_seconds")
            timing_breakdown["dataset_preparation_seconds"] = prepare_metadata.get("dataset_preparation_seconds")
            timing_breakdown["test_single_points_seconds"] = prepare_metadata.get("test_single_points_seconds")
        timing_breakdown["training_prediction_seconds"] = prepare_metadata.get("training_prediction_seconds")
        (result_dir / "timing_breakdown.json").write_text(
            json.dumps(json_safe(timing_breakdown), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        effective_size = int(prepare_metadata.get("effective_size", size))
        if run_mode == DATASET_ONLY_RUN_MODE:
            checkpoint_path = None
            signed_checkpoint = {"path": None, "sha256": None, "relative_path": None, "selection": None}
            checkpoint_warning = "Checkpoint not produced in dataset_only mode."
            checkpoint_manifest_path = training_dir / "checkpoint_manifest.json"
        else:
            training_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = find_latest_checkpoint(training_dir, config)
            signed_checkpoint = checkpoint_metadata(checkpoint_path, training_dir)
            checkpoint_warning = checkpoint_selection_warning(training_dir, checkpoint_path)
            checkpoint_manifest_path = write_checkpoint_manifest(training_dir, signed_checkpoint, checkpoint_warning)
        source_pseudos = (
            sorted((PIPELINES["md"].root / "dataset").glob("*.psf"))
            if key == "md"
            else sorted((PIPELINES["atom_displacement"].root / "base").glob("*.psf"))
        )
        run_artifact_hashes = {
            "basis": files_content_digest(sorted((result_dir / "basis").glob("*.ion.xml"))),
            "pseudopotentials": files_content_digest(source_pseudos),
            "rendered_run_fdf": files_content_digest(sorted((result_dir / "structures").glob("*/RUN.fdf"))),
            "pipeline_config": files_content_digest([result_dir / "pipeline_config.yaml"]),
            "checkpoint_manifest": files_content_digest([checkpoint_manifest_path]),
            "graph2mat_config": files_content_digest([training_dir / "config.yaml"]),
            "split_manifests": files_content_digest(
                [
                    *sorted((result_dir / "splits").glob("*_manifest.csv")),
                    result_dir / "splits" / "split_summary.json",
                ]
            ),
            "reference_matrices": files_content_digest(
                [path for path in sorted((result_dir / "siesta_hamiltonians").rglob("*")) if path.is_file()]
            ),
            "prediction_matrices": files_content_digest(
                [path for path in sorted((result_dir / "predicted_hamiltonians").rglob("*")) if path.is_file()]
            ),
            "metric_manifest": files_content_digest([result_dir / "metrics" / "manifest.json"]),
        }
        graph2mat_config_provenance = read_json_file(training_dir / "config_provenance.json")
        checkpoint_manifest_payload = read_json_file(checkpoint_manifest_path)
        checkpoint_graph2mat_provenance = checkpoint_manifest_payload.get("graph2mat_config_provenance")
        material_provenance = flatten_material_provenance(
            read_json_file(dataset_dir / "material_provenance.json"),
            read_json_file(dataset_dir / "samples_manifest.json"),
            graph2mat_config_provenance,
            checkpoint_graph2mat_provenance if isinstance(checkpoint_graph2mat_provenance, dict) else {},
            {
                "dataset_recipe": recipe_metadata,
                "dataset_recipe_parameters": recipe_metadata.get("generation_parameters")
                or recipe_metadata.get("generation_parameters_json"),
                "graph2mat_config_hash": run_artifact_hashes.get("graph2mat_config"),
                "split_manifest_hash": run_artifact_hashes.get("split_manifests"),
                "reference_matrix_sha256": run_artifact_hashes.get("reference_matrices"),
                "prediction_matrix_sha256": run_artifact_hashes.get("prediction_matrices"),
            },
        )
        benchmark_metadata = benchmark_metadata_from_config(config, recipe_metadata)
        pipeline_git = git_metadata_for_path(REPO_ROOT)
        graph2mat_git = graph2mat_git_metadata(config)
        dataset_sample_ids: list[str] = []
        for split_manifest in sorted((result_dir / "splits").glob("*_manifest.csv")):
            for row in read_csv_rows(split_manifest):
                sample_id = str(row.get("sample_id") or row.get("sample") or "")
                if sample_id:
                    dataset_sample_ids.append(sample_id)
        manifest_seed = prepare_metadata.get("seed")
        if manifest_seed in (None, ""):
            manifest_seed = recipe_metadata.get("seed")
        training_base = self._training_base_metadata(
            key,
            dataset_label,
            effective_size,
            dataset_dir,
            result_dir,
            run_id,
            prepare_metadata,
        )
        if run_mode == DATASET_ONLY_RUN_MODE:
            training_index = None
            training_tag = None
        else:
            training_index = self._next_training_index(
                result_group,
                training_base,
                current_result_dir=result_dir,
            )
            training_tag = self._training_tag(str(training_base["dataset_label"]), training_index)
        manifest = {
            "pipeline": key,
            "method_id": "siesta_fc_cartesian" if key == "atom_displacement" else key,
            "dataset_label": dataset_label,
            "dataset_size": size,
            "requested_dataset_size": size,
            "effective_dataset_size": effective_size,
            "run_id": run_id,
            "returncode": returncode,
            "pipeline_elapsed_seconds": pipeline_elapsed_seconds,
            "workspace": str(workspace),
            "dataset_dir": str(dataset_dir),
            "training_dir": str(training_dir),
            "result_dir": str(result_dir),
            "reused_dataset_dir": prepare_metadata.get("reused_dataset_dir"),
            "reused_result_dir": prepare_metadata.get("reused_result_dir"),
            "reused_run_id": prepare_metadata.get("reused_run_id"),
            "dataset_reused": bool(prepare_metadata.get("dataset_reused")),
            "training_tag": training_tag,
            "training_index": training_index,
            "training_base_method_id": training_base["method_id"],
            "training_base_dataset_label": training_base["dataset_label"],
            "training_base_dataset_size": training_base["dataset_size"],
            "training_base_dataset_dir": training_base["dataset_dir"],
            "training_base_result_dir": training_base["result_dir"],
            "training_base_run_id": training_base["run_id"],
            "model_checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "model_checkpoint_metadata": signed_checkpoint,
            "model_checkpoint_sha256": signed_checkpoint.get("sha256"),
            "checkpoint_manifest": str(checkpoint_manifest_path),
            "checkpoint_selection_warning": checkpoint_warning,
            "artifact_hashes": run_artifact_hashes,
            "pipeline_git": pipeline_git,
            "pipeline_commit": pipeline_git.get("commit"),
            "graph2mat_git": graph2mat_git,
            "graph2mat_commit": graph2mat_git.get("commit"),
            "material_provenance": material_provenance,
            **{key: material_provenance.get(key) for key in MATERIAL_FLAT_FIELDS if key in material_provenance},
            "dataset_sample_ids": sorted(set(dataset_sample_ids)),
            "dataset_sample_hash": sample_set_hash(dataset_sample_ids),
            "seed": manifest_seed,
            "dataset_recipe": recipe_metadata,
            "recipe_id": recipe_metadata.get("recipe_id"),
            "recipe_label": recipe_metadata.get("recipe_label"),
            "block_id": recipe_metadata.get("block_id"),
            "block_label": recipe_metadata.get("block_label"),
            "training_plan_index": recipe_metadata.get("training_plan_index"),
            "training_plan_label": recipe_metadata.get("training_plan_label"),
            "training_plan_display_label": recipe_metadata.get("training_plan_display_label"),
            "training_plan_settings": recipe_metadata.get("training_plan_settings"),
            "training_plan_source_dataset_label": recipe_metadata.get("training_plan_source_dataset_label"),
            "sweep_index": recipe_metadata.get("sweep_index"),
            "sweep_label": recipe_metadata.get("sweep_label"),
            "sweep_parameters": recipe_metadata.get("sweep_parameters"),
            "hidden_irreps_dimension": recipe_metadata.get("hidden_irreps_dimension"),
            "recipe_set_hash": recipe_set_hash(recipe_metadata) if recipe_metadata else "",
            "generation_parameters_json": recipe_metadata.get("generation_parameters_json"),
            "predicted_hamiltonians": prediction_count,
            "siesta_hamiltonians": reference_count,
            "timing_breakdown": timing_breakdown,
            "siesta_counts": siesta_counts,
            "performance": config.get("performance", {}),
            "training_settings": config.get("training", {}).get("ui_training_settings", {}),
            "training_hyperparameters": config.get("training", {}),
            "benchmark_metadata": benchmark_metadata,
            "benchmark_method_id": benchmark_metadata.get("benchmark_method_id"),
            "architecture": benchmark_metadata.get("architecture"),
            "readout": benchmark_metadata.get("readout"),
            "hamiltonian_context": benchmark_metadata.get("hamiltonian_context"),
            "context_enabled": benchmark_metadata.get("context_enabled"),
            "loss": benchmark_metadata.get("loss"),
            "loss_kwargs": benchmark_metadata.get("loss_kwargs"),
            "training_stages": benchmark_metadata.get("training_stages"),
            "training_seed": benchmark_metadata.get("seed"),
            "evaluation_config": {
                "metric_version": METRIC_VERSION,
                "hamiltonian_metrics_manifest": str(result_dir / "metrics" / "manifest.json"),
                "reference_policy": benchmark_metadata.get("dataset_reference_policy"),
            },
            "generated_samples": prepare_metadata.get("generated_samples"),
            "completed_samples": prepare_metadata.get("completed_samples"),
            "fc_generated_samples": prepare_metadata.get("fc_generated_samples"),
            "fc_completed_samples": prepare_metadata.get("fc_completed_samples"),
            "metrics": read_metrics_summary(result_dir / "sample_metrics.csv"),
            "hamiltonian_evaluation": evaluation_metrics,
            "run_mode": run_mode,
            "scientific_status": "dataset_only" if run_mode == DATASET_ONLY_RUN_MODE else "pending",
        }
        (result_dir / "manifest.json").write_text(
            json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._append(
            f"[UI] Resultados archivados en {result_dir} | "
            f"predichos: {prediction_count} | siesta: {reference_count}\n"
        )
        return manifest

    def _evaluate_hamiltonian_metrics(
        self,
        key: str,
        config: dict[str, Any],
        result_dir: Path,
    ) -> dict[str, Any]:
        spec_key = "atom_displacement" if key in {"random_cartesian", "siesta_fc_cartesian"} else key
        spec = PIPELINES[spec_key]
        venv_activate = resolve_pipeline_path(spec, config["paths"]["venv_activate"])
        python = venv_activate.parent / "python"
        if not python.exists():
            python = Path(sys.executable)
        script = COMPARISON_ROOT / "scripts" / "evaluate_hamiltonian_metrics.py"
        command = [str(python), str(script), str(result_dir)]
        metric_workers = int((config.get("performance", {}) or {}).get("max_parallel_metric_jobs") or 1)
        if metric_workers > 1:
            command.extend(["--workers", str(metric_workers)])
        if result_dir_needs_kpoint_metrics(result_dir):
            command.append("--enable-kpoint-metrics")
            self._append(
                "[UI] K-grid Monkhorst-Pack no-gamma detectada; "
                "se activa evaluacion Hamiltoniana k-point-aware.\n"
            )
        self._append(f"[UI] Calculando metricas sparse/espectro/DOS: {' '.join(command)}\n")
        started_at = time.time()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **performance_env(config.get("performance", {}))},
        )
        elapsed = time.time() - started_at
        if result.stdout.strip():
            self._append(result.stdout.strip() + "\n")
        if result.stderr.strip():
            self._append(result.stderr.strip() + "\n")

        manifest_path = result_dir / "eigenvalues" / "manifest.json"
        if manifest_path.exists():
            try:
                summary = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                summary = {"exists": True, "error": str(exc)}
        else:
            summary = {"exists": False}
        summary["returncode"] = result.returncode
        summary["evaluation_time_seconds"] = elapsed
        if result.returncode == 0:
            self._append(
                "[UI] Metricas Hamiltonianas archivadas: "
                f"{summary.get('samples_compared', 0)} muestras comparadas.\n"
            )
        else:
            self._append(
                "[WARN] Evaluacion Hamiltoniana termino con codigo "
                f"{result.returncode}. Revisa {manifest_path}.\n"
            )
        fatal_errors = summary.get("fatal_errors") or summary.get("errors") or []
        if result.returncode != 0 or fatal_errors:
            raise RuntimeError(
                "Evaluacion Hamiltoniana fallo con errores fatales: "
                + json.dumps(fatal_errors, ensure_ascii=False)[:1200]
            )
        if summary.get("structural_metrics_error"):
            raise RuntimeError(
                "Evaluacion Hamiltoniana sin basis orbital valida: "
                f"{summary.get('structural_basis_error') or summary.get('structural_metrics_unavailable')}"
            )
        return summary

    def _run_local_script(
        self,
        command: list[str],
        *,
        label: str,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self._append(f"[UI] {label}: {' '.join(command)}\n")
        result = subprocess.run(
            command,
            cwd=cwd or REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if result.stdout.strip():
            self._append(result.stdout.strip() + "\n")
        if result.stderr.strip():
            self._append(result.stderr.strip() + "\n")
        return result

    def _split_manifest_for_result(self, result: dict[str, Any], split_name: str) -> Path:
        result_dir = Path(str(result["result_dir"]))
        split_root = result_dir / "splits"
        for name in (f"{split_name}_valid_manifest.csv", f"{split_name}_manifest.csv"):
            candidate = split_root / name
            if candidate.exists():
                return candidate
        raise RuntimeError(f"Missing {split_name} manifest for {result.get('pipeline')}: {result_dir}")

    def _basis_files_glob_for_result(self, result: dict[str, Any]) -> str:
        config = load_config(Path(str(result["result_dir"])) / "pipeline_config.yaml")
        dataset_dir = Path(config["paths"]["dataset_dir"])
        candidates = [
            dataset_dir / "basis",
            dataset_dir / "MD_steps" / "basis",
            Path(str(config["paths"].get("relaxed_dir", ""))),
        ]
        for directory in candidates:
            if directory.exists() and list(directory.glob("*.ion.xml")):
                return str(directory / "*.ion.xml")
        configured = (
            config.get("prediction", {}).get("data", {}).get("basis_files")
            or config.get("training", {}).get("data", {}).get("basis_files")
        )
        if configured:
            return str(configured)
        raise RuntimeError(f"No basis .ion.xml files found for {result.get('pipeline')} result.")

    def _python_for_result(self, result: dict[str, Any], config: dict[str, Any]) -> str:
        pipeline = str(result.get("pipeline"))
        spec = PIPELINES.get("atom_displacement" if pipeline == "random_cartesian" else pipeline)
        venv_activate = config.get("paths", {}).get("venv_activate")
        if spec is not None and venv_activate:
            python = resolve_pipeline_path(spec, str(venv_activate)).parent / "python"
            if python.exists():
                return str(python)
        return sys.executable

    def _n_matrix_components_for_result(self, config: dict[str, Any]) -> int:
        for section in ("prediction", "testing", "training"):
            data = config.get(section, {}).get("data", {})
            value = data.get("n_matrix_components")
            if value not in (None, ""):
                return int(value)
        raise RuntimeError(
            "Missing Graph2Mat data.n_matrix_components in prediction/testing/training config; "
            "cannot run cross prediction safely."
        )

    def _matrix_component_policy_for_result(self, config: dict[str, Any]) -> str:
        for section in ("prediction", "testing", "training"):
            data = config.get(section, {}).get("data", {})
            value = data.get("matrix_component_policy")
            if value not in (None, ""):
                return str(value)
        raise RuntimeError(
            "Missing Graph2Mat data.matrix_component_policy in prediction/testing/training config; "
            "cannot run cross prediction safely."
        )

    def _prepare_cross_result_dir(
        self,
        cross_result_dir: Path,
        prediction_dir: Path,
        test_manifest: Path,
        basis_files_glob: str | None = None,
    ) -> dict[str, int]:
        reset_output_directory(cross_result_dir)
        prediction_root = prediction_dir / "predicted_hamiltonians"
        if not prediction_root.exists():
            raise RuntimeError(f"Prediction output missing: {prediction_root}")
        shutil.copytree(prediction_root, cross_result_dir / "predicted_hamiltonians")
        copy_if_exists(prediction_dir / "prediction_summary.json", cross_result_dir / "prediction_summary.json")
        copy_if_exists(prediction_dir / "prediction_manifest.csv", cross_result_dir / "prediction_manifest.csv")
        copy_if_exists(test_manifest.parent / "frozen_test_manifest.json", cross_result_dir / "frozen_test_manifest.json")
        if basis_files_glob:
            basis_dir = cross_result_dir / "basis"
            basis_dir.mkdir(parents=True, exist_ok=True)
            for src in sorted(Path(path) for path in glob.glob(basis_files_glob)):
                if src.is_file():
                    shutil.copy2(src, basis_dir / src.name)

        rows = read_csv_rows(test_manifest)
        references = 0
        structures = 0
        for row in rows:
            sample_id = str(row.get("sample_id") or row.get("sample") or "")
            if not sample_id:
                continue
            hamiltonian_path = Path(str(row.get("hamiltonian_path") or ""))
            if hamiltonian_path.exists() and hamiltonian_path.is_file():
                dst = cross_result_dir / "siesta_hamiltonians" / sample_id / hamiltonian_path.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(hamiltonian_path, dst)
                references += 1
            structure_path = Path(str(row.get("structure_path") or ""))
            if structure_path.exists() and structure_path.is_file():
                dst = cross_result_dir / "structures" / sample_id / "RUN.fdf"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(structure_path, dst)
                structures += 1
            metadata_path = Path(str(row.get("metadata_path") or ""))
            if metadata_path.exists() and metadata_path.is_file():
                dst = cross_result_dir / "structures" / sample_id / metadata_path.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(metadata_path, dst)
        return {"references": references, "structures": structures}

    def _common_test_pair_id(self, md_result: dict[str, Any], atom_result: dict[str, Any]) -> str:
        return cross_pair_id(
            {"md": md_result, "atom_displacement": atom_result},
            ["md", "atom_displacement"],
        )

    def _run_cross_evaluation_legacy_binary(self, run_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
        runs = [run for run in manifest.get("runs", []) if run.get("returncode") == 0]
        md_runs = [run for run in runs if run.get("pipeline") == "md"]
        atom_runs = [run for run in runs if run.get("pipeline") == "atom_displacement"]
        summary: dict[str, Any] = {
            "ok": False,
            "warnings": [],
            "missing_cells": [],
            "common_tests": [],
            "cross_evaluations": [],
            "outputs": {},
        }
        if not md_runs or not atom_runs:
            summary["warnings"].append("Both MD and AtomDisplacement successful runs are required for cross evaluation.")
            self._append("[WARN] No hay runs exitosos de ambos metodos; se omite cross-evaluation.\n")
            return summary

        common_root = experiment_root(run_id) / "common_tests"
        cross_root = experiment_root(run_id) / "cross_evaluations"
        prediction_root = experiment_root(run_id) / "cross_predictions"
        summary_root = experiment_root(run_id) / "summary"
        test_sets = deduplicate_common_test_sets(manifest.get("test_sets") or DEFAULT_COMMON_TEST_SETS)
        summary_root.mkdir(parents=True, exist_ok=True)
        expected_grid = build_cross_evaluation_expected_grid(
            ["md", "siesta_fc_cartesian"],
            test_sets,
            experiment_id=run_id,
        )
        expected_grid_path = summary_root / "cross_evaluation_expected_grid.json"
        expected_grid_path.write_text(
            json.dumps(json_safe(expected_grid), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        common_root.mkdir(parents=True, exist_ok=True)
        cross_root.mkdir(parents=True, exist_ok=True)
        prediction_root.mkdir(parents=True, exist_ok=True)

        md_runs = sorted(md_runs, key=lambda run: (int(run.get("dataset_size", 0)), str(run.get("dataset_label", ""))))
        atom_runs = sorted(atom_runs, key=lambda run: (int(run.get("dataset_size", 0)), str(run.get("dataset_label", ""))))
        for md_result in md_runs:
            md_dataset_size = int(md_result.get("dataset_size", 0))
            md_budget = reference_budget_for_run(md_result)
            for atom_result in atom_runs:
                compute_mode = str(manifest.get("compute_budget_mode", "both"))
                if not should_compare_budget_pair(md_result, atom_result, atom_runs, compute_mode):
                    continue
                atom_dataset_size = int(atom_result.get("dataset_size", 0))
                atom_budget = reference_budget_for_run(atom_result)
                ratio = budget_ratio(md_budget, atom_budget)
                mismatch_warning = budget_warning(md_budget, atom_budget)
                pair_id = self._common_test_pair_id(md_result, atom_result)
                pair_common_dir = common_root / pair_id
                build_command = [
                    sys.executable,
                    str(COMPARISON_ROOT / "scripts" / "build_common_tests.py"),
                    "--md-test-manifest",
                    str(self._split_manifest_for_result(md_result, "test")),
                    "--atomdisp-test-manifest",
                    str(self._split_manifest_for_result(atom_result, "test")),
                    "--train-manifest",
                    str(self._split_manifest_for_result(md_result, "train")),
                    "--train-manifest",
                    str(self._split_manifest_for_result(atom_result, "train")),
                    "--output-dir",
                    str(pair_common_dir),
                    "--test-sets",
                    ",".join(test_sets),
                ]
                build_result = self._run_local_script(build_command, label=f"Construyendo common tests {pair_id}")
                if build_result.returncode != 0:
                    raise RuntimeError(f"Common test builder failed for {pair_id}.")
                summary["common_tests"].append(str(pair_common_dir))
                train_manifests = [
                    self._split_manifest_for_result(md_result, "train"),
                    self._split_manifest_for_result(atom_result, "train"),
                ]
                leakage_warnings_by_test_set: dict[str, str] = {}
                leakage_summary_by_test_set: dict[str, str] = {}
                leakage_scientific_status_by_test_set: dict[str, str] = {}
                leakage_severe_warnings_by_test_set: dict[str, list[str]] = {}
                for test_set in test_sets:
                    test_manifest = pair_common_dir / test_set / "test_manifest.csv"
                    if not test_manifest.exists():
                        continue
                    leakage_dir = pair_common_dir / "geometry_leakage" / test_set
                    leakage_command = [
                        sys.executable,
                        str(COMPARISON_ROOT / "scripts" / "check_geometry_leakage.py"),
                        "--train-manifest",
                        str(train_manifests[0]),
                        "--train-manifest",
                        str(train_manifests[1]),
                        "--test-manifest",
                        str(test_manifest),
                        "--output-dir",
                        str(leakage_dir),
                    ]
                    leakage_result = self._run_local_script(
                        leakage_command,
                        label=f"Chequeando leakage geometrico {pair_id} {test_set}",
                    )
                    leakage_summary_path = leakage_dir / "geometry_leakage_summary.json"
                    leakage_diagnostics: dict[str, Any] = {}
                    if leakage_summary_path.exists():
                        leakage_summary_by_test_set[test_set] = str(leakage_summary_path)
                        leakage_diagnostics = geometry_leakage_diagnostic_fields(
                            pair_id=pair_id,
                            test_set=test_set,
                            leakage_dir=leakage_dir,
                            summary_path=leakage_summary_path,
                        )
                        leakage_scientific_status_by_test_set[test_set] = str(
                            leakage_diagnostics.get("scientific_status", "")
                        )
                        leakage_severe_warnings_by_test_set[test_set] = [
                            str(item) for item in leakage_diagnostics.get("severe_warnings", []) or []
                        ]
                    if leakage_result.returncode != 0:
                        warning = str(
                            leakage_diagnostics.get("warning")
                            or f"Geometry leakage detected for {pair_id} {test_set}; see {leakage_dir}."
                        )
                        summary["warnings"].append(warning)
                        leakage_warnings_by_test_set[test_set] = warning
                        if STRICT_COMPARISON_MODE:
                            raise RuntimeError(warning)

                for train_result in (md_result, atom_result):
                    train_method = str(train_result["pipeline"])
                    checkpoint = train_result.get("model_checkpoint")
                    if not checkpoint or not Path(str(checkpoint)).exists():
                        warning = f"Missing checkpoint for {train_method} {train_result.get('dataset_label')}; skipping cross prediction."
                        summary["warnings"].append(warning)
                        self._append(f"[WARN] {warning}\n")
                        continue
                    basis_files = self._basis_files_glob_for_result(train_result)
                    train_config = load_config(Path(str(train_result["result_dir"])) / "pipeline_config.yaml")
                    train_python = self._python_for_result(train_result, train_config)
                    n_matrix_components = self._n_matrix_components_for_result(train_config)
                    matrix_component_policy = self._matrix_component_policy_for_result(train_config)
                    for test_set in test_sets:
                        test_manifest = pair_common_dir / test_set / "test_manifest.csv"
                        if not test_manifest.exists():
                            warning = f"Missing common test manifest {test_manifest}; skipping."
                            summary["warnings"].append(warning)
                            self._append(f"[WARN] {warning}\n")
                            continue
                        frozen_manifest_path = test_manifest.parent / "frozen_test_manifest.json"
                        frozen_test_hash = None
                        frozen_test_warning = ""
                        if frozen_manifest_path.exists():
                            frozen_payload = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
                            frozen_test_hash = frozen_payload.get("frozen_test_hash")
                        else:
                            frozen_test_warning = f"Missing frozen test manifest for {pair_id} {test_set}."
                            if STRICT_COMPARISON_MODE:
                                raise RuntimeError(frozen_test_warning)
                        cross_name = cross_result_name(pair_id, train_method, test_set)
                        prediction_dir = prediction_root / cross_name
                        predict_command = [
                            train_python,
                            str(COMPARISON_ROOT / "scripts" / "predict_model_on_dataset.py"),
                            "--checkpoint",
                            str(checkpoint),
                            "--train-method",
                            train_method,
                            "--test-set",
                            test_set,
                            "--test-manifest",
                            str(test_manifest),
                            "--basis-files",
                            basis_files,
                            "--output-dir",
                            str(prediction_dir),
                            "--accelerator",
                            str(
                                train_config.get("training", {})
                                .get("trainer", {})
                                .get("accelerator", "cpu")
                            ),
                        ]
                        predict_command.extend(
                            [
                                "--matrix-component-policy",
                                matrix_component_policy,
                                "--n-matrix-components",
                                str(n_matrix_components),
                            ]
                        )
                        loader_threads = (
                            train_config.get("training", {})
                            .get("data", {})
                            .get("loader_threads")
                        )
                        if loader_threads is not None:
                            predict_command.extend(["--loader-threads", str(loader_threads)])
                        matmul_precision = train_config.get("training", {}).get("torch_float32_matmul_precision")
                        if matmul_precision in {"high", "medium"}:
                            predict_command.extend(["--torch-float32-matmul-precision", str(matmul_precision)])
                        predict_command.append("--patch-graph2mat-basis-loading")
                        predict_start = time.time()
                        predict_result = self._run_local_script(
                            predict_command,
                            label=f"Prediccion cruzada {cross_name}",
                            env={**os.environ, **performance_env(train_config.get("performance", {}))},
                        )
                        prediction_time = time.time() - predict_start
                        if predict_result.returncode != 0:
                            raise RuntimeError(f"Cross prediction failed for {cross_name}.")

                        cross_result_dir = cross_root / cross_name
                        copy_counts = self._prepare_cross_result_dir(
                            cross_result_dir,
                            prediction_dir,
                            test_manifest,
                            basis_files,
                        )
                        evaluation_start = time.time()
                        evaluation_summary = self._evaluate_hamiltonian_metrics(
                            train_method,
                            train_config,
                            cross_result_dir,
                        )
                        evaluation_time = time.time() - evaluation_start
                        cross_manifest = {
                            "experiment_id": run_id,
                            "pair_id": pair_id,
                            "train_method": train_method,
                            "test_set": test_set,
                            "dataset_size": int(train_result.get("dataset_size", 0)),
                            "train_dataset_size": int(train_result.get("dataset_size", 0)),
                            "md_dataset_size": md_dataset_size,
                            "atom_dataset_size": atom_dataset_size,
                            "compute_budget_mode": compute_mode,
                            "md_siesta_reference_count": md_budget,
                            "atomdisp_siesta_reference_count": atom_budget,
                            "budget_ratio": ratio,
                            "budget_mismatch_warning": mismatch_warning,
                            "leakage_warning": leakage_warnings_by_test_set.get(test_set, ""),
                            "leakage_summary": leakage_summary_by_test_set.get(test_set, ""),
                            "leakage_scientific_status": leakage_scientific_status_by_test_set.get(test_set, ""),
                            "leakage_severe_warnings": leakage_severe_warnings_by_test_set.get(test_set, []),
                            "frozen_test_warning": frozen_test_warning,
                            "frozen_test_hash": frozen_test_hash,
                            "frozen_test_manifest": str(frozen_manifest_path) if frozen_manifest_path.exists() else "",
                            "siesta_settings_hash": manifest.get("siesta_settings_hash"),
                            "siesta_settings_warning": manifest.get("siesta_settings_warning", ""),
                            "model_config_hash": manifest.get("model_config_hash"),
                            "model_config_warning": manifest.get("model_config_warning", ""),
                            "basis_pseudopotential_warning": manifest.get("basis_pseudopotential_warning", ""),
                            "strict_comparison_mode": manifest.get("strict_comparison_mode", STRICT_COMPARISON_MODE),
                            "md_dataset_label": str(md_result.get("dataset_label", f"dataset_{md_dataset_size}")),
                            "atom_dataset_label": str(atom_result.get("dataset_label", f"dataset_{atom_dataset_size}")),
                            "seed": train_result.get("seed"),
                            "epoch": None,
                            "model_checkpoint": str(checkpoint),
                            "model_checkpoint_sha256": train_result.get("model_checkpoint_sha256"),
                            "checkpoint_manifest": train_result.get("checkpoint_manifest", ""),
                            "checkpoint_selection_warning": train_result.get("checkpoint_selection_warning", ""),
                            "reproducibility_warning": manifest.get("reproducibility_warning", ""),
                            "nested_subset_warning": train_result.get("nested_subset_warning", ""),
                            "prediction_dir": str(prediction_dir),
                            "siesta_reference_dir": str(cross_result_dir / "siesta_hamiltonians"),
                            "prediction_time_seconds": prediction_time,
                            "evaluation_time_seconds": evaluation_time,
                            "total_time_seconds": (
                                float(train_result.get("pipeline_elapsed_seconds") or 0.0)
                                + prediction_time
                                + evaluation_time
                            ),
                            "references": copy_counts["references"],
                            "structures": copy_counts["structures"],
                            "evaluation": evaluation_summary,
                        }
                        cross_manifest["method_provenance"] = build_method_provenance(
                            manifest,
                            selected_methods=["md", "siesta_fc_cartesian"],
                            runs=[md_result, atom_result],
                            frozen_test_manifests_by_test_set={
                                test_set: str(frozen_manifest_path) if frozen_manifest_path.exists() else ""
                            },
                        )
                        material_maps = material_maps_from_manifest(cross_manifest)
                        material_warning = material_compatibility_warning(material_maps)
                        cross_manifest.update(material_maps)
                        cross_manifest["method_provenance_warnings"] = sorted(
                            dict.fromkeys(
                                f"{method}: {warning}"
                                for method, provenance in cross_manifest["method_provenance"].items()
                                for warning in provenance.get("warnings", []) or []
                            )
                        )
                        cross_manifest["method_provenance_severe_warnings"] = sorted(
                            dict.fromkeys(
                                [
                                    *(
                                        f"{method}: {warning}"
                                        for method, provenance in cross_manifest["method_provenance"].items()
                                        for warning in provenance.get("severe_warnings", []) or []
                                    ),
                                    *([material_warning] if material_warning else []),
                                ]
                            )
                        )
                        if material_warning:
                            cross_manifest["material_compatibility_warning"] = material_warning
                        (cross_result_dir / "cross_evaluation_manifest.json").write_text(
                            json.dumps(json_safe(cross_manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                            encoding="utf-8",
                        )
                        summary["cross_evaluations"].append(cross_manifest)

        aggregate_command = [
            sys.executable,
            str(COMPARISON_ROOT / "scripts" / "aggregate_cross_metrics.py"),
            "--experiment-id",
            run_id,
            "--cross-root",
            str(cross_root),
            "--output-dir",
            str(summary_root),
            "--expected-grid",
            str(expected_grid_path),
            "--primary-metric",
            str((manifest.get("selected_metrics") or {}).get("primary_metric", DEFAULT_PRIMARY_METRIC)),
        ]
        aggregate_result = self._run_local_script(
            aggregate_command,
            label="Agregando metricas cruzadas",
            env={**os.environ, **performance_env(manifest.get("performance", {}))},
        )
        if aggregate_result.returncode != 0:
            raise RuntimeError("Cross metric aggregation failed.")

        completeness_path = summary_root / "cross_evaluation_completeness.json"
        completeness = {}
        if completeness_path.exists():
            completeness = json.loads(completeness_path.read_text(encoding="utf-8"))
        if completeness.get("scientific_status") == "invalid_incomplete_grid":
            missing_cells = sorted(
                dict.fromkeys(
                    [
                        *(str(cell) for cell in completeness.get("missing_cells", []) or []),
                        *(str(cell) for cell in completeness.get("missing_primary_metric_cells", []) or []),
                        *(str(cell) for cell in completeness.get("missing_context_cells", []) or []),
                        *(str(cell) for cell in completeness.get("missing_primary_metric_context_cells", []) or []),
                    ]
                )
            )
            missing_required_cells = [
                *(completeness.get("missing_cells", []) or []),
                *(completeness.get("missing_context_cells", []) or []),
            ]
            missing_primary_metric_cells = [
                *(completeness.get("missing_primary_metric_cells", []) or []),
                *(completeness.get("missing_primary_metric_context_cells", []) or []),
            ]
            extra_unexpected_cells = [
                *(completeness.get("extra_unexpected_cells", []) or []),
                *(completeness.get("extra_unexpected_context_cells", []) or []),
            ]
            invalid_recommendation = {
                "status": "invalid_incomplete_grid",
                "scientific_status": "invalid_incomplete_grid",
                "winner": None,
                "reason": "Incomplete cross-evaluation grid",
                "missing_cells": missing_cells,
                "missing_required_cells": missing_required_cells,
                "extra_unexpected_cells": extra_unexpected_cells,
                "missing_primary_metric_cells": missing_primary_metric_cells,
                "completeness_report": str(completeness_path),
            }
            (summary_root / "recommendation.json").write_text(
                json.dumps(invalid_recommendation, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            summary["missing_cells"].extend(missing_required_cells)
            summary["missing_cells"].extend(missing_primary_metric_cells)
            summary["warnings"].append("Incomplete cross-evaluation grid; winner analysis skipped.")
            summary["ok"] = False
            summary["outputs"] = {
                "common_tests": str(common_root),
                "cross_evaluations": str(cross_root),
                "cross_evaluation_metrics": str(summary_root / "cross_evaluation_metrics.csv"),
                "cross_evaluation_expected_grid": str(expected_grid_path),
                "cross_evaluation_completeness": str(completeness_path),
                "recommendation": str(summary_root / "recommendation.json"),
            }
            return summary

        winner_command = [
            sys.executable,
            str(COMPARISON_ROOT / "scripts" / "analyze_winners.py"),
            "--metrics-csv",
            str(summary_root / "cross_evaluation_metrics.csv"),
            "--output-dir",
            str(summary_root),
            "--primary-metric",
            str((manifest.get("selected_metrics") or {}).get("primary_metric", DEFAULT_PRIMARY_METRIC)),
            "--minimum-robust-seeds",
            str(manifest.get("minimum_robust_seeds", MINIMUM_ROBUST_SEEDS)),
        ]
        winner_result = self._run_local_script(
            winner_command,
            label="Analizando winners",
            env={**os.environ, **performance_env(manifest.get("performance", {}))},
        )
        if winner_result.returncode != 0:
            raise RuntimeError("Winner analysis failed.")

        summary["ok"] = True
        summary["outputs"] = {
            "common_tests": str(common_root),
            "cross_evaluations": str(cross_root),
            "cross_evaluation_metrics": str(summary_root / "cross_evaluation_metrics.csv"),
            "winner_summary": str(summary_root / "winner_summary.csv"),
            "recommendation": str(summary_root / "recommendation.json"),
        }
        return summary

    def _run_cross_evaluation(self, run_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
        runs = [run for run in manifest.get("runs", []) if run.get("returncode") == 0]
        selected_methods = [
            normalize_method_id(item)
            for item in manifest.get("selected_methods") or []
        ]
        if not selected_methods:
            selected_methods = sorted(
                {
                    normalize_method_id(run.get("method_id") or run.get("pipeline"))
                    for run in runs
                    if run.get("method_id") or run.get("pipeline")
                }
            )
        runs_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            method_id = normalize_method_id(
                run.get("method_id") or run.get("pipeline") or "",
                allow_unknown=True,
            )
            if method_id:
                runs_by_method[method_id].append(run)

        summary: dict[str, Any] = {
            "ok": False,
            "warnings": [],
            "missing_cells": [],
            "common_tests": [],
            "cross_evaluations": [],
            "outputs": {},
        }
        common_root = experiment_root(run_id) / "common_tests"
        cross_root = experiment_root(run_id) / "cross_evaluations"
        prediction_root = experiment_root(run_id) / "cross_predictions"
        summary_root = experiment_root(run_id) / "summary"
        test_sets = deduplicate_common_test_sets(
            manifest.get("test_sets") or [f"test_{method}" for method in selected_methods] + ["test_mixed"]
        )
        summary_root.mkdir(parents=True, exist_ok=True)
        expected_grid = build_cross_evaluation_expected_grid(
            selected_methods,
            test_sets,
            experiment_id=run_id,
        )
        expected_grid_path = summary_root / "cross_evaluation_expected_grid.json"
        expected_grid_path.write_text(
            json.dumps(json_safe(expected_grid), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        summary["outputs"]["cross_evaluation_expected_grid"] = str(expected_grid_path)

        missing_methods = [method for method in selected_methods if not runs_by_method.get(method)]
        if missing_methods:
            warning = f"Missing successful runs for selected methods: {missing_methods}."
            summary["warnings"].append(warning)
            summary["missing_cells"].append(warning)
            self._append(f"[WARN] {warning}\n")
            return summary
        if len(selected_methods) < 2:
            summary["warnings"].append("At least two selected methods are required for cross evaluation.")
            return summary

        common_root.mkdir(parents=True, exist_ok=True)
        cross_root.mkdir(parents=True, exist_ok=True)
        prediction_root.mkdir(parents=True, exist_ok=True)

        method_run_lists = [
            sorted(runs_by_method[method], key=lambda run: (int(run.get("dataset_size", 0)), str(run.get("dataset_label", ""))))
            for method in selected_methods
        ]
        expected_context_cells: list[dict[str, Any]] = []
        for combo in itertools.product(*method_run_lists):
            combo_by_method = {
                normalize_method_id(run.get("method_id") or run.get("pipeline"), allow_unknown=True): run
                for run in combo
            }
            pair_id = cross_pair_id(combo_by_method, selected_methods)
            for train_method in selected_methods:
                train_result = combo_by_method[train_method]
                for test_set in test_sets:
                    expected_context_cells.append(
                        {
                            "pair_id": pair_id,
                            "train_method": train_method,
                            "test_set": test_set,
                            "cell_id": f"{pair_id} :: {train_method} on {test_set}",
                            "train_dataset_label": train_result.get("dataset_label", ""),
                            "train_training_tag": train_result.get("training_tag"),
                            "train_training_plan_label": train_result.get("training_plan_label"),
                            "train_training_plan_index": train_result.get("training_plan_index"),
                        }
                    )
        expected_grid["expected_context_cell_count"] = len(expected_context_cells)
        expected_grid["expected_context_cells"] = expected_context_cells
        expected_grid_path.write_text(
            json.dumps(json_safe(expected_grid), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for combo in itertools.product(*method_run_lists):
            combo_by_method = {
                normalize_method_id(run.get("method_id") or run.get("pipeline"), allow_unknown=True): run
                for run in combo
            }
            dataset_size_by_method = {
                method: int(combo_by_method[method].get("dataset_size", 0))
                for method in selected_methods
            }
            dataset_label_by_method = {
                method: str(
                    combo_by_method[method].get(
                        "dataset_label",
                        f"dataset_{dataset_size_by_method.get(method, 0)}",
                    )
                )
                for method in selected_methods
            }
            recipe_id_by_method = {
                method: combo_by_method[method].get("recipe_id")
                for method in selected_methods
            }
            recipe_label_by_method = {
                method: combo_by_method[method].get("recipe_label")
                for method in selected_methods
            }
            recipe_set_hash_by_method = {
                method: combo_by_method[method].get("recipe_set_hash")
                for method in selected_methods
            }
            training_tag_by_method = {
                method: combo_by_method[method].get("training_tag")
                for method in selected_methods
            }
            training_plan_label_by_method = {
                method: combo_by_method[method].get("training_plan_label")
                for method in selected_methods
            }
            training_plan_settings_by_method = {
                method: combo_by_method[method].get("training_plan_settings")
                or combo_by_method[method].get("training_settings")
                for method in selected_methods
            }
            training_plan_settings_warning = ""
            if any(value for value in training_plan_settings_by_method.values()):
                settings_fingerprints = {
                    method: json.dumps(
                        json_safe(training_plan_settings_by_method.get(method) or {}),
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    for method in selected_methods
                }
                if len(set(settings_fingerprints.values())) > 1:
                    training_plan_settings_warning = (
                        "Training plan settings differ across methods for this cross-evaluation pair: "
                        + ", ".join(
                            f"{method}={settings_fingerprints[method]}"
                            for method in selected_methods
                        )
                    )
            pair_id = cross_pair_id(combo_by_method, selected_methods)
            pair_common_dir = common_root / pair_id
            build_command = [
                sys.executable,
                str(COMPARISON_ROOT / "scripts" / "build_common_tests.py"),
                "--output-dir",
                str(pair_common_dir),
                "--test-sets",
                ",".join(test_sets),
            ]
            for method in selected_methods:
                build_command.extend(
                    [
                        "--test-manifest",
                        f"{method}={self._split_manifest_for_result(combo_by_method[method], 'test')}",
                    ]
                )
            train_manifests = []
            for method in selected_methods:
                train_manifest = self._split_manifest_for_result(combo_by_method[method], "train")
                train_manifests.append(train_manifest)
                build_command.extend(["--train-manifest", str(train_manifest)])
            build_result = self._run_local_script(build_command, label=f"Construyendo common tests {pair_id}")
            if build_result.returncode != 0:
                raise RuntimeError(f"Common test builder failed for {pair_id}.")
            summary["common_tests"].append(str(pair_common_dir))

            leakage_warnings_by_test_set: dict[str, str] = {}
            leakage_summary_by_test_set: dict[str, str] = {}
            leakage_scientific_status_by_test_set: dict[str, str] = {}
            leakage_severe_warnings_by_test_set: dict[str, list[str]] = {}
            for test_set in test_sets:
                test_manifest = pair_common_dir / test_set / "test_manifest.csv"
                if not test_manifest.exists():
                    summary["missing_cells"].append(f"missing test manifest {pair_id} {test_set}")
                    continue
                leakage_dir = pair_common_dir / "geometry_leakage" / test_set
                leakage_command = [
                    sys.executable,
                    str(COMPARISON_ROOT / "scripts" / "check_geometry_leakage.py"),
                    "--test-manifest",
                    str(test_manifest),
                    "--output-dir",
                    str(leakage_dir),
                ]
                for train_manifest in train_manifests:
                    leakage_command.extend(["--train-manifest", str(train_manifest)])
                leakage_result = self._run_local_script(
                    leakage_command,
                    label=f"Chequeando leakage geometrico {pair_id} {test_set}",
                )
                leakage_summary_path = leakage_dir / "geometry_leakage_summary.json"
                leakage_diagnostics: dict[str, Any] = {}
                if leakage_summary_path.exists():
                    leakage_summary_by_test_set[test_set] = str(leakage_summary_path)
                    leakage_diagnostics = geometry_leakage_diagnostic_fields(
                        pair_id=pair_id,
                        test_set=test_set,
                        leakage_dir=leakage_dir,
                        summary_path=leakage_summary_path,
                    )
                    leakage_scientific_status_by_test_set[test_set] = str(
                        leakage_diagnostics.get("scientific_status", "")
                    )
                    leakage_severe_warnings_by_test_set[test_set] = [
                        str(item) for item in leakage_diagnostics.get("severe_warnings", []) or []
                    ]
                if leakage_result.returncode != 0:
                    warning = str(
                        leakage_diagnostics.get("warning")
                        or f"Geometry leakage detected for {pair_id} {test_set}; see {leakage_dir}."
                    )
                    summary["warnings"].append(warning)
                    leakage_warnings_by_test_set[test_set] = warning
                    if STRICT_COMPARISON_MODE:
                        raise RuntimeError(warning)

            performance = parse_performance_settings(
                manifest.get("performance", {}),
                compute_accelerator=str(manifest.get("compute_accelerator", "cpu")),
            )
            error_policy = str(performance.get("error_policy", "fail_fast"))
            prediction_workers = int(performance.get("max_parallel_prediction_jobs", 1) or 1)
            evaluation_workers = int(performance.get("max_parallel_evaluation_jobs", 1) or 1)
            if str(manifest.get("compute_accelerator", "")).lower() == "gpu" and prediction_workers > 1:
                self._append(
                    "[WARN] compute_accelerator=gpu: cross predictions se serializan "
                    "para no sobresuscribir una unica GPU.\n"
                )
                prediction_workers = 1
            prediction_tasks: list[tuple[str, Any]] = []
            for train_method in selected_methods:
                train_result = combo_by_method[train_method]
                checkpoint = train_result.get("model_checkpoint")
                if not checkpoint or not Path(str(checkpoint)).exists():
                    warning = f"Missing checkpoint for {train_method} {train_result.get('dataset_label')}; skipping cross prediction."
                    summary["warnings"].append(warning)
                    summary["missing_cells"].append(warning)
                    self._append(f"[WARN] {warning}\n")
                    continue
                basis_files = self._basis_files_glob_for_result(train_result)
                train_config = load_config(Path(str(train_result["result_dir"])) / "pipeline_config.yaml")
                train_python = self._python_for_result(train_result, train_config)
                n_matrix_components = self._n_matrix_components_for_result(train_config)
                matrix_component_policy = self._matrix_component_policy_for_result(train_config)
                for test_set in test_sets:
                    test_manifest = pair_common_dir / test_set / "test_manifest.csv"
                    test_method = test_set.removeprefix("test_")
                    test_method = normalize_method_id(test_method, allow_unknown=True)
                    if test_method == "mixed":
                        test_method = "mixed"
                    required_cell = f"{train_method} on {test_set}"
                    if not test_manifest.exists():
                        summary["missing_cells"].append(required_cell)
                        continue
                    frozen_manifest_path = test_manifest.parent / "frozen_test_manifest.json"
                    frozen_test_hash = None
                    frozen_test_warning = ""
                    if frozen_manifest_path.exists():
                        frozen_payload = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
                        frozen_test_hash = frozen_payload.get("frozen_test_hash")
                    else:
                        frozen_test_warning = f"Missing frozen test manifest for {pair_id} {test_set}."
                        summary["missing_cells"].append(required_cell)
                        if STRICT_COMPARISON_MODE:
                            raise RuntimeError(frozen_test_warning)
                    cross_name = cross_result_name(pair_id, train_method, test_set)
                    prediction_dir = prediction_root / cross_name
                    predict_command = [
                        train_python,
                        str(COMPARISON_ROOT / "scripts" / "predict_model_on_dataset.py"),
                        "--checkpoint",
                        str(checkpoint),
                        "--train-method",
                        train_method,
                        "--test-set",
                        test_set,
                        "--test-manifest",
                        str(test_manifest),
                        "--basis-files",
                        basis_files,
                        "--output-dir",
                        str(prediction_dir),
                        "--accelerator",
                        str(
                            train_config.get("training", {})
                            .get("trainer", {})
                            .get("accelerator", "cpu")
                        ),
                    ]
                    predict_command.extend(
                        [
                            "--matrix-component-policy",
                            matrix_component_policy,
                            "--n-matrix-components",
                            str(n_matrix_components),
                        ]
                    )
                    loader_threads = (
                        train_config.get("training", {})
                        .get("data", {})
                        .get("loader_threads")
                    )
                    if loader_threads is not None:
                        predict_command.extend(["--loader-threads", str(loader_threads)])
                    matmul_precision = train_config.get("training", {}).get("torch_float32_matmul_precision")
                    if matmul_precision in {"high", "medium"}:
                        predict_command.extend(["--torch-float32-matmul-precision", str(matmul_precision)])
                    predict_command.append("--patch-graph2mat-basis-loading")

                    def run_prediction(
                        *,
                        command: list[str] = list(predict_command),
                        cross_name: str = cross_name,
                        required_cell: str = required_cell,
                        train_config: dict[str, Any] = train_config,
                        train_method: str = train_method,
                        test_set: str = test_set,
                        test_method: str = test_method,
                        test_manifest: Path = test_manifest,
                        basis_files: str = basis_files,
                        prediction_dir: Path = prediction_dir,
                        train_result: dict[str, Any] = train_result,
                        checkpoint: Any = checkpoint,
                        frozen_manifest_path: Path = frozen_manifest_path,
                        frozen_test_hash: Any = frozen_test_hash,
                        frozen_test_warning: str = frozen_test_warning,
                    ) -> dict[str, Any]:
                        predict_start = time.time()
                        predict_result = self._run_local_script(
                            command,
                            label=f"Prediccion cruzada {cross_name}",
                            env={**os.environ, **performance_env(train_config.get("performance", {}))},
                        )
                        prediction_time = time.time() - predict_start
                        if predict_result.returncode != 0:
                            raise RuntimeError(f"{required_cell}: Cross prediction failed for {cross_name}.")
                        return {
                            "cross_name": cross_name,
                            "required_cell": required_cell,
                            "train_config": train_config,
                            "train_method": train_method,
                            "test_set": test_set,
                            "test_method": test_method,
                            "test_manifest": test_manifest,
                            "basis_files": basis_files,
                            "prediction_dir": prediction_dir,
                            "prediction_time": prediction_time,
                            "train_result": train_result,
                            "checkpoint": checkpoint,
                            "frozen_manifest_path": frozen_manifest_path,
                            "frozen_test_hash": frozen_test_hash,
                            "frozen_test_warning": frozen_test_warning,
                        }

                    prediction_tasks.append((cross_name, run_prediction))

            prediction_results, prediction_failures = self._run_callable_tasks(
                prediction_tasks,
                workers=prediction_workers,
                error_policy=error_policy,
                stage="cross_prediction",
            )
            if prediction_failures:
                summary["warnings"].extend(prediction_failures)
                summary["missing_cells"].extend(prediction_failures)

            evaluation_tasks: list[tuple[str, Any]] = []
            for prediction_payload in prediction_results:
                cross_name = str(prediction_payload["cross_name"])

                def run_evaluation(
                    *,
                    payload: dict[str, Any] = prediction_payload,
                    cross_name: str = cross_name,
                ) -> dict[str, Any]:
                    train_result = payload["train_result"]
                    train_method = str(payload["train_method"])
                    train_config = payload["train_config"]
                    test_set = str(payload["test_set"])
                    cross_result_dir = cross_root / cross_name
                    copy_counts = self._prepare_cross_result_dir(
                        cross_result_dir,
                        Path(payload["prediction_dir"]),
                        Path(payload["test_manifest"]),
                        str(payload["basis_files"]),
                    )
                    evaluation_start = time.time()
                    evaluation_summary = self._evaluate_hamiltonian_metrics(train_method, train_config, cross_result_dir)
                    evaluation_time = time.time() - evaluation_start
                    model_config_warning = " | ".join(
                        item
                        for item in (
                            manifest.get("model_config_warning", ""),
                            training_plan_settings_warning,
                        )
                        if item
                    )
                    cross_manifest = {
                        "experiment_id": run_id,
                        "pair_id": pair_id,
                        "train_method": train_method,
                        "test_set": test_set,
                        "test_method": payload["test_method"],
                        "dataset_size": int(train_result.get("dataset_size", 0)),
                        "train_dataset_size": int(train_result.get("dataset_size", 0)),
                        "dataset_size_by_method": dataset_size_by_method,
                        "dataset_label_by_method": dataset_label_by_method,
                        "recipe_id_by_method": recipe_id_by_method,
                        "recipe_label_by_method": recipe_label_by_method,
                        "recipe_set_hash_by_method": recipe_set_hash_by_method,
                        "training_tag_by_method": training_tag_by_method,
                        "training_plan_label_by_method": training_plan_label_by_method,
                        "training_plan_settings_by_method": training_plan_settings_by_method,
                        "recipe_set_hash": train_result.get("recipe_set_hash", ""),
                        "train_dataset_label": train_result.get("dataset_label", ""),
                        "train_recipe_id": train_result.get("recipe_id"),
                        "train_recipe_label": train_result.get("recipe_label"),
                        "train_block_id": train_result.get("block_id"),
                        "train_block_label": train_result.get("block_label"),
                        "train_generation_parameters_json": train_result.get("generation_parameters_json"),
                        "train_training_tag": train_result.get("training_tag"),
                        "train_training_index": train_result.get("training_index"),
                        "train_training_settings": train_result.get("training_settings"),
                        "train_training_plan_index": train_result.get("training_plan_index"),
                        "train_training_plan_label": train_result.get("training_plan_label"),
                        "train_training_plan_settings": train_result.get("training_plan_settings"),
                        "train_training_plan_source_dataset_label": train_result.get(
                            "training_plan_source_dataset_label"
                        ),
                        "train_sweep_index": train_result.get("sweep_index"),
                        "train_sweep_label": train_result.get("sweep_label"),
                        "train_sweep_parameters": train_result.get("sweep_parameters"),
                        "train_hidden_irreps_dimension": train_result.get("hidden_irreps_dimension"),
                        "training_tag": train_result.get("training_tag"),
                        "training_index": train_result.get("training_index"),
                        "training_settings": train_result.get("training_settings"),
                        "training_plan_index": train_result.get("training_plan_index"),
                        "training_plan_label": train_result.get("training_plan_label"),
                        "training_plan_settings": train_result.get("training_plan_settings"),
                        "training_plan_source_dataset_label": train_result.get(
                            "training_plan_source_dataset_label"
                        ),
                        "sweep_index": train_result.get("sweep_index"),
                        "sweep_label": train_result.get("sweep_label"),
                        "sweep_parameters": train_result.get("sweep_parameters"),
                        "hidden_irreps_dimension": train_result.get("hidden_irreps_dimension"),
                        "md_dataset_size": dataset_size_by_method.get("md"),
                        "atom_dataset_size": dataset_size_by_method.get("siesta_fc_cartesian"),
                        "random_dataset_size": dataset_size_by_method.get("random_cartesian"),
                        "md_dataset_label": dataset_label_by_method.get("md"),
                        "atom_dataset_label": dataset_label_by_method.get("siesta_fc_cartesian"),
                        "random_dataset_label": dataset_label_by_method.get("random_cartesian"),
                        "md_recipe_set_hash": recipe_set_hash_by_method.get("md"),
                        "atom_recipe_set_hash": recipe_set_hash_by_method.get("siesta_fc_cartesian"),
                        "random_recipe_set_hash": recipe_set_hash_by_method.get("random_cartesian"),
                        "compute_budget_mode": manifest.get("compute_budget_mode", "both"),
                        "leakage_warning": leakage_warnings_by_test_set.get(test_set, ""),
                        "leakage_summary": leakage_summary_by_test_set.get(test_set, ""),
                        "leakage_scientific_status": leakage_scientific_status_by_test_set.get(test_set, ""),
                        "leakage_severe_warnings": leakage_severe_warnings_by_test_set.get(test_set, []),
                        "frozen_test_warning": payload["frozen_test_warning"],
                        "frozen_test_hash": payload["frozen_test_hash"],
                        "frozen_test_manifest": str(payload["frozen_manifest_path"]) if Path(payload["frozen_manifest_path"]).exists() else "",
                        "siesta_settings_hash": manifest.get("siesta_settings_hash"),
                        "siesta_settings_warning": manifest.get("siesta_settings_warning", ""),
                        "model_config_hash": manifest.get("model_config_hash"),
                        "model_config_warning": model_config_warning,
                        "training_plan_settings_warning": training_plan_settings_warning,
                        "basis_pseudopotential_warning": manifest.get("basis_pseudopotential_warning", ""),
                        "strict_comparison_mode": manifest.get("strict_comparison_mode", STRICT_COMPARISON_MODE),
                        "seed": train_result.get("seed"),
                        "epoch": None,
                        "model_checkpoint": str(payload["checkpoint"]),
                        "model_checkpoint_sha256": train_result.get("model_checkpoint_sha256"),
                        "checkpoint_manifest": train_result.get("checkpoint_manifest", ""),
                        "checkpoint_selection_warning": train_result.get("checkpoint_selection_warning", ""),
                        "reproducibility_warning": manifest.get("reproducibility_warning", ""),
                        "nested_subset_warning": train_result.get("nested_subset_warning", ""),
                        "prediction_dir": str(payload["prediction_dir"]),
                        "siesta_reference_dir": str(cross_result_dir / "siesta_hamiltonians"),
                        "prediction_time_seconds": payload["prediction_time"],
                        "evaluation_time_seconds": evaluation_time,
                        "total_time_seconds": float(train_result.get("pipeline_elapsed_seconds") or 0.0)
                        + float(payload["prediction_time"])
                        + evaluation_time,
                        "references": copy_counts["references"],
                        "structures": copy_counts["structures"],
                        "evaluation": evaluation_summary,
                    }
                    cross_manifest["method_provenance"] = build_method_provenance(
                        manifest,
                        selected_methods=selected_methods,
                        runs=[combo_by_method[method] for method in selected_methods],
                        frozen_test_manifests_by_test_set={
                            test_set: str(payload["frozen_manifest_path"])
                            if Path(payload["frozen_manifest_path"]).exists()
                            else ""
                        },
                    )
                    material_maps = material_maps_from_manifest(cross_manifest)
                    material_warning = material_compatibility_warning(material_maps)
                    cross_manifest.update(material_maps)
                    if material_warning:
                        cross_manifest["material_compatibility_warning"] = material_warning
                    cross_manifest["method_provenance_warnings"] = sorted(
                        dict.fromkeys(
                            f"{method}: {warning}"
                            for method, provenance in cross_manifest["method_provenance"].items()
                            for warning in provenance.get("warnings", []) or []
                        )
                    )
                    cross_manifest["method_provenance_severe_warnings"] = sorted(
                        dict.fromkeys(
                            [
                                *(
                                    f"{method}: {warning}"
                                    for method, provenance in cross_manifest["method_provenance"].items()
                                    for warning in provenance.get("severe_warnings", []) or []
                                ),
                                *([material_warning] if material_warning else []),
                            ]
                        )
                    )
                    (cross_result_dir / "cross_evaluation_manifest.json").write_text(
                        json.dumps(json_safe(cross_manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                    return cross_manifest

                evaluation_tasks.append((cross_name, run_evaluation))

            evaluation_results, evaluation_failures = self._run_callable_tasks(
                evaluation_tasks,
                workers=evaluation_workers,
                error_policy=error_policy,
                stage="cross_evaluation",
            )
            if evaluation_failures:
                summary["warnings"].extend(evaluation_failures)
                summary["missing_cells"].extend(evaluation_failures)
            summary["cross_evaluations"].extend(evaluation_results)

        primary_metric = str((manifest.get("selected_metrics") or {}).get("primary_metric", DEFAULT_PRIMARY_METRIC))
        aggregate_command = [
            sys.executable,
            str(COMPARISON_ROOT / "scripts" / "aggregate_cross_metrics.py"),
            "--experiment-id",
            run_id,
            "--cross-root",
            str(cross_root),
            "--output-dir",
            str(summary_root),
            "--expected-grid",
            str(expected_grid_path),
            "--primary-metric",
            primary_metric,
        ]
        aggregate_result = self._run_local_script(
            aggregate_command,
            label="Agregando metricas cruzadas",
            env={**os.environ, **performance_env(manifest.get("performance", {}))},
        )
        if aggregate_result.returncode != 0:
            raise RuntimeError("Cross metric aggregation failed.")

        completeness_path = summary_root / "cross_evaluation_completeness.json"
        completeness = {}
        if completeness_path.exists():
            completeness = json.loads(completeness_path.read_text(encoding="utf-8"))
        if completeness.get("scientific_status") == "invalid_incomplete_grid":
            missing_cells = sorted(
                dict.fromkeys(
                    [
                        *(str(cell) for cell in completeness.get("missing_cells", []) or []),
                        *(str(cell) for cell in completeness.get("missing_primary_metric_cells", []) or []),
                        *(str(cell) for cell in completeness.get("missing_context_cells", []) or []),
                        *(str(cell) for cell in completeness.get("missing_primary_metric_context_cells", []) or []),
                    ]
                )
            )
            missing_required_cells = [
                *(completeness.get("missing_cells", []) or []),
                *(completeness.get("missing_context_cells", []) or []),
            ]
            missing_primary_metric_cells = [
                *(completeness.get("missing_primary_metric_cells", []) or []),
                *(completeness.get("missing_primary_metric_context_cells", []) or []),
            ]
            extra_unexpected_cells = [
                *(completeness.get("extra_unexpected_cells", []) or []),
                *(completeness.get("extra_unexpected_context_cells", []) or []),
            ]
            invalid_recommendation = {
                "status": "invalid_incomplete_grid",
                "scientific_status": "invalid_incomplete_grid",
                "winner": None,
                "reason": "Incomplete cross-evaluation grid",
                "missing_cells": missing_cells,
                "missing_required_cells": missing_required_cells,
                "extra_unexpected_cells": extra_unexpected_cells,
                "missing_primary_metric_cells": missing_primary_metric_cells,
                "completeness_report": str(completeness_path),
            }
            (summary_root / "recommendation.json").write_text(
                json.dumps(invalid_recommendation, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            summary["missing_cells"].extend(missing_required_cells)
            summary["missing_cells"].extend(missing_primary_metric_cells)
            summary["warnings"].append("Incomplete cross-evaluation grid; winner analysis skipped.")
            summary["ok"] = False
            summary["outputs"] = {
                "common_tests": str(common_root),
                "cross_evaluations": str(cross_root),
                "cross_evaluation_metrics": str(summary_root / "cross_evaluation_metrics.csv"),
                "cross_evaluation_expected_grid": str(expected_grid_path),
                "cross_evaluation_completeness": str(completeness_path),
                "recommendation": str(summary_root / "recommendation.json"),
            }
            return summary

        winner_command = [
            sys.executable,
            str(COMPARISON_ROOT / "scripts" / "analyze_winners.py"),
            "--metrics-csv",
            str(summary_root / "cross_evaluation_metrics.csv"),
            "--output-dir",
            str(summary_root),
            "--primary-metric",
            primary_metric,
            "--minimum-robust-seeds",
            str(manifest.get("minimum_robust_seeds", MINIMUM_ROBUST_SEEDS)),
        ]
        winner_result = self._run_local_script(
            winner_command,
            label="Analizando winners",
            env={**os.environ, **performance_env(manifest.get("performance", {}))},
        )
        if winner_result.returncode != 0:
            raise RuntimeError("Winner analysis failed.")

        summary["ok"] = bool(completeness.get("complete", True))
        summary["outputs"] = {
            "common_tests": str(common_root),
            "cross_evaluations": str(cross_root),
            "cross_evaluation_metrics": str(summary_root / "cross_evaluation_metrics.csv"),
            "cross_evaluation_expected_grid": str(expected_grid_path),
            "cross_evaluation_completeness": str(completeness_path),
            "winner_summary": str(summary_root / "winner_summary.csv"),
            "recommendation": str(summary_root / "recommendation.json"),
        }
        return summary


EXPERIMENT_RUNNER = ExperimentRunner()
G2M_DEEPH_RUNNER = Graph2MatDeepHBenchmarkRunner()


def all_status() -> dict[str, Any]:
    statuses = {key: runner.status() for key, runner in RUNNERS.items()}
    return {
        "running": any(status["running"] for status in statuses.values()) or MIXING_E2E_RUNNER.status().get("running"),
        "pipelines": statuses,
        "mixing_e2e": MIXING_E2E_RUNNER.status(),
    }


def clear_generated_dataset_outputs(*, dry_run: bool = False) -> dict[str, Any]:
    if (
        all_status().get("running")
        or EXPERIMENT_RUNNER.status().get("running")
        or G2M_DEEPH_RUNNER.status().get("running")
    ):
        raise RuntimeError("No se pueden borrar datasets mientras hay pipelines o experimentos en ejecucion.")
    return cleanup_generated_datasets(REPO_ROOT, dry_run=dry_run)


def generated_dataset_output_records() -> dict[str, Any]:
    return {
        "targets": generated_dataset_records(REPO_ROOT),
        "scope": "all_generated_artifacts",
        "roots": {
            "md_dataset": str(REPO_ROOT / "MD" / "dataset"),
            "atom_dataset": str(REPO_ROOT / "AtomDisplacement" / "dataset"),
            "comparison_workspaces": str(WORKSPACES_ROOT),
            "comparison_results": str(RESULTS_ROOT),
        },
    }


def cleanup_target_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]


def relative_to_repo(path: Path) -> str:
    candidate = path if path.is_absolute() else REPO_ROOT / path
    try:
        return candidate.resolve(strict=False).relative_to(REPO_ROOT.resolve(strict=False)).as_posix()
    except ValueError:
        return candidate.as_posix()


def plot_metric_dataset_records(*, include_bytes: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    groups = {
        "md": RESULTS_ROOT / "results_md",
        "siesta_fc_cartesian": RESULTS_ROOT / "results_atomdisp",
        "random_cartesian": RESULTS_ROOT / "results_random_cartesian",
    }
    for method_id, root in groups.items():
        if not root.exists():
            continue
        for manifest_path in archived_result_manifest_paths(root):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            result_dir = archived_manifest_result_dir(manifest, manifest_path)
            metric_row_counts = archived_run_plot_metric_row_counts(result_dir)
            metric_files = [key for key, count in metric_row_counts.items() if count > 0]
            if not metric_files:
                continue
            try:
                stat = result_dir.stat()
            except OSError:
                stat = None
            relative = relative_to_repo(result_dir)
            records.append(
                {
                    "id": cleanup_target_id(relative),
                    "name": result_dir.name,
                    "path": relative,
                    "relative_path": relative,
                    "kind": "plot_metric_run",
                    "method": manifest.get("method_id") or method_id,
                    "dataset_label": manifest.get("dataset_label") or result_dir.parent.name,
                    "dataset_size": manifest.get("requested_dataset_size") or manifest.get("dataset_size"),
                    "run_id": manifest.get("run_id") or result_dir.name.removeprefix("run_"),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds") if stat else None,
                    "bytes": directory_size_bytes(result_dir) if include_bytes else None,
                    "metric_files": metric_files,
                    "plot_metric_rows": sum(metric_row_counts.values()),
                    "warning": (
                        "Este elemento alimenta los plots. Al borrarlo desaparecera de los plots tras recargar."
                    ),
                }
            )
    return sorted(
        records,
        key=lambda item: (
            str(item.get("method") or ""),
            int(item.get("dataset_size") or 0),
            str(item.get("dataset_label") or ""),
            str(item.get("run_id") or ""),
        ),
    )


def plot_metric_dataset_output_records() -> dict[str, Any]:
    return {
        "targets": plot_metric_dataset_records(),
        "scope": "plot_metric_runs",
        "roots": {
            "md_results": str(RESULTS_ROOT / "results_md"),
            "atom_results": str(RESULTS_ROOT / "results_atomdisp"),
            "random_cartesian_results": str(RESULTS_ROOT / "results_random_cartesian"),
        },
    }


def deeph_comparison_plot_runs() -> list[dict[str, Any]]:
    """Expose fair Graph2Mat-vs-DeepH aggregate rows to the normal plot feed."""
    runs: list[dict[str, Any]] = []
    benchmark_root = RESULTS_ROOT / "graphene_w90_deeph_fair_benchmark"
    if not benchmark_root.exists():
        return runs
    for aggregate_path in sorted(benchmark_root.glob("**/comparison/aggregate_graph2mat_vs_deeph.csv")):
        comparison_dir = aggregate_path.parent
        run_dir = comparison_dir.parent
        rows = read_csv_rows(aggregate_path)
        manifest_path = comparison_dir / "comparison_manifest.json"
        comparison_manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                comparison_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                comparison_manifest = {}
        metric_manifest_path = run_dir / "eval" / "metrics" / "manifest.json"
        metric_manifest = load_json_object(metric_manifest_path)
        metric_safety = metric_manifest_safety_summary(metric_manifest)
        raw_manifest_path = run_dir / "raw_prepare" / "deeph_raw_manifest.json"
        raw_manifest: dict[str, Any] = {}
        if raw_manifest_path.exists():
            try:
                raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
            except Exception:
                raw_manifest = {}
        graph2mat_result_dir = Path(
            str(
                comparison_manifest.get("graph2mat_result_dir")
                or metric_manifest.get("graph2mat_result_dir")
                or ""
            )
        )
        graph2mat_manifest: dict[str, Any] = {}
        if graph2mat_result_dir:
            graph2mat_manifest_path = graph2mat_result_dir / "manifest.json"
            if graph2mat_manifest_path.exists():
                try:
                    graph2mat_manifest = json.loads(graph2mat_manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    graph2mat_manifest = {}
        benchmark_dataset_size = int(
            graph2mat_manifest.get("requested_dataset_size")
            or graph2mat_manifest.get("dataset_size")
            or raw_manifest.get("samples_requested")
            or raw_manifest.get("samples_ready")
            or 0
        )
        for row in rows:
            method = str(row.get("method") or row.get("pipeline") or "").strip()
            if not method:
                continue
            samples_compared = int(row.get("samples_compared") or row.get("test_samples") or 0)
            plot_dataset_size = benchmark_dataset_size or samples_compared
            kpoint_matrix_row = {
                "sample": f"{method}_aggregate",
                "h_mae_eV": row.get("h_mae_eV_mean"),
                "h_rmse_eV": row.get("h_rmse_eV_mean"),
                "relative_frobenius": row.get("relative_frobenius_mean"),
            }
            kpoint_spectral_row = {
                "sample": f"{method}_aggregate",
                "global_rmse_eV": row.get("global_rmse_eV_mean"),
                "low_energy_rmse_eV": row.get("low_energy_rmse_eV_mean"),
                "fermi_window_rmse_eV": row.get("fermi_window_rmse_eV_mean"),
                "frontier_window_rmse_eV": row.get("frontier_window_rmse_eV_mean"),
            }
            kpoint_dos_row = {
                "sample": f"{method}_aggregate",
                "dos_mae_500_fermi_window": row.get("dos_mae_500_fermi_window_mean"),
                "dos_wasserstein_eV": row.get("dos_wasserstein_eV_mean"),
            }
            samples_by_group = {
                "sparse": [],
                "spectral": [],
                "dos": [],
                "kpoint_matrix": [kpoint_matrix_row],
                "kpoint_matrix_per_k": [],
                "kpoint_spectral": [kpoint_spectral_row],
                "kpoint_dos": [kpoint_dos_row],
                "sparse_sweep": [],
                "dos_sweep": [],
                "matrix_spectrum": [],
                "orbital_pair_summary": [],
            }
            run_id = f"{run_dir.name}_{method}"
            runs.append(
                {
                    "pipeline": "deeph_comparison",
                    "method_id": method,
                    "label": "Graph2Mat vs DeepH",
                    "dataset_size": plot_dataset_size,
                    "effective_dataset_size": plot_dataset_size,
                    "requested_dataset_size": plot_dataset_size,
                    "run_id": run_id,
                    "result_dir": str(comparison_dir),
                    "dataset_label": f"{run_dir.name} · {method}",
                    "training_tag": method,
                    "training_plan_label": method,
                    "training_plan_display_label": method,
                    "material_label": "graphene",
                    "material_display_label": "graphene",
                    "metric_space": "kpoint_sampled",
                    **metric_safety,
                    "metric_manifest_path": str(metric_manifest_path),
                    "kpoint_metrics_enabled": True,
                    "kpoint_sampled_supported": True,
                    "kpoint_mesh": metric_manifest.get("kpoint_mesh") or comparison_manifest.get("kpoint_mesh"),
                    "kpoint_count": metric_manifest.get("kpoint_count") or comparison_manifest.get("kpoint_count"),
                    "kpoint_source": metric_manifest.get("kpoint_source") or "DeepH fair benchmark",
                    "uses_reference_overlap_k": True,
                    "complex_hamiltonians_supported_for_kpoint_metrics": True,
                    "means": {
                        "run": {},
                        "sparse": {},
                        "spectral": {},
                        "dos": {},
                        "kpoint_matrix": numeric_means([kpoint_matrix_row]),
                        "kpoint_matrix_per_k": {},
                        "kpoint_spectral": numeric_means([kpoint_spectral_row]),
                        "kpoint_dos": numeric_means([kpoint_dos_row]),
                        "sparse_sweep": {},
                        "dos_sweep": {},
                        "matrix_spectrum": {},
                        "orbital_pair_summary": {},
                    },
                    "samples": plot_sample_payload(samples_by_group),
                    "sample_row_counts": plot_sample_row_counts(samples_by_group),
                    "diagnostics": {
                        "severity": "info",
                        "status": "fair_deeph_comparison",
                        "warnings": [
                            "Aggregate Graph2Mat-vs-DeepH rows are plotted as one point per method; per-sample DeepH details live in the comparison directory."
                        ],
                    },
                    "metric_availability": {
                        "spectral": {},
                        "kpoint_matrix": metric_availability_for_rows(
                            [kpoint_matrix_row],
                            ["h_mae_eV", "h_rmse_eV", "relative_frobenius"],
                        ),
                        "kpoint_spectral": metric_availability_for_rows(
                            [kpoint_spectral_row],
                            SPECTRAL_PLOT_AVAILABILITY_METRICS + ["global_rmse_eV", "gap_abs_error_eV"],
                        ),
                        "kpoint_dos": metric_availability_for_rows(
                            [kpoint_dos_row],
                            ["dos_wasserstein_eV", "dos_mae_500_fermi_window"],
                        ),
                    },
                    "summary": {
                        "source": str(aggregate_path),
                        "comparison_manifest": str(manifest_path),
                        "test_samples_compared": samples_compared,
                        "plot_x_dataset_size": plot_dataset_size,
                    },
                    "diagnostic_outputs": {
                        "aggregate_graph2mat_vs_deeph": {
                            "exists": True,
                            "path": str(aggregate_path),
                        },
                        "final_report": {
                            "exists": (comparison_dir / "final_report.md").exists(),
                            "path": str(comparison_dir / "final_report.md"),
                        },
                    },
                }
            )
    return runs


def directory_size_bytes(path: Path) -> int:
    if path.is_symlink() or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def safe_remove_repo_target(relative_path: str) -> None:
    if not relative_path:
        raise RuntimeError("Registro de cleanup sin relative_path.")
    repo_root = REPO_ROOT.resolve(strict=False)
    target = (REPO_ROOT / relative_path).resolve(strict=False)
    try:
        target.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError(f"Ruta de cleanup fuera del repositorio: {relative_path}") from exc
    if target == repo_root:
        raise RuntimeError(f"No se borra la raiz del repositorio: {relative_path}")
    if target.is_symlink():
        raise RuntimeError(f"No se borra un enlace simbolico: {relative_path}")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def clear_selected_plot_metric_dataset_outputs(
    target_ids: list[str],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    records = plot_metric_dataset_records()
    by_id = {str(record["id"]): record for record in records}
    normalized_ids = list(dict.fromkeys(str(target_id) for target_id in target_ids if str(target_id).strip()))
    unknown = [target_id for target_id in normalized_ids if target_id not in by_id]
    if unknown:
        raise RuntimeError(f"IDs de run de plots no reconocidos o ya borrados: {unknown}.")
    selected = [by_id[target_id] for target_id in normalized_ids]
    removed = [str(record["relative_path"]) for record in selected]
    if not dry_run:
        for record in selected:
            safe_remove_repo_target(str(record["relative_path"]))
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "removed": removed,
        "selected": selected,
        "scope": "plot_metric_runs",
        "preserved": [
            "source datasets outside the selected plot-visible run dirs",
            "Comparison/workspaces",
            "code, configs and pseudopotentials",
        ],
    }


def clear_selected_generated_dataset_outputs(
    target_ids: list[str],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    if (
        all_status().get("running")
        or EXPERIMENT_RUNNER.status().get("running")
        or G2M_DEEPH_RUNNER.status().get("running")
    ):
        raise RuntimeError("No se pueden borrar datasets mientras hay pipelines o experimentos en ejecucion.")
    plot_ids = {str(record["id"]) for record in plot_metric_dataset_records()}
    normalized_ids = list(dict.fromkeys(str(target_id) for target_id in target_ids if str(target_id).strip()))
    if normalized_ids and all(target_id in plot_ids for target_id in normalized_ids):
        return clear_selected_plot_metric_dataset_outputs(normalized_ids, dry_run=dry_run)
    return cleanup_selected_generated_datasets(
        REPO_ROOT,
        target_ids=normalized_ids,
        dry_run=dry_run,
    )


def run_all(*, venv_activate_command: str | None = None) -> dict[str, Any]:
    venv_activate_path = apply_venv_activate_to_pipeline_configs(venv_activate_command)
    started: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key, runner in RUNNERS.items():
        try:
            started[key] = runner.start()
        except Exception as exc:
            errors[key] = str(exc)
    payload = all_status()
    payload["started"] = started
    payload["errors"] = errors
    payload["venv_activate"] = venv_activate_path
    if errors and not started:
        raise RuntimeError("; ".join(errors.values()))
    return payload


def stop_all() -> dict[str, Any]:
    for runner in RUNNERS.values():
        runner.stop()
    MIXING_E2E_RUNNER.stop()
    return all_status()


def result_summary() -> dict[str, Any]:
    md_predictions = sorted((REPO_ROOT / "MD" / "dataset" / "MD_steps").glob("*/ML_prediction.HSX"))
    atdisp_predictions = sorted(
        (REPO_ROOT / "AtomDisplacement" / "dataset" / "samples").glob("*/ML_prediction.HSX")
    )
    random_predictions = sorted(
        (REPO_ROOT / "AtomDisplacement" / "dataset" / "RandomCartesian_steps").glob("*/ML_prediction.HSX")
    )
    archived = archived_results_summary()
    return {
        "md": {
            "root": str(REPO_ROOT / "MD"),
            "metrics": str(REPO_ROOT / "MD" / "training" / "sample_metrics.csv"),
            "metrics_exists": (REPO_ROOT / "MD" / "training" / "sample_metrics.csv").exists(),
            "predictions": len(md_predictions),
            "prediction_glob": "MD/dataset/MD_steps/*/ML_prediction.HSX",
        },
        "atom_displacement": {
            "root": str(REPO_ROOT / "AtomDisplacement"),
            "metrics": str(REPO_ROOT / "AtomDisplacement" / "training" / "sample_metrics.csv"),
            "metrics_exists": (REPO_ROOT / "AtomDisplacement" / "training" / "sample_metrics.csv").exists(),
            "predictions": len(atdisp_predictions),
            "prediction_glob": "AtomDisplacement/dataset/samples/*/ML_prediction.HSX",
        },
        "random_cartesian": {
            "root": str(REPO_ROOT / "AtomDisplacement"),
            "metrics": str(REPO_ROOT / "AtomDisplacement" / "training" / "sample_metrics.csv"),
            "metrics_exists": (REPO_ROOT / "AtomDisplacement" / "training" / "sample_metrics.csv").exists(),
            "predictions": len(random_predictions),
            "prediction_glob": "AtomDisplacement/dataset/RandomCartesian_steps/*/ML_prediction.HSX",
        },
        "archived": archived,
    }


def archived_results_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {"md": [], "atom_displacement": [], "random_cartesian": []}
    groups = {
        "md": RESULTS_ROOT / "results_md",
        "atom_displacement": RESULTS_ROOT / "results_atomdisp",
        "random_cartesian": RESULTS_ROOT / "results_random_cartesian",
    }
    for key, root in groups.items():
        if not root.exists():
            continue
        for manifest_path in archived_result_manifest_paths(root):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            result_dir = archived_manifest_result_dir(manifest, manifest_path)
            item = dict(manifest)
            metric_manifest_path = result_dir / "metrics" / "manifest.json"
            metric_manifest = load_json_object(metric_manifest_path)
            safety_summary = metric_manifest_safety_summary(metric_manifest)
            for field, value in safety_summary.items():
                if value in (None, "", "unknown") and item.get(field) not in (None, ""):
                    continue
                item[field] = value
            item["metric_manifest_path"] = str(metric_manifest_path)
            item.setdefault("run_id", result_dir.name.removeprefix("run_"))
            item.setdefault("result_dir", str(result_dir))
            item.setdefault("dataset_label", result_dir.parent.name)
            item["diagnostic_outputs"] = archived_run_diagnostic_outputs(result_dir)
            summary[key].append(item)
    return summary


def archived_result_manifest_paths(root: Path) -> list[Path]:
    """Return archived run manifests for both legacy and recipe-based dataset names."""
    selected: dict[Path, Path] = {}
    for metrics_manifest in list(root.glob("*/run_*/metrics/manifest.json")) + list(root.glob("run_*/metrics/manifest.json")):
        selected[metrics_manifest.parent.parent.resolve()] = metrics_manifest
    for run_manifest in list(root.glob("*/run_*/manifest.json")) + list(root.glob("run_*/manifest.json")):
        selected[run_manifest.parent.resolve()] = run_manifest
    return sorted(selected.values())


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(
        json_safe(payload),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error_response(handler: BaseHTTPRequestHandler, exc: Exception, status: int = 400) -> None:
    json_response(handler, {"error": str(exc)}, status=status)


def parse_query_int(
    query: dict[str, list[str]],
    key: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    raw_value = query.get(key, [str(default)])[0]
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def bounded_log_payload(
    logs: list[str],
    *,
    since: int = 0,
    limit: int | None = DEFAULT_LOG_RESPONSE_LIMIT,
) -> dict[str, Any]:
    """Return a bounded log slice so the browser never downloads huge histories."""
    total = len(logs)
    requested_since = max(0, int(since or 0))
    start = min(requested_since, total)
    end = total
    effective_limit = (
        None
        if limit is None or int(limit) <= 0
        else min(int(limit), MAX_LOG_RESPONSE_LIMIT)
    )
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
                "[UI] Log truncado en la respuesta web: "
                f"se omitieron {dropped_lines} lineas antiguas. "
                "Los artefactos archivados conservan el run.log completo.\n"
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


# --------------------------------------------------------------------------- #
# ML vs SIESTA benchmark backend (thin bridge to the ml_vs_siesta package).
# All handlers are lightweight: no SIESTA, no training, no heavy model loading.
# --------------------------------------------------------------------------- #
ML_VS_SIESTA_EXAMPLE_CONFIG = COMPARISON_ROOT / "config" / "ml_vs_siesta_benchmark_example.yaml"
ML_VS_SIESTA_DEFAULT_OUTPUT = RESULTS_ROOT / "ml_vs_siesta_displacements"


def _ml_vs_siesta_module():
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import ml_vs_siesta  # noqa: PLC0415 - lazy import keeps startup light

    return ml_vs_siesta


def ml_vs_siesta_config_template_payload() -> dict[str, Any]:
    mvs = _ml_vs_siesta_module()
    config = mvs.load_benchmark_config(ML_VS_SIESTA_EXAMPLE_CONFIG)
    return {
        "config": config.to_dict(),
        "dry_run": mvs.benchmark_dry_run(config, siesta_output_dir=None),
        "example_config_path": str(ML_VS_SIESTA_EXAMPLE_CONFIG.relative_to(REPO_ROOT)),
    }


def ml_vs_siesta_dry_run_payload(body: dict[str, Any]) -> dict[str, Any]:
    mvs = _ml_vs_siesta_module()
    config = mvs.parse_benchmark_config(body.get("config") or {})
    return mvs.benchmark_dry_run(config, siesta_output_dir=body.get("output_dir"))


def ml_vs_siesta_generate_payload(body: dict[str, Any]) -> dict[str, Any]:
    mvs = _ml_vs_siesta_module()
    config = mvs.parse_benchmark_config(body.get("config") or {})
    dry_run = parse_bool(body.get("dry_run"), True)
    output_dir = Path(body.get("output_dir") or ML_VS_SIESTA_DEFAULT_OUTPUT)
    if not dry_run:
        resolved = output_dir.expanduser().resolve()
        results_root = RESULTS_ROOT.resolve()
        if resolved != results_root and results_root not in resolved.parents:
            raise RuntimeError(
                "output_dir must be under Comparison/results when dry_run is false."
            )
    return mvs.generate_siesta_displacement_inputs(config, output_dir, dry_run=dry_run)


def ml_vs_siesta_mix_payload(body: dict[str, Any]) -> dict[str, Any]:
    mvs = _ml_vs_siesta_module()
    small = body.get("small") or []
    large = body.get("large") or []
    ratios = body.get("ratios") or [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    mode = body.get("mode") or "add"
    seed = int(body.get("seed") or 0)
    return mvs.make_mixed_dataset_manifest(small, large, ratios=ratios, mode=mode, seed=seed)


def ml_vs_siesta_inspect_species_payload(body: dict[str, Any]) -> dict[str, Any]:
    mvs = _ml_vs_siesta_module()
    config = body.get("config") or {}
    report = mvs.inspect_species_support(config, new_species=body.get("new_species"))
    return report.to_dict()


def ml_vs_siesta_matrix_viewer_demo_payload() -> dict[str, Any]:
    """Deterministic fake payload so the Matrix Viewer renders without SIESTA."""
    mvs = _ml_vs_siesta_module()
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(0)
    n = 12
    base = rng.standard_normal((n, n))
    base = (base + base.T) / 2.0
    siesta = mvs.MatrixData(values=base, target="hamiltonian")
    graph2mat = mvs.MatrixData(
        values=base + 0.05 * rng.standard_normal((n, n)), target="hamiltonian"
    )
    deeph = mvs.MatrixData(
        values=base + 0.12 * rng.standard_normal((n, n)), target="hamiltonian"
    )
    payload = mvs.build_matrix_viewer_payload(
        target="hamiltonian", siesta=siesta, graph2mat=graph2mat, deeph=deeph
    )
    payload["note"] = "Synthetic demo payload (no SIESTA run); wire real matrices later."
    return payload


# --------------------------------------------------------------------------- #
# Mixing datasets sweep backend (small + 5x5x1 large -> merged -> MAE vs size).
# discover/plan are synchronous; launch runs in a background thread. Training is
# never auto-launched here: launch materializes merged datasets (dry-run/preview
# by default). Real training is wired via run_mixing_sweep(launch_fn=...).
# --------------------------------------------------------------------------- #
DATASETS_ROOT = COMPARISON_ROOT / "datasets"
MIXING_SWEEP_OUTPUT_ROOT = RESULTS_ROOT / "ml_vs_siesta_mixing_sweep"
MIXING_SMALL_ATOM_THRESHOLD = 10


def _display_path(path: Path) -> str:
    """Repo-relative path when possible, else the absolute path."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def mixing_discover_payload(threshold_atoms: int = MIXING_SMALL_ATOM_THRESHOLD) -> dict[str, Any]:
    mvs = _ml_vs_siesta_module()
    small: list[dict[str, Any]] = []
    large: list[dict[str, Any]] = []
    seen: set[str] = set()
    if DATASETS_ROOT.exists():
        for manifest_path in sorted(DATASETS_ROOT.rglob("frozen_split_manifest.json")):
            dataset_root = manifest_path.parent
            key = str(dataset_root)
            if key in seen:
                continue
            seen.add(key)
            try:
                samples = mvs.read_dataset_samples(dataset_root)
                n_atoms = mvs.dataset_atom_count(dataset_root)
            except Exception:  # noqa: BLE001 - skip unreadable datasets
                continue
            entry = {
                "root": _display_path(dataset_root),
                "n_snapshots": len(samples),
                "n_atoms": n_atoms,
            }
            if n_atoms is not None and n_atoms >= threshold_atoms:
                large.append(entry)
            else:
                small.append(entry)
    return {
        "threshold_atoms": threshold_atoms,
        "small": small,
        "large": large,
        "datasets_root": _display_path(DATASETS_ROOT),
    }


def _mixing_roots_from_body(body: dict[str, Any], key: str) -> dict[int, str]:
    raw = body.get(key) or {}
    result: dict[int, str] = {}
    for size, root in raw.items():
        resolved = Path(root)
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        result[int(size)] = str(resolved)
    return result


def mixing_plan_payload(body: dict[str, Any]) -> dict[str, Any]:
    mvs = _ml_vs_siesta_module()
    small = _mixing_roots_from_body(body, "small")
    large = _mixing_roots_from_body(body, "large")
    modes = tuple(body.get("modes") or ("add", "replace"))
    ratios = tuple(float(r) for r in (body.get("ratios") or (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)))
    sizes = [int(s) for s in body["sizes"]] if body.get("sizes") else None
    seed = int(body.get("seed") or 0)
    split_policy = str(body.get("split_policy") or "fixed_common_test")
    return mvs.plan_mixing_sweep_from_roots(
        small, large, sizes=sizes, modes=modes, ratios=ratios, seed=seed,
        split_policy=split_policy,
    )


def mixing_metrics_demo_payload() -> dict[str, Any]:
    """Synthetic MAE-vs-size curves so the chart renders before real training."""
    mvs = _ml_vs_siesta_module()
    records: list[dict[str, Any]] = []
    for size in (20, 40, 60, 100):
        for mode in ("add", "replace"):
            for ratio in (0.0, 0.5, 1.0):
                total = size + (round(ratio * size) if mode == "add" else 0)
                for model, base in (("graph2mat", 0.9), ("deeph", 1.4)):
                    mae = base / (total ** 0.5) * (1.0 - 0.15 * ratio)
                    records.append(
                        {
                            "size": size,
                            "mode": mode,
                            "ratio": ratio,
                            "total_size": total,
                            "model": model,
                            "h_mae_eV": round(mae, 5),
                        }
                    )
    payload = mvs.aggregate_mae_vs_size(records)
    payload["note"] = "Synthetic demo curves (no training); replace with real sweep records."
    return payload


def _mixing_payload_id(item: dict[str, Any]) -> str:
    """Stable id for a mixing payload, derived from (size, mode, ratio).

    This triple is the true identity of a mixing permutation (the
    materialized output_root is a deterministic function of it), so it must
    be computed the same way regardless of whether the caller is a bare
    plan/permutation dict, a live training record, or a persisted summary.
    Using output_root as the id instead would vary across machines/tmp dirs
    and fail to merge with canonical ids from other sources.
    """
    explicit = item.get("payload_id")
    if explicit:
        return str(explicit)
    size, mode, ratio = item.get("size"), item.get("mode"), item.get("ratio")
    if size is not None and mode is not None and ratio is not None:
        mvs = _ml_vs_siesta_module()
        return f"size{int(size)}_{mode}_{mvs.mixing_sweep._ratio_slug(float(ratio))}"
    return str(item.get("output_root") or "")


def _mixing_payloads_from_permutations(permutations: Any) -> list[dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    if not isinstance(permutations, list):
        return []
    for item in permutations:
        if not isinstance(item, dict):
            continue
        payload_id = _mixing_payload_id(item)
        payloads[payload_id] = {
            "id": payload_id,
            "label": f"size={item.get('size')} {item.get('mode')} ratio={item.get('ratio')}",
            "size": item.get("size"),
            "mode": item.get("mode"),
            "ratio": item.get("ratio"),
            "total_size": item.get("total_size"),
            "status": item.get("status"),
            "output_root": item.get("output_root"),
        }
    return sorted(
        payloads.values(),
        key=lambda item: (
            int(item["size"] or 0),
            str(item["mode"] or ""),
            float(item["ratio"] or 0),
            str(item["id"]),
        ),
    )


def _mixing_metrics_payload(records: list[dict[str, Any]], summary: dict[str, Any] | None = None) -> dict[str, Any]:
    mvs = _ml_vs_siesta_module()
    payload = mvs.aggregate_mae_vs_size(records)
    by_id = {str(item["id"]): item for item in payload.get("payloads", [])}
    metric_ids = set(by_id)
    for item in _mixing_payloads_from_permutations((summary or {}).get("permutations")):
        key = str(item["id"])
        if key in by_id:
            # aggregate_mae_vs_size's entry (has real metrics) wins on overlap;
            # fields unique to the permutation, like "status", still fill in.
            by_id[key] = {**item, **by_id[key]}
        else:
            by_id[key] = item
    for key, item in by_id.items():
        if key in metric_ids:
            item["status"] = "trained"
    payload["payloads"] = sorted(
        by_id.values(),
        key=lambda item: (
            int(item.get("size") or 0),
            str(item.get("mode") or ""),
            float(item.get("ratio") or 0),
            str(item.get("id") or ""),
        ),
    )
    return payload


_MIXING_TRAINING_POLL_SECONDS = 5.0


_NON_TEST_SPLITS = {"train", "training", "validation", "val"}
_TEST_SPLITS = {"test", "final_test", "final", "eval", "eval_input"}
# Path tokens that on their own prove a test split. "eval"/"eval_input" are
# generic directory names (a run evaluates train and validation too), so they
# are not proof: real eval_input dirs ship a manifest declaring split=test,
# which is read first. Guessing "test" from the directory name alone would let
# a train average masquerade as a test MAE.
_TEST_SPLIT_PATH_TOKENS = {"test", "final_test", "final"}


def _extract_model_h_mae_eV(results_payload: Any, models: tuple[str, ...]) -> dict[str, Any]:
    """Per-model ``h_mae_eV`` extraction from a runner results dict.

    Recursively walks the results/plot payload looking for nodes that carry both
    a method/model identifier and an ``h_mae_eV`` value. Candidates whose
    nearest ``split`` context is train/validation are discarded (the mixing
    curves are test-split metrics). If a model has multiple *distinct* surviving
    values the extraction fails loudly instead of silently picking one.
    Returns ``{model: {"h_mae_eV": float}}``; models without a usable metric are
    simply omitted (run_mixing_sweep skips records without an MAE).
    """
    models_lower = {m.lower(): m for m in models}
    # model -> list of (value, json_path)
    candidates: dict[str, list[tuple[float, str]]] = {}

    def visit(node: Any, method_hint: str | None, split_hint: str | None, path: str) -> None:
        if isinstance(node, dict):
            method = node.get("method") or node.get("model") or method_hint
            split = str(node.get("split") or split_hint or "").lower() or None
            value = node.get("h_mae_eV")
            key = str(method or "").lower()
            if value is not None and key in models_lower and split not in _NON_TEST_SPLITS:
                try:
                    candidates.setdefault(models_lower[key], []).append(
                        (float(value), f"{path}.h_mae_eV")
                    )
                except (TypeError, ValueError):
                    pass
            for child_key, child in node.items():
                child_hint = child_key if str(child_key).lower() in models_lower else method
                child_split = split
                if str(child_key).lower() in (_NON_TEST_SPLITS | _TEST_SPLITS):
                    child_split = str(child_key).lower()
                visit(child, child_hint, child_split, f"{path}.{child_key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, method_hint, split_hint, f"{path}[{index}]")

    visit(results_payload, None, None, "$")

    found: dict[str, Any] = {}
    for model, entries in candidates.items():
        distinct: list[tuple[float, str]] = []
        for value, path in entries:
            if not any(math.isclose(value, seen, rel_tol=1e-9, abs_tol=1e-15) for seen, _ in distinct):
                distinct.append((value, path))
        if len(distinct) > 1:
            detail = "; ".join(f"{path}={value!r}" for value, path in distinct)
            raise RuntimeError(
                f"Ambiguous h_mae_eV for model {model!r}: multiple distinct values in "
                f"runner results ({detail}). Refusing to pick one silently."
            )
        found[model] = {"h_mae_eV": distinct[0][0]}
    return found


def _mixing_run_ok(status: Any) -> bool:
    """Whether a terminal runner status represents a successful training run.

    Mirrors the runner's own convention (returncode 0 or 2 == ok) and treats an
    error message or a stop request as failure.
    """
    if not isinstance(status, dict):
        return False
    if status.get("error"):
        return False
    if status.get("stop_requested"):
        return False
    returncode = status.get("returncode")
    if returncode is not None:
        try:
            if int(returncode) not in (0, 2):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _mixing_metrics_from_common_csv(run_root: Any, models: tuple[str, ...]) -> dict[str, Any]:
    if not run_root:
        return {}
    path = Path(str(run_root)) / "common_metrics" / "summary" / "common_method_metrics.csv"
    if not path.is_file():
        return {}
    wanted = {model.lower(): model for model in models}
    found: dict[str, Any] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = str(row.get("method") or row.get("model") or "").lower()
            value = row.get("h_mae_eV") or row.get("h_mae_eV_mean")
            if model not in wanted or value in (None, ""):
                continue
            try:
                metrics = {"h_mae_eV": float(value)}
                rel = row.get("relative_frobenius") or row.get("relative_frobenius_mean")
                if rel not in (None, ""):
                    metrics["relative_frobenius"] = float(rel)
                found[wanted[model]] = metrics
            except ValueError:
                pass
    return found


def _mixing_csv_split_evidence(path: Path, root: Path) -> str | None:
    """Best-effort split provenance for a metrics CSV: manifest first, then path.

    Returns "test", a non-test split name, or None when no evidence exists.
    """
    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            split = str(manifest.get("split") or "").lower()
            if split:
                return split
        except (OSError, json.JSONDecodeError):
            pass
    try:
        parts = [part.lower() for part in path.relative_to(root).parts]
    except ValueError:
        parts = [part.lower() for part in path.parts]
    for part in parts:
        tokens = set(re.split(r"[^a-z]+", part))
        if tokens & _NON_TEST_SPLITS:
            return "train_or_validation"
        if tokens & _TEST_SPLIT_PATH_TOKENS:
            return "test"
    return None


def _mixing_sample_domain(sample_id: Any) -> str | None:
    """"small"/"large" from a merged sample id, or None if it carries no prefix.

    Merged datasets prefix every sample with the ``small__``/``large__`` tags
    from ``mixing_sweep``, so the structure domain survives into the per-sample
    metrics CSVs and needs no extra bookkeeping.
    """
    mvs = _ml_vs_siesta_module()
    text = str(sample_id or "")
    for prefix, domain in (
        (mvs.mixing_sweep._SMALL_PREFIX, "small"),
        (mvs.mixing_sweep._LARGE_PREFIX, "large"),
    ):
        if text.startswith(prefix):
            return domain
    return None


def _mixing_domain_reduction(by_domain: dict[str, list[float]]) -> dict[str, Any]:
    """Per-domain H-MAE plus the macro/worst reductions a mixing curve needs.

    A ratio sweep trades small-domain error against large-domain error, so the
    pooled mean alone cannot show the trade-off: it moves simply because the
    test composition moves. Mirrors the derivative metrics' by_domain reduction.
    """
    if not by_domain:
        return {}
    means = {
        domain: sum(vals) / len(vals) for domain, vals in sorted(by_domain.items())
    }
    return {
        "h_mae_eV_by_domain": means,
        "h_mae_eV_macro_domain": sum(means.values()) / len(means),
        "h_mae_eV_worst_domain": max(means.values()),
    }


def _mixing_domain_metrics(run_root: Any, model: str) -> dict[str, Any]:
    """Just the per-domain reductions for a run, or {} when unavailable."""
    metrics = (_mixing_metrics_from_run_metrics(run_root, model) or {}).get(model) or {}
    return {k: v for k, v in metrics.items() if k.startswith("h_mae_eV_")}


def _mixing_metrics_from_run_metrics(run_root: Any, model: str) -> dict[str, Any]:
    """Last-resort ``h_mae_eV`` from kpoint CSVs, restricted to test-split evidence.

    A CSV is used only when a sibling ``manifest.json`` or its path marks it as
    test/final split; train/validation CSVs are skipped so mixed-split averages
    can never masquerade as a test MAE. Only ``weighted_sample`` rows count when
    the ``row_type`` column exists (per-k rows would double count). No evidence
    at all -> the CSV is skipped and the caller reports the metric as missing.
    """
    if not run_root:
        return {}
    root = Path(str(run_root))
    values: list[float] = []
    rel_values: list[float] = []
    by_domain: dict[str, list[float]] = {}
    for path in sorted(root.rglob("kpoint_matrix_metrics.csv")):
        evidence = _mixing_csv_split_evidence(path, root)
        if evidence not in _TEST_SPLITS:
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row_type = row.get("row_type")
                    if row_type is not None and row_type != "weighted_sample":
                        continue
                    value = row.get("h_mae_eV")
                    if value in (None, ""):
                        continue
                    values.append(float(value))
                    domain = _mixing_sample_domain(row.get("sample"))
                    if domain:
                        by_domain.setdefault(domain, []).append(float(value))
                    rel = row.get("relative_frobenius")
                    if rel not in (None, ""):
                        rel_values.append(float(rel))
        except (OSError, ValueError):
            continue
    if not values:
        return {}
    metrics = {"h_mae_eV": sum(values) / len(values)}
    if rel_values:
        metrics["relative_frobenius"] = sum(rel_values) / len(rel_values)
    metrics.update(_mixing_domain_reduction(by_domain))
    return {model: metrics}


def _persist_mixing_summary(output_root: Path, summary: dict[str, Any]) -> None:
    """Write the canonical mixing summary (and chart payload) for UI consumers."""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "mixing_sweep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    records = summary.get("records") or []
    if records:
        mvs = _ml_vs_siesta_module()
        mvs.write_mae_vs_size_outputs(records, output_root, write_png=True)


def _mixing_runner_launch_fn(payload: dict[str, Any]) -> dict[str, Any]:
    """Synchronously drive the real Graph2Mat/DeepH runner for one merged dataset.

    Starts the runner on the merged ``dataset_root`` payload built by
    ``run_mixing_sweep``, blocks until training finishes, then extracts per-model
    ``h_mae_eV``. This spawns real training subprocesses (Graph2Mat / DeepH) and
    therefore needs the models installed and, for a full sweep, a GPU.

    Returns ``ok=False`` (with an ``error``) when the runner failed or produced no
    ``h_mae_eV`` for the requested models, so a failed permutation is never
    silently reported as "trained".
    """
    runner = Graph2MatDeepHBenchmarkRunner()
    models = tuple(payload.get("models") or payload.get("selected_methods") or ("graph2mat", "deeph"))
    runner.start(payload)
    while True:
        status = runner.status()
        if not status.get("running"):
            break
        time.sleep(_MIXING_TRAINING_POLL_SECONDS)
    results = runner.results()
    runner_status = results.get("status") or {}
    metrics = _extract_model_h_mae_eV(results, models)
    if len(metrics) < len(models):
        metrics.update(_mixing_metrics_from_common_csv(runner_status.get("run_root"), models))
    ok = _mixing_run_ok(runner_status)
    error: str | None = runner_status.get("error") if isinstance(runner_status, dict) else None
    missing_models = [model for model in models if model not in metrics]
    if ok and missing_models:
        ok = False
        error = error or f"runner produced no h_mae_eV for models: {missing_models}"
    return {
        "ok": ok,
        "error": error,
        "metrics": metrics,
        "missing_models": missing_models,
        "runner_status": runner_status,
    }


def _apply_training_weighting_policy(
    overrides: dict[str, Any],
    model: str,
    policy: str,
    domain_weighting: dict[str, Any] | None,
) -> None:
    """Map training_weighting_policy (audit Fase 9) to per-model overrides.

    graph2mat: loss=block_type_mse_per_structure/_per_domain (+ loss_kwargs).
    deeph: hyperparameter.training_weighting_policy (+ domain options).
    Explicit user overrides in ``hyperparams`` win (setdefault only).
    """
    if policy in ("", None, "legacy_elementwise"):
        return
    domain = dict(domain_weighting or {})
    if model == "graph2mat":
        if policy == "per_structure":
            overrides.setdefault("loss", "graph2mat.core.data.metrics.block_type_mse_per_structure")
        else:
            overrides.setdefault("loss", "graph2mat.core.data.metrics.block_type_mse_per_domain")
            overrides.setdefault(
                "loss_kwargs",
                {
                    "domain_threshold_atoms": int(domain.get("domain_threshold_atoms", 0)),
                    "small_domain_weight": float(domain.get("small_domain_weight", 0.5)),
                    "large_domain_weight": float(domain.get("large_domain_weight", 0.5)),
                },
            )
    elif model == "deeph":
        overrides.setdefault("training_weighting_policy", policy)
        if policy == "per_domain":
            overrides.setdefault(
                "domain_threshold_atoms", int(domain.get("domain_threshold_atoms", 0))
            )
            overrides.setdefault(
                "small_domain_weight", float(domain.get("small_domain_weight", 0.5))
            )
            overrides.setdefault(
                "large_domain_weight", float(domain.get("large_domain_weight", 0.5))
            )


def _run_mixing_sweep_parallel(
    small: dict[int, str],
    large: dict[int, str],
    output_root: Path,
    *,
    sizes: list[int] | None,
    modes: tuple[str, ...],
    ratios: tuple[float, ...],
    seed: int,
    models: tuple[str, ...],
    epochs: int | None,
    performance: dict[str, Any] | None,
    split_policy: str = "fixed_common_test",
    split_fractions: tuple[float, float, float] | None = None,
    temporal_gap: int | None = None,
    confirm_ghost_species_exemption: bool = False,
    hyperparams: dict[str, dict[str, Any]] | None = None,
    training_weighting_policy: str = "legacy_elementwise",
    domain_weighting: dict[str, Any] | None = None,
    early_stopping: dict[str, Any] | None = None,
    reconstructed_records: list[dict[str, Any]] | None = None,
    progress_fn: Callable[[dict[str, Any]], None] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Materialize ALL permutation datasets, then train with ONE runner invocation.

    The runner handles g2m_parallelism / deeph_parallelism internally, so all
    permutations run with true parallel training (e.g. 7 Graph2Mat + 5 DeepH at
    a time) instead of one permutation at a time.
    """
    mvs = _ml_vs_siesta_module()
    training_weighting_policy = str(training_weighting_policy or "legacy_elementwise")
    if training_weighting_policy not in ("legacy_elementwise", "per_structure", "per_domain"):
        raise RuntimeError(
            f"Unknown training_weighting_policy {training_weighting_policy!r}; "
            "use legacy_elementwise, per_structure or per_domain."
        )
    if training_weighting_policy == "per_domain" and not (domain_weighting or {}).get(
        "domain_threshold_atoms"
    ):
        raise RuntimeError(
            "training_weighting_policy='per_domain' requires "
            "domain_weighting.domain_threshold_atoms in the payload."
        )
    if log_fn is not None:
        log_fn(
            "[MIXING] Materializando datasets: "
            f"sizes={sizes or sorted(small)} modes={list(modes)} ratios={list(ratios)} "
            f"split_policy={split_policy} training_weighting_policy={training_weighting_policy}\n"
        )
    reconstructed_records = list(reconstructed_records or [])
    reconstructed_models_by_key: dict[tuple[int, str, float], set[str]] = defaultdict(set)
    for record in reconstructed_records:
        try:
            reconstructed_models_by_key[
                (
                    int(record["size"]),
                    str(record["mode"]),
                    round(float(record["ratio"]), 6),
                )
            ].add(str(record["model"]))
        except (KeyError, TypeError, ValueError):
            continue
    reconstructed_keys = {
        key
        for key, reconstructed_models in reconstructed_models_by_key.items()
        if set(models).issubset(reconstructed_models)
    }

    # ── Phase 1: materialise all datasets ──────────────────────────────────────
    summary_permutations = mvs.run_mixing_sweep(
        small,
        large,
        output_root,
        sizes=sizes,
        modes=modes,
        ratios=ratios,
        seed=seed,
        models=models,
        epochs=epochs,
        performance=performance,
        split_policy=split_policy,
        split_fractions=split_fractions,
        temporal_gap=(DEFAULT_MD_TEMPORAL_GAP if temporal_gap is None else int(temporal_gap)),
        confirm_ghost_species_exemption=confirm_ghost_species_exemption,
        dry_run=False,
        launch_fn=None,          # materialise only, no training yet
        progress_fn=progress_fn,
        skip_permutation_keys=reconstructed_keys,
    )

    materialized = [
        p for p in (summary_permutations.get("permutations") or [])
        if p.get("status") == "materialized"
    ]
    for perm in materialized:
        key = (
            int(perm.get("size")),
            str(perm.get("mode")),
            round(float(perm.get("ratio")), 6),
        )
        if key in reconstructed_keys:
            perm["status"] = "reconstructed"
            perm["reconstructed"] = True
    trainable = [p for p in materialized if not p.get("reconstructed")]
    if log_fn is not None:
        reconstructed_count = sum(
            1
            for entry in (summary_permutations.get("permutations") or [])
            if entry.get("status") == "reconstructed"
        )
        log_fn(
            "[MIXING] Datasets materializados: "
            f"{len(materialized)}/{len(summary_permutations.get('permutations') or [])}. "
            f"Reconstruidos={reconstructed_count}. "
            "Arrancando entrenamiento Graph2Mat/DeepH.\n"
        )
    if not trainable:
        summary_permutations["n_trained"] = 0
        summary_permutations["records"] = reconstructed_records
        _persist_mixing_summary(output_root, summary_permutations)
        return summary_permutations

    # ── Phase 2: one runner invocation with ALL datasets ──────────────────────
    datasets_list = [
        {
            "dataset_id": f"perm_{i}",
            "dataset_root": p["output_root"],
        }
        for i, p in enumerate(trainable)
    ]

    thread_count = None
    if performance:
        thread_count = (
            performance.get("deeph_num_threads")
            or performance.get("torch_num_threads")
            or performance.get("omp_num_threads")
        )

    manual_runs: list[dict[str, Any]] = []
    for i, _perm in enumerate(trainable):
        dataset_id = f"perm_{i}"
        for model in models:
            overrides: dict[str, Any] = dict((hyperparams or {}).get(model) or {})
            # The runner only reads the training seed from overrides
            # (seed_everything / seed); the manual_run "seed" field below is
            # bookkeeping. The sweep seed must win over any payload
            # hyperparam, otherwise every replicate trains from an identical
            # init and the error bars only measure dataset selection.
            if model == "graph2mat":
                overrides["seed_everything"] = int(seed)
            elif model == "deeph":
                overrides["seed"] = int(seed)
            if epochs is not None:
                if model == "graph2mat":
                    overrides["max_epochs"] = int(epochs)
                elif model == "deeph":
                    overrides["epochs"] = int(epochs)
            if model == "deeph" and thread_count not in (None, "", "null"):
                overrides["num_threads"] = int(thread_count)
            _apply_training_weighting_policy(
                overrides, model, training_weighting_policy, domain_weighting
            )
            manual_runs.append(
                {
                    "model": model,
                    "dataset_id": dataset_id,
                    "config_id": f"{model}_{dataset_id}",
                    "seed": seed,
                    "overrides": overrides,
                }
            )

    runner_payload: dict[str, Any] = {
        "dataset_mode": "reuse_validated",
        # Point at the first dataset so the runner passes validation;
        # per-run dataset_root is overridden via the training_sweep datasets list.
        "dataset_root": trainable[0]["output_root"],
        "output_root": str(output_root / "parallel_run"),
        "allow_regenerate_siesta": False,
        # Mixed datasets never carry strict-only provenance (siesta_version,
        # command-line, environment, execution log come from two merged
        # sources, not a single SIESTA run) -- same as the manual anchor
        # payloads (mixing_sanity_*_paper_ready_anchors_payload.json).
        "strict_dataset_validation": False,
        "training_sweep": {
            "enabled": True,
            "error_policy": "continue_on_error",
            "search_policy": {"strategy": "manual", "random_seed": seed},
            "manual_runs": manual_runs,
        },
        # Inject the full datasets list so expand_training_sweep resolves
        # each planned_run's dataset_root correctly.
        "_mixing_datasets_list": datasets_list,
    }
    if performance:
        runner_payload["performance"] = performance
    if epochs is not None:
        runner_payload["epochs"] = int(epochs)
        runner_payload["graph2mat_overrides"] = {"max_epochs": int(epochs)}
        runner_payload["deeph"] = {"epochs": int(epochs)}
    if early_stopping:
        runner_payload["early_stopping"] = dict(early_stopping)
    if thread_count not in (None, "", "null"):
        runner_payload.setdefault("deeph", {})["num_threads"] = int(thread_count)

    # Patch the runner so it uses our datasets list in expand_training_sweep.
    # We monkey-patch _training_sweep_datasets on the instance only.
    runner = Graph2MatDeepHBenchmarkRunner()

    def _patched_training_sweep_datasets(validation: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ANN001
        return datasets_list

    runner._training_sweep_datasets = _patched_training_sweep_datasets  # type: ignore[method-assign]

    runner.start(runner_payload)
    runner_log_offset = 0
    while True:
        if log_fn is not None:
            try:
                log_payload = runner.logs(since=runner_log_offset, limit=None)
                for line in log_payload.get("lines") or []:
                    log_fn(str(line))
                runner_log_offset = int(log_payload.get("offset") or runner_log_offset)
            except Exception as exc:
                log_fn(f"[MIXING][WARN] No se pudieron leer logs Graph2Mat/DeepH: {exc}\n")
        status = runner.status()
        if not status.get("running"):
            break
        time.sleep(_MIXING_TRAINING_POLL_SECONDS)
    if log_fn is not None:
        try:
            log_payload = runner.logs(since=runner_log_offset, limit=None)
            for line in log_payload.get("lines") or []:
                log_fn(str(line))
        except Exception as exc:
            log_fn(f"[MIXING][WARN] No se pudieron leer logs finales Graph2Mat/DeepH: {exc}\n")

    results = runner.results()
    runner_status = results.get("status") or {}
    run_root = runner_status.get("run_root") or str(output_root / "parallel_run")

    # ── Phase 3: collect per-permutation MAE from the common_metrics CSV ───────
    records: list[dict[str, Any]] = []
    requested_models = list(models)
    for i, perm in enumerate(trainable):
        dataset_id = f"perm_{i}"
        recorded_models: list[str] = []
        # The runner writes per-run metrics under run_root/sweep/<model>/<dataset_id>/…
        for model in models:
            sweep_run_root = Path(run_root) / "sweep" / model / dataset_id
            m = _mixing_metrics_from_common_csv(str(sweep_run_root), (model,))
            if not m:
                # Fallback: scan inside the materialized output_root itself
                m = _mixing_metrics_from_common_csv(perm["output_root"], (model,))
            if not m:
                m = _mixing_metrics_from_run_metrics(sweep_run_root, model)
            if model in m:
                recorded_models.append(model)
                # common_method_metrics.csv is already averaged over samples, so
                # it can never carry the domain split; take that from the
                # per-sample CSVs regardless of which path produced h_mae_eV.
                domains = _mixing_domain_metrics(sweep_run_root, model)
                records.append(
                    {
                        "size": perm.get("size"),
                        "mode": perm.get("mode"),
                        "ratio": perm.get("ratio"),
                        "seed": seed,
                        "total_size": perm.get("total_size"),
                        "model": model,
                        "h_mae_eV": m[model]["h_mae_eV"],
                        "relative_frobenius": m[model].get("relative_frobenius"),
                        **{k: v for k, v in domains.items()},
                        "output_root": perm["output_root"],
                    }
                )
        # Rewrite the per-permutation status (same semantics as run_mixing_sweep)
        # so failures never linger as "materialized".
        if not recorded_models:
            perm["status"] = "failed"
            perm["error"] = "no h_mae_eV produced"
        elif len(recorded_models) < len(requested_models):
            perm["status"] = "partial"
            perm["error"] = "missing h_mae_eV for: " + ", ".join(
                model for model in requested_models if model not in recorded_models
            )
        else:
            perm["status"] = "trained"

    def _count(status: str) -> int:
        return sum(
            1
            for entry in (summary_permutations.get("permutations") or [])
            if entry.get("status") == status
        )

    records.extend(reconstructed_records)
    summary_permutations["records"] = records
    summary_permutations["n_trained"] = _count("trained")
    summary_permutations["n_partial"] = _count("partial")
    summary_permutations["n_failed"] = _count("failed")
    summary_permutations["parallel_run_root"] = run_root
    summary_permutations["training_weighting_policy"] = training_weighting_policy
    if domain_weighting:
        summary_permutations["domain_weighting"] = dict(domain_weighting)
    for record in records:
        record.setdefault("training_weighting_policy", training_weighting_policy)
    _persist_mixing_summary(output_root, summary_permutations)
    return summary_permutations


class MixingSweepRunner:
    """Minimal background runner for the mixing sweep (preview / materialize)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = {"state": "idle"}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _latest_persisted_summary(self) -> dict[str, Any]:
        # The canonical write location is MIXING_SWEEP_OUTPUT_ROOT, which in
        # production lives under RESULTS_ROOT but may be pointed elsewhere.
        candidates = sorted(
            set(RESULTS_ROOT.glob("ml_vs_siesta_mixing_sweep*/mixing_sweep_summary.json"))
            | {MIXING_SWEEP_OUTPUT_ROOT / "mixing_sweep_summary.json"},
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        for summary_path in candidates:
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payload.setdefault("_summary_path", str(summary_path))
                return payload
        return {}

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            summary = self._status.get("summary") or {}
            live_records = self._status.get("live_records") or []
            live_payloads = self._status.get("payloads") or []
        # Use final records when complete, live accumulation while running.
        records = summary.get("records") or live_records
        if not records:
            persisted = self._latest_persisted_summary()
            records = persisted.get("records") or []
            if not summary:
                summary = persisted
        payload = _mixing_metrics_payload(records, summary)
        if live_payloads:
            by_id = {str(item["id"]): item for item in payload.get("payloads", [])}
            for item in live_payloads:
                if isinstance(item, dict) and item.get("id") is not None:
                    by_id.setdefault(str(item["id"]), item)
            payload["payloads"] = list(by_id.values())
        return payload

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("A mixing sweep is already running.")
            self._status = {"state": "starting", "started_at": time.time(), "permutations_done": 0}
            self._thread = threading.Thread(target=self._run, args=(dict(body),), daemon=True)
            self._thread.start()
        return {"state": "started"}

    def _run(self, body: dict[str, Any]) -> None:
        mvs = _ml_vs_siesta_module()
        try:
            small = _mixing_roots_from_body(body, "small")
            large = _mixing_roots_from_body(body, "large")
            modes = tuple(body.get("modes") or ("add", "replace"))
            ratios = tuple(float(r) for r in (body.get("ratios") or (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)))
            sizes = [int(s) for s in body["sizes"]] if body.get("sizes") else None
            from ml_vs_siesta.mixing_payload_schema import prevalidate_mixing_payload

            body = prevalidate_mixing_payload(body)
            seed = int(body.get("seed") or 0)
            models = tuple(body.get("models") or ("graph2mat", "deeph"))
            epochs = int(body["epochs"]) if body.get("epochs") not in (None, "") else None
            performance = body.get("performance") or None
            hyperparams = body.get("hyperparams") or None
            # Default fixed_common_test: with resplit_combined the test set
            # changes with each ratio, confounding MAE-vs-composition curves.
            split_policy = str(body.get("split_policy") or "fixed_common_test")
            temporal_gap = int(body["temporal_gap"]) if body.get("temporal_gap") not in (None, "") else None
            split_fractions = body.get("split_fractions") or None
            if split_fractions is not None:
                split_fractions = tuple(float(x) for x in split_fractions)
            confirm_ghost = bool(body.get("confirm_ghost_species_exemption"))
            training_weighting_policy = str(
                body.get("training_weighting_policy") or "legacy_elementwise"
            )
            domain_weighting = body.get("domain_weighting") or None
            early_stopping = body.get("early_stopping") or None
            reconstructed_records = body.get("reconstructed_records") or []
            if isinstance(reconstructed_records, str):
                try:
                    reconstructed_records = json.loads(
                        (REPO_ROOT / reconstructed_records).read_text(encoding="utf-8")
                    ).get("records") or []
                except (OSError, json.JSONDecodeError):
                    reconstructed_records = []
            # Actions:
            #   "preview"     -> plan only (no writes).
            #   "materialize" -> build merged datasets, no training.
            #   "train"       -> materialize + drive the real Graph2Mat/DeepH runner
            #                    per permutation (needs models installed + GPU).
            action = str(body.get("action") or "preview")
            dry_run = action == "preview"
            launch_fn = _mixing_runner_launch_fn if action == "train" else None

            live_records: list[dict[str, Any]] = []

            def progress(entry: dict[str, Any]) -> None:
                # Accumulate MAE records as they arrive so /metrics is live.
                launch = entry.get("launch") or {}
                for model, model_metrics in (launch.get("metrics") or {}).items():
                    h_mae = (model_metrics or {}).get("h_mae_eV")
                    if h_mae is not None:
                        live_records.append(
                            {
                                "size": entry.get("size"),
                                "mode": entry.get("mode"),
                                "ratio": entry.get("ratio"),
                                "total_size": entry.get("total_size"),
                                "model": model,
                                "h_mae_eV": float(h_mae),
                            }
                        )
                with self._lock:
                    self._status["permutations_done"] = self._status.get("permutations_done", 0) + 1
                    self._status["last_permutation"] = entry
                    self._status["live_records"] = list(live_records)

            with self._lock:
                self._status["state"] = "running"
                self._status["action"] = action
                try:
                    plan = mvs.plan_mixing_sweep_from_roots(
                        small, large, sizes=sizes, modes=modes, ratios=ratios, seed=seed,
                        split_policy=split_policy,
                    )
                    self._status["payloads"] = _mixing_payloads_from_permutations(plan.get("permutations"))
                except Exception:
                    self._status["payloads"] = []
            if action == "train":
                # Parallel path: materialise all datasets, then one runner
                # invocation with full g2m/deeph parallelism.
                summary = _run_mixing_sweep_parallel(
                    small,
                    large,
                    MIXING_SWEEP_OUTPUT_ROOT,
                    sizes=sizes,
                    modes=modes,
                    ratios=ratios,
                    seed=seed,
                    models=models,
                    epochs=epochs,
                    performance=performance,
                    split_policy=split_policy,
                    split_fractions=split_fractions,
                    temporal_gap=temporal_gap,
                    confirm_ghost_species_exemption=confirm_ghost,
                    hyperparams=hyperparams,
                    training_weighting_policy=training_weighting_policy,
                    domain_weighting=domain_weighting,
                    early_stopping=early_stopping,
                    reconstructed_records=reconstructed_records,
                    progress_fn=progress,
                )
            else:
                summary = mvs.run_mixing_sweep(
                    small,
                    large,
                    MIXING_SWEEP_OUTPUT_ROOT,
                    sizes=sizes,
                    modes=modes,
                    ratios=ratios,
                    seed=seed,
                    models=models,
                    epochs=epochs,
                    performance=performance,
                    split_policy=split_policy,
                    confirm_ghost_species_exemption=confirm_ghost,
                    dry_run=dry_run,
                    launch_fn=launch_fn,
                    progress_fn=progress,
                )
            with self._lock:
                self._status.update(
                    {
                        "state": "completed",
                        "finished_at": time.time(),
                        "summary": summary,
                        "n_permutations": summary.get("n_permutations"),
                        "n_trained": summary.get("n_trained", 0),
                        "n_partial": summary.get("n_partial", 0),
                        "n_failed": summary.get("n_failed", 0),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - surface to UI
            with self._lock:
                self._status.update({"state": "error", "error": str(exc), "finished_at": time.time()})


MIXING_SWEEP_RUNNER = MixingSweepRunner()


# --------------------------------------------------------------------------- #
# Cross-testing sweep backend (source×target pairs -> MAE vs training source).
# Mirrors the mixing sweep: discover/plan are synchronous; launch runs a
# background thread. Training is never auto-launched; "preview" is a dry plan,
# "materialize" builds composites, "train" drives the real runner per pair.
# --------------------------------------------------------------------------- #
CROSS_TESTING_SWEEP_OUTPUT_ROOT = RESULTS_ROOT / "ml_vs_siesta_cross_structure_sweep"
CROSS_TESTING_VACANCY_OUTPUT_ROOT = RESULTS_ROOT / "ml_vs_siesta_cross_structure_vacancy"
CROSS_TESTING_BILAYER_OUTPUT_ROOT = RESULTS_ROOT / "ml_vs_siesta_cross_structure_bilayer_moire"
CROSS_TESTING_CONFIG_ROOT = COMPARISON_ROOT / "config"
VACANCY_MATRIX_ERROR_LOCK = threading.Lock()


def _vacancy_matrix_error_ready(output_dir: Path, kind: str = "mae") -> bool:
    html_path = output_dir / f"matrix_error_{kind}.html"
    manifest_path = output_dir / "matrix_error_manifest.json"
    if not html_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (
            manifest.get("status") == "ok"
            and manifest.get("crop_empty") is True
            and manifest.get("visualization_version") == 2
            and manifest.get("aggregation_mode") == "dataset_mean"
            and int((manifest.get("outputs") or {}).get("n_averaged_samples") or 0) > 0
        )
    except (OSError, json.JSONDecodeError):
        return False


def _vacancy_matrix_error_inventory(output_root: Path = CROSS_TESTING_VACANCY_OUTPUT_ROOT) -> list[dict[str, Any]]:
    output_root = output_root.resolve()
    latest_by_payload: dict[str, dict[str, Any]] = {}
    for checkpoint_manifest in output_root.rglob("graph2mat/training/checkpoint_manifest.json"):
        run_root = checkpoint_manifest.parents[2]
        prediction_root = run_root / "graph2mat" / "prediction_structures" / "test"
        predictions = sorted(prediction_root.glob("*/ML_prediction.HSX"))
        config_path = run_root / "graph2mat" / "pipeline_config.yaml"
        if not predictions or not config_path.is_file():
            continue
        try:
            manifest = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checkpoint = Path(str(manifest.get("checkpoint_path") or manifest.get("source_checkpoint_path") or ""))
        if not checkpoint.is_absolute():
            checkpoint = checkpoint_manifest.parent / checkpoint
        if not checkpoint.exists():
            continue
        relative = run_root.relative_to(output_root)
        payload_id = relative.parts[0]
        run_id = hashlib.sha256(str(relative).encode("utf-8")).hexdigest()[:20]
        output_dir = run_root / "matrix_error_plot" / run_id
        item = {
            "id": run_id,
            "payload_id": payload_id,
            "run_name": run_root.name,
            "sample_count": len(predictions),
            "label": f"{payload_id} · media de {len(predictions)} predicciones",
            "run_root": _display_path(run_root),
            "prediction_root": _display_path(prediction_root),
            "checkpoint": _display_path(checkpoint),
            "cached": _vacancy_matrix_error_ready(output_dir),
            "modified_at": checkpoint_manifest.stat().st_mtime,
        }
        previous = latest_by_payload.get(payload_id)
        if previous is None or float(item["modified_at"]) > float(previous["modified_at"]):
            latest_by_payload[payload_id] = item
    return sorted(latest_by_payload.values(), key=lambda item: str(item["payload_id"]))


def vacancy_matrix_error_runs_payload() -> dict[str, Any]:
    runs = _vacancy_matrix_error_inventory()
    return {"runs": runs, "n_runs": len(runs)}


def _vacancy_matrix_error_run(run_id: str) -> dict[str, Any]:
    match = next((item for item in _vacancy_matrix_error_inventory() if item["id"] == run_id), None)
    if match is None:
        raise RuntimeError("Run Graph2Mat de vacante no reconocido.")
    return match


def generate_vacancy_matrix_error_plot(run_id: str) -> dict[str, Any]:
    run = _vacancy_matrix_error_run(str(run_id or ""))
    run_root = (REPO_ROOT / run["run_root"]).resolve()
    output_dir = run_root / "matrix_error_plot" / run_id
    html_path = output_dir / "matrix_error_mae.html"
    if not _vacancy_matrix_error_ready(output_dir):
        checkpoint_manifest = json.loads(
            (run_root / "graph2mat" / "training" / "checkpoint_manifest.json").read_text(encoding="utf-8")
        )
        checkpoint = Path(str(checkpoint_manifest.get("checkpoint_path") or checkpoint_manifest.get("source_checkpoint_path")))
        if not checkpoint.is_absolute():
            checkpoint = run_root / "graph2mat" / "training" / checkpoint
        config_path = run_root / "graph2mat" / "pipeline_config.yaml"
        prediction_root = (REPO_ROOT / run["prediction_root"]).resolve()
        command = [
            postprocess_python_executable(),
            str(COMPARISON_ROOT / "scripts" / "export_graph2mat_matrix_error_plot.py"),
            "--mode", "programmatic",
            "--ckpt-path", str(checkpoint.resolve()),
            "--config-yaml", str(config_path),
            "--test-runs", str(prediction_root / "*" / "RUN.fdf"),
            "--average-hsx-root", str(prediction_root),
            "--output-dir", str(output_dir),
            "--out-matrix", "hamiltonian",
            "--error-metric", "mae",
            "--crop-empty", "--no-stretch-matrix",
            "--no-show", "--no-store-in-logger", "--no-samplewise-metrics", "--no-save-png", "--save-html",
        ]
        with VACANCY_MATRIX_ERROR_LOCK:
            if not _vacancy_matrix_error_ready(output_dir):
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env={
                        **os.environ,
                        "PYTHONUNBUFFERED": "1",
                        "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2",
                    },
                    text=True,
                    capture_output=True,
                    timeout=1800,
                    check=False,
                )
                if completed.returncode != 0 or not _vacancy_matrix_error_ready(output_dir):
                    stderr_log = output_dir / "graph2mat_test_stderr.log"
                    detail = stderr_log.read_text(encoding="utf-8", errors="replace")[-4000:] if stderr_log.is_file() else completed.stderr[-4000:]
                    raise RuntimeError(f"PlotMatrixError falló para {run['label']}: {detail.strip()}")
    return {
        **run,
        "cached": True,
        "artifact_url": f"/api/cross-testing/vacancy/matrix-error/artifact?{urlencode({'run_id': run_id, 'kind': 'mae'})}",
    }


def _cross_testing_resolve_body(body: dict[str, Any]) -> dict[str, Any]:
    """Load an optional campaign payload, constrained to Comparison/config."""
    raw = str(body.get("payload_path") or "").strip()
    if not raw:
        return dict(body)
    path = Path(raw).expanduser()
    path = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    config_root = CROSS_TESTING_CONFIG_ROOT.resolve()
    if path.suffix.lower() != ".json":
        raise RuntimeError("cross-testing payload_path debe ser un archivo .json.")
    if config_root not in path.parents:
        raise RuntimeError("cross-testing payload_path debe estar dentro de Comparison/config.")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"cross-testing payload debe contener un objeto JSON: {path}")
    overrides = {key: value for key, value in body.items() if key != "payload_path"}
    return {**payload, **overrides, "payload_path": _display_path(path)}


def cross_testing_discover_payload() -> dict[str, Any]:
    """Every validated dataset that can be a source or target (single list)."""
    mvs = _ml_vs_siesta_module()
    datasets: list[dict[str, Any]] = []
    seen: set[str] = set()
    if DATASETS_ROOT.exists():
        for manifest_path in sorted(DATASETS_ROOT.rglob("frozen_split_manifest.json")):
            dataset_root = manifest_path.parent
            key = str(dataset_root)
            if key in seen:
                continue
            seen.add(key)
            try:
                samples = mvs.read_dataset_samples(dataset_root)
                n_atoms = mvs.dataset_atom_count(dataset_root)
            except Exception:  # noqa: BLE001 - skip unreadable datasets
                continue
            datasets.append(
                {
                    "root": _display_path(dataset_root),
                    "n_snapshots": len(samples),
                    "n_atoms": n_atoms,
                }
            )
    return {"datasets": datasets, "datasets_root": _display_path(DATASETS_ROOT)}


def _cross_testing_roots_from_body(body: dict[str, Any], key: str) -> list[str]:
    """Resolve a list of dataset roots from ``body[key]`` (list or N=root map)."""
    raw = body.get(key)
    roots: list[str] = []
    if isinstance(raw, dict):
        raw = list(raw.values())
    for root in raw or []:
        resolved = Path(str(root))
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        roots.append(str(resolved))
    return roots


def _cross_testing_pairs_from_body(body: dict[str, Any]) -> list[tuple[str, str]] | None:
    raw = body.get("pairs")
    if not raw:
        return None
    pairs: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        source = Path(str(item.get("source") or ""))
        target = Path(str(item.get("target") or ""))
        if not source.is_absolute():
            source = REPO_ROOT / source
        if not target.is_absolute():
            target = REPO_ROOT / target
        pairs.append((str(source), str(target)))
    return pairs


def _filter_cross_testing_plan_pairs(plan: dict[str, Any], pairs: list[tuple[str, str]] | None) -> dict[str, Any]:
    if not pairs:
        return plan
    allowed = {(str(Path(source)), str(Path(target))) for source, target in pairs}
    filtered = [
        perm for perm in plan.get("permutations") or []
        if (str(Path(str(perm.get("source_root")))), str(Path(str(perm.get("target_root"))))) in allowed
    ]
    return {
        **plan,
        "n_permutations": len(filtered),
        "n_compatible": sum(1 for p in filtered if p.get("status") == "compatible"),
        "n_incompatible": sum(1 for p in filtered if p.get("status") == "incompatible"),
        "permutations": filtered,
    }


def cross_testing_plan_payload(body: dict[str, Any]) -> dict[str, Any]:
    body = _cross_testing_resolve_body(body)
    mvs = _ml_vs_siesta_module()
    pairs = _cross_testing_pairs_from_body(body)
    if pairs:
        permutations: list[dict[str, Any]] = []
        warnings: list[str] = []
        for source, target in pairs:
            pair_plan = mvs.plan_cross_structure_sweep(
                [source],
                [target],
                confirm_ghost_species_exemption=bool(body.get("confirm_ghost_species_exemption")),
                confirm_incomplete_hamiltonian_semantics=bool(
                    body.get("confirm_incomplete_hamiltonian_semantics")
                ),
            )
            permutations.extend(pair_plan.get("permutations") or [])
            warnings.extend(pair_plan.get("warnings") or [])
        return {
            "schema": "ml_vs_siesta_cross_structure_sweep_plan_v1",
            "n_permutations": len(permutations),
            "n_compatible": sum(1 for p in permutations if p.get("status") == "compatible"),
            "n_incompatible": sum(1 for p in permutations if p.get("status") == "incompatible"),
            "permutations": permutations,
            "warnings": warnings,
        }
    sources = _cross_testing_roots_from_body(body, "sources")
    targets = _cross_testing_roots_from_body(body, "targets")
    plan = mvs.plan_cross_structure_sweep(
        sources,
        targets,
        confirm_ghost_species_exemption=bool(body.get("confirm_ghost_species_exemption")),
        confirm_incomplete_hamiltonian_semantics=bool(
            body.get("confirm_incomplete_hamiltonian_semantics")
        ),
    )
    return plan


def _cross_testing_payloads_from_permutations(permutations: Any) -> list[dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    if not isinstance(permutations, list):
        return []
    for item in permutations:
        if not isinstance(item, dict):
            continue
        payload_id = str(item.get("payload_id") or f"{item.get('source_id')}__to__{item.get('target_id')}")
        payloads[payload_id] = {
            "id": payload_id,
            "label": f"{item.get('source_id')} → {item.get('target_id')}",
            "source_id": item.get("source_id"),
            "target_id": item.get("target_id"),
            "source_system_label": item.get("source_system_label"),
            "target_system_label": item.get("target_system_label"),
            "source_n_snapshots": item.get("source_n_snapshots"),
            "status": item.get("status"),
            "output_root": item.get("output_root"),
            "reason": item.get("reason"),
        }
    return sorted(
        payloads.values(),
        key=lambda item: (
            str(item.get("target_id") or ""),
            int(item.get("source_n_snapshots") or 0),
            str(item["id"]),
        ),
    )


def _cross_testing_metrics_payload(
    records: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
    *,
    by_seed: bool = False,
) -> dict[str, Any]:
    mvs = _ml_vs_siesta_module()
    # Older completed summaries kept relative Frobenius inside model_launches
    # but not in records. Backfill it in memory so no prediction is repeated.
    relative_by_run: dict[tuple[str, str], float] = {}
    for permutation in (summary or {}).get("permutations") or []:
        payload_id = str(permutation.get("payload_id") or "")
        for model, launch in (permutation.get("model_launches") or {}).items():
            value = (((launch or {}).get("metrics") or {}).get(model) or {}).get(
                "relative_frobenius"
            )
            if value is not None:
                relative_by_run[(payload_id, str(model))] = float(value)
    enriched_records = []
    for record in records:
        enriched = dict(record)
        key = (str(record.get("payload_id") or ""), str(record.get("model") or ""))
        if enriched.get("relative_frobenius") is None and key in relative_by_run:
            enriched["relative_frobenius"] = relative_by_run[key]
        enriched_records.append(enriched)
    payload = mvs.aggregate_cross_structure_mae(enriched_records, by_seed=by_seed)
    by_id = {str(item["id"]): item for item in payload.get("payloads", [])}
    metric_ids = set(by_id)
    # Permutation-derived payloads carry plain (non-seed-prefixed) ids, so they
    # only merge cleanly with the aggregate payload ids in non-seed mode. In
    # seed-aware mode the aggregate already produced the seed{N}::-keyed payloads.
    if not by_seed:
        for item in _cross_testing_payloads_from_permutations((summary or {}).get("permutations")):
            key = str(item["id"])
            if key in by_id:
                by_id[key] = {**item, **by_id[key]}
            else:
                by_id[key] = item
    metric_status = "evaluated" if (summary or {}).get("action") == "predict_metrics" else "trained"
    for key, item in by_id.items():
        if key in metric_ids:
            item["status"] = metric_status
    payload["payloads"] = sorted(
        by_id.values(),
        key=lambda item: (
            str(item.get("target_id") or ""),
            int(item.get("source_n_snapshots") or 0),
            str(item.get("id") or ""),
        ),
    )
    return payload


def cross_testing_metrics_demo_payload() -> dict[str, Any]:
    """Synthetic MAE-vs-source curves so the chart renders before training."""
    mvs = _ml_vs_siesta_module()
    records: list[dict[str, Any]] = []
    for target_id in ("t_5x5",):
        for source_id, n in (("s_w90_10", 10), ("s_w90_20", 20), ("s_5x5_20", 20)):
            for model, base in (("graph2mat", 0.9), ("deeph", 1.4)):
                for seed in range(3):
                    mae = base / (n ** 0.5) * (1.0 + 0.03 * seed)
                    records.append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "source_system_label": "graphene_w90" if "w90" in source_id else "graphene_5x5",
                            "target_system_label": "graphene_5x5",
                            "payload_id": f"{source_id}__to__{target_id}",
                            "source_n_snapshots": n,
                            "model": model,
                            "seed": seed,
                            "h_mae_eV": round(mae, 5),
                        }
                    )
    payload = mvs.aggregate_cross_structure_mae(records)
    payload["note"] = "Synthetic demo curves (no training); replace with real sweep records."
    return payload


def _cross_testing_launch_fn(runner_payload: dict[str, Any]) -> dict[str, Any]:
    """Synchronously drive the real runner for one composite cross-structure dataset.

    Same convention as ``_mixing_runner_launch_fn``: returns ``ok=False`` (with an
    ``error``) when the runner failed or produced no ``h_mae_eV``, so a failed
    pair is never reported as "trained". Spawns real training subprocesses.
    """
    runner = Graph2MatDeepHBenchmarkRunner()
    models = tuple(
        runner_payload.get("models")
        or runner_payload.get("selected_methods")
        or ("graph2mat", "deeph")
    )
    runner.start(runner_payload)
    while True:
        status = runner.status()
        if not status.get("running"):
            break
        time.sleep(_MIXING_TRAINING_POLL_SECONDS)
    results = runner.results()
    runner_status = results.get("status") or {}
    metrics = _extract_model_h_mae_eV(results, models)
    if len(metrics) < len(models):
        metrics.update(_mixing_metrics_from_common_csv(runner_status.get("run_root"), models))
    ok = _mixing_run_ok(runner_status)
    error: str | None = runner_status.get("error") if isinstance(runner_status, dict) else None
    missing_models = [model for model in models if model not in metrics]
    if ok and missing_models:
        ok = False
        error = error or f"runner produced no h_mae_eV for models: {missing_models}"
    return {
        "ok": ok,
        "error": error,
        "metrics": metrics,
        "missing_models": missing_models,
        "runner_status": runner_status,
    }


class CrossStructureSweepRunner:
    """Minimal background runner for the cross-structure sweep."""

    def __init__(
        self,
        output_root: Path = CROSS_TESTING_SWEEP_OUTPUT_ROOT,
        *,
        merge_all_results: bool = False,
    ) -> None:
        self._output_root = output_root
        # When True, .metrics() merges the records of ALL *result.json files in
        # output_root (one per seed sweep) and aggregates per-seed, so the UI can
        # select individual seeds. Used by the vacancy runner.
        self._merge_all_results = merge_all_results
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = {"state": "idle"}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _merged_persisted_records(self) -> list[dict[str, Any]]:
        """Concatenate the flat ``records`` of every ``*result.json`` in the output
        root, de-duplicated by (payload_id, model, seed). Each seed sweep writes a
        separate result file with its own seed stamped on every record, so this
        surfaces all seeds at once instead of only the most recent file."""
        merged: dict[tuple[str, str, Any], dict[str, Any]] = {}
        paths = list(self._output_root.glob("*result.json"))
        summary_path = self._output_root / "cross_structure_sweep_summary.json"
        if summary_path.is_file():
            paths.append(summary_path)
        for path in sorted(paths):
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            for record in payload.get("records") or []:
                if not isinstance(record, dict):
                    continue
                key = (
                    str(record.get("payload_id") or ""),
                    str(record.get("model") or ""),
                    record.get("seed"),
                )
                merged[key] = record
        return list(merged.values())

    def _latest_persisted_summary(self) -> dict[str, Any]:
        candidates = [self._output_root / "cross_structure_sweep_summary.json"]
        candidates.extend(self._output_root.glob("*result.json"))
        candidates = sorted(
            (path for path in candidates if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        fallback: dict[str, Any] = {}
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            fallback = fallback or payload
            if payload.get("records"):
                return payload
        return fallback

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            summary = self._status.get("summary") or {}
            action = self._status.get("action")
            live_records = self._status.get("live_records") or []
            live_payloads = self._status.get("payloads") or []
        records = summary.get("records") or live_records
        if not records:
            if self._merge_all_results:
                records = self._merged_persisted_records()
            if not records:
                persisted = self._latest_persisted_summary()
                records = persisted.get("records") or []
                if not summary:
                    summary = persisted
        if action and not summary.get("action"):
            summary = {**summary, "action": action}
        payload = _cross_testing_metrics_payload(
            records, summary, by_seed=self._merge_all_results
        )
        if live_payloads:
            by_id = {str(item["id"]): item for item in payload.get("payloads", [])}
            for item in live_payloads:
                if isinstance(item, dict) and item.get("id") is not None:
                    by_id.setdefault(str(item["id"]), item)
            payload["payloads"] = list(by_id.values())
        return payload

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        body = _cross_testing_resolve_body(body)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("A cross-structure sweep is already running.")
            self._status = {
                "state": "starting",
                "action": str(body.get("action") or "preview"),
                "started_at": time.time(),
                "permutations_done": 0,
            }
            self._thread = threading.Thread(target=self._run, args=(dict(body),), daemon=True)
            self._thread.start()
        return {"state": "started"}

    def _run(self, body: dict[str, Any]) -> None:
        mvs = _ml_vs_siesta_module()
        try:
            sources = _cross_testing_roots_from_body(body, "sources")
            targets = _cross_testing_roots_from_body(body, "targets")
            pairs = _cross_testing_pairs_from_body(body)
            if pairs:
                sources = [source for source, _target in pairs]
                targets = [target for _source, target in pairs]
            models = tuple(body.get("models") or ("graph2mat", "deeph"))
            epochs = int(body["epochs"]) if body.get("epochs") not in (None, "") else None
            performance = body.get("performance") or None
            hyperparams = body.get("hyperparams") or None
            early_stopping = body.get("early_stopping") or None
            existing_artifacts = body.get("existing_artifacts") or None
            seed = int(body.get("seed") or 0)
            confirm_ghost = bool(body.get("confirm_ghost_species_exemption"))
            confirm_hamiltonian = bool(body.get("confirm_incomplete_hamiltonian_semantics"))
            strict_dataset_validation = parse_bool(body.get("strict_dataset_validation"), True)
            action = str(body.get("action") or "preview")
            launch_fn = _cross_testing_launch_fn if action in {"train", "predict_metrics"} else None

            live_records: list[dict[str, Any]] = []

            def progress(entry: dict[str, Any]) -> None:
                launch = entry.get("launch") or {}
                for model, model_metrics in (launch.get("metrics") or {}).items():
                    h_mae = (model_metrics or {}).get("h_mae_eV")
                    if h_mae is not None:
                        live_records.append(
                            {
                                "source_id": entry.get("source_id"),
                                "target_id": entry.get("target_id"),
                                "source_system_label": entry.get("source_system_label"),
                                "target_system_label": entry.get("target_system_label"),
                                "payload_id": entry.get("payload_id"),
                                "source_n_snapshots": entry.get("source_n_snapshots"),
                                "source_n_atoms": entry.get("source_n_atoms"),
                                "model": model,
                                "seed": seed,
                                "h_mae_eV": float(h_mae),
                                "relative_frobenius": (model_metrics or {}).get(
                                    "relative_frobenius"
                                ),
                            }
                        )
                with self._lock:
                    self._status["permutations_done"] = self._status.get("permutations_done", 0) + 1
                    self._status["last_permutation"] = entry
                    self._status["live_records"] = list(live_records)

            with self._lock:
                self._status["state"] = "running"
                self._status["action"] = action
                try:
                    plan = cross_testing_plan_payload(body)
                    self._status["payloads"] = _cross_testing_payloads_from_permutations(
                        plan.get("permutations")
                    )
                except Exception:  # noqa: BLE001 - surface later
                    self._status["payloads"] = []

            summary = mvs.run_cross_structure_sweep(
                sources,
                targets,
                self._output_root,
                pairs=pairs,
                models=models,
                epochs=epochs,
                hyperparams=hyperparams,
                early_stopping=early_stopping,
                existing_artifacts=existing_artifacts,
                performance=performance,
                seed=seed,
                confirm_ghost_species_exemption=confirm_ghost,
                confirm_incomplete_hamiltonian_semantics=confirm_hamiltonian,
                strict_dataset_validation=strict_dataset_validation,
                action=action,
                launch_fn=launch_fn,
                progress_fn=progress,
            )
            with self._lock:
                self._status.update(
                    {
                        "state": "completed",
                        "finished_at": time.time(),
                        "summary": summary,
                        "n_permutations": summary.get("n_permutations"),
                        "n_trained": summary.get("n_trained", 0),
                        "n_evaluated": summary.get("n_evaluated", 0),
                        "n_partial": summary.get("n_partial", 0),
                        "n_failed": summary.get("n_failed", 0),
                        "n_incompatible": summary.get("n_incompatible", 0),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - surface to UI
            with self._lock:
                self._status.update({"state": "error", "error": str(exc), "finished_at": time.time()})


CROSS_TESTING_RUNNER = CrossStructureSweepRunner()
CROSS_TESTING_VACANCY_RUNNER = CrossStructureSweepRunner(
    CROSS_TESTING_VACANCY_OUTPUT_ROOT, merge_all_results=True
)
CROSS_TESTING_BILAYER_RUNNER = CrossStructureSweepRunner(CROSS_TESTING_BILAYER_OUTPUT_ROOT)


class MoireSpectralCampaignRunner:
    def __init__(self) -> None:
        self.root = RESULTS_ROOT / "graphene_hbn_magic_angle_spectral"
        self.script = COMPARISON_ROOT / "scripts/run_graphene_hbn_moire_spectral_campaign.py"
        self.config = COMPARISON_ROOT / "config/graphene_hbn_magic_angle_spectral_campaign.json"

    def _read(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _pid_running(self, pid: Any) -> bool:
        try:
            os.kill(int(pid), 0)
        except (OSError, TypeError, ValueError):
            return False
        return True

    def plan(self) -> dict[str, Any]:
        manifest = self._read(self.root / "campaign_manifest.json")
        return {
            "campaign": manifest,
            "stages": [
                self._read(path)
                for path in sorted((self.root / "stages").glob("*.json"))
            ],
        }

    def status(self) -> dict[str, Any]:
        status = self._read(self.root / "status.json")
        disk = shutil.disk_usage(self.root if self.root.exists() else RESULTS_ROOT)
        status["process_alive"] = self._pid_running(status.get("pid"))
        status["persistent"] = True
        status["log_path"] = str(self.root / "campaign.log")
        status["disk"] = {
            "free_bytes": disk.free,
            "free_percent": 100.0 * disk.free / disk.total,
            "minimum_free_percent": 12.0,
        }
        return status

    def results(self) -> dict[str, Any]:
        status = self.status()
        pure_root = RESULTS_ROOT / "tbg_pure_graph2mat"
        summary = self._read(self.root / "summary/spectral_results.json")
        pure_summary = self._read(pure_root / "summary/spectral_results.json")
        if pure_summary.get("spectra"):
            summary = {
                **summary,
                "spectra": [
                    row for row in summary.get("spectra", [])
                    if row.get("material_system") != "pure_tbg"
                ] + pure_summary["spectra"],
            }
        training = self._read(self.root / "training/training_campaign_manifest.json")
        if status.get("running") and status.get("current_stage") == "train":
            controls = [
                self._read(path)
                for path in sorted((self.root / "training").glob("n*/control/status.json"))
            ]
            training = {
                **training,
                "previous_status": training.get("status"),
                "status": "running",
                "live_controls": controls,
            }
        projected_root = self.root / "spectra/graph2mat/n480/seed0"
        projected_progress = {
            "followup": self._read(self.root / "spectra/projected_followup_status.json"),
            "stages": {
                name: {
                    "completed_kpoints": len(list((projected_root / tier).glob("mulliken_[0-9][0-9][0-9].json"))),
                    "expected_kpoints": expected,
                    "manifest_status": self._read(projected_root / tier / "solver_manifest.json").get("status"),
                }
                for name, tier, expected in (
                    ("smoke", "tier_projected_smoke", 31),
                    ("kprime", "tier_projected_kprime_diagnostic", 3),
                    ("production", "tier_projected_production", 151),
                    ("dos_6x6", "tier_projected_dos_smoke", 36),
                )
            },
        }
        return {
            **self.plan(),
            "status": status,
            "summary": summary,
            "pure_tbg": {
                "status": self._read(pure_root / "status.json"),
                "precision_gate": self._read(pure_root / "precision_gate.json"),
                "summary": pure_summary,
            },
            "target": self._read(self.root / "target/moire_geometry.json"),
            "overlap": self._read(self.root / "overlap/overlap_manifest.json"),
            "coverage": self._read(self.root / "local_environment_coverage.json"),
            "solver_environment": self._read(self.root / "solver/environment.json"),
            "solver_environment_gpu_cudss": self._read(
                self.root / "solver/environment_gpu_cudss.json"
            ),
            "tracked_band_sweep": self._read(
                self.root / "spectra/tracked_band_sweep_status.json"
            ),
            "projected_progress": projected_progress,
            "reference_validation": {
                "overlap_only_vs_full": self._read(
                    self.root / "reference_validation/overlap_only_vs_full.json"
                ),
                "synthetic_sparse_dense": self._read(
                    self.root / "solver/synthetic_validation/validation.json"
                ),
                "physical_atoms6_sparse_dense": self._read(
                    self.root / "solver/physical_validation_atoms6/validation.json"
                ),
                "physical_atoms6_cpu_gpu": self._read(
                    self.root / "solver/physical_validation_atoms6/gpu_validation.json"
                ),
            },
            "nested_datasets": self._read(self.root / "training_data/nested_dataset_manifest.json"),
            "training": training,
            "ui_request": self._read(self.root / "ui_request.json"),
            "legacy": self._read(self.root / "legacy_results_inventory.json"),
        }

    def artifact_path(self, value: str) -> Path:
        requested = Path(value).expanduser()
        requested = requested.resolve() if requested.is_absolute() else (self.root / requested).resolve()
        roots = (self.root.resolve(), (RESULTS_ROOT / "tbg_pure_graph2mat").resolve())
        if not any(root == requested or root in requested.parents for root in roots):
            raise RuntimeError("Artefacto espectral fuera de la campaña no permitido.")
        if requested.suffix.lower() not in {".json", ".csv", ".log", ".txt", ".fdf", ".ini"}:
            raise RuntimeError("Tipo de artefacto espectral no permitido.")
        return requested

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.status()
        if current.get("process_alive"):
            raise RuntimeError(f"Spectral campaign already running with pid={current.get('pid')}")
        action = str(payload.get("action") or "resume")
        if action not in {"run", "resume", "generate-training-data", "build-target", "build-overlap", "train", "predict", "solve-bands", "solve-dos", "aggregate"}:
            raise RuntimeError(f"Unsupported spectral campaign action: {action}")
        self.root.mkdir(parents=True, exist_ok=True)
        selection = {
            "training_size": int(payload.get("training_size") or 480),
            "seed": int(payload.get("seed") or 0),
            "model": str(payload.get("model") or "all"),
            "resolution": str(payload.get("resolution") or "coarse"),
            "num_bands": int(payload.get("num_bands") or 256),
        }
        if selection["training_size"] not in {30, 60, 120, 240, 480}:
            raise RuntimeError("Unsupported spectral training_size")
        if selection["seed"] not in {0, 1, 2} or selection["model"] not in {"all", "graph2mat", "deeph"}:
            raise RuntimeError("Unsupported spectral seed/model selection")
        if selection["resolution"] not in {"coarse", "full"} or not 4 <= selection["num_bands"] <= 512:
            raise RuntimeError("Unsupported spectral resolution/num_bands")
        (self.root / "ui_request.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = [
            postprocess_python_executable(),
            str(self.script),
            action,
            "--config",
            str(self.config),
        ]
        with (self.root / "campaign.log").open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return {
            "accepted": True,
            "pid": process.pid,
            "action": action,
            "selection": selection,
            "command": command,
        }

    def stop(self) -> dict[str, Any]:
        completed = subprocess.run(
            [postprocess_python_executable(), str(self.script), "stop", "--config", str(self.config)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "accepted": completed.returncode == 0,
            "policy": "stop_after_current_stage",
            "returncode": completed.returncode,
            "status": self.status(),
        }


MOIRE_SPECTRAL_RUNNER = MoireSpectralCampaignRunner()


class ComparisonUIHandler(BaseHTTPRequestHandler):
    server_version = "ComparisonPipelineUI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[comparison-ui] {self.address_string()} - {fmt % args}\n")

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        try:
            if path == "/api/health":
                json_response(
                    self,
                    {
                        "ok": True,
                        "repo_root": str(REPO_ROOT),
                        "pipelines": {
                            key: {
                                "root": str(spec.root),
                                "config_path": str(spec.config_path),
                                "main_script": str(spec.main_script),
                            }
                            for key, spec in PIPELINES.items()
                        },
                    },
                )
            elif path == "/api/run/status":
                json_response(self, all_status())
            elif path == "/api/run/logs":
                query = parse_qs(parsed_url.query)
                key = query.get("pipeline", [""])[0]
                since = int(query.get("since", ["0"])[0])
                limit = parse_query_int(
                    query,
                    "limit",
                    DEFAULT_LOG_RESPONSE_LIMIT,
                    minimum=1,
                    maximum=MAX_LOG_RESPONSE_LIMIT,
                )
                if key not in RUNNERS:
                    raise RuntimeError("Pipeline no reconocido.")
                json_response(self, RUNNERS[key].logs(since=since, limit=limit))
            elif path == "/api/results":
                json_response(self, result_summary())
            elif path == "/api/plots":
                json_response(self, plot_data_summary())
            elif path == "/api/datasets/targets":
                query = parse_qs(urlparse(self.path).query)
                scope = str((query.get("scope") or ["plot_metrics"])[0] or "plot_metrics").strip()
                if scope in {"all", "all_generated_artifacts"}:
                    json_response(self, generated_dataset_output_records())
                else:
                    json_response(self, plot_metric_dataset_output_records())
            elif path == "/api/datasets/reusable":
                json_response(self, EXPERIMENT_RUNNER.reusable_dataset_candidates_payload())
            elif path == "/api/atom-fc-config":
                json_response(self, atom_fc_ui_config())
            elif path == "/api/performance-presets":
                json_response(self, performance_preset_catalog())
            elif path == "/api/material/presets":
                json_response(self, material_presets_payload())
            elif path == "/api/methods":
                json_response(self, {"methods": method_registry_payload(), "run_modes": sorted(RUN_MODES)})
            elif path == "/api/experiment/status":
                json_response(self, EXPERIMENT_RUNNER.status())
            elif path == "/api/experiment/logs":
                query = parse_qs(parsed_url.query)
                since = int(query.get("since", ["0"])[0])
                limit = parse_query_int(
                    query,
                    "limit",
                    DEFAULT_LOG_RESPONSE_LIMIT,
                    minimum=1,
                    maximum=MAX_LOG_RESPONSE_LIMIT,
                )
                json_response(self, EXPERIMENT_RUNNER.logs(since=since, limit=limit))
            elif path == "/api/g2m-deeph/status":
                json_response(self, G2M_DEEPH_RUNNER.status())
            elif path == "/api/g2m-deeph/datasets":
                query = parse_qs(parsed_url.query)
                root = (query.get("root") or [None])[0]
                json_response(self, G2M_DEEPH_RUNNER.available_datasets_payload(root=root))
            elif path == "/api/g2m-deeph/logs":
                query = parse_qs(parsed_url.query)
                since = int(query.get("since", ["0"])[0])
                limit = parse_query_int(
                    query,
                    "limit",
                    DEFAULT_LOG_RESPONSE_LIMIT,
                    minimum=1,
                    maximum=MAX_LOG_RESPONSE_LIMIT,
                )
                json_response(self, G2M_DEEPH_RUNNER.logs(since=since, limit=limit))
            elif path == "/api/g2m-deeph/results":
                json_response(self, G2M_DEEPH_RUNNER.results())
            elif path == "/api/g2m-deeph/plot-runs":
                json_response(self, _g2m_deeph_plot_runs_payload())
            elif path == "/api/g2m-deeph/plots":
                query = parse_qs(parsed_url.query, keep_blank_values=True)
                selected_run_ids = None
                if any(key in query for key in ("run_id", "run_ids", "selected_run_ids")):
                    selected_run_ids = set()
                    for key in ("run_id", "run_ids", "selected_run_ids"):
                        for value in query.get(key, []):
                            selected_run_ids.update(
                                item.strip()
                                for item in str(value).split(",")
                                if item.strip()
                            )
                json_response(self, G2M_DEEPH_RUNNER.plots(selected_run_ids=selected_run_ids))
            elif path == "/api/g2m-deeph/derivative-metrics":
                query = parse_qs(parsed_url.query)
                run_ids: list[str] = []
                for key in ("run_id", "run_ids", "selected_run_ids"):
                    for value in query.get(key, []):
                        run_ids.extend(item.strip() for item in str(value).split(",") if item.strip())
                json_response(self, g2m_deeph_derivative_metrics_multi_payload(run_ids))
            elif path == "/api/g2m-deeph/derivative-metrics/artifact":
                query = parse_qs(parsed_url.query)
                run_id = str((query.get("run_id") or [""])[0] or "").strip()
                kind = str((query.get("kind") or [""])[0] or "").strip()
                if not run_id or not kind:
                    raise RuntimeError("run_id y kind son obligatorios.")
                artifact_meta = G2M_DEEPH_DERIVATIVE_ARTIFACTS.get(kind)
                if artifact_meta is None:
                    raise RuntimeError(f"Artefacto derivative no permitido: {kind}")
                run_root = resolve_g2m_deeph_run_root(run_id)
                if run_root is None:
                    raise RuntimeError("run_id derivative no encontrado.")
                artifact_path = _g2m_deeph_derivative_artifact_path(run_root, kind)
                if artifact_path is None:
                    raise FileNotFoundError(kind)
                self._serve_file(artifact_path, content_type=str(artifact_meta["mime_type"]))
            elif path == "/api/ml-vs-siesta/config-template":
                json_response(self, ml_vs_siesta_config_template_payload())
            elif path == "/api/ml-vs-siesta/matrix-viewer-demo":
                json_response(self, ml_vs_siesta_matrix_viewer_demo_payload())
            elif path == "/api/mixing/discover":
                query = parse_qs(parsed_url.query)
                threshold = parse_query_int(query, "threshold_atoms", MIXING_SMALL_ATOM_THRESHOLD, minimum=1)
                json_response(self, mixing_discover_payload(threshold))
            elif path == "/api/mixing/status":
                json_response(self, MIXING_SWEEP_RUNNER.status())
            elif path == "/api/deeph/capabilities":
                json_response(self, deeph_capabilities_payload())
            elif path == "/api/run-inventory":
                json_response(self, run_inventory_payload())
            elif path == "/api/mixing-e2e/status":
                json_response(self, MIXING_E2E_RUNNER.status())
            elif path == "/api/mixing-e2e/logs":
                query = parse_qs(parsed_url.query)
                since = int(query.get("since", ["0"])[0])
                limit = parse_query_int(
                    query,
                    "limit",
                    DEFAULT_LOG_RESPONSE_LIMIT,
                    minimum=1,
                    maximum=MAX_LOG_RESPONSE_LIMIT,
                )
                json_response(self, MIXING_E2E_RUNNER.logs(since=since, limit=limit))
            elif path == "/api/mixing/metrics":
                json_response(self, MIXING_SWEEP_RUNNER.metrics())
            elif path == "/api/mixing/metrics-demo":
                json_response(self, mixing_metrics_demo_payload())
            elif path == "/api/cross-testing/discover":
                json_response(self, cross_testing_discover_payload())
            elif path == "/api/cross-testing/status":
                json_response(self, CROSS_TESTING_RUNNER.status())
            elif path == "/api/cross-testing/metrics":
                json_response(self, CROSS_TESTING_RUNNER.metrics())
            elif path == "/api/cross-testing/vacancy/status":
                json_response(self, CROSS_TESTING_VACANCY_RUNNER.status())
            elif path == "/api/cross-testing/vacancy/metrics":
                json_response(self, CROSS_TESTING_VACANCY_RUNNER.metrics())
            elif path == "/api/cross-testing/bilayer/status":
                json_response(self, CROSS_TESTING_BILAYER_RUNNER.status())
            elif path == "/api/cross-testing/bilayer/metrics":
                json_response(self, CROSS_TESTING_BILAYER_RUNNER.metrics())
            elif path == "/api/cross-testing/bilayer/spectral/plan":
                json_response(self, MOIRE_SPECTRAL_RUNNER.plan())
            elif path == "/api/cross-testing/bilayer/spectral/status":
                json_response(self, MOIRE_SPECTRAL_RUNNER.status())
            elif path == "/api/cross-testing/bilayer/spectral/results":
                json_response(self, MOIRE_SPECTRAL_RUNNER.results())
            elif path == "/api/cross-testing/bilayer/spectral/artifact":
                query = parse_qs(parsed_url.query)
                artifact = str((query.get("path") or [""])[0] or "").strip()
                if not artifact:
                    raise RuntimeError("path es obligatorio.")
                self._serve_file(MOIRE_SPECTRAL_RUNNER.artifact_path(artifact))
            elif path == "/api/cross-testing/vacancy/matrix-errors":
                json_response(self, vacancy_matrix_error_runs_payload())
            elif path == "/api/cross-testing/vacancy/matrix-error/artifact":
                query = parse_qs(parsed_url.query)
                run_id = str((query.get("run_id") or [""])[0])
                kind = str((query.get("kind") or ["mae"])[0]).lower()
                if kind not in {"mae", "rmse"}:
                    raise RuntimeError("Tipo de PlotMatrixError no permitido.")
                run = _vacancy_matrix_error_run(run_id)
                run_root = (REPO_ROOT / run["run_root"]).resolve()
                output_dir = run_root / "matrix_error_plot" / run_id
                if not _vacancy_matrix_error_ready(output_dir, kind):
                    raise RuntimeError("PlotMatrixError todavía no está listo.")
                self._serve_file(output_dir / f"matrix_error_{kind}.html")
            elif path == "/api/cross-testing/metrics-demo":
                json_response(self, cross_testing_metrics_demo_payload())
            elif path == "/api/g2m-deeph/dataset-size-minimum":
                json_response(self, dataset_size_minimum_payload())
            elif path == "/api/g2m-deeph/dataset-size-minimum/artifact":
                query = parse_qs(parsed_url.query)
                output_dir_text = str((query.get("output_dir") or [""])[0] or "").strip()
                filename = str((query.get("name") or [""])[0] or "").strip()
                if not output_dir_text or not filename:
                    raise RuntimeError("output_dir y name son obligatorios.")
                artifact_meta = DATASET_SIZE_MINIMUM_UI_ARTIFACTS.get(filename)
                if artifact_meta is None:
                    raise RuntimeError(f"Artefacto dataset-size-minimum no permitido: {filename}")
                output_dir = Path(output_dir_text).expanduser().resolve()
                if RESULTS_ROOT.resolve() not in output_dir.parents:
                    raise RuntimeError("output_dir fuera de Comparison/results no permitido.")
                summary_path = output_dir / "dataset_size_minimum_summary.json"
                if not summary_path.exists():
                    raise FileNotFoundError(summary_path)
                artifact_path = output_dir / filename
                self._serve_file(artifact_path, content_type=str(artifact_meta["mime_type"]))
            elif path == "/":
                self._serve_file(UI_DIR / "index.html")
            else:
                requested = (UI_DIR / path.lstrip("/")).resolve()
                if UI_DIR.resolve() not in requested.parents:
                    raise FileNotFoundError(path)
                self._serve_file(requested)
        except FileNotFoundError:
            error_response(self, RuntimeError("No encontrado."), HTTPStatus.NOT_FOUND)
        except Exception as exc:
            error_response(self, exc)

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/api/run":
                payload = read_json_body(self)
                json_response(
                    self,
                    run_all(venv_activate_command=payload.get("venv_activate_command")),
                    status=HTTPStatus.ACCEPTED,
                )
            elif path == "/api/run/stop":
                json_response(self, stop_all(), status=HTTPStatus.ACCEPTED)
            elif path == "/api/datasets/clear":
                payload = read_json_body(self)
                dry_run = parse_bool(payload.get("dry_run"), False)
                target_ids = payload.get("target_ids")
                if target_ids not in (None, ""):
                    if not isinstance(target_ids, list):
                        raise RuntimeError("target_ids debe ser una lista de IDs de datasets.")
                    result = clear_selected_generated_dataset_outputs(
                        [str(target_id) for target_id in target_ids],
                        dry_run=dry_run,
                    )
                elif parse_bool(payload.get("all"), False):
                    result = clear_generated_dataset_outputs(dry_run=dry_run)
                else:
                    raise RuntimeError("Borrado ambiguo: envia target_ids o all=true.")
                json_response(
                    self,
                    result,
                    status=HTTPStatus.ACCEPTED,
                )
            elif path == "/api/material/validate":
                payload = read_json_body(self)
                material_payload = payload.get("material", payload)
                json_response(self, material_validation_response(material_payload))
            elif path == "/api/g2m-deeph/validate-dataset":
                payload = read_json_body(self)
                json_response(self, G2M_DEEPH_RUNNER.validate_dataset_payload(payload))
            elif path == "/api/g2m-deeph/run":
                payload = read_json_body(self)
                json_response(
                    self,
                    G2M_DEEPH_RUNNER.start(payload),
                    status=HTTPStatus.ACCEPTED,
                )
            elif path == "/api/g2m-deeph/stop":
                json_response(
                    self,
                    G2M_DEEPH_RUNNER.stop(),
                    status=HTTPStatus.ACCEPTED,
                )
            elif path == "/api/ml-vs-siesta/dry-run":
                payload = read_json_body(self)
                json_response(self, ml_vs_siesta_dry_run_payload(payload))
            elif path == "/api/ml-vs-siesta/generate-displacements":
                payload = read_json_body(self)
                json_response(
                    self,
                    ml_vs_siesta_generate_payload(payload),
                    status=HTTPStatus.CREATED,
                )
            elif path == "/api/ml-vs-siesta/mix-datasets":
                payload = read_json_body(self)
                json_response(self, ml_vs_siesta_mix_payload(payload))
            elif path == "/api/ml-vs-siesta/inspect-species":
                payload = read_json_body(self)
                json_response(self, ml_vs_siesta_inspect_species_payload(payload))
            elif path == "/api/mixing/plan":
                payload = read_json_body(self)
                json_response(self, mixing_plan_payload(payload))
            elif path == "/api/mixing/launch":
                payload = read_json_body(self)
                json_response(self, MIXING_SWEEP_RUNNER.start(payload), status=HTTPStatus.ACCEPTED)
            elif path == "/api/cross-testing/plan":
                payload = read_json_body(self)
                json_response(self, cross_testing_plan_payload(payload))
            elif path == "/api/cross-testing/launch":
                payload = read_json_body(self)
                json_response(self, CROSS_TESTING_RUNNER.start(payload), status=HTTPStatus.ACCEPTED)
            elif path == "/api/cross-testing/vacancy/launch":
                payload = read_json_body(self)
                json_response(
                    self,
                    CROSS_TESTING_VACANCY_RUNNER.start(payload),
                    status=HTTPStatus.ACCEPTED,
                )
            elif path == "/api/cross-testing/bilayer/launch":
                payload = read_json_body(self)
                json_response(
                    self,
                    CROSS_TESTING_BILAYER_RUNNER.start(payload),
                    status=HTTPStatus.ACCEPTED,
                )
            elif path == "/api/cross-testing/bilayer/spectral/launch":
                payload = read_json_body(self)
                json_response(
                    self,
                    MOIRE_SPECTRAL_RUNNER.start(payload),
                    status=HTTPStatus.ACCEPTED,
                )
            elif path == "/api/cross-testing/bilayer/spectral/stop":
                json_response(
                    self,
                    MOIRE_SPECTRAL_RUNNER.stop(),
                    status=HTTPStatus.ACCEPTED,
                )
            elif path == "/api/cross-testing/vacancy/matrix-error":
                payload = read_json_body(self)
                json_response(self, generate_vacancy_matrix_error_plot(str(payload.get("run_id") or "")))
            elif path == "/api/mixing-e2e/start":
                payload = read_json_body(self)
                json_response(
                    self,
                    MIXING_E2E_RUNNER.start(
                        payload_path=payload.get("payload"),
                        manifest_path=payload.get("manifest_json"),
                        poll_seconds=float(payload.get("poll_seconds") or 30.0),
                    ),
                    status=HTTPStatus.ACCEPTED,
                )
            elif path == "/api/mixing-e2e/stop":
                json_response(self, MIXING_E2E_RUNNER.stop(), status=HTTPStatus.ACCEPTED)
            elif path == "/api/g2m-deeph/dataset-size-minimum/analyze":
                payload = read_json_body(self)
                json_response(
                    self,
                    run_dataset_size_minimum_analysis(payload),
                    status=HTTPStatus.CREATED,
                )
            elif path == "/api/g2m-deeph/dataset-size-minimum/preview":
                payload = read_json_body(self)
                json_response(self, dataset_size_minimum_preview(payload))
            elif path == "/api/experiment":
                payload = read_json_body(self)
                material_config = parse_material_payload(
                    payload.get("material"),
                    required="material" in payload,
                )
                if material_config is not None:
                    validate_material_payload(material_config, required=True)
                run_mode = parse_run_mode(payload.get("run_mode"))
                if run_mode == DEEPH_COMPARISON_RUN_MODE:
                    json_response(
                        self,
                        EXPERIMENT_RUNNER.start_deeph_comparison(payload.get("deeph_comparison") or {}),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
                deeph_comparison_options = (
                    parse_deeph_comparison_options(
                        payload.get("deeph_comparison") or {},
                        require_graph2mat_result=False,
                    )
                    if run_mode == GRAPH2MAT_DEEPH_RUN_MODE
                    else None
                )
                selected_methods = normalize_selected_methods(payload.get("selected_methods"))
                reusable_dataset_ids = parse_reusable_dataset_ids(payload.get("reusable_dataset_ids"))
                training_plan = parse_training_plan(payload.get("training_plan"))
                hyperparameter_sweep = parse_hyperparameter_sweep(payload.get("hyperparameter_sweep"))
                validate_training_plan_sweep_sources(training_plan, hyperparameter_sweep)
                if reusable_dataset_ids and not run_mode_skips_dataset_generation(run_mode):
                    raise RuntimeError(
                        "reusable_dataset_ids solo se puede usar con train_test_metrics_plots_only."
                    )
                if training_plan:
                    validate_training_plan_for_run_mode(training_plan, run_mode)
                selected_pipeline_keys = pipeline_keys_for_methods(selected_methods)
                method_options = payload.get("method_options") or {}
                method_random_options = (
                    method_options.get("random_cartesian")
                    if isinstance(method_options, dict)
                    else None
                )
                random_cartesian_options = parse_random_cartesian_options(
                    payload.get("random_cartesian_options") or method_random_options
                )
                raw_md_sizes = payload.get("md_sizes")
                displacement_options = parse_fc_displacement_options(
                    payload.get("fc_displacement_options")
                )
                fc_displacements = parse_fc_displacements(payload.get("fc_displacements"))
                structures_per_displacement = parse_structures_per_displacement(
                    payload.get("structures_per_displacement")
                )
                split_ratios = parse_split_ratios(payload.get("splits")) or dict(DEFAULT_SPLIT_RATIOS)
                random_seed = payload.get("random_seed")
                random_seed = None if random_seed in (None, "") else int(random_seed)
                max_datasets = parse_max_datasets(payload.get("max_datasets"))
                combination_mode = parse_combination_mode(payload.get("combination_mode"))
                split_mode = parse_split_mode(payload.get("split_mode"))
                reusable_split_policy = parse_reusable_split_policy(payload.get("reusable_split_policy"))
                raw_test_sets = payload.get("test_sets")
                if raw_test_sets in (None, "", []):
                    test_sets = [f"test_{method}" for method in selected_methods] + ["test_mixed"]
                else:
                    test_sets = parse_test_sets(raw_test_sets)
                primary_metric = str(payload.get("primary_metric") or DEFAULT_PRIMARY_METRIC).strip()
                compute_budget_mode = parse_compute_budget_mode(payload.get("compute_budget_mode"))
                compute_accelerator = parse_compute_accelerator(payload.get("compute_accelerator"))
                performance_settings = parse_performance_settings(
                    payload.get("performance"),
                    compute_accelerator=compute_accelerator,
                )
                training_settings = parse_training_settings(payload.get("training_settings"))
                training_plan_dataset_ids: list[str] = []
                if run_mode_skips_dataset_generation(run_mode):
                    for plan_item in training_plan:
                        for dataset_id in plan_item["reusable_dataset_ids"]:
                            if dataset_id not in training_plan_dataset_ids:
                                training_plan_dataset_ids.append(dataset_id)
                effective_reusable_dataset_ids = (
                    training_plan_dataset_ids if training_plan else reusable_dataset_ids
                )
                compute_accelerator = str(performance_settings["compute_accelerator"])
                raw_venv_activate_command = payload.get("venv_activate_command")
                venv_activate_command = (
                    str(raw_venv_activate_command).strip()
                    if raw_venv_activate_command is not None
                    else DEFAULT_VENV_ACTIVATE_COMMAND
                )
                if not venv_activate_command:
                    venv_activate_command = DEFAULT_VENV_ACTIVATE_COMMAND
                venv_activate_path = resolve_venv_activate_from_command(venv_activate_command)
                raw_atom_sizes = payload.get("atom_sizes")
                atom_dataset_specs = None
                fc_dataset_specs = None
                atom_sizes = []
                if "atom_displacement" in selected_pipeline_keys:
                    atom_config = load_config(PIPELINES["atom_displacement"].config_path)
                    force_constants = atom_config.get("structure", {}).get("force_constants", {}) or {}
                    if payload.get("max_datasets") in (None, ""):
                        max_datasets = parse_max_datasets(force_constants.get("max_datasets"), 100)
                    if payload.get("combination_mode") in (None, ""):
                        combination_mode = parse_combination_mode(
                            force_constants.get("combination_mode", "aligned")
                        )
                    limit = atom_fc_sample_limit(atom_config)
                    if limit is None:
                        raise RuntimeError("AtomDisplacement: FC no esta habilitado en la configuracion.")
                    if displacement_options:
                        atom_dataset_specs = build_fc_dataset_specs_from_options(
                            displacement_options,
                            combination_mode=combination_mode,
                            per_displacement_limit=limit,
                            split_ratios=split_ratios,
                            max_datasets=max_datasets,
                        )
                        atom_sizes = [int(spec["size"]) for spec in atom_dataset_specs]
                    else:
                        requested_atom_sizes = parse_sizes(raw_atom_sizes, [10])
                        atom_sizes, fc_dataset_specs = build_fc_dataset_specs(
                            requested_atom_sizes,
                            fc_displacements,
                            structures_per_displacement,
                            per_displacement_limit=limit,
                            split_ratios=split_ratios,
                        )
                if "md" in selected_pipeline_keys:
                    if atom_sizes and parse_bool(payload.get("sync_md_sizes"), False):
                        md_sizes = unique_ints_preserve_order(atom_sizes)
                    else:
                        md_sizes = parse_sizes(raw_md_sizes, atom_sizes or [10, 20])
                else:
                    md_sizes = []
                if "atom_displacement" not in selected_pipeline_keys:
                    fc_dataset_specs = None
                    atom_dataset_specs = None
                dataset_recipes_info = dataset_recipes_to_execution_specs(
                    payload.get("dataset_recipes"),
                    selected_methods=selected_methods,
                    split_ratios=split_ratios,
                    random_cartesian_defaults=random_cartesian_options,
                )
                if dataset_recipes_info:
                    md_recipe_specs = dataset_recipes_info.get("md_dataset_specs") or []
                    atom_recipe_specs = dataset_recipes_info.get("atom_dataset_specs") or []
                    random_recipe_specs = dataset_recipes_info.get("random_cartesian_dataset_specs") or []
                    if md_recipe_specs:
                        md_sizes = [int(spec["size"]) for spec in md_recipe_specs]
                    if atom_recipe_specs:
                        atom_dataset_specs = atom_recipe_specs
                        atom_sizes = [int(spec["size"]) for spec in atom_recipe_specs]
                        fc_dataset_specs = None
                    if random_recipe_specs:
                        random_cartesian_options = {
                            **random_cartesian_options,
                            "_dataset_specs": random_recipe_specs,
                        }
                    legacy_recipes = legacy_payload_to_dataset_recipes(
                        md_sizes=md_sizes,
                        atom_dataset_specs=atom_dataset_specs,
                        atom_sizes=atom_sizes,
                        fc_dataset_specs=fc_dataset_specs,
                        random_cartesian_options=random_cartesian_options,
                        selected_methods=selected_methods,
                    )
                    merged_recipes = dict(legacy_recipes)
                    merged_recipes.update(dataset_recipes_info.get("recipes") or {})
                    dataset_recipes_info["recipes"] = merged_recipes
                    dataset_recipes_info["recipe_set_hash"] = recipe_set_hash(merged_recipes)
                else:
                    legacy_recipes = legacy_payload_to_dataset_recipes(
                        md_sizes=md_sizes,
                        atom_dataset_specs=atom_dataset_specs,
                        atom_sizes=atom_sizes,
                        fc_dataset_specs=fc_dataset_specs,
                        random_cartesian_options=random_cartesian_options,
                        selected_methods=selected_methods,
                    )
                    dataset_recipes_info = {
                        "recipes": legacy_recipes,
                        "recipe_set_hash": recipe_set_hash(legacy_recipes),
                        "md_dataset_specs": [],
                        "atom_dataset_specs": atom_dataset_specs or [],
                        "random_cartesian_dataset_specs": random_cartesian_options.get("_dataset_specs") or [],
                    }
                if effective_reusable_dataset_ids:
                    reusable_info = EXPERIMENT_RUNNER.dataset_recipes_info_for_reusable_dataset_ids(
                        effective_reusable_dataset_ids,
                        selected_methods=selected_methods,
                    )
                    reusable_recipes = reusable_info.get("recipes") or {}
                    merged_recipes = dict(dataset_recipes_info.get("recipes") or {})
                    merged_recipes.update(reusable_recipes)
                    dataset_recipes_info["recipes"] = merged_recipes
                    if reusable_info.get("md_dataset_specs"):
                        dataset_recipes_info["md_dataset_specs"] = reusable_info["md_dataset_specs"]
                        md_sizes = [int(spec["size"]) for spec in reusable_info["md_dataset_specs"]]
                    if reusable_info.get("atom_dataset_specs"):
                        dataset_recipes_info["atom_dataset_specs"] = reusable_info["atom_dataset_specs"]
                        atom_dataset_specs = reusable_info["atom_dataset_specs"]
                        atom_sizes = [int(spec["size"]) for spec in reusable_info["atom_dataset_specs"]]
                        fc_dataset_specs = None
                    if reusable_info.get("random_cartesian_dataset_specs"):
                        dataset_recipes_info["random_cartesian_dataset_specs"] = reusable_info[
                            "random_cartesian_dataset_specs"
                        ]
                        random_cartesian_options = {
                            **random_cartesian_options,
                            "_dataset_specs": reusable_info["random_cartesian_dataset_specs"],
                        }
                    if training_plan:
                        plan_methods: list[str] = []
                        if reusable_info.get("md_dataset_specs"):
                            plan_methods.append("md")
                        if reusable_info.get("atom_dataset_specs"):
                            plan_methods.append("siesta_fc_cartesian")
                        if reusable_info.get("random_cartesian_dataset_specs"):
                            plan_methods.append("random_cartesian")
                        selected_methods = [method for method in selected_methods if method in plan_methods]
                        if not selected_methods:
                            selected_methods = plan_methods
                    dataset_recipes_info["recipe_set_hash"] = recipe_set_hash(merged_recipes)
                if hyperparameter_sweep.get("enabled"):
                    raw_sweep_payload = payload.get("hyperparameter_sweep") or {}
                    if not isinstance(raw_sweep_payload, dict):
                        raw_sweep_payload = {}
                    sweep_reusable_ids = parse_reusable_dataset_ids(
                        raw_sweep_payload.get("reusable_dataset_ids")
                    ) or list(effective_reusable_dataset_ids)
                    sweep_dataset_targets = parse_dataset_targets(
                        raw_sweep_payload.get("dataset_targets")
                    )
                    if run_mode_uses_planned_dataset_targets(run_mode) and not sweep_dataset_targets:
                        planned_targets = EXPERIMENT_RUNNER._planned_dataset_targets_for_specs(
                            md_specs=dataset_recipes_info.get("md_dataset_specs") or [],
                            atom_specs=dataset_recipes_info.get("atom_dataset_specs") or atom_dataset_specs or [],
                            random_specs=dataset_recipes_info.get("random_cartesian_dataset_specs")
                            or random_cartesian_options.get("_dataset_specs")
                            or [],
                            selected_methods=selected_methods,
                        )
                        sweep_dataset_targets = [
                            EXPERIMENT_RUNNER._planned_dataset_target_public(target)
                            for target in planned_targets
                        ]
                    training_plan = expand_hyperparameter_sweep_to_training_plan(
                        hyperparameter_sweep,
                        base_training_settings=training_settings,
                        run_mode=run_mode,
                        reusable_dataset_ids=sweep_reusable_ids,
                        dataset_targets=sweep_dataset_targets,
                    )
                    validate_training_plan_for_run_mode(training_plan, run_mode)
                json_response(
                    self,
                    EXPERIMENT_RUNNER.start(
                        md_sizes,
                        atom_sizes,
                        fc_dataset_specs,
                        atom_dataset_specs,
                        split_ratios,
                        random_seed,
                        split_mode,
                        test_sets,
                        primary_metric,
                        compute_budget_mode,
                        compute_accelerator,
                        selected_methods,
                        run_mode,
                        random_cartesian_options,
                        performance_settings,
                        training_settings,
                        venv_activate_path,
                        dataset_recipes_info,
                        reusable_split_policy,
                        training_plan,
                        material_config,
                        hyperparameter_sweep,
                        deeph_comparison_options,
                    ),
                    status=HTTPStatus.ACCEPTED,
                )
            elif path == "/api/experiment/stop":
                json_response(
                    self,
                    EXPERIMENT_RUNNER.stop(),
                    status=HTTPStatus.ACCEPTED,
                )
            else:
                raise FileNotFoundError(self.path)
        except FileNotFoundError:
            error_response(self, RuntimeError("No encontrado."), HTTPStatus.NOT_FOUND)
        except Exception as exc:
            error_response(self, exc)

    def _serve_file(self, path: Path, *, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        if content_type is None:
            content_type = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".csv": "text/csv; charset=utf-8",
                ".png": "image/png",
                ".pdf": "application/pdf",
            }.get(path.suffix, "application/octet-stream")
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        if path.suffix in {".html", ".css", ".js"}:
            self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def ui_display_urls(host: str, port: int) -> list[str]:
    host_text = str(host or "").strip()
    if host_text not in {"", "0.0.0.0", "::"}:
        return [f"http://{host_text}:{port}"]
    hosts = ["127.0.0.1"]
    try:
        hosts.extend(
            address
            for address in socket.gethostbyname_ex(socket.gethostname())[2]
            if address and not address.startswith("127.")
        )
    except OSError:
        pass
    return [f"http://{address}:{port}" for address in dict.fromkeys(hosts)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the combined comparison pipeline UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ComparisonUIHandler)
    print(f"Comparison Pipeline UI listening on {args.host}:{args.port}")
    for url in ui_display_urls(args.host, args.port):
        print(f"Open: {url}")
    print(f"Repo root: {REPO_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Comparison Pipeline UI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
