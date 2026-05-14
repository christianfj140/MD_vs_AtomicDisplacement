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
import select
import secrets
import shutil
import shlex
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
from urllib.parse import parse_qs, urlparse

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

try:
    import pty
except Exception:  # pragma: no cover - Windows fallback.
    pty = None

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).resolve().parents[1] / "ui"
COMPARISON_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = COMPARISON_ROOT / "results"
WORKSPACES_ROOT = COMPARISON_ROOT / "workspaces"
LOG_HEARTBEAT_SECONDS = 30.0
DEFAULT_LOG_RESPONSE_LIMIT = 2000
MAX_LOG_RESPONSE_LIMIT = 20000
METRIC_VERSION = "2026-05-08.frontier-window-v1"
DEFAULT_VENV_ACTIVATE_COMMAND = "source ${REPO_ROOT}/.venv/bin/activate"


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

RUN_MODES = {"dataset_only", "full_strict_pipeline"}


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
    mode = "full_strict_pipeline" if value in (None, "") else str(value).strip()
    if mode not in RUN_MODES:
        raise RuntimeError(
            "run_mode debe ser 'dataset_only' o 'full_strict_pipeline' "
            f"(recibido: {value!r})."
        )
    return mode


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
    return {
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


def stream_process_output(
    process: subprocess.Popen[str],
    append: Any,
    *,
    label: str,
    master_fd: int | None = None,
    eta_provider: Any | None = None,
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
                append(
                    "[UI] "
                    f"{label} sigue ejecutandose | PID {process.pid} | "
                    f"elapsed {format_duration(now - started_at)} | "
                    f"sin nueva salida {format_duration(now - last_output)} | "
                    f"ETA {format_duration(eta_seconds)}\n"
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
            append(
                "[UI] "
                f"{label} sigue ejecutandose | PID {process.pid} | "
                f"elapsed {format_duration(now - started_at)} | "
                f"sin nueva salida {format_duration(now - last_output)} | "
                f"ETA {format_duration(eta_seconds)}\n"
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
    if (
        not Path(os.path.expandvars(text.replace("${REPO_ROOT}", str(REPO_ROOT)))).expanduser().is_absolute()
        and not text.startswith("${GRAPH2MAT_VENV}")
    ):
        text = "${REPO_ROOT}/" + text
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
    mode = "block" if value in (None, "") else str(value).strip().lower()
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
}

PERFORMANCE_PRESET_IDS = {
    "soft",
    "balanced",
    "aggressive",
    "aggressive_local",
    "gpu_focused",
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
) -> dict[str, Any]:
    return {
        "max_parallel_siesta_jobs": int(siesta_jobs),
        "max_parallel_dataset_jobs": 1,
        "max_parallel_prediction_jobs": 1,
        "max_parallel_evaluation_jobs": int(evaluation_jobs),
        "max_parallel_metric_jobs": int(metric_jobs),
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
        "torch_float32_matmul_precision": "high" if accelerator != "cpu" else None,
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
    )
    gpu_focused = _base_profile(
        preset="gpu_focused",
        hardware=hw,
        accelerator="gpu" if has_cuda else "cpu",
        batch_size=_vram_batch_size(vram_gb, mode="gpu_focused") if has_cuda else 32,
        siesta_jobs=_clamp(physical // 6, 1, 3),
        evaluation_jobs=_clamp(logical // 8, 1, 3),
        metric_jobs=_clamp(logical // 3, 2, 8),
        threads=1,
        torch_threads=_clamp(logical // 2, 4, 16),
        store_in_memory=False if low_ram else True,
    )
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
        + int(settings.get("torch_num_threads") or 1)
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


def parse_training_settings(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        raw: dict[str, Any] = {}
    elif isinstance(value, dict):
        raw = value
    else:
        raise RuntimeError("training_settings debe ser un objeto.")
    settings: dict[str, Any] = {}
    for key in ("max_epochs", "batch_size", "num_interactions", "correlation"):
        parsed = parse_optional_positive_int(raw.get(key), f"training_settings.{key}")
        if parsed is not None:
            settings[key] = parsed
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
    return settings


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
    for key in ("num_interactions", "correlation", "max_ell"):
        if settings.get(key) is not None:
            model[key] = int(settings[key])
    if settings.get("optim_lr") is not None:
        model["optim_lr"] = float(settings["optim_lr"])
    for key in ("loss", "hidden_irreps"):
        if settings.get(key) is not None:
            model[key] = str(settings[key])
    training["ui_training_settings"] = dict(settings)


def aggressive_local_performance_defaults() -> dict[str, Any]:
    cores = max(1, os.cpu_count() or 1)
    siesta_jobs = max(1, min(max(1, cores // 2), 8))
    return {
        "max_parallel_siesta_jobs": siesta_jobs,
        "max_parallel_dataset_jobs": 1,
        "max_parallel_prediction_jobs": 1,
        "max_parallel_evaluation_jobs": min(cores, 8),
        "max_parallel_metric_jobs": cores,
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
    payload = {
        "checkpoint_path": metadata.get("path"),
        "checkpoint_sha256": metadata.get("sha256"),
        "relative_path": metadata.get("relative_path"),
        "selection_reason": "manifest" if not warning else "latest_version_fallback",
        "checkpoint_selection_warning": warning,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "training_dir": str(training_dir),
    }
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
            ("dos", "dos_wasserstein_eV"),
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
        if run_mode == "dataset_only" or skipped:
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
            result_dir = Path(manifest.get("result_dir", manifest_path.parent))
            if not result_dir.is_absolute():
                result_dir = manifest_path.parent / result_dir
            if not result_dir.exists():
                result_dir = manifest_path.parent
            sparse_rows = read_csv_rows(result_dir / "metrics" / "sparse_metrics.csv")
            spectral_rows = read_csv_rows(result_dir / "metrics" / "spectral_metrics.csv")
            dos_rows = read_csv_rows(result_dir / "metrics" / "dos_metrics.csv")
            sparse_sweep_rows = read_csv_rows(result_dir / "metrics" / "sparse_threshold_sweep.csv")
            dos_sweep_rows = read_csv_rows(result_dir / "metrics" / "dos_sigma_sweep.csv")
            relationship_rows = read_csv_rows(
                result_dir / "metrics" / "matrix_spectrum_relationship.csv"
            )
            if not sparse_rows and not spectral_rows and not dos_rows and not relationship_rows and not sparse_sweep_rows and not dos_sweep_rows:
                continue
            errors = manifest.get("errors", [])
            if not isinstance(errors, list):
                errors = []
            metric_manifest = {}
            metric_manifest_path = result_dir / "metrics" / "manifest.json"
            if metric_manifest_path.exists():
                try:
                    metric_manifest = json.loads(metric_manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    metric_manifest = {}
            metric_errors = metric_manifest.get("fatal_errors", metric_manifest.get("errors", []))
            if isinstance(metric_errors, list):
                errors = errors + metric_errors
            warnings = metric_manifest.get("warnings", [])
            if not isinstance(warnings, list):
                warnings = []
            runs.append(
                {
                    "pipeline": key,
                    "method_id": manifest.get(
                        "method_id",
                        "siesta_fc_cartesian" if key == "atom_displacement" else key,
                    ),
                    "label": PIPELINES[key].label if key in PIPELINES else "Random Cartesian",
                    "dataset_size": int(
                        manifest.get(
                            "requested_dataset_size",
                            manifest.get("dataset_size", 0),
                        )
                    ),
                    "effective_dataset_size": int(
                        manifest.get(
                            "effective_dataset_size",
                            manifest.get("dataset_size", 0),
                        )
                    ),
                    "requested_dataset_size": int(
                        manifest.get(
                            "requested_dataset_size",
                            manifest.get("dataset_size", 0),
                        )
                    ),
                    "run_id": str(manifest.get("run_id", manifest_path.parent.name.removeprefix("run_"))),
                    "result_dir": str(result_dir),
                    "dataset_label": manifest.get("dataset_label"),
                    "recipe_id": manifest.get("recipe_id"),
                    "recipe_label": manifest.get("recipe_label"),
                    "block_id": manifest.get("block_id"),
                    "block_label": manifest.get("block_label"),
                    "recipe_set_hash": manifest.get("recipe_set_hash"),
                    "dataset_recipe": manifest.get("dataset_recipe", {}),
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
                        "sparse_sweep": numeric_means(sparse_sweep_rows),
                        "dos_sweep": numeric_means(dos_sweep_rows),
                        "matrix_spectrum": numeric_means(relationship_rows),
                    },
                    "samples": {
                        "sparse": sparse_rows,
                        "spectral": spectral_rows,
                        "dos": dos_rows,
                        "sparse_sweep": sparse_sweep_rows,
                        "dos_sweep": dos_sweep_rows,
                        "matrix_spectrum": relationship_rows,
                    },
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
                    },
                    "summary": manifest.get("summary", {}),
                }
            )
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
        compatibility = {
            "metric_version": manifest.get("metric_version"),
            "molecule_system_name": manifest.get("molecule_system_name"),
            "siesta_settings_hash": manifest.get("siesta_settings_hash"),
            "model_config_hash": manifest.get("model_config_hash"),
            "test_sets": manifest.get("test_sets"),
            "selected_methods": manifest.get("selected_methods"),
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
        cross_experiments.append(
            {
                "experiment_id": experiment_dir.name,
                "metrics": rows,
                "metric_availability": cross_metric_availability(rows),
                "recommendation": recommendation,
                "manifest": manifest,
                "compatibility": compatibility,
                "compatibility_group_id": compatibility_group_id,
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
        split_mode: str = "block",
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
    ) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Ya hay una comparacion experimental en ejecucion.")
            selected_methods = normalize_selected_methods(selected_methods)
            run_mode = parse_run_mode(run_mode)
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
                ),
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
    ) -> dict[str, Any]:
        md_config = load_config(PIPELINES["md"].config_path)
        atom_config = load_config(PIPELINES["atom_displacement"].config_path)
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
            "scientific_status": "dataset_only" if run_mode == "dataset_only" else (
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
        if result.get("run_mode", manifest.get("run_mode")) != "dataset_only":
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
        split_mode: str = "block",
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
    ) -> None:
        split_ratios = split_ratios or dict(DEFAULT_SPLIT_RATIOS)
        selected_methods = normalize_selected_methods(selected_methods)
        run_mode = parse_run_mode(run_mode)
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
            if STRICT_COMPARISON_MODE and manifest.get("siesta_settings_warning"):
                raise RuntimeError(
                    "Strict comparison aborted: MD y AtomDisplacement tienen settings SIESTA distintas. "
                    "Revisa experiment_manifest.yaml: siesta_settings_mismatches."
                )
            if STRICT_COMPARISON_MODE and manifest.get("model_config_warning"):
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

            md_task_specs = dataset_recipes_info.get("md_dataset_specs") or [
                {"label": fallback_dataset_label("md", index), "size": size, "recipe_metadata": None}
                for index, size in enumerate(md_sizes)
            ]
            for md_spec in (md_task_specs if "md" in pipeline_keys else []):
                size = int(md_spec["size"])
                dataset_tasks.append((
                    f"md {md_spec.get('label', f'dataset_{size}')}",
                    lambda size=size, md_spec=md_spec: self._run_one(
                        "md",
                        size,
                        run_id,
                        dataset_label=str(md_spec.get("label") or f"dataset_{size}"),
                        recipe_metadata=md_spec.get("recipe_metadata"),
                        md_temperature_blocks=md_spec.get("temperature_blocks"),
                        split_ratios=split_ratios,
                        split_mode=split_mode,
                        run_mode=run_mode,
                        compute_accelerator=compute_accelerator,
                        performance=performance_settings,
                        training_settings=training_settings,
                        venv_activate_path=venv_activate_path,
                    ),
                ))
            atom_runs = dataset_recipes_info.get("atom_dataset_specs") or atom_dataset_specs or [
                {
                    "label": fallback_dataset_label("fc", index),
                    "size": size,
                    "displacements": fc_dataset_specs.get(size) if fc_dataset_specs else None,
                }
                for index, size in enumerate(atom_sizes)
            ]
            for atom_spec in (atom_runs if "atom_displacement" in pipeline_keys else []):
                atom_recipe_seed = (atom_spec.get("recipe_metadata") or {}).get("seed")
                atom_random_seed = atom_recipe_seed if atom_recipe_seed not in (None, "") else random_seed
                dataset_tasks.append((
                    f"atom_displacement {atom_spec['label']}",
                    lambda atom_spec=atom_spec, atom_random_seed=atom_random_seed: self._run_one(
                        "atom_displacement",
                        int(atom_spec["size"]),
                        run_id,
                        dataset_label=str(atom_spec["label"]),
                        fc_displacements=atom_spec.get("displacements"),
                        recipe_metadata=atom_spec.get("recipe_metadata"),
                        split_ratios=split_ratios,
                        random_seed=atom_random_seed,
                        split_mode=split_mode,
                        run_mode=run_mode,
                        compute_accelerator=compute_accelerator,
                        performance=performance_settings,
                        training_settings=training_settings,
                        venv_activate_path=venv_activate_path,
                    ),
                ))
            if "random_cartesian" in selected_methods:
                random_specs = dataset_recipes_info.get("random_cartesian_dataset_specs") or random_cartesian_options.get("_dataset_specs") or [
                    {
                        "label": fallback_dataset_label("rc", index),
                        "size": random_size,
                        "options": {**random_cartesian_options, "n_structures": random_size},
                        "recipe_metadata": None,
                    }
                    for index, random_size in enumerate(random_cartesian_sizes_from_options(random_cartesian_options))
                ]
                for random_spec in random_specs:
                    random_size = int(random_spec["size"])
                    size_options = {
                        **random_cartesian_options,
                        **dict(random_spec.get("options") or {}),
                        "n_structures": random_size,
                    }
                    size_options.pop("_dataset_specs", None)
                    dataset_tasks.append((
                        f"random_cartesian {random_spec.get('label', f'dataset_{random_size}')}",
                        lambda random_size=random_size, size_options=size_options, random_spec=random_spec: self._run_random_cartesian(
                            random_size,
                            run_id,
                            dataset_label=str(random_spec.get("label") or f"dataset_{random_size}"),
                            recipe_metadata=random_spec.get("recipe_metadata"),
                            split_ratios=split_ratios,
                            random_cartesian_options=size_options,
                            run_mode=run_mode,
                            compute_accelerator=compute_accelerator,
                            performance=performance_settings,
                            training_settings=training_settings,
                            venv_activate_path=venv_activate_path,
                        ),
                    ))
            dataset_results = self._run_dataset_tasks(
                dataset_tasks,
                manifest=manifest,
                workers=dataset_workers,
                error_policy=str(performance_settings.get("error_policy", "fail_fast")),
            )
            manifest["timing"]["counters"]["dataset_jobs"] = len(dataset_tasks)
            previous_by_method: dict[str, dict[str, Any]] = {}
            for result in sorted(dataset_results, key=lambda item: (str(item.get("method_id") or item.get("pipeline")), int(item.get("dataset_size") or 0), str(item.get("dataset_label") or ""))):
                method = str(result.get("method_id") or result.get("pipeline"))
                if method in {"md", "siesta_fc_cartesian", "atom_displacement"}:
                    nested_key = "atom_displacement" if method in {"siesta_fc_cartesian", "atom_displacement"} else "md"
                    self._annotate_nested_subset(result, previous_by_method.get(nested_key))
                    previous_by_method[nested_key] = result
                self._record_run_result(manifest, result)
                self._write_experiment_manifest(manifest)
            if run_mode == "dataset_only":
                manifest["cross_evaluation"] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "dataset_only",
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
        split_mode: str = "block",
        run_mode: str = "full_strict_pipeline",
        compute_accelerator: str = "cpu",
        performance: dict[str, Any] | None = None,
        training_settings: dict[str, Any] | None = None,
        venv_activate_path: str | None = None,
    ) -> dict[str, Any]:
        spec = PIPELINES[key]
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
        if key == "md":
            self._prepare_md_config(
                config,
                workspace,
                size,
                split_ratios,
                split_mode=split_mode,
                temperature_blocks=md_temperature_blocks,
            )
            original_steps = list(config.get("pipeline", {}).get("steps", []))
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
            if run_mode == "dataset_only":
                config["pipeline"]["steps"] = []
                write_yaml(config_snapshot_path, config)
                self._append("[UI] dataset_only: MD validado; no se entrena ni predice.\n")
                returncode = 0
            else:
                config["pipeline"]["steps"] = [
                    step
                    for step in original_steps
                    if step in {"run_md_training", "run_md_testing", "run_md_prediction"}
                ] or ["run_md_training", "run_md_testing", "run_md_prediction"]
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
            if run_mode == "dataset_only":
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
        random_cartesian_options: dict[str, Any] | None = None,
        run_mode: str = "dataset_only",
        compute_accelerator: str = "cpu",
        performance: dict[str, Any] | None = None,
        training_settings: dict[str, Any] | None = None,
        venv_activate_path: str | None = None,
    ) -> dict[str, Any]:
        spec = PIPELINES["atom_displacement"]
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
        pseudo_count = copy_pseudopotentials(PIPELINES["atom_displacement"].root / "base", base_dir)
        relaxed_counts = copy_relaxed_basis(PIPELINES["atom_displacement"].root / "relaxed", relaxed_dir)
        random_config = config.setdefault("structure", {}).setdefault("random_cartesian", {})
        random_config.update(random_cartesian_options or {})
        random_config["enabled"] = True
        random_config["n_structures"] = int(size)
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
        self._append(f"[UI] Random Cartesian config: {random_config}\n")
        with self._lock:
            log_start = len(self._logs)
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
        if run_mode != "dataset_only":
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
            self._append("[UI] Config temporal Random Cartesian escrita para entrenar/evaluar.\n")
            self._validate_split_manifests("atom_displacement", config, size)
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
        split_mode: str = "block",
        temperature_blocks: list[dict[str, Any]] | None = None,
    ) -> None:
        dataset_dir = workspace / "dataset"
        pseudo_count = copy_pseudopotentials(PIPELINES["md"].root / "dataset", dataset_dir)
        ratios = split_ratios or split_ratios_from_config(config)
        counts = validate_split_sizes(size, ratios, label=f"MD dataset_{size}")
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
        config["training"]["data"]["train_runs"] = "../dataset/splits/train/*/RUN.fdf"
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
            f"{ratios['train']}/{ratios['validation']}/{ratios['test']}.\n"
        )
        self._append(f"[UI] MD train_runs: {config['training']['data']['train_runs']}\n")
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
        config["training"]["data"].setdefault("n_matrix_components", 2)
        config["prediction"]["data"]["basis_files"] = basis_files_pattern
        config["prediction"]["data"].setdefault("n_matrix_components", 2)
        config["testing"]["data"]["basis_files"] = basis_files_pattern
        config["testing"]["data"].setdefault("n_matrix_components", 2)
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
        for validation_file in sorted((dataset_dir / "validation").glob("*/*.csv")):
            copy_if_exists(validation_file, result_dir / "validation" / validation_file.parent.name / validation_file.name)
        siesta_counts = self._single_point_counts(result_dir / "run_summary.json")
        if run_mode == "dataset_only":
            evaluation_metrics = {
                "skipped": True,
                "reason": "dataset_only",
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
        if run_mode == "dataset_only":
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
        }
        dataset_sample_ids: list[str] = []
        for split_manifest in sorted((result_dir / "splits").glob("*_manifest.csv")):
            for row in read_csv_rows(split_manifest):
                sample_id = str(row.get("sample_id") or row.get("sample") or "")
                if sample_id:
                    dataset_sample_ids.append(sample_id)
        manifest_seed = prepare_metadata.get("seed")
        if manifest_seed in (None, ""):
            manifest_seed = recipe_metadata.get("seed")
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
            "model_checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "model_checkpoint_metadata": signed_checkpoint,
            "model_checkpoint_sha256": signed_checkpoint.get("sha256"),
            "checkpoint_manifest": str(checkpoint_manifest_path),
            "checkpoint_selection_warning": checkpoint_warning,
            "artifact_hashes": run_artifact_hashes,
            "dataset_sample_ids": sorted(set(dataset_sample_ids)),
            "dataset_sample_hash": sample_set_hash(dataset_sample_ids),
            "seed": manifest_seed,
            "dataset_recipe": recipe_metadata,
            "recipe_id": recipe_metadata.get("recipe_id"),
            "recipe_label": recipe_metadata.get("recipe_label"),
            "block_id": recipe_metadata.get("block_id"),
            "block_label": recipe_metadata.get("block_label"),
            "recipe_set_hash": recipe_set_hash(recipe_metadata) if recipe_metadata else "",
            "generation_parameters_json": recipe_metadata.get("generation_parameters_json"),
            "predicted_hamiltonians": prediction_count,
            "siesta_hamiltonians": reference_count,
            "timing_breakdown": timing_breakdown,
            "siesta_counts": siesta_counts,
            "performance": config.get("performance", {}),
            "training_settings": config.get("training", {}).get("ui_training_settings", {}),
            "training_hyperparameters": config.get("training", {}),
            "generated_samples": prepare_metadata.get("generated_samples"),
            "completed_samples": prepare_metadata.get("completed_samples"),
            "fc_generated_samples": prepare_metadata.get("fc_generated_samples"),
            "fc_completed_samples": prepare_metadata.get("fc_completed_samples"),
            "metrics": read_metrics_summary(result_dir / "sample_metrics.csv"),
            "hamiltonian_evaluation": evaluation_metrics,
            "run_mode": run_mode,
            "scientific_status": "dataset_only" if run_mode == "dataset_only" else "pending",
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

    def _n_matrix_components_for_result(self, config: dict[str, Any]) -> int | None:
        for section in ("prediction", "testing", "training"):
            data = config.get(section, {}).get("data", {})
            value = data.get("n_matrix_components")
            if value not in (None, ""):
                return int(value)
        return None

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

    def _run_cross_evaluation(self, run_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
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
                        if n_matrix_components is not None:
                            predict_command.extend(["--n-matrix-components", str(n_matrix_components)])
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
                    ]
                )
            )
            invalid_recommendation = {
                "status": "invalid_incomplete_grid",
                "scientific_status": "invalid_incomplete_grid",
                "winner": None,
                "reason": "Incomplete cross-evaluation grid",
                "missing_cells": missing_cells,
                "missing_required_cells": completeness.get("missing_cells", []),
                "extra_unexpected_cells": completeness.get("extra_unexpected_cells", []),
                "missing_primary_metric_cells": completeness.get("missing_primary_metric_cells", []),
                "completeness_report": str(completeness_path),
            }
            (summary_root / "recommendation.json").write_text(
                json.dumps(invalid_recommendation, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            summary["missing_cells"].extend(completeness.get("missing_cells", []))
            summary["missing_cells"].extend(completeness.get("missing_primary_metric_cells", []))
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
                    if n_matrix_components is not None:
                        predict_command.extend(["--n-matrix-components", str(n_matrix_components)])
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
                        "recipe_set_hash": train_result.get("recipe_set_hash", ""),
                        "train_dataset_label": train_result.get("dataset_label", ""),
                        "train_recipe_id": train_result.get("recipe_id"),
                        "train_recipe_label": train_result.get("recipe_label"),
                        "train_block_id": train_result.get("block_id"),
                        "train_block_label": train_result.get("block_label"),
                        "train_generation_parameters_json": train_result.get("generation_parameters_json"),
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
                        "model_config_warning": manifest.get("model_config_warning", ""),
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
                    cross_manifest["method_provenance_warnings"] = sorted(
                        dict.fromkeys(
                            f"{method}: {warning}"
                            for method, provenance in cross_manifest["method_provenance"].items()
                            for warning in provenance.get("warnings", []) or []
                        )
                    )
                    cross_manifest["method_provenance_severe_warnings"] = sorted(
                        dict.fromkeys(
                            f"{method}: {warning}"
                            for method, provenance in cross_manifest["method_provenance"].items()
                            for warning in provenance.get("severe_warnings", []) or []
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
                    ]
                )
            )
            invalid_recommendation = {
                "status": "invalid_incomplete_grid",
                "scientific_status": "invalid_incomplete_grid",
                "winner": None,
                "reason": "Incomplete cross-evaluation grid",
                "missing_cells": missing_cells,
                "missing_required_cells": completeness.get("missing_cells", []),
                "extra_unexpected_cells": completeness.get("extra_unexpected_cells", []),
                "missing_primary_metric_cells": completeness.get("missing_primary_metric_cells", []),
                "completeness_report": str(completeness_path),
            }
            (summary_root / "recommendation.json").write_text(
                json.dumps(invalid_recommendation, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            summary["missing_cells"].extend(completeness.get("missing_cells", []))
            summary["missing_cells"].extend(completeness.get("missing_primary_metric_cells", []))
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


def all_status() -> dict[str, Any]:
    statuses = {key: runner.status() for key, runner in RUNNERS.items()}
    return {
        "running": any(status["running"] for status in statuses.values()),
        "pipelines": statuses,
    }


def clear_generated_dataset_outputs(*, dry_run: bool = False) -> dict[str, Any]:
    if all_status().get("running") or EXPERIMENT_RUNNER.status().get("running"):
        raise RuntimeError("No se pueden borrar datasets mientras hay pipelines o experimentos en ejecucion.")
    return cleanup_generated_datasets(REPO_ROOT, dry_run=dry_run)


def generated_dataset_output_records() -> dict[str, Any]:
    return {
        "targets": generated_dataset_records(REPO_ROOT),
        "roots": {
            "md_dataset": str(REPO_ROOT / "MD" / "dataset"),
            "atom_dataset": str(REPO_ROOT / "AtomDisplacement" / "dataset"),
            "comparison_workspaces": str(WORKSPACES_ROOT),
            "comparison_results": str(RESULTS_ROOT),
        },
    }


def clear_selected_generated_dataset_outputs(
    target_ids: list[str],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    if all_status().get("running") or EXPERIMENT_RUNNER.status().get("running"):
        raise RuntimeError("No se pueden borrar datasets mientras hay pipelines o experimentos en ejecucion.")
    return cleanup_selected_generated_datasets(
        REPO_ROOT,
        target_ids=target_ids,
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
            summary[key].append(manifest)
    return summary


def archived_result_manifest_paths(root: Path) -> list[Path]:
    """Return archived run manifests for both legacy and recipe-based dataset names."""
    candidates = list(root.glob("*/run_*/manifest.json"))
    candidates.extend(root.glob("run_*/manifest.json"))
    unique = {path.resolve(): path for path in candidates}
    return sorted(unique.values())


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
                json_response(self, generated_dataset_output_records())
            elif path == "/api/atom-fc-config":
                json_response(self, atom_fc_ui_config())
            elif path == "/api/performance-presets":
                json_response(self, performance_preset_catalog())
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
            elif path == "/api/experiment":
                payload = read_json_body(self)
                selected_methods = normalize_selected_methods(payload.get("selected_methods"))
                run_mode = parse_run_mode(payload.get("run_mode"))
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

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }.get(path.suffix, "application/octet-stream")
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the combined comparison pipeline UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ComparisonUIHandler)
    print(f"Comparison Pipeline UI listening on http://{args.host}:{args.port}")
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
