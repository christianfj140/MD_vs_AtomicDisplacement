#!/usr/bin/env python3
"""Compare Graph2Mat model/training settings across scientific methods.

This module deliberately ignores path-only training differences while retaining
the model and hyperparameter fields that affect the scientific comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from method_registry import normalize_method_id


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MD_CONFIG = REPO_ROOT / "MD" / "pipeline_config.yaml"
DEFAULT_ATOM_CONFIG = REPO_ROOT / "AtomDisplacement" / "pipeline_config.yaml"

IGNORED_DATA_KEYS = {
    "basis_files",
    "train_runs",
    "test_runs",
    "predict_structs",
    "runs_json",
}
IGNORED_TRAINER_KEYS = {
    "logger",
    "default_root_dir",
}
PATH_ONLY_KEY_NAMES = {
    "data_dir",
    "dataset_dir",
    "dataset_path",
    "log_dir",
    "logging_dir",
    "output_dir",
    "output_directory",
    "save_dir",
    "checkpoint_dir",
    "checkpoint_path",
}
PATH_ONLY_SUFFIXES = ("_path", "_paths", "_dir", "_dirs")
NON_SEVERE_TRAINER_KEYS = {
    "accelerator",
    "devices",
    "num_nodes",
    "strategy",
}
TOP_LEVEL_TRAINING_KEYS = {
    "torch_float32_matmul_precision",
}
COMPARABLE_SECTIONS = ("data", "model", "trainer", "training")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return value


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(normalize(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def settings_hash(settings: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(settings).encode("utf-8")).hexdigest()


def is_path_only_key(key: str) -> bool:
    normalized_key = str(key).strip().lower()
    return normalized_key in PATH_ONLY_KEY_NAMES or normalized_key.endswith(PATH_ONLY_SUFFIXES)


def comparable_training_settings(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {}) or {}
    data = {
        key: value
        for key, value in (training.get("data", {}) or {}).items()
        if key not in IGNORED_DATA_KEYS and not is_path_only_key(str(key))
    }
    trainer = {
        key: value
        for key, value in (training.get("trainer", {}) or {}).items()
        if key not in IGNORED_TRAINER_KEYS and not is_path_only_key(str(key))
    }
    top_level_training = {
        key: training.get(key)
        for key in TOP_LEVEL_TRAINING_KEYS
        if key in training
    }
    return {
        "data": data,
        "model": training.get("model", {}) or {},
        "trainer": trainer,
        "training": top_level_training,
    }


def mismatch_severity(section: str, key: str) -> str:
    if section == "trainer" and key in NON_SEVERE_TRAINER_KEYS:
        return "info"
    return "severe"


def model_mismatch_payload(
    *,
    method_a: str,
    method_b: str,
    section: str,
    key: str,
    value_a: Any,
    value_b: Any,
) -> dict[str, Any]:
    severity = mismatch_severity(section, key)
    return {
        "type": "model_config",
        "section": section,
        "key": key,
        "methods": [method_a, method_b],
        "values": {
            method_a: value_a,
            method_b: value_b,
        },
        "severity": severity,
        "scientifically_relevant": severity == "severe",
    }


def pairwise_mismatch_report(method_settings: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted(method_settings)
    mismatches: list[dict[str, Any]] = []
    for index, method_a in enumerate(methods):
        for method_b in methods[index + 1 :]:
            settings_a = method_settings[method_a]
            settings_b = method_settings[method_b]
            for section in COMPARABLE_SECTIONS:
                keys = sorted(set(settings_a.get(section, {})) | set(settings_b.get(section, {})))
                for key in keys:
                    value_a = normalize((settings_a.get(section, {}) or {}).get(key))
                    value_b = normalize((settings_b.get(section, {}) or {}).get(key))
                    if value_a == value_b:
                        continue
                    mismatches.append(
                        model_mismatch_payload(
                            method_a=method_a,
                            method_b=method_b,
                            section=section,
                            key=key,
                            value_a=value_a,
                            value_b=value_b,
                        )
                    )
    return mismatches


def compare_method_model_settings(
    configs_by_method: dict[str, dict[str, Any]],
    *,
    selected_methods: list[str] | None = None,
) -> dict[str, Any]:
    normalized_configs = {
        normalize_method_id(method, allow_unknown=True): config
        for method, config in configs_by_method.items()
    }
    methods = selected_methods or list(normalized_configs)
    canonical_methods = []
    for method in methods:
        method_id = normalize_method_id(method, allow_unknown=True)
        if method_id in normalized_configs and method_id not in canonical_methods:
            canonical_methods.append(method_id)
    method_settings = {
        method_id: comparable_training_settings(normalized_configs[method_id])
        for method_id in canonical_methods
    }
    pairwise_mismatches = pairwise_mismatch_report(method_settings)
    severe_mismatches = [mismatch for mismatch in pairwise_mismatches if mismatch.get("severity") == "severe"]
    warning_mismatches = [mismatch for mismatch in pairwise_mismatches if mismatch.get("severity") != "severe"]
    hash_by_method = {
        method_id: settings_hash(settings)
        for method_id, settings in method_settings.items()
    }
    warning = (
        "Graph2Mat model/training settings differ across selected methods."
        if pairwise_mismatches
        else ""
    )
    severe_warning = (
        "Scientifically relevant Graph2Mat model/training settings differ across selected methods."
        if severe_mismatches
        else ""
    )
    report = {
        "ok": not severe_mismatches,
        "selected_methods": canonical_methods,
        "model_config_hash": settings_hash(method_settings),
        "model_config_hash_by_method": hash_by_method,
        "method_model_settings": method_settings,
        "pairwise_mismatch_report": pairwise_mismatches,
        "severe_mismatches": severe_mismatches,
        "warning_mismatches": warning_mismatches,
        "warning": warning,
        "severe_warning": severe_warning,
    }
    if "md" in hash_by_method:
        report["md_model_config_hash"] = hash_by_method["md"]
    if "siesta_fc_cartesian" in hash_by_method:
        report["siesta_fc_cartesian_model_config_hash"] = hash_by_method["siesta_fc_cartesian"]
        report["atom_displacement_model_config_hash"] = hash_by_method["siesta_fc_cartesian"]
    if "random_cartesian" in hash_by_method:
        report["random_cartesian_model_config_hash"] = hash_by_method["random_cartesian"]
    return report


def compare_model_settings(md_config: dict[str, Any], atom_config: dict[str, Any]) -> dict[str, Any]:
    report = compare_method_model_settings(
        {"md": md_config, "siesta_fc_cartesian": atom_config},
        selected_methods=["md", "siesta_fc_cartesian"],
    )
    legacy_mismatches = []
    for mismatch in report["pairwise_mismatch_report"]:
        values = mismatch.get("values", {}) or {}
        legacy_mismatches.append(
            {
                "section": mismatch.get("section"),
                "key": mismatch.get("key"),
                "md": values.get("md"),
                "atom_displacement": values.get("siesta_fc_cartesian"),
                "severity": mismatch.get("severity"),
            }
        )
    report["mismatches"] = legacy_mismatches
    report["warning"] = (
        ""
        if not legacy_mismatches
        else "MD and AtomDisplacement Graph2Mat settings differ; comparison is not strict."
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md-config", type=Path, default=DEFAULT_MD_CONFIG)
    parser.add_argument("--atom-config", type=Path, default=DEFAULT_ATOM_CONFIG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = compare_model_settings(load_yaml(args.md_config), load_yaml(args.atom_config))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
