#!/usr/bin/env python3
"""Compare Graph2Mat training settings for strict MD-vs-AtomDisplacement runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


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
IGNORED_TRAINER_KEYS = {"logger"}


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


def comparable_training_settings(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {}) or {}
    data = {
        key: value
        for key, value in (training.get("data", {}) or {}).items()
        if key not in IGNORED_DATA_KEYS
    }
    trainer = {
        key: value
        for key, value in (training.get("trainer", {}) or {}).items()
        if key not in IGNORED_TRAINER_KEYS
    }
    return {
        "data": data,
        "model": training.get("model", {}) or {},
        "trainer": trainer,
    }


def compare_model_settings(md_config: dict[str, Any], atom_config: dict[str, Any]) -> dict[str, Any]:
    md_settings = comparable_training_settings(md_config)
    atom_settings = comparable_training_settings(atom_config)
    mismatches = []
    for section in ("data", "model", "trainer"):
        keys = sorted(set(md_settings.get(section, {})) | set(atom_settings.get(section, {})))
        for key in keys:
            md_value = normalize((md_settings.get(section, {}) or {}).get(key))
            atom_value = normalize((atom_settings.get(section, {}) or {}).get(key))
            if md_value != atom_value:
                mismatches.append(
                    {
                        "section": section,
                        "key": key,
                        "md": md_value,
                        "atom_displacement": atom_value,
                    }
                )
    return {
        "ok": not mismatches,
        "model_config_hash": settings_hash({"md": md_settings, "atom_displacement": atom_settings}),
        "md_model_config_hash": settings_hash(md_settings),
        "atom_displacement_model_config_hash": settings_hash(atom_settings),
        "mismatches": mismatches,
        "warning": "" if not mismatches else "MD and AtomDisplacement Graph2Mat settings differ; comparison is not strict.",
    }


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
