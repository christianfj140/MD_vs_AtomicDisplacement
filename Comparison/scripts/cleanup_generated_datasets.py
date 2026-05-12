#!/usr/bin/env python3
"""Safely remove generated dataset artifacts without touching source configs."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
KEEP_DATASET_FILENAMES = {".gitkeep"}
KEEP_DATASET_SUFFIXES = {".psf", ".psml"}
GENERATED_RESULTS_GROUPS = ("results_md", "results_atomdisp", "results_random_cartesian")


def _is_generated_results_dir(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith(".")


def _is_generated_experiment_dir(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith(".") or path.name in GENERATED_RESULTS_GROUPS:
        return False
    if path.name[:8].isdigit():
        return True
    generated_markers = (
        "experiment_manifest.yaml",
        "summary",
        "cross_evaluations",
        "cross_predictions",
        "common_tests",
    )
    return any((path / marker).exists() for marker in generated_markers)


def generated_dataset_targets(repo_root: Path = REPO_ROOT) -> list[Path]:
    targets: list[Path] = []
    for dataset_root in (repo_root / "MD" / "dataset", repo_root / "AtomDisplacement" / "dataset"):
        if dataset_root.exists():
            for child in dataset_root.iterdir():
                if child.name in KEEP_DATASET_FILENAMES or child.suffix in KEEP_DATASET_SUFFIXES:
                    continue
                targets.append(child)

    comparison = repo_root / "Comparison"
    workspaces = comparison / "workspaces"
    if workspaces.exists():
        targets.append(workspaces)

    results = comparison / "results"
    for group in GENERATED_RESULTS_GROUPS:
        root = results / group
        if root.exists():
            targets.extend(child for child in root.iterdir() if _is_generated_results_dir(child))
    if results.exists():
        targets.extend(
            child
            for child in results.iterdir()
            if _is_generated_experiment_dir(child)
        )
    return sorted(set(targets), key=lambda path: path.as_posix())


def cleanup_generated_datasets(
    repo_root: Path = REPO_ROOT,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    targets = generated_dataset_targets(repo_root)
    removed: list[str] = []
    for target in targets:
        removed.append(str(target))
        if dry_run:
            continue
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "removed": removed,
        "preserved": [
            "MD/pipeline_config.yaml",
            "AtomDisplacement/pipeline_config.yaml",
            "Comparison/config/shared_siesta_settings.yaml",
            "source code under */scripts and */ui",
            "pseudopotentials *.psf/*.psml in MD/dataset and AtomDisplacement/dataset",
        ],
    }
    if not dry_run:
        log_path = repo_root / "Comparison" / "generated_dataset_cleanup_manifest.json"
        log_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = cleanup_generated_datasets(args.repo_root, dry_run=args.dry_run)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
