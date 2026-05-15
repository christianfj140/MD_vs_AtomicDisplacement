#!/usr/bin/env python3
"""Safely remove generated dataset artifacts without touching source configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
KEEP_DATASET_FILENAMES = {".gitkeep"}
KEEP_DATASET_SUFFIXES = {".psf", ".psml"}
GENERATED_RESULTS_GROUPS = ("results_md", "results_atomdisp", "results_random_cartesian")
RESULTS_METHOD_BY_GROUP = {
    "results_md": "md",
    "results_atomdisp": "siesta_fc_cartesian",
    "results_random_cartesian": "random_cartesian",
}
CLEANUP_MANIFEST_RELATIVE_PATH = "Comparison/generated_dataset_cleanup_manifest.json"


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


def _relative_path(repo_root: Path, path: Path) -> str:
    repo_root = repo_root.resolve()
    candidate = path if path.is_absolute() else repo_root / path
    try:
        return candidate.absolute().relative_to(repo_root).as_posix()
    except ValueError:
        return candidate.as_posix()


def _target_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]


def _directory_size_bytes(path: Path) -> int:
    if path.is_symlink():
        return 0
    if not path.exists():
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


def _dataset_kind_and_method(repo_root: Path, path: Path) -> tuple[str, str]:
    relative = _relative_path(repo_root, path)
    parts = relative.split("/")
    if relative.startswith("MD/dataset/"):
        return "source_dataset", "md"
    if relative.startswith("AtomDisplacement/dataset/RandomCartesian_steps"):
        return "source_dataset", "random_cartesian"
    if relative.startswith("AtomDisplacement/dataset/"):
        return "source_dataset", "siesta_fc_cartesian"
    if relative == "Comparison/workspaces":
        return "workspace_cache", "mixed"
    if len(parts) >= 3 and parts[0] == "Comparison" and parts[1] == "results":
        if parts[2] in RESULTS_METHOD_BY_GROUP:
            return "archived_dataset", RESULTS_METHOD_BY_GROUP[parts[2]]
        return "experiment_results", "mixed"
    return "generated_artifact", "unknown"


def _manifest_metadata(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        return {}
    manifest_paths = []
    if path.is_dir():
        manifest_paths.extend(sorted(path.glob("run_*/manifest.json")))
        manifest_paths.extend(sorted(path.glob("*/manifest.json")))
        manifest_paths.extend(sorted(path.glob("experiment_manifest.yaml")))
    for manifest_path in manifest_paths:
        try:
            if manifest_path.suffix == ".json":
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                payload = {}
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def generated_dataset_records(repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    repo_root = repo_root.resolve()
    records: list[dict[str, Any]] = []
    for target in generated_dataset_targets(repo_root):
        relative = _relative_path(repo_root, target)
        kind, method = _dataset_kind_and_method(repo_root, target)
        metadata = _manifest_metadata(target)
        try:
            stat = target.lstat() if target.is_symlink() else target.stat()
        except OSError:
            stat = None
        records.append(
            {
                "id": _target_id(relative),
                "name": target.name,
                "path": relative,
                "relative_path": relative,
                "kind": kind,
                "method": metadata.get("method_id") or method,
                "dataset_label": metadata.get("dataset_label") or target.name,
                "dataset_size": metadata.get("dataset_size"),
                "run_id": metadata.get("run_id") or metadata.get("experiment_id"),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds") if stat else None,
                "bytes": _directory_size_bytes(target),
                "warning": (
                    "Borrar este artefacto fisico no edita recetas ni configuraciones; "
                    "puede dejar resultados historicos apuntando a una ruta eliminada."
                    if kind in {"source_dataset", "workspace_cache"}
                    else ""
                ),
            }
        )
    return records


def _safe_remove_target(repo_root: Path, target: Path) -> None:
    repo_root = repo_root.resolve()
    if target.is_symlink():
        raise RuntimeError(f"No se borra un enlace simbolico: {target}")
    resolved = target.resolve()
    if resolved == repo_root:
        raise RuntimeError(f"No se borra la raiz del repositorio: {target}")
    if repo_root not in resolved.parents:
        raise RuntimeError(f"Ruta fuera del repositorio: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def _target_from_record(repo_root: Path, record: dict[str, Any]) -> Path:
    relative_path = str(record.get("relative_path") or "").strip()
    if not relative_path:
        raise RuntimeError("Registro de cleanup sin relative_path portable.")
    raw_candidate = repo_root / relative_path
    candidate = raw_candidate.resolve(strict=False)
    repo_root_resolved = repo_root.resolve(strict=False)
    try:
        candidate.relative_to(repo_root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"Ruta de cleanup fuera del repositorio: {relative_path}") from exc
    return raw_candidate


def _portable_record(record: dict[str, Any]) -> dict[str, Any]:
    portable = dict(record)
    relative_path = str(portable.get("relative_path") or "")
    portable["path"] = relative_path
    return portable


def _write_cleanup_manifest(repo_root: Path, manifest: dict[str, Any]) -> None:
    log_path = repo_root / CLEANUP_MANIFEST_RELATIVE_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cleanup_selected_generated_datasets(
    repo_root: Path = REPO_ROOT,
    *,
    target_ids: list[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    target_ids = list(dict.fromkeys(str(target_id) for target_id in target_ids if str(target_id).strip()))
    if not target_ids:
        raise RuntimeError("Selecciona al menos un dataset generado para borrar.")
    records = generated_dataset_records(repo_root)
    by_id = {record["id"]: record for record in records}
    unknown = [target_id for target_id in target_ids if target_id not in by_id]
    if unknown:
        raise RuntimeError(f"IDs de dataset no reconocidos o ya borrados: {unknown}.")
    removed: list[str] = []
    selected_records = [by_id[target_id] for target_id in target_ids]
    for record in selected_records:
        target = _target_from_record(repo_root, record)
        removed.append(str(record["relative_path"]))
        if not dry_run:
            _safe_remove_target(repo_root, target)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "removed": removed,
        "selected": [_portable_record(record) for record in selected_records],
        "preserved": [
            "MD/pipeline_config.yaml",
            "AtomDisplacement/pipeline_config.yaml",
            "Comparison/config/shared_siesta_settings.yaml",
            "source code under */scripts and */ui",
            "pseudopotentials *.psf/*.psml in MD/dataset and AtomDisplacement/dataset",
        ],
    }
    if not dry_run:
        _write_cleanup_manifest(repo_root, manifest)
    return manifest


def cleanup_generated_datasets(
    repo_root: Path = REPO_ROOT,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    targets = generated_dataset_targets(repo_root)
    removed: list[str] = []
    for target in targets:
        removed.append(_relative_path(repo_root, target))
        if dry_run:
            continue
        _safe_remove_target(repo_root, target)
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
        _write_cleanup_manifest(repo_root, manifest)
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
