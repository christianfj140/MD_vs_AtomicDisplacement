#!/usr/bin/env python3
"""Verify paper-ready datasets declared by a Graph2Mat-vs-DeepH protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from benchmark_manifest import write_benchmark_manifests  # noqa: E402
from g2m_deeph_protocol import load_protocol  # noqa: E402
from g2m_deeph_training_sweep import json_safe  # noqa: E402
from joint_artifact_contract import G2M_DEEPH_BENCHMARK_PROFILE, validate_dataset  # noqa: E402


DATASET_VERIFICATION_SCHEMA = "graph2mat_deeph_protocol_dataset_verification_v1"
FORBIDDEN_REFERENCE_NAME = "ML_prediction.HSX"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON file must contain an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_path(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return base_dir / path


def forbidden_reference_findings(payload: Any, *, path: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{path}.{key}" if path else str(key)
            findings.extend(forbidden_reference_findings(value, path=child))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(forbidden_reference_findings(value, path=f"{path}[{index}]"))
    elif isinstance(payload, str) and FORBIDDEN_REFERENCE_NAME in payload:
        findings.append(f"{path}: {payload}")
    return findings


def split_counts(split_manifest: dict[str, Any]) -> dict[str, int]:
    raw = split_manifest.get("split_counts")
    if isinstance(raw, dict):
        counts: dict[str, int] = {}
        for split in ("train", "validation", "test"):
            try:
                counts[split] = int(raw.get(split) or 0)
            except (TypeError, ValueError):
                counts[split] = 0
        return counts
    counts = {"train": 0, "validation": 0, "test": 0}
    for row in split_manifest.get("rows") or []:
        if isinstance(row, dict) and row.get("split") in counts:
            counts[str(row["split"])] += 1
    return counts


def sample_dirs_from_split(split_manifest: dict[str, Any], *, split_path: Path) -> list[Path]:
    sample_dirs: list[Path] = []
    for row in split_manifest.get("rows") or []:
        if not isinstance(row, dict):
            continue
        raw = row.get("sample_dir")
        if not raw:
            continue
        sample_dirs.append(resolve_path(raw, base_dir=split_path.parent))
    return sample_dirs


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()


def maybe_write_manifests(
    *,
    dataset_root: Path,
    dataset: dict[str, Any],
    strict: bool,
    blockers: list[str],
    warnings: list[str],
) -> None:
    split_root = resolve_path(dataset.get("split_root") or dataset_root / "splits", base_dir=dataset_root)
    artifact_validation_path = resolve_path(
        dataset.get("artifact_validation") or dataset_root / "artifact_validation.json",
        base_dir=dataset_root,
    )
    if not split_root.exists():
        blockers.append(f"cannot write manifests because split_root is missing: {split_root}")
        return
    if not artifact_validation_path.exists():
        blockers.append(f"cannot write manifests because artifact_validation.json is missing: {artifact_validation_path}")
        return
    try:
        write_benchmark_manifests(
            dataset_root=dataset_root,
            split_root=split_root,
            artifact_validation_path=artifact_validation_path,
            strict_paper_ready_provenance=strict,
        )
        warnings.append(f"wrote benchmark_dataset_manifest.json and frozen_split_manifest.json from split_root={split_root}")
    except RuntimeError as exc:
        blockers.append(f"write_benchmark_manifests failed: {exc}")


def verify_dataset_entry(
    dataset: dict[str, Any],
    *,
    protocol_dir: Path,
    strict: bool,
    write_manifests: bool,
) -> dict[str, Any]:
    dataset_id = str(dataset.get("dataset_id") or "dataset").strip()
    dataset_root = resolve_path(dataset.get("dataset_root"), base_dir=protocol_dir)
    benchmark_path = resolve_path(dataset.get("benchmark_dataset_manifest"), base_dir=protocol_dir)
    split_path = resolve_path(dataset.get("frozen_split_manifest"), base_dir=protocol_dir)
    artifact_path = resolve_path(
        dataset.get("artifact_validation") or dataset_root / "artifact_validation.json",
        base_dir=protocol_dir,
    )
    blockers: list[str] = []
    warnings: list[str] = []
    evidence_paths = {
        "dataset_root": str(dataset_root),
        "benchmark_dataset_manifest": str(benchmark_path),
        "frozen_split_manifest": str(split_path),
        "artifact_validation": str(artifact_path),
    }
    if not dataset_root.exists() or not dataset_root.is_dir():
        blockers.append(f"dataset_root is missing or not a directory: {dataset_root}")
        return {
            "dataset_id": dataset_id,
            "status": "invalid",
            "evidence_paths": evidence_paths,
            "blockers": blockers,
            "warnings": warnings,
        }

    if write_manifests and (not benchmark_path.exists() or not split_path.exists()):
        maybe_write_manifests(
            dataset_root=dataset_root,
            dataset=dataset,
            strict=strict,
            blockers=blockers,
            warnings=warnings,
        )

    missing = [
        label
        for label, path in (
            ("artifact_validation", artifact_path),
            ("benchmark_dataset_manifest", benchmark_path),
            ("frozen_split_manifest", split_path),
        )
        if not path.exists()
    ]
    if missing:
        blockers.append("missing required dataset evidence: " + ", ".join(missing))
        return {
            "dataset_id": dataset_id,
            "status": "invalid",
            "evidence_paths": evidence_paths,
            "blockers": blockers,
            "warnings": warnings,
        }

    try:
        artifact_validation = read_json(artifact_path)
        benchmark_manifest = read_json(benchmark_path)
        split_manifest = read_json(split_path)
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        blockers.append(f"failed to read dataset evidence JSON: {exc}")
        return {
            "dataset_id": dataset_id,
            "status": "invalid",
            "evidence_paths": evidence_paths,
            "blockers": blockers,
            "warnings": warnings,
        }

    if artifact_validation.get("valid") is not True:
        blockers.append("artifact_validation.valid is not true")
    if benchmark_manifest.get("benchmark_ready") is not True or str(benchmark_manifest.get("validation_status") or "") == "invalid":
        blockers.append("benchmark_dataset_manifest is not benchmark_ready")
    if split_manifest.get("valid") is False:
        blockers.append("frozen_split_manifest.valid is false")
    counts = split_counts(split_manifest)
    empty_splits = [split for split, count in sorted(counts.items()) if count <= 0]
    if empty_splits:
        blockers.append("frozen split has empty split(s): " + ", ".join(empty_splits))

    recorded_split = benchmark_manifest.get("frozen_split_manifest")
    if isinstance(recorded_split, dict):
        recorded_hash = str(recorded_split.get("split_hash") or "").strip()
        split_hash = str(split_manifest.get("split_hash") or "").strip()
        if recorded_hash and split_hash and recorded_hash != split_hash:
            blockers.append(f"split_hash mismatch: benchmark={recorded_hash} frozen_split={split_hash}")
        recorded_path = str(recorded_split.get("path") or "").strip()
        if recorded_path:
            resolved_recorded = resolve_path(recorded_path, base_dir=benchmark_path.parent)
            if not _same_path(resolved_recorded, split_path):
                blockers.append(f"benchmark frozen_split_manifest.path does not match protocol path: {recorded_path}")

    forbidden = []
    for label, payload in (
        ("artifact_validation", artifact_validation),
        ("benchmark_dataset_manifest", benchmark_manifest),
        ("frozen_split_manifest", split_manifest),
    ):
        forbidden.extend(f"{label}:{finding}" for finding in forbidden_reference_findings(payload))
    if forbidden:
        blockers.append("forbidden reference path detected: " + "; ".join(forbidden[:8]))

    provenance = benchmark_manifest.get("provenance_status") if isinstance(benchmark_manifest.get("provenance_status"), dict) else {}
    if strict:
        if provenance.get("valid") is not True:
            missing_provenance = ", ".join(str(item) for item in provenance.get("missing") or []) or "provenance_status.valid is not true"
            blockers.append("strict provenance is incomplete: " + missing_provenance)
        sample_dirs = sample_dirs_from_split(split_manifest, split_path=split_path)
        dataset_validation = validate_dataset(
            dataset_root,
            snapshot_dirs=sample_dirs if sample_dirs else None,
            validation_profile=G2M_DEEPH_BENCHMARK_PROFILE,
        )
        if not dataset_validation.valid:
            blockers.extend(f"joint artifact/provenance validation failed: {error}" for error in dataset_validation.errors)
        warnings.extend(dataset_validation.warnings)

    return {
        "dataset_id": dataset_id,
        "status": "valid" if not blockers else "invalid",
        "evidence_paths": evidence_paths,
        "benchmark_ready": benchmark_manifest.get("benchmark_ready"),
        "artifact_validation_valid": artifact_validation.get("valid"),
        "split_counts": counts,
        "split_hash": split_manifest.get("split_hash"),
        "benchmark_dataset_id": benchmark_manifest.get("benchmark_dataset_id"),
        "provenance_status": provenance,
        "blockers": blockers,
        "warnings": warnings,
    }


def verify_protocol_datasets(
    *,
    protocol_path: Path,
    output_path: Path,
    strict: bool = False,
    write_manifests: bool = False,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    datasets = [
        verify_dataset_entry(
            dict(dataset),
            protocol_dir=protocol_path.parent,
            strict=strict,
            write_manifests=write_manifests,
        )
        for dataset in protocol.get("datasets") or []
        if isinstance(dataset, dict)
    ]
    blockers = [
        f"{dataset['dataset_id']}: {blocker}"
        for dataset in datasets
        for blocker in dataset.get("blockers") or []
    ]
    payload = {
        "schema": DATASET_VERIFICATION_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_hash": protocol.get("protocol_hash"),
        "protocol_path": str(protocol_path),
        "strict": strict,
        "write_manifests": write_manifests,
        "status": "valid" if not blockers else "invalid",
        "datasets": datasets,
        "blockers": blockers,
        "warnings": [
            f"{dataset['dataset_id']}: {warning}"
            for dataset in datasets
            for warning in dataset.get("warnings") or []
        ],
    }
    write_json(output_path, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--write-manifests", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = verify_protocol_datasets(
        protocol_path=args.protocol,
        output_path=args.output,
        strict=bool(args.strict),
        write_manifests=bool(args.write_manifests),
    )
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if payload.get("status") == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
