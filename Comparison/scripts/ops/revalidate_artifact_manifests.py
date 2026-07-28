#!/usr/bin/env python3
"""Revalidate archived artifact manifests without modifying historical data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from joint_artifact_contract import validate_recorded_snapshots  # noqa: E402


SCHEMA = "strict_artifact_revalidation_report_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON root is not an object")
    return payload


def revalidate_manifest(path: Path, *, cache: dict[Path, Any] | None = None) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "path": str(path),
            "sha256": file_sha256(path) if path.is_file() else "",
            "status": "invalid_manifest",
            "declared_valid": False,
            "declared_benchmark_ready": False,
            "revalidated_valid": False,
            "problems": [str(exc)],
        }
    results, errors = validate_recorded_snapshots(payload, base_dir=path.parent, cache=cache)
    benchmark_path = path.parent / "benchmark_dataset_manifest.json"
    benchmark = read_json(benchmark_path) if benchmark_path.exists() else {}
    declared_valid = payload.get("valid") is True
    revalidated_valid = bool(results) and not errors
    status = (
        "valid"
        if revalidated_valid
        else "quarantined"
        if declared_valid or benchmark.get("benchmark_ready") is True
        else "invalid"
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "status": status,
        "declared_valid": declared_valid,
        "declared_benchmark_ready": benchmark.get("benchmark_ready") is True,
        "revalidated_valid": revalidated_valid,
        "snapshot_count": len(results),
        "valid_snapshot_count": sum(result.valid for result in results),
        "invalid_snapshot_count": sum(not result.valid for result in results),
        "parser_status_counts": dict(
            sorted(
                Counter(
                    str(result.siesta_run_status.get("parser_status") or "missing_status")
                    for result in results
                ).items()
            )
        ),
        "problems": errors,
    }


def build_report(root: Path) -> dict[str, Any]:
    cache: dict[Path, Any] = {}
    rows = [
        revalidate_manifest(path, cache=cache)
        for path in sorted(Path(root).rglob("artifact_validation.json"))
    ]
    counts = Counter(str(row["status"]) for row in rows)
    return {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scan_root": str(Path(root).resolve()),
        "manifest_count": len(rows),
        "unique_snapshot_count": len(cache),
        "status_counts": dict(sorted(counts.items())),
        "all_revalidated_valid": bool(rows) and all(row["revalidated_valid"] for row in rows),
        "manifests": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "Comparison")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-invalid", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("manifest_count", "status_counts", "all_revalidated_valid")}))
    return 1 if args.fail_on_invalid and not report["all_revalidated_valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
