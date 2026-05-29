#!/usr/bin/env python3
"""Build a release manifest for external Graph2Mat-vs-DeepH artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


RELEASE_MANIFEST_SCHEMA = "graph2mat_deeph_artifact_release_manifest_v1"
FORBIDDEN_REFERENCE_NAME = "ML_prediction.HSX"
RAW_ARTIFACT_ROLE_BY_KEY = {
    "run_fdf": "siesta_run_fdf",
    "run_output": "siesta_run_output",
    "metadata": "siesta_metadata",
    "reference_hsx": "siesta_reference_hsx",
    "reference_tshs": "siesta_reference_tshs",
    "reference_tsde": "siesta_reference_tsde",
    "struct_out": "siesta_struct_out",
    "xv": "siesta_xv",
    "orb_indx": "siesta_orb_indx",
}
RUN_FIXED_FILES = {
    "run_training_sweep_manifest": (Path("sweep/training_sweep_manifest.json"), True),
    "run_ranking_summary": (Path("summary/ranking/ranking_summary.json"), False),
    "run_report_summary": (Path("summary/report/report_summary.json"), False),
    "run_final_statistics": (Path("summary/final_statistics/final_statistics.json"), False),
}
WORKFLOW_FIXED_FILES = {
    "workflow_validated_protocol": (Path("protocol/validated_protocol.json"), True),
    "workflow_search_plan": (Path("search/search_plan.json"), True),
    "workflow_selected_configs": (Path("selection/selected_configs.json"), True),
    "workflow_robust_rerun_plan": (Path("selection/robust_rerun_plan.json"), True),
    "workflow_final_statistics": (Path("final_test/final_statistics.json"), True),
    "workflow_final_report": (Path("report/report_summary.json"), True),
    "workflow_evidence_bundle": (Path("evidence/evidence_bundle_manifest.json"), True),
    "workflow_gate_status": (Path("gate_status.json"), False),
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def relative_path(path: Path, roots: list[Path]) -> str:
    for root in roots:
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    for root in roots:
        try:
            return str(path.resolve(strict=False).relative_to(root.resolve(strict=False)))
        except ValueError:
            continue
    return path.name


def resolve_artifact_path(value: Any, *, dataset_root: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    candidate = dataset_root / path
    return candidate


def release_file_entry(
    path: Path,
    *,
    role: str,
    source_group: str,
    allowed_roots: list[Path],
    required_for_reproduction: bool,
    required_for_audit: bool,
) -> dict[str, Any]:
    exists = path.exists()
    is_symlink = path.is_symlink()
    resolved = path.resolve(strict=False) if exists else path
    symlink_outside = bool(is_symlink and not any(is_within(resolved, root) for root in allowed_roots))
    safe_to_hash = exists and path.is_file() and not symlink_outside
    return {
        "role": role,
        "relative_path": relative_path(path, allowed_roots),
        "absolute_path_optional": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if safe_to_hash else None,
        "sha256": file_sha256(path) if safe_to_hash else None,
        "required_for_reproduction": required_for_reproduction,
        "required_for_audit": required_for_audit,
        "source_group": source_group,
        "is_symlink": is_symlink,
        "symlink_target": str(resolved) if is_symlink else "",
        "symlink_outside_allowed_roots": symlink_outside,
    }


def add_entry(
    entries: list[dict[str, Any]],
    path: Path,
    *,
    role: str,
    source_group: str,
    allowed_roots: list[Path],
    required_for_reproduction: bool,
    required_for_audit: bool,
    seen: set[tuple[str, str]],
) -> None:
    key = (role, str(path))
    if key in seen:
        return
    seen.add(key)
    entries.append(
        release_file_entry(
            path,
            role=role,
            source_group=source_group,
            allowed_roots=allowed_roots,
            required_for_reproduction=required_for_reproduction,
            required_for_audit=required_for_audit,
        )
    )


def add_dataset_entries(
    entries: list[dict[str, Any]],
    *,
    dataset_root: Path,
    allowed_roots: list[Path],
    seen: set[tuple[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    benchmark_manifest = read_json(dataset_root / "benchmark_dataset_manifest.json")
    frozen_split = read_json(dataset_root / "frozen_split_manifest.json")
    for role, path in (
        ("benchmark_dataset_manifest", dataset_root / "benchmark_dataset_manifest.json"),
        ("artifact_validation", dataset_root / "artifact_validation.json"),
        ("material_provenance", dataset_root / "material_provenance.json"),
        ("frozen_split_manifest", dataset_root / "frozen_split_manifest.json"),
    ):
        add_entry(
            entries,
            path,
            role=role,
            source_group="dataset",
            allowed_roots=allowed_roots,
            required_for_reproduction=True,
            required_for_audit=True,
            seen=seen,
        )
    for row in frozen_split.get("rows") or []:
        if not isinstance(row, dict):
            continue
        artifact_paths = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
        for artifact_key, role in sorted(RAW_ARTIFACT_ROLE_BY_KEY.items()):
            raw_path = artifact_paths.get(artifact_key) or row.get(f"{artifact_key}_path")
            if not raw_path:
                continue
            add_entry(
                entries,
                resolve_artifact_path(raw_path, dataset_root=dataset_root),
                role=role,
                source_group="dataset",
                allowed_roots=allowed_roots,
                required_for_reproduction=True,
                required_for_audit=True,
                seen=seen,
            )
    return benchmark_manifest, frozen_split


def scan_optional_run_entries(
    entries: list[dict[str, Any]],
    *,
    root: Path,
    source_group: str,
    allowed_roots: list[Path],
    seen: set[tuple[str, str]],
) -> dict[str, int]:
    counts = {
        "telemetry": 0,
        "equivalence": 0,
        "deeph_predictions": 0,
        "deeph_processed": 0,
        "graph2mat_predictions": 0,
    }
    if not root.exists():
        return counts
    for path in sorted(root.rglob("telemetry/*.json")):
        if path.is_file() or path.is_symlink():
            counts["telemetry"] += 1
            add_entry(
                entries,
                path,
                role="telemetry",
                source_group="telemetry",
                allowed_roots=allowed_roots,
                required_for_reproduction=False,
                required_for_audit=True,
                seen=seen,
            )
    for name in ("adapter_manifest.json", "raw_global_equivalence_evidence.json"):
        for path in sorted(root.rglob(name)):
            if path.is_file() or path.is_symlink():
                counts["equivalence"] += 1
                add_entry(
                    entries,
                    path,
                    role="deeph_equivalence_manifest" if name == "adapter_manifest.json" else "deeph_raw_global_equivalence_evidence",
                    source_group="equivalence",
                    allowed_roots=allowed_roots,
                    required_for_reproduction=False,
                    required_for_audit=True,
                    seen=seen,
                )
    for name in ("hamiltonians_pred.h5", "rh_pred.h5"):
        for path in sorted(root.rglob(name)):
            if path.is_file() or path.is_symlink():
                counts["deeph_predictions"] += 1
                add_entry(
                    entries,
                    path,
                    role="deeph_prediction_hdf5",
                    source_group=source_group,
                    allowed_roots=allowed_roots,
                    required_for_reproduction=True,
                    required_for_audit=True,
                    seen=seen,
                )
    for name in ("hamiltonians.h5", "overlaps.h5", "orbital_types.dat"):
        for path in sorted(root.rglob(name)):
            if path.is_file() or path.is_symlink():
                counts["deeph_processed"] += 1
                add_entry(
                    entries,
                    path,
                    role="deeph_processed_reference",
                    source_group=source_group,
                    allowed_roots=allowed_roots,
                    required_for_reproduction=True,
                    required_for_audit=True,
                    seen=seen,
                )
    for path in sorted(root.rglob("ML_prediction.HSX")):
        if path.is_file() or path.is_symlink():
            counts["graph2mat_predictions"] += 1
            add_entry(
                entries,
                path,
                role="graph2mat_prediction_hsx",
                source_group=source_group,
                allowed_roots=allowed_roots,
                required_for_reproduction=True,
                required_for_audit=True,
                seen=seen,
            )
    return counts


def add_fixed_root_entries(
    entries: list[dict[str, Any]],
    *,
    root: Path,
    files: dict[str, tuple[Path, bool]],
    source_group: str,
    allowed_roots: list[Path],
    seen: set[tuple[str, str]],
) -> None:
    for role, (relative, required) in sorted(files.items()):
        add_entry(
            entries,
            root / relative,
            role=role,
            source_group=source_group,
            allowed_roots=allowed_roots,
            required_for_reproduction=False,
            required_for_audit=required,
            seen=seen,
        )


def forbidden_reference_findings(entries: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    for row in entries:
        role = str(row.get("role") or "")
        path = str(row.get("relative_path") or row.get("absolute_path_optional") or "")
        if FORBIDDEN_REFERENCE_NAME in path and "reference" in role:
            findings.append(f"{role}: {path}")
    return findings


def missing_required(entries: list[dict[str, Any]], *, strict: bool) -> list[str]:
    missing: list[str] = []
    for row in entries:
        if not (row.get("required_for_reproduction") or row.get("required_for_audit")):
            continue
        if row.get("exists") is not True:
            missing.append(f"{row.get('role')}:{row.get('relative_path')}")
        elif row.get("symlink_outside_allowed_roots"):
            missing.append(f"{row.get('role')}:{row.get('relative_path')}:symlink_outside_allowed_roots")
        elif strict and row.get("sha256") in (None, ""):
            missing.append(f"{row.get('role')}:{row.get('relative_path')}:missing_sha256")
    return sorted(set(missing))


def required_role_missing(entries: list[dict[str, Any]], role_prefix: str) -> bool:
    return not any(str(row.get("role") or "").startswith(role_prefix) and row.get("exists") is True for row in entries)


def build_release_manifest(
    *,
    dataset_root: Path,
    run_root: Path | None = None,
    workflow_root: Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    run_root = Path(run_root) if run_root is not None else None
    workflow_root = Path(workflow_root) if workflow_root is not None else None
    allowed_roots = [dataset_root]
    if run_root is not None:
        allowed_roots.append(run_root)
    if workflow_root is not None:
        allowed_roots.append(workflow_root)

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    benchmark_manifest, frozen_split = add_dataset_entries(
        entries,
        dataset_root=dataset_root,
        allowed_roots=allowed_roots,
        seen=seen,
    )
    if run_root is not None:
        add_fixed_root_entries(
            entries,
            root=run_root,
            files=RUN_FIXED_FILES,
            source_group="run",
            allowed_roots=allowed_roots,
            seen=seen,
        )
        scan_optional_run_entries(
            entries,
            root=run_root,
            source_group="run",
            allowed_roots=allowed_roots,
            seen=seen,
        )
    if workflow_root is not None:
        add_fixed_root_entries(
            entries,
            root=workflow_root,
            files=WORKFLOW_FIXED_FILES,
            source_group="workflow",
            allowed_roots=allowed_roots,
            seen=seen,
        )
        scan_optional_run_entries(
            entries,
            root=workflow_root,
            source_group="workflow",
            allowed_roots=allowed_roots,
            seen=seen,
        )

    missing = missing_required(entries, strict=strict)
    if run_root is not None and required_role_missing(entries, "telemetry"):
        missing.append("telemetry:any")
    if run_root is not None and required_role_missing(entries, "deeph_equivalence"):
        missing.append("deeph_equivalence:any")
    if run_root is not None and required_role_missing(entries, "graph2mat_prediction_hsx"):
        missing.append("graph2mat_prediction_hsx:any")
    if workflow_root is not None and required_role_missing(entries, "workflow_final_report"):
        missing.append("workflow_final_report:any")
    forbidden = forbidden_reference_findings(entries)
    status = "complete"
    if forbidden:
        status = "invalid"
    elif missing:
        status = "invalid" if strict else "partial"
    return {
        "schema": RELEASE_MANIFEST_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_commit": repo_commit(),
        "dataset_root": str(dataset_root),
        "run_root": str(run_root) if run_root is not None else "",
        "workflow_root": str(workflow_root) if workflow_root is not None else "",
        "benchmark_dataset_id": benchmark_manifest.get("benchmark_dataset_id", ""),
        "split_hash": frozen_split.get("split_hash") or (benchmark_manifest.get("frozen_split_manifest") or {}).get("split_hash", ""),
        "artifact_contract_version": (
            frozen_split.get("artifact_contract_version")
            or benchmark_manifest.get("artifact_contract_version")
            or ""
        ),
        "files": entries,
        "missing_required": sorted(set(missing)),
        "forbidden_reference_findings": forbidden,
        "status": status,
        "external_storage_url": "",
        "upload_instructions": (
            "Upload every listed file to immutable external storage, keep relative_path stable, "
            "and record the storage URL plus this manifest hash in the paper/release notes."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--workflow-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_release_manifest(
        dataset_root=args.dataset_root,
        run_root=args.run_root,
        workflow_root=args.workflow_root,
        strict=bool(args.strict),
    )
    write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 1 if args.strict and manifest["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
