#!/usr/bin/env python3
"""Fail-closed checkpoint-lineage and holdout-independence audit for the EPC roadmap.

Reads a preregistration document (see ``docs/epc_preregistration_v2.json``)
that declares, per checkpoint used by an EPC evaluation path: the checkpoint
file, how it was selected, the manifests of its known training/holdout rows,
and the structures it is evaluated on. For each declared checkpoint this
script checks:

1. **Checkpoint selection was validation-only.** The declared
   ``checkpoint_selection.monitor_metric`` must be a validation metric
   (contain ``val``, not ``test``/``holdout``); anything else is either a
   confirmed leak (blocking) or unconfirmable (reported as unknown).
2. **Structure/manifest integrity.** Every declared checkpoint, manifest and
   evaluation-structure path is re-hashed; a mismatch against the declared
   sha256, or a missing file, is reported rather than silently ignored.
3. **Geometry independence.** Known training rows are compared against the
   declared EPC evaluation structures for exact/near-duplicate geometry,
   rotation/translation-invariant duplicates and MD-neighbour-frame overlap,
   reusing :mod:`check_geometry_leakage` (the same machinery
   ``Comparison/scripts/ops/audit_geometry_independence.py`` and the
   graph2mat/DeepH benchmark gate already rely on) rather than
   re-implementing the comparison.
4. **Declared provenance gaps never resolve to "independent".** If the
   preregistration document itself says a checkpoint's training provenance is
   incomplete (e.g. an MD-sourced training component with no row-level
   manifest), that checkpoint is reported ``unknown_incomplete_training_provenance``
   even if the rows this script *can* see show no overlap. Absence of
   evidence for the untraced portion is never treated as evidence of
   independence.

Usage::

    audit_epc_lineage.py --preregistration docs/epc_preregistration_v2.json --fail-on-unknown

Exit code is non-zero if any checkpoint has a confirmed lineage problem
(leakage, a hash mismatch, or a confirmed non-validation selection metric).
``--fail-on-unknown`` additionally makes an *unresolvable* finding (missing
manifest, un-declared metric, declared-incomplete provenance) block the exit
code too -- without it, unknowns are reported but do not fail the run, the
same relationship ``--fail-on-invalid``/``--fail-on-warning`` have elsewhere
in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_geometry_leakage import analyze as geometry_leakage_analyze  # noqa: E402
from check_geometry_leakage import structure_path as geometry_structure_path  # noqa: E402
from check_geometry_leakage import parse_run_fdf_geometry  # noqa: E402

SCHEMA = "epc_lineage_audit_v1"
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "Comparison" / "results" / "epc_repair" / "lineage_audit.json"

STATUS_INDEPENDENT = "independent"
STATUS_LEAKAGE_DETECTED = "leakage_detected"
STATUS_STRUCTURE_HASH_MISMATCH = "structure_hash_mismatch"
STATUS_CHECKPOINT_SELECTION_LEAK = "checkpoint_selection_leak"
STATUS_UNKNOWN_INCOMPLETE_TRAINING = "unknown_incomplete_training_provenance"
STATUS_UNKNOWN_SELECTION_METRIC = "unknown_checkpoint_selection_metric"
STATUS_UNKNOWN_MANIFEST_MISSING = "unknown_training_manifest_unreadable"

BLOCKING_ALWAYS = {STATUS_LEAKAGE_DETECTED, STATUS_STRUCTURE_HASH_MISMATCH, STATUS_CHECKPOINT_SELECTION_LEAK}
UNKNOWN_STATUSES = {
    STATUS_UNKNOWN_INCOMPLETE_TRAINING,
    STATUS_UNKNOWN_SELECTION_METRIC,
    STATUS_UNKNOWN_MANIFEST_MISSING,
}
# Priority order for collapsing several component findings into one status:
# the worst (most fail-closed) finding wins.
_STATUS_PRIORITY = [
    STATUS_LEAKAGE_DETECTED,
    STATUS_STRUCTURE_HASH_MISMATCH,
    STATUS_CHECKPOINT_SELECTION_LEAK,
    STATUS_UNKNOWN_INCOMPLETE_TRAINING,
    STATUS_UNKNOWN_SELECTION_METRIC,
    STATUS_UNKNOWN_MANIFEST_MISSING,
]


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path_text: str, *, base_dir: Path) -> Path:
    path = Path(str(path_text))
    return path if path.is_absolute() else (base_dir / path)


def rows_from_frozen_split_manifest(path: Path, *, split_field: str | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if split_field is not None and str(row.get("split") or "") != split_field:
            continue
        out.append({**row, "manifest_path": str(path)})
    return out


def checkpoint_selection_status(selection: dict[str, Any]) -> tuple[str, str]:
    metric = str(selection.get("monitor_metric") or "").strip()
    if not metric:
        return STATUS_UNKNOWN_SELECTION_METRIC, "checkpoint_selection.monitor_metric is not declared"
    lowered = metric.lower()
    if "test" in lowered or "holdout" in lowered:
        return (
            STATUS_CHECKPOINT_SELECTION_LEAK,
            f"monitor_metric {metric!r} references test/holdout data; checkpoint selection must be validation-only",
        )
    if "val" not in lowered:
        return STATUS_UNKNOWN_SELECTION_METRIC, f"monitor_metric {metric!r} cannot be confirmed validation-only"
    return STATUS_INDEPENDENT, f"monitor_metric {metric!r} is validation-only"


def _declared_paths(entry: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    out: list[tuple[str, str, str | None]] = []
    if entry.get("checkpoint_path"):
        out.append(("checkpoint", str(entry["checkpoint_path"]), entry.get("checkpoint_sha256")))
    for structure in entry.get("epc_evaluation_structures") or []:
        out.append((str(structure.get("label") or "structure"), str(structure.get("path")), structure.get("sha256")))
    for manifest in entry.get("known_training_manifests") or []:
        out.append((f"training_manifest:{manifest.get('path')}", str(manifest.get("path")), manifest.get("sha256")))
    for manifest in entry.get("known_holdout_manifests") or []:
        out.append((f"holdout_manifest:{manifest.get('path')}", str(manifest.get("path")), manifest.get("sha256")))
    return out


def _atom_count(row: dict[str, Any]) -> int | None:
    """Cheap O(n) atom count for a manifest row's structure, no pairwise work.

    Used to pre-filter candidate pairs before calling
    :func:`geometry_leakage_analyze`, whose per-pair aligned-RMSD and
    internal-distance-signature checks are O(atoms^2): comparing a 2-atom
    graphene cell against an 11164-atom MATBG supercell's own training rows is
    already known-impossible to be a duplicate by atom count alone, and
    letting the O(n^2) checks run anyway on a check that size is minutes of
    wasted work for an answer this comparison already has for free.
    """
    manifest_path = Path(str(row.get("manifest_path", ""))) if row.get("manifest_path") else None
    path = geometry_structure_path(row, manifest_path)
    if path is None or not path.exists():
        return None
    try:
        _species, coords = parse_run_fdf_geometry(path)
    except OSError:
        return None
    return len(coords) or None


def _combine_statuses(statuses: list[str]) -> str:
    present = set(statuses)
    for status in _STATUS_PRIORITY:
        if status in present:
            return status
    return STATUS_INDEPENDENT if present else STATUS_UNKNOWN_MANIFEST_MISSING


def audit_checkpoint_entry(entry: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    statuses: list[str] = []
    findings: list[str] = []

    sel_status, sel_message = checkpoint_selection_status(entry.get("checkpoint_selection") or {})
    statuses.append(sel_status)
    findings.append(sel_message)

    for label, path_text, declared_sha in _declared_paths(entry):
        path = resolve(path_text, base_dir=base_dir)
        actual_sha = sha256_file(path)
        if actual_sha is None:
            statuses.append(STATUS_UNKNOWN_MANIFEST_MISSING)
            findings.append(f"{label}: {path} does not exist; cannot verify sha256")
        elif declared_sha and actual_sha != declared_sha:
            statuses.append(STATUS_STRUCTURE_HASH_MISMATCH)
            findings.append(f"{label}: {path} sha256={actual_sha} does not match declared {declared_sha}")

    train_rows: list[dict[str, Any]] = []
    for manifest in entry.get("known_training_manifests") or []:
        manifest_path = resolve(str(manifest.get("path") or ""), base_dir=base_dir)
        try:
            rows = rows_from_frozen_split_manifest(manifest_path, split_field=manifest.get("split_field"))
        except (OSError, json.JSONDecodeError) as exc:
            statuses.append(STATUS_UNKNOWN_MANIFEST_MISSING)
            findings.append(f"training manifest {manifest_path} unreadable: {exc}")
            continue
        if not rows:
            statuses.append(STATUS_UNKNOWN_MANIFEST_MISSING)
            findings.append(
                f"training manifest {manifest_path} produced zero rows for split={manifest.get('split_field')!r}"
            )
        train_rows.extend(rows)

    test_rows: list[dict[str, Any]] = []
    for structure in entry.get("epc_evaluation_structures") or []:
        structure_path = resolve(str(structure.get("path") or ""), base_dir=base_dir)
        test_rows.append(
            {
                "sample_id": structure.get("label"),
                "structure_path": str(structure_path),
                "split": "epc_evaluation",
                "method": "epc_reference",
            }
        )

    report: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    if train_rows and test_rows:
        train_counts = {id(row): _atom_count(row) for row in train_rows}
        if any(count is None for count in train_counts.values()):
            statuses.append(STATUS_UNKNOWN_INCOMPLETE_TRAINING)
            findings.append("unreadable training geometries cannot be excluded by atom-count filtering")
        per_structure_summaries: list[dict[str, Any]] = []
        for test_row in test_rows:
            test_count = _atom_count(test_row)
            if test_count is None:
                statuses.append(STATUS_UNKNOWN_MANIFEST_MISSING)
                findings.append("unreadable evaluation geometry: exclusion unproven")
            filtered_train_rows = [
                row
                for row in train_rows
                if test_count is not None and train_counts.get(id(row)) == test_count
            ]
            sub_report, sub_summary = geometry_leakage_analyze(
                filtered_train_rows,
                [test_row],
                rmsd_threshold=1e-4,
                max_diff_threshold=1e-4,
                aligned_rmsd_threshold=1e-4,
                distance_threshold=1e-4,
                neighbor_frame_window=1,
            )
            sub_summary["evaluation_structure"] = test_row.get("sample_id")
            sub_summary["train_rows_compared"] = len(filtered_train_rows)
            sub_summary["train_rows_skipped_atom_count_mismatch"] = len(train_rows) - len(filtered_train_rows)
            report.extend(sub_report)
            per_structure_summaries.append(sub_summary)
        overall_ok = all(item.get("ok") for item in per_structure_summaries)
        summary = {
            "ok": overall_ok,
            "per_evaluation_structure": per_structure_summaries,
            "warning_messages": [
                message for item in per_structure_summaries for message in (item.get("warning_messages") or [])
            ],
            "severe_warnings": [
                message for item in per_structure_summaries for message in (item.get("severe_warnings") or [])
            ],
        }
        if overall_ok:
            statuses.append(STATUS_INDEPENDENT)
            findings.append(
                "no geometry overlap detected between known training rows and EPC evaluation structures "
                "(atom-count pre-filter applied before the pairwise geometry comparison)"
            )
        else:
            statuses.append(STATUS_LEAKAGE_DETECTED)
            findings.append(
                "geometry leakage detected between known training rows and EPC evaluation structures: "
                + "; ".join(summary["warning_messages"] or summary["severe_warnings"])
            )
    else:
        statuses.append(STATUS_UNKNOWN_MANIFEST_MISSING)
        findings.append("could not build both a training-row set and an evaluation-structure set to compare")

    if entry.get("training_provenance_incomplete"):
        statuses.append(STATUS_UNKNOWN_INCOMPLETE_TRAINING)
        findings.append(str(entry.get("training_provenance_gap") or "training provenance is declared incomplete"))

    return {
        "checkpoint_id": entry.get("checkpoint_id"),
        "checkpoint_path": entry.get("checkpoint_path"),
        "component_statuses": statuses,
        "findings": findings,
        "geometry_leakage_report_rows": len(report),
        "geometry_leakage_summary": summary,
        "status": _combine_statuses(statuses),
    }


def build_report(*, preregistration_path: Path) -> dict[str, Any]:
    doc = json.loads(preregistration_path.read_text(encoding="utf-8"))
    entries = doc.get("checkpoint_lineage") or []
    results = [audit_checkpoint_entry(entry, base_dir=REPO_ROOT) for entry in entries]

    blocking = [row for row in results if row["status"] in BLOCKING_ALWAYS]
    unknown = [row for row in results if row["status"] in UNKNOWN_STATUSES]
    independent_lineage_allowed = bool(results) and all(row["status"] == STATUS_INDEPENDENT for row in results)

    return {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "preregistration_path": str(preregistration_path),
        "preregistration_content_sha256_declared": doc.get("content_sha256"),
        "checkpoints_checked": len(results),
        "results": results,
        "summary": {
            "independent_count": sum(1 for row in results if row["status"] == STATUS_INDEPENDENT),
            "blocking_count": len(blocking),
            "unknown_count": len(unknown),
            "blocking_checkpoint_ids": [row["checkpoint_id"] for row in blocking],
            "unknown_checkpoint_ids": [row["checkpoint_id"] for row in unknown],
            "independent_lineage_allowed": independent_lineage_allowed,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fail-on-unknown",
        action="store_true",
        help="Also fail the exit code when a checkpoint's lineage cannot be resolved (missing "
        "manifest, undeclared selection metric, or declared-incomplete training provenance).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(preregistration_path=args.preregistration)
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True, ensure_ascii=False))
    if report["summary"]["blocking_count"] > 0:
        return 1
    if args.fail_on_unknown and report["summary"]["unknown_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
