#!/usr/bin/env python3
"""Strict SIESTA reference matrix selection shared by comparison scripts."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from joint_artifact_contract import read_system_label_from_fdf  # noqa: E402
from siesta_output_status import parse_siesta_output  # noqa: E402
from reference_provenance import validate_positive_reference_provenance  # noqa: E402


MATRIX_SUFFIXES = (".TSHS", ".HSX")
REFERENCE_SELECTION_POLICY = (
    "positive_siesta_provenance_v3: prefer one SystemLabel.TSHS, otherwise "
    "SystemLabel.HSX; require valid RUN.out plus matching reference/input/output/"
    "geometry hashes in a benchmark, split, derivative or local provenance manifest."
)


@dataclass(frozen=True)
class ReferenceSelection:
    path: Path | None
    reason: str
    ambiguous: bool
    candidate_count: int
    candidates: tuple[str, ...]
    provenance_status: str = "not_checked"
    provenance_manifest: str = ""
    provenance: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.path is not None and self.reason == "ok"

    @property
    def kind(self) -> str | None:
        return self.path.suffix if self.path is not None else None


def matrix_sort_key(path: Path) -> tuple[int, str]:
    numbers: list[int] = []
    for chunk in path.stem.replace("-", ".").replace("_", ".").split("."):
        if chunk.isdigit():
            numbers.append(int(chunk))
    return (numbers[-1] if numbers else 10**9, path.name)


def is_reference_candidate(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix in MATRIX_SUFFIXES
        and "ML_prediction" not in path.name
    )


def reference_candidates(sample_dir: Path) -> list[Path]:
    if not sample_dir.exists():
        return []
    return sorted(
        [
            path
            for suffix in MATRIX_SUFFIXES
            for path in sample_dir.glob(f"*{suffix}")
            if is_reference_candidate(path)
        ],
        key=matrix_sort_key,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_recorded_path(value: Any, manifest_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        cwd_path = Path.cwd() / path
        path = cwd_path if cwd_path.exists() else manifest_dir / path
    return path.resolve(strict=False)


def _hash_value(row: dict[str, Any], *keys: str) -> str:
    hashes = row.get("artifact_sha256") if isinstance(row.get("artifact_sha256"), dict) else {}
    for key in keys:
        value = hashes.get(key) or row.get(f"{key}_sha256")
        if isinstance(value, str) and value:
            return value
    return ""


def _provenance_from_row(
    row: dict[str, Any],
    *,
    candidate: Path,
    sample_dir: Path,
    manifest_dir: Path,
) -> dict[str, Any] | None:
    explicit = row.get("reference_provenance")
    if isinstance(explicit, dict):
        recorded_dir = _resolve_recorded_path(
            explicit.get("snapshot_dir") or row.get("reference_dir") or row.get("sample_dir"),
            manifest_dir,
        )
        recorded_reference = _resolve_recorded_path(
            explicit.get("reference_path") or row.get("reference_matrix"),
            manifest_dir,
        )
        if recorded_dir == sample_dir.resolve() or recorded_reference == candidate.resolve():
            return dict(explicit)

    artifacts = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    suffix_key = "tshs" if candidate.suffix == ".TSHS" else "hsx"
    reference_keys = (suffix_key, f"reference_{suffix_key}")
    reference_path = next(
        (
            _resolve_recorded_path(artifacts.get(key) or row.get(f"{key}_path"), manifest_dir)
            for key in reference_keys
            if artifacts.get(key) or row.get(f"{key}_path")
        ),
        None,
    )
    recorded_dir = _resolve_recorded_path(
        row.get("sample_dir") or row.get("reference_dir"),
        manifest_dir,
    )
    if recorded_dir != sample_dir.resolve() and reference_path != candidate.resolve():
        return None
    return {
        "status": "positive_siesta_provenance_valid",
        "system_label": row.get("system_label"),
        "reference_path": str(reference_path or ""),
        "reference_sha256": _hash_value(row, *reference_keys)
        or str(row.get("reference_matrix_sha256") or ""),
        "run_fdf_sha256": _hash_value(row, "run_fdf"),
        "run_out_sha256": _hash_value(row, "run_output")
        or str(row.get("run_out_sha256") or ""),
        "xv_sha256": _hash_value(row, "xv"),
        "struct_out_sha256": _hash_value(row, "struct_out"),
        "snapshot_dir": str(recorded_dir or ""),
    }


def _provenance_records(sample_dir: Path, candidate: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    local = sample_dir / "siesta_reference_provenance.json"
    if local.exists():
        records.append((local, _read_json(local)))
    for depth, parent in enumerate((sample_dir, *sample_dir.parents)):
        if depth > 8:
            break
        for name, key in (
            ("benchmark_dataset_manifest.json", "samples"),
            ("frozen_split_manifest.json", "rows"),
            ("derivative_siesta_reference_manifest.json", "rows"),
        ):
            manifest = parent / name
            if not manifest.exists():
                continue
            payload = _read_json(manifest)
            for row in payload.get(key) or []:
                if not isinstance(row, dict):
                    continue
                provenance = _provenance_from_row(
                    row,
                    candidate=candidate,
                    sample_dir=sample_dir,
                    manifest_dir=manifest.parent,
                )
                if provenance is not None:
                    records.append((manifest, provenance))
    return records


def _positive_provenance(
    sample_dir: Path,
    candidate: Path,
) -> tuple[bool, str, str, dict[str, Any]]:
    run_fdf = sample_dir / "RUN.fdf"
    system_label = read_system_label_from_fdf(run_fdf)
    if not system_label:
        return False, "missing_system_label_provenance", "", {}
    if candidate.name != f"{system_label}{candidate.suffix}":
        return False, "reference_filename_does_not_match_system_label", "", {}
    run_out = next((path for path in (sample_dir / "RUN.out", sample_dir / "siesta.out") if path.exists()), None)
    run_status = parse_siesta_output(run_out, run_fdf)
    if not run_status["valid"]:
        return False, f"invalid_siesta_execution:{run_status['parser_status']}", "", {}

    last_errors: list[str] = []
    for manifest, provenance in _provenance_records(sample_dir, candidate):
        errors = validate_positive_reference_provenance(
            provenance,
            sample_dir=sample_dir,
            reference_path=candidate,
        )
        if not errors:
            return True, "positive_siesta_provenance_valid", str(manifest), provenance
        last_errors = errors
    status = (
        f"invalid_positive_siesta_provenance:{last_errors[0]}"
        if last_errors
        else "missing_matching_positive_siesta_provenance"
    )
    return False, status, "", {"validation_errors": last_errors}


def choose_reference_matrix(
    sample_dir: Path,
    *,
    require_positive_provenance: bool = True,
) -> ReferenceSelection:
    sample_dir = Path(sample_dir)
    candidates = reference_candidates(sample_dir)
    candidate_names = tuple(path.name for path in candidates)
    if not candidates:
        return ReferenceSelection(None, "missing_reference_matrix", False, 0, candidate_names)

    tshs = [path for path in candidates if path.suffix == ".TSHS"]
    hsx = [path for path in candidates if path.suffix == ".HSX"]

    if len(tshs) == 1:
        selected = tshs[0]
    elif len(tshs) > 1:
        return ReferenceSelection(
            None,
            "ambiguous_reference_matrix_multiple_tshs",
            True,
            len(candidates),
            candidate_names,
        )
    elif len(hsx) == 1:
        selected = hsx[0]
    else:
        return ReferenceSelection(
            None,
            "ambiguous_reference_matrix_multiple_hsx",
            True,
            len(candidates),
            candidate_names,
        )
    if not require_positive_provenance:
        return ReferenceSelection(selected, "ok", False, len(candidates), candidate_names)
    ok, status, manifest, provenance = _positive_provenance(sample_dir, selected)
    return ReferenceSelection(
        selected if ok else None,
        "ok" if ok else status,
        False,
        len(candidates),
        candidate_names,
        provenance_status=status,
        provenance_manifest=manifest,
        provenance=provenance,
    )


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
