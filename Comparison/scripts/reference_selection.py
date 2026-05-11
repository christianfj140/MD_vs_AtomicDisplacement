#!/usr/bin/env python3
"""Strict SIESTA reference matrix selection shared by comparison scripts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


MATRIX_SUFFIXES = (".TSHS", ".HSX")
REFERENCE_SELECTION_POLICY = (
    "strict_single_reference_v1: prefer exactly one non-predicted .TSHS; "
    "if no .TSHS exists, allow exactly one non-predicted .HSX; reject ambiguity."
)


@dataclass(frozen=True)
class ReferenceSelection:
    path: Path | None
    reason: str
    ambiguous: bool
    candidate_count: int
    candidates: tuple[str, ...]

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


def choose_reference_matrix(sample_dir: Path) -> ReferenceSelection:
    candidates = reference_candidates(sample_dir)
    candidate_names = tuple(path.name for path in candidates)
    if not candidates:
        return ReferenceSelection(None, "missing_reference_matrix", False, 0, candidate_names)

    tshs = [path for path in candidates if path.suffix == ".TSHS"]
    hsx = [path for path in candidates if path.suffix == ".HSX"]

    if len(tshs) == 1:
        return ReferenceSelection(tshs[0], "ok", False, len(candidates), candidate_names)
    if len(tshs) > 1:
        return ReferenceSelection(
            None,
            "ambiguous_reference_matrix_multiple_tshs",
            True,
            len(candidates),
            candidate_names,
        )
    if len(hsx) == 1:
        return ReferenceSelection(hsx[0], "ok", False, len(candidates), candidate_names)
    return ReferenceSelection(
        None,
        "ambiguous_reference_matrix_multiple_hsx",
        True,
        len(candidates),
        candidate_names,
    )


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
