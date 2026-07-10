"""Deterministic input signatures for cached derivative artifacts (audit Fase 5).

A cached ``.npz`` may only be reused when its sidecar metadata carries an
``input_signature_sha256`` that matches the signature recomputed from the
CURRENT inputs (checkpoint, code, structure, direction, dtype, method).
Anything else — no metadata, no signature (legacy), mismatch, unreadable or
non-finite payload — must be recomputed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

INPUT_SIGNATURE_SCHEMA = "derivative_input_signature_v1"

CACHE_VALID = "valid"
CACHE_LEGACY_UNVERIFIED = "legacy_unverified"
CACHE_SIGNATURE_MISMATCH = "signature_mismatch"
CACHE_MISSING_METADATA = "missing_metadata"
CACHE_UNREADABLE = "unreadable"
CACHE_NON_FINITE = "non_finite"


def file_sha256(path: str | Path | None) -> str | None:
    if path in (None, ""):
        return None
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_signature_sha256(payload: dict[str, Any]) -> str:
    """Canonical sha256 over the signature payload (order-independent)."""
    encoded = json.dumps(
        {"schema": INPUT_SIGNATURE_SCHEMA, **payload},
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cached_result_status(
    npz_path: str | Path,
    metadata_path: str | Path,
    expected_signature: str,
) -> str:
    """Classify an existing cached derivative for reuse (never raises)."""
    npz_path = Path(npz_path)
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        return CACHE_MISSING_METADATA
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return CACHE_UNREADABLE
    stored = metadata.get("input_signature_sha256")
    if not stored:
        return CACHE_LEGACY_UNVERIFIED
    if str(stored) != str(expected_signature):
        return CACHE_SIGNATURE_MISMATCH
    try:
        import numpy as np
        from scipy import sparse

        matrix = sparse.load_npz(npz_path)
        if matrix.data.size and not bool(np.all(np.isfinite(matrix.data))):
            return CACHE_NON_FINITE
        expected_shape = metadata.get("matrix_shape")
        if expected_shape and [int(x) for x in expected_shape] != [int(x) for x in matrix.shape]:
            return CACHE_SIGNATURE_MISMATCH
    except Exception:  # noqa: BLE001 - unreadable/corrupt payloads must not be reused
        return CACHE_UNREADABLE
    return CACHE_VALID
