"""Fase 5 (audit): derivative cache signatures — stale results must not be reused."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "shared"))

from artifact_signature import (  # noqa: E402
    CACHE_LEGACY_UNVERIFIED,
    CACHE_MISSING_METADATA,
    CACHE_NON_FINITE,
    CACHE_SIGNATURE_MISMATCH,
    CACHE_UNREADABLE,
    CACHE_VALID,
    cached_result_status,
    input_signature_sha256,
)

BASE_PAYLOAD = {
    "model": "graph2mat",
    "checkpoint_sha256": "abc",
    "repository_commits": {"MD": "sha1"},
    "structure_fdf_sha256": "fdf",
    "dtype": "float64",
    "atom_index": 0,
    "axis_index": 1,
}


def _write_cache(tmp_path: Path, signature: str | None, data=None, shape=None):
    matrix = sparse.csr_matrix(
        np.asarray(data if data is not None else [[1.0, 0.0], [0.0, 2.0]])
    )
    npz = tmp_path / "d.npz"
    with npz.open("wb") as handle:
        sparse.save_npz(handle, matrix)
    metadata = {"matrix_shape": shape or list(matrix.shape)}
    if signature is not None:
        metadata["input_signature_sha256"] = signature
    meta = tmp_path / "d.json"
    meta.write_text(json.dumps(metadata), encoding="utf-8")
    return npz, meta


def test_signature_is_deterministic_and_sensitive():
    sig = input_signature_sha256(BASE_PAYLOAD)
    assert sig == input_signature_sha256(dict(BASE_PAYLOAD))
    for key, value in [
        ("checkpoint_sha256", "OTHER"),        # different checkpoint
        ("structure_fdf_sha256", "OTHER"),      # different coordinates
        ("dtype", "float32"),                   # different dtype
        ("repository_commits", {"MD": "sha2"}),  # different code
        ("atom_index", 1),
    ]:
        assert input_signature_sha256({**BASE_PAYLOAD, key: value}) != sig, key


def test_matching_signature_is_valid(tmp_path):
    sig = input_signature_sha256(BASE_PAYLOAD)
    npz, meta = _write_cache(tmp_path, sig)
    assert cached_result_status(npz, meta, sig) == CACHE_VALID


def test_mismatched_signature_rejected(tmp_path):
    npz, meta = _write_cache(tmp_path, "stale-signature")
    sig = input_signature_sha256(BASE_PAYLOAD)
    assert cached_result_status(npz, meta, sig) == CACHE_SIGNATURE_MISMATCH


def test_legacy_without_signature_rejected(tmp_path):
    npz, meta = _write_cache(tmp_path, None)
    sig = input_signature_sha256(BASE_PAYLOAD)
    assert cached_result_status(npz, meta, sig) == CACHE_LEGACY_UNVERIFIED


def test_missing_metadata_rejected(tmp_path):
    npz, meta = _write_cache(tmp_path, "x")
    meta.unlink()
    assert cached_result_status(npz, meta, "x") == CACHE_MISSING_METADATA


def test_non_finite_payload_rejected(tmp_path):
    sig = input_signature_sha256(BASE_PAYLOAD)
    npz, meta = _write_cache(tmp_path, sig, data=[[np.nan, 0.0], [0.0, 1.0]])
    assert cached_result_status(npz, meta, sig) == CACHE_NON_FINITE


def test_shape_mismatch_rejected(tmp_path):
    sig = input_signature_sha256(BASE_PAYLOAD)
    npz, meta = _write_cache(tmp_path, sig, shape=[4, 4])
    assert cached_result_status(npz, meta, sig) == CACHE_SIGNATURE_MISMATCH


def test_corrupt_npz_rejected(tmp_path):
    sig = input_signature_sha256(BASE_PAYLOAD)
    npz, meta = _write_cache(tmp_path, sig)
    npz.write_bytes(b"not-an-npz")
    assert cached_result_status(npz, meta, sig) == CACHE_UNREADABLE
