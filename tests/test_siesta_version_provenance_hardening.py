"""C3 audit fix: SIESTA version probe/gate must reject non-version noise."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "shared", REPO_ROOT / "MD" / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from benchmark_manifest import (  # noqa: E402
    extract_siesta_version_from_text,
    looks_like_siesta_version,
    provenance_status,
)
from regrab_siesta_provenance import regrab_datasets, regrab_provenance_payload  # noqa: E402

X11_NOISE = "Authorization required, but no authorization protocol specified"
BUILD_INFO = (
    f"{X11_NOISE}\n\n{X11_NOISE}\n\n"
    "Executable      : siesta\n"
    "Version         : 5.4.2-11-g4e9a46060\n"
    "Architecture    : x86_64\n"
    "Compiler version: GNU-13.3.0\n"
)


def _completed(stdout: str, returncode: int = 0) -> types.SimpleNamespace:
    return types.SimpleNamespace(stdout=stdout, returncode=returncode)


def _probe_with_output(stdout: str) -> dict:
    import generate_md_dataset as gen

    with mock.patch.object(gen.subprocess, "run", return_value=_completed(stdout)):
        return gen.probe_siesta_version("siesta")


def test_version_text_validation() -> None:
    assert not looks_like_siesta_version(X11_NOISE)
    assert not looks_like_siesta_version("")
    assert not looks_like_siesta_version(None)
    assert looks_like_siesta_version("5.4.2-11-g4e9a46060")
    assert looks_like_siesta_version("SIESTA 4.1.5")


def test_extract_version_prefers_build_info_line() -> None:
    assert extract_siesta_version_from_text(BUILD_INFO) == "5.4.2-11-g4e9a46060"
    assert extract_siesta_version_from_text("SIESTA 4.1.5\n") == "SIESTA 4.1.5"
    assert extract_siesta_version_from_text(X11_NOISE) is None
    assert extract_siesta_version_from_text("") is None
    assert extract_siesta_version_from_text(None) is None


def test_probe_rejects_noise_output() -> None:
    result = _probe_with_output(X11_NOISE)
    assert result["siesta_version"] == ""
    assert result["siesta_version_probe"]["status"] == "unverified"


def test_probe_detects_real_version_inside_noisy_output() -> None:
    result = _probe_with_output(BUILD_INFO)
    assert result["siesta_version"] == "5.4.2-11-g4e9a46060"
    assert result["siesta_version_probe"]["status"] == "detected"


def test_strict_provenance_gate_rejects_noise_version(tmp_path: Path) -> None:
    material = {
        "label": "graphene",
        "basis_file_sha256": {"C.ion.xml": "abc"},
        "pseudopotential_sha256": {"C": "def"},
        "fdf_sha256": "0123",
        "siesta_command_line": "siesta < RUN.fdf",
        "environment": {"python_version": "3.12", "platform": "linux"},
        "siesta_version": X11_NOISE,
    }
    status = provenance_status(tmp_path, material, strict_paper_ready=True)
    assert status["siesta_version_provenance"] is False
    assert "siesta_version_provenance" in status["missing"]

    material["siesta_version"] = "5.4.2-11-g4e9a46060"
    status = provenance_status(tmp_path, material, strict_paper_ready=True)
    assert status["siesta_version_provenance"] is True


def test_regrab_fixes_noise_version_from_build_info(tmp_path: Path) -> None:
    root = tmp_path / "datasets" / "sample"
    root.mkdir(parents=True)
    payload = {
        "siesta_version": X11_NOISE,
        "siesta_build_info": BUILD_INFO,
        "siesta_version_probe": {"status": "detected", "attempts": []},
    }
    path = root / "material_provenance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    # Dry-run reports the change without writing.
    changes = regrab_datasets(tmp_path / "datasets", apply=False)
    assert [c["status"] for c in changes] == ["would_apply"]
    assert json.loads(path.read_text())["siesta_version"] == X11_NOISE

    # Apply fixes in place; a second run is a no-op (idempotent).
    changes = regrab_datasets(tmp_path / "datasets", apply=True)
    assert changes[0]["new_version"] == "5.4.2-11-g4e9a46060"
    fixed = json.loads(path.read_text())
    assert fixed["siesta_version"] == "5.4.2-11-g4e9a46060"
    assert fixed["siesta_version_probe"]["status"] == "regrabbed_from_build_info"
    assert regrab_datasets(tmp_path / "datasets", apply=True) == []


def test_regrab_leaves_valid_and_unrecoverable_payloads_alone() -> None:
    valid = {"siesta_version": "5.4.2", "siesta_build_info": BUILD_INFO}
    assert regrab_provenance_payload(valid) is None
    assert valid["siesta_version"] == "5.4.2"

    unrecoverable = {"siesta_version": X11_NOISE, "siesta_build_info": X11_NOISE}
    assert regrab_provenance_payload(unrecoverable) is None
    assert unrecoverable["siesta_version"] == X11_NOISE
