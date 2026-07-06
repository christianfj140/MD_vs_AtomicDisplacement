from __future__ import annotations

from pathlib import Path

from Comparison.scripts.run_mixing_e2e_payload_once import expected_paths_present, resolve_repo_path


def test_resolve_repo_path_maps_relative_paths_inside_repo() -> None:
    resolved = resolve_repo_path("Comparison/config/ml_vs_siesta_mixing_e2e_20_50_80_payload.json")
    assert resolved.is_absolute()
    assert resolved.name == "ml_vs_siesta_mixing_e2e_20_50_80_payload.json"


def test_expected_paths_present_requires_all_paths(tmp_path: Path) -> None:
    existing = tmp_path / "exists"
    existing.mkdir()
    missing = tmp_path / "missing"
    assert expected_paths_present([existing]) is True
    assert expected_paths_present([existing, missing]) is False
