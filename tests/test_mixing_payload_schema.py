"""Fase 15 (audit): payload schema v2, migration and fail-closed validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))

from ml_vs_siesta.mixing_payload_schema import (  # noqa: E402
    MIXING_PAYLOAD_SCHEMA_V2,
    migrate_mixing_payload,
    prevalidate_mixing_payload,
    validate_mixing_payload,
)

V2_OK = {
    "schema": MIXING_PAYLOAD_SCHEMA_V2,
    "action": "train",
    "modes": ["add", "replace"],
    "ratios": [0.0, 0.5, 1.0],
    "split_policy": "fixed_stratified_test",
    "training_weighting_policy": "per_structure",
    "models": ["graph2mat", "deeph"],
    "seeds": [0, 1, 2],
    "small": {"20": "datasets/small20"},
    "large": {"20": "datasets/large20"},
}


def test_valid_v2_passes():
    assert validate_mixing_payload(V2_OK) == []
    assert prevalidate_mixing_payload(dict(V2_OK))["schema"] == MIXING_PAYLOAD_SCHEMA_V2


def test_legacy_payload_is_migrated_with_warnings():
    legacy = {"action": "preview", "modes": ["add"], "ratios": [0.5], "seed": 3}
    migrated, warnings = migrate_mixing_payload(legacy)
    assert migrated["schema"] == MIXING_PAYLOAD_SCHEMA_V2
    assert migrated["migrated_from"] == "v1_schemaless"
    assert migrated["split_policy"] == "fixed_common_test"
    assert migrated["training_weighting_policy"] == "legacy_elementwise"
    assert migrated["selection_seeds"] == [3]
    assert warnings  # silent defaults are surfaced, never hidden
    assert validate_mixing_payload(migrated) == []


def test_unknown_schema_is_rejected():
    with pytest.raises(ValueError, match="Unknown mixing payload schema"):
        migrate_mixing_payload({"schema": "mixing_payload_schema_v99"})


@pytest.mark.parametrize(
    "patch,fragment",
    [
        ({"ratios": [1.5]}, "outside"),
        ({"modes": ["multiply"]}, "mode"),
        ({"split_policy": "random"}, "split_policy"),
        ({"training_weighting_policy": "bogus"}, "training_weighting_policy"),
        ({"models": ["schnet"]}, "unknown model"),
        ({"seeds": ["a"]}, "not an integer"),
        ({"small": None}, "missing 'small'"),
        (
            {"training_weighting_policy": "per_domain"},
            "domain_threshold_atoms",
        ),
    ],
)
def test_invalid_payload_fails_before_materializing(patch, fragment):
    payload = {**V2_OK, **patch}
    with pytest.raises(ValueError, match="nothing was materialized"):
        prevalidate_mixing_payload(payload)
    errors = validate_mixing_payload(payload)
    assert any(fragment in error for error in errors), errors


def test_per_domain_with_threshold_passes():
    payload = {
        **V2_OK,
        "training_weighting_policy": "per_domain",
        "domain_weighting": {"domain_threshold_atoms": 10},
    }
    assert validate_mixing_payload(payload) == []
