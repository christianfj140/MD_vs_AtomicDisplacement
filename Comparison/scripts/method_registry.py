#!/usr/bin/env python3
"""Canonical scientific method identifiers for the comparison workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScientificMethod:
    method_id: str
    display_name: str
    legacy_aliases: tuple[str, ...]
    results_dir: str
    frozen_test_set: str
    is_baseline: bool = False


METHOD_REGISTRY: dict[str, ScientificMethod] = {
    "md": ScientificMethod(
        method_id="md",
        display_name="MD",
        legacy_aliases=(),
        results_dir="results_md",
        frozen_test_set="test_md",
        is_baseline=True,
    ),
    "siesta_fc_cartesian": ScientificMethod(
        method_id="siesta_fc_cartesian",
        display_name="SIESTA FC Cartesian",
        legacy_aliases=("atom_displacement", "atomdisp"),
        results_dir="results_atomdisp",
        frozen_test_set="test_siesta_fc_cartesian",
    ),
    "random_cartesian": ScientificMethod(
        method_id="random_cartesian",
        display_name="Random Cartesian",
        legacy_aliases=(),
        results_dir="results_random_cartesian",
        frozen_test_set="test_random_cartesian",
    ),
}

CANONICAL_METHOD_IDS = tuple(METHOD_REGISTRY)
BASELINE_METHOD_ID = "md"

METHOD_ID_ALIASES: dict[str, str] = {
    alias: spec.method_id
    for spec in METHOD_REGISTRY.values()
    for alias in (spec.method_id, *spec.legacy_aliases)
}

TEST_SET_ALIASES = {
    "test_atomdisp": "test_siesta_fc_cartesian",
}


def normalize_method_id(value: Any, *, allow_unknown: bool = False) -> str:
    """Return a canonical scientific method ID."""
    text = str(value or "").strip()
    canonical = METHOD_ID_ALIASES.get(text)
    if canonical is not None:
        return canonical
    if allow_unknown:
        return text
    raise ValueError(f"Unknown scientific method ID: {text!r}")


def get_method(value: Any) -> ScientificMethod:
    """Return method metadata for a canonical ID or legacy alias."""
    return METHOD_REGISTRY[normalize_method_id(value)]


def normalize_test_set_id(value: Any) -> str:
    """Return a canonical frozen test-set ID when it encodes a method alias."""
    text = str(value or "").strip()
    if text in TEST_SET_ALIASES:
        return TEST_SET_ALIASES[text]
    if text.startswith("test_"):
        suffix = text.removeprefix("test_")
        if suffix == "mixed":
            return "test_mixed"
        try:
            return f"test_{normalize_method_id(suffix)}"
        except ValueError:
            return text
    return text


def normalize_method_mapping(value: Any) -> Any:
    """Normalize method-keyed dictionaries without changing non-dict values."""
    if not isinstance(value, dict):
        return value
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        normalized[normalize_method_id(key, allow_unknown=True)] = item
    return normalized
