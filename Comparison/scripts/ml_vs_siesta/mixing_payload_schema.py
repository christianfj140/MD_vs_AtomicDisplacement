"""Versioned mixing payload schema + fail-closed prevalidation (audit Fase 15).

``mixing_payload_schema_v2`` makes every scientific choice explicit — split
policy, evaluation scope, loss weighting policy, seeds — so no silent default
decides an experiment. Legacy (v1, schema-less) payloads are migrated with
their historical semantics and flagged as migrated.
"""

from __future__ import annotations

from typing import Any

MIXING_PAYLOAD_SCHEMA_V2 = "mixing_payload_schema_v2"

_VALID_ACTIONS = {"preview", "materialize", "train"}
_VALID_MODES = {"add", "replace"}
_VALID_SPLIT_POLICIES = {
    "fixed_stratified_test",
    "fixed_common_test",
    "fixed_common_test_small_only",
    "resplit_combined",
}
_VALID_WEIGHTING = {"legacy_elementwise", "per_structure", "per_domain"}


def migrate_mixing_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return a v2 payload and migration warnings. Never mutates the input."""
    warnings: list[str] = []
    migrated = dict(payload)
    if migrated.get("schema") == MIXING_PAYLOAD_SCHEMA_V2:
        return migrated, warnings
    if migrated.get("schema"):
        raise ValueError(f"Unknown mixing payload schema {migrated['schema']!r}")
    migrated["schema"] = MIXING_PAYLOAD_SCHEMA_V2
    migrated["migrated_from"] = "v1_schemaless"
    if "split_policy" not in migrated:
        migrated["split_policy"] = "fixed_common_test"
        warnings.append("split_policy defaulted to fixed_common_test (small-only test)")
    if "training_weighting_policy" not in migrated:
        migrated["training_weighting_policy"] = "legacy_elementwise"
        warnings.append(
            "training_weighting_policy defaulted to legacy_elementwise "
            "(large structures dominate the loss)"
        )
    if "seeds" not in migrated and "selection_seeds" not in migrated:
        migrated["selection_seeds"] = [int(migrated.get("seed") or 0)]
        warnings.append("single selection seed; <3 seeds means exploratory results")
    return migrated, warnings


def validate_mixing_payload(payload: dict[str, Any]) -> list[str]:
    """Validate a v2 payload BEFORE any materialization. Returns error strings."""
    errors: list[str] = []
    if payload.get("schema") != MIXING_PAYLOAD_SCHEMA_V2:
        errors.append(f"schema must be {MIXING_PAYLOAD_SCHEMA_V2}, got {payload.get('schema')!r}")

    action = str(payload.get("action") or "preview")
    if action not in _VALID_ACTIONS:
        errors.append(f"action {action!r} not in {sorted(_VALID_ACTIONS)}")

    for mode in payload.get("modes") or []:
        if str(mode) not in _VALID_MODES:
            errors.append(f"mode {mode!r} not in {sorted(_VALID_MODES)}")

    for ratio in payload.get("ratios") or []:
        try:
            value = float(ratio)
        except (TypeError, ValueError):
            errors.append(f"ratio {ratio!r} is not a number")
            continue
        if not 0.0 <= value <= 1.0:
            errors.append(f"ratio {value} outside [0, 1]")

    split_policy = str(payload.get("split_policy") or "")
    if split_policy not in _VALID_SPLIT_POLICIES:
        errors.append(f"split_policy {split_policy!r} not in {sorted(_VALID_SPLIT_POLICIES)}")

    weighting = str(payload.get("training_weighting_policy") or "")
    if weighting not in _VALID_WEIGHTING:
        errors.append(
            f"training_weighting_policy {weighting!r} not in {sorted(_VALID_WEIGHTING)}"
        )
    if weighting == "per_domain":
        domain = payload.get("domain_weighting") or {}
        if not domain.get("domain_threshold_atoms"):
            errors.append("per_domain requires domain_weighting.domain_threshold_atoms")

    for key in ("models",):
        for model in payload.get(key) or []:
            if str(model) not in {"graph2mat", "deeph"}:
                errors.append(f"unknown model {model!r}")

    seeds = payload.get("seeds") or payload.get("selection_seeds") or []
    for seed in seeds:
        if not isinstance(seed, int):
            errors.append(f"seed {seed!r} is not an integer")

    for side in ("small", "large"):
        mapping = payload.get(side)
        if action != "preview" and not mapping:
            errors.append(f"missing {side!r} dataset mapping")
        if mapping and not isinstance(mapping, dict):
            errors.append(f"{side!r} must map size -> dataset_root")

    return errors


def prevalidate_mixing_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Migrate + validate; raise with every error listed on failure."""
    migrated, warnings = migrate_mixing_payload(payload)
    errors = validate_mixing_payload(migrated)
    if errors:
        raise ValueError(
            "Invalid mixing payload (nothing was materialized): " + "; ".join(errors)
        )
    if warnings:
        migrated.setdefault("migration_warnings", []).extend(warnings)
    return migrated
