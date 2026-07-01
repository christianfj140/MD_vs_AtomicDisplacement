"""Species-support diagnostics + a safe model-expansion stub (no training).

These helpers only *inspect* and *report*. They never initialize embeddings,
never touch model weights and never train. Actual expansion is delegated to a
model's own hook if it exists; otherwise a clear ``NotImplementedError`` is
raised with a diagnostic report attached.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SpeciesTransferConfigError(RuntimeError):
    """Raised when a species-transfer config is inconsistent."""


@dataclass
class SpeciesSupportReport:
    """Diagnostic report for adding one or more species to a model/config."""

    supported_species: list[str]
    orbital_basis: dict[str, list[str]]
    supported_species_pairs: list[list[str]]
    requested_new_species: list[str]
    missing_species: list[str]
    missing_species_pairs: list[list[str]]
    requires_new_embeddings: bool
    requires_new_heads: bool
    expandable: bool
    status: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported_species": self.supported_species,
            "orbital_basis": self.orbital_basis,
            "supported_species_pairs": self.supported_species_pairs,
            "requested_new_species": self.requested_new_species,
            "missing_species": self.missing_species,
            "missing_species_pairs": self.missing_species_pairs,
            "requires_new_embeddings": self.requires_new_embeddings,
            "requires_new_heads": self.requires_new_heads,
            "expandable": self.expandable,
            "status": self.status,
            "notes": self.notes,
        }


def _as_config_dict(model_or_config: Any) -> dict[str, Any]:
    if isinstance(model_or_config, dict):
        return model_or_config
    if isinstance(model_or_config, (str, Path)):
        path = Path(model_or_config)
        if not path.is_file():
            raise FileNotFoundError(f"Config not found: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(text) or {}
        return json.loads(text)
    # Duck-typed model object: read commonly-named attributes.
    config: dict[str, Any] = {}
    for attr in ("supported_species", "species", "orbital_basis", "expandable"):
        if hasattr(model_or_config, attr):
            config[attr] = getattr(model_or_config, attr)
    return config


def _extract_supported_species(config: dict[str, Any]) -> list[str]:
    for key in ("supported_species", "species", "base_species"):
        value = config.get(key)
        if value:
            if isinstance(value, dict):
                return [str(s) for s in value.keys()]
            return [str(s) for s in value]
    basis = config.get("orbital_basis")
    if isinstance(basis, dict):
        return [str(s) for s in basis.keys()]
    return []


def inspect_species_support(
    model_or_config: Any,
    new_species: list[str] | str | None = None,
) -> SpeciesSupportReport:
    """Report which species/pairs a model or config supports and what's missing.

    ``model_or_config`` may be a dict, a path to YAML/JSON, or a duck-typed model
    exposing ``supported_species`` / ``orbital_basis`` / ``expandable``.
    """
    config = _as_config_dict(model_or_config)
    supported = _extract_supported_species(config)
    orbital_basis_raw = config.get("orbital_basis") or {}
    orbital_basis = {
        str(k): [str(o) for o in (v or [])] for k, v in orbital_basis_raw.items()
    }
    pairs_raw = config.get("supported_species_pairs")
    if pairs_raw:
        supported_pairs = [sorted(str(x) for x in pair) for pair in pairs_raw]
    else:
        supported_pairs = [
            sorted([a, b])
            for i, a in enumerate(supported)
            for b in supported[i:]
        ]

    if new_species is None:
        requested = list(config.get("new_species") or [])
    elif isinstance(new_species, str):
        requested = [new_species]
    else:
        requested = list(new_species)
    requested = [str(s) for s in requested]

    missing = [s for s in requested if s not in supported]
    all_species = supported + [s for s in requested if s not in supported]
    supported_pair_set = {tuple(p) for p in supported_pairs}
    missing_pairs: list[list[str]] = []
    for i, a in enumerate(all_species):
        for b in all_species[i:]:
            pair = tuple(sorted([a, b]))
            if pair not in supported_pair_set and (a in missing or b in missing):
                missing_pairs.append(list(pair))

    requires_new_embeddings = bool(missing)
    requires_new_heads = bool(missing_pairs)
    expandable = bool(config.get("expandable", False))

    if not missing:
        status = "supported"
    elif expandable:
        status = "partially_supported"
    else:
        status = "not_implemented"

    notes: list[str] = []
    if missing:
        notes.append(
            f"Species {missing} are not in the current basis; "
            f"new embeddings would be required."
        )
    if missing_pairs:
        notes.append(
            f"Species pairs {missing_pairs} have no output block; "
            f"new heads would be required."
        )
    if missing and not expandable:
        notes.append(
            "Config does not advertise modular expansion "
            "(set 'expandable: true' only if the model truly supports it)."
        )

    return SpeciesSupportReport(
        supported_species=supported,
        orbital_basis=orbital_basis,
        supported_species_pairs=supported_pairs,
        requested_new_species=requested,
        missing_species=missing,
        missing_species_pairs=missing_pairs,
        requires_new_embeddings=requires_new_embeddings,
        requires_new_heads=requires_new_heads,
        expandable=expandable,
        status=status,
        notes=notes,
    )


def load_species_transfer_config(payload: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Validate a species-transfer config and detect which species are new.

    Enforces that any species referenced as ``new_species`` is not silently
    assumed to be part of the base; returns the normalized config plus a
    ``detected_new_species`` list (new ∖ base).
    """
    if isinstance(payload, (str, Path)):
        config = _as_config_dict(payload)
    else:
        config = dict(payload)

    block = config.get("species_transfer", config)
    if not isinstance(block, dict):
        raise SpeciesTransferConfigError("species_transfer must be a mapping.")

    base_species = [str(s) for s in (block.get("base_species") or [])]
    new_species = [str(s) for s in (block.get("new_species") or [])]
    if not base_species:
        raise SpeciesTransferConfigError("species_transfer.base_species is required.")

    detected_new = [s for s in new_species if s not in base_species]
    init = str(block.get("initialize_new_embeddings", "random"))
    if init not in {"random", "zeros", "copy"}:
        raise SpeciesTransferConfigError(
            f"initialize_new_embeddings {init!r} not in "
            "{'random', 'zeros', 'copy'}."
        )
    return {
        "base_species": base_species,
        "new_species": new_species,
        "detected_new_species": detected_new,
        "initialize_new_embeddings": init,
        "freeze_backbone_initially": bool(block.get("freeze_backbone_initially", True)),
        "replay_old_species_data": bool(block.get("replay_old_species_data", True)),
    }


def prepare_species_expansion(
    model: Any,
    old_config: dict[str, Any],
    new_config: dict[str, Any],
) -> dict[str, Any]:
    """Prepare (never execute) a species expansion on ``model``.

    If ``model`` exposes an ``expand_species`` (or ``prepare_species_expansion``)
    hook, it is invoked and its result returned. Otherwise a clear
    ``NotImplementedError`` is raised with the diagnostic report attached, so the
    unsupported case fails loudly rather than silently.
    """
    old_species = _extract_supported_species(old_config)
    new_species = _extract_supported_species(new_config) or [
        str(s) for s in (new_config.get("new_species") or [])
    ]
    added = [s for s in new_species if s not in old_species]
    report = inspect_species_support(old_config, new_species=added)

    for hook_name in ("expand_species", "prepare_species_expansion"):
        hook = getattr(model, hook_name, None)
        if callable(hook):
            return {
                "status": "delegated",
                "hook": hook_name,
                "result": hook(old_config, new_config),
                "report": report.to_dict(),
            }

    error = NotImplementedError(
        "Model does not expose a modular species-expansion hook "
        "(expand_species / prepare_species_expansion). "
        f"Adding {added or new_species} needs new embeddings/heads and is not "
        "implemented here. See the diagnostic report attached to this error."
    )
    error.report = report.to_dict()  # type: ignore[attr-defined]
    raise error
