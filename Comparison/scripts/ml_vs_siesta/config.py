"""Benchmark configuration parsing + validation (no heavy logic).

The config is intentionally small: it parses a YAML/JSON/dict payload, validates
types and enum values, and fills defaults. Loading a config never touches SIESTA,
models or the filesystem beyond reading the config file itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_MODELS = ("graph2mat", "deeph")
SUPPORTED_TARGETS = ("hamiltonian", "density_matrix", "overlap")
SUPPORTED_DIRECTIONS = ("x", "y", "z")


class BenchmarkConfigError(RuntimeError):
    """Raised when a benchmark config is malformed."""


@dataclass
class SystemConfig:
    input_structure: str | None = None
    supercell: tuple[int, int, int] = (5, 5, 1)
    central_atom: Any = "auto"  # "auto" or an integer index


@dataclass
class DerivativesConfig:
    enabled: bool = False
    displacement: float = 0.01
    directions: tuple[str, ...] = SUPPORTED_DIRECTIONS


@dataclass
class SpeciesTransferConfig:
    enabled: bool = False
    base_species: tuple[str, ...] = ()
    new_species: tuple[str, ...] = ()
    initialize_new_embeddings: str = "random"
    freeze_backbone_initially: bool = True
    replay_old_species_data: bool = True


@dataclass
class BenchmarkConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    derivatives: DerivativesConfig = field(default_factory=DerivativesConfig)
    models: tuple[str, ...] = ("graph2mat", "deeph")
    targets: tuple[str, ...] = SUPPORTED_TARGETS
    dataset_mixing_enabled: bool = False
    species_transfer: SpeciesTransferConfig = field(
        default_factory=SpeciesTransferConfig
    )
    ui_enable_matrix_viewer: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": {
                "input_structure": self.system.input_structure,
                "supercell": list(self.system.supercell),
                "central_atom": self.system.central_atom,
            },
            "derivatives": {
                "enabled": self.derivatives.enabled,
                "displacement": self.derivatives.displacement,
                "directions": list(self.derivatives.directions),
            },
            "models": {"enabled": list(self.models)},
            "matrices": {"targets": list(self.targets)},
            "dataset_mixing": {"enabled": self.dataset_mixing_enabled},
            "species_transfer": {
                "enabled": self.species_transfer.enabled,
                "base_species": list(self.species_transfer.base_species),
                "new_species": list(self.species_transfer.new_species),
                "initialize_new_embeddings": (
                    self.species_transfer.initialize_new_embeddings
                ),
                "freeze_backbone_initially": (
                    self.species_transfer.freeze_backbone_initially
                ),
                "replay_old_species_data": (
                    self.species_transfer.replay_old_species_data
                ),
            },
            "ui": {"enable_matrix_viewer": self.ui_enable_matrix_viewer},
        }


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise BenchmarkConfigError(f"'{label}' must be a mapping, got {type(value).__name__}.")
    return value


def _parse_bool(value: Any, label: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "t", "1", "yes", "on"}:
            return True
        if token in {"false", "f", "0", "no", "off"}:
            return False
    raise BenchmarkConfigError(f"'{label}' must be a boolean, got {value!r}.")


def _parse_supercell(value: Any) -> tuple[int, int, int]:
    if value is None:
        return (5, 5, 1)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise BenchmarkConfigError("system.supercell must be a list of 3 integers.")
    try:
        reps = tuple(int(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigError("system.supercell entries must be integers.") from exc
    if any(r < 1 for r in reps):
        raise BenchmarkConfigError("system.supercell entries must be >= 1.")
    return reps  # type: ignore[return-value]


def _parse_central_atom(value: Any) -> Any:
    if value in (None, "auto"):
        return "auto"
    if isinstance(value, bool):
        raise BenchmarkConfigError("system.central_atom must be 'auto' or an integer.")
    if isinstance(value, int):
        if value < 0:
            raise BenchmarkConfigError("system.central_atom index must be >= 0.")
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        parsed = int(value)
        if parsed < 0:
            raise BenchmarkConfigError("system.central_atom index must be >= 0.")
        return parsed
    raise BenchmarkConfigError("system.central_atom must be 'auto' or an integer.")


def _parse_enum_list(
    value: Any,
    allowed: tuple[str, ...],
    label: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or not value:
        raise BenchmarkConfigError(f"'{label}' must be a non-empty list.")
    result: list[str] = []
    for item in value:
        token = str(item).strip().lower()
        if token not in allowed:
            raise BenchmarkConfigError(
                f"'{label}' contains unsupported value {item!r}; "
                f"allowed: {sorted(allowed)}."
            )
        if token not in result:
            result.append(token)
    return tuple(result)


def parse_benchmark_config(payload: dict[str, Any]) -> BenchmarkConfig:
    """Validate a raw config mapping and return a :class:`BenchmarkConfig`."""
    payload = _require_mapping(payload, "config")

    system_raw = _require_mapping(payload.get("system"), "system")
    system = SystemConfig(
        input_structure=(
            str(system_raw["input_structure"])
            if system_raw.get("input_structure") not in (None, "")
            else None
        ),
        supercell=_parse_supercell(system_raw.get("supercell")),
        central_atom=_parse_central_atom(system_raw.get("central_atom")),
    )

    derivatives_raw = _require_mapping(payload.get("derivatives"), "derivatives")
    displacement = derivatives_raw.get("displacement", 0.01)
    try:
        displacement = float(displacement)
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigError("derivatives.displacement must be a number.") from exc
    if displacement <= 0:
        raise BenchmarkConfigError("derivatives.displacement must be positive.")
    derivatives = DerivativesConfig(
        enabled=_parse_bool(derivatives_raw.get("enabled"), "derivatives.enabled", False),
        displacement=displacement,
        directions=_parse_enum_list(
            derivatives_raw.get("directions"),
            SUPPORTED_DIRECTIONS,
            "derivatives.directions",
            SUPPORTED_DIRECTIONS,
        ),
    )

    models_raw = _require_mapping(payload.get("models"), "models")
    models = _parse_enum_list(
        models_raw.get("enabled"),
        SUPPORTED_MODELS,
        "models.enabled",
        ("graph2mat", "deeph"),
    )

    matrices_raw = _require_mapping(payload.get("matrices"), "matrices")
    targets = _parse_enum_list(
        matrices_raw.get("targets"),
        SUPPORTED_TARGETS,
        "matrices.targets",
        SUPPORTED_TARGETS,
    )

    dataset_mixing_raw = _require_mapping(payload.get("dataset_mixing"), "dataset_mixing")
    dataset_mixing_enabled = _parse_bool(
        dataset_mixing_raw.get("enabled"), "dataset_mixing.enabled", False
    )

    species_raw = _require_mapping(payload.get("species_transfer"), "species_transfer")
    species_transfer = SpeciesTransferConfig(
        enabled=_parse_bool(species_raw.get("enabled"), "species_transfer.enabled", False),
        base_species=tuple(str(s) for s in (species_raw.get("base_species") or [])),
        new_species=tuple(str(s) for s in (species_raw.get("new_species") or [])),
        initialize_new_embeddings=str(
            species_raw.get("initialize_new_embeddings", "random")
        ),
        freeze_backbone_initially=_parse_bool(
            species_raw.get("freeze_backbone_initially"),
            "species_transfer.freeze_backbone_initially",
            True,
        ),
        replay_old_species_data=_parse_bool(
            species_raw.get("replay_old_species_data"),
            "species_transfer.replay_old_species_data",
            True,
        ),
    )

    ui_raw = _require_mapping(payload.get("ui"), "ui")
    ui_enable_matrix_viewer = _parse_bool(
        ui_raw.get("enable_matrix_viewer"), "ui.enable_matrix_viewer", True
    )

    return BenchmarkConfig(
        system=system,
        derivatives=derivatives,
        models=models,
        targets=targets,
        dataset_mixing_enabled=dataset_mixing_enabled,
        species_transfer=species_transfer,
        ui_enable_matrix_viewer=ui_enable_matrix_viewer,
        raw=payload,
    )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load a benchmark config from a YAML or JSON file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise BenchmarkConfigError(f"Config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    payload: Any
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - env dependent
            raise BenchmarkConfigError("PyYAML is required to read YAML configs.") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise BenchmarkConfigError(f"Config root must be a mapping: {config_path}")
    return parse_benchmark_config(payload)
