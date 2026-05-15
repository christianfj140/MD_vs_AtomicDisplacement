"""Small material preset resolver built on top of material_bundle validation."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on runtime environment.
    raise RuntimeError(
        "PyYAML is required to read material preset YAML files."
    ) from exc

from material_bundle import (
    MaterialBundleError,
    ValidatedMaterialBundle,
    validate_material_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESET_DIR = REPO_ROOT / "materials"
LEGACY_DEFAULT_PRESET = "h2o"
PRESET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ResolvedMaterialBundle:
    validated: ValidatedMaterialBundle
    preset: str | None
    source: str
    warning: str | None = None

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = self.validated.to_manifest_dict()
        payload.update(
            {
                "preset": self.preset,
                "material_source": self.source,
                "warning": self.warning,
            }
        )
        return payload


def _validate_preset_name(name: Any) -> str:
    text = str(name or "").strip()
    if not PRESET_NAME_PATTERN.fullmatch(text):
        raise MaterialBundleError(
            "material preset name must contain only letters, numbers, '.', '_' or '-' "
            "and must start with a letter or number."
        )
    return text


def material_preset_path(name: str, *, preset_dir: Path = DEFAULT_PRESET_DIR) -> Path:
    preset_name = _validate_preset_name(name)
    return preset_dir / preset_name / "material.yaml"


def load_material_preset(
    name: str,
    *,
    preset_dir: Path = DEFAULT_PRESET_DIR,
) -> dict[str, Any]:
    path = material_preset_path(name, preset_dir=preset_dir)
    if not path.is_file():
        raise MaterialBundleError(f"Material preset {name!r} does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("material"), dict):
        raise MaterialBundleError(f"Material preset must contain a material mapping: {path}")
    return data


def _material_section(config: dict[str, Any]) -> dict[str, Any] | None:
    material = config.get("material")
    if material is None:
        return None
    if not isinstance(material, dict):
        raise MaterialBundleError("material config must be a mapping.")
    return material


def _merge_preset_material(preset_config: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    material = copy.deepcopy(preset_config["material"])
    for key, value in override.items():
        if key == "preset" or value in (None, ""):
            continue
        material[key] = value
    return {"material": material}


def resolve_material_bundle(
    config: dict[str, Any],
    *,
    base_dir: str | Path = REPO_ROOT,
    preset_dir: str | Path = DEFAULT_PRESET_DIR,
    default_preset: str = LEGACY_DEFAULT_PRESET,
    allow_legacy_default: bool = True,
    allow_absolute_paths: bool = True,
) -> ResolvedMaterialBundle:
    """Resolve an explicit material section or legacy H2O default.

    This helper does not change pipeline behavior by itself. Callers can use it
    when they are ready to validate material inputs before running a pipeline.
    """

    preset_root = Path(preset_dir)
    material = _material_section(config)
    warning = None
    if material is None:
        if not allow_legacy_default:
            raise MaterialBundleError(
                "No material section configured. Add material.preset or a full material bundle."
            )
        preset = _validate_preset_name(default_preset)
        preset_config = load_material_preset(preset, preset_dir=preset_root)
        source = "legacy_default_preset"
        warning = (
            f"No material section configured; using backward-compatible material preset {preset!r}."
        )
        material_config = preset_config
    elif material.get("preset") not in (None, ""):
        preset = _validate_preset_name(material["preset"])
        preset_config = load_material_preset(preset, preset_dir=preset_root)
        source = "explicit_preset"
        material_config = _merge_preset_material(preset_config, material)
    else:
        preset = None
        source = "explicit_bundle"
        material_config = {"material": material}

    validation_base_dir = REPO_ROOT if source in {"legacy_default_preset", "explicit_preset"} else base_dir
    validated = validate_material_config(
        material_config,
        base_dir=validation_base_dir,
        allow_absolute_paths=allow_absolute_paths,
    )
    return ResolvedMaterialBundle(
        validated=validated,
        preset=preset,
        source=source,
        warning=warning,
    )
