"""Shared configuration helpers for the MD pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on runtime environment.
    raise RuntimeError(
        "PyYAML is required to read pipeline_config.yaml. Install pyyaml in the "
        "environment used to run these scripts."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "pipeline_config.yaml"


def load_pipeline_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise RuntimeError(f"No existe el archivo de configuración: {path}")

    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise RuntimeError(f"La configuración debe ser un diccionario YAML: {path}")

    config["_config_path"] = path
    config["_config_dir"] = path.parent
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = (
        "paths",
        "commands",
        "md",
        "training",
        "checkpoint",
        "testing",
        "prediction",
        "pipeline",
    )
    for section in required_sections:
        if section not in config:
            raise RuntimeError(f"Falta la sección '{section}' en pipeline_config.yaml.")

    md = config["md"]
    if int(md["steps"]) <= 0:
        raise RuntimeError("md.steps debe ser mayor que cero.")
    if len(md["lattice_vectors"]) != 3:
        raise RuntimeError("md.lattice_vectors debe contener exactamente 3 vectores.")
    if not md["species"]:
        raise RuntimeError("md.species debe contener al menos una especie.")
    if not md["atoms"]:
        raise RuntimeError("md.atoms debe contener al menos un átomo.")


def config_dir(config: dict[str, Any]) -> Path:
    return Path(config["_config_dir"])


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return config_dir(config) / path


def paths(config: dict[str, Any]) -> dict[str, Path]:
    raw_paths = config["paths"]
    dataset_dir = resolve_path(config, raw_paths["dataset_dir"])
    training_dir = resolve_path(config, raw_paths["training_dir"])
    return {
        "dataset_dir": dataset_dir,
        "training_dir": training_dir,
        "run_fdf_path": dataset_dir / raw_paths["run_fdf_name"],
        "run_out_path": dataset_dir / raw_paths["run_out_name"],
        "training_config_path": training_dir / raw_paths["training_config_name"],
        "venv_activate": resolve_path(config, raw_paths["venv_activate"]),
    }


def command(config: dict[str, Any], name: str) -> str:
    return str(config["commands"][name])


def _fdf_bool(value: bool) -> str:
    return "t" if value else "f"


def _format_float(value: float) -> str:
    return f"{float(value):.8f}"


def render_run_fdf(config: dict[str, Any]) -> str:
    md = config["md"]
    lines = [
        f"# Run {md['steps']} steps of a {md['type_of_run']} MD",
        f"MD.TypeOfRun {md['type_of_run']}",
        f"MD.Steps {md['steps']}",
        "",
        "# Basis set.",
        f"PAO.BasisSize {md['basis_size']}",
        "",
        "# Matrix outputs.",
        f"TS.HS.Save {_fdf_bool(bool(md['save_hs']))}",
        f"TS.DE.Save {_fdf_bool(bool(md['save_de']))}",
        "",
        "# Lua store script.",
        f"Lua.Script {md['lua_script']}",
        "",
        "# Auxiliary cell handling.",
        f"ForceAuxCell {_fdf_bool(bool(md['force_aux_cell']))}",
        "",
        "# Structure.",
        f"LatticeConstant {md['lattice_constant']['value']} {md['lattice_constant']['unit']}",
        "%block LatticeVectors",
    ]

    for vector in md["lattice_vectors"]:
        lines.append(" ".join(_format_float(component) for component in vector))

    lines.extend(
        [
            "%endblock LatticeVectors",
            "",
            f"NumberOfSpecies {len(md['species'])}",
            "%block ChemicalSpeciesLabel",
        ]
    )

    for species in md["species"]:
        lines.append(
            f"{species['index']} {species['atomic_number']} {species['symbol']}"
        )

    lines.extend(
        [
            "%endblock ChemicalSpeciesLabel",
            "",
            f"NumberOfAtoms {len(md['atoms'])}",
            f"AtomicCoordinatesFormat {md['coordinates_format']}",
            "%block AtomicCoordinatesAndAtomicSpecies",
        ]
    )

    for index, atom in enumerate(md["atoms"], start=1):
        x, y, z = atom["position"]
        lines.append(
            f"{_format_float(x)}  {_format_float(y)} {_format_float(z)} "
            f"{atom['species_index']} # {index}: {atom['label']}"
        )

    lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    return "\n".join(lines) + "\n"


def render_training_config(config: dict[str, Any]) -> str:
    training_config = config["training"]
    return "# Generated from ../pipeline_config.yaml\n" + yaml.safe_dump(
        training_config,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )


def checkpoint_version(path: Path) -> int:
    for part in path.parts:
        if part.startswith("version_"):
            suffix = part.removeprefix("version_")
            if suffix.isdigit():
                return int(suffix)
    return -1


def resolve_checkpoint(config: dict[str, Any]) -> str:
    pipeline_paths = paths(config)
    training_dir = pipeline_paths["training_dir"]
    checkpoint_config = config["checkpoint"]
    configured_path = checkpoint_config.get("path")

    if configured_path:
        configured_path = str(configured_path)
        configured_abs = training_dir / configured_path
        if configured_abs.exists():
            return configured_path

    if bool(checkpoint_config.get("auto_best", True)):
        candidates = sorted(training_dir.glob(str(checkpoint_config["search_glob"])))
        if candidates:
            selection = str(checkpoint_config.get("selection", "latest_version"))
            if selection != "latest_version":
                raise RuntimeError(
                    "checkpoint.selection solo soporta actualmente "
                    "el valor 'latest_version'."
                )

            latest_version = max(checkpoint_version(path) for path in candidates)
            latest_candidates = [
                path for path in candidates if checkpoint_version(path) == latest_version
            ]
            if len(latest_candidates) != 1:
                rel_candidates = "\n".join(
                    f"  - {path.relative_to(training_dir).as_posix()}"
                    for path in latest_candidates
                )
                raise RuntimeError(
                    "Se encontró más de un checkpoint best-*.ckpt dentro de la "
                    f"última versión version_{latest_version}. Define "
                    "checkpoint.path en pipeline_config.yaml con uno de estos "
                    f"valores:\n{rel_candidates}"
                )

            selected = latest_candidates[0]
            rel_path = selected.relative_to(training_dir).as_posix()
            print(
                "[INFO] Se usará automáticamente el checkpoint de la versión "
                f"más nueva: {rel_path}"
            )
            return rel_path

    raise RuntimeError(
        "No se encontró ningún checkpoint best-*.ckpt válido. Ajusta "
        "checkpoint.path o checkpoint.search_glob en pipeline_config.yaml."
    )
