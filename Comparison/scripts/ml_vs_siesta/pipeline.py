"""End-to-end dry-run: validate the whole benchmark plan without heavy work.

``benchmark_dry_run`` performs only lightweight validation. It never launches
SIESTA, never trains, and never loads heavy models. Missing optional inputs
(e.g. an absent input structure) are reported as warnings, not crashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .fdf_io import REFERENCE_LABEL, _displacement_labels
from .structure import (
    direction_name,
    find_central_atom,
    make_supercell,
    structure_from_fdf,
)


def benchmark_dry_run(
    config: BenchmarkConfig,
    *,
    siesta_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a benchmark config end-to-end and return a JSON-able summary.

    The summary contains one entry per validation stage plus an overall ``ok``
    flag and a ``warnings`` list. This is a *plan*, not an execution.
    """
    checks: dict[str, Any] = {}
    warnings: list[str] = []

    # 1. Config (already parsed if we got a BenchmarkConfig).
    checks["config"] = {"ok": True, "detail": config.to_dict()}

    # 2/3/4. Structure → supercell → central atom → displacements.
    supercell = None
    central_atom = None
    structure_path = config.system.input_structure
    if not structure_path:
        checks["structure"] = {"ok": False, "detail": "system.input_structure not set."}
        warnings.append("system.input_structure not set; skipping structure checks.")
    elif not Path(structure_path).is_file():
        checks["structure"] = {
            "ok": False,
            "detail": f"input_structure not found: {structure_path}",
        }
        warnings.append(f"input_structure not found: {structure_path}")
    else:
        primitive = structure_from_fdf(structure_path)
        supercell = make_supercell(primitive, config.system.supercell)
        checks["structure"] = {
            "ok": True,
            "detail": {
                "primitive_atoms": primitive.n_atoms,
                "species": sorted(set(primitive.symbols)),
            },
        }
        checks["supercell"] = {
            "ok": True,
            "detail": {
                "reps": list(config.system.supercell),
                "supercell_atoms": supercell.n_atoms,
            },
        }
        if config.system.central_atom == "auto":
            central_atom = find_central_atom(supercell)
        else:
            central_atom = int(config.system.central_atom)
        in_range = 0 <= central_atom < supercell.n_atoms
        checks["central_atom"] = {
            "ok": in_range,
            "detail": {
                "index": central_atom,
                "symbol": supercell.symbols[central_atom] if in_range else None,
                "mode": config.system.central_atom,
            },
        }
        if not in_range:
            warnings.append(f"central_atom {central_atom} out of range.")

    # 5. Displacements.
    if config.derivatives.enabled:
        checks["displacements"] = {
            "ok": True,
            "detail": {
                "displacement": config.derivatives.displacement,
                "directions": [direction_name(d) for d in config.derivatives.directions],
                "labels": _displacement_labels(config.derivatives.directions),
            },
        }
    else:
        checks["displacements"] = {"ok": True, "detail": "derivatives disabled."}

    # 6. Expected SIESTA paths.
    labels = [REFERENCE_LABEL]
    if config.derivatives.enabled:
        labels += _displacement_labels(config.derivatives.directions)
    if siesta_output_dir is not None:
        base = Path(siesta_output_dir)
        expected = {label: str(base / label / "RUN.fdf") for label in labels}
    else:
        expected = {label: f"<output_dir>/{label}/RUN.fdf" for label in labels}
    checks["siesta_paths"] = {"ok": True, "detail": expected}

    # 7. Predictors.
    checks["predictors"] = {
        "ok": bool(config.models),
        "detail": {"configured": list(config.models)},
    }

    # 8. Matrix targets.
    checks["targets"] = {"ok": bool(config.targets), "detail": list(config.targets)}

    # 9. UI options.
    checks["ui"] = {"ok": True, "detail": {"enable_matrix_viewer": config.ui_enable_matrix_viewer}}

    # 10. Dataset / species options.
    checks["dataset_mixing"] = {
        "ok": True,
        "detail": {"enabled": config.dataset_mixing_enabled},
    }
    species = config.species_transfer
    detected_new = [s for s in species.new_species if s not in species.base_species]
    checks["species_transfer"] = {
        "ok": True,
        "detail": {
            "enabled": species.enabled,
            "base_species": list(species.base_species),
            "new_species": list(species.new_species),
            "detected_new_species": detected_new,
        },
    }

    overall_ok = all(entry.get("ok", False) for entry in checks.values())
    return {
        "schema": "ml_vs_siesta_benchmark_dry_run_v1",
        "ok": overall_ok,
        "warnings": warnings,
        "checks": checks,
    }
