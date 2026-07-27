"""Write SIESTA ``.fdf`` inputs for a reference supercell and ±h displacements.

Reuses ``shared/fdf_materialization.materialize_sample_fdf`` so the emitted FDFs
keep the base file's species / pseudopotential / run blocks intact and only swap
the coordinates + lattice. This module never launches SIESTA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .structure import (
    direction_name,
    find_central_atom,
    make_displaced_structures,
    make_supercell,
    structure_from_fdf,
)

REFERENCE_LABEL = "reference"


def _displacement_labels(directions: tuple[str, ...]) -> list[str]:
    labels: list[str] = []
    for direction in directions:
        name = direction_name(direction)
        labels.append(f"{name}_plus")
        labels.append(f"{name}_minus")
    return labels


def generate_siesta_displacement_inputs(
    config: BenchmarkConfig,
    output_dir: str | Path,
    *,
    base_fdf: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate reference + ±h displacement FDFs and a ``metadata.json``.

    Parameters
    ----------
    config:
        Parsed benchmark config providing supercell, central atom, displacement
        and directions.
    output_dir:
        Destination directory. One subdirectory per structure is created
        (``reference``, ``x_plus``, ``x_minus``, ...), each with a ``RUN.fdf``.
    base_fdf:
        Base SIESTA FDF; defaults to ``config.system.input_structure``.
    dry_run:
        When True, nothing is written; the returned metadata lists the files
        that *would* be generated.

    Returns
    -------
    dict
        The metadata payload (also written to ``output_dir/metadata.json`` unless
        ``dry_run``).
    """
    from fdf_materialization import materialize_sample_fdf  # shared/

    source = base_fdf or config.system.input_structure
    if not source:
        raise ValueError(
            "No base FDF provided: set system.input_structure or pass base_fdf."
        )
    base = Path(source)
    if not base.is_file():
        raise FileNotFoundError(f"Base FDF not found: {base}")

    output_dir = Path(output_dir)
    primitive = structure_from_fdf(base)
    supercell = make_supercell(primitive, config.system.supercell)

    if config.system.central_atom == "auto":
        central_atom = find_central_atom(supercell)
    else:
        central_atom = int(config.system.central_atom)
        if central_atom < 0 or central_atom >= supercell.n_atoms:
            raise ValueError(
                f"central_atom {central_atom} out of range for supercell of "
                f"{supercell.n_atoms} atoms."
            )

    directions = config.derivatives.directions
    displacement = config.derivatives.displacement

    generated: dict[str, dict[str, Any]] = {}

    def _emit(label: str, structure) -> None:
        run_dir = output_dir / label
        fdf_path = run_dir / "RUN.fdf"
        record: dict[str, Any] = {
            "label": label,
            "fdf": str(fdf_path),
            "n_atoms": structure.n_atoms,
        }
        if not dry_run:
            materialized = materialize_sample_fdf(
                base,
                fdf_path,
                positions_ang=[tuple(p) for p in structure.positions.tolist()],
                atom_species=structure.species_index,
                lattice_vectors_ang=[tuple(v) for v in structure.cell.tolist()],
                system_label=f"{base.stem}_{label}",
                # Derivative reference/plus/minus fdfs must be single-point SCF:
                # inherited MD settings evolve the geometry away from +-delta.
                single_point=True,
            )
            record["fdf_sha256"] = materialized.metadata.get("materialized_fdf_sha256")
        generated[label] = record

    _emit(REFERENCE_LABEL, supercell)
    for direction in directions:
        plus, minus = make_displaced_structures(
            supercell, central_atom, direction, displacement
        )
        _emit(f"{direction_name(direction)}_plus", plus)
        _emit(f"{direction_name(direction)}_minus", minus)

    metadata: dict[str, Any] = {
        "schema": "ml_vs_siesta_displacement_inputs_v1",
        "base_fdf": str(base),
        "central_atom_index": central_atom,
        "central_atom_symbol": supercell.symbols[central_atom],
        "supercell_reps": list(config.system.supercell),
        "displacement": displacement,
        "directions": [direction_name(d) for d in directions],
        "reference_label": REFERENCE_LABEL,
        "displacement_labels": _displacement_labels(directions),
        "supercell_atom_count": supercell.n_atoms,
        "generated_files": generated,
        "dry_run": dry_run,
    }

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        metadata["metadata_json"] = str(output_dir / "metadata.json")

    return metadata
