"""Lightweight atomic structure + supercell / displacement utilities.

We keep a tiny, numpy-based :class:`BenchmarkStructure` instead of hard-wiring
ASE/sisl so that the whole benchmark toolkit stays fast and trivially testable
on CPU. Reading real SIESTA ``.fdf`` files reuses ``shared/fdf_materialization``.

Directions accept ``"x"``/``"y"``/``"z"`` or ``0``/``1``/``2`` throughout.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_AXIS_BY_NAME = {"x": 0, "y": 1, "z": 2}


class StructureError(RuntimeError):
    """Raised when a structure operation gets invalid input."""


def normalize_direction(direction: Any) -> int:
    """Return the cartesian axis index (0, 1 or 2) for a direction token."""
    if isinstance(direction, bool):  # bool is an int subclass; reject explicitly.
        raise StructureError(f"Invalid direction: {direction!r}")
    if isinstance(direction, (int, np.integer)):
        axis = int(direction)
    else:
        axis = _AXIS_BY_NAME.get(str(direction).strip().lower(), -1)
    if axis not in (0, 1, 2):
        raise StructureError(
            f"Invalid direction {direction!r}; use 'x'/'y'/'z' or 0/1/2."
        )
    return axis


def direction_name(direction: Any) -> str:
    """Return the canonical ``x``/``y``/``z`` name for a direction token."""
    return "xyz"[normalize_direction(direction)]


@dataclass
class BenchmarkStructure:
    """Minimal atomic structure.

    Attributes
    ----------
    symbols:
        Per-atom chemical symbol (e.g. ``["C", "C"]``).
    positions:
        ``(N, 3)`` cartesian coordinates in Angstrom.
    cell:
        ``(3, 3)`` lattice vectors (rows) in Angstrom.
    pbc:
        Periodic-boundary flags per lattice vector.
    species_index:
        Optional per-atom 1-based SIESTA species index (from the source FDF).
    source_atom:
        Optional provenance: for a supercell replica, the index of the atom in
        the *original* structure it came from. ``None`` for primitive cells.
    cell_offset:
        Optional provenance: the integer lattice translation ``(i, j, k)`` used
        to build each replica. ``None`` for primitive cells.
    metadata:
        Free-form provenance dictionary.
    """

    symbols: list[str]
    positions: np.ndarray
    cell: np.ndarray
    pbc: tuple[bool, bool, bool] = (True, True, True)
    species_index: list[int] | None = None
    source_atom: list[int] | None = None
    cell_offset: list[tuple[int, int, int]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        self.cell = np.asarray(self.cell, dtype=float).reshape(3, 3)
        self.symbols = [str(sym) for sym in self.symbols]
        if len(self.symbols) != len(self.positions):
            raise StructureError(
                f"symbols ({len(self.symbols)}) and positions "
                f"({len(self.positions)}) length mismatch."
            )
        if self.species_index is not None:
            self.species_index = [int(s) for s in self.species_index]
            if len(self.species_index) != len(self.symbols):
                raise StructureError("species_index length mismatch with atoms.")
        self.pbc = tuple(bool(flag) for flag in self.pbc)  # type: ignore[assignment]

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    def copy(self) -> "BenchmarkStructure":
        return BenchmarkStructure(
            symbols=list(self.symbols),
            positions=self.positions.copy(),
            cell=self.cell.copy(),
            pbc=self.pbc,
            species_index=list(self.species_index) if self.species_index else None,
            source_atom=list(self.source_atom) if self.source_atom else None,
            cell_offset=list(self.cell_offset) if self.cell_offset else None,
            metadata=copy.deepcopy(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "positions": self.positions.tolist(),
            "cell": self.cell.tolist(),
            "pbc": list(self.pbc),
            "species_index": list(self.species_index) if self.species_index else None,
            "source_atom": list(self.source_atom) if self.source_atom else None,
            "n_atoms": self.n_atoms,
        }


def structure_from_fdf(fdf_path: str | Path) -> BenchmarkStructure:
    """Build a :class:`BenchmarkStructure` from a SIESTA ``.fdf`` file.

    Reuses ``shared/fdf_materialization`` (Ang coordinates only).
    """
    from fdf_materialization import extract_fdf_structure  # shared/

    fdf = extract_fdf_structure(Path(fdf_path))
    species_label = {sp.index: sp.label for sp in fdf.species}
    symbols = [species_label[atom.species_index] for atom in fdf.atoms]
    positions = np.array([atom.position_ang for atom in fdf.atoms], dtype=float)
    cell = np.array(fdf.lattice_vectors_ang, dtype=float)
    species_index = [atom.species_index for atom in fdf.atoms]
    return BenchmarkStructure(
        symbols=symbols,
        positions=positions,
        cell=cell,
        species_index=species_index,
        metadata={"source_fdf": str(Path(fdf_path))},
    )


def make_supercell(
    structure: BenchmarkStructure,
    reps: Iterable[int] = (5, 5, 1),
) -> BenchmarkStructure:
    """Replicate ``structure`` by integer ``reps`` along each lattice vector.

    Species, positions, cell and periodicity are preserved. Every replica atom
    records its ``source_atom`` (index in the original structure) and its
    ``cell_offset`` so the original → replica mapping stays fully traceable.
    """
    reps_t = tuple(int(r) for r in reps)
    if len(reps_t) != 3 or any(r < 1 for r in reps_t):
        raise StructureError(f"reps must be three positive integers, got {reps!r}")

    nx, ny, nz = reps_t
    new_positions: list[np.ndarray] = []
    new_symbols: list[str] = []
    new_species: list[int] = [] if structure.species_index is not None else None  # type: ignore[assignment]
    source_atom: list[int] = []
    cell_offset: list[tuple[int, int, int]] = []
    a, b, c = structure.cell

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = i * a + j * b + k * c
                for atom_index in range(structure.n_atoms):
                    new_positions.append(structure.positions[atom_index] + shift)
                    new_symbols.append(structure.symbols[atom_index])
                    if new_species is not None:
                        new_species.append(structure.species_index[atom_index])
                    source_atom.append(atom_index)
                    cell_offset.append((i, j, k))

    new_cell = structure.cell * np.array(reps_t, dtype=float)[:, None]
    metadata = dict(structure.metadata)
    metadata["supercell_reps"] = list(reps_t)
    return BenchmarkStructure(
        symbols=new_symbols,
        positions=np.array(new_positions, dtype=float),
        cell=new_cell,
        pbc=structure.pbc,
        species_index=new_species,
        source_atom=source_atom,
        cell_offset=cell_offset,
        metadata=metadata,
    )


def find_central_atom(
    structure: BenchmarkStructure,
    species: str | None = None,
) -> int:
    """Return the index of the atom closest to the structure's geometric center.

    The reference center is the centroid of the atomic positions (cartesian),
    which is robust for slabs with vacuum (unlike the raw cell center). Ties are
    broken deterministically by the lowest atom index.

    Parameters
    ----------
    species:
        Optional chemical symbol filter. When given, only atoms of that species
        are considered candidates (the center is still the full-structure
        centroid). Kept minimal on purpose; extend later if needed.
    """
    if structure.n_atoms == 0:
        raise StructureError("Cannot find a central atom in an empty structure.")

    center = structure.positions.mean(axis=0)
    candidates = range(structure.n_atoms)
    if species is not None:
        candidates = [
            idx for idx in candidates if structure.symbols[idx] == str(species)
        ]
        if not candidates:
            raise StructureError(
                f"No atoms of species {species!r} found in structure."
            )

    best_index = -1
    best_distance = np.inf
    for idx in candidates:
        distance = float(np.linalg.norm(structure.positions[idx] - center))
        if distance < best_distance - 1e-12:
            best_distance = distance
            best_index = idx
    return best_index


def make_displaced_structures(
    structure: BenchmarkStructure,
    atom_index: int,
    direction: Any,
    displacement: float,
) -> tuple[BenchmarkStructure, BenchmarkStructure]:
    """Return ``(plus, minus)`` copies with a single atom moved by ``±h``.

    Only one coordinate of one atom changes. The input structure is never
    mutated. ``displacement`` must be strictly positive.
    """
    if not isinstance(atom_index, (int, np.integer)) or isinstance(atom_index, bool):
        raise StructureError(f"atom_index must be an int, got {atom_index!r}")
    atom_index = int(atom_index)
    if atom_index < 0 or atom_index >= structure.n_atoms:
        raise StructureError(
            f"atom_index {atom_index} out of range [0, {structure.n_atoms})."
        )
    axis = normalize_direction(direction)
    h = float(displacement)
    if not np.isfinite(h) or h <= 0:
        raise StructureError(f"displacement must be positive, got {displacement!r}")

    plus = structure.copy()
    minus = structure.copy()
    plus.positions[atom_index, axis] += h
    minus.positions[atom_index, axis] -= h
    provenance = {
        "displaced_atom": atom_index,
        "direction": direction_name(axis),
        "direction_axis": axis,
        "displacement": h,
    }
    plus.metadata = {**plus.metadata, **provenance, "sign": "+"}
    minus.metadata = {**minus.metadata, **provenance, "sign": "-"}
    return plus, minus
