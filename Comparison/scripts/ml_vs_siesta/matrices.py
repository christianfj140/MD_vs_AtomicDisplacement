"""Matrix container, compatibility checks, SIESTA loading and error metrics.

``compute_matrix_error`` deliberately keeps the *numeric* error computation
independent of any plotting/UI code so it can be reused by the Matrix Viewer
backend without importing the heavy ``export_graph2mat_matrix_error_plot`` CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_TARGETS = ("hamiltonian", "density_matrix", "overlap")

# Candidate on-disk filenames for each target, most-specific first. ``.npy`` /
# ``.npz`` are the lightweight fixture format used by tests and dry-runs.
_TARGET_FILENAMES = {
    "hamiltonian": ("hamiltonian.npy", "H.npy", "hamiltonian.npz"),
    "density_matrix": ("density_matrix.npy", "DM.npy", "density_matrix.npz"),
    "overlap": ("overlap.npy", "S.npy", "overlap.npz"),
}


class MatrixCompatibilityError(RuntimeError):
    """Raised when two matrices cannot be compared."""


@dataclass
class MatrixData:
    """A single matrix plus the metadata needed to compare it against another."""

    values: np.ndarray
    target: str
    basis: Any | None = None
    atom_order: list[int] | None = None
    orbital_labels: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        self.target = str(self.target)
        if self.atom_order is not None:
            self.atom_order = [int(a) for a in self.atom_order]
        if self.orbital_labels is not None:
            self.orbital_labels = [str(o) for o in self.orbital_labels]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.values.shape)

    def to_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "target": self.target,
            "shape": list(self.shape),
            "atom_order": self.atom_order,
            "orbital_labels": self.orbital_labels,
            "metadata": self.metadata,
        }
        if include_values:
            payload["values"] = np.asarray(self.values).tolist()
        return payload


def validate_matrix_compatible(a: MatrixData, b: MatrixData) -> None:
    """Raise :class:`MatrixCompatibilityError` if ``a`` and ``b`` cannot compare.

    Checks shape, target, atom order (when both provide it), orbital labels
    (when both provide them) and units (``metadata["units"]`` when both set it).
    """
    if a.shape != b.shape:
        raise MatrixCompatibilityError(
            f"shape mismatch: {a.shape} vs {b.shape}."
        )
    if a.target != b.target:
        raise MatrixCompatibilityError(
            f"target mismatch: {a.target!r} vs {b.target!r}."
        )
    if a.atom_order is not None and b.atom_order is not None:
        if a.atom_order != b.atom_order:
            raise MatrixCompatibilityError("atom_order mismatch.")
    if a.orbital_labels is not None and b.orbital_labels is not None:
        if a.orbital_labels != b.orbital_labels:
            raise MatrixCompatibilityError("orbital_labels mismatch.")
    unit_a = a.metadata.get("units")
    unit_b = b.metadata.get("units")
    if unit_a is not None and unit_b is not None and unit_a != unit_b:
        raise MatrixCompatibilityError(
            f"units mismatch: {unit_a!r} vs {unit_b!r}."
        )


@dataclass
class MatrixErrorSummary:
    """Numeric error summary between a reference and a predicted matrix."""

    target: str
    shape: tuple[int, ...]
    mae: float
    rmse: float
    max_abs_error: float
    relative_mae: float | None
    n_elements: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "shape": list(self.shape),
            "mae": self.mae,
            "rmse": self.rmse,
            "max_abs_error": self.max_abs_error,
            "relative_mae": self.relative_mae,
            "n_elements": self.n_elements,
        }


def compute_matrix_error(
    reference: MatrixData,
    predicted: MatrixData,
) -> MatrixErrorSummary:
    """Compute MAE / RMSE / max-abs / relative-MAE between two matrices."""
    validate_matrix_compatible(reference, predicted)
    ref = np.asarray(reference.values, dtype=float)
    pred = np.asarray(predicted.values, dtype=float)
    diff = pred - ref
    abs_diff = np.abs(diff)
    mae = float(abs_diff.mean()) if abs_diff.size else 0.0
    rmse = float(np.sqrt((diff**2).mean())) if diff.size else 0.0
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    ref_scale = float(np.abs(ref).mean()) if ref.size else 0.0
    # Relative MAE only when the reference magnitude is large enough to be stable.
    relative_mae = mae / ref_scale if ref_scale > 1e-12 else None
    return MatrixErrorSummary(
        target=reference.target,
        shape=reference.shape,
        mae=mae,
        rmse=rmse,
        max_abs_error=max_abs,
        relative_mae=relative_mae,
        n_elements=int(ref.size),
    )


def _load_matrix_file(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as data:
            key = "values" if "values" in data else data.files[0]
            return np.asarray(data[key])
    return np.asarray(np.load(path))


def load_siesta_matrices(
    path: str | Path,
    targets: list[str],
) -> dict[str, MatrixData]:
    """Load SIESTA reference matrices for the requested targets.

    The lightweight fixture/dry-run format is a directory containing
    ``<target>.npy`` (or ``H.npy``/``DM.npy``/``S.npy``) with an optional
    ``<target>.meta.json`` sidecar (``atom_order`` / ``orbital_labels`` / units).

    Heavy SIESTA parsers (HSX/TSHS via sisl) already live in the derivative
    scripts and are *not* duplicated here. When a target file cannot be found a
    clear error names the missing file.
    """
    base = Path(path)
    if not base.is_dir():
        raise FileNotFoundError(f"SIESTA matrix directory not found: {base}")

    out: dict[str, MatrixData] = {}
    for target in targets:
        if target not in _TARGET_FILENAMES:
            raise ValueError(f"Unsupported target {target!r}.")
        matrix_path: Path | None = None
        for candidate in _TARGET_FILENAMES[target]:
            candidate_path = base / candidate
            if candidate_path.is_file():
                matrix_path = candidate_path
                break
        if matrix_path is None:
            expected = ", ".join(_TARGET_FILENAMES[target])
            raise FileNotFoundError(
                f"No SIESTA matrix file for target {target!r} in {base}. "
                f"Expected one of: {expected}."
            )
        values = _load_matrix_file(matrix_path)
        metadata: dict[str, Any] = {"source": str(matrix_path)}
        atom_order = None
        orbital_labels = None
        meta_path = base / f"{target}.meta.json"
        if meta_path.is_file():
            sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
            atom_order = sidecar.get("atom_order")
            orbital_labels = sidecar.get("orbital_labels")
            if "units" in sidecar:
                metadata["units"] = sidecar["units"]
            metadata["sidecar"] = str(meta_path)
        out[target] = MatrixData(
            values=values,
            target=target,
            atom_order=atom_order,
            orbital_labels=orbital_labels,
            metadata=metadata,
        )
    return out


def _orbital_atom_index(matrix: MatrixData, n_orbitals: int) -> list[int] | None:
    """Best-effort orbital→atom mapping from matrix metadata.

    Recognized metadata keys (first match wins):
    - ``orbital_atom_index``: explicit list, one atom index per orbital.
    - ``orbitals_per_atom`` + ``atom_order``: counts expanded in atom order.
    """
    mapping = matrix.metadata.get("orbital_atom_index")
    if mapping is not None:
        return [int(a) for a in mapping]
    per_atom = matrix.metadata.get("orbitals_per_atom")
    if per_atom is not None:
        atoms = matrix.atom_order or list(range(len(per_atom)))
        expanded: list[int] = []
        for atom, count in zip(atoms, per_atom):
            expanded.extend([int(atom)] * int(count))
        if len(expanded) == n_orbitals:
            return expanded
    return None


def compute_error_by_species_pair(
    reference: MatrixData,
    predicted: MatrixData,
    structure,
) -> dict[str, Any]:
    """Group matrix errors by ordered species pair (e.g. ``C-C``, ``C-H``).

    Requires an orbital→atom mapping in the matrix metadata (see
    :func:`_orbital_atom_index`). When the mapping is unavailable a structured
    warning is returned instead of raising.
    """
    validate_matrix_compatible(reference, predicted)
    ref = np.asarray(reference.values, dtype=float)
    pred = np.asarray(predicted.values, dtype=float)
    if ref.ndim != 2:
        return {
            "pairs": {},
            "warnings": [
                f"species-pair errors need a 2D matrix, got shape {reference.shape}."
            ],
        }
    n_rows, n_cols = ref.shape
    row_atoms = _orbital_atom_index(reference, n_rows) or _orbital_atom_index(
        predicted, n_rows
    )
    col_atoms = _orbital_atom_index(reference, n_cols) or _orbital_atom_index(
        predicted, n_cols
    )
    if row_atoms is None or col_atoms is None:
        return {
            "pairs": {},
            "warnings": [
                "No orbital→atom mapping available (set metadata "
                "'orbital_atom_index' or 'orbitals_per_atom'); "
                "cannot group by species pair."
            ],
        }

    symbols = list(getattr(structure, "symbols", []))
    warnings: list[str] = []
    diff = np.abs(pred - ref)

    def _symbol(atom_index: int) -> str | None:
        if 0 <= atom_index < len(symbols):
            return symbols[atom_index]
        return None

    buckets: dict[str, list[float]] = {}
    ref_buckets: dict[str, list[float]] = {}
    for i in range(n_rows):
        sym_i = _symbol(row_atoms[i])
        if sym_i is None:
            continue
        for j in range(n_cols):
            sym_j = _symbol(col_atoms[j])
            if sym_j is None:
                continue
            pair = "-".join(sorted((sym_i, sym_j)))
            buckets.setdefault(pair, []).append(float(diff[i, j]))
            ref_buckets.setdefault(pair, []).append(abs(float(ref[i, j])))

    if not buckets:
        warnings.append("No orbital pair matched a known species symbol.")

    pairs: dict[str, Any] = {}
    for pair, errors in buckets.items():
        errors_arr = np.asarray(errors, dtype=float)
        ref_arr = np.asarray(ref_buckets[pair], dtype=float)
        ref_scale = float(ref_arr.mean()) if ref_arr.size else 0.0
        mae = float(errors_arr.mean())
        pairs[pair] = {
            "mae": mae,
            "rmse": float(np.sqrt((errors_arr**2).mean())),
            "max_abs_error": float(errors_arr.max()),
            "relative_mae": mae / ref_scale if ref_scale > 1e-12 else None,
            "n_elements": int(errors_arr.size),
        }
    return {"pairs": pairs, "warnings": warnings}
