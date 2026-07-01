"""High-level comparisons + finite-difference derivatives for ML predictors.

Nothing here launches SIESTA or trains. Finite differences reuse
:func:`make_displaced_structures` from :mod:`structure`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .matrices import (
    MatrixData,
    compute_matrix_error,
    load_siesta_matrices,
    validate_matrix_compatible,
)
from .predictors import MatrixPredictor
from .structure import direction_name, make_displaced_structures, normalize_direction


def compare_model_to_siesta(
    siesta_dir,
    predictor: MatrixPredictor,
    structure,
    targets: list[str],
) -> dict[str, Any]:
    """Compare a predictor's matrices against SIESTA references.

    Loads SIESTA matrices, asks the predictor for the same targets, validates
    compatibility and returns a serializable ``{target: error_summary_dict}``.
    """
    siesta = load_siesta_matrices(siesta_dir, list(targets))
    predicted = predictor.predict(structure, list(targets))
    summary: dict[str, Any] = {}
    for target in targets:
        if target not in siesta:
            raise KeyError(f"SIESTA reference missing target {target!r}.")
        if target not in predicted:
            raise KeyError(f"Predictor {predictor.name} did not return {target!r}.")
        validate_matrix_compatible(siesta[target], predicted[target])
        summary[target] = compute_matrix_error(
            siesta[target], predicted[target]
        ).to_dict()
    return {
        "model": predictor.name,
        "targets": list(targets),
        "errors": summary,
    }


def finite_difference_matrix_derivative(
    predictor: MatrixPredictor,
    structure,
    atom_index: int,
    direction: Any,
    displacement: float,
    targets: list[str],
) -> dict[str, MatrixData]:
    """Central finite-difference derivative ``(M_plus - M_minus) / (2h)``.

    Returns one :class:`MatrixData` per target holding the derivative matrix.
    """
    axis = normalize_direction(direction)
    h = float(displacement)
    plus_structure, minus_structure = make_displaced_structures(
        structure, atom_index, axis, h
    )
    m_plus = predictor.predict(plus_structure, list(targets))
    m_minus = predictor.predict(minus_structure, list(targets))

    out: dict[str, MatrixData] = {}
    for target in targets:
        if target not in m_plus or target not in m_minus:
            raise KeyError(f"Predictor {predictor.name} did not return {target!r}.")
        validate_matrix_compatible(m_plus[target], m_minus[target])
        derivative = (
            np.asarray(m_plus[target].values, dtype=float)
            - np.asarray(m_minus[target].values, dtype=float)
        ) / (2.0 * h)
        out[target] = MatrixData(
            values=derivative,
            target=target,
            atom_order=m_plus[target].atom_order,
            orbital_labels=m_plus[target].orbital_labels,
            metadata={
                "derivative": True,
                "displaced_atom": int(atom_index),
                "direction": direction_name(axis),
                "displacement": h,
                "model": predictor.name,
                "units": "matrix/Angstrom",
            },
        )
    return out


def torch_finite_difference_matrix_derivative(
    model_fn,
    positions,
    atom_index: int,
    direction: Any,
    displacement: float,
):
    """Differentiable central finite difference kept entirely in torch.

    ``model_fn(positions) -> torch.Tensor`` must stay differentiable w.r.t. model
    parameters; this helper never converts to numpy. Import of torch is guarded
    so the rest of the package works without torch installed.

    Returns the derivative tensor ``(M_plus - M_minus) / (2h)``.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "PyTorch is required for torch_finite_difference_matrix_derivative."
        ) from exc

    axis = normalize_direction(direction)
    h = float(displacement)
    if h <= 0:
        raise ValueError("displacement must be positive.")
    positions = torch.as_tensor(positions)
    atom_index = int(atom_index)
    if atom_index < 0 or atom_index >= positions.shape[0]:
        raise IndexError(f"atom_index {atom_index} out of range.")

    step = torch.zeros_like(positions)
    step[atom_index, axis] = h
    m_plus = model_fn(positions + step)
    m_minus = model_fn(positions - step)
    return (m_plus - m_minus) / (2.0 * h)


def compare_derivatives_to_siesta(
    siesta_derivative_dir,
    predictor: MatrixPredictor,
    structure,
    atom_index: int,
    directions,
    displacement: float,
    targets: list[str],
) -> dict[str, Any]:
    """Compare ML finite-difference derivatives against SIESTA derivatives.

    ``siesta_derivative_dir`` is expected to already contain SIESTA derivative
    matrices laid out per direction as ``<dir>/`` subdirectories (each holding
    ``<target>.npy`` in the fixture format). This never runs SIESTA. Errors are
    reported per target / direction / model / displacement / atom.
    """
    from pathlib import Path

    base = Path(siesta_derivative_dir)
    results: dict[str, Any] = {
        "model": predictor.name,
        "displaced_atom": int(atom_index),
        "displacement": float(displacement),
        "directions": {},
    }
    for direction in directions:
        name = direction_name(direction)
        ml_derivative = finite_difference_matrix_derivative(
            predictor, structure, atom_index, direction, displacement, list(targets)
        )
        siesta_dir = base / name
        siesta_derivative = load_siesta_matrices(siesta_dir, list(targets))
        per_target: dict[str, Any] = {}
        for target in targets:
            validate_matrix_compatible(
                siesta_derivative[target], ml_derivative[target]
            )
            per_target[target] = compute_matrix_error(
                siesta_derivative[target], ml_derivative[target]
            ).to_dict()
        results["directions"][name] = {"errors": per_target}
    return results
