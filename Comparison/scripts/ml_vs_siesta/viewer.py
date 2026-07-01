"""Matrix Viewer payloads (backend-only, UI-agnostic, JSON serializable).

``prepare_matrix_plot_payload`` is the common function that both the legacy
``plotMatrixError`` fallback and the new explicit Graph2Mat matrix view can feed
into. It accepts a raw matrix and an optional error matrix and produces a
heatmap-ready payload without importing any plotting library.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .matrices import MatrixData, compute_matrix_error


def _matrix_values(matrix: MatrixData | np.ndarray | None) -> np.ndarray | None:
    if matrix is None:
        return None
    if isinstance(matrix, MatrixData):
        return np.asarray(matrix.values, dtype=float)
    return np.asarray(matrix, dtype=float)


def _downsample(values: np.ndarray, max_size: int) -> tuple[np.ndarray, bool]:
    """Cap the heatmap grid so payloads stay small. Returns (values, truncated)."""
    if values.ndim != 2:
        return values, False
    rows, cols = values.shape
    if rows <= max_size and cols <= max_size:
        return values, False
    return values[:max_size, :max_size], True


def prepare_matrix_plot_payload(
    matrix: MatrixData | np.ndarray | None,
    error: MatrixData | np.ndarray | None = None,
    *,
    target: str | None = None,
    label: str | None = None,
    max_size: int = 128,
) -> dict[str, Any]:
    """Build a heatmap-ready payload from a raw matrix and/or an error matrix.

    Supports both the raw-matrix view (Graph2Mat/SIESTA) and the error view used
    by the old ``plotMatrixError`` fallback. Scales ``linear`` and ``log_abs``
    are advertised; the front-end applies the transform.
    """
    values = _matrix_values(matrix)
    error_values = _matrix_values(error)
    payload: dict[str, Any] = {
        "label": label,
        "target": target or (matrix.target if isinstance(matrix, MatrixData) else None),
        "scales": ["linear", "log_abs"],
    }
    if values is not None:
        capped, truncated = _downsample(values, max_size)
        payload["matrix"] = {
            "values": capped.tolist(),
            "shape": [int(s) for s in values.shape],
            "min": float(np.min(values)) if values.size else 0.0,
            "max": float(np.max(values)) if values.size else 0.0,
            "abs_max": float(np.max(np.abs(values))) if values.size else 0.0,
            "truncated": truncated,
        }
    if error_values is not None:
        capped_err, truncated_err = _downsample(error_values, max_size)
        payload["error"] = {
            "values": capped_err.tolist(),
            "shape": [int(s) for s in error_values.shape],
            "mae": float(np.abs(error_values).mean()) if error_values.size else 0.0,
            "rmse": float(np.sqrt((error_values**2).mean())) if error_values.size else 0.0,
            "max_abs_error": float(np.abs(error_values).max()) if error_values.size else 0.0,
            "truncated": truncated_err,
        }
    return payload


def build_matrix_viewer_payload(
    *,
    target: str,
    siesta: MatrixData | None = None,
    graph2mat: MatrixData | None = None,
    deeph: MatrixData | None = None,
    max_size: int = 128,
) -> dict[str, Any]:
    """Assemble the full Matrix Viewer payload for one target.

    Returns available matrices (SIESTA / Graph2Mat / DeepH), model−SIESTA
    differences and MAE/RMSE/max metrics for whichever models are present.
    """
    matrices: dict[str, Any] = {}
    for name, matrix in (
        ("siesta", siesta),
        ("graph2mat", graph2mat),
        ("deeph", deeph),
    ):
        if matrix is not None:
            matrices[name] = prepare_matrix_plot_payload(
                matrix, target=target, label=name, max_size=max_size
            )

    differences: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    siesta_values = _matrix_values(siesta)
    for name, matrix in (("graph2mat", graph2mat), ("deeph", deeph)):
        if matrix is None or siesta is None:
            continue
        diff = _matrix_values(matrix) - siesta_values
        differences[f"{name}_minus_siesta"] = prepare_matrix_plot_payload(
            diff, target=target, label=f"{name} − SIESTA", max_size=max_size
        )
        metrics[name] = compute_matrix_error(siesta, matrix).to_dict()

    return {
        "target": target,
        "available": sorted(matrices.keys()),
        "matrices": matrices,
        "differences": differences,
        "metrics": metrics,
    }


def build_derivative_viewer_payload(
    *,
    target: str,
    atom_index: int,
    direction: str,
    displacement: float,
    ml_derivative: MatrixData,
    siesta_derivative: MatrixData | None = None,
    model: str | None = None,
    max_size: int = 128,
) -> dict[str, Any]:
    """Assemble a derivative-view payload, reusing the matrix plot payload.

    Shows the ML derivative matrix and, when SIESTA is available, the error
    against it.
    """
    error_matrix = None
    metrics = None
    if siesta_derivative is not None:
        error_matrix = _matrix_values(ml_derivative) - _matrix_values(siesta_derivative)
        metrics = compute_matrix_error(siesta_derivative, ml_derivative).to_dict()
    return {
        "target": target,
        "model": model or ml_derivative.metadata.get("model"),
        "displaced_atom": int(atom_index),
        "direction": direction,
        "displacement": float(displacement),
        "derivative": prepare_matrix_plot_payload(
            ml_derivative,
            error_matrix,
            target=target,
            label="d(matrix)/d(pos)",
            max_size=max_size,
        ),
        "metrics": metrics,
    }
