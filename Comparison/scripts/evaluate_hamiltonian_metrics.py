#!/usr/bin/env python3
"""Evaluate sparse, spectral and total-DOS metrics for archived Hamiltonians."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg
from scipy import sparse
import sisl


SUPPORT_THRESHOLD = 1e-12
FERMI_WINDOW_EV = 2.0
DOS_SIGMA_EV = 0.10
DOS_POINTS = 1000


@dataclass
class MatrixData:
    path: Path
    hamiltonian: sparse.csr_matrix
    overlap: sparse.csr_matrix | None
    own_eigenvalues: np.ndarray
    fermi_level: float | None
    fermi_level_source: str | None
    orthogonal: bool
    has_overlap: bool
    overlap_error: str | None


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_dirs(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}
    return {
        path.name: path
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_dir()
    }


def matrix_sort_key(path: Path) -> tuple[int, str]:
    numbers: list[int] = []
    for chunk in path.stem.replace("-", ".").replace("_", ".").split("."):
        if chunk.isdigit():
            numbers.append(int(chunk))
    return (numbers[-1] if numbers else 10**9, path.name)


def find_prediction(sample_dir: Path) -> Path | None:
    direct = sample_dir / "ML_prediction.HSX"
    if direct.exists():
        return direct
    matches = sorted(sample_dir.glob("*ML_prediction*.HSX"), key=matrix_sort_key)
    return matches[0] if matches else None


def find_reference(sample_dir: Path) -> Path | None:
    matrices = [
        path
        for path in sorted(
            list(sample_dir.glob("*.TSHS")) + list(sample_dir.glob("*.HSX")),
            key=matrix_sort_key,
        )
        if path.name != "ML_prediction.HSX"
    ]
    return matrices[0] if matrices else None


def read_matrix(path: Path) -> MatrixData:
    sile = sisl.get_sile(str(path))
    hamiltonian_obj = sile.read_hamiltonian()
    hamiltonian = hamiltonian_obj.tocsr(0).astype(float)
    overlap = None
    has_overlap = False
    overlap_error = None
    try:
        overlap_obj = sile.read_overlap()
        overlap = overlap_obj.tocsr().astype(float)
        has_overlap = overlap is not None
    except Exception as exc:  # pragma: no cover - backend dependent.
        overlap_error = str(exc)
    try:
        own_eigenvalues = np.asarray(hamiltonian_obj.eigh(), dtype=float)
    except Exception:
        own_eigenvalues = np.asarray([], dtype=float)
    try:
        fermi_level = float(sile.read_fermi_level())
        fermi_level_source = "siesta_file"
    except Exception:
        fermi_level = None
        fermi_level_source = "unavailable"
    return MatrixData(
        path=path,
        hamiltonian=hamiltonian,
        overlap=overlap,
        own_eigenvalues=own_eigenvalues,
        fermi_level=fermi_level,
        fermi_level_source=fermi_level_source,
        orthogonal=bool(getattr(hamiltonian_obj, "orthogonal", False)),
        has_overlap=has_overlap,
        overlap_error=overlap_error,
    )


def sparse_norm(matrix: sparse.spmatrix) -> float:
    return float(np.sqrt(np.abs(matrix).power(2).sum()))


def hermiticity_defect(matrix: sparse.csr_matrix) -> float:
    denominator = sparse_norm(matrix)
    if denominator == 0:
        return math.nan
    return sparse_norm(matrix - matrix.getH()) / denominator


def csr_value_dict(matrix: sparse.csr_matrix, threshold: float) -> dict[tuple[int, int], complex]:
    coo = matrix.tocoo()
    values: dict[tuple[int, int], complex] = {}
    for row, col, value in zip(coo.row, coo.col, coo.data, strict=False):
        if abs(value) > threshold:
            values[(int(row), int(col))] = value
    return values


def mean_abs(values: list[complex]) -> float:
    return float(np.mean(np.abs(values))) if values else math.nan


def rmse(values: list[complex]) -> float:
    return float(np.sqrt(np.mean(np.abs(values) ** 2))) if values else math.nan


def sparse_metrics(sample: str, reference: MatrixData, predicted: MatrixData) -> dict[str, Any]:
    ref_values = csr_value_dict(reference.hamiltonian, SUPPORT_THRESHOLD)
    pred_values = csr_value_dict(predicted.hamiltonian, SUPPORT_THRESHOLD)
    ref_support = set(ref_values)
    pred_support = set(pred_values)
    union_support = ref_support | pred_support
    intersection = ref_support & pred_support

    false_zeros = ref_support - pred_support
    false_nonzeros = pred_support - ref_support
    deltas_ref = [pred_values.get(index, 0.0) - ref_values[index] for index in ref_support]
    deltas_pred = [pred_values[index] - ref_values.get(index, 0.0) for index in pred_support]
    deltas_union = [
        pred_values.get(index, 0.0) - ref_values.get(index, 0.0)
        for index in union_support
    ]
    ref_fro = float(np.sqrt(sum(abs(value) ** 2 for value in ref_values.values())))
    ref_pattern_fro = float(np.sqrt(sum(abs(value) ** 2 for value in deltas_ref)))
    union_fro = float(np.sqrt(sum(abs(value) ** 2 for value in deltas_union)))
    ref_l1 = float(sum(abs(value) for value in ref_values.values()))
    union_l1 = float(sum(abs(value) for value in deltas_union))
    precision = len(intersection) / len(pred_support) if pred_support else math.nan
    recall = len(intersection) / len(ref_support) if ref_support else math.nan
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else math.nan
    )
    n_entries = reference.hamiltonian.shape[0] * reference.hamiltonian.shape[1]
    return {
        "sample": sample,
        "n_orbitals": reference.hamiltonian.shape[0],
        "n_entries": n_entries,
        "ref_nnz": len(ref_support),
        "pred_nnz": len(pred_support),
        "union_nnz": len(union_support),
        "ref_density": len(ref_support) / n_entries if n_entries else math.nan,
        "pred_density": len(pred_support) / n_entries if n_entries else math.nan,
        "mae_ref_eV": mean_abs(deltas_ref),
        "rmse_ref_eV": rmse(deltas_ref),
        "mae_pred_eV": mean_abs(deltas_pred),
        "rmse_pred_eV": rmse(deltas_pred),
        "mae_union_eV": mean_abs(deltas_union),
        "rmse_union_eV": rmse(deltas_union),
        "max_abs_error_union_eV": float(max((abs(value) for value in deltas_union), default=math.nan)),
        "relative_frobenius_ref": ref_pattern_fro / ref_fro if ref_fro else math.nan,
        "relative_frobenius_union": union_fro / ref_fro if ref_fro else math.nan,
        "relative_l1_union": union_l1 / ref_l1 if ref_l1 else math.nan,
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
        "false_zeros": len(false_zeros),
        "false_nonzeros": len(false_nonzeros),
        "false_zero_rate": len(false_zeros) / len(ref_support) if ref_support else math.nan,
        "false_nonzero_rate": len(false_nonzeros) / len(pred_support) if pred_support else math.nan,
        "weighted_false_zeros_eV": float(sum(abs(ref_values[index]) for index in false_zeros)),
        "weighted_false_nonzeros_eV": float(sum(abs(pred_values[index]) for index in false_nonzeros)),
        "hermiticity_ref": hermiticity_defect(reference.hamiltonian),
        "hermiticity_pred": hermiticity_defect(predicted.hamiltonian),
    }


def symmetrized_dense(matrix: sparse.csr_matrix) -> np.ndarray:
    dense = matrix.toarray()
    return np.asarray((dense + dense.conj().T) / 2.0, dtype=float)


def generalized_eigenvalues(
    hamiltonian: sparse.csr_matrix,
    overlap: sparse.csr_matrix | None,
) -> np.ndarray:
    dense_h = symmetrized_dense(hamiltonian)
    if overlap is None:
        return np.linalg.eigvalsh(dense_h)
    dense_s = symmetrized_dense(overlap)
    return np.asarray(
        scipy.linalg.eigh(dense_h, dense_s, eigvals_only=True, check_finite=False),
        dtype=float,
    )


def eigenvalue_rows(values: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"band": index, "eigenvalue_eV": float(value)}
        for index, value in enumerate(values)
    ]


def band_gap(values: np.ndarray, fermi_level: float | None) -> float:
    if fermi_level is None or values.size == 0:
        return math.nan
    occupied = values[values <= fermi_level]
    unoccupied = values[values > fermi_level]
    if occupied.size == 0 or unoccupied.size == 0:
        return math.nan
    return float(unoccupied[0] - occupied[-1])


def eigen_error_metrics(
    reference: np.ndarray,
    predicted: np.ndarray,
    fermi_level: float | None,
    fermi_level_source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_bands = min(reference.size, predicted.size)
    reference = reference[:n_bands]
    predicted = predicted[:n_bands]
    errors = predicted - reference
    band_rows = [
        {
            "band": index,
            "siesta_eV": float(reference[index]),
            "predicted_eV": float(predicted[index]),
            "error_eV": float(errors[index]),
            "abs_error_eV": float(abs(errors[index])),
            "siesta_minus_fermi_eV": None
            if fermi_level is None
            else float(reference[index] - fermi_level),
        }
        for index in range(n_bands)
    ]

    occupied_mask = np.zeros(n_bands, dtype=bool)
    fermi_mask = np.zeros(n_bands, dtype=bool)
    if fermi_level is not None:
        occupied_mask = reference <= fermi_level
        fermi_mask = np.abs(reference - fermi_level) <= FERMI_WINDOW_EV

    def masked_mae(mask: np.ndarray) -> float:
        return float(np.mean(np.abs(errors[mask]))) if np.any(mask) else math.nan

    def masked_rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(errors[mask] ** 2))) if np.any(mask) else math.nan

    gap_ref = band_gap(reference, fermi_level)
    gap_pred = band_gap(predicted, fermi_level)
    metrics = {
        "n_compared_bands": n_bands,
        "fermi_ref_eV": fermi_level,
        "fermi_level_source": fermi_level_source,
        "global_mae_eV": float(np.mean(np.abs(errors))) if n_bands else math.nan,
        "global_rmse_eV": float(np.sqrt(np.mean(errors**2))) if n_bands else math.nan,
        "global_max_abs_error_eV": float(np.max(np.abs(errors))) if n_bands else math.nan,
        "global_mean_signed_error_eV": float(np.mean(errors)) if n_bands else math.nan,
        "occupied_bands": int(np.count_nonzero(occupied_mask)),
        "occupied_mae_eV": masked_mae(occupied_mask),
        "occupied_rmse_eV": masked_rmse(occupied_mask),
        "fermi_window_eV": FERMI_WINDOW_EV,
        "fermi_window_bands": int(np.count_nonzero(fermi_mask)),
        "fermi_window_mae_eV": masked_mae(fermi_mask),
        "fermi_window_rmse_eV": masked_rmse(fermi_mask),
        "gap_ref_eV": gap_ref,
        "gap_pred_eV": gap_pred,
        "gap_abs_error_eV": abs(gap_pred - gap_ref)
        if gap_ref == gap_ref and gap_pred == gap_pred
        else math.nan,
    }
    return band_rows, metrics


def gaussian_dos(values: np.ndarray, grid: np.ndarray, sigma: float) -> np.ndarray:
    prefactor = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    dos = np.zeros_like(grid, dtype=float)
    for value in values:
        dos += prefactor * np.exp(-0.5 * ((grid - value) / sigma) ** 2)
    return dos


def normalized_density(density: np.ndarray, dx: float) -> np.ndarray:
    area = float(np.sum(density) * dx)
    if area <= 0:
        return density
    return density / area


def wasserstein_from_grid(a_density: np.ndarray, b_density: np.ndarray, dx: float) -> float:
    a_norm = normalized_density(a_density, dx)
    b_norm = normalized_density(b_density, dx)
    cdf_delta = np.cumsum(a_norm - b_norm) * dx
    return float(np.sum(np.abs(cdf_delta)) * dx)


def dos_for_sample(reference: np.ndarray, predicted: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    combined = np.concatenate([reference, predicted])
    if combined.size == 0:
        return [], {
            "dos_wasserstein_eV": math.nan,
            "dos_l1": math.nan,
            "dos_l2": math.nan,
            "energy_min_eV": math.nan,
            "energy_max_eV": math.nan,
        }
    margin = max(5.0 * DOS_SIGMA_EV, 0.05 * float(np.ptp(combined) if combined.size > 1 else 1.0))
    energy_min = float(np.min(combined) - margin)
    energy_max = float(np.max(combined) + margin)
    grid = np.linspace(energy_min, energy_max, DOS_POINTS)
    dx = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
    ref_dos = gaussian_dos(reference, grid, DOS_SIGMA_EV)
    pred_dos = gaussian_dos(predicted, grid, DOS_SIGMA_EV)
    ref_norm = normalized_density(ref_dos, dx)
    pred_norm = normalized_density(pred_dos, dx)
    rows = [
        {
            "energy_eV": float(grid[index]),
            "siesta_dos": float(ref_dos[index]),
            "predicted_dos": float(pred_dos[index]),
            "siesta_dos_normalized": float(ref_norm[index]),
            "predicted_dos_normalized": float(pred_norm[index]),
        }
        for index in range(grid.size)
    ]
    metrics = {
        "dos_sigma_eV": DOS_SIGMA_EV,
        "dos_grid_points": DOS_POINTS,
        "energy_min_eV": energy_min,
        "energy_max_eV": energy_max,
        "dos_wasserstein_eV": wasserstein_from_grid(ref_dos, pred_dos, dx),
        "dos_l1": float(np.sum(np.abs(ref_norm - pred_norm)) * dx),
        "dos_l2": float(np.sqrt(np.sum((ref_norm - pred_norm) ** 2) * dx)),
    }
    return rows, metrics


def summarize_numeric(rows: list[dict[str, Any]], skip: set[str]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key in skip or value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values.setdefault(key, []).append(number)
    return {
        key: {
            "mean": float(np.mean(column)),
            "std": float(np.std(column)),
            "min": float(np.min(column)),
            "max": float(np.max(column)),
        }
        for key, column in values.items()
        if column
    }


def pearson_correlation(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        try:
            x = float(row[x_key])
            y = float(row[y_key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if len(pairs) < 2:
        return math.nan
    x_values = np.asarray([pair[0] for pair in pairs], dtype=float)
    y_values = np.asarray([pair[1] for pair in pairs], dtype=float)
    if float(np.std(x_values)) == 0.0 or float(np.std(y_values)) == 0.0:
        return math.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def matrix_spectrum_rows(
    sparse_rows: list[dict[str, Any]],
    spectral_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    spectral_by_sample = {str(row["sample"]): row for row in spectral_rows}
    rows: list[dict[str, Any]] = []
    for sparse_row in sparse_rows:
        spectral_row = spectral_by_sample.get(str(sparse_row["sample"]))
        if spectral_row is None:
            continue
        rows.append(
            {
                "sample": sparse_row["sample"],
                "mae_ref_eV": sparse_row.get("mae_ref_eV"),
                "rmse_ref_eV": sparse_row.get("rmse_ref_eV"),
                "rmse_union_eV": sparse_row.get("rmse_union_eV"),
                "relative_frobenius_union": sparse_row.get("relative_frobenius_union"),
                "support_f1": sparse_row.get("support_f1"),
                "global_rmse_eV": spectral_row.get("global_rmse_eV"),
                "fermi_window_rmse_eV": spectral_row.get("fermi_window_rmse_eV"),
                "gap_abs_error_eV": spectral_row.get("gap_abs_error_eV"),
                "fermi_level_source": spectral_row.get("fermi_level_source"),
                "fermi_metric_available": spectral_row.get("fermi_level_source") == "siesta_file",
            }
        )
    return rows


def matrix_spectrum_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fermi_rows = [
        row
        for row in rows
        if row.get("fermi_level_source") == "siesta_file"
    ]
    return {
        "samples": len(rows),
        "fermi_samples": len(fermi_rows),
        "corr_mae_ref_vs_global_rmse": pearson_correlation(rows, "mae_ref_eV", "global_rmse_eV"),
        "corr_rmse_ref_vs_global_rmse": pearson_correlation(rows, "rmse_ref_eV", "global_rmse_eV"),
        "corr_frobenius_vs_fermi_rmse": pearson_correlation(
            fermi_rows,
            "relative_frobenius_union",
            "fermi_window_rmse_eV",
        ),
        "corr_support_f1_vs_fermi_rmse": pearson_correlation(fermi_rows, "support_f1", "fermi_window_rmse_eV"),
    }


def extract(result_dir: Path) -> dict[str, Any]:
    prediction_root = result_dir / "predicted_hamiltonians"
    reference_root = result_dir / "siesta_hamiltonians"
    eigen_root = result_dir / "eigenvalues"
    metrics_root = result_dir / "metrics"
    dos_root = result_dir / "dos"
    prediction_dirs = sample_dirs(prediction_root)
    reference_dirs = sample_dirs(reference_root)
    sample_names = sorted(set(prediction_dirs) | set(reference_dirs))

    sparse_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    dos_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for sample in sample_names:
        predicted_path = find_prediction(prediction_dirs[sample]) if sample in prediction_dirs else None
        reference_path = find_reference(reference_dirs[sample]) if sample in reference_dirs else None
        if predicted_path is None or reference_path is None:
            errors.append(
                {
                    "sample": sample,
                    "kind": "input",
                    "error": "Missing predicted or reference Hamiltonian.",
                }
            )
            continue

        try:
            reference = read_matrix(reference_path)
            predicted = read_matrix(predicted_path)
        except Exception as exc:
            errors.append({"sample": sample, "kind": "read_matrix", "error": str(exc)})
            continue

        for kind, data in (("siesta", reference), ("predicted", predicted)):
            overlap_rows.append(
                {
                    "sample": sample,
                    "kind": kind,
                    "matrix_path": str(data.path),
                    "n_bands": int(data.hamiltonian.shape[0]),
                    "orthogonal": data.orthogonal,
                    "has_overlap": data.has_overlap,
                    "overlap_error": data.overlap_error,
                    "fermi_level_eV": data.fermi_level,
                }
            )

        try:
            sparse_rows.append(sparse_metrics(sample, reference, predicted))
        except Exception as exc:
            errors.append({"sample": sample, "kind": "sparse_metrics", "error": str(exc)})

        try:
            ref_eig = generalized_eigenvalues(reference.hamiltonian, reference.overlap)
            pred_eig = generalized_eigenvalues(predicted.hamiltonian, reference.overlap)
            write_csv(eigen_root / "siesta" / f"{sample}.csv", ["band", "eigenvalue_eV"], eigenvalue_rows(ref_eig))
            write_csv(eigen_root / "predicted" / f"{sample}.csv", ["band", "eigenvalue_eV"], eigenvalue_rows(pred_eig))
            fermi_level = reference.fermi_level
            fermi_source = reference.fermi_level_source or "unavailable"
            if fermi_level is None or not math.isfinite(fermi_level):
                errors.append(
                    {
                        "sample": sample,
                        "kind": "missing_fermi_level",
                        "error": (
                            "SIESTA reference does not provide a Fermi level; "
                            "near-Fermi, occupied-band, and gap metrics were left unavailable."
                        ),
                    }
                )
                fermi_level = None
                fermi_source = "unavailable"
            band_rows, spectral_metrics = eigen_error_metrics(
                ref_eig,
                pred_eig,
                fermi_level,
                fermi_source,
            )
            write_csv(
                eigen_root / "band_errors" / f"{sample}.csv",
                ["band", "siesta_eV", "predicted_eV", "error_eV", "abs_error_eV", "siesta_minus_fermi_eV"],
                band_rows,
            )
            spectral_rows.append(
                {
                    "sample": sample,
                    "siesta_bands": int(ref_eig.size),
                    "predicted_bands": int(pred_eig.size),
                    "overlap_source": "siesta_reference",
                    "hamiltonian_symmetrized_for_spectrum": True,
                    **spectral_metrics,
                }
            )
            dos_grid_rows, dos_metrics = dos_for_sample(ref_eig, pred_eig)
            write_csv(
                dos_root / f"{sample}.csv",
                ["energy_eV", "siesta_dos", "predicted_dos", "siesta_dos_normalized", "predicted_dos_normalized"],
                dos_grid_rows,
            )
            dos_rows.append({"sample": sample, **dos_metrics})
        except Exception as exc:
            errors.append({"sample": sample, "kind": "spectral_or_dos_metrics", "error": str(exc)})

    sparse_fields = [
        "sample",
        "n_orbitals",
        "n_entries",
        "ref_nnz",
        "pred_nnz",
        "union_nnz",
        "ref_density",
        "pred_density",
        "mae_ref_eV",
        "rmse_ref_eV",
        "mae_pred_eV",
        "rmse_pred_eV",
        "mae_union_eV",
        "rmse_union_eV",
        "max_abs_error_union_eV",
        "relative_frobenius_ref",
        "relative_frobenius_union",
        "relative_l1_union",
        "support_precision",
        "support_recall",
        "support_f1",
        "false_zeros",
        "false_nonzeros",
        "false_zero_rate",
        "false_nonzero_rate",
        "weighted_false_zeros_eV",
        "weighted_false_nonzeros_eV",
        "hermiticity_ref",
        "hermiticity_pred",
    ]
    spectral_fields = [
        "sample",
        "siesta_bands",
        "predicted_bands",
        "overlap_source",
        "hamiltonian_symmetrized_for_spectrum",
        "n_compared_bands",
        "fermi_ref_eV",
        "fermi_level_source",
        "global_mae_eV",
        "global_rmse_eV",
        "global_max_abs_error_eV",
        "global_mean_signed_error_eV",
        "occupied_bands",
        "occupied_mae_eV",
        "occupied_rmse_eV",
        "fermi_window_eV",
        "fermi_window_bands",
        "fermi_window_mae_eV",
        "fermi_window_rmse_eV",
        "gap_ref_eV",
        "gap_pred_eV",
        "gap_abs_error_eV",
    ]
    relationship_rows = matrix_spectrum_rows(sparse_rows, spectral_rows)
    relationship_fields = [
        "sample",
        "mae_ref_eV",
        "rmse_ref_eV",
        "rmse_union_eV",
        "relative_frobenius_union",
        "support_f1",
        "global_rmse_eV",
        "fermi_window_rmse_eV",
        "gap_abs_error_eV",
        "fermi_level_source",
        "fermi_metric_available",
    ]
    dos_fields = [
        "sample",
        "dos_sigma_eV",
        "dos_grid_points",
        "energy_min_eV",
        "energy_max_eV",
        "dos_wasserstein_eV",
        "dos_l1",
        "dos_l2",
    ]
    overlap_fields = [
        "sample",
        "kind",
        "matrix_path",
        "n_bands",
        "orthogonal",
        "has_overlap",
        "overlap_error",
        "fermi_level_eV",
    ]

    write_csv(metrics_root / "sparse_metrics.csv", sparse_fields, sparse_rows)
    write_csv(metrics_root / "spectral_metrics.csv", spectral_fields, spectral_rows)
    write_csv(metrics_root / "dos_metrics.csv", dos_fields, dos_rows)
    write_csv(metrics_root / "matrix_spectrum_relationship.csv", relationship_fields, relationship_rows)
    write_csv(eigen_root / "eigenvalue_metrics.csv", spectral_fields, spectral_rows)
    write_csv(eigen_root / "overlap_summary.csv", overlap_fields, overlap_rows)

    summary = {
        "sparse": summarize_numeric(sparse_rows, {"sample"}),
        "spectral": summarize_numeric(spectral_rows, {"sample", "overlap_source", "hamiltonian_symmetrized_for_spectrum"}),
        "dos": summarize_numeric(dos_rows, {"sample"}),
        "matrix_spectrum": matrix_spectrum_summary(relationship_rows),
    }
    manifest = {
        "result_dir": str(result_dir),
        "samples_seen": len(sample_names),
        "samples_compared": len(spectral_rows),
        "sparse_samples": len(sparse_rows),
        "dos_samples": len(dos_rows),
        "overlap_entries": len(overlap_rows),
        "support_threshold": SUPPORT_THRESHOLD,
        "fermi_window_eV": FERMI_WINDOW_EV,
        "dos_sigma_eV": DOS_SIGMA_EV,
        "dos_points": DOS_POINTS,
        "errors": errors,
        "summary": summary,
        "outputs": {
            "metrics_root": str(metrics_root),
            "sparse_metrics": str(metrics_root / "sparse_metrics.csv"),
            "spectral_metrics": str(metrics_root / "spectral_metrics.csv"),
            "dos_metrics": str(metrics_root / "dos_metrics.csv"),
            "matrix_spectrum_relationship": str(metrics_root / "matrix_spectrum_relationship.csv"),
            "eigenvalues_siesta": str(eigen_root / "siesta"),
            "eigenvalues_predicted": str(eigen_root / "predicted"),
            "band_errors": str(eigen_root / "band_errors"),
            "dos": str(dos_root),
            "overlap_summary": str(eigen_root / "overlap_summary.csv"),
        },
    }
    (eigen_root / "manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (metrics_root / "manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    manifest = extract(args.result_dir)
    print(json.dumps(json_safe(manifest), ensure_ascii=False, allow_nan=False))
    return 0 if not manifest["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
