#!/usr/bin/env python3
"""Extract eigenvalues from archived comparison Hamiltonians."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sisl


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
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


def read_matrix_eigenvalues(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    sile = sisl.get_sile(str(path))
    hamiltonian = sile.read_hamiltonian()
    eigenvalues = np.asarray(hamiltonian.eigh(), dtype=float)
    overlap_error = None
    has_overlap = False
    try:
        overlap = sile.read_overlap()
        has_overlap = overlap is not None
    except Exception as exc:  # pragma: no cover - depends on file backend.
        overlap_error = str(exc)
    metadata = {
        "matrix_path": str(path),
        "n_bands": int(eigenvalues.size),
        "orthogonal": bool(getattr(hamiltonian, "orthogonal", False)),
        "has_overlap": has_overlap,
        "overlap_error": overlap_error,
    }
    return eigenvalues, metadata


def eigenvalue_rows(values: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"band": index, "eigenvalue_eV": float(value)}
        for index, value in enumerate(values)
    ]


def compare_rows(siesta: np.ndarray, predicted: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, float]]:
    n_bands = min(siesta.size, predicted.size)
    siesta = siesta[:n_bands]
    predicted = predicted[:n_bands]
    errors = predicted - siesta
    rows = [
        {
            "band": index,
            "siesta_eV": float(siesta[index]),
            "predicted_eV": float(predicted[index]),
            "error_eV": float(errors[index]),
            "abs_error_eV": float(abs(errors[index])),
        }
        for index in range(n_bands)
    ]
    metrics = {
        "n_compared_bands": float(n_bands),
        "mae_eV": float(np.mean(np.abs(errors))) if n_bands else math.nan,
        "rmse_eV": float(np.sqrt(np.mean(errors**2))) if n_bands else math.nan,
        "max_abs_error_eV": float(np.max(np.abs(errors))) if n_bands else math.nan,
        "mean_signed_error_eV": float(np.mean(errors)) if n_bands else math.nan,
    }
    return rows, metrics


def extract(result_dir: Path) -> dict[str, Any]:
    prediction_root = result_dir / "predicted_hamiltonians"
    reference_root = result_dir / "siesta_hamiltonians"
    output_root = result_dir / "eigenvalues"
    prediction_dirs = sample_dirs(prediction_root)
    reference_dirs = sample_dirs(reference_root)
    sample_names = sorted(set(prediction_dirs) | set(reference_dirs))

    metrics_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for sample in sample_names:
        predicted_path = find_prediction(prediction_dirs[sample]) if sample in prediction_dirs else None
        reference_path = find_reference(reference_dirs[sample]) if sample in reference_dirs else None
        sample_values: dict[str, np.ndarray] = {}

        for kind, matrix_path in (("siesta", reference_path), ("predicted", predicted_path)):
            if matrix_path is None:
                continue
            try:
                values, metadata = read_matrix_eigenvalues(matrix_path)
            except Exception as exc:
                errors.append({"sample": sample, "kind": kind, "error": str(exc)})
                continue
            sample_values[kind] = values
            write_csv(
                output_root / kind / f"{sample}.csv",
                ["band", "eigenvalue_eV"],
                eigenvalue_rows(values),
            )
            overlap_rows.append(
                {
                    "sample": sample,
                    "kind": kind,
                    "matrix_path": metadata["matrix_path"],
                    "n_bands": metadata["n_bands"],
                    "orthogonal": metadata["orthogonal"],
                    "has_overlap": metadata["has_overlap"],
                    "overlap_error": metadata["overlap_error"],
                }
            )

        if "siesta" not in sample_values or "predicted" not in sample_values:
            continue
        band_rows, sample_metrics = compare_rows(sample_values["siesta"], sample_values["predicted"])
        write_csv(
            output_root / "band_errors" / f"{sample}.csv",
            ["band", "siesta_eV", "predicted_eV", "error_eV", "abs_error_eV"],
            band_rows,
        )
        metrics_rows.append(
            {
                "sample": sample,
                "siesta_bands": int(sample_values["siesta"].size),
                "predicted_bands": int(sample_values["predicted"].size),
                **sample_metrics,
            }
        )

    write_csv(
        output_root / "overlap_summary.csv",
        ["sample", "kind", "matrix_path", "n_bands", "orthogonal", "has_overlap", "overlap_error"],
        overlap_rows,
    )
    write_csv(
        output_root / "eigenvalue_metrics.csv",
        [
            "sample",
            "siesta_bands",
            "predicted_bands",
            "n_compared_bands",
            "mae_eV",
            "rmse_eV",
            "max_abs_error_eV",
            "mean_signed_error_eV",
        ],
        metrics_rows,
    )
    manifest = {
        "result_dir": str(result_dir),
        "samples_seen": len(sample_names),
        "samples_compared": len(metrics_rows),
        "overlap_entries": len(overlap_rows),
        "errors": errors,
        "outputs": {
            "siesta": str(output_root / "siesta"),
            "predicted": str(output_root / "predicted"),
            "band_errors": str(output_root / "band_errors"),
            "metrics": str(output_root / "eigenvalue_metrics.csv"),
            "overlap_summary": str(output_root / "overlap_summary.csv"),
        },
    }
    (output_root / "manifest.json").write_text(
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
