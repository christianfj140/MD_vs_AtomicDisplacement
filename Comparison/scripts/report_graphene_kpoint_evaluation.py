#!/usr/bin/env python3
"""Summarize the graphene k-point Hamiltonian evaluation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from evaluate_hamiltonian_metrics import parse_monkhorst_pack_kgrid
from reference_selection import choose_reference_matrix


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def percentile(values: list[float], fraction: float) -> float:
    clean = sorted(finite(values))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    pos = fraction * (len(clean) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return clean[lower]
    return clean[lower] * (upper - pos) + clean[upper] * (pos - lower)


def summarize_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = finite([as_float(row.get(metric)) for row in rows])
    if not values:
        return {
            "metric": metric,
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "p90": math.nan,
            "max": math.nan,
        }
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "metric": metric,
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def top_rows(rows: list[dict[str, str]], metric: str, limit: int = 10) -> list[dict[str, Any]]:
    ranked = [
        (as_float(row.get(metric)), row)
        for row in rows
        if math.isfinite(as_float(row.get(metric)))
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[dict[str, Any]] = []
    for value, row in ranked[:limit]:
        selected.append(
            {
                "rank_metric": metric,
                "rank_value": value,
                "sample": row.get("sample", ""),
                "k_index": row.get("k_index", ""),
                "k_label": row.get("k_label", ""),
                "kx": row.get("kx", ""),
                "ky": row.get("ky", ""),
                "kz": row.get("kz", ""),
                "h_mae_eV": row.get("h_mae_eV", ""),
                "h_rmse_eV": row.get("h_rmse_eV", ""),
                "relative_frobenius": row.get("relative_frobenius", ""),
                "global_rmse_eV": row.get("global_rmse_eV", ""),
                "low_energy_rmse_eV": row.get("low_energy_rmse_eV", ""),
                "fermi_window_rmse_eV": row.get("fermi_window_rmse_eV", ""),
                "frontier_window_rmse_eV": row.get("frontier_window_rmse_eV", ""),
                "dos_mae_500_fermi_window": row.get("dos_mae_500_fermi_window", ""),
            }
        )
    return selected


def reference_audit(result_dir: Path) -> dict[str, Any]:
    reference_root = result_dir / "siesta_hamiltonians"
    prediction_root = result_dir / "predicted_hamiltonians"
    sample_dirs = sorted([path for path in reference_root.iterdir() if path.is_dir()]) if reference_root.exists() else []
    selected: list[str] = []
    forbidden: list[str] = []
    missing_or_bad: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        selection = choose_reference_matrix(sample_dir)
        if selection.ok and selection.path is not None:
            selected.append(str(selection.path))
            if "ML_prediction.HSX" in selection.path.name:
                forbidden.append(str(selection.path))
        else:
            missing_or_bad.append(
                {
                    "sample": sample_dir.name,
                    "reason": selection.reason,
                    "candidates": list(selection.candidates),
                }
            )
    prediction_dirs = sorted([path for path in prediction_root.iterdir() if path.is_dir()]) if prediction_root.exists() else []
    return {
        "reference_sample_dirs": len(sample_dirs),
        "prediction_sample_dirs": len(prediction_dirs),
        "selected_references": len(selected),
        "selected_reference_suffixes": sorted({Path(path).suffix for path in selected}),
        "forbidden_ml_prediction_references": forbidden,
        "missing_or_bad_references": missing_or_bad,
    }


def kgrid_summary(result_dir: Path) -> dict[str, Any]:
    structure_root = result_dir / "structures"
    structure_dirs = sorted([path for path in structure_root.iterdir() if path.is_dir()]) if structure_root.exists() else []
    meshes: set[tuple[int, int, int]] = set()
    counts: set[int] = set()
    gamma_only = 0
    errors: list[dict[str, str]] = []
    for sample_dir in structure_dirs:
        grid = parse_monkhorst_pack_kgrid(sample_dir / "RUN.fdf")
        if grid is None:
            continue
        if not grid.ok or grid.mesh is None:
            errors.append({"sample": sample_dir.name, "error": grid.error or "invalid_kgrid"})
            continue
        meshes.add(tuple(grid.mesh))
        counts.add(len(grid.fractional_kpoints))
        if grid.is_gamma_only:
            gamma_only += 1
    return {
        "structure_sample_dirs": len(structure_dirs),
        "meshes": [list(mesh) for mesh in sorted(meshes)],
        "kpoint_counts": sorted(counts),
        "gamma_only_samples": gamma_only,
        "errors": errors,
    }


def format_value(value: Any, scale: float = 1.0, digits: int = 4) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number * scale:.{digits}f}"


def report(args: argparse.Namespace) -> None:
    result_dir = args.result_dir
    gamma_dir = args.gamma_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(result_dir / "metrics" / "manifest.json")
    matrix_rows = read_csv(result_dir / "metrics" / "kpoint_matrix_metrics.csv")
    matrix_weighted = [row for row in matrix_rows if row.get("row_type") == "weighted_sample"]
    matrix_per_k = [row for row in matrix_rows if row.get("row_type") == "per_k"]
    spectral_rows = read_csv(result_dir / "metrics" / "kpoint_spectral_metrics.csv")
    dos_rows = read_csv(result_dir / "metrics" / "kpoint_dos_metrics.csv")

    summary_rows: list[dict[str, Any]] = []
    for section, rows, metrics in [
        (
            "kpoint_matrix_weighted_sample",
            matrix_weighted,
            ["h_mae_eV", "h_rmse_eV", "relative_frobenius", "hermiticity_ref", "hermiticity_pred"],
        ),
        (
            "kpoint_spectral_weighted_sample",
            spectral_rows,
            [
                "global_mae_eV",
                "global_rmse_eV",
                "low_energy_rmse_eV",
                "fermi_window_rmse_eV",
                "frontier_window_rmse_eV",
                "occupied_rmse_eV",
                "gap_abs_error_eV",
            ],
        ),
        (
            "kpoint_dos_weighted_sample",
            dos_rows,
            ["dos_wasserstein_eV", "dos_l1", "dos_l2", "dos_mae_500_fermi_window"],
        ),
    ]:
        for metric in metrics:
            summary = summarize_metric(rows, metric)
            summary_rows.append({"section": section, **summary})

    worst_samples: list[dict[str, Any]] = []
    for metric, rows in [
        ("h_rmse_eV", matrix_weighted),
        ("h_mae_eV", matrix_weighted),
        ("global_rmse_eV", spectral_rows),
        ("low_energy_rmse_eV", spectral_rows),
        ("fermi_window_rmse_eV", spectral_rows),
        ("frontier_window_rmse_eV", spectral_rows),
        ("dos_mae_500_fermi_window", dos_rows),
    ]:
        worst_samples.extend(top_rows(rows, metric, limit=5))

    worst_kpoints: list[dict[str, Any]] = []
    for metric in ["h_rmse_eV", "h_mae_eV", "relative_frobenius"]:
        worst_kpoints.extend(top_rows(matrix_per_k, metric, limit=10))

    audit = reference_audit(result_dir)
    kgrid = kgrid_summary(result_dir)
    gamma_audit = reference_audit(gamma_dir)
    gamma_kgrid = kgrid_summary(gamma_dir)
    gamma_has_predictions = gamma_audit["prediction_sample_dirs"] > 0

    write_csv(
        output_dir / "aggregate_kpoint_summary.csv",
        ["section", "metric", "count", "mean", "std", "min", "p90", "max"],
        summary_rows,
    )
    write_csv(
        output_dir / "worst_samples.csv",
        [
            "rank_metric",
            "rank_value",
            "sample",
            "k_index",
            "k_label",
            "kx",
            "ky",
            "kz",
            "h_mae_eV",
            "h_rmse_eV",
            "relative_frobenius",
            "global_rmse_eV",
            "low_energy_rmse_eV",
            "fermi_window_rmse_eV",
            "frontier_window_rmse_eV",
            "dos_mae_500_fermi_window",
        ],
        worst_samples,
    )
    write_csv(
        output_dir / "worst_kpoints.csv",
        [
            "rank_metric",
            "rank_value",
            "sample",
            "k_index",
            "k_label",
            "kx",
            "ky",
            "kz",
            "h_mae_eV",
            "h_rmse_eV",
            "relative_frobenius",
            "global_rmse_eV",
            "low_energy_rmse_eV",
            "fermi_window_rmse_eV",
            "frontier_window_rmse_eV",
            "dos_mae_500_fermi_window",
        ],
        worst_kpoints,
    )
    (output_dir / "reference_audit.json").write_text(
        json.dumps(
            {
                "kpoint_candidate": audit,
                "kpoint_grid": kgrid,
                "gamma_workaround": gamma_audit,
                "gamma_grid": gamma_kgrid,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def metric_mean(section: str, metric: str) -> str:
        row = next((item for item in summary_rows if item["section"] == section and item["metric"] == metric), None)
        return format_value(row["mean"] if row else math.nan, scale=1000.0 if metric.endswith("_eV") else 1.0)

    report_text = f"""# Graphene k-point evaluation final report

Date: 2026-05-21

## Executive summary

The original graphene 6x6x1 candidate was re-evaluated successfully with
`--enable-kpoint-metrics`; no retraining or dataset regeneration was performed.

```text
samples seen: {manifest.get("samples_seen")}
samples compared: {manifest.get("samples_compared")}
failed samples: {manifest.get("samples_failed")}
k-point mesh: {manifest.get("kpoint_mesh")}
k-point count: {manifest.get("kpoint_count")}
k-point metrics enabled: {manifest.get("kpoint_metrics_enabled")}
fatal errors: {len(manifest.get("fatal_errors") or [])}
```

The scientifically relevant result is the k-point-aware evaluation. The
gamma-only workaround is not a physically equivalent graphene benchmark and, in
this archive, has no `predicted_hamiltonians`, so it cannot be compared as a
completed model run.

## Run paths

K-point dataset:

```text
{args.dataset_dir}
```

Evaluated candidate:

```text
{result_dir}
```

Gamma-only workaround:

```text
{gamma_dir}
```

## Command used

```bash
.venv/bin/python Comparison/scripts/evaluate_hamiltonian_metrics.py \\
  {result_dir} \\
  --enable-kpoint-metrics \\
  --workers 1
```

The captured stdout/stderr are:

```text
{output_dir / "final_revaluation.stdout.json"}
{output_dir / "final_revaluation.stderr.txt"}
```

## Reference validity

```text
reference sample dirs: {audit["reference_sample_dirs"]}
prediction sample dirs: {audit["prediction_sample_dirs"]}
selected references: {audit["selected_references"]}
reference suffixes: {audit["selected_reference_suffixes"]}
forbidden ML_prediction.HSX references: {len(audit["forbidden_ml_prediction_references"])}
bad/missing references: {len(audit["missing_or_bad_references"])}
```

The selected ground truth references are real SIESTA references. No
`ML_prediction.HSX` file was selected as ground truth.

## Aggregate k-point metrics

Means over the 60 held-out samples:

| Metric | Mean |
| --- | ---: |
| weighted H(k) MAE | {metric_mean("kpoint_matrix_weighted_sample", "h_mae_eV")} meV |
| weighted H(k) RMSE | {metric_mean("kpoint_matrix_weighted_sample", "h_rmse_eV")} meV |
| weighted relative Frobenius | {metric_mean("kpoint_matrix_weighted_sample", "relative_frobenius")} |
| global spectral RMSE | {metric_mean("kpoint_spectral_weighted_sample", "global_rmse_eV")} meV |
| low-energy RMSE | {metric_mean("kpoint_spectral_weighted_sample", "low_energy_rmse_eV")} meV |
| Fermi-window RMSE | {metric_mean("kpoint_spectral_weighted_sample", "fermi_window_rmse_eV")} meV |
| frontier-window RMSE | {metric_mean("kpoint_spectral_weighted_sample", "frontier_window_rmse_eV")} meV |
| DOS 500-point Fermi-window MAE | {format_value(next((item["mean"] for item in summary_rows if item["section"] == "kpoint_dos_weighted_sample" and item["metric"] == "dos_mae_500_fermi_window"), math.nan), digits=6)} |

Full aggregate statistics are in:

```text
{output_dir / "aggregate_kpoint_summary.csv"}
```

## Worst samples and k-points

Worst-sample summaries are in:

```text
{output_dir / "worst_samples.csv"}
```

Worst per-k matrix rows are in:

```text
{output_dir / "worst_kpoints.csv"}
```

The most important ranking columns are `rank_metric`, `rank_value`, `sample`,
`k_index`, `h_rmse_eV`, `global_rmse_eV`, `low_energy_rmse_eV`,
`fermi_window_rmse_eV`, and `dos_mae_500_fermi_window`.

## Gamma-only workaround caveat

Gamma workaround path:

```text
{gamma_dir}
```

Diagnostic state:

```text
structures: {gamma_kgrid["structure_sample_dirs"]}
references: {gamma_audit["reference_sample_dirs"]}
predictions: {gamma_audit["prediction_sample_dirs"]}
mesh: {gamma_kgrid["meshes"]}
completed prediction/evaluation available: {gamma_has_predictions}
```

This is not a valid physical comparator for the 6x6x1 graphene result. It is a
debug workaround for the old gamma-only evaluator. Since it has no archived
predictions, no model metric comparison is reported.

## Output paths

K-point evaluation outputs:

```text
{result_dir / "metrics" / "kpoint_matrix_metrics.csv"}
{result_dir / "metrics" / "kpoint_spectral_metrics.csv"}
{result_dir / "metrics" / "kpoint_dos_metrics.csv"}
{result_dir / "eigenvalues" / "kpoints.csv"}
{result_dir / "eigenvalues" / "kpoint_band_errors"}
```

Diagnostic report outputs:

```text
{output_dir / "final_report.md"}
{output_dir / "aggregate_kpoint_summary.csv"}
{output_dir / "worst_samples.csv"}
{output_dir / "worst_kpoints.csv"}
{output_dir / "reference_audit.json"}
```

## Remaining risks

- K-points are reconstructed from `RUN.fdf`; no SIESTA `.KP` files are archived
  here to verify ordering/weights against SIESTA output.
- The gamma-only archive is dataset-only for this purpose and cannot serve as a
  completed metric comparator.
- These metrics are now physically meaningful for the sampled mesh, but they are
  not a high-symmetry band-path benchmark.
"""
    (output_dir / "final_report.md").write_text(report_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("Comparison/results/results_md/md_dataset1_7rff2z_graphene_default_huber_b0p01/run_20260520_173620"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("Comparison/results/results_md/MD_dataset1_7RfF2Z/run_20260520_173620"),
    )
    parser.add_argument(
        "--gamma-dir",
        type=Path,
        default=Path("Comparison/results/results_md/MD_dataset1_1vDd4u/run_20260521_093114"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Comparison/results/diagnostics/graphene_kpoint_evaluation"),
    )
    report(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
