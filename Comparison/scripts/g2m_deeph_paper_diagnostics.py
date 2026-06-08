#!/usr/bin/env python3
"""Post-process existing G2M-vs-DeepH artifacts into paper-style diagnostics.

This script is intentionally read-only with respect to benchmark computation:
it does not train, infer, run SIESTA, run Graph2Mat, run DeepH, or materialize
new Hamiltonian/eigenvalue predictions. It only reads existing CSV/JSON
artifacts and writes derived plots and summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IID600_ROOT = REPO_ROOT / "Comparison" / "results" / "g2m_deeph_iid600_phaseB_intermediate_spectral_refine_v1"
DEFAULT_IID1000_ROOT = REPO_ROOT / "Comparison" / "results" / "g2m_deeph_iid1000_phaseB_transfer_spectral_refine_v1"
DEFAULT_IID600_RUN = "paper_ready_final70_iid600_20260602_123539"
DEFAULT_IID1000_RUN = "paper_ready_final70_iid1000_20260602_123539"
DEFAULT_BAND_ROOT = REPO_ROOT / "Comparison" / "results" / "graphene_band_comparison_winners"

WINNER_CONFIGS = {
    "iid600": {"deeph": "DH-T600-13", "graph2mat": "G2M-T600-26"},
    "iid1000": {"deeph": "DH-T1000-03", "graph2mat": "G2M-T1000-03"},
}

DATASET_LABELS = {
    "graphene_w90_phase1_iid600": "iid600",
    "graphene_w90_phase1_iid1000": "iid1000",
}

FORBIDDEN_COMPUTE_COMMANDS = (
    "deeph-train",
    "deeph-preprocess",
    "deeph-inference",
    "graph2mat fit",
    "graph2mat test",
    "graph2mat predict",
    "siesta",
    "gnubands",
)

PLOT_COLORS = {"deeph": "#d62728", "graph2mat": "#1f77b4", "siesta": "#111111"}


def _configure_csv() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    _configure_csv()
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        writer.writerows([{key: json_safe(row.get(key)) for key in fieldnames} for row in rows])


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def parse_json_field(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], fraction: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = fraction * (len(clean) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return clean[lower]
    return clean[lower] * (upper - pos) + clean[upper] * (pos - lower)


def metric_from_row(row: dict[str, Any], metric: str) -> float | None:
    for key in (metric, f"{metric}_mean"):
        value = number(row.get(key))
        if value is not None:
            return value
    for field in ("final_test_metrics", "test_metrics", "validation_metrics"):
        metrics = parse_json_field(row.get(field))
        for key in (metric, f"{metric}_mean"):
            value = number(metrics.get(key))
            if value is not None:
                return value
    if metric == "low_energy_rmse_eV":
        value = number(row.get("final_test_metric_value"))
        if value is not None:
            return value
    return None


def telemetry_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return parse_json_field(row.get("telemetry"))


def metrics_dir_from_row(row: dict[str, Any]) -> Path | None:
    for key in ("final_test_metrics_path", "metrics_path", "validation_metrics_path"):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.name == "manifest.json":
            return path.parent
        if path.is_dir():
            return path
    return None


def workflow_training_metrics_path(root: Path) -> Path:
    return root / "final_test" / "sweep" / "training_sweep_metrics.csv"


def run_training_metrics_path(root: Path, run_name: str) -> Path:
    return root / "runs" / run_name / "sweep" / "training_sweep_metrics.csv"


def load_final_rows(root: Path, run_name: str) -> list[dict[str, str]]:
    final_path = workflow_training_metrics_path(root)
    if final_path.exists():
        return read_csv(final_path)
    return read_csv(run_training_metrics_path(root, run_name))


def dataset_key(dataset_id: str) -> str:
    return DATASET_LABELS.get(dataset_id, dataset_id)


def model_key(model: str) -> str:
    normalized = str(model or "").strip().lower()
    if normalized in {"g2m", "graph2mat"}:
        return "graph2mat"
    if normalized == "deeph":
        return "deeph"
    return normalized


def selected_winner_rows(rows: list[dict[str, str]], key: str) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    winners = WINNER_CONFIGS[key]
    for row in rows:
        model = model_key(row.get("model", ""))
        config = row.get("selected_config_id") or row.get("config_id") or ""
        if winners.get(model) == config:
            selected.append(row)
    return selected


def representative_rows(rows: list[dict[str, str]], metric: str = "low_energy_rmse_eV") -> dict[tuple[str, str], dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(dataset_key(row.get("dataset_id", "")), model_key(row.get("model", "")))].append(row)
    chosen: dict[tuple[str, str], dict[str, str]] = {}
    for group, items in grouped.items():
        values = [(metric_from_row(item, metric), item) for item in items]
        clean = [(value, item) for value, item in values if value is not None]
        if not clean:
            continue
        center = mean([value for value, _item in clean])
        if center is None:
            continue
        clean.sort(key=lambda item: abs(item[0] - center))
        chosen[group] = clean[0][1]
    return chosen


def load_dos_sample_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        metrics_dir = metrics_dir_from_row(row)
        if metrics_dir is None:
            continue
        path = metrics_dir / "kpoint_dos_metrics.csv"
        if not path.exists():
            path = metrics_dir / "dos_metrics.csv"
        for item in read_csv(path):
            out.append(
                {
                    "dataset_key": dataset_key(row.get("dataset_id", "")),
                    "dataset_id": row.get("dataset_id", ""),
                    "model": model_key(row.get("model", "")),
                    "selected_config_id": row.get("selected_config_id") or row.get("config_id"),
                    "seed": row.get("seed") or row.get("final_seed"),
                    "run_id": row.get("run_id"),
                    "sample": item.get("sample", ""),
                    "dos_mae_500_fermi_window": number(item.get("dos_mae_500_fermi_window")),
                    "dos_wasserstein_eV": number(item.get("dos_wasserstein_eV")),
                    "dos_l1": number(item.get("dos_l1")),
                    "dos_l2": number(item.get("dos_l2")),
                }
            )
    return [row for row in out if row["dos_mae_500_fermi_window"] is not None or row["dos_wasserstein_eV"] is not None]


def best_median_worst(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    clean = [row for row in rows if number(row.get(metric)) is not None]
    clean.sort(key=lambda row: float(row[metric]))
    if not clean:
        return []
    mid = len(clean) // 2
    picks = [("best", clean[0]), ("median", clean[mid]), ("worst", clean[-1])]
    return [{**row, "rank_label": label, "rank_metric": metric} for label, row in picks]


def aggregate_rows(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, items in sorted(grouped.items()):
        result = {key: value for key, value in zip(keys, key_values)}
        result["n"] = len(items)
        for metric in metrics:
            values = [float(item[metric]) for item in items if number(item.get(metric)) is not None]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = std(values)
            result[f"{metric}_min"] = min(values) if values else None
            result[f"{metric}_median"] = percentile(values, 0.5)
            result[f"{metric}_max"] = max(values) if values else None
        out.append(result)
    return out


def linear_regression_summary(x_values: list[float], y_values: list[float]) -> dict[str, Any]:
    pairs = [(x, y) for x, y in zip(x_values, y_values) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return {"n": len(pairs), "r2": None, "slope": None, "intercept": None, "mae": None, "rmse": None}
    xs = [item[0] for item in pairs]
    ys = [item[1] for item in pairs]
    x_mean = mean(xs) or 0.0
    y_mean = mean(ys) or 0.0
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denom if denom else 0.0
    intercept = y_mean - slope * x_mean
    pred = [slope * x + intercept for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    errors = [y - x for x, y in pairs]
    return {
        "n": len(pairs),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0,
        "slope": slope,
        "intercept": intercept,
        "mae": mean([abs(error) for error in errors]),
        "rmse": math.sqrt(mean([error * error for error in errors]) or 0.0),
    }


def load_orbital_pair_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if model_key(row.get("model", "")) != "graph2mat":
            continue
        metrics_dir = metrics_dir_from_row(row)
        if metrics_dir is None:
            continue
        path = metrics_dir / "orbital_pair_metrics.csv"
        for item in read_csv(path):
            out.append(
                {
                    "dataset_key": dataset_key(row.get("dataset_id", "")),
                    "model": "graph2mat",
                    "selected_config_id": row.get("selected_config_id") or row.get("config_id"),
                    "seed": row.get("seed") or row.get("final_seed"),
                    "sample": item.get("sample", ""),
                    "row_orbital_label": item.get("row_orbital_label", ""),
                    "col_orbital_label": item.get("col_orbital_label", ""),
                    "row_orbital_index": item.get("row_orbital_index", ""),
                    "col_orbital_index": item.get("col_orbital_index", ""),
                    "mae_union_eV": number(item.get("mae_union_eV")),
                    "mae_union_meV": number(item.get("mae_union_meV")),
                    "rmse_union_eV": number(item.get("rmse_union_eV")),
                    "r2_union": number(item.get("r2_union")),
                    "mean_abs_ref_eV": number(item.get("mean_abs_ref_eV")),
                    "mean_signed_error_eV": number(item.get("mean_signed_error_eV")),
                }
            )
    return out


def load_kpoint_matrix_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        metrics_dir = metrics_dir_from_row(row)
        if metrics_dir is None:
            continue
        path = metrics_dir / "kpoint_matrix_metrics.csv"
        for item in read_csv(path):
            if item.get("row_type") not in {"weighted_sample", "per_k", ""}:
                continue
            out.append(
                {
                    "dataset_key": dataset_key(row.get("dataset_id", "")),
                    "model": model_key(row.get("model", "")),
                    "selected_config_id": row.get("selected_config_id") or row.get("config_id"),
                    "seed": row.get("seed") or row.get("final_seed"),
                    "sample": item.get("sample", ""),
                    "row_type": item.get("row_type", ""),
                    "h_mae_eV": number(item.get("h_mae_eV")),
                    "h_rmse_eV": number(item.get("h_rmse_eV")),
                    "relative_frobenius": number(item.get("relative_frobenius")),
                    "hermiticity_pred": number(item.get("hermiticity_pred")),
                }
            )
    return out


def matrix_summary_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("row_type") in {"weighted_sample", ""}]


def load_seed_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        telemetry = telemetry_from_row(row)
        out.append(
            {
                "dataset_key": dataset_key(row.get("dataset_id", "")),
                "dataset_id": row.get("dataset_id", ""),
                "model": model_key(row.get("model", "")),
                "selected_config_id": row.get("selected_config_id") or row.get("config_id"),
                "seed": row.get("seed") or row.get("final_seed"),
                "low_energy_rmse_eV": metric_from_row(row, "low_energy_rmse_eV"),
                "fermi_window_rmse_eV": metric_from_row(row, "fermi_window_rmse_eV"),
                "frontier_window_rmse_eV": metric_from_row(row, "frontier_window_rmse_eV"),
                "global_band_rmse": metric_from_row(row, "global_band_rmse"),
                "dos_mae_500_fermi_window": metric_from_row(row, "dos_mae_500_fermi_window"),
                "dos_wasserstein_eV": metric_from_row(row, "dos_wasserstein_eV"),
                "gpu_hours_total": number(telemetry.get("gpu_hours_total")),
                "peak_gpu_memory_mb": number(telemetry.get("peak_gpu_memory_mb")),
                "inference_seconds_total": number(telemetry.get("inference_seconds_total")),
                "run_id": row.get("run_id"),
            }
        )
    return out


def latest_band_parent(band_root: Path) -> Path | None:
    parents = sorted({path.parent for path in band_root.glob("gkm_fdf_dirac_diagnostic_*/*_native_bands_ref")})
    return parents[-1] if parents else None


def band_energy_for_plot(row: dict[str, str], method: str) -> float | None:
    if method == "siesta":
        return number(row.get("energy_aligned_eV")) if row.get("energy_aligned_eV") not in (None, "") else number(row.get("energy_eV"))
    return number(row.get("energy_eV"))


def load_band_map(path: Path, method: str) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in read_csv(path):
        try:
            k_index = int(float(row.get("k_index", 0) or 0))
            band_index = int(float(row.get("band_index", 0) or 0))
        except ValueError:
            continue
        energy = band_energy_for_plot(row, method)
        if energy is None:
            continue
        out[(k_index, band_index)] = {
            "energy_eV": energy,
            "sample_id": row.get("sample_id", ""),
            "k_distance": number(row.get("k_distance")),
            "k_label": row.get("k_label", ""),
            "segment": row.get("segment", ""),
        }
    return out


def corrected_band_residual_rows_from_bands(dataset_dir: Path, dataset_key_value: str) -> list[dict[str, Any]]:
    siesta = load_band_map(dataset_dir / "bands_siesta.csv", "siesta")
    if not siesta:
        return []
    out: list[dict[str, Any]] = []
    for method, filename in [("graph2mat", "bands_graph2mat.csv"), ("deeph", "bands_deeph.csv")]:
        pred = load_band_map(dataset_dir / filename, method)
        if not pred:
            continue
        for key, pred_row in pred.items():
            ref_row = siesta.get(key)
            if ref_row is None:
                continue
            delta = float(pred_row["energy_eV"]) - float(ref_row["energy_eV"])
            k_index, band_index = key
            out.append(
                {
                    "dataset_key": dataset_key_value,
                    "model": method,
                    "sample_id": pred_row.get("sample_id", ""),
                    "k_index": k_index,
                    "k_distance": pred_row.get("k_distance"),
                    "band_index": band_index,
                    "error_eV": delta,
                    "abs_error_eV": abs(delta),
                    "k_label": pred_row.get("k_label", ""),
                    "segment": pred_row.get("segment", ""),
                    "energy_convention": "siesta_energy_aligned_vs_prediction_already_fermi_aligned",
                }
            )
    return out


def load_band_residual_rows(band_root: Path) -> list[dict[str, Any]]:
    parent = latest_band_parent(band_root)
    if parent is None:
        return []
    out: list[dict[str, Any]] = []
    for dataset_dir in sorted(parent.glob("*_native_bands_ref")):
        key = "iid1000" if "iid1000" in dataset_dir.name else "iid600" if "iid600" in dataset_dir.name else dataset_dir.name
        corrected_rows = corrected_band_residual_rows_from_bands(dataset_dir, key)
        if corrected_rows:
            out.extend(corrected_rows)
            continue
        for method, filename in [("graph2mat", "band_errors_graph2mat.csv"), ("deeph", "band_errors_deeph.csv")]:
            for row in read_csv(dataset_dir / filename):
                out.append(
                    {
                        "dataset_key": key,
                        "model": method,
                        "sample_id": row.get("sample_id", ""),
                        "k_index": int(float(row.get("k_index", 0) or 0)),
                        "k_distance": number(row.get("k_distance")),
                        "band_index": int(float(row.get("band_index", 0) or 0)),
                        "error_eV": number(row.get("error_eV")),
                        "abs_error_eV": number(row.get("abs_error_eV")),
                    }
                )
    return out


def corrected_dirac_minus_fermi(item: dict[str, Any], model: str) -> tuple[float | None, str]:
    if model != "siesta":
        value = number(item.get("dirac_energy_eV"))
        return value, "prediction_already_fermi_aligned"
    return number(item.get("dirac_minus_fermi_eV")), "raw_minus_reference_fermi"


def load_dirac_rows(band_root: Path) -> list[dict[str, Any]]:
    parent = latest_band_parent(band_root)
    if parent is None:
        return []
    out: list[dict[str, Any]] = []
    for dataset_dir in sorted(parent.glob("*_native_bands_ref")):
        key = "iid1000" if "iid1000" in dataset_dir.name else "iid600" if "iid600" in dataset_dir.name else dataset_dir.name
        data = read_json(dataset_dir / "dirac_diagnostic.json")
        methods = data.get("methods") if isinstance(data.get("methods"), dict) else {}
        for method_key, item in methods.items():
            if not isinstance(item, dict):
                continue
            model = model_key(method_key) or method_key
            dirac_minus_fermi, convention = corrected_dirac_minus_fermi(item, model)
            warnings = item.get("warnings", [])
            if isinstance(warnings, list):
                filtered_warnings = [warning for warning in warnings if "Dirac-Fermi shift" not in str(warning)]
                if dirac_minus_fermi is not None and abs(float(dirac_minus_fermi)) > 0.05:
                    filtered_warnings.append(f"Corrected Dirac-Fermi shift {float(dirac_minus_fermi) * 1000.0:.3f} meV exceeds 50.000 meV")
                warnings_text = "; ".join(filtered_warnings)
            else:
                warnings_text = ""
            out.append(
                {
                    "dataset_key": key,
                    "model": model,
                    "method_label": item.get("method", method_key),
                    "status": item.get("status"),
                    "gap_eV": number(item.get("gap_eV")),
                    "gap_meV": (number(item.get("gap_eV")) or 0.0) * 1000.0 if number(item.get("gap_eV")) is not None else None,
                    "dirac_energy_eV": number(item.get("dirac_energy_eV")),
                    "dirac_minus_fermi_eV": dirac_minus_fermi,
                    "dirac_fermi_convention": convention,
                    "fermi_level_eV": number(item.get("fermi_level_eV")),
                    "warnings": warnings_text,
                }
            )
    return out


def gate_release_rows(dataset_roots: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, root in dataset_roots.items():
        gate = read_json(root / "gate_status.json")
        release = read_json(root / "release_manifest.json")
        evidence_files = list((root / "equivalence_strict").glob("*/deeph_raw_global_equivalence_preflight.json"))
        proven = 0
        failed = 0
        for path in evidence_files:
            data = read_json(path)
            status = str(data.get("equivalence_status") or data.get("status") or "").lower()
            if status == "proven":
                proven += 1
            elif status:
                failed += 1
        rows.append(
            {
                "dataset_key": key,
                "gate_claim_status": gate.get("claim_status"),
                "gate_robust_claim_allowed": gate.get("robust_claim_allowed"),
                "gate_blockers": "; ".join(gate.get("blockers", [])) if isinstance(gate.get("blockers"), list) else "",
                "release_status": release.get("status") or release.get("strict_status"),
                "release_strict": release.get("strict") or release.get("strict_mode"),
                "equivalence_files": len(evidence_files),
                "equivalence_proven": proven,
                "equivalence_failed": failed,
            }
        )
    return rows


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def run_plot_step(
    outputs: list[str],
    warnings: list[dict[str, Any]],
    kind: str,
    callback: Any,
    *args: Any,
) -> None:
    try:
        outputs.extend(callback(*args))
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            fallback_outputs = fallback_plot(kind, *args)
            outputs.extend(fallback_outputs)
            warnings.append(
                {
                    "kind": "missing_plot_dependency",
                    "plot": kind,
                    "fallback_outputs": fallback_outputs,
                    "message": "matplotlib is not installed; generated a PIL summary card instead of the full plot.",
                }
            )
            return
        raise


def save_figure(fig: Any, output_dir: Path, stem: str, formats: list[str]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        paths.append(str(path))
    return paths


def save_pil_card(output_dir: Path, stem: str, title: str, lines: list[str], formats: list[str]) -> list[str]:
    from PIL import Image, ImageDraw, ImageFont

    output_dir.mkdir(parents=True, exist_ok=True)
    width = 1600
    line_height = 32
    margin = 48
    title_height = 58
    height = max(520, margin * 2 + title_height + line_height * (len(lines) + 2))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        text_font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
    draw.rectangle((0, 0, width, 12), fill="#1f77b4")
    draw.text((margin, margin), title, fill="#111111", font=title_font)
    y = margin + title_height
    for line in lines:
        draw.text((margin, y), line, fill="#222222", font=text_font)
        y += line_height
    paths: list[str] = []
    for fmt in formats:
        if fmt not in {"png", "pdf"}:
            continue
        path = output_dir / f"{stem}.{fmt}"
        if fmt == "pdf":
            image.save(path, "PDF", resolution=220.0)
        else:
            image.save(path)
        paths.append(str(path))
    return paths


def fallback_metric_lines(rows: list[dict[str, Any]], metrics: list[str]) -> list[str]:
    lines: list[str] = [f"Rows: {len(rows)}"]
    datasets = sorted({str(row.get("dataset_key", "")) for row in rows if row.get("dataset_key")})
    models = sorted({str(row.get("model", "")) for row in rows if row.get("model")})
    if datasets:
        lines.append(f"Datasets: {', '.join(datasets)}")
    if models:
        lines.append(f"Models: {', '.join(models)}")
    for dataset in datasets or [""]:
        for model in models or [""]:
            subset = [
                row
                for row in rows
                if (not dataset or str(row.get("dataset_key", "")) == dataset)
                and (not model or str(row.get("model", "")) == model)
            ]
            if not subset:
                continue
            label = " / ".join(item for item in [dataset, model] if item)
            lines.append(f"{label}: n={len(subset)}")
            for metric in metrics:
                values = [float(row[metric]) for row in subset if number(row.get(metric)) is not None]
                med = percentile(values, 0.5)
                avg = mean(values)
                if values and med is not None and avg is not None:
                    lines.append(
                        f"  {metric}: mean={avg:.6g}, median={med:.6g}, min={min(values):.6g}, max={max(values):.6g}"
                    )
    return lines[:30]


def fallback_plot(kind: str, *args: Any) -> list[str]:
    output_dir = next((arg for arg in args if isinstance(arg, Path)), None)
    formats = next((arg for arg in reversed(args) if isinstance(arg, list)), ["png", "pdf"])
    rows = next((arg for arg in args if isinstance(arg, list)), [])
    if output_dir is None:
        return []
    metric_map = {
        "dos_distribution": ["dos_mae_500_fermi_window", "dos_wasserstein_eV"],
        "seed_uncertainty": ["low_energy_rmse_eV", "fermi_window_rmse_eV", "dos_mae_500_fermi_window"],
        "pareto": ["gpu_hours_total", "low_energy_rmse_eV", "dos_mae_500_fermi_window"],
        "orbital_pair_heatmaps": ["mae_union_meV", "rmse_union_eV", "r2_union"],
        "matrix_metric_distribution": ["h_mae_eV", "h_rmse_eV", "relative_frobenius"],
        "band_residuals": ["error_eV", "abs_error_eV"],
        "dirac_diagnostics": ["gap_meV", "dirac_minus_fermi_eV"],
    }
    title = f"{kind.replace('_', ' ').title()} (PIL fallback summary)"
    lines = [
        "matplotlib is not installed in this Python environment.",
        "This PNG/PDF is a metric summary generated from existing artifacts only.",
    ]
    if isinstance(rows, list):
        lines.extend(fallback_metric_lines(rows, metric_map.get(kind, [])))
    return save_pil_card(output_dir, f"{kind}_summary_fallback", title, lines, formats)


def plot_dos_distribution(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> list[str]:
    plt = import_matplotlib()
    outputs: list[str] = []
    for key in sorted({row["dataset_key"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset_key"] == key]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for col, metric in enumerate(["dos_mae_500_fermi_window", "dos_wasserstein_eV"]):
            for model in ["graph2mat", "deeph"]:
                values = [float(row[metric]) for row in dataset_rows if row["model"] == model and number(row.get(metric)) is not None]
                if not values:
                    continue
                axes[0][col].hist(values, bins=24, alpha=0.45, color=PLOT_COLORS[model], label=model)
                ordered = sorted(values)
                y = [(i + 1) / len(ordered) for i in range(len(ordered))]
                axes[1][col].plot(ordered, y, color=PLOT_COLORS[model], label=model, linewidth=2)
            axes[0][col].set_title(f"{key} {metric} distribution")
            axes[0][col].set_xlabel(metric)
            axes[0][col].set_ylabel("count")
            axes[0][col].legend()
            axes[1][col].set_title(f"{key} {metric} CDF")
            axes[1][col].set_xlabel(metric)
            axes[1][col].set_ylabel("cumulative fraction")
            axes[1][col].grid(alpha=0.25)
            axes[1][col].legend()
        fig.suptitle(f"DOS diagnostics from existing final-test metrics: {key}", fontsize=14)
        outputs.extend(save_figure(fig, output_dir, f"dos_distribution_{key}", formats))
        plt.close(fig)
    return outputs


def plot_seed_uncertainty(seed_rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> list[str]:
    plt = import_matplotlib()
    metrics = ["low_energy_rmse_eV", "fermi_window_rmse_eV", "dos_mae_500_fermi_window"]
    summary = aggregate_rows(seed_rows, ["dataset_key", "model", "selected_config_id"], metrics)
    outputs: list[str] = []
    for key in sorted({row["dataset_key"] for row in summary}):
        rows = [row for row in summary if row["dataset_key"] == key]
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
        for ax, metric in zip(axes, metrics):
            labels = [f"{row['model']}\n{row['selected_config_id']}" for row in rows]
            means = [row.get(f"{metric}_mean") or math.nan for row in rows]
            errors = [row.get(f"{metric}_std") or 0.0 for row in rows]
            colors = [PLOT_COLORS.get(str(row["model"]), "#777777") for row in rows]
            ax.bar(labels, means, yerr=errors, color=colors, alpha=0.82, capsize=4)
            ax.set_title(metric)
            ax.tick_params(axis="x", labelrotation=35)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Seed uncertainty across final runs: {key}", fontsize=14)
        outputs.extend(save_figure(fig, output_dir, f"seed_uncertainty_{key}", formats))
        plt.close(fig)
    return outputs


def plot_pareto(seed_rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> list[str]:
    plt = import_matplotlib()
    summary = aggregate_rows(
        seed_rows,
        ["dataset_key", "model", "selected_config_id"],
        ["low_energy_rmse_eV", "dos_mae_500_fermi_window", "gpu_hours_total"],
    )
    outputs: list[str] = []
    for key in sorted({row["dataset_key"] for row in summary}):
        rows = [row for row in summary if row["dataset_key"] == key]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for row in rows:
            model = str(row["model"])
            x = row.get("gpu_hours_total_mean")
            if x is None:
                continue
            axes[0].scatter(x, row.get("low_energy_rmse_eV_mean"), color=PLOT_COLORS.get(model, "#777777"), s=70)
            axes[1].scatter(x, row.get("dos_mae_500_fermi_window_mean"), color=PLOT_COLORS.get(model, "#777777"), s=70)
            for ax in axes:
                ax.annotate(str(row["selected_config_id"]), (x, ax.collections[-1].get_offsets()[-1][1]), fontsize=8)
        axes[0].set_title("Low-energy RMSE vs GPU-hours")
        axes[0].set_xlabel("GPU-hours mean per seed")
        axes[0].set_ylabel("low_energy_rmse_eV")
        axes[1].set_title("DOS MAE vs GPU-hours")
        axes[1].set_xlabel("GPU-hours mean per seed")
        axes[1].set_ylabel("dos_mae_500_fermi_window")
        for ax in axes:
            add_pareto_diagonal_guides(ax)
            ax.grid(alpha=0.25)
        fig.suptitle(f"Accuracy/cost Pareto from stored telemetry: {key}", fontsize=14)
        outputs.extend(save_figure(fig, output_dir, f"pareto_accuracy_cost_{key}", formats))
        plt.close(fig)
    return outputs


def add_pareto_diagonal_guides(ax: Any) -> None:
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    x_upper = ax.get_xlim()[1]
    y_upper = ax.get_ylim()[1]
    if x_upper <= 0.0 or y_upper <= 0.0:
        return
    for xs, ys in [([0.0, x_upper], [0.0, y_upper]), ([0.0, x_upper], [y_upper, 0.0])]:
        ax.plot(
            xs,
            ys,
            color="#8c8c8c",
            linestyle="--",
            linewidth=1.2,
            alpha=0.55,
            zorder=0,
        )


def plot_orbital_pair_heatmaps(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> list[str]:
    plt = import_matplotlib()
    outputs: list[str] = []
    for key in sorted({row["dataset_key"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset_key"] == key and number(row.get("mae_union_meV")) is not None]
        if not dataset_rows:
            continue
        labels = sorted({row["row_orbital_label"] for row in dataset_rows} | {row["col_orbital_label"] for row in dataset_rows})
        if not labels:
            continue
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in dataset_rows:
            grouped[(row["row_orbital_label"], row["col_orbital_label"])].append(float(row["mae_union_meV"]))
        matrix = []
        for r_label in labels:
            matrix_row = []
            for c_label in labels:
                matrix_row.append(mean(grouped.get((r_label, c_label), [])) or math.nan)
            matrix.append(matrix_row)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        image = ax.imshow(matrix, cmap="magma")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(f"Graph2Mat orbital-pair MAE ({key})")
        ax.set_xlabel("column orbital")
        ax.set_ylabel("row orbital")
        fig.colorbar(image, ax=ax, label="MAE meV")
        outputs.extend(save_figure(fig, output_dir, f"graph2mat_orbital_pair_mae_{key}", formats))
        plt.close(fig)
    return outputs


def plot_matrix_metric_distribution(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> list[str]:
    plt = import_matplotlib()
    outputs: list[str] = []
    sample_rows = [row for row in rows if row.get("row_type") in {"weighted_sample", ""}]
    for key in sorted({row["dataset_key"] for row in sample_rows}):
        dataset_rows = [row for row in sample_rows if row["dataset_key"] == key]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for ax, metric in zip(axes, ["h_mae_eV", "h_rmse_eV", "relative_frobenius"]):
            for model in ["graph2mat", "deeph"]:
                values = [float(row[metric]) for row in dataset_rows if row["model"] == model and number(row.get(metric)) is not None]
                if values:
                    ax.hist(values, bins=24, alpha=0.45, color=PLOT_COLORS[model], label=model)
            ax.set_title(metric)
            ax.set_xlabel(metric)
            ax.set_ylabel("count")
            ax.legend()
        fig.suptitle(f"Hamiltonian matrix metric distributions: {key}", fontsize=14)
        outputs.extend(save_figure(fig, output_dir, f"matrix_metric_distribution_{key}", formats))
        plt.close(fig)
    return outputs


def plot_band_residuals(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> list[str]:
    plt = import_matplotlib()
    from matplotlib.colors import TwoSlopeNorm

    outputs: list[str] = []
    for key in sorted({row["dataset_key"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset_key"] == key and number(row.get("error_eV")) is not None]
        if not dataset_rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for ax, model in zip(axes, ["graph2mat", "deeph"]):
            model_rows = [row for row in dataset_rows if row["model"] == model]
            if not model_rows:
                ax.set_axis_off()
                continue
            max_k = max(row["k_index"] for row in model_rows)
            max_band = max(row["band_index"] for row in model_rows)
            matrix = [[math.nan for _ in range(max_k + 1)] for _ in range(max_band + 1)]
            for row in model_rows:
                matrix[row["band_index"]][row["k_index"]] = float(row["error_eV"])
            finite = [abs(float(row["error_eV"])) for row in model_rows]
            vmax = percentile(finite, 0.98) or max(finite)
            image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
            ax.set_title(model)
            ax.set_xlabel("k index")
            ax.set_ylabel("band index")
            fig.colorbar(image, ax=ax, label="E_pred - E_SIESTA (eV)")
        fig.suptitle(f"Band residuals from existing band comparison CSVs: {key}", fontsize=14)
        outputs.extend(save_figure(fig, output_dir, f"band_residuals_{key}", formats))
        plt.close(fig)
    return outputs


def plot_dirac(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> list[str]:
    plt = import_matplotlib()
    outputs: list[str] = []
    for key in sorted({row["dataset_key"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset_key"] == key]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        labels = [str(row["method_label"]) for row in dataset_rows]
        colors = [PLOT_COLORS.get(str(row["model"]), "#111111") for row in dataset_rows]
        axes[0].bar(labels, [row.get("gap_meV") or math.nan for row in dataset_rows], color=colors, alpha=0.85)
        axes[0].set_title("K-point gap")
        axes[0].set_ylabel("gap meV")
        axes[1].bar(labels, [(row.get("dirac_minus_fermi_eV") or math.nan) * 1000.0 for row in dataset_rows], color=colors, alpha=0.85)
        axes[1].set_title("Dirac - Fermi")
        axes[1].set_ylabel("meV")
        for ax in axes:
            ax.tick_params(axis="x", labelrotation=25)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Dirac diagnostics from existing band artifacts: {key}", fontsize=14)
        outputs.extend(save_figure(fig, output_dir, f"dirac_diagnostics_{key}", formats))
        plt.close(fig)
    return outputs


def summarize_outputs(output_dir: Path, outputs: list[str], warnings: list[dict[str, Any]], rows: dict[str, list[dict[str, Any]]]) -> None:
    lines = [
        "# G2M-vs-DeepH Paper Diagnostics",
        "",
        "Post-processing only: no training, inference, SIESTA, Graph2Mat, DeepH, or new eigenvalue computation was run.",
        "",
        "## Generated Figures",
    ]
    for path in outputs:
        lines.append(f"- `{Path(path).relative_to(REPO_ROOT) if Path(path).is_absolute() and REPO_ROOT in Path(path).parents else path}`")
    lines.extend(["", "## Tables"])
    for name in sorted(rows):
        lines.append(f"- `{name}`: {len(rows[name])} rows")
    if warnings:
        lines.extend(["", "## Warnings"])
        for warning in warnings:
            lines.append(f"- `{warning.get('kind')}`: {warning.get('message')}")
    (output_dir / "paper_diagnostics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]

    dataset_roots = {"iid600": args.iid600_root, "iid1000": args.iid1000_root}
    run_names = {"iid600": args.run_iid600, "iid1000": args.run_iid1000}
    warnings: list[dict[str, Any]] = []
    outputs: list[str] = []

    all_rows: list[dict[str, str]] = []
    for key, root in dataset_roots.items():
        rows = load_final_rows(root, run_names[key])
        if not rows:
            warnings.append({"kind": "missing_artifact", "dataset_key": key, "message": f"No final training metrics found under {root}"})
        all_rows.extend(selected_winner_rows(rows, key))

    if not all_rows:
        raise RuntimeError("No winner rows were found in existing final metrics.")

    seed_rows = load_seed_rows(all_rows)
    dos_rows = load_dos_sample_metrics(all_rows)
    dos_summary_rows = aggregate_rows(dos_rows, ["dataset_key", "model", "selected_config_id"], ["dos_mae_500_fermi_window", "dos_wasserstein_eV"])
    dos_rank_rows: list[dict[str, Any]] = []
    for key in sorted({row["dataset_key"] for row in dos_rows}):
        for model in ["graph2mat", "deeph"]:
            model_rows = [row for row in dos_rows if row["dataset_key"] == key and row["model"] == model]
            dos_rank_rows.extend(best_median_worst(model_rows, "dos_mae_500_fermi_window"))

    orbital_rows = load_orbital_pair_rows(all_rows)
    orbital_summary_rows = aggregate_rows(
        orbital_rows,
        ["dataset_key", "model", "selected_config_id", "row_orbital_label", "col_orbital_label"],
        ["mae_union_meV", "rmse_union_eV", "r2_union", "mean_abs_ref_eV"],
    )
    if not orbital_rows:
        warnings.append(
            {
                "kind": "missing_artifact",
                "message": (
                    "No non-empty orbital_pair_metrics.csv rows were found; orbital-pair heatmaps were skipped. "
                    "Graph2Mat source manifests report missing .ion.xml basis files, and DeepH has no orbital_pair_metrics.csv."
                ),
            }
        )
    elif not any(row.get("model") == "deeph" for row in orbital_rows):
        warnings.append(
            {
                "kind": "missing_artifact",
                "message": "DeepH orbital_pair_metrics.csv was not found; orbital-pair heatmaps are Graph2Mat-only.",
            }
        )

    matrix_rows = load_kpoint_matrix_rows(all_rows)
    matrix_summary_rows = aggregate_rows(
        matrix_summary_source_rows(matrix_rows),
        ["dataset_key", "model", "selected_config_id"],
        ["h_mae_eV", "h_rmse_eV", "relative_frobenius", "hermiticity_pred"],
    )
    band_rows = load_band_residual_rows(args.band_root)
    if not band_rows:
        warnings.append({"kind": "missing_artifact", "message": f"No existing band residual CSVs found under {args.band_root}"})
    dirac_rows = load_dirac_rows(args.band_root)
    if not dirac_rows:
        warnings.append({"kind": "missing_artifact", "message": f"No existing dirac_diagnostic.json files found under {args.band_root}"})
    gate_rows = gate_release_rows(dataset_roots)

    table_rows = {
        "seed_metrics.csv": seed_rows,
        "dos_sample_metrics.csv": dos_rows,
        "dos_summary.csv": dos_summary_rows,
        "dos_best_median_worst.csv": dos_rank_rows,
        "orbital_pair_summary.csv": orbital_summary_rows,
        "matrix_metric_summary.csv": matrix_summary_rows,
        "band_residuals.csv": band_rows,
        "dirac_diagnostics.csv": dirac_rows,
        "equivalence_gate_release_table.csv": gate_rows,
    }
    for filename, rows in table_rows.items():
        write_csv(output_dir / filename, rows)

    if dos_rows:
        run_plot_step(outputs, warnings, "dos_distribution", plot_dos_distribution, dos_rows, output_dir, formats)
    else:
        warnings.append({"kind": "missing_artifact", "message": "No DOS metric rows found; DOS distribution plots were skipped."})
    run_plot_step(outputs, warnings, "seed_uncertainty", plot_seed_uncertainty, seed_rows, output_dir, formats)
    run_plot_step(outputs, warnings, "pareto", plot_pareto, seed_rows, output_dir, formats)
    if orbital_rows:
        run_plot_step(outputs, warnings, "orbital_pair_heatmaps", plot_orbital_pair_heatmaps, orbital_rows, output_dir, formats)
    if matrix_rows:
        run_plot_step(outputs, warnings, "matrix_metric_distribution", plot_matrix_metric_distribution, matrix_rows, output_dir, formats)
    if band_rows:
        run_plot_step(outputs, warnings, "band_residuals", plot_band_residuals, band_rows, output_dir, formats)
    if dirac_rows:
        run_plot_step(outputs, warnings, "dirac_diagnostics", plot_dirac, dirac_rows, output_dir, formats)

    if not any("dos_curve" in row for row in dos_rows):
        warnings.append(
            {
                "kind": "missing_artifact",
                "message": "No stored full DOS curves were discovered; best/median/worst DOS overlays were not generated.",
            }
        )

    manifest = {
        "script": "Comparison/scripts/g2m_deeph_paper_diagnostics.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "post_processing_only": True,
        "strict_non_compute_rules": {
            "no_training": True,
            "no_inference": True,
            "no_siesta": True,
            "no_graph2mat_cli": True,
            "no_deeph_cli": True,
            "forbidden_compute_commands": list(FORBIDDEN_COMPUTE_COMMANDS),
        },
        "inputs": {
            "iid600_root": str(args.iid600_root),
            "iid1000_root": str(args.iid1000_root),
            "run_iid600": args.run_iid600,
            "run_iid1000": args.run_iid1000,
            "band_root": str(args.band_root),
            "winner_configs": WINNER_CONFIGS,
        },
        "outputs": outputs,
        "tables": {name: str(output_dir / name) for name in table_rows},
        "warnings": warnings,
        "status": "ok" if outputs else "tables_only",
    }
    write_json(output_dir / "paper_diagnostics_manifest.json", manifest)
    summarize_outputs(output_dir, outputs, warnings, table_rows)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iid600-root", type=Path, default=DEFAULT_IID600_ROOT)
    parser.add_argument("--iid1000-root", type=Path, default=DEFAULT_IID1000_ROOT)
    parser.add_argument("--run-iid600", default=DEFAULT_IID600_RUN)
    parser.add_argument("--run-iid1000", default=DEFAULT_IID1000_RUN)
    parser.add_argument("--band-root", type=Path, default=DEFAULT_BAND_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "Comparison" / "results" / f"g2m_deeph_paper_diagnostics_{timestamp}",
    )
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated figure formats.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_diagnostics(args)
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir), "figures": len(manifest["outputs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
