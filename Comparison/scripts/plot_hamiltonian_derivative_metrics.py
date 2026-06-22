#!/usr/bin/env python3
"""Build diagnostic plot payloads from derivative metric CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


PLOT_SCHEMA = "hamiltonian_derivative_plot_payload_v1"
MANIFEST_SCHEMA = "hamiltonian_derivative_plot_manifest_v1"
TITLE = "Hamiltonian derivative diagnostics"
REFERENCE_LABEL = "Reference: finite differences of SIESTA Hamiltonians"
FORCE_CONSTANTS_LABEL = "SIESTA force constants are not treated as dH/dR"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _model_label(name: str) -> str:
    normalized = str(name or "").strip().lower()
    if normalized == "graph2mat":
        return "Graph2Mat"
    if normalized == "deeph":
        return "DeepH"
    return normalized or "Unknown"


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return sum(clean) / len(clean) if clean else None


def _sorted_unique(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _infer_model_from_rows(rows: list[dict[str, str]], fallback: str) -> str:
    for row in rows:
        source_model = str(row.get("source_model") or "").strip()
        if source_model:
            return source_model.lower()
    stem = fallback.lower()
    if "graph2mat" in stem or "g2m" in stem:
        return "graph2mat"
    if "deeph" in stem:
        return "deeph"
    return fallback.lower()


def _normalize_root(root: Path) -> Path:
    root = Path(root)
    if (root / "manifest.json").exists():
        return root
    return root if root.name == "derivative_metrics" else root / "derivative_metrics"


def _warning(code: str, message: str, *, severity: str = "warning", **details: Any) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "details": {str(key): json_safe(value) for key, value in details.items()},
    }


def load_derivative_root(root: Path, *, forced_model: str | None = None) -> dict[str, Any]:
    derivative_root = _normalize_root(root)
    manifest = read_json(derivative_root / "manifest.json")
    metric_rows = read_csv_rows(derivative_root / "derivative_matrix_metrics.csv")
    hermiticity_rows = read_csv_rows(derivative_root / "derivative_hermiticity.csv")
    stencil_rows = read_csv_rows(derivative_root / "stencil_status.csv")
    model = (forced_model or _infer_model_from_rows(metric_rows, derivative_root.parent.name or derivative_root.name)).lower()
    warnings: list[dict[str, Any]] = []
    scientific_status = str(manifest.get("scientific_status") or "diagnostic_only")
    if scientific_status != "presentation_ready":
        warnings.append(
            _warning(
                "scientific_status_diagnostic",
                "Derivative diagnostics are not in presentation-ready status.",
                scientific_status=scientific_status,
            )
        )
    if manifest.get("fatal_errors"):
        warnings.append(
            _warning(
                "fatal_errors_present",
                "Derivative metric evaluation reported fatal errors; plots are partial diagnostics.",
                severity="severe",
                count=len(manifest.get("fatal_errors") or []),
            )
        )
    if int(manifest.get("stencils_failed") or 0) > 0:
        warnings.append(
            _warning(
                "failed_stencils_present",
                "Some finite-difference stencils failed and are excluded from metric plots.",
                stencils_failed=int(manifest.get("stencils_failed") or 0),
            )
        )
    if not metric_rows:
        warnings.append(
            _warning(
                "no_metric_rows",
                "No derivative matrix metric rows are available; the payload is diagnostic metadata only.",
            )
        )
    return {
        "root": derivative_root,
        "model": model,
        "model_label": _model_label(model),
        "manifest": manifest,
        "metric_rows": metric_rows,
        "hermiticity_rows": hermiticity_rows,
        "stencil_rows": stencil_rows,
        "warnings": warnings,
    }


def _combined_rows(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        model = dataset["model"]
        model_label = dataset["model_label"]
        for row in dataset["metric_rows"]:
            rows.append(
                {
                    "model": model,
                    "model_label": model_label,
                    "sample": str(row.get("sample") or ""),
                    "atom_index_zero_based": row.get("atom_index_zero_based"),
                    "axis": str(row.get("axis") or ""),
                    "delta_ang": number(row.get("delta_ang")),
                    "finite_difference_method": str(row.get("finite_difference_method") or ""),
                    "dh_mae_union_eV_per_Ang": number(row.get("dh_mae_union_eV_per_Ang")),
                    "dh_rmse_union_eV_per_Ang": number(row.get("dh_rmse_union_eV_per_Ang")),
                    "dh_relative_frobenius_ref": number(row.get("dh_relative_frobenius_ref")),
                    "dh_false_zero_rate": number(row.get("dh_false_zero_rate")),
                    "dh_false_nonzero_rate": number(row.get("dh_false_nonzero_rate")),
                    "dh_support_f1": number(row.get("dh_support_f1")),
                    "dh_support_changed": _bool(row.get("dh_support_changed"))
                    or bool(number(row.get("dh_false_zero_rate")) not in (None, 0.0))
                    or bool(number(row.get("dh_false_nonzero_rate")) not in (None, 0.0)),
                }
            )
    return rows


def _combined_hermiticity_rows(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        model = dataset["model"]
        model_label = dataset["model_label"]
        for row in dataset["hermiticity_rows"]:
            rows.append(
                {
                    "model": model,
                    "model_label": model_label,
                    "sample": str(row.get("sample") or ""),
                    "finite_difference_method": str(row.get("finite_difference_method") or ""),
                    "dH_ref_hermiticity_defect": number(row.get("dH_ref_hermiticity_defect")),
                    "dH_pred_hermiticity_defect": number(row.get("dH_pred_hermiticity_defect")),
                    "dH_hermiticity_error_delta": number(row.get("dH_hermiticity_error_delta")),
                }
            )
    return rows


def _plot_common_metadata(methods_seen: list[str]) -> dict[str, Any]:
    return {
        "title": TITLE,
        "reference_label": REFERENCE_LABEL,
        "force_constants_label": FORCE_CONSTANTS_LABEL,
        "diagnostic_only": True,
        "units": {"derivative": "eV/Ang", "delta": "Ang"},
        "methods_seen": methods_seen,
        "method_label": "Method: " + ", ".join(methods_seen or ["unknown"]),
    }


def _grouped_bar_plot(
    plot_id: str,
    title: str,
    metrics: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": plot_id,
        "kind": "grouped_bar",
        "title": title,
        "subtitle": TITLE,
        "reference_label": REFERENCE_LABEL,
        "force_constants_label": FORCE_CONSTANTS_LABEL,
        "rows": rows,
        "metrics": metrics,
        "diagnostic_only": True,
    }


def _scatter_plot(
    plot_id: str,
    title: str,
    x_title: str,
    y_title: str,
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
) -> dict[str, Any]:
    return {
        "id": plot_id,
        "kind": "scatter",
        "title": title,
        "subtitle": TITLE,
        "reference_label": REFERENCE_LABEL,
        "force_constants_label": FORCE_CONSTANTS_LABEL,
        "x_title": x_title,
        "y_title": y_title,
        "x_key": "x",
        "y_key": metric_key,
        "rows": rows,
        "diagnostic_only": True,
    }


def _aggregate_by_model(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[float | None]] = {}
    labels: dict[str, str] = {}
    for row in rows:
        model = str(row.get("model") or "")
        labels[model] = str(row.get("model_label") or model)
        buckets.setdefault(model, []).append(number(row.get(key)))
    results: list[dict[str, Any]] = []
    for model in sorted(buckets):
        results.append({"method": labels[model], key: _mean(buckets[model])})
    return results


def _aggregate_model_metrics(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, str] = {}
    for row in rows:
        model = str(row.get("model") or "")
        labels[model] = str(row.get("model_label") or model)
        buckets.setdefault(model, []).append(row)
    results: list[dict[str, Any]] = []
    for model in sorted(buckets):
        group = buckets[model]
        aggregated = {"method": labels[model]}
        for key in keys:
            aggregated[key] = _mean([number(item.get(key)) for item in group])
        results.append(aggregated)
    return results


def _aggregate_axis(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[float | None]] = {}
    for row in rows:
        axis = str(row.get("axis") or "")
        model_label = str(row.get("model_label") or row.get("model") or "")
        if not axis:
            continue
        buckets.setdefault((axis, model_label), []).append(number(row.get(key)))
    results: list[dict[str, Any]] = []
    for (axis, model_label), values in sorted(buckets.items()):
        results.append({"axis": axis, "method": model_label, key: _mean(values)})
    return results


def _aggregate_atom(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, str], list[float | None]] = {}
    for row in rows:
        atom = number(row.get("atom_index_zero_based"))
        if atom is None:
            continue
        model_label = str(row.get("model_label") or row.get("model") or "")
        buckets.setdefault((int(atom), model_label), []).append(number(row.get(key)))
    results: list[dict[str, Any]] = []
    for (atom, model_label), values in sorted(buckets.items()):
        results.append({"atom_index_zero_based": atom, "method": model_label, key: _mean(values)})
    return results


def _support_diagnostic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, str] = {}
    for row in rows:
        model = str(row.get("model") or "")
        labels[model] = str(row.get("model_label") or model)
        buckets.setdefault(model, []).append(row)
    results: list[dict[str, Any]] = []
    for model in sorted(buckets):
        group = buckets[model]
        support_change_fraction = _mean([1.0 if _bool(item.get("dh_support_changed")) else 0.0 for item in group])
        results.append(
            {
                "method": labels[model],
                "support_change_fraction": support_change_fraction,
                "dh_false_zero_rate": _mean([number(item.get("dh_false_zero_rate")) for item in group]),
                "dh_false_nonzero_rate": _mean([number(item.get("dh_false_nonzero_rate")) for item in group]),
            }
        )
    return results


def _error_vs_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        delta = number(row.get("delta_ang"))
        mae = number(row.get("dh_mae_union_eV_per_Ang"))
        if delta is None or mae is None:
            continue
        result.append(
            {
                "method": row.get("model_label"),
                "sample": row.get("sample"),
                "x": delta,
                "dh_mae_union_eV_per_Ang": mae,
                "finite_difference_method": row.get("finite_difference_method"),
            }
        )
    return result


def _paired_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, Any, str, Any, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("sample") or ""),
            row.get("atom_index_zero_based"),
            str(row.get("axis") or ""),
            row.get("delta_ang"),
            str(row.get("finite_difference_method") or ""),
        )
        model = str(row.get("model") or "")
        keyed.setdefault(key, {})[model] = row
    results: list[dict[str, Any]] = []
    for key, models in sorted(keyed.items()):
        if "graph2mat" not in models or "deeph" not in models:
            continue
        graph2mat_row = models["graph2mat"]
        deeph_row = models["deeph"]
        results.append(
            {
                "sample": key[0],
                "atom_index_zero_based": key[1],
                "axis": key[2],
                "delta_ang": key[3],
                "finite_difference_method": key[4],
                "graph2mat_dh_mae_union_eV_per_Ang": number(graph2mat_row.get("dh_mae_union_eV_per_Ang")),
                "deeph_dh_mae_union_eV_per_Ang": number(deeph_row.get("dh_mae_union_eV_per_Ang")),
                "graph2mat_dh_rmse_union_eV_per_Ang": number(graph2mat_row.get("dh_rmse_union_eV_per_Ang")),
                "deeph_dh_rmse_union_eV_per_Ang": number(deeph_row.get("dh_rmse_union_eV_per_Ang")),
            }
        )
    return results


def _scientific_warnings(datasets: list[dict[str, Any]], metric_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for dataset in datasets:
        warnings.extend(dataset["warnings"])
    if len(_sorted_unique([row.get("delta_ang") for row in metric_rows if row.get("delta_ang") is not None])) <= 1:
        warnings.append(
            _warning(
                "single_delta_only",
                "Only one displacement delta is available; error-vs-delta is limited to a single-x diagnostic.",
            )
        )
    if not paired_rows and len({dataset["model"] for dataset in datasets}) < 2:
        warnings.append(
            _warning(
                "paired_comparison_unavailable",
                "Graph2Mat vs DeepH paired comparison is unavailable because only one model root was provided.",
            )
        )
    elif not paired_rows:
        warnings.append(
            _warning(
                "paired_comparison_unmatched",
                "Graph2Mat and DeepH roots were provided, but no shared derivative stencil keys matched for paired comparison.",
            )
        )
    return warnings


def build_derivative_plot_payload(
    *,
    derivative_roots: list[Path],
    graph2mat_root: Path | None = None,
    deeph_root: Path | None = None,
) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    if graph2mat_root is not None:
        datasets.append(load_derivative_root(graph2mat_root, forced_model="graph2mat"))
    if deeph_root is not None:
        datasets.append(load_derivative_root(deeph_root, forced_model="deeph"))
    for root in derivative_roots:
        datasets.append(load_derivative_root(root))
    metric_rows = _combined_rows(datasets)
    hermiticity_rows = _combined_hermiticity_rows(datasets)
    paired_rows = _paired_comparison_rows(metric_rows)
    methods_seen = _sorted_unique(
        sorted(
            {
                str(row.get("finite_difference_method") or "")
                for row in metric_rows
                if str(row.get("finite_difference_method") or "")
            }
        )
    )
    meta = _plot_common_metadata(methods_seen)

    plots = [
        _scatter_plot(
            "error_vs_delta",
            "Error vs delta",
            "delta Ang",
            "dH MAE eV/Ang",
            _error_vs_delta_rows(metric_rows),
            metric_key="dh_mae_union_eV_per_Ang",
        ),
        _grouped_bar_plot(
            "dh_mae_by_model",
            "dH MAE by model",
            [{"key": "dh_mae_union_eV_per_Ang", "label": "dH MAE", "unit": "eV/Ang"}],
            _aggregate_by_model(metric_rows, "dh_mae_union_eV_per_Ang"),
        ),
        _grouped_bar_plot(
            "dh_rmse_by_model",
            "dH RMSE by model",
            [{"key": "dh_rmse_union_eV_per_Ang", "label": "dH RMSE", "unit": "eV/Ang"}],
            _aggregate_by_model(metric_rows, "dh_rmse_union_eV_per_Ang"),
        ),
        _grouped_bar_plot(
            "relative_frobenius_by_model",
            "Relative Frobenius by model",
            [{"key": "dh_relative_frobenius_ref", "label": "Relative Frobenius", "unit": ""}],
            _aggregate_by_model(metric_rows, "dh_relative_frobenius_ref"),
        ),
        _grouped_bar_plot(
            "error_by_atom_index_zero_based",
            "Error by atom index",
            [{"key": "dh_mae_union_eV_per_Ang", "label": "dH MAE", "unit": "eV/Ang"}],
            _aggregate_atom(metric_rows, "dh_mae_union_eV_per_Ang"),
        ),
        _grouped_bar_plot(
            "error_by_axis",
            "Error by axis",
            [{"key": "dh_mae_union_eV_per_Ang", "label": "dH MAE", "unit": "eV/Ang"}],
            _aggregate_axis(metric_rows, "dh_mae_union_eV_per_Ang"),
        ),
        _grouped_bar_plot(
            "hermiticity_defect_by_model",
            "Hermiticity defect by model",
            [
                {"key": "dH_pred_hermiticity_defect", "label": "Predicted defect", "unit": ""},
                {"key": "dH_hermiticity_error_delta", "label": "Error delta", "unit": ""},
            ],
            _aggregate_model_metrics(
                hermiticity_rows,
                ["dH_pred_hermiticity_defect", "dH_hermiticity_error_delta"],
            ),
        ),
        _grouped_bar_plot(
            "support_change_false_zero_false_nonzero",
            "Support-change and false-zero diagnostics",
            [
                {"key": "support_change_fraction", "label": "Support change fraction", "unit": ""},
                {"key": "dh_false_zero_rate", "label": "False-zero rate", "unit": ""},
                {"key": "dh_false_nonzero_rate", "label": "False-nonzero rate", "unit": ""},
            ],
            _support_diagnostic_rows(metric_rows),
        ),
        _scatter_plot(
            "graph2mat_vs_deeph_paired_comparison",
            "Graph2Mat vs DeepH paired comparison",
            "Graph2Mat dH MAE eV/Ang",
            "DeepH dH MAE eV/Ang",
            [
                {
                    "sample": row["sample"],
                    "atom_index_zero_based": row["atom_index_zero_based"],
                    "axis": row["axis"],
                    "delta_ang": row["delta_ang"],
                    "finite_difference_method": row["finite_difference_method"],
                    "x": row["graph2mat_dh_mae_union_eV_per_Ang"],
                    "deeph_dh_mae_union_eV_per_Ang": row["deeph_dh_mae_union_eV_per_Ang"],
                }
                for row in paired_rows
            ],
            metric_key="deeph_dh_mae_union_eV_per_Ang",
        ),
    ]
    warnings = _scientific_warnings(datasets, metric_rows, paired_rows)
    return {
        "schema": PLOT_SCHEMA,
        "available": bool(metric_rows or hermiticity_rows),
        "diagnostic_only": True,
        "scientific_status": "diagnostic_only",
        "title": TITLE,
        "reference_label": REFERENCE_LABEL,
        "force_constants_label": FORCE_CONSTANTS_LABEL,
        "units": {"derivative": "eV/Ang", "delta": "Ang"},
        "methods_seen": methods_seen,
        "method_label": meta["method_label"],
        "plots": plots,
        "scientific_warnings": warnings,
        "inputs": {
            "roots": [str(dataset["root"]) for dataset in datasets],
            "models": [dataset["model_label"] for dataset in datasets],
        },
        "summary": {
            "metric_rows": len(metric_rows),
            "hermiticity_rows": len(hermiticity_rows),
            "paired_rows": len(paired_rows),
        },
    }


def write_derivative_plot_outputs(
    *,
    derivative_roots: list[Path],
    output_dir: Path,
    graph2mat_root: Path | None = None,
    deeph_root: Path | None = None,
) -> dict[str, Any]:
    payload = build_derivative_plot_payload(
        derivative_roots=derivative_roots,
        graph2mat_root=graph2mat_root,
        deeph_root=deeph_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "derivative_plot_payload.json"
    manifest_path = output_dir / "derivative_plot_manifest.json"
    write_json(payload_path, payload)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "title": TITLE,
        "reference_label": REFERENCE_LABEL,
        "force_constants_label": FORCE_CONSTANTS_LABEL,
        "diagnostic_only": True,
        "plot_payload": str(payload_path),
        "outputs": {"plot_payload": str(payload_path)},
        "scientific_warnings": payload.get("scientific_warnings") or [],
        "available": bool(payload.get("available")),
    }
    write_json(manifest_path, manifest)
    return {
        "payload": payload,
        "manifest": manifest,
        "payload_path": payload_path,
        "manifest_path": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derivative-root", action="append", default=[])
    parser.add_argument("--graph2mat-root", type=Path, default=None)
    parser.add_argument("--deeph-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    derivative_roots = [Path(item) for item in args.derivative_root]
    if not derivative_roots and args.graph2mat_root is None and args.deeph_root is None:
        raise SystemExit("Provide at least one derivative root or a Graph2Mat/DeepH pair.")
    result = write_derivative_plot_outputs(
        derivative_roots=derivative_roots,
        graph2mat_root=args.graph2mat_root,
        deeph_root=args.deeph_root,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "plot_payload": str(result["payload_path"]),
                "plot_manifest": str(result["manifest_path"]),
                "available": bool(result["payload"].get("available")),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
