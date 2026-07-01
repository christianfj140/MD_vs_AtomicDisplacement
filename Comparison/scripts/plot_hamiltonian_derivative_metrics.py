#!/usr/bin/env python3
"""Build diagnostic plot payloads from derivative metric CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any


PLOT_SCHEMA = "hamiltonian_derivative_plot_payload_v1"
MANIFEST_SCHEMA = "hamiltonian_derivative_plot_manifest_v1"
TITLE = "Hamiltonian derivative diagnostics"
REFERENCE_LABEL = "Reference: finite differences of SIESTA Hamiltonians"
FORCE_CONSTANTS_LABEL = "SIESTA force constants are not treated as dH/dR"
MAX_DATASET_SIZE_ATOM_SERIES = 12
DATASET_SIZE_METRICS = [
    "dh_mae_union_eV_per_Ang",
    "dh_rmse_union_eV_per_Ang",
    "dh_relative_frobenius_ref",
    "dh_signal_to_noise_ratio",
    "dh_signal_over_abs_h_ref",
    "dh_abs_h_pred_rel_error_ref",
    "dh_support_f1",
    "dh_false_zero_rate",
    "dh_false_nonzero_rate",
    "dh_relative_frobenius_union_robust",
    "dh_relative_l1_union_robust",
    "dh_pearson_union",
    "dh_spearman_union",
    "dh_residual_mean_union_eV_per_Ang",
    "dh_residual_std_union_eV_per_Ang",
    "dh_residual_median_union_eV_per_Ang",
    "dh_residual_bias_over_mae_union",
    "dh_residual_abs_p90_union_eV_per_Ang",
    "dh_residual_abs_p95_union_eV_per_Ang",
    "dh_residual_abs_p99_union_eV_per_Ang",
]
ROBUST_DERIVATIVE_METRICS = [
    "dh_relative_frobenius_union_robust",
    "dh_relative_l1_union_robust",
]
CORRELATION_RESIDUAL_DERIVATIVE_METRICS = [
    "dh_pearson_union",
    "dh_spearman_union",
    "dh_residual_mean_union_eV_per_Ang",
    "dh_residual_std_union_eV_per_Ang",
    "dh_residual_median_union_eV_per_Ang",
    "dh_residual_abs_p90_union_eV_per_Ang",
    "dh_residual_abs_p95_union_eV_per_Ang",
    "dh_residual_abs_p99_union_eV_per_Ang",
    "dh_residual_bias_over_mae_union",
]


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


def _int_value(value: Any) -> int | None:
    parsed = number(value)
    if parsed is None:
        return None
    integer = int(parsed)
    return integer if math.isclose(parsed, integer, rel_tol=0.0, abs_tol=1e-9) else None


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


def _resolve_path(value: Any, *, base_dir: Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir or Path.cwd()) / path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _split_counts(split_manifest: dict[str, Any]) -> dict[str, int]:
    raw = split_manifest.get("split_counts")
    if isinstance(raw, dict):
        counts: dict[str, int] = {}
        for split in ("train", "validation", "test"):
            counts[split] = _int_value(raw.get(split)) or 0
        return counts
    counts = {"train": 0, "validation": 0, "test": 0}
    rows = split_manifest.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("split") in counts:
                counts[str(row["split"])] += 1
    return counts


def _metadata_from_counts(
    *,
    dataset_root: Path | None,
    dataset_id: str,
    counts: dict[str, int],
    source: str,
) -> dict[str, Any]:
    n_train = counts.get("train") or None
    n_validation = counts.get("validation") or None
    n_test = counts.get("test") or None
    n_total = sum(value for value in counts.values() if value)
    x_dataset_size = n_train if n_train is not None else (n_total or None)
    x_dataset_size_kind = "N_train" if n_train is not None else ("N_total" if x_dataset_size is not None else "")
    return {
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root) if dataset_root is not None else "",
        "n_train": n_train,
        "n_validation": n_validation,
        "n_test": n_test,
        "n_total": n_total or None,
        "x_dataset_size": x_dataset_size,
        "x_dataset_size_kind": x_dataset_size_kind,
        "dataset_size_source": source,
        "warnings": [],
    }


def _metadata_from_dataset_root(
    dataset_root: Path | None,
    *,
    dataset_id: str = "",
    source: str = "frozen_split_manifest",
) -> dict[str, Any] | None:
    if dataset_root is None:
        return None
    split_path = dataset_root / "frozen_split_manifest.json"
    split_manifest = read_json(split_path)
    counts = _split_counts(split_manifest)
    if any(counts.values()):
        return _metadata_from_counts(
            dataset_root=dataset_root,
            dataset_id=dataset_id or dataset_root.name,
            counts=counts,
            source=source,
        )
    benchmark_manifest = read_json(dataset_root / "benchmark_dataset_manifest.json")
    n_total = _int_value(benchmark_manifest.get("total_snapshots")) or _int_value(benchmark_manifest.get("valid_snapshots"))
    samples = benchmark_manifest.get("samples")
    if n_total is None and isinstance(samples, list):
        n_total = len(samples)
    if n_total is None:
        return None
    return {
        "dataset_id": dataset_id or str(benchmark_manifest.get("dataset_id") or dataset_root.name),
        "dataset_root": str(dataset_root),
        "n_train": None,
        "n_validation": None,
        "n_test": None,
        "n_total": n_total,
        "x_dataset_size": n_total,
        "x_dataset_size_kind": "N_total",
        "dataset_size_source": "benchmark_dataset_manifest",
        "warnings": [],
    }


def _metadata_from_explicit_payload(payload: dict[str, Any], *, base_dir: Path | None = None) -> dict[str, Any] | None:
    metadata = payload.get("dataset_size_metadata")
    if isinstance(metadata, dict):
        payload = {**payload, **metadata}
    dataset_root = _resolve_path(
        payload.get("dataset_root") or payload.get("source_dataset_root"),
        base_dir=base_dir,
    )
    dataset_id = str(payload.get("dataset_id") or payload.get("id") or "").strip()
    if dataset_root is not None:
        resolved = _metadata_from_dataset_root(dataset_root, dataset_id=dataset_id)
        if resolved is not None:
            return resolved
    n_train = _int_value(payload.get("n_train") or payload.get("N_train") or payload.get("train_count"))
    n_validation = _int_value(payload.get("n_validation") or payload.get("N_validation") or payload.get("validation_count"))
    n_test = _int_value(payload.get("n_test") or payload.get("N_test") or payload.get("test_count"))
    n_total = _int_value(payload.get("n_total") or payload.get("N_total") or payload.get("dataset_size") or payload.get("total_snapshot_count"))
    if n_total is None and any(value is not None for value in (n_train, n_validation, n_test)):
        n_total = sum(value or 0 for value in (n_train, n_validation, n_test))
    x_dataset_size = n_train if n_train is not None else n_total
    if x_dataset_size is None:
        return None
    return {
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root) if dataset_root is not None else "",
        "n_train": n_train,
        "n_validation": n_validation,
        "n_test": n_test,
        "n_total": n_total,
        "x_dataset_size": x_dataset_size,
        "x_dataset_size_kind": "N_train" if n_train is not None else "N_total",
        "dataset_size_source": "explicit_payload",
        "warnings": [],
    }


def _metadata_from_stage_manifest(manifest_path: Path) -> dict[str, Any] | None:
    payload = read_json(manifest_path)
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    sweep_record = extra.get("sweep_record") if isinstance(extra.get("sweep_record"), dict) else {}
    merged = {
        **context,
        "dataset_id": sweep_record.get("dataset_id") or context.get("dataset_id") or "",
        "dataset_root": sweep_record.get("dataset_root") or context.get("dataset_root") or "",
    }
    split_path = _resolve_path(context.get("frozen_split_manifest_path"), base_dir=manifest_path.parent)
    if split_path is not None:
        split_manifest = read_json(split_path)
        counts = _split_counts(split_manifest)
        if any(counts.values()):
            dataset_root = _resolve_path(merged.get("dataset_root"), base_dir=manifest_path.parent)
            return _metadata_from_counts(
                dataset_root=dataset_root,
                dataset_id=str(merged.get("dataset_id") or (dataset_root.name if dataset_root is not None else "")),
                counts=counts,
                source="frozen_split_manifest",
            )
    return _metadata_from_explicit_payload(merged, base_dir=manifest_path.parent)


def _candidate_stage_manifests(derivative_root: Path, manifest: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    result_dir = _resolve_path(manifest.get("result_dir"), base_dir=derivative_root)
    anchors = [derivative_root, derivative_root.parent]
    if result_dir is not None:
        anchors.append(result_dir)
        anchors.extend(list(result_dir.parents[:4]))
    anchors.extend(list(derivative_root.parents[:6]))
    seen: set[str] = set()
    for anchor in anchors:
        for relative in (
            "graph2mat_manifest.json",
            "deeph_manifest.json",
            "graph2mat/graph2mat_manifest.json",
            "deeph/deeph_manifest.json",
        ):
            candidate = anchor / relative
            key = str(candidate.resolve(strict=False))
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def _training_sweep_metadata(derivative_root: Path) -> dict[str, Any] | None:
    root = derivative_root.resolve(strict=False)
    manifest_paths: list[Path] = []
    for ancestor in [derivative_root, *derivative_root.parents]:
        for candidate in (ancestor / "training_sweep_manifest.json", ancestor / "sweep" / "training_sweep_manifest.json"):
            if candidate.exists() and candidate not in manifest_paths:
                manifest_paths.append(candidate)
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path)
        runs = [row for row in manifest.get("runs") or [] if isinstance(row, dict)]
        matched_workflow: dict[str, Any] | None = None
        for workflow in manifest.get("derivative_workflows") or []:
            if not isinstance(workflow, dict):
                continue
            workflow_manifest = _resolve_path(workflow.get("derivative_workflow_manifest_path"), base_dir=manifest_path.parent.parent)
            workflow_root = workflow_manifest.parent if workflow_manifest is not None else None
            if workflow_root is not None and _is_relative_to(root, workflow_root):
                matched_workflow = workflow
                break
        matched_runs: list[dict[str, Any]] = []
        if matched_workflow is not None:
            for key in ("child_result_dir", "graph2mat_child_result_dir", "deeph_child_result_dir"):
                child_root = _resolve_path(matched_workflow.get(key), base_dir=manifest_path.parent.parent)
                if child_root is None:
                    continue
                matched_runs.extend(
                    row
                    for row in runs
                    if _resolve_path(row.get("run_root"), base_dir=manifest_path.parent.parent) == child_root
                )
            dataset_root = _resolve_path(matched_workflow.get("dataset_root"), base_dir=manifest_path.parent.parent)
            dataset_id = str(matched_workflow.get("dataset_id") or matched_workflow.get("run_id") or "").strip()
            if dataset_root is None and matched_runs:
                dataset_root = _resolve_path(matched_runs[0].get("dataset_root"), base_dir=manifest_path.parent.parent)
            if not dataset_id and matched_runs:
                dataset_id = str(matched_runs[0].get("dataset_id") or "").strip()
            resolved = _metadata_from_dataset_root(dataset_root, dataset_id=dataset_id)
            if resolved is not None:
                return resolved
        for row in runs:
            run_root = _resolve_path(row.get("run_root"), base_dir=manifest_path.parent.parent)
            if run_root is not None and _is_relative_to(root, run_root):
                return _metadata_from_explicit_payload(row, base_dir=manifest_path.parent.parent)
    return None


def _metadata_inferred_from_path(derivative_root: Path) -> dict[str, Any] | None:
    for part in reversed(derivative_root.parts):
        match = None
        for pattern in (r"(?:n[_-]?train|train)[_-]?(\d+)", r"(?:n[_-]?total|snap|snapshot|size|iid)[_-]?(\d+)"):
            match = re.search(pattern, part.lower())
            if match:
                break
        if match:
            value = int(match.group(1))
            kind = "N_train" if "train" in part.lower() else "N_total"
            warning = _warning(
                "dataset_size_inferred_from_path",
                "Dataset size was inferred from a path token; prefer explicit frozen split metadata.",
                dataset_size=value,
                path=str(derivative_root),
            )
            return {
                "dataset_id": "",
                "dataset_root": "",
                "n_train": value if kind == "N_train" else None,
                "n_validation": None,
                "n_test": None,
                "n_total": value if kind == "N_total" else None,
                "x_dataset_size": value,
                "x_dataset_size_kind": kind,
                "dataset_size_source": "inferred_from_path",
                "warnings": [warning],
            }
    return None


def resolve_dataset_size_metadata(derivative_root: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    derivative_root = _normalize_root(derivative_root)
    manifest = manifest or read_json(derivative_root / "manifest.json")
    for resolver in (
        lambda: _metadata_from_explicit_payload(manifest, base_dir=derivative_root),
        lambda: next(
            (
                metadata
                for metadata in (_metadata_from_stage_manifest(path) for path in _candidate_stage_manifests(derivative_root, manifest) if path.exists())
                if metadata is not None
            ),
            None,
        ),
        lambda: _training_sweep_metadata(derivative_root),
        lambda: _metadata_inferred_from_path(derivative_root),
    ):
        metadata = resolver()
        if metadata is not None and metadata.get("x_dataset_size") is not None:
            return metadata
    return {
        "dataset_id": "",
        "dataset_root": "",
        "n_train": None,
        "n_validation": None,
        "n_test": None,
        "n_total": None,
        "x_dataset_size": None,
        "x_dataset_size_kind": "",
        "dataset_size_source": "missing",
        "warnings": [
            _warning(
                "dataset_size_metadata_missing",
                "Dataset-size metadata is unavailable; metric-vs-dataset-size plots were not fabricated.",
                derivative_root=str(derivative_root),
            )
        ],
    }


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
    quantile_rows = read_csv_rows(derivative_root / "derivative_ref_abs_quantile_metrics.csv")
    group_metrics = read_json(derivative_root / "derivative_group_metrics.json")
    onsite_offsite_rows = read_csv_rows(derivative_root / "derivative_onsite_offsite_metrics.csv")
    onsite_offsite_payload = read_json(derivative_root / "derivative_onsite_offsite_metrics.json")
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
    if not quantile_rows:
        warnings.append(
            _warning(
                "derivative_ref_abs_quantile_metrics_missing",
                "derivative_ref_abs_quantile_metrics.csv is unavailable; abs-reference quantile derivative plots were not fabricated.",
                severity="info",
            )
        )
    if not group_metrics:
        warnings.append(
            _warning(
                "derivative_group_metrics_missing",
                "derivative_group_metrics.json is unavailable; robust atom/axis derivative group plots were not fabricated.",
                severity="info",
            )
        )
    if not onsite_offsite_rows:
        if onsite_offsite_payload.get("available") is False:
            reason = str(onsite_offsite_payload.get("reason") or "onsite_offsite_metrics_unavailable")
            warnings.append(
                _warning(
                    "derivative_onsite_offsite_metrics_unavailable",
                    f"Onsite/offsite derivative metrics are unavailable: {reason}.",
                    severity="info",
                    reason=reason,
                )
            )
        else:
            warnings.append(
                _warning(
                    "derivative_onsite_offsite_metrics_missing",
                    "derivative_onsite_offsite_metrics.csv/json is unavailable; onsite/offsite derivative plots were not fabricated.",
                    severity="info",
                )
            )
    dataset_size_metadata = resolve_dataset_size_metadata(derivative_root, manifest)
    warnings.extend(dataset_size_metadata.get("warnings") or [])
    return {
        "root": derivative_root,
        "model": model,
        "model_label": _model_label(model),
        "manifest": manifest,
        "dataset_size_metadata": dataset_size_metadata,
        "metric_rows": metric_rows,
        "quantile_rows": quantile_rows,
        "group_metrics": group_metrics,
        "onsite_offsite_rows": onsite_offsite_rows,
        "onsite_offsite_payload": onsite_offsite_payload,
        "hermiticity_rows": hermiticity_rows,
        "stencil_rows": stencil_rows,
        "warnings": warnings,
    }


def _combined_rows(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        model = dataset["model"]
        model_label = dataset["model_label"]
        dataset_size_metadata = dataset.get("dataset_size_metadata") or {}
        for row in dataset["metric_rows"]:
            rows.append(
                {
                    "dataset_id": dataset_size_metadata.get("dataset_id") or "",
                    "dataset_root": dataset_size_metadata.get("dataset_root") or "",
                    "n_train": dataset_size_metadata.get("n_train"),
                    "n_validation": dataset_size_metadata.get("n_validation"),
                    "n_test": dataset_size_metadata.get("n_test"),
                    "n_total": dataset_size_metadata.get("n_total"),
                    "x_dataset_size": dataset_size_metadata.get("x_dataset_size"),
                    "x_dataset_size_kind": dataset_size_metadata.get("x_dataset_size_kind") or "",
                    "dataset_size_source": dataset_size_metadata.get("dataset_size_source") or "missing",
                    "model": model,
                    "model_label": model_label,
                    "sample": str(row.get("sample") or ""),
                    "atom_index_zero_based": row.get("atom_index_zero_based"),
                    "axis": str(row.get("axis") or ""),
                    "delta_ang": number(row.get("delta_ang")),
                    "finite_difference_method": str(row.get("finite_difference_method") or ""),
                    "invalid_geometry": _bool(row.get("invalid_geometry"))
                    or _bool(row.get("geometry_validation_failed")),
                    "geometry_validation_failed": _bool(row.get("geometry_validation_failed"))
                    or _bool(row.get("invalid_geometry")),
                    "geometry_issue_codes": str(row.get("geometry_issue_codes") or ""),
                    "dh_mae_union_eV_per_Ang": number(row.get("dh_mae_union_eV_per_Ang")),
                    "dh_rmse_union_eV_per_Ang": number(row.get("dh_rmse_union_eV_per_Ang")),
                    "dh_relative_frobenius_ref": number(row.get("dh_relative_frobenius_ref")),
                    "dh_signal_to_noise_ratio": number(row.get("dh_signal_to_noise_ratio")),
                    "dh_signal_over_abs_h_ref": number(row.get("dh_signal_over_abs_h_ref")),
                    "dh_abs_h_pred_rel_error_ref": number(row.get("dh_abs_h_pred_rel_error_ref")),
                    "dh_relative_frobenius_union_robust": number(row.get("dh_relative_frobenius_union_robust")),
                    "dh_relative_l1_union_robust": number(row.get("dh_relative_l1_union_robust")),
                    "dh_norm_ref_union_fro": number(row.get("dh_norm_ref_union_fro")),
                    "dh_norm_error_union_fro": number(row.get("dh_norm_error_union_fro")),
                    "dh_norm_ref_l1_union": number(row.get("dh_norm_ref_l1_union")),
                    "dh_norm_error_l1_union": number(row.get("dh_norm_error_l1_union")),
                    "dh_pearson_union": number(row.get("dh_pearson_union")),
                    "dh_spearman_union": number(row.get("dh_spearman_union")),
                    "dh_residual_mean_union_eV_per_Ang": number(row.get("dh_residual_mean_union_eV_per_Ang")),
                    "dh_residual_std_union_eV_per_Ang": number(row.get("dh_residual_std_union_eV_per_Ang")),
                    "dh_residual_median_union_eV_per_Ang": number(row.get("dh_residual_median_union_eV_per_Ang")),
                    "dh_residual_abs_p90_union_eV_per_Ang": number(row.get("dh_residual_abs_p90_union_eV_per_Ang")),
                    "dh_residual_abs_p95_union_eV_per_Ang": number(row.get("dh_residual_abs_p95_union_eV_per_Ang")),
                    "dh_residual_abs_p99_union_eV_per_Ang": number(row.get("dh_residual_abs_p99_union_eV_per_Ang")),
                    "dh_residual_bias_over_mae_union": number(row.get("dh_residual_bias_over_mae_union")),
                    "dh_false_zero_rate": number(row.get("dh_false_zero_rate")),
                    "dh_false_nonzero_rate": number(row.get("dh_false_nonzero_rate")),
                    "dh_support_f1": number(row.get("dh_support_f1")),
                    "dh_support_changed": _bool(row.get("dh_support_changed"))
                    or bool(number(row.get("dh_false_zero_rate")) not in (None, 0.0))
                    or bool(number(row.get("dh_false_nonzero_rate")) not in (None, 0.0)),
                }
            )
    return rows


def _valid_geometry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not (_bool(row.get("invalid_geometry")) or _bool(row.get("geometry_validation_failed")))
    ]


def _invalid_geometry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _bool(row.get("invalid_geometry")) or _bool(row.get("geometry_validation_failed"))
    ]


def _combined_hermiticity_rows(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        model = dataset["model"]
        model_label = dataset["model_label"]
        dataset_size_metadata = dataset.get("dataset_size_metadata") or {}
        for row in dataset["hermiticity_rows"]:
            rows.append(
                {
                    "dataset_id": dataset_size_metadata.get("dataset_id") or "",
                    "dataset_root": dataset_size_metadata.get("dataset_root") or "",
                    "n_train": dataset_size_metadata.get("n_train"),
                    "n_validation": dataset_size_metadata.get("n_validation"),
                    "n_test": dataset_size_metadata.get("n_test"),
                    "n_total": dataset_size_metadata.get("n_total"),
                    "x_dataset_size": dataset_size_metadata.get("x_dataset_size"),
                    "x_dataset_size_kind": dataset_size_metadata.get("x_dataset_size_kind") or "",
                    "dataset_size_source": dataset_size_metadata.get("dataset_size_source") or "missing",
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


def _combined_quantile_rows(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        model = dataset["model"]
        model_label = dataset["model_label"]
        dataset_size_metadata = dataset.get("dataset_size_metadata") or {}
        for row in dataset["quantile_rows"]:
            rows.append(
                {
                    "dataset_id": dataset_size_metadata.get("dataset_id") or "",
                    "dataset_root": dataset_size_metadata.get("dataset_root") or "",
                    "n_train": dataset_size_metadata.get("n_train"),
                    "n_validation": dataset_size_metadata.get("n_validation"),
                    "n_test": dataset_size_metadata.get("n_test"),
                    "n_total": dataset_size_metadata.get("n_total"),
                    "x_dataset_size": dataset_size_metadata.get("x_dataset_size"),
                    "x_dataset_size_kind": dataset_size_metadata.get("x_dataset_size_kind") or "",
                    "dataset_size_source": dataset_size_metadata.get("dataset_size_source") or "missing",
                    "model": model,
                    "model_label": model_label,
                    "sample": str(row.get("sample") or ""),
                    "source_model": str(row.get("source_model") or model),
                    "reference_source": str(row.get("reference_source") or ""),
                    "base_sample_id": str(row.get("base_sample_id") or ""),
                    "atom_index_zero_based": row.get("atom_index_zero_based"),
                    "axis": str(row.get("axis") or ""),
                    "delta_ang": number(row.get("delta_ang")),
                    "finite_difference_method": str(row.get("finite_difference_method") or ""),
                    "support_threshold": number(row.get("support_threshold")),
                    "quantile_bin": _int_value(row.get("quantile_bin")),
                    "n_entries": _int_value(row.get("n_entries")),
                    "abs_ref_min_eV_per_Ang": number(row.get("abs_ref_min_eV_per_Ang")),
                    "abs_ref_max_eV_per_Ang": number(row.get("abs_ref_max_eV_per_Ang")),
                    "abs_ref_mean_eV_per_Ang": number(row.get("abs_ref_mean_eV_per_Ang")),
                    "dh_error_mae_eV_per_Ang": number(row.get("dh_error_mae_eV_per_Ang")),
                    "dh_error_rmse_eV_per_Ang": number(row.get("dh_error_rmse_eV_per_Ang")),
                    "dh_error_relative_l1_robust": number(row.get("dh_error_relative_l1_robust")),
                }
            )
    return rows


def _combined_group_rows(datasets: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        group_metrics = dataset.get("group_metrics") or {}
        source_rows = group_metrics.get(key) if isinstance(group_metrics.get(key), list) else []
        for row in source_rows:
            source_model = str(row.get("source_model") or dataset["model"])
            model = source_model.lower()
            rows.append(
                {
                    "model": model,
                    "model_label": _model_label(model),
                    "source_model": source_model,
                    "atom_index_zero_based": row.get("atom_index_zero_based"),
                    "axis": str(row.get("axis") or ""),
                    "atom_axis": f"{row.get('atom_index_zero_based')}:{row.get('axis')}",
                    "n_stencils": _int_value(row.get("n_stencils")),
                    "dh_relative_frobenius_union_robust_mean": number(row.get("dh_relative_frobenius_union_robust_mean")),
                    "dh_relative_frobenius_union_robust_median": number(row.get("dh_relative_frobenius_union_robust_median")),
                    "dh_mae_union_eV_per_Ang_mean": number(row.get("dh_mae_union_eV_per_Ang_mean")),
                    "dh_mae_union_eV_per_Ang_median": number(row.get("dh_mae_union_eV_per_Ang_median")),
                    "dh_rmse_union_eV_per_Ang_mean": number(row.get("dh_rmse_union_eV_per_Ang_mean")),
                    "dh_rmse_union_eV_per_Ang_median": number(row.get("dh_rmse_union_eV_per_Ang_median")),
                    "dh_relative_l1_union_robust_mean": number(row.get("dh_relative_l1_union_robust_mean")),
                    "dh_relative_l1_union_robust_median": number(row.get("dh_relative_l1_union_robust_median")),
                    "dh_relative_frobenius_union_robust_pooled": number(
                        row.get("dh_relative_frobenius_union_robust_pooled")
                    ),
                    "dh_relative_l1_union_robust_pooled": number(row.get("dh_relative_l1_union_robust_pooled")),
                }
            )
    return rows


def _combined_onsite_offsite_rows(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_size_metadata = dataset.get("dataset_size_metadata") or {}
        for row in dataset["onsite_offsite_rows"]:
            source_model = str(row.get("source_model") or dataset["model"])
            model = source_model.lower()
            rows.append(
                {
                    "dataset_id": dataset_size_metadata.get("dataset_id") or "",
                    "dataset_root": dataset_size_metadata.get("dataset_root") or "",
                    "n_train": dataset_size_metadata.get("n_train"),
                    "n_validation": dataset_size_metadata.get("n_validation"),
                    "n_test": dataset_size_metadata.get("n_test"),
                    "n_total": dataset_size_metadata.get("n_total"),
                    "x_dataset_size": dataset_size_metadata.get("x_dataset_size"),
                    "x_dataset_size_kind": dataset_size_metadata.get("x_dataset_size_kind") or "",
                    "dataset_size_source": dataset_size_metadata.get("dataset_size_source") or "missing",
                    "model": model,
                    "model_label": _model_label(model),
                    "source_model": source_model,
                    "sample": str(row.get("sample") or ""),
                    "atom_index_zero_based": row.get("atom_index_zero_based"),
                    "axis": str(row.get("axis") or ""),
                    "delta_ang": number(row.get("delta_ang")),
                    "finite_difference_method": str(row.get("finite_difference_method") or ""),
                    "dh_onsite_relative_frobenius_robust": number(row.get("dh_onsite_relative_frobenius_robust")),
                    "dh_onsite_mae_eV_per_Ang": number(row.get("dh_onsite_mae_eV_per_Ang")),
                    "dh_onsite_rmse_eV_per_Ang": number(row.get("dh_onsite_rmse_eV_per_Ang")),
                    "dh_offsite_relative_frobenius_robust": number(row.get("dh_offsite_relative_frobenius_robust")),
                    "dh_offsite_mae_eV_per_Ang": number(row.get("dh_offsite_mae_eV_per_Ang")),
                    "dh_offsite_rmse_eV_per_Ang": number(row.get("dh_offsite_rmse_eV_per_Ang")),
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


def _metric_unit(key: str) -> str:
    if "eV_per_Ang" in key or key.endswith("_eV_per_Ang"):
        return "eV/Ang"
    if "eV2_per_Ang2" in key:
        return "eV^2/Ang^2"
    return ""


def _metric_higher_is_better(key: str) -> bool:
    return any(token in key for token in ("support_f1", "precision", "recall", "pearson", "spearman", "cosine_similarity"))


def _plot_semantics(
    *,
    data_grain: str,
    aggregation: str,
    x_semantics: str,
    y_semantics: str,
    metric_keys: list[str],
    units: dict[str, str] | None = None,
) -> dict[str, Any]:
    higher = bool(metric_keys) and all(_metric_higher_is_better(key) for key in metric_keys)
    return {
        "data_grain": data_grain,
        "aggregation": aggregation,
        "x_semantics": x_semantics,
        "y_semantics": y_semantics,
        "lower_is_better": not higher,
        "higher_is_better": higher,
        "units": units or {key: _metric_unit(key) for key in metric_keys},
        "diagnostic_only": True,
    }


def _grouped_bar_plot(
    plot_id: str,
    title: str,
    metrics: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_keys = [str(metric.get("key") or "") for metric in metrics if metric.get("key")]
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
        **_plot_semantics(
            data_grain="model_mean",
            aggregation="mean over rows grouped by model",
            x_semantics="model or grouped diagnostic category",
            y_semantics="mean derivative diagnostic metric",
            metric_keys=metric_keys,
            units={str(metric.get("key")): str(metric.get("unit") or "") for metric in metrics if metric.get("key")},
        ),
    }


def _grouped_bar_plot_with_x(
    plot_id: str,
    title: str,
    metrics: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    x_key: str,
) -> dict[str, Any]:
    plot = _grouped_bar_plot(plot_id, title, metrics, rows)
    plot["x_key"] = x_key
    plot["data_grain"] = f"{x_key}_mean"
    plot["aggregation"] = f"mean over rows grouped by {x_key}"
    plot["x_semantics"] = x_key
    return plot


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
        **_plot_semantics(
            data_grain="sample_stencil_delta",
            aggregation="none",
            x_semantics=x_title,
            y_semantics=y_title,
            metric_keys=[metric_key],
            units={"x": "Ang" if "delta" in x_title.lower() else "", metric_key: _metric_unit(metric_key)},
        ),
    }


def _dataset_size_scatter_plot(
    plot_id: str,
    title: str,
    y_title: str,
    metric_key: str,
    rows: list[dict[str, Any]],
    *,
    x_dataset_size_kind: str,
    metrics: list[dict[str, Any]] | None = None,
    series_key: str = "model_label",
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    plot = _scatter_plot(
        plot_id,
        title,
        f"{x_dataset_size_kind} snapshots",
        y_title,
        rows,
        metric_key=metric_key,
    )
    plot.update(
        {
            "x_key": "x_dataset_size",
            "series_key": series_key,
            "dataset_size_plot": True,
            "data_grain": "model_dataset_size_mean",
            "aggregation": "mean over rows grouped by model and dataset size",
            "x_semantics": (
                "x_dataset_size is N_train when available, otherwise N_total; "
                "see row x_dataset_size_kind and dataset_size_source."
            ),
            "y_semantics": y_title,
            "units": {
                "x_dataset_size": "snapshots",
                **{
                    str(metric.get("key")): str(metric.get("unit") or "")
                    for metric in (metrics or [{"key": metric_key, "unit": _metric_unit(metric_key)}])
                    if metric.get("key")
                },
            },
            "lower_is_better": not all(
                _metric_higher_is_better(str(metric.get("key") or metric_key))
                for metric in (metrics or [{"key": metric_key}])
            ),
            "higher_is_better": all(
                _metric_higher_is_better(str(metric.get("key") or metric_key))
                for metric in (metrics or [{"key": metric_key}])
            ),
            "dataset_size_sources": sorted(
                {
                    str(row.get("dataset_size_source") or "missing")
                    for row in rows
                }
            ),
            "warnings": warnings or [],
        }
    )
    if metrics:
        plot["metrics"] = metrics
    return plot


def _quantile_plot(
    plot_id: str,
    title: str,
    y_title: str,
    metric_key: str,
    rows: list[dict[str, Any]],
    *,
    metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    plot = _scatter_plot(plot_id, title, "abs |dH_ref| quantile bin", y_title, rows, metric_key=metric_key)
    plot.update(
        {
            "x_key": "quantile_bin",
            "series_key": "model_label",
            "data_grain": "model_quantile_mean",
            "aggregation": "mean over rows grouped by model and reference-magnitude quantile bin",
            "x_semantics": "quantile bin of absolute reference derivative magnitude",
        }
    )
    if metrics:
        plot["metrics"] = metrics
    return plot


def _aggregate_quantile_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        quantile_bin = row.get("quantile_bin")
        if quantile_bin is None:
            continue
        buckets.setdefault((str(row.get("model_label") or ""), int(quantile_bin)), []).append(row)
    results: list[dict[str, Any]] = []
    for (model_label, quantile_bin), group in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        results.append(
            {
                "model_label": model_label,
                "quantile_bin": quantile_bin,
                "n_entries_total": sum(_int_value(row.get("n_entries")) or 0 for row in group),
                "dh_error_mae_eV_per_Ang": _mean([row.get("dh_error_mae_eV_per_Ang") for row in group]),
                "dh_error_rmse_eV_per_Ang": _mean([row.get("dh_error_rmse_eV_per_Ang") for row in group]),
                "dh_error_relative_l1_robust": _mean([row.get("dh_error_relative_l1_robust") for row in group]),
            }
        )
    return results


def _preferred_group_metric_key(rows: list[dict[str, Any]], prefix: str) -> str:
    pooled = f"{prefix}_pooled"
    return pooled if any(row.get(pooled) is not None for row in rows) else f"{prefix}_mean"


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


def _aggregate_metric_vs_dataset_size(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        x_dataset_size = _int_value(row.get("x_dataset_size"))
        model = str(row.get("model") or "")
        if x_dataset_size is None or not model:
            continue
        buckets.setdefault((model, x_dataset_size), []).append(row)
    results: list[dict[str, Any]] = []
    for (model, x_dataset_size), group in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        stencils = {
            (
                str(item.get("sample") or ""),
                item.get("atom_index_zero_based"),
                str(item.get("axis") or ""),
                item.get("delta_ang"),
                str(item.get("finite_difference_method") or ""),
            )
            for item in group
        }
        aggregated: dict[str, Any] = {
            "model": model,
            "model_label": str(group[0].get("model_label") or model),
            "x_dataset_size": x_dataset_size,
            "x_dataset_size_kind": str(group[0].get("x_dataset_size_kind") or ""),
            "n_train": group[0].get("n_train"),
            "n_total": group[0].get("n_total"),
            "dataset_size_source": ", ".join(str(value) for value in _sorted_unique([item.get("dataset_size_source") for item in group if item.get("dataset_size_source")])),
            "n_rows": len(group),
            "n_stencils": len(stencils),
            "delta_values": _sorted_unique(sorted({item.get("delta_ang") for item in group if item.get("delta_ang") is not None})),
            "atom_indices": _sorted_unique(sorted({item.get("atom_index_zero_based") for item in group if item.get("atom_index_zero_based") not in (None, "")})),
            "axes": _sorted_unique(sorted({str(item.get("axis")) for item in group if str(item.get("axis") or "")})),
            "dataset_ids": _sorted_unique([str(item.get("dataset_id") or "") for item in group if str(item.get("dataset_id") or "")]),
        }
        for key in DATASET_SIZE_METRICS:
            aggregated[key] = _mean([number(item.get(key)) for item in group])
        results.append(aggregated)
    return results


def _delta_series_label(model_label: str, delta_ang: float) -> str:
    return f"{model_label} Δ={delta_ang:g} Å"


def _aggregate_metric_vs_dataset_size_by_delta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        x_dataset_size = _int_value(row.get("x_dataset_size"))
        delta_ang = number(row.get("delta_ang"))
        model = str(row.get("model") or "")
        finite_difference_method = str(row.get("finite_difference_method") or "")
        if x_dataset_size is None or delta_ang is None or not model:
            continue
        buckets.setdefault((model, x_dataset_size, delta_ang, finite_difference_method), []).append(row)
    results: list[dict[str, Any]] = []
    for (model, x_dataset_size, delta_ang, finite_difference_method), group in sorted(
        buckets.items(),
        key=lambda item: (item[0][0], item[0][2], item[0][1], item[0][3]),
    ):
        model_label = str(group[0].get("model_label") or model)
        stencils = {
            (
                str(item.get("sample") or ""),
                item.get("atom_index_zero_based"),
                str(item.get("axis") or ""),
                item.get("delta_ang"),
                str(item.get("finite_difference_method") or ""),
            )
            for item in group
        }
        aggregated: dict[str, Any] = {
            "model": model,
            "model_label": model_label,
            "delta_ang": delta_ang,
            "finite_difference_method": finite_difference_method,
            "x_dataset_size": x_dataset_size,
            "x_dataset_size_kind": str(group[0].get("x_dataset_size_kind") or ""),
            "n_train": group[0].get("n_train"),
            "n_total": group[0].get("n_total"),
            "dataset_size_source": ", ".join(str(value) for value in _sorted_unique([item.get("dataset_size_source") for item in group if item.get("dataset_size_source")])),
            "n_rows": len(group),
            "n_stencils": len(stencils),
            "series_label": _delta_series_label(model_label, delta_ang),
        }
        for key in DATASET_SIZE_METRICS:
            aggregated[key] = _mean([number(item.get(key)) for item in group])
        results.append(aggregated)
    return results


def _aggregate_metric_vs_dataset_size_by_key(rows: list[dict[str, Any]], key: str, label: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        x_dataset_size = _int_value(row.get("x_dataset_size"))
        model = str(row.get("model") or "")
        raw_value = row.get(key)
        if x_dataset_size is None or not model or raw_value in (None, ""):
            continue
        value = str(raw_value)
        buckets.setdefault((model, x_dataset_size, value), []).append(row)
    results: list[dict[str, Any]] = []
    for (model, x_dataset_size, value), group in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][2], item[0][1])):
        model_label = str(group[0].get("model_label") or model)
        stencils = {
            (
                str(item.get("sample") or ""),
                item.get("atom_index_zero_based"),
                str(item.get("axis") or ""),
                item.get("delta_ang"),
                str(item.get("finite_difference_method") or ""),
            )
            for item in group
        }
        aggregated: dict[str, Any] = {
            "model": model,
            "model_label": model_label,
            key: value,
            "x_dataset_size": x_dataset_size,
            "x_dataset_size_kind": str(group[0].get("x_dataset_size_kind") or ""),
            "n_train": group[0].get("n_train"),
            "n_total": group[0].get("n_total"),
            "dataset_size_source": ", ".join(str(item) for item in _sorted_unique([row.get("dataset_size_source") for row in group if row.get("dataset_size_source")])),
            "n_rows": len(group),
            "n_stencils": len(stencils),
            "delta_values": _sorted_unique(sorted({item.get("delta_ang") for item in group if item.get("delta_ang") is not None})),
            "series_label": f"{model_label} {label}={value}",
        }
        for metric in DATASET_SIZE_METRICS:
            aggregated[metric] = _mean([number(item.get(metric)) for item in group])
        results.append(aggregated)
    return results


def _aggregate_rows_vs_dataset_size(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        x_dataset_size = _int_value(row.get("x_dataset_size"))
        model = str(row.get("model") or "")
        if x_dataset_size is None or not model:
            continue
        buckets.setdefault((model, x_dataset_size), []).append(row)
    results: list[dict[str, Any]] = []
    for (model, x_dataset_size), group in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        result: dict[str, Any] = {
            "model": model,
            "model_label": str(group[0].get("model_label") or model),
            "x_dataset_size": x_dataset_size,
            "x_dataset_size_kind": str(group[0].get("x_dataset_size_kind") or ""),
            "n_train": group[0].get("n_train"),
            "n_total": group[0].get("n_total"),
            "dataset_size_source": ", ".join(str(value) for value in _sorted_unique([item.get("dataset_size_source") for item in group if item.get("dataset_size_source")])),
            "n_rows": len(group),
            "n_stencils": len(group),
            "dataset_ids": _sorted_unique([str(item.get("dataset_id") or "") for item in group if str(item.get("dataset_id") or "")]),
        }
        for metric in metrics:
            result[metric] = _mean([number(item.get(metric)) for item in group])
        results.append(result)
    return results


def _dataset_size_plot_result(
    rows: list[dict[str, Any]],
    *,
    hermiticity_rows: list[dict[str, Any]] | None = None,
    onsite_offsite_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    missing_count = sum(1 for row in rows if row.get("x_dataset_size") is None)
    available_rows = [row for row in rows if row.get("x_dataset_size") is not None]
    if missing_count:
        warnings.append(
            _warning(
                "dataset_size_metadata_missing",
                "Dataset-size metadata is unavailable for some derivative metric rows; those rows were excluded from dataset-size plots.",
                missing_rows=missing_count,
            )
        )
    if not available_rows:
        if not warnings:
            warnings.append(
                _warning(
                    "dataset_size_metadata_missing",
                    "Dataset-size metadata is unavailable; metric-vs-dataset-size plots were not fabricated.",
                )
            )
        return {"plots": [], "warnings": warnings, "aggregated_rows": []}
    unique_sizes = sorted({_int_value(row.get("x_dataset_size")) for row in available_rows if _int_value(row.get("x_dataset_size")) is not None})
    if len(unique_sizes) < 2:
        warnings.append(
            _warning(
                "dataset_size_plots_unavailable_single_dataset_size",
                "Metric-vs-dataset-size plots require at least two unique dataset sizes.",
                dataset_sizes=unique_sizes,
            )
        )
        return {"plots": [], "warnings": warnings, "aggregated_rows": []}
    inferred_rows = [row for row in available_rows if row.get("dataset_size_source") == "inferred_from_path"]
    if inferred_rows:
        warnings.append(
            _warning(
                "dataset_size_inferred_from_path",
                "Some dataset sizes were inferred from paths; prefer explicit frozen split metadata.",
                inferred_rows=len(inferred_rows),
            )
        )
    aggregated_rows = _aggregate_metric_vs_dataset_size(available_rows)
    delta_aggregated_rows = _aggregate_metric_vs_dataset_size_by_delta(available_rows)
    axis_aggregated_rows = _aggregate_metric_vs_dataset_size_by_key(available_rows, "axis", "axis")
    atom_values = {
        str(row.get("atom_index_zero_based"))
        for row in available_rows
        if row.get("atom_index_zero_based") not in (None, "")
    }
    atom_aggregated_rows = (
        _aggregate_metric_vs_dataset_size_by_key(available_rows, "atom_index_zero_based", "atom")
        if atom_values and len(atom_values) <= MAX_DATASET_SIZE_ATOM_SERIES
        else []
    )
    if len(atom_values) > MAX_DATASET_SIZE_ATOM_SERIES:
        warnings.append(
            _warning(
                "dataset_size_atom_plot_too_many_series",
                "Displaced-atom metric-vs-dataset-size plot was not fabricated because it would create too many series.",
                unique_displaced_atoms=len(atom_values),
                max_series=MAX_DATASET_SIZE_ATOM_SERIES,
            )
        )
    x_kinds = [str(row.get("x_dataset_size_kind") or "") for row in aggregated_rows if str(row.get("x_dataset_size_kind") or "")]
    x_kind = x_kinds[0] if len(set(x_kinds)) == 1 else "dataset size"
    plots = [
        _dataset_size_scatter_plot(
            "dh_mae_vs_dataset_size",
            "dH MAE vs dataset size",
            "dH MAE eV/Ang",
            "dh_mae_union_eV_per_Ang",
            aggregated_rows,
            x_dataset_size_kind=x_kind,
            warnings=warnings,
        ),
        _dataset_size_scatter_plot(
            "dh_rmse_vs_dataset_size",
            "dH RMSE vs dataset size",
            "dH RMSE eV/Ang",
            "dh_rmse_union_eV_per_Ang",
            aggregated_rows,
            x_dataset_size_kind=x_kind,
            warnings=warnings,
        ),
        _dataset_size_scatter_plot(
            "relative_frobenius_vs_dataset_size",
            "Relative Frobenius vs dataset size",
            "Relative Frobenius",
            "dh_relative_frobenius_ref",
            aggregated_rows,
            x_dataset_size_kind=x_kind,
            warnings=warnings,
        ),
        _dataset_size_scatter_plot(
            "signal_to_noise_vs_dataset_size",
            "dH signal-to-noise ratio vs dataset size",
            "Signal-to-noise ratio",
            "dh_signal_to_noise_ratio",
            aggregated_rows,
            x_dataset_size_kind=x_kind,
            warnings=warnings,
        ),
        _dataset_size_scatter_plot(
            "support_f1_vs_dataset_size",
            "Support F1 vs dataset size",
            "Support F1",
            "dh_support_f1",
            aggregated_rows,
            x_dataset_size_kind=x_kind,
            warnings=warnings,
        ),
        _dataset_size_scatter_plot(
            "support_error_rates_vs_dataset_size",
            "Support error rates vs dataset size",
            "Support error rate",
            "dh_false_zero_rate",
            aggregated_rows,
            x_dataset_size_kind=x_kind,
            metrics=[
                {"key": "dh_false_zero_rate", "label": "False-zero rate", "unit": ""},
                {"key": "dh_false_nonzero_rate", "label": "False-nonzero rate", "unit": ""},
            ],
            warnings=warnings,
        ),
    ]
    if delta_aggregated_rows:
        plots.extend(
            [
                _dataset_size_scatter_plot(
                    "dh_mae_vs_dataset_size_by_delta",
                    "dH MAE vs dataset size by delta",
                    "dH MAE eV/Ang",
                    "dh_mae_union_eV_per_Ang",
                    delta_aggregated_rows,
                    x_dataset_size_kind=x_kind,
                    series_key="series_label",
                    warnings=warnings,
                ),
                _dataset_size_scatter_plot(
                    "dh_rmse_vs_dataset_size_by_delta",
                    "dH RMSE vs dataset size by delta",
                    "dH RMSE eV/Ang",
                    "dh_rmse_union_eV_per_Ang",
                    delta_aggregated_rows,
                    x_dataset_size_kind=x_kind,
                    series_key="series_label",
                    warnings=warnings,
                ),
                _dataset_size_scatter_plot(
                    "relative_frobenius_vs_dataset_size_by_delta",
                    "Relative Frobenius vs dataset size by delta",
                    "Relative Frobenius",
                    "dh_relative_frobenius_ref",
                    delta_aggregated_rows,
                    x_dataset_size_kind=x_kind,
                    series_key="series_label",
                    warnings=warnings,
                ),
            ]
        )
    if axis_aggregated_rows:
        plots.append(
            _dataset_size_scatter_plot(
                "dh_mae_vs_dataset_size_by_axis",
                "dH MAE vs dataset size by axis",
                "dH MAE eV/Ang",
                "dh_mae_union_eV_per_Ang",
                axis_aggregated_rows,
                x_dataset_size_kind=x_kind,
                series_key="series_label",
                warnings=warnings,
            )
        )
        if any(row.get("dh_relative_frobenius_union_robust") is not None for row in axis_aggregated_rows):
            plots.append(
                _dataset_size_scatter_plot(
                    "robust_frobenius_vs_dataset_size_by_axis",
                    "Robust Frobenius vs dataset size by axis",
                    "Robust relative Frobenius",
                    "dh_relative_frobenius_union_robust",
                    axis_aggregated_rows,
                    x_dataset_size_kind=x_kind,
                    series_key="series_label",
                    warnings=warnings,
                )
            )
    if atom_aggregated_rows:
        plots.append(
            _dataset_size_scatter_plot(
                "dh_mae_vs_dataset_size_by_displaced_atom",
                "dH MAE vs dataset size by displaced atom",
                "dH MAE eV/Ang",
                "dh_mae_union_eV_per_Ang",
                atom_aggregated_rows,
                x_dataset_size_kind=x_kind,
                series_key="series_label",
                warnings=warnings,
            )
        )
    if any(row.get("dh_relative_frobenius_union_robust") is not None for row in aggregated_rows):
        plots.append(
            _dataset_size_scatter_plot(
                "robust_relative_frobenius_vs_dataset_size",
                "Robust relative Frobenius vs dataset size",
                "Robust relative Frobenius",
                "dh_relative_frobenius_union_robust",
                aggregated_rows,
                x_dataset_size_kind=x_kind,
                warnings=warnings,
            )
        )
    if any(row.get("dh_relative_l1_union_robust") is not None for row in aggregated_rows):
        plots.append(
            _dataset_size_scatter_plot(
                "robust_relative_l1_vs_dataset_size",
                "Robust relative L1 vs dataset size",
                "Robust relative L1",
                "dh_relative_l1_union_robust",
                aggregated_rows,
                x_dataset_size_kind=x_kind,
                warnings=warnings,
            )
        )
    if any(row.get(metric) is not None for row in aggregated_rows for metric in ("dh_pearson_union", "dh_spearman_union")):
        plots.append(
            _dataset_size_scatter_plot(
                "derivative_correlation_vs_dataset_size",
                "Derivative correlation vs dataset size",
                "Derivative correlation",
                "dh_pearson_union",
                aggregated_rows,
                x_dataset_size_kind=x_kind,
                metrics=[
                    {"key": "dh_pearson_union", "label": "Pearson", "unit": ""},
                    {"key": "dh_spearman_union", "label": "Spearman", "unit": ""},
                ],
                warnings=warnings,
            )
        )
    if any(
        row.get(metric) is not None
        for row in aggregated_rows
        for metric in (
            "dh_residual_mean_union_eV_per_Ang",
            "dh_residual_std_union_eV_per_Ang",
            "dh_residual_median_union_eV_per_Ang",
            "dh_residual_bias_over_mae_union",
        )
    ):
        plots.append(
            _dataset_size_scatter_plot(
                "derivative_residual_summary_vs_dataset_size",
                "Derivative residual summary vs dataset size",
                "Derivative residual summary",
                "dh_residual_mean_union_eV_per_Ang",
                aggregated_rows,
                x_dataset_size_kind=x_kind,
                metrics=[
                    {"key": "dh_residual_mean_union_eV_per_Ang", "label": "Mean residual", "unit": "eV/Ang"},
                    {"key": "dh_residual_std_union_eV_per_Ang", "label": "Residual std", "unit": "eV/Ang"},
                    {"key": "dh_residual_median_union_eV_per_Ang", "label": "Median residual", "unit": "eV/Ang"},
                    {"key": "dh_residual_bias_over_mae_union", "label": "Bias over MAE", "unit": ""},
                ],
                warnings=warnings,
            )
        )
    if any(
        row.get(metric) is not None
        for row in aggregated_rows
        for metric in (
            "dh_residual_abs_p90_union_eV_per_Ang",
            "dh_residual_abs_p95_union_eV_per_Ang",
            "dh_residual_abs_p99_union_eV_per_Ang",
        )
    ):
        plots.append(
            _dataset_size_scatter_plot(
                "derivative_residual_tail_vs_dataset_size",
                "Derivative residual tail vs dataset size",
                "Derivative residual tail eV/Ang",
                "dh_residual_abs_p90_union_eV_per_Ang",
                aggregated_rows,
                x_dataset_size_kind=x_kind,
                metrics=[
                    {"key": "dh_residual_abs_p90_union_eV_per_Ang", "label": "Abs residual p90", "unit": "eV/Ang"},
                    {"key": "dh_residual_abs_p95_union_eV_per_Ang", "label": "Abs residual p95", "unit": "eV/Ang"},
                    {"key": "dh_residual_abs_p99_union_eV_per_Ang", "label": "Abs residual p99", "unit": "eV/Ang"},
                ],
                warnings=warnings,
            )
        )
    hermiticity_metrics = ["dH_pred_hermiticity_defect", "dH_hermiticity_error_delta"]
    hermiticity_dataset_rows = _aggregate_rows_vs_dataset_size(hermiticity_rows or [], hermiticity_metrics)
    if any(row.get(metric) is not None for row in hermiticity_dataset_rows for metric in hermiticity_metrics):
        plots.append(
            _dataset_size_scatter_plot(
                "derivative_hermiticity_vs_dataset_size",
                "Derivative hermiticity vs dataset size",
                "Hermiticity defect",
                "dH_pred_hermiticity_defect",
                hermiticity_dataset_rows,
                x_dataset_size_kind=x_kind,
                metrics=[
                    {"key": "dH_pred_hermiticity_defect", "label": "Predicted defect", "unit": ""},
                    {"key": "dH_hermiticity_error_delta", "label": "Error delta", "unit": ""},
                ],
                warnings=warnings,
            )
        )
    onsite_offsite_metrics = [
        "dh_onsite_relative_frobenius_robust",
        "dh_offsite_relative_frobenius_robust",
        "dh_onsite_mae_eV_per_Ang",
        "dh_offsite_mae_eV_per_Ang",
    ]
    onsite_offsite_dataset_rows = _aggregate_rows_vs_dataset_size(onsite_offsite_rows or [], onsite_offsite_metrics)
    if any(row.get(metric) is not None for row in onsite_offsite_dataset_rows for metric in onsite_offsite_metrics):
        plots.append(
            _dataset_size_scatter_plot(
                "onsite_offsite_derivative_error_vs_dataset_size",
                "Onsite/offsite derivative error vs dataset size",
                "Onsite/offsite derivative error",
                "dh_onsite_relative_frobenius_robust",
                onsite_offsite_dataset_rows,
                x_dataset_size_kind=x_kind,
                metrics=[
                    {"key": "dh_onsite_relative_frobenius_robust", "label": "Onsite robust relative Frobenius", "unit": ""},
                    {"key": "dh_offsite_relative_frobenius_robust", "label": "Offsite robust relative Frobenius", "unit": ""},
                    {"key": "dh_onsite_mae_eV_per_Ang", "label": "Onsite dH MAE", "unit": "eV/Ang"},
                    {"key": "dh_offsite_mae_eV_per_Ang", "label": "Offsite dH MAE", "unit": "eV/Ang"},
                ],
                warnings=warnings,
            )
        )
    return {"plots": plots, "warnings": warnings, "aggregated_rows": aggregated_rows}


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
                "invalid_geometry": _bool(row.get("invalid_geometry")),
                "geometry_validation_failed": _bool(row.get("geometry_validation_failed")),
                "geometry_issue_codes": row.get("geometry_issue_codes") or "",
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
    invalid_geometry = _invalid_geometry_rows(metric_rows)
    if invalid_geometry:
        warnings.append(
            _warning(
                "invalid_geometry_rows_excluded_from_aggregates",
                "Geometry-invalid derivative rows were excluded from aggregate model-level plots.",
                severity="severe",
                invalid_rows=len(invalid_geometry),
                samples=sorted({str(row.get("sample") or "") for row in invalid_geometry if row.get("sample")}),
            )
        )
    if metric_rows and not any(
        row.get(metric) is not None
        for row in metric_rows
        for metric in ROBUST_DERIVATIVE_METRICS
    ):
        warnings.append(
            _warning(
                "robust_derivative_metrics_missing",
                "Robust derivative metrics are unavailable in derivative_matrix_metrics.csv; robust diagnostic plots were not fabricated.",
            )
        )
    if metric_rows and not any(
        row.get(metric) is not None
        for row in metric_rows
        for metric in CORRELATION_RESIDUAL_DERIVATIVE_METRICS
    ):
        warnings.append(
            _warning(
                "derivative_correlation_or_residual_metrics_missing",
                "Derivative correlation/residual metrics are unavailable in derivative_matrix_metrics.csv; correlation/residual diagnostic plots were not fabricated.",
            )
        )
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
    quantile_rows = _combined_quantile_rows(datasets)
    quantile_aggregate_rows = _aggregate_quantile_rows(quantile_rows)
    group_by_atom_rows = _combined_group_rows(datasets, "by_atom")
    group_by_axis_rows = _combined_group_rows(datasets, "by_axis")
    group_by_atom_axis_rows = _combined_group_rows(datasets, "by_atom_axis")
    onsite_offsite_rows = _combined_onsite_offsite_rows(datasets)
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
    aggregate_metric_rows = _valid_geometry_rows(metric_rows)
    dataset_size_plot_result = _dataset_size_plot_result(
        aggregate_metric_rows,
        hermiticity_rows=hermiticity_rows,
        onsite_offsite_rows=onsite_offsite_rows,
    )

    diagnostic_plots = [
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
            _aggregate_by_model(aggregate_metric_rows, "dh_mae_union_eV_per_Ang"),
        ),
        _grouped_bar_plot(
            "dh_rmse_by_model",
            "dH RMSE by model",
            [{"key": "dh_rmse_union_eV_per_Ang", "label": "dH RMSE", "unit": "eV/Ang"}],
            _aggregate_by_model(aggregate_metric_rows, "dh_rmse_union_eV_per_Ang"),
        ),
        _grouped_bar_plot(
            "relative_frobenius_by_model",
            "Relative Frobenius by model",
            [{"key": "dh_relative_frobenius_ref", "label": "Relative Frobenius", "unit": ""}],
            _aggregate_by_model(aggregate_metric_rows, "dh_relative_frobenius_ref"),
        ),
        _grouped_bar_plot(
            "error_by_atom_index_zero_based",
            "Error by atom index",
            [{"key": "dh_mae_union_eV_per_Ang", "label": "dH MAE", "unit": "eV/Ang"}],
            _aggregate_atom(aggregate_metric_rows, "dh_mae_union_eV_per_Ang"),
        ),
        _grouped_bar_plot(
            "error_by_axis",
            "Error by axis",
            [{"key": "dh_mae_union_eV_per_Ang", "label": "dH MAE", "unit": "eV/Ang"}],
            _aggregate_axis(aggregate_metric_rows, "dh_mae_union_eV_per_Ang"),
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
            _support_diagnostic_rows(aggregate_metric_rows),
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
    if any(row.get("dh_relative_frobenius_union_robust") is not None for row in aggregate_metric_rows):
        diagnostic_plots.append(
            _grouped_bar_plot(
                "relative_frobenius_union_robust_by_model",
                "Robust relative Frobenius by model",
                [{"key": "dh_relative_frobenius_union_robust", "label": "Robust relative Frobenius", "unit": ""}],
                _aggregate_by_model(aggregate_metric_rows, "dh_relative_frobenius_union_robust"),
            )
        )
    if any(row.get("dh_relative_l1_union_robust") is not None for row in aggregate_metric_rows):
        diagnostic_plots.append(
            _grouped_bar_plot(
                "relative_l1_union_robust_by_model",
                "Robust relative L1 by model",
                [{"key": "dh_relative_l1_union_robust", "label": "Robust relative L1", "unit": ""}],
                _aggregate_by_model(aggregate_metric_rows, "dh_relative_l1_union_robust"),
            )
        )
    if any(row.get(metric) is not None for row in aggregate_metric_rows for metric in ROBUST_DERIVATIVE_METRICS):
        diagnostic_plots.append(
            _grouped_bar_plot(
                "robust_primary_metrics_by_model",
                "Robust primary metrics by model",
                [
                    {"key": "dh_relative_frobenius_union_robust", "label": "Robust relative Frobenius", "unit": ""},
                    {"key": "dh_relative_l1_union_robust", "label": "Robust relative L1", "unit": ""},
                    {"key": "dh_mae_union_eV_per_Ang", "label": "dH MAE", "unit": "eV/Ang"},
                    {"key": "dh_rmse_union_eV_per_Ang", "label": "dH RMSE", "unit": "eV/Ang"},
                ],
                _aggregate_model_metrics(
                    aggregate_metric_rows,
                    [
                        "dh_relative_frobenius_union_robust",
                        "dh_relative_l1_union_robust",
                        "dh_mae_union_eV_per_Ang",
                        "dh_rmse_union_eV_per_Ang",
                    ],
                ),
            )
        )
    if any(row.get(metric) is not None for row in aggregate_metric_rows for metric in ("dh_pearson_union", "dh_spearman_union")):
        diagnostic_plots.append(
            _grouped_bar_plot(
                "derivative_correlation_by_model",
                "Derivative correlation by model",
                [
                    {"key": "dh_pearson_union", "label": "Pearson", "unit": ""},
                    {"key": "dh_spearman_union", "label": "Spearman", "unit": ""},
                ],
                _aggregate_model_metrics(aggregate_metric_rows, ["dh_pearson_union", "dh_spearman_union"]),
            )
        )
    if any(
        row.get(metric) is not None
        for row in aggregate_metric_rows
        for metric in (
            "dh_residual_mean_union_eV_per_Ang",
            "dh_residual_std_union_eV_per_Ang",
            "dh_residual_median_union_eV_per_Ang",
            "dh_residual_bias_over_mae_union",
        )
    ):
        diagnostic_plots.append(
            _grouped_bar_plot(
                "derivative_residual_summary_by_model",
                "Derivative residual summary by model",
                [
                    {"key": "dh_residual_mean_union_eV_per_Ang", "label": "Mean residual", "unit": "eV/Ang"},
                    {"key": "dh_residual_std_union_eV_per_Ang", "label": "Residual std", "unit": "eV/Ang"},
                    {"key": "dh_residual_median_union_eV_per_Ang", "label": "Median residual", "unit": "eV/Ang"},
                    {"key": "dh_residual_bias_over_mae_union", "label": "Bias over MAE", "unit": ""},
                ],
                _aggregate_model_metrics(
                    aggregate_metric_rows,
                    [
                        "dh_residual_mean_union_eV_per_Ang",
                        "dh_residual_std_union_eV_per_Ang",
                        "dh_residual_median_union_eV_per_Ang",
                        "dh_residual_bias_over_mae_union",
                    ],
                ),
            )
        )
    if any(
        row.get(metric) is not None
        for row in aggregate_metric_rows
        for metric in (
            "dh_residual_abs_p90_union_eV_per_Ang",
            "dh_residual_abs_p95_union_eV_per_Ang",
            "dh_residual_abs_p99_union_eV_per_Ang",
        )
    ):
        diagnostic_plots.append(
            _grouped_bar_plot(
                "derivative_residual_tail_by_model",
                "Derivative residual tail by model",
                [
                    {"key": "dh_residual_abs_p90_union_eV_per_Ang", "label": "Abs residual p90", "unit": "eV/Ang"},
                    {"key": "dh_residual_abs_p95_union_eV_per_Ang", "label": "Abs residual p95", "unit": "eV/Ang"},
                    {"key": "dh_residual_abs_p99_union_eV_per_Ang", "label": "Abs residual p99", "unit": "eV/Ang"},
                ],
                _aggregate_model_metrics(
                    aggregate_metric_rows,
                    [
                        "dh_residual_abs_p90_union_eV_per_Ang",
                        "dh_residual_abs_p95_union_eV_per_Ang",
                        "dh_residual_abs_p99_union_eV_per_Ang",
                    ],
                ),
            )
        )
    if quantile_aggregate_rows:
        diagnostic_plots.append(
            _quantile_plot(
                "derivative_error_by_abs_ref_quantile",
                "Derivative error by |dH_ref| quantile",
                "Derivative error eV/Ang",
                "dh_error_mae_eV_per_Ang",
                quantile_aggregate_rows,
                metrics=[
                    {"key": "dh_error_mae_eV_per_Ang", "label": "dH error MAE", "unit": "eV/Ang"},
                    {"key": "dh_error_rmse_eV_per_Ang", "label": "dH error RMSE", "unit": "eV/Ang"},
                ],
            )
        )
        diagnostic_plots.append(
            _quantile_plot(
                "derivative_relative_l1_by_abs_ref_quantile",
                "Derivative relative L1 by |dH_ref| quantile",
                "Robust relative L1",
                "dh_error_relative_l1_robust",
                quantile_aggregate_rows,
                metrics=[
                    {"key": "dh_error_relative_l1_robust", "label": "Robust relative L1", "unit": ""},
                ],
            )
        )
    atom_fro_key = _preferred_group_metric_key(group_by_atom_rows, "dh_relative_frobenius_union_robust")
    axis_fro_key = _preferred_group_metric_key(group_by_axis_rows, "dh_relative_frobenius_union_robust")
    axis_l1_key = _preferred_group_metric_key(group_by_axis_rows, "dh_relative_l1_union_robust")
    atom_axis_fro_key = _preferred_group_metric_key(group_by_atom_axis_rows, "dh_relative_frobenius_union_robust")
    atom_axis_l1_key = _preferred_group_metric_key(group_by_atom_axis_rows, "dh_relative_l1_union_robust")
    if group_by_atom_rows:
        diagnostic_plots.append(
            _grouped_bar_plot_with_x(
                "robust_error_by_displaced_atom",
                "Robust derivative error by displaced atom",
                [{"key": atom_fro_key, "label": "Robust relative Frobenius", "unit": ""}],
                group_by_atom_rows,
                x_key="atom_index_zero_based",
            )
        )
    if group_by_axis_rows:
        diagnostic_plots.append(
            _grouped_bar_plot_with_x(
                "robust_error_by_axis",
                "Robust derivative error by axis",
                [
                    {"key": axis_fro_key, "label": "Robust relative Frobenius", "unit": ""},
                    {"key": axis_l1_key, "label": "Robust relative L1", "unit": ""},
                ],
                group_by_axis_rows,
                x_key="axis",
            )
        )
    if group_by_atom_axis_rows:
        diagnostic_plots.append(
            _grouped_bar_plot_with_x(
                "robust_error_by_atom_axis",
                "Robust derivative error by atom and axis",
                [
                    {"key": atom_axis_fro_key, "label": "Robust relative Frobenius", "unit": ""},
                    {"key": atom_axis_l1_key, "label": "Robust relative L1", "unit": ""},
                ],
                group_by_atom_axis_rows,
                x_key="atom_axis",
            )
        )
    if onsite_offsite_rows:
        onsite_offsite_metric_keys = [
            "dh_onsite_relative_frobenius_robust",
            "dh_offsite_relative_frobenius_robust",
            "dh_onsite_mae_eV_per_Ang",
            "dh_offsite_mae_eV_per_Ang",
        ]
        diagnostic_plots.append(
            _grouped_bar_plot(
                "onsite_offsite_derivative_error",
                "Onsite/offsite derivative error by model",
                [
                    {"key": "dh_onsite_relative_frobenius_robust", "label": "Onsite robust relative Frobenius", "unit": ""},
                    {"key": "dh_offsite_relative_frobenius_robust", "label": "Offsite robust relative Frobenius", "unit": ""},
                    {"key": "dh_onsite_mae_eV_per_Ang", "label": "Onsite dH MAE", "unit": "eV/Ang"},
                    {"key": "dh_offsite_mae_eV_per_Ang", "label": "Offsite dH MAE", "unit": "eV/Ang"},
                ],
                _aggregate_model_metrics(onsite_offsite_rows, onsite_offsite_metric_keys),
            )
        )
    dataset_size_plots = dataset_size_plot_result["plots"]
    plots = [*dataset_size_plots, *diagnostic_plots] if dataset_size_plots else diagnostic_plots
    dataset_size_plot_ids = [plot["id"] for plot in dataset_size_plots]
    diagnostic_plot_ids = [plot["id"] for plot in diagnostic_plots]
    warnings = _scientific_warnings(datasets, metric_rows, paired_rows)
    warnings.extend(dataset_size_plot_result["warnings"])
    return {
        "schema": PLOT_SCHEMA,
        "available": bool(metric_rows or hermiticity_rows or onsite_offsite_rows),
        "diagnostic_only": True,
        "scientific_status": "diagnostic_only",
        "title": TITLE,
        "reference_label": REFERENCE_LABEL,
        "force_constants_label": FORCE_CONSTANTS_LABEL,
        "units": {"derivative": "eV/Ang", "delta": "Ang"},
        "methods_seen": methods_seen,
        "method_label": meta["method_label"],
        "plots": plots,
        "primary_plot_ids": dataset_size_plot_ids or diagnostic_plot_ids,
        "dataset_size_plot_ids": dataset_size_plot_ids,
        "diagnostic_plot_ids": diagnostic_plot_ids,
        "scientific_warnings": warnings,
        "inputs": {
            "roots": [str(dataset["root"]) for dataset in datasets],
            "models": [dataset["model_label"] for dataset in datasets],
        },
        "summary": {
            "metric_rows": len(metric_rows),
            "hermiticity_rows": len(hermiticity_rows),
            "onsite_offsite_rows": len(onsite_offsite_rows),
            "paired_rows": len(paired_rows),
            "dataset_size_rows": len(dataset_size_plot_result["aggregated_rows"]),
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
