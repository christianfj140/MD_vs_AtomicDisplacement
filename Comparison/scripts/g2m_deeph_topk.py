#!/usr/bin/env python3
"""Validation-only top-k selection and robust rerun planning for G2M-vs-DeepH."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from g2m_deeph_test_blindness import ROBUST_VALIDATION_STAGE, SEARCH_STAGE
from g2m_deeph_training_sweep import json_safe, stable_hash


SELECTED_CONFIGS_SCHEMA = "graph2mat_deeph_selected_configs_v1"
ROBUST_RERUN_PLAN_SCHEMA = "graph2mat_deeph_robust_rerun_plan_v1"
VALIDATION_SPLITS = {"validation", "val"}
COMPLETED_STATUSES = {"completed"}
VAL_SPECTRAL_COMPOSITE = "val_spectral_composite"
SPECTRAL_COMPOSITE_COMPONENTS: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    ("low_energy_rmse_eV", 0.30, ("low_energy_rmse_eV", "low_energy_rmse_eV_mean")),
    ("fermi_window_rmse_eV", 0.20, ("fermi_window_rmse_eV", "fermi_window_rmse_eV_mean")),
    ("frontier_window_rmse_eV", 0.15, ("frontier_window_rmse_eV", "frontier_window_rmse_eV_mean")),
    ("global_band_rmse", 0.15, ("global_band_rmse", "global_band_rmse_eV", "global_rmse_eV", "global_rmse_eV_mean")),
    ("dos_wasserstein", 0.10, ("dos_wasserstein", "dos_wasserstein_eV", "dos_wasserstein_eV_mean")),
    ("dos_mae_near_fermi", 0.10, ("dos_mae_near_fermi", "dos_mae_500_fermi_window", "dos_mae_500_fermi_window_mean")),
)


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_from_mapping(mapping: dict[str, Any], metric: str) -> float | None:
    for key in (metric, f"{metric}_mean", "value", "mean", "best_validation_value"):
        if key not in mapping:
            continue
        value = mapping.get(key)
        if isinstance(value, dict):
            nested = _metric_from_mapping(value, metric)
            if nested is not None:
                return nested
        number = finite_number(value)
        if number is not None:
            return number
    return None


def validation_metric_value(record: dict[str, Any], metric: str) -> float | None:
    """Return a validation metric value without consulting test metrics."""

    validation_metrics = record.get("validation_metrics")
    if isinstance(validation_metrics, dict):
        direct = _metric_from_mapping(validation_metrics, metric)
        if direct is not None:
            return direct

    metric_rows = record.get("metric_rows") or record.get("metrics")
    if isinstance(metric_rows, list):
        for row in metric_rows:
            if not isinstance(row, dict):
                continue
            split = str(row.get("metric_split") or row.get("split") or row.get("evaluation_split") or "").lower()
            if split not in VALIDATION_SPLITS:
                continue
            name = str(row.get("metric") or row.get("metric_key") or row.get("name") or "").strip()
            if name and name not in {metric, f"{metric}_mean"}:
                continue
            direct = _metric_from_mapping(row, metric)
            if direct is not None:
                return direct

    early_stopping = record.get("early_stopping")
    if isinstance(early_stopping, dict):
        metric_name = str(
            early_stopping.get("validation_metric_name")
            or early_stopping.get("metric")
            or early_stopping.get("selection_metric")
            or ""
        ).strip()
        if metric_name in {"", metric}:
            direct = finite_number(early_stopping.get("best_validation_value"))
            if direct is not None:
                return direct

    telemetry = record.get("telemetry")
    if isinstance(telemetry, dict):
        metric_name = str(telemetry.get("validation_metric") or telemetry.get("selection_metric") or "").strip()
        if metric_name in {"", metric}:
            direct = finite_number(telemetry.get("best_validation_value"))
            if direct is not None:
                return direct

    split = str(record.get("metric_split") or record.get("split") or record.get("evaluation_split") or "").lower()
    if split in VALIDATION_SPLITS:
        return _metric_from_mapping(record, metric)
    return None


def _validation_metric_value_any(record: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        value = validation_metric_value(record, alias)
        if value is not None:
            return value
    return None


def _median_scale(values: list[float]) -> float:
    clean = sorted(value for value in values if math.isfinite(value) and value > 0)
    if not clean:
        return 1.0
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) / 2.0


def validation_spectral_composite_values(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Compute validation-only spectral composite scores for complete records.

    The score is fail-closed: a record must expose all required validation
    spectral/DOS components. Test rows are ignored by validation_metric_value.
    Components are normalized by the median finite positive component value
    across eligible records, then combined by the preregistered weights.
    """

    component_values: dict[int, dict[str, float]] = {}
    for record in records:
        values: dict[str, float] = {}
        for component, _weight, aliases in SPECTRAL_COMPOSITE_COMPONENTS:
            value = _validation_metric_value_any(record, aliases)
            if value is None:
                break
            values[component] = value
        if len(values) == len(SPECTRAL_COMPOSITE_COMPONENTS):
            component_values[id(record)] = values
    scales = {
        component: _median_scale([values[component] for values in component_values.values()])
        for component, _weight, _aliases in SPECTRAL_COMPOSITE_COMPONENTS
    }
    scores: dict[int, dict[str, Any]] = {}
    for record in records:
        values = component_values.get(id(record))
        if values is None:
            continue
        terms = {
            component: {
                "value": values[component],
                "scale": scales[component],
                "weight": weight,
                "normalized": values[component] / scales[component],
                "weighted": weight * (values[component] / scales[component]),
            }
            for component, weight, _aliases in SPECTRAL_COMPOSITE_COMPONENTS
        }
        scores[id(record)] = {
            "score": sum(item["weighted"] for item in terms.values()),
            "components": values,
            "scales": scales,
            "terms": terms,
        }
    return scores


def _completed_records(records: list[dict[str, Any]], metric: str, *, allow_diagnostic: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("status") or "") not in COMPLETED_STATUSES:
            continue
        if not allow_diagnostic and (
            record.get("diagnostic_only") is True
            or str(record.get("scientific_status") or "").lower() == "diagnostic_only"
            or str(record.get("comparability_status") or "").lower() == "diagnostic_only"
        ):
            continue
        value = validation_metric_value(record, metric)
        if value is None:
            continue
        selected.append(record)
    return selected


def _completed_records_for_status(records: list[dict[str, Any]], *, allow_diagnostic: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("status") or "") not in COMPLETED_STATUSES:
            continue
        if not allow_diagnostic and (
            record.get("diagnostic_only") is True
            or str(record.get("scientific_status") or "").lower() == "diagnostic_only"
            or str(record.get("comparability_status") or "").lower() == "diagnostic_only"
        ):
            continue
        selected.append(record)
    return selected


def _group_key(record: dict[str, Any], grouping: str) -> tuple[str, ...]:
    model = str(record.get("model") or "")
    dataset_id = str(record.get("dataset_id") or "")
    if grouping in {"model_dataset", "dataset_model", ""}:
        return (model, dataset_id)
    if grouping in {"model", "per_model"}:
        return (model,)
    raise RuntimeError("top-k grouping must be model_dataset or model.")


def _selection_row(
    record: dict[str, Any],
    *,
    metric: str,
    mode: str,
    rank: int,
    grouping: str,
    value_override: float | None = None,
    composite_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = value_override if value_override is not None else validation_metric_value(record, metric)
    if value is None:
        raise RuntimeError(f"Selected record has no validation metric {metric}: {record.get('config_id')}")
    row = {
        "rank": rank,
        "index": record.get("index"),
        "grouping": grouping,
        "model": str(record.get("model") or ""),
        "dataset_id": str(record.get("dataset_id") or ""),
        "dataset_root": str(record.get("dataset_root") or ""),
        "config_id": str(record.get("config_id") or ""),
        "config_hash": str(record.get("config_hash") or ""),
        "run_root": str(record.get("run_root") or ""),
        "original_search_run_id": str(record.get("run_id") or record.get("run_root") or record.get("config_id") or ""),
        "selection_metric": metric,
        "selection_mode": mode,
        "validation_metric_value": value,
        "common": json_safe(dict(record.get("common") or {})),
        "overrides": json_safe(dict(record.get("overrides") or {})),
        "search_strategy": str(record.get("search_strategy") or ""),
        "search_seed": record.get("search_seed"),
        "search_trial_index": record.get("search_trial_index"),
        "protocol_id": str(record.get("protocol_id") or ""),
        "protocol_hash": str(record.get("protocol_hash") or ""),
        "protocol_stage": SEARCH_STAGE,
        "source_status": str(record.get("status") or ""),
    }
    if composite_details:
        row["validation_composite"] = json_safe(composite_details)
    return row


def select_top_configs(
    records: list[dict[str, Any]],
    *,
    metric: str,
    mode: str,
    k_per_model: int,
    grouping: str = "model_dataset",
    allow_diagnostic: bool = False,
) -> dict[str, Any]:
    """Select top-k completed configs by validation metric only."""

    if not metric:
        raise RuntimeError("selection metric is required.")
    normalized_mode = str(mode or "").lower()
    if normalized_mode not in {"min", "max"}:
        raise RuntimeError("selection mode must be min or max.")
    if k_per_model <= 0:
        raise RuntimeError("top_k_selection.k_per_model must be positive.")
    composite_scores: dict[int, dict[str, Any]] = {}
    if metric == VAL_SPECTRAL_COMPOSITE:
        status_candidates = _completed_records_for_status(records, allow_diagnostic=allow_diagnostic)
        composite_scores = validation_spectral_composite_values(status_candidates)
        candidates = [record for record in status_candidates if id(record) in composite_scores]
    else:
        candidates = _completed_records(records, metric, allow_diagnostic=allow_diagnostic)
    if not candidates:
        raise RuntimeError(f"No completed configs have validation metric {metric!r} for top-k selection.")

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for record in candidates:
        groups.setdefault(_group_key(record, grouping), []).append(record)
    rows: list[dict[str, Any]] = []
    for group, group_records in sorted(groups.items()):
        if len(group_records) < k_per_model:
            raise RuntimeError(
                f"Insufficient completed configs for group {'/'.join(group)}: "
                f"need {k_per_model}, found {len(group_records)}."
            )
        group_records.sort(
            key=lambda row: (
                -(composite_scores[id(row)]["score"] if metric == VAL_SPECTRAL_COMPOSITE else validation_metric_value(row, metric) or 0.0)
                if normalized_mode == "max"
                else (composite_scores[id(row)]["score"] if metric == VAL_SPECTRAL_COMPOSITE else validation_metric_value(row, metric) or 0.0),
                str(row.get("config_id") or ""),
                str(row.get("run_root") or ""),
            )
        )
        for rank, record in enumerate(group_records[:k_per_model], start=1):
            rows.append(
                _selection_row(
                    record,
                    metric=metric,
                    mode=normalized_mode,
                    rank=rank,
                    grouping=grouping,
                    value_override=composite_scores[id(record)]["score"] if metric == VAL_SPECTRAL_COMPOSITE else None,
                    composite_details=composite_scores.get(id(record)) if metric == VAL_SPECTRAL_COMPOSITE else None,
                )
            )
    return {
        "schema": SELECTED_CONFIGS_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol_stage": SEARCH_STAGE,
        "metric": metric,
        "mode": normalized_mode,
        "k_per_model": int(k_per_model),
        "grouping": grouping,
        "uses_test_metrics": False,
        "selection_source": "validation_only",
        "selected_count": len(rows),
        "selected_configs": rows,
    }


def _with_seed(selected: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    common = dict(selected.get("common") or {})
    overrides = dict(selected.get("overrides") or {})
    model = str(selected.get("model") or "")
    common["seed"] = int(seed)
    if model == "graph2mat":
        overrides["seed_everything"] = int(seed)
    elif model == "deeph":
        overrides["seed"] = int(seed)
    else:
        raise RuntimeError(f"Unsupported selected model for robust rerun: {model}")
    return common, overrides


def generate_robust_rerun_plan(
    selected_manifest: dict[str, Any],
    *,
    final_seeds: list[int],
    stage: str = ROBUST_VALIDATION_STAGE,
) -> dict[str, Any]:
    """Expand selected configs to deterministic multi-seed robust rerun records."""

    if not final_seeds:
        raise RuntimeError("final_seeds must not be empty for robust rerun planning.")
    clean_seeds: list[int] = []
    for seed in final_seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise RuntimeError("final_seeds must contain non-negative integers.")
        if seed not in clean_seeds:
            clean_seeds.append(seed)
    selected_rows = selected_manifest.get("selected_configs")
    if not isinstance(selected_rows, list) or not selected_rows:
        raise RuntimeError("selected_configs manifest contains no selected configs.")

    planned: list[dict[str, Any]] = []
    for selected in selected_rows:
        if not isinstance(selected, dict):
            continue
        for seed in clean_seeds:
            common, overrides = _with_seed(selected, seed)
            identity = {
                "stage": stage,
                "model": selected.get("model"),
                "dataset_id": selected.get("dataset_id"),
                "source_config_id": selected.get("config_id"),
                "seed": seed,
                "common": common,
                "overrides": overrides,
            }
            run_hash = stable_hash(identity, length=12)
            planned.append(
                {
                    "stage": stage,
                    "status": "planned",
                    "model": str(selected.get("model") or ""),
                    "dataset_id": str(selected.get("dataset_id") or ""),
                    "dataset_root": str(selected.get("dataset_root") or ""),
                    "selected_config_id": str(selected.get("config_id") or ""),
                    "original_search_run_id": str(selected.get("original_search_run_id") or ""),
                    "selection_rank": selected.get("rank"),
                    "selection_metric": selected_manifest.get("metric"),
                    "validation_metric_value": selected.get("validation_metric_value"),
                    "seed": seed,
                    "final_seed": seed,
                    "config_id": f"{selected.get('config_id')}_seed{seed}_{run_hash}",
                    "config_hash": run_hash,
                    "run_id": f"{stage}/{selected.get('model')}/{selected.get('dataset_id')}/{selected.get('config_id')}/seed_{seed}_{run_hash}",
                    "common": json_safe(common),
                    "overrides": json_safe(overrides),
                    "protocol_id": str(selected.get("protocol_id") or ""),
                    "protocol_hash": str(selected.get("protocol_hash") or ""),
                    "source_selected_config": json_safe(selected),
                }
            )
    return {
        "schema": ROBUST_RERUN_PLAN_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol_stage": stage,
        "final_seeds": clean_seeds,
        "selected_count": len(selected_rows),
        "planned_run_count": len(planned),
        "planned_runs": planned,
        "selection_metric": selected_manifest.get("metric"),
        "selection_mode": selected_manifest.get("mode"),
        "uses_test_metrics": False,
    }


def _csv_safe(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            safe[key] = json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False)
        else:
            safe[key] = value
    return safe


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_rows = [_csv_safe(row) for row in rows]
    fieldnames: list[str] = []
    for row in safe_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        writer.writerows(safe_rows)


def write_selection_artifacts(
    output_dir: Path,
    selected_manifest: dict[str, Any],
    robust_plan: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_json = output_dir / "selected_configs.json"
    robust_json = output_dir / "robust_rerun_plan.json"
    selected_csv = output_dir / "selected_configs.csv"
    robust_csv = output_dir / "robust_rerun_plan.csv"
    selected_json.write_text(
        json.dumps(selected_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    robust_json.write_text(
        json.dumps(robust_plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(selected_csv, list(selected_manifest.get("selected_configs") or []))
    _write_csv(robust_csv, list(robust_plan.get("planned_runs") or []))
    return {
        "selected_configs_json": str(selected_json),
        "selected_configs_csv": str(selected_csv),
        "robust_rerun_plan_json": str(robust_json),
        "robust_rerun_plan_csv": str(robust_csv),
    }


def selection_policy_from_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    selection = protocol.get("selection") if isinstance(protocol.get("selection"), dict) else {}
    top_k = protocol.get("top_k_selection") if isinstance(protocol.get("top_k_selection"), dict) else {}
    fallback_top_k = payload.get("top_k_selection") if isinstance(payload.get("top_k_selection"), dict) else {}
    metric = str(
        selection.get("metric")
        or top_k.get("metric")
        or fallback_top_k.get("metric")
        or payload.get("selection_metric")
        or "val_loss"
    ).strip()
    mode = str(selection.get("mode") or payload.get("selection_mode") or "min").strip().lower()
    k_per_model = int(top_k.get("k_per_model") or fallback_top_k.get("k_per_model") or payload.get("top_k_per_model") or 1)
    grouping = str(top_k.get("grouping") or fallback_top_k.get("grouping") or "model_dataset").strip() or "model_dataset"
    final_seeds = protocol.get("final_seeds") if isinstance(protocol.get("final_seeds"), list) else payload.get("final_seeds")
    if not isinstance(final_seeds, list):
        final_seeds = []
    return {
        "metric": metric,
        "mode": mode,
        "k_per_model": k_per_model,
        "grouping": grouping,
        "final_seeds": final_seeds,
    }
