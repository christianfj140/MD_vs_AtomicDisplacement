#!/usr/bin/env python3
"""Paper-ready protocol validation for the Graph2Mat-vs-DeepH benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from g2m_deeph_training_sweep import DEEPH_KEYS, FORBIDDEN_DEEPH_KEYS, GRAPH2MAT_KEYS, json_safe
from graph2mat_sweep_config import GRAPH2MAT_READOUT_FAMILIES


SCHEMA_NAME = "graph2mat_deeph_benchmark_protocol_v1"
ALLOWED_MODELS = {"graph2mat", "deeph"}
ALLOWED_SEARCH_STRATEGIES = {"grid", "manual", "random", "latin_hypercube"}
ALLOWED_BUDGET_MODES = {"equal_n_trials", "equal_gpu_hours_per_model"}
ALLOWED_MODES = {"min", "max"}
ALLOWED_FINAL_TEST_POLICIES = {"locked_until_final"}
ALLOWED_DEEPH_EQUIVALENCE_POLICIES = {"fail_closed_unless_proven"}
SUPPORTED_DEEPH_CRITERIA = {"MaskMSELoss"}
SUPPORTED_DEEPH_OPTIMIZERS = {"sgd", "sgdm", "adam", "adamW", "adagrad", "RMSprop", "lbfgs"}
DISALLOWED_FINAL_CLAIM_METRICS = {
    "loss",
    "metric_value",
    "test_loss",
    "train_loss",
    "training_loss",
    "val_loss",
    "validation_loss",
    "validation_metric",
}
ALLOWED_FINAL_CLAIM_METRICS = {
    "dos_mae_500_fermi_window",
    "dos_mae_500_fermi_window_mean",
    "dos_wasserstein_eV",
    "dos_wasserstein_eV_mean",
    "fermi_window_rmse_eV",
    "fermi_window_rmse_eV_mean",
    "frontier_window_rmse_eV",
    "frontier_window_rmse_eV_mean",
    "global_rmse_eV",
    "global_rmse_eV_mean",
    "h_mae_eV",
    "h_mae_eV_mean",
    "h_mse_eV",
    "h_mse_eV_mean",
    "h_rmse_eV",
    "h_rmse_eV_mean",
    "low_energy_rmse_eV",
    "low_energy_rmse_eV_mean",
    "relative_frobenius",
    "relative_frobenius_mean",
    "spectral_composite_score",
    "spectral_composite_score_mean",
    "test_spectral_composite_score",
    "test_spectral_composite_score_mean",
}
VALIDATION_COMPOSITE_METRICS = {"val_spectral_composite"}

REQUIRED_REFERENCE_ARTIFACTS = {
    "RUN.fdf",
    "SystemLabel.TSHS",
    "SystemLabel.TSDE",
    "SystemLabel.HSX",
    "SystemLabel.STRUCT_OUT",
    "SystemLabel.XV",
    "SystemLabel.ORB_INDX",
    "metadata.json",
}
REQUIRED_TELEMETRY_FIELDS = {
    "wall_clock_seconds",
    "gpu_hours",
    "peak_gpu_memory_mb",
    "samples_per_second",
    "matrix_blocks_per_second",
    "best_validation_epoch",
}
REQUIRED_TOP_LEVEL_FIELDS = {
    "protocol_id",
    "version",
    "datasets",
    "reference_artifacts",
    "models",
    "selection",
    "early_stopping",
    "search_policy",
    "budget_policy",
    "final_seeds",
    "top_k_selection",
    "final_evaluation",
    "final_test_policy",
    "required_telemetry",
    "deeph_comparability",
}
PROTOCOL_GRAPH2MAT_KEYS = GRAPH2MAT_KEYS | {
    "optim_lr",
    "store_in_memory",
    "accelerator",
    "log_every_n_steps",
}


def _require_object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{field} must be an object.")
    return value


def _require_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{field} must be a list.")
    if not value:
        raise RuntimeError(f"{field} must not be empty.")
    return value


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if value is None:
        raise RuntimeError(f"{field} is required.")
    text = str(value).strip()
    if not text:
        raise RuntimeError(f"{field} must not be empty.")
    return text


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{field} must be a positive integer.")
    return value


def _require_nonnegative_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise RuntimeError(f"{field} must be a non-negative number.")
    return float(value)


def _contains_test_scope(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "test"
    if isinstance(value, dict):
        return any(_contains_test_scope(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_test_scope(item) for item in value)
    return False


def _numeric_values_from_space_spec(value: Any) -> list[float]:
    if isinstance(value, dict):
        if "choices" in value and isinstance(value["choices"], list):
            raw_values = value["choices"]
        elif "value" in value:
            raw_values = [value["value"]]
        elif "fixed" in value:
            raw_values = [value["fixed"]]
        else:
            raw_values = [value.get("min", value.get("low"))]
    elif isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]
    numbers: list[float] = []
    for item in raw_values:
        if isinstance(item, bool):
            continue
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            continue
    return numbers


def _values_from_space_spec(value: Any) -> list[Any]:
    if isinstance(value, dict):
        if "choices" in value and isinstance(value["choices"], list):
            return list(value["choices"])
        if "value" in value:
            return [value["value"]]
        if "fixed" in value:
            return [value["fixed"]]
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def protocol_hash(protocol: dict[str, Any]) -> str:
    """Return a stable hash for a validated or raw protocol dictionary."""
    payload = copy.deepcopy(protocol)
    payload.pop("protocol_hash", None)
    encoded = json.dumps(json_safe(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_datasets(protocol: dict[str, Any]) -> None:
    datasets = _require_list(protocol.get("datasets"), field="datasets")
    seen: set[str] = set()
    for index, item in enumerate(datasets):
        dataset = _require_object(item, field=f"datasets[{index}]")
        dataset_id = _require_nonempty_string(dataset.get("dataset_id"), field=f"datasets[{index}].dataset_id")
        if dataset_id in seen:
            raise RuntimeError(f"Duplicate protocol dataset_id: {dataset_id}")
        seen.add(dataset_id)
        for key in ("dataset_root", "frozen_split_manifest", "benchmark_dataset_manifest"):
            _require_nonempty_string(dataset.get(key), field=f"datasets[{index}].{key}")


def _validate_reference_artifacts(protocol: dict[str, Any]) -> None:
    section = _require_object(protocol.get("reference_artifacts"), field="reference_artifacts")
    required = set(str(item) for item in _require_list(section.get("required"), field="reference_artifacts.required"))
    missing = sorted(REQUIRED_REFERENCE_ARTIFACTS - required)
    if missing:
        raise RuntimeError("reference_artifacts.required is missing: " + ", ".join(missing))
    forbidden_reference = str(section.get("forbid_as_reference") or "")
    if "ML_prediction.HSX" not in forbidden_reference and "ML_prediction.HSX" not in set(section.get("forbidden", [])):
        raise RuntimeError("reference_artifacts must explicitly forbid ML_prediction.HSX as reference.")


def _validate_models(protocol: dict[str, Any]) -> None:
    models = _require_object(protocol.get("models"), field="models")
    missing = sorted(ALLOWED_MODELS - set(models))
    if missing:
        raise RuntimeError("models is missing required model sections: " + ", ".join(missing))
    for model_name in sorted(ALLOWED_MODELS):
        model = _require_object(models.get(model_name), field=f"models.{model_name}")
        if model.get("enabled") is not True:
            raise RuntimeError(f"models.{model_name}.enabled must be true for the final benchmark protocol.")
        search_space = _require_object(model.get("search_space"), field=f"models.{model_name}.search_space")
        if not search_space:
            raise RuntimeError(f"models.{model_name}.search_space must not be empty.")
        allowed = PROTOCOL_GRAPH2MAT_KEYS if model_name == "graph2mat" else DEEPH_KEYS
        forbidden = sorted(set(search_space) & FORBIDDEN_DEEPH_KEYS) if model_name == "deeph" else []
        if forbidden:
            raise RuntimeError(
                "models.deeph.search_space cannot change split/preprocess/physics keys: "
                + ", ".join(forbidden)
            )
        unknown = sorted(set(search_space) - allowed - FORBIDDEN_DEEPH_KEYS)
        if unknown:
            raise RuntimeError(f"models.{model_name}.search_space has unsupported keys: {', '.join(unknown)}.")
        if model_name == "graph2mat":
            batch_values = _numeric_values_from_space_spec(search_space.get("batch_size"))
            if not batch_values or min(batch_values) > 128:
                raise RuntimeError(
                    "models.graph2mat.search_space.batch_size must include at least one small/medium "
                    "batch size <= 128 for paper-ready fairness."
                )
            if "readout" in search_space:
                raw_readout_values = _values_from_space_spec(search_space.get("readout"))
                if not raw_readout_values:
                    raise RuntimeError(
                        "models.graph2mat.search_space.readout must use choices, value, or fixed."
                    )
                readout_values = {
                    str(item).strip().lower()
                    for item in raw_readout_values
                    if str(item).strip()
                }
                unsupported = sorted(readout_values - GRAPH2MAT_READOUT_FAMILIES)
                if unsupported:
                    raise RuntimeError(
                        "models.graph2mat.search_space.readout has unsupported values: "
                        + ", ".join(unsupported)
                        + ". Use one of: "
                        + ", ".join(sorted(GRAPH2MAT_READOUT_FAMILIES))
                        + "."
                    )
        else:
            if "optimizer" in search_space:
                raw_optimizer_values = _values_from_space_spec(search_space.get("optimizer"))
                optimizer_values = {
                    str(item).strip()
                    for item in raw_optimizer_values
                    if str(item).strip()
                }
                unsupported = sorted(optimizer_values - SUPPORTED_DEEPH_OPTIMIZERS)
                if unsupported:
                    raise RuntimeError(
                        "models.deeph.search_space.optimizer has unsupported values for the "
                        "current DeepH-pack hamiltonian target: "
                        + ", ".join(unsupported)
                        + ". Use one of: "
                        + ", ".join(sorted(SUPPORTED_DEEPH_OPTIMIZERS))
                        + "."
                    )
            if "criterion" in search_space:
                raw_criterion_values = _values_from_space_spec(search_space.get("criterion"))
                criterion_values = {
                    str(item).strip()
                    for item in raw_criterion_values
                    if str(item).strip()
                }
                unsupported = sorted(criterion_values - SUPPORTED_DEEPH_CRITERIA)
                if unsupported:
                    raise RuntimeError(
                        "models.deeph.search_space.criterion has unsupported values for the "
                        "current DeepH-pack hamiltonian target: "
                        + ", ".join(unsupported)
                        + ". Use one of: "
                        + ", ".join(sorted(SUPPORTED_DEEPH_CRITERIA))
                        + "."
                    )


def _validate_selection(protocol: dict[str, Any]) -> str:
    section = _require_object(protocol.get("selection"), field="selection")
    metric = _require_nonempty_string(section.get("metric"), field="selection.metric")
    mode = _require_nonempty_string(section.get("mode"), field="selection.mode")
    if mode not in ALLOWED_MODES:
        raise RuntimeError(f"selection.mode must be one of: {', '.join(sorted(ALLOWED_MODES))}.")
    split = _require_nonempty_string(section.get("split"), field="selection.split")
    if split != "validation":
        raise RuntimeError("selection.split must be validation; test metrics cannot select configs.")
    source = _require_nonempty_string(section.get("source"), field="selection.source")
    if source != "validation_only":
        raise RuntimeError("selection.source must be validation_only.")
    if _contains_test_scope(section):
        raise RuntimeError("selection must not reference test metrics.")
    return metric


def _validate_early_stopping(protocol: dict[str, Any], *, selection_metric: str) -> None:
    section = _require_object(protocol.get("early_stopping"), field="early_stopping")
    metric = _require_nonempty_string(section.get("metric"), field="early_stopping.metric")
    if metric != selection_metric and not (
        selection_metric in VALIDATION_COMPOSITE_METRICS
        and metric in {"val_loss", "validation_loss"}
    ):
        raise RuntimeError("early_stopping.metric must match selection.metric for the final protocol.")
    mode = _require_nonempty_string(section.get("mode"), field="early_stopping.mode")
    if mode not in ALLOWED_MODES:
        raise RuntimeError(f"early_stopping.mode must be one of: {', '.join(sorted(ALLOWED_MODES))}.")
    _require_positive_int(section.get("patience"), field="early_stopping.patience")
    _require_nonnegative_number(section.get("min_delta"), field="early_stopping.min_delta")
    _require_positive_int(section.get("max_epochs"), field="early_stopping.max_epochs")


def _validate_search_policy(protocol: dict[str, Any]) -> None:
    section = _require_object(protocol.get("search_policy"), field="search_policy")
    strategy = _require_nonempty_string(section.get("strategy"), field="search_policy.strategy")
    if strategy not in ALLOWED_SEARCH_STRATEGIES:
        raise RuntimeError(
            "search_policy.strategy must be one of: " + ", ".join(sorted(ALLOWED_SEARCH_STRATEGIES)) + "."
        )
    if strategy == "manual":
        manual = _require_object(protocol.get("manual_search_plan"), field="manual_search_plan")
        rows = _require_list(manual.get("planned_runs"), field="manual_search_plan.planned_runs")
        for index, row in enumerate(rows):
            item = _require_object(row, field=f"manual_search_plan.planned_runs[{index}]")
            model = _require_nonempty_string(item.get("model"), field=f"manual_search_plan.planned_runs[{index}].model")
            if model not in ALLOWED_MODELS:
                raise RuntimeError(f"manual_search_plan.planned_runs[{index}].model must be graph2mat or deeph.")
            _require_nonempty_string(item.get("config_id") or item.get("id"), field=f"manual_search_plan.planned_runs[{index}].config_id")
            overrides = _require_object(item.get("overrides"), field=f"manual_search_plan.planned_runs[{index}].overrides")
            if model == "deeph" and "criterion" in overrides:
                criterion = str(overrides.get("criterion") or "").strip()
                if criterion not in SUPPORTED_DEEPH_CRITERIA:
                    raise RuntimeError(
                        f"manual_search_plan.planned_runs[{index}].overrides.criterion "
                        "has unsupported values for the current DeepH-pack hamiltonian target: "
                        f"{criterion!r}. Use one of: {', '.join(sorted(SUPPORTED_DEEPH_CRITERIA))}."
                    )
            if model == "deeph" and "optimizer" in overrides:
                optimizer = str(overrides.get("optimizer") or "").strip()
                if optimizer not in SUPPORTED_DEEPH_OPTIMIZERS:
                    raise RuntimeError(
                        f"manual_search_plan.planned_runs[{index}].overrides.optimizer "
                        "has unsupported values for the current DeepH-pack hamiltonian target: "
                        f"{optimizer!r}. Use one of: {', '.join(sorted(SUPPORTED_DEEPH_OPTIMIZERS))}."
                    )
    elif strategy in {"random", "latin_hypercube"}:
        _require_positive_int(section.get("n_trials_per_model"), field="search_policy.n_trials_per_model")
        if "random_seed" not in section:
            raise RuntimeError("search_policy.random_seed is required for randomized search strategies.")
    elif not any(key in section for key in ("max_configs_per_model", "n_trials_per_model", "gpu_hours_per_model")):
        raise RuntimeError("search_policy.grid requires max_configs_per_model, n_trials_per_model, or gpu_hours_per_model.")


def _validate_budget_policy(protocol: dict[str, Any]) -> None:
    section = _require_object(protocol.get("budget_policy"), field="budget_policy")
    mode = _require_nonempty_string(section.get("mode"), field="budget_policy.mode")
    if mode not in ALLOWED_BUDGET_MODES:
        raise RuntimeError("budget_policy.mode must be equal_n_trials or equal_gpu_hours_per_model.")
    if mode == "equal_n_trials":
        _require_positive_int(section.get("n_trials_per_model"), field="budget_policy.n_trials_per_model")
    else:
        value = section.get("gpu_hours_per_model")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise RuntimeError("budget_policy.gpu_hours_per_model must be a positive number.")


def _validate_final_seeds(protocol: dict[str, Any]) -> None:
    seeds = _require_list(protocol.get("final_seeds"), field="final_seeds")
    if len(seeds) < 3:
        raise RuntimeError("final_seeds must include at least 3 seeds for the final benchmark protocol.")
    for index, seed in enumerate(seeds):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise RuntimeError(f"final_seeds[{index}] must be a non-negative integer.")
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("final_seeds must not contain duplicates.")


def _validate_top_k_selection(protocol: dict[str, Any], *, selection_metric: str) -> None:
    section = _require_object(protocol.get("top_k_selection"), field="top_k_selection")
    _require_positive_int(section.get("k_per_model"), field="top_k_selection.k_per_model")
    split = _require_nonempty_string(section.get("split"), field="top_k_selection.split")
    if split != "validation":
        raise RuntimeError("top_k_selection.split must be validation; test metrics cannot select top-k configs.")
    metric = _require_nonempty_string(section.get("metric"), field="top_k_selection.metric")
    if metric != selection_metric:
        raise RuntimeError("top_k_selection.metric must match selection.metric.")
    if section.get("uses_test_metrics") is not False:
        raise RuntimeError("top_k_selection.uses_test_metrics must be false.")
    if _contains_test_scope(section):
        raise RuntimeError("top_k_selection must not reference test metrics.")


def _validate_final_metric_name(metric: str, *, field: str) -> None:
    normalized = metric.strip()
    if normalized in DISALLOWED_FINAL_CLAIM_METRICS:
        raise RuntimeError(f"{field} must be a scientific final metric, not {normalized}.")
    if normalized not in ALLOWED_FINAL_CLAIM_METRICS:
        raise RuntimeError(
            f"{field} is unsupported for final scientific claims: {normalized}. "
            "Use one of: " + ", ".join(sorted(ALLOWED_FINAL_CLAIM_METRICS)) + "."
        )


def _validate_final_evaluation(protocol: dict[str, Any]) -> None:
    section = _require_object(protocol.get("final_evaluation"), field="final_evaluation")
    primary_metric = _require_nonempty_string(
        section.get("primary_metric"),
        field="final_evaluation.primary_metric",
    )
    _validate_final_metric_name(primary_metric, field="final_evaluation.primary_metric")
    mode = _require_nonempty_string(section.get("mode"), field="final_evaluation.mode")
    if mode not in ALLOWED_MODES:
        raise RuntimeError(f"final_evaluation.mode must be one of: {', '.join(sorted(ALLOWED_MODES))}.")
    secondary = section.get("secondary_metrics", [])
    if secondary is None:
        secondary = []
    if not isinstance(secondary, list):
        raise RuntimeError("final_evaluation.secondary_metrics must be a list.")
    for index, metric in enumerate(secondary):
        name = _require_nonempty_string(metric, field=f"final_evaluation.secondary_metrics[{index}]")
        _validate_final_metric_name(name, field=f"final_evaluation.secondary_metrics[{index}]")
    practical = section.get("practical_match")
    if practical is not None:
        practical_section = _require_object(practical, field="final_evaluation.practical_match")
        if practical_section.get("relative_gap_max") is not None:
            value = practical_section.get("relative_gap_max")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise RuntimeError("final_evaluation.practical_match.relative_gap_max must be a positive number.")
        if practical_section.get("absolute_gap_meV_max") is not None:
            _require_nonnegative_number(
                practical_section.get("absolute_gap_meV_max"),
                field="final_evaluation.practical_match.absolute_gap_meV_max",
            )
        if "requires_cost_noninferior" in practical_section and not isinstance(
            practical_section.get("requires_cost_noninferior"),
            bool,
        ):
            raise RuntimeError("final_evaluation.practical_match.requires_cost_noninferior must be boolean.")


def _validate_final_test_policy(protocol: dict[str, Any]) -> None:
    section = _require_object(protocol.get("final_test_policy"), field="final_test_policy")
    policy = _require_nonempty_string(section.get("policy"), field="final_test_policy.policy")
    if policy not in ALLOWED_FINAL_TEST_POLICIES:
        raise RuntimeError("final_test_policy.policy must be locked_until_final.")
    if section.get("locked_during_search") is not True:
        raise RuntimeError("final_test_policy.locked_during_search must be true.")
    if section.get("evaluate_once_after_selection") is not True:
        raise RuntimeError("final_test_policy.evaluate_once_after_selection must be true.")
    test_split = _require_nonempty_string(section.get("test_split"), field="final_test_policy.test_split")
    if test_split != "test":
        raise RuntimeError("final_test_policy.test_split must be test.")


def _validate_required_telemetry(protocol: dict[str, Any]) -> None:
    fields = set(str(item) for item in _require_list(protocol.get("required_telemetry"), field="required_telemetry"))
    missing = sorted(REQUIRED_TELEMETRY_FIELDS - fields)
    if missing:
        raise RuntimeError("required_telemetry is missing: " + ", ".join(missing))


def _validate_deeph_comparability(protocol: dict[str, Any]) -> None:
    section = _require_object(protocol.get("deeph_comparability"), field="deeph_comparability")
    policy = _require_nonempty_string(
        section.get("adapter_equivalence_policy"),
        field="deeph_comparability.adapter_equivalence_policy",
    )
    if policy not in ALLOWED_DEEPH_EQUIVALENCE_POLICIES:
        raise RuntimeError("deeph_comparability.adapter_equivalence_policy must be fail_closed_unless_proven.")
    if section.get("robust_winner_requires_proven_equivalence") is not True:
        raise RuntimeError("deeph_comparability.robust_winner_requires_proven_equivalence must be true.")
    if section.get("diagnostic_if_unproven") is not True:
        raise RuntimeError("deeph_comparability.diagnostic_if_unproven must be true.")


def validate_protocol(value: Any) -> dict[str, Any]:
    """Validate and return a canonical copy of a final benchmark protocol."""
    protocol = copy.deepcopy(_require_object(value, field="protocol"))
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(protocol))
    if missing:
        raise RuntimeError("Missing protocol fields: " + ", ".join(missing))
    protocol_id = _require_nonempty_string(protocol.get("protocol_id"), field="protocol_id")
    version = _require_nonempty_string(protocol.get("version"), field="version")
    protocol["protocol_id"] = protocol_id
    protocol["version"] = version
    if "schema" in protocol and protocol["schema"] != SCHEMA_NAME:
        raise RuntimeError(f"schema must be {SCHEMA_NAME}.")

    _validate_datasets(protocol)
    _validate_reference_artifacts(protocol)
    _validate_models(protocol)
    selection_metric = _validate_selection(protocol)
    _validate_early_stopping(protocol, selection_metric=selection_metric)
    _validate_search_policy(protocol)
    _validate_budget_policy(protocol)
    _validate_final_seeds(protocol)
    _validate_top_k_selection(protocol, selection_metric=selection_metric)
    _validate_final_evaluation(protocol)
    _validate_final_test_policy(protocol)
    _validate_required_telemetry(protocol)
    _validate_deeph_comparability(protocol)

    protocol["schema"] = SCHEMA_NAME
    protocol["protocol_hash"] = protocol_hash(protocol)
    return protocol


def load_protocol(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_protocol(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Graph2Mat-vs-DeepH paper benchmark protocol JSON.")
    parser.add_argument("protocol", type=Path, help="Path to the protocol JSON file.")
    parser.add_argument("--print-json", action="store_true", help="Print the canonical validated protocol JSON.")
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    if args.print_json:
        print(json.dumps(protocol, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "schema": protocol["schema"],
                    "protocol_id": protocol["protocol_id"],
                    "protocol_hash": protocol["protocol_hash"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
