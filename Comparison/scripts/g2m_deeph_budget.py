#!/usr/bin/env python3
"""Search budget accounting for Graph2Mat-vs-DeepH training sweeps."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BUDGET_SCHEMA = "graph2mat_deeph_search_budget_v1"
MODELS = ("graph2mat", "deeph")
BUDGET_MODE_NONE = "uncontrolled"
BUDGET_MODE_EQUAL_N_TRIALS = "equal_n_trials"
BUDGET_MODE_EQUAL_GPU_HOURS = "equal_gpu_hours_per_model"


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def record_gpu_hours(record: dict[str, Any]) -> float | None:
    telemetry = record.get("telemetry") if isinstance(record.get("telemetry"), dict) else {}
    value = finite_number(telemetry.get("gpu_hours_total")) if telemetry else None
    if value is not None:
        return value
    if record.get("telemetry_path"):
        payload = _read_json(Path(str(record["telemetry_path"])))
        value = finite_number(payload.get("gpu_hours_total"))
        if value is not None:
            return value
    return None


def normalize_budget_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    policy = policy if isinstance(policy, dict) else {}
    mode = str(policy.get("mode") or BUDGET_MODE_NONE).strip().lower()
    if mode not in {BUDGET_MODE_NONE, BUDGET_MODE_EQUAL_N_TRIALS, BUDGET_MODE_EQUAL_GPU_HOURS}:
        raise RuntimeError("budget_policy.mode must be equal_n_trials or equal_gpu_hours_per_model.")
    normalized: dict[str, Any] = {"mode": mode}
    if mode == BUDGET_MODE_EQUAL_N_TRIALS:
        n_trials = policy.get("n_trials_per_model")
        if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials <= 0:
            raise RuntimeError("budget_policy.n_trials_per_model must be a positive integer.")
        normalized["n_trials_per_model"] = int(n_trials)
    elif mode == BUDGET_MODE_EQUAL_GPU_HOURS:
        gpu_hours = finite_number(policy.get("gpu_hours_per_model"))
        if gpu_hours is None or gpu_hours <= 0:
            raise RuntimeError("budget_policy.gpu_hours_per_model must be a positive number.")
        normalized["gpu_hours_per_model"] = gpu_hours
        normalized["edge_policy"] = str(
            policy.get("edge_policy")
            or "allow_overshoot_by_one_scheduler_batch"
        )
    return normalized


@dataclass
class BudgetTracker:
    budget_policy: dict[str, Any] | None = None
    consumed_gpu_hours_by_model: dict[str, float] = field(default_factory=lambda: {model: 0.0 for model in MODELS})
    completed_trials_by_model: dict[str, int] = field(default_factory=lambda: {model: 0 for model in MODELS})
    reserved_trials_by_model: dict[str, int] = field(default_factory=lambda: {model: 0 for model in MODELS})
    skipped_trials_by_model: dict[str, int] = field(default_factory=lambda: {model: 0 for model in MODELS})
    budget_exhaustion_reason: dict[str, str] = field(default_factory=dict)
    accounting_errors: list[str] = field(default_factory=list)
    skipped_runs: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.budget_policy = normalize_budget_policy(self.budget_policy)

    @property
    def mode(self) -> str:
        return str((self.budget_policy or {}).get("mode") or BUDGET_MODE_NONE)

    @property
    def gpu_hours_per_model(self) -> float | None:
        return finite_number((self.budget_policy or {}).get("gpu_hours_per_model"))

    def add_completed(self, record: dict[str, Any], *, source: str = "completed") -> None:
        model = str(record.get("model") or "")
        if model not in MODELS or record.get("status") != "completed":
            return
        if self.reserved_trials_by_model[model] > 0:
            self.reserved_trials_by_model[model] -= 1
        self.completed_trials_by_model[model] += 1
        gpu_hours = record_gpu_hours(record)
        if gpu_hours is None:
            message = (
                f"Missing gpu_hours_total for completed {model} run "
                f"{record.get('config_id') or '<unknown>'}; budget accounting is incomplete."
            )
            if self.mode == BUDGET_MODE_EQUAL_GPU_HOURS:
                self.accounting_errors.append(message)
                raise RuntimeError(message)
            self.accounting_errors.append(message)
            return
        self.consumed_gpu_hours_by_model[model] += gpu_hours
        limit = self.gpu_hours_per_model
        if self.mode == BUDGET_MODE_EQUAL_GPU_HOURS and limit is not None:
            consumed = self.consumed_gpu_hours_by_model[model]
            if consumed >= limit:
                self.budget_exhaustion_reason[model] = (
                    f"consumed_gpu_hours {consumed:.6g} reached budget {limit:.6g} "
                    f"after {source}; no new {model} trials will be scheduled"
                )

    def add_completed_many(self, records: list[dict[str, Any]], *, source: str = "completed") -> None:
        for record in records:
            self.add_completed(record, source=source)

    def can_schedule(self, record: dict[str, Any]) -> bool:
        model = str(record.get("model") or "")
        if model not in MODELS:
            return True
        if self.mode == BUDGET_MODE_EQUAL_N_TRIALS:
            limit = int((self.budget_policy or {}).get("n_trials_per_model") or 0)
            if self.completed_trials_by_model[model] + self.reserved_trials_by_model[model] >= limit:
                self.budget_exhaustion_reason.setdefault(
                    model,
                    f"scheduled_trials reached n_trials_per_model={limit}",
                )
                return False
        if self.mode == BUDGET_MODE_EQUAL_GPU_HOURS:
            limit = self.gpu_hours_per_model
            if limit is not None and self.consumed_gpu_hours_by_model[model] >= limit:
                self.budget_exhaustion_reason.setdefault(
                    model,
                    f"consumed_gpu_hours {self.consumed_gpu_hours_by_model[model]:.6g} "
                    f"reached budget {limit:.6g}",
                )
                return False
        return True

    def reserve(self, record: dict[str, Any]) -> None:
        model = str(record.get("model") or "")
        if model in MODELS:
            self.reserved_trials_by_model[model] += 1

    def release(self, record: dict[str, Any]) -> None:
        model = str(record.get("model") or "")
        if model in MODELS and self.reserved_trials_by_model[model] > 0:
            self.reserved_trials_by_model[model] -= 1

    def skip_for_budget(self, record: dict[str, Any]) -> dict[str, Any]:
        model = str(record.get("model") or "")
        limit = self.gpu_hours_per_model
        reason = self.budget_exhaustion_reason.get(model)
        if not reason:
            reason = (
                f"{model} GPU-hour budget exhausted"
                if limit is not None
                else f"{model} budget exhausted"
            )
        skipped = {
            **record,
            "status": "skipped_budget_exhausted",
            "budget_skip_reason": reason,
        }
        if model in MODELS:
            self.skipped_trials_by_model[model] += 1
        self.skipped_runs.append(skipped)
        return skipped

    def summary(self) -> dict[str, Any]:
        limit = self.gpu_hours_per_model
        return {
            "schema": BUDGET_SCHEMA,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "budget_policy": self.budget_policy,
            "budget_gpu_hours_per_model": limit,
            "consumed_gpu_hours_by_model": dict(self.consumed_gpu_hours_by_model),
            "completed_trials_by_model": dict(self.completed_trials_by_model),
            "reserved_trials_by_model": dict(self.reserved_trials_by_model),
            "skipped_trials_by_model": dict(self.skipped_trials_by_model),
            "budget_exhaustion_reason": dict(self.budget_exhaustion_reason),
            "budget_accounting_status": "failed" if self.accounting_errors else "complete",
            "budget_accounting_errors": list(self.accounting_errors),
            "edge_policy": (self.budget_policy or {}).get("edge_policy", ""),
            "skipped_runs": list(self.skipped_runs),
        }


def write_budget_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
