#!/usr/bin/env python3
"""Common validation-based early stopping policy for Graph2Mat-vs-DeepH."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ALLOWED_METRIC_MODES = {"min", "max"}
DEEPh_VAL_LOSS_RE = re.compile(r"Epoch #(?P<epoch>\d+).*?Val loss:\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
METRIC_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class EarlyStoppingPolicy:
    validation_metric_name: str
    metric_mode: str
    patience: int
    min_delta: float
    max_epochs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_metric_name": self.validation_metric_name,
            "metric_mode": self.metric_mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "max_epochs": self.max_epochs,
        }


@dataclass
class ValidationEvent:
    epoch: int
    value: float


@dataclass
class EarlyStoppingTracker:
    policy: EarlyStoppingPolicy
    best_epoch: int | None = None
    best_validation_value: float | None = None
    epochs_trained: int = 0
    validation_checks: int = 0
    checks_since_improvement: int = 0
    stop_triggered: bool = False
    events: list[ValidationEvent] = field(default_factory=list)

    def improved(self, value: float) -> bool:
        if self.best_validation_value is None:
            return True
        if self.policy.metric_mode == "min":
            return value < self.best_validation_value - self.policy.min_delta
        return value > self.best_validation_value + self.policy.min_delta

    def update(self, *, epoch: int, value: float) -> bool:
        if not math.isfinite(float(value)):
            raise RuntimeError(f"Validation metric is non-finite at epoch {epoch}: {value!r}")
        self.events.append(ValidationEvent(epoch=int(epoch), value=float(value)))
        self.epochs_trained = max(self.epochs_trained, int(epoch))
        self.validation_checks += 1
        if self.improved(float(value)):
            self.best_epoch = int(epoch)
            self.best_validation_value = float(value)
            self.checks_since_improvement = 0
        else:
            self.checks_since_improvement += 1
        if self.checks_since_improvement >= self.policy.patience:
            self.stop_triggered = True
        return self.stop_triggered

    def metadata(self, *, failed: bool = False, interrupted: bool = False) -> dict[str, Any]:
        if not self.events:
            raise RuntimeError(
                f"Missing validation metric {self.policy.validation_metric_name!r}; "
                "early stopping/checkpoint selection must fail closed."
            )
        if failed:
            stop_reason = "failed"
        elif interrupted:
            stop_reason = "interrupted"
        elif self.stop_triggered:
            stop_reason = "early_stopping"
        elif self.epochs_trained >= self.policy.max_epochs:
            stop_reason = "max_epochs"
        else:
            stop_reason = "completed"
        return {
            **self.policy.to_dict(),
            "best_epoch": self.best_epoch,
            "best_validation_value": self.best_validation_value,
            "epochs_trained": self.epochs_trained,
            "validation_checks": self.validation_checks,
            "checks_since_improvement": self.checks_since_improvement,
            "stop_reason": stop_reason,
        }


class DeepHEarlyStoppingObserver:
    def __init__(self, policy: EarlyStoppingPolicy):
        self.tracker = EarlyStoppingTracker(policy)
        self.stop_reason: str | None = None

    def __call__(self, line: str) -> str | None:
        event = parse_deeph_validation_line(line)
        if event is None:
            return None
        if self.tracker.update(epoch=event.epoch, value=event.value):
            self.stop_reason = (
                f"early_stopping: {self.tracker.checks_since_improvement} validation checks "
                f"without improvement in {self.tracker.policy.validation_metric_name}"
            )
            return self.stop_reason
        return None

    def metadata(self) -> dict[str, Any]:
        return self.tracker.metadata()


def _references_test_metric(metric: str) -> bool:
    return "test" in [token.lower() for token in METRIC_TOKEN_RE.findall(metric)]


def parse_early_stopping_policy(payload: dict[str, Any] | None) -> EarlyStoppingPolicy | None:
    payload = payload or {}
    raw = payload.get("early_stopping")
    if raw in (None, "", False):
        return None
    if not isinstance(raw, dict):
        raise RuntimeError("early_stopping must be an object.")
    if raw.get("enabled") is False:
        return None
    metric = str(raw.get("metric") or raw.get("validation_metric_name") or "").strip()
    if not metric:
        raise RuntimeError("early_stopping.metric is required.")
    if _references_test_metric(metric):
        raise RuntimeError("early_stopping.metric must not reference test metrics.")
    mode = str(raw.get("mode") or raw.get("metric_mode") or "").strip().lower()
    if mode not in ALLOWED_METRIC_MODES:
        raise RuntimeError("early_stopping.mode must be min or max.")
    try:
        patience = int(raw["patience"])
        min_delta = float(raw["min_delta"])
        max_epochs = int(raw["max_epochs"])
    except KeyError as exc:
        raise RuntimeError(f"early_stopping.{exc.args[0]} is required.") from exc
    except (TypeError, ValueError) as exc:
        raise RuntimeError("early_stopping.patience, min_delta and max_epochs must be numeric.") from exc
    if patience <= 0:
        raise RuntimeError("early_stopping.patience must be positive.")
    if min_delta < 0:
        raise RuntimeError("early_stopping.min_delta must be non-negative.")
    if max_epochs <= 0:
        raise RuntimeError("early_stopping.max_epochs must be positive.")
    return EarlyStoppingPolicy(
        validation_metric_name=metric,
        metric_mode=mode,
        patience=patience,
        min_delta=min_delta,
        max_epochs=max_epochs,
    )


def parse_deeph_validation_line(line: str) -> ValidationEvent | None:
    match = DEEPh_VAL_LOSS_RE.search(line)
    if not match:
        return None
    return ValidationEvent(epoch=int(match.group("epoch")), value=float(match.group("value")))


def graph2mat_early_stopping_callbacks(policy: EarlyStoppingPolicy) -> list[dict[str, Any]]:
    return [
        {
            "class_path": "EarlyStopping",
            "init_args": {
                "monitor": policy.validation_metric_name,
                "mode": policy.metric_mode,
                "patience": policy.patience,
                "min_delta": policy.min_delta,
                "strict": True,
            },
        }
    ]


def tensorboard_policy_metadata(training_dir: Path, policy: EarlyStoppingPolicy) -> dict[str, Any]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        raise RuntimeError("TensorBoard event reader is required for Graph2Mat early stopping metadata.") from exc
    event_files = sorted(
        (training_dir / "lightning_logs").rglob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not event_files:
        raise RuntimeError(
            f"Missing validation metric {policy.validation_metric_name!r}; no TensorBoard event files under {training_dir}."
        )
    tracker = EarlyStoppingTracker(policy)
    found = False
    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = set(accumulator.Tags().get("scalars", []))
        if policy.validation_metric_name not in tags:
            continue
        epoch_by_step = {}
        if "epoch" in tags:
            epoch_by_step = {
                int(item.step): int(float(item.value))
                for item in accumulator.Scalars("epoch")
            }
        for item in accumulator.Scalars(policy.validation_metric_name):
            found = True
            epoch = epoch_by_step.get(int(item.step), int(item.step))
            tracker.update(epoch=epoch, value=float(item.value))
    if not found:
        raise RuntimeError(
            f"Missing validation metric {policy.validation_metric_name!r}; "
            f"event files did not contain that scalar under {training_dir}."
        )
    return tracker.metadata()
