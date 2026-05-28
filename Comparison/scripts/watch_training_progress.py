#!/usr/bin/env python3
"""Print compact Lightning training progress from TensorBoard event files."""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tensorboard.backend.event_processing import event_accumulator


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACES_ROOT = REPO_ROOT / "Comparison" / "workspaces"


def format_value(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 1:
        return f"{number:.4f}"
    return f"{number:.6f}"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def find_latest_training_dir() -> Path:
    candidates = [
        path
        for path in WORKSPACES_ROOT.rglob("training")
        if (path / "lightning_logs").exists()
    ]
    if not candidates:
        raise SystemExit(f"No training/lightning_logs found under {WORKSPACES_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def normalize_training_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "lightning_logs":
        return path.parent
    if (path / "lightning_logs").exists():
        return path
    for parent in [path, *path.parents]:
        if parent.name == "training" and (parent / "lightning_logs").exists():
            return parent
    raise SystemExit(f"Could not find lightning_logs for: {path}")


def latest_event_file(training_dir: Path) -> Path | None:
    event_files = sorted(
        (training_dir / "lightning_logs").rglob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
    )
    return event_files[-1] if event_files else None


def latest_scalar(accumulator: event_accumulator.EventAccumulator, tag: str) -> tuple[int | None, float | None]:
    tags = set(accumulator.Tags().get("scalars", []))
    if tag not in tags:
        return None, None
    values = accumulator.Scalars(tag)
    if not values:
        return None, None
    item = values[-1]
    return int(item.step), float(item.value)


def progress_line(training_dir: Path) -> str | None:
    event_file = latest_event_file(training_dir)
    if event_file is None:
        return None
    config = load_yaml(training_dir / "config.yaml")
    max_epochs = (config.get("trainer") or {}).get("max_epochs")
    accumulator = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    epoch_step, epoch = latest_scalar(accumulator, "epoch")
    train_step, train_loss = latest_scalar(accumulator, "train_loss_epoch")
    val_step, val_loss = latest_scalar(accumulator, "val_loss")
    _, train_step_loss = latest_scalar(accumulator, "train_loss_step")
    _, val_node = latest_scalar(accumulator, "val_node_smooth_l1")
    _, val_edge = latest_scalar(accumulator, "val_edge_smooth_l1")
    _, beta = latest_scalar(accumulator, "val_smooth_l1_beta")
    step = max(
        [item for item in (epoch_step, train_step, val_step) if item is not None],
        default=None,
    )
    pieces = [datetime.now().strftime("%H:%M:%S")]
    if epoch is not None:
        epoch_text = str(int(epoch))
        if max_epochs not in (None, ""):
            epoch_text += f"/{max_epochs}"
        pieces.append(f"epoch={epoch_text}")
    if step is not None:
        pieces.append(f"step={step}")
    if train_loss is not None:
        pieces.append(f"train_epoch={format_value(train_loss)}")
    if train_step_loss is not None:
        pieces.append(f"train_step={format_value(train_step_loss)}")
    if val_loss is not None:
        pieces.append(f"val={format_value(val_loss)}")
    if val_node is not None:
        pieces.append(f"val_node={format_value(val_node)}")
    if val_edge is not None:
        pieces.append(f"val_edge={format_value(val_edge)}")
    if beta is not None:
        pieces.append(f"beta={format_value(beta)}")
    checkpoints = sorted(
        (training_dir / "lightning_logs").rglob("checkpoints/*.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )
    if checkpoints:
        pieces.append(f"ckpt={checkpoints[-1].name}")
    return " | ".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="training dir or lightning_logs dir. Defaults to the latest workspace training dir.",
    )
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    training_dir = normalize_training_dir(args.path) if args.path else find_latest_training_dir()
    print(f"[watch] training_dir={training_dir}", flush=True)
    last_line = None
    while True:
        line = progress_line(training_dir)
        if line and line != last_line:
            print(line, flush=True)
            last_line = line
        elif line is None:
            print(f"{datetime.now().strftime('%H:%M:%S')} | waiting for events...", flush=True)
        if args.once:
            break
        time.sleep(max(1.0, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
