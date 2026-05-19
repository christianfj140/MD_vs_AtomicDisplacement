#!/usr/bin/env python3
"""Diagnose one-sample H2O overfit before MatrixWriter reconstruction.

This script loads the isolated one-sample H-only overfit checkpoint, rebuilds the
same train datamodule, runs ``model(batch)`` directly, and compares predicted
node/edge labels against the training labels.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TORCH_COMPAT_DIR = REPO_ROOT / "scripts" / "torch_serialization_compat"
sys.path.insert(0, str(TORCH_COMPAT_DIR))

from torch_safe_globals import allow_graph2mat_checkpoint_globals  # noqa: E402

allow_graph2mat_checkpoint_globals()

from graph2mat.tools.lightning import MatrixDataModule  # noqa: E402
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel  # noqa: E402


DEFAULT_WORKSPACE = (
    REPO_ROOT
    / "Comparison"
    / "results"
    / "diagnostics"
    / "h2o_hamiltonian"
    / "one_sample_overfit_h_only"
)


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def latest_checkpoint(training_dir: Path) -> Path:
    candidates = sorted(
        (training_dir / "logs" / "one_sample_h_only").glob("version_*/checkpoints/best-*.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]
    fallback = training_dir / "logs" / "one_sample_h_only" / "version_0" / "checkpoints" / "last.ckpt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "No one-sample checkpoint found under "
        f"{training_dir / 'logs' / 'one_sample_h_only'}"
    )


def tensor_to_float_list(tensor: torch.Tensor, limit: int | None = None) -> list[float]:
    flat = tensor.detach().cpu().double().reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    return [float(value) for value in flat.tolist()]


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def compare_labels(prediction: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    pred = prediction.detach().cpu().double().reshape(-1)
    ref = reference.detach().cpu().double().reshape(-1)
    shape_match = tuple(prediction.shape) == tuple(reference.shape)
    comparable = shape_match and pred.numel() == ref.numel()
    result: dict[str, Any] = {
        "prediction_shape": list(prediction.shape),
        "reference_shape": list(reference.shape),
        "shape_match": shape_match,
        "n_values": int(ref.numel()),
        "comparable": comparable,
    }
    if not comparable:
        result.update(
            {
                "mae": None,
                "rmse": None,
                "max_abs": None,
                "first20": [],
            }
        )
        return result

    error = pred - ref
    abs_error = error.abs()
    rmse = torch.sqrt(torch.mean(error * error))
    result.update(
        {
            "mae": finite_or_none(float(abs_error.mean().item())),
            "rmse": finite_or_none(float(rmse.item())),
            "max_abs": finite_or_none(float(abs_error.max().item())),
            "first20": [
                {
                    "index": index,
                    "reference": float(ref[index].item()),
                    "prediction": float(pred[index].item()),
                    "error": float(error[index].item()),
                }
                for index in range(min(20, ref.numel()))
            ],
        }
    )
    return result


def last_metric_values(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {"available": False, "path": str(metrics_path), "reason": "missing_metrics_csv"}
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    if not rows:
        return {"available": False, "path": str(metrics_path), "reason": "empty_metrics_csv"}
    wanted = ("epoch", "step", "train_loss_epoch", "val_loss", "val_node_mean", "val_edge_mean")
    values: dict[str, Any] = {"available": True, "path": str(metrics_path)}
    for key in wanted:
        raw = ""
        for row in reversed(rows):
            raw = row.get(key, "")
            if raw != "":
                break
        if raw == "":
            values[key] = None
            continue
        try:
            values[key] = float(raw)
        except ValueError:
            values[key] = raw
    return values


def build_datamodule_kwargs(data: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "out_matrix": data["out_matrix"],
        "basis_files": data["basis_files"],
        "train_runs": data["train_runs"],
        "val_runs": data.get("val_runs", data["train_runs"]),
        "test_runs": data.get("test_runs", data["train_runs"]),
        "symmetric_matrix": bool(data["symmetric_matrix"]),
        "sub_point_matrix": bool(data.get("sub_point_matrix", False)),
        "batch_size": int(data.get("batch_size", 1)),
        "store_in_memory": bool(data.get("store_in_memory", True)),
    }
    signature = inspect.signature(MatrixDataModule).parameters
    if "n_matrix_components" in signature:
        kwargs["n_matrix_components"] = int(data["n_matrix_components"])
    if "matrix_component_policy" in signature:
        kwargs["matrix_component_policy"] = data["matrix_component_policy"]
    if "loader_threads" in signature and data.get("loader_threads") is not None:
        kwargs["loader_threads"] = int(data["loader_threads"])
    return kwargs


def render_first20_table(title: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"### {title}", "", "| index | reference | prediction | error |", "|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            "| {index} | {reference:.10g} | {prediction:.10g} | {error:.10g} |".format(
                **row
            )
        )
    return "\n".join(lines)


def verdict(node: dict[str, Any], edge: dict[str, Any]) -> dict[str, str]:
    if not node["comparable"] or not edge["comparable"]:
        return {
            "status": "INCONCLUSIVE",
            "answer": "Shapes do not match, so direct label fit cannot be evaluated.",
        }
    worst_mae = max(float(node["mae"]), float(edge["mae"]))
    worst_rmse = max(float(node["rmse"]), float(edge["rmse"]))
    if worst_mae <= 1e-3 and worst_rmse <= 2e-3:
        return {
            "status": "B",
            "answer": "The model fit direct training labels to low-meV/sub-meV scale; investigate reconstruction/writer/evaluation.",
        }
    return {
        "status": "A",
        "answer": "The model did not fit the one-sample training labels before MatrixWriter/reconstruction.",
    }


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    node = payload["node_labels"]
    edge = payload["edge_labels"]
    decision = payload["verdict"]
    lines = [
        "# Direct Label Diagnostic",
        "",
        f"Workspace: `{payload['workspace']}`",
        f"Training dir: `{payload['training_dir']}`",
        f"Config: `{payload['config_path']}`",
        f"Checkpoint: `{payload['checkpoint_path']}`",
        "",
        "## Verdict",
        "",
        f"**{decision['status']}**: {decision['answer']}",
        "",
        "## H-only Policy",
        "",
        f"- `out_matrix`: `{payload['data_policy'].get('out_matrix')}`",
        f"- `matrix_component_policy`: `{payload['data_policy'].get('matrix_component_policy')}`",
        f"- `n_matrix_components`: `{payload['data_policy'].get('n_matrix_components')}`",
        f"- `symmetric_matrix`: `{payload['data_policy'].get('symmetric_matrix')}`",
        f"- `sub_point_matrix`: `{payload['data_policy'].get('sub_point_matrix')}`",
        "",
        "## Label Metrics",
        "",
        "| target | reference shape | prediction shape | MAE | RMSE | max abs |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| node | `{node['reference_shape']}` | `{node['prediction_shape']}` | "
            f"{node['mae']} | {node['rmse']} | {node['max_abs']} |"
        ),
        (
            f"| edge | `{edge['reference_shape']}` | `{edge['prediction_shape']}` | "
            f"{edge['mae']} | {edge['rmse']} | {edge['max_abs']} |"
        ),
        "",
        "## Final Logged Metrics",
        "",
        f"Metrics CSV: `{payload['training_metrics'].get('path')}`",
        "",
        "```json",
        json.dumps(payload["training_metrics"], indent=2, sort_keys=True),
        "```",
        "",
        render_first20_table("First 20 Node Label Values", node["first20"]),
        "",
        render_first20_table("First 20 Edge Label Values", edge["first20"]),
        "",
        "## Command",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
        f"Result: `{payload['script_result']}`",
        "",
        "## Notes",
        "",
        "- This diagnostic calls `model(batch)` directly and does not use `MatrixWriter`.",
        "- The comparison is against the first train batch rebuilt from the isolated one-sample config.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare one-sample H-only direct model labels before MatrixWriter."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--training-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    training_dir = (args.training_dir or workspace / "training").resolve()
    config_path = (args.config or training_dir / "config.yaml").resolve()
    checkpoint_path = (args.checkpoint or latest_checkpoint(training_dir)).resolve()
    output_dir = (args.output_dir or workspace / "direct_label_diagnostic").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    datamodule_kwargs = build_datamodule_kwargs(data)

    with working_directory(training_dir):
        model = LitMACEMatrixModel.load_from_checkpoint(str(checkpoint_path))
        model.eval()
        datamodule = MatrixDataModule(**datamodule_kwargs)
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        with torch.no_grad():
            output = model(batch)

    if not isinstance(output, dict):
        raise TypeError(f"Expected model(batch) to return dict, got {type(output).__name__}")
    for key in ("node_labels", "edge_labels"):
        if key not in output:
            raise KeyError(f"model(batch) output is missing {key!r}; keys={sorted(output)}")
    for attr in ("point_labels", "edge_labels"):
        if not hasattr(batch, attr):
            raise AttributeError(f"Batch is missing {attr!r}")

    node = compare_labels(output["node_labels"], batch.point_labels)
    edge = compare_labels(output["edge_labels"], batch.edge_labels)
    metrics_path = training_dir / "logs" / "one_sample_h_only" / "version_0" / "metrics.csv"
    payload = {
        "workspace": str(workspace),
        "training_dir": str(training_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "command": " ".join([sys.executable, *sys.argv]),
        "script_result": "completed_successfully",
        "data_policy": {
            "out_matrix": data.get("out_matrix"),
            "symmetric_matrix": data.get("symmetric_matrix"),
            "sub_point_matrix": data.get("sub_point_matrix"),
            "matrix_component_policy": data.get("matrix_component_policy"),
            "n_matrix_components": data.get("n_matrix_components"),
        },
        "batch_type": type(batch).__name__,
        "model_output_keys": sorted(output),
        "node_labels": node,
        "edge_labels": edge,
        "training_metrics": last_metric_values(metrics_path),
        "verdict": verdict(node, edge),
    }

    json_path = output_dir / "direct_label_diagnostic.json"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_report(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "verdict": payload["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
