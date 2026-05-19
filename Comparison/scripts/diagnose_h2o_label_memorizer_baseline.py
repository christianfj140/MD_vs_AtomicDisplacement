#!/usr/bin/env python3
"""Train a trivial one-sample label memorizer for the H2O H-only diagnostic.

The model intentionally has no graph readout. Its only learnable parameters are
one vector matching ``batch.point_labels`` and one vector matching
``batch.edge_labels``. If this baseline reaches near-zero direct-label and
Hamiltonian reconstruction error, the loss/data/reconstruction path can fit the
one-sample target and the remaining bottleneck is upstream in the MACE/Graph2Mat
readout.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))

from evaluate_hamiltonian_metrics import read_matrix  # noqa: E402
from graph2mat.core.data.metrics import OrbitalMatrixMetric  # noqa: E402
from graph2mat.tools.lightning import MatrixDataModule  # noqa: E402


DEFAULT_SOURCE_WORKSPACE = (
    REPO_ROOT
    / "Comparison"
    / "results"
    / "diagnostics"
    / "h2o_hamiltonian"
    / "one_sample_capacity_sweep"
    / "i3_h64_lr001_e600"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Comparison"
    / "results"
    / "diagnostics"
    / "h2o_hamiltonian"
    / "label_memorizer_baseline"
)


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


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


def resolve_loss(loss_path: str) -> type[OrbitalMatrixMetric]:
    """Resolve the same Graph2Mat metric path used in YAML configs."""
    aliases = {
        "graph2mat.metrics": "graph2mat.core.data.metrics",
    }
    module_name, _, attr = loss_path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid loss path: {loss_path!r}")
    module_name = aliases.get(module_name, module_name)
    module = importlib.import_module(module_name)
    loss_cls = getattr(module, attr)
    if not isinstance(loss_cls, type) or not issubclass(loss_cls, OrbitalMatrixMetric):
        raise TypeError(f"{loss_path!r} did not resolve to an OrbitalMatrixMetric class")
    return loss_cls


class LabelMemorizer(pl.LightningModule):
    """Lightning module with exactly two learnable label vectors."""

    def __init__(
        self,
        node_reference: torch.Tensor,
        edge_reference: torch.Tensor,
        *,
        loss_cls: type[OrbitalMatrixMetric],
        basis_table: Any,
        init: str,
    ) -> None:
        super().__init__()
        if init == "zeros":
            node_initial = torch.zeros_like(node_reference.detach())
            edge_initial = torch.zeros_like(edge_reference.detach())
        elif init == "mean":
            node_initial = torch.full_like(
                node_reference.detach(), float(node_reference.detach().mean().item())
            )
            edge_initial = torch.full_like(
                edge_reference.detach(), float(edge_reference.detach().mean().item())
            )
        else:
            raise ValueError(f"Unsupported init policy: {init}")

        self.node_labels_param = torch.nn.Parameter(node_initial)
        self.edge_labels_param = torch.nn.Parameter(edge_initial)
        self.loss_fn = loss_cls()
        self.basis_table = basis_table

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        return {
            "node_labels": self.node_labels_param,
            "edge_labels": self.edge_labels_param,
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out = self(batch)
        loss, stats = self.loss_fn(
            nodes_pred=out["node_labels"],
            nodes_ref=batch["point_labels"],
            edges_pred=out["edge_labels"],
            edges_ref=batch["edge_labels"],
            batch=batch,
            basis_table=self.basis_table,
            log_verbose=True,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=False)
        for key, value in stats.items():
            self.log(f"train_{key}", value, on_step=False, on_epoch=True, logger=False)
        return loss

    def configure_optimizers(self):
        raise RuntimeError(
            "This diagnostic trains the LightningModule parameters with an explicit "
            "local optimizer schedule instead of Trainer.configure_optimizers()."
        )


def compare_tensors(prediction: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    pred = prediction.detach().cpu().double().reshape(-1)
    ref = reference.detach().cpu().double().reshape(-1)
    shape_match = tuple(prediction.shape) == tuple(reference.shape)
    result: dict[str, Any] = {
        "prediction_shape": list(prediction.shape),
        "reference_shape": list(reference.shape),
        "shape_match": shape_match,
        "n_values": int(ref.numel()),
    }
    if not shape_match or pred.numel() != ref.numel():
        result.update({"mae": None, "rmse": None, "max_abs": None})
        return result
    delta = pred - ref
    abs_delta = delta.abs()
    result.update(
        {
            "mae": float(abs_delta.mean().item()),
            "rmse": float(torch.sqrt(torch.mean(delta * delta)).item()),
            "max_abs": float(abs_delta.max().item()),
            "first20": [
                {
                    "index": index,
                    "reference": float(ref[index].item()),
                    "prediction": float(pred[index].item()),
                    "error": float(delta[index].item()),
                }
                for index in range(min(20, ref.numel()))
            ],
        }
    )
    return result


def dense_h(matrix: Any) -> np.ndarray:
    if not hasattr(matrix, "tocsr"):
        raise TypeError(f"Object {type(matrix).__name__} does not expose tocsr()")
    return np.asarray(matrix.tocsr(0).toarray(), dtype=float)


def support_metrics(reference: np.ndarray, prediction: np.ndarray, threshold: float) -> dict[str, Any]:
    ref_support = np.abs(reference) > threshold
    pred_support = np.abs(prediction) > threshold
    tp = int(np.logical_and(ref_support, pred_support).sum())
    fp = int(np.logical_and(~ref_support, pred_support).sum())
    fn = int(np.logical_and(ref_support, ~pred_support).sum())
    precision = math.nan if tp + fp == 0 else tp / (tp + fp)
    recall = math.nan if tp + fn == 0 else tp / (tp + fn)
    f1 = math.nan if not math.isfinite(precision + recall) or precision + recall == 0 else (
        2 * precision * recall / (precision + recall)
    )
    return {
        "reference_nnz": int(ref_support.sum()),
        "prediction_nnz": int(pred_support.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
    }


def compare_h(reference: np.ndarray, prediction: np.ndarray, threshold: float) -> dict[str, Any]:
    if reference.shape != prediction.shape:
        return {
            "comparable": False,
            "reason": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "prediction_shape": list(prediction.shape),
        }
    delta = prediction - reference
    abs_delta = np.abs(delta)
    ref_norm = float(np.linalg.norm(reference))
    rel_fro = math.nan if ref_norm == 0 else float(np.linalg.norm(delta) / ref_norm)
    result = {
        "comparable": True,
        "reference_shape": list(reference.shape),
        "prediction_shape": list(prediction.shape),
        "support_threshold": threshold,
        "mae_eV": float(abs_delta.mean()),
        "mae_meV": float(abs_delta.mean() * 1000.0),
        "rmse_eV": float(np.sqrt(np.mean(delta * delta))),
        "rmse_meV": float(np.sqrt(np.mean(delta * delta)) * 1000.0),
        "max_abs_eV": float(abs_delta.max()),
        "relative_frobenius": rel_fro,
        "allclose_1e_8": bool(np.allclose(reference, prediction, atol=1e-8, rtol=1e-8)),
        "allclose_1e_5": bool(np.allclose(reference, prediction, atol=1e-5, rtol=1e-8)),
    }
    result.update(support_metrics(reference, prediction, threshold))
    return result


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return sanitize(value.detach().cpu().numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_learning_rate_schedule(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one learning rate is required")
    if any(value <= 0 for value in values):
        raise ValueError(f"Learning rates must be positive: {values}")
    return values


def verdict(node: dict[str, Any], edge: dict[str, Any], h_metrics: dict[str, Any]) -> dict[str, str]:
    direct_ok = (
        node.get("mae") is not None
        and edge.get("mae") is not None
        and float(node["mae"]) <= 1e-5
        and float(edge["mae"]) <= 1e-5
    )
    h_ok = (
        h_metrics.get("comparable")
        and float(h_metrics.get("mae_eV", math.inf)) <= 1e-5
        and float(h_metrics.get("max_abs_eV", math.inf)) <= 1e-4
    )
    if direct_ok and h_ok:
        return {
            "status": "PASS",
            "summary": (
                "The trivial label memorizer reached numerical-zero direct-label "
                "and H reconstruction error. The loss/data/reconstruction path can "
                "memorize the one-sample target; the MACE/Graph2Mat readout remains "
                "the likely bottleneck."
            ),
        }
    if direct_ok and not h_ok:
        return {
            "status": "FAIL_RECONSTRUCTION",
            "summary": (
                "The trivial memorizer fit direct labels, but reconstructed H is not "
                "near-zero. Investigate reconstruction/MatrixWriter/evaluation."
            ),
        }
    return {
        "status": "FAIL_TRAINING_OR_LOSS",
        "summary": (
            "The trivial memorizer did not fit direct labels to near-zero. "
            "Investigate loss wiring, optimizer behavior, or target tensors."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    node = payload["direct_label_metrics"]["node"]
    edge = payload["direct_label_metrics"]["edge"]
    h_metrics = payload["h_reconstruction_metrics"]
    lines = [
        "# H2O Label Memorizer Baseline",
        "",
        f"Source workspace: `{payload['source_workspace']}`",
        f"Training dir: `{payload['training_dir']}`",
        f"Config: `{payload['config_path']}`",
        f"Reference: `{payload['reference_path']}`",
        "",
        "## Verdict",
        "",
        f"**{payload['verdict']['status']}**: {payload['verdict']['summary']}",
        "",
        "## Policy",
        "",
        f"- `out_matrix`: `{payload['data_policy'].get('out_matrix')}`",
        f"- `matrix_component_policy`: `{payload['data_policy'].get('matrix_component_policy')}`",
        f"- `n_matrix_components`: `{payload['data_policy'].get('n_matrix_components')}`",
        f"- `symmetric_matrix`: `{payload['data_policy'].get('symmetric_matrix')}`",
        f"- loss: `{payload['loss_path']}`",
        "",
        "## Direct Label Metrics",
        "",
        "| target | shape | MAE eV | RMSE eV | max abs eV |",
        "|---|---:|---:|---:|---:|",
        f"| node | `{node['reference_shape']}` | {node['mae']} | {node['rmse']} | {node['max_abs']} |",
        f"| edge | `{edge['reference_shape']}` | {edge['mae']} | {edge['rmse']} | {edge['max_abs']} |",
        "",
        "## H Reconstruction Metrics",
        "",
        "| MAE meV | RMSE meV | max abs eV | rel Frobenius | support F1 | allclose 1e-8 |",
        "|---:|---:|---:|---:|---:|---|",
        (
            f"| {h_metrics.get('mae_meV')} | {h_metrics.get('rmse_meV')} | "
            f"{h_metrics.get('max_abs_eV')} | {h_metrics.get('relative_frobenius')} | "
            f"{h_metrics.get('support_f1')} | {h_metrics.get('allclose_1e_8')} |"
        ),
        "",
        "## Training",
        "",
        f"- epochs requested: `{payload['training']['epochs']}`",
        f"- epochs run: `{payload['training']['epochs_run']}`",
        f"- learning-rate schedule: `{payload['training']['learning_rate_schedule']}`",
        f"- init: `{payload['training']['init']}`",
        f"- runtime seconds: `{payload['training']['runtime_sec']}`",
        f"- final loss: `{payload['training']['final_loss']}`",
        f"- best loss: `{payload['training']['best_loss']}`",
        "",
        "## Command",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
        "## First 20 Values",
        "",
        "See `label_memorizer_baseline.json` for first-20 node/edge reference/prediction/error values.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a trivial H2O one-sample direct label memorizer."
    )
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--training-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--epochs-per-stage", type=int, default=2000)
    parser.add_argument(
        "--learning-rate-schedule",
        default="0.05,0.01,0.002,0.0004,0.00008,0.000016",
        help="Comma-separated Adam learning-rate stages for the L1 Graph2Mat loss.",
    )
    parser.add_argument("--init", choices=("zeros", "mean"), default="zeros")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-5,
        help="Support threshold for the near-identity reconstructed H comparison.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    source_workspace = args.source_workspace.resolve()
    training_dir = (args.training_dir or source_workspace / "training").resolve()
    config_path = (args.config or training_dir / "config.yaml").resolve()
    reference_path = (
        args.reference or source_workspace / "dataset" / "samples" / "md_94" / "siesta.TSHS"
    ).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    loss_path = config.get("model", {}).get("loss", "graph2mat.metrics.block_type_mae")
    loss_cls = resolve_loss(loss_path)
    datamodule_kwargs = build_datamodule_kwargs(data)
    lr_schedule = parse_learning_rate_schedule(args.learning_rate_schedule)
    epochs_per_stage = args.epochs_per_stage
    if args.epochs is not None:
        if len(lr_schedule) != 1:
            raise ValueError("--epochs is only supported with a single learning rate")
        epochs_per_stage = args.epochs

    pl.seed_everything(args.seed, workers=True)

    with working_directory(training_dir):
        datamodule = MatrixDataModule(**datamodule_kwargs)
        datamodule.setup("fit")
        processor = datamodule.data_processor
        batch = next(iter(datamodule.train_dataloader()))
        model = LabelMemorizer(
            batch.point_labels.detach().clone(),
            batch.edge_labels.detach().clone(),
            loss_cls=loss_cls,
            basis_table=processor.basis_table,
            init=args.init,
        )
        start = time.perf_counter()
        best_loss = math.inf
        epochs_run = 0
        stage_summaries: list[dict[str, Any]] = []
        for stage_index, learning_rate in enumerate(lr_schedule):
            optimizer = torch.optim.Adam(
                [model.node_labels_param, model.edge_labels_param],
                lr=learning_rate,
            )
            stage_best = math.inf
            for _ in range(epochs_per_stage):
                optimizer.zero_grad()
                out = model(batch)
                loss, _ = model.loss_fn(
                    nodes_pred=out["node_labels"],
                    nodes_ref=batch["point_labels"],
                    edges_pred=out["edge_labels"],
                    edges_ref=batch["edge_labels"],
                    batch=batch,
                    basis_table=processor.basis_table,
                )
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().cpu().item())
                best_loss = min(best_loss, loss_value)
                stage_best = min(stage_best, loss_value)
                epochs_run += 1
            with torch.no_grad():
                stage_out = model(batch)
                stage_loss, stage_stats = model.loss_fn(
                    nodes_pred=stage_out["node_labels"],
                    nodes_ref=batch["point_labels"],
                    edges_pred=stage_out["edge_labels"],
                    edges_ref=batch["edge_labels"],
                    batch=batch,
                    basis_table=processor.basis_table,
                    log_verbose=True,
                )
            stage_summaries.append(
                {
                    "stage": stage_index,
                    "learning_rate": learning_rate,
                    "epochs": epochs_per_stage,
                    "best_loss": stage_best,
                    "final_loss": float(stage_loss.detach().cpu().item()),
                    "final_stats": sanitize(stage_stats),
                }
            )
        runtime_sec = time.perf_counter() - start

        with torch.no_grad():
            output = model(batch)
            final_loss, final_stats = model.loss_fn(
                nodes_pred=output["node_labels"],
                nodes_ref=batch["point_labels"],
                edges_pred=output["edge_labels"],
                edges_ref=batch["edge_labels"],
                batch=batch,
                basis_table=processor.basis_table,
                log_verbose=True,
            )
            predictions = {
                "node_labels": output["node_labels"].detach().cpu(),
                "edge_labels": output["edge_labels"].detach().cpu(),
            }
            reconstructed = list(
                processor.yield_from_batch(batch, predictions=predictions, as_matrix=True)
            )

    if len(reconstructed) != 1:
        raise RuntimeError(f"Expected one reconstructed matrix, got {len(reconstructed)}")

    reference_h = np.asarray(read_matrix(reference_path).hamiltonian.toarray(), dtype=float)
    prediction_h = dense_h(reconstructed[0])
    node_metrics = compare_tensors(predictions["node_labels"], batch.point_labels)
    edge_metrics = compare_tensors(predictions["edge_labels"], batch.edge_labels)
    h_metrics = compare_h(reference_h, prediction_h, args.threshold)

    payload = {
        "source_workspace": str(source_workspace),
        "training_dir": str(training_dir),
        "config_path": str(config_path),
        "reference_path": str(reference_path),
        "output_dir": str(output_dir),
        "command": " ".join([sys.executable, *sys.argv]),
        "loss_path": loss_path,
        "loss_class": f"{loss_cls.__module__}.{loss_cls.__name__}",
        "data_policy": {
            "out_matrix": data.get("out_matrix"),
            "symmetric_matrix": data.get("symmetric_matrix"),
            "sub_point_matrix": data.get("sub_point_matrix"),
            "matrix_component_policy": data.get("matrix_component_policy"),
            "n_matrix_components": data.get("n_matrix_components"),
            "batch_size": data.get("batch_size"),
        },
        "batch": {
            "type": type(batch).__name__,
            "point_labels_shape": list(batch.point_labels.shape),
            "edge_labels_shape": list(batch.edge_labels.shape),
        },
        "training": {
            "epochs": epochs_per_stage * len(lr_schedule),
            "epochs_per_stage": epochs_per_stage,
            "epochs_run": epochs_run,
            "learning_rate_schedule": lr_schedule,
            "init": args.init,
            "runtime_sec": runtime_sec,
            "final_loss": float(final_loss.detach().cpu().item()),
            "best_loss": best_loss,
            "final_stats": sanitize(final_stats),
            "stages": stage_summaries,
        },
        "direct_label_metrics": {
            "node": node_metrics,
            "edge": edge_metrics,
        },
        "h_reconstruction_metrics": h_metrics,
    }
    payload["verdict"] = verdict(node_metrics, edge_metrics, h_metrics)

    json_path = output_dir / "label_memorizer_baseline.json"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(sanitize(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, sanitize(payload))
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "verdict": payload["verdict"],
                "node_mae": node_metrics["mae"],
                "edge_mae": edge_metrics["mae"],
                "h_mae_meV": h_metrics["mae_meV"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
