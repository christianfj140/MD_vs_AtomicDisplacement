#!/usr/bin/env python3
"""Diagnose why MACE/Graph2Mat readout does not memorize one H2O sample.

This script is intentionally diagnostic-only. It loads the best one-sample
H-only checkpoint from the capacity sweep, inspects parameter counts/readout
grouping/features, and runs two small controlled probes:

1. readout-only training with fixed MACE node features;
2. MACE-bypass training with learnable per-node features feeding the unchanged
   Graph2Mat readout.

It does not modify production configs or training code.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
import torch
import yaml
from mace.modules.utils import get_edge_vectors_and_lengths

REPO_ROOT = Path(__file__).resolve().parents[2]
TORCH_COMPAT_DIR = REPO_ROOT / "scripts" / "torch_serialization_compat"
sys.path.insert(0, str(TORCH_COMPAT_DIR))
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))

from torch_safe_globals import allow_graph2mat_checkpoint_globals  # noqa: E402

allow_graph2mat_checkpoint_globals()

from evaluate_hamiltonian_metrics import read_matrix  # noqa: E402
from graph2mat.core.data.metrics import block_type_mae  # noqa: E402
from graph2mat.tools.lightning import MatrixDataModule  # noqa: E402
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel  # noqa: E402


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
    / "readout_bottleneck_analysis"
)
ATOM_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 14: "Si"}


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
    if not candidates:
        raise FileNotFoundError(f"No best checkpoint found under {training_dir}")
    return candidates[-1]


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


def count_trainable_parameters(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def species_from_type(point_type: int) -> str:
    return ATOM_SYMBOLS.get(int(point_type), str(point_type))


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


def compare_tensors(prediction: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    pred = prediction.detach().cpu().double().reshape(-1)
    ref = reference.detach().cpu().double().reshape(-1)
    result: dict[str, Any] = {
        "prediction_shape": list(prediction.shape),
        "reference_shape": list(reference.shape),
        "shape_match": tuple(prediction.shape) == tuple(reference.shape),
        "n_values": int(ref.numel()),
    }
    if pred.numel() != ref.numel():
        result.update({"mae": None, "rmse": None, "max_abs": None})
        return result
    delta = pred - ref
    abs_delta = delta.abs()
    result.update(
        {
            "mae": float(abs_delta.mean().item()),
            "rmse": float(torch.sqrt(torch.mean(delta * delta)).item()),
            "max_abs": float(abs_delta.max().item()),
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
    return {"support_precision": precision, "support_recall": recall, "support_f1": f1}


def compare_h(reference: np.ndarray, prediction: np.ndarray, threshold: float) -> dict[str, Any]:
    if reference.shape != prediction.shape:
        return {
            "comparable": False,
            "reference_shape": list(reference.shape),
            "prediction_shape": list(prediction.shape),
        }
    delta = prediction - reference
    abs_delta = np.abs(delta)
    ref_norm = float(np.linalg.norm(reference))
    result = {
        "comparable": True,
        "support_threshold": threshold,
        "mae_eV": float(abs_delta.mean()),
        "mae_meV": float(abs_delta.mean() * 1000.0),
        "rmse_eV": float(np.sqrt(np.mean(delta * delta))),
        "rmse_meV": float(np.sqrt(np.mean(delta * delta)) * 1000.0),
        "max_abs_eV": float(abs_delta.max()),
        "relative_frobenius": math.nan if ref_norm == 0 else float(np.linalg.norm(delta) / ref_norm),
        "allclose_1e_5": bool(np.allclose(reference, prediction, atol=1e-5, rtol=1e-8)),
    }
    result.update(support_metrics(reference, prediction, threshold))
    return result


def read_first_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        try:
            return next(csv.DictReader(stream))
        except StopIteration:
            return {}


def read_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return rows if limit is None else rows[:limit]


def make_readout_inputs(model: LitMACEMatrixModel, batch: Any) -> tuple[Any, torch.Tensor, torch.Tensor]:
    matrix_model = model.model
    mace_out = matrix_model.mace(batch, compute_force=False)
    vectors, lengths = get_edge_vectors_and_lengths(
        positions=batch["positions"],
        edge_index=batch["edge_index"],
        shifts=batch["shifts"],
    )
    edge_attrs = matrix_model.mace.spherical_harmonics(vectors)
    edge_feats = matrix_model.mace.radial_embedding(
        lengths, batch["node_attrs"], batch["edge_index"], matrix_model.mace.atomic_numbers
    )
    if isinstance(edge_feats, tuple):
        edge_feats = edge_feats[0]
    data_for_readout = copy.copy(batch)
    data_for_readout["edge_attrs"] = edge_attrs
    data_for_readout["edge_feats"] = edge_feats
    node_feats = mace_out["node_feats"].detach()
    edge_messages = None
    readout = matrix_model.matrix_readouts
    if readout.preprocessing_edges is not None:
        with torch.no_grad():
            _, edge_messages = readout.preprocessing_edges(
                data=data_for_readout,
                node_feats=node_feats,
            )
    return data_for_readout, node_feats, edge_messages.detach() if edge_messages is not None else None


def pairwise_feature_stats(features: torch.Tensor, labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = features.detach().cpu().double()
    for i in range(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            vi = values[i].reshape(-1)
            vj = values[j].reshape(-1)
            denom = float(torch.linalg.norm(vi).item() * torch.linalg.norm(vj).item())
            cosine = math.nan if denom == 0 else float(torch.dot(vi, vj).item() / denom)
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "label_i": labels[i],
                    "label_j": labels[j],
                    "euclidean": float(torch.linalg.norm(vi - vj).item()),
                    "cosine": cosine,
                    "max_abs_delta": float((vi - vj).abs().max().item()),
                    "near_identical": bool(torch.allclose(vi, vj, atol=1e-8, rtol=1e-8)),
                }
            )
    return rows


def operation_input_degeneracy(readout: Any, batch: Any, node_feats: torch.Tensor, edge_messages: torch.Tensor | None) -> dict[str, Any]:
    node_labels = [f"{species_from_type(t)} atom {i}" for i, t in enumerate(batch.point_types.detach().cpu().tolist())]
    node_pairs = pairwise_feature_stats(node_feats, node_labels)
    edge_rows: list[dict[str, Any]] = []
    if edge_messages is None:
        return {"node_pairwise": node_pairs, "edge_operation_pairwise": edge_rows}

    graph2mat_edge_types = readout.edge_types_to_graph2mat[batch.edge_types]
    edge_index = batch.edge_index
    for module_key in readout.interactions.keys():
        point_type, neigh_type, edge_type = map(int, module_key[1:-1].split(","))
        mask = abs(graph2mat_edge_types) == abs(edge_type)
        if not bool(mask.any()):
            continue
        indices = torch.where(mask)[0]
        local_edge_types = graph2mat_edge_types[mask]
        if point_type == neigh_type:
            if readout.symmetric:
                forward_indices = indices[0::2]
                reverse_indices = indices[1::2]
            else:
                forward_indices = indices
                reverse_indices = indices
        else:
            forward_indices = indices[local_edge_types == edge_type]
            reverse_indices = indices[local_edge_types != edge_type]
        samples: list[torch.Tensor] = []
        labels: list[str] = []
        for local_idx, forward_idx in enumerate(forward_indices.tolist()):
            if local_idx >= len(reverse_indices):
                continue
            reverse_idx = int(reverse_indices[local_idx].item())
            sender = int(edge_index[0, forward_idx].item())
            receiver = int(edge_index[1, forward_idx].item())
            vec = torch.cat(
                [
                    node_feats[sender].detach().reshape(-1).cpu(),
                    node_feats[receiver].detach().reshape(-1).cpu(),
                    edge_messages[forward_idx].detach().reshape(-1).cpu(),
                    edge_messages[reverse_idx].detach().reshape(-1).cpu(),
                ]
            )
            samples.append(vec)
            labels.append(f"{species_from_type(batch.point_types[sender])}{sender}->{species_from_type(batch.point_types[receiver])}{receiver}")
        if len(samples) >= 2:
            for row in pairwise_feature_stats(torch.stack(samples), labels):
                row["operation_key"] = module_key
                edge_rows.append(row)
        elif len(samples) == 1:
            edge_rows.append(
                {
                    "operation_key": module_key,
                    "n_samples": 1,
                    "note": "single_sample_no_pairwise_degeneracy_test",
                    "label": labels[0],
                }
            )
    return {"node_pairwise": node_pairs, "edge_operation_pairwise": edge_rows}


def parameter_report(model: LitMACEMatrixModel) -> dict[str, int]:
    readout = model.model.matrix_readouts
    return {
        "total_trainable": count_trainable_parameters(model),
        "mace_trainable": count_trainable_parameters(model.model.mace),
        "graph2mat_readout_trainable": count_trainable_parameters(readout),
        "node_readout_trainable": count_trainable_parameters(readout.self_interactions),
        "edge_preprocessing_trainable": count_trainable_parameters(readout.preprocessing_edges),
        "edge_block_readout_trainable": count_trainable_parameters(readout.interactions),
        "edge_total_readout_trainable": count_trainable_parameters(readout.preprocessing_edges)
        + count_trainable_parameters(readout.interactions),
    }


def readout_grouping_report(readout: Any, batch: Any) -> dict[str, Any]:
    node_types = readout.types_to_graph2mat[batch.point_types]
    edge_types = readout.edge_types_to_graph2mat[batch.edge_types]
    node_rows: list[dict[str, Any]] = []
    for idx, block in enumerate(readout.self_interactions):
        point = readout.graph2mat_table.basis[idx]
        mask = node_types == idx
        node_rows.append(
            {
                "operation_index": idx,
                "point_type": int(point.type),
                "species": species_from_type(int(point.type)),
                "block_shape": list(getattr(block, "block_shape", ())) if block is not None else [],
                "block_size": int(getattr(block, "block_size", 0)) if block is not None else 0,
                "irreps_out": str(getattr(block, "_irreps_out", "")) if block is not None else "",
                "operation_class": type(getattr(block, "operation", None)).__name__ if block is not None else "",
                "n_blocks_using_operation": int(mask.sum().item()),
                "labels_covered": int(mask.sum().item() * getattr(block, "block_size", 0)) if block is not None else 0,
                "trainable_params": count_trainable_parameters(block),
                "shares_multiple_physical_blocks": bool(mask.sum().item() > 1),
            }
        )
    edge_rows: list[dict[str, Any]] = []
    for module_key, block in readout.interactions.items():
        point_type, neigh_type, edge_type = map(int, module_key[1:-1].split(","))
        mask = abs(edge_types) == abs(edge_type)
        if point_type == neigh_type and readout.symmetric:
            n_blocks = int(mask.sum().item() // 2)
        elif point_type == neigh_type:
            n_blocks = int(mask.sum().item())
        else:
            n_blocks = int((edge_types[mask] == edge_type).sum().item()) if bool(mask.any()) else 0
        i_basis = readout.graph2mat_table.basis[point_type]
        j_basis = readout.graph2mat_table.basis[neigh_type]
        edge_rows.append(
            {
                "operation_key": module_key,
                "row_point_type": int(i_basis.type),
                "col_point_type": int(j_basis.type),
                "species_pair": f"{species_from_type(int(i_basis.type))}-{species_from_type(int(j_basis.type))}",
                "block_shape": list(getattr(block, "block_shape", ())) if block is not None else [],
                "block_size": int(getattr(block, "block_size", 0)) if block is not None else 0,
                "irreps_out": str(getattr(block, "_irreps_out", "")) if block is not None else "",
                "operation_class": type(getattr(block, "operation", None)).__name__ if block is not None else "",
                "n_blocks_using_operation": n_blocks,
                "labels_covered": int(n_blocks * getattr(block, "block_size", 0)) if block is not None else 0,
                "trainable_params": count_trainable_parameters(block),
                "shares_multiple_physical_blocks": bool(n_blocks > 1),
            }
        )
    return {
        "basis_grouping": readout.basis_grouping,
        "symmetric": bool(readout.symmetric),
        "node_operations": node_rows,
        "edge_operations": edge_rows,
        "summary": readout.summary,
    }


def reconstruct_h_metrics(processor: Any, batch: Any, output: dict[str, torch.Tensor], reference_h: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = {
        "node_labels": output["node_labels"].detach().cpu(),
        "edge_labels": output["edge_labels"].detach().cpu(),
    }
    matrices = list(processor.yield_from_batch(batch, predictions=predictions, as_matrix=True))
    if len(matrices) != 1:
        return {"status": "error", "reason": f"expected_one_matrix_got_{len(matrices)}"}
    return compare_h(reference_h, dense_h(matrices[0]), threshold)


def train_readout_probe(
    *,
    name: str,
    readout: torch.nn.Module,
    data_for_readout: Any,
    node_feats: torch.Tensor,
    batch: Any,
    processor: Any,
    reference_h: np.ndarray,
    schedule: list[float],
    steps_per_stage: int,
    threshold: float,
    train_node_feats: bool = False,
) -> dict[str, Any]:
    readout.train()
    if train_node_feats:
        node_feats_param = torch.nn.Parameter(node_feats.detach().clone())
        node_feats_param.data += torch.randn_like(node_feats_param) * 1e-3
        params = [node_feats_param, *list(readout.parameters())]
    else:
        node_feats_param = None
        fixed_node_feats = node_feats.detach()
        params = list(readout.parameters())
    loss_fn = block_type_mae()
    best_loss = math.inf
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    final_output: dict[str, torch.Tensor] | None = None
    for stage_index, lr in enumerate(schedule):
        optimizer = torch.optim.Adam(params, lr=lr)
        stage_best = math.inf
        for _ in range(steps_per_stage):
            optimizer.zero_grad()
            current_node_feats = node_feats_param if node_feats_param is not None else fixed_node_feats
            node_labels, edge_labels = readout(data=data_for_readout, node_feats=current_node_feats)
            loss, _ = loss_fn(
                nodes_pred=node_labels,
                nodes_ref=batch["point_labels"],
                edges_pred=edge_labels,
                edges_ref=batch["edge_labels"],
                batch=batch,
                basis_table=processor.basis_table,
            )
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            best_loss = min(best_loss, loss_value)
            stage_best = min(stage_best, loss_value)
        with torch.no_grad():
            current_node_feats = node_feats_param if node_feats_param is not None else fixed_node_feats
            node_labels, edge_labels = readout(data=data_for_readout, node_feats=current_node_feats)
            stage_loss, stage_stats = loss_fn(
                nodes_pred=node_labels,
                nodes_ref=batch["point_labels"],
                edges_pred=edge_labels,
                edges_ref=batch["edge_labels"],
                batch=batch,
                basis_table=processor.basis_table,
                log_verbose=True,
            )
            final_output = {"node_labels": node_labels, "edge_labels": edge_labels}
            history.append(
                {
                    "stage": stage_index,
                    "learning_rate": lr,
                    "steps": steps_per_stage,
                    "stage_best_loss": stage_best,
                    "stage_final_loss": float(stage_loss.detach().cpu().item()),
                    "stage_final_stats": sanitize(stage_stats),
                }
            )
    runtime = time.perf_counter() - start
    assert final_output is not None
    node_metrics = compare_tensors(final_output["node_labels"], batch.point_labels)
    edge_metrics = compare_tensors(final_output["edge_labels"], batch.edge_labels)
    h_metrics = reconstruct_h_metrics(processor, batch, final_output, reference_h, threshold)
    reached = (
        float(node_metrics["mae"]) <= 1e-5
        and float(edge_metrics["mae"]) <= 1e-5
        and h_metrics.get("comparable")
        and float(h_metrics["mae_eV"]) <= 1e-5
    )
    return {
        "name": name,
        "train_node_feats": train_node_feats,
        "schedule": schedule,
        "steps_per_stage": steps_per_stage,
        "total_steps": steps_per_stage * len(schedule),
        "runtime_sec": runtime,
        "best_loss": best_loss,
        "history": history,
        "direct_node": node_metrics,
        "direct_edge": edge_metrics,
        "h_reconstruction": h_metrics,
        "reached_numerical_zero": bool(reached),
    }


def top_residual_tables(workspace: Path) -> dict[str, Any]:
    metrics_dir = workspace / "evaluation" / "metrics"
    block_rows = read_csv_rows(metrics_dir / "block_metrics.csv")
    orbital_rows = read_csv_rows(metrics_dir / "orbital_pair_metrics.csv")
    species_rows = read_csv_rows(metrics_dir / "species_pair_metrics.csv")

    def by_float(rows: list[dict[str, str]], key: str, limit: int) -> list[dict[str, str]]:
        return sorted(rows, key=lambda row: float(row.get(key, "nan")), reverse=True)[:limit]

    return {
        "sparse": read_first_csv_row(metrics_dir / "sparse_metrics.csv"),
        "block_top_by_mae": by_float(block_rows, "mae_union_eV", 12),
        "species_pair_top_by_mae": by_float(species_rows, "mae_union_eV", 12),
        "orbital_pair_top_by_mae_meV": by_float(orbital_rows, "mae_union_meV", 20),
    }


def executive_verdict(payload: dict[str, Any]) -> dict[str, str]:
    readout_only = payload["training_probes"]["readout_only_fixed_mace_features"]
    bypass = payload["training_probes"]["mace_bypass_learnable_node_features"]
    feature_pairs = payload["feature_degeneracy"]["node_pairwise"]
    identical_pairs = [row for row in feature_pairs if row.get("near_identical")]
    if bypass.get("reached_numerical_zero") and not readout_only.get("reached_numerical_zero"):
        return {
            "status": "A_FEATURE_OR_OPTIMIZATION_BOTTLENECK",
            "summary": (
                "Graph2Mat readout can memorize when fed learnable per-node features, "
                "but not when constrained to the frozen MACE features from the failed checkpoint. "
                "The evidence points to MACE feature generation and/or optimization into the readout, "
                "not a fundamental label/reconstruction problem."
            ),
        }
    if readout_only.get("reached_numerical_zero"):
        return {
            "status": "C_OPTIMIZATION_BOTTLENECK",
            "summary": (
                "The existing readout can memorize with frozen MACE features when optimized directly. "
                "The full MACE run likely failed due to joint optimization dynamics rather than readout capacity."
            ),
        }
    if not bypass.get("reached_numerical_zero"):
        return {
            "status": "B_READOUT_CONSTRAINT_OR_PROBE_LIMIT",
            "summary": (
                "Neither fixed-feature readout-only nor learnable-node-feature bypass reached numerical zero "
                "within this diagnostic budget. This points to readout/equivariance constraints or insufficient "
                "probe optimization; inspect the residual concentration and grouping."
            ),
        }
    if identical_pairs:
        return {
            "status": "A_FEATURE_DEGENERACY",
            "summary": "Some readout inputs are identical for different physical blocks; MACE feature degeneracy is a bottleneck candidate.",
        }
    return {
        "status": "D_INCONCLUSIVE",
        "summary": "The diagnostic evidence is mixed; see probe metrics and feature-distance tables.",
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    verdict = payload["executive_verdict"]
    counts = payload["parameter_counts"]
    baseline = payload["baseline_model_metrics"]
    ro = payload["training_probes"]["readout_only_fixed_mace_features"]
    bp = payload["training_probes"]["mace_bypass_learnable_node_features"]
    lines = [
        "# H2O Readout Bottleneck Analysis",
        "",
        f"Source workspace: `{payload['source_workspace']}`",
        f"Checkpoint: `{payload['checkpoint_path']}`",
        "",
        "## Executive Verdict",
        "",
        f"**{verdict['status']}**: {verdict['summary']}",
        "",
        "## Parameter Counts",
        "",
        "| group | trainable params |",
        "|---|---:|",
    ]
    for key, value in counts.items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            f"Label scalars: node `{payload['label_counts']['node_labels']}`, edge `{payload['label_counts']['edge_labels']}`, total `{payload['label_counts']['total_labels']}`.",
            "",
            "## Readout Grouping",
            "",
            f"- `basis_grouping`: `{payload['readout_grouping']['basis_grouping']}`",
            f"- `symmetric`: `{payload['readout_grouping']['symmetric']}`",
            "",
            "### Node Operations",
            "",
            "| op | species | block shape | irreps_out | blocks using op | labels covered | params | shared? |",
            "|---:|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["readout_grouping"]["node_operations"]:
        lines.append(
            f"| {row['operation_index']} | {row['species']} | `{row['block_shape']}` | `{row['irreps_out']}` | "
            f"{row['n_blocks_using_operation']} | {row['labels_covered']} | {row['trainable_params']} | {row['shares_multiple_physical_blocks']} |"
        )
    lines.extend(
        [
            "",
            "### Edge Operations",
            "",
            "| op | species pair | block shape | irreps_out | blocks using op | labels covered | params | shared? |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["readout_grouping"]["edge_operations"]:
        lines.append(
            f"| `{row['operation_key']}` | {row['species_pair']} | `{row['block_shape']}` | `{row['irreps_out']}` | "
            f"{row['n_blocks_using_operation']} | {row['labels_covered']} | {row['trainable_params']} | {row['shares_multiple_physical_blocks']} |"
        )
    lines.extend(
        [
            "",
            "## Feature Degeneracy",
            "",
            "### Node Feature Pairwise Distances",
            "",
            "| pair | euclidean | cosine | max abs delta | near identical |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["feature_degeneracy"]["node_pairwise"]:
        lines.append(
            f"| {row['label_i']} vs {row['label_j']} | {row['euclidean']} | {row['cosine']} | {row['max_abs_delta']} | {row['near_identical']} |"
        )
    lines.extend(
        [
            "",
            "### Edge Readout Input Pairwise Distances",
            "",
            "| operation | pair | euclidean | cosine | max abs delta | near identical |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["feature_degeneracy"]["edge_operation_pairwise"]:
        pair = row.get("label", f"{row.get('label_i', '')} vs {row.get('label_j', '')}")
        lines.append(
            f"| `{row.get('operation_key')}` | {pair} | {row.get('euclidean', '')} | {row.get('cosine', '')} | {row.get('max_abs_delta', '')} | {row.get('near_identical', '')} |"
        )
    lines.extend(
        [
            "",
            "## Block-Level Residuals From Best MACE Checkpoint",
            "",
            f"- Direct node MAE: `{baseline['direct_node']['mae']}` eV",
            f"- Direct edge MAE: `{baseline['direct_edge']['mae']}` eV",
            f"- H MAE: `{baseline['h_reconstruction']['mae_meV']}` meV",
            "",
            "Top atom-pair blocks by MAE:",
            "",
            "| row atom | col atom | species | n | MAE eV | RMSE eV | max eV |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["block_level_residuals"]["block_top_by_mae"][:8]:
        lines.append(
            f"| {row.get('row_atom')} | {row.get('col_atom')} | {row.get('row_species')}-{row.get('col_species')} | "
            f"{row.get('n_entries')} | {row.get('mae_union_eV')} | {row.get('rmse_union_eV')} | {row.get('max_abs_error_union_eV')} |"
        )
    lines.extend(
        [
            "",
            "Top orbital-pair groups by MAE:",
            "",
            "| species pair | row orb | col orb | n | MAE meV | RMSE eV | max eV |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["block_level_residuals"]["orbital_pair_top_by_mae_meV"][:12]:
        lines.append(
            f"| {row.get('species_pair')} | {row.get('row_orbital_index')} | {row.get('col_orbital_index')} | "
            f"{row.get('n_entries')} | {row.get('mae_union_meV')} | {row.get('rmse_union_eV')} | {row.get('max_abs_error_union_eV')} |"
        )
    lines.extend(
        [
            "",
            "## Training Probes",
            "",
            "| probe | steps | node MAE eV | edge MAE eV | H MAE meV | H RMSE meV | reached near zero | runtime s |",
            "|---|---:|---:|---:|---:|---:|---|---:|",
            (
                f"| readout-only fixed MACE features | {ro['total_steps']} | {ro['direct_node']['mae']} | "
                f"{ro['direct_edge']['mae']} | {ro['h_reconstruction'].get('mae_meV')} | "
                f"{ro['h_reconstruction'].get('rmse_meV')} | {ro['reached_numerical_zero']} | {ro['runtime_sec']} |"
            ),
            (
                f"| MACE-bypass learnable node features | {bp['total_steps']} | {bp['direct_node']['mae']} | "
                f"{bp['direct_edge']['mae']} | {bp['h_reconstruction'].get('mae_meV')} | "
                f"{bp['h_reconstruction'].get('rmse_meV')} | {bp['reached_numerical_zero']} | {bp['runtime_sec']} |"
            ),
            "",
            "## Fully Unconstrained Baseline",
            "",
            f"- Status: `{payload['unconstrained_label_memorizer'].get('verdict', {}).get('status')}`",
            f"- H MAE: `{payload['unconstrained_label_memorizer'].get('h_reconstruction_metrics', {}).get('mae_meV')}` meV",
            "",
            "## Caveats",
            "",
            "- The MACE-bypass probe uses learnable per-node features because the current `E3nnGraph2Mat` API derives edge messages through `preprocessing_edges`; it does not expose independent learnable edge feature injection without changing the readout API.",
            "- These probes are diagnostic, not production hyperparameter recommendations.",
            "",
            "## Commands",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose H2O Graph2Mat/MACE readout bottleneck.")
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--readout-steps-per-stage", type=int, default=600)
    parser.add_argument("--bypass-steps-per-stage", type=int, default=600)
    parser.add_argument("--learning-rate-schedule", default="0.003,0.0006,0.00012")
    parser.add_argument("--threshold", type=float, default=1e-5)
    args = parser.parse_args()

    source_workspace = args.source_workspace.resolve()
    training_dir = source_workspace / "training"
    config_path = training_dir / "config.yaml"
    checkpoint_path = latest_checkpoint(training_dir)
    reference_path = source_workspace / "dataset" / "samples" / "md_94" / "siesta.TSHS"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    schedule = [float(item.strip()) for item in args.learning_rate_schedule.split(",") if item.strip()]
    reference_h = np.asarray(read_matrix(reference_path).hamiltonian.toarray(), dtype=float)

    with working_directory(training_dir):
        datamodule = MatrixDataModule(**build_datamodule_kwargs(data))
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        processor = datamodule.data_processor
        model = LitMACEMatrixModel.load_from_checkpoint(str(checkpoint_path))
        model.eval()
        with torch.no_grad():
            baseline_output = model(batch)
            data_for_readout, node_feats, edge_messages = make_readout_inputs(model, batch)

        baseline = {
            "direct_node": compare_tensors(baseline_output["node_labels"], batch.point_labels),
            "direct_edge": compare_tensors(baseline_output["edge_labels"], batch.edge_labels),
            "h_reconstruction": reconstruct_h_metrics(processor, batch, baseline_output, reference_h, args.threshold),
        }
        readout = model.model.matrix_readouts
        params = parameter_report(model)
        grouping = readout_grouping_report(readout, batch)
        features = operation_input_degeneracy(readout, batch, node_feats, edge_messages)

        readout_only_model = LitMACEMatrixModel.load_from_checkpoint(str(checkpoint_path))
        bypass_model = LitMACEMatrixModel.load_from_checkpoint(str(checkpoint_path))

        readout_only = train_readout_probe(
            name="readout_only_fixed_mace_features",
            readout=readout_only_model.model.matrix_readouts,
            data_for_readout=data_for_readout,
            node_feats=node_feats,
            batch=batch,
            processor=processor,
            reference_h=reference_h,
            schedule=schedule,
            steps_per_stage=args.readout_steps_per_stage,
            threshold=args.threshold,
            train_node_feats=False,
        )
        bypass = train_readout_probe(
            name="mace_bypass_learnable_node_features",
            readout=bypass_model.model.matrix_readouts,
            data_for_readout=data_for_readout,
            node_feats=node_feats,
            batch=batch,
            processor=processor,
            reference_h=reference_h,
            schedule=schedule,
            steps_per_stage=args.bypass_steps_per_stage,
            threshold=args.threshold,
            train_node_feats=True,
        )

    label_memorizer_path = (
        REPO_ROOT
        / "Comparison"
        / "results"
        / "diagnostics"
        / "h2o_hamiltonian"
        / "label_memorizer_baseline"
        / "label_memorizer_baseline.json"
    )
    label_memorizer = json.loads(label_memorizer_path.read_text(encoding="utf-8")) if label_memorizer_path.exists() else {}

    payload = {
        "source_workspace": str(source_workspace),
        "training_dir": str(training_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "reference_path": str(reference_path),
        "output_dir": str(output_dir),
        "command": " ".join([sys.executable, *sys.argv]),
        "data_policy": {
            "out_matrix": data.get("out_matrix"),
            "symmetric_matrix": data.get("symmetric_matrix"),
            "sub_point_matrix": data.get("sub_point_matrix"),
            "matrix_component_policy": data.get("matrix_component_policy"),
            "n_matrix_components": data.get("n_matrix_components"),
        },
        "label_counts": {
            "node_labels": int(batch.point_labels.numel()),
            "edge_labels": int(batch.edge_labels.numel()),
            "total_labels": int(batch.point_labels.numel() + batch.edge_labels.numel()),
        },
        "parameter_counts": params,
        "readout_grouping": grouping,
        "feature_degeneracy": features,
        "baseline_model_metrics": baseline,
        "block_level_residuals": top_residual_tables(source_workspace),
        "training_probes": {
            "readout_only_fixed_mace_features": readout_only,
            "mace_bypass_learnable_node_features": bypass,
        },
        "unconstrained_label_memorizer": label_memorizer,
    }
    payload["executive_verdict"] = executive_verdict(payload)

    json_path = output_dir / "readout_bottleneck_analysis.json"
    report_path = output_dir / "report.md"
    json_path.write_text(json.dumps(sanitize(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report_path, sanitize(payload))
    print(
        json.dumps(
            {
                "json": str(json_path),
                "report": str(report_path),
                "verdict": payload["executive_verdict"],
                "readout_only_h_mae_meV": readout_only["h_reconstruction"].get("mae_meV"),
                "bypass_h_mae_meV": bypass["h_reconstruction"].get("mae_meV"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
