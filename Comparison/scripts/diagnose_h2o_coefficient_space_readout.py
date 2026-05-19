#!/usr/bin/env python3
"""Diagnose Graph2Mat readout failures in irreducible-coefficient space.

This script is diagnostic-only. It works below the matrix-block reconstruction
layer: target Hamiltonian blocks are projected to the same irreducible
coefficients used by ``E3nnIrrepsMatrixBlock`` before ``change_of_basis`` is
applied. It then compares and trains the readout in that coefficient space.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TORCH_COMPAT_DIR = REPO_ROOT / "scripts" / "torch_serialization_compat"
sys.path.insert(0, str(TORCH_COMPAT_DIR))
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))

from torch_safe_globals import allow_graph2mat_checkpoint_globals  # noqa: E402

allow_graph2mat_checkpoint_globals()

from diagnose_h2o_readout_bottleneck import (  # noqa: E402
    DEFAULT_SOURCE_WORKSPACE,
    build_datamodule_kwargs,
    compare_h,
    compare_tensors,
    dense_h,
    latest_checkpoint,
    make_readout_inputs,
    reconstruct_h_metrics,
    sanitize,
)
from evaluate_hamiltonian_metrics import read_matrix  # noqa: E402
from graph2mat.core.data.metrics import block_type_mae  # noqa: E402
from graph2mat.tools.lightning import MatrixDataModule  # noqa: E402
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Comparison"
    / "results"
    / "diagnostics"
    / "h2o_hamiltonian"
    / "coefficient_space_readout_analysis"
)


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True).strip()
    except Exception:
        return None


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def project_block_to_coeffs_torch(block: torch.Tensor, change_of_basis: torch.Tensor) -> torch.Tensor:
    if block.ndim == 2:
        block = block.unsqueeze(0)
    return torch.einsum("zij,nij->nz", change_of_basis.to(block), block)


def coeffs_to_block_torch(coeffs: torch.Tensor, change_of_basis: torch.Tensor) -> torch.Tensor:
    if coeffs.ndim == 1:
        coeffs = coeffs.unsqueeze(0)
    return torch.einsum("nz,zij->nij", coeffs, change_of_basis.to(coeffs))


def coeff_metrics(prediction: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    pred = prediction.detach().cpu().double().reshape(-1)
    ref = reference.detach().cpu().double().reshape(-1)
    delta = pred - ref
    abs_delta = delta.abs()
    return {
        "n_coefficients": int(ref.numel()),
        "mae": float(abs_delta.mean().item()) if ref.numel() else math.nan,
        "rmse": float(torch.sqrt(torch.mean(delta * delta)).item()) if ref.numel() else math.nan,
        "max_abs": float(abs_delta.max().item()) if ref.numel() else math.nan,
    }


def concatenate_coeffs(coeffs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for _, value in sorted(coeffs.items())])


def target_coefficients_and_reconstructed_labels(readout: Any, processor: Any, batch: Any) -> dict[str, Any]:
    table = processor.basis_table
    point_types = batch.point_types.detach().cpu().numpy().astype(int)
    point_labels = batch.point_labels.detach().cpu().reshape(-1)
    point_pointers = table.point_block_pointer(point_types)
    graph_node_types = readout.types_to_graph2mat[batch.point_types].detach().cpu().numpy().astype(int)

    node_coeffs: dict[str, list[torch.Tensor]] = {}
    node_reconstructed: list[torch.Tensor] = []
    node_block_rows: list[dict[str, Any]] = []

    for atom_index, point_type in enumerate(point_types):
        op_key = f"node:{graph_node_types[atom_index]}"
        operation = readout.self_interactions[int(graph_node_types[atom_index])]
        shape = tuple(int(x) for x in table.point_block_shape[:, point_type])
        block = point_labels[point_pointers[atom_index] : point_pointers[atom_index + 1]].reshape(shape)
        coeff = project_block_to_coeffs_torch(block, operation.change_of_basis).squeeze(0)
        reconstructed = coeffs_to_block_torch(coeff, operation.change_of_basis).squeeze(0)
        node_coeffs.setdefault(op_key, []).append(coeff)
        node_reconstructed.append(reconstructed.reshape(-1))
        node_block_rows.append(
            {
                "op_key": op_key,
                "atom_index": int(atom_index),
                "point_type": int(point_type),
                "block_shape": list(shape),
                "coefficients": int(coeff.numel()),
                "reconstruction_max_abs_eV": float((reconstructed - block).abs().max().item()),
            }
        )

    edge_types_full = batch.edge_types.detach().cpu().numpy().astype(int)
    edge_types = edge_types_full[::2]
    edge_labels = batch.edge_labels.detach().cpu().reshape(-1)
    edge_pointers = table.edge_block_pointer(edge_types)
    edge_coeffs: dict[str, list[torch.Tensor]] = {}
    edge_reconstructed: list[torch.Tensor] = []
    edge_block_rows: list[dict[str, Any]] = []

    for unique_edge_index, edge_type in enumerate(edge_types):
        module_key, operation = edge_module_for_unique_type(readout, int(edge_type))
        op_key = f"edge:{module_key}"
        shape = tuple(int(x) for x in table.edge_block_shape[:, abs(edge_type)])
        block = edge_labels[edge_pointers[unique_edge_index] : edge_pointers[unique_edge_index + 1]].reshape(shape)
        coeff = project_block_to_coeffs_torch(block, operation.change_of_basis).squeeze(0)
        reconstructed = coeffs_to_block_torch(coeff, operation.change_of_basis).squeeze(0)
        edge_coeffs.setdefault(op_key, []).append(coeff)
        edge_reconstructed.append(reconstructed.reshape(-1))
        edge_block_rows.append(
            {
                "op_key": op_key,
                "unique_edge_index": int(unique_edge_index),
                "edge_type": int(edge_type),
                "block_shape": list(shape),
                "coefficients": int(coeff.numel()),
                "reconstruction_max_abs_eV": float((reconstructed - block).abs().max().item()),
            }
        )

    return {
        "node_coeffs": {key: torch.stack(value) for key, value in node_coeffs.items()},
        "edge_coeffs": {key: torch.stack(value) for key, value in edge_coeffs.items()},
        "node_labels": torch.cat(node_reconstructed),
        "edge_labels": torch.cat(edge_reconstructed),
        "node_blocks": node_block_rows,
        "edge_blocks": edge_block_rows,
    }


def edge_module_for_unique_type(readout: Any, edge_type: int) -> tuple[str, Any]:
    graph_edge_type = int(abs(readout.edge_types_to_graph2mat[torch.tensor([edge_type])][0].item()))
    for module_key, operation in readout.interactions.items():
        _, _, op_edge_type = map(int, module_key[1:-1].split(","))
        if abs(op_edge_type) == graph_edge_type:
            return module_key, operation
    raise KeyError(f"No readout edge operation found for edge type {edge_type}")


def compute_edge_messages(readout: Any, data_for_readout: Any, node_feats: torch.Tensor, edge_messages: torch.Tensor | None) -> torch.Tensor:
    if edge_messages is not None:
        return edge_messages
    if readout.preprocessing_edges is None:
        raise ValueError("edge_messages must be provided when readout.preprocessing_edges is None")
    preprocessing_out = readout.preprocessing_edges(data=data_for_readout, node_feats=node_feats)
    if not isinstance(preprocessing_out, tuple) or len(preprocessing_out) != 2:
        raise TypeError("Expected preprocessing_edges to return (node_feats_for_edges, edge_messages)")
    return preprocessing_out[1]


def edge_operation_inputs(
    readout: Any,
    batch: Any,
    node_feats: torch.Tensor,
    edge_messages: torch.Tensor,
    module_key: str,
) -> dict[str, Any]:
    point_type, neigh_type, edge_type = map(int, module_key[1:-1].split(","))
    graph2mat_edge_types = readout.edge_types_to_graph2mat[batch.edge_types]
    mask = abs(graph2mat_edge_types) == abs(edge_type)
    if not bool(mask.any()):
        return {}
    type_edge_index = batch.edge_index[:, mask]
    filtered_edge_messages = edge_messages[mask]
    if point_type == neigh_type:
        if readout.symmetric:
            i_edges = slice(0, None, 2)
            j_edges = slice(1, None, 2)
        else:
            i_edges = slice(None)
            j_edges = slice(None)
    else:
        local_types = graph2mat_edge_types[mask]
        i_edges = local_types == edge_type
        j_edges = ~i_edges
    operation = readout.interactions[module_key]
    n_expected_inputs = len(getattr(getattr(operation, "operation", None), "tensor_products", []))
    edge_tuple = (filtered_edge_messages[i_edges], filtered_edge_messages[j_edges])
    node_tuple = (
        node_feats[type_edge_index[0, i_edges]],
        node_feats[type_edge_index[1, i_edges]],
    )
    if n_expected_inputs <= 1:
        return {"edge_messages": edge_tuple}
    return {"edge_messages": edge_tuple, "node_feats": node_tuple}


def predicted_coefficients(
    readout: Any,
    data_for_readout: Any,
    batch: Any,
    node_feats: torch.Tensor,
    edge_messages: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    graph_node_types = readout.types_to_graph2mat[batch.point_types]
    coeffs: dict[str, torch.Tensor] = {}
    for op_index, operation in enumerate(readout.self_interactions):
        if operation is None:
            continue
        mask = graph_node_types == op_index
        if not bool(mask.any()):
            continue
        coeffs[f"node:{op_index}"] = operation.operation(node_feats=node_feats[mask])

    edge_messages = compute_edge_messages(readout, data_for_readout, node_feats, edge_messages)
    for module_key, operation in readout.interactions.items():
        if operation is None:
            continue
        inputs = edge_operation_inputs(readout, batch, node_feats, edge_messages, module_key)
        if not inputs:
            continue
        if getattr(operation, "symm_transpose", False):
            block = operation(**inputs)
            coeffs[f"edge:{module_key}"] = project_block_to_coeffs_torch(block, operation.change_of_basis)
        else:
            coeffs[f"edge:{module_key}"] = operation.operation(**inputs)
    return coeffs


def compare_coefficients(predicted: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(target):
        if key not in predicted:
            rows.append({"op_key": key, "status": "missing_prediction"})
            continue
        metrics = coeff_metrics(predicted[key], target[key].to(predicted[key]))
        rows.append(
            {
                "op_key": key,
                "status": "ok",
                "prediction_shape": list(predicted[key].shape),
                "target_shape": list(target[key].shape),
                **metrics,
            }
        )
    return rows


def coefficient_loss(predicted: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for key, target_value in target.items():
        losses.append(torch.mean(torch.abs(predicted[key] - target_value.to(predicted[key]))))
    return torch.stack(losses).mean()


def train_coefficient_space(
    *,
    readout: torch.nn.Module,
    data_for_readout: Any,
    batch: Any,
    node_feats: torch.Tensor,
    target_coeffs: dict[str, torch.Tensor],
    processor: Any,
    reference_h: np.ndarray,
    schedule: list[float],
    steps_per_stage: int,
    threshold: float,
) -> dict[str, Any]:
    readout.train()
    node_feats = node_feats.detach()
    params = list(readout.parameters())
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    start = time.perf_counter()
    final_output = None
    for stage, lr in enumerate(schedule):
        optimizer = torch.optim.Adam(params, lr=lr)
        stage_best = math.inf
        for _ in range(steps_per_stage):
            optimizer.zero_grad()
            pred_coeffs = predicted_coefficients(readout, data_for_readout, batch, node_feats)
            loss = coefficient_loss(pred_coeffs, target_coeffs)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu().item())
            best_loss = min(best_loss, value)
            stage_best = min(stage_best, value)
        with torch.no_grad():
            pred_coeffs = predicted_coefficients(readout, data_for_readout, batch, node_feats)
            stage_loss = coefficient_loss(pred_coeffs, target_coeffs)
            node_labels, edge_labels = readout(data=data_for_readout, node_feats=node_feats)
            final_output = {"node_labels": node_labels, "edge_labels": edge_labels}
            history.append(
                {
                    "stage": int(stage),
                    "learning_rate": float(lr),
                    "steps": int(steps_per_stage),
                    "stage_best_loss": float(stage_best),
                    "stage_final_loss": float(stage_loss.detach().cpu().item()),
                    "coefficient_errors": compare_coefficients(pred_coeffs, target_coeffs),
                    "direct_node": compare_tensors(node_labels, batch.point_labels),
                    "direct_edge": compare_tensors(edge_labels, batch.edge_labels),
                }
            )
    assert final_output is not None
    runtime = time.perf_counter() - start
    pred_coeffs = predicted_coefficients(readout, data_for_readout, batch, node_feats)
    h_metrics = reconstruct_h_metrics(processor, batch, final_output, reference_h, threshold)
    return {
        "name": "coefficient_space_readout_training",
        "schedule": schedule,
        "steps_per_stage": steps_per_stage,
        "total_steps": steps_per_stage * len(schedule),
        "runtime_sec": runtime,
        "best_loss": best_loss,
        "final_coefficient_loss": float(coefficient_loss(pred_coeffs, target_coeffs).detach().cpu().item()),
        "coefficient_errors": compare_coefficients(pred_coeffs, target_coeffs),
        "direct_node": compare_tensors(final_output["node_labels"], batch.point_labels),
        "direct_edge": compare_tensors(final_output["edge_labels"], batch.edge_labels),
        "h_reconstruction": h_metrics,
        "history": history,
        "reached_numerical_zero": bool(
            h_metrics.get("comparable")
            and float(h_metrics["mae_eV"]) <= 1e-5
            and float(coefficient_loss(pred_coeffs, target_coeffs).detach().cpu().item()) <= 1e-5
        ),
    }


def train_independent_edge_bypass(
    *,
    readout: torch.nn.Module,
    data_for_readout: Any,
    batch: Any,
    initial_node_feats: torch.Tensor,
    initial_edge_messages: torch.Tensor,
    processor: Any,
    reference_h: np.ndarray,
    schedule: list[float],
    steps_per_stage: int,
    threshold: float,
) -> dict[str, Any]:
    class LearnableEdgeMessages(torch.nn.Module):
        def __init__(self, initial: torch.Tensor):
            super().__init__()
            self.edge_messages = torch.nn.Parameter(initial.detach().clone())
            with torch.no_grad():
                self.edge_messages.add_(torch.randn_like(self.edge_messages) * 1e-3)

        def forward(self, data: Any, node_feats: torch.Tensor) -> tuple[None, torch.Tensor]:
            return None, self.edge_messages

    readout.train()
    readout.preprocessing_edges = LearnableEdgeMessages(initial_edge_messages)
    node_feats_param = torch.nn.Parameter(initial_node_feats.detach().clone())
    with torch.no_grad():
        node_feats_param.add_(torch.randn_like(node_feats_param) * 1e-3)
    params = [node_feats_param, *list(readout.parameters())]
    loss_fn = block_type_mae()
    best_loss = math.inf
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    final_output = None
    for stage, lr in enumerate(schedule):
        optimizer = torch.optim.Adam(params, lr=lr)
        stage_best = math.inf
        for _ in range(steps_per_stage):
            optimizer.zero_grad()
            node_labels, edge_labels = readout(
                data=data_for_readout,
                node_feats=node_feats_param,
            )
            loss, _ = loss_fn(
                nodes_pred=node_labels,
                nodes_ref=batch.point_labels,
                edges_pred=edge_labels,
                edges_ref=batch.edge_labels,
                batch=batch,
                basis_table=processor.basis_table,
            )
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu().item())
            best_loss = min(best_loss, value)
            stage_best = min(stage_best, value)
        with torch.no_grad():
            node_labels, edge_labels = readout(
                data=data_for_readout,
                node_feats=node_feats_param,
            )
            stage_loss, stage_stats = loss_fn(
                nodes_pred=node_labels,
                nodes_ref=batch.point_labels,
                edges_pred=edge_labels,
                edges_ref=batch.edge_labels,
                batch=batch,
                basis_table=processor.basis_table,
                log_verbose=True,
            )
            final_output = {"node_labels": node_labels, "edge_labels": edge_labels}
            history.append(
                {
                    "stage": int(stage),
                    "learning_rate": float(lr),
                    "steps": int(steps_per_stage),
                    "stage_best_loss": float(stage_best),
                    "stage_final_loss": float(stage_loss.detach().cpu().item()),
                    "stage_final_stats": sanitize(stage_stats),
                    "direct_node": compare_tensors(node_labels, batch.point_labels),
                    "direct_edge": compare_tensors(edge_labels, batch.edge_labels),
                }
            )
    assert final_output is not None
    runtime = time.perf_counter() - start
    h_metrics = reconstruct_h_metrics(processor, batch, final_output, reference_h, threshold)
    return {
        "name": "independent_edge_message_bypass",
        "schedule": schedule,
        "steps_per_stage": steps_per_stage,
        "total_steps": steps_per_stage * len(schedule),
        "runtime_sec": runtime,
        "best_loss": best_loss,
        "direct_node": compare_tensors(final_output["node_labels"], batch.point_labels),
        "direct_edge": compare_tensors(final_output["edge_labels"], batch.edge_labels),
        "h_reconstruction": h_metrics,
        "history": history,
        "learned_node_features_shape": list(node_feats_param.shape),
        "learned_edge_messages_shape": list(readout.preprocessing_edges.edge_messages.shape),
        "reached_numerical_zero": bool(
            h_metrics.get("comparable")
            and float(h_metrics["mae_eV"]) <= 1e-5
            and compare_tensors(final_output["edge_labels"], batch.edge_labels)["mae"] <= 1e-5
        ),
    }


def operation_coeff_vector(
    readout: Any,
    data_for_readout: Any,
    batch: Any,
    node_feats: torch.Tensor,
    edge_messages: torch.Tensor,
    op_key: str,
) -> torch.Tensor:
    if op_key.startswith("node:"):
        op_index = int(op_key.split(":", 1)[1])
        graph_node_types = readout.types_to_graph2mat[batch.point_types]
        mask = graph_node_types == op_index
        return readout.self_interactions[op_index].operation(node_feats=node_feats[mask]).reshape(-1)
    module_key = op_key.split(":", 1)[1]
    operation = readout.interactions[module_key]
    inputs = edge_operation_inputs(readout, batch, node_feats, edge_messages, module_key)
    if getattr(operation, "symm_transpose", False):
        block = operation(**inputs)
        coeff = project_block_to_coeffs_torch(block, operation.change_of_basis)
        return coeff.reshape(-1)
    return operation.operation(**inputs).reshape(-1)


def randomized_rank_for_operation(
    *,
    readout: Any,
    data_for_readout: Any,
    batch: Any,
    node_feats: torch.Tensor,
    edge_messages: torch.Tensor,
    op_key: str,
    max_directions: int,
    eps: float,
) -> dict[str, Any]:
    module = readout.self_interactions[int(op_key.split(":", 1)[1])] if op_key.startswith("node:") else readout.interactions[op_key.split(":", 1)[1]]
    params = [param for param in module.parameters() if param.requires_grad]
    base = operation_coeff_vector(readout, data_for_readout, batch, node_feats, edge_messages, op_key).detach()
    output_dim = int(base.numel())
    n_directions = min(max_directions, max(output_dim * 2 + 10, 32))
    columns: list[torch.Tensor] = []
    generator = torch.Generator(device=base.device)
    generator.manual_seed(1729 + sum(ord(char) for char in op_key))
    with torch.no_grad():
        for _ in range(n_directions):
            directions = [torch.randn(param.shape, generator=generator, device=param.device, dtype=param.dtype) for param in params]
            for param, direction in zip(params, directions):
                param.add_(direction, alpha=eps)
            plus = operation_coeff_vector(readout, data_for_readout, batch, node_feats, edge_messages, op_key).detach()
            for param, direction in zip(params, directions):
                param.add_(direction, alpha=-2.0 * eps)
            minus = operation_coeff_vector(readout, data_for_readout, batch, node_feats, edge_messages, op_key).detach()
            for param, direction in zip(params, directions):
                param.add_(direction, alpha=eps)
            columns.append(((plus - minus) / (2.0 * eps)).double().cpu())
    sample_matrix = torch.stack(columns, dim=1).numpy()
    singular_values = np.linalg.svd(sample_matrix, compute_uv=False)
    tolerance = float(np.max(singular_values) * max(sample_matrix.shape) * np.finfo(float).eps) if singular_values.size else math.nan
    rel_tolerance = float(np.max(singular_values) * 1e-6) if singular_values.size else math.nan
    effective_rank = int(np.sum(singular_values > max(tolerance, rel_tolerance))) if singular_values.size else 0
    nonzero = singular_values[singular_values > max(tolerance, rel_tolerance)] if singular_values.size else np.array([])
    condition = math.inf if nonzero.size == 0 else float(nonzero.max() / nonzero.min())
    return {
        "op_key": op_key,
        "output_dim": output_dim,
        "random_directions": int(n_directions),
        "local_rank_lower_bound": effective_rank,
        "full_rank_observed": bool(effective_rank >= min(output_dim, n_directions)),
        "singular_value_max": float(singular_values.max()) if singular_values.size else math.nan,
        "singular_value_min_effective": float(nonzero.min()) if nonzero.size else math.nan,
        "condition_estimate": condition,
        "rank_tolerance": max(tolerance, rel_tolerance) if singular_values.size else math.nan,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sanitize(rows))


def final_verdict(payload: dict[str, Any]) -> dict[str, str]:
    coeff_train = payload["coefficient_space_training"]
    bypass = payload["independent_edge_message_bypass"]
    rank_rows = payload["jacobian_rank"]
    low_rank = [
        row
        for row in rank_rows
        if row["local_rank_lower_bound"] < min(row["output_dim"], row["random_directions"])
    ]
    if coeff_train.get("reached_numerical_zero"):
        return {
            "status": "C_MATRIX_LOSS_OR_JOINT_OPTIMIZATION_CONDITIONING",
            "summary": (
                "The readout can reach numerical-zero when optimized directly in coefficient space. "
                "The previous matrix-space/readout training failure is therefore likely a loss conditioning "
                "or joint optimization issue, not a coefficient representability issue."
            ),
        }
    if bypass.get("reached_numerical_zero"):
        return {
            "status": "B_MACE_EDGE_FEATURES_INSUFFICIENT",
            "summary": (
                "Independent learnable node features and edge messages let the unchanged readout memorize. "
                "The bottleneck is MACE feature generation and/or Graph2Mat edge preprocessing."
            ),
        }
    if low_rank:
        return {
            "status": "A_READOUT_OPERATION_LOCAL_RANK_LIMIT",
            "summary": (
                "At least one operation has a randomized local rank below the coefficient dimension. "
                "This is evidence that the current readout operation is locally constrained."
            ),
        }
    return {
        "status": "D_INCONCLUSIVE_OR_OPTIMIZATION_HARD",
        "summary": (
            "Coefficient-space and independent-edge bypass probes did not reach numerical zero within "
            "the diagnostic budget, while randomized rank did not prove a hard local rank limit. "
            "The remaining evidence points to difficult readout optimization/conditioning or a nonlinear "
            "readout bottleneck not resolved by these short probes."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    verdict = payload["final_verdict"]
    sanity = payload["target_coefficient_reconstruction"]
    coeff_train = payload["coefficient_space_training"]
    bypass = payload["independent_edge_message_bypass"]
    lines = [
        "# H2O Coefficient-Space Readout Analysis",
        "",
        f"Source workspace: `{payload['source_workspace']}`",
        f"Checkpoint: `{payload['checkpoint_path']}`",
        "",
        "## Executive Verdict",
        "",
        f"**{verdict['status']}**: {verdict['summary']}",
        "",
        "## Target Coefficient Reconstruction Sanity Check",
        "",
        f"- H MAE: `{sanity['h_reconstruction'].get('mae_meV')}` meV",
        f"- H RMSE: `{sanity['h_reconstruction'].get('rmse_meV')}` meV",
        f"- max abs: `{sanity['h_reconstruction'].get('max_abs_eV')}` eV",
        f"- support F1: `{sanity['h_reconstruction'].get('support_f1')}`",
        "",
        "## Best Checkpoint Coefficient Errors",
        "",
        "| op | n coeff | MAE | RMSE | max abs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["checkpoint_coefficient_errors"]:
        lines.append(f"| `{row['op_key']}` | {row.get('n_coefficients', '')} | {row.get('mae', '')} | {row.get('rmse', '')} | {row.get('max_abs', '')} |")
    lines.extend(
        [
            "",
            "## Coefficient-Space Training",
            "",
            f"- steps: `{coeff_train['total_steps']}`",
            f"- final coefficient loss: `{coeff_train['final_coefficient_loss']}`",
            f"- H MAE: `{coeff_train['h_reconstruction'].get('mae_meV')}` meV",
            f"- H RMSE: `{coeff_train['h_reconstruction'].get('rmse_meV')}` meV",
            f"- reached numerical zero: `{coeff_train['reached_numerical_zero']}`",
            "",
            "## Independent Edge-Message Bypass",
            "",
            f"- steps: `{bypass['total_steps']}`",
            f"- learned node features: `{bypass['learned_node_features_shape']}`",
            f"- learned edge messages: `{bypass['learned_edge_messages_shape']}`",
            f"- H MAE: `{bypass['h_reconstruction'].get('mae_meV')}` meV",
            f"- H RMSE: `{bypass['h_reconstruction'].get('rmse_meV')}` meV",
            f"- reached numerical zero: `{bypass['reached_numerical_zero']}`",
            "",
            "## Randomized Jacobian/Rank Diagnostics",
            "",
            "| op | output dim | directions | rank lower bound | full rank observed | condition estimate |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    for row in payload["jacobian_rank"]:
        lines.append(
            f"| `{row['op_key']}` | {row['output_dim']} | {row['random_directions']} | "
            f"{row['local_rank_lower_bound']} | {row['full_rank_observed']} | {row['condition_estimate']} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- JSON: `{payload['output_files']['json']}`",
            f"- Coefficient errors CSV: `{payload['output_files']['coefficient_errors_csv']}`",
            f"- Rank CSV: `{payload['output_files']['rank_csv']}`",
            "",
            "## Commands",
            "",
            "```bash",
            payload["command"],
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reference_h_path(source_workspace: Path) -> Path:
    candidates = sorted((source_workspace / "dataset" / "samples").glob("*/siesta.TSHS"))
    if not candidates:
        raise FileNotFoundError(f"No reference siesta.TSHS found under {source_workspace / 'dataset' / 'samples'}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--learning-rate-schedule", default="0.0001,0.00002")
    parser.add_argument("--coeff-steps-per-stage", type=int, default=160)
    parser.add_argument("--bypass-steps-per-stage", type=int, default=160)
    parser.add_argument("--rank-max-directions", type=int, default=180)
    parser.add_argument("--rank-eps", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=1e-5)
    args = parser.parse_args()

    source_workspace = args.source_workspace.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = source_workspace / "training"
    config_path = training_dir / "config.yaml"
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else latest_checkpoint(training_dir)
    schedule = [float(item) for item in args.learning_rate_schedule.split(",") if item.strip()]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = dict(config["data"])
    reference_path = reference_h_path(source_workspace)
    reference_h = np.asarray(read_matrix(reference_path).hamiltonian.toarray(), dtype=float)

    with working_directory(training_dir):
        datamodule = MatrixDataModule(**build_datamodule_kwargs(data))
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        processor = datamodule.data_processor
        model = LitMACEMatrixModel.load_from_checkpoint(str(checkpoint_path))
        model.eval()
        with torch.no_grad():
            data_for_readout, node_feats, edge_messages = make_readout_inputs(model, batch)
            readout = model.model.matrix_readouts
            target_pack = target_coefficients_and_reconstructed_labels(readout, processor, batch)
            target_coeffs = {**target_pack["node_coeffs"], **target_pack["edge_coeffs"]}
            target_output = {
                "node_labels": target_pack["node_labels"],
                "edge_labels": target_pack["edge_labels"],
            }
            target_h_metrics = reconstruct_h_metrics(processor, batch, target_output, reference_h, args.threshold)
            checkpoint_coeffs = predicted_coefficients(readout, data_for_readout, batch, node_feats)
            checkpoint_coeff_errors = compare_coefficients(checkpoint_coeffs, target_coeffs)
            checkpoint_output = model(batch)
            checkpoint_h_metrics = reconstruct_h_metrics(processor, batch, checkpoint_output, reference_h, args.threshold)

        coeff_model = LitMACEMatrixModel.load_from_checkpoint(str(checkpoint_path))
        with torch.no_grad():
            coeff_data_for_readout, coeff_node_feats, _ = make_readout_inputs(coeff_model, batch)
        coeff_training = train_coefficient_space(
            readout=coeff_model.model.matrix_readouts,
            data_for_readout=coeff_data_for_readout,
            batch=batch,
            node_feats=coeff_node_feats,
            target_coeffs=target_coeffs,
            processor=processor,
            reference_h=reference_h,
            schedule=schedule,
            steps_per_stage=args.coeff_steps_per_stage,
            threshold=args.threshold,
        )

        bypass_model = LitMACEMatrixModel.load_from_checkpoint(str(checkpoint_path))
        with torch.no_grad():
            bypass_data_for_readout, bypass_node_feats, bypass_edge_messages = make_readout_inputs(bypass_model, batch)
        bypass = train_independent_edge_bypass(
            readout=bypass_model.model.matrix_readouts,
            data_for_readout=bypass_data_for_readout,
            batch=batch,
            initial_node_feats=bypass_node_feats,
            initial_edge_messages=bypass_edge_messages,
            processor=processor,
            reference_h=reference_h,
            schedule=schedule,
            steps_per_stage=args.bypass_steps_per_stage,
            threshold=args.threshold,
        )

        rank_model = LitMACEMatrixModel.load_from_checkpoint(str(checkpoint_path))
        rank_model.eval()
        with torch.no_grad():
            rank_data_for_readout, rank_node_feats, rank_edge_messages = make_readout_inputs(rank_model, batch)
        rank_rows = []
        for op_key in sorted(target_coeffs):
            rank_rows.append(
                randomized_rank_for_operation(
                    readout=rank_model.model.matrix_readouts,
                    data_for_readout=rank_data_for_readout,
                    batch=batch,
                    node_feats=rank_node_feats,
                    edge_messages=rank_edge_messages,
                    op_key=op_key,
                    max_directions=args.rank_max_directions,
                    eps=args.rank_eps,
                )
            )

    graph2mat_root = Path("/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat")
    coefficient_errors_csv = output_dir / "checkpoint_coefficient_errors.csv"
    rank_csv = output_dir / "jacobian_rank.csv"
    json_path = output_dir / "coefficient_space_readout_analysis.json"
    report_path = output_dir / "report.md"
    output_files = {
        "json": str(json_path),
        "coefficient_errors_csv": str(coefficient_errors_csv),
        "rank_csv": str(rank_csv),
        "report": str(report_path),
    }
    payload = {
        "source_workspace": str(source_workspace),
        "training_dir": str(training_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "reference_path": str(reference_path),
        "command": " ".join([sys.executable, *sys.argv]),
        "repository_context": {
            "pipeline_branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT),
            "pipeline_commit": command_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
            "graph2mat_branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=graph2mat_root),
            "graph2mat_commit": command_output(["git", "rev-parse", "HEAD"], cwd=graph2mat_root),
        },
        "data_policy": {
            "out_matrix": data.get("out_matrix"),
            "symmetric_matrix": data.get("symmetric_matrix"),
            "sub_point_matrix": data.get("sub_point_matrix"),
            "matrix_component_policy": data.get("matrix_component_policy"),
            "n_matrix_components": data.get("n_matrix_components"),
        },
        "target_coefficient_reconstruction": {
            "node_blocks": target_pack["node_blocks"],
            "edge_blocks": target_pack["edge_blocks"],
            "h_reconstruction": target_h_metrics,
        },
        "checkpoint_coefficient_errors": checkpoint_coeff_errors,
        "checkpoint_h_reconstruction": checkpoint_h_metrics,
        "coefficient_space_training": coeff_training,
        "independent_edge_message_bypass": bypass,
        "jacobian_rank": rank_rows,
        "output_files": output_files,
    }
    payload["final_verdict"] = final_verdict(payload)

    write_csv(
        coefficient_errors_csv,
        checkpoint_coeff_errors,
        ["op_key", "status", "prediction_shape", "target_shape", "n_coefficients", "mae", "rmse", "max_abs"],
    )
    write_csv(
        rank_csv,
        rank_rows,
        [
            "op_key",
            "output_dim",
            "random_directions",
            "local_rank_lower_bound",
            "full_rank_observed",
            "singular_value_max",
            "singular_value_min_effective",
            "condition_estimate",
            "rank_tolerance",
        ],
    )
    safe_payload = sanitize(payload)
    json_path.write_text(json.dumps(safe_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report_path, safe_payload)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "report": str(report_path),
                "verdict": safe_payload["final_verdict"],
                "coefficient_training_h_mae_meV": safe_payload["coefficient_space_training"]["h_reconstruction"].get("mae_meV"),
                "independent_edge_bypass_h_mae_meV": safe_payload["independent_edge_message_bypass"]["h_reconstruction"].get("mae_meV"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
