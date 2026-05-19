#!/usr/bin/env python3
"""Project one-sample H2O Hamiltonian labels onto Graph2Mat/e3nn block bases.

This is a diagnostic-only script. It checks whether the H-only SIESTA target
blocks are in the linear subspace spanned by each Graph2Mat block operation's
``change_of_basis`` tensor. No production training/evaluation code is changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
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
    latest_checkpoint,
    sanitize,
)
from graph2mat.bindings.e3nn.irreps_tools import ReducedTensorProducts  # noqa: E402
from graph2mat.tools.lightning import MatrixDataModule  # noqa: E402
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Comparison"
    / "results"
    / "diagnostics"
    / "h2o_hamiltonian"
    / "representability_projection"
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


def species_for_point_type(table: Any, point_type: int) -> str:
    basis = table.basis[int(point_type)]
    return str(getattr(basis, "type", point_type))


def species_symbol(species: str) -> str:
    return {"1": "H", "6": "C", "7": "N", "8": "O", "14": "Si"}.get(str(species), str(species))


def projection_metrics(block: np.ndarray, change_of_basis: np.ndarray) -> dict[str, Any]:
    """Least-squares project ``block`` onto span(change_of_basis)."""
    target = np.asarray(block, dtype=np.float64).reshape(-1)
    cob = np.asarray(change_of_basis, dtype=np.float64)
    design = cob.reshape(cob.shape[0], -1).T
    coeffs, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    reconstructed = design @ coeffs
    delta = reconstructed - target
    abs_delta = np.abs(delta)
    ref_norm = float(np.linalg.norm(target))
    return {
        "basis_dimension": int(design.shape[1]),
        "block_dimension": int(design.shape[0]),
        "basis_rank": int(rank),
        "rank_deficiency": int(design.shape[1] - rank),
        "condition_number": (
            math.inf
            if singular_values.size == 0 or float(np.min(singular_values)) == 0.0
            else float(np.max(singular_values) / np.min(singular_values))
        ),
        "mae_eV": float(abs_delta.mean()) if abs_delta.size else math.nan,
        "mae_meV": float(abs_delta.mean() * 1000.0) if abs_delta.size else math.nan,
        "rmse_eV": float(np.sqrt(np.mean(delta * delta))) if delta.size else math.nan,
        "rmse_meV": float(np.sqrt(np.mean(delta * delta)) * 1000.0) if delta.size else math.nan,
        "max_abs_eV": float(abs_delta.max()) if abs_delta.size else math.nan,
        "relative_frobenius": math.nan if ref_norm == 0.0 else float(np.linalg.norm(delta) / ref_norm),
        "target_norm": ref_norm,
        "coeff_norm": float(np.linalg.norm(coeffs)),
        "reconstructed_block": reconstructed.reshape(block.shape),
        "delta_block": delta.reshape(block.shape),
    }


def compact_projection(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"reconstructed_block", "delta_block"}
    }


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def node_projection_rows(processor: Any, readout: Any, batch: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table = processor.basis_table
    point_types = tensor_to_numpy(batch.point_types).astype(int)
    labels = tensor_to_numpy(batch.point_labels).reshape(-1)
    pointers = table.point_block_pointer(point_types)
    rows: list[dict[str, Any]] = []
    orbital_rows: list[dict[str, Any]] = []
    graph2mat_node_types = tensor_to_numpy(readout.types_to_graph2mat[batch.point_types]).astype(int)

    for atom_index, point_type in enumerate(point_types):
        species = species_symbol(species_for_point_type(table, int(point_type)))
        shape = tuple(int(x) for x in table.point_block_shape[:, point_type])
        block = labels[pointers[atom_index] : pointers[atom_index + 1]].reshape(shape)
        op_index = int(graph2mat_node_types[atom_index])
        operation = readout.self_interactions[op_index]
        current = projection_metrics(block, tensor_to_numpy(operation.change_of_basis))

        point_basis = readout.graph2mat_table.basis[op_index]
        nonsymmetric_rtp = ReducedTensorProducts(
            "ij",
            i=point_basis.e3nn_irreps,
            j=point_basis.e3nn_irreps,
        )
        nonsymmetric = projection_metrics(block, tensor_to_numpy(nonsymmetric_rtp.change_of_basis))

        row = {
            "kind": "node",
            "atom_index": int(atom_index),
            "row_atom": int(atom_index),
            "col_atom": int(atom_index),
            "species_pair": f"{species}-{species}",
            "operation_key": str(op_index),
            "symmetry": "ij=ji",
            "variant": "current_symmetric_self",
            "block_shape": list(shape),
            "irreps_out": str(getattr(operation, "_irreps_out", "")),
            "change_of_basis_shape": list(tensor_to_numpy(operation.change_of_basis).shape),
            "symmetry_variant_non_symmetric": compact_projection(nonsymmetric),
            "target_asymmetry_max_abs_eV": float(np.abs(block - block.T).max()) if shape[0] == shape[1] else math.nan,
            **compact_projection(current),
        }
        rows.append(row)
        delta = current["delta_block"]
        for i in range(shape[0]):
            for j in range(shape[1]):
                orbital_rows.append(
                    {
                        "kind": "node",
                        "species_pair": row["species_pair"],
                        "row_orbital_index": int(i),
                        "col_orbital_index": int(j),
                        "n_entries": 1,
                        "abs_error_eV": float(abs(delta[i, j])),
                        "signed_error_eV": float(delta[i, j]),
                        "block_ref": f"atom_{atom_index}",
                    }
                )
    return rows, orbital_rows


def edge_module_for_type(readout: Any, edge_type: int) -> tuple[str, Any]:
    graph_edge_type = int(abs(readout.edge_types_to_graph2mat[torch.tensor([edge_type])][0].item()))
    for module_key, operation in readout.interactions.items():
        _, _, op_edge_type = map(int, module_key[1:-1].split(","))
        if abs(op_edge_type) == graph_edge_type:
            return module_key, operation
    raise KeyError(f"No readout edge operation found for edge type {edge_type}")


def edge_projection_rows(processor: Any, readout: Any, batch: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table = processor.basis_table
    edge_types_all = tensor_to_numpy(batch.edge_types).astype(int)
    edge_types = edge_types_all[::2]
    labels = tensor_to_numpy(batch.edge_labels).reshape(-1)
    pointers = table.edge_block_pointer(edge_types)
    edge_index = tensor_to_numpy(batch.edge_index).astype(int)
    point_types = tensor_to_numpy(batch.point_types).astype(int)
    rows: list[dict[str, Any]] = []
    orbital_rows: list[dict[str, Any]] = []

    for unique_edge_index, edge_type in enumerate(edge_types):
        shape = tuple(int(x) for x in table.edge_block_shape[:, abs(edge_type)])
        block = labels[pointers[unique_edge_index] : pointers[unique_edge_index + 1]].reshape(shape)
        module_key, operation = edge_module_for_type(readout, int(edge_type))
        current = projection_metrics(block, tensor_to_numpy(operation.change_of_basis))

        forward_edge = unique_edge_index * 2
        row_atom = int(edge_index[0, forward_edge])
        col_atom = int(edge_index[1, forward_edge])
        row_species = species_symbol(species_for_point_type(table, int(point_types[row_atom])))
        col_species = species_symbol(species_for_point_type(table, int(point_types[col_atom])))
        row = {
            "kind": "edge",
            "edge_pair_index": int(unique_edge_index),
            "row_atom": row_atom,
            "col_atom": col_atom,
            "edge_type": int(edge_type),
            "species_pair": f"{row_species}-{col_species}",
            "operation_key": module_key,
            "symmetry": "ij",
            "variant": "current_edge",
            "block_shape": list(shape),
            "irreps_out": str(getattr(operation, "_irreps_out", "")),
            "change_of_basis_shape": list(tensor_to_numpy(operation.change_of_basis).shape),
            "symm_transpose": bool(getattr(operation, "symm_transpose", False)),
            **compact_projection(current),
        }
        rows.append(row)
        delta = current["delta_block"]
        for i in range(shape[0]):
            for j in range(shape[1]):
                orbital_rows.append(
                    {
                        "kind": "edge",
                        "species_pair": row["species_pair"],
                        "row_orbital_index": int(i),
                        "col_orbital_index": int(j),
                        "n_entries": 1,
                        "abs_error_eV": float(abs(delta[i, j])),
                        "signed_error_eV": float(delta[i, j]),
                        "block_ref": f"{row_atom}_{col_atom}",
                    }
                )
    return rows, orbital_rows


def aggregate_orbital_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["kind"],
            row["species_pair"],
            row["row_orbital_index"],
            row["col_orbital_index"],
        )
        groups.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for (kind, species_pair, row_orb, col_orb), values in groups.items():
        abs_errors = np.array([item["abs_error_eV"] for item in values], dtype=float)
        signed_errors = np.array([item["signed_error_eV"] for item in values], dtype=float)
        result.append(
            {
                "kind": kind,
                "species_pair": species_pair,
                "row_orbital_index": int(row_orb),
                "col_orbital_index": int(col_orb),
                "n_entries": int(len(values)),
                "mae_eV": float(abs_errors.mean()),
                "mae_meV": float(abs_errors.mean() * 1000.0),
                "rmse_eV": float(np.sqrt(np.mean(signed_errors * signed_errors))),
                "rmse_meV": float(np.sqrt(np.mean(signed_errors * signed_errors)) * 1000.0),
                "max_abs_eV": float(abs_errors.max()),
                "mean_signed_error_eV": float(signed_errors.mean()),
                "block_refs": ",".join(sorted(str(item["block_ref"]) for item in values)),
            }
        )
    return sorted(result, key=lambda item: item["mae_eV"], reverse=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def top_rows_by_float(rows: list[dict[str, str]], key: str, limit: int) -> list[dict[str, str]]:
    def value(row: dict[str, str]) -> float:
        try:
            return float(row.get(key, "nan"))
        except ValueError:
            return math.nan

    return sorted(rows, key=value, reverse=True)[:limit]


def projection_verdict(rows: list[dict[str, Any]]) -> dict[str, str]:
    finite_max = [
        float(row["max_abs_eV"])
        for row in rows
        if row.get("max_abs_eV") is not None and math.isfinite(float(row["max_abs_eV"]))
    ]
    worst = max(finite_max) if finite_max else math.nan
    if math.isfinite(worst) and worst <= 1e-6:
        return {
            "status": "A_TARGET_BLOCKS_REPRESENTABLE",
            "summary": (
                "Every target block projects back through the current Graph2Mat/e3nn change_of_basis "
                "to numerical precision. The one-sample failure is therefore not explained by a "
                "SIESTA block outside the current reduced-tensor-product subspace."
            ),
        }
    if math.isfinite(worst) and worst <= 1e-4:
        return {
            "status": "A_TARGET_BLOCKS_REPRESENTABLE_WITH_FLOAT_TOLERANCE",
            "summary": (
                "Projection residuals are tiny but above a strict 1e-6 eV tolerance. This still points "
                "away from a representability failure and toward optimization/readout training."
            ),
        }
    if math.isfinite(worst):
        return {
            "status": "B_TARGET_BLOCKS_NOT_REPRESENTABLE",
            "summary": (
                "At least one target block has a non-negligible projection residual under the current "
                "Graph2Mat/e3nn block basis. Inspect the block and orbital residual tables."
            ),
        }
    return {
        "status": "C_INCONCLUSIVE",
        "summary": "No finite projection residuals were produced; inspect script errors and input paths.",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(sanitize(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_block_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "kind",
        "row_atom",
        "col_atom",
        "species_pair",
        "operation_key",
        "symmetry",
        "variant",
        "block_shape",
        "irreps_out",
        "change_of_basis_shape",
        "basis_dimension",
        "block_dimension",
        "basis_rank",
        "rank_deficiency",
        "condition_number",
        "target_asymmetry_max_abs_eV",
        "mae_eV",
        "mae_meV",
        "rmse_eV",
        "rmse_meV",
        "max_abs_eV",
        "relative_frobenius",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sanitize(rows))


def write_orbital_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "kind",
        "species_pair",
        "row_orbital_index",
        "col_orbital_index",
        "n_entries",
        "mae_eV",
        "mae_meV",
        "rmse_eV",
        "rmse_meV",
        "max_abs_eV",
        "mean_signed_error_eV",
        "block_refs",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sanitize(rows))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    verdict = payload["executive_verdict"]
    lines = [
        "# H2O Representability Projection",
        "",
        f"Source workspace: `{payload['source_workspace']}`",
        f"Checkpoint: `{payload['checkpoint_path']}`",
        "",
        "## Executive Verdict",
        "",
        f"**{verdict['status']}**: {verdict['summary']}",
        "",
        "## Repository Context",
        "",
        f"- Pipeline branch/commit: `{payload['repository_context']['pipeline_branch']}` / `{payload['repository_context']['pipeline_commit']}`",
        f"- Graph2Mat branch/commit: `{payload['repository_context']['graph2mat_branch']}` / `{payload['repository_context']['graph2mat_commit']}`",
        "",
        "## Basis And Symmetry",
        "",
        f"- basis convention: `{payload['basis']['basis_convention']}`",
        f"- symmetric matrix: `{payload['data_policy']['symmetric_matrix']}`",
        f"- self-block symmetry tested: `ij=ji` plus non-symmetric `ij` comparison",
        f"- edge-block symmetry tested: `ij`",
        f"- n_matrix_components: `{payload['data_policy']['n_matrix_components']}`",
        "",
        "## Projection Residuals By Block",
        "",
        "| kind | atoms | species | op | symmetry | dim/rank | MAE meV | RMSE meV | max eV | rel Frobenius | asym max eV |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["projection_blocks"]:
        lines.append(
            f"| {row['kind']} | {row['row_atom']}-{row['col_atom']} | {row['species_pair']} | `{row['operation_key']}` | "
            f"`{row['symmetry']}` | {row['basis_dimension']}/{row['basis_rank']} | {row['mae_meV']} | "
            f"{row['rmse_meV']} | {row['max_abs_eV']} | {row['relative_frobenius']} | "
            f"{row.get('target_asymmetry_max_abs_eV', '')} |"
        )
    lines.extend(
        [
            "",
            "## Top Projection Residuals By Orbital Pair",
            "",
            "| kind | species pair | row orb | col orb | n | MAE meV | RMSE meV | max eV | refs |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["projection_orbital_top"]:
        lines.append(
            f"| {row['kind']} | {row['species_pair']} | {row['row_orbital_index']} | {row['col_orbital_index']} | "
            f"{row['n_entries']} | {row['mae_meV']} | {row['rmse_meV']} | {row['max_abs_eV']} | {row['block_refs']} |"
        )
    lines.extend(
        [
            "",
            "## Comparison Against Trained MACE Residuals",
            "",
            f"- Worst projection block max abs error: `{payload['projection_summary']['worst_block_max_abs_eV']}` eV",
            f"- Worst projection orbital-pair MAE: `{payload['projection_summary']['worst_orbital_mae_meV']}` meV",
            "- Top trained MACE block residuals from the existing evaluation CSV:",
            "",
            "| row atom | col atom | species | n | MAE eV | RMSE eV | max eV |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["trained_mace_residuals"]["block_top_by_mae"][:8]:
        species_pair = row.get("species_pair") or "-".join(
            item for item in [row.get("row_species", ""), row.get("col_species", "")] if item
        )
        lines.append(
            f"| {row.get('row_atom', '')} | {row.get('col_atom', '')} | {species_pair} | "
            f"{row.get('n_entries', '')} | {row.get('mae_union_eV', '')} | {row.get('rmse_union_eV', '')} | "
            f"{row.get('max_abs_error_union_eV', '')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Output Files",
            "",
            f"- JSON: `{payload['output_files']['json']}`",
            f"- Block CSV: `{payload['output_files']['block_csv']}`",
            f"- Orbital CSV: `{payload['output_files']['orbital_csv']}`",
            "",
            "## Commands",
            "",
            "```bash",
            payload["command"],
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    source_workspace = args.source_workspace.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_dir = source_workspace / "training"
    config_path = training_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training config: {config_path}")
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else latest_checkpoint(training_dir)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = dict(config["data"])

    with working_directory(training_dir):
        datamodule = MatrixDataModule(**build_datamodule_kwargs(data))
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        processor = datamodule.data_processor
        model = LitMACEMatrixModel.load_from_checkpoint(str(checkpoint_path))
        model.eval()
        readout = model.model.matrix_readouts

        node_rows, node_orbitals = node_projection_rows(processor, readout, batch)
        edge_rows, edge_orbitals = edge_projection_rows(processor, readout, batch)

    projection_blocks = node_rows + edge_rows
    projection_orbitals = aggregate_orbital_rows(node_orbitals + edge_orbitals)
    projection_orbital_top = projection_orbitals[:20]
    verdict = projection_verdict(projection_blocks)
    worst_block = max((float(row["max_abs_eV"]) for row in projection_blocks), default=math.nan)
    worst_orbital = max((float(row["mae_meV"]) for row in projection_orbitals), default=math.nan)

    metrics_dir = source_workspace / "evaluation" / "metrics"
    trained_residuals = {
        "block_top_by_mae": top_rows_by_float(read_csv_rows(metrics_dir / "block_metrics.csv"), "mae_union_eV", 12),
        "orbital_pair_top_by_mae_meV": top_rows_by_float(
            read_csv_rows(metrics_dir / "orbital_pair_metrics.csv"), "mae_union_meV", 20
        ),
    }
    if verdict["status"].startswith("A_"):
        interpretation = (
            "Projection residuals are at numerical precision while the trained MACE checkpoint has large "
            "O-H/H-O and O-O residuals. The SIESTA H blocks are representable by the current e3nn "
            "block bases; the remaining failure is therefore optimization/readout training dynamics, "
            "feature generation, or the specific nonlinear map into those coefficients."
        )
    elif verdict["status"].startswith("B_"):
        interpretation = (
            "Projection residuals are non-negligible. This supports the hypothesis that current "
            "Graph2Mat/e3nn block bases, symmetry assumptions, or orbital convention exclude part "
            "of the SIESTA target."
        )
    else:
        interpretation = "The projection diagnostic did not produce a decisive finite result."

    json_path = output_dir / "representability_projection.json"
    block_csv = output_dir / "projection_block_metrics.csv"
    orbital_csv = output_dir / "projection_orbital_pair_metrics.csv"
    report_path = output_dir / "report.md"

    graph2mat_root = Path("/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat")
    payload = {
        "source_workspace": str(source_workspace),
        "training_dir": str(training_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
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
        "basis": {
            "basis_convention": getattr(processor.basis_table, "basis_convention", None),
            "types": [str(item) for item in getattr(processor.basis_table, "types", [])],
            "basis_sizes": sanitize(getattr(processor.basis_table, "basis_size", [])),
        },
        "projection_blocks": projection_blocks,
        "projection_orbital_top": projection_orbital_top,
        "projection_summary": {
            "worst_block_max_abs_eV": worst_block,
            "worst_orbital_mae_meV": worst_orbital,
            "n_blocks": len(projection_blocks),
            "n_orbital_groups": len(projection_orbitals),
        },
        "trained_mace_residuals": trained_residuals,
        "executive_verdict": verdict,
        "interpretation": interpretation,
        "output_files": {
            "json": str(json_path),
            "block_csv": str(block_csv),
            "orbital_csv": str(orbital_csv),
            "report": str(report_path),
        },
    }

    safe_payload = sanitize(payload)
    write_json(json_path, safe_payload)
    write_block_csv(block_csv, projection_blocks)
    write_orbital_csv(orbital_csv, projection_orbitals)
    write_markdown(report_path, safe_payload)

    print(
        json.dumps(
            {
                "json": str(json_path),
                "report": str(report_path),
                "verdict": verdict,
                "worst_block_max_abs_eV": worst_block,
                "worst_orbital_mae_meV": worst_orbital,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
