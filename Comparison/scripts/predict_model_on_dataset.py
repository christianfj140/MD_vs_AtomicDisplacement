#!/usr/bin/env python3
"""Predict Hamiltonians for a manifest-defined test dataset.

Examples
--------
Run a trained MD model on a frozen AtomDisplacement test set::

    python Comparison/scripts/predict_model_on_dataset.py \
      --checkpoint Comparison/workspaces/exp/md/dataset_10/training/lightning_logs/.../best.ckpt \
      --train-method md \
      --test-set test_atomdisp \
      --test-manifest Comparison/results/exp/common_tests/test_atomdisp/test_manifest.csv \
      --basis-files "../dataset/basis/*.ion.xml" \
      --output-dir Comparison/results/exp/cross_predictions/md_on_test_atomdisp
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any

TORCH_COMPAT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "torch_serialization_compat"
sys.path.insert(0, str(TORCH_COMPAT_DIR))
SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
from torch_safe_globals import (
    allow_graph2mat_checkpoint_globals,
    apply_torch_float32_matmul_precision,
)
from graph2mat_material_config import (
    resolve_matrix_component_policy,
    validate_model_matrix_component_policy,
)

allow_graph2mat_checkpoint_globals()

EDGE_LABEL_CONSUMPTION_ERROR = "Predicted edge labels were not fully consumed by yield_from_batch"
EDGE_LABEL_CONSUMPTION_PATTERN = re.compile(r"consumed=(\d+), total=(\d+)")
PREDICTION_OUTPUT_FILE = "ML_prediction.HSX"


def cast_low_precision_tensors_to_float32(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: cast_low_precision_tensors_to_float32(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cast_low_precision_tensors_to_float32(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cast_low_precision_tensors_to_float32(item) for item in value)
    if getattr(value, "is_floating_point", lambda: False)() and value.element_size() < 4:
        return value.float()
    return value


def apply_mace_node_chunking(model: Any, chunk_size: int, torch: Any) -> None:
    if chunk_size <= 0:
        raise ValueError("mace_node_chunk_size must be positive")

    class ChunkedContraction(torch.nn.Module):
        def __init__(self, contraction: Any):
            super().__init__()
            self.contraction = contraction

        def forward(self, x: Any, y: Any) -> Any:
            return torch.cat(
                [
                    self.contraction(x[start : start + chunk_size], y[start : start + chunk_size])
                    for start in range(0, x.shape[0], chunk_size)
                ],
                dim=0,
            )

    for product in model.model.mace.products:
        product.symmetric_contractions = ChunkedContraction(product.symmetric_contractions)


def apply_mace_edge_chunking(model: Any, chunk_size: int, torch: Any) -> None:
    if chunk_size <= 0:
        raise ValueError("mace_edge_chunk_size must be positive")

    for interaction in model.model.mace.interactions:
        if hasattr(interaction, "conv_fusion"):
            raise RuntimeError("MACE edge chunking does not support fused convolutions")

        def forward(
            self,
            node_attrs,
            node_feats,
            edge_attrs,
            edge_feats,
            edge_index,
            cutoff=None,
            lammps_class=None,
            lammps_natoms=(0, 0),
            first_layer=False,
        ):
            n_real = lammps_natoms[0] if lammps_class is not None else None
            sc = self.skip_tp(node_feats, node_attrs)
            node_feats = self.linear_up(node_feats)
            node_feats = self.handle_lammps(
                node_feats,
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
                first_layer=first_layer,
            )
            message = None
            for start in range(0, edge_index.shape[1], chunk_size):
                stop = start + chunk_size
                chunk_index = edge_index[:, start:stop]
                tp_weights = self.conv_tp_weights(edge_feats[start:stop])
                if cutoff is not None:
                    tp_weights = tp_weights * cutoff[start:stop]
                mji = self.conv_tp(
                    node_feats[chunk_index[0]],
                    edge_attrs[start:stop],
                    tp_weights,
                )
                if message is None:
                    message = mji.new_zeros((node_feats.shape[0], mji.shape[1]))
                message.index_add_(0, chunk_index[1], mji)
            if message is None:
                raise RuntimeError("MACE edge chunking received an empty graph")
            message = self.truncate_ghosts(message, n_real)
            node_attrs = self.truncate_ghosts(node_attrs, n_real)
            sc = self.truncate_ghosts(sc, n_real)
            message = self.linear(message) / self.avg_num_neighbors
            return self.reshape(message), sc

        interaction.forward = types.MethodType(forward, interaction)


def apply_graph2mat_edge_chunking(model: Any, chunk_size: int, torch: Any) -> None:
    if chunk_size <= 0:
        raise ValueError("graph2mat_edge_chunk_size must be positive")

    readout = model.model.matrix_readouts
    preprocessor = readout.preprocessing_edges

    def preprocessing_forward(self, data, node_feats):
        sender = data["edge_index"][0]
        edge_attrs = data["edge_attrs"]
        edge_feats = data["edge_feats"]
        node_feats = self.linear_up(node_feats)
        messages = []
        for start in range(0, sender.shape[0], chunk_size):
            stop = start + chunk_size
            weights = self.conv_tp_weights(edge_feats[start:stop])
            messages.append(
                self.linear(
                    self.conv_tp(
                        node_feats[sender[start:stop]],
                        edge_attrs[start:stop],
                        weights,
                    )
                )
            )
        return None, torch.cat(messages, dim=0)

    preprocessor.forward = types.MethodType(preprocessing_forward, preprocessor)

    for block in readout.interactions.values():
        operation = block.operation

        def operation_forward(self, edge_feats, edge_messages, node_feats):
            outputs = []
            for start in range(0, edge_feats[0].shape[0], chunk_size):
                stop = start + chunk_size
                scalar_feats = torch.concatenate(
                    (
                        self.nodes_linear(node_feats[0][start:stop]),
                        self.nodes_linear(node_feats[1][start:stop]),
                        edge_feats[0][start:stop],
                    ),
                    dim=1,
                )
                weights = self.edge_tp_weights(scalar_feats)
                outputs.append(
                    self.output_linear(
                        self.edges_tp(
                            edge_messages[0][start:stop],
                            edge_messages[1][start:stop],
                            weights,
                        )
                    )
                )
            return torch.cat(outputs, dim=0)

        operation.forward = types.MethodType(operation_forward, operation)


def apply_graph2mat_node_chunking(model: Any, chunk_size: int, torch: Any) -> None:
    if chunk_size <= 0:
        raise ValueError("graph2mat_node_chunk_size must be positive")

    for block in model.model.matrix_readouts.self_interactions:
        operation = block.operation

        def operation_forward(self, **node_kwargs):
            node_feats = None
            for value in node_kwargs.values():
                node_feats = value if node_feats is None else node_feats + value
            if node_feats is None:
                raise RuntimeError("Graph2Mat node chunking received no node features")
            return torch.cat(
                [
                    self.tsq(node_feats[start : start + chunk_size])
                    for start in range(0, node_feats.shape[0], chunk_size)
                ],
                dim=0,
            )

        operation.forward = types.MethodType(operation_forward, operation)


def incomplete_prediction_error(exc: ValueError) -> RuntimeError:
    return RuntimeError(
        "Graph2Mat MatrixWriter reported incomplete prediction export: "
        f"{exc}. Existing output files are not accepted as proof of complete, "
        "sample-mapped Hamiltonian predictions."
    )


def edge_label_consumption_counts(exc: ValueError) -> tuple[int, int] | None:
    match = EDGE_LABEL_CONSUMPTION_PATTERN.search(str(exc))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def recoverable_edge_label_consumption_error(exc: ValueError) -> bool:
    """Detect Graph2Mat's symmetric same-species edge accounting mismatch."""

    if EDGE_LABEL_CONSUMPTION_ERROR not in str(exc):
        return False
    counts = edge_label_consumption_counts(exc)
    if counts is None:
        return False
    consumed, total = counts
    return consumed > total


def warn_recovered_edge_label_consumption(exc: ValueError, *, batch_idx: int | None = None) -> None:
    suffix = f" batch={batch_idx}" if batch_idx is not None else ""
    print(
        "[WARN] Graph2Mat MatrixWriter reported an edge-label consumption "
        f"mismatch{suffix}: {exc}. Continuing and validating all prediction "
        "outputs after export.",
        file=sys.stderr,
    )


def remove_tree_with_retries(path: Path, *, attempts: int = 5) -> bool:
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            time.sleep(0.05 * (attempt + 1))
    return False


def reset_output_directory(path: Path) -> None:
    if path.exists():
        stale_root = path.parent / ".stale"
        stale_root.mkdir(parents=True, exist_ok=True)
        stale_path = stale_root / f"{path.name}.{os.getpid()}.{time.time_ns()}"
        try:
            path.rename(stale_path)
        except FileNotFoundError:
            pass
        except OSError:
            if not remove_tree_with_retries(path):
                raise
        else:
            remove_tree_with_retries(stale_path)
    path.mkdir(parents=True, exist_ok=False)


def strict_matrix_writer_class(matrix_writer_cls: type) -> type:
    class StrictMatrixWriter(matrix_writer_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._warned_edge_label_consumption = False

        def _on_batch_end(self, split, trainer, pl_module, prediction, batch, batch_idx, dataloader_idx):
            prediction = cast_low_precision_tensors_to_float32(prediction)
            try:
                return super()._on_batch_end(split, trainer, pl_module, prediction, batch, batch_idx, dataloader_idx)
            except ValueError as exc:
                if EDGE_LABEL_CONSUMPTION_ERROR not in str(exc):
                    raise
                if recoverable_edge_label_consumption_error(exc):
                    if not self._warned_edge_label_consumption:
                        warn_recovered_edge_label_consumption(exc, batch_idx=batch_idx)
                        self._warned_edge_label_consumption = True
                    return None
                raise incomplete_prediction_error(exc) from exc

    return StrictMatrixWriter


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["sample_id", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_sample_inputs(rows: list[dict[str, str]], workspace: Path) -> list[dict[str, Any]]:
    copied = []
    structs_dir = workspace / "predict_structures"
    reset_output_directory(structs_dir)
    for row in rows:
        sample_id = row["sample_id"]
        sample_dir = structs_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        src_structure = Path(row["structure_path"])
        if not src_structure.exists():
            raise RuntimeError(f"Missing structure for {sample_id}: {src_structure}")
        dst_structure = sample_dir / "RUN.fdf"
        shutil.copy2(src_structure, dst_structure)
        # Copy only structural/basis side inputs. Reference Hamiltonians are
        # deliberately not copied into the prediction input tree; evaluation
        # reads them through the manifest/reference paths instead.
        source_parent = src_structure.parent
        for pattern in ("*.psf", "*.ion", "*.ion.xml", "*.XV"):
            for src in source_parent.glob(pattern):
                if src.name == "ML_prediction.HSX":
                    continue
                dst = sample_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
        copied.append(
            {
                **row,
                "prediction_structure_path": str(dst_structure),
                "reference_hamiltonian_copied_to_input": False,
            }
        )
    return copied


def validate_manifest_rows(rows: list[dict[str, str]], manifest_path: Path) -> None:
    if not rows:
        raise RuntimeError(f"Test manifest has no samples: {manifest_path}")
    seen: set[str] = set()
    duplicates: list[str] = []
    missing_fields: list[str] = []
    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id") or "").strip()
        structure_path = str(row.get("structure_path") or "").strip()
        if not sample_id or not structure_path:
            missing_fields.append(str(index))
            continue
        if sample_id in seen:
            duplicates.append(sample_id)
        seen.add(sample_id)
    if missing_fields:
        raise RuntimeError(
            f"Test manifest rows missing sample_id or structure_path at {manifest_path}: "
            + ", ".join(missing_fields[:10])
        )
    if duplicates:
        unique_duplicates = sorted(set(duplicates))
        raise RuntimeError(
            f"Duplicate sample_id values in test manifest {manifest_path}: "
            + ", ".join(unique_duplicates[:10])
        )


def sample_list(values: list[str]) -> str:
    head = ", ".join(values[:10])
    if len(values) > 10:
        return f"{head}, ... (+{len(values) - 10} more)"
    return head


def validate_prediction_outputs(
    prediction_rows: list[dict[str, Any]],
    workspace: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not prediction_rows:
        raise RuntimeError("No prediction rows were produced.")

    expected_ids = [str(row.get("sample_id") or "") for row in prediction_rows]
    expected_set = set(expected_ids)
    if len(expected_set) != len(expected_ids):
        duplicates = sorted({sample_id for sample_id in expected_ids if expected_ids.count(sample_id) > 1})
        raise RuntimeError("Prediction validation failed: duplicate sample ids: " + sample_list(duplicates))

    workspace_root = workspace / "predict_structures"
    archive_root = output_dir / "predicted_hamiltonians"
    unexpected_workspace = sorted(
        path.parent.name
        for path in workspace_root.glob(f"*/{PREDICTION_OUTPUT_FILE}")
        if path.parent.name not in expected_set
    )
    unexpected_archive = sorted(
        path.parent.name
        for path in archive_root.glob(f"*/{PREDICTION_OUTPUT_FILE}")
        if path.parent.name not in expected_set
    )

    missing: list[str] = []
    path_mismatches: list[str] = []
    empty_outputs: list[str] = []
    for row in prediction_rows:
        sample_id = str(row["sample_id"])
        expected_workspace = workspace_root / sample_id / PREDICTION_OUTPUT_FILE
        expected_archive = archive_root / sample_id / PREDICTION_OUTPUT_FILE
        row_prediction = Path(str(row.get("prediction_path") or ""))
        if str(row.get("status") or "") != "predicted":
            missing.append(sample_id)
            continue
        if not expected_workspace.exists() or not expected_archive.exists():
            missing.append(sample_id)
            continue
        if row_prediction.resolve() != expected_archive.resolve():
            path_mismatches.append(sample_id)
            continue
        if expected_workspace.stat().st_size <= 0 or expected_archive.stat().st_size <= 0:
            empty_outputs.append(sample_id)

    problems: list[str] = []
    if missing:
        problems.append("missing prediction for samples: " + sample_list(missing))
    if path_mismatches:
        problems.append("prediction path mismatch for samples: " + sample_list(path_mismatches))
    if empty_outputs:
        problems.append("empty prediction output for samples: " + sample_list(empty_outputs))
    if unexpected_workspace:
        problems.append("unexpected workspace predictions: " + sample_list(unexpected_workspace))
    if unexpected_archive:
        problems.append("unexpected archived predictions: " + sample_list(unexpected_archive))
    if problems:
        raise RuntimeError("Prediction validation failed: " + "; ".join(problems))

    return {
        "validated": True,
        "status": "validated",
        "expected_samples": len(prediction_rows),
        "validated_samples": len(prediction_rows),
        "checks": [
            "sample_count",
            "sample_id_mapping",
            "workspace_prediction_path",
            "archived_prediction_path",
            "nonempty_output_file",
            "no_unexpected_prediction_files",
        ],
        "matrix_shape_validation": "deferred_to_hamiltonian_metrics",
    }


def run_prediction(args: argparse.Namespace, predict_glob: str, *, basis_files_absolute: str) -> None:
    import inspect
    import pytorch_lightning as pl
    import torch
    from graph2mat.core.data.processing import MatrixDataProcessor
    from graph2mat.tools.lightning import MatrixDataModule, MatrixWriter
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    from graph2mat import AtomicTableWithEdges

    matrix_component_policy, n_matrix_components = resolve_cli_matrix_component_policy(args)
    if args.patch_graph2mat_basis_loading:
        original = MatrixDataProcessor.get_config_kwargs

        def patched(self: MatrixDataProcessor, obj: object) -> dict[str, object]:
            kwargs = original(self, obj)
            kwargs.pop("basis", None)
            return kwargs

        MatrixDataProcessor.get_config_kwargs = patched

    allow_graph2mat_checkpoint_globals()
    # The checkpoint's saved basis_files hparam is a path relative to the cwd
    # used at training time, which does not always match the directory layout
    # reconstructed by checkpoint_training_dir() (e.g. after a run root was
    # moved or restructured). Build the basis table directly from the
    # absolute --basis-files CLI arg and pass it as a load_from_checkpoint
    # override instead of relying on that glob being resolvable from cwd.
    basis_table = AtomicTableWithEdges.from_basis_glob(
        Path(basis_files_absolute).parent.glob(Path(basis_files_absolute).name)
    )
    model = LitMACEMatrixModel.load_from_checkpoint(
        str(args.checkpoint), weights_only=False, basis_table=basis_table
    )
    if args.mace_node_chunk_size is not None:
        apply_mace_node_chunking(model, args.mace_node_chunk_size, torch)
    if args.mace_edge_chunk_size is not None:
        apply_mace_edge_chunking(model, args.mace_edge_chunk_size, torch)
    if args.graph2mat_edge_chunk_size is not None:
        apply_graph2mat_edge_chunking(model, args.graph2mat_edge_chunk_size, torch)
    if args.graph2mat_node_chunk_size is not None:
        apply_graph2mat_node_chunking(model, args.graph2mat_node_chunk_size, torch)
    datamodule_kwargs: dict[str, Any] = {
        "out_matrix": args.out_matrix,
        "symmetric_matrix": bool(args.symmetric_matrix),
        "sub_point_matrix": bool(args.sub_point_matrix),
        "basis_files": args.basis_files,
        "predict_structs": predict_glob,
        "store_in_memory": bool(args.store_in_memory),
    }
    if "n_matrix_components" in inspect.signature(MatrixDataModule).parameters:
        datamodule_kwargs["n_matrix_components"] = n_matrix_components
        if "matrix_component_policy" in inspect.signature(MatrixDataModule).parameters:
            datamodule_kwargs["matrix_component_policy"] = matrix_component_policy
        else:
            raise RuntimeError(
                "MatrixDataModule does not expose matrix_component_policy; this official "
                "benchmark cannot enforce H-only target semantics with this Graph2Mat version."
            )
        validate_model_matrix_component_policy(
            model,
            matrix_component_policy=matrix_component_policy,
            n_matrix_components=n_matrix_components,
            context="cross prediction",
        )
    else:
        raise RuntimeError(
            "MatrixDataModule does not expose n_matrix_components; this official "
            "benchmark cannot enforce H-only target semantics with this Graph2Mat version."
        )
    if "batch_size" in inspect.signature(MatrixDataModule).parameters:
        datamodule_kwargs["batch_size"] = 1
    if "loader_threads" in inspect.signature(MatrixDataModule).parameters and args.loader_threads is not None:
        datamodule_kwargs["loader_threads"] = int(args.loader_threads)
    datamodule = MatrixDataModule(**datamodule_kwargs)
    StrictMatrixWriter = strict_matrix_writer_class(MatrixWriter)
    callbacks = [StrictMatrixWriter(output_file=PREDICTION_OUTPUT_FILE, splits=["predict"])]
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        precision=args.precision,
        logger=False,
        callbacks=callbacks,
    )
    try:
        trainer.predict(model, datamodule=datamodule, ckpt_path=None)
    except ValueError as exc:
        if EDGE_LABEL_CONSUMPTION_ERROR not in str(exc):
            raise
        if recoverable_edge_label_consumption_error(exc):
            warn_recovered_edge_label_consumption(exc)
        else:
            raise incomplete_prediction_error(exc) from exc


def checkpoint_training_dir(checkpoint: Path) -> Path:
    for parent in checkpoint.parents:
        if parent.name == "lightning_logs":
            return parent.parent
    return checkpoint.parent


def relative_pattern(pattern: str, base: Path) -> str:
    if os.path.isabs(pattern):
        return os.path.relpath(pattern, base).replace("\\", "/")
    return pattern


def normalize_pattern_for_workdir(pattern: str, *, source_cwd: Path, target_cwd: Path) -> str:
    """Re-anchor a CLI path/glob so it stays valid after changing directories."""

    if os.path.isabs(pattern):
        absolute_pattern = pattern
    else:
        absolute_pattern = os.path.abspath(source_cwd / pattern)
    return os.path.relpath(absolute_pattern, target_cwd).replace("\\", "/")


def collect_predictions(rows: list[dict[str, Any]], workspace: Path, output_dir: Path) -> list[dict[str, Any]]:
    prediction_rows = []
    prediction_root = output_dir / "predicted_hamiltonians"
    reset_output_directory(prediction_root)
    for row in rows:
        sample_id = row["sample_id"]
        source_sample_dir = workspace / "predict_structures" / sample_id
        source_prediction = source_sample_dir / PREDICTION_OUTPUT_FILE
        destination_dir = prediction_root / sample_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_prediction = destination_dir / PREDICTION_OUTPUT_FILE
        status = "predicted" if source_prediction.exists() else "missing_prediction"
        if source_prediction.exists():
            shutil.copy2(source_prediction, destination_prediction)
        prediction_rows.append(
            {
                **row,
                "prediction_path": str(destination_prediction) if destination_prediction.exists() else "",
                "status": status,
            }
        )
    return prediction_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--train-method",
        required=True,
        choices=["md", "atom_displacement", "siesta_fc_cartesian", "random_cartesian"],
    )
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basis-files", required=True)
    parser.add_argument("--out-matrix", default="hamiltonian")
    parser.add_argument("--symmetric-matrix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sub-point-matrix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--store-in-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument(
        "--precision",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        default="32-true",
    )
    parser.add_argument("--mace-node-chunk-size", type=int)
    parser.add_argument("--mace-edge-chunk-size", type=int)
    parser.add_argument("--graph2mat-edge-chunk-size", type=int)
    parser.add_argument("--graph2mat-node-chunk-size", type=int)
    parser.add_argument("--n-matrix-components", type=int, default=None)
    parser.add_argument(
        "--matrix-component-policy",
        choices=["h_only", "spin_h_only", "raw_components", "h_and_overlap"],
        default=None,
    )
    parser.add_argument("--loader-threads", type=int, default=None)
    parser.add_argument("--torch-float32-matmul-precision", choices=["high", "medium"], default=None)
    parser.add_argument("--patch-graph2mat-basis-loading", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_cli_matrix_component_policy(args: argparse.Namespace) -> tuple[str, int]:
    return resolve_matrix_component_policy(
        {
            "matrix_component_policy": args.matrix_component_policy,
            "n_matrix_components": args.n_matrix_components,
        },
        context="prediction CLI",
    )


def main() -> int:
    args = build_parser().parse_args()
    resolve_cli_matrix_component_policy(args)
    if args.loader_threads is not None and args.loader_threads <= 0:
        raise RuntimeError("--loader-threads must be a positive integer.")
    apply_torch_float32_matmul_precision(args.torch_float32_matmul_precision)
    if not args.checkpoint.exists():
        raise RuntimeError(f"Checkpoint does not exist: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workspace = args.output_dir / "workspace"
    rows = read_rows(args.test_manifest)
    validate_manifest_rows(rows, args.test_manifest)
    start = time.time()
    copied_rows = copy_sample_inputs(rows, workspace)
    run_cwd = checkpoint_training_dir(args.checkpoint)
    source_cwd = Path.cwd()
    absolute_basis_files = os.path.abspath(source_cwd / args.basis_files)
    args.basis_files = normalize_pattern_for_workdir(
        str(args.basis_files),
        source_cwd=source_cwd,
        target_cwd=run_cwd,
    )
    predict_glob = os.path.relpath(
        workspace / "predict_structures" / "*" / "RUN.fdf",
        run_cwd,
    ).replace("\\", "/")
    prediction_status = "dry_run"
    error = None
    if not args.dry_run:
        cwd = Path.cwd()
        try:
            os.chdir(run_cwd)
            run_prediction(args, predict_glob, basis_files_absolute=absolute_basis_files)
            prediction_status = "completed"
        except (RuntimeError, ValueError, OSError) as exc:
            prediction_status = "failed"
            error = str(exc)
        finally:
            os.chdir(cwd)
    prediction_rows = collect_predictions(copied_rows, workspace, args.output_dir)
    validation_summary: dict[str, Any] = {
        "validated": False,
        "status": f"skipped_prediction_status_{prediction_status}",
        "expected_samples": len(rows),
        "validated_samples": 0,
        "checks": [],
        "matrix_shape_validation": "not_performed",
    }
    if prediction_status == "completed":
        try:
            validation_summary = validate_prediction_outputs(prediction_rows, workspace, args.output_dir)
        except RuntimeError as exc:
            prediction_status = "failed_validation"
            error = str(exc)
            validation_summary = {
                "validated": False,
                "status": "failed",
                "error": str(exc),
                "expected_samples": len(rows),
                "validated_samples": sum(1 for row in prediction_rows if row["status"] == "predicted"),
                "checks": [
                    "sample_count",
                    "sample_id_mapping",
                    "workspace_prediction_path",
                    "archived_prediction_path",
                    "nonempty_output_file",
                    "no_unexpected_prediction_files",
                ],
                "matrix_shape_validation": "not_performed",
            }
    for row in prediction_rows:
        row["train_method"] = args.train_method
        row["test_set"] = args.test_set
        row["model_checkpoint"] = str(args.checkpoint)
        row["prediction_command_status"] = prediction_status
        row["prediction_output_validated"] = bool(validation_summary.get("validated"))
        row["prediction_validation_status"] = str(validation_summary.get("status") or "")
    write_rows(args.output_dir / "prediction_manifest.csv", prediction_rows)
    summary = {
        "ok": prediction_status == "completed" and bool(validation_summary.get("validated")),
        "status": prediction_status,
        "error": error,
        "train_method": args.train_method,
        "test_set": args.test_set,
        "checkpoint": str(args.checkpoint),
        "samples": len(rows),
        "predicted_samples": sum(1 for row in prediction_rows if row["status"] == "predicted"),
        "prediction_time_seconds": time.time() - start,
        "outputs": {
            "prediction_manifest": str(args.output_dir / "prediction_manifest.csv"),
            "predicted_hamiltonians": str(args.output_dir / "predicted_hamiltonians"),
        },
        "input_reference_files_copied": False,
        "prediction_output_validation": validation_summary,
    }
    (args.output_dir / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
