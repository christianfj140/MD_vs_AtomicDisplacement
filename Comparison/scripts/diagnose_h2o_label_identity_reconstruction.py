#!/usr/bin/env python3
"""Diagnose H2O label identity reconstruction through Graph2Mat paths.

The diagnostic uses the one-sample H-only overfit workspace and feeds the
reference batch labels back as predictions. If identity labels cannot reconstruct
the reference Hamiltonian, the issue is in the data/reconstruction path rather
than model capacity.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TORCH_COMPAT_DIR = REPO_ROOT / "scripts" / "torch_serialization_compat"
sys.path.insert(0, str(TORCH_COMPAT_DIR))
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))

from torch_safe_globals import allow_graph2mat_checkpoint_globals  # noqa: E402

allow_graph2mat_checkpoint_globals()

from evaluate_hamiltonian_metrics import read_matrix  # noqa: E402
from graph2mat.tools.lightning import MatrixDataModule  # noqa: E402


DEFAULT_WORKSPACE = (
    REPO_ROOT
    / "Comparison"
    / "results"
    / "diagnostics"
    / "h2o_hamiltonian"
    / "one_sample_overfit_h_only"
)
EDGE_CONSUMPTION_RE = re.compile(r"consumed=(\d+), total=(\d+)")


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
        "mae": float(abs_delta.mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(abs_delta.max()),
        "relative_frobenius": rel_fro,
        "allclose_1e_8": bool(np.allclose(reference, prediction, atol=1e-8, rtol=1e-8)),
        "allclose_1e_5": bool(np.allclose(reference, prediction, atol=1e-5, rtol=1e-8)),
        "allclose_1e_10": bool(np.allclose(reference, prediction, atol=1e-10, rtol=1e-10)),
    }
    result.update(support_metrics(reference, prediction, threshold))
    return result


def failure_payload(exc: BaseException) -> dict[str, Any]:
    message = str(exc)
    match = EDGE_CONSUMPTION_RE.search(message)
    payload = {
        "status": "error",
        "error_type": type(exc).__name__,
        "error": message,
        "traceback": traceback.format_exc(),
        "edge_label_consumption_error": "Predicted edge labels were not fully consumed" in message,
    }
    if match:
        payload["edge_label_consumed"] = int(match.group(1))
        payload["edge_label_total"] = int(match.group(2))
    return payload


def run_path(
    name: str,
    fn: Callable[[], Any],
    *,
    reference_h: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    try:
        output = fn()
        if isinstance(output, tuple):
            output = list(output)
        if isinstance(output, list):
            if len(output) != 1:
                raise RuntimeError(f"Expected one reconstructed matrix, got {len(output)}")
            output = output[0]
        if not hasattr(output, "tocsr"):
            raise TypeError(f"Path returned {type(output).__name__}, not a matrix")
        prediction_h = dense_h(output)
        result = {
            "status": "ok",
            "matrix_type": type(output).__name__,
            "spin": str(getattr(output, "spin", "")),
            "orthogonal": bool(getattr(output, "orthogonal", True)),
            "metrics": compare_h(reference_h, prediction_h, threshold),
        }
    except Exception as exc:
        result = failure_payload(exc)
    result["path"] = name
    return result


def sanitize_debug_report(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): sanitize_debug_report(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_debug_report(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def overall_verdict(paths: dict[str, dict[str, Any]]) -> dict[str, str]:
    def identity_ok(item: dict[str, Any]) -> bool:
        if item["status"] != "ok":
            return False
        metrics = item["metrics"]
        return (
            metrics["comparable"]
            and metrics["mae"] <= 1e-6
            and metrics["rmse"] <= 1e-6
            and metrics["max_abs"] <= 1e-5
            and metrics["support_f1"] == 1.0
        )

    direct_paths = [
        paths["direct_example_existing_labels"],
        paths["direct_example_identity_predictions"],
        paths["batched_yield_existing_labels_as_matrix"],
    ]
    direct_ok = all(identity_ok(item) for item in direct_paths)
    prediction_paths = [
        paths["batched_yield_identity_predictions_as_matrix"],
        paths["batch_matrix_from_data_identity_predictions"],
        paths["matrixwriter_like_identity_predictions"],
    ]
    prediction_errors = [item for item in prediction_paths if item["status"] != "ok"]
    all_paths_ok = all(identity_ok(item) for item in paths.values())
    if direct_ok and prediction_errors:
        return {
            "status": "FAIL_BATCHED_PREDICTION_RECONSTRUCTION",
            "summary": (
                "Reference labels reconstruct the reference Hamiltonian through direct/example paths, "
                "but Graph2Mat's batched predictions path fails before MatrixWriter output."
            ),
        }
    if all_paths_ok:
        return {
            "status": "PASS",
            "summary": (
                "Identity labels reconstruct the reference Hamiltonian on all checked paths "
                "to float32-level numerical noise; no checked path triggered consumed/total."
            ),
        }
    return {
        "status": "FAIL_DIRECT_RECONSTRUCTION",
        "summary": (
            "At least one direct reconstruction path failed or was not numerically identical. "
            "This indicates a lower-level data/reconstruction issue."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# H2O Label Identity Reconstruction Diagnostic",
        "",
        f"Workspace: `{payload['workspace']}`",
        f"Training dir: `{payload['training_dir']}`",
        f"Config: `{payload['config_path']}`",
        f"Reference: `{payload['reference_path']}`",
        "",
        "## Verdict",
        "",
        f"**{payload['verdict']['status']}**: {payload['verdict']['summary']}",
        "",
        "## H-only Policy",
        "",
        f"- `out_matrix`: `{payload['data_policy'].get('out_matrix')}`",
        f"- `matrix_component_policy`: `{payload['data_policy'].get('matrix_component_policy')}`",
        f"- `n_matrix_components`: `{payload['data_policy'].get('n_matrix_components')}`",
        f"- `symmetric_matrix`: `{payload['data_policy'].get('symmetric_matrix')}`",
        f"- `sub_point_matrix`: `{payload['data_policy'].get('sub_point_matrix')}`",
        "",
        "## Edge Accounting Debug",
        "",
        "```json",
        json.dumps(payload["edge_accounting_debug"], indent=2, sort_keys=True),
        "```",
        "",
        "## Reconstruction Paths",
        "",
        "| path | status | MAE | RMSE | max abs | rel Frobenius | support F1 | edge consumed | edge total |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in payload["paths"].items():
        metrics = result.get("metrics") or {}
        lines.append(
            "| {name} | {status} | {mae} | {rmse} | {max_abs} | {rel_fro} | {f1} | {consumed} | {total} |".format(
                name=name,
                status=result["status"],
                mae=metrics.get("mae", ""),
                rmse=metrics.get("rmse", ""),
                max_abs=metrics.get("max_abs", ""),
                rel_fro=metrics.get("relative_frobenius", ""),
                f1=metrics.get("support_f1", ""),
                consumed=result.get("edge_label_consumed", ""),
                total=result.get("edge_label_total", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Error Details",
            "",
        ]
    )
    for name, result in payload["paths"].items():
        if result["status"] == "ok":
            continue
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Error type: `{result.get('error_type')}`",
                f"- Edge-label consumption error: `{result.get('edge_label_consumption_error')}`",
                "",
                "```text",
                result.get("error", ""),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Command",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
            f"Result: `{payload['script_result']}`",
            "",
            "## Interpretation",
            "",
            "- Direct/example reconstruction using the labels already stored on the example tests Graph2Mat's single-example data path.",
            "- Batched identity reconstruction tests the same `yield_from_batch(..., predictions=...)` slicing path used by prediction writers.",
            "- MatrixWriter-like identity reconstruction performs `yield_from_batch(..., as_matrix=False)` followed by `convert_to(...)`, matching the callback's core logic without writing files.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run H2O identity-label reconstruction diagnostics."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--training-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=1e-12)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    training_dir = (args.training_dir or workspace / "training").resolve()
    config_path = (args.config or training_dir / "config.yaml").resolve()
    reference_path = (args.reference or workspace / "dataset" / "samples" / "md_94" / "siesta.TSHS").resolve()
    output_dir = (args.output_dir or workspace / "label_identity_reconstruction").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    datamodule_kwargs = build_datamodule_kwargs(data)
    reference_h = np.asarray(read_matrix(reference_path).hamiltonian.toarray(), dtype=float)

    with working_directory(training_dir):
        datamodule = MatrixDataModule(**datamodule_kwargs)
        datamodule.setup("fit")
        processor = datamodule.data_processor
        batch = next(iter(datamodule.train_dataloader()))
        example = batch.get_example(0)
        predictions = {
            "node_labels": batch.point_labels,
            "edge_labels": batch.edge_labels,
        }
        example_predictions = {
            "node_labels": example.point_labels,
            "edge_labels": example.edge_labels,
        }

        def matrixwriter_like_no_predictions():
            outputs = []
            for matrix_data in processor.yield_from_batch(batch, as_matrix=False):
                outputs.append(
                    matrix_data.convert_to(
                        processor.default_out_format,
                        matrix_component_policy=getattr(processor, "matrix_component_policy", None),
                    )
                )
            return outputs

        def matrixwriter_like_identity_predictions():
            outputs = []
            for matrix_data in processor.yield_from_batch(
                batch, predictions=predictions, as_matrix=False
            ):
                outputs.append(
                    matrix_data.convert_to(
                        processor.default_out_format,
                        matrix_component_policy=getattr(processor, "matrix_component_policy", None),
                    )
                )
            return outputs

        paths = {
            "direct_example_existing_labels": run_path(
                "direct_example_existing_labels",
                lambda: processor.matrix_from_data(example),
                reference_h=reference_h,
                threshold=args.threshold,
            ),
            "direct_example_identity_predictions": run_path(
                "direct_example_identity_predictions",
                lambda: processor.matrix_from_data(example, predictions=example_predictions),
                reference_h=reference_h,
                threshold=args.threshold,
            ),
            "batched_yield_existing_labels_as_matrix": run_path(
                "batched_yield_existing_labels_as_matrix",
                lambda: list(processor.yield_from_batch(batch, as_matrix=True)),
                reference_h=reference_h,
                threshold=args.threshold,
            ),
            "batched_yield_identity_predictions_as_matrix": run_path(
                "batched_yield_identity_predictions_as_matrix",
                lambda: list(
                    processor.yield_from_batch(batch, predictions=predictions, as_matrix=True)
                ),
                reference_h=reference_h,
                threshold=args.threshold,
            ),
            "batch_matrix_from_data_identity_predictions": run_path(
                "batch_matrix_from_data_identity_predictions",
                lambda: processor.matrix_from_data(batch, predictions=predictions),
                reference_h=reference_h,
                threshold=args.threshold,
            ),
            "matrixwriter_like_existing_labels": run_path(
                "matrixwriter_like_existing_labels",
                matrixwriter_like_no_predictions,
                reference_h=reference_h,
                threshold=args.threshold,
            ),
            "matrixwriter_like_identity_predictions": run_path(
                "matrixwriter_like_identity_predictions",
                matrixwriter_like_identity_predictions,
                reference_h=reference_h,
                threshold=args.threshold,
            ),
        }
        try:
            debug = processor._debug_edge_label_accounting(batch, predictions=predictions)
        except Exception as exc:
            debug = failure_payload(exc)

    payload = {
        "workspace": str(workspace),
        "training_dir": str(training_dir),
        "config_path": str(config_path),
        "reference_path": str(reference_path),
        "output_dir": str(output_dir),
        "command": " ".join([sys.executable, *sys.argv]),
        "script_result": "completed_successfully",
        "threshold": args.threshold,
        "data_policy": {
            "out_matrix": data.get("out_matrix"),
            "symmetric_matrix": data.get("symmetric_matrix"),
            "sub_point_matrix": data.get("sub_point_matrix"),
            "matrix_component_policy": data.get("matrix_component_policy"),
            "n_matrix_components": data.get("n_matrix_components"),
        },
        "reference_shape": list(reference_h.shape),
        "edge_accounting_debug": sanitize_debug_report(debug),
        "paths": paths,
    }
    payload["verdict"] = overall_verdict(paths)

    json_path = output_dir / "label_identity_reconstruction.json"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "verdict": payload["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
