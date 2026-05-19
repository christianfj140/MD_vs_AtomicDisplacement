#!/usr/bin/env python3
"""Non-destructive diagnostics for the H2O Hamiltonian prediction path."""

from __future__ import annotations

import argparse
import glob
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
import sisl

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_hamiltonian_metrics import (  # noqa: E402
    generalized_eigenvalues,
    eigen_error_metrics,
    hermiticity_defect,
    read_matrix,
    sparse_metrics,
)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": int(values.size), "finite": 0}
    return {
        "n": int(values.size),
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "mean_abs": float(np.mean(np.abs(finite))),
    }


def csr_difference_stats(reference: sparse.spmatrix, other: sparse.spmatrix) -> dict[str, Any]:
    diff = (other - reference).tocsr()
    n_entries = reference.shape[0] * reference.shape[1]
    ref_norm = float(sparse.linalg.norm(reference))
    diff_norm = float(sparse.linalg.norm(diff))
    ref_support = set(zip(*reference.nonzero(), strict=False))
    other_support = set(zip(*other.nonzero(), strict=False))
    precision = len(ref_support & other_support) / len(other_support) if other_support else math.nan
    recall = len(ref_support & other_support) / len(ref_support) if ref_support else math.nan
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and precision + recall > 0
        else math.nan
    )
    return {
        "shape": list(reference.shape),
        "reference_nnz": int(reference.nnz),
        "other_nnz": int(other.nnz),
        "pattern_equal": ref_support == other_support,
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
        "diff_nnz": int(diff.nnz),
        "max_abs_error": float(np.max(np.abs(diff.data))) if diff.nnz else 0.0,
        "mae_all_entries": float(np.sum(np.abs(diff.data)) / n_entries) if n_entries else math.nan,
        "rmse_all_entries": float(np.sqrt(diff.multiply(diff).sum() / n_entries)) if n_entries else math.nan,
        "frobenius_error": diff_norm,
        "relative_frobenius_error": diff_norm / ref_norm if ref_norm else math.nan,
    }


def matrix_summary(path: Path) -> dict[str, Any]:
    data = read_matrix(path)
    return {
        "path": str(path),
        "shape": list(data.hamiltonian.shape),
        "hamiltonian_nnz": int(data.hamiltonian.nnz),
        "component_count_for_metrics": int(data.component_count),
        "spin_kind": data.spin_kind,
        "orthogonal": bool(data.orthogonal),
        "has_overlap": bool(data.has_overlap),
        "overlap_shape": list(data.overlap.shape) if data.overlap is not None else None,
        "overlap_nnz": int(data.overlap.nnz) if data.overlap is not None else None,
        "overlap_error": data.overlap_error,
        "fermi_level": data.fermi_level,
        "fermi_level_source": data.fermi_level_source,
        "hermiticity_relative_frobenius": hermiticity_defect(data.hamiltonian),
        "hamiltonian_values": finite_stats(data.hamiltonian.data),
    }


def raw_sisl_component_summary(path: Path) -> dict[str, Any]:
    hamiltonian = sisl.get_sile(str(path)).read_hamiltonian()
    raw_dim = getattr(hamiltonian, "dim", None)
    raw_dim = int(raw_dim() if callable(raw_dim) else raw_dim or 1)
    components = []
    for index in range(raw_dim):
        csr = hamiltonian.tocsr(index)
        components.append(
            {
                "index": index,
                "shape": list(csr.shape),
                "nnz": int(csr.nnz),
                "norm": float(sparse.linalg.norm(csr)),
                "values": finite_stats(csr.data),
            }
        )
    return {
        "path": str(path),
        "raw_dim": raw_dim,
        "spin": str(getattr(hamiltonian, "spin", "")) or None,
        "orthogonal": bool(getattr(hamiltonian, "orthogonal", False)),
        "shape": list(getattr(hamiltonian, "shape", ())),
        "components": components,
    }


def label_array_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "shape": list(values.shape),
        "component_count": int(values.shape[1]) if values.ndim == 2 else 1,
        "components": [finite_stats(values[:, index]) for index in range(values.shape[1])],
    }


def load_graph2mat_labels(
    *,
    run_fdf: Path,
    matrix_path: Path,
    matrix_name: str,
    basis_files: list[Path],
    output_dir: Path,
    n_matrix_components: int,
) -> tuple[Any, Any]:
    from graph2mat import AtomicTableWithEdges, MatrixDataProcessor
    from graph2mat.bindings.torch.data import TorchBasisMatrixData

    basis_table = AtomicTableWithEdges.from_basis_glob(basis_files)
    processor = MatrixDataProcessor(
        basis_table=basis_table,
        out_matrix="hamiltonian",
        symmetric_matrix=True,
        sub_point_matrix=False,
        n_matrix_components=n_matrix_components,
    )
    with tempfile.TemporaryDirectory(prefix="h2o_graph2mat_", dir=output_dir) as tmp:
        sample_dir = Path(tmp)
        shutil.copy2(run_fdf, sample_dir / "RUN.fdf")
        shutil.copy2(matrix_path, sample_dir / matrix_name)
        sample = TorchBasisMatrixData.new(
            sample_dir / "RUN.fdf",
            data_processor=processor,
            labels=True,
        )
        return sample, processor


def graph2mat_roundtrip(
    *,
    run_fdf: Path,
    reference_path: Path,
    predicted_path: Path | None,
    basis_files: list[Path],
    output_dir: Path,
    n_matrix_components: int,
) -> dict[str, Any]:
    if not basis_files:
        return {"status": "NOT_RUN", "reason": "no basis .ion.xml files provided"}

    try:
        reference_sample, processor = load_graph2mat_labels(
            run_fdf=run_fdf,
            matrix_path=reference_path,
            matrix_name="siesta.TSHS",
            basis_files=basis_files,
            output_dir=output_dir,
            n_matrix_components=n_matrix_components,
        )
        reconstructed = processor.matrix_from_data(reference_sample, threshold=None)
        reconstructed_h = reconstructed.tocsr(0) if hasattr(reconstructed, "tocsr") else reconstructed.tocsr()
        reference_h = read_matrix(reference_path).hamiltonian
        result: dict[str, Any] = {
            "status": "PASS",
            "basis_files": [str(path) for path in basis_files],
            "reference_point_labels": label_array_stats(reference_sample.point_labels.numpy()),
            "reference_edge_labels": label_array_stats(reference_sample.edge_labels.numpy()),
            "roundtrip_hamiltonian": csr_difference_stats(reference_h, reconstructed_h),
        }
        if predicted_path is not None and predicted_path.exists():
            predicted_sample, _processor = load_graph2mat_labels(
                run_fdf=run_fdf,
                matrix_path=predicted_path,
                matrix_name="siesta.HSX",
                basis_files=basis_files,
                output_dir=output_dir,
                n_matrix_components=n_matrix_components,
            )
            pred_point = predicted_sample.point_labels.numpy()
            pred_edge = predicted_sample.edge_labels.numpy()
            ref_point = reference_sample.point_labels.numpy()
            ref_edge = reference_sample.edge_labels.numpy()
            common_point_components = min(ref_point.shape[1], pred_point.shape[1])
            common_edge_components = min(ref_edge.shape[1], pred_edge.shape[1])
            component_errors = []
            for index in range(min(common_point_components, common_edge_components)):
                point_error = pred_point[:, index] - ref_point[:, index]
                edge_error = pred_edge[:, index] - ref_edge[:, index]
                component_errors.append(
                    {
                        "component": index,
                        "point_mae": float(np.mean(np.abs(point_error))),
                        "point_rmse": float(np.sqrt(np.mean(point_error**2))),
                        "point_max_abs": float(np.max(np.abs(point_error))),
                        "edge_mae": float(np.mean(np.abs(edge_error))),
                        "edge_rmse": float(np.sqrt(np.mean(edge_error**2))),
                        "edge_max_abs": float(np.max(np.abs(edge_error))),
                        "block_type_mae_contribution": float(
                            np.mean(np.abs(point_error)) + np.mean(np.abs(edge_error))
                        ),
                    }
                )
            result["predicted_point_labels"] = label_array_stats(pred_point)
            result["predicted_edge_labels"] = label_array_stats(pred_edge)
            result["label_component_errors"] = component_errors
            if component_errors:
                h_only = component_errors[0]["block_type_mae_contribution"]
                all_common = float(
                    np.mean(np.abs(pred_point[:, :common_point_components] - ref_point[:, :common_point_components]))
                    + np.mean(np.abs(pred_edge[:, :common_edge_components] - ref_edge[:, :common_edge_components]))
                )
                result["loss_weighting_probe"] = {
                    "h_component_only_node_plus_edge_mae": h_only,
                    "all_common_components_node_plus_edge_mae": all_common,
                    "common_point_components": int(common_point_components),
                    "common_edge_components": int(common_edge_components),
                }
        return result
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def find_matrix(directory: Path, preferred: str | None = None) -> Path | None:
    if preferred:
        candidate = directory / preferred
        if candidate.exists():
            return candidate
    candidates = sorted(
        [path for path in directory.glob("*.TSHS")] + [path for path in directory.glob("*.HSX")]
    )
    candidates = [path for path in candidates if path.name != "ML_prediction.HSX"]
    return candidates[0] if candidates else None


def available_sample_ids(evaluation_root: Path) -> list[str]:
    structures = evaluation_root / "structures"
    if not structures.exists():
        return []
    return sorted(path.name for path in structures.iterdir() if path.is_dir())


def sample_paths(evaluation_root: Path, sample: str) -> dict[str, Path | None]:
    return {
        "run_fdf": evaluation_root / "structures" / sample / "RUN.fdf",
        "reference": find_matrix(evaluation_root / "siesta_hamiltonians" / sample),
        "prediction": evaluation_root / "predicted_hamiltonians" / sample / "ML_prediction.HSX",
    }


def spectral_probe(reference: Any, predicted: Any, wrong_reference: Any | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    ref_eig = generalized_eigenvalues(reference.hamiltonian, reference.overlap)
    pred_eig = generalized_eigenvalues(predicted.hamiltonian, reference.overlap)
    _rows, metrics = eigen_error_metrics(
        ref_eig,
        pred_eig,
        reference.fermi_level,
        reference.fermi_level_source or "unavailable",
    )
    result["predicted_h_with_reference_s"] = metrics
    if wrong_reference is not None and wrong_reference.overlap is not None:
        wrong_ref_eig = generalized_eigenvalues(reference.hamiltonian, wrong_reference.overlap)
        _rows, wrong_ref_metrics = eigen_error_metrics(
            ref_eig,
            wrong_ref_eig,
            reference.fermi_level,
            reference.fermi_level_source or "unavailable",
        )
        wrong_pred_eig = generalized_eigenvalues(predicted.hamiltonian, wrong_reference.overlap)
        _rows, wrong_pred_metrics = eigen_error_metrics(
            ref_eig,
            wrong_pred_eig,
            reference.fermi_level,
            reference.fermi_level_source or "unavailable",
        )
        result["reference_h_with_wrong_s"] = wrong_ref_metrics
        result["predicted_h_with_wrong_s"] = wrong_pred_metrics
    return result


def overlap_probe(reference_path: Path, predicted_path: Path, wrong_reference_path: Path | None) -> dict[str, Any]:
    reference_sile = sisl.get_sile(str(reference_path))
    predicted_sile = sisl.get_sile(str(predicted_path))
    ref_s = reference_sile.read_overlap().tocsr()
    pred_s = predicted_sile.read_overlap().tocsr()
    predicted_h = predicted_sile.read_hamiltonian()
    result: dict[str, Any] = {
        "reference_overlap": {
            "shape": list(ref_s.shape),
            "nnz": int(ref_s.nnz),
            "values": finite_stats(ref_s.data),
        },
        "predicted_read_overlap_vs_reference_overlap": csr_difference_stats(ref_s, pred_s),
    }
    for index in range(1, 4):
        try:
            component = predicted_h.tocsr(index)
        except Exception:
            continue
        result[f"predicted_component_{index}_vs_reference_overlap"] = csr_difference_stats(ref_s, component)
    if wrong_reference_path is not None and wrong_reference_path.exists():
        wrong_s = sisl.get_sile(str(wrong_reference_path)).read_overlap().tocsr()
        result["wrong_reference_overlap_vs_reference_overlap"] = csr_difference_stats(ref_s, wrong_s)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--second-sample", default=None)
    parser.add_argument("--basis-glob", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("Comparison/results/diagnostics/h2o_hamiltonian"))
    parser.add_argument("--n-matrix-components", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation_root = args.evaluation_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basis_pattern = str(args.basis_glob)
    basis_matches = glob.glob(basis_pattern)
    if not basis_matches and not Path(basis_pattern).is_absolute():
        basis_matches = glob.glob(str(REPO_ROOT / basis_pattern))
    basis_files = sorted(Path(path).resolve() for path in basis_matches if Path(path).is_file())

    paths = sample_paths(evaluation_root, args.sample)
    missing = [key for key, path in paths.items() if path is None or not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required sample inputs for {args.sample}: {', '.join(missing)}")

    sample_ids = available_sample_ids(evaluation_root)
    second_sample = args.second_sample
    if second_sample is None:
        second_sample = next((sample for sample in sample_ids if sample != args.sample), None)
    wrong_paths = sample_paths(evaluation_root, second_sample) if second_sample else {}

    reference = read_matrix(paths["reference"])
    predicted = read_matrix(paths["prediction"])
    wrong_reference = None
    wrong_reference_path = wrong_paths.get("reference") if wrong_paths else None
    if wrong_reference_path is not None and wrong_reference_path.exists():
        wrong_reference = read_matrix(wrong_reference_path)

    sparse = sparse_metrics(args.sample, reference, predicted)
    payload = {
        "evaluation_root": str(evaluation_root),
        "sample": args.sample,
        "second_sample_for_wrong_overlap": second_sample,
        "inputs": {
            "run_fdf": str(paths["run_fdf"]),
            "reference": str(paths["reference"]),
            "prediction": str(paths["prediction"]),
            "basis_files": [str(path) for path in basis_files],
        },
        "reference_matrix": matrix_summary(paths["reference"]),
        "prediction_matrix": matrix_summary(paths["prediction"]),
        "reference_raw_sisl_components": raw_sisl_component_summary(paths["reference"]),
        "prediction_raw_sisl_components": raw_sisl_component_summary(paths["prediction"]),
        "sparse_metrics_subset": {
            key: sparse.get(key)
            for key in (
                "mae_ref_eV",
                "rmse_ref_eV",
                "mae_union_eV",
                "rmse_union_eV",
                "mse_union_eV2",
                "r2_union",
                "support_f1",
                "relative_frobenius_union",
                "max_abs_error_union_eV",
                "hermiticity_ref",
                "hermiticity_pred",
            )
        },
        "graph2mat_roundtrip": graph2mat_roundtrip(
            run_fdf=paths["run_fdf"],
            reference_path=paths["reference"],
            predicted_path=paths["prediction"],
            basis_files=basis_files,
            output_dir=args.output_dir,
            n_matrix_components=args.n_matrix_components,
        ),
        "overlap_probe": overlap_probe(paths["reference"], paths["prediction"], wrong_reference_path),
        "spectral_probe": spectral_probe(reference, predicted, wrong_reference),
        "single_sample_overfit": {
            "status": "NOT_RUN",
            "reason": (
                "Training a fresh MACE/Graph2Mat model is intentionally not performed by this "
                "read-only diagnostic script. Use the report's reproduction recipe to run it in "
                "an isolated workspace."
            ),
        },
    }

    output = args.output_dir / "h2o_hamiltonian_diagnostics.json"
    output.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_safe({"output": str(output), "sample": args.sample}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
