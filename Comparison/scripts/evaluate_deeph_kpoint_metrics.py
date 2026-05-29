#!/usr/bin/env python3
"""Evaluate DeepH H5 block predictions with the repository k-point metrics."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from configparser import ConfigParser
from pathlib import Path
from typing import Any

import numpy as np

from deeph_fair_utils import (
    DEFAULT_DEEPH_REPO,
    DeepHFairBenchmarkError,
    count_orbitals_from_orbital_types,
    deeph_subprocess_env,
    load_split_samples,
    read_json,
    run_git_commit,
    run_subprocess_streaming,
    sample_limit_by_split,
    sha256_file,
    write_csv_rows,
    write_deeph_config,
    write_json,
)
from deeph_prediction_adapter import DeepHPredictionAdapterError, adapt_deeph_prediction_sample
from evaluate_hamiltonian_metrics import (
    LOW_ENERGY_ALIGNMENT,
    LOW_ENERGY_N_STATES,
    complex_generalized_eigenvalues,
    complex_hermiticity_defect,
    complex_matrix_error_metrics,
    eigen_error_metrics,
    kpoint_weighted_dos_metrics,
    low_energy_metrics_from_eigenvalues,
    parse_monkhorst_pack_kgrid,
    read_matrix,
    weighted_metric_mean,
    weighted_metric_rmse,
)


def parse_block_key(key: str) -> tuple[tuple[int, int, int], int, int]:
    values = json.loads(key)
    if not isinstance(values, list) or len(values) != 5:
        raise DeepHFairBenchmarkError(f"Invalid DeepH block key: {key}")
    return (int(values[0]), int(values[1]), int(values[2])), int(values[3]) - 1, int(values[4]) - 1


def assemble_hk(block_h5: Path, sample_dir: Path, kpoint: tuple[float, float, float]) -> np.ndarray:
    import h5py

    orbital_counts = count_orbitals_from_orbital_types(sample_dir / "orbital_types.dat")
    offsets = np.cumsum([0, *orbital_counts])
    matrix = np.zeros((int(offsets[-1]), int(offsets[-1])), dtype=np.complex128)
    with h5py.File(block_h5, "r") as handle:
        for key in handle.keys():
            lattice_r, atom_i, atom_j = parse_block_key(key)
            if atom_i < 0 or atom_j < 0 or atom_i >= len(orbital_counts) or atom_j >= len(orbital_counts):
                raise DeepHFairBenchmarkError(f"Block key atom index out of range in {block_h5}: {key}")
            block = np.asarray(handle[key][()])
            phase = np.exp(2j * np.pi * float(np.dot(np.asarray(kpoint, dtype=float), np.asarray(lattice_r, dtype=float))))
            r0, r1 = int(offsets[atom_i]), int(offsets[atom_i + 1])
            c0, c1 = int(offsets[atom_j]), int(offsets[atom_j + 1])
            if block.shape != (r1 - r0, c1 - c0):
                raise DeepHFairBenchmarkError(
                    f"Block shape mismatch in {block_h5}: {key} has {block.shape}, expected {(r1-r0, c1-c0)}"
                )
            matrix[r0:r1, c0:c1] += block * phase
    return matrix


def copy_processed_sample(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def generate_prediction(
    *,
    sample_dir: Path,
    work_dir: Path,
    trained_model_dir: Path,
    deeph_repo: Path,
    python: Path,
    disable_cuda: bool,
    device: str,
) -> Path:
    copy_processed_sample(sample_dir, work_dir)
    config = ConfigParser()
    config.read(deeph_repo / "deeph" / "inference" / "inference_default.ini")
    config.set("basic", "work_dir", str(work_dir.resolve()))
    config.set("basic", "OLP_dir", str(work_dir.resolve()))
    config.set("basic", "interface", "openmx")
    config.set("basic", "trained_model_dir", json.dumps([str(trained_model_dir.resolve())]))
    config.set("basic", "task", "[3, 4]")
    config.set("basic", "disable_cuda", str(disable_cuda))
    config.set("basic", "device", device)
    config.set("basic", "huge_structure", "False")
    config.set("basic", "restore_blocks_py", "True")
    config.set("basic", "with_grad", "False")
    config.set("graph", "radius", "-1")
    config.set("graph", "create_from_DFT", "True")
    config_path = work_dir / "deeph_inference_config.ini"
    write_deeph_config(config_path, config)
    command = [
        str(python),
        "-u",
        str(deeph_repo / "deeph" / "scripts" / "inference.py"),
        "--config",
        str(config_path.resolve()),
    ]
    returncode = run_subprocess_streaming(
        command,
        cwd=deeph_repo,
        env=deeph_subprocess_env(deeph_repo),
        stdout_path=work_dir / "deeph_inference_stdout.log",
        stderr_path=work_dir / "deeph_inference_stderr.log",
        prefix=f"[DEEPh][inference:{sample_dir.name}] ",
    )
    if returncode != 0:
        raise DeepHFairBenchmarkError(f"DeepH inference failed for {sample_dir.name}: {returncode}")
    prediction = work_dir / "hamiltonians_pred.h5"
    if not prediction.exists():
        raise DeepHFairBenchmarkError(f"DeepH inference did not create {prediction}")
    return prediction


def graph2mat_reference_path(graph2mat_result_dir: Path, sample: str) -> Path | None:
    sample_dir = graph2mat_result_dir / "siesta_hamiltonians" / sample
    for suffix in (".TSHS", ".HSX"):
        matches = sorted(path for path in sample_dir.glob(f"*{suffix}") if path.name != "ML_prediction.HSX")
        if matches:
            return matches[0]
    return None


def evaluate_sample(args: argparse.Namespace, sample, rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    processed_sample = args.processed_dir / sample.sample
    if not processed_sample.exists():
        return {"sample": sample.sample, "status": "missing_processed_sample", "error": str(processed_sample)}
    run_fdf = args.graph2mat_result_dir / "structures" / sample.sample / "RUN.fdf"
    if not run_fdf.exists() and sample.structure_path:
        run_fdf = sample.structure_path
    kgrid = parse_monkhorst_pack_kgrid(run_fdf)
    if kgrid is None or not kgrid.ok:
        return {"sample": sample.sample, "status": "missing_or_invalid_kgrid", "error": kgrid.error if kgrid else "missing"}
    prediction = args.predictions_dir / sample.sample / args.prediction_filename
    if args.generate_predictions:
        prediction = generate_prediction(
            sample_dir=processed_sample,
            work_dir=args.predictions_dir / sample.sample,
            trained_model_dir=args.trained_model_dir,
            deeph_repo=args.deeph_repo,
            python=args.python,
            disable_cuda=args.disable_cuda,
            device=args.device,
        )
    if not prediction.exists():
        local_frame_prediction = args.predictions_dir / sample.sample / "rh_pred.h5"
        if local_frame_prediction.exists():
            prediction = local_frame_prediction
        else:
            return {"sample": sample.sample, "status": "missing_prediction_h5", "error": str(prediction)}
    ref_h5 = processed_sample / "hamiltonians.h5"
    overlap_h5 = processed_sample / "overlaps.h5"
    for required in (ref_h5, overlap_h5, processed_sample / "orbital_types.dat"):
        if not required.exists():
            return {"sample": sample.sample, "status": "missing_processed_artifact", "error": str(required)}
    try:
        adapter = adapt_deeph_prediction_sample(
            work_dir=prediction.parent,
            processed_sample_dir=processed_sample,
            sample_id=sample.sample,
            prediction_filename=args.prediction_filename,
        )
    except DeepHPredictionAdapterError as exc:
        return {"sample": sample.sample, "status": "adapter_failed", "error": str(exc)}
    if not adapter.metrics_ready or adapter.prediction_path is None:
        return {
            "sample": sample.sample,
            "status": "prediction_not_common_metric_ready",
            "error": adapter.diagnostic_reason,
            "adapter": adapter.to_dict(),
        }
    prediction = Path(adapter.prediction_path)
    adapter_fields = adapter.metric_fields()

    fermi_level = None
    fermi_source = "unavailable"
    reference_matrix = graph2mat_reference_path(args.graph2mat_result_dir, sample.sample)
    if reference_matrix is not None:
        matrix_data = read_matrix(reference_matrix)
        fermi_level = matrix_data.fermi_level
        fermi_source = matrix_data.fermi_level_source or "siesta_reference"

    per_k_matrix: list[dict[str, Any]] = []
    per_k_spectral: list[dict[str, Any]] = []
    all_ref_eigs: list[float] = []
    all_pred_eigs: list[float] = []
    all_weights: list[float] = []
    eigen_root = args.output_dir / "eigenvalues"
    for k_index, (kpoint, weight) in enumerate(zip(kgrid.fractional_kpoints, kgrid.weights, strict=True)):
        ref_h = assemble_hk(ref_h5, processed_sample, kpoint)
        pred_h = assemble_hk(prediction, processed_sample, kpoint)
        overlap = assemble_hk(overlap_h5, processed_sample, kpoint)
        matrix_metrics = complex_matrix_error_metrics(ref_h, pred_h)
        row = {
            "sample": sample.sample,
            "row_type": "per_k",
            "k_index": k_index,
            "k_label": f"k{k_index:04d}",
            "kx": kpoint[0],
            "ky": kpoint[1],
            "kz": kpoint[2],
            "k_weight": weight,
            "kpoint_mesh": json.dumps(kgrid.mesh),
            "kpoint_shifts": json.dumps(kgrid.shifts),
            "kpoint_source": kgrid.source_directive,
            "n_orbitals": ref_h.shape[0],
            "n_entries": ref_h.size,
            "h_mae_eV": matrix_metrics["mae_eV"],
            "h_rmse_eV": matrix_metrics["rmse_eV"],
            "h_mse_eV2": matrix_metrics["mse_eV2"],
            "h_max_abs_error_eV": matrix_metrics["max_abs_error_eV"],
            "relative_frobenius": matrix_metrics["relative_frobenius"],
            "hermiticity_ref": complex_hermiticity_defect(ref_h),
            "hermiticity_pred": complex_hermiticity_defect(pred_h),
            "uses_reference_overlap_k": True,
            "target_component_policy": "deeph_hamiltonian_h5_blocks",
            "prediction_own_overlap_used": False,
            **adapter_fields,
        }
        rows["kpoint_matrix"].append(row)
        per_k_matrix.append(row)
        ref_eig = complex_generalized_eigenvalues(ref_h, overlap)
        pred_eig = complex_generalized_eigenvalues(pred_h, overlap)
        all_ref_eigs.extend(ref_eig.tolist())
        all_pred_eigs.extend(pred_eig.tolist())
        all_weights.extend([float(weight)] * min(ref_eig.size, pred_eig.size))
        band_rows, spectral = eigen_error_metrics(ref_eig, pred_eig, fermi_level, fermi_source)
        low_energy = (
            {
                "low_energy_requested_states": args.low_energy_n_states,
                "low_energy_n_states": None,
                "low_energy_mae_eV": math.nan,
                "low_energy_rmse_eV": math.nan,
                "low_energy_max_abs_error_eV": math.nan,
                "low_energy_alignment": args.low_energy_alignment,
                "low_energy_aligned_rmse_eV": math.nan,
                "low_energy_overlap_used": True,
                "low_energy_overlap_required": True,
                "low_energy_solver": "scipy.linalg.eigh_generalized_kpoint",
                "low_energy_warning": "low-energy metrics disabled by CLI option.",
            }
            if args.disable_low_energy
            else low_energy_metrics_from_eigenvalues(
                ref_eig,
                pred_eig,
                n_states=args.low_energy_n_states,
                alignment=args.low_energy_alignment,
            )
        )
        spectral_row = {
            "sample": sample.sample,
            "k_index": k_index,
            "k_weight": weight,
            "uses_reference_overlap_k": True,
            **spectral,
            **low_energy,
        }
        per_k_spectral.append(spectral_row)
        write_csv_rows(eigen_root / "siesta" / f"{sample.sample}_k{k_index:04d}.csv", [{"band": i, "eigenvalue_eV": v} for i, v in enumerate(ref_eig)])
        write_csv_rows(eigen_root / "predicted" / f"{sample.sample}_k{k_index:04d}.csv", [{"band": i, "eigenvalue_eV": v} for i, v in enumerate(pred_eig)])
        write_csv_rows(eigen_root / "kpoint_band_errors" / f"{sample.sample}_k{k_index:04d}.csv", band_rows)

    rows["kpoint_matrix"].append(
        {
            "sample": sample.sample,
            "row_type": "weighted_sample",
            "k_index": "",
            "k_label": "weighted",
            "k_weight": 1.0,
            "kpoint_mesh": json.dumps(kgrid.mesh),
            "kpoint_shifts": json.dumps(kgrid.shifts),
            "kpoint_source": kgrid.source_directive,
            "n_orbitals": per_k_matrix[0]["n_orbitals"],
            "n_entries": per_k_matrix[0]["n_entries"],
            "h_mae_eV": weighted_metric_mean(per_k_matrix, "h_mae_eV"),
            "h_rmse_eV": weighted_metric_mean(per_k_matrix, "h_rmse_eV"),
            "relative_frobenius": weighted_metric_mean(per_k_matrix, "relative_frobenius"),
            "hermiticity_ref": weighted_metric_mean(per_k_matrix, "hermiticity_ref"),
            "hermiticity_pred": weighted_metric_mean(per_k_matrix, "hermiticity_pred"),
            "uses_reference_overlap_k": True,
            "target_component_policy": "deeph_hamiltonian_h5_blocks",
            "prediction_own_overlap_used": False,
            **adapter_fields,
        }
    )
    spectral_weighted = {
        "sample": sample.sample,
        "kpoint_count": len(kgrid.fractional_kpoints),
        "kpoint_mesh": json.dumps(kgrid.mesh),
        "kpoint_shifts": json.dumps(kgrid.shifts),
        "kpoint_source": kgrid.source_directive,
        "uses_reference_overlap_k": True,
        "global_mae_eV": weighted_metric_mean(per_k_spectral, "global_mae_eV"),
        "global_rmse_eV": weighted_metric_rmse(per_k_spectral, "global_rmse_eV"),
        "low_energy_requested_states": args.low_energy_n_states,
        "low_energy_n_states": weighted_metric_mean(per_k_spectral, "low_energy_n_states"),
        "low_energy_mae_eV": weighted_metric_mean(per_k_spectral, "low_energy_mae_eV"),
        "low_energy_rmse_eV": weighted_metric_rmse(per_k_spectral, "low_energy_rmse_eV"),
        "low_energy_max_abs_error_eV": max(
            (
                float(row["low_energy_max_abs_error_eV"])
                for row in per_k_spectral
                if math.isfinite(float(row["low_energy_max_abs_error_eV"]))
            ),
            default=math.nan,
        ),
        "low_energy_alignment": args.low_energy_alignment,
        "low_energy_aligned_rmse_eV": weighted_metric_rmse(per_k_spectral, "low_energy_aligned_rmse_eV"),
        "low_energy_overlap_used": True,
        "low_energy_overlap_required": True,
        "low_energy_solver": "scipy.linalg.eigh_generalized_kpoint",
        "low_energy_warning": "; ".join(
            sorted(
                {
                    str(row.get("low_energy_warning"))
                    for row in per_k_spectral
                    if str(row.get("low_energy_warning") or "").strip()
                }
            )
        ),
        "fermi_window_rmse_eV": weighted_metric_rmse(per_k_spectral, "fermi_window_rmse_eV"),
        "frontier_window_rmse_eV": weighted_metric_rmse(per_k_spectral, "frontier_window_rmse_eV"),
        "gap_abs_error_eV": weighted_metric_mean(per_k_spectral, "gap_abs_error_eV"),
        "fermi_ref_eV": fermi_level,
        "fermi_level_source": fermi_source,
        **adapter_fields,
    }
    rows["kpoint_spectral"].append(spectral_weighted)
    dos = kpoint_weighted_dos_metrics(
        np.asarray(all_ref_eigs, dtype=float),
        np.asarray(all_pred_eigs, dtype=float),
        np.asarray(all_weights, dtype=float),
        fermi_level,
    )
    rows["kpoint_dos"].append(
        {
            "sample": sample.sample,
            "kpoint_count": len(kgrid.fractional_kpoints),
            "kpoint_mesh": json.dumps(kgrid.mesh),
            "kpoint_shifts": json.dumps(kgrid.shifts),
            "kpoint_source": kgrid.source_directive,
            "weighted_eigenvalue_count": len(all_ref_eigs),
            **adapter_fields,
            **dos,
        }
    )
    rows["kpoints"].extend(
        {
            "sample": sample.sample,
            "k_index": index,
            "kx": kpoint[0],
            "ky": kpoint[1],
            "kz": kpoint[2],
            "weight": weight,
            "mesh": json.dumps(kgrid.mesh),
            "shifts": json.dumps(kgrid.shifts),
        }
        for index, (kpoint, weight) in enumerate(zip(kgrid.fractional_kpoints, kgrid.weights, strict=True))
    )
    return {
        "sample": sample.sample,
        "status": "ok",
        "prediction": str(prediction),
        "prediction_sha256": sha256_file(prediction),
        "reference_h5": str(ref_h5),
        "overlap_h5": str(overlap_h5),
        "kpoint_count": len(kgrid.fractional_kpoints),
        "adapter": adapter.to_dict(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.output_dir / "metrics"
    eigen_dir = args.output_dir / "eigenvalues"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    eigen_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        sample
        for sample in sample_limit_by_split(load_split_samples(args.graph2mat_result_dir.resolve()), args.sample_limit_per_split)
        if sample.split == args.split
    ]
    rows: dict[str, list[dict[str, Any]]] = {"kpoint_matrix": [], "kpoint_spectral": [], "kpoint_dos": [], "kpoints": []}
    statuses = [evaluate_sample(args, sample, rows) for sample in samples]
    write_csv_rows(metrics_dir / "kpoint_matrix_metrics.csv", rows["kpoint_matrix"])
    write_csv_rows(metrics_dir / "kpoint_spectral_metrics.csv", rows["kpoint_spectral"])
    write_csv_rows(metrics_dir / "kpoint_dos_metrics.csv", rows["kpoint_dos"])
    write_csv_rows(eigen_dir / "kpoints.csv", rows["kpoints"])
    write_csv_rows(metrics_dir / "sample_status.csv", statuses)
    failed = [status for status in statuses if status.get("status") != "ok"]
    manifest = {
        "stage": "deeph_kpoint_metrics",
        "graph2mat_result_dir": str(args.graph2mat_result_dir.resolve()),
        "processed_dir": str(args.processed_dir.resolve()),
        "predictions_dir": str(args.predictions_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "split": args.split,
        "samples_seen": len(samples),
        "samples_compared": len(statuses) - len(failed),
        "samples_failed": len(failed),
        "kpoint_metrics_enabled": True,
        "uses_reference_overlap_k": True,
        "prediction_own_overlap_used": False,
        "prediction_adapter": "deeph_hdf5_prediction_adapter",
        "prediction_adapter_version": "deeph_hdf5_prediction_adapter_v1",
        "sample_status": statuses,
        "pipeline_git": run_git_commit(Path.cwd()),
        "deepH_git": run_git_commit(args.deeph_repo.resolve()),
    }
    write_json(metrics_dir / "manifest.json", manifest)
    if failed and args.fail_closed:
        raise DeepHFairBenchmarkError(f"DeepH k-point evaluation failed for {len(failed)} samples; see {metrics_dir / 'manifest.json'}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph2mat-result-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trained-model-dir", type=Path, default=None)
    parser.add_argument("--deeph-repo", type=Path, default=DEFAULT_DEEPH_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_DEEPH_REPO / ".venv" / "bin" / "python")
    parser.add_argument("--prediction-filename", default="hamiltonians_pred.h5")
    parser.add_argument("--generate-predictions", action="store_true")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--sample-limit-per-split", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--fail-closed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-low-energy", action="store_true")
    parser.add_argument("--low-energy-n-states", type=int, default=LOW_ENERGY_N_STATES)
    parser.add_argument("--low-energy-alignment", default=LOW_ENERGY_ALIGNMENT, choices=["none", "global_shift"])
    args = parser.parse_args()
    if args.generate_predictions and args.trained_model_dir is None:
        parser.error("--generate-predictions requires --trained-model-dir")
    return args


def main() -> None:
    try:
        manifest = run(parse_args())
    except DeepHFairBenchmarkError as exc:
        raise SystemExit(f"[DEEPh-FAIR][ERROR] {exc}") from exc
    print(
        "[DEEPh-FAIR] k-point eval complete: "
        f"{manifest['samples_compared']}/{manifest['samples_seen']} compared; "
        f"manifest={Path(manifest['output_dir']) / 'metrics' / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
