#!/usr/bin/env python3
"""Run SIESTA single-point calculations for generated material samples."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import contextlib
import io
import time
from pathlib import Path
from typing import Any

from atom_displacement_utils import (
    ATDIS_STEPS_DIR_NAME,
    DATASET_DIR,
    PIPELINE_CONFIG,
    PIPELINE_PATHS,
    generated_sample_dirs,
    prepare_sample_material_inputs,
    require_command,
    run_siesta_in_dir,
    update_sample_execution_metadata,
    validate_sample_dir,
    write_validation_outputs,
    write_json,
)
from pipeline_config_utils import command


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_random_cartesian_hashes() -> None:
    manifest_path = DATASET_DIR / "RandomCartesian_steps" / "dataset_manifest.json"
    if not manifest_path.exists():
        return
    dataset_root = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if manifest.get("method_id") != "random_cartesian":
        return
    matrix_hashes = {}
    for matrix in sorted(list(dataset_root.glob("sample_*/*.TSHS")) + list(dataset_root.glob("sample_*/*.HSX"))):
        matrix_hashes[matrix.relative_to(dataset_root).as_posix()] = file_sha256(matrix)
    manifest["matrix_file_hashes"] = matrix_hashes
    for sample in manifest.get("samples", []):
        sample_dir = Path(str(sample.get("sample_dir", "")))
        matrices = sorted(list(sample_dir.glob("*.TSHS")) + list(sample_dir.glob("*.HSX")))
        if matrices:
            sample["matrix_file"] = str(matrices[0])
            sample["matrix_file_sha256"] = file_sha256(matrices[0])
        metadata_path = sample_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                material = metadata.get("material") if isinstance(metadata.get("material"), dict) else {}
                execution = metadata.get("siesta_execution") if isinstance(metadata.get("siesta_execution"), dict) else {}
                reference = metadata.get("reference_matrix") if isinstance(metadata.get("reference_matrix"), dict) else {}
                sample["material_label"] = material.get("label")
                sample["fdf_sha256"] = metadata.get("fdf_sha256")
                sample["pseudopotential_sha256"] = metadata.get("pseudopotential_sha256")
                sample["basis_file_sha256"] = metadata.get("basis_file_sha256")
                sample["job_completed"] = execution.get("job_completed")
                sample["scf_converged"] = execution.get("scf_converged")
                sample["reference_matrix_sha256"] = reference.get("sha256")
    write_json(manifest_path, manifest)
    artifact_path = dataset_root / "artifact_hashes.json"
    artifact = {}
    if artifact_path.exists():
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception:
            artifact = {}
    artifact["matrices"] = matrix_hashes
    write_json(artifact_path, artifact)


def build_argument_parser() -> argparse.ArgumentParser:
    single_points = PIPELINE_CONFIG["single_points"]
    parser = argparse.ArgumentParser(description="Lanza SIESTA para todas las muestras generadas.")
    parser.add_argument("--limit", type=int, default=single_points["limit"])
    parser.add_argument("--rerun", action="store_true", default=bool(single_points["rerun"]))
    parser.add_argument("--workers", type=int, default=int(single_points.get("workers", 1)))
    parser.add_argument(
        "--allow-unvalidated-matrices",
        action="store_true",
        default=bool(single_points.get("allow_unvalidated_matrices", False)),
        help=(
            "Solo para depuracion: permite reutilizar matrices existentes aunque "
            "RUN.out no demuestre completion y convergencia SCF."
        ),
    )
    return parser


def process_sample(sample_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], str]:
    started_at = time.time()
    log = io.StringIO()
    def emit(message: str) -> None:
        log.write(f"[{sample_dir.name}] {message}\n")

    try:
        material_manifest = prepare_sample_material_inputs(sample_dir)
        copied = material_manifest.get("pseudopotentials_copied_to_sample", {})
        if copied:
            emit(f"Material pseudos copied: {', '.join(sorted(copied.values()))}")
    except (RuntimeError, OSError) as exc:
        validation = validate_sample_dir(
            sample_dir,
            allow_unvalidated_matrices=args.allow_unvalidated_matrices,
        )
        emit(f"ERROR material preparation: {exc}")
        return (
            {
                "id": sample_dir.name,
                "status": "failed",
                "error": str(exc),
                "validation_reason": validation["validation_reason"],
                "wall_time_seconds": time.time() - started_at,
            },
            validation,
            log.getvalue(),
        )

    validation = validate_sample_dir(
        sample_dir,
        allow_unvalidated_matrices=args.allow_unvalidated_matrices,
    )
    if validation["valid"] and not args.rerun:
        if validation.get("unsafe_unvalidated_matrices"):
            emit("SKIP UNSAFE: matriz aceptada sin validacion completa de RUN.out/SCF")
        else:
            emit("SKIP validada: Hamiltoniano y SIESTA convergido")
        summary = {
            "id": sample_dir.name,
            "status": (
                "skipped_unvalidated"
                if validation.get("unsafe_unvalidated_matrices")
                else "skipped_validated"
            ),
            "validation_reason": validation["validation_reason"],
            "wall_time_seconds": time.time() - started_at,
        }
        update_sample_execution_metadata(sample_dir, validation, summary)
        return summary, validation, log.getvalue()

    emit("RUN")
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            run_siesta_in_dir(sample_dir, sample_dir / PIPELINE_CONFIG["paths"]["run_out_name"])
        validation = validate_sample_dir(
            sample_dir,
            allow_unvalidated_matrices=args.allow_unvalidated_matrices,
        )
        summary_row = {
            "id": sample_dir.name,
            "status": "completed" if validation["valid"] else "completed_invalid",
            "validation_reason": validation["validation_reason"],
            "wall_time_seconds": time.time() - started_at,
        }
        update_sample_execution_metadata(sample_dir, validation, summary_row)
        return summary_row, validation, log.getvalue()
    except Exception as exc:
        validation = validate_sample_dir(
            sample_dir,
            allow_unvalidated_matrices=args.allow_unvalidated_matrices,
        )
        emit(f"ERROR {exc}")
        summary_row = {
            "id": sample_dir.name,
            "status": "failed",
            "error": str(exc),
            "validation_reason": validation["validation_reason"],
            "wall_time_seconds": time.time() - started_at,
        }
        update_sample_execution_metadata(sample_dir, validation, summary_row)
        return (
            summary_row,
            validation,
            log.getvalue(),
        )


def main() -> int:
    args = build_argument_parser().parse_args()
    require_command(command(PIPELINE_CONFIG, "shell"))
    require_command(command(PIPELINE_CONFIG, "siesta"))
    if args.allow_unvalidated_matrices:
        print(
            "[WARN] UNSAFE_UNVALIDATED_MATRIX_REFERENCE mode enabled: "
            "existing matrices may be accepted without RUN.out completion/SCF proof."
        )

    if args.rerun:
        atdis_steps_dir = DATASET_DIR / ATDIS_STEPS_DIR_NAME
        if atdis_steps_dir.exists():
            shutil.rmtree(atdis_steps_dir)

    sample_dirs = generated_sample_dirs()
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]

    print("=== Single-point SIESTA runs ===")
    print(f"[INFO] Muestras detectadas: {len(sample_dirs)}")
    workers = max(1, int(args.workers or 1))
    print(f"[INFO] Workers single-point: {workers}")

    if workers == 1 or len(sample_dirs) <= 1:
        results = []
        for sample_dir in sample_dirs:
            result = process_sample(sample_dir, args)
            print(result[2], end="")
            results.append(result)
    else:
        indexed_results: list[tuple[int, dict[str, Any], dict[str, Any], str]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_sample, sample_dir, args): index
                for index, sample_dir in enumerate(sample_dirs)
            }
            for future in concurrent.futures.as_completed(futures):
                summary_row, validation, log = future.result()
                indexed_results.append((futures[future], summary_row, validation, log))
        indexed_results.sort(key=lambda item: item[0])
        for _index, _summary_row, _validation, log in indexed_results:
            print(log, end="")
        results = [(summary_row, validation, log) for _, summary_row, validation, log in indexed_results]

    summary = [summary_row for summary_row, _validation, _log in results]
    validation_rows = [validation for _summary_row, validation, _log in results]

    write_json(PIPELINE_PATHS["run_summary_path"], summary)
    validation_summary = write_validation_outputs(DATASET_DIR / "validation", validation_rows)
    refresh_random_cartesian_hashes()
    print("[OK] Resumen de ejecucion guardado en dataset/run_summary.json")
    print(f"[OK] Validacion de muestras guardada en {validation_summary['outputs']['validation_summary']}")
    if validation_summary.get("unsafe_unvalidated_samples"):
        print(
            "[WARN] UNSAFE_UNVALIDATED_MATRIX_REFERENCE: "
            f"{validation_summary['unsafe_unvalidated_samples']} sample(s) accepted in unsafe mode."
        )
    if validation_summary["invalid_samples"] and not args.allow_unvalidated_matrices:
        print(
            "[ERROR] Single-point validation failed: "
            f"{validation_summary['invalid_samples']} invalid samples."
        )
        return 2
            
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
