#!/usr/bin/env python3
"""Run SIESTA single-point calculations for all generated H2O samples."""

from __future__ import annotations

import argparse
import shutil

from atom_displacement_utils import (
    ATDIS_STEPS_DIR_NAME,
    DATASET_DIR,
    PIPELINE_CONFIG,
    PIPELINE_PATHS,
    generated_sample_dirs,
    require_command,
    run_siesta_in_dir,
    validate_sample_dir,
    write_validation_outputs,
    write_json,
)
from pipeline_config_utils import command


def build_argument_parser() -> argparse.ArgumentParser:
    single_points = PIPELINE_CONFIG["single_points"]
    parser = argparse.ArgumentParser(description="Lanza SIESTA para todas las muestras generadas.")
    parser.add_argument("--limit", type=int, default=single_points["limit"])
    parser.add_argument("--rerun", action="store_true", default=bool(single_points["rerun"]))
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


def main() -> int:
    args = build_argument_parser().parse_args()
    require_command(command(PIPELINE_CONFIG, "shell"))
    require_command(command(PIPELINE_CONFIG, "siesta"))

    if args.rerun:
        atdis_steps_dir = DATASET_DIR / ATDIS_STEPS_DIR_NAME
        if atdis_steps_dir.exists():
            shutil.rmtree(atdis_steps_dir)

    sample_dirs = generated_sample_dirs()
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]

    print("=== Single-point SIESTA runs ===")
    print(f"[INFO] Muestras detectadas: {len(sample_dirs)}")

    summary = []
    validation_rows = []
    for sample_dir in sample_dirs:
        validation = validate_sample_dir(
            sample_dir,
            allow_unvalidated_matrices=args.allow_unvalidated_matrices,
        )
        if validation["valid"] and not args.rerun:
            print(f"[SKIP] {sample_dir.name} validada: Hamiltoniano y SIESTA convergido")
            summary.append(
                {
                    "id": sample_dir.name,
                    "status": "skipped_validated",
                    "validation_reason": validation["validation_reason"],
                }
            )
            validation_rows.append(validation)
            continue

        print(f"[RUN] {sample_dir.name}")
        try:
            run_siesta_in_dir(sample_dir, sample_dir / PIPELINE_CONFIG["paths"]["run_out_name"])
            validation = validate_sample_dir(
                sample_dir,
                allow_unvalidated_matrices=args.allow_unvalidated_matrices,
            )
            summary.append(
                {
                    "id": sample_dir.name,
                    "status": "completed" if validation["valid"] else "completed_invalid",
                    "validation_reason": validation["validation_reason"],
                }
            )
        except Exception as exc:
            validation = validate_sample_dir(
                sample_dir,
                allow_unvalidated_matrices=args.allow_unvalidated_matrices,
            )
            summary.append(
                {
                    "id": sample_dir.name,
                    "status": "failed",
                    "error": str(exc),
                    "validation_reason": validation["validation_reason"],
                }
            )
            print(f"[ERROR] {sample_dir.name}: {exc}")
        validation_rows.append(validation)

    write_json(PIPELINE_PATHS["run_summary_path"], summary)
    validation_summary = write_validation_outputs(DATASET_DIR / "validation", validation_rows)
    print("[OK] Resumen de ejecucion guardado en dataset/run_summary.json")
    print(f"[OK] Validacion de muestras guardada en {validation_summary['outputs']['validation_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
