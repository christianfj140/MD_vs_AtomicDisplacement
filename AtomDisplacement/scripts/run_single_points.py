#!/usr/bin/env python3
"""Run SIESTA single-point calculations for all generated H2O samples."""

from __future__ import annotations

import argparse

from atom_displacement_utils import (
    PIPELINE_CONFIG,
    PIPELINE_PATHS,
    SAMPLES_DIR,
    require_command,
    run_siesta_in_dir,
    sample_run_status,
    write_json,
)
from pipeline_config_utils import command


def build_argument_parser() -> argparse.ArgumentParser:
    single_points = PIPELINE_CONFIG["single_points"]
    parser = argparse.ArgumentParser(description="Lanza SIESTA para todas las muestras generadas.")
    parser.add_argument("--limit", type=int, default=single_points["limit"])
    parser.add_argument("--rerun", action="store_true", default=bool(single_points["rerun"]))
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    require_command(command(PIPELINE_CONFIG, "shell"))
    require_command(command(PIPELINE_CONFIG, "siesta"))

    sample_dirs = sorted(path for path in SAMPLES_DIR.glob("sample_*") if path.is_dir())
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]

    print("=== Single-point SIESTA runs ===")
    print(f"[INFO] Muestras detectadas: {len(sample_dirs)}")

    summary = []
    for sample_dir in sample_dirs:
        status = sample_run_status(sample_dir)
        if status["job_completed"] and not args.rerun:
            print(f"[SKIP] {sample_dir.name} ya completada")
            summary.append({"id": sample_dir.name, "status": "skipped_completed"})
            continue

        print(f"[RUN] {sample_dir.name}")
        try:
            run_siesta_in_dir(sample_dir, sample_dir / PIPELINE_CONFIG["paths"]["run_out_name"])
            summary.append({"id": sample_dir.name, "status": "completed"})
        except Exception as exc:
            summary.append({"id": sample_dir.name, "status": "failed", "error": str(exc)})
            print(f"[ERROR] {sample_dir.name}: {exc}")

    write_json(PIPELINE_PATHS["run_summary_path"], summary)
    print("[OK] Resumen de ejecucion guardado en dataset/run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
