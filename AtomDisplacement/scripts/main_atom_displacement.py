#!/usr/bin/env python3
"""Run the full atom-displacement pipeline in order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pipeline_config_utils import load_pipeline_config, write_generated_inputs

SCRIPTS_DIR = Path(__file__).resolve().parent
STEP_SCRIPTS = {
    "run_relaxation": SCRIPTS_DIR / "run_relaxation.py",
    "generate_atom_displacement_dataset": SCRIPTS_DIR / "generate_atom_displacement_dataset.py",
    "generate_random_cartesian_dataset": SCRIPTS_DIR / "generate_random_cartesian_dataset.py",
    "run_single_points": SCRIPTS_DIR / "run_single_points.py",
    "normalize_fc_steps": SCRIPTS_DIR / "normalize_fc_steps.py",
    "collect_atom_displacement_dataset": SCRIPTS_DIR / "collect_atom_displacement_dataset.py",
    "run_atdisp_training": SCRIPTS_DIR / "run_atdisp_training.py",
    "run_atdisp_testing": SCRIPTS_DIR / "run_atdisp_testing.py",
    "run_atdisp_prediction": SCRIPTS_DIR / "run_atdisp_prediction.py",
}
TEST_STEPS = {"run_atdisp_testing"}


def run_step(script_path: Path) -> None:
    cmd = [sys.executable, str(script_path)]
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"El paso {script_path.name} fallo con codigo {result.returncode}"
        )


def main() -> int:
    config = load_pipeline_config()
    skip_model_test = bool(config.get("pipeline", {}).get("skip_model_test", False))
    print("=== Full atom-displacement pipeline ===")
    for step in config["pipeline"]["steps"]:
        if skip_model_test and step in TEST_STEPS:
            print(f"[SKIP] {step}: pipeline.skip_model_test=true")
            continue
        if step == "render_inputs":
            write_generated_inputs(config)
            print("[OK] Archivos derivados sincronizados desde pipeline_config.yaml")
            continue
        if step not in STEP_SCRIPTS:
            raise RuntimeError(f"Paso de pipeline desconocido: {step}")
        run_step(STEP_SCRIPTS[step])
    print("[OK] Pipeline completado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
