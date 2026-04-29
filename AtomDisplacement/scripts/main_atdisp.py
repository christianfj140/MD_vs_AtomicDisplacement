#!/usr/bin/env python3
"""Run the AtomDisplacement train-test-predict pipeline in order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pipeline_config_utils import load_pipeline_config, write_generated_inputs

SCRIPTS_DIR = Path(__file__).resolve().parent
STEP_SCRIPTS = {
    "run_relaxation": SCRIPTS_DIR / "run_relaxation.py",
    "generate_atom_displacement_dataset": SCRIPTS_DIR / "generate_atom_displacement_dataset.py",
    "run_single_points": SCRIPTS_DIR / "run_single_points.py",
    "normalize_fc_steps": SCRIPTS_DIR / "normalize_fc_steps.py",
    "collect_atom_displacement_dataset": SCRIPTS_DIR / "collect_atom_displacement_dataset.py",
    "run_atdisp_training": SCRIPTS_DIR / "run_atdisp_training.py",
    "run_atdisp_testing": SCRIPTS_DIR / "run_atdisp_testing.py",
    "run_atdisp_prediction": SCRIPTS_DIR / "run_atdisp_prediction.py",
}


def run_step(script_path: Path) -> None:
    if not script_path.exists():
        raise RuntimeError(f"No existe el script requerido: {script_path}")

    cmd = [sys.executable, str(script_path)]
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Fallo el paso {script_path.name} con codigo {result.returncode}."
        )


def main() -> int:
    config = load_pipeline_config()
    print("=== Pipeline AtomDisplacement ===")
    for step in config["pipeline"]["steps"]:
        if step == "render_inputs":
            write_generated_inputs(config)
            print("[OK] Archivos derivados sincronizados desde pipeline_config.yaml")
            continue
        if step not in STEP_SCRIPTS:
            raise RuntimeError(f"Paso de pipeline desconocido: {step}")
        run_step(STEP_SCRIPTS[step])

    print("\n=== Pipeline AtomDisplacement completado correctamente ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
