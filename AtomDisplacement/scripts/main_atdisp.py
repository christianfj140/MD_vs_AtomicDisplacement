#!/usr/bin/env python3
"""Run the AtomDisplacement train-test-predict pipeline in order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPTS = [
    SCRIPTS_DIR / "run_atdisp_training.py",
    SCRIPTS_DIR / "run_atdisp_testing.py",
    SCRIPTS_DIR / "run_atdisp_prediction.py",
]


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
    print("=== Pipeline AtomDisplacement: train + test + predict ===")
    for step in PIPELINE_SCRIPTS:
        run_step(step)

    print("\n=== Pipeline AtomDisplacement completado correctamente ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
