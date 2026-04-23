#!/usr/bin/env python3
"""Run the full atom-displacement pipeline in order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PIPELINE = [
    SCRIPTS_DIR / "run_relaxation.py",
    SCRIPTS_DIR / "generate_atom_displacement_dataset.py",
    SCRIPTS_DIR / "run_single_points.py",
    SCRIPTS_DIR / "collect_atom_displacement_dataset.py",
]


def run_step(script_path: Path) -> None:
    cmd = [sys.executable, str(script_path)]
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"El paso {script_path.name} fallo con codigo {result.returncode}"
        )


def main() -> int:
    print("=== Full atom-displacement pipeline ===")
    for step in PIPELINE:
        run_step(step)
    print("[OK] Pipeline completado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
