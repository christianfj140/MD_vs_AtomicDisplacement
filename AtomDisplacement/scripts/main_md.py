#!/usr/bin/env python3
"""Script principal para ejecutar el pipeline MD completo en orden.

Orden estricto de ejecución:
1) generate_md_dataset.py
2) run_md_training.py
3) run_md_testing.py
4) run_md_prediction.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPTS = [
    SCRIPTS_DIR / "generate_md_dataset.py",
    SCRIPTS_DIR / "run_md_training.py",
    SCRIPTS_DIR / "run_md_testing.py",
    SCRIPTS_DIR / "run_md_prediction.py",
]


def run_step(script_path: Path) -> None:
    if not script_path.exists():
        raise RuntimeError(f"No existe el script requerido: {script_path}")

    cmd = [sys.executable, str(script_path)]
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"Falló el paso {script_path.name} con código {result.returncode}."
        )


def main() -> int:
    print("=== Pipeline MD completo ===")
    for step in PIPELINE_SCRIPTS:
        run_step(step)

    print("\n=== Pipeline MD completo finalizado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
