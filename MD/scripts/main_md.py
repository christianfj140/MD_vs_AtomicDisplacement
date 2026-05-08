#!/usr/bin/env python3
"""Run the full MD pipeline in the order declared in pipeline_config.yaml."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from md_pipeline_config import load_pipeline_config

SCRIPTS_DIR = Path(__file__).resolve().parent
SCRIPT_BY_STEP = {
    "generate_md_dataset": SCRIPTS_DIR / "generate_md_dataset.py",
    "run_md_training": SCRIPTS_DIR / "run_md_training.py",
    "run_md_testing": SCRIPTS_DIR / "run_md_testing.py",
    "run_md_prediction": SCRIPTS_DIR / "run_md_prediction.py",
}
TEST_STEPS = {"run_md_testing"}


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
    config = load_pipeline_config()
    skip_model_test = bool(config.get("pipeline", {}).get("skip_model_test", False))
    pipeline_scripts = []
    for step in config["pipeline"]["steps"]:
        if skip_model_test and step in TEST_STEPS:
            print(f"[SKIP] {step}: pipeline.skip_model_test=true")
            continue
        pipeline_scripts.append(SCRIPT_BY_STEP[step])

    print("=== Pipeline MD completo ===")
    for step in pipeline_scripts:
        run_step(step)

    print("\n=== Pipeline MD completo finalizado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
