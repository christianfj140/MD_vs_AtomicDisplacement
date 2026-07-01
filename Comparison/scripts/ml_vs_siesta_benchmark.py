#!/usr/bin/env python3
"""Entrypoint for the ML vs SIESTA benchmark CLI.

Thin wrapper so the package can be run as a script:

    python Comparison/scripts/ml_vs_siesta_benchmark.py generate-siesta-displacements \\
        --config Comparison/config/ml_vs_siesta_benchmark_example.yaml \\
        --output runs/example --dry-run

Never launches SIESTA and never trains.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ml_vs_siesta.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
