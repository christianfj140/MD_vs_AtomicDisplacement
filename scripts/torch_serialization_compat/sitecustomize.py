"""Auto-register safe PyTorch globals for subprocess CLI entrypoints."""

from __future__ import annotations

import os

try:
    from torch_safe_globals import (
        allow_graph2mat_checkpoint_globals,
        apply_torch_float32_matmul_precision,
        apply_torch_num_threads,
    )

    allow_graph2mat_checkpoint_globals()
    apply_torch_float32_matmul_precision(os.environ.get("TORCH_FLOAT32_MATMUL_PRECISION"))
    apply_torch_num_threads(os.environ.get("TORCH_NUM_THREADS"))
except Exception:
    pass
