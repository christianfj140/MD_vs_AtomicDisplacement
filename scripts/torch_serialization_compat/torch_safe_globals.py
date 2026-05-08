"""PyTorch checkpoint compatibility helpers for Graph2Mat/MACE runs."""

from __future__ import annotations

import os
from pathlib import Path


COMPAT_DIR = Path(__file__).resolve().parent


def allow_graph2mat_checkpoint_globals() -> None:
    """Allow benign globals used by e3nn/Graph2Mat checkpoints."""

    try:
        import torch.serialization
    except ImportError:
        return

    torch.serialization.add_safe_globals([slice])


def env_with_torch_compat(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment where Python auto-loads this compat module."""

    patched = dict(os.environ if env is None else env)
    pythonpath = patched.get("PYTHONPATH")
    compat_path = str(COMPAT_DIR)
    patched["PYTHONPATH"] = (
        compat_path if not pythonpath else os.pathsep.join([compat_path, pythonpath])
    )
    return patched
