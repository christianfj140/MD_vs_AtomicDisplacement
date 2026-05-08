"""PyTorch checkpoint compatibility helpers for Graph2Mat/MACE runs."""

from __future__ import annotations

import os
from pathlib import Path


COMPAT_DIR = Path(__file__).resolve().parent
VALID_FLOAT32_MATMUL_PRECISIONS = {"high", "medium"}
THREAD_ENV_BY_SETTING = {
    "omp_num_threads": "OMP_NUM_THREADS",
    "mkl_num_threads": "MKL_NUM_THREADS",
    "openblas_num_threads": "OPENBLAS_NUM_THREADS",
    "torch_num_threads": "TORCH_NUM_THREADS",
}


def allow_graph2mat_checkpoint_globals() -> None:
    """Allow benign globals used by e3nn/Graph2Mat checkpoints."""

    try:
        import torch.serialization
    except ImportError:
        return

    torch.serialization.add_safe_globals([slice])


def apply_torch_float32_matmul_precision(precision: str | None) -> None:
    """Apply an opt-in float32 matmul precision setting when torch is present."""

    if precision in (None, "", "null"):
        return
    normalized = str(precision).strip().lower()
    if normalized not in VALID_FLOAT32_MATMUL_PRECISIONS:
        raise ValueError(
            "training.torch_float32_matmul_precision must be one of: "
            "null, high, medium"
        )
    try:
        import torch
    except ImportError:
        return
    torch.set_float32_matmul_precision(normalized)


def apply_torch_num_threads(value: str | None) -> None:
    """Apply an opt-in PyTorch intra-op thread count when torch is present."""

    if value in (None, "", "null"):
        return
    try:
        threads = int(str(value))
    except ValueError as exc:
        raise ValueError("torch_num_threads must be a positive integer") from exc
    if threads <= 0:
        raise ValueError("torch_num_threads must be a positive integer")
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(threads)


def env_with_torch_compat(
    env: dict[str, str] | None = None,
    *,
    matmul_precision: str | None = None,
    performance: dict[str, object] | None = None,
) -> dict[str, str]:
    """Return an environment where Python auto-loads this compat module."""

    patched = dict(os.environ if env is None else env)
    pythonpath = patched.get("PYTHONPATH")
    compat_path = str(COMPAT_DIR)
    patched["PYTHONPATH"] = (
        compat_path if not pythonpath else os.pathsep.join([compat_path, pythonpath])
    )
    if matmul_precision not in (None, "", "null"):
        normalized = str(matmul_precision).strip().lower()
        if normalized not in VALID_FLOAT32_MATMUL_PRECISIONS:
            raise ValueError(
                "training.torch_float32_matmul_precision must be one of: "
                "null, high, medium"
            )
        patched["TORCH_FLOAT32_MATMUL_PRECISION"] = normalized
    for setting, env_name in THREAD_ENV_BY_SETTING.items():
        value = (performance or {}).get(setting)
        if value in (None, "", "null"):
            continue
        threads = int(value)
        if threads <= 0:
            raise ValueError(f"{setting} must be a positive integer")
        patched[env_name] = str(threads)
    return patched
