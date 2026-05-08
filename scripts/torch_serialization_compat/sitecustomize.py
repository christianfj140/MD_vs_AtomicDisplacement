"""Auto-register safe PyTorch globals for subprocess CLI entrypoints."""

from __future__ import annotations

try:
    from torch_safe_globals import allow_graph2mat_checkpoint_globals

    allow_graph2mat_checkpoint_globals()
except Exception:
    pass
