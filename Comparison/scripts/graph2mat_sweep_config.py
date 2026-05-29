#!/usr/bin/env python3
"""Reusable Graph2Mat sweep helpers.

This module intentionally mirrors the Graph2Mat sweep keys already exposed by
the Comparison Experiment tab, but keeps the pure validation/expansion logic
small enough to be reused by the dedicated Graph2Mat-vs-DeepH runner.
"""

from __future__ import annotations

import re
from typing import Any


HIDDEN_IRREPS_TERM_RE = re.compile(r"^(?:(\d+)\s*x\s*)?(\d+)\s*([eoEO])$")

GRAPH2MAT_SIMPLE_NODE_BLOCK = "graph2mat.bindings.e3nn.E3nnSimpleNodeBlock"
GRAPH2MAT_SIMPLE_EDGE_BLOCK = "graph2mat.bindings.e3nn.E3nnSimpleEdgeBlock"
GRAPH2MAT_EDGE_BLOCK_NODE_MIX = "graph2mat.bindings.e3nn.E3nnEdgeBlockNodeMix"
GRAPH2MAT_EDGE_MESSAGE_BLOCK = "graph2mat.bindings.e3nn.E3nnEdgeMessageBlock"
GRAPH2MAT_READOUT_FAMILIES = {"default", "edge_node_mix"}
GRAPH2MAT_READOUT_MODEL_KEYS = {
    "node_block_readout",
    "edge_block_readout",
    "preprocessing_edges",
    "preprocessing_edges_reuse_nodes",
}
GRAPH2MAT_READOUT_OVERRIDES = {
    "default": {
        "node_block_readout": GRAPH2MAT_SIMPLE_NODE_BLOCK,
        "edge_block_readout": GRAPH2MAT_SIMPLE_EDGE_BLOCK,
        "preprocessing_edges": GRAPH2MAT_EDGE_MESSAGE_BLOCK,
        "preprocessing_edges_reuse_nodes": False,
    },
    "edge_node_mix": {
        "node_block_readout": GRAPH2MAT_SIMPLE_NODE_BLOCK,
        "edge_block_readout": GRAPH2MAT_EDGE_BLOCK_NODE_MIX,
        "preprocessing_edges": GRAPH2MAT_EDGE_MESSAGE_BLOCK,
        "preprocessing_edges_reuse_nodes": True,
    },
}

GRAPH2MAT_SWEEP_KEYS = {
    "max_epochs",
    "optim_lr",
    "batch_size",
    "loader_threads",
    "seed_everything",
    "loss",
    "loss_kwargs",
    "num_interactions",
    "correlation",
    "max_ell",
    "hidden_irreps",
    "hidden_irreps_channels",
    "readout",
    *GRAPH2MAT_READOUT_MODEL_KEYS,
}


def parse_hidden_irreps_terms(value: str, name: str = "hidden_irreps") -> list[dict[str, Any]]:
    terms: list[dict[str, Any]] = []
    for raw_term in str(value).split("+"):
        term = raw_term.strip()
        if not term:
            raise RuntimeError(f"{name}: empty term in hidden_irreps.")
        match = HIDDEN_IRREPS_TERM_RE.fullmatch(term)
        if not match:
            raise RuntimeError(
                f'{name}: "{term}" is not a valid Irreps term. Use NxLe, for example 32x1o.'
            )
        multiplier = int(match.group(1) or "1")
        ell = int(match.group(2))
        parity = match.group(3).lower()
        if multiplier <= 0:
            raise RuntimeError(f"{name}: multiplier must be positive.")
        terms.append({"multiplier": multiplier, "ell": ell, "parity": parity})
    return terms


def expected_hidden_irreps(multiplier: int, max_ell: int) -> str:
    return " + ".join(
        f"{int(multiplier)}x{ell}{'e' if ell % 2 == 0 else 'o'}"
        for ell in range(int(max_ell) + 1)
    )


def build_hidden_irreps(channels: int, max_ell: int) -> str:
    if int(channels) <= 0:
        raise RuntimeError("hidden_irreps_channels must be a positive integer.")
    if int(max_ell) < 0:
        raise RuntimeError("max_ell must be >= 0 when generating hidden_irreps.")
    return expected_hidden_irreps(int(channels), int(max_ell))


def hidden_irreps_dimension(hidden_irreps: str) -> int:
    terms = parse_hidden_irreps_terms(hidden_irreps)
    dimension = sum(int(term["multiplier"]) * (2 * int(term["ell"]) + 1) for term in terms)
    if dimension <= 0:
        raise RuntimeError("hidden_irreps dimension must be positive.")
    return dimension


def validate_hidden_irreps(value: str, max_ell: int | None, name: str = "hidden_irreps") -> None:
    terms = parse_hidden_irreps_terms(value, name)
    multipliers = {int(term["multiplier"]) for term in terms}
    if len(multipliers) != 1:
        raise RuntimeError(f"{name}: all terms must use the same channel multiplier.")
    seen: set[int] = set()
    for term in terms:
        ell = int(term["ell"])
        if ell in seen:
            raise RuntimeError(f"{name}: l={ell} appears more than once.")
        seen.add(ell)
        expected_parity = "e" if ell % 2 == 0 else "o"
        if str(term["parity"]) != expected_parity:
            raise RuntimeError(f"{name}: unexpected parity for l={ell}; use {ell}{expected_parity}.")
    if max_ell is None:
        return
    max_seen = max(seen)
    multiplier = next(iter(multipliers))
    expected = expected_hidden_irreps(multiplier, max_ell)
    if max_seen != int(max_ell):
        raise RuntimeError(f"{name}: lmax={max_seen} does not match max_ell={max_ell}. Use {expected}.")
    missing = [ell for ell in range(int(max_ell) + 1) if ell not in seen]
    if missing:
        raise RuntimeError(f"{name}: missing ell values {missing}. Use {expected}.")


def _normalize_readout_family(value: Any) -> str:
    family = str(value).strip().lower()
    if family not in GRAPH2MAT_READOUT_FAMILIES:
        raise RuntimeError(
            "graph2mat.readout must be one of: "
            + ", ".join(sorted(GRAPH2MAT_READOUT_FAMILIES))
            + "."
        )
    return family


def _apply_readout_family(normalized: dict[str, Any]) -> None:
    if "readout" not in normalized:
        return
    family = _normalize_readout_family(normalized["readout"])
    normalized["readout"] = family
    for key, value in GRAPH2MAT_READOUT_OVERRIDES[family].items():
        if key in normalized and normalized[key] != value:
            raise RuntimeError(
                f"graph2mat.readout={family} conflicts with explicit {key}={normalized[key]!r}."
            )
        normalized[key] = value


def normalize_graph2mat_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(overrides) - GRAPH2MAT_SWEEP_KEYS)
    if unknown:
        raise RuntimeError(f"Unsupported Graph2Mat sweep keys: {', '.join(unknown)}.")
    normalized = dict(overrides)
    if normalized.get("hidden_irreps") is not None and normalized.get("hidden_irreps_channels") is not None:
        raise RuntimeError("Use hidden_irreps or hidden_irreps_channels, not both.")
    if normalized.get("hidden_irreps_channels") is not None:
        if normalized.get("max_ell") is None:
            raise RuntimeError("hidden_irreps_channels requires max_ell.")
        normalized["hidden_irreps"] = build_hidden_irreps(
            int(normalized["hidden_irreps_channels"]),
            int(normalized["max_ell"]),
        )
        normalized.pop("hidden_irreps_channels", None)
    if normalized.get("hidden_irreps") is not None:
        max_ell = normalized.get("max_ell")
        validate_hidden_irreps(
            str(normalized["hidden_irreps"]),
            int(max_ell) if max_ell is not None else None,
            "graph2mat.hidden_irreps",
        )
    _apply_readout_family(normalized)
    return normalized
