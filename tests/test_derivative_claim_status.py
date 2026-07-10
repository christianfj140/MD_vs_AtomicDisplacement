"""Fase 11 (audit): three-comparison protocol + degrading claim ladder."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))

from derivative_claim_status import (  # noqa: E402
    CLAIM_DIAGNOSTIC_ONLY,
    CLAIM_INVALID,
    CLAIM_PAPER_READY,
    CLAIM_VALIDATED_AGAINST_SIESTA,
    CLAIM_VALIDATED_MODEL_DERIVATIVE,
    COMPARISON_MODEL_AUTOGRAD_VS_MODEL_FD,
    COMPARISON_MODEL_AUTOGRAD_VS_SIESTA_FD,
    COMPARISON_MODEL_FD_VS_SIESTA_FD,
    comparison_kind,
    derive_claim_status,
)

FULL = dict(
    comparison_valid=True,
    values_finite=True,
    model_fd_gate_status="pass",
    base_equivalence_status="proven",
    siesta_converged=True,
    reproducibility_status="pinned_clean",
    fixed_test=True,
    n_seeds=3,
)


def test_comparison_kinds_are_distinguished():
    assert comparison_kind("autograd_graph2mat", "siesta_finite_difference") == (
        COMPARISON_MODEL_AUTOGRAD_VS_SIESTA_FD
    )
    assert comparison_kind("finite_difference_graph2mat", "siesta_finite_difference") == (
        COMPARISON_MODEL_FD_VS_SIESTA_FD
    )
    assert comparison_kind("autograd_deeph", "model_finite_difference") == (
        COMPARISON_MODEL_AUTOGRAD_VS_MODEL_FD
    )


def test_full_evidence_is_paper_ready():
    result = derive_claim_status(**FULL)
    assert result["claim_status"] == CLAIM_PAPER_READY
    assert result["reasons"] == []


def test_non_finite_is_invalid():
    result = derive_claim_status(**{**FULL, "values_finite": False})
    assert result["claim_status"] == CLAIM_INVALID


def test_missing_model_fd_gate_degrades_to_diagnostic():
    result = derive_claim_status(**{**FULL, "model_fd_gate_status": "not_run"})
    assert result["claim_status"] == CLAIM_DIAGNOSTIC_ONLY
    assert any("model_fd_gate" in reason for reason in result["reasons"])


def test_unproven_base_equivalence_caps_at_model_derivative():
    result = derive_claim_status(**{**FULL, "base_equivalence_status": "unproven"})
    assert result["claim_status"] == CLAIM_VALIDATED_MODEL_DERIVATIVE


def test_dirty_repo_caps_at_siesta_validated():
    result = derive_claim_status(**{**FULL, "reproducibility_status": "pinned_dirty"})
    assert result["claim_status"] == CLAIM_VALIDATED_AGAINST_SIESTA
    assert any("reproducibility" in reason for reason in result["reasons"])


def test_insufficient_seeds_blocks_paper_ready():
    result = derive_claim_status(**{**FULL, "n_seeds": 2})
    assert result["claim_status"] == CLAIM_VALIDATED_AGAINST_SIESTA


def test_unconverged_siesta_blocks_siesta_claims():
    result = derive_claim_status(**{**FULL, "siesta_converged": False})
    assert result["claim_status"] == CLAIM_VALIDATED_MODEL_DERIVATIVE
