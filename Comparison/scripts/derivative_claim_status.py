#!/usr/bin/env python3
"""Derivative claim ladder (audit Fase 11).

Three distinct comparisons, never merged into one metric:

- A ``model_autograd_vs_model_fd``: mathematical consistency of the model's
  own derivative (no SIESTA involved).
- B ``model_fd_vs_siesta_fd``: model error measured with the same numerical
  method on both sides.
- C ``model_autograd_vs_siesta_fd``: the final result.

Claim ladder (each level requires everything below it):

    invalid < diagnostic_only < validated_model_derivative
            < validated_against_siesta < paper_ready
"""

from __future__ import annotations

from typing import Any

CLAIM_INVALID = "invalid"
CLAIM_DIAGNOSTIC_ONLY = "diagnostic_only"
CLAIM_VALIDATED_MODEL_DERIVATIVE = "validated_model_derivative"
CLAIM_VALIDATED_AGAINST_SIESTA = "validated_against_siesta"
CLAIM_PAPER_READY = "paper_ready"

CLAIM_LADDER = (
    CLAIM_INVALID,
    CLAIM_DIAGNOSTIC_ONLY,
    CLAIM_VALIDATED_MODEL_DERIVATIVE,
    CLAIM_VALIDATED_AGAINST_SIESTA,
    CLAIM_PAPER_READY,
)

COMPARISON_MODEL_AUTOGRAD_VS_MODEL_FD = "model_autograd_vs_model_fd"
COMPARISON_MODEL_FD_VS_SIESTA_FD = "model_fd_vs_siesta_fd"
COMPARISON_MODEL_AUTOGRAD_VS_SIESTA_FD = "model_autograd_vs_siesta_fd"


def comparison_kind(
    predicted_derivative_method: str | None,
    reference_derivative_method: str | None,
) -> str | None:
    """Which of the three protocol comparisons a metrics manifest encodes."""
    predicted = str(predicted_derivative_method or "").lower()
    reference = str(reference_derivative_method or "").lower()
    predicted_autograd = "autograd" in predicted
    if "siesta" in reference:
        return (
            COMPARISON_MODEL_AUTOGRAD_VS_SIESTA_FD
            if predicted_autograd
            else COMPARISON_MODEL_FD_VS_SIESTA_FD
        )
    if reference and predicted:
        return COMPARISON_MODEL_AUTOGRAD_VS_MODEL_FD if predicted_autograd else None
    return None


def derive_claim_status(
    *,
    comparison_valid: bool,
    values_finite: bool,
    model_fd_gate_status: str | None,
    base_equivalence_status: str | None,
    siesta_converged: bool = True,
    reproducibility_status: str | None = None,
    fixed_test: bool = False,
    n_seeds: int = 0,
    blocking_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate the claim ladder; claims degrade automatically on missing evidence.

    - ``validated_model_derivative`` needs a passing model-autograd-vs-model-FD
      gate (comparison A).
    - ``validated_against_siesta`` additionally needs proven basis equivalence,
      converged SIESTA references and a valid comparison.
    - ``paper_ready`` additionally needs pinned_clean repositories, a frozen
      test, >= 3 seeds and no blocking warnings.
    """
    blocking_warnings = list(blocking_warnings or [])
    reasons: list[str] = []

    if not comparison_valid or not values_finite:
        if not comparison_valid:
            reasons.append("comparison_not_run_or_failed")
        if not values_finite:
            reasons.append("non_finite_values")
        return {"claim_status": CLAIM_INVALID, "reasons": reasons}

    status = CLAIM_DIAGNOSTIC_ONLY
    if model_fd_gate_status == "pass":
        status = CLAIM_VALIDATED_MODEL_DERIVATIVE
    else:
        reasons.append(f"model_fd_gate_status={model_fd_gate_status or 'not_run'}")

    if status == CLAIM_VALIDATED_MODEL_DERIVATIVE:
        if base_equivalence_status == "proven" and siesta_converged:
            status = CLAIM_VALIDATED_AGAINST_SIESTA
        else:
            if base_equivalence_status != "proven":
                reasons.append(f"base_equivalence_status={base_equivalence_status or 'unavailable'}")
            if not siesta_converged:
                reasons.append("siesta_not_converged")

    if status == CLAIM_VALIDATED_AGAINST_SIESTA:
        paper_blockers = []
        if reproducibility_status != "pinned_clean":
            paper_blockers.append(f"reproducibility_status={reproducibility_status or 'unavailable'}")
        if not fixed_test:
            paper_blockers.append("no_fixed_test")
        if n_seeds < 3:
            paper_blockers.append(f"n_seeds={n_seeds}<3")
        if blocking_warnings:
            paper_blockers.append(f"blocking_warnings={blocking_warnings}")
        if paper_blockers:
            reasons.extend(paper_blockers)
        else:
            status = CLAIM_PAPER_READY

    return {"claim_status": status, "reasons": reasons}
