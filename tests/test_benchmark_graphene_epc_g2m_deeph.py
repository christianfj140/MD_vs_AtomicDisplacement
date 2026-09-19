from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))

from benchmark_graphene_epc_g2m_deeph import derivative_metrics, matrix_metrics


def test_benchmark_metrics_have_the_declared_normalization() -> None:
    reference = np.eye(2)
    candidate = 2 * reference
    derivative = derivative_metrics(candidate, reference)
    assert np.isclose(derivative["D_MAE_eV_per_Ang"], 0.5)
    assert np.isclose(derivative["D_relative_error"], 1.0)
    assert np.isclose(derivative["D_cosine"], 1.0)
    assert np.isclose(derivative["D_best_fit_scale_diagnostic"], 0.5)
    metrics = matrix_metrics(candidate, reference)
    assert np.isclose(metrics["G_relative_error"], 1.0)
    assert np.isclose(metrics["G_norm_ratio"], 2.0)
    assert np.isclose(metrics["W_over_reference"], 4.0)
