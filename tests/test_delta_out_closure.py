from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Comparison/scripts"), str(ROOT / "shared")]


def test_delta_out_vanishes_when_observation_and_full_basis_are_identical():
    from certify_delta_out_closure import delta_out

    s = np.array([[1.2, 0.1], [0.1, 1.0]])
    k = np.array([[2.0, 0.3], [0.3, -1.0]])
    derivative = np.array([[0.4, 0.2], [0.2, -0.1]])
    left = np.array([[0.0, 0.1], [-0.1, 0.0]])
    right = left.T
    out, delta_a, g_a, residual = delta_out(
        derivative, k, s, left, right, np.eye(2), s, left, right)
    assert np.allclose(out, 0.0)
    assert np.allclose(delta_a, g_a)
    assert residual < 1e-14
