from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Comparison/scripts"), str(ROOT / "shared")]

from certify_full_basis_connection_onlys_v2 import CASES  # noqa: E402


def test_v2_covers_exactly_the_existing_delta_out_basis_cases():
    assert CASES == (("n_tzp_d12_14", "C1_x"), ("n_tzp_d12_14", "C1_z"),
                     ("n_qzp_d12_14", "C1_x"), ("n_qzp_d12_14", "C1_z"),
                     ("n_qzp_d10_12", "C1_x"))


def test_v3_cross_derivative_uses_the_five_point_weights():
    from certify_full_basis_connection_onlys_v3 import cross_derivative
    from run_displaced_projection_sentinel import H_ANG

    class Overlap:
        def __init__(self, value): self.value = value
        def Sk(self, k, format): return np.full((2, 2), self.value)

    overlaps = {("C1_x", step): Overlap(step * H_ANG) for step in (-2, -1, 1, 2)}
    right, left = cross_derivative(overlaps, "C1_x", 1, (0, 0, 0))
    assert np.allclose(right, 1.0)
    assert np.allclose(left, 1.0)


def test_v4_enforces_the_exact_adjoint_identity():
    from certify_full_basis_connection_semantic_v4 import enforce_adjoint

    right = np.array([[1j, 2 + 3j], [4, -2j]])
    left = np.array([[-1.1j, 4.2], [2 - 2.8j, 2.1j]])
    corrected_right, corrected_left = enforce_adjoint(right, left)
    assert np.array_equal(corrected_left, corrected_right.conj().T)


def test_v5_backward_error_is_scale_stable():
    from certify_full_basis_connection_semantic_v5 import solve

    matrix = np.array([[2.0, 0.2], [0.2, 1.0]])
    rhs = 1e-14 * np.eye(2)
    solution, residual = solve(matrix, rhs)
    assert np.allclose(matrix @ solution, rhs)
    assert residual < 1e-14
