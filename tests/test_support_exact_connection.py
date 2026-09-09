from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))
from certify_support_exact_connection import (  # noqa: E402
    SOLVE_RESIDUAL_TOL,
    SUPPORT_LIMIT_EV_PER_ANG,
    _hermitian_solve,
    _propagate,
    _reconstruct,
    _support_blocks,
)


def _synthetic(no=7, ni=3, seed=0):
    """S_L/S_R with the exact moving-PAO support, plus an anti-Hermitian B_I."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(no)
    mask[:ni] = 1.0
    off = rng.normal(size=(no, no)) + 1j * rng.normal(size=(no, no))
    block = rng.normal(size=(ni, ni)) + 1j * rng.normal(size=(ni, ni))
    b_i = np.zeros((no, no), dtype=complex)
    b_i[:ni, :ni] = block - block.conj().T  # anti-Hermitian, as required
    # Only columns in I move for S_R; only rows in I move for S_L.
    right = (1.0 - mask)[:, None] * off * mask[None, :] + b_i
    left = mask[:, None] * (-off.conj().T) * (1.0 - mask)[None, :] - b_i
    return mask, left, right, b_i


def test_reconstruction_recovers_S_L_S_R_given_B_I():
    mask, left, right, b_i = _synthetic()
    d_s = left + right
    rec_left, rec_right = _reconstruct(d_s, mask)
    assert np.allclose(rec_right + b_i, right, atol=1e-12)
    assert np.allclose(rec_left - b_i, left, atol=1e-12)
    assert np.allclose(rec_left + b_i + rec_right - b_i, d_s, atol=1e-12)


def test_support_structure_blocks_vanish():
    mask, left, right, _ = _synthetic()
    out_out, in_in = _support_blocks(left + right, mask)
    assert np.linalg.norm(out_out) < 1e-12  # outside/outside is identically zero
    assert np.linalg.norm(in_in) < 1e-12  # B_I cancels between S_L and S_R in D_S


def test_B_I_is_antihermitian_and_not_recoverable_from_D_S():
    mask, left, right, b_i = _synthetic()
    assert np.allclose(b_i, -b_i.conj().T, atol=1e-12)
    # D_S carries no trace of B_I: the same D_S arises for any B_I.
    other = np.zeros_like(b_i)
    other[:3, :3] = np.diag([1j, -2j, 3j])
    assert np.allclose((left - other) + (right + other), left + right, atol=1e-12)


def test_B_I_zero_closure_is_insensitive_to_B_I():
    """The B_I=0 closure diagnostic cannot detect B_I -- it must not read as a pass."""
    mask, left, right, b_i = _synthetic()
    d_s = left + right
    rec_left, rec_right = _reconstruct(d_s, mask)
    closure = d_s - rec_left - rec_right
    # Closure is exactly the J.D.J block, which vanishes whatever B_I is.
    assert np.linalg.norm(closure) < 1e-12
    assert np.linalg.norm(b_i) > 1.0  # yet B_I is large and simply invisible here


def test_solve_residual_recorded_and_below_tolerance():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    overlap = a @ a.conj().T + 9 * np.eye(9)  # positive definite
    rhs = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    solution, residual = _hermitian_solve(overlap, rhs)
    assert residual < SOLVE_RESIDUAL_TOL
    assert np.allclose(overlap @ solution, rhs, atol=1e-9)


def test_propagate_returns_error_and_solve_residual():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    overlap = a @ a.conj().T + 6 * np.eye(6)
    k0 = rng.normal(size=(6, 6))
    k0 = k0 + k0.T
    zero = np.zeros((6, 6))
    error, residual = _propagate(zero, zero, overlap, k0)
    assert error == 0.0  # no perturbation -> no propagated error
    assert residual < SOLVE_RESIDUAL_TOL
    assert SUPPORT_LIMIT_EV_PER_ANG == 2.5e-4  # 0.25 meV/Ang in eV/Ang
