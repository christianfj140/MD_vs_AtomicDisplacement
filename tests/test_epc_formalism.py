"""C04 / E-F_001-S6: the moving-PAO formalism contract, exercised through its API.

The algebra itself was established in ``test_epc_moving_pao_formalism.py`` (C03b)
on a toy where the exact answer is known by construction: an explicit finite
basis ``V(t)`` and an explicit operator ``Hhat(t)``, so ``PAO-covariant response`` is
available for comparison instead of assumed. Those fixtures are reused here on
purpose — the production module must reproduce the same numbers on the same
model, not on a friendlier one.

What this file adds is the *contract*: which formalism_id comes out, what the
API refuses to do, and which quantities survive a gauge rotation. Everything is
float64/complex128.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))

from test_epc_moving_pao_formalism import (  # noqa: E402
    N_BASIS,
    _matrices,
    _model_at,
    _pencil,
    _toy,
)

from epc_formalism import (  # noqa: E402
    DEGRADED_FORMALISM_IDS,
    FORMALISM_ID,
    FORMALISM_ID_IGNORE_OVERLAP,
    FORMALISM_ID_SYMMETRIC,
    REPRESENTATION_D_S,
    REPRESENTATION_S_L_S_R,
    EpcFormalismError,
    basis_response_D_S,
    basis_response_S_L_S_R,
    check_overlap_response_consistency,
    epc_matrix_elements,
    formalism_contract,
    gauge_invariant_block_metrics,
    pao_covariant_response,
    zero_point_amplitude_ang,
)

TOL = 1e-10


def _response(m: dict, *, intra: bool = True):
    return basis_response_S_L_S_R(
        m["S_left"], m["S_right"], intra_atomic_included=intra, backend="toy"
    )


def _g(m: dict, eps: np.ndarray, c: np.ndarray):
    return epc_matrix_elements(
        m["D_H"], _response(m), C_row=c, C_col=c, eps_row=eps, eps_col=eps
    )


def _with_spectrum(m: dict, c: np.ndarray, new_eps: np.ndarray) -> dict:
    """Rebuild H so the pencil (H, S) really has spectrum ``new_eps`` with the same C."""
    inv_c = np.linalg.inv(c)
    h = inv_c.conj().T @ np.diag(new_eps) @ inv_c
    out = dict(m)
    out["H"] = 0.5 * (h + h.conj().T)
    return out


def test_formalism_id_is_the_one_the_memo_closed():
    memo = (REPO_ROOT / "docs" / "epc_formalismo_pao_movil.md").read_text(encoding="utf-8")
    assert FORMALISM_ID in memo
    assert set(DEGRADED_FORMALISM_IDS) == {FORMALISM_ID_SYMMETRIC, FORMALISM_ID_IGNORE_OVERLAP}
    contract = formalism_contract()
    assert contract["basis_response_representation"] == REPRESENTATION_S_L_S_R
    assert contract["units"]["g"] == "eV/Ang"


def test_pao_covariant_response_reproduces_the_operator_derivative_for_a_complete_basis():
    """(F1) is exact when the PAO span is the whole space: no Pulay term left."""
    toy = _toy(seed=1, n_exact=N_BASIS, n_basis=N_BASIS)
    m = _matrices(toy)
    delta = pao_covariant_response(m["D_H"], m["H"], m["S"], _response(m))
    assert delta.dtype == np.complex128
    assert np.allclose(delta, m["Delta_exact"], atol=TOL, rtol=0)


def test_contraction_equals_pao_covariant_response_without_touching_S():
    """(F2) is (F1) contracted; the production path never inverts S."""
    toy = _toy(seed=2)
    m = _matrices(toy)
    eps, c = _pencil(m["H"], m["S"])
    delta = pao_covariant_response(m["D_H"], m["H"], m["S"], _response(m))
    block = _g(m, eps, c)
    assert block.formalism_id == FORMALISM_ID
    assert block.intra_atomic_included is True
    assert np.allclose(block.values, c.conj().T @ delta @ c, atol=TOL, rtol=0)


def test_generalized_hellmann_feynman_diagonal():
    """diag g = d eps_n / du, checked against finite differences of the pencil."""
    toy = _toy(seed=3)
    m = _matrices(toy)
    eps, c = _pencil(m["H"], m["S"])

    step = 1e-5
    numeric = (_pencil(*_model_at(toy, step))[0] - _pencil(*_model_at(toy, -step))[0]) / (2 * step)

    diagonal = np.diag(_g(m, eps, c).values)
    assert np.allclose(diagonal.imag, 0.0, atol=TOL)
    assert np.allclose(diagonal.real, numeric, atol=1e-8, rtol=0)


def test_g_is_invariant_under_a_non_orthogonal_change_of_basis():
    """phi -> phi T with T constant (memo section 7): every raw matrix moves, g does not."""
    toy = _toy(seed=4)
    m = _matrices(toy)
    eps, c = _pencil(m["H"], m["S"])
    reference = _g(m, eps, c).values

    rng = np.random.default_rng(23)
    t = rng.normal(size=(N_BASIS, N_BASIS)) + 1j * rng.normal(size=(N_BASIS, N_BASIS))
    t += 3.0 * np.eye(N_BASIS)  # well conditioned, decidedly not unitary
    assert abs(np.linalg.norm(t.conj().T @ t - np.eye(N_BASIS))) > 1.0

    rotated = {key: t.conj().T @ m[key] @ t for key in ("H", "S", "D_H", "S_left", "S_right")}
    c_rot = np.linalg.solve(t, c)
    assert np.allclose(c_rot.conj().T @ rotated["S"] @ c_rot, np.eye(N_BASIS), atol=1e-9)

    assert np.allclose(_g(rotated, eps, c_rot).values, reference, atol=TOL, rtol=0)


def test_degenerate_block_metrics_are_gauge_invariant():
    """C -> C U inside an exactly degenerate block: g -> U^dag g U, metrics fixed."""
    toy = _toy(seed=5)
    m = _matrices(toy)
    eps, c = _pencil(m["H"], m["S"])
    eps = eps.copy()
    eps[1] = eps[0]  # exact two-fold degeneracy
    m = _with_spectrum(m, c, eps)

    rng = np.random.default_rng(7)
    seed_block = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    u = np.eye(N_BASIS, dtype=complex)
    u[:2, :2] = np.linalg.qr(seed_block)[0]

    reference = _g(m, eps, c).values
    rotated = _g(m, eps, c @ u).values
    assert np.allclose(rotated, u.conj().T @ reference @ u, atol=TOL, rtol=0)

    before = gauge_invariant_block_metrics(reference[:2, :2])
    after = gauge_invariant_block_metrics(rotated[:2, :2])
    assert np.allclose(before["singular_values"], after["singular_values"], atol=TOL, rtol=0)
    assert before["frobenius_norm"] == pytest.approx(after["frobenius_norm"], abs=TOL)
    # Teeth: the band-indexed elements themselves are pure gauge.
    assert not np.allclose(np.abs(reference[:2, :2]), np.abs(rotated[:2, :2]), atol=1e-6)


def test_symmetric_truncation_error_vanishes_with_the_energy_split():
    """The D_S-only form is wrong by (eps_m - eps_n) A: harmless only near degeneracy."""
    toy = _toy(seed=6)
    m = _matrices(toy)
    eps0, c = _pencil(m["H"], m["S"])
    response = _response(m)
    degraded = basis_response_D_S(response.total, backend="toy")

    errors = []
    for split in (1e-1, 1e-3, 1e-5):
        eps = eps0.copy()
        eps[1] = eps[0] + split
        m_split = _with_spectrum(m, c, eps)
        exact = epc_matrix_elements(
            m_split["D_H"], response, C_row=c, C_col=c, eps_row=eps, eps_col=eps
        )
        symmetric = epc_matrix_elements(
            m_split["D_H"], degraded, C_row=c, C_col=c, eps_row=eps, eps_col=eps
        )
        assert symmetric.formalism_id == FORMALISM_ID_SYMMETRIC
        assert symmetric.formalism_id in DEGRADED_FORMALISM_IDS
        # Exact on the diagonal at any split, and inside the near-degenerate block.
        assert np.allclose(np.diag(exact.values), np.diag(symmetric.values), atol=TOL, rtol=0)
        errors.append(abs(exact.values[0, 1] - symmetric.values[0, 1]))

    assert errors[0] > 1e-3  # not a vacuous test at a physical splitting
    assert errors[0] / errors[1] == pytest.approx(100.0, rel=1e-6)
    assert errors[1] / errors[2] == pytest.approx(100.0, rel=1e-6)


def test_a_sum_only_response_cannot_be_split_back_into_S_L_and_S_R():
    """D_S fixes the hermitian part of S*Gamma; A is independent information."""
    toy = _toy(seed=7)
    m = _matrices(toy)
    degraded = basis_response_D_S(m["D_S"], backend="toy")
    assert degraded.representation == REPRESENTATION_D_S
    assert degraded.S_left is None and degraded.S_right is None
    assert degraded.intra_atomic_included is False

    with pytest.raises(EpcFormalismError, match="not recoverable"):
        _ = degraded.antisymmetric
    with pytest.raises(EpcFormalismError, match="separately"):
        pao_covariant_response(m["D_H"], m["H"], m["S"], degraded)
    with pytest.raises(EpcFormalismError, match="no S_L/S_R split"):
        check_overlap_response_consistency(degraded, m["D_S"])


def test_consistency_check_ties_S_L_S_R_to_D_S_without_determining_them():
    """S_L + S_R = D_S and S_L = S_R^dag pass; a different pair passes just as well."""
    toy = _toy(seed=8)
    m = _matrices(toy)
    report = check_overlap_response_consistency(_response(m), m["D_S"])
    assert report["passes"]
    assert report["sum_residual"] < TOL * np.linalg.norm(m["D_S"])
    assert report["antisymmetric_norm"] > 1e-3

    # Same D_S, same hermiticity identity, different A -> the check cannot tell.
    shift = 1j * np.diag(np.arange(1.0, N_BASIS + 1.0))  # anti-hermitian
    other = basis_response_S_L_S_R(
        m["S_left"] + shift,
        m["S_right"] - shift,
        intra_atomic_included=True,
        backend="toy",
    )
    other_report = check_overlap_response_consistency(other, m["D_S"])
    assert other_report["passes"]
    assert other_report["antisymmetric_norm"] != pytest.approx(report["antisymmetric_norm"])

    eps, c = _pencil(m["H"], m["S"])
    g_ref = _g(m, eps, c).values
    g_other = epc_matrix_elements(
        m["D_H"], other, C_row=c, C_col=c, eps_row=eps, eps_col=eps
    ).values
    assert np.linalg.norm(g_ref - g_other) > 1e-2 * np.linalg.norm(g_ref)
    assert np.allclose(np.diag(g_ref), np.diag(g_other), atol=TOL, rtol=0)  # diagonal is safe


def test_D_H_cannot_be_labelled_g():
    """No API path returns C^dag D_H C as the production coupling."""
    toy = _toy(seed=9)
    m = _matrices(toy)
    eps, c = _pencil(m["H"], m["S"])

    with pytest.raises(EpcFormalismError, match="not g"):
        epc_matrix_elements(m["D_H"], None, C_row=c, C_col=c, eps_row=eps, eps_col=eps)

    control = epc_matrix_elements(
        m["D_H"], None, C_row=c, C_col=c, eps_row=eps, eps_col=eps, control_negative=True
    )
    assert control.formalism_id == FORMALISM_ID_IGNORE_OVERLAP
    assert control.is_degraded
    assert not _g(m, eps, c).is_degraded
    assert np.linalg.norm(control.values - _g(m, eps, c).values) > 1e-3


def test_missing_intra_atomic_term_is_declared_not_silent():
    """A response built from dS/dR alone must carry the flag into the block."""
    toy = _toy(seed=10)
    m = _matrices(toy)
    eps, c = _pencil(m["H"], m["S"])
    block = epc_matrix_elements(
        m["D_H"],
        _response(m, intra=False),
        C_row=c,
        C_col=c,
        eps_row=eps,
        eps_col=eps,
    )
    assert block.formalism_id == FORMALISM_ID
    assert block.intra_atomic_included is False
    assert block.basis_response_backend == "toy"


def test_zero_point_amplitude_matches_the_memo():
    assert zero_point_amplitude_ang(12.011, 0.196) == pytest.approx(0.0298, abs=5e-4)
    with pytest.raises(EpcFormalismError):
        zero_point_amplitude_ang(12.011, 0.0)
