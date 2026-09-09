"""C03b / E-F_001-S5: the algebra of the moving-PAO EPC memo, executed.

Every claim in ``docs/epc_formalismo_pao_movil.md`` that can be reduced to
linear algebra is checked here on a toy model where the *exact* answer is known
by construction, because the memo builds the finite basis explicitly:

    |phi_mu(t)> = column mu of V(t),   V(t) = V0 + t V1 + t^2/2 V2   (M x N)
    Hhat(t)     = Hop0 + t Hop1 + t^2/2 Hop2                        (M x M)

so that  S = V^dag V,  H = V^dag Hhat V,  and the exact operator perturbation
``PAO-covariant response = V0^dag Hop1 V0`` is available for comparison instead of being
assumed. ``M > N`` is a genuinely incomplete basis; ``M == N`` is a complete
one. Polynomials in ``t`` are used so the first derivatives are exact, not
finite differences.

No production module is imported: C04 has not written ``epc_formalism.py`` yet,
and GO-1 must be evaluable without it.
"""

from __future__ import annotations

import numpy as np
import pytest

M_EXACT = 9
N_BASIS = 5


def _toy(seed: int = 0, n_exact: int = M_EXACT, n_basis: int = N_BASIS) -> dict:
    """V0,V1,V2 (n_exact x n_basis) and Hermitian Hop0,Hop1,Hop2."""
    rng = np.random.default_rng(seed)

    def cplx(rows: int, cols: int) -> np.ndarray:
        return rng.normal(size=(rows, cols)) + 1j * rng.normal(size=(rows, cols))

    def herm(size: int) -> np.ndarray:
        a = cplx(size, size)
        return a + a.conj().T

    return {
        "V0": cplx(n_exact, n_basis),
        "V1": cplx(n_exact, n_basis),
        "V2": cplx(n_exact, n_basis),
        "Hop0": herm(n_exact),
        "Hop1": herm(n_exact),
        "Hop2": herm(n_exact),
    }


def _matrices(toy: dict) -> dict:
    """The observable matrix data at t=0 plus the objects behind it."""
    v0, v1, hop0, hop1 = toy["V0"], toy["V1"], toy["Hop0"], toy["Hop1"]
    s = v0.conj().T @ v0
    h = v0.conj().T @ hop0 @ v0
    s_left = v1.conj().T @ v0  # <d phi_mu | phi_nu>
    s_right = v0.conj().T @ v1  # <phi_mu | d phi_nu>
    return {
        "S": s,
        "H": h,
        "S_left": s_left,
        "S_right": s_right,
        "D_S": s_left + s_right,
        "D_H": v1.conj().T @ hop0 @ v0 + v0.conj().T @ hop1 @ v0 + v0.conj().T @ hop0 @ v1,
        "Delta_exact": v0.conj().T @ hop1 @ v0,
    }


def _pencil(h: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve H C = S C eps with C^dag S C = I, via a Cholesky factor of S."""
    chol = np.linalg.cholesky(s)
    inv = np.linalg.inv(chol)
    eps, u = np.linalg.eigh(inv @ h @ inv.conj().T)
    return eps, inv.conj().T @ u


def _model_at(toy: dict, t: float) -> tuple[np.ndarray, np.ndarray]:
    v = toy["V0"] + t * toy["V1"] + 0.5 * t * t * toy["V2"]
    hop = toy["Hop0"] + t * toy["Hop1"] + 0.5 * t * t * toy["Hop2"]
    return v.conj().T @ hop @ v, v.conj().T @ v


def _reconstructed(m: dict) -> np.ndarray:
    """Memo Eq. (F1): D_H - S_L S^-1 H - H S^-1 S_R."""
    inv_s = np.linalg.inv(m["S"])
    return m["D_H"] - m["S_left"] @ inv_s @ m["H"] - m["H"] @ inv_s @ m["S_right"]


def _g_exact_form(m: dict, eps: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Memo Eq. (F2): g_mn = (D_H)_mn - eps_n (S_L)_mn - eps_m (S_R)_mn."""
    d_h = c.conj().T @ m["D_H"] @ c
    s_l = c.conj().T @ m["S_left"] @ c
    s_r = c.conj().T @ m["S_right"] @ c
    return d_h - s_l * eps[None, :] - s_r * eps[:, None]


def _g_symmetric_form(m: dict, eps: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Memo Eq. (F3): the D_S-only truncation."""
    d_h = c.conj().T @ m["D_H"] @ c
    d_s = c.conj().T @ m["D_S"] @ c
    return d_h - 0.5 * (eps[:, None] + eps[None, :]) * d_s


def test_reconstruction_is_exact_for_a_complete_basis():
    """With M == N the projection identity is exact, so no Pulay term survives."""
    toy = _toy(seed=1, n_exact=N_BASIS, n_basis=N_BASIS)
    m = _matrices(toy)
    assert np.allclose(_reconstructed(m), m["Delta_exact"], atol=1e-10, rtol=0)


def test_incomplete_basis_residual_is_exactly_the_out_of_span_term():
    """The ONLY assumption is the span-restricted resolution of identity."""
    toy = _toy(seed=2)
    m = _matrices(toy)
    v0, v1, hop0 = toy["V0"], toy["V1"], toy["Hop0"]
    projector = v0 @ np.linalg.inv(m["S"]) @ v0.conj().T
    out_of_span = np.eye(v0.shape[0]) - projector
    pulay = v1.conj().T @ out_of_span @ hop0 @ v0 + v0.conj().T @ hop0 @ out_of_span @ v1

    residual = _reconstructed(m) - m["Delta_exact"]
    assert np.linalg.norm(pulay) > 1e-3  # the test would be vacuous otherwise
    assert np.allclose(residual, pulay, atol=1e-10, rtol=0)


def test_generalized_hellmann_feynman_diagonal():
    """d eps_n = C^dag (D_H - eps_n D_S) C holds for the INCOMPLETE basis too."""
    toy = _toy(seed=3)
    m = _matrices(toy)
    eps, c = _pencil(m["H"], m["S"])

    step = 1e-5
    plus = _pencil(*_model_at(toy, step))[0]
    minus = _pencil(*_model_at(toy, -step))[0]
    numeric = (plus - minus) / (2 * step)

    predicted = (np.diag(c.conj().T @ m["D_H"] @ c) - eps * np.diag(c.conj().T @ m["D_S"] @ c)).real
    assert np.allclose(numeric, predicted, atol=1e-8, rtol=0)

    # ... and the exact contraction reproduces it on the diagonal as well.
    assert np.allclose(np.diag(_g_exact_form(m, eps, c)).real, numeric, atol=1e-8, rtol=0)


def test_symmetric_form_differs_only_by_the_antisymmetric_overlap_response():
    """g_exact = g_symmetric + (eps_m - eps_n) * A_mn, A = (S_L - S_R)/2."""
    toy = _toy(seed=4)
    m = _matrices(toy)
    eps, c = _pencil(m["H"], m["S"])
    a_block = c.conj().T @ (0.5 * (m["S_left"] - m["S_right"])) @ c
    correction = (eps[:, None] - eps[None, :]) * a_block

    assert np.allclose(_g_exact_form(m, eps, c), _g_symmetric_form(m, eps, c) + correction,
                       atol=1e-10, rtol=0)
    # The correction is real, off-diagonal only, and not negligible.
    assert np.allclose(np.diag(correction), 0.0, atol=1e-12)
    assert np.linalg.norm(correction) > 1e-2 * np.linalg.norm(_g_exact_form(m, eps, c))


def test_D_H_and_D_S_alone_do_not_determine_g():
    """Two models with identical H, S, D_H, D_S and different PAO-projected coupling.

    Built as the memo says: move the basis by an extra in-span rotation
    ``V1 -> V1 + V0 X`` with ``S X = i K`` anti-hermitian (so D_S is untouched),
    then absorb the induced change of D_H into the operator derivative Hop1
    (unobservable from matrix data). Complete basis, so no Pulay term hides the
    effect.
    """
    toy = _toy(seed=5, n_exact=N_BASIS, n_basis=N_BASIS)
    m = _matrices(toy)
    s, h = m["S"], m["H"]

    k = np.diag(np.arange(1.0, N_BASIS + 1.0))  # hermitian, not commuting with eps
    x = np.linalg.solve(s, 1j * k)
    shift = x.conj().T @ h + h @ x  # hermitian

    other = dict(toy)
    other["V1"] = toy["V1"] + toy["V0"] @ x
    # Delta' = Delta - shift, i.e. Hop1' = V0^-dag (Delta - shift) V0^-1.
    inv_v0 = np.linalg.inv(toy["V0"])
    other["Hop1"] = inv_v0.conj().T @ (m["Delta_exact"] - shift) @ inv_v0
    other["Hop1"] = 0.5 * (other["Hop1"] + other["Hop1"].conj().T)
    n = _matrices(other)

    for key in ("S", "H", "D_S", "D_H"):
        assert np.allclose(m[key], n[key], atol=1e-8, rtol=0), key
    assert not np.allclose(m["S_left"], n["S_left"], atol=1e-6)

    eps, c = _pencil(h, s)
    g_a, g_b = _g_exact_form(m, eps, c), _g_exact_form(n, eps, c)
    # Same observable matrix data, different g: only the off-diagonal moves.
    assert np.allclose(np.diag(g_a), np.diag(g_b), atol=1e-8, rtol=0)
    assert np.linalg.norm(g_a - g_b) > 1e-2 * np.linalg.norm(g_a)


def test_energy_origin_shift_leaves_g_invariant_but_moves_raw_D_H():
    """H -> H + cS (a rigid E_F convention change) must not touch g."""
    toy = _toy(seed=6)
    m = _matrices(toy)
    eps, c = _pencil(m["H"], m["S"])
    shift = 0.37

    shifted = dict(m)
    shifted["H"] = m["H"] + shift * m["S"]
    shifted["D_H"] = m["D_H"] + shift * m["D_S"]
    eps_shifted, c_shifted = _pencil(shifted["H"], shifted["S"])
    assert np.allclose(eps_shifted, eps + shift, atol=1e-9, rtol=0)

    g_ref = _g_exact_form(m, eps, c)
    g_shift = _g_exact_form(shifted, eps_shifted, c_shifted)
    assert np.allclose(np.abs(g_ref), np.abs(g_shift), atol=1e-8, rtol=0)

    # Teeth: the naive "D_H is the coupling" answer is NOT invariant.
    naive_ref = c.conj().T @ m["D_H"] @ c
    naive_shift = c_shifted.conj().T @ shifted["D_H"] @ c_shifted
    assert not np.allclose(np.abs(naive_ref), np.abs(naive_shift), atol=1e-6)


def test_g_block_is_covariant_under_degenerate_subspace_rotation():
    """C -> C U inside a degenerate block rotates the g block, nothing else."""
    toy = _toy(seed=7)
    m = _matrices(toy)
    eps, c = _pencil(m["H"], m["S"])

    # Force an exact two-fold degeneracy by rebuilding H from a degenerate spectrum.
    eps = eps.copy()
    eps[1] = eps[0]
    inv_c = np.linalg.inv(c)
    m = dict(m)
    m["H"] = inv_c.conj().T @ np.diag(eps) @ inv_c
    m["H"] = 0.5 * (m["H"] + m["H"].conj().T)

    rng = np.random.default_rng(11)
    block = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    u_block = np.linalg.qr(block)[0]
    u = np.eye(N_BASIS, dtype=complex)
    u[:2, :2] = u_block
    c_rot = c @ u

    g_ref = _g_exact_form(m, eps, c)
    g_rot = _g_exact_form(m, eps, c_rot)
    assert np.allclose(g_rot, u.conj().T @ g_ref @ u, atol=1e-10, rtol=0)
    # The gauge-invariant summary of the degenerate block is unchanged.
    assert np.allclose(np.linalg.svd(g_rot[:2, :2], compute_uv=False),
                       np.linalg.svd(g_ref[:2, :2], compute_uv=False), atol=1e-10, rtol=0)


def test_intra_atomic_antisymmetric_response_is_invisible_to_D_S():
    """S_L + S_R = 0 on a same-atom block, so dS/dR cannot carry A there."""
    # One atom, two orbitals, rigidly transported: phi(t) = T(t) phi(0) with T
    # unitary, so S(t) is constant while <d phi_mu | phi_nu> is not zero.
    generator = np.array([[0.0, 1.0], [-1.0, 0.0]])  # real antisymmetric
    v0 = np.eye(2)
    v1 = generator @ v0
    s_left = v1.conj().T @ v0
    s_right = v0.conj().T @ v1
    assert np.allclose(s_left + s_right, 0.0, atol=1e-12)  # invisible to D_S
    assert np.linalg.norm(0.5 * (s_left - s_right)) > 0.5  # but A is not zero


def test_zero_point_amplitude_constant():
    """sqrt(hbar^2 / (2 M omega)) in Angstrom for M in amu and hbar*omega in eV."""
    hbar = 1.054571817e-34
    amu = 1.66053906660e-27
    ev = 1.602176634e-19
    constant = hbar * hbar / (amu * ev) * 1e20  # Angstrom^2
    assert constant == pytest.approx(4.1802e-3, rel=1e-3)
    carbon_optical = np.sqrt(constant / (2 * 12.011 * 0.196))
    assert carbon_optical == pytest.approx(0.0298, abs=5e-4)  # ~0.03 Ang, graphene E2g
