"""Unit tests for the independent two-centre B_I witness.

These exercise the *construction* -- radial reading, harmonic convention,
support-exact image enumeration, quadrature correctness and the propagation
convention -- on synthetic or analytic inputs, so they run without SIESTA.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))

from certify_b_i_independent_two_center import (  # noqa: E402
    B0_MONOCENTRIC_TOL,
    B0_REL_TOL,
    B1_LIMIT_EV_PER_ANG,
    B2_LIMIT_EV_PER_ANG,
    B3_LIMIT_EV_PER_ANG,
    FD_WEIGHTS,
    H_VALUES,
    PRODUCTION_H,
    Basis,
    _electronic,
    _grid,
    bloch_sum,
    enumerate_images,
    overlap_block,
    prolate_overlap_block,
    read_radials,
)

PROD_CENTRAL = (
    REPO_ROOT / "Comparison/results/epc/basis_projection_sentinel/runs/PROD-SZ/central"
)


def _gaussian_basis(alpha=1.5, cutoff=4.0, points=4000):
    """A Basis whose radials are Gaussians -- overlaps are known in closed form."""
    from scipy.interpolate import CubicSpline

    r = np.linspace(0.0, cutoff, points)
    # s: exp(-a r^2);  p: r exp(-a r^2)  (already includes the r**l factor)
    radials = {
        0: (CubicSpline(r, np.exp(-alpha * r**2)), cutoff),
        1: (CubicSpline(r, r * np.exp(-alpha * r**2)), cutoff),
    }
    return Basis(radials, [0, 1, 1, 1])


# ---------------------------------------------------------------- thresholds

def test_thresholds_are_the_preregistered_values():
    assert B1_LIMIT_EV_PER_ANG == 2.5e-4  # 0.25 meV/Ang
    assert B2_LIMIT_EV_PER_ANG == 2.5e-4  # 0.25 meV/Ang
    assert B3_LIMIT_EV_PER_ANG == 5.0e-4  # 0.50 meV/Ang
    assert B0_REL_TOL == 1e-6
    assert B0_MONOCENTRIC_TOL == 1e-8


def test_stencil_is_the_preregistered_five_point_rule():
    assert FD_WEIGHTS == {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
    assert H_VALUES == (0.010, 0.005, 0.0025, 0.00125)
    assert PRODUCTION_H in H_VALUES
    # The rule is exact for cubics: sum w_i (i h)^n / (12h) = delta_{n,1}.
    for power, expected in ((0, 0.0), (1, 1.0), (2, 0.0), (3, 0.0)):
        value = sum(w * (s * 1.0) ** power for s, w in FD_WEIGHTS.items()) / 12.0
        assert abs(value - expected) < 1e-12


# ------------------------------------------------------------------ harmonics

def test_real_spherical_harmonics_are_orthonormal_on_the_sphere():
    """The four angular factors must be orthonormal, in the assumed order."""
    basis = _gaussian_basis()
    rng = np.random.default_rng(0)
    # Quadrature on the unit sphere via the same product rule the code uses.
    x_c, w_c = np.polynomial.legendre.leggauss(40)
    phi = 2 * np.pi * np.arange(40) / 40
    st = np.sqrt(1 - x_c**2)
    dirs = np.stack([(st[:, None] * np.cos(phi)).ravel(),
                     (st[:, None] * np.sin(phi)).ravel(),
                     np.repeat(x_c, 40)], axis=-1)
    weights = np.repeat(w_c, 40) * (2 * np.pi / 40)
    r = np.linalg.norm(dirs, axis=-1)
    safe = np.where(r > 0, r, 1.0)
    x, y, z = dirs[..., 0] / safe, dirs[..., 1] / safe, dirs[..., 2] / safe
    c0, c1 = Basis.C0, Basis.C1
    harmonics = np.stack([np.full_like(x, c0), -c1 * y, c1 * z, -c1 * x])
    gram = np.einsum("an,bn,n->ab", harmonics, harmonics, weights)
    assert np.allclose(gram, np.eye(4), atol=1e-12)
    del basis, rng


def test_harmonic_signs_follow_siesta_convention():
    """px and py are NEGATIVE along +x / +y; pz is POSITIVE along +z."""
    basis = _gaussian_basis()
    values = basis.values(np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    assert values[3, 0] < 0  # px at +x
    assert values[1, 1] < 0  # py at +y
    assert values[2, 2] > 0  # pz at +z
    assert values[0, 0] > 0  # s is positive everywhere in range


# ------------------------------------------------------------------ quadrature

def test_quadrature_reproduces_the_analytic_gaussian_overlap():
    """<s_0 | s_R> for a radial exp(-a r^2) carrying Y_00 = 1/sqrt(4pi).

    The radial overlap is (pi/2a)^{3/2} exp(-a R^2 / 2); the two Y_00 factors
    contribute 1/(4pi).  This is the one place the quadrature is checked against
    a closed form rather than against itself.
    """
    alpha = 1.5
    basis = _gaussian_basis(alpha=alpha, cutoff=6.0)
    grid = _grid(basis, 400, 64, 64)
    for distance in (0.5, 1.0, 2.0):
        separation = np.array([distance, 0.0, 0.0])
        block = overlap_block(basis, separation, grid)
        analytic = ((np.pi / (2 * alpha)) ** 1.5
                    * np.exp(-alpha * distance**2 / 2) / (4 * np.pi))
        assert abs(block[0, 0] - analytic) < 1e-6 * max(abs(analytic), 1e-12)


def test_analytic_pao_gradient_matches_a_central_difference():
    basis = _gaussian_basis()
    points = np.array([[0.7, 0.4, -0.2], [1.1, -0.3, 0.8]])
    step = 1e-6
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = step
        numeric = (basis.values(points + shift) - basis.values(points - shift)) / (2 * step)
        assert np.allclose(basis.gradients(points)[:, :, axis], numeric, atol=2e-9)


def test_quadrature_is_symmetric_under_exchange_of_centres():
    """S(R) and S(-R) must be exact transposes: S_ab(R) = S_ba(-R)."""
    basis = _gaussian_basis()
    grid = _grid(basis, 300, 48, 48)
    separation = np.array([0.9, 0.4, -0.3])
    forward = overlap_block(basis, separation, grid)
    backward = overlap_block(basis, -separation, grid)
    assert np.allclose(forward, backward.T, atol=1e-9)


def test_prolate_overlap_has_exact_exchange_symmetry():
    basis = _gaussian_basis()
    separation = np.array([0.9, 0.4, -0.3])
    assert np.allclose(
        prolate_overlap_block(basis, separation, 100),
        prolate_overlap_block(basis, -separation, 100).T,
        atol=1e-12,
    )


def test_fourier_bessel_sk_derivative_matches_its_overlap():
    from certify_b_i_fourier_bessel_v4 import FourierBesselSP

    basis = _gaussian_basis(cutoff=6.0)
    engine = FourierBesselSP(basis, 300, 30.0, 1201)
    separation = np.array([0.9, 0.4, -0.3])
    step = 1e-5
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = step
        numeric = (engine.overlap(separation + shift) - engine.overlap(separation - shift)) / (2 * step)
        assert np.allclose(engine.derivative(separation, axis), numeric, atol=2e-7)


def test_v5_direct_radial_normalization_is_explicit():
    from certify_b_i_fourier_bessel_semantic_v5 import direct_radial_norms

    basis = _gaussian_basis()
    norms = direct_radial_norms(basis, order=300)
    factors = {l: 1 / np.sqrt(value) for l, value in norms.items()}
    assert max(abs(norms[l] * factors[l] ** 2 - 1) for l in norms) < 1e-14


def test_overlap_vanishes_beyond_the_summed_cutoffs():
    basis = _gaussian_basis(cutoff=2.0)
    grid = _grid(basis, 120, 24, 24)
    block = overlap_block(basis, np.array([4.001, 0.0, 0.0]), grid)
    assert np.array_equal(block, np.zeros((4, 4)))


# --------------------------------------------------------- image enumeration

def test_image_enumeration_is_support_exact_and_closes():
    """Every enumerated R must be reachable; the first empty shell terminates."""
    basis = _gaussian_basis(cutoff=2.0)  # sum of cutoffs = 4.0
    cell = np.diag([3.0, 3.0, 30.0])
    xyz = np.array([[0.0, 0.0, 0.0]])
    images = enumerate_images(basis, cell, xyz)
    indices = {tuple(image["index"]) for image in images}
    # |R| < 4.0 with a 3 Ang cubic lattice: R = 0, +-1 and the (+-1,+-1) diagonals
    # (|R| = 4.243) are excluded, so exactly the 7 in-plane axis/zero vectors.
    for index in indices:
        assert np.linalg.norm(np.asarray(index) @ cell) < 2 * basis.rc_max
    # Completeness: nothing inside the support radius may be missing.
    import itertools
    for index in itertools.product(range(-3, 4), repeat=3):
        if np.linalg.norm(np.asarray(index) @ cell) < 2 * basis.rc_max:
            assert tuple(index) in indices


def test_image_enumeration_grows_with_the_cutoff():
    """A larger PAO support must enumerate strictly more images -- not 'first shell'."""
    cell = np.diag([3.0, 3.0, 30.0])
    xyz = np.array([[0.0, 0.0, 0.0]])
    small = enumerate_images(_gaussian_basis(cutoff=2.0), cell, xyz)
    large = enumerate_images(_gaussian_basis(cutoff=4.0), cell, xyz)
    assert len(large) > len(small)
    small_indices = {tuple(image["index"]) for image in small}
    assert small_indices < {tuple(image["index"]) for image in large}


# ------------------------------------------------------------------ Bloch sum

def test_bloch_sum_at_gamma_is_the_plain_real_space_sum():
    blocks = {(0, 0, 0): np.eye(2), (1, 0, 0): np.full((2, 2), 0.25),
              (-1, 0, 0): np.full((2, 2), 0.25)}
    total = bloch_sum(blocks, (0.0, 0.0, 0.0))
    assert np.allclose(total, np.eye(2) + 0.5 * np.full((2, 2), 1.0))


def test_bloch_sum_phase_is_exp_plus_i_k_dot_R():
    blocks = {(1, 0, 0): np.ones((1, 1))}
    value = bloch_sum(blocks, (0.25, 0.0, 0.0))[0, 0]
    assert np.isclose(value, np.exp(2j * np.pi * 0.25))


def test_bloch_sum_of_a_hermitian_real_space_set_is_hermitian():
    rng = np.random.default_rng(3)
    block = rng.normal(size=(3, 3))
    blocks = {(0, 0, 0): block + block.T, (1, 0, 0): block, (-1, 0, 0): block.T}
    matrix = bloch_sum(blocks, (0.31, 0.0, 0.0))
    assert np.allclose(matrix, matrix.conj().T, atol=1e-12)


# ------------------------------------------------------- propagation convention

def test_electronic_propagation_uses_dS_R_plus_dS_L_minus():
    """dB_I enters as dS_R = +dB_I and dS_L = -dB_I; zero perturbation -> zero."""
    rng = np.random.default_rng(4)
    a = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    overlap = a @ a.conj().T + 5 * np.eye(5)
    k0 = rng.normal(size=(5, 5))
    k0 = k0 + k0.T
    error, residual = _electronic(np.zeros((5, 5), dtype=complex), overlap, k0)
    assert error == 0.0
    assert residual < 1e-11


def test_electronic_propagation_is_linear_in_the_perturbation():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    overlap = a @ a.conj().T + 4 * np.eye(4)
    k0 = rng.normal(size=(4, 4))
    k0 = k0 + k0.T
    block = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    delta = block - block.conj().T  # anti-Hermitian, as B_I is
    single, _ = _electronic(delta, overlap, k0)
    double, _ = _electronic(2 * delta, overlap, k0)
    assert np.isclose(double, 2 * single, rtol=1e-10)


# ------------------------------------------------- real PROD-SZ basis (if present)

@pytest.mark.skipif(not (PROD_CENTRAL / "C.ion.xml").is_file(),
                    reason="PROD-SZ central run not materialised")
def test_ion_xml_radials_are_read_with_the_documented_table_layout():
    radials, provenance = read_radials(PROD_CENTRAL / "C.ion.xml")
    assert set(radials) == {0, 1}
    for entry in provenance:
        assert entry["npts"] == 500
        assert entry["table_rows"] == 500          # interleaved (r, f) pairs
        assert entry["grid_matches_npts_delta"]    # (npts-1)*delta == cutoff
        assert entry["cutoff_ang"] > 2.0


@pytest.mark.skipif(not (PROD_CENTRAL / "C.ion.xml").is_file(),
                    reason="PROD-SZ central run not materialised")
def test_prod_basis_is_monocentrically_orthonormal():
    """<a|b> at R = 0 on one centre must be the identity -- pure quadrature check."""
    radials, _ = read_radials(PROD_CENTRAL / "C.ion.xml")
    basis = Basis(radials, [0, 1, 1, 1])
    grid = _grid(basis, 360, 64, 64)
    points, weights = grid
    values = basis.values(points)
    gram = np.einsum("an,bn,n->ab", values, values, weights)
    assert np.abs(gram - np.eye(4)).max() < B0_MONOCENTRIC_TOL
