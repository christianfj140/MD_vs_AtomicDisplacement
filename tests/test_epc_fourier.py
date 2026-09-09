"""D01 / E-F_001-S29: the q-space convention module.

Every adapter is checked twice: against an ``X(k)`` built directly from the
external definition (so the algebra is right, not just self-consistent), and by
a round trip (so it is exactly invertible). The two gauge statements that decide
whether a finite-``q`` calculation is right are runnable here:

* an **origin shift** moves nothing in the cell gauge;
* an **atom rewritten in another periodic image** moves the cell-gauge ``X(k)``
  by ``W X W^dag`` and moves the atom gauge not at all — and, because both are
  congruences, moves no generalized eigenvalue in either.

Machine precision means ``1e-13`` relative here: everything under test is a
product of unit-modulus phases, not a physical approximation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sla

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from epc_fourier import (  # noqa: E402
    ATOMIC_PHASE_ATOM,
    ATOMIC_PHASE_CELL,
    BLOCK_COLUMN_HOME,
    BLOCK_ROW_HOME,
    CANONICAL,
    CONVENTION_ID,
    EXTERNAL_CONVENTIONS,
    QConventionError,
    QSpaceConventions,
    atomic_phase,
    bloch_sum,
    convention_fields,
    effective_fourier_sign,
    external_conventions,
    fold_to_first_bz,
    fractional_positions,
    gauge_unitary,
    image_relabel_unitary,
    k_plus_q,
    matrix_at_k,
    minus_q_matrix,
    qspace_contract,
    relabel_home_images,
    require_same_conventions,
    to_canonical_at_k,
    from_canonical_at_k,
    to_canonical_vector_at_k,
)

TOL = 1e-13

ALL_CONVENTIONS = [
    QSpaceConventions(fourier_sign=sign, atomic_phase=phase, block_orientation=orientation)
    for sign in (1, -1)
    for phase in (ATOMIC_PHASE_CELL, ATOMIC_PHASE_ATOM)
    for orientation in (BLOCK_ROW_HOME, BLOCK_COLUMN_HOME)
]

# One orbital per atom keeps the fixture readable; the map orbital -> tau is
# what the adapters actually consume, so a wider basis changes nothing. The
# table reaches two cells along a1 while the blocks are non-zero only inside one
# — a cutoff, which is what makes relabeling by one cell representable.
IMAGES = np.array(
    [[x, y, 0] for x in (-2, -1, 0, 1, 2) for y in (-1, 0, 1)], dtype=np.int64
)
INDEX = {tuple(image): position for position, image in enumerate(IMAGES.tolist())}
NONZERO = np.abs(IMAGES[:, 0]) <= 1
TAU = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0], [0.21, 0.13, 0.0]])
K = np.array([0.17, -0.31, 0.0])
Q = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])


def hermitian_blocks(seed: int = 4) -> np.ndarray:
    """Real blocks with ``X(R)^T = X(-R)``, i.e. a hermitian ``X(k)``."""
    rng = np.random.default_rng(seed)
    blocks = rng.normal(size=(IMAGES.shape[0], TAU.shape[0], TAU.shape[0]))
    blocks[~NONZERO] = 0.0
    for position, image in enumerate(IMAGES.tolist()):
        partner = INDEX[tuple(-np.array(image))]
        if partner < position:
            continue
        if partner == position:  # R = 0 is its own partner: symmetrise it
            blocks[position] = 0.5 * (blocks[position] + blocks[position].T)
        else:
            blocks[partner] = blocks[position].T
    return blocks


def positive_definite_blocks(seed: int = 7) -> np.ndarray:
    """An overlap-like block set: hermitian at every k and positive definite."""
    blocks = 0.08 * hermitian_blocks(seed)
    blocks[INDEX[(0, 0, 0)]] += np.eye(TAU.shape[0])
    return blocks


def external_matrix(blocks: np.ndarray, k, conventions: QSpaceConventions) -> np.ndarray:
    """``X(k)`` as the external provider itself would write it (the ground truth)."""
    sign = effective_fourier_sign(conventions)
    matrix = bloch_sum(blocks, IMAGES, k, sign=sign)
    if conventions.atomic_phase == ATOMIC_PHASE_ATOM:
        # X^atom_{mu,nu} = exp(-i s k.tau_mu) X^cell_{mu,nu} exp(+i s k.tau_nu)
        phase = atomic_phase(k, TAU, sign)
        matrix = phase.conj()[:, None] * matrix * phase[None, :]
    return matrix


# --------------------------------------------------------------------------- #
# Adapters: external -> canonical
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("conventions", ALL_CONVENTIONS)
def test_adapter_reproduces_the_canonical_matrix(conventions):
    blocks = hermitian_blocks()
    canonical = bloch_sum(blocks, IMAGES, K)
    converted = to_canonical_at_k(
        external_matrix(blocks, K, conventions), K, conventions, orbital_positions_frac=TAU
    )
    assert np.allclose(converted, canonical, atol=TOL)


@pytest.mark.parametrize("conventions", ALL_CONVENTIONS)
def test_adapter_round_trip_is_machine_precision(conventions):
    blocks = hermitian_blocks()
    external = external_matrix(blocks, K, conventions)
    back = from_canonical_at_k(
        to_canonical_at_k(external, K, conventions, orbital_positions_frac=TAU),
        K,
        conventions,
        orbital_positions_frac=TAU,
    )
    assert np.max(np.abs(back - external)) < TOL


@pytest.mark.parametrize("name", sorted(EXTERNAL_CONVENTIONS))
def test_every_declared_backend_round_trips(name):
    conventions = external_conventions(name)
    blocks = hermitian_blocks()
    external = external_matrix(blocks, K, conventions)
    canonical = to_canonical_at_k(external, K, conventions, orbital_positions_frac=TAU)
    assert np.allclose(canonical, bloch_sum(blocks, IMAGES, K), atol=TOL)
    assert np.max(
        np.abs(from_canonical_at_k(canonical, K, conventions, orbital_positions_frac=TAU) - external)
    ) < TOL


def test_unknown_backend_raises_instead_of_getting_a_phase_patch():
    with pytest.raises(QConventionError, match="no declared q-space conventions"):
        external_conventions("some_new_code")


def test_block_orientation_is_a_fourier_sign_flip():
    """A column-home table summed with +1 is the row-home table summed with -1."""
    blocks = hermitian_blocks()
    index = {tuple(image): position for position, image in enumerate(IMAGES.tolist())}
    column_home = np.stack([blocks[index[tuple(-np.array(i))]] for i in IMAGES.tolist()])
    conventions = QSpaceConventions(block_orientation=BLOCK_COLUMN_HOME)
    assert np.allclose(
        matrix_at_k(column_home, IMAGES, K, conventions), bloch_sum(blocks, IMAGES, K), atol=TOL
    )
    assert effective_fourier_sign(conventions) == -1


def test_matrix_level_sign_flip_refuses_complex_real_space_blocks():
    conventions = QSpaceConventions(fourier_sign=-1)
    with pytest.raises(QConventionError, match="real-space blocks are real"):
        to_canonical_at_k(np.eye(3, dtype=complex), K, conventions, real_space_blocks_real=False)


def test_atom_gauge_needs_the_positions():
    with pytest.raises(QConventionError, match="orbital_positions_frac"):
        to_canonical_at_k(np.eye(3), K, QSpaceConventions(atomic_phase=ATOMIC_PHASE_ATOM))


# --------------------------------------------------------------------------- #
# Eigenvectors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("conventions", ALL_CONVENTIONS)
def test_converted_eigenvectors_solve_the_canonical_pencil(conventions):
    h_blocks, s_blocks = hermitian_blocks(1), positive_definite_blocks(2)
    h_ext = external_matrix(h_blocks, K, conventions)
    s_ext = external_matrix(s_blocks, K, conventions)
    eps, vectors = sla.eigh(h_ext, s_ext)

    canonical_vectors = to_canonical_vector_at_k(
        vectors, K, conventions, positions_frac=TAU
    )
    h = bloch_sum(h_blocks, IMAGES, K)
    s = bloch_sum(s_blocks, IMAGES, K)
    residual = h @ canonical_vectors - s @ canonical_vectors * eps[None, :]
    assert np.max(np.abs(residual)) < 1e-10
    # ... and the eigenvalues never depended on the convention in the first place
    assert np.allclose(eps, sla.eigh(h, s)[0], atol=1e-10)
    back = to_canonical_vector_at_k(
        canonical_vectors, K, conventions, positions_frac=TAU, inverse=True
    )
    assert np.max(np.abs(back - vectors)) < TOL


def test_phonon_sign_ignores_block_orientation():
    conventions = QSpaceConventions(phonon_fourier_sign=-1, block_orientation=BLOCK_COLUMN_HOME)
    eigenvector = np.array([[1.0 + 2.0j, 0.0, 0.0], [0.0, 1.0j, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(
        to_canonical_vector_at_k(eigenvector, Q, conventions, phonon=True),
        eigenvector.conj(),
        atol=TOL,
    )


# --------------------------------------------------------------------------- #
# Origin shifts and atoms crossing a periodic boundary
# --------------------------------------------------------------------------- #


def test_origin_shift_moves_nothing():
    """tau -> tau + t with no image relabeling: the cell gauge does not see it."""
    blocks = hermitian_blocks()
    shift = np.array([0.37, -0.11, 0.0])
    atom_conventions = QSpaceConventions(atomic_phase=ATOMIC_PHASE_ATOM)
    before = external_matrix(blocks, K, atom_conventions)
    phase = atomic_phase(K, TAU + shift, 1)
    after = phase.conj()[:, None] * bloch_sum(blocks, IMAGES, K) * phase[None, :]
    # The global exp(i k.t) cancels in the congruence, so even the atom gauge is
    # unchanged; the canonical matrix is untouched by construction.
    assert np.max(np.abs(after - before)) < TOL


def test_atom_across_the_boundary_is_a_declared_unitary_not_a_new_matrix():
    h_blocks, s_blocks = hermitian_blocks(1), positive_definite_blocks(2)
    shifts = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int64)  # atom 1 crosses +a1

    h_relabelled = relabel_home_images(h_blocks, IMAGES, shifts)
    s_relabelled = relabel_home_images(s_blocks, IMAGES, shifts)
    w = image_relabel_unitary(K, shifts)

    h, s = bloch_sum(h_blocks, IMAGES, K), bloch_sum(s_blocks, IMAGES, K)
    h_new = bloch_sum(h_relabelled, IMAGES, K)
    s_new = bloch_sum(s_relabelled, IMAGES, K)
    assert np.max(np.abs(h_new - w[:, None] * h * w.conj()[None, :])) < TOL
    assert np.max(np.abs(s_new - w[:, None] * s * w.conj()[None, :])) < TOL

    # A congruence by a unitary: not one eigenvalue moves.
    assert np.allclose(sla.eigh(h_new, s_new)[0], sla.eigh(h, s)[0], atol=1e-10)

    # The atom gauge, with tau updated to the new image, is invariant outright.
    atom = QSpaceConventions(atomic_phase=ATOMIC_PHASE_ATOM)
    old_atom_gauge = to_canonical_at_k(
        h, K, atom, orbital_positions_frac=TAU, inverse=True
    )
    new_atom_gauge = to_canonical_at_k(
        h_new, K, atom, orbital_positions_frac=TAU + shifts, inverse=True
    )
    assert np.max(np.abs(new_atom_gauge - old_atom_gauge)) < TOL


def test_relabeling_refuses_to_drop_a_block_off_the_image_table():
    blocks = hermitian_blocks()
    shifts = np.array([[0, 0, 0], [2, 0, 0], [0, 0, 0]], dtype=np.int64)
    with pytest.raises(QConventionError, match="not closed"):
        relabel_home_images(blocks, IMAGES, shifts)


def test_fractional_positions_round_trip():
    cell = np.array([[2.46, 0.0, 0.0], [-1.23, 2.13, 0.0], [0.0, 0.0, 20.0]])
    cartesian = TAU @ cell
    assert np.allclose(fractional_positions(cartesian, cell), TAU, atol=TOL)


# --------------------------------------------------------------------------- #
# k+q, folding, and q <-> -q
# --------------------------------------------------------------------------- #


def test_k_plus_q_and_folding():
    total = k_plus_q(K, Q)
    assert np.allclose(total, K + Q, atol=TOL)
    folded, g = fold_to_first_bz(total)
    assert np.allclose(folded + g, total, atol=TOL)
    assert np.all(folded >= -0.5) and np.all(folded < 0.5)
    assert g.dtype == np.int64

    blocks = hermitian_blocks()
    # Folding is the identity in the cell gauge ...
    assert np.allclose(
        bloch_sum(blocks, IMAGES, folded), bloch_sum(blocks, IMAGES, total), atol=TOL
    )
    # ... and a diagonal unitary in the atom gauge, which is why the gauge is
    # declared instead of assumed.
    assert not np.allclose(gauge_unitary(g, TAU), 1.0, atol=1e-6)


def test_minus_q_is_conjugation_for_real_blocks():
    blocks = hermitian_blocks()
    at_q = bloch_sum(blocks, IMAGES, Q)
    assert np.allclose(bloch_sum(blocks, IMAGES, -Q), minus_q_matrix(at_q), atol=TOL)
    assert np.allclose(atomic_phase(-Q, TAU), atomic_phase(Q, TAU).conj(), atol=TOL)
    with pytest.raises(QConventionError, match="real real-space blocks"):
        minus_q_matrix(at_q, real_space_blocks_real=False)


def test_phonon_minus_q_follows_the_same_rule():
    """The phonon side of the -q convention, through the provider that uses it."""
    from phonon_provider import ForceConstants

    rng = np.random.default_rng(11)
    cell = np.array([[2.46, 0.0, 0.0], [-1.23, 2.13, 0.0], [0.0, 0.0, 20.0]])
    images = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)
    matrix = rng.normal(size=(3, 2, 3, 2, 3))
    matrix[2] = matrix[1].transpose(2, 3, 0, 1)  # Phi(-R) = Phi(R)^T
    matrix[0] = 0.5 * (matrix[0] + matrix[0].transpose(2, 3, 0, 1))
    fc = ForceConstants(
        matrix=matrix,
        cell_images=images,
        cell_ang=cell,
        positions_ang=np.array([[0.0, 0.0, 0.0], [1.23, 0.71, 0.0]]),
        masses_amu=np.array([12.011, 12.011]),
    )
    assert np.allclose(fc.dynamical_matrix(-Q), fc.dynamical_matrix(Q).conj(), atol=TOL)


# --------------------------------------------------------------------------- #
# What artifacts declare
# --------------------------------------------------------------------------- #


def test_convention_fields_are_the_same_object_everywhere():
    from compute_epc_matrix_elements import RawDerivative  # noqa: F401  (import guard)

    fields = convention_fields()
    assert fields["convention_id"] == CONVENTION_ID
    assert fields["fourier_sign"] == 1 and fields["phonon_fourier_sign"] == 1
    assert fields["block_orientation"] == BLOCK_ROW_HOME
    require_same_conventions(fields, convention_fields(), what="two Gamma artifacts")


def test_gamma_response_cannot_be_relabelled_at_finite_q():
    with pytest.raises(QConventionError, match="not q-aware"):
        convention_fields(Q, q_aware_source=False)
    assert convention_fields(Q, q_aware_source=True)["q"] == list(Q)


def test_artifacts_phased_differently_cannot_be_contracted():
    left = convention_fields(Q, q_aware_source=True)
    right = convention_fields()
    with pytest.raises(QConventionError, match="do not share the q-space convention"):
        require_same_conventions(left, right, what="D_H and the basis response")


def test_artifacts_are_only_signed_in_the_canon():
    with pytest.raises(QConventionError, match="adapter boundary"):
        convention_fields(conventions=QSpaceConventions(fourier_sign=-1))


def test_contract_publishes_the_adapters():
    contract = qspace_contract()
    assert contract["convention_id"] == CONVENTION_ID
    assert contract["canonical"] == CANONICAL.to_dict()
    assert set(contract["adapters"]) == set(EXTERNAL_CONVENTIONS)
    assert "image_relabeling" in contract["gauge_covariance"]


def test_basis_response_and_raw_derivative_share_the_record():
    """The q-aware half of the basis-response contract (S29 requirement)."""
    from epc_basis_response import basis_response_contract

    assert basis_response_contract()["convention_id"] == CONVENTION_ID
    assert basis_response_contract()["conventions"] == convention_fields()
