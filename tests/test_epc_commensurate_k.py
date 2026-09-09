#!/usr/bin/env python3
"""D01/D03 / E-F_001-S30: the commensurate q != 0 reference, on a toy chain.

The toy is a periodic two-atom chain tripled into a commensurate supercell for
``q = (1/3, 0, 0)``: small enough to differentiate by hand, structured enough
that everything S30 claims is falsifiable — the cos/sin split, the complex
reconstruction, the ``exp(2i pi q.R)`` transformation law that tells a genuine
finite-``q`` response from a Gamma one, and the ``k -> k+q`` selection by Bloch
character rather than by band index.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import epc_commensurate_k as ck  # noqa: E402
import epc_basis_response as ebr  # noqa: E402
import epc_fourier as qf  # noqa: E402
import phonon_provider as pp  # noqa: E402
from compute_epc_matrix_elements import (  # noqa: E402
    PaoCovariantResponse,
    RawDerivative,
    dense_eigenspace,
)
from epc_subspaces import random_gauge_rotation, subspace_metrics, subspace_overlap  # noqa: E402

# --------------------------------------------------------------------------- #
# The toy: a two-atom chain, one orbital per atom, overlapping neighbours
# --------------------------------------------------------------------------- #

A_ANG = 2.5
PRIMITIVE_CELL = np.diag([A_ANG, 12.0, 12.0])
PRIMITIVE_POSITIONS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
Q = (1.0 / 3.0, 0.0, 0.0)
DELTA = 1e-4
ONSITE = (-1.0, 1.0)
T0, S0, LAMBDA = 2.4, 0.18, 0.9


def _matrices(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``H``, ``S`` of the ring, minimum image along x. Real and symmetric."""
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    length = 3.0 * A_ANG
    dx = positions[:, 0][:, None] - positions[:, 0][None, :]
    dx -= length * np.rint(dx / length)
    distance = np.sqrt(dx**2 + (positions[:, 1][:, None] - positions[:, 1][None, :]) ** 2
                       + (positions[:, 2][:, None] - positions[:, 2][None, :]) ** 2)
    decay = np.exp(-distance / LAMBDA)
    off = ~np.eye(n, dtype=bool)
    H = np.where(off, -T0 * decay, 0.0)
    S = np.where(off, S0 * decay, 0.0) + np.eye(n)
    H[np.diag_indices(n)] = [ONSITE[index % 2] for index in range(n)]
    return H, S


def _cell() -> ck.CommensurateCell:
    return ck.CommensurateCell(
        q_primitive=Q,
        matrix=ck.diagonal_supercell_matrix(Q),
        primitive_cell_ang=PRIMITIVE_CELL,
        primitive_positions_ang=PRIMITIVE_POSITIONS,
    )


def _mode(cell: ck.CommensurateCell) -> ck.ComplexMode:
    phonon = pp.PhononMode(
        q_fractional=np.asarray(Q),
        branch=3,
        frequency_ev=0.2,
        eigenvector=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]) / np.sqrt(2.0),
        masses_amu=np.array([12.011, 12.011]),
        cell_ang=PRIMITIVE_CELL,
        positions_ang=PRIMITIVE_POSITIONS,
        conventions=pp.CANONICAL_CONVENTIONS,
    )
    return ck.complex_mode(cell, phonon, name="toy_optical")


def _half(cell: ck.CommensurateCell, direction, *, q=(0.0, 0.0, 0.0), q_aware=False):
    """One real directional response of the supercell: ``D_H``, ``S_L``, ``S_R``.

    ``S_L`` is the bra half of ``dS`` and ``S_R`` the ket half, taken atom by
    atom, so ``S_L + S_R == D_S`` holds by construction rather than by fiat.
    """
    positions = cell.positions_ang
    vectors = np.asarray(direction.vectors)
    plus, minus = _matrices(positions + DELTA * vectors), _matrices(positions - DELTA * vectors)
    D_H = (plus[0] - minus[0]) / (2.0 * DELTA)
    S_left = np.zeros_like(plus[1])
    S_right = np.zeros_like(plus[1])
    for atom in range(positions.shape[0]):
        single = np.zeros_like(vectors)
        single[atom] = vectors[atom]
        derivative = (
            _matrices(positions + DELTA * single)[1] - _matrices(positions - DELTA * single)[1]
        ) / (2.0 * DELTA)
        S_left[atom, :] += derivative[atom, :]
        S_right[:, atom] += derivative[:, atom]
    context = ebr.ResponseContext(
        geometry_signature="toy_chain_geometry",
        basis_signature="toy_chain_basis",
        electronic_derivative_backend="toy_tight_binding",
        direction=direction,
        q=q,
        q_aware_source=q_aware,
    )
    raw = RawDerivative(
        blocks={"D_H": D_H},
        isc_off=((0, 0, 0),),
        backend="toy_tight_binding",
        method="frozen_central_difference",
        direction=direction,
        geometry_signature="toy_chain_geometry",
        basis_signature="toy_chain_basis",
        topology_sha256="toy_topology",
        delta_or_jvp=DELTA,
        q=q,
        q_aware_source=q_aware,
    )
    response = ebr.from_S_L_S_R(
        S_left,
        S_right,
        context,
        backend="toy_tight_binding",
        method="frozen_central_difference",
        intra_atomic_included=True,
    )
    return PaoCovariantResponse(raw=raw, response=response)


def _response(cell=None, mode=None) -> ck.ComplexModeResponse:
    cell = cell or _cell()
    mode = mode or _mode(cell)
    return ck.ComplexModeResponse(
        mode=mode, cos=_half(cell, mode.cos), sin=_half(cell, mode.sin)
    )


# --------------------------------------------------------------------------- #
# The cell
# --------------------------------------------------------------------------- #


def test_supercell_is_commensurate_and_maps_atoms():
    cell = _cell()
    assert cell.matrix.tolist() == np.diag([3, 1, 1]).tolist()
    assert cell.multiplicity == 3
    assert cell.atom_count == 6
    assert cell.images[0].tolist() == [0, 0, 0]
    assert np.allclose(cell.cell_ang[0], [3.0 * A_ANG, 0.0, 0.0])
    assert np.allclose(cell.positions_ang[:, 0], [0.0, 1.0, 2.5, 3.5, 5.0, 6.0])
    # q folds onto the supercell Gamma: that is the whole point of route Q-A.
    assert np.allclose(cell.fold(Q), 0.0)
    assert cell.folds_together((0.0, 0.0, 0.0), Q)


def test_incommensurate_supercell_is_refused():
    with pytest.raises(ck.CommensurateError, match="not commensurate"):
        ck.CommensurateCell(
            q_primitive=Q,
            matrix=np.diag([2, 1, 1]),
            primitive_cell_ang=PRIMITIVE_CELL,
            primitive_positions_ang=PRIMITIVE_POSITIONS,
        )


# --------------------------------------------------------------------------- #
# The mode: two real halves that rebuild one complex pattern
# --------------------------------------------------------------------------- #


def test_cos_and_sin_are_separate_artifacts_that_rebuild_the_complex_pattern():
    cell = _cell()
    mode = _mode(cell)
    expected = np.concatenate(
        [
            np.exp(2j * np.pi * float(np.asarray(Q) @ image))
            * np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
            / np.sqrt(2.0)
            * pp.zero_point_amplitude_ang(12.011, 0.2)
            for image in cell.images
        ]
    )
    assert np.allclose(mode.pattern_ang, expected, atol=1e-12)
    # Separately named, separately hashed, separately signable.
    assert mode.cos.name.endswith("__cos") and mode.sin.name.endswith("__sin")
    assert mode.cos.direction_hash != mode.sin.direction_hash
    assert mode.complex_mode_hash == _mode(_cell()).complex_mode_hash


def test_a_synthetic_displacement_is_not_a_mode():
    with pytest.raises(ck.CommensurateError, match="not a phonon"):
        ck.complex_mode(_cell(), object(), name="fake")


def test_a_gamma_phased_block_cannot_carry_the_finite_q_pattern():
    cell = _cell()
    mode = _mode(cell)
    with pytest.raises(ck.CommensurateError, match="signed at q"):
        ck.ComplexModeResponse(
            mode=mode,
            cos=_half(cell, mode.cos, q=Q, q_aware=True),
            sin=_half(cell, mode.sin),
        )


def test_halves_must_be_the_modes_own_directions():
    cell = _cell()
    mode = _mode(cell)
    with pytest.raises(ck.CommensurateError, match="not along the mode"):
        ck.ComplexModeResponse(mode=mode, cos=_half(cell, mode.sin), sin=_half(cell, mode.sin))


# --------------------------------------------------------------------------- #
# The complex response is a genuine q != 0 object
# --------------------------------------------------------------------------- #


def _translate(matrix: np.ndarray, cells: int) -> np.ndarray:
    """Relabel orbitals by ``l -> l + cells`` on the ring (2 orbitals per cell)."""
    order = np.concatenate([(np.arange(2) + 2 * ((image + cells) % 3)) for image in range(3)])
    return matrix[np.ix_(order, order)]


def test_complex_D_H_carries_the_q_phase_under_a_primitive_translation():
    response = _response()
    D_H = response.D_H_at_k((0.0, 0.0, 0.0))
    H, _ = _matrices(response.mode.cell.positions_ang)
    for cells in (1, 2):
        # The unperturbed matrix is translation invariant ...
        assert np.allclose(_translate(H, cells), H, atol=1e-12)
        # ... and the response of a q-modulated displacement is not: it picks up
        # exactly exp(2i pi q.R), which is what a Gamma response cannot do.
        phase = np.exp(2j * np.pi * cells * Q[0])
        assert np.allclose(_translate(D_H, cells), phase * D_H, atol=1e-8)


def test_basis_response_follows_the_same_combination_as_D_H():
    response = _response()
    combined = response.basis_response_at_k((0.0, 0.0, 0.0))
    mode = response.mode
    for name in ("S_left", "S_right"):
        expected = mode.cos_amplitude_ang * getattr(
            response.cos.response.at_k((0.0, 0.0, 0.0)), name
        ) + 1j * mode.sin_amplitude_ang * getattr(
            response.sin.response.at_k((0.0, 0.0, 0.0)), name
        )
        assert np.allclose(getattr(combined, name), expected, atol=1e-12)
    # S_L + S_R is the D_S of the same complex pattern, by construction.
    positions = mode.cell.positions_ang
    pattern = mode.pattern_ang
    D_S = (
        _matrices(positions + DELTA * pattern.real)[1]
        - _matrices(positions - DELTA * pattern.real)[1]
        + 1j
        * (
            _matrices(positions + DELTA * pattern.imag)[1]
            - _matrices(positions - DELTA * pattern.imag)[1]
        )
    ) / (2.0 * DELTA)
    assert np.allclose(combined.total, D_S, atol=1e-6)


# --------------------------------------------------------------------------- #
# k -> k+q by subspace, not by band index
# --------------------------------------------------------------------------- #


def _window():
    cell = _cell()
    H, S = _matrices(cell.positions_ang)
    return cell, H, S, dense_eigenspace((0.0, 0.0, 0.0), H, S, label="toy_gamma")


def test_unfolding_weights_partition_the_window_over_the_folding_star():
    cell, _, S, window = _window()
    image_index = cell.orbital_image_index([1, 1])
    weights = np.stack(
        [
            ck.unfolding_weights(
                window.C, S, ck.bloch_character_basis(image_index, cell.images, (k, 0.0, 0.0))
            )
            for k in (0.0, 1.0 / 3.0, 2.0 / 3.0)
        ]
    )
    # The star is complete: every state's character adds up to one over it.
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-10)
    # Each primitive k carries two of the six states' worth of character ...
    assert [int(row.sum().round()) for row in weights] == [2, 2, 2]
    # ... but four states are half k and half k+q, because k and -k are
    # degenerate at the folded point and the solver mixed them. This is exactly
    # why the module projects instead of selecting by band index.
    assert np.isclose(np.sort(weights[1])[-1], 0.5, atol=1e-9)


def test_k_to_k_plus_q_report_is_a_candidate_block_between_two_characters():
    cell, H, S, window = _window()
    response = _response(cell=cell)
    report = ck.k_to_k_plus_q_report(
        response,
        window,
        H,
        S,
        k_primitive=(0.0, 0.0, 0.0),
        orbital_image_index=cell.orbital_image_index([1, 1]),
    )
    assert report["status"] == ck.CANDIDATE_STATUS
    assert np.allclose(report["k_plus_q_primitive"], Q)
    assert report["initial_subspace"]["dimension"] == 2
    assert report["final_subspace"]["dimension"] == 2
    # The two characters are orthogonal subspaces of one window, so no state was
    # claimed twice: this is the check that replaces "the band indices differ".
    assert report["subspace_separation"]["projector_trace_overlap"] < 1e-16
    assert report["initial_subspace"]["maximum_ritz_relative_residual"] < 1e-10
    values = np.asarray(report["g"]["values_real"]) + 1j * np.asarray(report["g"]["values_imaginary"])
    assert values.shape == (2, 2)
    assert np.linalg.norm(values) > 0.0
    assert report["perturbation"]["convention"]["q_representation"] == ck.Q_REPRESENTATION
    assert report["perturbation"]["convention"]["q_primitive"] == list(Q)
    # The artifacts underneath stay Gamma artifacts of the supercell.
    assert report["perturbation"]["halves"]["cos"]["raw_derivative"]["q"] == [0.0, 0.0, 0.0]


def test_the_projection_survives_a_gauge_rotation_of_the_window():
    """The character subspace is a property of the window, not of its basis."""
    cell, H, S, window = _window()
    basis = ck.bloch_character_basis(cell.orbital_image_index([1, 1]), cell.images, Q)
    reference, _ = ck.project_character(window, H, S, basis, label="k_plus_q")
    rotated = ck.Eigenspace(
        label=window.label,
        k=window.k,
        C=window.C @ random_gauge_rotation([list(range(window.state_count))],
                                           window.state_count,
                                           np.random.default_rng(7)),
        eps=window.eps,
        identity_error=window.identity_error,
        identity_tolerance=window.identity_tolerance,
        resolutions_ev=window.resolutions_ev,
        cluster_labels=window.cluster_labels,
    )
    projected, _ = ck.project_character(rotated, H, S, basis, label="k_plus_q")
    metrics = subspace_metrics(subspace_overlap(projected.C, S, reference.C))
    assert projected.state_count == reference.state_count
    assert metrics["projector_frobenius_distance"] < 1e-8
    assert np.allclose(np.sort(projected.eps), np.sort(reference.eps), atol=1e-10)


def test_contract_is_data_and_says_it_is_not_go5():
    contract = ck.commensurate_contract()
    assert contract["status"] == ck.CANDIDATE_STATUS
    assert contract["q_representation"] == ck.Q_REPRESENTATION


# --------------------------------------------------------------------------- #
# E-F_001-S31: the round trips GO-5 requires on top of what S30 built
# --------------------------------------------------------------------------- #


def test_D_H_of_minus_q_is_the_complex_conjugate_of_D_H_of_q():
    """S29's ``MINUS_Q_CONVENTION`` (``X(-q) = conj(X(q))``), carried through
    the full cos/sin combination into ``D_H(q)`` and into the physical,
    gauge-invariant ``g`` block -- not just the raw halves.
    """
    cell, H, S, window = _window()
    response = _response(cell=cell)
    q_minus = tuple(-value for value in Q)
    cell_minus = ck.CommensurateCell(
        q_primitive=q_minus,
        matrix=ck.diagonal_supercell_matrix(q_minus),
        primitive_cell_ang=PRIMITIVE_CELL,
        primitive_positions_ang=PRIMITIVE_POSITIONS,
    )
    # Same |det M| = 3, same lattice: -q folds onto the same supercell as q.
    assert np.allclose(cell_minus.positions_ang, cell.positions_ang, atol=1e-12)
    phonon_minus = pp.PhononMode(
        q_fractional=np.asarray(q_minus),
        branch=3,
        frequency_ev=0.2,
        eigenvector=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]) / np.sqrt(2.0),
        masses_amu=np.array([12.011, 12.011]),
        cell_ang=PRIMITIVE_CELL,
        positions_ang=PRIMITIVE_POSITIONS,
        conventions=pp.CANONICAL_CONVENTIONS,
    )
    mode_minus = ck.complex_mode(cell_minus, phonon_minus, name="toy_optical__minus_q")
    response_minus = ck.ComplexModeResponse(
        mode=mode_minus,
        cos=_half(cell_minus, mode_minus.cos),
        sin=_half(cell_minus, mode_minus.sin),
    )
    assert np.allclose(
        response.D_H_at_k((0.0, 0.0, 0.0)),
        np.conj(response_minus.D_H_at_k((0.0, 0.0, 0.0))),
        atol=1e-8,
    )

    image_index = cell.orbital_image_index([1, 1])
    report_q = ck.k_to_k_plus_q_report(
        response, window, H, S, k_primitive=(0.0, 0.0, 0.0), orbital_image_index=image_index
    )
    report_minus = ck.k_to_k_plus_q_report(
        response_minus, window, H, S, k_primitive=(0.0, 0.0, 0.0), orbital_image_index=image_index
    )
    g_q = np.asarray(report_q["g"]["values_real"]) + 1j * np.asarray(report_q["g"]["values_imaginary"])
    g_minus = np.asarray(report_minus["g"]["values_real"]) + 1j * np.asarray(
        report_minus["g"]["values_imaginary"]
    )
    assert np.allclose(g_q, np.conj(g_minus), atol=1e-6)
    assert np.isclose(
        report_q["g"]["frobenius_norm"], report_minus["g"]["frobenius_norm"], atol=1e-10
    )


def _permute_half(half: PaoCovariantResponse, order: np.ndarray) -> PaoCovariantResponse:
    """Relabel one real half by ``order``: which primitive cell is called home.

    This is what a cell-origin shift *and* an atom crossing the periodic
    boundary both are here -- the ring has no distinguished origin, so
    renaming which index is cell 0 is the whole transformation (roadmap
    Fase 7). Exercised on this module's own artifacts, on top of what
    tests/test_epc_fourier.py already certifies for the generic Bloch-sum
    primitives (``image_relabel_unitary`` / ``relabel_home_images``).
    """
    ix = np.ix_(order, order)
    raw = half.raw
    new_raw = RawDerivative(
        blocks={"D_H": raw.blocks["D_H"][0][ix][None, ...]},
        isc_off=raw.isc_off,
        backend=raw.backend,
        method=raw.method,
        direction=raw.direction,
        geometry_signature=raw.geometry_signature,
        basis_signature=raw.basis_signature,
        topology_sha256=raw.topology_sha256,
        delta_or_jvp=raw.delta_or_jvp,
        q=raw.q,
        q_aware_source=raw.q_aware_source,
    )
    response = half.response
    new_response = ebr.from_S_L_S_R(
        response.blocks["S_left"][0][ix],
        response.blocks["S_right"][0][ix],
        response.context,
        backend=response.backend,
        method=response.method,
        intra_atomic_included=response.intra_atomic_included,
    )
    return PaoCovariantResponse(raw=new_raw, response=new_response)


def test_g_is_invariant_under_a_relabeling_that_crosses_the_periodic_boundary():
    """The PAO-projected coupling -- GO-6's gauge-invariant singular values and
    Frobenius norm -- must not move under a cell-origin shift / an atom
    crossing the periodic boundary, even though the raw ``D_H`` does pick up
    a phase under the same relabeling
    (``test_complex_D_H_carries_the_q_phase_under_a_primitive_translation``).
    """
    cell, H, S, window = _window()
    response = _response(cell=cell)
    image_index = cell.orbital_image_index([1, 1])
    report = ck.k_to_k_plus_q_report(
        response, window, H, S, k_primitive=(0.0, 0.0, 0.0), orbital_image_index=image_index
    )
    for cells in (1, 2):
        order = np.concatenate([(np.arange(2) + 2 * ((image + cells) % 3)) for image in range(3)])
        H_t, S_t = _matrices(cell.positions_ang[order])
        assert np.allclose(H_t, _translate(H, cells), atol=1e-12)  # sanity: the same relabeling
        window_t = dense_eigenspace((0.0, 0.0, 0.0), H_t, S_t, label=f"toy_gamma__shift{cells}")
        response_t = ck.ComplexModeResponse(
            mode=response.mode,
            cos=_permute_half(response.cos, order),
            sin=_permute_half(response.sin, order),
        )
        report_t = ck.k_to_k_plus_q_report(
            response_t,
            window_t,
            H_t,
            S_t,
            k_primitive=(0.0, 0.0, 0.0),
            orbital_image_index=image_index[order],
        )
        assert np.isclose(
            report["g"]["frobenius_norm"], report_t["g"]["frobenius_norm"], atol=1e-10
        )
        assert np.allclose(
            sorted(report["g"]["singular_values"]),
            sorted(report_t["g"]["singular_values"]),
            atol=1e-9,
        )


def test_alternative_atomic_phase_convention_round_trips_to_the_same_pattern():
    """A provider that writes the atomic phase into the eigenvector instead of
    the cell (DFTBephy-style, ``EXTERNAL_CONVENTIONS["dftbephy"]``) must
    reproduce the identical ``complex_mode`` pattern once converted through
    ``phonon_provider.to_canonical_conventions`` -- the round trip S29
    promises, exercised through this module's own ``complex_mode`` boundary
    rather than only at the generic adapter level.
    """
    cell = _cell()
    canonical = _mode(cell)
    eigenvector = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]) / np.sqrt(2.0)
    tau = qf.atomic_phase(
        np.asarray(Q), qf.fractional_positions(PRIMITIVE_POSITIONS, PRIMITIVE_CELL), sign=1
    )
    atom_gauge_eigenvector = eigenvector.astype(np.complex128) * np.conj(tau)[:, None]
    mode_set = pp.PhononModeSet(
        q_fractional=np.asarray(Q),
        frequencies_ev=np.array([0.2]),
        eigenvectors=atom_gauge_eigenvector[None, :, :],
        masses_amu=np.array([12.011, 12.011]),
        cell_ang=PRIMITIVE_CELL,
        positions_ang=PRIMITIVE_POSITIONS,
        conventions=pp.PhononConventions(atomic_phase=pp.ATOMIC_PHASE_ATOM),
        provider="toy_external_dftbephy_like",
        provider_version="1",
        force_source={},
        primitive_mapping={},
        fc_range={},
        asr_policy=pp.ASR_RAW,
        asr_residual_ev_ang2=0.0,
        asr_residual_raw_ev_ang2=0.0,
    )
    converted = pp.to_canonical_conventions(mode_set)
    assert converted.conventions.is_canonical
    from_external = ck.complex_mode(cell, converted.mode(0), name="toy_optical__from_external")
    assert np.allclose(from_external.pattern_ang, canonical.pattern_ang, atol=1e-12)


def test_go5_decision_is_data_and_names_exactly_what_blocks_it():
    decision = ck.go5_decision()
    assert decision["algebra_layer"] == "PASS"
    assert decision["go5"] == "NO_GO"
    assert decision["physical_layer"] == "NOT_ATTEMPTED"
    assert all(status == "PASS" for status in decision["checks"].values())
    assert "GO-4" in decision["blocked_by"]
