"""C08 / E-F_001-S10: the finite-difference perturbation space.

The three families the graphene-Gamma experiment needs (one Cartesian, one
non-trivial collective, one uniform translation), their single normalisation,
the delta sweep that is only ever *swept*, and the neighbour-topology gate.

The one-hot equivalence is checked against the real
``build_hamiltonian_derivative_stencils.displaced_positions`` -- the previous
behaviour -- not against a re-statement of it, and the PAO radii are read from
the repository's real graphene orbital contract rather than an invented payload.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from build_hamiltonian_derivative_stencils import displaced_positions  # noqa: E402
from fd_perturbation_space import (  # noqa: E402
    DEFAULT_DELTA_CENTER_ANG,
    KIND_COLLECTIVE,
    KIND_ONE_HOT,
    KIND_UNIFORM_TRANSLATION,
    MIN_PLATEAU_DELTA_COUNT,
    NORMALIZATION,
    Direction,
    FdPerturbationError,
    assert_topology_preserved,
    atom_cutoff_radii_ang,
    breathing_like,
    collective,
    default_direction_set,
    delta_sweep,
    delta_sweep_status,
    displace,
    normalize,
    one_hot,
    optical_like,
    random_collective,
    select_delta_from_plateau,
    shear_like,
    signs_for_method,
    synthetic_benchmark_direction_set,
    topology_margin,
    uniform_translation,
)

# Two carbons 1.42 Ang apart in a box wide enough that no image is a neighbour
# at the 2.0 Ang test cutoff.
DIMER_POSITIONS = [[0.0, 0.0, 0.0], [1.42, 0.0, 0.0]]
DIMER_CELL = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]

GRAPHENE_FDF = REPO_ROOT / "materials" / "graphene" / "RUN.fdf"
GRAPHENE_BASIS = REPO_ROOT / "materials" / "graphene_common" / "basis"
GRAPHENE_A_ANG = 2.46
GRAPHENE_POSITIONS = [[0.0, 0.0, 0.0], [GRAPHENE_A_ANG / 2.0, GRAPHENE_A_ANG / (2.0 * math.sqrt(3.0)), 0.0]]
GRAPHENE_CELL = [
    [GRAPHENE_A_ANG, 0.0, 0.0],
    [GRAPHENE_A_ANG / 2.0, GRAPHENE_A_ANG * math.sqrt(3.0) / 2.0, 0.0],
    [0.0, 0.0, 20.0],
]


# --------------------------------------------------------------------------- #
# Direction space and normalisation
# --------------------------------------------------------------------------- #


def test_default_set_is_the_three_required_families():
    kinds = [direction.kind for direction in default_direction_set(4)]
    assert kinds == [KIND_ONE_HOT, KIND_COLLECTIVE, KIND_UNIFORM_TRANSLATION]


@pytest.mark.parametrize("direction", default_direction_set(5, seed=7))
def test_every_family_is_unit_frobenius(direction):
    assert direction.normalization == NORMALIZATION
    assert direction.vectors.shape == (5, 3)
    assert float(np.linalg.norm(direction.vectors)) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Synthetic resource-benchmark directions (E-F_001-S37 / D06): breathing-like,
# shear-like, optical-like, random -- non-phonon coordinate patterns of the
# right *shape* for a two-layer stack, never a dynamical-matrix eigenvector.
# --------------------------------------------------------------------------- #

BILAYER_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.42, 0.0, 0.0],
        [0.0, 0.0, 3.35],
        [1.42, 0.0, 3.35],
    ]
)


def test_breathing_like_moves_layers_oppositely_along_z():
    direction = breathing_like(BILAYER_POSITIONS)
    assert direction.kind == KIND_COLLECTIVE
    bottom_z = direction.vectors[:2, 2]
    top_z = direction.vectors[2:, 2]
    assert np.all(bottom_z < 0.0)
    assert np.all(top_z > 0.0)
    assert np.allclose(direction.vectors[:, :2], 0.0)


def test_shear_like_moves_layers_oppositely_in_plane():
    direction = shear_like(BILAYER_POSITIONS, "x")
    bottom_x = direction.vectors[:2, 0]
    top_x = direction.vectors[2:, 0]
    assert np.all(bottom_x < 0.0)
    assert np.all(top_x > 0.0)
    assert np.allclose(direction.vectors[:, 1:], 0.0)


def test_shear_like_rejects_out_of_plane_axis():
    with pytest.raises(FdPerturbationError):
        shear_like(BILAYER_POSITIONS, "z")


def test_optical_like_alternates_sign_by_atom_index():
    direction = optical_like(4, seed=0)
    even_vectors = direction.vectors[0::2]
    odd_vectors = direction.vectors[1::2]
    assert np.allclose(even_vectors, -odd_vectors[: len(even_vectors)])


def test_synthetic_benchmark_direction_set_is_the_four_required_families():
    directions = synthetic_benchmark_direction_set(BILAYER_POSITIONS, seed=1)
    assert [direction.name for direction in directions] == [
        "breathing_like",
        "shear_like_x",
        "optical_like",
        "random_seed1",
    ]
    for direction in directions:
        assert direction.kind == KIND_COLLECTIVE
        assert float(np.linalg.norm(direction.vectors)) == pytest.approx(1.0, abs=1e-12)
        # These are timing/shape probes only; nothing here may claim to be a
        # diagonalised phonon eigenvector.
        assert "phonon" not in direction.name


def test_direction_refuses_an_unnormalised_pattern():
    with pytest.raises(FdPerturbationError, match="unit-Frobenius"):
        Direction(name="raw", kind=KIND_COLLECTIVE, vectors=np.full((2, 3), 0.5))


@pytest.mark.parametrize(
    "vectors",
    [np.zeros((2, 3)), np.ones((2, 4)), np.ones(3), [[1.0, 0.0, np.nan]]],
    ids=["zero", "wrong_width", "wrong_rank", "non_finite"],
)
def test_normalize_rejects_degenerate_patterns(vectors):
    with pytest.raises(FdPerturbationError):
        normalize(vectors)


def test_collective_direction_is_non_trivial():
    direction = random_collective(6, seed=3)
    # Not one-hot: more than one atom and more than one component move.
    assert np.count_nonzero(np.linalg.norm(direction.vectors, axis=1) > 1e-9) == 6
    # Not a translation: atoms do not move alike.
    assert np.linalg.norm(direction.vectors - direction.vectors[0]) > 1e-3
    assert random_collective(6, seed=3).direction_hash == direction.direction_hash
    assert random_collective(6, seed=4).direction_hash != direction.direction_hash


def test_direction_hash_ignores_the_name_but_not_the_pattern():
    vectors = random_collective(3, seed=1).vectors
    assert collective(vectors, name="a").direction_hash == collective(vectors, name="b").direction_hash
    assert collective(vectors, name="a").direction_hash != collective(2.0 * vectors + 0.5, name="a").direction_hash


def test_one_hot_metadata_carries_atom_and_axis():
    metadata = one_hot(3, 2, "z").to_metadata()
    assert metadata["atom_index_zero_based"] == 2
    assert (metadata["axis"], metadata["axis_index"]) == ("z", 2)
    assert metadata["direction_kind"] == KIND_ONE_HOT
    assert metadata["direction_max_atom_norm"] == pytest.approx(1.0)
    # A collective direction has no single atom/axis to report.
    assert "atom_index_zero_based" not in random_collective(3).to_metadata()


@pytest.mark.parametrize("axis", ["w", 3, -1, ""])
def test_one_hot_rejects_an_unknown_axis(axis):
    # -1 is a mistake, not "the z axis": integer axes are not wrapped around.
    with pytest.raises(FdPerturbationError, match="Unsupported axis"):
        one_hot(2, 0, axis)
    with pytest.raises(FdPerturbationError, match="Unsupported translation axis"):
        uniform_translation(2, axis)


def test_one_hot_rejects_an_atom_outside_the_structure():
    with pytest.raises(FdPerturbationError, match="outside a structure"):
        one_hot(2, 2, "x")


# --------------------------------------------------------------------------- #
# Displacement: the one-hot path must be the historical path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("atom_index", [0, 1])
@pytest.mark.parametrize("axis, axis_index", [("x", 0), ("y", 1), ("z", 2)])
@pytest.mark.parametrize("sign", [1, -1])
def test_one_hot_reproduces_the_previous_stencil_bit_for_bit(atom_index, axis, axis_index, sign):
    delta = 0.007
    legacy = displaced_positions(
        [tuple(position) for position in DIMER_POSITIONS],
        atom_index_zero_based=atom_index,
        axis_index=axis_index,
        signed_delta=sign * delta,
    )
    new = displace(DIMER_POSITIONS, one_hot(2, atom_index, axis), sign * delta)
    assert new == legacy


def test_collective_displacement_is_R_plus_delta_v():
    direction = random_collective(2, seed=11)
    moved = np.asarray(displace(DIMER_POSITIONS, direction, 0.01))
    assert moved == pytest.approx(np.asarray(DIMER_POSITIONS) + 0.01 * direction.vectors, abs=1e-15)


def test_central_pair_shares_ordering_and_is_symmetric_about_the_base():
    direction = random_collective(2, seed=5)
    delta = 0.01
    plus, minus = (displace(DIMER_POSITIONS, direction, sign * delta) for sign in signs_for_method("central"))
    assert signs_for_method("central") == (1, -1)
    assert len(plus) == len(minus) == len(DIMER_POSITIONS)
    # Same atom ordering, so R+ and R- straddle R atom by atom.
    assert np.asarray(plus) + np.asarray(minus) == pytest.approx(2.0 * np.asarray(DIMER_POSITIONS), abs=1e-15)


def test_uniform_translation_leaves_every_interatomic_distance_unchanged():
    direction = uniform_translation(2, [1.0, 1.0, 0.0])
    moved = np.asarray(displace(DIMER_POSITIONS, direction, 0.05))
    base = np.asarray(DIMER_POSITIONS)
    assert np.linalg.norm(moved[1] - moved[0]) == pytest.approx(np.linalg.norm(base[1] - base[0]), abs=1e-14)
    # Unit-Frobenius spreads delta over the atoms: each moves delta/sqrt(N).
    assert np.linalg.norm(moved[0] - base[0]) == pytest.approx(0.05 / math.sqrt(2), abs=1e-14)


def test_displace_refuses_a_direction_of_the_wrong_size():
    with pytest.raises(FdPerturbationError, match="spans"):
        displace(DIMER_POSITIONS, random_collective(3), 0.01)


def test_unsupported_method_is_refused():
    with pytest.raises(FdPerturbationError):
        signs_for_method("five_point")


# --------------------------------------------------------------------------- #
# Amplitudes: swept, never preferred
# --------------------------------------------------------------------------- #


def test_default_sweep_is_three_deltas_around_the_operating_centre():
    sweep = delta_sweep()
    assert len(sweep) == MIN_PLATEAU_DELTA_COUNT >= 3
    assert sweep[1] == pytest.approx(DEFAULT_DELTA_CENTER_ANG)
    assert sweep[0] < sweep[1] < sweep[2]
    assert sweep[1] / sweep[0] == pytest.approx(sweep[2] / sweep[1])


@pytest.mark.parametrize("kwargs", [{"count": 2}, {"count": 4}, {"ratio": 1.0}, {"center_ang": 0.0}])
def test_sweep_rejects_shapes_that_cannot_show_a_plateau(kwargs):
    with pytest.raises(FdPerturbationError):
        delta_sweep(**{"center_ang": 0.01, **kwargs})


def test_single_delta_is_flagged_as_unable_to_show_a_plateau():
    status = delta_sweep_status([0.01])
    assert status["delta_plateau_ready"] is False
    assert status["delta_sweep_status"] == "insufficient_for_plateau"
    assert delta_sweep_status(delta_sweep())["delta_plateau_ready"] is True


def test_delta_is_selected_from_the_observed_plateau():
    # Flat at 0.005-0.02, truncation error takes over at 0.04.
    selection = select_delta_from_plateau({0.005: 1.00, 0.01: 1.01, 0.02: 1.02, 0.04: 1.60})
    assert selection["plateau_delta_ang"] == [0.005, 0.01, 0.02]
    # Largest amplitude inside the plateau: best signal over the FD noise floor.
    assert selection["selected_delta_ang"] == 0.02


def test_no_plateau_means_no_delta_rather_than_a_default():
    with pytest.raises(FdPerturbationError, match="No plateau"):
        select_delta_from_plateau({0.005: 1.0, 0.01: 2.0, 0.02: 4.0})
    with pytest.raises(FdPerturbationError, match="at least"):
        select_delta_from_plateau({0.005: 1.0, 0.01: 1.0})


# --------------------------------------------------------------------------- #
# Neighbour topology
# --------------------------------------------------------------------------- #


def test_small_delta_preserves_topology_and_reports_the_margin():
    margin = topology_margin(
        DIMER_POSITIONS,
        direction=one_hot(2, 1, "y"),
        delta_ang=0.1,
        cutoff_ang=2.0,
        lattice_vectors_ang=DIMER_CELL,
    )
    assert margin.preserved
    assert margin.crossing_pairs == ()
    assert margin.neighbor_pair_count == 1
    assert margin.cutoff_mode == "pair_cutoff"
    assert sorted(margin.geometries) == ["base", "minus", "plus"]
    # Closest approach to the 2.0 Ang cutoff surface: the displaced dimer.
    assert margin.min_margin_ang == pytest.approx(2.0 - math.hypot(1.42, 0.1), abs=1e-12)
    assert margin.max_pair_distance_change_ang == pytest.approx(math.hypot(1.42, 0.1) - 1.42, abs=1e-12)
    assert assert_topology_preserved(margin) is margin


def test_a_perturbation_that_crosses_the_cutoff_is_refused():
    margin = topology_margin(
        DIMER_POSITIONS,
        direction=one_hot(2, 1, "x"),
        delta_ang=0.6,
        cutoff_ang=2.0,
        lattice_vectors_ang=DIMER_CELL,
    )
    assert not margin.preserved
    crossing = margin.crossing_pairs[0]
    assert crossing["in_neighbor_list"] == {"base": True, "plus": False, "minus": True}
    with pytest.raises(FdPerturbationError, match="neighbour topology"):
        assert_topology_preserved(margin, context="graphene gamma")


def test_a_crossing_through_a_periodic_image_is_caught():
    # Cell 4.0 Ang: the image pair sits at 2.58 Ang, just inside a 2.6 cutoff,
    # and only the -delta geometry pushes it out. Nothing changes in the cell.
    margin = topology_margin(
        DIMER_POSITIONS,
        direction=one_hot(2, 1, "x"),
        delta_ang=0.1,
        cutoff_ang=2.6,
        lattice_vectors_ang=[[4.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
    )
    assert not margin.preserved
    images = {tuple(entry["image"]) for entry in margin.crossing_pairs}
    assert images == {(1, 0, 0)}
    crossing = margin.crossing_pairs[0]
    assert crossing["distances_ang"]["minus"] == pytest.approx(2.68, abs=1e-9)
    assert crossing["in_neighbor_list"]["minus"] is False


def test_periodic_images_are_not_double_counted():
    # Each of the two atoms has exactly one +x and one -x image partner within a
    # 2.5 Ang cutoff of a 4.0 Ang chain; the lexicographic half counts each once.
    margin = topology_margin(
        [[0.0, 0.0, 0.0]],
        direction=one_hot(1, 0, "y"),
        delta_ang=0.01,
        cutoff_ang=4.5,
        lattice_vectors_ang=[[4.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
    )
    # Images (1,0,0) and (2,0,0) at 4.0 and 8.0 Ang: only the first is inside.
    assert margin.neighbor_pair_count == 1
    assert margin.preserved


def test_uniform_translation_can_never_change_topology():
    margin = topology_margin(
        DIMER_POSITIONS,
        direction=uniform_translation(2, "x"),
        delta_ang=0.5,
        cutoff_ang=2.0,
        lattice_vectors_ang=DIMER_CELL,
    )
    assert margin.preserved
    assert margin.max_pair_distance_change_ang == pytest.approx(0.0, abs=1e-12)


def test_per_atom_radii_use_the_siesta_pair_rule():
    margin = topology_margin(
        DIMER_POSITIONS,
        direction=one_hot(2, 1, "x"),
        delta_ang=0.01,
        cutoff_ang=[0.7, 0.8],
        lattice_vectors_ang=DIMER_CELL,
    )
    assert margin.cutoff_mode == "per_atom_radius_sum"
    # r_i + r_j = 1.5 < 1.42? no: the pair sits just inside 1.5 Ang.
    assert margin.neighbor_pair_count == 1
    assert margin.min_margin_ang == pytest.approx(1.5 - 1.43, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"delta_ang": 0.0}, "delta_ang must be positive"),
        ({"cutoff_ang": [1.0, 2.0, 3.0]}, "scalar or one radius per atom"),
        ({"cutoff_ang": 0.0}, "cutoff_ang must be positive"),
        ({"lattice_vectors_ang": [[1.0, 0.0, 0.0]]}, r"shape \(3, 3\)"),
        ({"lattice_vectors_ang": [[1.0, 0.0, 0.0]] * 3}, "singular"),
    ],
)
def test_topology_margin_rejects_ill_posed_requests(kwargs, match):
    request = {
        "direction": one_hot(2, 1, "x"),
        "delta_ang": 0.01,
        "cutoff_ang": 2.0,
        "lattice_vectors_ang": DIMER_CELL,
        **kwargs,
    }
    with pytest.raises(FdPerturbationError, match=match):
        topology_margin(DIMER_POSITIONS, **request)


# --------------------------------------------------------------------------- #
# Cutoff radii from the real orbital contract
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not GRAPHENE_FDF.is_file(), reason="graphene material inputs are not in this checkout")
def test_graphene_pao_radii_come_from_the_real_orbital_contract():
    from orbital_contract import build_orbital_contract

    contract = build_orbital_contract("graphene", GRAPHENE_FDF, GRAPHENE_BASIS)
    radii = atom_cutoff_radii_ang(contract)
    assert len(radii) == contract["atom_count"] == 2
    # Largest C PAO cutoff of the shipped C.ion.xml, converted from Bohr.
    largest_bohr = max(
        float(orbital["cutoff_bohr"]) for species in contract["species"] for orbital in species["orbitals"]
    )
    assert radii[0] == pytest.approx(largest_bohr * 0.529177210903, rel=1e-12)
    assert radii == [radii[0]] * 2

    # A collective displacement at the top of the sweep keeps the 5.15 Ang pair
    # list of graphene -- but by only ~0.01 Ang: with a PAO cutoff that large the
    # margin is what decides, which is exactly why it is reported and not assumed.
    delta = max(delta_sweep())
    direction = random_collective(2, seed=2)
    margin = topology_margin(
        GRAPHENE_POSITIONS,
        direction=direction,
        delta_ang=delta,
        cutoff_ang=radii,
        lattice_vectors_ang=GRAPHENE_CELL,
    )
    assert margin.preserved
    assert margin.cutoff_mode == "per_atom_radius_sum"
    assert margin.neighbor_pair_count > 0
    assert 0.0 < margin.min_margin_ang < 0.05
    # No pair can move by more than 2 * delta * max|v_i|.
    assert margin.max_pair_distance_change_ang <= 2.0 * delta * direction.max_atom_norm + 1e-12


def test_orbital_contract_without_cutoffs_is_refused():
    with pytest.raises(FdPerturbationError, match="no finite orbital cutoff"):
        atom_cutoff_radii_ang({"species": [{"label": "C", "orbitals": []}], "atoms": [{"species_label": "C"}]})
    with pytest.raises(FdPerturbationError, match="carries no atom list"):
        atom_cutoff_radii_ang({"species": [{"label": "C", "orbitals": [{"cutoff_bohr": 4.0}]}], "atoms": []})
    with pytest.raises(FdPerturbationError, match="No basis cutoff"):
        atom_cutoff_radii_ang(
            {"species": [{"label": "C", "orbitals": [{"cutoff_bohr": 4.0}]}], "atoms": [{"species_label": "H"}]}
        )
