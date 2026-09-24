from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import w90_displacement_sampler_family as sampler  # noqa: E402


def hex_supercell_geometry(n_cells: int = 5) -> sampler.Geometry:
    """Synthetic n_cells x n_cells graphene supercell (2 atoms/cell), no fdf/sisl I/O."""

    a1 = np.array([2.2005, -1.27045927, 0.0])
    a2 = np.array([2.2005, 1.27045927, 0.0])
    a3 = np.array([0.0, 0.0, 29.34])
    basis = np.array([[0.0, 0.0, 0.0], [0.4899, -0.4899, 0.0]]) @ np.array([a1, a2, a3])
    positions = []
    for i in range(n_cells):
        for j in range(n_cells):
            shift = i * a1 + j * a2
            for atom in basis:
                positions.append(atom + shift)
    lattice = np.array([a1 * n_cells, a2 * n_cells, a3])
    positions = np.array(positions)
    return sampler.Geometry(
        positions_ang=positions,
        lattice_ang=lattice,
        atomic_numbers=tuple(6 for _ in positions),
    )


@pytest.fixture(scope="module")
def graphene_primitive() -> sampler.Geometry:
    return sampler.load_graphene_primitive()


@pytest.fixture(scope="module")
def graphene_5x5() -> sampler.Geometry:
    return sampler.load_graphene_5x5()


# --------------------------------------------------------------------------
# axial_radial: radial-shell + k=1 correctness at R=0.03, 2D_in and 3D
# --------------------------------------------------------------------------


def test_axial_radial_2d_in_radius_and_directions(graphene_primitive):
    configs = sampler.generate_axial_radial(graphene_primitive, k=1, dim="2D_in", radii_ang=(0.03,))
    assert len(configs) == 4
    for config in configs:
        (vector,) = config.displacements_ang.values()
        norm = math.sqrt(sum(component**2 for component in vector))
        assert norm == pytest.approx(0.03, abs=1e-9)
        assert vector[2] == pytest.approx(0.0, abs=1e-12)
        assert config.metadata["k"] == 1
        assert config.metadata["k_over_n"] == pytest.approx(1 / graphene_primitive.n_atoms)
        assert config.metadata["participation_ratio"] == pytest.approx(1 / graphene_primitive.n_atoms)
        assert config.metadata["connected"] is True
        assert config.metadata["max_displacement_ang"] == pytest.approx(0.03, abs=1e-9)
    directions = {tuple(np.round(c.metadata["direction"], 6)) for c in configs}
    expected = {
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
    }
    assert directions == expected


def test_axial_radial_3d_radius_and_directions(graphene_primitive):
    configs = sampler.generate_axial_radial(graphene_primitive, k=1, dim="3D", radii_ang=(0.03,))
    assert len(configs) == 6
    for config in configs:
        (vector,) = config.displacements_ang.values()
        norm = math.sqrt(sum(component**2 for component in vector))
        assert norm == pytest.approx(0.03, abs=1e-9)
    directions = {tuple(np.round(c.metadata["direction"], 6)) for c in configs}
    assert len(directions) == 6
    assert all(np.count_nonzero(np.array(d)) == 1 for d in directions)


def test_axial_radial_declared_shells_match_all_amplitudes(graphene_primitive):
    configs = sampler.generate_axial_radial(graphene_primitive, k=1, dim="3D")
    radii = sorted({config.metadata["amplitude_ang"] for config in configs})
    assert radii == sorted(sampler.ALL_AMPLITUDES_ANG)
    assert len(configs) == len(sampler.ALL_AMPLITUDES_ANG) * 6


# --------------------------------------------------------------------------
# angular_shell: uniform (2D) / quasi-uniform (3D) angular coverage at R=0.03
# --------------------------------------------------------------------------


def test_angular_shell_2d_in_is_uniform(graphene_primitive):
    n_points = 12
    configs = sampler.generate_angular_shell(graphene_primitive, k=1, dim="2D_in", radius_ang=0.03, n_points=n_points)
    assert len(configs) == n_points
    angles = []
    for config in configs:
        (vector,) = config.displacements_ang.values()
        assert vector[2] == pytest.approx(0.0, abs=1e-12)
        norm = math.sqrt(sum(component**2 for component in vector))
        assert norm == pytest.approx(0.03, abs=1e-9)
        angles.append(math.atan2(vector[1], vector[0]) % (2 * math.pi))
    angles.sort()
    gaps = np.diff(angles + [angles[0] + 2 * math.pi])
    expected_gap = 2 * math.pi / n_points
    assert np.allclose(gaps, expected_gap, atol=1e-9)


def test_angular_shell_3d_is_quasi_uniform(graphene_primitive):
    n_points = 64
    configs = sampler.generate_angular_shell(graphene_primitive, k=1, dim="3D", radius_ang=0.03, n_points=n_points)
    assert len(configs) == n_points
    vectors = np.array([list(config.displacements_ang.values())[0] for config in configs])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 0.03, atol=1e-9)
    # Quasi-uniform sphere coverage: z spans close to [-R, R] and the mean is ~0
    # (no clustering at either pole), unlike a naive single-axis or great-circle sample.
    z_values = vectors[:, 2]
    assert z_values.max() > 0.9 * 0.03
    assert z_values.min() < -0.9 * 0.03
    assert abs(float(z_values.mean())) < 0.05 * 0.03


def test_angular_shell_rejects_1d_dimensionality(graphene_primitive):
    with pytest.raises(ValueError):
        sampler.generate_angular_shell(graphene_primitive, k=1, dim="1D_in", radius_ang=0.03)


# --------------------------------------------------------------------------
# active-atom selection: k, k/N, connectedness metadata
# --------------------------------------------------------------------------


def test_select_active_atoms_k1_is_trivially_connected(graphene_primitive):
    active = sampler.select_active_atoms(graphene_primitive, 1, center_index=0)
    assert active.indices == (0,)
    assert active.connected is True
    assert active.shell_span == 0
    assert active.pair_distance_ang is None


def test_select_active_atoms_k2_bonded_pair_is_first_shell_and_connected(graphene_primitive):
    active = sampler.select_active_atoms(graphene_primitive, 2, center_index=0, pair_mode="bonded")
    assert active.k == 2
    assert active.connected is True
    assert active.shell_span == 1
    assert active.pair_distance_ang == pytest.approx(1.467, abs=0.05)


def test_select_active_atoms_k2_arbitrary_pair_is_disconnected(graphene_5x5):
    bonded = sampler.select_active_atoms(graphene_5x5, 2, center_index=0, pair_mode="bonded")
    arbitrary = sampler.select_active_atoms(graphene_5x5, 2, center_index=0, pair_mode="arbitrary")
    assert arbitrary.connected is False
    assert arbitrary.pair_distance_ang > bonded.pair_distance_ang
    assert arbitrary.shell_span > bonded.shell_span


def test_select_active_atoms_k3_is_out_of_scope(graphene_primitive):
    with pytest.raises(ValueError):
        sampler.select_active_atoms(graphene_primitive, 3)


def test_select_active_atoms_k_equals_n_atoms_selects_every_atom(graphene_5x5):
    active = sampler.select_active_atoms(graphene_5x5, 50)
    assert active.indices == tuple(range(50))
    assert active.k == 50
    assert active.connected is True
    assert active.pair_distance_ang is None


def test_select_active_atoms_arbitrary_k_is_nested_connected(graphene_5x5):
    selections = [sampler.select_active_atoms(graphene_5x5, k) for k in (2, 5, 12, 32, 50)]
    assert all(selection.connected for selection in selections)
    assert all(set(left.indices) < set(right.indices) for left, right in zip(selections, selections[1:]))
    assert [selection.k for selection in selections] == [2, 5, 12, 32, 50]


# --------------------------------------------------------------------------
# 5x5 compatibility: N > 2 active-subset selection without N=2 assumptions
# --------------------------------------------------------------------------


def test_compatibility_preview_accepts_supercell_without_n2_assumption(graphene_5x5):
    assert graphene_5x5.n_atoms == 50
    preview_k1 = sampler.compatibility_preview(graphene_5x5, 1, center_index=10)
    preview_k2 = sampler.compatibility_preview(graphene_5x5, 2, center_index=10)
    assert preview_k1["n_atoms_total"] == 50
    assert preview_k2["n_atoms_total"] == 50
    assert preview_k1["k"] == 1
    assert preview_k2["k"] == 2
    assert preview_k2["connected"] is True
    assert 10 in preview_k2["active_atom_indices"]


def test_compatibility_preview_on_synthetic_supercell():
    geometry = hex_supercell_geometry(n_cells=6)
    assert geometry.n_atoms == 72
    preview = sampler.compatibility_preview(geometry, 2, center_index=40)
    assert preview["n_atoms_total"] == 72
    assert preview["connected"] is True


# --------------------------------------------------------------------------
# local_pair_modes: 5 modes, correlation sign matches the mode's physics
# --------------------------------------------------------------------------


def test_local_pair_modes_cover_five_mode_types_and_correlations(graphene_primitive):
    configs = sampler.generate_local_pair_modes(graphene_primitive, dim="3D", amplitudes_ang=(0.03,))
    modes = {config.metadata["mode"] for config in configs}
    assert modes == {"longitudinal", "transverse_ip", "transverse_opp", "parallel", "antiparallel"}
    by_mode: dict[str, list] = {}
    for config in configs:
        by_mode.setdefault(config.metadata["mode"], []).append(config)

    for mode in ("longitudinal", "transverse_ip", "transverse_opp"):
        for config in by_mode[mode]:
            assert config.metadata["correlation"] == pytest.approx(-1.0, abs=1e-9)
            assert config.metadata["connected"] is True

    for config in by_mode["parallel"]:
        assert config.metadata["correlation"] == pytest.approx(1.0, abs=1e-9)
    for config in by_mode["antiparallel"]:
        assert config.metadata["correlation"] == pytest.approx(-1.0, abs=1e-9)


def test_local_pair_modes_amplitude_sets_displacement_magnitude(graphene_primitive):
    configs = sampler.generate_local_pair_modes(graphene_primitive, dim="3D", amplitudes_ang=(0.05,))
    for config in configs:
        for vector in config.displacements_ang.values():
            norm = math.sqrt(sum(component**2 for component in vector))
            assert norm == pytest.approx(0.05, abs=1e-9)


# --------------------------------------------------------------------------
# sobol_sparse + nested learning curve prefixes
# --------------------------------------------------------------------------


def test_sobol_sparse_reproducible_and_bounded(graphene_primitive):
    first = sampler.generate_sobol_sparse(graphene_primitive, k=1, dim="3D", amplitude_ang=0.05, max_n=32, seed=7)
    second = sampler.generate_sobol_sparse(graphene_primitive, k=1, dim="3D", amplitude_ang=0.05, max_n=32, seed=7)
    for a, b in zip(first, second):
        assert a.displacements_ang == b.displacements_ang
    for config in first:
        for vector in config.displacements_ang.values():
            assert all(abs(component) <= 0.05 + 1e-12 for component in vector)


def test_nested_prefixes_are_subsets_for_sobol_and_axial_radial(graphene_primitive):
    sobol_configs = sampler.generate_sobol_sparse(
        graphene_primitive, k=1, dim="3D", amplitude_ang=0.05, max_n=256, seed=3
    )
    prefixes = sampler.nested_learning_curve_prefixes(sobol_configs)
    assert set(prefixes) == set(sampler.NESTED_LEARNING_CURVE_SIZES)
    for size, prefix in prefixes.items():
        assert prefix == sobol_configs[:size]
        assert len(prefix) == size

    axial_configs = sampler.generate_axial_radial(graphene_primitive, k=1, dim="3D")
    axial_prefixes = sampler.nested_learning_curve_prefixes(axial_configs, sizes=(6, 12, 18))
    assert axial_prefixes[6] == axial_configs[:6]
    assert axial_prefixes[6] == axial_prefixes[12][:6]
    assert axial_prefixes[12] == axial_prefixes[18][:12]


# --------------------------------------------------------------------------
# latin_hypercube: frozen-test-only sampler (S5 common_displacement)
# --------------------------------------------------------------------------


def test_latin_hypercube_reproducible_and_amplitude_in_range(graphene_primitive):
    first = sampler.generate_latin_hypercube(graphene_primitive, k=1, dim="3D", n_structures=16, seed=11)
    second = sampler.generate_latin_hypercube(graphene_primitive, k=1, dim="3D", n_structures=16, seed=11)
    assert len(first) == 16
    for a, b in zip(first, second):
        assert a.displacements_ang == b.displacements_ang
    lo, hi = sampler.COMMON_DISPLACEMENT_AMPLITUDE_RANGE_ANG
    for config in first:
        assert lo <= config.metadata["amplitude_ang"] <= hi
        for vector in config.displacements_ang.values():
            assert all(abs(component) <= hi + 1e-9 for component in vector)


def test_latin_hypercube_mixes_amplitudes_and_differs_from_sobol(graphene_primitive):
    lhs_configs = sampler.generate_latin_hypercube(graphene_primitive, k=2, dim="2D_in", n_structures=24, seed=5)
    amplitudes = {round(config.metadata["amplitude_ang"], 6) for config in lhs_configs}
    assert len(amplitudes) > 1  # amplitude is sampled per-structure, not fixed like every training family

    sobol_configs = sampler.generate_sobol_sparse(graphene_primitive, k=2, dim="2D_in", amplitude_ang=0.05, max_n=24, seed=5)
    lhs_vectors = [config.displacements_ang for config in lhs_configs]
    sobol_vectors = [config.displacements_ang for config in sobol_configs]
    assert lhs_vectors != sobol_vectors


# --------------------------------------------------------------------------
# random_cartesian: amplitude/dimensionality controls + reproducibility
# --------------------------------------------------------------------------


def test_random_cartesian_respects_amplitude_and_dimensionality(graphene_primitive):
    configs = sampler.generate_random_cartesian(
        graphene_primitive, k=1, dim="1D_z", amplitude_ang=0.08, n_structures=20, seed=42
    )
    assert len(configs) == 20
    for config in configs:
        (vector,) = config.displacements_ang.values()
        assert vector[0] == pytest.approx(0.0, abs=1e-12)
        assert vector[1] == pytest.approx(0.0, abs=1e-12)
        assert abs(vector[2]) <= 0.08 + 1e-9


def test_random_cartesian_seed_is_reproducible(graphene_primitive):
    first = sampler.generate_random_cartesian(
        graphene_primitive, k=2, dim="2D_in", amplitude_ang=0.03, n_structures=5, seed=99
    )
    second = sampler.generate_random_cartesian(
        graphene_primitive, k=2, dim="2D_in", amplitude_ang=0.03, n_structures=5, seed=99
    )
    for a, b in zip(first, second):
        assert a.displacements_ang == b.displacements_ang


# --------------------------------------------------------------------------
# MD family: contract only, no frame generation
# --------------------------------------------------------------------------


def test_md_family_contract_does_not_generate_frames():
    contract = sampler.md_family_contract()
    assert contract["status"] == "not_generated_by_this_module"
    assert contract["generator"] == "MD/scripts/generate_md_dataset.py"


if __name__ == "__main__":
    import unittest

    unittest.main()
