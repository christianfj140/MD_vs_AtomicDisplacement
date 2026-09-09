"""E-F_001-S34: the AB bilayer direction/layer geometry, isolated from SIESTA and torch.

Only the pure-geometry pieces are exercised here (layer labelling, the three
directions, mode matching): the raw derivatives, basis response and phonons
are measured by the run itself against the material's own SIESTA campaign and
written into its report, the same split ``test_run_graphene_gamma_epc_paths.py``
uses for the graphene runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

import run_ab_bilayer_epc_paths as ab  # noqa: E402

AB_FDF = REPO_ROOT / "materials/bilayer_graphene_AB/RUN.fdf"


def _ab_positions() -> np.ndarray:
    from fdf_materialization import extract_fdf_structure

    return np.asarray(extract_fdf_structure(AB_FDF).positions_ang, dtype=np.float64)


def test_layer_of_atom_from_positions_matches_the_material():
    positions = _ab_positions()
    layers = ab.layer_of_atom_from_positions(positions)
    assert layers.tolist() == [0, 0, 1, 1]


def test_layer_of_atom_from_positions_refuses_a_single_layer():
    positions = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.7, 1.2, 0.0]])
    with pytest.raises(ab.AbBilayerEpcError):
        ab.layer_of_atom_from_positions(positions)


def test_build_directions_are_unit_frobenius_and_layer_consistent():
    positions = _ab_positions()
    cell = np.array([[2.48, 0.0, 0.0], [-1.24, 2.147743, 0.0], [0.0, 0.0, 20.0]])
    layers = ab.layer_of_atom_from_positions(positions)
    directions = ab.build_directions(positions, cell, layers)
    assert set(directions) == {"shear", "layer_breathing", "intralayer_control"}
    for direction in directions.values():
        assert direction.vectors.shape == (4, 3)
        assert np.isclose(np.linalg.norm(direction.vectors), 1.0)

    shear = directions["shear"].vectors
    assert np.allclose(shear[layers == 0, 1:], 0.0)
    assert np.allclose(shear[layers == 1, 1:], 0.0)
    assert np.all(shear[layers == 0, 0] > 0.0)
    assert np.all(shear[layers == 1, 0] < 0.0)

    breathing = directions["layer_breathing"].vectors
    assert np.all(breathing[layers == 0, 2] < 0.0)
    assert np.all(breathing[layers == 1, 2] > 0.0)

    control = directions["intralayer_control"].vectors
    assert np.allclose(control[layers == 1], 0.0), "the control pattern leaves the other layer at rest"
    assert np.linalg.norm(control[layers == 0]) > 0.0


def test_match_mode_picks_the_branch_with_maximum_overlap():
    class ToyModeSet:
        frequencies_ev = np.array([0.001, 0.05, 0.2])
        eigenvectors = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ]
        )

    import fd_perturbation_space as fdp

    direction = fdp.collective(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]), name="probe")
    match = ab.match_mode(direction, ToyModeSet())
    assert match["branch"] == 1
    assert match["overlap"] == pytest.approx(1.0)
