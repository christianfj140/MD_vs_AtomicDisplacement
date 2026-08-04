import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))

from evaluate_checkpoint_spectral_metrics import (  # noqa: E402
    inverse_sqrt,
    subspace_metrics,
    selection_verdict,
    single_offset_metrics,
    rmse_with_offset,
    frozen_offsets,
)


def test_inverse_sqrt_whitens_the_overlap() -> None:
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(8, 8))
    overlap = basis @ basis.T + 8.0 * np.eye(8)
    root, minimum, condition = inverse_sqrt(overlap)
    assert np.allclose(root @ overlap @ root, np.eye(8), atol=1e-10)
    eigenvalues = np.linalg.eigvalsh(overlap)
    assert minimum == pytest.approx(eigenvalues.min())
    assert condition == pytest.approx(eigenvalues.max() / eigenvalues.min())


def test_inverse_sqrt_rejects_indefinite_overlap() -> None:
    with pytest.raises(RuntimeError, match="positive definite"):
        inverse_sqrt(np.diag([1.0, -1.0]))


def test_subspace_metrics_isolate_a_rigid_shift_and_the_first_order_shift() -> None:
    reference = np.array([-9.0, -0.3, -0.1, 0.2, 0.4, 8.0])
    predicted = reference + 0.05  # rigid shift only
    vectors = np.eye(6)
    delta_h = np.diag([0.0, 0.05, 0.05, 0.05, 0.05, 0.0])
    selection = np.argsort(np.abs(reference))[:4]
    metrics = subspace_metrics(reference, predicted, vectors, delta_h, selection)
    assert metrics["n_states"] == 4
    # The window must exclude the +/-9 eV states, so the span stays small.
    assert metrics["reference_min_eV"] == pytest.approx(-0.3)
    assert metrics["reference_max_eV"] == pytest.approx(0.4)
    assert metrics["rmse_eV"] == pytest.approx(0.05)
    assert metrics["global_shift_eV"] == pytest.approx(-0.05)
    assert metrics["aligned_rmse_eV"] == pytest.approx(0.0, abs=1e-12)
    # First-order shifts are the eigenvalues of C^dag dH C.
    assert metrics["projected_max_abs_first_order_shift_eV"] == pytest.approx(0.05)


def test_first_order_shifts_do_not_depend_on_the_basis_inside_a_degenerate_subspace() -> None:
    """Rotating the two degenerate vectors must not change the reported shifts."""
    reference = np.array([-0.1, -0.1, 5.0])
    predicted = reference.copy()
    delta_h = np.array([[0.02, 0.03, 0.0], [0.03, -0.02, 0.0], [0.0, 0.0, 0.5]])
    selection = np.array([0, 1])
    plain = subspace_metrics(reference, predicted, np.eye(3), delta_h, selection)
    angle = 0.7
    rotation = np.eye(3)
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rotated = subspace_metrics(reference, predicted, rotation, delta_h, selection)
    assert plain["projected_first_order_shifts_eV"] == pytest.approx(
        rotated["projected_first_order_shifts_eV"]
    )
    # The diagonal, by contrast, does move -- which is why it is not reported.
    assert np.diag(delta_h)[:2] != pytest.approx(
        np.diag(rotation[:, :2].T @ delta_h @ rotation[:, :2])
    )


def test_subspace_metrics_handles_an_empty_selection() -> None:
    metrics = subspace_metrics(np.array([1.0]), np.array([1.0]), np.eye(1),
                               np.zeros((1, 1)), np.array([], dtype=int))
    assert metrics == {"n_states": 0}


def test_selection_ignores_kpoints_without_frontier_states() -> None:
    report = {
        "checkpoints": {
            "a.ckpt": {"per_sample": [
                {"kpoint": "K", "frontier_n_states": 4, "frontier_rmse_eV": 0.03},
                {"kpoint": "Gamma", "frontier_n_states": 4, "frontier_rmse_eV": 9.0},
            ]},
            "b.ckpt": {"per_sample": [
                {"kpoint": "K", "frontier_n_states": 4, "frontier_rmse_eV": 0.01},
                {"kpoint": "Gamma", "frontier_n_states": 4, "frontier_rmse_eV": 0.5},
            ]},
        }
    }
    verdict = selection_verdict(report)
    # Gamma must not leak in: a.ckpt scores its K value alone, not the 9 eV.
    assert verdict["mean_eV"]["a.ckpt"] == pytest.approx(0.03)
    assert verdict["spectrally_best"] == "b.ckpt"


def test_selection_flags_a_difference_too_small_to_act_on() -> None:
    report = {
        "checkpoints": {
            "a.ckpt": {"per_sample": [{"kpoint": "K", "frontier_n_states": 4, "frontier_rmse_eV": 0.0663}]},
            "b.ckpt": {"per_sample": [{"kpoint": "K", "frontier_n_states": 4, "frontier_rmse_eV": 0.0665}]},
        }
    }
    assert selection_verdict(report)["separation_is_meaningful"] is False


def _row(rmse, global_shift, kpoint="K"):
    return {"kpoint": kpoint, "frontier_n_states": 4,
            "frontier_rmse_eV": rmse, "frontier_global_shift_eV": global_shift,
            "frontier_aligned_rmse_eV": 0.0}


def test_offset_sign_removes_a_pure_bias() -> None:
    """A constant +30 meV error must be cancelled by the fitted offset, not doubled."""
    delta = np.full(4, 0.030)
    rows = [_row(float(np.sqrt(np.mean(delta**2))), -float(np.mean(delta)))]
    metrics = single_offset_metrics(rows)
    assert metrics["offset_eV"] == pytest.approx(0.030)
    assert metrics["rmse_raw_eV"] == pytest.approx(0.030)
    # Sign error would give 60 meV instead of 0.
    assert metrics["rmse_with_this_split_offset_eV"] == pytest.approx(0.0, abs=1e-12)


def test_rmse_with_offset_matches_a_direct_computation() -> None:
    """The closed form must agree with recomputing from the raw errors."""
    rng = np.random.default_rng(3)
    deltas = [rng.normal(0.04, 0.01, size=4) for _ in range(5)]
    rows = [_row(float(np.sqrt(np.mean(d**2))), -float(np.mean(d))) for d in deltas]
    c = 0.037
    direct = np.sqrt(np.mean([np.mean((d - c) ** 2) for d in deltas]))
    assert rmse_with_offset(rows, c) == pytest.approx(direct)


def test_frozen_offset_is_per_checkpoint_and_survives_the_round_trip(tmp_path) -> None:
    """validation -> freeze -> test must carry one offset per checkpoint, not a shared one."""
    validation = {"checkpoints": {
        "a.ckpt": {"single_offset": {"offset_eV": 0.041}},
        "b.ckpt": {"single_offset": {"offset_eV": 0.017}},
    }}
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(validation), encoding="utf-8")
    offsets = frozen_offsets(path)
    assert offsets == {"a.ckpt": pytest.approx(0.041), "b.ckpt": pytest.approx(0.017)}
    # Applying a.ckpt's offset to b.ckpt is exactly the footgun this replaced:
    # it must give a different (worse) number than b's own.
    rows_b = [_row(0.020, -0.017)]
    assert rmse_with_offset(rows_b, offsets["b.ckpt"]) < rmse_with_offset(rows_b, offsets["a.ckpt"])


def test_frozen_offsets_rejects_a_report_without_calibration(tmp_path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"checkpoints": {"a.ckpt": {}}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no single_offset"):
        frozen_offsets(path)
    assert frozen_offsets(None) == {}


def test_registry_distance_uses_the_hexagonal_metric_not_the_fractional_one() -> None:
    """A Euclidean distance on fractional coordinates ranks registries wrongly."""
    from build_registry_grid_probe import registry_distance_ang

    cell = np.array([[2.48, 0.0, 0.0], [-1.24, 2.1477, 0.0], [0.0, 0.0, 20.0]])
    # AA, AB and BA are seen registries, so their distance must be exactly zero.
    for seen in [(0.0, 0.0), (1 / 3, 2 / 3), (2 / 3, 1 / 3)]:
        assert registry_distance_ang(seen, cell) == pytest.approx(0.0, abs=1e-9)
    # (0.5,0.5) and (0.5,0.0) differ in the fractional metric (0.236 vs 0.373)
    # but the hexagonal cell puts them at the same Cartesian distance. The
    # tolerance is 1e-4 because the fdf stores a2 as 2.1477, so the cell is only
    # hexagonal to five decimals; a wrong metric would differ by ~50%.
    assert registry_distance_ang((0.5, 0.5), cell) == pytest.approx(
        registry_distance_ang((0.5, 0.0), cell), rel=1e-4
    )
    # Periodicity: shifting by a whole lattice vector changes nothing.
    assert registry_distance_ang((1.25, 0.5), cell) == pytest.approx(
        registry_distance_ang((0.25, 0.5), cell)
    )


def test_symmetry_fingerprint_is_invariant_under_the_honeycomb_rotations() -> None:
    """Registries related by C3/C2 are physically the same and must share a class."""
    from build_registry_grid_probe import symmetry_fingerprint

    cell = np.array([[2.48, 0.0, 0.0], [-1.24, 2.1477, 0.0], [0.0, 0.0, 20.0]])
    basis = cell[:2, :2].T
    inverse = np.linalg.inv(basis)

    def rotate(fractional, theta):
        matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return tuple((inverse @ (matrix @ (basis @ np.array(fractional)))) % 1.0)

    for point in [(0.25, 0.5), (1 / 12, 5 / 12), (0.5, 0.5), (1 / 3, 2 / 3)]:
        reference = symmetry_fingerprint(point, cell)
        for theta in (2 * np.pi / 3, 4 * np.pi / 3, np.pi):
            assert symmetry_fingerprint(rotate(point, theta), cell) == reference


def test_symmetry_classes_partition_the_grid_into_hexagonal_orbits() -> None:
    """A 12x12 registry grid must collapse to orbits of size 1, 3, 6 or 12."""
    from collections import Counter

    from build_registry_grid_probe import grid_points, symmetry_fingerprint

    cell = np.array([[2.48, 0.0, 0.0], [-1.24, 2.1477, 0.0], [0.0, 0.0, 20.0]])
    classes: dict = {}
    for point in grid_points(12):
        classes.setdefault(symmetry_fingerprint(point, cell), []).append(point)
    assert sum(len(v) for v in classes.values()) == 144
    # Orbits of a hexagonal registry space: 1 (AA, full stabiliser), 2 (the AB/BA
    # pair, 3-fold stabiliser each), 3 (the saddle registries) and the generic 6/12.
    assert set(Counter(len(v) for v in classes.values())) <= {1, 2, 3, 6, 12}
    assert classes[symmetry_fingerprint((1 / 3, 2 / 3), cell)] == pytest.approx(
        classes[symmetry_fingerprint((2 / 3, 1 / 3), cell)]
    ), "AB and BA are C2 images of each other and must share a class"
    # Far fewer classes than points: that collapse is the whole point of the split.
    assert len(classes) < 30
