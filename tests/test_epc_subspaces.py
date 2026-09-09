"""C16 / E-F_001-S19: GO-6 gauge-invariant subspace metrics in the S metric.

What has to hold before graphene K or any TBG validation is allowed to name a
band: inside a cluster the band index is gauge, so every claim must survive
``C -> C U``. These tests do that on both kinds of input the roadmap names —
toy pencils with an *exact* degeneracy (built spectrum-first, because a random
H never has one) and the eigenspaces C15 actually persists — and they check the
two things that make the machinery trustworthy rather than merely invariant:

* the clusters come from the resolution the residual and the window
  conditioning imply, never from an index or a hand-picked eV;
* a rotation that crosses a *resolved* gap is not a gauge and is caught.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from epc_subspaces import (  # noqa: E402
    ENERGY_RESOLUTION_POLICY,
    GATE_ID,
    METRIC_TOLERANCE_POLICY,
    SubspaceGateError,
    block_metrics,
    certify,
    cluster_groups,
    cross_cluster_mixing,
    eigenvalue_resolutions_ev,
    main,
    near_degenerate_clusters,
    random_gauge_rotation,
    require_go6,
    subspace_metric_tolerances,
    subspace_metrics,
    subspace_overlap,
    window_metric,
)
from run_deeph_sparse_spectrum import degenerate_subspace_labels  # noqa: E402

from test_eigenspace_persistence import _pencil, write_eigenspace  # noqa: E402

# One exactly degenerate pair at 0.1 eV among otherwise well separated states.
DEGENERATE_SPECTRUM = np.array([-3.0, -0.2, 0.1, 0.1, 0.5, 4.0])
SIZE = DEGENERATE_SPECTRUM.size


def toy_window(seed: int = 11, spectrum: np.ndarray = DEGENERATE_SPECTRUM) -> dict:
    """An S-orthonormal window of a toy pencil, with the diagnostics C15 stores."""
    h, overlap = _pencil(seed, size=spectrum.size, spectrum=spectrum)
    energies, coefficients = eigh(h, overlap)
    residuals = np.array([
        np.linalg.norm(h @ coefficients[:, n] - energies[n] * overlap @ coefficients[:, n])
        / (
            np.linalg.norm(h @ coefficients[:, n])
            + abs(energies[n]) * np.linalg.norm(overlap @ coefficients[:, n])
        )
        for n in range(energies.size)
    ])
    window = coefficients.conj().T @ overlap @ coefficients
    return {
        "H": h,
        "S": overlap,
        "energies": energies,
        "C": coefficients,
        "residuals": residuals,
        "condition": float(np.linalg.cond(window)),
        "norbits": spectrum.size,
    }


def toy_clusters(window: dict) -> tuple[list[int], list[list[int]], np.ndarray]:
    resolutions = eigenvalue_resolutions_ev(
        window["energies"], window["residuals"], window["condition"], window["norbits"]
    )
    labels = near_degenerate_clusters(window["energies"], resolutions)
    return labels, cluster_groups(labels), resolutions


# -- resolution-driven clustering -------------------------------------------


def test_an_exact_degeneracy_is_one_cluster_and_the_resolved_gaps_are_not() -> None:
    labels, groups, resolutions = toy_clusters(toy_window())
    # the pair at 0.1 eV shares a label; nothing else does
    assert labels == [0, 1, 2, 2, 3, 4]
    assert [len(group) for group in groups] == [1, 1, 2, 1, 1]
    # and it is one cluster because the split is below the resolution, not
    # because indices 2 and 3 were declared degenerate somewhere
    assert 0.0 < resolutions.max() < 1e-10


def test_clustering_follows_the_residual_not_the_energies() -> None:
    """The same spectrum splits or merges purely on how converged the states are."""
    energies = np.array([-1.0, 0.0, 1e-9, 1.0])
    converged = eigenvalue_resolutions_ev(energies, 0.0, 1.0, 8)
    assert near_degenerate_clusters(energies, converged) == [0, 1, 2, 3]

    loose = eigenvalue_resolutions_ev(energies, 1e-8, 1.0, 8)
    assert loose.min() > 1e-9  # the solver cannot tell the two apart
    assert near_degenerate_clusters(energies, loose) == [0, 1, 1, 2]


def test_resolution_grows_with_the_residual_and_the_window_conditioning() -> None:
    energies = np.array([-1.0, 1.0])
    reference = eigenvalue_resolutions_ev(energies, 1e-9, 1.0, 8)
    assert np.allclose(eigenvalue_resolutions_ev(energies, 1e-7, 1.0, 8), 100.0 * reference)
    # cond enters as sqrt(cond) from ||r||_{S^-1} times the identity budget's cond
    scaled = eigenvalue_resolutions_ev(energies, 1e-9, 100.0, 8)
    assert np.allclose(scaled, 10.0 * reference)


def test_a_constant_resolution_reproduces_the_solver_labelling() -> None:
    """No second, incompatible notion of degeneracy enters with this module."""
    energies = np.array([-1.0, 0.1, 0.1 + 1e-12, 0.1 + 2e-12, 0.9])
    for tolerance in (1e-6, 1e-15):
        assert near_degenerate_clusters(energies, tolerance) == degenerate_subspace_labels(
            energies, tolerance
        )


def test_a_chain_of_unresolved_gaps_stays_one_cluster() -> None:
    energies = np.array([0.0, 1e-9, 2e-9, 3e-9])
    assert near_degenerate_clusters(energies, 2e-9) == [0, 0, 0, 0]
    with pytest.raises(SubspaceGateError, match="ascending"):
        near_degenerate_clusters(np.array([1.0, 0.0]), 1e-6)


# -- the metrics, on toy windows with an exact degeneracy --------------------


def test_a_gauge_rotation_moves_every_entry_and_no_invariant() -> None:
    window = toy_window()
    _, groups, _ = toy_clusters(window)
    rotation = random_gauge_rotation(groups, SIZE, np.random.default_rng(0))
    overlap = subspace_overlap(window["C"], window["S"], window["C"] @ rotation)

    assert np.abs(overlap - np.eye(SIZE)).max() > 0.5  # the gauge is a real rotation
    tolerances = subspace_metric_tolerances(
        np.abs(window["C"].conj().T @ window["S"] @ window["C"] - np.eye(SIZE)).max(),
        0.0,
        dimension=2,
        orbital_count=SIZE,
    )
    for group in groups:
        block = np.ix_(list(group), list(group))
        metrics = subspace_metrics(overlap[block])
        assert np.allclose(metrics["singular_values"], 1.0, atol=tolerances["singular_value"])
        assert metrics["maximum_principal_angle_rad"] <= tolerances["principal_angle_rad"]
        assert (
            metrics["projector_frobenius_distance"]
            <= tolerances["projector_frobenius_distance"]
        )


def test_mixing_across_a_resolved_gap_is_not_a_gauge_and_is_caught() -> None:
    window = toy_window()
    _, groups, _ = toy_clusters(window)
    mixing = cross_cluster_mixing(groups, SIZE, angle=np.pi / 4)
    overlap = subspace_overlap(window["C"], window["S"], window["C"] @ mixing)
    group = list(groups[0])
    metrics = subspace_metrics(overlap[np.ix_(group, group)])
    assert metrics["maximum_principal_angle_rad"] == pytest.approx(np.pi / 4, abs=1e-9)
    assert metrics["projector_frobenius_distance"] == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(SubspaceGateError, match="at least two clusters"):
        cross_cluster_mixing([[0, 1]], 2)


def test_the_projector_distance_is_the_explicit_one_without_building_a_projector() -> None:
    """``sqrt(d_A + d_B - 2 sum sigma^2)`` against ``P = C C† S`` built by hand.

    The projectors are S-self-adjoint, so the distance is the Frobenius norm of
    ``S^{1/2}(P_A - P_B)S^{-1/2}`` — the identity is what keeps a no_u x no_u
    object out of the calculation at MATBG size. Agreement is asserted to the
    module's own tolerance because ``sqrt(d_A + d_B - 2 sum sigma^2)`` cancels
    catastrophically for two equal spans: that sqrt is exactly why the projector
    tolerance carries a square root the singular values do not.
    """
    window = toy_window()
    overlap, coefficients = window["S"], window["C"]
    values, vectors = np.linalg.eigh(overlap)
    root = vectors @ np.diag(np.sqrt(values)) @ vectors.conj().T
    inverse_root = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.conj().T
    identity_error = np.abs(coefficients.conj().T @ overlap @ coefficients - np.eye(SIZE)).max()
    tolerance = subspace_metric_tolerances(
        identity_error, identity_error, dimension=3, orbital_count=SIZE
    )["projector_frobenius_distance"]

    left = coefficients[:, :3]
    for right in (coefficients[:, 1:4], coefficients[:, 3:], coefficients[:, :3]):
        projector_left = left @ left.conj().T @ overlap
        projector_right = right @ right.conj().T @ overlap
        explicit = np.linalg.norm(root @ (projector_left - projector_right) @ inverse_root)
        metrics = subspace_metrics(subspace_overlap(left, overlap, right))
        assert metrics["projector_frobenius_distance"] == pytest.approx(explicit, abs=tolerance)
    # the three cases are genuinely different subspaces, not three zeros
    assert subspace_metrics(subspace_overlap(left, overlap, coefficients[:, 3:]))[
        "projector_frobenius_distance"
    ] == pytest.approx(np.sqrt(6.0), abs=tolerance)

    with pytest.raises(SubspaceGateError, match="different bases"):
        subspace_overlap(left, overlap, coefficients[:2, :2])
    with pytest.raises(SubspaceGateError, match="2-D"):
        subspace_metrics(np.zeros(3))


def test_block_metrics_survive_the_gauge_that_scrambles_the_matrix_elements() -> None:
    """A coupling block goes to ``U_a† g_ab U_b``: entries are gauge, sigma is not."""
    window = toy_window()
    labels, groups, _ = toy_clusters(window)
    generator = np.random.default_rng(3)
    coupling = generator.normal(size=(SIZE, SIZE)) + 1j * generator.normal(size=(SIZE, SIZE))
    rotation = random_gauge_rotation(groups, SIZE, generator)
    rotated = rotation.conj().T @ coupling @ rotation

    before = block_metrics(coupling, labels, labels)
    after = block_metrics(rotated, labels, labels)
    assert len(before) == len(groups) ** 2
    assert np.abs(rotated - coupling).max() > 0.1
    for first, second in zip(before, after):
        assert first["row_states"] == second["row_states"]
        assert np.allclose(first["singular_values"], second["singular_values"], atol=1e-12)
        assert first["frobenius_norm"] == pytest.approx(second["frobenius_norm"], abs=1e-12)
        assert "matrix_elements" not in first  # band-indexed entries stay unpublished

    with pytest.raises(SubspaceGateError, match="cluster labels"):
        block_metrics(coupling[:2, :2], labels, labels)


def test_tolerances_are_derived_from_the_identity_error_with_their_own_powers() -> None:
    tolerances = subspace_metric_tolerances(1e-10, 3e-10, dimension=2, orbital_count=8)
    assert tolerances["identity_error"] == pytest.approx(4e-10)
    assert tolerances["singular_value"] == pytest.approx(4e-10)
    # the angle is not Lipschitz at sigma = 1 and the projector distance carries d
    assert tolerances["principal_angle_rad"] == pytest.approx(np.sqrt(2 * 4e-10))
    assert tolerances["projector_frobenius_distance"] == pytest.approx(np.sqrt(8 * 4e-10))
    assert tolerances["policy"] == METRIC_TOLERANCE_POLICY
    # a perfectly clean window still cannot claim better than float64 roundoff
    assert subspace_metric_tolerances(0.0, 0.0, dimension=1, orbital_count=8)["identity_error"] > 0


# -- the gate, on persisted C15 eigenspaces ---------------------------------


def persisted(directory: Path, kpoints: int = 2, **overrides) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for k_index in range(kpoints):
        write_eigenspace(
            directory,
            k_index,
            seed=11 + k_index,
            states=SIZE,
            size=SIZE,
            spectrum=DEGENERATE_SPECTRUM,
        )
        if overrides:
            path = directory / f"eigenspace_{k_index:03d}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.update(overrides)
            path.write_text(json.dumps(payload), encoding="utf-8")


def _written(tmp_path: Path) -> Path:
    directory = tmp_path / "eigenspaces"
    persisted(directory)
    return directory


def test_the_persisted_window_metric_reproduces_the_explicit_cross_gram(tmp_path: Path) -> None:
    """``I + D`` is the window's own metric, which is why the gate needs no S."""
    persisted(tmp_path, kpoints=1)
    window = toy_window(seed=11)
    from run_deeph_sparse_spectrum import load_persisted_eigenspace

    payload = load_persisted_eigenspace(tmp_path, 0)
    assert np.allclose(payload["coefficients"], window["C"])
    rotation = random_gauge_rotation(
        cluster_groups(toy_clusters(window)[0]), SIZE, np.random.default_rng(0)
    )
    explicit = subspace_overlap(window["C"], window["S"], window["C"] @ rotation)
    assert np.abs(window_metric(payload) @ rotation - explicit).max() < 1e-12


def test_go6_passes_on_persisted_eigenspaces_with_a_resolved_gap(tmp_path: Path) -> None:
    report = certify(_written(tmp_path), 2)
    assert report["verdict"] == "PASS"
    assert report["gate_id"] == GATE_ID and report["gate"] == "GO-6"
    assert report["summary"]["windows_passed"] == 2
    assert report["summary"]["windows_with_a_resolved_gap"] == 2
    assert report["eigenspace_status"] == "valid"
    assert report["metric_tolerance_policy"] == METRIC_TOLERANCE_POLICY
    assert report["energy_resolution_policy"] == ENERGY_RESOLUTION_POLICY
    assert report["eigenspace_identity_tolerance_policy"]
    for row in report["windows"]:
        assert row["cluster_labels"] == [0, 1, 2, 2, 3, 4]
        assert [len(entry["states"]) for entry in row["clusters"]] == [1, 1, 2, 1, 1]
        assert all(entry["gauge_invariant"] for entry in row["clusters"])
        assert row["cross_cluster_discrimination"]["detected"] is True
        assert row["tolerances"]["principal_angle_rad"] > 0
    assert require_go6(report) is report


def test_a_window_the_solver_cannot_resolve_certifies_nothing(tmp_path: Path) -> None:
    """Every gap unresolved means no discrimination was exercised: not a PASS."""
    directory = tmp_path / "loose"
    persisted(directory, kpoints=1, generalized_relative_residual=[1.0] * SIZE)
    report = certify(directory, 1)
    assert report["windows"][0]["cluster_count"] == 1
    assert report["windows"][0]["cross_cluster_discrimination"]["applicable"] is False
    assert report["verdict"] == "NO_GO"
    with pytest.raises(SubspaceGateError, match="GO-6 is NO_GO"):
        require_go6(report)


def test_require_go6_refuses_anything_that_is_not_a_passing_certification(tmp_path: Path) -> None:
    with pytest.raises(SubspaceGateError, match="has not been certified"):
        require_go6(tmp_path / "absent.json")
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"gate_id": "go5", "verdict": "PASS"}), encoding="utf-8")
    with pytest.raises(SubspaceGateError, match="not a go6"):
        require_go6(path)


def test_cli_writes_the_report_and_exits_on_the_verdict(tmp_path: Path) -> None:
    output = tmp_path / "report"
    arguments = [
        "--eigenspace-dir", str(_written(tmp_path)),
        "--kpoint-count", "2",
        "--output-dir", str(output),
    ]
    assert main(arguments) == 0
    written = require_go6(output / "go6_subspace_metrics.json")
    assert written["ticket"].startswith("C16")
    assert written["gauge_seed"] == 0 and written["separation_factor"] == 1.0
