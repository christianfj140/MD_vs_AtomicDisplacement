#!/usr/bin/env python3
"""E-F_001-S40: Q-C's real-space kernel (and Q-B's cross-check), built on the
real production JVP (compute_graph2mat_directional_derivative), not on
epc_qb_qc_prototypes' hand-rolled finite differences."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import epc_qb_qc_graph2mat_kernel as g2m  # noqa: E402
from graph2mat_autograd_derivatives import compute_graph2mat_directional_derivative  # noqa: E402


# --------------------------------------------------------------------------- #
# S33 item 1: the real-JVP kernel construction is exact and O(1) in shells
# --------------------------------------------------------------------------- #


def test_kernel_table_uses_exactly_two_jvp_calls():
    table = g2m.kernel_table_graph2mat(shells=4)
    assert table["jvp_calls"] == 2
    assert table["materialized_jacobian"] is False


def test_phase_aware_kernel_uses_exactly_two_jvp_calls():
    phase_aware = g2m.phase_aware_kernel_graph2mat((0.2, 0.0, 0.0), shells=4)
    assert phase_aware["jvp_calls"] == 2
    assert phase_aware["materialized_jacobian"] is False


def test_kernel_table_matches_the_toy_prototype():
    """The real-JVP construction must agree with the hand-rolled FD toy it
    replaces (epc_qb_qc_prototypes.kernel_table), to FD noise."""
    import epc_qb_qc_prototypes as proto

    table = g2m.kernel_table_graph2mat(shells=1)
    toy = proto.kernel_table(g2m.DEFAULT_DIRECTION_VECTORS)
    for shell in (-1, 0, 1):
        assert np.abs(table["A_H"][shell] - toy["A_H"][shell]).max() < 5e-9
        assert np.abs(table["B_H"][shell] - toy["B_H"][shell]).max() < 5e-9
        assert np.abs(table["A_S"][shell] - toy["A_S"][shell]).max() < 5e-9
        assert np.abs(table["B_S"][shell] - toy["B_S"][shell]).max() < 5e-9


# --------------------------------------------------------------------------- #
# S33 item 3: the five S31-style checks
# --------------------------------------------------------------------------- #


def test_real_kernel_checks_all_pass():
    report = g2m.real_kernel_checks()
    assert report["all_pass"], report
    assert set(report["checks"]) == {
        "qc_and_qb_agree_on_real_graph2mat",
        "q_and_minus_q",
        "cell_origin_shift",
        "atom_across_periodic_boundary",
        "alternative_atomic_phase_convention",
    }
    assert all(status == "PASS" for status in report["checks"].values())


# --------------------------------------------------------------------------- #
# S33 item 2: topology margin (reused fd_perturbation_space.topology_margin)
# --------------------------------------------------------------------------- #


def test_topology_margin_is_preserved_at_a_small_physical_delta():
    report = g2m.topology_margin_report(delta_ang=0.05)
    assert report["topology_preserved"] is True
    assert report["crossing_pairs"] == []


def test_topology_margin_crosses_at_a_large_enough_delta():
    report = g2m.topology_margin_report(delta_ang=2.0)
    assert report["topology_preserved"] is False
    assert report["crossing_pairs"]


# --------------------------------------------------------------------------- #
# S33 item 6: GPU preflight extends resolve_jvp_backend, never a silent fallback
# --------------------------------------------------------------------------- #


def test_gpu_preflight_records_requested_and_effective_backend():
    report = g2m.gpu_preflight_report(requested_backend="cpu")
    assert report["requested_backend"] == "cpu"
    assert report["effective_backend"] == "cpu"
    assert report["backend_preflight"] == "not_required"


def test_gpu_preflight_never_silently_falls_back():
    """Whatever CUDA availability this host has, the record must say why."""
    report = g2m.gpu_preflight_report(requested_backend="cuda")
    assert report["requested_backend"] == "cuda"
    if report["effective_backend"] == "cpu":
        assert report["backend_fallback_reason"]
    else:
        assert report["effective_backend"] == "cuda"
        assert report["backend_preflight"] == "passed"


# --------------------------------------------------------------------------- #
# S33 item 8: sparse serialisation round trip (C11)
# --------------------------------------------------------------------------- #


def test_sparse_round_trip_is_reproducible():
    report = g2m.sparse_round_trip_check()
    assert report["nnz"] > 0
    assert report["reproducible_across_calls"] is True
    assert report["matches_result_metadata_nnz"] is True


# --------------------------------------------------------------------------- #
# S33 item 5: raw_derivative_kernel_table cache/restart
# --------------------------------------------------------------------------- #


def test_kernel_table_cache_restart_round_trips_without_recomputation():
    table = g2m.kernel_table_graph2mat()
    node = g2m.kernel_table_dag_node(table)
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "kernel_table.npz"
        metadata_path = Path(tmpdir) / "kernel_table.json"
        g2m.save_kernel_table_cache(table, node, npz_path, metadata_path)
        loaded, status = g2m.load_kernel_table_cache(node, npz_path, metadata_path)

    assert status == "valid"
    assert loaded is not None
    assert loaded["loaded_from_cache"] is True
    before = g2m.k_to_k_plus_q_kernel_graph2mat(table, (0.15, 0.0, 0.0), (0.2, 0.0, 0.0))
    after = g2m.k_to_k_plus_q_kernel_graph2mat(loaded, (0.15, 0.0, 0.0), (0.2, 0.0, 0.0))
    assert np.abs(before["D_H"] - after["D_H"]).max() == 0.0


def test_kernel_table_cache_rejects_a_different_direction():
    table = g2m.kernel_table_graph2mat()
    node = g2m.kernel_table_dag_node(table)
    other_table = g2m.kernel_table_graph2mat(direction=[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    other_node = g2m.kernel_table_dag_node(other_table)
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "kernel_table.npz"
        metadata_path = Path(tmpdir) / "kernel_table.json"
        g2m.save_kernel_table_cache(table, node, npz_path, metadata_path)
        loaded, status = g2m.load_kernel_table_cache(other_node, npz_path, metadata_path)
    assert status == "signature_mismatch"
    assert loaded is None


# --------------------------------------------------------------------------- #
# S33 item 7: shell-scaling proxy sweep
# --------------------------------------------------------------------------- #


def test_scaling_sweep_shows_jvp_calls_do_not_grow_with_shells():
    report = g2m.scaling_sweep_graph2mat((1, 4, 16))
    assert report["jvp_calls_independent_of_shells"] is True
    assert report["matbg_edge_count_measured"] is False
    for point in report["points"]:
        assert point["jvp_calls"] == 2


# --------------------------------------------------------------------------- #
# GO-8b acceptance criterion: a Gamma JVP reused for q != 0 fails automatically
# --------------------------------------------------------------------------- #


def _gamma_jvp_metadata():
    model = g2m.PeriodicShellGraphModel()
    home0 = torch.as_tensor(g2m.PRIMITIVE_POSITIONS_ANG, dtype=torch.float64)
    batch = g2m.KernelBatch(positions=torch.cat([home0, home0.clone()]))
    dvec = torch.as_tensor(g2m.DEFAULT_DIRECTION_VECTORS, dtype=torch.float64)
    zero = torch.zeros_like(dvec)
    result = compute_graph2mat_directional_derivative(g2m.PeriodicShellGraphModel(), batch, torch.cat([zero, dvec]))
    return {"schema": result.metadata["schema"]}


def test_k_to_k_plus_q_kernel_rejects_a_gamma_jvp_payload_for_qneq0():
    fake_table = _gamma_jvp_metadata()
    with pytest.raises(g2m.GammaJvpReuseError):
        g2m.k_to_k_plus_q_kernel_graph2mat(fake_table, (0.1, 0.0, 0.0), (0.2, 0.0, 0.0))


def test_k_to_k_plus_q_kernel_allows_a_gamma_jvp_payload_only_at_qeq0():
    """q == 0 degenerates to a plain Gamma response; only q != 0 is rejected."""
    fake_table = _gamma_jvp_metadata()
    with pytest.raises(KeyError):
        # Not a real kernel table (no A_H/B_H/...), so it still fails -- but
        # for a KeyError, not GammaJvpReuseError: the guard itself must not
        # fire at q == 0.
        g2m.k_to_k_plus_q_kernel_graph2mat(fake_table, (0.1, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_q_b_response_rejects_a_gamma_jvp_payload():
    fake_phase_aware = _gamma_jvp_metadata()
    with pytest.raises(g2m.GammaJvpReuseError):
        g2m.q_b_response_graph2mat(fake_phase_aware, (0.1, 0.0, 0.0))


# --------------------------------------------------------------------------- #
# Blocker B4 rationale: the plain shared-position JVP cannot separate the
# home/neighbour roles of a self-pair edge -- which is exactly why two
# independent position leaves are needed at all (module docstring).
# --------------------------------------------------------------------------- #


def test_periodic_shell_graph_model_has_self_pair_edges_at_nonzero_shells():
    model = g2m.PeriodicShellGraphModel(shells=1)
    self_pairs = [(shell, mu, nu) for shell, mu, nu in model.edges if mu == nu]
    assert self_pairs, "the model must exercise the mu == nu, shell != 0 case B4 is about"
