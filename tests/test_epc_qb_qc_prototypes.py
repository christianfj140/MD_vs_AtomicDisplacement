#!/usr/bin/env python3
"""E-F_001-S32: Q-B and Q-C prototypes agree with the Q-A reference and with
each other, on the toy periodic chain, at every commensurate electronic k."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import epc_qb_qc_prototypes as proto  # noqa: E402

DIRECTION = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]) / np.sqrt(2.0)
Q = (1.0 / 3.0, 0.0, 0.0)
FD_TOLERANCE = 5e-9  # central difference at delta=1e-5: O(delta^2) ~ 1e-10, generous margin


# --------------------------------------------------------------------------- #
# Correctness: all three routes agree where Q-A is well-defined
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k_fraction", [0.0, 1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0])
def test_qb_and_qc_agree_with_the_commensurate_reference(k_fraction):
    k_primitive = (k_fraction, 0.0, 0.0)
    reference = proto.q_a_reference(Q, DIRECTION, k_primitive)
    table = proto.kernel_table(DIRECTION)
    qc = proto.k_to_k_plus_q_kernel(table, k_primitive, Q)
    phase_aware = proto.phase_aware_kernel(DIRECTION, Q)
    qb = proto.q_b_response(phase_aware, k_primitive)

    assert np.abs(reference["D_H"] - qc["D_H"]).max() < FD_TOLERANCE
    assert np.abs(reference["D_H"] - qb["D_H"]).max() < FD_TOLERANCE
    assert np.abs(reference["D_S"] - qc["D_S"]).max() < FD_TOLERANCE
    assert np.abs(reference["D_S"] - qb["D_S"]).max() < FD_TOLERANCE


def test_qb_and_qc_agree_with_each_other_directly():
    table = proto.kernel_table(DIRECTION)
    phase_aware = proto.phase_aware_kernel(DIRECTION, Q)
    for k_fraction in (0.0, 1.0 / 3.0, 2.0 / 3.0):
        k_primitive = (k_fraction, 0.0, 0.0)
        qc = proto.k_to_k_plus_q_kernel(table, k_primitive, Q)
        qb = proto.q_b_response(phase_aware, k_primitive)
        assert np.abs(qc["D_H"] - qb["D_H"]).max() < FD_TOLERANCE


def test_compare_routes_reports_worst_case_below_tolerance_and_declares_limits():
    report = proto.compare_routes(Q, DIRECTION)
    assert report["status"] == "prototype__algebra_only"
    assert report["worst_case_agreement_D_H"] < FD_TOLERANCE
    assert "reference_limits" in report
    assert "gpu_preflight" in report
    assert report["not_materialized"]


# --------------------------------------------------------------------------- #
# Q-C is valid at incommensurate k too (the reference is what is restricted)
# --------------------------------------------------------------------------- #


def test_qc_and_qb_still_agree_with_each_other_at_incommensurate_k():
    """Q-A cannot be asked here (module docstring), but Q-B/Q-C are plain
    real-space-to-k sums, defined for any k -- and must still agree."""
    table = proto.kernel_table(DIRECTION)
    phase_aware = proto.phase_aware_kernel(DIRECTION, Q)
    k_primitive = (0.11, 0.0, 0.0)
    qc = proto.k_to_k_plus_q_kernel(table, k_primitive, Q)
    qb = proto.q_b_response(phase_aware, k_primitive)
    assert np.abs(qc["D_H"] - qb["D_H"]).max() < FD_TOLERANCE


# --------------------------------------------------------------------------- #
# No (n_outputs, N, 3) Jacobian: every kernel stays shell-sized
# --------------------------------------------------------------------------- #


def test_kernel_table_never_grows_with_the_number_of_atoms_queried():
    table = proto.kernel_table(DIRECTION, shells=6)
    assert len(table["A_H"]) == 2 * 6 + 1
    for block in table["A_H"].values():
        assert block.shape == (2, 2)


def test_direction_shape_is_validated():
    with pytest.raises(proto.PrototypeError):
        proto.compare_routes(Q, np.zeros((3, 3)))


# --------------------------------------------------------------------------- #
# Scaling: Q-C's one-time build must eventually beat Q-B's per-q rebuild
# --------------------------------------------------------------------------- #


def test_q_c_pays_fd_once_and_q_b_pays_fd_per_q():
    q_values = [(0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (0.3, 0.0, 0.0)]
    k_values = [(0.05, 0.0, 0.0)]
    report = proto.scaling_report(q_values, k_values, shells=4)
    assert report["q_c_pays_fd_once"] is True
    assert report["q_b_pays_fd_per_q"] is True


def test_scaling_sweep_finds_a_crossover_in_a_reasonable_range():
    sweep = proto.scaling_sweep(n_q_values=[1, 10, 40, 120], n_k=5, shells=4)
    assert sweep["crossover_observed"], (
        "expected Q-C's amortized kernel_table() to overtake Q-B's per-q rebuild "
        f"somewhere in n_q up to 120; points were {sweep['points']}"
    )


# --------------------------------------------------------------------------- #
# E-F_001-S33: the production route decision is data, and rejects Gamma-JVP reuse
# --------------------------------------------------------------------------- #


def test_production_route_decision_stays_blocked_and_rejects_gamma_jvp_reuse():
    decision = proto.production_route_decision()
    assert decision["status"] == "BLOCKED"
    assert decision["primary_route"] is None
    assert decision["validation_only_route"] is None
    assert decision["candidate_after_go5"] == "Q-C"
    assert decision["rejects_gamma_jvp_reuse"] is True
    assert "gamma_jvp_reuse_rejection" in decision
    assert decision["no_production_started"] is True
    assert decision["blocked_by"].startswith("GO-5 NO_GO")


def test_production_route_decision_covers_all_four_axes_with_reproducible_evidence():
    decision = proto.production_route_decision()
    for axis in ("accuracy", "basis_response", "cost", "cache_restart"):
        assert axis in decision["axes"]
        assert "verdict" in decision["axes"][axis]

    # cost/cache_restart evidence must come from the same reproducible sweep
    # shape as an independent measurement (wall-clock seconds vary run to run).
    fresh_sweep = proto.scaling_sweep(n_q_values=(1, 5, 20, 80), n_k=5, shells=4)
    assert decision["axes"]["cost"]["evidence"]["n_k"] == fresh_sweep["n_k"]
    assert decision["axes"]["cost"]["evidence"]["shells"] == fresh_sweep["shells"]
    assert [p["n_q"] for p in decision["evidence"]["scaling_sweep"]["points"]] == [
        p["n_q"] for p in fresh_sweep["points"]
    ]
    assert decision["axes"]["cost"]["evidence"]["crossover_n_q"] in (1, 5, 20, 80, None)


def test_production_route_decision_lists_additional_tests_before_go8b():
    decision = proto.production_route_decision()
    tests = decision["additional_tests_required_before_go8b"]
    assert len(tests) >= 5
    joined = " ".join(tests)
    assert "real Graph2Mat" in joined
    assert "restart" in joined
    assert "GPU preflight" in joined


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
