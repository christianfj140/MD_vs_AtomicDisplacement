#!/usr/bin/env python3
"""E-F_001-S40: GO-8b certification of the scalable MATBG q != 0 response."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import certify_qneq0_matbg_response as cert  # noqa: E402
import epc_qb_qc_prototypes as prototypes  # noqa: E402


def test_route_mechanics_suite_passes_today():
    certification = cert.certify_qneq0_matbg_response_go8b()
    assert certification["route_mechanics_suite"]["status"] == "PASS", certification["route_mechanics_suite"]


def test_graphene_k_agreement_is_blocked_by_go5_not_fabricated():
    certification = cert.certify_qneq0_matbg_response_go8b()
    suite = certification["graphene_k_agreement_suite"]
    assert suite["status"] == "NOT_RUN"
    assert suite["blocked_by"] == "NO_GO"
    entry = suite["checks"][0]
    assert entry["applicable"] is False


def test_overall_verdict_is_no_go8b_until_go5_clears():
    certification = cert.certify_qneq0_matbg_response_go8b()
    assert certification["go8b_status"] == "NO-GO-8b"
    assert certification["matbg_qneq0_production_authorized"] is False
    assert certification["gate"] == "GO-8b"


def test_every_route_mechanics_check_is_present():
    certification = cert.certify_qneq0_matbg_response_go8b()
    names = {entry["check"] for entry in certification["route_mechanics_suite"]["checks"]}
    assert names == {
        "qc_and_qb_agree_on_real_graph2mat",
        "q_and_minus_q",
        "cell_origin_shift",
        "atom_across_periodic_boundary",
        "alternative_atomic_phase_convention",
        "topology_margin_preserved",
        "gpu_preflight_recorded",
        "sparse_serialization_round_trip",
        "kernel_table_cache_restart",
        "jvp_calls_independent_of_shells",
        "gamma_jvp_reuse_for_qneq0_rejected",
    }


def test_require_go8b_certified_response_raises_on_no_go():
    certification = cert.certify_qneq0_matbg_response_go8b()
    with pytest.raises(cert.MatbgQneq0ResponseCertificationError):
        cert.require_go8b_certified_qneq0_response(certification)


def test_require_go8b_certified_response_raises_on_malformed_object():
    with pytest.raises(cert.MatbgQneq0ResponseCertificationError):
        cert.require_go8b_certified_qneq0_response({"gate": "some_other_gate", "go8b_status": "GO-8b"})
    with pytest.raises(cert.MatbgQneq0ResponseCertificationError):
        cert.require_go8b_certified_qneq0_response({})


def test_require_go8b_certified_response_accepts_a_passing_certification():
    passing = {"gate": "GO-8b", "go8b_status": "GO-8b", "extra": 1}
    assert cert.require_go8b_certified_qneq0_response(passing) == passing


def test_primary_route_tracks_s33_live_instead_of_repeating_its_name():
    """Regression: this module used to hardcode "Q-C" here, so it kept
    describing Q-C as an already-selected primary route even after S33
    itself was corrected to report no route selected while GO-5 is NO_GO --
    an audit correctly read that drift as the DAG being violated.
    """
    decision = prototypes.production_route_decision()
    certification = cert.certify_qneq0_matbg_response_go8b()
    assert certification["primary_route"] == decision["primary_route"]
    assert certification["validation_only_route"] == decision["validation_only_route"]
    # Today GO-5 is NO_GO, so S33 itself reports no route selected.
    assert decision["primary_route"] is None
    assert certification["primary_route"] is None
    # The kernel this module tests is still, factually, built against Q-C --
    # that is a separate claim from "Q-C has been selected".
    assert certification["kernel_mechanics_built_against_route"] == "Q-C"
