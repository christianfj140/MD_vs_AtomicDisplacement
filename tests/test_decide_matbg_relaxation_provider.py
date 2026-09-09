#!/usr/bin/env python3
"""E-F_001-S47: the scalable MATBG relaxation provider GO-10/NO-GO-10 decision."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import decide_matbg_relaxation_provider as dr  # noqa: E402


def test_verdict_is_no_go_with_no_candidate_selected():
    decision = dr.matbg_relaxation_provider_decision()
    assert decision["verdict"] == "NO-GO-10"
    assert decision["selected_provider"] is None
    assert decision["gate"] == "GO-10"


def test_no_go_never_claims_a_relaxed_geometry_was_produced():
    decision = dr.matbg_relaxation_provider_decision()
    assert decision["relaxed_geometry_produced"] is False


def test_every_candidate_declares_every_mechanism_and_a_blocking_reason():
    decision = dr.matbg_relaxation_provider_decision()
    assert len(decision["candidates"]) >= 3
    for candidate in decision["candidates"]:
        assert set(candidate["coverage"]) == set(dr.RELAXATION_MECHANISMS)
        assert isinstance(candidate["defensible"], bool)
        if not candidate["defensible"]:
            assert candidate["blocking_reason"]
        assert candidate["coverage_basis"]


def test_dft_route_is_ruled_out_on_cost_not_on_physics():
    decision = dr.matbg_relaxation_provider_decision()
    dft = next(c for c in decision["candidates"] if c["id"] == "dft_relaxation_siesta")
    assert all(dft["coverage"].values())  # no physical mechanism is missing
    assert not dft["defensible"]
    assert "intractable" in dft["blocking_reason"]


def test_classical_potential_route_is_blocked_on_missing_interlayer_term():
    decision = dr.matbg_relaxation_provider_decision()
    classical = next(
        c for c in decision["candidates"] if c["id"] == "classical_empirical_potential"
    )
    assert classical["coverage"]["in_plane_lattice_relaxation"] is True
    assert classical["coverage"]["out_of_plane_corrugation"] is False
    assert classical["coverage"]["ab_ba_domain_formation"] is False
    assert "interlayer" in classical["blocking_reason"]


def test_acceptance_criteria_keep_rigid_and_relaxed_statuses_distinct():
    decision = dr.matbg_relaxation_provider_decision()
    criteria = decision["acceptance_criteria_for_a_future_relaxation"]
    assert "geometry_signature" in criteria
    assert "status_label" in criteria
    assert "rigid" in criteria["status_label"] and "relaxed" in criteria["status_label"]


def test_importability_probe_reflects_the_real_environment():
    """The classical-potential candidate's availability is measured, not hard-coded."""
    decision = dr.matbg_relaxation_provider_decision()
    classical = next(
        c for c in decision["candidates"] if c["id"] == "classical_empirical_potential"
    )
    availability = classical["availability"]["intralayer_bond_order_potential_importable"]
    for module_name, importable in availability.items():
        assert importable == dr._importable(module_name)


def test_a_candidate_would_flip_the_verdict_if_it_became_defensible(monkeypatch):
    """NO-GO-10 must be earned, not hard-coded -- one candidate flipping flips the verdict."""

    original = dr._candidates

    def _patched_candidates():
        candidates = original()
        candidates[0] = dict(candidates[0], defensible=True)
        return candidates

    monkeypatch.setattr(dr, "_candidates", _patched_candidates)
    decision = dr.matbg_relaxation_provider_decision()
    assert decision["verdict"].startswith("GO-10:")
    assert decision["selected_provider"] == decision["candidates"][0]["id"]


def test_self_test_runs_end_to_end():
    dr._self_test()
