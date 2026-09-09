#!/usr/bin/env python3
"""E-F_001-S38 / D05B: the scalable MATBG PhononProvider GO-8a/NO-GO-8a decision."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import decide_matbg_phonon_provider as dp  # noqa: E402


def test_verdict_is_no_go_with_no_candidate_selected():
    decision = dp.matbg_phonon_provider_decision()
    assert decision["verdict"] == "NO-GO-8a"
    assert decision["selected_provider"] is None
    assert decision["gate"] == "GO-8a"


def test_no_go_never_claims_a_full_ks_coupling_was_produced():
    decision = dp.matbg_phonon_provider_decision()
    assert decision["full_ks_coupling_produced"] is False


def test_every_candidate_declares_every_coverage_axis_and_a_blocking_reason():
    decision = dp.matbg_phonon_provider_decision()
    assert len(decision["candidates"]) >= 4
    for candidate in decision["candidates"]:
        assert set(candidate["coverage"]) == set(dp.COVERAGE_AXES)
        assert isinstance(candidate["defensible"], bool)
        if not candidate["defensible"]:
            assert candidate["blocking_reason"]
        assert candidate["coverage_basis"]


def test_dft_route_is_ruled_out_on_cost_not_on_physics():
    decision = dp.matbg_phonon_provider_decision()
    dft = next(c for c in decision["candidates"] if c["id"] == "dft_force_constants_siesta")
    assert all(dft["coverage"].values())  # no physical mechanism is missing
    assert not dft["defensible"]
    assert "intractable" in dft["blocking_reason"]


def test_classical_potential_route_is_blocked_on_missing_interlayer_term():
    decision = dp.matbg_phonon_provider_decision()
    classical = next(
        c for c in decision["candidates"] if c["id"] == "classical_empirical_potential"
    )
    assert classical["coverage"]["acoustic"] is True
    assert classical["coverage"]["shear"] is False
    assert classical["coverage"]["breathing"] is False
    assert classical["coverage"]["moire"] is False
    assert "interlayer" in classical["blocking_reason"]


def test_moon_koshino_route_is_blocked_on_missing_total_energy_functional():
    decision = dp.matbg_phonon_provider_decision()
    tb = next(
        c
        for c in decision["candidates"]
        if c["id"] == "moon_koshino_tight_binding_electronic"
    )
    assert all(value is False for value in tb["coverage"].values())
    assert tb["availability"]["reference_module_computes_forces"] is False


def test_benchmark_suite_cites_existing_fixtures_not_new_numbers():
    decision = dp.matbg_phonon_provider_decision()
    suite = decision["benchmark_suite_required_before_go8a"]
    assert "small_system" in suite and "matbg_scale" in suite
    gamma = suite["small_system"]["graphene_gamma_acoustic_and_optical"]
    assert "test_phonon_provider.py" in gamma["reference"]
    assert suite["matbg_scale"]["geometry_signature_match"]


def test_importability_probe_reflects_the_real_environment():
    """The classical-potential candidate's availability is measured, not hard-coded."""
    decision = dp.matbg_phonon_provider_decision()
    classical = next(
        c for c in decision["candidates"] if c["id"] == "classical_empirical_potential"
    )
    availability = classical["availability"]["intralayer_bond_order_potential_importable"]
    for module_name, importable in availability.items():
        assert importable == dp._importable(module_name)


def test_a_candidate_would_flip_the_verdict_if_it_became_defensible(monkeypatch):
    """NO-GO-8a must be earned, not hard-coded -- one candidate flipping flips the verdict."""

    original = dp._candidates

    def _patched_candidates():
        candidates = original()
        candidates[0] = dict(candidates[0], defensible=True)
        return candidates

    monkeypatch.setattr(dp, "_candidates", _patched_candidates)
    decision = dp.matbg_phonon_provider_decision()
    assert decision["verdict"].startswith("GO-8a:")
    assert decision["selected_provider"] == decision["candidates"][0]["id"]


def test_self_test_runs_end_to_end():
    dp._self_test()
