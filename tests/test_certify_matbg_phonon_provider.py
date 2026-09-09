"""E-F_001-S39: the GO-8a certification for the scalable MATBG ``PhononProvider``.

S38 already lives as a decision (``NO-GO-8a``, no candidate selected); this
gate is what turns that decision into a certification artifact that carries
the ``artifact_signature`` schema split (``physical_phonon`` vs
``synthetic_test_displacement``) and the ``require_go8a_certified_matbg_phonons``
consumer-side lock. Both suites are exercised: ``small_system`` against the
real archived fixtures already on disk (no new numbers), and ``matbg_scale``
both in its current blocked state and, with a real (small, stand-in) provider
plugged in, to prove the suite this module runs the day GO-8a reopens is not
dead code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import certify_matbg_phonon_provider as cert  # noqa: E402
import phonon_provider as pp  # noqa: E402
from artifact_signature import PHYSICAL_PHONON  # noqa: E402


# --------------------------------------------------------------------------- #
# Current repository state: NO-GO-8a, no provider.
# --------------------------------------------------------------------------- #


def test_no_provider_is_no_go8a_and_produces_no_physical_phonon():
    certification = cert.certify_matbg_phonon_provider_go8a()
    assert certification["gate"] == "GO-8a"
    assert certification["go8a_status"] == "NO-GO-8a"
    assert certification["physical_phonon_produced"] is False
    assert certification["physical_phonon"] is None
    assert certification["artifact_kind"] is None


def test_small_system_suite_passes_against_real_archived_fixtures():
    certification = cert.certify_matbg_phonon_provider_go8a()
    small_system = certification["small_system_suite"]
    assert small_system["status"] == "PASS"
    by_name = {entry["check"]: entry for entry in small_system["checks"]}
    assert by_name["graphene_gamma_acoustic_and_optical"]["passed"] is True
    assert by_name["acoustic_sum_rule_raw_vs_corrected"]["passed"] is True
    assert by_name["ab_bilayer_interlayer_sectors"]["passed"] is True


def test_matbg_scale_suite_is_blocked_not_run_and_carries_s38_reasons():
    certification = cert.certify_matbg_phonon_provider_go8a()
    matbg_scale = certification["matbg_scale_suite"]
    decision = certification["decision"]
    assert matbg_scale["status"] == "NOT_RUN"
    assert matbg_scale["blocked_by"] == decision["verdict"] == "NO-GO-8a"
    assert matbg_scale["selected_provider"] is None
    assert matbg_scale["reopen_if"] == decision["reopen_if"]
    assert len(matbg_scale["checks"]) == len(
        decision["benchmark_suite_required_before_go8a"]["matbg_scale"]
    )
    for entry in matbg_scale["checks"]:
        assert entry["applicable"] is False
        assert entry["passed"] is True  # not-applicable checks never fail the gate
        assert decision["verdict"] in entry["detail"]


def test_require_go8a_certified_matbg_phonons_refuses_a_no_go_certification():
    certification = cert.certify_matbg_phonon_provider_go8a()
    with pytest.raises(cert.MatbgPhononCertificationError, match="NO-GO-8a"):
        cert.require_go8a_certified_matbg_phonons(certification)


def test_require_go8a_certified_matbg_phonons_refuses_a_malformed_object():
    with pytest.raises(cert.MatbgPhononCertificationError):
        cert.require_go8a_certified_matbg_phonons({"gate": "GO-8a", "go8a_status": "GO-8a"})


# --------------------------------------------------------------------------- #
# The reopen path: matbg_scale actually runs against a real provider.
#
# There is no defensible MATBG-scale candidate today (that is S38's verdict),
# so this stands a small archived SiestaFcPhononProvider (graphene, 2 atoms)
# in for "a future provider on the target geometry" -- it exercises exactly
# the checks _certify_matbg_scale_suite runs, on a real PhononModeSet, rather
# than leaving that function's only test coverage be "never called".
# --------------------------------------------------------------------------- #


def test_matbg_scale_suite_runs_for_real_against_a_provider_on_matching_geometry(monkeypatch):
    """The suite actually executes (not ``NOT_RUN``) once a provider exists.

    This stands a small archived provider in for "a future MATBG-scale
    candidate" to prove ``_certify_matbg_scale_suite`` is live code, not to
    assert it always clears every check on this stand-in: the geometry,
    acoustic and provenance checks are expected to hold on real archived
    data, but ``dynamical_matrix_hermiticity``'s tolerance is derived from
    toy-scale roundoff and is not guaranteed to clear on a real SIESTA FC
    run, so its outcome is read from the report rather than assumed.
    """
    if not cert.GRAPHENE_FC_RUN.is_dir():
        pytest.skip(f"no archived FC run at {cert.GRAPHENE_FC_RUN}")
    stand_in_fdf = cert.GRAPHENE_FC_RUN / "RUN.fdf"
    provider = pp.SiestaFcPhononProvider.from_run_dir(cert.GRAPHENE_FC_RUN, apply_asr=True)
    monkeypatch.setattr(cert, "TARGET_ATOM_COUNT", 2)

    certification = cert.certify_matbg_phonon_provider_go8a(provider, target_fdf=stand_in_fdf)

    matbg_scale = certification["matbg_scale_suite"]
    assert matbg_scale["status"] in ("PASS", "FAIL")  # measured, never NOT_RUN once a provider runs
    by_name = {entry["check"]: entry for entry in matbg_scale["checks"]}
    assert by_name["geometry_signature_match"]["passed"] is True
    assert by_name["acoustic_branches_vanish_at_gamma_after_asr"]["passed"] is True
    assert by_name["resource_preflight"]["passed"] is True
    assert by_name["provenance"]["passed"] is True
    assert by_name["dynamical_matrix_hermiticity"]["applicable"] is True
    assert by_name["dynamical_matrix_hermiticity"]["hermiticity_ev_ang2_amu"] >= 0.0

    all_passed = all(entry["passed"] for entry in matbg_scale["checks"])
    assert matbg_scale["status"] == ("PASS" if all_passed else "FAIL")
    assert certification["go8a_status"] == ("GO-8a" if all_passed else "NO-GO-8a")
    assert certification["physical_phonon_produced"] is all_passed

    if all_passed:
        assert certification["artifact_kind"] == PHYSICAL_PHONON
        assert certification["physical_phonon"]["artifact_kind"] == PHYSICAL_PHONON
        physical_phonon = cert.require_go8a_certified_matbg_phonons(certification)
        assert physical_phonon["artifact_kind"] == PHYSICAL_PHONON
    else:
        assert certification["physical_phonon"] is None
        with pytest.raises(cert.MatbgPhononCertificationError):
            cert.require_go8a_certified_matbg_phonons(certification)


def test_matbg_scale_suite_fails_on_geometry_mismatch(monkeypatch):
    if not cert.GRAPHENE_FC_RUN.is_dir():
        pytest.skip(f"no archived FC run at {cert.GRAPHENE_FC_RUN}")
    stand_in_fdf = cert.GRAPHENE_FC_RUN / "RUN.fdf"
    provider = pp.SiestaFcPhononProvider.from_run_dir(cert.GRAPHENE_FC_RUN, apply_asr=True)
    # Leave TARGET_ATOM_COUNT at the real (31, 30) target: a 2-atom provider
    # must not silently certify against an 11 164-atom geometry.
    certification = cert.certify_matbg_phonon_provider_go8a(provider, target_fdf=stand_in_fdf)

    matbg_scale = certification["matbg_scale_suite"]
    by_name = {entry["check"]: entry for entry in matbg_scale["checks"]}
    assert by_name["geometry_signature_match"]["passed"] is False
    assert matbg_scale["status"] == "FAIL"
    assert certification["go8a_status"] == "NO-GO-8a"
    assert certification["physical_phonon_produced"] is False
    with pytest.raises(cert.MatbgPhononCertificationError):
        cert.require_go8a_certified_matbg_phonons(certification)
