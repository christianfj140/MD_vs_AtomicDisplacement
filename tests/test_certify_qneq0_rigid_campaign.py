"""E-F_001-S45: GO-9 certification of the rigid-MATBG q != 0 (non-Gamma) branch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import certify_qneq0_rigid_campaign as cert  # noqa: E402


def _certify(tmp_path):
    return cert.certify_rigid_qneq0_campaign_go9(tmp_path)


def test_gate_authorization_suite_passes_against_the_real_frozen_protocol(tmp_path):
    certification = _certify(tmp_path)
    assert certification["gate_authorization_suite"]["status"] == "PASS", certification["gate_authorization_suite"]


def test_go5_and_go8b_dependency_suite_reads_live_gates_and_is_a_conjunction(tmp_path):
    certification = _certify(tmp_path)
    suite = certification["go5_and_go8b_dependency_suite"]
    assert suite["status"] == "PASS", suite
    assert suite["go5_decision"]["go5"] == "NO_GO"
    assert suite["go8b_certification"]["go8b_status"] == "NO-GO-8b"
    assert suite["full_ks_gates_open"] is False


def test_go9_gate_is_a_strict_conjunction_not_a_substitution():
    assert cert._qneq0_go9_gate(go5_status="PASS", go8b_status="GO-8b") is True
    assert cert._qneq0_go9_gate(go5_status="PASS", go8b_status="NO-GO-8b") is False
    assert cert._qneq0_go9_gate(go5_status="NO_GO", go8b_status="GO-8b") is False
    assert cert._qneq0_go9_gate(go5_status="NO_GO", go8b_status="NO-GO-8b") is False


def test_route_mechanics_and_reproducibility_suite_passes_on_the_real_kernel(tmp_path):
    certification = _certify(tmp_path)
    suite = certification["route_mechanics_and_reproducibility_suite"]
    assert suite["status"] == "PASS", suite


def test_phase_and_basis_response_guard_suite_passes(tmp_path):
    certification = _certify(tmp_path)
    suite = certification["phase_and_basis_response_guard_suite"]
    assert suite["status"] == "PASS", suite
    detection_check = next(
        entry for entry in suite["checks"]
        if entry["check"] == "phase_sign_error_is_detected_despite_well_formed_matrices"
    )
    assert detection_check["worst_abs_diff"] > 1e-3


def test_resource_budget_suite_is_honestly_not_run_not_fabricated(tmp_path):
    certification = _certify(tmp_path)
    suite = certification["resource_budget_suite"]
    assert suite["status"] == "NOT_RUN"
    assert "S44" in suite["blocked_by"]
    entry = suite["checks"][0]
    assert entry["applicable"] is False
    assert "live_machine_headroom_snapshot" in suite


def test_overall_verdict_is_no_go9_until_go5_and_go8b_and_resources_are_measured(tmp_path):
    certification = _certify(tmp_path)
    assert certification["go9_status"] == "NO-GO-9"
    assert certification["rigid_qneq0_campaign_certified"] is False
    assert certification["gate"] == "GO-9 (q != 0 branch)"


def test_final_and_forbidden_publication_labels_are_pinned(tmp_path):
    certification = _certify(tmp_path)
    assert certification["final_publication_label"] == "rigid_tbg_model"
    assert "quantitative_relaxed_matbg_prediction" in certification["forbidden_publication_labels"]
    assert "g_mn_nu" in certification["forbidden_publication_labels"]
    assert "phonon_eigenmode" in certification["forbidden_publication_labels"]


def test_require_go9_certified_raises_on_no_go(tmp_path):
    certification = _certify(tmp_path)
    with pytest.raises(cert.RigidQneq0CampaignCertificationError):
        cert.require_go9_certified_qneq0_rigid_campaign(certification)


def test_require_go9_certified_raises_on_malformed_object():
    with pytest.raises(cert.RigidQneq0CampaignCertificationError):
        cert.require_go9_certified_qneq0_rigid_campaign({"gate": "some_other_gate", "go9_status": "GO-9"})
    with pytest.raises(cert.RigidQneq0CampaignCertificationError):
        cert.require_go9_certified_qneq0_rigid_campaign({})


def test_require_go9_certified_accepts_a_passing_certification():
    passing = {"gate": cert.GATE, "go9_status": "GO-9", "extra": 1}
    assert cert.require_go9_certified_qneq0_rigid_campaign(passing) == passing
