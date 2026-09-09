"""E-F_001-S43: GO-9 certification of the rigid-Gamma raw-derivative campaign."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import certify_rigid_gamma_campaign as cert  # noqa: E402


def _certify(tmp_path):
    return cert.certify_rigid_gamma_campaign_go9(tmp_path)


def test_gate_authorization_suite_passes_against_the_real_frozen_protocol(tmp_path):
    certification = _certify(tmp_path)
    assert certification["gate_authorization_suite"]["status"] == "PASS", certification["gate_authorization_suite"]


def test_reproducibility_suite_passes_cold_warm_and_restart(tmp_path):
    certification = _certify(tmp_path)
    assert certification["reproducibility_suite"]["status"] == "PASS", certification["reproducibility_suite"]


def test_reorder_and_interruption_suite_passes(tmp_path):
    certification = _certify(tmp_path)
    assert certification["reorder_and_interruption_suite"]["status"] == "PASS", certification["reorder_and_interruption_suite"]


def test_synthetic_dependency_guard_suite_passes(tmp_path):
    certification = _certify(tmp_path)
    assert certification["synthetic_dependency_guard_suite"]["status"] == "PASS", certification["synthetic_dependency_guard_suite"]


def test_resource_budget_suite_is_honestly_not_run_not_fabricated(tmp_path):
    certification = _certify(tmp_path)
    suite = certification["resource_budget_suite"]
    assert suite["status"] == "NOT_RUN"
    assert suite["blocked_by"] == "S42 Sec. 6 (no measured production run)"
    entry = suite["checks"][0]
    assert entry["applicable"] is False
    assert "live_machine_headroom_snapshot" in suite


def test_overall_verdict_is_no_go9_until_a_real_run_is_measured(tmp_path):
    certification = _certify(tmp_path)
    assert certification["go9_status"] == "NO-GO-9"
    assert certification["rigid_gamma_campaign_certified"] is False
    assert certification["gate"] == "GO-9"


def test_final_and_forbidden_publication_labels_are_pinned(tmp_path):
    certification = _certify(tmp_path)
    assert certification["final_publication_label"] == "rigid_tbg_model"
    assert "quantitative_relaxed_matbg_prediction" in certification["forbidden_publication_labels"]
    assert "g_mn_nu" in certification["forbidden_publication_labels"]
    assert "phonon_eigenmode" in certification["forbidden_publication_labels"]


def test_certifier_never_touches_the_real_production_campaign_state(tmp_path):
    real_status = REPO_ROOT / "Comparison/results/epc/gamma_epc/matbg/artifact_status.json"
    existed_before = real_status.exists()
    _certify(tmp_path)
    assert real_status.exists() == existed_before, (
        "the certifier must be fully isolated to its own tmp_path (EPC_ROOT/EPC_ARRAYS_DIR/EPC_STATUS_PATH "
        "all patched) -- this is the exact bug this ticket found and fixed in "
        "tests/test_run_tbg_gamma_rigid_epc.py, which used to leak a 'synthetic failure' entry into this file"
    )


def test_require_go9_certified_raises_on_no_go(tmp_path):
    certification = _certify(tmp_path)
    with pytest.raises(cert.RigidGammaCampaignCertificationError):
        cert.require_go9_certified_rigid_gamma_campaign(certification)


def test_require_go9_certified_raises_on_malformed_object():
    with pytest.raises(cert.RigidGammaCampaignCertificationError):
        cert.require_go9_certified_rigid_gamma_campaign({"gate": "some_other_gate", "go9_status": "GO-9"})
    with pytest.raises(cert.RigidGammaCampaignCertificationError):
        cert.require_go9_certified_rigid_gamma_campaign({})


def test_require_go9_certified_accepts_a_passing_certification():
    passing = {"gate": "GO-9", "go9_status": "GO-9", "extra": 1}
    assert cert.require_go9_certified_rigid_gamma_campaign(passing) == passing
