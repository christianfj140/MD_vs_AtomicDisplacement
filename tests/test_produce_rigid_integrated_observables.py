"""E-F_001-S54: produce and interpret rigid-model integrated EPC observables.

Builds fresh S43/S45 rigid GO-9 certifications in isolation (self-contained,
no GPU/checkpoint/SIESTA) and checks S54 reads them live, refuses correctly
while g_set is empty, and never licenses a quantitative claim the evidence
does not support.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "Comparison/scripts"
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import certify_qneq0_rigid_campaign as go9_qneq0  # noqa: E402
import certify_rigid_gamma_campaign as go9_gamma  # noqa: E402
import produce_rigid_integrated_observables as mod  # noqa: E402


@pytest.fixture(scope="module")
def rigid_reports(tmp_path_factory):
    s43_dir = tmp_path_factory.mktemp("s43")
    s45_dir = tmp_path_factory.mktemp("s45")
    s43 = go9_gamma.certify_rigid_gamma_campaign_go9(s43_dir)
    s45 = go9_qneq0.certify_rigid_qneq0_campaign_go9(s45_dir)
    s43_path = s43_dir / "s43.json"
    s45_path = s45_dir / "s45.json"
    s43_path.write_text(json.dumps(s43), encoding="utf-8")
    s45_path.write_text(json.dumps(s45), encoding="utf-8")
    return s43_path, s45_path, s43, s45


def test_rigid_go9_status_reads_live_not_hardcoded(rigid_reports):
    s43_path, s45_path, s43, s45 = rigid_reports
    status = mod.rigid_go9_status(s43_path, s45_path)
    assert status["gamma"]["go9_status"] == s43["go9_status"]
    assert status["qneq0"]["go9_status"] == s45["go9_status"]


def test_produce_blocks_today_and_never_fabricates(rigid_reports):
    s43_path, s45_path, _s43, _s45 = rigid_reports
    result = mod.produce(s43_path, s45_path)

    assert result["state"] == "blocked"
    assert result["result_status"] == mod.RESULT_STATUS == "candidate_rigid_tbg"
    assert result["g_set"]["status"] == "not_attempted"
    assert set(result["observables"]) == set(mod.OBSERVABLES)
    for row in result["observables"].values():
        assert row["status"] == "not_attempted"
        assert row["reason"] == "blocked by g_set"
    for backend_field in (
        "electronic_derivative_backend",
        "basis_response_backend",
        "phonon_backend",
        "epc_backend_class",
    ):
        assert result[backend_field] == "not_attempted"
    assert result["blocking"]


def test_interpret_licenses_no_quantitative_claim(rigid_reports):
    s43_path, s45_path, _s43, _s45 = rigid_reports
    production = mod.produce(s43_path, s45_path)
    interpretation = mod.interpret(production)

    assert interpretation["all_licensed_claims_hold"] is True
    assert interpretation["verdict"] == "NO_QUANTITATIVE_CLAIM_LICENSED"
    forbidden_claims = {row["claim"] for row in interpretation["forbidden"]}
    for label in mod.FORBIDDEN_PUBLICATION_LABELS:
        assert label in forbidden_claims
    assert "Tc" in forbidden_claims


def test_interpret_fails_closed_if_a_branch_falsely_claims_go9(rigid_reports):
    s43_path, s45_path, _s43, _s45 = rigid_reports
    production = mod.produce(s43_path, s45_path)
    tampered = {**production, "rigid_go9_status": {**production["rigid_go9_status"]}}
    tampered["rigid_go9_status"]["gamma"] = {**tampered["rigid_go9_status"]["gamma"], "go9_status": "GO-9"}

    interpretation = mod.interpret(tampered)
    assert interpretation["licensed"][1]["licensed"] is False
    assert interpretation["all_licensed_claims_hold"] is False
    assert interpretation["verdict"] == "VALIDATION_FAILED"


def test_run_writes_a_summary_citing_the_real_s53_protocol(tmp_path, rigid_reports):
    s43_path, s45_path, _s43, _s45 = rigid_reports
    result = mod.run(s43_path, s45_path, output_dir=tmp_path)

    assert result["schema"] == mod.SCHEMA
    assert result["ticket"] == "E-F_001-S54"
    assert result["overall_claim"] == "no_execution_authorised"
    assert result["interpretation"]["verdict"] == "NO_QUANTITATIVE_CLAIM_LICENSED"
    for label in ("quantitative_relaxed_matbg_prediction", "g_mn_nu", "phonon_eigenmode", "Tc"):
        assert label in result["forbidden_publication_labels"]
    assert (tmp_path / mod.REPORT_NAME).is_file()


def test_main_exits_zero_on_the_expected_refusal(tmp_path, rigid_reports, capsys):
    s43_path, s45_path, _s43, _s45 = rigid_reports
    exit_code = mod.main(
        ["--s43-report", str(s43_path), "--s45-report", str(s45_path), "--output-dir", str(tmp_path)]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "no_execution_authorised" in captured.out
