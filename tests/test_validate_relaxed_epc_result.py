"""E-F_001-S52: validate and interpret the relaxed-MATBG EPC result.

Builds a fresh S50 protocol, a fresh S51 refusal and fresh S43/S45 rigid GO-9
certifications in isolation (all three producers are already cheap and
self-contained -- no GPU, no checkpoint, no SIESTA) and checks S52 reads them
correctly, catches tampering, and never licenses a claim the evidence does
not support.
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
import preregister_matbg_relaxed_epc as epc_prereg  # noqa: E402
import run_matbg_relaxed_epc as s51  # noqa: E402
import validate_relaxed_epc_result as mod  # noqa: E402


@pytest.fixture()
def rigs(tmp_path, monkeypatch):
    """A self-contained S50 protocol + S51 summary, both on real files on disk."""
    protocol = epc_prereg.build_protocol()
    protocol_path = tmp_path / "prereg" / epc_prereg.PROTOCOL_NAME
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    result_root = tmp_path / "relaxed_epc"
    monkeypatch.setattr(s51.epc_prereg, "load_protocol", lambda *_a, **_k: protocol)
    monkeypatch.setattr(s51, "EPC_STATUS_PATH", result_root / "artifact_status.json")
    monkeypatch.setattr(s51, "EPC_SUMMARY_PATH", result_root / "relaxed_epc_summary.json")
    s51.run_relaxed_epc()

    return {
        "protocol_path": protocol_path,
        "summary_path": result_root / "relaxed_epc_summary.json",
        "result_root": result_root,
    }


@pytest.fixture(scope="module")
def rigid_reports(tmp_path_factory):
    s43 = go9_gamma.certify_rigid_gamma_campaign_go9(tmp_path_factory.mktemp("s43"))
    s45 = go9_qneq0.certify_rigid_qneq0_campaign_go9(tmp_path_factory.mktemp("s45"))
    return s43, s45


def test_protocol_chain_is_verified_for_an_untampered_pair(rigs):
    report = mod.protocol_chain_report(
        rigs["protocol_path"], rigs["summary_path"], result_root=rigs["result_root"]
    )
    assert report["protocol_intact"] is True
    assert report["results_cite_the_protocol"] is True
    for row in report["per_branch_blocking_reasons_unmodified"].values():
        assert row["matches_frozen_protocol"] is True
    assert report["verified"] is True


def test_protocol_chain_flags_a_blocking_list_edited_after_the_fact(rigs):
    summary = json.loads(rigs["summary_path"].read_text(encoding="utf-8"))
    summary["branches"]["gamma"]["blocking"] = ["a laxer reason invented after the fact"]
    rigs["summary_path"].write_text(json.dumps(summary), encoding="utf-8")

    report = mod.protocol_chain_report(
        rigs["protocol_path"], rigs["summary_path"], result_root=rigs["result_root"]
    )
    assert report["per_branch_blocking_reasons_unmodified"]["gamma"]["matches_frozen_protocol"] is False
    assert report["verified"] is False


def test_no_leaked_claims_report_is_clean_by_default(rigs):
    report = mod.no_leaked_claims_report(rigs["summary_path"])
    assert report["clean"] is True
    assert report["offending_fields"] == []


def test_no_leaked_claims_report_catches_a_forbidden_label(rigs):
    summary = json.loads(rigs["summary_path"].read_text(encoding="utf-8"))
    summary["branches"]["qneq0"]["claim"] = "g_mn_nu"
    rigs["summary_path"].write_text(json.dumps(summary), encoding="utf-8")

    report = mod.no_leaked_claims_report(rigs["summary_path"])
    assert report["clean"] is False
    assert any("qneq0.claim" in row for row in report["offending_fields"])


def test_rigid_comparison_reads_go9_status_live_not_hardcoded(rigs, rigid_reports):
    s43, s45 = rigid_reports
    report = mod.rigid_comparison_report(rigs["summary_path"], s43, s45)
    assert report["rigid"]["gamma"]["go9_status"] == s43["go9_status"]
    assert report["rigid"]["qneq0"]["go9_status"] == s45["go9_status"]
    assert set(report["relaxed"]) == set(s51.BRANCHES)
    assert report["either_geometry_has_a_certified_full_ks_coupling"] is False


def test_forbidden_labels_include_the_relaxed_vs_rigid_delta():
    assert "relaxed_vs_rigid_epc_quantitative_delta" in mod.FORBIDDEN_PUBLICATION_LABELS
    for label in s51.FORBIDDEN_PUBLICATION_LABELS:
        assert label in mod.FORBIDDEN_PUBLICATION_LABELS


def test_validate_and_interpret_assembles_a_full_report(rigs, rigid_reports):
    s43, s45 = rigid_reports
    report = mod.validate_and_interpret(
        rigs["protocol_path"], rigs["summary_path"], s43, s45, result_root=rigs["result_root"]
    )
    assert report["schema"] == mod.SCHEMA
    assert report["ticket"] == "E-F_001-S52"
    assert report["claims"]["all_licensed_claims_hold"] is True
    assert report["verdict"] == "NO_QUANTITATIVE_CLAIM_LICENSED"
    forbidden_claims = {row["claim"] for row in report["claims"]["forbidden"]}
    assert "quantitative_relaxed_matbg_prediction" in forbidden_claims
    assert "relaxed_vs_rigid_epc_quantitative_delta" in forbidden_claims


def test_validate_and_interpret_fails_closed_when_the_protocol_chain_breaks(rigs, rigid_reports):
    s43, s45 = rigid_reports
    summary = json.loads(rigs["summary_path"].read_text(encoding="utf-8"))
    summary["branches"]["gamma"]["blocking"] = ["tampered"]
    rigs["summary_path"].write_text(json.dumps(summary), encoding="utf-8")

    report = mod.validate_and_interpret(
        rigs["protocol_path"], rigs["summary_path"], s43, s45, result_root=rigs["result_root"]
    )
    assert report["claims"]["all_licensed_claims_hold"] is False
    assert report["verdict"] == "VALIDATION_FAILED"
