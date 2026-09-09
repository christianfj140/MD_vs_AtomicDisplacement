"""E-F_001-S55: relaxed-MATBG integrated-observables convergence protocol.

GO-10 (S47) is hardcoded NO-GO-10 today, so this ticket must read that live
and refuse -- never fabricate a relaxed observable and never silently reuse
S53's rigid convergence ladder as if it applied to the relaxed geometry.
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

import converge_relaxed_integrated_observables as mod  # noqa: E402


def test_relaxed_g_set_status_reads_go10_live_and_finds_no_g(tmp_path):
    status = mod.relaxed_g_set_status(tmp_path / "no_such_summary.json")
    assert status["go10_verdict"] == "NO-GO-10"
    assert status["any_branch_has_g"] is False
    assert set(status["branches"]) == set(mod.s51.BRANCHES)


def test_build_protocol_blocks_today_and_does_not_copy_s53_conclusions(tmp_path):
    protocol = mod.build_protocol(summary_path=tmp_path / "missing.json")

    assert protocol["ready_to_execute"] is False
    assert protocol["blocking"]
    assert protocol["scope"]["inherits_rigid_conclusions"] is False
    # the mesh/broadening ladders are reused as *starting points*, not results.
    assert protocol["mesh"]["default_density_ladder_starting_point"] == list(mod.s53.DEFAULT_MESH_DENSITY_LADDER)
    assert "must_be_reevaluated_against" in protocol["mesh"]
    assert protocol["frozen_content_sha256"] == mod.prereg.freeze_hash(protocol)


def test_rigid_vs_relaxed_delta_report_never_fires_with_missing_side():
    report = mod.rigid_vs_relaxed_delta_report(None, None, None, None)
    assert report["status"] == "not_attempted"

    report = mod.rigid_vs_relaxed_delta_report(1.0, 0.1, None, None)
    assert report["status"] == "not_attempted"


def test_rigid_vs_relaxed_delta_report_gates_on_combined_uncertainty():
    below = mod.rigid_vs_relaxed_delta_report(1.0, 0.05, 1.02, 0.05)
    assert below["status"] == "computed"
    assert below["exceeds_combined_uncertainty"] is False
    assert below["interpretable"] is False

    above = mod.rigid_vs_relaxed_delta_report(1.0, 0.01, 1.5, 0.01)
    assert above["exceeds_combined_uncertainty"] is True
    assert above["interpretable"] is True


def test_produce_blocks_and_never_fabricates(tmp_path):
    result = mod.produce(tmp_path / "missing_summary.json", tmp_path / "missing_rigid.json")

    assert result["state"] == "blocked"
    assert result["result_status"] == mod.RESULT_STATUS == "candidate_relaxed_tbg"
    assert result["g_set"]["status"] == "not_attempted"
    assert set(result["observables"]) == set(mod.OBSERVABLES)
    for row in result["observables"].values():
        assert row["status"] == "not_attempted"
    for row in result["rigid_vs_relaxed_delta"].values():
        assert row["status"] == "not_attempted"
    for backend_field in (
        "electronic_derivative_backend", "basis_response_backend", "phonon_backend", "epc_backend_class",
    ):
        assert result[backend_field] == "not_attempted"
    assert result["backend"]["effective_backend"] == "cpu"


def test_interpret_licenses_no_quantitative_claim(tmp_path):
    production = mod.produce(tmp_path / "missing_summary.json", tmp_path / "missing_rigid.json")
    interpretation = mod.interpret(production)

    assert interpretation["all_licensed_claims_hold"] is True
    assert interpretation["verdict"] == "NO_QUANTITATIVE_CLAIM_LICENSED"
    forbidden_claims = {row["claim"] for row in interpretation["forbidden"]}
    for label in mod.FORBIDDEN_PUBLICATION_LABELS:
        assert label in forbidden_claims
    assert "Tc" in forbidden_claims
    assert "relaxed_vs_rigid_integrated_observable_quantitative_delta" in forbidden_claims


def test_interpret_fails_closed_if_a_delta_is_falsely_marked_computed(tmp_path):
    production = mod.produce(tmp_path / "missing_summary.json", tmp_path / "missing_rigid.json")
    tampered = {**production, "rigid_vs_relaxed_delta": dict(production["rigid_vs_relaxed_delta"])}
    tampered["rigid_vs_relaxed_delta"]["g_squared"] = {"status": "computed"}

    interpretation = mod.interpret(tampered)
    assert interpretation["licensed"][2]["licensed"] is False
    assert interpretation["all_licensed_claims_hold"] is False
    assert interpretation["verdict"] == "VALIDATION_FAILED"


def test_run_writes_a_report_citing_the_frozen_protocol(tmp_path):
    result = mod.run(tmp_path / "missing_summary.json", tmp_path / "missing_rigid.json", output_dir=tmp_path)

    assert result["ticket"] == mod.TICKET == "E-F_001-S55"
    assert result["overall_claim"] == "no_execution_authorised"
    assert result["interpretation"]["verdict"] == "NO_QUANTITATIVE_CLAIM_LICENSED"
    output_path = tmp_path / mod.REPORT_NAME
    assert output_path.is_file()
    on_disk = json.loads(output_path.read_text(encoding="utf-8"))
    assert on_disk["production"]["preregistration_sha256"] == result["production"]["preregistration_sha256"]


def test_main_exits_zero_on_the_expected_refusal(tmp_path, capsys):
    exit_code = mod.main([
        "--summary", str(tmp_path / "missing_summary.json"),
        "--s54-report", str(tmp_path / "missing_rigid.json"),
        "--protocol-dir", str(tmp_path / "protocol"),
        "--output-dir", str(tmp_path / "result"),
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "no_execution_authorised" in captured.out


def test_verify_detects_tampering(tmp_path):
    protocol_dir = tmp_path / "protocol"
    protocol = mod.build_protocol(summary_path=tmp_path / "missing.json")
    protocol_dir.mkdir(parents=True)
    protocol_path = protocol_dir / mod.PROTOCOL_NAME
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    intact = mod.verify(protocol_path)
    assert intact["verified"] is True

    tampered = {**protocol, "ready_to_execute": True}
    protocol_path.write_text(json.dumps(tampered), encoding="utf-8")
    caught = mod.verify(protocol_path)
    assert caught["verified"] is False

    with pytest.raises(mod.PreregistrationError):
        mod.require_frozen_protocol(protocol_path)
