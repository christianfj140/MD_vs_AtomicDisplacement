"""C21 publishes C19/C20's terminal finite-PAO verdict without new physics."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))

import evaluate_epc_metrics as go4  # noqa: E402
import preregister_graphene_gamma_epc as prereg  # noqa: E402

PROTOCOL_PATH = prereg.DEFAULT_OUTPUT_DIR / prereg.PROTOCOL_NAME
SUMMARY_PATH = go4.DEFAULT_RESULT_ROOT / go4.SUMMARY_NAME
FC_FD_DETAIL_PATH = go4.DEFAULT_RESULT_ROOT / "embedded_read_only_split_fc_fd_v2.json"

candidate_present = pytest.mark.skipif(
    not (PROTOCOL_PATH.is_file() and SUMMARY_PATH.is_file()),
    reason="the graphene Gamma candidate has not been produced",
)


@pytest.fixture
def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@candidate_present
def test_verdict_matches_the_two_levels(summary, protocol):
    report = go4.adjudicate(summary, protocol)
    assert report["levels"][go4.LEVEL_DERIVATIVE]["verdict"] == "PASS"
    assert report["levels"]["finite_PAO"]["verdict"] == "NO_GO:model_accuracy"
    assert report["levels"][go4.LEVEL_EPC]["verdict"] == "BLOCKED"
    assert report["verdict"] == "NO_GO:model_accuracy"
    assert report["levels"]["finite_PAO"]["gates_failed"] == [7]


@candidate_present
def test_only_a_pass_is_labelled_validated(summary, protocol):
    report = go4.adjudicate(summary, protocol)
    assert report["label"] is None
    assert report["candidate_result_class"] == "quantitative_g"
    assert report["full_KS"] == "BLOCKED: Delta_out"


@candidate_present
def test_every_failure_names_exactly_one_known_layer(summary, protocol):
    report = go4.adjudicate(summary, protocol)
    failed = {row["gate"] for row in report["gates"] if not row["passed"]}
    charged = {row["gate"]: row["layer"] for row in report["failure_attribution"]}
    assert set(charged) == failed
    assert set(charged.values()) <= set(go4.LAYERS)
    assert charged == {7: "model"}


@candidate_present
def test_model_accuracy_is_the_terminal_finite_pao_outcome(summary, protocol):
    report = go4.adjudicate(summary, protocol)
    statement = report["model_statement"]
    assert statement["charged_to_the_model"] is True
    assert statement["fine_tuning_ticket_authorized"] is False
    assert statement["quantitative_for_g"] is False
    assert statement["bound_covers_result"] is False
    assert statement["bound_holds_on_result"] is False
    assert statement["transfers_to_validation_split"] is True
    assert report["recommended_next_work"]
    assert "not authorized" in report["claim"]["claim"]


@candidate_present
def test_frozen_rules_are_applied_unchanged(summary, protocol):
    report = go4.adjudicate(summary, protocol)
    failed = {row["id"] for row in report["frozen_checks"] if not row["passed"]}
    assert failed == {"uniform_translation_null", "symmetry_of_the_doublet_at_K"}
    assert all(
        row["nonblocking_scope_finding"]
        for row in report["frozen_checks"]
        if row["id"] in failed
    )


@candidate_present
def test_a_three_path_failure_is_not_charged_to_a_backend(summary, protocol):
    report = go4.adjudicate(summary, protocol)
    findings = {row["check"] for row in report["nonblocking_protocol_findings"]}
    assert findings == {"uniform_translation_null", "symmetry_of_the_doublet_at_K"}
    assert summary["blocking_checks_failed"] == []


@candidate_present
def test_the_claim_rung_is_the_no_go_rung(summary, protocol):
    report = go4.adjudicate(summary, protocol)
    assert report["claim"]["rung"] == 2
    assert report["claim"]["also_active"] == []


@candidate_present
def test_a_missing_measurement_stops_the_adjudication(summary, protocol):
    """Absence is never a pass."""
    mutilated = copy.deepcopy(summary)
    mutilated["checks"] = [
        row
        for row in mutilated["checks"]
        if row.get("check") != "basis_response_present_and_required"
    ]
    with pytest.raises(go4.AdjudicationError, match="basis_response_present_and_required"):
        go4.adjudicate(mutilated, protocol)


@candidate_present
def test_a_candidate_that_does_not_cite_the_protocol_is_refused(summary, protocol):
    mutilated = copy.deepcopy(summary)
    mutilated["preregistration_sha256"] = "0" * 64
    with pytest.raises(go4.AdjudicationError, match="does not cite"):
        go4.adjudicate(mutilated, protocol)


@candidate_present
def test_an_edited_protocol_is_refused(summary, protocol):
    edited = copy.deepcopy(protocol)
    edited["tolerances"]["constants"]["G_ADEQUACY_THRESHOLD"] = 10.0
    with pytest.raises(go4.AdjudicationError, match="edited"):
        go4.adjudicate(summary, edited)


@candidate_present
def test_tau_model_records_what_it_may_not_be_derived_from(summary, protocol):
    """The one rule keeping a global matrix norm out of tau_model must be shown honoured."""
    outcome = go4.gate_6_model_against_reference(summary, protocol)
    assert outcome["not_derived_from"], "FORBIDDEN_TAU_MODEL_SOURCES was not carried through"
    assert any("global matrix norm" in row for row in outcome["not_derived_from"])
    assert "C_f^dag" in outcome["contracted_in"]

    stripped = copy.deepcopy(summary)
    stripped["tolerances"]["definition"]["tau_model"]["forbidden_sources"] = []
    with pytest.raises(go4.AdjudicationError, match="FORBIDDEN_TAU_MODEL_SOURCES"):
        go4.gate_6_model_against_reference(stripped, protocol)


@candidate_present
def test_gate_5_uses_the_validated_c14c_provider(summary, protocol):
    outcome = go4.gate_5_basis_response(summary, protocol)
    assert outcome["passed"] is True
    assert outcome["c14c_verdict"] == "PASS"
    assert outcome["provider_validated"] is True
    assert all(row["passed"] for row in outcome["evidence"])


@candidate_present
def test_persisted_ds_fc_fd_detail_round_trips_without_recalculation():
    report = json.loads(FC_FD_DETAIL_PATH.read_text(encoding="utf-8"))
    ds_rows = [row for row in report["comparisons"] if row["kind"] == "D_S"]
    assert ds_rows and report["verdict"] == "PASS"
    for row in ds_rows:
        for delta, detail in row["detail_by_delta"].items():
            assert detail["fc"] and detail["fd"] and detail["difference"]
            assert detail["difference_frobenius"] == pytest.approx(
                row["by_norm"]["frobenius"]["discrepancy_by_delta"][delta]
            )
            assert all(len(source["sha256"]) == 64 for source in detail["sources"].values())


@candidate_present
def test_a_clean_candidate_would_pass_and_be_labelled(summary, protocol):
    """The PASS publication branch follows the already-decided C20 verdict."""
    clean = copy.deepcopy(summary)
    clean["GO4_finite_PAO"] = "PASS"
    clean["verdict"] = "PASS"
    clean["graph2mat_quantitative_for_g"] = True
    clean["tolerances"]["tau_model"]["value"] = 0.01
    for split in ("validation", "result", "result_graph2mat_frozen"):
        for row in clean["tolerances"]["tau_model"][split]:
            row["relative_subspace_error"] = 0.005
    report = go4.adjudicate(clean, protocol)
    assert report["verdict"] == "PASS"
    assert report["label"] is None
    assert report["claim"]["rung"] == 1
    assert report["model_statement"]["charged_to_the_model"] is False
    assert report["model_statement"]["quantitative_for_g"] is True
