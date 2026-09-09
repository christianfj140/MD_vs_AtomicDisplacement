"""E-F_001-S27: the interpretation of the graphene Gamma residual.

The interpretation produces no physics, so what is worth testing is that it
cannot quietly become a second measurement of the gate: that the relative
subspace errors it decomposes are bit-for-bit the ones the frozen protocol
already recorded, that a result which does not cite the frozen protocol is
refused, that the retraining ticket follows the frozen claim ladder in both
directions, and that a null direction cannot be counted as the model's best
case.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))

import evaluate_epc_metrics as go4  # noqa: E402
import interpret_gamma_epc_error as s27  # noqa: E402
import preregister_graphene_gamma_epc as prereg  # noqa: E402

PROTOCOL_PATH = prereg.DEFAULT_OUTPUT_DIR / prereg.PROTOCOL_NAME
SUMMARY_PATH = s27.DEFAULT_RESULT_ROOT / go4.SUMMARY_NAME
VERDICT_PATH = s27.DEFAULT_RESULT_ROOT / go4.VERDICT_NAME
REPORT_PATH = s27.DEFAULT_OUTPUT_DIR / s27.REPORT_NAME

candidate_present = pytest.mark.skipif(
    not (PROTOCOL_PATH.is_file() and SUMMARY_PATH.is_file() and VERDICT_PATH.is_file()),
    reason="the graphene Gamma candidate has not been produced",
)
report_present = pytest.mark.skipif(
    not REPORT_PATH.is_file(), reason="the S27 interpretation has not been produced"
)


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def verdict() -> dict:
    return json.loads(VERDICT_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# the interpretation is not a second measurement
# --------------------------------------------------------------------------


@report_present
@candidate_present
def test_relative_subspace_errors_reproduce_the_frozen_ones(report, summary):
    """Every r the report decomposes is the r the frozen protocol recorded."""
    recomputed = {
        (row["direction"], row["k"]): row["windows"]["frozen_window"]["relative_subspace_error"]
        for row in report["subspace_level"]["rows"]
        if row["model_path"] == "graph2mat_jvp"
    }
    frozen = [
        entry
        for section in ("calibration", "validation", "result")
        for entry in summary["tolerances"]["tau_model"][section]
        if entry["model_path"] == "graph2mat_jvp"
    ]
    assert frozen, "the candidate records no tau_model entries to compare against"
    for entry in frozen:
        assert recomputed[(entry["direction"], entry["k"])] == pytest.approx(
            entry["relative_subspace_error"], rel=1e-9, abs=1e-12
        )


@report_present
def test_tau_model_and_threshold_are_read_not_rederived(report, summary):
    axis = report["sensitivity"]["tolerances"]
    assert axis["tau_model_frozen"] == summary["tolerances"]["tau_model"]["value"]
    assert (
        axis["adequacy_threshold"]
        == summary["tolerances"]["definition"]["constants"]["G_ADEQUACY_THRESHOLD"]
    )
    assert report["conclusion"]["adequacy"]["quantitative_for_g"] is False


@candidate_present
def test_a_result_that_does_not_cite_the_protocol_is_refused(tmp_path):
    """The hash guard fires before anything is interpreted."""
    for name in (go4.SUMMARY_NAME, go4.VERDICT_NAME):
        payload = json.loads((s27.DEFAULT_RESULT_ROOT / name).read_text(encoding="utf-8"))
        payload["preregistration_sha256"] = "0" * 64
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    for name in (s27.PATH_SIESTA, *s27.MODEL_PATHS):
        source = s27.DEFAULT_RESULT_ROOT / f"path__{name}.json"
        (tmp_path / f"path__{name}.json").write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    with pytest.raises(s27.InterpretationError, match="not the same experiment"):
        s27.load_inputs(tmp_path, PROTOCOL_PATH, s27.DEFAULT_C14_REPORT)


# --------------------------------------------------------------------------
# the residual carries no basis response
# --------------------------------------------------------------------------


@report_present
def test_the_basis_response_cancels_in_the_residual(report):
    cancellation = report["basis_response_cancellation"]
    assert cancellation["holds"] is True
    assert cancellation["worst_relative"] < 1e-9
    assert all(
        entry["basis_response_blocks_identical_across_paths"]
        for entry in cancellation["per_case"]
    )


@report_present
def test_the_open_intra_atomic_term_cannot_rescue_adequacy(report):
    rescue = report["intra_atomic_rescue"]
    assert rescue["smallest_required_multiple_of_dS"] > 1.0
    assert rescue["answer"].startswith("no")


def test_bloch_sum_uses_the_declared_convention():
    """``X(k) = sum_l exp(+2i pi k.R_l) X(R_l)``, checked against a hand sum."""
    isc_off = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)
    blocks = np.arange(3 * 2 * 2, dtype=np.float64).reshape(3, 2, 2)
    k = [0.25, 0.0, 0.0]
    expected = (
        blocks[0]
        + np.exp(2j * np.pi * 0.25) * blocks[1]
        + np.exp(-2j * np.pi * 0.25) * blocks[2]
    )
    assert np.allclose(s27.bloch_sum(isc_off, blocks, k), expected)


def test_dense_to_field_keys_by_lattice_vector():
    isc_off = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    blocks = np.zeros((2, 2, 2))
    blocks[1, 0, 1] = 3.5
    field = s27.dense_to_field(isc_off, blocks)
    assert field == {(0, 1, (1, 0, 0)): 3.5}


# --------------------------------------------------------------------------
# the window rule and the null-direction exclusion
# --------------------------------------------------------------------------


@candidate_present
def test_window_selections_come_from_the_electron_count():
    protocol = prereg.load_protocol(PROTOCOL_PATH)
    selections = s27.window_selections(protocol, 4)
    below = int(protocol["electronic"]["window"]["states_below"])
    assert selections["frozen_window"] == [0, 1, 2, 3]
    assert selections["neutrality_pair"] == [below - 1, below]
    assert selections["occupied_block"] == list(range(below))


@report_present
def test_the_null_direction_is_excluded_from_the_adequacy_statistic(report):
    """A direction with no derivative signal cannot be the model's best case."""
    axis = report["sensitivity"]["subspace_selection"]
    assert axis["null_directions_excluded"], "the null direction classification was not applied"
    excluded = set(axis["null_directions_excluded"])
    assert axis["smallest_relative_subspace_error"]["direction"] not in excluded
    assert axis["smallest_over_threshold"] > 1.0
    assert report["conclusion"]["adequacy"]["robust_to_subspace_choice"] is True


# --------------------------------------------------------------------------
# the retraining ticket follows the frozen ladder, in both directions
# --------------------------------------------------------------------------


def _conclude(inputs_stub) -> dict:
    return s27.conclude(
        inputs_stub,
        derivative_rows=inputs_stub.derivative_rows,
        subspace_rows=inputs_stub.subspace_rows,
        cancellation={"consequence": "measured", "worst_relative": 0.0},
        axes=inputs_stub.axes,
    )


class _Stub:
    """The minimum an interpretation needs to reach the ticket decision."""

    def __init__(self, verdict: dict, summary: dict, c14: dict):
        self.verdict = verdict
        self.summary = summary
        self.c14 = c14
        self.derivative_rows = [
            {
                "direction": "e2g_bond_longitudinal",
                "reference_frobenius": 54.0,
                "best_fit_scale": 0.62,
                "cosine": 0.94,
                "relative_frobenius": 0.67,
                "relative_after_best_fit_scale": 0.33,
            }
        ]
        self.subspace_rows = [
            {
                "direction": "e2g_bond_longitudinal",
                "k": "K",
                "windows": {
                    "neutrality_pair": {
                        "reference_frobenius": 13.5,
                        "relative_subspace_error": 0.77,
                        "relative_after_derivative_level_scale": 0.09,
                    }
                },
            }
        ]
        self.axes = {"subspace_selection": {"smallest_over_threshold": 5.6}}


@candidate_present
def test_ticket_stays_closed_while_a_layer_is_open(verdict, summary):
    c14_report = json.loads(s27.DEFAULT_C14_REPORT.read_text(encoding="utf-8"))
    conclusion = _conclude(_Stub(verdict, summary, c14_report))
    ticket = conclusion["retraining_ticket"]
    assert ticket["authorized"] is False
    assert ticket["status"] == "closed"
    assert {row["layer"] for row in ticket["blocked_by"]} == {"basis_response", "formalism"}
    assert ticket["activation_evidence"], "a closed ticket must still say what would open it"
    assert all(
        {"condition", "artifact", "becomes_true_when"} <= set(entry)
        for entry in ticket["activation_evidence"]
    )


@candidate_present
def test_ticket_opens_only_at_the_rung_the_ladder_names(verdict, summary):
    """The ladder requires the independent promotion prerequisites as well."""
    c14_report = json.loads(s27.DEFAULT_C14_REPORT.read_text(encoding="utf-8"))
    promoted = copy.deepcopy(verdict)
    promoted["claim"]["rung"] = s27.RETRAINING_RUNG
    promoted["model_statement"]["fine_tuning_ticket_authorized"] = True
    with pytest.raises(s27.InterpretationError, match="disagree"):
        _conclude(_Stub(promoted, summary, c14_report))
    from epc_claim_policy import claim_policy
    gates = dict.fromkeys(
        ("gauge_derivative_audit", "graph2mat_target_gauge_contract", "numerical_PAO_convergence",
         "checkpoint_lineage", "final_case_exclusion", "gpu_runtime", "GO-5", "GO-7"), "PASS")
    # Every gate green is still not enough: Graph2Mat's target lives in the
    # H_abs - E_F.S gauge and the PAO reference in H_abs - c_vac.S, and
    # (c_vac - E_F)' = 3.4869 meV/Ang against a 5 meV/Ang budget. The ticket
    # stays shut until the final comparator is verified to apply that conversion.
    promoted["model_statement"]["claim_policy"] = claim_policy(dict(gates))
    with pytest.raises(s27.InterpretationError, match="disagree"):
        _conclude(_Stub(promoted, summary, c14_report))
    promoted["model_statement"]["claim_policy"] = claim_policy(
        {**gates, "final_comparator_gauge_conversion_verified": True})
    conclusion = _conclude(_Stub(promoted, summary, c14_report))
    assert conclusion["retraining_ticket"]["authorized"] is True


@candidate_present
def test_a_verdict_that_contradicts_the_ladder_is_refused(verdict, summary):
    c14_report = json.loads(s27.DEFAULT_C14_REPORT.read_text(encoding="utf-8"))
    inconsistent = copy.deepcopy(verdict)
    inconsistent["model_statement"]["fine_tuning_ticket_authorized"] = True
    with pytest.raises(s27.InterpretationError, match="disagree"):
        _conclude(_Stub(inconsistent, summary, c14_report))


@report_present
def test_the_no_go_is_not_reported_as_a_result_about_graphene(report):
    claims = {entry["claim"] for entry in report["conclusion"]["what_the_residual_is_not"]}
    assert "it is not a negative physical result about graphene" in claims
    assert report["conclusion"]["retraining_ticket"]["why_the_evidence_survives_the_blockers"]
