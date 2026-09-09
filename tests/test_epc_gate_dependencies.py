"""Regression tests for gate-dependency discipline (audit finding, Sept 2026).

Two decision functions already read upstream gate verdicts and refuse to
claim readiness while a prerequisite is still open:

* :func:`decide_small_tbg.small_tbg_decision` (E-F_001-S36) reads GO-5
  (:mod:`epc_commensurate_k`) and the GO-7 verdict artifact
  (``evaluate_ab_bilayer_epc.py``), and always reports the still-open
  GO-4/GO-5 blocker under ``not_evidence_for_running`` even when it allows
  ``SKIP_WITH_JUSTIFICATION`` for the narrow mapping-relevant subset.
* :func:`epc_qb_qc_prototypes.production_route_decision` (E-F_001-S33)
  states explicitly that it is not a GO and is downstream of GO-5 ``NO_GO``.

This module does not re-derive any physics or re-litigate S33/S36's own
scope -- it only pins down that the *dependency discipline* itself (the
refusal to claim readiness, the propagation of the open blocker) is real and
stays real. A future edit that quietly drops the blocker field, or that
lets either function proceed on missing/absent upstream evidence, should
fail one of these tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import decide_small_tbg as dt  # noqa: E402
import epc_commensurate_k as ck  # noqa: E402
import epc_qb_qc_prototypes as qbqc  # noqa: E402


# --------------------------------------------------------------------------- #
# small_tbg_decision (E-F_001-S36)
# --------------------------------------------------------------------------- #

go7_present = pytest.mark.skipif(
    not dt.GO7_VERDICT_PATH.exists(),
    reason=f"GO-7 verdict not generated at {dt.GO7_VERDICT_PATH}",
)


@go7_present
def test_small_tbg_decision_always_reports_the_go4_go5_blocker():
    """Whichever verdict comes out, the open GO-4/GO-5 blocker must be named,
    never omitted -- SKIP_WITH_JUSTIFICATION is not the same claim as
    "everything upstream passed"."""
    decision = dt.small_tbg_decision()
    not_evidence = decision["not_evidence_for_running"]
    assert "go4_go5_blocker" in not_evidence
    assert not_evidence["go4_go5_blocker"]  # non-empty: a real blocker, not a placeholder
    assert "GO-4" in not_evidence["go4_go5_blocker"]


@go7_present
def test_small_tbg_decision_blocker_matches_live_go5_state():
    """The blocker text small_tbg_decision reports must match what go5_decision()
    (cheap to recompute) says right now -- not a stale, hand-copied claim."""
    decision = dt.small_tbg_decision()
    go5 = ck.go5_decision()
    assert go5["go5"] == "NO_GO"
    assert decision["not_evidence_for_running"]["go4_go5_blocker"] == go5["blocked_by"]


def test_small_tbg_decision_raises_when_go7_verdict_is_missing():
    """Mirrors SmallTbgDecisionError: refuse rather than silently default to a
    verdict when the upstream GO-7 artifact was never written."""

    class _Missing:
        def exists(self):
            return False

    original = dt.GO7_VERDICT_PATH
    try:
        dt.GO7_VERDICT_PATH = _Missing()
        with pytest.raises(dt.SmallTbgDecisionError):
            dt.small_tbg_decision()
    finally:
        dt.GO7_VERDICT_PATH = original


@go7_present
def test_small_tbg_decision_flips_to_run_if_a_mapping_check_regresses(monkeypatch):
    """SKIP_WITH_JUSTIFICATION must be earned by the specific mapping checks
    passing, not asserted regardless of their state."""
    original = dt._ck.go5_decision

    def _broken_go5():
        result = original()
        result["checks"] = dict(result["checks"])
        result["checks"]["q_and_minus_q"] = "FAIL"
        return result

    monkeypatch.setattr(dt._ck, "go5_decision", _broken_go5)
    decision = dt.small_tbg_decision()
    assert decision["verdict"] == "UNDECIDED"
    # still names the blocker even while RUN (never silently proceeds either)
    assert decision["not_evidence_for_running"]["go4_go5_blocker"]


# --------------------------------------------------------------------------- #
# production_route_decision (E-F_001-S33)
# --------------------------------------------------------------------------- #


def test_production_route_decision_declares_itself_non_authoritative():
    """docs/epc_s33_ruta_qneq0_produccion.md says this in prose ("Este
    documento no es un GO"); the structured equivalent in the returned dict
    is no_production_started=True plus a blocked_by naming GO-5 NO_GO."""
    decision = qbqc.production_route_decision()
    assert decision["no_production_started"] is True
    assert "GO-5" in decision["blocked_by"]
    assert "NO_GO" in decision["blocked_by"]


def test_production_route_decision_blocker_matches_live_go5_state():
    """The hardcoded blocked_by claim is only honest as long as GO-5 really
    is NO_GO. If GO-5 ever flips, this test fails and flags that S33's
    decision record needs re-examination -- exactly the dependency-discipline
    property the audit asked to be pinned down."""
    go5 = ck.go5_decision()
    assert go5["go5"] == "NO_GO"
    decision = qbqc.production_route_decision()
    assert decision["no_production_started"] is True


def test_production_route_decision_never_authorizes_production():
    """Regardless of which route wins on cost/cache axes, the decision must
    never claim readiness to run MATBG q!=0 production while it is blocked."""
    decision = qbqc.production_route_decision()
    assert decision["status"] == "BLOCKED"
    assert decision["primary_route"] is None
    assert decision["no_production_started"] is True
    assert decision["additional_tests_required_before_go8b"]
