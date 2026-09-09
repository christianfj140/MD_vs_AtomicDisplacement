#!/usr/bin/env python3
"""E-F_001-S36 / D05: the small-TBG RUN / SKIP_WITH_JUSTIFICATION decision."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import decide_small_tbg as dt  # noqa: E402


def test_go7_verdict_artifact_is_present_for_this_decision():
    if not dt.GO7_VERDICT_PATH.exists():
        pytest.skip(f"GO-7 verdict not generated at {dt.GO7_VERDICT_PATH}")


def test_decision_is_undecided_until_both_upstream_gates_pass():
    if not dt.GO7_VERDICT_PATH.exists():
        pytest.skip(f"GO-7 verdict not generated at {dt.GO7_VERDICT_PATH}")

    decision = dt.small_tbg_decision()

    assert decision["verdict"] == "UNDECIDED"
    assert decision["status"] == "BLOCKED"

    go5_checks = decision["risks_closed"]["commensurate_supercell_and_q_periodicity_algebra"][
        "checks"
    ]
    assert set(go5_checks) == set(dt.GO5_MAPPING_RELEVANT_CHECKS)
    assert all(v == "PASS" for v in go5_checks.values())

    go7_checks = decision["risks_closed"]["multi_layer_geometry_construction"]["checks"]
    assert go7_checks
    assert not all(go7_checks.values())

    # The decision must not hide that GO-4/GO-5 remain blocked -- SKIP is not
    # the same as "everything upstream passed".
    assert "GO-4" in decision["not_evidence_for_running"]["go4_go5_blocker"]


def test_a_failing_mapping_check_remains_undecided(monkeypatch):
    """SKIP must be earned by the specific mapping checks, not asserted."""
    if not dt.GO7_VERDICT_PATH.exists():
        pytest.skip(f"GO-7 verdict not generated at {dt.GO7_VERDICT_PATH}")

    original = dt._ck.go5_decision

    def _broken_go5():
        result = original()
        result["checks"] = dict(result["checks"])
        result["checks"]["cell_origin_shift"] = "FAIL"
        return result

    monkeypatch.setattr(dt._ck, "go5_decision", _broken_go5)
    decision = dt.small_tbg_decision()
    assert decision["verdict"] == "UNDECIDED"


def test_missing_go7_verdict_raises_instead_of_defaulting():
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
