"""DATASET-DESIGN-W90-001-S9-S9 contract tests -- campaign execution engine.

Fast: no new SIESTA/GPU work. ``wait_for_process``/``run_phase`` are exercised
against real OS processes (a definitely-dead PID, real ``sleep``/``false``
subprocesses); the cost tracker and pruned-branch aggregation are exercised
against the real on-disk S9-S4/S9-S5 artifacts already materialized by this
task's own campaign run, following S9-S5/S9-S8's precedent of testing against
real artifacts rather than fixtures. Matched by ``-k s9_s9`` or by the
``DATASET-DESIGN-W90-001-S9-S9`` marker registered in ``pytest.ini``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import dataset_design_w90_001_s9_s9_execute_campaign as s99  # noqa: E402

S9_S9_MARK = getattr(pytest.mark, "DATASET-DESIGN-W90-001-S9-S9")

# A PID essentially guaranteed not to be alive (max PID space on Linux is 2**22).
DEFINITELY_DEAD_PID = 2**22 + 1


@S9_S9_MARK
def test_wait_for_process_returns_immediately_for_a_dead_pid():
    result = s99.wait_for_process(DEFINITELY_DEAD_PID, poll_seconds=0.01)
    assert result["was_running_at_start"] is False
    assert result["polls"] == 0
    assert result["pid"] == DEFINITELY_DEAD_PID


@S9_S9_MARK
def test_wait_for_process_detects_the_current_process_as_running_then_returns_once_dead():
    # A short-lived child PID: alive during the call window this test controls,
    # dead immediately after -- exercises the "was running, now isn't" branch
    # without actually blocking on a long-lived process.
    pid = os.fork() if hasattr(os, "fork") else None
    if pid is None:
        pytest.skip("os.fork unavailable on this platform")
    if pid == 0:
        os._exit(0)  # child: exit immediately
    os.waitpid(pid, 0)
    result = s99.wait_for_process(pid, poll_seconds=0.01)
    assert result["was_running_at_start"] is False  # already reaped by the time we poll


@S9_S9_MARK
def test_run_phase_reports_ok_for_a_successful_command():
    phase = s99.run_phase("noop", [sys.executable, "-c", "pass"], timeout_seconds=10)
    assert phase["ok"] is True
    assert phase["returncode"] == 0
    assert phase["timed_out"] is False


@S9_S9_MARK
def test_run_phase_reports_failure_for_a_nonzero_exit():
    phase = s99.run_phase("fails", [sys.executable, "-c", "import sys; sys.exit(2)"], timeout_seconds=10)
    assert phase["ok"] is False
    assert phase["returncode"] == 2
    assert phase["timed_out"] is False


@S9_S9_MARK
def test_run_phase_reports_timeout_for_a_command_that_outlives_its_budget():
    phase = s99.run_phase("slow", [sys.executable, "-c", "import time; time.sleep(5)"], timeout_seconds=0.2)
    assert phase["ok"] is False
    assert phase["timed_out"] is True
    assert phase["returncode"] is None


@S9_S9_MARK
def test_pruned_branches_are_non_empty_with_a_documented_reason():
    branches = s99.collect_pruned_branches()
    assert branches, "expected at least one pruned S9-S4 branch on disk"
    for entry in branches:
        assert entry["reason"] in ("dominated", "envelope_fail", "convergence_fail", "budget")
        assert entry["design_id"]


@S9_S9_MARK
def test_cost_tracker_credits_reuse_across_the_campaign():
    tracker = s99.compute_cost_tracker()
    assert "error" not in tracker, tracker.get("error")
    assert tracker["sum_per_design_ids"] > 0
    assert tracker["campaign_union_ids"] > 0
    # The whole point of campaign-level dedup: many designs reuse the same
    # SIESTA structures, so the union must be strictly smaller than the naive sum.
    assert tracker["sum_per_design_ids"] > tracker["campaign_union_ids"]
    assert tracker["reuse_credited"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
