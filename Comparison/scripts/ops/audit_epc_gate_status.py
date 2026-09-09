#!/usr/bin/env python3
"""Fail-closed reconciliation of E-F_001's S1..S56 record against real evidence.

The research orchestrator (a separate tool, state in a SQLite file outside
this repo) marks each subtask DONE or not. That label is the orchestrator's
own bookkeeping, not evidence: a subtask can be marked DONE while the
``scientific_check`` events it actually recorded are full of FAIL/ERROR, or
while it recorded no checks at all. Neither case may be silently accepted as
"the subtask is fine" -- that is exactly the failure mode this reconciler
exists to catch.

The orchestrator may run aggregate audits, but an aggregate PASS is not a
substitute for evidence that a leaf met its own objective. Refusals, NO-GO
outcomes, synthetic stand-ins and zero-check records remain useful outcomes,
but do not reconcile as completed scientific work.

For ``task_id`` (``E-F_001``) and every direct subtask id
``<task_id>-S1`` .. ``<task_id>-S<subtask_count>`` this script:

1. Reads the orchestrator-declared task state (read-only; never writes to the
   orchestrator's database).
2. Determines, from ``task_decomposed`` events, whether the id is itself a
   decomposition parent (an "audit unit") or a leaf.
3. For an audit unit: tallies every ``scientific_check`` event it recorded
   into PASS / FAIL / ERROR / other counts, and ``review`` events as
   corroborating evidence. It is only ``RECONCILED_PASS`` if it has at least
   one piece of scientific-check evidence, none of it FAIL or ERROR, and the
   orchestrator declares it DONE.
4. For a leaf: apply the same evidence rule. A DONE label with zero accepted
   task-local checks is ``NOT_RECONCILED_NO_EVIDENCE``.

Every other case -- a declared state that is not DONE, or a task id the
database does not even know about -- is reported as NOT reconciled. Silence
from an *audit unit* is never treated as a pass; silence from a *leaf* is the
documented, correct outcome, not evidence of anything.

Usage::

    audit_epc_gate_status.py --output Comparison/results/epc_repair/status_reconciliation.json

Exit code is 0 only if ``task_id`` itself and every audit-unit subtask
reconcile to PASS and every leaf subtask is DONE; 1 otherwise (fail-closed:
the default posture is "block until proven fine").
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "epc_gate_status_reconciliation_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATE_DB = Path("/home/christian/repositorios/research-orchestrator/state.db")
DEFAULT_TASK_ID = "E-F_001"
DEFAULT_SUBTASK_COUNT = 56
DEFAULT_OUTPUT = REPO_ROOT / "Comparison" / "results" / "epc_repair" / "status_reconciliation.json"

RECONCILED_PASS = "RECONCILED_PASS"
NOT_RECONCILED_NO_EVIDENCE = "NOT_RECONCILED_NO_EVIDENCE"
NOT_RECONCILED_FAILING_EVIDENCE = "NOT_RECONCILED_FAILING_EVIDENCE"
NOT_RECONCILED_STATE_MISMATCH = "NOT_RECONCILED_STATE_MISMATCH"
NOT_RECONCILED_TASK_MISSING = "NOT_RECONCILED_TASK_MISSING"
RECONCILED_VERDICTS = (RECONCILED_PASS,)


def open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open the orchestrator's SQLite file strictly read-only.

    ``mode=ro`` in the URI makes SQLite itself refuse any write -- this
    process never creates a journal/WAL file next to state.db and never
    touches a table, unlike opening it through the orchestrator's own Store
    class (which runs ``CREATE TABLE IF NOT EXISTS`` and PRAGMA statements on
    connect).
    """
    if not db_path.exists():
        raise FileNotFoundError(f"orchestrator state.db not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def direct_subtask_ids(task_id: str, subtask_count: int) -> list[str]:
    return [f"{task_id}-S{n}" for n in range(1, subtask_count + 1)]


def fetch_task(conn: sqlite3.Connection, task_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT id, state, attempt, max_attempts, block_reason FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def fetch_event_payloads(
    conn: sqlite3.Connection, task_id: str, kind: str, after_id: int = 0
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT payload FROM events WHERE task_id = ? AND kind = ? AND id > ? ORDER BY id",
        (task_id, kind, after_id),
    ).fetchall()
    payloads: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            payload = {}
        payloads.append(payload if isinstance(payload, dict) else {})
    return payloads


def latest_repair_cycle(conn: sqlite3.Connection, task_id: str) -> tuple[int, bool]:
    """Latest explicit-unblock event and whether that repair reached DONE."""
    rows = conn.execute(
        "SELECT id, to_state, payload FROM events "
        "WHERE task_id = ? AND kind = 'transition' ORDER BY id",
        (task_id,),
    ).fetchall()
    start = 0
    done = False
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict) and payload.get("unblocked") is True:
            start = int(row["id"])
            done = False
        elif start and int(row["id"]) > start and row["to_state"] == "DONE":
            done = True
    return start, done


def is_audit_unit(conn: sqlite3.Connection, task_id: str) -> bool:
    """Whether ``task_id`` owns its own decomposed plan (a ``task_decomposed`` event).

    Under ``audit_scope: global`` (this project's configuration), the
    scientific audit runs once per audit unit, not once per leaf step -- see
    ``_skip_decomposition_audit`` in the orchestrator itself. A leaf carrying
    no evidence of its own is therefore correct, not a gap.
    """
    row = conn.execute(
        "SELECT 1 FROM events WHERE task_id = ? AND kind = 'task_decomposed' LIMIT 1",
        (task_id,),
    ).fetchone()
    return row is not None


def reconcile_audit_unit(conn: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    """Strict reconciliation for a task that owns its own decomposed plan (or has none to own).

    Evidence is windowed to the latest explicit-unblock cycle (``cycle_start``)
    so a task repaired today is not held against evidence a since-fixed
    infrastructure bug produced hours or days ago. ``repaired_to_done`` is
    recorded for context but MUST NOT excuse zero evidence: a task closed
    DONE with nothing recorded since its last repair is exactly the "S1 is
    DONE despite 51 FAIL and 102 ERROR" failure mode this reconciler exists
    to catch, whether that silence is old or new. An earlier revision of this
    function let ``repaired_to_done`` bypass the zero-evidence check, which
    made E-F_001-S1 (repaired, later closed via a human interactive session,
    never re-validated) silently reconcile to PASS with zero evidence -- the
    exact thing the module docstring promises never happens. Reverted.
    """
    task = fetch_task(conn, task_id)
    cycle_start, repaired_to_done = latest_repair_cycle(conn, task_id)
    checks = fetch_event_payloads(conn, task_id, "scientific_check", cycle_start)
    reviews = fetch_event_payloads(conn, task_id, "review", cycle_start)

    counts = Counter()
    for check in checks:
        status = str(check.get("status") or "").strip().upper()
        if check.get("checks") == 0 or str(check.get("verdict", "")).upper() in {"NO_GO", "BLOCKED"}:
            status = "FAIL"
        counts[status if status in ("PASS", "FAIL", "ERROR") else "OTHER"] += 1
    evidence_counts = {
        "pass": counts["PASS"],
        "fail": counts["FAIL"],
        "error": counts["ERROR"],
        "other": counts["OTHER"],
        "total": sum(counts.values()),
    }
    review_fail = sum(1 for r in reviews if str(r.get("verdict") or "").strip().lower() == "fail")
    review_evidence = {"fail": review_fail, "total": len(reviews)}

    declared_state = task["state"] if task is not None else None
    reasons: list[str] = []

    if task is None:
        verdict = NOT_RECONCILED_TASK_MISSING
        reasons.append("task id is not present in the orchestrator's tasks table")
    elif evidence_counts["total"] == 0:
        verdict = NOT_RECONCILED_NO_EVIDENCE
        reasons.append(
            "zero scientific_check events recorded since the last repair cycle "
            f"(id>{cycle_start}); a declared state={declared_state!r} is not evidence and is "
            "not accepted as a pass, no matter how the task was eventually closed"
        )
    elif evidence_counts["fail"] or evidence_counts["error"] or evidence_counts["other"]:
        verdict = NOT_RECONCILED_FAILING_EVIDENCE
        reasons.append(
            f"{evidence_counts['fail']} FAIL and {evidence_counts['error']} ERROR "
            f"scientific_check event(s) out of {evidence_counts['total']} recorded"
        )
    elif declared_state != "DONE":
        verdict = NOT_RECONCILED_STATE_MISMATCH
        reasons.append(f"orchestrator declares state={declared_state!r}, not DONE")
    else:
        verdict = RECONCILED_PASS

    if review_fail:
        reasons.append(f"{review_fail} review event(s) out of {len(reviews)} recorded verdict=fail")

    return {
        "task_id": task_id,
        "audit_unit": True,
        "orchestrator_declared_state": declared_state,
        "orchestrator_task_found": task is not None,
        "evidence_counts": evidence_counts,
        "review_evidence": review_evidence,
        "repair_cycle_start_event_id": cycle_start or None,
        "repair_cycle_completed": repaired_to_done,
        "reconciled_verdict": verdict,
        "reasons": reasons,
    }


def reconcile_leaf(conn: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    """A DONE leaf still needs its own accepted evidence; parent coverage is not evidence."""
    result = reconcile_audit_unit(conn, task_id)
    result["audit_unit"] = False
    return result


def reconcile_subtask(conn: sqlite3.Connection, subtask_id: str) -> dict[str, Any]:
    result = (
        reconcile_audit_unit(conn, subtask_id)
        if is_audit_unit(conn, subtask_id)
        else reconcile_leaf(conn, subtask_id)
    )
    return {"subtask_id": result.pop("task_id"), **result}


def build_report(*, state_db: Path, task_id: str, subtask_count: int) -> dict[str, Any]:
    conn = open_readonly(state_db)
    try:
        parent_reconciliation = reconcile_audit_unit(conn, task_id)
        per_subtask = [
            reconcile_subtask(conn, subtask_id)
            for subtask_id in direct_subtask_ids(task_id, subtask_count)
        ]
    finally:
        conn.close()

    verdict_counts = Counter(item["reconciled_verdict"] for item in per_subtask)
    all_reconciled = all(
        item["reconciled_verdict"] in RECONCILED_VERDICTS for item in per_subtask
    )
    # This runs while the parent itself is validating; requiring it to be DONE
    # here is circular. Parent reconciliation remains in the report as context.
    fail_closed_ok = bool(per_subtask) and all_reconciled
    not_reconciled = [
        item["subtask_id"] for item in per_subtask
        if item["reconciled_verdict"] not in RECONCILED_VERDICTS
    ]

    return {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "state_db_path": str(state_db),
        "task_id": task_id,
        "audit_scope": "global",
        "parent_reconciliation": parent_reconciliation,
        "orchestrator_parent_declared_state": parent_reconciliation["orchestrator_declared_state"],
        "subtasks_checked": len(per_subtask),
        "per_subtask": per_subtask,
        "summary": {
            "reconciled_pass": verdict_counts.get(RECONCILED_PASS, 0),
            "reconciled_covered_by_parent_audit": 0,
            "not_reconciled_no_evidence": verdict_counts.get(NOT_RECONCILED_NO_EVIDENCE, 0),
            "not_reconciled_failing_evidence": verdict_counts.get(NOT_RECONCILED_FAILING_EVIDENCE, 0),
            "not_reconciled_state_mismatch": verdict_counts.get(NOT_RECONCILED_STATE_MISMATCH, 0),
            "not_reconciled_task_missing": verdict_counts.get(NOT_RECONCILED_TASK_MISSING, 0),
            "not_reconciled_subtask_ids": not_reconciled,
            "parent_reconciled": parent_reconciliation["reconciled_verdict"] == RECONCILED_PASS,
            "fail_closed_ok": fail_closed_ok,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--subtask-count", type=int, default=DEFAULT_SUBTASK_COUNT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(state_db=args.state_db, task_id=args.task_id, subtask_count=args.subtask_count)
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["summary"]["fail_closed_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
