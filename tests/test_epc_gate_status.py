from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "ops" / "audit_epc_gate_status.py"
SPEC = importlib.util.spec_from_file_location("audit_epc_gate_status", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REAL_STATE_DB = Path("/home/christian/repositorios/research-orchestrator/state.db")


def test_successful_execution_does_not_override_negative_science(tmp_path):
    path = tmp_path / "state.db"
    make_fake_orchestrator_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        for evidence in ({"status": "PASS", "checks": 0},
                         {"status": "PASS", "verdict": "NO_GO"},
                         {"status": "PASS", "verdict": "BLOCKED"}):
            conn.execute("DELETE FROM events WHERE task_id='T-S3' AND kind='scientific_check'")
            conn.execute("INSERT INTO events (task_id,kind,payload) VALUES ('T-S3','scientific_check',?)",
                         (json.dumps(evidence),))
            assert MODULE.reconcile_subtask(conn, "T-S3")["reconciled_verdict"] != MODULE.RECONCILED_PASS


def make_fake_orchestrator_db(path: Path) -> None:
    """A tiny stand-in for the orchestrator's state.db, same three tables."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT, type TEXT, state TEXT,
            implementer TEXT, reviewer TEXT, attempt INTEGER, max_attempts INTEGER,
            dependencies TEXT, requires_human_approval INTEGER, block_reason TEXT,
            spec TEXT, created_at TEXT, updated_at TEXT
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, run_id TEXT, task_id TEXT,
            kind TEXT, from_state TEXT, to_state TEXT, payload TEXT
        );
        """
    )

    def add_task(task_id: str, state: str) -> None:
        conn.execute(
            "INSERT INTO tasks (id, title, type, state, attempt, max_attempts, dependencies, "
            "requires_human_approval, spec, created_at, updated_at) "
            "VALUES (?, 't', 'agent', ?, 1, 5, '[]', 0, '{}', 'now', 'now')",
            (task_id, state),
        )

    def add_event(task_id: str, kind: str, payload: dict, to_state: str | None = None) -> None:
        conn.execute(
            "INSERT INTO events (ts, run_id, task_id, kind, to_state, payload) "
            "VALUES ('now', 'run', ?, ?, ?, ?)",
            (task_id, kind, to_state, json.dumps(payload)),
        )

    # T: the parent task itself, an audit unit with clean evidence, so
    # subtask-level fixtures below drive fail_closed_ok on their own.
    add_task("T", "DONE")
    add_event("T", "scientific_check", {"status": "PASS", "validator": "parent"})

    # S1: an audit unit (has task_decomposed) that is DONE, but every recorded
    # check is FAIL/ERROR -- must NOT reconcile.
    add_task("T-S1", "DONE")
    add_event("T-S1", "task_decomposed", {"subtasks": [{"id": "T-S1-S1"}]})
    add_event("T-S1", "scientific_check", {"status": "ERROR", "validator": "v1"})
    add_event("T-S1", "scientific_check", {"status": "FAIL", "validator": "v2"})
    add_event("T-S1", "review", {"verdict": "fail"})

    # S2: a leaf (no task_decomposed) that is DONE with zero scientific_check
    # events -- parent evidence never covers this child's missing checks.
    add_task("T-S2", "DONE")
    add_event("T-S2", "transition", {})

    # S3: an audit unit, DONE, all evidence PASS -- must reconcile.
    add_task("T-S3", "DONE")
    add_event("T-S3", "task_decomposed", {"subtasks": [{"id": "T-S3-S1"}]})
    add_event("T-S3", "scientific_check", {"status": "PASS", "validator": "v1"})
    add_event("T-S3", "scientific_check", {"status": "PASS", "validator": "v2"})

    # S4: a leaf that is not DONE (still BLOCKED) -- must NOT reconcile even
    # though it carries clean evidence: both state and evidence matter.
    add_task("T-S4", "BLOCKED")
    add_event("T-S4", "scientific_check", {"status": "PASS", "validator": "v1"})

    # S6: an audit unit, DONE, zero scientific_check events -- must NOT
    # reconcile (an audit unit's silence is never accepted as a pass).
    add_task("T-S6", "DONE")
    add_event("T-S6", "task_decomposed", {"subtasks": [{"id": "T-S6-S1"}]})

    # S7: the S1 shape -- repaired (unblocked), THEN closed DONE with zero
    # scientific_check events recorded since that repair. Must still NOT
    # reconcile: being "repaired" is not evidence, and must never bypass the
    # zero-evidence check no matter how the task was eventually closed.
    add_task("T-S7", "DONE")
    add_event("T-S7", "task_decomposed", {"subtasks": [{"id": "T-S7-S1"}]})
    add_event("T-S7", "scientific_check", {"status": "FAIL", "validator": "stale"})
    add_event("T-S7", "transition", {"unblocked": True, "resumed_at": "REVIEW"})
    add_event("T-S7", "transition", {}, to_state="DONE")

    conn.commit()
    conn.close()


class GateStatusReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "fake_state.db"
        make_fake_orchestrator_db(self.db_path)

    def test_failing_evidence_is_not_reconciled(self) -> None:
        report = MODULE.build_report(state_db=self.db_path, task_id="T", subtask_count=4)
        by_id = {row["subtask_id"]: row for row in report["per_subtask"]}
        self.assertEqual(by_id["T-S1"]["reconciled_verdict"], MODULE.NOT_RECONCILED_FAILING_EVIDENCE)
        self.assertEqual(by_id["T-S1"]["evidence_counts"]["fail"], 1)
        self.assertEqual(by_id["T-S1"]["evidence_counts"]["error"], 1)

    def test_zero_evidence_is_not_silently_passed_for_an_audit_unit(self) -> None:
        report = MODULE.build_report(state_db=self.db_path, task_id="T", subtask_count=6)
        by_id = {row["subtask_id"]: row for row in report["per_subtask"]}
        self.assertEqual(by_id["T-S6"]["reconciled_verdict"], MODULE.NOT_RECONCILED_NO_EVIDENCE)
        self.assertEqual(by_id["T-S6"]["evidence_counts"]["total"], 0)
        self.assertTrue(by_id["T-S6"]["audit_unit"])

    def test_being_repaired_and_later_closed_done_does_not_excuse_zero_evidence(self) -> None:
        """The exact E-F_001-S1 shape: repaired, then closed DONE, no new evidence since."""
        report = MODULE.build_report(state_db=self.db_path, task_id="T", subtask_count=7)
        by_id = {row["subtask_id"]: row for row in report["per_subtask"]}
        self.assertEqual(by_id["T-S7"]["reconciled_verdict"], MODULE.NOT_RECONCILED_NO_EVIDENCE)
        self.assertEqual(by_id["T-S7"]["evidence_counts"]["total"], 0)
        self.assertTrue(by_id["T-S7"]["repair_cycle_completed"])

    def test_leaf_with_zero_evidence_is_not_reconciled(self) -> None:
        report = MODULE.build_report(state_db=self.db_path, task_id="T", subtask_count=4)
        by_id = {row["subtask_id"]: row for row in report["per_subtask"]}
        self.assertEqual(
            by_id["T-S2"]["reconciled_verdict"], MODULE.NOT_RECONCILED_NO_EVIDENCE
        )
        self.assertFalse(by_id["T-S2"]["audit_unit"])
        self.assertEqual(by_id["T-S2"]["evidence_counts"]["total"], 0)

    def test_clean_passing_evidence_reconciles(self) -> None:
        report = MODULE.build_report(state_db=self.db_path, task_id="T", subtask_count=4)
        by_id = {row["subtask_id"]: row for row in report["per_subtask"]}
        self.assertEqual(by_id["T-S3"]["reconciled_verdict"], MODULE.RECONCILED_PASS)

    def test_non_done_state_is_not_reconciled_even_with_clean_evidence(self) -> None:
        report = MODULE.build_report(state_db=self.db_path, task_id="T", subtask_count=4)
        by_id = {row["subtask_id"]: row for row in report["per_subtask"]}
        self.assertEqual(by_id["T-S4"]["reconciled_verdict"], MODULE.NOT_RECONCILED_STATE_MISMATCH)

    def test_missing_task_id_is_reported_not_skipped(self) -> None:
        report = MODULE.build_report(state_db=self.db_path, task_id="T", subtask_count=5)
        by_id = {row["subtask_id"]: row for row in report["per_subtask"]}
        self.assertEqual(by_id["T-S5"]["reconciled_verdict"], MODULE.NOT_RECONCILED_TASK_MISSING)
        self.assertFalse(by_id["T-S5"]["orchestrator_task_found"])

    def test_fail_closed_ok_is_false_unless_every_subtask_reconciles(self) -> None:
        report = MODULE.build_report(state_db=self.db_path, task_id="T", subtask_count=4)
        self.assertFalse(report["summary"]["fail_closed_ok"])
        self.assertEqual(report["summary"]["reconciled_pass"], 1)
        self.assertEqual(report["summary"]["reconciled_covered_by_parent_audit"], 0)
        self.assertIn("T-S1", report["summary"]["not_reconciled_subtask_ids"])
        self.assertIn("T-S2", report["summary"]["not_reconciled_subtask_ids"])
        self.assertIn("T-S4", report["summary"]["not_reconciled_subtask_ids"])

    def test_fail_closed_ok_true_when_every_subtask_reconciles(self) -> None:
        clean_db = Path(self.tmp.name) / "clean_state.db"
        conn = sqlite3.connect(str(clean_db))
        conn.executescript(
            "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, type TEXT, state TEXT, "
            "attempt INTEGER, max_attempts INTEGER, dependencies TEXT, "
            "requires_human_approval INTEGER, block_reason TEXT, spec TEXT, created_at TEXT, updated_at TEXT);"
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, run_id TEXT, "
            "task_id TEXT, kind TEXT, from_state TEXT, to_state TEXT, payload TEXT);"
        )
        conn.execute(
            "INSERT INTO tasks (id, title, type, state, attempt, max_attempts, dependencies, "
            "requires_human_approval, spec, created_at, updated_at) "
            "VALUES ('C', 't', 'agent', 'BLOCKED', 1, 5, '[]', 0, '{}', 'now', 'now')"
        )
        conn.execute(
            "INSERT INTO events (ts, run_id, task_id, kind, payload) "
            "VALUES ('now', 'run', 'C', 'scientific_check', ?)",
            (json.dumps({"status": "PASS", "validator": "parent"}),),
        )
        conn.execute(
            "INSERT INTO tasks (id, title, type, state, attempt, max_attempts, dependencies, "
            "requires_human_approval, spec, created_at, updated_at) "
            "VALUES ('C-S1', 't', 'agent', 'DONE', 1, 5, '[]', 0, '{}', 'now', 'now')"
        )
        conn.execute(
            "INSERT INTO events (ts, run_id, task_id, kind, payload) "
            "VALUES ('now', 'run', 'C-S1', 'scientific_check', ?)",
            (json.dumps({"status": "PASS", "validator": "v1"}),),
        )
        conn.commit()
        conn.close()

        single = MODULE.build_report(state_db=clean_db, task_id="C", subtask_count=1)
        self.assertEqual(single["subtasks_checked"], 1)
        self.assertTrue(single["summary"]["fail_closed_ok"])

    def test_vacuous_zero_subtask_report_is_not_treated_as_a_pass(self) -> None:
        empty = MODULE.build_report(state_db=self.db_path, task_id="T", subtask_count=0)
        self.assertEqual(empty["subtasks_checked"], 0)
        self.assertFalse(empty["summary"]["fail_closed_ok"])

    def test_readonly_open_never_writes_to_the_db_file(self) -> None:
        before = self.db_path.stat().st_mtime_ns
        MODULE.build_report(state_db=self.db_path, task_id="T", subtask_count=4)
        after = self.db_path.stat().st_mtime_ns
        self.assertEqual(before, after)
        self.assertFalse((self.db_path.parent / (self.db_path.name + "-wal")).exists())

    def test_missing_db_file_raises_rather_than_silently_passing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            MODULE.build_report(state_db=self.db_path.parent / "does_not_exist.db", task_id="T", subtask_count=1)

    def test_main_writes_report_and_returns_fail_closed_exit_code(self) -> None:
        output = Path(self.tmp.name) / "status_reconciliation.json"
        exit_code = MODULE.main(
            [
                "--state-db",
                str(self.db_path),
                "--task-id",
                "T",
                "--subtask-count",
                "4",
                "--output",
                str(output),
            ]
        )
        self.assertEqual(exit_code, 1)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], MODULE.SCHEMA)
        self.assertFalse(payload["summary"]["fail_closed_ok"])


@unittest.skipUnless(REAL_STATE_DB.exists(), "orchestrator state.db not available in this environment")
class RealOrchestratorReconciliationTests(unittest.TestCase):
    """Guards against a future version of this tool quietly learning to lie.

    These assertions are invariants of fail-closed reconciliation, not pinned
    snapshots of the orchestrator's current data: whatever E-F_001 looks like
    later, a subtask with zero scientific_check evidence must never reconcile
    to PASS, and a subtask with recorded FAIL/ERROR evidence must never
    reconcile to PASS either.
    """

    def test_no_evidence_and_no_failing_evidence_subtasks_never_reconcile_pass(self) -> None:
        report = MODULE.build_report(state_db=REAL_STATE_DB, task_id="E-F_001", subtask_count=56)
        self.assertEqual(report["subtasks_checked"], 56)
        for row in report["per_subtask"]:
            if not row["audit_unit"]:
                # Parent coverage is never substituted for task-local evidence.
                self.assertIn(
                    row["reconciled_verdict"],
                    (MODULE.RECONCILED_PASS,
                     MODULE.NOT_RECONCILED_NO_EVIDENCE,
                     MODULE.NOT_RECONCILED_FAILING_EVIDENCE,
                     MODULE.NOT_RECONCILED_STATE_MISMATCH,
                     MODULE.NOT_RECONCILED_TASK_MISSING),
                )
                if row["evidence_counts"]["total"] == 0:
                    self.assertNotEqual(row["reconciled_verdict"], MODULE.RECONCILED_PASS)
                continue
            if row["evidence_counts"]["total"] == 0:
                # Being repaired-then-closed-DONE must never excuse zero
                # evidence -- that is exactly the E-F_001-S1 failure mode.
                self.assertNotEqual(row["reconciled_verdict"], MODULE.RECONCILED_PASS)
            if row["evidence_counts"]["fail"] or row["evidence_counts"]["error"]:
                self.assertNotEqual(row["reconciled_verdict"], MODULE.RECONCILED_PASS)

    def test_leaf_subtasks_are_never_silently_treated_as_evidence_bearing(self) -> None:
        """Confirms the real E-F_001-S2..S56 shape: leaves, not evidence gaps."""
        report = MODULE.build_report(state_db=REAL_STATE_DB, task_id="E-F_001", subtask_count=56)
        by_id = {row["subtask_id"]: row for row in report["per_subtask"]}
        leaves = [f"E-F_001-S{n}" for n in range(2, 57)]
        for subtask_id in leaves:
            self.assertFalse(by_id[subtask_id]["audit_unit"], subtask_id)

    def test_does_not_write_to_the_real_state_db(self) -> None:
        before = REAL_STATE_DB.stat().st_mtime_ns
        MODULE.build_report(state_db=REAL_STATE_DB, task_id="E-F_001", subtask_count=56)
        after = REAL_STATE_DB.stat().st_mtime_ns
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
