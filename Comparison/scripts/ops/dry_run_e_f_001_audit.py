#!/usr/bin/env python3
"""Dry-run aggregate audit for E-F_001: no SIESTA, no C19/C20, no orchestrator
run, no verdict changes. Aggregates the outputs of the scope linter (B), the
live interface audit (C), and the canonical-evidence resolver (D/I) into one
report, cross-referenced against the scope contract (A).

This script does not compute anything new: it reads three already-produced
JSON artifacts and the scope contract, and reports what they say honestly --
including when section 3 (in-scope ACTIVE_BLOCKERS) is not empty.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "Comparison/results/epc_repair"

SCOPE_CONTRACT = RESULTS / "E-F_001_scope_contract_v3.json"
CLAIM_LINT = RESULTS / "claim_scope_lint.json"
INTERFACE_AUDIT = RESULTS / "interface_audit.json"
CANONICAL_EVIDENCE = RESULTS / "canonical_evidence_resolution.json"
S1_S56 = RESULTS / "s1_s56_reconciliation_v2.json"
OUTPUT = RESULTS / "dry_run_audit_v3.json"

LINTER_SCRIPT = REPO_ROOT / "Comparison/scripts/lint_epc_e_f_001_claim_scope.py"
INTERFACE_SCRIPT = REPO_ROOT / "Comparison/scripts/ops/audit_epc_interfaces.py"
RESOLVER_SCRIPT = REPO_ROOT / "Comparison/scripts/ops/resolve_canonical_evidence.py"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _rerun(script: Path) -> None:
    """Regenerate an input artifact by re-running its own producing script,
    so this dry-run reports the CURRENT state, not a stale prior run. Exit
    code is ignored here -- a FAIL/NO_GO is itself the signal this script
    reports, not a reason to abort aggregation."""
    subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, capture_output=True)


def main() -> int:
    # Regenerate the three inputs fresh (read-only against state.db; no
    # SIESTA, no C19/C20, no orchestrator `run`/`unblock`).
    _rerun(LINTER_SCRIPT)
    _rerun(INTERFACE_SCRIPT)
    _rerun(RESOLVER_SCRIPT)

    contract = _load(SCOPE_CONTRACT)
    lint = _load(CLAIM_LINT)
    interfaces = _load(INTERFACE_AUDIT)
    canonical = _load(CANONICAL_EVIDENCE)
    s1_s56 = _load(S1_S56)

    excluded_topics = {
        t.split(" (")[0].strip().lower()
        for t in (contract or {}).get("excluded_topics", [])
    } if contract else set()

    # -- 1. scope violations (linter) ----------------------------------------
    scope_violations = (lint or {}).get("violations", [])

    # -- 2. interface incompatibilities (live audit) -------------------------
    interface_problems = []
    if interfaces:
        interface_problems.extend(
            row for row in interfaces.get("table", []) if row.get("compatible") is False
        )
        interface_problems.extend(
            {"kind": "missing_target", **s}
            for s in interfaces.get("script_spotchecks", []) if not s.get("exists")
        )
        interface_problems.extend(
            {"kind": "unknown_flags", **s}
            for s in interfaces.get("script_spotchecks", []) if s.get("unknown_flags")
        )

    # -- 3. in-scope ACTIVE_BLOCKERS ------------------------------------------
    # Cross-reference the reconciler's ACTIVE_BLOCKER list against the scope
    # contract: an ACTIVE_BLOCKER whose OWN domain is explicitly out of scope
    # (full_KS, ml_checkpoint_selection, MATBG, qneq0, C19C20) does not count
    # here, even though the row itself is "ACTIVE_BLOCKER" classification --
    # that classification means "unresolved", not "blocks E-F_001's claim".
    # An "unclassified" row is NOT excluded: an unreviewed finding must not
    # be assumed out of scope by default, so it counts as in-scope until a
    # human or a new FINDING_RULES rule says otherwise.
    OUT_OF_SCOPE_DOMAINS = {"full_KS", "ml_checkpoint_selection", "MATBG", "qneq0", "C19C20"}
    in_scope_active_blockers = []
    if canonical:
        for row in canonical.get("reconciliation_table", []):
            if row.get("classification") != "ACTIVE_BLOCKER":
                continue
            if row.get("domain") in OUT_OF_SCOPE_DOMAINS:
                continue
            in_scope_active_blockers.append(row)
    s1_s56_gamma_blockers = (s1_s56 or {}).get("graphene_Gamma_active_blocker_steps", [])

    # -- 4. OUT_OF_SCOPE findings (informational) -----------------------------
    out_of_scope_findings = [
        row for row in (canonical or {}).get("reconciliation_table", [])
        if row.get("classification") == "OUT_OF_SCOPE"
    ]

    # -- 5. SUPERSEDED findings (informational) -------------------------------
    superseded_findings = [
        row for row in (canonical or {}).get("reconciliation_table", [])
        if row.get("classification") == "SUPERSEDED"
    ]

    report = {
        "schema": "epc_e_f_001_dry_run_audit_v3",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task_id": "E-F_001",
        "no_execution": {
            "siesta_run": False, "c19_c20_run": False,
            "orchestrator_run_or_unblock_called": False,
            "verdicts_changed": False,
        },
        "inputs": {
            "scope_contract": str(SCOPE_CONTRACT.relative_to(REPO_ROOT)) if contract else None,
            "claim_scope_lint": str(CLAIM_LINT.relative_to(REPO_ROOT)) if lint else None,
            "interface_audit": str(INTERFACE_AUDIT.relative_to(REPO_ROOT)) if interfaces else None,
            "canonical_evidence_resolution": str(CANONICAL_EVIDENCE.relative_to(REPO_ROOT)) if canonical else None,
            "s1_s56_reconciliation_v2": str(S1_S56.relative_to(REPO_ROOT)) if s1_s56 else None,
        },
        "section_1_scope_violations": {
            "verdict": (lint or {}).get("verdict"),
            "n_violations": len(scope_violations),
            "violations": scope_violations,
        },
        "section_2_interface_incompatibilities": {
            "n_problems": len(interface_problems),
            "problems": interface_problems,
        },
        "section_3_in_scope_active_blockers": {
            "n_blockers": len(in_scope_active_blockers),
            "blockers": in_scope_active_blockers,
            "s1_s56_graphene_Gamma_active_blocker_steps": s1_s56_gamma_blockers,
            "empty": len(in_scope_active_blockers) == 0 and not s1_s56_gamma_blockers,
        },
        "section_4_out_of_scope_findings": {
            "n_findings": len(out_of_scope_findings),
            "findings": out_of_scope_findings,
        },
        "section_5_superseded_findings": {
            "n_findings": len(superseded_findings),
            "findings": superseded_findings,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print(f"dry_run_e_f_001_audit -> {OUTPUT}")
    print(f"1. scope violations: {report['section_1_scope_violations']['n_violations']}")
    print(f"2. interface incompatibilities: {report['section_2_interface_incompatibilities']['n_problems']}")
    print(f"3. in-scope ACTIVE_BLOCKERS: {report['section_3_in_scope_active_blockers']['n_blockers']} "
          f"(empty={report['section_3_in_scope_active_blockers']['empty']})")
    print(f"4. OUT_OF_SCOPE findings (informational): {report['section_4_out_of_scope_findings']['n_findings']}")
    print(f"5. SUPERSEDED findings (informational): {report['section_5_superseded_findings']['n_findings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
