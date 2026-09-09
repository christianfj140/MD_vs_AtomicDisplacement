#!/usr/bin/env python3
"""Live interface audit: does E-F_001's recorded ``scientific_validation_plan``
actually build against the orchestrator's REAL validator classes right now?

Reuses the orchestrator's own introspection point (``inspect.signature`` on
the real classes in ``research_orchestrator.validators``, the module that
actually builds and runs these specs -- see ``validators.build`` and
``validators.REGISTRY``) rather than a hand-maintained list of "what a
validator probably accepts". A planner-imagined kwarg (``axes``, ``metric``,
``metrics``, ``parameters``, ``observable``, ``required_values``, ``kinds``)
is invisible until compared against the live constructor; that comparison is
this script's whole job.

Read-only against research-orchestrator's state.db: no event is written, no
row is mutated.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ORCHESTRATOR_ROOT = Path("/home/christian/repositorios/research-orchestrator")
sys.path.insert(0, str(ORCHESTRATOR_ROOT / "src"))

from research_orchestrator import validators as V  # noqa: E402

SCHEMA = "epc_interface_audit_v1"
DEFAULT_STATE_DB = ORCHESTRATOR_ROOT / "state.db"
DEFAULT_TASK_ID = "E-F_001"
DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc_repair/interface_audit.json"

VALIDATOR_CLASSES = {
    "command": V.CommandValidator,
    "numerical_sanity": V.NumericalSanityValidator,
    "physical_tolerance": V.PhysicalToleranceValidator,
    "convergence": V.ConvergenceValidator,
    "symmetry": V.SymmetryValidator,
    "literature": V.LiteratureValidator,
}


def real_signature(validator_type: str) -> set[str] | None:
    cls = VALIDATOR_CLASSES.get(validator_type)
    if cls is None:
        return None
    sig = inspect.signature(cls.__init__)
    return {p for p in sig.parameters if p not in ("self",)}


def open_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def latest_plan_checks(conn: sqlite3.Connection, task_id: str) -> list[dict[str, Any]]:
    row = conn.execute(
        "SELECT payload FROM events WHERE task_id = ? AND kind = 'scientific_validation_plan' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None:
        return []
    payload = json.loads(row["payload"])
    return payload.get("checks", [])


def audit_check(index: int, check: dict[str, Any]) -> dict[str, Any]:
    validator = check.get("validator") or {}
    validator_type = validator.get("type")
    given_fields = {k for k in validator.keys() if k != "type"}
    accepted = real_signature(validator_type) if validator_type else None

    if not validator:
        return {
            "index": index, "check": check.get("check", "")[:120], "validator_type": None,
            "expected_interface": [], "actual_interface": [], "compatible": None,
            "action": "not automated (no validator spec); answered by an auditor, not a gate",
        }
    if validator_type not in VALIDATOR_CLASSES:
        return {
            "index": index, "check": check.get("check", "")[:120], "validator_type": validator_type,
            "expected_interface": sorted(given_fields), "actual_interface": [],
            "compatible": False,
            "action": f"unknown validator type {validator_type!r}; known: {sorted(VALIDATOR_CLASSES)}",
        }

    bad_fields = given_fields - accepted
    compatible = not bad_fields
    if compatible:
        action = "spec is fully compatible"
    else:
        action = "drop field(s): " + ", ".join(sorted(bad_fields))

    return {
        "index": index,
        "check": check.get("check", "")[:120],
        "validator_type": validator_type,
        "expected_interface": sorted(given_fields),
        "actual_interface": sorted(accepted),
        "extra_fields_not_accepted": sorted(bad_fields),
        "compatible": compatible,
        "action": action,
    }


def _find_targets(argv: list[str]) -> list[str]:
    """Every argv element that names a script or test file this repo should
    contain: a ``.py`` path (the validated script itself, or a pytest test
    target), skipping the interpreter (argv[0]) and ``-m``/``-q`` flags."""
    return [a for a in argv[1:] if a.endswith(".py") and not a.startswith("-")]


def _spot_check_argv(argv: list[str]) -> list[dict[str, Any]]:
    """For a command-type check's argv, confirm every referenced .py target
    (the validated script itself, or pytest test files) exists in the repo,
    and, for a script under Comparison/scripts that exposes --help, that
    every flag the spec passes is one the script actually accepts."""
    results = []
    for target in _find_targets(argv):
        target_path = (REPO_ROOT / target).resolve() if not Path(target).is_absolute() else Path(target)
        exists = target_path.is_file()
        entry: dict[str, Any] = {"script": target, "exists": exists}
        if not exists:
            results.append(entry)
            continue
        try:
            target_path.relative_to(REPO_ROOT / "Comparison/scripts")
        except ValueError:
            results.append(entry)
            continue  # a test file, not a CLI script with --help to check

        given_flags = sorted({a for a in argv[1:] if a.startswith("--")})
        # Use the SAME interpreter the recorded validator spec itself names
        # (argv[0], typically the repo's .venv) -- a bare system python3
        # lacks this repo's dependencies (numpy, sisl, ...) and a
        # ModuleNotFoundError before argparse runs would misreport every
        # flag as unknown.
        python_bin = argv[0] if argv and Path(argv[0]).is_file() else (shutil.which("python3") or "python3")
        try:
            proc = subprocess.run(
                [python_bin, str(target_path), "--help"],
                capture_output=True, text=True, timeout=30, cwd=REPO_ROOT,
            )
            help_text = proc.stdout + proc.stderr
            entry["help_ran"] = proc.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            entry["help_ran"] = False
            entry["error"] = str(exc)
            results.append(entry)
            continue
        entry["flags_checked"] = given_flags
        entry["unknown_flags"] = [f for f in given_flags if f not in help_text]
        results.append(entry)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-script-spotcheck", action="store_true",
                         help="skip running --help on referenced scripts (faster, less thorough)")
    args = parser.parse_args()

    conn = open_readonly(args.state_db)
    try:
        checks = latest_plan_checks(conn, args.task_id)
    finally:
        conn.close()

    table = [audit_check(i, check) for i, check in enumerate(checks)]

    script_spotchecks = []
    if not args.skip_script_spotcheck:
        checked_scripts: set[str] = set()
        for check in checks:
            validator = check.get("validator") or {}
            if validator.get("type") != "command":
                continue
            argv = validator.get("argv") or []
            for result in _spot_check_argv(argv):
                if result["script"] not in checked_scripts:
                    checked_scripts.add(result["script"])
                    script_spotchecks.append(result)

    n_incompatible = sum(1 for r in table if r["compatible"] is False)
    n_not_automated = sum(1 for r in table if r["compatible"] is None)
    n_compatible = sum(1 for r in table if r["compatible"] is True)
    n_missing_targets = sum(1 for s in script_spotchecks if not s["exists"])
    n_unknown_flags = sum(1 for s in script_spotchecks if s.get("unknown_flags"))

    report = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task_id": args.task_id,
        "state_db_path": str(args.state_db),
        "validator_registry_source": "research_orchestrator.validators.REGISTRY (live introspection)",
        "known_validator_types": sorted(VALIDATOR_CLASSES),
        "n_checks_total": len(table),
        "n_compatible": n_compatible,
        "n_incompatible": n_incompatible,
        "n_not_automated": n_not_automated,
        "n_referenced_targets_missing": n_missing_targets,
        "n_scripts_with_unknown_flags": n_unknown_flags,
        "table": table,
        "script_spotchecks": script_spotchecks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"interface_audit: {n_compatible} compatible, {n_incompatible} incompatible, "
          f"{n_not_automated} not-automated (of {len(table)} checks)")
    print(f"{'idx':>3} {'type':<18} {'compatible':<11} action")
    for row in table:
        print(f"{row['index']:>3} {str(row['validator_type']):<18} {str(row['compatible']):<11} {row['action'][:100]}")
    if script_spotchecks:
        print(f"\nscript/test spot-checks ({n_missing_targets} missing, "
              f"{n_unknown_flags} with unknown flags):")
        for s in script_spotchecks:
            if not s["exists"]:
                print(f"  {s['script']}: MISSING")
                continue
            flag_note = f"unknown flags: {s['unknown_flags']}" if s.get("unknown_flags") else "flags OK"
            print(f"  {s['script']}: exists=True {flag_note}")
    print(f"-> {args.output}")
    return 0 if (n_incompatible == 0 and n_missing_targets == 0 and n_unknown_flags == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
