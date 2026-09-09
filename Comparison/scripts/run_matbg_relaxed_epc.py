#!/usr/bin/env python3
"""E-F_001-S51: relaxed-MATBG EPC production stage (Fase 11).

S50 (``preregister_matbg_relaxed_epc.py``) froze the protocol for both
branches (``gamma``, ``qneq0``) and left ``ready_to_execute = False``
unconditionally: ``no_relaxed_geometry`` is appended to every branch's
``blocking`` list without exception, because S47
(``decide_matbg_relaxation_provider``) found **NO-GO-10** -- no relaxation
provider in this environment is defensible for the production ``(31, 30)``
cell, so no relaxed positions file exists anywhere in this repository.

There is therefore nothing to build a batch, a direction or a delta sweep
against, and this module never attempts to. Its only job is what the roadmap
actually asks a step at this position in the DAG to do while every upstream
gate is closed: read S50's gate live (never re-derive it), refuse execution
through the same ``require_frozen_protocol`` every other stage in this
pipeline is refused through, and persist that refusal as a granular,
idempotent, per-branch artifact -- ``raw_responses``/``pao_covariant_response``/``g`` each
``status: "not_attempted"`` with a reason, and every backend field
(``electronic_derivative_backend``/``basis_response_backend``/
``phonon_backend``/``epc_backend_class``) recorded rather than omitted -- so a
future execution, once S47 reopens, has exactly one place
(``produce_branch``'s ``except PreregistrationError`` branch) to replace with
a real computation instead of a second script to write from scratch.

Producer: :func:`run_relaxed_epc`, :func:`produce_branch`. Depends on:
``docs/epc_s47_matbg_relaxation_decision.md`` (S47, the GO-10 verdict this
module cannot change) and ``preregister_matbg_relaxed_epc.py`` (S50, the
frozen protocol and gate this module reads).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import preregister_matbg_relaxed_epc as epc_prereg  # noqa: E402
from artifact_signature import file_sha256, input_signature_sha256  # noqa: E402

prereg = epc_prereg.prereg
PreregistrationError = prereg.PreregistrationError

SCHEMA = "epc_relaxed_matbg_production_v1"
TICKET = "E-F_001-S51"
RESULT_STATUS = "candidate_relaxed_tbg"
BRANCHES: tuple[str, ...] = ("gamma", "qneq0")

EPC_ROOT = epc_prereg.DEFAULT_RESULT_ROOT
EPC_STATUS_PATH = EPC_ROOT / "artifact_status.json"
EPC_SUMMARY_PATH = EPC_ROOT / "relaxed_epc_summary.json"

FORBIDDEN_PUBLICATION_LABELS: tuple[str, ...] = (
    "quantitative_relaxed_matbg_prediction",
    "g_mn_nu",
    "phonon_eigenmode",
)

_NOT_ATTEMPTED = "not_attempted"


class RelaxedEpcProductionError(RuntimeError):
    """A relaxed-MATBG EPC production step could not be authorised or run."""


def read_json(path: Path) -> dict[str, Any]:
    payload = prereg._read_json(Path(path))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prereg._json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def branch_signature(*, protocol_sha256: str | None, branch: str) -> str:
    return input_signature_sha256(
        {
            "schema": "epc_relaxed_matbg_production_signature_v1",
            "preregistration_sha256": protocol_sha256,
            "branch": branch,
            "code_sha256": file_sha256(Path(__file__)),
        }
    )


def produce_branch(branch: str, *, protocol: dict[str, Any]) -> dict[str, Any]:
    """Attempt E-F_001-S51 for one branch; refuses rather than fabricating.

    Idempotent: a cached ``blocked`` entry whose signature (protocol hash,
    branch, this file's own hash) already matches is reused without
    re-touching the gate, so repeated or restarted runs do not recompute
    anything -- consistent with every other producer in this pipeline.
    """
    if branch not in BRANCHES:
        raise RelaxedEpcProductionError(f"unknown branch {branch!r}")

    protocol_sha256 = protocol.get("frozen_content_sha256")
    signature = branch_signature(protocol_sha256=protocol_sha256, branch=branch)
    status = read_json(EPC_STATUS_PATH)
    status["preregistration_sha256"] = protocol_sha256
    status.setdefault("branches", {})
    existing = status["branches"].get(branch, {})
    if existing.get("state") == "blocked" and existing.get("signature_sha256") == signature:
        return {**existing, "reused": True}

    try:
        epc_prereg.require_frozen_protocol(branch=branch)
        # Unreachable while S50 keeps ready_to_execute=False for both
        # branches (today, unconditionally). Left as live code, not a
        # hardcoded skip, so a future GO-10 reopen exercises this gate for
        # real instead of falling through a branch that was never written.
        raise RelaxedEpcProductionError(
            f"branch {branch!r} reports ready_to_execute=True but no raw-response/"
            "pao_covariant_response/g computation exists yet in this module to run against a relaxed "
            "geometry -- implement it here before removing this guard"
        )
    except PreregistrationError as exc:
        branch_protocol = protocol["branches"][branch]
        blocking = list(branch_protocol.get("blocking") or [])
        claim = branch_protocol["claim_ladder"][-1]["claim"]
        result = {
            "state": "blocked",
            "signature_sha256": signature,
            "result_status": RESULT_STATUS,
            "branch": branch,
            "preregistration_sha256": protocol_sha256,
            "gate_error": str(exc),
            "blocking": blocking,
            "raw_responses": {
                "status": _NOT_ATTEMPTED,
                "reason": "no relaxed geometry exists (S47: NO-GO-10, decide_matbg_relaxation_"
                "provider); there is no positions file to build a batch, a direction or a delta "
                "sweep against",
            },
            "pao_covariant_response": {"status": _NOT_ATTEMPTED, "reason": "blocked by raw_responses"},
            "g": {"status": _NOT_ATTEMPTED, "reason": "blocked by pao_covariant_response"},
            "electronic_derivative_backend": _NOT_ATTEMPTED,
            "basis_response_backend": _NOT_ATTEMPTED,
            "phonon_backend": _NOT_ATTEMPTED,
            "epc_backend_class": _NOT_ATTEMPTED,
            "artifact_kind": "not_produced",
            "claim": claim,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    status["branches"][branch] = result
    write_json(EPC_STATUS_PATH, status)
    return result


def run_relaxed_epc(branches: tuple[str, ...] = BRANCHES) -> dict[str, Any]:
    """E-F_001-S51 entry point: attempt every requested branch, cite S50, never fabricate."""
    protocol = epc_prereg.load_protocol()
    results = {branch: produce_branch(branch, protocol=protocol) for branch in branches}
    all_blocked = all(row["state"] == "blocked" for row in results.values())

    summary = {
        "schema": SCHEMA,
        "ticket": TICKET,
        "gate": "S50 preregistration (preregister_matbg_relaxed_epc); every branch stays blocked "
        "until S47/GO-10 reopens",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_status": RESULT_STATUS,
        "preregistration_sha256": protocol.get("frozen_content_sha256"),
        "branches": results,
        "forbidden_publication_labels": list(FORBIDDEN_PUBLICATION_LABELS),
        "overall_claim": "no_execution_authorised" if all_blocked else "partial_execution",
    }
    write_json(EPC_SUMMARY_PATH, summary)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--branch", choices=(*BRANCHES, "all"), default="all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    branches = BRANCHES if args.branch == "all" else (args.branch,)
    summary = run_relaxed_epc(branches)

    print(f"[{TICKET}] relaxed-MATBG EPC production: {summary['overall_claim']}  ->  {EPC_SUMMARY_PATH}")
    for name, row in summary["branches"].items():
        print(f"[{TICKET}]   branch {name:<7} state={row['state']}  claim={row['claim']}")
        for reason in row.get("blocking", []):
            print(f"[{TICKET}]     blocked by: {reason}")
    return 0 if summary["overall_claim"] == "no_execution_authorised" else 1


if __name__ == "__main__":
    raise SystemExit(main())
