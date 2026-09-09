#!/usr/bin/env python3
"""E-F_001-S54: produce and interpret rigid-model integrated EPC observables (Fase 12).

Roadmap Fase 12 (``|g|^2, gamma_qnu, lambda_qnu, lambda, alpha2F(omega)``) is
``DEFERRED``: "Solo despues de GO-10 o, para un resultado explicitamente
rigid-model, despues de GO-9." S53 (``preregister_integrated_observables_
convergence.py``) already froze the numerical protocol (Gamma-centered
Monkhorst-Pack k/q mesh, modal completeness, smearing reused from
``epc_occupations.py``, broadening/DOS normalization, the plateau/uncertainty
evidence rule) and validated all of it against synthetic test functions with
a known limit -- never against a fabricated ``g``. It left
``ready_to_execute=False`` for one reason only: ``g_set`` (the protocol's
sole physical dependency, per ``shared/artifact_signature.py``'s own DAG) is
empty, because both rigid branches read live as ``NO-GO-9``:

    certify_rigid_gamma_campaign.py   (S43) -> go9_status == "NO-GO-9"
    certify_qneq0_rigid_campaign.py   (S45) -> go9_status == "NO-GO-9"

This ticket is the execution step S53 gated. It does not recompute S53's
math or S43/S45's certifications. It reads the S53 protocol live, and:

* if some day it reports ``ready_to_execute=True``, computes ``|g|^2``,
  ``gamma_qnu``, ``lambda_qnu``, ``lambda`` and ``alpha2F`` from a
  GO-9-certified ``g_set`` using the converged mesh/broadening from S53's own
  ladder -- that branch is unreachable today and intentionally left as live
  code (not a hardcoded skip) so a future GO-9 reopen exercises it for real;
* today, refuses and persists that refusal as a granular artifact (each
  observable ``not_attempted`` with a reason cascading from ``g_set``, every
  backend field recorded rather than omitted), then interprets that refusal:
  which claims the evidence licenses (none quantitative) and which stay
  forbidden regardless of gate outcome (``Tc`` explicitly, per roadmap scope).

Producer: :func:`produce`, :func:`interpret`, :func:`run`. Depends on:
``preregister_integrated_observables_convergence.py`` (S53, the frozen
protocol and gate this module reads) and the on-disk GO-9 certification
reports from ``certify_rigid_gamma_campaign.py`` (S43) and
``certify_qneq0_rigid_campaign.py`` (S45).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import certify_qneq0_rigid_campaign as go9_qneq0  # noqa: E402
import certify_rigid_gamma_campaign as go9_gamma  # noqa: E402
import preregister_integrated_observables_convergence as s53  # noqa: E402

prereg = s53.prereg
PreregistrationError = prereg.PreregistrationError

SCHEMA = "epc_rigid_integrated_observables_v1"
TICKET = "E-F_001-S54"
MEMO = "docs/epc_s54_rigid_integrated_observables.md"
RESULT_STATUS = "candidate_rigid_tbg"
GATE = (
    "S53 convergence protocol (preregister_integrated_observables_convergence); execution "
    "additionally requires a GO-9-certified g_set on either rigid branch (S43 or S45)"
)

DEFAULT_S43_REPORT_PATH = go9_gamma.DEFAULT_OUTPUT_DIR / go9_gamma.REPORT_NAME
DEFAULT_S45_REPORT_PATH = go9_qneq0.DEFAULT_OUTPUT_DIR / go9_qneq0.REPORT_NAME
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/integrated_observables/matbg"
REPORT_NAME = "rigid_integrated_observables_result.json"

# roadmap Fase 12: "|g|^2, gamma_qnu, lambda_qnu, lambda, alpha2F(omega)"
OBSERVABLES: tuple[str, ...] = ("g_squared", "gamma_qnu", "lambda_qnu", "lambda_total", "alpha2f")

FORBIDDEN_PUBLICATION_LABELS: tuple[str, ...] = (
    "quantitative_relaxed_matbg_prediction",
    "g_mn_nu",
    "phonon_eigenmode",
    "Tc",
    "rigid_tbg_model",  # only licensed once a branch's own GO-9 certification says so
)

_NOT_ATTEMPTED = "not_attempted"


class RigidIntegratedObservablesError(RuntimeError):
    """The rigid-model integrated EPC observables could not be authorised or computed."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = prereg._read_json(Path(path))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prereg._json_safe(dict(value)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# production: attempt |g|^2, gamma_qnu, lambda_qnu, lambda, alpha2F; refuse
# rather than fabricate while g_set is empty.
# --------------------------------------------------------------------------- #


def rigid_go9_status(
    s43_report: Path = DEFAULT_S43_REPORT_PATH, s45_report: Path = DEFAULT_S45_REPORT_PATH
) -> dict[str, Any]:
    """Live GO-9 read for both rigid branches, from the on-disk S43/S45 certifications."""
    s43 = _read_json(s43_report)
    s45 = _read_json(s45_report)
    return {
        "gamma": {"go9_status": s43.get("go9_status"), "final_publication_label": s43.get("final_publication_label")},
        "qneq0": {"go9_status": s45.get("go9_status"), "final_publication_label": s45.get("final_publication_label")},
    }


def produce(
    s43_report: Path = DEFAULT_S43_REPORT_PATH, s45_report: Path = DEFAULT_S45_REPORT_PATH
) -> dict[str, Any]:
    """E-F_001-S54 production step: attempt every rigid observable, never fabricate."""
    protocol = s53.build_protocol(gamma_campaign_certification=s43_report, qneq0_campaign_certification=s45_report)
    go9 = rigid_go9_status(s43_report, s45_report)

    try:
        if not protocol.get("ready_to_execute"):
            raise PreregistrationError(f"the S53 protocol is not ready to execute: {protocol.get('blocking')}")
        # Unreachable while both rigid branches stay NO-GO-9 (today,
        # unconditionally). Left as live code, not a hardcoded skip, so a
        # future GO-9 reopen exercises this gate for real instead of falling
        # through a branch that was never written.
        raise RigidIntegratedObservablesError(
            "protocol reports ready_to_execute=True but no g_set -> observable computation exists "
            "yet in this module to run against a GO-9-certified rigid campaign -- implement it here "
            "(converged mesh/broadening from the S53 ladder) before removing this guard"
        )
    except PreregistrationError as exc:
        result = {
            "state": "blocked",
            "result_status": RESULT_STATUS,
            "gate_error": str(exc),
            "blocking": list(protocol.get("blocking", [])),
            "g_set": {"status": _NOT_ATTEMPTED, "reason": str(exc)},
            "observables": {name: {"status": _NOT_ATTEMPTED, "reason": "blocked by g_set"} for name in OBSERVABLES},
            "rigid_go9_status": go9,
            "electronic_derivative_backend": _NOT_ATTEMPTED,
            "basis_response_backend": _NOT_ATTEMPTED,
            "phonon_backend": _NOT_ATTEMPTED,
            "epc_backend_class": _NOT_ATTEMPTED,
            "artifact_kind": "not_produced",
            "preregistration_sha256": protocol.get("frozen_content_sha256"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    return result


# --------------------------------------------------------------------------- #
# interpretation: which claims the refusal above licenses, and which stay
# forbidden regardless of gate outcome.
# --------------------------------------------------------------------------- #


def interpret(production: Mapping[str, Any]) -> dict[str, Any]:
    go9 = production["rigid_go9_status"]
    any_go9 = any(row["go9_status"] == "GO-9" for row in go9.values())
    licensed = [
        {
            "claim": "rigid_integrated_observables_protocol_correctly_applied",
            "licensed": production["state"] == "blocked" and bool(production.get("blocking")),
            "evidence": "the S53 protocol was read live and its own blocking list is reproduced verbatim",
        },
        {
            "claim": "rigid_integrated_observables_no_execution_authorised",
            "licensed": not any_go9,
            "evidence": go9,
        },
    ]
    forbidden = [
        {
            "claim": label,
            "reason": (
                "Tc is explicitly out of scope per the roadmap and S53's own protocol scope"
                if label == "Tc"
                else "no g has been computed or certified GO-9 on either rigid branch; there is no "
                "coupling constant to square, weight by phonon occupation or sum over a k/q mesh"
            ),
        }
        for label in FORBIDDEN_PUBLICATION_LABELS
    ]
    all_hold = all(row["licensed"] for row in licensed)
    return {
        "licensed": licensed,
        "forbidden": forbidden,
        "rule": "an observable claim is licensed only if g_set itself is licensed; this step never "
        "widens GO-9 or substitutes a selected-mode/synthetic result for the modal sum an "
        "integrated observable needs",
        "all_licensed_claims_hold": all_hold,
        "verdict": "NO_QUANTITATIVE_CLAIM_LICENSED" if all_hold else "VALIDATION_FAILED",
    }


def run(
    s43_report: Path = DEFAULT_S43_REPORT_PATH,
    s45_report: Path = DEFAULT_S45_REPORT_PATH,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    production = produce(s43_report, s45_report)
    interpretation = interpret(production)
    result = {
        "schema": SCHEMA,
        "ticket": TICKET,
        "memo": MEMO,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_status": RESULT_STATUS,
        "production": production,
        "interpretation": interpretation,
        "forbidden_publication_labels": list(FORBIDDEN_PUBLICATION_LABELS),
        "overall_claim": "no_execution_authorised" if production["state"] == "blocked" else "partial_execution",
    }
    _write_json(Path(output_dir) / REPORT_NAME, result)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s43-report", type=Path, default=DEFAULT_S43_REPORT_PATH)
    parser.add_argument("--s45-report", type=Path, default=DEFAULT_S45_REPORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run(args.s43_report, args.s45_report, output_dir=args.output_dir)

    output = args.output_dir / REPORT_NAME
    print(f"[{TICKET}] rigid integrated observables: {result['overall_claim']}  ->  {output}")
    print(f"[{TICKET}]   interpretation verdict: {result['interpretation']['verdict']}")
    for branch, row in result["production"]["rigid_go9_status"].items():
        print(f"[{TICKET}]   rigid {branch:<7} go9={row['go9_status']}")
    for reason in result["production"].get("blocking", []):
        print(f"[{TICKET}]   blocked by: {reason}")
    for row in result["interpretation"]["forbidden"]:
        print(f"[{TICKET}]   forbidden: {row['claim']}")
    return 0 if result["interpretation"]["verdict"] == "NO_QUANTITATIVE_CLAIM_LICENSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
