#!/usr/bin/env python3
"""E-F_001-S55: relaxed-MATBG integrated-observables convergence protocol (Fase 12 / GO-10).

S53 (``preregister_integrated_observables_convergence.py``) froze the k/q mesh
generator, modal-completeness rule, smearing contract, broadening/DOS
normalization and plateau-evidence rule for the *rigid* branch, gated on a
GO-9-certified ``g_set``. S54 (``produce_rigid_integrated_observables.py``)
executes that protocol and, with both rigid branches still ``NO-GO-9``,
refuses -- correctly, since ``shared/artifact_signature.py``'s own DAG makes
``integrated_observable`` depend on nothing but ``g_set``.

This ticket's Objective is explicit that the relaxed geometry's mesh, modes,
smearing, broadening, occupations and normalization must be determined "sin
heredar automaticamente la convergencia rigida" -- so a relaxed run cannot
just replay S53/S54's frozen conclusions. It reuses S53's *machinery*
(``monkhorst_pack_2d``, ``modal_completeness_report``, the smearing/broadening/
DOS contracts, ``mesh_convergence_report``) because that machinery is generic
numerical infrastructure, not a rigid-model result -- but it freezes its own
protocol, under its own id, gated on its own precondition: a relaxed ``g_set``
(S51, ``run_matbg_relaxed_epc.py``), which itself requires GO-10 (S47,
``decide_matbg_relaxation_provider.py``, read live, never assumed). Today
S47 is unconditionally ``NO-GO-10`` -- no relaxed geometry exists anywhere in
this repository -- so S51 leaves every branch ``blocked`` and this protocol's
``ready_to_execute`` stays ``False``.

The Acceptance criteria also require that any relaxed-vs-rigid difference be
compared against the *combined* uncertainty of both sides before it is
interpreted physically. This module carries that as a generic, symmetric
comparison rule (:func:`rigid_vs_relaxed_delta_report`) that fires only once
*both* sides carry a real computed value -- reading S54's rigid result live
rather than assuming it is any further along than certified.

Producer: :func:`build_protocol`, :func:`produce`, :func:`interpret`,
:func:`run`. Depends on: ``preregister_integrated_observables_convergence.py``
(S53, the reused machinery), ``run_matbg_relaxed_epc.py`` (S51, the relaxed
``g_set`` this protocol gates on), ``decide_matbg_relaxation_provider.py``
(S47, the live GO-10 verdict) and ``produce_rigid_integrated_observables.py``
(S54, the rigid side of the required comparison).
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

import decide_matbg_relaxation_provider as go10_decision  # noqa: E402
import preregister_integrated_observables_convergence as s53  # noqa: E402
import produce_rigid_integrated_observables as s54  # noqa: E402
import run_matbg_relaxed_epc as s51  # noqa: E402

prereg = s53.prereg
PreregistrationError = prereg.PreregistrationError

SCHEMA = "epc_preregistration_v1"
PROTOCOL_ID = "relaxed_integrated_observables_convergence_v1"
TICKET = "E-F_001-S55"
MEMO = "docs/epc_s55_relaxed_integrated_observables_convergence.md"
GATE = (
    "Fase-12 convergence-protocol pre-registration for the relaxed geometry; execution "
    "additionally requires a relaxed g_set on either branch (S51/run_matbg_relaxed_epc), which "
    "itself requires GO-10 (S47/decide_matbg_relaxation_provider, read live)"
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/preregistration/matbg"
PROTOCOL_NAME = "epc_relaxed_integrated_observables_convergence_protocol.json"
DEFAULT_RESULT_DIR = REPO_ROOT / "Comparison/results/epc/integrated_observables/matbg"
REPORT_NAME = "relaxed_integrated_observables_result.json"
DEFAULT_S51_SUMMARY_PATH = s51.EPC_SUMMARY_PATH
DEFAULT_S54_REPORT_PATH = s54.DEFAULT_OUTPUT_DIR / s54.REPORT_NAME

RESULT_STATUS = "candidate_relaxed_tbg"
OBSERVABLES: tuple[str, ...] = s54.OBSERVABLES
_NOT_ATTEMPTED = "not_attempted"

FORBIDDEN_PUBLICATION_LABELS: tuple[str, ...] = (
    *s54.FORBIDDEN_PUBLICATION_LABELS,
    "relaxed_vs_rigid_integrated_observable_quantitative_delta",
)


class RelaxedIntegratedObservablesError(RuntimeError):
    """The relaxed-MATBG integrated observables could not be authorised or computed."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = prereg._read_json(Path(path))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prereg._json_safe(dict(value)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Precondition: a relaxed g_set. GO-10 (S47) is read live, not assumed, and
# S51's per-branch state is read live too, so a future GO-10 reopen (and a
# future S51 that stops raising its "unreachable" guard) is picked up here
# without editing this gate.
# --------------------------------------------------------------------------- #


def relaxed_g_set_status(summary_path: Path = DEFAULT_S51_SUMMARY_PATH) -> dict[str, Any]:
    go10 = go10_decision.matbg_relaxation_provider_decision()
    summary = _read_json(summary_path)
    branches = summary.get("branches") or {}
    per_branch = {
        name: {"state": (branches.get(name) or {}).get("state"),
               "result_status": (branches.get(name) or {}).get("result_status")}
        for name in s51.BRANCHES
    }
    any_branch_has_g = any(row["state"] not in (None, "blocked") for row in per_branch.values())
    return {
        "go10_verdict": go10["verdict"],
        "go10_relaxed_geometry_produced": go10["relaxed_geometry_produced"],
        "branches": per_branch,
        "any_branch_has_g": any_branch_has_g,
    }


def preconditions(summary_path: Path = DEFAULT_S51_SUMMARY_PATH) -> list[dict[str, Any]]:
    status = relaxed_g_set_status(summary_path)
    return [
        prereg._precondition(
            "relaxed g_set (S51, gated by GO-10/S47)",
            summary_path,
            status,
            verdict="PASS" if status["any_branch_has_g"] else status["go10_verdict"],
            required=True,
        )
    ]


def physics_review(gates: list[dict[str, Any]]) -> dict[str, Any]:
    any_g = any(row["status"] == "PASS" for row in gates)
    items = [
        {
            "item": "mesh_smearing_broadening_dos_machinery",
            "decision": "approved",
            "evidence": {
                "reused_from": "preregister_integrated_observables_convergence.py (S53)",
                "functions": ["monkhorst_pack_2d", "modal_completeness_report", "broadening_kernel",
                               "gaussian_broadened_dos", "mesh_convergence_report"],
            },
            "basis": "generic numerical infrastructure validated against synthetic functions with a "
            "known limit; not a rigid-model conclusion, so reusing the code (not the rigid ladder's "
            "output values) does not smuggle in inherited rigid convergence",
        },
        {
            "item": "no_inherited_rigid_convergence",
            "decision": "approved",
            "evidence": {"rule": "any mesh-density/broadening-width ladder actually evaluated by a "
                         "future execution of this protocol must be run against the relaxed g_set's "
                         "own values; S53's frozen rigid plateau/uncertainty conclusions are never "
                         "copied into this protocol's own convergence_evidence"},
            "basis": "this ticket's Objective: mallas/modos/smearing/broadening/occupations/"
            "normalizacion determined for the relaxed geometry, not inherited from the rigid one",
        },
        {
            "item": "rigid_vs_relaxed_delta_rule",
            "decision": "approved",
            "evidence": {"function": "rigid_vs_relaxed_delta_report",
                         "rule": "a relaxed-vs-rigid difference is interpreted physically only once "
                         "it exceeds the combined (relaxed + rigid) declared uncertainty; fires only "
                         "when both sides carry a real computed value"},
            "basis": "this ticket's Acceptance criteria: differences with the rigid model must exceed "
            "combined uncertainties before physical interpretation",
        },
        {
            "item": "g_set",
            "decision": "approved" if any_g else "blocked",
            "evidence": {row["name"]: row["status"] for row in gates},
            "basis": "artifact_signature.py: integrated_observable depends only on g_set "
            "(pao_projected_mode_coupling, physical); S47 is NO-GO-10 today, so S51 leaves every relaxed "
            "branch blocked and g_set is empty -- no sweep over a real relaxed coupling constant is "
            "authorised",
        },
    ]
    return {
        "reviewer": "automated_precondition_review (evidence-bound)",
        "items": items,
        "status": "APPROVED" if all(row["decision"] == "approved" for row in items) else "BLOCKED",
        "human_sign_off": None,
    }


def build_protocol(*, summary_path: Path = DEFAULT_S51_SUMMARY_PATH) -> dict[str, Any]:
    gates = preconditions(summary_path)
    review = physics_review(gates)

    blocking = [f"{row['name']}: {row['status']}" for row in gates if row["status"] != "PASS"]

    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "ticket": TICKET,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_content_sha256": None,
        "scope": {
            "in_scope": ["k_mesh", "q_mesh", "modal_completeness", "smearing", "broadening",
                         "occupations", "spin_valley_weights", "dos_normalization_2d",
                         "rigid_vs_relaxed_delta"],
            "explicitly_out_of_scope": ["Tc"],
            "inherits_rigid_conclusions": False,
        },
        "mesh": {
            "generator": "monkhorst_pack_2d(nx, ny)  # reused from S53, generic machinery",
            "convention": "Gamma-centered, fractional [0, 1)^2, shared by k and q (same moire BZ)",
            "default_density_ladder_starting_point": list(s53.DEFAULT_MESH_DENSITY_LADDER),
            "must_be_reevaluated_against": "the relaxed g_set's own eigenspectrum/phonon data; this "
            "ladder is a starting point, not a plateau result carried over from S53",
        },
        "modal_completeness": s53.modal_completeness_report(0, n_atoms_per_cell=s53.TARGET_ATOM_COUNT),
        "smearing": s53.epc_occupations.occupations_contract(),
        "broadening": {
            "functions": list(s53.BROADENING_FUNCTIONS),
            "default_width_ladder_ev_starting_point": list(s53.DEFAULT_BROADENING_LADDER_EV),
            "distinct_from_smearing": "smearing fills electronic states (roadmap XII); broadening "
            "is the observable-level energy-conserving delta (integrated_observable node)",
        },
        "spin_valley": {"valley_degeneracy": s53.VALLEY_DEGENERACY,
                         "formula": "spin_degeneracy * valley_degeneracy"},
        "dos_normalization": s53.dos_normalization_contract(),
        "convergence_evidence_rule": {
            "method": "mesh_convergence_report(name, densities, values, tol)  # reused from S53",
            "default_relative_tol": s53.DEFAULT_PLATEAU_RELATIVE_TOL,
            "rule": "plateau requires the last relative difference on the relaxed ladder <= tol; "
            "otherwise the residual itself is the declared uncertainty -- never silently dropped, "
            "and never substituted with S53's rigid-ladder residual",
        },
        "rigid_comparison_rule": {
            "method": "rigid_vs_relaxed_delta_report",
            "rule": "a relaxed-vs-rigid difference is licensed for physical interpretation only if "
            "abs(relaxed - rigid) > (relaxed_uncertainty + rigid_uncertainty); otherwise the "
            "difference is declared indistinguishable from convergence noise",
            "reads_rigid_from": str(DEFAULT_S54_REPORT_PATH),
        },
        "preconditions": gates,
        "physics_review": review,
        "blocking": blocking,
        "ready_to_execute": not blocking and review["status"] == "APPROVED",
        "backend": {
            "requested_backend": "cpu",
            "effective_backend": "cpu",
            "backend_preflight": "not_required",
            "backend_fallback_reason": "this ticket only freezes mesh/smearing/broadening/DOS math "
            "(reused from S53) and validates it against synthetic functions (negligible cost); the "
            "GPU preflight applies to the future relaxed g_set campaign this protocol gates, not to "
            "this ticket's own math",
        },
        "result_dir": str(DEFAULT_RESULT_DIR),
    }
    protocol["frozen_content_sha256"] = prereg.freeze_hash(protocol)
    return prereg._json_safe(protocol)


def load_protocol(path: Path = DEFAULT_OUTPUT_DIR / PROTOCOL_NAME) -> dict[str, Any]:
    payload = prereg._read_json(Path(path))
    if payload is None:
        raise PreregistrationError(f"no pre-registered protocol at {path}")
    return payload


def verify(path: Path = DEFAULT_OUTPUT_DIR / PROTOCOL_NAME) -> dict[str, Any]:
    protocol = load_protocol(path)
    recomputed = prereg.freeze_hash(protocol)
    stored = protocol.get("frozen_content_sha256")
    return {
        "protocol": str(path),
        "protocol_id": protocol.get("protocol_id"),
        "frozen_content_sha256": stored,
        "recomputed_sha256": recomputed,
        "intact": stored == recomputed,
        "verified": stored == recomputed,
    }


def require_frozen_protocol(path: Path = DEFAULT_OUTPUT_DIR / PROTOCOL_NAME) -> dict[str, Any]:
    protocol = load_protocol(path)
    if prereg.freeze_hash(protocol) != protocol.get("frozen_content_sha256"):
        raise PreregistrationError(f"{path} has been edited since it was frozen")
    if not protocol.get("ready_to_execute"):
        raise PreregistrationError(
            f"the protocol is frozen but its preconditions are not met: {protocol.get('blocking')}"
        )
    return protocol


# --------------------------------------------------------------------------- #
# rigid <-> relaxed comparison: symmetric, generic, fires only when both sides
# carry a real value. Never used to interpret a difference between two
# not_attempted placeholders.
# --------------------------------------------------------------------------- #


def rigid_vs_relaxed_delta_report(
    rigid_value: float | None, rigid_uncertainty: float | None,
    relaxed_value: float | None, relaxed_uncertainty: float | None,
) -> dict[str, Any]:
    have_both = rigid_value is not None and relaxed_value is not None
    if not have_both:
        return {"status": _NOT_ATTEMPTED, "reason": "one or both sides have no computed value"}
    combined_uncertainty = float((rigid_uncertainty or 0.0) + (relaxed_uncertainty or 0.0))
    delta = float(relaxed_value - rigid_value)
    exceeds = abs(delta) > combined_uncertainty
    return {
        "status": "computed",
        "rigid_value": float(rigid_value),
        "relaxed_value": float(relaxed_value),
        "delta": delta,
        "combined_uncertainty": combined_uncertainty,
        "exceeds_combined_uncertainty": exceeds,
        "interpretable": exceeds,
        "reading": "difference is physically interpretable" if exceeds
        else "difference is indistinguishable from convergence/numerical noise at this mesh/broadening",
    }


# --------------------------------------------------------------------------- #
# production: attempt every observable + the rigid comparison; never
# fabricate while the relaxed g_set is empty.
# --------------------------------------------------------------------------- #


def produce(
    summary_path: Path = DEFAULT_S51_SUMMARY_PATH, s54_report: Path = DEFAULT_S54_REPORT_PATH
) -> dict[str, Any]:
    protocol = build_protocol(summary_path=summary_path)
    g_status = relaxed_g_set_status(summary_path)
    rigid_report = _read_json(s54_report)
    rigid_observables = ((rigid_report.get("production") or {}).get("observables")) or {
        name: {"status": _NOT_ATTEMPTED} for name in OBSERVABLES
    }

    try:
        if not protocol.get("ready_to_execute"):
            raise PreregistrationError(f"the protocol is not ready to execute: {protocol.get('blocking')}")
        # Unreachable while GO-10 (S47) stays NO-GO-10 (today, unconditionally)
        # and S51 keeps every relaxed branch blocked. Left as live code, not a
        # hardcoded skip, so a future GO-10 reopen exercises this gate for
        # real instead of falling through a branch that was never written.
        raise RelaxedIntegratedObservablesError(
            "protocol reports ready_to_execute=True but no relaxed-integrated-observable computation "
            "exists yet in this module to run against a real relaxed g_set -- implement it here "
            "(independently converged mesh/broadening for the relaxed geometry, never copied from "
            "S53's rigid ladder) before removing this guard"
        )
    except PreregistrationError as exc:
        observables = {name: {"status": _NOT_ATTEMPTED, "reason": "blocked by g_set"} for name in OBSERVABLES}
        delta = {
            name: rigid_vs_relaxed_delta_report(None, None, None, None) for name in OBSERVABLES
        }
        result = {
            "state": "blocked",
            "result_status": RESULT_STATUS,
            "gate_error": str(exc),
            "blocking": list(protocol.get("blocking", [])),
            "g_set": {"status": _NOT_ATTEMPTED, "reason": str(exc)},
            "observables": observables,
            "relaxed_g_set_status": g_status,
            "rigid_observables": {name: row.get("status") for name, row in rigid_observables.items()},
            "rigid_vs_relaxed_delta": delta,
            "electronic_derivative_backend": _NOT_ATTEMPTED,
            "basis_response_backend": _NOT_ATTEMPTED,
            "phonon_backend": _NOT_ATTEMPTED,
            "epc_backend_class": _NOT_ATTEMPTED,
            "artifact_kind": "not_produced",
            "backend": protocol["backend"],
            "preregistration_sha256": protocol.get("frozen_content_sha256"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    return result


def interpret(production: Mapping[str, Any]) -> dict[str, Any]:
    g_status = production["relaxed_g_set_status"]
    delta_computed = any(row["status"] == "computed" for row in production["rigid_vs_relaxed_delta"].values())
    licensed = [
        {
            "claim": "relaxed_integrated_observables_protocol_correctly_applied",
            "licensed": production["state"] == "blocked" and bool(production.get("blocking")),
            "evidence": "this protocol was built live and its own blocking list is reproduced verbatim",
        },
        {
            "claim": "relaxed_integrated_observables_no_execution_authorised",
            "licensed": not g_status["any_branch_has_g"],
            "evidence": g_status,
        },
        {
            "claim": "no_relaxed_vs_rigid_interpretation_without_exceeding_combined_uncertainty",
            "licensed": not delta_computed,
            "evidence": "every rigid_vs_relaxed_delta entry is not_attempted: neither side has a "
            "computed observable, so no interpretation is drawn from an uncomputed difference",
        },
    ]
    forbidden = [
        {
            "claim": label,
            "reason": (
                "Tc is explicitly out of scope per the roadmap"
                if label == "Tc"
                else "no g has been computed on the relaxed geometry (S47: NO-GO-10) or certified "
                "GO-9 on the rigid one either; there is no measured pair of numbers and no validated "
                "uncertainty to bound a relaxed-vs-rigid integrated-observable difference"
            ),
        }
        for label in FORBIDDEN_PUBLICATION_LABELS
    ]
    all_hold = all(row["licensed"] for row in licensed)
    return {
        "licensed": licensed,
        "forbidden": forbidden,
        "rule": "an observable claim is licensed only if g_set itself is licensed on the relaxed "
        "geometry; a relaxed-vs-rigid claim is licensed only once both sides carry a real computed "
        "value AND the difference exceeds their combined declared uncertainty",
        "all_licensed_claims_hold": all_hold,
        "verdict": "NO_QUANTITATIVE_CLAIM_LICENSED" if all_hold else "VALIDATION_FAILED",
    }


def run(
    summary_path: Path = DEFAULT_S51_SUMMARY_PATH,
    s54_report: Path = DEFAULT_S54_REPORT_PATH,
    *,
    output_dir: Path = DEFAULT_RESULT_DIR,
) -> dict[str, Any]:
    production = produce(summary_path, s54_report)
    interpretation = interpret(production)
    result = {
        "schema": "epc_relaxed_integrated_observables_v1",
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


def _selfcheck() -> None:
    """The one runnable check this module's non-trivial logic needs (ponytail)."""
    # rigid_vs_relaxed_delta_report: symmetric, fires only with both sides present.
    both_missing = rigid_vs_relaxed_delta_report(None, None, None, None)
    assert both_missing["status"] == _NOT_ATTEMPTED

    below_noise = rigid_vs_relaxed_delta_report(1.0, 0.05, 1.02, 0.05)
    assert below_noise["status"] == "computed" and not below_noise["exceeds_combined_uncertainty"]
    assert not below_noise["interpretable"]

    above_noise = rigid_vs_relaxed_delta_report(1.0, 0.01, 1.5, 0.01)
    assert above_noise["exceeds_combined_uncertainty"] and above_noise["interpretable"]

    # Live gate reads: GO-10 is NO-GO-10 today, so g_set must read as absent.
    status = relaxed_g_set_status()
    assert status["go10_verdict"] == "NO-GO-10"
    assert status["any_branch_has_g"] is False

    protocol = build_protocol()
    assert protocol["ready_to_execute"] is False
    assert protocol["blocking"]
    assert prereg.freeze_hash(protocol) == protocol["frozen_content_sha256"]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, default=DEFAULT_S51_SUMMARY_PATH)
    parser.add_argument("--s54-report", type=Path, default=DEFAULT_S54_REPORT_PATH)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    protocol_path = args.protocol_dir / PROTOCOL_NAME

    if args.verify:
        report = verify(protocol_path)
        print(f"pre-registration {report['protocol_id']}: {'INTACT' if report['intact'] else 'TAMPERED'} "
              f"{report['frozen_content_sha256']}")
        return 0 if report["verified"] else 1

    _selfcheck()
    protocol = build_protocol(summary_path=args.summary)
    args.protocol_dir.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"pre-registration {protocol['protocol_id']} -> {protocol_path}")
    print(f"  frozen_content_sha256 : {protocol['frozen_content_sha256']}")
    print(f"  physics review        : {protocol['physics_review']['status']}")
    print(f"  ready to execute      : {protocol['ready_to_execute']}")
    for reason in protocol["blocking"]:
        print(f"    blocked by: {reason}")

    result = run(args.summary, args.s54_report, output_dir=args.output_dir)
    output = args.output_dir / REPORT_NAME
    print(f"[{TICKET}] relaxed integrated observables: {result['overall_claim']}  ->  {output}")
    print(f"[{TICKET}]   interpretation verdict: {result['interpretation']['verdict']}")
    for reason in result["production"].get("blocking", []):
        print(f"[{TICKET}]   blocked by: {reason}")
    for row in result["interpretation"]["forbidden"]:
        print(f"[{TICKET}]   forbidden: {row['claim']}")
    return 0 if result["interpretation"]["verdict"] == "NO_QUANTITATIVE_CLAIM_LICENSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
