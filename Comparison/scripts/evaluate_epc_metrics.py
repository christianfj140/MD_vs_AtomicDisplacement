#!/usr/bin/env python3
"""C20 / E-F_001-S26 — GO-4 adjudication of the graphene Gamma candidate.

Applies the pre-registered protocol **unchanged** to the candidate produced by
``run_graphene_gamma_epc_paths.py`` and emits a verdict for the two levels the
roadmap separates: derivative-level and full-KS-EPC-level.

This script computes no physics. Every number it judges was measured by S25
under the frozen protocol; re-deriving any of it here would be measuring the
result a second time with the freedom to measure it differently. What is new is
the adjudication: the nine PASS gates of roadmap section IX.D, evaluated against
recorded evidence, with each failure charged to exactly one layer.

Three rules govern this file, and they are why it is separate from the run:

* **The protocol is not edited to fit the result.** Two frozen pass rules were
  not met. They are reported as failures. The measurement that explains each one
  is carried alongside, but an explanation is not a pass, and amending a rule
  after seeing what it rejects would undo the point of freezing it.
* **Absence is never a pass.** Every gate resolves against a named piece of
  evidence and raises if it is missing. A gate that cannot find its evidence
  fails the run rather than defaulting to True.
* **The floor: "if 1-5 fail, the ML is not to blame."** Gate 6 is the only gate
  that can charge the model, and it may only do so once gates 1-5 have all
  passed. Otherwise the model measurement is reported and explicitly not
  charged, and the fine-tuning ticket stays closed.

Usage::

    .venv/bin/python Comparison/scripts/evaluate_epc_metrics.py
    .venv/bin/python Comparison/scripts/evaluate_epc_metrics.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import preregister_graphene_gamma_epc as prereg  # noqa: E402

SCHEMA = "epc_go4_adjudication_v1"
TICKET = "C20 / E-F_001-S26"
GATE = "GO-4"

DEFAULT_RESULT_ROOT = prereg.DEFAULT_RESULT_ROOT
SUMMARY_NAME = "graphene_gamma_epc_summary.json"
VERDICT_NAME = "go4_verdict.json"

LEVEL_DERIVATIVE = "derivative_level"
LEVEL_EPC = "full_ks_epc_level"

#: The six layers a failure may be charged to. The acceptance criterion is that
#: a failure names exactly one of them, so the set is closed and no gate may
#: invent a seventh.
LAYERS = ("reference", "numerics", "formalism", "basis_response", "phonons", "model")

#: The label that only a PASS may carry.
VALIDATED_LABEL = "validated_small_system_result"

#: Frozen check ids whose evaluated counterpart in the S25 summary carries a
#: longer name. The id on the left is the protocol's; the name on the right is
#: where the measurement lives. Anything not listed here shares its name.
FROZEN_CHECK_ALIASES = {
    "siesta_fc_equals_explicit_fd": "e2g_directions_fc_matches_explicit_fd",
    "jvp_equals_frozen": "jvp_equals_frozen_within_the_frozen_floor",
}

#: A frozen rule that the run classified as asking for more than the formalism
#: it gates can deliver. It is still a failure; this only fixes which layer
#: owns it, and the classification is the run's, made before this adjudication.
FROZEN_RULE_SCOPE_FINDING = "frozen_pass_rule_inconsistent_with_the_formalism_it_gates"


class AdjudicationError(RuntimeError):
    """The candidate cannot be judged under this protocol at all."""


# --------------------------------------------------------------------------
# evidence access -- every reader raises rather than defaulting to a pass
# --------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AdjudicationError(f"missing artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def check_row(summary: Mapping[str, Any], name: str) -> dict[str, Any]:
    """The recorded check called ``name``, or a refusal.

    A gate whose evidence is absent must stop the adjudication. Treating a
    missing check as satisfied is how an unevaluated gate becomes a silent PASS.
    """
    for row in summary.get("checks", ()):
        if row.get("check") == name:
            return dict(row)
    raise AdjudicationError(
        f"the candidate records no check named {name!r}; a gate cannot be judged "
        "against evidence that was never produced"
    )


def check_rows(summary: Mapping[str, Any], names: Sequence[str]) -> list[dict[str, Any]]:
    return [check_row(summary, name) for name in names]


def prefixed_check_rows(summary: Mapping[str, Any], prefix: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in summary.get("checks", ()) if str(row.get("check", "")).startswith(prefix)]
    if not rows:
        raise AdjudicationError(f"the candidate records no check starting with {prefix!r}")
    return rows


def _passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(bool(row.get("passed")) for row in rows)


def _digest(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"check": row.get("check"), "passed": bool(row.get("passed"))} for row in rows]


def precondition(protocol: Mapping[str, Any], name: str) -> dict[str, Any]:
    for row in protocol.get("preconditions", ()):
        if row.get("name") == name:
            return dict(row)
    raise AdjudicationError(f"the frozen protocol declares no precondition named {name!r}")


# --------------------------------------------------------------------------
# the nine gates of roadmap section IX.D
# --------------------------------------------------------------------------


def gate_1_siesta_fc_equals_fd(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    rows = check_rows(summary, ["go2_certified", "e2g_directions_fc_matches_explicit_fd"])
    return {"passed": _passed(rows), "evidence": _digest(rows)}


def gate_2_jvp_equals_coordinates(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    row = check_row(summary, "jvp_equals_coordinate_combination")
    return {
        "passed": bool(row.get("passed")),
        "evidence": _digest([row]),
        "rule": row.get("rule"),
        "sub_checks_failed": row.get("failed"),
    }


def gate_3_jvp_equals_frozen(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    row = check_row(summary, "jvp_equals_frozen_within_the_frozen_floor")
    budget = summary.get("tolerances", {}).get("tau_backend", {})
    entries = [
        {"pair": key, "tau_backend": value.get("tau_backend"), "limit": value.get("limit"),
         "passes": bool(value.get("passes")), "split": value.get("split")}
        for key, value in sorted(budget.items())
    ]
    if not entries:
        raise AdjudicationError("no tau_backend budget was recorded; gate 3 has no evidence")
    return {
        "passed": bool(row.get("passed")) and all(entry["passes"] for entry in entries),
        "evidence": _digest([row]),
        "tau_backend": entries,
    }


def gate_4_hellmann_feynman(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    row = check_row(summary, "generalized_hellmann_feynman_diagonal")
    return {"passed": bool(row.get("passed")), "evidence": _digest([row]), "rule": row.get("rule")}


def gate_5_basis_response(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Gate 5 has two halves, and the candidate satisfies only one of them.

    The protocol's own check asks whether every PAO-covariant response was *built from* a
    basis-response artifact of the production formalism. It passes. The roadmap
    gate asks for a provider that is *validated* -- C14C -- and that precondition
    is NO_GO. Reading only the first half would let an unvalidated provider
    through on the strength of having been used.
    """
    used = check_row(summary, "basis_response_present_and_required")
    shared = check_row(summary, "one_basis_response_per_direction_across_paths")
    certification = precondition(protocol, "C14C basis response")
    validated = certification.get("verdict") == "PASS"
    return {
        "passed": bool(used.get("passed")) and bool(shared.get("passed")) and validated,
        "evidence": _digest([used, shared]),
        "provider_validated": validated,
        "c14c_verdict": certification.get("verdict"),
        "unresolved_term": next(
            (row for row in summary.get("limitations", ()) if isinstance(row, Mapping)), None
        ),
    }


def gate_6_model_against_reference(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Graph2Mat vs SIESTA, as an error on g and never on a matrix.

    Two separate questions, kept separate: does the pre-registered bound *hold*
    on the result split (and did it transfer to the validation split first), and
    is the bound narrow enough to call the checkpoint quantitative. The first is
    the gate; the second is the adequacy statement that decides the claim rung.
    """
    tolerances = summary.get("tolerances", {})
    tau = tolerances.get("tau_model")
    if not tau or "value" not in tau:
        raise AdjudicationError("no tau_model was recorded; gate 6 has no evidence")
    # the frozen definition, including what tau_model may never be derived from
    definition = tolerances.get("definition", {}).get("tau_model", {})
    forbidden = list(definition.get("forbidden_sources", ()) or ())
    if not forbidden:
        raise AdjudicationError(
            "the candidate records no FORBIDDEN_TAU_MODEL_SOURCES; the one rule that keeps a "
            "global matrix norm out of tau_model cannot be shown to have been honoured"
        )
    transfer = check_row(summary, "tau_model_transfers_to_the_validation_split")
    value = float(tau["value"])
    threshold = float(tau["adequacy_threshold"])
    validation_rows = list(tau.get("validation", ()))
    if not validation_rows:
        raise AdjudicationError("tau_model recorded no validation split; gate 6 has no evidence")
    transfers = max(float(row["relative_subspace_error"]) for row in validation_rows) <= value
    result_rows = list(tau.get("result", ())) + list(tau.get("result_graph2mat_frozen", ()))
    if not result_rows:
        raise AdjudicationError("tau_model recorded no result split; gate 6 has no evidence")
    worst = max(float(row["relative_subspace_error"]) for row in result_rows)
    covered = worst <= value
    return {
        "passed": bool(transfer.get("passed")) and transfers and covered,
        "evidence": _digest([transfer]),
        "tau_model": value,
        "worst_relative_subspace_error_on_result": worst,
        "bound_covers_result": covered,
        "transfers_to_validation_split": transfers,
        "adequacy_threshold": threshold,
        "quantitative_for_g": transfers and covered and value <= threshold,
        "contracted_in": definition.get("contracted_in"),
        "not_derived_from": forbidden,
    }


def gate_7_gauge_invariant_subspaces(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    rows = check_rows(
        summary,
        ["go6_subspace_metrics_pass", "electronic_gauge_invariance", "phonon_doublet_gauge_invariance"],
    )
    go6 = summary.get("go6", {})
    return {
        "passed": _passed(rows) and go6.get("verdict") == "PASS",
        "evidence": _digest(rows),
        "go6_verdict": go6.get("verdict"),
    }


def gate_8_acoustic_translation(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    row = check_row(summary, "uniform_translation_null")
    return {
        "passed": bool(row.get("passed")),
        "evidence": _digest([row]),
        "rule": row.get("rule"),
        "acoustic_diagonal_null": row.get("acoustic_diagonal_null"),
        "derivative_level_null": row.get("derivative_level_null"),
        "reproduces_the_commutator_identity": row.get("reproduces_the_commutator_identity"),
        "paths_failing": _paths_failing(row),
    }


def gate_9_no_convention_explains_a_difference(
    summary: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """The audit axis: units, normalization, phase, overlap, basis, symmetry.

    Each axis is a way a disagreement could be an artifact of bookkeeping rather
    than physics. The gate passes only if every axis is closed.
    """
    experiment = protocol.get("experiment", {})
    axes = {
        "units": check_rows(summary, ["units_round_trip"]),
        "normalization": check_rows(
            summary,
            [
                "phonon_zero_point_amplitude_matches_preregistration",
                "phonon_artifact_is_the_preregistered_one",
            ],
        ),
        "phase": prefixed_check_rows(summary, "spectrum_matches_preregistration__"),
        "overlap": check_rows(summary, ["one_basis_response_per_direction_across_paths"]),
        "basis": check_rows(
            summary,
            [
                "e2g_campaign_orbital_contract_identical",
                "geometry_signature_matches_preregistration",
                "phonon_geometry_signature_shared",
            ],
        ),
        "symmetry": check_rows(summary, ["symmetry_of_the_doublet_at_K"]),
    }
    report = {
        name: {"closed": _passed(rows), "evidence": _digest(rows)} for name, rows in axes.items()
    }
    report["phase"]["q_label"] = experiment.get("q_label")
    report["phase"]["k_plus_q_equals_k"] = experiment.get("k_plus_q_equals_k")
    symmetry_row = axes["symmetry"][0]
    report["symmetry"]["paths_failing"] = _paths_failing(symmetry_row)
    report["symmetry"]["member_ratio_spread_across_paths"] = symmetry_row.get(
        "member_ratio_spread_across_paths"
    )
    return {
        "passed": all(entry["closed"] for entry in report.values()),
        "axes": report,
        "open_axes": sorted(name for name, entry in report.items() if not entry["closed"]),
    }


#: The three conceptual paths the experiment compares.
PATHS = ("graph2mat_frozen", "graph2mat_jvp", "siesta_reference")


def _paths_failing(row: Mapping[str, Any]) -> list[str]:
    """Which of the three paths a per-path check failed on.

    A rule that fails identically on all three paths cannot be a property of any
    one backend, which is what keeps such a failure off the model's account. Some
    checks flag the verdict per row and some only record the measurement per
    path; when the check as a whole failed and no row claims to pass, every path
    it measured failed it.
    """
    rows = row.get("rows")
    if not isinstance(rows, list):
        return []
    entries = [entry for entry in rows if isinstance(entry, Mapping)]
    flagged = {str(entry["path"]) for entry in entries if entry.get("passes") is False}
    if flagged or any(entry.get("passes") is True for entry in entries):
        return sorted(flagged)
    if row.get("passed") is False:
        return sorted({str(entry["path"]) for entry in entries if entry.get("path")})
    return []


GATES: tuple[dict[str, Any], ...] = (
    {
        "number": 1,
        "id": "siesta_fc_derivative_equals_explicit_fd",
        "level": LEVEL_DERIVATIVE,
        "layer": "reference",
        "question": "does SIESTA's FC dHSdR reproduce an explicit central FD of the same H and S "
                    "inside the delta plateau?",
        "resolve": gate_1_siesta_fc_equals_fd,
    },
    {
        "number": 2,
        "id": "jvp_equals_coordinate_combination",
        "level": LEVEL_DERIVATIVE,
        "layer": "numerics",
        "question": "does the arbitrary-direction Graph2Mat JVP equal the combination of "
                    "coordinate JVPs?",
        "resolve": gate_2_jvp_equals_coordinates,
    },
    {
        "number": 3,
        "id": "jvp_equals_frozen_graph2mat",
        "level": LEVEL_DERIVATIVE,
        "layer": "numerics",
        "question": "does the JVP equal the same model's own frozen central difference, inside "
                    "the frozen side's floor?",
        "resolve": gate_3_jvp_equals_frozen,
    },
    {
        "number": 4,
        "id": "generalized_hellmann_feynman_diagonal",
        "level": LEVEL_EPC,
        "layer": "formalism",
        "question": "is the diagonal of the physical operator the generalized Hellmann-Feynman "
                    "derivative of the eigenvalue?",
        "resolve": gate_4_hellmann_feynman,
    },
    {
        "number": 5,
        "id": "basis_response_validated_and_required",
        "level": LEVEL_EPC,
        "layer": "basis_response",
        "question": "is the basis-response provider validated, and does the formalism_id require "
                    "it explicitly?",
        "resolve": gate_5_basis_response,
    },
    {
        "number": 6,
        "id": "full_ks_epc_within_pre_registered_tau_model",
        "level": LEVEL_EPC,
        "layer": "model",
        "question": "is Graph2Mat vs SIESTA inside the pre-registered tau_model, defined in the "
                    "electronic subspace that carries g?",
        "resolve": gate_6_model_against_reference,
    },
    {
        "number": 7,
        "id": "gauge_invariant_subspace_metrics_coherent",
        "level": LEVEL_EPC,
        "layer": "formalism",
        "question": "are the GO-6 subspace invariants unmoved by electronic and phonon gauge "
                    "rotations?",
        "resolve": gate_7_gauge_invariant_subspaces,
    },
    {
        "number": 8,
        "id": "uniform_translation_acoustic_check",
        "level": LEVEL_EPC,
        "layer": "formalism",
        "question": "does the uniform-translation direction satisfy the frozen acoustic rule?",
        "resolve": gate_8_acoustic_translation,
    },
    {
        "number": 9,
        "id": "no_difference_explained_by_convention",
        "level": LEVEL_EPC,
        "layer": "formalism",
        "question": "is every difference free of a units, normalization, phase, overlap, basis or "
                    "symmetry explanation?",
        "resolve": gate_9_no_convention_explains_a_difference,
    },
)


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------


def attribute(gate: Mapping[str, Any], outcome: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    """Charge one failure to exactly one of the six layers.

    The default is the gate's own layer. Two corrections apply, and both are read
    off measurements rather than chosen here:

    * a frozen rule the run classified as asking for more than the formalism it
      gates, and which fails identically on all three paths, is a statement about
      the rule's scope -- it stays on ``formalism`` and cannot reach ``model``;
    * gate 6 may only charge the model once gates 1-5 have passed. That is the
      roadmap's floor, and it is the difference between a model that is wrong and
      a model measured against a reference that is not yet validated.
    """
    finding = next(
        (
            row
            for row in summary.get("preregistration_findings", ())
            if row.get("check") in (gate["id"], _summary_check_of(gate))
        ),
        None,
    )
    layer = gate["layer"]
    detail: dict[str, Any] = {"gate": gate["number"], "gate_id": gate["id"], "layer": layer}
    if finding is not None and finding.get("kind") == FROZEN_RULE_SCOPE_FINDING:
        detail["kind"] = FROZEN_RULE_SCOPE_FINDING
        detail["scope"] = (
            "the frozen pass rule is stronger than the physics it gates; the rule is applied "
            "unchanged and reported as failed, and amending it is not this adjudication's call"
        )
        detail["resolution"] = finding.get("resolution")
        detail["measurement"] = finding.get("measurement")
    paths = outcome.get("paths_failing") or (
        outcome.get("axes", {}).get("symmetry", {}).get("paths_failing") if "axes" in outcome else None
    )
    if paths:
        detail["paths_failing"] = paths
        detail["backend_specific"] = len(paths) < 3
        if len(paths) == 3:
            detail["not_charged_to"] = ["model", "reference", "numerics"]
            detail["why"] = (
                "the rule fails identically on all three paths, so it is not a property of any "
                "one backend"
            )
    if layer not in LAYERS:
        raise AdjudicationError(f"gate {gate['number']} names an unknown layer {layer!r}")
    return detail


def _summary_check_of(gate: Mapping[str, Any]) -> str | None:
    return {
        "uniform_translation_acoustic_check": "uniform_translation_null",
        "no_difference_explained_by_convention": "symmetry_of_the_doublet_at_K",
    }.get(gate["id"])


# --------------------------------------------------------------------------
# adjudication
# --------------------------------------------------------------------------


def frozen_check_status(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every frozen check, by the stage it was frozen with.

    Independent of the nine gates: the gates ask the roadmap's questions, this
    asks whether the protocol's own 13 checks were met. Both must hold.
    """
    rows = []
    for spec in protocol.get("checks", ()):
        name = FROZEN_CHECK_ALIASES.get(spec["id"], spec["id"])
        row = check_row(summary, name)
        rows.append(
            {
                "id": spec["id"],
                "evaluated_as": name,
                "stage": spec["stage"],
                "passed": bool(row.get("passed")),
            }
        )
    return rows


def adjudicate(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    stored = protocol.get("frozen_content_sha256")
    if summary.get("preregistration_sha256") != stored:
        raise AdjudicationError(
            "the candidate does not cite the frozen protocol; it was not produced under it"
        )
    if prereg.freeze_hash(protocol) != stored:
        raise AdjudicationError("the protocol was edited after being frozen; nothing can be judged against it")
    if summary.get("GO4_finite_PAO"):
        return publish_finite_pao(summary, protocol)

    outcomes = []
    for gate in GATES:
        outcome = gate["resolve"](summary, protocol)
        row = {
            "gate": gate["number"],
            "id": gate["id"],
            "level": gate["level"],
            "layer": gate["layer"],
            "question": gate["question"],
            "passed": bool(outcome["passed"]),
        }
        row.update({key: value for key, value in outcome.items() if key != "passed"})
        outcomes.append(row)

    frozen = frozen_check_status(summary, protocol)
    verdicts = {}
    for level in (LEVEL_DERIVATIVE, LEVEL_EPC):
        stage = "derivative" if level == LEVEL_DERIVATIVE else "epc"
        gate_rows = [row for row in outcomes if row["level"] == level]
        frozen_rows = [row for row in frozen if row["stage"] == stage]
        failed_gates = [row["gate"] for row in gate_rows if not row["passed"]]
        failed_frozen = [row["id"] for row in frozen_rows if not row["passed"]]
        verdicts[level] = {
            "verdict": "PASS" if not failed_gates and not failed_frozen else "NO_GO",
            "gates": [row["gate"] for row in gate_rows],
            "gates_failed": failed_gates,
            "frozen_checks": len(frozen_rows),
            "frozen_checks_failed": failed_frozen,
        }

    attributions = [
        attribute(next(g for g in GATES if g["number"] == row["gate"]), row, summary)
        for row in outcomes
        if not row["passed"]
    ]

    gates_1_to_5_pass = all(row["passed"] for row in outcomes if row["gate"] <= 5)
    from epc_claim_policy import claim_policy
    promotion = claim_policy(summary.get("scientific_gates", {}))
    model = next(row for row in outcomes if row["gate"] == 6)
    model_statement = {
        "bound_holds_on_result": bool(model["bound_covers_result"]),
        "transfers_to_validation_split": bool(model["transfers_to_validation_split"]),
        "tau_model": model["tau_model"],
        "adequacy_threshold": model["adequacy_threshold"],
        "quantitative_for_g": bool(model["quantitative_for_g"]),
        "charged_to_the_model": gates_1_to_5_pass and not model["passed"],
        "claim_policy": promotion,
        "fine_tuning_ticket_authorized": gates_1_to_5_pass and not model["quantitative_for_g"]
        and promotion["fine_tuning_candidate"] != "BLOCKED",
        "floor": (
            "gates 1-5 all pass, so a residual disagreement may be charged to the model"
            if gates_1_to_5_pass
            else "gates 1-5 do not all pass, so no disagreement is charged to the model: "
                 "'if 1-5 fail, the ML is not to blame'"
        ),
    }

    overall = (
        "PASS"
        if all(entry["verdict"] == "PASS" for entry in verdicts.values())
        else "NO_GO"
    )
    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration_sha256": stored,
        "protocol_id": protocol.get("protocol_id"),
        "candidate": summary.get("ticket"),
        "candidate_result_class": summary.get("result_class"),
        # The adjudication judges the candidate; it does not promote it. The
        # status stays the candidate's own -- a GO-4 PASS is still `candidate`
        # until an independent validation reproduces it, which is a separate
        # axis from this verdict.
        "status": summary.get("status"),
        "formalism_id": summary.get("formalism_id"),
        "verdict": overall,
        "levels": verdicts,
        "gates": outcomes,
        "frozen_checks": frozen,
        "failure_attribution": attributions,
        "model_statement": model_statement,
        "claim": select_claim(overall, verdicts, summary, protocol, model_statement),
        "claim_policy": promotion,
        "label": VALIDATED_LABEL if overall == "PASS" and promotion["final_claim"] != "BLOCKED" else None,
        "label_rule": (
            f"a PASS on both levels plus independent claim-policy prerequisites is required for {VALIDATED_LABEL}; this candidate "
            f"keeps result_class {summary.get('result_class')!r} and status "
            f"{summary.get('status')!r}"
        ),
        "limitations": summary.get("limitations"),
        "blocking": protocol.get("blocking"),
    }


def publish_finite_pao(
    summary: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """C21: publish C19/C20's terminal finite-PAO verdict without recomputing it."""
    verdict = summary.get("GO4_finite_PAO")
    if verdict not in {"PASS", "NO_GO:model_accuracy"}:
        raise AdjudicationError(f"non-terminal GO4_finite_PAO verdict: {verdict!r}")
    if summary.get("blocking_checks_failed"):
        raise AdjudicationError(
            f"finite-PAO summary has blocking failures: {summary['blocking_checks_failed']}"
        )

    specs = (
        (1, "read_only_preflight", "reference"),
        (2, "e2g_directions_fc_matches_explicit_fd", "reference"),
        (3, "jvp_equals_coordinate_combination", "numerics"),
        (4, "jvp_equals_frozen_within_the_frozen_floor", "numerics"),
        (5, "basis_response_present_and_required", "basis_response"),
        (6, "go6_subspace_metrics_pass", "formalism"),
    )
    gates = []
    for number, name, layer in specs:
        row = check_row(summary, name)
        gates.append(
            {
                "gate": number,
                "id": name,
                "level": "finite_PAO",
                "layer": layer,
                "question": row.get("detail"),
                "passed": bool(row.get("passed")),
                "evidence": _digest([row]),
            }
        )
    tau = summary["tolerances"]["tau_model"]
    transfer = check_row(summary, "tau_model_transfers_to_the_validation_split")
    validation_rows = list(tau.get("validation", ()))
    if not validation_rows:
        raise AdjudicationError("tau_model recorded no validation split; gate 7 has no evidence")
    result_rows = list(tau.get("result", ())) + list(tau.get("result_graph2mat_frozen", ()))
    if not result_rows:
        raise AdjudicationError("tau_model recorded no result split; gate 7 has no evidence")
    bound_covers_result = max(float(row["relative_subspace_error"]) for row in result_rows) <= float(
        tau["value"]
    )
    transfers_to_validation_split = (
        bool(transfer.get("passed"))
        and max(float(row["relative_subspace_error"]) for row in validation_rows) <= float(tau["value"])
    )
    model_quantitative = (
        transfers_to_validation_split
        and bound_covers_result
        and float(tau["value"]) <= float(tau["adequacy_threshold"])
    )
    gates.append(
        {
            "gate": 7,
            "id": "tau_model_adequacy",
            "level": "finite_PAO",
            "layer": "model",
            "question": "is tau_model within the preregistered quantitative-accuracy threshold?",
            "passed": model_quantitative,
            "evidence": [],
        }
    )
    if any(not row["passed"] for row in gates[:-1]):
        raise AdjudicationError("a prerequisite gate failed despite blocking_checks_failed=[]")

    verdict = "PASS" if model_quantitative else "NO_GO:model_accuracy"
    findings = list(summary.get("preregistration_findings") or [])
    ladder = list(protocol["claim_ladder"])
    claim_index = 0 if verdict == "PASS" else 1
    claim = {**ladder[claim_index], "rung": claim_index + 1, "also_active": []}
    if verdict == "NO_GO:model_accuracy":
        claim["preregistered_claim"] = claim.get("claim")
        claim["claim"] = (
            "siesta_reference_g_validated__graph2mat_g_not_quantitative; "
            "model work is recommended but not authorized"
        )
    claim["publishable"] = claim["claim"]
    frozen = frozen_check_status(summary, protocol)
    finding_names = {row.get("check") for row in findings}
    for row in frozen:
        row["nonblocking_scope_finding"] = row["evaluated_as"] in finding_names

    return {
        "schema": SCHEMA,
        "ticket": "C21 / E-F_001-S28",
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration_sha256": protocol.get("frozen_content_sha256"),
        "protocol_id": protocol.get("protocol_id"),
        "candidate": summary.get("ticket"),
        "candidate_result_class": summary.get("result_class"),
        "result_class": summary.get("result_class"),
        "status": "complete",
        "scientific_verdict": "NO_GO" if verdict == "NO_GO:model_accuracy" else "PASS",
        "scientific_reason": "model_accuracy" if verdict == "NO_GO:model_accuracy" else None,
        "formalism_id": summary.get("formalism_id"),
        "verdict": verdict,
        "GO4_finite_PAO": verdict,
        "full_KS": summary.get("full_KS"),
        "levels": {
            LEVEL_DERIVATIVE: {"verdict": "PASS", "gates_failed": []},
            "finite_PAO": {"verdict": verdict, "gates_failed": [] if verdict == "PASS" else [7]},
            LEVEL_EPC: {"verdict": "BLOCKED", "reason": "Delta_out"},
        },
        "gates": gates,
        "frozen_checks": frozen,
        "nonblocking_protocol_findings": findings,
        "failure_attribution": [] if verdict == "PASS" else [
            {
                "gate": 7,
                "gate_id": "tau_model_adequacy",
                "layer": "model",
                "kind": "model_accuracy",
            }
        ],
        "model_statement": {
            "tau_model": tau.get("value"),
            "adequacy_threshold": tau.get("adequacy_threshold"),
            "quantitative_for_g": model_quantitative,
            "bound_covers_result": bound_covers_result,
            "bound_holds_on_result": bound_covers_result,
            "transfers_to_validation_split": transfers_to_validation_split,
            "charged_to_the_model": verdict == "NO_GO:model_accuracy",
            "fine_tuning_ticket_authorized": False,
            "floor": "reference, numerics, basis response and subspace gates passed; full_KS remains blocked by Delta_out",
        },
        "claim": claim,
        "recommended_next_work": (
            "fine-tune or replace the Graph2Mat checkpoint against the frozen finite-PAO target"
            if verdict == "NO_GO:model_accuracy"
            else None
        ),
        "label": None,
        "label_rule": "finite-PAO validation is complete; NO_GO:model_accuracy is a terminal result, not a validated-model label",
        "limitations": summary.get("limitations"),
        "blocking": [],
    }


def select_claim(
    overall: str,
    verdicts: Mapping[str, Mapping[str, Any]],
    summary: Mapping[str, Any],
    protocol: Mapping[str, Any],
    model_statement: Mapping[str, Any],
) -> dict[str, Any]:
    """Which rung of the frozen claim ladder this candidate reaches.

    The rungs are not mutually exclusive, so they are applied in the order the
    protocol's own outcomes imply: a failing derivative-level or formalism check
    is the strongest statement and outranks the rest.
    """
    ladder = list(protocol.get("claim_ladder", ()))
    if not ladder:
        raise AdjudicationError("the frozen protocol carries no claim ladder")
    if overall == "PASS":
        rung = 0 if model_statement["quantitative_for_g"] else 1
    else:
        rung = 2
    entry = dict(ladder[rung])
    entry["rung"] = rung + 1
    entry["also_active"] = [
        dict(row, rung=index + 1)
        for index, row in enumerate(ladder)
        if index != rung and "intra_atomic" in str(row.get("outcome", ""))
    ]
    entry["publishable"] = (
        summary.get("claim")
        if overall != "PASS"
        else "the validated three-path result at the rung selected above"
    )
    return entry


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=prereg.DEFAULT_OUTPUT_DIR / prereg.PROTOCOL_NAME,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-check the frozen hash and that every result cites it, then adjudicate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    protocol = prereg.load_protocol(args.protocol)
    summary = _read_json(Path(args.result_root) / SUMMARY_NAME)

    if args.verify:
        status = prereg.verify(args.protocol, result_root=args.result_root)
        if not status["intact"]:
            print("protocol hash mismatch: the pre-registration was edited after freezing")
            return 1
        if status["results_not_citing_the_protocol"]:
            print(f"results not citing the protocol: {status['results_not_citing_the_protocol']}")
            return 1

    report = adjudicate(summary, protocol)
    destination = Path(args.result_root) / VERDICT_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    frozen_paths = [
        Path(args.protocol),
        Path(args.result_root) / SUMMARY_NAME,
        Path(args.result_root) / "embedded_read_only_split_fc_fd_v2.json",
        destination,
    ]
    snapshot = {
        "schema": "epc_c21_frozen_snapshot_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration_sha256": report["preregistration_sha256"],
        "files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in frozen_paths
            if path.is_file()
        },
    }
    with (Path(args.result_root) / "c21_frozen_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"{GATE} adjudication of {report['candidate']}")
    for level, entry in report["levels"].items():
        print(f"  {level:<20} {entry['verdict']}", end="")
        if entry.get("gates_failed"):
            print(f"  (gates {entry['gates_failed']} failed)", end="")
        if entry.get("frozen_checks_failed"):
            print(f"  (frozen: {entry['frozen_checks_failed']})", end="")
        print()
    print(f"  overall              {report['verdict']}")
    for row in report["failure_attribution"]:
        print(f"    gate {row['gate']} -> {row['layer']}")
    print(f"  label                {report['label']}")
    print(f"  written              {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
