#!/usr/bin/env python3
"""Does ``epc_claim_policy`` block a graphene-Gamma-local claim *by association*?

The concern this audit answers: ``claim_policy`` lists ``GO-5`` and ``GO-7``
among the prerequisites of ``fine_tuning_candidate``. GO-5 is the q != 0
commensurate-K gate and GO-7 is the AB-bilayer/S33-S36 gate. If
``fine_tuning_candidate`` means *"a fine-tuning candidate arising from the
graphene Gamma residual"*, then requiring two gates about other systems and
other momenta would be blocking a local conclusion by association.

This script only **reports** the semantics it finds and recommends. It changes
no policy and mutates no gate: adjusting ``epc_claim_policy`` is a decision for
the physics owner, not a side effect of an audit.

The answer is read out of the real consumers of the field, not from its name.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Comparison/scripts"), str(ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402

SCHEMA = "epc_claim_policy_scope_audit_v1"
DEFAULT_OUTPUT = ROOT / "Comparison/results/epc/claim_policy_scope"

POLICY = ROOT / "Comparison/scripts/epc_claim_policy.py"
CONSUMERS = {
    "evaluate_epc_metrics.py": ROOT / "Comparison/scripts/evaluate_epc_metrics.py",
    "interpret_gamma_epc_error.py": ROOT / "Comparison/scripts/interpret_gamma_epc_error.py",
    "quantify_checkpoint_derivative_error.py": ROOT / "Comparison/scripts/quantify_checkpoint_derivative_error.py",
    "epc_commensurate_k.py": ROOT / "Comparison/scripts/epc_commensurate_k.py",
    "evaluate_ab_bilayer_epc.py": ROOT / "Comparison/scripts/evaluate_ab_bilayer_epc.py",
}


def _cite(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(str(path))}


def build_report() -> dict[str, Any]:
    from epc_claim_policy import claim_policy

    all_pass = {
        name: "PASS"
        for name in ("gauge_derivative_audit", "graph2mat_target_gauge_contract",
                     "numerical_PAO_convergence", "checkpoint_lineage", "final_case_exclusion",
                     "gpu_runtime", "GO-5", "GO-7")
    }
    # What the policy does today, executed rather than described.
    without_go5_go7 = claim_policy(
        {**all_pass, "GO-5": "NO_GO", "GO-7": "NO_GO",
         "final_comparator_gauge_conversion_verified": True}
    )
    local_only = claim_policy(
        {**all_pass, "final_comparator_gauge_conversion_verified": True}
    )

    return {
        "schema": SCHEMA,
        "gate": "epc_claim_policy_scope",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "REPORTED_NOT_CHANGED",
        "question": (
            "does claim_policy block a graphene-Gamma-local conclusion by association with "
            "the global q != 0 / TBG gates GO-5 and GO-7?"
        ),
        "observed_semantics": {
            "fine_tuning_candidate_is_global_not_gamma_local": True,
            "evidence": [
                {
                    "where": "evaluate_epc_metrics.py, model_statement.fine_tuning_ticket_authorized",
                    "code": "gates_1_to_5_pass and not model['quantitative_for_g'] "
                            "and promotion['fine_tuning_candidate'] != 'BLOCKED'",
                    "reading": (
                        "the field authorises a RETRAINING TICKET against the model as a whole. "
                        "It is not the name of the graphene-Gamma residual, and it is not a "
                        "per-system quantity: one model gets retrained, so the decision to "
                        "retrain is legitimately global."
                    ),
                    "source": _cite(CONSUMERS["evaluate_epc_metrics.py"]),
                },
                {
                    "where": "interpret_gamma_epc_error.py, retraining ticket authorisation",
                    "code": "authorised = rung == RETRAINING_RUNG and "
                            "promotion['fine_tuning_candidate'] != 'BLOCKED'",
                    "reading": (
                        "the same global reading, cross-checked against the frozen claim "
                        "ladder; a disagreement raises rather than being reconciled."
                    ),
                    "source": _cite(CONSUMERS["interpret_gamma_epc_error.py"]),
                },
            ],
            "separately_scoped_fields_already_exist": {
                "observable": "finite-PAO covariant response in PROD-SZ -- already Gamma-local",
                "full_KS": "governed by delta_out_closure alone, independently of GO-5/GO-7",
                "reading": (
                    "the policy already distinguishes scopes where the physics differs. "
                    "full_KS does not consult GO-5 or GO-7 at all."
                ),
            },
        },
        "why_the_association_concern_does_not_hold": {
            "finding": "the dependency runs the OPPOSITE way from the concern",
            "detail": (
                "GO-5 is not an independent q != 0 obstacle that could contaminate a local "
                "claim. epc_commensurate_k.go5_decision() records algebra_layer = PASS, "
                "physical_layer = NOT_ATTEMPTED, go5 = NO_GO, blocked_by = 'GO-4 "
                "full-KS-EPC-level', blocking_gate = 'gate 5 "
                "(basis_response_validated_and_required_by_formalism_id): the intra-atomic "
                "term is unresolved (C14C)'. GO-4 is the graphene *Gamma* adjudication. So "
                "GO-5 is NO_GO *because the Gamma-local gate is open*, and it flips the day "
                "GO-4 does. GO-7 in turn fails closed unless the candidate records "
                "GO-4 = PASS and GO-5 = PASS. Removing GO-5/GO-7 from the prerequisite list "
                "would therefore not unblock a Gamma-local claim: the same C14C intra-atomic "
                "defect that blocks them blocks GO-4 directly."
            ),
            "sources": [
                _cite(CONSUMERS["epc_commensurate_k.py"]),
                _cite(CONSUMERS["evaluate_ab_bilayer_epc.py"]),
            ],
        },
        "executed_policy_probes": {
            "all_gates_pass_including_GO5_GO7_plus_gauge": local_only,
            "same_but_GO5_GO7_failing": without_go5_go7,
            "note": (
                "the probes are hypothetical gate dictionaries used to exhibit the policy's "
                "branching. They assert nothing about the real gates, which today keep "
                "checkpoint_lineage = NO_GO and the gauge conversion unverified."
            ),
        },
        "go5_go7_still_block_what_they_must": {
            "S33_S36_q_neq_0_small_TBG": "BLOCKED, unchanged by this audit",
            "confirmation": (
                "nothing here relaxes GO-5 or GO-7, and no q != 0, MATBG, S33/S36 or "
                "small-TBG artifact was read, written or re-adjudicated."
            ),
        },
        "recommendation": {
            "change_epc_claim_policy": False,
            "rationale": (
                "1. fine_tuning_candidate is demonstrably a GLOBAL retraining decision in both "
                "of its consumers, so listing global gates among its prerequisites is correct, "
                "not an association error. "
                "2. Even granting the local reading, removing GO-5/GO-7 would not unblock "
                "anything: they are downstream of GO-4, whose own gate 5 (C14C intra-atomic "
                "term) is the real obstruction. The block is causal, not associative. "
                "3. The policy already carries a Gamma-local field -- `observable` -- and a "
                "separately-scoped `full_KS`, so a local conclusion has somewhere to live "
                "that does not consult GO-5/GO-7."
            ),
            "if_a_gamma_local_claim_is_ever_wanted_as_its_own_field": (
                "add a NEW key (e.g. gamma_local_candidate) with its own explicitly named "
                "prerequisites, rather than widening or weakening fine_tuning_candidate. "
                "Never relabel the existing field: downstream reports and the frozen claim "
                "ladder in interpret_gamma_epc_error.py both read it with the global meaning, "
                "and interpret_gamma_epc_error raises when the two disagree."
            ),
            "decision_owner": "physics owner; this audit reports and does not change policy",
        },
        "policy_source": _cite(POLICY),
        "scope": (
            "reads source and executes claim_policy on hypothetical gate dictionaries only. "
            "No gate re-adjudicated, no policy modified, no SIESTA, no training."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = build_report()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "claim_policy_scope.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"epc_claim_policy_scope = {report['verdict']}")
    print(f"  fine_tuning_candidate is global: "
          f"{report['observed_semantics']['fine_tuning_candidate_is_global_not_gamma_local']}")
    print(f"  {report['why_the_association_concern_does_not_hold']['finding']}")
    print(f"  recommend changing policy: "
          f"{report['recommendation']['change_epc_claim_policy']}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
