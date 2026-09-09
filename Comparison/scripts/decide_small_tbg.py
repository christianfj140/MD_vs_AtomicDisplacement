#!/usr/bin/env python3
"""E-F_001-S36 / D05: keep small commensurate TBG UNDECIDED until GO-5/GO-7 pass.

Fase 9 of the roadmap asks a narrow question: does anything about *twist* --
a moire supercell built from two independently rotated per-layer
commensuration matrices, not a single-lattice supercell (K) or an untwisted
two-layer stack (AB) -- introduce a mapping/gauge ambiguity that graphene K
and the AB bilayer cannot exercise? "More complexity" is explicitly not an
acceptable reason on its own (roadmap Fase 9, gate 7).

This module computes no new physics. It reads the two verdicts that already
exist -- :func:`epc_commensurate_k.go5_decision` (GO-5, always cheap to
recompute) and the GO-7 verdict written by ``evaluate_ab_bilayer_epc.py`` (an
artifact, read as-is, never re-derived here: the same C02 rule the rest of
this repository follows) -- and asks specifically which of their checks are
about layer/twist geometry construction, as opposed to the unrelated blockers
(C14C's intra-atomic basis-response term; GO-8a's phonon branch-matching
robustness) that would apply identically to a small TBG and therefore are not
evidence for building one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import epc_commensurate_k as _ck  # noqa: E402
import run_ab_bilayer_epc_paths as _ab  # noqa: E402

SCHEMA = "epc_small_tbg_decision_v1"
TICKET = "E-F_001-S36"
MEMO = "docs/epc_s36_small_tbg_decision.md"

#: Where evaluate_ab_bilayer_epc.py writes its verdict (VERDICT_NAME there).
GO7_VERDICT_PATH = _ab.DEFAULT_RESULT_ROOT / "go7_verdict.json"

#: GO-7 checks whose failure would be a layer/twist geometry-mapping problem,
#: as opposed to a phonon-provider or basis-response problem. Mirrors the
#: ``layer`` field evaluate_ab_bilayer_epc.py._check() attaches to each row.
GO7_MAPPING_RELEVANT_LAYERS = ("geometry_construction", "reference")

#: GO-5 algebra-layer checks: all of them are about supercell/q-periodicity/
#: complex-mode/subspace machinery that a moire cell would reuse verbatim
#: (Sec. 2 of docs/epc_s30_graphene_k_conmensurable.md), so all are relevant.
GO5_MAPPING_RELEVANT_CHECKS = (
    "k_to_k_plus_q_by_subspace",
    "q_and_minus_q",
    "cell_origin_shift",
    "atom_across_periodic_boundary",
    "alternative_atomic_phase_convention",
)


class SmallTbgDecisionError(RuntimeError):
    """Raised when the evidence this decision depends on is missing."""


def _load_go7_verdict() -> dict[str, Any] | None:
    if not GO7_VERDICT_PATH.exists():
        return None
    return json.loads(GO7_VERDICT_PATH.read_text())


def small_tbg_decision() -> dict[str, Any]:
    """The RUN / SKIP_WITH_JUSTIFICATION decision, as data.

    Returns a dict rather than a bare string so that *which* risks are
    closed, and by which artifact, stays inspectable -- the same rule
    ``go5_decision()`` and ``evaluate_ab_bilayer_epc.adjudicate()`` follow:
    nothing here is a prose-only claim.
    """
    go5 = _ck.go5_decision()
    go7 = _load_go7_verdict()

    go5_mapping_checks = {name: go5["checks"][name] for name in GO5_MAPPING_RELEVANT_CHECKS}
    go5_mapping_pass = all(v == "PASS" for v in go5_mapping_checks.values())

    if go7 is None:
        raise SmallTbgDecisionError(
            f"GO-7 verdict not found at {GO7_VERDICT_PATH}; run "
            "run_ab_bilayer_epc_paths.py then evaluate_ab_bilayer_epc.py before "
            "deciding -- this module never infers a verdict that was not written."
        )

    go7_mapping_checks = {
        row["check"]: row["passed"]
        for row in go7["checks"]
        if row["layer"] in GO7_MAPPING_RELEVANT_LAYERS
    }
    go7_mapping_pass = all(go7_mapping_checks.values())
    go7_non_mapping_failures = [
        row["check"] for row in go7["checks"] if not row["passed"] and row["layer"] not in GO7_MAPPING_RELEVANT_LAYERS
    ]

    upstream_pass = go5.get("go5") == "PASS" and go7.get("verdict") == "PASS"
    mapping_risk_closed = upstream_pass and go5_mapping_pass and go7_mapping_pass
    verdict = "SKIP_WITH_JUSTIFICATION" if mapping_risk_closed else "UNDECIDED"

    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "memo": MEMO,
        "verdict": verdict,
        "status": "DECIDED" if upstream_pass else "BLOCKED",
        "risks_closed": {
            "commensurate_supercell_and_q_periodicity_algebra": {
                "closed_by": "GO-5 algebra layer (docs/epc_s31_graphene_k_go5_verdict.md)",
                "checks": go5_mapping_checks,
            },
            "multi_layer_geometry_construction": {
                "closed_by": "GO-7 geometry_construction/reference checks "
                "(evaluate_ab_bilayer_epc.py go7_verdict.json)",
                "checks": go7_mapping_checks,
            },
            "twist_specific_moire_mapping_at_production_scale": {
                "closed_by": "materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf is the "
                "single geometry shared by run_tbg_pure_graph2mat_campaign.py and the "
                "independent Moon-Koshino run_tbg_tight_binding.py; the latter measures "
                "v*/v_F = 0.012 (docs/tbg_tight_binding_reference.md), the magic-angle "
                "suppression signature, on that exact geometry -- evidence the per-layer "
                "commensuration/ORB_INDX mapping is already correct at the real target "
                "scale, not a hypothesis a smaller practice cell would newly test",
            },
        },
        "not_evidence_for_running": {
            "reason": "the following are open, but apply identically to Gamma, K, AB, and any "
            "TBG regardless of size or twist, so a small TBG would not close them -- it would "
            "measure the same attributed blocks a third time, exactly what "
            "docs/epc_s31_graphene_k_go5_verdict.md Sec. 3 already refused to do for K",
            "go4_go5_blocker": go5["blocked_by"],
            "go7_non_mapping_failures": go7_non_mapping_failures,
        },
        "reopen_if": "a genuinely twist-specific ambiguity surfaces during GO-8a/GO-8b work "
        "that the production (31,30) geometry + Moon-Koshino cross-check did not exercise "
        "(e.g. a PAO-projected coupling gauge tied to the rotation-center choice) -- then build the small "
        "cell as an algebra-only toy analogous to S30's chain, not as a PAO-covariant coupling campaign, "
        "since GO-1/GO-4 remain the blocker for physical numbers regardless of cell size.",
    }


def _self_test() -> None:
    decision = small_tbg_decision()
    assert decision["verdict"] in ("UNDECIDED", "RUN", "SKIP_WITH_JUSTIFICATION")
    assert decision["risks_closed"]["commensurate_supercell_and_q_periodicity_algebra"]["checks"]
    assert decision["risks_closed"]["multi_layer_geometry_construction"]["checks"]
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    _self_test()
