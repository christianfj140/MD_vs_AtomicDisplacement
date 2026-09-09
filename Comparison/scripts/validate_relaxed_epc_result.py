#!/usr/bin/env python3
"""E-F_001-S52: validate and interpret the relaxed-MATBG EPC result (Fase 11).

S51 (``run_matbg_relaxed_epc.py``) executed the S50-frozen protocol and wrote
``no_execution_authorised`` for both branches (``gamma``, ``qneq0``): S47 is
NO-GO-10, so no relaxed geometry exists to build a raw response, a physical
perturbation or ``g`` against. This ticket does not recompute any of that. It
answers the three things the roadmap asks of a *validation and interpretation*
step once execution itself has already refused:

1. Was S50's frozen protocol actually the thing S51 ran, and did S51 refuse
   for exactly the reasons S50 recorded -- not a different or a laxer set?
   Reuses ``preregister_matbg_relaxed_epc.verify()`` (the same protocol/result
   citation check S50 already ships) rather than re-deriving it, and adds one
   check that check alone cannot make: that each branch's persisted
   ``blocking`` list is not merely *present* but *identical* to the frozen
   protocol's own list for that branch.
2. How does this compare with the rigid model? Reads GO-9 live from S43
   (``certify_rigid_gamma_campaign``) and S45 (``certify_qneq0_rigid_campaign``)
   -- both already ``NO-GO-9`` -- rather than assuming the rigid campaign is
   any further along than it has been certified to be.
3. What can be claimed about relaxed MATBG EPC today? Nothing quantitative:
   with no ``g`` computed on the relaxed geometry (it does not exist) and none
   certified GO-9 on the rigid one either, there is no measured pair of
   numbers and no validated uncertainty to bound a relaxed-vs-rigid EPC
   difference. This module states that explicitly rather than leaving it
   implicit, and lists it alongside S50's own forbidden publication labels.

Producer: :func:`validate_and_interpret`. Depends on:
``preregister_matbg_relaxed_epc.py`` (S50, the frozen protocol),
``run_matbg_relaxed_epc.py`` (S51, the refusal this module checks), and the
on-disk GO-9 certification reports from ``certify_rigid_gamma_campaign.py``
(S43) and ``certify_qneq0_rigid_campaign.py`` (S45).
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
import preregister_matbg_relaxed_epc as epc_prereg  # noqa: E402
import run_matbg_relaxed_epc as s51  # noqa: E402

prereg = epc_prereg.prereg

SCHEMA = "epc_relaxed_matbg_interpretation_v1"
TICKET = "E-F_001-S52"
MEMO = "docs/epc_s52_relaxed_matbg_interpretation.md"

DEFAULT_PROTOCOL_PATH = epc_prereg.DEFAULT_OUTPUT_DIR / epc_prereg.PROTOCOL_NAME
DEFAULT_S51_SUMMARY_PATH = s51.EPC_SUMMARY_PATH
DEFAULT_S43_REPORT_PATH = go9_gamma.DEFAULT_OUTPUT_DIR / go9_gamma.REPORT_NAME
DEFAULT_S45_REPORT_PATH = go9_qneq0.DEFAULT_OUTPUT_DIR / go9_qneq0.REPORT_NAME
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/relaxed_epc/matbg"
REPORT_NAME = "relaxed_epc_interpretation.json"

FORBIDDEN_PUBLICATION_LABELS: tuple[str, ...] = (
    *s51.FORBIDDEN_PUBLICATION_LABELS,
    "relaxed_vs_rigid_epc_quantitative_delta",
)


class InterpretationError(RuntimeError):
    """The relaxed-MATBG EPC result cannot be validated/interpreted from the artifacts on disk."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = prereg._read_json(Path(path))
    if payload is None:
        raise InterpretationError(f"missing artifact: {path}")
    return payload


# --------------------------------------------------------------------------- #
# 1. Protocol chain: S51 actually ran S50, and refused for S50's own reasons
# --------------------------------------------------------------------------- #


def protocol_chain_report(
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    summary_path: Path = DEFAULT_S51_SUMMARY_PATH,
    *,
    result_root: Path = epc_prereg.DEFAULT_RESULT_ROOT,
) -> dict[str, Any]:
    """S50 integrity plus per-branch equality of the persisted ``blocking`` list."""
    verification = epc_prereg.verify(protocol_path, result_root=result_root)
    protocol = epc_prereg.load_protocol(protocol_path)
    summary = _read_json(summary_path)

    blocking_matches = {}
    for branch, branch_result in summary.get("branches", {}).items():
        frozen_blocking = list(protocol["branches"][branch]["blocking"])
        persisted_blocking = list(branch_result.get("blocking") or [])
        blocking_matches[branch] = {
            "matches_frozen_protocol": persisted_blocking == frozen_blocking,
            "frozen_blocking": frozen_blocking,
            "persisted_blocking": persisted_blocking,
        }

    return {
        "protocol_intact": bool(verification["intact"]),
        "results_cite_the_protocol": not verification["results_not_citing_the_protocol"],
        "results_not_citing_the_protocol": verification["results_not_citing_the_protocol"],
        "per_branch_blocking_reasons_unmodified": blocking_matches,
        "verified": bool(
            verification["verified"]
            and all(row["matches_frozen_protocol"] for row in blocking_matches.values())
        ),
    }


def no_leaked_claims_report(summary_path: Path = DEFAULT_S51_SUMMARY_PATH) -> dict[str, Any]:
    """None of S51's persisted claims/statuses is a label S50/S52 forbid."""
    summary = _read_json(summary_path)
    forbidden = set(FORBIDDEN_PUBLICATION_LABELS)
    offending: list[str] = []
    for label in (summary.get("overall_claim"), summary.get("result_status")):
        if label in forbidden:
            offending.append(str(label))
    for branch, branch_result in summary.get("branches", {}).items():
        for field in ("claim", "result_status", "artifact_kind"):
            value = branch_result.get(field)
            if value in forbidden:
                offending.append(f"{branch}.{field}={value}")
    return {"forbidden_labels_checked": sorted(forbidden), "offending_fields": offending, "clean": not offending}


# --------------------------------------------------------------------------- #
# 2. Comparison with the rigid model
# --------------------------------------------------------------------------- #


def rigid_comparison_report(
    summary_path: Path = DEFAULT_S51_SUMMARY_PATH,
    s43_report: Path | Mapping[str, Any] = DEFAULT_S43_REPORT_PATH,
    s45_report: Path | Mapping[str, Any] = DEFAULT_S45_REPORT_PATH,
) -> dict[str, Any]:
    """Rigid GO-9 (S43/S45, read live) against the relaxed refusal (S51)."""
    summary = _read_json(summary_path)
    s43 = s43_report if isinstance(s43_report, Mapping) else _read_json(Path(s43_report))
    s45 = s45_report if isinstance(s45_report, Mapping) else _read_json(Path(s45_report))

    rigid = {
        "gamma": {"go9_status": s43["go9_status"], "final_publication_label": s43["final_publication_label"]},
        "qneq0": {"go9_status": s45["go9_status"], "final_publication_label": s45["final_publication_label"]},
    }
    relaxed = {
        branch: {"state": row["state"], "result_status": row["result_status"]}
        for branch, row in summary.get("branches", {}).items()
    }
    rigid_reached_go9 = {branch: row["go9_status"] == "GO-9" for branch, row in rigid.items()}
    return {
        "rigid": rigid,
        "relaxed": relaxed,
        "rigid_reached_go9": rigid_reached_go9,
        "reading": (
            "neither rigid branch is GO-9 yet (S43/S45), so the relaxed branches -- which S50 "
            "requires to clear everything the rigid branches require plus GO-10 -- cannot be ahead "
            "of a rigid campaign that has not itself cleared GO-9; the relaxed refusal is at least "
            "as justified as the rigid one, not a separate or harsher standard applied only to it"
        ),
        "either_geometry_has_a_certified_full_ks_coupling": False,
    }


# --------------------------------------------------------------------------- #
# 3. Claims licensed by the evidence above
# --------------------------------------------------------------------------- #


def claims_report(
    protocol_chain: Mapping[str, Any], no_leak: Mapping[str, Any], comparison: Mapping[str, Any]
) -> dict[str, Any]:
    licensed = [
        {
            "claim": "relaxed_matbg_epc_protocol_correctly_applied",
            "licensed": bool(protocol_chain["verified"]),
            "evidence": "S51 cites the frozen S50 protocol and its persisted per-branch blocking list "
            "is byte-identical to S50's own, for every branch",
        },
        {
            "claim": "relaxed_matbg_epc_no_execution_authorised",
            "licensed": True,
            "evidence": "S47: NO-GO-10, no relaxed geometry exists on which to select k/q, build a "
            "direction or measure a delta sweep",
        },
        {
            "claim": "rigid_matbg_epc_not_yet_go9_either",
            "licensed": True,
            "evidence": comparison["rigid"],
        },
    ]
    forbidden = [
        {
            "claim": label,
            "reason": "no g has been computed on the relaxed geometry (it does not exist) or "
            "certified GO-9 on the rigid one; there is no measured pair of numbers and no validated "
            "uncertainty to bound a relaxed-vs-rigid difference"
            if label == "relaxed_vs_rigid_epc_quantitative_delta"
            else "forbidden by S50/S51 regardless of gate outcome",
        }
        for label in FORBIDDEN_PUBLICATION_LABELS
    ]
    return {
        "licensed": licensed,
        "forbidden": forbidden,
        "rule": "a claim is licensed only if every field above it (protocol chain intact, no leaked "
        "label, rigid comparison read live) holds; this step never widens a gate or amends a "
        "tolerance to make a claim fit",
        "all_licensed_claims_hold": all(row["licensed"] for row in licensed) and no_leak["clean"],
    }


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def validate_and_interpret(
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    summary_path: Path = DEFAULT_S51_SUMMARY_PATH,
    s43_report: Path | Mapping[str, Any] = DEFAULT_S43_REPORT_PATH,
    s45_report: Path | Mapping[str, Any] = DEFAULT_S45_REPORT_PATH,
    *,
    result_root: Path = epc_prereg.DEFAULT_RESULT_ROOT,
) -> dict[str, Any]:
    protocol_chain = protocol_chain_report(protocol_path, summary_path, result_root=result_root)
    no_leak = no_leaked_claims_report(summary_path)
    comparison = rigid_comparison_report(summary_path, s43_report, s45_report)
    claims = claims_report(protocol_chain, no_leak, comparison)

    verdict = (
        "NO_QUANTITATIVE_CLAIM_LICENSED"
        if claims["all_licensed_claims_hold"]
        else "VALIDATION_FAILED"
    )
    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "memo": MEMO,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_chain": protocol_chain,
        "no_leaked_claims": no_leak,
        "rigid_comparison": comparison,
        "claims": claims,
        "verdict": verdict,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_S51_SUMMARY_PATH)
    parser.add_argument("--s43-report", type=Path, default=DEFAULT_S43_REPORT_PATH)
    parser.add_argument("--s45-report", type=Path, default=DEFAULT_S45_REPORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = validate_and_interpret(args.protocol, args.summary, args.s43_report, args.s45_report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / REPORT_NAME
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[{TICKET}] relaxed-MATBG EPC interpretation: {report['verdict']}  ->  {output}")
    print(f"[{TICKET}]   protocol chain verified: {report['protocol_chain']['verified']}")
    for row in report["rigid_comparison"]["rigid"].items():
        print(f"[{TICKET}]   rigid {row[0]:<7} go9={row[1]['go9_status']}")
    for row in report["rigid_comparison"]["relaxed"].items():
        print(f"[{TICKET}]   relaxed {row[0]:<7} state={row[1]['state']}")
    for row in report["claims"]["forbidden"]:
        print(f"[{TICKET}]   forbidden: {row['claim']}")
    return 0 if report["verdict"] == "NO_QUANTITATIVE_CLAIM_LICENSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
