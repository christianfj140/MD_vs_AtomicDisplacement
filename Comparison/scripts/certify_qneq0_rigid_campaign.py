#!/usr/bin/env python3
"""E-F_001-S45: GO-9 certification of the rigid-MATBG q != 0 (non-Gamma) branch.

S44 (``preregister_matbg_qneq0_rigid_epc.py``, ``docs/epc_s44_matbg_qneq0_rigid_preregistration.md``)
froze the non-Gamma protocol but could not authorise any execution: unlike the
Gamma branch (S41/S42), whose claim ladder pre-authorises a narrow
raw-derivative rung regardless of GO-8a, the q != 0 protocol's own claim
ladder requires GO-8a *and* GO-8b before anything beyond route mechanics, and
``ready_to_execute`` is unconditionally ``False`` today (``q_selection`` and
``qaware_basis_response_producer_missing`` are always-present blockers, S44
Sec. 1). There is therefore no ``epc_qneq0_rigid()`` execution stage to
certify the way S43 certified ``epc_gamma_rigid()`` -- this ticket instead
certifies the thing the roadmap actually asks for at this stage: that GO-5
and GO-8b are read live (not asserted), that GO-9(q != 0) is a strict
conjunction of both (neither substitutes for the other), that the mechanics
S40 already proved on the real production JVP still hold (cache/restart,
GPU preflight, sparse round-trip, scaling, Gamma-JVP-reuse rejection --
reused verbatim from ``certify_qneq0_matbg_response.py``, not re-implemented),
and -- the specific failure mode this ticket exists to close -- that a phase
or basis-response defect forces ``NO-GO`` even when the raw ``D_H`` matrices
involved are individually well-formed (finite, no crash).

Verdict today: ``NO-GO-9``. GO-5 is ``NO_GO`` (blocked by GO-4/C14C),
``GO-8b`` is ``NO-GO-8b`` (blocked by the same GO-5), and no real execution
of the q != 0 branch against the production ``(31, 30)`` checkpoint has ever
been measured, so ``resource_budget_suite`` stays honestly ``NOT_RUN``
(mirroring ``certify_rigid_gamma_campaign.py``, S43).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import benchmark_matbg_synthetic_displacements as matbg_bench  # noqa: E402
import certify_qneq0_matbg_response as go8b_cert  # noqa: E402
import epc_commensurate_k as ck  # noqa: E402
import epc_qb_qc_graph2mat_kernel as g2m_kernel  # noqa: E402
import preregister_matbg_qneq0_rigid_epc as qneq0_prereg  # noqa: E402

SCHEMA = "rigid_qneq0_campaign_go9_certification_v1"
TICKET = "E-F_001-S45"
MEMO = "docs/epc_s45_rigid_qneq0_campaign_certification.md"
GATE = "GO-9 (q != 0 branch)"
FINAL_PUBLICATION_LABEL = "rigid_tbg_model"
FORBIDDEN_PUBLICATION_LABELS = ("quantitative_relaxed_matbg_prediction", "g_mn_nu", "phonon_eigenmode")

DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/certification/matbg"
REPORT_NAME = "s45_rigid_qneq0_campaign_go9_certification.json"

# A phase-convention bug plausible enough to be real: flipping the sign of q
# (S40's own check_q_and_minus_q already proves q and -q give genuinely
# different, non-trivial results, so this is not a degenerate corruption).
_GUARD_K_PRIMITIVE = (0.15, 0.0, 0.0)
_GUARD_Q_PRIMITIVE = (0.2, 0.0, 0.0)


class RigidQneq0CampaignCertificationError(RuntimeError):
    """A GO-9 (q != 0) certification was built, read, or trusted in a way this gate forbids."""


def check(name: str, passed: bool, detail: str, *, applicable: bool = True, **extra: Any) -> dict[str, Any]:
    return {
        "check": name,
        "passed": bool(passed) if applicable else True,
        "applicable": bool(applicable),
        "detail": detail,
        **extra,
    }


# --------------------------------------------------------------------------- #
# Suite A: gate authorization / provenance against the real, frozen S44 protocol
# --------------------------------------------------------------------------- #


def _certify_gate_authorization(tmp_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    protocol = qneq0_prereg.load_protocol()

    verification = qneq0_prereg.verify()
    checks.append(
        check(
            "protocol_hash_intact", verification["intact"],
            f"recomputed_sha256={verification['recomputed_sha256']} stored={verification['frozen_content_sha256']}",
        )
    )

    tampered_path = tmp_dir / "tampered_protocol.json"
    tampered = dict(protocol)
    tampered["frozen_content_sha256"] = "tampered"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    rejected = False
    try:
        qneq0_prereg.require_frozen_protocol(tampered_path)
    except qneq0_prereg.PreregistrationError:
        rejected = True
    checks.append(check("tampered_protocol_is_rejected", rejected, "an edited protocol must raise PreregistrationError"))

    blocked, blocked_reason = False, None
    try:
        qneq0_prereg.require_frozen_protocol()
    except qneq0_prereg.PreregistrationError as error:
        blocked, blocked_reason = True, str(error)
    checks.append(
        check(
            "real_protocol_correctly_refuses_execution_today", blocked,
            f"require_frozen_protocol() on the real S44 protocol must raise while ready_to_execute "
            f"is False: {blocked_reason if blocked else 'did NOT raise -- GO-9 violation'}",
        )
    )

    positions = qneq0_prereg.read_xyz_positions_ang(qneq0_prereg.DEFAULT_POSITIONS_XYZ)
    directions = qneq0_prereg.build_candidate_directions(positions)
    frozen_sectors = protocol["phonon"]["candidate_sectors"]
    mismatches = [
        name for name, direction in directions.items()
        if name not in frozen_sectors or direction.direction_hash != frozen_sectors[name]["direction_hash"]
    ]
    checks.append(
        check(
            "rebuilt_directions_match_frozen_protocol_hashes", not mismatches,
            f"{len(directions)} directions rebuilt from the real (31,30) geometry; mismatches={mismatches}",
        )
    )
    checks.append(
        check(
            "frozen_q_candidates_match_module_constants",
            protocol["experiment"]["q_fractional_primary"] == list(qneq0_prereg.Q_PRIMARY["k"])
            and protocol["experiment"]["q_fractional_cross_check"] == list(qneq0_prereg.Q_CROSSCHECK["k"]),
            f"q_fractional_primary={protocol['experiment']['q_fractional_primary']!r} "
            f"q_fractional_cross_check={protocol['experiment']['q_fractional_cross_check']!r}",
        )
    )

    passed = all(entry["passed"] for entry in checks if entry["applicable"])
    return {"status": "PASS" if passed else "FAIL", "checks": checks}


# --------------------------------------------------------------------------- #
# Suite B: GO-5 and GO-8b are read live and combined as a strict conjunction
# --------------------------------------------------------------------------- #


def _qneq0_go9_gate(*, go5_status: str, go8b_status: str) -> bool:
    """GO-9 (q != 0 branch) may only become GO once GO-5 *and* GO-8b both
    independently read PASS. Neither substitutes for the other: GO-8b's own
    acceptance criterion requires graphene-K agreement (GO-5), and GO-5 does
    not by itself certify the MATBG-scale route mechanics GO-8b covers.
    """
    return go5_status == "PASS" and go8b_status == "GO-8b"


def _certify_go5_and_go8b_dependency() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    go5 = ck.go5_decision()
    go8b = go8b_cert.certify_qneq0_matbg_response_go8b()

    checks.append(
        check(
            "go8b_graphene_k_agreement_blocked_by_live_go5",
            go8b["graphene_k_agreement_suite"]["blocked_by"] == go5["go5"],
            f"go8b's own graphene_k_agreement_suite.blocked_by={go8b['graphene_k_agreement_suite']['blocked_by']!r} "
            f"must equal the live GO-5 decision ({go5['go5']!r}), proving the dependency is wired "
            "to the real gate, not a hardcoded label",
        )
    )

    truth_table = {
        (go5_pass, go8b_pass): _qneq0_go9_gate(
            go5_status="PASS" if go5_pass else "NO_GO", go8b_status="GO-8b" if go8b_pass else "NO-GO-8b"
        )
        for go5_pass in (True, False)
        for go8b_pass in (True, False)
    }
    expected_table = {(True, True): True, (True, False): False, (False, True): False, (False, False): False}
    checks.append(
        check(
            "go9_is_a_strict_conjunction_of_go5_and_go8b", truth_table == expected_table,
            f"truth_table[(go5_pass, go8b_pass) -> go9_pass] = {truth_table}; either gate alone "
            "failing must fail the whole branch, and neither may substitute for the other",
        )
    )

    full_ks_gates_open = _qneq0_go9_gate(go5_status=go5["go5"], go8b_status=go8b["go8b_status"])
    checks.append(
        check(
            "current_go5_and_go8b_status_correctly_yields_blocked",
            full_ks_gates_open is False,
            f"today go5={go5['go5']!r}, go8b_status={go8b['go8b_status']!r} -> "
            f"full_ks_gates_open={full_ks_gates_open}; the branch may not claim GO-9 until both "
            "flip independently",
        )
    )

    passed = all(entry["passed"] for entry in checks if entry["applicable"])
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "go5_decision": go5,
        "go8b_certification": go8b,
        "full_ks_gates_open": full_ks_gates_open,
    }


# --------------------------------------------------------------------------- #
# Suite C: route mechanics + cache/restart/GPU/sparse/scaling controls,
# reused verbatim from S40's own certifier -- not re-implemented.
# --------------------------------------------------------------------------- #


def _certify_route_mechanics_and_reproducibility() -> dict[str, Any]:
    return go8b_cert._certify_route_mechanics()


# --------------------------------------------------------------------------- #
# Suite D: a phase or basis-response defect must force NO-GO even when the
# raw D_H matrices involved are individually well-formed.
# --------------------------------------------------------------------------- #


def _certify_phase_and_basis_response_guard() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    table = g2m_kernel.kernel_table_graph2mat()
    correct = g2m_kernel.k_to_k_plus_q_kernel_graph2mat(table, _GUARD_K_PRIMITIVE, _GUARD_Q_PRIMITIVE)
    wrong_q = tuple(-float(value) for value in _GUARD_Q_PRIMITIVE)
    corrupted_phase_aware = g2m_kernel.phase_aware_kernel_graph2mat(wrong_q)
    corrupted = g2m_kernel.q_b_response_graph2mat(corrupted_phase_aware, _GUARD_K_PRIMITIVE)
    worst = float(np.abs(correct["D_H"] - corrupted["D_H"]).max())
    both_well_formed = bool(np.all(np.isfinite(correct["D_H"])) and np.all(np.isfinite(corrupted["D_H"])))
    checks.append(
        check(
            "phase_sign_error_is_detected_despite_well_formed_matrices",
            both_well_formed and worst > 1e-3,
            f"D_H(k+q,k) worst_abs_diff={worst:.3e} between the correct q and a sign-flipped q "
            f"(both matrices finite/well-formed: {both_well_formed}); a mismatch this large would "
            "fail qc_and_qb_agree_on_real_graph2mat (tolerance 1e-9) rather than silently pass a "
            "wrong D_H(q) that looks like a perfectly ordinary matrix",
            worst_abs_diff=worst,
        )
    )

    protocol = qneq0_prereg.load_protocol()
    basis_response_check = next(
        entry for entry in protocol["checks"] if entry["id"] == "qaware_basis_response_present_and_required"
    )
    checks.append(
        check(
            "qaware_basis_response_check_is_registered_and_unmet",
            "no such producer exists" in basis_response_check["pass_rule"],
            f"pass_rule={basis_response_check['pass_rule']!r}",
        )
    )
    checks.append(
        check(
            "basis_response_gap_alone_keeps_physics_review_blocked",
            protocol["physics_review"]["status"] == "BLOCKED" and protocol["ready_to_execute"] is False,
            f"even with route_mechanics_suite = PASS (S40, Suite C above), the frozen protocol's "
            f"own physics_review stays {protocol['physics_review']['status']!r} and "
            f"ready_to_execute={protocol['ready_to_execute']!r} solely because of the qaware "
            "basis-response gap",
        )
    )

    ladder = protocol["claim_ladder"]
    mechanics_rung = next(row for row in ladder if "mechanics_validated_no_full_ks_coupling" in row["claim"])
    physical_rung = next(row for row in ladder if "qneq0_epc_validated" in row["claim"])
    checks.append(
        check(
            "claim_ladder_ceiling_is_mechanics_only_while_basis_response_is_missing",
            "route-mechanics checks pass" in mechanics_rung["outcome"]
            and "q-aware basis response" in physical_rung["outcome"],
            f"mechanics_rung.outcome={mechanics_rung['outcome']!r} stays the ceiling until the "
            f"physical rung's own precondition ({physical_rung['outcome']!r}) is met",
        )
    )

    passed = all(entry["passed"] for entry in checks if entry["applicable"])
    return {"status": "PASS" if passed else "FAIL", "checks": checks}


# --------------------------------------------------------------------------- #
# Suite E: resource budget -- honestly blocked, no real GPU run exists yet
# --------------------------------------------------------------------------- #


def _certify_resource_budget() -> dict[str, Any]:
    margins = matbg_bench.resource_margins()
    reason = (
        "no execution stage for the q != 0 branch has ever been run against the real (31, 30) "
        "checkpoint (docs/epc_s44_matbg_qneq0_rigid_preregistration.md Sec. 4: even the "
        "mechanics-only kernel-table build was never measured at this system's real edge count, "
        "S37/S40 OOM'd before any real MATBG edge count was recorded); the roadmap's own compute "
        "policy requires a one-direction preflight before a batch run, not a resource claim "
        "fabricated from idle machine headroom"
    )
    return {
        "status": "NOT_RUN",
        "blocked_by": "S44 Sec. 4 (no measured production run of the q != 0 branch)",
        "checks": [
            check(
                "campaign_peak_resource_usage_within_roadmap_margins", True, reason, applicable=False,
                requirement="roadmap Section X preflight (28 GiB VRAM / 62 GB RAM / 12% free disk)",
            )
        ],
        "live_machine_headroom_snapshot": margins,
    }


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def certify_rigid_qneq0_campaign_go9(tmp_dir: Path) -> dict[str, Any]:
    gate_authorization = _certify_gate_authorization(tmp_dir)
    go5_go8b_dependency = _certify_go5_and_go8b_dependency()
    route_mechanics_and_reproducibility = _certify_route_mechanics_and_reproducibility()
    phase_and_basis_response_guard = _certify_phase_and_basis_response_guard()
    resource_budget = _certify_resource_budget()

    suites = {
        "gate_authorization_suite": gate_authorization,
        "go5_and_go8b_dependency_suite": go5_go8b_dependency,
        "route_mechanics_and_reproducibility_suite": route_mechanics_and_reproducibility,
        "phase_and_basis_response_guard_suite": phase_and_basis_response_guard,
        "resource_budget_suite": resource_budget,
    }
    mechanism_ok = all(suite["status"] == "PASS" for suite in suites.values())
    go9 = mechanism_ok and go5_go8b_dependency["full_ks_gates_open"]
    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "memo": MEMO,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **suites,
        "go9_status": "GO-9" if go9 else "NO-GO-9",
        "rigid_qneq0_campaign_certified": bool(go9),
        "final_publication_label": FINAL_PUBLICATION_LABEL,
        "forbidden_publication_labels": list(FORBIDDEN_PUBLICATION_LABELS),
    }


def require_go9_certified_qneq0_rigid_campaign(certification: dict) -> dict:
    if certification.get("gate") != GATE or certification.get("go9_status") != "GO-9":
        raise RigidQneq0CampaignCertificationError(
            f"GO-9 (q != 0 branch) is not certified ({certification.get('go9_status')!r}): no "
            f"non-Gamma rigid-MATBG result may be published as {FINAL_PUBLICATION_LABEL!r}, and "
            f"{FORBIDDEN_PUBLICATION_LABELS!r} stay forbidden regardless"
        )
    return certification


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    import tempfile

    with tempfile.TemporaryDirectory(prefix="s45_rigid_qneq0_certify_") as tmpdir:
        certification = certify_rigid_qneq0_campaign_go9(Path(tmpdir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / REPORT_NAME
    output.write_text(json.dumps(_json_safe(certification), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"S45 rigid q!=0 campaign {GATE} certification: {certification['go9_status']}  ->  {output}")
    for suite_name in (
        "gate_authorization_suite", "go5_and_go8b_dependency_suite", "route_mechanics_and_reproducibility_suite",
        "phase_and_basis_response_guard_suite", "resource_budget_suite",
    ):
        suite = certification[suite_name]
        print(f"  {suite_name}: {suite['status']}")
        for entry in suite["checks"]:
            state = "n/a " if not entry["applicable"] else ("ok  " if entry["passed"] else "FAIL")
            print(f"    [{state}] {entry['check']}: {entry['detail']}")
    return 0 if certification["go9_status"] == "GO-9" else 1


if __name__ == "__main__":
    raise SystemExit(main())
