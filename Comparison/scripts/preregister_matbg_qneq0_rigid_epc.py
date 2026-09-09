#!/usr/bin/env python3
"""E-F_001-S44: pre-register the rigid-MATBG q != 0 (non-Gamma) EPC branch.

The roadmap scopes this ticket explicitly as conditional -- "Ejecutar, solo
si se desea una rama no-Gamma, modos fisicos MATBG mediante el provider
GO-8a y la respuesta q-aware GO-8b" -- and D08B pins its hard dependencies:
GO-5, GO-8a, GO-8 and GO-8b. Today:

- GO-8a is ``NO-GO-8a`` (S39, ``certify_matbg_phonon_provider.py``): no
  scalable, validated MATBG :class:`phonon_provider.PhononProvider` exists.
- GO-8b is ``NO-GO-8b`` (S40, ``certify_qneq0_matbg_response.py``): the Q-C
  kernel-table / Q-B phase-aware mechanics are proven on the real production
  JVP, but ``graphene_k_agreement_suite`` stays ``NOT_RUN`` because GO-5
  itself is ``NO_GO`` (blocked by GO-4/C14C's intra-atomic basis-response
  term).
- GO-9's own Gamma-branch campaign certifier (S43,
  ``certify_rigid_gamma_campaign.py``) is also ``NO-GO-9``
  (``resource_budget_suite`` stays ``NOT_RUN``: no GPU run of
  ``epc_gamma_rigid()`` against the real 44 656-orbital checkpoint has ever
  been measured).

So this ticket cannot "produce" a physical q != 0 mode today -- doing so
would mean either (a) reusing the Gamma JVP as a silent q != 0 substitute,
the exact failure mode GO-8b's own guard (``GammaJvpReuseError``,
S40 Sec. 7) exists to reject, or (b) fabricating a phonon that no certified
:class:`PhononProvider` produced (roadmap B7). What this ticket does instead
is what S41 did for the Gamma branch before GO-8a existed: freeze the
non-Gamma protocol -- candidate q vectors, the k -> k+q electronic pairing,
which route (Q-C primary / Q-B cross-check, never plain Gamma JVP) any future
execution must use, the tolerances, the claim ladder -- so a future
execution step, once GO-8a *and* GO-8b both close, has nothing left to
decide after seeing ``g``. Every helper below is reused from
:mod:`preregister_matbg_gamma_rigid_epc` (S41) and
:mod:`preregister_graphene_gamma_epc` (C19/C20) rather than re-implemented;
this module only adds what is actually different about q != 0: two more
required gates, the q candidates, the q-aware reference route, and the
route_mechanics-style checks S40 already exercised on the real kernel.

Units: Ang for displacements/geometry, eV for energies, fractional
reciprocal-lattice coordinates for q/k (``epc_fourier.CONVENTION_ID``, S29).
Narrative: ``docs/epc_s44_matbg_qneq0_rigid_preregistration.md``.
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

import epc_fourier  # noqa: E402
import preregister_matbg_gamma_rigid_epc as gamma_prereg  # noqa: E402

prereg = gamma_prereg.prereg
go8a_cert = gamma_prereg.go8a_cert
go8b_cert = gamma_prereg.go8b_cert
fdp = gamma_prereg.fdp

PreregistrationError = prereg.PreregistrationError

SCHEMA = "epc_preregistration_v1"
PROTOCOL_ID = "matbg_rigid_qneq0_epc_v1"
TICKET = "E-F_001-S44"
GATE = "GO-9 pre-registration (q != 0 branch); execution additionally requires GO-1..GO-6, GO-8a and GO-8b"

TARGET_FDF = gamma_prereg.TARGET_FDF
TARGET_ATOM_COUNT = gamma_prereg.TARGET_ATOM_COUNT
TARGET_ORBITAL_COUNT = gamma_prereg.TARGET_ORBITAL_COUNT
DEFAULT_POSITIONS_XYZ = gamma_prereg.DEFAULT_POSITIONS_XYZ
DEFAULT_NEUTRALITY_ESTIMATE = gamma_prereg.DEFAULT_NEUTRALITY_ESTIMATE
DEFAULT_MATBG_CERTIFICATION_DIR = gamma_prereg.DEFAULT_MATBG_CERTIFICATION_DIR
DEFAULT_BASIS_RESPONSE = gamma_prereg.DEFAULT_BASIS_RESPONSE
DEFAULT_GO6_REPORT = gamma_prereg.DEFAULT_GO6_REPORT
DEFAULT_GAMMA_PROTOCOL = gamma_prereg.DEFAULT_OUTPUT_DIR / gamma_prereg.PROTOCOL_NAME

DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/preregistration/matbg"
PROTOCOL_NAME = "epc_matbg_rigid_qneq0_protocol.json"
DEFAULT_RESULT_ROOT = REPO_ROOT / "Comparison/results/epc/qneq0_epc/matbg"

read_xyz_positions_ang = gamma_prereg.read_xyz_positions_ang
geometry_report = gamma_prereg.geometry_report
build_candidate_directions = gamma_prereg.build_candidate_directions
occupation_report = gamma_prereg.occupation_report

# --------------------------------------------------------------------------- #
# What is reused verbatim from the Gamma protocol (S41): electronic k
# candidates, window/occupation contract, delta sweep, resource margins, the
# three synthetic-displacement direction families. None of that changes with
# q -- only *what q is added on top of k* is new here.
# --------------------------------------------------------------------------- #

K_CANDIDATES = gamma_prereg.K_CANDIDATES
K_CALIBRATION = gamma_prereg.K_CALIBRATION
RESULT_DIRECTION_NAMES = gamma_prereg.RESULT_DIRECTION_NAMES
VALIDATION_DIRECTION_NAMES = gamma_prereg.VALIDATION_DIRECTION_NAMES
CALIBRATION_DIRECTION_NAMES = gamma_prereg.CALIBRATION_DIRECTION_NAMES
WINDOW_STATES_BELOW = gamma_prereg.WINDOW_STATES_BELOW
WINDOW_STATES_ABOVE = gamma_prereg.WINDOW_STATES_ABOVE
MU_CONVENTION = gamma_prereg.MU_CONVENTION
MIN_WINDOW_GAP_EV = gamma_prereg.MIN_WINDOW_GAP_EV
DEGENERACY_TOL_EV = gamma_prereg.DEGENERACY_TOL_EV
DELTA_SWEEP_ANG = gamma_prereg.DELTA_SWEEP_ANG
MIN_SIGNAL_TO_FLOOR = gamma_prereg.MIN_SIGNAL_TO_FLOOR
BACKEND_NOISE_MARGIN = gamma_prereg.BACKEND_NOISE_MARGIN
VRAM_MARGIN_GIB = gamma_prereg.VRAM_MARGIN_GIB
RAM_MARGIN_GIB = gamma_prereg.RAM_MARGIN_GIB
BASIS_RESPONSE_CONTRACT = dict(gamma_prereg.BASIS_RESPONSE_CONTRACT)

# --------------------------------------------------------------------------- #
# What is new: the phonon-momentum q candidates. [DESIGN, unverified]: same
# moire-BZ fractional convention as gamma_prereg.K_DEGENERATE/K_CALIBRATION,
# same "moire_hexagonal_convention_checked" caveat -- not re-derived here,
# reused with the caveat intact. K (Kohn-anomaly analogue, Piscanec et al.
# 2004; low-energy moire modes near non-Gamma points, Lu et al. 2022) is the
# literature-motivated primary candidate for a non-Gamma branch; a second,
# generic point is kept only as a mechanics cross-check (Q-C vs Q-B agree
# identically at any q by construction, S40), never as a second physical
# result.
Q_PRIMARY = {**gamma_prereg.K_DEGENERATE, "role": "primary_non_gamma_candidate_pending_go8a_go8b"}
Q_CROSSCHECK = {**K_CANDIDATES[0], "label": "q_generic_1", "role": "route_mechanics_cross_check_only"}

FORBIDDEN_TAU_MODEL_SOURCES = gamma_prereg.FORBIDDEN_TAU_MODEL_SOURCES

SPLITS: dict[str, dict[str, Any]] = {
    "calibration": {
        "purpose": "fix FD floors and the delta plateau once a real measurement exists. May be "
        "inspected freely; nothing measured here is a result.",
        "directions": CALIBRATION_DIRECTION_NAMES,
        "k_labels": ("gamma",),
        "q_labels": ("q_generic_1",),
        "fc_ranges": (),
    },
    "validation": {
        "purpose": "route-mechanics cross-check: Q-C kernel-table contraction vs Q-B per-q "
        "phase-aware response must agree at machine precision on the real Graph2Mat JVP (S40's "
        "own route_mechanics_suite), independent of any physical claim.",
        "directions": VALIDATION_DIRECTION_NAMES,
        "k_labels": ("k_generic_2",),
        "q_labels": ("q_generic_1",),
        "fc_ranges": (),
    },
    "result": {
        "purpose": "the pre-registered non-Gamma experiment: the K-point candidate, target of a "
        "physical validation once GO-8a and GO-8b both close. Evaluated last, never used to set "
        "a tolerance.",
        "directions": RESULT_DIRECTION_NAMES,
        "k_labels": ("k_generic_1",),
        "q_labels": ("K",),
        "fc_ranges": (),
    },
}

# q-aware production route: Q-C (primary) / Q-B (cross-check) through the
# real production JVP -- never the plain Gamma-periodic JVP graph2mat_jvp
# path S41's own protocol used, which is exactly the substitution GO-8b's
# GammaJvpReuseError guard exists to reject (S40 Sec. 7).
REFERENCE_PATHS: tuple[dict[str, Any], ...] = (
    {
        "path": "graph2mat_qc_kernel_table",
        "electronic_derivative_backend": "graph2mat",
        "derivative_method": "qc_kernel_table",
        "artifact": "q-independent real-JVP kernel table "
        "(epc_qb_qc_graph2mat_kernel.kernel_table_graph2mat), contracted per q via "
        "k_to_k_plus_q_kernel_graph2mat -- exactly 2 real JVP calls total, independent of q "
        "(S40 Sec. 4 item 5/7)",
        "internal_check": "raises GammaJvpReuseError if fed a plain q=0 directional-JVP payload "
        "instead of a kernel_table_graph2mat one (S40 Sec. 7)",
        "new_runs_required": "one kernel-table build (2 JVP calls) against the existing "
        "tbg_pure_graph2mat checkpoint, not retrained; reused for every q afterwards",
    },
    {
        "path": "graph2mat_qb_phase_aware",
        "electronic_derivative_backend": "graph2mat",
        "derivative_method": "qb_phase_aware_jvp",
        "artifact": "per-q phase-aware directional response "
        "(epc_qb_qc_graph2mat_kernel.phase_aware_kernel_graph2mat / q_b_response_graph2mat) -- "
        "2 real JVP calls per q",
        "internal_check": "agrees with graph2mat_qc_kernel_table to machine precision on the "
        "real production JVP (S40: worst_abs_diff=1.11e-16)",
        "new_runs_required": "one phase-aware build per q evaluated (2 JVP calls each)",
    },
)
SIESTA_REFERENCE_EXCLUDED = (
    gamma_prereg.SIESTA_REFERENCE_EXCLUDED
    + " A q != 0 SIESTA reference would additionally need a commensurate supercell containing "
    "that q as a reciprocal-lattice vector of the auxiliary cell (roadmap Fase 7 route Q-A), "
    "strictly more expensive than the already-intractable q=0 FC campaign S38 measured -- so no "
    "SIESTA path is proposed for this branch either."
)

QAWARE_BASIS_RESPONSE_GAP = (
    "no q-aware basis-response producer exists in this repository: epc_basis_response.py (C14B) "
    "builds PAO-covariant response only from a real-space SIESTA FC.Save.dHS campaign object "
    "(campaign.certified_pairs()/dhsdr_path), which S38 already measured intractable at MATBG "
    "scale even at q=0. Roadmap GO-8b's own acceptance criterion requires this response to be "
    "'tambien q-aware' with the same Fourier/R-vector/atomic-phase conventions as D_H(q) -- that "
    "producer does not exist yet, independent of whether C14C's intra-atomic term ever resolves."
)

TOLERANCES: dict[str, dict[str, Any]] = {
    "tau_num": dict(gamma_prereg.TOLERANCES["tau_num"]),
    "tau_backend": {
        **gamma_prereg.TOLERANCES["tau_backend"],
        "what": "Q-C kernel-table contraction vs Q-B phase-aware response of the same Graph2Mat "
        "model, at the same q (route_mechanics_suite, not JVP-vs-frozen as in the Gamma branch)",
        "definition": "tau_backend = || C_W^dag (Delta_QC(q) - Delta_QB(q)) C_W ||_F",
    },
    "tau_model": dict(gamma_prereg.TOLERANCES["tau_model"]),
}

CHECKS: tuple[dict[str, Any], ...] = (
    {
        "id": "qc_and_qb_agree_on_real_graph2mat",
        "stage": "derivative",
        "applies_to": ("graph2mat_qc_kernel_table", "graph2mat_qb_phase_aware"),
        "pass_rule": "worst_abs_diff at machine precision (S40 route_mechanics_suite: 1.11e-16)",
    },
    {
        "id": "q_and_minus_q",
        "stage": "derivative",
        "applies_to": ("graph2mat_qc_kernel_table", "graph2mat_qb_phase_aware"),
        "pass_rule": "epc_fourier.minus_q_matrix round-trip at machine precision",
    },
    {
        "id": "cell_origin_shift",
        "stage": "derivative",
        "applies_to": ("graph2mat_qc_kernel_table", "graph2mat_qb_phase_aware"),
        "pass_rule": "invariant to a shift of the periodic-cell origin, at machine precision",
    },
    {
        "id": "atom_across_periodic_boundary",
        "stage": "derivative",
        "applies_to": ("graph2mat_qc_kernel_table", "graph2mat_qb_phase_aware"),
        "pass_rule": "re-expressing an atom in a different periodic image leaves the result "
        "unchanged (S40 Sec. 3), at machine precision",
    },
    {
        "id": "alternative_atomic_phase_convention",
        "stage": "derivative",
        "applies_to": ("graph2mat_qc_kernel_table", "graph2mat_qb_phase_aware"),
        "pass_rule": "atom-gauge vs bond-gauge atomic phase convention agree once converted "
        "through epc_fourier's explicit unitary (roadmap Section XV closing paragraph)",
    },
    {
        "id": "topology_margin_preserved",
        "stage": "derivative",
        "applies_to": ("graph2mat_qc_kernel_table", "graph2mat_qb_phase_aware"),
        "pass_rule": "fd_perturbation_space.topology_margin: 0 crossing pairs under the delta "
        "sweep used to build the kernel table",
    },
    {
        "id": "gamma_jvp_reuse_rejected",
        "stage": "provenance",
        "applies_to": ("all_qneq0_artifacts",),
        "pass_rule": "k_to_k_plus_q_kernel_graph2mat / q_b_response_graph2mat raise "
        "GammaJvpReuseError for any q != 0 query against a payload that is not a real "
        "kernel_table_graph2mat / phase_aware_kernel_graph2mat schema (S40 Sec. 7); this "
        "protocol never authorises a plain q=0 directional-JVP artifact for a q != 0 claim",
    },
    {
        "id": "route_agrees_with_q_a_on_real_graphene_k",
        "stage": "cross_system_transfer",
        "applies_to": ("graph2mat_qc_kernel_table", "graph2mat_qb_phase_aware"),
        "pass_rule": "must have passed on graphene K (GO-5) before this system's result is used "
        "for anything beyond route mechanics; status today NOT_RUN, blocked by GO-5=NO_GO",
    },
    {
        "id": "qaware_basis_response_present_and_required",
        "stage": "epc",
        "applies_to": ("graph2mat_qc_kernel_table", "graph2mat_qb_phase_aware"),
        "pass_rule": "PAO-covariant response(q) is built only from a basis-response artifact that is itself "
        "q-aware under the same Fourier/R-vector/atomic-phase conventions; no such producer "
        "exists (see QAWARE_BASIS_RESPONSE_GAP) so this check cannot pass today",
    },
    {
        "id": "no_forbidden_labels",
        "stage": "provenance",
        "applies_to": ("all_result_artifacts",),
        "pass_rule": "every artifact this protocol's execution writes stays tagged "
        "shared.artifact_signature.SYNTHETIC_TEST_DISPLACEMENT; PHYSICAL_PHONON and any "
        "'phonon_eigenmode'/'g_mn_nu' label require a GO-8a-certified PhononProvider payload, "
        "never this protocol's own candidate q",
    },
)

CLAIM_LADDER: tuple[dict[str, Any], ...] = (
    {
        "outcome": "GO-8a becomes GO-8a, GO-8b becomes GO-8b, C14C becomes GO and every check "
        "passes (including a q-aware basis response and graphene-K agreement)",
        "claim": "rigid_tbg_qneq0_epc_validated (GO-9, non-Gamma branch); Graph2Mat g(q) is "
        "quantitative for this mode/k/q on the rigid (31, 30) cell. Publication label: "
        "rigid_tbg_model -- never quantitative_relaxed_matbg_prediction",
    },
    {
        "outcome": "route-mechanics checks pass (qc_and_qb_agree_on_real_graph2mat, q_and_minus_q, "
        "cell_origin_shift, atom_across_periodic_boundary, alternative_atomic_phase_convention, "
        "topology_margin_preserved, gamma_jvp_reuse_rejected) but GO-8a and/or GO-8b are still "
        "NO-GO",
        "claim": "graph2mat_qneq0_mechanics_validated_no_full_ks_coupling; identical ceiling to the "
        "Gamma branch's graph2mat_derivative_validated_no_full_ks_coupling, applied to the q != 0 "
        "route instead of the plain JVP -- no g_mn_nu may be published",
    },
    {
        "outcome": "any route-mechanics check fails",
        "claim": "NO_GO, attributed to the failing layer (kernel construction, phase convention, "
        "topology); never to the model",
    },
    {
        "outcome": "current state: GO-8a NO-GO-8a, GO-8b NO-GO-8b, GO-5 NO_GO, no q-aware "
        "basis-response producer exists",
        "claim": "no execution is authorised at all for this branch; this pre-registration "
        "exists to freeze what a future execution must not be allowed to choose after seeing g, "
        "not to run anything today",
    },
)


# --------------------------------------------------------------------------- #
# Preconditions: GO-8a and GO-8b are BOTH required here (unlike the Gamma
# protocol, S41, where GO-8b was informational-only). GO-9's own Gamma-branch
# campaign certification (S43) is cited informationally: the roadmap
# recommends running D08A before D08B but does not make it a hard dependency
# if every other gate already passed (roadmap Fase 10d).
# --------------------------------------------------------------------------- #


def preconditions(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    go8a = prereg._read_json(paths["go8a"])
    go8b = prereg._read_json(paths["go8b"])
    c14c = prereg._read_json(paths["c14c"])
    go6 = prereg._read_json(paths["go6"])
    go9 = prereg._read_json(paths["go9"])

    rows = [
        prereg._precondition(
            "GO-8a MATBG PhononProvider", paths["go8a"], go8a,
            verdict="PASS" if (go8a or {}).get("go8a_status") == "GO-8a" else (go8a or {}).get("go8a_status"),
            required=True,
        ),
        prereg._precondition(
            "GO-8b MATBG q!=0 response", paths["go8b"], go8b,
            verdict="PASS" if (go8b or {}).get("go8b_status") == "GO-8b" else (go8b or {}).get("go8b_status"),
            required=True,
        ),
        prereg._precondition(
            "C14C basis response (global formalism gate)", paths["c14c"], c14c,
            verdict=(c14c or {}).get("verdict"), required=True,
        ),
        prereg._precondition(
            "GO-6 subspace metrics (global gate)", paths["go6"], go6,
            verdict=(go6 or {}).get("verdict"), required=True,
        ),
        prereg._precondition(
            "GO-9 rigid-Gamma campaign certification", paths["go9"], go9,
            verdict="PASS" if (go9 or {}).get("go9_status") == "GO-9" else (go9 or {}).get("go9_status"),
            required=False,
        ),
    ]
    rows[-1]["note"] = "informational only: roadmap Fase 10d recommends but does not require a " \
        "passed Gamma-branch (D08A) campaign before a q != 0 (D08B) one"
    if go8a is not None:
        rows[0]["physical_phonon_produced"] = bool(go8a.get("physical_phonon_produced"))
    return rows


def physics_review(
    *, geometry: Mapping[str, Any], gates: list[dict[str, Any]], q_selection_measured: bool
) -> dict[str, Any]:
    gate_status = {row["name"]: row["status"] for row in gates}
    required_gates_pass = all(
        gate_status.get(name) == "PASS"
        for name in (
            "GO-8a MATBG PhononProvider",
            "GO-8b MATBG q!=0 response",
            "C14C basis response (global formalism gate)",
            "GO-6 subspace metrics (global gate)",
        )
    )
    items = [
        {
            "item": "candidate_q",
            "decision": "approved",
            "evidence": {"primary": Q_PRIMARY, "cross_check_only": Q_CROSSCHECK},
            "basis": "K is the literature-motivated non-Gamma candidate (Piscanec et al. 2004 "
            "Kohn-anomaly analogue; Lu et al. 2022 low-energy moire phonon evolution with twist); "
            "approval here is of the *selection*, not of any physical mode -- see qaware "
            "basis-response and GO-8a/GO-8b below for why no g(q) may yet be computed",
        },
        {
            "item": "q_selection",
            "decision": "approved" if q_selection_measured else "blocked",
            "evidence": {"rule_frozen": True, "candidates_frozen": True, "route_measured": q_selection_measured},
            "basis": "the candidate q list is frozen by value; contracting the frozen "
            "graph2mat_qc_kernel_table at a live q is deferred to execution, gated by GO-8a/GO-8b "
            "and by the same resource preflight the Gamma branch requires (roadmap Section X)",
        },
        {
            "item": "references",
            "decision": "approved" if geometry["atom_count_matches_target"] else "blocked",
            "evidence": {"paths": [row["path"] for row in REFERENCE_PATHS], "geometry": dict(geometry)},
            "basis": "Q-C (primary) / Q-B (cross-check) through the real production JVP only; "
            "never the plain Gamma-periodic JVP the Gamma protocol used, and never a SIESTA path "
            "(SIESTA_REFERENCE_EXCLUDED, strictly more intractable than the already-excluded q=0 case)",
        },
        {
            "item": "qaware_basis_response",
            "decision": "blocked",
            "evidence": {"gap": QAWARE_BASIS_RESPONSE_GAP},
            "basis": "GO-8b's own acceptance criterion requires the basis response to be q-aware "
            "under the same conventions as D_H(q); no producer exists for this repository at any "
            "system today, so this item is blocked by design, not by omission",
        },
        {
            "item": "gates",
            "decision": "approved" if required_gates_pass else "blocked",
            "evidence": gate_status,
            "basis": "GO-8a, GO-8b, C14C and GO-6 are required before any PAO-projected coupling(q) on this "
            "or any system; their current status is read live from the certification artifacts, "
            "not asserted",
        },
    ]
    return {
        "reviewer": "automated_precondition_review (evidence-bound)",
        "items": items,
        "status": "APPROVED" if all(row["decision"] == "approved" for row in items) else "BLOCKED",
        "human_sign_off": None,
    }


def build_protocol(
    *,
    target_fdf: Path = TARGET_FDF,
    positions_xyz: Path = DEFAULT_POSITIONS_XYZ,
    neutrality_estimate: Path = DEFAULT_NEUTRALITY_ESTIMATE,
    certification_dir: Path = DEFAULT_MATBG_CERTIFICATION_DIR,
    basis_response_report: Path = DEFAULT_BASIS_RESPONSE,
    go6_report: Path = DEFAULT_GO6_REPORT,
    gamma_campaign_certification: Path = DEFAULT_MATBG_CERTIFICATION_DIR / "s43_rigid_gamma_campaign_go9_certification.json",
) -> dict[str, Any]:
    prereg.assert_splits_disjoint(SPLITS)

    geometry = geometry_report(positions_xyz, target_fdf)
    if not geometry["atom_count_matches_target"]:
        raise PreregistrationError(
            f"{positions_xyz} has {geometry['atom_count']} atoms, target is {TARGET_ATOM_COUNT}"
        )
    positions = read_xyz_positions_ang(positions_xyz)
    directions = build_candidate_directions(positions)
    occupations = occupation_report(neutrality_estimate)

    gate_paths = {
        "go8a": certification_dir / go8a_cert.REPORT_NAME,
        "go8b": certification_dir / go8b_cert.REPORT_NAME,
        "c14c": basis_response_report,
        "go6": go6_report,
        "go9": gamma_campaign_certification,
    }
    gates = preconditions(gate_paths)
    review = physics_review(geometry=geometry, gates=gates, q_selection_measured=False)

    blocking = [
        f"{row['name']}: {row['status']}" for row in gates if row["required_before_execution"] and row["status"] != "PASS"
    ]
    blocking.append("q_selection: no live k -> k+q electronic contraction measured yet at any "
                     "candidate q (rule and candidates are frozen; the measurement itself is "
                     "deferred to a resource-preflight-gated execution step)")
    blocking.append("qaware_basis_response_producer_missing: " + QAWARE_BASIS_RESPONSE_GAP)
    blocking.append("moire_hexagonal_convention_checked: the K fractional label is the graphene "
                     "convention carried over, not yet verified against this structure's own "
                     "reciprocal lattice (same open item as the Gamma protocol, S41)")

    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "ticket": TICKET,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_content_sha256": None,
        "experiment": {
            "system": "rigid magic-angle TBG (31, 30), materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf",
            "q_fractional_primary": list(Q_PRIMARY["k"]),
            "q_label_primary": Q_PRIMARY["label"],
            "q_fractional_cross_check": list(Q_CROSSCHECK["k"]),
            "q_convention": epc_fourier.CONVENTION_ID,
            "k_plus_q_equals_k": False,
            "geometry": geometry,
            "publication_label_if_validated": "rigid_tbg_model",
            "forbidden_publication_labels": ["quantitative_relaxed_matbg_prediction"],
        },
        "phonon": {
            "status": "no PhononProvider certified for this geometry (GO-8a: NO-GO-8a as of S39); "
            "no q-aware response certified either (GO-8b: NO-GO-8b as of S40)",
            "candidate_sectors": {
                name: directions[name].to_dict()
                for name in (*RESULT_DIRECTION_NAMES, *VALIDATION_DIRECTION_NAMES, *CALIBRATION_DIRECTION_NAMES)
            },
            "candidate_q": {"primary": Q_PRIMARY, "cross_check_only": Q_CROSSCHECK},
            "artifact_kind_of_candidates": "synthetic_test_displacement",
            "artifact_kind_forbidden_until_go8a_and_go8b": "physical_phonon",
        },
        "electronic": {
            "window": {
                "states_below": WINDOW_STATES_BELOW,
                "states_above": WINDOW_STATES_ABOVE,
                "rule": "identical to the Gamma protocol's window rule, applied independently at "
                "k and at k+q",
            },
            "occupations": occupations,
            "degeneracy_tol_ev": DEGENERACY_TOL_EV,
            "min_window_gap_ev": MIN_WINDOW_GAP_EV,
            "k_candidates": list(K_CANDIDATES),
            "k_calibration": list(K_CALIBRATION),
            "k_selection_rule": "identical rule to preregister_graphene_gamma_epc.select_result_k "
            "and to the Gamma protocol, applied independently to the measured spectrum at k and at "
            "k+q (epc_fourier.k_plus_q)",
            "k_selection_status": "pending_measurement",
            "subspace_machinery": "Comparison/scripts/epc_subspaces.py (GO-6): principal angles, "
            "singular values and projector distances, evaluated at both k and k+q windows",
        },
        "references": {
            "paths": list(REFERENCE_PATHS),
            "siesta_reference_excluded_reason": SIESTA_REFERENCE_EXCLUDED,
            "basis_response": {**BASIS_RESPONSE_CONTRACT, "must_be_q_aware": True, "gap": QAWARE_BASIS_RESPONSE_GAP},
            "formalism_id": gamma_prereg.BASIS_RESPONSE_CONTRACT["formalism_id"],
        },
        "splits": {
            **{name: {**split, "directions": list(split["directions"]), "k_labels": list(split["k_labels"]),
                      "q_labels": list(split["q_labels"]), "fc_ranges": list(split["fc_ranges"])}
               for name, split in SPLITS.items()},
            "disjoint_in": "(direction, k, q) triples and directions",
        },
        "perturbation": {
            "normalization": fdp.NORMALIZATION,
            "method": "central",
            "delta_sweep_ang": list(DELTA_SWEEP_ANG),
        },
        "tolerances": {
            **TOLERANCES,
            "constants": {
                "MIN_SIGNAL_TO_FLOOR": MIN_SIGNAL_TO_FLOOR,
                "BACKEND_NOISE_MARGIN": BACKEND_NOISE_MARGIN,
            },
            "resource_margins": {"VRAM_MARGIN_GIB": VRAM_MARGIN_GIB, "RAM_MARGIN_GIB": RAM_MARGIN_GIB},
        },
        "checks": list(CHECKS),
        "claim_ladder": list(CLAIM_LADDER),
        "preconditions": gates,
        "physics_review": review,
        "blocking": blocking,
        "ready_to_execute": not blocking and review["status"] == "APPROVED",
        "result_root": str(DEFAULT_RESULT_ROOT),
    }
    protocol["frozen_content_sha256"] = prereg.freeze_hash(protocol)
    return prereg._json_safe(protocol)


def load_protocol(path: Path = DEFAULT_OUTPUT_DIR / PROTOCOL_NAME) -> dict[str, Any]:
    payload = prereg._read_json(Path(path))
    if payload is None:
        raise PreregistrationError(f"no pre-registered protocol at {path}")
    return payload


def verify(
    path: Path = DEFAULT_OUTPUT_DIR / PROTOCOL_NAME, *, result_root: Path = DEFAULT_RESULT_ROOT
) -> dict[str, Any]:
    protocol = load_protocol(path)
    recomputed = prereg.freeze_hash(protocol)
    stored = protocol.get("frozen_content_sha256")
    results = []
    for report in sorted(Path(result_root).rglob("*.json")) if Path(result_root).is_dir() else []:
        payload = prereg._read_json(report) or {}
        cited = payload.get("preregistration_sha256")
        results.append({"report": str(report), "preregistration_sha256": cited, "cites_this_protocol": cited == stored})
    uncited = [row for row in results if not row["cites_this_protocol"]]
    return {
        "protocol": str(path),
        "protocol_id": protocol.get("protocol_id"),
        "frozen_content_sha256": stored,
        "recomputed_sha256": recomputed,
        "intact": stored == recomputed,
        "results_checked": results,
        "results_not_citing_the_protocol": [row["report"] for row in uncited],
        "verified": stored == recomputed and not uncited,
    }


def require_frozen_protocol(path: Path = DEFAULT_OUTPUT_DIR / PROTOCOL_NAME) -> dict[str, Any]:
    """Gate for a future execution step: refuse to compute anything outside this protocol."""
    protocol = load_protocol(path)
    if prereg.freeze_hash(protocol) != protocol.get("frozen_content_sha256"):
        raise PreregistrationError(f"{path} has been edited since it was frozen")
    if not protocol.get("ready_to_execute"):
        raise PreregistrationError(
            f"the protocol is frozen but its preconditions are not met: {protocol.get('blocking')}"
        )
    return protocol


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-fdf", type=Path, default=TARGET_FDF)
    parser.add_argument("--positions-xyz", type=Path, default=DEFAULT_POSITIONS_XYZ)
    parser.add_argument("--neutrality-estimate", type=Path, default=DEFAULT_NEUTRALITY_ESTIMATE)
    parser.add_argument("--certification-dir", type=Path, default=DEFAULT_MATBG_CERTIFICATION_DIR)
    parser.add_argument("--basis-response-report", type=Path, default=DEFAULT_BASIS_RESPONSE)
    parser.add_argument("--go6-report", type=Path, default=DEFAULT_GO6_REPORT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--sign-off", default=None, metavar="NAME")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    output = args.output_dir / PROTOCOL_NAME

    if args.verify:
        report = verify(output)
        print(f"pre-registration {report['protocol_id']}: "
              f"{'INTACT' if report['intact'] else 'TAMPERED'}  {report['frozen_content_sha256']}")
        for row in report["results_not_citing_the_protocol"]:
            print(f"  result does not cite the protocol: {row}")
        return 0 if report["verified"] else 1

    if args.sign_off:
        protocol = load_protocol(output)
        protocol["physics_review"]["human_sign_off"] = {
            "reviewer": str(args.sign_off),
            "signed_at": datetime.now(timezone.utc).isoformat(),
            "signed_content_sha256": protocol["frozen_content_sha256"],
        }
        protocol["frozen_content_sha256"] = prereg.freeze_hash(protocol)
    else:
        protocol = build_protocol(
            target_fdf=args.target_fdf,
            positions_xyz=args.positions_xyz,
            neutrality_estimate=args.neutrality_estimate,
            certification_dir=args.certification_dir,
            basis_response_report=args.basis_response_report,
            go6_report=args.go6_report,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")

    print(f"pre-registration {protocol['protocol_id']} -> {output}")
    print(f"  frozen_content_sha256 : {protocol['frozen_content_sha256']}")
    print(f"  candidate q           : {protocol['phonon']['candidate_q']['primary']['label']}")
    print(f"  physics review        : {protocol['physics_review']['status']}")
    print(f"  ready to execute      : {protocol['ready_to_execute']}")
    for reason in protocol["blocking"]:
        print(f"    blocked by: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
