#!/usr/bin/env python3
"""E-F_001-S50: pre-register the EPC protocol for the relaxed magic-angle TBG target (Fase 11).

Fase 11 ("Relaxed magic-angle TBG") sits behind GO-10, and two upstream tickets already
answer, as data, whether anything relaxed exists to measure:

- ``decide_matbg_relaxation_provider`` (S47): ``NO-GO-10`` -- no relaxation provider in this
  environment is defensible for the production ``(31, 30)`` cell. ``relaxed_geometry_produced``
  is hard-coded ``False``. No relaxed positions file exists anywhere in this repository.
- ``certify_relaxed_geometry_transferability`` (S48): a *local-stacking* bound (AA/AB/BA,
  the registries Nam & Koshino 2017 say relaxation redistributes area among) -- ``PASS``, but
  explicitly ``licenses_go10=False``. It is checkpoint self-consistency on existing small
  cells, not a model-vs-SIESTA comparison on an actual relaxed geometry.
- ``certify_relaxed_matbg_phonon_provider`` (S49): a bound built from archived small-system
  SIESTA FC data -- ``bound_verdict=PASS``, but ``relaxed_physical_phonon_produced=False``
  (both GO-10 and GO-8a are NO-GO; either alone blocks a relaxed ``physical_phonon``).

So this ticket cannot freeze a *measured* relaxed protocol -- there is no relaxed geometry to
measure. What it does, mirroring what S41 did for the rigid Gamma branch before GO-8a existed
and what S44 did for the rigid q != 0 branch before GO-8b existed, is freeze the *rule*: which
gates a future execution must additionally clear beyond the rigid campaign's own GO-1..GO-9
(S47's GO-10, S48's and S49's bounds superseded by their real relaxed-geometry equivalents),
which artifacts a relaxed run may and may not reuse from the rigid one, and why. Two branches
are frozen separately, per the roadmap's own Gamma/q!=0 split (Fase 7's GO-5 gate, Fase 10's
GO-8b gate): ``gamma`` (q=0) requires everything the rigid Gamma branch (S41) requires, plus
GO-10; ``qneq0`` requires everything the rigid q != 0 branch (S44) requires -- GO-8a *and*
GO-8b -- plus GO-10. Neither branch is ready today.

Every reusable formula, direction family, k/q candidate, tolerance definition and check this
module needs already exists in :mod:`preregister_matbg_gamma_rigid_epc` (S41) and
:mod:`preregister_matbg_qneq0_rigid_epc` (S44); both are imported and reused, never
re-implemented. What is genuinely new here is what changes when the *geometry* changes: a
relaxed cell gets its own ``geometry_cell_species_sha256`` (roadmap Section V), which by the
artifact DAG invalidates every artifact keyed to the rigid one -- H, S, eigenpairs, phonons,
raw derivatives and any numeric tolerance measured on them. The rigid protocols' tolerance
*definitions* (formulas contracted in the electronic window) transfer unchanged; their
*computed numbers* do not, and this module forbids copying them (``cost_model`` /
``tolerances_policy`` below) -- the acceptance criterion "el protocolo no reutiliza
tolerancias o caches incompatibles de la geometria rigida" is enforced by construction, not by
a runtime check, because no rigid number is ever written into this protocol's own
``tolerances``/``checks`` sections in the first place.

Units: Ang for displacements/geometry, eV for energies, fractional reciprocal-lattice
coordinates for q/k (``epc_fourier.CONVENTION_ID``, S29).
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
import preregister_matbg_gamma_rigid_epc as gamma_prereg  # noqa: E402
import preregister_matbg_qneq0_rigid_epc as qneq0_prereg  # noqa: E402

prereg = gamma_prereg.prereg
PreregistrationError = prereg.PreregistrationError

SCHEMA = "epc_preregistration_v1"
PROTOCOL_ID = "matbg_relaxed_epc_v1"
TICKET = "E-F_001-S50"
GATE = (
    "GO-10 pre-registration; execution additionally requires GO-1..GO-9 (gamma branch) or "
    "GO-1..GO-9 plus GO-8b (qneq0 branch), applied to the relaxed geometry's own artifacts"
)

RIGID_TARGET_FDF = gamma_prereg.TARGET_FDF
TARGET_ATOM_COUNT = gamma_prereg.TARGET_ATOM_COUNT
DEFAULT_MATBG_CERTIFICATION_DIR = gamma_prereg.DEFAULT_MATBG_CERTIFICATION_DIR
DEFAULT_BASIS_RESPONSE = gamma_prereg.DEFAULT_BASIS_RESPONSE
DEFAULT_GO6_REPORT = gamma_prereg.DEFAULT_GO6_REPORT
DEFAULT_S48_REPORT = REPO_ROOT / "Comparison/results/epc/relaxed_transferability/local_stacking_bound.json"
DEFAULT_S49_REPORT = (
    REPO_ROOT / "Comparison/results/epc/certification/matbg/s49_relaxed_matbg_phonon_provider_bound.json"
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/preregistration/matbg"
PROTOCOL_NAME = "epc_matbg_relaxed_protocol.json"
DEFAULT_RESULT_ROOT = REPO_ROOT / "Comparison/results/epc/relaxed_epc/matbg"

# --------------------------------------------------------------------------- #
# Reused verbatim from the rigid protocols: direction family names/generator,
# k/q candidates, window/occupation rule shape, delta sweep, tolerance
# FORMULAS (not their computed numbers), resource margins, basis-response
# contract. None of this is re-derived for the relaxed target.
# --------------------------------------------------------------------------- #

RESULT_DIRECTION_NAMES = gamma_prereg.RESULT_DIRECTION_NAMES
VALIDATION_DIRECTION_NAMES = gamma_prereg.VALIDATION_DIRECTION_NAMES
CALIBRATION_DIRECTION_NAMES = gamma_prereg.CALIBRATION_DIRECTION_NAMES
WINDOW_STATES_BELOW = gamma_prereg.WINDOW_STATES_BELOW
WINDOW_STATES_ABOVE = gamma_prereg.WINDOW_STATES_ABOVE
DEGENERACY_TOL_EV = gamma_prereg.DEGENERACY_TOL_EV
MIN_WINDOW_GAP_EV = gamma_prereg.MIN_WINDOW_GAP_EV
DELTA_SWEEP_ANG = gamma_prereg.DELTA_SWEEP_ANG
MIN_SIGNAL_TO_FLOOR = gamma_prereg.MIN_SIGNAL_TO_FLOOR
BACKEND_NOISE_MARGIN = gamma_prereg.BACKEND_NOISE_MARGIN
VRAM_MARGIN_GIB = gamma_prereg.VRAM_MARGIN_GIB
RAM_MARGIN_GIB = gamma_prereg.RAM_MARGIN_GIB
FORBIDDEN_TAU_MODEL_SOURCES = gamma_prereg.FORBIDDEN_TAU_MODEL_SOURCES
BASIS_RESPONSE_CONTRACT = dict(gamma_prereg.BASIS_RESPONSE_CONTRACT)
QAWARE_BASIS_RESPONSE_GAP = qneq0_prereg.QAWARE_BASIS_RESPONSE_GAP
SENSITIVITY_AXES = gamma_prereg.SENSITIVITY_AXES

K_CANDIDATES = gamma_prereg.K_CANDIDATES
K_DEGENERATE = gamma_prereg.K_DEGENERATE
K_CALIBRATION = gamma_prereg.K_CALIBRATION
Q_PRIMARY = qneq0_prereg.Q_PRIMARY
Q_CROSSCHECK = qneq0_prereg.Q_CROSSCHECK

SPLITS_GAMMA = gamma_prereg.SPLITS
SPLITS_QNEQ0 = qneq0_prereg.SPLITS

REFERENCE_PATHS_GAMMA = gamma_prereg.REFERENCE_PATHS
REFERENCE_PATHS_QNEQ0 = qneq0_prereg.REFERENCE_PATHS

SIESTA_REFERENCE_EXCLUDED_GAMMA = (
    gamma_prereg.SIESTA_REFERENCE_EXCLUDED
    + " Relaxation redistributes atomic positions but does not change the atom count (roadmap "
    "acceptance criteria, S47), so the same 6 * 11164 = 66984 displaced-SCF-solve estimate S38 "
    "measured intractable for the rigid cell applies at least as strongly here; no cheaper "
    "relaxed-cell SIESTA route is proposed."
)
SIESTA_REFERENCE_EXCLUDED_QNEQ0 = (
    qneq0_prereg.SIESTA_REFERENCE_EXCLUDED
    + " Same atom-count argument as the Gamma branch: relaxation does not reduce the cost "
    "S38/S44 already found intractable for a commensurate q != 0 supercell of the rigid cell."
)

_RELAXED_ONLY_CHECKS: tuple[dict[str, Any], ...] = (
    {
        "id": "relaxed_geometry_signature_distinct_from_rigid",
        "stage": "geometry",
        "applies_to": ("all_relaxed_artifacts",),
        "pass_rule": "geometry_cell_species_sha256 of the relaxed target differs from the "
        "rigid (31, 30) cell's; equality would mean no relaxation occurred and blocks every "
        "downstream artifact by the same rule the artifact DAG (roadmap Section V) already "
        "applies to any geometry change",
    },
    {
        "id": "relaxation_acceptance_criteria_recorded",
        "stage": "provenance",
        "applies_to": ("all_relaxed_artifacts",),
        "pass_rule": "the relaxed geometry artifact records every field of "
        "decide_matbg_relaxation_provider's acceptance_criteria_for_a_future_relaxation "
        "(forces, corrugation, domain_areas, interlayer_spacing_distribution, "
        "geometry_signature, provenance, status_label='relaxed') -- a missing field blocks "
        "GO-10 regardless of which candidate produced the geometry",
    },
    {
        "id": "no_rigid_numeric_reuse",
        "stage": "provenance",
        "applies_to": ("all_relaxed_artifacts",),
        "pass_rule": "every tau_num/tau_backend/plateau/topology-margin NUMBER this branch "
        "reports is measured from the relaxed geometry's own delta sweep; none is copied from "
        "S37/S38/S40/S41/S44's rigid-cell measurements -- only the tolerance FORMULA and the "
        "first-try delta amplitudes (DELTA_SWEEP_ANG) are reused, per tolerances_policy",
    },
    {
        "id": "local_stacking_and_phonon_bounds_passed",
        "stage": "epc",
        "applies_to": ("all_relaxed_artifacts",),
        "pass_rule": "S48 (local_stacking_transferability_bound) and S49 "
        "(relaxed_matbg_phonon_provider_bound) both PASS as a necessary, not sufficient, proxy "
        "before the real relaxed-geometry equivalents (structure/H/derivative transferability "
        "and a physical relaxed phonon) are measured on the actual relaxed cell",
    },
    {
        "id": "relaxed_symmetry_labels_unchecked",
        "stage": "geometry",
        "applies_to": ("k_selection", "q_selection"),
        "pass_rule": "the little-group labels (identity/C3v/C2v) carried over from the rigid "
        "K_CANDIDATES/K_DEGENERATE/K_CALIBRATION/Q_PRIMARY table are verified against the "
        "relaxed cell's own reciprocal lattice and point group, not assumed -- domain formation "
        "(Nam & Koshino 2017) can lower the symmetry the rigid cell had at the same fractional "
        "coordinate; unchecked at freeze time (see blocking)",
    },
)

CHECKS_GAMMA: tuple[dict[str, Any], ...] = (*gamma_prereg.CHECKS, *_RELAXED_ONLY_CHECKS)
CHECKS_QNEQ0: tuple[dict[str, Any], ...] = (*qneq0_prereg.CHECKS, *_RELAXED_ONLY_CHECKS)

TOLERANCES_POLICY = (
    "tau_num/tau_backend/tau_model are defined by the same formulas as the rigid protocols "
    "(a norm of C_W^dag Delta C_W, never of Delta alone) -- those definitions are geometry-"
    "independent and are reused unmodified. DELTA_SWEEP_ANG is reused only as the first-try "
    "amplitude list to probe the relaxed geometry's own plateau; the plateau itself, and every "
    "number derived from it, must be remeasured. No numeric tolerance computed on the rigid "
    "(31, 30) cell (S37/S40/S41/S44) may be copied into a relaxed report."
)

COST_MODEL = {
    "policy": "the A/B contraction cost MODEL (roadmap Section X: T_A = T_JVP + T_serialize + "
    "N_k * T_contract; T_B ~= N_blocks * T_VJP) is reused unmodified -- it is a formula, not a "
    "measurement. Its OPERANDS (wall time, peak RSS/VRAM, sparse derivative NNZ) measured on "
    "the rigid cell (S37 synthetic-displacement benchmark, S40 kernel-table cost) are not "
    "assumed to transfer and are not copied into this protocol.",
    "reason": "relaxation (Nam & Koshino 2017 domain formation) redistributes atomic positions "
    "without changing the atom count (S47 acceptance criteria), but a redistributed geometry "
    "can still gain or lose neighbour-cutoff crossings relative to the rigid cell, which "
    "changes edge count / NNZ and therefore every wall-time and memory number the rigid "
    "benchmark measured -- topology_margin_preserved must be re-evaluated on the relaxed "
    "topology, not inherited from the rigid one.",
    "must_remeasure_on_relaxed_topology": (
        "graph2mat forward/JVP wall time", "peak RSS", "peak VRAM", "derivative sparse NNZ",
        "topology_margin crossing count", "sparse eigensolver time per k",
        "contraction time per (k, q, nu)",
    ),
    "resource_margins_reused": {"VRAM_MARGIN_GIB": VRAM_MARGIN_GIB, "RAM_MARGIN_GIB": RAM_MARGIN_GIB},
}

CLAIM_LADDER_GAMMA: tuple[dict[str, Any], ...] = (
    {
        "outcome": "GO-10 becomes GO-10:<provider> with a relaxed geometry meeting S47's "
        "acceptance criteria; GO-8a becomes GO-8a on that relaxed geometry itself (not the "
        "rigid one, and not S49's archived-FC proxy); C14C becomes GO; and every rigid-Gamma "
        "check (jvp_equals_coordinate_combination, jvp_equals_frozen, "
        "generalized_hellmann_feynman_diagonal, uniform_translation_null, "
        "electronic_gauge_invariance, basis_response_present_and_required) plus every "
        "relaxed-only check above passes when remeasured on the relaxed topology",
        "claim": "relaxed_tbg_gamma_epc_validated (GO-10, Gamma branch); Graph2Mat g is "
        "quantitative for this mode/k on the relaxed target geometry. Publication label: "
        "relaxed_tbg_model -- quantitative_relaxed_matbg_prediction additionally requires the "
        "Fase 12 integrated-observable convergence gates (k/q mesh, broadening, occupations) "
        "and is never claimed by this ticket alone",
    },
    {
        "outcome": "derivative-level checks pass on the relaxed geometry once one exists, but "
        "GO-8a on that relaxed geometry is still NO-GO-8a",
        "claim": "graph2mat_derivative_validated_no_full_ks_coupling_relaxed; parallel ceiling to the "
        "rigid branch's graph2mat_derivative_validated_no_full_ks_coupling; no g_mn_nu may be "
        "published (roadmap B7)",
    },
    {
        "outcome": "any derivative-level or relaxed-only check fails",
        "claim": "NO_GO, attributed to the failing layer (relaxation provider, JVP backend, "
        "topology, formalism); never to the model",
    },
    {
        "outcome": "current state: GO-10 is NO-GO-10 (S47: no relaxation provider in this "
        "environment is defensible), so no relaxed geometry exists to measure; S48/S49 are "
        "proxy bounds only, not measurements on the relaxed cell itself",
        "claim": "no execution is authorised at all for this branch; this pre-registration "
        "exists to freeze what a future execution must not be allowed to choose after seeing "
        "g, mirroring S41's role for the rigid Gamma branch before GO-8a existed",
    },
)

CLAIM_LADDER_QNEQ0: tuple[dict[str, Any], ...] = (
    {
        "outcome": "everything CLAIM_LADDER_GAMMA's top rung requires, plus GO-8b becomes "
        "GO-8b on the relaxed geometry (a q-aware response and a q-aware basis response, both "
        "measured on the relaxed topology) and route-mechanics checks agree at machine "
        "precision",
        "claim": "relaxed_tbg_qneq0_epc_validated (GO-10, non-Gamma branch); Graph2Mat g(q) is "
        "quantitative for this mode/k/q on the relaxed target geometry. Publication label: "
        "relaxed_tbg_model -- same quantitative_relaxed_matbg_prediction restriction as the "
        "Gamma branch",
    },
    {
        "outcome": "route-mechanics checks pass on the relaxed geometry once one exists, but "
        "GO-10 and/or GO-8a and/or GO-8b are still NO-GO",
        "claim": "graph2mat_qneq0_mechanics_validated_no_full_ks_coupling_relaxed; identical ceiling "
        "to the rigid q != 0 branch's own no-full-KS-coupling claim, applied to the relaxed target",
    },
    {
        "outcome": "any route-mechanics or relaxed-only check fails",
        "claim": "NO_GO, attributed to the failing layer; never to the model",
    },
    {
        "outcome": "current state: GO-10 NO-GO-10, GO-8a NO-GO-8a, GO-8b NO-GO-8b, GO-5 NO_GO, "
        "no q-aware basis-response producer exists at any geometry",
        "claim": "no execution is authorised at all for this branch; this pre-registration "
        "exists only to freeze the rule, mirroring S44's role for the rigid q != 0 branch",
    },
)


def preconditions(paths: Mapping[str, Path], *, branch: str) -> list[dict[str, Any]]:
    """Gate state for one branch. GO-10/S48/S49 are common; GO-8b is qneq0-only-required."""
    if branch not in ("gamma", "qneq0"):
        raise PreregistrationError(f"unknown branch {branch!r}")

    go10 = go10_decision.matbg_relaxation_provider_decision()
    s48 = prereg._read_json(paths["s48"])
    s49 = prereg._read_json(paths["s49"])
    go8a = prereg._read_json(paths["go8a"])
    go8b = prereg._read_json(paths["go8b"])
    c14c = prereg._read_json(paths["c14c"])
    go6 = prereg._read_json(paths["go6"])

    rows = [
        prereg._precondition(
            "GO-10 MATBG relaxation provider", Path(go10_decision.__file__), go10,
            verdict="PASS" if go10["verdict"].startswith("GO-10:") else go10["verdict"],
            required=True,
        ),
        prereg._precondition(
            "S48 local-stacking transferability bound", paths["s48"], s48,
            verdict=(s48 or {}).get("verdict"), required=True,
        ),
        prereg._precondition(
            "S49 relaxed phonon-provider bound", paths["s49"], s49,
            verdict=(s49 or {}).get("bound_verdict"), required=True,
        ),
        prereg._precondition(
            "GO-8a MATBG PhononProvider", paths["go8a"], go8a,
            verdict="PASS" if (go8a or {}).get("go8a_status") == "GO-8a" else (go8a or {}).get("go8a_status"),
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
            "GO-8b MATBG q!=0 response", paths["go8b"], go8b,
            verdict="PASS" if (go8b or {}).get("go8b_status") == "GO-8b" else (go8b or {}).get("go8b_status"),
            required=(branch == "qneq0"),
        ),
    ]
    rows[0]["relaxed_geometry_produced"] = bool(go10.get("relaxed_geometry_produced"))
    rows[-1]["note"] = (
        "required: GO-9's own q != 0 branch (S44) needs GO-8b" if branch == "qneq0"
        else "informational only: this branch is Gamma (q=0); GO-8b gates non-Gamma production"
    )
    return rows


def physics_review(*, gates: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    gate_status = {row["name"]: row["status"] for row in gates}
    required_names = [row["name"] for row in gates if row["required_before_execution"]]
    required_gates_pass = all(gate_status.get(name) == "PASS" for name in required_names)
    items = [
        {
            "item": "relaxed_geometry",
            "decision": "blocked",
            "evidence": {"go10_status": gate_status.get("GO-10 MATBG relaxation provider")},
            "basis": "S47 found no defensible relaxation provider for this environment "
            "(NO-GO-10); no relaxed positions exist anywhere in this repository, so nothing "
            "below can be measured yet -- only frozen by rule",
        },
        {
            "item": "proxy_bounds",
            "decision": "approved" if (
                gate_status.get("S48 local-stacking transferability bound") == "PASS"
                and gate_status.get("S49 relaxed phonon-provider bound") == "PASS"
            ) else "blocked",
            "evidence": {
                "s48": gate_status.get("S48 local-stacking transferability bound"),
                "s49": gate_status.get("S49 relaxed phonon-provider bound"),
            },
            "basis": "S48/S49 both PASS as proxies (local stacking registries, archived FC "
            "data) but explicitly license neither GO-10 nor a physical relaxed phonon "
            "(licenses_go10=False, relaxed_physical_phonon_produced=False) -- approval here is "
            "of the proxy bound, not of any relaxed-geometry claim",
        },
        {
            "item": "candidate_selection",
            "decision": "approved",
            "evidence": {"reused_from": "preregister_matbg_gamma_rigid_epc / "
                         "preregister_matbg_qneq0_rigid_epc", "branch": branch},
            "basis": "direction families, k/q candidates and the selection rule are frozen by "
            "value, reused unmodified from the rigid protocols; approval here is of the "
            "selection, not of any physical mode measured on a relaxed geometry that does not "
            "exist",
        },
        {
            "item": "references",
            "decision": "approved",
            "evidence": {"paths": [
                row["path"] for row in (REFERENCE_PATHS_QNEQ0 if branch == "qneq0" else REFERENCE_PATHS_GAMMA)
            ]},
            "basis": "same two-path (or Q-C/Q-B) Graph2Mat-only reference set as the rigid "
            "branch; a SIESTA reference is excluded at least as strongly on the relaxed cell "
            "(same atom count, roadmap S47 acceptance criteria)",
        },
        {
            "item": "gates",
            "decision": "approved" if required_gates_pass else "blocked",
            "evidence": gate_status,
            "basis": "every required gate's status is read live from its own certification "
            "artifact or decision function, never asserted",
        },
    ]
    return {
        "reviewer": "automated_precondition_review (evidence-bound)",
        "items": items,
        "status": "APPROVED" if all(row["decision"] == "approved" for row in items) else "BLOCKED",
        "human_sign_off": None,
    }


def _branch_protocol(
    branch: str,
    *,
    gate_paths: Mapping[str, Path],
) -> dict[str, Any]:
    splits = SPLITS_QNEQ0 if branch == "qneq0" else SPLITS_GAMMA
    prereg.assert_splits_disjoint(splits)
    gates = preconditions(gate_paths, branch=branch)
    review = physics_review(gates=gates, branch=branch)

    blocking = [
        f"{row['name']}: {row['status']}" for row in gates
        if row["required_before_execution"] and row["status"] != "PASS"
    ]
    blocking.append(
        "no_relaxed_geometry: S47 found NO-GO-10; no relaxed positions file exists to select "
        "k/q, build directions or measure any tolerance against"
    )
    blocking.append(
        "relaxed_symmetry_labels_unchecked: little-group labels are carried over from the "
        "rigid convention, not verified against a relaxed reciprocal lattice that does not "
        "yet exist"
    )
    if branch == "qneq0":
        blocking.append("qaware_basis_response_producer_missing: " + QAWARE_BASIS_RESPONSE_GAP)

    is_qneq0 = branch == "qneq0"
    protocol: dict[str, Any] = {
        "branch": branch,
        "experiment": {
            "system": "relaxed magic-angle TBG target geometry (not yet produced, GO-10: "
            f"{go10_decision.matbg_relaxation_provider_decision()['verdict']}); rigid reference "
            "geometry materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf",
            "target_atom_count": TARGET_ATOM_COUNT,
            "q_fractional_primary": list(Q_PRIMARY["k"]) if is_qneq0 else [0.0, 0.0, 0.0],
            "q_label_primary": Q_PRIMARY["label"] if is_qneq0 else "Gamma",
            "q_fractional_cross_check": list(Q_CROSSCHECK["k"]) if is_qneq0 else None,
            "k_plus_q_equals_k": not is_qneq0,
            "geometry_produced": False,
            "publication_label_if_validated": "relaxed_tbg_model",
            "forbidden_publication_labels": ["quantitative_relaxed_matbg_prediction"],
        },
        "electronic": {
            "window": {"states_below": WINDOW_STATES_BELOW, "states_above": WINDOW_STATES_ABOVE},
            "occupations": "pending_measurement: no relaxed neutrality estimate exists (requires "
            "a converged relaxed H/S export, itself gated by GO-10)",
            "degeneracy_tol_ev": DEGENERACY_TOL_EV,
            "min_window_gap_ev": MIN_WINDOW_GAP_EV,
            "k_candidates": list(K_CANDIDATES),
            "k_degenerate": K_DEGENERATE if not is_qneq0 else None,
            "k_calibration": list(K_CALIBRATION),
            "q_candidates": {"primary": Q_PRIMARY, "cross_check_only": Q_CROSSCHECK} if is_qneq0 else None,
            "k_selection_status": "pending_measurement",
            "subspace_machinery": "Comparison/scripts/epc_subspaces.py (GO-6), reused unmodified",
        },
        "phonon": {
            "status": "no relaxed geometry (GO-10: NO-GO-10) and no MATBG-scale PhononProvider "
            "(GO-8a: NO-GO-8a) exist; S49's bound is a proxy, not a measurement on this branch's "
            "own target",
            "candidate_direction_names": {
                "result": list(RESULT_DIRECTION_NAMES),
                "validation": list(VALIDATION_DIRECTION_NAMES),
                "calibration": list(CALIBRATION_DIRECTION_NAMES),
            },
            "artifact_kind_of_candidates": "synthetic_test_displacement",
            "artifact_kind_forbidden_until_go10_and_go8a": "physical_phonon",
        },
        "references": {
            "paths": list(REFERENCE_PATHS_QNEQ0 if is_qneq0 else REFERENCE_PATHS_GAMMA),
            "siesta_reference_excluded_reason": (
                SIESTA_REFERENCE_EXCLUDED_QNEQ0 if is_qneq0 else SIESTA_REFERENCE_EXCLUDED_GAMMA
            ),
            "basis_response": (
                {**BASIS_RESPONSE_CONTRACT, "must_be_q_aware": True, "gap": QAWARE_BASIS_RESPONSE_GAP}
                if is_qneq0 else BASIS_RESPONSE_CONTRACT
            ),
            "formalism_id": BASIS_RESPONSE_CONTRACT["formalism_id"],
        },
        "splits": {
            **{
                name: {
                    **split,
                    "directions": list(split["directions"]),
                    "k_labels": list(split["k_labels"]),
                    **({"q_labels": list(split["q_labels"])} if "q_labels" in split else {}),
                    "fc_ranges": list(split["fc_ranges"]),
                }
                for name, split in splits.items()
            },
        },
        "perturbation": {
            "normalization": gamma_prereg.fdp.NORMALIZATION,
            "method": "central",
            "delta_sweep_ang_first_try": list(DELTA_SWEEP_ANG),
        },
        "tolerances_policy": TOLERANCES_POLICY,
        "cost_model": COST_MODEL,
        "checks": list(CHECKS_QNEQ0 if is_qneq0 else CHECKS_GAMMA),
        "sensitivity": list(SENSITIVITY_AXES) if not is_qneq0 else None,
        "claim_ladder": list(CLAIM_LADDER_QNEQ0 if is_qneq0 else CLAIM_LADDER_GAMMA),
        "preconditions": gates,
        "physics_review": review,
        "blocking": blocking,
        "ready_to_execute": not blocking and review["status"] == "APPROVED",
    }
    return protocol


def build_protocol(
    *,
    certification_dir: Path = DEFAULT_MATBG_CERTIFICATION_DIR,
    basis_response_report: Path = DEFAULT_BASIS_RESPONSE,
    go6_report: Path = DEFAULT_GO6_REPORT,
    s48_report: Path = DEFAULT_S48_REPORT,
    s49_report: Path = DEFAULT_S49_REPORT,
) -> dict[str, Any]:
    gate_paths = {
        "s48": s48_report,
        "s49": s49_report,
        "go8a": certification_dir / gamma_prereg.go8a_cert.REPORT_NAME,
        "go8b": certification_dir / gamma_prereg.go8b_cert.REPORT_NAME,
        "c14c": basis_response_report,
        "go6": go6_report,
    }

    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "ticket": TICKET,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_content_sha256": None,
        "branches": {
            "gamma": _branch_protocol("gamma", gate_paths=gate_paths),
            "qneq0": _branch_protocol("qneq0", gate_paths=gate_paths),
        },
        "ready_to_execute": False,
        "result_root": str(DEFAULT_RESULT_ROOT),
    }
    protocol["ready_to_execute"] = any(
        branch["ready_to_execute"] for branch in protocol["branches"].values()
    )
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


def require_frozen_protocol(
    path: Path = DEFAULT_OUTPUT_DIR / PROTOCOL_NAME, *, branch: str = "gamma"
) -> dict[str, Any]:
    """Gate for a future execution step: refuse to compute anything outside this protocol."""
    protocol = load_protocol(path)
    if prereg.freeze_hash(protocol) != protocol.get("frozen_content_sha256"):
        raise PreregistrationError(f"{path} has been edited since it was frozen")
    branch_protocol = protocol["branches"][branch]
    if not branch_protocol.get("ready_to_execute"):
        raise PreregistrationError(
            f"the protocol is frozen but branch {branch!r}'s preconditions are not met: "
            f"{branch_protocol.get('blocking')}"
        )
    return protocol


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--certification-dir", type=Path, default=DEFAULT_MATBG_CERTIFICATION_DIR)
    parser.add_argument("--basis-response-report", type=Path, default=DEFAULT_BASIS_RESPONSE)
    parser.add_argument("--go6-report", type=Path, default=DEFAULT_GO6_REPORT)
    parser.add_argument("--s48-report", type=Path, default=DEFAULT_S48_REPORT)
    parser.add_argument("--s49-report", type=Path, default=DEFAULT_S49_REPORT)
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
        for branch_protocol in protocol["branches"].values():
            branch_protocol["physics_review"]["human_sign_off"] = {
                "reviewer": str(args.sign_off),
                "signed_at": datetime.now(timezone.utc).isoformat(),
                "signed_content_sha256": protocol["frozen_content_sha256"],
            }
        protocol["frozen_content_sha256"] = prereg.freeze_hash(protocol)
    else:
        protocol = build_protocol(
            certification_dir=args.certification_dir,
            basis_response_report=args.basis_response_report,
            go6_report=args.go6_report,
            s48_report=args.s48_report,
            s49_report=args.s49_report,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")

    print(f"pre-registration {protocol['protocol_id']} -> {output}")
    print(f"  frozen_content_sha256 : {protocol['frozen_content_sha256']}")
    for name, branch in protocol["branches"].items():
        print(f"  branch {name:<7} physics review={branch['physics_review']['status']} "
              f"ready_to_execute={branch['ready_to_execute']}")
        for reason in branch["blocking"]:
            print(f"    blocked by: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
