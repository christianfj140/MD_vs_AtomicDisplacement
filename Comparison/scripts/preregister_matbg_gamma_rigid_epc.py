#!/usr/bin/env python3
"""E-F_001-S41: pre-register the rigid-MATBG Gamma (q=0) EPC campaign.

Fase 10d of the roadmap ("Physical selected-mode campaign"), scoped down to
what a pre-registration can freeze *today*: S38/S39 already certified
NO-GO-8a (no scalable, validated MATBG ``PhononProvider`` exists) and C14C
(basis response) is NO_GO on its own unresolved intra-atomic term. Neither
gate is closed by this ticket. What this ticket freezes is everything a
future GO-9 execution must not be allowed to choose *after* seeing ``g``: the
handful of candidate Gamma sectors, the electronic k-selection rule and
candidate list, the window/occupation contract, the splits, the three
tolerances and the claim ladder -- written to one file and hashed, mirroring
``preregister_graphene_gamma_epc.py`` (C19/C20), whose generic utilities
(``freeze_hash``, the split-disjointness guards, the JSON helpers) this
module imports and reuses rather than re-implementing.

Two differences from the graphene protocol are structural, not oversights:

``no live k/mode measurement``
    Graphene's script measures the candidate spectra live (2-atom ``eigh``,
    seconds). The rigid ``(31, 30)`` cell is 44 656 orbitals; a live sparse
    generalized solve per candidate k is exactly the cost this repository's
    compute policy requires a preflight before, not a side effect of freezing
    a document. This protocol therefore freezes the *rule* and the *ordered
    candidate list* (:func:`fd_perturbation_space`-style, same shape as
    graphene's ``K_CANDIDATES``) and records ``k_selection.status =
    "pending_measurement"``: a future run applies
    ``preregister_graphene_gamma_epc.select_result_k`` -- already generic --
    to real measured spectra, it does not re-derive the rule.

``no SIESTA reference path``
    S38 measured that an exact SIESTA ``FC.Save.dHS`` campaign on this cell
    needs ``6 * 11164 = 66984`` displaced SCF solves and called that
    intractable at this scale. ``REFERENCE_PATHS`` here has two entries, not
    three: Graph2Mat JVP vs. Graph2Mat frozen central difference of the *same*
    model. A disagreement between them is a backend bug; it is not, and
    cannot become, a claim about the model's accuracy against DFT.

Every candidate direction is built with :func:`fd_perturbation_space` on the
real ``(31, 30)`` geometry and stays tagged
:data:`shared.artifact_signature.SYNTHETIC_TEST_DISPLACEMENT` throughout this
protocol -- it is a benchmark/candidate pattern, never a phonon eigenvector,
per :data:`CLAIM_LADDER` and roadmap B7. Nothing in this module computes a
directional derivative, a JVP or a ``g``: the whole point is that it runs
before any of those exist.

Units: Ang for displacements and geometry, eV for energies. Narrative and
frozen values: ``docs/epc_s41_matbg_gamma_rigid_preregistration.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fd_perturbation_space as fdp  # noqa: E402
import preregister_graphene_gamma_epc as prereg  # noqa: E402
import certify_matbg_phonon_provider as go8a_cert  # noqa: E402
import certify_qneq0_matbg_response as go8b_cert  # noqa: E402
from epc_formalism import (  # noqa: E402
    FORMALISM_ID,
    FORMALISM_ID_IGNORE_OVERLAP,
    FORMALISM_ID_SYMMETRIC,
    REPRESENTATION_S_L_S_R,
)
from reference_provenance import geometry_cell_species_sha256  # noqa: E402

PreregistrationError = prereg.PreregistrationError

SCHEMA = "epc_preregistration_v1"
PROTOCOL_ID = "matbg_rigid_gamma_epc_v1"
TICKET = "E-F_001-S41"
GATE = "GO-9 pre-registration (Gamma branch); execution additionally requires GO-8a and GO-1..GO-6"

TARGET_FDF = REPO_ROOT / "materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf"
TARGET_ATOM_COUNT = go8a_cert.TARGET_ATOM_COUNT  # 11164, S38's own constant
TARGET_ORBITAL_COUNT = 44656  # roadmap contract; cross-checked against neutrality_estimate.json

DEFAULT_POSITIONS_XYZ = (
    REPO_ROOT
    / "Comparison/results/tbg_pure_graph2mat/overlap/twisted_bilayer_graphene_1p084549deg.xyz"
)
DEFAULT_NEUTRALITY_ESTIMATE = REPO_ROOT / "Comparison/results/tbg_pure_graph2mat/neutrality_estimate.json"
DEFAULT_MATBG_CERTIFICATION_DIR = go8a_cert.DEFAULT_OUTPUT_DIR
DEFAULT_BASIS_RESPONSE = prereg.DEFAULT_BASIS_RESPONSE
DEFAULT_GO6_REPORT = prereg.DEFAULT_GO6_REPORT

DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/preregistration/matbg"
PROTOCOL_NAME = "epc_matbg_rigid_gamma_protocol.json"
DEFAULT_RESULT_ROOT = REPO_ROOT / "Comparison/results/epc/gamma_epc/matbg"

# --------------------------------------------------------------------------- #
# Frozen choices
# --------------------------------------------------------------------------- #

# Moire-BZ fractional coordinates, same convention graphene's hexagonal cell
# uses (the (31,30) commensuration is a hexagonal supercell by construction).
# [DESIGN, unverified]: which of these labels is actually Gamma/K/M of *this*
# specific twisted supercell's reciprocal lattice is checked, not assumed --
# see the ``moire_hexagonal_convention_checked`` check below.
K_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"label": "k_generic_1", "k": (0.29, 0.11, 0.0), "little_group": "identity",
     "role": "stage_1_non_degenerate_candidate_pending_measurement"},
    {"label": "k_generic_2", "k": (0.21, 0.37, 0.0), "little_group": "identity",
     "role": "stage_1_non_degenerate_candidate_pending_measurement"},
)
K_DEGENERATE = {
    "label": "K", "k": (2.0 / 3.0, 1.0 / 3.0, 0.0), "little_group": "C3v",
    "basis": "moire-BZ analogue of graphene K; the flat-band near-degeneracy there is measured "
    "in this repository only by the Moon-Koshino tight-binding model (docs/tbg_tight_binding_reference.md, "
    "S=I, no ion-ion/DFT cross-check), not yet by the Graph2Mat/SIESTA production Hamiltonian",
}
K_CALIBRATION = (
    {"label": "gamma", "k": (0.0, 0.0, 0.0), "little_group": "moire_point_group",
     "basis": "the only electronic k with a persisted eigenspace artifact in this repository today "
     "(epc_synthetic_benchmark/eigenspace_probe_manual), produced manually rather than by an "
     "automated, signed campaign step -- cited with that caveat, not as a certified measurement"},
    {"label": "M", "k": (0.5, 0.0, 0.0), "little_group": "C2v"},
)

# The four families of Comparison/scripts/benchmark_matbg_synthetic_displacements.py
# (S37/D06), the same generic fd_perturbation_space functions applied to this
# structure's own positions -- not re-implemented here.
RESULT_DIRECTION_NAMES = ("breathing_like", "shear_like_x", "optical_like")
VALIDATION_DIRECTION_NAMES = ("random_seed0",)
CALIBRATION_DIRECTION_NAMES = ("translation_x",)

WINDOW_STATES_BELOW = 2
WINDOW_STATES_ABOVE = 2
MU_CONVENTION = "exported_hamiltonian_fermi_zero"

MIN_WINDOW_GAP_EV = 0.05
DEGENERACY_TOL_EV = 1e-6  # the value this repository's real solver run actually used (S37 probe)

DELTA_SWEEP_ANG = (0.005, 0.01, 0.02)  # same sweep GO-2/S37 already use elsewhere
MIN_SIGNAL_TO_FLOOR = 10.0
BACKEND_NOISE_MARGIN = 10.0

# roadmap Section X operational headroom, same constants S37/S40 already froze.
VRAM_MARGIN_GIB = 28.0
RAM_MARGIN_GIB = 62.0

FORBIDDEN_TAU_MODEL_SOURCES = (
    "relative Frobenius of D_H at graphene scale (C14, a different system)",
    "MAE or Frobenius of H/D_H matrix elements over all orbital pairs",
    "any global matrix norm converted into a percentage of g",
    "the S37/S40 JVP-cost scaling result, which measures wall time and NNZ, not accuracy",
)

SPLITS: dict[str, dict[str, Any]] = {
    "calibration": {
        "purpose": "fix FD floors and the delta plateau once a real measurement exists. May be "
        "inspected freely; nothing measured here is a result.",
        "directions": CALIBRATION_DIRECTION_NAMES,
        "k_labels": ("gamma", "M"),
        "fc_ranges": (),
    },
    "validation": {
        "purpose": "test whether a calibrated bound transfers to a direction and a k that took "
        "part in no fit. Used once, after tolerances are frozen.",
        "directions": VALIDATION_DIRECTION_NAMES,
        "k_labels": ("k_generic_2",),
        "fc_ranges": (),
    },
    "result": {
        "purpose": "the pre-registered experiment: the three candidate Gamma sectors S37/D06 "
        "already benchmarked for cost, now the target of a physical validation once GO-8a "
        "closes. Evaluated last, never used to set a tolerance.",
        "directions": RESULT_DIRECTION_NAMES,
        "k_labels": ("k_generic_1", "K"),
        "fc_ranges": (),
    },
}

REFERENCE_PATHS: tuple[dict[str, Any], ...] = (
    {
        "path": "graph2mat_jvp",
        "electronic_derivative_backend": "graph2mat",
        "derivative_method": "jvp",
        "artifact": "directional double-backward JVP on the frozen neighbour topology "
        "(graph2mat_autograd_derivatives.compute_graph2mat_directional_derivative)",
        "internal_check": "equals the coordinate-JVP combination to float64 roundoff (GO-3)",
        "new_runs_required": "one JVP call per result direction; the checkpoint is the existing "
        "tbg_pure_graph2mat one, not retrained",
    },
    {
        "path": "graph2mat_frozen",
        "electronic_derivative_backend": "graph2mat",
        "derivative_method": "frozen_central_difference",
        "artifact": "[H(R + delta v) - H(R - delta v)] / (2 delta) of the same model",
        "internal_check": "delta plateau; topology unchanged at every amplitude",
        "new_runs_required": "none; forward passes only",
    },
)
SIESTA_REFERENCE_EXCLUDED = (
    "an exact SIESTA FC.Save.dHS campaign on this cell needs 6 * 11164 = 66984 displaced SCF "
    "solves (S38, docs/epc_s38_matbg_phonon_provider_decision.md Section 3) -- measured "
    "intractable at this scale, not merely expensive. Until a cheaper certified SIESTA route or "
    "GO-8a exists, the two Graph2Mat paths are the only references this protocol may cite; a "
    "disagreement between them is a backend bug in the same weights, never a DFT-accuracy claim."
)

BASIS_RESPONSE_CONTRACT = {
    "formalism_id": FORMALISM_ID,
    "representation": REPRESENTATION_S_L_S_R,
    "provider": "Comparison/scripts/epc_basis_response.py (C14B), gated by C14C -- the same "
    "global formalism gate graphene's protocol depends on; this ticket adds no new physics",
    "diagnostic_variants_allowed": (FORMALISM_ID_SYMMETRIC, FORMALISM_ID_IGNORE_OVERLAP),
    "diagnostic_variants_are_never_the_result": True,
    "unresolved_term_blocking_production": "intra_atomic_basis_response (C14C memo A2): NO_GO "
    "on graphene already blocks PAO-covariant response everywhere this formalism_id is used, MATBG included",
}

TOLERANCES: dict[str, dict[str, Any]] = {
    "tau_num": {
        "what": "numerical floor of each path: finite difference, plateau curvature, dtype",
        "contracted_in": "the S-orthonormal window C_W at the k being measured",
        "definition": "tau_num = max over plateau amplitude pairs (a, b) of "
        "|| C_W^dag (Delta(delta_a) - Delta(delta_b)) C_W ||_F -- identical construction to "
        "GO-2/graphene's tau_num, evaluated on this system's own delta sweep",
        "unit": "eV/Ang",
        "pass_rule": "the result is INCONCLUSIVE unless "
        "|| C_W^dag Delta C_W ||_F >= MIN_SIGNAL_TO_FLOOR * tau_num",
    },
    "tau_backend": {
        "what": "JVP vs frozen central difference of the same Graph2Mat model",
        "contracted_in": "the S-orthonormal window C_W at the k being measured",
        "definition": "tau_backend = || C_W^dag (Delta_JVP - Delta_frozen) C_W ||_F",
        "pass_rule": "tau_backend <= BACKEND_NOISE_MARGIN * tau_num(graph2mat_frozen); same "
        "function of the same weights, so anything above the frozen side's own floor is a bug",
        "unit": "eV/Ang",
    },
    "tau_model": {
        "what": "no SIESTA reference exists at this scale (see SIESTA_REFERENCE_EXCLUDED), so "
        "there is no model-vs-DFT tolerance this protocol can define for MATBG itself",
        "definition": "undefined pending either a cheaper certified SIESTA route or transfer of "
        "graphene's GO-4 tau_model with an explicit, tested transferability argument -- neither "
        "exists yet, so tau_model is NOT frozen by this protocol",
        "forbidden_sources": FORBIDDEN_TAU_MODEL_SOURCES,
        "consequence": "no claim on this system may cross above "
        "'graph2mat_derivative_validated_no_full_ks_coupling' in CLAIM_LADDER until this is resolved",
    },
}

CHECKS: tuple[dict[str, Any], ...] = (
    {
        "id": "jvp_equals_coordinate_combination",
        "stage": "derivative",
        "applies_to": ("graph2mat_jvp",),
        "pass_rule": "relative Frobenius <= 1e-11 in float64 (GO-3 roundoff tolerance)",
    },
    {
        "id": "jvp_equals_frozen",
        "stage": "derivative",
        "applies_to": ("graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "tau_backend rule",
    },
    {
        "id": "topology_unchanged",
        "stage": "derivative",
        "applies_to": ("graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "no neighbour pair crosses its cutoff under +- max(delta) v for any result "
        "direction; status today is not_yet_measured (S40 item 7: no real MATBG edge count has "
        "ever been measured, S37 OOM'd first) -- this check blocks execution, it is not assumed",
    },
    {
        "id": "generalized_hellmann_feynman_diagonal",
        "stage": "epc",
        "applies_to": ("graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "d(eps_n) from C_n^dag (D_H - eps_n D_S) C_n equals the central FD of the "
        "eigenvalue itself, inside tau_num",
    },
    {
        "id": "uniform_translation_null",
        "stage": "epc",
        "applies_to": ("graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "g of translation_x is <= tau_num in absolute value at every result k "
        "(acoustic control)",
    },
    {
        "id": "electronic_gauge_invariance",
        "stage": "epc",
        "applies_to": ("graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "under a block-diagonal random unitary on the window "
        "(epc_subspaces.random_gauge_rotation) singular values and Frobenius norm of every "
        "cluster block move by less than GO-6's metric tolerance, while cross_cluster_mixing "
        "moves them far more",
    },
    {
        "id": "units_round_trip",
        "stage": "epc",
        "applies_to": ("graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "eV/Ang * Ang = eV end to end (once a physical amplitude exists via GO-8a)",
    },
    {
        "id": "basis_response_present_and_required",
        "stage": "epc",
        "applies_to": ("graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "PAO-covariant response is built only from a basis-response artifact whose formalism_id "
        "and representation match BASIS_RESPONSE_CONTRACT; a missing artifact raises",
    },
    {
        "id": "moire_hexagonal_convention_checked",
        "stage": "geometry",
        "applies_to": ("k_selection",),
        "pass_rule": "the (2/3, 1/3, 0) / (1/2, 0, 0) fractional labels are checked against this "
        "structure's own reciprocal lattice (from geometry_cell_species_sha256's parsed cell), "
        "not assumed from graphene's convention; unchecked at freeze time (see blocking)",
    },
    {
        "id": "no_forbidden_labels",
        "stage": "provenance",
        "applies_to": ("all_result_artifacts",),
        "pass_rule": "every artifact this protocol's execution writes stays tagged "
        "shared.artifact_signature.SYNTHETIC_TEST_DISPLACEMENT; PHYSICAL_PHONON and any "
        "'phonon_eigenmode'/'g_mn_nu' label require a GO-8a-certified PhononProvider payload "
        "(phonon_provider.require_physical_phonon), never this protocol's own candidate patterns",
    },
)

SENSITIVITY_AXES: tuple[dict[str, Any], ...] = (
    {
        "axis": "delta",
        "vary": "the three amplitudes of DELTA_SWEEP_ANG",
        "tolerance": "plateau; a direction without a plateau is INCONCLUSIVE",
    },
    {
        "axis": "formalism_variant",
        "vary": f"{FORMALISM_ID} vs {FORMALISM_ID_SYMMETRIC} vs {FORMALISM_ID_IGNORE_OVERLAP}",
        "tolerance": "diagnostic only: the difference is the size of the term each variant "
        "drops, reported, not tolerated",
    },
    {
        "axis": "window_size",
        "vary": "the frozen 2+2 window vs 3+3",
        "tolerance": "gauge-invariant metrics of the common sub-block must not move by more "
        "than tau_num",
    },
)

CLAIM_LADDER: tuple[dict[str, Any], ...] = (
    {
        "outcome": "GO-8a becomes GO-8a, C14C becomes GO, and every check passes",
        "claim": "rigid_tbg_gamma_epc_validated (GO-9, Gamma branch); Graph2Mat g is "
        "quantitative for this mode/k on the rigid (31, 30) cell. Publication label: "
        "rigid_tbg_model -- never quantitative_relaxed_matbg_prediction (roadmap Fase 10, GO-10)",
    },
    {
        "outcome": "derivative-level checks pass (jvp_equals_coordinate_combination, "
        "jvp_equals_frozen, generalized_hellmann_feynman_diagonal) but GO-8a is still NO-GO-8a",
        "claim": "graph2mat_derivative_validated_no_full_ks_coupling; the three result-split "
        "directions stay tagged synthetic_test_displacement; no g_mn_nu may be published "
        "(roadmap B7 / S39 verdict)",
    },
    {
        "outcome": "any derivative-level check fails",
        "claim": "NO_GO, attributed to the failing layer (JVP backend, topology, formalism); "
        "never to the model",
    },
    {
        "outcome": "signal below MIN_SIGNAL_TO_FLOOR * tau_num",
        "claim": "INCONCLUSIVE; not a PASS and not a FAIL",
    },
    {
        "outcome": "C14C intra_atomic_basis_response still unresolved (current state)",
        "claim": "no quantitative g at all, on any system, including MATBG once GO-8a closes; "
        "at most a two-path Graph2Mat derivative validation plus a bound whose width is C14C's "
        "intra-atomic sensitivity",
    },
)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def read_xyz_positions_ang(path: Path) -> np.ndarray:
    """Minimal stdlib XYZ reader: element x y z, no dependency beyond numpy."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise PreregistrationError(f"{path}: not a valid xyz file")
    count = int(lines[0].strip())
    rows = []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    array = np.asarray(rows, dtype=np.float64)
    if array.shape != (count, 3):
        raise PreregistrationError(f"{path}: header declares {count} atoms, parsed {array.shape[0]}")
    return array


def geometry_report(positions_xyz: Path, target_fdf: Path) -> dict[str, Any]:
    positions = read_xyz_positions_ang(positions_xyz)
    atom_count = int(positions.shape[0])
    return {
        "target_fdf": str(target_fdf),
        "positions_source": str(positions_xyz),
        "atom_count": atom_count,
        "atom_count_matches_target": atom_count == TARGET_ATOM_COUNT,
        "geometry_signature": geometry_cell_species_sha256(target_fdf),
        "positions_sha256": prereg._sha256_of(positions.tolist()),
    }


def build_candidate_directions(positions_ang: np.ndarray) -> dict[str, Any]:
    """The five directions this protocol's splits reference, by name."""
    atom_count = int(positions_ang.shape[0])
    benchmark_set = fdp.synthetic_benchmark_direction_set(positions_ang, seed=0)
    translation = fdp.uniform_translation(atom_count, "x")
    directions = {d.name: d for d in (*benchmark_set, translation)}
    expected = set(RESULT_DIRECTION_NAMES) | set(VALIDATION_DIRECTION_NAMES) | set(CALIBRATION_DIRECTION_NAMES)
    missing = expected - set(directions)
    if missing:
        raise PreregistrationError(f"fd_perturbation_space did not produce {sorted(missing)}")
    return directions


# --------------------------------------------------------------------------- #
# Electronic occupation contract (measured from the real, already-computed
# neutrality estimate -- not fabricated for this protocol)
# --------------------------------------------------------------------------- #


def occupation_report(neutrality_estimate_path: Path) -> dict[str, Any]:
    payload = prereg._read_json(neutrality_estimate_path)
    if payload is None:
        raise PreregistrationError(
            f"no neutrality estimate at {neutrality_estimate_path}: this protocol freezes the "
            "occupation contract from a real, already-computed artifact, not by assumption"
        )
    return {
        "source": str(neutrality_estimate_path),
        "source_sha256": prereg._file_sha256(neutrality_estimate_path),
        "matrix_dimension": payload.get("matrix_dimension"),
        "matrix_dimension_matches_target_orbitals": payload.get("matrix_dimension") == TARGET_ORBITAL_COUNT,
        "neutral_electrons": payload.get("neutral_electrons"),
        "target_occupied_bands_per_k": payload.get("target_occupied_bands_per_k"),
        "shift_ev": payload.get("shift_eV"),
        "spin_degeneracy": payload.get("spin_degeneracy"),
        "mu_convention": MU_CONVENTION,
        "scope": payload.get("scope"),
    }


# --------------------------------------------------------------------------- #
# Preconditions
# --------------------------------------------------------------------------- #


def preconditions(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    go8a = prereg._read_json(paths["go8a"])
    go8b = prereg._read_json(paths["go8b"])
    c14c = prereg._read_json(paths["c14c"])
    go6 = prereg._read_json(paths["go6"])

    rows = [
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
            required=False,
        ),
    ]
    rows[-1]["note"] = "informational only: this protocol is Gamma (q=0); GO-8b gates non-Gamma production"
    if go8a is not None:
        rows[0]["physical_phonon_produced"] = bool(go8a.get("physical_phonon_produced"))
    return rows


# --------------------------------------------------------------------------- #
# Physics review
# --------------------------------------------------------------------------- #


def physics_review(
    *, geometry: Mapping[str, Any], gates: list[dict[str, Any]], k_selection_measured: bool
) -> dict[str, Any]:
    gate_status = {row["name"]: row["status"] for row in gates}
    items = [
        {
            "item": "candidate_sectors",
            "decision": "approved",
            "evidence": {
                "directions": list(RESULT_DIRECTION_NAMES),
                "source": "fd_perturbation_space.synthetic_benchmark_direction_set, applied to "
                "the real (31, 30) positions -- the same call S37/D06 already benchmarked for cost",
            },
            "basis": "shear/breathing/optical are the three geometric families the AB-bilayer "
            "GO-7 gate and S37's cost model already exercise on this exact system; approval here "
            "is of the *selection*, not of any physical mode -- see basis_response/GO-8a below "
            "for why no g may yet be computed from them",
        },
        {
            "item": "k_selection",
            "decision": "approved" if k_selection_measured else "blocked",
            "evidence": {"rule_frozen": True, "candidates_frozen": True, "spectra_measured": k_selection_measured},
            "basis": "the rule and candidate list are frozen by value; applying "
            "preregister_graphene_gamma_epc.select_result_k to a real measured spectrum is "
            "deferred to execution, gated by its own resource preflight (roadmap Section X) -- "
            "not run inside this pre-registration",
        },
        {
            "item": "references",
            "decision": "approved" if geometry["atom_count_matches_target"] else "blocked",
            "evidence": {"paths": [row["path"] for row in REFERENCE_PATHS], "geometry": dict(geometry)},
            "basis": "two Graph2Mat paths only (SIESTA_REFERENCE_EXCLUDED); their disagreement "
            "is a backend bug in the same weights, never a claim about DFT accuracy",
        },
        {
            "item": "error_budget",
            "decision": "blocked",
            "evidence": {
                "tau_num": TOLERANCES["tau_num"]["definition"],
                "tau_backend": TOLERANCES["tau_backend"]["pass_rule"],
                "tau_model": "undefined (see TOLERANCES['tau_model'])",
                "forbidden_sources": list(FORBIDDEN_TAU_MODEL_SOURCES),
            },
            "basis": "tau_num/tau_backend are fully defined and contracted in the electronic "
            "subspace; tau_model cannot be defined until a SIESTA reference or a tested "
            "graphene-transfer argument exists for this system -- so the item stays blocked by "
            "design, not by omission",
        },
        {
            "item": "gates",
            "decision": "approved" if gate_status.get("GO-8a MATBG PhononProvider") == "PASS"
            and gate_status.get("C14C basis response (global formalism gate)") == "PASS"
            and gate_status.get("GO-6 subspace metrics (global gate)") == "PASS"
            else "blocked",
            "evidence": gate_status,
            "basis": "GO-8a, C14C and GO-6 are required before any PAO-projected coupling on this or any "
            "system; their current status is read live from the certification artifacts, not "
            "asserted",
        },
    ]
    return {
        "reviewer": "automated_precondition_review (evidence-bound)",
        "items": items,
        "status": "APPROVED" if all(row["decision"] == "approved" for row in items) else "BLOCKED",
        "human_sign_off": None,
    }


# --------------------------------------------------------------------------- #
# Build, freeze, verify
# --------------------------------------------------------------------------- #


def build_protocol(
    *,
    target_fdf: Path = TARGET_FDF,
    positions_xyz: Path = DEFAULT_POSITIONS_XYZ,
    neutrality_estimate: Path = DEFAULT_NEUTRALITY_ESTIMATE,
    certification_dir: Path = DEFAULT_MATBG_CERTIFICATION_DIR,
    basis_response_report: Path = DEFAULT_BASIS_RESPONSE,
    go6_report: Path = DEFAULT_GO6_REPORT,
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
    }
    gates = preconditions(gate_paths)
    review = physics_review(geometry=geometry, gates=gates, k_selection_measured=False)

    blocking = [
        f"{row['name']}: {row['status']}" for row in gates if row["required_before_execution"] and row["status"] != "PASS"
    ]
    blocking.append("k_selection: no live electronic spectrum measured yet at any candidate k "
                     "(rule and candidates are frozen; the measurement itself is deferred to a "
                     "resource-preflight-gated execution step)")
    blocking.append("moire_hexagonal_convention_checked: the K/M fractional labels are the "
                     "graphene convention carried over, not yet verified against this structure's "
                     "own reciprocal lattice")

    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "ticket": TICKET,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_content_sha256": None,
        "experiment": {
            "system": "rigid magic-angle TBG (31, 30), materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf",
            "q_fractional": [0.0, 0.0, 0.0],
            "q_label": "Gamma",
            "k_plus_q_equals_k": True,
            "geometry": geometry,
            "publication_label_if_validated": "rigid_tbg_model",
            "forbidden_publication_labels": ["quantitative_relaxed_matbg_prediction"],
        },
        "phonon": {
            "status": "no PhononProvider certified for this geometry (GO-8a: NO-GO-8a as of S39)",
            "candidate_sectors": {
                name: directions[name].to_dict()
                for name in (*RESULT_DIRECTION_NAMES, *VALIDATION_DIRECTION_NAMES, *CALIBRATION_DIRECTION_NAMES)
            },
            "artifact_kind_of_candidates": "synthetic_test_displacement",
            "artifact_kind_forbidden_until_go8a": "physical_phonon",
        },
        "electronic": {
            "window": {
                "states_below": WINDOW_STATES_BELOW,
                "states_above": WINDOW_STATES_ABOVE,
                "rule": "same shape as graphene's window; the occupied index comes from the "
                "measured neutrality estimate, never from where a gap happens to be",
            },
            "occupations": occupations,
            "degeneracy_tol_ev": DEGENERACY_TOL_EV,
            "min_window_gap_ev": MIN_WINDOW_GAP_EV,
            "k_candidates": list(K_CANDIDATES),
            "k_degenerate": K_DEGENERATE,
            "k_calibration": list(K_CALIBRATION),
            "k_selection_rule": f"first candidate with an identity little group whose window "
            f"spacings and window-to-outside gaps all exceed {MIN_WINDOW_GAP_EV} eV -- identical "
            f"rule to preregister_graphene_gamma_epc.select_result_k, applied here to real "
            f"measured spectra only at execution time",
            "k_selection_status": "pending_measurement",
            "subspace_machinery": "Comparison/scripts/epc_subspaces.py (GO-6): principal angles, "
            "singular values and projector distances. No band index is a claim.",
        },
        "references": {
            "paths": list(REFERENCE_PATHS),
            "siesta_reference_excluded_reason": SIESTA_REFERENCE_EXCLUDED,
            "basis_response": BASIS_RESPONSE_CONTRACT,
            "formalism_id": FORMALISM_ID,
        },
        "splits": {
            **{name: {**split, "directions": list(split["directions"]), "k_labels": list(split["k_labels"]),
                      "fc_ranges": list(split["fc_ranges"])}
               for name, split in SPLITS.items()},
            "disjoint_in": "(direction, k) pairs and directions",
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
        "sensitivity": list(SENSITIVITY_AXES),
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
    print(f"  candidate sectors     : {list(protocol['phonon']['candidate_sectors'])}")
    print(f"  physics review        : {protocol['physics_review']['status']}")
    print(f"  ready to execute      : {protocol['ready_to_execute']}")
    for reason in protocol["blocking"]:
        print(f"    blocked by: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
