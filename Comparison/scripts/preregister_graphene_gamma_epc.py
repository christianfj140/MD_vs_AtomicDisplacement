#!/usr/bin/env python3
"""C19-C20 / E-F_001-S23: the graphene Gamma EPC experiment, pre-registered.

Everything that could be chosen *after* seeing ``g`` is chosen here, written to
one file and hashed. The perturbation, the electronic states, the three
reference paths, the split that keeps calibration out of the result, the three
tolerances, the PASS/FAIL rules and what may be claimed under each outcome are
frozen **before** ``compute_epc_matrix_elements.py`` exists; C20 then either
reproduces the frozen hash or its report is not a pre-registered result.

What is frozen, and why each choice is not free:

``mode``
    The Gamma ``E_2g`` doublet of C18, from the ``sc5x5x1`` ASR-corrected
    artifact -- the range whose IFC tail is 1.3e-3 and which C18 gated against
    ``sc1x1x1`` *at the same effective k mesh* (0.055% on the resolved
    branches). A degenerate doublet has no canonical member, so one is fixed by
    a rule and not by an index: the member of the 2D span with maximum overlap
    on the mass-weighted bond-stretch pattern of the cell
    (:func:`bond_reference_pattern`), its partner being the orthogonal
    complement inside the same span. Band 4 vs band 5 is the solver's gauge;
    "the one that stretches the bond" is not.

``k``
    Stage 1 is a *generic* k: no mirror, no rotation axis, so no matrix element
    is forced to zero by symmetry and a disagreement cannot hide inside a
    protected zero. It is selected from an ordered candidate list by a frozen
    rule (first candidate whose window spacings all exceed
    :data:`MIN_WINDOW_GAP_EV`) and then stored *by value* together with the
    spectrum that selected it. Stage 2 is K, where the Dirac doublet turns the
    same measurement into a subspace one, read only through GO-6 invariants.

``tolerances``
    ``tau_num``/``tau_backend``/``tau_model`` are defined **inside the window's
    subspace**: every one of them is a norm of ``C_f^dag dDelta C_i``, never a
    norm of ``dDelta`` alone. C14 measured 0.67 relative Frobenius on ``D_H``
    and said so in its own verdict: that number is not a tolerance on ``g`` and
    :data:`FORBIDDEN_TAU_MODEL_SOURCES` names it explicitly so a later reviewer
    can check that nobody converted it into a percentage of ``g``.

``splits``
    Calibration fixes floors and the transfer factor; validation tests whether
    the calibrated bound transfers; the result set is touched last and never
    enters a tolerance. The three sets are disjoint in ``(direction, k)`` and
    :func:`assert_splits_disjoint` refuses to build a protocol where they are
    not.

The script measures at freeze time only what must be frozen by value: the
spectrum at the candidate k points (from the certified equilibrium TSHS), the
gauge-fixed doublet (from the C18 artifact), the topology margin of the ``E_2g``
excursion, and the state of every upstream gate. It computes no ``g``: the
whole point is that it runs before the result exists.

Units: eV for energies and frequencies, Ang for displacements, eV/Ang for
``D_H`` and for ``g`` before the zero-point factor, eV after it.

Narrative and frozen values: ``docs/epc_c19c20_preregistracion_graphene_gamma.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fd_perturbation_space as fdp  # noqa: E402
from artifact_signature import input_signature_sha256  # noqa: E402
from epc_formalism import (  # noqa: E402
    FORMALISM_ID,
    FORMALISM_ID_IGNORE_OVERLAP,
    FORMALISM_ID_SYMMETRIC,
    REPRESENTATION_S_L_S_R,
    zero_point_amplitude_ang,
)

SCHEMA = "epc_preregistration_v1"
PROTOCOL_ID = "graphene_gamma_epc_v1"
TICKET = "C19-C20 / E-F_001-S23"
GATE = "GO-4"

DEFAULT_REFERENCE_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene"
DEFAULT_PHONON_ROOT = REPO_ROOT / "Comparison/results/epc/phonons/graphene"
DEFAULT_CERTIFICATION_DIR = REPO_ROOT / "Comparison/results/epc/certification/graphene"
DEFAULT_BASIS_RESPONSE = (
    REPO_ROOT / "Comparison/results/epc/basis_response/graphene/basis_response_certification.json"
)
DEFAULT_CHECKPOINT_ERROR = (
    REPO_ROOT
    / "Comparison/results/epc/checkpoint_derivative_error/graphene/checkpoint_derivative_error.json"
)
DEFAULT_GO6_REPORT = REPO_ROOT / "Comparison/results/epc/subspaces/go6_subspace_metrics.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/preregistration/graphene"
PROTOCOL_NAME = "epc_graphene_gamma_protocol.json"

# Where C19/C20 will write. Anything already there must cite this protocol's
# hash, or the "pre" in pre-registration is a claim and not a fact.
DEFAULT_RESULT_ROOT = REPO_ROOT / "Comparison/results/epc/gamma_epc/graphene"

# The FC range whose modes are the result, and the one C18 gated it against at
# a fixed effective k mesh. sc3x3x1 ran at a different mesh, so its 30 cm^-1
# offset is k sampling (the Kohn anomaly needs K in the mesh), not IFC range.
RESULT_FC_RANGE = "sc5x5x1"
RANGE_SENSITIVITY_PARTNER = "sc1x1x1"
RESULT_ASR_POLICY = "corrected"

# --------------------------------------------------------------------------- #
# Frozen choices
# --------------------------------------------------------------------------- #

# Ordered; the first candidate that satisfies the rule wins stage 1, the next
# one becomes the validation k. Fractional coordinates of the 60-degree
# hexagonal primitive cell of materials/graphene.
K_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "label": "k_generic_1",
        "k": (0.29, 0.11, 0.0),
        "little_group": "identity",
        "role": "stage_1_non_degenerate_candidate",
    },
    {
        "label": "k_generic_2",
        "k": (0.21, 0.37, 0.0),
        "little_group": "identity",
        "role": "stage_1_non_degenerate_candidate",
    },
    {
        "label": "M",
        "k": (0.5, 0.0, 0.0),
        "little_group": "C2v",
        "role": "calibration_only__high_symmetry",
    },
)

# K of the same cell: the Dirac degeneracy stage 2 needs. Not a candidate for
# stage 1 by construction.
K_DEGENERATE = {"label": "K", "k": (2.0 / 3.0, 1.0 / 3.0, 0.0), "little_group": "C3v"}

# Calibration k points. Disjoint from the result set: no tolerance may be fitted
# where the result is measured.
K_CALIBRATION = (
    {"label": "gamma", "k": (0.0, 0.0, 0.0), "little_group": "D6h"},
    {"label": "M", "k": (0.5, 0.0, 0.0), "little_group": "C2v"},
)

# Two states below and two above the neutrality point. q = Gamma, so k + q = k
# and one window serves both indices of g_mn.
WINDOW_STATES_BELOW = 2
WINDOW_STATES_ABOVE = 2

# Graphene: 2 C x 4 valence electrons, unpolarised.
ELECTRON_COUNT = 8
SPIN_DEGENERACY = 2
MU_CONVENTION = "exported_hamiltonian_fermi_zero"

# Stage 1 is only non-degenerate if the window's spacings clear this. It is
# 50x the 1e-3 eV at which C14C clusters states, which is itself the scale at
# which the SCF grid breaks the symmetry.
MIN_WINDOW_GAP_EV = 0.05
DEGENERACY_TOL_EV = 1e-3

# Amplitudes of the collective E2g stencil: the sweep GO-2 certified, reused so
# the result direction inherits a plateau protocol that already has a floor.
DELTA_SWEEP_ANG = (0.005, 0.01, 0.02)

# A result whose signal does not clear its own numerical floor by this factor is
# INCONCLUSIVE, never PASS. (Derivative campaigns in this repository have been
# burnt by an SNR below 1 before.)
MIN_SIGNAL_TO_FLOOR = 10.0

# JVP vs frozen central difference of the *same* model: the frozen side has its
# own FD floor, and agreement is judged against it, not against a round number.
# Same margin validate_graph2mat_jvp.py already uses (FD_NOISE_MARGIN).
BACKEND_NOISE_MARGIN = 10.0

# tau_model = TRANSFER_FACTOR x (worst calibrated subspace-relative error). Both
# numbers are C14's, kept identical so the two gates cannot drift apart.
TRANSFER_FACTOR = 1.5
ATTRIBUTION_MARGIN = 3.0

# A priori: what "the checkpoint is good enough for a quantitative g" means.
# Chosen before measuring, and deliberately not the 0.05 of the derivative-level
# gate reinterpreted -- this one is a relative error on the g block itself.
G_ADEQUACY_THRESHOLD = 0.05

FORBIDDEN_TAU_MODEL_SOURCES = (
    "relative Frobenius of D_H (C14 worst_relative_frobenius)",
    "MAE or Frobenius of H/D_H matrix elements over all orbital pairs",
    "any global matrix norm converted into a percentage of g",
)

SPLITS: dict[str, dict[str, Any]] = {
    "calibration": {
        "purpose": "fix the FD floors, the delta plateau and the transfer factor. "
        "May be inspected freely; nothing measured here is a result.",
        "directions": ("atom0000_x", "translation_x"),
        "k_labels": ("gamma", "M"),
        "fc_ranges": (RANGE_SENSITIVITY_PARTNER, "sc3x3x1"),
    },
    "validation": {
        "purpose": "test whether the calibrated bound transfers to a direction and a k "
        "that took part in no fit. Used once, after the tolerances are frozen.",
        "directions": ("random_seed0",),
        "k_labels": ("k_generic_2",),
        "fc_ranges": (),
    },
    "result": {
        "purpose": "the pre-registered experiment. Evaluated last, and never used to "
        "set, widen or justify a tolerance.",
        "directions": ("e2g_bond_longitudinal", "e2g_bond_transverse"),
        "k_labels": ("k_generic_1", "K"),
        "fc_ranges": (RESULT_FC_RANGE,),
    },
}

# K appears in C14C, but only as the place its gauge-covariance check rotates a
# degenerate subspace: no g and no tolerance was fitted there. Recorded rather
# than hidden, because the disjointness rule below is about tolerance fitting.
SPLIT_NOTES = (
    "C14C evaluated the basis-response gauge check at Gamma and K. No g value and no "
    "tolerance of this protocol comes from those runs; K enters the result set as the "
    "degenerate stage and its coupling block has never been computed.",
    "The three FD directions of GO-2 were certified before this protocol existed. Two of "
    "them are the calibration set here precisely because their floors are already measured; "
    "the E2g directions are new and enter no floor.",
)

REFERENCE_PATHS: tuple[dict[str, Any], ...] = (
    {
        "path": "siesta_reference",
        "electronic_derivative_backend": "siesta",
        "derivative_method": "siesta_fc_dHS",
        "artifact": "*.dHSdR.nc of the FC branch, contracted onto the E2g direction",
        "internal_check": "explicit central FD of the same H and S from a +- TSHS pair "
        "along the same collective direction; the two must agree inside GO-2's tau_FD "
        "before this path is a reference for anything",
        "new_runs_required": "one +- SIESTA pair per E2g member per delta of the sweep "
        "(2 x 2 x 3 = 12 single-point runs, same physics fingerprint as the certified "
        "campaign), driven by run_epc_siesta_reference.py",
    },
    {
        "path": "graph2mat_jvp",
        "electronic_derivative_backend": "graph2mat",
        "derivative_method": "jvp",
        "artifact": "directional double-backward JVP on the frozen neighbour topology",
        "internal_check": "equals the coordinate-JVP combination sum_i v_i J e_i to "
        "float64 roundoff (GO-3, 1e-11 relative Frobenius)",
        "new_runs_required": "none beyond the JVP itself; the checkpoint is C14's",
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

# Every path is contracted through the same basis response and the same
# formalism. A path that skips it is a diagnostic with a name, not a variant.
BASIS_RESPONSE_CONTRACT = {
    "formalism_id": FORMALISM_ID,
    "representation": REPRESENTATION_S_L_S_R,
    "provider": "Comparison/scripts/epc_basis_response.py (C14B), gated by C14C",
    "diagnostic_variants_allowed": (FORMALISM_ID_SYMMETRIC, FORMALISM_ID_IGNORE_OVERLAP),
    "diagnostic_variants_are_never_the_result": True,
    "unresolved_term_blocking_production": "intra_atomic_basis_response (memo A2): "
    "dS/dR is identically blind to it, so it needs the analytic one-centre integral over "
    "the .ion.xml radials. Until it exists, C14C is NO_GO and no PAO-covariant response may be built.",
}

TOLERANCES: dict[str, dict[str, Any]] = {
    "tau_num": {
        "what": "numerical floor of each path: finite difference, plateau curvature, dtype",
        "contracted_in": "the S-orthonormal window C_W at the k being measured",
        "definition": "tau_num = max over plateau amplitude pairs (a, b) of "
        "|| C_W^dag (Delta(delta_a) - Delta(delta_b)) C_W ||_F, with C_W the S-orthonormal "
        "window at the k being measured. Same construction as GO-2's tau_FD, evaluated "
        "after the contraction instead of before it.",
        "measured_from": "the result direction's own delta sweep; GO-2's tau_FD is the "
        "prior, not the value",
        "unit": "eV/Ang (eV after the zero-point factor)",
        "pass_rule": "every check that compares two objects is judged against tau_num; and "
        "the result is INCONCLUSIVE unless "
        "|| C_W^dag Delta C_W ||_F >= MIN_SIGNAL_TO_FLOOR * tau_num",
    },
    "tau_backend": {
        "what": "JVP vs frozen central difference of the same Graph2Mat model",
        "contracted_in": "the S-orthonormal window C_W at the k being measured",
        "definition": "tau_backend = || C_W^dag (Delta_JVP - Delta_frozen) C_W ||_F",
        "pass_rule": "tau_backend <= BACKEND_NOISE_MARGIN * tau_num(graph2mat_frozen). "
        "The two objects are the same function of the same weights, so anything above the "
        "frozen side's own floor is a bug, not a tolerance.",
        "unit": "eV/Ang",
    },
    "tau_model": {
        "what": "Graph2Mat vs certified SIESTA, as an error on g and not on a matrix",
        "contracted_in": "delta_g = C_f^dag delta_Delta C_i, the same window that carries g",
        "definition": "propagate the derivative-level residual through the same electronic "
        "subspace: delta_g = C_f^dag (Delta_G2M - Delta_SIESTA) C_i, and take the relative "
        "subspace error r = ||delta_g||_F / ||C_f^dag Delta_SIESTA C_i||_F",
        "calibration": "tau_model = TRANSFER_FACTOR * max(r) over the calibration split "
        "(directions x k of SPLITS['calibration']), excluding null directions, whose "
        "denominator is zero by construction and which are judged against tau_num instead",
        "validation": "the validation split must satisfy r <= tau_model before tau_model is "
        "applied to the result. If it does not, the bound does not transfer and the verdict "
        "is NO_GO for the model path -- widening tau_model post hoc is forbidden",
        "adequacy": f"the checkpoint is quantitative for g only if tau_model <= "
        f"{G_ADEQUACY_THRESHOLD}",
        "attribution": "a disagreement is charged to the model only if it exceeds "
        "ATTRIBUTION_MARGIN * tau_num and every numerical check has passed",
        "forbidden_sources": FORBIDDEN_TAU_MODEL_SOURCES,
        "unit": "dimensionless (relative), applied to the g block",
    },
}

CHECKS: tuple[dict[str, Any], ...] = (
    {
        "id": "siesta_fc_equals_explicit_fd",
        "stage": "derivative",
        "applies_to": ("siesta_reference",),
        "pass_rule": "discrepancy(D_H, D_S) between the FC dHSdR contraction and the "
        "explicit +- FD of the same direction is inside the plateau's tau_FD, with no "
        "index, image, phase or unit mismatch (GO-2 protocol, rerun on the E2g directions)",
    },
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
        "applies_to": ("siesta_reference", "graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "no neighbour pair crosses its cutoff under +- max(delta) v for either "
        "E2g member; measured at freeze time and re-asserted at run time",
    },
    {
        "id": "generalized_hellmann_feynman_diagonal",
        "stage": "epc",
        "applies_to": ("siesta_reference", "graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "d(eps_n) from C_n^dag (D_H - eps_n D_S) C_n equals the central FD of "
        "the eigenvalue itself, inside tau_num. Uses D_S and not the split, so it is the one "
        "identity that holds without the unresolved intra-atomic term",
    },
    {
        "id": "uniform_translation_null",
        "stage": "epc",
        "applies_to": ("siesta_reference", "graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "g of the uniform-translation direction is <= tau_num in absolute "
        "value at every k of the result set (acoustic control; relative metrics are "
        "undefined for a null direction and are not reported for it)",
    },
    {
        "id": "electronic_gauge_invariance",
        "stage": "epc",
        "applies_to": ("siesta_reference", "graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "under a block-diagonal random unitary on the window "
        "(epc_subspaces.random_gauge_rotation) the singular values and Frobenius norm of "
        "every cluster block move by less than GO-6's metric tolerance, while "
        "cross_cluster_mixing moves them far more (the check must have teeth)",
    },
    {
        "id": "phonon_doublet_gauge_invariance",
        "stage": "epc",
        "applies_to": ("siesta_reference", "graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "rotating the E2g doublet by a random 2x2 orthogonal matrix leaves "
        "the Frobenius norm and the singular values of the two-mode g block invariant to "
        "1e-10 relative; the gauge-fixed member is a label for reporting, never the claim",
    },
    {
        "id": "symmetry_of_the_doublet_at_K",
        "stage": "epc",
        "applies_to": ("siesta_reference",),
        "pass_rule": "at K the two E2g members give equal doublet-block Frobenius norms to "
        "within the measured symmetry-breaking floor of the SCF (the K degeneracy split "
        "recorded at freeze time); an inequality above that floor is a broken convention, "
        "not physics",
    },
    {
        "id": "units_round_trip",
        "stage": "epc",
        "applies_to": ("siesta_reference", "graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "eV/Ang * Ang = eV end to end: the zero-point amplitude recomputed "
        "from (M, omega) reproduces the frozen value to 1e-12 relative, cm^-1 <-> eV "
        "round-trips through the C18 constant, and D_S/S_L/S_R are in 1/Ang",
    },
    {
        "id": "basis_response_present_and_required",
        "stage": "epc",
        "applies_to": ("siesta_reference", "graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "PAO-covariant response is built only from a basis-response artifact whose "
        "formalism_id and representation match BASIS_RESPONSE_CONTRACT; a missing artifact "
        "raises, and the degraded variants may only appear labelled as diagnostics",
    },
    {
        "id": "on_shell_antisymmetric_term",
        "stage": "epc",
        "applies_to": ("siesta_reference",),
        "pass_rule": "report the symmetric-form error restricted to the pairs the mode can "
        "connect (|eps_m - eps_n| <= hbar*omega of the result mode) against tau_num. C14C "
        "measured this only at C_row = C_col with no q; here it is measured with the real "
        "mode. No claim that A is negligible may be made without this number",
    },
    {
        "id": "intra_atomic_uncertainty_declared",
        "stage": "epc",
        "applies_to": ("siesta_reference", "graph2mat_jvp", "graph2mat_frozen"),
        "pass_rule": "the published g carries the C14C intra-atomic sensitivity times the "
        "magnitude of the unresolved term as an explicit uncertainty; if that term is still "
        "unresolved the result is a bound and is labelled as such",
    },
)

SENSITIVITY_AXES: tuple[dict[str, Any], ...] = (
    {
        "axis": "fc_range",
        "vary": f"{RESULT_FC_RANGE} vs {RANGE_SENSITIVITY_PARTNER} (same effective k mesh)",
        "tolerance": "relative change of ||g|| <= 0.02, the same tolerance C18 froze for "
        "the frequencies; the doublet spans differ by 2.1e-8 rad, so any larger move in g "
        "is not the phonon",
    },
    {
        "axis": "asr_policy",
        "vary": "corrected vs raw IFC",
        "tolerance": "reported side by side, never merged; the raw acoustic residual is "
        "carried into the report so 'the acoustic modes are zero' is never a claim the "
        "correction smuggled in",
    },
    {
        "axis": "delta",
        "vary": "the three amplitudes of DELTA_SWEEP_ANG",
        "tolerance": "plateau; a direction without a plateau is INCONCLUSIVE",
    },
    {
        "axis": "formalism_variant",
        "vary": f"{FORMALISM_ID} vs {FORMALISM_ID_SYMMETRIC} vs {FORMALISM_ID_IGNORE_OVERLAP}",
        "tolerance": "diagnostic only: the difference is the size of the term each variant "
        "drops, and it is reported, not tolerated",
    },
    {
        "axis": "window_size",
        "vary": "the frozen 2+2 window vs 3+3",
        "tolerance": "the gauge-invariant metrics of the common sub-block must not move by "
        "more than tau_num; a result that depends on where the window was cut is not a "
        "result",
    },
)

CLAIM_LADDER: tuple[dict[str, Any], ...] = (
    {
        "outcome": "every check passes and tau_model <= adequacy",
        "claim": "graphene_gamma_epc_validated (GO-4 PASS); Graph2Mat g is quantitative "
        "for this mode at this k",
    },
    {
        "outcome": "numerical and formalism checks pass, tau_model > adequacy",
        "claim": "siesta_reference_g_validated__graph2mat_g_not_quantitative; opens the "
        "fine-tuning ticket, which stays closed otherwise",
    },
    {
        "outcome": "any derivative-level or formalism check fails",
        "claim": "NO_GO. The failure is attributed to the failing layer and never to the "
        "model: 'if 1-5 fail, the ML is not to blame'",
    },
    {
        "outcome": "signal below MIN_SIGNAL_TO_FLOOR * tau_num",
        "claim": "INCONCLUSIVE; the experiment did not resolve g, which is not a PASS and "
        "not a FAIL",
    },
    {
        "outcome": "intra_atomic_basis_response still unresolved",
        "claim": "no quantitative g at all. What may be published is the three-path "
        "derivative validation plus a bound on g with the intra-atomic sensitivity as its "
        "width",
    },
)


class PreregistrationError(RuntimeError):
    """The protocol cannot be built, or a frozen protocol failed verification."""


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def _pairs(split: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {(d, k) for d in split["directions"] for k in split["k_labels"]}


def assert_splits_disjoint(splits: Mapping[str, Mapping[str, Any]] = SPLITS) -> None:
    """No ``(direction, k)`` may serve two purposes: that is what leakage is."""
    names = list(splits)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = _pairs(splits[first]) & _pairs(splits[second])
            if shared:
                raise PreregistrationError(
                    f"splits {first!r} and {second!r} share {sorted(shared)}; a tolerance "
                    "fitted where the result is measured is leakage"
                )
            shared_directions = set(splits[first]["directions"]) & set(splits[second]["directions"])
            if shared_directions:
                raise PreregistrationError(
                    f"splits {first!r} and {second!r} share directions {sorted(shared_directions)}"
                )


def assert_selection_outside_fitted_splits(selected: str) -> None:
    """The k the rule picks must be the declared result k, and only that.

    ``assert_splits_disjoint`` only sees the static table. The stage-1 k is
    chosen at freeze time by a rule over measured spectra, so if the first
    candidate ever failed it, the rule would silently promote the *validation*
    k to the result -- fitting and measuring in the same place. That is leakage
    arriving through the back door, and it stops here.
    """
    for name in ("calibration", "validation"):
        if selected in SPLITS[name]["k_labels"]:
            raise PreregistrationError(
                f"the frozen rule selected k {selected!r}, which the {name} split already "
                "claims: a tolerance fitted where the result is measured is leakage"
            )
    if selected not in SPLITS["result"]["k_labels"]:
        raise PreregistrationError(
            f"the frozen rule selected k {selected!r}, which the result split does not "
            f"declare ({sorted(SPLITS['result']['k_labels'])})"
        )


# --------------------------------------------------------------------------- #
# The mode: fixing one member of a degenerate doublet, reproducibly
# --------------------------------------------------------------------------- #


def minimum_image_vector(
    origin: Sequence[float], target: Sequence[float], cell: Any
) -> np.ndarray:
    """``target - origin`` folded to the shortest periodic image."""
    cell = np.asarray(cell, dtype=np.float64)
    raw = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    best = raw
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                candidate = raw + i * cell[0] + j * cell[1] + k * cell[2]
                if np.linalg.norm(candidate) < np.linalg.norm(best):
                    best = candidate
    return best


def bond_reference_pattern(
    positions_ang: Any, cell_ang: Any, masses_amu: Any
) -> dict[str, Any]:
    """The mass-weighted bond-stretch pattern of a two-atom cell.

    Atom 0 moves against the bond, atom 1 along it, so the pattern is the
    optical (opposed-motion) in-plane displacement whose direction is fixed by
    the geometry alone -- no eigenvector, no index, no solver gauge. Returned
    mass-weighted and unit-norm, i.e. in the space the phonon eigenvectors live
    in (``e = sqrt(M) u``, ``sum |e|^2 = 1``).
    """
    positions = np.asarray(positions_ang, dtype=np.float64)
    masses = np.asarray(masses_amu, dtype=np.float64)
    if positions.shape != (2, 3) or masses.shape != (2,):
        raise PreregistrationError(
            "the bond reference is defined for the two-atom cell of graphene; "
            f"got positions {positions.shape} and masses {masses.shape}"
        )
    bond = minimum_image_vector(positions[0], positions[1], cell_ang)
    unit = bond / np.linalg.norm(bond)
    displacement = np.stack([-unit, unit])
    weighted = displacement * np.sqrt(masses)[:, None]
    return {
        "definition": "atom 0 along -b, atom 1 along +b, with b the minimum-image bond "
        "vector; mass-weighted and normalised",
        "bond_vector_ang": bond.tolist(),
        "pattern_mass_weighted": (weighted / np.linalg.norm(weighted)).tolist(),
        "in_plane_normal": np.cross([0.0, 0.0, 1.0], unit).tolist(),
    }


def gauge_fix_doublet(
    eigenvectors: Any,
    branches: Sequence[int],
    reference: Mapping[str, Any],
    *,
    labels: Sequence[str] = ("e2g_bond_longitudinal", "e2g_bond_transverse"),
) -> dict[str, Any]:
    """Rotate a degenerate pair onto the frozen reference pattern.

    The first member is the unit vector of the doublet's span with maximum
    overlap on ``reference``; the second is its orthogonal complement *inside
    the same span*, with its sign fixed by the in-plane normal. Any other
    2x2 rotation spans the same subspace and must give the same physics --
    which is exactly what ``phonon_doublet_gauge_invariance`` checks.
    """
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    if len(branches) != 2:
        raise PreregistrationError(f"the E2g sector must be a doublet; got {list(branches)}")
    basis = np.stack([vectors[int(b)].reshape(-1) for b in branches])
    target = np.asarray(reference["pattern_mass_weighted"], dtype=np.float64).reshape(-1)

    coefficients = basis @ target
    overlap = float(np.linalg.norm(coefficients))
    if overlap < 1e-12:
        raise PreregistrationError(
            "the bond-stretch reference is orthogonal to the E2g doublet; the gauge rule "
            "cannot fix a member and the protocol must not guess one"
        )
    longitudinal = (coefficients @ basis) / overlap
    transverse = (np.array([-coefficients[1], coefficients[0]]) @ basis) / overlap
    normal = np.asarray(reference["in_plane_normal"], dtype=np.float64)
    if float(transverse.reshape(-1, 3)[0] @ normal) < 0.0:
        transverse = -transverse

    members = []
    for label, vector in zip(labels, (longitudinal, transverse)):
        shaped = vector.reshape(-1, 3)
        members.append(
            {
                "label": label,
                "eigenvector_mass_weighted": shaped.tolist(),
                "overlap_with_reference": float(shaped.reshape(-1) @ target),
                "eigenvector_sha256": _sha256_of(shaped.tolist()),
            }
        )
    return {
        "branches": [int(b) for b in branches],
        "gauge_rule": "member 1 maximises the overlap with the mass-weighted bond-stretch "
        "pattern inside the doublet span; member 2 is its orthogonal complement in that "
        "span, signed by the in-plane normal",
        "reference": dict(reference),
        "reference_lies_in_the_doublet": bool(overlap >= 0.999),
        "reference_projection_norm": overlap,
        "members": members,
    }


def mode_direction(
    member: Mapping[str, Any], masses_amu: Any, *, name: str
) -> fdp.Direction:
    """The unit-Frobenius cartesian displacement pattern of one mode member.

    ``u = e / sqrt(M)`` normalised: the perturbation-space convention of C08, so
    ``delta`` means the same amplitude here as for every other direction in the
    repository. The physical (zero-point) amplitude enters afterwards, through
    :func:`epc_formalism.zero_point_amplitude_ang`, and never through ``v``.
    """
    weighted = np.asarray(member["eigenvector_mass_weighted"], dtype=np.float64)
    masses = np.asarray(masses_amu, dtype=np.float64)
    return fdp.collective(weighted / np.sqrt(masses)[:, None], name=name)


# --------------------------------------------------------------------------- #
# The electronic states
# --------------------------------------------------------------------------- #


def window_indices(
    eigenvalues: Any, *, below: int = WINDOW_STATES_BELOW, above: int = WINDOW_STATES_ABOVE
) -> list[int]:
    """The ``below + above`` states straddling the neutrality point.

    The occupied count comes from the declared electron count, not from where a
    gap happens to be: an artifact that counts its own bands would move with the
    result.
    """
    occupied = ELECTRON_COUNT // SPIN_DEGENERACY
    values = np.asarray(eigenvalues, dtype=np.float64)
    if occupied - below < 0 or occupied + above > values.size:
        raise PreregistrationError(
            f"a {below}+{above} window around band {occupied} does not fit in "
            f"{values.size} states"
        )
    return list(range(occupied - below, occupied + above))


def spectrum_report(label: str, k: Sequence[float], eigenvalues: Any) -> dict[str, Any]:
    """Everything the k rule needs, measured: the window and its spacings."""
    values = np.asarray(eigenvalues, dtype=np.float64)
    indices = window_indices(values)
    window = values[indices]
    outside = min(
        float(window[0] - values[indices[0] - 1]) if indices[0] > 0 else np.inf,
        float(values[indices[-1] + 1] - window[-1]) if indices[-1] + 1 < values.size else np.inf,
    )
    return {
        "label": label,
        "k_fractional": [float(component) for component in k],
        "eigenvalues_ev": [float(value) for value in values],
        "window_indices": indices,
        "window_eigenvalues_ev": [float(value) for value in window],
        "min_window_gap_ev": float(np.min(np.diff(window))),
        "min_gap_to_states_outside_window_ev": outside,
    }


def select_result_k(
    spectra: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The frozen rule: first generic candidate whose window is fully resolved.

    Both the winner and every candidate it beat are recorded, so the choice can
    be audited without rerunning anything.
    """
    by_label = {row["label"]: row for row in spectra}
    trace = []
    winner = None
    for candidate in candidates:
        row = by_label.get(candidate["label"])
        if row is None:
            raise PreregistrationError(f"no measured spectrum for candidate {candidate['label']!r}")
        generic = candidate["little_group"] == "identity"
        resolved = (
            row["min_window_gap_ev"] >= MIN_WINDOW_GAP_EV
            and row["min_gap_to_states_outside_window_ev"] >= MIN_WINDOW_GAP_EV
        )
        accepted = bool(generic and resolved and winner is None)
        trace.append(
            {
                "label": candidate["label"],
                "little_group": candidate["little_group"],
                "generic": generic,
                "min_window_gap_ev": row["min_window_gap_ev"],
                "resolved": bool(resolved),
                "accepted": accepted,
                # The intrinsic reason first: a candidate that would have been
                # rejected anyway must say so, or the trace credits it with a
                # near miss it never had.
                "rejected_because": (
                    None
                    if accepted
                    else "high-symmetry little group: a symmetry-protected zero could hide "
                    "a disagreement"
                    if not generic
                    else f"window spacing below {MIN_WINDOW_GAP_EV} eV"
                    if not resolved
                    else "already selected"
                ),
            }
        )
        if accepted:
            winner = candidate["label"]
    if winner is None:
        raise PreregistrationError(
            "no candidate k satisfies the frozen rule; the rule is not to be relaxed after "
            "looking at the spectra"
        )
    assert_selection_outside_fitted_splits(winner)
    return {
        "rule": f"first candidate with an identity little group whose window spacings and "
        f"window-to-outside gaps all exceed {MIN_WINDOW_GAP_EV} eV",
        "selected": winner,
        "candidates": trace,
    }


def siesta_spectra(
    tshs: Path, points: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Eigenvalues of the certified equilibrium pencil at each k (needs sisl)."""
    try:
        import sisl
    except ImportError as error:  # pragma: no cover - environment dependent
        raise PreregistrationError(
            f"sisl is required to freeze the k selection by value: {error}"
        ) from error
    hamiltonian = sisl.get_sile(str(tshs)).read_hamiltonian()
    return [
        spectrum_report(point["label"], point["k"], hamiltonian.eigh(k=list(point["k"])))
        for point in points
    ]


# --------------------------------------------------------------------------- #
# Upstream gates
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _precondition(
    name: str, path: Path, payload: Mapping[str, Any] | None, *, verdict: str | None, required: bool
) -> dict[str, Any]:
    if payload is None:
        status = "MISSING"
    elif verdict is None:
        status = "PRESENT"
    else:
        status = "PASS" if verdict == "PASS" else str(verdict)
    return {
        "name": name,
        "path": str(path),
        "status": status,
        "verdict": verdict,
        "required_before_execution": bool(required),
        "sha256": _file_sha256(path),
    }


def preconditions(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    """State of every gate this experiment consumes. Missing is a state, not an error."""
    go2 = _read_json(paths["go2"])
    c18 = _read_json(paths["c18"])
    c14c = _read_json(paths["c14c"])
    c14 = _read_json(paths["c14"])
    go6 = _read_json(paths["go6"])

    rows = [
        _precondition("GO-2 siesta derivative", paths["go2"], go2,
                      verdict=(go2 or {}).get("verdict"), required=True),
        _precondition("C18 gamma phonons", paths["c18"], c18,
                      verdict=(c18 or {}).get("verdict"), required=True),
        _precondition("C14C basis response", paths["c14c"], c14c,
                      verdict=(c14c or {}).get("verdict"), required=True),
        _precondition("GO-6 subspace metrics", paths["go6"], go6,
                      verdict=(go6 or {}).get("verdict"), required=True),
        _precondition("C14 checkpoint derivative error", paths["c14"], c14,
                      verdict=None, required=False),
    ]
    if c14 is not None:
        rows[-1]["decision"] = (c14.get("verdict") or {}).get("decision")
        rows[-1]["go3_passed"] = bool(
            ((c14.get("internal_validation_go3") or {}).get("summary") or {}).get("passed")
        )
        rows[-1]["derivative_level_relative_frobenius"] = (c14.get("verdict") or {}).get(
            "worst_relative_frobenius"
        )
        rows[-1]["not_a_tolerance_on_g"] = (c14.get("verdict") or {}).get("not_a_tolerance_on_g")
    if c14c is not None:
        rows[2]["provider_checks_passed"] = bool(c14c.get("provider_checks_passed"))
    return rows


def geometry_consistency(
    reference_manifest: Mapping[str, Any], phonon_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """The phonon and the derivative must be derivatives of the *same* structure."""
    fields = {
        "material_fdf_sha256": (
            reference_manifest.get("material_fdf_sha256"),
            phonon_manifest.get("material_fdf_sha256"),
        ),
        "base_ang_fdf_sha256": (
            reference_manifest.get("base_ang_fdf_sha256"),
            phonon_manifest.get("base_ang_fdf_sha256"),
        ),
        "orbital_contract_hash": (
            (reference_manifest.get("orbital_contract") or {}).get("orbital_contract_hash"),
            (phonon_manifest.get("orbital_contract") or {}).get("orbital_contract_hash"),
        ),
        "basis_contract_hash": (
            (reference_manifest.get("orbital_contract") or {}).get("basis_contract_hash"),
            (phonon_manifest.get("orbital_contract") or {}).get("basis_contract_hash"),
        ),
        "siesta_binary_sha256": (
            (reference_manifest.get("siesta_runtime_ref") or {}).get("binary_sha256"),
            (phonon_manifest.get("siesta_runtime_ref") or {}).get("binary_sha256"),
        ),
    }
    rows = [
        {"field": name, "derivative_campaign": left, "phonon_campaign": right,
         "identical": left is not None and left == right}
        for name, (left, right) in fields.items()
    ]
    return {"fields": rows, "consistent": all(row["identical"] for row in rows)}


# --------------------------------------------------------------------------- #
# The physics review, bound to evidence
# --------------------------------------------------------------------------- #


def error_budget_report(selected_k: str) -> dict[str, Any]:
    """Is the budget a budget, or a promise?

    Approving it means checking the four things that make ``tau_model`` mean
    something later: all three tolerances defined, every one of them a norm
    *inside* the electronic subspace, ``tau_model`` calibrated and transfer
    tested on splits that do not touch the result, and its forbidden sources
    written down so the C14 matrix-level number cannot be recycled into a
    percentage of ``g``.
    """
    names = ("tau_num", "tau_backend", "tau_model")
    defined = {name: set(TOLERANCES[name]) >= {"what", "definition", "unit"} for name in names}
    contracted = {name: bool(TOLERANCES[name].get("contracted_in")) for name in names}
    model = set(TOLERANCES["tau_model"]) >= {
        "calibration",
        "validation",
        "adequacy",
        "attribution",
        "forbidden_sources",
    }
    try:
        assert_splits_disjoint()
        assert_selection_outside_fitted_splits(selected_k)
    except PreregistrationError as error:
        leakage: str | None = str(error)
    else:
        leakage = None
    return {
        "tolerances_defined": defined,
        "tolerances_contracted_in_the_subspace": contracted,
        "tau_model_calibration_contract_complete": model,
        "leakage": leakage,
        "complete": all(defined.values())
        and all(contracted.values())
        and model
        and bool(FORBIDDEN_TAU_MODEL_SOURCES)
        and leakage is None,
    }


def physics_review(
    *,
    mode: Mapping[str, Any],
    k_selection: Mapping[str, Any],
    spectra: Mapping[str, Mapping[str, Any]],
    c18: Mapping[str, Any] | None,
    gates: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Approve mode, k, references and error budget -- or record why not.

    Every decision is a condition on measured evidence, so the approval can be
    re-derived instead of trusted. The human slot is separate and optional: it
    signs the same hash, it does not replace these conditions.
    """
    identification = ((c18 or {}).get("range_stability") or {})
    doublet_stable = any(
        check.get("check") == "chosen_subspaces_are_stable_in_the_ifc_range" and check.get("passed")
        for check in identification.get("checks", [])
    )
    degenerate = spectra.get(K_DEGENERATE["label"], {})
    split = (
        float(np.min(np.abs(np.diff(np.asarray(degenerate["window_eigenvalues_ev"])))))
        if degenerate
        else None
    )
    gate_status = {row["name"]: row["status"] for row in gates}
    budget = error_budget_report(k_selection["selected"])
    items = [
        {
            "item": "mode",
            "decision": "approved"
            if (c18 or {}).get("verdict") == "PASS"
            and doublet_stable
            and mode["reference_lies_in_the_doublet"]
            else "blocked",
            "evidence": {
                "c18_verdict": (c18 or {}).get("verdict"),
                "doublet_stable_in_ifc_range": doublet_stable,
                "reference_projection_norm": mode["reference_projection_norm"],
                "frequency_ev": mode["frequency_ev"],
            },
            "basis": "the Gamma E2g of graphene is the mode whose coupling is the physical "
            "benchmark of this system (Piscanec 2004, DOI 10.1103/PhysRevLett.93.185503); "
            "C18 identified it without imposing an eigenvector",
        },
        {
            "item": "k",
            "decision": "approved"
            if k_selection["selected"] is not None
            and spectra[k_selection["selected"]]["min_window_gap_ev"] >= MIN_WINDOW_GAP_EV
            else "blocked",
            "evidence": {
                "stage_1": k_selection["selected"],
                "stage_1_min_window_gap_ev": spectra[k_selection["selected"]]["min_window_gap_ev"],
                "stage_2": K_DEGENERATE["label"],
                "stage_2_degeneracy_split_ev": split,
            },
            "basis": "stage 1 isolates the operator where no symmetry protects a zero and "
            "no near-degeneracy makes the eigenvectors ill-conditioned; stage 2 repeats it "
            "on the Dirac doublet, read only through GO-6 invariants",
        },
        {
            "item": "references",
            "decision": "approved"
            if gate_status.get("GO-2 siesta derivative") == "PASS" and geometry["consistent"]
            else "blocked",
            "evidence": {
                "gates": gate_status,
                "geometry_consistent_across_campaigns": geometry["consistent"],
                "paths": [row["path"] for row in REFERENCE_PATHS],
            },
            "basis": "the SIESTA path is a reference only because GO-2 reproduced it "
            "against explicit finite differences; the two Graph2Mat paths are the same "
            "weights through two different differentiations, so their disagreement is a "
            "backend bug and never physics",
        },
        {
            "item": "error_budget",
            "decision": "approved" if budget["complete"] else "blocked",
            "evidence": {
                "tau_num": TOLERANCES["tau_num"]["definition"],
                "tau_backend": TOLERANCES["tau_backend"]["pass_rule"],
                "tau_model_propagation": TOLERANCES["tau_model"]["definition"],
                "forbidden_sources": list(FORBIDDEN_TAU_MODEL_SOURCES),
                "adequacy_threshold": G_ADEQUACY_THRESHOLD,
                "resolution_rule": f"signal >= {MIN_SIGNAL_TO_FLOOR} x tau_num",
                **budget,
            },
            "basis": "each tolerance is a norm in the same electronic subspace the result "
            "is contracted in, calibrated on a disjoint split and transfer-tested before "
            "it is applied; no global matrix norm becomes a percentage of g",
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_of(payload: Any) -> str:
    return input_signature_sha256({"payload": _json_safe(payload)})


def _file_sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Fields that may change without changing the protocol. Everything else is
# frozen, measurements included.
VOLATILE_FIELDS = ("generated_at", "frozen_content_sha256")


def freeze_hash(protocol: Mapping[str, Any]) -> str:
    """Hash of everything but the volatile fields: the pre-registration identity."""
    return input_signature_sha256(
        {key: _json_safe(value) for key, value in protocol.items() if key not in VOLATILE_FIELDS}
    )


def build_protocol(
    *,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    phonon_root: Path = DEFAULT_PHONON_ROOT,
    certification_dir: Path = DEFAULT_CERTIFICATION_DIR,
    basis_response_report: Path = DEFAULT_BASIS_RESPONSE,
    checkpoint_error_report: Path = DEFAULT_CHECKPOINT_ERROR,
    go6_report: Path = DEFAULT_GO6_REPORT,
    spectra_provider: Callable[[Path, Sequence[Mapping[str, Any]]], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Measure what must be frozen by value, and assemble the protocol."""
    assert_splits_disjoint()

    reference_manifest = _read_json(reference_root / "epc_siesta_reference_manifest.json")
    phonon_manifest = _read_json(phonon_root / "graphene_gamma_phonon_manifest.json")
    mode_payload = _read_json(
        phonon_root / "ranges" / RESULT_FC_RANGE / f"gamma_modes_asr_{RESULT_ASR_POLICY}.json"
    )
    if reference_manifest is None or phonon_manifest is None or mode_payload is None:
        raise PreregistrationError(
            "the protocol is frozen against real artifacts: the SIESTA reference manifest, "
            "the phonon manifest and the result-range modes must all exist"
        )

    sector = (mode_payload.get("sector_report") or {}).get("e2g_candidate") or {}
    if not sector.get("branches"):
        raise PreregistrationError(
            f"{RESULT_FC_RANGE} has no identified E2g doublet; C18 must identify the sector "
            "before it can be pre-registered"
        )

    reference = bond_reference_pattern(
        mode_payload["positions_ang"], mode_payload["cell_ang"], mode_payload["masses_amu"]
    )
    doublet = gauge_fix_doublet(
        mode_payload["eigenvectors_real"], sector["branches"], reference
    )
    frequencies = np.asarray(mode_payload["frequencies_ev"], dtype=np.float64)
    frequency_ev = float(np.mean(frequencies[list(sector["branches"])]))
    mass_amu = float(np.mean(np.asarray(mode_payload["masses_amu"], dtype=np.float64)))
    doublet["frequency_ev"] = frequency_ev
    doublet["frequency_spread_ev"] = float(sector.get("frequency_spread_ev", 0.0))
    doublet["zero_point_amplitude_ang"] = zero_point_amplitude_ang(mass_amu, frequency_ev)

    directions = [
        mode_direction(member, mode_payload["masses_amu"], name=member["label"])
        for member in doublet["members"]
    ]
    cutoff = float((reference_manifest["topology_margin"][0])["cutoff_ang"])
    topology = [
        {
            "direction_name": direction.name,
            "delta_ang": max(DELTA_SWEEP_ANG),
            **fdp.topology_margin(
                mode_payload["positions_ang"],
                direction=direction,
                delta_ang=max(DELTA_SWEEP_ANG),
                cutoff_ang=cutoff,
                lattice_vectors_ang=mode_payload["cell_ang"],
                method="central",
            ).to_dict(),
        }
        for direction in directions
    ]

    points = list(K_CANDIDATES) + [K_DEGENERATE] + [
        point for point in K_CALIBRATION if point["label"] not in {c["label"] for c in K_CANDIDATES}
    ]
    provider = spectra_provider or siesta_spectra
    spectra = provider(
        reference_root / "runs/equilibrium/equilibrium.TSHS", points
    )
    by_label = {row["label"]: row for row in spectra}
    k_selection = select_result_k(spectra, K_CANDIDATES)

    gate_paths = {
        "go2": certification_dir / "go2_certification.json",
        "c18": certification_dir / "c18_gamma_phonon_certification.json",
        "c14c": basis_response_report,
        "c14": checkpoint_error_report,
        "go6": go6_report,
    }
    gates = preconditions(gate_paths)
    geometry = geometry_consistency(reference_manifest, phonon_manifest)
    review = physics_review(
        mode=doublet,
        k_selection=k_selection,
        spectra=by_label,
        c18=_read_json(gate_paths["c18"]),
        gates=gates,
        geometry=geometry,
    )

    blocking = [
        f"{row['name']}: {row['status']}"
        for row in gates
        if row["required_before_execution"] and row["status"] != "PASS"
    ]
    if not geometry["consistent"]:
        blocking.append("phonon and derivative campaigns do not share a geometry/basis signature")

    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "ticket": TICKET,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_content_sha256": None,
        "experiment": {
            "system": "graphene primitive cell (2 C), materials/graphene/RUN.fdf",
            "q_fractional": [0.0, 0.0, 0.0],
            "q_label": "Gamma",
            "k_plus_q_equals_k": True,
            "stage_1": "one non-degenerate k: the operator-level test",
            "stage_2": f"the degenerate block at {K_DEGENERATE['label']}: the same operator "
            "read through GO-6 subspace invariants",
            "geometry_signature": mode_payload.get("geometry_signature"),
            "geometry_consistency": geometry,
        },
        "phonon": {
            "provider": mode_payload.get("provider"),
            "contract_id": mode_payload.get("contract_id"),
            "fc_range": RESULT_FC_RANGE,
            "asr_policy": RESULT_ASR_POLICY,
            "artifact": str(
                phonon_root / "ranges" / RESULT_FC_RANGE / f"gamma_modes_asr_{RESULT_ASR_POLICY}.json"
            ),
            "artifact_sha256": _file_sha256(
                phonon_root / "ranges" / RESULT_FC_RANGE / f"gamma_modes_asr_{RESULT_ASR_POLICY}.json"
            ),
            "conventions": mode_payload.get("conventions"),
            "masses_amu": mode_payload.get("masses_amu"),
            "doublet": doublet,
            "amplitude_contract": "u_kappa = A_zp * e_kappa / sqrt(M_kappa / M_ref) with "
            "A_zp = sqrt(hbar / (2 M omega)); for graphene the two masses are equal, so the "
            "unit-Frobenius direction is the mass-weighted eigenvector itself and g[eV] = "
            "A_zp * g[eV/Ang]",
        },
        "perturbation": {
            "normalization": fdp.NORMALIZATION,
            "method": "central",
            "delta_sweep_ang": list(DELTA_SWEEP_ANG),
            "directions": [direction.to_dict() for direction in directions],
            "topology_margin": topology,
            "topology_preserved": all(row["topology_preserved"] for row in topology),
        },
        "electronic": {
            "window": {
                "states_below": WINDOW_STATES_BELOW,
                "states_above": WINDOW_STATES_ABOVE,
                "rule": "the occupied count comes from the declared electron count, never "
                "from where a gap happens to be",
            },
            "occupations": {
                "electron_count": ELECTRON_COUNT,
                "spin_degeneracy": SPIN_DEGENERACY,
                "mu_convention": MU_CONVENTION,
                "smearing": "not part of any artifact upstream of g (roadmap XII)",
            },
            "degeneracy_tol_ev": DEGENERACY_TOL_EV,
            "min_window_gap_ev": MIN_WINDOW_GAP_EV,
            "k_selection": k_selection,
            "k_result_non_degenerate": by_label[k_selection["selected"]],
            "k_result_degenerate": by_label[K_DEGENERATE["label"]],
            "k_calibration": [by_label[point["label"]] for point in K_CALIBRATION],
            "k_validation": by_label[SPLITS["validation"]["k_labels"][0]],
            "subspace_machinery": "Comparison/scripts/epc_subspaces.py (GO-6): principal "
            "angles, singular values and projector distances. No band index is a claim.",
        },
        "references": {
            "paths": list(REFERENCE_PATHS),
            "basis_response": BASIS_RESPONSE_CONTRACT,
            "formalism_id": FORMALISM_ID,
            "equilibrium_run": str(reference_root / "runs/equilibrium"),
            "equilibrium_tshs_sha256": _file_sha256(
                reference_root / "runs/equilibrium/equilibrium.TSHS"
            ),
            "siesta_runtime_ref": reference_manifest.get("siesta_runtime_ref"),
            "provenance_limitation": "VERIFIED_BINARY_ONLY propagates to every artifact "
            "derived from this reference",
        },
        "splits": {
            **{name: {**split, "directions": list(split["directions"]),
                      "k_labels": list(split["k_labels"]),
                      "fc_ranges": list(split["fc_ranges"])}
               for name, split in SPLITS.items()},
            "disjoint_in": "(direction, k) pairs and directions",
            "notes": list(SPLIT_NOTES),
        },
        "tolerances": {
            **TOLERANCES,
            "constants": {
                "MIN_SIGNAL_TO_FLOOR": MIN_SIGNAL_TO_FLOOR,
                "BACKEND_NOISE_MARGIN": BACKEND_NOISE_MARGIN,
                "TRANSFER_FACTOR": TRANSFER_FACTOR,
                "ATTRIBUTION_MARGIN": ATTRIBUTION_MARGIN,
                "G_ADEQUACY_THRESHOLD": G_ADEQUACY_THRESHOLD,
            },
            "priors": {
                "go2_tau_fd": ((_read_json(gate_paths["go2"]) or {}).get("tau_fd") or {}).get(
                    "global_by_kind"
                ),
                "go3_tolerances": (
                    (_read_json(gate_paths["c14"]) or {}).get("internal_validation_go3") or {}
                ).get("tolerances"),
                "c14_derivative_level_relative_frobenius": (
                    (_read_json(gate_paths["c14"]) or {}).get("verdict") or {}
                ).get("worst_relative_frobenius"),
                "c14_is_not_tau_model": "recorded as a prior on the derivative level only; "
                "converting it into a percentage of g is forbidden by "
                "FORBIDDEN_TAU_MODEL_SOURCES",
            },
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
    protocol["frozen_content_sha256"] = freeze_hash(protocol)
    return _json_safe(protocol)


def load_protocol(path: Path = DEFAULT_OUTPUT_DIR / PROTOCOL_NAME) -> dict[str, Any]:
    payload = _read_json(Path(path))
    if payload is None:
        raise PreregistrationError(f"no pre-registered protocol at {path}")
    return payload


def verify(
    path: Path = DEFAULT_OUTPUT_DIR / PROTOCOL_NAME,
    *,
    result_root: Path = DEFAULT_RESULT_ROOT,
) -> dict[str, Any]:
    """Is the frozen file intact, and did every result cite it?

    The second half is what makes the "pre" checkable: a report in the result
    root that does not carry this hash was not produced under this protocol.
    """
    protocol = load_protocol(path)
    recomputed = freeze_hash(protocol)
    stored = protocol.get("frozen_content_sha256")
    results = []
    for report in sorted(Path(result_root).rglob("*.json")) if Path(result_root).is_dir() else []:
        payload = _read_json(report) or {}
        cited = payload.get("preregistration_sha256")
        results.append(
            {"report": str(report), "preregistration_sha256": cited, "cites_this_protocol": cited == stored}
        )
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
    path: Path = DEFAULT_OUTPUT_DIR / PROTOCOL_NAME,
) -> dict[str, Any]:
    """Gate for C19/C20: refuse to compute g outside the pre-registered protocol."""
    protocol = load_protocol(path)
    if freeze_hash(protocol) != protocol.get("frozen_content_sha256"):
        raise PreregistrationError(
            f"{path} has been edited since it was frozen; a protocol changed after the fact "
            "is not a pre-registration"
        )
    if protocol.get("physics_review", {}).get("status") != "APPROVED":
        raise PreregistrationError(
            f"the physics review is {protocol.get('physics_review', {}).get('status')!r}: "
            f"{[row for row in protocol['physics_review']['items'] if row['decision'] != 'approved']}"
        )
    if not protocol.get("ready_to_execute"):
        raise PreregistrationError(
            "the protocol is frozen but its preconditions are not met: "
            f"{protocol.get('blocking')}"
        )
    return protocol


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--phonon-root", type=Path, default=DEFAULT_PHONON_ROOT)
    parser.add_argument("--certification-dir", type=Path, default=DEFAULT_CERTIFICATION_DIR)
    parser.add_argument("--basis-response-report", type=Path, default=DEFAULT_BASIS_RESPONSE)
    parser.add_argument("--checkpoint-error-report", type=Path, default=DEFAULT_CHECKPOINT_ERROR)
    parser.add_argument("--go6-report", type=Path, default=DEFAULT_GO6_REPORT)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="do not rebuild: check the frozen file's hash and that every result cites it",
    )
    parser.add_argument(
        "--sign-off",
        default=None,
        metavar="NAME",
        help="add a human signature to the frozen protocol (re-freezes: the signature is "
        "part of the protocol)",
    )
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
        protocol["frozen_content_sha256"] = freeze_hash(protocol)
    else:
        protocol = build_protocol(
            reference_root=args.reference_root,
            phonon_root=args.phonon_root,
            certification_dir=args.certification_dir,
            basis_response_report=args.basis_response_report,
            checkpoint_error_report=args.checkpoint_error_report,
            go6_report=args.go6_report,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")

    print(f"pre-registration {protocol['protocol_id']} -> {output}")
    print(f"  frozen_content_sha256 : {protocol['frozen_content_sha256']}")
    print(f"  mode                  : E2g doublet, branches {protocol['phonon']['doublet']['branches']}, "
          f"omega = {protocol['phonon']['doublet']['frequency_ev']:.6f} eV "
          f"({protocol['phonon']['fc_range']}, ASR {protocol['phonon']['asr_policy']})")
    print(f"  stage 1 k             : {protocol['electronic']['k_result_non_degenerate']['label']} "
          f"{protocol['electronic']['k_result_non_degenerate']['k_fractional']} "
          f"(min window gap {protocol['electronic']['k_result_non_degenerate']['min_window_gap_ev']:.3f} eV)")
    print(f"  stage 2 k             : {protocol['electronic']['k_result_degenerate']['label']}")
    print(f"  physics review        : {protocol['physics_review']['status']}")
    print(f"  ready to execute      : {protocol['ready_to_execute']}")
    for reason in protocol["blocking"]:
        print(f"    blocked by: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
