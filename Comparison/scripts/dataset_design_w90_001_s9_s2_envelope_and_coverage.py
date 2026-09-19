#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9-S2: frozen envelope/domain/density/coverage spec.

Design-only artifact -- generates no geometries, runs no SIESTA, trains no
models. It defines, for the W90 graphene displacement dataset:

  (a) the common physical envelope (max per-atom displacement norm, Ang),
  (b) training-domain levels R_train_max and the OOD boundary rule,
  (c) test amplitudes,
  (d) per-sampler-family density/resolution axes with explicit meaning,
  (e) the family x dim x k x domain x resolution x seed coverage matrix,
  (f) the cost-accounting schema (per-design vs campaign-union SIESTA IDs).

Reuses the family names, dimensionalities and amplitude constants already
frozen in ``w90_displacement_sampler_family.py`` (S2) rather than
re-declaring them, and grounds ``coverage_matrix`` "executed" cells in the
real run directories left by the S3 pilot / S4 learning-curve campaign
(``Comparison/results/dataset_design_w90_001_s4/runs``) instead of asserting
untested state.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import w90_displacement_sampler_family as sampler  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parents[1]
S4_RUNS_DIR = REPO_ROOT / "Comparison" / "results" / "dataset_design_w90_001_s4" / "runs"
OUTPUT_JSON = REPO_ROOT / "Comparison" / "results" / "dataset_design_w90_001_s9_s2" / "envelope_spec.json"

# --------------------------------------------------------------------------
# (a)-(c) Envelope, domain levels, test amplitudes
# --------------------------------------------------------------------------

FAMILIES: tuple[str, ...] = (
    "axial_radial",
    "angular_shell",
    "local_pair_modes",
    "sobol_sparse",
    "random_cartesian",
    "latin_hypercube",
)
DIMENSIONALITIES: tuple[str, ...] = sampler.DIMENSIONALITIES
K_VALUES: tuple[int, ...] = (1, 2)

NEAR_LINEAR_PROBE_ANG: float = 0.01
R_TRAIN_MAX_LEVELS_ANG: tuple[float, ...] = (0.03, 0.05, 0.08)
OOD_CONDITIONAL_ANG: float = 0.12
assert OOD_CONDITIONAL_ANG == sampler.OOD_AMPLITUDE_ANG
DOMAIN_LEVELS_ANG: tuple[float, ...] = R_TRAIN_MAX_LEVELS_ANG + (OOD_CONDITIONAL_ANG,)
TEST_AMPLITUDES_ANG: tuple[float, ...] = (0.01, 0.03, 0.05, 0.08, 0.10, 0.12)

# 0.12 Ang training inclusion is gated per-k, decided from a direct probe
# (dataset_design_w90_001_s9_ood012_probe.py, dataset_design_w90_001_s9_ood012_k2_probe.py)
# and two recorded expert consultations (DATASET-DESIGN-W90-001-S9 event log,
# requests EXP-ood012-manual and EXP-ood012-k2-tradeoff), not extrapolated
# from the 0.08 Ang delta_nl numbers alone. See OOD_CONDITIONAL_JUSTIFICATION.
#
# k=1: unconditionally enabled. N(r) = ||H(+r)+H(-r)-2H0|| / (||H(+r)-H0||+||H(-r)-H0||)
#      grows smoothly through 0.12 Ang (0.12-0.13 there, up from 0.01-0.02 at 0.01 Ang),
#      C(r) stays ~1.0 (no cancellation artifact), J_eff drift vs the 0.01 Ang reference
#      stays under 4%. Expert: "encouraging... I would be comfortable considering k=1
#      first."
#
# k=2: enabled for `local_pair_modes` ONLY (longitudinal/transverse_ip/transverse_opp --
#      genuine relative bond-stretch/bend motion), flagged PROVISIONAL pending the
#      ablation study in dataset_design_w90_001_s9_ood012_k2_ablation.py. `axial_radial`
#      k=2 stays disabled: its two active atoms move in the SAME direction by
#      construction, which for a 2-atom graphene cell is a near-rigid basis shift --
#      confirmed via independent single-atom Jacobians (|J_atom0+J_atom1| ~1000x smaller
#      than |J_atom0|, |J_atom1| individually) to be a near-symmetry cancellation, not a
#      probe of correlated response either way; training on it would fit noise, not
#      physics. For local_pair_modes, N(r) is ~2x k=1's at matched amplitude but C(r)
#      stays 0.97-1.0 (clean, non-cancelling signal) and grows smoothly -- expert's
#      revised reading: larger curvature alone is not a reason to exclude a physically
#      relevant domain from training, but it does mean this region needs denser boundary
#      sampling and a held-out test before being trusted the way k=1 is.
OOD_CONDITIONAL_INCLUDED_IN_TRAINING_FOR_K: frozenset[int] = frozenset({1, 2})
OOD_CONDITIONAL_K2_FAMILIES: frozenset[str] = frozenset({"local_pair_modes"})
OOD_CONDITIONAL_K2_PROVISIONAL: bool = True
OOD_CONDITIONAL_JUSTIFICATION: str = (
    "0.12 Ang training inclusion is conditional on evidence that the "
    "harmonic/near-linear regime (probed at 0.01 Ang) still bounds the "
    "dH/dR response at 0.08 Ang within the stencil validation tolerance "
    "used by hamiltonian_derivative_stencil.py; without that evidence, "
    "training past 0.08 Ang risks fitting anharmonic response the S1-S8 "
    "pipeline was never designed to sample densely. Resolved per-k by direct "
    "probe and expert consultation -- see the block comment above "
    "OOD_CONDITIONAL_INCLUDED_IN_TRAINING_FOR_K."
)


def ood_boundary_rule(r_train_max_ang: float, test_amplitude_ang: float) -> str:
    """Per-model OOD classification: closed-interval convention.

    A test amplitude at exactly R_train_max is treated as boundary/in-domain
    (matches how R_TRAIN_MAX_LEVELS_ANG itself is drawn from
    TEST_AMPLITUDES_ANG); strictly greater is OOD.

    NOTE: this takes a *nominal* batch amplitude. Box samplers
    (random_cartesian/sobol_sparse) realize a range of effective per-atom
    displacements at a given nominal amplitude, so a batch can straddle the
    OOD boundary even though every point shares one nominal label -- use
    ``effective_ood_status`` on the geometries actually evaluated whenever
    per-sample ``max_displacement_ang`` is available; this function remains
    for pure-nominal contexts (documentation, coverage matrix, tests) where
    no materialized geometry exists yet.
    """

    return "in_domain" if test_amplitude_ang <= r_train_max_ang + 1e-9 else "ood"


def effective_ood_status(r_train_max_ang: float, effective_displacements_ang: list[float]) -> dict[str, Any]:
    """OOD classification from each geometry's own realized max per-atom displacement.

    ``effective_displacements_ang``: one value per sample actually evaluated
    (``Sample.max_displacement_ang``), not the batch's nominal amplitude
    label. A batch is "mixed" when some points fall in-domain and others OOD
    under the same nominal label -- exactly the confound a nominal-amplitude
    rule cannot see.
    """

    if not effective_displacements_ang:
        return {"status": "unknown", "reason": "no effective displacements available", "n": 0}
    statuses = [ood_boundary_rule(r_train_max_ang, d) for d in effective_displacements_ang]
    n_ood = sum(1 for s in statuses if s == "ood")
    n_in = len(statuses) - n_ood
    if n_ood == 0:
        status = "in_domain"
    elif n_in == 0:
        status = "ood"
    else:
        status = "mixed"
    return {
        "status": status,
        "n": len(statuses),
        "n_in_domain": n_in,
        "n_ood": n_ood,
        "min_effective_displacement_ang": min(effective_displacements_ang),
        "max_effective_displacement_ang": max(effective_displacements_ang),
        "r_train_max_ang": r_train_max_ang,
    }


# --------------------------------------------------------------------------
# (d) Density/resolution axes, per sampler family
# --------------------------------------------------------------------------

# TODO(open, not a bug): axial_radial/local_pair_modes cap at levels (1,2,4), which
# yields only 2-24 real points in the pilot (vs up to 256 for the box samplers) --
# the pilot never screens axial_radial in the 24-256 point regime, only cheap or (if
# a cheap design wins Pareto) whatever S9-S5 extrapolates it to later. Widening these
# levels (e.g. to mirror angular_shell's (8,16,32)) would close that coverage gap but
# requires a real SIESTA + pilot-training rerun for the family, not just a viz change.
DENSITY_LEVELS: dict[str, dict[str, Any]] = {
    "axial_radial": {
        "axis": "n_radial_shells",
        "meaning": "number of radial amplitude shells swept per canonical axis direction (len(radii_ang))",
        "levels": (1, 2, 4),
    },
    "angular_shell": {
        "axis": "n_points",
        "meaning": "angular points on the fixed-radius shell (uniform ring in 2D_in, Fibonacci sphere in 3D)",
        "levels": (8, 16, 32),
    },
    "local_pair_modes": {
        "axis": "n_amplitude_points",
        "meaning": "number of radial amplitudes swept per bond-frame mode (len(amplitudes_ang)); mode count itself is fixed at 5",
        "levels": (1, 2, 4),
    },
    "sobol_sparse": {
        "axis": "max_n",
        "meaning": "scrambled-Sobol net size; any power-of-two prefix is itself a valid nested net",
        "levels": (16, 64, 256),
    },
    "random_cartesian": {
        "axis": "n_structures",
        "meaning": "number of i.i.d. uniform/gaussian draws inside the per-axis amplitude box",
        "levels": (16, 64, 256),
    },
    "latin_hypercube": {
        "axis": "n_structures",
        "meaning": "number of Latin Hypercube points (direction and amplitude stratified jointly)",
        "levels": (16, 64, 256),
    },
}


def is_applicable(family: str, dim: str, k: int) -> tuple[bool, str]:
    """Architectural applicability of (family, dim, k), independent of domain/resolution."""

    if family == "angular_shell" and dim not in ("2D_in", "3D"):
        return False, (
            "generate_angular_shell only supports 2D_in (uniform ring) and 3D "
            "(quasi-uniform sphere); raises ValueError for 1D_in/1D_z"
        )
    if family == "local_pair_modes" and k != 2:
        return False, "pair modes require two active atoms (k=2) by construction"
    return True, ""


_STOCHASTIC_FAMILIES = ("sobol_sparse", "random_cartesian", "latin_hypercube")
_SEEDS_BY_FAMILY: dict[str, tuple[Any, ...]] = {
    family: (0, 1, 2) if family in _STOCHASTIC_FAMILIES else ("deterministic",) for family in FAMILIES
}

# --------------------------------------------------------------------------
# (e) Coverage matrix: family x dim x k x domain x resolution x seed
# --------------------------------------------------------------------------

_RUN_DIR_RE = re.compile(r"^(?P<family>[a-z_]+)__(?P<dim>[0-9A-Za-z_]+)__k(?P<k>[0-9]+)__n(?P<n>[0-9]+)__seed(?P<seed>[0-9]+)$")
PILOT_R_TRAIN_MAX_ANG: float = 0.08  # max(PILOT_AMPLITUDES_ANG) in run_w90_displacement_siesta_pilot.py
PILOT_DENSITY_LEVEL: int = 64  # PILOT_MAX_N in run_w90_displacement_siesta_pilot.py
PILOT_SEED: int = 0


def _scan_executed_family_dim_k(runs_dir: Path) -> set[tuple[str, str, int]]:
    """(family, dim, k) triplets with at least one completed S4 pilot run directory."""

    if not runs_dir.is_dir():
        return set()
    found: set[tuple[str, str, int]] = set()
    for entry in runs_dir.iterdir():
        match = _RUN_DIR_RE.match(entry.name)
        if match:
            found.add((match.group("family"), match.group("dim"), int(match.group("k"))))
    return found


def _cell_state(
    family: str, dim: str, k: int, domain: float, resolution: int | str, seed: Any,
    executed_combos: set[tuple[str, str, int]],
) -> tuple[str, str]:
    applicable, reason = is_applicable(family, dim, k)
    if not applicable:
        return "not_applicable", reason

    if abs(domain - OOD_CONDITIONAL_ANG) < 1e-9:
        if k not in OOD_CONDITIONAL_INCLUDED_IN_TRAINING_FOR_K:
            return "pruned", (
                f"0.12 Ang training domain is not enabled for k={k}; see "
                "OOD_CONDITIONAL_INCLUDED_IN_TRAINING_FOR_K and OOD_CONDITIONAL_JUSTIFICATION"
            )
        if k == 2 and family not in OOD_CONDITIONAL_K2_FAMILIES:
            return "pruned", (
                f"0.12 Ang training domain is enabled for k=2 only for "
                f"{sorted(OOD_CONDITIONAL_K2_FAMILIES)} (genuine relative bond motion); "
                f"'{family}' at k=2 moves both active atoms in the same direction, "
                "confirmed to be a near-rigid-shift cancellation, not evidence either way "
                "-- see OOD_CONDITIONAL_JUSTIFICATION"
            )

    if family == "latin_hypercube":
        return "pending", (
            "generate_latin_hypercube is implemented but the S3 training-pool "
            "generator (run_w90_displacement_siesta_pilot.py) does not call it "
            "yet -- it is currently wired only into the S5 common_displacement "
            "frozen test. Listed here as a training family per this task's "
            "scope; execution is pending S3 rewiring."
        )

    if (
        (family, dim, k) in executed_combos
        and abs(domain - PILOT_R_TRAIN_MAX_ANG) < 1e-9
        and resolution == PILOT_DENSITY_LEVEL
        and seed == PILOT_SEED
    ):
        return "executed", (
            f"recorded under Comparison/results/dataset_design_w90_001_s4/runs/"
            f"{family}__{dim}__k{k}__n{PILOT_DENSITY_LEVEL}__seed{PILOT_SEED}"
        )

    return "pending", "no execution recorded yet for this cell"


def build_coverage_matrix(runs_dir: Path = S4_RUNS_DIR) -> list[dict[str, Any]]:
    executed_combos = _scan_executed_family_dim_k(runs_dir)
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        levels = DENSITY_LEVELS[family]["levels"]
        seeds = _SEEDS_BY_FAMILY[family]
        for dim in DIMENSIONALITIES:
            for k in K_VALUES:
                for domain in DOMAIN_LEVELS_ANG:
                    for resolution in levels:
                        for seed in seeds:
                            state, reason = _cell_state(family, dim, k, domain, resolution, seed, executed_combos)
                            rows.append(
                                {
                                    "family": family,
                                    "dim": dim,
                                    "k": k,
                                    "domain_r_train_max_ang": domain,
                                    "resolution": resolution,
                                    "seed": seed,
                                    "state": state,
                                    "reason": reason,
                                }
                            )
    return rows


# --------------------------------------------------------------------------
# (f) Cost accounting schema
# --------------------------------------------------------------------------

COST_SCHEMA: dict[str, Any] = {
    "unique_siesta_ids": {
        "definition": "distinct displaced-geometry IDs required by a single design, no cross-design dedup",
        "formula": "unique_siesta_ids(design) = |geometry_ids(design)|",
    },
    "union_siesta_ids": {
        "definition": "distinct displaced-geometry IDs across the whole campaign, deduplicated by geometry content hash",
        "formula": "union_siesta_ids(campaign) = |union over designs in campaign of geometry_ids(design)|",
    },
    "reproducible_cost": {
        "definition": "SIESTA compute to regenerate one design from scratch (no reuse of prior campaign output)",
        "formula": "reproducible_cost(design) = unique_siesta_ids(design) * cost_per_siesta_point",
    },
    "incremental_cost": {
        "definition": "SIESTA compute to add one more design given a campaign that already materialized some geometries",
        "formula": "incremental_cost(design | already_run) = |geometry_ids(design) minus union_siesta_ids(already_run)| * cost_per_siesta_point",
    },
    "per_design_cost": {
        "definition": "sum of each design's own unique_siesta_ids, double-counting geometries shared across designs (e.g. a training pool and a frozen test that happen to reuse the same displacement)",
        "formula": "per_design_cost(campaign) = sum over designs in campaign of unique_siesta_ids(design) * cost_per_siesta_point",
    },
    "campaign_union_cost": {
        "definition": "true SIESTA spend for the campaign: each distinct geometry priced once regardless of how many designs reference it",
        "formula": "campaign_union_cost(campaign) = union_siesta_ids(campaign) * cost_per_siesta_point",
    },
}


# --------------------------------------------------------------------------
# Assembly + I/O
# --------------------------------------------------------------------------


def build_spec() -> dict[str, Any]:
    return {
        "envelope": {
            "definition": "max per-atom displacement norm (Angstrom) across all active atoms in a configuration",
            "units": "Angstrom",
            "near_linear_probe_ang": NEAR_LINEAR_PROBE_ANG,
            "r_train_max_levels_ang": list(R_TRAIN_MAX_LEVELS_ANG),
            "ood_conditional_ang": OOD_CONDITIONAL_ANG,
            "ood_conditional_included_in_training_for_k": sorted(OOD_CONDITIONAL_INCLUDED_IN_TRAINING_FOR_K),
            "ood_conditional_k2_families": sorted(OOD_CONDITIONAL_K2_FAMILIES),
            "ood_conditional_k2_provisional": OOD_CONDITIONAL_K2_PROVISIONAL,
            "ood_conditional_justification": OOD_CONDITIONAL_JUSTIFICATION,
            "test_amplitudes_ang": list(TEST_AMPLITUDES_ANG),
            "ood_boundary_rule": (
                "a test amplitude a is in-domain for a model trained at R_train_max "
                "if a <= R_train_max (closed interval), else OOD"
            ),
        },
        "dimensionalities": list(DIMENSIONALITIES),
        "k_values": list(K_VALUES),
        "families": list(FAMILIES),
        "density_levels": DENSITY_LEVELS,
        "coverage_matrix": build_coverage_matrix(),
        "cost_schema": COST_SCHEMA,
    }


REQUIRED_TOP_KEYS = (
    "envelope", "dimensionalities", "k_values", "families", "density_levels", "coverage_matrix", "cost_schema",
)
REQUIRED_ENVELOPE_KEYS = (
    "units", "near_linear_probe_ang", "r_train_max_levels_ang", "ood_conditional_ang",
    "ood_conditional_justification", "test_amplitudes_ang", "ood_boundary_rule",
)
REQUIRED_COVERAGE_ROW_KEYS = ("family", "dim", "k", "domain_r_train_max_ang", "resolution", "seed", "state", "reason")
VALID_STATES = {"executed", "pruned", "not_applicable", "pending"}


def validate_spec(spec: dict[str, Any]) -> None:
    missing_top = [key for key in REQUIRED_TOP_KEYS if key not in spec]
    if missing_top:
        raise ValueError(f"envelope_spec missing top-level keys: {missing_top}")

    envelope = spec["envelope"]
    missing_env = [key for key in REQUIRED_ENVELOPE_KEYS if key not in envelope]
    if missing_env:
        raise ValueError(f"envelope_spec['envelope'] missing keys: {missing_env}")
    for level in envelope["r_train_max_levels_ang"]:
        if not (0.0 < level < 0.2):
            raise ValueError(f"R_train_max level {level} is not a finite, physically reasonable Angstrom value (<0.2)")

    if not spec["coverage_matrix"]:
        raise ValueError("coverage_matrix must not be empty")
    for row in spec["coverage_matrix"]:
        missing_row = [key for key in REQUIRED_COVERAGE_ROW_KEYS if key not in row]
        if missing_row:
            raise ValueError(f"coverage_matrix row missing keys {missing_row}: {row}")
        if row["state"] not in VALID_STATES:
            raise ValueError(f"coverage_matrix row has invalid state {row['state']!r}: {row}")

    for family, info in spec["density_levels"].items():
        if len({d for d, dim, k in ((r["family"], r["dim"], r["k"]) for r in spec["coverage_matrix"]) if d == family}) == 0:
            raise ValueError(f"density_levels declares {family!r} but coverage_matrix has no rows for it")
        if len(info["levels"]) < 3:
            raise ValueError(f"density_levels[{family!r}] has fewer than 3 resolution levels")


def write_spec_json(path: Path = OUTPUT_JSON) -> Path:
    spec = build_spec()
    validate_spec(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _demo() -> None:
    spec = build_spec()
    validate_spec(spec)
    assert ood_boundary_rule(0.08, 0.05) == "in_domain"
    assert ood_boundary_rule(0.08, 0.08) == "in_domain"
    assert ood_boundary_rule(0.08, 0.10) == "ood"
    states = {row["state"] for row in spec["coverage_matrix"]}
    assert states <= VALID_STATES
    assert "executed" in states, "expected at least one grounded 'executed' cell from the S4 pilot runs"
    print(f"OK: {len(spec['coverage_matrix'])} coverage_matrix rows, states={sorted(states)}")


if __name__ == "__main__":
    written = write_spec_json()
    _demo()
    print(f"wrote {written}")
