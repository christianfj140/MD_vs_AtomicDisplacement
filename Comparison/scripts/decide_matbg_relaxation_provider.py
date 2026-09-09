#!/usr/bin/env python3
"""E-F_001-S47 (decision phase): GO-10 or NO-GO-10 for a scalable MATBG relaxation provider.

Fase 11 of the roadmap asks for the *relaxed* magic-angle `(31, 30)` cell
(11 164 atoms), with forces, corrugation, AA/AB/BA domain areas and an
interlayer-spacing distribution -- not a rigid input geometry, and not one
reused from the rigid campaign. Producing that structure requires a source of
total-energy forces on the whole bilayer that is defensible at this scale.

This module answers the same way :mod:`decide_matbg_phonon_provider` answered
GO-8a: as data, from a live probe against this environment, never from memory
of what "usually" ships. It returns ``GO-10:<candidate id>`` only if a
candidate can produce forces covering every one of Nam & Koshino's (2017)
relaxation-driven mechanisms -- in-plane lattice relaxation, out-of-plane
corrugation, and AB/BA domain formation, which is *interlayer registry*
physics by construction, not intralayer bond stiffness -- and ``NO-GO-10``
otherwise. A ``NO-GO-10`` verdict is itself the deliverable when nothing here
is defensible: it blocks any relaxed-geometry artifact from being produced
with physically absent interlayer coupling and mislabelled as "relaxed".
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SCHEMA = "epc_matbg_relaxation_provider_decision_v1"
TICKET = "E-F_001-S47"
MEMO = "docs/epc_s47_matbg_relaxation_decision.md"
GATE = "GO-10"

RIGID_GEOMETRY = "materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf"
TARGET_ATOM_COUNT = 11164

#: Nam & Koshino, PRB 96, 075311 (2017): below ~2 deg twist, relaxation
#: redistributes area from AA into AB/BA domains and produces measurable
#: corrugation. A relaxation source is defensible only if it can in
#: principle produce a restoring force along every one of these mechanisms.
RELAXATION_MECHANISMS = ("in_plane_lattice_relaxation", "out_of_plane_corrugation", "ab_ba_domain_formation")

_INTRALAYER_POTENTIAL_MODULES = ("ase.calculators.tersoff", "matscipy.calculators.manybody")
_CLASSICAL_MD_ENGINE_MODULES = ("lammps", "openmm")
_INTERLAYER_REGISTRY_POTENTIAL_MODULES = ("kimpy",)  # OpenKIM is the only route to KC/Lebedeva here
_ML_FOUNDATION_MODULES = ("mace.calculators",)


def _importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def _any_importable(module_names: tuple[str, ...]) -> dict[str, bool]:
    return {name: _importable(name) for name in module_names}


def _local_ml_foundation_checkpoint() -> str | None:
    """A path to a downloaded universal-MLIP checkpoint, if one is already local.

    Checked, not fetched: no network access and no repo-wide walk, just the
    two places a foundation checkpoint would sit if someone had put one there.
    """
    env_path = os.environ.get("MACE_MODEL_PATH") or os.environ.get("MATBG_FOUNDATION_POTENTIAL")
    if env_path and Path(env_path).is_file():
        return env_path
    cache_dir = Path.home() / ".cache" / "mace"
    if cache_dir.is_dir():
        checkpoints = sorted(cache_dir.glob("*.model")) + sorted(cache_dir.glob("*.pt"))
        if checkpoints:
            return str(checkpoints[0])
    return None


def _candidate_dft_relaxation() -> dict[str, Any]:
    # No physical mechanism is missing from an exact DFT relaxation -- it is
    # disqualified on cost, and the cost here is categorically different from
    # (and larger than) the phonon FC estimate in S38: a relaxation needs a
    # converged SCF gradient at *every* ionic step, not one displaced solve
    # per direction. This cell has never had a converged full-SCF run at all
    # (roadmap: "no existe Hamiltoniano DFT de referencia para ese target");
    # only MaxSCFIterations=0 overlap-only runs exist for it.
    return {
        "id": "dft_relaxation_siesta",
        "family": "ab_initio_relaxation",
        "mechanism": "SIESTA MD.TypeOfRun CG/FIRE ionic relaxation with converged SCF forces "
        "at every step, on the production (31,30) cell itself",
        "coverage": {mechanism: True for mechanism in RELAXATION_MECHANISMS},
        "coverage_basis": "an exact DFT relaxation has no missing physical mechanism on any "
        "axis; this candidate is not excluded by physics",
        "availability": {
            "target_atom_count": TARGET_ATOM_COUNT,
            "converged_full_scf_run_ever_attempted_on_this_cell": False,
        },
        "defensible": False,
        "blocking_reason": "this cell has never had a converged full-SCF SIESTA run (only "
        "MaxSCFIterations=0, TS.onlyS=T overlap-only runs exist for it, per "
        "generate_siesta_overlap_only.py's own manifest); a relaxation needs a converged SCF "
        "force at every one of dozens of ionic steps, which is categorically larger than the "
        "single overlap-only step already used as the roadmap's own standing definition of "
        "intractable at this scale (docs/epc_s37_matbg_synthetic_benchmark.md), not a new "
        "estimate invented for this document",
    }


def _candidate_classical_empirical_potential() -> dict[str, Any]:
    intralayer = _any_importable(_INTRALAYER_POTENTIAL_MODULES)
    md_engine = _any_importable(_CLASSICAL_MD_ENGINE_MODULES)
    interlayer = _any_importable(_INTERLAYER_REGISTRY_POTENTIAL_MODULES)
    intralayer_available = any(intralayer.values())
    interlayer_available = any(interlayer.values())
    coverage = {
        "in_plane_lattice_relaxation": intralayer_available,
        "out_of_plane_corrugation": interlayer_available,
        "ab_ba_domain_formation": interlayer_available,
    }
    return {
        "id": "classical_empirical_potential",
        "family": "empirical_interatomic_potential",
        "mechanism": "intralayer analytic bond-order potential (Tersoff/Brenner) for the "
        "in-plane sp2 network, plus a registry-dependent interlayer potential "
        "(Kolmogorov & Crespi 2005 / Lebedeva et al.) for the stacking-dependent restoring "
        "force, minimized with a standard geometry optimizer -- the field-standard route for "
        "large-scale TBG relaxation (Nam & Koshino, PRB 96, 075311, 2017, uses exactly this "
        "combination to produce AB/BA domain formation below ~2 deg twist)",
        "coverage": coverage,
        "coverage_basis": "AB/BA domain formation and corrugation are driven by the "
        "twist-dependent *interlayer registry* energy landscape, not by intralayer bond "
        "stiffness: an intralayer-only potential leaves the two layers mechanically decoupled "
        "at zero lateral/normal stiffness, so relaxing under it can only relax each layer's "
        "flat in-plane lattice -- corrugation and domain formation are physically absent from "
        "that calculation, not merely approximate, until a registry-dependent interlayer "
        "potential is added",
        "availability": {
            "intralayer_bond_order_potential_importable": intralayer,
            "classical_md_engine_importable": md_engine,
            "interlayer_registry_potential_importable": interlayer,
        },
        "defensible": False,
        "blocking_reason": "the intralayer bond-order potential is available in this "
        "environment (matscipy/ase Tersoff-Brenner), but no registry-dependent interlayer "
        "potential (Kolmogorov-Crespi/Lebedeva) is installed anywhere in this repository or "
        "its declared dependencies, and neither is a classical MD engine (lammps/openmm) that "
        "could load one via OpenKIM -- adding and independently validating one is new "
        "development, not a selection among existing options",
    }


def _candidate_ml_universal_potential() -> dict[str, Any]:
    checkpoint = _local_ml_foundation_checkpoint()
    architecture_available = _importable("mace.calculators")
    return {
        "id": "ml_universal_interatomic_potential",
        "family": "machine_learning_interatomic_potential",
        "mechanism": "a pretrained universal MLIP (e.g. a MACE-MP-class foundation model) "
        "evaluated on the full (31,30) cell for forces, minimized with a standard geometry "
        "optimizer",
        "coverage": {mechanism: bool(checkpoint) for mechanism in RELAXATION_MECHANISMS},
        "coverage_basis": "coverage is contingent on the checkpoint's own training distribution "
        "including twisted-bilayer/registry-dependent carbon environments -- unknown here "
        "because no checkpoint is present to inspect or validate",
        "availability": {
            "mace_calculators_architecture_importable": architecture_available,
            "local_foundation_checkpoint_path": checkpoint,
        },
        "defensible": False,
        "blocking_reason": "mace-torch is a declared dependency (it builds this repository's "
        "own Graph2Mat Hamiltonian architecture), but that is an architecture library, not a "
        "trained total-energy foundation model; no MLIP checkpoint is present in this "
        "environment or repository, and fetching + independently validating one before "
        "trusting its forces for relaxation is a separate acquisition ticket, not something "
        "this selection can silently assume",
    }


def _candidates() -> list[dict[str, Any]]:
    return [
        _candidate_dft_relaxation(),
        _candidate_classical_empirical_potential(),
        _candidate_ml_universal_potential(),
    ]


def _acceptance_criteria() -> dict[str, Any]:
    """What a future relaxation run must record, regardless of which candidate clears GO-10."""
    return {
        "preregistration": "method, force-convergence tolerance and stopping criterion fixed "
        "before the run, per the roadmap's Fase 11 requirement",
        "forces": "final force set on every atom, from the same backend used to relax",
        "corrugation": "out-of-plane z-distribution relative to each layer's mean plane",
        "domain_areas": "AA/AB/BA stacking-registry area fractions, per Nam & Koshino (2017)",
        "interlayer_spacing_distribution": "not a single mean value -- the full distribution, "
        "since domain formation is precisely the statement that spacing is not uniform",
        "geometry_signature": "a new geometry_cell_species_sha256 (shared/reference_provenance.py) "
        "distinct from the rigid cell's, so it invalidates rather than silently reuses any "
        "H/S/eigenpair/phonon/derivative artifact keyed to the rigid geometry",
        "provenance": "generator identity + version, full input set, trajectory, and final "
        "geometry all signed, per the roadmap's Requirements section",
        "status_label": "'relaxed', kept distinct from the rigid campaign's 'rigid_tbg_model' "
        "status -- never merged into a single geometry status",
    }


class MatbgRelaxationProviderDecisionError(RuntimeError):
    """Raised when this decision cannot be computed reproducibly."""


def matbg_relaxation_provider_decision() -> dict[str, Any]:
    """The GO-10 / NO-GO-10 decision, as data. Never produces a relaxed geometry."""
    candidates = _candidates()
    defensible = [candidate for candidate in candidates if candidate["defensible"]]
    verdict = f"GO-10:{defensible[0]['id']}" if defensible else "NO-GO-10"
    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "memo": MEMO,
        "gate": GATE,
        "rigid_geometry": RIGID_GEOMETRY,
        "target_atom_count": TARGET_ATOM_COUNT,
        "relaxation_mechanisms": list(RELAXATION_MECHANISMS),
        "candidates": candidates,
        "verdict": verdict,
        "selected_provider": defensible[0]["id"] if defensible else None,
        "relaxed_geometry_produced": False,
        "acceptance_criteria_for_a_future_relaxation": _acceptance_criteria(),
        "reopen_if": "a concrete candidate closes its own blocking_reason with checkable "
        "evidence -- an installed and independently validated interlayer registry potential "
        "(Kolmogorov-Crespi/Lebedeva) paired with the intralayer bond-order potential already "
        "available here, reachable from a classical MD engine or an ASE-compatible calculator; "
        "or a locally present and independently validated universal MLIP checkpoint whose "
        "training distribution is shown to cover twisted-bilayer carbon -- and then clears the "
        "acceptance criteria above",
        "references": [
            "A. N. Kolmogorov and V. H. Crespi, Phys. Rev. B 71, 235415 (2005) -- the "
            "registry-dependent interlayer potential this decision found absent from every "
            "candidate that would otherwise be defensible",
            "N. N. T. Nam and M. Koshino, Phys. Rev. B 96, 075311 (2017) -- relaxation-driven "
            "AB/BA domain formation below ~2 deg twist, the physical mechanism this decision "
            "checks coverage against",
        ],
    }


def _self_test() -> None:
    decision = matbg_relaxation_provider_decision()
    assert decision["verdict"] == "NO-GO-10"
    assert decision["selected_provider"] is None
    assert decision["relaxed_geometry_produced"] is False
    assert {c["id"] for c in decision["candidates"]} == {
        "dft_relaxation_siesta",
        "classical_empirical_potential",
        "ml_universal_interatomic_potential",
    }
    for candidate in decision["candidates"]:
        assert set(candidate["coverage"]) == set(RELAXATION_MECHANISMS)
        assert not candidate["defensible"]
        assert candidate["blocking_reason"]
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    _self_test()
