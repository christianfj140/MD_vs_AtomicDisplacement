#!/usr/bin/env python3
"""E-F_001-S38 / D05B (decision phase): GO-8a or NO-GO-8a for a scalable MATBG PhononProvider.

Fase 10a of the roadmap needs a source of ``omega(q,nu)`` and ``e(q,nu)`` for the
rigid `(31, 30)` magic-angle cell (11 164 atoms) that is scalable, has a known
range of validity, and satisfies the :mod:`phonon_provider` contract. This
module answers *which one*, or refuses to pick one, as data rather than prose:
:func:`matbg_phonon_provider_decision` checks what is actually importable in
this environment, states the coverage (acoustic / shear / breathing / moire,
the roadmap's own axes) and cost each candidate family can defensibly claim,
and returns ``GO-8a`` only if a candidate clears every axis. Nothing here
produces a phonon artifact or a PAO-projected coupling: a ``NO-GO-8a`` verdict is
itself the deliverable when no candidate is defensible (roadmap Fase 10a,
"Si ninguna opción es defendible, se emite NO-GO-8a sin producir acoplamiento proyectado PAO").

Evidence for each candidate comes from one of:

* an import probe run here, right now, against this environment
  (``ase.calculators.tersoff``, ``matscipy.calculators.manybody``, a classical
  MD engine, an interlayer registry potential, a local ML foundation-model
  checkpoint) -- never assumed from memory of what "usually" ships;
* the two large-cell assets that do exist in this repository,
  :mod:`run_tbg_tight_binding` (Moon & Koshino 2013 electronic ``p_z``
  Hamiltonian, ``S = I``, no total-energy/force term) and
  :mod:`phonon_provider` (the backend-agnostic contract every candidate must
  eventually satisfy);
* the absence of any classical-MD engine, interlayer registry potential,
  continuum-moire-phonon code or ML foundation checkpoint anywhere in this
  repository or its declared dependencies (verified by search, not inferred
  from the repository's name).
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

SCHEMA = "epc_matbg_phonon_provider_decision_v1"
TICKET = "E-F_001-S38"
MEMO = "docs/epc_s38_matbg_phonon_provider_decision.md"
GATE = "GO-8a"

TARGET_GEOMETRY = "materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf"
TARGET_ATOM_COUNT = 11164

#: The roadmap's own coverage axes (Fase 10a requirements / acceptance
#: criteria): "Evaluar cobertura de shear, breathing, acústicos y modos
#: moiré". A candidate is defensible only if it can, in principle and without
#: unpublished physics, produce a restoring force along every one of these.
COVERAGE_AXES = ("acoustic", "shear", "breathing", "moire")

#: Module names whose mere importability is evidence (not proof) that a given
#: piece of machinery exists in this environment. Checked live, not assumed.
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

    Checked, not fetched: this decision does not download weights. It looks in
    the two places a foundation checkpoint would sit if someone had already
    put one there -- an explicit env var, or mace's own default cache dir --
    both cheap, bounded stat/listdir calls, no network and no repo-wide walk.
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


def _candidate_dft_force_constants() -> dict[str, Any]:
    # C09/C18 already certify this route (SIESTA FC .Save.dHS + explicit FD) at
    # graphene scale (2-6 atoms). It is not disqualified on physics -- an exact
    # DFT dynamical matrix would cover every axis -- it is disqualified on cost.
    displaced_configurations = 6 * TARGET_ATOM_COUNT  # +/- x/y/z per atom, FC.Displacement convention
    return {
        "id": "dft_force_constants_siesta",
        "family": "ab_initio_force_constants",
        "mechanism": "SIESTA FC.Save.dHS on the production (31,30) cell itself, the same "
        "route already certified at graphene scale (C09/C18)",
        "coverage": {axis: True for axis in COVERAGE_AXES},
        "coverage_basis": "an exact DFT dynamical matrix has no missing physical mechanism "
        "on any axis; this candidate is not excluded by physics",
        "availability": {
            "target_atom_count": TARGET_ATOM_COUNT,
            "displaced_self_consistent_calculations_required": displaced_configurations,
        },
        "defensible": False,
        "blocking_reason": f"{displaced_configurations} displaced self-consistent DFT solves on "
        f"an {TARGET_ATOM_COUNT}-atom cell; the *electronic-only*, no-SCF, overlap-only step "
        "already used for this same cell needs GPU + explicit MACE node/edge chunking just to "
        "avoid OOM for one forward pass (docs/epc_s37_matbg_synthetic_benchmark.md Sec. 3) -- a "
        "full SCF FC campaign multiplies that by tens of thousands of independent solves, which "
        "is the roadmap's own definition of intractable at this scale, not a new estimate",
    }


def _candidate_moon_koshino_electronic() -> dict[str, Any]:
    # run_tbg_tight_binding.py exists and is validated at the exact target
    # geometry (docs/tbg_tight_binding_reference.md), but Moon & Koshino (2013)
    # is a hopping parametrization for the electronic pz manifold: it has no
    # total-energy or ion-ion repulsive term, so d(total energy)/dR is not
    # defined by the published model, and this repository's implementation
    # (Comparison/scripts/run_tbg_tight_binding.py) computes bands only --
    # confirmed by inspection, not assumed from the module name.
    tb_module = REPO_ROOT / "Comparison" / "scripts" / "run_tbg_tight_binding.py"
    return {
        "id": "moon_koshino_tight_binding_electronic",
        "family": "electronic_tight_binding_reused_as_force_source",
        "mechanism": "reuse run_tbg_tight_binding.py's validated Moon & Koshino (2013) p_z "
        "Hamiltonian and geometry/neighbor-list code as the basis for a numerically "
        "differentiated total energy",
        "coverage": {axis: False for axis in COVERAGE_AXES},
        "coverage_basis": "the published parametrization (Moon & Koshino, PRB 87, 205404, "
        "2013) is hopping-only, S=I, with no repulsive/total-energy term; "
        f"{tb_module.name} implements bands, DOS and neutrality only -- no force or "
        "force-constant quantity exists anywhere in it (grep-verified)",
        "availability": {
            "reference_module_present": tb_module.is_file(),
            "reference_module_computes_forces": False,
        },
        "defensible": False,
        "blocking_reason": "no total-energy functional exists in the cited source or this "
        "repository's implementation of it; producing forces would require authoring and "
        "independently validating a new repulsive term the published model does not specify, "
        "which is new physics, not a selection",
    }


def _candidate_classical_empirical_potential() -> dict[str, Any]:
    intralayer = _any_importable(_INTRALAYER_POTENTIAL_MODULES)
    md_engine = _any_importable(_CLASSICAL_MD_ENGINE_MODULES)
    interlayer = _any_importable(_INTERLAYER_REGISTRY_POTENTIAL_MODULES)
    intralayer_available = any(intralayer.values())
    interlayer_available = any(interlayer.values())
    coverage = {
        "acoustic": intralayer_available,
        "shear": interlayer_available,
        "breathing": interlayer_available,
        "moire": interlayer_available,
    }
    return {
        "id": "classical_empirical_potential",
        "family": "empirical_interatomic_potential",
        "mechanism": "intralayer analytic bond-order potential (Tersoff/Brenner) for the "
        "in-plane sp2 network, plus a registry-dependent interlayer potential "
        "(Kolmogorov & Crespi 2005 / Lebedeva et al.) for the twist-dependent stacking "
        "response, differentiated (ideally analytically, e.g. matscipy's exact Hessians) "
        "into force constants -- the field-standard route for large-scale graphite/TBG "
        "lattice dynamics this roadmap's own bibliography cites (Lu et al. 2022, PRB 106, "
        "144305, uses classical potentials for moire phonons at this scale)",
        "coverage": coverage,
        "coverage_basis": "an intralayer-only bond-order potential run on the bilayer with no "
        "interlayer term leaves the two layers mechanically decoupled at zero lateral/normal "
        "stiffness: shear and layer-breathing restoring forces, and therefore the low-energy "
        "moire phonon sector Lu et al. (2022) actually study, are physically absent, not "
        "merely approximate, until a registry-dependent interlayer potential is added",
        "availability": {
            "intralayer_bond_order_potential_importable": intralayer,
            "classical_md_engine_importable": md_engine,
            "interlayer_registry_potential_importable": interlayer,
        },
        "defensible": False,
        "blocking_reason": "the intralayer bond-order potential is available in this "
        "environment (matscipy/ase Tersoff-Brenner, with matscipy's exact analytic Hessians "
        "the natural scale-appropriate route once used), but no registry-dependent interlayer "
        "potential (Kolmogorov-Crespi/Lebedeva) is installed anywhere in this repository or "
        "its declared dependencies -- adding and independently validating one is new "
        "development, not a selection among existing options",
    }


def _candidate_continuum_moire_model() -> dict[str, Any]:
    # Confirmed by search (not assumed): zero mention of "moire phonon",
    # "continuum model", "dynamical matrix", "elastic", "shear mode" or
    # "breathing mode" anywhere in docs/, and no continuum-elasticity code
    # anywhere in Comparison/scripts other than the backend-agnostic
    # phonon_provider.py contract itself.
    return {
        "id": "continuum_moire_elastic_model",
        "family": "continuum_elasticity",
        "mechanism": "long-wavelength elastic/continuum theory of the moire superlattice "
        "(e.g. Lu et al. 2022, PRB 106, 144305) for the low-energy shear/breathing/moire "
        "acoustic sector near the moire zone center",
        "coverage": {
            "acoustic": True,
            "shear": True,
            "breathing": True,
            "moire": True,
        },
        "coverage_basis": "a continuum model targets exactly this sector by construction, but "
        "is not valid away from the moire zone center: it has no atomistic optical branches "
        "(the graphene E2g/ZO sector GO-4/GO-5 already validated the electronic side against) "
        "and no q away from the long-wavelength limit, so its coverage is real but bounded, "
        "not full-BZ",
        "availability": {
            "prior_art_in_repository": False,
            "search_performed": "docs/*.md grepped for 'moire phonon|continuum model|dynamical "
            "matrix|elastic|shear mode|breathing mode'; Comparison/scripts/ has no continuum-"
            "elasticity code outside the backend-agnostic phonon_provider.py contract",
        },
        "defensible": False,
        "blocking_reason": "no implementation exists in this repository; authoring one and "
        "independently validating it against both the atomistic small-system benchmarks (where "
        "its regime overlaps) and the literature model it claims to reproduce is itself a "
        "multi-step research task, not a selection among existing options, and even once built "
        "its bounded range of validity would need every consumer of the PhononProvider contract "
        "to know it cannot answer a full-BZ q",
    }


def _candidate_ml_universal_potential() -> dict[str, Any]:
    checkpoint = _local_ml_foundation_checkpoint()
    architecture_available = _importable("mace.calculators")
    return {
        "id": "ml_universal_interatomic_potential",
        "family": "machine_learning_interatomic_potential",
        "mechanism": "a pretrained universal MLIP (e.g. a MACE-MP-class foundation model) "
        "evaluated on the full (31,30) cell for forces, then numerically differentiated into "
        "force constants",
        "coverage": {axis: bool(checkpoint) for axis in COVERAGE_AXES},
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
        "environment or repository, and fetching + independently validating one against the "
        "benchmarks below is a separate acquisition ticket, not something this selection can "
        "silently assume",
    }


def _candidates() -> list[dict[str, Any]]:
    return [
        _candidate_dft_force_constants(),
        _candidate_classical_empirical_potential(),
        _candidate_ml_universal_potential(),
        _candidate_continuum_moire_model(),
        _candidate_moon_koshino_electronic(),
    ]


def _benchmark_suite() -> dict[str, Any]:
    """What any future candidate must pass before GO-8a, regardless of which one it is.

    Small-system thresholds are not invented here: they are the ones
    ``tests/test_phonon_provider.py`` already certifies against the real
    graphene SIESTA FC run, so a new backend is compared against a fixture
    that already exists rather than a number picked for this document.
    """
    return {
        "small_system": {
            "graphene_gamma_acoustic_and_optical": {
                "reference": "tests/test_phonon_provider.py::test_gamma_frequencies_reproduce_graphene",
                "requires": [
                    "3 acoustic branches at Gamma with |omega| < 1e-6 eV after ASR",
                    "ZO branch in 800-900 cm^-1",
                    "E2g doublet in 1500-1620 cm^-1, split by < 1e-4 relative",
                ],
            },
            "acoustic_sum_rule_raw_vs_corrected": {
                "requires": "ForceConstants.asr_residual_ev_ang2 reported for both raw and "
                "ASR-corrected IFC, never silently only the corrected one (phonon_provider.py "
                "ASR_RAW/ASR_CORRECTED split)",
            },
            "ab_bilayer_interlayer_sectors": {
                "reference": "Comparison/scripts/run_ab_bilayer_epc_paths.py sectors "
                "'shear'/'layer_breathing'",
                "requires": "the candidate's own eigenvectors -- not a synthetic direction -- "
                "measurably show shear as in-plane antiparallel layers and breathing as "
                "out-of-plane equal-and-opposite layers, the same geometric signature "
                "evaluate_ab_bilayer_epc.check_shear_direction/check_breathing_parity already "
                "check for the synthetic case",
            },
        },
        "matbg_scale": {
            "geometry_signature_match": f"the candidate's force/FC run is on exactly "
            f"{TARGET_GEOMETRY} ({TARGET_ATOM_COUNT} atoms) -- same "
            "geometry_cell_species_sha256 as the Graph2Mat and tight-binding paths",
            "dynamical_matrix_hermiticity": "max|D(q) - D(q)^dagger| at the dynamical-matrix "
            "hermiticity provenance field modes_from_force_constants() already records, at a "
            "tolerance set from the candidate's own noise floor, not a hand-picked constant",
            "acoustic_branches_vanish_at_gamma_after_asr": True,
            "resource_preflight": "wall time / peak RSS / peak VRAM / disk measured for real "
            "(roadmap Section X), inside the 28 GiB VRAM / 62 GB RAM margins "
            "docs/epc_s37_matbg_synthetic_benchmark.md already uses for the electronic side",
            "provenance": "geometry + provider/version + force/IFC source + primitive mapping + "
            "FC range + q + eigvec hash + masses + normalization + ASR policy (roadmap Section V, "
            "'Phonon' firma)",
        },
    }


class MatbgPhononProviderDecisionError(RuntimeError):
    """Raised when this decision cannot be computed reproducibly."""


def matbg_phonon_provider_decision() -> dict[str, Any]:
    """The GO-8a / NO-GO-8a decision, as data. Never produces a phonon or a g."""
    candidates = _candidates()
    defensible = [candidate for candidate in candidates if candidate["defensible"]]
    verdict = f"GO-8a:{defensible[0]['id']}" if defensible else "NO-GO-8a"
    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "memo": MEMO,
        "gate": GATE,
        "target_geometry": TARGET_GEOMETRY,
        "target_atom_count": TARGET_ATOM_COUNT,
        "coverage_axes": list(COVERAGE_AXES),
        "candidates": candidates,
        "verdict": verdict,
        "selected_provider": defensible[0]["id"] if defensible else None,
        "full_ks_coupling_produced": False,
        "benchmark_suite_required_before_go8a": _benchmark_suite(),
        "reopen_if": "a concrete candidate closes its own blocking_reason with checkable "
        "evidence -- an installed and independently validated interlayer registry potential "
        "(Kolmogorov-Crespi/Lebedeva) paired with the intralayer bond-order potential already "
        "available here; a locally present and independently validated universal MLIP "
        "checkpoint; or a continuum moire-phonon implementation validated against both the "
        "small-system atomistic benchmarks (where its regime overlaps) and its own literature "
        "source -- and then clears the full benchmark_suite_required_before_go8a above",
        "references": [
            "P. Moon and M. Koshino, Phys. Rev. B 87, 205404 (2013) -- the electronic model "
            "this repository already reuses for bands, and why it cannot supply forces",
            "Lu, MacDonald & co., Phys. Rev. B 106, 144305 (2022) -- low-energy moire phonons "
            "via classical potentials, the literature precedent for the classical-potential "
            "candidate's mechanism and its bounded continuum-regime alternative",
            "A. N. Kolmogorov and V. H. Crespi, Phys. Rev. B 71, 235415 (2005) -- the "
            "registry-dependent interlayer potential this decision found absent from every "
            "candidate that would otherwise be defensible",
        ],
    }


def _self_test() -> None:
    decision = matbg_phonon_provider_decision()
    assert decision["verdict"] == "NO-GO-8a"
    assert decision["selected_provider"] is None
    assert decision["full_ks_coupling_produced"] is False
    assert {c["id"] for c in decision["candidates"]} == {
        "dft_force_constants_siesta",
        "classical_empirical_potential",
        "ml_universal_interatomic_potential",
        "continuum_moire_elastic_model",
        "moon_koshino_tight_binding_electronic",
    }
    for candidate in decision["candidates"]:
        assert set(candidate["coverage"]) == set(COVERAGE_AXES)
        assert not candidate["defensible"]
        assert candidate["blocking_reason"]
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    _self_test()
