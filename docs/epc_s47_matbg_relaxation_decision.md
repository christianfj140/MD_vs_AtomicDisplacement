# E-F_001-S47 — relaxed MATBG geometry: GO-10 or NO-GO-10

Fase 11 of the roadmap asks for the *relaxed* magic-angle `(31, 30)` cell
(11 164 atoms) — forces, corrugation, AA/AB/BA domain areas, an
interlayer-spacing distribution — produced without reusing any artifact that
depends on the rigid geometry. This ticket is the *selection* step: name a
force source defensible for whole-bilayer relaxation at this scale, or refuse
to, without producing a relaxed geometry either way.

Producer: [`Comparison/scripts/decide_matbg_relaxation_provider.py`](../Comparison/scripts/decide_matbg_relaxation_provider.py)
(`matbg_relaxation_provider_decision()`). Tests:
[`tests/test_decide_matbg_relaxation_provider.py`](../tests/test_decide_matbg_relaxation_provider.py).
Depends on: `materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf` (the
rigid target geometry), [[c20-graphene-gamma-three-paths]] and
`docs/epc_s38_matbg_phonon_provider_decision.md` (the sibling GO-8a decision
this ticket mirrors and reuses the same live-import-probe method from).

Evidence conventions: **[CODE]** verified in this repository, **[MEAS]**
measured here (a live import probe, not a memory of what "usually" ships),
**[LIT]** a literature citation, **[DESIGN]** decision frozen in this
document.

## 1. Verdict

**NO-GO-10.** No candidate clears every relaxation mechanism with what is
actually present in this repository and its declared dependencies.
`matbg_relaxation_provider_decision()["relaxed_geometry_produced"]` is
hard-coded `False` so no downstream consumer can misread a refusal as a
result, and no relaxed-geometry artifact is written by this ticket.

## 2. The three candidates considered

| id | mechanism | in-plane relaxation | corrugation | AB/BA domains | blocked on |
| --- | --- | :-: | :-: | :-: | --- |
| `dft_relaxation_siesta` | SIESTA `MD.TypeOfRun CG/FIRE` ionic relaxation with converged SCF forces at every step | ✅ | ✅ | ✅ | **cost**: no converged full-SCF run has ever existed for this cell; a relaxation needs dozens of them |
| `classical_empirical_potential` | intralayer Tersoff/Brenner + interlayer Kolmogorov-Crespi, minimized | ✅ | ❌ | ❌ | **missing interlayer term**: no KC/Lebedeva potential in this environment |
| `ml_universal_interatomic_potential` | pretrained foundation MLIP, minimized | ❌ | ❌ | ❌ | **no checkpoint present**, coverage unknown without one |

Coverage is checked against Nam & Koshino's (2017) own mechanism: relaxation
below ~2° twist redistributes area from AA into AB/BA domains and produces
measurable corrugation, and both effects are driven by the *interlayer
registry* energy landscape, not by intralayer bond stiffness. A candidate
that cannot represent interlayer registry physics cannot produce domain
formation or corrugation, regardless of how well it relaxes each layer's
flat in-plane lattice.

## 3. `dft_relaxation_siesta` — right physics, a different and larger cost than S38's FC estimate

**[CODE]** `shared/siesta_run_fdf.py` already knows how to emit
`MD.TypeOfRun CG`/relaxation FDF blocks (used today for the unrelated
`AtomDisplacement/scripts/run_relaxation.py` H2O reference), and SIESTA force
convergence under a converged SCF loop is standard, certified methodology —
nothing about the *physics* of a DFT relaxation is missing on any mechanism.

**[MEAS]** The cost is not comparable to S38's `6 * N_atoms` displaced-FC
estimate — it is worse in a different way. `generate_siesta_overlap_only.py`
sets `MaxSCFIterations=0`, `TS.onlyS=T` for this exact cell precisely
*because* a converged SCF solve has never been attempted on it
(`reference_hamiltonian_generated=False` in its own manifest). A relaxation
needs a *converged* SCF force at every one of dozens of ionic steps.
`docs/epc_s37_matbg_synthetic_benchmark.md` §3 already measured that even the
electronic-only, no-SCF, single forward pass for this cell needs GPU plus
explicit MACE node/edge chunking just to avoid OOM — and that is the
Graph2Mat-only step, categorically cheaper than one converged DFT SCF cycle.
Multiplying an unattempted, already-marginal single step by dozens of ionic
steps is the roadmap's own standing definition of intractable at this scale.

## 4. `classical_empirical_potential` — same missing piece as S38, now blocking relaxation instead of phonons

**[LIT]** Nam & Koshino, *Phys. Rev. B* 96, 075311 (2017) relax TBG with
exactly the field-standard combination: an intralayer bond-order potential
plus a registry-dependent interlayer potential (Kolmogorov & Crespi 2005 /
Lebedeva et al.), and report that the interlayer term is what drives AB/BA
domain formation and corrugation below ~2° twist.

**[MEAS]** Reusing the same live probe as `decide_matbg_phonon_provider.py`:
`ase.calculators.tersoff` and `matscipy.calculators.manybody` import
successfully (intralayer, available today); `kimpy`, `lammps` and `openmm` do
not import (no route to a registry-dependent interlayer potential, and no
classical MD engine that could load one via OpenKIM).

**[DESIGN]** Coverage: `in_plane_lattice_relaxation = True` (the intralayer
potential alone can relax each layer's flat lattice); `out_of_plane_corrugation
= ab_ba_domain_formation = False`. Minimizing the bilayer under an
intralayer-only potential leaves the two layers mechanically decoupled at
zero lateral/normal interlayer stiffness — there is no restoring force to
redistribute AA area into AB/BA domains or to buckle the layers out of plane
by registry, so both target observables of this ticket are **physically
absent from that calculation, not merely approximated**. This is not a new
finding: it is the same blocking condition S38 already established for
phonons, now shown to block relaxation itself, since domain formation and
phonon shear/breathing restoring forces come from the same missing physics.

## 5. `ml_universal_interatomic_potential` — architecture present, no trained weights

**[MEAS]** `mace.calculators` imports successfully (an architecture library
already used to build this repository's own Graph2Mat Hamiltonian model, not
a trained total-energy foundation model). `_local_ml_foundation_checkpoint()`
checks `$MACE_MODEL_PATH`, `$MATBG_FOUNDATION_POTENTIAL` and
`~/.cache/mace/*`; none is present.

**[DESIGN]** Fetching a foundation checkpoint and independently validating
that its training distribution covers twisted-bilayer, registry-dependent
carbon before trusting its forces for relaxation is a separate acquisition
ticket, out of scope for this selection.

## 6. Acceptance criteria recorded now for whichever candidate eventually clears GO-10

Recorded so a later relaxation attempt is judged against fixed criteria, not
criteria invented after the fact
(`matbg_relaxation_provider_decision()["acceptance_criteria_for_a_future_relaxation"]`):
force-convergence tolerance and stopping criterion preregistered before the
run; final forces on every atom; the out-of-plane corrugation distribution;
AA/AB/BA area fractions; the full interlayer-spacing *distribution* (not a
single mean, since non-uniform spacing is exactly the point); a new
`geometry_cell_species_sha256` (`shared/reference_provenance.py`) distinct
from the rigid cell's, so it invalidates rather than silently reuses any
rigid-geometry-keyed H/S/eigenpair/phonon/derivative artifact; full
generator/input/trajectory/final-geometry provenance signing; and a
`'relaxed'` status label kept distinct from the rigid campaign's
`'rigid_tbg_model'` status.

## 7. Reopen condition

GO-10 reopens when a concrete candidate closes its own `blocking_reason` with
checkable evidence — an installed and independently validated interlayer
registry potential (Kolmogorov-Crespi/Lebedeva) paired with the intralayer
bond-order potential already available here, reachable from a classical MD
engine or an ASE-compatible calculator; or a locally present and
independently validated universal MLIP checkpoint whose training
distribution is shown to cover twisted-bilayer carbon — and then clears §6's
acceptance criteria. `matbg_relaxation_provider_decision()` is a live probe,
not a cached verdict: installing any of the above and rerunning it flips
`verdict` to `GO-10:<candidate id>` automatically
(`test_a_candidate_would_flip_the_verdict_if_it_became_defensible`).

## 8. Reproduce

```bash
.venv/bin/python Comparison/scripts/decide_matbg_relaxation_provider.py
.venv/bin/pytest -q tests/test_decide_matbg_relaxation_provider.py
```
