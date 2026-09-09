# D05B / E-F_001-S38 — scalable MATBG `PhononProvider`: GO-8a or NO-GO-8a

Fase 10a of the roadmap: before any displacement on the rigid `(31, 30)`
magic-angle cell (11 164 atoms) may be called a physical phonon eigenmode
rather than a `synthetic_test_displacement`, a scalable, validated source of
`omega(q,nu)` and `e(q,nu)` for that exact geometry has to exist. This ticket
is the *selection* step: compare candidate providers on literature grounding,
coverage (acoustic / shear / breathing / moire — the roadmap's own axes),
cost and provenance, and either name one or refuse to, without producing a
phonon or a PAO-projected coupling either way.

Producer: [`Comparison/scripts/decide_matbg_phonon_provider.py`](../Comparison/scripts/decide_matbg_phonon_provider.py)
(`matbg_phonon_provider_decision()`). Tests:
[`tests/test_decide_matbg_phonon_provider.py`](../tests/test_decide_matbg_phonon_provider.py).
Depends on: `materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf` (the
target geometry), [[tbg_tight_binding_reference]] (`run_tbg_tight_binding.py`,
one of the candidates), `Comparison/scripts/phonon_provider.py` (the contract
every candidate must eventually satisfy).

Evidence conventions: **[CODE]** verified in this repository, **[MEAS]**
measured here (a live import probe, not a memory of what "usually" ships),
**[LIT]** a literature citation, **[DESIGN]** decision frozen in this
document.

---

## 1. Verdict

**NO-GO-8a.** No candidate clears every coverage axis with what is actually
present in this repository and its declared dependencies. This is the
deliverable, not a stalled ticket: the roadmap explicitly allows it ("Si
ninguna opción es defendible, se emite NO-GO-8a sin producir acoplamiento proyectado PAO"), and
`matbg_phonon_provider_decision()["full_ks_coupling_produced"]` is hard-coded
`False` so no downstream consumer can misread a refusal as a result.

## 2. The five candidates considered

| id | mechanism | acoustic | shear | breathing | moire | blocked on |
| --- | --- | :-: | :-: | :-: | :-: | --- |
| `dft_force_constants_siesta` | exact SIESTA `FC.Save.dHS` on the production cell | ✅ | ✅ | ✅ | ✅ | **cost**: 66 984 displaced SCF solves |
| `classical_empirical_potential` | intralayer Tersoff/Brenner + interlayer Kolmogorov-Crespi | ✅ | ❌ | ❌ | ❌ | **missing interlayer term**: no KC/Lebedeva potential in this environment |
| `ml_universal_interatomic_potential` | pretrained foundation MLIP, differentiated for forces | ❌ | ❌ | ❌ | ❌ | **no checkpoint present**, coverage unknown without one |
| `continuum_moire_elastic_model` | Lu et al. (2022)-style long-wavelength elasticity | ✅ | ✅ | ✅ | ✅ | **no implementation exists**; and bounded to the moire zone center even once built |
| `moon_koshino_tight_binding_electronic` | reuse `run_tbg_tight_binding.py`'s validated electronic model as a force source | ❌ | ❌ | ❌ | ❌ | **no total-energy functional** in the cited source or this repository |

Coverage is a claim about physical mechanism, checked against what evidence
actually supports it (§3–§7), not a wishlist.

## 3. `dft_force_constants_siesta` — right physics, wrong cost

**[CODE]** SIESTA `FC.Save.dHS` is already the certified route at graphene
scale: C09/C18 (`docs/epc_c18_graphene_gamma_phonons.md`) validated it against
explicit finite differences on a 2-atom cell. Nothing about the *physics* of
an exact DFT dynamical matrix is missing on any coverage axis.

**[MEAS]** The cost is not: `FC.Displacement` moves each atom `+/-` along
`x, y, z`, i.e. `6 * N_atoms` displaced self-consistent solves —
`6 * 11164 = 66984` for this cell. `docs/epc_s37_matbg_synthetic_benchmark.md`
§3 already measured that the *electronic-only, no-SCF, overlap-only* step for
this same cell needs GPU plus explicit MACE node/edge chunking just to avoid
OOM on a single forward pass. A full SCF FC campaign is tens of thousands of
independent DFT solves on top of that. This is the roadmap's own standing
definition of "intractable at this scale" (roadmap §I, `run_tbg_pure_graph2mat_campaign.py`'s
own contract: "no existe Hamiltoniano DFT de referencia para ese target"),
not a new estimate invented for this document.

## 4. `classical_empirical_potential` — the literature-standard route, missing one piece

**[LIT]** Classical potentials are the field-standard tool for large-scale
graphite/TBG lattice dynamics; this repository's own roadmap bibliography
already cites Lu et al., *Phys. Rev. B* 106, 144305 (2022) for low-energy
moire phonons at this kind of scale. The standard combination is an
intralayer analytic bond-order potential (Tersoff/Brenner-family) for the
in-plane `sp2` network plus a registry-dependent interlayer potential —
Kolmogorov & Crespi, *Phys. Rev. B* 71, 235415 (2005), or Lebedeva et al. —
for the twist-dependent stacking response that actually produces the shear
and breathing restoring forces.

**[MEAS]** Live import probe in this environment:
`ase.calculators.tersoff` and `matscipy.calculators.manybody` both import
successfully — the intralayer half, including matscipy's exact analytic
Hessians (the scale-appropriate way to build force constants for a system
this large without finite-difference noise or cost), is available today.
`kimpy` (the only route to an OpenKIM-distributed Kolmogorov-Crespi model
this repository's dependencies expose) does not import; neither does
`lammps` or `openmm`. **No registry-dependent interlayer potential is
installed anywhere in this repository or its declared dependencies.**

**[DESIGN]** Coverage: `acoustic = True` (the intralayer potential alone
gives a physically meaningful in-plane/out-of-plane acoustic branch);
`shear = breathing = moire = False`. Running the bilayer through an
intralayer-only potential with no interlayer term at all leaves the two
layers mechanically decoupled — zero lateral and normal stiffness between
them — so the shear and layer-breathing restoring forces, and with them the
low-energy moire phonon sector Lu et al. (2022) actually studies, are
**physically absent from that calculation, not merely approximated**. Adding
and independently validating an interlayer registry potential is new
development, not a selection among what already exists here.

## 5. `ml_universal_interatomic_potential` — architecture present, no trained weights

**[MEAS]** `mace.calculators` imports successfully — `mace-torch` is already
a declared dependency, because it builds this repository's own Graph2Mat
Hamiltonian architecture. That is an *architecture* library, not a trained
total-energy foundation model. `_local_ml_foundation_checkpoint()` checks
`$MACE_MODEL_PATH`, `$MATBG_FOUNDATION_POTENTIAL` and `~/.cache/mace/*` for
an already-downloaded checkpoint; none is present. No weights exist here to
inspect, let alone validate against the benchmarks in §8, so coverage on
every axis is reported unknown-as-`False` rather than assumed.

**[DESIGN]** Fetching a foundation checkpoint is out of scope for a selection
ticket: it is a separate acquisition-plus-validation ticket (network fetch,
license terms, and — most importantly — independent validation that its
training distribution actually covers twisted-bilayer, registry-dependent
carbon environments before its forces are trusted for this system).

## 6. `continuum_moire_elastic_model` — the right regime, unbuilt and inherently bounded

**[MEAS]** `docs/*.md` grepped for `moire phonon|continuum model|dynamical
matrix|elastic|kirigami|Lu et al|shear mode|breathing mode` returns no
matches anywhere except `phonon_provider.py`'s own dynamical-matrix
machinery. There is no continuum-elasticity code anywhere in
`Comparison/scripts/` beyond the backend-agnostic contract itself.

**[DESIGN]** Coverage is real but bounded even if built: a continuum model of
this kind targets exactly the low-energy shear/breathing/moire-acoustic
sector near the moire zone center by construction, but has no atomistic
optical branches (the graphene E2g/ZO sector GO-4/GO-5 already validated the
electronic side against) and no `q` away from the long-wavelength limit.
Authoring one and independently validating it against both the small-system
atomistic benchmarks (§8, where its regime overlaps) and its own literature
source is itself a multi-step research task, not a choice among existing
options — and every future consumer of the `PhononProvider` contract would
need to know its `q`-range is bounded, which the contract's `fc_range` field
already has a place for but nothing currently populates.

## 7. `moon_koshino_tight_binding_electronic` — reuses the wrong half of an existing asset

**[CODE]** `run_tbg_tight_binding.py` is validated at the exact target
geometry (`docs/tbg_tight_binding_reference.md`) and is the only other
large-cell asset in this repository. But Moon & Koshino, *Phys. Rev. B* 87,
205404 (2013) is a hopping parametrization for the electronic `p_z`
manifold — one orbital per carbon, `S = I`, no ion-ion repulsive or
total-energy term. Grep-verified: `run_tbg_tight_binding.py` computes bands,
DOS and a neutrality Fermi level; no force or force-constant quantity exists
anywhere in it. Producing forces from this model would mean authoring and
independently validating a repulsive term the published source does not
specify — new physics, not a reuse.

## 8. Benchmark suite any future candidate must clear

Recorded now so a later attempt is judged against fixed criteria, not
criteria invented after the fact (`matbg_phonon_provider_decision()["benchmark_suite_required_before_go8a"]`):

**Small system** (reuses existing fixtures, no new numbers):

- `graphene_gamma_acoustic_and_optical` — the exact thresholds
  `tests/test_phonon_provider.py::test_gamma_frequencies_reproduce_graphene`
  already certifies against the real SIESTA FC run: 3 acoustic branches
  `< 1e-6 eV` after ASR, ZO in 800–900 cm⁻¹, E2g doublet in 1500–1620 cm⁻¹
  split by `< 1e-4` relative;
- `acoustic_sum_rule_raw_vs_corrected` — both the raw and ASR-corrected
  residual reported, per the existing `ASR_RAW`/`ASR_CORRECTED` split;
- `ab_bilayer_interlayer_sectors` — the candidate's own eigenvectors, not a
  synthetic direction, must measurably show shear as in-plane antiparallel
  layers and breathing as out-of-plane equal-and-opposite layers, the same
  geometric signature `evaluate_ab_bilayer_epc.check_shear_direction` /
  `check_breathing_parity` already check for the synthetic case in
  `run_ab_bilayer_epc_paths.py`.

**MATBG scale**:

- `geometry_signature_match` — the candidate's force/FC run is on exactly
  `materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf` (11 164 atoms),
  same `geometry_cell_species_sha256` as the Graph2Mat and tight-binding
  paths;
- `dynamical_matrix_hermiticity` — `modes_from_force_constants()` already
  records this provenance field; a candidate's own noise floor sets the
  tolerance, not a hand-picked constant;
- acoustic branches vanish at Γ after ASR;
- `resource_preflight` — wall time / peak RSS / peak VRAM / disk measured
  for real, inside the 28 GiB VRAM / 62 GB RAM margins
  `docs/epc_s37_matbg_synthetic_benchmark.md` already established for the
  electronic side;
- provenance per roadmap Section V's "Phonon" firma: geometry + provider/
  version + force/IFC source + primitive mapping + FC range + q + eigvec
  hash + masses + normalization + ASR policy.

## 9. Reopen condition

GO-8a reopens when a concrete candidate closes its own `blocking_reason`
with checkable evidence — an installed and independently validated
interlayer registry potential (Kolmogorov-Crespi/Lebedeva) paired with the
intralayer bond-order potential already available here; a locally present
and independently validated universal MLIP checkpoint; or a continuum
moire-phonon implementation validated against both the small-system
atomistic benchmarks and its own literature source — and then clears the
full benchmark suite in §8. `matbg_phonon_provider_decision()` is a live
probe, not a cached verdict: installing any of the above and rerunning it
flips `verdict` to `GO-8a:<candidate id>` automatically
(`test_a_candidate_would_flip_the_verdict_if_it_became_defensible`).

## 10. Reproduce

```bash
.venv/bin/python Comparison/scripts/decide_matbg_phonon_provider.py
.venv/bin/pytest -q tests/test_decide_matbg_phonon_provider.py
```
