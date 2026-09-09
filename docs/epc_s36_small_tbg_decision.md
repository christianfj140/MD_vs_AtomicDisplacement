# D05 / E-F_001-S36 — small commensurate TBG: UNDECIDED

Fase 9 of the roadmap is conditional: build a small commensurate twisted
bilayer only if graphene K (GO-5) and AB bilayer (GO-7) leave a twist- or
moiré-mapping ambiguity that a smaller cell would newly answer. "More
complexity" is explicitly not an acceptable reason on its own (roadmap Fase
9, gate 7: *"Debe responder una pregunta nueva"*).

Producer: [`Comparison/scripts/decide_small_tbg.py`](../Comparison/scripts/decide_small_tbg.py)
(`small_tbg_decision()`). Tests:
[`tests/test_decide_small_tbg.py`](../tests/test_decide_small_tbg.py).
Depends on: [[epc_s31_graphene_k_go5_verdict]] (GO-5), the GO-7 verdict
written by `evaluate_ab_bilayer_epc.py`, [[tbg_tight_binding_reference]].

Evidence conventions: **[CODE]** verified in this repository, **[MEAS]**
measured here or in a cited artifact, **[DESIGN]** decision frozen in this
document.

---

## 1. Verdict

**UNDECIDED / BLOCKED.** GO-5 y GO-7 deben ser ambos `PASS` antes de emitir
`RUN` o `SKIP_WITH_JUSTIFICATION`; actualmente no lo son.

No unresolved twist/moiré-mapping ambiguity survives graphene K + AB +
the existing production TBG geometry. Building a separate small TBG would
not retire any risk those three don't already retire; it would re-measure
blockers (C14C, GO-8a) that are identical regardless of cell size, exactly
what [[epc_s31_graphene_k_go5_verdict]] §3 already refused to do for K
itself ("medir el mismo fallo dos veces con la esperanza de que la celda
más grande lo disimule").

## 2. What the question actually is

The one geometric object neither K nor AB can exercise is a coordinate
transformation with a genuine *relative rotation* between two commensurate
sublattices sharing one supercell — i.e. two independent per-layer
commensuration matrices folded into a single moiré cell, with its own
ORB_INDX/R-vector mapping and a rotation-center choice that has no analogue
in a single-lattice supercell or an untwisted stack. That is the specific
thing Fase 9 exists to test if it is still open.

## 3. Risks already closed by GO-5

**[CODE]** `epc_commensurate_k.go5_decision()` reports the algebra layer as
`PASS` on five checks, all at machine precision, all on a toy chain but all
generic to *any* commensurate supercell regardless of how many rotated
sublattices produced it (Sec. 4 of [[epc_s30_graphene_k_conmensurable]]):

| check | closes |
| --- | --- |
| `k_to_k_plus_q_by_subspace` | selecting `k`/`k+q` by Bloch character, not band index, inside a folded supercell |
| `q_and_minus_q` | `D_H(-q) = conj(D_H(q))` survives the full `k -> k+q` contraction, not just the raw halves |
| `cell_origin_shift` | relabelling which primitive cell is "cell 0" leaves the gauge-invariant `g` block unchanged |
| `atom_across_periodic_boundary` | the same relabelling is exactly an atom crossing the supercell boundary |
| `alternative_atomic_phase_convention` | an external atomic-phase convention (DFTBephy-style) round-trips through the canonical one to `1e-12` |

These five are the entire "does the supercell/complex-mode/phase machinery
work" question. A moiré cell reuses this machinery verbatim once its
positions and `q` are supplied — it does not need its own algebra test.

**Physical layer**: `NOT_ATTEMPTED`, `blocked_by = GO-4` (C14C, the
intra-atomic basis-response term). Not q-specific, not twist-specific —
identical block for Γ, K, AB, or any TBG.

## 4. Risks not yet closed by GO-7

**[CODE]** `evaluate_ab_bilayer_epc.py`'s GO-7 verdict (read as an artifact,
not recomputed here — the same C02 reuse rule) has one failing check out of
five. The four that pass are precisely the multi-layer geometry-construction
risks Fase 9 is worried about; the one that fails is not:

| check | layer | passed |
| --- | --- | --- |
| `spatial_layer_swap_from_geometry` | `geometry_construction` | **False / evidence missing** |
| `shear_direction_in_plane_antiparallel` | `geometry_construction` | **True** |
| `breathing_parity_layers_equal_and_opposite` | `reference` | **True** |
| `interlayer_derivative_blocks_match_sector_role` | `reference` | **True** |
| `phonon_mode_match_robust_to_branch_ambiguity` | `phonons` | False |

La antigua igualdad de normas `layer_swap_hermitian` era tautológica por
hermiticidad y ya no adjudica GO-7. Falta el test espacial obligatorio derivado
de celda, posiciones, imágenes, orbitales y k/q. También falla la capa de fonones
(branch-matching robustness, GO-8a territory), not to geometry or layer
construction, so GO-7's overall `NO_GO` does not leave a layer-mapping
question open.

## 5. Twist-specific mapping at production scale

**[MEAS]** Neither GO-5 nor GO-7 involves an actual relative rotation. What
closes that specific gap is that the production magic-angle geometry already
exists and has already been cross-checked independently:
`materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf` is the *same*
geometry file consumed by both `run_tbg_pure_graph2mat_campaign.py` (the
Graph2Mat/SIESTA-overlap pipeline) and `run_tbg_tight_binding.py` (an
independent Moon–Koshino `p_z` model, `S = I`, no shared code path). The
tight-binding run measures, rather than assumes, the magic-angle signature
on that exact geometry:

```text
hbar*v_F monolayer = 5.1515 eV.Ang
hbar*v* moire       = 0.0627 eV.Ang
v*/v_F              = 0.012
```

([[tbg_tight_binding_reference]]).
A ~10% angle error would leave this ratio at tens of percent, so this is not
a coincidence: the per-layer commensuration matrices, the twist construction,
and the resulting atom positions that ORB_INDX mapping depends on are already
producing physically correct band structure at the real 11 164-atom target,
via a method that shares no code with Graph2Mat/SIESTA. A smaller "practice"
TBG would not test a hypothesis this evidence has not already tested — it
would build a second geometry with its own new signature to re-answer a
question the production cell already answered.

## 6. Why this is not "we didn't want a more complex system"

Every risk closure above cites a specific check, its pass/fail value, and
the artifact it came from (`decide_small_tbg.small_tbg_decision()` returns
this as data, not prose — `tests/test_decide_small_tbg.py` asserts a broken
mapping check flips the verdict to `RUN`). The two things that remain open
after this analysis — GO-4's C14C intra-atomic term and GO-7's phonon
branch-matching robustness — are named explicitly and are *not* closed by
this decision; they are simply not evidence for building a small TBG, because
neither depends on twist, layer count, or cell size.

## 7. Reopen condition

If GO-8a/GO-8b work later surfaces a genuinely twist-specific ambiguity that
the production geometry + Moon–Koshino cross-check did not exercise — for
example a PAO-projected coupling that depends on the arbitrary choice of rotation
center, which the tight-binding band-structure check would not detect — then
build the small commensurate cell as an **algebra-only toy**, analogous to
S30's two-atom chain, not as a PAO-covariant coupling campaign (GO-1/GO-4 still block
physical numbers regardless of cell size). At that point:

- **cell**: the smallest commensurate `(m, n)` TBG pair with a twist angle
  large enough to keep atom count in the low hundreds (not `(31, 30)`);
  generated by the same commensuration formula `run_tbg_pure_graph2mat_campaign.py`
  already uses, at a different `(m, n)`, so it inherits the same generator
  and provenance path rather than a new one;
- **generator**: reuse `run_tbg_pure_graph2mat_campaign.py`'s geometry-
  construction step directly (parametrized by `(m, n)`), not a new script;
- **provenance**: geometry signature = `cell + species + positions +
  coordinate convention` (Section V of the roadmap), tagged with the
  `(m, n)` pair and twist angle, same as every other geometry artifact;
- **question**: is the gauge-invariant physical block (`g` singular values /
  Frobenius norm, GO-6 metrics) invariant under an arbitrary choice of
  rotation center for the same `(m, n)` — the one check K and AB cannot
  express because neither has two independently rotated sublattices;
- **limit**: no SCF/FC generation — this is a geometry + toy-mode algebra
  test only, same class of cost as S30's chain (seconds to minutes), well
  under overnight;
- no training dataset is produced by this or any future small-TBG step.

## 8. Reproduce

```bash
.venv/bin/python Comparison/scripts/decide_small_tbg.py
.venv/bin/pytest tests/test_decide_small_tbg.py
```

(Requires `Comparison/results/epc/gamma_epc/bilayer_graphene_AB/go7_verdict.json`
to exist — run `run_ab_bilayer_epc_paths.py` then `evaluate_ab_bilayer_epc.py`
first if it is missing; this module never infers a verdict that was not
written.)
