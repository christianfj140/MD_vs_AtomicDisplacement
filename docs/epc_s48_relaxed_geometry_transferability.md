# E-F_001-S48 — Graph2Mat transferability across local stacking registries (a bound, not GO-10)

Fase 11 of the roadmap requires, before any quantitative relaxed-MATBG claim,
that Graph2Mat's structure handling, `H` prediction and directional
derivatives pass a **transferability** check, not merely a bands-match check.
[`docs/epc_s47_matbg_relaxation_decision.md`](epc_s47_matbg_relaxation_decision.md)
(S47) already found **NO-GO-10**: no relaxation provider in this environment
is defensible for the production `(31, 30)` cell, so no relaxed MATBG
geometry exists, and none is produced by this ticket either. This ticket is
the fallback Fase 11 itself names: "directional-derivative validation
Graph2Mat en muestras locales/sistemas pequeños representativos".

Producer:
[`Comparison/scripts/certify_relaxed_geometry_transferability.py`](../Comparison/scripts/certify_relaxed_geometry_transferability.py)
(`certify()`). Tests:
[`tests/test_certify_relaxed_geometry_transferability.py`](../tests/test_certify_relaxed_geometry_transferability.py).
Depends on: `docs/epc_s47_matbg_relaxation_decision.md` (the GO-10 status this
ticket reads but cannot change),
[[c20-graphene-gamma-three-paths]] (GO-3, the internal JVP validation this
ticket runs on new geometries), and
[[checkpoint-pointbasis-R-drift]] (the known basis-table drift this ticket
treats as non-blocking, matching `shared/run_inventory.py`'s own
`DECLARED_ONLY -> PRESENT_UNVERIFIED` mapping).

## 1. What "local samples representative of the relaxed geometry" means here

Nam & Koshino, *Phys. Rev. B* 96, 075311 (2017) — the same reference S47 used
to reject every relaxation candidate lacking interlayer registry physics —
identify AA, AB and BA as exactly the stacking registries a relaxed twisted
bilayer locally interpolates between below ~2° twist. This repository already
has all three as independent commensurate bilayer materials
(`materials/bilayer_graphene_AA`, `materials/bilayer_graphene_AB`,
`materials/bilayer_graphene_BA`, 4 atoms / 16 orbitals each). Certifying the
checkpoint's structure/H/derivative machinery across those three registries is
a real, cheap test of local-environment transfer, in the specific sense Fase
11 asks for when no relaxed geometry exists to test on directly.

**This is not GO-10.** GO-10 additionally requires a relaxed geometry with
its own provenance and a model-vs-SIESTA accuracy comparison on it (S47 §6).
This ticket cannot produce either, since S47 is NO-GO-10. The status label
this ticket can license is `local_stacking_transferability_bound`, and the
report sets `licenses_go10: false` and
`quantitative_relaxed_matbg_claim_licensed: false` unconditionally — a PASS
verdict here narrows the risk before a relaxed geometry exists; it does not
substitute for one.

## 2. Three checks per material, all reused, none new

- **`structure`**: `shared/orbital_contract.certify_systems()` (C03) — the
  same basis/species/orbital contract machinery already certifying graphene,
  AB and the rigid TBG target, run here against AA/AB/BA with the production
  checkpoint.
- **`h`**: the checkpoint's forward Hamiltonian prediction on each material's
  geometry is finite and produces a well-formed sparse `(row, col, R)`
  support, reusing
  `quantify_checkpoint_derivative_error.load_model_and_batch` /
  `model_predictions` (C14's own model-loading path). No SIESTA reference
  exists for AA/BA (only AB has a `dHSdR.nc` campaign, S34/S35), so this is a
  well-formedness check, not a DFT-accuracy one.
- **`derivative`**: GO-3
  (`validate_graph2mat_jvp.validate_directional_jvp`) — the directional JVP
  equals the coordinate-JVP combination and equals the model's own frozen
  central difference, on this material's own neighbour topology, for the
  default direction set plus this material's stacking-specific patterns
  (`shear`, `layer_breathing`, `intralayer_control`), reused unmodified from
  `run_ab_bilayer_epc_paths.build_directions` — a function already generic
  over any bilayer geometry (it derives layers from atomic `z`, never from an
  AB-specific index).

## 3. The `DECLARED_ONLY` orbital-contract status is not a failure here

Running C03 against this checkpoint gives `DECLARED_ONLY` for **every**
material already in production under it — graphene, AB, and the rigid
`(31,30)` target — not only for AA/BA:

```
graphene                              DECLARED_ONLY
bilayer_graphene_AB                   DECLARED_ONLY
twisted_bilayer_graphene_1p084549deg  DECLARED_ONLY
```

The cause is the known checkpoint `PointBasis.R` / edge-cutoff drift
([[checkpoint-pointbasis-R-drift]]): the checkpoint's stored radial cutoff and
the installed sisl's rebuilt cutoff differ for the same `.ion.xml`, an
`absent`-level informational check, not a species/count/order mismatch.
`shared/run_inventory.py` already treats `DECLARED_ONLY` as
`PRESENT_UNVERIFIED`, not as invalid. `certify_material`'s structure check
follows that same convention: it fails only on `orbital_contract.INVALID`, and
records the exact status rather than silently upgrading or hiding it.
Requiring strict `VERIFIED` here would fail this ticket for a condition that
applies to every material in the repository, not to anything specific to
relaxed-geometry transferability.

## 4. Result

Both backends were run (`--backend cpu` and the default `--backend cuda`,
which passed its preflight with no fallback). All three materials:

```
bilayer_graphene_AA   status=PASS   structure=DECLARED_ONLY  h=finite (2816 nnz)  go3=PASS (6 directions)
bilayer_graphene_AB   status=PASS   structure=DECLARED_ONLY  h=finite (2784 nnz)  go3=PASS (6 directions)
bilayer_graphene_BA   status=PASS   structure=DECLARED_ONLY  h=finite (2784 nnz)  go3=PASS (6 directions)
verdict: PASS  (GO-10: NO-GO-10)
```

The checkpoint's structure handling, forward `H` and GO-3 internal derivative
consistency transfer across all three local stacking registries. GO-10
remains NO-GO-10 (S47) regardless: no relaxed geometry exists, so no
model-vs-SIESTA accuracy comparison on one has been or could be run here.

## 5. Reopen condition

Re-run whenever the checkpoint, the AA/AB/BA material files, or `decide_matbg_relaxation_provider`'s
verdict change — a live measurement, not a cached one. If S47 ever flips to a
`GO-10:<candidate>` verdict, a follow-up ticket must additionally validate
transferability *on the produced relaxed geometry itself* (S47 §6's
acceptance criteria) before this bound's PASS can be treated as covering it.

## 6. Reproduce

```bash
.venv/bin/python Comparison/scripts/certify_relaxed_geometry_transferability.py --backend cpu
.venv/bin/pytest -q tests/test_certify_relaxed_geometry_transferability.py
```
