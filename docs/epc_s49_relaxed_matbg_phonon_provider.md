# E-F_001-S49 — relaxed-MATBG `PhononProvider` bound (Fase 11)

The roadmap objective is to "producir y certificar un `PhononProvider` cuya
geometry signature corresponda exactamente a la MATBG relajada". Two
independent gates already answer whether that can exist today, both as data
this ticket imports and reuses rather than recomputing:
[`docs/epc_s47_matbg_relaxation_decision.md`](epc_s47_matbg_relaxation_decision.md)
(S47, `decide_matbg_relaxation_provider.matbg_relaxation_provider_decision`)
found **NO-GO-10** — no relaxed MATBG geometry exists — and
[`docs/epc_s39_matbg_phonon_provider_certification.md`](epc_s39_matbg_phonon_provider_certification.md)
(S39, `certify_matbg_phonon_provider.certify_matbg_phonon_provider_go8a`)
found **NO-GO-8a** — no MATBG-scale `PhononProvider` exists at all, on any
geometry. Either alone blocks a `relaxed` `physical_phonon` artifact; this
ticket cannot produce one and does not fabricate a relaxed geometry or a
MATBG phonon to work around either refusal.

Producer:
[`Comparison/scripts/certify_relaxed_matbg_phonon_provider.py`](../Comparison/scripts/certify_relaxed_matbg_phonon_provider.py)
(`certify_relaxed_matbg_phonon_provider_bound()`,
`require_relaxed_matbg_physical_phonon()`). Tests:
[`tests/test_certify_relaxed_matbg_phonon_provider.py`](../tests/test_certify_relaxed_matbg_phonon_provider.py).
Depends on: `decide_matbg_relaxation_provider.py` (S47),
`certify_matbg_phonon_provider.py` (S39/S38), `phonon_provider.py` (the
`PhononProvider` contract), `epc_subspaces.py` (the gauge-invariant
eigenvector comparison), and the archived graphene-Gamma and AB-bilayer FC
runs this ticket measures.

Evidence conventions: **[CODE]** verified in this repository, **[MEAS]**
measured here against real archived data, **[DESIGN]** decision frozen in
this document.

---

## 1. Verdict

**Compound NO-GO.** `certify_relaxed_matbg_phonon_provider_bound()` reports
`go10_status = "NO-GO-10"`, `go8a_status = "NO-GO-8a"`,
`relaxed_geometry_available = False` and
`relaxed_physical_phonon_produced = False` unconditionally in this
repository's current state; `relaxed_physical_phonon` is `None`.
`require_relaxed_matbg_physical_phonon()` raises
`RelaxedMatbgPhononBoundError` on this report, so no downstream consumer can
misread the bound below as a usable phonon.

**[MEAS]** Reproduced:

```
$ .venv/bin/python Comparison/scripts/certify_relaxed_matbg_phonon_provider.py
[E-F_001-S49] relaxed-MATBG PhononProvider bound: PASS  ->  Comparison/results/epc/certification/matbg/s49_relaxed_matbg_phonon_provider_bound.json
[E-F_001-S49]   go10=NO-GO-10  go8a=NO-GO-8a  relaxed_physical_phonon_produced=False
[E-F_001-S49]   displacement_sensitivity_suite: PASS
[E-F_001-S49]     [ok  ] graphene_gamma_d0p005: ...
[E-F_001-S49]     [ok  ] graphene_gamma_d0p01: ...
[E-F_001-S49]     [ok  ] graphene_gamma_d0p02: ...
[E-F_001-S49]   local_stacking_phonon_suite: PASS
[E-F_001-S49]     [n/a ] bilayer_graphene_AA_phonon: no archived FC run under .../bilayer_graphene_AA/runs; ...
[E-F_001-S49]     [ok  ] bilayer_graphene_AB_phonon: ...
[E-F_001-S49]     [n/a ] bilayer_graphene_BA_phonon: no archived FC run under .../bilayer_graphene_BA/runs; ...
```

The top-level `PASS` is the **bound's own verdict**, not GO-8a and not GO-10:
it means the two measurable suites below hold against the real data this
repository has today. It is printed and stored distinctly from `go10_status`
/ `go8a_status` precisely so it cannot be misread as either gate passing.

## 2. What this ticket can still measure without a relaxed geometry or a MATBG provider

Fase 11's Requirements name three things: reevaluate frequencies,
eigenvectors, masses, ASR, Fourier convention and shear/breathing/moire
modes; measure sensitivity to the provider and to relevant parameters;
record backend, resources and geometry provenance. None of the first
requires a relaxed `(31, 30)` cell or a working MATBG-scale provider to be
partially exercised against real small-system data already archived in this
repository — so this ticket runs two suites, both reusing the exact
`PhononProvider` contract already certified at C18/S39, on nothing new.

### 2.1 `displacement_sensitivity_suite` — parameter sensitivity, with real numbers

**[CODE]** Three archived graphene-Gamma SIESTA FC runs already exist at
displacement magnitudes 0.005, 0.01 and 0.02 Ang
(`Comparison/results/epc/siesta_reference/graphene/runs/fc_dhs_d0p{005,01,02}`).
**[MEAS]** Each is put through the exact acoustic/ZO/E2g/ASR signature S39
already certifies (3 acoustic branches `< 1e-6 eV` after ASR, ZO in
800–900 cm⁻¹, E2g doublet in 1500–1620 cm⁻¹ split `< 1e-4` relative, raw ASR
residual measurably nonzero and corrected at roundoff) — all three pass
independently.

The cross-delta spread is then measured, not asserted against an invented
tolerance: ZO moves `0.03 cm⁻¹` and the E2g mean moves `0.91 cm⁻¹` across the
full 0.005→0.02 Ang range, and the E2g eigenvector subspace (`epc_subspaces
.subspace_overlap`/`subspace_metrics`, gauge-invariant under rotation within
the degenerate pair) rotates by a maximum principal angle of
`2.1e-8 rad` between the two extreme deltas — machine-precision stable. This
is reported under `spread_diagnostics` with `"applicable": true` but no
`passed` field of its own: no delta-sensitivity tolerance has been
independently derived for this repository the way, e.g., the dynamical-matrix
hermiticity tolerance in `certify_matbg_phonon_provider` is derived from the
mode set's own scale, so this ticket reports the number rather than invent a
threshold it cannot defend. The regression test only bounds it loosely
(`< 5 cm^-1`, far inside the >100 cm⁻¹ branch separation S39 already
certifies) so a real regression would still fail CI.

### 2.2 `local_stacking_phonon_suite` — real phonons on whichever registry has FC data

**[CODE]** Nam & Koshino (2017) identify AA, AB and BA as exactly the local
stacking registries a relaxed twisted bilayer interpolates between below
~2° twist — the same citation S47/S48 use to reject relaxation candidates
missing interlayer registry physics and to license AA/AB/BA as local
samples. **[MEAS]** Only `bilayer_graphene_AB` has an archived FC run
(`fc_dhs_d0p02`, 4 atoms). Its real phonon modes — not a synthetic direction,
not an electronic `D_H` derivative — are diagonalised and put through the
same acoustic/ASR contract check: 12 modes, 3 unambiguous acoustic branches
after ASR, raw residual `6.2e-2 -> 3.1e-15 eV/Ang²` corrected. Its geometry
signature, backend (`siesta_fc`), wall time and peak RSS delta are all
recorded per the "registrar backend, recursos y provenance" requirement.

AA and BA have **no** archived FC run
(`Comparison/results/epc/siesta_reference/bilayer_graphene_{AA,BA}/runs` do
not exist). They are reported `applicable: False` with the missing path
named in the detail string, and `_certify_local_stacking_phonons` treats
non-applicable checks as `passed: True` for the gate (they do not fail it)
while never counting them toward `materials_with_archived_fc`. This is the
concrete implementation of the acceptance criterion "ningún phonon rígido se
marca compatible salvo demostración explícita": AA/BA are not asserted
compatible by proximity to AB, they are named as unmeasured.

Two non-acoustic branches are additionally reported as *diagnostic*
breathing/shear candidates — the highest/lowest out-of-plane projection
weight among AB's optical branches, from `phonon_provider.mode_sector_report`
(already certified, generic, no new eigenvector logic here). This is
explicitly **not** the GO-7 shear/breathing adjudication
(`evaluate_ab_bilayer_epc`, which judges an electronic `D_H` derivative on
synthetic directions) — the field's own `shear_breathing_policy` string says
so, so a reader cannot conflate the two artifacts.

### 2.3 Moiré modes: out of reach, stated rather than skipped silently

Fase 11's Requirements also name "modos ... moiré". No MATBG-scale
`PhononProvider` exists (`go8a_status = "NO-GO-8a"`), so there is nothing to
Fourier-transform into low-energy moiré phonon branches (Lu et al., 2022) —
this is recorded in `references` as explicitly blocked on GO-8a rather than
omitted.

## 3. The GO-8a contract is reused verbatim, not re-derived

**[CODE]** `certify_relaxed_matbg_phonon_provider_bound(provider=...,
relaxed_target_fdf=...)` forwards both arguments straight into
`certify_matbg_phonon_provider.certify_matbg_phonon_provider_go8a` — the
exact same function, exact same checks
(`geometry_signature_match`/`dynamical_matrix_hermiticity`/
`acoustic_branches_vanish_at_gamma_after_asr`/`resource_preflight`/
`provenance`) S39 already certifies for the rigid target. This is how the
acceptance criterion "los artifacts relaxed `physical_phonon` pasan el mismo
contrato GO-8a" is met: by reuse, not by a second gate invented for the
relaxed case. **[MEAS]**
`test_go8a_contract_is_reused_verbatim_when_a_provider_is_supplied` stands
the same small archived graphene provider S39's own reopen test uses in for
"a future relaxed-geometry candidate" and checks the certification this
module produces is identical to calling S39's certifier directly.

Crucially, `relaxed_physical_phonon_produced` requires **both**
`relaxed_geometry_available` (from S47/GO-10) **and** `go8a_status ==
"GO-8a"` — `test_relaxed_physical_phonon_requires_both_go10_and_go8a` proves
a provider that clears GO-8a on its own (via the same graphene stand-in,
monkeypatched to a 2-atom target so the geometry check passes) still cannot
flip the relaxed verdict while S47 stays NO-GO-10: geometry existence and
provider existence are independent blockers, and this module refuses to let
one substitute for the other.

## 4. GO/NO-GO

**Relaxed-MATBG `physical_phonon`: NOT PRODUCED.** Blocked on both S47
(NO-GO-10: no relaxed geometry) and S39/S38 (NO-GO-8a: no MATBG-scale
provider). Reopens only when **both** upstream gates reopen — S47's own
`reopen_if` (an interlayer-registry-capable relaxation candidate) and S38's
own `reopen_if` (a MATBG-scale phonon candidate) — with the resulting
provider and the relaxed geometry's own `RUN.fdf` passed into this module's
`provider`/`relaxed_target_fdf` arguments, which the reopen test above
already exercises as live code, not a dead branch.

The bound itself — displacement sensitivity and local-stacking phonon
contract checks against real archived data — is `PASS` today, which narrows
risk (the `PhononProvider` contract's frequency/eigenvector/ASR machinery is
demonstrably stable across displacement magnitude and reproduces correctly
on a second, interlayer small system) but licenses nothing quantitative
about the relaxed target: `licenses_go8a_for_relaxed_geometry` and
`quantitative_relaxed_matbg_phonon_claim_licensed` are both hard-coded
`False`.

## 5. Reproduce

```bash
.venv/bin/python Comparison/scripts/certify_relaxed_matbg_phonon_provider.py
.venv/bin/pytest -q tests/test_certify_relaxed_matbg_phonon_provider.py
```
