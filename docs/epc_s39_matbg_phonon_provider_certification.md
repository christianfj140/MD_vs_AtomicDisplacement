# E-F_001-S39 — GO-8a certification of the scalable MATBG `PhononProvider`

Fase 10a of the roadmap. S38 (`docs/epc_s38_matbg_phonon_provider_decision.md`)
is the *selection* step and already concluded `NO-GO-8a`: no candidate is
defensible against the roadmap's own coverage axes. This ticket is the
*certification* step — it turns that decision into a certification artifact
that carries the `physical_phonon` / `synthetic_test_displacement` schema
split from `shared/artifact_signature.py`, runs the exact benchmark suite
S38 Section 8 pre-registered, and provides the consumer-side gate any future
MATBG selected-mode campaign must call before treating a mode as physical.

Producer: [`Comparison/scripts/certify_matbg_phonon_provider.py`](../Comparison/scripts/certify_matbg_phonon_provider.py)
(`certify_matbg_phonon_provider_go8a()`, `require_go8a_certified_matbg_phonons()`).
Tests: [`tests/test_certify_matbg_phonon_provider.py`](../tests/test_certify_matbg_phonon_provider.py).
Depends on: `decide_matbg_phonon_provider.py` (S38), `phonon_provider.py` (the
`PhononProvider` contract and `SiestaFcPhononProvider` backend),
`evaluate_ab_bilayer_epc.py` (the GO-7 adjudication), the archived graphene
Gamma FC run and the persisted AB-bilayer GO-7 report.

Evidence conventions: **[CODE]** verified in this repository, **[MEAS]**
measured here, **[DESIGN]** decision frozen in this document.

---

## 1. Verdict

**NO-GO-8a**, inherited from S38 and re-certified here against evidence
rather than re-asserted. `certify_matbg_phonon_provider_go8a()["go8a_status"]`
is `"NO-GO-8a"`, `["physical_phonon_produced"]` is `False`, and
`["physical_phonon"]` is `None` — no MATBG mode is produced, physical or
otherwise, by this ticket.

**[MEAS]** Reproduced by running the certifier with no candidate provider
(what this repository has today):

```
$ .venv/bin/python Comparison/scripts/certify_matbg_phonon_provider.py
S39 MATBG PhononProvider GO-8a certification: NO-GO-8a
  small_system_suite: PASS
    [ok  ] graphene_gamma_acoustic_and_optical: ...
    [ok  ] acoustic_sum_rule_raw_vs_corrected: ...
    [ok  ] ab_bilayer_interlayer_sectors: ...
  matbg_scale_suite: NOT_RUN
    [n/a ] geometry_signature_match: blocked by NO-GO-8a: ...
    [n/a ] dynamical_matrix_hermiticity: blocked by NO-GO-8a: ...
    [n/a ] acoustic_branches_vanish_at_gamma_after_asr: blocked by NO-GO-8a: ...
    [n/a ] resource_preflight: blocked by NO-GO-8a: ...
    [n/a ] provenance: blocked by NO-GO-8a: ...
```

## 2. Two independent suites, one gate

The certifier splits the S38 Section 8 benchmark suite exactly along its own
`small_system` / `matbg_scale` boundary, because the two run under
completely different conditions:

**[CODE] `small_system`** runs unconditionally, against real archived SIESTA
FC data already on disk — no candidate provider is needed because the
`PhononProvider` contract itself (`SiestaFcPhononProvider`) is already
validated at graphene/AB-bilayer scale (C18/GO-7). Three checks:

- `graphene_gamma_acoustic_and_optical` — the exact thresholds
  `tests/test_phonon_provider.py::test_gamma_frequencies_reproduce_graphene`
  already certifies (3 acoustic branches `< 1e-6 eV`, ZO in 800–900 cm⁻¹, E2g
  doublet in 1500–1620 cm⁻¹ split `< 1e-4` relative), re-measured here rather
  than imported as a cached verdict;
- `acoustic_sum_rule_raw_vs_corrected` — raw ASR residual measurably nonzero,
  corrected residual at roundoff, neither number substituted for the other;
- `ab_bilayer_interlayer_sectors` — the candidate's own eigenvectors (not a
  synthetic direction) reproduce the shear/breathing geometric signature
  `evaluate_ab_bilayer_epc.check_shear_direction` /
  `check_breathing_parity` already check, read through the persisted GO-7
  report rather than recomputed.

**[MEAS]** All three pass against the artifacts present in this repository
today (`Comparison/results/epc/siesta_reference/graphene/runs/fc_dhs_d0p01`,
`Comparison/results/epc/...ab_bilayer_epc_three_sectors.json`); the
`small_system_suite` status is `PASS`.

**[CODE] `matbg_scale`** can only run against a real `PhononProvider`
instance for `materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf`. Since
S38's live decision is `NO-GO-8a` (`selected_provider` is `None`), this suite
is reported `NOT_RUN` — every one of its five checks
(`geometry_signature_match`, `dynamical_matrix_hermiticity`,
`acoustic_branches_vanish_at_gamma_after_asr`, `resource_preflight`,
`provenance`) is `applicable: False`, carrying S38's own `blocking_reason`
and `reopen_if` text verbatim rather than a generic "skipped" label. This is
the mechanism that keeps S38's limitation visible to every downstream
consumer of this certification, not just to a reader of the S38 memo.

## 3. Schema separation is structural, not a label

**[CODE]** `certify_matbg_phonon_provider_go8a()["artifact_kind"]` is `None`
whenever `go8a_status != "GO-8a"`; the only way to obtain a
`physical_phonon`-tagged payload is for `matbg_scale_suite` to actually run
and pass against a real provider (`_certify_matbg_scale_suite`, which calls
`phonon_provider.require_physical_phonon` on the provider's own output —
a `synthetic_test_displacement` cannot pass through it silently).
`require_go8a_certified_matbg_phonons()` re-checks both the gate name and
the artifact kind before returning anything, so a consumer script cannot
mistake a `NO-GO-8a` report — or a malformed one missing the expected keys —
for a usable phonon; both raise `MatbgPhononCertificationError`
(`tests/test_certify_matbg_phonon_provider.py::test_require_go8a_certified_matbg_phonons_refuses_a_no_go_certification`,
`::test_require_go8a_certified_matbg_phonons_refuses_a_malformed_object`).

## 4. The `matbg_scale` suite is live code, not a dead branch

**[MEAS]** `certify_matbg_phonon_provider_go8a(provider=...)` accepts a real
`PhononProvider` today — the reopen path S38 Section 9 describes does not
require rewriting this module. Standing a small archived
`SiestaFcPhononProvider` (graphene, 2 atoms) in for "a future MATBG-scale
candidate" and pointing `target_fdf` at its own `RUN.fdf` exercises every
`matbg_scale` check for real:

- `geometry_signature_match`, `acoustic_branches_vanish_at_gamma_after_asr`,
  `resource_preflight` and `provenance` pass on this stand-in;
- `dynamical_matrix_hermiticity`'s tolerance (`64 * eps * scale`, `scale`
  derived from the mode set's own frequency range) is a toy-scale estimate
  and is not guaranteed to clear on real SIESTA FC data — on this stand-in
  it measures `~9e-12 eV/(Ang^2 amu)` against a `~1e-13` tolerance. This is
  read from the report, not assumed, by both the certifier (which fails the
  suite honestly rather than rounding it up) and its test
  (`test_matbg_scale_suite_runs_for_real_against_a_provider_on_matching_geometry`).
  A real GO-8a candidate's own noise floor — not this stand-in's — is what
  S38 Section 8 requires the tolerance to be checked against; this is
  recorded here as an open calibration point for whichever future ticket
  supplies a real candidate, not fixed speculatively now.
- with `TARGET_ATOM_COUNT` left at the real `(31, 30)` value (11 164), the
  same 2-atom stand-in correctly fails `geometry_signature_match` rather
  than certifying a mismatched geometry
  (`test_matbg_scale_suite_fails_on_geometry_mismatch`).

## 5. GO/NO-GO

**GO-8a: NOT MET.** No candidate provider exists to certify at MATBG scale;
`matbg_scale_suite` stays `NOT_RUN` until S38 reopens per its own Section 9
criteria. Per roadmap Fase 10a, no physical selected-mode EPC production
campaign on MATBG is authorized. Only `synthetic_test_displacement`
benchmarking (D06/S37) is permitted on the `(31, 30)` target in the
meantime.

## 6. Reproduce

```bash
.venv/bin/python Comparison/scripts/certify_matbg_phonon_provider.py
.venv/bin/pytest -q tests/test_certify_matbg_phonon_provider.py
```
