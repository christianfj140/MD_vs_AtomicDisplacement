# E-F_001-S44 — pre-register the rigid-MATBG q != 0 (non-Gamma) EPC branch

Fase 10d of the roadmap, D08B ("Extend selected PAO-covariant coupling to MATBG q != 0. Requiere GO-5,
GO-8a, GO-8 y GO-8b"). The ticket text is explicitly conditional: "Ejecutar, solo si se desea una
rama no-Gamma, modos fisicos MATBG mediante el provider GO-8a y la respuesta q-aware GO-8b."

Producer: `Comparison/scripts/preregister_matbg_qneq0_rigid_epc.py`.
Tests: `tests/test_preregister_matbg_qneq0_rigid_epc.py`.
Artifact: `Comparison/results/epc/preregistration/matbg/epc_matbg_rigid_qneq0_protocol.json`.

## 1. Why this ticket does not execute anything physical

The gate registry today:

- **GO-8a**: `NO-GO-8a` (S39, `certify_matbg_phonon_provider.py`) — no scalable, validated MATBG
  `PhononProvider`.
- **GO-8b**: `NO-GO-8b` (S40, `certify_qneq0_matbg_response.py`) — the Q-C/Q-B mechanics pass on
  the real production JVP, but `graphene_k_agreement_suite` stays `NOT_RUN` because GO-5 is
  `NO_GO` (blocked by GO-4/C14C's intra-atomic basis-response term).
- **GO-9 (Gamma branch)**: `NO-GO-9` (S43, `certify_rigid_gamma_campaign.py`) — even the simpler
  Gamma-only campaign has never had `resource_budget_suite` measured against the real
  44 656-orbital checkpoint.

Both of S44's own required inputs — the provider and the q-aware response — are `NO-GO`.
Producing a q != 0 mode today would require either reusing the plain q=0 Gamma JVP as a silent
substitute (exactly the failure `GammaJvpReuseError` exists to reject, S40 Sec. 7) or fabricating
a phonon no certified `PhononProvider` produced (roadmap B7). Neither is acceptable, so this
ticket does what S41 did for the Gamma branch before GO-8a existed: freeze the protocol, not run
it.

## 2. What is new relative to the Gamma protocol (S41)

`preregister_matbg_qneq0_rigid_epc.py` imports and reuses
`preregister_matbg_gamma_rigid_epc.py` directly for everything that does not change with q:
geometry/candidate-direction/occupation reporting, the electronic k candidates and window rule,
the delta sweep, resource margins, and the three synthetic-displacement direction families. Only
what is genuinely different about a non-Gamma branch is new:

- **Two more required gates.** GO-8a and GO-8b are both `required=True` preconditions here
  (S41's protocol carried GO-8b as informational only, since it is Gamma-only). GO-9's own
  Gamma-branch campaign certification (S43) stays informational — the roadmap recommends running
  D08A before D08B but does not make it a hard dependency (Fase 10d closing paragraph).
- **q candidates.** `Q_PRIMARY` reuses the Gamma protocol's `K_DEGENERATE` point relabelled as a
  phonon-momentum candidate — literature-motivated (Piscanec et al. 2004 Kohn-anomaly analogue;
  Lu et al. 2022 low-energy moire-phonon evolution with twist), not a physical claim. A second,
  generic point is kept only as a route-mechanics cross-check, never a second result.
- **The reference route is Q-C/Q-B, never the plain Gamma JVP.** `REFERENCE_PATHS` points at
  `epc_qb_qc_graph2mat_kernel.kernel_table_graph2mat` (primary) and
  `phase_aware_kernel_graph2mat`/`q_b_response_graph2mat` (cross-check) — the real-JVP q-aware
  kernel S40 built and certified the mechanics of, not `graph2mat_jvp`/`graph2mat_frozen`.
- **A structural gap beyond C14C: no q-aware basis-response producer exists.**
  `epc_basis_response.py` (C14B) builds `PAO-covariant response` only from a real-space SIESTA
  `FC.Save.dHS` campaign object, which S38 already measured intractable at MATBG scale even at
  `q=0`. GO-8b's own acceptance criterion requires the basis response to be "tambien q-aware"
  under the same Fourier/R-vector/atomic-phase conventions as `D_H(q)`. That producer does not
  exist for any system in this repository today — independent of whether C14C's intra-atomic term
  ever resolves. `QAWARE_BASIS_RESPONSE_GAP` records this explicitly and blocks the
  `qaware_basis_response` physics-review item.
- **Checks mirror S40's `route_mechanics_suite`** (`qc_and_qb_agree_on_real_graph2mat`,
  `q_and_minus_q`, `cell_origin_shift`, `atom_across_periodic_boundary`,
  `alternative_atomic_phase_convention`, `topology_margin_preserved`,
  `gamma_jvp_reuse_rejected`) instead of the Gamma branch's JVP-vs-frozen check, plus
  `route_agrees_with_q_a_on_real_graphene_k` (blocked by GO-5) and
  `qaware_basis_response_present_and_required`.

## 3. Result

```
$ .venv/bin/python Comparison/scripts/preregister_matbg_qneq0_rigid_epc.py
pre-registration matbg_rigid_qneq0_epc_v1 -> .../epc_matbg_rigid_qneq0_protocol.json
  candidate q           : K
  physics review        : BLOCKED
  ready to execute      : False
    blocked by: GO-8a MATBG PhononProvider: NO-GO-8a
    blocked by: GO-8b MATBG q!=0 response: NO-GO-8b
    blocked by: C14C basis response (global formalism gate): NO_GO
    blocked by: q_selection: no live k -> k+q electronic contraction measured yet ...
    blocked by: qaware_basis_response_producer_missing: ...
    blocked by: moire_hexagonal_convention_checked: ...
```

`ready_to_execute = False`. `require_frozen_protocol()` raises rather than let a future execution
step run against an unmet precondition or a tampered file — same contract as S41's
`require_frozen_protocol`, verified by
`test_require_frozen_protocol_refuses_to_run_while_blocked` and
`test_require_frozen_protocol_detects_tampering`.

## 4. What a future execution step must do differently from S42

S42 (`run_tbg_pure_graph2mat_campaign.epc_gamma_rigid()`) could run its raw-derivative rung today
because the plain JVP needs nothing beyond the checkpoint and a direction. A q != 0 analogue
cannot follow that shortcut: even the mechanics-only rung this protocol authorises once GO-8a and
GO-8b close still needs the Q-C kernel-table build (`epc_qb_qc_graph2mat_kernel`) against the real
`(31, 30)` topology, which S37/S40 never measured (OOM before any real MATBG edge count was
recorded). Any execution step must therefore run the same representative-displacement resource
preflight GO-8 requires before attempting a real kernel-table build on this system, exactly as
S42 Sec. 6 already required for the simpler Gamma JVP.

## 5. Reproduce

```bash
.venv/bin/python Comparison/scripts/preregister_matbg_qneq0_rigid_epc.py
.venv/bin/python Comparison/scripts/preregister_matbg_qneq0_rigid_epc.py --verify
.venv/bin/pytest -q tests/test_preregister_matbg_qneq0_rigid_epc.py
```
