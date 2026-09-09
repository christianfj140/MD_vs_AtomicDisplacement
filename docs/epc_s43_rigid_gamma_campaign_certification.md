# E-F_001-S43 — GO-9 certification of the rigid-Gamma raw-derivative campaign

S42 (`run_tbg_pure_graph2mat_campaign.epc_gamma_rigid()`,
`docs/epc_s42_matbg_gamma_rigid_execution.md`) implemented the S41-authorised rung of the
rigid-MATBG-Γ campaign but its own Sec. 6 flagged an explicit open item: it had never been
exercised end to end, not even with the GPU/model boundary mocked. This ticket is that
validation — reproducibility, order-independence, deliberate-interruption survival, a guard
against any synthetic artifact satisfying a physical dependency, and a resource-budget check —
implemented as a new certifier, `Comparison/scripts/certify_rigid_gamma_campaign.py`, mirroring
the `certify_*_go8*.py` pattern already established by S39/S40.

Producer: `certify_rigid_gamma_campaign.certify_rigid_gamma_campaign_go9()`.
Tests: `tests/test_certify_rigid_gamma_campaign.py`.
Artifact: `Comparison/results/epc/certification/matbg/s43_rigid_gamma_campaign_go9_certification.json`.

## 1. A real bug this audit found and fixed first

`tests/test_run_tbg_gamma_rigid_epc.py` monkeypatched `camp.EPC_ARRAYS_DIR` to a `tmp_path` but
never touched `camp.EPC_STATUS_PATH` — a separate module-level global pointing at the *real*
`Comparison/results/epc/gamma_epc/matbg/artifact_status.json`. Every run of that test file's
`test_raw_derivative_stage_records_failure_without_crashing` therefore overwrote the real
production campaign status with a fabricated `{"raw_derivative__translation_x": {"error":
"RuntimeError: synthetic failure", "state": "failed", ...}}` entry — a `synthetic_test_displacement`
contract's own machinery, corrupting the exact reproducibility signal S43 was asked to audit. Fixed
by monkeypatching `EPC_STATUS_PATH` alongside `EPC_ARRAYS_DIR` in both tests that call
`epc_raw_derivative_for_direction` directly; the stray file (untracked, gitignored, so purely local
contamination) was deleted. `certify_rigid_gamma_campaign.py` never repeats this mistake: every
mocked run patches `EPC_ROOT`, `EPC_ARRAYS_DIR` *and* `EPC_STATUS_PATH` to an isolated `tmp_path`,
and `test_certifier_never_touches_the_real_production_campaign_state` asserts it.

## 2. What each suite actually exercises

Every suite below runs the real `epc_gamma_rigid()` (or, for reordering/interruption, its own
`epc_raw_derivative_for_direction` helper) against the real, already-frozen S41 protocol
(`Comparison/results/epc/preregistration/matbg/epc_matbg_rigid_gamma_protocol.json`) and the real
`(31, 30)` geometry (`preregister_matbg_gamma_rigid_epc.DEFAULT_POSITIONS_XYZ`, 11 164 atoms) — only
the GPU/model boundary (`load_model_and_batch`, `resolve_jvp_backend`, `batch_topology_hash`,
`directional_derivative_field`, `frozen_derivative_field`) is faked, with a deterministic
per-direction field keyed off each direction's own `direction_hash` (same value for JVP and every
`delta`, so `jvp_equals_frozen_within_backend_margin` passes for a real, non-trivial reason rather
than a rigged constant).

- **`gate_authorization_suite`**: `epc_authorize` against the real on-disk protocol grants exactly
  the S41-frozen rung; a tampered protocol is rejected; the five directions rebuilt from the real
  geometry hash-match the frozen protocol's `phonon.candidate_sectors` (the invariant
  `epc_gamma_rigid()` itself asserts before doing anything else).
- **`reproducibility_suite`**: cold cache (fresh `artifact_status.json`) vs warm cache (same file,
  second call) vs a simulated restart (third call against the same on-disk state) all produce
  bitwise-identical `epc_gamma_rigid_summary.json` content (timestamps and the per-call `reused`
  flag excluded — those are the *expected* difference between a cache hit and a cold run, not
  drift) with zero extra JVP/frozen calls on the warm/restart paths. A changed checkpoint correctly
  invalidates the cache and forces recompute, proving the cache hits above were signature-checked.
- **`reorder_and_interruption_suite`**: the five directions processed forward vs reversed into two
  independent status dicts produce identical per-direction artifacts (each keyed independently, no
  order-dependent shared state). A simulated crash mid-campaign (`directional_derivative_field`
  raises for the second direction after the first already completed) is recorded `state: "failed"`
  without touching the first direction's `completed` entry; reloading the persisted status from disk
  (simulated restart) reuses the completed direction without recompute and recomputes only the
  failed one, converging to exactly the same result as an uninterrupted run.
- **`synthetic_dependency_guard_suite`**: every one of `basis_response`/`eigenspace`/`pao_covariant_response`/
  `g`/`phonon` stays `status: "not_attempted"`; every artifact (top-level and per-direction) is
  tagged `synthetic_test_displacement`; `forbidden_publication_labels` names
  `quantitative_relaxed_matbg_prediction`, `g_mn_nu` and `phonon_eigenmode`, none of which is ever
  used as the live `claim`/`result_status`; `result_status` stays `candidate_rigid_tbg` (never the
  eventual publication label `rigid_tbg_model`); `claim` never exceeds the S41-authorised rung.
- **`resource_budget_suite`**: honestly `NOT_RUN`. No real GPU run of `epc_gamma_rigid()` against
  the production 44 656-orbital checkpoint has ever been measured (S42 Sec. 6). Reporting a
  wall/RAM/VRAM/disk verdict from a live machine snapshot with zero actual campaign usage would be
  fabricating agreement with the roadmap's own Section X margins (28 GiB VRAM / 62 GB RAM / 12%
  free disk), reused here via `benchmark_matbg_synthetic_displacements.resource_margins()` purely
  as informative context, not a pass/fail claim — the same honesty pattern S40's
  `graphene_k_agreement_suite` already established for a different blocked gate.

## 3. Verdict

`go9_status = "NO-GO-9"`: four of five suites `PASS`, `resource_budget_suite` stays `NOT_RUN` until
a real GPU run against the production checkpoint is measured (S42 Sec. 6's own open item — a
one-direction-at-a-time preflight, per this repository's compute policy, not a five-direction batch
launched blind). `final_publication_label` is pinned to `rigid_tbg_model`;
`quantitative_relaxed_matbg_prediction` (plus `g_mn_nu`, `phonon_eigenmode`) stay listed as
forbidden regardless of gate outcome. `require_go9_certified_rigid_gamma_campaign()` raises on
anything short of `GO-9`, mirroring `require_go8b_certified_qneq0_response`.

## 4. Reproduce

```bash
.venv/bin/pytest -q tests/test_certify_rigid_gamma_campaign.py tests/test_run_tbg_gamma_rigid_epc.py
.venv/bin/python Comparison/scripts/certify_rigid_gamma_campaign.py
```
