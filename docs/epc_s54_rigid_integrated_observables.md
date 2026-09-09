# E-F_001-S54 — Produce and interpret rigid-model integrated observables (Fase 12)

Objective: "Calcular `|g|^2, gamma(q,nu), lambda(q,nu), lambda y alpha2F` para el modelo rigido
unicamente con los parametros convergidos."

Roadmap Fase 12 is `DEFERRED`: "Solo despues de GO-10 o, para un resultado explicitamente
rigid-model, despues de GO-9." S53 (`preregister_integrated_observables_convergence.py`) froze the
convergence protocol (Gamma-centered `monkhorst_pack_2d` k/q mesh, `3 * n_atoms_per_cell` modal
completeness, smearing reused verbatim from `epc_occupations.py`, a distinct broadening/DOS
normalization contract, spin/valley weights, the plateau/uncertainty evidence rule) and validated
every piece of that machinery against synthetic test functions with a known limit — never against a
fabricated `g`. It left `ready_to_execute=False` for exactly one reason:
`shared/artifact_signature.py`'s `integrated_observable` node depends only on `g_set`
(`pao_projected_mode_coupling`, physical), and `g_set` is empty, because both rigid branches read live as
`NO-GO-9`:

- `certify_rigid_gamma_campaign.py` (S43) → `go9_status == "NO-GO-9"`
- `certify_qneq0_rigid_campaign.py` (S45) → `go9_status == "NO-GO-9"`

Re-verified live for this ticket (2026-09-03): both certifications still read `NO-GO-9` — S43 on an
unmeasured resource budget (`resource_budget_suite: NOT_RUN`, no real GPU run of `epc_gamma_rigid()`
against the production `(31,30)` checkpoint has ever been measured), S45 on GO-5/GO-8b plus the same
unmeasured resource budget. There is therefore no PAO-projected coupling anywhere in this repository, on
either branch, today.

Producer: `Comparison/scripts/produce_rigid_integrated_observables.py`
(`produce()`, `interpret()`, `run()`). Tests:
`tests/test_produce_rigid_integrated_observables.py`. Artifact:
`Comparison/results/epc/integrated_observables/matbg/rigid_integrated_observables_result.json`.
Depends on `preregister_integrated_observables_convergence.py` (S53, the frozen protocol and gate
this module reads) and the on-disk GO-9 certification reports from `certify_rigid_gamma_campaign.py`
(S43) and `certify_qneq0_rigid_campaign.py` (S45).

## 1. What this ticket does

Reads S53's protocol live (rebuilt fresh each run — the protocol's own math is negligible cost, so
there is no staleness risk in not caching it), and:

- if `ready_to_execute` is ever `True`, computes `|g|^2`, `gamma_qnu`, `lambda_qnu`, `lambda` and
  `alpha2F` from a GO-9-certified `g_set`, using the converged mesh/broadening density from S53's own
  ladder rather than a single hand-picked value — that branch is unreachable today and is left as
  live code (not a hardcoded skip), so a future GO-9 reopen exercises it for real instead of adding a
  second script from scratch;
- today, refuses and persists that refusal as a granular artifact: `g_set` carries the actual
  `PreregistrationError` reason, and every downstream observable (`g_squared`, `gamma_qnu`,
  `lambda_qnu`, `lambda_total`, `alpha2f`) is `not_attempted` with `"blocked by g_set"`, the same
  cascading-dependency pattern S51 used for `raw_responses -> pao_covariant_response -> g`. Every backend field
  (`electronic_derivative_backend`, `basis_response_backend`, `phonon_backend`, `epc_backend_class`)
  is recorded as `not_attempted` rather than omitted.

`interpret()` then states plainly which claims that refusal licenses (the protocol was correctly
applied; no execution is authorised — both `licensed`) and which stay forbidden regardless of gate
outcome: S43/S45's own labels (`quantitative_relaxed_matbg_prediction`, `g_mn_nu`,
`phonon_eigenmode`), `rigid_tbg_model` (the eventual final label, only licensed once a branch's own
GO-9 certification says so — not by this ticket), and `Tc` (explicitly out of scope per the roadmap
and S53's own protocol scope).

## 2. What is not done

No sweep runs against a real coupling constant, DOS or `alpha2F`: `g_set` is empty on both branches.
No k/q-mesh or broadening-width convergence ladder is exercised against physics (S53 already
validated that machinery against synthetic functions; running it again here on the same synthetic
data would add nothing). No GPU workload exists for this ticket: the observable math this ticket
would run, once unblocked, is the cheap sum/broadening arithmetic S53 already benchmarked as
negligible — the GPU cost lives entirely upstream, in the rigid campaigns' own `g_set` production
(S42/S43/S44/S45), which this ticket does not re-run.

## 3. Verdict

`NO_QUANTITATIVE_CLAIM_LICENSED`. Both `licensed` claims hold against the real, live S43/S45/S53
artifacts (`interpretation.all_licensed_claims_hold = True`); every forbidden label — including `Tc`
— stays out of the produced artifact. This is not a GO of any kind: it is the same honest procedural
statement S52 made for the relaxed branch, applied here to the rigid one. Re-run whenever S43 or S45
change; the moment either flips to `GO-9`, this module's `produce()` becomes the one place to replace
the `RigidIntegratedObservablesError` guard with a real `g_set -> |g|^2/gamma/lambda/alpha2F`
computation against S53's converged mesh/broadening ladder.

## 4. Reproduce

```bash
.venv/bin/pytest -q tests/test_produce_rigid_integrated_observables.py
.venv/bin/python Comparison/scripts/produce_rigid_integrated_observables.py
```
