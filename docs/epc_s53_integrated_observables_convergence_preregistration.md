# E-F_001-S53 — Convergence protocol for integrated rigid-model observables (Fase 12)

Roadmap Fase 12 (`|g|^2, gamma_qnu, lambda_qnu, lambda, alpha2F(omega)`) is `DEFERRED`: "Solo
despues de GO-10 o, para un resultado explicitamente rigid-model, despues de GO-9."
`shared/artifact_signature.py`'s own DAG makes the dependency mechanical: the
`integrated_observable` node's only dependency is `g_set` (one or more `pao_projected_mode_coupling`
artifacts). Read live today:

- `certify_rigid_gamma_campaign.py` (S43) → `go9_status == "NO-GO-9"`
- `certify_qneq0_rigid_campaign.py` (S45) → `go9_status == "NO-GO-9"`

so `g_set` is empty on both branches — there is no PAO-projected coupling anywhere in this repository.
Sweeping a k/q mesh, a broadening width or a DOS normalization against a coupling constant that
does not exist would fabricate a plateau. This ticket does what S41/S44/S50 already established
for a gated ticket: freeze the numerical convergence protocol a future execution must use, and gate
any real execution behind a live read of GO-9 rather than skip the ticket or fake a result.

Producer: `Comparison/scripts/preregister_integrated_observables_convergence.py`.
Tests: `tests/test_preregister_integrated_observables_convergence.py`.
Artifact: `Comparison/results/epc/preregistration/matbg/epc_integrated_observables_convergence_protocol.json`.

## 1. What is frozen

- **k/q mesh**: `monkhorst_pack_2d(nx, ny)`, a Gamma-centered fractional 2D generator shared by
  both k and q (both live in the same moire Brillouin zone).
- **Modal completeness**: `3 * n_atoms_per_cell` phonon branches per q (33 492 for the `(31, 30)`
  target's 11 164 atoms). A selected-mode subset (roadmap Fase 10d) may validate physics but is
  rejected as a substitute for the modal sum an integrated observable needs — this ticket's own
  Requirements state it explicitly ("selected modes aislados no bastan").
- **Smearing**: reused verbatim from `epc_occupations.py` (roadmap XII contract) — not redeclared.
- **Broadening**: a second, distinct energy-conserving delta for the observable itself (Gaussian or
  Lorentzian kernel, `broadening_kernel`), kept separate from electronic smearing because
  `artifact_signature.py`'s `integrated_observable` node already carries `smearing` and
  `broadening` as two independent fields.
- **Spin/valley weights**: `spin_degeneracy * VALLEY_DEGENERACY` (`VALLEY_DEGENERACY = 2`,
  graphene/TBG's two inequivalent K, K' valleys, Piscanec et al. 2004).
- **2D DOS normalization**: `g(E) = (spin_degeneracy * valley_degeneracy / cell_area_ang2) *
  sum_k weight_k * sum_n kernel(E - eps_kn, width)`, units states/eV/Ang^2; its sum rule
  (`integral g(E) dE == spin_valley_mult`, independent of width/kind) is exercised in
  `_selfcheck()` against synthetic eigenvalues, never against a fabricated `g`.
- **Plateau/uncertainty evidence rule**: `mesh_convergence_report(name, densities, values, tol)` —
  plateau requires the last relative difference on a convergence ladder to fall below `tol`;
  otherwise that residual is reported as the declared uncertainty rather than dropped. This is the
  acceptance-criteria mechanism itself ("cada parametro tiene evidencia de plateau o incertidumbre
  declarada"), validated here on synthetic test functions with a known limit.

`Tc` is explicitly listed as out of scope in the protocol's own `scope.explicitly_out_of_scope`.

## 2. What is not done

No sweep runs against a real coupling constant, DOS, or `alpha2F`: `g_set` is empty. The
`physics_review` item `g_set` reads `blocked`, citing both GO-9 branches live; `ready_to_execute`
is `False`. No GPU workload exists for this ticket either — the frozen backend record documents
that explicitly rather than fabricating a preflight for math this cheap.

## 3. Verdict

`ready_to_execute = False`. All non-`g_set` protocol items (mesh, modal completeness, smearing,
broadening/DOS, convergence-evidence rule) are `approved` — the machinery is written, tested and
frozen. The single blocking item is physical: no branch has cleared GO-9. Re-run whenever S43 or
S45 change; the moment either does, this protocol's mesh/DOS/broadening machinery applies to that
branch's `g_set` unchanged.

## 4. Reproduce

```bash
.venv/bin/pytest -q tests/test_preregister_integrated_observables_convergence.py
.venv/bin/python Comparison/scripts/preregister_integrated_observables_convergence.py
```
