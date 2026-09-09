# E-F_001 overnight remediation — 2026-09-05

## Outcome

E-F_001 remains **BLOCKED**. No orchestrator attempt was started and `state.db`
was opened only by the read-only reconciler. No commit or push was made.

## Implemented

- The production graphene route now uses the certified vacuum gauge
  `H(R)-c_vacuum(R)S(R)` for FC, explicit FD, Richardson and the finite-PAO
  covariant response. It refuses a campaign without a matching PASS gauge
  artifact. The Graph2Mat label conversion includes both `(E_F-c_vacuum)D_S`
  and `(E_F'-c_vacuum')S`.
- Added and executed the E2g gauge campaign. Both base and E2g gauge artifacts
  pass and contain file hashes, directional offsets and gauge-aligned
  Richardson rows.
- Replaced the former overclaiming perturbation terminology and DAG kinds with
  `pao_covariant_response` and `pao_projected_mode_coupling`. Historical JSON
  snapshots remain readable/displayable; they are not rewritten as new evidence.
- S33 is BLOCKED with no selected route while GO-5 is not PASS. S36 is
  UNDECIDED unless GO-5 and GO-7 are both PASS. S34 refuses before compute and
  S35 fails closed without recorded upstream PASS gates.
- GO-7 now requires an inversion derived from the real AB cell, positions,
  lattice-image translations, ORB_INDX orbital permutation/parity and declared
  Gamma k/q. The conjugate-block norm equality is no longer a check.
- Reconciliation no longer promotes zero-check leaves through a parent audit.
- The backend auditor consumes runtime manifests. A real Graph2Mat checkpoint
  preflight ran on CUDA and records device and actual dtypes.
- Checkpoint attribution is `UNDECIDED` while gauge/reference/basis closure is
  not attached; no tuning recommendation is emitted.

## Commands and results

```text
.venv/bin/python Comparison/scripts/run_epc_convergence_sweeps.py --only gauge_e2g
  16/16 SIESTA runs certified

.venv/bin/python Comparison/scripts/certify_epc_energy_zero.py
.venv/bin/python Comparison/scripts/certify_epc_energy_zero.py --reference-root Comparison/results/epc/convergence/gauge_e2g
  PASS / PASS

.venv/bin/python Comparison/scripts/certify_epc_numerical_convergence.py
  PASS

.venv/bin/python Comparison/scripts/certify_siesta_dhsdr.py
  GO-2 PASS, 6/6 comparisons

.venv/bin/python Comparison/scripts/certify_gamma_normalization_asr.py
  C18 PASS

.venv/bin/python Comparison/scripts/preflight_graph2mat_jvp.py
  passed on cuda:0; model=torch.float64; positions=torch.float64

.venv/bin/python Comparison/scripts/audit_epc_backends.py --fail-on-silent-cpu --require-runtime
  runtime_evidence_valid=true; silent_count=0

.venv/bin/python Comparison/scripts/cross_basis_projection_preflight.py
  PASS; four central-only SIESTA references (DZP/TZP), no displacement stencil

.venv/bin/python Comparison/scripts/run_displaced_projection_sentinel.py
  expected fail-closed NO_GO: all displaced C/M stability checks pass, but
  LS-vs-metric-restored derivatives differ by 7.2--35.4 meV/Ang (>1 meV/Ang)

.venv/bin/python Comparison/scripts/nested_basis_identity_preflight.py
  PASS: exact PROD-SZ radial identity, selected-overlap error <=2.8e-15

.venv/bin/python Comparison/scripts/run_nested_basis_sentinel.py
  PASS: constant selector remains exact for C1:x/C1:z and all five FD points

.venv/bin/python Comparison/scripts/run_nested_basis_ladder.py
  expected fail-closed NO_GO: cardinality changes 39.7--69.7 meV/Ang and
  final diffuse-range changes 77.7--135.6 meV/Ang (>5 meV/Ang)

.venv/bin/python Comparison/scripts/certify_production_basis_connection.py
  PASS without new DFT: propagated D_S residual <=0.544 meV/Ang, h-step
  residual <=0.149 meV/Ang, cross-vs-XML intra-atomic <=0.0029 meV/Ang

.venv/bin/python Comparison/scripts/certify_reference_basis_convergence_covariant.py
  expected fail-closed NO_GO: final diffuse-range now PASS (2.1--4.7
  meV/Ang), but N-DZP->N-TZP cardinality remains 15.9--18.2 meV/Ang for C1:x

.venv/bin/python Comparison/scripts/diagnose_nested_basis_central.py
  DIAGNOSTIC_ONLY; reads energy, density, bands, aligned VT/VH, K0 and S
  conditioning from existing central artifacts; no gate promotion

.venv/bin/python Comparison/scripts/evaluate_ab_bilayer_epc.py
  expected NO_GO: spatial AB PASS; upstream GO-4/GO-5 FAIL; phonon match FAIL

.venv/bin/python Comparison/scripts/ops/audit_epc_gate_status.py
  expected fail-closed: 56 NOT_RECONCILED_NO_EVIDENCE; 0 parent-audit promotions

.venv/bin/python -m pytest tests/ -q
  2713 passed, 1 skipped, 3 historical unrelated failures

Focused tests after the final edits:
  264 passed; final schema/compatibility pass: 116 passed
```

The three full-suite failures are the pre-existing Plotly `shape:` assertion
and two graphene/hBN legacy-dataset assertions. None imports or exercises the
EPC files changed here.

## Artifacts

- `Comparison/results/epc/convergence/gauge_delta/energy_zero_alignment.json`
- `Comparison/results/epc/convergence/gauge_e2g/energy_zero_alignment.json`
- `Comparison/results/epc/convergence/numerical_convergence.json`
- `Comparison/results/epc/convergence/gauge_e2g/certification/go2_certification.json`
- `Comparison/results/epc/certification/graphene/c18_gamma_phonon_certification.json`
- `Comparison/results/epc/gamma_epc/graphene_gauge_aligned_smoke/graphene_gamma_epc_summary.json`
- `Comparison/results/epc/gamma_epc/graphene/graphene_gamma_epc_summary.json`
- `Comparison/results/epc/gamma_epc/graphene/go4_verdict.json`
- `Comparison/results/epc/checkpoint_derivative_error/graphene/checkpoint_derivative_error.json`
- `Comparison/results/epc/error_interpretation/graphene/gamma_error_interpretation.json`
- `Comparison/results/epc/preflight/graph2mat_jvp_runtime.json`
- `Comparison/results/epc/gamma_epc/bilayer_graphene_AB/go7_verdict.json`
- `Comparison/results/epc_repair/backend_provenance_audit.json`
- `Comparison/results/epc_repair/status_reconciliation.json`
- `Comparison/results/epc/basis_projection_preflight/cross_basis_projection_preflight.json`
- `Comparison/results/epc/basis_projection_sentinel/displaced_projection_sentinel.json`
- `Comparison/results/epc/nested_basis_identity_preflight/nested_basis_identity_preflight.json`
- `Comparison/results/epc/nested_basis_sentinel/displaced_nested_basis_sentinel.json`
- `Comparison/results/epc/nested_basis_ladder/reference_basis_convergence_raw.json`
- `Comparison/results/epc/production_basis_connection/production_basis_connection.json`
- `Comparison/results/epc/reference_basis_convergence_covariant/reference_basis_convergence_covariant.json`
- `Comparison/results/epc/nested_basis_ladder/central_diagnostics.json`

## Scientific blockers deliberately left open

1. `Delta_out` and the intra-atomic PAO basis-response term remain unbounded;
   full-KS EPC is not claimed.
2. The central-geometry cross-basis projection preflight now passes for
   PROD-SZ, DZP(0.02/0.01 Ry), and TZP(0.01/0.005 Ry). The displacement
   stencils remain deliberately unlaunched: future derivatives must recompute
   `M(R_j,k)`, project `H-c_vacuum*S` at every geometry, and only then apply the
   finite-difference stencil. This can validate only the fixed production-PAO
   claim; `Delta_out`/full-KS remain phase 2.
   The subsequent displaced TZP(.005 Ry) sentinel correctly returns `NO_GO`:
   its C/M stability checks pass, but LS-versus-metric-restored projection
   sensitivity is 7.2--35.4 meV/Ang. Per the preregistered decision tree the
   remaining DZP/TZP stencils were not launched.
   A contract-preserving replacement was then tested without altering that
   verdict. Its exact-selector identity preflight and displaced sentinel pass,
   proving that PROD-SZ is literally embedded in the enriched bases. The raw
   ladder nevertheless returns `NO_GO`: both the cardinality increment and
   the final diffuse-range increment exceed the preregistered 5 meV/Ang and
   15 meV/Ang-outlier limits. No TZDP or `Delta_out` work was launched.
   The subsequently certified PROD-SZ one-sided connection passes, including
   its intra-atomic cross-geometry/XML check. Applying that connection makes
   the final diffuse-range increment pass, but the covariant cardinality
   increment remains NO_GO for C1:x (15.9--18.2 meV/Ang RMS). Therefore the
   raw NO_GO remains historical evidence and covariant reference convergence
   is separately, correctly NO_GO; no extended ladder was launched blindly.

## Independent PROD-SZ intra-atomic connection follow-up (2026-09-06)

The fixed-grid two-centre gate remains `NO_GO`.  Two additive follow-ups were
run without SIESTA or new DFT:

- `certify_b_i_analytic_two_center.py`: analytic PAO gradients remove the
  displacement stencil, but fixed-grid quadrature still fails for `C1_x`.
- `certify_b_i_prolate_two_center.py`: support-adapted prolate quadrature passes
  its 320--240 and h=0.00125--0.0025 checks; B3 still narrowly fails for `C1_x`
  (0.606 meV/Ang worst, 37/400 k points).
- `certify_b_i_prolate_confirmation.py`: the preregistered order-400 B3 passes
  for both directions (`C1_x` 0.319, `C1_z` 0.231 meV/Ang worst), but the dense
  400--320 convergence check for `C1_x` is 0.313 meV/Ang and therefore remains
  `NO_GO` against the unchanged 0.25 meV/Ang limit.  The declared stop policy
  applies: no further refinement or downstream `Delta_out` promotion.

Thus the independent construction strongly corroborates `.onlyS`, but the
full `B_I` validation remains unclosed because its own numerical uncertainty
does not meet the preregistered budget.

The subsequent s/p-only Fourier--Bessel/Slater--Koster v4 removes both the
moving support boundary and the centre FD.  Its dense internal convergence is
0.00496 meV/Ang (`C1_x`) and 0.00152 meV/Ang (`C1_z`), and the exploratory
comparison to `.onlyS` is below 0.20 meV/Ang.  Nevertheless V4-A remains
`NO_GO`: the monocentric Fourier reconstruction is 8.42e-10 versus the fixed
1e-10 limit.  A single preregistered q-tail/normalisation confirmation reduces
this to 2.16e-10 while keeping the dense electronic change below 0.0035
meV/Ang, but does not pass.  The final `.onlyS` gate therefore remains
`BLOCKED`; no further q refinement or downstream promotion was performed.

The corrected semantic v5 keeps all those historical `NO_GO` results and
separates `(120,raw)`, `(180,raw)` and `(180,normalized)`.  It passes:

- q-tail effect: 0.00344 meV/Ang (`C1_x`) and 0.00104 meV/Ang (`C1_z`);
- radial-normalisation effect: below 2e-6 meV/Ang in both directions;
- normalized radial norm error: 2.22e-16 and exact anti-Hermiticity;
- `.onlyS` confirmation: 0.200 meV/Ang (`C1_x`) and 0.121 meV/Ang (`C1_z`)
  worst case over all 400 k points, below the unchanged 0.5 meV/Ang gate.

`B_I_fourier_bessel_semantic_closure_v5 = PASS`, explicitly recorded as a
corrected-method confirmation rather than a blind holdout.  This closes the
independent PROD-SZ intra-atomic connection only; it does not retroactively
change v1--v4 or itself adjudicate `Delta_out`/full-KS.
4. The remaining C1:x cardinality residual was diagnosed before further DFT.
   Shared N-DZP/N-TZP radial functions (including polarization) are identical;
   the raw matrix residual is dominated by on-site s-p and, at K, interatomic
   p-p blocks. A strictly nested QZ prefix then passed at central and displaced
   geometries. The final QZ increments pass without changing thresholds:
   QZ-TZ is 1.78--2.04 meV/Ang for C1:x and 0.45--0.58 meV/Ang for C1:z;
   QZ D10/12-D12/14 is 2.15--2.61 meV/Ang at worst-case C1:x. Therefore
   `reference_basis_convergence=PASS` only for the SIESTA covariant reference
   expressed in fixed PROD-SZ. Historical NO_GOs remain recorded; this is not
   evidence for `Delta_out` closure or a full-KS claim.
5. A zero-DFT dense-k audit re-evaluated the final QZ cardinality/range
   increments on all 400 k points read from the TSHS. It passes: the worst
   case is QZ D10/12--D12/14 at k=(0.5,0.5,0), with 2.813 meV/Ang RMS and
   3.466 meV/Ang maximum.
6. The original Cartesian `full_basis_connection_v1=NO_GO` remains historical.
   A two-point `.onlyS` v2 also remains `NO_GO`; matching the five-point D_S
   stencil in v3 reduces the propagated closure below 0.0004 meV/Ang, but its
   independently accumulated left/right blocks miss the exact adjoint identity
   at about 1e-9 and therefore fail the unchanged 1e-10 auxiliary threshold.
   Semantic v4 projects onto the exact identity and keeps every electronic
   effect below 1 meV/Ang, but remains `NO_GO` because a relative solve residual
   normalized by a near-zero RHS reaches 1.24e-11.  V5 retains those verdicts
   and uses the scale-stable Cholesky backward error; all five TZ/QZ cases pass,
   with maximum backward error 1.03e-16.  Thus `full_basis_connection=PASS`
   under the corrected-method scope, without new SCF/DFT.
7. `delta_out_closure_v1` was then evaluated on all 400 k points using the same
   self-consistent B Hamiltonian on both sides of
   `Delta_A^[B] - E^dagger Delta_B E`.  The derivative-Hamiltonian terms cancel
   numerically to 1.5e-16 relative, so the result is not an FD artifact.  The
   gate is `NO_GO`: final cardinality increments are 336.2 meV/Ang (`C1_x`)
   and 166.8 meV/Ang (`C1_z`), while the QZ diffuse-range increment is
   684.5 meV/Ang.  The finite-basis `captured_exterior_response(QZ+D12/14)`
   reaches 5.69--8.90 eV/Ang for `C1_x` and 2.69--4.31 eV/Ang for `C1_z`,
   but is not a converged physical value of `Delta_out`; consequently
   `delta_out_negligible=NO_GO`.  No full-KS promotion or further DFT was made.
8. A zero-SIESTA `delta_out_subspace_diagnostic` explicitly built the Schur
   complement `span(B) minus span(A)`, its normalized modes, shell weights and
   spectral-cutoff sweep.  Its strict mode-sum equivalence gate is `NO_GO`:
   the complement sum differs from the current estimator by 1.9e-5--6.6e-5
   relative, exposing the residual inconsistency between the embedded PROD
   block of the full-basis connection and the separately certified PROD
   connection.  This is not hidden or regularized.  The remaining diagnostics
   are nevertheless stable under orthogonal, scaled and triangular complement
   gauges to at most 1.52e-11 relative.  Modes below relative lambda=1e-3 do
   not generate the multi-eV response: retaining only better-conditioned modes
   already leaves roughly 5.5--8.9 eV/Ang (`C1_x`) and 2.6--4.5 eV/Ang
   (`C1_z`).  The largest individual modes have relative lambda about
   0.56--0.76 and 85--94% `3d_polarization` weight.  Cross-overlap diagnostics
   also show that the QZ D10/12 and D12/14 complements are far from identical
   (sigma_min 0.18--0.56 at the three diagnostic k points).  Thus the evidence
   points to a well-conditioned polarization channel plus a materially changing
   complement, not an explosion driven primarily by near-null overlap modes.
9. `captured_exterior_response_embedded_v3` closes the semantic bridge without
   changing either earlier `NO_GO`: using the embedded full-basis connection on
   both sides and Hermitian representatives of the raw TSHS matrices reproduces
   the modal sum to 1.03e-11 relative.  The separately certified PROD connection
   differs by only 0.188 meV/Ang (`C1_x`) and 0.109 meV/Ang (`C1_z`), below the
   historical 1 meV/Ang connection budget.  Across all 400 k points the positive
   norm-weighted d-sector fraction remains systematic (roughly 0.53--0.80), and
   near-null modes remain irrelevant.  This certifies the diagnostic algebra,
   not `delta_out_closure`, which remains `NO_GO`.
10. The directed `QZDP_nested_identity_preflight` was attempted before any
    displaced calculation.  SIESTA's perturbative `P 2` generator rejected the
    second d zeta as degenerate for shell-specific split norms 0.15, 0.30, 0.50,
    0.80 and 0.99.  The preflight is therefore recorded fail-closed as `NO_GO`
    at atomic-basis generation: no candidate TSHS exists, no prefix-identity
    claim is made, and the authorized `C1_x` stencil/polarization probe was not
    launched.
11. A separate co-moving auxiliary-d preflight then tested whether the causal
    d-sector question could be isolated without modifying the QZP parent.  A
    regular ghost-C (`Z=-6`) requires its occupied 2s/2p shells, the special
    floating-Bessel species (`Z=-100`) requires this build's occupied 7s/5f
    shells, and a zero-occupation synthetic ghost (`Z=-206`) cannot reuse the
    ordinary C pseudopotential (`synthetic species not detailed`).  None yields
    the required five-d-only candidate, so `ghost_d_augmentation_preflight` is
    fail-closed `NO_GO` at species generation.  No TSHS, Schur-complement gate,
    displaced SCF or polarization-radial probe was produced.
12. The final authorized route first exercised the certified QZP `.ion` files
    unchanged through `User.Basis T`.  Both declared species packages were read,
    their byte hashes, radial functions and ordering are identical, total energy
    is identical, the Fermi-level difference is 2.33e-7 eV and the reconstructed
    density differs by 2.22e-9 relative.  Nevertheless the prospectively fixed
    round-trip limits are not met: the worst dense-k overlap discrepancy is
    2.63e-9 relative against 1e-12, and the worst metric Hamiltonian RMS is
    5.53e-5 eV against 1e-5 eV.  Therefore
    `user_basis_roundtrip_preflight=NO_GO`; no custom `.ion`, orthogonal d2,
    displaced stencil or full-KS promotion was attempted.
3. GO-5 is still NO_GO. GO-7 remains NO_GO despite passing its now-mandatory
   spatial test, because its upstream gates and phonon robustness do not pass.
4. The gauge-aligned graphene smoke run honestly returns `CHECKS_FAIL` on two
   frozen scientific rules (uniform-translation full-block null and individual
   K-doublet member equality). It produced evidence but did not promote a gate.

Legacy JSON snapshots may still contain former field names. Active producers
emit the new names; the UI keeps a read-only fallback for old `physical_epc_level`
artifacts so historical evidence remains inspectable without being relabelled as
a new run.
