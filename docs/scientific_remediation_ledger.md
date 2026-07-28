# Scientific remediation ledger

Final engineering closure of the five-phase audit, 2026-07-28.

Status vocabulary:

- `PASS_ENGINEERING`: the shared implementation, fail-closed gate and regression
  tests are complete.
- `PRESERVED_PROVEN`: the original finding was refuted for an explicitly
  bounded scope and that evidence remains protected.
- `BLOCKED_FAIL_CLOSED`: the repository now refuses the claim because the
  required physical campaign or clean freeze does not exist. This is a correct
  scientific outcome, not fabricated completion.

## Findings

| ID | Initial finding | Change and affected scope | Validation evidence | Final status | Remaining limitation / allowed claim |
|---|---|---|---|---|---|
| F1-001 | Critical SCF false positive | Canonical parser in `shared/siesta_output_status.py`; joint contract, benchmark manifests, MD and release/ranking gates consume it; historical manifests revalidated without changing raw outputs | `strict_artifact_revalidation_report.json`; parser and gate tests | PASS_ENGINEERING | 658/660 manifests quarantined; only 2 valid. Historical graphene iid20 remains 0/20 valid |
| F1-002 | MD temperature without equilibration | Explicit initialization/equilibration/production schema, executed-step evidence and stability gates in MD generation | `md_temporal_campaign_status.json`; MD contract tests | BLOCKED_FAIL_CLOSED | Historical source is `scf_not_started`; no executed equilibration. Only diagnostic claims |
| F1-003 | Fixed temporal gap without independence proof | ACF, Sokal positive-lag statistical inefficiency, \(\tau_\mathrm{int}\), \(N_\mathrm{eff}\), gap rule and trajectory-block splits implemented | MD tests and temporal campaign audit | BLOCKED_FAIL_CLOSED | Historical data contain no usable temporal series or blocked split |
| F1-004 | FC seeds not physically independent | Seed semantics separated from geometry-family identity; manifests record geometry hashes and independence audit | FC/random tests; `geometry_seed_independence_status.json` | PASS_ENGINEERING | Historical manifests have zero recoverable geometry hashes, so replica-independence is not claimable |
| F1-005 | Random Cartesian split can be empty | Scientific mode requires at least three explicit/derivable families, ratio-aware grouped assignment, non-empty splits and isolation | random Cartesian, fairness and material smoke tests | PASS_ENGINEERING | Sample-id fallback is diagnostic only |
| F1-006 | Electronic comparability lacks convergence evidence | Explicit convergence-evidence schema and fail-closed pooling rule for cutoff, k density, SCF tolerance and conditional basis studies | `electronic_convergence_status.json`; electronic convergence tests | BLOCKED_FAIL_CLOSED | No executed predeclared convergence series; claims restricted to exact material/cell/electronic configuration |
| F1-007 | `si_vacancy` default is smoke | Distinct `production`, `smoke` and `diagnostic` profiles; promotion requires production profile | material preset/bundle tests | PASS_ENGINEERING | Production still needs executed convergence evidence |
| F1-008 | Snapshot unverifiable | Existing commit/hash inventory preserved and extended to three repos, executables, linkage, package versions and ABI probe | `reproducibility_inventory.json` | PRESERVED_PROVEN | Snapshot is pinned but all three repos are dirty; no new paper run allowed |
| F2-001 | Same matrix object not demonstrated | Existing raw-global evidence retained; release remains scoped to explicit `proven` manifests | physical preflight for `md_18`, `md_19` | PRESERVED_PROVEN | Graphene/supported-basis scope only |
| F2-002 | TSHS/HSX equivalence absent | Existing numerical H/S/spectral equivalence retained | raw-global evidence at 17 k-points | PRESERVED_PROVEN | No universal extrapolation |
| F2-003 | Generalized eigenproblem lacks overlap diagnostics | Added Hermiticity, \(\lambda_\min(S)\), condition number, normalized residual, \(c^\dagger Sc\) error and explicit no-silent-regularization policy | extended preflight and tests | PASS_ENGINEERING | Proven only for the two inspected physical samples |
| F2-004 | Renamed ML file accepted as reference | Positive provenance v3 binds reference, RUN.fdf/out, geometry, ORB_INDX, basis/pseudopotentials, SIESTA metadata and frozen split; legacy accepted only as diagnostic | adversarial rename, byte mutation, swap and ambiguity tests | PASS_ENGINEERING | Historical references without provenance remain diagnostic/quarantined |
| F2-005 | DeepH processed splits unproved | Existing 224/224 valid post-preprocessing audits preserved; gates still require them | split audit and ranking tests | PRESERVED_PROVEN | Each new run must emit its own audit |
| F2-006 | R convention / H(k) unevaluable | Existing 17-k-point physical validation preserved | raw-global preflight | PRESERVED_PROVEN | Scoped evidence only |
| F2-007 | Synthetic tests alone do not prove physics | Physical preflight remains separate from unit checks | two physical samples plus regression suite | PRESERVED_PROVEN | Unit tests never elevate scientific scope |
| F2-008 | Runtime/source provenance incomplete | Reproducibility inventory records exact commits, imports, interpreters, binaries, hashes, GPU/CUDA and dependency health; release requires `pinned_clean` | inventory and dirty-state negative tests | BLOCKED_FAIL_CLOSED | Three repositories are dirty |
| F3-001 | FD/autograd mismatch | Existing real-checkpoint Graph2Mat and DeepH comparisons preserved | autograd/FD suite and physical reports | PRESERVED_PROVEN | Internal same-model equivalence, not SIESTA derivative accuracy |
| F3-002 | Central stencil not evaluable | Pairing, axis, atom, delta, hashes, shapes, units and \(2\delta\) denominator remain validated | stencil and direct-prediction tests | PRESERVED_PROVEN | No extension beyond declared stencil |
| F3-003 | Missing \(dS/dR\) | Spectral derivative claims are blocked unless overlap derivatives exist; dH-only scope is explicit | derivative claim/gate tests | PASS_ENGINEERING | No \(dE/dR\) claim |
| F3-004 | Derivative units/calibration unclear | Explicit eV/Å metadata required; non-diagnostic paths require hashes/provenance | derivative tests and gate | BLOCKED_FAIL_CLOSED | Independent SIESTA physical calibration is absent |
| F3-005 | Delta stability not demonstrated | Predeclared delta protocol, minimum three deltas, convergence thresholds, noise/support gates and machine-readable report implemented | `derivative_campaign_status.json`; derivative gate tests | BLOCKED_FAIL_CLOSED | Real iid300 campaign fails support, delta, noise, ordering/gauge, hash and split requirements |
| F3-006 | Symmetry document overclaims | Rewritten to distinguish implementation checks from physical validation and to state dH-only scope | documentation review plus suite | PASS_ENGINEERING | Physical validation remains blocked as above |
| F3-007 | Adaptive test use | Validation-only adaptation, immutable final-test chronology and release/ranking gates enforced | test-blindness and Phase-6 analyzer tests | PASS_ENGINEERING | No completed Phase-6 campaign proves historical chronology |
| F4-001 | Absolute \(N_{\min}\) overclaim | Absolute crossing stays null when unobserved; diagnostic plateau/relative estimates are separately labelled | N_min execution audit and summary | PASS_ENGINEERING | \(N_{\min}^\mathrm{abs}=\) null for both models at 8/10/12 meV |
| F4-002 | Bootstrap hierarchy incomplete | Seed/config/block/fit/dataset-size hierarchy, 2,000 replicates and availability flags implemented | N_min suite and execution audit | BLOCKED_FAIL_CLOSED | Block/trajectory hierarchy unavailable in historical metrics |
| F4-003 | Fit-family sensitivity missing | Predeclared linear/quadratic/inverse/inverse-square/power-law-floor fits, predictive selection and leave-one-N-out diagnostics implemented | N_min summary | BLOCKED_FAIL_CLOSED | Locked power-law-floor is not predictive winner; no absolute crossing |
| F4-004 | Phase 6 not executed / too few seeds | Analyzer verifies physical runs, strict dataset identity and minimum seed policy | `phase6_campaign_status.json` | BLOCKED_FAIL_CLOSED | 0/21 runs, 3 configured seeds versus minimum 5, stale dataset identity |
| F4-005 | Selection/test chronology incomplete | Validation-only selection and one locked final-test evaluation are required by gate | blindness/release/analyzer tests | PASS_ENGINEERING | Empirical closure awaits Phase 6 |
| F4-006 | 10 meV treated as universal | Protocol labels threshold internal/material-specific and requires 8/10/12 meV sensitivity | threshold config and N_min audit | PASS_ENGINEERING | No threshold has an absolute crossing |
| F4-007 | Diagnostic artifacts can rank | Existing fail-closed ranking expanded to provenance, dirty state and current schemas | ranking/top-k/release tests | PRESERVED_PROVEN | Recheck when adding new ranking paths |
| IF-001 | Cross-model equivalence absent | Same bounded raw-global preservation as F2-001/F2-002 | physical preflight | PRESERVED_PROVEN | Not universal |
| IF-002 | Invalid SIESTA evidence | Same canonical parser/quarantine as F1-001 | strict revalidation | PASS_ENGINEERING | Historical paper claims blocked |
| IF-003 | H/S/H(k)/S(k)/E(k) chain absent | Same physical equivalence and generalized diagnostics as F2-001–F2-003 | extended preflight | PRESERVED_PROVEN | Two samples only |
| IF-004 | FD/autograd absent | Same bounded evidence as F3-001 | real-checkpoint tests/reports | PRESERVED_PROVEN | Not SIESTA derivative accuracy |
| IF-005 | Missing \(dS/dR\) | Same spectral claim block as F3-003 | derivative gate tests | PASS_ENGINEERING | dH-only |
| IF-006 | MD/FC/random independence missing | Scientific split/temporal/geometry gates implemented | temporal and geometry audits | BLOCKED_FAIL_CLOSED | Historical evidence insufficient |
| IF-007 | \(N_{\min}\) robustness missing | Full audit machinery executed at 2,000 replicates and three thresholds | N_min audit | BLOCKED_FAIL_CLOSED | No publishable absolute crossing/hierarchy |
| IF-008 | Test blindness partial | Automatic chronology gate implemented | blindness and Phase-6 tests | BLOCKED_FAIL_CLOSED | Phase 6 has 0/21 completed runs |
| IF-009 | Guardrails only documentary | Negative tests and fail-closed gates preserved and expanded | final full suite | PRESERVED_PROVEN | New paths must route through shared gates |
| IF-010 | Snapshot integrity absent | Exact pinned inventory exists; clean-state requirement added | reproducibility inventory | BLOCKED_FAIL_CLOSED | Must intentionally commit/freeze all three repos |

## Final validation anchors

- Strict historical revalidation:
  `Comparison/results/audit_remediation_20260728/strict_artifact_revalidation_report.json`
  — 660 manifests, 79,946 unique snapshots, 658 quarantined and 2 valid.
- Raw-global H/S preflight:
  `Comparison/results/audit_remediation_20260728/raw_global_overlap_diagnostics_h1e-8/deeph_raw_global_equivalence_preflight.json`
  — 2/2 physical samples proven, no regularization.
- N_min:
  `Comparison/results/audit_remediation_20260728/n_min_paper_audit/n_min_remediation_execution_audit.json`
  — 2,000/2,000 replicates and 8/10/12 meV sensitivity executed, scientific
  status diagnostic only.
- Full suite:
  `Comparison/results/audit_remediation_20260728/pytest_full_suite_final.xml`
  — 1,719 tests passed, 1 skipped, 43 subtests passed, zero failures/errors.

## Baseline and non-regression

- Initial focal baseline: 334 tests plus 2 subtests passed.
- Final integrated suite: 1,719 tests plus 43 subtests passed; 1
  environment-dependent test skipped.
- Repository commit inspected: `39fc96b508b97fb92c5603a9258d4caf1ce47fa5`.
- Pre-existing user work, including the graphene-vacancy builder and UI/config
  work, was preserved; no commit, reset, clean, push or destructive rewrite was
  performed.
