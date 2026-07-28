# Final scientific-computational remediation report

## Verdict

The remediation is complete at the engineering and fail-closed control layer.
The repository now rejects unsupported scientific promotion instead of
manufacturing missing evidence.

The repository is **not paper-ready today**. Its final state is
`BLOCKED_FAIL_CLOSED` because the physical campaigns needed to close MD
independence, electronic convergence, derivative calibration/stability,
Phase 6, absolute \(N_{\min}\), and a clean three-repository freeze do not
exist in the inspected artifacts.

This distinction matters:

- the implementation and regression suite pass;
- several bounded physical equivalence claims remain proven;
- broader publication claims are automatically blocked.

## Score

| Dimension | Score | Rationale |
|---|---:|---|
| Software engineering and automated controls | 9.0/10 | Shared contracts, adversarial provenance, atomic materialization, strict profiles and 1,719 passing tests |
| Reproducibility infrastructure | 8.5/10 | Exact three-repo/runtime inventory and fail-closed dirty-state gate; clean commits still absent |
| Existing scientific evidence | 5.0/10 | Strong bounded H/S equivalence, but historical SIESTA, MD, derivative, Phase-6 and N_min evidence is incomplete |
| Paper readiness | 5.0/10 | Correctly blocked; missing evidence cannot be repaired in code |
| Overall repository state | **7.0/10** | Technically robust and scientifically honest, but not yet a complete publishable campaign |

## Implemented outcome

1. A single strict SIESTA status parser now governs joint validation,
   manifests, MD and release/ranking decisions.
2. Revalidation scanned 660 manifests and 79,946 unique snapshots: 658 are
   quarantined and 2 remain valid. The critical graphene iid20 dataset is
   0/20 valid.
3. Production references require positive provenance tied to hashes,
   geometry, SIESTA execution and frozen dataset/split identity. Renaming an ML
   file no longer promotes it; legacy matrices are diagnostic only.
4. MD now has explicit equilibration/production semantics, ACF,
   \(\tau_\mathrm{int}\), \(N_\mathrm{eff}\), temporal-gap and block-split
   contracts. The historical campaign fails these gates honestly.
5. FC/random geometry identity is separated from training seed. Scientific
   random splits require at least three independent families and non-empty
   grouped train/validation/test sets.
6. Smoke, diagnostic and production profiles are distinct. Smoke results
   cannot reach paper/release status.
7. Electronic pooling requires predeclared convergence evidence.
8. Generalized eigensolvers now archive overlap positivity/conditioning,
   Hermiticity, residuals, \(c^\dagger Sc\) normalization and regularization
   status.
9. Derivative gates cover positive provenance, delta sweep, reference noise,
   support continuity, ordering/gauge, hashes, dataset independence and
   dH-only versus spectral scope.
10. Test blindness, seed-aware ranking and dirty-state exclusions are enforced
    by release/ranking gates.
11. N_min executes 2,000 hierarchical bootstrap replicates and 8/10/12 meV
    sensitivity. Absolute N_min remains null, as required by the data.
12. Phase 6 is machine-audited and blocked at 0/21 executions.

## Bounded positive physical result

The raw-global Graph2Mat/DeepH preflight remains `proven` for physical graphene
samples `md_18` and `md_19`, across 17 k-points each.

- \( \lambda_\min(S) \ge 0.1649 \)
- \( \kappa(S) \le 15.2814 \)
- maximum normalized generalized-eigenpair residual:
  \(1.1684\times10^{-13}\)
- maximum \(c^\dagger Sc\) normalization error:
  \(3.1087\times10^{-15}\)
- no regularization applied

This evidence is not extrapolated to other materials, bases or manifests.

## Remaining blockers and exact resumptions

### 1. Clean freeze

Review and intentionally commit the desired changes in all three repositories,
then rerun:

```bash
.venv/bin/python Comparison/scripts/ops/write_reproducibility_inventory.py \
  --output Comparison/results/audit_remediation_20260728/reproducibility_inventory.json
```

Required result: `reproducibility_status=pinned_clean`.

### 2. New MD paper-candidate dataset

Create a new config and output directory; never overwrite the historical data:

```bash
PIPELINE_CONFIG_PATH=/absolute/path/to/a/new-paper-candidate-config.yaml \
  .venv/bin/python MD/scripts/generate_md_dataset.py
```

The resulting manifest must show valid equilibration and production SIESTA
executions, stability thresholds passed, ACF/\(\tau_\mathrm{int}\),
\(N_\mathrm{eff}\), and a time-blocked split derived from the declared gap
rule.

### 3. Electronic convergence

Execute predeclared series with at least three points each for MeshCutoff,
reciprocal-space k density and SCF tolerance; add a basis series whenever the
claim changes basis. Pooling remains restricted until the generated evidence
passes `shared/electronic_convergence.py`.

### 4. Derivative references and gates

Regenerate into new scratch/log paths:

```bash
.venv/bin/python Comparison/scripts/ops/regenerate_derivative_siesta_references.py \
  --scratch /absolute/new/scratch \
  --log /absolute/new/logs/derivative_references.log \
  --reference-workers 1 \
  --threads 2
```

Then rerun the derivative metrics/gate with at least three predeclared deltas,
independent references, noise repeats and complete ordering/hash/split
metadata. Do not add spectral \(dE/dR\) claims without \(dS/dR\).

### 5. Phase 6

First replace the stale reusable dataset ID with a current strict
`benchmark_ready` identity and predeclare at least five seeds or a power
justification. With the local experiment service running:

```bash
curl -sS -X POST http://127.0.0.1:8770/api/experiment \
  -H 'Content-Type: application/json' \
  --data-binary @Comparison/config/h2o_phase6_hamiltonian_architecture_benchmark_payload.json

.venv/bin/python Comparison/scripts/analyze_md1000_phase6_benchmark.py
```

Required result: 21/21 completed physical runs, validation-only selection and
one locked final-test evaluation.

### 6. N_min

Rerun N_min only after the new strict datasets and temporal evidence exist.
Keep 2,000 replicates, the full hierarchy and 8/10/12 meV sensitivity.
Publication requires an observed/supported absolute crossing and a stable
predeclared fit policy; plateau estimates alone are diagnostic.

## Evidence index

- [Remediation ledger](scientific_remediation_ledger.md)
- [Strict artifact revalidation](../Comparison/results/audit_remediation_20260728/strict_artifact_revalidation_report.json)
- [Reproducibility inventory](../Comparison/results/audit_remediation_20260728/reproducibility_inventory.json)
- [MD temporal audit](../Comparison/results/audit_remediation_20260728/md_temporal_campaign_status.json)
- [Geometry independence audit](../Comparison/results/audit_remediation_20260728/geometry_seed_independence_status.json)
- [Electronic convergence status](../Comparison/results/audit_remediation_20260728/electronic_convergence_status.json)
- [Derivative campaign status](../Comparison/results/audit_remediation_20260728/derivative_campaign_status.json)
- [Raw-global preflight](../Comparison/results/audit_remediation_20260728/raw_global_overlap_diagnostics_h1e-8/deeph_raw_global_equivalence_preflight.json)
- [Phase-6 status](../Comparison/results/h2o_md1000_phase6_architecture_benchmark/phase6_campaign_status.json)
- [N_min execution audit](../Comparison/results/audit_remediation_20260728/n_min_paper_audit/n_min_remediation_execution_audit.json)
- [Final JUnit](../Comparison/results/audit_remediation_20260728/pytest_full_suite_final.xml)

## Verification

```text
1719 passed, 1 skipped, 151 warnings, 43 subtests passed
0 failures, 0 errors
578.68 s
```

Warnings are dependency deprecations plus the reproduced NumPy/netCDF4 ABI
warning. `pip check` passes and the minimal netCDF4 numeric roundtrip succeeds,
but the ABI warning remains recorded and is not silently ignored.
