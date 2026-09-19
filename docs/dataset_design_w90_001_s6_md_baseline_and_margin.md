# DATASET-DESIGN-W90-001-S6 — MD baseline and non-inferiority margin

Cites `docs/dataset_design_w90_001_s1_convention_manifest.md` (S1) and reuses
S4 (`dataset_design_w90_001_s4_train_learning_curves.py`) / S5
(`dataset_design_w90_001_s5_frozen_tests_generalization.py`) directly, per
S1 §4. Produced by
`Comparison/scripts/dataset_design_w90_001_s6_md_baseline_and_margin.py`;
numbers below are from the real run under
`Comparison/results/dataset_design_w90_001_s6/`.

## (a) MD train/frozen split verification — PASS, with a load-bearing caveat

`verify_md_split_integrity()` reads each of the 5 `MD_dataset*_*` runs'
`splits/split_summary.json` + train/validation/test manifests
(`blocked_with_gap` strategy, `temporal_gap=1` in every dataset) and confirms
no frame index is shared between two partitions in any dataset
(`no_partition_overlap: true`, `split_verified: true`,
`md_split_integrity_report.json`).

**Caveat (the central finding of this step):** across the whole repository,
only 20 MD frames anywhere still have a persisted SIESTA-computed
Hamiltonian on disk (`Comparison/results/results_md/MD_dataset*/run_*/
siesta_hamiltonians/<frame_index>/`), and **every single one of those 20
falls inside the *test* partition** of its dataset's split
(`persisted_outside_test_partition: []` for all 5 datasets). The
train/validation-partition Hamiltonians were written under
`Comparison/workspaces/<run_id>/...`, which has since been cleaned up —
`train_block_hamiltonians_available: false`. This is the same data loss S5's
docstring already flagged for its own `md_frozen` frame pool; S6 confirms it
is total, not partial.

MD frozen decorrelation (S5's `build_md_frozen_samples`, reused verbatim via
`check_md_frozen_decorrelation`) is verified: the slowest-decorrelating
configurational observable (mean bond angle / pyramidalization) gives
`g_used = 4`; the 7 frames S5 selects for `md_frozen` are all pairwise
spaced ≥ 4 apart (`decorrelation_verified: true`,
`md_frozen_decorrelation_report.json`).

## (b) MD baseline training at N_train ∈ {16, 32, 64} — not possible with current data

S5 already claims 7 of the 20 persisted frames as the `md_frozen` frozen
test. That leaves at most **13** frames anywhere in the repo that could
serve as an MD *training* pool disjoint from `md_frozen`
(`md_training_pool_indices`) — short of even N_train=16.
`run_md_baseline_combo()` (which trains/evaluates identically to S4's
`run_one_combo`/S5's evaluation the moment enough data exists) therefore
records every one of the 6 (N_train, seed) combos as
`insufficient_md_training_data`, `available_n=13`
(`md_baseline_metrics.csv`) — never fabricated.

Regenerating a real, densely-sampled MD trajectory to backfill this pool is
the same "separate, much larger integration project" S4 and S5 already
scoped out; S6 does not attempt it here.

Because zero MD baseline models trained, the `common_displacement` /
`ood_displacement` frozen-test SIESTA generation (S5's
`materialize_and_run`) was never invoked — there is nothing to evaluate on
them. The code path exists (`get_frozen_samples_by_test` in the S6 script)
and will run unmodified once a training combo succeeds.

## (c)/(d) Non-inferiority margin Δ_NI — not justified; documenting the limitation

Using only pilot/validation results (never the full 5-seed comparison),
`summarize_pilot_signal()` reads S4's `learning_curves.csv` (120 `ok`
models): relative-Frobenius H error spans **15.8%–22.6%** (median 18.6%),
H-MAE spans **0.23–0.48 eV** (median 0.35 eV).

The task's own candidate margin — 5% relative H-MAE — is examined against
that evidence and rejected by `fix_non_inferiority_margin()`, for two
independent reasons:

1. **No MD baseline exists to be non-inferior to.** (b) above.
2. **The candidate margin is not achievable by anything measured so far.**
   The *best* S4 pilot model's relative-Frobenius error (15.8%) is already
   3.2× the 5% candidate. The S4 pilot architecture (`num_interactions=1`,
   `hidden_irreps="10x0e + 10x1o + 10x2e"`, 60 epochs) is deliberately
   small/fast — not production-representative — so treating its error floor
   as evidence that 5% is reachable would not be defensible.

**Conclusion (Δ_NI is not fixed):** per task requirement (d), this S6 step's
conclusion — and any later step built on it — must be limited to *observed
error/cost and a Pareto front*, without an "equivalent to MD"
non-inferiority claim, until both:

- a real MD training pool is regenerated (distinct project, out of scope
  here and in S4/S5), and
- a production-representative (non-pilot) model accuracy ceiling exists to
  ground a practically-relevant margin.

Full numeric evidence: `Comparison/results/dataset_design_w90_001_s6/
non_inferiority_margin.json`, `md_split_integrity_report.json`,
`md_frozen_decorrelation_report.json`, `md_baseline_metrics.csv`,
`s6_summary.json`.

## Referenced by

Any later step that would build a Pareto frontier or claim MD-equivalence
must cite this file and may not assume a Δ_NI value beyond what is
documented here as "not fixed."
