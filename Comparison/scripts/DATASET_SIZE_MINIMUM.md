# Dataset Size Minimum

Read-only post-processing for Graph2Mat vs DeepH scaling sweeps. It estimates
`N_min` thresholds from existing metric tables and does **not** train models or
regenerate datasets.

## Nominal N vs effective N (N_eff)

`N_min` fits and threshold crossings use **nominal** dataset sizes:

- `N_train` / `N_total` from metric rows, `frozen_split_manifest.json`, or
  related manifests.

For MD-generated benchmarks, consecutive snapshots along a trajectory are often
**autocorrelated**. Nominal `N_train` therefore counts correlated frames, not
independent observations. Reporting only `N_min(N_train)` can be optimistic when
snapshot spacing is small or when train/validation/test splits do not enforce a
sufficient temporal gap.

### Temporal diagnostics (diagnostic only)

When dataset roots expose enough metadata (`split_summary.json`,
`frozen_split_manifest.json`, split manifests, per-sample `metadata.json`), the
analysis adds `temporal_diagnostics` to the summary:

| Field | Meaning |
|---|---|
| `nominal_n_train` | Train split size from manifests |
| `estimated_n_eff_train` | Diagnostic effective train size when a cheap scalar series exists |
| `autocorrelation_available` | Whether ACF / statistical_inefficiency / N_eff could be estimated |
| `autocorrelation_convention` | Registered convention id (`sokal_positive_lag_inefficiency_v1`) |
| `temporal_diagnostics.datasets[]` | Per-dataset block counts, `temporal_gap`, split strategy, warnings |
| `n_min_basis` | Always `nominal`; thresholds are computed on nominal N |
| `N_min_nominal` | Per-method nominal threshold map copied from the canonical criteria |
| `N_eff_diagnostic_available` | Whether `N_eff_over_N_nominal` could be estimated |
| `N_eff_over_N_nominal` | Diagnostic ratio used to contextualize nominal thresholds |
| `effective_samples_at_nominal_N_min_diagnostic` | Estimated effective sample count at the nominal `N_min`; diagnostic only; not an alternative threshold |
| `scientific_claim_status` | `diagnostic_only` when temporal evidence blocks paper-level claims |
| `paper_level_blockers` | Machine-readable blockers such as missing autocorrelation or very low N_eff |

Autocorrelation uses a lightweight scalar per snapshot when available (for
example total energy or displacement magnitude from `metadata.json`). It
computes:

- autocorrelation function up to a modest lag cap;
- `statistical_inefficiency = 1 + 2 Σ ρ(k)` (positive lags only; `tau_int` is a legacy alias);
- `N_eff = N / statistical_inefficiency`.

**Important:** `N_eff` is reported for transparency. The main `N_min` curves and
replicate-resampling CIs still use nominal `N` until a future protocol explicitly
switches to effective sizes.

The report/UI therefore uses this warning verbatim:

> N_min uses nominal N. If MD snapshots are autocorrelated, independent sample count can be lower. Check N_eff before using this as a paper-level claim.

If autocorrelation is unavailable, `scientific_claim_status` is
`diagnostic_only` and `paper_level_blockers` includes
`paper_blocked_if_autocorrelation_unavailable`. If `N_eff/N_nominal` is below
the configured diagnostic threshold, the status is also diagnostic-only with
`paper_blocked_if_n_eff_much_smaller_than_nominal`. Effective-N fields are
diagnostic context, not validated paper-level replacement thresholds. In
particular, `effective_samples_at_nominal_N_min_diagnostic` is the estimated
effective sample count corresponding to the nominal `N_min`; a true
effective-N threshold is not implemented in this analysis.

Backward-compatible aliases currently remain in JSON:

- `effective_samples_at_N_min_nominal`
- `N_min_eff_diagnostic` with
  `N_min_eff_diagnostic_deprecated_alias_for = "effective_samples_at_nominal_N_min_diagnostic"`

## Replicate-resampling CI

The canonical summary object is `replicate_bootstrap`. The legacy `bootstrap`
key is retained as a deprecated compatibility alias for existing UI/API
consumers.

The display label is **replicate-resampling CI**. It resamples replicate rows
within each `(method, dataset_size_x)` group. This is useful for seed/replicate
variation, but it is **not** a temporal/block bootstrap and does not capture:

- temporal autocorrelation;
- model-selection uncertainty;
- hyperparameter-selection uncertainty;
- dependence between dataset sizes.

When enabled, the summary/report/UI expose those limitations through warnings
and `limitations`.

## Fit policy

`power_law_floor` is the only fit marked `paper_candidate`, and only when the
constrained nonnegative fit succeeds and there are at least 5 observed dataset
sizes for that method. The fit can still run with fewer points for diagnostic
use; the paper-candidate gate is stricter than the mathematical minimum needed
to execute the fit. `power_law` remains a legacy alias for that same
constrained model.

The following fits are `diagnostic_only`: `linear`, `quadratic`, `inverse`,
`inverse_square`, `moving_average`, LOWESS variants, `cumulative_best`, and
`none`. If an unconstrained fit predicts negative error inside the fit domain,
it is marked invalid for N_min thresholding and the affected N_min values fall
back to observed points with an explicit warning.

## Leave-one-size-out fit stability

When `n_min_source=fit`, the summary also records
`fit_predictive_stability_by_left_out_N`. For each method and each observed
dataset size, the diagnostic removes that size, refits with the same
`n_min_fit_model`, and recomputes fitted `N_min_abs`, `N_min_rel_tol`, and
`N_min_plateau` where possible.

The summary reports:

- `n_leave_one_out_trials`
- `n_successful`
- `n_failed`
- `max_abs_delta_N_min`
- `max_relative_delta_N_min`
- `unstable_criteria`

For paper-level gating, the current conservative default is:

- block if any paper-relevant fitted criterion changes by more than one
  observed size step;
- block if too many leave-one-out fits fail.

Observed-only mode marks this diagnostic as not applicable.

### How to read warnings

- `temporal_metadata_missing_*` — no usable temporal manifests; only nominal N is known.
- `temporal_gap_le_1_*` — adjacent MD frames may leak across split boundaries.
- `autocorrelation_unavailable_*` — no cheap scalar series; N_eff is not estimated.
- `n_eff_much_smaller_than_nominal` — estimated N_eff is far below nominal N_train.
- `N is nominal; N_eff not estimated` — safe default message in UI/report.

Treat `N_min` as a **lower bound subject to temporal independence assumptions**
when MD snapshots drive the dataset.

## CLI

```bash
python Comparison/scripts/g2m_deeph_dataset_size_minimum.py \
  --run-root Comparison/results/<sweep> \
  --output-dir Comparison/results/dataset_size_minimum_<tag> \
  --threshold-mev 20 \
  --x-axis n_train
```

Outputs include `dataset_size_minimum_summary.json` with `temporal_diagnostics`.

`N_min_cost_eff` also records the selected `cost_basis`:

- `per_seed_mean` — mean GPU-hours per valid seed/replicate row; preserves historical behavior.
- `protocol_total` — summed GPU-hours across the valid rows required by the aggregated protocol.

Aggregated rows expose `gpu_hours_per_seed_mean`, `gpu_hours_protocol_total`,
and `gpu_hours_protocol_sem` when those values can be computed from the
available per-run costs.

## N_min criteria names

- `N_min_abs`: first dataset size crossing the absolute error threshold.
- `N_min_rel_tol`: first dataset size whose error is within relative tolerance
  of the best observed/fitted value, i.e. `best * (1 + relative_tolerance)`.
- `N_min_plateau`: first dataset size after which the remaining future gain is
  below the plateau threshold.
- `N_min_cost_eff`: lowest-cost dataset size among points within relative
  tolerance of the best observed value, using the selected `cost_basis`.

`N_min_rel95` is a deprecated JSON compatibility alias for `N_min_rel_tol`.
It is not a 95% confidence interval. The only 95% quantity in this analysis is
the optional replicate resampling confidence interval when `ci_level=0.95`.
