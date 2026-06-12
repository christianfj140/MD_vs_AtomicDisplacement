# Dataset Size Minimum

Read-only post-processing for Graph2Mat vs DeepH scaling sweeps. It estimates
`N_min` thresholds from existing metric tables and does **not** train models or
regenerate datasets.

## Metric-specific threshold presets

`threshold_mev` is now treated as a metric-specific protocol choice rather than
as a universal `20 meV` rule.

Current UI/backend presets are explicit per metric:

- `h_mae_eV_mean` — exploratory presets distinct from spectral metrics
- `h_rmse_eV` — exploratory presets distinct from H-MAE
- `low_energy_rmse_eV` — exploratory spectral presets
- `fermi_window_rmse_eV` — exploratory Fermi-window spectral presets

The summary JSON records:

- `threshold_basis`
- `threshold_reference`
- `threshold_interpretation`
- `threshold_metric_family`
- `threshold_is_user_defined`
- `threshold_preset_key`

Important semantics:

- `20 meV` is **not universal**
- metric-specific presets are currently exploratory defaults unless stronger
  justification is documented
- a manual threshold is recorded as `user_defined_exploratory`
- exploratory threshold choices should keep `scientific_claim_status` at
  `diagnostic_only` or add a paper blocker unless a stronger threshold protocol
  is supplied separately

## Aggregation mode

Aggregation mode is part of the reproducibility contract for Dataset Size
Minimum and should be treated as an explicit protocol choice, not a cosmetic UI
setting.

Preferred paper-ready mode:

- `mean_seeds_per_config`

Other supported modes:

- `best_config_mean` — paper-candidate only if the config-selection policy is
  locked and documented
- `mean_replicates` — diagnostic only
- `best_config` — diagnostic only

The summary JSON records:

- `requested_aggregation_mode`
- `actual_aggregation_mode`
- `aggregation_mode_classification`
- `aggregation_mode_classification_reason`
- `aggregation_mode_legacy_inferred`

If older CLI calls omit `aggregation_mode`, the current transition is
warning-first rather than hard-fail:

- the script still resolves the legacy fallback
  (`best_config` for one run root, `mean_replicates` for multiple);
- a warning is recorded in `warnings`;
- the summary marks the mode as legacy/inferred.

Old summaries without stored aggregation metadata still render in the UI/API,
but they are labeled as legacy inferred rather than fully explicit.

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
| `N_eff_by_dataset_size` | Diagnostic effective sample count keyed by nominal dataset size |
| `N_eff_over_N_by_dataset_size` | Per-size diagnostic ratio keyed by nominal dataset size |
| `autocorrelation_available_by_dataset_size` | Whether the per-size N_eff diagnostic was defensible for that size |
| `temporal_block_diagnostics_by_dataset_size` | Per-size block/trajectory diagnostics and per-dataset block summaries |
| `N_min_effective_diagnostic` | Effective-sample diagnostic at nominal `N_min` when the exact nominal size has a matching per-size `N_eff` entry |
| `effective_samples_at_nominal_N_min_diagnostic` | Estimated effective sample count at the nominal `N_min`; diagnostic only; not an alternative threshold |
| `scientific_claim_status` | `diagnostic_only` when temporal evidence blocks paper-level claims |
| `paper_level_blockers` | Machine-readable blockers such as missing autocorrelation or very low N_eff |

Autocorrelation uses a lightweight scalar per snapshot when available (for
example total energy or displacement magnitude from `metadata.json`). It
computes:

- autocorrelation function up to a modest lag cap;
- `statistical_inefficiency = 1 + 2 Σ ρ(k)` (positive lags only; `tau_int` is a legacy alias);
- `N_eff = N / statistical_inefficiency`.

Trajectory/block assumptions are conservative:

- prefer `trajectory_id` when available;
- else group by `block_id + temperature_K` when available;
- else allow a single implicit continuous block only when metadata proves a
  single ordered trajectory;
- otherwise mark autocorrelation as unavailable for paper-level claims.

The diagnostic does **not** concatenate unrelated blocks or mixed temperatures
into one continuous time series. Per-block diagnostics are reported separately,
including:

- `nominal_n`
- `scalar_used`
- `max_lag`
- `statistical_inefficiency`
- `n_eff`
- `warnings`

Aggregate `estimated_n_eff_train` is reported only when the grouping assumptions
are defensible. It remains diagnostic context, not a validated replacement for
nominal `N_min`.

The summary also records per-size temporal diagnostics:

- `N_eff_by_dataset_size`
- `N_eff_over_N_by_dataset_size`
- `autocorrelation_available_by_dataset_size`
- `temporal_block_diagnostics_by_dataset_size`

These fields are keyed by nominal dataset size and answer a stricter
reproducibility question than the global `N_eff/N_nominal` ratio: whether each
size used by the sweep has defensible temporal diagnostics. If any dataset size
used by a fitted `N_min` lacks defensible `N_eff`, paper-level status is
blocked via `paper_blocked_if_n_eff_by_dataset_size_incomplete`.

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
effective-N threshold is not implemented in this analysis. Likewise,
`N_min_effective_diagnostic` is only populated when the exact nominal threshold
size has a matching per-size `N_eff` diagnostic. It does **not** replace
`N_min_nominal`, and it does not make the result paper-ready by itself.

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

`N_min_cost_eff` is handled more conservatively than the nominal-size
thresholds:

- `BOOTSTRAP_N_MIN_CRITERIA` does **not** include `N_min_cost_eff`
- `replicate_bootstrap.cost_eff_ci_available = false`
- `replicate_bootstrap.cost_eff_ci_policy = "excluded_no_joint_metric_cost_resampling"`

Why: the current replicate-resampling diagnostic does not jointly resample
metric and cost under the selected `cost_basis`, so attaching a CI to
`N_min_cost_eff` would overstate what the bootstrap actually supports.

The UI/report therefore states explicitly that `N_min_cost_eff` has **no
replicate-resampling CI** in the current protocol. If rows are missing cost for
the selected basis, that is also surfaced in the bootstrap warnings rather than
being converted into a fake interval.

## Hierarchical uncertainty

The summary now also exposes `hierarchical_uncertainty` as a paper-readiness
audit layer. It is deliberately separate from `replicate_bootstrap`.

`replicate_bootstrap` remains a backward-compatible **diagnostic** interval for
row-level replicate resampling. It is **not** a paper-level uncertainty model.

`hierarchical_uncertainty` separates uncertainty sources instead of collapsing
them into one ambiguous interval:

- `seed` — seed-to-seed variability within the same base config
- `config` — config/hyperparameter selection variability across base configs
- `block` — block/trajectory temporal variability from per-block `N_eff / N`
- `fit_model` — selected fit policy plus successful alternative model statuses
- `dataset_size_dependence` — leave-one-size-out stability of fitted `N_min`

Paper-ready uncertainty requires explicit hierarchy support. If the necessary
metadata is missing, the analysis stays `diagnostic_only` and records machine-
readable blockers such as:

- `paper_uncertainty_seed_hierarchy_incomplete`
- `paper_uncertainty_config_hierarchy_incomplete`
- `paper_uncertainty_block_hierarchy_unavailable`
- `paper_uncertainty_fit_model_not_paper_candidate:*`
- `paper_uncertainty_fit_model_selection_not_paper_candidate:*`

Important semantics:

- no fallback from missing block hierarchy to row bootstrap is treated as
  paper-ready;
- block uncertainty requires explicit temporal grouping metadata and multiple
  defensible train blocks;
- `hierarchical_uncertainty` does not replace nominal `N_min` thresholds;
- deterministic seeds are used for the audit resampling so the JSON is
  reproducible across repeated runs with the same inputs.

## Fit policy

`power_law_floor` is the only fit marked `paper_candidate`, and only when the
constrained nonnegative fit succeeds and there are at least 5 observed dataset
sizes for that method. The fit can still run with fewer points for diagnostic
use; the paper-candidate gate is stricter than the mathematical minimum needed
to execute the fit. `power_law` remains a legacy alias for that same
constrained model.

For `power_law_floor`, alpha is no longer chosen only from a fixed coarse grid.
The current implementation uses a deterministic bounded search:

- coarse alpha scan on `[0.05, 4.0]`;
- local golden-section refinement around the best coarse alpha;
- constrained nonnegative re-fit of `E_inf` and `A` at each alpha evaluation.

The fit summary/report records:

- `alpha`
- `alpha_search_method`
- `alpha_bounds`
- `alpha_refinement_interval`
- `objective_evaluations`
- `sse`
- `rmse_mev`
- `fit_domain`
- `nonnegative_constraints_active`

This preserves the same model,

`y(N) = E_inf + A * N^(-alpha)` with `E_inf >= 0`, `A >= 0`, `alpha > 0`,

while giving a more stable alpha estimate for paper-candidate screening. A
power-law fit is still not sufficient by itself for a paper-level claim: the
leave-one-size-out stability gate must also pass.

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
- `autocorrelation_grouping_missing_or_ambiguous` — trajectory/block identity is
  not strong enough to justify a continuous ACF estimate.
- `autocorrelation_unavailable_mixed_temperatures` — different temperatures were
  detected and are not concatenated into one time series.
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
