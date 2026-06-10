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

Autocorrelation uses a lightweight scalar per snapshot when available (for
example total energy or displacement magnitude from `metadata.json`). It
computes:

- autocorrelation function up to a modest lag cap;
- `statistical_inefficiency = 1 + 2 Σ ρ(k)` (positive lags only; `tau_int` is a legacy alias);
- `N_eff = N / statistical_inefficiency`.

**Important:** `N_eff` is reported for transparency. The main `N_min` curves and
bootstrap CIs still use nominal `N` until a future protocol explicitly switches
to effective sizes.

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
