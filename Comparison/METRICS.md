# Comparison Metrics

This file documents the metrics produced by
`Comparison/scripts/evaluate_hamiltonian_metrics.py` for archived MD and
AtomDisplacement runs.

## Data Flow

Each archived result directory is expected to contain:

- `predicted_hamiltonians/<sample>/ML_prediction.HSX`
- `siesta_hamiltonians/<sample>/*.HSX` or `*.TSHS`

The evaluator loads both matrices with `sisl`, compares the predicted
Hamiltonian against the SIESTA reference, writes per-sample CSV files under
`metrics/`, and writes a JSON manifest with summaries and explicit errors.

The UI endpoint `/api/plots` reads those archived CSV files for both
`results_md` and `results_atomdisp`. If a manifest contains an absolute path
from another OS, the UI falls back to the archive directory beside the manifest.

## Sparse Matrix Metrics

The sparse Hamiltonian metrics are computed using a support threshold of
`1e-12`. They intentionally distinguish four comparison domains:

- Reference support: entries non-zero in SIESTA.
- Prediction support: entries non-zero in the ML prediction.
- Union support: entries non-zero in either SIESTA or the prediction.
- Full implicit matrix: only densities and entry counts are reported over the
  full matrix; error values are not averaged over all implicit zeros.

Reported metrics include:

- `mae_ref_eV`, `rmse_ref_eV`: errors on SIESTA non-zero entries.
- `mae_pred_eV`, `rmse_pred_eV`: errors on predicted non-zero entries.
- `mae_union_eV`, `rmse_union_eV`: errors on the union support.
- `relative_frobenius_ref`: Frobenius error on the SIESTA support, normalized by
  the SIESTA Frobenius norm.
- `relative_frobenius_union`: Frobenius error on the union support, normalized
  by the SIESTA Frobenius norm.
- `support_precision`, `support_recall`, `support_f1`: sparsity pattern quality.
- `false_zero_rate`: fraction of SIESTA support missing in the prediction.
- `false_nonzero_rate`: fraction of predicted support absent from SIESTA.
- `hermiticity_ref`, `hermiticity_pred`: relative Hermiticity defect.

Block-wise orbital and distance-dependent metrics are not computed yet because
the archived HSX/TSHS comparison path does not currently expose stable atom-pair
metadata in the result CSVs.

## Spectral Metrics

Eigenvalues are computed by diagonalizing the symmetrized Hamiltonian. When the
SIESTA overlap is available, the prediction is diagonalized using the SIESTA
reference overlap so both spectra are compared in the same generalized
eigenproblem.

Reported metrics include:

- `global_mae_eV`, `global_rmse_eV`: all compared eigenvalues.
- `occupied_mae_eV`, `occupied_rmse_eV`: only bands below the SIESTA Fermi level.
- `fermi_window_mae_eV`, `fermi_window_rmse_eV`: bands within +/- 2 eV of the
  SIESTA Fermi level.
- `gap_ref_eV`, `gap_pred_eV`, `gap_abs_error_eV`: gap metrics around the SIESTA
  Fermi level.

Important: near-Fermi, occupied-band, and gap metrics require a real Fermi level
read from the SIESTA reference file. The evaluator does not estimate or infer a
Fermi level. If SIESTA does not provide one, those metrics are left unavailable,
`fermi_level_source` is set to `unavailable`, and the manifest records a
`missing_fermi_level` error for that sample.

## DOS Metrics

The evaluator builds a Gaussian-broadened total DOS from reference and predicted
eigenvalues using `sigma = 0.10 eV` and a 1000-point energy grid. It reports:

- `dos_wasserstein_eV`: 1D transport distance between normalized DOS curves.
- `dos_l1`: L1 distance between normalized DOS curves.
- `dos_l2`: L2 distance between normalized DOS curves.

## Matrix-Spectrum Relationship

`metrics/matrix_spectrum_relationship.csv` joins sparse and spectral rows by
sample. This is used to test whether matrix improvements translate into
physically meaningful spectral improvements.

Per sample it records:

- `relative_frobenius_union`
- `mae_ref_eV`, `rmse_ref_eV`, `rmse_union_eV`
- `support_f1`
- `global_rmse_eV`
- `fermi_window_rmse_eV` when a real SIESTA Fermi level exists
- `gap_abs_error_eV` when a real SIESTA Fermi level exists

The manifest summary includes Pearson correlations for matrix error versus
global spectral RMSE, and Fermi-window correlations only over samples with a
real SIESTA Fermi level.

## Interpreting MD vs AtomDisplacement

The safest comparison is:

- MD prediction vs SIESTA reference.
- AtomDisplacement prediction vs SIESTA reference.
- Then compare the resulting MD and AtomDisplacement metric distributions on
  the same frozen SIESTA-referenced test set.
- Keep the chosen budget mode explicit:
  `equal_sample_count`, `equal_siesta_budget`, or `both`.

Matrix error alone should not be treated as the final answer. A lower matrix
MAE can still fail to improve frontier eigenvalues, gap, or DOS, so the report
keeps matrix, spectral, near-Fermi, DOS, and matrix-spectrum relationship
metrics separate.

For physical conclusions the UI/winner analysis should prefer, in order of
availability:

1. `fermi_window_rmse_eV`
2. `occupied_rmse_eV`
3. `relative_frobenius_union`
4. `dos_wasserstein_eV`
5. `global_rmse_eV` only as a secondary/report metric

Winner outputs preserve `experiment_id`, `seed`, dataset sizes, test set,
metric, and checkpoints. `pooled` aggregation is only produced when explicitly
requested with `--aggregation-mode pooled`. A single seed is reported as a
single-seed result, not as a robust winner.

Winner recommendations are marked `inconclusive` when machine-readable
validation warnings are present in the aggregated cross metrics, including
SIESTA settings mismatch, Graph2Mat model/config mismatch, or a severe budget
mismatch. In strict UI comparison mode, SIESTA/model mismatches and geometry
leakage abort before winner analysis.

`timing_breakdown.json` is written for new UI runs and includes the required
phase keys. Some legacy Graph2Mat/SIESTA entrypoints still expose only coarse
process timings, so missing phase values are explicit `null` values rather than
invented numbers.

## Validation and Leakage Diagnostics

Before a sample is considered scientifically valid, strict validation requires:

- `RUN.fdf`
- a non-predicted Hamiltonian reference; `.TSHS` is preferred over `.HSX`
  when both are present from the same SIESTA run
- `RUN.out`
- SIESTA `Job completed`
- `SCF cycle converged`

Validation artifacts are `valid_samples.csv`, `invalid_samples.csv` and
`validation_summary.json`. Typical invalid reasons include `missing_run_fdf`,
`missing_matrix`, `missing_output`, `job_not_completed`,
`scf_not_converged`, `ambiguous_reference_matrix`, and `parser_error`.

`Comparison/scripts/check_geometry_leakage.py` reports exact duplicates,
near-duplicate geometries, neighboring MD frames crossing splits, and
AtomDisplacement displacement-family leakage. It uses direct atom ordering from
`RUN.fdf`; it does not perform alignment/Kabsch.

For strict one-click comparisons, geometry leakage diagnostics are run after
common frozen tests are built and before cross predictions. Any detected leakage
invalidates that comparison rather than being hidden in a plot.
