# Comparison Metrics

This file documents the metrics produced by
`Comparison/scripts/evaluate_hamiltonian_metrics.py` for archived comparison
runs. The current workflow supports the canonical methods `md`,
`siesta_fc_cartesian`, and `random_cartesian`; legacy names such as
`atom_displacement` are normalized by `Comparison/scripts/method_registry.py`.

## Data Flow

Each archived result directory is expected to contain:

- `predicted_hamiltonians/<sample>/ML_prediction.HSX`
- `siesta_hamiltonians/<sample>/*.HSX` or `*.TSHS`

The evaluator loads both matrices with `sisl`, compares the predicted
Hamiltonian against the SIESTA reference, writes per-sample CSV files under
`metrics/`, and writes a JSON manifest with summaries and explicit errors.

Reference selection is fail-closed and shared by validation, archiving, legacy
eigenvalue extraction, and the metrics evaluator:

- exactly one non-predicted `.TSHS` is preferred;
- if no `.TSHS` exists, exactly one non-predicted `.HSX` is allowed;
- multiple `.TSHS` files, multiple fallback `.HSX` files, and any
  `ML_prediction.HSX` reference candidate are rejected.

`metrics/manifest.json` records `reference_selection_policy`, `fatal_errors`,
`warnings`, `samples_failed`, and per-sample `sample_status` rows containing the
selected reference/prediction paths and SHA256 hashes.

Archived run and cross-evaluation manifests also carry material provenance when
available: material label/source, species, atom count, FDF hash,
pseudopotential and basis hashes, SIESTA output flags, Graph2Mat config hash,
split manifest hash, dataset recipe, and aggregate reference/prediction matrix
hashes. Aggregated CSV/JSON rows preserve those fields. If cross-evaluation
detects different material compatibility hashes across methods, the aggregate
row receives an `INCOMPATIBLE_MATERIAL_PROVENANCE` severe warning and winner
analysis treats the comparison as not scientifically valid.

The UI endpoint `/api/plots` reads those archived CSV files from `results_md`,
`results_atomdisp`, and `results_random_cartesian`. If a manifest contains an
absolute path from another OS, the UI falls back to the archive directory beside
the manifest. When material provenance is present, `/api/plots` also exposes
compact material metadata (`material_label`, `material_species`,
`material_atom_count`, `material_identity_hash`, `material_compatibility_hash`,
and `material_display_label`) for archived runs and cross-evaluation rows.

Results plots are material-aware diagnostics. The UI shows material labels in
run labels and hover text, offers an `All materials` filter, and warns when
multiple material compatibility groups are visible. Material-mixed plots are
useful for inspection only; robust scientific comparisons require compatible
material hashes, basis files, pseudopotentials, and SIESTA settings.

The Results tab includes a "DeepH-comparable" diagnostic plot group when the
corresponding archived columns exist:

- Matrix MAE/RMSE in meV from `mae_union_meV`, `rmse_union_meV`,
  `mae_ref_meV`, and `rmse_ref_meV` in `metrics/sparse_metrics.csv`
  (lower is better).
- Matrix MSE from `mse_union_eV2` and `mse_ref_eV2` in
  `metrics/sparse_metrics.csv` (lower is better; units are eV^2).
- Matrix R2 from `r2_union` and `r2_ref` in `metrics/sparse_metrics.csv`
  (higher is better; unavailable for constant targets).
- DeepH-style DOS MAE from `dos_mae_500_fermi_window` in
  `metrics/dos_metrics.csv`; unavailable rows preserve
  `dos_window_unavailable_reason` rather than being treated as zero.
- Orbital-pair MAE heatmaps from `metrics/orbital_pair_summary.csv` using
  `mae_union_meV_mean` grouped by `species_pair`, `row_orbital_index`, and
  `col_orbital_index`. The full per-sample CSV remains
  `metrics/orbital_pair_metrics.csv`.

These plots are diagnostics for comparing repository outputs with DeepH-style
reports. They do not change winner defaults and do not imply exact DeepH H'
local-coordinate equivalence.

## Official H-only Target Semantics

The official Hamiltonian benchmark target is H-only:

```yaml
data:
  out_matrix: hamiltonian
  matrix_component_policy: h_only
  n_matrix_components: 1
  symmetric_matrix: true
```

Non-orthogonal SIESTA files may expose raw matrix components corresponding to
Hamiltonian and overlap data. The official benchmark trains and evaluates only
Hamiltonian component 0 as the target. It must not optimize or report a
non-spin water or graphene model as if an auxiliary S-like channel were a
physical second Hamiltonian or spin target.

Generated Graph2Mat configs, manifests, and metric rows record the matrix
target, component policy, and component count. Missing or mismatched target
policy is treated as legacy/unknown at read time and as a severe blocker for
new official runs.

Training loss is not the primary scientific comparison metric. Reports should
use the explicit H matrix, spectral, DOS, block/orbital, and safety provenance
fields described below.

For the paper-ready Graph2Mat-vs-DeepH workflow, final winner claims are made
only by the preregistered `final_evaluation.primary_metric` together with
`g2m_deeph_final_stats.py` and `g2m_deeph_gate_check.py`. Common H-MAE
summaries and UI recommendations are supporting Hamiltonian diagnostics unless
H-MAE itself was preregistered as the final metric. They must not override a
failed gate status, missing final statistics, or diagnostic-only DeepH
equivalence.

## DeepH Comparability Status

The repository exposes DeepH-comparable diagnostics, not a complete
reproduction of every DeepH benchmark. `metrics/manifest.json` records the same
policy under `deeph_comparability_status`.

The dedicated Graph2Mat-vs-DeepH workflow uses
`joint_graph2mat_deeph_artifact_contract_v1` and aggregates common metrics only
after both methods used the same validated snapshots and frozen split. The UI
summary and `/api/g2m-deeph/plots` payload report `valid_joint_one_pass_dataset`,
`valid_reused_joint_dataset`, `valid_repaired_dataset_with_warning`,
`invalid_*`, or `diagnostic_only`. If the status is `diagnostic_only` or
`invalid_*`, the UI must not declare a winner.

See `docs/graph2mat_deeph_benchmark.md` for the required `HSX`,
`STRUCT_OUT`, `XV`, `ORB_INDX`, `TSHS`, `TSDE` artifact contract and the rule
that missing DeepH artifacts must not trigger silent SIESTA repair.

Implemented repo-compatible analogues:

- Hamiltonian MAE/RMSE/MSE/R2 in `metrics/sparse_metrics.csv` on the repository
  reference, prediction, and union sparse supports.
- meV aliases for Hamiltonian MAE/RMSE, computed as `1000 * eV`.
- `dos_mae_500_fermi_window` in `metrics/dos_metrics.csv` when the SIESTA
  reference provides a real finite Fermi level.
- Orbital-pair CSV diagnostics in `metrics/orbital_pair_metrics.csv` and
  `metrics/orbital_pair_summary.csv` when structure and `.ion.xml` basis
  coverage are available.

Caveats:

- Matrix and orbital-pair metrics use the repository's raw/global Hamiltonian
  basis. They are not exact DeepH local-coordinate H' metrics unless a future
  validated H' transformation is implemented.
- DOS window and units are repository DOS diagnostics and may not be physically
  identical to DeepH's 2D material examples.
- Fermi-dependent metrics are unavailable, not estimated, when SIESTA does not
  provide a real finite Fermi level.

Future work not implemented in this repository phase:

| Area | Why it is future work |
| --- | --- |
| True high-symmetry k-path band structures | Requires explicit k-path input, k-resolved reference/predicted bands, and validation against SIESTA band-structure outputs. Monkhorst-Pack k-point metrics are available only through the explicit `--enable-kpoint-metrics` evaluator path; high-symmetry path band comparisons remain future work. |
| SOC/complex Hamiltonians | The k-point path supports complex Hermitian H(k)/S(k) for the validated non-SOC workflow, but spin-orbit, spinful, and ambiguous multi-component semantics are still unsupported. |
| Optical response, Berry quantities, electric susceptibility and shift current | Requires optical/Berry-response infrastructure, validated wavefunction or velocity/dipole data, and material-specific scientific checks. |
| Ensemble uncertainty | Requires an explicit ensemble protocol and calibrated reliability validation across independent model instances. |
| DeepH-vs-DFT scaling by system size | Requires controlled system-size series, DFT/DeepH timing protocol, and hardware-normalized scaling analysis. |

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
- `mse_ref_eV2`, `mse_pred_eV2`, `mse_union_eV2`: mean squared Hamiltonian
  error on the same support domains as the corresponding MAE/RMSE metrics.
  Units are eV^2.
- `r2_ref`, `r2_pred`, `r2_union`: coefficient of determination on the same
  support domains. R2 is complementary to absolute error metrics and is left
  unavailable when the reference target has zero variance or no comparable
  entries.
- `mae_*_meV`, `rmse_*_meV`: meV aliases for the existing eV MAE/RMSE metrics,
  computed as `1000 * *_eV`.
- `matrix_metric_target_space`: currently `raw_global_hamiltonian`, meaning the
  repository compares the archived global SIESTA Hamiltonian matrix against the
  predicted matrix. These metrics are DeepH-comparable scalar diagnostics, but
  they are not exact DeepH local-coordinate H' block metrics unless a future
  validated H' transformation is implemented.
- Target-semantics columns are repeated in sparse, spectral, DOS, overlap, and
  matrix-spectrum outputs: `metrics_schema_version`,
  `metrics_provenance_generation`, `target_component_policy`,
  `reference_component_count`, `prediction_component_count`,
  `reference_spin_kind`, `prediction_spin_kind`, `overlap_source`,
  `prediction_own_overlap_used`,
  `prediction_overlap_relative_frobenius_vs_reference`,
  `graph2mat_auxiliary_component_ignored`, and
  `prediction_self_contained_hsx_safe`. These fields make H-only versus
  multi-component or ambiguous spin-container predictions explicit in the CSVs.
- `relative_frobenius_ref`: Frobenius error on the SIESTA support, normalized by
  the SIESTA Frobenius norm.
- `relative_frobenius_union`: Frobenius error on the union support, normalized
  by the SIESTA Frobenius norm.
- `support_precision`, `support_recall`, `support_f1`: sparsity pattern quality.
- `false_zero_rate`: fraction of SIESTA support missing in the prediction.
- `false_nonzero_rate`: fraction of predicted support absent from SIESTA.
- `hermiticity_ref`, `hermiticity_pred`: relative Hermiticity defect.
- `metrics/sparse_threshold_sweep.csv`: sensitivity table at thresholds
  `1e-12`, `1e-10`, `1e-8`, `1e-6` with `mae_union_eV` and `rmse_union_eV`.
  This is a robustness diagnostic; it should not replace the canonical table.
- `metrics/component_channel_metrics.csv`: per-channel diagnostics for
  containers that expose multiple matrix components. Component 0 is treated as
  the Hamiltonian channel and is labeled with `component_target_label=H`,
  `component_units=eV`, and `component_is_official_hamiltonian_target=true`.
  Auxiliary channels are reported separately when both reference and prediction
  channels exist, or marked unavailable when the reference has no corresponding
  auxiliary target. Auxiliary-channel rows are never mixed into the main
  Hamiltonian MAE/RMSE/R2 metrics or into official winner metrics.

## Hamiltonian Derivative Sparse Metrics

Derivative metrics compare finite differences of Hamiltonian matrices,
`dH_pred/dR` versus `dH_ref/dR`, in the raw global Hamiltonian basis. They are
not force-constant metrics, phonon dynamical-matrix metrics, or finite
differences of forces. The valid SIESTA reference derivative is built from
Hamiltonian matrices, for example central difference:

In this repository, `dH/dR` means the derivative of Hamiltonian matrix elements
with respect to Cartesian atomic displacement. SIESTA force constants, `.FC`
files, dynamical matrices, and phonons are not `dH/dR` references and must not
be substituted for Hamiltonian finite differences.

```text
dH_ref/dR = (H_SIESTA(R + delta) - H_SIESTA(R - delta)) / (2 * delta)
```

The internal derivative metric target space is:

```text
raw_global_hamiltonian_derivative
```

Implemented sparse derivative metrics include:

- `dh_mae_ref_eV_per_Ang`, `dh_rmse_ref_eV_per_Ang`,
  `dh_mse_ref_eV2_per_Ang2`: derivative errors on the SIESTA derivative
  support.
- `dh_mae_pred_eV_per_Ang`, `dh_rmse_pred_eV_per_Ang`: derivative errors on
  the predicted derivative support.
- `dh_mae_union_eV_per_Ang`, `dh_rmse_union_eV_per_Ang`,
  `dh_max_abs_error_union_eV_per_Ang`: derivative errors on the union support.
- `dh_relative_frobenius_ref`, `dh_relative_frobenius_union`,
  `dh_relative_l1_union`, `dh_cosine_similarity_union`: relative and angular
  diagnostics over sparse derivative values.
- `dh_support_precision`, `dh_support_recall`, `dh_support_f1`,
  `dh_false_zero_rate`, `dh_false_nonzero_rate`: derivative sparsity-support
  diagnostics.
- `dh_hermiticity_ref`, `dh_hermiticity_pred`,
  `dh_hermiticity_error_delta`: Hermiticity diagnostics for reference and
  predicted derivative matrices.

Rows carry derivative-specific metadata: `sample`, `atom_index_zero_based`,
`axis`, `axis_index`, `delta_ang`, `finite_difference_method`,
`source_model`, `reference_source`, `derivative_units`, `hamiltonian_units`,
`displacement_units`, `matrix_metric_target_space`, and `comparison_status`.
Until all required comparability metadata gates are present, derivative metric
rows remain `diagnostic_only`. If the reference derivative norm is zero,
relative metrics are left unavailable as `NaN` and the row records an explicit
unavailability reason.

Required derivative artifacts are:

- `RUN.fdf`
- `metadata.json`
- SIESTA Hamiltonian reference in `.HSX` or `.TSHS`
- predicted `ML_prediction.HSX`
- explicit plus/minus displacement metadata
- `ORB_INDX` and basis/gauge evidence where available

`Comparison/scripts/evaluate_hamiltonian_derivative_metrics.py` evaluates these
metrics from already archived Hamiltonian references and predictions. It does
not run SIESTA, Graph2Mat, or DeepH, and it does not read force constants,
phonon outputs, dynamical matrices, or finite differences of forces. Example:

```bash
python3 Comparison/scripts/evaluate_hamiltonian_derivative_metrics.py \
  <result_dir> \
  --method central \
  --split test \
  --require-central \
  --diagnostic-only \
  --support-threshold 1e-12 \
  --overwrite
```

Implemented evaluator options are:

- `--method {central,forward,backward}`
- `--split {test,validation,train,all}`
- `--require-central`
- `--overwrite`
- `--diagnostic-only`
- `--support-threshold <float>`
- `--max-stencils <int>`
- `--output-dir <path>`
- `--source-model {graph2mat,deeph}`

The derivative evaluator writes:

- `derivative_metrics/manifest.json`
- `derivative_metrics/stencil_status.csv`
- `derivative_metrics/derivative_matrix_metrics.csv`
- `derivative_metrics/derivative_support_sweep.csv`
- `derivative_metrics/derivative_hermiticity.csv`
- `derivative_metrics/derivative_summary.json`

The manifest records `force_constants_used: false`,
`reference_definition: siesta_hamiltonian_finite_difference`, and
`paper_level: false`. `scientific_status` defaults to `diagnostic_only` and is
promoted only to `presentation_ready` for central finite differences when
required metadata, unit, shape, sign, finite-value, and Hermiticity gates pass.
No current derivative evaluator path promotes results to paper-level status.

The fail-closed derivative gate checker consumes those files and emits a report:

```bash
python3 Comparison/scripts/g2m_deeph_derivative_gate_check.py \
  --derivative-root <result_dir>/derivative_metrics \
  --output <result_dir>/derivative_metrics/derivative_gate_report.json
```

Or, for a staged benchmark run with both methods:

```bash
python3 Comparison/scripts/g2m_deeph_derivative_gate_check.py \
  --run-root <benchmark_run_root> \
  --output <benchmark_run_root>/common_metrics/summary/derivative_gate_report.json
```

The gate report contains:

- `scientific_status`
- `allowed_claims`
- `blocked_claims`
- `blockers`
- `warnings`
- `recommended_next_steps`
- `evidence_paths`

Derivative scientific gate levels are:

- `internal_diagnostic`
- `technical_presentation`
- `paper_level_candidate`
- `blocked`

Required blockers include:

- force constants used as reference
- reference definition other than SIESTA Hamiltonian finite difference
- no central stencils
- missing plus/minus pairing
- mismatched shapes, delta, or units
- missing or inconsistent atom indexing
- missing or inconsistent orbital ordering metadata
- high Hermiticity defect
- support discontinuity above threshold

Paper-level candidate status is additionally blocked without:

- basis/gauge evidence
- orbital ordering evidence
- delta sensitivity study
- independent dataset/split metadata
- proven Graph2Mat/DeepH equivalence when both are compared

Derivative limitations that must be stated explicitly in reports:

- gauge, basis, and orbital ordering can remain ambiguous
- neighbor-list or sparsity discontinuities may dominate derivative errors
- finite-difference sensitivity to `delta` must be checked
- ML prediction noise can be amplified by finite differences
- no force-constants comparison is implemented

Training loss is not an official winner metric. For H-only benchmark configs,
`loss: graph2mat.metrics.block_type_mae` is interpretable as Hamiltonian-only
only because the generated/validated Graph2Mat config requires
`matrix_component_policy: h_only` and `n_matrix_components: 1`. Reports should
therefore cite `h_matrix_mae_*`, `h_matrix_rmse_*`, spectral metrics, and
component-channel diagnostics rather than training loss alone.

Post-H-only/S_ref re-evaluations are schema-tagged with
`metrics_schema_version=h_only_sref_v2` and
`metrics_provenance_generation=post_h_only_sref_prediction_safety`. Legacy
metric manifests that lack this provenance are treated as unknown/unsafe for
official winner claims; cross aggregation emits
`LEGACY_METRICS_SCHEMA_UNKNOWN_REEVALUATE_POST_H_ONLY` and related target/HSX
safety warnings. To intentionally replace old files, rerun
`evaluate_hamiltonian_metrics.py --overwrite` after confirming the predictions
were produced with H-only semantics.

For gamma/single-matrix runs:

```bash
python3 Comparison/scripts/evaluate_hamiltonian_metrics.py <result_dir> --overwrite
```

For non-gamma Monkhorst-Pack graphene runs:

```bash
python3 Comparison/scripts/evaluate_hamiltonian_metrics.py <result_dir> \
  --enable-kpoint-metrics \
  --overwrite
```

Do not pool old pre-H-only metric rows with post-H-only rows unless the report
explicitly labels the comparison as historical or exploratory.

Diagnostic structural metrics are written from the real SIESTA/Graph2Mat basis.
The evaluator requires archived `.ion.xml` files and counts PAOs from each
species basis using the angular degeneracy `2*l+1`:

- `metrics/block_metrics.csv`: errors grouped by `(row_atom, col_atom)`.
- `metrics/species_pair_metrics.csv`: errors grouped by species pair.
- `metrics/distance_bin_metrics.csv`: errors grouped into `0-1.2`,
  `1.2-2.0`, `2.0-4.0`, and `>4.0 Ang` bins.
- `metrics/orbital_pair_metrics.csv`: heatmap-ready Hamiltonian errors grouped
  by sample, row/column species, and row/column local orbital index.
- `metrics/orbital_pair_summary.csv`: cross-sample count/mean/std/min/max
  summaries for the orbital-pair metrics.

Orbital-pair rows report union-support `mae_union_eV`, `mae_union_meV`,
`mse_union_eV2`, `rmse_union_eV`, `r2_union`, `max_abs_error_union_eV`,
`mean_abs_ref_eV`, and `mean_signed_error_eV`. The target space is
`raw_global_hamiltonian_orbital_basis`: this is the repository Hamiltonian
basis, not DeepH's local-coordinate H' block representation. The `.ion.xml`
parser currently uses reliably available PAO counts only, so orbital labels are
stable generated labels such as `orbital_0`; species columns are part of the key
so local index `0` for different species is never merged.

For a DeepH-style orbital-orbital heatmap diagnostic, filter
`metrics/orbital_pair_metrics.csv` to one `sample` and `species_pair`, then
pivot `mae_union_meV` by `row_orbital_index` and `col_orbital_index`.
`metrics/orbital_pair_summary.csv` provides cross-sample summaries with the
same grouping keys. These files are discoverable in `metrics/manifest.json`
under `outputs.orbital_pair_metrics` and `outputs.orbital_pair_summary`, and
the UI lists their archive paths when present. They remain diagnostic only and
are not primary winner metrics.

If the basis is missing, cannot be parsed, lacks a species present in `RUN.fdf`,
or its orbital count does not match the Hamiltonian dimension, the metrics
manifest records `structural_basis_error` and the UI strict path aborts the
evaluation. These metrics are diagnostic in the current strict comparison; they
do not decide the winner.

Structural distance-bin metrics currently use direct Cartesian distances only.
For material metadata marked as `bulk`, `crystal`, `periodic`, `solid`,
`surface`, or `slab`, periodic minimum-image distance handling is treated as
unsupported: block/species structural metrics may still be written, but
distance-bin rows are omitted and the manifest records a severe
`unsupported_periodic_distance_bins` warning.

## Spectral Metrics

Eigenvalues are computed by diagonalizing the symmetrized Hamiltonian. When the
SIESTA overlap is available, the prediction is diagonalized using the SIESTA
reference overlap so both spectra are compared in the same generalized
eigenproblem.

For non-orthogonal matrices, `overlap_source` is `siesta_reference` and
`prediction_own_overlap_used` is `false`. A prediction-owned overlap is checked
only as a safety diagnostic for using the predicted HSX as a standalone
generalized-eigenproblem input. If it differs from `S_ref`, the evaluator emits
a severe `prediction_overlap_mismatch` warning, records
`prediction_self_contained_hsx_safe=false`, and still computes spectra with the
reference overlap. Known Graph2Mat auxiliary two-component prediction containers
for an unpolarized reference are allowed only as a severe diagnostic case:
component 0 is used as H, the auxiliary component is ignored for the main
metrics, and spectra still use `S_ref`.

The metrics manifest also includes `prediction_artifact_semantics`, a compact
summary of how many predicted HSX files are safe or unsafe as standalone
Hamiltonian+overlap containers. Official reports should use this manifest field
to avoid presenting unsafe `ML_prediction.HSX` artifacts as physically
self-contained generalized-eigenproblem inputs.

Reported metrics include:

- `global_mae_eV`, `global_rmse_eV`: all compared eigenvalues.
- `low_energy_mae_eV`, `low_energy_rmse_eV`,
  `low_energy_max_abs_error_eV`: errors for the lowest electronic eigenvalues
  after sorting both spectra ascending. By default the evaluator compares the
  first 10 states, or fewer when the matrices contain fewer common states.
- `low_energy_n_states`: the actual number of low-energy states compared.
- `occupied_mae_eV`, `occupied_rmse_eV`: only bands below the SIESTA Fermi level.
- `fermi_window_mae_eV`, `fermi_window_rmse_eV`: bands within +/- 2 eV of the
  SIESTA Fermi level.
- `gap_ref_eV`, `gap_pred_eV`, `gap_abs_error_eV`: gap metrics around the SIESTA
  Fermi level.
- `homo_error_eV`, `lumo_error_eV`, `frontier_window_rmse_eV`: frontier orbital
  diagnostics when a real SIESTA Fermi level identifies HOMO/LUMO levels.
- Alignment diagnostics are additionally reported and never overwrite raw errors:
  `align_global_shift_eV`, `align_global_mae_eV`, `align_global_rmse_eV`,
  `align_fermi_*`, and `align_homo_*`.

Low-energy metrics are computed from the same reference/predicted Hamiltonian
pair, but they answer a different question from matrix MAE/RMSE: whether the
lowest electronic eigenvalues themselves are reproduced. The prediction is
solved with the SIESTA reference overlap when overlap is available, so both
spectra use the same generalized eigenproblem `H c = E S c`. If the reference
Hamiltonian is non-orthogonal and no supported overlap can be read, the
low-energy metrics are left unavailable and `low_energy_warning` records the
reason. Metric-time validation treats an invalid or missing required overlap as
a fatal evaluator error for strict comparisons; the evaluator does not invent an
identity overlap.

Compatibility gates are intentionally fail-closed for unsupported scientific
cases. The default gamma/single-matrix path rejects mismatched matrix shapes,
non-orthogonal references without a readable overlap, unsupported
multi-component matrices, spin metadata outside the supported unpolarized path,
and non-gamma k-point sampled FDF inputs. Non-gamma Monkhorst-Pack runs require
the explicit `--enable-kpoint-metrics` opt-in path, which constructs H(k) and
uses `S_ref(k)` for spectra. Unsupported spin-orbit, ambiguous component
semantics, missing overlaps, and invalid reference/prediction shapes remain
fatal. Matrix-level sparse metrics remain material-agnostic for supported
single-matrix Hamiltonians with compatible reference/prediction shapes.

Configuration defaults are equivalent to:

```yaml
evaluation:
  spectral:
    low_energy:
      enabled: true
      n_states: 10
      alignment: none
```

`alignment: none` is the raw metric used for scientific comparison. The optional
`global_shift` diagnostic may add `low_energy_aligned_rmse_eV`, but it never
replaces `low_energy_rmse_eV`.

Important: A lower matrix MAE/RMSE does not necessarily imply a better
electronic spectrum. Spectral metrics must be evaluated explicitly, and matrix
level and spectral errors should both be inspected before drawing physical
conclusions.

Important: near-Fermi, occupied-band, gap, and fixed-window DOS metrics require
a real Fermi level read from the SIESTA reference file. The evaluator does not
estimate or infer a Fermi level. If SIESTA does not provide one, those metrics
are left unavailable, `fermi_level_source` is set to `unavailable`, and the
manifest records a nonfatal `missing_fermi_level` warning for that sample. This
warning still blocks winner claims whenever the unavailable Fermi-dependent
metric is the selected primary metric.

Metric-time compatibility gates require identical Hamiltonian shapes and a
single supported matrix component. Complex Hamiltonian values are preserved
rather than cast to real values. Obvious unsupported multi-component matrices
are rejected until spin/k/component-aware metrics are implemented.

## DOS Metrics

The evaluator builds a Gaussian-broadened total DOS from reference and predicted
eigenvalues using `sigma = 0.10 eV` and a 1000-point energy grid. It reports:

- `dos_wasserstein_eV`: 1D transport distance between normalized DOS curves.
- `dos_l1`: L1 distance between normalized DOS curves.
- `dos_l2`: L2 distance between normalized DOS curves.
- `dos_mae_500_fermi_window`: MAE between raw Gaussian-broadened predicted and
  reference DOS values on exactly 500 points from `fermi_level - 6 eV` to
  `fermi_level + 6 eV`, aligned to the SIESTA reference Fermi level and using
  the default `sigma = 0.10 eV`.

Sensitivity diagnostics are written to `metrics/dos_sigma_sweep.csv` for sigma
values `[0.05, 0.10, 0.20, 0.40] eV`. The purpose is to avoid conclusions driven
by one arbitrary broadening parameter.

The fixed Fermi-window DOS metric is a repository-compatible analogue of
DeepH's DOS MAE benchmark, not a claim of identical physical units or exact
DeepH equivalence for molecular or H2O-only systems. If the reference SIESTA
Fermi level is missing or non-finite, the metric is written as unavailable
(`NaN`) with `dos_window_unavailable_reason = missing_fermi_level`; the
evaluator does not estimate Fermi from HOMO/LUMO.

## Matrix-Spectrum Relationship

`metrics/matrix_spectrum_relationship.csv` joins sparse and spectral rows by
sample. This is used to test whether matrix improvements translate into
physically meaningful spectral improvements.

Per sample it records:

- `relative_frobenius_union`
- `mae_ref_eV`, `rmse_ref_eV`, `mse_ref_eV2`, `r2_ref`
- `rmse_union_eV`, `mse_union_eV2`, `r2_union`
- `support_f1`
- `global_rmse_eV`
- `low_energy_rmse_eV`
- `fermi_window_rmse_eV` when a real SIESTA Fermi level exists
- `gap_abs_error_eV` when a real SIESTA Fermi level exists

The manifest summary includes both Pearson and Spearman correlations for matrix
error versus spectral errors. Spearman is the rank-robust companion and should
be preferred whenever heavy-tailed outliers are visible.

## Interpreting Method Comparisons

The safest comparison is:

- each selected method prediction vs its SIESTA reference;
- then compare the resulting metric distributions on the same frozen
  SIESTA-referenced test sets;
- Keep the chosen budget mode explicit:
  `equal_sample_count`, `equal_siesta_budget`, or `both`.

Matrix error alone should not be treated as the final answer. A lower matrix
MAE can still fail to improve frontier eigenvalues, gap, or DOS, so the report
keeps matrix, spectral, near-Fermi, DOS, and matrix-spectrum relationship
metrics separate.

The DeepH-comparable additions (`mse_*_eV2`, `r2_*`, meV MAE/RMSE aliases,
`dos_mae_500_fermi_window`, and the orbital-pair CSV diagnostics) are
repository-compatible analogues on the archived Hamiltonian/DOS space. Scalar
matrix and DOS additions flow into cross-evaluation reports when present.
Orbital-pair CSVs are higher-cardinality auxiliary diagnostics and are listed
as output artifacts rather than folded into `cross_evaluation_metrics.csv`.
Matrix MSE/R2/meV aliases are diagnostic or secondary winner metrics, while the
fixed-window DOS MAE remains Fermi-dependent and must not be substituted when
unavailable.

For physical conclusions the UI/winner analysis uses the selected primary metric
authoritatively and does not silently substitute another metric when the primary
metric is missing. The recommended primary order is:

1. `fermi_window_rmse_eV`
2. `occupied_rmse_eV`
3. `relative_frobenius_union`
4. `dos_wasserstein_eV`
5. `global_rmse_eV` only as a secondary/report metric

Winner outputs preserve `experiment_id`, `seed`, dataset sizes, test set,
metric, frozen-test hash, checkpoint manifest, and checkpoint hash. `pooled`
aggregation is only produced when explicitly requested with
`--aggregation-mode pooled`. Fewer than three seeds are reported as
`scientific_status: exploratory`; `robust_comparison` requires enough seeds, a
complete cross-evaluation grid for the selected methods and frozen test sets, a
complete primary metric, and no severe warnings.

Wins only on a method's own frozen test set, for example only on
`test_siesta_fc_cartesian` or only on `test_random_cartesian`, are treated as
distribution-specific diagnostics, not cross-generalization winners. A
conservative winner should hold on `test_md` and/or `test_mixed` with the same
frozen references, enough seeds, a complete primary metric, and no severe
warnings.

Winner recommendations are marked `inconclusive` when machine-readable
validation warnings are present in the aggregated cross metrics, including
SIESTA settings mismatch, Graph2Mat model/config mismatch, or a severe budget
mismatch. In strict UI comparison mode, SIESTA/model mismatches and geometry
leakage abort before winner analysis.

Hamiltonian reference output controls are strict SIESTA settings for this
benchmark: mismatches in `Save.HS`, `TS.HS.Save`, `TS.DE.Save`, or `XML.Write`
are treated as severe because they decide whether comparable Hamiltonian,
overlap, and XML reference artifacts are generated.

`timing_breakdown.json` is written for new UI runs and includes the required
phase keys. Some legacy Graph2Mat/SIESTA entrypoints still expose only coarse
process timings, so missing phase values are explicit `null` values rather than
invented numbers.

## Validation and Leakage Diagnostics

Before a sample is considered scientifically valid, strict validation requires:

- `RUN.fdf`
- a non-predicted Hamiltonian reference following the shared strict reference
  policy above
- `RUN.out`
- SIESTA `Job completed`
- `SCF cycle converged`

Validation artifacts are `valid_samples.csv`, `invalid_samples.csv` and
`validation_summary.json`. Typical invalid reasons include `missing_run_fdf`,
`missing_matrix`, `missing_output`, `job_not_completed`,
`scf_not_converged`, `ambiguous_reference_matrix`, and `parser_error`.

`Comparison/scripts/check_geometry_leakage.py` reports exact duplicates,
near-duplicate geometries, neighboring MD frames crossing splits, and
AtomDisplacement displacement-family leakage. In addition to raw-coordinate
checks, it records a translation/rotation-invariant internal-distance signature
by species and atom order. It still does not solve arbitrary atom permutation or
full PBC equivalence.

For strict one-click comparisons, geometry leakage diagnostics are run after
common frozen tests are built and before cross predictions. Any detected leakage
invalidates that comparison rather than being hidden in a plot.
