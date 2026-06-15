# Dataset Size Minimum Analysis

Postprocesado de solo lectura. No se ha ejecutado entrenamiento, inferencia, SIESTA, Graph2Mat ni DeepH.

## Inputs

- `/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/results/graphene_w90_snapshot_scaling_reuse_10_1000_4seeds/graphene_w90_snapshot_scaling_reuse_10_1000_4seeds_20260610_122311`
- `/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/results/graphene_w90_snapshot_scaling_1100_1300_4seeds_followup/graphene_w90_snapshot_scaling_1100_1300_4seeds_followup_20260612_104920`

## Configuracion

- Primary metric: `h_mae_eV_mean` convertido a meV
- Threshold absoluto: `10` meV
- Threshold policy: basis `explicit_threshold_publication_protocol`, reference `graphene_dataset_size_minimum_paper_protocol_v1`, metric_family `hamiltonian_element_error_mev`, user_defined=False
- Threshold interpretation: Explicit threshold publication protocol supplied by the user. Paper-candidate use still depends on the recorded sensitivity audit.
- Threshold protocol file: `/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/config/graphene_dataset_size_minimum_paper_threshold_protocol.json`
- Threshold physical rationale: Paper-audit Hamiltonian MAE threshold for the locked Graphene W90 Graph2Mat-vs-DeepH scaling protocol. This threshold is documented explicitly for publication-facing N_min analysis and is not an exploratory UI preset.
- Threshold applicability: `["h_mae_eV_mean"]`
- Threshold sensitivity recommendation: Audit the lower, main, and upper Hamiltonian MAE thresholds before treating nominal N_min as a paper-level claim.
- 20 meV is not universal; threshold presets are metric-specific and exploratory unless explicitly justified.
- Eje x: `n_train`
- Cost basis for `N_min_cost_eff`: `protocol_total` (protocol total GPU-hours across required seeds/replicates)
- Power-law paper-candidate gate: at least `5` observed dataset sizes per method.

## Cobertura

- Grupos config agregados: 544
- Mejores metodo/tamano: 136
- Metodos: deeph, graph2mat
- Tamanos: 6, 14, 19, 22, 30, 38, 43, 46, 54, 59, 62, 70, 75, 78, 94, 118, 139, 142, 158, 174, 198, 222, 238, 278, 318, 398, 478, 558, 638, 718, 798, 878, 958, 1038

## N_min por metodo

| Metodo | Best observado meV | N_min_abs | N_min_rel_tol | N_min_plateau | N_min_cost_eff |
|---|---:|---:|---:|---:|---:|
| deeph | 2.747 | - | 907 | 901 | 1038 |
| graph2mat | 12.077 | - | 887 | 880 | 1038 |

## Fits


### deeph
| Modelo | Estado | Politica | RMSE meV | SSE | alpha | alpha search | evals | R2 | Coeficientes |
|---|---|---|---:|---:|---:|---|---:|---:|---|
| linear | ok | diagnostic_only | 20.254 | - | - | - | - | 0.463 | `[55.89807864815743, -0.06298987686006134]` |
| quadratic | ok | diagnostic_only | 18.256 | - | - | - | - | 0.564 | `[65.85655339412065, -0.17001343388561677, 0.00011664470870132048]` |
| inverse | ok | diagnostic_only | 21.952 | - | - | - | - | 0.369 | `[28.973492712002386, 552.8398607452867]` |
| inverse_square | ok | diagnostic_only | 25.475 | - | - | - | - | 0.151 | `[36.09838774307996, 2271.312284691196]` |
| power_law_floor | ok | paper_candidate | 17.216 | 20154.662 | 0.361 | coarse_grid_plus_golden_section | 176 | 0.612 | `[0.0, 206.43842036716487, 0.361054293627424]` |

### graph2mat
| Modelo | Estado | Politica | RMSE meV | SSE | alpha | alpha search | evals | R2 | Coeficientes |
|---|---|---|---:|---:|---:|---|---:|---:|---|
| linear | ok | diagnostic_only | 13.610 | - | - | - | - | 0.673 | `[61.16587982604503, -0.06531345587241358]` |
| quadratic | ok | diagnostic_only | 7.428 | - | - | - | - | 0.902 | `[74.11272692275573, -0.20445299921640192, 0.00015164784233735428]` |
| inverse | ok | diagnostic_only | 18.969 | - | - | - | - | 0.364 | `[35.06873791823008, 472.17582121175747]` |
| inverse_square | ok | diagnostic_only | 22.308 | - | - | - | - | 0.120 | `[41.39614403975614, 1745.9529183429795]` |
| power_law_floor | ok | paper_candidate | 11.137 | 8434.041 | 0.309 | coarse_grid_plus_golden_section | 176 | 0.781 | `[0.0, 185.03679092758725, 0.3092942488508934]` |

## Replicate-resampling CI

- Label: replicate-resampling CI
- Enabled: True
- Scope: row-level replicate/seed resampling within `(method, dataset_size_x)`; not temporal/block bootstrap and not full scientific uncertainty.
- Limitations:
  - does not model temporal autocorrelation
  - does not model model-selection uncertainty
  - does not model hyperparameter-selection uncertainty
  - does not model dependence between dataset sizes
  - N_min_cost_eff has no replicate-resampling CI in this protocol
  - N_min_cost_eff is excluded from replicate-resampling CI because this diagnostic does not jointly resample cost and metric under the selected cost_basis.
- Replicate resampling warnings: `["replicate_bootstrap_row_level_replicates_only", "replicate_bootstrap_no_temporal_or_block_bootstrap", "replicate_bootstrap_does_not_capture_model_selection_uncertainty", "replicate_bootstrap_does_not_capture_hyperparameter_selection_uncertainty", "replicate_bootstrap_does_not_capture_dependence_between_dataset_sizes", "replicate_bootstrap_excludes_n_min_cost_eff", "replicate_bootstrap_excludes_n_min_cost_eff"]`
- N_min_cost_eff CI available: False
- N_min_cost_eff CI policy: `excluded_no_joint_metric_cost_resampling`

## Hierarchical uncertainty

- Label: hierarchical uncertainty (paper-readiness audit)
- Status: `diagnostic_only`
- Paper-ready: False
- This layer separates seed variability, config/hyperparameter selection variability, block/trajectory temporal variability, fit/model-selection variability, and dataset-size dependence.
- Level `block`: available=True, sufficient=False, blockers=`["paper_uncertainty_block_hierarchy_incomplete"]`
- Level `config`: available=True, sufficient=True, blockers=`[]`
- Level `dataset_size_dependence`: available=True, sufficient=True, blockers=`[]`
- Level `fit_model`: available=True, sufficient=True, blockers=`[]`
- Level `seed`: available=True, sufficient=True, blockers=`[]`
- Hierarchical uncertainty blockers: `["paper_uncertainty_block_hierarchy_incomplete"]`

## Fit stability (leave-one-size-out)

- Status: `ok`
- Method `deeph`: trials=34, successful=34, failed=0, unstable_criteria=`[]`
- Method `graph2mat`: trials=34, successful=34, failed=0, unstable_criteria=`[]`

## Threshold sensitivity

- Status: `ok`
- Thresholds audited (meV): `[8.0, 10.0, 12.0]`
- Method `deeph`: span=-, allowed_delta=3, unstable=False
  series: `[{"threshold_mev": 8.0, "N_min_abs": null, "N_min_rel_tol": 907, "N_min_plateau": 901, "source": "fit"}, {"threshold_mev": 10.0, "N_min_abs": null, "N_min_rel_tol": 907, "N_min_plateau": 901, "source": "fit"}, {"threshold_mev": 12.0, "N_min_abs": null, "N_min_rel_tol": 907, "N_min_plateau": 901, "source": "fit"}]`
  blockers: `["paper_blocked_if_threshold_sensitivity_missing_n_min_abs:deeph"]`
- Method `graph2mat`: span=-, allowed_delta=3, unstable=False
  series: `[{"threshold_mev": 8.0, "N_min_abs": null, "N_min_rel_tol": 887, "N_min_plateau": 880, "source": "fit"}, {"threshold_mev": 10.0, "N_min_abs": null, "N_min_rel_tol": 887, "N_min_plateau": 880, "source": "fit"}, {"threshold_mev": 12.0, "N_min_abs": null, "N_min_rel_tol": 887, "N_min_plateau": 880, "source": "fit"}]`
  blockers: `["paper_blocked_if_threshold_sensitivity_missing_n_min_abs:graph2mat"]`

## Temporal diagnostics (MD snapshot independence)

- Status: Estimated N_eff range: {"min": 5.81805340162577, "median": 43.50843638373937, "max": 311.53863042186526} (N_eff = N / statistical_inefficiency; N_min still uses nominal N)
- Nominal N_train (metadata): 1038
- Estimated N_eff_train: `{"min": 5.81805340162577, "median": 43.50843638373937, "max": 311.53863042186526}`
- Autocorrelation diagnostic available: True

| Dataset size (nominal N_train) | N_eff diagnostic | N_eff/N_nominal | Autocorrelation available | Blocks/datasets |
|---:|---:|---:|---|---|
| 6 | - | - | no | 1 dataset(s), 2 block entry(ies) |
| 14 | 5.818 | 0.416 | yes | 1 dataset(s), 2 block entry(ies) |
| 19 | 6.961 | 0.366 | yes | 1 dataset(s), 2 block entry(ies) |
| 22 | 8.164 | 0.371 | yes | 1 dataset(s), 2 block entry(ies) |
| 30 | 10.376 | 0.346 | yes | 1 dataset(s), 2 block entry(ies) |
| 38 | 12.551 | 0.330 | yes | 1 dataset(s), 2 block entry(ies) |
| 43 | 14.028 | 0.326 | yes | 1 dataset(s), 2 block entry(ies) |
| 46 | 14.678 | 0.319 | yes | 1 dataset(s), 2 block entry(ies) |
| 54 | 16.974 | 0.314 | yes | 1 dataset(s), 2 block entry(ies) |
| 59 | 19.074 | 0.323 | yes | 1 dataset(s), 2 block entry(ies) |
| 62 | 19.879 | 0.321 | yes | 1 dataset(s), 2 block entry(ies) |
| 70 | 21.880 | 0.313 | yes | 1 dataset(s), 2 block entry(ies) |
| 75 | 23.343 | 0.311 | yes | 1 dataset(s), 2 block entry(ies) |
| 78 | 24.594 | 0.315 | yes | 1 dataset(s), 2 block entry(ies) |
| 94 | 29.271 | 0.311 | yes | 1 dataset(s), 2 block entry(ies) |
| 118 | 36.063 | 0.306 | yes | 1 dataset(s), 2 block entry(ies) |
| 139 | 42.316 | 0.304 | yes | 1 dataset(s), 2 block entry(ies) |
| 142 | 43.508 | 0.306 | yes | 1 dataset(s), 2 block entry(ies) |
| 158 | 48.758 | 0.309 | yes | 1 dataset(s), 2 block entry(ies) |
| 174 | 53.244 | 0.306 | yes | 1 dataset(s), 2 block entry(ies) |
| 198 | 60.094 | 0.304 | yes | 1 dataset(s), 2 block entry(ies) |
| 222 | 67.469 | 0.304 | yes | 1 dataset(s), 2 block entry(ies) |
| 238 | 72.296 | 0.304 | yes | 1 dataset(s), 2 block entry(ies) |
| 278 | 84.738 | 0.305 | yes | 1 dataset(s), 2 block entry(ies) |
| 318 | 96.208 | 0.303 | yes | 1 dataset(s), 2 block entry(ies) |
| 398 | 120.186 | 0.302 | yes | 1 dataset(s), 2 block entry(ies) |
| 478 | 143.892 | 0.301 | yes | 1 dataset(s), 2 block entry(ies) |
| 558 | 168.116 | 0.301 | yes | 1 dataset(s), 2 block entry(ies) |
| 638 | 191.950 | 0.301 | yes | 1 dataset(s), 2 block entry(ies) |
| 718 | 215.913 | 0.301 | yes | 1 dataset(s), 2 block entry(ies) |
| 798 | 239.642 | 0.300 | yes | 1 dataset(s), 2 block entry(ies) |
| 878 | 263.470 | 0.300 | yes | 1 dataset(s), 2 block entry(ies) |
| 958 | 287.685 | 0.300 | yes | 1 dataset(s), 2 block entry(ies) |
| 1038 | 311.539 | 0.300 | yes | 1 dataset(s), 2 block entry(ies) |
- N_min basis: `nominal`
- Claim mode: requested `paper_candidate` -> `diagnostic`
- Scientific claim status: `diagnostic_only`
- N_min protocol: source `fit` -> `fit`, fit `power_law_floor` -> `power_law_floor`, aggregation `mean_seeds_per_config` -> `mean_seeds_per_config`
- Aggregation mode policy: `paper_candidate` (`paper_ready_seed_mean_per_config`)
- N_min fit policy: `diagnostic_only`
- Paper-level blockers: `["paper_blocked_if_autocorrelation_grouping_mixed_temperatures", "paper_blocked_if_autocorrelation_scalar_series_unavailable", "paper_blocked_if_n_eff_by_dataset_size_incomplete", "paper_blocked_if_n_eff_much_smaller_than_nominal", "paper_blocked_if_temporal_gap_le_1", "paper_blocked_if_threshold_sensitivity_missing_n_min_abs:deeph", "paper_blocked_if_threshold_sensitivity_missing_n_min_abs:graph2mat", "paper_uncertainty_block_hierarchy_incomplete"]`
- N_eff / N_nominal: 0.042
- N_min_effective_diagnostic: `{"deeph": {"N_min_abs": null, "N_min_rel_tol": null, "N_min_plateau": null, "N_min_cost_eff": 311.53863042186526}, "graph2mat": {"N_min_abs": null, "N_min_rel_tol": null, "N_min_plateau": null, "N_min_cost_eff": 311.53863042186526}}`
- Effective samples at nominal N_min are diagnostic only; true effective-N thresholding is not implemented.
- Dataset `joint_graph2mat_deeph_388c061cff2dc93a`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_e0c5ec6729b2ae47`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_89b954cc51269556`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_7308e65b5bb16dd5`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_d8fa4edf62cefc43`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_75ff8acad293fdb5`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_04b2dba22684600f`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_3b9bddd25bddef44`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_a1d80aa1a5b8804b`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_e203814eb3f30888`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_5ef5cbecd6604297`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_ba252c5447bcae03`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_e738449d8c357184`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_400da82939a6a553`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_4a1c835d3473d1ec`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_7d215344130cb740`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_ba2d1a1b2fdc8442`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_e549d837a0089531`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_305a8177d7677270`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_153c64920bab291b`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_d05cd3de7d92f383`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_3fa2028c1758c996`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_0715fccd0f55d320`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_c67eb56ce1b760e6`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_a3929e822c712fb0`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_0d20809fb1af2764`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_998da195e51aa1b6`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_ceced8c3b19a44be`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_9f798a08c12538c0`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_2c10d843ef2f89f1`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_6debbfd8cd28d88d`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_620d2bda2ab33223`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_51b29617399dcd22`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True
- Dataset `joint_graph2mat_deeph_e9e80796d0c4e73d`: blocks=3, strategy=blocked_with_gap, temporal_gap=1, blocked_split=True

N_min uses nominal N. If MD snapshots are autocorrelated, independent sample count can be lower. Check N_eff before using this as a paper-level claim. Effective-N values are diagnostics only and do not replace nominal N_min without a stronger validated protocol.


## Warnings / blockers

- autocorrelation_grouping_invalid_mixed_temperatures
- autocorrelation_unavailable_mixed_temperatures
- autocorrelation_unavailable_no_cheap_scalar_series
- block_uncertainty_requires_multiple_temporal_blocks_per_dataset
- n_eff_much_smaller_than_nominal
- paper_blocked_if_autocorrelation_grouping_mixed_temperatures
- paper_blocked_if_autocorrelation_scalar_series_unavailable
- paper_blocked_if_n_eff_by_dataset_size_incomplete
- paper_blocked_if_n_eff_much_smaller_than_nominal
- paper_blocked_if_temporal_gap_le_1
- paper_blocked_if_threshold_sensitivity_missing_n_min_abs:deeph
- paper_blocked_if_threshold_sensitivity_missing_n_min_abs:graph2mat
- paper_uncertainty_block_hierarchy_incomplete
- replicate_bootstrap_does_not_capture_dependence_between_dataset_sizes
- replicate_bootstrap_does_not_capture_hyperparameter_selection_uncertainty
- replicate_bootstrap_does_not_capture_model_selection_uncertainty
- replicate_bootstrap_excludes_n_min_cost_eff
- replicate_bootstrap_no_temporal_or_block_bootstrap
- replicate_bootstrap_row_level_replicates_only
- temporal_gap_le_1_adjacent_frames_may_leak

## Outputs

- `.test_tmp/paper_audit_check/dataset_size_minimum_results.csv`
- `.test_tmp/paper_audit_check/dataset_size_minimum_summary.json`
- `.test_tmp/paper_audit_check/dataset_size_minimum_report.md`
