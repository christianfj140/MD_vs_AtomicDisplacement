# Dossier 4A — Claims, rankings y gates

## Objeto de revisión

Auditar qué claims sobreviven a los datos: seeds, intervalos, checkpoint selection, test blindness, leakage temporal, agregación, umbrales y gates.

## Condiciones del contexto

- Contexto ciego: no contiene informes de auditoría previos ni prompts históricos.
- Las líneas prefijadas `NNNNN |` conservan la numeración del archivo fuente.
- Los extractos Python omiten funciones operativas sin semántica científica directa.
- Ausencia de un archivo o función en este dossier significa `no evaluable`, no `correcto`.

## `docs/phase6_hamiltonian_architecture_benchmark.md`

SHA-256: `9c797473808d12c9459dc22900c1c14eea3e8d954c322158046552e78ba68a00`

```md
00001 | # Phase 6 H2O Hamiltonian Architecture Benchmark
00002 | 
00003 | This benchmark compares the corrected H-only Graph2Mat baseline against
00004 | Hamiltonian-specific architecture options exposed by an editable Graph2Mat
00005 | checkout.
00006 | 
00007 | Related documents:
00008 | 
00009 | - `README.md` for the current repository scope and common validation commands.
00010 | - `docs/workflows.md` for the main comparison UI flow.
00011 | - `docs/graph2mat_deeph_benchmark.md` for the stricter joint benchmark rules.
00012 | 
00013 | ## Payload
00014 | 
00015 | Use:
00016 | 
00017 | ```text
00018 | Comparison/config/h2o_phase6_hamiltonian_architecture_benchmark_payload.json
00019 | ```
00020 | 
00021 | The payload is intentionally `train_test_metrics_plots_only` and reuses the
00022 | archived MD dataset/splits through:
00023 | 
00024 | ```text
00025 | reusable_dataset_ids: ["98c89bd85937f6a2"]
00026 | reusable_split_policy: preserve_archived_splits
00027 | selected_methods: ["md"]
00028 | ```
00029 | 
00030 | Replace the dataset id only if you intentionally want a different archived
00031 | dataset. The benchmark should not regenerate references and must not use
00032 | `ML_prediction.HSX` as ground truth.
00033 | 
00034 | ## Matrix
00035 | 
00036 | The implemented matrix has three seeds for each candidate:
00037 | 
00038 | - `baseline_default_mae`
00039 | - `baseline_default_huber_b0p01`
00040 | - `baseline_default_mse`
00041 | - `context_default_huber_b0p01`
00042 | - `context_hamiltonian_readout_huber_b0p01`
00043 | - `context_hamiltonian_readout_staged_composite`
00044 | - `diagnostic_dense_mse`
00045 | 
00046 | The dense readout entry is diagnostic only and must not be ranked as a
00047 | production result.
00048 | 
00049 | ## Scientific Guardrails
00050 | 
00051 | Every plan entry sets:
00052 | 
00053 | ```yaml
00054 | out_matrix: hamiltonian
00055 | matrix_component_policy: h_only
00056 | n_matrix_components: 1
00057 | symmetric_matrix: true
00058 | ```
00059 | 
00060 | The archive/evaluator path uses the strict reference selector. It accepts real
00061 | SIESTA `.TSHS` / `.HSX` references, rejects `ML_prediction.HSX`, and records the
00062 | reference policy in each manifest.
00063 | 
00064 | Spectral and DOS metrics must be interpreted only when the metrics manifest
00065 | records the post-H-only/S_ref provenance, including target policy, component
00066 | counts, overlap source, and prediction-HSX safety fields. Phase-6 rows lacking
00067 | that provenance are legacy or unknown and should be regenerated with
00068 | `Comparison/scripts/evaluate_hamiltonian_metrics.py` before they are used for a
00069 | winner claim.
00070 | 
00071 | The matrix metrics in this benchmark are repository raw-global-H diagnostics.
00072 | They are useful for internal Graph2Mat comparisons, but they are not exact
00073 | DeepH H-prime local-frame block metrics unless a future validated H-prime
00074 | transform is added.
00075 | 
00076 | ## Manifests
00077 | 
00078 | Per-run manifests now record benchmark metadata:
00079 | 
00080 | - pipeline and Graph2Mat git commits
00081 | - architecture/readout/context fields
00082 | - loss and loss kwargs
00083 | - staged training metadata
00084 | - seed
00085 | - H-only target policy
00086 | - reference policy and evaluation manifest path
00087 | 
00088 | ## Running
00089 | 
00090 | Load the JSON payload in the UI or POST it to the existing experiment endpoint.
00091 | Run a one-epoch smoke first by copying the payload and changing every
00092 | `max_epochs` to `1`.
00093 | 
00094 | Full benchmark cost is high: 21 trainings on the MD 1140-snapshot dataset.
```

## `docs/derivative_smoke_validation_note.md`

SHA-256: `434fc80d16cbaaf1e2342edcf17806a46b6057d5a39bb15f95f1622cfa8ee607`

```md
00001 | # Derivative Smoke Validation Note
00002 | 
00003 | - Repo root: `/home/christian/repositorios/MD_vs_AtomicDisplacement`
00004 | - Graph2Mat smoke root: `Comparison/results/derivative_smoke/graph2mat_derivative_result`
00005 | - DeepH smoke root: `Comparison/results/derivative_smoke/deeph_derivative_result`
00006 | 
00007 | This note records a completed local derivative smoke for Graph2Mat and a completed local DeepH derivative smoke using the same 5-sample stencil/reference set. It validates workflow plumbing and UI/backend integration only. It does not claim paper-ready derivative results, and it does not claim any scientific winner.
00008 | 
00009 | ## Smoke Scope
00010 | 
00011 | - Stencil samples: `5`
00012 | - Sample ids:
00013 |   - `md_270_base`
00014 |   - `md_270_atom0000_x_d0.005_plus`
00015 |   - `md_270_atom0000_x_d0.005_minus`
00016 |   - `md_270_atom0000_x_d0.01_plus`
00017 |   - `md_270_atom0000_x_d0.01_minus`
00018 | - SIESTA reference status: `5/5 ok`
00019 | 
00020 | ## Graph2Mat Smoke
00021 | 
00022 | - Checkpoint path: `Comparison/results/derivative_smoke/graph2mat_derivative_result/graph2mat/training/lightning_logs/my_first_model/version_0/checkpoints/best-22.ckpt`
00023 | - Prediction status: `5/5 predicted`
00024 | - Local smoke outputs include:
00025 |   - `derivative_metrics/graph2mat/manifest.json`
00026 |   - `derivative_metrics/graph2mat/derivative_matrix_metrics.csv`
00027 |   - `derivative_metrics/graph2mat/derivative_delta_stability.json`
00028 |   - `derivative_metrics/graph2mat/summary/derivative_gate_report.json`
00029 |   - `derivative_metrics/graph2mat/summary/derivative_plot_payload.json`
00030 |   - `derivative_metrics/graph2mat/summary/derivative_plot_manifest.json`
00031 |   - `derivative_artifact_validation.json`
00032 | 
00033 | ## DeepH Smoke
00034 | 
00035 | - Runtime note: sibling `DeepH-pack` source override via `PYTHONPATH=/home/christian/repositorios/DeepH-pack`
00036 | - Reused compatible DeepH model path: `Comparison/results/graphene_w90_snapshot_scaling_150_1000_dense/graphene_w90_snapshot_scaling_150_1000_dense_20260609_132720/sweep/deeph/graphene_w90_scale_iid150/DH-T1000-04-anchor-N150/deeph/train`
00037 | - DeepH prediction status: `5/5 predicted`
00038 | - DeepH derivative metrics path: `Comparison/results/derivative_smoke/deeph_derivative_result/derivative_metrics/deeph/derivative_matrix_metrics.csv`
00039 | - Local smoke outputs also include:
00040 |   - `Comparison/results/derivative_smoke/deeph_derivative_result/derivative_metrics/summary/derivative_gate_report.json`
00041 |   - `Comparison/results/derivative_smoke/deeph_derivative_result/derivative_metrics/summary/derivative_plot_payload.json`
00042 |   - `Comparison/results/derivative_smoke/deeph_derivative_result/derivative_metrics/summary/derivative_plot_manifest.json`
00043 |   - `Comparison/results/derivative_smoke/deeph_derivative_result/derivative_artifact_validation.json`
00044 | 
00045 | ## Paired Comparison And UI
00046 | 
00047 | - Paired Graph2Mat-vs-DeepH comparison status: available
00048 | - Paired comparison count: `2`
00049 | - UI/backend run id: `deeph_derivative_result`
00050 | - UI endpoint verification completed for:
00051 |   - `/api/g2m-deeph/plot-runs`
00052 |   - `/api/g2m-deeph/derivative-metrics?run_id=deeph_derivative_result`
00053 | - Verified UI/backend sections:
00054 |   - derivative status summary
00055 |   - gate report
00056 |   - warning/blocker table
00057 |   - artifact links
00058 |   - derivative plots
00059 |   - paired comparison plot
00060 | 
00061 | ## Gate Interpretation
00062 | 
00063 | - Gate status: `blocked`
00064 | - `blocked` is the expected outcome for this smoke. The workflow completed, produced the expected derivative diagnostics, and preserved conservative scientific gating instead of overclaiming.
00065 | - DeepH equivalence remains diagnostic-only/unproven against raw/global HSX ordering/sign/unit conventions.
00066 | 
00067 | ## Release Framing
00068 | 
00069 | - This validates workflow plumbing and integration only, not paper-level derivative science.
00070 | - Do not treat this smoke as paper-ready derivative evidence.
00071 | - Do not treat this smoke as a scientific winner decision.
00072 | - The paths above refer to local smoke outputs from the validation run and are documented here as runtime evidence, not as committed release artifacts.
```

## `Comparison/config/h2o_phase6_hamiltonian_architecture_benchmark_payload.json`

SHA-256: `455c99fab0d39c0de0c25a17929d77d44282892380d20dc47266214be056b204`

```json
00001 | {
00002 |   "material": {
00003 |     "mode": "preset",
00004 |     "preset": "h2o"
00005 |   },
00006 |   "selected_methods": [
00007 |     "md"
00008 |   ],
00009 |   "run_mode": "train_test_metrics_plots_only",
00010 |   "reusable_dataset_ids": [
00011 |     "98c89bd85937f6a2"
00012 |   ],
00013 |   "reusable_split_policy": "preserve_archived_splits",
00014 |   "splits": {
00015 |     "train": 0.8,
00016 |     "validation": 0.1,
00017 |     "test": 0.1
00018 |   },
00019 |   "split_mode": "blocked_with_gap",
00020 |   "test_sets": [
00021 |     "test_md"
00022 |   ],
00023 |   "primary_metric": "mae_ref_meV",
00024 |   "compute_budget_mode": "equal_sample_count",
00025 |   "compute_accelerator": "gpu",
00026 |   "performance": {
00027 |     "preset": "aggressive",
00028 |     "max_parallel_dataset_jobs": 1,
00029 |     "max_parallel_prediction_jobs": 1,
00030 |     "max_parallel_evaluation_jobs": 8,
00031 |     "max_parallel_metric_jobs": 24,
00032 |     "max_parallel_siesta_jobs": 8,
00033 |     "omp_num_threads": 3,
00034 |     "mkl_num_threads": 3,
00035 |     "openblas_num_threads": 3,
00036 |     "numexpr_num_threads": 24,
00037 |     "torch_num_threads": 12,
00038 |     "compute_accelerator": "gpu",
00039 |     "batch_size": 256,
00040 |     "store_in_memory": true,
00041 |     "reuse_validated_siesta_outputs": true,
00042 |     "enable_experiment_cache": false,
00043 |     "error_policy": "fail_fast",
00044 |     "torch_float32_matmul_precision": "high"
00045 |   },
00046 |   "training_plan": [
00047 |     {
00048 |       "label": "phase6_baseline_default_mae_seed0",
00049 |       "display_label": "Phase 6 baseline default readout MAE seed 0",
00050 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00051 |       "training_settings": {
00052 |         "seed_everything": 0,
00053 |         "max_epochs": 550,
00054 |         "optim_lr": 0.005,
00055 |         "batch_size": 96,
00056 |         "loader_threads": 4,
00057 |         "num_interactions": 3,
00058 |         "correlation": 2,
00059 |         "max_ell": 3,
00060 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00061 |         "loss": "graph2mat.core.data.metrics.block_type_mae",
00062 |         "loss_kwargs": {},
00063 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00064 |         "model": {"readout": "default"},
00065 |         "benchmark_metadata": {"benchmark_method_id": "baseline_default_mae", "architecture": "default", "readout": "default", "context_enabled": false}
00066 |       }
00067 |     },
00068 |     {
00069 |       "label": "phase6_baseline_default_mae_seed1",
00070 |       "display_label": "Phase 6 baseline default readout MAE seed 1",
00071 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00072 |       "training_settings": {
00073 |         "seed_everything": 1,
00074 |         "max_epochs": 550,
00075 |         "optim_lr": 0.005,
00076 |         "batch_size": 96,
00077 |         "loader_threads": 4,
00078 |         "num_interactions": 3,
00079 |         "correlation": 2,
00080 |         "max_ell": 3,
00081 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00082 |         "loss": "graph2mat.core.data.metrics.block_type_mae",
00083 |         "loss_kwargs": {},
00084 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00085 |         "model": {"readout": "default"},
00086 |         "benchmark_metadata": {"benchmark_method_id": "baseline_default_mae", "architecture": "default", "readout": "default", "context_enabled": false}
00087 |       }
00088 |     },
00089 |     {
00090 |       "label": "phase6_baseline_default_mae_seed2",
00091 |       "display_label": "Phase 6 baseline default readout MAE seed 2",
00092 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00093 |       "training_settings": {
00094 |         "seed_everything": 2,
00095 |         "max_epochs": 550,
00096 |         "optim_lr": 0.005,
00097 |         "batch_size": 96,
00098 |         "loader_threads": 4,
00099 |         "num_interactions": 3,
00100 |         "correlation": 2,
00101 |         "max_ell": 3,
00102 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00103 |         "loss": "graph2mat.core.data.metrics.block_type_mae",
00104 |         "loss_kwargs": {},
00105 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00106 |         "model": {"readout": "default"},
00107 |         "benchmark_metadata": {"benchmark_method_id": "baseline_default_mae", "architecture": "default", "readout": "default", "context_enabled": false}
00108 |       }
00109 |     },
00110 |     {
00111 |       "label": "phase6_baseline_default_huber_b0p01_seed0",
00112 |       "display_label": "Phase 6 baseline default readout Huber beta 0.01 seed 0",
00113 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00114 |       "training_settings": {
00115 |         "seed_everything": 0,
00116 |         "max_epochs": 550,
00117 |         "optim_lr": 0.005,
00118 |         "batch_size": 96,
00119 |         "loader_threads": 4,
00120 |         "num_interactions": 3,
00121 |         "correlation": 2,
00122 |         "max_ell": 3,
00123 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00124 |         "loss": "graph2mat.core.data.metrics.block_type_huber",
00125 |         "loss_kwargs": {"beta": 0.01},
00126 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00127 |         "model": {"readout": "default"},
00128 |         "benchmark_metadata": {"benchmark_method_id": "baseline_default_huber_b0p01", "architecture": "default", "readout": "default", "context_enabled": false}
00129 |       }
00130 |     },
00131 |     {
00132 |       "label": "phase6_baseline_default_huber_b0p01_seed1",
00133 |       "display_label": "Phase 6 baseline default readout Huber beta 0.01 seed 1",
00134 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00135 |       "training_settings": {
00136 |         "seed_everything": 1,
00137 |         "max_epochs": 550,
00138 |         "optim_lr": 0.005,
00139 |         "batch_size": 96,
00140 |         "loader_threads": 4,
00141 |         "num_interactions": 3,
00142 |         "correlation": 2,
00143 |         "max_ell": 3,
00144 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00145 |         "loss": "graph2mat.core.data.metrics.block_type_huber",
00146 |         "loss_kwargs": {"beta": 0.01},
00147 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00148 |         "model": {"readout": "default"},
00149 |         "benchmark_metadata": {"benchmark_method_id": "baseline_default_huber_b0p01", "architecture": "default", "readout": "default", "context_enabled": false}
00150 |       }
00151 |     },
00152 |     {
00153 |       "label": "phase6_baseline_default_huber_b0p01_seed2",
00154 |       "display_label": "Phase 6 baseline default readout Huber beta 0.01 seed 2",
00155 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00156 |       "training_settings": {
00157 |         "seed_everything": 2,
00158 |         "max_epochs": 550,
00159 |         "optim_lr": 0.005,
00160 |         "batch_size": 96,
00161 |         "loader_threads": 4,
00162 |         "num_interactions": 3,
00163 |         "correlation": 2,
00164 |         "max_ell": 3,
00165 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00166 |         "loss": "graph2mat.core.data.metrics.block_type_huber",
00167 |         "loss_kwargs": {"beta": 0.01},
00168 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00169 |         "model": {"readout": "default"},
00170 |         "benchmark_metadata": {"benchmark_method_id": "baseline_default_huber_b0p01", "architecture": "default", "readout": "default", "context_enabled": false}
00171 |       }
00172 |     },
00173 |     {
00174 |       "label": "phase6_baseline_default_mse_seed0",
00175 |       "display_label": "Phase 6 baseline default readout MSE seed 0",
00176 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00177 |       "training_settings": {
00178 |         "seed_everything": 0,
00179 |         "max_epochs": 550,
00180 |         "optim_lr": 0.005,
00181 |         "batch_size": 96,
00182 |         "loader_threads": 4,
00183 |         "num_interactions": 3,
00184 |         "correlation": 2,
00185 |         "max_ell": 3,
00186 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00187 |         "loss": "graph2mat.core.data.metrics.block_type_mse",
00188 |         "loss_kwargs": {},
00189 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00190 |         "model": {"readout": "default"},
00191 |         "benchmark_metadata": {"benchmark_method_id": "baseline_default_mse", "architecture": "default", "readout": "default", "context_enabled": false}
00192 |       }
00193 |     },
00194 |     {
00195 |       "label": "phase6_baseline_default_mse_seed1",
00196 |       "display_label": "Phase 6 baseline default readout MSE seed 1",
00197 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00198 |       "training_settings": {
00199 |         "seed_everything": 1,
00200 |         "max_epochs": 550,
00201 |         "optim_lr": 0.005,
00202 |         "batch_size": 96,
00203 |         "loader_threads": 4,
00204 |         "num_interactions": 3,
00205 |         "correlation": 2,
00206 |         "max_ell": 3,
00207 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00208 |         "loss": "graph2mat.core.data.metrics.block_type_mse",
00209 |         "loss_kwargs": {},
00210 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00211 |         "model": {"readout": "default"},
00212 |         "benchmark_metadata": {"benchmark_method_id": "baseline_default_mse", "architecture": "default", "readout": "default", "context_enabled": false}
00213 |       }
00214 |     },
00215 |     {
00216 |       "label": "phase6_baseline_default_mse_seed2",
00217 |       "display_label": "Phase 6 baseline default readout MSE seed 2",
00218 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00219 |       "training_settings": {
00220 |         "seed_everything": 2,
00221 |         "max_epochs": 550,
00222 |         "optim_lr": 0.005,
00223 |         "batch_size": 96,
00224 |         "loader_threads": 4,
00225 |         "num_interactions": 3,
00226 |         "correlation": 2,
00227 |         "max_ell": 3,
00228 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00229 |         "loss": "graph2mat.core.data.metrics.block_type_mse",
00230 |         "loss_kwargs": {},
00231 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00232 |         "model": {"readout": "default"},
00233 |         "benchmark_metadata": {"benchmark_method_id": "baseline_default_mse", "architecture": "default", "readout": "default", "context_enabled": false}
00234 |       }
00235 |     },
00236 |     {
00237 |       "label": "phase6_context_default_huber_b0p01_seed0",
00238 |       "display_label": "Phase 6 Hamiltonian context default readout Huber beta 0.01 seed 0",
00239 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00240 |       "training_settings": {
00241 |         "seed_everything": 0,
00242 |         "max_epochs": 550,
00243 |         "optim_lr": 0.005,
00244 |         "batch_size": 96,
00245 |         "loader_threads": 4,
00246 |         "num_interactions": 3,
00247 |         "correlation": 2,
00248 |         "max_ell": 3,
00249 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00250 |         "loss": "graph2mat.core.data.metrics.block_type_huber",
00251 |         "loss_kwargs": {"beta": 0.01},
00252 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00253 |         "model": {"readout": "default", "hamiltonian_context": {"enabled": true}},
00254 |         "benchmark_metadata": {"benchmark_method_id": "context_default_huber_b0p01", "architecture": "hamiltonian_context", "readout": "default", "context_enabled": true}
00255 |       }
00256 |     },
00257 |     {
00258 |       "label": "phase6_context_default_huber_b0p01_seed1",
00259 |       "display_label": "Phase 6 Hamiltonian context default readout Huber beta 0.01 seed 1",
00260 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00261 |       "training_settings": {
00262 |         "seed_everything": 1,
00263 |         "max_epochs": 550,
00264 |         "optim_lr": 0.005,
00265 |         "batch_size": 96,
00266 |         "loader_threads": 4,
00267 |         "num_interactions": 3,
00268 |         "correlation": 2,
00269 |         "max_ell": 3,
00270 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00271 |         "loss": "graph2mat.core.data.metrics.block_type_huber",
00272 |         "loss_kwargs": {"beta": 0.01},
00273 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00274 |         "model": {"readout": "default", "hamiltonian_context": {"enabled": true}},
00275 |         "benchmark_metadata": {"benchmark_method_id": "context_default_huber_b0p01", "architecture": "hamiltonian_context", "readout": "default", "context_enabled": true}
00276 |       }
00277 |     },
00278 |     {
00279 |       "label": "phase6_context_default_huber_b0p01_seed2",
00280 |       "display_label": "Phase 6 Hamiltonian context default readout Huber beta 0.01 seed 2",
00281 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00282 |       "training_settings": {
00283 |         "seed_everything": 2,
00284 |         "max_epochs": 550,
00285 |         "optim_lr": 0.005,
00286 |         "batch_size": 96,
00287 |         "loader_threads": 4,
00288 |         "num_interactions": 3,
00289 |         "correlation": 2,
00290 |         "max_ell": 3,
00291 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00292 |         "loss": "graph2mat.core.data.metrics.block_type_huber",
00293 |         "loss_kwargs": {"beta": 0.01},
00294 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00295 |         "model": {"readout": "default", "hamiltonian_context": {"enabled": true}},
00296 |         "benchmark_metadata": {"benchmark_method_id": "context_default_huber_b0p01", "architecture": "hamiltonian_context", "readout": "default", "context_enabled": true}
00297 |       }
00298 |     },
00299 |     {
00300 |       "label": "phase6_context_hamreadout_huber_b0p01_seed0",
00301 |       "display_label": "Phase 6 Hamiltonian context Hamiltonian readout Huber beta 0.01 seed 0",
00302 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00303 |       "training_settings": {
00304 |         "seed_everything": 0,
00305 |         "max_epochs": 550,
00306 |         "optim_lr": 0.005,
00307 |         "batch_size": 96,
00308 |         "loader_threads": 4,
00309 |         "num_interactions": 3,
00310 |         "correlation": 2,
00311 |         "max_ell": 3,
00312 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00313 |         "loss": "graph2mat.core.data.metrics.block_type_huber",
00314 |         "loss_kwargs": {"beta": 0.01},
00315 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00316 |         "model": {"readout": "hamiltonian", "hamiltonian_context": {"enabled": true}},
00317 |         "benchmark_metadata": {"benchmark_method_id": "context_hamiltonian_readout_huber_b0p01", "architecture": "hamiltonian_context", "readout": "hamiltonian", "context_enabled": true}
00318 |       }
00319 |     },
00320 |     {
00321 |       "label": "phase6_context_hamreadout_huber_b0p01_seed1",
00322 |       "display_label": "Phase 6 Hamiltonian context Hamiltonian readout Huber beta 0.01 seed 1",
00323 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00324 |       "training_settings": {
00325 |         "seed_everything": 1,
00326 |         "max_epochs": 550,
00327 |         "optim_lr": 0.005,
00328 |         "batch_size": 96,
00329 |         "loader_threads": 4,
00330 |         "num_interactions": 3,
00331 |         "correlation": 2,
00332 |         "max_ell": 3,
00333 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00334 |         "loss": "graph2mat.core.data.metrics.block_type_huber",
00335 |         "loss_kwargs": {"beta": 0.01},
00336 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00337 |         "model": {"readout": "hamiltonian", "hamiltonian_context": {"enabled": true}},
00338 |         "benchmark_metadata": {"benchmark_method_id": "context_hamiltonian_readout_huber_b0p01", "architecture": "hamiltonian_context", "readout": "hamiltonian", "context_enabled": true}
00339 |       }
00340 |     },
00341 |     {
00342 |       "label": "phase6_context_hamreadout_huber_b0p01_seed2",
00343 |       "display_label": "Phase 6 Hamiltonian context Hamiltonian readout Huber beta 0.01 seed 2",
00344 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00345 |       "training_settings": {
00346 |         "seed_everything": 2,
00347 |         "max_epochs": 550,
00348 |         "optim_lr": 0.005,
00349 |         "batch_size": 96,
00350 |         "loader_threads": 4,
00351 |         "num_interactions": 3,
00352 |         "correlation": 2,
00353 |         "max_ell": 3,
00354 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00355 |         "loss": "graph2mat.core.data.metrics.block_type_huber",
00356 |         "loss_kwargs": {"beta": 0.01},
00357 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00358 |         "model": {"readout": "hamiltonian", "hamiltonian_context": {"enabled": true}},
00359 |         "benchmark_metadata": {"benchmark_method_id": "context_hamiltonian_readout_huber_b0p01", "architecture": "hamiltonian_context", "readout": "hamiltonian", "context_enabled": true}
00360 |       }
00361 |     },
00362 |     {
00363 |       "label": "phase6_context_hamreadout_staged_seed0",
00364 |       "display_label": "Phase 6 Hamiltonian context Hamiltonian readout staged coefficient/composite seed 0",
00365 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00366 |       "training_settings": {
00367 |         "seed_everything": 0,
00368 |         "max_epochs": 550,
00369 |         "optim_lr": 0.005,
00370 |         "batch_size": 96,
00371 |         "loader_threads": 4,
00372 |         "num_interactions": 3,
00373 |         "correlation": 2,
00374 |         "max_ell": 3,
00375 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00376 |         "loss": "graph2mat.core.data.metrics.block_type_huber",
00377 |         "loss_kwargs": {"beta": 0.01},
00378 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00379 |         "model": {"readout": "hamiltonian", "hamiltonian_context": {"enabled": true}, "return_coefficients": true},
00380 |         "training": {"training_stages": [{"id": "coefficient_warmup", "loss": "graph2mat.metrics.coefficient_space_mse"}, {"id": "composite_huber", "loss": "graph2mat.core.data.metrics.block_type_huber", "loss_kwargs": {"beta": 0.01}}]},
00381 |         "benchmark_metadata": {"benchmark_method_id": "context_hamiltonian_readout_staged_composite", "architecture": "hamiltonian_context", "readout": "hamiltonian", "context_enabled": true, "training_stages": [{"id": "coefficient_warmup"}, {"id": "composite_huber"}]}
00382 |       }
00383 |     },
00384 |     {
00385 |       "label": "phase6_context_hamreadout_staged_seed1",
00386 |       "display_label": "Phase 6 Hamiltonian context Hamiltonian readout staged coefficient/composite seed 1",
00387 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00388 |       "training_settings": {
00389 |         "seed_everything": 1,
00390 |         "max_epochs": 550,
00391 |         "optim_lr": 0.005,
00392 |         "batch_size": 96,
00393 |         "loader_threads": 4,
00394 |         "num_interactions": 3,
00395 |         "correlation": 2,
00396 |         "max_ell": 3,
00397 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00398 |         "loss": "graph2mat.core.data.metrics.block_type_huber",
00399 |         "loss_kwargs": {"beta": 0.01},
00400 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00401 |         "model": {"readout": "hamiltonian", "hamiltonian_context": {"enabled": true}, "return_coefficients": true},
00402 |         "training": {"training_stages": [{"id": "coefficient_warmup", "loss": "graph2mat.metrics.coefficient_space_mse"}, {"id": "composite_huber", "loss": "graph2mat.core.data.metrics.block_type_huber", "loss_kwargs": {"beta": 0.01}}]},
00403 |         "benchmark_metadata": {"benchmark_method_id": "context_hamiltonian_readout_staged_composite", "architecture": "hamiltonian_context", "readout": "hamiltonian", "context_enabled": true, "training_stages": [{"id": "coefficient_warmup"}, {"id": "composite_huber"}]}
00404 |       }
00405 |     },
00406 |     {
00407 |       "label": "phase6_context_hamreadout_staged_seed2",
00408 |       "display_label": "Phase 6 Hamiltonian context Hamiltonian readout staged coefficient/composite seed 2",
00409 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00410 |       "training_settings": {
00411 |         "seed_everything": 2,
00412 |         "max_epochs": 550,
00413 |         "optim_lr": 0.005,
00414 |         "batch_size": 96,
00415 |         "loader_threads": 4,
00416 |         "num_interactions": 3,
00417 |         "correlation": 2,
00418 |         "max_ell": 3,
00419 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00420 |         "loss": "graph2mat.core.data.metrics.block_type_huber",
00421 |         "loss_kwargs": {"beta": 0.01},
00422 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00423 |         "model": {"readout": "hamiltonian", "hamiltonian_context": {"enabled": true}, "return_coefficients": true},
00424 |         "training": {"training_stages": [{"id": "coefficient_warmup", "loss": "graph2mat.metrics.coefficient_space_mse"}, {"id": "composite_huber", "loss": "graph2mat.core.data.metrics.block_type_huber", "loss_kwargs": {"beta": 0.01}}]},
00425 |         "benchmark_metadata": {"benchmark_method_id": "context_hamiltonian_readout_staged_composite", "architecture": "hamiltonian_context", "readout": "hamiltonian", "context_enabled": true, "training_stages": [{"id": "coefficient_warmup"}, {"id": "composite_huber"}]}
00426 |       }
00427 |     },
00428 |     {
00429 |       "label": "phase6_diagnostic_dense_mse_seed0",
00430 |       "display_label": "Phase 6 diagnostic dense readout MSE seed 0",
00431 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00432 |       "training_settings": {
00433 |         "seed_everything": 0,
00434 |         "max_epochs": 550,
00435 |         "optim_lr": 0.005,
00436 |         "batch_size": 96,
00437 |         "loader_threads": 4,
00438 |         "num_interactions": 3,
00439 |         "correlation": 2,
00440 |         "max_ell": 3,
00441 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00442 |         "loss": "graph2mat.core.data.metrics.block_type_mse",
00443 |         "loss_kwargs": {},
00444 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00445 |         "model": {"readout": "diagnostic_dense", "node_block_readout": "graph2mat.bindings.e3nn.E3nnDiagnosticDenseNodeBlock", "edge_block_readout": "graph2mat.bindings.e3nn.E3nnDiagnosticDenseEdgeBlock"},
00446 |         "benchmark_metadata": {"benchmark_method_id": "diagnostic_dense_mse", "architecture": "diagnostic_dense", "readout": "diagnostic_dense", "context_enabled": false, "diagnostic_only": true, "notes": "Non-equivariant diagnostic upper bound; do not rank as production."}
00447 |       }
00448 |     },
00449 |     {
00450 |       "label": "phase6_diagnostic_dense_mse_seed1",
00451 |       "display_label": "Phase 6 diagnostic dense readout MSE seed 1",
00452 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00453 |       "training_settings": {
00454 |         "seed_everything": 1,
00455 |         "max_epochs": 550,
00456 |         "optim_lr": 0.005,
00457 |         "batch_size": 96,
00458 |         "loader_threads": 4,
00459 |         "num_interactions": 3,
00460 |         "correlation": 2,
00461 |         "max_ell": 3,
00462 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00463 |         "loss": "graph2mat.core.data.metrics.block_type_mse",
00464 |         "loss_kwargs": {},
00465 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00466 |         "model": {"readout": "diagnostic_dense", "node_block_readout": "graph2mat.bindings.e3nn.E3nnDiagnosticDenseNodeBlock", "edge_block_readout": "graph2mat.bindings.e3nn.E3nnDiagnosticDenseEdgeBlock"},
00467 |         "benchmark_metadata": {"benchmark_method_id": "diagnostic_dense_mse", "architecture": "diagnostic_dense", "readout": "diagnostic_dense", "context_enabled": false, "diagnostic_only": true, "notes": "Non-equivariant diagnostic upper bound; do not rank as production."}
00468 |       }
00469 |     },
00470 |     {
00471 |       "label": "phase6_diagnostic_dense_mse_seed2",
00472 |       "display_label": "Phase 6 diagnostic dense readout MSE seed 2",
00473 |       "reusable_dataset_ids": ["98c89bd85937f6a2"],
00474 |       "training_settings": {
00475 |         "seed_everything": 2,
00476 |         "max_epochs": 550,
00477 |         "optim_lr": 0.005,
00478 |         "batch_size": 96,
00479 |         "loader_threads": 4,
00480 |         "num_interactions": 3,
00481 |         "correlation": 2,
00482 |         "max_ell": 3,
00483 |         "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o",
00484 |         "loss": "graph2mat.core.data.metrics.block_type_mse",
00485 |         "loss_kwargs": {},
00486 |         "data": {"out_matrix": "hamiltonian", "matrix_component_policy": "h_only", "n_matrix_components": 1, "symmetric_matrix": true},
00487 |         "model": {"readout": "diagnostic_dense", "node_block_readout": "graph2mat.bindings.e3nn.E3nnDiagnosticDenseNodeBlock", "edge_block_readout": "graph2mat.bindings.e3nn.E3nnDiagnosticDenseEdgeBlock"},
00488 |         "benchmark_metadata": {"benchmark_method_id": "diagnostic_dense_mse", "architecture": "diagnostic_dense", "readout": "diagnostic_dense", "context_enabled": false, "diagnostic_only": true, "notes": "Non-equivariant diagnostic upper bound; do not rank as production."}
00489 |       }
00490 |     }
00491 |   ],
00492 |   "venv_activate_command": "source /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/activate"
00493 | }
```

## `Comparison/scripts/g2m_deeph_rank_runs.py`

SHA-256: `7e145dd1558931038573edee62f16f99586f2fd6b95338df66175af080c3db0d`

```py
00001 | #!/usr/bin/env python3
00002 | """Best-run ranking for Graph2Mat-vs-DeepH benchmark outputs."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import argparse
00007 | import csv
00008 | import json
00009 | import math
00010 | import time
00011 | from collections import defaultdict
00012 | from pathlib import Path
00013 | from typing import Any
00014 | 
00015 | from analyze_winners import (
00016 |     mean,
00017 |     metric_lower_is_better as _legacy_metric_lower_is_better,
00018 |     metric_policy_role as _legacy_metric_policy_role,
00019 |     seed_stability_status,
00020 |     stddev,
00021 |     valid_stability_seeds,
00022 |     warning_items,
00023 | )
00024 | from deeph_prediction_adapter import (
00025 |     EQUIVALENCE_PROVEN_RAW_GLOBAL,
00026 |     EQUIVALENCE_STATUS_PROVEN,
00027 |     EQUIVALENCE_STATUS_UNPROVEN,
00028 | )
00029 | from g2m_deeph_metrics import summarize_method
00030 | 
00031 | 
00032 | SCHEMA = "graph2mat_deeph_run_ranking_v1"
00033 | MODELS = ("graph2mat", "deeph")
00034 | METRIC_FAIL_POLICY_FAIL_CLOSED = "fail_closed"
00035 | METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY = "diagnostic_only"
00036 | VALID_DATASET_STATUSES = {"valid_joint_one_pass_dataset", "valid_reused_joint_dataset"}
00037 | VALID_ARTIFACT_CONTRACT_STATUSES = {"valid", "benchmark_ready"}
00038 | VALID_SPLIT_AUDIT_STATUSES = {"valid"}
00039 | VALID_COMPARABILITY_STATUSES = {"valid", "valid_joint_one_pass_dataset", "valid_reused_joint_dataset"}
00040 | RECOMMENDATION_STATUS_VALUES = {
00041 |     "robust_graph2mat_win",
00042 |     "robust_deeph_win",
00043 |     "exploratory_graph2mat_win",
00044 |     "exploratory_deeph_win",
00045 |     "no_robust_winner",
00046 |     "diagnostic_only",
00047 |     "invalid_incomplete_grid",
00048 |     "invalid_incompatible_splits",
00049 |     "invalid_incompatible_artifacts",
00050 |     "invalid_missing_provenance",
00051 |     "invalid_prediction_format",
00052 |     "invalid_metric_policy",
00053 |     "invalid_unverified_deeph_split",
00054 |     "unstable_across_seeds",
00055 | }
00056 | STATUS_PRIORITY = (
00057 |     "invalid_incompatible_splits",
00058 |     "invalid_incompatible_artifacts",
00059 |     "invalid_missing_provenance",
00060 |     "invalid_unverified_deeph_split",
00061 |     "invalid_prediction_format",
00062 |     "invalid_metric_policy",
00063 |     "invalid_incomplete_grid",
00064 |     "unstable_across_seeds",
00065 |     "diagnostic_only",
00066 | )
00067 | DIAGNOSTIC_GATES = {
00068 |     "diagnostic_only",
00069 |     "metric_fail_policy_diagnostic_only",
00070 |     "deeph_adapter_equivalence_not_proven",
00071 | }
00072 | PRIMARY_METRIC_PRIORITY = [
00073 |     "low_energy_rmse_eV",
00074 |     "fermi_window_rmse_eV",
00075 |     "frontier_window_rmse_eV",
00076 |     "dos_mae_500_fermi_window",
00077 |     "dos_wasserstein_eV",
00078 |     "relative_frobenius",
00079 |     "relative_frobenius_union",
00080 |     "h_mae_eV",
00081 | ]
00082 | DIAGNOSTIC_ONLY_METRICS = {
00083 |     "global_rmse_eV",
00084 |     "global_mae_eV",
00085 |     "support_precision",
00086 |     "support_recall",
00087 |     "support_f1",
00088 |     "hermiticity_pred",
00089 |     "hermiticity_error",
00090 |     "antihermitian_norm",
00091 | }
00092 | RECOMMENDATION_GRADE_METRICS = {
00093 |     "low_energy_rmse_eV",
00094 |     "fermi_window_rmse_eV",
00095 |     "frontier_window_rmse_eV",
00096 |     "dos_mae_500_fermi_window",
00097 |     "dos_wasserstein_eV",
00098 |     "relative_frobenius",
00099 |     "relative_frobenius_union",
00100 |     "h_mae_eV",
00101 | }
00102 | METRIC_ALIASES = {
00103 |     "h_mae_eV_mean": "h_mae_eV",
00104 |     "h_rmse_eV_mean": "h_rmse_eV",
00105 |     "h_mse_eV2_mean": "h_mse_eV2",
00106 |     "r2_mean": "r2",
00107 |     "relative_frobenius_mean": "relative_frobenius",
00108 |     "support_precision_mean": "support_precision",
00109 |     "support_recall_mean": "support_recall",
00110 |     "support_f1_mean": "support_f1",
00111 |     "hermiticity_pred_mean": "hermiticity_pred",
00112 |     "global_rmse_eV_mean": "global_rmse_eV",
00113 |     "low_energy_rmse_eV_mean": "low_energy_rmse_eV",
00114 |     "fermi_window_rmse_eV_mean": "fermi_window_rmse_eV",
00115 |     "frontier_window_rmse_eV_mean": "frontier_window_rmse_eV",
00116 |     "dos_mae_500_fermi_window_mean": "dos_mae_500_fermi_window",
00117 | }
00118 | CANONICAL_TO_SOURCE = {value: key for key, value in METRIC_ALIASES.items()}
00119 | HIGHER_IS_BETTER = {"r2", "support_f1"}
00120 | 
00121 | 
00122 | def json_safe(value: Any) -> Any:
00123 |     if isinstance(value, Path):
00124 |         return str(value)
00125 |     if isinstance(value, float):
00126 |         return value if math.isfinite(value) else None
00127 |     if isinstance(value, dict):
00128 |         return {str(key): json_safe(item) for key, item in sorted(value.items())}
00129 |     if isinstance(value, (list, tuple)):
00130 |         return [json_safe(item) for item in value]
00131 |     return value
00132 | 
00133 | 
00134 | def read_json(path: Path | None) -> dict[str, Any]:
00135 |     if path is None or not path.exists():
00136 |         return {}
00137 |     payload = json.loads(path.read_text(encoding="utf-8"))
00138 |     return payload if isinstance(payload, dict) else {}
00139 | 
00140 | 
00141 | def write_json(path: Path, payload: dict[str, Any]) -> None:
00142 |     path.parent.mkdir(parents=True, exist_ok=True)
00143 |     path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
00144 | 
00145 | 
00146 | def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
00147 |     path.parent.mkdir(parents=True, exist_ok=True)
00148 |     if fieldnames is None:
00149 |         fieldnames = sorted({key for row in rows for key in row}) or ["status"]
00150 |     with path.open("w", encoding="utf-8", newline="") as handle:
00151 |         writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
00152 |         writer.writeheader()
00153 |         writer.writerows([{key: json_safe(value) for key, value in row.items()} for row in rows])
00154 | 
00155 | 
00156 | def number(value: Any) -> float:
00157 |     try:
00158 |         result = float(value)
00159 |     except (TypeError, ValueError):
00160 |         return math.nan
00161 |     return result if math.isfinite(result) else math.nan
00162 | 
00163 | 
00164 | def finite(value: Any) -> bool:
00165 |     return math.isfinite(number(value))
00166 | 
00167 | 
00168 | def normalize_model(value: Any) -> str:
00169 |     text = str(value or "").strip().lower().replace("-", "_")
00170 |     aliases = {
00171 |         "g2m": "graph2mat",
00172 |         "graph_2_mat": "graph2mat",
00173 |         "deep_h": "deeph",
00174 |         "deeph_pack": "deeph",
00175 |     }
00176 |     return aliases.get(text, text)
00177 | 
00178 | 
00179 | def canonical_metric(metric: str) -> str:
00180 |     text = str(metric or "").strip()
00181 |     if text in METRIC_ALIASES:
00182 |         return METRIC_ALIASES[text]
00183 |     if text.endswith("_mean") and text.removesuffix("_mean") in RECOMMENDATION_GRADE_METRICS | DIAGNOSTIC_ONLY_METRICS:
00184 |         return text.removesuffix("_mean")
00185 |     return text
00186 | 
00187 | 
00188 | def source_metric(metric: str) -> str:
00189 |     canonical = canonical_metric(metric)
00190 |     return CANONICAL_TO_SOURCE.get(canonical, metric)
00191 | 
00192 | 
00193 | def metric_lower_is_better(metric: str) -> bool:
00194 |     canonical = canonical_metric(metric)
00195 |     if canonical in HIGHER_IS_BETTER:
00196 |         return False
00197 |     return _legacy_metric_lower_is_better(canonical)
00198 | 
00199 | 
00200 | def metric_policy_role(metric: str) -> str:
00201 |     canonical = canonical_metric(metric)
00202 |     if canonical in DIAGNOSTIC_ONLY_METRICS:
00203 |         return "diagnostic_only"
00204 |     if canonical in RECOMMENDATION_GRADE_METRICS:
00205 |         return "recommendation_grade"
00206 |     legacy = _legacy_metric_policy_role(canonical)
00207 |     return legacy if legacy != "unknown_metric" else "diagnostic_only"
00208 | 
00209 | 
00210 | def severe_warning_items(*values: Any) -> list[Any]:
00211 |     severe: list[Any] = []
00212 |     for value in values:
00213 |         if isinstance(value, list):
00214 |             for item in value:
00215 |                 if isinstance(item, dict) and str(item.get("severity") or "").lower() == "severe":
00216 |                     severe.append(item)
00217 |                 elif isinstance(item, str) and "severe" in item.lower():
00218 |                     severe.append(item)
00219 |         elif isinstance(value, dict):
00220 |             if str(value.get("severity") or "").lower() == "severe":
00221 |                 severe.append(value)
00222 |         else:
00223 |             for item in warning_items(value):
00224 |                 if "severe" in str(item).lower():
00225 |                     severe.append(item)
00226 |     return severe
00227 | 
00228 | 
00229 | def warning_status(severe_warnings: list[Any]) -> str:
00230 |     return "severe" if severe_warnings else "ok"
00231 | 
00232 | 
00233 | def metric_fail_policy_warning(policy: str) -> dict[str, str] | None:
00234 |     if policy != METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY:
00235 |         return None
00236 |     return {
00237 |         "severity": "severe",
00238 |         "kind": "metric_fail_policy_diagnostic_only",
00239 |         "message": "Metrics were produced in explicit fail-open diagnostic mode; robust ranking is disabled.",
00240 |     }
00241 | 
00242 | 
00243 | def deeph_adapter_equivalence_warning(
00244 |     status: str,
00245 |     *,
00246 |     equivalence_status: str = "",
00247 |     reason: str = "",
00248 | ) -> dict[str, str]:
00249 |     return {
00250 |         "severity": "severe",
00251 |         "kind": "deeph_adapter_equivalence_not_proven",
00252 |         "adapter_equivalence_status": status or "missing",
00253 |         "equivalence_status": equivalence_status or "missing",
00254 |         "diagnostic_only_reason": reason or "DeepH raw/global equivalence evidence is not proven.",
00255 |         "message": "DeepH prediction equivalence to Graph2Mat raw/global HSX is not proven.",
00256 |     }
00257 | 
00258 | 
00259 | def deeph_adapter_equivalence_proven(row: dict[str, Any]) -> bool:
00260 |     if normalize_model(row.get("model")) != "deeph":
00261 |         return True
00262 |     adapter_status = str(row.get("adapter_equivalence_status") or "")
00263 |     equivalence_status = str(row.get("equivalence_status") or "")
00264 |     if equivalence_status:
00265 |         return adapter_status == EQUIVALENCE_PROVEN_RAW_GLOBAL and equivalence_status == EQUIVALENCE_STATUS_PROVEN
00266 |     return adapter_status == EQUIVALENCE_PROVEN_RAW_GLOBAL
00267 | 
00268 | 
00269 | def deeph_adapter_status(rows: list[dict[str, Any]]) -> str:
00270 |     statuses = sorted(
00271 |         {
00272 |             str(row.get("adapter_equivalence_status") or "")
00273 |             for row in rows
00274 |             if normalize_model(row.get("model")) == "deeph" and str(row.get("adapter_equivalence_status") or "")
00275 |         }
00276 |     )
00277 |     return statuses[0] if len(statuses) == 1 else ",".join(statuses) if statuses else "missing"
00278 | 
00279 | 
00280 | def deeph_equivalence_status(rows: list[dict[str, Any]]) -> str:
00281 |     statuses = sorted(
00282 |         {
00283 |             str(row.get("equivalence_status") or "")
00284 |             for row in rows
00285 |             if normalize_model(row.get("model")) == "deeph" and str(row.get("equivalence_status") or "")
00286 |         }
00287 |     )
00288 |     if statuses:
00289 |         return statuses[0] if len(statuses) == 1 else ",".join(statuses)
00290 |     adapter_status = deeph_adapter_status(rows)
00291 |     return EQUIVALENCE_STATUS_PROVEN if adapter_status == EQUIVALENCE_PROVEN_RAW_GLOBAL else EQUIVALENCE_STATUS_UNPROVEN
00292 | 
00293 | 
00294 | def split_audit_status(rows: list[dict[str, Any]]) -> str:
00295 |     statuses = sorted(
00296 |         {
00297 |             str(row.get("split_audit_status") or "")
00298 |             for row in rows
00299 |             if normalize_model(row.get("model")) == "deeph" and str(row.get("split_audit_status") or "")
00300 |         }
00301 |     )
00302 |     return statuses[0] if len(statuses) == 1 else ",".join(statuses) if statuses else "missing"
00303 | 
00304 | 
00305 | def dataset_metadata(dataset_root: Path | None) -> dict[str, Any]:
00306 |     if dataset_root is None:
00307 |         return {}
00308 |     split = read_json(dataset_root / "frozen_split_manifest.json")
00309 |     manifest = read_json(dataset_root / "benchmark_dataset_manifest.json")
00310 |     compatibility_hash = (
00311 |         manifest.get("dataset_compatibility_hash")
00312 |         or manifest.get("material_compatibility_hash")
00313 |         or manifest.get("benchmark_dataset_id")
00314 |         or manifest.get("siesta_input_sha256")
00315 |         or str(dataset_root)
00316 |     )
00317 |     provenance = manifest.get("provenance_status") or {}
00318 |     provenance_valid = bool(provenance.get("valid")) if isinstance(provenance, dict) and provenance else bool(manifest.get("benchmark_ready"))
00319 |     return {
00320 |         "dataset_id": manifest.get("benchmark_dataset_id") or dataset_root.name,
00321 |         "dataset_label": manifest.get("material_label") or dataset_root.name,
00322 |         "dataset_recipe_id": manifest.get("dataset_recipe_id") or "",
00323 |         "dataset_compatibility_hash": compatibility_hash,
00324 |         "frozen_split_hash": split.get("split_hash") or (manifest.get("frozen_split_manifest") or {}).get("split_hash") or "",
00325 |         "artifact_contract_status": "valid" if manifest.get("benchmark_ready") else "unknown",
00326 |         "provenance_status": "valid" if provenance_valid else "invalid",
00327 |         "required_provenance_present": provenance_valid,
00328 |         "dataset_scientific_status": "valid" if manifest.get("benchmark_ready") else "unknown",
00329 |     }
00330 | 
00331 | 
00332 | def deeph_manifest_metadata(record: dict[str, Any]) -> dict[str, Any]:
00333 |     if normalize_model(record.get("model")) != "deeph":
00334 |         return {"split_audit_status": "not_applicable"}
00335 |     manifest_path = record.get("deeph_manifest_path")
00336 |     manifest = read_json(Path(str(manifest_path))) if manifest_path else {}
00337 |     audit = manifest.get("split_audit") or {}
00338 |     status = (
00339 |         record.get("split_audit_status")
00340 |         or manifest.get("split_audit_status")
00341 |         or audit.get("status")
00342 |         or "missing"
00343 |     )
00344 |     return {
00345 |         "split_audit_status": str(status),
00346 |         "split_audit_path": str(manifest.get("split_audit_path") or record.get("split_audit_path") or ""),
00347 |     }
00348 | 
00349 | 
00350 | def metric_root_for_record(record: dict[str, Any]) -> Path | None:
00351 |     run_root = Path(str(record.get("run_root") or ""))
00352 |     model = normalize_model(record.get("model"))
00353 |     if not str(run_root) or not run_root.exists():
00354 |         return None
00355 |     candidates = (
00356 |         [run_root / "metrics" / "graph2mat" / "eval_input" / "metrics", run_root / "common_metrics" / "graph2mat_eval" / "metrics"]
00357 |         if model == "graph2mat"
00358 |         else [run_root / "metrics" / "deeph" / "eval" / "metrics", run_root / "common_metrics" / "deeph_eval" / "metrics"]
00359 |     )
00360 |     for candidate in candidates:
00361 |         if candidate.exists():
00362 |             return candidate
00363 |     return None
00364 | 
00365 | 
00366 | def timing_seconds(record: dict[str, Any]) -> dict[str, float | None]:
00367 |     train = number((record.get("train_run") or {}).get("elapsed_seconds"))
00368 |     predict = number((record.get("predict_run") or {}).get("elapsed_seconds"))
00369 |     metrics = number((record.get("metrics_run") or {}).get("elapsed_seconds"))
00370 |     preprocess = number((record.get("preprocess_run") or {}).get("elapsed_seconds"))
00371 |     inference = sum(
00372 |         number(run.get("elapsed_seconds"))
00373 |         for run in record.get("inference_runs") or []
00374 |         if isinstance(run, dict) and finite(run.get("elapsed_seconds"))
00375 |     )
00376 |     values = [train, predict, metrics, preprocess, inference]
00377 |     total = sum(value for value in values if math.isfinite(value))
00378 |     return {
00379 |         "training_time_seconds": train if math.isfinite(train) else None,
00380 |         "prediction_time_seconds": predict if math.isfinite(predict) else (inference if inference else None),
00381 |         "preprocess_time_seconds": preprocess if math.isfinite(preprocess) else None,
00382 |         "evaluation_time_seconds": metrics if math.isfinite(metrics) else None,
00383 |         "total_time_seconds": total if total else None,
00384 |     }
00385 | 
00386 | 
00387 | def telemetry_fields(record: dict[str, Any]) -> dict[str, Any]:
00388 |     telemetry = record.get("telemetry") if isinstance(record.get("telemetry"), dict) else {}
00389 |     if not telemetry and record.get("telemetry_path"):
00390 |         telemetry = read_json(Path(str(record.get("telemetry_path"))))
00391 |     if not isinstance(telemetry, dict) or not telemetry:
00392 |         return {
00393 |             "telemetry_status": "unavailable",
00394 |             "wall_clock_seconds_total": None,
00395 |             "gpu_hours_total": None,
00396 |             "gpu_hours_to_best_validation": None,
00397 |             "wall_clock_seconds_to_best_validation": None,
00398 |             "peak_gpu_memory_mb": None,
00399 |             "samples_per_second": None,
00400 |             "matrix_blocks_per_second": None,
00401 |             "epochs_trained": None,
00402 |             "best_validation_epoch": None,
00403 |             "telemetry_warnings": ["telemetry unavailable"],
00404 |         }
00405 |     return {
00406 |         "telemetry_status": telemetry.get("telemetry_status") or "partial",
00407 |         "wall_clock_seconds_total": telemetry.get("wall_clock_seconds_total"),
00408 |         "gpu_hours_total": telemetry.get("gpu_hours_total"),
00409 |         "gpu_hours_to_best_validation": telemetry.get("gpu_hours_to_best_validation"),
00410 |         "wall_clock_seconds_to_best_validation": telemetry.get("wall_clock_seconds_to_best_validation"),
00411 |         "peak_gpu_memory_mb": telemetry.get("peak_gpu_memory_mb"),
00412 |         "samples_per_second": telemetry.get("samples_per_second"),
00413 |         "matrix_blocks_per_second": telemetry.get("matrix_blocks_per_second"),
00414 |         "epochs_trained": telemetry.get("epochs_trained"),
00415 |         "best_validation_epoch": telemetry.get("best_validation_epoch"),
00416 |         "telemetry_warnings": telemetry.get("telemetry_warnings") or [],
00417 |         "hardware": telemetry.get("hardware") or {},
00418 |     }
00419 | 
00420 | 
00421 | def early_stopping_fields(record: dict[str, Any]) -> dict[str, Any]:
00422 |     metadata = record.get("early_stopping") if isinstance(record.get("early_stopping"), dict) else {}
00423 |     if not metadata:
00424 |         return {}
00425 |     return {
00426 |         "validation_metric_name": metadata.get("validation_metric_name"),
00427 |         "metric_mode": metadata.get("metric_mode"),
00428 |         "early_stopping_patience": metadata.get("patience"),
00429 |         "early_stopping_min_delta": metadata.get("min_delta"),
00430 |         "early_stopping_max_epochs": metadata.get("max_epochs"),
00431 |         "early_stopping_best_epoch": metadata.get("best_epoch"),
00432 |         "early_stopping_best_validation_value": metadata.get("best_validation_value"),
00433 |         "early_stopping_epochs_trained": metadata.get("epochs_trained"),
00434 |         "early_stopping_stop_reason": metadata.get("stop_reason"),
00435 |     }
00436 | 
00437 | 
00438 | def row_from_training_record(record: dict[str, Any]) -> dict[str, Any]:
00439 |     model = normalize_model(record.get("model"))
00440 |     if model not in MODELS:
00441 |         raise RuntimeError(f"Unsupported Graph2Mat/DeepH ranking model: {record.get('model')}")
00442 |     if not record.get("config_id"):
00443 |         raise RuntimeError("Training sweep run is missing config_id.")
00444 |     dataset_root = Path(str(record.get("dataset_root") or "")) if record.get("dataset_root") else None
00445 |     metadata = dataset_metadata(dataset_root)
00446 |     metrics_root = metric_root_for_record(record)
00447 |     method_summary: dict[str, Any] = {}
00448 |     metric_warnings: list[Any] = []
00449 |     if metrics_root is not None:
00450 |         method_summary = summarize_method(model, metrics_root)
00451 |         metric_manifest = read_json(metrics_root / "manifest.json")
00452 |         metric_warnings.extend(metric_manifest.get("warnings") or [])
00453 |         metric_warnings.extend(metric_manifest.get("fatal_errors") or [])
00454 |     metric_fail_policy = str(record.get("metric_fail_policy") or METRIC_FAIL_POLICY_FAIL_CLOSED)
00455 |     metric_policy_warning = metric_fail_policy_warning(metric_fail_policy)
00456 |     if metric_policy_warning:
00457 |         metric_warnings.append(metric_policy_warning)
00458 |     deeph_manifest = deeph_manifest_metadata(record)
00459 |     adapter_equivalence_status = str(
00460 |         method_summary.get("adapter_equivalence_status")
00461 |         or record.get("adapter_equivalence_status")
00462 |         or ""
00463 |     )
00464 |     equivalence_status = str(
00465 |         method_summary.get("equivalence_status")
00466 |         or record.get("equivalence_status")
00467 |         or (EQUIVALENCE_STATUS_PROVEN if adapter_equivalence_status == EQUIVALENCE_PROVEN_RAW_GLOBAL else "")
00468 |     )
00469 |     equivalence_scope = str(method_summary.get("equivalence_scope") or record.get("equivalence_scope") or "")
00470 |     equivalence_gate = method_summary.get("equivalence_gate") if isinstance(method_summary.get("equivalence_gate"), dict) else {}
00471 |     diagnostic_reason = str(
00472 |         method_summary.get("diagnostic_only_reason")
00473 |         or record.get("diagnostic_only_reason")
00474 |         or equivalence_gate.get("diagnostic_only_reason")
00475 |         or ""
00476 |     )
00477 |     if model == "deeph" and not deeph_adapter_equivalence_proven(
00478 |         {
00479 |             "model": model,
00480 |             "adapter_equivalence_status": adapter_equivalence_status,
00481 |             "equivalence_status": equivalence_status,
00482 |         }
00483 |     ):
00484 |         metric_warnings.append(
00485 |             deeph_adapter_equivalence_warning(
00486 |                 adapter_equivalence_status,
00487 |                 equivalence_status=equivalence_status,
00488 |                 reason=diagnostic_reason,
00489 |             )
00490 |         )
00491 |     severe = severe_warning_items(record.get("severe_warnings"), record.get("warnings"), metric_warnings)
00492 |     method_status = method_summary.get("method_status") or ("missing_metrics" if record.get("status") == "completed" else record.get("status"))
00493 |     diagnostic_only = bool(method_summary.get("diagnostic_only")) or metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY
00494 |     if model == "deeph" and not deeph_adapter_equivalence_proven(
00495 |         {
00496 |             "model": model,
00497 |             "adapter_equivalence_status": adapter_equivalence_status,
00498 |             "equivalence_status": equivalence_status,
00499 |         }
00500 |     ):
00501 |         diagnostic_only = True
00502 |     comparability_status = "diagnostic_only" if diagnostic_only else "valid"
00503 |     if method_status not in {"ok", "dry_run", "failed", "missing_metrics", None}:
00504 |         comparability_status = "invalid_prediction_format"
00505 |     overrides = record.get("overrides") if isinstance(record.get("overrides"), dict) else {}
00506 |     common = record.get("common") if isinstance(record.get("common"), dict) else {}
00507 |     epochs = overrides.get("max_epochs") or overrides.get("epochs") or common.get("epochs")
00508 |     row = {
00509 |         "benchmark_id": Path(str(record.get("run_root") or "")).parts[-5] if record.get("run_root") and len(Path(str(record.get("run_root"))).parts) >= 5 else "",
00510 |         **metadata,
00511 |         "model": model,
00512 |         "config_id": str(record.get("config_id")),
00513 |         "config_label": str(record.get("config_label") or record.get("config_id")),
00514 |         "config_hash": str(record.get("config_hash") or ""),
00515 |         "epochs": epochs,
00516 |         "epoch_label": f"{epochs} epochs" if epochs not in (None, "") else "",
00517 |         "seed": (record.get("common") or {}).get("seed")
00518 |         or (record.get("overrides") or {}).get("seed")
00519 |         or (record.get("overrides") or {}).get("seed_everything")
00520 |         or "unknown",
00521 |         "run_status": str(record.get("status") or "unknown"),
00522 |         "protocol_stage": str(record.get("protocol_stage") or "exploratory"),
00523 |         "metric_split": str(record.get("metric_split") or ("test" if record.get("metrics_run") else "")),
00524 |         "test_metrics_locked": bool(record.get("test_metrics_locked")),
00525 |         "test_metrics_status": str(record.get("test_metrics_status") or ""),
00526 |         "method_status": method_status,
00527 |         "prediction_dir": str(Path(str(record.get("run_root") or "")) / "graph2mat" / "prediction_structures")
00528 |         if model == "graph2mat"
00529 |         else str(Path(str(record.get("run_root") or "")) / "deeph" / "inference"),
00530 |         "reference_dir": str(dataset_root or ""),
00531 |         "run_dir": str(record.get("run_root") or ""),
00532 |         "metrics_root": str(metrics_root or ""),
00533 |         "metric_fail_policy": metric_fail_policy,
00534 |         "fail_open_metric_outputs": metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
00535 |         "comparability_status": comparability_status,
00536 |         "scientific_status": comparability_status,
00537 |         "warning_status": warning_status(severe),
00538 |         "severe_warnings": severe,
00539 |         "diagnostic_only": diagnostic_only,
00540 |         "adapter_equivalence_status": adapter_equivalence_status,
00541 |         "equivalence_status": equivalence_status,
00542 |         "equivalence_scope": equivalence_scope,
00543 |         "diagnostic_only_reason": diagnostic_reason,
00544 |         "raw_global_equivalence_proven": bool(method_summary.get("raw_global_equivalence_proven")),
00545 |         **deeph_manifest,
00546 |         **timing_seconds(record),
00547 |         **telemetry_fields(record),
00548 |         **early_stopping_fields(record),
00549 |     }
00550 |     for key, value in method_summary.items():
00551 |         if key.endswith("_mean"):
00552 |             row[key] = value
00553 |     return row
00554 | 
00555 | 
00556 | def rows_from_common_metrics(
00557 |     common_metrics_manifest: dict[str, Any],
00558 |     *,
00559 |     dataset_root: Path | None = None,
00560 |     frozen_split_manifest_path: Path | None = None,
00561 |     dataset_manifest_path: Path | None = None,
00562 | ) -> list[dict[str, Any]]:
00563 |     metadata = dataset_metadata(dataset_root)
00564 |     if frozen_split_manifest_path:
00565 |         split = read_json(frozen_split_manifest_path)
00566 |         metadata["frozen_split_hash"] = split.get("split_hash") or metadata.get("frozen_split_hash") or ""
00567 |     if dataset_manifest_path:
00568 |         manifest = read_json(dataset_manifest_path)
00569 |         metadata["dataset_compatibility_hash"] = (
00570 |             manifest.get("dataset_compatibility_hash")
00571 |             or manifest.get("material_compatibility_hash")
00572 |             or manifest.get("benchmark_dataset_id")
00573 |             or metadata.get("dataset_compatibility_hash")
00574 |         )
00575 |         metadata["artifact_contract_status"] = "valid" if manifest.get("benchmark_ready") else metadata.get("artifact_contract_status", "unknown")
00576 |     metric_fail_policy = str(common_metrics_manifest.get("metric_fail_policy") or METRIC_FAIL_POLICY_FAIL_CLOSED)
00577 |     policy_warning = metric_fail_policy_warning(metric_fail_policy)
00578 |     severe_inputs = [common_metrics_manifest.get("warnings")]
00579 |     if policy_warning:
00580 |         severe_inputs.append([policy_warning])
00581 |     severe = severe_warning_items(*severe_inputs)
00582 |     manifest_status = str(common_metrics_manifest.get("status") or "diagnostic_only")
00583 |     if metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY:
00584 |         manifest_status = "diagnostic_only"
00585 |     rows: list[dict[str, Any]] = []
00586 |     for item in common_metrics_manifest.get("summary_rows") or []:
00587 |         model = normalize_model(item.get("method"))
00588 |         if model not in MODELS:
00589 |             continue
00590 |         adapter_equivalence_status = str(item.get("adapter_equivalence_status") or "")
00591 |         equivalence_status = str(
00592 |             item.get("equivalence_status")
00593 |             or (EQUIVALENCE_STATUS_PROVEN if adapter_equivalence_status == EQUIVALENCE_PROVEN_RAW_GLOBAL else "")
00594 |         )
00595 |         equivalence_scope = str(item.get("equivalence_scope") or "")
00596 |         equivalence_gate = item.get("equivalence_gate") if isinstance(item.get("equivalence_gate"), dict) else {}
00597 |         diagnostic_reason = str(item.get("diagnostic_only_reason") or equivalence_gate.get("diagnostic_only_reason") or "")
00598 |         item_severe = list(severe)
00599 |         if model == "deeph" and not deeph_adapter_equivalence_proven(
00600 |             {
00601 |                 "model": model,
00602 |                 "adapter_equivalence_status": adapter_equivalence_status,
00603 |                 "equivalence_status": equivalence_status,
00604 |             }
00605 |         ):
00606 |             item_severe.append(
00607 |                 deeph_adapter_equivalence_warning(
00608 |                     adapter_equivalence_status,
00609 |                     equivalence_status=equivalence_status,
00610 |                     reason=diagnostic_reason,
00611 |                 )
00612 |             )
00613 |         row = {
00614 |             "benchmark_id": "",
00615 |             **metadata,
00616 |             "model": model,
00617 |             "config_id": f"default_{model}",
00618 |             "config_label": f"default_{model}",
00619 |             "config_hash": "",
00620 |             "seed": "unknown",
00621 |             "run_status": "completed",
00622 |             "method_status": item.get("method_status") or "ok",
00623 |             "prediction_dir": "",
00624 |             "reference_dir": str(dataset_root or ""),
00625 |             "run_dir": "",
00626 |             "metrics_root": str(item.get("metrics_root") or ""),
00627 |             "metric_fail_policy": metric_fail_policy,
00628 |             "fail_open_metric_outputs": metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
00629 |             "comparability_status": manifest_status,
00630 |             "scientific_status": manifest_status,
00631 |             "warning_status": warning_status(item_severe),
00632 |             "severe_warnings": item_severe,
00633 |             "diagnostic_only": bool(item.get("diagnostic_only"))
00634 |             or manifest_status == "diagnostic_only"
00635 |             or metric_fail_policy == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
00636 |             "adapter_equivalence_status": adapter_equivalence_status,
00637 |             "equivalence_status": equivalence_status,
00638 |             "equivalence_scope": equivalence_scope,
00639 |             "diagnostic_only_reason": diagnostic_reason,
00640 |             "raw_global_equivalence_proven": bool(item.get("raw_global_equivalence_proven")),
00641 |             "split_audit_status": str(item.get("split_audit_status") or ("missing" if model == "deeph" else "not_applicable")),
00642 |             "split_audit_path": str(item.get("split_audit_path") or ""),
00643 |             "training_time_seconds": None,
00644 |             "prediction_time_seconds": None,
00645 |             "preprocess_time_seconds": None,
00646 |             "evaluation_time_seconds": None,
00647 |             "total_time_seconds": None,
00648 |         }
00649 |         for key, value in item.items():
00650 |             if key.endswith("_mean"):
00651 |                 row[key] = value
00652 |         if model == "deeph" and not deeph_adapter_equivalence_proven(row):
00653 |             row["diagnostic_only"] = True
00654 |             row["comparability_status"] = "diagnostic_only"
00655 |             row["scientific_status"] = "diagnostic_only"
00656 |         rows.append(row)
00657 |     return rows
00658 | 
00659 | 
00660 | def load_metric_rows(
00661 |     *,
00662 |     training_sweep_manifest_path: Path | None = None,
00663 |     common_metrics_manifest_path: Path | None = None,
00664 |     dataset_root: Path | None = None,
00665 |     frozen_split_manifest_path: Path | None = None,
00666 |     dataset_manifest_path: Path | None = None,
00667 | ) -> list[dict[str, Any]]:
00668 |     rows: list[dict[str, Any]] = []
00669 |     training = read_json(training_sweep_manifest_path)
00670 |     for record in training.get("runs") or []:
00671 |         if not isinstance(record, dict):
00672 |             continue
00673 |         rows.append(row_from_training_record(record))
00674 |     common = read_json(common_metrics_manifest_path)
00675 |     if common:
00676 |         rows.extend(
00677 |             rows_from_common_metrics(
00678 |                 common,
00679 |                 dataset_root=dataset_root,
00680 |                 frozen_split_manifest_path=frozen_split_manifest_path,
00681 |                 dataset_manifest_path=dataset_manifest_path,
00682 |             )
00683 |         )
00684 |     return rows
00685 | 
00686 | 
00687 | def finite_metric(row: dict[str, Any], metric: str) -> float | None:
00688 |     value = number(row.get(source_metric(metric)))
00689 |     return value if math.isfinite(value) else None
00690 | 
00691 | 
00692 | def row_is_robust_eligible(row: dict[str, Any], metric: str) -> bool:
00693 |     return not row_gate_failures(row, metric)
00694 | 
00695 | 
00696 | def row_gate_failures(row: dict[str, Any], metric: str) -> list[str]:
00697 |     failures: list[str] = []
00698 |     model = normalize_model(row.get("model"))
00699 |     if row.get("run_status") not in {"completed", "ok"}:
00700 |         failures.append("invalid_incomplete_grid")
00701 |     if finite_metric(row, metric) is None:
00702 |         failures.append("invalid_incomplete_grid")
00703 |     if row.get("method_status") != "ok":
00704 |         failures.append("invalid_prediction_format")
00705 |     if str(row.get("artifact_contract_status") or "") not in VALID_ARTIFACT_CONTRACT_STATUSES:
00706 |         failures.append("invalid_incompatible_artifacts")
00707 |     if row.get("required_provenance_present") is not True or str(row.get("provenance_status") or "") != "valid":
00708 |         failures.append("invalid_missing_provenance")
00709 |     comparability = str(row.get("comparability_status") or "")
00710 |     if comparability not in VALID_COMPARABILITY_STATUSES:
00711 |         failures.append(comparability if comparability.startswith("invalid_") else "diagnostic_only")
00712 |     if row.get("warning_status") == "severe":
00713 |         failures.append("severe_warnings")
00714 |     if metric_policy_role(metric) == "diagnostic_only":
00715 |         failures.append("invalid_metric_policy")
00716 |     if row.get("fail_open_metric_outputs") or row.get("metric_fail_policy") == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY:
00717 |         failures.append("metric_fail_policy_diagnostic_only")
00718 |     if model == "deeph":
00719 |         if not deeph_adapter_equivalence_proven(row):
00720 |             failures.append("deeph_adapter_equivalence_not_proven")
00721 |         split_status = str(row.get("split_audit_status") or "missing")
00722 |         if split_status not in VALID_SPLIT_AUDIT_STATUSES:
00723 |             failures.append(split_status if split_status == "invalid_incompatible_splits" else "invalid_unverified_deeph_split")
00724 |     return sorted(set(failures))
00725 | 
00726 | 
00727 | def status_from_gates(gates_failed: list[str]) -> str:
00728 |     gates = set(gates_failed)
00729 |     if gates & DIAGNOSTIC_GATES:
00730 |         return "diagnostic_only"
00731 |     if "missing_model" in gates or "missing_primary_metric" in gates:
00732 |         return "invalid_incomplete_grid"
00733 |     for status in STATUS_PRIORITY:
00734 |         if status in gates:
00735 |             return status
00736 |     if "insufficient_seeds" in gates:
00737 |         return "no_robust_winner"
00738 |     if "severe_warnings" in gates:
00739 |         return "no_robust_winner"
00740 |     return "no_robust_winner"
00741 | 
00742 | 
00743 | def passed_gates(gates_failed: list[str]) -> list[str]:
00744 |     failed = set(gates_failed)
00745 |     gate_map = {
00746 |         "complete_required_grid": {"invalid_incomplete_grid", "missing_model", "missing_primary_metric"},
00747 |         "valid_artifact_contract": {"invalid_incompatible_artifacts"},
00748 |         "required_provenance": {"invalid_missing_provenance"},
00749 |         "same_frozen_split": {"invalid_incompatible_splits"},
00750 |         "deeph_split_audit_pass": {"invalid_unverified_deeph_split"},
00751 |         "adapter_equivalence_pass": {"deeph_adapter_equivalence_not_proven"},
00752 |         "production_fail_closed_metrics": {"metric_fail_policy_diagnostic_only"},
00753 |         "recommendation_grade_metric": {"invalid_metric_policy"},
00754 |         "valid_prediction_format": {"invalid_prediction_format"},
00755 |         "no_severe_warnings": {"severe_warnings"},
00756 |         "seed_stability": {"unstable_across_seeds", "insufficient_seeds"},
00757 |     }
00758 |     return [gate for gate, blockers in gate_map.items() if not (failed & blockers)]
00759 | 
00760 | 
00761 | def choose_primary_metric(rows: list[dict[str, Any]]) -> str | None:
00762 |     for metric in PRIMARY_METRIC_PRIORITY:
00763 |         values_by_model: dict[str, list[float]] = defaultdict(list)
00764 |         for row in rows:
00765 |             if row.get("model") not in MODELS:
00766 |                 continue
00767 |             value = finite_metric(row, metric)
00768 |             if value is not None:
00769 |                 values_by_model[str(row["model"])].append(value)
00770 |         if all(values_by_model.get(model) for model in MODELS):
00771 |             return metric
00772 |     return None
00773 | 
00774 | 
00775 | def rank_metric_groups(rows: list[dict[str, Any]], metric: str, *, scope: str = "dataset") -> list[dict[str, Any]]:
00776 |     groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
00777 |     for row in rows:
00778 |         value = finite_metric(row, metric)
00779 |         if value is None:
00780 |             continue
00781 |         dataset_key = str(row.get("dataset_id") or "unknown") if scope == "dataset" else "all"
00782 |         groups[(dataset_key, str(row.get("model")), str(row.get("config_id")))].append(row)
00783 |     summaries: list[dict[str, Any]] = []
00784 |     for (dataset_id, model, config_id), group_rows in groups.items():
00785 |         values = [finite_metric(row, metric) for row in group_rows]
00786 |         clean = [value for value in values if value is not None]
00787 |         seeds = [row.get("seed") for row in group_rows]
00788 |         severe = any(row.get("warning_status") == "severe" for row in group_rows)
00789 |         eligible = all(row_is_robust_eligible(row, metric) for row in group_rows)
00790 |         group_gate_failures = sorted({failure for row in group_rows for failure in row_gate_failures(row, metric)})
00791 |         seed_status = seed_stability_status(seeds, [model], has_severe_warning=severe)
00792 |         sample = group_rows[0]
00793 |         summaries.append(
00794 |             {
00795 |                 "scope": scope,
00796 |                 "dataset_id": dataset_id,
00797 |                 "model": model,
00798 |                 "config_id": config_id,
00799 |                 "config_label": sample.get("config_label") or config_id,
00800 |                 "metric": metric,
00801 |                 "metric_column": source_metric(metric),
00802 |                 "mean": mean(clean),
00803 |                 "std": stddev(clean),
00804 |                 "n_samples": len(clean),
00805 |                 "n_seeds": len(set(str(seed) for seed in seeds)),
00806 |                 "valid_seed_count": len(valid_stability_seeds(seeds)),
00807 |                 "seed_stability_status": seed_status,
00808 |                 "comparability_status": sample.get("comparability_status"),
00809 |                 "scientific_status": "robust_candidate" if eligible and seed_status == "robust_candidate" else seed_status,
00810 |                 "warning_status": "severe" if severe else "ok",
00811 |                 "severe_warnings": [item for row in group_rows for item in row.get("severe_warnings") or []],
00812 |                 "adapter_equivalence_status": sample.get("adapter_equivalence_status"),
00813 |                 "raw_global_equivalence_proven": sample.get("raw_global_equivalence_proven"),
00814 |                 "split_audit_status": sample.get("split_audit_status"),
00815 |                 "run_dir": sample.get("run_dir"),
00816 |                 "prediction_dir": sample.get("prediction_dir"),
00817 |                 "dataset_compatibility_hash": sample.get("dataset_compatibility_hash"),
00818 |                 "frozen_split_hash": sample.get("frozen_split_hash"),
00819 |                 "total_time_seconds": mean([number(row.get("total_time_seconds")) for row in group_rows if finite(row.get("total_time_seconds"))]),
00820 |                 "robust_eligible": eligible,
00821 |                 "gates_failed": group_gate_failures,
00822 |                 "metric_policy_role": metric_policy_role(metric),
00823 |             }
00824 |         )
00825 |     lower = metric_lower_is_better(metric)
00826 |     ranked: list[dict[str, Any]] = []
00827 |     by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
00828 |     for item in summaries:
00829 |         by_bucket[(str(item["dataset_id"]), str(item["model"]))].append(item)
00830 |     for bucket, items in sorted(by_bucket.items()):
00831 |         sorted_items = sorted(
00832 |             items,
00833 |             key=lambda item: (
00834 |                 number(item["mean"]) if lower else -number(item["mean"]),
00835 |                 str(item["config_id"]),
00836 |             ),
00837 |         )
00838 |         previous_value: float | None = None
00839 |         previous_rank = 0
00840 |         for index, item in enumerate(sorted_items, start=1):
00841 |             value = number(item["mean"])
00842 |             rank = previous_rank if previous_value is not None and value == previous_value else index
00843 |             previous_value = value
00844 |             previous_rank = rank
00845 |             ranked.append({**item, "rank": rank, "tie": rank != index})
00846 |     return ranked
00847 | 
00848 | 
00849 | def build_metric_rankings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
00850 |     metrics = sorted({canonical_metric(key) for row in rows for key, value in row.items() if key.endswith("_mean") and finite(value)})
00851 |     rankings: list[dict[str, Any]] = []
00852 |     for metric in metrics:
00853 |         rankings.extend(rank_metric_groups(rows, metric, scope="dataset"))
00854 |     return rankings
00855 | 
00856 | 
00857 | def best_runs_by_model(rows: list[dict[str, Any]], primary_metric: str | None) -> list[dict[str, Any]]:
00858 |     if not primary_metric:
00859 |         return []
00860 |     dataset_rankings = [row for row in rank_metric_groups(rows, primary_metric, scope="dataset") if row["rank"] == 1]
00861 |     global_rankings = [row for row in rank_metric_groups(rows, primary_metric, scope="global") if row["rank"] == 1]
00862 |     return [*dataset_rankings, *global_rankings]
00863 | 
00864 | 
00865 | def pairwise_comparisons(best_rows: list[dict[str, Any]], *, baseline_model: str = "graph2mat") -> list[dict[str, Any]]:
00866 |     by_dataset_metric: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
00867 |     for row in best_rows:
00868 |         if row.get("scope") != "dataset":
00869 |             continue
00870 |         by_dataset_metric[(str(row.get("dataset_id")), str(row.get("metric")))][str(row.get("model"))] = row
00871 |     pairs: list[dict[str, Any]] = []
00872 |     challenger_model = "deeph" if baseline_model == "graph2mat" else "graph2mat"
00873 |     for (dataset_id, metric), methods in sorted(by_dataset_metric.items()):
00874 |         baseline = methods.get(baseline_model)
00875 |         challenger = methods.get(challenger_model)
00876 |         if not baseline or not challenger:
00877 |             pairs.append(
00878 |                 {
00879 |                     "dataset_id": dataset_id,
00880 |                     "metric": metric,
00881 |                     "status": "non_comparative",
00882 |                     "winner": None,
00883 |                     "reason": "Both Graph2Mat and DeepH are required for pairwise comparison.",
00884 |                 }
00885 |             )
00886 |             continue
00887 |         gates_failed: list[str] = []
00888 |         for key, label in (
00889 |             ("frozen_split_hash", "invalid_incompatible_splits"),
00890 |             ("dataset_compatibility_hash", "invalid_incompatible_artifacts"),
00891 |         ):
00892 |             if baseline.get(key) != challenger.get(key):
00893 |                 gates_failed.append(label)
00894 |         if not baseline.get("robust_eligible"):
00895 |             gates_failed.extend(baseline.get("gates_failed") or ["invalid_prediction_format"])
00896 |         if not challenger.get("robust_eligible"):
00897 |             gates_failed.extend(challenger.get("gates_failed") or ["invalid_prediction_format"])
00898 |         if challenger_model == "deeph" and not deeph_adapter_equivalence_proven(challenger):
00899 |             gates_failed.append("deeph_adapter_equivalence_not_proven")
00900 |         if baseline_model == "deeph" and not deeph_adapter_equivalence_proven(baseline):
00901 |             gates_failed.append("deeph_adapter_equivalence_not_proven")
00902 |         lower = metric_lower_is_better(metric)
00903 |         baseline_value = number(baseline.get("mean"))
00904 |         challenger_value = number(challenger.get("mean"))
00905 |         gates_failed = sorted(set(gates_failed))
00906 |         if gates_failed:
00907 |             status = status_from_gates(gates_failed)
00908 |             winner = None
00909 |             improvement = None
00910 |         else:
00911 |             challenger_better = challenger_value < baseline_value if lower else challenger_value > baseline_value
00912 |             winner = challenger_model if challenger_better else baseline_model
00913 |             status = "comparable"
00914 |             if baseline_value:
00915 |                 improvement = (
00916 |                     (baseline_value - challenger_value) / abs(baseline_value) * 100.0
00917 |                     if lower
00918 |                     else (challenger_value - baseline_value) / abs(baseline_value) * 100.0
00919 |                 )
00920 |             else:
00921 |                 improvement = None
00922 |         pairs.append(
00923 |             {
00924 |                 "dataset_id": dataset_id,
00925 |                 "metric": metric,
00926 |                 "baseline_model": baseline_model,
00927 |                 "challenger_model": challenger_model,
00928 |                 "baseline_config_id": baseline.get("config_id"),
00929 |                 "challenger_config_id": challenger.get("config_id"),
00930 |                 "baseline_value": baseline_value,
00931 |                 "challenger_value": challenger_value,
00932 |                 "absolute_difference": challenger_value - baseline_value,
00933 |                 "percent_improvement_challenger_vs_baseline": improvement,
00934 |                 "lower_is_better": lower,
00935 |                 "winner": winner,
00936 |                 "status": status,
00937 |                 "gates_failed": gates_failed,
00938 |             }
00939 |         )
00940 |     return pairs
00941 | 
00942 | 
00943 | def build_recommendation(
00944 |     *,
00945 |     rows: list[dict[str, Any]],
00946 |     best_rows: list[dict[str, Any]],
00947 |     pairs: list[dict[str, Any]],
00948 |     primary_metric: str | None,
00949 |     minimum_robust_seeds: int = 3,
00950 | ) -> dict[str, Any]:
00951 |     models_seen = sorted({str(row.get("model")) for row in rows if row.get("model")})
00952 |     gates_failed: list[str] = []
00953 |     if set(models_seen) != set(MODELS):
00954 |         status = "invalid_incomplete_grid"
00955 |         return {
00956 |             "status": status,
00957 |             "scientific_status": status,
00958 |             "winner": None,
00959 |             "winning_model": None,
00960 |             "primary_metric": primary_metric,
00961 |             "reason": "Both graph2mat and deeph runs are required.",
00962 |             "gates_passed": [],
00963 |             "gates_failed": ["missing_model"],
00964 |             "models_seen": models_seen,
00965 |             "status_values": sorted(RECOMMENDATION_STATUS_VALUES),
00966 |         }
00967 |     if not primary_metric:
00968 |         status = "invalid_incomplete_grid"
00969 |         return {
00970 |             "status": status,
00971 |             "scientific_status": status,
00972 |             "winner": None,
00973 |             "winning_model": None,
00974 |             "primary_metric": None,
00975 |             "reason": "No shared finite primary metric is available.",
00976 |             "gates_passed": [],
00977 |             "gates_failed": ["missing_primary_metric"],
00978 |             "status_values": sorted(RECOMMENDATION_STATUS_VALUES),
00979 |         }
00980 |     if metric_policy_role(primary_metric) == "diagnostic_only":
00981 |         gates_failed.append("invalid_metric_policy")
00982 |     comparable_pairs = [pair for pair in pairs if pair.get("metric") == primary_metric and pair.get("status") == "comparable"]
00983 |     if not comparable_pairs:
00984 |         pair_failures = sorted({failure for pair in pairs for failure in pair.get("gates_failed") or []})
00985 |         gates_failed.extend(pair_failures or ["invalid_incomplete_grid"])
00986 |     gates_failed.extend(failure for row in rows for failure in row_gate_failures(row, primary_metric))
00987 |     severe = [item for row in rows for item in row.get("severe_warnings") or []]
00988 |     if severe:
00989 |         gates_failed.append("severe_warnings")
00990 |     if any(row.get("fail_open_metric_outputs") or row.get("metric_fail_policy") == METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY for row in rows):
00991 |         gates_failed.append("metric_fail_policy_diagnostic_only")
00992 |     if any(normalize_model(row.get("model")) == "deeph" and not deeph_adapter_equivalence_proven(row) for row in rows):
00993 |         gates_failed.append("deeph_adapter_equivalence_not_proven")
00994 |     primary_best = [row for row in best_rows if row.get("scope") == "global" and row.get("metric") == primary_metric]
00995 |     valid_seed_counts = [int(row.get("valid_seed_count") or 0) for row in primary_best]
00996 |     if valid_seed_counts and min(valid_seed_counts) < minimum_robust_seeds:
00997 |         gates_failed.append("insufficient_seeds")
00998 |     if any(row.get("seed_stability_status") == "unstable" for row in primary_best):
00999 |         gates_failed.append("unstable_across_seeds")
01000 | 
01001 |     best_by_model = {str(row.get("model")): row for row in primary_best}
01002 |     if set(best_by_model) != set(MODELS):
01003 |         gates_failed.append("invalid_incomplete_grid")
01004 | 
01005 |     winner = None
01006 |     if best_by_model:
01007 |         lower = metric_lower_is_better(primary_metric)
01008 |         ordered = sorted(
01009 |             best_by_model.values(),
01010 |             key=lambda row: (
01011 |                 number(row.get("mean")) if lower else -number(row.get("mean")),
01012 |                 str(row.get("model")),
01013 |             ),
01014 |         )
01015 |         winner = str(ordered[0].get("model")) if ordered else None
01016 | 
01017 |     gates_failed = sorted(set(gates_failed))
01018 |     hard_failures = [failure for failure in gates_failed if failure not in {"insufficient_seeds"}]
01019 |     if hard_failures:
01020 |         status = status_from_gates(hard_failures)
01021 |         scientific_status = status
01022 |         winner = None
01023 |     elif "insufficient_seeds" in gates_failed:
01024 |         status = f"exploratory_{winner}_win" if winner in MODELS else "no_robust_winner"
01025 |         scientific_status = "exploratory_only"
01026 |     else:
01027 |         status = f"robust_{winner}_win" if winner in MODELS else "no_robust_winner"
01028 |         scientific_status = "robust_comparison" if winner in MODELS else "not_scientifically_valid"
01029 | 
01030 |     return {
01031 |         "status": status,
01032 |         "scientific_status": scientific_status,
01033 |         "winner": winner if status.startswith(("robust_", "exploratory_")) else None,
01034 |         "winning_model": winner if status.startswith(("robust_", "exploratory_")) else None,
01035 |         "winning_config_id": (best_by_model.get(winner) or {}).get("config_id") if winner else None,
01036 |         "winning_dataset_id": (best_by_model.get(winner) or {}).get("dataset_id") if winner else None,
01037 |         "primary_metric": primary_metric,
01038 |         "primary_metric_column": source_metric(primary_metric) if primary_metric else None,
01039 |         "reason": (
01040 |             f"{winner} has the best {primary_metric}, but seed count makes this exploratory."
01041 |             if status.startswith("exploratory_")
01042 |             else f"{winner} has the best {primary_metric} and all robust gates passed."
01043 |             if status.startswith("robust_")
01044 |             else "Scientific gates prevent a robust winner."
01045 |         ),
01046 |         "limitations": gates_failed,
01047 |         "gates_passed": passed_gates(gates_failed),
01048 |         "gates_failed": gates_failed,
01049 |         "best_graph2mat": best_by_model.get("graph2mat"),
01050 |         "best_deeph": best_by_model.get("deeph"),
01051 |         "pairwise_evidence": comparable_pairs,
01052 |         "seed_stability": {row.get("model"): row.get("seed_stability_status") for row in primary_best},
01053 |         "metric_policy": metric_policy_role(primary_metric) if primary_metric else "missing",
01054 |         "comparability_status": "valid" if not hard_failures else ("diagnostic_only" if status == "diagnostic_only" else status),
01055 |         "adapter_equivalence_status": deeph_adapter_status(rows),
01056 |         "equivalence_status": deeph_equivalence_status(rows),
01057 |         "split_audit_status": split_audit_status(rows),
01058 |         "status_values": sorted(RECOMMENDATION_STATUS_VALUES),
01059 |         "warnings": severe,
01060 |     }
01061 | 
01062 | 
01063 | def pareto_frontier(rows: list[dict[str, Any]], primary_metric: str | None) -> list[dict[str, Any]]:
01064 |     if not primary_metric:
01065 |         return []
01066 |     candidates: list[dict[str, Any]] = []
01067 |     for row in rows:
01068 |         value = finite_metric(row, primary_metric)
01069 |         time_value = number(row.get("total_time_seconds"))
01070 |         if value is None:
01071 |             continue
01072 |         if not row_is_robust_eligible(row, primary_metric):
01073 |             continue
01074 |         candidates.append(
01075 |             {
01076 |                 "model": row.get("model"),
01077 |                 "dataset_id": row.get("dataset_id"),
01078 |                 "config_id": row.get("config_id"),
01079 |                 "metric": primary_metric,
01080 |                 "metric_value": value,
01081 |                 "total_time_seconds": time_value if math.isfinite(time_value) else None,
01082 |                 "train_time_seconds": row.get("training_time_seconds"),
01083 |                 "predict_time_seconds": row.get("prediction_time_seconds"),
01084 |                 "preprocess_time_seconds": row.get("preprocess_time_seconds"),
01085 |                 "gpu_hours_total": row.get("gpu_hours_total"),
01086 |                 "gpu_hours_to_best_validation": row.get("gpu_hours_to_best_validation"),
01087 |                 "peak_gpu_memory_mb": row.get("peak_gpu_memory_mb"),
01088 |                 "samples_per_second": row.get("samples_per_second"),
01089 |                 "matrix_blocks_per_second": row.get("matrix_blocks_per_second"),
01090 |                 "timing_reliability_status": "available" if math.isfinite(time_value) else "timing_unavailable",
01091 |             }
01092 |         )
01093 |     frontier: list[dict[str, Any]] = []
01094 |     for candidate in candidates:
01095 |         time_a = number(candidate.get("total_time_seconds"))
01096 |         dominated = False
01097 |         for other in candidates:
01098 |             if other is candidate:
01099 |                 continue
01100 |             time_b = number(other.get("total_time_seconds"))
01101 |             if not math.isfinite(time_a) or not math.isfinite(time_b):
01102 |                 continue
01103 |             no_worse = other["metric_value"] <= candidate["metric_value"] and time_b <= time_a
01104 |             strictly_better = other["metric_value"] < candidate["metric_value"] or time_b < time_a
01105 |             if no_worse and strictly_better:
01106 |                 dominated = True
01107 |                 break
01108 |         if not dominated:
01109 |             frontier.append(candidate)
01110 |     return [
01111 |         {**row, "pareto_rank": index + 1, "pareto_status": "robust_frontier"}
01112 |         for index, row in enumerate(sorted(frontier, key=lambda item: (item["metric_value"], number(item.get("total_time_seconds")), str(item.get("config_id")))))
01113 |     ]
01114 | 
01115 | 
01116 | def rank_graph2mat_deeph_runs(
01117 |     *,
01118 |     run_root: Path,
01119 |     output_dir: Path | None = None,
01120 |     training_sweep_manifest_path: Path | None = None,
01121 |     common_metrics_manifest_path: Path | None = None,
01122 |     dataset_root: Path | None = None,
01123 |     frozen_split_manifest_path: Path | None = None,
01124 |     dataset_manifest_path: Path | None = None,
01125 |     minimum_robust_seeds: int = 3,
01126 | ) -> dict[str, Any]:
01127 |     run_root = Path(run_root)
01128 |     output_dir = Path(output_dir or run_root / "summary" / "ranking")
01129 |     training_sweep_manifest_path = training_sweep_manifest_path or run_root / "sweep" / "training_sweep_manifest.json"
01130 |     common_metrics_manifest_path = common_metrics_manifest_path or run_root / "common_metrics" / "summary" / "common_summary.json"
01131 |     rows = load_metric_rows(
01132 |         training_sweep_manifest_path=training_sweep_manifest_path if training_sweep_manifest_path.exists() else None,
01133 |         common_metrics_manifest_path=common_metrics_manifest_path if common_metrics_manifest_path.exists() else None,
01134 |         dataset_root=dataset_root,
01135 |         frozen_split_manifest_path=frozen_split_manifest_path,
01136 |         dataset_manifest_path=dataset_manifest_path,
01137 |     )
01138 |     primary_metric = choose_primary_metric(rows)
01139 |     metric_rankings = build_metric_rankings(rows)
01140 |     best_rows = best_runs_by_model(rows, primary_metric)
01141 |     model_config_rankings = [row for row in metric_rankings if row.get("metric") == primary_metric] if primary_metric else []
01142 |     pairs = pairwise_comparisons(best_rows)
01143 |     recommendation = build_recommendation(
01144 |         rows=rows,
01145 |         best_rows=best_rows,
01146 |         pairs=pairs,
01147 |         primary_metric=primary_metric,
01148 |         minimum_robust_seeds=minimum_robust_seeds,
01149 |     )
01150 |     pareto = pareto_frontier(rows, primary_metric)
01151 |     manifest = {
01152 |         "schema": SCHEMA,
01153 |         "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
01154 |         "run_root": str(run_root),
01155 |         "output_dir": str(output_dir),
01156 |         "primary_metric": primary_metric,
01157 |         "metric_rows_count": len(rows),
01158 |         "ranking_outputs": {
01159 |             "best_runs_by_model": str(output_dir / "best_runs_by_model.json"),
01160 |             "model_config_rankings": str(output_dir / "model_config_rankings.json"),
01161 |             "metric_rankings_by_model": str(output_dir / "metric_rankings_by_model.json"),
01162 |             "pairwise_graph2mat_vs_deeph": str(output_dir / "pairwise_graph2mat_vs_deeph.json"),
01163 |             "pareto_accuracy_cost": str(output_dir / "pareto_accuracy_cost.json"),
01164 |             "recommendation": str(output_dir / "recommendation.json"),
01165 |             "best_overall": str(output_dir / "best_overall.json"),
01166 |         },
01167 |         "recommendation": recommendation,
01168 |         "best_runs_by_model": best_rows,
01169 |         "pairwise_graph2mat_vs_deeph": pairs,
01170 |         "pareto_accuracy_cost": pareto,
01171 |     }
01172 |     write_csv(output_dir / "normalized_run_metrics.csv", rows)
01173 |     write_json(output_dir / "normalized_run_metrics.json", {"rows": rows})
01174 |     write_csv(output_dir / "best_runs_by_model.csv", best_rows)
01175 |     write_json(output_dir / "best_runs_by_model.json", {"rows": best_rows})
01176 |     write_csv(output_dir / "model_config_rankings.csv", model_config_rankings)
01177 |     write_json(output_dir / "model_config_rankings.json", {"rows": model_config_rankings})
01178 |     write_csv(output_dir / "metric_rankings_by_model.csv", metric_rankings)
01179 |     write_json(output_dir / "metric_rankings_by_model.json", {"rows": metric_rankings})
01180 |     write_csv(output_dir / "pairwise_graph2mat_vs_deeph.csv", pairs)
01181 |     write_json(output_dir / "pairwise_graph2mat_vs_deeph.json", {"rows": pairs})
01182 |     write_csv(output_dir / "pareto_accuracy_cost.csv", pareto)
01183 |     write_json(output_dir / "pareto_accuracy_cost.json", {"rows": pareto})
01184 |     write_json(output_dir / "recommendation.json", recommendation)
01185 |     write_json(output_dir / "best_overall.json", {"recommendation": recommendation, "best_runs_by_model": best_rows})
01186 |     write_json(output_dir / "ranking_summary.json", manifest)
01187 |     return manifest
01188 | 
01189 | 
01190 | def parse_args() -> argparse.Namespace:
01191 |     parser = argparse.ArgumentParser(description=__doc__)
01192 |     parser.add_argument("run_root", type=Path)
01193 |     parser.add_argument("--output-dir", type=Path, default=None)
01194 |     parser.add_argument("--training-sweep-manifest", type=Path, default=None)
01195 |     parser.add_argument("--common-metrics-manifest", type=Path, default=None)
01196 |     parser.add_argument("--dataset-root", type=Path, default=None)
01197 |     parser.add_argument("--frozen-split-manifest", type=Path, default=None)
01198 |     parser.add_argument("--dataset-manifest", type=Path, default=None)
01199 |     parser.add_argument("--minimum-robust-seeds", type=int, default=3)
01200 |     return parser.parse_args()
01201 | 
01202 | 
01203 | def main() -> None:
01204 |     args = parse_args()
01205 |     manifest = rank_graph2mat_deeph_runs(
01206 |         run_root=args.run_root,
01207 |         output_dir=args.output_dir,
01208 |         training_sweep_manifest_path=args.training_sweep_manifest,
01209 |         common_metrics_manifest_path=args.common_metrics_manifest,
01210 |         dataset_root=args.dataset_root,
01211 |         frozen_split_manifest_path=args.frozen_split_manifest,
01212 |         dataset_manifest_path=args.dataset_manifest,
01213 |         minimum_robust_seeds=args.minimum_robust_seeds,
01214 |     )
01215 |     print(json.dumps(json_safe({"status": manifest["recommendation"]["status"], "output_dir": manifest["output_dir"]}), ensure_ascii=False))
01216 | 
01217 | 
01218 | if __name__ == "__main__":
01219 |     main()
```

## `Comparison/scripts/g2m_deeph_final_stats.py`

SHA-256: `d970fa6b8b9e10470077e09711e45e190b7037728271ebd1c0b9c2b2b8bb9db5`

```py
00001 | #!/usr/bin/env python3
00002 | """Final statistical aggregation and winner gates for G2M-vs-DeepH benchmarks."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import argparse
00007 | import csv
00008 | import json
00009 | import math
00010 | import random
00011 | import statistics
00012 | import time
00013 | from pathlib import Path
00014 | from typing import Any
00015 | 
00016 | from deeph_prediction_adapter import (
00017 |     EQUIVALENCE_PROVEN_RAW_GLOBAL,
00018 |     EQUIVALENCE_STATUS_PROVEN,
00019 |     EQUIVALENCE_STATUS_UNPROVEN,
00020 | )
00021 | 
00022 | 
00023 | FINAL_STATS_SCHEMA = "graph2mat_deeph_final_statistics_v1"
00024 | MODELS = ("graph2mat", "deeph")
00025 | FINAL_TEST_STAGE = "final_test"
00026 | TEST_SPLITS = {"test"}
00027 | VALID_MODES = {"min", "max"}
00028 | T_CRITICAL_95 = {
00029 |     1: 12.706,
00030 |     2: 4.303,
00031 |     3: 3.182,
00032 |     4: 2.776,
00033 |     5: 2.571,
00034 |     6: 2.447,
00035 |     7: 2.365,
00036 |     8: 2.306,
00037 |     9: 2.262,
00038 |     10: 2.228,
00039 |     11: 2.201,
00040 |     12: 2.179,
00041 |     13: 2.160,
00042 |     14: 2.145,
00043 |     15: 2.131,
00044 |     16: 2.120,
00045 |     17: 2.110,
00046 |     18: 2.101,
00047 |     19: 2.093,
00048 |     20: 2.086,
00049 |     21: 2.080,
00050 |     22: 2.074,
00051 |     23: 2.069,
00052 |     24: 2.064,
00053 |     25: 2.060,
00054 |     26: 2.056,
00055 |     27: 2.052,
00056 |     28: 2.048,
00057 |     29: 2.045,
00058 |     30: 2.042,
00059 | }
00060 | 
00061 | 
00062 | def finite_number(value: Any) -> float | None:
00063 |     try:
00064 |         number = float(value)
00065 |     except (TypeError, ValueError):
00066 |         return None
00067 |     return number if math.isfinite(number) else None
00068 | 
00069 | 
00070 | def json_safe(value: Any) -> Any:
00071 |     if isinstance(value, Path):
00072 |         return str(value)
00073 |     if isinstance(value, dict):
00074 |         return {str(key): json_safe(item) for key, item in value.items()}
00075 |     if isinstance(value, (list, tuple)):
00076 |         return [json_safe(item) for item in value]
00077 |     if isinstance(value, float):
00078 |         return value if math.isfinite(value) else None
00079 |     return value
00080 | 
00081 | 
00082 | def read_json(path: Path | None) -> dict[str, Any]:
00083 |     if path is None or not path.exists():
00084 |         return {}
00085 |     try:
00086 |         payload = json.loads(path.read_text(encoding="utf-8"))
00087 |     except (OSError, json.JSONDecodeError):
00088 |         return {}
00089 |     return payload if isinstance(payload, dict) else {}
00090 | 
00091 | 
00092 | def write_json(path: Path, payload: dict[str, Any]) -> None:
00093 |     path.parent.mkdir(parents=True, exist_ok=True)
00094 |     path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
00095 | 
00096 | 
00097 | def _csv_safe(row: dict[str, Any]) -> dict[str, Any]:
00098 |     safe: dict[str, Any] = {}
00099 |     for key, value in row.items():
00100 |         if isinstance(value, (dict, list)):
00101 |             safe[key] = json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False)
00102 |         else:
00103 |             safe[key] = json_safe(value)
00104 |     return safe
00105 | 
00106 | 
00107 | def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
00108 |     path.parent.mkdir(parents=True, exist_ok=True)
00109 |     safe_rows = [_csv_safe(row) for row in rows]
00110 |     fieldnames: list[str] = []
00111 |     for row in safe_rows:
00112 |         for key in row:
00113 |             if key not in fieldnames:
00114 |                 fieldnames.append(key)
00115 |     with path.open("w", encoding="utf-8", newline="") as handle:
00116 |         writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"])
00117 |         writer.writeheader()
00118 |         writer.writerows(safe_rows)
00119 | 
00120 | 
00121 | def mean(values: list[float]) -> float | None:
00122 |     return sum(values) / len(values) if values else None
00123 | 
00124 | 
00125 | def stddev(values: list[float]) -> float | None:
00126 |     if not values:
00127 |         return None
00128 |     return statistics.stdev(values) if len(values) > 1 else 0.0
00129 | 
00130 | 
00131 | def stderr(values: list[float]) -> float | None:
00132 |     if len(values) < 2:
00133 |         return None
00134 |     return (statistics.stdev(values) / math.sqrt(len(values)))
00135 | 
00136 | 
00137 | def confidence_interval(values: list[float], *, confidence_level: float = 0.95) -> dict[str, Any]:
00138 |     if len(values) < 2:
00139 |         return {
00140 |             "method": "unavailable",
00141 |             "confidence_level": confidence_level,
00142 |             "low": None,
00143 |             "high": None,
00144 |             "reason": "at least two seeds are required for a seed-level confidence interval",
00145 |         }
00146 |     center = mean(values)
00147 |     se = stderr(values)
00148 |     if center is None or se is None:
00149 |         return {"method": "unavailable", "confidence_level": confidence_level, "low": None, "high": None}
00150 |     if abs(confidence_level - 0.95) > 1e-9:
00151 |         critical = 1.96
00152 |         method = "normal_approximation"
00153 |     else:
00154 |         critical = T_CRITICAL_95.get(len(values) - 1, 1.96)
00155 |         method = "student_t_seed_mean"
00156 |     return {
00157 |         "method": method,
00158 |         "confidence_level": confidence_level,
00159 |         "low": center - critical * se,
00160 |         "high": center + critical * se,
00161 |     }
00162 | 
00163 | 
00164 | def metric_value(row: dict[str, Any], metric: str) -> float | None:
00165 |     for key in (
00166 |         "final_test_metric_value",
00167 |         "test_metric_value",
00168 |         metric,
00169 |         f"{metric}_mean",
00170 |         "metric_value",
00171 |     ):
00172 |         if key in row:
00173 |             number = finite_number(row.get(key))
00174 |             if number is not None:
00175 |                 return number
00176 |     metrics = row.get("final_test_metrics") or row.get("test_metrics")
00177 |     if isinstance(metrics, dict):
00178 |         for key in (metric, f"{metric}_mean"):
00179 |             number = finite_number(metrics.get(key))
00180 |             if number is not None:
00181 |                 return number
00182 |     return None
00183 | 
00184 | 
00185 | def protocol_stage(row: dict[str, Any]) -> str:
00186 |     return str(row.get("protocol_stage") or row.get("stage") or "").strip().lower()
00187 | 
00188 | 
00189 | def metric_split(row: dict[str, Any]) -> str:
00190 |     return str(row.get("metric_split") or row.get("split") or row.get("evaluation_split") or "").strip().lower()
00191 | 
00192 | 
00193 | def final_test_row(row: dict[str, Any], metric: str) -> bool:
00194 |     if str(row.get("status") or row.get("run_status") or "completed") not in {"completed", "ok"}:
00195 |         return False
00196 |     if metric_value(row, metric) is None:
00197 |         return False
00198 |     stage = protocol_stage(row)
00199 |     split = metric_split(row)
00200 |     return stage == FINAL_TEST_STAGE or split in TEST_SPLITS
00201 | 
00202 | 
00203 | def protocol_violations(rows: list[dict[str, Any]]) -> list[str]:
00204 |     violations: list[str] = []
00205 |     for index, row in enumerate(rows):
00206 |         if not isinstance(row, dict):
00207 |             continue
00208 |         split = metric_split(row)
00209 |         stage = protocol_stage(row)
00210 |         if split in TEST_SPLITS and stage != FINAL_TEST_STAGE:
00211 |             violations.append(
00212 |                 f"row {row.get('config_id') or index} contains test metrics outside final_test stage"
00213 |             )
00214 |     return violations
00215 | 
00216 | 
00217 | def seed_value(row: dict[str, Any]) -> Any:
00218 |     common = row.get("common") if isinstance(row.get("common"), dict) else {}
00219 |     overrides = row.get("overrides") if isinstance(row.get("overrides"), dict) else {}
00220 |     return row.get("seed", common.get("seed", overrides.get("seed_everything", overrides.get("seed"))))
00221 | 
00222 | 
00223 | def selected_config_id(row: dict[str, Any]) -> str:
00224 |     """Return the validation-selected config identity for final seed grouping."""
00225 | 
00226 |     for key in ("selected_config_id", "base_config_id", "parent_config_id"):
00227 |         value = str(row.get(key) or "").strip()
00228 |         if value:
00229 |             return value
00230 |     source = row.get("source_selected_config") if isinstance(row.get("source_selected_config"), dict) else {}
00231 |     value = str(source.get("config_id") or "").strip()
00232 |     if value:
00233 |         return value
00234 |     return str(row.get("config_id") or "").strip()
00235 | 
00236 | 
00237 | def compute_field(row: dict[str, Any], key: str) -> float | None:
00238 |     value = finite_number(row.get(key))
00239 |     if value is not None:
00240 |         return value
00241 |     telemetry = row.get("telemetry") if isinstance(row.get("telemetry"), dict) else {}
00242 |     return finite_number(telemetry.get(key))
00243 | 
00244 | 
00245 | def per_system_values(row: dict[str, Any], metric: str) -> list[float]:
00246 |     values: list[float] = []
00247 |     for key in ("per_system_metrics", "sample_metrics", "system_metrics"):
00248 |         raw = row.get(key)
00249 |         if not isinstance(raw, list):
00250 |             continue
00251 |         for item in raw:
00252 |             if not isinstance(item, dict):
00253 |                 continue
00254 |             value = metric_value(item, metric)
00255 |             if value is not None:
00256 |                 values.append(value)
00257 |     return values
00258 | 
00259 | 
00260 | def _bool_field(value: Any) -> bool:
00261 |     if isinstance(value, bool):
00262 |         return value
00263 |     if isinstance(value, (int, float)):
00264 |         return bool(value)
00265 |     text = str(value or "").strip().lower()
00266 |     return text in {"1", "true", "yes", "y", "on"}
00267 | 
00268 | 
00269 | def _candidate_adapter_manifest_paths(row: dict[str, Any]) -> list[Path]:
00270 |     candidates: list[Path] = []
00271 |     for key in ("adapter_manifest_path", "deeph_adapter_manifest_path"):
00272 |         value = str(row.get(key) or "").strip()
00273 |         if value:
00274 |             candidates.append(Path(value))
00275 | 
00276 |     run_root = str(row.get("run_root") or "").strip()
00277 |     if run_root:
00278 |         candidates.append(Path(run_root) / "deeph" / "inference" / "adapter_manifest.json")
00279 | 
00280 |     deeph_manifest = str(row.get("deeph_manifest_path") or "").strip()
00281 |     if deeph_manifest:
00282 |         deeph_root = Path(deeph_manifest).parent
00283 |         candidates.append(deeph_root / "inference" / "adapter_manifest.json")
00284 | 
00285 |     metrics_path = str(row.get("metrics_path") or row.get("final_test_metrics_path") or "").strip()
00286 |     if metrics_path:
00287 |         path = Path(metrics_path)
00288 |         for parent in path.parents:
00289 |             if parent.name == "metrics":
00290 |                 candidates.append(parent.parent / "adapter_manifest.json")
00291 |                 break
00292 | 
00293 |     unique: list[Path] = []
00294 |     seen: set[str] = set()
00295 |     for path in candidates:
00296 |         key = str(path)
00297 |         if key not in seen:
00298 |             unique.append(path)
00299 |             seen.add(key)
00300 |     return unique
00301 | 
00302 | 
00303 | def _first_string(values: Any) -> str:
00304 |     if isinstance(values, list):
00305 |         for value in values:
00306 |             text = str(value or "").strip()
00307 |             if text:
00308 |                 return text
00309 |     return str(values or "").strip()
00310 | 
00311 | 
00312 | def _deeph_equivalence_fields(row: dict[str, Any]) -> dict[str, Any]:
00313 |     """Resolve DeepH equivalence from row fields, falling back to adapter manifests."""
00314 | 
00315 |     status = str(row.get("adapter_equivalence_status") or row.get("deeph_adapter_equivalence_status") or "").strip()
00316 |     equivalence_status = str(row.get("equivalence_status") or row.get("deeph_equivalence_status") or "").strip()
00317 |     comparability_status = str(row.get("comparability_status") or row.get("deeph_comparability_status") or "").strip()
00318 |     diagnostic_only = _bool_field(row.get("diagnostic_only") or row.get("deeph_diagnostic_only"))
00319 |     source = "row" if status or equivalence_status else ""
00320 | 
00321 |     needs_manifest = status != EQUIVALENCE_PROVEN_RAW_GLOBAL or equivalence_status != EQUIVALENCE_STATUS_PROVEN
00322 |     if needs_manifest:
00323 |         for path in _candidate_adapter_manifest_paths(row):
00324 |             payload = read_json(path)
00325 |             if not payload:
00326 |                 continue
00327 |             gate = payload.get("equivalence_gate") if isinstance(payload.get("equivalence_gate"), dict) else {}
00328 |             manifest_status = _first_string(payload.get("adapter_equivalence_statuses"))
00329 |             manifest_equivalence_status = _first_string(payload.get("equivalence_statuses"))
00330 |             proven_count = finite_number(payload.get("raw_global_equivalence_proven_count")) or 0.0
00331 |             robust_allowed = gate.get("robust_claim_allowed") is True
00332 | 
00333 |             if robust_allowed and proven_count > 0:
00334 |                 status = EQUIVALENCE_PROVEN_RAW_GLOBAL
00335 |                 equivalence_status = EQUIVALENCE_STATUS_PROVEN
00336 |                 comparability_status = comparability_status or "valid"
00337 |                 diagnostic_only = False
00338 |                 source = str(path)
00339 |                 break
00340 | 
00341 |             status = status or manifest_status
00342 |             equivalence_status = equivalence_status or manifest_equivalence_status
00343 |             if gate:
00344 |                 diagnostic_only = gate.get("diagnostic_only") is True
00345 |             source = str(path)
00346 |             break
00347 | 
00348 |     if not equivalence_status:
00349 |         equivalence_status = (
00350 |             EQUIVALENCE_STATUS_PROVEN if status == EQUIVALENCE_PROVEN_RAW_GLOBAL else EQUIVALENCE_STATUS_UNPROVEN
00351 |         )
00352 |     return {
00353 |         "adapter_equivalence_status": status,
00354 |         "equivalence_status": equivalence_status,
00355 |         "comparability_status": comparability_status,
00356 |         "diagnostic_only": diagnostic_only,
00357 |         "source": source,
00358 |     }
00359 | 
00360 | 
00361 | def bootstrap_ci(values: list[float], *, iterations: int = 1000, confidence_level: float = 0.95, seed: int = 0) -> dict[str, Any]:
00362 |     if len(values) < 2:
00363 |         return {
00364 |             "method": "unavailable",
00365 |             "confidence_level": confidence_level,
00366 |             "low": None,
00367 |             "high": None,
00368 |             "reason": "per-system metrics unavailable or too small for bootstrap",
00369 |         }
00370 |     rng = random.Random(seed)
00371 |     means: list[float] = []
00372 |     for _ in range(max(1, int(iterations))):
00373 |         sample = [values[rng.randrange(len(values))] for _ in values]
00374 |         means.append(sum(sample) / len(sample))
00375 |     means.sort()
00376 |     alpha = max(0.0, min(1.0, 1.0 - confidence_level))
00377 |     low_index = min(max(int((alpha / 2.0) * len(means)), 0), len(means) - 1)
00378 |     high_index = min(max(int((1.0 - alpha / 2.0) * len(means)) - 1, 0), len(means) - 1)
00379 |     return {
00380 |         "method": "bootstrap_per_system_mean",
00381 |         "confidence_level": confidence_level,
00382 |         "iterations": int(iterations),
00383 |         "low": means[low_index],
00384 |         "high": means[high_index],
00385 |     }
00386 | 
00387 | 
00388 | def aggregate_final_seed_metrics(
00389 |     rows: list[dict[str, Any]],
00390 |     *,
00391 |     metric: str,
00392 |     expected_seeds: list[int] | None = None,
00393 |     confidence_level: float = 0.95,
00394 |     bootstrap_iterations: int = 1000,
00395 | ) -> list[dict[str, Any]]:
00396 |     groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
00397 |     for row in rows:
00398 |         if final_test_row(row, metric):
00399 |             groups.setdefault(
00400 |                 (
00401 |                     str(row.get("model") or ""),
00402 |                     str(row.get("dataset_id") or ""),
00403 |                     selected_config_id(row),
00404 |                 ),
00405 |                 [],
00406 |             ).append(row)
00407 |     expected = set(expected_seeds or [])
00408 |     summaries: list[dict[str, Any]] = []
00409 |     for (model, dataset_id, config_id), group_rows in sorted(groups.items()):
00410 |         values = [value for row in group_rows if (value := metric_value(row, metric)) is not None]
00411 |         seeds = sorted({seed_value(row) for row in group_rows if seed_value(row) not in (None, "")}, key=str)
00412 |         gpu_hours = [value for row in group_rows if (value := compute_field(row, "gpu_hours_total")) is not None]
00413 |         peak_memory = [value for row in group_rows if (value := compute_field(row, "peak_gpu_memory_mb")) is not None]
00414 |         samples_per_second = [value for row in group_rows if (value := compute_field(row, "samples_per_second")) is not None]
00415 |         matrix_blocks_per_second = [
00416 |             value for row in group_rows if (value := compute_field(row, "matrix_blocks_per_second")) is not None
00417 |         ]
00418 |         per_system = [value for row in group_rows for value in per_system_values(row, metric)]
00419 |         missing_expected = sorted(expected - {int(seed) for seed in seeds if isinstance(seed, int) or str(seed).isdigit()})
00420 |         diagnostic_reasons: list[str] = []
00421 |         if model == "deeph":
00422 |             for row in group_rows:
00423 |                 equivalence = _deeph_equivalence_fields(row)
00424 |                 status = str(equivalence.get("adapter_equivalence_status") or "")
00425 |                 equivalence_status = str(equivalence.get("equivalence_status") or "")
00426 |                 if status != EQUIVALENCE_PROVEN_RAW_GLOBAL or equivalence_status != EQUIVALENCE_STATUS_PROVEN:
00427 |                     diagnostic_reasons.append(
00428 |                         "deeph adapter equivalence not proven: "
00429 |                         f"adapter={status or 'missing'} equivalence={equivalence_status or 'missing'}"
00430 |                     )
00431 |                     break
00432 |                 if bool(equivalence.get("diagnostic_only")) or str(equivalence.get("comparability_status") or "").lower() == "diagnostic_only":
00433 |                     diagnostic_reasons.append("deeph metrics are diagnostic_only")
00434 |                     break
00435 |         summaries.append(
00436 |             {
00437 |                 "model": model,
00438 |                 "dataset_id": dataset_id,
00439 |                 "selected_config_id": config_id,
00440 |                 "config_id": config_id,
00441 |                 "final_run_config_ids": sorted(
00442 |                     {str(row.get("config_id") or "") for row in group_rows if str(row.get("config_id") or "").strip()}
00443 |                 ),
00444 |                 "metric": metric,
00445 |                 "mean": mean(values),
00446 |                 "std": stddev(values),
00447 |                 "stderr": stderr(values),
00448 |                 "n_seeds_completed": len(seeds),
00449 |                 "expected_seed_count": len(expected) if expected else None,
00450 |                 "completed_seeds": seeds,
00451 |                 "missing_seeds": missing_expected,
00452 |                 "confidence_interval": confidence_interval(values, confidence_level=confidence_level),
00453 |                 "bootstrap_ci": bootstrap_ci(
00454 |                     per_system,
00455 |                     iterations=bootstrap_iterations,
00456 |                     confidence_level=confidence_level,
00457 |                     seed=0,
00458 |                 )
00459 |                 if per_system
00460 |                 else {
00461 |                     "method": "unavailable",
00462 |                     "confidence_level": confidence_level,
00463 |                     "low": None,
00464 |                     "high": None,
00465 |                     "reason": "per-system metrics unavailable",
00466 |                 },
00467 |                 "gpu_hours_mean": mean(gpu_hours),
00468 |                 "gpu_hours_std": stddev(gpu_hours),
00469 |                 "peak_gpu_memory_mb_mean": mean(peak_memory),
00470 |                 "peak_gpu_memory_mb_std": stddev(peak_memory),
00471 |                 "samples_per_second_mean": mean(samples_per_second),
00472 |                 "samples_per_second_std": stddev(samples_per_second),
00473 |                 "matrix_blocks_per_second_mean": mean(matrix_blocks_per_second),
00474 |                 "matrix_blocks_per_second_std": stddev(matrix_blocks_per_second),
00475 |                 "robust_claim_allowed_by_comparability": not diagnostic_reasons,
00476 |                 "diagnostic_only_reason": "; ".join(sorted(set(diagnostic_reasons))),
00477 |             }
00478 |         )
00479 |     return summaries
00480 | 
00481 | 
00482 | def _metric_better(a: float, b: float, *, mode: str) -> bool:
00483 |     return a < b if mode == "min" else a > b
00484 | 
00485 | 
00486 | def _ci_separated(best: dict[str, Any], other: dict[str, Any], *, mode: str) -> bool:
00487 |     best_ci = best.get("confidence_interval") if isinstance(best.get("confidence_interval"), dict) else {}
00488 |     other_ci = other.get("confidence_interval") if isinstance(other.get("confidence_interval"), dict) else {}
00489 |     best_low = finite_number(best_ci.get("low"))
00490 |     best_high = finite_number(best_ci.get("high"))
00491 |     other_low = finite_number(other_ci.get("low"))
00492 |     other_high = finite_number(other_ci.get("high"))
00493 |     if None in {best_low, best_high, other_low, other_high}:
00494 |         return False
00495 |     return best_high < other_low if mode == "min" else best_low > other_high
00496 | 
00497 | 
00498 | def pareto_frontier(summary_rows: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
00499 |     rows = []
00500 |     for row in summary_rows:
00501 |         value = finite_number(row.get("mean"))
00502 |         cost = finite_number(row.get("gpu_hours_mean"))
00503 |         if value is None:
00504 |             continue
00505 |         item = {
00506 |             "model": row.get("model"),
00507 |             "dataset_id": row.get("dataset_id"),
00508 |             "selected_config_id": row.get("selected_config_id") or row.get("config_id"),
00509 |             "config_id": row.get("config_id") or row.get("selected_config_id"),
00510 |             "metric": row.get("metric"),
00511 |             "metric_value": value,
00512 |             "gpu_hours": cost,
00513 |             "peak_gpu_memory_mb": row.get("peak_gpu_memory_mb_mean"),
00514 |             "pareto_dominated": False,
00515 |             "pareto_status": "frontier",
00516 |         }
00517 |         rows.append(item)
00518 |     for row in rows:
00519 |         cost_a = finite_number(row.get("gpu_hours"))
00520 |         value_a = finite_number(row.get("metric_value"))
00521 |         if cost_a is None or value_a is None:
00522 |             row["pareto_status"] = "cost_unavailable"
00523 |             continue
00524 |         dominated = False
00525 |         for other in rows:
00526 |             if other is row:
00527 |                 continue
00528 |             cost_b = finite_number(other.get("gpu_hours"))
00529 |             value_b = finite_number(other.get("metric_value"))
00530 |             if cost_b is None or value_b is None:
00531 |                 continue
00532 |             metric_no_worse = value_b <= value_a if mode == "min" else value_b >= value_a
00533 |             metric_better = value_b < value_a if mode == "min" else value_b > value_a
00534 |             cost_no_worse = cost_b <= cost_a
00535 |             cost_better = cost_b < cost_a
00536 |             if metric_no_worse and cost_no_worse and (metric_better or cost_better):
00537 |                 dominated = True
00538 |                 break
00539 |         row["pareto_dominated"] = dominated
00540 |         row["pareto_status"] = "dominated" if dominated else "frontier"
00541 |     return rows
00542 | 
00543 | 
00544 | def decide_winners(
00545 |     summary_rows: list[dict[str, Any]],
00546 |     *,
00547 |     mode: str = "min",
00548 |     min_final_seeds: int = 3,
00549 |     expected_seeds: list[int] | None = None,
00550 |     tolerance: float = 0.0,
00551 |     compute_accuracy_threshold: float | None = None,
00552 | ) -> dict[str, Any]:
00553 |     if mode not in VALID_MODES:
00554 |         raise RuntimeError("mode must be min or max.")
00555 |     gates_failed: list[str] = []
00556 |     expected_count = len(expected_seeds or [])
00557 |     for row in summary_rows:
00558 |         model = str(row.get("model") or "unknown")
00559 |         dataset_id = str(row.get("dataset_id") or "dataset")
00560 |         config_id = str(row.get("selected_config_id") or row.get("config_id") or "config")
00561 |         completed = int(row.get("n_seeds_completed") or 0)
00562 |         required = max(min_final_seeds, expected_count)
00563 |         if completed < required:
00564 |             gates_failed.append(f"incomplete_final_seeds:{model}/{dataset_id}/{config_id}")
00565 |         if row.get("robust_claim_allowed_by_comparability") is not True:
00566 |             gates_failed.append(f"diagnostic_only:{model}/{dataset_id}/{config_id}")
00567 | 
00568 |     def row_sort_key(row: dict[str, Any]) -> tuple[float, str]:
00569 |         value = finite_number(row.get("mean"))
00570 |         if value is None:
00571 |             value = math.inf if mode == "min" else -math.inf
00572 |         return (value if mode == "min" else -value, str(row.get("selected_config_id") or row.get("config_id") or ""))
00573 | 
00574 |     def best_model_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
00575 |         metric_rows = [row for row in rows if finite_number(row.get("mean")) is not None]
00576 |         if not metric_rows:
00577 |             return None
00578 |         metric_rows.sort(key=row_sort_key)
00579 |         return metric_rows[0]
00580 | 
00581 |     dataset_decisions: list[dict[str, Any]] = []
00582 |     dataset_ids = sorted({str(row.get("dataset_id") or "") for row in summary_rows if str(row.get("dataset_id") or "")})
00583 |     if not dataset_ids:
00584 |         gates_failed.append("missing_dataset")
00585 | 
00586 |     for dataset_id in dataset_ids:
00587 |         dataset_rows = [row for row in summary_rows if str(row.get("dataset_id") or "") == dataset_id]
00588 |         best_by_model = {
00589 |             model: best_model_row([row for row in dataset_rows if str(row.get("model") or "") == model])
00590 |             for model in MODELS
00591 |         }
00592 |         dataset_gates: list[str] = []
00593 |         if any(row is None for row in best_by_model.values()):
00594 |             dataset_gates.append(f"missing_model:{dataset_id}")
00595 |         metric_rows = [row for row in best_by_model.values() if row is not None]
00596 |         if len(metric_rows) < 2:
00597 |             dataset_gates.append(f"missing_metric:{dataset_id}")
00598 |         precision_winner = None
00599 |         effect_size = None
00600 |         ci_rule_passed = False
00601 |         best: dict[str, Any] | None = None
00602 |         other: dict[str, Any] | None = None
00603 |         if len(metric_rows) >= 2:
00604 |             metric_rows.sort(key=row_sort_key)
00605 |             best, other = metric_rows[0], metric_rows[1]
00606 |             best_mean = finite_number(best.get("mean"))
00607 |             other_mean = finite_number(other.get("mean"))
00608 |             if best_mean is not None and other_mean is not None:
00609 |                 effect_size = (other_mean - best_mean) if mode == "min" else (best_mean - other_mean)
00610 |                 ci_rule_passed = _ci_separated(best, other, mode=mode)
00611 |                 if effect_size > tolerance and ci_rule_passed:
00612 |                     precision_winner = str(best.get("model") or "")
00613 |                 elif effect_size <= tolerance:
00614 |                     dataset_gates.append(f"precision_difference_within_tolerance:{dataset_id}")
00615 |                 elif not ci_rule_passed:
00616 |                     dataset_gates.append(f"confidence_intervals_overlap_or_unavailable:{dataset_id}")
00617 |         gates_failed.extend(dataset_gates)
00618 |         dataset_decisions.append(
00619 |             {
00620 |                 "dataset_id": dataset_id,
00621 |                 "precision_winner": precision_winner,
00622 |                 "winner_config_id": (best or {}).get("selected_config_id") or (best or {}).get("config_id"),
00623 |                 "runner_up_model": (other or {}).get("model"),
00624 |                 "runner_up_config_id": (other or {}).get("selected_config_id") or (other or {}).get("config_id"),
00625 |                 "effect_size_best_vs_second": effect_size,
00626 |                 "ci_rule_passed": ci_rule_passed,
00627 |                 "gates_failed": sorted(set(dataset_gates)),
00628 |                 "best_config_by_model": {
00629 |                     model: {
00630 |                         "selected_config_id": row.get("selected_config_id") or row.get("config_id"),
00631 |                         "mean": row.get("mean"),
00632 |                         "confidence_interval": row.get("confidence_interval"),
00633 |                     }
00634 |                     for model, row in best_by_model.items()
00635 |                     if row is not None
00636 |                 },
00637 |             }
00638 |         )
00639 | 
00640 |     robust_dataset_winners = [
00641 |         str(item.get("precision_winner"))
00642 |         for item in dataset_decisions
00643 |         if item.get("precision_winner") and not item.get("gates_failed")
00644 |     ]
00645 |     precision_winner = None
00646 |     effect_size = None
00647 |     ci_rule_passed = all(bool(item.get("ci_rule_passed")) for item in dataset_decisions) if dataset_decisions else False
00648 |     if dataset_decisions:
00649 |         if len(robust_dataset_winners) != len(dataset_decisions):
00650 |             gates_failed.append("missing_robust_dataset_winner")
00651 |         elif len(set(robust_dataset_winners)) == 1:
00652 |             precision_winner = robust_dataset_winners[0]
00653 |             effect_sizes = [
00654 |                 finite_number(item.get("effect_size_best_vs_second"))
00655 |                 for item in dataset_decisions
00656 |                 if finite_number(item.get("effect_size_best_vs_second")) is not None
00657 |             ]
00658 |             effect_size = min(effect_sizes) if effect_sizes else None
00659 |         else:
00660 |             gates_failed.append("dataset_winners_disagree")
00661 |     if gates_failed:
00662 |         precision_winner = None
00663 | 
00664 |     compute_winner = None
00665 |     if compute_accuracy_threshold is not None:
00666 |         eligible = []
00667 |         for row in summary_rows:
00668 |             value = finite_number(row.get("mean"))
00669 |             cost = finite_number(row.get("gpu_hours_mean"))
00670 |             if value is None or cost is None:
00671 |                 continue
00672 |             meets = value <= compute_accuracy_threshold if mode == "min" else value >= compute_accuracy_threshold
00673 |             if meets:
00674 |                 eligible.append(row)
00675 |         eligible.sort(key=lambda row: (finite_number(row.get("gpu_hours_mean")) or math.inf, str(row.get("model"))))
00676 |         compute_winner = eligible[0].get("model") if eligible else None
00677 | 
00678 |     pareto_rows = pareto_frontier(summary_rows, mode=mode)
00679 |     frontier = [row for row in pareto_rows if row.get("pareto_status") == "frontier"]
00680 |     pareto_winner = frontier[0].get("model") if len(frontier) == 1 else None
00681 | 
00682 |     robust_claim_allowed = not gates_failed and precision_winner is not None
00683 |     diagnostic_reasons = [row.get("diagnostic_only_reason") for row in summary_rows if row.get("diagnostic_only_reason")]
00684 |     return {
00685 |         "precision_winner": precision_winner,
00686 |         "compute_winner": compute_winner,
00687 |         "pareto_winner": pareto_winner,
00688 |         "practical_pareto_winner": pareto_winner,
00689 |         "robust_claim_allowed": robust_claim_allowed,
00690 |         "gates_failed": sorted(set(gates_failed)),
00691 |         "gates_passed": [] if gates_failed else [
00692 |             "complete_final_seeds",
00693 |             "statistical_ci_rule",
00694 |             "deeph_equivalence_or_not_needed",
00695 |             "final_test_protocol",
00696 |         ],
00697 |         "diagnostic_only_reason": "; ".join(sorted(set(diagnostic_reasons))),
00698 |         "effect_size_best_vs_second": effect_size,
00699 |         "ci_rule_passed": ci_rule_passed,
00700 |         "dataset_decisions": dataset_decisions,
00701 |         "tolerance": tolerance,
00702 |         "compute_accuracy_threshold": compute_accuracy_threshold,
00703 |         "pareto_frontier": pareto_rows,
00704 |     }
00705 | 
00706 | 
00707 | def load_rows(run_root: Path | str) -> list[dict[str, Any]]:
00708 |     root = Path(str(run_root))
00709 |     candidates = [
00710 |         root / "summary" / "ranking" / "normalized_run_metrics.json",
00711 |         root / "summary" / "report" / "best_validation_summary.json",
00712 |         root / "sweep" / "training_sweep_manifest.json",
00713 |     ]
00714 |     for path in candidates:
00715 |         payload = read_json(path)
00716 |         rows = payload.get("rows") or payload.get("runs")
00717 |         if isinstance(rows, list):
00718 |             return [dict(row) for row in rows if isinstance(row, dict)]
00719 |     return []
00720 | 
00721 | 
00722 | def final_statistics_report(
00723 |     *,
00724 |     run_root: Path | str,
00725 |     output_dir: Path | None = None,
00726 |     metric: str,
00727 |     mode: str = "min",
00728 |     expected_seeds: list[int] | None = None,
00729 |     min_final_seeds: int = 3,
00730 |     tolerance: float = 0.0,
00731 |     compute_accuracy_threshold: float | None = None,
00732 |     bootstrap_iterations: int = 1000,
00733 | ) -> dict[str, Any]:
00734 |     rows = load_rows(run_root)
00735 |     violations = protocol_violations(rows)
00736 |     summary_rows = aggregate_final_seed_metrics(
00737 |         rows,
00738 |         metric=metric,
00739 |         expected_seeds=expected_seeds,
00740 |         bootstrap_iterations=bootstrap_iterations,
00741 |     )
00742 |     winners = decide_winners(
00743 |         summary_rows,
00744 |         mode=mode,
00745 |         min_final_seeds=min_final_seeds,
00746 |         expected_seeds=expected_seeds,
00747 |         tolerance=tolerance,
00748 |         compute_accuracy_threshold=compute_accuracy_threshold,
00749 |     )
00750 |     if violations:
00751 |         winners["robust_claim_allowed"] = False
00752 |         winners["gates_failed"] = sorted(set([*winners.get("gates_failed", []), "protocol_violation_test_metrics_outside_final_stage"]))
00753 |         winners["diagnostic_only_reason"] = "; ".join([winners.get("diagnostic_only_reason", ""), *violations]).strip("; ")
00754 |     output = output_dir or Path(str(run_root)) / "summary" / "final_statistics"
00755 |     manifest = {
00756 |         "schema": FINAL_STATS_SCHEMA,
00757 |         "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
00758 |         "run_root": str(run_root),
00759 |         "output_dir": str(output),
00760 |         "metric": metric,
00761 |         "mode": mode,
00762 |         "expected_seeds": expected_seeds or [],
00763 |         "min_final_seeds": min_final_seeds,
00764 |         "protocol_violations": violations,
00765 |         "final_seed_summary": summary_rows,
00766 |         "winner_decision": winners,
00767 |         "outputs": {
00768 |             "final_seed_summary_csv": str(output / "final_seed_summary.csv"),
00769 |             "final_seed_summary_json": str(output / "final_seed_summary.json"),
00770 |             "pareto_frontier_csv": str(output / "pareto_frontier.csv"),
00771 |             "pareto_frontier_json": str(output / "pareto_frontier.json"),
00772 |             "winner_decision_json": str(output / "winner_decision.json"),
00773 |             "final_statistics_json": str(output / "final_statistics.json"),
00774 |         },
00775 |     }
00776 |     write_csv(output / "final_seed_summary.csv", summary_rows)
00777 |     write_json(output / "final_seed_summary.json", {"rows": summary_rows})
00778 |     write_csv(output / "pareto_frontier.csv", winners.get("pareto_frontier") or [])
00779 |     write_json(output / "pareto_frontier.json", {"rows": winners.get("pareto_frontier") or []})
00780 |     write_json(output / "winner_decision.json", winners)
00781 |     write_json(output / "final_statistics.json", manifest)
00782 |     return manifest
00783 | 
00784 | 
00785 | def parse_args() -> argparse.Namespace:
00786 |     parser = argparse.ArgumentParser(description=__doc__)
00787 |     parser.add_argument("run_root", type=Path)
00788 |     parser.add_argument("--output-dir", type=Path, default=None)
00789 |     parser.add_argument("--metric", required=True)
00790 |     parser.add_argument("--mode", choices=("min", "max"), default="min")
00791 |     parser.add_argument("--expected-seeds", default="")
00792 |     parser.add_argument("--min-final-seeds", type=int, default=3)
00793 |     parser.add_argument("--tolerance", type=float, default=0.0)
00794 |     parser.add_argument("--compute-accuracy-threshold", type=float, default=None)
00795 |     parser.add_argument("--bootstrap-iterations", type=int, default=1000)
00796 |     return parser.parse_args()
00797 | 
00798 | 
00799 | def main() -> None:
00800 |     args = parse_args()
00801 |     expected = [int(item) for item in args.expected_seeds.split(",") if item.strip()] if args.expected_seeds else None
00802 |     manifest = final_statistics_report(
00803 |         run_root=args.run_root,
00804 |         output_dir=args.output_dir,
00805 |         metric=args.metric,
00806 |         mode=args.mode,
00807 |         expected_seeds=expected,
00808 |         min_final_seeds=args.min_final_seeds,
00809 |         tolerance=args.tolerance,
00810 |         compute_accuracy_threshold=args.compute_accuracy_threshold,
00811 |         bootstrap_iterations=args.bootstrap_iterations,
00812 |     )
00813 |     print(json.dumps(json_safe(manifest), indent=2, sort_keys=True, ensure_ascii=False))
00814 | 
00815 | 
00816 | if __name__ == "__main__":
00817 |     main()
```

## `Comparison/scripts/g2m_deeph_gate_check.py`

SHA-256: `ff0737b0a12ce4087c186c2a9a31df021aa9cdb0bc6aa53523d813ea361dd198`

```py
00001 | #!/usr/bin/env python3
00002 | """Fail-closed robust-claim gate checker for Graph2Mat-vs-DeepH benchmarks."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import argparse
00007 | import json
00008 | import sys
00009 | import time
00010 | from pathlib import Path
00011 | from typing import Any
00012 | 
00013 | from deeph_prediction_adapter import (
00014 |     EQUIVALENCE_PROVEN_RAW_GLOBAL,
00015 |     EQUIVALENCE_SCOPE_RAW_GLOBAL,
00016 |     EQUIVALENCE_STATUS_PROVEN,
00017 | )
00018 | from g2m_deeph_protocol import protocol_hash, validate_protocol
00019 | from g2m_deeph_training_sweep import json_safe
00020 | 
00021 | 
00022 | GATE_STATUS_SCHEMA = "graph2mat_deeph_gate_status_v1"
00023 | FORBIDDEN_REFERENCE_NAME = "ML_prediction.HSX"
00024 | CLAIM_STATUS_PRIORITY = (
00025 |     "invalid_protocol",
00026 |     "invalid_missing_evidence",
00027 |     "invalid_dataset",
00028 |     "invalid_equivalence",
00029 |     "invalid_final_statistics",
00030 |     "invalid_telemetry",
00031 |     "diagnostic_only",
00032 | )
00033 | 
00034 | 
00035 | def read_json_strict(path: Path) -> dict[str, Any]:
00036 |     try:
00037 |         payload = json.loads(path.read_text(encoding="utf-8"))
00038 |     except FileNotFoundError as exc:
00039 |         raise RuntimeError(f"missing JSON file: {path}") from exc
00040 |     except json.JSONDecodeError as exc:
00041 |         raise RuntimeError(f"malformed JSON file: {path}: {exc}") from exc
00042 |     if not isinstance(payload, dict):
00043 |         raise RuntimeError(f"JSON payload must be an object: {path}")
00044 |     return payload
00045 | 
00046 | 
00047 | def write_json(path: Path, payload: dict[str, Any]) -> None:
00048 |     path.parent.mkdir(parents=True, exist_ok=True)
00049 |     path.write_text(
00050 |         json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
00051 |         encoding="utf-8",
00052 |     )
00053 | 
00054 | 
00055 | def resolve_path(value: Any, *, base_dir: Path) -> Path:
00056 |     path = Path(str(value or ""))
00057 |     if path.is_absolute():
00058 |         return path
00059 |     cwd_path = Path.cwd() / path
00060 |     if cwd_path.exists():
00061 |         return cwd_path
00062 |     return base_dir / path
00063 | 
00064 | 
00065 | def gate(
00066 |     gate_id: str,
00067 |     status: str,
00068 |     *,
00069 |     severity: str,
00070 |     message: str,
00071 |     evidence_paths: list[Path | str] | None = None,
00072 |     claim_status: str | None = None,
00073 | ) -> dict[str, Any]:
00074 |     return {
00075 |         "id": gate_id,
00076 |         "status": status,
00077 |         "severity": severity,
00078 |         "claim_status": claim_status,
00079 |         "evidence_paths": [str(path) for path in (evidence_paths or []) if str(path)],
00080 |         "message": message,
00081 |     }
00082 | 
00083 | 
00084 | def pass_gate(gate_id: str, message: str, evidence_paths: list[Path | str] | None = None) -> dict[str, Any]:
00085 |     return gate(gate_id, "pass", severity="info", message=message, evidence_paths=evidence_paths)
00086 | 
00087 | 
00088 | def fail_gate(
00089 |     gate_id: str,
00090 |     message: str,
00091 |     *,
00092 |     claim_status: str,
00093 |     evidence_paths: list[Path | str] | None = None,
00094 | ) -> dict[str, Any]:
00095 |     return gate(
00096 |         gate_id,
00097 |         "fail",
00098 |         severity="blocker",
00099 |         message=message,
00100 |         evidence_paths=evidence_paths,
00101 |         claim_status=claim_status,
00102 |     )
00103 | 
00104 | 
00105 | def candidate_final_statistics_paths(workflow_root: Path | None, run_root: Path | None) -> list[Path]:
00106 |     paths: list[Path] = []
00107 |     if workflow_root is not None:
00108 |         paths.append(workflow_root / "final_test" / "final_statistics.json")
00109 |     if run_root is not None:
00110 |         paths.append(run_root / "summary" / "final_statistics" / "final_statistics.json")
00111 |     return paths
00112 | 
00113 | 
00114 | def candidate_evidence_bundle_paths(workflow_root: Path | None) -> list[Path]:
00115 |     return [workflow_root / "evidence" / "evidence_bundle_manifest.json"] if workflow_root is not None else []
00116 | 
00117 | 
00118 | def find_existing(paths: list[Path]) -> Path | None:
00119 |     for path in paths:
00120 |         if path.exists():
00121 |             return path
00122 |     return None
00123 | 
00124 | 
00125 | def split_counts(split_manifest: dict[str, Any]) -> dict[str, int]:
00126 |     raw_counts = split_manifest.get("split_counts")
00127 |     if isinstance(raw_counts, dict):
00128 |         counts: dict[str, int] = {}
00129 |         for split in ("train", "validation", "test"):
00130 |             try:
00131 |                 counts[split] = int(raw_counts.get(split) or 0)
00132 |             except (TypeError, ValueError):
00133 |                 counts[split] = 0
00134 |         return counts
00135 |     counts = {"train": 0, "validation": 0, "test": 0}
00136 |     for row in split_manifest.get("rows") or []:
00137 |         if not isinstance(row, dict):
00138 |             continue
00139 |         split = str(row.get("split") or "").strip()
00140 |         if split in counts:
00141 |             counts[split] += 1
00142 |     return counts
00143 | 
00144 | 
00145 | def forbidden_reference_findings(payload: Any, *, path: str = "") -> list[str]:
00146 |     findings: list[str] = []
00147 |     if isinstance(payload, dict):
00148 |         for key, value in payload.items():
00149 |             child_path = f"{path}.{key}" if path else str(key)
00150 |             if "forbidden" in str(key).lower():
00151 |                 continue
00152 |             findings.extend(forbidden_reference_findings(value, path=child_path))
00153 |     elif isinstance(payload, list):
00154 |         for index, value in enumerate(payload):
00155 |             findings.extend(forbidden_reference_findings(value, path=f"{path}[{index}]"))
00156 |     elif isinstance(payload, str) and FORBIDDEN_REFERENCE_NAME in payload:
00157 |         findings.append(f"{path}: {payload}")
00158 |     return findings
00159 | 
00160 | 
00161 | def validate_protocol_gate(protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
00162 |     raw_protocol = read_json_strict(protocol_path)
00163 |     try:
00164 |         protocol = validate_protocol(raw_protocol)
00165 |     except RuntimeError as exc:
00166 |         fallback = dict(raw_protocol)
00167 |         if "protocol_hash" not in fallback:
00168 |             try:
00169 |                 fallback["protocol_hash"] = protocol_hash(fallback)
00170 |             except Exception:
00171 |                 fallback["protocol_hash"] = ""
00172 |         return (
00173 |             fallback,
00174 |             raw_protocol,
00175 |             [
00176 |                 fail_gate(
00177 |                     "protocol_valid",
00178 |                     str(exc),
00179 |                     claim_status="invalid_protocol",
00180 |                     evidence_paths=[protocol_path],
00181 |                 )
00182 |             ],
00183 |         )
00184 |     return protocol, raw_protocol, [pass_gate("protocol_valid", "Protocol validates.", [protocol_path])]
00185 | 
00186 | 
00187 | def validate_dataset_gates(protocol: dict[str, Any], *, protocol_dir: Path) -> list[dict[str, Any]]:
00188 |     gates: list[dict[str, Any]] = []
00189 |     present_paths: list[Path] = []
00190 |     ready_failures: list[str] = []
00191 |     split_failures: list[str] = []
00192 |     forbidden_findings: list[str] = []
00193 |     for index, dataset in enumerate(protocol.get("datasets") or []):
00194 |         if not isinstance(dataset, dict):
00195 |             ready_failures.append(f"datasets[{index}] is not an object")
00196 |             continue
00197 |         dataset_id = str(dataset.get("dataset_id") or f"dataset_{index}")
00198 |         dataset_root = resolve_path(dataset.get("dataset_root"), base_dir=protocol_dir)
00199 |         benchmark_path = resolve_path(dataset.get("benchmark_dataset_manifest"), base_dir=protocol_dir)
00200 |         split_path = resolve_path(dataset.get("frozen_split_manifest"), base_dir=protocol_dir)
00201 |         artifact_path = resolve_path(
00202 |             dataset.get("artifact_validation") or dataset_root / "artifact_validation.json",
00203 |             base_dir=protocol_dir,
00204 |         )
00205 |         required_paths = [benchmark_path, split_path, artifact_path]
00206 |         missing = [path for path in required_paths if not path.exists()]
00207 |         if missing:
00208 |             gates.append(
00209 |                 fail_gate(
00210 |                     f"dataset_{dataset_id}_manifests_present",
00211 |                     "Dataset evidence is missing: " + ", ".join(str(path) for path in missing),
00212 |                     claim_status="invalid_missing_evidence",
00213 |                     evidence_paths=required_paths,
00214 |                 )
00215 |             )
00216 |             continue
00217 |         present_paths.extend(required_paths)
00218 |         benchmark = read_json_strict(benchmark_path)
00219 |         split = read_json_strict(split_path)
00220 |         artifact = read_json_strict(artifact_path)
00221 |         if benchmark.get("benchmark_ready") is not True or str(benchmark.get("validation_status") or "valid") == "invalid":
00222 |             ready_failures.append(f"{dataset_id}: benchmark_dataset_manifest is not benchmark_ready")
00223 |         if artifact.get("valid") is not True:
00224 |             ready_failures.append(f"{dataset_id}: artifact_validation.valid is not true")
00225 |         if split.get("valid") is False:
00226 |             split_failures.append(f"{dataset_id}: frozen_split_manifest.valid is false")
00227 |         counts = split_counts(split)
00228 |         missing_splits = [name for name, count in sorted(counts.items()) if count <= 0]
00229 |         if missing_splits:
00230 |             split_failures.append(f"{dataset_id}: empty split(s): {', '.join(missing_splits)}")
00231 |         for label, payload in (
00232 |             (f"{dataset_id}:benchmark_dataset_manifest", benchmark),
00233 |             (f"{dataset_id}:frozen_split_manifest", split),
00234 |             (f"{dataset_id}:artifact_validation", artifact),
00235 |         ):
00236 |             for finding in forbidden_reference_findings(payload):
00237 |                 forbidden_findings.append(f"{label}:{finding}")
00238 |     if present_paths:
00239 |         gates.append(pass_gate("dataset_manifests_present", "Dataset manifest files are present.", present_paths))
00240 |     if ready_failures:
00241 |         gates.append(
00242 |             fail_gate(
00243 |                 "dataset_benchmark_ready",
00244 |                 "; ".join(ready_failures),
00245 |                 claim_status="invalid_dataset",
00246 |                 evidence_paths=present_paths,
00247 |             )
00248 |         )
00249 |     elif present_paths:
00250 |         gates.append(pass_gate("dataset_benchmark_ready", "Dataset manifests are benchmark_ready.", present_paths))
00251 |     if split_failures:
00252 |         gates.append(
00253 |             fail_gate(
00254 |                 "frozen_split_nonempty",
00255 |                 "; ".join(split_failures),
00256 |                 claim_status="invalid_dataset",
00257 |                 evidence_paths=present_paths,
00258 |             )
00259 |         )
00260 |     elif present_paths:
00261 |         gates.append(pass_gate("frozen_split_nonempty", "Frozen splits have non-empty train/validation/test rows.", present_paths))
00262 |     if forbidden_findings:
00263 |         gates.append(
00264 |             fail_gate(
00265 |                 "forbidden_reference_absent",
00266 |                 "Forbidden reference artifact detected: " + "; ".join(forbidden_findings[:10]),
00267 |                 claim_status="invalid_dataset",
00268 |                 evidence_paths=present_paths,
00269 |             )
00270 |         )
00271 |     elif present_paths:
00272 |         gates.append(pass_gate("forbidden_reference_absent", "No ML_prediction.HSX reference paths found.", present_paths))
00273 |     return gates
00274 | 
00275 | 
00276 | def validate_selection_gate(protocol: dict[str, Any]) -> dict[str, Any]:
00277 |     selection = protocol.get("selection") if isinstance(protocol.get("selection"), dict) else {}
00278 |     top_k = protocol.get("top_k_selection") if isinstance(protocol.get("top_k_selection"), dict) else {}
00279 |     if (
00280 |         selection.get("split") == "validation"
00281 |         and selection.get("source") == "validation_only"
00282 |         and top_k.get("split") == "validation"
00283 |         and top_k.get("uses_test_metrics") is False
00284 |     ):
00285 |         return pass_gate("selection_validation_only", "Selection and top-k are validation-only.")
00286 |     return fail_gate(
00287 |         "selection_validation_only",
00288 |         "Selection/top-k policy is not validation-only.",
00289 |         claim_status="invalid_protocol",
00290 |     )
00291 | 
00292 | 
00293 | def load_final_statistics(path: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
00294 |     if path is None:
00295 |         return {}, [
00296 |             fail_gate(
00297 |                 "final_statistics_present",
00298 |                 "final_statistics.json is missing.",
00299 |                 claim_status="invalid_missing_evidence",
00300 |             )
00301 |         ]
00302 |     return read_json_strict(path), [pass_gate("final_statistics_present", "Final statistics manifest exists.", [path])]
00303 | 
00304 | 
00305 | def validate_final_statistics_gates(
00306 |     final_stats: dict[str, Any],
00307 |     *,
00308 |     final_stats_path: Path | None,
00309 |     expected_seed_count: int,
00310 | ) -> list[dict[str, Any]]:
00311 |     if not final_stats:
00312 |         return []
00313 |     gates: list[dict[str, Any]] = []
00314 |     summary_rows = [row for row in final_stats.get("final_seed_summary") or [] if isinstance(row, dict)]
00315 |     seed_failures = []
00316 |     for row in summary_rows:
00317 |         completed = int(row.get("n_seeds_completed") or 0)
00318 |         model = str(row.get("model") or "unknown")
00319 |         dataset = str(row.get("dataset_id") or "dataset")
00320 |         if completed < expected_seed_count:
00321 |             seed_failures.append(f"{model}/{dataset}: {completed} < {expected_seed_count}")
00322 |     if not summary_rows:
00323 |         seed_failures.append("final_seed_summary is empty")
00324 |     if seed_failures:
00325 |         gates.append(
00326 |             fail_gate(
00327 |                 "final_seeds_complete",
00328 |                 "Incomplete final seeds: " + "; ".join(seed_failures),
00329 |                 claim_status="invalid_final_statistics",
00330 |                 evidence_paths=[final_stats_path] if final_stats_path else None,
00331 |             )
00332 |         )
00333 |     else:
00334 |         gates.append(pass_gate("final_seeds_complete", "Final seed counts meet the protocol requirement.", [final_stats_path] if final_stats_path else None))
00335 | 
00336 |     winners = final_stats.get("winner_decision") if isinstance(final_stats.get("winner_decision"), dict) else {}
00337 |     if winners.get("robust_claim_allowed") is True:
00338 |         gates.append(pass_gate("final_statistics_robust", "Final statistics allow a robust claim.", [final_stats_path] if final_stats_path else None))
00339 |     else:
00340 |         reason = "; ".join(str(item) for item in winners.get("gates_failed") or []) or str(
00341 |             winners.get("diagnostic_only_reason") or "winner_decision.robust_claim_allowed is not true"
00342 |         )
00343 |         gates.append(
00344 |             fail_gate(
00345 |                 "final_statistics_robust",
00346 |                 reason,
00347 |                 claim_status="diagnostic_only" if "diagnostic" in reason else "invalid_final_statistics",
00348 |                 evidence_paths=[final_stats_path] if final_stats_path else None,
00349 |             )
00350 |         )
00351 |     return gates
00352 | 
00353 | 
00354 | def validate_telemetry_gate(final_stats: dict[str, Any], *, final_stats_path: Path | None) -> dict[str, Any]:
00355 |     missing: list[str] = []
00356 |     for row in final_stats.get("final_seed_summary") or []:
00357 |         if not isinstance(row, dict):
00358 |             continue
00359 |         label = f"{row.get('model') or 'unknown'}/{row.get('dataset_id') or 'dataset'}"
00360 |         if row.get("gpu_hours_mean") is None:
00361 |             missing.append(f"{label}: gpu_hours_mean")
00362 |         if row.get("peak_gpu_memory_mb_mean") is None:
00363 |             missing.append(f"{label}: peak_gpu_memory_mb_mean")
00364 |     if not final_stats.get("final_seed_summary"):
00365 |         missing.append("final_seed_summary")
00366 |     if missing:
00367 |         return fail_gate(
00368 |             "telemetry_complete",
00369 |             "Missing telemetry fields: " + ", ".join(missing),
00370 |             claim_status="invalid_telemetry",
00371 |             evidence_paths=[final_stats_path] if final_stats_path else None,
00372 |         )
00373 |     return pass_gate("telemetry_complete", "GPU-hours and peak GPU memory are present.", [final_stats_path] if final_stats_path else None)
00374 | 
00375 | 
00376 | def evidence_bundle_path_from_args(workflow_root: Path | None) -> Path | None:
00377 |     return find_existing(candidate_evidence_bundle_paths(workflow_root))
00378 | 
00379 | 
00380 | def validate_evidence_bundle_gate(evidence_path: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
00381 |     if evidence_path is None:
00382 |         return {}, [
00383 |             fail_gate(
00384 |                 "evidence_bundle_complete",
00385 |                 "evidence_bundle_manifest.json is missing.",
00386 |                 claim_status="invalid_missing_evidence",
00387 |             )
00388 |         ]
00389 |     payload = read_json_strict(evidence_path)
00390 |     missing_required = list(payload.get("missing_required") or [])
00391 |     files = [entry for entry in payload.get("files") or [] if isinstance(entry, dict)]
00392 |     for entry in files:
00393 |         if entry.get("required") is True:
00394 |             path_text = str(entry.get("path") or "")
00395 |             if entry.get("exists") is not True:
00396 |                 missing_required.append(str(entry.get("label") or path_text or "required_file"))
00397 |             elif path_text and not Path(path_text).exists():
00398 |                 missing_required.append(str(entry.get("label") or path_text))
00399 |     if str(payload.get("status") or "") != "complete":
00400 |         missing_required.append(f"bundle status is {payload.get('status') or 'missing'}")
00401 |     if missing_required:
00402 |         return payload, [
00403 |             fail_gate(
00404 |                 "evidence_bundle_complete",
00405 |                 "Evidence bundle is incomplete: " + ", ".join(sorted(set(missing_required))),
00406 |                 claim_status="invalid_missing_evidence",
00407 |                 evidence_paths=[evidence_path],
00408 |             )
00409 |         ]
00410 |     return payload, [pass_gate("evidence_bundle_complete", "Evidence bundle is complete.", [evidence_path])]
00411 | 
00412 | 
00413 | def adapter_manifest_paths(
00414 |     *,
00415 |     run_root: Path | None,
00416 |     evidence_bundle: dict[str, Any],
00417 | ) -> list[Path]:
00418 |     paths: list[Path] = []
00419 |     for entry in evidence_bundle.get("files") or []:
00420 |         if not isinstance(entry, dict):
00421 |             continue
00422 |         label = str(entry.get("label") or "")
00423 |         path = str(entry.get("path") or "")
00424 |         if "adapter_manifest" in label and path:
00425 |             paths.append(Path(path))
00426 |     if run_root is not None and run_root.exists():
00427 |         paths.extend(sorted(run_root.rglob("adapter_manifest.json")))
00428 |     unique: dict[str, Path] = {}
00429 |     for path in paths:
00430 |         unique[str(path)] = path
00431 |     return list(unique.values())
00432 | 
00433 | 
00434 | def adapter_manifest_is_proven(payload: dict[str, Any]) -> bool:
00435 |     statuses = [str(item) for item in payload.get("adapter_equivalence_statuses") or [] if str(item)]
00436 |     equivalence_statuses = [str(item) for item in payload.get("equivalence_statuses") or [] if str(item)]
00437 |     scopes = [str(item) for item in payload.get("equivalence_scopes") or [] if str(item)]
00438 |     gate_payload = payload.get("equivalence_gate") if isinstance(payload.get("equivalence_gate"), dict) else {}
00439 |     return (
00440 |         bool(payload.get("robust_matrix_metrics_allowed"))
00441 |         and gate_payload.get("robust_claim_allowed") is True
00442 |         and bool(statuses)
00443 |         and all(status == EQUIVALENCE_PROVEN_RAW_GLOBAL for status in statuses)
00444 |         and bool(equivalence_statuses)
00445 |         and all(status == EQUIVALENCE_STATUS_PROVEN for status in equivalence_statuses)
00446 |         and bool(scopes)
00447 |         and all(scope == EQUIVALENCE_SCOPE_RAW_GLOBAL for scope in scopes)
00448 |     )
00449 | 
00450 | 
00451 | def validate_deeph_equivalence_gate(
00452 |     protocol: dict[str, Any],
00453 |     *,
00454 |     run_root: Path | None,
00455 |     evidence_bundle: dict[str, Any],
00456 | ) -> dict[str, Any]:
00457 |     deeph = protocol.get("models", {}).get("deeph") if isinstance(protocol.get("models"), dict) else {}
00458 |     if not isinstance(deeph, dict) or deeph.get("enabled") is not True:
00459 |         return pass_gate("deeph_equivalence_proven", "DeepH is not enabled in this protocol.")
00460 |     paths = adapter_manifest_paths(run_root=run_root, evidence_bundle=evidence_bundle)
00461 |     if not paths:
00462 |         return fail_gate(
00463 |             "deeph_equivalence_proven",
00464 |             "No DeepH adapter_manifest.json was found.",
00465 |             claim_status="invalid_equivalence",
00466 |         )
00467 |     failures: list[str] = []
00468 |     for path in paths:
00469 |         if not path.exists():
00470 |             failures.append(f"{path}: missing")
00471 |             continue
00472 |         payload = read_json_strict(path)
00473 |         if not adapter_manifest_is_proven(payload):
00474 |             failures.append(f"{path}: raw/global equivalence is not proven")
00475 |     if failures:
00476 |         return fail_gate(
00477 |             "deeph_equivalence_proven",
00478 |             "; ".join(failures),
00479 |             claim_status="invalid_equivalence",
00480 |             evidence_paths=paths,
00481 |         )
00482 |     return pass_gate("deeph_equivalence_proven", "DeepH raw/global equivalence is proven.", paths)
00483 | 
00484 | 
00485 | def claim_status_from_gates(gates: list[dict[str, Any]]) -> str:
00486 |     failed_statuses = {
00487 |         str(gate.get("claim_status"))
00488 |         for gate in gates
00489 |         if gate.get("status") == "fail" and gate.get("claim_status")
00490 |     }
00491 |     if not failed_statuses:
00492 |         return "robust_allowed"
00493 |     for status in CLAIM_STATUS_PRIORITY:
00494 |         if status in failed_statuses:
00495 |             return status
00496 |     return "invalid_missing_evidence"
00497 | 
00498 | 
00499 | def required_next_actions(gates: list[dict[str, Any]]) -> list[str]:
00500 |     actions: list[str] = []
00501 |     for item in gates:
00502 |         if item.get("status") != "fail":
00503 |             continue
00504 |         actions.append(f"{item.get('id')}: {item.get('message')}")
00505 |     return actions
00506 | 
00507 | 
00508 | def build_gate_status(
00509 |     *,
00510 |     protocol_path: Path,
00511 |     workflow_root: Path | None = None,
00512 |     run_root: Path | None = None,
00513 | ) -> dict[str, Any]:
00514 |     workflow_root = Path(workflow_root) if workflow_root is not None else None
00515 |     run_root = Path(run_root) if run_root is not None else None
00516 |     protocol, _raw_protocol, gates = validate_protocol_gate(protocol_path)
00517 |     protocol_dir = protocol_path.parent
00518 |     if protocol.get("protocol_id"):
00519 |         gates.append(validate_selection_gate(protocol))
00520 |     gates.extend(validate_dataset_gates(protocol, protocol_dir=protocol_dir))
00521 | 
00522 |     final_stats_path = find_existing(candidate_final_statistics_paths(workflow_root, run_root))
00523 |     final_stats, final_stats_gates = load_final_statistics(final_stats_path)
00524 |     gates.extend(final_stats_gates)
00525 |     expected_seed_count = max(3, len(protocol.get("final_seeds") or []))
00526 |     gates.extend(
00527 |         validate_final_statistics_gates(
00528 |             final_stats,
00529 |             final_stats_path=final_stats_path,
00530 |             expected_seed_count=expected_seed_count,
00531 |         )
00532 |     )
00533 |     if final_stats:
00534 |         gates.append(validate_telemetry_gate(final_stats, final_stats_path=final_stats_path))
00535 | 
00536 |     evidence_bundle, evidence_gates = validate_evidence_bundle_gate(evidence_bundle_path_from_args(workflow_root))
00537 |     gates.extend(evidence_gates)
00538 |     gates.append(validate_deeph_equivalence_gate(protocol, run_root=run_root, evidence_bundle=evidence_bundle))
00539 | 
00540 |     status = claim_status_from_gates(gates)
00541 |     blockers = [str(item.get("message") or "") for item in gates if item.get("status") == "fail"]
00542 |     warnings = [str(item.get("message") or "") for item in gates if item.get("status") == "warn"]
00543 |     return {
00544 |         "schema": GATE_STATUS_SCHEMA,
00545 |         "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
00546 |         "protocol_id": protocol.get("protocol_id"),
00547 |         "protocol_hash": protocol.get("protocol_hash"),
00548 |         "protocol_path": str(protocol_path),
00549 |         "workflow_root": str(workflow_root) if workflow_root is not None else "",
00550 |         "run_root": str(run_root) if run_root is not None else "",
00551 |         "robust_claim_allowed": status == "robust_allowed",
00552 |         "diagnostic_only": status != "robust_allowed",
00553 |         "claim_status": status,
00554 |         "gates": gates,
00555 |         "blockers": blockers,
00556 |         "warnings": warnings,
00557 |         "required_next_actions": required_next_actions(gates),
00558 |     }
00559 | 
00560 | 
00561 | def error_status(*, protocol_path: Path | None, workflow_root: Path | None, run_root: Path | None, error: str) -> dict[str, Any]:
00562 |     gate_payload = fail_gate(
00563 |         "gate_check_error",
00564 |         error,
00565 |         claim_status="invalid_protocol" if protocol_path else "invalid_missing_evidence",
00566 |         evidence_paths=[protocol_path] if protocol_path else None,
00567 |     )
00568 |     return {
00569 |         "schema": GATE_STATUS_SCHEMA,
00570 |         "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
00571 |         "protocol_id": "",
00572 |         "protocol_hash": "",
00573 |         "protocol_path": str(protocol_path) if protocol_path is not None else "",
00574 |         "workflow_root": str(workflow_root) if workflow_root is not None else "",
00575 |         "run_root": str(run_root) if run_root is not None else "",
00576 |         "robust_claim_allowed": False,
00577 |         "diagnostic_only": True,
00578 |         "claim_status": gate_payload["claim_status"],
00579 |         "gates": [gate_payload],
00580 |         "blockers": [error],
00581 |         "warnings": [],
00582 |         "required_next_actions": [f"gate_check_error: {error}"],
00583 |     }
00584 | 
00585 | 
00586 | def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
00587 |     parser = argparse.ArgumentParser(description=__doc__)
00588 |     parser.add_argument("--protocol", type=Path, required=True)
00589 |     parser.add_argument("--workflow-root", type=Path, default=None)
00590 |     parser.add_argument("--run-root", type=Path, default=None)
00591 |     parser.add_argument("--output", type=Path, required=True)
00592 |     return parser.parse_args(argv)
00593 | 
00594 | 
00595 | def main(argv: list[str] | None = None) -> int:
00596 |     args = parse_args(argv)
00597 |     try:
00598 |         payload = build_gate_status(
00599 |             protocol_path=args.protocol,
00600 |             workflow_root=args.workflow_root,
00601 |             run_root=args.run_root,
00602 |         )
00603 |     except RuntimeError as exc:
00604 |         payload = error_status(
00605 |             protocol_path=args.protocol,
00606 |             workflow_root=args.workflow_root,
00607 |             run_root=args.run_root,
00608 |             error=str(exc),
00609 |         )
00610 |         write_json(args.output, payload)
00611 |         print(str(exc), file=sys.stderr)
00612 |         return 1
00613 |     write_json(args.output, payload)
00614 |     print(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False))
00615 |     return 0
00616 | 
00617 | 
00618 | if __name__ == "__main__":
00619 |     raise SystemExit(main())
```

## `Comparison/scripts/g2m_deeph_early_stopping.py`

SHA-256: `75a74cff9a0f5a3f67002cc77a98d13c3a9d03a7bd53ae64e7f366e1569de6c1`

```py
00001 | #!/usr/bin/env python3
00002 | """Common validation-based early stopping policy for Graph2Mat-vs-DeepH."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import math
00007 | import re
00008 | from dataclasses import dataclass, field
00009 | from pathlib import Path
00010 | from typing import Any
00011 | 
00012 | 
00013 | ALLOWED_METRIC_MODES = {"min", "max"}
00014 | DEEPh_VAL_LOSS_RE = re.compile(r"Epoch #(?P<epoch>\d+).*?Val loss:\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
00015 | METRIC_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
00016 | 
00017 | 
00018 | @dataclass(frozen=True)
00019 | class EarlyStoppingPolicy:
00020 |     validation_metric_name: str
00021 |     metric_mode: str
00022 |     patience: int
00023 |     min_delta: float
00024 |     max_epochs: int
00025 | 
00026 |     def to_dict(self) -> dict[str, Any]:
00027 |         return {
00028 |             "validation_metric_name": self.validation_metric_name,
00029 |             "metric_mode": self.metric_mode,
00030 |             "patience": self.patience,
00031 |             "min_delta": self.min_delta,
00032 |             "max_epochs": self.max_epochs,
00033 |         }
00034 | 
00035 | 
00036 | @dataclass
00037 | class ValidationEvent:
00038 |     epoch: int
00039 |     value: float
00040 | 
00041 | 
00042 | @dataclass
00043 | class EarlyStoppingTracker:
00044 |     policy: EarlyStoppingPolicy
00045 |     best_epoch: int | None = None
00046 |     best_validation_value: float | None = None
00047 |     epochs_trained: int = 0
00048 |     validation_checks: int = 0
00049 |     checks_since_improvement: int = 0
00050 |     stop_triggered: bool = False
00051 |     events: list[ValidationEvent] = field(default_factory=list)
00052 | 
00053 |     def improved(self, value: float) -> bool:
00054 |         if self.best_validation_value is None:
00055 |             return True
00056 |         if self.policy.metric_mode == "min":
00057 |             return value < self.best_validation_value - self.policy.min_delta
00058 |         return value > self.best_validation_value + self.policy.min_delta
00059 | 
00060 |     def update(self, *, epoch: int, value: float) -> bool:
00061 |         if not math.isfinite(float(value)):
00062 |             raise RuntimeError(f"Validation metric is non-finite at epoch {epoch}: {value!r}")
00063 |         self.events.append(ValidationEvent(epoch=int(epoch), value=float(value)))
00064 |         self.epochs_trained = max(self.epochs_trained, int(epoch))
00065 |         self.validation_checks += 1
00066 |         if self.improved(float(value)):
00067 |             self.best_epoch = int(epoch)
00068 |             self.best_validation_value = float(value)
00069 |             self.checks_since_improvement = 0
00070 |         else:
00071 |             self.checks_since_improvement += 1
00072 |         if self.checks_since_improvement >= self.policy.patience:
00073 |             self.stop_triggered = True
00074 |         return self.stop_triggered
00075 | 
00076 |     def metadata(self, *, failed: bool = False, interrupted: bool = False) -> dict[str, Any]:
00077 |         if not self.events:
00078 |             raise RuntimeError(
00079 |                 f"Missing validation metric {self.policy.validation_metric_name!r}; "
00080 |                 "early stopping/checkpoint selection must fail closed."
00081 |             )
00082 |         if failed:
00083 |             stop_reason = "failed"
00084 |         elif interrupted:
00085 |             stop_reason = "interrupted"
00086 |         elif self.stop_triggered:
00087 |             stop_reason = "early_stopping"
00088 |         elif self.epochs_trained >= self.policy.max_epochs:
00089 |             stop_reason = "max_epochs"
00090 |         else:
00091 |             stop_reason = "completed"
00092 |         return {
00093 |             **self.policy.to_dict(),
00094 |             "best_epoch": self.best_epoch,
00095 |             "best_validation_value": self.best_validation_value,
00096 |             "epochs_trained": self.epochs_trained,
00097 |             "validation_checks": self.validation_checks,
00098 |             "checks_since_improvement": self.checks_since_improvement,
00099 |             "stop_reason": stop_reason,
00100 |         }
00101 | 
00102 | 
00103 | class DeepHEarlyStoppingObserver:
00104 |     def __init__(self, policy: EarlyStoppingPolicy):
00105 |         self.tracker = EarlyStoppingTracker(policy)
00106 |         self.stop_reason: str | None = None
00107 | 
00108 |     def __call__(self, line: str) -> str | None:
00109 |         event = parse_deeph_validation_line(line)
00110 |         if event is None:
00111 |             return None
00112 |         if self.tracker.update(epoch=event.epoch, value=event.value):
00113 |             self.stop_reason = (
00114 |                 f"early_stopping: {self.tracker.checks_since_improvement} validation checks "
00115 |                 f"without improvement in {self.tracker.policy.validation_metric_name}"
00116 |             )
00117 |             return self.stop_reason
00118 |         return None
00119 | 
00120 |     def metadata(self) -> dict[str, Any]:
00121 |         return self.tracker.metadata()
00122 | 
00123 | 
00124 | def _references_test_metric(metric: str) -> bool:
00125 |     return "test" in [token.lower() for token in METRIC_TOKEN_RE.findall(metric)]
00126 | 
00127 | 
00128 | def parse_early_stopping_policy(payload: dict[str, Any] | None) -> EarlyStoppingPolicy | None:
00129 |     payload = payload or {}
00130 |     raw = payload.get("early_stopping")
00131 |     if raw in (None, "", False):
00132 |         return None
00133 |     if not isinstance(raw, dict):
00134 |         raise RuntimeError("early_stopping must be an object.")
00135 |     if raw.get("enabled") is False:
00136 |         return None
00137 |     metric = str(raw.get("metric") or raw.get("validation_metric_name") or "").strip()
00138 |     if not metric:
00139 |         raise RuntimeError("early_stopping.metric is required.")
00140 |     if _references_test_metric(metric):
00141 |         raise RuntimeError("early_stopping.metric must not reference test metrics.")
00142 |     mode = str(raw.get("mode") or raw.get("metric_mode") or "").strip().lower()
00143 |     if mode not in ALLOWED_METRIC_MODES:
00144 |         raise RuntimeError("early_stopping.mode must be min or max.")
00145 |     try:
00146 |         patience = int(raw["patience"])
00147 |         min_delta = float(raw["min_delta"])
00148 |         max_epochs = int(raw["max_epochs"])
00149 |     except KeyError as exc:
00150 |         raise RuntimeError(f"early_stopping.{exc.args[0]} is required.") from exc
00151 |     except (TypeError, ValueError) as exc:
00152 |         raise RuntimeError("early_stopping.patience, min_delta and max_epochs must be numeric.") from exc
00153 |     if patience <= 0:
00154 |         raise RuntimeError("early_stopping.patience must be positive.")
00155 |     if min_delta < 0:
00156 |         raise RuntimeError("early_stopping.min_delta must be non-negative.")
00157 |     if max_epochs <= 0:
00158 |         raise RuntimeError("early_stopping.max_epochs must be positive.")
00159 |     return EarlyStoppingPolicy(
00160 |         validation_metric_name=metric,
00161 |         metric_mode=mode,
00162 |         patience=patience,
00163 |         min_delta=min_delta,
00164 |         max_epochs=max_epochs,
00165 |     )
00166 | 
00167 | 
00168 | def parse_deeph_validation_line(line: str) -> ValidationEvent | None:
00169 |     match = DEEPh_VAL_LOSS_RE.search(line)
00170 |     if not match:
00171 |         return None
00172 |     return ValidationEvent(epoch=int(match.group("epoch")), value=float(match.group("value")))
00173 | 
00174 | 
00175 | def graph2mat_early_stopping_callbacks(policy: EarlyStoppingPolicy) -> list[dict[str, Any]]:
00176 |     return [
00177 |         {
00178 |             "class_path": "EarlyStopping",
00179 |             "init_args": {
00180 |                 "monitor": policy.validation_metric_name,
00181 |                 "mode": policy.metric_mode,
00182 |                 "patience": policy.patience,
00183 |                 "min_delta": policy.min_delta,
00184 |                 "strict": True,
00185 |             },
00186 |         }
00187 |     ]
00188 | 
00189 | 
00190 | def tensorboard_policy_metadata(training_dir: Path, policy: EarlyStoppingPolicy) -> dict[str, Any]:
00191 |     try:
00192 |         from tensorboard.backend.event_processing import event_accumulator
00193 |     except Exception as exc:
00194 |         raise RuntimeError("TensorBoard event reader is required for Graph2Mat early stopping metadata.") from exc
00195 |     event_files = sorted(
00196 |         (training_dir / "lightning_logs").rglob("events.out.tfevents.*"),
00197 |         key=lambda path: path.stat().st_mtime,
00198 |     )
00199 |     if not event_files:
00200 |         raise RuntimeError(
00201 |             f"Missing validation metric {policy.validation_metric_name!r}; no TensorBoard event files under {training_dir}."
00202 |         )
00203 |     tracker = EarlyStoppingTracker(policy)
00204 |     found = False
00205 |     for event_file in event_files:
00206 |         accumulator = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
00207 |         accumulator.Reload()
00208 |         tags = set(accumulator.Tags().get("scalars", []))
00209 |         if policy.validation_metric_name not in tags:
00210 |             continue
00211 |         epoch_by_step = {}
00212 |         if "epoch" in tags:
00213 |             epoch_by_step = {
00214 |                 int(item.step): int(float(item.value))
00215 |                 for item in accumulator.Scalars("epoch")
00216 |             }
00217 |         for item in accumulator.Scalars(policy.validation_metric_name):
00218 |             found = True
00219 |             epoch = epoch_by_step.get(int(item.step), int(item.step))
00220 |             tracker.update(epoch=epoch, value=float(item.value))
00221 |     if not found:
00222 |         raise RuntimeError(
00223 |             f"Missing validation metric {policy.validation_metric_name!r}; "
00224 |             f"event files did not contain that scalar under {training_dir}."
00225 |         )
00226 |     return tracker.metadata()
```

## `tests/test_g2m_deeph_rank_runs.py`

SHA-256: `3296ecef2e0a4ff4c5f2e8888180ba17892f0b99e83e74587413e7625c48fb37`

```py
00001 | import json
00002 | import sys
00003 | import tempfile
00004 | import unittest
00005 | from pathlib import Path
00006 | 
00007 | 
00008 | REPO_ROOT = Path(__file__).resolve().parents[1]
00009 | SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
00010 | if str(SCRIPTS_DIR) not in sys.path:
00011 |     sys.path.insert(0, str(SCRIPTS_DIR))
00012 | 
00013 | from g2m_deeph_rank_runs import (  # noqa: E402
00014 |     build_recommendation,
00015 |     choose_primary_metric,
00016 |     load_metric_rows,
00017 |     pairwise_comparisons,
00018 |     pareto_frontier,
00019 |     rank_graph2mat_deeph_runs,
00020 |     rank_metric_groups,
00021 |     row_from_training_record,
00022 | )
00023 | from deeph_prediction_adapter import (  # noqa: E402
00024 |     EQUIVALENCE_PROVEN_RAW_GLOBAL,
00025 |     EQUIVALENCE_STATUS_UNPROVEN,
00026 | )
00027 | 
00028 | 
00029 | def valid_metric_row(model: str, *, value: float = 0.1, seed: int = 1, **overrides) -> dict:
00030 |     row = {
00031 |         "model": model,
00032 |         "dataset_id": "d",
00033 |         "config_id": f"{model}_cfg",
00034 |         "seed": seed,
00035 |         "run_status": "completed",
00036 |         "method_status": "ok",
00037 |         "comparability_status": "valid",
00038 |         "artifact_contract_status": "valid",
00039 |         "required_provenance_present": True,
00040 |         "provenance_status": "valid",
00041 |         "warning_status": "ok",
00042 |         "metric_fail_policy": "fail_closed",
00043 |         "low_energy_rmse_eV_mean": value,
00044 |     }
00045 |     if model == "deeph":
00046 |         row.update(
00047 |             {
00048 |                 "adapter_equivalence_status": EQUIVALENCE_PROVEN_RAW_GLOBAL,
00049 |                 "raw_global_equivalence_proven": True,
00050 |                 "split_audit_status": "valid",
00051 |             }
00052 |         )
00053 |     row.update(overrides)
00054 |     return row
00055 | 
00056 | 
00057 | def best_row(model: str, *, mean: float, seeds: int = 3, **overrides) -> dict:
00058 |     row = {
00059 |         "scope": "global",
00060 |         "dataset_id": "all",
00061 |         "model": model,
00062 |         "config_id": f"{model}_cfg",
00063 |         "metric": "low_energy_rmse_eV",
00064 |         "mean": mean,
00065 |         "valid_seed_count": seeds,
00066 |         "seed_stability_status": "robust_candidate" if seeds >= 3 else "exploratory_only",
00067 |     }
00068 |     row.update(overrides)
00069 |     return row
00070 | 
00071 | 
00072 | def write_json(path: Path, payload: dict) -> None:
00073 |     path.parent.mkdir(parents=True, exist_ok=True)
00074 |     path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
00075 | 
00076 | 
00077 | def write_dataset(root: Path, split_hash: str = "split-a", compatibility: str = "compat-a") -> None:
00078 |     root.mkdir(parents=True, exist_ok=True)
00079 |     write_json(root / "frozen_split_manifest.json", {"valid": True, "split_hash": split_hash, "rows": []})
00080 |     write_json(
00081 |         root / "benchmark_dataset_manifest.json",
00082 |         {
00083 |             "benchmark_ready": True,
00084 |             "benchmark_dataset_id": compatibility,
00085 |             "material_label": "graphene",
00086 |             "frozen_split_manifest": {"split_hash": split_hash},
00087 |         },
00088 |     )
00089 | 
00090 | 
00091 | def write_metric_root(root: Path, *, h_mae: float, low_energy: float, r2: float = 0.5) -> None:
00092 |     root.mkdir(parents=True, exist_ok=True)
00093 |     (root / "kpoint_matrix_metrics.csv").write_text(
00094 |         "row_type,sample,h_mae_eV,h_rmse_eV,h_mse_eV2,relative_frobenius,hermiticity_pred\n"
00095 |         f"weighted_sample,s1,{h_mae},{h_mae * 2},{h_mae * h_mae},0.2,0.0\n",
00096 |         encoding="utf-8",
00097 |     )
00098 |     (root / "sparse_metrics.csv").write_text(
00099 |         "sample,mae_union_eV,rmse_union_eV,mse_union_eV2,r2_union,support_f1\n"
00100 |         f"s1,{h_mae},{h_mae * 2},{h_mae * h_mae},{r2},0.7\n",
00101 |         encoding="utf-8",
00102 |     )
00103 |     (root / "kpoint_spectral_metrics.csv").write_text(
00104 |         "sample,global_rmse_eV,low_energy_rmse_eV,fermi_window_rmse_eV,frontier_window_rmse_eV\n"
00105 |         f"s1,{low_energy * 2},{low_energy},{low_energy * 1.1},{low_energy * 1.2}\n",
00106 |         encoding="utf-8",
00107 |     )
00108 |     (root / "kpoint_dos_metrics.csv").write_text(
00109 |         "sample,dos_mae_500_fermi_window\n"
00110 |         f"s1,{low_energy * 0.1}\n",
00111 |         encoding="utf-8",
00112 |     )
00113 |     write_json(root / "manifest.json", {"uses_reference_overlap_k": True, "kpoint_metrics_enabled": True, "warnings": []})
00114 | 
00115 | 
00116 | def write_deeph_adapter_manifest(metrics_root: Path, *, proven: bool = True) -> None:
00117 |     status = EQUIVALENCE_PROVEN_RAW_GLOBAL if proven else "diagnostic_local_frame_only"
00118 |     write_json(
00119 |         metrics_root.parent / "adapter_manifest.json",
00120 |         {
00121 |             "adapter_equivalence_statuses": [status],
00122 |             "diagnostic_only_count": 0 if proven else 1,
00123 |             "raw_global_equivalence_proven_count": 1 if proven else 0,
00124 |             "robust_matrix_metrics_allowed": proven,
00125 |             "samples": [
00126 |                 {
00127 |                     "sample_id": "s1",
00128 |                     "adapter_equivalence_status": status,
00129 |                     "diagnostic_only": not proven,
00130 |                 }
00131 |             ],
00132 |         },
00133 |     )
00134 | 
00135 | 
00136 | def write_run(
00137 |     base: Path,
00138 |     dataset: Path,
00139 |     *,
00140 |     model: str,
00141 |     config_id: str,
00142 |     seed: int,
00143 |     h_mae: float,
00144 |     low_energy: float,
00145 |     seconds: float = 10.0,
00146 |     metric_fail_policy: str = "fail_closed",
00147 | ) -> dict:
00148 |     run_root = base / "sweep" / model / dataset.name / f"{config_id}_{seed}"
00149 |     metrics_root = (
00150 |         run_root / "metrics" / "graph2mat" / "eval_input" / "metrics"
00151 |         if model == "graph2mat"
00152 |         else run_root / "metrics" / "deeph" / "eval" / "metrics"
00153 |     )
00154 |     write_metric_root(metrics_root, h_mae=h_mae, low_energy=low_energy)
00155 |     deeph_manifest_path = ""
00156 |     if model == "deeph":
00157 |         write_deeph_adapter_manifest(metrics_root, proven=True)
00158 |         deeph_manifest_path = str(run_root / "deeph" / "deeph_manifest.json")
00159 |         write_json(Path(deeph_manifest_path), {"split_audit_status": "valid", "split_audit": {"status": "valid"}})
00160 |     return {
00161 |         "model": model,
00162 |         "dataset_id": dataset.name,
00163 |         "dataset_root": str(dataset),
00164 |         "config_id": config_id,
00165 |         "config_hash": config_id,
00166 |         "common": {"seed": seed},
00167 |         "status": "completed",
00168 |         "run_root": str(run_root),
00169 |         "train_run": {"elapsed_seconds": seconds},
00170 |         "predict_run": {"elapsed_seconds": seconds / 10.0},
00171 |         "metrics_run": {"elapsed_seconds": seconds / 20.0},
00172 |         "metric_fail_policy": metric_fail_policy,
00173 |         "deeph_manifest_path": deeph_manifest_path,
00174 |     }
00175 | 
00176 | 
00177 | class Graph2MatDeepHRankingTests(unittest.TestCase):
00178 |     def test_loader_reads_valid_training_sweep_run(self) -> None:
00179 |         with tempfile.TemporaryDirectory() as tmp:
00180 |             root = Path(tmp)
00181 |             dataset = root / "dataset"
00182 |             write_dataset(dataset)
00183 |             record = write_run(root, dataset, model="graph2mat", config_id="g2m_a", seed=1, h_mae=0.2, low_energy=0.3)
00184 |             manifest = root / "sweep" / "training_sweep_manifest.json"
00185 |             write_json(manifest, {"runs": [record]})
00186 | 
00187 |             rows = load_metric_rows(training_sweep_manifest_path=manifest)
00188 | 
00189 |             self.assertEqual(rows[0]["model"], "graph2mat")
00190 |             self.assertEqual(rows[0]["config_id"], "g2m_a")
00191 |             self.assertAlmostEqual(float(rows[0]["low_energy_rmse_eV_mean"]), 0.3)
00192 | 
00193 |     def test_missing_config_id_fails_clearly(self) -> None:
00194 |         with self.assertRaisesRegex(RuntimeError, "config_id"):
00195 |             row_from_training_record({"model": "graph2mat", "status": "completed"})
00196 | 
00197 |     def test_primary_metric_ignores_non_finite_values(self) -> None:
00198 |         rows = [
00199 |             {"model": "graph2mat", "low_energy_rmse_eV_mean": "nan", "h_mae_eV_mean": 0.2},
00200 |             {"model": "deeph", "low_energy_rmse_eV_mean": 0.1, "h_mae_eV_mean": 0.3},
00201 |         ]
00202 | 
00203 |         self.assertEqual(choose_primary_metric(rows), "h_mae_eV")
00204 | 
00205 |     def test_ranking_lower_and_higher_directions(self) -> None:
00206 |         rows = [
00207 |             {"model": "graph2mat", "dataset_id": "d", "config_id": "a", "seed": 1, "run_status": "completed", "method_status": "ok", "comparability_status": "valid", "low_energy_rmse_eV_mean": 0.2, "r2_mean": 0.4},
00208 |             {"model": "graph2mat", "dataset_id": "d", "config_id": "b", "seed": 1, "run_status": "completed", "method_status": "ok", "comparability_status": "valid", "low_energy_rmse_eV_mean": 0.1, "r2_mean": 0.9},
00209 |         ]
00210 | 
00211 |         lower = rank_metric_groups(rows, "low_energy_rmse_eV")
00212 |         higher = rank_metric_groups(rows, "r2")
00213 | 
00214 |         self.assertEqual([row for row in lower if row["rank"] == 1][0]["config_id"], "b")
00215 |         self.assertEqual([row for row in higher if row["rank"] == 1][0]["config_id"], "b")
00216 | 
00217 |     def test_pairwise_blocks_incompatible_split_hash(self) -> None:
00218 |         best_rows = [
00219 |             {"scope": "dataset", "dataset_id": "d", "model": "graph2mat", "config_id": "g", "metric": "low_energy_rmse_eV", "mean": 0.2, "frozen_split_hash": "a", "dataset_compatibility_hash": "c", "robust_eligible": True},
00220 |             {
00221 |                 "scope": "dataset",
00222 |                 "dataset_id": "d",
00223 |                 "model": "deeph",
00224 |                 "config_id": "d",
00225 |                 "metric": "low_energy_rmse_eV",
00226 |                 "mean": 0.1,
00227 |                 "frozen_split_hash": "b",
00228 |                 "dataset_compatibility_hash": "c",
00229 |                 "robust_eligible": True,
00230 |                 "adapter_equivalence_status": EQUIVALENCE_PROVEN_RAW_GLOBAL,
00231 |             },
00232 |         ]
00233 | 
00234 |         pairs = pairwise_comparisons(best_rows)
00235 | 
00236 |         self.assertEqual(pairs[0]["status"], "invalid_incompatible_splits")
00237 |         self.assertIsNone(pairs[0]["winner"])
00238 | 
00239 |     def test_single_seed_recommendation_is_exploratory_not_robust(self) -> None:
00240 |         best_rows = [
00241 |             best_row("graph2mat", mean=0.2, seeds=1),
00242 |             best_row("deeph", mean=0.1, seeds=1),
00243 |         ]
00244 |         pairs = [{"metric": "low_energy_rmse_eV", "status": "comparable", "winner": "deeph"}]
00245 |         rows = [valid_metric_row("graph2mat", value=0.2), valid_metric_row("deeph", value=0.1)]
00246 | 
00247 |         rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=pairs, primary_metric="low_energy_rmse_eV")
00248 | 
00249 |         self.assertEqual(rec["status"], "exploratory_deeph_win")
00250 |         self.assertEqual(rec["scientific_status"], "exploratory_only")
00251 | 
00252 |     def test_severe_warning_blocks_robust_winner(self) -> None:
00253 |         best_rows = [
00254 |             best_row("graph2mat", mean=0.2),
00255 |             best_row("deeph", mean=0.1),
00256 |         ]
00257 |         rows = [
00258 |             valid_metric_row("graph2mat", value=0.2, severe_warnings=["severe overlap"], warning_status="severe"),
00259 |             valid_metric_row("deeph", value=0.1),
00260 |         ]
00261 | 
00262 |         rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")
00263 | 
00264 |         self.assertIsNone(rec["winner"])
00265 |         self.assertIn("severe_warnings", rec["gates_failed"])
00266 | 
00267 |     def test_metric_fail_policy_diagnostic_blocks_robust_winner(self) -> None:
00268 |         best_rows = [
00269 |             best_row("graph2mat", mean=0.2),
00270 |             best_row("deeph", mean=0.1),
00271 |         ]
00272 |         rows = [
00273 |             valid_metric_row("graph2mat", value=0.2),
00274 |             valid_metric_row("deeph", value=0.1, metric_fail_policy="diagnostic_only", fail_open_metric_outputs=True),
00275 |         ]
00276 | 
00277 |         rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")
00278 | 
00279 |         self.assertIsNone(rec["winner"])
00280 |         self.assertEqual(rec["status"], "diagnostic_only")
00281 |         self.assertIn("metric_fail_policy_diagnostic_only", rec["gates_failed"])
00282 | 
00283 |     def test_missing_provenance_gets_explicit_status(self) -> None:
00284 |         best_rows = [best_row("graph2mat", mean=0.1), best_row("deeph", mean=0.2)]
00285 |         rows = [
00286 |             valid_metric_row("graph2mat", value=0.1, required_provenance_present=False, provenance_status="invalid"),
00287 |             valid_metric_row("deeph", value=0.2),
00288 |         ]
00289 | 
00290 |         rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")
00291 | 
00292 |         self.assertEqual(rec["status"], "invalid_missing_provenance")
00293 |         self.assertIn("invalid_missing_provenance", rec["gates_failed"])
00294 | 
00295 |     def test_deeph_split_audit_missing_gets_explicit_status(self) -> None:
00296 |         best_rows = [best_row("graph2mat", mean=0.2), best_row("deeph", mean=0.1)]
00297 |         rows = [
00298 |             valid_metric_row("graph2mat", value=0.2),
00299 |             valid_metric_row("deeph", value=0.1, split_audit_status="missing"),
00300 |         ]
00301 | 
00302 |         rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")
00303 | 
00304 |         self.assertEqual(rec["status"], "invalid_unverified_deeph_split")
00305 |         self.assertIn("invalid_unverified_deeph_split", rec["gates_failed"])
00306 | 
00307 |     def test_incomplete_grid_gets_explicit_status(self) -> None:
00308 |         rows = [valid_metric_row("graph2mat", value=0.1)]
00309 | 
00310 |         rec = build_recommendation(rows=rows, best_rows=[best_row("graph2mat", mean=0.1)], pairs=[], primary_metric="low_energy_rmse_eV")
00311 | 
00312 |         self.assertEqual(rec["status"], "invalid_incomplete_grid")
00313 |         self.assertIn("missing_model", rec["gates_failed"])
00314 | 
00315 |     def test_stable_robust_graph2mat_win(self) -> None:
00316 |         best_rows = [best_row("graph2mat", mean=0.1), best_row("deeph", mean=0.2)]
00317 |         rows = [valid_metric_row("graph2mat", value=0.1), valid_metric_row("deeph", value=0.2)]
00318 | 
00319 |         rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")
00320 | 
00321 |         self.assertEqual(rec["status"], "robust_graph2mat_win")
00322 |         self.assertEqual(rec["winner"], "graph2mat")
00323 |         self.assertEqual(rec["adapter_equivalence_status"], EQUIVALENCE_PROVEN_RAW_GLOBAL)
00324 |         self.assertEqual(rec["split_audit_status"], "valid")
00325 | 
00326 |     def test_stable_robust_deeph_win(self) -> None:
00327 |         best_rows = [best_row("graph2mat", mean=0.2), best_row("deeph", mean=0.1)]
00328 |         rows = [valid_metric_row("graph2mat", value=0.2), valid_metric_row("deeph", value=0.1)]
00329 | 
00330 |         rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")
00331 | 
00332 |         self.assertEqual(rec["status"], "robust_deeph_win")
00333 |         self.assertEqual(rec["winner"], "deeph")
00334 | 
00335 |     def test_training_record_fail_open_metrics_are_diagnostic_only(self) -> None:
00336 |         with tempfile.TemporaryDirectory() as tmp:
00337 |             root = Path(tmp)
00338 |             dataset = root / "dataset"
00339 |             write_dataset(dataset)
00340 |             record = write_run(
00341 |                 root,
00342 |                 dataset,
00343 |                 model="deeph",
00344 |                 config_id="deeph_diag",
00345 |                 seed=1,
00346 |                 h_mae=0.1,
00347 |                 low_energy=0.1,
00348 |                 metric_fail_policy="diagnostic_only",
00349 |             )
00350 | 
00351 |             row = row_from_training_record(record)
00352 | 
00353 |             self.assertEqual(row["comparability_status"], "diagnostic_only")
00354 |             self.assertTrue(row["diagnostic_only"])
00355 |             self.assertTrue(row["fail_open_metric_outputs"])
00356 | 
00357 |     def test_deeph_adapter_diagnostic_blocks_robust_winner(self) -> None:
00358 |         best_rows = [
00359 |             best_row("graph2mat", mean=0.2),
00360 |             best_row("deeph", mean=0.1),
00361 |         ]
00362 |         rows = [
00363 |             valid_metric_row("graph2mat", value=0.2),
00364 |             valid_metric_row("deeph", value=0.1, adapter_equivalence_status="diagnostic_local_frame_only", raw_global_equivalence_proven=False),
00365 |         ]
00366 | 
00367 |         rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")
00368 | 
00369 |         self.assertIsNone(rec["winner"])
00370 |         self.assertEqual(rec["status"], "diagnostic_only")
00371 |         self.assertIn("deeph_adapter_equivalence_not_proven", rec["gates_failed"])
00372 | 
00373 |     def test_unproven_equivalence_status_blocks_even_with_proven_adapter_label(self) -> None:
00374 |         best_rows = [
00375 |             best_row("graph2mat", mean=0.2),
00376 |             best_row("deeph", mean=0.1),
00377 |         ]
00378 |         rows = [
00379 |             valid_metric_row("graph2mat", value=0.2),
00380 |             valid_metric_row(
00381 |                 "deeph",
00382 |                 value=0.1,
00383 |                 adapter_equivalence_status=EQUIVALENCE_PROVEN_RAW_GLOBAL,
00384 |                 equivalence_status=EQUIVALENCE_STATUS_UNPROVEN,
00385 |             ),
00386 |         ]
00387 | 
00388 |         rec = build_recommendation(
00389 |             rows=rows,
00390 |             best_rows=best_rows,
00391 |             pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}],
00392 |             primary_metric="low_energy_rmse_eV",
00393 |         )
00394 | 
00395 |         self.assertIsNone(rec["winner"])
00396 |         self.assertEqual(rec["status"], "diagnostic_only")
00397 |         self.assertIn("deeph_adapter_equivalence_not_proven", rec["gates_failed"])
00398 | 
00399 |     def test_deeph_missing_adapter_status_is_not_robust_eligible(self) -> None:
00400 |         rows = [
00401 |             {"model": "deeph", "dataset_id": "d", "config_id": "local", "seed": 1, "run_status": "completed", "method_status": "ok", "comparability_status": "valid", "low_energy_rmse_eV_mean": 0.1, "adapter_equivalence_status": "diagnostic_local_frame_only"},
00402 |         ]
00403 | 
00404 |         ranked = rank_metric_groups(rows, "low_energy_rmse_eV")
00405 | 
00406 |         self.assertFalse(ranked[0]["robust_eligible"])
00407 |         self.assertEqual(ranked[0]["adapter_equivalence_status"], "diagnostic_local_frame_only")
00408 | 
00409 |     def test_pareto_excludes_dominated_runs(self) -> None:
00410 |         rows = [
00411 |             valid_metric_row("graph2mat", value=0.4, config_id="slow_bad", total_time_seconds=20),
00412 |             valid_metric_row("graph2mat", value=0.2, config_id="fast_good", total_time_seconds=10),
00413 |         ]
00414 | 
00415 |         frontier = pareto_frontier(rows, "low_energy_rmse_eV")
00416 | 
00417 |         self.assertEqual([row["config_id"] for row in frontier], ["fast_good"])
00418 | 
00419 |     def test_ranker_writes_outputs_and_exploratory_winner(self) -> None:
00420 |         with tempfile.TemporaryDirectory() as tmp:
00421 |             root = Path(tmp)
00422 |             dataset = root / "dataset"
00423 |             write_dataset(dataset)
00424 |             runs = [
00425 |                 write_run(root, dataset, model="graph2mat", config_id="g2m_a", seed=1, h_mae=0.2, low_energy=0.2),
00426 |                 write_run(root, dataset, model="deeph", config_id="deeph_a", seed=1, h_mae=0.1, low_energy=0.1),
00427 |             ]
00428 |             write_json(root / "sweep" / "training_sweep_manifest.json", {"runs": runs})
00429 | 
00430 |             manifest = rank_graph2mat_deeph_runs(run_root=root)
00431 | 
00432 |             self.assertEqual(manifest["recommendation"]["status"], "exploratory_deeph_win")
00433 |             self.assertTrue((root / "summary" / "ranking" / "recommendation.json").exists())
00434 |             self.assertTrue((root / "summary" / "ranking" / "pareto_accuracy_cost.csv").exists())
00435 | 
00436 | 
00437 | if __name__ == "__main__":
00438 |     unittest.main()
```

## `tests/test_g2m_deeph_final_stats.py`

SHA-256: `37a291395a41b737a2268104d122cc110d0c14846a6cd6b480f1529fdd940231`

```py
00001 | import json
00002 | import sys
00003 | import tempfile
00004 | import unittest
00005 | from pathlib import Path
00006 | 
00007 | 
00008 | REPO_ROOT = Path(__file__).resolve().parents[1]
00009 | SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
00010 | if str(SCRIPTS_DIR) not in sys.path:
00011 |     sys.path.insert(0, str(SCRIPTS_DIR))
00012 | 
00013 | from deeph_prediction_adapter import EQUIVALENCE_PROVEN_RAW_GLOBAL, EQUIVALENCE_STATUS_PROVEN, EQUIVALENCE_STATUS_UNPROVEN  # noqa: E402
00014 | from g2m_deeph_final_stats import (  # noqa: E402
00015 |     aggregate_final_seed_metrics,
00016 |     bootstrap_ci,
00017 |     decide_winners,
00018 |     final_statistics_report,
00019 |     protocol_violations,
00020 | )
00021 | 
00022 | 
00023 | def write_json(path: Path, payload: dict) -> None:
00024 |     path.parent.mkdir(parents=True, exist_ok=True)
00025 |     path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
00026 | 
00027 | 
00028 | def final_row(
00029 |     model: str,
00030 |     seed: int,
00031 |     value: float,
00032 |     *,
00033 |     gpu_hours: float = 1.0,
00034 |     peak_memory: float = 1000.0,
00035 |     proven_deeph: bool = True,
00036 |     stage: str = "final_test",
00037 |     split: str = "test",
00038 |     **overrides,
00039 | ) -> dict:
00040 |     row = {
00041 |         "status": "completed",
00042 |         "model": model,
00043 |         "dataset_id": "dataset_a",
00044 |         "config_id": f"{model}_cfg",
00045 |         "seed": seed,
00046 |         "protocol_stage": stage,
00047 |         "metric_split": split,
00048 |         "low_energy_rmse_eV_mean": value,
00049 |         "telemetry": {
00050 |             "gpu_hours_total": gpu_hours,
00051 |             "peak_gpu_memory_mb": peak_memory,
00052 |             "samples_per_second": 10.0 / gpu_hours,
00053 |             "matrix_blocks_per_second": 100.0 / gpu_hours,
00054 |         },
00055 |         "per_system_metrics": [
00056 |             {"sample_id": "s1", "low_energy_rmse_eV_mean": value * 0.9},
00057 |             {"sample_id": "s2", "low_energy_rmse_eV_mean": value * 1.1},
00058 |         ],
00059 |     }
00060 |     if model == "deeph":
00061 |         row["adapter_equivalence_status"] = (
00062 |             EQUIVALENCE_PROVEN_RAW_GLOBAL if proven_deeph else "diagnostic_local_frame_only"
00063 |         )
00064 |         row["comparability_status"] = "valid" if proven_deeph else "diagnostic_only"
00065 |         row["diagnostic_only"] = not proven_deeph
00066 |     row.update(overrides)
00067 |     return row
00068 | 
00069 | 
00070 | class Graph2MatDeepHFinalStatsTests(unittest.TestCase):
00071 |     def test_mean_std_aggregation_and_compute_summary(self) -> None:
00072 |         rows = [
00073 |             final_row("graph2mat", 0, 0.10, gpu_hours=2.0),
00074 |             final_row("graph2mat", 1, 0.20, gpu_hours=4.0),
00075 |             final_row("graph2mat", 2, 0.15, gpu_hours=3.0),
00076 |         ]
00077 | 
00078 |         summary = aggregate_final_seed_metrics(rows, metric="low_energy_rmse_eV", expected_seeds=[0, 1, 2])
00079 | 
00080 |         self.assertEqual(summary[0]["n_seeds_completed"], 3)
00081 |         self.assertAlmostEqual(summary[0]["mean"], 0.15)
00082 |         self.assertAlmostEqual(summary[0]["gpu_hours_mean"], 3.0)
00083 |         self.assertEqual(summary[0]["missing_seeds"], [])
00084 | 
00085 |     def test_incomplete_seeds_block_robust_winner(self) -> None:
00086 |         summaries = aggregate_final_seed_metrics(
00087 |             [
00088 |                 final_row("graph2mat", 0, 0.10),
00089 |                 final_row("graph2mat", 1, 0.11),
00090 |                 final_row("deeph", 0, 0.09),
00091 |                 final_row("deeph", 1, 0.10),
00092 |             ],
00093 |             metric="low_energy_rmse_eV",
00094 |             expected_seeds=[0, 1, 2],
00095 |         )
00096 | 
00097 |         decision = decide_winners(
00098 |             summaries,
00099 |             expected_seeds=[0, 1, 2],
00100 |             min_final_seeds=3,
00101 |             mode="min",
00102 |         )
00103 | 
00104 |         self.assertFalse(decision["robust_claim_allowed"])
00105 |         self.assertIn("incomplete_final_seeds:graph2mat/dataset_a/graph2mat_cfg", decision["gates_failed"])
00106 |         self.assertIn("incomplete_final_seeds:deeph/dataset_a/deeph_cfg", decision["gates_failed"])
00107 | 
00108 |     def test_bootstrap_ci_when_per_system_metrics_exist(self) -> None:
00109 |         ci = bootstrap_ci([0.1, 0.2, 0.3, 0.4], iterations=200, seed=7)
00110 | 
00111 |         self.assertEqual(ci["method"], "bootstrap_per_system_mean")
00112 |         self.assertIsNotNone(ci["low"])
00113 |         self.assertIsNotNone(ci["high"])
00114 |         self.assertLessEqual(ci["low"], ci["high"])
00115 | 
00116 |     def test_missing_per_system_metrics_is_explicit(self) -> None:
00117 |         row = final_row("graph2mat", 0, 0.10)
00118 |         row.pop("per_system_metrics")
00119 | 
00120 |         summary = aggregate_final_seed_metrics([row], metric="low_energy_rmse_eV")
00121 | 
00122 |         self.assertEqual(summary[0]["bootstrap_ci"]["method"], "unavailable")
00123 |         self.assertIn("per-system metrics unavailable", summary[0]["bootstrap_ci"]["reason"])
00124 | 
00125 |     def test_deeph_diagnostic_only_gate_blocks_robust_claim(self) -> None:
00126 |         summaries = aggregate_final_seed_metrics(
00127 |             [
00128 |                 final_row("graph2mat", 0, 0.20),
00129 |                 final_row("graph2mat", 1, 0.21),
00130 |                 final_row("graph2mat", 2, 0.22),
00131 |                 final_row("deeph", 0, 0.10, proven_deeph=False),
00132 |                 final_row("deeph", 1, 0.11, proven_deeph=False),
00133 |                 final_row("deeph", 2, 0.12, proven_deeph=False),
00134 |             ],
00135 |             metric="low_energy_rmse_eV",
00136 |             expected_seeds=[0, 1, 2],
00137 |         )
00138 | 
00139 |         decision = decide_winners(summaries, expected_seeds=[0, 1, 2], mode="min")
00140 | 
00141 |         self.assertFalse(decision["robust_claim_allowed"])
00142 |         self.assertIn("diagnostic_only:deeph/dataset_a/deeph_cfg", decision["gates_failed"])
00143 |         self.assertIn("deeph adapter equivalence not proven", decision["diagnostic_only_reason"])
00144 | 
00145 |     def test_unproven_formal_equivalence_status_blocks_robust_claim(self) -> None:
00146 |         rows = [
00147 |             final_row("graph2mat", 0, 0.20),
00148 |             final_row("graph2mat", 1, 0.21),
00149 |             final_row("graph2mat", 2, 0.22),
00150 |             final_row("deeph", 0, 0.10, proven_deeph=True, equivalence_status=EQUIVALENCE_STATUS_UNPROVEN),
00151 |             final_row("deeph", 1, 0.11, proven_deeph=True, equivalence_status=EQUIVALENCE_STATUS_UNPROVEN),
00152 |             final_row("deeph", 2, 0.12, proven_deeph=True, equivalence_status=EQUIVALENCE_STATUS_UNPROVEN),
00153 |         ]
00154 | 
00155 |         summaries = aggregate_final_seed_metrics(rows, metric="low_energy_rmse_eV", expected_seeds=[0, 1, 2])
00156 |         decision = decide_winners(summaries, expected_seeds=[0, 1, 2], mode="min")
00157 | 
00158 |         self.assertFalse(decision["robust_claim_allowed"])
00159 |         self.assertIn("diagnostic_only:deeph/dataset_a/deeph_cfg", decision["gates_failed"])
00160 |         self.assertIn("equivalence=unproven", decision["diagnostic_only_reason"])
00161 | 
00162 |     def test_deeph_adapter_manifest_discovery_can_prove_equivalence(self) -> None:
00163 |         with tempfile.TemporaryDirectory() as tmp:
00164 |             run_root = Path(tmp) / "deeph_run"
00165 |             write_json(
00166 |                 run_root / "deeph" / "inference" / "adapter_manifest.json",
00167 |                 {
00168 |                     "adapter_equivalence_statuses": [EQUIVALENCE_PROVEN_RAW_GLOBAL],
00169 |                     "equivalence_statuses": [EQUIVALENCE_STATUS_PROVEN],
00170 |                     "raw_global_equivalence_proven_count": 1,
00171 |                     "equivalence_gate": {
00172 |                         "robust_claim_allowed": True,
00173 |                         "diagnostic_only": False,
00174 |                     },
00175 |                 },
00176 |             )
00177 |             rows = [
00178 |                 final_row(
00179 |                     "deeph",
00180 |                     0,
00181 |                     0.10,
00182 |                     run_root=str(run_root),
00183 |                     adapter_equivalence_status="",
00184 |                     comparability_status="",
00185 |                     diagnostic_only=False,
00186 |                 )
00187 |             ]
00188 | 
00189 |             summary = aggregate_final_seed_metrics(rows, metric="low_energy_rmse_eV")
00190 | 
00191 |             self.assertTrue(summary[0]["robust_claim_allowed_by_comparability"])
00192 |             self.assertEqual(summary[0]["diagnostic_only_reason"], "")
00193 | 
00194 |     def test_top_k_configs_are_not_mixed_in_final_seed_summary(self) -> None:
00195 |         rows = [
00196 |             final_row("graph2mat", 0, 0.10, selected_config_id="g_a", config_id="g_a_seed0"),
00197 |             final_row("graph2mat", 1, 0.10, selected_config_id="g_a", config_id="g_a_seed1"),
00198 |             final_row("graph2mat", 2, 0.10, selected_config_id="g_a", config_id="g_a_seed2"),
00199 |             final_row("graph2mat", 0, 0.40, selected_config_id="g_b", config_id="g_b_seed0"),
00200 |             final_row("graph2mat", 1, 0.40, selected_config_id="g_b", config_id="g_b_seed1"),
00201 |             final_row("graph2mat", 2, 0.40, selected_config_id="g_b", config_id="g_b_seed2"),
00202 |             final_row("deeph", 0, 0.30, selected_config_id="d_a", config_id="d_a_seed0"),
00203 |             final_row("deeph", 1, 0.30, selected_config_id="d_a", config_id="d_a_seed1"),
00204 |             final_row("deeph", 2, 0.30, selected_config_id="d_a", config_id="d_a_seed2"),
00205 |         ]
00206 | 
00207 |         summaries = aggregate_final_seed_metrics(rows, metric="low_energy_rmse_eV", expected_seeds=[0, 1, 2])
00208 |         by_config = {row["selected_config_id"]: row for row in summaries}
00209 |         decision = decide_winners(summaries, expected_seeds=[0, 1, 2], mode="min")
00210 | 
00211 |         self.assertEqual(set(by_config), {"g_a", "g_b", "d_a"})
00212 |         self.assertAlmostEqual(by_config["g_a"]["mean"], 0.10)
00213 |         self.assertAlmostEqual(by_config["g_b"]["mean"], 0.40)
00214 |         self.assertTrue(decision["robust_claim_allowed"])
00215 |         self.assertEqual(decision["precision_winner"], "graph2mat")
00216 |         self.assertEqual(decision["dataset_decisions"][0]["winner_config_id"], "g_a")
00217 | 
00218 |     def test_winner_tolerance_and_ci_rules(self) -> None:
00219 |         summaries = aggregate_final_seed_metrics(
00220 |             [
00221 |                 final_row("graph2mat", 0, 0.10),
00222 |                 final_row("graph2mat", 1, 0.10),
00223 |                 final_row("graph2mat", 2, 0.10),
00224 |                 final_row("deeph", 0, 0.30),
00225 |                 final_row("deeph", 1, 0.30),
00226 |                 final_row("deeph", 2, 0.30),
00227 |             ],
00228 |             metric="low_energy_rmse_eV",
00229 |             expected_seeds=[0, 1, 2],
00230 |         )
00231 | 
00232 |         decision = decide_winners(summaries, expected_seeds=[0, 1, 2], mode="min", tolerance=0.01)
00233 | 
00234 |         self.assertTrue(decision["robust_claim_allowed"])
00235 |         self.assertEqual(decision["precision_winner"], "graph2mat")
00236 |         self.assertTrue(decision["ci_rule_passed"])
00237 | 
00238 |     def test_compute_winner_threshold_logic(self) -> None:
00239 |         summaries = aggregate_final_seed_metrics(
00240 |             [
00241 |                 final_row("graph2mat", 0, 0.10, gpu_hours=5.0),
00242 |                 final_row("graph2mat", 1, 0.10, gpu_hours=5.0),
00243 |                 final_row("graph2mat", 2, 0.10, gpu_hours=5.0),
00244 |                 final_row("deeph", 0, 0.20, gpu_hours=1.0),
00245 |                 final_row("deeph", 1, 0.20, gpu_hours=1.0),
00246 |                 final_row("deeph", 2, 0.20, gpu_hours=1.0),
00247 |             ],
00248 |             metric="low_energy_rmse_eV",
00249 |             expected_seeds=[0, 1, 2],
00250 |         )
00251 | 
00252 |         decision = decide_winners(
00253 |             summaries,
00254 |             expected_seeds=[0, 1, 2],
00255 |             mode="min",
00256 |             compute_accuracy_threshold=0.25,
00257 |         )
00258 | 
00259 |         self.assertEqual(decision["compute_winner"], "deeph")
00260 | 
00261 |     def test_protocol_violation_when_test_metrics_appear_in_wrong_stage(self) -> None:
00262 |         rows = [final_row("graph2mat", 0, 0.10, stage="search", split="test")]
00263 | 
00264 |         violations = protocol_violations(rows)
00265 | 
00266 |         self.assertTrue(violations)
00267 |         self.assertIn("outside final_test", violations[0])
00268 | 
00269 |     def test_final_statistics_report_serializes_outputs_and_protocol_violation(self) -> None:
00270 |         with tempfile.TemporaryDirectory() as tmp:
00271 |             root = Path(tmp)
00272 |             rows = [
00273 |                 final_row("graph2mat", 0, 0.10, stage="search", split="test"),
00274 |                 final_row("deeph", 0, 0.20),
00275 |             ]
00276 |             write_json(root / "summary" / "ranking" / "normalized_run_metrics.json", {"rows": rows})
00277 | 
00278 |             report = final_statistics_report(
00279 |                 run_root=root,
00280 |                 metric="low_energy_rmse_eV",
00281 |                 expected_seeds=[0],
00282 |                 min_final_seeds=1,
00283 |             )
00284 | 
00285 |             self.assertFalse(report["winner_decision"]["robust_claim_allowed"])
00286 |             self.assertIn("protocol_violation_test_metrics_outside_final_stage", report["winner_decision"]["gates_failed"])
00287 |             self.assertTrue((root / "summary" / "final_statistics" / "winner_decision.json").exists())
00288 | 
00289 | 
00290 | if __name__ == "__main__":
00291 |     unittest.main()
```

## `tests/test_g2m_deeph_gate_check.py`

SHA-256: `6321d867aa7f8ccd58e96436f5822dd1e7f976f766423b6d6a565797c641db1a`

```py
00001 | import json
00002 | import sys
00003 | import tempfile
00004 | import unittest
00005 | from pathlib import Path
00006 | 
00007 | 
00008 | REPO_ROOT = Path(__file__).resolve().parents[1]
00009 | SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
00010 | if str(SCRIPTS_DIR) not in sys.path:
00011 |     sys.path.insert(0, str(SCRIPTS_DIR))
00012 | 
00013 | from deeph_prediction_adapter import EQUIVALENCE_PROVEN_RAW_GLOBAL  # noqa: E402
00014 | from g2m_deeph_gate_check import build_gate_status, main  # noqa: E402
00015 | 
00016 | 
00017 | def write_json(path: Path, payload: dict) -> None:
00018 |     path.parent.mkdir(parents=True, exist_ok=True)
00019 |     path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
00020 | 
00021 | 
00022 | def protocol_payload(dataset_root: Path) -> dict:
00023 |     return {
00024 |         "protocol_id": "gate_check_protocol_unit",
00025 |         "version": "1.0",
00026 |         "datasets": [
00027 |             {
00028 |                 "dataset_id": "joint_a",
00029 |                 "dataset_root": str(dataset_root),
00030 |                 "benchmark_dataset_manifest": str(dataset_root / "benchmark_dataset_manifest.json"),
00031 |                 "frozen_split_manifest": str(dataset_root / "frozen_split_manifest.json"),
00032 |             }
00033 |         ],
00034 |         "reference_artifacts": {
00035 |             "required": [
00036 |                 "RUN.fdf",
00037 |                 "SystemLabel.TSHS",
00038 |                 "SystemLabel.TSDE",
00039 |                 "SystemLabel.HSX",
00040 |                 "SystemLabel.STRUCT_OUT",
00041 |                 "SystemLabel.XV",
00042 |                 "SystemLabel.ORB_INDX",
00043 |                 "metadata.json",
00044 |             ],
00045 |             "forbidden": ["ML_prediction.HSX"],
00046 |             "forbid_as_reference": "ML_prediction.HSX",
00047 |         },
00048 |         "models": {
00049 |             "graph2mat": {
00050 |                 "enabled": True,
00051 |                 "search_space": {
00052 |                     "optim_lr": {"choices": [0.001, 0.003]},
00053 |                     "batch_size": {"choices": [64, 128]},
00054 |                     "max_epochs": {"value": 20},
00055 |                     "hidden_irreps": {"choices": ["16x0e + 16x1o + 16x2e"]},
00056 |                     "num_interactions": {"value": 2},
00057 |                     "correlation": {"value": 2},
00058 |                     "max_ell": {"value": 2},
00059 |                 },
00060 |             },
00061 |             "deeph": {
00062 |                 "enabled": True,
00063 |                 "search_space": {
00064 |                     "learning_rate": {"choices": [0.0001, 0.0003]},
00065 |                     "batch_size": {"choices": [2, 4]},
00066 |                     "epochs": {"value": 20},
00067 |                     "atom_fea_len": {"value": 64},
00068 |                     "edge_fea_len": {"value": 128},
00069 |                     "num_l": {"value": 4},
00070 |                     "if_lcmp": {"value": True},
00071 |                 },
00072 |             },
00073 |         },
00074 |         "selection": {
00075 |             "split": "validation",
00076 |             "metric": "val_loss",
00077 |             "mode": "min",
00078 |             "source": "validation_only",
00079 |         },
00080 |         "early_stopping": {
00081 |             "metric": "val_loss",
00082 |             "mode": "min",
00083 |             "patience": 5,
00084 |             "min_delta": 0.0,
00085 |             "max_epochs": 20,
00086 |         },
00087 |         "search_policy": {"strategy": "random", "n_trials_per_model": 2, "random_seed": 1},
00088 |         "budget_policy": {"mode": "equal_n_trials", "n_trials_per_model": 2},
00089 |         "final_seeds": [0, 1, 2],
00090 |         "top_k_selection": {
00091 |             "k_per_model": 1,
00092 |             "split": "validation",
00093 |             "metric": "val_loss",
00094 |             "uses_test_metrics": False,
00095 |         },
00096 |         "final_evaluation": {
00097 |             "primary_metric": "low_energy_rmse_eV",
00098 |             "mode": "min",
00099 |             "secondary_metrics": ["fermi_window_rmse_eV", "dos_wasserstein_eV", "h_mae_eV"],
00100 |         },
00101 |         "final_test_policy": {
00102 |             "policy": "locked_until_final",
00103 |             "test_split": "test",
00104 |             "locked_during_search": True,
00105 |             "evaluate_once_after_selection": True,
00106 |         },
00107 |         "required_telemetry": [
00108 |             "wall_clock_seconds",
00109 |             "gpu_hours",
00110 |             "peak_gpu_memory_mb",
00111 |             "samples_per_second",
00112 |             "matrix_blocks_per_second",
00113 |             "best_validation_epoch",
00114 |         ],
00115 |         "deeph_comparability": {
00116 |             "adapter_equivalence_policy": "fail_closed_unless_proven",
00117 |             "robust_winner_requires_proven_equivalence": True,
00118 |             "diagnostic_if_unproven": True,
00119 |         },
00120 |     }
00121 | 
00122 | 
00123 | class Graph2MatDeepHGateCheckTests(unittest.TestCase):
00124 |     def setUp(self) -> None:
00125 |         self.tmp = tempfile.TemporaryDirectory()
00126 |         self.root = Path(self.tmp.name)
00127 |         self.dataset_root = self.root / "dataset"
00128 |         self.workflow_root = self.root / "workflow"
00129 |         self.run_root = self.workflow_root / "runs" / "final"
00130 |         self.protocol_path = self.root / "protocol.json"
00131 |         self.output_path = self.root / "gate_status.json"
00132 |         self.dataset_root.mkdir(parents=True)
00133 |         self.run_root.mkdir(parents=True)
00134 |         self.write_complete_fixture()
00135 | 
00136 |     def tearDown(self) -> None:
00137 |         self.tmp.cleanup()
00138 | 
00139 |     def write_complete_fixture(self) -> None:
00140 |         write_json(self.protocol_path, protocol_payload(self.dataset_root))
00141 |         write_json(
00142 |             self.dataset_root / "benchmark_dataset_manifest.json",
00143 |             {
00144 |                 "schema": "joint_graph2mat_deeph_benchmark_manifest_v1",
00145 |                 "benchmark_ready": True,
00146 |                 "validation_status": "valid",
00147 |                 "frozen_split_manifest": {
00148 |                     "path": str(self.dataset_root / "frozen_split_manifest.json"),
00149 |                     "split_counts": {"train": 1, "validation": 1, "test": 1},
00150 |                     "valid": True,
00151 |                 },
00152 |             },
00153 |         )
00154 |         write_json(
00155 |             self.dataset_root / "frozen_split_manifest.json",
00156 |             {
00157 |                 "schema": "joint_graph2mat_deeph_frozen_split_manifest_v1",
00158 |                 "valid": True,
00159 |                 "split_hash": "split-hash",
00160 |                 "split_counts": {"train": 1, "validation": 1, "test": 1},
00161 |                 "rows": [
00162 |                     {"sample_id": "s0", "split": "train", "artifact_paths": {"reference_hsx": "s0.HSX"}},
00163 |                     {"sample_id": "s1", "split": "validation", "artifact_paths": {"reference_hsx": "s1.HSX"}},
00164 |                     {"sample_id": "s2", "split": "test", "artifact_paths": {"reference_hsx": "s2.HSX"}},
00165 |                 ],
00166 |             },
00167 |         )
00168 |         write_json(
00169 |             self.dataset_root / "artifact_validation.json",
00170 |             {
00171 |                 "valid": True,
00172 |                 "snapshots": [
00173 |                     {"valid": True, "present_artifacts": {"hsx": "s0.HSX"}},
00174 |                     {"valid": True, "present_artifacts": {"hsx": "s1.HSX"}},
00175 |                     {"valid": True, "present_artifacts": {"hsx": "s2.HSX"}},
00176 |                 ],
00177 |             },
00178 |         )
00179 |         write_json(
00180 |             self.workflow_root / "final_test" / "final_statistics.json",
00181 |             {
00182 |                 "schema": "graph2mat_deeph_final_statistics_v1",
00183 |                 "expected_seeds": [0, 1, 2],
00184 |                 "final_seed_summary": [
00185 |                     {
00186 |                         "model": "graph2mat",
00187 |                         "dataset_id": "joint_a",
00188 |                         "n_seeds_completed": 3,
00189 |                         "gpu_hours_mean": 1.0,
00190 |                         "peak_gpu_memory_mb_mean": 1000.0,
00191 |                     },
00192 |                     {
00193 |                         "model": "deeph",
00194 |                         "dataset_id": "joint_a",
00195 |                         "n_seeds_completed": 3,
00196 |                         "gpu_hours_mean": 1.2,
00197 |                         "peak_gpu_memory_mb_mean": 1200.0,
00198 |                         "robust_claim_allowed_by_comparability": True,
00199 |                     },
00200 |                 ],
00201 |                 "winner_decision": {"robust_claim_allowed": True, "gates_failed": []},
00202 |             },
00203 |         )
00204 |         evidence = self.run_root / "deeph" / "raw_global_equivalence_evidence.json"
00205 |         write_json(evidence, {"equivalence_status": "proven", "equivalence_scope": "raw_global"})
00206 |         adapter = self.run_root / "deeph" / "adapter_manifest.json"
00207 |         write_json(
00208 |             adapter,
00209 |             {
00210 |                 "schema": "deeph_hdf5_prediction_adapter_v1",
00211 |                 "robust_matrix_metrics_allowed": True,
00212 |                 "adapter_equivalence_statuses": [EQUIVALENCE_PROVEN_RAW_GLOBAL],
00213 |                 "equivalence_statuses": ["proven"],
00214 |                 "equivalence_scopes": ["raw_global"],
00215 |                 "equivalence_evidence_paths": [str(evidence)],
00216 |                 "equivalence_gate": {"robust_claim_allowed": True, "diagnostic_only": False},
00217 |             },
00218 |         )
00219 |         required_files = [
00220 |             ("protocol", self.protocol_path),
00221 |             ("benchmark_dataset_manifest", self.dataset_root / "benchmark_dataset_manifest.json"),
00222 |             ("frozen_split_manifest", self.dataset_root / "frozen_split_manifest.json"),
00223 |             ("artifact_validation", self.dataset_root / "artifact_validation.json"),
00224 |             ("final_statistics", self.workflow_root / "final_test" / "final_statistics.json"),
00225 |             ("deeph_adapter_manifest", adapter),
00226 |         ]
00227 |         write_json(
00228 |             self.workflow_root / "evidence" / "evidence_bundle_manifest.json",
00229 |             {
00230 |                 "schema": "graph2mat_deeph_final_evidence_bundle_v1",
00231 |                 "status": "complete",
00232 |                 "missing_required": [],
00233 |                 "files": [
00234 |                     {
00235 |                         "label": label,
00236 |                         "path": str(path),
00237 |                         "required": True,
00238 |                         "exists": path.exists(),
00239 |                     }
00240 |                     for label, path in required_files
00241 |                 ],
00242 |             },
00243 |         )
00244 | 
00245 |     def gate_status(self) -> dict:
00246 |         return build_gate_status(
00247 |             protocol_path=self.protocol_path,
00248 |             workflow_root=self.workflow_root,
00249 |             run_root=self.run_root,
00250 |         )
00251 | 
00252 |     def test_all_pass_synthetic_fixture_allows_robust_claim(self) -> None:
00253 |         status = self.gate_status()
00254 | 
00255 |         self.assertTrue(status["robust_claim_allowed"])
00256 |         self.assertEqual(status["claim_status"], "robust_allowed")
00257 |         self.assertFalse([gate for gate in status["gates"] if gate["status"] == "fail"])
00258 | 
00259 |     def test_missing_frozen_split_manifest_blocks_robust_claim(self) -> None:
00260 |         (self.dataset_root / "frozen_split_manifest.json").unlink()
00261 | 
00262 |         status = self.gate_status()
00263 | 
00264 |         self.assertFalse(status["robust_claim_allowed"])
00265 |         self.assertEqual(status["claim_status"], "invalid_missing_evidence")
00266 |         self.assertIn("dataset_joint_a_manifests_present", {gate["id"] for gate in status["gates"] if gate["status"] == "fail"})
00267 | 
00268 |     def test_missing_final_evaluation_blocks_paper_ready_claim(self) -> None:
00269 |         protocol = protocol_payload(self.dataset_root)
00270 |         protocol.pop("final_evaluation")
00271 |         write_json(self.protocol_path, protocol)
00272 | 
00273 |         status = self.gate_status()
00274 | 
00275 |         self.assertFalse(status["robust_claim_allowed"])
00276 |         self.assertEqual(status["claim_status"], "invalid_protocol")
00277 |         self.assertIn("protocol_valid", {gate["id"] for gate in status["gates"] if gate["status"] == "fail"})
00278 | 
00279 |     def test_deeph_adapter_unproven_blocks_robust_claim(self) -> None:
00280 |         adapter = self.run_root / "deeph" / "adapter_manifest.json"
00281 |         write_json(
00282 |             adapter,
00283 |             {
00284 |                 "schema": "deeph_hdf5_prediction_adapter_v1",
00285 |                 "robust_matrix_metrics_allowed": False,
00286 |                 "adapter_equivalence_statuses": ["invalid_orbital_order_unknown"],
00287 |                 "equivalence_statuses": ["unproven"],
00288 |                 "equivalence_scopes": ["deeph_processed_blockwise_global_hdf5"],
00289 |                 "equivalence_gate": {"robust_claim_allowed": False, "diagnostic_only": True},
00290 |             },
00291 |         )
00292 | 
00293 |         status = self.gate_status()
00294 | 
00295 |         self.assertFalse(status["robust_claim_allowed"])
00296 |         self.assertEqual(status["claim_status"], "invalid_equivalence")
00297 |         self.assertIn("deeph_equivalence_proven", {gate["id"] for gate in status["gates"] if gate["status"] == "fail"})
00298 | 
00299 |     def test_missing_telemetry_blocks_cost_claim(self) -> None:
00300 |         stats_path = self.workflow_root / "final_test" / "final_statistics.json"
00301 |         stats = json.loads(stats_path.read_text(encoding="utf-8"))
00302 |         stats["final_seed_summary"][1].pop("gpu_hours_mean")
00303 |         write_json(stats_path, stats)
00304 | 
00305 |         status = self.gate_status()
00306 | 
00307 |         self.assertFalse(status["robust_claim_allowed"])
00308 |         self.assertEqual(status["claim_status"], "invalid_telemetry")
00309 |         self.assertIn("telemetry_complete", {gate["id"] for gate in status["gates"] if gate["status"] == "fail"})
00310 | 
00311 |     def test_ml_prediction_reference_path_blocks_dataset_gate(self) -> None:
00312 |         split_path = self.dataset_root / "frozen_split_manifest.json"
00313 |         split = json.loads(split_path.read_text(encoding="utf-8"))
00314 |         split["rows"][2]["artifact_paths"]["reference_hsx"] = "ML_prediction.HSX"
00315 |         write_json(split_path, split)
00316 | 
00317 |         status = self.gate_status()
00318 | 
00319 |         self.assertFalse(status["robust_claim_allowed"])
00320 |         self.assertEqual(status["claim_status"], "invalid_dataset")
00321 |         self.assertIn("forbidden_reference_absent", {gate["id"] for gate in status["gates"] if gate["status"] == "fail"})
00322 | 
00323 |     def test_malformed_json_returns_nonzero_cli_exit(self) -> None:
00324 |         self.protocol_path.write_text("{\n", encoding="utf-8")
00325 | 
00326 |         exit_code = main(
00327 |             [
00328 |                 "--protocol",
00329 |                 str(self.protocol_path),
00330 |                 "--workflow-root",
00331 |                 str(self.workflow_root),
00332 |                 "--run-root",
00333 |                 str(self.run_root),
00334 |                 "--output",
00335 |                 str(self.output_path),
00336 |             ]
00337 |         )
00338 | 
00339 |         self.assertEqual(exit_code, 1)
00340 |         payload = json.loads(self.output_path.read_text(encoding="utf-8"))
00341 |         self.assertFalse(payload["robust_claim_allowed"])
00342 |         self.assertEqual(payload["claim_status"], "invalid_protocol")
00343 | 
00344 | 
00345 | if __name__ == "__main__":
00346 |     unittest.main()
```

## `Comparison/scripts/g2m_deeph_paper_diagnostics.py` — extractos seleccionados

SHA-256 del archivo completo: `4d095b29a9d13b21b3e4caa07f553856497f0476451c18b134700233654d0792`

### Cabecera, imports y constantes iniciales

```py
00001 | #!/usr/bin/env python3
00002 | """Post-process existing G2M-vs-DeepH artifacts into paper-style diagnostics.
00003 | 
00004 | This script is intentionally read-only with respect to benchmark computation:
00005 | it does not train, infer, run SIESTA, run Graph2Mat, run DeepH, or materialize
00006 | new Hamiltonian/eigenvalue predictions. It only reads existing CSV/JSON
00007 | artifacts and writes derived plots and summaries.
00008 | """
00009 | 
00010 | from __future__ import annotations
00011 | 
00012 | import argparse
00013 | import csv
00014 | import json
00015 | import math
00016 | import statistics
00017 | import sys
00018 | import time
00019 | from collections import defaultdict
00020 | from pathlib import Path
00021 | from typing import Any
00022 | 
00023 | 
00024 | REPO_ROOT = Path(__file__).resolve().parents[2]
00025 | DEFAULT_IID600_ROOT = REPO_ROOT / "Comparison" / "results" / "g2m_deeph_iid600_phaseB_intermediate_spectral_refine_v1"
00026 | DEFAULT_IID1000_ROOT = REPO_ROOT / "Comparison" / "results" / "g2m_deeph_iid1000_phaseB_transfer_spectral_refine_v1"
00027 | DEFAULT_IID600_RUN = "paper_ready_final70_iid600_20260602_123539"
00028 | DEFAULT_IID1000_RUN = "paper_ready_final70_iid1000_20260602_123539"
00029 | DEFAULT_BAND_ROOT = REPO_ROOT / "Comparison" / "results" / "graphene_band_comparison_winners"
00030 | 
00031 | WINNER_CONFIGS = {
00032 |     "iid600": {"deeph": "DH-T600-13", "graph2mat": "G2M-T600-26"},
00033 |     "iid1000": {"deeph": "DH-T1000-03", "graph2mat": "G2M-T1000-03"},
00034 | }
00035 | 
00036 | DATASET_LABELS = {
00037 |     "graphene_w90_phase1_iid600": "iid600",
00038 |     "graphene_w90_phase1_iid1000": "iid1000",
00039 | }
00040 | 
00041 | FORBIDDEN_COMPUTE_COMMANDS = (
00042 |     "deeph-train",
00043 |     "deeph-preprocess",
00044 |     "deeph-inference",
00045 |     "graph2mat fit",
00046 |     "graph2mat test",
00047 |     "graph2mat predict",
00048 |     "siesta",
00049 |     "gnubands",
00050 | )
00051 | 
00052 | PLOT_COLORS = {"deeph": "#d62728", "graph2mat": "#1f77b4", "siesta": "#111111"}
00053 | 
```

### `representative_rows` — líneas 229–244

```py
00229 | def representative_rows(rows: list[dict[str, str]], metric: str = "low_energy_rmse_eV") -> dict[tuple[str, str], dict[str, str]]:
00230 |     grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
00231 |     for row in rows:
00232 |         grouped[(dataset_key(row.get("dataset_id", "")), model_key(row.get("model", "")))].append(row)
00233 |     chosen: dict[tuple[str, str], dict[str, str]] = {}
00234 |     for group, items in grouped.items():
00235 |         values = [(metric_from_row(item, metric), item) for item in items]
00236 |         clean = [(value, item) for value, item in values if value is not None]
00237 |         if not clean:
00238 |             continue
00239 |         center = mean([value for value, _item in clean])
00240 |         if center is None:
00241 |             continue
00242 |         clean.sort(key=lambda item: abs(item[0] - center))
00243 |         chosen[group] = clean[0][1]
00244 |     return chosen
```

### `best_median_worst` — líneas 275–282

```py
00275 | def best_median_worst(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
00276 |     clean = [row for row in rows if number(row.get(metric)) is not None]
00277 |     clean.sort(key=lambda row: float(row[metric]))
00278 |     if not clean:
00279 |         return []
00280 |     mid = len(clean) // 2
00281 |     picks = [("best", clean[0]), ("median", clean[mid]), ("worst", clean[-1])]
00282 |     return [{**row, "rank_label": label, "rank_metric": metric} for label, row in picks]
```

### `aggregate_rows` — líneas 285–301

```py
00285 | def aggregate_rows(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
00286 |     grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
00287 |     for row in rows:
00288 |         grouped[tuple(row.get(key) for key in keys)].append(row)
00289 |     out: list[dict[str, Any]] = []
00290 |     for key_values, items in sorted(grouped.items()):
00291 |         result = {key: value for key, value in zip(keys, key_values)}
00292 |         result["n"] = len(items)
00293 |         for metric in metrics:
00294 |             values = [float(item[metric]) for item in items if number(item.get(metric)) is not None]
00295 |             result[f"{metric}_mean"] = mean(values)
00296 |             result[f"{metric}_std"] = std(values)
00297 |             result[f"{metric}_min"] = min(values) if values else None
00298 |             result[f"{metric}_median"] = percentile(values, 0.5)
00299 |             result[f"{metric}_max"] = max(values) if values else None
00300 |         out.append(result)
00301 |     return out
```

### `linear_regression_summary` — líneas 304–326

```py
00304 | def linear_regression_summary(x_values: list[float], y_values: list[float]) -> dict[str, Any]:
00305 |     pairs = [(x, y) for x, y in zip(x_values, y_values) if math.isfinite(x) and math.isfinite(y)]
00306 |     if len(pairs) < 2:
00307 |         return {"n": len(pairs), "r2": None, "slope": None, "intercept": None, "mae": None, "rmse": None}
00308 |     xs = [item[0] for item in pairs]
00309 |     ys = [item[1] for item in pairs]
00310 |     x_mean = mean(xs) or 0.0
00311 |     y_mean = mean(ys) or 0.0
00312 |     denom = sum((x - x_mean) ** 2 for x in xs)
00313 |     slope = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denom if denom else 0.0
00314 |     intercept = y_mean - slope * x_mean
00315 |     pred = [slope * x + intercept for x in xs]
00316 |     ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
00317 |     ss_tot = sum((y - y_mean) ** 2 for y in ys)
00318 |     errors = [y - x for x, y in pairs]
00319 |     return {
00320 |         "n": len(pairs),
00321 |         "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0,
00322 |         "slope": slope,
00323 |         "intercept": intercept,
00324 |         "mae": mean([abs(error) for error in errors]),
00325 |         "rmse": math.sqrt(mean([error * error for error in errors]) or 0.0),
00326 |     }
```

### `gate_release_rows` — líneas 558–586

```py
00558 | def gate_release_rows(dataset_roots: dict[str, Path]) -> list[dict[str, Any]]:
00559 |     rows: list[dict[str, Any]] = []
00560 |     for key, root in dataset_roots.items():
00561 |         gate = read_json(root / "gate_status.json")
00562 |         release = read_json(root / "release_manifest.json")
00563 |         evidence_files = list((root / "equivalence_strict").glob("*/deeph_raw_global_equivalence_preflight.json"))
00564 |         proven = 0
00565 |         failed = 0
00566 |         for path in evidence_files:
00567 |             data = read_json(path)
00568 |             status = str(data.get("equivalence_status") or data.get("status") or "").lower()
00569 |             if status == "proven":
00570 |                 proven += 1
00571 |             elif status:
00572 |                 failed += 1
00573 |         rows.append(
00574 |             {
00575 |                 "dataset_key": key,
00576 |                 "gate_claim_status": gate.get("claim_status"),
00577 |                 "gate_robust_claim_allowed": gate.get("robust_claim_allowed"),
00578 |                 "gate_blockers": "; ".join(gate.get("blockers", [])) if isinstance(gate.get("blockers"), list) else "",
00579 |                 "release_status": release.get("status") or release.get("strict_status"),
00580 |                 "release_strict": release.get("strict") or release.get("strict_mode"),
00581 |                 "equivalence_files": len(evidence_files),
00582 |                 "equivalence_proven": proven,
00583 |                 "equivalence_failed": failed,
00584 |             }
00585 |         )
00586 |     return rows
```

### `build_diagnostics` — líneas 964–1095

```py
00964 | def build_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
00965 |     output_dir = args.output_dir
00966 |     output_dir.mkdir(parents=True, exist_ok=True)
00967 |     formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
00968 | 
00969 |     dataset_roots = {"iid600": args.iid600_root, "iid1000": args.iid1000_root}
00970 |     run_names = {"iid600": args.run_iid600, "iid1000": args.run_iid1000}
00971 |     warnings: list[dict[str, Any]] = []
00972 |     outputs: list[str] = []
00973 | 
00974 |     all_rows: list[dict[str, str]] = []
00975 |     for key, root in dataset_roots.items():
00976 |         rows = load_final_rows(root, run_names[key])
00977 |         if not rows:
00978 |             warnings.append({"kind": "missing_artifact", "dataset_key": key, "message": f"No final training metrics found under {root}"})
00979 |         all_rows.extend(selected_winner_rows(rows, key))
00980 | 
00981 |     if not all_rows:
00982 |         raise RuntimeError("No winner rows were found in existing final metrics.")
00983 | 
00984 |     seed_rows = load_seed_rows(all_rows)
00985 |     dos_rows = load_dos_sample_metrics(all_rows)
00986 |     dos_summary_rows = aggregate_rows(dos_rows, ["dataset_key", "model", "selected_config_id"], ["dos_mae_500_fermi_window", "dos_wasserstein_eV"])
00987 |     dos_rank_rows: list[dict[str, Any]] = []
00988 |     for key in sorted({row["dataset_key"] for row in dos_rows}):
00989 |         for model in ["graph2mat", "deeph"]:
00990 |             model_rows = [row for row in dos_rows if row["dataset_key"] == key and row["model"] == model]
00991 |             dos_rank_rows.extend(best_median_worst(model_rows, "dos_mae_500_fermi_window"))
00992 | 
00993 |     orbital_rows = load_orbital_pair_rows(all_rows)
00994 |     orbital_summary_rows = aggregate_rows(
00995 |         orbital_rows,
00996 |         ["dataset_key", "model", "selected_config_id", "row_orbital_label", "col_orbital_label"],
00997 |         ["mae_union_meV", "rmse_union_eV", "r2_union", "mean_abs_ref_eV"],
00998 |     )
00999 |     if not orbital_rows:
01000 |         warnings.append(
01001 |             {
01002 |                 "kind": "missing_artifact",
01003 |                 "message": (
01004 |                     "No non-empty orbital_pair_metrics.csv rows were found; orbital-pair heatmaps were skipped. "
01005 |                     "Graph2Mat source manifests report missing .ion.xml basis files, and DeepH has no orbital_pair_metrics.csv."
01006 |                 ),
01007 |             }
01008 |         )
01009 |     elif not any(row.get("model") == "deeph" for row in orbital_rows):
01010 |         warnings.append(
01011 |             {
01012 |                 "kind": "missing_artifact",
01013 |                 "message": "DeepH orbital_pair_metrics.csv was not found; orbital-pair heatmaps are Graph2Mat-only.",
01014 |             }
01015 |         )
01016 | 
01017 |     matrix_rows = load_kpoint_matrix_rows(all_rows)
01018 |     matrix_summary_rows = aggregate_rows(
01019 |         matrix_summary_source_rows(matrix_rows),
01020 |         ["dataset_key", "model", "selected_config_id"],
01021 |         ["h_mae_eV", "h_rmse_eV", "relative_frobenius", "hermiticity_pred"],
01022 |     )
01023 |     band_rows = load_band_residual_rows(args.band_root)
01024 |     if not band_rows:
01025 |         warnings.append({"kind": "missing_artifact", "message": f"No existing band residual CSVs found under {args.band_root}"})
01026 |     dirac_rows = load_dirac_rows(args.band_root)
01027 |     if not dirac_rows:
01028 |         warnings.append({"kind": "missing_artifact", "message": f"No existing dirac_diagnostic.json files found under {args.band_root}"})
01029 |     gate_rows = gate_release_rows(dataset_roots)
01030 | 
01031 |     table_rows = {
01032 |         "seed_metrics.csv": seed_rows,
01033 |         "dos_sample_metrics.csv": dos_rows,
01034 |         "dos_summary.csv": dos_summary_rows,
01035 |         "dos_best_median_worst.csv": dos_rank_rows,
01036 |         "orbital_pair_summary.csv": orbital_summary_rows,
01037 |         "matrix_metric_summary.csv": matrix_summary_rows,
01038 |         "band_residuals.csv": band_rows,
01039 |         "dirac_diagnostics.csv": dirac_rows,
01040 |         "equivalence_gate_release_table.csv": gate_rows,
01041 |     }
01042 |     for filename, rows in table_rows.items():
01043 |         write_csv(output_dir / filename, rows)
01044 | 
01045 |     if dos_rows:
01046 |         run_plot_step(outputs, warnings, "dos_distribution", plot_dos_distribution, dos_rows, output_dir, formats)
01047 |     else:
01048 |         warnings.append({"kind": "missing_artifact", "message": "No DOS metric rows found; DOS distribution plots were skipped."})
01049 |     run_plot_step(outputs, warnings, "seed_uncertainty", plot_seed_uncertainty, seed_rows, output_dir, formats)
01050 |     run_plot_step(outputs, warnings, "pareto", plot_pareto, seed_rows, output_dir, formats)
01051 |     if orbital_rows:
01052 |         run_plot_step(outputs, warnings, "orbital_pair_heatmaps", plot_orbital_pair_heatmaps, orbital_rows, output_dir, formats)
01053 |     if matrix_rows:
01054 |         run_plot_step(outputs, warnings, "matrix_metric_distribution", plot_matrix_metric_distribution, matrix_rows, output_dir, formats)
01055 |     if band_rows:
01056 |         run_plot_step(outputs, warnings, "band_residuals", plot_band_residuals, band_rows, output_dir, formats)
01057 |     if dirac_rows:
01058 |         run_plot_step(outputs, warnings, "dirac_diagnostics", plot_dirac, dirac_rows, output_dir, formats)
01059 | 
01060 |     if not any("dos_curve" in row for row in dos_rows):
01061 |         warnings.append(
01062 |             {
01063 |                 "kind": "missing_artifact",
01064 |                 "message": "No stored full DOS curves were discovered; best/median/worst DOS overlays were not generated.",
01065 |             }
01066 |         )
01067 | 
01068 |     manifest = {
01069 |         "script": "Comparison/scripts/g2m_deeph_paper_diagnostics.py",
01070 |         "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
01071 |         "post_processing_only": True,
01072 |         "strict_non_compute_rules": {
01073 |             "no_training": True,
01074 |             "no_inference": True,
01075 |             "no_siesta": True,
01076 |             "no_graph2mat_cli": True,
01077 |             "no_deeph_cli": True,
01078 |             "forbidden_compute_commands": list(FORBIDDEN_COMPUTE_COMMANDS),
01079 |         },
01080 |         "inputs": {
01081 |             "iid600_root": str(args.iid600_root),
01082 |             "iid1000_root": str(args.iid1000_root),
01083 |             "run_iid600": args.run_iid600,
01084 |             "run_iid1000": args.run_iid1000,
01085 |             "band_root": str(args.band_root),
01086 |             "winner_configs": WINNER_CONFIGS,
01087 |         },
01088 |         "outputs": outputs,
01089 |         "tables": {name: str(output_dir / name) for name in table_rows},
01090 |         "warnings": warnings,
01091 |         "status": "ok" if outputs else "tables_only",
01092 |     }
01093 |     write_json(output_dir / "paper_diagnostics_manifest.json", manifest)
01094 |     summarize_outputs(output_dir, outputs, warnings, table_rows)
01095 |     return manifest
```
