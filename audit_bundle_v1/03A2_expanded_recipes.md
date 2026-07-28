# Dossier 1B — Recetas expandidas de datasets

## Objeto de revisión

Auditar composición y escalado de las recetas MD, FC y random Cartesian, incluyendo temperaturas, amplitudes, seeds, tamaños y repetición de bloques.

## Condiciones del contexto

- Contexto ciego: no contiene informes de auditoría previos ni prompts históricos.
- Las líneas prefijadas `NNNNN |` conservan la numeración del archivo fuente.
- Los extractos Python omiten funciones operativas sin semántica científica directa.
- Ausencia de un archivo o función en este dossier significa `no evaluable`, no `correcto`.

## `MD/pipeline_config.yaml`

SHA-256: `668b7c5adf47e1e10480c5cd8d1ac68c93e9dd130e20b6a96eefa470c55ec240`

```yaml
00001 | paths:
00002 |   dataset_dir: ${REPO_ROOT}/MD/dataset
00003 |   training_dir: ${REPO_ROOT}/MD/training
00004 |   run_fdf_name: RUN.fdf
00005 |   run_out_name: RUN.out
00006 |   training_config_name: config.yaml
00007 |   venv_activate: ${REPO_ROOT}/.venv/bin/activate
00008 | material:
00009 |   preset: graphene
00010 | commands:
00011 |   graph2mat: graph2mat
00012 |   siesta: siesta
00013 |   shell: bash
00014 | md:
00015 |   system_name: Graphene MD dataset
00016 |   system_label: siesta
00017 |   run_fdf_template: ${REPO_ROOT}/materials/graphene/RUN.fdf
00018 |   type_of_run: verlet
00019 |   # 70 = 6+1+1 split frames + 2*30 temporal-gap frames (see splits below).
00020 |   steps: 70
00021 |   temperature_K: 300.0
00022 |   timestep_fs: 1.0
00023 |   ensemble: nve
00024 |   thermostat: none
00025 |   write_md_history: true
00026 |   temperature_blocks: []
00027 |   basis_size: SZP
00028 |   basis_type: split
00029 |   energy_shift: 0.275 eV
00030 |   mesh_cutoff: 250 Ry
00031 |   xc_functional: LDA
00032 |   xc_authors: PZ
00033 |   max_scf_iterations: 200
00034 |   solution_method: diagon
00035 |   dm_mixing_weight: 0.02
00036 |   dm_number_pulay: 3
00037 |   dm_tolerance: 1.d-5
00038 |   dm_require_energy_convergence: T
00039 |   dm_energy_tolerance: 1.e-5 eV
00040 |   spin_polarized: F
00041 |   fix_spin: F
00042 |   non_collinear_spin: F
00043 |   xml_write: true
00044 |   save_hs_file: true
00045 |   save_hs: true
00046 |   save_de: true
00047 |   lua_script: md_store.lua
00048 |   force_aux_cell: true
00049 |   lattice_constant:
00050 |     value: 1.0
00051 |     unit: Ang
00052 |   lattice_vectors:
00053 |   - - 6.5025
00054 |     - 3.75422012541
00055 |     - 0.0
00056 |   - - 2.1675
00057 |     - -1.25140670847
00058 |     - 0.0
00059 |   - - 0.0
00060 |     - 0.0
00061 |     - 14.45
00062 |   species:
00063 |   - index: 1
00064 |     atomic_number: 6
00065 |     symbol: C
00066 |   coordinates_format: Ang
00067 |   kgrid_monkhorst_pack:
00068 |   - - 1
00069 |     - 0
00070 |     - 0
00071 |     - 0.0
00072 |   - - 0
00073 |     - 1
00074 |     - 0
00075 |     - 0.0
00076 |   - - 0
00077 |     - 0
00078 |     - 1
00079 |     - 0.0
00080 |   atoms:
00081 |   - label: C
00082 |     species_index: 1
00083 |     position:
00084 |     - 0.0
00085 |     - 0.0
00086 |     - 0.0
00087 |   - label: C
00088 |     species_index: 1
00089 |     position:
00090 |     - 1.445
00091 |     - 0.0
00092 |     - 0.0
00093 |   - label: C
00094 |     species_index: 1
00095 |     position:
00096 |     - 2.1675
00097 |     - 1.25141
00098 |     - 0.0
00099 |   - label: C
00100 |     species_index: 1
00101 |     position:
00102 |     - 3.6125
00103 |     - 1.25141
00104 |     - 0.0
00105 |   - label: C
00106 |     species_index: 1
00107 |     position:
00108 |     - 4.335
00109 |     - 2.50281
00110 |     - 0.0
00111 |   - label: C
00112 |     species_index: 1
00113 |     position:
00114 |     - 5.78
00115 |     - 2.50281
00116 |     - 0.0
00117 | training:
00118 |   torch_float32_matmul_precision: null
00119 |   data:
00120 |     out_matrix: hamiltonian
00121 |     symmetric_matrix: true
00122 |     sub_point_matrix: false
00123 |     matrix_component_policy: h_only
00124 |     n_matrix_components: 1
00125 |     basis_files: ../dataset/MD_steps/basis/*.ion.xml
00126 |     train_runs: ../dataset/splits/train/*/RUN.fdf
00127 |     val_runs: ../dataset/splits/validation/*/RUN.fdf
00128 |     batch_size: 10
00129 |     store_in_memory: true
00130 |   model:
00131 |     num_interactions: 1
00132 |     correlation: 1
00133 |     max_ell: 2
00134 |     hidden_irreps: 10x0e + 10x1o + 10x2e
00135 |     loss: graph2mat.metrics.block_type_mae
00136 |     optim_lr: 0.005
00137 |   trainer:
00138 |     accelerator: cpu
00139 |     logger:
00140 |       class_path: TensorBoardLogger
00141 |       init_args:
00142 |         name: my_first_model
00143 |         save_dir: lightning_logs
00144 |     max_epochs: 200
00145 | checkpoint:
00146 |   path: null
00147 |   auto_best: true
00148 |   search_glob: lightning_logs/**/checkpoints/best-*.ckpt
00149 |   selection: latest_version
00150 | testing:
00151 |   test_runs: ../dataset/splits/test/*/RUN.fdf
00152 |   callbacks:
00153 |     plot_matrix_error: false
00154 |     show_plot: false
00155 |     samplewise_metrics_logger: false
00156 | prediction:
00157 |   predict_structs: ../dataset/splits/test/*/RUN.fdf
00158 |   output_file: ML_prediction.HSX
00159 |   callbacks:
00160 |     matrix_writer: true
00161 | pipeline:
00162 |   skip_model_test: false
00163 |   steps:
00164 |   - generate_md_dataset
00165 |   - run_md_training
00166 |   - run_md_testing
00167 |   - run_md_prediction
00168 | performance:
00169 |   max_parallel_siesta_jobs: 1
00170 |   max_parallel_dataset_jobs: 1
00171 |   max_parallel_prediction_jobs: 1
00172 |   max_parallel_evaluation_jobs: 1
00173 |   max_parallel_metric_jobs: 1
00174 |   omp_num_threads: null
00175 |   mkl_num_threads: null
00176 |   openblas_num_threads: null
00177 |   numexpr_num_threads: null
00178 |   torch_num_threads: null
00179 |   compute_accelerator: cpu
00180 |   batch_size: null
00181 |   store_in_memory: null
00182 |   reuse_validated_siesta_outputs: true
00183 |   enable_experiment_cache: false
00184 |   error_policy: fail_fast
00185 |   preset: null
00186 |   torch_float32_matmul_precision: null
00187 | splits:
00188 |   enabled: true
00189 |   # Leakage-safer scientific default for MD trajectories: consecutive frames
00190 |   # are 1 fs apart, and carbon vibrational periods are ~20-40 fs, so a gap of
00191 |   # 30 frames (~30 fs, about one vibrational period) is the minimum for the
00192 |   # buffered blocks to be meaningfully decorrelated. A gap of 1 frame (1 fs)
00193 |   # is physically meaningless. md.steps must cover
00194 |   # train+validation+test + 2*temporal_gap frames.
00195 |   strategy: blocked_with_gap
00196 |   temporal_gap: 30
00197 |   block_order: train,validation,test
00198 |   train: 6
00199 |   validation: 1
00200 |   test: 1
```

## `AtomDisplacement/pipeline_config.yaml`

SHA-256: `7904a92df49cf33a117252e6371c051b404b2c8924c1a759c956551f7ae5abed`

```yaml
00001 | paths:
00002 |   base_dir: base
00003 |   relaxed_dir: relaxed
00004 |   dataset_dir: dataset
00005 |   samples_dir: dataset/samples
00006 |   collected_dir: dataset/collected
00007 |   training_dir: training
00008 |   run_fdf_name: RUN.fdf
00009 |   run_out_name: RUN.out
00010 |   training_config_name: config.yaml
00011 |   runs_json_name: runs.json
00012 |   samples_manifest_name: samples_manifest.json
00013 |   run_summary_name: run_summary.json
00014 |   collected_json_name: water_atom_displacement_dataset.json
00015 |   collected_csv_name: water_atom_displacement_summary.csv
00016 |   venv_activate: ${REPO_ROOT}/.venv/bin/activate
00017 | material:
00018 |   preset: h2o
00019 | atomic_displacement:
00020 |   recipe: generic_cartesian
00021 |   amplitude_ang: 0.03
00022 |   selected_species: null
00023 |   include_base: true
00024 |   overwrite: false
00025 | commands:
00026 |   graph2mat: graph2mat
00027 |   siesta: siesta
00028 |   shell: bash
00029 |   python: python
00030 | generation:
00031 |   sample_id_format: sample_{index:04d}
00032 | structure:
00033 |   molecule_name: H2O
00034 |   relaxation:
00035 |     system_name: H2O molecule relaxation
00036 |     system_label: h2o_relax
00037 |   single_point:
00038 |     system_name_template: H2O {sample_id}
00039 |     title: Force-constant calculation for H2O atom displacements
00040 |   force_constants:
00041 |     enabled: true
00042 |     displacement: 0.05 Ang
00043 |     target_count: null
00044 |     include_reference: true
00045 |     expand_amplitudes: false
00046 |     subsampling:
00047 |       method: spread
00048 |       seed: 0
00049 |     random_seed: 42
00050 |     combination_mode: aligned
00051 |     max_datasets: 100
00052 |     displacements:
00053 |       0.02 Ang:
00054 |       - 5
00055 |       - 7
00056 |       - 9
00057 |       0.03 Ang:
00058 |       - 4
00059 |       - 6
00060 |       - 8
00061 |       0.05 Ang:
00062 |       - 2
00063 |       - 10
00064 |       - 13
00065 |     allow_missing_matrix: true
00066 |     first_atom: 1
00067 |     last_atom: null
00068 |     lua_script: md_store.lua
00069 |     save_tshs: true
00070 |     save_tsde: true
00071 |     save_dhs: true
00072 |     dHdR_tolerance: -1 Ry/Bohr
00073 |     dSdR_tolerance: -1 1/Bohr
00074 |   random_cartesian:
00075 |     enabled: false
00076 |     recipe: legacy_components
00077 |     n_structures: 100
00078 |     seed: 1234
00079 |     distribution: gaussian
00080 |     sigma_ang: 0.03
00081 |     uniform_range_ang: 0.05
00082 |     move_atoms: all
00083 |     species_filter: []
00084 |     min_distance_ang: 0.65
00085 |     max_attempts_per_structure: 100
00086 |     remove_center_of_mass_translation: true
00087 |     components:
00088 |       atom_displacement:
00089 |         enabled: true
00090 |         distribution: gaussian
00091 |         sigma_ang: 0.03
00092 |         uniform_range_ang: 0.05
00093 |         move_atoms: all
00094 |         species_filter: []
00095 |         remove_center_of_mass_translation: true
00096 |       bond_displacement:
00097 |         enabled: false
00098 |         distribution: gaussian
00099 |         sigma_ang: 0.01
00100 |         uniform_range_ang: 0.02
00101 |         min_delta_ang: null
00102 |         max_delta_ang: null
00103 |         min_bond_ang: 0.70
00104 |         max_bond_ang: 1.30
00105 |         bonds: h2o_oh
00106 |       angle_displacement:
00107 |         enabled: false
00108 |         distribution: gaussian
00109 |         sigma_deg: 3.0
00110 |         uniform_range_deg: 5.0
00111 |         min_delta_deg: null
00112 |         max_delta_deg: null
00113 |         min_angle_deg: 80.0
00114 |         max_angle_deg: 130.0
00115 |         angles: h2o_hoh
00116 |     validation:
00117 |       min_distance_ang: 0.65
00118 |       max_rmsd_from_reference_ang: null
00119 |       max_attempts_per_structure: 100
00120 |   lattice_constant:
00121 |     value: 1.0
00122 |     unit: Ang
00123 |   lattice_vectors:
00124 |   - - 15.0
00125 |     - 0.0
00126 |     - 0.0
00127 |   - - 0.0
00128 |     - 15.0
00129 |     - 0.0
00130 |   - - 0.0
00131 |     - 0.0
00132 |     - 15.0
00133 |   coordinates_format: Ang
00134 |   species:
00135 |   - index: 1
00136 |     atomic_number: 8
00137 |     symbol: O
00138 |   - index: 2
00139 |     atomic_number: 1
00140 |     symbol: H
00141 |   atoms:
00142 |   - label: O
00143 |     species_index: 1
00144 |     position:
00145 |     - 7.5
00146 |     - 7.5
00147 |     - 7.619262
00148 |   - label: H
00149 |     species_index: 2
00150 |     position:
00151 |     - 7.5
00152 |     - 8.263239
00153 |     - 7.022953
00154 |   - label: H
00155 |     species_index: 2
00156 |     position:
00157 |     - 7.5
00158 |     - 6.836839
00159 |     - 7.022953
00160 |   kgrid_monkhorst_pack:
00161 |   - - 1
00162 |     - 0
00163 |     - 0
00164 |     - 0.0
00165 |   - - 0
00166 |     - 1
00167 |     - 0
00168 |     - 0.0
00169 |   - - 0
00170 |     - 0
00171 |     - 1
00172 |     - 0.0
00173 |   siesta:
00174 |     ForceAuxCell: T
00175 |     MeshCutoff: 200 Ry
00176 |     PAO.BasisType: split
00177 |     PAO.BasisSize: DZP
00178 |     PAO.EnergyShift: 0.03 eV
00179 |     XC.functional: GGA
00180 |     XC.authors: PBE
00181 |     MaxSCFIterations: 200
00182 |     SolutionMethod: diagon
00183 |     DM.MixingWeight: 0.02
00184 |     DM.NumberPulay: 3
00185 |     DM.Tolerance: 1.d-5
00186 |     DM.Require.Energy.Convergence: T
00187 |     DM.Energy.Tolerance: 1.e-5 eV
00188 |     SCF.MixAfterConvergence: F
00189 |     SpinPolarized: F
00190 |     FixSpin: F
00191 |     NonCollinearSpin: F
00192 |     DM.InitSpinAF: F
00193 |     ON.UseSaveLWF: T
00194 |     DM.UseSaveDM: T
00195 |     UseSaveData: T
00196 |     LongOutput: T
00197 |     WriteCoorXmol: T
00198 |     WriteCoorStep: T
00199 |     WriteForces: T
00200 |     Save.HS: T
00201 |     XML.Write: T
00202 |   relaxation_md:
00203 |     MD.TypeOfRun: CG
00204 |     MD.NumCGsteps: 200
00205 |     MD.MaxForceTol: 0.04 eV/Ang
00206 |     MD.MaxCGDispl: 0.03 Bohr
00207 |     MD.VariableCell: F
00208 |     MD.ConstantVolume: T
00209 |     MD.UseSaveXV: T
00210 |     MD.UseSaveCG: T
00211 |     WriteMDHistory: T
00212 |   single_point_overrides:
00213 |     DM.UseSaveDM: F
00214 | splits:
00215 |   train: 0.8
00216 |   validation: 0.1
00217 |   test: 0.1
00218 | single_points:
00219 |   limit: null
00220 |   rerun: false
00221 |   # Scientific hardening default: references are reusable only when RUN.out
00222 |   # proves job completion and SCF convergence. Use the CLI debug flag only for
00223 |   # unsafe local recovery/debug workflows.
00224 |   allow_unvalidated_matrices: false
00225 |   workers: 1
00226 | training:
00227 |   torch_float32_matmul_precision: null
00228 |   data:
00229 |     out_matrix: hamiltonian
00230 |     symmetric_matrix: true
00231 |     sub_point_matrix: false
00232 |     matrix_component_policy: h_only
00233 |     n_matrix_components: 1
00234 |     basis_files: ../dataset/FC_steps/basis/*.ion.xml
00235 |     runs_json: runs.json
00236 |     val_runs: ../dataset/validation_samples/*/RUN.fdf
00237 |     batch_size: 10
00238 |     store_in_memory: true
00239 |   model:
00240 |     num_interactions: 1
00241 |     correlation: 1
00242 |     max_ell: 2
00243 |     hidden_irreps: 10x0e + 10x1o + 10x2e
00244 |     loss: graph2mat.metrics.block_type_mae
00245 |     optim_lr: 0.005
00246 |   trainer:
00247 |     accelerator: cpu
00248 |     logger:
00249 |       class_path: TensorBoardLogger
00250 |       init_args:
00251 |         name: atom_displacement_model
00252 |         save_dir: lightning_logs
00253 |     max_epochs: 200
00254 |   fit_args:
00255 |   - models
00256 |   - mace
00257 |   - main
00258 |   - fit
00259 |   - -c
00260 |   - config.yaml
00261 |   min_completed_samples: 2
00262 |   tensorboard_hint: tensorboard --logdir lightning_logs
00263 | checkpoint:
00264 |   path: null
00265 |   auto_best: true
00266 |   search_glob: lightning_logs/**/checkpoints/best-*.ckpt
00267 |   selection: latest_version
00268 | testing:
00269 |   sample_index: 0
00270 |   data:
00271 |     out_matrix: hamiltonian
00272 |     symmetric_matrix: true
00273 |     sub_point_matrix: false
00274 |     basis_files: ../dataset/FC_steps/basis/*.ion.xml
00275 |     store_in_memory: true
00276 |     matrix_component_policy: h_only
00277 |     n_matrix_components: 1
00278 |   callbacks:
00279 |     plot_matrix_error: true
00280 |     show_plot: false
00281 |     store_plot_in_logger: false
00282 |     samplewise_metrics_logger: true
00283 |     output_file: sample_metrics.csv
00284 | prediction:
00285 |   data:
00286 |     out_matrix: hamiltonian
00287 |     symmetric_matrix: true
00288 |     sub_point_matrix: false
00289 |     basis_files: ../dataset/FC_steps/basis/*.ion.xml
00290 |     predict_structs: ../dataset/FC_steps/*/RUN.fdf
00291 |     store_in_memory: true
00292 |     matrix_component_policy: h_only
00293 |     n_matrix_components: 1
00294 |   callbacks:
00295 |     matrix_writer: true
00296 |     output_file: ML_prediction.HSX
00297 | pipeline:
00298 |   skip_model_test: false
00299 |   steps:
00300 |   - render_inputs
00301 |   - generate_atom_displacement_dataset
00302 |   - run_single_points
00303 |   - normalize_fc_steps
00304 |   - collect_atom_displacement_dataset
00305 |   - run_atdisp_training
00306 |   - run_atdisp_testing
00307 |   - run_atdisp_prediction
00308 | performance:
00309 |   max_parallel_siesta_jobs: 1
00310 |   max_parallel_dataset_jobs: 1
00311 |   max_parallel_prediction_jobs: 1
00312 |   max_parallel_evaluation_jobs: 1
00313 |   max_parallel_metric_jobs: 1
00314 |   omp_num_threads: null
00315 |   mkl_num_threads: null
00316 |   openblas_num_threads: null
00317 |   numexpr_num_threads: null
00318 |   torch_num_threads: null
00319 |   compute_accelerator: cpu
00320 |   batch_size: null
00321 |   store_in_memory: null
00322 |   reuse_validated_siesta_outputs: true
00323 |   enable_experiment_cache: false
00324 |   error_policy: fail_fast
00325 |   preset: null
00326 |   torch_float32_matmul_precision: null
```

## `Comparison/dataset_recipes/h2o_efficiency_reliable_3seed_train_200epochs.json` — vista compacta

SHA-256 del JSON completo: `c08c130ec6b2fd79fcb82da51546a5750305aac0374c69207d20359f1510bc8a`. Las listas de más de ocho elementos conservan longitud, dos primeros y dos últimos elementos; esta vista no sustituye al artefacto para verificaciones cuantitativas.

```json
00001 | {
00002 |   "md": [
00003 |     {
00004 |       "recipe_id": "md_h2o_weighted_temp_190",
00005 |       "label": "H2O MD weighted temperature ladder 190",
00006 |       "comparison_role": "initial_efficiency",
00007 |       "scientific_note": "Small MD baseline for initial efficiency checks. Weighted 300-1000 K ladder concentrated toward 300-700 K.",
00008 |       "blocks": {
00009 |         "_list_length": 15,
00010 |         "_first_two": [
00011 |           {
00012 |             "block_id": "md_T300",
00013 |             "n_snapshots": 18,
00014 |             "temperature_K": 300,
00015 |             "seed": 1001
00016 |           },
00017 |           {
00018 |             "block_id": "md_T350",
00019 |             "n_snapshots": 17,
00020 |             "temperature_K": 350,
00021 |             "seed": 1002
00022 |           }
00023 |         ],
00024 |         "_last_two": [
00025 |           {
00026 |             "block_id": "md_T950",
00027 |             "n_snapshots": 8,
00028 |             "temperature_K": 950,
00029 |             "seed": 1014
00030 |           },
00031 |           {
00032 |             "block_id": "md_T1000",
00033 |             "n_snapshots": 8,
00034 |             "temperature_K": 1000,
00035 |             "seed": 1015
00036 |           }
00037 |         ]
00038 |       }
00039 |     },
00040 |     {
00041 |       "recipe_id": "md_h2o_weighted_temp_380",
00042 |       "label": "H2O MD weighted temperature ladder 380",
00043 |       "comparison_role": "initial_efficiency",
00044 |       "scientific_note": "Intermediate small MD baseline. Weighted 300-1000 K ladder concentrated toward 300-700 K.",
00045 |       "blocks": {
00046 |         "_list_length": 15,
00047 |         "_first_two": [
00048 |           {
00049 |             "block_id": "md_T300",
00050 |             "n_snapshots": 36,
00051 |             "temperature_K": 300,
00052 |             "seed": 1051
00053 |           },
00054 |           {
00055 |             "block_id": "md_T350",
00056 |             "n_snapshots": 34,
00057 |             "temperature_K": 350,
00058 |             "seed": 1052
00059 |           }
00060 |         ],
00061 |         "_last_two": [
00062 |           {
00063 |             "block_id": "md_T950",
00064 |             "n_snapshots": 16,
00065 |             "temperature_K": 950,
00066 |             "seed": 1064
00067 |           },
00068 |           {
00069 |             "block_id": "md_T1000",
00070 |             "n_snapshots": 16,
00071 |             "temperature_K": 1000,
00072 |             "seed": 1065
00073 |           }
00074 |         ]
00075 |       }
00076 |     },
00077 |     {
00078 |       "recipe_id": "md_h2o_weighted_temp_570",
00079 |       "label": "H2O MD weighted temperature ladder 570",
00080 |       "comparison_role": "primary_baseline",
00081 |       "scientific_note": "Primary MD baseline for method-efficiency comparisons. Weighted 300-1000 K ladder concentrated toward 300-700 K.",
00082 |       "blocks": {
00083 |         "_list_length": 15,
00084 |         "_first_two": [
00085 |           {
00086 |             "block_id": "md_T300",
00087 |             "n_snapshots": 54,
00088 |             "temperature_K": 300,
00089 |             "seed": 1101
00090 |           },
00091 |           {
00092 |             "block_id": "md_T350",
00093 |             "n_snapshots": 51,
00094 |             "temperature_K": 350,
00095 |             "seed": 1102
00096 |           }
00097 |         ],
00098 |         "_last_two": [
00099 |           {
00100 |             "block_id": "md_T950",
00101 |             "n_snapshots": 24,
00102 |             "temperature_K": 950,
00103 |             "seed": 1114
00104 |           },
00105 |           {
00106 |             "block_id": "md_T1000",
00107 |             "n_snapshots": 24,
00108 |             "temperature_K": 1000,
00109 |             "seed": 1115
00110 |           }
00111 |         ]
00112 |       }
00113 |     },
00114 |     {
00115 |       "recipe_id": "md_h2o_weighted_temp_760",
00116 |       "label": "H2O MD weighted temperature ladder 760",
00117 |       "comparison_role": "strong_baseline",
00118 |       "scientific_note": "Strong MD baseline. Weighted 300-1000 K ladder concentrated toward 300-700 K.",
00119 |       "blocks": {
00120 |         "_list_length": 15,
00121 |         "_first_two": [
00122 |           {
00123 |             "block_id": "md_T300",
00124 |             "n_snapshots": 72,
00125 |             "temperature_K": 300,
00126 |             "seed": 1151
00127 |           },
00128 |           {
00129 |             "block_id": "md_T350",
00130 |             "n_snapshots": 68,
00131 |             "temperature_K": 350,
00132 |             "seed": 1152
00133 |           }
00134 |         ],
00135 |         "_last_two": [
00136 |           {
00137 |             "block_id": "md_T950",
00138 |             "n_snapshots": 32,
00139 |             "temperature_K": 950,
00140 |             "seed": 1164
00141 |           },
00142 |           {
00143 |             "block_id": "md_T1000",
00144 |             "n_snapshots": 32,
00145 |             "temperature_K": 1000,
00146 |             "seed": 1165
00147 |           }
00148 |         ]
00149 |       }
00150 |     },
00151 |     {
00152 |       "recipe_id": "md_h2o_weighted_temp_1140",
00153 |       "label": "H2O MD weighted temperature ladder 1140",
00154 |       "comparison_role": "very_strong_baseline",
00155 |       "scientific_note": "Very strong MD baseline for large-RC comparisons. Weighted 300-1000 K ladder concentrated toward 300-700 K.",
00156 |       "blocks": {
00157 |         "_list_length": 15,
00158 |         "_first_two": [
00159 |           {
00160 |             "block_id": "md_T300",
00161 |             "n_snapshots": 108,
00162 |             "temperature_K": 300,
00163 |             "seed": 1201
00164 |           },
00165 |           {
00166 |             "block_id": "md_T350",
00167 |             "n_snapshots": 102,
00168 |             "temperature_K": 350,
00169 |             "seed": 1202
00170 |           }
00171 |         ],
00172 |         "_last_two": [
00173 |           {
00174 |             "block_id": "md_T950",
00175 |             "n_snapshots": 48,
00176 |             "temperature_K": 950,
00177 |             "seed": 1214
00178 |           },
00179 |           {
00180 |             "block_id": "md_T1000",
00181 |             "n_snapshots": 48,
00182 |             "temperature_K": 1000,
00183 |             "seed": 1215
00184 |           }
00185 |         ]
00186 |       }
00187 |     }
00188 |   ],
00189 |   "siesta_fc_cartesian": [
00190 |     {
00191 |       "recipe_id": "fc_h2o_core_95",
00192 |       "label": "H2O FC compact core 95",
00193 |       "comparison_role": "low_budget_local_core",
00194 |       "scientific_note": "Compact FC core for low-budget local harmonic coverage.",
00195 |       "blocks": [
00196 |         {
00197 |           "block_id": "fc_A0040",
00198 |           "displacement": "0.004 Ang",
00199 |           "n_structures": 19
00200 |         },
00201 |         {
00202 |           "block_id": "fc_A0050",
00203 |           "displacement": "0.005 Ang",
00204 |           "n_structures": 19
00205 |         },
00206 |         {
00207 |           "block_id": "fc_A0060",
00208 |           "displacement": "0.006 Ang",
00209 |           "n_structures": 19
00210 |         },
00211 |         {
00212 |           "block_id": "fc_A0070",
00213 |           "displacement": "0.007 Ang",
00214 |           "n_structures": 19
00215 |         },
00216 |         {
00217 |           "block_id": "fc_A0080",
00218 |           "displacement": "0.008 Ang",
00219 |           "n_structures": 19
00220 |         }
00221 |       ]
00222 |     },
00223 |     {
00224 |       "recipe_id": "fc_h2o_core_190",
00225 |       "label": "H2O FC harmonic core 190",
00226 |       "comparison_role": "initial_efficiency",
00227 |       "scientific_note": "Core FC ladder.",
00228 |       "blocks": {
00229 |         "_list_length": 10,
00230 |         "_first_two": [
00231 |           {
00232 |             "block_id": "fc_A0040",
00233 |             "displacement": "0.004 Ang",
00234 |             "n_structures": 19
00235 |           },
00236 |           {
00237 |             "block_id": "fc_A0050",
00238 |             "displacement": "0.005 Ang",
00239 |             "n_structures": 19
00240 |           }
00241 |         ],
00242 |         "_last_two": [
00243 |           {
00244 |             "block_id": "fc_A0120",
00245 |             "displacement": "0.012 Ang",
00246 |             "n_structures": 19
00247 |           },
00248 |           {
00249 |             "block_id": "fc_A0130",
00250 |             "displacement": "0.013 Ang",
00251 |             "n_structures": 19
00252 |           }
00253 |         ]
00254 |       }
00255 |     },
00256 |     {
00257 |       "recipe_id": "fc_h2o_mid_380",
00258 |       "label": "H2O FC mid-density 380",
00259 |       "comparison_role": "medium_efficiency",
00260 |       "scientific_note": "Medium-density FC ladder.",
00261 |       "blocks": {
00262 |         "_list_length": 20,
00263 |         "_first_two": [
00264 |           {
00265 |             "block_id": "fc_A0040",
00266 |             "displacement": "0.004 Ang",
00267 |             "n_structures": 19
00268 |           },
00269 |           {
00270 |             "block_id": "fc_A0050",
00271 |             "displacement": "0.005 Ang",
00272 |             "n_structures": 19
00273 |           }
00274 |         ],
00275 |         "_last_two": [
00276 |           {
00277 |             "block_id": "fc_A0230",
00278 |             "displacement": "0.023 Ang",
00279 |             "n_structures": 19
00280 |           },
00281 |           {
00282 |             "block_id": "fc_A0260",
00283 |             "displacement": "0.026 Ang",
00284 |             "n_structures": 19
00285 |           }
00286 |         ]
00287 |       }
00288 |     },
00289 |     {
00290 |       "recipe_id": "fc_h2o_dense_570",
00291 |       "label": "H2O FC dense 570",
00292 |       "comparison_role": "primary_fc_reference",
00293 |       "scientific_note": "Dense FC ladder through 0.065 Ang and saturation tail.",
00294 |       "blocks": {
00295 |         "_list_length": 30,
00296 |         "_first_two": [
00297 |           {
00298 |             "block_id": "fc_A0040",
00299 |             "displacement": "0.004 Ang",
00300 |             "n_structures": 19
00301 |           },
00302 |           {
00303 |             "block_id": "fc_A0050",
00304 |             "displacement": "0.005 Ang",
00305 |             "n_structures": 19
00306 |           }
00307 |         ],
00308 |         "_last_two": [
00309 |           {
00310 |             "block_id": "fc_A0600",
00311 |             "displacement": "0.060 Ang",
00312 |             "n_structures": 19
00313 |           },
00314 |           {
00315 |             "block_id": "fc_A0650",
00316 |             "displacement": "0.065 Ang",
00317 |             "n_structures": 19
00318 |           }
00319 |         ]
00320 |       }
00321 |     },
00322 |     {
00323 |       "recipe_id": "fc_h2o_saturation_760",
00324 |       "label": "H2O FC saturation 760",
00325 |       "comparison_role": "saturation_reference",
00326 |       "scientific_note": "Saturation-only FC reference; tests redundancy rather than broad diversity.",
00327 |       "blocks": {
00328 |         "_list_length": 40,
00329 |         "_first_two": [
00330 |           {
00331 |             "block_id": "fc_A0040",
00332 |             "displacement": "0.004 Ang",
00333 |             "n_structures": 19
00334 |           },
00335 |           {
00336 |             "block_id": "fc_A0050",
00337 |             "displacement": "0.005 Ang",
00338 |             "n_structures": 19
00339 |           }
00340 |         ],
00341 |         "_last_two": [
00342 |           {
00343 |             "block_id": "fc_A1350",
00344 |             "displacement": "0.135 Ang",
00345 |             "n_structures": 19
00346 |           },
00347 |           {
00348 |             "block_id": "fc_A1500",
00349 |             "displacement": "0.150 Ang",
00350 |             "n_structures": 19
00351 |           }
00352 |         ]
00353 |       }
00354 |     }
00355 |   ],
00356 |   "random_cartesian": [
00357 |     {
00358 |       "recipe_id": "rc_h2o_multisigma_190",
00359 |       "label": "H2O Random Cartesian multisigma 190",
00360 |       "comparison_role": "initial_efficiency",
00361 |       "scientific_note": "Multi-sigma random Cartesian dataset. Blocks are one structure each with unique seeds so current spread splitting does not place the same RC family across train/validation/test.",
00362 |       "blocks": {
00363 |         "_list_length": 190,
00364 |         "_first_two": [
00365 |           {
00366 |             "block_id": "rc_sigma0060_0001",
00367 |             "n_structures": 1,
00368 |             "distribution": "gaussian",
00369 |             "sigma_ang": 0.006,
00370 |             "seed": 210001,
00371 |             "min_distance_ang": 0.65,
00372 |             "remove_center_of_mass_translation": true
00373 |           },
00374 |           {
00375 |             "block_id": "rc_sigma0060_0002",
00376 |             "n_structures": 1,
00377 |             "distribution": "gaussian",
00378 |             "sigma_ang": 0.006,
00379 |             "seed": 210002,
00380 |             "min_distance_ang": 0.65,
00381 |             "remove_center_of_mass_translation": true
00382 |           }
00383 |         ],
00384 |         "_last_two": [
00385 |           {
00386 |             "block_id": "rc_sigma0500_0024",
00387 |             "n_structures": 1,
00388 |             "distribution": "gaussian",
00389 |             "sigma_ang": 0.05,
00390 |             "seed": 210189,
00391 |             "min_distance_ang": 0.65,
00392 |             "remove_center_of_mass_translation": true
00393 |           },
00394 |           {
00395 |             "block_id": "rc_sigma0500_0025",
00396 |             "n_structures": 1,
00397 |             "distribution": "gaussian",
00398 |             "sigma_ang": 0.05,
00399 |             "seed": 210190,
00400 |             "min_distance_ang": 0.65,
00401 |             "remove_center_of_mass_translation": true
00402 |           }
00403 |         ]
00404 |       }
00405 |     },
00406 |     {
00407 |       "recipe_id": "rc_h2o_multisigma_380",
00408 |       "label": "H2O Random Cartesian multisigma 380",
00409 |       "comparison_role": "initial_efficiency",
00410 |       "scientific_note": "Multi-sigma random Cartesian dataset. Blocks are one structure each with unique seeds so current spread splitting does not place the same RC family across train/validation/test.",
00411 |       "blocks": {
00412 |         "_list_length": 380,
00413 |         "_first_two": [
00414 |           {
00415 |             "block_id": "rc_sigma0060_0001",
00416 |             "n_structures": 1,
00417 |             "distribution": "gaussian",
00418 |             "sigma_ang": 0.006,
00419 |             "seed": 220001,
00420 |             "min_distance_ang": 0.65,
00421 |             "remove_center_of_mass_translation": true
00422 |           },
00423 |           {
00424 |             "block_id": "rc_sigma0060_0002",
00425 |             "n_structures": 1,
00426 |             "distribution": "gaussian",
00427 |             "sigma_ang": 0.006,
00428 |             "seed": 220002,
00429 |             "min_distance_ang": 0.65,
00430 |             "remove_center_of_mass_translation": true
00431 |           }
00432 |         ],
00433 |         "_last_two": [
00434 |           {
00435 |             "block_id": "rc_sigma0500_0049",
00436 |             "n_structures": 1,
00437 |             "distribution": "gaussian",
00438 |             "sigma_ang": 0.05,
00439 |             "seed": 220379,
00440 |             "min_distance_ang": 0.65,
00441 |             "remove_center_of_mass_translation": true
00442 |           },
00443 |           {
00444 |             "block_id": "rc_sigma0500_0050",
00445 |             "n_structures": 1,
00446 |             "distribution": "gaussian",
00447 |             "sigma_ang": 0.05,
00448 |             "seed": 220380,
00449 |             "min_distance_ang": 0.65,
00450 |             "remove_center_of_mass_translation": true
00451 |           }
00452 |         ]
00453 |       }
00454 |     },
00455 |     {
00456 |       "recipe_id": "rc_h2o_multisigma_570",
00457 |       "label": "H2O Random Cartesian multisigma 570",
00458 |       "comparison_role": "primary_random_reference",
00459 |       "scientific_note": "Multi-sigma random Cartesian dataset. Blocks are one structure each with unique seeds so current spread splitting does not place the same RC family across train/validation/test.",
00460 |       "blocks": {
00461 |         "_list_length": 570,
00462 |         "_first_two": [
00463 |           {
00464 |             "block_id": "rc_sigma0060_0001",
00465 |             "n_structures": 1,
00466 |             "distribution": "gaussian",
00467 |             "sigma_ang": 0.006,
00468 |             "seed": 230001,
00469 |             "min_distance_ang": 0.65,
00470 |             "remove_center_of_mass_translation": true
00471 |           },
00472 |           {
00473 |             "block_id": "rc_sigma0060_0002",
00474 |             "n_structures": 1,
00475 |             "distribution": "gaussian",
00476 |             "sigma_ang": 0.006,
00477 |             "seed": 230002,
00478 |             "min_distance_ang": 0.65,
00479 |             "remove_center_of_mass_translation": true
00480 |           }
00481 |         ],
00482 |         "_last_two": [
00483 |           {
00484 |             "block_id": "rc_sigma0500_0074",
00485 |             "n_structures": 1,
00486 |             "distribution": "gaussian",
00487 |             "sigma_ang": 0.05,
00488 |             "seed": 230569,
00489 |             "min_distance_ang": 0.65,
00490 |             "remove_center_of_mass_translation": true
00491 |           },
00492 |           {
00493 |             "block_id": "rc_sigma0500_0075",
00494 |             "n_structures": 1,
00495 |             "distribution": "gaussian",
00496 |             "sigma_ang": 0.05,
00497 |             "seed": 230570,
00498 |             "min_distance_ang": 0.65,
00499 |             "remove_center_of_mass_translation": true
00500 |           }
00501 |         ]
00502 |       }
00503 |     },
00504 |     {
00505 |       "recipe_id": "rc_h2o_multisigma_760",
00506 |       "label": "H2O Random Cartesian multisigma 760",
00507 |       "comparison_role": "medium_random_scaling",
00508 |       "scientific_note": "Multi-sigma random Cartesian dataset. Blocks are one structure each with unique seeds so current spread splitting does not place the same RC family across train/validation/test.",
00509 |       "blocks": {
00510 |         "_list_length": 760,
00511 |         "_first_two": [
00512 |           {
00513 |             "block_id": "rc_sigma0060_0001",
00514 |             "n_structures": 1,
00515 |             "distribution": "gaussian",
00516 |             "sigma_ang": 0.006,
00517 |             "seed": 240001,
00518 |             "min_distance_ang": 0.65,
00519 |             "remove_center_of_mass_translation": true
00520 |           },
00521 |           {
00522 |             "block_id": "rc_sigma0060_0002",
00523 |             "n_structures": 1,
00524 |             "distribution": "gaussian",
00525 |             "sigma_ang": 0.006,
00526 |             "seed": 240002,
00527 |             "min_distance_ang": 0.65,
00528 |             "remove_center_of_mass_translation": true
00529 |           }
00530 |         ],
00531 |         "_last_two": [
00532 |           {
00533 |             "block_id": "rc_sigma0500_0099",
00534 |             "n_structures": 1,
00535 |             "distribution": "gaussian",
00536 |             "sigma_ang": 0.05,
00537 |             "seed": 240759,
00538 |             "min_distance_ang": 0.65,
00539 |             "remove_center_of_mass_translation": true
00540 |           },
00541 |           {
00542 |             "block_id": "rc_sigma0500_0100",
00543 |             "n_structures": 1,
00544 |             "distribution": "gaussian",
00545 |             "sigma_ang": 0.05,
00546 |             "seed": 240760,
00547 |             "min_distance_ang": 0.65,
00548 |             "remove_center_of_mass_translation": true
00549 |           }
00550 |         ]
00551 |       }
00552 |     },
00553 |     {
00554 |       "recipe_id": "rc_h2o_multisigma_1140",
00555 |       "label": "H2O Random Cartesian multisigma 1140",
00556 |       "comparison_role": "large_random_scaling",
00557 |       "scientific_note": "Multi-sigma random Cartesian dataset. Blocks are one structure each with unique seeds so current spread splitting does not place the same RC family across train/validation/test.",
00558 |       "blocks": {
00559 |         "_list_length": 1140,
00560 |         "_first_two": [
00561 |           {
00562 |             "block_id": "rc_sigma0060_0001",
00563 |             "n_structures": 1,
00564 |             "distribution": "gaussian",
00565 |             "sigma_ang": 0.006,
00566 |             "seed": 250001,
00567 |             "min_distance_ang": 0.65,
00568 |             "remove_center_of_mass_translation": true
00569 |           },
00570 |           {
00571 |             "block_id": "rc_sigma0060_0002",
00572 |             "n_structures": 1,
00573 |             "distribution": "gaussian",
00574 |             "sigma_ang": 0.006,
00575 |             "seed": 250002,
00576 |             "min_distance_ang": 0.65,
00577 |             "remove_center_of_mass_translation": true
00578 |           }
00579 |         ],
00580 |         "_last_two": [
00581 |           {
00582 |             "block_id": "rc_sigma0500_0149",
00583 |             "n_structures": 1,
00584 |             "distribution": "gaussian",
00585 |             "sigma_ang": 0.05,
00586 |             "seed": 251139,
00587 |             "min_distance_ang": 0.65,
00588 |             "remove_center_of_mass_translation": true
00589 |           },
00590 |           {
00591 |             "block_id": "rc_sigma0500_0150",
00592 |             "n_structures": 1,
00593 |             "distribution": "gaussian",
00594 |             "sigma_ang": 0.05,
00595 |             "seed": 251140,
00596 |             "min_distance_ang": 0.65,
00597 |             "remove_center_of_mass_translation": true
00598 |           }
00599 |         ]
00600 |       }
00601 |     },
00602 |     {
00603 |       "recipe_id": "rc_h2o_multisigma_1520",
00604 |       "label": "H2O Random Cartesian multisigma 1520",
00605 |       "comparison_role": "large_random_scaling",
00606 |       "scientific_note": "Multi-sigma random Cartesian dataset. Blocks are one structure each with unique seeds so current spread splitting does not place the same RC family across train/validation/test.",
00607 |       "blocks": {
00608 |         "_list_length": 1520,
00609 |         "_first_two": [
00610 |           {
00611 |             "block_id": "rc_sigma0060_0001",
00612 |             "n_structures": 1,
00613 |             "distribution": "gaussian",
00614 |             "sigma_ang": 0.006,
00615 |             "seed": 260001,
00616 |             "min_distance_ang": 0.65,
00617 |             "remove_center_of_mass_translation": true
00618 |           },
00619 |           {
00620 |             "block_id": "rc_sigma0060_0002",
00621 |             "n_structures": 1,
00622 |             "distribution": "gaussian",
00623 |             "sigma_ang": 0.006,
00624 |             "seed": 260002,
00625 |             "min_distance_ang": 0.65,
00626 |             "remove_center_of_mass_translation": true
00627 |           }
00628 |         ],
00629 |         "_last_two": [
00630 |           {
00631 |             "block_id": "rc_sigma0500_0199",
00632 |             "n_structures": 1,
00633 |             "distribution": "gaussian",
00634 |             "sigma_ang": 0.05,
00635 |             "seed": 261519,
00636 |             "min_distance_ang": 0.65,
00637 |             "remove_center_of_mass_translation": true
00638 |           },
00639 |           {
00640 |             "block_id": "rc_sigma0500_0200",
00641 |             "n_structures": 1,
00642 |             "distribution": "gaussian",
00643 |             "sigma_ang": 0.05,
00644 |             "seed": 261520,
00645 |             "min_distance_ang": 0.65,
00646 |             "remove_center_of_mass_translation": true
00647 |           }
00648 |         ]
00649 |       }
00650 |     },
00651 |     {
00652 |       "recipe_id": "rc_h2o_multisigma_2280",
00653 |       "label": "H2O Random Cartesian multisigma 2280",
00654 |       "comparison_role": "massive_random_scaling",
00655 |       "scientific_note": "Multi-sigma random Cartesian dataset. Blocks are one structure each with unique seeds so current spread splitting does not place the same RC family across train/validation/test.",
00656 |       "blocks": {
00657 |         "_list_length": 2280,
00658 |         "_first_two": [
00659 |           {
00660 |             "block_id": "rc_sigma0060_0001",
00661 |             "n_structures": 1,
00662 |             "distribution": "gaussian",
00663 |             "sigma_ang": 0.006,
00664 |             "seed": 270001,
00665 |             "min_distance_ang": 0.65,
00666 |             "remove_center_of_mass_translation": true
00667 |           },
00668 |           {
00669 |             "block_id": "rc_sigma0060_0002",
00670 |             "n_structures": 1,
00671 |             "distribution": "gaussian",
00672 |             "sigma_ang": 0.006,
00673 |             "seed": 270002,
00674 |             "min_distance_ang": 0.65,
00675 |             "remove_center_of_mass_translation": true
00676 |           }
00677 |         ],
00678 |         "_last_two": [
00679 |           {
00680 |             "block_id": "rc_sigma0500_0299",
00681 |             "n_structures": 1,
00682 |             "distribution": "gaussian",
00683 |             "sigma_ang": 0.05,
00684 |             "seed": 272279,
00685 |             "min_distance_ang": 0.65,
00686 |             "remove_center_of_mass_translation": true
00687 |           },
00688 |           {
00689 |             "block_id": "rc_sigma0500_0300",
00690 |             "n_structures": 1,
00691 |             "distribution": "gaussian",
00692 |             "sigma_ang": 0.05,
00693 |             "seed": 272280,
00694 |             "min_distance_ang": 0.65,
00695 |             "remove_center_of_mass_translation": true
00696 |           }
00697 |         ]
00698 |       }
00699 |     },
00700 |     {
00701 |       "recipe_id": "rc_h2o_multisigma_3500",
00702 |       "label": "H2O Random Cartesian multisigma 3500",
00703 |       "comparison_role": "massive_random_scaling",
00704 |       "scientific_note": "Multi-sigma random Cartesian dataset. Blocks are one structure each with unique seeds so current spread splitting does not place the same RC family across train/validation/test.",
00705 |       "blocks": {
00706 |         "_list_length": 3500,
00707 |         "_first_two": [
00708 |           {
00709 |             "block_id": "rc_sigma0060_0001",
00710 |             "n_structures": 1,
00711 |             "distribution": "gaussian",
00712 |             "sigma_ang": 0.006,
00713 |             "seed": 280001,
00714 |             "min_distance_ang": 0.65,
00715 |             "remove_center_of_mass_translation": true
00716 |           },
00717 |           {
00718 |             "block_id": "rc_sigma0060_0002",
00719 |             "n_structures": 1,
00720 |             "distribution": "gaussian",
00721 |             "sigma_ang": 0.006,
00722 |             "seed": 280002,
00723 |             "min_distance_ang": 0.65,
00724 |             "remove_center_of_mass_translation": true
00725 |           }
00726 |         ],
00727 |         "_last_two": [
00728 |           {
00729 |             "block_id": "rc_sigma0500_0499",
00730 |             "n_structures": 1,
00731 |             "distribution": "gaussian",
00732 |             "sigma_ang": 0.05,
00733 |             "seed": 283499,
00734 |             "min_distance_ang": 0.65,
00735 |             "remove_center_of_mass_translation": true
00736 |           },
00737 |           {
00738 |             "block_id": "rc_sigma0500_0500",
00739 |             "n_structures": 1,
00740 |             "distribution": "gaussian",
00741 |             "sigma_ang": 0.05,
00742 |             "seed": 283500,
00743 |             "min_distance_ang": 0.65,
00744 |             "remove_center_of_mass_translation": true
00745 |           }
00746 |         ]
00747 |       }
00748 |     }
00749 |   ]
00750 | }
```

## `Comparison/dataset_recipes/h2o_recommended_190_570_1140.json` — vista compacta

SHA-256 del JSON completo: `08f3eeb7da5f4828b098c3f96b6aa71d896f156540092675b208dd6084a756e2`. Las listas de más de ocho elementos conservan longitud, dos primeros y dos últimos elementos; esta vista no sustituye al artefacto para verificaciones cuantitativas.

```json
00001 | {
00002 |   "md": [
00003 |     {
00004 |       "recipe_id": "md_h2o_weighted_temp_190",
00005 |       "label": "H2O MD weighted temperature ladder 190",
00006 |       "comparison_role": "primary_equal_N",
00007 |       "scientific_note": "Weighted 300-1000 K ladder concentrated toward 300-700 K.",
00008 |       "blocks": {
00009 |         "_list_length": 15,
00010 |         "_first_two": [
00011 |           {
00012 |             "block_id": "md_T300",
00013 |             "n_snapshots": 18,
00014 |             "temperature_K": 300,
00015 |             "seed": 1001
00016 |           },
00017 |           {
00018 |             "block_id": "md_T350",
00019 |             "n_snapshots": 17,
00020 |             "temperature_K": 350,
00021 |             "seed": 1002
00022 |           }
00023 |         ],
00024 |         "_last_two": [
00025 |           {
00026 |             "block_id": "md_T950",
00027 |             "n_snapshots": 8,
00028 |             "temperature_K": 950,
00029 |             "seed": 1014
00030 |           },
00031 |           {
00032 |             "block_id": "md_T1000",
00033 |             "n_snapshots": 8,
00034 |             "temperature_K": 1000,
00035 |             "seed": 1015
00036 |           }
00037 |         ]
00038 |       }
00039 |     },
00040 |     {
00041 |       "recipe_id": "md_h2o_weighted_temp_570",
00042 |       "label": "H2O MD weighted temperature ladder 570",
00043 |       "comparison_role": "medium_exploratory",
00044 |       "scientific_note": "Second-stage MD dataset for MD vs random_cartesian scaling.",
00045 |       "blocks": {
00046 |         "_list_length": 15,
00047 |         "_first_two": [
00048 |           {
00049 |             "block_id": "md_T300",
00050 |             "n_snapshots": 54,
00051 |             "temperature_K": 300,
00052 |             "seed": 1101
00053 |           },
00054 |           {
00055 |             "block_id": "md_T350",
00056 |             "n_snapshots": 51,
00057 |             "temperature_K": 350,
00058 |             "seed": 1102
00059 |           }
00060 |         ],
00061 |         "_last_two": [
00062 |           {
00063 |             "block_id": "md_T950",
00064 |             "n_snapshots": 24,
00065 |             "temperature_K": 950,
00066 |             "seed": 1114
00067 |           },
00068 |           {
00069 |             "block_id": "md_T1000",
00070 |             "n_snapshots": 24,
00071 |             "temperature_K": 1000,
00072 |             "seed": 1115
00073 |           }
00074 |         ]
00075 |       }
00076 |     },
00077 |     {
00078 |       "recipe_id": "md_h2o_weighted_temp_1140",
00079 |       "label": "H2O MD weighted temperature ladder 1140",
00080 |       "comparison_role": "large_learning_curve",
00081 |       "scientific_note": "Large MD dataset for checking whether MD continues improving after 570 structures.",
00082 |       "blocks": {
00083 |         "_list_length": 15,
00084 |         "_first_two": [
00085 |           {
00086 |             "block_id": "md_T300",
00087 |             "n_snapshots": 108,
00088 |             "temperature_K": 300,
00089 |             "seed": 1201
00090 |           },
00091 |           {
00092 |             "block_id": "md_T350",
00093 |             "n_snapshots": 102,
00094 |             "temperature_K": 350,
00095 |             "seed": 1202
00096 |           }
00097 |         ],
00098 |         "_last_two": [
00099 |           {
00100 |             "block_id": "md_T950",
00101 |             "n_snapshots": 48,
00102 |             "temperature_K": 950,
00103 |             "seed": 1214
00104 |           },
00105 |           {
00106 |             "block_id": "md_T1000",
00107 |             "n_snapshots": 48,
00108 |             "temperature_K": 1000,
00109 |             "seed": 1215
00110 |           }
00111 |         ]
00112 |       }
00113 |     }
00114 |   ],
00115 |   "siesta_fc_cartesian": [
00116 |     {
00117 |       "recipe_id": "fc_h2o_harmonic_core_190",
00118 |       "label": "H2O FC harmonic core 190",
00119 |       "comparison_role": "primary_equal_N",
00120 |       "scientific_note": "Dense small-amplitude frozen-phonon core: 10 amplitudes x 19 structures.",
00121 |       "blocks": {
00122 |         "_list_length": 10,
00123 |         "_first_two": [
00124 |           {
00125 |             "block_id": "fc_A0005",
00126 |             "displacement": "0.005 Ang",
00127 |             "n_structures": 19
00128 |           },
00129 |           {
00130 |             "block_id": "fc_A0010",
00131 |             "displacement": "0.010 Ang",
00132 |             "n_structures": 19
00133 |           }
00134 |         ],
00135 |         "_last_two": [
00136 |           {
00137 |             "block_id": "fc_A0045",
00138 |             "displacement": "0.045 Ang",
00139 |             "n_structures": 19
00140 |           },
00141 |           {
00142 |             "block_id": "fc_A0050",
00143 |             "displacement": "0.050 Ang",
00144 |             "n_structures": 19
00145 |           }
00146 |         ]
00147 |       }
00148 |     },
00149 |     {
00150 |       "recipe_id": "fc_h2o_saturation_570",
00151 |       "label": "H2O FC saturation 570",
00152 |       "comparison_role": "optional_saturation_reference",
00153 |       "scientific_note": "Optional saturation reference only; not recommended as the automatic second main benchmark.",
00154 |       "blocks": {
00155 |         "_list_length": 30,
00156 |         "_first_two": [
00157 |           {
00158 |             "block_id": "fc_A0005",
00159 |             "displacement": "0.005 Ang",
00160 |             "n_structures": 19
00161 |           },
00162 |           {
00163 |             "block_id": "fc_A0010",
00164 |             "displacement": "0.010 Ang",
00165 |             "n_structures": 19
00166 |           }
00167 |         ],
00168 |         "_last_two": [
00169 |           {
00170 |             "block_id": "fc_A0145",
00171 |             "displacement": "0.145 Ang",
00172 |             "n_structures": 19
00173 |           },
00174 |           {
00175 |             "block_id": "fc_A0150",
00176 |             "displacement": "0.150 Ang",
00177 |             "n_structures": 19
00178 |           }
00179 |         ]
00180 |       }
00181 |     }
00182 |   ],
00183 |   "random_cartesian": [
00184 |     {
00185 |       "recipe_id": "rc_h2o_multisigma_190",
00186 |       "label": "H2O Random Cartesian multisigma 190",
00187 |       "comparison_role": "primary_equal_N",
00188 |       "scientific_note": "Multi-sigma local Cartesian perturbations concentrated around 0.010-0.020 Ang.",
00189 |       "blocks": [
00190 |         {
00191 |           "block_id": "rc_sigma0005",
00192 |           "n_structures": 25,
00193 |           "distribution": "gaussian",
00194 |           "sigma_ang": 0.005,
00195 |           "seed": 2001,
00196 |           "min_distance_ang": 0.65,
00197 |           "remove_center_of_mass_translation": true
00198 |         },
00199 |         {
00200 |           "block_id": "rc_sigma0010",
00201 |           "n_structures": 45,
00202 |           "distribution": "gaussian",
00203 |           "sigma_ang": 0.01,
00204 |           "seed": 2002,
00205 |           "min_distance_ang": 0.65,
00206 |           "remove_center_of_mass_translation": true
00207 |         },
00208 |         {
00209 |           "block_id": "rc_sigma0020",
00210 |           "n_structures": 55,
00211 |           "distribution": "gaussian",
00212 |           "sigma_ang": 0.02,
00213 |           "seed": 2003,
00214 |           "min_distance_ang": 0.65,
00215 |           "remove_center_of_mass_translation": true
00216 |         },
00217 |         {
00218 |           "block_id": "rc_sigma0035",
00219 |           "n_structures": 40,
00220 |           "distribution": "gaussian",
00221 |           "sigma_ang": 0.035,
00222 |           "seed": 2004,
00223 |           "min_distance_ang": 0.65,
00224 |           "remove_center_of_mass_translation": true
00225 |         },
00226 |         {
00227 |           "block_id": "rc_sigma0050",
00228 |           "n_structures": 25,
00229 |           "distribution": "gaussian",
00230 |           "sigma_ang": 0.05,
00231 |           "seed": 2005,
00232 |           "min_distance_ang": 0.65,
00233 |           "remove_center_of_mass_translation": true
00234 |         }
00235 |       ]
00236 |     },
00237 |     {
00238 |       "recipe_id": "rc_h2o_multisigma_570",
00239 |       "label": "H2O Random Cartesian multisigma 570",
00240 |       "comparison_role": "medium_exploratory",
00241 |       "scientific_note": "Second-stage random_cartesian dataset for MD vs random_cartesian scaling.",
00242 |       "blocks": [
00243 |         {
00244 |           "block_id": "rc_sigma0005",
00245 |           "n_structures": 75,
00246 |           "distribution": "gaussian",
00247 |           "sigma_ang": 0.005,
00248 |           "seed": 2101,
00249 |           "min_distance_ang": 0.65,
00250 |           "remove_center_of_mass_translation": true
00251 |         },
00252 |         {
00253 |           "block_id": "rc_sigma0010",
00254 |           "n_structures": 135,
00255 |           "distribution": "gaussian",
00256 |           "sigma_ang": 0.01,
00257 |           "seed": 2102,
00258 |           "min_distance_ang": 0.65,
00259 |           "remove_center_of_mass_translation": true
00260 |         },
00261 |         {
00262 |           "block_id": "rc_sigma0020",
00263 |           "n_structures": 165,
00264 |           "distribution": "gaussian",
00265 |           "sigma_ang": 0.02,
00266 |           "seed": 2103,
00267 |           "min_distance_ang": 0.65,
00268 |           "remove_center_of_mass_translation": true
00269 |         },
00270 |         {
00271 |           "block_id": "rc_sigma0035",
00272 |           "n_structures": 120,
00273 |           "distribution": "gaussian",
00274 |           "sigma_ang": 0.035,
00275 |           "seed": 2104,
00276 |           "min_distance_ang": 0.65,
00277 |           "remove_center_of_mass_translation": true
00278 |         },
00279 |         {
00280 |           "block_id": "rc_sigma0050",
00281 |           "n_structures": 75,
00282 |           "distribution": "gaussian",
00283 |           "sigma_ang": 0.05,
00284 |           "seed": 2105,
00285 |           "min_distance_ang": 0.65,
00286 |           "remove_center_of_mass_translation": true
00287 |         }
00288 |       ]
00289 |     },
00290 |     {
00291 |       "recipe_id": "rc_h2o_multisigma_1140",
00292 |       "label": "H2O Random Cartesian multisigma 1140",
00293 |       "comparison_role": "large_learning_curve",
00294 |       "scientific_note": "Large random_cartesian dataset for checking whether additional samples keep improving metrics.",
00295 |       "blocks": [
00296 |         {
00297 |           "block_id": "rc_sigma0005",
00298 |           "n_structures": 150,
00299 |           "distribution": "gaussian",
00300 |           "sigma_ang": 0.005,
00301 |           "seed": 2201,
00302 |           "min_distance_ang": 0.65,
00303 |           "remove_center_of_mass_translation": true
00304 |         },
00305 |         {
00306 |           "block_id": "rc_sigma0010",
00307 |           "n_structures": 270,
00308 |           "distribution": "gaussian",
00309 |           "sigma_ang": 0.01,
00310 |           "seed": 2202,
00311 |           "min_distance_ang": 0.65,
00312 |           "remove_center_of_mass_translation": true
00313 |         },
00314 |         {
00315 |           "block_id": "rc_sigma0020",
00316 |           "n_structures": 330,
00317 |           "distribution": "gaussian",
00318 |           "sigma_ang": 0.02,
00319 |           "seed": 2203,
00320 |           "min_distance_ang": 0.65,
00321 |           "remove_center_of_mass_translation": true
00322 |         },
00323 |         {
00324 |           "block_id": "rc_sigma0035",
00325 |           "n_structures": 240,
00326 |           "distribution": "gaussian",
00327 |           "sigma_ang": 0.035,
00328 |           "seed": 2204,
00329 |           "min_distance_ang": 0.65,
00330 |           "remove_center_of_mass_translation": true
00331 |         },
00332 |         {
00333 |           "block_id": "rc_sigma0050",
00334 |           "n_structures": 150,
00335 |           "distribution": "gaussian",
00336 |           "sigma_ang": 0.05,
00337 |           "seed": 2205,
00338 |           "min_distance_ang": 0.65,
00339 |           "remove_center_of_mass_translation": true
00340 |         }
00341 |       ]
00342 |     },
00343 |     {
00344 |       "recipe_id": "rc_h2o_multisigma_1800",
00345 |       "label": "H2O Random Cartesian multisigma 1800",
00346 |       "comparison_role": "extended_random_cartesian_scaling",
00347 |       "scientific_note": "Extra random_cartesian scaling point to test whether multisigma sampling can overtake MD at larger N.",
00348 |       "blocks": [
00349 |         {
00350 |           "block_id": "rc_sigma0005",
00351 |           "n_structures": 237,
00352 |           "distribution": "gaussian",
00353 |           "sigma_ang": 0.005,
00354 |           "seed": 2301,
00355 |           "min_distance_ang": 0.65,
00356 |           "remove_center_of_mass_translation": true
00357 |         },
00358 |         {
00359 |           "block_id": "rc_sigma0010",
00360 |           "n_structures": 426,
00361 |           "distribution": "gaussian",
00362 |           "sigma_ang": 0.01,
00363 |           "seed": 2302,
00364 |           "min_distance_ang": 0.65,
00365 |           "remove_center_of_mass_translation": true
00366 |         },
00367 |         {
00368 |           "block_id": "rc_sigma0020",
00369 |           "n_structures": 521,
00370 |           "distribution": "gaussian",
00371 |           "sigma_ang": 0.02,
00372 |           "seed": 2303,
00373 |           "min_distance_ang": 0.65,
00374 |           "remove_center_of_mass_translation": true
00375 |         },
00376 |         {
00377 |           "block_id": "rc_sigma0035",
00378 |           "n_structures": 379,
00379 |           "distribution": "gaussian",
00380 |           "sigma_ang": 0.035,
00381 |           "seed": 2304,
00382 |           "min_distance_ang": 0.65,
00383 |           "remove_center_of_mass_translation": true
00384 |         },
00385 |         {
00386 |           "block_id": "rc_sigma0050",
00387 |           "n_structures": 237,
00388 |           "distribution": "gaussian",
00389 |           "sigma_ang": 0.05,
00390 |           "seed": 2305,
00391 |           "min_distance_ang": 0.65,
00392 |           "remove_center_of_mass_translation": true
00393 |         }
00394 |       ]
00395 |     },
00396 |     {
00397 |       "recipe_id": "rc_h2o_multisigma_2500",
00398 |       "label": "H2O Random Cartesian multisigma 2500",
00399 |       "comparison_role": "extended_random_cartesian_scaling",
00400 |       "scientific_note": "Extra random_cartesian scaling point to test whether multisigma sampling can overtake MD at larger N.",
00401 |       "blocks": [
00402 |         {
00403 |           "block_id": "rc_sigma0005",
00404 |           "n_structures": 329,
00405 |           "distribution": "gaussian",
00406 |           "sigma_ang": 0.005,
00407 |           "seed": 2401,
00408 |           "min_distance_ang": 0.65,
00409 |           "remove_center_of_mass_translation": true
00410 |         },
00411 |         {
00412 |           "block_id": "rc_sigma0010",
00413 |           "n_structures": 592,
00414 |           "distribution": "gaussian",
00415 |           "sigma_ang": 0.01,
00416 |           "seed": 2402,
00417 |           "min_distance_ang": 0.65,
00418 |           "remove_center_of_mass_translation": true
00419 |         },
00420 |         {
00421 |           "block_id": "rc_sigma0020",
00422 |           "n_structures": 724,
00423 |           "distribution": "gaussian",
00424 |           "sigma_ang": 0.02,
00425 |           "seed": 2403,
00426 |           "min_distance_ang": 0.65,
00427 |           "remove_center_of_mass_translation": true
00428 |         },
00429 |         {
00430 |           "block_id": "rc_sigma0035",
00431 |           "n_structures": 526,
00432 |           "distribution": "gaussian",
00433 |           "sigma_ang": 0.035,
00434 |           "seed": 2404,
00435 |           "min_distance_ang": 0.65,
00436 |           "remove_center_of_mass_translation": true
00437 |         },
00438 |         {
00439 |           "block_id": "rc_sigma0050",
00440 |           "n_structures": 329,
00441 |           "distribution": "gaussian",
00442 |           "sigma_ang": 0.05,
00443 |           "seed": 2405,
00444 |           "min_distance_ang": 0.65,
00445 |           "remove_center_of_mass_translation": true
00446 |         }
00447 |       ]
00448 |     },
00449 |     {
00450 |       "recipe_id": "rc_h2o_multisigma_3500",
00451 |       "label": "H2O Random Cartesian multisigma 3500",
00452 |       "comparison_role": "extended_random_cartesian_scaling",
00453 |       "scientific_note": "Extra random_cartesian scaling point to test whether multisigma sampling can overtake MD at larger N.",
00454 |       "blocks": [
00455 |         {
00456 |           "block_id": "rc_sigma0005",
00457 |           "n_structures": 460,
00458 |           "distribution": "gaussian",
00459 |           "sigma_ang": 0.005,
00460 |           "seed": 2501,
00461 |           "min_distance_ang": 0.65,
00462 |           "remove_center_of_mass_translation": true
00463 |         },
00464 |         {
00465 |           "block_id": "rc_sigma0010",
00466 |           "n_structures": 829,
00467 |           "distribution": "gaussian",
00468 |           "sigma_ang": 0.01,
00469 |           "seed": 2502,
00470 |           "min_distance_ang": 0.65,
00471 |           "remove_center_of_mass_translation": true
00472 |         },
00473 |         {
00474 |           "block_id": "rc_sigma0020",
00475 |           "n_structures": 1013,
00476 |           "distribution": "gaussian",
00477 |           "sigma_ang": 0.02,
00478 |           "seed": 2503,
00479 |           "min_distance_ang": 0.65,
00480 |           "remove_center_of_mass_translation": true
00481 |         },
00482 |         {
00483 |           "block_id": "rc_sigma0035",
00484 |           "n_structures": 738,
00485 |           "distribution": "gaussian",
00486 |           "sigma_ang": 0.035,
00487 |           "seed": 2504,
00488 |           "min_distance_ang": 0.65,
00489 |           "remove_center_of_mass_translation": true
00490 |         },
00491 |         {
00492 |           "block_id": "rc_sigma0050",
00493 |           "n_structures": 460,
00494 |           "distribution": "gaussian",
00495 |           "sigma_ang": 0.05,
00496 |           "seed": 2505,
00497 |           "min_distance_ang": 0.65,
00498 |           "remove_center_of_mass_translation": true
00499 |         }
00500 |       ]
00501 |     }
00502 |   ]
00503 | }
```

## `Comparison/dataset_recipes/h2o_scientific_285_750_1500_improved.json` — vista compacta

SHA-256 del JSON completo: `e65b93bdc2ed3d2535756f53b732f3295e449b423084ae637d311af81008f9af`. Las listas de más de ocho elementos conservan longitud, dos primeros y dos últimos elementos; esta vista no sustituye al artefacto para verificaciones cuantitativas.

```json
00001 | {
00002 |   "md": {
00003 |     "_list_length": 9,
00004 |     "_first_two": [
00005 |       {
00006 |         "recipe_id": "md_h2o_weighted_ladder_285_seed1",
00007 |         "label": "H2O MD weighted temperature ladder 285 seed 1",
00008 |         "seed": 1001,
00009 |         "comparison_role": "primary_equal_N_285",
00010 |         "scientific_note": "Weighted 300-1000 K ladder: broad thermal coverage with less dominance of the 950-1000 K tail.",
00011 |         "blocks": {
00012 |           "_list_length": 15,
00013 |           "_first_two": [
00014 |             {
00015 |               "block_id": "md_h2o_weighted_ladder_285_seed1_T300_n28",
00016 |               "label": "MD 300 K, 28 snapshots",
00017 |               "temperature_K": 300,
00018 |               "n_snapshots": 28
00019 |             },
00020 |             {
00021 |               "block_id": "md_h2o_weighted_ladder_285_seed1_T350_n26",
00022 |               "label": "MD 350 K, 26 snapshots",
00023 |               "temperature_K": 350,
00024 |               "n_snapshots": 26
00025 |             }
00026 |           ],
00027 |           "_last_two": [
00028 |             {
00029 |               "block_id": "md_h2o_weighted_ladder_285_seed1_T950_n13",
00030 |               "label": "MD 950 K, 13 snapshots",
00031 |               "temperature_K": 950,
00032 |               "n_snapshots": 13
00033 |             },
00034 |             {
00035 |               "block_id": "md_h2o_weighted_ladder_285_seed1_T1000_n13",
00036 |               "label": "MD 1000 K, 13 snapshots",
00037 |               "temperature_K": 1000,
00038 |               "n_snapshots": 13
00039 |             }
00040 |           ]
00041 |         }
00042 |       },
00043 |       {
00044 |         "recipe_id": "md_h2o_weighted_ladder_285_seed2",
00045 |         "label": "H2O MD weighted temperature ladder 285 seed 2",
00046 |         "seed": 1002,
00047 |         "comparison_role": "primary_equal_N_285",
00048 |         "scientific_note": "Weighted 300-1000 K ladder: broad thermal coverage with less dominance of the 950-1000 K tail.",
00049 |         "blocks": {
00050 |           "_list_length": 15,
00051 |           "_first_two": [
00052 |             {
00053 |               "block_id": "md_h2o_weighted_ladder_285_seed2_T300_n26",
00054 |               "label": "MD 300 K, 26 snapshots",
00055 |               "temperature_K": 300,
00056 |               "n_snapshots": 26
00057 |             },
00058 |             {
00059 |               "block_id": "md_h2o_weighted_ladder_285_seed2_T350_n25",
00060 |               "label": "MD 350 K, 25 snapshots",
00061 |               "temperature_K": 350,
00062 |               "n_snapshots": 25
00063 |             }
00064 |           ],
00065 |           "_last_two": [
00066 |             {
00067 |               "block_id": "md_h2o_weighted_ladder_285_seed2_T950_n13",
00068 |               "label": "MD 950 K, 13 snapshots",
00069 |               "temperature_K": 950,
00070 |               "n_snapshots": 13
00071 |             },
00072 |             {
00073 |               "block_id": "md_h2o_weighted_ladder_285_seed2_T1000_n12",
00074 |               "label": "MD 1000 K, 12 snapshots",
00075 |               "temperature_K": 1000,
00076 |               "n_snapshots": 12
00077 |             }
00078 |           ]
00079 |         }
00080 |       }
00081 |     ],
00082 |     "_last_two": [
00083 |       {
00084 |         "recipe_id": "md_h2o_weighted_ladder_1500_seed2",
00085 |         "label": "H2O MD weighted temperature ladder 1500 seed 2",
00086 |         "seed": 1202,
00087 |         "comparison_role": "large_scale_MD_vs_RC",
00088 |         "scientific_note": "Large learning-curve point for MD vs RC with three seeds; run only after checking whether 750 is still improving.",
00089 |         "blocks": {
00090 |           "_list_length": 15,
00091 |           "_first_two": [
00092 |             {
00093 |               "block_id": "md_h2o_weighted_ladder_1500_seed2_T300_n140",
00094 |               "label": "MD 300 K, 140 snapshots",
00095 |               "temperature_K": 300,
00096 |               "n_snapshots": 140
00097 |             },
00098 |             {
00099 |               "block_id": "md_h2o_weighted_ladder_1500_seed2_T350_n136",
00100 |               "label": "MD 350 K, 136 snapshots",
00101 |               "temperature_K": 350,
00102 |               "n_snapshots": 136
00103 |             }
00104 |           ],
00105 |           "_last_two": [
00106 |             {
00107 |               "block_id": "md_h2o_weighted_ladder_1500_seed2_T950_n60",
00108 |               "label": "MD 950 K, 60 snapshots",
00109 |               "temperature_K": 950,
00110 |               "n_snapshots": 60
00111 |             },
00112 |             {
00113 |               "block_id": "md_h2o_weighted_ladder_1500_seed2_T1000_n54",
00114 |               "label": "MD 1000 K, 54 snapshots",
00115 |               "temperature_K": 1000,
00116 |               "n_snapshots": 54
00117 |             }
00118 |           ]
00119 |         }
00120 |       },
00121 |       {
00122 |         "recipe_id": "md_h2o_weighted_ladder_1500_seed3",
00123 |         "label": "H2O MD weighted temperature ladder 1500 seed 3",
00124 |         "seed": 1203,
00125 |         "comparison_role": "large_scale_MD_vs_RC",
00126 |         "scientific_note": "Large learning-curve point for MD vs RC with three seeds; run only after checking whether 750 is still improving.",
00127 |         "blocks": {
00128 |           "_list_length": 15,
00129 |           "_first_two": [
00130 |             {
00131 |               "block_id": "md_h2o_weighted_ladder_1500_seed3_T300_n136",
00132 |               "label": "MD 300 K, 136 snapshots",
00133 |               "temperature_K": 300,
00134 |               "n_snapshots": 136
00135 |             },
00136 |             {
00137 |               "block_id": "md_h2o_weighted_ladder_1500_seed3_T350_n132",
00138 |               "label": "MD 350 K, 132 snapshots",
00139 |               "temperature_K": 350,
00140 |               "n_snapshots": 132
00141 |             }
00142 |           ],
00143 |           "_last_two": [
00144 |             {
00145 |               "block_id": "md_h2o_weighted_ladder_1500_seed3_T950_n50",
00146 |               "label": "MD 950 K, 50 snapshots",
00147 |               "temperature_K": 950,
00148 |               "n_snapshots": 50
00149 |             },
00150 |             {
00151 |               "block_id": "md_h2o_weighted_ladder_1500_seed3_T1000_n50",
00152 |               "label": "MD 1000 K, 50 snapshots",
00153 |               "temperature_K": 1000,
00154 |               "n_snapshots": 50
00155 |             }
00156 |           ]
00157 |         }
00158 |       }
00159 |     ]
00160 |   },
00161 |   "siesta_fc_cartesian": {
00162 |     "_list_length": 9,
00163 |     "_first_two": [
00164 |       {
00165 |         "recipe_id": "fc_h2o_low_amp_subsampled_125_seed1",
00166 |         "label": "H2O FC low-amplitude subsampled ladder 125 seed 1",
00167 |         "seed": 2126,
00168 |         "comparison_role": "budget_curve_FC_125",
00169 |         "scientific_note": "Budget-curve FC point with 25 low-amplitude displacements x 5 selected FC steps. Counts stay below the high-count regime that can select the zero-reference geometry.",
00170 |         "blocks": {
00171 |           "_list_length": 25,
00172 |           "_first_two": [
00173 |             {
00174 |               "block_id": "fc_125_s1_A0p005_n5",
00175 |               "label": "FC 0.005 Ang, 5 selected structures",
00176 |               "displacement": "0.005 Ang",
00177 |               "n_structures": 5
00178 |             },
00179 |             {
00180 |               "block_id": "fc_125_s1_A0p008_n5",
00181 |               "label": "FC 0.008 Ang, 5 selected structures",
00182 |               "displacement": "0.008 Ang",
00183 |               "n_structures": 5
00184 |             }
00185 |           ],
00186 |           "_last_two": [
00187 |             {
00188 |               "block_id": "fc_125_s1_A0p075_n5",
00189 |               "label": "FC 0.075 Ang, 5 selected structures",
00190 |               "displacement": "0.075 Ang",
00191 |               "n_structures": 5
00192 |             },
00193 |             {
00194 |               "block_id": "fc_125_s1_A0p08_n5",
00195 |               "label": "FC 0.080 Ang, 5 selected structures",
00196 |               "displacement": "0.080 Ang",
00197 |               "n_structures": 5
00198 |             }
00199 |           ]
00200 |         }
00201 |       },
00202 |       {
00203 |         "recipe_id": "fc_h2o_low_amp_subsampled_125_seed2",
00204 |         "label": "H2O FC low-amplitude subsampled ladder 125 seed 2",
00205 |         "seed": 2127,
00206 |         "comparison_role": "budget_curve_FC_125",
00207 |         "scientific_note": "Budget-curve FC point with 25 low-amplitude displacements x 5 selected FC steps. Counts stay below the high-count regime that can select the zero-reference geometry.",
00208 |         "blocks": {
00209 |           "_list_length": 25,
00210 |           "_first_two": [
00211 |             {
00212 |               "block_id": "fc_125_s2_A0p006_n5",
00213 |               "label": "FC 0.006 Ang, 5 selected structures",
00214 |               "displacement": "0.006 Ang",
00215 |               "n_structures": 5
00216 |             },
00217 |             {
00218 |               "block_id": "fc_125_s2_A0p009_n5",
00219 |               "label": "FC 0.009 Ang, 5 selected structures",
00220 |               "displacement": "0.009 Ang",
00221 |               "n_structures": 5
00222 |             }
00223 |           ],
00224 |           "_last_two": [
00225 |             {
00226 |               "block_id": "fc_125_s2_A0p077_n5",
00227 |               "label": "FC 0.077 Ang, 5 selected structures",
00228 |               "displacement": "0.077 Ang",
00229 |               "n_structures": 5
00230 |             },
00231 |             {
00232 |               "block_id": "fc_125_s2_A0p082_n5",
00233 |               "label": "FC 0.082 Ang, 5 selected structures",
00234 |               "displacement": "0.082 Ang",
00235 |               "n_structures": 5
00236 |             }
00237 |           ]
00238 |         }
00239 |       }
00240 |     ],
00241 |     "_last_two": [
00242 |       {
00243 |         "recipe_id": "fc_h2o_low_amp_subsampled_285_seed2",
00244 |         "label": "H2O FC low-amplitude subsampled ladder 285 seed 2",
00245 |         "seed": 2002,
00246 |         "comparison_role": "primary_equal_N_285",
00247 |         "scientific_note": "Uses more amplitudes with fewer FC steps per amplitude because zero-reference FC geometries are invisible in this repo's benchmark splits.",
00248 |         "blocks": {
00249 |           "_list_length": 32,
00250 |           "_first_two": [
00251 |             {
00252 |               "block_id": "fc_s2_A0p006_n9",
00253 |               "label": "FC 0.006 Ang, 9 selected structures",
00254 |               "displacement": "0.006 Ang",
00255 |               "n_structures": 9
00256 |             },
00257 |             {
00258 |               "block_id": "fc_s2_A0p009_n9",
00259 |               "label": "FC 0.009 Ang, 9 selected structures",
00260 |               "displacement": "0.009 Ang",
00261 |               "n_structures": 9
00262 |             }
00263 |           ],
00264 |           "_last_two": [
00265 |             {
00266 |               "block_id": "fc_s2_A0p116_n9",
00267 |               "label": "FC 0.116 Ang, 9 selected structures",
00268 |               "displacement": "0.116 Ang",
00269 |               "n_structures": 9
00270 |             },
00271 |             {
00272 |               "block_id": "fc_s2_A0p125_n6",
00273 |               "label": "FC 0.125 Ang, 6 selected structures",
00274 |               "displacement": "0.125 Ang",
00275 |               "n_structures": 6
00276 |             }
00277 |           ]
00278 |         }
00279 |       },
00280 |       {
00281 |         "recipe_id": "fc_h2o_low_amp_subsampled_285_seed3",
00282 |         "label": "H2O FC low-amplitude subsampled ladder 285 seed 3",
00283 |         "seed": 2003,
00284 |         "comparison_role": "primary_equal_N_285",
00285 |         "scientific_note": "Uses more amplitudes with fewer FC steps per amplitude because zero-reference FC geometries are invisible in this repo's benchmark splits.",
00286 |         "blocks": {
00287 |           "_list_length": 32,
00288 |           "_first_two": [
00289 |             {
00290 |               "block_id": "fc_s3_A0p004_n9",
00291 |               "label": "FC 0.004 Ang, 9 selected structures",
00292 |               "displacement": "0.004 Ang",
00293 |               "n_structures": 9
00294 |             },
00295 |             {
00296 |               "block_id": "fc_s3_A0p007_n9",
00297 |               "label": "FC 0.007 Ang, 9 selected structures",
00298 |               "displacement": "0.007 Ang",
00299 |               "n_structures": 9
00300 |             }
00301 |           ],
00302 |           "_last_two": [
00303 |             {
00304 |               "block_id": "fc_s3_A0p114_n9",
00305 |               "label": "FC 0.114 Ang, 9 selected structures",
00306 |               "displacement": "0.114 Ang",
00307 |               "n_structures": 9
00308 |             },
00309 |             {
00310 |               "block_id": "fc_s3_A0p122_n6",
00311 |               "label": "FC 0.122 Ang, 6 selected structures",
00312 |               "displacement": "0.122 Ang",
00313 |               "n_structures": 6
00314 |             }
00315 |           ]
00316 |         }
00317 |       }
00318 |     ]
00319 |   },
00320 |   "random_cartesian": {
00321 |     "_list_length": 9,
00322 |     "_first_two": [
00323 |       {
00324 |         "recipe_id": "rc_h2o_multisigma_285_seed1",
00325 |         "label": "H2O Random Cartesian multisigma 285 seed 1",
00326 |         "seed": 3001,
00327 |         "comparison_role": "primary_equal_N_285",
00328 |         "scientific_note": "Multi-sigma low-amplitude RC around equilibrium; avoids a single arbitrary sigma.",
00329 |         "blocks": [
00330 |           {
00331 |             "block_id": "rc_h2o_multisigma_285_seed1_sigma0p005_n45",
00332 |             "label": "RC gaussian sigma 0.005 Ang, 45 structures",
00333 |             "n_structures": 45,
00334 |             "distribution": "gaussian",
00335 |             "sigma_ang": 0.005,
00336 |             "min_distance_ang": 0.65,
00337 |             "remove_center_of_mass_translation": true
00338 |           },
00339 |           {
00340 |             "block_id": "rc_h2o_multisigma_285_seed1_sigma0p01_n65",
00341 |             "label": "RC gaussian sigma 0.010 Ang, 65 structures",
00342 |             "n_structures": 65,
00343 |             "distribution": "gaussian",
00344 |             "sigma_ang": 0.01,
00345 |             "min_distance_ang": 0.65,
00346 |             "remove_center_of_mass_translation": true
00347 |           },
00348 |           {
00349 |             "block_id": "rc_h2o_multisigma_285_seed1_sigma0p02_n70",
00350 |             "label": "RC gaussian sigma 0.020 Ang, 70 structures",
00351 |             "n_structures": 70,
00352 |             "distribution": "gaussian",
00353 |             "sigma_ang": 0.02,
00354 |             "min_distance_ang": 0.65,
00355 |             "remove_center_of_mass_translation": true
00356 |           },
00357 |           {
00358 |             "block_id": "rc_h2o_multisigma_285_seed1_sigma0p035_n60",
00359 |             "label": "RC gaussian sigma 0.035 Ang, 60 structures",
00360 |             "n_structures": 60,
00361 |             "distribution": "gaussian",
00362 |             "sigma_ang": 0.035,
00363 |             "min_distance_ang": 0.65,
00364 |             "remove_center_of_mass_translation": true
00365 |           },
00366 |           {
00367 |             "block_id": "rc_h2o_multisigma_285_seed1_sigma0p05_n45",
00368 |             "label": "RC gaussian sigma 0.050 Ang, 45 structures",
00369 |             "n_structures": 45,
00370 |             "distribution": "gaussian",
00371 |             "sigma_ang": 0.05,
00372 |             "min_distance_ang": 0.65,
00373 |             "remove_center_of_mass_translation": true
00374 |           }
00375 |         ]
00376 |       },
00377 |       {
00378 |         "recipe_id": "rc_h2o_multisigma_285_seed2",
00379 |         "label": "H2O Random Cartesian multisigma 285 seed 2",
00380 |         "seed": 3002,
00381 |         "comparison_role": "primary_equal_N_285",
00382 |         "scientific_note": "Multi-sigma low-amplitude RC around equilibrium; avoids a single arbitrary sigma.",
00383 |         "blocks": [
00384 |           {
00385 |             "block_id": "rc_h2o_multisigma_285_seed2_sigma0p005_n50",
00386 |             "label": "RC gaussian sigma 0.005 Ang, 50 structures",
00387 |             "n_structures": 50,
00388 |             "distribution": "gaussian",
00389 |             "sigma_ang": 0.005,
00390 |             "min_distance_ang": 0.65,
00391 |             "remove_center_of_mass_translation": true
00392 |           },
00393 |           {
00394 |             "block_id": "rc_h2o_multisigma_285_seed2_sigma0p01_n60",
00395 |             "label": "RC gaussian sigma 0.010 Ang, 60 structures",
00396 |             "n_structures": 60,
00397 |             "distribution": "gaussian",
00398 |             "sigma_ang": 0.01,
00399 |             "min_distance_ang": 0.65,
00400 |             "remove_center_of_mass_translation": true
00401 |           },
00402 |           {
00403 |             "block_id": "rc_h2o_multisigma_285_seed2_sigma0p02_n70",
00404 |             "label": "RC gaussian sigma 0.020 Ang, 70 structures",
00405 |             "n_structures": 70,
00406 |             "distribution": "gaussian",
00407 |             "sigma_ang": 0.02,
00408 |             "min_distance_ang": 0.65,
00409 |             "remove_center_of_mass_translation": true
00410 |           },
00411 |           {
00412 |             "block_id": "rc_h2o_multisigma_285_seed2_sigma0p035_n60",
00413 |             "label": "RC gaussian sigma 0.035 Ang, 60 structures",
00414 |             "n_structures": 60,
00415 |             "distribution": "gaussian",
00416 |             "sigma_ang": 0.035,
00417 |             "min_distance_ang": 0.65,
00418 |             "remove_center_of_mass_translation": true
00419 |           },
00420 |           {
00421 |             "block_id": "rc_h2o_multisigma_285_seed2_sigma0p05_n45",
00422 |             "label": "RC gaussian sigma 0.050 Ang, 45 structures",
00423 |             "n_structures": 45,
00424 |             "distribution": "gaussian",
00425 |             "sigma_ang": 0.05,
00426 |             "min_distance_ang": 0.65,
00427 |             "remove_center_of_mass_translation": true
00428 |           }
00429 |         ]
00430 |       }
00431 |     ],
00432 |     "_last_two": [
00433 |       {
00434 |         "recipe_id": "rc_h2o_multisigma_1500_seed2",
00435 |         "label": "H2O Random Cartesian multisigma 1500 seed 2",
00436 |         "seed": 3202,
00437 |         "comparison_role": "large_scale_MD_vs_RC",
00438 |         "scientific_note": "Large RC learning-curve point matched to MD-1500, with three seeds.",
00439 |         "blocks": [
00440 |           {
00441 |             "block_id": "rc_h2o_multisigma_1500_seed2_sigma0p005_n220",
00442 |             "label": "RC gaussian sigma 0.005 Ang, 220 structures",
00443 |             "n_structures": 220,
00444 |             "distribution": "gaussian",
00445 |             "sigma_ang": 0.005,
00446 |             "min_distance_ang": 0.65,
00447 |             "remove_center_of_mass_translation": true
00448 |           },
00449 |           {
00450 |             "block_id": "rc_h2o_multisigma_1500_seed2_sigma0p01_n340",
00451 |             "label": "RC gaussian sigma 0.010 Ang, 340 structures",
00452 |             "n_structures": 340,
00453 |             "distribution": "gaussian",
00454 |             "sigma_ang": 0.01,
00455 |             "min_distance_ang": 0.65,
00456 |             "remove_center_of_mass_translation": true
00457 |           },
00458 |           {
00459 |             "block_id": "rc_h2o_multisigma_1500_seed2_sigma0p02_n380",
00460 |             "label": "RC gaussian sigma 0.020 Ang, 380 structures",
00461 |             "n_structures": 380,
00462 |             "distribution": "gaussian",
00463 |             "sigma_ang": 0.02,
00464 |             "min_distance_ang": 0.65,
00465 |             "remove_center_of_mass_translation": true
00466 |           },
00467 |           {
00468 |             "block_id": "rc_h2o_multisigma_1500_seed2_sigma0p035_n360",
00469 |             "label": "RC gaussian sigma 0.035 Ang, 360 structures",
00470 |             "n_structures": 360,
00471 |             "distribution": "gaussian",
00472 |             "sigma_ang": 0.035,
00473 |             "min_distance_ang": 0.65,
00474 |             "remove_center_of_mass_translation": true
00475 |           },
00476 |           {
00477 |             "block_id": "rc_h2o_multisigma_1500_seed2_sigma0p05_n200",
00478 |             "label": "RC gaussian sigma 0.050 Ang, 200 structures",
00479 |             "n_structures": 200,
00480 |             "distribution": "gaussian",
00481 |             "sigma_ang": 0.05,
00482 |             "min_distance_ang": 0.65,
00483 |             "remove_center_of_mass_translation": true
00484 |           }
00485 |         ]
00486 |       },
00487 |       {
00488 |         "recipe_id": "rc_h2o_multisigma_1500_seed3",
00489 |         "label": "H2O Random Cartesian multisigma 1500 seed 3",
00490 |         "seed": 3203,
00491 |         "comparison_role": "large_scale_MD_vs_RC",
00492 |         "scientific_note": "Large RC learning-curve point matched to MD-1500, with three seeds.",
00493 |         "blocks": [
00494 |           {
00495 |             "block_id": "rc_h2o_multisigma_1500_seed3_sigma0p005_n180",
00496 |             "label": "RC gaussian sigma 0.005 Ang, 180 structures",
00497 |             "n_structures": 180,
00498 |             "distribution": "gaussian",
00499 |             "sigma_ang": 0.005,
00500 |             "min_distance_ang": 0.65,
00501 |             "remove_center_of_mass_translation": true
00502 |           },
00503 |           {
00504 |             "block_id": "rc_h2o_multisigma_1500_seed3_sigma0p01_n370",
00505 |             "label": "RC gaussian sigma 0.010 Ang, 370 structures",
00506 |             "n_structures": 370,
00507 |             "distribution": "gaussian",
00508 |             "sigma_ang": 0.01,
00509 |             "min_distance_ang": 0.65,
00510 |             "remove_center_of_mass_translation": true
00511 |           },
00512 |           {
00513 |             "block_id": "rc_h2o_multisigma_1500_seed3_sigma0p02_n410",
00514 |             "label": "RC gaussian sigma 0.020 Ang, 410 structures",
00515 |             "n_structures": 410,
00516 |             "distribution": "gaussian",
00517 |             "sigma_ang": 0.02,
00518 |             "min_distance_ang": 0.65,
00519 |             "remove_center_of_mass_translation": true
00520 |           },
00521 |           {
00522 |             "block_id": "rc_h2o_multisigma_1500_seed3_sigma0p035_n340",
00523 |             "label": "RC gaussian sigma 0.035 Ang, 340 structures",
00524 |             "n_structures": 340,
00525 |             "distribution": "gaussian",
00526 |             "sigma_ang": 0.035,
00527 |             "min_distance_ang": 0.65,
00528 |             "remove_center_of_mass_translation": true
00529 |           },
00530 |           {
00531 |             "block_id": "rc_h2o_multisigma_1500_seed3_sigma0p05_n200",
00532 |             "label": "RC gaussian sigma 0.050 Ang, 200 structures",
00533 |             "n_structures": 200,
00534 |             "distribution": "gaussian",
00535 |             "sigma_ang": 0.05,
00536 |             "min_distance_ang": 0.65,
00537 |             "remove_center_of_mass_translation": true
00538 |           }
00539 |         ]
00540 |       }
00541 |     ]
00542 |   }
00543 | }
```

## `Comparison/dataset_recipes/scientific_large_3seed_equalN.json` — vista compacta

SHA-256 del JSON completo: `3d6a71b77f35ed3360618be66b41143a190dac245e14cacb462d91072d7b9041`. Las listas de más de ocho elementos conservan longitud, dos primeros y dos últimos elementos; esta vista no sustituye al artefacto para verificaciones cuantitativas.

```json
00001 | {
00002 |   "md": {
00003 |     "_list_length": 9,
00004 |     "_first_two": [
00005 |       {
00006 |         "recipe_id": "md_ladder_285_seed101",
00007 |         "label": "MD ladder 285 (balanced 15-temperature ladder, 19 snapshots per temperature), seed 101",
00008 |         "seed": 101,
00009 |         "blocks": {
00010 |           "_list_length": 15,
00011 |           "_first_two": [
00012 |             {
00013 |               "block_id": "md_T300_n19",
00014 |               "label": "19 snapshots @ 300 K",
00015 |               "temperature_K": 300,
00016 |               "n_snapshots": 19,
00017 |               "seed": 101001
00018 |             },
00019 |             {
00020 |               "block_id": "md_T350_n19",
00021 |               "label": "19 snapshots @ 350 K",
00022 |               "temperature_K": 350,
00023 |               "n_snapshots": 19,
00024 |               "seed": 101002
00025 |             }
00026 |           ],
00027 |           "_last_two": [
00028 |             {
00029 |               "block_id": "md_T950_n19",
00030 |               "label": "19 snapshots @ 950 K",
00031 |               "temperature_K": 950,
00032 |               "n_snapshots": 19,
00033 |               "seed": 101014
00034 |             },
00035 |             {
00036 |               "block_id": "md_T1000_n19",
00037 |               "label": "19 snapshots @ 1000 K",
00038 |               "temperature_K": 1000,
00039 |               "n_snapshots": 19,
00040 |               "seed": 101015
00041 |             }
00042 |           ]
00043 |         }
00044 |       },
00045 |       {
00046 |         "recipe_id": "md_ladder_285_seed202",
00047 |         "label": "MD ladder 285 (balanced 15-temperature ladder, 19 snapshots per temperature), seed 202",
00048 |         "seed": 202,
00049 |         "blocks": {
00050 |           "_list_length": 15,
00051 |           "_first_two": [
00052 |             {
00053 |               "block_id": "md_T300_n19",
00054 |               "label": "19 snapshots @ 300 K",
00055 |               "temperature_K": 300,
00056 |               "n_snapshots": 19,
00057 |               "seed": 202001
00058 |             },
00059 |             {
00060 |               "block_id": "md_T350_n19",
00061 |               "label": "19 snapshots @ 350 K",
00062 |               "temperature_K": 350,
00063 |               "n_snapshots": 19,
00064 |               "seed": 202002
00065 |             }
00066 |           ],
00067 |           "_last_two": [
00068 |             {
00069 |               "block_id": "md_T950_n19",
00070 |               "label": "19 snapshots @ 950 K",
00071 |               "temperature_K": 950,
00072 |               "n_snapshots": 19,
00073 |               "seed": 202014
00074 |             },
00075 |             {
00076 |               "block_id": "md_T1000_n19",
00077 |               "label": "19 snapshots @ 1000 K",
00078 |               "temperature_K": 1000,
00079 |               "n_snapshots": 19,
00080 |               "seed": 202015
00081 |             }
00082 |           ]
00083 |         }
00084 |       }
00085 |     ],
00086 |     "_last_two": [
00087 |       {
00088 |         "recipe_id": "md_ladder_760_seed202",
00089 |         "label": "MD ladder 760 (strong 15-temperature ladder, 50-51 snapshots per temperature), seed 202",
00090 |         "seed": 202,
00091 |         "blocks": {
00092 |           "_list_length": 15,
00093 |           "_first_two": [
00094 |             {
00095 |               "block_id": "md_T300_n51",
00096 |               "label": "51 snapshots @ 300 K",
00097 |               "temperature_K": 300,
00098 |               "n_snapshots": 51,
00099 |               "seed": 202001
00100 |             },
00101 |             {
00102 |               "block_id": "md_T350_n51",
00103 |               "label": "51 snapshots @ 350 K",
00104 |               "temperature_K": 350,
00105 |               "n_snapshots": 51,
00106 |               "seed": 202002
00107 |             }
00108 |           ],
00109 |           "_last_two": [
00110 |             {
00111 |               "block_id": "md_T950_n50",
00112 |               "label": "50 snapshots @ 950 K",
00113 |               "temperature_K": 950,
00114 |               "n_snapshots": 50,
00115 |               "seed": 202014
00116 |             },
00117 |             {
00118 |               "block_id": "md_T1000_n50",
00119 |               "label": "50 snapshots @ 1000 K",
00120 |               "temperature_K": 1000,
00121 |               "n_snapshots": 50,
00122 |               "seed": 202015
00123 |             }
00124 |           ]
00125 |         }
00126 |       },
00127 |       {
00128 |         "recipe_id": "md_ladder_760_seed303",
00129 |         "label": "MD ladder 760 (strong 15-temperature ladder, 50-51 snapshots per temperature), seed 303",
00130 |         "seed": 303,
00131 |         "blocks": {
00132 |           "_list_length": 15,
00133 |           "_first_two": [
00134 |             {
00135 |               "block_id": "md_T300_n51",
00136 |               "label": "51 snapshots @ 300 K",
00137 |               "temperature_K": 300,
00138 |               "n_snapshots": 51,
00139 |               "seed": 303001
00140 |             },
00141 |             {
00142 |               "block_id": "md_T350_n51",
00143 |               "label": "51 snapshots @ 350 K",
00144 |               "temperature_K": 350,
00145 |               "n_snapshots": 51,
00146 |               "seed": 303002
00147 |             }
00148 |           ],
00149 |           "_last_two": [
00150 |             {
00151 |               "block_id": "md_T950_n50",
00152 |               "label": "50 snapshots @ 950 K",
00153 |               "temperature_K": 950,
00154 |               "n_snapshots": 50,
00155 |               "seed": 303014
00156 |             },
00157 |             {
00158 |               "block_id": "md_T1000_n50",
00159 |               "label": "50 snapshots @ 1000 K",
00160 |               "temperature_K": 1000,
00161 |               "n_snapshots": 50,
00162 |               "seed": 303015
00163 |             }
00164 |           ]
00165 |         }
00166 |       }
00167 |     ]
00168 |   },
00169 |   "siesta_fc_cartesian": {
00170 |     "_list_length": 9,
00171 |     "_first_two": [
00172 |       {
00173 |         "recipe_id": "fc_multiamp_285_seed101",
00174 |         "label": "FC multi-amplitude 285 (15 amplitudes), seed 101",
00175 |         "seed": 101,
00176 |         "blocks": {
00177 |           "_list_length": 15,
00178 |           "_first_two": [
00179 |             {
00180 |               "block_id": "fc_d0p010_n19",
00181 |               "label": "19 structures @ 0.010 Ang",
00182 |               "displacement": "0.010 Ang",
00183 |               "n_structures": 19
00184 |             },
00185 |             {
00186 |               "block_id": "fc_d0p020_n19",
00187 |               "label": "19 structures @ 0.020 Ang",
00188 |               "displacement": "0.020 Ang",
00189 |               "n_structures": 19
00190 |             }
00191 |           ],
00192 |           "_last_two": [
00193 |             {
00194 |               "block_id": "fc_d0p140_n19",
00195 |               "label": "19 structures @ 0.140 Ang",
00196 |               "displacement": "0.140 Ang",
00197 |               "n_structures": 19
00198 |             },
00199 |             {
00200 |               "block_id": "fc_d0p150_n19",
00201 |               "label": "19 structures @ 0.150 Ang",
00202 |               "displacement": "0.150 Ang",
00203 |               "n_structures": 19
00204 |             }
00205 |           ]
00206 |         }
00207 |       },
00208 |       {
00209 |         "recipe_id": "fc_multiamp_285_seed202",
00210 |         "label": "FC multi-amplitude 285 (15 amplitudes), seed 202",
00211 |         "seed": 202,
00212 |         "blocks": {
00213 |           "_list_length": 15,
00214 |           "_first_two": [
00215 |             {
00216 |               "block_id": "fc_d0p010_n19",
00217 |               "label": "19 structures @ 0.010 Ang",
00218 |               "displacement": "0.010 Ang",
00219 |               "n_structures": 19
00220 |             },
00221 |             {
00222 |               "block_id": "fc_d0p020_n19",
00223 |               "label": "19 structures @ 0.020 Ang",
00224 |               "displacement": "0.020 Ang",
00225 |               "n_structures": 19
00226 |             }
00227 |           ],
00228 |           "_last_two": [
00229 |             {
00230 |               "block_id": "fc_d0p140_n19",
00231 |               "label": "19 structures @ 0.140 Ang",
00232 |               "displacement": "0.140 Ang",
00233 |               "n_structures": 19
00234 |             },
00235 |             {
00236 |               "block_id": "fc_d0p150_n19",
00237 |               "label": "19 structures @ 0.150 Ang",
00238 |               "displacement": "0.150 Ang",
00239 |               "n_structures": 19
00240 |             }
00241 |           ]
00242 |         }
00243 |       }
00244 |     ],
00245 |     "_last_two": [
00246 |       {
00247 |         "recipe_id": "fc_multiamp_760_seed202",
00248 |         "label": "FC multi-amplitude 760 (40 amplitudes), seed 202",
00249 |         "seed": 202,
00250 |         "blocks": {
00251 |           "_list_length": 40,
00252 |           "_first_two": [
00253 |             {
00254 |               "block_id": "fc_d0p005_n19",
00255 |               "label": "19 structures @ 0.005 Ang",
00256 |               "displacement": "0.005 Ang",
00257 |               "n_structures": 19
00258 |             },
00259 |             {
00260 |               "block_id": "fc_d0p010_n19",
00261 |               "label": "19 structures @ 0.010 Ang",
00262 |               "displacement": "0.010 Ang",
00263 |               "n_structures": 19
00264 |             }
00265 |           ],
00266 |           "_last_two": [
00267 |             {
00268 |               "block_id": "fc_d0p195_n19",
00269 |               "label": "19 structures @ 0.195 Ang",
00270 |               "displacement": "0.195 Ang",
00271 |               "n_structures": 19
00272 |             },
00273 |             {
00274 |               "block_id": "fc_d0p200_n19",
00275 |               "label": "19 structures @ 0.200 Ang",
00276 |               "displacement": "0.200 Ang",
00277 |               "n_structures": 19
00278 |             }
00279 |           ]
00280 |         }
00281 |       },
00282 |       {
00283 |         "recipe_id": "fc_multiamp_760_seed303",
00284 |         "label": "FC multi-amplitude 760 (40 amplitudes), seed 303",
00285 |         "seed": 303,
00286 |         "blocks": {
00287 |           "_list_length": 40,
00288 |           "_first_two": [
00289 |             {
00290 |               "block_id": "fc_d0p005_n19",
00291 |               "label": "19 structures @ 0.005 Ang",
00292 |               "displacement": "0.005 Ang",
00293 |               "n_structures": 19
00294 |             },
00295 |             {
00296 |               "block_id": "fc_d0p010_n19",
00297 |               "label": "19 structures @ 0.010 Ang",
00298 |               "displacement": "0.010 Ang",
00299 |               "n_structures": 19
00300 |             }
00301 |           ],
00302 |           "_last_two": [
00303 |             {
00304 |               "block_id": "fc_d0p195_n19",
00305 |               "label": "19 structures @ 0.195 Ang",
00306 |               "displacement": "0.195 Ang",
00307 |               "n_structures": 19
00308 |             },
00309 |             {
00310 |               "block_id": "fc_d0p200_n19",
00311 |               "label": "19 structures @ 0.200 Ang",
00312 |               "displacement": "0.200 Ang",
00313 |               "n_structures": 19
00314 |             }
00315 |           ]
00316 |         }
00317 |       }
00318 |     ]
00319 |   },
00320 |   "random_cartesian": {
00321 |     "_list_length": 9,
00322 |     "_first_two": [
00323 |       {
00324 |         "recipe_id": "rc_multisigma_285_seed101",
00325 |         "label": "Random Cartesian multisigma 285 (5 sigmas), seed 101",
00326 |         "seed": 101,
00327 |         "blocks": [
00328 |           {
00329 |             "block_id": "rc_sigma_0p010_n57",
00330 |             "label": "57 gaussian structures @ sigma 0.010 Ang",
00331 |             "n_structures": 57,
00332 |             "distribution": "gaussian",
00333 |             "sigma_ang": 0.01,
00334 |             "seed": 101001,
00335 |             "min_distance_ang": 0.65,
00336 |             "move_atoms": "all",
00337 |             "remove_center_of_mass_translation": true
00338 |           },
00339 |           {
00340 |             "block_id": "rc_sigma_0p020_n57",
00341 |             "label": "57 gaussian structures @ sigma 0.020 Ang",
00342 |             "n_structures": 57,
00343 |             "distribution": "gaussian",
00344 |             "sigma_ang": 0.02,
00345 |             "seed": 101002,
00346 |             "min_distance_ang": 0.65,
00347 |             "move_atoms": "all",
00348 |             "remove_center_of_mass_translation": true
00349 |           },
00350 |           {
00351 |             "block_id": "rc_sigma_0p030_n57",
00352 |             "label": "57 gaussian structures @ sigma 0.030 Ang",
00353 |             "n_structures": 57,
00354 |             "distribution": "gaussian",
00355 |             "sigma_ang": 0.03,
00356 |             "seed": 101003,
00357 |             "min_distance_ang": 0.65,
00358 |             "move_atoms": "all",
00359 |             "remove_center_of_mass_translation": true
00360 |           },
00361 |           {
00362 |             "block_id": "rc_sigma_0p050_n57",
00363 |             "label": "57 gaussian structures @ sigma 0.050 Ang",
00364 |             "n_structures": 57,
00365 |             "distribution": "gaussian",
00366 |             "sigma_ang": 0.05,
00367 |             "seed": 101004,
00368 |             "min_distance_ang": 0.65,
00369 |             "move_atoms": "all",
00370 |             "remove_center_of_mass_translation": true
00371 |           },
00372 |           {
00373 |             "block_id": "rc_sigma_0p080_n57",
00374 |             "label": "57 gaussian structures @ sigma 0.080 Ang",
00375 |             "n_structures": 57,
00376 |             "distribution": "gaussian",
00377 |             "sigma_ang": 0.08,
00378 |             "seed": 101005,
00379 |             "min_distance_ang": 0.65,
00380 |             "move_atoms": "all",
00381 |             "remove_center_of_mass_translation": true
00382 |           }
00383 |         ]
00384 |       },
00385 |       {
00386 |         "recipe_id": "rc_multisigma_285_seed202",
00387 |         "label": "Random Cartesian multisigma 285 (5 sigmas), seed 202",
00388 |         "seed": 202,
00389 |         "blocks": [
00390 |           {
00391 |             "block_id": "rc_sigma_0p010_n57",
00392 |             "label": "57 gaussian structures @ sigma 0.010 Ang",
00393 |             "n_structures": 57,
00394 |             "distribution": "gaussian",
00395 |             "sigma_ang": 0.01,
00396 |             "seed": 202001,
00397 |             "min_distance_ang": 0.65,
00398 |             "move_atoms": "all",
00399 |             "remove_center_of_mass_translation": true
00400 |           },
00401 |           {
00402 |             "block_id": "rc_sigma_0p020_n57",
00403 |             "label": "57 gaussian structures @ sigma 0.020 Ang",
00404 |             "n_structures": 57,
00405 |             "distribution": "gaussian",
00406 |             "sigma_ang": 0.02,
00407 |             "seed": 202002,
00408 |             "min_distance_ang": 0.65,
00409 |             "move_atoms": "all",
00410 |             "remove_center_of_mass_translation": true
00411 |           },
00412 |           {
00413 |             "block_id": "rc_sigma_0p030_n57",
00414 |             "label": "57 gaussian structures @ sigma 0.030 Ang",
00415 |             "n_structures": 57,
00416 |             "distribution": "gaussian",
00417 |             "sigma_ang": 0.03,
00418 |             "seed": 202003,
00419 |             "min_distance_ang": 0.65,
00420 |             "move_atoms": "all",
00421 |             "remove_center_of_mass_translation": true
00422 |           },
00423 |           {
00424 |             "block_id": "rc_sigma_0p050_n57",
00425 |             "label": "57 gaussian structures @ sigma 0.050 Ang",
00426 |             "n_structures": 57,
00427 |             "distribution": "gaussian",
00428 |             "sigma_ang": 0.05,
00429 |             "seed": 202004,
00430 |             "min_distance_ang": 0.65,
00431 |             "move_atoms": "all",
00432 |             "remove_center_of_mass_translation": true
00433 |           },
00434 |           {
00435 |             "block_id": "rc_sigma_0p080_n57",
00436 |             "label": "57 gaussian structures @ sigma 0.080 Ang",
00437 |             "n_structures": 57,
00438 |             "distribution": "gaussian",
00439 |             "sigma_ang": 0.08,
00440 |             "seed": 202005,
00441 |             "min_distance_ang": 0.65,
00442 |             "move_atoms": "all",
00443 |             "remove_center_of_mass_translation": true
00444 |           }
00445 |         ]
00446 |       }
00447 |     ],
00448 |     "_last_two": [
00449 |       {
00450 |         "recipe_id": "rc_multisigma_760_seed202",
00451 |         "label": "Random Cartesian multisigma 760 (10 sigmas), seed 202",
00452 |         "seed": 202,
00453 |         "blocks": {
00454 |           "_list_length": 10,
00455 |           "_first_two": [
00456 |             {
00457 |               "block_id": "rc_sigma_0p005_n76",
00458 |               "label": "76 gaussian structures @ sigma 0.005 Ang",
00459 |               "n_structures": 76,
00460 |               "distribution": "gaussian",
00461 |               "sigma_ang": 0.005,
00462 |               "seed": 202001,
00463 |               "min_distance_ang": 0.65,
00464 |               "move_atoms": "all",
00465 |               "remove_center_of_mass_translation": true
00466 |             },
00467 |             {
00468 |               "block_id": "rc_sigma_0p010_n76",
00469 |               "label": "76 gaussian structures @ sigma 0.010 Ang",
00470 |               "n_structures": 76,
00471 |               "distribution": "gaussian",
00472 |               "sigma_ang": 0.01,
00473 |               "seed": 202002,
00474 |               "min_distance_ang": 0.65,
00475 |               "move_atoms": "all",
00476 |               "remove_center_of_mass_translation": true
00477 |             }
00478 |           ],
00479 |           "_last_two": [
00480 |             {
00481 |               "block_id": "rc_sigma_0p100_n76",
00482 |               "label": "76 gaussian structures @ sigma 0.100 Ang",
00483 |               "n_structures": 76,
00484 |               "distribution": "gaussian",
00485 |               "sigma_ang": 0.1,
00486 |               "seed": 202009,
00487 |               "min_distance_ang": 0.65,
00488 |               "move_atoms": "all",
00489 |               "remove_center_of_mass_translation": true
00490 |             },
00491 |             {
00492 |               "block_id": "rc_sigma_0p130_n76",
00493 |               "label": "76 gaussian structures @ sigma 0.130 Ang",
00494 |               "n_structures": 76,
00495 |               "distribution": "gaussian",
00496 |               "sigma_ang": 0.13,
00497 |               "seed": 202010,
00498 |               "min_distance_ang": 0.65,
00499 |               "move_atoms": "all",
00500 |               "remove_center_of_mass_translation": true
00501 |             }
00502 |           ]
00503 |         }
00504 |       },
00505 |       {
00506 |         "recipe_id": "rc_multisigma_760_seed303",
00507 |         "label": "Random Cartesian multisigma 760 (10 sigmas), seed 303",
00508 |         "seed": 303,
00509 |         "blocks": {
00510 |           "_list_length": 10,
00511 |           "_first_two": [
00512 |             {
00513 |               "block_id": "rc_sigma_0p005_n76",
00514 |               "label": "76 gaussian structures @ sigma 0.005 Ang",
00515 |               "n_structures": 76,
00516 |               "distribution": "gaussian",
00517 |               "sigma_ang": 0.005,
00518 |               "seed": 303001,
00519 |               "min_distance_ang": 0.65,
00520 |               "move_atoms": "all",
00521 |               "remove_center_of_mass_translation": true
00522 |             },
00523 |             {
00524 |               "block_id": "rc_sigma_0p010_n76",
00525 |               "label": "76 gaussian structures @ sigma 0.010 Ang",
00526 |               "n_structures": 76,
00527 |               "distribution": "gaussian",
00528 |               "sigma_ang": 0.01,
00529 |               "seed": 303002,
00530 |               "min_distance_ang": 0.65,
00531 |               "move_atoms": "all",
00532 |               "remove_center_of_mass_translation": true
00533 |             }
00534 |           ],
00535 |           "_last_two": [
00536 |             {
00537 |               "block_id": "rc_sigma_0p100_n76",
00538 |               "label": "76 gaussian structures @ sigma 0.100 Ang",
00539 |               "n_structures": 76,
00540 |               "distribution": "gaussian",
00541 |               "sigma_ang": 0.1,
00542 |               "seed": 303009,
00543 |               "min_distance_ang": 0.65,
00544 |               "move_atoms": "all",
00545 |               "remove_center_of_mass_translation": true
00546 |             },
00547 |             {
00548 |               "block_id": "rc_sigma_0p130_n76",
00549 |               "label": "76 gaussian structures @ sigma 0.130 Ang",
00550 |               "n_structures": 76,
00551 |               "distribution": "gaussian",
00552 |               "sigma_ang": 0.13,
00553 |               "seed": 303010,
00554 |               "min_distance_ang": 0.65,
00555 |               "move_atoms": "all",
00556 |               "remove_center_of_mass_translation": true
00557 |             }
00558 |           ]
00559 |         }
00560 |       }
00561 |     ]
00562 |   }
00563 | }
```
