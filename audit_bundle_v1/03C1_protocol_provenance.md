# Dossier 2A — Protocolo, splits y procedencia

## Objeto de revisión

Auditar si ambos modelos resuelven el mismo problema: base y orden orbital, vectores R, espín, overlap, referencia energética, k-points, splits y gates.

## Condiciones del contexto

- Contexto ciego: no contiene informes de auditoría previos ni prompts históricos.
- Las líneas prefijadas `NNNNN |` conservan la numeración del archivo fuente.
- Los extractos Python omiten funciones operativas sin semántica científica directa.
- Ausencia de un archivo o función en este dossier significa `no evaluable`, no `correcto`.

## `docs/graph2mat_deeph_benchmark.md`

SHA-256: `590fd2890f524a83c69b3be0c3553baff0892ec9fa9d225a83ea6c2aa4bad3f7`

```md
00001 | # Graph2Mat vs DeepH Joint Benchmark
00002 | 
00003 | This document describes the controlled Graph2Mat vs DeepH workflow for
00004 | graphene Hamiltonians generated with the repository SIESTA/Wannier90 stack.
00005 | The goal is not to reproduce the DeepH paper exactly. The goal is a
00006 | cross-architecture benchmark where the DFT backend, basis, pseudopotentials,
00007 | snapshots, splits, references, overlap policy, and metrics are fixed.
00008 | 
00009 | Companion documents:
00010 | 
00011 | - `README.md` for the repository entrypoints and material-bundle overview.
00012 | - `docs/workflows.md` for the UI/API workflow map.
00013 | - `docs/data_and_outputs.md` for the artifact and result layout.
00014 | 
00015 | ## Why The Joint Artifact Contract Exists
00016 | 
00017 | Graph2Mat can train from SIESTA snapshots that contain only the artifacts used
00018 | by the Graph2Mat path. DeepH's SIESTA preprocessing needs a different raw
00019 | artifact set. The joint benchmark therefore uses the versioned contract:
00020 | 
00021 | ```text
00022 | joint_graph2mat_deeph_artifact_contract_v1
00023 | ```
00024 | 
00025 | A snapshot is benchmark-ready only when one SIESTA calculation produced and
00026 | archived every artifact needed by both methods. Missing artifacts are a
00027 | scientific blocker, not a warning to ignore.
00028 | 
00029 | ## Old Failure Mode
00030 | 
00031 | The previous Graph2Mat-only dataset archive stored snapshots like:
00032 | 
00033 | ```text
00034 | RUN.fdf
00035 | graphene.TSHS
00036 | graphene.TSDE
00037 | graphene.XV
00038 | metadata.json
00039 | ```
00040 | 
00041 | That was enough for the Graph2Mat/evaluator path, but not enough for DeepH.
00042 | DeepH with the SIESTA interface requires at least:
00043 | 
00044 | ```text
00045 | graphene.HSX
00046 | graphene.STRUCT_OUT
00047 | graphene.XV
00048 | graphene.ORB_INDX
00049 | ```
00050 | 
00051 | Because `graphene.HSX`, `graphene.STRUCT_OUT`, and `graphene.ORB_INDX` were
00052 | not archived, the recovery path had to rerun SIESTA once per snapshot. On the
00053 | 1500-snapshot graphene dataset this cost roughly 3.8-3.9 minutes per snapshot,
00054 | nearly 100 sequential hours.
00055 | 
00056 | The fix is one-pass artifact generation: run SIESTA once per snapshot during
00057 | dataset generation and archive all Graph2Mat and DeepH artifacts immediately.
00058 | The normal benchmark workflow must not silently rerun SIESTA to repair missing
00059 | DeepH files. Any repair path must be explicit, opt-in, and visibly marked
00060 | slow/expensive.
00061 | 
00062 | ## Required Per-Snapshot Artifacts
00063 | 
00064 | A joint Graph2Mat+DeepH benchmark snapshot must contain:
00065 | 
00066 | ```text
00067 | RUN.fdf
00068 | RUN.out or siesta.out
00069 | SystemLabel.TSHS
00070 | SystemLabel.TSDE
00071 | SystemLabel.HSX
00072 | SystemLabel.STRUCT_OUT
00073 | SystemLabel.XV
00074 | SystemLabel.ORB_INDX
00075 | metadata.json
00076 | ```
00077 | 
00078 | Dataset-level provenance must also make the material identity traceable and
00079 | must include basis and pseudopotential provenance or hashes when required by
00080 | the workflow. For graphene the `SystemLabel` is normally `graphene`, but the
00081 | validator resolves the label from real dataset metadata and FDF content rather
00082 | than assuming every dataset is graphene.
00083 | 
00084 | The validator writes or consumes:
00085 | 
00086 | ```text
00087 | artifact_validation.json
00088 | benchmark_dataset_manifest.json
00089 | frozen_split_manifest.json
00090 | ```
00091 | 
00092 | `benchmark_dataset_manifest.json` records the artifact contract and dataset
00093 | status. `frozen_split_manifest.json` records the exact train/validation/test
00094 | sample IDs and artifact hashes used by both Graph2Mat and DeepH.
00095 | 
00096 | ## Dataset Modes
00097 | 
00098 | The UI and runner distinguish three dataset modes.
00099 | 
00100 | | Mode | Meaning | Scientific status |
00101 | | --- | --- | --- |
00102 | | Clean one-pass dataset | SIESTA generated all joint artifacts during the original dataset generation. | Preferred. Can be robust if all later compatibility checks pass. |
00103 | | Reused validated joint dataset | Existing dataset is validated against the joint artifact contract before training. | Valid reuse if manifests, hashes, splits, material, basis and pseudos are compatible. |
00104 | | Repaired dataset | Missing artifacts are regenerated only after explicit user opt-in. | Must be marked `valid_repaired_dataset_with_warning` or diagnostic/exploratory depending on provenance. |
00105 | 
00106 | If a Graph2Mat-only legacy dataset is missing `HSX`, `STRUCT_OUT`, or
00107 | `ORB_INDX`, the validator classifies it as invalid or repair-required. The
00108 | normal one-click benchmark does not repair it silently.
00109 | 
00110 | ## Running A Smoke Benchmark
00111 | 
00112 | Use this for UI/API plumbing and manifest validation, not for publication
00113 | metrics.
00114 | 
00115 | The smallest backend smoke is the dedicated script:
00116 | 
00117 | ```bash
00118 | python3 Comparison/scripts/g2m_deeph_smoke.py \
00119 |   --dry-run \
00120 |   --output-root Comparison/results/g2m_deeph_smoke_dry_run
00121 | ```
00122 | 
00123 | The dry-run never launches SIESTA, Graph2Mat training, or DeepH. It exercises
00124 | the Graph2Mat vs DeepH runner in dataset-generation planning mode and writes:
00125 | 
00126 | ```text
00127 | smoke_manifest.json
00128 | artifact_validation.json
00129 | benchmark_manifest.json
00130 | recommendation.json
00131 | logs/smoke.log
00132 | ```
00133 | 
00134 | `artifact_validation.json` records the required joint snapshot contract:
00135 | `RUN.fdf`, `SystemLabel.TSHS`, `SystemLabel.TSDE`, `SystemLabel.HSX`,
00136 | `SystemLabel.STRUCT_OUT`, `SystemLabel.XV`, `SystemLabel.ORB_INDX`, and
00137 | `metadata.json`. In dry-run mode those files are not claimed to exist; the
00138 | smoke only verifies that the planned workflow knows the required artifact set.
00139 | 
00140 | A real tiny smoke is opt-in and should be run manually only on a machine with
00141 | the required executables:
00142 | 
00143 | ```bash
00144 | RUN_G2M_DEEPH_REAL_SMOKE=1 \
00145 | SIESTA_COMMAND=siesta \
00146 | DEEPH_PACK_ROOT=/path/to/DeepH-pack \
00147 | python3 Comparison/scripts/g2m_deeph_smoke.py \
00148 |   --tiny-real \
00149 |   --sample-limit 6 \
00150 |   --epochs 1 \
00151 |   --output-root Comparison/results/g2m_deeph_smoke_real
00152 | ```
00153 | 
00154 | The real smoke checks SIESTA one-pass joint artifact generation, Graph2Mat,
00155 | DeepH preprocess/train/inference, fail-closed metrics, manifests, and an honest
00156 | diagnostic/recommendation summary. If `RUN_G2M_DEEPH_REAL_SMOKE=1` or any
00157 | dependency is missing, the command writes `status: skipped` rather than
00158 | pretending the smoke passed.
00159 | 
00160 | 1. Start the UI:
00161 | 
00162 |    ```bash
00163 |    python3 Comparison/scripts/pipeline_ui.py
00164 |    ```
00165 | 
00166 | 2. Open the `G2M vs DeepH` tab.
00167 | 
00168 | 3. Select `Reuse existing validated joint dataset`.
00169 | 
00170 | 4. Point `Dataset root` to a tiny dataset that already contains the joint
00171 |    artifacts and manifests. The default persistent location is
00172 |    `Comparison/datasets/graphene_w90_joint`, separated from `results/` and
00173 |    `workspaces/`.
00174 | 
00175 | 5. Click `Validate dataset artifacts`. The artifact table must show all
00176 |    snapshots valid and no missing `HSX`, `STRUCT_OUT`, `XV`, or `ORB_INDX`.
00177 | 
00178 | 6. For a backend dry-run smoke through the API, keep the UI server running and
00179 |    use:
00180 | 
00181 |    ```bash
00182 |    curl -sS -X POST http://127.0.0.1:8770/api/g2m-deeph/run \
00183 |      -H 'Content-Type: application/json' \
00184 |      -d '{
00185 |        "dataset_root": "Comparison/datasets/graphene_w90_joint",
00186 |        "system_label": "graphene",
00187 |        "dry_run": true,
00188 |        "graph2mat_overrides": {"max_epochs": 1},
00189 |        "deeph": {"epochs": 1, "batch_size": 1}
00190 |      }'
00191 |    ```
00192 | 
00193 | The dry-run checks runner wiring and writes local configs/manifests where the
00194 | runner phase supports it, but it does not prove model accuracy.
00195 | 
00196 | ## Optional derivative post-processing
00197 | 
00198 | Hamiltonian derivative diagnostics are optional post-processing outputs for
00199 | technical internal diagnostic use. They do not replace the main H-vs-H
00200 | benchmark, and if they are not computed the benchmark remains valid for the
00201 | standard Hamiltonian metrics and winner logic.
00202 | 
00203 | The derivative evaluator remains optional and diagnostic-only, but the
00204 | benchmark runner can now invoke it automatically at the end of
00205 | `common_metrics` when `derivative_metrics.enabled=true`. The same logic is
00206 | also available as a reusable backfill CLI for already completed sweeps.
00207 | 
00208 | Manual/offline execution is still supported:
00209 | 
00210 | ```bash
00211 | python3 Comparison/scripts/evaluate_hamiltonian_derivative_metrics.py \
00212 |   --run-root <benchmark_run_root> \
00213 |   --output-dir <benchmark_run_root>/common_metrics/graph2mat_eval/derivative_metrics
00214 | 
00215 | python3 Comparison/scripts/g2m_deeph_derivative_gate_check.py \
00216 |   --run-root <benchmark_run_root> \
00217 |   --output <benchmark_run_root>/common_metrics/summary/derivative_gate_report.json
00218 | 
00219 | python3 Comparison/scripts/g2m_deeph_runner.py \
00220 |   --backfill-derivatives-from-training-sweep <benchmark_run_root>/sweep/training_sweep_manifest.json \
00221 |   --overwrite \
00222 |   --diagnostic-only \
00223 |   --split test \
00224 |   --method central \
00225 |   --require-central \
00226 |   --support-threshold 1e-12
00227 | ```
00228 | 
00229 | Expected derivative artifacts live under `derivative_metrics/` and include:
00230 | 
00231 | - `manifest.json`
00232 | - `derivative_matrix_metrics.csv`
00233 | - `derivative_hermiticity.csv`
00234 | - `stencil_status.csv`
00235 | - `derivative_summary.json`
00236 | 
00237 | The benchmark UI also expects:
00238 | 
00239 | - `common_metrics/summary/derivative_plots/derivative_plot_payload.json`
00240 | - `common_metrics/summary/derivative_plots/derivative_plot_manifest.json`
00241 | - `common_metrics/summary/derivative_gate_report.json`
00242 | 
00243 | The valid derivative reference is finite differences of SIESTA Hamiltonians:
00244 | 
00245 | ```text
00246 | (H_SIESTA(R + delta) - H_SIESTA(R - delta)) / (2 * delta)
00247 | ```
00248 | 
00249 | No force-constants comparison is implemented. SIESTA force constants, phonons,
00250 | dynamical matrices, and finite differences of forces are not treated as
00251 | `dH/dR`.
00252 | 
00253 | For presentation, explain the derivative gate report as a scientific status
00254 | summary for these optional diagnostics: it says what evidence is present,
00255 | whether the result should stay at `internal_diagnostic` or
00256 | `technical_presentation`, and why no derivative winner claim is allowed by
00257 | default.
00258 | 
00259 | The derivative plot payload can include dataset-size plots for the UI. Each
00260 | point is normally an aggregate over derivative metric rows/stencils for one
00261 | model and dataset size. `x_dataset_size` is preferably `N_train`; if split
00262 | metadata is unavailable, it falls back to `N_total`. Plot families include
00263 | primary dH errors, robust relative errors, correlation/residual diagnostics,
00264 | delta-conditioned trends, guarded axis/atom trends, and Hermiticity plus
00265 | onsite/offsite diagnostics versus dataset size. Treat all of them as
00266 | post-processing diagnostics, not paper-ready winner evidence.
00267 | 
00268 | For the paper-ready control plane, use the synthetic staged workflow smoke:
00269 | 
00270 | ```bash
00271 | python3 Comparison/scripts/g2m_deeph_smoke.py \
00272 |   --paper-workflow-dry-run \
00273 |   --output-dir Comparison/results/g2m_deeph_paper_workflow_smoke
00274 | ```
00275 | 
00276 | This command creates a minimal synthetic joint dataset fixture and exercises the
00277 | final workflow wiring: protocol validation, dataset verification, search-plan
00278 | generation, dry-run search payload, validation-only top-k, final-seed plan,
00279 | synthetic final statistics, diagnostic report, gate checker, and release
00280 | manifest. It writes `smoke_summary.json` with
00281 | `scientific_status: not_a_scientific_run`. A successful control-plane smoke is
00282 | not benchmark evidence; it deliberately keeps robust claims blocked unless real
00283 | dataset manifests, telemetry, final statistics, and DeepH equivalence evidence
00284 | are supplied.
00285 | 
00286 | ## Running A Serious Benchmark
00287 | 
00288 | For a real comparison:
00289 | 
00290 | 1. Generate a new joint dataset with the MD generation path or reuse a dataset
00291 |    that already passed the joint artifact contract. The generation path must
00292 |    use the expanded SIESTA store list and must validate artifacts before
00293 |    splits/training:
00294 | 
00295 |    ```bash
00296 |    PIPELINE_CONFIG_PATH=MD/pipeline_config.yaml \
00297 |      python3 MD/scripts/generate_md_dataset.py
00298 |    ```
00299 | 
00300 |    For production, use a copied or experiment-specific config rather than
00301 |    editing global defaults during an active run.
00302 | 
00303 | 2. Store reusable joint datasets under `Comparison/datasets/`. The default
00304 |    graphene root used by the UI/backend is
00305 |    `Comparison/datasets/graphene_w90_joint`.
00306 | 
00307 | 3. Confirm these files exist at the dataset root:
00308 | 
00309 |    ```text
00310 |    artifact_validation.json
00311 |    benchmark_dataset_manifest.json
00312 |    frozen_split_manifest.json
00313 |    ```
00314 | 
00315 | 4. Start the UI and open `G2M vs DeepH`.
00316 | 
00317 | ## Paper-Ready Protocol Contract
00318 | 
00319 | Exploratory sweeps remain useful for development, but the final/publicable
00320 | Graph2Mat-vs-DeepH comparison should be preregistered before looking at test
00321 | metrics. The machine-readable protocol contract is:
00322 | 
00323 | ```text
00324 | graph2mat_deeph_benchmark_protocol_v1
00325 | ```
00326 | 
00327 | Validate an example protocol with:
00328 | 
00329 | ```bash
00330 | python3 Comparison/scripts/g2m_deeph_protocol.py \
00331 |   Comparison/config/g2m_deeph_paper_protocol_v1_example.json
00332 | ```
00333 | 
00334 | Before generating a search plan, verify every dataset declared by the protocol.
00335 | This command reads existing manifests only; it does not run SIESTA or repair
00336 | artifacts:
00337 | 
00338 | ```bash
00339 | python3 Comparison/scripts/g2m_deeph_verify_protocol_datasets.py \
00340 |   --protocol Comparison/config/g2m_deeph_paper_protocol_v1_example.json \
00341 |   --output Comparison/results/g2m_deeph_dataset_verification.json \
00342 |   --strict
00343 | ```
00344 | 
00345 | The verifier fails closed if `dataset_root`, `artifact_validation.json`,
00346 | `benchmark_dataset_manifest.json`, `frozen_split_manifest.json`, strict
00347 | SIESTA/environment provenance, non-empty train/validation/test splits, split
00348 | hash links, or forbidden-reference checks do not pass. To finalize manifests
00349 | from an existing `split_root`, use `--write-manifests` explicitly; without that
00350 | flag the verifier never creates or freezes dataset manifests.
00351 | 
00352 | The staged final workflow can also be driven from one CLI. The workflow root is
00353 | an artifact directory; every stage writes a `stages/<stage>.json` manifest and
00354 | refuses to continue when required previous artifacts are missing.
00355 | 
00356 | ```bash
00357 | WORKFLOW_ROOT=Comparison/results/g2m_deeph_final_workflow
00358 | PROTOCOL=Comparison/config/g2m_deeph_paper_protocol_v1_example.json
00359 | 
00360 | python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00361 |   --stage validate-protocol \
00362 |   --protocol "$PROTOCOL" \
00363 |   --workflow-root "$WORKFLOW_ROOT" \
00364 |   --verify-datasets
00365 | 
00366 | python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00367 |   --stage generate-search-plan \
00368 |   --workflow-root "$WORKFLOW_ROOT"
00369 | 
00370 | python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00371 |   --stage run-search \
00372 |   --workflow-root "$WORKFLOW_ROOT" \
00373 |   --dataset-id <dataset-id>
00374 | 
00375 | python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00376 |   --stage select-top-k \
00377 |   --workflow-root "$WORKFLOW_ROOT"
00378 | 
00379 | python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00380 |   --stage generate-final-seeds \
00381 |   --workflow-root "$WORKFLOW_ROOT"
00382 | 
00383 | python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00384 |   --stage run-final \
00385 |   --workflow-root "$WORKFLOW_ROOT" \
00386 |   --dataset-id <dataset-id>
00387 | 
00388 | python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00389 |   --stage evaluate-final-test \
00390 |   --workflow-root "$WORKFLOW_ROOT" \
00391 |   --final-run-root "$WORKFLOW_ROOT/runs/<final-run-id>"
00392 | 
00393 | python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00394 |   --stage generate-report \
00395 |   --workflow-root "$WORKFLOW_ROOT" \
00396 |   --report-run-root "$WORKFLOW_ROOT/runs/<final-run-id>"
00397 | ```
00398 | 
00399 | Before any robust DeepH claim, generate numerical raw/global equivalence
00400 | evidence from the frozen samples. The preflight compares DeepH processed
00401 | `hamiltonians.h5`/`overlaps.h5` against the raw SIESTA/Graph2Mat HSX/TSHS
00402 | reference and writes adapter-discoverable
00403 | `raw_global_equivalence_evidence.json` files:
00404 | 
00405 | ```bash
00406 | python3 Comparison/scripts/deeph_raw_global_equivalence_preflight.py \
00407 |   --frozen-split-manifest Comparison/datasets/<frozen-dataset-id>/frozen_split_manifest.json \
00408 |   --graph2mat-result-dir "$WORKFLOW_ROOT/runs/<final-run-id>" \
00409 |   --deeph-processed-dir "$WORKFLOW_ROOT/runs/<final-run-id>/deeph/processed" \
00410 |   --deeph-predictions-dir "$WORKFLOW_ROOT/runs/<final-run-id>/deeph/inference" \
00411 |   --sample-limit 5 \
00412 |   --output-dir "$WORKFLOW_ROOT/deeph_raw_global_equivalence"
00413 | ```
00414 | 
00415 | If any required file or numerical check is missing, the evidence status is
00416 | `failed` and the DeepH adapter remains diagnostic-only. Do not hand-write this
00417 | evidence JSON; it must come from the preflight command and be included in the
00418 | release/evidence bundle.
00419 | 
00420 | After final statistics are available, run the fail-closed gate checker before
00421 | generating any claim-bearing final report or making any robust claim:
00422 | 
00423 | ```bash
00424 | python3 Comparison/scripts/g2m_deeph_gate_check.py \
00425 |   --protocol "$PROTOCOL" \
00426 |   --workflow-root "$WORKFLOW_ROOT" \
00427 |   --run-root "$WORKFLOW_ROOT/runs/<final-run-id>" \
00428 |   --output "$WORKFLOW_ROOT/gate_status.json"
00429 | ```
00430 | 
00431 | Only `claim_status=robust_allowed` and `robust_claim_allowed=true` allow a
00432 | paper-style winner claim. Any `invalid_*` or `diagnostic_only` status means the
00433 | run can be reported only as blocked/diagnostic, with the listed
00434 | `required_next_actions` used as the remediation checklist.
00435 | 
00436 | The repository intentionally keeps large generated matrices, predictions,
00437 | telemetry, and result directories out of git. Before archiving or sharing a
00438 | paper-ready run, write a release manifest that hashes every required external
00439 | file and flags missing or forbidden evidence:
00440 | 
00441 | ```bash
00442 | python3 Comparison/scripts/g2m_deeph_release_manifest.py \
00443 |   --dataset-root Comparison/datasets/<frozen-dataset-id> \
00444 |   --run-root "$WORKFLOW_ROOT/runs/<final-run-id>" \
00445 |   --workflow-root "$WORKFLOW_ROOT" \
00446 |   --output "$WORKFLOW_ROOT/artifact_release_manifest.json" \
00447 |   --strict
00448 | ```
00449 | 
00450 | In strict mode the manifest exits nonzero unless all required dataset manifests,
00451 | raw SIESTA artifacts, run telemetry, equivalence/adapter manifests, final
00452 | statistics, final report, and workflow evidence are present and hashable.
00453 | `ML_prediction.HSX` is allowed only as a Graph2Mat prediction artifact; if it
00454 | appears in any reference role, the release manifest is `invalid`.
00455 | Symlinks are hashable only when their resolved target remains under one of the
00456 | explicit roots passed on the command line; external symlinks are recorded as
00457 | unsafe missing evidence.
00458 | 
00459 | Use `--dry-run` with `run-search` or `run-final` to materialize the exact runner
00460 | payload without launching SIESTA, Graph2Mat, or DeepH. Search stages are
00461 | test-blind; `select-top-k` fails closed if search inputs contain test metrics.
00462 | If the protocol declares more than one dataset, runner stages require
00463 | `--dataset-id`; the workflow writes `selected_dataset_id`,
00464 | `executed_dataset_ids`, and all protocol dataset IDs into the stage manifests
00465 | so no dataset can be silently reduced to `datasets[0]`.
00466 | 
00467 | The protocol records:
00468 | 
00469 | ```text
00470 | protocol_id and version
00471 | datasets and frozen split manifests
00472 | joint SIESTA reference artifact requirements
00473 | Graph2Mat and DeepH model-specific search spaces
00474 | validation-only selection metric
00475 | common early stopping policy
00476 | search and budget policy
00477 | final seed list
00478 | top-k rule based only on validation metrics
00479 | locked final test policy
00480 | required telemetry fields
00481 | DeepH equivalence fail-closed policy
00482 | ```
00483 | 
00484 | This contract intentionally does not force identical learning rates or batch
00485 | sizes. Those values are model-specific because Graph2Mat and DeepH have
00486 | different architectures and different memory/throughput behavior. Fairness is
00487 | expressed through shared datasets, splits, references, metrics, validation
00488 | selection, hardware class, telemetry, and either equal trials or equal
00489 | GPU-hour budgets.
00490 | 
00491 | Protocol search spaces can be expanded as the old finite grid, or sampled with
00492 | deterministic random/Latin-hypercube search. Sampled spaces support:
00493 | 
00494 | ```text
00495 | {"choices": [...]}
00496 | {"value": ...} or {"fixed": ...}
00497 | {"type": "int", "min": ..., "max": ...}
00498 | {"distribution": "uniform", "min": ..., "max": ...}
00499 | {"distribution": "loguniform", "min": ..., "max": ...}
00500 | ```
00501 | 
00502 | Graph2Mat supports a benchmark-level `readout` search key with two verified
00503 | families:
00504 | 
00505 | ```text
00506 | default
00507 | edge_node_mix
00508 | ```
00509 | 
00510 | `edge_node_mix` is expanded before execution to the Graph2Mat model keys
00511 | `node_block_readout`, `edge_block_readout`, `preprocessing_edges`, and
00512 | `preprocessing_edges_reuse_nodes`. The mapped classes are the existing
00513 | Graph2Mat e3nn readout/preprocessing classes, so no unknown model argument is
00514 | sent downstream. Configs that omit `readout` keep the historical Graph2Mat
00515 | defaults.
00516 | 
00517 | The runner writes the preregistered plan before launching jobs:
00518 | 
00519 | ```text
00520 | sweep/search_plan.json
00521 | ```
00522 | 
00523 | For sampled search this artifact records the strategy, random seed, protocol
00524 | id/hash, sampled dimensions, every planned config, and any duplicate sampled
00525 | configs produced by a tiny search space.
00526 | 
00527 | Search budget accounting is written to:
00528 | 
00529 | ```text
00530 | sweep/budget_summary.json
00531 | ```
00532 | 
00533 | For `equal_n_trials`, the scheduler reserves at most the configured number of
00534 | trials per model and reports actual GPU-hours consumed. For
00535 | `equal_gpu_hours_per_model`, the scheduler follows the preregistered search-plan
00536 | order and starts new trials only while the model has remaining consumed
00537 | GPU-hour budget. Because true trial cost is known only after completion, the
00538 | last scheduled trial, or scheduler batch when parallelism is enabled, may
00539 | overshoot the exact budget; the overshoot and exhaustion reason are reported.
00540 | Completed trials with missing `gpu_hours_total` make budget accounting fail
00541 | closed rather than being treated as zero cost.
00542 | 
00543 | The final test split is locked during search. Test metrics must not be used for
00544 | checkpoint selection, top-k selection, search-space refinement, or early
00545 | stopping. If DeepH prediction equivalence is not proven by the adapter, robust
00546 | winner claims remain fail-closed and DeepH-derived comparisons stay
00547 | diagnostic-only.
00548 | 
00549 | ## Cost Telemetry
00550 | 
00551 | Training and evaluation runs write optional versioned cost telemetry:
00552 | 
00553 | ```text
00554 | graph2mat_deeph_cost_telemetry_v1
00555 | ```
00556 | 
00557 | For each completed Graph2Mat or DeepH config the runner writes:
00558 | 
00559 | ```text
00560 | telemetry/graph2mat.json
00561 | telemetry/deeph.json
00562 | ```
00563 | 
00564 | Training-sweep records also embed the same telemetry summary. The fields include
00565 | total wall-clock time, per-phase wall-clock time, GPU-hours when process-level
00566 | GPU activity can be observed, peak GPU memory, samples/s, matrix-blocks/s when
00567 | available, epochs trained, best validation epoch/value, hardware metadata, and
00568 | warnings for unavailable fields.
00569 | 
00570 | Telemetry is fail-transparent. CPU runs, missing `nvidia-smi`, missing
00571 | TensorBoard validation events, or unavailable matrix-block counts do not crash
00572 | the benchmark. They mark the telemetry as `partial` or `unavailable` and explain
00573 | which fields could not be collected. The runner never fabricates GPU-hours,
00574 | memory, or throughput values.
00575 | 
00576 | ## Common Early Stopping
00577 | 
00578 | Production payloads can define a shared validation-only stopping policy:
00579 | 
00580 | ```json
00581 | {
00582 |   "early_stopping": {
00583 |     "metric": "val_loss",
00584 |     "mode": "min",
00585 |     "patience": 30,
00586 |     "min_delta": 0.0,
00587 |     "max_epochs": 600
00588 |   }
00589 | }
00590 | ```
00591 | 
00592 | When this policy is present, Graph2Mat receives a Lightning `EarlyStopping`
00593 | callback monitoring the configured validation metric and uses the same
00594 | `max_epochs`. DeepH does not expose a patience/min-delta INI API, so the runner
00595 | monitors DeepH's per-epoch validation-loss log lines and terminates the DeepH
00596 | process only after an epoch has completed and the common patience rule is
00597 | exhausted. DeepH's native threshold-based stopping is neutralized in this mode
00598 | so the benchmark policy is the active stopping rule.
00599 | 
00600 | Every completed run records early-stopping/checkpoint-selection metadata:
00601 | 
00602 | ```text
00603 | validation_metric_name
00604 | metric_mode
00605 | patience
00606 | min_delta
00607 | max_epochs
00608 | best_epoch
00609 | best_validation_value
00610 | epochs_trained
00611 | stop_reason
00612 | ```
00613 | 
00614 | If the configured validation metric is missing or non-finite, the run fails
00615 | closed. Test metrics are not accepted for early stopping or checkpoint
00616 | selection.
00617 | 
00618 | 5. Use `Reuse existing validated joint dataset` unless you are intentionally
00619 |    generating a new dataset through an approved one-pass path.
00620 | 
00621 | 6. Click `Validate dataset artifacts`.
00622 | 
00623 | 7. Set Graph2Mat and DeepH training settings.
00624 | 
00625 | 8. Click `Run Graph2Mat vs DeepH benchmark`.
00626 | 
00627 | 9. Watch the phase progress, streaming logs, artifact table, timing table,
00628 |    metric summary and plots in the same tab.
00629 | 
00630 | The runner phases are:
00631 | 
00632 | ```text
00633 | validate_inputs
00634 | generate_or_validate_joint_dataset
00635 | freeze_splits
00636 | graph2mat_train
00637 | graph2mat_predict
00638 | deeph_preprocess
00639 | deeph_train
00640 | deeph_predict
00641 | common_metrics
00642 | plots_and_summary
00643 | complete
00644 | ```
00645 | 
00646 | ## Interpreting Comparability Status
00647 | 
00648 | The final summary and plots expose a scientific status. A winner is allowed
00649 | only when the status is scientifically valid and no severe warning blocks the
00650 | recommendation.
00651 | 
00652 | Important status values:
00653 | 
00654 | | Status | Interpretation |
00655 | | --- | --- |
00656 | | `valid_joint_one_pass_dataset` | Preferred: artifacts were generated in one pass and checks passed. |
00657 | | `valid_reused_joint_dataset` | Existing dataset passed the joint contract and compatibility checks. |
00658 | | `valid_repaired_dataset_with_warning` | Repaired data may be useful, but provenance must be inspected carefully. |
00659 | | `invalid_missing_artifacts` | Missing required artifacts; do not compare. |
00660 | | `invalid_incompatible_splits` | Graph2Mat and DeepH did not use identical sample IDs/splits. |
00661 | | `invalid_incompatible_basis_or_pseudos` | Basis/material/pseudopotential provenance is incompatible or unknown. |
00662 | | `invalid_prediction_format` | Prediction outputs could not be compared safely. |
00663 | | `diagnostic_only` | Metrics may help debugging, but no robust winner is declared. |
00664 | 
00665 | If status is `diagnostic_only` or starts with `invalid_`, the UI must say
00666 | `No robust winner` even if one method has a lower numeric error.
00667 | 
00668 | ## DeepH Equivalence Gate
00669 | 
00670 | DeepH prediction artifacts now carry a formal equivalence audit in addition to
00671 | the adapter-specific status:
00672 | 
00673 | | Field | Meaning |
00674 | | --- | --- |
00675 | | `equivalence_status` | One of `proven`, `failed`, `unproven`, or `not_applicable`. |
00676 | | `equivalence_scope` | The representation whose convention was checked, such as `raw_global` or `deeph_processed_blockwise_global_hdf5`. |
00677 | | `equivalence_evidence_paths` | Files used as evidence for the audit. |
00678 | | `equivalence_gate.robust_claim_allowed` | Whether DeepH can participate in robust winner claims. |
00679 | 
00680 | Only `equivalence_status=proven` with raw/global Hamiltonian equivalence allows
00681 | robust matrix-metric claims. Current DeepH HDF5 outputs may still appear in
00682 | diagnostic tables, but if raw/global HSX units, orbital order, R-vector
00683 | convention, and support semantics are not proven, ranking and final statistics
00684 | must remain fail-closed.
00685 | 
00686 | ## Ground Truth And Overlap Policy
00687 | 
00688 | Ground truth always comes from SIESTA reference artifacts:
00689 | 
00690 | ```text
00691 | SystemLabel.HSX
00692 | SystemLabel.TSHS
00693 | SystemLabel.STRUCT_OUT
00694 | SystemLabel.XV
00695 | SystemLabel.ORB_INDX
00696 | ```
00697 | 
00698 | Never use `ML_prediction.HSX` or any DeepH/Graph2Mat prediction as ground
00699 | truth.
00700 | 
00701 | For non-orthogonal SIESTA references, official spectral and DOS metrics use
00702 | the SIESTA reference overlap, `S_ref` or `S_ref(k)`, for predicted spectra
00703 | whenever it is available. A prediction-owned overlap is used only if it is
00704 | explicitly validated within the evaluator tolerance. Unsafe predicted HSX files
00705 | are marked by fields such as:
00706 | 
00707 | ```text
00708 | prediction_self_contained_hsx_safe
00709 | prediction_own_overlap_used
00710 | prediction_overlap_relative_frobenius_vs_reference
00711 | overlap_source
00712 | graph2mat_auxiliary_component_ignored
00713 | ```
00714 | 
00715 | ## What Remains Diagnostic-Only
00716 | 
00717 | Some outputs are intentionally diagnostic until additional physical equivalence
00718 | is validated:
00719 | 
00720 | - DeepH HDF5 predictions adapted through a local reader/converter are
00721 |   diagnostic-only if units, orbital order, R-vector convention, basis or frame
00722 |   equivalence is not fully proven.
00723 | - Repository raw/global Hamiltonian metrics are not exact DeepH local-frame
00724 |   H-prime metrics.
00725 | - High-symmetry k-path band comparisons are not implied by Monkhorst-Pack
00726 |   k-grid metrics.
00727 | - Unsupported spin-orbit, spinful, ambiguous multi-component or incompatible
00728 |   non-orthogonal cases must fail or become diagnostic-only.
00729 | - Missing Fermi level means near-Fermi and fixed-window DOS metrics are
00730 |   unavailable; the evaluator does not invent a substitute Fermi level.
00731 | 
00732 | The benchmark should be presented as:
00733 | 
00734 | ```text
00735 | Controlled cross-architecture benchmark for learned Hamiltonians under a
00736 | unified SIESTA/W90 pipeline.
00737 | ```
00738 | 
00739 | not as an exact reproduction of the DeepH paper.
00740 | 
00741 | ## Paper-Ready Reports
00742 | 
00743 | After a search or final benchmark run, generate machine-readable learning-curve
00744 | and accuracy/cost summaries with:
00745 | 
00746 | ```bash
00747 | python3 Comparison/scripts/g2m_deeph_report.py \
00748 |   Comparison/results/<benchmark_run_id> \
00749 |   --metric val_loss \
00750 |   --mode min
00751 | ```
00752 | 
00753 | The report is written under `summary/report/` and includes:
00754 | 
00755 | - `learning_curve.csv/json`: validation error vs epoch, cumulative wall-clock,
00756 |   cumulative GPU-hours, samples seen and matrix blocks seen when available.
00757 | - `best_validation_summary.csv/json`: best validation epoch/value, GPU-hours
00758 |   to best validation, wall-clock to best validation and peak GPU memory.
00759 | - `pareto_accuracy_cost.csv/json`: accuracy/cost Pareto table with explicit
00760 |   dominated/frontier status.
00761 | - `final_comparison.json`: exploratory accuracy, compute and practical Pareto
00762 |   diagnostics from available run artifacts.
00763 | - `final_report.json/md`: fail-closed final claim report. It declares no
00764 |   winner unless both `final_statistics` and `gate_status` allow robust claims.
00765 | 
00766 | Search-stage reports ignore test metrics. If telemetry or per-epoch curves are
00767 | missing in older artifacts, the rows remain readable and the missing fields are
00768 | reported through explicit warning/status fields instead of being silently filled.
00769 | 
00770 | For final multi-seed test aggregation and winner gates, run:
00771 | 
00772 | ```bash
00773 | python3 Comparison/scripts/g2m_deeph_final_stats.py \
00774 |   Comparison/results/<benchmark_run_id> \
00775 |   --metric low_energy_rmse_eV \
00776 |   --mode min \
00777 |   --expected-seeds 0,1,2,3,4 \
00778 |   --min-final-seeds 3
00779 | ```
00780 | 
00781 | This writes `summary/final_statistics/` with seed mean/std/stderr, seed-level
00782 | confidence intervals, optional per-system bootstrap CIs, compute summaries, a
00783 | Pareto table and a fail-closed winner decision. Robust claims are blocked if
00784 | final seeds are incomplete, test metrics appear outside `final_test`, or DeepH
00785 | raw/global equivalence is not proven.
00786 | 
00787 | For a claim-bearing paper report, run the gate checker and pass both evidence
00788 | files to the report generator:
00789 | 
00790 | ```bash
00791 | python3 Comparison/scripts/g2m_deeph_gate_check.py \
00792 |   --protocol "$PROTOCOL" \
00793 |   --workflow-root "$WORKFLOW_ROOT" \
00794 |   --run-root Comparison/results/<benchmark_run_id> \
00795 |   --output "$WORKFLOW_ROOT/gate_status.json"
00796 | 
00797 | python3 Comparison/scripts/g2m_deeph_report.py \
00798 |   Comparison/results/<benchmark_run_id> \
00799 |   --metric low_energy_rmse_eV \
00800 |   --mode min \
00801 |   --final-statistics "$WORKFLOW_ROOT/final_test/final_statistics.json" \
00802 |   --gate-status "$WORKFLOW_ROOT/gate_status.json"
00803 | ```
00804 | 
00805 | If `gate_status.json` is missing or blocks the claim, `final_report.json/md`
00806 | must report `robust_claim_allowed=false` and leave `accuracy_winner`,
00807 | `cost_winner`, and `pareto_winner` empty. H-MAE/common-summary recommendations
00808 | remain supporting Hamiltonian diagnostics unless H-MAE is explicitly
00809 | preregistered as `final_evaluation.primary_metric`; they cannot override the
00810 | gate checker.
00811 | 
00812 | ## Paper-Ready Reviewer Runbook
00813 | 
00814 | This is the strict path for a defensible Graph2Mat-vs-DeepH comparison. It is
00815 | separate from exploratory UI runs and smoke tests. Do not use DeepH paper
00816 | numbers as a direct baseline for this workflow; they are external context only
00817 | unless the same frozen SIESTA dataset, splits, references, metrics and hardware
00818 | contract are used.
00819 | 
00820 | Set the common paths once:
00821 | 
00822 | ```bash
00823 | PROTOCOL=Comparison/config/g2m_deeph_paper_protocol_v1_example.json
00824 | WORKFLOW_ROOT=Comparison/results/g2m_deeph_final_workflow
00825 | DATASET_ROOT=Comparison/datasets/<frozen-dataset-id>
00826 | FINAL_RUN_ROOT="$WORKFLOW_ROOT/runs/<final-run-id>"
00827 | ```
00828 | 
00829 | 1. **Required external artifacts**
00830 | 
00831 |    A paper-ready dataset and run need the external files that are intentionally
00832 |    not stored in git: SIESTA raw matrices, split manifests, model predictions,
00833 |    telemetry, equivalence evidence, final statistics and reports. The required
00834 |    per-snapshot SIESTA files are `RUN.fdf`, `RUN.out` or `siesta.out`,
00835 |    `SystemLabel.TSHS`, `SystemLabel.TSDE`, `SystemLabel.HSX`,
00836 |    `SystemLabel.STRUCT_OUT`, `SystemLabel.XV`, `SystemLabel.ORB_INDX`, and
00837 |    `metadata.json`. `ML_prediction.HSX` is never a reference file.
00838 |    `ML_prediction.HSX` is never a reference in metric staging, manifests or
00839 |    release evidence.
00840 | 
00841 | 2. **Dataset freeze/verify**
00842 | 
00843 |    ```bash
00844 |    python3 Comparison/scripts/g2m_deeph_verify_protocol_datasets.py \
00845 |      --protocol "$PROTOCOL" \
00846 |      --output "$WORKFLOW_ROOT/dataset_verification.json" \
00847 |      --strict
00848 |    ```
00849 | 
00850 |    This must pass before search. It verifies dataset roots, manifests, strict
00851 |    SIESTA/environment provenance, non-empty train/validation/test splits,
00852 |    split-hash linkage and forbidden-reference checks.
00853 | 
00854 | 3. **Protocol validation**
00855 | 
00856 |    ```bash
00857 |    python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00858 |      --stage validate-protocol \
00859 |      --protocol "$PROTOCOL" \
00860 |      --workflow-root "$WORKFLOW_ROOT" \
00861 |      --verify-datasets
00862 |    ```
00863 | 
00864 |    The protocol must include `final_evaluation.primary_metric`; validation
00865 |    metrics such as `val_loss` are allowed for selection but not for final
00866 |    scientific claims.
00867 | 
00868 | 4. **Dataset release manifest**
00869 | 
00870 |    ```bash
00871 |    python3 Comparison/scripts/g2m_deeph_release_manifest.py \
00872 |      --dataset-root "$DATASET_ROOT" \
00873 |      --output "$WORKFLOW_ROOT/dataset_release_manifest.json" \
00874 |      --strict
00875 |    ```
00876 | 
00877 |    Run this early to confirm that external dataset evidence is hashable. Run it
00878 |    again with `--run-root "$FINAL_RUN_ROOT"` and
00879 |    `--workflow-root "$WORKFLOW_ROOT"` after final evaluation to capture
00880 |    predictions, telemetry, final stats and reports.
00881 | 
00882 | 5. **DeepH raw/global equivalence preflight**
00883 | 
00884 |    ```bash
00885 |    python3 Comparison/scripts/deeph_raw_global_equivalence_preflight.py \
00886 |      --frozen-split-manifest "$DATASET_ROOT/frozen_split_manifest.json" \
00887 |      --graph2mat-result-dir "$FINAL_RUN_ROOT" \
00888 |      --deeph-processed-dir "$FINAL_RUN_ROOT/deeph/processed" \
00889 |      --deeph-predictions-dir "$FINAL_RUN_ROOT/deeph/inference" \
00890 |      --sample-limit 5 \
00891 |      --output-dir "$WORKFLOW_ROOT/deeph_raw_global_equivalence" \
00892 |      --fail-closed
00893 |    ```
00894 | 
00895 |    Missing `raw_global_equivalence_evidence.json`, failed shape/unit/orbital
00896 |    order/R-vector/spin/support/H(k)/S_ref/eigenvalue checks, or hand-written
00897 |    evidence keeps DeepH diagnostic-only.
00898 | 
00899 | 6. **Search plan generation**
00900 | 
00901 |    ```bash
00902 |    python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00903 |      --stage generate-search-plan \
00904 |      --workflow-root "$WORKFLOW_ROOT"
00905 |    ```
00906 | 
00907 |    This writes the preregistered search plan before jobs run. The search space
00908 |    may be grid, deterministic random or Latin hypercube according to the
00909 |    validated protocol.
00910 | 
00911 | 7. **Test-blind search**
00912 | 
00913 |    ```bash
00914 |    python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00915 |      --stage run-search \
00916 |      --workflow-root "$WORKFLOW_ROOT" \
00917 |      --dataset-id <dataset-id>
00918 |    ```
00919 | 
00920 |    Use `--dry-run` first to inspect the exact runner payload. Search artifacts
00921 |    must not contain test metrics, and search/checkpoint choices must use only
00922 |    validation metrics.
00923 | 
00924 | 8. **Validation-only top-k**
00925 | 
00926 |    ```bash
00927 |    python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00928 |      --stage select-top-k \
00929 |      --workflow-root "$WORKFLOW_ROOT"
00930 |    ```
00931 | 
00932 |    Top-k uses the protocol validation metric and fails closed if test metrics
00933 |    appear in search inputs.
00934 | 
00935 | 9. **Final multi-seed plan**
00936 | 
00937 |    ```bash
00938 |    python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00939 |      --stage generate-final-seeds \
00940 |      --workflow-root "$WORKFLOW_ROOT"
00941 |    ```
00942 | 
00943 |    The output must preserve selected hyperparameters exactly and expand only
00944 |    seed/run metadata. Final robust claims require the configured seed count.
00945 | 
00946 | 10. **Final training and locked test evaluation**
00947 | 
00948 |     ```bash
00949 |     python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00950 |       --stage run-final \
00951 |       --workflow-root "$WORKFLOW_ROOT" \
00952 |       --dataset-id <dataset-id>
00953 | 
00954 |     python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00955 |       --stage evaluate-final-test \
00956 |       --workflow-root "$WORKFLOW_ROOT" \
00957 |       --final-run-root "$FINAL_RUN_ROOT"
00958 |     ```
00959 | 
00960 |     Test metrics are produced only in the final stage and are interpreted using
00961 |     `final_evaluation.primary_metric`, not `selection.metric`.
00962 | 
00963 | 11. **Final statistics and Pareto**
00964 | 
00965 |     The workflow stage above writes final statistics when run through
00966 |     `evaluate-final-test`. For standalone aggregation:
00967 | 
00968 |     ```bash
00969 |     python3 Comparison/scripts/g2m_deeph_final_stats.py \
00970 |       "$FINAL_RUN_ROOT" \
00971 |       --metric low_energy_rmse_eV \
00972 |       --mode min \
00973 |       --expected-seeds 0,1,2,3,4 \
00974 |       --min-final-seeds 3 \
00975 |       --output-dir "$WORKFLOW_ROOT/final_test"
00976 |     ```
00977 | 
00978 |     The final statistics report must include seed mean/std, uncertainty where
00979 |     computable, GPU-hours, peak memory, throughput and Pareto summaries.
00980 | 
00981 | 12. **Gate checker**
00982 | 
00983 |     ```bash
00984 |     python3 Comparison/scripts/g2m_deeph_gate_check.py \
00985 |       --protocol "$PROTOCOL" \
00986 |       --workflow-root "$WORKFLOW_ROOT" \
00987 |       --run-root "$FINAL_RUN_ROOT" \
00988 |       --output "$WORKFLOW_ROOT/gate_status.json"
00989 |     ```
00990 | 
00991 |     This is the single reviewer-facing answer for robust claims. Anything other
00992 |     than `claim_status=robust_allowed` with `robust_claim_allowed=true` blocks a
00993 |     winner claim.
00994 | 
00995 | 13. **Final report and evidence bundle**
00996 | 
00997 |     ```bash
00998 |     python3 Comparison/scripts/g2m_deeph_final_workflow.py \
00999 |       --stage generate-report \
01000 |       --workflow-root "$WORKFLOW_ROOT" \
01001 |       --report-run-root "$FINAL_RUN_ROOT"
01002 | 
01003 |     python3 Comparison/scripts/g2m_deeph_report.py \
01004 |       "$FINAL_RUN_ROOT" \
01005 |       --metric low_energy_rmse_eV \
01006 |       --mode min \
01007 |       --final-statistics "$WORKFLOW_ROOT/final_test/final_statistics.json" \
01008 |       --gate-status "$WORKFLOW_ROOT/gate_status.json"
01009 |     ```
01010 | 
01011 |     The report may include diagnostic tables, but final winners come only from
01012 |     final statistics plus gate status. The evidence bundle and release manifest
01013 |     should be archived together.
01014 | 
01015 | ## Optional Hamiltonian Derivative Diagnostics
01016 | 
01017 | Derivative metrics are an optional postprocess and are disabled by default.
01018 | They compare finite differences of Hamiltonian matrices, not force constants,
01019 | phonon dynamical matrices, or finite differences of forces.
01020 | 
01021 | For this repository, `dH/dR` means the derivative of Hamiltonian matrix
01022 | elements with respect to Cartesian atomic displacement. The valid reference is
01023 | the finite-difference SIESTA Hamiltonian derivative:
01024 | 
01025 | ```text
01026 | (H_SIESTA(R + delta) - H_SIESTA(R - delta)) / (2 * delta)
01027 | ```
01028 | 
01029 | SIESTA force constants, `.FC` files, dynamical matrices, and phonons are not
01030 | valid `dH/dR` references here. They must not be substituted for finite
01031 | differences of SIESTA Hamiltonians.
01032 | 
01033 | ```yaml
01034 | derivative_metrics:
01035 |   enabled: false
01036 |   finite_difference_method: central
01037 |   split: test
01038 |   require_central: true
01039 |   diagnostic_only: true
01040 |   support_threshold: 1e-12
01041 | ```
01042 | 
01043 | When enabled, the runner calls
01044 | `Comparison/scripts/evaluate_hamiltonian_derivative_metrics.py` after common H
01045 | metrics. Derivative failures are recorded as diagnostic derivative manifests and
01046 | must not hide or rewrite existing H metric outputs. Aggregation preserves the
01047 | `graph2mat` and `deeph` method labels, adds derivative rows only when
01048 | `derivative_metrics/*.csv` outputs exist, and adds recommendation notes only as
01049 | diagnostics. Derivative metrics are not winner metrics and do not enable
01050 | paper-level claims. `derivative_delta_stability.json` only documents that a
01051 | delta sweep was run; it does not prove convergence unless the thresholds used
01052 | for the sweep are documented.
01053 | 
01054 | ### Required derivative artifacts
01055 | 
01056 | Derivative comparisons assume archived Hamiltonian artifacts already exist. The
01057 | minimum expected evidence is:
01058 | 
01059 | - `RUN.fdf`
01060 | - `metadata.json`
01061 | - SIESTA Hamiltonian reference such as `*.HSX` or `*.TSHS`
01062 | - predicted Hamiltonian `ML_prediction.HSX`
01063 | - explicit plus/minus displacement metadata
01064 | - `ORB_INDX` and basis/gauge evidence where available
01065 | 
01066 | For central finite differences, plus/minus pairing must be unambiguous. Missing
01067 | pairing, mismatched delta, mismatched units, inconsistent atom indexing, or
01068 | missing orbital-ordering evidence must keep the derivative result in a blocked
01069 | or diagnostic-only state.
01070 | 
01071 | ### Derivative CLI usage
01072 | 
01073 | The derivative evaluator consumes archived references and predictions; it does
01074 | not rerun SIESTA, Graph2Mat, or DeepH.
01075 | 
01076 | ```bash
01077 | python3 Comparison/scripts/evaluate_hamiltonian_derivative_metrics.py \
01078 |   <result_dir> \
01079 |   --method central \
01080 |   --split test \
01081 |   --require-central \
01082 |   --diagnostic-only \
01083 |   --support-threshold 1e-12 \
01084 |   --overwrite
01085 | ```
01086 | 
01087 | Implemented options include:
01088 | 
01089 | - `--method {central,forward,backward}`
01090 | - `--split {test,validation,train,all}`
01091 | - `--require-central`
01092 | - `--diagnostic-only`
01093 | - `--support-threshold <float>`
01094 | - `--max-stencils <int>`
01095 | - `--output-dir <path>`
01096 | - `--source-model {graph2mat,deeph}`
01097 | - `--overwrite`
01098 | 
01099 | Outputs are written under `derivative_metrics/`:
01100 | 
01101 | - `manifest.json`
01102 | - `stencil_status.csv`
01103 | - `derivative_matrix_metrics.csv`
01104 | - `derivative_support_sweep.csv`
01105 | - `derivative_hermiticity.csv`
01106 | - `derivative_summary.json`
01107 | 
01108 | The fail-closed derivative gate checker reads those outputs and classifies what
01109 | can honestly be claimed:
01110 | 
01111 | ```bash
01112 | python3 Comparison/scripts/g2m_deeph_derivative_gate_check.py \
01113 |   --derivative-root <result_dir>/derivative_metrics \
01114 |   --output <result_dir>/derivative_metrics/derivative_gate_report.json
01115 | ```
01116 | 
01117 | If a benchmark run already staged both methods, the same checker can discover
01118 | the two derivative roots from the run directory:
01119 | 
01120 | ```bash
01121 | python3 Comparison/scripts/g2m_deeph_derivative_gate_check.py \
01122 |   --run-root <benchmark_run_root> \
01123 |   --output <benchmark_run_root>/common_metrics/summary/derivative_gate_report.json
01124 | ```
01125 | 
01126 | ### Derivative metrics and units
01127 | 
01128 | Derivative metrics are reported in `eV/Ang` and remain diagnostic-only by
01129 | default. The current derivative outputs include:
01130 | 
01131 | - derivative MAE and RMSE on reference, predicted, and union supports
01132 | - relative Frobenius, relative L1, and cosine diagnostics
01133 | - support precision, recall, F1, false-zero, and false-nonzero diagnostics
01134 | - Hermiticity diagnostics for reference and predicted derivative matrices
01135 | 
01136 | Rows also carry derivative metadata such as `atom_index_zero_based`, `axis`,
01137 | `delta_ang`, `finite_difference_method`, `derivative_units`, and
01138 | `comparison_status`.
01139 | 
01140 | ### Derivative scientific gates
01141 | 
01142 | The derivative gate checker emits one of four statuses:
01143 | 
01144 | - `internal_diagnostic`
01145 | - `technical_presentation`
01146 | - `paper_level_candidate`
01147 | - `blocked`
01148 | 
01149 | Fail-closed blockers include:
01150 | 
01151 | - `force_constants_used=true`
01152 | - `reference_definition != siesta_hamiltonian_finite_difference`
01153 | - no central stencils
01154 | - missing plus/minus pairing
01155 | - mismatched shapes
01156 | - mismatched `delta_ang`
01157 | - mismatched units
01158 | - missing or inconsistent atom indexing
01159 | - missing or inconsistent orbital ordering metadata
01160 | - high Hermiticity defect
01161 | - support pattern discontinuity above threshold
01162 | 
01163 | Paper-level candidate status is additionally blocked without:
01164 | 
01165 | - documented delta thresholds
01166 | - basis/gauge evidence
01167 | - orbital-ordering evidence
01168 | - independent dataset/split metadata
01169 | - split leakage audit
01170 | - proven Graph2Mat/DeepH equivalence when both methods are compared
01171 | 
01172 | ### Derivative limitations
01173 | 
01174 | Current derivative comparisons must be interpreted conservatively:
01175 | 
01176 | - gauge, basis, and orbital ordering may still be ambiguous across tools
01177 | - neighbor-list or sparsity discontinuities can dominate derivative errors
01178 | - delta sensitivity can change rankings or invalidate naive comparisons
01179 | - ML prediction noise may be amplified by finite differences
01180 | - no force-constants comparison is implemented or scientifically accepted here
01181 | 
01182 | ## Claim Checklist
01183 | 
01184 | | Gate | Evidence file or field | Robust claim requirement |
01185 | | --- | --- | --- |
01186 | | Dataset contract | `artifact_validation.json`, `benchmark_dataset_manifest.json` | dataset is `benchmark_ready` and all required SIESTA artifacts exist |
01187 | | Frozen splits | `frozen_split_manifest.json` | train/validation/test are non-empty and split hashes match |
01188 | | References | release manifest, metric manifests | no `ML_prediction.HSX` or model prediction is used as reference |
01189 | | Selection metric | protocol `selection` and top-k artifacts | validation-only selection; no test metrics in search/top-k |
01190 | | Final metric | protocol `final_evaluation.primary_metric` | final claims use preregistered spectral/DOS/Hamiltonian metric, not `val_loss` |
01191 | | DeepH equivalence | `raw_global_equivalence_evidence.json`, adapter manifest | raw/global equivalence is proven, otherwise DeepH is diagnostic-only |
01192 | | Telemetry | per-run telemetry and final stats | GPU-hours, peak memory and throughput are present for cost claims |
01193 | | Seeds/statistics | `final_statistics.json` | configured final seeds complete and uncertainty is reported honestly |
01194 | | Gate status | `gate_status.json` | `robust_claim_allowed=true` and `claim_status=robust_allowed` |
01195 | | Release bundle | `artifact_release_manifest.json` | all required external evidence is hashable and complete |
01196 | 
01197 | ## Allowed And Forbidden Claims
01198 | 
01199 | Allowed before gates pass:
01200 | 
01201 | - The repository can run the staged benchmark control plane.
01202 | - A run is exploratory, diagnostic-only or blocked, with listed reasons.
01203 | - H-MAE, spectral, DOS, timing and telemetry tables can be inspected as
01204 |   diagnostics if their provenance is clear.
01205 | 
01206 | Allowed only after all gates pass:
01207 | 
01208 | - Graph2Mat/DeepH accuracy winner for the preregistered final metric.
01209 | - Compute winner for a configured accuracy threshold.
01210 | - Practical/Pareto winner from non-dominated accuracy-cost fronts.
01211 | 
01212 | Forbidden:
01213 | 
01214 | - Direct winner claims from DeepH paper numbers.
01215 | - Spectral superiority from H-MAE alone unless H-MAE was preregistered as the
01216 |   final metric.
01217 | - Any claim using test metrics for search, top-k, early stopping or checkpoint
01218 |   selection.
01219 | - Cost-efficiency claims when GPU-hours, memory or throughput telemetry is
01220 |   missing.
01221 | - Robust DeepH claims when `raw_global_equivalence_evidence.json` is missing,
01222 |   unproven or failed.
01223 | - Treating smoke output as scientific benchmark evidence.
01224 | 
01225 | ## Diagnostic-Only Examples
01226 | 
01227 | The final report must remain diagnostic-only in these common cases:
01228 | 
01229 | - DeepH adapter reports `unproven`, `failed`, `diagnostic_only`, unknown units,
01230 |   unknown orbital order, or R-vector mismatch.
01231 | - `low_energy_rmse_eV`, Fermi-window or DOS metrics are unavailable for one
01232 |   model but are required by `final_evaluation`.
01233 | - Final seeds are incomplete or fewer than `--min-final-seeds`.
01234 | - Telemetry is `partial` or `unavailable` for a compute claim.
01235 | - Dataset manifests are missing, repaired without provenance, or point to
01236 |   different split hashes.
01237 | - Search artifacts include test metrics.
01238 | 
01239 | ## Troubleshooting Blocked Gates
01240 | 
01241 | | Blocker | What to check |
01242 | | --- | --- |
01243 | | Missing provenance | Regenerate or attach SIESTA version/build, command line, FDF hash, basis/pseudo hashes and environment manifest. |
01244 | | Missing frozen split | Run the dataset manifest builder or verifier with `--write-manifests` only after confirming the intended split root. |
01245 | | Forbidden reference | Remove any `ML_prediction.HSX` or prediction path from reference fields and rerun metric staging. |
01246 | | DeepH diagnostic-only | Run the raw/global equivalence preflight and include generated evidence plus adapter manifests in the bundle. |
01247 | | Missing telemetry | Inspect per-run telemetry JSON; rerun affected jobs if GPU-hours, peak memory or throughput are required for claims. |
01248 | | Incomplete seeds | Resume final selected configs until the expected seed list is complete, then rerun final stats. |
01249 | | Smoke passed but gates fail | Expected behavior: smoke validates wiring only and writes `not_a_scientific_run`. Supply real evidence before claims. |
```

## `docs/cross_structure_evaluation.md`

SHA-256: `a003ee2e721c52f21e7fadc1e73808344139f1a54049dd18b0e0ad3f5eaf5a36`

```md
00001 | # Cross-Structure Evaluation
00002 | 
00003 | Cross-structure evaluation measures structural transfer: train and validate on
00004 | one validated MD dataset, then evaluate on the held-out test split of another
00005 | validated MD dataset. It is an out-of-distribution test when the source and
00006 | target structures differ, for example primitive graphene to a 5x5 graphene
00007 | supercell.
00008 | 
00009 | ## Split Contract
00010 | 
00011 | The materialized composite dataset is runner-ready and keeps split membership
00012 | frozen:
00013 | 
00014 | ```text
00015 | train      <- source dataset train split only
00016 | validation <- source dataset validation split only
00017 | test       <- target dataset test split only
00018 | ```
00019 | 
00020 | The source test split is excluded. The target train and validation splits are
00021 | excluded. No random re-splitting, ratio split, `_split_pool`, or SIESTA run is
00022 | performed.
00023 | 
00024 | Materialized sample ids are role-prefixed:
00025 | 
00026 | ```text
00027 | source_train__<original_id>
00028 | source_validation__<original_id>
00029 | target_test__<original_id>
00030 | ```
00031 | 
00032 | ## Compatibility
00033 | 
00034 | The implementation reuses the existing ML-vs-SIESTA compatibility checks. It
00035 | fails closed for different real species, real-species basis hashes,
00036 | pseudopotential hashes, active ghost target spaces, blocking DFT settings, and
00037 | the joint Graph2Mat/DeepH artifact contract.
00038 | 
00039 | It also fails closed when Hamiltonian target semantics are incomplete: H-only
00040 | policy, one matrix component, spin semantics, and real/complex representation
00041 | must be explicit enough for a production cross-structure run. Legacy preview or
00042 | development payloads may set `confirm_incomplete_hamiltonian_semantics=true`;
00043 | that confirmation is written into provenance.
00044 | 
00045 | Atom count, cell dimensions, lattice vectors, raw Hamiltonian dimensions, system
00046 | label, and raw Monkhorst-Pack integers may differ. K-point sampling is compared
00047 | through the existing reciprocal-spacing logic, so primitive-cell and supercell
00048 | k-grids can be compatible even when the integer grids differ.
00049 | 
00050 | ## Test Blindness
00051 | 
00052 | The target structure never contributes to training or validation. The source
00053 | structure never contributes to the cross-structure test split. Provenance stores
00054 | a leakage report checking role/split membership, unique materialized ids, and
00055 | unique canonical source artifact identities. Target test membership is inherited
00056 | from the target dataset frozen split and is deterministic across repeated
00057 | materializations.
00058 | 
00059 | `train` rejects `runner_payload.training_sweep` for this workflow; that keeps
00060 | target-test metrics out of hyperparameter search/model selection in the
00061 | payload-driven entry point.
00062 | 
00063 | ## Payload
00064 | 
00065 | Use:
00066 | 
00067 | ```bash
00068 | .venv/bin/python Comparison/scripts/run_cross_structure_payload.py \
00069 |   Comparison/config/graphene_w90_to_5x5_cross_structure_preview_payload.json
00070 | ```
00071 | 
00072 | Supported actions:
00073 | 
00074 | ```text
00075 | preview      validate and report counts/compatibility; write nothing
00076 | materialize  build the composite dataset; do not train
00077 | train        materialize/reuse the composite dataset and launch the existing runner
00078 | predict_metrics  materialize/reuse and evaluate existing checkpoints; never train
00079 | ```
00080 | 
00081 | For `train`, keep the CLI process alive and persist runner status:
00082 | 
00083 | ```bash
00084 | .venv/bin/python Comparison/scripts/run_cross_structure_payload.py payload.json \
00085 |   --status-json Comparison/results/cross_structure/status.json \
00086 |   --manifest-json Comparison/results/cross_structure/manifest.json \
00087 |   --poll-seconds 30
00088 | ```
00089 | 
00090 | Important payload fields:
00091 | 
00092 | ```json
00093 | {
00094 |   "schema": "g2m_deeph_cross_structure_run_v1",
00095 |   "action": "preview",
00096 |   "source_dataset_root": "${REPO_ROOT}/Comparison/datasets/...",
00097 |   "target_dataset_root": "${REPO_ROOT}/Comparison/datasets/...",
00098 |   "composite_dataset_root": "${REPO_ROOT}/Comparison/results/.../dataset",
00099 |   "run_output_root": "${REPO_ROOT}/Comparison/results/.../training",
00100 |   "link": true,
00101 |   "overwrite": false,
00102 |   "runner_payload": {
00103 |     "selected_methods": ["graph2mat", "deeph"],
00104 |     "allow_diagnostic_metrics": false,
00105 |     "metric_fail_policy": "fail_closed",
00106 |     "graph2mat_overrides": {},
00107 |     "deeph": {},
00108 |     "performance": {}
00109 |   }
00110 | }
00111 | ```
00112 | 
00113 | For `train`, protected runner fields are forced by the wrapper and cannot be
00114 | overridden inside `runner_payload`:
00115 | 
00116 | ```json
00117 | {
00118 |   "dataset_mode": "reuse_validated",
00119 |   "dataset_root": "<composite_dataset_root>",
00120 |   "output_root": "<run_output_root>",
00121 |   "allow_regenerate_siesta": false
00122 | }
00123 | ```
00124 | 
00125 | ## Graphene 5x5 monovacancy campaign
00126 | 
00127 | The preset `materials/graphene_5x5_vacancy` is the pristine 5x5 cell with the
00128 | ideal central carbon at fractional position `(0.5, 0.5, 0.0)` removed. It has 49
00129 | carbons, is unrelaxed and non-spin-polarized, and shares the pristine PAO basis,
00130 | pseudopotential and electronic settings. The target builder deletes the same
00131 | zero-based atom index from pristine test snapshots and records both the ideal
00132 | site and the actual MD-displaced position in each `metadata.json`.
00133 | 
00134 | First inspect the transformation. This writes no final dataset and never invokes
00135 | SIESTA:
00136 | 
00137 | ```bash
00138 | .venv/bin/python Comparison/scripts/build_graphene_5x5_vacancy_target.py \
00139 |   --source-dataset Comparison/datasets/graphene_5x5_snapshot_scaling_20_50_80/graphene_5x5_scale_iid20 \
00140 |   --output-root /tmp/graphene_5x5_vacancy_dry_run \
00141 |   --source-split test \
00142 |   --limit 2 \
00143 |   --atom-index 24 \
00144 |   --dry-run
00145 | ```
00146 | 
00147 | Generate the real static SIESTA references when compute is available:
00148 | 
00149 | ```bash
00150 | .venv/bin/python Comparison/scripts/build_graphene_5x5_vacancy_target.py \
00151 |   --source-dataset Comparison/datasets/graphene_5x5_snapshot_scaling_complete_paper_ready_missing/graphene_5x5_scale_iid600 \
00152 |   --output-root Comparison/datasets/graphene_5x5_vacancy \
00153 |   --source-split test \
00154 |   --limit 20 \
00155 |   --atom-index 24 \
00156 |   --siesta-command siesta
00157 | ```
00158 | 
00159 | The output contains only a frozen `test` split. Every sample is checked against
00160 | the existing joint Graph2Mat/DeepH artifact contract before the dataset is
00161 | marked ready. An existing output is preserved unless `--overwrite` is passed;
00162 | a failed SIESTA run removes the partial replacement and restores any previous
00163 | target.
00164 | 
00165 | The campaign payload is reproducible rather than hand-maintained:
00166 | 
00167 | ```bash
00168 | .venv/bin/python Comparison/scripts/ops/build_cross_predict_metrics_payload.py
00169 | ```
00170 | 
00171 | It creates
00172 | `Comparison/config/graphene_w90_5x5_to_vacancy_predict_metrics_payload.json`,
00173 | with one w90→vacancy and one 5x5→vacancy pair per catalogued source size. Preview
00174 | it without loading a model:
00175 | 
00176 | ```bash
00177 | .venv/bin/python Comparison/scripts/run_cross_structure_sweep_payload.py \
00178 |   Comparison/config/graphene_w90_5x5_to_vacancy_predict_metrics_payload.json \
00179 |   --action preview
00180 | ```
00181 | 
00182 | Preview fails closed as incompatible, with the missing-dataset reason, until
00183 | `Comparison/datasets/graphene_5x5_vacancy` exists and validates. Before the real
00184 | evaluation, stage the archived w90 checkpoints referenced by the same payload:
00185 | 
00186 | ```bash
00187 | .venv/bin/python Comparison/scripts/ops/prepare_cross_predict_metrics_artifacts.py \
00188 |   Comparison/config/graphene_w90_5x5_to_vacancy_predict_metrics_payload.json
00189 | ```
00190 | 
00191 | Then evaluate. `predict_metrics` sets `predict_metrics_only=true` and passes the
00192 | catalogued Graph2Mat training directories and DeepH model directories to the
00193 | runner; no fitting, epoch loop or checkpoint modification occurs:
00194 | 
00195 | ```bash
00196 | .venv/bin/python Comparison/scripts/run_cross_structure_sweep_payload.py \
00197 |   Comparison/config/graphene_w90_5x5_to_vacancy_predict_metrics_payload.json \
00198 |   --output-root Comparison/results/ml_vs_siesta_cross_structure_sweep/vacancy \
00199 |   --result-json Comparison/results/ml_vs_siesta_cross_structure_sweep/vacancy/result.json
00200 | ```
00201 | 
00202 | For the UI, start `python3 Comparison/scripts/pipeline_ui.py`, open **Cross
00203 | testing**, and use **Cross testing con vacante — checkpoints existentes**. Its
00204 | payload field is restricted to JSON files below `Comparison/config`; the preview
00205 | and evaluation buttons use their own log, live status and MAE chart, without
00206 | altering the normal cross-testing plot or its payload selection.
00207 | The evaluation button always overrides the action to `predict_metrics`.
00208 | 
00209 | Results are written below the selected sweep output root, one stable
00210 | `<source_id>__to__<target_id>` directory per pair. The summary is
00211 | `cross_structure_sweep_summary.json`; model metric directories contain the
00212 | existing `h_mae_eV`, `h_rmse_eV`, relative/Frobenius and spectral summaries plus
00213 | `block_metrics.csv`, `species_pair_metrics.csv` and `orbital_pair_metrics.csv`
00214 | when available. The UI plots `h_mae_eV` in meV versus source training snapshots
00215 | as four distinct source/model curves.
00216 | 
00217 | To compare the pristine 5x5 baseline with the vacancy target, match the same
00218 | source checkpoint/model and compare their final-test metrics. In
00219 | `block_metrics.csv`, `row_atom == col_atom` identifies on-site blocks and
00220 | `row_atom != col_atom` identifies off-site blocks. Do not interpret
00221 | `distance_bin_metrics.csv` as a valid local periodic analysis for graphene: the
00222 | current evaluator intentionally disables those bins because it does not apply
00223 | the periodic minimum-image convention. DOS, LDOS, IPR and spin analysis are not
00224 | part of this initial campaign.
00225 | 
00226 | ## Graphene/hBN bilayer → twisted-moire campaign
00227 | 
00228 | This variant trains one Graph2Mat and one DeepH on a **single** dataset that
00229 | fuses the AA/AB1/AB2 graphene-hBN stackings, then cross tests them against an
00230 | independent **twisted-moire** supercell (many more atoms). The three stackings
00231 | share species (C/B/N), PAO basis and pseudopotentials, so the fused pool is
00232 | physically meaningful and the planner accepts the bilayer→moire pair (more
00233 | atoms and a larger cell are allowed; species/basis/pseudo hashes must match).
00234 | 
00235 | **Material bundles.** `materials/graphene_hBN_{AA,AB1,AB2}/` each hold a static
00236 | electronic `RUN.fdf` template (no MD block; the MD layer is appended at
00237 | generation time) and a `material.yaml` that points pseudopotentials and basis at
00238 | the shared `materials/graphene_hBN_common/{pseudos,basis}` directory (symlinks to
00239 | `C.psf`, `B.psml`, `N.psml`, `C.ion.xml`, `B.ion.xml`, `N.ion.xml`).
00240 | 
00241 | **Phase 1 — per-stacking MD datasets (payload only, no new code).** One small MD
00242 | dataset per stacking via `run_g2m_deeph_payload_once.py`:
00243 | 
00244 | ```bash
00245 | for S in AA AB1 AB2; do
00246 |   .venv/bin/python Comparison/scripts/run_g2m_deeph_payload_once.py \
00247 |     Comparison/config/graphene_hbn_${S}_md30_payload.json \
00248 |     --status-json /tmp/${S}_md_status.json \
00249 |     --manifest-json /tmp/${S}_md_manifest.json
00250 | done
00251 | ```
00252 | 
00253 | Each writes `Comparison/datasets/graphene_hBN_<S>_md30` with `*.HSX`,
00254 | `*.ORB_INDX`, `*.STRUCT_OUT` present for all three species (the shared MD
00255 | pipeline config keeps `Write.OrbitalIndex`, `XML.Write` and `SaveHS`).
00256 | 
00257 | **Phase 2 — fuse into one train pool.** `build_graphene_hbn_bilayer_train_dataset.py`
00258 | copies the train+validation snapshots of the three datasets into one
00259 | `dataset_root` with re-indexed, stacking-prefixed ids. It never re-runs SIESTA
00260 | and fails closed if basis or pseudopotential hashes differ between stackings:
00261 | 
00262 | ```bash
00263 | # The sweep nests each dataset under its recipe slug (graphene_hbn_<s>_md30).
00264 | .venv/bin/python Comparison/scripts/build_graphene_hbn_bilayer_train_dataset.py \
00265 |   --source-dataset Comparison/datasets/graphene_hBN_AA_md30/graphene_hbn_aa_md30 \
00266 |   --source-dataset Comparison/datasets/graphene_hBN_AB1_md30/graphene_hbn_ab1_md30 \
00267 |   --source-dataset Comparison/datasets/graphene_hBN_AB2_md30/graphene_hbn_ab2_md30 \
00268 |   --output-root Comparison/datasets/graphene_hBN_bilayer_train
00269 | ```
00270 | 
00271 | `material_provenance.json` records the mixture (three source datasets + hashes).
00272 | 
00273 | **Phase 3 — train one G2M + one DeepH on the fused pool** with a small-epoch
00274 | snapshot-scaling payload whose `dataset_root` is `graphene_hBN_bilayer_train`.
00275 | Persist the checkpoints where the Phase 5 payload expects them
00276 | (`graph2mat_training_dir` with `checkpoint_manifest.json` + `*.ckpt`;
00277 | `deeph_save_dir` with `config.ini` + `best_state_dict.pkl`).
00278 | 
00279 | **Phase 4 — twisted-moire target.** `build_graphene_hbn_moire_target.py` builds
00280 | the standard periodic commensurate cell: graphene uses the `(m,n)` supercell
00281 | basis and hBN the rotated `(n,m)` basis. For `(1,2)` the coincidence index is 7,
00282 | so the physically periodic bilayer has 28 atoms (14 per layer), then static
00283 | SIESTA runs into a frozen `test` split:
00284 | 
00285 | ```bash
00286 | # geometry-only preview (no SIESTA)
00287 | .venv/bin/python Comparison/scripts/build_graphene_hbn_moire_target.py \
00288 |   --approximant 2 --commensurate-angle 1,2 --dry-run
00289 | 
00290 | # real static references
00291 | .venv/bin/python Comparison/scripts/build_graphene_hbn_moire_target.py \
00292 |   --approximant 2 --commensurate-angle 1,2 --limit 1 --overwrite \
00293 |   --output-root Comparison/datasets/graphene_hBN_moire_22deg --siesta-command siesta
00294 | ```
00295 | 
00296 | **Physics caveat (documented, not hidden).** Graphene and hBN are naturally
00297 | incommensurate. This smoke target uses the 2.480 Å graphene lattice for both
00298 | layers, so hBN is biaxially compressed by `(2.480/2.504 - 1)*100 = -0.9585%`
00299 | relative to the recorded 2.504 Å native reference. The strain is calculated
00300 | from the written geometry, not stored as a mismatch proxy. The builder also
00301 | measures the minimum atom distance over 3x3 in-plane periodic images before
00302 | writing/running SIESTA and aborts below `--min-atom-distance` (1.2 Å by default).
00303 | For `(m,n)=(1,2)`, the written layer bases reproduce the calculated ~21.79°
00304 | twist. This remains a smoke-scale commensurate surrogate, not a relaxed
00305 | paper-ready incommensurate moire.
00306 | 
00307 | **Phase 5 — predict_metrics payload** (reproducible via the ops generator):
00308 | 
00309 | ```bash
00310 | .venv/bin/python Comparison/scripts/ops/build_cross_predict_metrics_payload.py \
00311 |   --bilayer-output Comparison/config/graphene_hbn_bilayer_to_moire_predict_metrics_payload.json \
00312 |   --bilayer-source Comparison/datasets/graphene_hBN_bilayer_train \
00313 |   --bilayer-target Comparison/datasets/graphene_hBN_moire_22deg
00314 | ```
00315 | 
00316 | The payload has a single `bilayer_to_moire` pair; `existing_artifacts` is keyed
00317 | by the source dataset basename (`graphene_hBN_bilayer_train`). Preview then
00318 | evaluate:
00319 | 
00320 | ```bash
00321 | .venv/bin/python Comparison/scripts/run_cross_structure_sweep_payload.py \
00322 |   Comparison/config/graphene_hbn_bilayer_to_moire_predict_metrics_payload.json --action preview
00323 | 
00324 | .venv/bin/python Comparison/scripts/run_cross_structure_sweep_payload.py \
00325 |   Comparison/config/graphene_hbn_bilayer_to_moire_predict_metrics_payload.json \
00326 |   --action predict_metrics \
00327 |   --output-root Comparison/results/ml_vs_siesta_cross_structure_bilayer_moire
00328 | ```
00329 | 
00330 | `records[]` carry `h_mae_eV` for both models; the payload id is
00331 | `graphene_hBN_bilayer__to__graphene_hBN_moire_<ang>` (auto-derived, never
00332 | hardcoded).
00333 | 
00334 | **Phase 6 — UI.** Start `python3 Comparison/scripts/pipeline_ui.py`, open **Cross
00335 | testing**, scroll to the last subsection **Cross testing bicapa grafeno/hBN →
00336 | moiré rotado**. Its payload field is restricted to JSON under `Comparison/config`
00337 | and defaults to the Phase 5 payload; preview/evaluate/metrics use their own log,
00338 | status and MAE chart on `/api/cross-testing/bilayer/*` without touching the
00339 | normal or vacancy subsections.
00340 | 
00341 | ## Outputs
00342 | 
00343 | `materialize` writes:
00344 | 
00345 | ```text
00346 | splits/{train,validation,test}/
00347 | splits/{train,validation,test}_manifest.csv
00348 | artifact_validation.json
00349 | benchmark_dataset_manifest.json
00350 | frozen_split_manifest.json
00351 | material_provenance.json
00352 | cross_structure_dataset_provenance.json
00353 | dataset_compatibility_report.json
00354 | ```
00355 | 
00356 | The split CSV rows and frozen manifest rows preserve `evaluation_mode`,
00357 | `transfer_direction`, role, original sample id, original split, original source
00358 | root, system label, atom count when available, and artifact validation status.
00359 | `cross_structure_dataset_provenance.json` also stores the source-artifact
00360 | identity used for leakage checks and actual link/copy counts. If `train` sees an
00361 | existing composite dataset with matching source/target roots and split hashes,
00362 | it recalculates leakage from the current frozen rows before reusing it unless
00363 | `overwrite=true`.
00364 | 
00365 | ## Example
00366 | 
00367 | The committed example uses existing local datasets:
00368 | 
00369 | ```text
00370 | source: Comparison/datasets/graphene_w90_snapshot_scaling/graphene_w90_scale_iid10
00371 | target: Comparison/datasets/graphene_5x5_snapshot_scaling_20_50_80/graphene_5x5_scale_iid20
00372 | ```
00373 | 
00374 | It previews a 2-atom graphene to 50-atom graphene 5x5 transfer case.
00375 | 
00376 | ## Limitations
00377 | 
00378 | The payload wrapper does not support `training_sweep` in cross-structure mode;
00379 | use fixed runner settings for training. The vacancy workflow itself performs no
00380 | retraining and only evaluates existing checkpoints.
00381 | Metric formulas are unchanged, so compare normalized/per-entry metrics across
00382 | structures rather than unnormalized total norms. Some older local datasets may
00383 | lack explicit Hamiltonian semantics; those require the explicit development
00384 | confirmation above or updated provenance before production materialization.
```

## `Comparison/config/g2m_deeph_paper_protocol_v1_example.json`

SHA-256: `1e4db60859ed9fc01ca0d728b9d87dfd764e8f6880f8a8fc8cbb5bd55d3577de`

```json
00001 | {
00002 |   "protocol_id": "graphene_w90_g2m_deeph_final_v1",
00003 |   "version": "1.0",
00004 |   "datasets": [
00005 |     {
00006 |       "dataset_id": "graphene_w90_phase1_iid300",
00007 |       "dataset_root": "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid300",
00008 |       "benchmark_dataset_manifest": "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid300/benchmark_dataset_manifest.json",
00009 |       "frozen_split_manifest": "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid300/frozen_split_manifest.json",
00010 |       "compatibility_hash": "recorded_in_benchmark_dataset_manifest"
00011 |     },
00012 |     {
00013 |       "dataset_id": "graphene_w90_phase1_iid600",
00014 |       "dataset_root": "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid600",
00015 |       "benchmark_dataset_manifest": "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid600/benchmark_dataset_manifest.json",
00016 |       "frozen_split_manifest": "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid600/frozen_split_manifest.json",
00017 |       "compatibility_hash": "recorded_in_benchmark_dataset_manifest"
00018 |     }
00019 |   ],
00020 |   "reference_artifacts": {
00021 |     "contract": "joint_graph2mat_deeph_artifact_contract_v1",
00022 |     "required": [
00023 |       "RUN.fdf",
00024 |       "SystemLabel.TSHS",
00025 |       "SystemLabel.TSDE",
00026 |       "SystemLabel.HSX",
00027 |       "SystemLabel.STRUCT_OUT",
00028 |       "SystemLabel.XV",
00029 |       "SystemLabel.ORB_INDX",
00030 |       "metadata.json"
00031 |     ],
00032 |     "forbidden": [
00033 |       "ML_prediction.HSX"
00034 |     ],
00035 |     "forbid_as_reference": "ML_prediction.HSX"
00036 |   },
00037 |   "models": {
00038 |     "graph2mat": {
00039 |       "enabled": true,
00040 |       "search_space": {
00041 |         "optim_lr": {
00042 |           "choices": [
00043 |             0.0012,
00044 |             0.003,
00045 |             0.005,
00046 |             0.01
00047 |           ]
00048 |         },
00049 |         "batch_size": {
00050 |           "choices": [
00051 |             32,
00052 |             48,
00053 |             64,
00054 |             96,
00055 |             128,
00056 |             256
00057 |           ]
00058 |         },
00059 |         "max_epochs": {
00060 |           "value": 600
00061 |         },
00062 |         "hidden_irreps": {
00063 |       "choices": [
00064 |         "24x0e + 24x1o + 24x2e + 24x3o",
00065 |         "32x0e + 32x1o + 32x2e + 32x3o",
00066 |         "48x0e + 48x1o + 48x2e + 48x3o",
00067 |         "64x0e + 64x1o + 64x2e + 64x3o"
00068 |       ]
00069 |         },
00070 |         "num_interactions": {
00071 |           "value": 3
00072 |         },
00073 |         "correlation": {
00074 |           "choices": [
00075 |             2,
00076 |             3
00077 |           ]
00078 |         },
00079 |         "max_ell": {
00080 |           "value": 3
00081 |         },
00082 |         "loss": {
00083 |           "choices": [
00084 |             "graph2mat.metrics.block_type_mse",
00085 |             "graph2mat.metrics.block_type_huber"
00086 |           ]
00087 |         },
00088 |         "loss_kwargs": {
00089 |           "choices": [
00090 |             {},
00091 |             {
00092 |               "beta": 0.003
00093 |             },
00094 |             {
00095 |               "beta": 0.01
00096 |             }
00097 |           ]
00098 |         },
00099 |         "readout": {
00100 |           "choices": [
00101 |             "default",
00102 |             "edge_node_mix"
00103 |           ]
00104 |         }
00105 |       }
00106 |     },
00107 |     "deeph": {
00108 |       "enabled": true,
00109 |       "search_space": {
00110 |         "learning_rate": {
00111 |           "distribution": "loguniform",
00112 |           "min": 0.00003,
00113 |           "max": 0.001
00114 |         },
00115 |         "batch_size": {
00116 |           "choices": [
00117 |             4,
00118 |             8
00119 |           ]
00120 |         },
00121 |         "epochs": {
00122 |           "value": 600
00123 |         },
00124 |         "optimizer": {
00125 |           "value": "adamW"
00126 |         },
00127 |         "weight_decay": {
00128 |           "choices": [
00129 |             0.0,
00130 |             0.0001
00131 |           ]
00132 |         },
00133 |         "criterion": {
00134 |           "value": "MaskMSELoss"
00135 |         },
00136 |         "atom_fea_len": {
00137 |           "choices": [
00138 |             64,
00139 |             128
00140 |           ]
00141 |         },
00142 |         "edge_fea_len": {
00143 |           "value": 128
00144 |         },
00145 |         "gauss_stop": {
00146 |           "value": 6
00147 |         },
00148 |         "num_l": {
00149 |           "choices": [
00150 |             4
00151 |           ]
00152 |         },
00153 |         "if_edge_update": {
00154 |           "value": true
00155 |         },
00156 |         "if_lcmp": {
00157 |           "value": true
00158 |         },
00159 |         "normalization": {
00160 |           "value": "LayerNorm"
00161 |         },
00162 |         "atom_update_net": {
00163 |           "value": "CGConv"
00164 |         },
00165 |         "retain_edge_fea": {
00166 |           "value": true
00167 |         }
00168 |       }
00169 |     }
00170 |   },
00171 |   "selection": {
00172 |     "split": "validation",
00173 |     "metric": "val_loss",
00174 |     "mode": "min",
00175 |     "source": "validation_only"
00176 |   },
00177 |   "early_stopping": {
00178 |     "metric": "val_loss",
00179 |     "mode": "min",
00180 |     "patience": 30,
00181 |     "min_delta": 0.0,
00182 |     "max_epochs": 600
00183 |   },
00184 |   "search_policy": {
00185 |     "strategy": "latin_hypercube",
00186 |     "n_trials_per_model": 40,
00187 |     "random_seed": 20260528
00188 |   },
00189 |   "budget_policy": {
00190 |     "mode": "equal_n_trials",
00191 |     "n_trials_per_model": 40
00192 |   },
00193 |   "final_seeds": [
00194 |     0,
00195 |     1,
00196 |     2,
00197 |     3,
00198 |     4
00199 |   ],
00200 |   "top_k_selection": {
00201 |     "k_per_model": 2,
00202 |     "split": "validation",
00203 |     "metric": "val_loss",
00204 |     "uses_test_metrics": false
00205 |   },
00206 |   "final_evaluation": {
00207 |     "primary_metric": "low_energy_rmse_eV",
00208 |     "mode": "min",
00209 |     "secondary_metrics": [
00210 |       "fermi_window_rmse_eV",
00211 |       "frontier_window_rmse_eV",
00212 |       "dos_wasserstein_eV",
00213 |       "h_mae_eV"
00214 |     ],
00215 |     "practical_match": {
00216 |       "relative_gap_max": 1.1,
00217 |       "absolute_gap_meV_max": null,
00218 |       "requires_cost_noninferior": true
00219 |     }
00220 |   },
00221 |   "final_test_policy": {
00222 |     "policy": "locked_until_final",
00223 |     "test_split": "test",
00224 |     "locked_during_search": true,
00225 |     "evaluate_once_after_selection": true
00226 |   },
00227 |   "required_telemetry": [
00228 |     "wall_clock_seconds",
00229 |     "gpu_hours",
00230 |     "peak_gpu_memory_mb",
00231 |     "samples_per_second",
00232 |     "matrix_blocks_per_second",
00233 |     "best_validation_epoch"
00234 |   ],
00235 |   "deeph_comparability": {
00236 |     "adapter_equivalence_policy": "fail_closed_unless_proven",
00237 |     "robust_winner_requires_proven_equivalence": true,
00238 |     "diagnostic_if_unproven": true
00239 |   }
00240 | }
```

## `Comparison/config/graphene_w90_g2m_deeph_weekend_iid1000_paper_ready_v1.json`

SHA-256: `775a9ae8cbb8187076d11071e42f57028a79a79c7290113977ff867de3df4111`

```json
00001 | {
00002 |   "protocol_id": "graphene_w90_g2m_deeph_weekend_iid1000_paper_ready_v1",
00003 |   "version": "1.1",
00004 |   "datasets": [
00005 |     {
00006 |       "dataset_id": "graphene_w90_phase1_iid1000",
00007 |       "dataset_root": "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000",
00008 |       "benchmark_dataset_manifest": "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/benchmark_dataset_manifest.json",
00009 |       "frozen_split_manifest": "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/frozen_split_manifest.json",
00010 |       "artifact_validation": "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/artifact_validation.json",
00011 |       "compatibility_hash": "recorded_in_benchmark_dataset_manifest",
00012 |       "role": "search_and_validation"
00013 |     }
00014 |   ],
00015 |   "reference_artifacts": {
00016 |     "contract": "joint_graph2mat_deeph_artifact_contract_v1",
00017 |     "required": [
00018 |       "RUN.fdf",
00019 |       "SystemLabel.TSHS",
00020 |       "SystemLabel.TSDE",
00021 |       "SystemLabel.HSX",
00022 |       "SystemLabel.STRUCT_OUT",
00023 |       "SystemLabel.XV",
00024 |       "SystemLabel.ORB_INDX",
00025 |       "metadata.json"
00026 |     ],
00027 |     "forbidden": [
00028 |       "ML_prediction.HSX"
00029 |     ],
00030 |     "forbid_as_reference": "ML_prediction.HSX"
00031 |   },
00032 |   "models": {
00033 |     "graph2mat": {
00034 |       "enabled": true,
00035 |       "search_space": {
00036 |         "optim_lr": {
00037 |           "choices": [
00038 |             0.0012,
00039 |             0.003,
00040 |             0.005,
00041 |             0.01
00042 |           ]
00043 |         },
00044 |         "batch_size": {
00045 |           "choices": [
00046 |             32,
00047 |             48,
00048 |             64,
00049 |             96,
00050 |             128,
00051 |             256
00052 |           ]
00053 |         },
00054 |         "max_epochs": {
00055 |           "value": 600
00056 |         },
00057 |         "hidden_irreps": {
00058 |           "choices": [
00059 |             "24x0e + 24x1o + 24x2e + 24x3o",
00060 |             "32x0e + 32x1o + 32x2e + 32x3o",
00061 |             "48x0e + 48x1o + 48x2e + 48x3o",
00062 |             "64x0e + 64x1o + 64x2e + 64x3o"
00063 |           ]
00064 |         },
00065 |         "num_interactions": {
00066 |           "value": 3
00067 |         },
00068 |         "correlation": {
00069 |           "choices": [
00070 |             2,
00071 |             3
00072 |           ]
00073 |         },
00074 |         "max_ell": {
00075 |           "value": 3
00076 |         },
00077 |         "loss": {
00078 |           "choices": [
00079 |             "graph2mat.metrics.block_type_mse",
00080 |             "graph2mat.metrics.block_type_huber"
00081 |           ]
00082 |         },
00083 |         "loss_kwargs": {
00084 |           "choices": [
00085 |             {},
00086 |             {
00087 |               "beta": 0.003
00088 |             },
00089 |             {
00090 |               "beta": 0.01
00091 |             }
00092 |           ]
00093 |         },
00094 |         "readout": {
00095 |           "choices": [
00096 |             "default",
00097 |             "edge_node_mix"
00098 |           ]
00099 |         },
00100 |         "seed_everything": {
00101 |           "value": 0
00102 |         }
00103 |       }
00104 |     },
00105 |     "deeph": {
00106 |       "enabled": true,
00107 |       "search_space": {
00108 |         "learning_rate": {
00109 |           "distribution": "loguniform",
00110 |           "min": 3e-05,
00111 |           "max": 0.001
00112 |         },
00113 |         "batch_size": {
00114 |           "choices": [
00115 |             2,
00116 |             4,
00117 |             8
00118 |           ]
00119 |         },
00120 |         "epochs": {
00121 |           "value": 600
00122 |         },
00123 |         "optimizer": {
00124 |           "value": "adamW"
00125 |         },
00126 |         "weight_decay": {
00127 |           "choices": [
00128 |             0.0,
00129 |             1e-05,
00130 |             0.0001
00131 |           ]
00132 |         },
00133 |         "criterion": {
00134 |           "value": "MaskMSELoss"
00135 |         },
00136 |         "atom_fea_len": {
00137 |           "choices": [
00138 |             64,
00139 |             128,
00140 |             256
00141 |           ]
00142 |         },
00143 |         "edge_fea_len": {
00144 |           "choices": [
00145 |             32,
00146 |             64,
00147 |             128
00148 |           ]
00149 |         },
00150 |         "gauss_stop": {
00151 |           "value": 6
00152 |         },
00153 |         "num_l": {
00154 |           "choices": [
00155 |             4
00156 |           ]
00157 |         },
00158 |         "if_edge_update": {
00159 |           "value": true
00160 |         },
00161 |         "if_lcmp": {
00162 |           "value": true
00163 |         },
00164 |         "normalization": {
00165 |           "value": "LayerNorm"
00166 |         },
00167 |         "atom_update_net": {
00168 |           "value": "CGConv"
00169 |         },
00170 |         "retain_edge_fea": {
00171 |           "value": true
00172 |         },
00173 |         "seed": {
00174 |           "value": 0
00175 |         }
00176 |       }
00177 |     }
00178 |   },
00179 |   "selection": {
00180 |     "split": "validation",
00181 |     "metric": "val_loss",
00182 |     "mode": "min",
00183 |     "source": "validation_only"
00184 |   },
00185 |   "early_stopping": {
00186 |     "metric": "val_loss",
00187 |     "mode": "min",
00188 |     "patience": 70,
00189 |     "min_delta": 0.0,
00190 |     "max_epochs": 600
00191 |   },
00192 |   "search_policy": {
00193 |     "strategy": "latin_hypercube",
00194 |     "n_trials_per_model": 40,
00195 |     "random_seed": 20260528
00196 |   },
00197 |   "budget_policy": {
00198 |     "mode": "equal_n_trials",
00199 |     "n_trials_per_model": 40
00200 |   },
00201 |   "final_seeds": [
00202 |     0,
00203 |     1,
00204 |     2,
00205 |     3,
00206 |     4
00207 |   ],
00208 |   "top_k_selection": {
00209 |     "k_per_model": 5,
00210 |     "split": "validation",
00211 |     "metric": "val_loss",
00212 |     "uses_test_metrics": false
00213 |   },
00214 |   "final_evaluation": {
00215 |     "primary_metric": "low_energy_rmse_eV",
00216 |     "mode": "min",
00217 |     "secondary_metrics": [
00218 |       "fermi_window_rmse_eV",
00219 |       "frontier_window_rmse_eV",
00220 |       "dos_wasserstein_eV",
00221 |       "h_mae_eV"
00222 |     ],
00223 |     "practical_match": {
00224 |       "relative_gap_max": 1.1,
00225 |       "absolute_gap_meV_max": null,
00226 |       "requires_cost_noninferior": true
00227 |     }
00228 |   },
00229 |   "final_test_policy": {
00230 |     "policy": "locked_until_final",
00231 |     "test_split": "test",
00232 |     "locked_during_search": true,
00233 |     "evaluate_once_after_selection": true
00234 |   },
00235 |   "required_telemetry": [
00236 |     "wall_clock_seconds",
00237 |     "gpu_hours",
00238 |     "peak_gpu_memory_mb",
00239 |     "samples_per_second",
00240 |     "matrix_blocks_per_second",
00241 |     "best_validation_epoch"
00242 |   ],
00243 |   "deeph_comparability": {
00244 |     "adapter_equivalence_policy": "fail_closed_unless_proven",
00245 |     "robust_winner_requires_proven_equivalence": true,
00246 |     "diagnostic_if_unproven": true
00247 |   },
00248 |   "performance": {
00249 |     "compute_accelerator": "gpu",
00250 |     "torch_mixed_precision": "bf16-mixed",
00251 |     "torch_float32_matmul_precision": "high",
00252 |     "max_parallel_graph2mat_training_jobs": 3,
00253 |     "max_parallel_deeph_training_jobs": 2,
00254 |     "graph2mat_log_every_n_steps": 1,
00255 |     "graph2mat_check_val_every_n_epoch": 1,
00256 |     "graph2mat_checkpoint_every_n_epochs": 1,
00257 |     "graph2mat_require_cuequivariance": true,
00258 |     "torch_num_threads": 4,
00259 |     "omp_num_threads": 4,
00260 |     "mkl_num_threads": 4,
00261 |     "openblas_num_threads": 4,
00262 |     "numexpr_num_threads": 4
00263 |   },
00264 |   "notes": {
00265 |     "purpose": "Weekend paper-ready search canary/full sweep on iid1000 with small/medium Graph2Mat batches. Final ID/OOD locked evaluation remains a later stage after validation-only selection.",
00266 |     "created_after_stopping_run": "g2m_deeph_perf1024_cueq_clean_20260528_140516",
00267 |     "test_blind_policy": "selection and early stopping use validation val_loss only; final claims use final_evaluation.primary_metric only after final stage."
00268 |   }
00269 | }
```

## `Comparison/scripts/g2m_deeph_protocol.py`

SHA-256: `279d8448ac4b61dac40c73c9116c2e648dbe1b160714c8c47e9b77da7c4b5056`

```py
00001 | #!/usr/bin/env python3
00002 | """Paper-ready protocol validation for the Graph2Mat-vs-DeepH benchmark."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import argparse
00007 | import copy
00008 | import hashlib
00009 | import json
00010 | from pathlib import Path
00011 | from typing import Any
00012 | 
00013 | from g2m_deeph_training_sweep import DEEPH_KEYS, FORBIDDEN_DEEPH_KEYS, GRAPH2MAT_KEYS, json_safe
00014 | from graph2mat_sweep_config import GRAPH2MAT_READOUT_FAMILIES
00015 | 
00016 | 
00017 | SCHEMA_NAME = "graph2mat_deeph_benchmark_protocol_v1"
00018 | ALLOWED_MODELS = {"graph2mat", "deeph"}
00019 | ALLOWED_SEARCH_STRATEGIES = {"grid", "manual", "random", "latin_hypercube"}
00020 | ALLOWED_BUDGET_MODES = {"equal_n_trials", "equal_gpu_hours_per_model"}
00021 | ALLOWED_MODES = {"min", "max"}
00022 | ALLOWED_FINAL_TEST_POLICIES = {"locked_until_final"}
00023 | ALLOWED_DEEPH_EQUIVALENCE_POLICIES = {"fail_closed_unless_proven"}
00024 | SUPPORTED_DEEPH_CRITERIA = {"MaskMSELoss"}
00025 | SUPPORTED_DEEPH_OPTIMIZERS = {"sgd", "sgdm", "adam", "adamW", "adagrad", "RMSprop", "lbfgs"}
00026 | DISALLOWED_FINAL_CLAIM_METRICS = {
00027 |     "loss",
00028 |     "metric_value",
00029 |     "test_loss",
00030 |     "train_loss",
00031 |     "training_loss",
00032 |     "val_loss",
00033 |     "validation_loss",
00034 |     "validation_metric",
00035 | }
00036 | ALLOWED_FINAL_CLAIM_METRICS = {
00037 |     "dos_mae_500_fermi_window",
00038 |     "dos_mae_500_fermi_window_mean",
00039 |     "dos_wasserstein_eV",
00040 |     "dos_wasserstein_eV_mean",
00041 |     "fermi_window_rmse_eV",
00042 |     "fermi_window_rmse_eV_mean",
00043 |     "frontier_window_rmse_eV",
00044 |     "frontier_window_rmse_eV_mean",
00045 |     "global_rmse_eV",
00046 |     "global_rmse_eV_mean",
00047 |     "h_mae_eV",
00048 |     "h_mae_eV_mean",
00049 |     "h_mse_eV",
00050 |     "h_mse_eV_mean",
00051 |     "h_rmse_eV",
00052 |     "h_rmse_eV_mean",
00053 |     "low_energy_rmse_eV",
00054 |     "low_energy_rmse_eV_mean",
00055 |     "relative_frobenius",
00056 |     "relative_frobenius_mean",
00057 |     "spectral_composite_score",
00058 |     "spectral_composite_score_mean",
00059 |     "test_spectral_composite_score",
00060 |     "test_spectral_composite_score_mean",
00061 | }
00062 | VALIDATION_COMPOSITE_METRICS = {"val_spectral_composite"}
00063 | 
00064 | REQUIRED_REFERENCE_ARTIFACTS = {
00065 |     "RUN.fdf",
00066 |     "SystemLabel.TSHS",
00067 |     "SystemLabel.TSDE",
00068 |     "SystemLabel.HSX",
00069 |     "SystemLabel.STRUCT_OUT",
00070 |     "SystemLabel.XV",
00071 |     "SystemLabel.ORB_INDX",
00072 |     "metadata.json",
00073 | }
00074 | REQUIRED_TELEMETRY_FIELDS = {
00075 |     "wall_clock_seconds",
00076 |     "gpu_hours",
00077 |     "peak_gpu_memory_mb",
00078 |     "samples_per_second",
00079 |     "matrix_blocks_per_second",
00080 |     "best_validation_epoch",
00081 | }
00082 | REQUIRED_TOP_LEVEL_FIELDS = {
00083 |     "protocol_id",
00084 |     "version",
00085 |     "datasets",
00086 |     "reference_artifacts",
00087 |     "models",
00088 |     "selection",
00089 |     "early_stopping",
00090 |     "search_policy",
00091 |     "budget_policy",
00092 |     "final_seeds",
00093 |     "top_k_selection",
00094 |     "final_evaluation",
00095 |     "final_test_policy",
00096 |     "required_telemetry",
00097 |     "deeph_comparability",
00098 | }
00099 | PROTOCOL_GRAPH2MAT_KEYS = GRAPH2MAT_KEYS | {
00100 |     "optim_lr",
00101 |     "store_in_memory",
00102 |     "accelerator",
00103 |     "log_every_n_steps",
00104 | }
00105 | 
00106 | 
00107 | def _require_object(value: Any, *, field: str) -> dict[str, Any]:
00108 |     if not isinstance(value, dict):
00109 |         raise RuntimeError(f"{field} must be an object.")
00110 |     return value
00111 | 
00112 | 
00113 | def _require_list(value: Any, *, field: str) -> list[Any]:
00114 |     if not isinstance(value, list):
00115 |         raise RuntimeError(f"{field} must be a list.")
00116 |     if not value:
00117 |         raise RuntimeError(f"{field} must not be empty.")
00118 |     return value
00119 | 
00120 | 
00121 | def _require_nonempty_string(value: Any, *, field: str) -> str:
00122 |     if value is None:
00123 |         raise RuntimeError(f"{field} is required.")
00124 |     text = str(value).strip()
00125 |     if not text:
00126 |         raise RuntimeError(f"{field} must not be empty.")
00127 |     return text
00128 | 
00129 | 
00130 | def _require_positive_int(value: Any, *, field: str) -> int:
00131 |     if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
00132 |         raise RuntimeError(f"{field} must be a positive integer.")
00133 |     return value
00134 | 
00135 | 
00136 | def _require_nonnegative_number(value: Any, *, field: str) -> float:
00137 |     if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
00138 |         raise RuntimeError(f"{field} must be a non-negative number.")
00139 |     return float(value)
00140 | 
00141 | 
00142 | def _contains_test_scope(value: Any) -> bool:
00143 |     if isinstance(value, str):
00144 |         return value.strip().lower() == "test"
00145 |     if isinstance(value, dict):
00146 |         return any(_contains_test_scope(item) for item in value.values())
00147 |     if isinstance(value, list):
00148 |         return any(_contains_test_scope(item) for item in value)
00149 |     return False
00150 | 
00151 | 
00152 | def _numeric_values_from_space_spec(value: Any) -> list[float]:
00153 |     if isinstance(value, dict):
00154 |         if "choices" in value and isinstance(value["choices"], list):
00155 |             raw_values = value["choices"]
00156 |         elif "value" in value:
00157 |             raw_values = [value["value"]]
00158 |         elif "fixed" in value:
00159 |             raw_values = [value["fixed"]]
00160 |         else:
00161 |             raw_values = [value.get("min", value.get("low"))]
00162 |     elif isinstance(value, list):
00163 |         raw_values = value
00164 |     else:
00165 |         raw_values = [value]
00166 |     numbers: list[float] = []
00167 |     for item in raw_values:
00168 |         if isinstance(item, bool):
00169 |             continue
00170 |         try:
00171 |             numbers.append(float(item))
00172 |         except (TypeError, ValueError):
00173 |             continue
00174 |     return numbers
00175 | 
00176 | 
00177 | def _values_from_space_spec(value: Any) -> list[Any]:
00178 |     if isinstance(value, dict):
00179 |         if "choices" in value and isinstance(value["choices"], list):
00180 |             return list(value["choices"])
00181 |         if "value" in value:
00182 |             return [value["value"]]
00183 |         if "fixed" in value:
00184 |             return [value["fixed"]]
00185 |         return []
00186 |     if isinstance(value, list):
00187 |         return list(value)
00188 |     return [value]
00189 | 
00190 | 
00191 | def protocol_hash(protocol: dict[str, Any]) -> str:
00192 |     """Return a stable hash for a validated or raw protocol dictionary."""
00193 |     payload = copy.deepcopy(protocol)
00194 |     payload.pop("protocol_hash", None)
00195 |     encoded = json.dumps(json_safe(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
00196 |     return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
00197 | 
00198 | 
00199 | def _validate_datasets(protocol: dict[str, Any]) -> None:
00200 |     datasets = _require_list(protocol.get("datasets"), field="datasets")
00201 |     seen: set[str] = set()
00202 |     for index, item in enumerate(datasets):
00203 |         dataset = _require_object(item, field=f"datasets[{index}]")
00204 |         dataset_id = _require_nonempty_string(dataset.get("dataset_id"), field=f"datasets[{index}].dataset_id")
00205 |         if dataset_id in seen:
00206 |             raise RuntimeError(f"Duplicate protocol dataset_id: {dataset_id}")
00207 |         seen.add(dataset_id)
00208 |         for key in ("dataset_root", "frozen_split_manifest", "benchmark_dataset_manifest"):
00209 |             _require_nonempty_string(dataset.get(key), field=f"datasets[{index}].{key}")
00210 | 
00211 | 
00212 | def _validate_reference_artifacts(protocol: dict[str, Any]) -> None:
00213 |     section = _require_object(protocol.get("reference_artifacts"), field="reference_artifacts")
00214 |     required = set(str(item) for item in _require_list(section.get("required"), field="reference_artifacts.required"))
00215 |     missing = sorted(REQUIRED_REFERENCE_ARTIFACTS - required)
00216 |     if missing:
00217 |         raise RuntimeError("reference_artifacts.required is missing: " + ", ".join(missing))
00218 |     forbidden_reference = str(section.get("forbid_as_reference") or "")
00219 |     if "ML_prediction.HSX" not in forbidden_reference and "ML_prediction.HSX" not in set(section.get("forbidden", [])):
00220 |         raise RuntimeError("reference_artifacts must explicitly forbid ML_prediction.HSX as reference.")
00221 | 
00222 | 
00223 | def _validate_models(protocol: dict[str, Any]) -> None:
00224 |     models = _require_object(protocol.get("models"), field="models")
00225 |     missing = sorted(ALLOWED_MODELS - set(models))
00226 |     if missing:
00227 |         raise RuntimeError("models is missing required model sections: " + ", ".join(missing))
00228 |     for model_name in sorted(ALLOWED_MODELS):
00229 |         model = _require_object(models.get(model_name), field=f"models.{model_name}")
00230 |         if model.get("enabled") is not True:
00231 |             raise RuntimeError(f"models.{model_name}.enabled must be true for the final benchmark protocol.")
00232 |         search_space = _require_object(model.get("search_space"), field=f"models.{model_name}.search_space")
00233 |         if not search_space:
00234 |             raise RuntimeError(f"models.{model_name}.search_space must not be empty.")
00235 |         allowed = PROTOCOL_GRAPH2MAT_KEYS if model_name == "graph2mat" else DEEPH_KEYS
00236 |         forbidden = sorted(set(search_space) & FORBIDDEN_DEEPH_KEYS) if model_name == "deeph" else []
00237 |         if forbidden:
00238 |             raise RuntimeError(
00239 |                 "models.deeph.search_space cannot change split/preprocess/physics keys: "
00240 |                 + ", ".join(forbidden)
00241 |             )
00242 |         unknown = sorted(set(search_space) - allowed - FORBIDDEN_DEEPH_KEYS)
00243 |         if unknown:
00244 |             raise RuntimeError(f"models.{model_name}.search_space has unsupported keys: {', '.join(unknown)}.")
00245 |         if model_name == "graph2mat":
00246 |             batch_values = _numeric_values_from_space_spec(search_space.get("batch_size"))
00247 |             if not batch_values or min(batch_values) > 128:
00248 |                 raise RuntimeError(
00249 |                     "models.graph2mat.search_space.batch_size must include at least one small/medium "
00250 |                     "batch size <= 128 for paper-ready fairness."
00251 |                 )
00252 |             if "readout" in search_space:
00253 |                 raw_readout_values = _values_from_space_spec(search_space.get("readout"))
00254 |                 if not raw_readout_values:
00255 |                     raise RuntimeError(
00256 |                         "models.graph2mat.search_space.readout must use choices, value, or fixed."
00257 |                     )
00258 |                 readout_values = {
00259 |                     str(item).strip().lower()
00260 |                     for item in raw_readout_values
00261 |                     if str(item).strip()
00262 |                 }
00263 |                 unsupported = sorted(readout_values - GRAPH2MAT_READOUT_FAMILIES)
00264 |                 if unsupported:
00265 |                     raise RuntimeError(
00266 |                         "models.graph2mat.search_space.readout has unsupported values: "
00267 |                         + ", ".join(unsupported)
00268 |                         + ". Use one of: "
00269 |                         + ", ".join(sorted(GRAPH2MAT_READOUT_FAMILIES))
00270 |                         + "."
00271 |                     )
00272 |         else:
00273 |             if "optimizer" in search_space:
00274 |                 raw_optimizer_values = _values_from_space_spec(search_space.get("optimizer"))
00275 |                 optimizer_values = {
00276 |                     str(item).strip()
00277 |                     for item in raw_optimizer_values
00278 |                     if str(item).strip()
00279 |                 }
00280 |                 unsupported = sorted(optimizer_values - SUPPORTED_DEEPH_OPTIMIZERS)
00281 |                 if unsupported:
00282 |                     raise RuntimeError(
00283 |                         "models.deeph.search_space.optimizer has unsupported values for the "
00284 |                         "current DeepH-pack hamiltonian target: "
00285 |                         + ", ".join(unsupported)
00286 |                         + ". Use one of: "
00287 |                         + ", ".join(sorted(SUPPORTED_DEEPH_OPTIMIZERS))
00288 |                         + "."
00289 |                     )
00290 |             if "criterion" in search_space:
00291 |                 raw_criterion_values = _values_from_space_spec(search_space.get("criterion"))
00292 |                 criterion_values = {
00293 |                     str(item).strip()
00294 |                     for item in raw_criterion_values
00295 |                     if str(item).strip()
00296 |                 }
00297 |                 unsupported = sorted(criterion_values - SUPPORTED_DEEPH_CRITERIA)
00298 |                 if unsupported:
00299 |                     raise RuntimeError(
00300 |                         "models.deeph.search_space.criterion has unsupported values for the "
00301 |                         "current DeepH-pack hamiltonian target: "
00302 |                         + ", ".join(unsupported)
00303 |                         + ". Use one of: "
00304 |                         + ", ".join(sorted(SUPPORTED_DEEPH_CRITERIA))
00305 |                         + "."
00306 |                     )
00307 | 
00308 | 
00309 | def _validate_selection(protocol: dict[str, Any]) -> str:
00310 |     section = _require_object(protocol.get("selection"), field="selection")
00311 |     metric = _require_nonempty_string(section.get("metric"), field="selection.metric")
00312 |     mode = _require_nonempty_string(section.get("mode"), field="selection.mode")
00313 |     if mode not in ALLOWED_MODES:
00314 |         raise RuntimeError(f"selection.mode must be one of: {', '.join(sorted(ALLOWED_MODES))}.")
00315 |     split = _require_nonempty_string(section.get("split"), field="selection.split")
00316 |     if split != "validation":
00317 |         raise RuntimeError("selection.split must be validation; test metrics cannot select configs.")
00318 |     source = _require_nonempty_string(section.get("source"), field="selection.source")
00319 |     if source != "validation_only":
00320 |         raise RuntimeError("selection.source must be validation_only.")
00321 |     if _contains_test_scope(section):
00322 |         raise RuntimeError("selection must not reference test metrics.")
00323 |     return metric
00324 | 
00325 | 
00326 | def _validate_early_stopping(protocol: dict[str, Any], *, selection_metric: str) -> None:
00327 |     section = _require_object(protocol.get("early_stopping"), field="early_stopping")
00328 |     metric = _require_nonempty_string(section.get("metric"), field="early_stopping.metric")
00329 |     if metric != selection_metric and not (
00330 |         selection_metric in VALIDATION_COMPOSITE_METRICS
00331 |         and metric in {"val_loss", "validation_loss"}
00332 |     ):
00333 |         raise RuntimeError("early_stopping.metric must match selection.metric for the final protocol.")
00334 |     mode = _require_nonempty_string(section.get("mode"), field="early_stopping.mode")
00335 |     if mode not in ALLOWED_MODES:
00336 |         raise RuntimeError(f"early_stopping.mode must be one of: {', '.join(sorted(ALLOWED_MODES))}.")
00337 |     _require_positive_int(section.get("patience"), field="early_stopping.patience")
00338 |     _require_nonnegative_number(section.get("min_delta"), field="early_stopping.min_delta")
00339 |     _require_positive_int(section.get("max_epochs"), field="early_stopping.max_epochs")
00340 | 
00341 | 
00342 | def _validate_search_policy(protocol: dict[str, Any]) -> None:
00343 |     section = _require_object(protocol.get("search_policy"), field="search_policy")
00344 |     strategy = _require_nonempty_string(section.get("strategy"), field="search_policy.strategy")
00345 |     if strategy not in ALLOWED_SEARCH_STRATEGIES:
00346 |         raise RuntimeError(
00347 |             "search_policy.strategy must be one of: " + ", ".join(sorted(ALLOWED_SEARCH_STRATEGIES)) + "."
00348 |         )
00349 |     if strategy == "manual":
00350 |         manual = _require_object(protocol.get("manual_search_plan"), field="manual_search_plan")
00351 |         rows = _require_list(manual.get("planned_runs"), field="manual_search_plan.planned_runs")
00352 |         for index, row in enumerate(rows):
00353 |             item = _require_object(row, field=f"manual_search_plan.planned_runs[{index}]")
00354 |             model = _require_nonempty_string(item.get("model"), field=f"manual_search_plan.planned_runs[{index}].model")
00355 |             if model not in ALLOWED_MODELS:
00356 |                 raise RuntimeError(f"manual_search_plan.planned_runs[{index}].model must be graph2mat or deeph.")
00357 |             _require_nonempty_string(item.get("config_id") or item.get("id"), field=f"manual_search_plan.planned_runs[{index}].config_id")
00358 |             overrides = _require_object(item.get("overrides"), field=f"manual_search_plan.planned_runs[{index}].overrides")
00359 |             if model == "deeph" and "criterion" in overrides:
00360 |                 criterion = str(overrides.get("criterion") or "").strip()
00361 |                 if criterion not in SUPPORTED_DEEPH_CRITERIA:
00362 |                     raise RuntimeError(
00363 |                         f"manual_search_plan.planned_runs[{index}].overrides.criterion "
00364 |                         "has unsupported values for the current DeepH-pack hamiltonian target: "
00365 |                         f"{criterion!r}. Use one of: {', '.join(sorted(SUPPORTED_DEEPH_CRITERIA))}."
00366 |                     )
00367 |             if model == "deeph" and "optimizer" in overrides:
00368 |                 optimizer = str(overrides.get("optimizer") or "").strip()
00369 |                 if optimizer not in SUPPORTED_DEEPH_OPTIMIZERS:
00370 |                     raise RuntimeError(
00371 |                         f"manual_search_plan.planned_runs[{index}].overrides.optimizer "
00372 |                         "has unsupported values for the current DeepH-pack hamiltonian target: "
00373 |                         f"{optimizer!r}. Use one of: {', '.join(sorted(SUPPORTED_DEEPH_OPTIMIZERS))}."
00374 |                     )
00375 |     elif strategy in {"random", "latin_hypercube"}:
00376 |         _require_positive_int(section.get("n_trials_per_model"), field="search_policy.n_trials_per_model")
00377 |         if "random_seed" not in section:
00378 |             raise RuntimeError("search_policy.random_seed is required for randomized search strategies.")
00379 |     elif not any(key in section for key in ("max_configs_per_model", "n_trials_per_model", "gpu_hours_per_model")):
00380 |         raise RuntimeError("search_policy.grid requires max_configs_per_model, n_trials_per_model, or gpu_hours_per_model.")
00381 | 
00382 | 
00383 | def _validate_budget_policy(protocol: dict[str, Any]) -> None:
00384 |     section = _require_object(protocol.get("budget_policy"), field="budget_policy")
00385 |     mode = _require_nonempty_string(section.get("mode"), field="budget_policy.mode")
00386 |     if mode not in ALLOWED_BUDGET_MODES:
00387 |         raise RuntimeError("budget_policy.mode must be equal_n_trials or equal_gpu_hours_per_model.")
00388 |     if mode == "equal_n_trials":
00389 |         _require_positive_int(section.get("n_trials_per_model"), field="budget_policy.n_trials_per_model")
00390 |     else:
00391 |         value = section.get("gpu_hours_per_model")
00392 |         if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
00393 |             raise RuntimeError("budget_policy.gpu_hours_per_model must be a positive number.")
00394 | 
00395 | 
00396 | def _validate_final_seeds(protocol: dict[str, Any]) -> None:
00397 |     seeds = _require_list(protocol.get("final_seeds"), field="final_seeds")
00398 |     if len(seeds) < 3:
00399 |         raise RuntimeError("final_seeds must include at least 3 seeds for the final benchmark protocol.")
00400 |     for index, seed in enumerate(seeds):
00401 |         if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
00402 |             raise RuntimeError(f"final_seeds[{index}] must be a non-negative integer.")
00403 |     if len(set(seeds)) != len(seeds):
00404 |         raise RuntimeError("final_seeds must not contain duplicates.")
00405 | 
00406 | 
00407 | def _validate_top_k_selection(protocol: dict[str, Any], *, selection_metric: str) -> None:
00408 |     section = _require_object(protocol.get("top_k_selection"), field="top_k_selection")
00409 |     _require_positive_int(section.get("k_per_model"), field="top_k_selection.k_per_model")
00410 |     split = _require_nonempty_string(section.get("split"), field="top_k_selection.split")
00411 |     if split != "validation":
00412 |         raise RuntimeError("top_k_selection.split must be validation; test metrics cannot select top-k configs.")
00413 |     metric = _require_nonempty_string(section.get("metric"), field="top_k_selection.metric")
00414 |     if metric != selection_metric:
00415 |         raise RuntimeError("top_k_selection.metric must match selection.metric.")
00416 |     if section.get("uses_test_metrics") is not False:
00417 |         raise RuntimeError("top_k_selection.uses_test_metrics must be false.")
00418 |     if _contains_test_scope(section):
00419 |         raise RuntimeError("top_k_selection must not reference test metrics.")
00420 | 
00421 | 
00422 | def _validate_final_metric_name(metric: str, *, field: str) -> None:
00423 |     normalized = metric.strip()
00424 |     if normalized in DISALLOWED_FINAL_CLAIM_METRICS:
00425 |         raise RuntimeError(f"{field} must be a scientific final metric, not {normalized}.")
00426 |     if normalized not in ALLOWED_FINAL_CLAIM_METRICS:
00427 |         raise RuntimeError(
00428 |             f"{field} is unsupported for final scientific claims: {normalized}. "
00429 |             "Use one of: " + ", ".join(sorted(ALLOWED_FINAL_CLAIM_METRICS)) + "."
00430 |         )
00431 | 
00432 | 
00433 | def _validate_final_evaluation(protocol: dict[str, Any]) -> None:
00434 |     section = _require_object(protocol.get("final_evaluation"), field="final_evaluation")
00435 |     primary_metric = _require_nonempty_string(
00436 |         section.get("primary_metric"),
00437 |         field="final_evaluation.primary_metric",
00438 |     )
00439 |     _validate_final_metric_name(primary_metric, field="final_evaluation.primary_metric")
00440 |     mode = _require_nonempty_string(section.get("mode"), field="final_evaluation.mode")
00441 |     if mode not in ALLOWED_MODES:
00442 |         raise RuntimeError(f"final_evaluation.mode must be one of: {', '.join(sorted(ALLOWED_MODES))}.")
00443 |     secondary = section.get("secondary_metrics", [])
00444 |     if secondary is None:
00445 |         secondary = []
00446 |     if not isinstance(secondary, list):
00447 |         raise RuntimeError("final_evaluation.secondary_metrics must be a list.")
00448 |     for index, metric in enumerate(secondary):
00449 |         name = _require_nonempty_string(metric, field=f"final_evaluation.secondary_metrics[{index}]")
00450 |         _validate_final_metric_name(name, field=f"final_evaluation.secondary_metrics[{index}]")
00451 |     practical = section.get("practical_match")
00452 |     if practical is not None:
00453 |         practical_section = _require_object(practical, field="final_evaluation.practical_match")
00454 |         if practical_section.get("relative_gap_max") is not None:
00455 |             value = practical_section.get("relative_gap_max")
00456 |             if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
00457 |                 raise RuntimeError("final_evaluation.practical_match.relative_gap_max must be a positive number.")
00458 |         if practical_section.get("absolute_gap_meV_max") is not None:
00459 |             _require_nonnegative_number(
00460 |                 practical_section.get("absolute_gap_meV_max"),
00461 |                 field="final_evaluation.practical_match.absolute_gap_meV_max",
00462 |             )
00463 |         if "requires_cost_noninferior" in practical_section and not isinstance(
00464 |             practical_section.get("requires_cost_noninferior"),
00465 |             bool,
00466 |         ):
00467 |             raise RuntimeError("final_evaluation.practical_match.requires_cost_noninferior must be boolean.")
00468 | 
00469 | 
00470 | def _validate_final_test_policy(protocol: dict[str, Any]) -> None:
00471 |     section = _require_object(protocol.get("final_test_policy"), field="final_test_policy")
00472 |     policy = _require_nonempty_string(section.get("policy"), field="final_test_policy.policy")
00473 |     if policy not in ALLOWED_FINAL_TEST_POLICIES:
00474 |         raise RuntimeError("final_test_policy.policy must be locked_until_final.")
00475 |     if section.get("locked_during_search") is not True:
00476 |         raise RuntimeError("final_test_policy.locked_during_search must be true.")
00477 |     if section.get("evaluate_once_after_selection") is not True:
00478 |         raise RuntimeError("final_test_policy.evaluate_once_after_selection must be true.")
00479 |     test_split = _require_nonempty_string(section.get("test_split"), field="final_test_policy.test_split")
00480 |     if test_split != "test":
00481 |         raise RuntimeError("final_test_policy.test_split must be test.")
00482 | 
00483 | 
00484 | def _validate_required_telemetry(protocol: dict[str, Any]) -> None:
00485 |     fields = set(str(item) for item in _require_list(protocol.get("required_telemetry"), field="required_telemetry"))
00486 |     missing = sorted(REQUIRED_TELEMETRY_FIELDS - fields)
00487 |     if missing:
00488 |         raise RuntimeError("required_telemetry is missing: " + ", ".join(missing))
00489 | 
00490 | 
00491 | def _validate_deeph_comparability(protocol: dict[str, Any]) -> None:
00492 |     section = _require_object(protocol.get("deeph_comparability"), field="deeph_comparability")
00493 |     policy = _require_nonempty_string(
00494 |         section.get("adapter_equivalence_policy"),
00495 |         field="deeph_comparability.adapter_equivalence_policy",
00496 |     )
00497 |     if policy not in ALLOWED_DEEPH_EQUIVALENCE_POLICIES:
00498 |         raise RuntimeError("deeph_comparability.adapter_equivalence_policy must be fail_closed_unless_proven.")
00499 |     if section.get("robust_winner_requires_proven_equivalence") is not True:
00500 |         raise RuntimeError("deeph_comparability.robust_winner_requires_proven_equivalence must be true.")
00501 |     if section.get("diagnostic_if_unproven") is not True:
00502 |         raise RuntimeError("deeph_comparability.diagnostic_if_unproven must be true.")
00503 | 
00504 | 
00505 | def validate_protocol(value: Any) -> dict[str, Any]:
00506 |     """Validate and return a canonical copy of a final benchmark protocol."""
00507 |     protocol = copy.deepcopy(_require_object(value, field="protocol"))
00508 |     missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(protocol))
00509 |     if missing:
00510 |         raise RuntimeError("Missing protocol fields: " + ", ".join(missing))
00511 |     protocol_id = _require_nonempty_string(protocol.get("protocol_id"), field="protocol_id")
00512 |     version = _require_nonempty_string(protocol.get("version"), field="version")
00513 |     protocol["protocol_id"] = protocol_id
00514 |     protocol["version"] = version
00515 |     if "schema" in protocol and protocol["schema"] != SCHEMA_NAME:
00516 |         raise RuntimeError(f"schema must be {SCHEMA_NAME}.")
00517 | 
00518 |     _validate_datasets(protocol)
00519 |     _validate_reference_artifacts(protocol)
00520 |     _validate_models(protocol)
00521 |     selection_metric = _validate_selection(protocol)
00522 |     _validate_early_stopping(protocol, selection_metric=selection_metric)
00523 |     _validate_search_policy(protocol)
00524 |     _validate_budget_policy(protocol)
00525 |     _validate_final_seeds(protocol)
00526 |     _validate_top_k_selection(protocol, selection_metric=selection_metric)
00527 |     _validate_final_evaluation(protocol)
00528 |     _validate_final_test_policy(protocol)
00529 |     _validate_required_telemetry(protocol)
00530 |     _validate_deeph_comparability(protocol)
00531 | 
00532 |     protocol["schema"] = SCHEMA_NAME
00533 |     protocol["protocol_hash"] = protocol_hash(protocol)
00534 |     return protocol
00535 | 
00536 | 
00537 | def load_protocol(path: Path | str) -> dict[str, Any]:
00538 |     path = Path(path)
00539 |     with path.open("r", encoding="utf-8") as handle:
00540 |         payload = json.load(handle)
00541 |     return validate_protocol(payload)
00542 | 
00543 | 
00544 | def main(argv: list[str] | None = None) -> int:
00545 |     parser = argparse.ArgumentParser(description="Validate a Graph2Mat-vs-DeepH paper benchmark protocol JSON.")
00546 |     parser.add_argument("protocol", type=Path, help="Path to the protocol JSON file.")
00547 |     parser.add_argument("--print-json", action="store_true", help="Print the canonical validated protocol JSON.")
00548 |     args = parser.parse_args(argv)
00549 | 
00550 |     protocol = load_protocol(args.protocol)
00551 |     if args.print_json:
00552 |         print(json.dumps(protocol, indent=2, sort_keys=True))
00553 |     else:
00554 |         print(
00555 |             json.dumps(
00556 |                 {
00557 |                     "status": "valid",
00558 |                     "schema": protocol["schema"],
00559 |                     "protocol_id": protocol["protocol_id"],
00560 |                     "protocol_hash": protocol["protocol_hash"],
00561 |                 },
00562 |                 indent=2,
00563 |                 sort_keys=True,
00564 |             )
00565 |         )
00566 |     return 0
00567 | 
00568 | 
00569 | if __name__ == "__main__":
00570 |     raise SystemExit(main())
```

## `tests/test_g2m_deeph_protocol.py`

SHA-256: `fe6eea241d89aabfde9fbd95ab0336f5e35021188f4a58d4c00dc26e6b61ddd3`

```py
00001 | import copy
00002 | import sys
00003 | import unittest
00004 | from pathlib import Path
00005 | 
00006 | 
00007 | REPO_ROOT = Path(__file__).resolve().parents[1]
00008 | SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
00009 | if str(SCRIPTS_DIR) not in sys.path:
00010 |     sys.path.insert(0, str(SCRIPTS_DIR))
00011 | 
00012 | from g2m_deeph_protocol import SCHEMA_NAME, load_protocol, validate_protocol  # noqa: E402
00013 | 
00014 | 
00015 | def valid_protocol() -> dict:
00016 |     return {
00017 |         "protocol_id": "paper_protocol_unit",
00018 |         "version": "1.0",
00019 |         "datasets": [
00020 |             {
00021 |                 "dataset_id": "joint_a",
00022 |                 "dataset_root": "Comparison/datasets/joint_a",
00023 |                 "benchmark_dataset_manifest": "Comparison/datasets/joint_a/benchmark_dataset_manifest.json",
00024 |                 "frozen_split_manifest": "Comparison/datasets/joint_a/frozen_split_manifest.json",
00025 |             }
00026 |         ],
00027 |         "reference_artifacts": {
00028 |             "required": [
00029 |                 "RUN.fdf",
00030 |                 "SystemLabel.TSHS",
00031 |                 "SystemLabel.TSDE",
00032 |                 "SystemLabel.HSX",
00033 |                 "SystemLabel.STRUCT_OUT",
00034 |                 "SystemLabel.XV",
00035 |                 "SystemLabel.ORB_INDX",
00036 |                 "metadata.json",
00037 |             ],
00038 |             "forbidden": ["ML_prediction.HSX"],
00039 |             "forbid_as_reference": "ML_prediction.HSX",
00040 |         },
00041 |         "models": {
00042 |             "graph2mat": {
00043 |                 "enabled": True,
00044 |                 "search_space": {
00045 |                     "optim_lr": [0.0003, 0.001],
00046 |                     "batch_size": [64, 128, 256],
00047 |                     "max_epochs": [200],
00048 |                     "hidden_irreps": ["32x0e + 32x1o + 32x2e + 32x3o"],
00049 |                     "num_interactions": [3],
00050 |                     "correlation": [2],
00051 |                     "max_ell": [3],
00052 |                     "readout": {"choices": ["default", "edge_node_mix"]},
00053 |                 },
00054 |             },
00055 |             "deeph": {
00056 |                 "enabled": True,
00057 |                 "search_space": {
00058 |                     "learning_rate": [0.0001, 0.0003],
00059 |                     "batch_size": [4, 8],
00060 |                     "epochs": [200],
00061 |                     "atom_fea_len": [64],
00062 |                     "edge_fea_len": [128],
00063 |                     "num_l": [4],
00064 |                     "if_lcmp": [True],
00065 |                 },
00066 |             },
00067 |         },
00068 |         "selection": {
00069 |             "split": "validation",
00070 |             "metric": "low_energy_rmse_eV",
00071 |             "mode": "min",
00072 |             "source": "validation_only",
00073 |         },
00074 |         "early_stopping": {
00075 |             "metric": "low_energy_rmse_eV",
00076 |             "mode": "min",
00077 |             "patience": 30,
00078 |             "min_delta": 0.0,
00079 |             "max_epochs": 200,
00080 |         },
00081 |         "search_policy": {
00082 |             "strategy": "latin_hypercube",
00083 |             "n_trials_per_model": 20,
00084 |             "random_seed": 20260528,
00085 |         },
00086 |         "budget_policy": {
00087 |             "mode": "equal_gpu_hours_per_model",
00088 |             "gpu_hours_per_model": 24.0,
00089 |         },
00090 |         "final_seeds": [0, 1, 2],
00091 |         "top_k_selection": {
00092 |             "k_per_model": 2,
00093 |             "split": "validation",
00094 |             "metric": "low_energy_rmse_eV",
00095 |             "uses_test_metrics": False,
00096 |         },
00097 |         "final_evaluation": {
00098 |             "primary_metric": "low_energy_rmse_eV",
00099 |             "mode": "min",
00100 |             "secondary_metrics": [
00101 |                 "fermi_window_rmse_eV",
00102 |                 "frontier_window_rmse_eV",
00103 |                 "dos_wasserstein_eV",
00104 |                 "h_mae_eV",
00105 |             ],
00106 |             "practical_match": {
00107 |                 "relative_gap_max": 1.10,
00108 |                 "absolute_gap_meV_max": None,
00109 |                 "requires_cost_noninferior": True,
00110 |             },
00111 |         },
00112 |         "final_test_policy": {
00113 |             "policy": "locked_until_final",
00114 |             "test_split": "test",
00115 |             "locked_during_search": True,
00116 |             "evaluate_once_after_selection": True,
00117 |         },
00118 |         "required_telemetry": [
00119 |             "wall_clock_seconds",
00120 |             "gpu_hours",
00121 |             "peak_gpu_memory_mb",
00122 |             "samples_per_second",
00123 |             "matrix_blocks_per_second",
00124 |             "best_validation_epoch",
00125 |         ],
00126 |         "deeph_comparability": {
00127 |             "adapter_equivalence_policy": "fail_closed_unless_proven",
00128 |             "robust_winner_requires_proven_equivalence": True,
00129 |             "diagnostic_if_unproven": True,
00130 |         },
00131 |     }
00132 | 
00133 | 
00134 | class Graph2MatDeepHProtocolTests(unittest.TestCase):
00135 |     def test_valid_final_benchmark_protocol(self) -> None:
00136 |         protocol = validate_protocol(valid_protocol())
00137 | 
00138 |         self.assertEqual(protocol["schema"], SCHEMA_NAME)
00139 |         self.assertIn("protocol_hash", protocol)
00140 |         self.assertEqual(protocol["selection"]["split"], "validation")
00141 |         self.assertEqual(protocol["final_evaluation"]["primary_metric"], "low_energy_rmse_eV")
00142 | 
00143 |     def test_example_protocol_loads(self) -> None:
00144 |         protocol = load_protocol(REPO_ROOT / "Comparison" / "config" / "g2m_deeph_paper_protocol_v1_example.json")
00145 | 
00146 |         self.assertEqual(protocol["schema"], SCHEMA_NAME)
00147 |         self.assertEqual(protocol["budget_policy"]["mode"], "equal_n_trials")
00148 |         self.assertLessEqual(
00149 |             min(protocol["models"]["graph2mat"]["search_space"]["batch_size"]["choices"]),
00150 |             128,
00151 |         )
00152 | 
00153 |     def test_missing_required_fields_fail(self) -> None:
00154 |         protocol = valid_protocol()
00155 |         protocol.pop("final_test_policy")
00156 | 
00157 |         with self.assertRaisesRegex(RuntimeError, "Missing protocol fields: final_test_policy"):
00158 |             validate_protocol(protocol)
00159 | 
00160 |     def test_missing_final_evaluation_fails(self) -> None:
00161 |         protocol = valid_protocol()
00162 |         protocol.pop("final_evaluation")
00163 | 
00164 |         with self.assertRaisesRegex(RuntimeError, "Missing protocol fields: final_evaluation"):
00165 |             validate_protocol(protocol)
00166 | 
00167 |     def test_final_evaluation_rejects_validation_loss(self) -> None:
00168 |         protocol = valid_protocol()
00169 |         protocol["selection"]["metric"] = "val_loss"
00170 |         protocol["early_stopping"]["metric"] = "val_loss"
00171 |         protocol["top_k_selection"]["metric"] = "val_loss"
00172 |         protocol["final_evaluation"]["primary_metric"] = "val_loss"
00173 | 
00174 |         with self.assertRaisesRegex(RuntimeError, "final_evaluation.primary_metric"):
00175 |             validate_protocol(protocol)
00176 | 
00177 |     def test_final_evaluation_rejects_unsupported_metrics(self) -> None:
00178 |         protocol = valid_protocol()
00179 |         protocol["final_evaluation"]["primary_metric"] = "private_dashboard_metric"
00180 | 
00181 |         with self.assertRaisesRegex(RuntimeError, "unsupported for final scientific claims"):
00182 |             validate_protocol(protocol)
00183 | 
00184 |     def test_invalid_budget_policy_fails(self) -> None:
00185 |         protocol = valid_protocol()
00186 |         protocol["budget_policy"] = {"mode": "equal_gpu_hours_per_model"}
00187 | 
00188 |         with self.assertRaisesRegex(RuntimeError, "gpu_hours_per_model"):
00189 |             validate_protocol(protocol)
00190 | 
00191 |     def test_deeph_search_space_rejects_plain_mse_loss(self) -> None:
00192 |         protocol = valid_protocol()
00193 |         protocol["models"]["deeph"]["search_space"]["criterion"] = {"value": "MSELoss"}
00194 | 
00195 |         with self.assertRaisesRegex(RuntimeError, "models.deeph.search_space.criterion"):
00196 |             validate_protocol(protocol)
00197 | 
00198 |     def test_deeph_search_space_rejects_wrong_optimizer_spelling(self) -> None:
00199 |         protocol = valid_protocol()
00200 |         protocol["models"]["deeph"]["search_space"]["optimizer"] = {"value": "AdamW"}
00201 | 
00202 |         with self.assertRaisesRegex(RuntimeError, "models.deeph.search_space.optimizer"):
00203 |             validate_protocol(protocol)
00204 | 
00205 |     def test_deeph_manual_plan_rejects_plain_mse_loss(self) -> None:
00206 |         protocol = valid_protocol()
00207 |         protocol["search_policy"] = {"strategy": "manual", "random_seed": 1}
00208 |         protocol["manual_search_plan"] = {
00209 |             "planned_runs": [
00210 |                 {
00211 |                     "model": "deeph",
00212 |                     "config_id": "DH-bad",
00213 |                     "overrides": {
00214 |                         "batch_size": 4,
00215 |                         "learning_rate": 0.0001,
00216 |                         "criterion": "MSELoss",
00217 |                     },
00218 |                 }
00219 |             ]
00220 |         }
00221 | 
00222 |         with self.assertRaisesRegex(RuntimeError, "overrides.criterion"):
00223 |             validate_protocol(protocol)
00224 | 
00225 |     def test_deeph_manual_plan_rejects_wrong_optimizer_spelling(self) -> None:
00226 |         protocol = valid_protocol()
00227 |         protocol["search_policy"] = {"strategy": "manual", "random_seed": 1}
00228 |         protocol["manual_search_plan"] = {
00229 |             "planned_runs": [
00230 |                 {
00231 |                     "model": "deeph",
00232 |                     "config_id": "DH-bad",
00233 |                     "overrides": {
00234 |                         "batch_size": 4,
00235 |                         "learning_rate": 0.0001,
00236 |                         "criterion": "MaskMSELoss",
00237 |                         "optimizer": "AdamW",
00238 |                     },
00239 |                 }
00240 |             ]
00241 |         }
00242 | 
00243 |         with self.assertRaisesRegex(RuntimeError, "overrides.optimizer"):
00244 |             validate_protocol(protocol)
00245 | 
00246 |         protocol = valid_protocol()
00247 |         protocol["budget_policy"] = {"mode": "same_number_of_epochs", "n_trials_per_model": 10}
00248 |         with self.assertRaisesRegex(RuntimeError, "budget_policy.mode"):
00249 |             validate_protocol(protocol)
00250 | 
00251 |     def test_invalid_early_stopping_policy_fails(self) -> None:
00252 |         protocol = valid_protocol()
00253 |         protocol["early_stopping"]["patience"] = 0
00254 | 
00255 |         with self.assertRaisesRegex(RuntimeError, "early_stopping.patience"):
00256 |             validate_protocol(protocol)
00257 | 
00258 |         protocol = valid_protocol()
00259 |         protocol["early_stopping"]["metric"] = "training_loss"
00260 |         with self.assertRaisesRegex(RuntimeError, "early_stopping.metric must match selection.metric"):
00261 |             validate_protocol(protocol)
00262 | 
00263 |     def test_validation_spectral_composite_allows_val_loss_early_stopping(self) -> None:
00264 |         protocol = valid_protocol()
00265 |         protocol["selection"]["metric"] = "val_spectral_composite"
00266 |         protocol["top_k_selection"]["metric"] = "val_spectral_composite"
00267 |         protocol["early_stopping"]["metric"] = "val_loss"
00268 |         protocol["search_policy"] = {"strategy": "manual", "random_seed": 20260529, "max_runs": 2}
00269 |         protocol["manual_search_plan"] = {
00270 |             "planned_runs": [
00271 |                 {
00272 |                     "model": "graph2mat",
00273 |                     "config_id": "G2M-A01",
00274 |                     "overrides": {"batch_size": 32, "optim_lr": 0.003},
00275 |                 },
00276 |                 {
00277 |                     "model": "deeph",
00278 |                     "config_id": "DH-A01",
00279 |                     "overrides": {"batch_size": 4, "learning_rate": 3e-5},
00280 |                 },
00281 |             ]
00282 |         }
00283 |         protocol["final_evaluation"]["primary_metric"] = "test_spectral_composite_score"
00284 | 
00285 |         validated = validate_protocol(protocol)
00286 | 
00287 |         self.assertEqual(validated["selection"]["metric"], "val_spectral_composite")
00288 |         self.assertEqual(validated["early_stopping"]["metric"], "val_loss")
00289 |         self.assertEqual(validated["search_policy"]["strategy"], "manual")
00290 |         self.assertEqual(validated["final_evaluation"]["primary_metric"], "test_spectral_composite_score")
00291 | 
00292 |     def test_manual_search_strategy_requires_preregistered_plan(self) -> None:
00293 |         protocol = valid_protocol()
00294 |         protocol["search_policy"] = {"strategy": "manual", "random_seed": 20260529}
00295 | 
00296 |         with self.assertRaisesRegex(RuntimeError, "manual_search_plan"):
00297 |             validate_protocol(protocol)
00298 | 
00299 |     def test_invalid_final_test_policy_fails(self) -> None:
00300 |         protocol = valid_protocol()
00301 |         protocol["final_test_policy"]["locked_during_search"] = False
00302 | 
00303 |         with self.assertRaisesRegex(RuntimeError, "locked_during_search"):
00304 |             validate_protocol(protocol)
00305 | 
00306 |     def test_model_specific_search_spaces_may_differ(self) -> None:
00307 |         protocol = validate_protocol(valid_protocol())
00308 |         graph2mat_space = protocol["models"]["graph2mat"]["search_space"]
00309 |         deeph_space = protocol["models"]["deeph"]["search_space"]
00310 | 
00311 |         self.assertEqual(graph2mat_space["optim_lr"], [0.0003, 0.001])
00312 |         self.assertEqual(graph2mat_space["batch_size"], [64, 128, 256])
00313 |         self.assertEqual(graph2mat_space["readout"]["choices"], ["default", "edge_node_mix"])
00314 |         self.assertEqual(deeph_space["learning_rate"], [0.0001, 0.0003])
00315 |         self.assertNotEqual(graph2mat_space["batch_size"], deeph_space["batch_size"])
00316 | 
00317 |     def test_graph2mat_readout_search_space_rejects_unknown_family(self) -> None:
00318 |         protocol = valid_protocol()
00319 |         protocol["models"]["graph2mat"]["search_space"]["readout"] = {"choices": ["edge_node_mix", "mystery"]}
00320 | 
00321 |         with self.assertRaisesRegex(RuntimeError, "search_space.readout has unsupported values"):
00322 |             validate_protocol(protocol)
00323 | 
00324 |     def test_graph2mat_final_protocol_requires_small_batch_search(self) -> None:
00325 |         protocol = valid_protocol()
00326 |         protocol["models"]["graph2mat"]["search_space"]["batch_size"] = [512, 1024]
00327 | 
00328 |         with self.assertRaisesRegex(RuntimeError, "batch size <= 128"):
00329 |             validate_protocol(protocol)
00330 | 
00331 |     def test_test_metrics_cannot_select_configs(self) -> None:
00332 |         protocol = valid_protocol()
00333 |         protocol["selection"]["split"] = "test"
00334 | 
00335 |         with self.assertRaisesRegex(RuntimeError, "test metrics cannot select configs"):
00336 |             validate_protocol(protocol)
00337 | 
00338 |         protocol = valid_protocol()
00339 |         protocol["top_k_selection"]["uses_test_metrics"] = True
00340 |         with self.assertRaisesRegex(RuntimeError, "uses_test_metrics must be false"):
00341 |             validate_protocol(protocol)
00342 | 
00343 |     def test_deeph_equivalence_policy_is_fail_closed(self) -> None:
00344 |         protocol = validate_protocol(valid_protocol())
00345 | 
00346 |         self.assertEqual(
00347 |             protocol["deeph_comparability"]["adapter_equivalence_policy"],
00348 |             "fail_closed_unless_proven",
00349 |         )
00350 | 
00351 |         invalid = valid_protocol()
00352 |         invalid["deeph_comparability"]["adapter_equivalence_policy"] = "warn_only"
00353 |         with self.assertRaisesRegex(RuntimeError, "fail_closed_unless_proven"):
00354 |             validate_protocol(invalid)
00355 | 
00356 |     def test_deeph_search_space_rejects_split_and_physics_keys(self) -> None:
00357 |         protocol = valid_protocol()
00358 |         protocol["models"]["deeph"]["search_space"]["train_ratio"] = [0.8]
00359 | 
00360 |         with self.assertRaisesRegex(RuntimeError, "cannot change split/preprocess/physics"):
00361 |             validate_protocol(protocol)
00362 | 
00363 |     def test_required_telemetry_fields_must_be_present(self) -> None:
00364 |         protocol = valid_protocol()
00365 |         protocol["required_telemetry"].remove("gpu_hours")
00366 | 
00367 |         with self.assertRaisesRegex(RuntimeError, "required_telemetry is missing: gpu_hours"):
00368 |             validate_protocol(protocol)
00369 | 
00370 | 
00371 | if __name__ == "__main__":
00372 |     unittest.main()
```
