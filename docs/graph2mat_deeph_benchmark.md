# Graph2Mat vs DeepH Joint Benchmark

This document describes the controlled Graph2Mat vs DeepH workflow for
graphene Hamiltonians generated with the repository SIESTA/Wannier90 stack.
The goal is not to reproduce the DeepH paper exactly. The goal is a
cross-architecture benchmark where the DFT backend, basis, pseudopotentials,
snapshots, splits, references, overlap policy, and metrics are fixed.

## Why The Joint Artifact Contract Exists

Graph2Mat can train from SIESTA snapshots that contain only the artifacts used
by the Graph2Mat path. DeepH's SIESTA preprocessing needs a different raw
artifact set. The joint benchmark therefore uses the versioned contract:

```text
joint_graph2mat_deeph_artifact_contract_v1
```

A snapshot is benchmark-ready only when one SIESTA calculation produced and
archived every artifact needed by both methods. Missing artifacts are a
scientific blocker, not a warning to ignore.

## Old Failure Mode

The previous Graph2Mat-only dataset archive stored snapshots like:

```text
RUN.fdf
graphene.TSHS
graphene.TSDE
graphene.XV
metadata.json
```

That was enough for the Graph2Mat/evaluator path, but not enough for DeepH.
DeepH with the SIESTA interface requires at least:

```text
graphene.HSX
graphene.STRUCT_OUT
graphene.XV
graphene.ORB_INDX
```

Because `graphene.HSX`, `graphene.STRUCT_OUT`, and `graphene.ORB_INDX` were
not archived, the recovery path had to rerun SIESTA once per snapshot. On the
1500-snapshot graphene dataset this cost roughly 3.8-3.9 minutes per snapshot,
nearly 100 sequential hours.

The fix is one-pass artifact generation: run SIESTA once per snapshot during
dataset generation and archive all Graph2Mat and DeepH artifacts immediately.
The normal benchmark workflow must not silently rerun SIESTA to repair missing
DeepH files. Any repair path must be explicit, opt-in, and visibly marked
slow/expensive.

## Required Per-Snapshot Artifacts

A joint Graph2Mat+DeepH benchmark snapshot must contain:

```text
RUN.fdf
RUN.out or siesta.out
SystemLabel.TSHS
SystemLabel.TSDE
SystemLabel.HSX
SystemLabel.STRUCT_OUT
SystemLabel.XV
SystemLabel.ORB_INDX
metadata.json
```

Dataset-level provenance must also make the material identity traceable and
must include basis and pseudopotential provenance or hashes when required by
the workflow. For graphene the `SystemLabel` is normally `graphene`, but the
validator resolves the label from real dataset metadata and FDF content rather
than assuming every dataset is graphene.

The validator writes or consumes:

```text
artifact_validation.json
benchmark_dataset_manifest.json
frozen_split_manifest.json
```

`benchmark_dataset_manifest.json` records the artifact contract and dataset
status. `frozen_split_manifest.json` records the exact train/validation/test
sample IDs and artifact hashes used by both Graph2Mat and DeepH.

## Dataset Modes

The UI and runner distinguish three dataset modes.

| Mode | Meaning | Scientific status |
| --- | --- | --- |
| Clean one-pass dataset | SIESTA generated all joint artifacts during the original dataset generation. | Preferred. Can be robust if all later compatibility checks pass. |
| Reused validated joint dataset | Existing dataset is validated against the joint artifact contract before training. | Valid reuse if manifests, hashes, splits, material, basis and pseudos are compatible. |
| Repaired dataset | Missing artifacts are regenerated only after explicit user opt-in. | Must be marked `valid_repaired_dataset_with_warning` or diagnostic/exploratory depending on provenance. |

If a Graph2Mat-only legacy dataset is missing `HSX`, `STRUCT_OUT`, or
`ORB_INDX`, the validator classifies it as invalid or repair-required. The
normal one-click benchmark does not repair it silently.

## Running A Smoke Benchmark

Use this for UI/API plumbing and manifest validation, not for publication
metrics.

The smallest backend smoke is the dedicated script:

```bash
python3 Comparison/scripts/g2m_deeph_smoke.py \
  --dry-run \
  --output-root Comparison/results/g2m_deeph_smoke_dry_run
```

The dry-run never launches SIESTA, Graph2Mat training, or DeepH. It exercises
the Graph2Mat vs DeepH runner in dataset-generation planning mode and writes:

```text
smoke_manifest.json
artifact_validation.json
benchmark_manifest.json
recommendation.json
logs/smoke.log
```

`artifact_validation.json` records the required joint snapshot contract:
`RUN.fdf`, `SystemLabel.TSHS`, `SystemLabel.TSDE`, `SystemLabel.HSX`,
`SystemLabel.STRUCT_OUT`, `SystemLabel.XV`, `SystemLabel.ORB_INDX`, and
`metadata.json`. In dry-run mode those files are not claimed to exist; the
smoke only verifies that the planned workflow knows the required artifact set.

A real tiny smoke is opt-in and should be run manually only on a machine with
the required executables:

```bash
RUN_G2M_DEEPH_REAL_SMOKE=1 \
SIESTA_COMMAND=siesta \
DEEPH_PACK_ROOT=/path/to/DeepH-pack \
python3 Comparison/scripts/g2m_deeph_smoke.py \
  --tiny-real \
  --sample-limit 6 \
  --epochs 1 \
  --output-root Comparison/results/g2m_deeph_smoke_real
```

The real smoke checks SIESTA one-pass joint artifact generation, Graph2Mat,
DeepH preprocess/train/inference, fail-closed metrics, manifests, and an honest
diagnostic/recommendation summary. If `RUN_G2M_DEEPH_REAL_SMOKE=1` or any
dependency is missing, the command writes `status: skipped` rather than
pretending the smoke passed.

1. Start the UI:

   ```bash
   python3 Comparison/scripts/pipeline_ui.py
   ```

2. Open the `G2M vs DeepH` tab.

3. Select `Reuse existing validated joint dataset`.

4. Point `Dataset root` to a tiny dataset that already contains the joint
   artifacts and manifests. The default persistent location is
   `Comparison/datasets/graphene_w90_joint`, separated from `results/` and
   `workspaces/`.

5. Click `Validate dataset artifacts`. The artifact table must show all
   snapshots valid and no missing `HSX`, `STRUCT_OUT`, `XV`, or `ORB_INDX`.

6. For a backend dry-run smoke through the API, keep the UI server running and
   use:

   ```bash
   curl -sS -X POST http://127.0.0.1:8770/api/g2m-deeph/run \
     -H 'Content-Type: application/json' \
     -d '{
       "dataset_root": "Comparison/datasets/graphene_w90_joint",
       "system_label": "graphene",
       "dry_run": true,
       "graph2mat_overrides": {"max_epochs": 1},
       "deeph": {"epochs": 1, "batch_size": 1}
     }'
   ```

The dry-run checks runner wiring and writes local configs/manifests where the
runner phase supports it, but it does not prove model accuracy.

## Running A Serious Benchmark

For a real comparison:

1. Generate a new joint dataset with the MD generation path or reuse a dataset
   that already passed the joint artifact contract. The generation path must
   use the expanded SIESTA store list and must validate artifacts before
   splits/training:

   ```bash
   PIPELINE_CONFIG_PATH=MD/pipeline_config.yaml \
     python3 MD/scripts/generate_md_dataset.py
   ```

   For production, use a copied or experiment-specific config rather than
   editing global defaults during an active run.

2. Store reusable joint datasets under `Comparison/datasets/`. The default
   graphene root used by the UI/backend is
   `Comparison/datasets/graphene_w90_joint`.

3. Confirm these files exist at the dataset root:

   ```text
   artifact_validation.json
   benchmark_dataset_manifest.json
   frozen_split_manifest.json
   ```

4. Start the UI and open `G2M vs DeepH`.

## Paper-Ready Protocol Contract

Exploratory sweeps remain useful for development, but the final/publicable
Graph2Mat-vs-DeepH comparison should be preregistered before looking at test
metrics. The machine-readable protocol contract is:

```text
graph2mat_deeph_benchmark_protocol_v1
```

Validate an example protocol with:

```bash
python3 Comparison/scripts/g2m_deeph_protocol.py \
  Comparison/config/g2m_deeph_paper_protocol_v1_example.json
```

The staged final workflow can also be driven from one CLI. The workflow root is
an artifact directory; every stage writes a `stages/<stage>.json` manifest and
refuses to continue when required previous artifacts are missing.

```bash
WORKFLOW_ROOT=Comparison/results/g2m_deeph_final_workflow
PROTOCOL=Comparison/config/g2m_deeph_paper_protocol_v1_example.json

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage validate-protocol \
  --protocol "$PROTOCOL" \
  --workflow-root "$WORKFLOW_ROOT"

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage generate-search-plan \
  --workflow-root "$WORKFLOW_ROOT"

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage run-search \
  --workflow-root "$WORKFLOW_ROOT"

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage select-top-k \
  --workflow-root "$WORKFLOW_ROOT"

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage generate-final-seeds \
  --workflow-root "$WORKFLOW_ROOT"

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage run-final \
  --workflow-root "$WORKFLOW_ROOT"

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage evaluate-final-test \
  --workflow-root "$WORKFLOW_ROOT" \
  --final-run-root "$WORKFLOW_ROOT/runs/<final-run-id>"

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage generate-report \
  --workflow-root "$WORKFLOW_ROOT" \
  --report-run-root "$WORKFLOW_ROOT/runs/<final-run-id>"
```

Use `--dry-run` with `run-search` or `run-final` to materialize the exact runner
payload without launching SIESTA, Graph2Mat, or DeepH. Search stages are
test-blind; `select-top-k` fails closed if search inputs contain test metrics.

The protocol records:

```text
protocol_id and version
datasets and frozen split manifests
joint SIESTA reference artifact requirements
Graph2Mat and DeepH model-specific search spaces
validation-only selection metric
common early stopping policy
search and budget policy
final seed list
top-k rule based only on validation metrics
locked final test policy
required telemetry fields
DeepH equivalence fail-closed policy
```

This contract intentionally does not force identical learning rates or batch
sizes. Those values are model-specific because Graph2Mat and DeepH have
different architectures and different memory/throughput behavior. Fairness is
expressed through shared datasets, splits, references, metrics, validation
selection, hardware class, telemetry, and either equal trials or equal
GPU-hour budgets.

Protocol search spaces can be expanded as the old finite grid, or sampled with
deterministic random/Latin-hypercube search. Sampled spaces support:

```text
{"choices": [...]}
{"value": ...} or {"fixed": ...}
{"type": "int", "min": ..., "max": ...}
{"distribution": "uniform", "min": ..., "max": ...}
{"distribution": "loguniform", "min": ..., "max": ...}
```

The runner writes the preregistered plan before launching jobs:

```text
sweep/search_plan.json
```

For sampled search this artifact records the strategy, random seed, protocol
id/hash, sampled dimensions, every planned config, and any duplicate sampled
configs produced by a tiny search space.

Search budget accounting is written to:

```text
sweep/budget_summary.json
```

For `equal_n_trials`, the scheduler reserves at most the configured number of
trials per model and reports actual GPU-hours consumed. For
`equal_gpu_hours_per_model`, the scheduler follows the preregistered search-plan
order and starts new trials only while the model has remaining consumed
GPU-hour budget. Because true trial cost is known only after completion, the
last scheduled trial, or scheduler batch when parallelism is enabled, may
overshoot the exact budget; the overshoot and exhaustion reason are reported.
Completed trials with missing `gpu_hours_total` make budget accounting fail
closed rather than being treated as zero cost.

The final test split is locked during search. Test metrics must not be used for
checkpoint selection, top-k selection, search-space refinement, or early
stopping. If DeepH prediction equivalence is not proven by the adapter, robust
winner claims remain fail-closed and DeepH-derived comparisons stay
diagnostic-only.

## Cost Telemetry

Training and evaluation runs write optional versioned cost telemetry:

```text
graph2mat_deeph_cost_telemetry_v1
```

For each completed Graph2Mat or DeepH config the runner writes:

```text
telemetry/graph2mat.json
telemetry/deeph.json
```

Training-sweep records also embed the same telemetry summary. The fields include
total wall-clock time, per-phase wall-clock time, GPU-hours when process-level
GPU activity can be observed, peak GPU memory, samples/s, matrix-blocks/s when
available, epochs trained, best validation epoch/value, hardware metadata, and
warnings for unavailable fields.

Telemetry is fail-transparent. CPU runs, missing `nvidia-smi`, missing
TensorBoard validation events, or unavailable matrix-block counts do not crash
the benchmark. They mark the telemetry as `partial` or `unavailable` and explain
which fields could not be collected. The runner never fabricates GPU-hours,
memory, or throughput values.

## Common Early Stopping

Production payloads can define a shared validation-only stopping policy:

```json
{
  "early_stopping": {
    "metric": "val_loss",
    "mode": "min",
    "patience": 30,
    "min_delta": 0.0,
    "max_epochs": 600
  }
}
```

When this policy is present, Graph2Mat receives a Lightning `EarlyStopping`
callback monitoring the configured validation metric and uses the same
`max_epochs`. DeepH does not expose a patience/min-delta INI API, so the runner
monitors DeepH's per-epoch validation-loss log lines and terminates the DeepH
process only after an epoch has completed and the common patience rule is
exhausted. DeepH's native threshold-based stopping is neutralized in this mode
so the benchmark policy is the active stopping rule.

Every completed run records early-stopping/checkpoint-selection metadata:

```text
validation_metric_name
metric_mode
patience
min_delta
max_epochs
best_epoch
best_validation_value
epochs_trained
stop_reason
```

If the configured validation metric is missing or non-finite, the run fails
closed. Test metrics are not accepted for early stopping or checkpoint
selection.

5. Use `Reuse existing validated joint dataset` unless you are intentionally
   generating a new dataset through an approved one-pass path.

6. Click `Validate dataset artifacts`.

7. Set Graph2Mat and DeepH training settings.

8. Click `Run Graph2Mat vs DeepH benchmark`.

9. Watch the phase progress, streaming logs, artifact table, timing table,
   metric summary and plots in the same tab.

The runner phases are:

```text
validate_inputs
generate_or_validate_joint_dataset
freeze_splits
graph2mat_train
graph2mat_predict
deeph_preprocess
deeph_train
deeph_predict
common_metrics
plots_and_summary
complete
```

## Interpreting Comparability Status

The final summary and plots expose a scientific status. A winner is allowed
only when the status is scientifically valid and no severe warning blocks the
recommendation.

Important status values:

| Status | Interpretation |
| --- | --- |
| `valid_joint_one_pass_dataset` | Preferred: artifacts were generated in one pass and checks passed. |
| `valid_reused_joint_dataset` | Existing dataset passed the joint contract and compatibility checks. |
| `valid_repaired_dataset_with_warning` | Repaired data may be useful, but provenance must be inspected carefully. |
| `invalid_missing_artifacts` | Missing required artifacts; do not compare. |
| `invalid_incompatible_splits` | Graph2Mat and DeepH did not use identical sample IDs/splits. |
| `invalid_incompatible_basis_or_pseudos` | Basis/material/pseudopotential provenance is incompatible or unknown. |
| `invalid_prediction_format` | Prediction outputs could not be compared safely. |
| `diagnostic_only` | Metrics may help debugging, but no robust winner is declared. |

If status is `diagnostic_only` or starts with `invalid_`, the UI must say
`No robust winner` even if one method has a lower numeric error.

## DeepH Equivalence Gate

DeepH prediction artifacts now carry a formal equivalence audit in addition to
the adapter-specific status:

| Field | Meaning |
| --- | --- |
| `equivalence_status` | One of `proven`, `failed`, `unproven`, or `not_applicable`. |
| `equivalence_scope` | The representation whose convention was checked, such as `raw_global` or `deeph_processed_blockwise_global_hdf5`. |
| `equivalence_evidence_paths` | Files used as evidence for the audit. |
| `equivalence_gate.robust_claim_allowed` | Whether DeepH can participate in robust winner claims. |

Only `equivalence_status=proven` with raw/global Hamiltonian equivalence allows
robust matrix-metric claims. Current DeepH HDF5 outputs may still appear in
diagnostic tables, but if raw/global HSX units, orbital order, R-vector
convention, and support semantics are not proven, ranking and final statistics
must remain fail-closed.

## Ground Truth And Overlap Policy

Ground truth always comes from SIESTA reference artifacts:

```text
SystemLabel.HSX
SystemLabel.TSHS
SystemLabel.STRUCT_OUT
SystemLabel.XV
SystemLabel.ORB_INDX
```

Never use `ML_prediction.HSX` or any DeepH/Graph2Mat prediction as ground
truth.

For non-orthogonal SIESTA references, official spectral and DOS metrics use
the SIESTA reference overlap, `S_ref` or `S_ref(k)`, for predicted spectra
whenever it is available. A prediction-owned overlap is used only if it is
explicitly validated within the evaluator tolerance. Unsafe predicted HSX files
are marked by fields such as:

```text
prediction_self_contained_hsx_safe
prediction_own_overlap_used
prediction_overlap_relative_frobenius_vs_reference
overlap_source
graph2mat_auxiliary_component_ignored
```

## What Remains Diagnostic-Only

Some outputs are intentionally diagnostic until additional physical equivalence
is validated:

- DeepH HDF5 predictions adapted through a local reader/converter are
  diagnostic-only if units, orbital order, R-vector convention, basis or frame
  equivalence is not fully proven.
- Repository raw/global Hamiltonian metrics are not exact DeepH local-frame
  H-prime metrics.
- High-symmetry k-path band comparisons are not implied by Monkhorst-Pack
  k-grid metrics.
- Unsupported spin-orbit, spinful, ambiguous multi-component or incompatible
  non-orthogonal cases must fail or become diagnostic-only.
- Missing Fermi level means near-Fermi and fixed-window DOS metrics are
  unavailable; the evaluator does not invent a substitute Fermi level.

The benchmark should be presented as:

```text
Controlled cross-architecture benchmark for learned Hamiltonians under a
unified SIESTA/W90 pipeline.
```

not as an exact reproduction of the DeepH paper.

## Paper-Ready Reports

After a search or final benchmark run, generate machine-readable learning-curve
and accuracy/cost summaries with:

```bash
python3 Comparison/scripts/g2m_deeph_report.py \
  Comparison/results/<benchmark_run_id> \
  --metric val_loss \
  --mode min
```

The report is written under `summary/report/` and includes:

- `learning_curve.csv/json`: validation error vs epoch, cumulative wall-clock,
  cumulative GPU-hours, samples seen and matrix blocks seen when available.
- `best_validation_summary.csv/json`: best validation epoch/value, GPU-hours
  to best validation, wall-clock to best validation and peak GPU memory.
- `pareto_accuracy_cost.csv/json`: accuracy/cost Pareto table with explicit
  dominated/frontier status.
- `final_comparison.json`: separate accuracy, compute and practical Pareto
  claims.

Search-stage reports ignore test metrics. If telemetry or per-epoch curves are
missing in older artifacts, the rows remain readable and the missing fields are
reported through explicit warning/status fields instead of being silently filled.

For final multi-seed test aggregation and winner gates, run:

```bash
python3 Comparison/scripts/g2m_deeph_final_stats.py \
  Comparison/results/<benchmark_run_id> \
  --metric low_energy_rmse_eV \
  --mode min \
  --expected-seeds 0,1,2,3,4 \
  --min-final-seeds 3
```

This writes `summary/final_statistics/` with seed mean/std/stderr, seed-level
confidence intervals, optional per-system bootstrap CIs, compute summaries, a
Pareto table and a fail-closed winner decision. Robust claims are blocked if
final seeds are incomplete, test metrics appear outside `final_test`, or DeepH
raw/global equivalence is not proven.
