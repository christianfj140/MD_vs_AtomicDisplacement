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

For the paper-ready control plane, use the synthetic staged workflow smoke:

```bash
python3 Comparison/scripts/g2m_deeph_smoke.py \
  --paper-workflow-dry-run \
  --output-dir Comparison/results/g2m_deeph_paper_workflow_smoke
```

This command creates a minimal synthetic joint dataset fixture and exercises the
final workflow wiring: protocol validation, dataset verification, search-plan
generation, dry-run search payload, validation-only top-k, final-seed plan,
synthetic final statistics, diagnostic report, gate checker, and release
manifest. It writes `smoke_summary.json` with
`scientific_status: not_a_scientific_run`. A successful control-plane smoke is
not benchmark evidence; it deliberately keeps robust claims blocked unless real
dataset manifests, telemetry, final statistics, and DeepH equivalence evidence
are supplied.

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

Before generating a search plan, verify every dataset declared by the protocol.
This command reads existing manifests only; it does not run SIESTA or repair
artifacts:

```bash
python3 Comparison/scripts/g2m_deeph_verify_protocol_datasets.py \
  --protocol Comparison/config/g2m_deeph_paper_protocol_v1_example.json \
  --output Comparison/results/g2m_deeph_dataset_verification.json \
  --strict
```

The verifier fails closed if `dataset_root`, `artifact_validation.json`,
`benchmark_dataset_manifest.json`, `frozen_split_manifest.json`, strict
SIESTA/environment provenance, non-empty train/validation/test splits, split
hash links, or forbidden-reference checks do not pass. To finalize manifests
from an existing `split_root`, use `--write-manifests` explicitly; without that
flag the verifier never creates or freezes dataset manifests.

The staged final workflow can also be driven from one CLI. The workflow root is
an artifact directory; every stage writes a `stages/<stage>.json` manifest and
refuses to continue when required previous artifacts are missing.

```bash
WORKFLOW_ROOT=Comparison/results/g2m_deeph_final_workflow
PROTOCOL=Comparison/config/g2m_deeph_paper_protocol_v1_example.json

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage validate-protocol \
  --protocol "$PROTOCOL" \
  --workflow-root "$WORKFLOW_ROOT" \
  --verify-datasets

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage generate-search-plan \
  --workflow-root "$WORKFLOW_ROOT"

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage run-search \
  --workflow-root "$WORKFLOW_ROOT" \
  --dataset-id <dataset-id>

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage select-top-k \
  --workflow-root "$WORKFLOW_ROOT"

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage generate-final-seeds \
  --workflow-root "$WORKFLOW_ROOT"

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage run-final \
  --workflow-root "$WORKFLOW_ROOT" \
  --dataset-id <dataset-id>

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage evaluate-final-test \
  --workflow-root "$WORKFLOW_ROOT" \
  --final-run-root "$WORKFLOW_ROOT/runs/<final-run-id>"

python3 Comparison/scripts/g2m_deeph_final_workflow.py \
  --stage generate-report \
  --workflow-root "$WORKFLOW_ROOT" \
  --report-run-root "$WORKFLOW_ROOT/runs/<final-run-id>"
```

Before any robust DeepH claim, generate numerical raw/global equivalence
evidence from the frozen samples. The preflight compares DeepH processed
`hamiltonians.h5`/`overlaps.h5` against the raw SIESTA/Graph2Mat HSX/TSHS
reference and writes adapter-discoverable
`raw_global_equivalence_evidence.json` files:

```bash
python3 Comparison/scripts/deeph_raw_global_equivalence_preflight.py \
  --frozen-split-manifest Comparison/datasets/<frozen-dataset-id>/frozen_split_manifest.json \
  --graph2mat-result-dir "$WORKFLOW_ROOT/runs/<final-run-id>" \
  --deeph-processed-dir "$WORKFLOW_ROOT/runs/<final-run-id>/deeph/processed" \
  --deeph-predictions-dir "$WORKFLOW_ROOT/runs/<final-run-id>/deeph/inference" \
  --sample-limit 5 \
  --output-dir "$WORKFLOW_ROOT/deeph_raw_global_equivalence"
```

If any required file or numerical check is missing, the evidence status is
`failed` and the DeepH adapter remains diagnostic-only. Do not hand-write this
evidence JSON; it must come from the preflight command and be included in the
release/evidence bundle.

After final statistics are available, run the fail-closed gate checker before
generating any claim-bearing final report or making any robust claim:

```bash
python3 Comparison/scripts/g2m_deeph_gate_check.py \
  --protocol "$PROTOCOL" \
  --workflow-root "$WORKFLOW_ROOT" \
  --run-root "$WORKFLOW_ROOT/runs/<final-run-id>" \
  --output "$WORKFLOW_ROOT/gate_status.json"
```

Only `claim_status=robust_allowed` and `robust_claim_allowed=true` allow a
paper-style winner claim. Any `invalid_*` or `diagnostic_only` status means the
run can be reported only as blocked/diagnostic, with the listed
`required_next_actions` used as the remediation checklist.

The repository intentionally keeps large generated matrices, predictions,
telemetry, and result directories out of git. Before archiving or sharing a
paper-ready run, write a release manifest that hashes every required external
file and flags missing or forbidden evidence:

```bash
python3 Comparison/scripts/g2m_deeph_release_manifest.py \
  --dataset-root Comparison/datasets/<frozen-dataset-id> \
  --run-root "$WORKFLOW_ROOT/runs/<final-run-id>" \
  --workflow-root "$WORKFLOW_ROOT" \
  --output "$WORKFLOW_ROOT/artifact_release_manifest.json" \
  --strict
```

In strict mode the manifest exits nonzero unless all required dataset manifests,
raw SIESTA artifacts, run telemetry, equivalence/adapter manifests, final
statistics, final report, and workflow evidence are present and hashable.
`ML_prediction.HSX` is allowed only as a Graph2Mat prediction artifact; if it
appears in any reference role, the release manifest is `invalid`.
Symlinks are hashable only when their resolved target remains under one of the
explicit roots passed on the command line; external symlinks are recorded as
unsafe missing evidence.

Use `--dry-run` with `run-search` or `run-final` to materialize the exact runner
payload without launching SIESTA, Graph2Mat, or DeepH. Search stages are
test-blind; `select-top-k` fails closed if search inputs contain test metrics.
If the protocol declares more than one dataset, runner stages require
`--dataset-id`; the workflow writes `selected_dataset_id`,
`executed_dataset_ids`, and all protocol dataset IDs into the stage manifests
so no dataset can be silently reduced to `datasets[0]`.

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

Graph2Mat supports a benchmark-level `readout` search key with two verified
families:

```text
default
edge_node_mix
```

`edge_node_mix` is expanded before execution to the Graph2Mat model keys
`node_block_readout`, `edge_block_readout`, `preprocessing_edges`, and
`preprocessing_edges_reuse_nodes`. The mapped classes are the existing
Graph2Mat e3nn readout/preprocessing classes, so no unknown model argument is
sent downstream. Configs that omit `readout` keep the historical Graph2Mat
defaults.

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
- `final_comparison.json`: exploratory accuracy, compute and practical Pareto
  diagnostics from available run artifacts.
- `final_report.json/md`: fail-closed final claim report. It declares no
  winner unless both `final_statistics` and `gate_status` allow robust claims.

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

For a claim-bearing paper report, run the gate checker and pass both evidence
files to the report generator:

```bash
python3 Comparison/scripts/g2m_deeph_gate_check.py \
  --protocol "$PROTOCOL" \
  --workflow-root "$WORKFLOW_ROOT" \
  --run-root Comparison/results/<benchmark_run_id> \
  --output "$WORKFLOW_ROOT/gate_status.json"

python3 Comparison/scripts/g2m_deeph_report.py \
  Comparison/results/<benchmark_run_id> \
  --metric low_energy_rmse_eV \
  --mode min \
  --final-statistics "$WORKFLOW_ROOT/final_test/final_statistics.json" \
  --gate-status "$WORKFLOW_ROOT/gate_status.json"
```

If `gate_status.json` is missing or blocks the claim, `final_report.json/md`
must report `robust_claim_allowed=false` and leave `accuracy_winner`,
`cost_winner`, and `pareto_winner` empty. H-MAE/common-summary recommendations
remain supporting Hamiltonian diagnostics unless H-MAE is explicitly
preregistered as `final_evaluation.primary_metric`; they cannot override the
gate checker.

## Paper-Ready Reviewer Runbook

This is the strict path for a defensible Graph2Mat-vs-DeepH comparison. It is
separate from exploratory UI runs and smoke tests. Do not use DeepH paper
numbers as a direct baseline for this workflow; they are external context only
unless the same frozen SIESTA dataset, splits, references, metrics and hardware
contract are used.

Set the common paths once:

```bash
PROTOCOL=Comparison/config/g2m_deeph_paper_protocol_v1_example.json
WORKFLOW_ROOT=Comparison/results/g2m_deeph_final_workflow
DATASET_ROOT=Comparison/datasets/<frozen-dataset-id>
FINAL_RUN_ROOT="$WORKFLOW_ROOT/runs/<final-run-id>"
```

1. **Required external artifacts**

   A paper-ready dataset and run need the external files that are intentionally
   not stored in git: SIESTA raw matrices, split manifests, model predictions,
   telemetry, equivalence evidence, final statistics and reports. The required
   per-snapshot SIESTA files are `RUN.fdf`, `RUN.out` or `siesta.out`,
   `SystemLabel.TSHS`, `SystemLabel.TSDE`, `SystemLabel.HSX`,
   `SystemLabel.STRUCT_OUT`, `SystemLabel.XV`, `SystemLabel.ORB_INDX`, and
   `metadata.json`. `ML_prediction.HSX` is never a reference file.
   `ML_prediction.HSX` is never a reference in metric staging, manifests or
   release evidence.

2. **Dataset freeze/verify**

   ```bash
   python3 Comparison/scripts/g2m_deeph_verify_protocol_datasets.py \
     --protocol "$PROTOCOL" \
     --output "$WORKFLOW_ROOT/dataset_verification.json" \
     --strict
   ```

   This must pass before search. It verifies dataset roots, manifests, strict
   SIESTA/environment provenance, non-empty train/validation/test splits,
   split-hash linkage and forbidden-reference checks.

3. **Protocol validation**

   ```bash
   python3 Comparison/scripts/g2m_deeph_final_workflow.py \
     --stage validate-protocol \
     --protocol "$PROTOCOL" \
     --workflow-root "$WORKFLOW_ROOT" \
     --verify-datasets
   ```

   The protocol must include `final_evaluation.primary_metric`; validation
   metrics such as `val_loss` are allowed for selection but not for final
   scientific claims.

4. **Dataset release manifest**

   ```bash
   python3 Comparison/scripts/g2m_deeph_release_manifest.py \
     --dataset-root "$DATASET_ROOT" \
     --output "$WORKFLOW_ROOT/dataset_release_manifest.json" \
     --strict
   ```

   Run this early to confirm that external dataset evidence is hashable. Run it
   again with `--run-root "$FINAL_RUN_ROOT"` and
   `--workflow-root "$WORKFLOW_ROOT"` after final evaluation to capture
   predictions, telemetry, final stats and reports.

5. **DeepH raw/global equivalence preflight**

   ```bash
   python3 Comparison/scripts/deeph_raw_global_equivalence_preflight.py \
     --frozen-split-manifest "$DATASET_ROOT/frozen_split_manifest.json" \
     --graph2mat-result-dir "$FINAL_RUN_ROOT" \
     --deeph-processed-dir "$FINAL_RUN_ROOT/deeph/processed" \
     --deeph-predictions-dir "$FINAL_RUN_ROOT/deeph/inference" \
     --sample-limit 5 \
     --output-dir "$WORKFLOW_ROOT/deeph_raw_global_equivalence" \
     --fail-closed
   ```

   Missing `raw_global_equivalence_evidence.json`, failed shape/unit/orbital
   order/R-vector/spin/support/H(k)/S_ref/eigenvalue checks, or hand-written
   evidence keeps DeepH diagnostic-only.

6. **Search plan generation**

   ```bash
   python3 Comparison/scripts/g2m_deeph_final_workflow.py \
     --stage generate-search-plan \
     --workflow-root "$WORKFLOW_ROOT"
   ```

   This writes the preregistered search plan before jobs run. The search space
   may be grid, deterministic random or Latin hypercube according to the
   validated protocol.

7. **Test-blind search**

   ```bash
   python3 Comparison/scripts/g2m_deeph_final_workflow.py \
     --stage run-search \
     --workflow-root "$WORKFLOW_ROOT" \
     --dataset-id <dataset-id>
   ```

   Use `--dry-run` first to inspect the exact runner payload. Search artifacts
   must not contain test metrics, and search/checkpoint choices must use only
   validation metrics.

8. **Validation-only top-k**

   ```bash
   python3 Comparison/scripts/g2m_deeph_final_workflow.py \
     --stage select-top-k \
     --workflow-root "$WORKFLOW_ROOT"
   ```

   Top-k uses the protocol validation metric and fails closed if test metrics
   appear in search inputs.

9. **Final multi-seed plan**

   ```bash
   python3 Comparison/scripts/g2m_deeph_final_workflow.py \
     --stage generate-final-seeds \
     --workflow-root "$WORKFLOW_ROOT"
   ```

   The output must preserve selected hyperparameters exactly and expand only
   seed/run metadata. Final robust claims require the configured seed count.

10. **Final training and locked test evaluation**

    ```bash
    python3 Comparison/scripts/g2m_deeph_final_workflow.py \
      --stage run-final \
      --workflow-root "$WORKFLOW_ROOT" \
      --dataset-id <dataset-id>

    python3 Comparison/scripts/g2m_deeph_final_workflow.py \
      --stage evaluate-final-test \
      --workflow-root "$WORKFLOW_ROOT" \
      --final-run-root "$FINAL_RUN_ROOT"
    ```

    Test metrics are produced only in the final stage and are interpreted using
    `final_evaluation.primary_metric`, not `selection.metric`.

11. **Final statistics and Pareto**

    The workflow stage above writes final statistics when run through
    `evaluate-final-test`. For standalone aggregation:

    ```bash
    python3 Comparison/scripts/g2m_deeph_final_stats.py \
      "$FINAL_RUN_ROOT" \
      --metric low_energy_rmse_eV \
      --mode min \
      --expected-seeds 0,1,2,3,4 \
      --min-final-seeds 3 \
      --output-dir "$WORKFLOW_ROOT/final_test"
    ```

    The final statistics report must include seed mean/std, uncertainty where
    computable, GPU-hours, peak memory, throughput and Pareto summaries.

12. **Gate checker**

    ```bash
    python3 Comparison/scripts/g2m_deeph_gate_check.py \
      --protocol "$PROTOCOL" \
      --workflow-root "$WORKFLOW_ROOT" \
      --run-root "$FINAL_RUN_ROOT" \
      --output "$WORKFLOW_ROOT/gate_status.json"
    ```

    This is the single reviewer-facing answer for robust claims. Anything other
    than `claim_status=robust_allowed` with `robust_claim_allowed=true` blocks a
    winner claim.

13. **Final report and evidence bundle**

    ```bash
    python3 Comparison/scripts/g2m_deeph_final_workflow.py \
      --stage generate-report \
      --workflow-root "$WORKFLOW_ROOT" \
      --report-run-root "$FINAL_RUN_ROOT"

    python3 Comparison/scripts/g2m_deeph_report.py \
      "$FINAL_RUN_ROOT" \
      --metric low_energy_rmse_eV \
      --mode min \
      --final-statistics "$WORKFLOW_ROOT/final_test/final_statistics.json" \
      --gate-status "$WORKFLOW_ROOT/gate_status.json"
    ```

    The report may include diagnostic tables, but final winners come only from
    final statistics plus gate status. The evidence bundle and release manifest
    should be archived together.

## Optional Hamiltonian Derivative Diagnostics

Derivative metrics are an optional postprocess and are disabled by default.
They compare finite differences of Hamiltonian matrices, not force constants,
phonon dynamical matrices, or finite differences of forces.

For this repository, `dH/dR` means the derivative of Hamiltonian matrix
elements with respect to Cartesian atomic displacement. The valid reference is
the finite-difference SIESTA Hamiltonian derivative:

```text
(H_SIESTA(R + delta) - H_SIESTA(R - delta)) / (2 * delta)
```

SIESTA force constants, `.FC` files, dynamical matrices, and phonons are not
valid `dH/dR` references here. They must not be substituted for finite
differences of SIESTA Hamiltonians.

```yaml
derivative_metrics:
  enabled: false
  finite_difference_method: central
  split: test
  require_central: true
  diagnostic_only: true
  support_threshold: 1e-12
```

When enabled, the runner calls
`Comparison/scripts/evaluate_hamiltonian_derivative_metrics.py` after common H
metrics. Derivative failures are recorded as diagnostic derivative manifests and
must not hide or rewrite existing H metric outputs. Aggregation preserves the
`graph2mat` and `deeph` method labels, adds derivative rows only when
`derivative_metrics/*.csv` outputs exist, and adds recommendation notes only as
diagnostics. Derivative metrics are not winner metrics and do not enable
paper-level claims.

### Required derivative artifacts

Derivative comparisons assume archived Hamiltonian artifacts already exist. The
minimum expected evidence is:

- `RUN.fdf`
- `metadata.json`
- SIESTA Hamiltonian reference such as `*.HSX` or `*.TSHS`
- predicted Hamiltonian `ML_prediction.HSX`
- explicit plus/minus displacement metadata
- `ORB_INDX` and basis/gauge evidence where available

For central finite differences, plus/minus pairing must be unambiguous. Missing
pairing, mismatched delta, mismatched units, inconsistent atom indexing, or
missing orbital-ordering evidence must keep the derivative result in a blocked
or diagnostic-only state.

### Derivative CLI usage

The derivative evaluator consumes archived references and predictions; it does
not rerun SIESTA, Graph2Mat, or DeepH.

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

Implemented options include:

- `--method {central,forward,backward}`
- `--split {test,validation,train,all}`
- `--require-central`
- `--diagnostic-only`
- `--support-threshold <float>`
- `--max-stencils <int>`
- `--output-dir <path>`
- `--source-model {graph2mat,deeph}`
- `--overwrite`

Outputs are written under `derivative_metrics/`:

- `manifest.json`
- `stencil_status.csv`
- `derivative_matrix_metrics.csv`
- `derivative_support_sweep.csv`
- `derivative_hermiticity.csv`
- `derivative_summary.json`

The fail-closed derivative gate checker reads those outputs and classifies what
can honestly be claimed:

```bash
python3 Comparison/scripts/g2m_deeph_derivative_gate_check.py \
  --derivative-root <result_dir>/derivative_metrics \
  --output <result_dir>/derivative_metrics/derivative_gate_report.json
```

If a benchmark run already staged both methods, the same checker can discover
the two derivative roots from the run directory:

```bash
python3 Comparison/scripts/g2m_deeph_derivative_gate_check.py \
  --run-root <benchmark_run_root> \
  --output <benchmark_run_root>/common_metrics/summary/derivative_gate_report.json
```

### Derivative metrics and units

Derivative metrics are reported in `eV/Ang` and remain diagnostic-only by
default. The current derivative outputs include:

- derivative MAE and RMSE on reference, predicted, and union supports
- relative Frobenius, relative L1, and cosine diagnostics
- support precision, recall, F1, false-zero, and false-nonzero diagnostics
- Hermiticity diagnostics for reference and predicted derivative matrices

Rows also carry derivative metadata such as `atom_index_zero_based`, `axis`,
`delta_ang`, `finite_difference_method`, `derivative_units`, and
`comparison_status`.

### Derivative scientific gates

The derivative gate checker emits one of four statuses:

- `internal_diagnostic`
- `technical_presentation`
- `paper_level_candidate`
- `blocked`

Fail-closed blockers include:

- `force_constants_used=true`
- `reference_definition != siesta_hamiltonian_finite_difference`
- no central stencils
- missing plus/minus pairing
- mismatched shapes
- mismatched `delta_ang`
- mismatched units
- missing or inconsistent atom indexing
- missing or inconsistent orbital ordering metadata
- high Hermiticity defect
- support pattern discontinuity above threshold

Paper-level candidate status is additionally blocked without:

- basis/gauge evidence
- orbital-ordering evidence
- delta sensitivity study
- independent dataset/split metadata
- proven Graph2Mat/DeepH equivalence when both methods are compared

### Derivative limitations

Current derivative comparisons must be interpreted conservatively:

- gauge, basis, and orbital ordering may still be ambiguous across tools
- neighbor-list or sparsity discontinuities can dominate derivative errors
- delta sensitivity can change rankings or invalidate naive comparisons
- ML prediction noise may be amplified by finite differences
- no force-constants comparison is implemented or scientifically accepted here

## Claim Checklist

| Gate | Evidence file or field | Robust claim requirement |
| --- | --- | --- |
| Dataset contract | `artifact_validation.json`, `benchmark_dataset_manifest.json` | dataset is `benchmark_ready` and all required SIESTA artifacts exist |
| Frozen splits | `frozen_split_manifest.json` | train/validation/test are non-empty and split hashes match |
| References | release manifest, metric manifests | no `ML_prediction.HSX` or model prediction is used as reference |
| Selection metric | protocol `selection` and top-k artifacts | validation-only selection; no test metrics in search/top-k |
| Final metric | protocol `final_evaluation.primary_metric` | final claims use preregistered spectral/DOS/Hamiltonian metric, not `val_loss` |
| DeepH equivalence | `raw_global_equivalence_evidence.json`, adapter manifest | raw/global equivalence is proven, otherwise DeepH is diagnostic-only |
| Telemetry | per-run telemetry and final stats | GPU-hours, peak memory and throughput are present for cost claims |
| Seeds/statistics | `final_statistics.json` | configured final seeds complete and uncertainty is reported honestly |
| Gate status | `gate_status.json` | `robust_claim_allowed=true` and `claim_status=robust_allowed` |
| Release bundle | `artifact_release_manifest.json` | all required external evidence is hashable and complete |

## Allowed And Forbidden Claims

Allowed before gates pass:

- The repository can run the staged benchmark control plane.
- A run is exploratory, diagnostic-only or blocked, with listed reasons.
- H-MAE, spectral, DOS, timing and telemetry tables can be inspected as
  diagnostics if their provenance is clear.

Allowed only after all gates pass:

- Graph2Mat/DeepH accuracy winner for the preregistered final metric.
- Compute winner for a configured accuracy threshold.
- Practical/Pareto winner from non-dominated accuracy-cost fronts.

Forbidden:

- Direct winner claims from DeepH paper numbers.
- Spectral superiority from H-MAE alone unless H-MAE was preregistered as the
  final metric.
- Any claim using test metrics for search, top-k, early stopping or checkpoint
  selection.
- Cost-efficiency claims when GPU-hours, memory or throughput telemetry is
  missing.
- Robust DeepH claims when `raw_global_equivalence_evidence.json` is missing,
  unproven or failed.
- Treating smoke output as scientific benchmark evidence.

## Diagnostic-Only Examples

The final report must remain diagnostic-only in these common cases:

- DeepH adapter reports `unproven`, `failed`, `diagnostic_only`, unknown units,
  unknown orbital order, or R-vector mismatch.
- `low_energy_rmse_eV`, Fermi-window or DOS metrics are unavailable for one
  model but are required by `final_evaluation`.
- Final seeds are incomplete or fewer than `--min-final-seeds`.
- Telemetry is `partial` or `unavailable` for a compute claim.
- Dataset manifests are missing, repaired without provenance, or point to
  different split hashes.
- Search artifacts include test metrics.

## Troubleshooting Blocked Gates

| Blocker | What to check |
| --- | --- |
| Missing provenance | Regenerate or attach SIESTA version/build, command line, FDF hash, basis/pseudo hashes and environment manifest. |
| Missing frozen split | Run the dataset manifest builder or verifier with `--write-manifests` only after confirming the intended split root. |
| Forbidden reference | Remove any `ML_prediction.HSX` or prediction path from reference fields and rerun metric staging. |
| DeepH diagnostic-only | Run the raw/global equivalence preflight and include generated evidence plus adapter manifests in the bundle. |
| Missing telemetry | Inspect per-run telemetry JSON; rerun affected jobs if GPU-hours, peak memory or throughput are required for claims. |
| Incomplete seeds | Resume final selected configs until the expected seed list is complete, then rerun final stats. |
| Smoke passed but gates fail | Expected behavior: smoke validates wiring only and writes `not_a_scientific_run`. Supply real evidence before claims. |
