# Cross-Structure Evaluation

Cross-structure evaluation measures structural transfer: train and validate on
one validated MD dataset, then evaluate on the held-out test split of another
validated MD dataset. It is an out-of-distribution test when the source and
target structures differ, for example primitive graphene to a 5x5 graphene
supercell.

## Split Contract

The materialized composite dataset is runner-ready and keeps split membership
frozen:

```text
train      <- source dataset train split only
validation <- source dataset validation split only
test       <- target dataset test split only
```

The source test split is excluded. The target train and validation splits are
excluded. No random re-splitting, ratio split, `_split_pool`, or SIESTA run is
performed.

Materialized sample ids are role-prefixed:

```text
source_train__<original_id>
source_validation__<original_id>
target_test__<original_id>
```

## Compatibility

The implementation reuses the existing ML-vs-SIESTA compatibility checks. It
fails closed for different real species, real-species basis hashes,
pseudopotential hashes, active ghost target spaces, blocking DFT settings, and
the joint Graph2Mat/DeepH artifact contract.

It also fails closed when Hamiltonian target semantics are incomplete: H-only
policy, one matrix component, spin semantics, and real/complex representation
must be explicit enough for a production cross-structure run. Legacy preview or
development payloads may set `confirm_incomplete_hamiltonian_semantics=true`;
that confirmation is written into provenance.

Atom count, cell dimensions, lattice vectors, raw Hamiltonian dimensions, system
label, and raw Monkhorst-Pack integers may differ. K-point sampling is compared
through the existing reciprocal-spacing logic, so primitive-cell and supercell
k-grids can be compatible even when the integer grids differ.

## Test Blindness

The target structure never contributes to training or validation. The source
structure never contributes to the cross-structure test split. Provenance stores
a leakage report checking role/split membership, unique materialized ids, and
unique canonical source artifact identities. Target test membership is inherited
from the target dataset frozen split and is deterministic across repeated
materializations.

`train` rejects `runner_payload.training_sweep` for this workflow; that keeps
target-test metrics out of hyperparameter search/model selection in the
payload-driven entry point.

## Payload

Use:

```bash
.venv/bin/python Comparison/scripts/run_cross_structure_payload.py \
  Comparison/config/graphene_w90_to_5x5_cross_structure_preview_payload.json
```

Supported actions:

```text
preview      validate and report counts/compatibility; write nothing
materialize  build the composite dataset; do not train
train        materialize/reuse the composite dataset and launch the existing runner
predict_metrics  materialize/reuse and evaluate existing checkpoints; never train
```

For `train`, keep the CLI process alive and persist runner status:

```bash
.venv/bin/python Comparison/scripts/run_cross_structure_payload.py payload.json \
  --status-json Comparison/results/cross_structure/status.json \
  --manifest-json Comparison/results/cross_structure/manifest.json \
  --poll-seconds 30
```

Important payload fields:

```json
{
  "schema": "g2m_deeph_cross_structure_run_v1",
  "action": "preview",
  "source_dataset_root": "${REPO_ROOT}/Comparison/datasets/...",
  "target_dataset_root": "${REPO_ROOT}/Comparison/datasets/...",
  "composite_dataset_root": "${REPO_ROOT}/Comparison/results/.../dataset",
  "run_output_root": "${REPO_ROOT}/Comparison/results/.../training",
  "link": true,
  "overwrite": false,
  "runner_payload": {
    "selected_methods": ["graph2mat", "deeph"],
    "allow_diagnostic_metrics": false,
    "metric_fail_policy": "fail_closed",
    "graph2mat_overrides": {},
    "deeph": {},
    "performance": {}
  }
}
```

For `train`, protected runner fields are forced by the wrapper and cannot be
overridden inside `runner_payload`:

```json
{
  "dataset_mode": "reuse_validated",
  "dataset_root": "<composite_dataset_root>",
  "output_root": "<run_output_root>",
  "allow_regenerate_siesta": false
}
```

## Graphene 5x5 monovacancy campaign

The preset `materials/graphene_5x5_vacancy` is the pristine 5x5 cell with the
ideal central carbon at fractional position `(0.5, 0.5, 0.0)` removed. It has 49
carbons, is unrelaxed and non-spin-polarized, and shares the pristine PAO basis,
pseudopotential and electronic settings. The target builder deletes the same
zero-based atom index from pristine test snapshots and records both the ideal
site and the actual MD-displaced position in each `metadata.json`.

First inspect the transformation. This writes no final dataset and never invokes
SIESTA:

```bash
.venv/bin/python Comparison/scripts/build_graphene_5x5_vacancy_target.py \
  --source-dataset Comparison/datasets/graphene_5x5_snapshot_scaling_20_50_80/graphene_5x5_scale_iid20 \
  --output-root /tmp/graphene_5x5_vacancy_dry_run \
  --source-split test \
  --limit 2 \
  --atom-index 24 \
  --dry-run
```

Generate the real static SIESTA references when compute is available:

```bash
.venv/bin/python Comparison/scripts/build_graphene_5x5_vacancy_target.py \
  --source-dataset Comparison/datasets/graphene_5x5_snapshot_scaling_complete_paper_ready_missing/graphene_5x5_scale_iid600 \
  --output-root Comparison/datasets/graphene_5x5_vacancy \
  --source-split test \
  --limit 20 \
  --atom-index 24 \
  --siesta-command siesta
```

The output contains only a frozen `test` split. Every sample is checked against
the existing joint Graph2Mat/DeepH artifact contract before the dataset is
marked ready. An existing output is preserved unless `--overwrite` is passed;
a failed SIESTA run removes the partial replacement and restores any previous
target.

The campaign payload is reproducible rather than hand-maintained:

```bash
.venv/bin/python Comparison/scripts/ops/build_cross_predict_metrics_payload.py
```

It creates
`Comparison/config/graphene_w90_5x5_to_vacancy_predict_metrics_payload.json`,
with one w90→vacancy and one 5x5→vacancy pair per catalogued source size. Preview
it without loading a model:

```bash
.venv/bin/python Comparison/scripts/run_cross_structure_sweep_payload.py \
  Comparison/config/graphene_w90_5x5_to_vacancy_predict_metrics_payload.json \
  --action preview
```

Preview fails closed as incompatible, with the missing-dataset reason, until
`Comparison/datasets/graphene_5x5_vacancy` exists and validates. Before the real
evaluation, stage the archived w90 checkpoints referenced by the same payload:

```bash
.venv/bin/python Comparison/scripts/ops/prepare_cross_predict_metrics_artifacts.py \
  Comparison/config/graphene_w90_5x5_to_vacancy_predict_metrics_payload.json
```

Then evaluate. `predict_metrics` sets `predict_metrics_only=true` and passes the
catalogued Graph2Mat training directories and DeepH model directories to the
runner; no fitting, epoch loop or checkpoint modification occurs:

```bash
.venv/bin/python Comparison/scripts/run_cross_structure_sweep_payload.py \
  Comparison/config/graphene_w90_5x5_to_vacancy_predict_metrics_payload.json \
  --output-root Comparison/results/ml_vs_siesta_cross_structure_sweep/vacancy \
  --result-json Comparison/results/ml_vs_siesta_cross_structure_sweep/vacancy/result.json
```

For the UI, start `python3 Comparison/scripts/pipeline_ui.py`, open **Cross
testing**, and use **Cross testing con vacante — checkpoints existentes**. Its
payload field is restricted to JSON files below `Comparison/config`; the preview
and evaluation buttons use their own log, live status and MAE chart, without
altering the normal cross-testing plot or its payload selection.
The evaluation button always overrides the action to `predict_metrics`.

Results are written below the selected sweep output root, one stable
`<source_id>__to__<target_id>` directory per pair. The summary is
`cross_structure_sweep_summary.json`; model metric directories contain the
existing `h_mae_eV`, `h_rmse_eV`, relative/Frobenius and spectral summaries plus
`block_metrics.csv`, `species_pair_metrics.csv` and `orbital_pair_metrics.csv`
when available. The UI plots `h_mae_eV` in meV versus source training snapshots
as four distinct source/model curves.

To compare the pristine 5x5 baseline with the vacancy target, match the same
source checkpoint/model and compare their final-test metrics. In
`block_metrics.csv`, `row_atom == col_atom` identifies on-site blocks and
`row_atom != col_atom` identifies off-site blocks. Do not interpret
`distance_bin_metrics.csv` as a valid local periodic analysis for graphene: the
current evaluator intentionally disables those bins because it does not apply
the periodic minimum-image convention. DOS, LDOS, IPR and spin analysis are not
part of this initial campaign.

## Outputs

`materialize` writes:

```text
splits/{train,validation,test}/
splits/{train,validation,test}_manifest.csv
artifact_validation.json
benchmark_dataset_manifest.json
frozen_split_manifest.json
material_provenance.json
cross_structure_dataset_provenance.json
dataset_compatibility_report.json
```

The split CSV rows and frozen manifest rows preserve `evaluation_mode`,
`transfer_direction`, role, original sample id, original split, original source
root, system label, atom count when available, and artifact validation status.
`cross_structure_dataset_provenance.json` also stores the source-artifact
identity used for leakage checks and actual link/copy counts. If `train` sees an
existing composite dataset with matching source/target roots and split hashes,
it recalculates leakage from the current frozen rows before reusing it unless
`overwrite=true`.

## Example

The committed example uses existing local datasets:

```text
source: Comparison/datasets/graphene_w90_snapshot_scaling/graphene_w90_scale_iid10
target: Comparison/datasets/graphene_5x5_snapshot_scaling_20_50_80/graphene_5x5_scale_iid20
```

It previews a 2-atom graphene to 50-atom graphene 5x5 transfer case.

## Limitations

The payload wrapper does not support `training_sweep` in cross-structure mode;
use fixed runner settings for training. The vacancy workflow itself performs no
retraining and only evaluates existing checkpoints.
Metric formulas are unchanged, so compare normalized/per-entry metrics across
structures rather than unnormalized total norms. Some older local datasets may
lack explicit Hamiltonian semantics; those require the explicit development
confirmation above or updated provenance before production materialization.
