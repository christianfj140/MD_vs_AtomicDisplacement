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

No new UI plot is implemented. The payload wrapper does not support
`training_sweep` in cross-structure mode; use fixed runner settings for now.
Metric formulas are unchanged, so compare normalized/per-entry metrics across
structures rather than unnormalized total norms. Some older local datasets may
lack explicit Hamiltonian semantics; those require the explicit development
confirmation above or updated provenance before production materialization.
