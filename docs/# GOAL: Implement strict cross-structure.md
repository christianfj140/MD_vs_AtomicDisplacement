# GOAL: Implement strict cross-structure training and evaluation runs

## Role

You are a senior Python and computational-materials engineering agent working directly in:

`/home/christian/repositorios/MD_vs_AtomicDisplacement`

Implement a minimal, scientifically valid, fully tested workflow that allows models to be trained and validated on MD snapshots from one atomic structure and evaluated on MD snapshots from another structure.

Canonical example:

* Train and validate on primitive graphene snapshots with 2 atoms.
* Test on graphene 5×5 supercell snapshots with 50 atoms.

The implementation must support both Graph2Mat and DeepH through the existing joint benchmark runner.

Do not merely write a plan. Inspect the repository, implement the feature, add tests, run validation and report the result.

---

## Primary objective

Introduce a cross-structure evaluation mode with the following strict split contract:

```text
train      <- source/training structure only
validation <- source/training structure only
test       <- target/evaluation structure only
```

For the graphene example:

```text
train      <- 2-atom graphene dataset, source train split
validation <- 2-atom graphene dataset, source validation split
test       <- 50-atom graphene dataset, target test split
```

The target structure must never contribute snapshots to training or validation.

The source structure must never contribute snapshots to the cross-structure test split.

Do not randomly re-split or combine the two datasets.

---

## Repository context already identified

The current repository already contains relevant infrastructure. Inspect it before editing, especially:

* `Comparison/scripts/ml_vs_siesta/mixed_dataset_materialize.py`
* `Comparison/scripts/ml_vs_siesta/dataset_compatibility.py`
* `Comparison/scripts/ml_vs_siesta/mixing_sweep.py`
* `Comparison/scripts/ml_vs_siesta/__init__.py`
* `Comparison/scripts/run_mixing_e2e_payload_once.py`
* `Comparison/scripts/g2m_deeph_runner.py`
* `Comparison/scripts/g2m_deeph_metrics.py`
* `Comparison/scripts/evaluate_hamiltonian_metrics.py`
* `shared/benchmark_manifest.py`
* `shared/joint_artifact_contract.py`
* `tests/test_ml_vs_siesta_mixing.py`
* `tests/test_g2m_deeph_runner.py`
* `docs/graph2mat_deeph_benchmark.md`
* `docs/ml_vs_siesta_benchmark.md`
* representative primitive-cell and graphene 5×5 payloads under `Comparison/config/`

The current mixing implementation already provides:

* reading samples from `frozen_split_manifest.json`;
* snapshot validation;
* safe output-root handling;
* symlink-or-copy materialization;
* deterministic benchmark and frozen-split manifests;
* dataset compatibility checks;
* material provenance;
* a `reuse_validated` runner payload;
* CPU-only fake-dataset tests.

Reuse these capabilities. Do not independently reimplement the same domain rules.

The GitHub repository was observed at commit:

`542a1ff0ecf1e2d51decb987b12a3f21b7c7a216`

Record the actual local `HEAD` before editing. Do not reset the repository or assume the local checkout has not advanced.

---

## Mandatory initial inspection

Before modifying code, inspect the repository and provide a concise internal summary of:

1. Actual current commit and worktree status.
2. Python version and dependency/tooling configuration.
3. How `Graph2MatDeepHBenchmarkRunner` consumes:

   * `dataset_root`;
   * split manifests;
   * `dataset_mode="reuse_validated"`;
   * Graph2Mat training and inference;
   * DeepH preprocessing, training and inference;
   * Hamiltonian metrics.
4. Whether training sweeps subset only the training split or rebuild/re-split the complete dataset.
5. Which metadata from split CSV rows survives into `frozen_split_manifest.json`.
6. Which existing helpers can be reused safely.
7. Whether Graph2Mat or DeepH contains any explicit assumption that train and test samples have the same number of atoms.
8. Relevant existing tests and actual validation commands.

Do not invent files, functions, classes, imports, routes, configuration keys, environment variables, CLI flags, dependencies or APIs. Verify everything in the repository first.

---

## Required design

### 1. Composite cross-structure dataset

Implement a materializer that constructs a runner-ready dataset root from two already validated dataset roots:

* `source_dataset_root`
* `target_dataset_root`

The materialized dataset must contain:

```text
<cross_dataset_root>/
├── splits/
│   ├── train/
│   ├── validation/
│   ├── test/
│   ├── train_manifest.csv
│   ├── validation_manifest.csv
│   └── test_manifest.csv
├── artifact_validation.json
├── benchmark_dataset_manifest.json
├── frozen_split_manifest.json
├── material_provenance.json
├── cross_structure_dataset_provenance.json
├── dataset_compatibility_report.json
└── required shared basis/pseudopotential artifacts
```

Use the repository’s actual expected file names if they differ.

### 2. Exact split membership

Read the frozen manifests of both source datasets.

Populate the new dataset as follows:

```text
composite train:
    rows whose split is "train" in source_dataset_root

composite validation:
    rows whose split is "validation" in source_dataset_root

composite test:
    rows whose split is "test" in target_dataset_root
```

Default behavior must exclude:

* source `test` rows;
* target `train` rows;
* target `validation` rows.

Do not perform `_split_pool`, random splitting, blocked splitting or ratio-based selection.

Preserve the original split membership exactly.

Fail clearly when any required split is absent or empty.

### 3. Neutral terminology

This feature is not limited to “small” and “large” structures.

Use neutral public terminology:

* `source`
* `target`
* `train_structure`
* `test_structure`
* `cross_structure`

Do not expose `small_root` and `large_root` as the new public API.

Existing internal compatibility functions may still be called with their current parameter names. Do not broadly rename the existing mixing API.

### 4. Collision-safe sample IDs

Prefix materialized sample IDs so collisions are impossible:

```text
source_train__<original_id>
source_validation__<original_id>
target_test__<original_id>
```

The exact syntax may follow an existing repository convention, but it must encode both role and original identity.

Each split row must preserve at least:

* materialized `sample_id`;
* `sample_dir`;
* assigned split;
* source/target role;
* original source root;
* original sample ID;
* original split;
* system label;
* atom count, when available;
* artifact validation status.

### 5. Snapshot materialization

Follow the existing mixed-dataset materializer:

* symlink snapshot artifacts by default;
* fall back to copying where symlinks are unavailable;
* never propagate stale `ML_prediction.HSX` files;
* validate every destination snapshot with the joint artifact contract;
* regenerate manifests using the shared manifest builder;
* never launch SIESTA;
* never silently repair missing artifacts;
* never modify either source dataset.

Follow the existing safe-output-root policy. Never recursively delete an arbitrary user path.

Do not leave a partially valid dataset root after a materialization failure. Reuse an existing staging/atomic-output helper if available; otherwise implement the smallest safe approach consistent with repository conventions.

---

## Scientific compatibility requirements

Atom count and lattice size are intentionally allowed to differ.

For example:

```text
source atom count = 2
target atom count = 50
```

This difference must not be treated as an incompatibility.

The following must be checked before materialization.

### Blocking compatibility conditions

Fail closed when any of these differ or cannot be validated adequately for a production run:

1. Active real chemical species.
2. Orbital basis definition and basis hashes for every active real species.
3. Pseudopotential hashes for every active real species.
4. Hamiltonian target semantics:

   * H-only policy;
   * matrix component count;
   * supported spin semantics;
   * real/complex representation constraints.
5. Relevant DFT target settings already treated as blocking by the repository:

   * exchange-correlation settings;
   * mesh cutoff;
   * electronic temperature;
   * SCF tolerance;
   * spin configuration;
   * other existing blocking fingerprint fields.
6. Active ghost-species target space.
7. Required SIESTA artifacts and joint Graph2Mat/DeepH artifact contract.

### Differences that may be valid

Do not require exact equality of:

* atom count;
* lattice vectors;
* cell dimensions;
* raw Monkhorst–Pack mesh integers;
* number of Hamiltonian rows or columns;
* system label.

For primitive-cell versus supercell calculations, compare k-point sampling using the repository’s existing reciprocal-space density or spacing logic rather than requiring the same raw k-grid.

Record all non-blocking sampling differences in the compatibility report.

Do not weaken the current compatibility validation globally. Add stricter cross-structure validation around the existing helpers where necessary.

In particular, do not treat different pseudopotentials as a warning for this workflow: they must block the run.

---

## Provenance contract

Write:

`cross_structure_dataset_provenance.json`

Use a versioned schema such as:

`ml_vs_siesta_cross_structure_dataset_provenance_v1`

It must contain at least:

```json
{
  "schema": "ml_vs_siesta_cross_structure_dataset_provenance_v1",
  "evaluation_mode": "cross_structure",
  "evaluation_scope": "target_structure_only",
  "validation_scope": "source_structure_only",
  "source_dataset_root": "...",
  "target_dataset_root": "...",
  "source_split_hash": "...",
  "target_split_hash": "...",
  "source_system_labels": ["..."],
  "target_system_labels": ["..."],
  "source_atom_counts": [2],
  "target_atom_counts": [50],
  "train_count": 0,
  "validation_count": 0,
  "test_count": 0,
  "source_train_ids": [],
  "source_validation_ids": [],
  "target_test_ids": [],
  "compatibility": {},
  "leakage_check": {},
  "run_inventory": {}
}
```

Adapt field names to repository conventions, but preserve the information.

Also record:

* hashes or stable identities of selected samples;
* whether files were linked or copied;
* exact original split of every materialized sample;
* transfer direction, for example `2_atoms_to_50_atoms`;
* compatibility status;
* warnings;
* repository commit/worktree reproducibility status.

Do not copy the primitive dataset’s provenance and pretend it represents the composite dataset.

The composite `material_provenance.json` must explicitly state that train/validation and test come from different structures.

---

## Leakage and test-blindness requirements

Implement explicit checks proving:

1. No target sample appears in train or validation.
2. No source sample appears in test.
3. No materialized sample ID occurs in more than one split.
4. No canonical source artifact identity occurs in more than one split.
5. The target test split remains unchanged between repeated runs using the same target dataset.
6. Target test metrics are not used for early stopping, model selection or hyperparameter search.
7. Validation remains source-domain validation by default.

The leakage report must be persisted in provenance.

If the repository’s current `training_sweep` implementation rebuilds or re-splits the dataset in a way that could move target samples into training, reject that combination explicitly instead of silently producing an invalid run.

Support a training sweep only when repository inspection proves it merely subsets the existing source training split while leaving validation and target test membership frozen.

---

## Public payload and CLI

Implement a lightweight payload-driven entry point consistent with the existing repository patterns.

A preferred shape is:

```json
{
  "schema": "g2m_deeph_cross_structure_run_v1",
  "action": "preview",
  "case_id": "graphene_2atom_to_graphene_5x5",
  "source_dataset_root": "${REPO_ROOT}/...",
  "target_dataset_root": "${REPO_ROOT}/...",
  "composite_dataset_root": "${REPO_ROOT}/Comparison/results/.../dataset",
  "run_output_root": "${REPO_ROOT}/Comparison/results/.../training",
  "link": true,
  "overwrite": false,
  "runner_payload": {
    "selected_methods": [
      "graph2mat",
      "deeph"
    ],
    "allow_diagnostic_metrics": false,
    "metric_fail_policy": "fail_closed",
    "graph2mat_overrides": {},
    "deeph": {},
    "performance": {}
  }
}
```

Use the repository’s existing environment-variable and repository-relative path expansion logic. Do not create a second incompatible path parser.

Supported actions:

### `preview`

* Validate payload structure.
* Read both frozen manifests.
* Run compatibility checks.
* Calculate expected counts and provenance.
* Perform no writes.
* Launch no training.
* Launch no SIESTA.

### `materialize`

* Perform all preview validation.
* Build and validate the composite dataset.
* Do not train.

### `train`

* Materialize or reuse the validated composite dataset.
* Construct the standard runner payload.
* Launch the existing `Graph2MatDeepHBenchmarkRunner`.
* Stream or persist runner logs consistently with existing scripts.
* Return the actual runner result and failure status.

When constructing the runner payload, force:

```json
{
  "dataset_mode": "reuse_validated",
  "dataset_root": "<composite_dataset_root>",
  "output_root": "<run_output_root>",
  "allow_regenerate_siesta": false
}
```

Reject conflicting attempts to override these protected fields inside `runner_payload`.

Pass through supported existing runner configuration such as:

* selected methods;
* Graph2Mat overrides;
* DeepH options;
* epoch settings;
* performance configuration;
* modular workflow configuration;
* metric configuration.

Do not duplicate the complete runner configuration schema. Prefer a nested pass-through `runner_payload` and override only the protected cross-structure fields.

Use a dedicated CLI name consistent with the repository. A likely file is:

`Comparison/scripts/run_cross_structure_payload.py`

Choose a different name only if repository inspection identifies a stronger existing convention.

Avoid importing the very large UI backend merely to launch a single runner unless the repository architecture makes that necessary.

---

## Recommended module structure

Prefer a small dedicated module, for example:

`Comparison/scripts/ml_vs_siesta/cross_structure_materialize.py`

It should contain focused, testable functions for:

* parsing source and target split rows;
* planning the composite split;
* compatibility validation;
* leakage validation;
* materialization;
* provenance construction;
* preview generation.

Export the stable public functions through:

`Comparison/scripts/ml_vs_siesta/__init__.py`

Do not turn this into a generic dataset framework.

Do not introduce manager, service, registry or factory classes unless a real repository contract requires them.

Reuse or minimally extract shared helpers from `mixed_dataset_materialize.py` only when that removes actual duplicated domain logic.

Do not change existing mixing semantics.

---

## Runner and metric integration

The resulting composite dataset must work with the existing:

* Graph2Mat training path;
* Graph2Mat target-structure prediction path;
* DeepH preprocessing and training path;
* DeepH target-structure prediction path;
* Hamiltonian metric evaluator;
* common metric aggregation.

Do not modify metric formulas merely because matrix dimensions differ.

The existing per-entry MAE and RMSE metrics remain meaningful when computed on each target snapshot.

Preserve and expose target structure metadata in metric/run records:

* target atom count;
* target system label;
* source atom count;
* source system label;
* transfer direction;
* evaluation mode;
* source and target split hashes.

Do not interpret an unnormalised total Frobenius norm as directly comparable between structures of different sizes.

Relative Frobenius, per-entry MAE/RMSE and other already normalised metrics may continue to be reported according to their existing definitions.

Do not add new scientific metrics unless required to prevent an incorrect aggregation.

Ensure cross-structure records cannot be silently grouped with ordinary in-domain runs solely because they share the same training size. At minimum, persist a grouping key such as:

```text
evaluation_mode=cross_structure
transfer_direction=<source>_to_<target>
```

No new UI plot is required in this task.

---

## Example payload

After implementing the generic feature, create one documented example payload for:

```text
2-atom graphene -> 50-atom graphene 5×5
```

Inspect the repository to determine the actual current primitive-graphene and graphene-5×5 dataset conventions.

Do not invent a dataset directory and claim it exists.

If concrete generated datasets are not committed to the repository, use clearly documented `${REPO_ROOT}`-relative placeholders and explain how to replace them.

The example should default to:

```json
{
  "action": "preview",
  "selected_methods": ["graph2mat", "deeph"],
  "allow_regenerate_siesta": false,
  "metric_fail_policy": "fail_closed"
}
```

Do not launch a real training campaign during implementation.

---

## Documentation

Add a concise document describing:

1. What cross-structure evaluation measures.
2. Why it is an out-of-distribution structural-transfer test.
3. Exact source/target split semantics.
4. Compatibility requirements.
5. Why atom count may differ.
6. Why raw k-grid integers may differ while k-point density remains compatible.
7. How test blindness is preserved.
8. Payload fields.
9. Preview, materialize and train commands.
10. Output files and provenance.
11. Limitations.
12. How to run the 2-atom-to-50-atom example.

Do not describe unimplemented functionality.

---

## Tests

Use the existing pytest style and fake-artifact helpers.

Tests must be CPU-only and must not require:

* SIESTA;
* CUDA;
* Graph2Mat training;
* DeepH training;
* network access;
* real production datasets.

Create focused tests, preferably in:

`tests/test_cross_structure_evaluation.py`

### Required materialization tests

Create a fake source dataset with:

```text
2 atoms
non-empty train split
non-empty validation split
non-empty test split
```

Create a fake target dataset with:

```text
50 atoms
non-empty train split
non-empty validation split
non-empty test split
```

Verify that the composite dataset contains:

```text
train      -> only source 2-atom snapshots
validation -> only source 2-atom snapshots
test       -> only target 50-atom snapshots
```

Assert that:

* source test snapshots are absent;
* target train snapshots are absent;
* target validation snapshots are absent;
* materialized IDs are unique;
* split counts are exact;
* split assignment is deterministic;
* every snapshot passes the joint artifact validation;
* benchmark and frozen manifests are regenerated;
* provenance records both structures;
* leakage report passes;
* target test membership remains stable across repeated materializations.

### Required compatibility tests

Verify failure for:

* different real species;
* different real-species orbital basis;
* different pseudopotential hashes;
* incompatible active ghost species;
* incompatible blocking DFT settings;
* missing target test split;
* missing source train split;
* missing source validation split;
* invalid source or target frozen manifest;
* incompatible Hamiltonian target semantics, when represented in available artifacts/provenance.

Verify that different atom counts do not fail.

Verify that different but density-consistent k-point meshes do not fail merely because their integer grids differ.

### Required payload tests

Verify:

* `preview` writes nothing;
* `materialize` writes the dataset but launches no runner;
* `train` constructs a standard `reuse_validated` runner payload;
* protected runner fields cannot be overridden;
* `allow_regenerate_siesta` remains false;
* Graph2Mat and DeepH settings pass through unchanged;
* a fake launch function receives the correct composite dataset root;
* launch failure is propagated rather than converted into success;
* unsupported or unsafe training-sweep behavior fails clearly.

### Regression tests

Run and preserve the existing behavior covered by:

* `tests/test_ml_vs_siesta_mixing.py`
* relevant dataset and `reuse_validated` tests in `tests/test_g2m_deeph_runner.py`

Do not modify existing tests merely to weaken their assertions.

---

## Validation

Discover and use the repository’s actual environment and commands.

Run targeted tests first. At minimum, run the equivalent of:

```bash
python -m pytest -q tests/test_cross_structure_evaluation.py
python -m pytest -q tests/test_ml_vs_siesta_mixing.py
python -m pytest -q tests/test_g2m_deeph_runner.py -k "dataset or reuse or split"
```

Adjust only when the repository’s actual test layout requires it.

Then run relevant broader validation that is reasonably affordable:

```bash
python -m pytest -q <other directly affected tests>
```

Run the repository’s configured linter, formatter check and type checker only if they already exist.

Also execute the new example payload in `preview` mode and verify:

* no output dataset was created;
* source and target structures were detected;
* source and target atom counts were reported;
* expected split counts were reported;
* compatibility status was explicit;
* no SIESTA, training or prediction process was launched.

If a command cannot be run, state exactly which command, why it could not run and what remains unverified.

Do not claim that a check passed unless it was actually executed.

---

## Scope restrictions

Do not:

* launch a real SIESTA campaign;
* launch real Graph2Mat or DeepH training;
* modify generated production datasets;
* change existing mixing ratios or split-policy semantics;
* broadly refactor `g2m_deeph_runner.py`;
* redesign the UI;
* modify `Comparison/ui/index.html`;
* alter metric formulas without demonstrated necessity;
* add a dependency;
* add silent compatibility overrides;
* allow test data into training or validation;
* use target-test metrics for early stopping;
* hardcode machine-specific paths;
* use `shell=True` for payload-controlled commands;
* swallow validation errors;
* introduce broad `except Exception` handling as a shortcut;
* perform formatting-only rewrites of unrelated files.

Touch `pipeline_ui.py` only if repository inspection proves that a minimal backend hook is required to make the new payload executable. If touched, keep the edit local and add focused tests.

---

## Acceptance criteria

The task is complete only when all of the following are true:

1. A payload can designate independent source and target dataset roots.
2. Composite train and validation contain only source-structure snapshots.
3. Composite test contains only target-structure snapshots.
4. A 2-atom source and 50-atom target pass when their physical target definitions are compatible.
5. Different basis, pseudopotential, species or target semantics fail closed.
6. No random re-splitting occurs.
7. Target-test blindness is preserved.
8. The composite dataset passes the existing joint dataset validator.
9. The existing runner accepts it through `reuse_validated`.
10. Both Graph2Mat and DeepH can be selected in the generated runner payload.
11. Cross-structure provenance and transfer direction are persisted.
12. Existing mixing behavior remains unchanged.
13. Tests cover success, incompatibility, leakage, payload actions and backward compatibility.
14. Targeted validation passes or failures are reported honestly.
15. No unnecessary dependency or broad refactor is introduced.

---

## Final response format

Return:

### 1. Repository inspection

* actual commit;
* worktree state;
* relevant files inspected;
* existing behavior found;
* invariants preserved.

### 2. Implemented design

* materialization flow;
* split contract;
* compatibility policy;
* runner integration;
* provenance and leakage protection.

### 3. Files changed

For every file:

* path;
* purpose;
* concise explanation of the change.

### 4. Tests

* tests added or modified;
* behavior covered;
* relevant edge and failure cases.

### 5. Validation evidence

For every command:

* exact command;
* exit status;
* result;
* relevant failure output when applicable.

### 6. Example usage

Show the exact preview command using the example payload.

### 7. Remaining limitations

State only genuine limitations or unverified behavior.

### 8. Explicit non-changes

Confirm that:

* no real training was launched;
* no SIESTA calculation was launched;
* existing mixing semantics were not changed;
* the UI was not changed unless strictly required.
