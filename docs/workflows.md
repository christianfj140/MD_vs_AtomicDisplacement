# Workflows

## Main Ways To Use The Repository

The repository has three main usage patterns:

1. Run the comparison UI and execute MD, SIESTA FC Cartesian, or Random
   Cartesian experiments.
2. Run the standalone MD or AtomDisplacement pipelines for debugging or local
   development.
3. Run the Graph2Mat-vs-DeepH benchmark tooling for paper-ready comparison
   workflows and diagnostics.

## Comparison UI Workflow

Start the main UI with:

```bash
python3 Comparison/scripts/pipeline_ui.py
```

The server listens on `127.0.0.1:8770` by default.

### Inputs

- `Comparison/config/*.json` payloads and dataset recipes.
- A material bundle selected in the `Experiment` tab.
- Optional archived datasets under `Comparison/datasets/`.
- External executables such as `siesta`, `graph2mat`, and any DeepH commands
  required by the selected run mode.

### Typical Order

1. Choose one or more canonical methods: `md`, `siesta_fc_cartesian`, or
   `random_cartesian`.
2. Choose the run mode:
   - `dataset_only`
   - `full_strict_pipeline`
   - `train_test_metrics_plots_only`
3. Adjust dataset recipes, splits, and training settings.
4. Validate the material bundle.
5. Launch the experiment.

### Outputs

- `Comparison/results/<run_id>/experiment_manifest.yaml`
- `Comparison/results/<run_id>/performance_report.json`
- `Comparison/results/<run_id>/summary/recommendation.json`
- `Comparison/results/<run_id>/metrics/*.csv`
- `Comparison/results/results_<method>/<dataset_label>/run_<run_id>/manifest.json`

### Common Failure Points

- No method selected.
- Invalid or incomplete material bundle.
- Missing SIESTA, Graph2Mat, or DeepH executables.
- Reused archived datasets missing the required manifests or split files.
- Old datasets with incompatible provenance hashes or missing joint artifacts.

## MD Standalone Workflow

Start the UI with:

```bash
python3 MD/scripts/pipeline_ui.py
```

or run the pipeline directly with:

```bash
python3 MD/scripts/main_md.py
```

### Inputs

- `MD/pipeline_config.yaml`
- `materials/graphene/RUN.fdf`
- `MD/dataset/`
- A local `.venv` and the external tools named in the config

### Order Of Execution

`MD/scripts/main_md.py` follows the step list in `MD/pipeline_config.yaml`:

1. `generate_md_dataset`
2. `run_md_training`
3. `run_md_testing`
4. `run_md_prediction`

If `pipeline.skip_model_test` is true, the testing step is skipped.

### Outputs

- `MD/dataset/`
- `MD/training/`
- pipeline logs and generated matrices under the dataset and training roots

### Common Failure Points

- Missing local `.venv`.
- `siesta` or `graph2mat` not available in `PATH`.
- Invalid `pipeline_config.yaml`.
- Missing dataset directories or stale outputs from earlier runs.

## AtomDisplacement Standalone Workflow

Start the UI with:

```bash
python3 AtomDisplacement/scripts/pipeline_ui.py
```

or run the pipeline directly with:

```bash
python3 AtomDisplacement/scripts/main_atom_displacement.py
```

### Inputs

- `AtomDisplacement/pipeline_config.yaml`
- `AtomDisplacement/base/`
- `AtomDisplacement/relaxed/`
- `AtomDisplacement/dataset/`
- A local `.venv` and the external tools named in the config

### Order Of Execution

`AtomDisplacement/scripts/main_atom_displacement.py` follows the configured
pipeline step list, which can include:

1. `run_relaxation`
2. `generate_atom_displacement_dataset`
3. `generate_random_cartesian_dataset`
4. `run_single_points`
5. `normalize_fc_steps`
6. `collect_atom_displacement_dataset`
7. `run_atdisp_training`
8. `run_atdisp_testing`
9. `run_atdisp_prediction`

The config can also request `render_inputs`, which synchronizes derived files
from `pipeline_config.yaml`.

### Outputs

- `AtomDisplacement/dataset/FC_steps/`
- `AtomDisplacement/dataset/RandomCartesian_steps/`
- `AtomDisplacement/dataset/collected/`
- `AtomDisplacement/dataset/samples_manifest.json`
- `AtomDisplacement/dataset/run_summary.json`
- `AtomDisplacement/training/`

### Common Failure Points

- Missing relaxed or base inputs.
- Missing pseudopotentials or basis files for the chosen material.
- Unconverged or incomplete SIESTA single points.
- Invalid sample counts or split settings in the pipeline config.

## Graph2Mat vs DeepH Workflow

Start the combined comparison UI with:

```bash
python3 Comparison/scripts/pipeline_ui.py
```

The benchmark runbook and claim checklist are documented in
`docs/graph2mat_deeph_benchmark.md`.

### Inputs

- A validated joint dataset under `Comparison/datasets/graphene_w90_joint/`
  or another dataset that satisfies the same contract.
- `artifact_validation.json`
- `benchmark_dataset_manifest.json`
- `frozen_split_manifest.json`
- External SIESTA, Graph2Mat, and DeepH executables

### Final Workflow Stages

`Comparison/scripts/g2m_deeph_final_workflow.py` exposes the following stages:

1. `validate-protocol`
2. `generate-search-plan`
3. `run-search`
4. `select-top-k`
5. `generate-final-seeds`
6. `run-final`
7. `run-final-test`
8. `evaluate-final-test`
9. `generate-report`

Each stage writes a JSON manifest under the workflow root so the run can be
audited later.

### Outputs

- staged workflow manifests under `<workflow-root>/stages/`
- final reports and evidence bundles under `<workflow-root>/report/`
- aggregated metrics and final-statistics JSON files
- gate status and selection artifacts for the final/publicable run

### Modular Hamiltonian And Derivative Modes

The Graph2Mat-vs-DeepH runner accepts modular workflow payloads through the
existing UI/API route:

```bash
python3 Comparison/scripts/pipeline_ui.py
curl -X POST http://127.0.0.1:8770/api/g2m-deeph/run \
  -H 'Content-Type: application/json' \
  --data @payload.json
```

The examples below are minimal payload fragments. Add the same dataset,
training, DeepH, Python, and output settings used by your normal benchmark
payload when a mode includes Hamiltonian training or prediction.

H-only benchmark, with derivative stages disabled:

```json
{
  "workflow_mode": "hamiltonian_only",
  "dataset_root": "Comparison/datasets/graphene_w90_joint",
  "output_root": "Comparison/results/graphene_w90_g2m_deeph_benchmark"
}
```

Derivative-stencils-only, starting from MD/base snapshots. MD snapshots are base geometries `R0`; they are not derivative stencils until this stage writes explicit displaced geometries:

```json
{
  "workflow_mode": "derivative_stencils_only",
  "derivative": {
    "enabled": true,
    "source_dataset_root": "Comparison/datasets/graphene_w90_joint",
    "output_root": "Comparison/results/derivative_stencils_demo",
    "method": "central",
    "delta_ang": 0.01,
    "base_split": "test",
    "max_base_snapshots": 2,
    "atoms": ["0"],
    "axes": ["x", "y", "z"],
    "overwrite": false,
    "skip_if_exists": true
  }
}
```

Adaptive derivative base-selection smoke, without training, predictions, or
SIESTA. This creates synthetic split manifests for `n_test=10`, `80`, and
`110`, then runs stencil build and, when local Python dependencies are
available, geometry validation:

```bash
python3 Comparison/scripts/smoke_adaptive_derivative_selection.py \
  --payload Comparison/config/adaptive_derivative_selection_smoke.json
```

Expected compact output:

```json
{"status": "ok", "cases": [{"label": "n_test_10", "selected_base_snapshot_count": 10, "geometry_validation_status": "ok"}, {"label": "n_test_80", "selected_base_snapshot_count": 20, "geometry_validation_status": "ok"}, {"label": "n_test_110", "selected_base_snapshot_count": 22, "geometry_validation_status": "ok"}]}
```

Interpretation: each case checks `K = min(n_test, max(20, ceil(0.20*n_test)))`;
when `n_test < 20`, all test snapshots are selected. Full manifests are written
under `Comparison/results/adaptive_derivative_selection_smoke/stencils/*/`.
If `numpy` is missing, `geometry_validation_status` is
`skipped_missing_dependency`; the manifest K checks still run.

Derivative-metrics-only, using existing finite-displacement stencil artifacts
and Hamiltonians:

```json
{
  "workflow_mode": "derivative_metrics_only",
  "derivative": {
    "enabled": true,
    "result_dir": "Comparison/results/derivative_stencils_demo",
    "method": "central",
    "base_split": "test",
    "skip_if_exists": true
  }
}
```

Hamiltonian benchmark followed by derivative postprocess:

```json
{
  "workflow_mode": "h_then_derivative_postprocess",
  "dataset_root": "Comparison/datasets/graphene_w90_joint",
  "output_root": "Comparison/results/graphene_w90_g2m_deeph_benchmark",
  "derivative": {
    "enabled": true,
    "method": "central",
    "base_split": "test",
    "skip_if_exists": true
  }
}
```

Hamiltonian benchmark followed by the full derivative workflow:

```json
{
  "workflow_mode": "h_then_derivative_full",
  "dataset_root": "Comparison/datasets/graphene_w90_joint",
  "output_root": "Comparison/results/graphene_w90_g2m_deeph_benchmark",
  "derivative": {
    "enabled": true,
    "source_dataset_root": "Comparison/datasets/graphene_w90_joint",
    "output_root": "Comparison/results/graphene_w90_g2m_deeph_benchmark/derivative_workflow",
    "method": "central",
    "delta_ang": [0.005, 0.01, 0.02],
    "base_split": "test",
    "max_base_snapshots": 2,
    "atoms": ["0"],
    "axes": ["x", "y", "z"],
    "skip_if_exists": true
  }
}
```

`full_end_to_end` is also accepted for the full Hamiltonian plus derivative
workflow. `derivative` is the canonical payload key; `derivatives` is accepted
as a backward-compatible alias when it contains the same object.

In training sweeps and `full_strict_pipeline` runs, derivative stencil
generation uses each completed child run's dataset root. It does not fall back
to one shared global `derivative.source_dataset_root` across all children.

For `h_then_derivative_full` and `full_end_to_end`, derivative prediction stages
can use explicit model artifacts such as `derivative.graph2mat_checkpoint` and
`derivative.deeph_model_dir`. When the normal H workflow has just completed,
the runner can infer those model artifacts from the Graph2Mat checkpoint manifest
and the DeepH save directory. To reuse already-produced derivative
predictions instead of running prediction commands, provide
`derivative.graph2mat_existing_prediction_root` and/or
`derivative.deeph_existing_prediction_root`.

`derivative.siesta_command` defaults to `"siesta"` for derivative SIESTA
reference Hamiltonian staging. Override it only when the reference command
itself needs to change.

When both Graph2Mat and DeepH derivative prediction stages are enabled, the
runner may materialize separate model-specific derivative result roots, such as
`graph2mat_derivative_result/` and `deeph_derivative_result/`, so that each
model's predicted Hamiltonians and metrics remain unambiguous.

A stencil is the set of displaced geometries around a base geometry. Finite differences are the formula applied to Hamiltonians evaluated on that stencil.
The recommended benchmark method is a central stencil with `R0`, `R+`, and
`R-`, using the central finite difference:

```text
dH/dR ~= (H(R+) - H(R-)) / (2 * delta)
```

For compatibility with older examples: a central stencil with `R+` and `R-`
also keeps the matching base geometry `R0` when geometry validation is part of
the workflow. `delta_ang` may be one numeric value, a numeric string, a list
such as `[0.005, 0.01, 0.02]`, or a comma-separated string such as
`"0.005,0.01,0.02"`. A delta sweep is the preferred way to check whether the
finite difference is stable.

Forward or backward finite differences are supported only as fallback or
diagnostic modes. SIESTA force constants, `.FC` files, phonons, dynamical
matrices, and finite differences of forces are not used as `dH/dR` references.
The derivative reference is finite differences of SIESTA Hamiltonians.

Keep every `R0`/`R+`/`R-` family in the same split. MD snapshots are valid base
geometries, but arbitrary MD snapshots from different splits or unrelated
families are not valid derivative stencils for one another.

Paper-level derivative claims require more evidence than these modular stages
produce by default: delta stability, basis/gauge/orbital-order compatibility,
and reference-noise checks should be documented before treating results as
paper-level conclusions.

Expected derivative artifact layout:

```text
<derivative-result-root>/
  derivative_stencil_manifest.json
  derivative_geometry_validation.csv
  derivative_geometry_validation.json
  structures/<sample>/RUN.fdf
  structures/<sample>/metadata.json
  siesta_hamiltonians/<sample>/*.HSX|*.TSHS
  siesta_hamiltonians/derivative_siesta_reference_manifest.json
  predicted_hamiltonians/<sample>/ML_prediction.HSX
  predicted_hamiltonians/derivative_graph2mat_prediction_manifest.json
  predicted_hamiltonians/derivative_deeph_prediction_manifest.json
  derivative_metrics/<model>/manifest.json
  derivative_metrics/<model>/derivative_matrix_metrics.csv
  derivative_metrics/<model>/derivative_support_sweep.csv
  derivative_metrics/<model>/derivative_hermiticity.csv
  derivative_metrics/<model>/derivative_summary.json
  derivative_metrics/summary/derivative_gate_report.json
  derivative_metrics/summary/derivative_model_comparison/
    derivative_model_comparison_summary.json
    derivative_model_paired_comparison.csv
```

If model-specific derivative roots are used, treat the generic layout above as
the layout inside each model-specific derivative result root. For example:

```text
<derivative-result-root>/
  graph2mat_derivative_result/
    derivative_stencil_manifest.json
    structures/<sample>/metadata.json
    structures/<sample>/RUN.fdf
    siesta_hamiltonians/<sample>/*.HSX|*.TSHS
    predicted_hamiltonians/<sample>/ML_prediction.HSX
    derivative_metrics/graph2mat/manifest.json
    derivative_metrics/graph2mat/derivative_delta_stability.json
    derivative_metrics/summary/derivative_gate_report.json
  deeph_derivative_result/
    derivative_stencil_manifest.json
    structures/<sample>/metadata.json
    structures/<sample>/RUN.fdf
    siesta_hamiltonians/<sample>/*.HSX|*.TSHS
    predicted_hamiltonians/<sample>/ML_prediction.HSX
    derivative_metrics/deeph/manifest.json
    derivative_metrics/deeph/derivative_delta_stability.json
    derivative_metrics/summary/derivative_gate_report.json
```

### Common Failure Points

- The protocol does not validate.
- A dataset is missing the joint artifact contract files.
- The selected dataset and the frozen split are not compatible.
- DeepH preprocessing cannot find `HSX`, `STRUCT_OUT`, `XV`, or `ORB_INDX`.
- The final workflow is run before validation-based top-k selection has been
  completed.

## Validation And Smoke Checks

The repository also includes dedicated validation commands:

```bash
python3 Comparison/scripts/g2m_deeph_smoke.py --dry-run
python3 Comparison/scripts/verify_dataset_integrity.py --dry-run
python3 -m unittest tests/test_comparison_workflow.py
```

The smoke scripts are useful for plumbing validation, but they do not prove
scientific accuracy by themselves.

### Minimal Derivative Smoke Check

Use these checks for a lightweight derivative workflow sanity pass. This smoke
validates plumbing and artifact layout, not paper-level science.

```bash
python3 -m unittest tests/test_validate_derivative_workflow_artifacts.py
python3 -m unittest tests/test_g2m_deeph_documentation.py
python3 Comparison/scripts/validate_derivative_workflow_artifacts.py \
  <derivative_result_root> \
  --output-json <summary.json>
```

Golden payload examples live in `Comparison/config/`:

- `derivative_stencils_only_minimal.json`
- `derivative_metrics_only_existing_artifacts.json`
- `h_then_derivative_full_smoke.json`

For a minimal staged smoke path, use one base snapshot, one atom, one axis,
and two deltas. In this layout, `graph2mat_derivative_result/` is the active
smoke root and `deeph_derivative_result/` is the staged DeepH prediction root:

```bash
SMOKE_ROOT=Comparison/results/derivative_smoke
STENCIL_ROOT="$SMOKE_ROOT/graph2mat_derivative_result"
DEEPH_ROOT="$SMOKE_ROOT/deeph_derivative_result"

python3 Comparison/scripts/build_hamiltonian_derivative_stencils.py \
  --source-dataset-root Comparison/datasets/PLACEHOLDER_joint_dataset \
  --output-stencil-root "$STENCIL_ROOT" \
  --method central \
  --delta-ang 0.005 0.01 \
  --split test \
  --max-base-snapshots 1 \
  --atoms 0 \
  --axes x \
  --overwrite

python3 Comparison/scripts/validate_hamiltonian_derivative_geometry.py \
  "$STENCIL_ROOT" \
  --output-dir "$STENCIL_ROOT" \
  --method central \
  --source-model graph2mat \
  --split test \
  --require-central

python3 Comparison/scripts/run_hamiltonian_derivative_siesta_references.py \
  --stencil-root "$STENCIL_ROOT" \
  --output-reference-root "$STENCIL_ROOT/siesta_hamiltonians" \
  --existing-reference-root <existing_reference_root>

python3 Comparison/scripts/run_hamiltonian_derivative_predictions.py \
  --stencil-root "$STENCIL_ROOT" \
  --model graph2mat \
  --output-root "$STENCIL_ROOT/predicted_hamiltonians" \
  --existing-prediction-root <existing_graph2mat_prediction_root>

python3 Comparison/scripts/run_hamiltonian_derivative_predictions.py \
  --stencil-root "$STENCIL_ROOT" \
  --model deeph \
  --output-root "$DEEPH_ROOT/predicted_hamiltonians" \
  --existing-prediction-root <existing_deeph_prediction_root>

python3 Comparison/scripts/evaluate_hamiltonian_derivative_metrics.py \
  "$STENCIL_ROOT" \
  --method central \
  --split test \
  --require-central \
  --diagnostic-only \
  --support-threshold 1e-12 \
  --max-stencils 1 \
  --source-model graph2mat \
  --output-dir "$STENCIL_ROOT/derivative_metrics/graph2mat"

python3 Comparison/scripts/g2m_deeph_derivative_gate_check.py \
  --derivative-root "$STENCIL_ROOT/derivative_metrics/graph2mat" \
  --output "$STENCIL_ROOT/derivative_metrics/graph2mat/summary/derivative_gate_report.json"

python3 Comparison/scripts/validate_derivative_workflow_artifacts.py \
  "$SMOKE_ROOT" \
  --output-json "$SMOKE_ROOT/derivative_artifact_validation.json"
```

If real SIESTA is available, omit `--existing-reference-root` and let the
default `siesta_command="siesta"` run. If real Graph2Mat or DeepH predictions
are available, omit `--existing-prediction-root` and use the normal checkpoint
or model-dir path required by that backend. The staged smoke above is the
preferred path when expensive runs are unavailable.

The expected key outputs are:

- `derivative_stencil_manifest.json`
- `derivative_geometry_validation.json`
- `structures/<sample>/RUN.fdf`
- `structures/<sample>/metadata.json`
- `siesta_hamiltonians/derivative_siesta_reference_manifest.json`
- `predicted_hamiltonians/derivative_graph2mat_prediction_manifest.json`
- `predicted_hamiltonians/derivative_deeph_prediction_manifest.json`
- `derivative_metrics/graph2mat/manifest.json`
- `derivative_metrics/graph2mat/derivative_matrix_metrics.csv`
- `derivative_metrics/graph2mat/derivative_delta_stability.json`
- `derivative_metrics/graph2mat/summary/derivative_gate_report.json`
- `derivative_artifact_validation.json`
