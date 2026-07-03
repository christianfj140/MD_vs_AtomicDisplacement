# Data And Outputs

## Input Data Types

### Pipeline Configuration

- `MD/pipeline_config.yaml`
- `AtomDisplacement/pipeline_config.yaml`
- UI-edited JSON/YAML payloads written by the pipeline servers

These files define paths, commands, training settings, split settings, and the
pipeline step order.

### Benchmark Recipes And Protocols

- `Comparison/dataset_recipes/*.json`
- `Comparison/config/*.json`
- `Comparison/config/*.py` payload generators
- `Comparison/scripts/g2m_deeph_protocol.py` protocol validation inputs

These files describe dataset recipes, run configuration, and the final
Graph2Mat-vs-DeepH protocol.

### Material Bundles

- `materials/*/RUN.fdf`
- `materials/*/pseudos/*`
- `materials/*/basis/*`
- `materials/*/material.yaml`
- `shared/material_bundle.py`

The material bundle validator expects the FDF file, pseudopotential coverage,
and optional basis coverage to match the species declared in the FDF file.
Versioned presets currently shipped in-tree are:

- `materials/h2o/`
- `materials/graphene/`
- `materials/graphene_5x2/`
- `materials/graphene_5x5/`
- `materials/si_amorphous/`
- `materials/si_vacancy/`

### Snapshot And Dataset Artifacts

The benchmark workflows operate on snapshot directories that include:

- `RUN.fdf`
- `RUN.out` or `siesta.out`
- `SystemLabel.TSHS`
- `SystemLabel.TSDE`
- `SystemLabel.HSX`
- `SystemLabel.STRUCT_OUT`
- `SystemLabel.XV`
- `SystemLabel.ORB_INDX`
- `metadata.json`

The joint benchmark also expects dataset-level manifests such as:

- `artifact_validation.json`
- `benchmark_dataset_manifest.json`
- `frozen_split_manifest.json`

## Output Locations

### MD

- `MD/dataset/MD_steps/`
- `MD/dataset/splits/`
- `MD/training/`

The MD config points `dataset_dir` and `training_dir` at these locations.

### AtomDisplacement

- `AtomDisplacement/dataset/FC_steps/`
- `AtomDisplacement/dataset/RandomCartesian_steps/`
- `AtomDisplacement/dataset/collected/`
- `AtomDisplacement/dataset/samples_manifest.json`
- `AtomDisplacement/dataset/run_summary.json`
- `AtomDisplacement/training/`

The atom-displacement config also names these summary files explicitly in
`AtomDisplacement/pipeline_config.yaml`.

### Comparison

- `Comparison/workspaces/` for scratch space
- `Comparison/results/` for archived runs
- `Comparison/results/<run_id>/` for run-specific manifests, metrics, plots,
  and recommendations
- `Comparison/results/results_md/`
- `Comparison/results/results_atomdisp/`
- `Comparison/results/results_random_cartesian/`
- `Comparison/results/dataset_size_minimum_*/` for dataset-size-minimum
  summaries, reports, PDFs, and PNGs

The benchmark runner writes structured manifests, metrics CSVs, and summary
JSON files into these directories.

### DeepH Joint Dataset

- `Comparison/datasets/graphene_w90_joint/`

This directory is the persistent local home for reusable benchmark datasets.
It is expected to remain separate from `Comparison/workspaces/` and
`Comparison/results/`.

## Naming Conventions

- Canonical method IDs: `md`, `siesta_fc_cartesian`, `random_cartesian`.
- Legacy aliases: `atom_displacement`, `atomdisp`.
- Result directories: `results_md`, `results_atomdisp`, `results_random_cartesian`.
- Run directories: `run_<run_id>`.
- Sample IDs in the atom-displacement config are generated as
  `sample_{index:04d}`.

These names are part of the current repository contract and are used by the UI
and the test suite.

## What Gets Written Where

### Manifests And Reports

- `manifest.json` files record run metadata, selected inputs, and hashes.
- `*_manifest.json` files record dataset or artifact validation state.
- `recommendation.json` captures the comparison summary written by the main
  comparison workflow.
- `performance_report.json` records performance-oriented run information.
- staged `stages/<stage>.json` files record Graph2Mat-vs-DeepH final-workflow
  status transitions.
- `dataset_size_minimum_summary.json` and
  `dataset_size_minimum_report.md` summarize postprocessed minimum-size claims.

### Metrics And Plots

Comparison runs write metric CSV files such as:

- `metrics/sparse_metrics.csv`
- `metrics/spectral_metrics.csv`
- `metrics/dos_metrics.csv`
- `metrics/matrix_spectrum_relationship.csv`
- `metrics/orbital_pair_metrics.csv`
- `metrics/orbital_pair_summary.csv`

The exact set depends on the workflow stage and the available artifacts.
Derivative-specific layouts can additionally write:

- `derivative_stencil_manifest.json`
- `derivative_geometry_validation.json`
- `derivative_metrics/<model>/manifest.json`
- `derivative_metrics/<model>/derivative_matrix_metrics.csv`
- `derivative_metrics/<model>/derivative_hermiticity.csv`
- `derivative_metrics/summary/derivative_gate_report.json`

### Logs And Intermediate Files

The workflow runners also create:

- process logs for SIESTA, Graph2Mat, and DeepH subprocesses
- staged configs in the relevant workspace
- prediction outputs such as `ML_prediction.HSX`
- frozen test manifests under `common_tests/`
- optional mixed-dataset manifests and staging payloads for `ML vs SIESTA`
  helpers

## Reproducibility Notes

- Outputs under `Comparison/results/` are only reproducible if the same input
  dataset, split manifests, material bundle, and external toolchain are
  available.
- Archived comparison results may depend on external SIESTA or DeepH runs and
  should be treated as provenance-bearing artifacts, not as rebuildable cache.
- Some benchmark outputs are explicitly marked diagnostic-only or exploratory
  when the required evidence is incomplete.

## Known Data Dependencies

- The comparison workflows rely on local filesystem layout conventions.
- The benchmark runners expect external executables such as `siesta`,
  `graph2mat`, and the DeepH command set to be discoverable in the current
  environment.
- The material-bundle validator assumes FDF species declarations are the source
  of truth for pseudopotential coverage.
