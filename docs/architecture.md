# Architecture

## Overview

This repository contains three related but distinct workflow families:

- `MD/` for the molecular-dynamics pipeline.
- `AtomDisplacement/` for H2O atom-displacement workflows and the generic
  Cartesian displacement path.
- `Comparison/` for the main comparison UI, comparison runners, metrics, and
  Graph2Mat-vs-DeepH benchmark tooling.

Around those families, the repository also carries:

- `materials/` as the versioned source of truth for bundled materials;
- `shared/` validators for material bundles, SIESTA FDF materialization, and
  joint Graph2Mat/DeepH dataset contracts;
- `docs/` as the user-facing map of the current workflow surface.

The repository is organized as a file-driven workflow system. The Python
scripts read YAML or JSON configuration, validate local SIESTA artifacts,
launch external tools, and write manifests, metrics, and reports back to the
workspace.

## Repository Layout

| Path | Responsibility |
| --- | --- |
| `MD/` | Standalone MD pipeline, its config, dataset, and UI. |
| `AtomDisplacement/` | Standalone atom-displacement pipeline, relaxed/base inputs, dataset, and UI. |
| `Comparison/` | Shared comparison workflows, benchmark configs, datasets, UI, results, and docs. |
| `shared/` | Common validation and material-bundle helpers used by more than one workflow family. |
| `materials/` | Versioned material bundles such as `h2o`, `graphene`, `graphene_5x2`, `graphene_5x5`, `si_amorphous`, and `si_vacancy`. |
| `configs/` | Auxiliary Graph2Mat config files. |
| `scripts/` | Environment helpers and Torch serialization compatibility shims. |
| `tests/` | `unittest`-style regression tests for the workflows and helper modules. |
| `docs/` | Repository documentation, including the benchmark runbook. |

## Major Modules

### Comparison

- `Comparison/scripts/pipeline_ui.py` serves the combined HTTP UI and exposes
  an API for reading and patching run payloads, starting runs, and streaming
  logs. It writes per-run configuration snapshots into the selected workspace
  and result directories.
- `Comparison/ui/` is the browser client for the comparison UI, including
  experiment orchestration, `G2M vs DeepH`, `ML vs SIESTA`, and
  dataset-size-minimum panels.
- `Comparison/scripts/g2m_deeph_runner.py` is the backend runner for the
  Graph2Mat-vs-DeepH benchmark. It stages datasets, runs Graph2Mat and DeepH
  phases, and writes manifests and metrics.
- `Comparison/scripts/g2m_deeph_final_workflow.py` implements the staged
  public/final workflow with explicit stages such as protocol validation,
  search, selection, final runs, final-test evaluation, and report generation.
- `Comparison/scripts/g2m_deeph_dataset_size_minimum.py` performs the
  postprocessed dataset-size-minimum analysis used by the UI and archived
  summaries.
- `Comparison/scripts/ml_vs_siesta_benchmark.py` is a lightweight CLI and UI
  bridge for `ML vs SIESTA` planning, displacement generation, mixing, and
  dry-run validation.
- `Comparison/scripts/g2m_deeph_protocol.py` validates the paper-ready
  benchmark protocol schema.
- `Comparison/scripts/deeph_config.py` builds DeepH preprocess/train/
  inference configs and validates the required joint SIESTA artifacts.
- `Comparison/scripts/method_registry.py` is the canonical source for method
  identifiers, display names, legacy aliases, and result-directory names.

### MD and AtomDisplacement

- `MD/scripts/main_md.py` runs the MD pipeline steps in the order declared in
  `MD/pipeline_config.yaml`.
- `MD/scripts/pipeline_ui.py` exposes the MD pipeline through a local HTTP UI
  and starts `MD/scripts/main_md.py` in a subprocess.
- `AtomDisplacement/scripts/main_atom_displacement.py` runs the atom-displacement
  pipeline step list declared in `AtomDisplacement/pipeline_config.yaml`.
- `AtomDisplacement/scripts/pipeline_ui.py` exposes the atom-displacement
  pipeline through a local HTTP UI and starts
  `AtomDisplacement/scripts/main_atdisp.py`.
- `AtomDisplacement/scripts/generate_generic_cartesian_displacement_dataset.py`
  is the generic Cartesian displacement generator referenced by the README and
  the tests.

### Shared Helpers

- `shared/material_bundle.py` validates material bundles, extracts species from
  the FDF file, checks pseudopotential and basis coverage, and records hashes.
- `shared/material_presets.py` resolves named presets such as `h2o`.
- `shared/joint_artifact_contract.py` validates the benchmark snapshot contract
  used by the Graph2Mat-vs-DeepH workflow.
- `shared/benchmark_manifest.py` centralizes dataset/run manifest generation
  for benchmark-style outputs and dataset reuse.
- `shared/siesta_run_fdf.py` and related helpers materialize SIESTA FDF inputs
  and preserve provenance.

## Data Flow

The main comparison flow is:

1. Read a selected material bundle, dataset recipe, or archived dataset
   manifest.
2. Validate the bundle and the local SIESTA snapshot artifacts.
3. Create an isolated workspace under `Comparison/workspaces/`.
4. Run the selected pipeline stages for Graph2Mat and, when requested, DeepH.
5. Write results, metrics, plots, manifests, and recommendations under
   `Comparison/results/<run_id>/`.
6. Aggregate the outputs into comparison summaries and UI-visible tables.

The same `Comparison` surface also serves auxiliary API families such as:

- `/api/material/*` for preset discovery and bundle validation;
- `/api/g2m-deeph/*` for joint-dataset validation, benchmark runs, plots,
  derivative metrics, and dataset-size-minimum summaries;
- `/api/ml-vs-siesta/*` and `/api/mixing/*` for the lightweight benchmark
  helpers and mixed-dataset planning/materialization.

The MD and AtomDisplacement flows follow the same pattern on a smaller scale:
configuration is read from the local `pipeline_config.yaml`, the pipeline runs
the declared step scripts, and outputs are written into the corresponding
`dataset/` and `training/` directories.

## Public Versus Internal Interfaces

### Stable or user-facing surfaces

- `Comparison/scripts/pipeline_ui.py`
- `MD/scripts/pipeline_ui.py`
- `AtomDisplacement/scripts/pipeline_ui.py`
- `Comparison/scripts/g2m_deeph_final_workflow.py`
- `Comparison/scripts/g2m_deeph_protocol.py`
- `Comparison/scripts/g2m_deeph_smoke.py`
- `Comparison/scripts/g2m_deeph_dataset_size_minimum.py`
- `Comparison/scripts/ml_vs_siesta_benchmark.py`
- `Comparison/scripts/verify_dataset_integrity.py`
- `Comparison/scripts/evaluate_hamiltonian_metrics.py`
- `MD/scripts/main_md.py`
- `AtomDisplacement/scripts/main_atom_displacement.py`

These are the entry points a new user or contributor is most likely to invoke.

### Internal implementation details

- Helper modules whose names end in `_utils.py`, `_config_utils.py`, or
  `_adapter.py`.
- Generated workspace contents under `Comparison/workspaces/`.
- Temporary manifests, logs, and intermediate configs written by the runners.
- UI-specific helper functions and private methods inside the HTTP server
  scripts.

The internal helpers are still important for the current implementation, but
they should be treated as changeable details rather than stable APIs.
