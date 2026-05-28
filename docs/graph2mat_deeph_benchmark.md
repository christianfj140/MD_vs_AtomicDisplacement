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
