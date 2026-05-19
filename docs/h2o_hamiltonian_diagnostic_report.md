# H2O Hamiltonian Diagnostic Report

Date: 2026-05-19

## 1. Executive verdict

The strongest finding is not a SIESTA read/write failure. A direct Graph2Mat
roundtrip of one archived H2O `siesta.TSHS` preserves Hamiltonian component 0
with only numerical noise:

- max absolute H error: `1.782011906925618e-06 eV`
- all-entry H MAE: `6.178601904766942e-08 eV`
- relative Frobenius H error: `3.212062328196608e-08`
- support F1: `1.0`

The serious red flag is target semantics: the non-orthogonal SIESTA H2O TSHS
stores raw components `(H, S)`. Graph2Mat training labels expose two components
for the reference sample, while the prediction file is written as a polarized
non-orthogonal Hamiltonian container with raw components `(H0, H1, S_identity)`.
For H2O this is not a physical spin target. The repository metric evaluator
mostly protects matrix/spectral metrics by comparing `tocsr(0)` and using the
reference overlap `S_ref`, but the training loss appears to optimize H and S-like
channels together.

On the checked sample, H-channel label error is large while the S-like channel is
small:

- H node+edge MAE contribution: `2.11240291595459`
- common two-component node+edge MAE contribution: `1.0706387758255005`
- S-like component node+edge MAE contribution: `0.028874581679701805`

This can make training loss look much better than Hamiltonian accuracy. It is
the most likely pipeline-level reason the model misses DeepH-level H errors.

## 2. Repository context inspected

Main repository:

- path: `/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement`
- branch: `main`
- commit: `4dd1944648624d8f44d841d02348c26ce0e11439`
- dirty files at inspection time included prior working changes in
  `AtomDisplacement/pipeline_config.yaml`, `Comparison/METRICS.md`,
  `Comparison/scripts/pipeline_ui.py`, `Comparison/ui/app.js`,
  `Comparison/ui/index.html`, `MD/pipeline_config.yaml`,
  `MD/scripts/generate_md_dataset.py`, `README.md`,
  `tests/test_comparison_workflow.py`, and `materials/graphene/`.

Graph2Mat checkout used by the installed environment:

- python: `/home/christian/graph2mat-env/bin/python`
- package file: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/src/graph2mat/__init__.py`
- version: `0.0.13`
- branch: `hamiltonian-spin-colineal-support`
- commit: `ec4f7421b14ded32389a858bbf25e0fde2203d3f`

Primary H2O diagnostic sample:

- evaluation root: `Comparison/results/20260504_123253/cross_evaluations/dataset_100__dataset_0_d0p01_10__d0p02_10__d0p03_10__d0p04_10__d0p05_10__d0p06_10__d0p07_10__d0p08_10__d0p09_10__d0p10_10__atom_displacement__on__test_md`
- sample: `md_94`
- structure: `structures/md_94/RUN.fdf`
- reference: `siesta_hamiltonians/md_94/siesta.TSHS`
- prediction: `predicted_hamiltonians/md_94/ML_prediction.HSX`
- basis files: `MD/dataset/MD_steps/basis/H.ion.xml`,
  `MD/dataset/MD_steps/basis/O.ion.xml`
- wrong-S sensitivity sample: `md_91`

Relevant implementation paths inspected:

- `Comparison/scripts/evaluate_hamiltonian_metrics.py`
- `Comparison/scripts/predict_model_on_dataset.py`
- `MD/scripts/run_md_training.py`
- `MD/scripts/run_md_prediction.py`
- `MD/scripts/run_md_testing.py`
- `MD/scripts/md_pipeline_config.py`
- `MD/pipeline_config.yaml`
- `MD/training/config.yaml`
- `AtomDisplacement/pipeline_config.yaml`
- `AtomDisplacement/scripts/run_atdisp_prediction.py`
- `Comparison/scripts/pipeline_ui.py`
- `Comparison/scripts/write_graph2mat_configs.py`
- `Comparison/METRICS.md`
- `README.md`
- `requirements-graph2mat.txt`
- `tests/test_comparison_workflow.py`
- `tests/test_metrics_material_compatibility.py`
- Graph2Mat:
  - `src/graph2mat/core/data/configuration.py`
  - `src/graph2mat/core/data/processing.py`
  - `src/graph2mat/core/data/sparse.py`
  - `src/graph2mat/core/data/metrics.py`
  - `src/graph2mat/tools/lightning/data.py`
  - `src/graph2mat/tools/lightning/model.py`
  - `src/graph2mat/tools/lightning/callbacks.py`

## 3. Diagnostic table

| Diagnostic | Status | Evidence | Numerical result | Likely implication | Next action |
|---|---|---|---|---|---|
| 1. Single-sample overfit | NOT RUN | No fresh MACE fit was launched in this audit. Full training is nontrivial and should be isolated from production results. | Not available. | Cannot yet distinguish memorization failure from generalization failure. | Run the one-sample overfit recipe after fixing or explicitly controlling H-only target semantics. |
| 2. SIESTA H read/write Graph2Mat roundtrip | PASS | `TorchBasisMatrixData.new(... labels=True)` plus `MatrixDataProcessor.matrix_from_data(... threshold=None)` on `md_94`. | relative Frobenius `3.212062328196608e-08`; support F1 `1.0`; max abs `1.782011906925618e-06 eV`. | Orbital ordering, basis mapping, and reconstruction of H component 0 are probably not the main failure. | Keep this as a regression test when changing target component handling. |
| 3. Symmetry/Hermiticity | PASS | Direct sparse Hermiticity checks on reference and prediction H component 0. | reference relative defect `8.910017839076371e-11`; prediction relative defect `0.0`. | The checked H2O matrices are real/Hermitian enough; bad spectra are not explained by non-Hermitian H. | Check more samples after fixing target semantics. |
| 4. Overlap S validation | FAIL | Evaluation uses `S_ref` for spectra, but the prediction container's own overlap is not the reference overlap. | predicted `read_overlap()` vs `S_ref` relative Frobenius `0.7212945472064474`; predicted component 1 vs `S_ref` relative Frobenius `0.11059358783973933`; wrong sample `S` vs `S_ref` relative Frobenius `0.07717461605063569`. | Production spectral metrics are paired with `S_ref`, but `ML_prediction.HSX` is not a physically self-contained non-spin H2O Hamiltonian+S file. External use of the predicted HSX overlap is unsafe. | Keep spectral evaluation on `S_ref`; redesign prediction writing so the auxiliary component is not represented as physical spin and document/preserve S explicitly. |
| 5. Scale, units, normalization, loss weighting | FAIL | Graph2Mat labels for reference are `[219, 2]` point and `[155, 2]` edge; prediction labels are `[219, 3]` point and `[155, 3]` edge. Graph2Mat `block_type_mae` returns node mean absolute error plus edge mean absolute error. | H component node MAE `1.4565106630325317 eV`; edge MAE `0.6558921933174133 eV`; S-like component node MAE `0.005997497122734785`; edge MAE `0.022877084091305733`. | The current loss can average a dimensionless/easier S-like channel with H in eV, making loss scale scientifically misaligned with Hamiltonian accuracy. | Train/evaluate an H-only target or explicitly split H and S losses with documented weights. |

## 4. Evidence and commands

Diagnostic output:

- JSON: `Comparison/results/diagnostics/h2o_hamiltonian/h2o_hamiltonian_diagnostics.json`
- script: `Comparison/scripts/diagnose_h2o_hamiltonian_pipeline.py`

Command run:

```bash
/home/christian/graph2mat-env/bin/python Comparison/scripts/diagnose_h2o_hamiltonian_pipeline.py \
  --evaluation-root Comparison/results/20260504_123253/cross_evaluations/dataset_100__dataset_0_d0p01_10__d0p02_10__d0p03_10__d0p04_10__d0p05_10__d0p06_10__d0p07_10__d0p08_10__d0p09_10__d0p10_10__atom_displacement__on__test_md \
  --sample md_94 \
  --second-sample md_91 \
  --basis-glob 'MD/dataset/MD_steps/basis/*.ion.xml' \
  --output-dir Comparison/results/diagnostics/h2o_hamiltonian
```

Sparse metrics on `md_94`:

- `mae_ref_eV`: `1.6070844148621144`
- `rmse_ref_eV`: `2.292263756856704`
- `mae_union_eV`: `1.6070844148621144`
- `rmse_union_eV`: `2.292263756856704`
- `mse_union_eV2`: `5.254473130998812`
- `r2_union`: `0.8691910600246407`
- `support_f1`: `0.8267148014440433`
- `relative_frobenius_union`: `0.36023784139603376`
- `max_abs_error_union_eV`: `7.997146011682471`

Reference matrix:

- shape: `[23, 23]`
- `nnz`: `529`
- metric component count: `1`
- raw sisl dimension: `2`
- raw component 0: H, norm `114.71413553064836`
- raw component 1: S, norm `6.924104100152673`
- spin: `Spin{unpolarized}`
- orthogonal: `False`
- overlap readable: `True`
- Fermi level: `-4.503062250825039 eV`

Prediction matrix:

- shape: `[23, 23]`
- `nnz`: `529`
- metric component count: `2`
- raw sisl dimension: `3`
- raw component 0: predicted H, norm `114.40752621982047`
- raw component 1: spin-like/S-like auxiliary, norm `6.901418235528647`
- raw component 2: overlap as read by sisl, norm `4.795831523312719`
- spin: `Spin{polarized}`
- orthogonal: `False`
- Fermi level read from prediction: `0.0` but metrics use reference Fermi.

Spectral probe:

- `H_pred` with `S_ref`: global RMSE `46.927428911097806 eV`, Fermi-window RMSE `6.533595400803923 eV`
- `H_ref` with wrong `S` from `md_91`: global RMSE `3.3744806857341794 eV`, Fermi-window RMSE `0.04234734421844433 eV`
- `H_pred` with wrong `S` from `md_91`: global RMSE `45.91751115765589 eV`, Fermi-window RMSE `6.4314279949689155 eV`

Interpretation: S mismatch is dangerous and non-negligible, but for this checked
prediction the huge spectral error is dominated by H error, not only by S
selection. The repository evaluator uses `reference.overlap` for predicted
spectra in `Comparison/scripts/evaluate_hamiltonian_metrics.py:646-648` and
records `overlap_source = siesta_reference`.

## 5. Code-path findings

SIESTA and metric loading:

- `evaluate_hamiltonian_metrics.read_matrix()` reads `sile.read_hamiltonian()`,
  takes `hamiltonian_obj.tocsr(0)` as H, and separately calls `sile.read_overlap()`
  (`Comparison/scripts/evaluate_hamiltonian_metrics.py:763-790`).
- Spectral metrics solve `H_ref` with `S_ref` and `H_pred` with `S_ref`
  (`Comparison/scripts/evaluate_hamiltonian_metrics.py:646-648`).
- The evaluator already has a compatibility warning for Graph2Mat auxiliary
  predictions (`graph2mat_auxiliary_component_ignored`) at
  `Comparison/scripts/evaluate_hamiltonian_metrics.py:930-969`.

Graph2Mat conversion:

- Graph2Mat builds `OrbitalConfiguration` by calling the sisl reader associated
  with `out_matrix`; for `hamiltonian`, it calls `read_hamiltonian`
  (`graph2mat/core/data/configuration.py:585-610`).
- For non-orthogonal SIESTA Hamiltonians, the raw sisl sparse data includes H and
  S components.
- When converting multiple CSR components back to a sisl Hamiltonian, Graph2Mat
  currently treats two components as polarized spin and three components as
  polarized spin plus overlap (`graph2mat/core/data/sparse.py:438-466`).

Training/loss:

- Current configs use `out_matrix: hamiltonian`, `symmetric_matrix: true`,
  `n_matrix_components: 2`, and `loss: graph2mat.metrics.block_type_mae` in the
  active pipeline configs.
- `block_type_mae` computes `abs(node_error).mean() + abs(edge_error).mean()`
  (`graph2mat/core/data/metrics.py:212-221`).
- `MatrixDataModule` passes `n_matrix_components` to `MatrixDataProcessor`, but
  the checked labels still expose H/S-like components from the non-orthogonal
  matrix path (`graph2mat/tools/lightning/data.py:240-267`).

## 6. Ranked likely root causes

1. H and S are mixed as training target components for a non-spin H2O task.
   - Evidence supporting it: reference Graph2Mat labels are two-component;
     raw reference sisl dim is 2; prediction raw sisl dim is 3 and spin metadata
     is `Spin{polarized}`; the S-like channel error is tiny relative to the H
     channel.
   - Evidence against it: repository sparse/spectral metrics compare H component
     0 and therefore are not directly using S as H at evaluation time.
   - How to confirm/falsify: run a one-sample overfit after forcing an H-only
     label path. If H-only overfit reaches sub-meV while current two-component
     training does not, this is confirmed.
   - Suggested fix: introduce an explicit H-only target extraction policy for
     non-orthogonal Hamiltonians, or split H and S into separate named targets
     with explicit loss weights. Do not encode the second target as physical spin.

2. Prediction HSX is not a self-contained physical non-spin H2O Hamiltonian.
   - Evidence supporting it: predicted `read_overlap()` differs from `S_ref` by
     relative Frobenius `0.7212945472064474`; prediction is read as
     `Spin{polarized}`.
   - Evidence against it: repository spectral metrics use `S_ref`, so this is
     not the immediate cause inside the current evaluator.
   - How to confirm/falsify: read `ML_prediction.HSX` with independent tools and
     compare eigenvalues using its own overlap vs `S_ref`.
   - Suggested fix: write predictions in a documented H-only container or archive
     the matching reference S next to predictions and forbid external use of the
     predicted file's own overlap for spectral science.

3. Model/loss capacity and objective are likely too weak after the data-path bug.
   - Evidence supporting it: current configs inspected include small settings
     such as `num_interactions: 1`, `correlation: 1`, `max_ell: 2`, and
     `10x0e + 10x1o + 10x2e` in several active/generated configs.
   - Evidence against it: a single-sample overfit has not yet been run, so
     capacity cannot be separated from target bugs.
   - How to confirm/falsify: run one-sample overfit with H-only labels and a
     larger model.
   - Suggested fix: only after H-only target semantics pass roundtrip/overfit,
     tune capacity, learning rate, batch size, and block weighting.

4. Raw-global-H metrics are not DeepH H-prime metrics.
   - Evidence supporting it: repository metric target space is
     `raw_global_hamiltonian`.
   - Evidence against it: this does not explain bad self-consistency, but it
     matters for comparing absolute numbers to DeepH.
   - How to confirm/falsify: implement/validate a local-frame H-prime transform.
   - Suggested fix: keep reporting current metrics honestly as repo-compatible
     diagnostics until H-prime exists.

## 7. Recommended fixes, not implemented here

1. Add an explicit Hamiltonian target component policy.
   - For non-orthogonal SIESTA H2O, train H component 0 only unless a separate
     overlap objective is deliberately requested.
   - Add a regression test that `TorchBasisMatrixData.new(... out_matrix="hamiltonian")`
     for non-spin H2O exposes exactly one H target component under that policy.

2. Fix or wrap MatrixWriter output semantics.
   - Avoid representing auxiliary H/S channels as spin-polarized Hamiltonians for
     a non-spin task.
   - Ensure any predicted file either has a valid documented overlap or is never
     used as a self-contained generalized-eigenproblem input.

3. Run the single-sample overfit test.
   - Train one H2O sample as train/val/test/predict.
   - Acceptance target: H matrix MAE should approach sub-meV or very low meV if
     labels, ordering, reconstruction, and capacity are correct.
   - If it stalls at tens of meV after H-only cleanup, inspect model capacity,
     optimizer, and label ordering in the model output path.

4. Revisit loss weighting.
   - Report node/edge H losses separately.
   - Consider block-normalized H-only MAE/MSE or a weighted H loss that does not
     dilute H with dimensionless S.

5. Re-evaluate spectral metrics after H-only predictions.
   - Keep using `S_ref` for predicted spectra unless a validated predicted S path
     exists.
   - Add a hard warning when users try to use `ML_prediction.HSX` own overlap.

## 8. Reproduction commands

Repository inspection:

```bash
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short

/home/christian/graph2mat-env/bin/python - <<'PY'
import graph2mat, inspect, sys
print('python', sys.executable)
print('graph2mat', getattr(graph2mat, '__version__', 'unknown'))
print('graph2mat_file', inspect.getfile(graph2mat))
PY

git -C /home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat branch --show-current
git -C /home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat rev-parse HEAD
git -C /home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat status --short
```

Diagnostic script:

```bash
python3 -m py_compile Comparison/scripts/diagnose_h2o_hamiltonian_pipeline.py

/home/christian/graph2mat-env/bin/python Comparison/scripts/diagnose_h2o_hamiltonian_pipeline.py \
  --evaluation-root Comparison/results/20260504_123253/cross_evaluations/dataset_100__dataset_0_d0p01_10__d0p02_10__d0p03_10__d0p04_10__d0p05_10__d0p06_10__d0p07_10__d0p08_10__d0p09_10__d0p10_10__atom_displacement__on__test_md \
  --sample md_94 \
  --second-sample md_91 \
  --basis-glob 'MD/dataset/MD_steps/basis/*.ion.xml' \
  --output-dir Comparison/results/diagnostics/h2o_hamiltonian
```

Targeted validation run:

```bash
python3 -m unittest tests.test_comparison_workflow.ComparisonWorkflowTests.test_md_rewrite_makes_effective_geometries_differ_when_xv_differs tests.test_comparison_workflow.ComparisonWorkflowTests.test_md_step_xv_path_accepts_unique_system_label_xv
```

Single-sample overfit recipe, not run in this audit:

1. Create an isolated workspace under
   `Comparison/results/diagnostics/h2o_hamiltonian/one_sample_overfit/`.
2. Copy `structures/md_94/RUN.fdf` and
   `siesta_hamiltonians/md_94/siesta.TSHS` into train, validation, test, and
   prediction sample directories.
3. Copy H/O `.ion.xml` basis files.
4. Generate a Graph2Mat config with `batch_size: 1`, fixed seed if supported,
   `store_in_memory: true`, `symmetric_matrix: true`, and an H-only target policy.
5. Run the verified training command pattern:

```bash
graph2mat models mace main fit -c config.yaml
```

This recipe is intentionally marked unverified until the H-only target policy is
implemented or explicitly configured.

## 9. What remains unverified

- A true single-sample overfit was not run.
- Only one primary H2O sample (`md_94`) plus one wrong-overlap sample (`md_91`)
  was numerically diagnosed.
- No production Graph2Mat or repository training logic was changed.
- No broad re-evaluation was run after the diagnostic script.
- The exact API/design for an H-only non-orthogonal Hamiltonian target still
  needs to be chosen and tested.
