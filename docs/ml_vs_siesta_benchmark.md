# ML vs SIESTA benchmark

Infrastructure for a benchmark flow that compares Graph2Mat / DeepH matrix
predictions (and their finite-difference derivatives) against SIESTA references.

> **This toolkit is infrastructure only.** It never launches SIESTA, never trains
> a model and never needs a GPU. It builds inputs, validates plans, computes
> errors and serves UI payloads. Heavy model inference and real SIESTA parsing
> stay in the existing `Comparison/scripts` runners and are plugged in through
> thin adapters.

Current integration points:

- CLI: `Comparison/scripts/ml_vs_siesta_benchmark.py`
- UI/API bridge: `Comparison/scripts/pipeline_ui.py`
- Routes: `/api/ml-vs-siesta/*` and `/api/mixing/*`

## Where things live

- Package: `Comparison/scripts/ml_vs_siesta/`
- CLI entrypoint: `Comparison/scripts/ml_vs_siesta_benchmark.py`
- Example config: `Comparison/config/ml_vs_siesta_benchmark_example.yaml`
- Example structure: `Comparison/config/ml_vs_siesta_example_structure.fdf`
- Tests: `tests/test_ml_vs_siesta_benchmark.py`
- UI: sidebar tab **ML vs SIESTA** (backend routes under `/api/ml-vs-siesta/*`)

## Config

See `Comparison/config/ml_vs_siesta_benchmark_example.yaml`. Fields:

- `system.input_structure`, `system.supercell` (default `[5, 5, 1]`),
  `system.central_atom` (`auto` or an integer)
- `derivatives.enabled`, `derivatives.displacement`, `derivatives.directions`
- `models.enabled` (`graph2mat`, `deeph`)
- `matrices.targets` (`hamiltonian`, `density_matrix`, `overlap`)
- `dataset_mixing.enabled`
- `species_transfer.*`
- `ui.enable_matrix_viewer`

```python
from ml_vs_siesta import load_benchmark_config
config = load_benchmark_config("Comparison/config/ml_vs_siesta_benchmark_example.yaml")
```

## Generate 5×5×1 `.fdf` inputs (reference + ±h)

```bash
python Comparison/scripts/ml_vs_siesta_benchmark.py generate-siesta-displacements \
    --config Comparison/config/ml_vs_siesta_benchmark_example.yaml \
    --output Comparison/results/ml_vs_siesta_displacements
```

Add `--dry-run` to only list the files that would be written. This produces
`reference/RUN.fdf`, `x_plus/`, `x_minus/`, … `z_minus/` and a `metadata.json`
with the central atom, supercell reps, displacement and file paths. It **does
not** run SIESTA.

## Prepare derivatives

Derivatives are central finite differences of matrix predictions:

```python
from ml_vs_siesta import finite_difference_matrix_derivative, compare_derivatives_to_siesta
```

- `finite_difference_matrix_derivative(predictor, structure, atom, dir, h, targets)`
  returns `(M_plus − M_minus) / (2h)` per target.
- `compare_derivatives_to_siesta(...)` compares those against SIESTA derivative
  matrices already present on disk (fixture format: `<dir>/<target>.npy`).
- A differentiable, torch-only variant is available:
  `torch_finite_difference_matrix_derivative(...)`.

Plug a real model by wrapping its inference in a predictor:

```python
from ml_vs_siesta import Graph2MatPredictor
predictor = Graph2MatPredictor(predict_fn=my_inference_callable)  # callable(structure, targets)
```

Without `predict_fn`, `Graph2MatPredictor`/`DeepHPredictor` raise a clear
`NotImplementedError` pointing at the existing runners.

## Datasets small/large

```bash
python Comparison/scripts/ml_vs_siesta_benchmark.py mix-datasets \
    --small small.json --large large.json --mode add \
    --ratios 0.0,0.1,0.25,0.5,0.75,1.0 --seed 0 \
    --output manifest.json --configs-dir configs/mixed_datasets
```

- `mode=add`: keep all small, append a fraction of large.
- `mode=replace`: keep total size constant, swap small for large.
- Output format inferred from the suffix (`.json` / `.yaml` / `.csv`).
- `classify_dataset_by_size(dataset, threshold_atoms)` splits by atom count.

## Diagnose species transfer

```bash
python Comparison/scripts/ml_vs_siesta_benchmark.py inspect-species \
    --config model_or_species_config.yaml --new-species H
```

`inspect_species_support(...)` reports supported species/pairs, missing blocks,
whether new embeddings/heads are needed and a status
(`supported` / `partially_supported` / `not_implemented`).
`prepare_species_expansion(...)` delegates to a model's own expansion hook if it
exists, otherwise raises a clear `NotImplementedError` with the report attached.
**No weights are modified and no training happens.**

## End-to-end dry-run

```bash
python Comparison/scripts/ml_vs_siesta_benchmark.py benchmark-dry-run \
    --config Comparison/config/ml_vs_siesta_benchmark_example.yaml
```

Validates config → structure → supercell → central atom → displacements →
expected SIESTA paths → predictors → targets → UI options → dataset/species
options, and prints a JSON summary. Nothing heavy runs.

## Mixing datasets sweep (small + 5×5×1 large → MAE vs size)

Study whether mixing small-cell (2-atom) datasets with large 5×5×1 supercell
datasets improves Hamiltonian prediction. For each dataset size `N`, two mixing
modes are swept:

- **add**: keep 100% of small(N), append a fraction of large(N) → total grows.
- **replace**: hold the total at `N`, sweep composition 100/0 → 0/100 small/large.

Each permutation becomes a real, runner-ready merged `dataset_root` (its own
combined train/val/test split), so Graph2Mat + DeepH can train/predict/evaluate
per permutation and the result is plotted as **MAE vs total dataset size**, one
curve per `(mode, ratio, model)`.

### Modules

- `Comparison/scripts/ml_vs_siesta/mixed_dataset_materialize.py` —
  `materialize_mixed_dataset(...)` turns a selection of small/large sample ids
  into a merged `dataset_root` (merged frozen split + regenerated benchmark
  manifests, reusing `shared/benchmark_manifest.write_benchmark_manifests`). It
  validates species/basis compatibility first (`validate_datasets_compatible`).
- `Comparison/scripts/ml_vs_siesta/mixing_sweep.py` — `plan_mixing_sweep` (pure)
  and `run_mixing_sweep` (materialize + train via an injectable `launch_fn`).
- `Comparison/scripts/ml_vs_siesta/plot_mixing_mae_vs_size.py` —
  `aggregate_mae_vs_size` / `write_mae_vs_size_outputs` (JSON + PNG).

### CLI (dry-run preview)

```bash
python Comparison/scripts/ml_vs_siesta_benchmark.py mixing-sweep \
    --small 20=Comparison/datasets/.../graphene_w90_scale_iid20 \
    --large 20=Comparison/datasets/.../graphene_5x5x1_iid20 \
    --modes add,replace --ratios 0.0,0.2,0.4,0.6,0.8,1.0 --dry-run
```

Drop `--dry-run` and pass `--output-root Comparison/results/runs/mix` to
materialize the merged datasets (still no training — training is launched via
the runner / UI).

### Placing datasets

- **Small** datasets already exist (`Comparison/datasets/graphene_*_scale_iidN`).
- **Large 5×5×1** datasets do not exist yet: generate the 5×5×1 references with
  SIESTA (heavy, done by you), then place each size-`N` dataset the same way. The
  small and large datasets **must share the carbon PAO basis and species**
  (validated automatically; a mismatch raises a clear error).

### One-click flow (UI)

Open the **Mixing datasets** sidebar tab:

1. *Descubrir datasets* — lists available small (2-atom) and large (5×5×1)
   datasets grouped by snapshot count.
2. *Configurar sweep* — small/large `N=path` maps, modes, ratios, seed, models.
3. *Previsualizar (dry-run)* — shows the permutation table (`/api/mixing/plan`).
4. *Materializar merged datasets* — builds the merged `dataset_root`s in the
   background (`/api/mixing/launch`, polled via `/api/mixing/status`).
5. *MAE vs tamaño* — chart of `h_mae_eV` vs total size; *Cargar demo* shows a
   synthetic curve, *Cargar métricas reales* reads `/api/mixing/metrics` once
   training has produced records.

> The real flow is **preview → materialize → train (real) → load metrics**.
> *Previsualizar* is a dry-run plan; *Materializar* builds the merged
> `dataset_root`s (cheap, snapshots are symlinked); the **Entrenar sweep
> (real)** button (`action == "train"` on `/api/mixing/launch`) materializes
> everything and then drives ONE Graph2Mat/DeepH runner invocation with full
> training parallelism — this launches real training subprocesses and needs
> the models installed (GPU recommended). *Cargar métricas reales* then reads
> `/api/mixing/metrics`, which reflects per-permutation `h_mae_eV` records and
> `trained`/`partial`/`failed` statuses. Programmatic use without the UI: pass
> a `launch_fn` to `run_mixing_sweep` returning
> `{"metrics": {model: {"h_mae_eV": ...}}}`.

#### Split policy

`materialize_mixed_dataset` / `run_mixing_sweep` accept
`split_policy` (UI payload key `split_policy`):

- `"resplit_combined"` (default, legacy behaviour): the merged pool is
  re-split by seed, so the test set changes with the selection — an MAE
  improvement between ratios can come from the test set changing, not from
  the composition.
- `"fixed_common_test"` (**recommended for scientific analysis**): the test
  set is a fixed fraction of the small pool derived only from
  `small_root` + `seed`, so all permutations of the same size/seed share
  exactly the same test snapshots.

Each merged dataset also writes a self-contained
`mixed_dataset_provenance.json` (mode, ratio + semantics, seed, selected ids,
split policy, compatibility check) so any mixture can be reproduced.

#### Graph2Mat autograd derivatives (dH_pred/dR)

The autograd route (`run_graph2mat_autograd_derivative_predictions.py`) is
**CPU-only** (MACE's TorchScript modules reject the vectorized batched
backward on CUDA) and differentiates the model with the **fixed neighbor
topology of the base structure** — the jacobian is exact up to cutoff-induced
connectivity changes. A real-checkpoint smoke test
(`tests/test_graph2mat_autograd_derivatives.py`, marked `slow`; set
`G2M_AUTOGRAD_SMOKE_CKPT` to point at a checkpoint) validates autograd against
a finite difference of the model itself.

## UI section

Open the pipeline UI and select the **ML vs SIESTA** sidebar tab. It has five
panels: SIESTA 5×5×1 generation, Matrix Viewer (heatmap + MAE/RMSE/max),
derivatives (reusing the Matrix Viewer), datasets small/large, and species
transfer. The Matrix Viewer "Cargar demo" button renders a synthetic payload so
the heatmap works before real matrices are wired.

## What this does NOT do (yet)

- It does **not** train Graph2Mat or DeepH.
- It does **not** launch SIESTA (you run SIESTA yourself on the generated FDFs).
- It does **not** ship heavy HSX/TSHS parsing here — those live in the existing
  derivative scripts; the loader here uses the lightweight `.npy` fixture format
  and names any missing file clearly.
- Real model inference must be supplied via a predictor `predict_fn`.
