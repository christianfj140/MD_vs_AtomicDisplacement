# ML vs SIESTA benchmark

Infrastructure for a benchmark flow that compares Graph2Mat / DeepH matrix
predictions (and their finite-difference derivatives) against SIESTA references.

> **This toolkit is infrastructure only.** It never launches SIESTA, never trains
> a model and never needs a GPU. It builds inputs, validates plans, computes
> errors and serves UI payloads. Heavy model inference and real SIESTA parsing
> stay in the existing `Comparison/scripts` runners and are plugged in through
> thin adapters.

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
