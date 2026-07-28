# Phase 6 H2O Hamiltonian Architecture Benchmark

This benchmark compares the corrected H-only Graph2Mat baseline against
Hamiltonian-specific architecture options exposed by an editable Graph2Mat
checkout.

Related documents:

- `README.md` for the current repository scope and common validation commands.
- `docs/workflows.md` for the main comparison UI flow.
- `docs/graph2mat_deeph_benchmark.md` for the stricter joint benchmark rules.

## Payload

Use:

```text
Comparison/config/h2o_phase6_hamiltonian_architecture_benchmark_payload.json
```

The payload is intentionally `train_test_metrics_plots_only` and reuses the
archived MD dataset/splits through:

```text
reusable_dataset_ids: ["98c89bd85937f6a2"]
reusable_split_policy: preserve_archived_splits
selected_methods: ["md"]
```

Replace the dataset id only if you intentionally want a different archived
dataset. The benchmark should not regenerate references and must not use
`ML_prediction.HSX` as ground truth.

## Matrix

The current payload has three seeds for each candidate:

- `baseline_default_mae`
- `baseline_default_huber_b0p01`
- `baseline_default_mse`
- `context_default_huber_b0p01`
- `context_hamiltonian_readout_huber_b0p01`
- `context_hamiltonian_readout_staged_composite`
- `diagnostic_dense_mse`

The dense readout entry is diagnostic only and must not be ranked as a
production result.

Three seeds are insufficient for a paper-level architecture claim under the
current gate. Before the final campaign, predeclare at least five independent
seeds or archive a power justification.

## Scientific Guardrails

Every plan entry sets:

```yaml
out_matrix: hamiltonian
matrix_component_policy: h_only
n_matrix_components: 1
symmetric_matrix: true
```

The archive/evaluator path uses the strict reference selector. It accepts real
SIESTA `.TSHS` / `.HSX` references, rejects `ML_prediction.HSX`, and records the
reference policy in each manifest.

Spectral and DOS metrics must be interpreted only when the metrics manifest
records the post-H-only/S_ref provenance, including target policy, component
counts, overlap source, and prediction-HSX safety fields. Phase-6 rows lacking
that provenance are legacy or unknown and should be regenerated with
`Comparison/scripts/evaluate_hamiltonian_metrics.py` before they are used for a
winner claim.

The matrix metrics in this benchmark are repository raw-global-H diagnostics.
They are useful for internal Graph2Mat comparisons, but they are not exact
DeepH H-prime local-frame block metrics unless a future validated H-prime
transform is added.

## Manifests

Per-run manifests now record benchmark metadata:

- pipeline and Graph2Mat git commits
- architecture/readout/context fields
- loss and loss kwargs
- staged training metadata
- seed
- H-only target policy
- reference policy and evaluation manifest path

## Running

Load the JSON payload in the UI or POST it to the existing experiment endpoint.
Run a one-epoch smoke first by copying the payload and changing every
`max_epochs` to `1`.

Full benchmark cost is high: 21 trainings on the MD 1140-snapshot dataset.
