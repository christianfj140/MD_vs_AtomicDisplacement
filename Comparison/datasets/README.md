# Comparison Datasets

This directory is the persistent local home for reusable benchmark datasets.
It is intentionally separate from:

- `Comparison/workspaces/`, which is temporary per-run scratch space.
- `Comparison/results/`, which stores model outputs, metrics and reports.

Large generated SIESTA/Graph2Mat/DeepH artifacts are ignored by git. Keep only
small documentation or manifest templates under version control.

The default Graph2Mat vs DeepH dataset root is:

```text
Comparison/datasets/graphene_w90_joint/
```

That dataset is benchmark-ready only after it contains the joint artifact
contract files, including `artifact_validation.json`,
`benchmark_dataset_manifest.json` and `frozen_split_manifest.json`.
