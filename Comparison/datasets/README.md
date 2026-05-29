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

For paper-ready Graph2Mat-vs-DeepH claims, this directory is not sufficient by
itself unless the external artifact bundle is also released. Use:

```bash
python3 Comparison/scripts/g2m_deeph_verify_protocol_datasets.py \
  --protocol Comparison/config/g2m_deeph_paper_protocol_v1_example.json \
  --output Comparison/results/g2m_deeph_dataset_verification.json \
  --strict

python3 Comparison/scripts/g2m_deeph_release_manifest.py \
  --dataset-root Comparison/datasets/<frozen-dataset-id> \
  --output Comparison/results/g2m_deeph_dataset_release_manifest.json \
  --strict
```

The release manifest records sizes and SHA-256 hashes for external files that
are intentionally ignored by git. Missing or unhashable required evidence keeps
the benchmark diagnostic-only.
