# Derivative Smoke Validation Note

- Repo root: `/home/christian/repositorios/MD_vs_AtomicDisplacement`
- Graph2Mat smoke root: `Comparison/results/derivative_smoke/graph2mat_derivative_result`
- DeepH smoke root: `Comparison/results/derivative_smoke/deeph_derivative_result`

This note records a completed local derivative smoke for Graph2Mat and a completed local DeepH derivative smoke using the same 5-sample stencil/reference set. It validates workflow plumbing and UI/backend integration only. It does not claim paper-ready derivative results, and it does not claim any scientific winner.

## Smoke Scope

- Stencil samples: `5`
- Sample ids:
  - `md_270_base`
  - `md_270_atom0000_x_d0.005_plus`
  - `md_270_atom0000_x_d0.005_minus`
  - `md_270_atom0000_x_d0.01_plus`
  - `md_270_atom0000_x_d0.01_minus`
- SIESTA reference status: `5/5 ok`

## Graph2Mat Smoke

- Checkpoint path: `Comparison/results/derivative_smoke/graph2mat_derivative_result/graph2mat/training/lightning_logs/my_first_model/version_0/checkpoints/best-22.ckpt`
- Prediction status: `5/5 predicted`
- Local smoke outputs include:
  - `derivative_metrics/graph2mat/manifest.json`
  - `derivative_metrics/graph2mat/derivative_matrix_metrics.csv`
  - `derivative_metrics/graph2mat/derivative_delta_stability.json`
  - `derivative_metrics/graph2mat/summary/derivative_gate_report.json`
  - `derivative_metrics/graph2mat/summary/derivative_plot_payload.json`
  - `derivative_metrics/graph2mat/summary/derivative_plot_manifest.json`
  - `derivative_artifact_validation.json`

## DeepH Smoke

- Runtime note: sibling `DeepH-pack` source override via `PYTHONPATH=/home/christian/repositorios/DeepH-pack`
- Reused compatible DeepH model path: `Comparison/results/graphene_w90_snapshot_scaling_150_1000_dense/graphene_w90_snapshot_scaling_150_1000_dense_20260609_132720/sweep/deeph/graphene_w90_scale_iid150/DH-T1000-04-anchor-N150/deeph/train`
- DeepH prediction status: `5/5 predicted`
- DeepH derivative metrics path: `Comparison/results/derivative_smoke/deeph_derivative_result/derivative_metrics/deeph/derivative_matrix_metrics.csv`
- Local smoke outputs also include:
  - `Comparison/results/derivative_smoke/deeph_derivative_result/derivative_metrics/summary/derivative_gate_report.json`
  - `Comparison/results/derivative_smoke/deeph_derivative_result/derivative_metrics/summary/derivative_plot_payload.json`
  - `Comparison/results/derivative_smoke/deeph_derivative_result/derivative_metrics/summary/derivative_plot_manifest.json`
  - `Comparison/results/derivative_smoke/deeph_derivative_result/derivative_artifact_validation.json`

## Paired Comparison And UI

- Paired Graph2Mat-vs-DeepH comparison status: available
- Paired comparison count: `2`
- UI/backend run id: `deeph_derivative_result`
- UI endpoint verification completed for:
  - `/api/g2m-deeph/plot-runs`
  - `/api/g2m-deeph/derivative-metrics?run_id=deeph_derivative_result`
- Verified UI/backend sections:
  - derivative status summary
  - gate report
  - warning/blocker table
  - artifact links
  - derivative plots
  - paired comparison plot

## Gate Interpretation

- Gate status: `blocked`
- `blocked` is the expected outcome for this smoke. The workflow completed, produced the expected derivative diagnostics, and preserved conservative scientific gating instead of overclaiming.
- DeepH equivalence remains diagnostic-only/unproven against raw/global HSX ordering/sign/unit conventions.

## Release Framing

- This validates workflow plumbing and integration only, not paper-level derivative science.
- Do not treat this smoke as paper-ready derivative evidence.
- Do not treat this smoke as a scientific winner decision.
- The paths above refer to local smoke outputs from the validation run and are documented here as runtime evidence, not as committed release artifacts.
