# Graph2Mat Derivative Smoke Validation Note

- Repo root: `/home/christian/repositorios/MD_vs_AtomicDisplacement`
- Smoke root: `Comparison/results/derivative_smoke/graph2mat_derivative_result`

This smoke validates derivative-workflow plumbing for the Graph2Mat-only path. It does not validate DeepH derivative support, and it does not establish paper-ready derivative science.

## Smoke Contents

- Stencil samples: 5 total
- Sample ids:
  - `md_270_base`
  - `md_270_atom0000_x_d0.005_plus`
  - `md_270_atom0000_x_d0.005_minus`
  - `md_270_atom0000_x_d0.01_plus`
  - `md_270_atom0000_x_d0.01_minus`

## Stage Status

- SIESTA reference status: `5/5 ok` from `siesta_hamiltonians/derivative_siesta_reference_manifest.json`
- Graph2Mat checkpoint path: `Comparison/results/derivative_smoke/graph2mat_derivative_result/graph2mat/training/lightning_logs/my_first_model/version_0/checkpoints/best-22.ckpt`
- Prediction status: `5/5 predicted` from `predicted_hamiltonians/derivative_graph2mat_prediction_manifest.json`

## Produced Outputs

- Metrics manifest: `derivative_metrics/graph2mat/manifest.json`
- Matrix metrics: `derivative_metrics/graph2mat/derivative_matrix_metrics.csv`
- Delta stability: `derivative_metrics/graph2mat/derivative_delta_stability.json`
- Gate report: `derivative_metrics/graph2mat/summary/derivative_gate_report.json`
- Plot payload: `derivative_metrics/graph2mat/summary/derivative_plot_payload.json`
- Plot manifest: `derivative_metrics/graph2mat/summary/derivative_plot_manifest.json`
- Artifact validator: `derivative_artifact_validation.json`

## Gate Interpretation

- Gate `scientific_status`: `blocked`
- Blocked is expected for this smoke. The gate report correctly fail-closes on missing or incomplete derivative-comparability evidence, including:
  - missing or inconsistent orbital-ordering metadata
  - support-pattern discontinuity / false-zero-false-nonzero activity
  - missing paper-level convergence, gauge/order, reference-noise, and independent-dataset evidence

This is the expected outcome for a plumbing smoke: the workflow completed, emitted the expected derivative artifacts, and preserved conservative scientific gating instead of overclaiming.
