# Known Limitations

## Verified Limitations

- The workflows depend on external executables such as `siesta`, `graph2mat`,
  and, for the joint benchmark, the DeepH command set.
- The repository does not ship a centralized packaging or linting config at the
  root level.
- Several workflows are only safe when the expected local `.venv/` exists.
- The comparison workflow uses the canonical methods `md`,
  `siesta_fc_cartesian`, and `random_cartesian`; legacy aliases are supported,
  but they are still legacy aliases.
- `ML_prediction.HSX` is not automatically safe to treat as a standalone
  Hamiltonian+overlap reference. The benchmark code and docs explicitly warn
  that spectral metrics may need the SIESTA reference overlap instead.
- Several user-facing flows are file-driven and manifest-driven, so moving or
  partially editing archived run directories can invalidate reuse.

## Temporal Leakage In Archived Snapshot-Scaling Datasets (Audit C2)

The archived `*_iid*` snapshot-scaling datasets (e.g.
`graphene_w90_scale_iid*`, `graphene_5x5_scale_iid*`) were generated from MD
temperature blocks (150 K / 300 K / 450 K trajectories of consecutive frames,
1 fs apart) and split with `blocked_with_gap` using `temporal_gap: 1`. This
has three consequences that apply to every number already produced from these
datasets. The user decision is to document them, not to regenerate the
datasets (regeneration would require re-running SIESTA):

1. **The reported test MAE is 450 K extrapolation, not in-distribution
   error.** In these datasets the train split contains the 150 K and 300 K
   trajectories only, while validation and test are both carved from the
   450 K trajectory. Models are therefore evaluated on a temperature regime
   they never saw in training.
2. **Validation is a temporal twin of test.** Validation and test are the
   early and late halves of the *same* 450 K trajectory, separated by a
   single 1 fs frame. Carbon vibrational periods are ~20-40 fs, so a 1-frame
   gap does not decorrelate the blocks: checkpoint selection / early stopping
   is informed by configurations nearly identical to the test set. The
   locked-test policy holds formally but is materially eroded.
3. **`*_iid*` is a misnomer.** The samples are consecutive MD frames, not
   independent draws. Treat any "MAE vs dataset size" trend from these
   datasets as interpolation along nearly continuous trajectories.

Any already-published or archived result computed on these splits needs this
note attached. New datasets are generated with `temporal_gap: 30` by default
(~one carbon vibrational period at 1 fs/frame); the old datasets keep their
frozen splits for reproducibility of past runs.

## Scientific And Computational Caveats

- The repository exposes DeepH-comparable diagnostics, but it does not claim to
  reproduce every DeepH paper feature or every DeepH metric exactly.
- The final/publicable Graph2Mat-vs-DeepH workflow is intentionally
  fail-closed. Missing dataset evidence, missing artifact hashes, or ambiguous
  provenance should block robust claims rather than be guessed around.
- Some outputs are marked diagnostic-only or exploratory when provenance is not
  strong enough for a robust benchmark claim.
- Cross-material plots and mixed-provenance comparisons are useful for
  inspection, but they are not a substitute for a compatibility-hash matched
  benchmark.
- The `ML vs SIESTA` toolkit is infrastructure and validation glue; it is not a
  full training or production inference runner.
- Dataset-size-minimum reports are postprocessed summaries over archived runs;
  they do not repair weak provenance or upgrade exploratory runs into
  publication-ready evidence.

## Fragile Areas

- File-based workspaces can be invalidated by stale outputs from older runs.
- Dataset reuse only works when the stored manifests, split files, and material
  provenance remain compatible.
- Several scripts rely on exact file names such as `RUN.fdf`, `RUN.out`,
  `artifact_validation.json`, and `frozen_split_manifest.json`.
- Generated results are only as trustworthy as the external SIESTA and DeepH
  runs that produced them.
- Some workflows assume POSIX-style command examples in docs even when the
  local operator is on Windows or another shell environment.

## Validation Gaps

- The repository has broad `unittest` coverage, but not every external-tool
  workflow can be exercised in a pure unit test.
- Live SIESTA, Graph2Mat, and DeepH execution is environment-dependent and is
  not fully verified by the lightweight checks.
- The documentation can confirm command names and file paths, but it cannot
  guarantee that the external executables are installed or that their versions
  match a particular publication baseline.

## Assumptions That Remain Uncertain

- The current environment may or may not have the required external binaries
  available.
- Existing archived results may include a mix of fresh and legacy runs, so the
  docs should be read as repository facts, not as a guarantee that all archived
  outputs were produced with the same toolchain.
- Runtime, throughput, and scientific comparability can vary substantially with
  the local machine and the installed external dependencies.
- Not every archived dataset necessarily satisfies the latest joint artifact,
  material-provenance, derivative, or H-only/S_ref expectations until it is
  explicitly revalidated or re-evaluated.

## Audit 2026-07-10: mixing and autograd validation contract

The small/large mixing and autograd-derivative corrections are documented in
`docs/mixing_and_autograd_validation_contract.md` (semantics, gates, claim
ladder) and `docs/audit_corrections_implementation_report.md` (what changed,
what was measured, what remains). Highlights of remaining limitations:

- Effective composition reports node blocks and matrix elements but not edge
  blocks (neighbour lists would be required).
- Derivative metrics expose micro/macro/per-domain reductions; absolute-H
  metrics keep their historical aggregation.
- DeepH autograd-vs-FD validation in float32 (production dtype) is only
  conclusive around delta = 1e-4 Ang (FD cancellation noise elsewhere).
- `paper_ready` claims additionally require `pinned_clean` repositories; any
  uncommitted change anywhere downgrades runs to `pinned_dirty`.
