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

## Fragile Areas

- File-based workspaces can be invalidated by stale outputs from older runs.
- Dataset reuse only works when the stored manifests, split files, and material
  provenance remain compatible.
- Several scripts rely on exact file names such as `RUN.fdf`, `RUN.out`,
  `artifact_validation.json`, and `frozen_split_manifest.json`.
- Generated results are only as trustworthy as the external SIESTA and DeepH
  runs that produced them.

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
