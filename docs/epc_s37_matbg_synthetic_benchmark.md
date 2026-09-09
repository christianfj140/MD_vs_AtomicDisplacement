# D06 / E-F_001-S37 — MATBG synthetic-displacement resource benchmark

Fase 10b of the roadmap: measure, on the rigid `(31, 30)` magic-angle cell,
the wall time / RSS / VRAM / sparse NNZ / disk cost of the Graph2Mat
directional JVP, its sparse serialisation, the "contract before materialise"
VJP alternative, and one persisted electronic eigenspace — using synthetic,
explicitly non-phonon displacements. This is a computational cost model, not
a PAO-covariant coupling calculation: GO-1/GO-4 (moving-PAO formalism) and GO-8a (a
validated MATBG `PhononProvider`) are still open, so nothing here is
contracted into a Hamiltonian-electron coupling.

Producer: [`Comparison/scripts/benchmark_matbg_synthetic_displacements.py`](../Comparison/scripts/benchmark_matbg_synthetic_displacements.py).
New primitives it depends on:
[`fd_perturbation_space.synthetic_benchmark_direction_set`](../Comparison/scripts/fd_perturbation_space.py)
(breathing/shear/optical/random) and
[`graph2mat_autograd_derivatives.compute_graph2mat_vjp_contraction`](../Comparison/scripts/graph2mat_autograd_derivatives.py)
(the contract-before-materialize primitive, sibling to the existing
`compute_graph2mat_directional_derivative` JVP). Tests:
[`tests/test_benchmark_matbg_synthetic_displacements.py`](../tests/test_benchmark_matbg_synthetic_displacements.py),
plus additions to `tests/test_fd_perturbation_space.py` and
`tests/test_graph2mat_autograd_derivatives.py`.

Evidence conventions: **[CODE]** verified in this repository, **[MEAS]**
measured here, **[DESIGN]** decision frozen in this document, **[OPEN]**
question left for follow-up.

---

## 1. Verdict

**Cost model partially measured; GO-8 is NOT yet PASS for the naive
unchunked JVP path on `(31, 30)`.**

- **[MEAS]** The per-unit machinery (directional JVP, VJP contraction, the
  materialize-once/contract-before-materialize crossover, and the real sparse
  solver's persisted-eigenspace stage) is validated end to end against real
  Graph2Mat/solver artifacts — just not yet at `(31, 30)` scale for the JVP
  half, for a resource reason recorded below, not a code defect.
- **[MEAS]** The persisted-eigenspace stage *does* run successfully at full
  `(31, 30)` scale (44656 orbitals): one gamma-point window solved and
  persisted in 19 s wall / 3.85 GiB RSS via the existing `gpu_cudss` Julia
  backend, comfortably inside the 28 GiB VRAM / 62 GB RAM roadmap margins.
- **[MEAS]** The naive (no MACE node/edge chunking) directional-JVP path —
  checkpoint load + one `resolve_jvp_backend` CUDA preflight, before any
  synthetic direction is even measured — drove this machine's RSS past 30+ GiB
  within under a minute and was heading for a full-system OOM; a safety
  watchdog killed it at ~5 GiB available rather than let the kernel OOM-killer
  pick a victim on this shared host. This is real, not inferred: see §3.

## 2. What the four direction families are

**[CODE]** `fd_perturbation_space.synthetic_benchmark_direction_set` builds,
from the structure's own cartesian positions:

| direction | construction | probes |
| --- | --- | --- |
| `breathing_like` | halves split by median `z`, `+/-z` | interlayer-normal response |
| `shear_like_x` | halves split by median `z`, antiparallel in-plane | interlayer in-plane response |
| `optical_like` | even/odd atom index, antiparallel along a random axis | alternating-atom response |
| `random_seed{N}` | `fd_perturbation_space.random_collective` | arbitrary-`v` control |

All four are `kind = "collective"` `Direction` objects (unit-Frobenius, same
`fd_perturbation_space` machinery graphene-Γ/K already use) — no mass
weighting, no frequency, no phonon provider. The benchmark script tags every
artifact it writes `displacement_class = "synthetic_test_displacement"` and
runs `assert_no_forbidden_labels` (refusing `phonon_eigenmode`/`physical_g`/
`g_mn_nu` anywhere in a payload) on every JSON it writes — this is exercised
by `LabelGuardTests` in the test file, not just asserted in prose.

## 3. Real run against `(31, 30)`: what happened

**[MEAS]** Checkpoint: `Comparison/results/tbg_registry_spectral_loss/training/checkpoints/spectral-best-epoch=247-step=03472.ckpt`
(the same one `run_tbg_pure_graph2mat_campaign.py` uses in production).
Structure: `materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf` (11164
atoms, 44656 orbitals). `--accelerator cuda` (RTX 5090, 32 GB, free at the
time of the attempt).

Two attempts, both instrumented with a 3 s `ps`/`\/proc/meminfo` poll:

1. First attempt: no external monitor; the process was silently reaped with
   no Python traceback in `stderr` (consistent with `SIGKILL`, which discards
   unflushed buffers) while system swap was at 7.8/8.0 GiB used.
2. Second attempt, with the poller and a watchdog that sends `SIGTERM`/
   `SIGKILL` once `/proc/meminfo MemAvailable` drops under ~6 GiB (to protect
   the other processes on this shared host — this machine also runs the
   user's IDE and other agents): RSS climbed non-monotonically (GC between
   samples) from near 0 to 34 GiB in under 50 seconds, entirely *inside*
   `load_checkpoint_and_batch` + the `resolve_jvp_backend` CUDA preflight —
   i.e. before any of the four synthetic directions had even started. The
   watchdog fired at `avail_kib=4927696` (~4.7 GiB) and killed it cleanly;
   memory recovered to 51 GiB free immediately after.

**[DESIGN]** Root-caused, not guessed: `run_graph2mat_autograd_derivative_predictions.py`
(the existing vectorized-VJP jacobian route) already documents that the full
`(31, 30)` forward needs MACE node/edge chunking + `bf16-mixed` precision
(`--mace-node-chunk-size 512 --mace-edge-chunk-size 8192`, used by
`run_tbg_pure_graph2mat_campaign.py`'s own H-prediction stage) to fit in
memory/VRAM; this benchmark's loader deliberately did **not** apply that
chunking (module docstring: "no chunking: the directional JVP is one
double-backward, not the batched-VJP jacobian route that needed it"). That
assumption held for every checkpoint/structure pair this session validated
against (small smoke systems, §4) but does not hold for the full 44656-orbital
graph: MACE's message-passing forward pass itself — not the double-backward —
is the part that needs bounding at this scale, independent of which
downstream derivative primitive consumes it.

**[OPEN]** Re-attempt with MACE node/edge chunking applied to the forward
closure (mirroring `predict_model_on_dataset.apply_mace_node_chunking`/
`apply_mace_edge_chunking`), or on a host with materially more free RAM. This
is the concrete next step before GO-8 can be marked PASS for the JVP/VJP-
contraction half of the cost model at production scale.

## 4. Real validation of the measurement machinery itself

**[MEAS]** Because the full-scale JVP attempt could not complete safely, the
JVP/VJP-contraction/crossover/caching machinery was instead validated for
real against a small, already-checkpoint-gated Graph2Mat model (the same
`graphene_w90_scale_iid12` smoke checkpoint `tests/test_graph2mat_autograd_derivatives.py`
and `tests/test_validate_graph2mat_jvp.py` already use, 2 atoms / 8 orbitals):

- CUDA preflight: `requested_backend=cuda`, `effective_backend=cuda`,
  `backend_preflight=passed` — the real double backward ran on GPU.
- All four synthetic directions: `materialize_once` (JVP + sparse
  serialisation) between 0.02-0.22 s; `contract_before_materialize` (VJP
  contraction) between 0.013-0.056 s; both correctly returned the same
  `nnz=992`/`matrix_shape=[8, 200]` sparse layout.
- Crossover table: for this tiny system, `contract_before_materialize` wins
  at `n_blocks=1` and `materialize_once` wins from `n_blocks≈3` up (the
  formula's own `crossover_n_blocks≈2.98` matches the grid's own switch) —
  reproducing the roadmap's own qualitative expectation ("si se usan muchos
  k... A probablemente gana") from measured numbers, not assumed ones.
- Cache: re-running the identical command produced `cache.hits = 4/4`
  (`input_signature_sha256` match), i.e. the per-direction restart/reuse
  granularity works.
- `resource_margins`: `peak_rss_gib≈1.8`, `peak_vram_gib≈0.017`, both well
  inside the 62 GB / 28 GiB roadmap margins (expected, at this tiny scale).

**[MEAS]** The eigenspace stage was then run against the *real* `(31, 30)`
solver input (`Comparison/results/tbg_pure_graph2mat/prediction/solver_input`,
the actual production H/S at 44656 orbitals) with a cheap gamma-only, 4-band,
`--persist-eigenspaces` probe: `status=completed`, `s_orthonormal=True`,
`maximum_generalized_relative_residual≈3.5e-10`, `wall_clock=0:19.25`,
`max_rss_gib≈3.85`, cuDSS `analysis 0.42 s / factorization 0.83 s / solve
0.18 s (36 solves)`. This *is* full-scale, real evidence for the
eigenspace-cost half of GO-8.

## 5. A bug this benchmark's probe surfaced and fixed

**[CODE]** `run_deeph_sparse_spectrum.run()` read `egvals.dat` (one row per
band, one column per k point) via `np.loadtxt(...).T`. For any DOS job that
solves exactly one k point, the file has a single column, so `np.loadtxt`
collapses it to a 1-D array; `.T` on a 1-D array is a no-op, so
`len(raw_energies)` silently read as the *band* count instead of `1`. This
never surfaced before because production DOS runs always used multi-k meshes
(`8x8`, `16x16`) and the one existing single-k-point run
(`fermi_inertia_preflight`) never combined it with `--persist-eigenspaces`.
It is exactly the "cheap gamma-only probe + persist-eigenspaces" combination
this benchmark introduces that hit it — `persisted_eigenspace_data` then
looked for `eigenspace_001.json`, which was never written, and raised
`FileNotFoundError`.

Fixed by reshaping the loaded array to a column vector before transposing
when it is 1-D (`Comparison/scripts/run_deeph_sparse_spectrum.py`, the
`elif returncode == 0:` DOS branch). Regression coverage:
`tests/test_deeph_sparse_spectrum.py` continues to pass unchanged (15/15);
the fix itself is exercised for real by §4's eigenspace run, which failed
before the fix and passed after it, on the same real `(31, 30)` input.

## 6. GO-8 status

| component | status |
| --- | --- |
| synthetic direction generators (breathing/shear/optical/random) | **[MEAS]** implemented, tested, real-run validated |
| materialize-once JVP + sparse serialisation | **[MEAS]** real-run validated (small system); **[OPEN]** at `(31, 30)` scale, needs MACE chunking |
| contract-before-materialize VJP | **[MEAS]** real-run validated (small system); same `(31, 30)`-scale open item |
| materialize-once vs. contract-before-materialize crossover | **[MEAS]** measured and internally consistent |
| persisted electronic eigenspace | **[MEAS]** real-run validated *at full `(31, 30)` scale* |
| cache/restart granularity (`per_direction`, `per_solver_call`) | **[MEAS]** verified (4/4 cache hits on rerun) |
| CUDA preflight, backend provenance | **[MEAS]** real, non-silent (`resolve_jvp_backend`) |
| RAM/VRAM/disk margin accounting | **[MEAS]** implemented against measured peaks, not just "available" memory |
| labeling guard (never `phonon_eigenmode`/PAO-projected coupling) | **[MEAS]** enforced and tested |

**GO-8 = NOT YET PASS.** The blocking item is narrow and identified: apply
MACE node/edge chunking (or more host RAM) to the directional-JVP forward
closure before it is safe to run unattended against the full `(31, 30)`
structure on hardware at this RAM tier. Nothing else in the cost model is
blocked; the eigenspace half is already measured at full scale.
