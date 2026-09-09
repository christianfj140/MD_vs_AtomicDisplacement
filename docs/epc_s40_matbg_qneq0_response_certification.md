# E-F_001-S40 — GO-8b certification of the scalable MATBG `q != 0` response

Fase 10aB of the roadmap. S33 (`docs/epc_s33_ruta_qneq0_produccion.md`) picked route Q-C as the
primary `q != 0` production route for MATBG (Q-B kept as a per-`q` numerical cross-check) and
pre-registered eight things `additional_tests_required_before_go8b` had to demonstrate first —
all still open at S33, since that ticket ran Q-B/Q-C only on a pure-numpy toy chain, never
through the real Graph2Mat production JVP. This ticket builds the real implementation and runs
that exact list.

Producer: [`Comparison/scripts/epc_qb_qc_graph2mat_kernel.py`](../Comparison/scripts/epc_qb_qc_graph2mat_kernel.py)
(the real-JVP Q-C/Q-B implementation) and
[`Comparison/scripts/certify_qneq0_matbg_response.py`](../Comparison/scripts/certify_qneq0_matbg_response.py)
(`certify_qneq0_matbg_response_go8b()`, `require_go8b_certified_qneq0_response()`).
Tests: [`tests/test_epc_qb_qc_graph2mat_kernel.py`](../tests/test_epc_qb_qc_graph2mat_kernel.py),
[`tests/test_certify_qneq0_matbg_response.py`](../tests/test_certify_qneq0_matbg_response.py).
Depends on: [[epc_s33_ruta_qneq0_produccion]], [[epc_s32_qb_qc_prototypes]],
[[epc_s31_graphene_k_go5_verdict]], `graph2mat_autograd_derivatives.compute_graph2mat_directional_derivative`
(C10/C11), `fd_perturbation_space.topology_margin`, `epc_fourier` (S29), `shared/artifact_signature.py`.

Evidence conventions: **[CODE]** verified in this repository, **[MEAS]** measured here,
**[DESIGN]** decision frozen in this document.

---

## 1. Verdict

**NO-GO-8b**, and honestly so: `certify_qneq0_matbg_response_go8b()["go8b_status"]` is
`"NO-GO-8b"`. This is not a failure of this ticket's work — it is the same inherited-blocker
pattern S38/S39 (GO-8a) already established: everything that *can* run today runs and passes;
the one piece that cannot (agreement against real graphene K) stays explicitly `NOT_RUN`,
blocked by GO-5 (`NO_GO`, itself blocked by GO-4/C14C's unresolved intra-atomic basis-response
term), not silently skipped or assumed.

```
$ .venv/bin/python Comparison/scripts/certify_qneq0_matbg_response.py
S40 MATBG q!=0 response GO-8b certification: NO-GO-8b
  route_mechanics_suite: PASS
    [ok  ] qc_and_qb_agree_on_real_graph2mat: worst_abs_diff=1.11e-16
    [ok  ] q_and_minus_q: worst_abs_diff=0.0
    [ok  ] cell_origin_shift: worst_abs_diff=0.0
    [ok  ] atom_across_periodic_boundary: worst_abs_diff=2.29e-16
    [ok  ] alternative_atomic_phase_convention: worst_abs_diff=1.11e-16
    [ok  ] topology_margin_preserved: min_margin_ang=0.1793 at cutoff 3.75, 0 crossing pair(s)
    [ok  ] gpu_preflight_recorded: requested='cuda' effective='cuda' preflight='passed'
    [ok  ] sparse_serialization_round_trip: nnz=10 shape=[10, 10] reproducible=True
    [ok  ] kernel_table_cache_restart: reload status='valid', diff after restart=0.0e+00
    [ok  ] jvp_calls_independent_of_shells: True; real MATBG edge count measured: False
    [ok  ] gamma_jvp_reuse_for_qneq0_rejected: raised GammaJvpReuseError as required
  graphene_k_agreement_suite: NOT_RUN
    [n/a ] route_agrees_with_q_a_on_real_graphene_k: blocked by NO_GO: GO-4 ...
```

## 2. Why the toy still isn't graphene K, and why that's the right scope for this ticket

**[CODE]** S32's toy chain (`epc_qb_qc_prototypes.py`) proved Q-B/Q-C's *algebra* — the
real-space-to-`(k,q)` identity, the `q`-independence of Q-C's table, the cost trade-off — using
pure numpy finite differences. It never touched
`graph2mat_autograd_derivatives.compute_graph2mat_directional_derivative`, so none of S33's eight
"additional tests" could be exercised: there was no real JVP to run a topology-margin check on,
no real sparse mapping to round-trip through, no real `resolve_jvp_backend` preflight to extend.

**[DESIGN]** This ticket closes that gap the same way S32 chose its own toy: on a small periodic
model (`PeriodicShellGraphModel` / `PeriodicShellReplicatedGraphModel`, a 1D two-atom chain,
literally the same physical constants as `epc_qb_qc_prototypes` — same `T0_EV`, `S0`,
`LAMBDA_ANG`, `CUTOFF_RADIUS_ANG`), but differentiated through the *real* production JVP instead
of hand-rolled finite differences. Running graphene K for real would only re-report the same
open GO-4/C14C gate a third time (S32 already made this exact call for its own toy, S31 for
`epc_commensurate_k`) — not exercise anything new about the mechanics S33 asked this ticket to
prove.

## 3. Why two independent position "copies" are needed at all

**[CODE]** A real production Graph2Mat/MACE graph has exactly one position leaf per base-cell
atom; `edge_index`/`shifts` route each edge to the periodic image it needs, but moving
`positions[i]` moves *every* image of atom `i` identically. That is why the existing plain
directional JVP (`compute_graph2mat_directional_derivative`) is Gamma-periodic (`q=0`) by
construction (roadmap Section II, blocker B4): it cannot tell "the home copy of atom `i`" apart
from "the image-`l` copy of atom `i`" for a self-pair edge (`mu == nu`, `shell != 0`, e.g. an
atom's coupling to its own periodic image), because both roles read the same tensor entry.

`epc_qb_qc_prototypes.pair_block` sidesteps this on the numpy toy by calling itself on two
*independent* explicit position arrays (a home array and a neighbour array) per shell — there is
no shared tensor to conflate. `PeriodicShellGraphModel` ports that same trick onto the real
production JVP by expanding the batch's position tensor into independent "home" and
"neighbour-image" leaves (one neighbour block, shared across shells — sufficient for Q-C, whose
neighbour tangent carries no `q`); `PeriodicShellReplicatedGraphModel` uses one *independent*
neighbour block *per shell* (needed for Q-B, whose neighbour tangent is weighted by
`exp(i q . R_shell)` and must therefore differ shell to shell). Both are differentiated through
the same, unmodified `compute_graph2mat_directional_derivative` — nothing in this module
reimplements autograd. `tests/test_epc_qb_qc_graph2mat_kernel.py::test_periodic_shell_graph_model_has_self_pair_edges_at_nonzero_shells`
confirms the model actually exercises this case, not just the easy `mu != nu` one.

**[MEAS]** The payoff: Q-C's whole `q`-independent kernel table needs exactly **2** real JVP
calls, independent of `shells` (against `epc_qb_qc_prototypes.kernel_table`'s
`4 * (2 * shells + 1)` finite differences); Q-B needs exactly **2** real JVP calls *per `q`*
(against the toy's per-shell Python loop for the same `q`). Nothing of shape
`[n_outputs, n_atoms, 3]` is ever materialised.
`tests/test_epc_qb_qc_graph2mat_kernel.py::test_kernel_table_matches_the_toy_prototype` confirms
the real-JVP numbers agree with the hand-rolled toy to FD noise (`< 5e-9`).

## 4. S33's eight items, one by one

| # | requirement | status | evidence |
| --- | --- | --- | --- |
| 1 | Q-C (+ Q-B cross-check) on the real Graph2Mat JVP | **done** | `kernel_table_graph2mat`/`phase_aware_kernel_graph2mat`, both call `compute_graph2mat_directional_derivative` unmodified |
| 2 | topology-margin test | **done** | `topology_margin_report()` reuses `fd_perturbation_space.topology_margin` directly; `min_margin_ang=0.179` at the physical `delta` used, `0` crossing pairs |
| 3 | the five S31-style checks on the real kernel | **done** | `real_kernel_checks()`: `qc_and_qb_agree_on_real_graph2mat`, `q_and_minus_q`, `cell_origin_shift`, `atom_across_periodic_boundary`, `alternative_atomic_phase_convention` — all PASS at `< 3e-16` (machine precision; the real JVP is exact, no FD noise floor) |
| 4 | agreement vs Q-A on real graphene K | **blocked** | `NOT_RUN`, `epc_commensurate_k.go5_decision()` carried verbatim into `graphene_k_agreement_suite` |
| 5 | `raw_derivative_kernel_table` cache artifact + restart test | **done** | new `EPC_ARTIFACT_KINDS["raw_derivative_kernel_table"]` node (Section 5 below); `save_kernel_table_cache`/`load_kernel_table_cache` reuse the *existing* `artifact_signature.cached_result_status` mechanism (not a new cache manager); restart test confirms `0.0` diff after reload and that a changed direction is correctly rejected as `signature_mismatch` |
| 6 | GPU preflight on the real kernel build | **done** | `gpu_preflight_report()` calls `resolve_jvp_backend` (the existing CUDA-preflight function) directly on the real periodic model/batch — extends it, does not repeat it; on this host, `requested=cuda effective=cuda preflight=passed` |
| 7 | `scaling_sweep` at the real MATBG shell/edge count | **partial, honestly labelled** | see Section 6 — no real MATBG edge count has ever been measured (S37 OOM'd first); `scaling_sweep_graph2mat` uses `shells` as an explicit proxy and says so in its own output (`matbg_edge_count_measured: False`) |
| 8 | sparse serialization round-trip through C11 | **done** | `sparse_round_trip_check()`: two calls through `derivative_prediction_to_sparse_matrices` (via `compute_graph2mat_directional_derivative(..., data_processor=...)`) produce byte-identical sparse matrices |

## 5. The `raw_derivative_kernel_table` artifact kind

**[CODE]** S33 Section 2 flagged a signature gap: `raw_derivative`'s existing fields cover the
final per-`(k, q)` result of either route, but had no separate node for Q-C's `q`-free table,
so a process restart would redo the table build even though it is `q`-independent. This ticket
adds `raw_derivative_kernel_table` to `shared/artifact_signature.py`'s `EPC_ARTIFACT_KINDS`:

```
fields: derivative_backend, derivative_method, direction_hash, topology_sha256,
        delta_or_jvp, shells, dtype, units
deps:   hamiltonian (electronic_hamiltonian)
```

Deliberately no `q` or `mode` field — that independence from `q` is the entire point (S33
Section 2). `raw_derivative` gained an *optional* `kernel_table` dependency
(`("raw_derivative_kernel_table",)`) so a per-`(k, q)` result built by contracting a table can
declare which table it consumed. `tests/test_artifact_signature.py` was extended (`build_dag()`
now builds one, wired as `raw_derivative`'s `kernel_table` dep) and all 25 existing DAG tests
still pass unmodified otherwise — geometry invalidation, broadening isolation, occupation
isolation all still hold with the new node in the graph.

## 6. Item 7's honest limit: no real MATBG edge count exists yet

**[CODE]** `docs/epc_s37_matbg_synthetic_benchmark.md` already measured that the naive
(unchunked) directional-JVP forward on the real `(31, 30)` checkpoint drove this host's RSS past
30+ GiB and was killed by a safety watchdog *before* any structure-specific edge count was
recorded. There is therefore no real MATBG shell/edge count to scale `scaling_sweep_graph2mat`
to — inventing one would be exactly the kind of fabricated number this repository's own
conventions forbid.

**[MEAS]** `scaling_sweep_graph2mat` instead sweeps `shells` as an explicit, labelled proxy
(`(1, 4, 16, 64)`) and reports two things plainly: `jvp_calls_independent_of_shells: True` (the
load-bearing result — real JVP cost does not scale with neighbour-shell count, unlike the toy's
FD cost) and `matbg_edge_count_measured: False` with `matbg_edge_count_blocking_reason` pointing
at S37. A further honesty check: this toy's fixed physical cutoff caps `edges_in_topology` at
the same value (`10`) for every `shells` tried above the cutoff's own shell radius, so
`edge_count_grows_with_shells` is `False` too — the sweep is evidence that *JVP calls* don't
scale with `shells`, not evidence about how a *real* MATBG edge count would scale. Re-measuring
against a real `(31, 30)` topology remains open, blocked on the same resource issue S37 recorded
(chunked forward / more host RAM), not on anything this ticket could resolve.

## 7. The Gamma-JVP-reuse guard (GO-8b's own acceptance criterion)

**[CODE]** GO-8b explicitly requires "ninguna reutilización del JVP Gamma como sustituto
silencioso de una perturbación `q != 0`" and S33 published `rejects_gamma_jvp_reuse: True` as a
decision, not yet as running code. This ticket makes it a structural, automatic failure:
`k_to_k_plus_q_kernel_graph2mat` and `q_b_response_graph2mat` both check their input payload's
`"schema"` field before doing anything else, and raise `GammaJvpReuseError` for any `q != 0`
query against a payload that is not actually a `kernel_table_graph2mat`/
`phase_aware_kernel_graph2mat` output — in particular, against a plain
`compute_graph2mat_directional_derivative` result (`DIRECTIONAL_DERIVATIVE_SCHEMA`, a genuine
q=0 Gamma-periodic JVP of one cell). `tests/test_epc_qb_qc_graph2mat_kernel.py::test_k_to_k_plus_q_kernel_rejects_a_gamma_jvp_payload_for_qneq0`
and `::test_q_b_response_rejects_a_gamma_jvp_payload` exercise this directly; the certifier's own
`gamma_jvp_reuse_for_qneq0_rejected` check re-runs it as part of `route_mechanics_suite`, so a
future regression that silently widened the guard's schema check would fail CI, not just a
docstring promise. At `q == 0` the guard does not fire — that is a legitimate degenerate case,
not the failure mode GO-8b is about.

## 8. What this proves and what it does not

- **Proves:** the chosen production route (Q-C, plus Q-B as cross-check) is implemented against
  the real Graph2Mat production JVP, not just the S32 toy; its topology margin, cache/restart,
  GPU preflight and sparse serialisation are all real, measured, and passing; the five S31-style
  invariance checks hold at machine precision on the real kernel; a Gamma JVP fed into a `q != 0`
  contraction fails automatically rather than silently.
- **Does not prove:** agreement with Q-A on real graphene K (blocked by GO-4/GO-5, item 4, and
  by design not attempted here), nor a real measured MATBG edge/shell count (blocked by S37's
  resource limit, item 7, partial). `go8b_status` is `NO-GO-8b` and stays that way until GO-5
  clears; `matbg_qneq0_production_authorized` is `False`.

## 9. Reproduce

```bash
.venv/bin/python Comparison/scripts/epc_qb_qc_graph2mat_kernel.py
.venv/bin/python Comparison/scripts/certify_qneq0_matbg_response.py
.venv/bin/pytest -q tests/test_epc_qb_qc_graph2mat_kernel.py tests/test_certify_qneq0_matbg_response.py tests/test_artifact_signature.py
```
