# E-F_001-S42 — rigid-MATBG Γ raw-derivative execution stage

Fase 10d of the roadmap ("Physical selected-mode campaign"), the first *execution* step
against the protocol S41 froze
(`Comparison/results/epc/preregistration/matbg/epc_matbg_rigid_gamma_protocol.json`). S41 itself
is explicit that `ready_to_execute = False` and names exactly one rung of its own claim ladder
this repository may still produce today: the two-path (`graph2mat_jvp` vs `graph2mat_frozen`)
raw-derivative validation, while GO-8a stays `NO-GO-8a` and C14C stays `NO_GO`. This ticket
implements exactly that rung, as a new stage of the existing campaign runner
(`Comparison/scripts/run_tbg_pure_graph2mat_campaign.py`), not a parallel script.

Producer: `run_tbg_pure_graph2mat_campaign.epc_gamma_rigid()`, wired behind `--epc-gamma-rigid`.
Tests: `tests/test_run_tbg_gamma_rigid_epc.py`.
Artifacts: `Comparison/results/epc/gamma_epc/matbg/{epc_gamma_rigid_summary.json,
artifact_status.json, arrays/*.npz}`.

Evidence tags: **[CODE]** verified in this repository, **[MEAS]** measured against the real,
already-frozen S41 protocol, **[DESIGN]** decision made in this ticket, **[OPEN]** open.

---

## 1. What this stage computes, and why not more

S41 §5/§6 froze two structural facts this stage inherits rather than re-derives:

- **no SIESTA reference exists at this scale.** `REFERENCE_PATHS` has two entries, not three:
  `graph2mat_jvp` and `graph2mat_frozen`, both of the *same* checkpoint. A disagreement between
  them is a backend bug, never a DFT-accuracy claim (S38: an exact `FC.Save.dHS` campaign on this
  cell needs `6 * 11164 = 66984` displaced SCF solves).
- **`PAO-covariant response` and `g` need a basis-response artifact** (`S_L`/`S_R`, per `FORMALISM_ID`'s
  `REPRESENTATION_S_L_S_R`). The only producer this repository has
  (`certify_basis_response.response_for_direction` / `epc_basis_response.py`) is built around a
  certified SIESTA FC-style campaign object (`campaign.certified_pairs()`,
  `campaign.dhsdr_path(delta)`) — the same one S38 measured intractable here, independent of
  which `electronic_derivative_backend` produced `D_H`. **[CODE]** This is a structural gap, not
  merely C14C's live `NO_GO` verdict: even if C14C closed tomorrow, MATBG would still have no
  basis-response producer until a cheaper SIESTA route or an analytic PAO-overlap derivative
  exists (roadmap Section XI, Case 2). `epc_gamma_rigid()` therefore never attempts to build
  `PAO-covariant response` or `g`; every artifact it writes records `basis_response`, `eigenspace`,
  `pao_covariant_response` and `g` as `status: "not_attempted"` with this reason, rather than a silently
  missing field.
- **`k_selection` is `pending_measurement`.** S41 froze the *rule* and the *candidate list*
  for `k_generic_1`/`k_generic_2`/`K`/`M`, not a measured spectrum — and
  `moire_hexagonal_convention_checked` is separately unverified for the same k labels. Persisting
  an eigenspace at any of them before that measurement exists would pre-empt it rather than reuse
  it, so this stage does not touch the eigensolver at all. **[DESIGN]**

What is left, and what this stage actually produces, is exactly S41's claim ladder's second rung:
`D_H` of the five pre-registered directions (`translation_x`, `random_seed0`, `breathing_like`,
`shear_like_x`, `optical_like`), by `graph2mat_jvp` and by `graph2mat_frozen`, agreement between
the two (`tau_backend`), and the honest bookkeeping of everything downstream that is still
blocked. Claim ceiling: `graph2mat_derivative_validated_no_full_ks_coupling`.

## 2. Gate: `epc_authorize`

Mirrors `run_graphene_gamma_epc_paths.authorize()`: verifies the protocol's `frozen_content_sha256`
is intact, then checks every entry of the frozen `blocking` list starts with one of four
pre-authorised prefixes (`GO-8a MATBG PhononProvider`, `C14C basis response (global formalism
gate)`, `k_selection`, `moire_hexagonal_convention_checked`). Any other blocking reason raises
`EpcGammaRigidError` rather than being silently absorbed — a protocol blocked for a reason nobody
accounted for is not a protocol this stage may run against. **[CODE]**

**[MEAS]** Run against the real, currently-frozen S41 protocol on disk, `epc_authorize` grants:

```
authorised_claim   : graph2mat_derivative_validated_no_full_ks_coupling; the three result-split
                      directions stay tagged synthetic_test_displacement; no g_mn_nu may be
                      published (roadmap B7 / S39 verdict)
blocking            : GO-8a MATBG PhononProvider: NO-GO-8a
                      C14C basis response (global formalism gate): NO_GO
                      k_selection: no live electronic spectrum measured yet at any candidate k ...
                      moire_hexagonal_convention_checked: the K/M fractional labels are ...
result_status       : candidate_rigid_tbg
```

i.e. today's real gate state is exactly the one this ticket is scoped for.

## 3. What is reused, what is computed once

- **Checkpoint, geometry, positions:** the campaign's own gated checkpoint (`main()`'s
  `checkpoint` variable — external-gate-accepted or trained-and-gated, unchanged), the target
  `RUN.fdf` `prepare_target()` already wrote, and the positions `overlap()` already exported
  (`epc_prereg.DEFAULT_POSITIONS_XYZ`, the same file S41 froze the direction hashes from). No H,
  S or eigenpair is recomputed by this stage. **[CODE]**
- **Model load and CUDA preflight: once.** `quantify_checkpoint_derivative_error.load_model_and_batch`
  loads the checkpoint and builds one batch; `graph2mat_autograd_derivatives.resolve_jvp_backend`
  runs the one real forward-plus-double-backward CUDA probe and returns the effective backend —
  never a synthetic probe, per its own docstring. The `JvpBackendRecord` (requested, effective,
  preflight outcome, fallback reason) is recorded in the summary and is never silently dropped on
  a CPU fallback (compute policy). **[CODE]**
- **Per direction:** one JVP (`compute_graph2mat_directional_derivative` via
  `directional_derivative_field`) and one frozen central difference per `delta_sweep_ang` entry
  (`frozen_derivative_field`), all against the one loaded model/batch. `tau_backend` is the
  Frobenius disagreement between the JVP and the frozen side at the largest δ; the plateau spread
  of the frozen side across `delta_sweep_ang` is `tau_num`; the check
  `jvp_equals_frozen_within_backend_margin` follows S41's own `BACKEND_NOISE_MARGIN` rule.
  `topology_unchanged` is recorded `"not_measured"` rather than assumed: the batch's own topology
  hash is invariant to the displaced-forward substitution by construction (positions are
  substituted locally, `edge_index`/`shifts` never rebuilt), so it cannot certify that no
  neighbour pair crossed its cutoff — S40 already found the real MATBG edge count "OOM'd before
  it could be measured", and this stage does not pretend otherwise. **[CODE]/[OPEN]**

## 4. Per-artifact status and restart

`Comparison/results/epc/gamma_epc/matbg/artifact_status.json` carries one entry per
`raw_derivative__<direction>` artifact, states `pending → running → completed | failed`
(`invalid` is reserved for a future stage that rejects an already-`completed` artifact against a
stricter check, e.g. a tightened `tau_backend` margin). The signature
(`checkpoint sha256, direction_hash, delta_sweep_ang, neighbour topology hash, this file's own
sha256`) is recomputed on every call; a `completed` entry whose signature still matches is reused
without re-touching the model, satisfying "no recompute" across repeated `--epc-gamma-rigid` runs
and across a mid-campaign restart. A raised exception inside one direction is caught and recorded
`state: "failed"` with the exception text, rather than aborting every other direction's artifact.
**[CODE]**

## 5. Result status and forbidden labels

Every artifact this stage writes — completed, failed, or the top-level summary — carries
`result_status: "candidate_rigid_tbg"` (this ticket's own acceptance criterion) and
`artifact_kind: synthetic_test_displacement`. The summary's `forbidden_publication_labels`
explicitly lists `quantitative_relaxed_matbg_prediction`, `g_mn_nu` and `phonon_eigenmode`: none
of the five directions may be described as a phonon, and no numeric `g` exists in this stage's
output to mislabel in the first place. **[CODE]**

## 6. What this ticket does not do

**[OPEN]** `epc_gamma_rigid()` has been validated against the real, on-disk S41 protocol
(`epc_authorize`, §2 above) and against mocked heavy dependencies (`tests/test_run_tbg_gamma_rigid_epc.py`)
but has **not** been executed end to end against the real 44 656-orbital checkpoint in this
ticket. S37's own synthetic-displacement benchmark on this exact cell "OOM'd before it could
measure" a comparable JVP-class operation (`docs/epc_s37_matbg_synthetic_benchmark.md`), and this
repository's compute policy requires a cheap preflight before, not instead of, a GPU-heavy run at
this scale. Running `--epc-gamma-rigid` for the first time against the real checkpoint is
therefore the next step, and it should be preceded by the same representative-displacement
resource preflight GO-8 already requires (wall time, peak VRAM, one direction at a time) rather
than launched as a five-direction batch blind. Nothing in this ticket's own machinery presupposes
which outcome that preflight finds.

## 7. Reproduce

```bash
# Gate check against the real, frozen protocol (cheap; no GPU, no checkpoint load):
.venv/bin/python -c "
import sys; sys.path.insert(0, 'Comparison/scripts'); sys.path.insert(0, 'shared')
import run_tbg_pure_graph2mat_campaign as camp, preregister_matbg_gamma_rigid_epc as prereg
import json; print(json.dumps(camp.epc_authorize(prereg.load_protocol()), indent=2))
"

# Orchestration/idempotency unit tests (mocked, no GPU/checkpoint):
.venv/bin/pytest -q tests/test_run_tbg_gamma_rigid_epc.py

# Full stage, after the existing pipeline (GPU, real checkpoint -- see Section 6 first):
.venv/bin/python Comparison/scripts/run_tbg_pure_graph2mat_campaign.py \
    --checkpoint Comparison/results/tbg_registry_spectral_loss/training/checkpoints/spectral-best-epoch=247-step=03472.ckpt \
    --precision-gate Comparison/results/tbg_pure_graph2mat/precision_gate.json \
    --epc-gamma-rigid
```
