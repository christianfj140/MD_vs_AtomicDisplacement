# E-F_001-S52 — Validate and interpret the relaxed-MATBG EPC result (Fase 11)

S51 (`run_matbg_relaxed_epc.py`, `Comparison/results/epc/relaxed_epc/matbg/relaxed_epc_summary.json`)
executed the S50-frozen protocol and wrote `no_execution_authorised` for both branches (`gamma`,
`qneq0`): S47 is NO-GO-10, no relaxed geometry exists anywhere in this repository, so there is no
raw response, PAO-covariant response or mode coupling to compute. This ticket does not recompute any physics.
It is the objective's own "aplicar el protocolo prerregistrado, comparar con el modelo rígido y
emitir claims relaxed-MATBG solo dentro de las incertidumbres validadas": validate that S51 actually
ran the frozen S50 protocol and refused for S50's own reasons, compare that refusal against the
rigid campaign's own certified state, and state explicitly which claims that evidence licenses and
which it forbids.

Producer: [`Comparison/scripts/validate_relaxed_epc_result.py`](../Comparison/scripts/validate_relaxed_epc_result.py)
(`validate_and_interpret()`). Tests:
[`tests/test_validate_relaxed_epc_result.py`](../tests/test_validate_relaxed_epc_result.py). Depends
on `preregister_matbg_relaxed_epc.py` (S50), `run_matbg_relaxed_epc.py` (S51),
`certify_rigid_gamma_campaign.py` (S43) and `certify_qneq0_rigid_campaign.py` (S45), all read live,
none re-implemented.

Evidence conventions: **[CODE]** verified in this repository, **[MEAS]** measured here against real
(or freshly regenerated) artifacts, **[DESIGN]** decision frozen in this document.

## 1. Three questions, each answered by reading an existing artifact, not by recomputing one

### 1.1 Did S51 actually run S50's protocol, and refuse for S50's own reasons?

**[CODE]** `protocol_chain_report()` reuses `preregister_matbg_relaxed_epc.verify()` verbatim for the
part it already does — the on-disk protocol's `frozen_content_sha256` matches its own recomputed
hash, and every `*.json` under the relaxed-EPC result root cites that hash — and adds the one check
`verify()` cannot make on its own: that each branch's *persisted* `blocking` list
(`relaxed_epc_summary.json`) is not merely non-empty but byte-identical to the *frozen* protocol's
own `blocking` list for that branch. A refusal that cites the right protocol but a different (e.g.
shortened or reworded) blocking list would still pass `verify()` and would still fail this check —
exactly the failure mode a validation step exists to catch, distinct from an authorization step.

**[MEAS]** Against the real on-disk S50/S51 artifacts: `protocol_intact=True`,
`results_cite_the_protocol=True`, both branches' persisted blocking lists match the frozen protocol
exactly, `verified=True`.

### 1.2 How does this compare with the rigid model?

**[CODE]** `rigid_comparison_report()` reads GO-9 live from
`Comparison/results/epc/certification/matbg/s43_rigid_gamma_campaign_go9_certification.json` (S43)
and `.../s45_rigid_qneq0_campaign_go9_certification.json` (S45) — the same certifiers this roadmap
already treats as the authoritative GO-9 verdict — rather than assuming or re-deriving how far the
rigid campaign has progressed.

**[MEAS]** Both rigid branches are `NO-GO-9` today (S43: blocked only on an unmeasured resource
budget; S45: blocked on GO-5/GO-8b plus the same unmeasured resource budget). S50 already requires
the relaxed branches to clear everything the rigid branches require *plus* GO-10
(`docs/epc_s50...` — S50's own module docstring), so a relaxed branch cannot legitimately be ahead of
a rigid campaign that has not itself cleared GO-9. The relaxed refusal is at least as justified as
the rigid one; it is not a separate or harsher standard invented only for the relaxed case.

### 1.3 What can be claimed about relaxed-MATBG EPC today?

**[DESIGN]** `claims_report()` licenses exactly three claims — the protocol was correctly applied,
no execution is authorised, and the rigid campaign itself is not GO-9 either — and forbids
`quantitative_relaxed_matbg_prediction`, `g_mn_nu` and `phonon_eigenmode` (S50's own list) plus one
new label this ticket adds: `relaxed_vs_rigid_epc_quantitative_delta`. No `g` has been computed on
the relaxed geometry (it does not exist) or certified GO-9 on the rigid one; there is no measured
pair of numbers and therefore no validated uncertainty to bound a relaxed-vs-rigid EPC difference.
Naming this label explicitly, rather than leaving "no comparison exists" implicit, closes the one
claim a reader could otherwise be tempted to construct from S43/S45's and S51's numbers side by side.

## 2. Verdict

`NO_QUANTITATIVE_CLAIM_LICENSED`. `claims["all_licensed_claims_hold"]` is `True` against the real
on-disk artifacts today: the protocol chain is intact, no forbidden label has leaked into S51's
output, and the rigid comparison is read live. This is not a GO of any kind — it is the honest
statement that every claim available today is procedural (the refusal was correctly derived), not
physical (no coupling was measured, on either geometry).

## 3. Reopen condition

Re-run whenever S50, S51, S43 or S45 change. If S47 ever reopens GO-10 and a relaxed geometry is
produced, a follow-up ticket executes the relaxed branches for real; this validator's own
`rigid_comparison_report` would then need to compare an actual relaxed `g` against a GO-9-certified
rigid one, which is a different, and currently unwritten, comparison — not an extension of this one.

## 4. Reproduce

```bash
.venv/bin/pytest -q tests/test_validate_relaxed_epc_result.py
.venv/bin/python Comparison/scripts/validate_relaxed_epc_result.py
```
