# E-F_001-S45 — GO-9 certification of the rigid-MATBG q != 0 (non-Gamma) branch

S44 (`preregister_matbg_qneq0_rigid_epc.py`,
`docs/epc_s44_matbg_qneq0_rigid_preregistration.md`) froze the non-Gamma protocol but authorised no
execution: unlike the Gamma branch (S41/S42), whose claim ladder pre-authorises a narrow
raw-derivative rung regardless of GO-8a, the q != 0 protocol always carries `q_selection` and
`qaware_basis_response_producer_missing` as unconditional blockers, so `ready_to_execute` is `False`
today and there is no `epc_qneq0_rigid()` execution stage the way S43 certified `epc_gamma_rigid()`.
This ticket certifies what the roadmap actually requires at this stage instead: GO-5 and GO-8b read
live and combined correctly, the route-mechanics/reproducibility controls S40 already proved on the
real production JVP, and — the specific failure mode this ticket exists to close — that a phase or
basis-response defect forces `NO-GO` even when the raw `D_H` matrices involved are individually
well-formed.

Producer: `Comparison/scripts/certify_qneq0_rigid_campaign.certify_rigid_qneq0_campaign_go9()`.
Tests: `tests/test_certify_qneq0_rigid_campaign.py`.
Artifact: `Comparison/results/epc/certification/matbg/s45_rigid_qneq0_campaign_go9_certification.json`.

## 1. What each suite exercises

- **`gate_authorization_suite`**: the real, on-disk S44 protocol's content hash matches its own
  `frozen_content_sha256` (`preregister_matbg_qneq0_rigid_epc.verify()`); a tampered copy is
  rejected; `require_frozen_protocol()` correctly raises against the real protocol today (it is
  frozen but its preconditions are not met); the five candidate directions rebuilt from the real
  `(31, 30)` geometry hash-match the frozen `phonon.candidate_sectors`; the frozen `Q_PRIMARY`/
  `Q_CROSSCHECK` fractional coordinates match the module's own constants.
- **`go5_and_go8b_dependency_suite`**: reads `epc_commensurate_k.go5_decision()` and
  `certify_qneq0_matbg_response.certify_qneq0_matbg_response_go8b()` live (not asserted), confirms
  GO-8b's own `graphene_k_agreement_suite.blocked_by` equals the live GO-5 decision (proving the
  dependency is wired to the real gate rather than a hardcoded label), and proves via an explicit
  truth table that `_qneq0_go9_gate()` is a strict conjunction of both — neither GO-5 nor GO-8b can
  substitute for the other. `full_ks_gates_open` reads `False` today because both gates are
  currently `NO_GO`/`NO-GO-8b`.
- **`route_mechanics_and_reproducibility_suite`**: reuses
  `certify_qneq0_matbg_response._certify_route_mechanics()` verbatim — the same real-JVP mechanics,
  cache/restart-with-stale-rejection, GPU preflight, sparse round trip, shell-scaling and
  Gamma-JVP-reuse guard S40 already certified. Not re-implemented here.
- **`phase_and_basis_response_guard_suite`**: (1) feeds a sign-flipped `q` into the real Q-B route
  and confirms the resulting `D_H(k+q, k)` disagrees with the correct Q-C route by `0.66` (worst
  abs diff) while both matrices stay individually finite/well-formed — proof that a phase-convention
  bug produces a *wrong* matrix that still *looks* ordinary, and that the existing agreement check
  (tolerance `1e-9`) would catch it; (2) confirms the frozen protocol's own
  `qaware_basis_response_present_and_required` check is registered and currently unmeetable
  (`QAWARE_BASIS_RESPONSE_GAP`); (3) confirms that gap alone keeps `physics_review.status ==
  "BLOCKED"` and `ready_to_execute == False` even though `route_mechanics_suite` already reads
  `PASS` on the real kernel (S40) — a route-mechanics pass never promotes the claim past the
  mechanics-only rung while the basis-response gap stands.
- **`resource_budget_suite`**: honestly `NOT_RUN`. No execution of the q != 0 branch has ever been
  measured against the real `(31, 30)` checkpoint (S37/S40 OOM'd before any real MATBG edge count
  was recorded), so reporting a wall/RAM/VRAM/disk verdict would fabricate agreement with the
  roadmap's own Section X margins — the same honesty pattern S43 established for the Gamma branch.

## 2. Verdict

`go9_status = "NO-GO-9"`. `gate_authorization_suite`, `go5_and_go8b_dependency_suite`,
`route_mechanics_and_reproducibility_suite` and `phase_and_basis_response_guard_suite` all read
`PASS` (the *mechanisms* — provenance, gate wiring, route mechanics, and the phase/basis-response
guard — all work correctly), but `go9` additionally requires `full_ks_gates_open` (both GO-5 and
GO-8b actually `PASS`/`GO-8b`) and `resource_budget_suite == "PASS"`; neither holds today. GO-9
(q != 0 branch) therefore stays `NO-GO-9` until GO-4/GO-5 close and a real GPU run of the branch is
measured. `final_publication_label` is pinned to `rigid_tbg_model`;
`quantitative_relaxed_matbg_prediction` (plus `g_mn_nu`, `phonon_eigenmode`) stay forbidden
regardless of gate outcome. `require_go9_certified_qneq0_rigid_campaign()` raises on anything short
of `GO-9`, mirroring `require_go9_certified_rigid_gamma_campaign` (S43) and
`require_go8b_certified_qneq0_response` (S40).

## 3. Reproduce

```bash
.venv/bin/pytest -q tests/test_certify_qneq0_rigid_campaign.py
.venv/bin/python Comparison/scripts/certify_qneq0_rigid_campaign.py
```
