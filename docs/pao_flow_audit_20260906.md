# PAO contract audit — 2026-09-06

Scope: finite-PAO covariant response in PROD-SZ (`Delta_PAO_cov`, `g_PAO`),
not a full-KS operator derivative. No new DFT, bases, MATBG or finite-q campaigns
were launched. Historical numerical artifacts and gate verdicts are not rewritten.

## Evidence and remaining blockers

- `gauge_derivative_audit`: PASS on 400 k points for C1:x,z, using existing
  five-point PROD-SZ stencils. Worst electronic identity residuals are
  0.002051 and 0.000457 meV/Ang respectively, below the existing 1 meV/Ang
  connection budget. The moving-vacuum primary gauge is unchanged. The
  identity has a finite-stencil product-rule residual, not exact floating-point
  equality. Installed sisl Fortran source subtracts `Ef*S` when reading TSHS;
  adding the TSHS Fermi level times S restores the stored Hamiltonian energy
  zero. This is source verification, not a second independent binary reader.
- Numerical PAO convergence remains UNADJUDICATED. The prospective campaign
  is in `pao_numerical_convergence_protocol_v2.json`; no propagated convergence
  numbers are fabricated for settings not yet run.
- GO-7 remains NO_GO. Its independently reconstructed spatial AB transformation
  now runs inside the adjudicator and passes on real artifacts. Upstream gates
  and phonon mode matching still fail. S33 cannot choose Q-C while GO-5 fails;
  S36/small-TBG remain BLOCKED/UNDECIDED unless GO-5 and GO-7 both pass.
- GPU runtime evidence exists for the Graph2Mat JVP preflight (CUDA device,
  precision, requested/effective backend, preflight result, fallback reason).
  The runtime audit accepts that manifest, not just the static source scan.
  This does not certify an unexecuted MATBG or finite-q job.
- Checkpoint lineage remains UNKNOWN: zero independently cleared checkpoints;
  final-case exclusion is unproven. Already inspected cases are not blind
  holdouts. Fine-tuning and final claims remain BLOCKED by the shared claim
  policy, even when a numerical comparison itself is executable.
- Reconciliation found 56 children without individual scientific-check evidence
  in the audited cycle. None is marked covered by the parent. This is an
  evidence gap, not proof that every earlier calculation failed. No state.db
  changes were made.

Historical status remains: embedded exterior-response v3 PASS; Delta_out
closure and negligibility NO_GO; QZDP, ghost-d and User.Basis round-trip
preflights NO_GO; full_KS BLOCKED. Executing a NO_GO gate correctly does not
satisfy its physical objective.

## Reproduce read-only audits

Run from the physics repository with `.venv/bin/python`:

```text
Comparison/scripts/audit_gauge_derivative.py
Comparison/scripts/audit_epc_backends.py --require-runtime --output Comparison/results/epc/pao_flow_audit/backend_audit.json
Comparison/scripts/evaluate_ab_bilayer_epc.py --result-root Comparison/results/epc/pao_flow_audit/ab
Comparison/scripts/audit_epc_lineage.py --preregistration docs/epc_preregistration_v1.json --fail-on-unknown --output Comparison/results/epc/pao_flow_audit/lineage_audit.json
Comparison/scripts/ops/audit_epc_gate_status.py --output Comparison/results/epc/pao_flow_audit/reconciliation.json
```

Nonzero exits for GO-7, lineage and reconciliation are intentional fail-closed
outcomes. Acceptance of this audit must not promote those scientific gates.
