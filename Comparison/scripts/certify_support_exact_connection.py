#!/usr/bin/env python3
"""Gate 1 of the support-exact one-sided connection (schema v2).

Only the PAOs centred on the displaced atom ``I`` move, so with ``J_I`` the
orbital selector of ``I`` and ``D = d_{R_Iu} S = S_L + S_R``:

    S_R = S_R . J_I          (only columns in I move)
    S_L = J_I . S_L          (only rows in I move)

hence, off the I-I block, ``D`` fixes the one-sided connection exactly:

    S_R = (1 - J).D.J + B_I ,    S_L = J.D.(1 - J) - B_I ,  B_I = J.S_R.J

This script certifies the *support structure* that makes that reconstruction
legitimate -- ``(1-J).D.(1-J) = 0`` and ``J.D.J = 0`` -- by propagating those
blocks to the electronic quantity, and builds the support-exact off-site parts
of ``S_L``/``S_R`` from the existing ``D_S``.  ``B_I`` (the I-I block, which is
NOT purely monocentric in a periodic cell: it also carries C1-C1 terms between
different lattice images) is NOT determined here and the v2 gate therefore
stays open; see ``b_i_status`` in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from audit_reference_basis_dense_k import _runs  # noqa: E402
from certify_full_basis_connection import CASES, WEIGHTS  # noqa: E402
from certify_production_basis_connection import LIMIT_EV_PER_ANG  # noqa: E402
from cross_basis_projection_preflight import K_POINTS  # noqa: E402
from run_displaced_projection_sentinel import H_ANG  # noqa: E402
from run_nested_basis_ladder import _rms  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/support_exact_full_basis_connection"
DISPLACED_ATOM = 0  # _geometry() in certify_production_basis_connection displaces xyz[0]
# 0.25 meV/Ang, expressed in the eV/Ang the code works in.
SUPPORT_LIMIT_EV_PER_ANG = 2.5e-4
SOLVE_RESIDUAL_TOL = 1e-11


def _selector(geometry, atom: int = DISPLACED_ATOM) -> np.ndarray:
    """Diagonal 0/1 mask J_I selecting the orbitals centred on ``atom``."""
    mask = np.zeros(geometry.no)
    mask[geometry.a2o(atom, all=True)] = 1.0
    return mask


def _hermitian_solve(overlap: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve S X = rhs by Cholesky (S is positive definite), with residual.

    Never forms S^-1, and never truncates eigenvalues -- that would change the
    very subspace under validation.
    """
    symmetric = (overlap + overlap.conj().T) / 2
    solution = cho_solve(cho_factor(symmetric, lower=True), rhs)
    residual = float(
        np.linalg.norm(symmetric @ solution - rhs) / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    )
    return solution, residual


def _propagate(delta_left, delta_right, overlap, k0):
    """dDelta = -dS_L S^-1 K - K S^-1 dS_R, via Hermitian solves only."""
    x, res_x = _hermitian_solve(overlap, k0)
    y, res_y = _hermitian_solve(overlap, delta_right)
    correction = -(delta_left @ x + k0 @ y)
    return _rms(correction, overlap)[0], max(res_x, res_y)


def _reconstruct(d_s: np.ndarray, mask: np.ndarray):
    """Support-exact off-site blocks of S_L, S_R from D_S (B_I excluded)."""
    outside = 1.0 - mask
    right = outside[:, None] * d_s * mask[None, :]  # row not in I, col in I
    left = mask[:, None] * d_s * outside[None, :]  # row in I, col not in I
    return left, right


def _support_blocks(d_s: np.ndarray, mask: np.ndarray):
    """The two blocks theory says must vanish: (1-J).D.(1-J) and J.D.J."""
    outside = 1.0 - mask
    return outside[:, None] * d_s * outside[None, :], mask[:, None] * d_s * mask[None, :]


def evaluate(output: Path) -> dict:
    import sisl

    k_points = np.asarray(K_POINTS)
    cases = []
    for basis, direction in CASES:
        runs = _runs(basis, direction)
        _, central, fermi, vacuum, _ = _matrix_data(runs[0])
        hamiltonians = {step: _matrix_data(path)[1] for step, path in runs.items()}
        mask = _selector(central.geometry)
        rows, failures, saved = [], [], {}
        for ik, k in enumerate(k_points):
            s0 = np.asarray(central.Sk(k=k, format="array"))
            k0 = np.asarray(central.Hk(k=k, format="array")) + (fermi - vacuum) * s0
            d_s = sum(
                weight * np.asarray(hamiltonians[step].Sk(k=k, format="array"))
                for step, weight in WEIGHTS.items()
            ) / (12 * H_ANG)
            scale = max(np.linalg.norm(d_s), np.finfo(float).tiny)

            out_out, in_in = _support_blocks(d_s, mask)
            # Propagate each theoretically-zero block as a one-sided perturbation
            # of the connection: it would otherwise be silently absorbed.
            e_out, res_out = _propagate(out_out.conj().T, out_out, s0, k0)
            e_in, res_in = _propagate(in_in.conj().T, in_in, s0, k0)
            e_support = max(e_out, e_in)

            left, right = _reconstruct(d_s, mask)
            # Diagnostic only: the B_I = 0 reconstruction. NOT the v2 verdict.
            closure = d_s - left - right
            e_closure_b0, res_closure = _propagate(closure / 2, closure / 2, s0, k0)
            adjoint_b0 = float(
                np.linalg.norm(left - right.conj().T) / max(np.linalg.norm(left), np.finfo(float).tiny)
            )
            solve_residual = max(res_out, res_in, res_closure)

            if e_support >= SUPPORT_LIMIT_EV_PER_ANG:
                failures.append(f"k={ik}: propagated support residual={e_support:.6g} eV/Ang")
            if solve_residual >= SOLVE_RESIDUAL_TOL:
                failures.append(f"k={ik}: Hermitian solve residual={solve_residual:.6g}")

            rows.append({
                "k_reduced": k.tolist(),
                "outside_outside_relative": float(np.linalg.norm(out_out) / scale),
                "inside_inside_relative": float(np.linalg.norm(in_in) / scale),
                "outside_outside_propagated_ev_per_ang": e_out,
                "inside_inside_propagated_ev_per_ang": e_in,
                "support_propagated_ev_per_ang": e_support,
                "solve_residual_relative": solve_residual,
                "diagnostic_B_I_zero": {
                    "closure_propagated_ev_per_ang": e_closure_b0,
                    "adjoint_relative": adjoint_b0,
                    "note": (
                        "DIAGNOSTIC LOWER BOUND, NOT the v2 verdict, and NOT evidence "
                        "that B_I = 0. D_S - S_L - S_R reduces identically to the J.D.J "
                        "block, which the support check above already certifies as zero, "
                        "so this closure is insensitive to B_I by construction: any "
                        "anti-Hermitian B_I added to S_R and subtracted from S_L leaves "
                        "it unchanged. It bounds nothing about the true connection error."
                    ),
                },
            })
            saved[f"S_L_offsite_k{ik}"] = left
            saved[f"S_R_offsite_k{ik}"] = right
            saved[f"D_S_k{ik}"] = d_s
        output.mkdir(parents=True, exist_ok=True)
        artifact = output / f"{basis}_{direction}_support_exact.npz"
        np.savez_compressed(artifact, k_points=k_points, orbital_mask_I=mask, **saved)
        cases.append({
            "basis": basis, "direction": direction,
            "displaced_atom_index": DISPLACED_ATOM,
            "orbitals_in_I": int(mask.sum()), "orbitals_total": int(mask.size),
            "verdict": "PASS" if not failures else "NO_GO",
            "failure_reasons": failures, "k_results": rows,
            "artifact": str(artifact), "artifact_sha256": file_sha256(artifact),
        })
        print(f"support structure {basis} {direction}: {cases[-1]['verdict']}", flush=True)

    passed = all(case["verdict"] == "PASS" for case in cases)
    report = {
        "schema": "support_exact_full_basis_connection_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate": "support_exact_full_basis_connection_v2",
        "verdict": "BLOCKED",
        "gate_1_support_structure": "PASS" if passed else "NO_GO",
        "k_count": len(k_points),
        "h_ang": H_ANG,
        "stencil": "five-point central",
        "thresholds": {
            "support_propagated_ev_per_ang": SUPPORT_LIMIT_EV_PER_ANG,
            "connection_propagated_ev_per_ang": LIMIT_EV_PER_ANG,
            "solve_residual_relative": SOLVE_RESIDUAL_TOL,
        },
        "threshold_note": (
            "support limit is 0.25 meV/Ang = 2.5e-4 eV/Ang; the preregistered v2 "
            "connection limit stays 1 meV/Ang = 1e-3 eV/Ang"
        ),
        "linear_algebra": "Cholesky/Hermitian solves only; no explicit inverse, no pseudoinverse, no eigenvalue truncation",
        "b_i_status": "NOT_DETERMINED",
        "b_i_note": (
            "B_I = J.S_R.J is the only piece D_S cannot determine: J.D.J = 0 because "
            "the displaced atom and its periodic images move together, yet B_I(k) = "
            "sum_R exp(i k.R) <phi_{alpha,0}|d_{R_Iu} phi_{beta,R}> is nonzero and is "
            "NOT purely monocentric -- with 10-14 bohr PAOs in this cell it contains "
            "C1-C1 terms across lattice images R != 0. Obtaining it needs the "
            "TS.S.Save/.onlyS route or two-centre radial-angular integrals, which are "
            "out of scope here. Gate v2 cannot be adjudicated until B_I exists."
        ),
        "cases": cases,
        "claim_scope": (
            "Gate 1 only: support structure of D_S plus support-exact off-site "
            "reconstruction of S_L/S_R. No B_I, no v2 connection verdict, no new SIESTA, "
            "no Delta_out, no full-KS claim. full_basis_connection_v1 = NO_GO stands as "
            "methodological history; delta_out_closure and full_KS remain BLOCKED."
        ),
    }
    path = output / "support_exact_full_basis_connection.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"support-exact connection gate 1: {report['gate_1_support_structure']} "
          f"(gate v2 {report['verdict']}: B_I {report['b_i_status']}) -> {path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    return 0 if evaluate(args.output)["gate_1_support_structure"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
